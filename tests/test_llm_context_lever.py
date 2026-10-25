"""The lever that actually changes a server's window — and what it must not break.

A window is chosen at SERVER START: ollama reads ``OLLAMA_CONTEXT_LENGTH`` from its
environment (its ``/v1`` endpoint ignores a per-request ``num_ctx`` — measured on 0.33.2),
vLLM takes ``--max-model-len`` on its command line. Neither can be changed on a running
server, so the lever RECREATES the container from its own inspect output with one value
swapped, exactly as the vLLM model switch already does.

THE HAZARD THIS SUITE EXISTS FOR: a recreate that rebuilds the container from assumptions
turns a window change into a silent redeploy. Docker's create body does NOT inherit the
container's network endpoints, so a payload that forgets them reattaches ``broker-ollama``
to its network WITHOUT its aliases — every client dialling it by name gets NXDOMAIN, and
the box's whole LLM service is gone. Network identity, binds, devices, GPU requests and
the restart policy must survive verbatim; only the window may change.

The original value is remembered in a container LABEL, so "no queued job asks for a
window" can restore what the operator configured instead of freezing the last override
forever.
"""
from __future__ import annotations

import pytest

from queue_workflows.llm_backends import server_context as sc

OLLAMA_INSPECT = {
    "Id": "abc123def4567890abcdef",
    "Config": {
        "Image": "ollama/ollama:0.33.2",
        "Cmd": ["serve"],
        "Env": ["PATH=/usr/bin", "OLLAMA_KEEP_ALIVE=5m", "OLLAMA_NUM_PARALLEL=4"],
        "Labels": {"com.docker.compose.service": "ollama"},
        "ExposedPorts": {"11434/tcp": {}},
    },
    "HostConfig": {
        "NetworkMode": "acme_default",
        "Binds": ["/srv/ollama:/root/.ollama"],
        "RestartPolicy": {"Name": "unless-stopped"},
        "DeviceRequests": [{"Count": -1, "Capabilities": [["gpu"]]}],
        "PortBindings": {"11434/tcp": [{"HostPort": "11434"}]},
    },
    "NetworkSettings": {"Networks": {
        "acme_default": {"Aliases": ["ollama", "abc123def456"],
                             "IPAMConfig": {"IPv4Address": "172.20.0.5"},
                             "NetworkID": "n1", "IPAddress": "172.20.0.5"},
        "hw": {"Aliases": ["broker-ollama"], "NetworkID": "n2"},
    }},
}

VLLM_INSPECT = {
    "Id": "0f0f0f0f0f0f0f",
    "Config": {"Image": "vllm/vllm-openai:latest",
               "Cmd": ["--model", "/models/chat", "--gpu-memory-utilization", "0.85"],
               "Env": ["NVIDIA_VISIBLE_DEVICES=all"], "Labels": {}},
    "HostConfig": {"NetworkMode": "host"},
    "NetworkSettings": {"Networks": {"host": {"Aliases": None, "NetworkID": "h"}}},
}


def _env(payload) -> dict[str, str]:
    return dict(e.split("=", 1) for e in payload["Env"])


# ── reading what a server is configured for ──────────────────────────────────


def test_the_configured_window_is_read_from_where_each_server_takes_it():
    assert sc.configured_context(OLLAMA_INSPECT, "ollama") is None
    assert sc.configured_context(VLLM_INSPECT, "vllm") is None
    with_ctx = {**OLLAMA_INSPECT, "Config": {
        **OLLAMA_INSPECT["Config"], "Env": ["OLLAMA_CONTEXT_LENGTH=131072"]}}
    assert sc.configured_context(with_ctx, "ollama") == 131072
    for argv in (["--max-model-len", "32768"], ["--max-model-len=32768"]):
        assert sc.configured_context(
            {"Config": {"Cmd": argv}}, "vllm") == 32768


def test_a_nonsense_value_reads_as_no_window_not_as_a_crash():
    assert sc.configured_context(
        {"Config": {"Env": ["OLLAMA_CONTEXT_LENGTH=lots"]}}, "ollama") is None
    assert sc.configured_context({"Config": {"Cmd": ["--max-model-len"]}}, "vllm") is None
    assert sc.configured_context({}, "ollama") is None


# ── the plan: what one window change does to the container ───────────────────


def test_ollama_gets_the_window_in_its_environment_and_keeps_the_rest():
    plan = sc.plan_context(OLLAMA_INSPECT, "ollama", 131072, name="broker-ollama")
    env = _env(plan.payload)
    assert env["OLLAMA_CONTEXT_LENGTH"] == "131072"
    assert env["OLLAMA_KEEP_ALIVE"] == "5m" and env["OLLAMA_NUM_PARALLEL"] == "4"
    assert plan.payload["Cmd"] == ["serve"]
    assert plan.payload["Image"] == "ollama/ollama:0.33.2"


def test_vllm_gets_the_window_on_its_command_line_in_the_form_it_already_uses():
    plan = sc.plan_context(VLLM_INSPECT, "vllm", 65536, name="vllm")
    assert plan.payload["Cmd"] == ["--model", "/models/chat",
                                   "--gpu-memory-utilization", "0.85",
                                   "--max-model-len", "65536"]
    equals_form = {**VLLM_INSPECT, "Config": {
        **VLLM_INSPECT["Config"], "Cmd": ["--model", "m", "--max-model-len=8192"]}}
    plan2 = sc.plan_context(equals_form, "vllm", 65536, name="vllm")
    assert plan2.payload["Cmd"] == ["--model", "m", "--max-model-len=65536"]


def test_a_server_already_at_that_window_is_left_alone():
    """A needless recreate costs the box its whole LLM service for a weight reload."""
    at_131k = {**OLLAMA_INSPECT, "Config": {
        **OLLAMA_INSPECT["Config"], "Env": ["OLLAMA_CONTEXT_LENGTH=131072"]}}
    assert sc.plan_context(at_131k, "ollama", 131072, name="broker-ollama") is None


# ── the original value, so "no need" can mean "back to the operator's" ───────


def test_the_first_override_records_what_the_operator_had_configured():
    plan = sc.plan_context(OLLAMA_INSPECT, "ollama", 131072, name="broker-ollama")
    assert plan.payload["Labels"][sc.DEFAULT_LABEL] == ""     # nothing was set
    assert plan.payload["Labels"]["com.docker.compose.service"] == "ollama"

    configured = {**OLLAMA_INSPECT, "Config": {
        **OLLAMA_INSPECT["Config"], "Env": ["OLLAMA_CONTEXT_LENGTH=32768"]}}
    plan2 = sc.plan_context(configured, "ollama", 131072, name="broker-ollama")
    assert plan2.payload["Labels"][sc.DEFAULT_LABEL] == "32768"


def test_a_second_override_does_not_overwrite_the_remembered_original():
    """Else two switches in a row would make 131072 look like what the operator chose."""
    overridden = {**OLLAMA_INSPECT, "Config": {
        **OLLAMA_INSPECT["Config"],
        "Env": ["OLLAMA_CONTEXT_LENGTH=131072"],
        "Labels": {**OLLAMA_INSPECT["Config"]["Labels"], sc.DEFAULT_LABEL: "32768"}}}
    plan = sc.plan_context(overridden, "ollama", 65536, name="broker-ollama")
    assert _env(plan.payload)["OLLAMA_CONTEXT_LENGTH"] == "65536"
    assert plan.payload["Labels"][sc.DEFAULT_LABEL] == "32768"


def test_restoring_puts_back_the_operators_value_and_forgets_the_override():
    overridden = {**OLLAMA_INSPECT, "Config": {
        **OLLAMA_INSPECT["Config"],
        "Env": ["OLLAMA_CONTEXT_LENGTH=131072", "OLLAMA_KEEP_ALIVE=5m"],
        "Labels": {sc.DEFAULT_LABEL: "32768"}}}
    plan = sc.plan_context(overridden, "ollama", None, name="broker-ollama")
    assert _env(plan.payload)["OLLAMA_CONTEXT_LENGTH"] == "32768"
    assert sc.DEFAULT_LABEL not in plan.payload["Labels"]


def test_restoring_a_window_the_operator_never_set_removes_it_entirely():
    overridden = {**OLLAMA_INSPECT, "Config": {
        **OLLAMA_INSPECT["Config"],
        "Env": ["OLLAMA_CONTEXT_LENGTH=131072"], "Labels": {sc.DEFAULT_LABEL: ""}}}
    plan = sc.plan_context(overridden, "ollama", None, name="broker-ollama")
    assert "OLLAMA_CONTEXT_LENGTH" not in _env(plan.payload)

    vllm_overridden = {**VLLM_INSPECT, "Config": {
        **VLLM_INSPECT["Config"], "Cmd": ["--model", "m", "--max-model-len", "131072"],
        "Labels": {sc.DEFAULT_LABEL: ""}}}
    plan2 = sc.plan_context(vllm_overridden, "vllm", None, name="vllm")
    assert plan2.payload["Cmd"] == ["--model", "m"]


def test_restoring_a_server_nobody_overrode_is_not_a_recreate():
    assert sc.plan_context(OLLAMA_INSPECT, "ollama", None, name="broker-ollama") is None


def test_restore_never_strips_a_window_the_operator_set_themselves():
    """No label means WE never changed it, so whatever is configured IS the operator's —
    "restore the default" must not read that as "remove the default"."""
    operators_own = {**OLLAMA_INSPECT, "Config": {
        **OLLAMA_INSPECT["Config"], "Env": ["OLLAMA_CONTEXT_LENGTH=32768"]}}
    assert sc.plan_context(operators_own, "ollama", None, name="broker-ollama") is None


# ── the redeploy hazard ──────────────────────────────────────────────────────


def test_the_network_identity_survives_or_every_client_loses_the_server():
    """Docker's create body does not inherit endpoints. Drop them and the container
    comes back on the network WITHOUT its aliases: every peer dialling `ollama` by name
    gets NXDOMAIN and the box's LLM service is gone while the container looks healthy."""
    plan = sc.plan_context(OLLAMA_INSPECT, "ollama", 131072, name="broker-ollama")
    endpoints = plan.payload["NetworkingConfig"]["EndpointsConfig"]
    assert list(endpoints) == ["acme_default"], "the create attaches ONE network"
    assert endpoints["acme_default"]["Aliases"] == ["ollama"]
    assert endpoints["acme_default"]["IPAMConfig"] == {"IPv4Address": "172.20.0.5"}
    # the rest are connected after create — losing one silently halves reachability
    assert plan.connect == [("hw", {"Aliases": ["broker-ollama"]})]


def test_the_containers_own_id_is_not_carried_over_as_an_alias():
    """Docker adds the short id as an alias itself; re-sending the OLD one names the
    replacement after a container that no longer exists."""
    plan = sc.plan_context(OLLAMA_INSPECT, "ollama", 131072, name="broker-ollama")
    aliases = plan.payload["NetworkingConfig"]["EndpointsConfig"]["acme_default"]["Aliases"]
    assert "abc123def456" not in aliases


def test_a_host_network_container_gets_no_endpoints_at_all():
    """Docker refuses a create that both shares the host namespace and names networks."""
    plan = sc.plan_context(VLLM_INSPECT, "vllm", 65536, name="vllm")
    assert "NetworkingConfig" not in plan.payload
    assert plan.connect == []


def test_everything_the_operator_configured_survives_the_recreate():
    plan = sc.plan_context(OLLAMA_INSPECT, "ollama", 131072, name="broker-ollama")
    host = plan.payload["HostConfig"]
    assert host["Binds"] == ["/srv/ollama:/root/.ollama"]
    assert host["RestartPolicy"] == {"Name": "unless-stopped"}
    assert host["DeviceRequests"][0]["Capabilities"] == [["gpu"]]
    assert host["PortBindings"]["11434/tcp"][0]["HostPort"] == "11434"
    assert host["NetworkMode"] == "acme_default"
    assert plan.payload["ExposedPorts"] == {"11434/tcp": {}}


def test_the_vllm_model_switch_payload_is_unchanged_by_the_generalisation():
    """vllm_sidecar.recreate_payload is the older caller of the same builder."""
    from queue_workflows.llm_backends.vllm_sidecar import recreate_payload

    payload = recreate_payload(VLLM_INSPECT, ["--model", "/models/new"])
    assert payload["Cmd"] == ["--model", "/models/new"]
    assert payload["Image"] == "vllm/vllm-openai:latest"
    assert payload["Env"] == ["NVIDIA_VISIBLE_DEVICES=all"]


# ── the lever itself (docker I/O, faked) ─────────────────────────────────────


class _Docker:
    """A fake docker Engine API: records calls, answers from a script."""

    def __init__(self, inspect=OLLAMA_INSPECT, *, statuses=None):
        self.calls: list[tuple[str, str]] = []
        self.bodies: dict[str, dict] = {}
        self._inspect = inspect
        self._statuses = statuses or {}

    def __call__(self, method, path, body=None):
        self.calls.append((method, path))
        if body is not None:
            self.bodies[path] = body
        for prefix, answer in self._statuses.items():
            if path.startswith(prefix):
                return answer
        if method == "GET":
            return (200, self._inspect) if self._inspect is not None else (404, None)
        return 204, None


@pytest.fixture(autouse=True)
def _no_cache():
    from queue_workflows import llm_context_lever
    llm_context_lever.reset_cache_for_tests()
    yield
    llm_context_lever.reset_cache_for_tests()


def test_the_lever_recreates_the_container_in_the_order_that_keeps_the_name():
    from queue_workflows import llm_context_lever

    docker = _Docker()
    outcome = llm_context_lever.set_context("ollama", 131072, api_fn=docker)

    assert outcome["status"] == "started"
    verbs = [f"{m} {p.split('?')[0]}" for m, p in docker.calls]
    assert verbs == [
        "GET /containers/broker-ollama/json",
        "POST /containers/broker-ollama/stop",
        "DELETE /containers/broker-ollama",
        "POST /containers/create",
        "POST /networks/hw/connect",
        "POST /containers/broker-ollama/start",
    ]
    created = next(b for p, b in docker.bodies.items() if p.startswith("/containers/create"))
    assert _env(created)["OLLAMA_CONTEXT_LENGTH"] == "131072"


def test_what_the_lever_just_changed_is_re_read_not_served_from_the_cache():
    """The switch asks 'is it serving my window yet?' immediately after the recreate. A
    cached answer from before it would say no for the whole TTL — and the switch would
    conclude its own change never landed."""
    from queue_workflows import llm_context_lever

    docker = _Docker()
    assert llm_context_lever.describe("ollama", api_fn=docker).configured is None
    llm_context_lever.set_context("ollama", 131072, api_fn=docker)
    docker._inspect = {**OLLAMA_INSPECT, "Config": {
        **OLLAMA_INSPECT["Config"], "Env": ["OLLAMA_CONTEXT_LENGTH=131072"]}}
    assert llm_context_lever.describe("ollama", api_fn=docker).configured == 131072


def test_a_server_already_at_the_window_is_not_touched():
    from queue_workflows import llm_context_lever

    at_131k = {**OLLAMA_INSPECT, "State": {"Running": True}, "Config": {
        **OLLAMA_INSPECT["Config"], "Env": ["OLLAMA_CONTEXT_LENGTH=131072"]}}
    docker = _Docker(at_131k)
    outcome = llm_context_lever.set_context("ollama", 131072, api_fn=docker)

    assert outcome["status"] == "already_running"
    assert [m for m, _ in docker.calls] == ["GET"], "no write on a no-op"


def test_no_such_container_is_permanent_not_a_retry_forever():
    from queue_workflows import llm_context_lever

    docker = _Docker(None)
    outcome = llm_context_lever.set_context("ollama", 131072, api_fn=docker)
    assert outcome["status"] == "permanent"
    assert "broker-ollama" in outcome["reason"]


def test_a_start_that_loses_the_port_race_is_retried_not_reported(monkeypatch):
    """Docker's remove returns before the old endpoint's port binding is torn down, so the
    replacement's start can fail with 'driver failed programming external connectivity'
    on the same port, typically right after a recreate. One more try a moment
    later succeeds; reporting the first as transient starts a 30 s retry loop instead."""
    from queue_workflows import llm_context_lever

    monkeypatch.setattr(llm_context_lever.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    class _Racy(_Docker):
        def __call__(self, method, path, body=None):
            if method == "POST" and path.endswith("/start"):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    self.calls.append((method, path))
                    return 500, {"message": "failed to set up container networking: driver "
                                            "failed programming external connectivity on "
                                            "endpoint vllm: Bind for 0.0.0.0:8000 failed"}
            return super().__call__(method, path, body)

    docker = _Racy()
    outcome = llm_context_lever.set_context("ollama", 131072, api_fn=docker)
    assert outcome["status"] == "started"
    assert attempts["n"] == 2


def test_a_port_held_by_another_container_is_permanent_not_a_retry_loop(monkeypatch):
    """box-c after a reboot: app3d-web had taken host port 8000 before vLLM. Every start
    then failed 'Bind for 0.0.0.0:8000 failed: port is already allocated' — reported
    transient, retried every 30 s for the whole warm window. Once the race retries are
    spent, a port another container holds is not going to free itself: the switch must
    give up now and say which port."""
    from queue_workflows import llm_context_lever

    monkeypatch.setattr(llm_context_lever.time, "sleep", lambda s: None)
    docker = _Docker(statuses={"/containers/broker-ollama/start": (
        500, {"message": "driver failed programming external connectivity on endpoint "
                         "broker-ollama: Bind for 0.0.0.0:11434 failed: port is already allocated"})})
    outcome = llm_context_lever.set_context("ollama", 131072, api_fn=docker)
    assert outcome["status"] == "permanent"
    assert "host port 11434 is held by another container" in outcome["reason"]


def test_a_stopped_container_whose_config_already_matches_is_started(monkeypatch):
    """The seeded sidecar: created with the right argv, never started. 'Already configured
    for that window' used to be reported as already_running — and nothing started, so the
    switch warmed for a server that did not exist."""
    from queue_workflows import llm_context_lever

    stopped = {**OLLAMA_INSPECT, "State": {"Running": False}, "Config": {
        **OLLAMA_INSPECT["Config"], "Env": ["OLLAMA_CONTEXT_LENGTH=131072"]}}
    docker = _Docker(stopped)
    outcome = llm_context_lever.set_context("ollama", 131072, api_fn=docker)
    assert outcome["status"] == "started"
    assert [m for m, _ in docker.calls] == ["GET", "POST"]
    assert docker.calls[-1][1].startswith("/containers/broker-ollama/start")


def test_a_running_container_whose_config_already_matches_is_left_alone():
    from queue_workflows import llm_context_lever

    running = {**OLLAMA_INSPECT, "State": {"Running": True}, "Config": {
        **OLLAMA_INSPECT["Config"], "Env": ["OLLAMA_CONTEXT_LENGTH=131072"]}}
    docker = _Docker(running)
    assert llm_context_lever.set_context("ollama", 131072, api_fn=docker)["status"] == "already_running"
    assert [m for m, _ in docker.calls] == ["GET"]


def test_a_refused_create_is_permanent_and_a_busy_daemon_is_transient():
    from queue_workflows import llm_context_lever

    refused = _Docker(statuses={"/containers/create": (400, {"message": "invalid mount"})})
    assert llm_context_lever.set_context("ollama", 131072, api_fn=refused)["status"] == "permanent"

    busy = _Docker(statuses={"/containers/create": (500, {"message": "daemon busy"})})
    assert llm_context_lever.set_context("ollama", 131072, api_fn=busy)["status"] == "transient"


def test_an_unreachable_socket_is_permanent_as_configured():
    from queue_workflows import llm_context_lever

    def no_socket(*_a, **_k):
        raise FileNotFoundError("/var/run/docker.sock")

    outcome = llm_context_lever.set_context("ollama", 131072, api_fn=no_socket)
    assert outcome["status"] == "permanent"
    assert "docker" in outcome["reason"].lower()


def test_a_daemon_that_times_out_mid_cycle_is_transient():
    from queue_workflows import llm_context_lever

    def timeout(*_a, **_k):
        raise TimeoutError("timed out")

    assert llm_context_lever.set_context(
        "ollama", 131072, api_fn=timeout)["status"] == "transient"


def test_a_host_wired_lever_replaces_the_built_in_one_entirely():
    """A deployment whose server is not a local container (k8s, another host) wires its
    own; the engine must then never touch docker."""
    from queue_workflows import config as _config
    from queue_workflows import llm_context_lever

    seen: list[tuple] = []

    def lever(kind, context_length):
        seen.append((kind, context_length))
        return {"status": "started", "reason": "k8s rollout"}

    cfg = _config.get_config()
    cfg.llm_context_lever_fn = lever
    try:
        docker = _Docker()
        outcome = llm_context_lever.set_context("ollama", 131072, api_fn=docker)
        assert outcome == {"status": "started", "reason": "k8s rollout"}
        assert seen == [("ollama", 131072)]
        assert docker.calls == []
        assert llm_context_lever.available("ollama") is True
    finally:
        cfg.llm_context_lever_fn = None


def test_a_raising_host_lever_is_transient_not_an_exception_into_the_switch():
    from queue_workflows import config as _config
    from queue_workflows import llm_context_lever

    cfg = _config.get_config()
    cfg.llm_context_lever_fn = lambda kind, n: 1 / 0
    try:
        outcome = llm_context_lever.set_context("ollama", 131072, api_fn=_Docker())
        assert outcome["status"] == "transient"
        assert "ZeroDivisionError" in outcome["reason"]
    finally:
        cfg.llm_context_lever_fn = None


def test_the_kill_switch_makes_the_lever_unavailable_and_inert(monkeypatch):
    from queue_workflows import llm_context_lever

    monkeypatch.setenv(llm_context_lever.DISABLE_ENV, "1")
    docker = _Docker()
    assert llm_context_lever.available("ollama", api_fn=docker) is False
    outcome = llm_context_lever.set_context("ollama", 131072, api_fn=docker)
    assert outcome["status"] == "permanent"
    assert docker.calls == [], "disabled means no docker call at all"


def test_what_a_box_can_be_asked_for_is_read_from_the_container_and_cached():
    from queue_workflows import llm_context_lever

    docker = _Docker({**OLLAMA_INSPECT, "Config": {
        **OLLAMA_INSPECT["Config"], "Env": ["OLLAMA_CONTEXT_LENGTH=32768"]}})
    info = llm_context_lever.describe("ollama", api_fn=docker)
    assert (info.exists, info.configured, info.name) == (True, 32768, "broker-ollama")
    llm_context_lever.describe("ollama", api_fn=docker)
    assert len(docker.calls) == 1, "the heartbeat must not inspect every tick"
    assert llm_context_lever.available("ollama", api_fn=docker) is True


def test_a_box_without_that_container_has_no_lever_for_it():
    from queue_workflows import llm_context_lever

    docker = _Docker(None)
    assert llm_context_lever.describe("ollama", api_fn=docker).exists is False
    assert llm_context_lever.available("ollama", api_fn=docker) is False


def test_the_container_names_are_the_deployments_to_choose(monkeypatch):
    from queue_workflows import llm_context_lever

    assert llm_context_lever.container_name("ollama") == "broker-ollama"
    assert llm_context_lever.container_name("vllm") == "vllm"
    monkeypatch.setenv("QUEUE_WORKFLOWS_OLLAMA_CONTAINER", "ollama-box")
    assert llm_context_lever.container_name("ollama") == "ollama-box"


def test_a_start_that_fails_leaves_no_half_started_container_behind(monkeypatch):
    """The daemon can lose the port race and still leave the process running with no
    network endpoint (box-c: vLLM logged 'Application startup complete' on a
    container with `Networks: {}`). The next tick would then read it as running and
    configured — 'already_running' — and wait 30 minutes for a server nobody can
    reach. A failed start is stopped before it is reported."""
    from queue_workflows import llm_context_lever

    monkeypatch.setattr(llm_context_lever.time, "sleep", lambda s: None)
    docker = _Docker(statuses={"/containers/broker-ollama/start": (
        500, {"message": "Bind for 0.0.0.0:11434 failed: port is already allocated"})})
    outcome = llm_context_lever.set_context("ollama", 131072, api_fn=docker)
    assert outcome["status"] == "permanent"
    assert docker.calls[-1] == ("POST", "/containers/broker-ollama/stop?t=5")
    assert docker.calls[-2] == ("POST", "/containers/broker-ollama/start")


def test_a_running_container_with_no_network_endpoint_is_broken_not_ready():
    """Running, configured for the window — and attached to nothing: unreachable on
    every address the topology dials. Stop it and say so, rather than 'already_running'."""
    from queue_workflows import llm_context_lever

    detached = {**OLLAMA_INSPECT, "State": {"Running": True},
                "Config": {**OLLAMA_INSPECT["Config"], "Env": ["OLLAMA_CONTEXT_LENGTH=131072"]},
                "NetworkSettings": {"Networks": {}}}
    docker = _Docker(detached)
    outcome = llm_context_lever.set_context("ollama", 131072, api_fn=docker)
    assert outcome["status"] == "permanent"
    assert "network" in outcome["reason"]
    assert ("POST", "/containers/broker-ollama/stop?t=5") in docker.calls


def test_host_networking_has_no_endpoint_to_miss():
    from queue_workflows import llm_context_lever

    host_net = {**VLLM_INSPECT, "State": {"Running": True},
                "NetworkSettings": {"Networks": {}}}
    docker = _Docker(host_net)
    outcome = llm_context_lever.set_context("vllm", None, api_fn=docker)
    assert outcome["status"] == "already_running"
    assert [m for m, _ in docker.calls] == ["GET"]


# ── the vLLM profile: read off the container's own log ──────────────────────

VLLM_LOG = (
    "(EngineCore pid=170) INFO Model loading took 18.74 GiB memory and 126.99 seconds\n"
    "(EngineCore pid=170) INFO Available KV cache memory: 42.23 GiB\n"
    "(EngineCore pid=170) INFO The current --gpu-memory-utilization=0.4200 is equivalent to x\n"
    "(EngineCore pid=170) INFO GPU KV cache size: 684,056 tokens\n"
    "(EngineCore pid=170) INFO Maximum concurrency for 262,144 tokens per request: 2.61x\n"
)


def _framed(text: str, stream: int = 1) -> bytes:
    """Docker's multiplexed log stream: 8-byte header (type, 0,0,0, big-endian length)."""
    out = b""
    for line in text.splitlines(keepends=True):
        data = line.encode()
        out += bytes([stream, 0, 0, 0]) + len(data).to_bytes(4, "big") + data
    return out


def test_docker_log_frames_are_demultiplexed_and_a_tty_log_is_passed_through():
    from queue_workflows.llm_backends import docker_api

    assert docker_api.demux_stream(_framed(VLLM_LOG, 2)) == VLLM_LOG
    assert docker_api.demux_stream(VLLM_LOG.encode()) == VLLM_LOG
    assert docker_api.demux_stream(b"") == ""
    truncated = _framed(VLLM_LOG)[:-7]                     # a cut frame at the end
    assert "18.74 GiB" in docker_api.demux_stream(truncated)


VLLM_RUNNING = {**VLLM_INSPECT, "Id": "0f0f0f0f0f0f0f", "State": {"Running": True,
                "StartedAt": "2024-01-01T00:00:00.1Z"}, "Config": {
    **VLLM_INSPECT["Config"],
    "Cmd": ["--model", "/models/chat", "--served-model-name", "chat-model-a:latest",
            "--gpu-memory-utilization", "0.42", "--max-num-seqs", "8", "--max-model-len", "262144"],
    "Labels": {"queue_workflows.llm_context.max_num_seqs": "16"}}}


def test_describe_reports_what_the_plan_and_the_observation_need():
    from queue_workflows import llm_context_lever
    from queue_workflows.llm_backends import server_context as sc

    info = llm_context_lever.describe("vllm", api_fn=_Docker(VLLM_RUNNING))
    assert info.running and info.container_id == "0f0f0f0f0f0f0f"
    assert info.started_at == "2024-01-01T00:00:00.1Z"
    assert info.utilization == pytest.approx(0.42) and info.utilization_now == pytest.approx(0.42)
    assert info.max_num_seqs == 16, "the operator's remembered value"
    assert info.max_num_seqs_now == 8
    assert info.served_name == "chat-model-a:latest"
    assert info.fingerprint == sc.argv_fingerprint(VLLM_RUNNING["Config"]["Cmd"])


def test_the_startup_log_is_fetched_over_the_socket_and_parsed_into_a_profile():
    from queue_workflows import llm_context_lever

    class _WithLogs(_Docker):
        def __call__(self, method, path, body=None):
            self.calls.append((method, path))
            if "/logs" in path:
                return 200, _framed(VLLM_LOG)
            return super().__call__(method, path, body)

    docker = _WithLogs(VLLM_RUNNING)
    profile = llm_context_lever.observe_vllm(total_bytes=121 * 2**30, api_fn=docker,
                                             raw_fn=docker)
    assert profile is not None
    assert profile.kv_tokens == 684_056 and profile.utilization == pytest.approx(0.42)
    assert any(p.startswith("/containers/vllm/logs?") and "stdout=1" in p and "stderr=1" in p
               for _, p in docker.calls)


def test_no_container_or_no_complete_start_is_no_profile():
    from queue_workflows import llm_context_lever

    assert llm_context_lever.observe_vllm(total_bytes=121 * 2**30, api_fn=_Docker(None),
                                          raw_fn=lambda *a, **k: (200, b"")) is None
    assert llm_context_lever.observe_vllm(total_bytes=121 * 2**30, api_fn=_Docker(VLLM_RUNNING),
                                          raw_fn=lambda *a, **k: (200, b"still loading")) is None

    def boom(*a, **k):
        raise OSError("socket gone")
    assert llm_context_lever.observe_vllm(total_bytes=121 * 2**30, api_fn=_Docker(VLLM_RUNNING),
                                          raw_fn=boom) is None


def test_set_context_passes_the_concurrency_to_the_recreate():
    from queue_workflows import llm_context_lever
    from queue_workflows.llm_backends import server_context as sc

    docker = _Docker(VLLM_RUNNING)
    outcome = llm_context_lever.set_context("vllm", None, api_fn=docker, utilization=0.31,
                                            max_num_seqs=3)
    assert outcome["status"] == "started"
    created = docker.bodies["/containers/create?name=vllm"]
    assert sc.argv_max_seqs(created["Cmd"]) == 3
    assert sc.argv_utilization(created["Cmd"]) == pytest.approx(0.31)


def test_the_startup_lines_are_fetched_by_TIME_not_by_tail():
    """A vLLM that started hours ago and had been serving through a
    load balancer since: the API server logs every request, so a `tail=` read never
    reached back to the four startup lines and the box got no profile. The container
    says when it started — read THAT window."""
    from queue_workflows import llm_context_lever

    asked: list[str] = []

    def raw(method, path, **kw):
        asked.append(path)
        return 200, _framed(VLLM_LOG)

    profile = llm_context_lever.observe_vllm(total_bytes=121 * 2**30,
                                             api_fn=_Docker(VLLM_RUNNING), raw_fn=raw)
    assert profile is not None
    path, = asked
    assert "since=" in path and "until=" in path and "tail=" not in path
    import urllib.parse as up

    q = dict(up.parse_qsl(path.split("?", 1)[1]))
    started = 1704067200                      # 2024-01-01T00:00:00Z
    assert int(q["since"]) == started - 1
    assert int(q["until"]) == started + llm_context_lever.STARTUP_WINDOW_S


def test_a_container_whose_start_time_is_unreadable_falls_back_to_the_tail():
    from queue_workflows import llm_context_lever

    asked: list[str] = []

    def raw(method, path, **kw):
        asked.append(path)
        return 200, _framed(VLLM_LOG)

    no_time = {**VLLM_RUNNING, "State": {"Running": True, "StartedAt": "0001-01-01T00:00:00Z"}}
    assert llm_context_lever.observe_vllm(total_bytes=121 * 2**30, api_fn=_Docker(no_time),
                                          raw_fn=raw) is not None
    assert "tail=" in asked[0] and "since=" not in asked[0]
