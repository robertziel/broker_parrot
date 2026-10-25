"""ONE LLM DAEMON PER BOX — a second ollama / vLLM container is stopped on sight.

THE FAILURE MODE. A benchmark starts a second ``ollama/ollama`` container
(``ollama-bench``, on another port, the production model volume mounted read-only)
beside the engine's ``broker-ollama``, and loads the same model into it: two resident
copies on a unified-memory box, most of its RAM gone — and NOTHING in the engine can
see it. The residency
enforcer probes the servers it was configured with; a daemon on another port is invisible,
so "one model per box" holds on paper while the box holds two. Stopped by hand, such a
container can be back within a minute: whatever started it starts it again.

So the rule is enforced where a second daemon can be SEEN — the docker socket — and every
tick, the way the parked-box quiesce already kills resurrected servers: a box that runs
the engine's own ollama (or vLLM) container may run NO OTHER container of that kind.

What this deliberately does NOT do: touch a box whose configured daemon is not running
(a dev box with an unrelated ollama is left alone — the rule is "no second daemon beside
OURS", not "no ollama anywhere"); match images by substring (``llm-gateway:latest`` is a
proxy, not a server); or raise into the heartbeat.
"""
from __future__ import annotations

import pytest

from queue_workflows import rogue_llm


def _c(name: str, image: str, state: str = "running", ports=(11434,)) -> dict:
    """One entry of docker's ``GET /containers/json`` — names carry a leading slash."""
    return {"Names": [f"/{name}"], "Image": image, "State": state,
            "Ports": [{"PrivatePort": p} for p in ports]}


OURS = _c("broker-ollama", "ollama/ollama:0.33.2-rocm")
BENCH = _c("ollama-bench", "ollama/ollama:0.33.2-rocm")
LB = _c("llm-gateway", "llm-gateway:latest", ports=(8010,))


@pytest.fixture(autouse=True)
def _fresh():
    rogue_llm.reset_for_tests()
    yield
    rogue_llm.reset_for_tests()


# ── the decision (pure) ──────────────────────────────────────────────────────


def test_a_second_ollama_beside_ours_is_a_rogue():
    rogues = rogue_llm.find_rogues([OURS, BENCH, LB], ollama_name="broker-ollama",
                                   vllm_name="vllm")
    assert [(r.name, r.kind, r.ours) for r in rogues] == [
        ("ollama-bench", "ollama", "broker-ollama")]


def test_our_own_daemon_is_never_a_rogue():
    assert rogue_llm.find_rogues([OURS], ollama_name="broker-ollama", vllm_name="vllm") == []


def test_the_load_balancer_is_not_a_server_however_its_image_is_named():
    """``llm-gateway:latest`` contains 'vllm'. A substring match would stop the proxy every
    box's clients dial — the image REPOSITORY has to be the server's."""
    assert rogue_llm.find_rogues([OURS, LB], ollama_name="broker-ollama", vllm_name="vllm") == []


def test_without_our_daemon_running_nothing_is_a_rogue():
    """The rule is 'no second daemon beside OURS'. A dev box running its own ollama, or a
    box whose production ollama is down, is not policed."""
    assert rogue_llm.find_rogues([BENCH], ollama_name="broker-ollama", vllm_name="vllm") == []
    stopped_ours = _c("broker-ollama", "ollama/ollama:0.33.2-rocm", state="exited")
    assert rogue_llm.find_rogues([stopped_ours, BENCH], ollama_name="broker-ollama",
                                 vllm_name="vllm") == []


def test_a_container_that_is_not_running_is_not_a_rogue():
    exited = _c("ollama-bench", "ollama/ollama:0.33.2-rocm", state="exited")
    assert rogue_llm.find_rogues([OURS, exited], ollama_name="broker-ollama",
                                 vllm_name="vllm") == []


def test_the_configured_names_are_the_deployments_own():
    mine = _c("ollama-box", "ollama/ollama:latest")
    other = _c("broker-ollama", "ollama/ollama:latest")
    rogues = rogue_llm.find_rogues([mine, other], ollama_name="ollama-box", vllm_name="vllm")
    assert [r.name for r in rogues] == ["broker-ollama"]


@pytest.mark.parametrize("image,kind", [
    ("ollama/ollama:0.33.2-rocm", "ollama"),
    ("ollama/ollama", "ollama"),
    ("registry.local:5000/ollama/ollama@sha256:abc", "ollama"),
    ("vllm/vllm-openai:latest", "vllm"),
    ("vllm/vllm-openai:v0.9.1", "vllm"),
    ("llm-gateway:latest", None),
    ("content-worker:latest", None),
    ("searxng/searxng:latest", None),
    ("", None),
])
def test_image_kind_matches_the_servers_repositories_only(image, kind):
    assert rogue_llm.image_kind(image) == kind


def test_a_second_vllm_beside_ours_is_a_rogue_too():
    ours = _c("vllm", "vllm/vllm-openai:latest", ports=(8000,))
    extra = _c("vllm-vision", "vllm/vllm-openai:latest", ports=(8001,))
    rogues = rogue_llm.find_rogues([ours, extra], ollama_name="broker-ollama", vllm_name="vllm")
    assert [(r.name, r.kind) for r in rogues] == [("vllm-vision", "vllm")]


# ── the sweep (docker I/O, faked) ────────────────────────────────────────────


class _Docker:
    def __init__(self, listing, *, fail_stop_for=()):
        self.listing = listing
        self.calls: list[tuple[str, str]] = []
        self.fail_stop_for = set(fail_stop_for)

    def __call__(self, method, path, body=None):
        self.calls.append((method, path))
        if method == "GET":
            return 200, self.listing
        name = path.split("/containers/")[1].split("/")[0]
        if name in self.fail_stop_for:
            return 500, {"message": "daemon busy"}
        return 204, None


def test_the_sweep_stops_every_rogue_it_sees(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_OLLAMA_CONTAINER", "broker-ollama")
    docker = _Docker([OURS, BENCH, LB])
    stopped = rogue_llm.sweep(api_fn=docker, host="box-a-gpu")
    assert stopped == ["ollama-bench"]
    assert docker.calls == [("GET", "/containers/json"),
                            ("POST", "/containers/ollama-bench/stop?t=5")]


def test_report_mode_names_the_rogue_but_leaves_it_running(monkeypatch):
    monkeypatch.setenv(rogue_llm.MODE_ENV, "report")
    docker = _Docker([OURS, BENCH])
    assert rogue_llm.sweep(api_fn=docker, host="box-a-gpu") == ["ollama-bench"]
    assert [m for m, _ in docker.calls] == ["GET"], "reported, not stopped"


def test_the_kill_switch_makes_it_inert(monkeypatch):
    monkeypatch.setenv(rogue_llm.DISABLE_ENV, "1")
    docker = _Docker([OURS, BENCH])
    assert rogue_llm.sweep(api_fn=docker, host="box-a-gpu") == []
    assert docker.calls == []


def test_a_box_without_the_socket_is_a_quiet_no_op_that_does_not_retry_every_tick():
    """The gpu heartbeat ticks every 10 s and most claim containers have no socket. One
    failed connect per minute, not one per tick."""
    calls = {"n": 0}

    def no_socket(*_a, **_k):
        calls["n"] += 1
        raise FileNotFoundError("/var/run/docker.sock")

    clock = {"t": 1000.0}
    for _ in range(5):
        assert rogue_llm.sweep(api_fn=no_socket, host="x", now_fn=lambda: clock["t"]) == []
        clock["t"] += 10.0
    assert calls["n"] == 1
    clock["t"] += 60.0
    rogue_llm.sweep(api_fn=no_socket, host="x", now_fn=lambda: clock["t"])
    assert calls["n"] == 2


def test_a_stop_that_fails_does_not_stop_the_sweep_or_raise():
    extra = _c("ollama-two", "ollama/ollama:latest")
    docker = _Docker([OURS, BENCH, extra], fail_stop_for={"ollama-bench"})
    assert rogue_llm.sweep(api_fn=docker, host="box-a-gpu") == ["ollama-two"]


def test_a_docker_error_on_the_listing_never_raises():
    def boom(*_a, **_k):
        return 500, {"message": "daemon busy"}

    assert rogue_llm.sweep(api_fn=boom, host="box-a-gpu") == []


# ── where it runs ────────────────────────────────────────────────────────────


def test_the_gpu_heartbeat_sweeps_before_it_looks_at_residency():
    """The residency probe must see a box the guard has already cleaned, and the guard
    must run wherever the socket is — in a typical deployment that is the claim worker on
    the GPU boxes and the orchestrator on the control box, so both tick it."""
    import inspect

    from queue_workflows import claim_worker, node_pool

    tick = inspect.getsource(claim_worker.ClaimWorker._box_lease_tick)
    code = "\n".join(l.split("#", 1)[0] for l in tick.splitlines())
    assert "rogue_llm.sweep(" in code
    assert code.index("rogue_llm.sweep(") < code.index("_refresh_box_llm_state()")
    pool = inspect.getsource(node_pool.NodePool._tick)
    assert "_sweep_rogue_llm" in pool


def test_the_orchestrator_sweep_is_interval_gated(monkeypatch):
    from queue_workflows import node_pool

    calls = {"n": 0}
    monkeypatch.setattr(rogue_llm, "sweep", lambda **kw: calls.__setitem__("n", calls["n"] + 1) or [])
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._rogue_llm_interval_s = 0.0
    pool._sweep_rogue_llm()
    pool._rogue_llm_interval_s = 60.0
    pool._sweep_rogue_llm()
    pool._sweep_rogue_llm()
    assert calls["n"] == 1


# ── the GPU-service rule ─────────────────────────────────────────────────────
#
# The failure mode: after a reboot a sibling project's GPU service (``app3d-web-1``, a
# FastAPI with ``restart: unless-stopped``, ``--gpus all``) comes up before vLLM and
# takes host port 8000 — the port every client dials vLLM on. The switch recreates
# vLLM, its start loses the port, and the box serves nothing for the whole warm window.
# Its worker holds GPU memory the admission gate can only see as "less available". The
# rule for the GPU boxes: nothing uses the GPU there except what the broker arbitrates.


def _gpu(name, image, *, cid, restart="unless-stopped", project=None, gpu=True,
         runtime="runc", devices=()):
    listing = {"Id": cid, "Names": [f"/{name}"], "Image": image, "State": "running",
               "Labels": ({"com.docker.compose.project": project} if project else {})}
    inspected = {"Id": cid, "Name": f"/{name}", "State": {"Running": True}, "HostConfig": {
        "DeviceRequests": ([{"Count": -1, "Capabilities": [["gpu"]]}] if gpu else []),
        "Runtime": runtime, "Devices": [{"PathOnHost": d} for d in devices],
        "RestartPolicy": {"Name": restart}}}
    return listing, inspected


class _DockerGpu:
    """Answers the listing AND per-container inspects; records every call and body."""

    def __init__(self, *containers):
        self.listing = [c[0] for c in containers]
        self.inspects = {c[1]["Id"]: c[1] for c in containers}
        self.names = {c[0]["Names"][0].lstrip("/"): c[1]["Id"] for c in containers}
        self.calls: list[tuple[str, str]] = []
        self.bodies: dict[str, object] = {}

    def __call__(self, method, path, body=None):
        self.calls.append((method, path))
        if body is not None:
            self.bodies[path] = body
        if path == "/containers/json":
            return 200, self.listing
        if method == "GET":
            cid = path.split("/containers/")[1].split("/")[0]
            data = self.inspects.get(cid) or self.inspects.get(self.names.get(cid, ""))
            return (200, data) if data else (404, {"message": "no such container"})
        return 204, None

    def inspected(self) -> list[str]:
        return [p for m, p in self.calls if m == "GET" and p != "/containers/json"]


APP3D_WEB = _gpu("app3d-web-1", "app3d-worker-cuda13:latest", cid="p1", project="app3d")
APP3D_DB = _gpu("app3d-postgres-1", "postgres:16", cid="p2", project="app3d", gpu=False)
LANE = _gpu("content-worker-workers-1", "gpu-worker:latest", cid="l1",
            project="content-worker")
AGENT = _gpu("box-agent", "gpu-worker:latest", cid="a1", project="box-worker")
BROADCAST = _gpu("hw-broadcast", "python:3.12-slim", cid="h1")
TRAINER = _gpu("trainer-run-1", "model-trainer:latest", cid="t1", restart="no")
OUR_OLLAMA = _gpu("broker-ollama", "ollama/ollama:0.33.2", cid="o1")
ALLOW = "broker-ollama,vllm,content-worker,box-worker,hw-broadcast"


@pytest.fixture
def _allow(monkeypatch):
    monkeypatch.setenv(rogue_llm.ALLOW_ENV, ALLOW)
    monkeypatch.setenv("QUEUE_WORKFLOWS_OLLAMA_CONTAINER", "broker-ollama")
    monkeypatch.setenv("QUEUE_WORKFLOWS_VLLM_CONTAINER", "vllm")
    rogue_llm.reset_for_tests()


def test_a_gpu_service_outside_the_allow_list_is_stopped_and_pinned(_allow):
    """Stopped, AND its restart policy set to ``no`` — a ``docker stop`` alone is undone
    by dockerd at the next boot, which is exactly how app3d got there."""
    docker = _DockerGpu(OUR_OLLAMA, APP3D_WEB, LANE)
    assert rogue_llm.sweep(api_fn=docker, host="box-c-gpu") == ["app3d-web-1"]
    writes = [(m, p) for m, p in docker.calls if m == "POST"]
    assert writes == [("POST", "/containers/app3d-web-1/update"),
                      ("POST", "/containers/app3d-web-1/stop?t=10")]
    assert docker.bodies["/containers/app3d-web-1/update"] == {"RestartPolicy": {"Name": "no"}}


def test_allowed_by_name_prefix_or_compose_project(_allow):
    """``hw-broadcast`` by name, the lane and the box agent by their compose project."""
    docker = _DockerGpu(OUR_OLLAMA, LANE, AGENT, BROADCAST)
    assert rogue_llm.sweep(api_fn=docker, host="box-b-gpu") == []
    assert [m for m, _ in docker.calls if m == "POST"] == []


def test_our_own_servers_are_allowed_without_being_listed(monkeypatch):
    monkeypatch.setenv(rogue_llm.ALLOW_ENV, "content-worker")
    monkeypatch.setenv("QUEUE_WORKFLOWS_OLLAMA_CONTAINER", "broker-ollama")
    rogue_llm.reset_for_tests()
    docker = _DockerGpu(OUR_OLLAMA, LANE)
    assert rogue_llm.sweep(api_fn=docker, host="box-b-gpu") == []


def test_a_hand_launched_gpu_run_is_named_but_not_stopped(_allow, caplog):
    """THE RULE: never stop a hand-launched run on sight; the RAM guard stops it only
    if the box runs short of memory.

    Presence is not the offence — MEMORY is. A run someone launched by hand (a training
    job, a benchmark; ``restart=no``, so dockerd will never resurrect it) shares the box
    until the box runs short, and then the memory sentinel takes it (see
    :func:`stop_gpu_outsiders` and ``tests/test_memory_shedding.py``). A SERVICE is
    different and still goes on sight: it came back by itself at boot, took vLLM's port
    and would do so again every reboot."""
    docker = _DockerGpu(OUR_OLLAMA, TRAINER)
    with caplog.at_level("WARNING", logger="queue_workflows.rogue_llm"):
        assert rogue_llm.sweep(api_fn=docker, host="box-b-gpu") == []
        assert rogue_llm.sweep(api_fn=docker, host="box-b-gpu") == []
    named = [r for r in caplog.records if "trainer-run-1" in r.getMessage()]
    assert len(named) == 1, "named once, not every tick"
    assert "memory" in named[0].getMessage()
    assert [m for m, _ in docker.calls if m == "POST"] == []


def test_a_deployment_that_wants_them_gone_on_sight_can_say_so(_allow, monkeypatch):
    docker = _DockerGpu(OUR_OLLAMA, TRAINER)
    monkeypatch.setenv(rogue_llm.RUNS_ENV, "stop")
    rogue_llm.reset_for_tests()
    assert rogue_llm.sweep(api_fn=docker, host="box-b-gpu") == ["trainer-run-1"]


def test_the_ram_guard_stops_every_outsider_run_and_service_alike(_allow):
    """What the memory sentinel calls when the box is STILL short after the broker has
    given back its own servers: the outsiders go, largest offence first or not, because
    at that point the box is about to stop forking."""
    docker = _DockerGpu(OUR_OLLAMA, TRAINER, APP3D_WEB, LANE)
    stopped = rogue_llm.stop_gpu_outsiders(api_fn=docker, host="box-b-gpu",
                                           why="0.4 GB left")
    assert sorted(stopped) == ["app3d-web-1", "trainer-run-1"]
    assert ("POST", "/containers/trainer-run-1/stop?t=10") in docker.calls
    assert ("POST", "/containers/app3d-web-1/stop?t=10") in docker.calls
    assert not any(p.startswith("/containers/content-worker") for _m, p in docker.calls)


def test_the_ram_guard_ignores_the_report_modes(_allow, monkeypatch):
    """``report`` is about routine policing. A box in memory danger is not routine."""
    monkeypatch.setenv(rogue_llm.RUNS_ENV, "report")
    monkeypatch.setenv(rogue_llm.MODE_ENV, "report")
    rogue_llm.reset_for_tests()
    docker = _DockerGpu(OUR_OLLAMA, TRAINER, APP3D_WEB)
    assert sorted(rogue_llm.stop_gpu_outsiders(api_fn=docker, host="box-b-gpu", why="x")) == \
        ["app3d-web-1", "trainer-run-1"]


def test_the_ram_guard_does_nothing_without_an_allow_list(monkeypatch):
    """Unset, the engine has no idea what belongs on this box and stops nothing."""
    monkeypatch.delenv(rogue_llm.ALLOW_ENV, raising=False)
    rogue_llm.reset_for_tests()
    docker = _DockerGpu(OUR_OLLAMA, TRAINER, APP3D_WEB)
    assert rogue_llm.stop_gpu_outsiders(api_fn=docker, host="box-b-gpu", why="x") == []


def test_the_ram_guard_never_touches_our_own_servers(_allow, monkeypatch):
    """Our vLLM and ollama are shed by the sentinel's own path, which knows how to stop
    a server properly; this is for everything else. Shown with an allow-list that does
    NOT name them, so only the ours= exclusion can spare them."""
    monkeypatch.setenv(rogue_llm.ALLOW_ENV, "content-worker")
    rogue_llm.reset_for_tests()
    ours_vllm = _gpu("vllm", "vllm/vllm-openai:v0.23.0", cid="v1")
    docker = _DockerGpu(OUR_OLLAMA, ours_vllm, TRAINER)
    assert rogue_llm.stop_gpu_outsiders(api_fn=docker, host="box-b-gpu", why="x") == ["trainer-run-1"]
    assert not any("/containers/vllm/stop" in p or "/containers/broker-ollama/stop" in p
                   for _m, p in docker.calls)


def test_a_container_without_the_gpu_is_never_touched(_allow):
    """The sibling project's Postgres uses no GPU: not the guard's business, whatever
    its restart policy."""
    docker = _DockerGpu(OUR_OLLAMA, APP3D_DB)
    assert rogue_llm.sweep(api_fn=docker, host="box-c-gpu") == []
    assert [m for m, _ in docker.calls if m == "POST"] == []


@pytest.mark.parametrize("variant", [
    dict(gpu=False, runtime="nvidia"),
    dict(gpu=False, devices=("/dev/dri/renderD128",)),
    dict(gpu=False, devices=("/dev/kfd",)),
    dict(gpu=False, devices=("/dev/nvidia0",)),
])
def test_the_nvidia_runtime_and_the_gpu_device_nodes_count_as_gpu_access(_allow, variant):
    rogue = _gpu("parakeet-stt", "parakeet-stt:latest", cid="k1", project="parakeet", **variant)
    docker = _DockerGpu(OUR_OLLAMA, rogue)
    assert rogue_llm.sweep(api_fn=docker, host="box-c-gpu") == ["parakeet-stt"]


def test_without_an_allow_list_the_gpu_rule_is_off(monkeypatch):
    """A deployment that never configured it sees no new behaviour — not even an
    inspect per container."""
    monkeypatch.delenv(rogue_llm.ALLOW_ENV, raising=False)
    monkeypatch.setenv("QUEUE_WORKFLOWS_OLLAMA_CONTAINER", "broker-ollama")
    rogue_llm.reset_for_tests()
    docker = _DockerGpu(OUR_OLLAMA, APP3D_WEB)
    assert rogue_llm.sweep(api_fn=docker, host="box-c-gpu") == []
    assert docker.calls == [("GET", "/containers/json")]


def test_report_mode_names_a_rogue_gpu_service_without_stopping_it(_allow, monkeypatch, caplog):
    monkeypatch.setenv(rogue_llm.MODE_ENV, "report")
    docker = _DockerGpu(OUR_OLLAMA, APP3D_WEB)
    with caplog.at_level("ERROR", logger="queue_workflows.rogue_llm"):
        assert rogue_llm.sweep(api_fn=docker, host="box-c-gpu") == ["app3d-web-1"]
    assert [m for m, _ in docker.calls if m == "POST"] == []
    assert any("app3d-web-1" in r.getMessage() and "NOT stopping" in r.getMessage()
               for r in caplog.records)


def test_each_container_is_inspected_once_across_sweeps(_allow):
    """The listing is one call; GPU access and restart policy need an inspect each,
    and a container's HostConfig does not change while it runs — remember it by Id.
    An allowed container is never inspected at all: the allow-list is read from the
    listing's name and labels."""
    docker = _DockerGpu(OUR_OLLAMA, LANE, AGENT, TRAINER)
    rogue_llm.sweep(api_fn=docker, host="box-b-gpu")
    rogue_llm.sweep(api_fn=docker, host="box-b-gpu")
    rogue_llm.sweep(api_fn=docker, host="box-b-gpu")
    assert docker.inspected() == ["/containers/t1/json"]


def test_a_gone_container_is_forgotten_so_its_facts_are_not_reused_when_it_returns(_allow, monkeypatch):
    """The facts are remembered by Id while the Id is listed. Once it leaves the listing
    they must go with it: a `docker update --restart` between two lives of the same
    container (same Id) would otherwise be read from the stale entry. (Shown on a run
    the guard is only NAMING, so the first sweep leaves the listing unchanged.)"""
    monkeypatch.setenv(rogue_llm.RUNS_ENV, "report")
    rogue_llm.reset_for_tests()
    docker = _DockerGpu(OUR_OLLAMA, TRAINER)
    assert rogue_llm.sweep(api_fn=docker, host="box-b-gpu") == []          # restart=no: named
    docker.listing = [OUR_OLLAMA[0]]
    rogue_llm.sweep(api_fn=docker, host="box-b-gpu")                        # gone
    again = _gpu("trainer-run-1", "model-trainer:latest", cid="t1")   # same Id, a service now
    docker = _DockerGpu(OUR_OLLAMA, again)
    assert rogue_llm.sweep(api_fn=docker, host="box-b-gpu") == ["trainer-run-1"]
    assert docker.inspected() == ["/containers/t1/json"]
    assert ("POST", "/containers/trainer-run-1/update") in docker.calls, \
        "read fresh as a SERVICE, so the restart policy is pinned this time"


def test_a_stop_that_fails_is_not_counted_as_acted_and_does_not_raise(_allow):
    class _NoStop(_DockerGpu):
        def __call__(self, method, path, body=None):
            if path.endswith("/stop?t=10"):
                self.calls.append((method, path))
                return 500, {"message": "daemon busy"}
            return super().__call__(method, path, body)
    docker = _NoStop(OUR_OLLAMA, APP3D_WEB)
    assert rogue_llm.sweep(api_fn=docker, host="box-c-gpu") == []


def test_a_container_that_is_not_running_is_not_inspected_or_touched(_allow):
    """`docker ps -a` shows the exited ones too when the listing is asked for all;
    a stopped service holds no GPU and no port."""
    exited = _gpu("app3d-web-1", "app3d-worker-cuda13:latest", cid="p9", project="app3d")
    exited[0]["State"] = "exited"
    docker = _DockerGpu(OUR_OLLAMA, exited)
    assert rogue_llm.sweep(api_fn=docker, host="box-c-gpu") == []
    assert docker.calls == [("GET", "/containers/json")]


def test_an_update_that_fails_still_stops_the_service(_allow):
    class _NoUpdate(_DockerGpu):
        def __call__(self, method, path, body=None):
            if path.endswith("/update"):
                self.calls.append((method, path))
                return 500, {"message": "daemon busy"}
            return super().__call__(method, path, body)
    docker = _NoUpdate(OUR_OLLAMA, APP3D_WEB)
    assert rogue_llm.sweep(api_fn=docker, host="box-c-gpu") == ["app3d-web-1"]
    assert ("POST", "/containers/app3d-web-1/stop?t=10") in docker.calls


def test_a_parked_worker_sweeps_too():
    """A parked box is the one most likely to be borrowed for something else, and it is
    still the box that serves the broker's models the moment it is enabled."""
    import inspect

    from queue_workflows import claim_worker

    src = inspect.getsource(claim_worker.ClaimWorker._build_park_quiesce)
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    assert "rogue_llm.sweep(" in code
