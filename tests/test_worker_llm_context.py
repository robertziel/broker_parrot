"""The box's side of the window: what it observes, what it advertises, what it asks for.

Three things the GPU worker must get right, and a failure behind each:

  * OBSERVE. What window does my server actually serve? For ollama that is its
    CONFIGURED value, not whatever a loaded runner happens to sit at: another client can
    load a runner natively at 262144, and the next ``/v1`` call reloads it at the
    configured default. Believing the runner would make the box claim a job whose prompt
    then gets silently truncated — the whole failure this feature exists to prevent.
  * ADVERTISE. How big could it go? Against the memory the box has FREE, and only if it
    has a lever at all: a box that cannot restart its server cannot grow, so its ceiling
    is what it serves. Advertising more would make the fleet tag nothing and wait for a
    switch that can never happen.
  * ASK. Request a resize only when the queue actually needs one — and put the server
    back if the resize left it not answering.
"""
from __future__ import annotations

import pytest

from queue_workflows import claim_worker, llm_context, llm_context_lever, llm_probe
from queue_workflows import node_queue, server_switch
from queue_workflows.llm_backends import factory as _factory

PS_LOADED = [{"name": "chat-model-a:latest", "size": 18_714_627_931,
              "size_vram": 18_714_627_931, "context_length": 32768}]
SIZES = {"chat-model-a:latest": 17_741_872_154}
INFO = {"general.architecture": "qwen35", "qwen35.context_length": 262144,
        "qwen35.block_count": 36, "qwen35.attention.head_count_kv": 8,
        "qwen35.attention.head_count": 32, "qwen35.embedding_length": 4096,
        "qwen35.full_attention_interval": 4}


def _worker(served=("ollama",), *, switch=None):
    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-a-gpu"
            self.queue = "gpu"
            self._box_served = list(served)
            self._switch = switch or server_switch.ServerSwitch(label="box-a-gpu")
            self._box_lease = None
            self.model_cache = None
            self.lease_s = 60
            self.host_priority = 0
    w = _W()
    w._get_box_lease = lambda: None
    w._gpu_active_count = lambda: 0
    w._physical_box = lambda: "box-a"
    return w


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    llm_context_lever.reset_cache_for_tests()
    monkeypatch.setattr(_factory, "resolve_base_url",
                        lambda kind="ollama": f"http://{kind}:1")
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: list(PS_LOADED))
    monkeypatch.setattr(llm_probe, "ollama_model_sizes", lambda url, **k: dict(SIZES))
    monkeypatch.setattr(llm_probe, "ollama_model_info", lambda url, m, **k: dict(INFO))
    monkeypatch.setattr(llm_probe, "vllm_max_model_len", lambda url, **k: 32768)
    monkeypatch.setattr(llm_probe, "vllm_kv_capacity_tokens", lambda url, **k: 98304)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 60 * 2**30)
    yield
    llm_context_lever.reset_cache_for_tests()


def _lever(monkeypatch, *, configured=32768, exists=True):
    info = llm_context_lever.ServerContainer(
        kind="ollama", name="ollama", exists=exists, running=True,
        configured=configured)
    monkeypatch.setattr(llm_context_lever, "describe",
                        lambda kind, **k: info if kind == "ollama" else
                        llm_context_lever.ServerContainer(kind=kind, name=kind,
                                                          exists=exists))
    monkeypatch.setattr(llm_context_lever, "available", lambda kind, **k: exists)


# ── observe ──────────────────────────────────────────────────────────────────


def test_the_window_it_serves_is_the_configured_one_not_a_runners(monkeypatch):
    """A runner sitting at 262144 because another client asked natively says nothing
    about what OUR /v1 calls get — the configured value does."""
    _lever(monkeypatch, configured=32768)
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [
        {**PS_LOADED[0], "context_length": 262144}])
    serving, _ceiling = _worker()._observe_llm_context()
    assert serving == 32768


def test_without_a_lever_to_ask_the_loaded_runner_is_the_best_it_can_say(monkeypatch):
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k:
                        llm_context_lever.ServerContainer(kind=kind, name="x",
                                                          exists=False))
    monkeypatch.setattr(llm_context_lever, "available", lambda kind, **k: False)
    serving, ceiling = _worker()._observe_llm_context()
    assert serving == 32768
    assert ceiling == 32768, "no lever ⇒ it cannot grow, whatever the free memory says"


def test_a_box_with_room_advertises_the_window_it_could_switch_to(monkeypatch):
    _lever(monkeypatch, configured=32768)
    serving, ceiling = _worker()._observe_llm_context()
    assert serving == 32768
    assert ceiling is not None and ceiling > serving
    assert ceiling <= 262144, "never beyond what the model itself supports"


def test_the_ceiling_moves_with_the_memory_the_box_has_free(monkeypatch):
    _lever(monkeypatch, configured=32768)
    roomy = _worker()._observe_llm_context()[1]
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 2 * 2**30)
    tight = _worker()._observe_llm_context()[1]
    assert tight < roomy


def test_a_box_configured_beyond_its_model_serves_only_what_the_model_supports(
        monkeypatch):
    """ollama CLAMPS a request to the model's own maximum — e.g. a 65536 ask on a
    32768 model loads 32768. Advertising the configured number would make the box claim
    jobs it then truncates."""
    _lever(monkeypatch, configured=262144)
    monkeypatch.setattr(llm_probe, "ollama_model_info", lambda url, m, **k: {
        **INFO, "qwen35.context_length": 32768})
    serving, ceiling = _worker()._observe_llm_context()
    assert serving == 32768
    assert ceiling is not None and ceiling <= 32768


def test_a_box_serving_two_kinds_advertises_the_smaller_window(monkeypatch):
    """A sized job on a co-resident box may resolve to either server, so the honest
    claim filter is the SMALLER of the two — the larger would route jobs into the
    server that truncates them."""
    _lever(monkeypatch, configured=131072)
    monkeypatch.setattr(llm_probe, "vllm_max_model_len", lambda url, **k: 32768)
    serving, ceiling = _worker(served=("ollama", "vllm"))._observe_llm_context()
    assert serving == 32768
    assert ceiling is not None and ceiling >= 131072, (
        "the ceiling is the roomiest kind's — that is what a switch could reach")


def test_a_server_that_says_nothing_about_its_window_is_unknown(monkeypatch):
    _lever(monkeypatch, configured=None)
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [])
    assert _worker()._observe_llm_context() == (None, None)


def test_an_idle_ollama_still_knows_the_window_its_daemon_serves(monkeypatch):
    """THE IDLE-BOX TRAP. ollama's daemon is always up; only the MODEL unloads — after
    5 minutes of quiet here. A box with no lever to ask (no docker socket) then has no
    runner to read either, so it published NOTHING and, for as long as it stayed idle,
    could neither claim a sized job nor avoid being told it was unassignable. The next
    unsized job would load the model and the window would reappear: a sized job starved
    behind a fact the box already knew.

    The window is a property of the DAEMON (its configured default, bounded by the
    model), not of whichever runner happens to be resident, so the last one OBSERVED
    stands until something contradicts it."""
    _lever(monkeypatch, configured=None, exists=False)   # no lever, no configured value
    w = _worker()
    assert w._observe_llm_context()[0] == 32768     # observed from the loaded runner

    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [])   # idled out
    assert w._observe_llm_context() == (32768, 32768)


def test_it_never_invents_a_window_it_has_not_seen(monkeypatch):
    """A worker that has never observed one says unknown — memory, not a guess."""
    _lever(monkeypatch, configured=None, exists=False)
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [])
    monkeypatch.setattr(node_queue, "last_published_context", lambda *a, **k: None)
    assert _worker()._observe_llm_context() == (None, None)


def test_a_restarted_worker_resumes_the_window_it_last_published(monkeypatch):
    """A restart erases the in-process memory, and a box restarted while its model is
    unloaded would be windowless all over again — which is exactly the state box-a was
    in when this was found. Its own last heartbeat is the same observation, one process
    ago."""
    _lever(monkeypatch, configured=None, exists=False)
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [])
    reads: list = []
    monkeypatch.setattr(node_queue, "last_published_context",
                        lambda *a, **k: reads.append(a) or 131072)
    w = _worker()
    assert w._observe_llm_context() == (131072, 131072)
    w._observe_llm_context()
    w._observe_llm_context()
    assert len(reads) == 1, "read back ONCE per process, not every heartbeat"


def test_the_read_back_is_never_used_for_a_stopped_vllm(monkeypatch):
    _lever(monkeypatch, configured=None, exists=False)
    monkeypatch.setattr(llm_probe, "vllm_max_model_len", lambda url, **k: None)
    monkeypatch.setattr(node_queue, "last_published_context",
                        lambda *a, **k: pytest.fail("a stopped vLLM serves nothing"))
    assert _worker(served=("vllm",))._observe_llm_context() == (None, None)


def test_a_fresh_observation_replaces_the_remembered_one(monkeypatch):
    _lever(monkeypatch, configured=None, exists=False)
    w = _worker()
    w._observe_llm_context()
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [
        {**PS_LOADED[0], "context_length": 8192}])
    assert w._observe_llm_context()[0] == 8192
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [])
    assert w._observe_llm_context()[0] == 8192, "the LAST window seen, not the first"


def test_a_stopped_vllm_is_not_remembered_as_if_it_were_serving(monkeypatch):
    """The asymmetry that makes the memory safe: an unloaded ollama MODEL still has a
    daemon that will reload it at the same window, but an unreadable vLLM means the
    SERVER is gone — remembering its window would route sized jobs at nothing."""
    _lever(monkeypatch, exists=False)
    w = _worker(served=("vllm",))
    assert w._observe_llm_context()[0] == 32768
    monkeypatch.setattr(llm_probe, "vllm_max_model_len", lambda url, **k: None)
    assert w._observe_llm_context() == (None, None)


def test_a_box_serving_nothing_has_no_window(monkeypatch):
    _lever(monkeypatch)
    assert _worker(served=())._observe_llm_context() == (None, None)


def test_a_vllm_box_reads_its_window_from_the_server_it_runs(monkeypatch):
    _lever(monkeypatch)
    serving, ceiling = _worker(served=("vllm",))._observe_llm_context()
    assert serving == 32768
    assert ceiling == 32768, "without a stated model maximum it cannot prove more"


def test_a_vllm_box_told_its_models_maximum_can_grow_into_its_kv_pool(monkeypatch):
    """vLLM's KV pool is sized by gpu_memory_utilization, so a bigger window costs no
    new memory — it only has to fit the pool the server already owns."""
    _lever(monkeypatch)
    monkeypatch.setenv("QUEUE_WORKFLOWS_VLLM_MAX_CONTEXT", "262144")
    serving, ceiling = _worker(served=("vllm",))._observe_llm_context()
    assert (serving, ceiling) == (32768, 98304)


def test_the_kill_switch_freezes_every_ceiling_at_what_is_served(monkeypatch):
    _lever(monkeypatch, configured=32768)
    monkeypatch.setenv(llm_context_lever.DISABLE_ENV, "1")
    assert _worker()._observe_llm_context() == (32768, 32768)


def test_a_box_whose_last_resize_failed_stops_promising_a_bigger_one(monkeypatch):
    """The quarantine is what moves the election on to another box — and what makes the
    unassignable tag honest about a job nobody can actually grow into."""
    _lever(monkeypatch, configured=32768)
    switch = server_switch.ServerSwitch(label="box-a-gpu")
    w = _worker(switch=switch)
    switch.request("ollama", context_length=131072, resize=True)
    switch.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    switch.report_start_outcome(server_switch.StartOutcome.permanent("no container"))
    assert w._observe_llm_context() == (32768, 32768)


# ── advertise ────────────────────────────────────────────────────────────────


def test_the_heartbeat_carries_both_numbers(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(node_queue, "upsert_worker_heartbeat",
                        lambda **kw: seen.update(kw))
    emitter = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-a-gpu",
        llm_context_fn=lambda: (32768, 131072))
    emitter.emit_once()
    assert seen["llm_context_length"] == 32768
    assert seen["llm_context_max"] == 131072


def test_a_probe_that_fails_publishes_unknown_not_a_crashed_heartbeat(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(node_queue, "upsert_worker_heartbeat",
                        lambda **kw: seen.update(kw))

    def boom():
        raise OSError("server gone")

    claim_worker.HeartbeatEmitter(queue="gpu", host_label="box-a-gpu",
                                  llm_context_fn=boom).emit_once()
    assert seen["llm_context_length"] is None and seen["llm_context_max"] is None


def test_a_cpu_worker_does_not_even_name_the_window_columns(monkeypatch):
    """Deploy order: a cpu worker of this engine can start against a database the
    orchestrator has not migrated to 0028 yet, and naming a column that does not exist
    would flag the whole lane dead over a field it never uses."""
    seen: dict = {}
    monkeypatch.setattr(node_queue, "upsert_worker_heartbeat",
                        lambda **kw: seen.update(kw))
    claim_worker.HeartbeatEmitter(queue="cpu", host_label="box-a-cpu").emit_once()
    assert "llm_context_length" not in seen and "llm_context_max" not in seen


# ── claim ────────────────────────────────────────────────────────────────────


def test_both_gpu_lanes_claim_against_the_window_they_serve(monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(node_queue, "claim_next_gpu_job",
                        lambda *a, **kw: seen.append(kw) or None)
    w = _worker()
    w._llm_context_state = (32768, 131072)
    w._llm_gpu_ok = True
    w._spill_paused = lambda: False
    w._gpu_capacity_for_claim = lambda: (None, [], False)
    w._known_models = lambda ids=None: []
    w._warm_state = lambda: ([], [])
    w._kind_caps = lambda: {}
    w._gpu_slot_budget = lambda: 1
    w._pool_gpu_serving_ok = lambda: True
    w._claim()
    w._claim_pool()
    assert [kw.get("serving_context") for kw in seen] == [32768, 32768]


# ── ask ──────────────────────────────────────────────────────────────────────


def test_a_queued_job_the_box_could_grow_into_opens_a_resize(monkeypatch):
    _lever(monkeypatch, configured=32768)
    monkeypatch.setattr(node_queue, "context_switch_need", lambda **kw: 130000)
    w = _worker()
    w._llm_context_state = (32768, 262144)

    assert w._maybe_request_context_switch() is True
    assert w._switch.resize is True
    assert w._switch.target == "ollama"
    assert w._switch.target_context == 130048, "rounded up to a whole granule"
    assert w._switch.blocks_claims() is True


def test_no_demand_no_restart(monkeypatch):
    _lever(monkeypatch, configured=32768)
    monkeypatch.setattr(node_queue, "context_switch_need", lambda **kw: None)
    w = _worker()
    w._llm_context_state = (32768, 262144)
    assert w._maybe_request_context_switch() is False
    assert w._switch.state == server_switch.IDLE


def test_a_box_with_no_lever_never_opens_one(monkeypatch):
    _lever(monkeypatch, exists=False)
    monkeypatch.setattr(node_queue, "context_switch_need", lambda **kw: 131072)
    w = _worker()
    w._llm_context_state = (32768, 262144)
    assert w._maybe_request_context_switch() is False


def test_a_switch_already_in_flight_is_not_disturbed_or_even_asked_about(monkeypatch):
    """Not even the demand READ: this runs on every heartbeat tick, and a box that is
    already out of service has nothing to decide."""
    _lever(monkeypatch, configured=32768)
    reads: list = []
    monkeypatch.setattr(node_queue, "context_switch_need",
                        lambda **kw: reads.append(kw) or 131072)
    w = _worker()
    w._llm_context_state = (32768, 262144)
    w._switch.request("vllm")
    assert w._maybe_request_context_switch() is False
    assert w._switch.target == "vllm" and w._switch.resize is False
    assert reads == []


def test_the_kill_switch_stops_the_asking_too(monkeypatch):
    _lever(monkeypatch, configured=32768)
    monkeypatch.setenv(llm_context_lever.DISABLE_ENV, "1")
    monkeypatch.setattr(node_queue, "context_switch_need", lambda **kw: 131072)
    w = _worker()
    w._llm_context_state = (32768, 262144)
    assert w._maybe_request_context_switch() is False


# ── perform ──────────────────────────────────────────────────────────────────


def test_the_resize_is_performed_by_the_lever_and_its_verdict_goes_back(monkeypatch):
    _lever(monkeypatch, configured=32768)
    calls: list[tuple] = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: calls.append((kind, n)) or
                        {"status": "started", "reason": "recreated"})
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: ["ollama"])
    w = _worker()
    w._switch.request("ollama", context_length=131072, resize=True)
    w._drive_server_switch()                      # DRAINING → WARMING, issues the start

    assert calls == [("ollama", 131072)]
    assert w._switch.last_start["status"] == "started"
    assert w._switch.state == server_switch.WARMING


def test_the_resize_completes_only_once_the_server_serves_that_window(monkeypatch):
    _lever(monkeypatch, configured=32768)
    monkeypatch.setattr(llm_context_lever, "set_context", lambda *a, **k: {"status": "started"})
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: ["ollama"])
    w = _worker()
    w._switch.request("ollama", context_length=131072, resize=True)
    w._drive_server_switch()
    assert w._switch.state == server_switch.WARMING, "answering at 32768 is not ready"

    _lever(monkeypatch, configured=131072)
    w._drive_server_switch()
    assert w._switch.state == server_switch.IDLE


def test_a_server_that_answers_but_cannot_say_its_window_is_not_ready_yet(monkeypatch):
    """Completing means resuming claims for jobs that need that window. A window the
    box cannot READ is not evidence it has one — the switch waits (and its own budget
    ends the wait) rather than assuming."""
    _lever(monkeypatch, configured=None)
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [])
    monkeypatch.setattr(llm_context_lever, "set_context", lambda *a, **k: {"status": "started"})
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: ["ollama"])
    w = _worker()
    w._switch.request("ollama", context_length=131072, resize=True)
    w._drive_server_switch()
    w._drive_server_switch()
    assert w._switch.state == server_switch.WARMING


def test_a_resize_that_left_the_server_dark_is_put_back(monkeypatch):
    """The dangerous failure: the container was recreated with a window its server
    cannot honour, so it never comes up. Aborting alone would leave the box with NO
    LLM at all — restoring the operator's own value is what brings it back."""
    _lever(monkeypatch, configured=131072)
    restored: list[tuple] = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: restored.append((kind, n)) or
                        {"status": "started"})
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: [])
    w = _worker(switch=server_switch.ServerSwitch(label="box-a-gpu",
                                                  warm_timeout_s=0.0))
    w._switch.request("ollama", context_length=131072, resize=True)
    w._drive_server_switch()                      # start issued
    w._drive_server_switch()                      # warm budget gone → abort

    assert w._switch.state == server_switch.IDLE
    assert restored[-1] == ("ollama", None), "back to the operator's own window"


def test_a_kind_switch_starts_the_incoming_server_at_the_window_asked_for(monkeypatch):
    """R4: a server being started anyway comes up at the window the queue needs, rather
    than coming up wrong and being restarted a minute later."""
    _lever(monkeypatch, configured=None)
    calls: list[tuple] = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: calls.append((kind, n)) or
                        {"status": "started"})
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: [])
    monkeypatch.setattr(llm_probe, "loaded_models", lambda url, **k: [])
    monkeypatch.setattr(llm_probe, "vllm_served_models", lambda url, **k: [])
    w = _worker(served=("ollama",))
    w._switch.request("vllm", context_length=65536)
    w._drive_server_switch()                      # DRAINING → SWITCHING (nothing loaded)
    w._drive_server_switch()                      # SWITCHING → WARMING: start it

    assert calls == [("vllm", 65536)]


def test_a_plain_kind_switch_still_uses_the_hosts_own_start_lever(monkeypatch):
    """No window asked for ⇒ byte-identical to the pre-0028 path."""
    from queue_workflows import config as _config

    started: list = []
    cfg = _config.get_config()
    cfg.vllm_start_fn = lambda model: started.append(model) or None
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda *a, **k: pytest.fail("must not touch the context lever"))
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: [])
    monkeypatch.setattr(llm_probe, "loaded_models", lambda url, **k: [])
    monkeypatch.setattr(llm_probe, "vllm_served_models", lambda url, **k: [])
    try:
        w = _worker()
        w._switch.request("vllm")
        w._drive_server_switch()
        w._drive_server_switch()
        assert started == [None]
    finally:
        cfg.vllm_start_fn = None
