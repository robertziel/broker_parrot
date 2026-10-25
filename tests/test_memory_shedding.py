"""A box in memory danger SHEDS its LLM servers — it is never parked.

THE DESIGN RULE: don't park. Stop and unload the server; if memory is missing, the
server will start again, adjusted to RAM. Never park. Parking is an
operator lever that takes a lane out of service until an operator returns it; a box
that has run out of memory needs the opposite — the biggest thing the engine controls
given back NOW, and the box left ON so that the next start is planned against what is
actually free (test_vllm_plan_worker.py) and everything that fits keeps flowing.

THE FAILURE IT PREVENTS: a large co-tenant training run lands on a unified-memory box
beside a serving vLLM and ollama's resident model; MemAvailable falls to a few GB or
less and sshd can no longer fork. Without the sentinel nothing in the engine gives
memory back, because the only memory guard ran on a START — a serving server was
never re-examined — and parking the lanes by hand is exactly what this design forbids.
"""
from __future__ import annotations

import pytest

from queue_workflows import claim_worker, llm_context, llm_context_lever, llm_probe, worker_control
from queue_workflows import server_switch
from queue_workflows.config import get_config
from queue_workflows.llm_backends import factory as _factory

GiB = 2**30


def _worker(served=("vllm",)):
    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
            self._box_served = list(served)
            self._switch = server_switch.ServerSwitch(label="box-b-gpu")
    w = _W()
    w._physical_box = lambda: "box-b"
    w._pool_parallelism_configured = lambda: 3
    w._get_box_lease = lambda: None
    w._gpu_active_count = lambda: 0
    w._refresh_box_llm_state = lambda: None
    w._maybe_request_context_switch = lambda: False
    w._box_held_model = None
    w._pool_inflight = 0
    w._inline_running = False
    return w


def _container(running):
    return llm_context_lever.ServerContainer(
        kind="vllm", name="vllm", exists=True, running=running, configured=262144,
        utilization=0.42, utilization_now=0.42, max_num_seqs=8, max_num_seqs_now=8,
        served_name="chat-model-a:latest", fingerprint="fp", container_id="c1",
        started_at="2024-01-01T00:00:00Z")


@pytest.fixture(autouse=True)
def _box(monkeypatch):
    llm_context_lever.reset_cache_for_tests()
    monkeypatch.setattr(_factory, "resolve_base_url", lambda kind="ollama": f"http://{kind}:1")
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [])
    monkeypatch.setattr(llm_probe, "ollama_model_sizes", lambda url, **k: {})
    monkeypatch.setattr(llm_probe, "ollama_model_info", lambda url, m, **k: {})
    monkeypatch.setattr(llm_probe, "vllm_max_model_len", lambda url, **k: 262144)
    monkeypatch.setattr(llm_probe, "vllm_served_models", lambda url, **k: ["chat-model-a:latest"])
    monkeypatch.setattr(llm_probe, "vllm_kv_capacity_tokens", lambda url, **k: None)
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: [])
    monkeypatch.setattr(llm_probe, "loaded_models", lambda url, **k: [])
    monkeypatch.setattr(llm_context_lever, "available", lambda kind, **k: True)
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: _container(running=True))
    monkeypatch.setattr(llm_context_lever, "observe_vllm", lambda **k: None)
    monkeypatch.setattr(llm_context, "mem_total_bytes", lambda *a, **k: 121 * GiB)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 60 * GiB)
    cfg = get_config()
    cfg.vllm_stop_fn = None
    cfg.vllm_start_fn = lambda model: None
    yield
    cfg.vllm_stop_fn = None
    cfg.vllm_start_fn = None
    llm_context_lever.reset_cache_for_tests()


def _dying(monkeypatch, *, available_gib=5, resident=("chat-model-a:latest",)):
    stopped: list = []
    unloaded: list = []
    get_config().vllm_stop_fn = lambda: stopped.append(True) or True
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: available_gib * GiB)
    monkeypatch.setattr(llm_probe, "loaded_models_info",
                        lambda url, **k: [{"name": n, "size": 17 * GiB} for n in resident])
    monkeypatch.setattr(llm_probe, "unload_ollama_models",
                        lambda url, names, **k: unloaded.extend(names))
    return stopped, unloaded


def test_a_dying_box_stops_its_serving_vllm_and_unloads_ollama(monkeypatch, caplog):
    stopped, unloaded = _dying(monkeypatch)
    w = _worker()
    with caplog.at_level("ERROR", logger="queue_workflows.claim_worker"):
        w._box_lease_tick()
    assert stopped == [True]
    assert unloaded == ["chat-model-a:latest"]
    assert any("SHED" in r.getMessage() and "5.0 GB" in r.getMessage() for r in caplog.records)


def test_shedding_never_parks(monkeypatch):
    """The whole point. worker_controls is the operator's; the engine does not write it
    when it sheds — the lane stays ON and keeps claiming what fits."""
    import inspect

    _dying(monkeypatch)
    w = _worker()
    writes: list = []
    monkeypatch.setattr(worker_control, "disable_worker",
                        lambda *a, **k: writes.append(a) or pytest.fail("parked!"))
    w._box_lease_tick()
    assert writes == []
    import ast, textwrap

    fn = claim_worker.ClaimWorker._shed_llm_if_box_is_dying
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    if ast.get_docstring(node):
        node.body = node.body[1:]
    code = ast.unparse(node)                    # docstring and comments gone: code only
    assert "disable_worker" not in code and "worker_controls" not in code


def test_a_box_that_is_fine_sheds_nothing(monkeypatch):
    stopped, unloaded = _dying(monkeypatch, available_gib=40)
    w = _worker()
    w._box_lease_tick()
    assert stopped == [] and unloaded == []


def test_an_unreadable_meminfo_is_not_danger(monkeypatch):
    stopped, unloaded = _dying(monkeypatch)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: None)
    w = _worker()
    w._box_lease_tick()
    assert stopped == [] and unloaded == []


def test_shedding_is_rate_limited_so_a_slow_release_is_not_hammered(monkeypatch):
    """Memory comes back over seconds, not instantly; a docker stop in flight and a
    second stop a tick later would race each other and spam the log."""
    stopped, _u = _dying(monkeypatch)
    w = _worker()
    now = [1000.0]
    monkeypatch.setattr(claim_worker.time, "monotonic", lambda: now[0])
    w._box_lease_tick()
    now[0] += 10
    w._box_lease_tick()
    assert stopped == [True], "one stop inside the cooldown"
    now[0] += claim_worker.ClaimWorker.SHED_COOLDOWN_S + 1
    w._box_lease_tick()
    assert stopped == [True, True], "still dying after the cooldown: shed again"


def test_a_warming_vllm_is_abandoned_not_just_stopped(monkeypatch):
    """A warm in flight is stopped AND its switch abandoned, so the tick does not go
    on to probe for a server it just gave back (the warm watchdog's contract)."""
    stopped, _u = _dying(monkeypatch, available_gib=60)
    w = _worker()
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert w._switch.state == "warming"
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 5 * GiB)
    w._box_lease_tick()
    assert stopped == [True]
    assert w._switch.state == "idle"


def test_after_shedding_the_next_start_is_planned_against_what_is_free(monkeypatch):
    """'The server will start again, adjusted to RAM': once memory returns and demand
    asks, the plan sizes the pool to what is free — not to what it had before."""
    from queue_workflows import node_queue

    stopped, _u = _dying(monkeypatch)
    w = _worker()
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: False)
    w._box_lease_tick()
    assert stopped == [True]
    # the trainer finishes: 70 GB free again; a client asks for vLLM. The container the
    # operator seeded allows a 0.75 pool; the plan takes only what two requests need.
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 70 * GiB)
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: llm_context_lever.ServerContainer(
        kind="vllm", name="vllm", exists=True, running=False, configured=262144,
        utilization=0.75, utilization_now=0.75, max_num_seqs=8, max_num_seqs_now=8,
        served_name="chat-model-a:latest", fingerprint="fp", container_id="c1",
        started_at="2024-01-01T00:00:00Z"))
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", {
        "weights_bytes": int(18.74 * GiB), "kv_bytes": int(42.23 * GiB), "kv_tokens": 684_056,
        "window": 262_144, "utilization": 0.42, "total_bytes": 121 * GiB, "fingerprint": "fp"})
    calls: list = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: calls.append(k) or {"status": "started"})
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    (kw,) = calls
    # 62 GB usable: weights 18.7 + 2 x 16.2 GB of cache, +5 % = 53.7 GB → 0.45 of 121,
    # not the 0.75 the operator allows and not the 0.42 it ran at before.
    assert kw["max_num_seqs"] == 2
    assert kw["utilization"] == pytest.approx(0.45, abs=0.011)


def test_no_stop_lever_is_reported_not_swallowed(monkeypatch, caplog):
    _dying(monkeypatch)
    get_config().vllm_stop_fn = None
    w = _worker()
    with caplog.at_level("ERROR", logger="queue_workflows.claim_worker"):
        w._box_lease_tick()
    assert any("no stop lever" in r.getMessage() for r in caplog.records)


def test_only_the_gpu_lane_sheds():
    """The cpu and download lanes hold no LLM server; they must not even look."""
    import inspect

    src = inspect.getsource(claim_worker.ClaimWorker._box_lease_tick)
    assert "_shed_llm_if_box_is_dying()" in src
    assert 'on_tick=(self._box_lease_tick if self.queue == "gpu" else None)' in \
        inspect.getsource(claim_worker.ClaimWorker)


# ── the escalation: when our own memory is not enough ────────────────────────
#
# THE DESIGN RULE: never stop a run merely because it is on the box; the RAM guard
# stops it only when memory is exceeded. Presence on the box is not the
# offence — memory is. So a non-broker GPU run shares the box freely, and the shed path
# reaches it only in the order that etiquette demands: the broker gives back ITS OWN
# servers first, re-reads, and only if the box is STILL below the floor does it stop
# what else is holding the card.


def _outsiders(monkeypatch, readings):
    """`readings` are consecutive MemAvailable answers (the second is read after our
    own servers have been stopped)."""
    from queue_workflows import rogue_llm

    seen = iter(readings)
    last = [readings[-1]]

    def available(*a, **k):
        try:
            last[0] = next(seen)
        except StopIteration:
            pass
        return last[0]
    monkeypatch.setattr(llm_context, "mem_available_bytes", available)
    stopped: list = []
    monkeypatch.setattr(rogue_llm, "stop_gpu_outsiders",
                        lambda **kw: stopped.append(kw.get("why") or "") or ["outsider-a"])
    return stopped


def test_the_broker_gives_back_its_own_first_and_only_then_the_outsiders(monkeypatch):
    """5 GB free; our vLLM and ollama go; still 4 GB; the outside trainer is what
    is left to give."""
    ours, unloaded = _dying(monkeypatch)
    outs = _outsiders(monkeypatch, [5 * GiB, 4 * GiB])
    w = _worker()
    w._box_lease_tick()
    assert ours == [True] and unloaded == ["chat-model-a:latest"]
    assert len(outs) == 1 and "4.0 GB" in outs[0]


def test_an_outsider_is_left_alone_when_our_own_memory_was_enough(monkeypatch):
    """The common case: the broker was the one holding the box, and giving that back
    fixed it. Nobody else's work is touched."""
    ours, _unloaded = _dying(monkeypatch)
    outs = _outsiders(monkeypatch, [5 * GiB, 60 * GiB])
    w = _worker()
    w._box_lease_tick()
    assert ours == [True]
    assert outs == []


def test_a_box_that_is_fine_never_reaches_the_outsiders(monkeypatch):
    ours, _unloaded = _dying(monkeypatch, available_gib=40)
    outs = _outsiders(monkeypatch, [40 * GiB, 40 * GiB])
    w = _worker()
    w._box_lease_tick()
    assert ours == [] and outs == []


def test_the_escalation_is_reported_as_what_it_is(monkeypatch, caplog):
    _dying(monkeypatch)
    _outsiders(monkeypatch, [2 * GiB, 1 * GiB])
    w = _worker()
    with caplog.at_level("ERROR", logger="queue_workflows.claim_worker"):
        w._box_lease_tick()
    assert any("still short" in r.getMessage().lower() and "outsider-a" in r.getMessage()
               for r in caplog.records), "say which outsiders went, and that ours had gone first"


def test_a_box_that_cannot_read_its_memory_after_shedding_touches_no_outsider(monkeypatch):
    """The 0015 convention on unknowns, at the sharpest end: stopping somebody else's
    training on a reading the engine does not have would be the worst false positive
    in the system."""
    from queue_workflows import rogue_llm

    _dying(monkeypatch)
    called: list = []
    monkeypatch.setattr(rogue_llm, "stop_gpu_outsiders", lambda **kw: called.append(kw) or [])
    w = _worker()
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: None)
    assert w._stop_gpu_outsiders_if_still_short() == []
    assert called == []
