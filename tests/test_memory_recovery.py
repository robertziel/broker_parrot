"""Memory recovery: when RAM has been back for a minute, vLLM may come back too.

The rule: monitor RAM usage; if RAM stays free for more than a minute and the box can
switch from ollama to vLLM, do it.

A memory refusal quarantines vLLM on the switch — 900 s, doubling — which is right for
a start that failed for a reason nobody can see change, and wrong for one refused over
MEMORY: the sentinel reads that number every second and knows exactly when the box can
afford the pool again. A hold caused by memory is lifted once MemAvailable has covered
what a worthwhile vLLM needs — weights, overhead and the minimum-concurrency worth of
cache, plus the reserve — for MEMORY_RECOVERY_S continuously; the demand logic then
requests the switch as it always does and the plan sizes the start to what is free. A
dip inside the minute starts the count again; a hold for any OTHER reason (a server
that never became ready, a lever failure) is not the sentinel's to lift.
"""
from __future__ import annotations

import pytest

from queue_workflows import claim_worker, llm_context, llm_context_lever, llm_probe, node_queue
from queue_workflows import memory_sentinel, server_switch
from queue_workflows.config import get_config
from queue_workflows.llm_backends import factory as _factory

GiB = 2**30
PROFILE = {"weights_bytes": int(18.74 * GiB), "kv_bytes": int(42.23 * GiB), "kv_tokens": 684_056,
           "window": 262_144, "utilization": 0.42, "total_bytes": 121 * GiB, "fingerprint": "fp"}


def _container(*, running=False):
    return llm_context_lever.ServerContainer(
        kind="vllm", name="vllm", exists=True, running=running, configured=262144,
        utilization=0.75, utilization_now=0.42, max_num_seqs=8, max_num_seqs_now=8,
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
    monkeypatch.setattr(llm_probe, "loaded_models_info", lambda url, **k: [])
    monkeypatch.setattr(llm_context_lever, "available", lambda kind, **k: True)
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: _container())
    monkeypatch.setattr(llm_context_lever, "observe_vllm", lambda **k: None)
    monkeypatch.setattr(llm_context, "mem_total_bytes", lambda *a, **k: 121 * GiB)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 20 * GiB)
    cfg = get_config()
    cfg.vllm_stop_fn = lambda: True
    cfg.vllm_start_fn = lambda model: None
    yield
    cfg.vllm_stop_fn = None
    cfg.vllm_start_fn = None
    llm_context_lever.reset_cache_for_tests()


def _worker(monkeypatch):
    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
            self._box_served = ["ollama"]
            self._switch = server_switch.ServerSwitch(label="box-b-gpu")
    w = _W()
    w._physical_box = lambda: "box-b"
    w._pool_parallelism_configured = lambda: 8
    w._get_box_lease = lambda: None
    w._gpu_active_count = lambda: 0
    w._box_held_model = None
    w._pool_inflight = 0
    w._inline_running = False
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: True)
    now = [1000.0]
    monkeypatch.setattr(claim_worker.time, "monotonic", lambda: now[0])
    return w, now


def _refused_over_memory(w):
    """A memory refusal: 20 GB free cannot hold one 262k request → quarantined."""
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", PROFILE)
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert w._switch.quarantined("vllm") is True
    assert "memory:" in (w._switch.quarantine_reason("vllm") or "")


def test_a_minute_of_headroom_lifts_a_memory_quarantine(monkeypatch):
    w, now = _worker(monkeypatch)
    _refused_over_memory(w)
    need = w._vllm_memory_need_bytes()
    assert need is not None
    for _ in range(int(claim_worker.ClaimWorker.MEMORY_RECOVERY_S) - 1):
        now[0] += 1
        w._memory_tick(need + 1)
        assert w._switch.quarantined("vllm") is True, "not a minute yet"
    now[0] += 2
    w._memory_tick(need + 1)
    assert w._switch.quarantined("vllm") is False


def test_a_dip_inside_the_minute_starts_the_count_again(monkeypatch):
    w, now = _worker(monkeypatch)
    _refused_over_memory(w)
    need = w._vllm_memory_need_bytes()
    for _ in range(40):
        now[0] += 1
        w._memory_tick(need + 1)
    now[0] += 1
    w._memory_tick(need - 1)                     # the trainer touched memory again
    for _ in range(40):
        now[0] += 1
        w._memory_tick(need + 1)
    assert w._switch.quarantined("vllm") is True, "40 s since the dip, not 60"
    for _ in range(21):
        now[0] += 1
        w._memory_tick(need + 1)
    assert w._switch.quarantined("vllm") is False


def test_what_is_needed_is_a_worthwhile_vllm_plus_the_reserve(monkeypatch):
    """With the profile: weights + overhead + MIN_CONCURRENCY x a request's cache, +5 %,
    + the reserve — the memory a start that would not be refused needs. Without a
    profile: the minimum pool + the reserve."""
    w, _now = _worker(monkeypatch)
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", PROFILE)
    p = llm_context.VllmProfile.from_record(PROFILE)
    per_seq = 262_144 * p.per_token_bytes
    expected = int((p.weights_bytes + p.overhead_bytes + 2 * per_seq) * 1.05) + llm_context.reserve_bytes()
    assert w._vllm_memory_need_bytes() == pytest.approx(expected, rel=1e-6)
    w.__dict__.pop("_vllm_need_cache", None)
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", {**PROFILE, "fingerprint": "other"})
    assert w._vllm_memory_need_bytes() == int(0.30 * 121 * GiB) + llm_context.reserve_bytes()


def test_a_hold_for_any_other_reason_is_not_the_sentinels_to_lift(monkeypatch):
    w, now = _worker(monkeypatch)
    w._switch._quarantine_locked("vllm", "did not become ready within 1800s")
    for _ in range(120):
        now[0] += 1
        w._memory_tick(200 * GiB)
    assert w._switch.quarantined("vllm") is True


def test_after_the_lift_the_start_is_planned_against_what_is_free(monkeypatch):
    """The end of the story: memory back for a minute, the demand logic asks, the
    plan sizes the pool to the free memory and vLLM comes up beside nothing."""
    w, now = _worker(monkeypatch)
    _refused_over_memory(w)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 90 * GiB)
    need = w._vllm_memory_need_bytes()
    for _ in range(61):
        now[0] += 1
        w._memory_tick(90 * GiB)
    assert w._switch.quarantined("vllm") is False
    calls: list = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: calls.append(k) or {"status": "started"})
    assert w._switch.request("vllm") is True
    w._drive_server_switch(); w._drive_server_switch()
    (kw,) = calls
    # 82 GB usable (90 free − 8 reserve), capped by the configured 0.75 = 90.75 → 82:
    # (82/1.05 − 18.74) / 16.18 = 3 requests, pool 0.61 — sized to what is free.
    assert kw["max_num_seqs"] == 3 and kw["utilization"] < 0.75
    assert need < 90 * GiB


def test_the_sentinel_ticks_both_the_shed_and_the_recovery(monkeypatch):
    from queue_workflows import claim_worker as cw

    class _W(cw.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
    w = _W()
    started: list = []
    monkeypatch.setattr(memory_sentinel.MemorySentinel, "start", lambda self: started.append(self) or self)
    w._arm_memory_sentinel()
    (s,) = started
    assert s._act == w._memory_tick
    import inspect

    src = inspect.getsource(cw.ClaimWorker._memory_tick)
    assert "_shed_llm_if_box_is_dying(" in src and "_note_memory_recovery(" in src


def test_the_need_is_recomputed_at_most_once_a_minute(monkeypatch):
    """One catalog read a minute, not one a second."""
    w, now = _worker(monkeypatch)
    reads: list = []
    real = node_queue.llm_profile
    monkeypatch.setattr(node_queue, "llm_profile", lambda *a, **k: reads.append(a) or real(*a, **k))
    for _ in range(30):
        now[0] += 1
        w._vllm_memory_need_bytes()
    assert len(reads) == 1
    now[0] += 61
    w._vllm_memory_need_bytes()
    assert len(reads) == 2
