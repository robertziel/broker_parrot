"""The gpu worker: a vLLM start planned from its remembered profile, and a box that
advertises the concurrency its cache really holds. See test_vllm_plan.py for the math."""
from __future__ import annotations

import pytest

from queue_workflows import claim_worker, llm_context, llm_context_lever, llm_probe, node_queue
from queue_workflows import server_switch
from queue_workflows.llm_backends import factory as _factory

GiB = 2**30
FP = "fingerprint-1"
RECORD = {"weights_bytes": int(18.74 * GiB), "kv_bytes": int(42.23 * GiB), "kv_tokens": 684_056,
          "window": 262_144, "utilization": 0.42, "total_bytes": 121 * GiB, "fingerprint": FP}


def _worker(served=("vllm",)):
    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
            self._box_served = list(served)
            self._switch = server_switch.ServerSwitch(label="box-b-gpu")
    w = _W()
    w._physical_box = lambda: "box-b"
    w._pool_parallelism_configured = lambda: 8
    w._get_box_lease = lambda: None
    w._gpu_active_count = lambda: 0
    return w


def _container(*, running, started_at="2024-01-01T00:00:00Z", fingerprint=FP, max_seqs=8):
    return llm_context_lever.ServerContainer(
        kind="vllm", name="vllm", exists=True, running=running, configured=262144,
        utilization=0.75, utilization_now=0.42, max_num_seqs=max_seqs, max_num_seqs_now=max_seqs,
        served_name="chat-model-a:latest", fingerprint=fingerprint, container_id="c1",
        started_at=started_at)


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
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: _container(running=False))
    monkeypatch.setattr(llm_context_lever, "observe_vllm", lambda **k: None)
    monkeypatch.setattr(llm_context, "mem_total_bytes", lambda *a, **k: 121 * GiB)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 110 * GiB)
    yield
    llm_context_lever.reset_cache_for_tests()


# ── the start ─────────────────────────────────────────────────────────────────


def test_with_a_remembered_profile_the_start_is_sized_to_need(monkeypatch):
    """A box with 110 GB free: the old rule starts at the operator's 0.75 (91 GB). With the
    profile of its last start: 4 requests of 262k fit → pool 0.73, --max-num-seqs 4."""
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", RECORD)
    calls: list = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: calls.append((kind, n, k)) or {"status": "started"})
    w = _worker()
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert len(calls) == 1
    kind, window, kw = calls[0]
    assert (kind, window) == ("vllm", None)
    assert kw["utilization"] == pytest.approx(0.73, abs=0.011)
    assert kw["max_num_seqs"] == 4


def test_a_profile_from_another_command_line_is_not_used(monkeypatch):
    """Re-quantised, a different KV dtype, another batch size: the numbers belong to
    another memory shape. The current rule applies — and 110 GB free fits the operator's
    pool, so the host's plain start is used."""
    from queue_workflows import config as _config

    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest",
                                  {**RECORD, "fingerprint": "other-flags"})
    started: list = []
    cfg = _config.get_config()
    cfg.vllm_start_fn = lambda model: started.append(model) or None
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda *a, **k: pytest.fail("no profile → no recreate"))
    try:
        w = _worker()
        w._switch.request("vllm")
        w._drive_server_switch(); w._drive_server_switch()
        assert started == [None]
    finally:
        cfg.vllm_start_fn = None


def test_a_box_that_cannot_hold_one_request_refuses_and_quarantines(monkeypatch):
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", RECORD)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 40 * GiB)
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda *a, **k: pytest.fail("must not start"))
    w = _worker()
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert w._switch.quarantined("vllm") is True


def test_the_plan_uses_the_window_the_switch_asked_for(monkeypatch):
    """A resize to 32k tokens costs 2 GB per request: all 8 fit in 36 GB — pool 0.31."""
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", RECORD)
    calls: list = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: calls.append((n, k)) or {"status": "started"})
    w = _worker()
    w._switch.request("vllm", 32_768)
    w._drive_server_switch(); w._drive_server_switch()
    (window, kw), = calls
    assert window == 32_768 and kw["max_num_seqs"] == 8
    assert kw["utilization"] == pytest.approx(0.31, abs=0.011)


# ── the observation ───────────────────────────────────────────────────────────


def _serving(monkeypatch, *, capacity=684_056, profile=True, started_at="2024-01-01T00:00:00Z"):
    monkeypatch.setattr(llm_context_lever, "describe",
                        lambda kind, **k: _container(running=True, started_at=started_at))
    monkeypatch.setattr(llm_probe, "vllm_kv_capacity_tokens", lambda url, **k: capacity)
    seen = {"n": 0}

    def observe(**k):
        seen["n"] += 1
        return llm_context.VllmProfile.from_record(RECORD) if profile else None
    monkeypatch.setattr(llm_context_lever, "observe_vllm", observe)
    return seen


def test_the_box_advertises_the_concurrency_its_cache_holds(monkeypatch):
    """684,056 tokens of cache at a 262,144 window = 2 full requests. PAR 8 → 2: the pool
    lane, the per-kind cap and the heartbeat's concurrency all say 2."""
    _serving(monkeypatch)
    w = _worker()
    assert w._pool_parallelism() == 8, "nothing observed yet: the operator's PAR"
    w._observe_llm_context()
    assert w._pool_parallelism() == 2
    assert w._gpu_slot_budget() == 2
    assert w._kind_caps()["vllm"] == 2


def test_the_starts_numbers_are_remembered_in_the_catalog(monkeypatch):
    _serving(monkeypatch)
    w = _worker()
    w._observe_llm_context()
    rec = node_queue.llm_profile("box-b-gpu", "vllm", "chat-model-a:latest")
    assert rec is not None and rec["kv_tokens"] == 684_056 and rec["fingerprint"] == FP


def test_the_observation_is_made_once_per_start(monkeypatch):
    seen = _serving(monkeypatch)
    w = _worker()
    w._observe_llm_context(); w._observe_llm_context(); w._observe_llm_context()
    assert seen["n"] == 1
    _serving(monkeypatch, started_at="2024-01-01T01:00:00Z")      # restarted: a new start
    w._observe_llm_context()
    assert seen["n"] == 1, "a fresh recorder; the old one saw its single call"


def test_the_metrics_gauge_alone_still_caps_par_when_the_log_has_no_profile(monkeypatch):
    _serving(monkeypatch, capacity=1_000_000, profile=False)
    w = _worker()
    w._observe_llm_context()
    assert w._pool_parallelism() == 3
    assert node_queue.llm_profile("box-b-gpu", "vllm", "chat-model-a:latest") is None


def test_a_log_that_never_profiles_is_given_up_on(monkeypatch):
    seen = _serving(monkeypatch, capacity=None, profile=False)
    w = _worker()
    for _ in range(claim_worker.ClaimWorker._VLLM_PROFILE_ATTEMPTS + 5):
        w._observe_llm_context()
    assert seen["n"] == claim_worker.ClaimWorker._VLLM_PROFILE_ATTEMPTS


def test_the_cap_is_lifted_when_vllm_stops_serving(monkeypatch):
    _serving(monkeypatch)
    w = _worker()
    w._observe_llm_context()
    assert w._pool_parallelism() == 2
    w._box_served = ["ollama"]
    w._observe_llm_context()
    assert w._pool_parallelism() == 8


def test_a_vllm_that_is_not_running_is_not_observed(monkeypatch):
    seen = _serving(monkeypatch)
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: _container(running=False))
    w = _worker()
    w._observe_llm_context()
    assert seen["n"] == 0 and w._pool_parallelism() == 8


# ── a pool already committed is not re-judged ──────────────────────────


def _stop_recorder(monkeypatch, *, available_gib, complete, started_at="2024-01-01T00:00:00Z"):
    from queue_workflows import config as _config

    stopped: list = []
    cfg = _config.get_config()
    cfg.vllm_stop_fn = lambda: stopped.append(True) or True
    # A real deployment always has a start lever wired; without one the switch reports every
    # start as permanently impossible and the test would measure that instead.
    cfg.vllm_start_fn = lambda model: None
    monkeypatch.setattr(llm_context_lever, "describe",
                        lambda kind, **k: _container(running=True, started_at=started_at))
    monkeypatch.setattr(llm_context_lever, "observe_vllm",
                        lambda **k: llm_context.VllmProfile.from_record(RECORD) if complete else None)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: available_gib * GiB)
    return stopped


def test_a_loaded_vllm_is_not_stopped_because_a_fresh_start_would_not_fit(monkeypatch):
    """Observed in production: a vLLM allocated its cache ('Available KV cache
    memory: … Maximum concurrency … Nx') after many minutes of loading. The
    worker was restarted shortly after; the fresh process read MemAvailable — low BECAUSE of
    that very pool — planned a START that could not fit and stopped the server that had
    just finished. The memory gate governs STARTS: a running server whose startup is
    complete is judged by whether the box is in DANGER, not by whether a start it is not
    making would fit. (With 40 GB still free, it is not in danger.)"""
    stopped = _stop_recorder(monkeypatch, available_gib=40, complete=True)
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda *a, **k: pytest.fail("no start: one is already running"))
    try:
        w = _worker()
        w._observe_llm_context()                 # the start is seen to be complete
        w._switch.request("vllm")
        w._drive_server_switch(); w._drive_server_switch()
        assert stopped == [], "a loaded server must survive a start it is not making"
        assert w._switch.quarantined("vllm") is not True
    finally:
        from queue_workflows import config as _config
        _config.get_config().vllm_stop_fn = None
        _config.get_config().vllm_start_fn = None


def test_the_danger_line_is_an_absolute_floor_not_a_multiple_of_the_reserve(monkeypatch):
    """Observed in production: the reserve was raised to 32 GB on a unified-memory box
    (8 GB never modelled the other GPU and CPU tenants that share the box) — and the
    danger line, written as 2x the reserve, moved to 64 GB with it. A box holding a
    legitimate 80 GB pool sits at ~34 GB free and would have been declared dying and
    had its server killed, every tick.

    The reserve is a PLANNING margin: how much the next start must leave behind. Danger
    is a different quantity — how little is left before fork() fails and the kernel
    starts refusing GPU allocations — and it does not scale with planning ambition.
    A box has died at MemAvailable 8 GB (MemFree under 2 GB), so the floor is absolute."""
    from queue_workflows import llm_context

    w = _worker()
    monkeypatch.setattr(llm_context, "reserve_bytes", lambda: 32 * GiB)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 34 * GiB)
    assert w._box_memory_in_danger() is False, "34 GB free is not a dying box"
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 8 * GiB)
    assert w._box_memory_in_danger() is True
    monkeypatch.setattr(llm_context, "reserve_bytes", lambda: 8 * GiB)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 34 * GiB)
    assert w._box_memory_in_danger() is False, "the floor does not move with the reserve"


def test_the_danger_floor_is_a_knob(monkeypatch):
    from queue_workflows import claim_worker as cw, llm_context

    w = _worker()
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 20 * GiB)
    assert w._box_memory_in_danger() is False
    monkeypatch.setenv("QUEUE_WORKFLOWS_LLM_DANGER_FLOOR_MB", "30720")
    assert w._box_memory_in_danger() is True
    monkeypatch.setenv("QUEUE_WORKFLOWS_LLM_DANGER_FLOOR_MB", "nonsense")
    assert w._box_memory_in_danger() is False, "a bad value falls back to the default"
    assert cw.DEFAULT_DANGER_FLOOR_MB == 12288


def test_a_box_actually_in_danger_stops_even_a_loaded_vllm(monkeypatch):
    """The other half of the same incident: MemAvailable around 13 GB —
    and the KERNEL had already logged `NVRM: … Out of memory`, because MemAvailable
    counts reclaimable page cache that a CUDA allocation on a unified-memory box cannot take
    (MemFree was under 2 GB). So the danger line sits above the reserve, not at it: below
    the DANGER FLOOR (an absolute number of free bytes, never a multiple of the
    reserve) the box is in trouble and the LLM server is the biggest thing the engine
    can give back."""
    stopped = _stop_recorder(monkeypatch, available_gib=8, complete=True)
    try:
        w = _worker()
        w._observe_llm_context()
        w._switch.request("vllm")
        w._drive_server_switch(); w._drive_server_switch()
        assert stopped == [True]
    finally:
        from queue_workflows import config as _config
        _config.get_config().vllm_stop_fn = None
        _config.get_config().vllm_start_fn = None


def test_a_vllm_still_loading_into_memory_the_box_lacks_is_stopped_at_any_reading(monkeypatch):
    """The case the stop was written for (37 GB free): a start whose pool
    is still AHEAD of it — no complete startup in its log — heading for 91 GB. The box
    is not in danger YET; it will be, because of this."""
    stopped = _stop_recorder(monkeypatch, available_gib=37, complete=False)
    try:
        w = _worker()
        w._observe_llm_context()
        w._switch.request("vllm")
        w._drive_server_switch(); w._drive_server_switch()
        assert stopped == [True]
        assert w._switch.quarantined("vllm") is True
    finally:
        from queue_workflows import config as _config
        _config.get_config().vllm_stop_fn = None
        _config.get_config().vllm_start_fn = None


def test_a_loaded_pool_is_the_one_this_start_took_not_an_older_one(monkeypatch):
    """The commitment belongs to a (container, start) pair — a restart at a bigger pool
    is a new start with its memory still ahead of it."""
    stopped = _stop_recorder(monkeypatch, available_gib=40, complete=True)
    try:
        w = _worker()
        w._observe_llm_context()
        restarted = _stop_recorder(monkeypatch, available_gib=40, complete=False,
                                   started_at="2024-01-01T02:00:00Z")
        w._observe_llm_context()
        w._switch.request("vllm")
        w._drive_server_switch(); w._drive_server_switch()
        assert stopped == [] and restarted == [True], \
            "a newer start is not covered by the older commitment"
    finally:
        from queue_workflows import config as _config
        _config.get_config().vllm_stop_fn = None
        _config.get_config().vllm_start_fn = None


def test_a_stopped_container_is_planned_as_a_start_however_lately_it_ran(monkeypatch):
    """The exemption is for a server that IS running. One the engine (or the park
    watchdog) stopped keeps its observation, and starting it again is a start like any
    other: its pool has to fit the box as it is now."""
    stopped = _stop_recorder(monkeypatch, available_gib=60, complete=True)
    calls: list = []
    try:
        w = _worker()
        # a vLLM-only box, so the "one request is not worth a vLLM" rule below does
        # not apply — this test is about a stopped container being a start.
        monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: False)
        w._observe_llm_context()                 # …which also remembered its profile
        monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: _container(running=False))
        monkeypatch.setattr(llm_context_lever, "set_context",
                            lambda kind, n, **k: calls.append(k) or {"status": "started"})
        w._switch.request("vllm")
        w._drive_server_switch(); w._drive_server_switch()
        (kw,) = calls
        # 52 GB usable: weights 18.7 + one 262k request 16.2, +5 % = 36.7 GB → 0.31, and
        # only ONE request fits. The profile it just took is what sizes its next start.
        assert kw["max_num_seqs"] == 1
        assert kw["utilization"] == pytest.approx(0.31, abs=0.011)
        assert stopped == []
    finally:
        from queue_workflows import config as _config
        _config.get_config().vllm_stop_fn = None
        _config.get_config().vllm_start_fn = None


def test_a_box_that_cannot_read_its_own_memory_is_not_called_endangered(monkeypatch):
    """The 0015 convention on unknowns, here too: only a KNOWN shortage acts. A box that
    cannot read /proc/meminfo must not have its loaded server stopped every tick."""
    stopped = _stop_recorder(monkeypatch, available_gib=40, complete=True)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: None)
    try:
        w = _worker()
        w._observe_llm_context()
        assert w._box_memory_in_danger() is False
        w._switch.request("vllm")
        w._drive_server_switch(); w._drive_server_switch()
        assert stopped == []
    finally:
        from queue_workflows import config as _config
        _config.get_config().vllm_stop_fn = None
        _config.get_config().vllm_start_fn = None


# ── one request is not worth a vLLM (a deliberate policy) ────────────────────
#
# vLLM's whole advantage over ollama is BATCHING: one pool, many requests in flight.
# When the memory left on a box can only pay for ONE concurrent request, that pool buys
# nothing ollama does not already give — ollama serialises at a fraction of the memory
# and needs no switch, no weight reload, no warm window. So a plan that can afford only
# one request is not a reason to start vLLM on a box that can serve ollama instead.


def _one_request_box(monkeypatch, *, ollama_here=True):
    """Enough memory for exactly one 262k request (36.7 GB of the 44 GB usable)."""
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", RECORD)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 52 * GiB)
    w = _worker()
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: ollama_here)
    return w


def test_a_pool_that_fits_one_request_does_not_start_vllm_where_ollama_can_serve(monkeypatch):
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda *a, **k: pytest.fail("ollama serves this better"))
    w = _one_request_box(monkeypatch)
    assert w._vllm_start_plan().max_num_seqs is None
    refusal = w._vllm_start_plan().refusal
    assert refusal and "1 concurrent request" in refusal and "ollama serves this box" in refusal
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert w._switch.quarantined("vllm") is True


def test_a_box_with_no_ollama_of_its_own_starts_that_single_slot_vllm_anyway(monkeypatch):
    """A vLLM-only box has no alternative: one request beats none."""
    calls: list = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: calls.append(k.get("max_num_seqs")) or
                        {"status": "started"})
    w = _one_request_box(monkeypatch, ollama_here=False)
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert calls == [1]


def test_two_requests_are_worth_it(monkeypatch):
    calls: list = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: calls.append(k.get("max_num_seqs")) or
                        {"status": "started"})
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", RECORD)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 70 * GiB)
    w = _worker()
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: True)
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert calls == [2]


def test_the_threshold_is_a_knob(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_VLLM_MIN_CONCURRENCY", "3")
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", RECORD)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 70 * GiB)
    w = _worker()
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: True)
    assert w._vllm_start_plan().refusal is not None      # 2 fit, 3 wanted
    monkeypatch.setenv("QUEUE_WORKFLOWS_VLLM_MIN_CONCURRENCY", "1")
    w.__dict__.pop("_vllm_decision", None)
    assert w._vllm_start_plan().refusal is None


def test_without_a_profile_the_rule_cannot_fire(monkeypatch):
    """No numbers, no count — the first start of a model still happens (and becomes
    the profile that makes this judgeable next time)."""
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 52 * GiB)
    w = _worker()
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: True)
    plan = w._vllm_start_plan()
    assert plan.refusal is None and plan.max_num_seqs is None


def test_a_box_serves_ollama_when_its_url_differs_from_vllms(monkeypatch):
    from queue_workflows.llm_backends import factory as f

    w = _worker()
    monkeypatch.setattr(f, "resolve_base_url",
                        lambda kind="ollama": "http://o:11434" if kind == "ollama" else "http://v:8000")
    assert w._box_has_ollama_endpoint() is True
    monkeypatch.setattr(f, "resolve_base_url", lambda kind="ollama": "http://same:8000")
    assert w._box_has_ollama_endpoint() is False
    monkeypatch.setattr(f, "resolve_base_url", lambda kind="ollama": "")
    assert w._box_has_ollama_endpoint() is False


# ── the box dying mid-warm ───────────────────────────────────────────────────
#
# Observed in production: a box's first start of a model had NO profile, so the
# plan used the configured seeded pool (most of the box) — which "fits"
# when most of the box is free. It then loaded next to the box's other tenants (other
# GPU and CPU work) and MemAvailable fell to a few GB; sshd could no longer
# fork, and the gpu heartbeat went minutes stale. Nothing intervened, because the POOL
# decision is deliberately made once per switch (a re-decision mid-load is what caused
# the recreate loop) and a start that is not being re-planned was never re-examined.
#
# Deciding the pool once is right. Watching the BOX is a different question, and it must
# keep being asked: while a server warms, a box that crosses into danger gives back the
# thing that is taking the memory.


def _warming(monkeypatch, *, available_gib):
    from queue_workflows import config as _config

    stopped: list = []
    cfg = _config.get_config()
    cfg.vllm_stop_fn = lambda: stopped.append(True) or True
    cfg.vllm_start_fn = lambda model: None
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: _container(running=True))
    monkeypatch.setattr(llm_context_lever, "observe_vllm", lambda **k: None)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 102 * GiB)
    w = _worker()
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: False)
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()     # DRAINING → WARMING, start issued
    assert w._switch.state == "warming", w._switch.state
    # the load eats the box
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: available_gib * GiB)
    return w, stopped


def test_a_warming_box_that_runs_out_of_memory_gives_the_server_back(monkeypatch):
    try:
        w, stopped = _warming(monkeypatch, available_gib=8)
        w._switch._last_try -= 999
        w._drive_server_switch()
        assert stopped == [True], "the box was dying; the pool it is taking must go back"
        assert w._switch.state != "warming", "and the switch must not keep warming"
    finally:
        from queue_workflows import config as _config
        _config.get_config().vllm_stop_fn = None
        _config.get_config().vllm_start_fn = None


def test_a_warm_that_is_merely_tight_is_left_to_finish(monkeypatch):
    """A vLLM load LOWERS MemAvailable by design — that is what taking a pool means.
    Only crossing the danger line acts; 'less than when we decided' must not."""
    try:
        w, stopped = _warming(monkeypatch, available_gib=30)
        w._switch._last_try -= 999
        w._drive_server_switch()
        assert stopped == []
        assert w._switch.state == "warming"
    finally:
        from queue_workflows import config as _config
        _config.get_config().vllm_stop_fn = None
        _config.get_config().vllm_start_fn = None


def test_the_pool_is_still_decided_only_once(monkeypatch):
    """The guard must not become a back door to re-planning: between the danger line
    and the decision, nothing is recomputed and nothing is recreated."""
    recreated: list = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: recreated.append(k) or {"status": "started"})
    try:
        w, stopped = _warming(monkeypatch, available_gib=30)
        w._switch._last_try -= 999
        w._drive_server_switch()
        assert recreated == [] and stopped == []
    finally:
        from queue_workflows import config as _config
        _config.get_config().vllm_stop_fn = None
        _config.get_config().vllm_start_fn = None


def test_a_box_that_cannot_read_its_memory_never_gives_up_a_warm(monkeypatch):
    try:
        w, stopped = _warming(monkeypatch, available_gib=8)
        monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: None)
        w._switch._last_try -= 999
        w._drive_server_switch()
        assert stopped == [] and w._switch.state == "warming"
    finally:
        from queue_workflows import config as _config
        _config.get_config().vllm_stop_fn = None
        _config.get_config().vllm_start_fn = None


def test_the_watchdog_does_not_touch_a_drain_or_an_ollama_target(monkeypatch):
    """It guards a vLLM POOL being taken. DRAINING takes no memory (the box is still
    serving what it had), and an ollama target loads per request — killing either
    because the box is tight would take a box out of service for nothing."""
    from queue_workflows import config as _config

    stopped: list = []
    cfg = _config.get_config()
    cfg.vllm_stop_fn = lambda: stopped.append(True) or True
    cfg.vllm_start_fn = lambda model: None
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 4 * GiB)
    try:
        draining = _worker()
        draining._pool_inflight = 1
        draining._switch.request("vllm")
        draining._drive_server_switch()
        assert draining._switch.state != "warming", "still on the way there"
        assert draining._abandon_warm_if_box_is_dying() is False
        assert stopped == []

        to_ollama = _worker()
        to_ollama._switch.request("ollama")
        to_ollama._switch._state = "warming"
        assert to_ollama._abandon_warm_if_box_is_dying() is False
        assert stopped == []
    finally:
        cfg.vllm_stop_fn = None
        cfg.vllm_start_fn = None


def test_the_tick_stops_there_when_the_warm_is_abandoned(monkeypatch):
    """Having given the server back, the same tick must not go on to probe for it and
    re-issue a start against the memory that just ran out."""
    from queue_workflows import config as _config

    started: list = []
    cfg = _config.get_config()
    cfg.vllm_stop_fn = lambda: True
    cfg.vllm_start_fn = lambda model: started.append(model) or None
    try:
        w, _stopped = _warming(monkeypatch, available_gib=8)
        started.clear()
        w._switch._last_try -= 999
        w._drive_server_switch()
        assert started == [], "no start after the abandon"
        assert "_vllm_decision" not in w.__dict__, "the next switch decides afresh"
    finally:
        cfg.vllm_stop_fn = None
        cfg.vllm_start_fn = None
