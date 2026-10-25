"""THE START PRE-CHECK, end to end. The contract: before a server starts, the RAM pre-check
    counts the available memory correctly, sets the vLLM server's concurrency to what fits, or
    runs ollama instead, or marks the job as not processable in the RAM available now and takes
    another one — with every edge case covered.

Each clause is a section below. The arithmetic itself is specified in
test_vllm_plan.py; this file is the CONTRACT read top to bottom: what is counted, what
is decided, and what happens to the job the box cannot take.
"""
from __future__ import annotations

import pytest

from queue_workflows import claim_worker, llm_context, llm_context_lever, llm_probe, node_queue
from queue_workflows import memory_sentinel, server_switch
from queue_workflows.llm_backends import factory as _factory
from tests._helpers import make_run

GiB = 2**30
MiB = 2**20

BOX_PROFILE = {"weights_bytes": int(18.74 * GiB), "kv_bytes": int(42.23 * GiB),
                 "kv_tokens": 684_056, "window": 262_144, "utilization": 0.42,
                 "total_bytes": 121 * GiB, "fingerprint": "fp"}
PROFILE = llm_context.VllmProfile.from_record(BOX_PROFILE)


def _plan(**kw):
    args = dict(total_bytes=121 * GiB, available_bytes=110 * GiB, reserve_bytes=8 * GiB,
                operator_utilization=0.75, operator_max_seqs=8, window=262_144,
                profile=PROFILE, min_utilization=0.30)
    args.update(kw)
    return llm_context.vllm_plan(**args)


# ── 1. available RAM is counted correctly ────────────────────────────────────


def test_the_budget_is_what_is_available_minus_the_reserve_capped_by_the_operator():
    """110 GB free, 8 GB reserve → 102 GB; the operator's 0.75 of 121 = 90.75 GB is
    the tighter cap. The plan never plans beyond the SMALLER of the two."""
    plan = _plan()
    assert plan.utilization <= 0.75
    tight = _plan(available_bytes=60 * GiB)
    assert tight.utilization <= (60 - 8) / 121 + 1e-9


@pytest.mark.parametrize("available_gib, reserve_gib", [(8, 8), (7, 8), (0, 8), (5, 0)])
def test_a_box_with_nothing_to_give_refuses_rather_than_planning_a_negative_pool(available_gib, reserve_gib):
    plan = _plan(available_bytes=available_gib * GiB, reserve_bytes=reserve_gib * GiB)
    assert plan.refusal is not None
    assert plan.utilization is None and plan.max_num_seqs is None
    no_profile = _plan(available_bytes=available_gib * GiB, reserve_bytes=reserve_gib * GiB,
                       profile=None)
    assert no_profile.refusal is not None


def test_unknown_readings_plan_nothing_and_refuse_nothing():
    """A box that cannot read its own memory must not wedge the queue (the 0015
    convention on unknowns): it starts as configured and the danger floor still
    watches it."""
    for kw in (dict(total_bytes=None), dict(available_bytes=None), dict(total_bytes=0)):
        plan = _plan(**kw)
        assert plan == llm_context.VllmPlan(None, None, None, plan.reason)


def test_exactly_at_the_minimum_utilization_is_allowed_and_a_hair_under_is_not():
    """No profile: the pool is the operator's value lowered to what is free, and the
    minimum is inclusive."""
    at = _plan(profile=None, available_bytes=int(0.30 * 121 * GiB) + 8 * GiB + 1)
    assert at.refusal is None and at.utilization == pytest.approx(0.30)
    under = _plan(profile=None, available_bytes=int(0.29 * 121 * GiB) + 8 * GiB)
    assert under.refusal is not None


def test_the_readings_come_from_meminfo_and_the_reserve_from_the_env(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       127500000 kB\nMemFree:          900000 kB\n"
                       "MemAvailable:   60000000 kB\nBuffers:               0 kB\n")
    assert llm_context.mem_total_bytes(str(meminfo)) == 127_500_000 * 1024
    assert llm_context.mem_available_bytes(str(meminfo)) == 60_000_000 * 1024
    assert llm_context.mem_available_bytes(str(tmp_path / "missing")) is None
    monkeypatch.setenv("QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB", "32768")
    assert llm_context.reserve_bytes() == 32768 * MiB
    monkeypatch.setenv("QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB", "not-a-number")
    assert llm_context.reserve_bytes() == llm_context.DEFAULT_RESERVE_MB * MiB


def test_a_profile_with_no_cache_measurement_cannot_count_requests():
    """kv_tokens 0 (a start that printed a zero-token cache) must not divide the plan
    by zero or promise infinite concurrency: it is no profile at all."""
    broken = llm_context.VllmProfile.from_record({**BOX_PROFILE, "kv_tokens": 0, "kv_bytes": 0})
    assert broken is not None and broken.per_token_bytes == 0.0
    plan = _plan(profile=broken, available_bytes=60 * GiB)
    assert plan.max_num_seqs is None
    assert plan.utilization == pytest.approx(0.42, abs=0.005), "today's rule, pool only"


def test_the_operators_ceiling_is_respected_even_when_it_is_the_default():
    """No --gpu-memory-utilization on the command line = vLLM's own 0.90."""
    plan = _plan(operator_utilization=None, window=32_768)
    assert plan.utilization <= llm_context.DEFAULT_VLLM_UTILIZATION


def test_a_zero_or_missing_max_num_seqs_plans_at_least_one_request():
    for seqs in (0, None, -3):
        plan = _plan(operator_max_seqs=seqs, window=32_768)
        assert plan.max_num_seqs == 1


# ── 2. the vLLM server gets the concurrency that fits ────────────────────────


def test_concurrency_is_what_fits_and_is_passed_as_max_num_seqs():
    """8 requests of 262k cost 8 x 16.2 GB; 90.75 GB usable holds 4 beside the
    weights → --max-num-seqs 4, pool sized for exactly those 4."""
    plan = _plan()
    assert plan.max_num_seqs == 4
    assert plan.utilization == pytest.approx(0.73, abs=0.011)
    plenty = _plan(window=32_768)
    assert plenty.max_num_seqs == 8, "all eight fit at a 32k window"


def test_the_live_box_advertises_that_concurrency_not_the_operators_par(monkeypatch):
    """Whatever --max-num-seqs says, the cache holds what it holds: 290,064 tokens at a
    262k window is ONE request, and that is what the heartbeat and the claim use."""
    w = _gpu_worker(monkeypatch, served=("vllm",))
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: _container(running=True))
    monkeypatch.setattr(llm_probe, "vllm_kv_capacity_tokens", lambda url, **k: 290_064)
    w._observe_llm_context()
    assert w._pool_parallelism() == 1
    assert w._gpu_slot_budget() == 1
    assert w._kind_caps()["vllm"] == 1


# ── 3. or ollama runs instead ─────────────────────────────────────────────────


def test_one_request_worth_of_memory_runs_ollama_instead(monkeypatch):
    """52 GB free → 44 GB usable → one 262k request (36.7 GB) fits, a second does not.
    vLLM at one request buys nothing ollama does not give; the start is refused and
    the box keeps serving ollama."""
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", BOX_PROFILE)
    w = _gpu_worker(monkeypatch, available_gib=52)
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: True)
    plan = w._vllm_start_plan()
    assert plan.refusal is not None and "ollama serves this box" in plan.refusal


def test_two_requests_worth_runs_vllm(monkeypatch):
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", BOX_PROFILE)
    w = _gpu_worker(monkeypatch, available_gib=70)
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: True)
    plan = w._vllm_start_plan()
    assert plan.refusal is None and plan.max_num_seqs == 2


def test_a_vllm_only_box_runs_the_single_request_vllm(monkeypatch):
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", BOX_PROFILE)
    w = _gpu_worker(monkeypatch, available_gib=52)
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: False)
    plan = w._vllm_start_plan()
    assert plan.refusal is None and plan.max_num_seqs == 1


# ── 4. a job that does not fit now is marked, and another one is taken ─────


def test_a_job_whose_model_does_not_fit_now_is_skipped_and_the_next_one_taken(monkeypatch):
    """17 GB free, 8 GB reserved: a 16.8 GB model cannot be loaded here, a 0.4 GB one
    can. The claim is asked only for what fits — the big job stays queued for a box
    that can hold it, the small job runs. Then memory returns and the big one is
    offered again."""
    w = _gpu_worker(monkeypatch, served=("ollama",), available_gib=17)
    run_id = make_run()
    big = node_queue.enqueue_node_job(run_id=run_id, node_id="big", node_module="llm.chat",
                                      queue="gpu", required_model="chat-expert-a:latest")
    small = node_queue.enqueue_node_job(run_id=run_id, node_id="small", node_module="llm.chat",
                                        queue="gpu", required_model="tiny:latest")
    w._observe_llm_context()
    offered = w._known_models(["chat-expert-a:latest", "tiny:latest"])
    assert offered == ["tiny:latest"]
    claimed = node_queue.claim_next_gpu_job(0, None, host="box-b-gpu", known_models=offered,
                                            served_types=["ollama"])
    assert claimed and claimed["id"] == small
    assert node_queue.claim_next_gpu_job(0, None, host="box-b-gpu", known_models=offered,
                                         served_types=["ollama"]) is None
    assert node_queue.get_node_job(big)["status"] == "queued"
    node_queue.mark_completed(small, context_delta={}, seconds=1.0)   # the box's one ollama slot frees
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 60 * GiB)
    w._observe_llm_context()
    assert "chat-expert-a:latest" in w._known_models(["chat-expert-a:latest", "tiny:latest"])
    claimed = node_queue.claim_next_gpu_job(0, None, host="box-b-gpu",
                                            known_models=w._known_models(["chat-expert-a:latest"]),
                                            served_types=["ollama"])
    assert claimed and claimed["id"] == big


def test_a_job_no_live_box_can_hold_in_its_free_memory_is_marked_not_processible():
    """The fleet-wide mark (migration 0015): every live gpu box advertises what it can
    load NOW (fits_models, memory-narrowed); a queued job whose model none of them
    lists is flagged, and un-flagged the moment one of them can."""
    run_id = make_run()
    big = node_queue.enqueue_node_job(run_id=run_id, node_id="big", node_module="llm.chat",
                                      queue="gpu", required_model="chat-expert-a:latest")
    node_queue.upsert_worker_heartbeat(host_label="box-b-gpu", queue="gpu", concurrency=1,
                                       known_models=["tiny:latest"], fits_models=["tiny:latest"],
                                       vram_total_mb=98304)
    flagged = node_queue.flag_unassignable_gpu_jobs()
    assert [j["id"] for j in flagged] == [big]
    row = node_queue.get_node_job(big)
    assert row["unassignable_at"] is not None and "chat-expert-a:latest" in str(row["unassignable_reason"])
    node_queue.upsert_worker_heartbeat(host_label="box-b-gpu", queue="gpu", concurrency=1,
                                       known_models=["tiny:latest", "chat-expert-a:latest"],
                                       fits_models=["tiny:latest", "chat-expert-a:latest"],
                                       vram_total_mb=98304)
    assert node_queue.flag_unassignable_gpu_jobs() == []
    assert node_queue.get_node_job(big)["unassignable_at"] is None


def test_the_heartbeat_carries_the_memory_narrowed_list_the_sweep_reads():
    """Without this, the fleet mark would judge by VRAM size only and never by the
    memory actually free: the heartbeat's known/fits lists pass through the worker's
    memory narrowing before they are written."""
    import inspect

    src = inspect.getsource(claim_worker.HeartbeatEmitter.emit_once)
    assert "known = self._known_models_fn(known)" in src
    assert "fits_models=self._fits_models(known, vram_total)" in src


# ── 5. the danger lines the sentinel reads ───────────────────────────────────


def test_the_floors_are_exclusive_at_the_boundary(monkeypatch):
    w = _gpu_worker(monkeypatch)
    floor = claim_worker._danger_floor_bytes()
    kill = memory_sentinel.kill_floor_bytes()
    assert floor > kill > 0
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: floor)
    assert w._box_memory_in_danger() is False, "at the floor is not under it"
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: floor - 1)
    assert w._box_memory_in_danger() is True


def test_a_kill_floor_above_the_danger_floor_still_sheds_at_once(monkeypatch):
    """A misconfiguration (kill 20 GB, danger 12 GB) must not create a band where
    nothing acts: the kill line is simply the stronger one."""
    from queue_workflows.config import get_config

    monkeypatch.setenv(memory_sentinel.KILL_FLOOR_ENV, str(20 * 1024))
    stopped: list = []
    get_config().vllm_stop_fn = lambda: stopped.append(True) or True
    monkeypatch.setattr(llm_probe, "loaded_models_info", lambda url, **k: [])
    try:
        w = _gpu_worker(monkeypatch, available_gib=15)
        monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: _container(running=True))
        assert w._shed_llm_if_box_is_dying() is True
        assert w._shed_llm_if_box_is_dying() is True, "the kill line ignores the cooldown"
        assert stopped == [True, True]
    finally:
        get_config().vllm_stop_fn = None


# ── helpers ───────────────────────────────────────────────────────────────────


def _container(*, running):
    return llm_context_lever.ServerContainer(
        kind="vllm", name="vllm", exists=True, running=running, configured=262144,
        utilization=0.75, utilization_now=0.42, max_num_seqs=8, max_num_seqs_now=8,
        served_name="chat-model-a:latest", fingerprint="fp", container_id="c1",
        started_at="2024-01-01T00:00:00Z")


def _gpu_worker(monkeypatch, *, served=("vllm",), available_gib=110):
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
    monkeypatch.setattr(_factory, "resolve_base_url", lambda kind="ollama": f"http://{kind}:1")
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [])
    monkeypatch.setattr(llm_probe, "ollama_model_sizes", lambda url, **k: {
        "chat-expert-a:latest": 16_800_000_000, "tiny:latest": 400_000_000,
        "chat-model-a:latest": 17_741_872_154})
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
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: available_gib * GiB)
    llm_context_lever.reset_cache_for_tests()
    return w
