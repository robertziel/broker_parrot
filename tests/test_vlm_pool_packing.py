"""FILL-BEFORE-SPILL packing for the no-model GPU (VLM) pool lane.

The GPU pool feeder POSTs no-model VLM jobs (``required_model IS NULL``) to a
per-host vLLM server, up to PAR (= ``worker_controls.llm_parallelism``) in
flight. Previously every vLLM machine ran an independent
``FOR UPDATE SKIP LOCKED`` feeder, so VLM jobs SPREAD across machines. The new
gate :func:`node_queue.vlm_pool_should_defer` bin-packs instead: a machine
defers a no-model claim this cycle IFF a FRESH gpu peer ranked strictly above it
— ``(concurrency DESC, host_label ASC)`` — still has free VLM capacity. So VLM
work fills the highest-ranked box first and spills only when it is full.

Covered:
- defer=True: a fresh higher-ranked peer (bigger PAR; or equal PAR + earlier
  host_label) with running-no-model count < its concurrency → M defers.
- defer=False: every higher-ranked peer is full → M claims.
- top-ranked machine (max PAR, earliest host_label) → never defers.
- a STALE higher-ranked peer is ignored → M claims.
- single machine (only M heartbeating) → never defers (byte-compat default).
- the INLINE diffusion lane (``require_model=True``) is unaffected by the gate.
- feeder integration: the pool feeder honours a True verdict (does not claim).
"""

from __future__ import annotations

import threading
import time

from queue_workflows import claim_worker, node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


# ── seeding helpers (domain-free) ────────────────────────────────────────────


def _beat(host: str, concurrency: int, *, age_s: float = 0.0,
          queue: str = "gpu") -> None:
    """Upsert a ``worker_heartbeats`` row for ``host`` advertising
    ``concurrency`` (= PAR). ``age_s`` > 0 pushes ``last_seen`` that many
    seconds into the past so the row reads STALE to a freshness window < age_s
    (``upsert_worker_heartbeat`` always writes ``last_seen = now()``)."""
    node_queue.upsert_worker_heartbeat(
        host_label=host, queue=queue, concurrency=concurrency,
    )
    if age_s > 0:
        with connection() as c, c.cursor() as cur:
            cur.execute(
                "UPDATE worker_heartbeats "
                "SET last_seen = now() - make_interval(secs => %s) "
                "WHERE host_label = %s AND queue = %s",
                (float(age_s), host, queue),
            )


def _running_no_model_gpu_jobs(host: str, n: int) -> list[str]:
    """Create ``n`` RUNNING no-model gpu jobs owned by ``host`` via the real
    claim path (enqueue queued no-model gpu row → claim with
    ``require_model=False`` stamps ``running`` + ``claimed_by=host``)."""
    ids: list[str] = []
    for i in range(n):
        run_id = make_run()
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"vlm-{host}-{i}", node_module="x",
            queue="gpu",
        )
        job = node_queue.claim_next_gpu_job(
            0, None, host=host, require_model=False,
        )
        assert job is not None and job["claimed_by"] == host
        assert job["required_model"] is None
        ids.append(job["id"])
    return ids


# ── vlm_pool_should_defer — the gate ─────────────────────────────────────────


def test_defer_true_when_higher_par_peer_has_free_capacity():
    """A fresh peer with a BIGGER PAR and free VLM capacity ranks above M → M
    (the lower-PAR box) defers this cycle."""
    _beat("big", concurrency=4)   # ranks above (PAR 4 > 2)
    _beat("small", concurrency=2)
    # big runs 1 of its 4 → free capacity remains.
    _running_no_model_gpu_jobs("big", 1)
    assert node_queue.vlm_pool_should_defer("small", 2) is True


def test_defer_true_when_equal_par_earlier_host_has_free_capacity():
    """Equal PAR is broken by ``host_label ASC`` — an earlier-named peer with
    free capacity ranks above M → M (later name) defers."""
    _beat("aaa", concurrency=2)   # equal PAR, earlier host → ranks above "bbb"
    _beat("bbb", concurrency=2)
    # aaa idle (0 running) → free capacity.
    assert node_queue.vlm_pool_should_defer("bbb", 2) is True


def test_defer_false_when_every_higher_peer_is_full():
    """When every higher-ranked peer is at capacity (running-no-model count >=
    its concurrency), nothing ranks above with room → M claims (no defer)."""
    _beat("big", concurrency=2)   # higher PAR than M(1)
    _beat("mid", concurrency=1)   # M
    # big is FULL (2 of 2 running no-model).
    _running_no_model_gpu_jobs("big", 2)
    assert node_queue.vlm_pool_should_defer("mid", 1) is False


def test_defer_false_equal_par_earlier_host_full():
    """Equal-PAR earlier-host peer that is FULL does not block M → M claims."""
    _beat("aaa", concurrency=2)   # equal PAR, earlier host, but FULL
    _beat("bbb", concurrency=2)
    _running_no_model_gpu_jobs("aaa", 2)   # 2 of 2 → full
    assert node_queue.vlm_pool_should_defer("bbb", 2) is False


def test_top_ranked_machine_never_defers():
    """The max-PAR / earliest-host machine has no peer ranked above it → it
    never defers, even with idle lower-ranked peers present (fills first)."""
    _beat("top", concurrency=8)
    _beat("mid", concurrency=4)   # idle, but ranks BELOW top
    _beat("low", concurrency=1)
    assert node_queue.vlm_pool_should_defer("top", 8) is False


def test_top_ranked_by_host_label_tiebreak_never_defers():
    """Among equal-PAR machines the earliest host_label is top-ranked → never
    defers (no strictly-higher peer exists; an equal-PAR equal-or-later host is
    NOT 'above')."""
    _beat("aaa", concurrency=4)   # earliest of the PAR-4 cohort
    _beat("zzz", concurrency=4)   # idle equal-PAR peer, but later name
    assert node_queue.vlm_pool_should_defer("aaa", 4) is False


def test_stale_higher_ranked_peer_is_ignored():
    """A higher-ranked peer whose heartbeat is OLDER than the freshness window
    does NOT count as 'above' — a dead top box must not block everyone. With
    only a stale higher peer present, M claims."""
    _beat("deadbig", concurrency=4, age_s=120)   # stale (> 30 s window), idle
    _beat("small", concurrency=2)
    assert node_queue.vlm_pool_should_defer("small", 2) is False
    # And with an explicit short window the same row is also ignored.
    assert node_queue.vlm_pool_should_defer("small", 2, stale_s=30) is False


def test_fresh_within_custom_window_still_defers():
    """Freshness honours the ``stale_s`` argument: a peer aged 10 s is fresh to
    the default 30 s window, so M still defers."""
    _beat("big", concurrency=4, age_s=10)   # fresh within 30 s
    _beat("small", concurrency=2)
    assert node_queue.vlm_pool_should_defer("small", 2) is True
    # Narrow the window below the peer's age → it goes stale → no defer.
    assert node_queue.vlm_pool_should_defer("small", 2, stale_s=5) is False


def test_single_machine_never_defers():
    """Only M heartbeating (no peer at all) → never defers. SAFE default that
    keeps single-box fleets + other consumers byte-identical to today."""
    _beat("solo", concurrency=4)
    assert node_queue.vlm_pool_should_defer("solo", 4) is False


def test_no_heartbeats_at_all_never_defers():
    """No ``worker_heartbeats`` rows whatsoever (a cold fleet) → never defers."""
    assert node_queue.vlm_pool_should_defer("ghost", 1) is False


def test_capacity_count_ignores_model_backed_running_jobs():
    """Free-capacity is measured over NO-MODEL gpu jobs only. A higher peer busy
    with a MODEL-BACKED (inline diffusion) job still counts as having free VLM
    capacity → M defers (the diffusion job is not a VLM slot)."""
    _beat("big", concurrency=2)
    _beat("small", concurrency=1)
    # big is running a model-backed diffusion job — NOT a no-model VLM job.
    run_id = make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="diff", node_module="x", queue="gpu",
        required_model="sdxl",
    )
    job = node_queue.claim_next_gpu_job(
        0, "sdxl", host="big", require_model=True,
    )
    assert job is not None and job["required_model"] == "sdxl"
    # big's no-model count is 0 < 2 → free VLM capacity → M defers.
    assert node_queue.vlm_pool_should_defer("small", 1) is True


def test_capacity_count_ignores_cpu_and_other_queues():
    """Capacity counts ``queue='gpu'`` only — a running CPU job owned by the
    higher peer doesn't consume its VLM capacity."""
    _beat("big", concurrency=1)
    _beat("small", concurrency=1, queue="gpu")
    # Make small rank below big via host_label (equal PAR): "big" < "small".
    run_id = make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="cpu", node_module="x", queue="cpu",
    )
    cpu_job = node_queue.claim_next_cpu_job(0, host="big")
    assert cpu_job is not None
    # big's no-model GPU count is still 0 < 1 → free → small defers.
    assert node_queue.vlm_pool_should_defer("small", 1) is True


# ── inline diffusion lane is unaffected ───────────────────────────────────────


class _Cache:
    current_model = None


def test_inline_lane_claim_ignores_defer_gate():
    """The INLINE diffusion lane (``_claim`` → ``require_model=True``) must NOT
    consult the defer gate: even with a fresh higher-ranked idle peer that WOULD
    make the pool lane defer, a model-backed claim proceeds normally."""
    # A higher-ranked idle peer that would force a POOL deferral for "small".
    _beat("big", concurrency=8)
    _beat("small", concurrency=2)
    # Sanity: the pool gate would defer for this machine.
    assert node_queue.vlm_pool_should_defer("small", 2) is True

    # But the inline lane claims a model-backed job regardless.
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="d", node_module="x", queue="gpu",
        required_model="sdxl",
    )
    w = claim_worker.ClaimWorker(queue="gpu", host="small", model_cache=_Cache())
    claimed = w._claim()
    assert claimed is not None and claimed["id"] == job_id
    assert claimed["status"] == "running"
    assert claimed["claimed_by"] == "small"


def test_inline_lane_does_not_call_defer_gate(monkeypatch):
    """Belt-and-braces: ``_claim`` (inline) never invokes
    ``vlm_pool_should_defer`` — the gate is pool-lane-only."""
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return False

    monkeypatch.setattr(node_queue, "vlm_pool_should_defer", spy)
    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_Cache())
    w._claim()   # inline lane claim (require_model=True), empty queue
    assert called["n"] == 0


# ── feeder integration: the pool feeder honours a True verdict ────────────────


def test_pool_feeder_defers_when_gate_true(monkeypatch):
    """When :func:`vlm_pool_should_defer` returns True, the feeder does NOT claim
    this cycle — a queued no-model job is left ``queued`` (no claim-then-release):
    the slot stays free and ``_pool_inflight`` is untouched."""
    from queue_workflows import worker_control
    monkeypatch.setattr(
        worker_control, "llm_config_for",
        lambda h, q: worker_control.LLMConfig(parallelism=2),
    )
    # Force the gate ON for this feeder's host.
    monkeypatch.setattr(node_queue, "vlm_pool_should_defer", lambda h, p: True)

    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="v", node_module="x", queue="gpu",
    )

    w = claim_worker.ClaimWorker(queue="gpu", host="small", model_cache=_Cache())
    w._start_pool_lane()
    try:
        # Give the feeder several cycles; it must keep deferring (never claim).
        deadline = time.time() + 1.5
        while time.time() < deadline:
            assert node_queue.get_node_job(job_id)["status"] == "queued", (
                "feeder claimed despite a True defer verdict"
            )
            assert w._pool_inflight == 0, "defer must not reserve an in-flight slot"
            time.sleep(0.05)
    finally:
        w.stop()
        w._stop_pool_lane()
    # Still queued at the end.
    assert node_queue.get_node_job(job_id)["status"] == "queued"


def test_pool_feeder_claims_when_gate_false(monkeypatch):
    """When the gate returns False, the feeder claims as before — the no-model
    job is taken (running/completed)."""
    from queue_workflows import worker_control
    import sys
    import types
    monkeypatch.setattr(
        worker_control, "llm_config_for",
        lambda h, q: worker_control.LLMConfig(parallelism=2),
    )
    monkeypatch.setattr(node_queue, "vlm_pool_should_defer", lambda h, p: False)

    # A fast no-op node so the claimed job runs to completion.
    import queue_workflows
    queue_workflows.set_node_module_package("qwf_pack_nodes")
    mod = types.ModuleType("qwf_pack_nodes.vlm_ok")
    mod.run = lambda *, out=None, inputs=None, cancel_event=None: {
        "context_delta": {}
    }
    sys.modules["qwf_pack_nodes.vlm_ok"] = mod

    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="v", node_module="vlm_ok", queue="gpu",
    )

    w = claim_worker.ClaimWorker(queue="gpu", host="solo", model_cache=_Cache())
    w._start_pool_lane()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if node_queue.get_node_job(job_id)["status"] in ("running", "completed"):
                break
            time.sleep(0.02)
        assert node_queue.get_node_job(job_id)["status"] in ("running", "completed"), (
            "feeder must claim when the defer gate is False"
        )
    finally:
        w.stop()
        w._stop_pool_lane()


def test_pool_feeder_uses_live_par_in_defer_check(monkeypatch):
    """The feeder passes the LIVE PAR (``_pool_parallelism()``) into the gate, so
    the defer decision reflects the operator's current ``--max-num-seqs``."""
    from queue_workflows import worker_control
    monkeypatch.setattr(
        worker_control, "llm_config_for",
        lambda h, q: worker_control.LLMConfig(parallelism=5),
    )
    seen: list[tuple[str, int]] = []
    stop = threading.Event()

    def spy(host, par):
        seen.append((host, par))
        stop.set()
        return True   # defer so nothing is claimed (keeps the test cheap)

    monkeypatch.setattr(node_queue, "vlm_pool_should_defer", spy)

    run_id = make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="v", node_module="x", queue="gpu",
    )

    w = claim_worker.ClaimWorker(queue="gpu", host="host-z", model_cache=_Cache())
    w._start_pool_lane()
    try:
        assert stop.wait(timeout=3.0), "feeder never invoked the defer gate"
    finally:
        w.stop()
        w._stop_pool_lane()
    assert ("host-z", 5) in seen, f"feeder did not pass live (host, PAR); saw {seen}"
