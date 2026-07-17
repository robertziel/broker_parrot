"""ONE centralized per-box slot arbiter — enforced at the CLAIM, in the DB.

THE DUPLICATE-JOBS FAILURE THIS ENDS. A GPU box's concurrency was guarded by
in-process counters spread across two lanes (`_pool_inflight`, `_inline_running`)
— independent callbacks that must all agree. They didn't: every box in the fleet
was observed running 2 concurrent GPU jobs against an advertised capacity of 1
(panel `NODE JOBS ⚠ 2/1`). Counter drift, a thread dying without finalizing its
row, or a second worker process all slip past process-local accounting, because
the accounting isn't where the truth is.

THE ARBITER. `claim_next_gpu_job(..., max_running=N)` refuses the claim inside
the claim statement itself when the box already has ≥ N `running` rows
(`claimed_by = host`, same queue + project). The DB count is the single source of
truth every claim path funnels through — inline lane, pool lane, any future lane,
even a second process on the box — and a zombie row keeps its slot occupied until
the lease reclaim frees it, so a box can never stack live work on top of a corpse.
On Postgres the claim serializes per box via an advisory xact lock so two
concurrent claimants can't both pass the count; SQLite's single-writer already
serializes. `max_running=None` keeps the old unbounded behavior (byte-compat).
"""

from __future__ import annotations

from queue_workflows import node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


def _seed(run_id, n=3):
    return [
        node_queue.enqueue_node_job(run_id=run_id, node_id=f"n{i}", node_module="x",
                                    queue="gpu", priority=100)
        for i in range(n)
    ]


def test_full_box_refuses_to_claim_a_second_job():
    run = make_run()
    _seed(run, 2)
    first = node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=1)
    assert first is not None
    # Box at capacity: a queued job exists, but THIS box must not take it.
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=1) is None


def test_slot_frees_on_terminal_and_claiming_resumes():
    run = make_run()
    _seed(run, 2)
    first = node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=1)
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=1) is None
    node_queue.mark_completed(first["id"], context_delta={}, seconds=1.0)
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=1) is not None


def test_max_running_two_admits_two_then_blocks_third():
    run = make_run()
    _seed(run, 3)
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=2) is not None
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=2) is not None
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=2) is None


def test_other_boxes_are_not_blocked_by_a_full_peer():
    run = make_run()
    _seed(run, 2)
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=1) is not None
    # box-a is full; box-b claims the spilled job — the whole point of refusing.
    assert node_queue.claim_next_gpu_job(0, host="box-b-gpu", max_running=1) is not None


def test_zombie_row_holds_the_slot_until_reclaimed():
    """A thread that dies without finalizing leaves its row `running`. The gate
    counts it — the box must NOT stack a live job on the corpse; the lease
    reclaim frees the slot, and only then does claiming resume."""
    run = make_run()
    _seed(run, 2)
    zombie = node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=1)
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=1) is None
    from queue_workflows.dialect import get_dialect
    with connection() as conn, conn.cursor() as cur:   # lease lapses (dead renewer)
        cur.execute(
            "UPDATE workflow_node_jobs SET lease_expires_at = "
            + get_dialect().past_seconds("%(s)s") + " WHERE id = %(id)s",
            {"s": 60, "id": zombie["id"]},
        )
    reclaimed = node_queue.reclaim_expired_leases()
    assert any(r["id"] == zombie["id"] for r in reclaimed)
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=1) is not None


def test_none_keeps_unbounded_legacy_behavior():
    run = make_run()
    _seed(run, 2)
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu") is not None
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu") is not None


def test_running_rows_of_other_queues_do_not_count():
    run = make_run()
    node_queue.enqueue_node_job(run_id=run, node_id="c", node_module="x",
                                queue="cpu", priority=100)
    _seed(run, 1)
    assert node_queue.claim_next_cpu_job(0, host="box-a-gpu") is not None
    # The cpu job running under the same label must not consume the gpu slot.
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", max_running=1) is not None


# ── the worker funnels BOTH lanes through the arbiter ─────────────────────────


def _worker(monkeypatch, par):
    from queue_workflows import claim_worker

    class _Cache:
        current_model = None

    w = claim_worker.ClaimWorker(queue="gpu", host="box-a-gpu", model_cache=_Cache())
    monkeypatch.setattr(w, "_pool_parallelism", lambda: par)
    return w


def test_inline_claim_passes_the_slot_budget(monkeypatch):
    from queue_workflows import claim_worker
    w = _worker(monkeypatch, par=1)
    seen = {}
    monkeypatch.setattr(claim_worker.node_queue, "claim_next_gpu_job",
                        lambda *a, **k: seen.update(k) or None)
    w._claim()
    assert seen["max_running"] == 1


def test_pool_claim_passes_the_slot_budget(monkeypatch):
    from queue_workflows import claim_worker, model_registry
    w = _worker(monkeypatch, par=4)
    monkeypatch.setattr(model_registry, "known_ids", lambda: [])
    seen = {}
    monkeypatch.setattr(claim_worker.node_queue, "claim_next_gpu_job",
                        lambda *a, **k: seen.update(k) or None)
    w._claim_pool()
    assert seen["max_running"] == 4
