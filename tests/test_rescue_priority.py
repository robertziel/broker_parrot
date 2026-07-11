"""Rescued work jumps the queue — ``is_priority`` on machine-loss requeues.

When a machine dies (its lease lapses) or an operator turns it OFF, the engine
already returns that machine's ``running`` node jobs to ``queued``. But it only
bumped the integer band (``LEAST(priority, 10)``), which merely moves the row
*within* the ordering — a peer sitting at band 0 still wins, and on GPU the
warm-model affinity tiebreak can still outrank it.

Rescued work must come back at the FRONT: ``is_priority`` sorts first in the
claim ORDER BY (``node_queue.py`` — ahead of the band AND the GPU warm-model
affinity), so a job that already burned wall-clock on a lost machine is the very
next thing a healthy peer picks up.

Scope: only the machine-loss paths (lease lapse, operator hard-stop). A terminal
parent still cancels rather than re-queues, and must NOT be flagged — flagging a
ghost row would park a priority job nobody can ever claim. ``ingest_jobs`` has no
``is_priority`` column, so its requeue path stays band-only.
"""

from __future__ import annotations

import pytest

import queue_workflows
from queue_workflows import node_queue
from queue_workflows.db import connection
from tests._helpers import force_lease, make_run, set_run_status


@pytest.fixture(autouse=True)
def _register_ingest_tasks():
    queue_workflows.register_ingest_task("run_fetch_all", lambda reason: {"ok": True})
    yield


def _make_run() -> str:
    return make_run(workflow_name="_rescue_priority_test")


def _running_node_job(host: str, *, queue: str = "gpu") -> str:
    """A ``running`` workflow_node_jobs row claimed_by ``host`` on ``queue``."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue=queue,
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET status='running', started_at=now(), claimed_by=%s, "
            "    lease_expires_at = now() + interval '600 seconds' "
            "WHERE id=%s",
            (host, job_id),
        )
    return job_id


# ── lease lapse (the machine died / wedged) ──────────────────────────────────


def test_lease_lapse_rescue_flags_highest_priority():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="rescued", node_module="x", queue="cpu",
        priority=100,
    )
    force_lease(job_id, expires_in_s=-30)

    node_queue.reclaim_expired_leases()

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
    assert row["is_priority"] is True, (
        "a job rescued off a lost machine must come back at the FRONT of the "
        "queue (is_priority), not merely inside band 10"
    )


def test_lease_lapse_rescue_does_not_flag_a_terminal_parent_row():
    """The cancel branch must stay unflagged: the claim SQL filters non-running
    parents, so a flagged ghost would sit at the head of the queue forever."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="ghost", node_module="x", queue="gpu",
        priority=100,
    )
    force_lease(job_id, expires_in_s=-30)
    set_run_status(run_id, "cancelled")

    node_queue.reclaim_expired_leases()

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "cancelled"
    assert row["is_priority"] is False


def test_rescued_job_is_claimed_before_a_better_banded_fresh_job():
    """Behavioural proof of "highest priority": a fresh job in the BEST band
    (0) still loses to a rescued job, because is_priority sorts ahead of the
    band. Without the flag the band-0 job would win."""
    run_id = _make_run()
    fresh = node_queue.enqueue_node_job(
        run_id=run_id, node_id="fresh", node_module="x", queue="cpu",
        priority=0,
    )
    rescued = node_queue.enqueue_node_job(
        run_id=run_id, node_id="rescued", node_module="x", queue="cpu",
        priority=100,
    )
    force_lease(rescued, expires_in_s=-30)
    node_queue.reclaim_expired_leases()

    first = node_queue.claim_next_cpu_job(0)
    assert first["id"] == rescued, (
        "rescued work must outrank even a band-0 fresh job; "
        f"claimed {first['node_id']!r} instead"
    )
    assert node_queue.claim_next_cpu_job(0)["id"] == fresh


def test_gpu_rescue_outranks_warm_model_affinity():
    """On GPU the rescued row must also beat the warm-model tiebreak — the
    machine that died may have held the only warm copy."""
    run_id = _make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="warm", node_module="x", queue="gpu",
        required_model="flux",
    )
    rescued = node_queue.enqueue_node_job(
        run_id=run_id, node_id="rescued", node_module="x", queue="gpu",
        required_model="sdxl",
    )
    force_lease(rescued, expires_in_s=-30)
    node_queue.reclaim_expired_leases()

    # Worker has 'flux' warm — normally it would claim the 'warm' job.
    first = node_queue.claim_next_gpu_job(0, current_model="flux")
    assert first["id"] == rescued


# ── operator hard-stop (the machine was turned OFF) ──────────────────────────


def test_operator_hard_stop_rescue_flags_highest_priority():
    job_id = _running_node_job("host-a", queue="gpu")

    assert node_queue.requeue_running_for_worker("host-a", "gpu") == 1

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
    assert row["is_priority"] is True


def test_hard_stop_rescue_still_does_not_burn_the_retry_cap():
    """Prioritising must not turn a redistribution into a watchdog retry."""
    job_id = _running_node_job("host-a", queue="gpu")
    node_queue.requeue_running_for_worker("host-a", "gpu")
    assert (node_queue.get_node_job(job_id).get("watchdog_retries") or 0) == 0


def test_ingest_hard_stop_rescue_is_band_only():
    """``ingest_jobs`` carries no ``is_priority`` column — its requeue path must
    stay band-only and must not reference the column (else it raises)."""
    job_id = node_queue.enqueue_ingest_job(
        task_name="run_fetch_all", queue="fetch", priority=100,
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE ingest_jobs "
            "SET status='running', started_at=now(), claimed_by='host-a', "
            "    lease_expires_at = now() + interval '600 seconds' "
            "WHERE id=%s",
            (job_id,),
        )

    assert node_queue.requeue_running_for_worker("host-a", "fetch") == 1

    row = node_queue.get_ingest_job(job_id)
    assert row["status"] == "queued"
    assert "is_priority" not in row
