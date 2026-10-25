"""``ingest_store`` — the ingest-job seam the engine's ingest call sites use.

The seam is a transparent delegate to the ``node_queue.*ingest*`` functions
against ``ingest_jobs`` (pg or sqlite), and the live call sites (the NodePool
reclaim sweep) must route through it.
"""

from __future__ import annotations

import pytest

import queue_workflows
from queue_workflows import ingest_store
from queue_workflows.db import connection


def _task(reason, args=None):
    return {"ok": True, "reason": reason}


def _ingest_jobs_count() -> int:
    """Rows in the Postgres ``ingest_jobs`` table (the pg-only engine table)."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM ingest_jobs")
        return int(cur.fetchone()["n"])


# ── pg path: the seam is a transparent delegate to node_queue ───────────────


def test_pg_path_delegates_to_node_queue_end_to_end():
    queue_workflows.register_ingest_task("probe_task", _task)
    from queue_workflows import node_queue

    jid = ingest_store.enqueue_ingest_job(
        task_name="probe_task", queue="fetch", reason="tick", args={"k": 1},
    )
    # Visible through the engine's own ingest_jobs reader (same row, same table).
    row = node_queue.get_ingest_job(jid)
    assert row is not None and row["status"] == "queued" and row["task_name"] == "probe_task"
    assert ingest_store.get_ingest_job(jid)["args"] == {"k": 1}

    claimed = ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=30)
    assert claimed is not None and claimed["id"] == jid and claimed["status"] == "running"
    assert ingest_store.renew_ingest_lease(jid, "w1", lease_s=30) is True

    done = ingest_store.mark_ingest_completed(jid, result={"n": 7}, seconds=0.1)
    assert done is not None and done["status"] == "completed"
    # Idempotent: a second terminal call is a no-op.
    assert ingest_store.mark_ingest_completed(jid, result={"n": 99}) is None
    assert node_queue.get_ingest_job(jid)["result"] == {"n": 7}


def test_pg_snapshot_matches_node_queue():
    queue_workflows.register_ingest_task("probe_task", _task)
    ingest_store.enqueue_ingest_job(task_name="probe_task", queue="fetch")
    from queue_workflows import node_queue

    assert ingest_store.ingest_snapshot() == node_queue.ingest_snapshot()


@pytest.mark.parametrize("backend_env", [None])  # pg path (default)
def test_enqueue_validation_rejects_before_write(backend_env):
    queue_workflows.register_ingest_task("probe_task", _task)
    before = _ingest_jobs_count()
    with pytest.raises(ValueError, match="ingest queue must be in"):
        ingest_store.enqueue_ingest_job(task_name="probe_task", queue="nope")
    with pytest.raises(ValueError, match="registered ingest task"):
        ingest_store.enqueue_ingest_job(task_name="ghost", queue="fetch")
    assert _ingest_jobs_count() == before  # nothing written on a rejected enqueue


def test_node_pool_ingest_reclaim_routes_through_seam(monkeypatch):
    """`NodePool._sweep_expired_ingest_leases` must call the ingest seam, not
    `node_queue` directly — proven with a spy, no live worker needed."""
    from queue_workflows import node_pool as _np

    calls: list[str] = []
    monkeypatch.setattr(
        ingest_store, "reclaim_expired_ingest_leases",
        lambda: calls.append("seam") or [],
    )
    pool = _np.NodePool()
    pool._ingest_reclaim_last_run = 0.0          # bypass the interval gate
    pool._ingest_reclaim_interval_s = 0.0
    pool._sweep_expired_ingest_leases()
    assert calls == ["seam"]
