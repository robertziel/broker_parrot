"""Ingest-job store — the one module the engine's ingest call sites (scheduler,
claim worker lease renewer + watchdog fail path, the NodePool reclaim sweep)
go through for the *flat* ingest-family queue (``ingest_jobs``).

WHY this module exists
----------------------
It is the single seam between the ingest call sites and the relational store
(``pg`` or ``sqlite``; sqlite speaks the same SQL via the dialect seam). Every
function delegates to the ``node_queue.*ingest*`` functions against
``ingest_jobs``, so the call sites depend on one small surface instead of
reaching into ``node_queue`` piecemeal, and a test can spy the seam (e.g. the
reclaim sweep) without a live worker.
"""

from __future__ import annotations

from typing import Any

from queue_workflows.node_queue import DEFAULT_LEASE_S


# ── public API (mirrors node_queue's ingest surface) ────────────────────────


def enqueue_ingest_job(
    *, task_name: str, queue: str, reason: str = "tick", priority: int = 100,
    args: dict[str, Any] | None = None, conn: Any = None,
    project: str | None = None,
) -> str:
    """Insert a fresh ``queued`` ingest job; return its id.

    ``conn`` (a host connection for a caller-controlled transaction) makes the
    NOTIFY ride the caller's transaction. ``project`` (migration 0017) is the
    tenant tag (``None`` ⇒ ``config.project``; see docs/broker.md)."""
    from queue_workflows import node_queue

    return node_queue.enqueue_ingest_job(
        task_name=task_name, queue=queue, reason=reason,
        priority=priority, args=args, conn=conn, project=project,
    )


def get_ingest_job(job_id: str) -> dict[str, Any] | None:
    from queue_workflows import node_queue

    return node_queue.get_ingest_job(job_id)


def claim_next_ingest_job(
    queue: str, *, host: str | None = None, lease_s: int = DEFAULT_LEASE_S,
) -> dict[str, Any] | None:
    """Atomically claim the next queued ingest job on ``queue`` (exactly-once
    under contention), or ``None``."""
    from queue_workflows import node_queue

    return node_queue.claim_next_ingest_job(queue, host=host, lease_s=lease_s)


def renew_ingest_lease(
    job_id: str, host: str, *, lease_s: int = DEFAULT_LEASE_S,
) -> bool:
    """Extend the lease iff the row is still ``running`` and owned by ``host``.
    Returns whether it renewed (the LeaseRenewer's heartbeat)."""
    from queue_workflows.db import connection

    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE ingest_jobs "
            "SET lease_expires_at = now() + make_interval(secs => %s) "
            "WHERE id = %s AND status = 'running' AND claimed_by = %s "
            "RETURNING id",
            (int(lease_s), job_id, host),
        )
        return cur.fetchone() is not None


def mark_ingest_completed(
    job_id: str, *, result: dict[str, Any] | None = None,
    seconds: float | None = None,
) -> dict[str, Any] | None:
    """Transition to ``completed`` (idempotent — ``None`` if already terminal)."""
    from queue_workflows import node_queue

    return node_queue.mark_ingest_completed(
        job_id, result=result, seconds=seconds,
    )


def mark_ingest_failed(
    job_id: str, *, error: str, seconds: float | None = None,
) -> dict[str, Any] | None:
    """Transition to ``failed`` (idempotent — ``None`` if already terminal)."""
    from queue_workflows import node_queue

    return node_queue.mark_ingest_failed(job_id, error=error, seconds=seconds)


def reclaim_expired_ingest_leases() -> list[dict[str, Any]]:
    """Re-queue ``running`` ingest jobs whose lease lapsed — the sole recovery
    path for an orphaned ingest row. Returns ``{id, task_name, queue}`` per row."""
    from queue_workflows import node_queue

    return node_queue.reclaim_expired_ingest_leases()


def ingest_snapshot() -> dict[str, Any]:
    """Per-queue depth + live-worker counts (the dashboard's queue indicator)."""
    from queue_workflows import node_queue

    return node_queue.ingest_snapshot()


__all__ = [
    "enqueue_ingest_job",
    "get_ingest_job",
    "claim_next_ingest_job",
    "renew_ingest_lease",
    "mark_ingest_completed",
    "mark_ingest_failed",
    "reclaim_expired_ingest_leases",
    "ingest_snapshot",
]
