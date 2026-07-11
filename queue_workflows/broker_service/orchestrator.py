"""Broker-side orchestration — the pull→grant permission inversion.

In the legacy engine a worker is autonomous: it *claims* a row for itself
(``FOR UPDATE SKIP LOCKED``) and self-manages its lifecycle. The v2 broker
inverts control: the worker **asks** and the **broker decides**. The broker owns
one shared CPU/GPU queue across every project (rows labelled by ``project``) and:

* **grants** a waiting worker the next job for its project/resource — but only if
  a cross-project **capacity** gate leaves a slot free (this is the broker
  arbitrating a shared core / GPU between competing projects);
* can **revoke** (kill) a job at any time — withdrawing the worker's permission,
  which the client-side gate observes and stops on;
* **health-checks** workers and, when one goes silent (or its grant lapses),
  marks it dead and **re-queues** its job so the next worker can be granted it.

Every terminal carries the engine's idempotency guard
(``WHERE status NOT IN ('done','failed','killed')``) so a duplicate/racey call
can't clobber a finalised row — the same contract the legacy engine preserves.

Dialect-portable: SQL is written in pyformat with :mod:`queue_workflows.dialect`
fragments (``now``, ``future_seconds``, ``past_seconds``, ``creation_order``,
``skip_locked``, ``qualify_returning``), so it runs unchanged on Postgres and on
the SQLite store (the sqlite connection translates paramstyle at execute time).
"""

from __future__ import annotations

import json as _json
import uuid
from typing import Any

from queue_workflows import db
from queue_workflows.broker_service.schema import RESOURCES
from queue_workflows.dialect import get_dialect

#: Columns returned to callers as a job dict (kept in sync with :func:`_row_to_job`).
_JOB_COLS = (
    "job_id", "project", "resource", "status", "priority",
    "granted_worker", "grant_expires_at", "payload", "result", "error",
)

#: Statuses that are terminal for a job (guard the terminal transitions).
_TERMINAL = ("done", "failed", "killed")


def _row_to_job(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalise a DB row into a job dict, decoding the JSON ``payload``/``result``
    text columns back into python objects (``NULL`` stays ``None``)."""
    if row is None:
        return None
    job = dict(row)
    for key in ("payload", "result"):
        val = job.get(key)
        job[key] = _json.loads(val) if isinstance(val, str) else val
    return job


# ── worker registry + liveness ───────────────────────────────────────────────


def register_worker(worker_id: str, *, project: str, resource: str) -> None:
    """Register (or refresh) a worker as ``waiting`` for a grant. Idempotent: a
    re-register refreshes ``last_seen`` + (project, resource) and revives a worker
    the health sweep had marked ``dead`` (it came back)."""
    if resource not in RESOURCES:
        raise ValueError(f"resource must be one of {sorted(RESOURCES)}, got {resource!r}")
    d = get_dialect()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO bw_workers (worker_id, project, resource, state, last_seen, registered_at) "
            f"VALUES (%(w)s, %(p)s, %(r)s, 'waiting', {d.now}, {d.now}) "
            "ON CONFLICT (worker_id) DO UPDATE SET "
            "project = excluded.project, resource = excluded.resource, "
            f"last_seen = {d.now}, "
            "state = CASE WHEN state = 'dead' THEN 'waiting' ELSE state END",
            {"w": worker_id, "p": project, "r": resource},
        )
        conn.commit()


def worker_heartbeat(worker_id: str) -> bool:
    """Refresh a worker's ``last_seen`` (the liveness signal the health sweep
    reads). Returns False if the worker isn't registered."""
    d = get_dialect()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE bw_workers SET last_seen = {d.now} WHERE worker_id = %(w)s RETURNING worker_id",
            {"w": worker_id},
        )
        row = cur.fetchone()
        conn.commit()
        return row is not None


def get_worker(worker_id: str) -> dict[str, Any] | None:
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT worker_id, project, resource, state, last_seen "
            "FROM bw_workers WHERE worker_id = %(w)s",
            {"w": worker_id},
        )
        return cur.fetchone()


# ── the shared queue ─────────────────────────────────────────────────────────


def submit_job(
    *, project: str, resource: str, priority: int = 100, payload: Any = None
) -> str:
    """Enqueue a job onto the shared ``resource`` queue under ``project``. Returns
    the generated job id. Lower ``priority`` = sooner (bands, like the engine)."""
    if resource not in RESOURCES:
        raise ValueError(f"resource must be one of {sorted(RESOURCES)}, got {resource!r}")
    d = get_dialect()
    job_id = uuid.uuid4().hex
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO bw_jobs (job_id, project, resource, status, priority, payload, created_at, updated_at) "
            f"VALUES (%(id)s, %(p)s, %(r)s, 'queued', %(pri)s, %(payload)s, {d.now}, {d.now})",
            {
                "id": job_id, "p": project, "r": resource, "pri": int(priority),
                "payload": _json.dumps(payload) if payload is not None else None,
            },
        )
        conn.commit()
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_JOB_COLS)} FROM bw_jobs WHERE job_id = %(id)s",
            {"id": job_id},
        )
        return _row_to_job(cur.fetchone())


# ── the grant decision (the inversion) ───────────────────────────────────────


def grant_next(
    worker_id: str, *, lease_s: float, capacity: int | None = None
) -> dict[str, Any] | None:
    """The broker decides whether — and what — a waiting worker may run.

    Atomically: (1) look up the worker's (project, resource); (2) if ``capacity``
    is set, DENY when the number of ``granted``/``running`` jobs on that resource
    **across all projects** already meets it (no free core/GPU — the cross-project
    arbitration point); (3) otherwise claim the next ``queued`` job for the
    worker's own project+resource (priority then FIFO), stamp the grant + a
    ``lease_s`` expiry, and flip the worker to ``running``. Returns the granted
    job dict, or ``None`` when denied / no eligible work.
    """
    d = get_dialect()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT worker_id, project, resource FROM bw_workers WHERE worker_id = %(w)s",
            {"w": worker_id},
        )
        worker = cur.fetchone()
        if worker is None:
            return None

        if capacity is not None:
            cur.execute(
                "SELECT COUNT(*) AS n FROM bw_jobs "
                "WHERE resource = %(r)s AND status IN ('granted', 'running')",
                {"r": worker["resource"]},
            )
            if cur.fetchone()["n"] >= capacity:
                return None  # broker denies — the shared resource is full

        cur.execute(
            "UPDATE bw_jobs SET status = 'granted', granted_worker = %(w)s, "
            f"grant_expires_at = {d.future_seconds('%(lease)s')}, updated_at = {d.now} "
            "WHERE job_id = (SELECT job_id FROM bw_jobs "
            "WHERE status = 'queued' AND resource = %(r)s AND project = %(p)s "
            f"ORDER BY priority ASC, {d.creation_order('bw_jobs')} ASC "
            f"LIMIT 1 {d.skip_locked}) "
            f"RETURNING {d.qualify_returning('bw_jobs', _JOB_COLS)}",
            {"w": worker_id, "lease": lease_s, "r": worker["resource"], "p": worker["project"]},
        )
        row = cur.fetchone()
        if row is None:
            return None
        cur.execute(
            f"UPDATE bw_workers SET state = 'running', last_seen = {d.now} WHERE worker_id = %(w)s",
            {"w": worker_id},
        )
        conn.commit()
        return _row_to_job(row)


def has_permission(job_id: str, worker_id: str) -> bool:
    """True iff ``worker_id`` currently holds a LIVE grant for ``job_id`` — i.e.
    the broker still permits it to run. Goes False the instant the broker revokes
    (kill), reassigns, or the grant lease lapses; the client gate polls this to
    know it must stop."""
    d = get_dialect()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 AS ok FROM bw_jobs "
            "WHERE job_id = %(id)s AND granted_worker = %(w)s "
            f"AND status IN ('granted', 'running') AND grant_expires_at > {d.now}",
            {"id": job_id, "w": worker_id},
        )
        return cur.fetchone() is not None


def begin_job(job_id: str, worker_id: str) -> dict[str, Any] | None:
    """The permitted worker confirms it started: ``granted → running``. Returns
    ``None`` if the grant isn't the worker's / was already revoked."""
    d = get_dialect()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE bw_jobs SET status = 'running', updated_at = {d.now} "
            "WHERE job_id = %(id)s AND granted_worker = %(w)s AND status = 'granted' "
            f"RETURNING {d.qualify_returning('bw_jobs', _JOB_COLS)}",
            {"id": job_id, "w": worker_id},
        )
        row = cur.fetchone()
        conn.commit()
        return _row_to_job(row)


def renew_grant(job_id: str, worker_id: str, *, lease_s: float) -> bool:
    """Extend the grant lease while the worker keeps running (its liveness token).
    Also refreshes the worker heartbeat. False if the worker no longer holds it."""
    d = get_dialect()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE bw_jobs SET grant_expires_at = {d.future_seconds('%(lease)s')}, updated_at = {d.now} "
            "WHERE job_id = %(id)s AND granted_worker = %(w)s AND status IN ('granted', 'running') "
            "RETURNING job_id",
            {"id": job_id, "w": worker_id, "lease": lease_s},
        )
        ok = cur.fetchone() is not None
        cur.execute(
            f"UPDATE bw_workers SET last_seen = {d.now} WHERE worker_id = %(w)s",
            {"w": worker_id},
        )
        conn.commit()
        return ok


# ── terminals (idempotent) + broker kill ─────────────────────────────────────


def _terminal(job_id: str, worker_id: str, *, status: str, result: Any, error: str | None):
    d = get_dialect()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE bw_jobs SET status = %(st)s, result = %(res)s, error = %(err)s, "
            f"granted_worker = NULL, updated_at = {d.now} "
            "WHERE job_id = %(id)s AND granted_worker = %(w)s "
            f"AND status NOT IN {_TERMINAL} "
            f"RETURNING {d.qualify_returning('bw_jobs', _JOB_COLS)}",
            {
                "st": status,
                "res": _json.dumps(result) if result is not None else None,
                "err": error, "id": job_id, "w": worker_id,
            },
        )
        row = cur.fetchone()
        if row is not None:  # free the worker to await its next grant
            cur.execute(
                "UPDATE bw_workers SET state = 'waiting' WHERE worker_id = %(w)s",
                {"w": worker_id},
            )
        conn.commit()
        return _row_to_job(row)


def complete_job(job_id: str, worker_id: str, *, result: Any = None) -> dict[str, Any] | None:
    """``running → done`` (idempotent). Returns ``None`` if already terminal /
    not the worker's grant."""
    return _terminal(job_id, worker_id, status="done", result=result, error=None)


def fail_job(job_id: str, worker_id: str, *, error: str | None = None) -> dict[str, Any] | None:
    """``running → failed`` (idempotent)."""
    return _terminal(job_id, worker_id, status="failed", result=None, error=error)


def revoke(job_id: str, *, requeue: bool = True, reason: str | None = None) -> dict[str, Any] | None:
    """The broker kills a job's grant at will. ``requeue=True`` (default) sends it
    back to ``queued`` (cleared grant → another worker can be granted it — the
    "kill and give permission for the next node_job" path); ``requeue=False``
    marks it ``killed`` (terminal). Either way the holder's permission is
    withdrawn (``has_permission`` goes False) and the holder is freed. Returns the
    updated job, or ``None`` if it wasn't granted/running."""
    d = get_dialect()
    new_status = "queued" if requeue else "killed"
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT granted_worker FROM bw_jobs WHERE job_id = %(id)s", {"id": job_id})
        held = cur.fetchone()
        holder = held["granted_worker"] if held else None
        cur.execute(
            "UPDATE bw_jobs SET status = %(st)s, granted_worker = NULL, grant_expires_at = NULL, "
            f"error = %(err)s, updated_at = {d.now} "
            "WHERE job_id = %(id)s AND status IN ('granted', 'running') "
            f"RETURNING {d.qualify_returning('bw_jobs', _JOB_COLS)}",
            {"st": new_status, "err": None if requeue else reason, "id": job_id},
        )
        row = cur.fetchone()
        if row is not None and holder is not None:
            cur.execute(
                "UPDATE bw_workers SET state = 'waiting' WHERE worker_id = %(w)s AND state = 'running'",
                {"w": holder},
            )
        conn.commit()
        return _row_to_job(row)


def sweep_unhealthy(*, stale_s: float) -> list[str]:
    """The broker's health check. Mark every worker whose heartbeat is older than
    ``stale_s`` as ``dead``, then re-queue any ``granted``/``running`` job whose
    holder is now dead **or** whose grant lease has lapsed — so a healthy worker
    can be granted it next. Returns the re-queued job ids. This is the sole
    recovery path for a worker that died mid-job."""
    d = get_dialect()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE bw_workers SET state = 'dead' "
            f"WHERE state != 'dead' AND last_seen < {d.past_seconds('%(stale)s')}",
            {"stale": stale_s},
        )
        cur.execute(
            "UPDATE bw_jobs SET status = 'queued', granted_worker = NULL, grant_expires_at = NULL, "
            f"error = NULL, updated_at = {d.now} "
            "WHERE status IN ('granted', 'running') AND ("
            f"grant_expires_at < {d.now} "
            "OR granted_worker IN (SELECT worker_id FROM bw_workers WHERE state = 'dead')"
            ") RETURNING job_id",
        )
        rows = cur.fetchall()
        conn.commit()
        return [r["job_id"] for r in rows]


# ── read views (power the web panel + JSON API) ──────────────────────────────


def list_jobs(*, project: str | None = None, status: str | None = None, limit: int = 100):
    """Recent jobs on the shared queue, newest first, with optional project/status
    filters. Read-only."""
    d = get_dialect()
    where, params = [], {"lim": int(limit)}
    if project is not None:
        where.append("project = %(project)s")
        params["project"] = project
    if status is not None:
        where.append("status = %(status)s")
        params["status"] = status
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_JOB_COLS)}, created_at, updated_at FROM bw_jobs "
            f"{clause} ORDER BY created_at DESC, {d.creation_order('bw_jobs')} DESC "
            "LIMIT %(lim)s",
            params,
        )
        return [_row_to_job(r) for r in cur.fetchall()]


def list_workers():
    """All registered workers, ordered by resource then id. Read-only."""
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT worker_id, project, resource, state, last_seen "
            "FROM bw_workers ORDER BY resource ASC, worker_id ASC"
        )
        return cur.fetchall()


def queue_counts(*, project: str | None = None):
    """Per ``(resource, status)`` job counts — powers the panel KPI strip."""
    params, clause = {}, ""
    if project is not None:
        clause = "WHERE project = %(project)s"
        params["project"] = project
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT resource, status, COUNT(*) AS n FROM bw_jobs {clause} "
            "GROUP BY resource, status",
            params or None,
        )
        return cur.fetchall()
