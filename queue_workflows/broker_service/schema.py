"""Clean-slate schema for the broker orchestration core (v2 control plane).

This is deliberately **outside** the engine's 0001–0017 migration chain. The v2
broker is a fresh design (the operator's "nuke the DB, no migrations" reset), so
it ships a single **idempotent** ``ensure_schema()`` (``CREATE TABLE IF NOT
EXISTS``) instead of an incremental, ledgered migration chain. It coexists with
the legacy engine tables in the same relational store (pg or sqlite) via the
:mod:`queue_workflows.db` / :mod:`queue_workflows.dialect` seam, so a later pass
can retire the old engine without a data migration.

Two tables, both keyed by caller-supplied TEXT ids (no serial/autoincrement — so
ids are backend-agnostic):

* ``bw_jobs``    — the SHARED CPU/GPU queue for ALL projects. Every row carries a
  ``project`` label; the broker (not the worker) decides which row a worker may
  run. ``status``: ``queued → granted → running → done | failed | killed`` (a
  ``killed``/expired grant is re-queued by the health sweep).
* ``bw_workers`` — the worker registry + liveness. ``state``: ``waiting`` (idle,
  awaiting a grant) / ``running`` (holds a grant) / ``dead`` (failed the health
  check; its grant is reclaimed).

Timestamps + JSON payloads are stored per-dialect (``TIMESTAMPTZ``/``TEXT`` and
``JSONB``-free ``TEXT`` holding ``json.dumps``) so the same reader works on both.
"""

from __future__ import annotations

from queue_workflows import db
from queue_workflows.dialect import is_sqlite

#: Valid resource lanes (the two shared queues). Host-defined lanes are a later pass.
RESOURCES = frozenset({"cpu", "gpu"})


def _ddl(sqlite: bool) -> list[str]:
    """The CREATE statements for the active dialect. pg uses ``TIMESTAMPTZ`` +
    ``now()`` defaults; sqlite uses ``TEXT`` ISO-8601 + ``datetime('now')`` (UTC,
    lexically comparable) — matching the engine's ``migrations_sqlite`` convention."""
    ts = "TEXT" if sqlite else "TIMESTAMPTZ"
    now = "(datetime('now'))" if sqlite else "now()"
    return [
        f"""CREATE TABLE IF NOT EXISTS bw_workers (
            worker_id     TEXT PRIMARY KEY,
            project       TEXT NOT NULL,
            resource      TEXT NOT NULL,
            state         TEXT NOT NULL DEFAULT 'waiting',
            last_seen     {ts} NOT NULL DEFAULT {now},
            registered_at {ts} NOT NULL DEFAULT {now}
        )""",
        f"""CREATE TABLE IF NOT EXISTS bw_jobs (
            job_id          TEXT PRIMARY KEY,
            project         TEXT NOT NULL,
            resource        TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'queued',
            priority        INTEGER NOT NULL DEFAULT 100,
            granted_worker  TEXT,
            grant_expires_at {ts},
            payload         TEXT,
            result          TEXT,
            error           TEXT,
            created_at      {ts} NOT NULL DEFAULT {now},
            updated_at      {ts} NOT NULL DEFAULT {now}
        )""",
        # The grant pick: next queued row for a (resource, project), priority then FIFO.
        "CREATE INDEX IF NOT EXISTS bw_jobs_claim_idx "
        "ON bw_jobs (resource, project, status, priority)",
        # The health sweep: find a dead worker's held rows.
        "CREATE INDEX IF NOT EXISTS bw_jobs_granted_worker_idx ON bw_jobs (granted_worker)",
        "CREATE INDEX IF NOT EXISTS bw_workers_liveness_idx ON bw_workers (state, last_seen)",
    ]


def ensure_schema() -> None:
    """Idempotently create the broker-service tables in the engine's relational
    store. Safe to call at every broker startup; ``CREATE … IF NOT EXISTS`` makes
    it a no-op once applied. No version ledger (this is the clean-slate design)."""
    with db.connection() as conn, conn.cursor() as cur:
        for stmt in _ddl(is_sqlite()):
            cur.execute(stmt)
        conn.commit()
