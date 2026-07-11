"""Broker orchestration core — the pull→grant permission inversion.

The v2 broker owns a SHARED, project-labelled CPU/GPU queue and DECIDES who runs:
a worker asks the broker for permission (``grant_next``); the broker grants the
next job for the worker's project/resource, subject to a cross-project capacity
gate; it can revoke (kill) a job at any time; and it health-checks workers,
marking an unhealthy one dead and re-queuing its job for the next grant.

These run on the hermetic SQLite store (``QUEUE_WORKFLOWS_TEST_SQLITE=1``) — no
Postgres server, no real DB touched. Time-dependent behaviour (grant expiry,
worker staleness) is driven deterministically with negative lease/stale windows
(a boundary already in the past/future), never ``sleep``.
"""

from __future__ import annotations

import pytest

from queue_workflows import broker_service as bs
from queue_workflows import db
from queue_workflows.dialect import get_dialect


def _backdate(table, key_col, key, ts_col, seconds):
    """Deterministically push a timestamp column ``seconds`` into the past — the
    portable way to simulate an expired grant / a stale worker without sleeping.
    (A negative lease can't be used: the dialect builds intervals by string
    concat, so a negative value yields an invalid SQLite modifier.)"""
    d = get_dialect()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET {ts_col} = {d.past_seconds('%(s)s')} WHERE {key_col} = %(k)s",
            {"s": seconds, "k": key},
        )
        conn.commit()


@pytest.fixture(autouse=True)
def _fresh_broker_schema():
    """Create the broker-service tables and wipe them between tests. They live
    alongside the engine tables in the same store; the engine conftest truncates
    only its own tables, so the broker tables are cleaned here."""
    bs.ensure_schema()
    with db.connection() as conn, conn.cursor() as cur:
        for tbl in ("bw_jobs", "bw_workers"):
            cur.execute(f"DELETE FROM {tbl}")
        conn.commit()
    yield


# ── enqueue + the grant decision ────────────────────────────────────────────


def test_grant_next_picks_priority_then_fifo():
    # Two default-priority jobs (j1 before j2) + one high-priority (lower int).
    j1 = bs.submit_job(project="A", resource="cpu", priority=100)
    j2 = bs.submit_job(project="A", resource="cpu", priority=100)
    j3 = bs.submit_job(project="A", resource="cpu", priority=50)
    bs.register_worker("wA", project="A", resource="cpu")

    # High priority first, then FIFO among equal priority.
    assert bs.grant_next("wA", lease_s=30)["job_id"] == j3
    bs.complete_job(j3, "wA")
    assert bs.grant_next("wA", lease_s=30)["job_id"] == j1
    bs.complete_job(j1, "wA")
    assert bs.grant_next("wA", lease_s=30)["job_id"] == j2


def test_grant_marks_job_granted_and_worker_running():
    j = bs.submit_job(project="A", resource="gpu")
    bs.register_worker("wA", project="A", resource="gpu")

    granted = bs.grant_next("wA", lease_s=30)
    assert granted["job_id"] == j
    assert granted["status"] == "granted"
    assert granted["granted_worker"] == "wA"
    assert bs.get_job(j)["status"] == "granted"
    assert bs.get_worker("wA")["state"] == "running"


def test_no_work_returns_none():
    bs.register_worker("wA", project="A", resource="cpu")
    assert bs.grant_next("wA", lease_s=30) is None


def test_worker_only_gets_its_own_project_jobs():
    bs.submit_job(project="B", resource="cpu")  # other project's work
    bs.register_worker("wA", project="A", resource="cpu")
    assert bs.grant_next("wA", lease_s=30) is None  # A has no jobs of its own


# ── cross-project capacity arbitration (the broker deciding) ────────────────


def test_capacity_gate_denies_across_projects():
    bs.submit_job(project="A", resource="cpu")
    bs.submit_job(project="B", resource="cpu")
    bs.register_worker("wA", project="A", resource="cpu")
    bs.register_worker("wB", project="B", resource="cpu")

    # capacity=1 cpu across ALL projects: A grabs the one slot...
    a = bs.grant_next("wA", lease_s=30, capacity=1)
    assert a is not None
    # ...B is denied even though B has queued work (broker arbitrates the shared core).
    assert bs.grant_next("wB", lease_s=30, capacity=1) is None
    # freeing the slot lets B in.
    bs.complete_job(a["job_id"], "wA")
    assert bs.grant_next("wB", lease_s=30, capacity=1) is not None


# ── terminals (idempotent) ──────────────────────────────────────────────────


def test_begin_then_complete_and_idempotent():
    j = bs.submit_job(project="A", resource="cpu")
    bs.register_worker("wA", project="A", resource="cpu")
    bs.grant_next("wA", lease_s=30)

    assert bs.begin_job(j, "wA")["status"] == "running"
    assert bs.complete_job(j, "wA", result={"ok": True})["status"] == "done"
    assert bs.get_job(j)["result"] == {"ok": True}
    # second terminal is a no-op (idempotency guard) and worker is freed.
    assert bs.complete_job(j, "wA") is None
    assert bs.get_worker("wA")["state"] == "waiting"


def test_fail_job():
    j = bs.submit_job(project="A", resource="cpu")
    bs.register_worker("wA", project="A", resource="cpu")
    bs.grant_next("wA", lease_s=30)
    assert bs.fail_job(j, "wA", error="boom")["status"] == "failed"
    assert bs.get_job(j)["error"] == "boom"


# ── broker kill: revoke ─────────────────────────────────────────────────────


def test_revoke_requeues_and_withdraws_permission():
    j = bs.submit_job(project="A", resource="cpu")
    bs.register_worker("wA", project="A", resource="cpu")
    bs.grant_next("wA", lease_s=30)
    assert bs.has_permission(j, "wA") is True

    bs.revoke(j, requeue=True, reason="operator")
    # the worker that held it must stop — permission is withdrawn.
    assert bs.has_permission(j, "wA") is False
    assert bs.get_job(j)["status"] == "queued"
    # a fresh worker can now be granted the requeued job.
    bs.register_worker("wA2", project="A", resource="cpu")
    assert bs.grant_next("wA2", lease_s=30)["job_id"] == j


def test_revoke_kill_without_requeue():
    j = bs.submit_job(project="A", resource="cpu")
    bs.register_worker("wA", project="A", resource="cpu")
    bs.grant_next("wA", lease_s=30)
    bs.revoke(j, requeue=False)
    assert bs.get_job(j)["status"] == "killed"
    assert bs.has_permission(j, "wA") is False


# ── broker health check: sweep unhealthy → reassign ─────────────────────────


def test_sweep_unhealthy_marks_dead_and_requeues():
    j = bs.submit_job(project="A", resource="cpu")
    bs.register_worker("wA", project="A", resource="cpu")
    bs.grant_next("wA", lease_s=30)

    # Worker went silent an hour ago → stale beyond a 1s threshold.
    _backdate("bw_workers", "worker_id", "wA", "last_seen", 3600)
    requeued = bs.sweep_unhealthy(stale_s=1)
    assert j in requeued
    assert bs.get_worker("wA")["state"] == "dead"
    assert bs.get_job(j)["status"] == "queued"
    assert bs.has_permission(j, "wA") is False
    # a healthy worker gets the reassigned job.
    bs.register_worker("wB", project="A", resource="cpu")
    assert bs.grant_next("wB", lease_s=30)["job_id"] == j


def test_sweep_reclaims_expired_grant_even_if_worker_fresh():
    j = bs.submit_job(project="A", resource="cpu")
    bs.register_worker("wA", project="A", resource="cpu")
    bs.grant_next("wA", lease_s=30)
    # The grant expired (worker stopped renewing) but the worker heartbeat is fresh.
    _backdate("bw_jobs", "job_id", j, "grant_expires_at", 5)
    assert bs.has_permission(j, "wA") is False  # expired grant grants no permission

    requeued = bs.sweep_unhealthy(stale_s=3600)  # worker fresh; only the grant expired
    assert j in requeued
    assert bs.get_job(j)["status"] == "queued"


# ── the client-side permission gate (thin wrapper the worker uses) ──────────


def test_client_gate_end_to_end():
    bs.submit_job(project="A", resource="cpu", payload={"n": 7})
    # ask_to_run registers + heartbeats + asks the broker for permission in one call.
    job = bs.ask_to_run("wA", project="A", resource="cpu", lease_s=30)
    assert job is not None and job["payload"] == {"n": 7}
    # while permitted, the worker runs; keep_permission renews + confirms not killed.
    assert bs.keep_permission(job["job_id"], "wA", lease_s=30) is True
    result = bs.finish(job["job_id"], "wA", result={"doubled": 14})
    assert result["status"] == "done"


def test_client_gate_denied_returns_none():
    # capacity 0 => broker denies everyone (no free cores).
    bs.submit_job(project="A", resource="cpu")
    assert bs.ask_to_run("wA", project="A", resource="cpu", lease_s=30, capacity=0) is None
