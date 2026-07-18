"""Over-claim reaper — the runtime SAFETY-FALLBACK for the per-box slot arbiter.

The slot arbiter (`claim_next_gpu_job(max_running=N)`) is claim-time PREVENTION.
If it is ever bypassed — a claim path that forgets `max_running`, a counter bug,
a race a future lane opens — a box can end up RUNNING more jobs than its
advertised capacity (the panel's `NODE JOBS ⚠ 2/1`). This reaper is the runtime
backstop: it re-queues the NEWEST over-capacity job(s) (keeping the already-
running oldest), so an over-claim self-heals within one sweep instead of two
jobs contending on one GPU until a lease lapses.

Scope: per (host, queue, project) within a DB — the same grouping the arbiter
uses. Requeue is resume-style (front-of-queue, NO `watchdog_retries` bump — an
over-claim isn't the job's fault), CAS on `status='running'` so a job that just
finished is never yanked, and JobStatusWatcher covers the double-run (clearing
`claimed_by` self-kills a still-alive claimant).
"""
from __future__ import annotations

import datetime as dt

from queue_workflows import node_pool, node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


# ── the pure victim picker (keep oldest, reap newest) ─────────────────────


def test_pick_victims_keeps_oldest_reaps_newest():
    rows = [("j1", 100), ("j2", 300), ("j3", 200)]      # (id, started ordinal)
    assert set(node_queue.pick_over_claim_victims(rows, 1)) == {"j2", "j3"}
    assert node_queue.pick_over_claim_victims(rows, 2) == ["j2"]   # only the newest


def test_pick_victims_none_when_within_capacity():
    assert node_queue.pick_over_claim_victims([("a", 1), ("b", 2)], 2) == []
    assert node_queue.pick_over_claim_victims([("a", 1)], 5) == []


def test_pick_victims_tie_break_by_id_is_deterministic():
    rows = [("b", 5), ("a", 5), ("c", 5)]               # equal start → id order
    assert node_queue.pick_over_claim_victims(rows, 1) == ["b", "c"]  # keep "a"


def test_pick_victims_none_started_at_is_kept_not_reaped():
    # a running row that somehow lacks started_at is anomalous — keep it, don't
    # aggressively reap (the lease/dead-worker sweeps own that case)
    rows = [("late", 100), ("nostart", None)]
    assert node_queue.pick_over_claim_victims(rows, 1) == ["late"]


# ── the reaper against a real store ───────────────────────────────────────


def _status(job_id):
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT status, claimed_by, is_priority, watchdog_retries "
                    "FROM workflow_node_jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def _set_started(ids_oldest_first):
    base = dt.datetime(2026, 7, 18, 12, 0, 0, tzinfo=dt.timezone.utc)
    with connection() as conn, conn.cursor() as cur:
        for i, jid in enumerate(ids_oldest_first):
            cur.execute("UPDATE workflow_node_jobs SET started_at = %s WHERE id = %s",
                        (base + dt.timedelta(seconds=i * 10), jid))


def _over_claim(host="box-a-gpu", n=3):
    """Force an over-claim: claim n gpu jobs on one box with the arbiter OFF
    (max_running=None) — simulating a prevention layer that failed."""
    run = make_run()
    for i in range(n):
        node_queue.enqueue_node_job(run_id=run, node_id=f"n{i}", node_module="x",
                                    queue="gpu", priority=100)
    claimed = []
    for _ in range(n):
        j = node_queue.claim_next_gpu_job(0, host=host, max_running=None)
        assert j is not None
        claimed.append(j["id"])
    return run, claimed


def test_reap_keeps_oldest_requeues_newest():
    run, claimed = _over_claim("box-a-gpu", 3)
    _set_started(claimed)                                  # claimed[0] oldest
    node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu", concurrency=1)

    reaped = node_queue.reap_over_claimed_boxes(queue="gpu")

    assert {r["job_id"] for r in reaped} == {claimed[1], claimed[2]}
    assert _status(claimed[0])["status"] == "running"      # the oldest keeps the slot
    for jid in (claimed[1], claimed[2]):
        row = _status(jid)
        assert row["status"] == "queued" and row["claimed_by"] is None
        assert row["is_priority"]                           # front-of-queue resume
        assert row["watchdog_retries"] == 0                 # not the job's fault


def test_reap_respects_capacity_of_two():
    run, claimed = _over_claim("box-a-gpu", 3)
    _set_started(claimed)
    node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu", concurrency=2)
    reaped = node_queue.reap_over_claimed_boxes(queue="gpu")
    assert {r["job_id"] for r in reaped} == {claimed[2]}    # only the single newest


def test_reap_noop_within_capacity():
    _run, claimed = _over_claim("box-a-gpu", 1)
    node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu", concurrency=1)
    assert node_queue.reap_over_claimed_boxes(queue="gpu") == []
    assert _status(claimed[0])["status"] == "running"


def test_reap_skips_a_box_with_no_heartbeat():
    # capacity unknown (no heartbeat) → don't touch it; the lease/dead-worker
    # sweeps own the "worker vanished" case
    _run, claimed = _over_claim("ghost-gpu", 3)
    assert node_queue.reap_over_claimed_boxes(queue="gpu") == []
    assert _status(claimed[0])["status"] == "running"


def test_reap_cancels_victims_of_a_terminal_run():
    run, claimed = _over_claim("box-a-gpu", 3)
    _set_started(claimed)
    node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu", concurrency=1)
    with connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE workflow_runs SET status = 'failed' WHERE id = %s", (run,))
    node_queue.reap_over_claimed_boxes(queue="gpu")
    for jid in (claimed[1], claimed[2]):
        assert _status(jid)["status"] == "cancelled"        # no ghost re-queue


# ── the NodePool sweep (interval-gated, disable-able wrapper) ──────────────


def _sweep_pool(*, disabled=False, interval_s=0.0, last_run=0.0, now=None):
    pool = node_pool.NodePool.__new__(node_pool.NodePool)
    pool._over_claim_disabled = disabled
    pool._over_claim_interval_s = interval_s
    pool._over_claim_last_run = last_run
    pool._over_claim_now_fn = (lambda: now) if now is not None else None
    return pool


def test_sweep_requeues_the_newest_over_capacity_job():
    _run, claimed = _over_claim("box-a-gpu", 3)
    _set_started(claimed)
    node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu", concurrency=1)
    _sweep_pool()._sweep_over_claimed_boxes()
    assert _status(claimed[0])["status"] == "running"       # oldest keeps the slot
    assert _status(claimed[1])["status"] == "queued"
    assert _status(claimed[2])["status"] == "queued"


def test_sweep_is_interval_gated():
    _run, claimed = _over_claim("box-a-gpu", 3)
    _set_started(claimed)
    node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu", concurrency=1)
    # last ran at t=100, now t=105, gate 10 s → still inside the window → no-op
    _sweep_pool(interval_s=10.0, last_run=100.0, now=105.0)._sweep_over_claimed_boxes()
    for jid in claimed:
        assert _status(jid)["status"] == "running"          # nothing reaped yet


def test_sweep_is_a_noop_when_disabled():
    _run, claimed = _over_claim("box-a-gpu", 3)
    _set_started(claimed)
    node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu", concurrency=1)
    _sweep_pool(disabled=True)._sweep_over_claimed_boxes()
    for jid in claimed:
        assert _status(jid)["status"] == "running"          # disable knob honored
