"""FAST zombie-job reclaim — kill a dead worker's jobs in seconds, not lease-length.

THE INCIDENT. Recreating a worker leaves its in-flight job as a ZOMBIE: a ``running``
row claimed by the dead process. Nothing touched it until the 600 s lease lapsed, so
the box's ONE slot sat occupied ~10 min while a live worker idled next to a backlog.
The lease is deliberately long (a live-but-busy worker's renewal hiccup must not get it
reclaimed), so the fix is NOT a shorter lease — it's using the FAST liveness signal
(worker_heartbeats, 10 s cadence, stale at 30 s) to reclaim the jobs of a worker that
is actually gone:

  1. **worker-boot self-reclaim** — a restarting worker re-queues its dead
     predecessor's rows immediately (seconds after the supervisor restart);
  2. **orchestrator fast dead-reclaim** — ``_sweep_dead_workers`` now also re-queues a
     flagged worker's jobs at flag time (~30 s), covering a worker that is gone and
     NOT coming back (container stopped, box down).

SAFETY. Both paths ride one atomic predicate: rows are re-queued ONLY while the
label's heartbeat is STALE (or absent). A multi-process lane (30 cpu procs share one
label) is safe because any live sibling keeps the heartbeat fresh — and a claim can
only happen AFTER the claimer's first beat, so "claimed row + stale heartbeat" can
never describe a live job. Double-run is prevented by the same guarantee every reclaim
relies on: clearing ``claimed_by`` trips a still-alive claimant's JobStatusWatcher.

Requires Postgres (heartbeat staleness uses ``make_interval``): run with
``QUEUE_WORKFLOWS_TEST_DB_URL=postgresql://...:.../qw_test``.
"""
from __future__ import annotations

from queue_workflows import claim_worker, node_pool, node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


def _age_heartbeat(host: str, queue: str, age_s: float) -> None:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE worker_heartbeats SET last_seen = now() - make_interval(secs => %s) "
            "WHERE host_label = %s AND queue = %s",
            (float(age_s), host, queue),
        )


def _zombie(host: str, *, queue: str = "gpu", heartbeat_age_s: float | None = 120.0) -> str:
    """A RUNNING row claimed by ``host`` whose owner process is 'dead': the label's
    heartbeat is aged past staleness (or absent when ``heartbeat_age_s`` is None)."""
    run_id = make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id=f"z-{host}", node_module="x", queue=queue,
    )
    job = node_queue.claim_next_gpu_job(0, None, host=host) if queue == "gpu" \
        else node_queue.claim_next_cpu_job(0, host=host)
    assert job is not None and job["claimed_by"] == host
    if heartbeat_age_s is None:
        with connection() as c, c.cursor() as cur:
            cur.execute(
                "DELETE FROM worker_heartbeats WHERE host_label = %s", (host,),
            )
    else:
        node_queue.upsert_worker_heartbeat(host_label=host, queue=queue, concurrency=1)
        _age_heartbeat(host, queue, heartbeat_age_s)
    return job["id"]


def _status(job_id: str) -> dict:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT status, claimed_by, is_priority, watchdog_retries "
            "FROM workflow_node_jobs WHERE id = %s", (job_id,),
        )
        return cur.fetchone()


# ── the atomic primitive: requeue-if-worker-stale ────────────────────────────


def test_stale_worker_zombie_is_requeued_to_the_front():
    jid = _zombie("box-dead-gpu")
    n = node_queue.requeue_running_if_worker_stale("box-dead-gpu", "gpu")
    row = _status(jid)
    assert n == 1
    assert row["status"] == "queued" and row["claimed_by"] is None
    assert row["is_priority"] is True                 # recovered work runs next
    assert row["watchdog_retries"] == 0               # resume-style, not a fault


def test_fresh_worker_job_is_NEVER_touched():
    # The multi-process-lane guarantee: a live sibling keeps the heartbeat fresh,
    # so a boot-reclaim can't yank a live job.
    jid = _zombie("box-live-gpu", heartbeat_age_s=0.0)   # fresh beat
    n = node_queue.requeue_running_if_worker_stale("box-live-gpu", "gpu")
    assert n == 0
    assert _status(jid)["status"] == "running"


def test_absent_heartbeat_counts_as_stale():
    # No heartbeat row at all ⇒ nobody with this label is alive ⇒ reclaim.
    jid = _zombie("box-nohb-gpu", heartbeat_age_s=None)
    n = node_queue.requeue_running_if_worker_stale("box-nohb-gpu", "gpu")
    assert n == 1 and _status(jid)["status"] == "queued"


def test_scoped_to_its_own_host_and_queue():
    victim = _zombie("box-a-gpu")
    bystander = _zombie("box-b-gpu")                     # different host, also stale
    node_queue.requeue_running_if_worker_stale("box-a-gpu", "gpu")
    assert _status(victim)["status"] == "queued"
    assert _status(bystander)["status"] == "running"     # untouched


def test_idempotent_second_call_is_a_noop():
    _zombie("box-idem-gpu")
    assert node_queue.requeue_running_if_worker_stale("box-idem-gpu", "gpu") == 1
    assert node_queue.requeue_running_if_worker_stale("box-idem-gpu", "gpu") == 0


# ── worker-boot self-reclaim ─────────────────────────────────────────────────


def test_worker_boot_reclaims_its_dead_predecessors_zombie():
    jid = _zombie("box-boot-gpu")
    w = claim_worker.ClaimWorker(queue="gpu", host="box-boot-gpu")
    w._reclaim_predecessor_zombies()
    assert _status(jid)["status"] == "queued"


def test_worker_boot_skips_when_a_sibling_is_alive():
    jid = _zombie("box-sib-cpu", queue="cpu", heartbeat_age_s=0.0)  # sibling beating
    w = claim_worker.ClaimWorker(queue="cpu", host="box-sib-cpu")
    w._reclaim_predecessor_zombies()
    assert _status(jid)["status"] == "running"           # live sibling's job survives


def test_run_forever_wires_the_boot_reclaim():
    # Source-level wiring pin: the startup path must call the boot reclaim after the
    # schema gate / park gate and before the claim loop.
    import inspect
    src = inspect.getsource(claim_worker.ClaimWorker.run_forever)
    assert "_reclaim_predecessor_zombies" in src


# ── orchestrator fast dead-reclaim (the flag now also frees the jobs) ─────────


def test_sweep_dead_workers_requeues_the_flagged_workers_jobs():
    jid = _zombie("box-swept-gpu")
    pool = node_pool.NodePool.__new__(node_pool.NodePool)
    pool._dead_worker_last_run = 0.0
    pool._dead_worker_interval_s = 0.0
    pool._sweep_dead_workers()
    row = _status(jid)
    assert row["status"] == "queued" and row["is_priority"] is True


def test_sweep_dead_workers_leaves_fresh_workers_jobs_alone():
    jid = _zombie("box-fresh-gpu", heartbeat_age_s=0.0)
    pool = node_pool.NodePool.__new__(node_pool.NodePool)
    pool._dead_worker_last_run = 0.0
    pool._dead_worker_interval_s = 0.0
    pool._sweep_dead_workers()
    assert _status(jid)["status"] == "running"


# ── frozen-lease sweep: per-JOB liveness, catches what heartbeats can't ───────
#
# A FAST worker restart (< heartbeat-staleness) hides the corpse: the old process
# died between beats, the new one starts beating the SAME label within the window,
# so both the boot-reclaim and the stale-heartbeat sweep see a "live" label while
# the predecessor's job rots. The unambiguous PER-JOB signal is the lease renewal
# itself: a live claimant advances lease_expires_at every ~10 s; a zombie's value
# FREEZES. The orchestrator samples running rows and re-queues any whose lease has
# not moved for the freeze window — with a CAS guard (reclaim only if the value
# still equals the frozen one we observed), so an in-between renewal aborts it.


def test_detect_frozen_leases_flags_only_after_the_window():
    detect = node_queue.detect_frozen_leases
    rows = [("j1", "L1")]
    to_reclaim, sample = detect({}, rows, now=100.0, frozen_after_s=35.0)
    assert to_reclaim == []                       # first sighting starts the clock
    to_reclaim, sample = detect(sample, rows, now=120.0, frozen_after_s=35.0)
    assert to_reclaim == []                       # 20 s frozen — under the window
    to_reclaim, sample = detect(sample, rows, now=140.0, frozen_after_s=35.0)
    assert to_reclaim == [("j1", "L1")]           # 40 s frozen — dead


def test_detect_frozen_leases_an_advancing_lease_resets_the_clock():
    detect = node_queue.detect_frozen_leases
    _, sample = detect({}, [("j1", "L1")], now=100.0, frozen_after_s=35.0)
    # renewal happened: value moved → clock restarts, never flagged
    to_reclaim, sample = detect(sample, [("j1", "L2")], now=140.0, frozen_after_s=35.0)
    assert to_reclaim == []
    to_reclaim, _ = detect(sample, [("j1", "L2")], now=170.0, frozen_after_s=35.0)
    assert to_reclaim == []                       # only 30 s since L2 appeared


def test_detect_frozen_leases_forgets_rows_that_left_running():
    detect = node_queue.detect_frozen_leases
    _, sample = detect({}, [("j1", "L1")], now=100.0, frozen_after_s=35.0)
    to_reclaim, sample = detect(sample, [], now=200.0, frozen_after_s=35.0)
    assert to_reclaim == [] and sample == {}      # finished/reclaimed elsewhere


def test_cas_requeue_frees_a_frozen_row_resume_style():
    jid = _zombie("box-frozen-gpu", heartbeat_age_s=0.0)   # heartbeat FRESH (the gap!)
    with connection() as c, c.cursor() as cur:
        cur.execute("SELECT lease_expires_at FROM workflow_node_jobs WHERE id=%s", (jid,))
        lease = cur.fetchone()["lease_expires_at"]
    n = node_queue.requeue_running_if_lease_frozen(jid, lease)
    row = _status(jid)
    assert n == 1
    assert row["status"] == "queued" and row["claimed_by"] is None
    assert row["is_priority"] is True and row["watchdog_retries"] == 0


def test_cas_requeue_aborts_when_the_lease_advanced_in_between():
    # The live-claimant guard: a renewal between our sample and the UPDATE means
    # the worker is alive — the CAS mismatch must make the reclaim a no-op.
    import datetime as _dt
    jid = _zombie("box-alive-gpu", heartbeat_age_s=0.0)
    with connection() as c, c.cursor() as cur:
        cur.execute("SELECT lease_expires_at FROM workflow_node_jobs WHERE id=%s", (jid,))
        stale_lease = cur.fetchone()["lease_expires_at"]
        cur.execute(   # the claimant renews (advances the lease) — portable form
            "UPDATE workflow_node_jobs SET lease_expires_at = %s WHERE id=%s",
            (stale_lease + _dt.timedelta(seconds=60), jid),
        )
    assert node_queue.requeue_running_if_lease_frozen(jid, stale_lease) == 0
    assert _status(jid)["status"] == "running"


def test_cas_requeue_cancels_instead_when_the_parent_run_is_terminal():
    jid = _zombie("box-term-gpu", heartbeat_age_s=0.0)
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_runs SET status='cancelled' WHERE id::text = "
            "(SELECT run_id FROM workflow_node_jobs WHERE id=%s)", (jid,),
        )
        cur.execute("SELECT lease_expires_at FROM workflow_node_jobs WHERE id=%s", (jid,))
        lease = cur.fetchone()["lease_expires_at"]
    assert node_queue.requeue_running_if_lease_frozen(jid, lease) == 1
    assert _status(jid)["status"] == "cancelled"   # never a ghost-queued row


def test_node_pool_frozen_sweep_end_to_end_with_virtual_clock():
    jid = _zombie("box-e2e-gpu", heartbeat_age_s=0.0)      # fresh heartbeat: the gap case
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._frozen_lease_interval_s = 0.0                    # no gate in-test
    fake_now = [1000.0]
    pool._frozen_lease_now_fn = lambda: fake_now[0]
    pool._sweep_frozen_leases()                            # sample
    assert _status(jid)["status"] == "running"
    fake_now[0] += 40.0                                    # > the default 35 s window
    pool._sweep_frozen_leases()                            # detect + CAS requeue
    row = _status(jid)
    assert row["status"] == "queued" and row["is_priority"] is True


# ── ghosts can no longer accumulate: orphan-cancel is ON by default ──────────


def test_orphan_cancel_sweep_is_on_by_default():
    from queue_workflows.config import EngineConfig
    assert EngineConfig().cancel_orphan_queued_jobs is True
