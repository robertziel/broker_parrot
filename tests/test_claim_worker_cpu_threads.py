"""The CPU claim lane may run N node-jobs in ONE process (``cpu_worker_threads``).

A claim worker has always been concurrency-1: claim a row, run the body inline on
the main thread, then loop. N-way parallelism meant N PROCESSES
(``deploy/run_claim_lane.sh``), and that stays the recommended default — for a
CPU-bound pure-Python node the GIL serialises threads, so processes are the only
real parallelism.

But the engine already runs ``execute_node`` concurrently: the GPU VLM **pool
lane** has a PAR-sized ``ThreadPoolExecutor`` fed by a reserve-slot-then-claim
loop. For an **IO-bound** cpu node (HTTP, DB waits, an ffmpeg subprocess) that
same shape is exactly what's wanted, without a supervisor script. This module
pins the opt-in knob and the threaded lane:

  * ``threads = 1`` (the default) is byte-identical to today — no executor is
    ever created and the body runs inline;
  * ``threads = N`` claims up to N jobs concurrently and NEVER N+1;
  * a watchdog trip still HARD-EXITS (a Python thread cannot be killed), but it
    first releases THIS process's other in-flight rows resume-style — and only
    this process's, never a sibling process that happens to share the
    ``host_label``.
"""

from __future__ import annotations

import sys
import time
import threading
import types

import pytest

import queue_workflows
from queue_workflows import claim_worker, config, node_queue
from tests._helpers import make_run


@pytest.fixture(autouse=True)
def _node_pkg():
    queue_workflows.set_node_module_package("qwf_thr_nodes")
    yield


@pytest.fixture(autouse=True)
def _no_abandon_watcher(monkeypatch):
    """Disarm the real :class:`JobStatusWatcher` for this module.

    Several tests here deliberately yank a row out from under a RUNNING body —
    that is the whole scenario for the stale-id sweep, the re-claim accounting
    and the operator hard-stop. Clearing ``claimed_by`` is precisely what the
    abandon-watcher exists to detect, and its response is ``os._exit(77)``: it
    would kill the PYTEST PROCESS, not fail a test.

    It stayed hidden on sqlite because those tests finish inside the watcher's
    2 s first poll; against real Postgres every round-trip is slower, the poll
    lands, and the run dies mid-file with exit 77 — several tests AFTER the one
    that armed it, since the thread outlives its test. The engine is behaving
    correctly (killing a worker whose row was reassigned is the anti-double-run
    guarantee); this module is about slot accounting and sweep scoping, so it
    stubs the watcher out rather than racing it."""
    class _InertWatcher:
        def __init__(self, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(claim_worker, "JobStatusWatcher", _InertWatcher)
    yield


def _make_run() -> str:
    return make_run(status="running", workflow_name="_cpu_threads_test", out_dir=None)


def _install_fake_node(name: str, run_fn):
    mod = types.ModuleType(f"qwf_thr_nodes.{name}")
    mod.run = run_fn
    sys.modules[f"qwf_thr_nodes.{name}"] = mod


def _status(job_id: str) -> str:
    return node_queue.get_node_job(job_id)["status"]


# ── the knob ────────────────────────────────────────────────────────────────


def test_default_is_one_thread(monkeypatch):
    """Unset everywhere ⇒ 1. The serial worker is still the default."""
    monkeypatch.delenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS", raising=False)
    monkeypatch.delenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS", raising=False)
    assert claim_worker.resolve_cpu_worker_threads() == 1
    assert config.get_config().cpu_worker_threads == 1


def test_configure_sets_the_thread_count(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS", raising=False)
    monkeypatch.delenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS", raising=False)
    queue_workflows.configure(cpu_worker_threads=6)
    assert claim_worker.resolve_cpu_worker_threads() == 6


def test_env_beats_configure(monkeypatch):
    """The operator override wins over the host's ``configure()`` — same order as
    every other tunable."""
    queue_workflows.configure(cpu_worker_threads=6)
    monkeypatch.setenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS", "4")
    assert claim_worker.resolve_cpu_worker_threads() == 4

    monkeypatch.delenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS")
    monkeypatch.setenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS", "3")
    assert claim_worker.resolve_cpu_worker_threads() == 3


@pytest.mark.parametrize("raw", ["0", "-2", "", "not-a-number"])
def test_bad_values_never_produce_a_zero_width_lane(monkeypatch, raw):
    """A malformed or sub-1 value floors at 1 — a worker that claims nothing is
    strictly worse than the default."""
    monkeypatch.delenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS", raising=False)
    monkeypatch.setenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS", raw)
    assert claim_worker.resolve_cpu_worker_threads() >= 1


def test_only_the_cpu_queue_is_threaded(monkeypatch):
    """gpu owns its OWN two-lane budget (inline diffusion + the PAR VLM pool) and
    ingest is untouched by this change — neither may be widened by the cpu knob."""
    monkeypatch.setenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS", "8")
    assert claim_worker.ClaimWorker(queue="cpu", host="h").threads == 8
    assert claim_worker.ClaimWorker(
        queue="gpu", host="h", model_cache=object()).threads == 1
    assert claim_worker.ClaimWorker(queue="fetch", host="h").threads == 1


def test_explicit_threads_arg_beats_the_env(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_CPU_WORKER_THREADS", "8")
    assert claim_worker.ClaimWorker(queue="cpu", host="h", threads=2).threads == 2


# ── threads=1 stays exactly what it was ─────────────────────────────────────


def test_single_threaded_worker_runs_inline_and_builds_no_executor():
    """The default path must not merely behave the same — it must not construct a
    pool at all, so the serial deployment carries no thread machinery."""
    run_id = _make_run()
    ran: dict = {}

    def run(*, out=None, inputs=None, cancel_event=None):
        ran["thread"] = threading.current_thread().name
        return {"context_delta": {"ok": True}}

    _install_fake_node("_thr_inline", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="_thr_inline", queue="cpu",
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-inline", threads=1)
    assert w.run_once() is True
    # Ran on the CALLING thread — inline, not handed to a pool.
    assert ran["thread"] == threading.current_thread().name
    assert w._cpu_pool is None
    assert _status(job_id) == "completed"


# ── the threaded lane ───────────────────────────────────────────────────────


class _GatedNode:
    """A node body that blocks until released, so a test can hold N jobs
    simultaneously in flight and observe the claim ceiling."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.started = threading.Semaphore(0)
        self.running = 0
        self._lock = threading.Lock()
        self.peak = 0

    def run(self, *, out=None, inputs=None, cancel_event=None):
        with self._lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        self.started.release()
        self.gate.wait(timeout=30)
        with self._lock:
            self.running -= 1
        return {"context_delta": {"ok": True}}


def _await_started(node: _GatedNode, n: int) -> None:
    for _ in range(n):
        assert node.started.acquire(timeout=15), "job never started"


def test_threaded_worker_holds_n_jobs_at_once_and_never_n_plus_one():
    """THE contract: N concurrent, N+1 never. The 4th ``run_once`` with 3 threads
    full must claim NOTHING (returning False so the caller idles on NOTIFY) —
    over-claiming would strand a row on a busy worker while a peer sat idle."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_gated", node.run)
    ids = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"n{i}", node_module="_thr_gated", queue="cpu",
        )
        for i in range(4)
    ]
    w = claim_worker.ClaimWorker(queue="cpu", host="host-thr", threads=3)
    try:
        assert [w.run_once() for _ in range(3)] == [True, True, True]
        _await_started(node, 3)
        # Slots full: the claim is not even attempted.
        assert w.run_once() is False

        running = [j for j in ids if _status(j) == "running"]
        queued = [j for j in ids if _status(j) == "queued"]
        assert len(running) == 3 and len(queued) == 1
        assert node.peak == 3
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)

    assert sorted(_status(j) for j in ids) == [
        "completed", "completed", "completed", "queued",
    ]


def test_a_freed_slot_admits_the_next_job():
    """Capacity is a live ceiling, not a one-shot budget: when a job finishes the
    slot comes back and the next ``run_once`` claims again."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_refill", node.run)
    ids = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"r{i}", node_module="_thr_refill", queue="cpu",
        )
        for i in range(2)
    ]
    j3 = node_queue.enqueue_node_job(
        run_id=run_id, node_id="r2", node_module="_thr_refill", queue="cpu",
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-refill", threads=2)
    try:
        assert w.run_once() is True
        assert w.run_once() is True
        _await_started(node, 2)
        assert w.run_once() is False          # both slots busy — j3 stays queued
        assert _status(j3) == "queued"

        # Release the two in flight; the freed slots make the SAME worker
        # claimable again.
        node.gate.set()
        for _ in range(50):
            if not w.inflight_job_ids():
                break
            threading.Event().wait(0.1)
        assert w.inflight_job_ids() == set()
        assert w.run_once() is True
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)

    assert all(_status(j) == "completed" for j in [*ids, j3])


def test_threaded_worker_reports_its_inflight_job_ids():
    """The trip sweep needs to know exactly which rows THIS process holds — the
    set is the whole reason the sweep can be per-process rather than host-wide."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_ids", node.run)
    ids = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"i{i}", node_module="_thr_ids", queue="cpu",
        )
        for i in range(2)
    ]
    w = claim_worker.ClaimWorker(queue="cpu", host="host-ids", threads=2)
    try:
        w.run_once()
        w.run_once()
        _await_started(node, 2)
        assert w.inflight_job_ids() == set(ids)
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)
    assert w.inflight_job_ids() == set()


# ── the trip path ───────────────────────────────────────────────────────────


def test_trip_releases_this_processes_siblings_without_burning_their_retries():
    """A trip kills the PROCESS (a thread cannot be killed), so the N-1 healthy
    siblings die with it. Waiting out their leases would idle them for a full
    lease period, so the trip releases them first — resume-style: back to
    ``queued``, front of the line, and NO ``watchdog_retries`` bump, because being
    on a box whose neighbour wedged is not a fault of theirs."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_trip", node.run)
    ids = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"t{i}", node_module="_thr_trip", queue="cpu",
        )
        for i in range(3)
    ]
    w = claim_worker.ClaimWorker(queue="cpu", host="host-trip", threads=3)
    try:
        for _ in range(3):
            w.run_once()
        _await_started(node, 3)
        held = w.inflight_job_ids()
        assert len(held) == 3

        tripped = sorted(held)[0]
        exits: list[int] = []
        # The exact wiring _run_node gives a watchdog in threaded mode.
        w._trip_exit(tripped, on_exit=exits.append)(75)

        assert exits == [75]
        for sib in held - {tripped}:
            row = node_queue.get_node_job(sib)
            assert row["status"] == "queued"
            assert row["claimed_by"] is None
            assert int(row["watchdog_retries"] or 0) == 0    # not their fault
        # The tripped row itself is NOT touched by the sweep — the watchdog's own
        # requeue-with-bump path owns it.
        assert _status(tripped) == "running"
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_the_sweep_never_touches_another_processes_rows():
    """The KILL line. The recommended lane runs N processes that SHARE one
    ``host_label``, so a host-wide sweep would yank a healthy sibling PROCESS's
    rows; each of those would then self-exit on its JobStatusWatcher and one
    wedge would cascade into killing the whole box's lane. The sweep is scoped to
    the ids this process actually holds."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_scope", node.run)
    mine = node_queue.enqueue_node_job(
        run_id=run_id, node_id="mine", node_module="_thr_scope", queue="cpu",
    )
    theirs = node_queue.enqueue_node_job(
        run_id=run_id, node_id="theirs", node_module="_thr_scope", queue="cpu",
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="shared-label", threads=2)
    try:
        w.run_once()
        _await_started(node, 1)
        assert w.inflight_job_ids() == {mine}

        # A DIFFERENT process of the same lane claims the other row under the
        # SAME host_label (exactly what run_claim_lane.sh produces).
        other = node_queue.claim_next_cpu_job(0, host="shared-label")
        assert other is not None and other["id"] == theirs

        w._trip_exit(mine, on_exit=lambda code: None)(75)

        row = node_queue.get_node_job(theirs)
        assert row["status"] == "running"
        assert row["claimed_by"] == "shared-label"
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_the_sweep_never_touches_a_row_that_left_this_process():
    """The STALE-ID case — the one an id-only sweep gets wrong.

    An id stays in the in-flight set until the body returns, but the ROW can
    leave this process meanwhile: a lease lapse, a host-wide corrective requeue,
    the over-claim reaper. If a sibling PROCESS under the SAME ``host_label``
    then re-claims it, an id-scoped sweep still matches every ``WHERE`` term
    (id, running, claimed_by, queue, project) and steals a live row — and that
    sibling's JobStatusWatcher compares against the SHARED label, so it can't
    even tell. The sweep must therefore pin the CLAIM, not just the row: a
    re-claim stamps a new ``started_at``, which makes the stale id a no-op."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_stale", node.run)
    a = node_queue.enqueue_node_job(
        run_id=run_id, node_id="sa", node_module="_thr_stale", queue="cpu",
    )
    b = node_queue.enqueue_node_job(
        run_id=run_id, node_id="sb", node_module="_thr_stale", queue="cpu",
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="shared-label", threads=2)
    try:
        w.run_once()
        w.run_once()
        _await_started(node, 2)
        assert w.inflight_job_ids() == {a, b}

        # Both rows are yanked out from under this process and re-claimed by a
        # SIBLING PROCESS sharing the host_label. Our in-flight set is now stale.
        node_queue.requeue_running_for_worker("shared-label", "cpu")
        reclaimed = [node_queue.claim_next_cpu_job(0, host="shared-label")
                     for _ in range(2)]
        assert all(r is not None for r in reclaimed)

        # A trip on one of OUR (now stale) ids must not touch the sibling's rows.
        w._trip_exit(a, on_exit=lambda code: None)(75)

        for row in (node_queue.get_node_job(a), node_queue.get_node_job(b)):
            assert row["status"] == "running", "sweep stole a re-claimed row"
            assert row["claimed_by"] == "shared-label"
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_a_claim_that_raises_does_not_leak_a_slot():
    """A reservation is taken BEFORE the claim, so every failure path must give
    it back. A leaked reservation is not a narrowed lane — it leaves
    ``_cpu_drained`` cleared forever, and drain/shutdown wait on that event
    UNBOUNDED, so the worker hangs at teardown."""
    w = claim_worker.ClaimWorker(queue="cpu", host="host-leak", threads=2)

    def boom(**kw):
        raise RuntimeError("claim blew up")

    w._claim = boom
    assert w.run_once() is False
    assert w.inflight_job_ids() == set()
    assert w._wait_for_inflight(timeout=1) is True, "drained event left cleared"

    # A BaseException (KeyboardInterrupt in an embedded host) must not leak
    # either — it propagates, but the slot comes back.
    def interrupt(**kw):
        raise KeyboardInterrupt

    w._claim = interrupt
    with pytest.raises(KeyboardInterrupt):
        w.run_once()
    assert w.inflight_job_ids() == set()
    assert w._wait_for_inflight(timeout=1) is True


def test_a_failed_submit_does_not_leak_a_slot():
    """The other half of the leak. By the time ``submit`` runs, the real job id
    has REPLACED the placeholder in the in-flight map — so releasing the
    placeholder frees nothing and the slot is held forever, leaving
    ``_cpu_drained`` cleared. Drain and the SIGTERM teardown then wait on that
    event UNBOUNDED: the exact permanent hang, reachable in production via
    ``RuntimeError: can't start new thread`` under the M×P×N composition the
    deploy docs recommend."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_submitfail", node.run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="sf", node_module="_thr_submitfail", queue="cpu",
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-submitfail", threads=2)

    class _DeadPool:
        def submit(self, *a, **kw):
            raise RuntimeError("can't start new thread")

    w._ensure_cpu_pool = lambda: _DeadPool()
    assert w.run_once() is False
    assert w.inflight_job_ids() == set()
    assert w._wait_for_inflight(timeout=1) is True, "drained event left cleared"
    # The row was claimed but never run — it falls to lease reclaim, unchanged.
    assert _status(job_id) == "running"


def test_a_successful_submit_releases_the_slot_exactly_once():
    """The mirror hazard of releasing too eagerly. Once ``submit`` succeeds the
    POOL WORKER owns the slot and frees it in its own ``finally``; if the claim
    path freed it too, the count would drop while the job still ran and an extra
    claim would be admitted — N+1. Exactly one release per job."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_once", node.run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="oc", node_module="_thr_once", queue="cpu",
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-once", threads=2)
    releases: list[str] = []
    real_release = w._release_slot
    w._release_slot = lambda tok: (releases.append(tok), real_release(tok))[1]
    try:
        assert w.run_once() is True
        _await_started(node, 1)
        # While it runs the slot is still held — the claim path must not have
        # released the token it handed to the pool.
        assert w.inflight_job_ids() == {job_id}
        assert releases == []
        node.gate.set()
        w.shutdown_threads(timeout=15)
        assert len(releases) == 1, f"slot released more than once: {releases}"
    finally:
        node.gate.set()
        w._release_slot = real_release


def test_a_re_claimed_row_does_not_collapse_two_slots_into_one():
    """A SLOT is not a job id. The row this process is running can be re-queued
    out from under it (lapsed lease, a corrective requeue, the over-claim reaper)
    and then re-claimed BY THIS SAME PROCESS — at which point an id-keyed map
    overwrites the first entry and two live bodies share one slot. The count
    under-reads, so the lane admits an extra claim; worse, the map can go EMPTY
    while a body still runs, which sets the drained event and lets an operator
    drain park a worker that is still working — the exact thing
    ``_drain_park_if_requested`` says it prevents. The slot must be keyed by the
    RESERVATION, which is unique per claim."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_reclaim", node.run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="rc", node_module="_thr_reclaim", queue="cpu",
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-reclaim", threads=3)
    try:
        assert w.run_once() is True
        _await_started(node, 1)

        # The row leaves this process and comes straight back to it.
        node_queue.requeue_running_for_worker("host-reclaim", "cpu")
        assert w.run_once() is True
        _await_started(node, 1)          # the SECOND body is now running too

        # TWO bodies are running, so TWO slots are held.
        assert len(w._cpu_inflight) == 2, "two claims collapsed into one slot"
        assert not w._cpu_drained.is_set(), "drained while two bodies run"

        # Releasing one must NOT declare the lane quiescent.
        node.gate.set()
        assert w.inflight_job_ids() == {job_id}
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)
    assert w._cpu_drained.is_set()


def test_the_pool_is_built_exactly_once_under_concurrent_first_use():
    """Check-then-assign on ``_cpu_pool`` without the lock lets concurrent first
    callers build SEPARATE pools; the losers are orphaned with work submitted to
    them and are invisible to ``shutdown_threads``. Unreachable with today's
    single claim thread — which is precisely why it must be closed now."""
    w = claim_worker.ClaimWorker(queue="cpu", host="host-poolrace", threads=4)
    barrier = threading.Barrier(4)
    pools: list[object] = []

    real_ctor = claim_worker._CpuLanePool

    def slow_ctor(*a, **kw):
        time.sleep(0.05)                      # widen the check-then-assign window
        return real_ctor(*a, **kw)

    claim_worker._CpuLanePool = slow_ctor
    try:
        def build():
            barrier.wait(timeout=10)
            pools.append(w._ensure_cpu_pool())
        threads = [threading.Thread(target=build) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert len({id(p) for p in pools}) == 1, "built more than one pool"
    finally:
        claim_worker._CpuLanePool = real_ctor
        w.shutdown_threads(timeout=5)


def test_thread_creation_failure_happens_before_any_work_is_accepted():
    """The subtlest slot bug of all, and why the lane owns its threads.

    ``ThreadPoolExecutor.submit`` puts the work item on its queue and only THEN
    lazily starts a worker::

        self._work_queue.put(w)        # the body WILL run
        self._adjust_thread_count()    # RuntimeError: can't start new thread

    So under thread pressure — exactly what a widened lane invites — submit
    raises while an existing worker still runs the body. Inferring "not
    submitted, give the slot back" from that exception frees a slot the job still
    occupies: the count under-reads, the ceiling breaks (N+1 rows `running`), and
    the drained event can fire mid-body so a drain parks a working worker.

    Owning the threads moves the refusal to ONE place — pool construction, before
    any work exists — so "submit raised ⇒ nothing was enqueued" is true rather
    than merely usual."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_nothread", node.run)
    ids = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"nt{i}", node_module="_thr_nothread", queue="cpu",
        )
        for i in range(3)
    ]
    w = claim_worker.ClaimWorker(queue="cpu", host="host-nothread", threads=2)
    try:
        assert w.run_once() is True
        _await_started(node, 1)

        # Claim a second job while the OS refuses new threads. With a lazily
        # thread-starting executor this is the trap: the item is already queued
        # (it WILL run once a worker frees) but submit raises, so the claim path
        # concludes "not submitted" and hands the slot back.
        real_start = threading.Thread.start
        threading.Thread.start = lambda self: (_ for _ in ()).throw(
            RuntimeError("can't start new thread"))
        try:
            w.run_once()
        finally:
            threading.Thread.start = real_start

        # Two claims are outstanding, so two slots are held...
        assert len(w._cpu_inflight) == 2, "a slot was freed while its body waits"
        # ...and the ceiling therefore still refuses a third.
        assert w.run_once() is False
        running = [j for j in ids if _status(j) == "running"]
        assert len(running) <= 2, f"N+1 rows running: {len(running)}"
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_submitting_to_a_shut_down_pool_raises_instead_of_black_holing_the_job():
    """The other half of "submit cannot fail after enqueuing". A bare ``put``
    onto a queue whose workers have all taken their sentinels LOOKS like success
    and never runs — the caller would mark the slot submitted, never release it,
    and both drain and shutdown would wait on the drained event forever (round
    2's hang, by another door). Refusing keeps the caller's inference honest."""
    pool = claim_worker._CpuLanePool(width=1)
    pool.start()
    pool.shutdown(wait=True, timeout=5)
    with pytest.raises(RuntimeError):
        pool.submit(lambda: None)


def test_a_shut_down_lane_releases_the_slot_rather_than_hanging():
    """End to end: the claim path must survive a shut-down pool with its slot
    accounting intact, because a leak there is an unbounded wait, not a slow
    lane."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_closed", node.run)
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="cl", node_module="_thr_closed", queue="cpu",
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-closed", threads=2)
    dead = claim_worker._CpuLanePool(width=1)
    dead.start()
    dead.shutdown(wait=True, timeout=5)
    w._ensure_cpu_pool = lambda: dead
    try:
        assert w.run_once() is False
        assert w.inflight_job_ids() == set()
        assert w._wait_for_inflight(timeout=1) is True
    finally:
        node.gate.set()


def test_submit_cannot_fail_after_the_work_is_enqueued():
    """The property the slot rule rests on, pinned directly: once the pool
    exists, handing it work never raises — so `submitted=False` can only mean
    the body will never run."""
    pool = claim_worker._CpuLanePool(width=2)
    pool.start()
    done = threading.Event()
    try:
        # Plain queue put: no thread is created here, so nothing to refuse.
        real_start = threading.Thread.start
        threading.Thread.start = lambda self: (_ for _ in ()).throw(
            RuntimeError("can't start new thread"))
        try:
            pool.submit(lambda: done.set())
        finally:
            threading.Thread.start = real_start
        assert done.wait(timeout=10), "submitted work never ran"
    finally:
        pool.shutdown(wait=True, timeout=5)


def test_concurrent_reservations_never_collide():
    """The reservation placeholder must be UNIQUE. Deriving it from the set's
    length let two concurrent reservers compute the same string, so ``set.add``
    silently no-op'd, the count under-read by one and an extra job was admitted
    past the ceiling — N+1, the KILL line."""
    w = claim_worker.ClaimWorker(queue="cpu", host="host-uniq", threads=8)
    # _reserve_slot returns (token, slots_held) and reserves ALL free capacity
    # pessimistically (migration 0024 — a job's width isn't known until the claim
    # returns), so filling the lane means shrinking each hold to its real weight,
    # exactly as _claim_threaded does. The TOKEN is what must be unique.
    seen = set()
    for i in range(8):
        held = w._reserve_slot()
        assert held is not None, f"lane refused reservation {i} before it was full"
        token, _free = held
        seen.add(token)
        w._set_slot_weight(token, f"job-{i}", None, 1)   # this one needs 1 slot
    assert len(seen) == 8, "reservation tokens collided"
    assert w.inflight_weight() == 8
    assert w._reserve_slot() is None           # ceiling reached
    assert w.run_once() is False               # ...so nothing is claimed


def test_shutdown_timeout_actually_bounds_the_wait():
    """``timeout`` must bound the call. Waiting for in-flight and then blocking
    again in ``pool.shutdown(wait=True)`` made the parameter decorative —
    ``cancel_futures`` only drops jobs that have not STARTED."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_bound", node.run)
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="b0", node_module="_thr_bound", queue="cpu",
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-bound", threads=2)
    try:
        w.run_once()
        _await_started(node, 1)
        started = time.monotonic()
        w.shutdown_threads(timeout=0.3)
        assert time.monotonic() - started < 2.0, "timeout did not bound the call"
    finally:
        node.gate.set()


def test_single_threaded_worker_gets_no_trip_wrapper():
    """At the default the watchdog keeps its stock ``os._exit`` — there are no
    siblings, so there is nothing to sweep and nothing to change."""
    w = claim_worker.ClaimWorker(queue="cpu", host="host-solo", threads=1)
    assert w._trip_exit("some-job") is None


# ── operator control / drain / teardown at N > 1 ────────────────────────────


def test_hard_stop_releases_every_inflight_job():
    """Operator OFF re-queues ALL N rows this worker holds (resume-style, no
    retry bump) — the existing host-wide sweep is correct here precisely because
    the whole lane is being turned off."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_off", node.run)
    ids = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"o{i}", node_module="_thr_off", queue="cpu",
        )
        for i in range(3)
    ]
    w = claim_worker.ClaimWorker(queue="cpu", host="host-off", threads=3)
    try:
        for _ in range(3):
            w.run_once()
        _await_started(node, 3)
        assert w.requeue_inflight_for_control() == 3
        for j in ids:
            row = node_queue.get_node_job(j)
            assert row["status"] == "queued"
            assert int(row["watchdog_retries"] or 0) == 0
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_drain_waits_for_inflight_before_parking(monkeypatch):
    """``stop_policy='drain'`` means "park, but keep the last job". Serial mode got
    that for free (``run_once`` didn't return until the body did); threaded mode
    returns the moment the job is SUBMITTED, so the park boundary must wait for
    in-flight to reach zero or a drain would park a worker with N jobs still
    running on it."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_drain", node.run)
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="d0", node_module="_thr_drain", queue="cpu",
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-drain", threads=2)
    parked: list[bool] = []
    monkeypatch.setattr(w, "_park_until_enabled", lambda: parked.append(True) or True)
    try:
        w.run_once()
        _await_started(node, 1)
        w.request_drain()

        done = threading.Event()
        threading.Thread(
            target=lambda: (w._drain_park_if_requested(), done.set()), daemon=True,
        ).start()
        # Still in flight ⇒ the park has not happened.
        assert not done.wait(timeout=0.5)
        assert parked == []

        node.gate.set()
        assert done.wait(timeout=15)
        assert parked == [True]
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_shutdown_waits_for_inflight_so_a_sigterm_drains_leases():
    """``run_claim_lane.sh`` forwards SIGTERM and relies on in-flight work
    finishing inside ``stop_grace_period``. A pool that dropped its jobs on
    shutdown would strand N rows until their leases lapsed."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_thr_term", node.run)
    ids = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"s{i}", node_module="_thr_term", queue="cpu",
        )
        for i in range(2)
    ]
    w = claim_worker.ClaimWorker(queue="cpu", host="host-term", threads=2)
    w.run_once()
    w.run_once()
    _await_started(node, 2)
    node.gate.set()
    w.shutdown_threads(timeout=15)
    assert all(_status(j) == "completed" for j in ids)
    assert w._cpu_pool is None


# ── the capacity gauge ──────────────────────────────────────────────────────


def test_threaded_cpu_worker_advertises_its_thread_count():
    """The panel sums ``worker_heartbeats.concurrency``; a 4-thread worker that
    advertised 1 would read ``CPU x/1`` while running four jobs. Display-only —
    the cpu claim is SKIP LOCKED and is never gated by this number."""
    w = claim_worker.ClaimWorker(queue="cpu", host="host-cap", threads=4)
    assert w.heartbeat._concurrency() == 4


def test_single_threaded_cpu_worker_still_advertises_one():
    w = claim_worker.ClaimWorker(queue="cpu", host="host-cap1", threads=1)
    assert w.heartbeat._concurrency() == 1
