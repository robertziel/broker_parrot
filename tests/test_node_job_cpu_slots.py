"""A node-job can reserve N slots of a widened cpu lane (``cpu_slots``).

``cpu_worker_threads`` says how many jobs ONE worker process runs at once. It says
nothing about how much of that worker a single job needs — every job counted as
exactly 1. So a node whose body legitimately spawns its own 5 threads (an
ffmpeg fan-out, a thread-pooled HTTP sweep) landed 5-to-a-process on a 5-wide
lane: 25 threads fighting over one interpreter, and no way to say otherwise.

``cpu_slots`` is that missing weight, stored per node-job (migration 0024, NULL =
1) and enforced in the CLAIM — so the reservation exists *before* the body starts,
not as an afterthought. Contracts pinned here:

  * a ``cpu_slots=N`` job occupies N of the worker's slots; the lane admits
    others only up to the remainder;
  * a job wider than the whole lane runs ALONE rather than starving forever
    (the weight is clamped to the lane width for the fit test);
  * a serial worker (``threads=1``, the default) is
    completely unaffected — it runs one job at a time, so exclusivity is already
    total and every ``cpu_slots`` value is claimable;
  * the weight is released on every path, exactly like a plain slot.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

import queue_workflows
from queue_workflows import claim_worker, node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


@pytest.fixture(autouse=True)
def _node_pkg():
    queue_workflows.set_node_module_package("qwf_slot_nodes")
    yield


@pytest.fixture(autouse=True)
def _no_abandon_watcher(monkeypatch):
    """Same rationale as the threaded-lane module: these tests reassign rows on
    purpose, and the real JobStatusWatcher answers that with ``os._exit(77)``,
    which kills the pytest process rather than failing a test (invisible on
    sqlite, fatal on pg)."""
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
    return make_run(status="running", workflow_name="_cpu_slots_test", out_dir=None)


def _install_fake_node(name: str, run_fn):
    mod = types.ModuleType(f"qwf_slot_nodes.{name}")
    mod.run = run_fn
    sys.modules[f"qwf_slot_nodes.{name}"] = mod


def _status(job_id: str) -> str:
    return node_queue.get_node_job(job_id)["status"]


def _slots(job_id: str):
    with connection() as c, c.cursor() as cur:
        cur.execute("SELECT cpu_slots FROM workflow_node_jobs WHERE id=%s", (job_id,))
        return cur.fetchone()["cpu_slots"]


class _GatedNode:
    def __init__(self) -> None:
        self.gate = threading.Event()
        self.started = threading.Semaphore(0)

    def run(self, *, out=None, inputs=None, cancel_event=None):
        self.started.release()
        self.gate.wait(timeout=30)
        return {"context_delta": {"ok": True}}


def _await_started(node: _GatedNode, n: int = 1) -> None:
    for _ in range(n):
        assert node.started.acquire(timeout=15), "job never started"


# ── the column ──────────────────────────────────────────────────────────────


def test_default_is_null_meaning_one_slot():
    """Every existing row and every unweighted enqueue stays NULL — the column is
    additive, so a pre-0024 workload is byte-identical."""
    run_id = _make_run()
    jid = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="m", queue="cpu",
    )
    assert _slots(jid) is None


def test_enqueue_stores_the_weight():
    run_id = _make_run()
    jid = node_queue.enqueue_node_job(
        run_id=run_id, node_id="b", node_module="m", queue="cpu", cpu_slots=5,
    )
    assert _slots(jid) == 5


def test_a_weight_of_one_is_stored_as_null():
    """1 IS the default, so it must not create a distinct 'weighted' shape —
    otherwise two spellings of the same thing drift apart."""
    run_id = _make_run()
    jid = node_queue.enqueue_node_job(
        run_id=run_id, node_id="c", node_module="m", queue="cpu", cpu_slots=1,
    )
    assert _slots(jid) is None


@pytest.mark.parametrize("bad", [0, -3])
def test_a_sub_one_weight_is_rejected_before_the_write(bad):
    """A zero/negative weight would make the fit test meaningless (and a 0-weight
    job would let the lane admit infinitely many). Fail loudly at the API edge,
    the same discipline as the queue-name and stray-model guards."""
    run_id = _make_run()
    with pytest.raises(ValueError):
        node_queue.enqueue_node_job(
            run_id=run_id, node_id="d", node_module="m", queue="cpu", cpu_slots=bad,
        )


def test_the_dispatcher_carries_cpu_slots_from_the_node_spec():
    """A workflow author declares it once in the DAG — like avoid_box — and the
    dispatcher stamps it on the fanned-out row."""
    from queue_workflows import dispatcher
    node = {"id": "n1", "module": "m", "queue": "cpu", "cpu_slots": 4}
    assert dispatcher._node_cpu_slots(node) == 4
    assert dispatcher._node_cpu_slots({"id": "n2", "module": "m"}) is None


# ── the claim fit test ──────────────────────────────────────────────────────


def test_a_serial_worker_ignores_cpu_slots_entirely():
    """THE compatibility contract. A threads=1 worker runs one job at a time, so
    it already gives that job the whole process — a cpu_slots=8 job must run
    there unchanged. Most deployments run threads=1; if this breaks, weighted
    rows become permanently unclaimable in production."""
    run_id = _make_run()
    ran = {}

    def run(*, out=None, inputs=None, cancel_event=None):
        ran["did"] = True
        return {"context_delta": {}}

    _install_fake_node("_slot_serial", run)
    jid = node_queue.enqueue_node_job(
        run_id=run_id, node_id="s", node_module="_slot_serial", queue="cpu",
        cpu_slots=8,
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-serial", threads=1)
    assert w.run_once() is True
    assert ran.get("did") is True
    assert _status(jid) == "completed"


def test_a_fat_job_holds_its_whole_weight_and_the_lane_admits_only_the_remainder():
    """cpu_slots=3 on a 4-wide lane: the fat job takes 3, exactly ONE 1-slot job
    fits beside it, and the next is refused."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_slot_fat", node.run)
    fat = node_queue.enqueue_node_job(
        run_id=run_id, node_id="fat", node_module="_slot_fat", queue="cpu",
        cpu_slots=3, priority=1,          # claimed first
    )
    thin = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"t{i}", node_module="_slot_fat", queue="cpu",
            priority=50,
        )
        for i in range(2)
    ]
    w = claim_worker.ClaimWorker(queue="cpu", host="host-fat", threads=4)
    try:
        assert w.run_once() is True            # the fat one
        _await_started(node)
        assert w.inflight_job_ids() == {fat}
        assert w.inflight_weight() == 3

        assert w.run_once() is True            # one thin job fits (3+1 = 4)
        _await_started(node)
        assert w.inflight_weight() == 4

        assert w.run_once() is False           # lane full — the 2nd thin waits
        running = [j for j in [fat, *thin] if _status(j) == "running"]
        assert len(running) == 2
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_a_job_wider_than_the_lane_runs_alone_instead_of_starving():
    """cpu_slots=9 on a 3-wide lane can never 'fit' arithmetically. Excluding it
    would strand it forever — no worker in the fleet is ever wider. So the weight
    is CLAMPED to the lane width for the fit test: it claims when the lane is
    idle and occupies the whole worker."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_slot_huge", node.run)
    huge = node_queue.enqueue_node_job(
        run_id=run_id, node_id="h", node_module="_slot_huge", queue="cpu",
        cpu_slots=9, priority=1,
    )
    other = node_queue.enqueue_node_job(
        run_id=run_id, node_id="o", node_module="_slot_huge", queue="cpu",
        priority=50,
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-huge", threads=3)
    try:
        assert w.run_once() is True
        _await_started(node)
        assert _status(huge) == "running"
        assert w.inflight_weight() == 3         # clamped to the lane width
        assert w.run_once() is False            # it owns the whole worker
        assert _status(other) == "queued"
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_a_fat_job_waits_rather_than_squeezing_in():
    """The reservation must bind BEFORE the body starts: with 1 slot free, a
    3-slot job is not claimed at all (it stays queued for a worker that can hold
    it), instead of starting oversubscribed."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_slot_wait", node.run)
    thin = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"w{i}", node_module="_slot_wait", queue="cpu",
            priority=1,
        )
        for i in range(2)
    ]
    fat = node_queue.enqueue_node_job(
        run_id=run_id, node_id="wf", node_module="_slot_wait", queue="cpu",
        cpu_slots=3, priority=50,
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-wait", threads=3)
    try:
        assert w.run_once() is True and w.run_once() is True   # both thin
        _await_started(node, 2)
        assert w.inflight_weight() == 2
        # 1 slot free, the only queued row needs 3 ⇒ nothing claimable.
        assert w.run_once() is False
        assert _status(fat) == "queued"
        assert all(_status(t) == "running" for t in thin)
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_the_weight_comes_back_when_the_job_finishes():
    """A released fat job frees ALL of its slots, not one."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_slot_rel", node.run)
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="r", node_module="_slot_rel", queue="cpu",
        cpu_slots=3,
    )
    w = claim_worker.ClaimWorker(queue="cpu", host="host-rel", threads=4)
    try:
        assert w.run_once() is True
        _await_started(node)
        assert w.inflight_weight() == 3
        node.gate.set()
        assert w._wait_for_inflight(timeout=15) is True
        assert w.inflight_weight() == 0
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_an_out_of_band_negative_weight_cannot_break_the_ceiling():
    """The weights are SUMMED, so the ceiling is only a ceiling while every
    weight is >= 1. ``COALESCE(cpu_slots, 1)`` maps NULL/0 to 1 but a NEGATIVE
    would pass straight through — satisfying the fit test unconditionally AND
    dragging the running total down, so the lane admits far more claims than it
    has pool threads. Rows claimed past that point sit ``running`` with no
    LeaseRenewer yet (it starts inside ``_run_node``), so their leases lapse,
    reclaim hands them to a peer, and the job double-runs.

    Unreachable through the API (``enqueue_node_job`` raises on < 1) — this is
    the raw-SQL / future-writer path, floored on BOTH sides (worker and SQL) so
    the two can never disagree."""
    run_id = _make_run()
    node = _GatedNode()
    _install_fake_node("_slot_neg", node.run)
    ids = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"n{i}", node_module="_slot_neg", queue="cpu",
        )
        for i in range(6)
    ]
    # Out-of-band write, bypassing the API guard entirely.
    with connection() as c, c.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET cpu_slots = -5 WHERE id = %s",
                    (ids[0],))

    w = claim_worker.ClaimWorker(queue="cpu", host="host-neg", threads=2)
    try:
        claimed = 0
        for _ in range(6):
            if w.run_once():
                claimed += 1
        assert claimed <= 2, f"lane admitted {claimed} claims on a 2-wide lane"
        assert w.inflight_weight() >= 1
        running = [j for j in ids if _status(j) == "running"]
        assert len(running) <= 2, f"{len(running)} rows running on a 2-wide lane"
    finally:
        node.gate.set()
        w.shutdown_threads(timeout=15)


def test_a_claim_that_finds_nothing_releases_the_provisional_weight():
    """The claim reserves the free capacity BEFORE it knows the row's weight (it
    cannot know sooner). An empty queue must give all of it back, or the lane
    silently narrows to nothing."""
    w = claim_worker.ClaimWorker(queue="cpu", host="host-prov", threads=4)
    assert w.run_once() is False
    assert w.inflight_weight() == 0
    assert w._wait_for_inflight(timeout=1) is True
