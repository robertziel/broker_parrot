"""The POOL lane's no-progress guard — opt-in per node, policed by the node's beats.

WHY. A pool-lane job (an LLM / VLM facade: the node POSTs to the box's ollama or vLLM
and waits) had no bound at all. ``_run_pool_node`` arms no wall-clock ``Watchdog`` and
its ``LeaseRenewer`` renews for as long as the process lives, so lease reclaim cannot
fire either: one request that never answers held the lane — and the box's LLM slot —
for ever, with the panel showing a healthy running job.

The inline lane's ``StallWatchdog`` is the right SHAPE (inert until the first beat,
then a no-progress window), but two of its assumptions are wrong for this lane:

  * **Its trip is confirmed against THIS container's GPU/RAM.** A pool job's GPU work
    happens in the ollama / vLLM container, so the worker container reads idle and
    static whether the generation is streaming happily or wedged. The confirmation
    says nothing here — the beat IS the progress signal — so the pool guard trips on
    the beat gap alone (``confirm_idle=False``).
  * **"Declares ``status_callback``" is not an opt-in on this lane.** ComfyUI renders
    run on the pool lane and beat once per sampler step, which on a video model is
    minutes apart; arming a fixed window for every node that declares the parameter
    would kill healthy renders. So the opt-in is explicit and carries the node's own
    window: a module-level ``POOL_STALL_TIMEOUT_S``. A node without it runs exactly
    as before (``status_callback=None``, no watchdog).

Everything else is the engine's one trip policy (``_watchdog_trip``): re-queue under the
retry cap, fail past it, hard-exit so the stuck thread actually stops.
"""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

import queue_workflows
from queue_workflows import claim_worker, node_queue
from tests._helpers import make_run


@pytest.fixture(autouse=True)
def _node_pkg():
    queue_workflows.set_node_module_package("qwf_cw_nodes")
    yield


def _install_node(name: str, run_fn, **attrs) -> None:
    mod = types.ModuleType(f"qwf_cw_nodes.{name}")
    mod.run = run_fn
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[f"qwf_cw_nodes.{name}"] = mod


class _NoStatusWatcher:
    """The pool lane arms a JobStatusWatcher, whose answer to a row re-queued out from
    under it is ``os._exit(77)``. The trip tests below re-queue on purpose, so the
    watcher is inert here (it has its own suite) — otherwise pytest itself dies."""

    def __init__(self, **_kw):
        pass

    def start(self):
        pass

    def stop(self):
        pass


@pytest.fixture
def no_status_watcher(monkeypatch):
    monkeypatch.setattr(claim_worker, "JobStatusWatcher", _NoStatusWatcher)


def _claim_pool_job(node_module: str) -> dict:
    run_id = make_run(workflow_name="_pool_stall_test", out_dir=None)
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module=node_module, queue="gpu",
    )
    job = node_queue.claim_next_gpu_job(0, None, host="host-a", require_model=False)
    assert job is not None and job["node_module"] == node_module
    return job


# ── the watchdog itself: a trip without the physical confirmation ────────────


def test_without_physical_confirmation_the_beat_gap_alone_trips():
    """``confirm_idle=False``: silence past the window trips, and the GPU/RAM samplers
    are not consulted at all — on this lane they measure the wrong container."""
    job = _claim_pool_job("_ps_raw")
    sampled: list[str] = []

    # BUSY and MOVING — the physical confirmation, if it ran, would spare the job.
    # (Recorded rather than raised: the watchdog swallows sampler errors.)
    def gpu_busy():
        sampled.append("gpu")
        return 99

    def ram_moving():
        sampled.append("ram")
        return 1_000_000 + 10_000 * len(sampled)

    exits: list[int] = []
    wd = claim_worker.StallWatchdog(
        job_id=job["id"], stall_timeout_s=0.05, poll_s=0.01,
        on_exit=exits.append, confirm_idle=False,
        gpu_sampler=gpu_busy, ram_sampler=ram_moving,
    )
    wd.start()
    wd.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()

    assert sampled == [], "the pool guard must not consult the worker's GPU/RAM"
    assert exits == [76]
    row = node_queue.get_node_job(job["id"])
    assert row["status"] == "queued" and row["watchdog_retries"] == 1, (
        "same trip policy as the inline lane: re-queue under the cap, run stays alive")


def test_without_physical_confirmation_it_is_still_inert_until_the_first_beat():
    """No beat yet ⇒ nothing to police. On the pool lane the first beat is the first
    streamed token, so a cold server-side model load is never timed."""
    job = _claim_pool_job("_ps_inert")
    exits: list[int] = []
    wd = claim_worker.StallWatchdog(
        job_id=job["id"], stall_timeout_s=0.05, poll_s=0.01,
        on_exit=exits.append, confirm_idle=False,
    )
    wd.start()
    time.sleep(0.3)
    wd.stop()
    assert exits == []
    assert node_queue.get_node_job(job["id"])["status"] == "running"


# ── the opt-in ───────────────────────────────────────────────────────────────


def _cb_run(*, out=None, status_callback=None):
    return {}


def _plain_run(*, out=None, inputs=None):
    return {}


@pytest.mark.parametrize(("name", "run_fn", "attrs", "expected"), [
    ("_ps_opted", _cb_run, {"POOL_STALL_TIMEOUT_S": 300}, 300.0),
    ("_ps_float", _cb_run, {"POOL_STALL_TIMEOUT_S": 45.5}, 45.5),
    # declares the parameter but not the window: a ComfyUI-style render — untouched
    ("_ps_no_attr", _cb_run, {}, None),
    # declares a window but cannot beat: arming it would kill every job it runs
    ("_ps_no_param", _plain_run, {"POOL_STALL_TIMEOUT_S": 300}, None),
    ("_ps_zero", _cb_run, {"POOL_STALL_TIMEOUT_S": 0}, None),
    ("_ps_negative", _cb_run, {"POOL_STALL_TIMEOUT_S": -5}, None),
    ("_ps_junk", _cb_run, {"POOL_STALL_TIMEOUT_S": "soon"}, None),
    ("_ps_bool", _cb_run, {"POOL_STALL_TIMEOUT_S": True}, None),
])
def test_the_guard_is_opt_in_per_node_module(name, run_fn, attrs, expected):
    _install_node(name, run_fn, **attrs)
    assert claim_worker.ClaimWorker._pool_stall_timeout_s(name) == expected


def test_a_window_below_the_floor_is_raised_to_it():
    """A typo'd ``5`` would kill every healthy job between two requests; the floor is
    the smallest window that still spans a prompt prefill."""
    _install_node("_ps_tiny", _cb_run, POOL_STALL_TIMEOUT_S=5)
    assert claim_worker.ClaimWorker._pool_stall_timeout_s("_ps_tiny") == \
        claim_worker.POOL_STALL_MIN_TIMEOUT_S


def test_an_unresolvable_module_is_not_armed():
    assert claim_worker.ClaimWorker._pool_stall_timeout_s("_ps_does_not_exist") is None


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_the_operator_kill_switch_disarms_it_fleet_wide(monkeypatch, value):
    _install_node("_ps_killed", _cb_run, POOL_STALL_TIMEOUT_S=300)
    monkeypatch.setenv("QUEUE_WORKFLOWS_DISABLE_POOL_STALL_WATCHDOG", value)
    assert claim_worker.ClaimWorker._pool_stall_timeout_s("_ps_killed") is None


def test_a_falsy_kill_switch_value_does_not_disarm_it(monkeypatch):
    _install_node("_ps_not_killed", _cb_run, POOL_STALL_TIMEOUT_S=300)
    monkeypatch.setenv("QUEUE_WORKFLOWS_DISABLE_POOL_STALL_WATCHDOG", "0")
    assert claim_worker.ClaimWorker._pool_stall_timeout_s("_ps_not_killed") == 300.0


# ── the wiring in _run_pool_node ─────────────────────────────────────────────


def _spy_stall(monkeypatch, record: list[dict]):
    real = claim_worker.StallWatchdog

    def spy(**kw):
        record.append(dict(kw))
        return real(**kw)

    monkeypatch.setattr(claim_worker, "StallWatchdog", spy)


def test_an_opted_in_node_gets_the_beat_and_the_guard(monkeypatch):
    built: list[dict] = []
    _spy_stall(monkeypatch, built)
    seen: dict = {}

    def run(*, out=None, status_callback=None):
        seen["callable"] = callable(status_callback)
        if status_callback:
            status_callback()                  # one streamed chunk
        return {"context_delta": {"ok": True}}

    _install_node("_ps_wired", run, POOL_STALL_TIMEOUT_S=420)
    job = _claim_pool_job("_ps_wired")
    worker = claim_worker.ClaimWorker(queue="gpu", host="host-a")

    assert worker._run_pool_node(job) is True
    assert seen["callable"] is True, "an opted-in node must receive the beat"
    assert len(built) == 1
    assert built[0]["stall_timeout_s"] == 420.0
    assert built[0]["confirm_idle"] is False, (
        "the worker container's GPU/RAM says nothing about a server-side generation")
    assert node_queue.get_node_job(job["id"])["status"] == "completed"


def test_a_node_that_did_not_opt_in_runs_exactly_as_before(monkeypatch):
    """Declaring ``status_callback`` alone changes nothing on the pool lane — the
    ComfyUI render that beats once a sampler step must not be put on a clock."""
    built: list[dict] = []
    _spy_stall(monkeypatch, built)
    seen: dict = {}

    def run(*, out=None, status_callback=None):
        seen["status_callback"] = status_callback
        return {"context_delta": {}}

    _install_node("_ps_render_like", run)
    job = _claim_pool_job("_ps_render_like")
    assert claim_worker.ClaimWorker(queue="gpu", host="host-a")._run_pool_node(job) is True
    assert built == [], "no watchdog for a node that did not opt in"
    assert seen["status_callback"] is None


def test_an_opted_in_node_that_stops_making_progress_is_requeued(
        monkeypatch, no_status_watcher):
    """End to end: the node streams, then hangs. The guard trips on the gap, the row
    goes back to the queue (retry 1 of the cap), and the exit is requested — which in
    production ends the process and, with it, the stuck request."""
    monkeypatch.setattr(claim_worker, "POOL_STALL_MIN_TIMEOUT_S", 0.05)
    at_trip: dict = {}
    tripped = threading.Event()
    real = claim_worker.StallWatchdog

    def fast_stall(**kw):
        def on_exit(code):
            at_trip["code"] = code
            at_trip["row"] = node_queue.get_node_job(kw["job_id"])
            tripped.set()
        kw.update(on_exit=on_exit, poll_s=0.01)
        return real(**kw)

    monkeypatch.setattr(claim_worker, "StallWatchdog", fast_stall)

    def run(*, out=None, status_callback=None):
        status_callback()                      # the first token arrives…
        if not tripped.wait(timeout=5.0):      # …and then nothing, ever
            raise AssertionError("the guard never tripped")
        raise RuntimeError("the process would be gone by now")

    _install_node("_ps_hangs", run, POOL_STALL_TIMEOUT_S=0.05)
    job = _claim_pool_job("_ps_hangs")
    claim_worker.ClaimWorker(queue="gpu", host="host-a")._run_pool_node(job)

    assert at_trip["code"] == 76
    assert at_trip["row"]["status"] == "queued"
    assert at_trip["row"]["watchdog_retries"] == 1
    assert at_trip["row"]["claimed_by"] is None


def test_the_guard_is_stopped_when_the_job_ends(monkeypatch):
    stopped: list[bool] = []
    real = claim_worker.StallWatchdog

    def spy(**kw):
        wd = real(**kw)
        original = wd.stop

        def stop():
            stopped.append(True)
            original()
        wd.stop = stop
        return wd

    monkeypatch.setattr(claim_worker, "StallWatchdog", spy)

    def run(*, out=None, status_callback=None):
        raise RuntimeError("the node failed")

    _install_node("_ps_fails", run, POOL_STALL_TIMEOUT_S=300)
    job = _claim_pool_job("_ps_fails")
    claim_worker.ClaimWorker(queue="gpu", host="host-a")._run_pool_node(job)
    assert stopped == [True], "a failed job must not leave its guard running"
    assert node_queue.get_node_job(job["id"])["status"] == "failed"
