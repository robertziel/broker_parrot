"""The 1 Hz memory sentinel: reads MemAvailable every second, hands it to the worker's
shedding, never decides a start and never parks. See memory_sentinel.py for the why."""
from __future__ import annotations

import threading

import pytest

from queue_workflows import memory_sentinel

GiB = 2**30


class _Clock:
    """A virtual clock: sleep() advances time and records the requested intervals."""

    def __init__(self):
        self.t = 0.0
        self.slept: list[float] = []

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def _drive(sentinel, ticks):
    for _ in range(ticks):
        sentinel.tick()


def test_it_reads_every_second_and_hands_the_number_over():
    seen: list[int] = []
    clock = _Clock()
    s = memory_sentinel.MemorySentinel(read_fn=lambda: 7 * GiB, act_fn=seen.append,
                                       sleep_fn=clock.sleep, label="box-b-gpu")
    stop_after = {"n": 0}

    def act(v):
        seen.append(v)
        stop_after["n"] += 1
        if stop_after["n"] == 3:
            s.stop()
    s._act = act
    s.run()
    assert seen == [7 * GiB] * 3
    assert clock.slept == [1.0, 1.0, 1.0], "once a second, the designed cadence"
    assert memory_sentinel.INTERVAL_S == 1.0


def test_an_unreadable_meminfo_is_skipped_not_acted_on():
    seen: list = []
    s = memory_sentinel.MemorySentinel(read_fn=lambda: None, act_fn=seen.append)
    _drive(s, 3)
    assert seen == [] and s.ticks == 3 and s.failures == 0


def test_a_read_that_raises_does_not_stop_the_sentinel():
    seen: list = []
    reads = iter([Exception("meminfo gone"), 5 * GiB])

    def read():
        v = next(reads)
        if isinstance(v, Exception):
            raise v
        return v
    s = memory_sentinel.MemorySentinel(read_fn=read, act_fn=seen.append)
    _drive(s, 2)
    assert seen == [5 * GiB] and s.failures == 1


def test_shedding_that_raises_does_not_stop_the_sentinel(caplog):
    seen: list = []

    def act(v):
        seen.append(v)
        if len(seen) == 1:
            raise RuntimeError("docker socket hiccup")
    s = memory_sentinel.MemorySentinel(read_fn=lambda: 1 * GiB, act_fn=act, label="box-c-gpu")
    with caplog.at_level("ERROR", logger="queue_workflows.memory_sentinel"):
        _drive(s, 2)
    assert seen == [1 * GiB, 1 * GiB] and s.failures == 1
    assert any("keeps watching" in r.getMessage() for r in caplog.records)


def test_the_thread_is_a_daemon_that_stops_on_request():
    gate = threading.Event()
    s = memory_sentinel.MemorySentinel(read_fn=lambda: 50 * GiB, act_fn=lambda v: gate.set(),
                                       interval_s=0.05)
    s.start()
    assert gate.wait(2.0), "the thread ticked"
    assert s.running and s._thread.daemon
    s.stop()
    s._thread.join(2.0)
    assert not s.running
    assert s.start() is s, "start() after stop() is a no-op that returns the sentinel"


def test_the_kill_floor_is_half_a_gigabyte_and_a_knob(monkeypatch):
    assert memory_sentinel.kill_floor_bytes() == 512 * 1024 * 1024
    monkeypatch.setenv(memory_sentinel.KILL_FLOOR_ENV, "2048")
    assert memory_sentinel.kill_floor_bytes() == 2048 * 1024 * 1024
    monkeypatch.setenv(memory_sentinel.KILL_FLOOR_ENV, "junk")
    assert memory_sentinel.kill_floor_bytes() == 512 * 1024 * 1024
    monkeypatch.setenv(memory_sentinel.KILL_FLOOR_ENV, "-5")
    assert memory_sentinel.kill_floor_bytes() == 0


def test_the_interval_has_a_floor_so_a_misconfiguration_cannot_spin():
    s = memory_sentinel.MemorySentinel(read_fn=lambda: 1, act_fn=lambda v: None, interval_s=0)
    assert s._interval == 0.05


# ── wired into the gpu lane ──────────────────────────────────────────────────


def test_the_gpu_worker_arms_the_sentinel_with_its_own_shedding(monkeypatch):
    from queue_workflows import claim_worker, llm_context

    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
    w = _W()
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 99 * GiB)
    started: list = []
    monkeypatch.setattr(memory_sentinel.MemorySentinel, "start",
                        lambda self: started.append(self) or self)
    w._arm_memory_sentinel()
    (s,) = started
    assert s._read is llm_context.mem_available_bytes
    assert s._act == w._memory_tick


def test_the_kill_switch_arms_nothing(monkeypatch):
    from queue_workflows import claim_worker

    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
    monkeypatch.setenv(memory_sentinel.DISABLE_ENV, "1")
    monkeypatch.setattr(memory_sentinel.MemorySentinel, "start",
                        lambda self: pytest.fail("armed despite the kill switch"))
    _W()._arm_memory_sentinel()


def test_arming_runs_at_startup_for_the_gpu_lane():
    import inspect

    from queue_workflows import claim_worker

    src = inspect.getsource(claim_worker.ClaimWorker.run_forever)
    assert "_arm_memory_sentinel()" in src
    assert src.index("_arm_llm_idle_reaper()") < src.index("_arm_memory_sentinel()")


def test_the_sentinel_module_never_touches_the_control_plane():
    """It reads one number and calls one action. No parking, no starts, no DB."""
    import inspect

    src = inspect.getsource(memory_sentinel)
    for forbidden in ("worker_control", "disable_worker", "node_queue", "set_context",
                      "vllm_start_fn"):
        assert forbidden not in src, forbidden
