"""The GPU worker's idle signal that drives the LLM idle-unload reaper.

`_gpu_idle_seconds()` is 0 while a job is in flight (a running job IS GPU use, and
the call refreshes the stamp so a long job never looks idle), and grows from the last
activity once the box is quiet. A successful claim stamps activity so a short job that
starts and finishes between two reaper polls still resets the countdown.
"""

from __future__ import annotations

from queue_workflows import claim_worker, model_registry


class _Cache:
    current_model = None


def _gpu_worker():
    return claim_worker.ClaimWorker(queue="gpu", host="box-a-gpu", model_cache=_Cache())


def test_idle_seconds_zero_while_a_pool_job_runs():
    w = _gpu_worker()
    w._pool_inflight = 2
    assert w._gpu_active_count() == 2
    assert w._gpu_idle_seconds() == 0.0


def test_idle_seconds_zero_while_inline_runs():
    w = _gpu_worker()
    w._inline_running = True
    assert w._gpu_active_count() == 1
    assert w._gpu_idle_seconds() == 0.0


def test_idle_seconds_grows_when_quiet(monkeypatch):
    w = _gpu_worker()
    t = {"now": 1000.0}
    monkeypatch.setattr(claim_worker.time, "monotonic", lambda: t["now"])
    w._mark_gpu_active()          # stamp at t=1000
    t["now"] = 1000.0 + 420.0     # 7 min later, still quiet
    assert w._gpu_active_count() == 0
    assert w._gpu_idle_seconds() == 420.0


def test_running_job_refreshes_the_stamp(monkeypatch):
    # A long job must never look idle: while active, each idle read re-stamps, so
    # when it finishes the countdown starts from ~then, not from the claim.
    w = _gpu_worker()
    t = {"now": 1000.0}
    monkeypatch.setattr(claim_worker.time, "monotonic", lambda: t["now"])
    w._mark_gpu_active()
    w._pool_inflight = 1
    t["now"] = 1000.0 + 9999.0    # a 2h+ job
    assert w._gpu_idle_seconds() == 0.0     # active → 0 AND re-stamped to now
    w._pool_inflight = 0
    t["now"] += 30.0
    assert w._gpu_idle_seconds() == 30.0    # idle grows from the refresh, not the claim


def test_successful_pool_claim_stamps_activity(monkeypatch):
    w = _gpu_worker()
    w._llm_gpu_ok = True
    monkeypatch.setattr(model_registry, "known_ids", lambda: [])
    monkeypatch.setattr(claim_worker.node_queue, "claim_next_gpu_job",
                        lambda *a, **k: {"id": "j"})
    marks = []
    monkeypatch.setattr(w, "_mark_gpu_active", lambda: marks.append(1))
    assert w._claim_pool() == {"id": "j"}
    assert marks == [1]


def test_empty_pool_claim_does_not_stamp(monkeypatch):
    w = _gpu_worker()
    w._llm_gpu_ok = True
    monkeypatch.setattr(model_registry, "known_ids", lambda: [])
    monkeypatch.setattr(claim_worker.node_queue, "claim_next_gpu_job",
                        lambda *a, **k: None)
    marks = []
    monkeypatch.setattr(w, "_mark_gpu_active", lambda: marks.append(1))
    assert w._claim_pool() is None
    assert marks == []
