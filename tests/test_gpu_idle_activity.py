"""The GPU worker's in-flight signal that guards the LLM idle-unload reaper.

`_gpu_active_count()` counts jobs currently executing (pool in-flight + the
inline lane); the reaper never unloads while it is > 0, so a claimed job's LLM
call can't lose its model mid-run.

HISTORY: this module used to test a claim-stamped idle CLOCK
(`_mark_gpu_active` / `_gpu_idle_seconds`) — removed when the idle
signal moved into the reaper itself, which now observes actual server-side use
(ollama ``/api/ps`` ``expires_at`` advances; see test_llm_idle_unload.py). The
claim clock was wrong both ways: memoized claims reset it without touching the
model, and non-queue clients' use was invisible to it.
"""

from __future__ import annotations

from queue_workflows import claim_worker


class _Cache:
    current_model = None


def _gpu_worker():
    return claim_worker.ClaimWorker(queue="gpu", host="box-a-gpu", model_cache=_Cache())


def test_active_count_counts_pool_inflight():
    w = _gpu_worker()
    w._pool_inflight = 2
    assert w._gpu_active_count() == 2


def test_active_count_counts_the_inline_lane():
    w = _gpu_worker()
    w._inline_running = True
    assert w._gpu_active_count() == 1


def test_active_count_zero_when_quiet():
    assert _gpu_worker()._gpu_active_count() == 0
