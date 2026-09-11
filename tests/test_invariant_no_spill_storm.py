"""INVARIANT: a box denied by its own lease must back off, not hot-loop.

Observed live, on both GPU boxes, twice within seven seconds of a lane restart:

    71 `SPILLING job <id>` lines in 6.0 s   (box-a)
    79 `SPILLING job <id>` lines in 6.5 s   (box-b)

— all for ONE job id. The mechanism is a loop with no brake. The previous worker
process died mid-switch, so the box-lease file still carried its `__server_switch__`
holder (TTL 120 s, no release on exit). The fresh worker claimed a job, was denied the
box by that dead holder, and `requeue_job_for_retry` put the row back at the FRONT of
the queue (`LEAST(priority, 10)` — deliberately, it is a resume, not a fault). The very
next claim returned the same row, which was denied again. Twelve DB round-trips a second
until the dead holder's TTL expired.

The same shape was recorded before, in an earlier fill-before-spill incident: "68
spills / 5 min — a hot claim->spill loop". It was diagnosed as a placement problem and
the loop itself was left in.

Spilling is correct: the job belongs on a box that can run it. Spilling the same job
twelve times a second is not — it is a denial-of-service against the queue, it burns the
row's history, and it hides the reason in a wall of identical lines. So a denied acquire
now pauses THIS worker's GPU claiming for a short, doubling, capped interval, and the
reason is logged once per window. A worker that is never denied is unaffected.
"""
from __future__ import annotations

import logging

import pytest

from queue_workflows import claim_worker


class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


class _Lease:
    """A box lease that grants or denies on command, like the flock file would."""

    def __init__(self, held=None):
        self.held = held
        self.acquired = []

    def acquire(self, model_id):
        if self.held is not None and self.held != model_id:
            return False
        self.acquired.append(model_id)
        return True

    def current(self):
        class _S:
            pass
        s = _S()
        s.model = self.held
        return s

    def renew(self, *a, **k):
        return True

    def release(self):
        self.held = None


def _worker(lease, clock, requeued):
    """A bare gpu ClaimWorker with only the box-lease collaborators wired — no DB, no
    threads (``__init__`` is deliberately skipped, the idiom this module already uses
    for claim-path tests)."""
    w = claim_worker.ClaimWorker.__new__(claim_worker.ClaimWorker)
    w.queue = "gpu"
    w.host = "box-a-gpu"
    w._box_lease = lease
    w._box_held_model = None
    w._box_active_jobs = 0
    w._spill_pause_until = 0.0
    w._spill_streak = 0
    w._spill_now = clock
    w._get_box_lease = lambda: lease
    w._effective_gpu_model = lambda job: "__llm_server__"
    w._release_llm_ram_unless_llm_owns_the_card = lambda model: None
    return w


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    """`requeue_job_for_retry` is the only DB call on this path."""
    calls = []
    monkeypatch.setattr(claim_worker.node_queue, "requeue_job_for_retry",
                        lambda job_id, *a, **k: calls.append(job_id))
    return calls


def _job(jid="job-1"):
    return {"id": jid, "node_id": "report/compose_report"}


# ── the brake ────────────────────────────────────────────────────────────────


def test_a_denied_acquire_spills_the_job_exactly_once():
    clock = _Clock()
    w = _worker(_Lease(held="__server_switch__"), clock, [])

    assert w._acquire_box_or_spill(_job()) is False
    assert w._spill_pause_until > clock()


def test_a_spill_PAUSES_this_workers_gpu_claiming():
    """THE LOOP. Without this the next claim returns the same front-of-queue row."""
    clock = _Clock()
    w = _worker(_Lease(held="__server_switch__"), clock, [])
    w._acquire_box_or_spill(_job())

    assert w._spill_paused() is True

    clock.advance(claim_worker.SPILL_PAUSE_S + 0.01)

    assert w._spill_paused() is False


def test_consecutive_spills_back_off_and_stay_capped():
    clock = _Clock()
    w = _worker(_Lease(held="__server_switch__"), clock, [])
    seen = []
    for _ in range(8):
        w._acquire_box_or_spill(_job())
        seen.append(round(w._spill_pause_until - clock(), 3))
        clock.advance(seen[-1] + 0.001)

    assert seen[0] == pytest.approx(claim_worker.SPILL_PAUSE_S)
    assert seen[1] > seen[0], "the backoff must grow while the denial persists"
    assert max(seen) <= claim_worker.SPILL_PAUSE_MAX_S
    assert seen[-1] == pytest.approx(claim_worker.SPILL_PAUSE_MAX_S)


def test_a_granted_acquire_RESETS_the_backoff():
    """A box that spilled once and then got the card must not stay slowed down."""
    clock = _Clock()
    lease = _Lease(held="__server_switch__")
    w = _worker(lease, clock, [])
    for _ in range(4):
        w._acquire_box_or_spill(_job())
        clock.advance(60)

    lease.held = None
    assert w._acquire_box_or_spill(_job()) is True

    assert w._spill_streak == 0
    assert w._spill_paused() is False


def test_the_pause_is_bounded_so_a_freed_box_resumes_promptly():
    """The dead holder that caused this expires on its own TTL; the brake must not
    outlast it by much or a healthy box sits idle."""
    from queue_workflows import gpu_model_lease

    assert claim_worker.SPILL_PAUSE_MAX_S <= gpu_model_lease.DEFAULT_TTL_S


# ── the claim gates honour it ────────────────────────────────────────────────


def test_the_inline_claim_gate_refuses_while_paused(monkeypatch):
    clock = _Clock()
    w = _worker(_Lease(held="__server_switch__"), clock, [])
    w._switch = type("S", (), {"blocks_claims": staticmethod(lambda: False)})()
    called = []
    monkeypatch.setattr(claim_worker.node_queue, "claim_next_gpu_job",
                        lambda *a, **k: called.append(1))

    w._acquire_box_or_spill(_job())

    assert w._claim() is None
    assert called == [], "a paused worker must not touch the claim SQL at all"


def test_the_pool_claim_gate_refuses_while_paused(monkeypatch):
    clock = _Clock()
    w = _worker(_Lease(held="__server_switch__"), clock, [])
    w._switch = type("S", (), {"blocks_claims": staticmethod(lambda: False)})()
    w._pool_gpu_serving_ok = lambda: True
    called = []
    monkeypatch.setattr(claim_worker.node_queue, "claim_next_gpu_job",
                        lambda *a, **k: called.append(1))

    w._acquire_box_or_spill(_job())

    assert w._claim_pool() is None
    assert called == []


# ── the wall of identical lines ──────────────────────────────────────────────


def test_the_spill_reason_is_logged_once_per_window_not_per_attempt(caplog):
    clock = _Clock()
    w = _worker(_Lease(held="__server_switch__"), clock, [])

    with caplog.at_level(logging.WARNING, logger="queue_workflows.claim_worker"):
        for _ in range(30):
            w._acquire_box_or_spill(_job())
            clock.advance(0.1)

    said = [r for r in caplog.records if "SPILLING" in r.getMessage()]
    assert len(said) == 1, f"expected one line per window, got {len(said)}"


def test_a_DIFFERENT_job_being_spilled_is_always_reported(caplog):
    clock = _Clock()
    w = _worker(_Lease(held="__server_switch__"), clock, [])

    with caplog.at_level(logging.WARNING, logger="queue_workflows.claim_worker"):
        w._acquire_box_or_spill(_job("job-1"))
        clock.advance(0.1)
        w._acquire_box_or_spill(_job("job-2"))

    said = [r.getMessage() for r in caplog.records if "SPILLING" in r.getMessage()]
    assert len(said) == 2
    assert "job-2" in said[1]


def test_every_spilled_job_is_still_requeued_even_when_the_line_is_suppressed(_no_db):
    """Rate-limiting the LOG must never rate-limit the WORK: each spilled job has to
    reach the queue again or it is lost."""
    clock = _Clock()
    w = _worker(_Lease(held="__server_switch__"), clock, [])
    for i in range(5):
        w._acquire_box_or_spill(_job(f"job-{i}"))
        clock.advance(0.1)

    assert _no_db == ["job-0", "job-1", "job-2", "job-3", "job-4"]
