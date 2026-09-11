"""INVARIANT: a switch may never hold a box out of service for a start that cannot work.

The outage: two GPU boxes sat at a ~50 % duty cycle for 3 h 08 m with 26 jobs
queued. Each had requested a switch to vLLM; the host's start lever answered
``HTTP 404: No such container`` on one box and an OCI mount error on the other — every
30 s, for the whole 1800 s warm window — and the machine read that as "not ready YET",
because a FAILED START was not one of the three observations ``tick`` accepts. Then the
demand starter re-requested after its cooldown, and an operator restart re-entered
DRAINING with FRESH phase clocks, extending the outage instead of ending it.

Three guarantees are pinned here, and they are the reason this file is an invariant and
not a topic module:

  * **A definitive failure ends the window on the SAME tick.** Not at the warm timeout.
  * **A kind that cannot start is QUARANTINED**, and the quarantine survives the restart
    that re-reads the record — otherwise the demand starter simply asks again.
  * **The budget belongs to the REQUEST, not to the process.** A restart inherits the
    remaining time; it can never re-arm a full window.

The corollary invariant, fuzzed at the bottom: ``blocks_claims()`` is true only while
there is something to protect — work in flight, an outgoing server still loaded, or a
start actually in flight.

Everything runs on a virtual clock: no sleeping, no network, no DB.
"""
from __future__ import annotations

import json

import pytest

from queue_workflows import server_switch as sw


class _Clock:
    def __init__(self, t=1000.0): self.t = float(t)
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def _m(**kw):
    """A switch with explicit budgets, a virtual monotonic clock AND a virtual wall
    clock (the record is wall-stamped so it survives a restart)."""
    kw.setdefault("drain_timeout_s", 60.0)
    kw.setdefault("unload_timeout_s", 60.0)
    kw.setdefault("warm_timeout_s", 600.0)
    clock = kw.pop("clock", None) or _Clock()
    wall = kw.pop("wall", None) or _Clock(1_700_000_000.0)
    m = sw.ServerSwitch(now_fn=clock, wall_clock_fn=wall, **kw)
    return m, clock, wall


def _reach_warming(m):
    """Drive a fresh switch to WARMING and return the actions of the last tick."""
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    return m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)


# ── the outcome value object ─────────────────────────────────────────────────


def test_a_started_outcome_is_acknowledged_and_a_permanent_one_is_not():
    assert sw.StartOutcome.started().acknowledged is True
    assert sw.StartOutcome.already_running().acknowledged is True
    assert sw.StartOutcome.transient("connection reset").acknowledged is False
    assert sw.StartOutcome.permanent("404").acknowledged is False


@pytest.mark.parametrize("value,status", [
    (None, sw.UNKNOWN),                                   # legacy hook: returns nothing
    (True, sw.STARTED),
    (False, sw.TRANSIENT),
    ({"status": "permanent", "reason": "no such container"}, sw.PERMANENT),
    ({"status": "already_running"}, sw.ALREADY_RUNNING),
    ({"status": "nonsense-from-a-host"}, sw.UNKNOWN),
    ("garbage", sw.UNKNOWN),
])
def test_coerce_accepts_every_shape_a_host_lever_may_return(value, status):
    """The hook contract must stay byte-compatible: a host that returns ``None`` (every
    host today) keeps today's behaviour, and no host return may ever raise into the
    switch."""
    assert sw.StartOutcome.coerce(value).status == status


def test_coerce_passes_a_real_outcome_through_unchanged():
    o = sw.StartOutcome.permanent("boom")
    assert sw.StartOutcome.coerce(o) is o


# ── (a) a definitive failure ends the window on the SAME tick ────────────────


def test_a_PERMANENT_start_failure_unblocks_claims_on_the_same_tick():
    """THE OUTAGE. Before: 1800 s of WARMING against a 404. After: the box is back in
    service the instant the lever says the start can never work."""
    m, _, _ = _m()
    assert _reach_warming(m).count(sw.START_INCOMING) == 1
    assert m.blocks_claims() is True

    actions = m.report_start_outcome(
        sw.StartOutcome.permanent("HTTP 404: No such container: vllm"))

    assert sw.ABORT in actions
    assert m.state == sw.IDLE
    assert m.blocks_claims() is False


def test_a_PERMANENT_failure_quarantines_the_kind_so_the_demand_starter_cannot_re_ask():
    m, _, _ = _m(quarantine_s=900.0)
    _reach_warming(m)
    m.report_start_outcome(sw.StartOutcome.permanent("no such container"))

    assert m.quarantined("vllm") is True
    assert m.request("vllm") is False          # the re-request loop is what looped
    assert m.state == sw.IDLE
    assert m.request("ollama") is True         # a DIFFERENT kind is unaffected


def test_the_quarantine_expires_so_a_repaired_container_is_tried_again():
    m, clock, wall = _m(quarantine_s=900.0)
    _reach_warming(m)
    m.report_start_outcome(sw.StartOutcome.permanent("404"))

    wall.advance(899); clock.advance(899)
    assert m.request("vllm") is False
    wall.advance(2); clock.advance(2)
    assert m.quarantined("vllm") is False
    assert m.request("vllm") is True


def test_repeated_failures_back_off_but_stay_capped():
    """The count must SURVIVE the expiry of the hold it caused — otherwise a
    permanently-broken target is retried at a constant rate for ever."""
    m, clock, wall = _m(quarantine_s=100.0, quarantine_max_s=400.0)
    for expected in (100.0, 200.0, 400.0, 400.0):
        step = m.quarantine_remaining_s("vllm") + 1      # just past the current hold
        wall.advance(step); clock.advance(step)
        assert m.request("vllm") is True
        _drive_to_warming_after_request(m)
        m.report_start_outcome(sw.StartOutcome.permanent("404"))
        assert m.quarantine_remaining_s("vllm") == pytest.approx(expected, abs=1.0)


def _drive_to_warming_after_request(m):
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)


def test_observing_the_kind_SERVING_clears_the_quarantine_at_once():
    """An operator who fixes and starts the container by hand must not wait out a
    4 h quarantine: the box observes it serving and the hold is gone."""
    m, _, _ = _m()
    _reach_warming(m)
    m.report_start_outcome(sw.StartOutcome.permanent("404"))
    assert m.quarantined("vllm") is True

    m.clear_quarantine("vllm")

    assert m.quarantined("vllm") is False
    assert m.request("vllm") is True


# ── (c) the quarantine survives the restart that re-reads the record ─────────


def test_the_quarantine_SURVIVES_a_process_restart(tmp_path):
    """Without this the loop simply resumes: the file is deleted on abort, the fresh
    process knows nothing, the demand starter re-requests, 30 more minutes are lost."""
    path = str(tmp_path / "switch.json")
    wall = _Clock(1_700_000_000.0)
    m1, _, _ = _m(persist_path=path, wall=wall, quarantine_s=900.0)
    _reach_warming(m1)
    m1.report_start_outcome(sw.StartOutcome.permanent("no such container"))

    m2 = sw.ServerSwitch(label="x", persist_path=path, now_fn=_Clock(),
                         wall_clock_fn=wall)

    assert m2.state == sw.IDLE                 # the dead switch is NOT resumed
    assert m2.quarantined("vllm") is True
    assert m2.request("vllm") is False


def test_a_long_forgotten_quarantine_leaves_no_record_behind(tmp_path):
    """The hold expires quickly, but the FAILURE COUNT is kept so the backoff can
    escalate. Only after a full cap-length with no retry is the record forgotten and
    the file removed."""
    import os
    path = str(tmp_path / "switch.json")
    wall = _Clock(1_700_000_000.0)
    m1, _, _ = _m(persist_path=path, wall=wall, quarantine_s=10.0,
                  quarantine_max_s=100.0)
    _reach_warming(m1)
    m1.report_start_outcome(sw.StartOutcome.permanent("404"))
    assert os.path.exists(path)

    wall.advance(11)                       # hold over, count remembered
    m2 = sw.ServerSwitch(label="x", persist_path=path, now_fn=_Clock(),
                         wall_clock_fn=wall, quarantine_s=10.0,
                         quarantine_max_s=100.0)
    assert m2.quarantined("vllm") is False
    assert os.path.exists(path)

    wall.advance(200)                      # a full cap-length with no retry
    sw.ServerSwitch(label="x", persist_path=path, now_fn=_Clock(),
                    wall_clock_fn=wall, quarantine_s=10.0, quarantine_max_s=100.0)

    assert not os.path.exists(path)


# ── (b) the budget belongs to the REQUEST, not to the process ────────────────


def test_a_restart_mid_switch_INHERITS_the_remaining_budget_and_never_re_arms(tmp_path):
    """Observed live: a switch was still blocking claims more than three hours after it
    was requested, because every restart re-entered DRAINING with a fresh clock. The
    deadline is now computed from the ORIGINAL request."""
    path = str(tmp_path / "switch.json")
    wall = _Clock(1_700_000_000.0)
    m1, _, _ = _m(persist_path=path, wall=wall)
    m1.request("vllm")
    total = m1.total_budget_s()

    wall.advance(total - 30)                   # 30 s of the original budget left
    m2 = sw.ServerSwitch(label="x", persist_path=path, now_fn=_Clock(),
                         wall_clock_fn=wall, drain_timeout_s=60.0,
                         unload_timeout_s=60.0, warm_timeout_s=600.0)

    assert m2.state == sw.DRAINING
    assert m2.remaining_budget_s() == pytest.approx(30.0, abs=1.0)


def test_the_ORIGINAL_deadline_aborts_the_switch_even_while_phases_keep_resetting():
    m, clock, wall = _m()
    m.request("vllm")
    deadline = m.total_budget_s()

    # In flight the whole time, so DRAINING never times out on its own phase clock.
    clock.advance(deadline + 1); wall.advance(deadline + 1)
    actions = m.tick(in_flight=1, outgoing_loaded=True, incoming_ready=False)

    assert sw.ABORT in actions
    assert m.blocks_claims() is False


def test_a_switch_whose_whole_budget_elapsed_before_the_restart_is_discarded(tmp_path):
    path = str(tmp_path / "switch.json")
    wall = _Clock(1_700_000_000.0)
    m1, _, _ = _m(persist_path=path, wall=wall)
    m1.request("vllm")

    wall.advance(m1.total_budget_s() + 1)
    m2 = sw.ServerSwitch(label="x", persist_path=path, now_fn=_Clock(),
                         wall_clock_fn=wall, drain_timeout_s=60.0,
                         unload_timeout_s=60.0, warm_timeout_s=600.0)

    assert m2.state == sw.IDLE
    assert m2.blocks_claims() is False


def test_a_v1_record_is_still_read_by_the_new_code(tmp_path):
    """Byte-compat: the record older engines persisted is ``{target, wall}``."""
    path = str(tmp_path / "switch.json")
    wall = _Clock(1_700_000_000.0)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"target": "vllm", "wall": wall() - 10}, fh)

    m = sw.ServerSwitch(label="x", persist_path=path, now_fn=_Clock(),
                        wall_clock_fn=wall)

    assert m.state == sw.DRAINING
    assert m.target == "vllm"
    assert m.remaining_budget_s() < m.total_budget_s()


# ── the ack budget: a start that never even happens ──────────────────────────


def test_a_TRANSIENT_failure_keeps_retrying_inside_the_ack_budget():
    m, clock, _ = _m(retry_every_s=30.0, start_ack_timeout_s=300.0)
    _reach_warming(m)
    m.report_start_outcome(sw.StartOutcome.transient("connection reset"))

    assert m.blocks_claims() is True            # still ours to finish
    clock.advance(31)
    assert sw.START_INCOMING in m.tick(
        in_flight=0, outgoing_loaded=False, incoming_ready=False)


def test_a_start_that_is_never_ACKNOWLEDGED_aborts_at_the_ack_budget_not_the_warm_one():
    """A lever that keeps failing transiently must not cost the full 1800 s window."""
    m, clock, _ = _m(retry_every_s=30.0, start_ack_timeout_s=300.0, warm_timeout_s=600.0)
    _reach_warming(m)

    for _ in range(11):                         # 11 × 31 s = 341 s > ack budget
        m.report_start_outcome(sw.StartOutcome.transient("connection reset"))
        clock.advance(31)
        actions = m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
        if sw.ABORT in actions:
            break

    assert m.state == sw.IDLE
    assert clock.t - 1000.0 < 600.0             # aborted on the ACK budget, not warm
    assert m.quarantined("vllm") is True


def test_an_acknowledged_start_gets_the_FULL_warm_window():
    """A healthy 887 s weight load must not be cut short by the ack budget."""
    m, clock, _ = _m(start_ack_timeout_s=300.0, warm_timeout_s=600.0)
    _reach_warming(m)
    m.report_start_outcome(sw.StartOutcome.started())

    clock.advance(400)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    assert m.state == sw.WARMING          # 400 s in, still loading: not our business
    assert m.blocks_claims() is True

    assert sw.DONE in m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=True)


def test_a_legacy_lever_returning_None_keeps_todays_behaviour():
    """A host lever written before start outcomes existed returns ``None``. UNKNOWN must be treated as
    acknowledged, or this change would shorten every healthy switch."""
    m, clock, _ = _m(start_ack_timeout_s=300.0, warm_timeout_s=600.0)
    _reach_warming(m)
    m.report_start_outcome(sw.StartOutcome.coerce(None))

    clock.advance(400)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    assert m.state == sw.WARMING
    assert m.blocks_claims() is True


def test_reporting_an_outcome_while_idle_is_a_harmless_no_op():
    m, _, _ = _m()
    assert m.report_start_outcome(sw.StartOutcome.permanent("x")) == []
    assert m.state == sw.IDLE


# ── the corollary invariant, fuzzed ──────────────────────────────────────────


@pytest.mark.parametrize("seed", range(40))
def test_a_switch_can_never_outlive_its_budget_or_a_definitive_failure(seed):
    """FUZZ. Two properties over any sequence of observations and lever outcomes:

    1. once the lever has said the start CANNOT work, the box is back in service and
       stays there (no phase, no retry, no re-request may re-block it);
    2. no switch holds the box past the budget its request was given — the property a
       restart used to break by re-arming a fresh window.

    Nothing loaded + no start in flight + still blocking IS the outage above.
    (SWITCHING with nothing loaded is legitimately blocking: the start is about to be
    issued on the next tick, and a claim in that gap reloads the outgoing server.)
    """
    import random
    rnd = random.Random(seed)
    m, clock, wall = _m(retry_every_s=5.0, start_ack_timeout_s=60.0,
                        warm_timeout_s=120.0, drain_timeout_s=30.0,
                        unload_timeout_s=30.0)
    m.request("vllm")
    t0 = wall()
    saw_permanent = False

    for _ in range(40):
        in_flight = rnd.choice([0, 0, 0, 1])
        outgoing = rnd.choice([True, False])
        ready = rnd.choice([False, False, False, True])
        actions = m.tick(in_flight=in_flight, outgoing_loaded=outgoing,
                         incoming_ready=ready)
        if sw.START_INCOMING in actions:
            outcome = rnd.choice([
                sw.StartOutcome.started(),
                sw.StartOutcome.transient("reset"),
                sw.StartOutcome.permanent("404"),
                sw.StartOutcome.coerce(None),
            ])
            m.report_start_outcome(outcome)
            saw_permanent = saw_permanent or outcome.fatal

        if saw_permanent:
            assert m.blocks_claims() is False, (
                "still out of service after the lever said the start cannot work "
                f"(state={m.state})")
            assert m.request("vllm") is False, "a quarantined kind was re-requested"

        if m.blocks_claims():
            assert wall() - t0 <= m.total_budget_s(), (
                f"held out of service past the budget of the original request: "
                f"state={m.state} elapsed={wall() - t0:.0f}s "
                f"budget={m.total_budget_s():.0f}s")

        step = rnd.choice([1, 5, 31])
        clock.advance(step); wall.advance(step)


# ── the seam must be WIRED, not merely present (audit delta D1) ─────────────


def test_a_SUCCESSFUL_switch_clears_the_kinds_quarantine():
    """AUDIT D2. The entry outlives its own expiry so a broken target escalates — but
    a completed switch proves the kind healthy. Without this every slow-but-successful
    load doubled the next hold until the box refused the kind for hours."""
    m, clock, wall = _m(quarantine_s=100.0)
    _reach_warming(m)
    m.report_start_outcome(sw.StartOutcome.permanent("404"))
    assert m.quarantined("vllm") is True
    wall.advance(101); clock.advance(101)

    assert m.request("vllm") is True                 # hold expired, retry allowed
    _drive_to_warming_after_request(m)
    assert sw.DONE in m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=True)

    _reach_warming(m)
    m.report_start_outcome(sw.StartOutcome.permanent("404"))

    assert m.quarantine_remaining_s("vllm") == pytest.approx(100.0, abs=1.0), (
        "the success should have reset the failure count, not doubled the next hold")


def test_the_engine_actually_FEEDS_the_levers_verdict_to_the_switch():
    """AUDIT D1, and the reason this test exists at all: the seam was written, unit
    tested and documented as 'called right after the START' — while the driver dropped
    the lever's return value on the floor. Every guarantee above was unreachable in the
    engine. A source scan is crude, but it is the thing that was missing."""
    import inspect
    from queue_workflows import claim_worker

    src = inspect.getsource(claim_worker.ClaimWorker._drive_server_switch)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())

    assert "report_start_outcome" in code, (
        "the START_INCOMING branch must report what the lever did")
    i = code.index("START_INCOMING")
    assert "report_start_outcome" in code[i:], (
        "the report must happen in the START branch, in the same tick")


def test_a_missing_start_lever_is_PERMANENT_not_a_1800_second_wait():
    """A worker whose host never wired a start lever could never complete a vllm
    switch, yet warmed for the full window on every attempt."""
    import inspect
    from queue_workflows import claim_worker

    code = inspect.getsource(claim_worker.ClaimWorker._drive_server_switch)
    assert "permanent(" in code and "no vllm start lever wired" in code


def test_the_deadline_survives_a_wall_clock_that_reads_ZERO():
    """AUDIT round 5, follow-up 3. The `is None` sentinel that replaced `if not
    self._requested_wall` was fixed in code and pinned by nothing: every other test
    injects a large wall clock, so mutating the guard back left the whole suite green.
    A fix nobody can break is a fix nobody is keeping."""
    zero = _Clock(0.0)
    m, clock, _ = _m(wall=zero)
    m.request("vllm")

    assert m.remaining_budget_s() == pytest.approx(m.total_budget_s(), abs=1.0)

    zero.advance(m.total_budget_s() + 1)
    clock.advance(1)

    # in_flight=1 would otherwise hold DRAINING open indefinitely
    assert sw.ABORT in m.tick(in_flight=1, outgoing_loaded=True, incoming_ready=False)
    assert m.blocks_claims() is False
