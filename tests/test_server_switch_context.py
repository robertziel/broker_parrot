"""Switching the WINDOW, not the server — the same out-of-service machine, one kind.

A job may need a bigger context window than this box's server was started with
(migration 0028). Giving it one means restarting that server, which has exactly the
hazards a kind switch has — in-flight requests die, the box must not claim while its
server is down — and one it does not: there is no OUTGOING server to free, because the
recreate replaces it in a single operation. So a resize reuses ``ServerSwitch`` (drain,
block claims, timeout-bounded, persisted across a restart) and differs in three places:

  * nothing is unloaded — the box goes straight to starting the new window;
  * the target is a WINDOW on the kind already serving, not another kind;
  * a failed resize quarantines RESIZING that kind, never the kind itself. A box whose
    window cannot be changed must still be able to switch to that server normally.
"""
from __future__ import annotations

import json

from queue_workflows import server_switch as sw


class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def _m(**kw):
    kw.setdefault("drain_timeout_s", 60.0)
    kw.setdefault("warm_timeout_s", 600.0)
    clock = kw.pop("clock", None) or _Clock()
    return sw.ServerSwitch(now_fn=clock, **kw), clock


# ── requesting one ───────────────────────────────────────────────────────────


def test_a_resize_carries_the_window_it_is_for():
    m, _ = _m()
    assert m.request("ollama", context_length=131072, resize=True) is True
    assert (m.target, m.target_context, m.resize) == ("ollama", 131072, True)
    assert m.blocks_claims() is True, "the server is about to restart under us"


def test_a_plain_kind_switch_still_has_no_window_of_its_own():
    m, _ = _m()
    m.request("vllm")
    assert (m.target_context, m.resize) == (None, False)


def test_a_kind_switch_may_also_name_the_window_to_start_at():
    """R4: a server being started anyway should come up at the window the queue needs,
    rather than come up wrong and be restarted a second time."""
    m, _ = _m()
    m.request("vllm", context_length=65536)
    assert (m.target, m.target_context, m.resize) == ("vllm", 65536, False)


def test_one_resize_at_a_time_like_every_other_switch():
    m, _ = _m()
    assert m.request("ollama", context_length=131072, resize=True) is True
    assert m.request("ollama", context_length=262144, resize=True) is False


# ── the sequence ─────────────────────────────────────────────────────────────


def test_a_resize_never_unloads_anything_it_goes_straight_to_the_restart():
    """There is no outgoing server to free: the lever replaces the container in one
    operation. Emitting UNLOAD here would stop the very server we are resizing and
    strand the box at 'waiting for the outgoing to disappear' until the timeout."""
    m, clock = _m()
    m.request("ollama", context_length=131072, resize=True)
    assert m.tick(in_flight=1, outgoing_loaded=True, incoming_ready=False) == []
    assert m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False) == [
        sw.START_INCOMING]
    assert m.state == sw.WARMING
    clock.advance(20)
    assert m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=True) == [sw.DONE]
    assert m.state == sw.IDLE
    assert m.target_context is None and m.resize is False


def test_a_kind_switch_still_waits_for_the_outgoing_server_to_go():
    """The two-copies-on-one-card rule is untouched by any of this."""
    m, _ = _m()
    m.request("vllm", context_length=65536)
    assert m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False) == [
        sw.UNLOAD_OUTGOING]
    assert m.state == sw.SWITCHING


def test_a_resize_that_never_arrives_aborts_on_its_own_budget():
    m, clock = _m()
    m.request("ollama", context_length=131072, resize=True)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    clock.advance(601)
    assert m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False) == [sw.ABORT]
    assert m.blocks_claims() is False


# ── quarantine: resizing a kind is not the kind ──────────────────────────────


def test_a_failed_resize_does_not_quarantine_the_server_kind_itself():
    """A box whose window cannot be changed (no docker socket, say) must still be able
    to switch TO that server — the two failures have nothing to do with each other."""
    m, _ = _m()
    m.request("ollama", context_length=131072, resize=True)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    m.report_start_outcome(sw.StartOutcome.permanent("no docker socket"))

    assert m.quarantined("ollama") is False
    assert m.resize_quarantined("ollama") is True
    assert m.request("ollama") is True, "a plain switch to ollama is unaffected"


def test_a_quarantined_resize_is_refused_until_it_expires():
    m, clock = _m(quarantine_s=100.0, quarantine_max_s=400.0)
    m.request("ollama", context_length=131072, resize=True)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    m.report_start_outcome(sw.StartOutcome.permanent("no such container"))

    assert m.request("ollama", context_length=131072, resize=True) is False
    assert m.resize_quarantine_remaining_s("ollama") > 0


def test_a_completed_resize_clears_the_hold_on_resizing():
    m, clock = _m()
    m.request("ollama", context_length=131072, resize=True)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    m.report_start_outcome(sw.StartOutcome.transient("daemon busy"))
    m.report_start_outcome(sw.StartOutcome.started())
    clock.advance(5)
    assert m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=True) == [sw.DONE]
    assert m.resize_quarantined("ollama") is False


def test_a_failed_kind_switch_still_quarantines_the_kind_not_resizing_it():
    m, _ = _m()
    m.request("vllm", context_length=65536)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    m.report_start_outcome(sw.StartOutcome.permanent("no such container"))
    assert m.quarantined("vllm") is True
    assert m.resize_quarantined("vllm") is False


# ── across a restart ─────────────────────────────────────────────────────────


def test_a_resize_in_flight_survives_the_worker_restart_that_interrupted_it(tmp_path):
    """The record is what stops a fresh process from claiming into a server that is
    still restarting — the race the whole machine exists to close. It must carry the
    window too, or the resumed switch would wait for a window it can no longer name."""
    path = str(tmp_path / "switch.json")
    m, _ = _m(persist_path=path)
    m.request("ollama", context_length=131072, resize=True)

    resumed, _ = _m(persist_path=path)
    assert resumed.state == sw.DRAINING
    assert (resumed.target, resumed.target_context, resumed.resize) == (
        "ollama", 131072, True)
    assert resumed.blocks_claims() is True


def test_a_resize_quarantine_survives_a_restart_like_every_other_one(tmp_path):
    path = str(tmp_path / "switch.json")
    m, _ = _m(persist_path=path, quarantine_s=300.0)
    m.request("ollama", context_length=131072, resize=True)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    m.report_start_outcome(sw.StartOutcome.permanent("no such container"))

    resumed, _ = _m(persist_path=path)
    assert resumed.resize_quarantined("ollama") is True
    assert resumed.quarantined("ollama") is False


def test_an_older_engines_record_still_resumes(tmp_path):
    """python/workflows-style forward compat: a v2 record written before windows
    existed has no context field, and must resume as the plain kind switch it was."""
    path = str(tmp_path / "switch.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"v": 2, "quarantine": {},
                   "switch": {"target": "vllm", "requested_wall": __import__("time").time(),
                              "phase": "warming", "start_acknowledged": True}}, fh)
    resumed, _ = _m(persist_path=path)
    assert resumed.target == "vllm"
    assert (resumed.target_context, resumed.resize) == (None, False)
