"""Stop an idle vLLM server after the TTL — the vLLM twin of the ollama idle unload.

WHY IT IS SERVER-OBSERVED, NOT COUNTER-DRIVEN. ``LLMSupervisor`` already stops an
idle vLLM, but its clock is the BACKEND'S IN-PROCESS counter
(``mark_request_start``/``mark_request_end``). That only ticks if the node routes its
HTTP through the backend — and a second consumer's nodes are handed a URL and POST it
themselves, as does any host-internal caller and any other project sharing the box.
Against this fleet that clock reads "idle" while the server is flat out, so wiring
the supervisor as-is would SIGTERM vLLM mid-generation.

This is the same lesson the ollama reaper learned live and records in
its module docstring: the one signal every client shares is the SERVER's. ollama
exposes it as ``/api/ps`` ``expires_at``; vLLM exposes it as the Prometheus counters
``vllm:num_requests_running`` (in flight now) and ``vllm:request_success_total``
(monotonic completions). Running > 0, or a completion count that ADVANCED since the
last tick, means used.
"""
from __future__ import annotations

from queue_workflows import llm_idle_unload as idle


# ── the pure decision ────────────────────────────────────────────────────────


def test_busy_server_is_never_stopped():
    assert idle.should_stop_idle_vllm(
        served=True, running=1, active=0, idle_s=99999, ttl_s=300) is False


def test_a_claimed_job_in_flight_blocks_the_stop():
    """`active` is the worker's own in-flight guard: a job claimed but not yet
    POSTing must not have its server yanked out from under it."""
    assert idle.should_stop_idle_vllm(
        served=True, running=0, active=1, idle_s=99999, ttl_s=300) is False


def test_idle_past_the_ttl_stops():
    assert idle.should_stop_idle_vllm(
        served=True, running=0, active=0, idle_s=300, ttl_s=300) is True


def test_idle_under_the_ttl_does_not():
    assert idle.should_stop_idle_vllm(
        served=True, running=0, active=0, idle_s=299, ttl_s=300) is False


def test_nothing_served_means_nothing_to_stop():
    assert idle.should_stop_idle_vllm(
        served=False, running=0, active=0, idle_s=99999, ttl_s=300) is False


def test_ttl_zero_disables():
    assert idle.should_stop_idle_vllm(
        served=True, running=0, active=0, idle_s=99999, ttl_s=0) is False


# ── the reaper, on a virtual clock ───────────────────────────────────────────


class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def _reaper(activity, *, ttl=300.0, active=0):
    clock = _Clock()
    stopped = []
    r = idle.VLLMIdleStopper(
        active_fn=lambda: active,
        activity_fn=lambda: activity[0],
        stop_fn=lambda: stopped.append(clock.t),
        ttl_s=ttl, now_fn=clock,
    )
    return r, clock, stopped


def test_a_completion_between_ticks_resets_the_clock():
    """The counter advancing means SOME client used it — queue node, host-internal
    HTTP, or another project on the box. All of them count."""
    act = [{"served": True, "running": 0, "finished": 10}]
    r, clock, stopped = _reaper(act)
    r.stop_idle_once()                      # first sighting stamps "used now"
    clock.advance(200)
    act[0] = {"served": True, "running": 0, "finished": 11}   # someone used it
    r.stop_idle_once()
    clock.advance(200)                      # 400s total, but only 200s since use
    r.stop_idle_once()
    assert stopped == [], "the clock restarted at the completion"


def test_a_frozen_counter_ages_to_the_stop():
    act = [{"served": True, "running": 0, "finished": 10}]
    r, clock, stopped = _reaper(act)
    r.stop_idle_once()
    clock.advance(301)
    r.stop_idle_once()
    assert len(stopped) == 1


def test_first_sighting_grants_a_full_ttl_of_grace():
    """A server we have never seen before was, at worst, just started — never stop
    it on the strength of a single observation."""
    act = [{"served": True, "running": 0, "finished": 500}]
    r, clock, stopped = _reaper(act)
    r.stop_idle_once()
    assert stopped == []


def test_in_flight_requests_hold_it_open_indefinitely():
    act = [{"served": True, "running": 2, "finished": 10}]
    r, clock, stopped = _reaper(act)
    r.stop_idle_once()
    clock.advance(10_000)
    r.stop_idle_once()
    assert stopped == []


def test_it_stops_only_once_not_every_tick():
    act = [{"served": True, "running": 0, "finished": 10}]
    r, clock, stopped = _reaper(act)
    r.stop_idle_once()
    clock.advance(301)
    r.stop_idle_once()
    act[0] = {"served": False, "running": 0, "finished": 10}   # it is down now
    clock.advance(301)
    r.stop_idle_once()
    assert len(stopped) == 1


def test_a_dead_probe_never_stops_anything():
    """Best-effort: if we cannot observe the server we must not act on a guess."""
    act = [{"served": False, "running": 0, "finished": 0}]
    r, clock, stopped = _reaper(act)
    r.stop_idle_once()
    clock.advance(10_000)
    r.stop_idle_once()
    assert stopped == []


def test_a_raising_probe_is_swallowed():
    def _boom():
        raise RuntimeError("no route to host")
    r = idle.VLLMIdleStopper(
        active_fn=lambda: 0, activity_fn=_boom, stop_fn=lambda: None, ttl_s=300.0,
    )
    r.stop_idle_once()          # must not raise
