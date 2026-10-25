"""ONE SERVER AT A TIME — the drain-based switch state machine.

Moving a box from ollama to vLLM is not "unload then start". vLLM needs minutes to
load a large model, and a box that keeps claiming during those minutes reloads the server it
just unloaded. The result: peak memory near the box's total with BOTH resident, a few GB
from OOM, followed by the incoming vLLM
losing the residency vote (it had no traffic yet) and being killed — forever.

The missing thing is an OUT-OF-SERVICE WINDOW, not an ordering. These pin it.

    IDLE ──request──> DRAINING ──in-flight 0──> SWITCHING ──started──> WARMING
      ^                                                                   │
      └───────────────────── ready / timeout ─────────────────────────────┘

Everything is a pure decision on a virtual clock: no sleeping, no network, no DB.
"""
from __future__ import annotations

import pytest

from queue_workflows import server_switch as sw


def _m(**kw):
    """A switch with fast, explicit timeouts so a test never depends on defaults."""
    kw.setdefault("drain_timeout_s", 60.0)
    kw.setdefault("warm_timeout_s", 600.0)
    clock = kw.pop("clock", None) or _Clock()
    m = sw.ServerSwitch(now_fn=clock, **kw)
    return m, clock


class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


# ── the resting state ────────────────────────────────────────────────────────


def test_idle_by_default_and_blocks_nothing():
    m, _ = _m()
    assert m.state == sw.IDLE
    assert m.blocks_claims() is False
    assert m.protected_kind() is None


def test_a_tick_while_idle_does_nothing():
    m, _ = _m()
    assert m.tick(in_flight=3, outgoing_loaded=True, incoming_ready=False) == []
    assert m.state == sw.IDLE


# ── requesting a switch ──────────────────────────────────────────────────────


def test_requesting_a_switch_enters_DRAINING_and_blocks_claims_at_once():
    """The whole point: stop taking work the OUTGOING server would serve, the moment
    the decision is made — not after it drains."""
    m, _ = _m()
    assert m.request("vllm") is True
    assert m.state == sw.DRAINING
    assert m.blocks_claims() is True


def test_requesting_the_same_switch_twice_is_a_no_op():
    m, _ = _m()
    m.request("vllm")
    assert m.request("vllm") is False, "already switching there"
    assert m.state == sw.DRAINING


def test_cannot_start_a_second_switch_while_one_is_running():
    m, _ = _m()
    m.request("vllm")
    assert m.request("ollama") is False
    assert m.target == "vllm"


# ── draining ─────────────────────────────────────────────────────────────────


def test_draining_waits_for_in_flight_to_reach_zero():
    m, _ = _m()
    m.request("vllm")
    assert m.tick(in_flight=2, outgoing_loaded=True, incoming_ready=False) == []
    assert m.state == sw.DRAINING
    assert m.blocks_claims() is True


def test_drained_box_unloads_FIRST_and_does_not_start_in_the_same_breath():
    """The unload and the start must not be issued together: an unload can
    silently fail (the lever returns what it actually freed, and it may free nothing),
    and starting anyway is exactly how the box ended up holding two full copies of the weights."""
    m, _ = _m()
    m.request("vllm")
    acts = m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    assert acts == [sw.UNLOAD_OUTGOING]
    assert sw.START_INCOMING not in acts
    assert m.state == sw.SWITCHING


def test_the_incoming_starts_only_once_the_outgoing_is_OBSERVED_gone():
    m, clock = _m()
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)   # unload issued
    clock.advance(5)
    acts = m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    assert acts == [sw.START_INCOMING]
    assert m.state == sw.WARMING


def test_an_unload_that_never_takes_effect_ABORTS_instead_of_starting():
    """The sharp end of the same rule: a silently-failed unload must cost a slow queue, not
    an OOM. The incoming server is never started on top of the outgoing one."""
    m, clock = _m(drain_timeout_s=60.0, unload_timeout_s=60.0)
    m.request("vllm")
    started = 0
    for _ in range(20):
        started += m.tick(
            in_flight=0, outgoing_loaded=True, incoming_ready=False,
        ).count(sw.START_INCOMING)
        clock.advance(10)
        if m.state == sw.IDLE:
            break
    assert started == 0, "must NEVER start while the outgoing server is still loaded"
    assert m.state == sw.IDLE, "and it must abort rather than block forever"


def test_a_stuck_unload_is_RETRIED_not_issued_once_and_forgotten():
    m, clock = _m(drain_timeout_s=600.0, retry_every_s=30.0)
    m.request("vllm")
    unloads = m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False).count(
        sw.UNLOAD_OUTGOING)
    for _ in range(4):
        clock.advance(31)
        unloads += m.tick(
            in_flight=0, outgoing_loaded=True, incoming_ready=False,
        ).count(sw.UNLOAD_OUTGOING)
    assert unloads >= 3, "a lever that did nothing must be pulled again"


def test_an_already_unloaded_outgoing_skips_straight_to_the_start():
    m, clock = _m()
    m.request("vllm")
    assert m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False) == []
    clock.advance(1)
    assert m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False) == [
        sw.START_INCOMING]


def test_a_failed_START_is_retried_so_the_box_is_not_left_with_NO_server():
    """Actions are emitted once; if the caller's start hook raises, the box
    would otherwise sit out the whole warm window having unloaded one server and
    started none."""
    m, clock = _m(retry_every_s=30.0, warm_timeout_s=600.0)
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    starts = m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False).count(
        sw.START_INCOMING)
    for _ in range(4):
        clock.advance(31)
        starts += m.tick(
            in_flight=0, outgoing_loaded=False, incoming_ready=False,
        ).count(sw.START_INCOMING)
    assert starts >= 3


def test_a_drain_that_never_finishes_ABORTS_rather_than_wedging_the_box():
    """THE KILL LINE. A box that can neither claim nor switch is worse than one that
    never switched — it silently stops doing any work at all."""
    m, clock = _m(drain_timeout_s=60.0)
    m.request("vllm")
    clock.advance(61)
    acts = m.tick(in_flight=1, outgoing_loaded=True, incoming_ready=False)
    assert acts == [sw.ABORT]
    assert m.state == sw.IDLE
    assert m.blocks_claims() is False, "the box must go back to work"


# ── warming ──────────────────────────────────────────────────────────────────


def test_the_incoming_server_is_PROTECTED_while_it_warms():
    """It has no traffic yet, so every "who is busy?" vote would evict it. That is
    exactly how vLLM got killed on every attempt."""
    m, _ = _m()
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)   # unload
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)  # start
    assert m.state == sw.WARMING
    assert m.protected_kind() == "vllm"


def test_claims_stay_blocked_until_the_incoming_server_is_READY():
    """Claiming during the warm window is what reloaded the outgoing server."""
    m, clock = _m()
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    clock.advance(10)
    assert m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False) == []
    assert m.blocks_claims() is True


def test_ready_completes_the_switch_and_resumes_service():
    m, clock = _m()
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    clock.advance(480)
    acts = m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=True)
    assert acts == [sw.DONE]
    assert m.state == sw.IDLE
    assert m.blocks_claims() is False
    assert m.protected_kind() is None


def test_a_warm_that_never_completes_ABORTS_and_resumes_service():
    """A vLLM that fails to boot must not take the box out of service forever."""
    m, clock = _m(warm_timeout_s=600.0)
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    clock.advance(601)
    acts = m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    assert acts == [sw.ABORT]
    assert m.state == sw.IDLE
    assert m.blocks_claims() is False


# ── the invariant the whole thing exists for ─────────────────────────────────


@pytest.mark.parametrize("in_flight", [0, 1, 5])
def test_the_box_NEVER_claims_while_two_kinds_could_be_loaded(in_flight):
    """From the request until the incoming server is ready, claims are blocked. There
    is no reachable state in which the box takes new work while a switch is mid-flight
    — that window is the only thing that can reload the outgoing server."""
    m, clock = _m()
    m.request("vllm")
    for _ in range(3):
        assert m.blocks_claims() is True
        m.tick(in_flight=in_flight, outgoing_loaded=True, incoming_ready=False)
        clock.advance(5)
        if m.state == sw.IDLE:        # aborted — that is allowed, and unblocks
            assert m.blocks_claims() is False
            return
    assert m.blocks_claims() is True


def test_unload_is_never_emitted_once_WARMING():
    """Unloading after the start would free the server we just brought up."""
    m, clock = _m()
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    assert m.state == sw.WARMING
    for _ in range(4):
        clock.advance(31)
        later = m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
        assert sw.UNLOAD_OUTGOING not in later, "must not free the incoming server"


def test_start_is_not_re_issued_faster_than_the_retry_interval():
    """A vLLM takes minutes to boot; ticking every 10 s must not spawn a start each
    time — only the bounded retry may re-issue it."""
    m, clock = _m(retry_every_s=120.0)
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    starts = m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False).count(
        sw.START_INCOMING)
    for _ in range(6):
        clock.advance(10)
        starts += m.tick(
            in_flight=0, outgoing_loaded=False, incoming_ready=False,
        ).count(sw.START_INCOMING)
    assert starts == 1


def test_a_completed_switch_can_be_followed_by_another():
    m, clock = _m()
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=True)
    assert m.state == sw.IDLE
    assert m.request("ollama") is True
    assert m.target == "ollama"


# ═══ WIRING — a state machine nothing calls is worthless ═════════════════════
#
# The unit tests above never touch claim_worker, which is exactly where the two real
# defects (start-on-failed-unload; actions never retried) lived.


def _gpu_worker():
    from queue_workflows import claim_worker
    from queue_workflows.server_switch import ServerSwitch

    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-a-gpu"
            self.queue = "gpu"
            self._switch = ServerSwitch(label="box-a-gpu")
    return _W()


def test_BOTH_claim_lanes_refuse_work_while_a_switch_is_in_flight():
    """One gated lane is not enough — the other would keep reloading the outgoing
    server, which is the whole failure this exists to stop."""
    w = _gpu_worker()
    assert w._switch.request("vllm") is True
    assert w._claim() is None, "inline lane must be out of service"
    assert w._claim_pool() is None, "pool lane must be out of service"


def test_both_lanes_serve_again_once_the_switch_aborts():
    """The KILL line, end to end: an aborted switch must put the box back to work."""
    import queue_workflows.node_queue as nq
    w = _gpu_worker()
    w._switch.request("vllm")
    # Drive it to abort via the drain timeout.
    w._switch._drain_timeout_s = 0.0
    w._switch.tick(in_flight=1, outgoing_loaded=True, incoming_ready=False)
    assert w._switch.state == sw.IDLE
    assert w._switch.blocks_claims() is False
    # The lanes no longer short-circuit; they now reach the real claim path.
    calls = []
    orig = nq.claim_next_gpu_job
    nq.claim_next_gpu_job = lambda *a, **k: calls.append(1) or None
    try:
        w._pool_gpu_serving_ok = lambda: True
        w._gpu_slot_budget = lambda: 1
        w._physical_box = lambda: "box-a"
        w._served_llm_types = lambda: ["ollama"]
        w._warm_state = lambda: ([], [])
        w.lease_s = 600
        w.host_priority = 0
        w._claim_pool()
        assert len(calls) == 1, "pool lane must attempt a claim after the abort"
        # …and the INLINE lane too (the test name says both, so the body
        # drives both).
        w.model_cache = type("MC", (), {"current_model": None})()
        w._gpu_capacity_for_claim = lambda: (None, [], False)
        w._claim()
        assert len(calls) == 2, "inline lane must attempt a claim after the abort"
    finally:
        nq.claim_next_gpu_job = orig


def test_the_enforcer_is_given_the_switchs_protected_kind(monkeypatch):
    """The protection has to REACH the vote. The
    first version of this test was a source grep — it stayed green with the wiring
    commented out. Now the real arm method runs and the captured constructor kwarg
    must BE the switch's bound method, so dead wiring fails."""
    from queue_workflows import claim_worker, model_residency

    captured = {}

    class _FakeEnforcer:
        def __init__(self, **kw):
            captured.update(kw)
        def start(self):
            pass

    monkeypatch.setattr(model_residency, "ModelResidencyEnforcer", _FakeEnforcer)
    w = _gpu_worker()
    w._collect_box_residents = lambda: []
    w._stop_rogue_vllm = lambda: None
    w._free_box_comfyui = lambda: None
    w._arm_model_residency_enforcer()
    assert captured, "the arm method must construct the enforcer"
    assert captured.get("protected_server_fn") == w._switch.protected_kind, (
        "the enforcer must be handed THE SWITCH'S protected_kind, live")
    # …and the callable actually reflects the switch state.
    assert captured["protected_server_fn"]() is None
    w._switch.request("vllm")
    w._switch.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    w._switch.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    assert captured["protected_server_fn"]() == "vllm"


# ═══ WARMING retries and the outgoing server ═════════════════════════════════


def test_the_WARMING_retry_never_starts_while_the_outgoing_is_back():
    """The start retry must not re-open the unload-first rule. The retry re-issued START_INCOMING
    without re-checking the world — but the outgoing server can REAPPEAR mid-warm (a
    peer worker on the same box reloads it; the switch is per-process). Starting then
    is the two-copies bug again, through the retry door. A fuzz run found violations
    before the fix, all in WARMING."""
    m, clock = _m(retry_every_s=30.0, warm_timeout_s=600.0)
    m.request("vllm")
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)    # unload
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)   # start
    assert m.state == sw.WARMING
    starts = 0
    for _ in range(6):
        clock.advance(31)
        starts += m.tick(
            in_flight=0, outgoing_loaded=True, incoming_ready=False,   # it came BACK
        ).count(sw.START_INCOMING)
    assert starts == 0, "START must never be (re-)issued while the outgoing is loaded"
    # …and once it is gone again, the retry resumes.
    clock.advance(31)
    assert sw.START_INCOMING in m.tick(
        in_flight=0, outgoing_loaded=False, incoming_ready=False)


def test_SWITCHING_retry_honours_the_stated_cadence():
    """Entering SWITCHING zeroed the retry clock, so the first retry
    fired on the very next tick (11 pulls in 300 s at retry_every_s=30)."""
    m, clock = _m(drain_timeout_s=600.0, retry_every_s=30.0)
    m.request("vllm")
    first = m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    assert first == [sw.UNLOAD_OUTGOING]
    clock.advance(5)                        # well inside the interval
    assert m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False) == []
    clock.advance(30)                       # past it
    assert m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False) == [
        sw.UNLOAD_OUTGOING]


def test_worst_case_out_of_service_is_bounded_by_all_three_phases():
    """The bound is drain + switching + warm — assert it as arithmetic so a fourth
    phase cannot creep in silently."""
    m, clock = _m(drain_timeout_s=60.0, warm_timeout_s=120.0)
    m.request("vllm")
    t0 = clock.t
    # worst case each phase: drain to the brink, switch to the brink, warm to abort
    clock.advance(59); m.tick(in_flight=1, outgoing_loaded=True, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)     # -> SWITCHING
    clock.advance(59); m.tick(in_flight=0, outgoing_loaded=True, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)    # -> WARMING
    clock.advance(121)
    acts = m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    assert acts == [sw.ABORT]
    assert clock.t - t0 <= 60 + 60 + 121 + 1
    assert m.blocks_claims() is False


def test_retry_interval_constant_is_exported():
    assert "DEFAULT_RETRY_EVERY_S" in sw.__all__


def test_blocks_claims_self_aborts_when_nobody_ticks():
    """The timeout escapes lived only in tick(), so a box
    whose heartbeat stopped (config-reachable kill-switch) stayed blocked forever.
    The claim path calls blocks_claims() every loop — it must be its own escape."""
    m, clock = _m(drain_timeout_s=60.0)
    m.request("vllm")
    assert m.blocks_claims() is True
    clock.advance(60 * 2 + 1)          # double the phase budget, zero ticks
    assert m.blocks_claims() is False, "must self-abort rather than wedge"
    assert m.state == sw.IDLE
    assert m.request("ollama") is True, "and the machine is reusable after"


def test_blocks_claims_does_not_preempt_a_ticking_switch():
    """Within the budget the normal machinery owns the decision."""
    m, clock = _m(drain_timeout_s=60.0)
    m.request("vllm")
    clock.advance(90)                   # past the budget, but under 2x
    assert m.blocks_claims() is True, "the tick-driven abort gets first refusal"


# ═══ ANALYSIS #1 — restart persistence + the cross-process box hold ══════════


def test_a_restart_mid_switch_RE_ENTERS_the_out_of_service_window(tmp_path):
    """The switch state was per-process, so a worker restart mid-switch forgot it —
    the fresh process resumed claiming while the incoming server warmed, which is
    the exact race the machine exists to close."""
    path = str(tmp_path / "server_switch_box-a-gpu.json")
    wall = [1000.0]
    m1 = sw.ServerSwitch(label="box-a-gpu", persist_path=path,
                         wall_clock_fn=lambda: wall[0])
    m1.request("vllm")
    assert m1.blocks_claims() is True
    # …process dies here; a NEW process constructs its switch…
    wall[0] += 60
    m2 = sw.ServerSwitch(label="box-a-gpu", persist_path=path,
                         wall_clock_fn=lambda: wall[0])
    assert m2.state == sw.DRAINING, "the fresh process re-enters the window"
    assert m2.target == "vllm"
    assert m2.blocks_claims() is True


def test_a_STALE_persisted_switch_is_ignored_and_removed(tmp_path):
    """A record from a long-dead switch must not take a healthy box out of service."""
    import os
    path = str(tmp_path / "s.json")
    wall = [1000.0]
    m1 = sw.ServerSwitch(label="x", persist_path=path, wall_clock_fn=lambda: wall[0])
    m1.request("vllm")
    wall[0] += m1._budget_total() + 1          # older than the whole switch budget
    m2 = sw.ServerSwitch(label="x", persist_path=path, wall_clock_fn=lambda: wall[0])
    assert m2.state == sw.IDLE
    assert not os.path.exists(path), "the stale record is cleaned up"


def test_a_completed_switch_leaves_no_record_behind(tmp_path):
    import os
    path = str(tmp_path / "s.json")
    m = sw.ServerSwitch(label="x", persist_path=path)
    m.request("vllm")
    assert os.path.exists(path)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=False)
    m.tick(in_flight=0, outgoing_loaded=False, incoming_ready=True)
    assert m.state == sw.IDLE
    assert not os.path.exists(path)


def test_the_switch_HOLDS_the_box_lease_and_releases_on_resolution(monkeypatch):
    """Cross-process half: peers of every project honour the flock box lease, so a
    switching worker holding a distinct sentinel makes them SPILL for the duration
    instead of reloading the server being replaced."""
    from queue_workflows import claim_worker

    events = []

    class _Lease:
        def acquire(self, model):
            events.append(("acquire", model)); return True
        def renew(self, model):
            events.append(("renew", model)); return True
        def release(self):
            events.append(("release", None))
        def current(self):
            return type("S", (), {"model": claim_worker.SWITCH_SLOT})()

    w = _gpu_worker()
    w._get_box_lease = lambda: _Lease()
    w._gpu_active_count = lambda: 0
    monkeypatch.setattr("queue_workflows.llm_probe.loaded_models", lambda url: [])
    monkeypatch.setattr("queue_workflows.llm_probe.probe_llm_servers",
                        lambda url, **k: ["vllm"])
    monkeypatch.setattr("queue_workflows.llm_probe.vllm_served_models",
                        lambda url, **k: [])
    monkeypatch.setattr(
        "queue_workflows.llm_backends.factory.resolve_base_url",
        lambda kind="ollama": f"http://{kind}:1")

    w._switch.request("vllm")
    w._drive_server_switch()                    # drain(0) → unload issued, lease held
    assert ("acquire", claim_worker.SWITCH_SLOT) in events
    w._drive_server_switch()                    # outgoing gone → start
    w._drive_server_switch()                    # incoming ready → DONE + release
    assert w._switch.state == sw.IDLE
    assert ("release", None) in events, "the box frees the moment the switch resolves"


def test_the_lease_tick_never_clobbers_the_switch_slot(monkeypatch):
    """_box_lease_tick drives the switch (which acquires SWITCH_SLOT)
    and then renewed its held MODEL one line later — renew() being acquire() on the
    single-slot store, the switch's hold survived 1 of 6 ticks and peers kept
    reloading the outgoing server. While a switch is active, the slot is the
    switch's; the held-model bookkeeping waits."""
    from queue_workflows import claim_worker

    slot_writes = []

    class _Lease:
        def acquire(self, model): slot_writes.append(model); return True
        def renew(self, model):   slot_writes.append(model); return True
        def release(self):        slot_writes.append(None)
        def current(self):        return type("S", (), {"model": slot_writes[-1] if slot_writes else None})()

    w = _gpu_worker()
    w._get_box_lease = lambda: _Lease()
    w._gpu_active_count = lambda: 0
    w._box_held_model = "chat-model-a:latest"        # a job's lease bookkeeping exists
    w._box_active_jobs = 1
    w._refresh_box_llm_state = lambda: None
    w._model_still_resident = lambda m: True
    monkeypatch.setattr("queue_workflows.llm_probe.loaded_models", lambda url: ["chat-model-a:latest"])
    monkeypatch.setattr("queue_workflows.llm_probe.probe_llm_servers", lambda url, **k: [])
    monkeypatch.setattr("queue_workflows.llm_probe.vllm_served_models", lambda url, **k: [])
    monkeypatch.setattr("queue_workflows.llm_backends.factory.resolve_base_url",
                        lambda kind="ollama": f"http://{kind}:1")

    w._switch.request("vllm")
    for _ in range(3):
        w._box_lease_tick()

    during = [x for x in slot_writes if x is not None]
    assert during, "the switch must be holding the slot"
    assert set(during) == {claim_worker.SWITCH_SLOT}, (
        f"NOTHING but SWITCH_SLOT may touch the lease during a switch — saw {set(during)}")


def test_switch_timeouts_are_env_overridable(monkeypatch):
    """A box loading vLLM weights from network storage took nearly its whole warm window —
    seconds of margin. The budget must be a deployment knob, not a constant edit."""
    from queue_workflows import claim_worker
    monkeypatch.setenv("QUEUE_WORKFLOWS_SWITCH_WARM_TIMEOUT_S", "2400")
    monkeypatch.setenv("QUEUE_WORKFLOWS_SWITCH_DRAIN_TIMEOUT_S", "120")
    drain, warm = claim_worker._switch_timeouts()
    assert (drain, warm) == (120.0, 2400.0)


def test_switch_timeout_defaults_have_real_headroom(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_SWITCH_WARM_TIMEOUT_S", raising=False)
    monkeypatch.delenv("QUEUE_WORKFLOWS_SWITCH_DRAIN_TIMEOUT_S", raising=False)
    from queue_workflows import claim_worker
    _, warm = claim_worker._switch_timeouts()
    assert warm >= 1800, "a slow network-storage load demands >2x headroom, not seconds"


def test_malformed_timeout_env_falls_through(monkeypatch):
    from queue_workflows import claim_worker
    monkeypatch.setenv("QUEUE_WORKFLOWS_SWITCH_WARM_TIMEOUT_S", "soon")
    _, warm = claim_worker._switch_timeouts()
    assert warm == 1800.0


# ═══ CAPABILITY — a box must never chase a server it cannot run ══════════════


def test_a_resumed_vllm_switch_on_a_vllm_less_box_is_ABANDONED(tmp_path, monkeypatch):
    """A box (ROCm, bare topology URL ⇒ no distinct vLLM endpoint)
    resumed a persisted vllm switch after a restart and sat OUT OF SERVICE warming
    a server that can never exist."""
    from queue_workflows import claim_worker
    path = str(tmp_path / "server_switch_box-c-gpu.json")
    m1 = sw.ServerSwitch(label="box-c-gpu", persist_path=path)
    m1.request("vllm")

    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_MODEL_LEASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "queue_workflows.llm_backends.factory.resolve_base_url",
        lambda kind="ollama": "http://same:11434")     # bare entry: both kinds one URL

    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-c-gpu"
            self.queue = "gpu"
    w = _W.__new__(_W)
    w.host, w.queue = "box-c-gpu", "gpu"
    # replicate the construction block
    from queue_workflows.server_switch import ServerSwitch
    w._switch = ServerSwitch(label=w.host, persist_path=path)
    assert w._switch.state == sw.DRAINING, "persistence resumed the doomed switch"
    if (w._switch.target == "vllm" and not claim_worker.ClaimWorker._box_has_vllm_endpoint(w)):
        w._switch.abandon("this box has no distinct vLLM endpoint")
    assert w._switch.state == sw.IDLE
    assert w._switch.blocks_claims() is False, "the box goes straight back to work"
    import os
    assert not os.path.exists(path), "the doomed record is cleared"


def test_demand_start_is_capability_gated(monkeypatch):
    """2000 pending vllm-preferring rows must not make a vllm-less box open a
    switch it can never complete."""
    from queue_workflows import claim_worker
    w = claim_worker.ClaimWorker.__new__(claim_worker.ClaimWorker)
    w.host, w.queue = "box-c-gpu", "gpu"
    monkeypatch.setattr(
        "queue_workflows.llm_backends.factory.resolve_base_url",
        lambda kind="ollama": "http://same:11434")
    assert claim_worker.ClaimWorker._box_has_vllm_endpoint(w) is False
    monkeypatch.setattr(
        "queue_workflows.llm_backends.factory.resolve_base_url",
        lambda kind="ollama": f"http://{kind}:1")
    assert claim_worker.ClaimWorker._box_has_vllm_endpoint(w) is True
