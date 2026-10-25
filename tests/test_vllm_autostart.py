"""Start a stopped vLLM when the queue justifies it — the other half of the rule.

The idle stopper frees a vLLM's memory after 5 unused minutes. On its own that is a
one-way door, and it closed one: stopped ⇒ not advertised ⇒ no job can prefer it ⇒
never used ⇒ stays stopped. A box can sit with vLLM `exited` and 0
requests served while thousands of jobs preferring it queue behind ollama.

The amortisation rule already says when a cold start is worth paying for: at least
``min_queue`` jobs waiting. The pair is deliberately ASYMMETRIC (a design rule) —

    unused for the TTL AND the queue is EMPTY  -> stop  (reclaim the VRAM)
    >= min_queue jobs waiting                  -> start (the cold start amortises)

Starting is gated high because the load costs minutes; stopping is gated at ZERO
because stopping with ANY work still waiting forces the very next job back onto
ollama — which reloads the same weights this box just held warm, then the queue
re-deepens and pays the vLLM cold start AGAIN. A warm server with work behind it
is never torn down.
"""
from __future__ import annotations

from queue_workflows import llm_idle_unload as idle


# ── the pure decision ────────────────────────────────────────────────────────


def test_start_when_the_queue_clears_the_threshold():
    assert idle.should_start_vllm(served=False, pending=5, min_queue=5) is True


def test_do_not_start_for_too_little_work():
    assert idle.should_start_vllm(served=False, pending=4, min_queue=5) is False


def test_never_start_one_that_is_already_serving():
    assert idle.should_start_vllm(served=True, pending=500, min_queue=5) is False


def test_threshold_zero_means_start_on_any_demand():
    assert idle.should_start_vllm(served=False, pending=1, min_queue=0) is True
    assert idle.should_start_vllm(served=False, pending=0, min_queue=0) is False


def test_no_demand_never_starts():
    assert idle.should_start_vllm(served=False, pending=0, min_queue=5) is False


# ── the daemon tick ──────────────────────────────────────────────────────────


class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def _mgr(activity, pending, *, ttl=300.0, min_queue=5, cooldown=120.0,
         may_start=None):
    clock = _Clock()
    started, stopped = [], []
    m = idle.VLLMIdleStopper(
        active_fn=lambda: 0,
        activity_fn=lambda: activity[0],
        stop_fn=lambda: stopped.append(clock.t),
        start_fn=lambda: started.append(clock.t),
        pending_fn=lambda: pending[0],
        min_queue=min_queue, restart_cooldown_s=cooldown,
        ttl_s=ttl, now_fn=clock, may_start_fn=may_start,
    )
    return m, clock, started, stopped


def test_a_stopped_vllm_with_a_deep_queue_is_started():
    act = [{"served": False, "running": 0, "finished": 0}]
    m, clock, started, _ = _mgr(act, [2403])
    m.tick()
    assert len(started) == 1


def test_a_stopped_vllm_with_a_shallow_queue_is_left_alone():
    act = [{"served": False, "running": 0, "finished": 0}]
    m, clock, started, _ = _mgr(act, [2])
    m.tick()
    assert started == []


def test_it_does_not_thrash_between_stop_and_start():
    """A box that just idled out must not be restarted by the very next tick — the
    queue behind it did not change, only our own action did.

    (Post-sticky semantics: the stop itself only fires when the queue is EMPTY —
    any waiting job keeps the warm server, see the hysteresis tests — so the
    thrash window is a stop under an empty queue followed by demand arriving.)"""
    pending = [0]                              # drained: the stop may fire
    act = [{"served": True, "running": 0, "finished": 7}]
    m, clock, started, stopped = _mgr(act, pending)
    m.tick()                                   # first sighting
    clock.advance(301)
    m.tick()                                   # idles out -> stop
    assert len(stopped) == 1
    act[0] = {"served": False, "running": 0, "finished": 7}
    pending[0] = 2403                          # demand arrives right after
    clock.advance(5)
    m.tick()                                   # inside the cooldown
    assert started == [], "must not restart what it just stopped"
    clock.advance(200)                          # past the cooldown
    m.tick()
    assert len(started) == 1, "…but demand does eventually win"


def test_start_is_not_attempted_repeatedly_while_it_boots():
    """A vLLM takes minutes to load; every tick in between still reads served=False
    and must not fire another start."""
    act = [{"served": False, "running": 0, "finished": 0}]
    m, clock, started, _ = _mgr(act, [2403], cooldown=120.0)
    m.tick()
    clock.advance(10); m.tick()
    clock.advance(10); m.tick()
    assert len(started) == 1


def test_no_start_hook_wired_is_a_no_op_not_a_crash():
    m = idle.VLLMIdleStopper(
        active_fn=lambda: 0,
        activity_fn=lambda: {"served": False, "running": 0, "finished": 0},
        stop_fn=lambda: None, start_fn=None, pending_fn=lambda: 999,
        min_queue=5, ttl_s=300.0,
    )
    m.tick()        # must not raise


def test_a_raising_pending_probe_never_starts_anything():
    def _boom():
        raise RuntimeError("db down")
    started = []
    m = idle.VLLMIdleStopper(
        active_fn=lambda: 0,
        activity_fn=lambda: {"served": False, "running": 0, "finished": 0},
        stop_fn=lambda: None, start_fn=lambda: started.append(1),
        pending_fn=_boom, min_queue=5, ttl_s=300.0,
    )
    m.tick()
    assert started == [], "an unreadable queue must not trigger a 7-minute start"


def _worker():
    from queue_workflows import claim_worker

    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-a-gpu"
    w = _W()
    w._collect_box_residents = lambda: []
    return w


def test_the_start_is_a_HANDOVER_ollama_freed_BEFORE_vllm_starts(monkeypatch):
    """ONE SERVER AT A TIME. Each server holds its own full copy of the model, so
    starting vLLM while ollama is still resident takes a unified-memory box to nearly
    all of its memory — a few GB from OOM. Order is the contract."""
    from queue_workflows import claim_worker, model_residency
    from queue_workflows.config import get_config

    order = []

    def _clear(kind, model, residents, **levers):
        order.append(f"clear:{kind}")
        levers["unload_ollama"](["chat-model-a:latest"])   # the real lever, exercised
        return []

    monkeypatch.setattr(model_residency, "clear_box_for", _clear)
    monkeypatch.setattr(
        "queue_workflows.llm_probe.unload_ollama_models",
        lambda url, models: order.append("ollama_unloaded"))
    monkeypatch.setattr(get_config(), "vllm_start_fn",
                        lambda m: order.append("vllm_started"), raising=False)

    assert _worker()._handover_to_vllm() is True
    assert order == ["clear:vllm", "ollama_unloaded", "vllm_started"], order


def test_a_failed_clear_must_not_start_vllm(monkeypatch):
    """Better a slow queue on ollama than two full copies on one card."""
    from queue_workflows import model_residency
    from queue_workflows.config import get_config

    started = []
    monkeypatch.setattr(
        model_residency, "clear_box_for",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ollama won't release")))
    monkeypatch.setattr(get_config(), "vllm_start_fn",
                        lambda m: started.append(1), raising=False)

    assert _worker()._handover_to_vllm() is False
    assert started == [], "a box that could not be freed must not gain a 2nd server"


def test_no_start_hook_means_no_handover(monkeypatch):
    from queue_workflows.config import get_config
    monkeypatch.setattr(get_config(), "vllm_start_fn", None, raising=False)
    assert _worker()._handover_to_vllm() is False


# ── hysteresis + backoff ─────────────────────────────────────────────────────


def test_idle_stop_refuses_while_the_queue_would_immediately_restart_it():
    """The stop side must be symmetric with the start side: a 5-min traffic stall
    under a 2000-deep queue must not unload a model the next tick pays minutes to reload."""
    act = [{"served": True, "running": 0, "finished": 7}]
    m, clock, started, stopped = _mgr(act, [2000])
    m.tick()                                # first sighting
    clock.advance(301)                      # idle past the TTL…
    m.tick()
    assert stopped == [], "…but 2000 jobs are waiting: keep the warm server"


def test_idle_stop_still_fires_when_the_queue_is_EMPTY():
    act = [{"served": True, "running": 0, "finished": 7}]
    m, clock, started, stopped = _mgr(act, [0])
    m.tick()
    clock.advance(301)
    m.tick()
    assert len(stopped) == 1


# ── sticky stop (design rule): don't hand back mid-stream ────────


def test_sticky_stop_ONE_waiting_job_keeps_the_warm_server():
    """The stop threshold is ZERO, not min_queue. At pending=1..min_queue-1 the old
    symmetric rule stopped the server; the next job then ran on ollama — reloading
    the very weights this box held warm — and the re-deepening queue paid the vLLM
    cold start a second time. Any waiting job keeps the server."""
    act = [{"served": True, "running": 0, "finished": 7}]
    m, clock, started, stopped = _mgr(act, [1], min_queue=5)
    m.tick()                                # first sighting
    clock.advance(301)                      # idle past the TTL…
    m.tick()
    assert stopped == [], "1 job still waiting: the warm server must survive"


def test_sticky_stop_applies_even_with_the_start_gate_disabled():
    """min_queue=0 disables the START amortisation gate ("always pay the start") —
    it must not also disable the KEEP rule, or a lull under a non-empty queue
    would still tear the warm server down."""
    act = [{"served": True, "running": 0, "finished": 7}]
    m, clock, started, stopped = _mgr(act, [3], min_queue=0)
    m.tick()
    clock.advance(301)
    m.tick()
    assert stopped == [], "work is waiting: min_queue=0 must not unstick the stop"


def test_start_defers_to_the_may_start_gate():
    """PARK RELEASES THE CARD: a worker heading into (or sitting
    in) an OFF control state must not ignite a multi-minute server start the
    parked box can never use — demand is some other box's business."""
    act = [{"served": False, "running": 0, "finished": 0}]
    m, clock, started, _ = _mgr(act, [2403], may_start=lambda: False)
    m.tick()
    assert started == [], "an OFF worker must not start a server"


def test_the_gate_open_still_starts_on_demand():
    act = [{"served": False, "running": 0, "finished": 0}]
    m, clock, started, _ = _mgr(act, [2403], may_start=lambda: True)
    m.tick()
    assert len(started) == 1


def test_a_raising_gate_fails_OPEN():
    """The gate is a control-plane read; a DB blip must degrade to TODAY'S
    behaviour (start on demand), not wedge every start on the fleet."""
    def _boom():
        raise RuntimeError("control-plane read failed")
    act = [{"served": False, "running": 0, "finished": 0}]
    m, clock, started, _ = _mgr(act, [2403], may_start=_boom)
    m.tick()
    assert len(started) == 1, "an unreadable gate must not block starts"


def test_a_target_that_never_comes_up_is_backed_off_exponentially():
    """A permanently-broken vLLM, re-requested every cooldown, keeps the box
    out of service most of the time. Each failed start doubles the wait; a
    server observed serving resets it."""
    act = [{"served": False, "running": 0, "finished": 0}]
    m, clock, started, stopped = _mgr(act, [2403], cooldown=100.0)
    m.tick()
    assert len(started) == 1
    clock.advance(150); m.tick()            # 1 failure ⇒ cooldown 200 — too soon
    assert len(started) == 1
    clock.advance(100); m.tick()            # 250 > 200 ⇒ second try
    assert len(started) == 2
    clock.advance(350); m.tick()            # 2 failures ⇒ 400 — too soon
    assert len(started) == 2
    clock.advance(100); m.tick()            # 450 > 400 ⇒ third try
    assert len(started) == 3
    # …and a server that finally SERVES resets the backoff entirely.
    act[0] = {"served": True, "running": 1, "finished": 1}
    m.tick()
    assert m._failed_starts == 0
