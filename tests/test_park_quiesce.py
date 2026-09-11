"""THE PARK WATCHDOG — a parked box's broker-managed LLM servers are KILLED.

Operator rule #3: "Create watchdog that kills any broker_parrot
servers if box is parked." Unconditional — no idle TTL, no activity exemption.

Why the first cut wasn't enough: rule #2's quiesce reused the idle reapers,
which spare a BUSY server (peer safety). Found live within hours: REMOTE
traffic — box-c POSTing chat completions over the fast iface — kept parked
box-a's vllm "busy" at 96 % GPU indefinitely (9 concurrent requests, all from
a peer on the private interconnect), and an out-of-band `docker run` had recreated the container
mid-park (a fresh CreatedAt, RestartCount reset — the engine's lever can
only START an existing container, so no engine path did that). An
activity-guarded parked box is a box that anyone can keep un-parked from the
outside.

Parked now means: NO LLM service on this box, full stop. A remote caller gets
connection-refused — exactly the pressure that fails it over to an ON box. The
watchdog kills on sight and keeps killing resurrections (the vllm stop lever is
a real `docker stop`, which `restart: unless-stopped` respects; ollama models
are unloaded with keep_alive:0, daemon left running — it is shared box infra).
"""
from __future__ import annotations

from queue_workflows import claim_worker, llm_idle_unload as idle


def _watchdog(activity, infos):
    stopped, unloaded = [], []
    q = idle.ParkQuiesce(
        vllm_activity_fn=lambda: activity[0],
        vllm_stop_fn=lambda: stopped.append(True),
        ollama_models_fn=lambda: infos[0],
        ollama_unload_fn=lambda models: unloaded.append(list(models)),
        label="park-test",
    )
    return q, stopped, unloaded


# ── vLLM half: kill on sight ─────────────────────────────────────────────────


def test_a_parked_vllm_is_stopped_ON_SIGHT_even_mid_request():
    """THE box-a regression: 9 concurrent remote requests
    kept a parked box serving at 96 % GPU. Busy is not an exemption — a parked
    box has no LLM service to offer, whoever is calling."""
    act = [{"served": True, "running": 9, "finished": 1234}]
    q, stopped, _ = _watchdog(act, [[]])
    q.tick()
    assert stopped == [True], "a serving vLLM on a parked box dies THIS tick"


def test_a_not_serving_vllm_is_a_noop():
    act = [{"served": False, "running": 0, "finished": 0}]
    q, stopped, unloaded = _watchdog(act, [[]])
    for _ in range(5):
        q.tick()
    assert stopped == [] and unloaded == []


def test_a_resurrected_vllm_is_stopped_every_time_it_reappears():
    """The `unless-stopped` fight, and the out-of-band `docker run` fight: every
    reappearance is met with another stop, for as long as the box is parked."""
    act = [{"served": True, "running": 0, "finished": 0}]
    q, stopped, _ = _watchdog(act, [[]])
    q.tick()
    act[0] = {"served": False, "running": 0, "finished": 0}
    q.tick()
    act[0] = {"served": True, "running": 3, "finished": 9}   # resurrected + busy
    q.tick()
    assert stopped == [True, True], "every resurrection is re-killed"


# ── ollama half: unload on sight ─────────────────────────────────────────────


def test_parked_ollama_models_are_unloaded_on_sight_even_in_active_use():
    """No mru grace: rule #2 kept an actively-used model for a peer; rule #3
    says a parked box serves NOBODY. The daemon itself survives (shared box
    infra, ~no RAM); only the models are unloaded."""
    infos = [[{"name": "chat-model-a:latest", "mru": 111.0},
              {"name": "nomic-embed", "mru": 222.0}]]
    q, _, unloaded = _watchdog([{"served": False, "running": 0,
                                 "finished": 0}], infos)
    q.tick()
    assert unloaded == [["chat-model-a:latest", "nomic-embed"]], \
        "every loaded model dies on sight, active or not"


def test_an_empty_ollama_is_a_noop():
    q, stopped, unloaded = _watchdog(
        [{"served": False, "running": 0, "finished": 0}], [[]])
    q.tick()
    assert unloaded == [] and stopped == []


def test_a_raising_probe_never_crashes_the_tick():
    def _boom():
        raise RuntimeError("server unreachable")
    q = idle.ParkQuiesce(
        vllm_activity_fn=_boom, vllm_stop_fn=lambda: None,
        ollama_models_fn=_boom, ollama_unload_fn=lambda m: None,
        label="park-test",
    )
    q.tick()        # must not raise — it lives on the park gate's loop


def test_one_half_raising_does_not_starve_the_other():
    """The vllm probe failing must not skip the ollama unload (and vice
    versa) — each half is independently best-effort."""
    def _boom():
        raise RuntimeError("vllm probe down")
    unloaded = []
    q = idle.ParkQuiesce(
        vllm_activity_fn=_boom, vllm_stop_fn=lambda: None,
        ollama_models_fn=lambda: [{"name": "chat-model-a:latest", "mru": 1.0}],
        ollama_unload_fn=lambda m: unloaded.append(list(m)),
        label="park-test",
    )
    q.tick()
    assert unloaded == [["chat-model-a:latest"]]


# ── the claim-worker builder (unchanged contract) ────────────────────────────


def _bare(queue: str):
    class _W(claim_worker.ClaimWorker):
        def __init__(self):     # bypass the heavy real __init__
            self.queue = queue
            self.host = f"{queue}-park-build-test"
    return _W()


def test_builder_is_gpu_lane_only():
    """A parked CPU worker must not touch the box's LLM servers — the card
    belongs to the gpu lane."""
    assert _bare("cpu")._build_park_quiesce() is None


def test_builder_honors_the_shared_kill_switch(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER", "1")
    assert _bare("gpu")._build_park_quiesce() is None


def test_builder_arms_for_a_gpu_worker(monkeypatch):
    # The suite's conftest silences the idle reapers globally; the builder must
    # honour that (previous test) — so prove the ARMED case with it cleared.
    monkeypatch.delenv("QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER", raising=False)
    monkeypatch.delenv("AI_LEADS_DISABLE_GPU_IDLE_REAPER", raising=False)
    tick = _bare("gpu")._build_park_quiesce()
    assert callable(tick), "a gpu worker's park gate must get a watchdog tick"
