"""THE PARK WATCHDOG — a parked box's broker-managed LLM servers are KILLED.

The rule: a watchdog kills any broker-managed LLM server on a parked box.
Unconditional — no idle TTL, no activity exemption.

Why an activity-guarded quiesce isn't enough: reusing the idle reapers spares a
BUSY server (peer safety). REMOTE traffic — a peer POSTing chat completions over
the fast interface — can keep a parked box's vllm "busy" at full GPU
indefinitely (N concurrent requests, all from a peer on the private
interconnect), and an out-of-band `docker run` had recreated the container
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
    """THE regression: N concurrent remote requests
    kept a parked box serving at full GPU. Busy is not an exemption — a parked
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
    monkeypatch.delenv("QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER", raising=False)
    tick = _bare("gpu")._build_park_quiesce()
    assert callable(tick), "a gpu worker's park gate must get a watchdog tick"


# ── THE BLIND SPOT and THE RESTART ───────────────────────────────────────────
#
# A lane begins a vLLM switch and is parked a minute later. A watchdog that stops vLLM only once
# it answers /metrics is blind to a server still loading its weights (it answers no HTTP probe),
# so for minutes a parked box goes on loading a large model. And seconds after that stop the SAME
# container can be started again while the lane is still parked (RestartCount 0) — the switch,
# still WARMING, re-issues its start on its 30 s retry.
# A park now reads the CONTAINER, and ends the switch it finds.

from tests.test_cluster_reservation import _box_b, _warm_up, box  # noqa: E402,F401


def test_the_watchdog_stops_a_vllm_that_is_still_loading_by_container_state():
    stopped = []
    q = idle.ParkQuiesce(
        vllm_activity_fn=lambda: {"served": False, "running": 0, "finished": 0},
        vllm_stop_fn=lambda: stopped.append(True),
        vllm_running_fn=lambda: True,
        ollama_models_fn=lambda: [], ollama_unload_fn=lambda m: None, label="box-c-gpu",
    )
    assert q.tick() == (True, [])
    assert stopped == [True], "a loading vLLM ran on because /metrics was silent"


def test_a_stopped_container_that_answers_nothing_is_left_alone():
    stopped = []
    q = idle.ParkQuiesce(
        vllm_activity_fn=lambda: {"served": False}, vllm_stop_fn=lambda: stopped.append(True),
        vllm_running_fn=lambda: False,
        ollama_models_fn=lambda: [], ollama_unload_fn=lambda m: None, label="park-test",
    )
    for _ in range(3):
        q.tick()
    assert stopped == []


def test_an_unreadable_container_falls_back_to_the_metrics_probe():
    """No docker socket reachable: the watchdog is exactly as good as it was, never worse."""
    def _no_socket():
        raise FileNotFoundError("/var/run/docker.sock")

    for served, expected in ((True, [True]), (False, [])):
        stopped = []
        q = idle.ParkQuiesce(
            vllm_activity_fn=lambda s=served: {"served": s, "running": 1},
            vllm_stop_fn=lambda: stopped.append(True), vllm_running_fn=_no_socket,
            ollama_models_fn=lambda: [], ollama_unload_fn=lambda m: None, label="park-test",
        )
        q.tick()
        assert stopped == expected


def test_a_drained_job_still_holds_off_a_running_container():
    """THE ONE EXEMPTION (test_park_quiesce_drain.py) is unchanged by reading the container."""
    stopped = []
    q = idle.ParkQuiesce(
        vllm_activity_fn=lambda: {"served": False}, vllm_stop_fn=lambda: stopped.append(True),
        vllm_running_fn=lambda: True, in_flight_fn=lambda: 1,
        ollama_models_fn=lambda: [], ollama_unload_fn=lambda m: None, label="park-test",
    )
    q.tick()
    assert stopped == []


def test_a_park_stops_the_loading_vllm_of_a_switch_in_flight(box, monkeypatch):
    w, parked, _ = _box_b(monkeypatch)
    _warm_up(w, box)
    parked[0] = True
    w._build_park_quiesce()()
    assert (box.vllm_running, box.stops) == (False, 1), "the park could not see a loading vLLM"


def test_a_park_ends_the_warm_up_so_nothing_starts_the_server_again(box, monkeypatch):
    """THE RESTART, replayed: loading, parked, answering at last, stopped — and then the
    switch's retry must not start it again."""
    from queue_workflows import server_switch

    w, parked, _ = _box_b(monkeypatch)
    _warm_up(w, box)
    box.vllm_answers = True                  # at last it answers /metrics
    parked[0] = True
    tick = w._build_park_quiesce()
    tick()
    assert box.stops == 1 and not box.vllm_running
    for _ in range(4):                       # a drain-park keeps its heartbeat, and its switch tick
        w.clock.advance(31)
        w._drive_server_switch()
        tick()
    assert box.starts == 1, "a PARKED box started its vLLM again: the warm-up outlived the park"
    assert w._switch.state == server_switch.IDLE
    assert w.request_server_switch("vllm") is False, "a parked box accepted a new switch"
    assert box.starts == 1


def test_the_heartbeat_ends_a_warm_up_on_a_parked_box_before_the_watchdog_ticks(box, monkeypatch):
    from queue_workflows import server_switch

    w, parked, _ = _box_b(monkeypatch)
    _warm_up(w, box)
    parked[0] = True
    w.clock.advance(31)
    w._drive_server_switch()
    assert w._switch.state == server_switch.IDLE
    assert box.starts == 1, "the switch re-issued its start on a parked box"
    assert box.vllm_running is False, "the loading vLLM of the abandoned switch was left running"


def test_a_persisted_switch_does_not_outlive_a_boot_park(box, monkeypatch, tmp_path):
    """"Never resume": a lane that restarts parked inherits the persisted switch (a restart
    re-enters it), and the park gate runs before the heartbeat could ever drive it."""
    import os

    from queue_workflows import server_switch

    path = str(tmp_path / "server_switch_box-b-gpu.json")
    assert server_switch.ServerSwitch(label="box-c-gpu", persist_path=path).request("vllm")
    w, parked, _ = _box_b(monkeypatch)
    w._switch = server_switch.ServerSwitch(label="box-c-gpu", persist_path=path)
    assert w._switch.state == server_switch.DRAINING, "precondition: the restart resumed the switch"
    parked[0] = True
    w._build_park_quiesce()()
    assert w._switch.state == server_switch.IDLE
    assert not os.path.exists(path), "the persisted switch would resume at the next restart"
