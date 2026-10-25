"""A CLUSTER RESERVATION — a gpu node-job that holds one box for a server the engine did not start.

The design rule: a cluster never interferes with parks anywhere in the vLLM service; it adds a
node-job with priority and waits for the engine to take it.

llm-gateway runs a TP=2 vLLM across several boxes in containers of its own (``cluster-vllm``).
It used to reserve the boxes by PARKING their gpu lanes — the operator's lever, which it must
never touch. The reservation is now an ordinary node-job, one per box:

    enqueue_node_job(queue='gpu', node_module='system.vllm_lease', server_types=['cluster'],
                     force_box=[<box>], gpu_slots='all', priority=0)   + prioritize_node_job

and the engine owes it five things:

  (a) the pinned box claims it although no box SERVES 'cluster' — at once on an idle box, after a
      drain on a busy one, and even mid-switch — and the claim starts no LLM server;
  (b) while it runs, the lane keeps the broker's OWN servers down: the ``vllm`` container is
      stopped by CONTAINER STATE (a vLLM still loading answers no /metrics), every ollama model
      is unloaded, a switch that began before the claim is abandoned, and nothing starts either
      server again until the job ends;
  (c) parks stay the operator's: nothing here writes the control plane, and a parked lane claims
      nothing (the park gate sits in front of every claim);
  (d) the external ranks survive the rogue-LLM sweep and the memory squeeze;
  (e) a re-queued reservation is re-claimed first by its pinned box.

WHY (b) IS BY CONTAINER STATE: two lanes began a vLLM switch and were parked about a minute
later. The park watchdog could not see the loading server until it answered /metrics minutes
after that, stopped it — and seconds later the still-WARMING switch re-issued its start (a fresh
StartedAt, RestartCount 0). tests/test_park_quiesce.py pins the park half.
"""
from __future__ import annotations

import inspect
import sys
import types
from datetime import datetime, timezone

import pytest

from queue_workflows import node_queue
from tests._helpers import force_lease, make_run


# ═══ the queue: what the claim, the sweeps and the demand count see (SQLite + Postgres) ═══════


def _reservation(box: str = "box-c", *, run: str | None = None, **kw) -> str:
    """A reservation exactly as llm-gateway writes one: the contract in the module docstring."""
    run = run or make_run(workflow_name="vllm_lb_cluster", out_dir=None)
    job_id = node_queue.enqueue_node_job(
        run_id=run, node_id=f"cluster-{box}", node_module="system.vllm_lease", queue="gpu",
        server_types=["cluster"], force_box=[box], gpu_slots="all", priority=0,
        inputs={"lease_id": run, "purpose": "llm-gateway/cluster#7", "model": "chat-model-a:latest",
                "max_hold_s": 86400},
        **kw)
    assert node_queue.prioritize_node_job(job_id) is not None
    return job_id


def _fleet(node: str, *, run: str | None = None, servers=("vllm",)) -> str:
    return node_queue.enqueue_node_job(
        run_id=run or make_run(), node_id=node, node_module="llm_chat", queue="gpu",
        server_types=list(servers))


def _pool_claim(host="box-c-gpu", box="box-c", served=("ollama",), **kw):
    """The pool lane's own claim arguments on a GB10 (PAR 3, the per-kind caps it passes)."""
    args = dict(host=host, max_running=3, box=box, require_model=False,
                served_types=list(served), kind_caps={"ollama": 1, "vllm": 3},
                serving_context=None)
    args.update(kw)
    return node_queue.claim_next_gpu_job(0, **args)


@pytest.mark.parametrize("served", [("ollama",), ("vllm",), ()])
def test_the_pinned_box_claims_a_reservation_no_box_serves(served):
    """(a) No box will ever advertise a 'cluster' server — the server is the LB's, not ours. The
    server-preference gate must not read that as "nobody can take it"."""
    job_id = _reservation("box-c")
    got = _pool_claim(served=served)
    assert got is not None and got["id"] == job_id, f"a box serving {served} refused its reservation"
    assert got["resolved_server"] == node_queue.CLUSTER_SERVER_KIND
    assert got["claimed_by"] == "box-c-gpu"


def test_no_other_box_and_no_inline_lane_takes_a_pinned_reservation():
    """The over-reach guard: the reservation is for ONE box."""
    _reservation("box-c")
    assert _pool_claim(host="box-b-gpu", box="box-b", served=()) is None
    assert _pool_claim(host="box-b-gpu", box="box-b", served=("vllm",)) is None
    assert node_queue.claim_next_gpu_job(
        0, host="box-c-gpu", max_running=3, box="box-c", require_model=True,
        served_types=["ollama"], pool_modules=["llm_chat"]) is None, "the inline lane took it"


def test_a_busy_box_drains_for_the_reservation_then_takes_it():
    """(a) on a BUSY box: nothing narrower is admitted while the reservation waits, the box drains,
    the reservation takes the whole box, and nothing fits beside it."""
    run = make_run()
    running = []
    for i in range(2):
        _fleet(f"fleet{i}", run=run)
        running.append(_pool_claim(served=("vllm",)))
    assert all(running)
    job_id = _reservation("box-c")
    _fleet("refill", run=run)
    assert _pool_claim(served=("vllm",)) is None, "a refill was admitted beside a waiting reservation"
    for row in running:
        node_queue.mark_completed(row["id"], context_delta={}, seconds=1.0)
    got = _pool_claim(served=("vllm",))
    assert got is not None and got["id"] == job_id
    assert _pool_claim(served=("vllm",)) is None, "work was admitted beside a held reservation"


def test_mid_switch_the_pool_lane_takes_a_reservation_and_nothing_else():
    """A switch takes the box out of service because a claim would reload the outgoing server. A
    reservation loads nothing and ends the switch, so it is the one claim a switching box makes."""
    run = make_run()
    _fleet("ordinary", run=run, servers=("ollama",))
    assert _pool_claim(reservations_only=True) is None, "ordinary work was claimed mid-switch"
    job_id = _reservation("box-c")
    got = _pool_claim(reservations_only=True)
    assert got is not None and got["id"] == job_id


def test_the_lb_contract_round_trip_release_while_queued():
    """The release the LB already uses for every lease: the run cancelled, its queued jobs
    cancelled. The box goes back to work on its very next claim."""
    from queue_workflows import run_store

    run = make_run(workflow_name="vllm_lb_cluster", out_dir=None)
    _fleet("busy")
    assert _pool_claim(served=("vllm",)) is not None
    job_id = _reservation("box-c", run=run)
    _fleet("waiting")
    assert _pool_claim(served=("vllm",)) is None, "the reservation did not hold the box"
    row = node_queue.get_node_job(job_id)
    assert (row["status"], row["claimed_by"]) == ("queued", None)

    run_store.update_run(run, status="cancelled", finished_at=datetime.now(timezone.utc))
    assert node_queue.cancel_queued_jobs_for_run(run) == 1
    assert node_queue.get_node_job(job_id)["status"] == "cancelled"
    got = _pool_claim(served=("vllm",))
    assert got is not None and got["node_id"] == "waiting", "the released box did not resume"


@pytest.mark.parametrize("requeue", ["lane_restart", "lease_lapsed", "operator_stop"])
def test_a_requeued_reservation_is_reclaimed_first_by_its_box(requeue):
    """(e) A lane restart, a lapsed lease or an operator hard-stop puts the running reservation back
    in the queue. It keeps its pin, its weight and the run-next flag, and its box takes it back
    before anything else."""
    job_id = _reservation("box-c")
    got = _pool_claim()
    assert got is not None and got["id"] == job_id
    _fleet("fleet", servers=("ollama",))
    if requeue == "lane_restart":
        assert node_queue.requeue_running_if_worker_stale("box-c-gpu", "gpu") == 1
    elif requeue == "lease_lapsed":
        force_lease(job_id, expires_in_s=-5)
        assert [r["id"] for r in node_queue.reclaim_expired_leases()] == [job_id]
    else:
        assert node_queue.requeue_running_for_worker("box-c-gpu", "gpu") == 1
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued" and row["is_priority"]
    assert list(row["force_box"]) == ["box-c"] and row["gpu_slots"] == "all"
    assert _pool_claim(host="box-b-gpu", box="box-b", served=()) is not None, "box-b took the fleet row"
    _fleet("fleet2", servers=("ollama",))
    again = _pool_claim()
    assert again is not None and again["id"] == job_id, "the fleet row jumped the re-queued reservation"


def test_the_stranded_sweep_never_flags_a_reservation():
    """No live worker serves 'cluster' and none ever will; a reservation needs no server. The sweep
    that names rows no server can take must not paint it red — while it still names a real one."""
    node_queue.upsert_worker_heartbeat(host_label="box-c-gpu", queue="gpu", concurrency=3,
                                       llm_servers_available=["ollama"])
    job_id = _reservation("box-c")
    stranded = _fleet("needs-vllm")
    assert [r["id"] for r in node_queue.flag_server_stranded_jobs()] == [stranded]
    assert node_queue.get_node_job(job_id)["unassignable_at"] is None


def test_pending_vllm_demand_counts_only_rows_this_box_could_run():
    """"One vLLM lease wakes every box": the demand starter counted every queued vLLM row on every
    box. A row pinned elsewhere, or avoiding this box, can never run here and must not start a
    server here. (An UNPLACED row still counts on every capable box — see the report.)"""
    run = make_run()
    for node, kw in (("anywhere", {}), ("pinned-box-b", {"force_box": ["box-b"]}),
                     ("avoids-box-c", {"avoid_box": ["box-c"]})):
        node_queue.enqueue_node_job(run_id=run, node_id=node, node_module="system.vllm_lease",
                                    queue="gpu", server_types=["vllm"], **kw)
    _reservation("box-c")
    assert node_queue.count_pending_for_server(None, "vllm", any_model=True) == 3, (
        "the unplaced count (the dispatch-time amortisation read) must not change")
    assert node_queue.count_pending_for_server(None, "vllm", any_model=True, box="box-c") == 1
    # box-b could run all three: the unplaced row, the row pinned to it, and the row that
    # avoids box-c.
    assert node_queue.count_pending_for_server(None, "vllm", any_model=True, box="box-b") == 3


def test_a_reservation_is_visible_to_its_box_and_only_while_it_is_live():
    """The read behind "must this box start nothing?": a reservation QUEUED for this box, or
    RUNNING on this lane — never one pinned elsewhere, never a released one."""
    run = make_run(workflow_name="vllm_lb_cluster", out_dir=None)
    job_id = _reservation("box-c", run=run)
    seen = node_queue.cluster_reservation_for(host="box-c-gpu", box="box-c")
    assert seen is not None and (seen["id"], seen["status"]) == (job_id, "queued")
    assert node_queue.cluster_reservation_for(host="box-b-gpu", box="box-b") is None
    assert _pool_claim()["id"] == job_id
    seen = node_queue.cluster_reservation_for(host="box-c-gpu", box="box-c")
    assert seen is not None and seen["status"] == "running"
    assert node_queue.cluster_reservation_for(host="box-b-gpu", box="box-b") is None

    other = make_run(workflow_name="vllm_lb_cluster", out_dir=None)
    _reservation("box-b", run=other)
    from queue_workflows import run_store
    run_store.update_run(other, status="cancelled", finished_at=datetime.now(timezone.utc))
    node_queue.cancel_queued_jobs_for_run(other)
    assert node_queue.cluster_reservation_for(host="box-b-gpu", box="box-b") is None


def test_the_executor_resolves_no_llm_server_for_a_reservation():
    """(a) "the claim starts no broker LLM server": resolving an LLM server for a row is how a
    dispatch reaches a backend (building one arms vLLM's supervisor), and for ['cluster'] it could
    only fall back to the box default and log a violated preference. A reservation has no server."""
    import queue_workflows
    from queue_workflows import node_executor

    queue_workflows.set_node_module_package("qwf_cr_nodes")
    seen: dict = {}

    def run(*, inputs=None, out=None, llm_server="not passed", cancel_event=None):
        seen["llm_server"] = llm_server
        return {"context_delta": {}}

    mod = types.ModuleType("qwf_cr_nodes.hold")
    mod.run = run
    sys.modules["qwf_cr_nodes.hold"] = mod
    run_id = make_run(workflow_name="vllm_lb_cluster", out_dir=None)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="cluster-box-c", node_module="hold", queue="gpu",
        server_types=["cluster"], force_box=["box-c"], gpu_slots="all", priority=0)

    def _resolver(job):
        raise AssertionError("a cluster reservation asked for an LLM server")

    queue_workflows.set_llm_server_resolver(_resolver)
    try:
        result = node_executor.execute_node(node_queue.get_node_job(job_id))
    finally:
        queue_workflows.set_llm_server_resolver(None)
    assert result == "completed"
    assert seen["llm_server"] is None


# ═══ the lane: holding the box (pure, with the box's levers faked) ════════════════════════════


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class _Box:
    """box-c as the engine's levers see it: the broker's own ``vllm`` container and ollama."""

    def __init__(self):
        self.vllm_running = False      # docker: the container is up (loading OR serving)
        self.vllm_answers = False      # vLLM answers HTTP (/metrics, /v1/models): loaded
        self.ollama: list[str] = []
        self.starts = 0
        self.stops = 0

    def start(self, _model=None):
        self.starts += 1
        self.vllm_running = True
        return {"status": "started", "reason": "ok"}

    def stop(self):
        self.stops += 1
        self.vllm_running = False
        self.vllm_answers = False
        return True


@pytest.fixture
def box(monkeypatch):
    from queue_workflows import llm_context_lever, llm_probe, rogue_llm
    from queue_workflows.config import get_config
    from queue_workflows.llm_backends import factory

    b = _Box()
    # The suite silences the engine's LLM-touching daemons; these tests are about exactly them.
    monkeypatch.delenv("QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER", raising=False)
    monkeypatch.delenv("QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER", raising=False)
    llm_context_lever.reset_cache_for_tests()
    monkeypatch.setattr(factory, "resolve_base_url", lambda kind="ollama": f"http://{kind}:1")
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: llm_context_lever.ServerContainer(
        kind=kind, name=kind, exists=True, running=(b.vllm_running if kind == "vllm" else True)))
    monkeypatch.setattr(llm_probe, "vllm_activity",
                        lambda url, **k: {"served": b.vllm_answers, "running": 0, "finished": 0})
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: (
        ["vllm"] if "vllm" in str(url) and b.vllm_answers
        else ["ollama"] if "ollama" in str(url) else []))
    monkeypatch.setattr(llm_probe, "vllm_served_models",
                        lambda url, **k: ["chat-model-b:latest"] if b.vllm_answers else [])
    monkeypatch.setattr(llm_probe, "loaded_models", lambda url, **k: list(b.ollama))
    monkeypatch.setattr(llm_probe, "loaded_models_info",
                        lambda url, **k: [{"name": n, "mru": 1.0} for n in b.ollama])

    def _unload(url, models, **k):
        b.ollama = [m for m in b.ollama if m not in set(models)]
        return list(models)

    monkeypatch.setattr(llm_probe, "unload_ollama_models", _unload)
    monkeypatch.setattr(rogue_llm, "sweep", lambda **k: [])
    cfg = get_config()
    monkeypatch.setattr(cfg, "vllm_start_fn", b.start)
    monkeypatch.setattr(cfg, "vllm_stop_fn", b.stop)
    return b


def _box_b(monkeypatch):
    """A gpu worker for box-c with every observation injectable: ``parked[0]`` is the control
    plane's word, ``reservation[0]`` what the queue says is reserved for this box."""
    from queue_workflows import claim_worker, server_switch, worker_control

    clock = _Clock()

    class _W(claim_worker.ClaimWorker):
        def __init__(self):     # bypass the heavy real __init__
            self.host = "box-c-gpu"
            self.queue = "gpu"
            self._switch = server_switch.ServerSwitch(label="box-c-gpu", now_fn=clock)

    w = _W()
    w.clock = clock
    w._box_name = "box-c"
    w._pool_inflight = 0
    w._inline_running = False
    w._box_held_model = None
    w._box_active_jobs = 0
    w._get_box_lease = lambda: None
    w._box_memory_in_danger = lambda: False
    w._vllm_start_decision = lambda: types.SimpleNamespace(
        utilization=None, max_num_seqs=None, refusal=None)
    parked, reservation = [False], [None]
    monkeypatch.setattr(worker_control, "desired_state_for",
                        lambda host, queue, **k: "off" if parked[0] else "on")
    monkeypatch.setattr(node_queue, "cluster_reservation_for",
                        lambda **k: reservation[0], raising=False)
    return w, parked, reservation


def _warm_up(w, box):
    """A switch to vLLM, begun before anything reserved the box, now loading weights."""
    from queue_workflows import server_switch

    assert w._switch.request("vllm")
    for _ in range(3):
        w._drive_server_switch()
        w.clock.advance(1)
    assert w._switch.state == server_switch.WARMING
    assert box.starts == 1 and box.vllm_running and not box.vllm_answers


def _held_row(job_id="c0ffee00-cluster"):
    return {"id": job_id, "run_id": "run-1", "node_id": "cluster-box-c",
            "node_module": "system.vllm_lease", "queue": "gpu", "required_model": None,
            "server_types": ["cluster"], "resolved_server": "cluster", "force_box": ["box-c"],
            "gpu_slots": "all", "inputs": {"purpose": "llm-gateway/cluster#7"}}


def test_a_hold_ends_a_switch_that_began_before_the_claim_and_keeps_the_servers_down(box, monkeypatch):
    """(b), the coordinator's replay: the switch was WARMING a vLLM when the reservation landed. The
    hold ends the switch, stops the loading server by container state, unloads ollama, refuses every
    start while it lasts — and hands the box back when it ends."""
    from queue_workflows import server_switch

    w, _parked, _reservation_ = _box_b(monkeypatch)
    _warm_up(w, box)
    box.ollama = ["chat-model-a:latest"]
    hold = w._begin_cluster_hold(_held_row())
    w._cluster_hold_tick()
    assert w._switch.state == server_switch.IDLE, "the switch outlived the claim"
    assert (box.vllm_running, box.stops) == (False, 1), "a vLLM still loading was left running"
    assert box.ollama == [], "ollama kept its model on a held box"

    for _ in range(4):                       # the heartbeat keeps ticking the whole time
        w.clock.advance(31)
        w._drive_server_switch()
        w._cluster_hold_tick()
    assert box.starts == 1, "the broker's vLLM was started while the box was held"
    assert w.request_server_switch("vllm") is False, "a held box accepted a switch"
    assert w._llm_starts_forbidden()

    box.vllm_running = True                  # out of band: someone starts it anyway…
    box.ollama = ["chat-model-a:latest"]          # …and a remote caller loads a model
    w._cluster_hold_tick()
    assert box.vllm_running is False and box.ollama == [], "a resurrection survived a held tick"

    w._end_cluster_hold(hold)
    assert w._llm_starts_forbidden() is None
    assert w.request_server_switch("vllm") is True, "normal service did not resume after the hold"


def test_a_reservation_queued_for_this_box_ends_a_switch_and_refuses_new_ones(box, monkeypatch):
    """(b), before the claim: a box queued for the cluster is about to give itself up — a switch it
    is still draining for must not go on to start a server."""
    from queue_workflows import server_switch

    w, _parked, reservation = _box_b(monkeypatch)
    assert w._switch.request("vllm")
    reservation[0] = {"id": "r-1", "status": "queued"}
    w._drive_server_switch()
    assert w._switch.state == server_switch.IDLE
    w.clock.advance(31)
    w._drive_server_switch()
    assert box.starts == 0
    assert w.request_server_switch("vllm") is False


def test_the_demand_starter_is_shut_while_a_reservation_is_queued_or_held(box, monkeypatch):
    """(b) "starts neither again": the idle stopper's START half answers to the same rule, and its
    demand count is judged by this box's placement."""
    from queue_workflows import llm_idle_unload

    monkeypatch.setattr(llm_idle_unload.VLLMIdleStopper, "ensure_started", lambda self: None)
    monkeypatch.setattr(llm_idle_unload.ExternalModelIdleReaper, "ensure_started", lambda self: None)
    w, _parked, reservation = _box_b(monkeypatch)
    seen: dict = {}
    monkeypatch.setattr(node_queue, "count_pending_for_server",
                        lambda *a, **k: seen.update(k) or 5)
    w._gpu_active_count = lambda: 0
    w._arm_llm_idle_reaper()
    stopper = w._vllm_idle_stopper
    assert stopper._may_start_fn() is True
    reservation[0] = {"id": "r-1", "status": "queued"}
    assert stopper._may_start_fn() is False, "a box queued for the cluster may still start a vLLM"
    reservation[0] = None
    hold = w._begin_cluster_hold(_held_row())
    assert stopper._may_start_fn() is False, "a held box may still start a vLLM"
    w._end_cluster_hold(hold)
    assert stopper._may_start_fn() is True
    stopper._pending_fn()
    assert seen.get("box") == "box-c", "pending vLLM demand ignored this box's placement"


def test_a_lane_that_restarts_under_a_reservation_starts_no_inference_server(box, monkeypatch):
    """(e) with (b): a restarted lane reaches its boot-time server start before it re-claims."""
    from queue_workflows.config import get_config

    w, _parked, reservation = _box_b(monkeypatch)
    started = []
    monkeypatch.setattr(get_config(), "inference_server_start_fn", lambda: started.append(True))
    reservation[0] = {"id": "r-1", "status": "running"}     # its dead predecessor's claim
    w._start_inference_server()
    assert started == []
    reservation[0] = None
    w._start_inference_server()
    assert started == [True]


def test_the_pool_lane_takes_a_reservation_mid_switch_or_with_no_llm_server(monkeypatch):
    """(a) The pool lane stops claiming mid-switch and when no GPU LLM server answers — right for
    LLM work, wrong for a reservation, which needs no server and ends the switch. It claims, and
    claims ONLY reservations, when one is queued for this box."""
    from queue_workflows import claim_worker, model_registry

    calls: list[dict] = []
    monkeypatch.setattr(node_queue, "claim_next_gpu_job", lambda *a, **k: calls.append(k) or None)
    monkeypatch.setattr(model_registry, "known_ids", lambda: [])
    reservation = [None]
    monkeypatch.setattr(node_queue, "cluster_reservation_for",
                        lambda **k: reservation[0], raising=False)
    w = claim_worker.ClaimWorker(queue="gpu", host="box-c-gpu")
    w._box_name = "box-c"
    w._pool_gpu_serving_ok = lambda: True
    w._served_llm_types = lambda: ["ollama"]
    assert w._switch.request("vllm")
    assert w._claim_pool() is None and calls == [], "a switching box with nothing reserved claimed"
    reservation[0] = {"id": "r-1", "status": "queued"}
    w._claim_pool()
    assert calls and calls[-1].get("reservations_only") is True, "mid-switch: no reservation claim"
    w._switch.abandon("test")
    calls.clear()
    w._pool_gpu_serving_ok = lambda: False
    w._claim_pool()
    assert calls and calls[-1].get("reservations_only") is True, "no LLM server: no reservation claim"
    calls.clear()
    w._pool_gpu_serving_ok = lambda: True
    w._claim_pool()
    assert calls and not calls[-1].get("reservations_only"), "a healthy box narrowed its claim"


def test_a_reservation_is_its_own_occupant_of_the_card(tmp_path):
    """(b) across projects: the box lease is how every project's workers on one card exclude each
    other. A reservation is neither the LLM server nor a ComfyUI render (the GB10s serve both), so
    it takes a slot of its own — and while it holds the card nobody else's occupant is granted."""
    from queue_workflows import claim_worker
    from queue_workflows import gpu_model_lease as gml

    w = claim_worker.ClaimWorker(queue="gpu", host="box-c-gpu")
    w._box_serves_comfyui = lambda: True
    slot = w._effective_gpu_model(_held_row())
    assert slot == gml.CLUSTER_RESERVATION_SLOT
    assert slot not in (gml.LLM_SERVER_SLOT, gml.COMFYUI_SERVER_SLOT)
    assert gml.is_sentinel_slot(slot) and not gml.is_llm_slot(slot)
    assert gml.describe_slot(slot) == "cluster reservation"

    def lease(holder):
        return gml.ModelLease(box_id="box-c", store=gml.FileLeaseStore(str(tmp_path)),
                              holder=holder, ttl_s=60, now_fn=lambda: 1000.0)

    assert lease("content-app").acquire(gml.CLUSTER_RESERVATION_SLOT) is True
    for other in (gml.LLM_SERVER_SLOT, gml.COMFYUI_SERVER_SLOT, "sdxl"):
        assert lease("app-a").acquire(other) is False, f"{other} was granted on a held card"


_BOX_ALLOW = ("ollama,vllm,content-worker,box-worker,crm-worker,batch-worker,"
                "hw-broadcast,cluster-vllm")          # a typical gpu box's allow-list


def _docker(containers, stopped):
    listing = [c for c, _ in containers]
    inspected = {c["Id"]: i for c, i in containers}

    def api(method, path, body=None, **_k):
        if method == "GET" and path == "/containers/json":
            return 200, listing
        if method == "GET" and path.endswith("/json"):
            return 200, inspected[path.split("/")[2]]
        if method == "POST" and path.split("?")[0].endswith("/stop"):
            stopped.append(path.split("/")[2])
            return 204, None
        if method == "POST" and path.endswith("/update"):
            return 200, {}
        return 404, {"message": "no such container"}

    return api


def _ctr(cid, name, image, *, restart="no"):
    return ({"Id": cid, "Names": [f"/{name}"], "Image": image, "State": "running", "Labels": {}},
            {"Id": cid, "HostConfig": {"DeviceRequests": [{"Capabilities": [["gpu"]]}],
                                       "RestartPolicy": {"Name": restart}}})


_BOX_WHILE_HELD = [
    _ctr("r0", "cluster-vllm", "registry.example.com/chat-model-a-cluster-vllm:v1"),
    _ctr("o1", "ollama", "ollama/ollama:0.33.2", restart="unless-stopped"),
    _ctr("x9", "someones-trainer", "pytorch/pytorch:2.8.0", restart="no"),
]


def test_the_ranks_survive_the_rogue_sweep(monkeypatch):
    """(d) The rank is not the broker's vLLM image and it is on the box's allow-list."""
    from queue_workflows import rogue_llm

    rogue_llm.reset_for_tests()
    monkeypatch.setenv(rogue_llm.ALLOW_ENV, _BOX_ALLOW)
    stopped: list[str] = []
    assert rogue_llm.image_kind("registry.example.com/chat-model-a-cluster-vllm:v1") is None
    assert rogue_llm.sweep(api_fn=_docker(_BOX_WHILE_HELD, stopped), host="box-c-gpu") == []
    assert stopped == []


def test_a_memory_squeeze_while_held_spares_the_ranks(box, monkeypatch):
    """(d) The ranks load most of the box's memory, and the lanes logged the danger floor while the pair ran. The
    sentinel's shed gives back the broker's own servers and then what the allow-list does not name —
    never the ranks."""
    from queue_workflows import llm_context, rogue_llm
    from queue_workflows.llm_backends import docker_api

    rogue_llm.reset_for_tests()
    monkeypatch.setenv(rogue_llm.ALLOW_ENV, _BOX_ALLOW)
    stopped: list[str] = []
    monkeypatch.setattr(docker_api, "call", _docker(_BOX_WHILE_HELD, stopped))
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 9 * 2**30)
    w, _parked, _reservation_ = _box_b(monkeypatch)
    hold = w._begin_cluster_hold(_held_row())
    try:
        w._shed_llm_if_box_is_dying()
    finally:
        w._end_cluster_hold(hold)
    assert stopped == ["someones-trainer"]


def test_no_reservation_code_path_writes_the_control_plane():
    """(c) Parks are the operator's. Pinned the way test_memory_shedding pins the shed path."""
    from queue_workflows import claim_worker, llm_idle_unload

    worker = claim_worker.ClaimWorker
    for fn in (worker._begin_cluster_hold, worker._start_cluster_hold_watch,
               worker._cluster_hold_tick, worker._end_cluster_hold, worker._abandon_switch,
               worker._llm_starts_forbidden, worker._reservation_queued_here,
               worker._build_park_quiesce, node_queue.cluster_reservation_for,
               llm_idle_unload.ParkQuiesce.tick):
        code = inspect.getsource(fn)
        for lever in ("set_worker_control", "disable_worker", "enable_worker",
                      "INSERT INTO worker_controls", "UPDATE worker_controls"):
            assert lever not in code, f"{fn.__qualname__} writes the control plane ({lever})"


def test_the_pool_lane_holds_for_exactly_as_long_as_the_reservation_runs(monkeypatch):
    """The wiring, through the real pool-lane runner and a real claimed row: held while the body
    runs, quiesced at once, released when it returns."""
    import queue_workflows
    from queue_workflows import claim_worker

    queue_workflows.set_node_module_package("qwf_cr_nodes")
    w = claim_worker.ClaimWorker(queue="gpu", host="box-c-gpu")
    w._box_name = "box-c"
    ticks: list = []
    monkeypatch.setattr(w, "_cluster_hold_tick", lambda hold=None: ticks.append(1), raising=False)
    seen: dict = {}

    def run(*, inputs=None, out=None, llm_server=None, cancel_event=None):
        seen["forbidden"] = w._llm_starts_forbidden()
        return {"context_delta": {}}

    mod = types.ModuleType("qwf_cr_nodes.pool_hold")
    mod.run = run
    sys.modules["qwf_cr_nodes.pool_hold"] = mod
    run_id = make_run(workflow_name="vllm_lb_cluster", out_dir=None)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="cluster-box-c", node_module="pool_hold", queue="gpu",
        server_types=["cluster"], force_box=["box-c"], gpu_slots="all", priority=0)
    job = node_queue.claim_next_gpu_job(0, None, host="box-c-gpu", box="box-c",
                                        require_model=False, max_running=3)
    assert job is not None and job["id"] == job_id
    assert w._run_pool_node(job) is True
    assert seen.get("forbidden"), "the lane did not hold the box while the reservation ran"
    assert ticks, "the hold never quiesced the box"
    assert w._llm_starts_forbidden() is None, "the hold outlived the job"
    assert node_queue.get_node_job(job_id)["status"] == "completed"


def test_a_cluster_claim_stops_classic_vllm_within_one_tick_warm_or_mid_switch(box, monkeypatch):
    """The design rule: when a cluster starts, a classic vLLM still running is shut down first to
    free RAM. At CLAIM time — within one tick of the hold, not after the idle TTL — on
    both shapes a box can be in: box-b with a warm classic vLLM serving (ollama holding a model
    too), and box-c mid-switch with a vLLM still loading. Each claims its reservation through the
    real claim and the real pool-lane runner; the FIRST tick after the claim finds the vLLM
    stopped, ollama empty and no switch — and the abandoned switch never starts it again."""
    import threading

    import queue_workflows
    from queue_workflows import claim_worker, server_switch

    queue_workflows.set_node_module_package("qwf_cr_nodes")
    ticked = threading.Event()
    after_tick: list[tuple] = []
    real_tick = claim_worker.ClaimWorker._cluster_hold_tick

    def spy(self, hold=None):
        real_tick(self, hold)
        after_tick.append((self.host, box.vllm_running, list(box.ollama), self._switch.state))
        ticked.set()

    monkeypatch.setattr(claim_worker.ClaimWorker, "_cluster_hold_tick", spy)

    def run(*, inputs=None, out=None, llm_server=None, cancel_event=None):
        ticked.wait(5.0)                         # the reservation holds until its first tick
        return {"context_delta": {}}

    mod = types.ModuleType("qwf_cr_nodes.cluster_body")
    mod.run = run
    sys.modules["qwf_cr_nodes.cluster_body"] = mod

    def lane(host, box_name):
        clock = _Clock()
        w = claim_worker.ClaimWorker(queue="gpu", host=host)
        w._switch = server_switch.ServerSwitch(label=host, now_fn=clock)
        w.clock = clock
        w._box_name = box_name
        w._box_memory_in_danger = lambda: False
        w._vllm_start_decision = lambda: types.SimpleNamespace(
            utilization=None, max_num_seqs=None, refusal=None)
        return w

    def claim_and_hold(w, box_name, **claim_kw):
        run_id = make_run(workflow_name="vllm_lb_cluster", out_dir=None)
        job_id = node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"cluster-{box_name}", node_module="cluster_body",
            queue="gpu", server_types=["cluster"], force_box=[box_name], gpu_slots="all",
            priority=0)
        node_queue.prioritize_node_job(job_id)
        job = node_queue.claim_next_gpu_job(0, None, host=w.host, box=box_name,
                                            require_model=False, max_running=3, **claim_kw)
        assert job is not None and job["id"] == job_id
        ticked.clear()
        after_tick.clear()
        assert w._run_pool_node(job) is True
        assert node_queue.get_node_job(job_id)["status"] == "completed"
        return after_tick[0]

    # box-b — a WARM classic vLLM, serving, and ollama holding a model beside it.
    box.vllm_running, box.vllm_answers, box.ollama = True, True, ["chat-model-a:latest"]
    box_b = lane("box-b-gpu", "box-b")
    assert claim_and_hold(box_b, "box-b", served_types=["vllm"]) == \
        ("box-b-gpu", False, [], server_switch.IDLE), "the warm classic vLLM outlived the first tick"
    assert box.stops == 1

    # box-c — MID-SWITCH: its vLLM is still loading (no /metrics) when the reservation lands.
    box.vllm_running, box.vllm_answers, box.ollama, box.starts, box.stops = False, False, [], 0, 0
    box_c = lane("box-c-gpu", "box-c")
    _warm_up(box_c, box)
    assert claim_and_hold(box_c, "box-c", served_types=[], reservations_only=True) == \
        ("box-c-gpu", False, [], server_switch.IDLE), "the loading vLLM outlived the first tick"
    for _ in range(3):
        box_c.clock.advance(31)
        box_c._drive_server_switch()
    assert (box.starts, box.stops) == (1, 1), "the abandoned switch started the vLLM again"


def test_a_reservation_that_never_gets_the_card_spills_and_holds_nothing(monkeypatch):
    """A card another project still occupies — a render, which a hold CANNOT clear — denies the
    reservation for the whole waiting ceiling (B4). Only then does it go back to the queue,
    still pinned here, and the lane holds nothing afterwards.

    (Before B4 the reservation spilled on the FIRST denial, which is what made the box flap
    open several times a minute; the ceiling is what turns that into one honest spill.)"""
    from queue_workflows import claim_worker

    monkeypatch.setattr(claim_worker, "CLUSTER_RESERVATION_ACQUIRE_MAX_S", 0.0)
    requeued: list = []
    monkeypatch.setattr(node_queue, "requeue_job_for_retry",
                        lambda jid: requeued.append(jid), raising=False)

    class _Denied:
        def acquire(self, model):
            return False

        def current(self):
            return types.SimpleNamespace(model="someone-elses-render")

        def release(self):
            pass

    w = claim_worker.ClaimWorker(queue="gpu", host="box-c-gpu")
    w._box_name = "box-c"
    w._get_box_lease = lambda: _Denied()
    assert w._run_pool_node(_held_row()) is False
    assert not w.__dict__.get("_cluster_holds")
    assert requeued == [_held_row()["id"]], "the reservation never went back to the queue"


# ═══ THE DEFECT: a held box went on serving LLM work ══════════════════════════════════════════
#
# box-c's lane log:
#
#   WARNING [cluster-hold] box-c-gpu: HOLDING this box for cluster reservation abc123
#   INFO    [claim-worker:gpu:pool] claimed abc123 (node=cluster-box-c model=None)
#   WARNING [llm] box-c-gpu: serving ollama from a REMOTE box's server at
#                 http://172.17.0.1:11434          … repeated every 10 s, THROUGH the hold
#
# `/api/ps` still reported a resident chat-model-a:latest on every box while the cluster ranks
# were starting; llm-gateway waited at "freeing RAM; box-b: <n> GiB free, need <m>" and gave up
# after 60 s. The lane was not dead (no restarts) — it was WORKING:
# it kept claiming llm_chat rows through the box's own ollama, and each one re-loaded the model
# the hold had just unloaded. Unloading after the fact is a race the hold loses every tick.


def test_a_held_box_claims_no_new_llm_pool_work(monkeypatch):
    """(1) THE HOLD MUST GATE THE CLAIM. While this lane holds its box for the cluster, the pool
    lane must not claim work that would occupy the card — the claim is the moment the card is
    given away, and it is the only moment the engine still controls."""
    from queue_workflows import claim_worker, model_registry

    calls: list[dict] = []
    monkeypatch.setattr(node_queue, "claim_next_gpu_job",
                        lambda *a, **k: calls.append(k) or {"id": "llm-chat-1"})
    monkeypatch.setattr(model_registry, "known_ids", lambda: [])
    monkeypatch.setattr(node_queue, "cluster_reservation_for", lambda **k: None, raising=False)
    w = claim_worker.ClaimWorker(queue="gpu", host="box-c-gpu")
    w._box_name = "box-c"
    w._pool_gpu_serving_ok = lambda: True

    assert w._claim_pool() == {"id": "llm-chat-1"}, "a free box must still claim ordinary work"
    calls.clear()

    hold = w._begin_cluster_hold(_held_row())
    try:
        assert w._claim_pool() is None, "a HELD box claimed new LLM pool work"
        assert calls == [], "a HELD box reached the DB claim at all"
    finally:
        w._end_cluster_hold(hold)
    assert w._claim_pool() == {"id": "llm-chat-1"}, "the box never claimed again after the hold"


def _pool_row(job_id: str) -> dict:
    return {"id": job_id, "run_id": f"run-{job_id}", "node_id": "llm_chat", "node_module": "llm_chat",
            "queue": "gpu", "required_model": "chat-model-a:latest", "server_types": ["ollama"]}


def _running_on_the_pool(w, *rows) -> None:
    """These rows RUN on the pool lane: registered with their cancel seams, each holding its slot."""
    import threading

    for row in rows:
        w._register_pool_job(row, threading.Event())
    w._pool_inflight = len(rows)


def test_a_held_box_evicts_at_once_while_its_feeder_is_mid_claim(box, monkeypatch, caplog):
    """Seen for over an hour on every held box: "[cluster-hold] … HELD for a
    cluster reservation and draining — 1 job(s) still in flight on this worker" — with ONLY the
    reservation running. The pool feeder reserves its slot BEFORE it asks for a job (`_pool_inflight += 1`,
    then `_claim_pool()`), and a held box keeps asking — the hold refuses, the slot goes back. Every hold
    tick that lands in that window counted a job that does not exist, and skipped the eviction: a chat
    client that reloads ollama on the held card is exactly what the hold is there to stop.
    A slot being claimed is not a job streaming from the card."""
    import logging

    w, _parked, _reservation_ = _box_b(monkeypatch)
    w._spill_now = w.clock
    box.ollama = ["chat-model-a:latest"]
    _running_on_the_pool(w, _held_row())         # the reservation's own body: the only job running…
    hold = w._begin_cluster_hold(_held_row())
    w._pool_inflight += 1                        # …while the feeder holds a slot for a claim in flight
    with caplog.at_level(logging.INFO, logger="queue_workflows.llm_idle_unload"):
        w._cluster_hold_tick()
    assert box.ollama == [], "a claim attempt was counted as a job streaming from the card"
    assert not any("draining" in r.getMessage() for r in caplog.records), \
        "the hold said it was draining a job that does not exist"
    w._end_cluster_hold(hold)


def test_work_already_in_flight_when_a_hold_begins_drains_under_a_bounded_grace(
        box, monkeypatch, caplog):
    """(3) THE JOB ALREADY IN FLIGHT: it DRAINS. The hold claims nothing new (above), lets the
    job it found finish, and SAYS SO on every tick — but the grace is BOUNDED, because a request
    that never ends must not starve the cluster (llm-gateway waits ~60 s for the RAM). Past the
    ceiling the card is taken anyway, out loud."""
    import logging

    from queue_workflows import claim_worker

    w, _parked, _reservation_ = _box_b(monkeypatch)
    w._spill_now = w.clock                       # the worker's own injectable monotonic seam
    box.vllm_running, box.vllm_answers = True, True
    box.ollama = ["chat-model-a:latest"]
    chat = _pool_row("llm-chat-1")
    _running_on_the_pool(w, _held_row(), chat)   # the reservation's own body + ONE llm_chat

    hold = w._begin_cluster_hold(_held_row())
    with caplog.at_level(logging.INFO, logger="queue_workflows.llm_idle_unload"):
        w._cluster_hold_tick()
    assert box.ollama == ["chat-model-a:latest"], "the hold evicted the model a live job was streaming"
    assert any("draining" in r.getMessage() for r in caplog.records), \
        "a draining hold never said so — the operator cannot tell waiting from wedged"

    w.clock.advance(claim_worker.CLUSTER_HOLD_DRAIN_MAX_S + 1.0)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="queue_workflows.llm_idle_unload"):
        w._cluster_hold_tick()
    assert (box.ollama, box.vllm_running) == ([], False), \
        "a job that never ends held the card for ever and starved the cluster"
    assert any("releasing it anyway" in r.getMessage() for r in caplog.records)

    w._unregister_pool_job(chat)                 # the llm_chat finished; only the hold remains
    w._pool_inflight = 1
    box.ollama = ["chat-model-a:latest"]
    w._cluster_hold_tick()
    assert box.ollama == [], "a drained box did not get its card taken"
    w._end_cluster_hold(hold)


def test_a_reservation_holds_its_box_on_a_short_lease_so_a_dead_holder_frees_it(monkeypatch):
    """(4) What releases a hold whose owner dies? Nothing enforces a
    hold whose lane is gone: the row stays 'running' until its lease lapses, and on the ordinary
    600 s job lease llm-gateway believes the box is held for ten minutes while the box serves whoever
    asks. A reservation's liveness IS its renewal (every 10 s), so it holds its box on a SHORT
    lease and the orchestrator's 5 s reclaim sweep re-queues a dead holder's row within ~a
    minute. Ordinary work keeps the ordinary lease."""
    import queue_workflows
    from queue_workflows import claim_worker

    queue_workflows.set_node_module_package("qwf_cr_nodes")
    leases: list[int] = []

    class _Recorder(claim_worker.LeaseRenewer):
        def __init__(self, **kw):
            leases.append(int(kw["lease_s"]))
            super().__init__(**kw)

        def start(self):    # the lease is what this test reads, not the thread
            pass

        def stop(self):
            pass

    monkeypatch.setattr(claim_worker, "LeaseRenewer", _Recorder)

    def run(*, inputs=None, out=None, llm_server=None, cancel_event=None):
        return {"context_delta": {}}

    for name in ("lease_hold", "lease_chat"):
        mod = types.ModuleType(f"qwf_cr_nodes.{name}")
        mod.run = run
        sys.modules[f"qwf_cr_nodes.{name}"] = mod

    w = claim_worker.ClaimWorker(queue="gpu", host="box-c-gpu")
    w._box_name = "box-c"
    w._acquire_box_or_spill = lambda job: True
    w._box_job_done = lambda: None
    monkeypatch.setattr(w, "_cluster_hold_tick", lambda hold=None: None, raising=False)

    res_id = node_queue.enqueue_node_job(
        run_id=make_run(workflow_name="vllm_lb_cluster", out_dir=None), node_id="cluster-box-c",
        node_module="lease_hold", queue="gpu", server_types=["cluster"], force_box=["box-c"],
        gpu_slots="all", priority=0)
    node_queue.prioritize_node_job(res_id)
    job = node_queue.claim_next_gpu_job(0, None, host="box-c-gpu", box="box-c",
                                        require_model=False, max_running=3)
    assert job is not None and job["id"] == res_id
    assert w._run_pool_node(job) is True

    chat_id = node_queue.enqueue_node_job(
        run_id=make_run(), node_id="chat", node_module="lease_chat", queue="gpu")
    job = node_queue.claim_next_gpu_job(0, None, host="box-c-gpu", box="box-c",
                                        require_model=False, max_running=3)
    assert job is not None and job["id"] == chat_id
    assert w._run_pool_node(job) is True

    assert leases == [claim_worker.CLUSTER_RESERVATION_LEASE_S, claim_worker.LEASE_S], (
        "a reservation holds its box on the ordinary job lease — a dead holder's box reads "
        "'held' to llm-gateway for the whole lease while nothing enforces it")


# ═══ B2/B4: the hold must not fight the lane it runs on ═══════════════════════════════════════


def test_a_hold_cancels_the_llm_pool_work_it_finds_in_flight(box, monkeypatch):
    """(B2) A hold does not wait 5 s for the run-cancel watcher to notice, and it must not evict
    under a live streamer: it CANCELS this lane's in-flight LLM-serving pool jobs the moment it
    begins. On box-b a row already `cancelled` kept its process running, and it
    still issued a FRESH /v1/chat/completions ~28 s later, reloading the weights the hold was
    unloading every 5 s. Work that does not touch the LLM is left alone — a render does not
    compete for the card's weights — and the reservation's own body is never cancelled: it IS
    the hold."""
    import threading

    w, _parked, _reservation_ = _box_b(monkeypatch)
    chat, render = threading.Event(), threading.Event()
    w._register_pool_job({"id": "chat-1", "node_module": "llm_chat",
                          "required_model": "chat-model-a:latest",
                          "server_types": ["ollama"]}, chat)
    w._register_pool_job({"id": "render-1", "node_module": "render_blender",
                          "required_model": None, "server_types": []}, render)

    hold = w._begin_cluster_hold(_held_row())
    try:
        assert chat.is_set(), "the hold left a live LLM client streaming from the card it took"
        assert not render.is_set(), "the hold cancelled a render that never touches the LLM"
    finally:
        w._end_cluster_hold(hold)


def test_a_reservation_waits_for_the_card_instead_of_flapping_against_the_box_lease(
        box, monkeypatch):
    """(B4) On box-c the box lease was held by the resident
    chat-model-a:latest, so the reservation's acquire was DENIED, it spilled, and it ENDED its hold —
    repeated HOLDING/RELEASED cycles within a minute or two, each `normal service resumes` re-opening the box to
    LLM claims. A reservation is the operator's decision that this box belongs to the cluster,
    so it WAITS for the card its own hold is clearing: no re-queue, and the hold is never handed
    back between attempts."""
    from queue_workflows import claim_worker

    monkeypatch.setattr(claim_worker, "CLUSTER_RESERVATION_ACQUIRE_POLL_S", 0.0)
    w, _parked, _reservation_ = _box_b(monkeypatch)
    w._spill_now = w.clock
    attempts = {"n": 0}

    class _Lease:
        def acquire(self, model):
            attempts["n"] += 1
            return attempts["n"] >= 3     # denied twice; the hold's unload then frees the card

        def current(self):
            return types.SimpleNamespace(model="chat-model-a:latest")

        def release(self):
            pass

    w._get_box_lease = lambda: _Lease()
    requeued: list = []
    monkeypatch.setattr(node_queue, "requeue_job_for_retry",
                        lambda jid: requeued.append(jid), raising=False)
    released: list = []
    real_end = claim_worker.ClaimWorker._end_cluster_hold
    monkeypatch.setattr(claim_worker.ClaimWorker, "_end_cluster_hold",
                        lambda self, hold: (released.append(hold.job_id), real_end(self, hold)))

    hold = w._begin_cluster_hold(_held_row())
    assert w._acquire_box_for_reservation(_held_row(), hold) is True, \
        "the reservation gave the box back rather than wait for the card"
    assert attempts["n"] == 3, "the reservation did not retry the acquire its hold was clearing"
    assert requeued == [], "the reservation re-queued itself instead of waiting for the card"
    assert released == [], "the hold was RELEASED between attempts — the box re-opened to LLM work"
    w._end_cluster_hold(hold)
