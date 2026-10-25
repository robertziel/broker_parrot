"""ONE MODEL per physical GPU box, across ALL projects — enforced in the run-path.

The box a job runs on can hold at most one model. broker_parrot is the single queuer
every project runs, so it owns the rule via a flock'd per-box lease
(:mod:`queue_workflows.gpu_model_lease`) that every GPU worker container on the card —
regardless of project or database — coordinates through. Before executing a GPU node,
the worker acquires the box for the job's EFFECTIVE model (its ``required_model``, or
the shared LLM-server slot for a no-model ollama job); if a live worker holds a
DIFFERENT model, the job is SPILLED (re-queued) so a second model never loads.

These pin the worker side: the effective-model key, the acquire-or-spill gate on both
lanes, the residency-tracked renew/release tick, and — with a real shared file lease —
that an LLM job and a diffusion job on ONE box mutually exclude.
"""
from __future__ import annotations

import pytest

import queue_workflows
from queue_workflows import claim_worker
from queue_workflows.config import get_config
from queue_workflows.gpu_model_lease import (
    COMFYUI_SERVER_SLOT, LLM_SERVER_SLOT, FileLeaseStore, ModelLease,
)


class _Cache:
    current_model = None


def _worker(host="box-a-gpu", lease=None):
    w = claim_worker.ClaimWorker(queue="gpu", host=host, model_cache=_Cache())
    if lease is not None:
        w._box_lease = lease
    return w


def _file_lease(tmp_path, holder, box="box-a"):
    return ModelLease(box_id=box, store=FileLeaseStore(str(tmp_path)),
                      holder=holder, ttl_s=60, now_fn=lambda: 1000.0)


# ── effective model key ──────────────────────────────────────────────────────


def test_effective_model_is_required_model_when_set():
    assert _worker()._effective_gpu_model({"required_model": "sdxl"}) == "sdxl"


def test_effective_model_is_llm_slot_for_a_no_model_job():
    assert _worker()._effective_gpu_model({"required_model": None}) == LLM_SERVER_SLOT
    assert _worker()._effective_gpu_model({}) == LLM_SERVER_SLOT


# ── a ComfyUI render is its OWN server kind, NOT the ollama LLM slot ──────────
#
# THE CROSS-PROJECT OVER-CLAIM THIS FIXES: a no-model ComfyUI render used to key to
# the SAME ``__llm_server__`` slot as a no-model ollama-dispatch job → the box lease
# treated them as the same "model" and SHARED the card → two GPU consumers on one box
# (fatal on a marginal GPU). A ComfyUI-serving worker's no-model job must key to a
# DISTINCT slot so it EXCLUDES an ollama job (and vice versa), while two ComfyUI
# renders still share. Signalled by a wired ComfyUI lifecycle or a configured ComfyUI URL.


def test_effective_model_is_comfyui_slot_when_the_worker_serves_comfyui():
    queue_workflows.set_comfyui_lifecycle(start_fn=lambda: None, stop_fn=None)
    assert _worker()._effective_gpu_model({"required_model": None}) == COMFYUI_SERVER_SLOT
    assert _worker()._effective_gpu_model({}) == COMFYUI_SERVER_SLOT
    assert _worker()._effective_gpu_model({"required_model": "sdxl"}) == "sdxl"  # real model still wins


def test_comfyui_url_also_signals_a_comfyui_serving_box():
    get_config().comfyui_url = "http://box:8188"
    assert _worker()._effective_gpu_model({"required_model": None}) == COMFYUI_SERVER_SLOT


def test_comfyui_render_and_ollama_job_MUTUALLY_EXCLUDE_on_one_box(monkeypatch, tmp_path):
    # The exact observed collision: an ollama dispatch and a ComfyUI render, both
    # no-model, on ONE physical box. Before the fix they shared __llm_server__; now
    # they are different server kinds → the second one spills.
    llm = _worker(host="llm-gpu", lease=_file_lease(tmp_path, "llm:1"))
    vid = _worker(host="render-gpu", lease=_file_lease(tmp_path, "render:1"))
    monkeypatch.setattr(llm, "_box_serves_comfyui", lambda: False)   # ollama dispatch
    monkeypatch.setattr(vid, "_box_serves_comfyui", lambda: True)    # ComfyUI render
    requeued = []
    monkeypatch.setattr(claim_worker.node_queue, "requeue_job_for_retry",
                        lambda jid: requeued.append(jid))
    assert llm._acquire_box_or_spill({"id": "score", "required_model": None}) is True
    assert vid._acquire_box_or_spill({"id": "render", "required_model": None}) is False
    assert requeued == ["render"]           # the render spilled — no 2/1
    assert vid._box_held_model is None


def test_comfyui_render_excludes_a_later_ollama_job_too(monkeypatch, tmp_path):
    # Symmetric: the render takes the box first, an ollama job is then refused.
    vid = _worker(host="render-gpu", lease=_file_lease(tmp_path, "render:1"))
    llm = _worker(host="llm-gpu", lease=_file_lease(tmp_path, "llm:1"))
    monkeypatch.setattr(vid, "_box_serves_comfyui", lambda: True)
    monkeypatch.setattr(llm, "_box_serves_comfyui", lambda: False)
    monkeypatch.setattr(claim_worker.node_queue, "requeue_job_for_retry", lambda jid: None)
    assert vid._acquire_box_or_spill({"id": "render", "required_model": None}) is True
    assert llm._acquire_box_or_spill({"id": "score", "required_model": None}) is False


def test_two_comfyui_renders_share_the_comfyui_slot(tmp_path):
    a = _file_lease(tmp_path, "a")
    b = _file_lease(tmp_path, "b")
    assert a.acquire(COMFYUI_SERVER_SLOT) is True
    assert b.acquire(COMFYUI_SERVER_SLOT) is True   # same kind ⇒ both may hold (no over-denial)


def test_residency_comfyui_slot_probes_comfyui_loaded(monkeypatch):
    from queue_workflows import llm_probe
    get_config().comfyui_url = "http://box:8188"
    monkeypatch.setattr(llm_probe, "comfyui_loaded", lambda url: True)
    assert _worker()._model_still_resident(COMFYUI_SERVER_SLOT) is True
    monkeypatch.setattr(llm_probe, "comfyui_loaded", lambda url: False)
    assert _worker()._model_still_resident(COMFYUI_SERVER_SLOT) is False


def test_residency_comfyui_slot_without_a_url_is_not_resident():
    # host-managed ComfyUI with comfyui_url unset ⇒ unprobeable ⇒ not-resident, so the
    # box RELEASES once the render's job count hits 0 (the active-jobs guard holds it
    # for the render's whole life). No false warm-hold on a card it can't see.
    assert _worker()._model_still_resident(COMFYUI_SERVER_SLOT) is False


# ── the acquire-or-spill gate ────────────────────────────────────────────────


def test_gate_grants_and_records_held_model(monkeypatch, tmp_path):
    w = _worker(lease=_file_lease(tmp_path, "w1"))
    assert w._acquire_box_or_spill({"id": "j", "required_model": "sdxl"}) is True
    assert w._box_held_model == "sdxl"


def test_gate_spills_and_requeues_when_a_peer_holds_another_model(monkeypatch, tmp_path):
    # A peer (different holder, same box + store) holds a DIFFERENT model.
    peer = _file_lease(tmp_path, "peer")
    assert peer.acquire("sdxl") is True
    w = _worker(lease=_file_lease(tmp_path, "me"))
    requeued = []
    monkeypatch.setattr(claim_worker.node_queue, "requeue_job_for_retry",
                        lambda jid: requeued.append(jid))
    ok = w._acquire_box_or_spill({"id": "j7", "required_model": None})  # wants LLM slot
    assert ok is False
    assert requeued == ["j7"]            # spilled to a free/matching box
    assert w._box_held_model is None     # never took the box


def test_two_no_model_jobs_share_the_llm_slot(tmp_path):
    a = _file_lease(tmp_path, "a")
    b = _file_lease(tmp_path, "b")
    assert a.acquire(LLM_SERVER_SLOT) is True
    assert b.acquire(LLM_SERVER_SLOT) is True     # same model ⇒ both may hold it (PAR)


def test_llm_job_and_diffusion_job_mutually_exclude_on_one_box(monkeypatch, tmp_path):
    # Two consumers on one box: one dispatches to ollama, one loads sdxl in-process.
    llm = _worker(host="llm-gpu", lease=_file_lease(tmp_path, "llm:1"))
    vid = _worker(host="render-gpu", lease=_file_lease(tmp_path, "render:1"))
    requeued = []
    monkeypatch.setattr(claim_worker.node_queue, "requeue_job_for_retry",
                        lambda jid: requeued.append(jid))
    # The LLM job takes the box first.
    assert llm._acquire_box_or_spill({"id": "score", "required_model": None}) is True
    # A diffusion job for a DIFFERENT model is refused → spills.
    assert vid._acquire_box_or_spill({"id": "render", "required_model": "sdxl"}) is False
    assert requeued == ["render"]


# ── the residency-tracked renew / release tick ───────────────────────────────


class _RenewRecordingLease:
    def __init__(self):
        self.renewed, self.released = [], 0

    def renew(self, m):
        self.renewed.append(m)
        return True

    def release(self):
        self.released += 1


def test_tick_renews_while_the_model_is_resident(monkeypatch):
    w = _worker()
    w._box_lease = _RenewRecordingLease()
    w._box_held_model = "sdxl"
    monkeypatch.setattr(w, "_model_still_resident", lambda held: True)
    w._box_lease_tick()
    assert w._box_lease.renewed == ["sdxl"] and w._box_held_model == "sdxl"


def test_tick_releases_once_the_model_unloads(monkeypatch):
    w = _worker()
    w._box_lease = _RenewRecordingLease()
    w._box_held_model = "sdxl"
    w._box_active_jobs = 0                     # no job running — safe to release
    monkeypatch.setattr(w, "_model_still_resident", lambda held: False)
    w._box_lease_tick()
    assert w._box_lease.released == 1 and w._box_held_model is None


def test_tick_HOLDS_through_a_residency_flicker_while_a_job_is_active(monkeypatch):
    # THE BUG THIS FIXES: a long ollama job's model can read 'not resident' for a beat
    # (probe blip, ollama keep_alive between calls, a peer's transient eviction). The
    # box must STAY held for the whole job — else the lease drops mid-job and never
    # re-acquires (acquire only runs at claim), leaving the box unprotected: a peer
    # would EVICT this running model instead of SPILLING.
    w = _worker()
    w._box_lease = _RenewRecordingLease()
    w._box_held_model = "__llm_server__"
    w._box_active_jobs = 1                     # a job is running on this worker
    monkeypatch.setattr(w, "_model_still_resident", lambda held: False)  # flicker!
    w._box_lease_tick()
    assert w._box_lease.released == 0                       # did NOT release
    assert w._box_lease.renewed == ["__llm_server__"]       # kept holding
    assert w._box_held_model == "__llm_server__"


def test_acquire_marks_a_job_active_and_done_decrements(monkeypatch, tmp_path):
    w = _worker(lease=_file_lease(tmp_path, "w1"))
    assert w._box_active_jobs == 0
    w._acquire_box_or_spill({"id": "j", "required_model": "sdxl"})
    assert w._box_active_jobs == 1
    w._box_job_done()
    assert w._box_active_jobs == 0
    # w1 still holds sdxl (warm-hold survives job-done) → a DIFFERENT-model job on
    # the same box is denied, and a spilled (denied) job must NOT mark active.
    monkeypatch.setattr(claim_worker.node_queue, "requeue_job_for_retry", lambda jid: None)
    w2 = _worker(lease=_file_lease(tmp_path, "w2"))
    assert w2._acquire_box_or_spill({"id": "j2", "required_model": "other-model"}) is False
    assert w2._box_active_jobs == 0


def test_tick_releases_after_the_job_is_done_and_the_model_unloads(monkeypatch):
    w = _worker()
    w._box_lease = _RenewRecordingLease()
    w._box_held_model = "sdxl"
    w._box_active_jobs = 1
    w._box_job_done()                          # job ended → count 0, warm-hold remains
    monkeypatch.setattr(w, "_model_still_resident", lambda held: False)  # now unloaded
    w._box_lease_tick()
    assert w._box_lease.released == 1 and w._box_held_model is None


def test_tick_is_a_noop_when_holding_nothing():
    w = _worker()
    w._box_lease = _RenewRecordingLease()
    w._box_held_model = None
    w._box_lease_tick()
    assert w._box_lease.renewed == [] and w._box_lease.released == 0


def test_residency_llm_slot_checks_ollama_loaded_models(monkeypatch):
    from queue_workflows import llm_probe
    from queue_workflows.llm_backends import factory
    monkeypatch.setattr(factory, "resolve_base_url", lambda t="ollama": "http://b:11434")
    monkeypatch.setattr(llm_probe, "loaded_models", lambda url: ["an LLM"])
    assert _worker()._model_still_resident(LLM_SERVER_SLOT) is True
    monkeypatch.setattr(llm_probe, "loaded_models", lambda url: [])
    assert _worker()._model_still_resident(LLM_SERVER_SLOT) is False


def test_residency_diffusion_checks_modelcache_current_model():
    w = _worker()
    w.model_cache.current_model = "sdxl"
    assert w._model_still_resident("sdxl") is True
    assert w._model_still_resident("other") is False


# ── the pool lane funnels through the same gate ──────────────────────────────


def test_pool_node_spills_without_executing_when_box_is_full(monkeypatch, tmp_path):
    peer = _file_lease(tmp_path, "peer")
    peer.acquire("sdxl")
    w = _worker(lease=_file_lease(tmp_path, "me"))
    monkeypatch.setattr(claim_worker.node_queue, "requeue_job_for_retry", lambda jid: None)
    ran = []
    monkeypatch.setattr(w, "_run_pool_lane_body", lambda *a, **k: ran.append(1), raising=False)
    assert w._run_pool_node({"id": "j", "required_model": None}) is False
    assert ran == []                     # execution never started


# ── SELF-MANAGING GPU work is its own occupant, not the LLM's ────────────────
#
# THE DEFECT: the lease's subject was "which MODEL is on the card", so a job that
# is not a model had to borrow a key that is. A self-managing node (Blender/EEVEE
# via `gpu_self_load_node_modules`) fell through to `__llm_server__` — so it
# SHARED the lease with ollama dispatches instead of excluding them, and two
# unrelated workloads ran on one card. The panel reported the LLM model "in use"
# while a Blender render held the GPU.


def _self_managing(monkeypatch, *modules):
    from queue_workflows import config as _config

    cfg = _config.get_config()
    monkeypatch.setattr(cfg, "gpu_self_load_node_modules", frozenset(modules),
                        raising=False)


def test_a_self_managing_node_gets_its_OWN_slot(monkeypatch):
    from queue_workflows.gpu_model_lease import self_managed_slot

    _self_managing(monkeypatch, "render_blender_reel")
    job = {"required_model": None, "node_module": "render_blender_reel"}
    assert _worker()._effective_gpu_model(job) == self_managed_slot("render_blender_reel")


def test_it_EXCLUDES_an_ollama_dispatch_on_the_same_box(monkeypatch, tmp_path):
    """The consequence that matters. Sharing the LLM slot meant the lease GRANTED
    both — the invariant it exists to enforce was off for exactly this class."""
    _self_managing(monkeypatch, "render_blender_reel")
    render = _worker(lease=_file_lease(tmp_path, "render"))
    llm = _worker(lease=_file_lease(tmp_path, "llm"))

    assert render._acquire_box_or_spill(
        {"id": "j1", "required_model": None, "node_module": "render_blender_reel"})
    # An ollama dispatch on the SAME physical box must now be refused and spill.
    assert not llm._acquire_box_or_spill(
        {"id": "j2", "required_model": None, "node_module": "llm_chat"})


def test_two_renders_of_the_SAME_node_still_share_the_card(monkeypatch, tmp_path):
    """Per-module, not one shared key: identical work co-tenants fine — it is
    unrelated work that must not."""
    _self_managing(monkeypatch, "render_blender_reel")
    a = _worker(lease=_file_lease(tmp_path, "a"))
    b = _worker(lease=_file_lease(tmp_path, "b"))
    job = {"required_model": None, "node_module": "render_blender_reel"}

    assert a._acquire_box_or_spill(dict(job, id="j1"))
    assert b._acquire_box_or_spill(dict(job, id="j2"))


def test_two_DIFFERENT_self_managing_nodes_exclude_each_other(monkeypatch, tmp_path):
    _self_managing(monkeypatch, "render_blender_reel", "video_model_a_render")
    a = _worker(lease=_file_lease(tmp_path, "a"))
    b = _worker(lease=_file_lease(tmp_path, "b"))

    assert a._acquire_box_or_spill(
        {"id": "j1", "required_model": None, "node_module": "render_blender_reel"})
    assert not b._acquire_box_or_spill(
        {"id": "j2", "required_model": None, "node_module": "video_model_a_render"})


def test_an_UNregistered_no_model_job_still_keys_to_the_llm_slot(monkeypatch):
    """Registration is the declaration. A gpu node that never declared itself
    self-managing keeps the old behaviour — this change adds a category, it does
    not reinterpret the ones that exist."""
    _self_managing(monkeypatch)
    assert _worker()._effective_gpu_model(
        {"required_model": None, "node_module": "llm_chat"}) == LLM_SERVER_SLOT


# ── taking the card reclaims the LLM's RAM ───────────────────────────────────


def test_a_non_llm_occupant_unloads_the_resident_ollama_weights(monkeypatch, tmp_path):
    """A resident model is not the same thing as a held lease: ollama's
    keep-alive holds ~29 GB AFTER its job finished and its lease was released, so
    a render acquires the box legitimately and then runs alongside VRAM it was
    just granted exclusive use of. Exclusivity that covers the bookkeeping and
    not the card is not exclusivity."""
    from queue_workflows import llm_probe

    _self_managing(monkeypatch, "render_blender_reel")
    unloaded: list[list[str]] = []
    monkeypatch.setattr(llm_probe, "loaded_models_info",
                        lambda *a, **k: [{"name": "chat-model-a:latest"}])
    monkeypatch.setattr(llm_probe, "unload_ollama_models",
                        lambda base, models: unloaded.append(list(models)))

    w = _worker(lease=_file_lease(tmp_path, "render"))
    assert w._acquire_box_or_spill(
        {"id": "j1", "required_model": None, "node_module": "render_blender_reel"})
    assert unloaded == [["chat-model-a:latest"]]


def test_an_LLM_job_does_NOT_unload_the_model_it_is_about_to_use(monkeypatch, tmp_path):
    """The obvious way to get this catastrophically wrong."""
    from queue_workflows import llm_probe

    _self_managing(monkeypatch, "render_blender_reel")
    unloaded: list[list[str]] = []
    monkeypatch.setattr(llm_probe, "loaded_models_info",
                        lambda *a, **k: [{"name": "chat-model-a:latest"}])
    monkeypatch.setattr(llm_probe, "unload_ollama_models",
                        lambda base, models: unloaded.append(list(models)))

    w = _worker(lease=_file_lease(tmp_path, "llm"))
    assert w._acquire_box_or_spill(
        {"id": "j1", "required_model": None, "node_module": "llm_chat"})
    assert unloaded == []


# ── slots are internal keys; humans get words ────────────────────────────────


def test_slots_describe_themselves_for_a_panel():
    from queue_workflows.gpu_model_lease import (
        COMFYUI_SERVER_SLOT, describe_slot, self_managed_slot,
    )

    assert describe_slot(None) == "idle"
    assert describe_slot(LLM_SERVER_SLOT) == "LLM server"
    assert describe_slot(COMFYUI_SERVER_SLOT) == "ComfyUI render"
    assert describe_slot(self_managed_slot("render_blender_reel")) == \
        "GPU job: render_blender_reel"
    assert describe_slot("sdxl") == "sdxl"


# ── the heartbeat gauge names the occupant ───────────────────────────────────


def _heartbeat(occupant=None, model=None):
    class _MC:
        current_model = model

    return claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-a-gpu", model_cache=_MC(),
        occupant_fn=(lambda: occupant))


def test_the_gauge_does_not_name_a_self_managing_occupant_as_a_MODEL():
    """``current_model`` names a MODEL. A self-managing occupant (e.g. a Blender
    render holding the card) is a running job, not a model, so the gauge reads None
    for it; the box still does not look idle, because its running job row says so."""
    from queue_workflows.gpu_model_lease import self_managed_slot

    hb = _heartbeat(occupant=self_managed_slot("render_blender_reel"))
    assert hb._current_model() is None


def test_a_real_in_process_model_still_wins_the_gauge():
    from queue_workflows.gpu_model_lease import self_managed_slot

    hb = _heartbeat(occupant=self_managed_slot("render_blender_reel"), model="sdxl")
    assert hb._current_model() == "sdxl"


def test_the_llm_slot_stays_null_so_the_server_is_not_double_reported():
    from queue_workflows.gpu_model_lease import LLM_SERVER_SLOT

    assert _heartbeat(occupant=LLM_SERVER_SLOT)._current_model() is None
    assert _heartbeat(occupant=None)._current_model() is None


# ── a box that serves BOTH: the slot follows the JOB, not the box ─────────────
#
# Seen on a box that serves ComfyUI (a configured URL) AND the
# LLM lanes, which the rule above never anticipated: every no-model job on such a box
# took COMFYUI_SERVER_SLOT — including `system.vllm_lease` and every LLM dispatch. Two
# consequences, and the second is the serious one:
#
#   * the panel showed "ComfyUI render" as the occupant beside vLLM's real model and
#     raised "⚠ 2 models resident — one GPU, contending for VRAM" while ComfyUI's queue
#     was empty and nvidia-smi showed one process;
#   * sharing one slot means NOT excluding: an LLM dispatch and a real ComfyUI render
#     both keyed to COMFYUI_SERVER_SLOT and the lease granted BOTH — two models on one
#     card, which is the very thing this lease exists to prevent.
#
# A job that resolves to an LLM server IS the LLM server's occupant, whatever else the
# box can serve.


def test_an_llm_job_takes_the_llm_slot_on_a_box_that_also_serves_comfyui():
    get_config().comfyui_url = "http://box:8188"
    w = _worker()
    assert w._effective_gpu_model({"server_types": ["vllm"]}) == LLM_SERVER_SLOT
    assert w._effective_gpu_model({"server_types": ["ollama", "vllm"]}) == LLM_SERVER_SLOT
    assert w._effective_gpu_model({"resolved_server": "ollama"}) == LLM_SERVER_SLOT
    assert w._effective_gpu_model({"resolved_server": "vllm", "server_types": []}) == LLM_SERVER_SLOT


def test_a_job_with_no_server_of_its_own_is_still_the_boxs_kind():
    get_config().comfyui_url = "http://box:8188"
    w = _worker()
    assert w._effective_gpu_model({"required_model": None}) == COMFYUI_SERVER_SLOT
    assert w._effective_gpu_model({"server_types": []}) == COMFYUI_SERVER_SLOT
    assert w._effective_gpu_model({"server_types": None, "resolved_server": None}) == COMFYUI_SERVER_SLOT


def test_a_real_model_and_a_self_managing_module_still_win_over_the_server_kind():
    get_config().comfyui_url = "http://box:8188"
    w = _worker()
    assert w._effective_gpu_model({"required_model": "sdxl", "server_types": ["vllm"]}) == "sdxl"
    get_config().gpu_self_load_node_modules = frozenset({"render.blender"})
    assert w._effective_gpu_model(
        {"node_module": "render.blender", "server_types": ["ollama"]}).startswith("__gpu__:")


def test_an_unknown_server_kind_is_not_treated_as_the_llm_server():
    get_config().comfyui_url = "http://box:8188"
    assert _worker()._effective_gpu_model({"server_types": ["tgi"]}) == COMFYUI_SERVER_SLOT


def test_an_llm_job_and_a_comfyui_render_exclude_each_other_on_one_such_box(monkeypatch, tmp_path):
    """The co-residency the mislabelling allowed: both were 'ComfyUI render', so the
    lease saw one occupant and let both on. The vLLM lease job and a render must not
    share a card."""
    get_config().comfyui_url = "http://box:8188"
    monkeypatch.setattr(claim_worker.node_queue, "requeue_job_for_retry", lambda jid: None)
    llm = _worker(host="box-b-gpu", lease=_file_lease(tmp_path, "llm:1"))
    render = _worker(host="box-b-gpu", lease=_file_lease(tmp_path, "render:1"))
    assert llm._acquire_box_or_spill(
        {"id": "lease", "required_model": None, "server_types": ["vllm"]}) is True
    assert render._acquire_box_or_spill({"id": "render", "required_model": None}) is False


# ── current_model carries a MODEL, or nothing ────────────────────────────────
#
# Seen in an operator panel: the second "model" is not a model, and showing it
# makes no sense. The box card showed
#
#     MODEL chat-model-a:latest      18 GB · in use
#     MODEL GPU job: render_job         content_app/gpu
#     ⚠ 2 models resident — one GPU, contending for VRAM
#
# The second row is a LEASE SLOT rendered for humans and published in
# `worker_heartbeats.current_model`. The LLM slot was already excluded there, with the
# right reason — "the server's OWN model reporting already covers it, and 'LLM server'
# beside 'chat-model-a:latest' would double-report one occupant" — and that reason
# applies to every sentinel: a ComfyUI render and a self-managing node are OCCUPANCY,
# not models. A field named current_model carries a model id or NULL; what the box is
# doing is already on the card, one section down, as its running jobs.


def _emitter(occupant, model=None):
    class _Cache:
        current_model = model
    return claim_worker.HeartbeatEmitter(
        host_label="box-b-gpu", queue="gpu", model_cache=(_Cache() if model else None),
        occupant_fn=lambda: occupant)


def test_a_real_model_on_the_card_is_reported():
    from queue_workflows.gpu_model_lease import self_managed_slot

    assert _emitter(None, model="sdxl")._current_model() == "sdxl"
    assert _emitter("sdxl")._current_model() == "sdxl", "a model id is a model, slot or not"
    assert _emitter(self_managed_slot("render.blender"), model="sdxl")._current_model() == "sdxl"


@pytest.mark.parametrize("slot", [
    LLM_SERVER_SLOT,
    COMFYUI_SERVER_SLOT,
    "__gpu__:render_job",
    "__gpu__:render_blender",
])
def test_a_sentinel_slot_is_not_a_model_and_is_not_published(slot):
    assert _emitter(slot)._current_model() is None


def test_an_idle_gpu_box_and_every_cpu_box_report_nothing():
    assert _emitter(None)._current_model() is None
    e = claim_worker.HeartbeatEmitter(host_label="box-a-cpu", queue="cpu",
                                      occupant_fn=lambda: "sdxl")
    assert e._current_model() is None


def test_every_sentinel_follows_the_convention_the_predicate_tests():
    """The predicate matches SHAPE (``__name__`` / ``__gpu__:<module>``) rather than a
    list somebody must remember to extend. That only works while every sentinel keeps
    the shape — so each one is pinned here, and a rename that breaks it fails loudly."""
    from queue_workflows import claim_worker as cw
    from queue_workflows import gpu_model_lease as gl

    for slot in (gl.LLM_SERVER_SLOT, gl.COMFYUI_SERVER_SLOT, cw.SWITCH_SLOT,
                 gl.self_managed_slot("render_blender_reel")):
        assert gl.is_sentinel_slot(slot), slot
    for model in ("chat-model-a:latest", "sdxl", "chat-expert-a", "chat-model-a:v2", None, ""):
        assert not gl.is_sentinel_slot(model), model
