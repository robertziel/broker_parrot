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
    llm = _worker(host="lm-gpu", lease=_file_lease(tmp_path, "lm:1"))
    vid = _worker(host="vg-gpu", lease=_file_lease(tmp_path, "vg:1"))
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
    vid = _worker(host="vg-gpu", lease=_file_lease(tmp_path, "vg:1"))
    llm = _worker(host="lm-gpu", lease=_file_lease(tmp_path, "lm:1"))
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
    llm = _worker(host="lm-gpu", lease=_file_lease(tmp_path, "lm:1"))
    vid = _worker(host="vg-gpu", lease=_file_lease(tmp_path, "vg:1"))
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
    monkeypatch.setattr(llm_probe, "loaded_models", lambda url: ["qwen"])
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
