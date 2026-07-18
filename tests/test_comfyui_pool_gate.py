"""The pool lane must claim ComfyUI render jobs on a box with NO LLM server.

The pool lane skips claiming when the box's LLM server isn't GPU-ready
(``_llm_gpu_ok`` False) — correct for VLM-facade jobs that POST to that LLM. But a
ComfyUI render box has no LLM at all; its no-model gpu jobs POST to ComfyUI, which
is a DIFFERENT server. So the gate must also pass when the box's ComfyUI is
reachable — else comfyui_render jobs sit unclaimed forever on a render-only box.
"""
from __future__ import annotations

import pytest

from queue_workflows import claim_worker, config


@pytest.fixture(autouse=True)
def _reset():
    cfg = config.get_config()
    saved = cfg.comfyui_url
    yield
    cfg.comfyui_url = saved


def _worker():
    class _Cache:
        current_model = None
        def require_model(self, m): return object()
        def mark_busy(self): ...
        def mark_idle(self): ...
    return claim_worker.ClaimWorker(queue="gpu", host="render-box", model_cache=_Cache())


def test_pool_gate_open_when_llm_gpu_ok():
    w = _worker()
    w._llm_gpu_ok = True
    assert w._pool_gpu_serving_ok() is True


def test_pool_gate_closed_when_no_llm_and_no_comfyui():
    w = _worker()
    w._llm_gpu_ok = False
    config.get_config().comfyui_url = ""      # no comfyui configured
    assert w._pool_gpu_serving_ok() is False


def test_pool_gate_open_when_comfyui_reachable_despite_no_llm(monkeypatch):
    """A render-only box: no LLM, but ComfyUI is up → pool lane must claim."""
    from queue_workflows import llm_probe
    w = _worker()
    w._llm_gpu_ok = False
    config.get_config().comfyui_url = "http://127.0.0.1:8188"
    monkeypatch.setattr(llm_probe, "comfyui_reachable", lambda url, **k: True)
    assert w._pool_gpu_serving_ok() is True


def test_pool_gate_closed_when_comfyui_configured_but_down(monkeypatch):
    from queue_workflows import llm_probe
    w = _worker()
    w._llm_gpu_ok = False
    config.get_config().comfyui_url = "http://127.0.0.1:8188"
    monkeypatch.setattr(llm_probe, "comfyui_reachable", lambda url, **k: False)
    assert w._pool_gpu_serving_ok() is False


def test_claim_pool_skips_when_gate_closed(monkeypatch):
    w = _worker()
    monkeypatch.setattr(w, "_pool_gpu_serving_ok", lambda: False)
    assert w._claim_pool() is None       # no DB touch when the gate is closed


# ── fill-before-spill must NOT starve a ComfyUI render box ────────────────────


def test_comfyui_serving_box_never_defers(monkeypatch):
    """Fill-before-spill consolidates VLM load onto one LLM box. A box whose pool gate
    opened via ComfyUI (no LLM answers) serves box-placed render jobs — deferring to a
    'fresher LLM peer' that can never claim those jobs starves the queue (observed 3x:
    GPU boxes sat idle on queued/force_box'd comfyui_video jobs while another box's
    LLM worker heartbeat kept them deferring)."""
    from queue_workflows import llm_probe
    w = _worker()
    w._llm_gpu_ok = False                                     # no LLM on this box
    config.get_config().comfyui_url = "http://127.0.0.1:8188"
    monkeypatch.setattr(llm_probe, "comfyui_reachable", lambda url, **k: True)
    assert w._pool_should_defer(par=1) is False               # never defer


def test_llm_serving_box_still_defers_by_heuristic(monkeypatch):
    from queue_workflows import node_queue
    w = _worker()
    w._llm_gpu_ok = True                                      # a genuine LLM box
    monkeypatch.setattr(node_queue, "vlm_pool_should_defer", lambda h, p: True)
    assert w._pool_should_defer(par=1) is True                # heuristic still applies
