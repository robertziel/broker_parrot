"""The GPU worker's residency-enforcer wiring: what it collects, how it kills.

`_collect_box_residents` must see every ollama-loaded model AND a genuine vLLM's
pinned model — without double-counting an ollama as a vLLM (ollama answers
``/v1/models`` too, so the vLLM leg only counts when its URL is distinct from
ollama's and probes as a real vLLM). `_stop_rogue_vllm` uses the host-wired sidecar
stop and is LOUD (not silent) when none is wired.
"""

from __future__ import annotations

import logging

from queue_workflows import claim_worker, config, llm_probe
from queue_workflows.llm_backends import factory


class _Cache:
    current_model = None


def _gpu_worker():
    return claim_worker.ClaimWorker(queue="gpu", host="box-a-gpu", model_cache=_Cache())


def _wire(monkeypatch, *, ollama_url, vllm_url, loaded, vllm_probe, served):
    urls = {"ollama": ollama_url, "vllm": vllm_url}
    monkeypatch.setattr(factory, "resolve_base_url", lambda t="ollama": urls[t])
    monkeypatch.setattr(llm_probe, "loaded_models_info",
                        lambda url, **k: list(loaded) if url == ollama_url else [])
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: list(vllm_probe))
    monkeypatch.setattr(llm_probe, "vllm_served_models",
                        lambda url, **k: list(served) if url == vllm_url else [])


def test_collects_ollama_models_with_recency(monkeypatch):
    _wire(monkeypatch, ollama_url="http://b:11434", vllm_url="http://b:11434",
          loaded=[{"name": "m1", "mru": 5.0}, {"name": "m2", "mru": 9.0}],
          vllm_probe=["ollama"], served=[])
    got = _gpu_worker()._collect_box_residents()
    assert [(r.server, r.model, r.mru) for r in got] == \
        [("ollama", "m1", 5.0), ("ollama", "m2", 9.0)]


def test_counts_a_real_vllm_on_a_distinct_url(monkeypatch):
    _wire(monkeypatch, ollama_url="http://b:11434", vllm_url="http://b:8000",
          loaded=[{"name": "m1", "mru": 5.0}], vllm_probe=["vllm"], served=["vlm-x"])
    got = _gpu_worker()._collect_box_residents()
    assert ("vllm", "vlm-x") in {(r.server, r.model) for r in got}


def test_never_double_counts_ollama_as_vllm(monkeypatch):
    # Same URL for both types (the common single-server box): the vLLM leg must
    # not run at all, even though ollama would answer /v1/models.
    _wire(monkeypatch, ollama_url="http://b:11434", vllm_url="http://b:11434",
          loaded=[{"name": "m1", "mru": 5.0}], vllm_probe=["ollama"], served=["m1"])
    got = _gpu_worker()._collect_box_residents()
    assert {r.server for r in got} == {"ollama"}


def test_distinct_url_that_probes_as_ollama_is_not_a_vllm(monkeypatch):
    _wire(monkeypatch, ollama_url="http://b:11434", vllm_url="http://b:8000",
          loaded=[{"name": "m1", "mru": 5.0}], vllm_probe=["ollama"], served=["x"])
    got = _gpu_worker()._collect_box_residents()
    assert {r.server for r in got} == {"ollama"}


def test_stop_rogue_vllm_uses_wired_hook(monkeypatch):
    stops = []
    monkeypatch.setattr(config.get_config(), "vllm_stop_fn", lambda: stops.append(1))
    _gpu_worker()._stop_rogue_vllm()
    assert stops == [1]


def test_stop_rogue_vllm_is_loud_without_a_hook(monkeypatch, caplog):
    monkeypatch.setattr(config.get_config(), "vllm_stop_fn", None)
    with caplog.at_level(logging.ERROR):
        _gpu_worker()._stop_rogue_vllm()
    assert any("no stop hook is wired" in r.getMessage() for r in caplog.records)
