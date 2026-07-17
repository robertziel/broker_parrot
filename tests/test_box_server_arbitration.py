"""ONE SERVER KIND per physical GPU box — reported, and cleared-then-loaded.

A box can be held by four serving paths: the ollama daemon, a vLLM sidecar, a ComfyUI
diffusion server, and the worker's own in-process ``ModelCache`` (native sdxl). The
recurring incident is TWO of them resident at once across two projects sharing one card.
broker_parrot is the single queuer, so it owns the rule: at most ONE (kind, model) may
be resident, and the engine both REPORTS which kind holds a box and, before a load,
CLEARS every other kind off the box (each via its own evict lever) so the second model
never forms.

These pin the pure core — the report (``held_server_types`` / ``describe_box_residency``)
and the kind-aware ``clear_box_for`` (which lever fires for which foreign kind, that the
same (kind, model) is never evicted, and that a still-dirty box after eviction RAISES).
"""
from __future__ import annotations

import pytest

from queue_workflows import llm_probe
from queue_workflows.model_residency import (
    ModelAlreadyLoadedError,
    Resident,
    clear_box_for,
    describe_box_residency,
    held_server_types,
)


def _levers():
    """Recording evict levers — one per serving kind."""
    rec = {"ollama": [], "vllm": 0, "comfyui": 0, "inprocess": 0}
    return rec, dict(
        unload_ollama=lambda models: rec["ollama"].extend(models),
        stop_vllm=lambda: rec.__setitem__("vllm", rec["vllm"] + 1),
        free_comfyui=lambda: rec.__setitem__("comfyui", rec["comfyui"] + 1),
        unload_inprocess=lambda: rec.__setitem__("inprocess", rec["inprocess"] + 1),
    )


# ── reporting: which server kind holds the box ───────────────────────────────


def test_held_server_types_lists_distinct_kinds():
    residents = [
        Resident(server="ollama", model="qwen"),
        Resident(server="comfyui", model="comfyui"),
    ]
    assert held_server_types(residents) == ["comfyui", "ollama"]


def test_held_server_types_empty_box():
    assert held_server_types([]) == []


def test_describe_box_residency_names_kind_and_model():
    d = describe_box_residency(
        [Resident(server="ollama", model="qwen")], label="box-c")
    assert "box-c" in d and "ollama" in d and "qwen" in d


def test_describe_box_residency_reports_idle():
    assert "no model" in describe_box_residency([], label="box-b").lower()


def test_describe_box_residency_flags_a_multi_kind_violation():
    d = describe_box_residency([
        Resident(server="ollama", model="qwen"),
        Resident(server="inprocess", model="sdxl"),
    ], label="box-c").lower()
    # Two kinds resident is the exact bug — the report must make it loud.
    assert "ollama" in d and "inprocess" in d


# ── clear_box_for: evict every OTHER kind, via its own lever ──────────────────


def test_inprocess_load_evicts_the_resident_ollama_via_keepalive0():
    # The scenario: one consumer wants sdxl in-process while ollama holds qwen.
    rec, levers = _levers()
    evicted = clear_box_for(
        llm_probe.INPROCESS, "sdxl",
        [Resident(server="ollama", model="qwen")],
        label="box-c", **levers)
    assert rec["ollama"] == ["qwen"]           # unloaded ollama
    assert [r.model for r in evicted] == ["qwen"]


def test_inprocess_load_evicts_comfyui_and_vllm_too():
    rec, levers = _levers()
    clear_box_for(
        llm_probe.INPROCESS, "sdxl",
        [Resident(server="comfyui", model="comfyui"),
         Resident(server="vllm", model="llama-70b")],
        label="box-c", **levers)
    assert rec["comfyui"] == 1 and rec["vllm"] == 1


def test_ollama_slot_load_evicts_comfyui_but_keeps_ollama():
    # An LLM-slot job (incoming model unspecified) clears the DIFFUSION kinds but must
    # NOT unload ollama's own model — that is the model it will serve.
    rec, levers = _levers()
    evicted = clear_box_for(
        llm_probe.OLLAMA, None,
        [Resident(server="ollama", model="qwen"),
         Resident(server="comfyui", model="comfyui")],
        label="box-c", **levers)
    assert rec["ollama"] == []                     # ollama kept
    assert rec["comfyui"] == 1                      # comfyui evicted
    assert [r.server for r in evicted] == ["comfyui"]


def test_same_kind_same_model_is_never_evicted():
    # Reloading the model the box already holds is a no-op, not a violation.
    rec, levers = _levers()
    evicted = clear_box_for(
        llm_probe.INPROCESS, "sdxl",
        [Resident(server="inprocess", model="sdxl")],
        label="box-c", **levers)
    assert evicted == [] and rec["inprocess"] == 0


def test_same_kind_different_model_is_evicted_when_a_concrete_model_is_given():
    rec, levers = _levers()
    evicted = clear_box_for(
        llm_probe.OLLAMA, "qwen",
        [Resident(server="ollama", model="llama-70b")],
        label="box-c", **levers)
    assert rec["ollama"] == ["llama-70b"]
    assert [r.model for r in evicted] == ["llama-70b"]


def test_clean_box_evicts_nothing():
    rec, levers = _levers()
    assert clear_box_for(llm_probe.INPROCESS, "sdxl", [], label="x", **levers) == []
    assert rec == {"ollama": [], "vllm": 0, "comfyui": 0, "inprocess": 0}


# ── the verify step: eviction that FAILS must not silently pass a load ────────


def test_reprobe_still_dirty_raises_after_eviction():
    # ollama refuses to drop the model (stuck) — the re-probe still sees it, so the
    # load must be REFUSED, never proceed onto a second model.
    _, levers = _levers()
    still_there = [Resident(server="ollama", model="qwen")]
    with pytest.raises(ModelAlreadyLoadedError) as e:
        clear_box_for(
            llm_probe.INPROCESS, "sdxl", still_there,
            reprobe=lambda: still_there, label="box-c", **levers)
    assert "qwen" in str(e.value) and "box-c" in str(e.value)


def test_reprobe_clean_returns_the_evicted_and_does_not_raise():
    _, levers = _levers()
    evicted = clear_box_for(
        llm_probe.INPROCESS, "sdxl",
        [Resident(server="ollama", model="qwen")],
        reprobe=lambda: [], label="box-c", **levers)  # box is clean after evict
    assert [r.model for r in evicted] == ["qwen"]


def test_reprobe_of_the_same_model_is_clean():
    # After clearing for sdxl, the re-probe legitimately sees sdxl (the loader
    # is warming it) — that is NOT foreign, so no raise.
    _, levers = _levers()
    evicted = clear_box_for(
        llm_probe.INPROCESS, "sdxl",
        [Resident(server="ollama", model="qwen")],
        reprobe=lambda: [Resident(server="inprocess", model="sdxl")],
        label="box-c", **levers)
    assert [r.model for r in evicted] == ["qwen"]


def test_a_broken_lever_does_not_stop_the_others():
    # ollama unload raises, but comfyui + vllm still get evicted (best-effort per lever).
    rec = {"comfyui": 0, "vllm": 0}
    def boom(models):
        raise OSError("ollama down")
    clear_box_for(
        llm_probe.INPROCESS, "sdxl",
        [Resident(server="ollama", model="qwen"),
         Resident(server="comfyui", model="comfyui"),
         Resident(server="vllm", model="x")],
        unload_ollama=boom,
        stop_vllm=lambda: rec.__setitem__("vllm", 1),
        free_comfyui=lambda: rec.__setitem__("comfyui", 1),
        label="box-c")
    assert rec == {"comfyui": 1, "vllm": 1}


# ── the wired gate: _box_load_gate clears the box before an in-process load ───


from queue_workflows import gpu_model_cache, model_residency
from queue_workflows.config import get_config
from queue_workflows.llm_backends import factory


@pytest.fixture
def _no_gate_sleep(monkeypatch):
    monkeypatch.setattr(gpu_model_cache, "_GATE_SLEEP", lambda s: None)


def test_gate_unloads_a_resident_ollama_then_permits_the_load(monkeypatch, _no_gate_sleep):
    monkeypatch.setattr(factory, "resolve_base_url", lambda t="ollama": "http://b:11434")
    unloaded: list[list[str]] = []
    monkeypatch.setattr(llm_probe, "unload_ollama_models",
                        lambda url, models, **k: unloaded.append(list(models)))
    # First probe: ollama holds qwen; after eviction the re-probe is clean.
    probes = iter([[Resident(server="ollama", model="qwen")],   # initial residents
                   []])                                            # settle re-probe: clean
    monkeypatch.setattr(model_residency, "probe_box_residents",
                        lambda **k: next(probes), raising=False)
    # Should NOT raise — the box was cleared.
    gpu_model_cache._box_load_gate("sdxl")
    assert unloaded == [["qwen"]]


def test_gate_refuses_when_the_box_cannot_be_cleared(monkeypatch, _no_gate_sleep):
    monkeypatch.setattr(factory, "resolve_base_url", lambda t="ollama": "http://b:11434")
    monkeypatch.setattr(llm_probe, "unload_ollama_models", lambda url, models, **k: None)
    stuck = [Resident(server="ollama", model="qwen")]
    monkeypatch.setattr(model_residency, "probe_box_residents",
                        lambda **k: list(stuck))  # never clears
    with pytest.raises(ModelAlreadyLoadedError):
        gpu_model_cache._box_load_gate("sdxl")


def test_gate_is_a_noop_on_an_idle_box(monkeypatch, _no_gate_sleep):
    monkeypatch.setattr(factory, "resolve_base_url", lambda t="ollama": "http://b:11434")
    called = []
    monkeypatch.setattr(llm_probe, "unload_ollama_models",
                        lambda url, models, **k: called.append(1))
    monkeypatch.setattr(model_residency, "probe_box_residents", lambda **k: [])
    gpu_model_cache._box_load_gate("sdxl")   # clean box → no evict, no raise
    assert called == []


def test_gate_frees_comfyui_when_configured(monkeypatch, _no_gate_sleep):
    monkeypatch.setattr(factory, "resolve_base_url", lambda t="ollama": "http://b:11434")
    monkeypatch.setattr(get_config(), "comfyui_url", "http://b:8188")
    freed = []
    monkeypatch.setattr(llm_probe, "comfyui_free", lambda url, **k: freed.append(url))
    probes = iter([[Resident(server="comfyui", model="comfyui")],   # initial residents
                   []])                                            # settle re-probe: clean
    monkeypatch.setattr(model_residency, "probe_box_residents",
                        lambda **k: next(probes))
    gpu_model_cache._box_load_gate("sdxl")
    assert freed == ["http://b:8188"]


# ── the background enforcer arbitrates ComfyUI + informs on change ────────────


def test_enforcer_evicts_comfyui_when_it_loses_to_the_desired_llm():
    # Both an LLM and a ComfyUI up on one box ⇒ the enforcer keeps the operator's
    # desired server and frees ComfyUI through its own lever.
    freed = []
    enf = model_residency.ModelResidencyEnforcer(
        collect_fn=lambda: [Resident(server="ollama", model="qwen", mru=9.0),
                            Resident(server="comfyui", model="comfyui", mru=0.0)],
        unload_ollama_fn=lambda models: None,
        stop_vllm_fn=lambda: None,
        free_comfyui_fn=lambda: freed.append(1),
        desired_server_fn=lambda: "ollama",
        cap=1, label="box-c")
    with pytest.raises(model_residency.ModelResidencyViolation):
        enf.enforce_once()
    assert freed == [1]


def test_enforcer_reports_held_server_type_only_on_change(caplog):
    import logging
    seq = iter([
        [Resident(server="ollama", model="qwen")],   # report: held by ollama:qwen
        [Resident(server="ollama", model="qwen")],   # same ⇒ NO new report
        [],                                             # change ⇒ report: idle
    ])
    enf = model_residency.ModelResidencyEnforcer(
        collect_fn=lambda: next(seq),
        unload_ollama_fn=lambda m: None, stop_vllm_fn=lambda: None,
        cap=1, label="box-c")
    with caplog.at_level(logging.INFO):
        enf.enforce_once(); enf.enforce_once(); enf.enforce_once()
    reports = [r.getMessage() for r in caplog.records
               if "held by" in r.getMessage() or "idle" in r.getMessage()]
    assert sum("ollama:qwen" in m for m in reports) == 1   # logged once, not twice
    assert any("idle" in m for m in reports)
