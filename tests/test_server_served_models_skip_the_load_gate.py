"""A server-served model must not go through the IN-PROCESS load gate.

The failure this closes, measured on a live fleet:

    node   llm_chat/llm_chat   required_model chat-expert-a   resolved_server inprocess
    error  MODEL LOAD BLOCKED on box-c-gpu: cleared the box for 'chat-expert-a' (inprocess)
           but chat-expert-a:latest[ollama] is STILL resident after eviction

`chat-expert-a` is served by ollama. Its ModelSpec is a SENTINEL whose loader returns the id and
puts nothing in VRAM. Running the clear-then-load gate for it evicted the very ollama copy the
node was about to call, and then refused the load because the survivor was reported under its
full tag while the load asked for the bare one. 482 `chat-model-a:latest` failures on that fleet
share the shape.
"""
from __future__ import annotations

import pytest

from queue_workflows import gpu_model_cache, model_registry
from queue_workflows.model_registry import ModelSpec


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(model_registry.MODELS)
    try:
        yield
    finally:
        model_registry.MODELS.clear()
        model_registry.MODELS.update(saved)


def test_a_served_by_model_never_probes_or_evicts(monkeypatch):
    """The whole gate must be skipped — not merely survive. Probing at all is the bug: the
    eviction it performs is what unloads the model the caller is about to dial."""
    model_registry.MODELS.clear()
    model_registry.register(ModelSpec(id="chat-expert-a", loader=lambda: "chat-expert-a",
                                      est_vram_gb=0.0, served_by=("ollama", "vllm")))

    called = {"probe": 0}

    def _boom(*a, **k):
        called["probe"] += 1
        raise AssertionError("probe_box_residents must not run for a server-served model")

    monkeypatch.setattr("queue_workflows.model_residency.probe_box_residents", _boom)
    gpu_model_cache._box_load_gate("chat-expert-a")      # must not raise
    assert called["probe"] == 0


def test_an_UNMARKED_model_still_goes_through_the_gate(monkeypatch):
    """The bypass must be narrow. An in-process model (tts-model-v1 — no `served_by`) is
    exactly what the one-model-per-box gate exists to protect, so it must still probe."""
    model_registry.MODELS.clear()
    model_registry.register(ModelSpec(id="tts-model-v1", loader=lambda: None))

    seen = {"probe": 0}
    monkeypatch.setattr("queue_workflows.model_residency.probe_box_residents",
                        lambda *a, **k: (seen.__setitem__("probe", seen["probe"] + 1), [])[1])
    monkeypatch.setattr("queue_workflows.model_residency.clear_box_for",
                        lambda *a, **k: [])
    gpu_model_cache._box_load_gate("tts-model-v1")
    assert seen["probe"] >= 1, "an in-process model must still be gated"


def test_an_UNREGISTERED_id_still_goes_through_the_gate(monkeypatch):
    """Fail-safe direction: an id with no spec is not evidence that it is server-served."""
    model_registry.MODELS.clear()
    seen = {"probe": 0}
    monkeypatch.setattr("queue_workflows.model_residency.probe_box_residents",
                        lambda *a, **k: (seen.__setitem__("probe", seen["probe"] + 1), [])[1])
    monkeypatch.setattr("queue_workflows.model_residency.clear_box_for", lambda *a, **k: [])
    gpu_model_cache._box_load_gate("something-nobody-registered")
    assert seen["probe"] >= 1


# ── the canonical-tag half (defence in depth) ────────────────────────
def test_a_resident_under_latest_is_NOT_foreign_to_the_bare_id():
    """`assert_can_load('chat-model-a', [Resident('chat-model-a:latest','ollama')])` must pass. ollama
    reports the resolved tag; a load asks for the bare one. Comparing them literally is what
    made a box refuse to load the model it already held."""
    from queue_workflows import model_residency as mr

    residents = [mr.Resident(model="chat-model-a:latest", server="ollama")]
    mr.assert_can_load("chat-model-a", residents, label="test-box")   # must not raise


def test_a_GENUINELY_foreign_resident_is_still_refused():
    """The canonicalisation must not become a blanket pass — otherwise one-model-per-box is
    off and the gate protects nothing."""
    from queue_workflows import model_residency as mr

    residents = [mr.Resident(model="chat-model-a:latest", server="ollama")]
    with pytest.raises(mr.ModelAlreadyLoadedError) as ei:
        mr.assert_can_load("chat-expert-a", residents, label="test-box")
    assert "chat-model-a:latest[ollama]" in str(ei.value)
