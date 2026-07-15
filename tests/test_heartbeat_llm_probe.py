"""The heartbeat advertises PROBED llm capability, not a static guess.

Pins the wiring that closes the incident (see ``test_llm_probe`` for the root cause):
``HeartbeatEmitter`` takes an injectable ``llm_servers_fn`` seam. Unset, it publishes
``config.llm_servers_available`` exactly as before (byte-compatible for cpu/ingest and
other consumers). The GPU worker wires it to a live endpoint probe, so a box that
serves no LLM advertises ``[]`` — and the "OLLAMA · ON" chip disappears — instead of
claiming an ollama it never had.
"""

from __future__ import annotations

import pytest

from queue_workflows import claim_worker, model_registry


@pytest.fixture
def capture(monkeypatch):
    """Drive HeartbeatEmitter.emit_once with no DB and a deterministic registry:
    capture the upsert kwargs, empty the model registry, and stub the VRAM probe."""
    box = {"kwargs": None}
    monkeypatch.setattr(
        claim_worker.node_queue, "upsert_worker_heartbeat",
        lambda **kw: box.__setitem__("kwargs", kw),
    )
    monkeypatch.setattr(model_registry, "known_ids", lambda: [])
    # config.llm_servers_available is the fallback path; the probe seam overrides it
    monkeypatch.setattr(
        claim_worker, "get_config",
        lambda: type("C", (), {"llm_servers_available": ["ollama"], "project": ""})(),
    )
    return box


def _emitter(monkeypatch, **kw):
    em = claim_worker.HeartbeatEmitter(queue="gpu", host_label="box-a-gpu", **kw)
    # VRAM probe is a stable hardware read; stub it so emit_once needs no GPU.
    monkeypatch.setattr(em, "_vram_total_mb", lambda: None)
    return em


def test_default_publishes_static_config(monkeypatch, capture):
    """No seam wired ⇒ the historical behaviour: whatever config advertises."""
    em = _emitter(monkeypatch)
    em.emit_once()
    assert capture["kwargs"]["llm_servers_available"] == ["ollama"]


def test_probe_seam_overrides_config(monkeypatch, capture):
    em = _emitter(monkeypatch, llm_servers_fn=lambda: [])  # probe says NOTHING
    em.emit_once()
    assert capture["kwargs"]["llm_servers_available"] == []  # not ["ollama"]


def test_probe_seam_reports_vllm(monkeypatch, capture):
    em = _emitter(monkeypatch, llm_servers_fn=lambda: ["vllm"])
    em.emit_once()
    assert capture["kwargs"]["llm_servers_available"] == ["vllm"]


def test_probe_seam_failure_does_not_crash_heartbeat(monkeypatch, capture):
    """A probe that raises must not take down the liveness signal — the row still
    upserts, and the field degrades to the static config value."""
    def boom():
        raise RuntimeError("probe blew up")

    em = _emitter(monkeypatch, llm_servers_fn=boom)
    em.emit_once()  # must not raise
    assert capture["kwargs"] is not None
    assert capture["kwargs"]["llm_servers_available"] == ["ollama"]  # fell back
