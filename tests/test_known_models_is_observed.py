"""The claim gate advertises what the box HOLDS — and is actually WIRED to do so.

`model_catalog.narrow_to_observed` is unit-tested next door. This file exists because a
correct helper that nothing calls is indistinguishable from one that does not exist: the
model catalog itself shipped complete, migrated, fleet-published and read by NOTHING, and
that is precisely what made `force_box` un-retirable. So these tests assert the seam is
fed, at both ends.
"""
from __future__ import annotations

import inspect

from queue_workflows.claim_worker import ClaimWorker, HeartbeatEmitter


# ── the narrowing itself, on the object that owns it ─────────────────
def _worker_stub(observed):
    """A bare ClaimWorker — `_known_models` reads only `self._observed_model_ids`, so it
    needs no DB, no queue and no constructor. Binding the unbound method is what keeps
    this test honest: it exercises the SHIPPED method, not a re-implementation."""
    stub = object.__new__(ClaimWorker)
    if observed is not None:
        stub._observed_model_ids = observed
    return stub


def test_a_declared_model_the_box_does_not_hold_is_not_advertised():
    """Through the REAL method, which consults the registry for what is server-served —
    so this also proves `_known_models` passes `narrowable` rather than narrowing blindly."""
    from queue_workflows import model_registry
    from queue_workflows.model_registry import ModelSpec

    saved = dict(model_registry.MODELS)
    try:
        for mid in ("chat-model-a:latest", "chat-expert-b", "chat-expert-a"):
            model_registry.register(ModelSpec(id=mid, loader=lambda: None,
                                              served_by=("ollama", "vllm")))
        stub = _worker_stub(["chat-model-a:latest", "chat-expert-b:latest"])
        assert ClaimWorker._known_models(
            stub, ["chat-model-a:latest", "chat-expert-b", "chat-expert-a"]
        ) == ["chat-model-a:latest", "chat-expert-b"]
    finally:
        model_registry.MODELS.clear()
        model_registry.MODELS.update(saved)


def test_an_UNMARKED_model_survives_the_real_method():
    """The wedge, end to end through `_known_models`: tts-model-v1 is registered with no
    `served_by`, so the probe's blindness to it must not delete it from the fleet."""
    from queue_workflows import model_registry
    from queue_workflows.model_registry import ModelSpec

    saved = dict(model_registry.MODELS)
    try:
        model_registry.MODELS.clear()
        model_registry.register(ModelSpec(id="chat-expert-b", loader=lambda: None,
                                          served_by=("ollama",)))
        model_registry.register(ModelSpec(id="tts-model-v1", loader=lambda: None))
        stub = _worker_stub(["chat-expert-b:latest"])
        assert ClaimWorker._known_models(stub, ["chat-expert-b", "tts-model-v1"]) == [
            "chat-expert-b", "tts-model-v1"]
    finally:
        model_registry.MODELS.clear()
        model_registry.MODELS.update(saved)


def test_a_worker_that_has_observed_NOTHING_advertises_its_full_registry():
    """Never wedge: a cold worker, or one whose probe has not run, must keep claiming."""
    for observed in (None, [], ()):
        assert ClaimWorker._known_models(
            _worker_stub(observed), ["a", "b"]) == ["a", "b"]


def test_narrowing_failure_falls_back_to_declared_rather_than_starving_the_box():
    """Fail-open. `_observed_model_ids` set to something unusable must not empty the
    advertisement — an exception here would otherwise stop every LLM claim on the box."""
    class Exploding:
        def __iter__(self):
            raise RuntimeError("boom")

    stub = object.__new__(ClaimWorker)
    stub._observed_model_ids = Exploding()
    assert ClaimWorker._known_models(stub, ["a", "b"]) == ["a", "b"]


# ── the WIRING: every consumer of the declared registry goes through it ──
def test_the_heartbeat_emitter_applies_the_injected_narrowing():
    """Behavioural, not source-level: emit_once must send the NARROWED list."""
    captured = {}

    emitter = HeartbeatEmitter(
        queue="gpu", host_label="box-b-gpu",
        llm_servers_fn=lambda: ["ollama"],
        known_models_fn=lambda declared: [m for m in declared if m != "chat-expert-a"],
    )

    import queue_workflows.claim_worker as cw

    real_upsert = cw.node_queue.upsert_worker_heartbeat
    real_known = cw.model_registry.known_ids if hasattr(cw, "model_registry") else None
    try:
        cw.node_queue.upsert_worker_heartbeat = lambda **kw: captured.update(kw)
        from queue_workflows import model_registry
        real_known = model_registry.known_ids
        model_registry.known_ids = lambda: ["chat-model-a:latest", "chat-expert-a"]
        emitter.emit_once()
    finally:
        cw.node_queue.upsert_worker_heartbeat = real_upsert
        if real_known is not None:
            from queue_workflows import model_registry
            model_registry.known_ids = real_known

    assert captured["known_models"] == ["chat-model-a:latest"], (
        "the heartbeat advertised the raw registry — `fits_models` derives from this "
        "list, so an un-narrowed advertisement propagates into every capacity view")


def test_the_emitter_default_is_IDENTITY_so_cpu_workers_are_unchanged():
    """cpu/ingest workers observe no models; their heartbeat must be byte-identical."""
    emitter = HeartbeatEmitter(queue="cpu", host_label="box-c-cpu")
    assert emitter._known_models_fn(["a", "b"]) == ["a", "b"]


def test_the_claim_worker_INJECTS_its_narrowing_into_the_heartbeat():
    """The end the behavioural test above cannot reach: ClaimWorker builds the emitter,
    and if it stops passing `known_models_fn` the default identity silently restores the
    old dishonest advertisement with every test still green."""
    src = inspect.getsource(ClaimWorker)
    assert "known_models_fn=self._known_models" in src


def test_BOTH_gpu_claim_call_sites_pass_the_narrowed_list():
    """The gate itself. `known_models` is what the claim SQL matches `required_model`
    (and `required_model_ids`) against — a raw `model_registry.known_ids()` here is a box
    claiming an expert it has never pulled, which is the exact failure `force_box` was
    holding shut."""
    src = inspect.getsource(ClaimWorker)
    assert "known_models=model_registry.known_ids()," not in src, (
        "a gpu claim advertises the DECLARED registry — every box registers the same "
        "MODELS dict at import, so this is not a capability statement")
    assert "known_models=fits," not in src, (
        "the VRAM-filtered claim advertises capacity, not possession")
    assert src.count("known_models=self._known_models(") == 2


def test_the_publisher_records_what_it_observed_for_the_gate_to_use():
    """The catalog publish and the claim gate must read ONE observation. If the publisher
    stops stamping `_observed_model_ids`, narrowing silently becomes a no-op and every
    box goes back to advertising the full registry — green tests, dead seam."""
    src = inspect.getsource(ClaimWorker._publish_model_catalog)
    assert "self._observed_model_ids" in src


# ── the regression that made this file's premise wrong ───────────────
def test_an_IN_PROCESS_model_survives_an_observation_that_cannot_see_it():
    """The bug this narrowing shipped with.

    The observation is ollama /api/tags + vLLM /v1/models. A model the worker loads
    IN-PROCESS (tts-model-v1, via the warm ModelCache) can NEVER appear there, so
    narrowing against it deleted the id from every box's advertisement at once and made
    every job that needed it unclaimable fleet-wide. Worse, it was self-concealing: the lane
    came back only while a box's ollama was BROKEN, because an empty observation
    fails open to the full registry.
    """
    from queue_workflows.model_catalog import narrow_to_observed

    declared = ["chat-expert-b", "tts-model-v1", "chat-model-a:latest"]
    observed = ["chat-expert-b:latest", "chat-model-a:latest"]   # servers cannot report tts-model-v1

    assert narrow_to_observed(declared, observed) == ["chat-expert-b", "chat-model-a:latest"], (
        "sanity: with no `narrowable` the in-process model IS dropped — that was the bug")
    assert narrow_to_observed(
        declared, observed, narrowable={"chat-expert-b", "chat-model-a:latest"},
    ) == declared, "an in-process model must survive a probe that cannot see it"


def test_a_server_served_model_the_box_lacks_is_STILL_dropped():
    """The protection must not become a blanket pass-through — otherwise the gate is
    back to advertising the declared registry and force_box was retired on nothing."""
    from queue_workflows.model_catalog import narrow_to_observed

    kept = narrow_to_observed(
        ["chat-expert-b", "chat-expert-a", "tts-model-v1"], ["chat-expert-b:latest"],
        narrowable={"chat-expert-b", "chat-expert-a"})
    assert kept == ["chat-expert-b", "tts-model-v1"], kept


def test_an_observation_matching_NOTHING_falls_back_to_declared_not_to_empty():
    """`[]` reads downstream as "no capability list", which turns the claim gate OFF
    (node_queue gates on `if known:`) — strictly WIDER than never narrowing. The fallback
    must be the declared list, which is at least as tight."""
    from queue_workflows.model_catalog import narrow_to_observed

    assert narrow_to_observed(["chat-expert-b"], ["something-else"]) == ["chat-expert-b"]


def test_the_registry_defaults_to_PROTECTED_so_an_unmarked_spec_is_never_narrowed():
    """Fail-safe direction, and it matters across projects: a host registers in-process
    torch/diffusers specs that know nothing about `served_by`. Unmarked must mean
    in-process, never 'assume a server has it'."""
    from queue_workflows import model_registry
    from queue_workflows.model_registry import ModelSpec

    saved = dict(model_registry.MODELS)
    try:
        model_registry.MODELS.clear()
        model_registry.register(ModelSpec(id="in-process", loader=lambda: None))
        model_registry.register(ModelSpec(id="served", loader=lambda: None,
                                          served_by=("ollama", "vllm")))
        assert model_registry.server_served_ids() == {"served"}
    finally:
        model_registry.MODELS.clear()
        model_registry.MODELS.update(saved)
