"""The ARTIFACT behind the role tag — resolved from observation, never declared into truth.

Exists because of an eight-day incident in a consumer deployment: the
`chat-expert-a` role tag served a GGUF built from a void merge — the untuned base — and every
job row truthfully recorded `resolved_model='chat-expert-a'` the whole time. A stable role
name can never answer "which bytes"; these tests pin the machinery that can.
"""
from __future__ import annotations

import inspect

from queue_workflows.model_catalog import CatalogEntry, artifact_role, resolve_artifact


def _e(model, digest="", kind="ollama", host="box-a-gpu", enabled=True):
    return CatalogEntry(host, kind, model, enabled=enabled, digest=digest)


# ── the grammar ──────────────────────────────────────────────────────
def test_artifact_role_parses_the_convention_and_only_the_convention():
    assert artifact_role("chat-expert-a-v6-q4km-ollama") == "chat-expert-a"
    assert artifact_role("chat-expert-a-v6-q4km-ollama:latest") == "chat-expert-a"
    assert artifact_role("chat-expert-b-v1-q4km-ollama") == "chat-expert-b"
    assert artifact_role("chat-expert-a-v6-bf16-vllm") == "chat-expert-a"
    # a ROLE tag is not an artifact tag — that distinction is the whole convention
    assert artifact_role("chat-expert-a") is None
    assert artifact_role("chat-model-a:latest") is None
    assert artifact_role("") is None


# ── digest match: the by-bytes answer ────────────────────────────────
def test_digest_match_ties_the_role_to_the_artifact_it_serves():
    entries = [
        _e("chat-expert-a:latest", "sha256-AAA"),
        _e("chat-expert-a-v6-q4km-ollama:latest", "sha256-AAA"),
        _e("chat-expert-b:latest", "sha256-BBB"),
        _e("chat-model-a:latest", "sha256-CCC"),
    ]
    assert resolve_artifact("chat-expert-a", "ollama", entries) == "chat-expert-a-v6-q4km-ollama"


def test_a_digest_equal_tag_of_ANOTHER_role_is_never_the_answer():
    """The grammar constrains the digest match: sharing bytes is not enough, the tag must
    name THIS role. (A copied tag under a different role name must not cross-label.)"""
    entries = [
        _e("chat-expert-a:latest", "sha256-AAA"),
        _e("chat-expert-b-v1-q4km-ollama:latest", "sha256-AAA"),   # same bytes, other role
    ]
    assert resolve_artifact("chat-expert-a", "ollama", entries) is None


def test_after_a_promotion_the_next_observation_wins():
    """The stamp must follow the re-point: v5 and v6 both installed, role now on v6."""
    entries = [
        _e("chat-expert-a:latest", "sha256-V6"),
        _e("chat-expert-a-v5-q4km-ollama:latest", "sha256-V5"),
        _e("chat-expert-a-v6-q4km-ollama:latest", "sha256-V6"),
    ]
    assert resolve_artifact("chat-expert-a", "ollama", entries) == "chat-expert-a-v6-q4km-ollama"


def test_declared_options_break_ties_but_cannot_conjure_evidence():
    both = [
        _e("chat-expert-a:latest", "sha256-X"),
        _e("chat-expert-a-v6-q4km-ollama:latest", "sha256-X"),
        _e("chat-expert-a-v6-q8-ollama:latest", "sha256-X"),
    ]
    assert resolve_artifact("chat-expert-a", "ollama", both,
                            options=("chat-expert-a-v6-q8-ollama",)) == "chat-expert-a-v6-q8-ollama"
    # with NO observed candidate, a declared option contributes nothing — stamping the
    # declaration would be the exact assertion-nobody-verifies failure again
    assert resolve_artifact("chat-expert-a", "ollama",
                            [_e("chat-expert-a:latest", "sha256-X")],
                            options=("chat-expert-a-v6-q4km-ollama",)) is None


# ── absent, never guessed ────────────────────────────────────────────
def test_no_digest_on_the_role_means_no_ollama_answer():
    entries = [
        _e("chat-expert-a:latest", ""),                        # server said nothing
        _e("chat-expert-a-v6-q4km-ollama:latest", "sha256-A"),
    ]
    assert resolve_artifact("chat-expert-a", "ollama", entries) is None


def test_disabled_rows_are_invisible():
    entries = [
        _e("chat-expert-a:latest", "sha256-A"),
        _e("chat-expert-a-v6-q4km-ollama:latest", "sha256-A", enabled=False),
    ]
    assert resolve_artifact("chat-expert-a", "ollama", entries) is None


# ── the vLLM shape: no digests, the served id IS the artifact ────────
def test_vllm_serving_the_artifact_directly_resolves_without_digests():
    entries = [_e("chat-expert-a-v6-bf16-vllm", "", kind="vllm")]
    assert resolve_artifact("chat-expert-a", "vllm", entries) == "chat-expert-a-v6-bf16-vllm"


# ── the registry side ────────────────────────────────────────────────
def test_server_model_options_is_empty_never_raising_for_the_unknown():
    from queue_workflows import model_registry
    from queue_workflows.model_registry import ModelSpec

    saved = dict(model_registry.MODELS)
    try:
        model_registry.MODELS.clear()
        model_registry.register(ModelSpec(
            id="chat-expert-a", loader=lambda: None, served_by=("ollama", "vllm"),
            server_models={"ollama": ("chat-expert-a-v6-q4km-ollama",),
                           "vllm": ("chat-expert-a-v6-bf16-vllm",)}))
        assert model_registry.server_model_options("chat-expert-a") == {
            "ollama": ("chat-expert-a-v6-q4km-ollama",),
            "vllm": ("chat-expert-a-v6-bf16-vllm",)}
        assert model_registry.server_model_options("nope") == {}
        model_registry.register(ModelSpec(id="bare", loader=lambda: None))
        assert model_registry.server_model_options("bare") == {}
    finally:
        model_registry.MODELS.clear()
        model_registry.MODELS.update(saved)


# ── the wiring: observation keeps digests, the claim stamps ──────────
def test_the_probe_carries_ollama_digests_into_catalog_entries():
    from queue_workflows.llm_probe import observe_box_models

    entries = observe_box_models(
        "box-a-gpu",
        urls_by_kind={"ollama": "http://x:11434", "vllm": "http://x:8000"},
        ollama_fn=lambda url, timeout_s: [("chat-expert-a:latest", "sha256-D"),
                                          ("chat-expert-a-v6-q4km-ollama:latest", "sha256-D")],
        vllm_fn=lambda url, timeout_s: ["chat-expert-a-v6-bf16-vllm"],
    )
    by_id = {(e.server_type, e.model_id): e.digest for e in entries}
    assert by_id[("ollama", "chat-expert-a:latest")] == "sha256-D"
    assert by_id[("vllm", "chat-expert-a-v6-bf16-vllm")] == ""
    assert resolve_artifact("chat-expert-a", "ollama", entries) == "chat-expert-a-v6-q4km-ollama"


def test_the_gpu_claim_path_calls_the_stamp():
    """Source-level, like the known_models wiring tests next door: the stamp must ride the
    successful claim, or the column stays NULL forever with every unit test green."""
    from queue_workflows.claim_worker import ClaimWorker

    src = inspect.getsource(ClaimWorker)
    assert "self._stamp_resolved_artifact(job)" in src
    assert "self._observed_entries = list(entries)" in src, (
        "the observation no longer keeps the full entries — artifact resolution has no "
        "digests to work from and silently stamps nothing")


def test_the_stamp_resolves_from_the_workers_own_observation():
    """Behavioural: a stub worker with a live observation stamps the digest-matched tag."""
    from queue_workflows import node_queue
    from queue_workflows.claim_worker import ClaimWorker

    stub = object.__new__(ClaimWorker)
    stub._observed_entries = [
        _e("chat-expert-a:latest", "sha256-Z"),
        _e("chat-expert-a-v6-q4km-ollama:latest", "sha256-Z"),
    ]
    captured = {}
    real = node_queue.set_resolved_artifact
    try:
        node_queue.set_resolved_artifact = lambda jid, art: captured.update(
            {"id": jid, "artifact": art})
        job = {"id": "j1", "resolved_model": "chat-expert-a", "resolved_server": "ollama"}
        ClaimWorker._stamp_resolved_artifact(stub, job)
    finally:
        node_queue.set_resolved_artifact = real
    assert captured == {"id": "j1", "artifact": "chat-expert-a-v6-q4km-ollama"}
    assert job["resolved_artifact"] == "chat-expert-a-v6-q4km-ollama"


def test_the_stamp_stays_silent_without_evidence():
    from queue_workflows import node_queue
    from queue_workflows.claim_worker import ClaimWorker

    stub = object.__new__(ClaimWorker)          # no _observed_entries at all
    calls = []
    real = node_queue.set_resolved_artifact
    try:
        node_queue.set_resolved_artifact = lambda *a: calls.append(a)
        job = {"id": "j1", "resolved_model": "chat-expert-a", "resolved_server": "ollama"}
        ClaimWorker._stamp_resolved_artifact(stub, job)
    finally:
        node_queue.set_resolved_artifact = real
    assert calls == [] and "resolved_artifact" not in job
