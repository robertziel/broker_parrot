"""Co-resident LLM backends — per-kind resolution (migration 0022).

Under Option A a box may run ollama AND vLLM at once, so the factory can no longer
hold ONE backend per ``(host, queue)``: asking for ollama must not tear down a live
vLLM (that would reset its request counters and state machine mid-flight). These pin
the re-keyed cache, the ordered-preference choice, and the per-kind topology address.
"""
from __future__ import annotations

import pytest

from queue_workflows.llm_backends import choose_server_type, topology
from queue_workflows.llm_backends.factory import BackendFactory


# ── choose_server_type: where the ORDER in server_types is interpreted ────────


@pytest.mark.parametrize("prefs,have,want", [
    (["vllm", "ollama"], ["ollama", "vllm"], "vllm"),   # first preference wins
    (["vllm", "ollama"], ["ollama"], "ollama"),         # falls back down the list
    (["ollama", "vllm"], ["ollama", "vllm"], "ollama"),  # order is the preference
    (["vllm"], ["ollama"], None),                        # no overlap ⇒ box default
    ([], ["ollama"], None),                              # no preference ⇒ box default
    (None, ["ollama"], None),
    (["vllm", "ollama"], [], "vllm"),                    # unprobed box still honours it
    (["vllm", "ollama"], None, "vllm"),
])
def test_choose_server_type(prefs, have, want):
    assert choose_server_type(prefs, have) == want


# ── the re-keyed cache ───────────────────────────────────────────────────────


def _factory(**kw):
    built = []

    def _build(server_type, base_url, parallelism, idle_ttl_s):
        class _Fake:
            def __init__(self):
                self.server_type = server_type
                self.base_url = base_url
            def shutdown(self):
                built.append(("shutdown", server_type))
        built.append(("build", server_type))
        return _Fake()

    f = BackendFactory(
        build_backend_fn=_build, supervisor_factory=lambda b: None,
        ollama_url="http://o:11434", vllm_url="http://v:8000", **kw,
    )
    return f, built


def test_two_kinds_coexist_for_one_host_queue():
    f, built = _factory()
    a = f.get_backend("h", "gpu", "ollama")
    b = f.get_backend("h", "gpu", "vllm")
    assert a is not b
    assert a.server_type == "ollama" and b.server_type == "vllm"
    assert ("shutdown", "ollama") not in built, "asking for vllm must not kill ollama"


def test_identity_is_preserved_per_kind():
    f, _ = _factory()
    assert f.get_backend("h", "gpu", "vllm") is f.get_backend("h", "gpu", "vllm")


def test_no_server_type_uses_the_box_default():
    f, _ = _factory()
    # No worker_controls row ⇒ LLMConfig default is ollama.
    assert f.get_backend("h", "gpu").server_type == "ollama"


# ── per-kind topology ────────────────────────────────────────────────────────


def test_topology_per_kind_urls(tmp_path):
    p = tmp_path / "topo.yml"
    p.write_text(
        "boxes:\n"
        "  box-a-gpu:\n"
        "    ollama: http://127.0.0.1:11434\n"
        "    vllm: http://127.0.0.1:8000\n"
    )
    topology._reset_cache_for_tests()
    assert topology.resolve_for(str(p), "box-a-gpu", "ollama") == "http://127.0.0.1:11434"
    assert topology.resolve_for(str(p), "box-a-gpu", "vllm") == "http://127.0.0.1:8000"


def test_bare_url_entry_still_answers_for_both_kinds(tmp_path):
    """Back-compat: every topology written before 0022 is a bare URL and must keep
    serving both kinds, or an existing deployment breaks on upgrade."""
    p = tmp_path / "topo.yml"
    p.write_text("boxes:\n  box-a-gpu: http://127.0.0.1:11434\n")
    topology._reset_cache_for_tests()
    assert topology.resolve_for(str(p), "box-a-gpu", "ollama") == "http://127.0.0.1:11434"
    assert topology.resolve_for(str(p), "box-a-gpu", "vllm") == "http://127.0.0.1:11434"


def test_missing_entry_resolves_none_so_the_caller_falls_back(tmp_path):
    p = tmp_path / "topo.yml"
    p.write_text("boxes:\n  other-gpu: http://x:11434\n")
    topology._reset_cache_for_tests()
    assert topology.resolve_for(str(p), "box-a-gpu", "vllm") is None
