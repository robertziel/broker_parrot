"""Per-box LLM TOPOLOGY (``queue_workflows.llm_backends.topology``) + the backend
factory's use of it.

WHAT THIS PINS. A deployment can map each worker box (by ``host_label``) to the LLM
server ROOT URL that box should dispatch to, via a small YAML. The loader parses the
bare-string + ``{url: …}`` forms, falls back to a ``default:`` row, and NEVER raises
on an absent/broken file (dispatch must not die on a bad topology file). The factory
then prefers the topology URL — keyed by THIS box's ``host_label`` env — over the
``ollama_url_env`` / localhost default, and falls back cleanly when unset.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from queue_workflows.config import get_config
from queue_workflows.llm_backends import factory as factory_mod
from queue_workflows.llm_backends import topology
from queue_workflows.llm_backends.factory import BackendFactory


@pytest.fixture(autouse=True)
def _reset():
    factory_mod.reset_default_for_tests()
    topology._reset_cache_for_tests()
    yield
    factory_mod.reset_default_for_tests()
    topology._reset_cache_for_tests()


def _write(tmp_path, body: str) -> str:
    p = tmp_path / "llm_topology.yml"
    p.write_text(textwrap.dedent(body))
    return str(p)


# ── the loader ───────────────────────────────────────────────────────────────


def test_resolve_bare_string_and_mapping_forms(tmp_path):
    path = _write(
        tmp_path,
        """
        boxes:
          box-b:     http://127.0.0.1:11434
          box-a-gpu: { url: http://box-b-fast:11434 }
        """,
    )
    assert topology.resolve(path, "box-b") == {"url": "http://127.0.0.1:11434"}
    assert topology.resolve(path, "box-a-gpu")["url"] == "http://box-b-fast:11434"


def test_resolve_falls_back_to_default_then_none(tmp_path):
    path = _write(tmp_path, "boxes:\n  default: http://box-b-fast:11434\n")
    assert topology.resolve(path, "whoever")["url"] == "http://box-b-fast:11434"


def test_resolve_none_for_missing_file_or_unmatched(tmp_path):
    assert topology.resolve(str(tmp_path / "nope.yml"), "box-b") is None
    path = _write(tmp_path, "boxes:\n  box-b: http://x:11434\n")
    assert topology.resolve(path, "other") is None  # no default row → None


def test_broken_file_resolves_to_none_not_raise(tmp_path):
    path = _write(tmp_path, "just a scalar, no boxes mapping\n")
    assert topology.resolve(path, "box-b") is None


def test_missing_pyyaml_resolves_to_none_not_raise(tmp_path, monkeypatch):
    """PyYAML is an OPTIONAL extra — psycopg stays the only hard runtime dep. A
    deployment that sets ``llm_topology_path`` without installing it must degrade to
    the env/localhost default (the module's "never breaks dispatch" contract), NOT
    blow up the node thread with ModuleNotFoundError."""
    path = _write(tmp_path, "boxes:\n  box-b: http://127.0.0.1:11434\n")
    # sys.modules[name] = None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "yaml", None)
    topology._reset_cache_for_tests()
    assert topology.resolve(path, "box-b") is None


# ── the factory honors it (keyed by host_label) ──────────────────────────────


def _plain_factory():
    """A factory with the DEFAULT URL resolution (no ``ollama_url`` override) so the
    topology → env → localhost-default chain is what decides ``base_url``; a fake
    build keeps it off the network."""
    built = []

    def build(server_type, base_url, parallelism, idle_ttl_s):
        b = type("B", (), {})()
        b.server_type, b.base_url = server_type, base_url
        b.shutdown = lambda: None
        built.append(b)
        return b

    f = BackendFactory(now_fn=lambda: 0.0, build_backend_fn=build)
    return f, built


def test_factory_uses_topology_url_keyed_by_host_label(tmp_path, monkeypatch):
    path = _write(tmp_path, "boxes:\n  box-a-gpu: http://box-b-fast:11434\n")
    cfg = get_config()
    monkeypatch.setattr(cfg, "llm_topology_path", path)
    monkeypatch.setenv(cfg.host_label_env, "box-a-gpu")
    f, _ = _plain_factory()
    # No DB row for this host → ollama; the URL comes from the TOPOLOGY (this box's
    # host_label), not the env / localhost default.
    assert f.get_backend("box-a-gpu", "gpu").base_url == "http://box-b-fast:11434"


def test_factory_falls_back_to_default_when_no_topology(monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(cfg, "llm_topology_path", None)
    monkeypatch.setenv(cfg.host_label_env, "box-a-gpu")
    f, _ = _plain_factory()
    assert f.get_backend("box-a-gpu", "gpu").base_url == factory_mod.DEFAULT_OLLAMA_URL


def test_resolve_base_url_returns_topology_without_building_a_backend(tmp_path, monkeypatch):
    """The heartbeat probe needs THIS box's URL without a DB read or a backend build.
    resolve_base_url gives the same topology→env→default answer get_backend would use."""
    path = _write(tmp_path, "boxes:\n  box-a-gpu: http://host.docker.internal:11434\n")
    cfg = get_config()
    monkeypatch.setattr(cfg, "llm_topology_path", path)
    monkeypatch.setenv(cfg.host_label_env, "box-a-gpu")
    f, built = _plain_factory()
    assert f.resolve_base_url() == "http://host.docker.internal:11434"
    assert built == []  # resolving a URL must NOT construct a backend


def test_resolve_base_url_defaults_when_unset(monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(cfg, "llm_topology_path", None)
    monkeypatch.setenv(cfg.host_label_env, "box-a-gpu")
    f, _ = _plain_factory()
    assert f.resolve_base_url() == factory_mod.DEFAULT_OLLAMA_URL
