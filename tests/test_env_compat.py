"""Env names — every knob is ``QUEUE_WORKFLOWS_*``; the retired ``AI_LEADS_*``
twins are an error, not a fallback.

Contract:

  1. the canonical name is read;
  2. a legacy ``AI_LEADS_*`` twin still set in the environment RAISES, naming
     its replacement — an un-migrated deploy fails at startup instead of
     silently losing its setting;
  3. a host-configured CUSTOM env name (``configure(db_url_env="MYAPP_DB_URL")``)
     is read verbatim;
  4. no engine module names an ``AI_LEADS_*`` variable any more (source scan).
"""
from __future__ import annotations

import pathlib

import pytest

from queue_workflows import envcompat


def test_canonical_is_read(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_FOO_KNOB", "canonical")
    assert envcompat.env_get("QUEUE_WORKFLOWS_FOO_KNOB") == "canonical"


def test_legacy_twin_raises_with_its_replacement(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_FOO_KNOB", raising=False)
    monkeypatch.setenv("AI_LEADS_FOO_KNOB", "legacy")
    with pytest.raises(RuntimeError, match="AI_LEADS_FOO_KNOB .*rename it to QUEUE_WORKFLOWS_FOO_KNOB"):
        envcompat.env_get("QUEUE_WORKFLOWS_FOO_KNOB")


def test_legacy_twin_raises_even_beside_the_canonical(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_FOO_KNOB", "canonical")
    monkeypatch.setenv("AI_LEADS_FOO_KNOB", "legacy")
    with pytest.raises(RuntimeError):
        envcompat.env_get("QUEUE_WORKFLOWS_FOO_KNOB")


def test_custom_name_read_verbatim(monkeypatch):
    monkeypatch.setenv("MYAPP_DB_URL", "postgresql://custom")
    assert envcompat.env_get("MYAPP_DB_URL") == "postgresql://custom"


def test_default_when_unset(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_FOO_KNOB", raising=False)
    monkeypatch.delenv("AI_LEADS_FOO_KNOB", raising=False)
    assert envcompat.env_get("QUEUE_WORKFLOWS_FOO_KNOB") is None
    assert envcompat.env_get("QUEUE_WORKFLOWS_FOO_KNOB", "dflt") == "dflt"


def test_a_real_knob_refuses_its_legacy_spelling(monkeypatch):
    from queue_workflows import claim_worker
    monkeypatch.setenv("QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES", "7")
    monkeypatch.delenv("AI_LEADS_WATCHDOG_MAX_RETRIES", raising=False)
    assert claim_worker._watchdog_max_retries() == 7
    monkeypatch.setenv("AI_LEADS_WATCHDOG_MAX_RETRIES", "5")
    with pytest.raises(RuntimeError, match="QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES"):
        claim_worker._watchdog_max_retries()


def test_env_name_defaults_are_canonical():
    from queue_workflows import config as _config
    c = _config.EngineConfig()
    assert c.db_url_env == "QUEUE_WORKFLOWS_DB_URL"
    assert c.host_label_env == "QUEUE_WORKFLOWS_HOST_LABEL"
    assert c.host_priority_env == "QUEUE_WORKFLOWS_GPU_CONSUMER_PRIORITY"
    assert c.ollama_url_env == "QUEUE_WORKFLOWS_OLLAMA_URL"
    assert c.vllm_url_env == "QUEUE_WORKFLOWS_VLLM_URL"
    assert c.container_prefix == ""  # no host-specific default


def test_no_legacy_env_names_in_source():
    pkg = pathlib.Path(envcompat.__file__).parent
    offenders = []
    for path in [*pkg.rglob("*.py"), *pkg.rglob("*.sql")]:
        if path.name == "envcompat.py":
            continue  # names the retired prefix to refuse it
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "AI_LEADS_" in line:
                offenders.append(f"{path.name}:{i}: {line.strip()[:70]}")
    assert not offenders, "retired AI_LEADS_* names in the engine:\n  " + "\n  ".join(offenders)
