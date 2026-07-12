"""Env-name compatibility — canonical ``QUEUE_WORKFLOWS_*`` with ``AI_LEADS_*`` fallback.

The engine was extracted from a host application whose env names it inherited.
As a standalone library its knobs are now canonically ``QUEUE_WORKFLOWS_*``;
every legacy ``AI_LEADS_*`` spelling keeps working as a silent fallback so an
existing deploy upgrades with zero .env changes. Contract:

  1. the canonical name wins when both are set;
  2. the legacy name alone still works;
  3. a host-configured CUSTOM env name (``configure(db_url_env="MYAPP_DB_URL")``)
     is read verbatim;
  4. no engine module reads ``AI_LEADS_*`` directly any more — everything goes
     through the compat helper (source-scan guard).
"""
from __future__ import annotations

import pathlib
import re

import pytest

from queue_workflows import envcompat


# ── the helper itself ────────────────────────────────────────────────────────

def test_canonical_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_FOO_KNOB", "canonical")
    monkeypatch.setenv("AI_LEADS_FOO_KNOB", "legacy")
    assert envcompat.env_get("QUEUE_WORKFLOWS_FOO_KNOB") == "canonical"
    # asking via the legacy spelling ALSO prefers the canonical value
    assert envcompat.env_get("AI_LEADS_FOO_KNOB") == "canonical"


def test_legacy_fallback(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_FOO_KNOB", raising=False)
    monkeypatch.setenv("AI_LEADS_FOO_KNOB", "legacy")
    assert envcompat.env_get("QUEUE_WORKFLOWS_FOO_KNOB") == "legacy"


def test_custom_name_read_verbatim(monkeypatch):
    monkeypatch.setenv("MYAPP_DB_URL", "postgresql://custom")
    assert envcompat.env_get("MYAPP_DB_URL") == "postgresql://custom"


def test_default_when_neither_set(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_FOO_KNOB", raising=False)
    monkeypatch.delenv("AI_LEADS_FOO_KNOB", raising=False)
    assert envcompat.env_get("QUEUE_WORKFLOWS_FOO_KNOB") is None
    assert envcompat.env_get("QUEUE_WORKFLOWS_FOO_KNOB", "dflt") == "dflt"


# ── knobs actually honour both spellings ─────────────────────────────────────

def test_watchdog_retries_canonical(monkeypatch):
    from queue_workflows import claim_worker
    monkeypatch.setenv("QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES", "7")
    monkeypatch.delenv("AI_LEADS_WATCHDOG_MAX_RETRIES", raising=False)
    assert claim_worker._watchdog_max_retries() == 7


def test_watchdog_retries_legacy(monkeypatch):
    from queue_workflows import claim_worker
    monkeypatch.delenv("QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES", raising=False)
    monkeypatch.setenv("AI_LEADS_WATCHDOG_MAX_RETRIES", "5")
    assert claim_worker._watchdog_max_retries() == 5


def test_db_url_honours_both_spellings(monkeypatch):
    """The db_url_env FIELD now defaults to the canonical name; resolution falls
    back to the legacy twin so an existing AI_LEADS_DB_URL deploy keeps working."""
    from queue_workflows import config as _config
    from queue_workflows import db
    assert _config.EngineConfig().db_url_env == "QUEUE_WORKFLOWS_DB_URL"
    monkeypatch.setattr(_config.get_config(), "db_url_env", "QUEUE_WORKFLOWS_DB_URL")
    monkeypatch.setenv("QUEUE_WORKFLOWS_DB_URL", "postgresql://canonical/db")
    monkeypatch.setenv("AI_LEADS_DB_URL", "postgresql://legacy/db")
    assert db.db_url() == "postgresql://canonical/db"
    monkeypatch.delenv("QUEUE_WORKFLOWS_DB_URL")
    assert db.db_url() == "postgresql://legacy/db"


def test_host_label_env_default_is_canonical():
    from queue_workflows import config as _config
    c = _config.EngineConfig()
    assert c.host_label_env == "QUEUE_WORKFLOWS_HOST_LABEL"
    assert c.host_priority_env == "QUEUE_WORKFLOWS_GPU_CONSUMER_PRIORITY"
    assert c.ollama_url_env == "QUEUE_WORKFLOWS_OLLAMA_URL"
    assert c.vllm_url_env == "QUEUE_WORKFLOWS_VLLM_URL"


# ── the guard: no direct legacy reads remain anywhere in the engine ─────────

def test_no_direct_legacy_env_reads_in_source():
    pkg = pathlib.Path(envcompat.__file__).parent
    offenders = []
    for py in pkg.rglob("*.py"):
        if py.name == "envcompat.py":
            continue  # the compat shim itself may name the prefix
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if re.search(r'environ\.get\(\s*"AI_LEADS_|getenv\(\s*"AI_LEADS_', line):
                offenders.append(f"{py.name}:{i}: {line.strip()[:70]}")
    assert not offenders, "direct AI_LEADS_* env reads (use envcompat.env_get):\n  " + "\n  ".join(offenders)
