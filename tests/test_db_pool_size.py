"""Pool sizing — the connection-budget knobs.

Context: a fleet's N-process claim lane (one concurrency-1 worker per core)
holds ~4-5 pooled connections per process under load — demand from the
worker's concurrent threads (claim/terminal txns, lease renewer, heartbeat,
watchers), not an eager floor (the pool has always floored at min_size=1).
The budget lever for such lanes is the CAP. Contract:

  * ``QUEUE_WORKFLOWS_DB_POOL_MAX`` caps the pool (default 10 — unchanged);
  * ``QUEUE_WORKFLOWS_DB_POOL_MIN`` (new) sets the floor (default 1 —
    exactly today's behavior);
  * min is CLAMPED to max, so no combination can crash pool construction;
  * both knobs resolve through envcompat (legacy AI_LEADS_* spellings work).
"""
from __future__ import annotations

from queue_workflows.db import _pool_sizes


def test_defaults_match_previous_behavior(monkeypatch):
    for k in ("QUEUE_WORKFLOWS_DB_POOL_MAX", "QUEUE_WORKFLOWS_DB_POOL_MIN",
              "AI_LEADS_DB_POOL_MAX", "AI_LEADS_DB_POOL_MIN"):
        monkeypatch.delenv(k, raising=False)
    assert _pool_sizes() == (1, 10)  # floor 1, cap 10 — exactly as before


def test_small_cap_keeps_floor_within_it(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_DB_POOL_MAX", "1")
    monkeypatch.setenv("QUEUE_WORKFLOWS_DB_POOL_MIN", "4")
    assert _pool_sizes() == (1, 1)  # min clamped to the cap — never a ValueError


def test_explicit_min(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_DB_POOL_MAX", "3")
    monkeypatch.setenv("QUEUE_WORKFLOWS_DB_POOL_MIN", "1")
    assert _pool_sizes() == (1, 3)


def test_min_above_max_clamps(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_DB_POOL_MAX", "3")
    monkeypatch.setenv("QUEUE_WORKFLOWS_DB_POOL_MIN", "9")
    assert _pool_sizes() == (3, 3)


def test_legacy_spellings_resolve(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_DB_POOL_MAX", raising=False)
    monkeypatch.delenv("QUEUE_WORKFLOWS_DB_POOL_MIN", raising=False)
    monkeypatch.setenv("AI_LEADS_DB_POOL_MAX", "2")
    monkeypatch.setenv("AI_LEADS_DB_POOL_MIN", "1")
    assert _pool_sizes() == (1, 2)


def test_garbage_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_DB_POOL_MAX", "nope")
    monkeypatch.setenv("QUEUE_WORKFLOWS_DB_POOL_MIN", "")
    assert _pool_sizes() == (1, 10)
