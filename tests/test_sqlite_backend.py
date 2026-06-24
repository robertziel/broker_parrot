"""SQLite engine backend — Phase 1: the connection/dialect seam.

Proves the SQLite compatibility layer in ``db.py`` gives psycopg parity for the
constructs the engine relies on: pyformat (``%s`` / ``%(name)s``) paramstyle,
JSON-obj→dict, JSON-array→list, TIMESTAMPTZ→aware ``datetime``, ``now()``
translation, the string-literal-aware translator (``strftime('%s')`` survives
while ``%s`` placeholders convert), ``::cast`` strip, ``LEAST``→``MIN``,
``FOR UPDATE [SKIP LOCKED]`` strip, ``RETURNING *``, and ``rowcount``.

These run against a throwaway SQLite FILE (per test, via tmp_path), independent
of the Postgres test DB the rest of the suite uses.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

import queue_workflows
from queue_workflows import db, dialect


@pytest.fixture
def sqlite_engine(tmp_path):
    """Point the engine's relational store at a fresh SQLite file, then restore
    the Postgres test config on teardown (this fixture tears down before the
    autouse pg-truncate fixture, so it must hand the engine back to Postgres)."""
    path = str(tmp_path / "qw.db")
    os.environ["QUEUE_WORKFLOWS_SQLITE_SMOKE_URL"] = path
    db.close_pool()
    queue_workflows.configure(
        db_backend="sqlite", db_url_env="QUEUE_WORKFLOWS_SQLITE_SMOKE_URL",
    )
    yield path
    db.close_pool()
    queue_workflows.configure(db_backend="pg", db_url_env="QUEUE_WORKFLOWS_TEST_DB_URL")


def test_dialect_selected_for_sqlite(sqlite_engine):
    assert dialect.is_sqlite() is True
    assert dialect.get_dialect().name == "sqlite"
    assert db.sqlite_path() == sqlite_engine


def _create(cur):
    # Real engine column NAMES so the row factory's by-name converters fire.
    cur.execute(
        """
        CREATE TABLE t (
            id           TEXT PRIMARY KEY,
            context      TEXT,
            known_models TEXT,
            priority     INTEGER NOT NULL DEFAULT 100,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )


def test_paramstyle_json_array_timestamp_roundtrip(sqlite_engine):
    from psycopg.types.json import Jsonb

    with db.connection() as conn, conn.cursor() as cur:
        _create(cur)
    # positional %s + Jsonb (adapted to JSON text) + a JSON-array string + now()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO t (id, context, known_models, created_at) "
            "VALUES (%s, %s, %s, now())",
            ("a", Jsonb({"k": 1, "nested": [1, 2]}), json.dumps(["sdxl", "qwen"])),
        )
        assert cur.rowcount == 1
    # named %(id)s + RETURNING-style read
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM t WHERE id = %(id)s", {"id": "a"})
        row = cur.fetchone()

    assert row["id"] == "a"
    assert row["context"] == {"k": 1, "nested": [1, 2]}     # JSON-obj → dict
    assert row["known_models"] == ["sdxl", "qwen"]          # JSON-array → list
    assert isinstance(row["created_at"], datetime)          # TIMESTAMPTZ → datetime
    assert row["created_at"].tzinfo is not None             # aware (UTC), psycopg parity


def test_translator_is_string_literal_aware(sqlite_engine):
    # strftime('%s', …) must SURVIVE (literal), while the %s placeholder converts.
    with db.connection() as conn, conn.cursor() as cur:
        _create(cur)
        cur.execute("INSERT INTO t (id) VALUES (%s)", ("a",))
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT CAST(strftime('%s', created_at) AS REAL) AS epoch, "
            "%s AS passthru FROM t WHERE id = %s",
            ("hello", "a"),
        )
        row = cur.fetchone()
    assert isinstance(row["epoch"], float) and row["epoch"] > 0
    assert row["passthru"] == "hello"


def test_mechanical_rewrites(sqlite_engine):
    # LEAST→MIN, ::cast strip, FOR UPDATE SKIP LOCKED strip in one statement.
    with db.connection() as conn, conn.cursor() as cur:
        _create(cur)
        cur.execute("INSERT INTO t (id, priority) VALUES (%s, %s)", ("a", 50))
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE t SET priority = LEAST(priority, 10) "
            "WHERE id = (SELECT id FROM t WHERE priority = 50 "
            "            ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) "
            "RETURNING priority, '{}'::jsonb AS j",
            (),
        )
        row = cur.fetchone()
    assert row["priority"] == 10        # LEAST→MIN applied
    assert row["j"] == "{}"             # ::jsonb stripped; 'j' not a known JSON col → text


def test_dialect_fragments_render_for_sqlite(sqlite_engine):
    d = dialect.get_dialect()
    assert d.skip_locked == ""
    assert "datetime('now'" in d.future_seconds("%(s)s")
    assert "strftime" in d.epoch("created_at")
    assert "json_each" in d.value_in_param_array("x", "%(p)s")
    assert d.array_param(["a", "b"]) == '["a", "b"]'   # list → JSON text for sqlite


def test_migration_chain_bootstraps_and_roundtrips(sqlite_engine):
    # The SQLite migration chain (migrations_sqlite/) applies to v17, creates the
    # full engine schema, and survives a full downgrade→0→re-bootstrap roundtrip.
    db.bootstrap()
    assert db.current_schema_version() == 17

    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = {r["name"] for r in cur.fetchall()}
        cur.execute("PRAGMA table_info(worker_heartbeats)")
        hb_pk = [r["name"] for r in cur.fetchall() if r["pk"]]

    assert {
        "workflow_runs", "workflow_node_jobs", "ingest_jobs", "worker_heartbeats",
        "worker_controls", "workflow_dispatch_events", "workflow_node_events",
        "workflow_input_submissions", "workflow_run_files",
    } <= tables
    assert hb_pk == ["host_label", "queue", "project"]   # migration 0017 PK rebuild

    reverted = db.downgrade(to_version=0)
    assert len(reverted) == 17
    assert db.current_schema_version() == 0
    db.bootstrap()
    assert db.current_schema_version() == 17


def test_commit_and_rollback_semantics(sqlite_engine):
    with db.connection() as conn, conn.cursor() as cur:
        _create(cur)
    # rollback on exception: the row must NOT persist
    with pytest.raises(RuntimeError):
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO t (id) VALUES (%s)", ("rollme",))
            raise RuntimeError("boom")
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM t WHERE id = %s", ("rollme",))
        assert cur.fetchone()["n"] == 0
