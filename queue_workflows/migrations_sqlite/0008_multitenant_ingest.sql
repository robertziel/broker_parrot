-- queue_workflows 0008 — multi-tenant ingest: per-job args + host-defined queue
-- names. SQLite: omit CHECK constraint drops (never existed in sqlite base).

-- (G2) Per-job arguments for parametrised ingest tasks — e.g. a host's
-- run_scenario(scenario_id). DEFAULT '{}' so every existing INSERT stays valid
-- and ai_leads' periodic sweeps (which carry no args) are unaffected.
ALTER TABLE ingest_jobs
    ADD COLUMN args TEXT NOT NULL DEFAULT '{}';

-- (G1) In Postgres: drop ingest_jobs_queue_check. On SQLite: no-op (never existed).
-- (G5) In Postgres: drop worker_heartbeats_queue_check. On SQLite: no-op (never existed).
