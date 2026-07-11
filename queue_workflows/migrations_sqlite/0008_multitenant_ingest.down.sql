-- Reverse 0008: drop per-job args column.
-- In Postgres: re-add the CHECK constraints. On SQLite: no-ops (never existed).

ALTER TABLE ingest_jobs
    DROP COLUMN args;
