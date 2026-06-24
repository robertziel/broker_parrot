-- Revert 0015. Drop the index + capacity/flag columns, and restore the 0011
-- event_type CHECK (without 'unassignable').

DROP INDEX IF EXISTS workflow_node_jobs_unassignable_idx;

ALTER TABLE workflow_node_jobs DROP COLUMN unassignable_reason;
ALTER TABLE workflow_node_jobs DROP COLUMN unassignable_at;

ALTER TABLE worker_heartbeats DROP COLUMN fits_models;
ALTER TABLE worker_heartbeats DROP COLUMN vram_total_mb;

-- NOTE: Postgres migration also drops and recreates the workflow_node_events
-- event_type CHECK constraint to remove 'unassignable'. SQLite does not support
-- modifying constraints after table creation; the constraint must be defined in
-- the base schema with the appropriate set of allowed event types.
