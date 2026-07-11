-- queue_workflows 0015 — capacity-aware GPU model assignment + unassignable flag.
-- SQLite port of the Postgres migration.

ALTER TABLE worker_heartbeats
    ADD COLUMN vram_total_mb INTEGER;

ALTER TABLE worker_heartbeats
    ADD COLUMN fits_models TEXT NOT NULL DEFAULT '[]';

ALTER TABLE workflow_node_jobs
    ADD COLUMN unassignable_at TEXT;

ALTER TABLE workflow_node_jobs
    ADD COLUMN unassignable_reason TEXT;

-- NOTE: Postgres migration also drops and recreates the workflow_node_events
-- event_type CHECK constraint to include 'unassignable'. SQLite does not support
-- modifying constraints after table creation; the constraint must be defined in
-- the base schema with the full set of allowed event types.

CREATE INDEX IF NOT EXISTS workflow_node_jobs_unassignable_idx
    ON workflow_node_jobs (queue, status)
    WHERE required_model IS NOT NULL;
