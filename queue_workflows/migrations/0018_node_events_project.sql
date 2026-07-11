-- Tag the forensic node-event log with the tenant project, so the queue ERROR LOG
-- (node_queue.error_snapshot / an operator Errors view) is project-scoped exactly like
-- the rest of the broker. Migration 0017 tagged runs / node_jobs / ingest_jobs /
-- worker_heartbeats but missed workflow_node_events; this closes the gap so every project's
-- queue errors are captured AND filterable without a join.
--
-- ``NOT NULL DEFAULT ''`` backfills existing rows to the single-tenant sentinel with no
-- separate backfill step; re-running on an already-migrated DB is a safe no-op.
ALTER TABLE workflow_node_events
    ADD COLUMN IF NOT EXISTS project text NOT NULL DEFAULT '';

-- The error console reads (one project, the failure kinds, newest-first) → covering index.
CREATE INDEX IF NOT EXISTS workflow_node_events_project_kind_idx
    ON workflow_node_events (project, event_type, created_at DESC);
