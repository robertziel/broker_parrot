-- SQLite variant of 0018 (see the pg copy for rationale): tag the node-event log with the
-- tenant project so error_snapshot / an operator Errors view is project-scoped. SQLite has
-- no ``ADD COLUMN IF NOT EXISTS``; the migration runner applies each version once, so a plain
-- ADD COLUMN is correct.
ALTER TABLE workflow_node_events
    ADD COLUMN project TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS workflow_node_events_project_kind_idx
    ON workflow_node_events (project, event_type, created_at DESC);
