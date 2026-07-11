-- queue_workflows 0006 — SQLite lease columns and indexing
-- (pg_notify wake primitives omitted; SQLite uses polling instead)

ALTER TABLE workflow_node_jobs ADD COLUMN claimed_by TEXT;
ALTER TABLE workflow_node_jobs ADD COLUMN lease_expires_at TEXT;

CREATE INDEX IF NOT EXISTS workflow_node_jobs_lease_idx
    ON workflow_node_jobs (lease_expires_at)
    WHERE status = 'queued';
