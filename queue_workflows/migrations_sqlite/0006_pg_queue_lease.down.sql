-- Revert 0006: drop the lease index and columns
DROP INDEX IF EXISTS workflow_node_jobs_lease_idx;

ALTER TABLE workflow_node_jobs DROP COLUMN lease_expires_at;
ALTER TABLE workflow_node_jobs DROP COLUMN claimed_by;
