-- Revert 0023.
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS resolved_server;
