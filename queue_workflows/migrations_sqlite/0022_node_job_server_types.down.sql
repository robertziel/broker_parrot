-- 0022 (sqlite) down — drop the per-node-job LLM server preference.
ALTER TABLE workflow_node_jobs DROP COLUMN server_types;
