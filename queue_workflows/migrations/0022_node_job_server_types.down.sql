-- Revert 0022 — drop the per-node-job LLM server preference.
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS server_types;
