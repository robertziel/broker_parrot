ALTER TABLE worker_heartbeats DROP COLUMN IF EXISTS llm_context_max;
ALTER TABLE worker_heartbeats DROP COLUMN IF EXISTS llm_context_length;
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS context_length;
