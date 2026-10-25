ALTER TABLE worker_heartbeats DROP COLUMN llm_context_max;
ALTER TABLE worker_heartbeats DROP COLUMN llm_context_length;
ALTER TABLE workflow_node_jobs DROP COLUMN context_length;
