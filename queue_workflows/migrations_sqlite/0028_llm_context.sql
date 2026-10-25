-- 0028 (sqlite twin) — the context window a node-job needs, and the one a box serves.
-- See the pg file for the full WHY. Same plain ADD COLUMN shape as 0024's twin.
ALTER TABLE workflow_node_jobs ADD COLUMN context_length INTEGER;
ALTER TABLE worker_heartbeats ADD COLUMN llm_context_length INTEGER;
ALTER TABLE worker_heartbeats ADD COLUMN llm_context_max INTEGER;
