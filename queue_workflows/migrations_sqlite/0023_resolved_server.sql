-- 0023 (sqlite) — resolved server kind stamped at claim. See migrations/0023.
ALTER TABLE workflow_node_jobs ADD COLUMN resolved_server TEXT;
