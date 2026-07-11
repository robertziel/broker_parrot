DROP INDEX IF EXISTS workflow_node_events_project_kind_idx;
ALTER TABLE workflow_node_events DROP COLUMN IF EXISTS project;
