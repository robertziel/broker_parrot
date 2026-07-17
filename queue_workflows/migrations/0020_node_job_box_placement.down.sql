-- 0020 down — drop the per-node-job box-placement columns.
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS force_box;
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS avoid_box;
