-- 0020 (sqlite) down — drop the box-placement columns.
ALTER TABLE workflow_node_jobs DROP COLUMN force_box;
ALTER TABLE workflow_node_jobs DROP COLUMN avoid_box;
