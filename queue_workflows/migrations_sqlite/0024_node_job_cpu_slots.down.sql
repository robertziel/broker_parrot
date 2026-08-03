-- Reverse 0024 (sqlite twin). Without the column every row is worth 1 slot
-- again — the pre-0024 behaviour.
ALTER TABLE workflow_node_jobs DROP COLUMN cpu_slots;
