-- Reverse 0030. Dropping the column loses every job's declared GPU weight, which is
-- exactly what a downgrade means here: without it the arbiter cannot weigh a row and
-- every row is worth 1 slot again — the pre-0030 behaviour.
ALTER TABLE workflow_node_jobs
    DROP COLUMN IF EXISTS gpu_slots;
