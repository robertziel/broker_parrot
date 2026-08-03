-- Reverse 0024. Dropping the column loses every job's declared CPU width, which
-- is exactly what a downgrade means here: without it the claim cannot weigh a
-- job and every row is worth 1 slot again — the pre-0024 behaviour.
ALTER TABLE workflow_node_jobs
    DROP COLUMN IF EXISTS cpu_slots;
