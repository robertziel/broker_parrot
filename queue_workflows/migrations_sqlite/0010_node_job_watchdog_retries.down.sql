-- Reverse of 0010. Drop the per-job watchdog re-queue counter.
ALTER TABLE workflow_node_jobs DROP COLUMN watchdog_retries;
