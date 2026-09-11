-- queue_workflows 0025 down — restore the cpu/gpu queue CHECK.
--
-- NB (the 0008.down caveat): re-adding the CHECK assumes only cpu/gpu rows
-- remain in workflow_node_jobs. If `download` rows exist, this fails — the
-- operator drains/cancels those rows first (they are engine-recoverable:
-- queued rows cancel, running rows finish or lease-lapse).

ALTER TABLE workflow_node_jobs
    ADD CONSTRAINT workflow_node_jobs_queue_check
    CHECK (queue IN ('cpu', 'gpu'));

DROP INDEX IF EXISTS workflow_node_jobs_dedup_idx;

ALTER TABLE workflow_node_jobs
    DROP COLUMN IF EXISTS dedup_key;
