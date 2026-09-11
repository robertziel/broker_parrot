-- queue_workflows 0025 — the `download` node-job queue: the queue-name
-- allow-list moves HOST-SIDE (the 0008 precedent, applied to
-- workflow_node_jobs).
--
-- WHY: downloads are WAIT (network I/O), not compute. Routed onto cpu they
-- burn a claim slot on wall-clock; run N-wide they saturate the shared uplink.
-- `download` is a third first-class NODE queue (full DAG citizenship —
-- depends_on edges, force_box placement, lease/watchdog/outbox) whose claim
-- enforces a fleet-wide per-project width cap (default 1) in
-- node_queue.claim_next_download_job. No new column is needed — only the
-- hardcoded cpu/gpu CHECK has to go.
--
-- Like 0008, the DB stops owning the allow-list entirely (rather than growing
-- the CHECK by one name): enqueue_node_job already validates queue host-side,
-- and any FUTURE lane needs no migration. The CHECK is matched by DEFINITION,
-- not name — the live table may carry the auto-named inline constraint
-- (`workflow_node_jobs_queue_check`) or an older host chain's name. The
-- discriminator is the quoted literal 'cpu': `workflow_node_jobs_check`
-- (queue = 'gpu' OR required_model IS NULL) mentions only 'gpu', and no other
-- CHECK on this table mentions 'cpu', so nothing else can match.
--
-- Idempotent: a re-run finds no matching constraint and drops nothing.

DO $$
DECLARE c record;
BEGIN
    FOR c IN
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'workflow_node_jobs'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%''cpu''%'
    LOOP
        EXECUTE format(
            'ALTER TABLE workflow_node_jobs DROP CONSTRAINT %I', c.conname);
    END LOOP;
END $$;

-- DEDUP KEY — "one file is never downloaded twice". An opt-in, host-provided
-- OUTPUT-IDENTITY string: when a node-job with the same (project, dedup_key)
-- has already COMPLETED, execute_node finalizes the new job as completed
-- (the twin's context_delta + a `_deduped_from` marker) WITHOUT invoking the
-- body. NULL (the default, and every existing row) = no dedup — byte-compat.
-- The engine only compares strings: a host whose outputs are box-local must
-- encode the location in the key (or omit it) — see docs. The partial index
-- serves exactly the twin lookup (completed rows with a key).
ALTER TABLE workflow_node_jobs
    ADD COLUMN IF NOT EXISTS dedup_key TEXT;

CREATE INDEX IF NOT EXISTS workflow_node_jobs_dedup_idx
    ON workflow_node_jobs (project, dedup_key)
    WHERE dedup_key IS NOT NULL AND status = 'completed';
