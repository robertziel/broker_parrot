-- 0025 (sqlite twin) — the `download` node-job queue. See the pg migration for
-- the why.
--
-- SQLite cannot drop a CHECK (no ALTER TABLE DROP CONSTRAINT), so this is the
-- one-time 12-step table REBUILD (the 0017 worker_heartbeats precedent):
-- recreate workflow_node_jobs in its exact head shape MINUS the cpu/gpu queue
-- CHECK, copy every row, swap, and recreate every index. After this, queue
-- names are host-validated only (enqueue_node_job) — any FUTURE lane needs no
-- rebuild.
--
-- Safe: nothing references workflow_node_jobs by FK (verified across the
-- chain), and the whole swap is wrapped in ONE transaction so a crash mid-way
-- rolls back to the pre-0025 table instead of leaving a half-rebuilt pair.
-- The column list below is the FULL head shape — 0002 base plus every column
-- later migrations added (0006 lease, 0010 watchdog_retries, 0015
-- unassignable_*, 0016 is_priority, 0017 project, 0020 avoid/force_box, 0022
-- server_types, 0023 resolved_server, 0024 cpu_slots) — because a rebuild
-- replaces the table wholesale.

BEGIN IMMEDIATE;

CREATE TABLE workflow_node_jobs_new (
    id                 TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    node_id            TEXT NOT NULL,
    node_module        TEXT NOT NULL,
    pipeline_name      TEXT,
    queue              TEXT NOT NULL,
    required_model     TEXT,
    status             TEXT NOT NULL,
    priority           INTEGER NOT NULL DEFAULT 100,
    worker_lane        INTEGER,
    inputs             TEXT NOT NULL DEFAULT '{}',
    resolved_inputs    TEXT,
    input_spec         TEXT,
    context_delta      TEXT NOT NULL DEFAULT '{}',
    host_label         TEXT,
    celery_task_id     TEXT,
    error              TEXT,
    vm_rss_mb_peak     INTEGER,
    seconds            REAL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    started_at         TEXT,
    finished_at        TEXT,
    claimed_by         TEXT,
    lease_expires_at   TEXT,
    watchdog_retries   INTEGER NOT NULL DEFAULT 0,
    unassignable_at    TEXT,
    unassignable_reason TEXT,
    is_priority        INTEGER NOT NULL DEFAULT 0,
    project            TEXT NOT NULL DEFAULT '',
    avoid_box          TEXT,
    force_box          TEXT,
    server_types       TEXT,
    resolved_server    TEXT,
    cpu_slots          INTEGER,
    -- NEW in 0025 (see the pg migration): opt-in output-identity for the
    -- "never download twice" short-circuit. NULL on every copied row.
    dedup_key          TEXT,
    UNIQUE (run_id, node_id),
    CONSTRAINT workflow_node_jobs_status_check CHECK (status IN (
        'queued', 'running', 'completed', 'failed', 'cancelled',
        'awaiting_input', 'skipped'
    )),
    CONSTRAINT workflow_node_jobs_check
        CHECK (queue = 'gpu' OR required_model IS NULL)
);

INSERT INTO workflow_node_jobs_new (
    id, run_id, node_id, node_module, pipeline_name, queue, required_model,
    status, priority, worker_lane, inputs, resolved_inputs, input_spec,
    context_delta, host_label, celery_task_id, error, vm_rss_mb_peak, seconds,
    created_at, started_at, finished_at, claimed_by, lease_expires_at,
    watchdog_retries, unassignable_at, unassignable_reason, is_priority,
    project, avoid_box, force_box, server_types, resolved_server, cpu_slots
)
SELECT
    id, run_id, node_id, node_module, pipeline_name, queue, required_model,
    status, priority, worker_lane, inputs, resolved_inputs, input_spec,
    context_delta, host_label, celery_task_id, error, vm_rss_mb_peak, seconds,
    created_at, started_at, finished_at, claimed_by, lease_expires_at,
    watchdog_retries, unassignable_at, unassignable_reason, is_priority,
    project, avoid_box, force_box, server_types, resolved_server, cpu_slots
FROM workflow_node_jobs;

DROP TABLE workflow_node_jobs;

ALTER TABLE workflow_node_jobs_new RENAME TO workflow_node_jobs;

-- Recreate every index the chain had built on the old table (0002 base +
-- 0006 lease + 0015 unassignable + 0017 project claim), verbatim.
CREATE INDEX IF NOT EXISTS workflow_node_jobs_claim_idx
    ON workflow_node_jobs (queue, priority, created_at)
    WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS workflow_node_jobs_run_idx
    ON workflow_node_jobs (run_id);
CREATE INDEX IF NOT EXISTS workflow_node_jobs_status_idx
    ON workflow_node_jobs (status);
CREATE INDEX IF NOT EXISTS workflow_node_jobs_model_idx
    ON workflow_node_jobs (required_model)
    WHERE queue = 'gpu' AND status = 'queued';
CREATE INDEX IF NOT EXISTS workflow_node_jobs_pipeline_idx
    ON workflow_node_jobs (pipeline_name);
CREATE INDEX IF NOT EXISTS workflow_node_jobs_host_label_idx
    ON workflow_node_jobs (host_label)
    WHERE host_label IS NOT NULL;
CREATE INDEX IF NOT EXISTS workflow_node_jobs_celery_task_id_idx
    ON workflow_node_jobs (celery_task_id)
    WHERE celery_task_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS workflow_node_jobs_lease_idx
    ON workflow_node_jobs (lease_expires_at)
    WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS workflow_node_jobs_unassignable_idx
    ON workflow_node_jobs (queue, status)
    WHERE required_model IS NOT NULL;
CREATE INDEX IF NOT EXISTS workflow_node_jobs_project_claim_idx
    ON workflow_node_jobs (queue, project, priority, created_at)
    WHERE status = 'queued';

-- NEW in 0025 — serves the completed-twin dedup lookup.
CREATE INDEX IF NOT EXISTS workflow_node_jobs_dedup_idx
    ON workflow_node_jobs (project, dedup_key)
    WHERE dedup_key IS NOT NULL AND status = 'completed';

COMMIT;
