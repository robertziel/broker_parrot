-- 0025 down (sqlite twin) — restore the cpu/gpu queue CHECK via the reverse
-- rebuild. Assumes only cpu/gpu rows remain (the pg down's caveat): a
-- `download` row would violate the re-added CHECK on the INSERT-SELECT and
-- roll the whole transaction back, leaving the 0025 table intact.

BEGIN IMMEDIATE;

CREATE TABLE workflow_node_jobs_old (
    id                 TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    node_id            TEXT NOT NULL,
    node_module        TEXT NOT NULL,
    pipeline_name      TEXT,
    queue              TEXT NOT NULL CHECK (queue IN ('cpu', 'gpu')),
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
    UNIQUE (run_id, node_id),
    CONSTRAINT workflow_node_jobs_status_check CHECK (status IN (
        'queued', 'running', 'completed', 'failed', 'cancelled',
        'awaiting_input', 'skipped'
    )),
    CONSTRAINT workflow_node_jobs_check
        CHECK (queue = 'gpu' OR required_model IS NULL)
);

INSERT INTO workflow_node_jobs_old (
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

ALTER TABLE workflow_node_jobs_old RENAME TO workflow_node_jobs;

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

COMMIT;
