-- queue_workflows 0019 (sqlite) — project (tenant) identity on the control plane.
--
-- Mirror of the Postgres 0019. See that file for the WHY (two projects sharing a
-- (host_label, queue) share ONE control row, so an ON/OFF or an LLM-config write
-- for one tenant hits the other).
--
-- SQLite has no ``ALTER TABLE … DROP CONSTRAINT``, so re-keying the PK means the
-- standard table-rebuild: create the new shape, copy, drop, rename. No triggers
-- exist on this chain (the NOTIFY payload change is Postgres-only).

ALTER TABLE worker_controls
    ADD COLUMN project TEXT NOT NULL DEFAULT '';

CREATE TABLE worker_controls_new (
    host_label      TEXT NOT NULL,
    queue           TEXT NOT NULL,
    project         TEXT NOT NULL DEFAULT '',
    desired_state   TEXT NOT NULL DEFAULT 'on',
    stop_policy     TEXT NOT NULL DEFAULT 'hard',
    requested_by    TEXT,
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    llm_server_type TEXT NOT NULL DEFAULT 'ollama'
        CHECK (llm_server_type IN ('ollama', 'vllm')),
    llm_parallelism INTEGER NOT NULL DEFAULT 1
        CHECK (llm_parallelism >= 1),
    vllm_idle_ttl_s INTEGER NOT NULL DEFAULT 60
        CHECK (vllm_idle_ttl_s >= 0),
    PRIMARY KEY (host_label, queue, project),
    CHECK (desired_state IN ('on', 'off'))
);

INSERT INTO worker_controls_new (
    host_label, queue, project, desired_state, stop_policy, requested_by,
    updated_at, llm_server_type, llm_parallelism, vllm_idle_ttl_s
)
SELECT host_label, queue, project, desired_state, stop_policy, requested_by,
       updated_at, llm_server_type, llm_parallelism, vllm_idle_ttl_s
FROM worker_controls;

DROP TABLE worker_controls;
ALTER TABLE worker_controls_new RENAME TO worker_controls;
