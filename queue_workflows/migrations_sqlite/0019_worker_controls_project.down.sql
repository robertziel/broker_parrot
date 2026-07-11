-- Revert 0019 (sqlite) — drop the tenant tag from the operator control plane.
--
-- DESTRUCTIVE, same rationale as the Postgres down-migration: the 2-col PK can't
-- be restored while two projects hold a row for the same (host_label, queue), so
-- the non-default tenants' control rows are dropped. Control rows are cheap
-- operator state, not queue work; a worker with no row defaults to ON.

DELETE FROM worker_controls WHERE project <> '';

CREATE TABLE worker_controls_old (
    host_label      TEXT NOT NULL,
    queue           TEXT NOT NULL,
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
    PRIMARY KEY (host_label, queue),
    CHECK (desired_state IN ('on', 'off'))
);

INSERT INTO worker_controls_old (
    host_label, queue, desired_state, stop_policy, requested_by,
    updated_at, llm_server_type, llm_parallelism, vllm_idle_ttl_s
)
SELECT host_label, queue, desired_state, stop_policy, requested_by,
       updated_at, llm_server_type, llm_parallelism, vllm_idle_ttl_s
FROM worker_controls;

DROP TABLE worker_controls;
ALTER TABLE worker_controls_old RENAME TO worker_controls;
