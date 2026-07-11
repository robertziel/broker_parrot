-- queue_workflows 0013 — per-machine LLM server config on worker_controls.
-- (notify function and trigger omitted for SQLite; uses polling wakeup instead
--  of LISTEN/NOTIFY. SQLite adds ONE column per ALTER TABLE — so the pg
--  migration's single multi-column ALTER becomes three statements.)

ALTER TABLE worker_controls
    ADD COLUMN llm_server_type TEXT NOT NULL DEFAULT 'ollama'
        CHECK (llm_server_type IN ('ollama', 'vllm'));

ALTER TABLE worker_controls
    ADD COLUMN llm_parallelism INTEGER NOT NULL DEFAULT 1
        CHECK (llm_parallelism >= 1);

ALTER TABLE worker_controls
    ADD COLUMN vllm_idle_ttl_s INTEGER NOT NULL DEFAULT 60
        CHECK (vllm_idle_ttl_s >= 0);
