ALTER TABLE worker_controls
    DROP COLUMN vllm_idle_ttl_s;

ALTER TABLE worker_controls
    DROP COLUMN llm_parallelism;

ALTER TABLE worker_controls
    DROP COLUMN llm_server_type;
