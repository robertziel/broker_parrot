-- 0028 — the CONTEXT WINDOW a node-job needs, and the one a GPU box serves.
--
-- WHY. An LLM job's prompt has to fit the server's context window, and when it does not
-- the failure is SILENT: ollama truncates the prompt to fit and answers HTTP 200.
-- Measured on ollama 0.33.2 — a ~12k-token prompt to a runner holding an 8192 context
-- came back 200 with usage.prompt_tokens = 4098. Nothing in the queue knew what size a
-- job needed or what size a box served, so nothing could stop that before it happened.
--
--   * workflow_node_jobs.context_length — the window (tokens) this job's requests need,
--     prompt + completion. NULL = no requirement, which is every row written before
--     this migration, so they claim exactly as they always did. Declared per node in
--     the pipeline schema (a literal, or a $ref the run supplies); validated host-side.
--
--   * worker_heartbeats.llm_context_length — the window this box's LLM server serves
--     RIGHT NOW, OBSERVED (ollama /api/ps for the loaded runner, vLLM max_model_len).
--     The claim refuses a row whose context_length exceeds it.
--
--   * worker_heartbeats.llm_context_max — the highest window this box could SWITCH to:
--     the model's own limit, bounded by the memory the box has free for the larger KV
--     cache, or simply llm_context_length when the box has no working lever to restart
--     its server with a bigger one. The orchestrator tags a row no live box's
--     llm_context_max reaches; a box whose max reaches a starving row switches up.
--
-- All three nullable and additive — byte-compatible for any reader or writer that has
-- never heard of them.

ALTER TABLE workflow_node_jobs
    ADD COLUMN IF NOT EXISTS context_length INTEGER;

ALTER TABLE worker_heartbeats
    ADD COLUMN IF NOT EXISTS llm_context_length INTEGER;

ALTER TABLE worker_heartbeats
    ADD COLUMN IF NOT EXISTS llm_context_max INTEGER;
