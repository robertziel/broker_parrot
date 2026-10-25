-- 0029 — what a model COSTS on a box, as its own server reported it.
--
-- WHY. vLLM takes its memory pool at start, as a fraction of the WHOLE machine on a
-- unified-memory box, and the engine sized that pool to "what is free" (never above the
-- configured value) — which on an empty box is most of its memory for a model whose weights
-- are a fraction of that.
-- What the model NEEDS was never known to anyone: it is printed once, in vLLM's own
-- startup log, and forgotten.
--
--   * llm_model_catalog.profile — JSON, per (host, server_type, model): the latest complete
--     start's numbers — weights_bytes, kv_bytes, kv_tokens, window, utilization,
--     total_bytes — plus the fingerprint of the command line they belong to (everything
--     but the three flags the engine itself changes: window, pool, concurrency). From it
--     the next start is sized to what N requests of the window need, and N to what fits.
--     NULL = never observed here; the catalog's observation upsert never touches it.
--
-- Nullable, additive, byte-compatible for any reader or writer that has never heard of it.

ALTER TABLE llm_model_catalog
    ADD COLUMN IF NOT EXISTS profile JSONB;
