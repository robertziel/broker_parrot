-- 0026 (sqlite twin) — the model catalog + the acceptable-models array.
-- See the pg migration for the WHY.
--
-- No table rebuild needed here: SQLite supports ADD COLUMN, and both new columns are
-- nullable with no CHECK. JSONB has no SQLite type — TEXT holds the same JSON and the
-- dialect layer already serialises it, exactly as it does for other JSON columns.
CREATE TABLE IF NOT EXISTS llm_model_catalog (
    host_label   TEXT    NOT NULL,
    server_type  TEXT    NOT NULL,
    model_id     TEXT    NOT NULL,
    enabled      INTEGER NOT NULL DEFAULT 1,
    observed_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (host_label, server_type, model_id)
);

CREATE INDEX IF NOT EXISTS llm_model_catalog_lookup
    ON llm_model_catalog (host_label, server_type) WHERE enabled = 1;
CREATE INDEX IF NOT EXISTS llm_model_catalog_by_model
    ON llm_model_catalog (model_id) WHERE enabled = 1;

ALTER TABLE workflow_node_jobs ADD COLUMN required_models TEXT;
-- Flat claim index; see the pg migration. TEXT holds the JSON array the
-- dialect's array_param already serialises for sqlite.
ALTER TABLE workflow_node_jobs ADD COLUMN required_model_ids TEXT;
ALTER TABLE workflow_node_jobs ADD COLUMN resolved_model  TEXT;
