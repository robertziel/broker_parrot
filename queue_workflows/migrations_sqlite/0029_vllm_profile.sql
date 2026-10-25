-- 0029 (sqlite twin) — a model's memory profile on a box. See the pg file for the WHY.
-- JSONB has no SQLite type — TEXT holds the same JSON, as 0026's required_models does.
ALTER TABLE llm_model_catalog ADD COLUMN profile TEXT;
