-- 0027 (sqlite twin) — digest on the catalog, artifact on the row. See the pg file for
-- the full WHY. SQLite has no IF NOT EXISTS on ADD COLUMN in older versions the tests
-- pin, so these follow the same shape 0026's twin uses.
ALTER TABLE llm_model_catalog ADD COLUMN digest TEXT NOT NULL DEFAULT '';
ALTER TABLE workflow_node_jobs ADD COLUMN resolved_artifact TEXT;
