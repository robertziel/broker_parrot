DROP INDEX IF EXISTS llm_model_catalog_by_model;
DROP INDEX IF EXISTS llm_model_catalog_lookup;
DROP TABLE IF EXISTS llm_model_catalog;
-- SQLite ≥3.35 supports DROP COLUMN; older builds need a rebuild, which a down-migration
-- of an additive change does not justify.
ALTER TABLE workflow_node_jobs DROP COLUMN required_models;
ALTER TABLE workflow_node_jobs DROP COLUMN required_model_ids;
ALTER TABLE workflow_node_jobs DROP COLUMN resolved_model;
