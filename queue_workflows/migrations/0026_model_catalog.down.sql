DROP INDEX IF EXISTS llm_model_catalog_by_model;
DROP INDEX IF EXISTS llm_model_catalog_lookup;
DROP TABLE IF EXISTS llm_model_catalog;
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS required_models;
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS required_model_ids;
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS resolved_model;
