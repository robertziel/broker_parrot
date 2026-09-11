ALTER TABLE llm_model_catalog DROP COLUMN IF EXISTS digest;
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS resolved_artifact;
