-- 0022 (sqlite) — per-node-job LLM SERVER PREFERENCE.
-- SQLite twin of migrations/0022: text[] → TEXT holding a JSON array (the dialect
-- writes via array_literal → json.dumps and matches via json_each / arrays_overlap),
-- NULL = unconstrained. Plain ADD COLUMN (SQLite has no ADD COLUMN IF NOT EXISTS;
-- the version ledger already guarantees single application).
ALTER TABLE workflow_node_jobs ADD COLUMN server_types TEXT;
