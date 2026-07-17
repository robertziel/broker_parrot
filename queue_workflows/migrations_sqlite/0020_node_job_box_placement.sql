-- 0020 (sqlite) — per-node-job PHYSICAL-BOX placement constraints.
-- SQLite twin of migrations/0020: text[] → TEXT holding a JSON array (the dialect
-- writes via array_literal → json.dumps and matches via json_each), NULL =
-- unconstrained. Plain ADD COLUMN (SQLite has no ADD COLUMN IF NOT EXISTS; the
-- version ledger already guarantees single application).
ALTER TABLE workflow_node_jobs ADD COLUMN avoid_box TEXT;
ALTER TABLE workflow_node_jobs ADD COLUMN force_box TEXT;
