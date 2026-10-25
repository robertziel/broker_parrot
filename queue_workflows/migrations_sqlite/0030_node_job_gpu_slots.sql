-- 0030 (sqlite twin) — per-node-job GPU weight: how much of the box's budget one job occupies.
-- See the pg file for the full WHY. Same plain ADD COLUMN shape as 0024's and 0028's twins
-- (SQLite has no ``ADD COLUMN IF NOT EXISTS``; the chain runner applies each file once).
ALTER TABLE workflow_node_jobs ADD COLUMN gpu_slots TEXT;
