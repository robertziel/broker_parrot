-- 0024 (sqlite twin) — per-node-job CPU WIDTH. See the pg migration for the why.
--
-- Bare ``ADD COLUMN``: SQLite supports neither ``IF NOT EXISTS`` on ADD COLUMN
-- nor ``ALTER TABLE ... ADD CONSTRAINT``, which is why the ">= 1" invariant is
-- enforced host-side in ``node_queue.enqueue_node_job`` instead of as a CHECK.
-- NULL = 1 = an ordinary job.
ALTER TABLE workflow_node_jobs ADD COLUMN cpu_slots INTEGER;
