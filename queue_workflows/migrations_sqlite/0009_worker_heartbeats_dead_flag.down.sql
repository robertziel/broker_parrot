DROP INDEX IF EXISTS worker_heartbeats_flagged_dead_idx;
ALTER TABLE worker_heartbeats DROP COLUMN last_flagged_dead_at;
