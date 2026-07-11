-- queue_workflows 0009 — worker_heartbeats.last_flagged_dead_at (stale-worker recovery marker).
-- Add a column to track when a worker was flagged as dead by the orchestrator.

ALTER TABLE worker_heartbeats ADD COLUMN last_flagged_dead_at TEXT;

-- Supports the supervisor's "which workers are flagged dead recently" poll.
CREATE INDEX IF NOT EXISTS worker_heartbeats_flagged_dead_idx
    ON worker_heartbeats (last_flagged_dead_at)
    WHERE last_flagged_dead_at IS NOT NULL;
