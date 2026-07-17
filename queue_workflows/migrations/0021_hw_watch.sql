-- queue_workflows 0021 — hw_watch_samples (two-tier hardware flight recorder).
--
-- WHY. hw_metrics is push-only (NOTIFY, no row retained), so when a GPU box
-- hard-dies there is NO persisted hardware trail to autopsy. The motivating
-- incident: a GB10 box's firmware thermal protection killed the machine with
-- zero kernel trace — the GPU asserted a HW slowdown (throttle mask 0x48)
-- seconds before an OS-invisible power cut, and only an ad-hoc side-channel
-- recorder caught it. This table is that flight recorder, engine-owned.
--
-- DESIGN. Append-only rows in two tiers, pruned on a NodePool sweep:
--   * 'detail'  — super-detailed samples (default every 2 s), retained 1 h.
--   * 'history' — coarse samples (default every 60 s), retained 24 h.
-- ``tier`` is free-form TEXT, not a CHECK, so future tiers slot in with no
-- schema change (the worker_controls.stop_policy precedent). ``data`` carries
-- the full sample (GPU temp/power/clocks/throttle-mask, thermal zones, hwmon,
-- meminfo, loadavg, disk) as JSONB — the payload shape may grow per vendor.
-- ``box`` is the PHYSICAL box identity (gpu_model_lease.default_box_id), the
-- same name the box lease / avoid_box use — NOT host_label — so cross-project
-- samples of one machine group together. Writes are best-effort telemetry
-- (own connection, swallow-on-failure, like workflow_node_events): a sample
-- blip can never take down a worker. No UPDATE path — append-only.
--
-- Additive + idempotent (IF NOT EXISTS) so re-running on a live DB is a no-op.

CREATE TABLE IF NOT EXISTS hw_watch_samples (
    id          BIGSERIAL PRIMARY KEY,
    host_label  TEXT NOT NULL,             -- emitting worker's label
    box         TEXT,                      -- physical box id (NULL = unknown)
    project     TEXT NOT NULL DEFAULT '',  -- tenant tag (0017 convention)
    tier        TEXT NOT NULL,             -- 'detail' | 'history' (free-form)
    data        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Retention-sweep predicate (prune deletes per tier by age).
CREATE INDEX IF NOT EXISTS hw_watch_samples_tier_created_idx
    ON hw_watch_samples (tier, created_at);

-- The hot read: one box's recent trail, newest first.
CREATE INDEX IF NOT EXISTS hw_watch_samples_host_created_idx
    ON hw_watch_samples (host_label, created_at);
