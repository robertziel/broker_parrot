-- hw_watch_samples: two-tier hardware flight recorder (see the pg twin for the
-- full WHY). 'detail' = super-detailed ring (default 2 s cadence, 1 h retention);
-- 'history' = coarse ring (60 s cadence, 24 h retention). Append-only; pruned by
-- a NodePool sweep. 'data' is the full JSON sample (GPU temp/power/throttle-mask,
-- thermal zones, hwmon, meminfo, load, disk); 'box' is the physical box identity
-- (the box-lease name), not host_label.

CREATE TABLE IF NOT EXISTS hw_watch_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    host_label  TEXT NOT NULL,
    box         TEXT,
    project     TEXT NOT NULL DEFAULT '',
    tier        TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now'))
);

CREATE INDEX IF NOT EXISTS hw_watch_samples_tier_created_idx
    ON hw_watch_samples (tier, created_at);

CREATE INDEX IF NOT EXISTS hw_watch_samples_host_created_idx
    ON hw_watch_samples (host_label, created_at);
