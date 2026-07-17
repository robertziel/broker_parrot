# hw_watch — the persisted hardware flight recorder (migration 0021)

## Why it exists

`hw_metrics` is push-only: every sample is a `NOTIFY` and gone. When a GPU box
hard-dies there is **no hardware trail to autopsy**. The motivating incident
(2026-07-17): a GB10 box's firmware thermal protection killed the machine on
every `sdxl` render — **zero kernel trace** (no Xid, no thermal event,
journal stops mid-write), because the trigger was a board-level sensor the OS
exposes no trip points for. The GPU asserted a hardware slowdown (throttle
mask `0x48`) **7 seconds before the power cut**; only an ad-hoc sync-per-line
shell recorder caught it. `hw_watch` is that recorder, engine-owned.

## What it stores

One table, `hw_watch_samples`, two tiers, both pruned automatically:

| tier      | default cadence | retention | purpose                          |
|-----------|-----------------|-----------|----------------------------------|
| `detail`  | every 2 s       | 1 h       | death-second forensics ring      |
| `history` | every 60 s      | 24 h      | "was it trending hot all day?"   |

Each row: `host_label`, `box` (the **physical** box identity —
`gpu_model_lease.default_box_id()`, the same name the box lease and
`avoid_box`/`force_box` use), `project`, `tier`, `created_at`, and a `data`
JSONB payload from `hw_watch.deep_sample()`:

- **gpu** (nvidia-smi or rocm-smi, vendor picked once): temp, power draw,
  SM clock, utilisation, pstate, and the **throttle-reason mask** — the
  smoking-gun field (`0x20` SW-thermal, `0x40` HW-thermal, `0x08` HW-brake,
  `0x04` SW-power-cap).
- **tz** — every `/sys/class/thermal` zone (`type` + milli-°C). On the GB10
  incident box the killer sensor was an `acpitz` zone reading ~94 °C while
  the GPU core showed a harmless 78–84 °C.
- **hwmon** — every hwmon temp input (NVMe, NIC, SoC…).
- **mem** — `/proc/meminfo` totals (unified-memory boxes live and die by
  host RAM); **load1**; **disk_root_used_mb**.

Every probe is individually guarded: a dead sensor records its empty shape,
never an exception.

## Contracts

- **Off by default.** `QUEUE_WORKFLOWS_HW_WATCH=1` enables persistence —
  existing deploys are byte-compatible. Knobs (canonical spelling; legacy
  `AI_LEADS_*` twins resolve for free):
  `QUEUE_WORKFLOWS_HW_WATCH_DETAIL_INTERVAL_S` (2),
  `…_HISTORY_INTERVAL_S` (60), `…_DETAIL_RETENTION_S` (3600),
  `…_HISTORY_RETENTION_S` (86400), `…_PRUNE_INTERVAL_S` (300, orchestrator).
- **Writes are best-effort telemetry** (own connection, swallow-on-failure —
  the `record_node_event` pattern). A sample blip, a dead DB, or a pre-0021
  schema can never take down a worker.
- **Append-only; no UPDATE path.** Growth is bounded by the two-tier prune
  (`prune_hw_watch`), run by `NodePool` every 5 min; the standalone CLI
  prunes for itself. Both deletes are dialect-portable (pg + sqlite).
- The cadence brain (`HwWatchRecorder`) is pure logic with injectable
  `now_fn`/`record_fn`/`sample_fn` seams — tested on a virtual clock
  (`tests/test_hw_watch.py`).

## How it runs

**Inside a worker (normal path):** the GPU claim worker's existing
`HwMetricsSampler` thread drives it. With hw-watch enabled the loop ticks at
the detail interval while the `NOTIFY hw_metrics` broadcast keeps its own 5 s
schedule — the live-dashboard contract is unchanged.

**Standalone (a box under investigation, workers parked):**

```bash
QUEUE_WORKFLOWS_DB_BACKEND=pg QUEUE_WORKFLOWS_DB_URL=postgresql://… \
    queue-hw-watch                  # records + prunes until killed
queue-hw-watch --once               # one sample per tier, then exit
```

The CLI bootstraps the engine chain idempotently (the `queue-broker`
standalone precedent), so it works against a fresh DB.

## Reading it back

```python
from queue_workflows import hw_watch
rows = hw_watch.recent_hw_samples(host_label="box-b", tier="detail",
                                  since_s=900)
# rows[0]["data"]["gpu"][0]["throttle_hex"]  → "0x0000000000000048"
# rows[0]["data"]["tz"]                      → [{"type": "acpitz", "milli_c": 94600}, …]
```

`data` comes back as a real dict on both backends (sqlite stores JSON text).

## Reading a crash

Pull the last detail rows before the heartbeat gap and look at, in order:
1. `gpu[].throttle_hex` — `0x40`/`0x08` bits appearing = hardware protection
   engaging (the last warning before a firmware kill).
2. `tz[]` — which zone was climbing toward its (possibly OS-invisible)
   limit; compare against a healthy box at the same workload.
3. `gpu[].sm_mhz` falling while `util_pct` stays pinned = clocks being pulled
   by protection.
4. `mem.available_kb` — rules unified-memory exhaustion in or out.
