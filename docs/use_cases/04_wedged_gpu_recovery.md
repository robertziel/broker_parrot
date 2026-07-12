# Use case 4 — a wedged GPU job recovers itself (the watchdog stack)

**Scenario.** A GPU node body stops making progress: a model load that will
never finish, an inference loop stalled at 0 % GPU, a driver-level hang. The
process is alive — leases keep renewing — so the lease-reclaim of
[use case 1](01_box_power_loss.md) never fires. Something else has to notice.

## Three in-process watchdogs, one policy

Every claimed job is bracketed by daemon watchdog threads, each hard-exiting
with a **distinct code** so the cause is readable straight off the exit status:

| Watchdog | Exit | Trips on |
|---|---|---|
| `Watchdog` (wall-clock) | 75 | `elapsed >= budget_for(job)` |
| `StallWatchdog` (no-progress) | 76 | beat gap ≥ 120 s — armed only after the first per-step `beat()`, so a minutes-long cold model load is never policed |
| `GpuHealthWatchdog` | 78 | per-container GPU idle **and** RAM static across a 300 s window (no GPU work + no memory movement ⇒ wedged); first window gets a 20-min load grace |

All three funnel through one policy point — **re-queue-and-retry, not fail**:
the row flips `running -> queued`, `watchdog_retries` increments, **no**
dispatch event is written (the *run* stays running; only this node re-runs),
and the process hard-exits so a fresh worker — usually a different box —
re-claims the job. Only when `watchdog_retries` reaches
`QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES` (default 3) does the trip fall back to *fail*
(terminal status + the `failed` dispatch event in one transaction). A single
transient wedge never kills a whole workflow.

## The layer in-process threads can't reach

A true **hardware hang** can defeat all three watchdogs — on a GIL-holding
hang the threads never run; on some ROCm hangs the box-level probe still reads
non-idle. The worker then sits wedged while its heartbeat freezes. The
orchestrator is a **separate process**, so its dead-worker sweep
(`flag_stale_workers_holding_running_jobs`, every 5 s) catches what the inside
view can't: a heartbeat stale > 30 s **while the worker still owns a running
job** ⇒ flag `last_flagged_dead_at` + an actionable `DEAD WORKER:` ERROR. The
JOB is recovered by the lease-reclaim as usual (front-of-queue); the flagged
PROCESS is for a host supervisor to bounce — the orchestrator can't safely
cross-host-kill it.

## Forensics: the append-only event log

Every attempt writes to `workflow_node_events` (migration `0011`): `claimed`,
`model_load_*`, `progress_beat`, `stall_*`, `gpu_health_trip`, `budget_trip`,
`requeued`, terminal events — with `attempt = watchdog_retries` tying the
tries of one node together. The mutable job row only shows the last attempt;
the event log shows the story. Best-effort writes on the hot path (an event
blip can never fail a claim), transactional writes for terminal/`requeued`
(they ride the state-change txn).

Full treatment: [`../watchdogs.md`](../watchdogs.md).
