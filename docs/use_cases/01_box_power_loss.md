# Use case 1 — a box powers off mid-job

**Scenario.** A worker box (say a GPU node) loses power — someone pulls the
plug, PSU dies, kernel panics — while its worker is mid-way through a claimed
job. Minutes-to-hours of wall-clock may already be invested in the job. Two
things must be true afterwards:

1. the job **must not wait its turn again** — it already queued once and burned
   run time; it should be *the next thing* a healthy peer picks up;
2. if the "dead" box comes back (power restored, network partition heals), the
   returning worker **must not double-run** the job — it has to receive a kill
   signal before it can write anything.

The engine does both, with no operator action.

## Timeline

```
t0      box claims job J (status=queued -> running, claimed_by=box-a,
        lease_expires_at = now + 600 s)
t0+…    worker renews the lease every 10 s while J runs
tX      ── power off ── renewals stop; heartbeats stop
tX+~30s the orchestrator's dead-worker sweep flags box-a
        (worker_heartbeats.last_flagged_dead_at) and logs an actionable
        "DEAD WORKER:" ERROR — the PROCESS is now a supervisor problem
tX+lease the lease lapses; the orchestrator's reclaim sweep (every 5 s)
        flips J: running -> queued, claimed_by cleared, and — because the
        parent run is still live — **is_priority = TRUE** and
        priority = LEAST(priority, 10)
tX+ε    the requeue re-fires the NOTIFY; every listening peer wakes
tX+ε'   a healthy peer claims J FIRST — is_priority DESC sorts ahead of
        the priority band AND ahead of GPU warm-model affinity
```

## Front of the queue, not the back

The reclaim (`node_queue.reclaim_expired_leases`) does two priority writes in
the same guarded `UPDATE`:

- **`is_priority = TRUE`** — the binary "run next" flag (migration `0016`).
  The claim `ORDER BY` is `is_priority DESC, priority ASC, …`, so a reclaimed
  job beats every band and even the warm-model affinity tiebreak. Work that
  already burned wall-clock on a lost machine is deliberately allowed to force
  a model swap on the healthy box — the box holding the warm copy is the one
  that just died.
- **`priority = LEAST(priority, 10)`** — the band bump, so even engines/paths
  that don't know about the flag treat it as urgent.

Two guards keep this honest:

- a job whose **parent run is already terminal** is *cancelled*, not requeued —
  the claim SQL filters non-running parents, and a flagged ghost would
  otherwise sit at the head of the queue forever;
- **ingest jobs** (the second job family) have no `is_priority` column — their
  reclaim (`reclaim_expired_ingest_leases`) is band-only (`LEAST(priority, 10)`),
  which still front-loads them within their queue.

## The kill signal — when the "dead" box comes back

Power returns. The box boots, the supervisor restarts the worker — but suppose
the OS never actually died (a network partition, a hung switch) and the OLD
worker process is still alive, still executing J's node body, and about to
write a terminal status for a job that has since been handed to a peer.

That worker kills **itself**: every claimed job is bracketed by a
`JobStatusWatcher` daemon thread that polls `workflow_node_jobs` (every 2 s)
and **hard-exits the process (`os._exit(77)`)** the instant the row is no
longer `claimed_by` this host. The reclaim cleared `claimed_by` — so the
zombie trips the watcher before it can double-write. `os._exit` is deliberate:
it is the only signal that reliably abandons a node body wedged deep in a CUDA
kernel, and it frees the GPU's VRAM with the process.

The watcher is scoped to `claimed_by`, **not** `status` — a worker's own
`mark_completed`/`mark_failed` leaves `claimed_by` intact, so finishing your
own job never trips it; only an external hand-off does.

Idempotency backstops the race window: terminal marks are
`UPDATE … WHERE status NOT IN ('completed','failed','cancelled')` — if the
zombie somehow finished a nanosecond before the exit, whichever write lands
second is a no-op, never a clobber.

## What the operator sees

- The dead-worker flag (`last_flagged_dead_at`) + the `DEAD WORKER:` log line
  say which host needs a power/supervisor intervention. The engine will not
  cross-host kill a machine — flagging is the correct boundary.
- The job itself needs **nothing**: it re-ran on a peer. A fresh heartbeat from
  the recovered box clears the flag.

## Mechanisms involved

| Mechanism | Where | Numbers |
|---|---|---|
| Lease renew | `claim_worker.LeaseRenewer` | every 10 s, lease 600 s |
| Reclaim sweep | `node_pool` → `node_queue.reclaim_expired_leases` | every 5 s (`QUEUE_WORKFLOWS_LEASE_RECLAIM_INTERVAL_S`) |
| Front-of-queue flag | migration `0016`, `is_priority DESC` in the claim `ORDER BY` | binary, ahead of band + affinity |
| Dead-worker flag | `node_queue.flag_stale_workers_holding_running_jobs` | heartbeat stale > 30 s (`QUEUE_WORKFLOWS_STALE_WORKER_AFTER_S`) |
| Zombie kill signal | `claim_worker.JobStatusWatcher` | poll 2 s, `os._exit(77)` |
| Double-write safety | idempotent terminal `WHERE` guard | any duplicate is a no-op |

See also: [02 — a box boots](02_box_boot_and_rejoin.md),
[04 — wedged GPU recovery](04_wedged_gpu_recovery.md),
[`../watchdogs.md`](../watchdogs.md).
