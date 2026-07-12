# Use case 3 — operator stops / parks a worker (ON/OFF control plane)

**Scenario.** You need box-b's GPU worker out of the pool *now* — to upgrade a
driver, to hand the GPU to something else, to stop a misbehaving deploy — and
you want its in-flight job redistributed, not failed.

## One row is the whole interface

Desired state is a row in `worker_controls` (migration `0012`), so anything
that can write SQL is an operator console:

```bash
queue-worker-control --queue gpu --off --host box-b     # CLI helper
```

```sql
-- …or plain SQL from any app sharing the DB; the trigger wakes the worker.
INSERT INTO worker_controls (host_label, queue, desired_state, stop_policy, requested_by)
VALUES ('box-b', 'gpu', 'off', 'hard', 'ops')
ON CONFLICT (host_label, queue) DO UPDATE
  SET desired_state = EXCLUDED.desired_state, updated_at = now();
```

A row trigger fires `pg_notify('worker_control', 'box-b:gpu')` **inside the
writer's transaction**, and the worker's `WorkerControlWatcher` (LISTEN + 5 s
safety poll) reacts immediately.

## What OFF does — hard stop, fault-free requeue

1. The in-flight job is **requeued resume-style, at the front**
   (`is_priority = TRUE` + band bump) — an operator stop is *not* a fault, so
   `watchdog_retries` is **not** bumped and nothing is marked failed.
2. The process **hard-exits (`os._exit(79)`)**. The node body runs inline on
   the worker's main thread, so process exit is the only thing that reliably
   stops a wedged CUDA kernel and *actually frees the VRAM*.
3. The supervisor (systemd/docker) restarts the worker; on boot it re-reads
   `worker_controls`, sees OFF, and **parks** — up and heartbeating, claiming
   nothing — until you flip it back:

```bash
queue-worker-control --queue gpu --on --host box-b      # resumes in place
```

## Design notes

- **Desired vs observed** are deliberately separate tables: `worker_controls`
  is what you *want*; `worker_heartbeats` is what *is*. An OFF must persist
  precisely while the worker isn't beating.
- `stop_policy` is free-form TEXT with a registry (`STOP_POLICIES`) — only
  `"hard"` exists today; `"drain"`/`"pause"` are reserved names that slot in
  with no schema change.
- A missing row = ON, and a pre-`0012` database is treated as all-ON — the
  control plane is strictly opt-in.
- Scope: per `(host_label, queue)` — parking box-b's `gpu` worker leaves its
  `cpu` worker untouched. On a shared broker the key includes `project`
  (migration `0019`), so two projects' workers on one box are controlled
  independently.

Full design rationale: [`../worker_control.md`](../worker_control.md).
