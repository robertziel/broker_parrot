# Use case 2 — a box boots (or reboots) and rejoins the fleet

**Scenario.** A machine powers on — first boot of a new fleet member, or the
recovery boot after [use case 1](01_box_power_loss.md). Its claim worker(s)
must join the fleet without racing migrations, without violating an operator's
OFF switch, and without double-running anything it held before the reboot.

## Boot sequence (what a claim worker actually does)

1. **Schema gate.** The worker never runs migrations. It calls
   `db.wait_for_schema(min_version)` and **blocks** until the orchestrator's
   `db.bootstrap()` has brought the ledger (`queue_schema_version`) far enough
   (each queue knows its minimum version). A brand-new box pointed at a
   brand-new DB simply waits for the orchestrator instead of racing it.
2. **Park check.** It reads `worker_controls` for its `(host_label, queue)`
   row. If the desired state is `off`, the worker **parks** — process up,
   heartbeating, but not claiming — until an operator turns it back ON. A row
   that doesn't exist means ON (default-on), and a DB predating the control
   plane is treated as all-ON, so old deployments boot unchanged.
3. **Heartbeat + LISTEN.** It starts the 10 s heartbeat
   (`worker_heartbeats` — which also *clears* a dead-worker flag from a prior
   crash), executes `LISTEN <channel>`, then **greedily drains** the queue:
   claim, run, claim again, until empty. A 1 s safety poll covers any dropped
   NOTIFY.
4. **Cold caches are fine.** A GPU worker boots with an empty warm
   `ModelCache`; the affinity tiebreak simply stops preferring this box until
   its first model load publishes `current_model` in the heartbeat.

## The rejoin hazard: work it held before the reboot

Whatever the box was running when it went down was already handled by the
lease-reclaim (front-of-queue requeue — see
[use case 1](01_box_power_loss.md)). By the time the box is back, its old job
is either running on a peer or already terminal. Three protections make the
rejoin safe:

- the rebooted worker holds **no claim state** — claims live in the DB, and its
  old rows now say `claimed_by = <someone else>` or a terminal status;
- if the process somehow *survived* (partition, not power loss), the
  `JobStatusWatcher` hard-exits it (`os._exit(77)`) the moment it observes the
  reassignment — the kill signal that prevents a double-run;
- terminal marks are idempotent (`WHERE status NOT IN (…)`), so even a
  photo-finish duplicate write is a no-op.

## Operator checklist for adding a brand-new box

```bash
# on the new box (env: DSN + host label)
queue-claim-worker --queue gpu     # blocks on wait_for_schema until ready
queue-claim-worker --queue cpu
```

Nothing to register centrally: the first heartbeat makes the box visible in
`worker_heartbeats`, and the claim loop makes it productive. To stage it
half-on, pre-write the `worker_controls` row `off` and flip it when ready —
see [03 — operator stop and park](03_operator_stop_and_park.md).
