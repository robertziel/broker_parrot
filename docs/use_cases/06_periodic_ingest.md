# Use case 6 — periodic background work (the ingest family)

**Scenario.** Every deployment has non-DAG background work: poll an upstream
API every 15 minutes, refresh a cache nightly, run a parametrised scenario on
demand. That's the engine's **second job family** — `ingest_jobs`: standalone
jobs on **your own queue names**, no DAG, no parent run.

## Register a task, schedule it — or enqueue directly

```python
queue_workflows.configure(
    ingest_queues=frozenset({"fetch", "scenario"}),   # your names; cpu/gpu are reserved
    ingest_default_budget_s=3600,                     # watchdog budget for these queues
)
queue_workflows.register_ingest_task("refresh_listings", refresh_listings)  # fn(reason) or fn(reason, args)
queue_workflows.set_ingest_schedule([ScheduleEntry(...)])                   # the queue-scheduler ticker fires these
```

The `queue-scheduler` process is a DB-native ticker — it sleeps to the next
scheduled minute and inserts `ingest_jobs` rows. No cron, no beat daemon.

## Atomic with your own writes

Direct enqueues accept a caller connection, so the job insert (and its NOTIFY)
**rides your transaction** — the job exists if and only if your domain row
does:

```python
with my_pool.connection() as conn:
    my_create_scenario(conn, scenario_id)
    node_queue.enqueue_ingest_job(
        task_name="run_scenario", queue="scenario",
        args={"scenario_id": scenario_id}, conn=conn,
    )
```

## Same machinery, one deliberate difference

Ingest jobs get the same claim (`FOR UPDATE SKIP LOCKED` + `ingest_job_ready`
NOTIFY), the same lease renew/reclaim, wall-clock budgets, and heartbeats
(`ingest_snapshot()` reports per-queue depth + workers). The difference: on a
machine-loss reclaim they take the band bump (`LEAST(priority, 10)`) but have
no `is_priority` column — front-of-band, not absolute-front. And a watchdog
trip **fails** an ingest job rather than retrying it (no parent run, no retry
counter) — periodic work's natural retry is its next scheduled tick.

Validation is host-side (registered `task_name`s, configured queue names), so
adding a queue is a `configure()` change, not a migration.
