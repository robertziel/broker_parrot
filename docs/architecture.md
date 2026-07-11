# 🧭 Architecture

*The mental model: Postgres is the message bus, three processes share it, and everything domain-specific is an injected hook.*

This doc is the map. For the liveness/lease/watchdog detail see [watchdogs](watchdogs.md); for table shapes see [schema](schema.md); for the full config-hook reference see [configuration](configuration.md); for how the v2 `broker_service` inverts part of this model see [broker](broker.md).

## 1. The database is the message bus

There is no separate broker process to run, monitor, or keep in sync with the source of truth. A piece of work is a *row*. Enqueuing it is an `INSERT`. Claiming it is a single atomic `UPDATE … FOR UPDATE SKIP LOCKED`.

The key correctness property is that **enqueue and wake happen in the same commit**. A row trigger (added in migration `0006` for `workflow_node_jobs`, `0007` for `ingest_jobs`) fires `pg_notify('node_job_ready', <queue>)` / `pg_notify('ingest_job_ready', <queue>)` on `INSERT` or on any `UPDATE` that flips a row's `status` to `queued`:

```sql
-- queue_workflows/migrations/0006_pg_queue_lease.sql
CREATE OR REPLACE FUNCTION notify_node_job_ready() RETURNS trigger AS $$
BEGIN
    IF NEW.status = 'queued' THEN
        PERFORM pg_notify('node_job_ready', NEW.queue);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER node_job_ready_notify
    AFTER INSERT OR UPDATE OF status ON workflow_node_jobs
    FOR EACH ROW EXECUTE FUNCTION notify_node_job_ready();
```

Because the `PERFORM pg_notify(...)` runs *inside the writer's own transaction*, there is no "row queued but nobody got woken" window and no "notified but the row isn't visible yet" race — either both land on commit or neither does. This is also why a lease reclaim (flipping a lapsed `running` row back to `queued`) re-wakes an idle worker for free: it's just another `UPDATE OF status`, so the same trigger fires.

A claim worker doesn't have to catch every NOTIFY to stay correct — `LISTEN` is paired with a 1 s safety poll (see §3), so a dropped notification only costs up to a second of latency, never a stuck job.

## 2. Three process roles, one Postgres

All three run as independent OS processes against the same database. None of them talk to each other directly — Postgres rows and NOTIFY channels are the only interface.

| Process | Entry point | Owns |
|---|---|---|
| **Orchestrator** | `orchestrator.py` → `node_pool.NodePool` | migrations, DAG dispatch, outbox drain, lease-reclaim sweeps, `InputListener`, dead-worker sweep |
| **Claim worker** | `claim_worker.ClaimWorker` | claiming + running exactly one job at a time on one queue |
| **Scheduler** | `scheduler.Ticker` | enqueuing periodic `ingest_jobs` rows |

### Orchestrator

`orchestrator.py` (console script `queue-orchestrator`) is the **only process that bootstraps migrations** (`db.bootstrap_from_schema()` then `db.bootstrap()`, both idempotent — safe to run on every boot). It then does two startup-health re-queues — `run_store.reenqueue_running_for_resume()` for orphaned `running` runs and `node_queue.reclaim_all_running_for_resume()` for orphaned `running` node-jobs left by a fleet-wide restart — before starting `NodePool` and blocking on `SIGINT`/`SIGTERM`.

`NodePool` runs a single background dispatch thread (`_dispatch_loop` → `_tick`, default poll every 0.5 s) that does, in order, on every tick:

1. **Expand freshly-queued `mode='node'` runs** — `run_store.list_queued_node_run_ids()` then `dispatcher.start_run(run_id)` for each, which enqueues the DAG's entry nodes as `workflow_node_jobs` rows.
2. **Drain the dispatch-event outbox** (`_drain_dispatch_events`, see §4).
3. **Sweep expired node-job leases** (`node_queue.reclaim_expired_leases`, interval-gated to ~5 s) — re-queues `running` rows whose lease lapsed.
4. **Sweep expired ingest leases** (`ingest_store.reclaim_expired_ingest_leases`) — the `ingest_jobs` twin of step 3.
5. **Sweep dead workers** (`node_queue.flag_stale_workers_holding_running_jobs`) — flags (does not kill) a worker whose heartbeat froze while it still owns a `running` row; see [watchdogs § last-resort layer](watchdogs.md).
6. **Prune old `workflow_node_events`** rows past the retention window.
7. **Orphan-cancel sweep** (opt-in via `configure(cancel_orphan_queued_jobs=True)`) — flips `queued` jobs of already-terminal runs to `cancelled`.
8. **Stuck-run reconciler** (`dispatcher.reconcile_run`, interval-gated to 5 min, fires instantly on the first tick) — re-drives runs the engine still calls non-terminal but that have no live node-job (e.g. a resume that landed on a cancelled-node dead end).
9. **Unassignable-job sweep** (`node_queue.flag_unassignable_gpu_jobs`) — red-flags queued GPU jobs no live machine has enough VRAM to hold (capacity-aware assignment, migration `0015`).

Plus the `InputListener` (polls `workflow_input_submissions`; when a value lands for an `awaiting_input` node it calls `dispatcher.resume_after_input` to unblock the DAG). **No node body ever runs in the orchestrator process** — it only moves rows and drains events.

### Claim worker

`claim_worker.ClaimWorker` (console script `queue-claim-worker --queue=<name>`) is **one process = one worker, concurrency-1 by contract** — there is no in-process thread pool claiming multiple jobs. `run_forever` opens a dedicated autocommit connection, issues `LISTEN <wake_channel>`, then loops:

```
LISTEN node_job_ready   (or ingest_job_ready for an ingest queue)
while running:
    claimed = claim_next_cpu_job / claim_next_gpu_job / claim_next_ingest_job(...)
    if claimed:
        run it through execute_node under a LeaseRenewer + watchdog(s)
        loop immediately — drain the queue greedily
    else:
        block on the notify with a 1 s safety-poll timeout, then re-loop
```

The 1 s poll timeout is the belt-and-suspenders for a dropped NOTIFY (network hiccup, a PG restart mid-listen) — worst case a claimable row waits ~1 s longer than it should. Which claim function runs depends on the queue name: `cpu`/`gpu` (the reserved `_NODE_QUEUES`) draw DAG node-jobs from `workflow_node_jobs`; anything in `config.ingest_queues` (default `{fetch, load}`) draws standalone jobs from `ingest_jobs`. Before entering the loop each worker blocks on `db.wait_for_schema(min_version)` — only the orchestrator bootstraps, so a worker waits for the schema instead of racing the migration run.

A GPU worker additionally owns the process-wide warm `ModelCache` (`gpu_model_cache.py`) and publishes its `current_model` to `worker_heartbeats` for the warm-model affinity tiebreak (§3). Every claimed job is bracketed by a `LeaseRenewer` and one or more watchdogs (wall-clock, stall, GPU-health) plus a `JobStatusWatcher`; the worker process as a whole runs one `WorkerControlWatcher` for the operator ON/OFF plane. All of that liveness machinery is covered in full in [watchdogs](watchdogs.md) and [worker_control](worker_control.md) — this doc only needs you to know it exists and brackets every claim.

### Scheduler

`scheduler.Ticker` (console script `queue-scheduler`) is a plain Python loop — **not** `pg_cron` — that sleeps to the next scheduled minute and `INSERT`s an `ingest_jobs` row per fire. It reads its schedule as host-injected data (`config.ingest_schedule`, empty by default, set via `queue_workflows.set_ingest_schedule([...])`). Firing is just another `INSERT` into `ingest_jobs`, so it rides the same `ingest_job_ready` trigger as any other producer — the scheduler has no special wake path.

## 3. The queue mechanism

The claim is a **single SQL statement**: a `FOR UPDATE SKIP LOCKED` subselect picks the next claimable row for a queue, and the outer `UPDATE` flips it `queued → running` and stamps `claimed_by` + `lease_expires_at` in the same statement (`node_queue._CLAIM_SQL`):

```sql
UPDATE workflow_node_jobs AS j
SET status = 'running',
    started_at = now(),
    worker_lane = %(worker_lane)s,
    claimed_by = %(host)s,
    lease_expires_at = {lease_expr}
WHERE j.id = (
    SELECT c.id FROM workflow_node_jobs c
    WHERE c.queue = %(queue)s
      AND c.status = 'queued'
      AND c.project = %(project)s
      AND EXISTS (
          SELECT 1 FROM workflow_runs r
          WHERE r.id = c.run_id
            AND r.status NOT IN ('cancelled', 'failed')
      )
      {capability}
    ORDER BY {order}
    {skip_locked}
    LIMIT 1
)
RETURNING *
```

`{order}` and `{capability}` are the **only** interpolations, and both are built exclusively from validated ints and fixed SQL fragments in `node_queue.py` — **never from caller-supplied strings**. That's what keeps this claim SQL-injection-safe despite the string formatting. The subselect also folds in a run-cancel guard so a worker can never claim a job whose parent run was already cancelled or failed out from under it.

**Claim ordering** (`ORDER BY`), CPU:

```
c.is_priority DESC, c.priority ASC, (created_at direction by host_priority)
```

GPU adds a warm-model affinity term between the two:

```
c.is_priority DESC, (required_model IS NOT DISTINCT FROM current_model) DESC, c.priority ASC, (created_at direction by host_priority)
```

- **`is_priority`** — a boolean "run next" flag (`node_queue.prioritize_node_job`, migration `0016`) that sorts first of everything, including warm-model affinity: an operator-flagged cold-model job preempts a warm one, because the reload is the accepted cost of "run this next."
- **Warm-model affinity** (GPU only) — rows whose `required_model` matches the claiming worker's `current_model` (`IS NOT DISTINCT FROM`, null-safe) sort ahead of the `priority` band, so consecutive same-model jobs avoid a reload.
- **`priority`** — an integer band (default `100`, lower = sooner).
- **`host_priority`-directed creation tiebreak** — normally oldest-first (FIFO within a band); an explicit negative `host_priority` (an "overflow" box) walks newest-first, so it naturally claims the tail rather than competing for the head with priority hosts.

GPU claiming also applies a **capability gate** (only claim a job whose `required_model` is in the worker's `known_models`, or that needs no model) and an optional **lane filter** (`require_model` True/False) that splits the GPU queue into an inline warm-model lane and a no-model VLM-pool lane so a two-lane GPU worker never over-claims the other lane's rows — see `node_queue.claim_next_gpu_job`'s docstring for the full lane-filter contract.

## 4. DAG dispatch + the durable outbox

`dispatcher.py` is **pure DAG-walk logic** — no queue-transport coupling, unit-testable without a worker pool. It does three things:

1. `start_run(run_id)` — reads the workflow/pipeline definition via the injected loader hooks and enqueues every node with no `depends_on`.
2. `on_node_completed` / `on_node_failed` / `on_node_awaiting_input` — given one node's terminal event, finds downstream nodes whose deps are all `completed`/`skipped` and enqueues them (or inserts a `skipped` marker per that node's `skip_if`); a failure cancels siblings and fails the run; an empty frontier completes the run.

The **worker → dispatcher handoff is an outbox**, not a direct call. When a worker finalizes a node (`node_executor.execute_node`), it writes the terminal status **and** a `workflow_dispatch_events` row in **one transaction**:

```python
with _db_connection() as conn, conn.cursor() as cur:
    row = node_queue.mark_completed_in_txn(cur, job_id, context_delta=..., seconds=..., vm_rss_mb_peak=...)
    if row is None:
        return "skipped"  # already terminal — duplicate delivery / claim-race loser
    node_queue.enqueue_dispatch_event_in_txn(cur, job["run_id"], job["node_id"], "completed")
```

This is deliberate decoupling: fan-out is never synchronously coupled to the worker that ran the node. A worker can crash the instant after this transaction commits and the graph still advances, because the event is already durable. The orchestrator's `_drain_dispatch_events` (step 2 of `_tick`, §2) pops unprocessed rows with its own `FOR UPDATE SKIP LOCKED` and calls the matching `dispatcher.on_node_*` callback. A callback that raises is retried on the next tick (`attempts` incremented, error recorded); after `_DISPATCH_MAX_ATTEMPTS` (10) the event is **poison-flagged** and the run is force-failed so operators see something instead of a silent stall, rather than retrying forever.

## 5. Two job families

The engine carries two independent job shapes over the same claim/lease/watchdog machinery:

| | DAG node-jobs | Ingest jobs |
|---|---|---|
| Table | `workflow_node_jobs` | `ingest_jobs` |
| Queues | `cpu` / `gpu` (reserved) | host-defined (`config.ingest_queues`, default `{fetch, load}`) |
| Enqueued by | the dispatcher, fanning out a run | the scheduler ticker, or any host directly |
| Shape | one node inside a pipeline DAG | a single self-contained task, no dependencies |
| Carries | resolved `$from` inputs from upstream nodes | optional per-job `args JSONB` |

Both families share atomic enqueue-with-notify, lease/reclaim, idempotent terminals, and the dispatch-outbox pattern (ingest jobs have no dispatcher fan-out, but `execute_node`'s ingest twin still writes its terminal atomically). See [schema](schema.md) for the table definitions and [configuration](configuration.md) for `ingest_queues` / `register_ingest_task` / `set_ingest_schedule`.

## 6. The host-agnostic seam

Everything domain-specific — which workflows exist, how to import a node module, which GPU models to register, what ingest tasks/schedule to run, how to resolve a `$from` ref, per-node setup/teardown — is an **injected hook** on a process-wide `EngineConfig` singleton (`config.py`), wired once via `queue_workflows.configure(...)` and the `set_*`/`register_*` helpers in `__init__.py`. **Every hook has a safe default**, so `import queue_workflows` + `configure()` against a reachable database runs standalone with no host wired at all (this is exactly what `tests/test_standalone_import.py` proves). `config.py` is a dependency-graph leaf — it imports nothing from other engine modules — so a host can extend the engine without the engine ever reaching "up" into a host.

The full hook reference (workflow/pipeline provider, node-module resolver, builtin-model registrar, ingest tasks/schedule/queues, invoke-context wrapper, ref resolver, `db_backend`/`project` selection) lives in [configuration](configuration.md) — this doc only needs you to know the shape exists and where the seam sits.

## Where to go next

- **Liveness in depth** (leases, the three watchdogs, the dead-worker last-resort layer) → [watchdogs](watchdogs.md)
- **Table-by-table schema, migration chain (0001–0019)** → [schema](schema.md)
- **All config hooks + env knobs** → [configuration](configuration.md)
- **Storage backend seam** (sqlite/pg vs redis/mongodb) → [storage_backends](storage_backends.md)
- **Operator ON/OFF control plane** → [worker_control](worker_control.md)
- **GPU model cache + LLM server resolution** → [gpu_and_llm](gpu_and_llm.md)
- **The v2 `broker_service`** — inverts this doc's autonomous-worker model into a central pull→grant broker; the process roles above still run underneath it → [broker](broker.md)
- **Deploying the fleet** → [deployment](deployment.md)
