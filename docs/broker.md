# 🦜 The shared broker — one queue for many projects

*How several projects share one Postgres queue with the `project` tag (migration 0017).*

## The shared broker (multitenant `project` tag, migration 0017)

### Why

Historically each project ran its **own** Postgres and its own engine instance, so a fleet of N projects was N isolated single-tenant deployments — N databases to patch, N sets of dashboards, no fleet-wide view. The engine's queue was already partitioned by *resource* (one `cpu` queue + one `gpu` queue, `workflow_node_jobs.queue`) but never by *project*: you could only see one project per DB. Many sibling projects sharing one DRY engine source makes "one queue per resource, shared across every project" the natural next step — DRY infrastructure, not just DRY code.

The model: **shared DB + per-project clients.**

```
                 Broker  =  ONE shared Postgres
        workflow_node_jobs(queue=gpu, project=alpha, …)
        workflow_node_jobs(queue=gpu, project=beta, …)
        ingest_jobs(queue=fetch, project=alpha, …)
        worker_heartbeats(host_label=host-a, queue=gpu, project=alpha, …)
        worker_heartbeats(host_label=host-a, queue=gpu, project=beta, …)
                              │
            LISTEN/NOTIFY + worker_control (project-scoped)
                              │
        ┌─────────────────┬──┴──────────────┬─────────────────┐
   [alpha client]    [beta client]    [gamma client]    …each holds its
    orchestrator +      orchestrator +     orchestrator +   OWN code, claims
    cpu/gpu workers     cpu/gpu workers    cpu/gpu workers   ONLY its project's
                                                              rows.
```

- **The broker is the shared Postgres.** It holds *every* project's jobs on the shared `cpu`/`gpu` (+ ingest) queues — one place to see the whole fleet, one set of counts.
- **Each project runs its own client** (orchestrator + claim workers + scheduler) pointed at the shared broker DB. Workers stay project-local because they must import that project's node code (`config.node_module_package` / `node_resolver`). A physical GPU box can run several projects' workers as separate processes/containers sharing the hardware; each claims only its own project's rows.
- **No process imports another project.** The tenant boundary is the `project` tag plus each client's own `config.project` — not code sharing.

### Design rule — exact-match-always

Every queue row carries `project TEXT NOT NULL DEFAULT ''`. **Enqueue** stamps it from the parent run (or `config.project`). **Claim** filters `AND project = <this client's project>` *unconditionally*, resolved from `config.project`. There is no "claim-any-on-empty" special case:

- **Single-tenant (default `project=""`):** every row is `''` and the filter `project=''` matches them all → byte-compatible with a pre-0017, one-Postgres-per-project deploy, zero host wiring.
- **Multi-tenant:** each client sets `config.project = X` (`configure(project="X")`), so it enqueues `X` and claims `X`. Claiming another tenant's row is not even expressible.

`project=""` is deliberately the single-tenant sentinel, not a config a host has to set. `project` **pools** tenants into one queue with a filter. See [configuration](configuration.md) for the full field docs.

### What `project` touches

| Area | Change |
|---|---|
| Schema (0017 + 0018/0019 follow-ups) | `project` column on `workflow_runs`, `workflow_node_jobs`, `ingest_jobs`, `worker_heartbeats`, `workflow_node_events` (0018), `worker_controls` (0019); `worker_heartbeats` PK widened to `(host_label, queue, project)` so two projects' workers can share a machine without clobbering each other's heartbeat row; project-aware claim indexes. |
| `config` / `configure` | `config.project: str`, default read from `QUEUE_WORKFLOWS_PROJECT` env, settable via `configure(project=...)`. |
| Enqueue paths | `enqueue_node_job` / `insert_skipped_job` / `enqueue_ingest_job` stamp `project` (default → `config.project`). |
| Claim paths | `claim_next_{cpu,gpu,ingest}_job` filter by `project` (default → `config.project`). |
| Recovery / telemetry | `upsert_worker_heartbeat`, `clear_worker_current_model`, `flag_stale_workers_holding_running_jobs`, `reclaim_all_running_for_resume`, `vlm_pool_should_defer`, `flag_unassignable_gpu_jobs`, `requeue_running_for_worker` are project-scoped — on a shared broker `host_label` is no longer globally unique, so any `claimed_by`/`host_label` join must also match `project`. |
| Snapshots | `snapshot` / `ingest_snapshot` / `fleet_snapshot` take an optional `project` filter (`None` = broker-wide). |
| Worker control (0019) | `worker_controls` keyed `(host_label, queue, project)`; every accessor defaults `project` to `config.project`, so `queue-worker-control --project` targets another tenant explicitly. See [worker_control](worker_control.md). |
| Genuinely un-scoped (safe) | `reclaim_expired_leases`/`reclaim_expired_ingest_leases` (act on a row in place — project travels with it), cancel/terminal paths keyed by `run_id`/`job_id`, `prune_node_events`, and `fleet_snapshot()` with no filter (the deliberate broker-wide cross-project view). |

### Standing up a shared broker

**Step 1 — the broker schema.** Point any project's orchestrator at the broker DB: `queue-orchestrator` runs `db.bootstrap()` on boot. A shared, multi-host broker is Postgres, so select `pg` explicitly (`QUEUE_WORKFLOWS_DB_BACKEND=pg`) — the library default is `sqlite` (v1.0.0 breaking default, see [configuration](configuration.md)). `db.bootstrap()` takes a Postgres advisory lock, so concurrent orchestrator boots against one shared broker are safe (the lock serializes; a late bootstrap that finds the chain already applied is a no-op).

**Step 2 — every process of every project points at that broker and names itself:**

```python
queue_workflows.configure(project="alpha", db_backend="pg", db_url_env="BROKER_DSN")
queue_workflows.configure(project="beta",  db_backend="pg", db_url_env="BROKER_DSN")
# … each then enqueues + claims ONLY its own project's rows on the ONE shared
#   cpu/gpu (+ ingest) queue.
```

This must run for **every** process of that project — orchestrator, claim workers, scheduler — or export `QUEUE_WORKFLOWS_PROJECT=<name>` once in the deploy environment (the env knob also reaches entrypoints that hand-roll their own `configure()`, e.g. standalone scripts, mirroring `QUEUE_WORKFLOWS_DB_BACKEND`).

**Step 3 — watch the consolidated queue across all projects:** `node_queue.list_projects()` enumerates the tenants and `node_queue.snapshot(project=p)` gives each one's `cpu`/`gpu` queued/running depth (`snapshot()` with no filter is the broker-wide view).

### Cutover — adopting a project name on an existing deploy

Migration 0017 backfills every pre-existing row to `project=''`. Because claiming is exact-match, the instant a running deploy switches to `configure(project="alpha")` its client stops seeing the backfilled `''` rows — any in-flight queued/running work would be stranded. Adopt a project name only on a **drained** queue, or run a one-time backfill in the same maintenance window:

```sql
UPDATE workflow_runs        SET project = 'alpha' WHERE project = '';
UPDATE workflow_node_jobs   SET project = 'alpha' WHERE project = '';
UPDATE ingest_jobs          SET project = 'alpha' WHERE project = '';
UPDATE worker_heartbeats    SET project = 'alpha' WHERE project = '';  -- or let stale rows age out
```

A deploy that stays single-tenant needs none of this.

### Operational notes

- **`worker_heartbeats` writes must go through `node_queue.upsert_worker_heartbeat`.** The PK is `(host_label, queue, project)`; a consumer upserting with a hand-rolled `ON CONFLICT (host_label, queue)` will fail with "no unique or exclusion constraint matching the ON CONFLICT specification".
- **The 0017 down-migration is only safe on a single-tenant (all-`''`) DB** — dropping `project` and re-adding the 2-column heartbeat PK would collapse two projects' rows sharing `(host_label, queue)` into duplicates.

## See also

- [architecture.md](architecture.md) — the three process roles, the lease/reclaim/watchdog model, and DAG dispatch.
- [configuration.md](configuration.md) — every `EngineConfig` field, including `project` and `db_backend`.
- [schema.md](schema.md) — the full per-table column reference.
- [storage_backends.md](storage_backends.md) — the `db_backend` seam (sqlite or pg).
- [worker_control.md](worker_control.md) — the operator ON/OFF plane (`worker_controls`, migrations 0012/0019).
- [watchdogs.md](watchdogs.md) — the in-process liveness watchdogs.
- [gpu_and_llm.md](gpu_and_llm.md) — GPU model cache and LLM backends.
- [deployment.md](deployment.md) — running these processes/containers in practice.
