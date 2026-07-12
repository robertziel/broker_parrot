# 🦜 The broker — the shared queue and the v2 control plane

*How ~35 sibling projects share one Postgres queue (Part A), and how the v2 `broker_service` inverts autonomous claiming into broker-granted permission (Part B).*

There are **two** related but distinct things named "broker" in this repo. Keep them separate:

| | Part A — the shared broker | Part B — `broker_service` (v2) |
|---|---|---|
| What it is | The **legacy engine's** own tables (`workflow_node_jobs`, `ingest_jobs`, …), multi-tenanted with a `project` column so one Postgres serves every project | A **new, separate** schema (`bw_jobs`, `bw_workers`) + a pull→grant control plane |
| Claim model | Worker **claims** its own row (`FOR UPDATE SKIP LOCKED`) — autonomous | Worker **asks**; the broker **decides** whether to grant it — inverted |
| Migration chain | `0001`–`0019`, ledgered | None — `ensure_schema()`, idempotent `CREATE TABLE IF NOT EXISTS` |
| Console entry point | `queue-broker` (bootstrap/inspect the shared schema) | `queue-broker-web`, `queue-broker-worker` |
| Status | Delivered, in production use | Software core delivered; web service + panel delivered; k8s-managed LLM servers are a later pass |

Both can point at the *same* Postgres — Part B's tables coexist with Part A's engine tables in one relational store (see [schema](schema.md)) — but they are independent queues with independent semantics. If you're deploying today, you are almost certainly using Part A. Part B is the operator's stated direction: the broker becomes a central orchestrator with a web service, arbitrating a shared core/GPU across projects and (eventually) Kubernetes-managed LLM servers (see [gpu_and_llm](gpu_and_llm.md)).

---

## Part A — the shared broker (multitenant `project` tag, migration 0017)

### Why

Historically each project ran its **own** Postgres and its own engine instance, so a fleet of N projects was N isolated single-tenant deployments — N databases to patch, N sets of dashboards, no fleet-wide view. The engine's queue was already partitioned by *resource* (one `cpu` queue + one `gpu` queue, `workflow_node_jobs.queue`) but never by *project*: you could only see one project per DB. `~35` sibling projects sharing one DRY engine source (see the repo's `CLAUDE.md`) makes "one queue per resource, shared across every project" the natural next step — DRY infrastructure, not just DRY code.

The model, decided by the operator: **shared DB + per-project clients.**

```
                 Broker  =  ONE shared Postgres
        workflow_node_jobs(queue=gpu, project=alpha, …)
        workflow_node_jobs(queue=gpu, project=alpha, …)
        ingest_jobs(queue=fetch, project=alpha, …)
        worker_heartbeats(host_label=host-a, queue=gpu, project=alpha, …)
        worker_heartbeats(host_label=host-a, queue=gpu, project=alpha, …)
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

`project=""` is deliberately the single-tenant sentinel, not a config a host has to set. Contrast this with `db_namespace` (the redis/mongo isolation knob, [storage_backends](storage_backends.md)): `db_namespace` **isolates** tenants on a shared redis/mongo server so they can't see each other; `project` **pools** tenants into one pg queue with a filter — the inverse operation. See [configuration](configuration.md) for both knobs' full field docs.

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

### The `queue-broker` console script (`queue_workflows/broker.py`)

`queue-broker` is the explicit "bootstrap the broker once, then point every project at it" entry point — it turns "one consolidated queue for all projects" into a config flip rather than new code. It imports only the client primitives (`config`, `db`, `node_queue`).

**Step 1 — stand up the broker schema once (idempotent):**

```bash
BROKER_DSN=postgresql://…/broker  queue-broker --db-backend pg --db-url-env BROKER_DSN
```

A shared, multi-host broker is Postgres, so pass `--db-backend pg` explicitly — the library default is `sqlite` (v1.0.0 breaking default, see [configuration](configuration.md)). You do not strictly have to run this *before* the projects: `db.bootstrap()` takes a Postgres advisory lock, so concurrent orchestrator boots against one shared broker are safe (the lock serializes; a late bootstrap that finds the chain already applied is a no-op).

**Step 2 — every process of every project points at that broker and names itself:**

```python
queue_workflows.configure(project="alpha", db_backend="pg", db_url_env="BROKER_DSN")
queue_workflows.configure(project="alpha",    db_backend="pg", db_url_env="BROKER_DSN")
# … each then enqueues + claims ONLY its own project's rows on the ONE shared
#   cpu/gpu (+ ingest) queue.
```

This must run for **every** process of that project — orchestrator, claim workers, scheduler — or export `QUEUE_WORKFLOWS_PROJECT=<name>` once in the deploy environment (the env knob also reaches entrypoints that hand-roll their own `configure()`, e.g. standalone scripts, mirroring `QUEUE_WORKFLOWS_DB_BACKEND`).

**Step 3 — watch the consolidated queue across all projects:**

```bash
BROKER_DSN=…  queue-broker --db-backend pg --db-url-env BROKER_DSN --status
```

`--status` prints the schema version plus every project sharing the broker with its `cpu`/`gpu` queued/running depth (via `node_queue.list_projects()` + `node_queue.snapshot(project=p)`), without bootstrapping:

```
broker schema version: 19
projects on this broker (2):
  alpha                  cpu[q0 r1]  gpu[q3 r1]
  alpha                     cpu[q1 r0]  gpu[q0 r0]
```

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
- **The `project` tag applies to the legacy relational engine path only.** The pluggable `StorageBackend` SPI (`backends/{postgres,redis,mongodb}.py`) has no `project` concept — its multi-tenancy is `db_namespace` (isolation, the inverse of pooling). Selecting redis/mongo does not re-home the orchestrator/worker, so `project` filtering does not apply there. See [storage_backends](storage_backends.md).
- **The 0017 down-migration is only safe on a single-tenant (all-`''`) DB** — dropping `project` and re-adding the 2-column heartbeat PK would collapse two projects' rows sharing `(host_label, queue)` into duplicates.

### Later phases

Cross-project GPU arbitration — today each project's client claims its own rows independently; a true shared queue could let something arbitrate fair-share/priority/preemption *across* projects. That "broker decides, worker asks" idea is exactly what Part B below builds, on a fresh schema rather than retrofitting the legacy tables.

---

## Part B — `broker_service`, the v2 pull→grant control plane

### The inversion

In the legacy engine (Part A) a worker is **autonomous**: it claims a row for itself with `FOR UPDATE SKIP LOCKED` and self-manages its own lifecycle (lease renewal, watchdogs, self-kill on cancel). `queue_workflows.broker_service` inverts this: **the worker asks, and the broker decides.**

The broker owns one shared CPU/GPU queue across every project and:

- **grants** a waiting worker the next job for its project + resource — but only if a cross-project **capacity** gate leaves a slot free (the broker arbitrating a shared core/GPU between competing projects);
- can **revoke** (kill) a job's grant at any time — withdrawing the worker's permission, which the client-side gate observes and stops on;
- **health-checks** workers, and when one goes silent (or its grant lapses) marks it dead and re-queues its job so the next worker can be granted it.

This is a genuinely separate control plane from Part A, not a rewrite of it — it is deliberately outside the `0001`–`0019` migration chain (see [schema](schema.md)) and coexists with the legacy engine tables in the same relational store so the existing suite stays green while this is built out. The stated direction is that this becomes the central orchestrator (web service + operator panel + Kubernetes-managed LLM servers), eventually retiring the old engine — but that retirement hasn't happened; both are live today.

### Schema — `bw_jobs` + `bw_workers` (no ledger)

`queue_workflows/broker_service/schema.py` ships a single idempotent `ensure_schema()` (`CREATE TABLE IF NOT EXISTS …`) instead of an incremental, ledgered migration chain — the operator's "clean-slate, no migrations" reset for this component. It is dialect-portable (pg `TIMESTAMPTZ`/`now()` vs. sqlite `TEXT` ISO-8601/`datetime('now')`) via the same `queue_workflows.dialect` seam the rest of the engine uses. Two tables, both keyed by caller-supplied TEXT ids (no serial/autoincrement, so ids are backend-agnostic):

| Table | Purpose | Key columns |
|---|---|---|
| `bw_jobs` | The **shared** cpu/gpu queue for **all** projects | `job_id` (PK, TEXT), `project`, `resource` (`cpu`\|`gpu`), `status`, `priority`, `granted_worker`, `grant_expires_at`, `payload`, `result`, `error`, `created_at`, `updated_at` |
| `bw_workers` | Worker registry + liveness | `worker_id` (PK, TEXT), `project`, `resource`, `state` (`waiting`\|`running`\|`dead`), `last_seen`, `registered_at` |

`bw_jobs.status` moves `queued → granted → running → done | failed | killed`; a `killed` or lease-expired grant is re-queued by the health sweep (`sweep_unhealthy`), not left dangling. Indexes: `bw_jobs_claim_idx (resource, project, status, priority)` backs the grant pick, `bw_jobs_granted_worker_idx (granted_worker)` backs the dead-worker sweep's job lookup, `bw_workers_liveness_idx (state, last_seen)` backs the staleness scan.

`RESOURCES = frozenset({"cpu", "gpu"})` — the only two resource lanes today; host-defined lanes (mirroring the engine's `ingest_queues`) are a later pass.

### The API surfaces (`queue_workflows.broker_service` `__all__`)

**Broker side** (`orchestrator.py`) — called by the broker's own web service, or directly by anything that *is* the broker:

| Function | Signature intent |
|---|---|
| `ensure_schema()` | Idempotently create `bw_jobs`/`bw_workers`. Call at every broker startup. |
| `register_worker(worker_id, *, project, resource)` | Register or refresh a worker as `waiting`. Idempotent — a re-register refreshes `last_seen`/(project, resource) and revives a worker the health sweep had marked `dead`. |
| `worker_heartbeat(worker_id)` | Refresh `last_seen` (the liveness signal the health sweep reads). Returns `False` if unregistered. |
| `submit_job(*, project, resource, priority=100, payload=None)` | Enqueue a job onto the shared `resource` queue under `project`. Returns the generated job id. Lower `priority` = sooner. |
| `grant_next(worker_id, *, lease_s, capacity=None)` | **The grant decision.** Atomically: look up the worker's (project, resource); if `capacity` is set, deny when `granted`+`running` jobs on that resource *across all projects* already meet it (the cross-project arbitration point); else claim the next `queued` job for the worker's own project+resource (priority, then FIFO), stamp the grant + `lease_s` expiry, flip the worker to `running`. Returns the granted job dict or `None` (denied / no eligible work). |
| `has_permission(job_id, worker_id)` | `True` iff `worker_id` currently holds a *live* grant — `status IN ('granted','running')` and `grant_expires_at` in the future. Goes `False` the instant the broker revokes, reassigns, or the lease lapses. |
| `begin_job(job_id, worker_id)` | The permitted worker confirms it started: `granted → running`. `None` if the grant isn't the worker's / was already revoked. |
| `renew_grant(job_id, worker_id, *, lease_s)` | Extend the grant lease while the worker keeps running (its liveness token); also refreshes the worker heartbeat. `False` if the worker no longer holds it. |
| `complete_job(job_id, worker_id, *, result=None)` | `running → done`, idempotent (guarded by `status NOT IN ('done','failed','killed')`). |
| `fail_job(job_id, worker_id, *, error=None)` | `running → failed`, same idempotency guard. |
| `revoke(job_id, *, requeue=True, reason=None)` | The broker kills a job's grant at will. `requeue=True` (default) sends it back to `queued` — "kill and give permission for the next job" — `requeue=False` marks it `killed` (terminal). Either way the holder's permission is withdrawn and the holder freed back to `waiting`. |
| `sweep_unhealthy(*, stale_s)` | The health check: mark every worker whose heartbeat is older than `stale_s` as `dead`, then re-queue any `granted`/`running` job whose holder is now dead **or** whose grant lease lapsed. Returns the re-queued job ids. This is the sole recovery path for a worker that died mid-job. |
| `get_job(job_id)` / `get_worker(worker_id)` | Point reads. |
| `list_jobs(*, project=None, status=None, limit=100)` | Recent jobs, newest first, optional filters. Read-only, powers the panel + `/api/jobs`. |
| `list_workers()` | All registered workers, ordered by resource then id. |
| `queue_counts(*, project=None)` | Per `(resource, status)` counts — powers the panel KPI strip. |

**Client gate** (`permission.py`) — see next section.

### The permission gate — the 4-call client contract (`permission.py`)

A client project's worker does **not** claim work or manage its own LLM servers. It talks to the broker through exactly four calls, which wrap the orchestrator primitives above from the worker's point of view:

```python
job = ask_to_run(worker_id, project=..., resource="gpu", lease_s=30.0, capacity=4)
# job is None → denied (capacity full) or no eligible work; poll again later.

# ... node body + LLM calls run here, bracketed periodically by:
if not keep_permission(job["job_id"], worker_id, lease_s=30.0):
    ...  # broker revoked/reassigned — STOP IMMEDIATELY, do not call finish/abort

finish(job["job_id"], worker_id, result={...})   # success
# or
abort(job["job_id"], worker_id, error="...")      # failure
```

- **`ask_to_run(worker_id, *, project, resource, lease_s=30.0, capacity=None)`** — `register_worker` + `grant_next` in one call. Returns the granted job dict, or `None`.
- **`keep_permission(job_id, worker_id, *, lease_s=30.0)`** — renews the grant lease **and** confirms the broker hasn't revoked it (`renew_grant` then `has_permission`). Returns `False` the moment the grant was killed or reassigned — the worker must stop immediately. Call this between units of cooperative work, the same shape as the legacy engine's `LeaseRenewer` but with an added kill-check.
- **`finish(job_id, worker_id, *, result=None)`** — `begin_job` then `complete_job`.
- **`abort(job_id, worker_id, *, error=None)`** — `begin_job` then `fail_job`.

The node body and the LLM calls (to the broker-managed ollama/vLLM servers) run *between* `ask_to_run` and `finish`/`abort`, bracketed by periodic `keep_permission` checks — the v2 analogue of the legacy engine's lease-renew-plus-watchdog bracket, but observed from the client side against a broker decision rather than a self-managed lease.

### The worker runtime (`worker.py`)

`worker.py` is the runnable loop a client worker container executes — the concrete form of the permission gate above. It does **not** claim work or run its own LLM servers; it asks, runs the granted handler only while permitted, and stops the instant the broker revokes.

A handler is `fn(job, cancel) -> dict | None`: it does the work and returns a JSON-able result. `cancel` is a `threading.Event` set when the broker withdraws permission mid-run — a long-running handler should poll `cancel.is_set()` / `cancel.wait(...)` and bail cooperatively. Handlers are resolved by `job["payload"]["handler"]`, falling back to the job's `resource`, and registered via:

```python
queue_workflows.register_broker_handler("my-handler-key", my_fn)
```

(`config.broker_handlers: dict[str, Callable]`, empty by default on a submit-only app.)

`run_once(worker_id, *, project, resource, handlers=None, lease_s=30.0, capacity=None, poll_s=1.0, on_grant=None)` is one iteration: ask for permission; if granted, spawn a background **watcher thread** that calls `keep_permission` every `poll_s` seconds — renewing the lease *and* detecting revocation (setting a local `cancel` event the handler observes) — while the handler runs on the calling thread; then report the outcome:

- handler raises → `abort(..., error=str(exc))`;
- `cancel.is_set()` or permission was lost by the time the handler returns → **do not** call `finish`/`abort` at all; the broker already owns the job (it was re-queued for the next grant);
- otherwise → `finish(..., result=...)`.

`run_forever(...)` loops `run_once` until a `stop_event`, sleeping `idle_sleep_s` on an empty poll (denied / no work) — same idle-backoff shape as the rest of the engine's loops, with `sleep_fn` injectable for tests.

Console script:

```bash
queue-broker-worker --worker-id host-a-gpu-0 --project alpha --resource gpu \
                     --lease-s 30 --capacity 4 --db-backend pg --db-url-env BROKER_DSN
```

On boot it calls `ensure_schema()`, warns if no handlers are registered, then runs `run_forever`.

### The web service + operator panel (`web.py`)

`queue_workflows.broker_service.web` is a pure-stdlib `http.server` service (`ThreadingHTTPServer`) — server-rendered, no JavaScript, `Cache-Control: no-store` on every dynamic response — matching the house style of the engine's other operator surfaces. It has two faces on one server:

**1. The worker/broker JSON API** — the network form of the pull→grant model, driven by `POST` with a JSON body:

| Route | Body | Effect |
|---|---|---|
| `POST /api/submit` | `project, resource, priority?, payload?` | `submit_job(...)` → `{"job_id": ...}` |
| `POST /api/ask` | `worker_id, project, resource, lease_s?, capacity?` | `ask_to_run(...)` → `{"granted": bool, "job": ...}` |
| `POST /api/heartbeat` | `worker_id` | `worker_heartbeat(...)` → `{"ok": bool}` |
| `POST /api/renew` | `job_id, worker_id, lease_s?` | `keep_permission(...)` → `{"permitted": bool}` |
| `POST /api/finish` | `job_id, worker_id, result?` | `finish(...)` → `{"job": ...}` |
| `POST /api/abort` | `job_id, worker_id, error?` | `abort(...)` → `{"job": ...}` |
| `POST /api/revoke` | `job_id, requeue?, reason?` | `revoke(...)` → `{"job": ...}` — the **operator's kill switch** |

A missing required field returns `400 {"error": "missing field '<name>'"}` (a bare `KeyError` caught at the route level); any other exception is caught and surfaced as `500` rather than crashing the threaded server.

**2. The read-only operator panel** — `GET /`: the shared, project-labelled cpu/gpu queue (a per-resource KPI strip broken down by every `bw_jobs.status`), the worker fleet with liveness state, and a `?project=` filter rendered as clickable chips. Auto-refreshes every 5 seconds (`<meta http-equiv="refresh" content="5">`). Also:

- `GET /api/jobs`, `GET /api/workers`, `GET /api/snapshot` — the same data as JSON (`{"counts", "workers", "jobs"}` for `snapshot`), each honoring `?project=`.
- `GET /healthz` → `200 ok`, plaintext, unauthenticated, checked first (before the auth gate).
- `GET /favicon.svg` — the `broker_parrot` brand SVG (a geometric parrot-head mark), served publicly and pre-auth since a browser favicon fetch carries no bearer token and the mark isn't sensitive. The same SVG string is inlined in the panel header.

**Auth:** none by default — the service is meant to sit behind an internal ingress. If `QUEUE_WORKFLOWS_BROKER_WEB_TOKEN` is set in the environment, every route except `/healthz` and `/favicon.svg` requires `Authorization: Bearer <token>`, else `401` with `WWW-Authenticate: Bearer realm="broker"`. Unset (the default) leaves the service open — a single shared token is a minimal gate; per-worker identity, RBAC, and TLS are later passes.

Console script:

```bash
queue-broker-web --host 0.0.0.0 --port 8787 --db-backend pg --db-url-env BROKER_DSN
```

Calls `ensure_schema()` on boot, then serves the panel at `/` and the API at `/api/*` until interrupted. The SQLite backend is opened `check_same_thread=False` specifically so the threading server can share one connection safely — relevant if you point `queue-broker-web` at the sqlite default rather than a pg `BROKER_DSN`.

### Idempotency

Every terminal transition in `orchestrator.py` (`complete_job`, `fail_job`, and `revoke`'s `killed` branch) carries the same guard the legacy engine uses: `WHERE ... AND status NOT IN ('done', 'failed', 'killed')` (`_terminal`'s `_TERMINAL` tuple), plus `AND granted_worker = %(w)s` so a stale/duplicate call from a worker that no longer holds the grant is a no-op. A duplicate `finish`/`abort` delivery, or a race between a client's `finish` and the broker's `revoke`, resolves to whichever hits the row first; the loser gets back `None` rather than clobbering the winner's `result`/`error`.

All SQL in `broker_service` is written in pyformat with `queue_workflows.dialect` fragments (`now`, `future_seconds`, `past_seconds`, `creation_order`, `skip_locked`, `qualify_returning`) — the same seam the rest of the engine uses — so the identical code runs unchanged on Postgres and on the sqlite store (paramstyle translated at execute time). See [storage_backends](storage_backends.md) for the dialect seam's broader role.

### Relationship to the operator's stated direction

Part B is explicitly the beginning of a larger move: the operator wants the broker to become a **central orchestrator** — this web service and panel, a true cross-project capacity gate (already partially here via `grant_next`'s `capacity` parameter), and Kubernetes-managed LLM servers that client workers call into rather than each project running its own ollama/vLLM sidecar. That last piece — the k8s-managed LLM layer — is covered separately in [gpu_and_llm](gpu_and_llm.md); this document covers only the queue/grant control plane that layer will sit behind. Retiring the legacy engine (Part A) in favor of `broker_service` is a stated future pass, not something that has happened — both remain live, and nothing here requires choosing one over the other yet.

---

## See also

- [architecture.md](architecture.md) — the legacy engine's three process roles, lease/reclaim/watchdog model, and DAG dispatch (Part A's foundation).
- [configuration.md](configuration.md) — every `EngineConfig` field, including `project`, `db_namespace`, `db_backend`, and `broker_handlers`.
- [schema.md](schema.md) — full per-table column reference for both the ledgered `0001`–`0019` chain and `broker_service`'s `bw_jobs`/`bw_workers`.
- [storage_backends.md](storage_backends.md) — the pluggable `db_backend`/`StorageBackend` seam that both Part A and Part B's dialect layer build on.
- [worker_control.md](worker_control.md) — the legacy engine's operator ON/OFF plane (`worker_controls`, migration 0012/0019) — the Part A analogue of Part B's `revoke`.
- [watchdogs.md](watchdogs.md) — the legacy engine's in-process liveness watchdogs — the Part A analogue of Part B's `sweep_unhealthy`.
- [gpu_and_llm.md](gpu_and_llm.md) — GPU model cache, LLM backends, and the Kubernetes-managed LLM direction Part B is heading toward.
- [deployment.md](deployment.md) — running these processes/containers in practice.
