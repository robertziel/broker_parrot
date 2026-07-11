# 🗄️ Database schema & migration chain

*How `queue_workflows` lays out its Postgres/SQLite tables, why each migration exists, and the idempotency contracts the engine code leans on.*

This is the single schema reference — it replaces the old `broker_db_schema.md`
(Postgres) and `client_db_schema.md` (SQLite) docs, which duplicated most of
this content twice. One schema, two dialect renderings, documented once.

> Sibling docs: [architecture.md](architecture.md) (the three process roles
> that read/write this schema), [configuration.md](configuration.md) (the
> `db_backend` / `db_url_env` / `project` knobs), [storage_backends.md](storage_backends.md)
> (the non-relational `redis`/`mongodb` backends, which do **not** use this
> schema), [watchdogs.md](watchdogs.md) (the liveness model behind the lease +
> `watchdog_retries` columns), [worker_control.md](worker_control.md) (the
> `worker_controls` table in depth), [broker.md](broker.md) (the v2
> `bw_jobs`/`bw_workers` schema, which is deliberately **not** part of this
> chain).

## 1. Two migration chains, one database

The engine owns exactly one migration chain: `queue_workflows/migrations/NNNN_*.sql`
(Postgres) with a paired `NNNN_*.down.sql` for every forward file, shipped as
package data and tracked in a version ledger, `queue_schema_version` (one row
per applied migration, `(version, applied_at)`).

- **`db.bootstrap()`** applies every pending forward migration, in order,
  idempotently — each file uses `IF NOT EXISTS` / `IF EXISTS` forms so
  re-running the whole chain against an already-migrated DB is a no-op. On
  Postgres it first takes `pg_advisory_xact_lock(hashtext(version_table))`
  **before** even creating the ledger table, so many processes calling
  `bootstrap()` against the *same* database (every project's orchestrator
  booting against one shared broker after a new migration ships) is safe: the
  lock holder applies the pending chain and commits, and every waiter's
  post-lock read of the ledger then sees what the winner just committed.
- **`db.downgrade(to_version=...)`** reverses migrations above `to_version`,
  newest first, running each `NNNN_*.down.sql` and deleting its ledger row.
  Raises if a version has no down file.
- **Only the orchestrator bootstraps.** Claim workers and the scheduler never
  call `bootstrap()` — they call **`db.wait_for_schema(min_version)`**, which
  polls `current_schema_version()` until the ledger reaches `min_version` (or
  raises `TimeoutError` after 120 s). This avoids two processes racing the
  migration run.
- **Per-queue minimum version** — `claim_worker.py` maps `_NODE_REQUIRED_VERSION
  = 6` (the lease columns + `node_job_ready` trigger a `cpu`/`gpu` worker's
  claim loop needs) and `_INGEST_REQUIRED_VERSION = 8` (multi-tenant ingest —
  host-defined queue names + `args`) via `ClaimWorker.await_schema()`. A
  `cpu`/`gpu` worker blocks until version ≥ 6; an ingest-family worker (the
  host's `config.ingest_queues`) blocks until ≥ 8.
- **Worker-control is read-optional**, not gated at all: `get_worker_control()`
  swallows Postgres's `UndefinedTable` and treats a missing `worker_controls`
  table — i.e. a DB that predates migration 0012 — as "every worker is ON".
  So the ON/OFF control plane degrades gracefully on an old schema instead of
  crashing claim workers that don't otherwise need it.

**A host with its own domain tables runs a *second* chain** on the same
Postgres, via `db.bootstrap(migrations_dir=<host dir>, version_table=<host
ledger>)` / the matching `downgrade`/`wait_for_schema` overloads — "two ORMs,
two chains, one Postgres." The engine never sees or migrates a host's tables;
`run_store` treats `parcel_id` on `workflow_runs` as a plain opaque nullable
column for exactly this reason (see §4).

## 2. SQLite vs Postgres — one engine, two dialects

There are **two parallel migration directories**, applied by the same
`db.bootstrap()` machinery, selected by the active `db_backend`:

| | Postgres (`migrations/`) | SQLite (`migrations_sqlite/`) |
|---|---|---|
| Selected by | `configure(db_backend="pg")` / `QUEUE_WORKFLOWS_DB_BACKEND=pg` | default as of v1.0.0 |
| Timestamps | `TIMESTAMPTZ`, `now()` default | `TEXT` ISO-8601 UTC, `strftime('%Y-%m-%d %H:%M:%f', 'now')` default |
| JSON | `JSONB` | `TEXT` (`json.dumps`/`json.loads` at the boundary) |
| Arrays (`text[]`) | native `text[]`, GIN index on `known_models` | JSON-array `TEXT` (no array index) |
| Booleans | `boolean` | `INTEGER` 0/1 |
| Serial PKs | `BIGSERIAL` | `INTEGER PRIMARY KEY` (rowid alias) |
| Wake mechanism | `pg_notify` triggers (`node_job_ready_notify`, `ingest_job_ready_notify`, `worker_control_notify`, `worker_llm_config_notify`) | **none** — SQLite has no LISTEN/NOTIFY; every wake loop falls back to its safety-poll cadence |
| Claim concurrency | `SELECT … FOR UPDATE SKIP LOCKED` | *(clause dropped)* — a single-statement `UPDATE … WHERE id = (SELECT … LIMIT 1)` is already atomic under SQLite's serialized writers (WAL + `busy_timeout`) |

The two chains carry the **same version numbers, the same table names, the
same columns, the same composite primary keys, the same `project` tenant tag**
— they diverge only in DDL syntax and the presence/absence of triggers. A
migration is added to both directories in lockstep; skipping one leaves the
chains out of sync for whichever dialect was missed.

Two seams make one Python codebase run unmodified against either dialect:

1. **`dialect.py`** (`queue_workflows/dialect.py`) — a process-wide `Dialect`
   object (`PgDialect` / `SqliteDialect`, chosen from `config.db_backend`) that
   the runtime *hot-path* SQL (claim, lease renew, reclaim) is spliced from —
   fragments like "now", "future offset", "array membership", "table exists"
   render differently per dialect but the calling code is dialect-agnostic.
   Representative fragments:

   | Fragment | Postgres | SQLite |
   |---|---|---|
   | current time | `now()` | `datetime('now')` |
   | future/past offset | `now() + make_interval(secs => %s)` | `datetime('now', ('+' \|\| %s \|\| ' seconds'))` |
   | seconds since epoch | `EXTRACT(EPOCH FROM col)` | `CAST(strftime('%s', col) AS REAL)` |
   | FIFO tiebreak | `EXTRACT(EPOCH FROM a.created_at)` | `a.rowid` (monotonic, never ties) |
   | null-safe equality | `a IS NOT DISTINCT FROM b` | `a IS b` |
   | scalar minimum | `LEAST(...)` | `MIN(...)` |
   | array membership | `val = ANY(arr::text[])` | `val IN (SELECT value FROM json_each(arr))` |
   | table exists | `to_regclass('public.' \|\| %s)` | `SELECT name FROM sqlite_master WHERE type='table' AND name = %s` |

2. **`db.py`**'s string-literal-aware translator (`_translate_sql_for_sqlite`)
   — for SQL that's still written directly with Postgres syntax (migrations,
   some call sites), a regex pipeline rewrites `FOR UPDATE [SKIP LOCKED]` away,
   fuses `now() ± make_interval(...)` into a single `datetime('now', ...)`
   call, strips `::cast` suffixes, maps `LEAST`/`GREATEST` → `MIN`/`MAX`, and
   converts the pyformat paramstyle (`%s` / `%(name)s`) to SQLite's (`?` /
   `:name`) — string literals are protected from these rewrites so e.g.
   `strftime('%s', …)` inside a literal survives untouched. A row-factory
   (`_sqlite_row_to_dict`) then restores psycopg-equivalent Python types on
   read: known JSON-object/array columns parse through `json.loads`, known
   timestamp columns parse into aware UTC `datetime`s, known boolean columns
   coerce `0`/`1` → `bool`.

Selecting the backend is a `configure()` concern, not a schema concern — see
[configuration.md](configuration.md) for `db_backend` / `db_url_env` defaults,
and [storage_backends.md](storage_backends.md) for the `redis`/`mongodb`
backends, which bypass this schema entirely via the `StorageBackend` SPI.

## 3. The chain, migration by migration (0001–0019)

Every migration below applies to *both* dialect directories in lockstep
(dialect differences per §2); the "why" is dialect-independent.

| # | Adds | Why |
|---|---|---|
| **0001** `queue_runs` | `workflow_runs`, `workflow_run_files` | The queue's substrate. `workflow_runs` is what a worker claims (`status='queued'` ordered by `priority, queued_at`); `parcel_id` is deliberately a plain nullable column, not an FK — the engine drops the host's `parcels` FK so the schema stands alone on a parcel-less DB. `workflow_run_files` is the per-run output-artifact manifest. |
| **0002** `node_jobs` | `workflow_node_jobs` | The node-per-job queue — the engine dispatches one DAG *node* at a time (not a whole pipeline step), landing on `cpu` (short-lived workers) or `gpu` (long-lived, warm-model-cache workers). Ships in its final consolidated shape (folds in what the original ai_leads chain reached across several later migrations: `pipeline_name`, `celery_task_id`, `resolved_inputs`, `host_label`, `input_spec`) rather than as discrete column-adds, since this is a fresh engine chain. |
| **0003** `input_submissions` | `workflow_input_submissions` | A durable store for user-submitted values on an `awaiting_input` node. Replaces a transient `pg_notify('input_submitted', …)` channel that dropped submissions across a listener restart; the `InputListener` polls `pending` rows instead. Ships with a partial-unique index on `(run_id, node_id) WHERE status IN ('pending','processing')` so only *in-flight* submissions collide — legitimate resubmission across retries doesn't 409. |
| **0004** `dispatch_events` | `workflow_dispatch_events` | The durable dispatcher **outbox**. A worker's terminal state write and its dispatch-event row land in one transaction, so the orchestrator's fan-out to downstream nodes is retryable and never synchronously coupled to the worker process. |
| **0005** `worker_heartbeats` | `worker_heartbeats` | The per-worker fleet capacity ledger (observed state). Each claim worker upserts `(host_label, queue)` and refreshes `last_seen` every ~10 s; consumers `SUM(concurrency)` over fresh rows for the "GPU 1/N" capacity gauge. Ships with `current_model` (warm-model affinity hint) and `known_models` (capability advertisement) already folded in. |
| **0006** `pg_queue_lease` | `workflow_node_jobs.claimed_by`/`lease_expires_at` + `node_job_ready` trigger (pg only) | The lease + wake primitives that make the queue *live*: a lapsed lease (no renewal) is the sole signal a reclaim sweep needs to re-queue an orphaned `running` row, and the trigger fires `pg_notify('node_job_ready', queue)` **inside the writer's transaction** whenever a row becomes `queued` — so an idle claim worker blocks on `LISTEN` with no "row queued but no wake" window. (SQLite has no trigger counterpart — it polls.) |
| **0007** `ingest_jobs` | `ingest_jobs` table + `ingest_job_ready` trigger (pg only) | The **second job family** — standalone periodic/parametrised work with no DAG, no parent run, no dispatch outbox. Carries the same claim/lease column shape as `workflow_node_jobs` so the lease-renew/reclaim machinery is reused, plus its own wake trigger. Relaxes the original ai_leads task-name CHECK — the host validates `task_name` against its registered dispatch map before enqueue, not a DB constraint, so the table is reusable by any project's periodic work. |
| **0008** `multitenant_ingest` | `ingest_jobs.args JSONB`; drops the `fetch`/`load` queue CHECK on `ingest_jobs` and the `cpu`/`gpu` CHECK on `worker_heartbeats` | Lets a **second consumer** (a non-DAG forecast service) route its own ingest queue names and carry per-job parameters, without forking the schema. The queue-name allow-list moves from a DB `CHECK` to host-side validation (`node_queue.enqueue_ingest_job`), mirroring what 0007 already did for `task_name`. Fully additive/backward-compatible — `ai_leads`' `fetch`/`load` no-args enqueues keep working unchanged. |
| **0009** `worker_heartbeats_dead_flag` | `worker_heartbeats.last_flagged_dead_at` | The last-resort recovery marker: the orchestrator (a separate, GIL-independent process) stamps this when a worker's heartbeat goes stale **while it still owns a `running` job** — a GPU hardware hang can wedge a worker process even after the *job* is safely reclaimed by the lease sweep, and nothing else flags the dead *process* for a host-supervisor to bounce. Nullable, cleared by the next successful heartbeat. |
| **0010** `node_job_watchdog_retries` | `workflow_node_jobs.watchdog_retries INTEGER NOT NULL DEFAULT 0` | Changes watchdog-trip policy from "fail the node (and the whole run)" to "re-queue and retry the node" for a transient wedge. This counter — deliberately **not** `workflow_dispatch_events.attempts`, a different retry budget on a different table — caps the re-queue loop; once it reaches `AI_LEADS_WATCHDOG_MAX_RETRIES` (default 3) the watchdog falls back to the old mark-failed path. |
| **0011** `node_events` | `workflow_node_events` (append-only) | `workflow_node_jobs` is one *mutable* row per `(run_id, node_id)` — a watchdog re-queue overwrites `claimed_by`/timing and only bumps `watchdog_retries`, so the prior attempt's forensics (which worker, how long, what tripped) are lost the instant the next attempt is claimed. This table is the durable, append-only per-attempt event log (`claimed`, `model_load_*`, `progress_beat`, `stall_*`, `gpu_health_trip`, `budget_trip`, `requeued`, `reassigned`, terminal states, …); `attempt` = `watchdog_retries` at emit time ties one node's tries together. |
| **0012** `worker_controls` | `worker_controls` table + `worker_control` NOTIFY trigger (pg only) | The operator worker ON/OFF control plane — **desired** state, deliberately a *separate* table from the *observed* `worker_heartbeats`, because an OFF state must persist precisely while the worker is not beating (exactly when its heartbeat row would age out of the freshness window). Keyed `(host_label, queue)` — same identity as the heartbeat and the claim's `claimed_by`/`queue` — because one host can run several per-queue workers. |
| **0013** `worker_controls_llm` | `worker_controls.llm_server_type`/`llm_parallelism`/`vllm_idle_ttl_s` + a second `worker_llm_config_changed` NOTIFY trigger (pg only) | Per-machine LLM-server config, operator-set, living next to the ON/OFF switch since it's the same desired-state row read by the same worker. A dedicated NOTIFY channel (payload `host\|queue`, not `host:queue`) so an LLM-config edit isn't mistaken for an ON/OFF flip by the hard-stop watcher; the trigger stays quiet on an UPDATE touching none of the three LLM columns. |
| **0014** `worker_heartbeats_llm_servers` | `worker_heartbeats.llm_servers_available text[]` | Observed LLM-server capability — analogous to `known_models` but for the LLM sidecar type(s) a machine can actually run (e.g. `{ollama}` universally, `{ollama,vllm}` on an NVIDIA host with the vllm sidecar). Gates the operator control-plane UI so a host that can't run vllm can't be toggled onto it. |
| **0015** `capacity_aware_assignment` | `worker_heartbeats.vram_total_mb`/`fits_models`; `workflow_node_jobs.unassignable_at`/`unassignable_reason`; adds `'unassignable'` to the 0011 `event_type` CHECK | Closes a gap where *any* GPU worker could claim *any* GPU model job regardless of whether it fit in VRAM — a too-big model got claimed anyway and OOM'd at load, or (if no machine in the fleet fits) sat queued forever with no visible reason. `fits_models` pushes the fit computation to the worker (which holds the model registry); `unassignable_at`/`_reason` is a red-flag stamp on a still-`queued` row, not a new terminal status — the condition is transient and clears itself if a bigger machine comes online. |
| **0016** `node_priority_flag` | `workflow_node_jobs.is_priority BOOLEAN NOT NULL DEFAULT FALSE` | An operator-settable "run next" flag on a queued node — sorted **first** in the claim `ORDER BY`, ahead of the integer `priority` band and (on GPU) ahead of the warm-model affinity tiebreak, so a flagged cold-model node preempts a warm one (the model reload is the accepted cost). |
| **0017** `project_tenant` | `project TEXT NOT NULL DEFAULT ''` on `workflow_runs`, `workflow_node_jobs`, `ingest_jobs`, `worker_heartbeats`; widens `worker_heartbeats`' PK to `(host_label, queue, project)`; project-aware claim indexes | Pools **multiple projects onto one shared broker Postgres** (previously one Postgres per project was assumed). `DEFAULT ''` keeps a single-tenant deploy byte-compatible — every row is `''`, the claim filter `project=''` matches everything, so behavior is unchanged with zero host wiring. **Breaking for raw-SQL heartbeat writers**: the 2-column `ON CONFLICT (host_label, queue)` no longer matches any constraint — callers must move to `node_queue.upsert_worker_heartbeat`. |
| **0018** `node_events_project` | `workflow_node_events.project TEXT NOT NULL DEFAULT ''` + `(project, event_type, created_at DESC)` index | 0017 tagged runs/node-jobs/ingest-jobs/heartbeats with `project` but missed the forensic event log — this closes the gap so an operator "Errors" view stays project-scoped without a join, matching the rest of the broker. |
| **0019** `worker_controls_project` | `worker_controls.project TEXT NOT NULL DEFAULT ''`; re-keys its PK to `(host_label, queue, project)` | The control-plane twin of 0017: on a shared broker, `host_label` is no longer globally unique (two projects can run a worker on the same machine+queue), so a 2-column `worker_controls` PK let one project's operator ON/OFF (or LLM-config write) silently hit the *other* project's worker. NOTIFY payloads are deliberately left unchanged (`host:queue` / `host\|queue`, no tenant segment) — both watchers treat the NOTIFY as a bare wake and re-read their own project's row, so a spurious cross-tenant wake is harmless. **Breaking for raw-SQL control writers** for the same reason as 0017. |

`0017`'s and `0019`'s down-migrations are the two genuinely **destructive**
reversals in the chain: collapsing the 3-column PK back to 2 columns only
works on a DB that never actually went multi-tenant (`0017`'s down throws if
two projects share a `(host_label, queue)`; `0019`'s down `DELETE`s every
non-`''`-project `worker_controls` row outright, since control rows are cheap
operator state, not queue work). Every other down-migration is a clean,
lossless reversal (aside from the explicitly lossy `0008` down, which drops
`args`).

## 4. Per-table reference (load-bearing tables)

Column tables below are the **Postgres** rendering; where SQLite's type or
default differs, it's noted inline (full type-mapping table in §2). All
`created_at`/timestamp defaults are `now()` on pg / the `strftime(...)` UTC
literal on SQLite — abbreviated to `now()` below for both.

### `workflow_runs` (0001, tenant-tagged 0017)

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | text | no | | PK |
| `parcel_id` | text | yes | | opaque host tag; engine drops the `parcels` FK |
| `workflow_name` | text | no | | which workflow/pipeline |
| `status` | text | no | | `queued`/`running`/`completed`/`failed`/`cancelled`/… |
| `priority` | smallint | no | `100` | lower = sooner |
| `current_step_id` | text | yes | | progress pointer |
| `progress_pct` | real | no | `0.0` | |
| `steps_done` | jsonb | no | `'[]'` | |
| `context` | jsonb | no | `'{}'` | accumulated run context |
| `input_spec` | jsonb | yes | | run-level awaiting-input spec |
| `error` | text | yes | | |
| `out_dir` | text | yes | | |
| `mode` | text | no | `'step'` | `CHECK (mode IN ('step','node'))` |
| `resume_count` | smallint | no | `0` | |
| `created_at`/`updated_at` | timestamptz | no | `now()` | |
| `queued_at`/`started_at`/`finished_at` | timestamptz | yes | | |
| `project` | text | no | `''` | tenant tag (0017) |

PK `(id)`. Indexes: `workflow_runs_claim_idx (priority, queued_at) WHERE status='queued'`,
`workflow_runs_status_idx (status)`, `workflow_runs_project_idx (project, status)`,
`workflow_runs_parcel_created_idx (parcel_id, created_at DESC)`. No triggers.

### `workflow_node_jobs` (0002, lease 0006, watchdog 0010, capacity 0015, priority 0016, tenant 0017)

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | text | no | | PK |
| `run_id` | text | no | | FK → `workflow_runs(id)` `ON DELETE CASCADE` |
| `node_id` | text | no | | logical id inside the workflow JSON |
| `node_module` | text | no | | importable name via the host node-module hook |
| `pipeline_name` | text | yes | | parent pipeline ref |
| `queue` | text | no | | `CHECK (queue IN ('cpu','gpu'))` |
| `required_model` | text | yes | | `CHECK (queue='gpu' OR required_model IS NULL)` |
| `status` | text | no | | `queued`/`running`/`completed`/`failed`/`cancelled`/`awaiting_input`/`skipped` |
| `priority` | smallint | no | `100` | |
| `worker_lane` | smallint | yes | | |
| `inputs` | jsonb | no | `'{}'` | may hold `$from`/`$value` refs |
| `resolved_inputs` | jsonb | yes | | execute-time `$from` snapshot |
| `input_spec` | jsonb | yes | | per-job awaiting-input spec |
| `context_delta` | jsonb | no | `'{}'` | merged into run context on success |
| `host_label` | text | yes | | claiming host (`COALESCE`d in at terminal) |
| `celery_task_id` | text | yes | | legacy, unused |
| `error` | text | yes | | |
| `vm_rss_mb_peak` | integer | yes | | |
| `seconds` | double precision | yes | | |
| `created_at`/`started_at`/`finished_at` | timestamptz | | `now()` on `created_at` | |
| `claimed_by` | text | yes | | lease owner (0006) |
| `lease_expires_at` | timestamptz | yes | | reclaim predicate (0006) |
| `watchdog_retries` | integer | no | `0` | re-queue counter (0010) |
| `unassignable_at`/`unassignable_reason` | timestamptz/text | yes | | capacity red flag (0015) |
| `is_priority` | boolean | no | `false` | "run next" (0016) |
| `project` | text | no | `''` | tenant tag (0017) |

PK `(id)`; **UNIQUE** `(run_id, node_id)`. Indexes: `workflow_node_jobs_claim_idx
(queue, priority, created_at) WHERE status='queued'`, `..._project_claim_idx
(queue, project, priority, created_at) WHERE status='queued'`, `..._lease_idx
(lease_expires_at) WHERE status='running'`, `..._model_idx (required_model)
WHERE queue='gpu' AND status='queued'` (warm-model affinity ordering),
`..._unassignable_idx (queue, status) WHERE required_model IS NOT NULL`, plus
`run_idx`/`status_idx`/`pipeline_idx`/`host_label_idx`/`celery_task_id_idx`.
Trigger (pg only): `node_job_ready_notify` `AFTER INSERT OR UPDATE OF status`
→ `pg_notify('node_job_ready', NEW.queue)` when `NEW.status='queued'`.

### `workflow_dispatch_events` (0004)

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | bigserial | no | | PK |
| `run_id` | text | no | | FK → `workflow_runs(id)` CASCADE |
| `node_id` | text | no | | terminated node |
| `kind` | text | no | | `CHECK (kind IN ('completed','failed','awaiting_input'))` |
| `processed_at` | timestamptz | yes | | NULL = unprocessed (the drain predicate) |
| `error` | text | yes | | last callback error |
| `attempts` | smallint | no | `0` | outbox-drain retry counter |
| `created_at` | timestamptz | no | `now()` | drain order |

PK `(id)`. Index: `..._unprocessed_idx (created_at) WHERE processed_at IS NULL`.
No triggers — the orchestrator polls this outbox itself.

### `worker_heartbeats` (0005, dead-flag 0009, LLM caps 0014, capacity 0015, tenant 0017)

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `host_label` | text | no | | PK part |
| `queue` | text | no | | PK part; `cpu`/`gpu`/ingest queue (CHECK dropped 0008) |
| `concurrency` | integer | no | | this worker's contribution (1 by contract) |
| `current_model` | text | yes | | GPU warm-model affinity hint |
| `known_models` | text[] | no | `'{}'` | registered model ids advertised |
| `last_seen` | timestamptz | no | `now()` | freshness filter (`> now() - 30s`) |
| `last_flagged_dead_at` | timestamptz | yes | | orchestrator dead-worker flag (0009) |
| `llm_servers_available` | text[] | no | `'{ollama}'` | observed LLM capability (0014) |
| `vram_total_mb` | integer | yes | | total GPU VRAM MB (0015) |
| `fits_models` | text[] | no | `'{}'` | model ids that fit this VRAM (0015) |
| `project` | text | no | `''` | tenant tag, PK part (0017) |

PK `(host_label, queue, project)` — **widened from 2 to 3 columns in 0017**.
Indexes: `last_seen_idx (last_seen)`, `known_models_gin` GIN (pg only),
`flagged_dead_idx (last_flagged_dead_at) WHERE ... IS NOT NULL`. No triggers.
**All writes must go through `node_queue.upsert_worker_heartbeat`** — a raw
`ON CONFLICT (host_label, queue)` no longer matches any constraint post-0017.

### `ingest_jobs` (0007, multitenant 0008, tenant 0017)

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | text | no | | PK |
| `task_name` | text | no | | host-validated, not a DB CHECK |
| `queue` | text | no | | host-defined; CHECK dropped in 0008 |
| `reason` | text | no | `'tick'` | `tick`/`boot`/`manual` |
| `status` | text | no | `'queued'` | `queued`/`running`/`completed`/`failed`/`cancelled` |
| `priority` | smallint | no | `100` | |
| `result` | jsonb | yes | | |
| `error` | text | yes | | |
| `seconds` | double precision | yes | | |
| `claimed_by` | text | yes | | |
| `lease_expires_at` | timestamptz | yes | | |
| `created_at`/`started_at`/`finished_at` | timestamptz | | `now()` on `created_at` | |
| `args` | jsonb | no | `'{}'` | per-job params for a parametrised task (0008) |
| `project` | text | no | `''` | tenant tag (0017) |

PK `(id)`. Indexes: `claim_idx (queue, priority, created_at) WHERE status='queued'`,
`project_claim_idx (queue, project, priority, created_at) WHERE status='queued'`,
`lease_idx (lease_expires_at) WHERE status='running'`. Trigger (pg only):
`ingest_job_ready_notify` → `pg_notify('ingest_job_ready', NEW.queue)`.

### `workflow_node_events` (0011, capacity vocabulary 0015, tenant 0018)

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | bigserial | no | | PK |
| `run_id` | text | no | | FK → `workflow_runs(id)` CASCADE |
| `node_id` | text | no | | |
| `job_id` | text | yes | | `workflow_node_jobs.id` at emit (survives row churn) |
| `attempt` | smallint | no | `0` | `= watchdog_retries` at emit — cross-attempt key |
| `event_type` | text | no | | `CHECK` — `claimed`, `model_load_start`, `model_load_done`, `progress_beat`, `stall_suspected`, `stall_trip`, `gpu_health_trip`, `budget_trip`, `requeued`, `reassigned`, `lease_renew`, `completed`, `failed`, `cancelled`, `error`, `unassignable` (0015) |
| `host_label`/`queue`/`model` | text | yes | | context at emit |
| `elapsed_s` | double precision | yes | | seconds in this attempt |
| `error` | text | yes | | trip reason, truncated |
| `detail` | jsonb | no | `'{}'` | free-form trip metrics |
| `created_at` | timestamptz | no | `now()` | |
| `project` | text | no | `''` | tenant tag (0018) |

PK `(id)`. Indexes: `node_idx (run_id, node_id, created_at)` (the hot per-node
timeline read), `created_idx (created_at)` (the `prune_node_events` retention
sweep), `project_kind_idx (project, event_type, created_at DESC)` (0018 — the
project-scoped error console). **Append-only — no UPDATE path**, so it adds no
new mutation invariant (§5). No triggers; writers insert directly.

### `worker_controls` (0012, LLM config 0013, tenant 0019)

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `host_label` | text | no | | PK part |
| `queue` | text | no | | PK part |
| `desired_state` | text | no | `'on'` | `CHECK (desired_state IN ('on','off'))` |
| `stop_policy` | text | no | `'hard'` | free-form (no CHECK); validated against the in-code `STOP_POLICIES` registry |
| `requested_by` | text | yes | | provenance, informational |
| `updated_at` | timestamptz | no | `now()` | |
| `llm_server_type` | text | no | `'ollama'` | `CHECK IN ('ollama','vllm')` (0013) |
| `llm_parallelism` | integer | no | `1` | sidecar concurrency, `CHECK >= 1` (0013) |
| `vllm_idle_ttl_s` | integer | no | `60` | `CHECK >= 0`; 0 disables the idle reap (0013) |
| `project` | text | no | `''` | tenant tag, PK part (0019) |

PK `(host_label, queue, project)` — **widened from 2 to 3 columns in 0019**.
No secondary indexes beyond the PK. Triggers (pg only): `worker_control_notify`
`AFTER INSERT OR UPDATE` → `pg_notify('worker_control', host_label||':'||queue)`
on every write; `worker_llm_config_notify` → `pg_notify('worker_llm_config_changed',
host_label||'|'||queue)`, silent unless an LLM column actually changed. See
[worker_control.md](worker_control.md) for the watcher that reads this table.

## 5. Idempotency contracts

Every terminal-state writer shares one shape, and it is load-bearing — do not
drop the `WHERE` clause when adding a new state transition:

```sql
UPDATE workflow_node_jobs
SET status = 'completed', ...
WHERE id = %s
  AND status NOT IN ('completed', 'failed', 'cancelled')
RETURNING *
```

`node_queue.mark_completed` / `mark_failed` / `mark_awaiting_input` (plus the
`_in_txn` variants used from inside the dispatch-outbox transaction) and their
ingest twins `mark_ingest_completed` / `mark_ingest_failed` all follow this
pattern: the `WHERE status NOT IN (...)` makes a duplicate delivery or a
claim-race loser a safe no-op — the function returns `None` when the row was
already terminal, rather than clobbering a freshly-finalized `context_delta`
(or `result`/`error`) with whatever a stray second call computed (often empty).
Any new state transition follows the same shape: gate the `UPDATE` on the
non-terminal set, `RETURNING *`, `None` on a no-op hit.

JSON payloads are pre-validated with `json.dumps(...)` **before** the state
mutation runs, so a bad payload fails fast rather than leaving the row in a
half-written state.

**The one deliberate exception is `workflow_node_events`** (§4) — append-only,
no `UPDATE` path at all, so it introduces no new mutation invariant to
preserve. Its terminal and `requeued` rows instead ride the *same transaction*
as the state change they record (the same outbox-atomicity pattern as
`workflow_dispatch_events`); every other event type is written best-effort, on
its own connection, swallowing failures, so an event-log blip can never fail
the load-bearing claim/terminal/watchdog path.

## 6. What's outside this chain: the v2 broker_service schema

`queue_workflows/broker_service/schema.py` defines a **second, separate, and
deliberately un-ledgered** schema for the v2 broker orchestration core:
`bw_jobs` (the shared cpu/gpu queue across all projects, `status`: `queued →
granted → running → done | failed | killed`) and `bw_workers` (the worker
registry + liveness, `state`: `waiting` / `running` / `dead`). It coexists in
the same relational store (pg or sqlite) via the same `db.connection()` /
`dialect.py` seam this document's chain uses, but it is created by a single
idempotent `ensure_schema()` (`CREATE TABLE IF NOT EXISTS`) — no `NNNN_*.sql`
files, no version ledger, no down-migrations. This is a deliberate "clean
slate, no migrations" design choice for the v2 control plane, not an
oversight; it coexists with the legacy 0001–0019 chain so a later pass can
retire the old engine tables without a data migration. See
[broker.md](broker.md) for the full v2 design and API.
