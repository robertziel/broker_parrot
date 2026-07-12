# 🚀 Deployment

*Running the engine in production: which process runs which console script, the recommended one-container/N-process claim lane, migrations at boot, backend selection, a worked fleet-cutover checklist, and how to run the test suite.*

See [architecture](architecture.md) for the process-role model this doc operationalizes, [schema](schema.md) for the migration chain, [storage_backends](storage_backends.md) for the `db_backend` seam, [configuration](configuration.md) for the full config-hook reference, [worker_control](worker_control.md) for the operator ON/OFF plane, [broker](broker.md) for the shared-broker/multitenant model, and [gpu_and_llm](gpu_and_llm.md) for GPU pool + LLM backend specifics.

## 1. Process roles → console scripts

Every process role in [architecture](architecture.md) ships as a `pip`-installed console script (`[project.scripts]` in `pyproject.toml`) and is equally runnable as `python -m queue_workflows.<module>` — useful when a host wraps the entry point instead of shelling out to the installed script.

| Script | Module | Role |
|---|---|---|
| `queue-orchestrator` | `queue_workflows.orchestrator:main` | Bootstraps migrations (`db.bootstrap`) then runs `NodePool` — dispatch loop, outbox drain, lease-reclaim sweeps, `InputListener`. **The only process that bootstraps.** |
| `queue-claim-worker --queue=<name>` | `queue_workflows.claim_worker:main` | One worker process, concurrency-1 by contract. `--queue` ∈ `{cpu, gpu}` (DAG node-jobs) ∪ `config.ingest_queues` (default `{fetch, load}`). |
| `queue-scheduler` | `queue_workflows.scheduler:main` | PG-native ingest ticker — sleeps to the next scheduled minute, enqueues `ingest_jobs` rows. |
| `queue-worker-control --queue=<name> --on\|--off` | `queue_workflows.worker_control:main` | Operator CLI for the ON/OFF control plane (migration 0012) — flips a `(host, queue)` `worker_controls` row; see [worker_control](worker_control.md). |
| `queue-broker` | `queue_workflows.broker:main` | Stand up / inspect the **shared broker** DB (migration 0017 multitenant model) — bootstrap once, then `--status` to view every project's queue depth. See [broker](broker.md). |
| `queue-broker-web` | `queue_workflows.broker_service.web:main` | v2 broker control-plane HTTP face: the worker/broker JSON API (`submit`/`ask`/`heartbeat`/`renew`/`finish`/`abort`/`revoke`) plus the read-only operator panel. |
| `queue-broker-worker` | `queue_workflows.broker_service.worker:main` | v2 worker that runs *behind the broker's permission gate* — asks for grants instead of claiming rows itself, stops the instant the broker revokes. |

The first five scripts are the **v1 client model** (a project runs its own orchestrator/workers/scheduler against a DB it owns, optionally a shared one via `queue-broker`). `queue-broker-web` + `queue-broker-worker` are the **v2 broker service** (pull→grant over HTTP) — see [broker](broker.md) for when to reach for which model.

## 2. Recommended default: one container, N claim PROCESSES per lane

**The recommended default deployment for a claim lane is ONE container running N concurrency-1 `queue-claim-worker` processes — not N docker replicas.**

### Why

A claim worker is one-job-per-process by contract (`architecture.md`: "scale by adding workers, not threads"). The naive way to get N-way parallelism is `deploy.replicas: N` — N containers, each running one worker process. But an idle claim worker is tiny (tens of MiB), so N replicas buys N containers' worth of per-container overhead (extra namespaces, one shim per container, scheduler churn, a cluttered `docker ps`) for **zero** additional throughput: each process still claims via `SELECT … FOR UPDATE SKIP LOCKED`, so what matters is the process count, not the container count. Running the N processes inside ONE container keeps identical N-way parallelism (N processes = N concurrent jobs, each independently claiming) while dropping the per-container tax.

### Panel display: advertise N, don't let it default to 1

All N processes share the container's `host_label` (e.g. `claim-cpu`), so the operator panel sees **one** `worker_heartbeats` row and **one** ON/OFF toggle controls all N of them — this is a feature, not a gap: it's exactly the ON/OFF granularity worker_control.md documents. But the panel sums `concurrency` over that row, and the engine's default only advertises capacity `1` for a cpu worker (only GPU advertises real parallelism), so an unmodified lane would read `CPU x/1` even with 30 processes claiming behind it. To read `CPU x/N`, a host that wants accurate panel capacity sets the process count on that heartbeat's `concurrency_fn` (internal to `HeartbeatEmitter`, not a `ClaimWorker.__init__` kwarg — wire it defensively, falling back to the stock entry point on any error). cpu claim itself is SKIP LOCKED and is **never** gated by this number — it's display-only.

Keep the `host_label` **shared** across the N processes in one container. Giving each process a unique label would also make the panel count N workers, but it would break the single ON/OFF toggle, since `worker_control` keys off an exact `host_label`.

### Reference supervisor

`deploy/run_claim_lane.sh <queue>` is the reference entrypoint: it spawns N processes of `$CLAIM_MODULE` (defaults to `queue_workflows.claim_worker`; a consumer typically points `CLAIM_MODULE` at its own thin wrapper), restarts any process that exits (replacing docker's per-container `restart: unless-stopped`), and forwards `SIGTERM` to every child so in-flight leases get a chance to drain within the compose `stop_grace_period`. If the whole container dies, docker's own restart policy covers that layer. N is read per queue:

| Queue | Env var | Default |
|---|---|---|
| `cpu` | `LM_WORKFLOW_CPU_WORKERS` | 30 |
| `gpu` | `LM_WORKFLOW_GPU_WORKERS` | 1 |
| anything else | `LM_WORKFLOW_LANE_WORKERS` | 1 |

```yaml
# docker-compose.yml sketch
workers-claim-cpu:
  image: your-image
  command: ["bash", "/app/deploy/run_claim_lane.sh", "cpu"]
  restart: unless-stopped
  stop_grace_period: 60s
  environment:
    - LM_WORKFLOW_CPU_WORKERS=32   # process count — was the replica count
  # NO deploy.replicas — ONE container.
```

GPU lanes are typically already single-process (one persistent warm `ModelCache` per host — see [gpu_and_llm](gpu_and_llm.md)), so in practice this mainly consolidates the `cpu` lane, where N is largest and the per-container overhead multiplies most.

## 3. Migrations at deploy time

Only the **orchestrator** calls `db.bootstrap()`. On Postgres it's concurrency-safe: it takes a `pg_advisory_xact_lock` keyed on the version table **before** even `CREATE TABLE IF NOT EXISTS`, so if many orchestrators (e.g. every project's orchestrator, all pointed at one shared broker) boot at once against the same DB, the lock serializes them — the winner applies the pending chain and commits, every waiter's post-lock read then sees the winner's work already done and applies nothing. On SQLite the advisory lock is a no-op; that's fine because SQLite is a single-machine deploy where, by convention, only the orchestrator ever calls `bootstrap()` — claim workers and the scheduler never do.

Every other process — claim workers, the scheduler — calls `db.wait_for_schema(min_version)` instead of bootstrapping, and **blocks** (polling `current_schema_version`, default 120 s timeout) until the ledger reaches the version that queue depends on, rather than racing the orchestrator's migration run. `min_version` is per queue-family: DAG node-jobs (`cpu`/`gpu`) require schema ≥ 6 (`_NODE_REQUIRED_VERSION`); ingest queues require schema ≥ 8 (`_INGEST_REQUIRED_VERSION`, migration 0008's multi-tenant ingest). This means the current chain (0001‥0019 — see [schema](schema.md)) is far ahead of what a worker actually gates on: worker-control (0012) is read-optional and a pre-0012 DB is treated as all-ON, so a fleet doesn't need to be fully caught up to schema HEAD for workers to run — only orchestrators applying new migrations need to.

A host with its own domain tables runs a **second** chain against its own ledger, by calling `db.bootstrap(migrations_dir=..., version_table=...)` with its own directory and version-table name — "two ORMs / two chains, one Postgres." The engine never sees or touches the host's chain.

**Deploy-time checklist:**
1. Start (or let your orchestrator container start) `queue-orchestrator` first, or at least don't gate other containers' health checks on anything before it — `bootstrap()` racing a fresh empty DB is safe, but a claim worker calling `wait_for_schema` before *any* orchestrator has ever run will simply block up to its timeout and then raise `TimeoutError`.
2. Claim workers / scheduler containers can start in parallel with the orchestrator — they'll block on `wait_for_schema` until it catches up, then proceed. No explicit ordering/`depends_on: service_healthy` dance is required for correctness (only for faster convergence).
3. Rolling a new migration into a live fleet: ship the new orchestrator image first (or alongside), let it apply the migration, then roll workers — they'll simply wait a little longer for the version bump if they land first.

## 4. Backend selection at deploy

As of v1.0.0 the library default is `db_backend="sqlite"` — the friendliest zero-config default for a reusable library, and the deliberate breaking change from the pre-1.0 `pg` default. A Postgres fleet must opt in, either in code:

```python
queue_workflows.configure(db_backend="pg", db_url_env="QUEUE_WORKFLOWS_DB_URL")
```

or via the env knob, which also reaches every standalone console script above (they have no host `configure()` call to hook):

```bash
export QUEUE_WORKFLOWS_DB_BACKEND=pg
```

`queue-broker` and the v2 `queue-broker-web` / `queue-broker-worker` scripts additionally take `--db-backend` / `--db-url-env` flags that override the env for that invocation. `redis`/`mongodb` are also selectable (`configure(db_backend="redis"|"mongodb")`) but are an **additive** durable-queue SPI, not a relational engine backend — selecting one does not re-home the orchestrator/dispatcher (see [storage_backends](storage_backends.md) for exactly what does and doesn't move onto them).

**The landmine to avoid on any cutover:** if you swap a shared engine checkout under multiple already-running projects without also flipping each one's `db_backend` to `pg` in the same change, every project that relied on the old `pg` default will read its Postgres DSN as a SQLite file path the instant the new engine goes live — total breakage, all at once, not a graceful one-project-at-a-time failure. Land the engine swap and the `db_backend="pg"` opt-in **together**. See §5 for the worked sequence.

## 5. Fleet cutover: N projects → one shared broker

This is the topology-specific checklist for moving several already-running projects, each with its **own** Postgres DB and its own queue, onto **one shared broker DB** using the `project` tag (migration 0017's multitenant model — see [broker](broker.md) for the generic model and API). The worked example uses three projects — `alpha`, `beta`, `gamma` — sharing a physical fleet, each starting on its own DB; treat "3 projects, 3 DSNs, 1 shared mount" as the pattern, not a requirement.

### Preconditions the operator decides first

1. **History**: fresh-start (abandon in-flight jobs in each project's old DB — usually fine for a live/pending-job control panel) vs. migrate (copy each old DB's rows into the broker, tagged by `project`).
2. **Broker host + DSN**: where the one shared Postgres "broker" DB lives (a new database on an existing instance, or a dedicated one).
3. **Every project's current DSN + configure() call site**: you need write access to each project's startup code (or its env), not just the one you're standing in.

### Why this is not a "pull + restart" — the landmine from §4

Swapping the shared engine checkout in place makes every process that reads it default to `sqlite`. Any project that hasn't yet been updated to pass `db_backend="pg"` will, the instant the new engine is live, read its `*_DB_URL` Postgres DSN as a SQLite file path — breaking **every** project sharing that checkout simultaneously, not just the one being migrated. The engine swap and each project's `db_backend="pg"` + broker-DSN opt-in must land as one atomic change, with the fleet stopped for the swap.

### Sequence (fleet stopped during the swap)

```text
0. Stage the new engine at a NEW path — do not overwrite the live checkout in
   place. Staging makes rollback = repoint the mount back to the old path.
     stage  <shared-mount>/queue_workflows_v1   (broker_parrot @ the target commit)

1. Stand up the broker DB (idempotent — brings the ledger to schema HEAD):
     QUEUE_WORKFLOWS_DB_BACKEND=pg BROKER_DSN=postgresql://…/broker \
       queue-broker --db-backend pg --db-url-env BROKER_DSN

2. STOP all N fleets (orchestrators + claim workers + schedulers, every project).

3. Per project, in its engine startup configure() (or via env for entry points
   that hand-roll their own configure()):
     configure(project="<project>", db_backend="pg", db_url_env="BROKER_DSN")
   # or, fleet-wide per project:
     export QUEUE_WORKFLOWS_DB_BACKEND=pg
     export QUEUE_WORKFLOWS_PROJECT=<project>
   # and point that project's *_DB_URL env name at the broker DSN (or set
   # BROKER_DSN and db_url_env="BROKER_DSN" directly) — the env knobs reach
   # every process, including standalone worker scripts with no host
   # configure() call.

4. Repoint the shared engine mount at the staged path (all N projects at once).

5. (migrate-history only) backfill each old DB's runs / node_jobs / ingest_jobs
   into the broker, tagging `project`; skip entirely under fresh-start.

6. START all N fleets — now pointed at the broker.

7. VERIFY:
     queue-broker --db-backend pg --db-url-env BROKER_DSN --status
   lists every project with a nonzero/expected cpu+gpu queue depth; the
   operator panel (queue-broker-web, or a host's own panel) shows ONE
   consolidated cpu/gpu queue, filterable by project.

ROLLBACK: repoint the mount back to the old engine path + each project's
   *_DB_URL back to its own DB; restart. Old per-project DBs are untouched
   under fresh-start, so rollback is non-destructive.
```

### Why this stays an operator-gated action, not something to automate blind

- It overwrites/repoints the engine **every** project on the shared mount uses — one mistake takes down all of them at once, compounded by the sqlite-default landmine.
- It requires editing **N separate project repos'** startup code (or their env), not just this one.
- It requires restarting **N live fleets** — typically needs interactive sudo/credentials the automation doesn't have.
- The history decision (fresh-start vs. migrate) is irreversible once made, and DSNs/access for projects other than the one you're standing in are usually not available to whatever is driving the cutover.

Prefer doing this interactively, one gated step at a time, with each step's outcome (schema version, queue depth, panel view) checked before moving to the next.

## 6. Running the test suite

```bash
pip install -e '.[test]'   # pytest, pytest-cov, redis>=5, pymongo>=4.4
```

The suite **requires a reachable Postgres** by default — it forces a `*_test`-suffixed database and creates it if missing (`tests/conftest.py`):

```bash
QUEUE_WORKFLOWS_TEST_DB_URL=postgresql://user:pw@host:port/queue_workflows_test python -m pytest
#   falls back to QUEUE_WORKFLOWS_DB_URL with its db name suffixed _test if unset
#   (conftest refuses to bootstrap against a non-_test-suffixed database name)

python -m pytest tests/test_node_queue.py             # one module
python -m pytest tests/test_node_queue.py::test_name  # one test
python -m pytest -k lease                              # by keyword
```

`conftest.py` truncates the engine tables between tests and resets injected config, so a hook one test wires never leaks into the next.

### Hermetic mode — no Postgres reachable

```bash
QUEUE_WORKFLOWS_TEST_SQLITE=1 python -m pytest
```

Runs the whole suite against a throwaway SQLite file instead (`QUEUE_WORKFLOWS_TEST_SQLITE_PATH`, auto-created), calling `configure(db_backend="sqlite", ...)`. Pg-specific behavior (raw `LISTEN`/trigger `NOTIFY`, some DDL `ALTER` paths) is skipped in this mode — expect a small number of pre-existing skips, not failures.

### Multi-backend contract suite

`tests/test_backend_contract.py` runs the same parametrized contract (enqueue / claim-exactly-once / lease+reclaim / idempotent terminals / the atomic outbox / wake / heartbeat / ON-OFF control) against every `StorageBackend`. Each backend **skips** independently if its server env var is unset or unreachable:

```bash
QUEUE_WORKFLOWS_TEST_REDIS_URL=redis://localhost:6379/0        python -m pytest tests/test_backend_contract.py
QUEUE_WORKFLOWS_TEST_MONGO_URL=mongodb://localhost:27017/?replicaSet=rs0  python -m pytest tests/test_backend_contract.py
```

Mongo needs a replica set (the backend uses multi-document transactions + a change stream — a standalone `mongod` will skip). See [storage_backends](storage_backends.md) for what each backend actually implements and the anti-leakage rule (no driver object — cursor/pipeline/session — crosses the `StorageBackend` port).
