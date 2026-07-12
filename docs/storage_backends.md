# 🗄️ Storage backends

*The `db_backend` seam: `sqlite`/`pg` run the full DAG engine; `redis`/`mongodb` are an additive, opt-in durable-queue SPI.*

## The selection

```python
import queue_workflows
queue_workflows.configure(db_backend="sqlite")   # "sqlite" (default) | "pg" | "redis" | "mongodb"
```

or the env knob (reaches the standalone console scripts too, which have no host
`configure()` call):

```bash
export QUEUE_WORKFLOWS_DB_BACKEND=pg
```

`db_backend` defaults to **`"sqlite"`** as of v1.0.0 — a daemon-less local file,
the friendliest zero-config default for a reusable library. A Postgres consumer
(existing hosts) must opt in with `configure(db_backend="pg")` or the
env var above; otherwise an `QUEUE_WORKFLOWS_DB_URL` Postgres DSN gets read as a SQLite
path. Aliases `postgres`/`postgresql` → `pg` and `mongo` → `mongodb` are accepted
and normalized; an unknown name raises `ValueError` at `configure()` time. See
[configuration.md](configuration.md) for the full `EngineConfig` reference.

## Two layers — know the boundary

**`sqlite` and `pg` are the two relational engine backends.** They run the
*entire* engine — orchestrator, claim loop, lease/reclaim, dispatch outbox,
watchdogs, worker control — through one dialect seam,
`queue_workflows/dialect.py`. `PgDialect` emits exactly the SQL the engine has
always used (`FOR UPDATE SKIP LOCKED`, `make_interval`, `pg_notify`);
`SqliteDialect` renders the same statements for a single SQLite file, with
`queue_workflows.db` translating pyformat placeholders to qmark/named at
execute time. Selecting between them changes *where the DAG runs*, not *what
runs it*. See [architecture.md](architecture.md) and
[schema.md](schema.md) for the relational schema those two share.

**`redis` and `mongodb` resolve a `StorageBackend`** instead
(`queue_workflows/backends/`, one provider module per backend). This is a
generic durable-queue SPI — enqueue / claim / lease / terminal / outbox / wake —
**not** a second home for the DAG engine. Selecting `redis`/`mongodb` does
**not** re-home the orchestrator, claim worker, or dispatcher; those still need
a relational backend (`sqlite` or `pg`) underneath. Wiring the DAG engine
end-to-end onto a non-relational backend is a later milestone — today the SPI
is a standalone pluggable durable queue a host can use directly (`get_backend()`
below), independent of whatever relational backend runs the DAG.

The redis and pymongo drivers **import lazily** — selecting `pg`/`sqlite` never
imports either, so a relational-only deploy needs neither installed. Selecting
a backend whose driver is missing raises an `ImportError` naming the extra to
install.

```python
from queue_workflows.backends import get_backend

be = get_backend()                       # bound to config.db_backend + db_namespace
jid = be.enqueue("cpu", {"task": "render"})
job = be.claim("cpu", worker="box-1", lease_s=30)
be.complete_with_event(job["id"], "completed", result={"ok": True})
```

## The SPI surface

The port lives in `queue_workflows/backends/base.py` (`StorageBackend`, an
`abc.ABC`). Every method below is abstract — each of the three providers
(`postgres.py`, `redis.py`, `mongodb.py`) implements the full set, pinned
identical by one contract suite.

| Group | Methods |
|---|---|
| Schema / lifecycle | `ensure_schema()`, `close()` |
| Enqueue / claim / lease | `enqueue(queue, payload, *, job_id=None, priority=0) -> str`, `claim(queue, worker, *, lease_s) -> Job \| None`, `renew_lease(job_id, worker, *, lease_s) -> bool`, `reclaim_expired(*, queue=None) -> list[str]`, `requeue_for_retry(job_id) -> Job \| None` |
| Terminal transitions (idempotent) | `mark_completed(job_id, *, result=None)`, `mark_failed(job_id, *, error=None)` |
| Atomic outbox | `complete_with_event(job_id, event_type, *, result=None, detail=None)`, `fail_with_event(job_id, event_type, *, error=None, detail=None)` |
| Reads | `get(job_id) -> Job \| None`, `counts(queue) -> dict[str, int]`, `events(*, since=0, limit=1000) -> list[Event]` |
| Wake | `notify(queue)`, `subscribe(*queues) -> WakeListener` |
| Heartbeat + operator control | `heartbeat(host, queue, *, current_model=None, stale_after_s=30.0)`, `workers(queue) -> list[dict]`, `set_control(host, queue, *, desired_state, stop_policy="hard", requested_by=None)`, `desired_state(host, queue) -> str` |

`claim`/`renew_lease`/`reclaim_expired`/`requeue_for_retry` reproduce the
lease-and-reclaim liveness model from [watchdogs.md](watchdogs.md) at the SPI
level; `set_control`/`desired_state` reproduce the operator ON/OFF plane from
[worker_control.md](worker_control.md) — `desired_state` returns `"off"` only
when an explicit OFF row exists, else `"on"` (absent ⇒ ON, the same
default-on contract the relational engine uses).

`mark_completed`/`mark_failed` and `complete_with_event`/`fail_with_event` are
all **idempotent**: a call against a job already in `TERMINAL_STATUSES`
(`completed`, `failed`) returns `None` and — critically for the outbox pair —
writes **no** event. That mirrors the relational engine's
`UPDATE … WHERE status NOT IN (terminal) RETURNING *` guard.

## Two honesty invariants

The base-class docstring names these explicitly, and `tests/test_backend_contract.py`
pins both:

1. **No leakage.** No SPI method takes or returns a driver handle — no psycopg
   cursor, no redis pipeline, no pymongo session/client in any signature. The
   outbox atomicity ("go terminal *and* append the event, both-or-neither") is
   exposed as one high-level call per backend, implemented atomically in that
   backend's own idiom. A host can never hold a transaction object across
   backends, so one driver's internals can't bleed into another's call sites.
2. **Namespace-bound.** Each backend instance is constructed with `(url,
   namespace)` and every key/row/document/collection it touches is scoped by
   that namespace — two tenants pointed at the same Redis or MongoDB server
   cannot enqueue, claim, read, count, or wake into each other's jobs. `""`
   normalizes to the literal `"default"` (`normalized_namespace`).

## Per-backend mechanics

| Guarantee | pg | redis | mongodb |
|---|---|---|---|
| Claim exactly-once | `FOR UPDATE SKIP LOCKED` | `ZPOPMIN` inside a **Lua** script | `find_one_and_update` |
| Atomic outbox | one **transaction** | one **Lua** script | one **multi-document transaction** |
| Wake | `LISTEN` / `pg_notify` (in-txn) | **pub/sub** (fire-and-forget) | **change stream** on a capped collection |
| Namespace isolation | `namespace` column, every query filtered | key prefix `qw:<namespace>:` | one **database** per namespace (`qw_<namespace>`) |

**pg** (`queue_workflows/backends/postgres.py`) is the reference adapter: it
uses its *own* small tables — `qw_jobs`, `qw_events`, `qw_workers`,
`qw_controls` — separate from the engine's `workflow_*`/dialect-driven schema,
so enabling the SPI never collides with or migrates a host's existing engine
tables.

**redis** (`backends/redis.py`) has no `SKIP LOCKED` and no cross-key ACID
transaction, so every atomic step — enqueue, claim, terminal+event, requeue —
runs as a registered **Lua script**, one indivisible server-side unit.
Priority + FIFO is a per-queue sorted set (score `-priority`, a zero-padded
monotonic sequence breaks ties); lease/reclaim is a per-queue *running* sorted
set scored by expiry, swept with `ZRANGEBYSCORE … now`. The wake is
**pub/sub**, so a subscriber that's down misses it — the same reason the
relational engine keeps a safety-poll behind `LISTEN`. Because keys are
derived server-side inside the scripts, this targets a **single Redis
instance**, not Cluster (which needs every key in one hash slot).

**mongodb** (`backends/mongodb.py`) claims via `find_one_and_update` on the
oldest, highest-priority `queued` document — the standard Mongo work-queue
claim, reproducing `SKIP LOCKED`'s effect because the first claimer's update
removes the doc from the `status:'queued'` filter before a second claimer can
match it. The atomic outbox is a genuine **multi-document transaction**
(terminal update + event insert, both-or-neither), and the wake is a
**change stream** on a capped `wake` collection. Both transactions and change
streams require a **replica set** — a single-node RS is sufficient, but a
standalone `mongod` fails loudly (`ensure_schema()` pings the server up front
to catch this at boot, not mid-claim).

## Testing

One parametrized contract suite, `tests/test_backend_contract.py`, runs against
all three live servers; a backend whose server is unreachable **SKIPs**, it
doesn't fail:

```bash
export QUEUE_WORKFLOWS_TEST_DB_URL=postgresql://user:pw@host:port/queue_workflows_test
export QUEUE_WORKFLOWS_TEST_REDIS_URL=redis://localhost:6379/0
export QUEUE_WORKFLOWS_TEST_MONGO_URL="mongodb://localhost:27017/?replicaSet=rs0"
python -m pytest tests/test_backend_contract.py
```

Each test gets a fresh random namespace, so the suite's own tests never see
each other's jobs — and the cross-namespace test doubles as the data-leakage
guard for invariant (2) above. `.[test]` (used by the repo's own test setup)
already pulls in `redis` + `pymongo`; a standalone consumer that only wants the
SPI adds `pip install 'queue_workflows[redis]'` or `'queue_workflows[mongodb]'`
without pulling the other in.

## `db_namespace` vs `project` — pooling and isolation are inverses

`db_namespace` (this doc) and `project` (migration `0017`+, see
[configuration.md](configuration.md) and [broker.md](broker.md)) both scope
"whose job is this" but solve opposite problems:

- **`db_namespace` isolates.** Two apps pointed at one shared Redis/MongoDB
  server get their own key prefix / database and **cannot see each other's
  jobs at all** — full separation on a shared backend.
- **`project` pools.** Multiple projects share **one** Postgres queue
  (`cpu`/`gpu`/ingest rows all carry a `project` column) and each client
  claims only rows whose `project` matches its own — a filter, not a
  partition, so one broker DB serves several tenants side by side.

They compose independently: a `pg`-backed host sets `project` to pool onto a
shared broker DB; a `redis`/`mongodb`-backed host sets `db_namespace` to
isolate on a shared server. Neither implies the other.

## The shared GPU pool is a separate `StorageBackend` selection

`gpu_pool_backend` (default `"redis"`) and `gpu_pool_namespace` address a
**pooled GPU worker queue independently of `db_backend`** — an app can run its
own DAG/run state on `db_backend="pg"` while its GPU workers claim
self-contained tasks from one shared Redis-backed pool that spans multiple
apps/boxes. It reuses the same `StorageBackend` SPI documented here, just
pointed at a different DSN/namespace pair. See
[gpu_and_llm.md](gpu_and_llm.md) for the pool's task shape and handler
registration.
