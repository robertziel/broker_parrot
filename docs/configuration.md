# Configuration ⚙️

*How a host wires `queue_workflows` — the `EngineConfig` singleton, every `configure()` keyword, and the `set_*`/`register_*` hook helpers.*

## The single most important design fact

`queue_workflows` couples to nothing domain-specific at import time. Everything
that would otherwise tie it to a particular application — where the DAG comes
from, which env var holds the DSN, how to import a node module, which LLM
server a job should call — is an **injected hook** held on one process-wide
`EngineConfig` singleton (`queue_workflows/config.py`). A host wires those
hooks once at startup, before launching a claim worker / scheduler /
orchestrator, via `queue_workflows.configure(**kwargs)` plus a family of
`set_*` / `register_*` helpers exported from `queue_workflows/__init__.py`.

**Every hook has a safe default.** `import queue_workflows` +
`queue_workflows.configure()` + a reachable database is enough to run the
engine standalone — no host wiring required (`tests/test_standalone_import.py`
pins this). `config.py` is deliberately a **leaf module**: it imports nothing
from any other engine module, so every engine module can
`from queue_workflows import config` with no import cycle. When you're tempted
to have an engine module reach "up" into host code, add a config hook with a
safe default instead — see [architecture.md](architecture.md) for how the
hooks are consumed once wired.

`configure()` is safe to call repeatedly: it's **additive and idempotent** —
only the keyword arguments you pass are mutated, everything else keeps its
current value, so a host (or a test) can call it again to adjust a subset
without resetting the rest.

## `configure(**kwargs)`

| kwarg | default | what it does |
|---|---|---|
| `db_url_env` | `"AI_LEADS_DB_URL"` | Env var name holding the queue DSN (Postgres DSN for `db_backend="pg"`, a filesystem path for `"sqlite"`). |
| `metrics_db_url_env` | `None` | Env var name holding the DSN hw-metrics telemetry publishes to / reads from — the shared "broker" Postgres, so every project shows the same fleet-wide hardware view. `None` falls back to `db_url_env` (a project whose queue DB already *is* the broker needs no extra wiring). Always a pg DSN regardless of `db_backend` — hw-metrics is NOTIFY-only and Postgres-only. |
| `video_model_ids` | `frozenset()` | GPU model ids on the tight per-job video-render wall-clock budget (`claim_worker.budget_for`). |
| `node_module_package` | `""` | Dotted package the default node-module resolver imports under (e.g. `"workflows.nodes"`). Empty ⇒ a stored `node_module` value is imported as a fully-qualified module name. |
| `host_label_env` | `"AI_LEADS_HOST_LABEL"` | Env var name holding this host's identity string (used as `claimed_by` / heartbeat key). |
| `host_priority_env` | `"AI_LEADS_GPU_CONSUMER_PRIORITY"` | Env var name holding this host's GPU claim-priority tiebreak. |
| `container_prefix` | `"ai_leads-"` | cgroup-attribution container-name prefix — `cgroup_attribution.py` sums the CPU/RAM slice owned by containers whose name starts with this. |
| `project` | `""` (or `QUEUE_WORKFLOWS_PROJECT`) | Tenant tag on every queue row (migration 0017) so a shared broker Postgres can pool multiple projects — see [The `project` tenant tag vs `db_namespace`](#the-project-tenant-tag-vs-db_namespace) below. |
| `ingest_queues` | `{"fetch", "load"}` | The valid ingest-family queue names (host-side validation since migration 0008 dropped the DB `CHECK`). **Rejects reuse of the reserved DAG queue names `cpu`/`gpu`** — raises `ValueError` naming the offending set. |
| `ingest_default_budget_s` | `3600` | Wall-clock budget (seconds) `claim_worker.budget_for` applies to ingest queues other than the built-in `fetch`/`load`. |
| `db_backend` | `"sqlite"` (or `QUEUE_WORKFLOWS_DB_BACKEND`) | Which storage engine the store resolves to: `"sqlite"` / `"pg"` (relational, via the dialect seam) or `"redis"` / `"mongodb"` (the `StorageBackend` SPI). See [the ai_leads-byte-compat default philosophy](#the-ai_leads-byte-compat-default-philosophy) below for the one breaking exception. |
| `db_namespace` | `""` | Logical namespace isolating this tenant's jobs on a **shared** redis/mongodb server (every key/collection is scoped by it); `""` means the literal namespace `"default"`. For `pg`/`sqlite` it scopes SPI rows via a `namespace` column. Inverse of `project` — see below. |
| `cancel_orphan_queued_jobs` | `False` | When `True`, `NodePool` periodically flips `queued` node-jobs whose parent run is already `cancelled`/`failed` to `cancelled`, cleaning up queue gauges. Default preserves pre-0.4 behaviour byte-for-byte. |
| `gpu_requires_local_device` | `True` | When `False`, a GPU worker with no local CUDA device does **not** self-park as blind — it's treated as a remote/thin client whose GPU nodes call an external inference server over HTTP and never touch `torch.cuda`, so the box-blind guard doesn't apply. |
| `disable_gpu_worker_hw_sampler` | `False` | When `True`, a GPU worker does not start the engine's flocked host hw-metrics sampler — for a host that publishes hw-metrics from its own (richer/different) sampler and would otherwise double-`NOTIFY` `hw_metrics`. |
| `vlm_pool_node_modules` | `frozenset()` | Node modules that are genuine VLM-facade (HTTP call to the per-host vLLM/ollama server) and therefore safe to run PAR-concurrently in the GPU pool lane. Non-empty ⇒ every *other* no-model GPU job routes to the conc-1 inline lane instead. |
| `gpu_self_load_node_modules` | `frozenset()` | GPU node modules that intentionally run without a cache-managed `model` and are exempt from the required-model guard (`dispatcher._assert_gpu_nodes_declare_model`) — in-process self-loaders not yet migrated onto the warm `ModelCache`. |
| `gpu_pool_backend` | `"redis"` | Which `StorageBackend` the shared GPU pool addresses — independent of `db_backend` (an app can keep `db_backend="pg"` for its own run/DAG state while pooled GPU workers across apps share one redis-backed pool). |
| `gpu_pool_url_env` | `"QUEUE_WORKFLOWS_GPU_POOL_URL"` | Env var name holding the pool store's DSN. |
| `gpu_pool_namespace` | `"gpu_pool"` | Logical tenant namespace for the shared GPU pool — every app/box that should share one fleet uses the same value. |

Two related `EngineConfig` fields hold env-var *names* but are **not** yet
exposed as `configure()` keywords — `redis_url_env` (default
`"QUEUE_WORKFLOWS_REDIS_URL"`) and `mongo_url_env` (default
`"QUEUE_WORKFLOWS_MONGO_URL"`), read by `queue_workflows/backends/__init__.py`
when `db_backend` selects `redis`/`mongodb`. Similarly `ollama_url_env`
(`"AI_LEADS_OLLAMA_URL"`) and `vllm_url_env` (`"AI_LEADS_VLLM_URL"`) name the
env vars the LLM backend factory reads for per-machine server root URLs — see
[gpu_and_llm.md](gpu_and_llm.md). Rename these by setting the env var itself,
or by mutating `get_config()` directly.

`gpu_pool_handlers` (via `register_pool_handler`) and `broker_handlers` (via
`register_broker_handler`) are also `EngineConfig` fields but are populated
through their own registration helpers, not `configure()` — see below.

## Hook-setter helpers

These live in `queue_workflows/__init__.py` alongside `configure()`. Each
wires one seam that plain keyword arguments can't express (a callable, not a
value).

| helper | signature | wires | default when unset |
|---|---|---|---|
| `set_node_module_package` | `(package: str) -> None` | Shorthand for `configure(node_module_package=...)` — the dotted package the default node-module resolver imports under. | Stored `node_module` treated as a fully-qualified module name. |
| `set_node_resolver` | `(resolver: Callable[[str], module]) -> None` | A fully custom node-module resolver, overriding `node_module_package`. The returned module must expose `run(...)`. | Falls back to the `node_module_package` resolver. |
| `set_builtin_model_registrar` | `(registrar: Callable[[], None]) -> None` | The idempotent builtin-model registrar — called by the `model_cache` empty-registry fallback and once at claim-worker/orchestrator startup. Should register `ModelSpec`s into `queue_workflows.model_registry`. | No-op (standalone engine has no models). |
| `set_workflow_provider` | `(load_workflow: Callable[[str], dict], pipeline_schema: Callable[[str], dict], *, resolve_ref: Callable[[Any, dict], Any] \| None = None) -> None` | Where the dispatcher reads the DAG from: `load_workflow(name)` for the workflow definition, `pipeline_schema(name)` for the node DAG, and optionally a custom `resolve_ref`. | `workflow_loader`/`pipeline_schema_loader` unset ⇒ callers error; `resolve_ref` defaults to the engine's own `refs.resolve_ref`. |
| `set_invoke_context` | `(factory: Callable[[dict, dict], ContextManager]) -> None` | A per-node invoke wrapper: given `(job, run)`, returns a context manager. `__enter__` does host setup (e.g. pin a run-context `ContextVar`, capture a live-mock flag) and yields a `finalize(context_delta) -> context_delta` callable `execute_node` applies only on success; `__exit__` tears down on every exit path. | Unset ⇒ the engine runs nodes directly, no wrapping. |
| `set_llm_servers_available` | `(servers: list[str]) -> None` | Declares which LLM server types this host can actually run, published in the worker heartbeat (migration 0014) so the queue UI gates its per-machine server-type control. | `["ollama"]` (the universal baseline). |
| `set_vllm_lifecycle` | `(stop_fn: Callable[[], bool], start_fn: Callable[[str], None]) -> None` | Wires the vllm-sidecar stop/start the idle `LLMSupervisor` and model-switch logic drive — `stop_fn()` frees VRAM, `start_fn(model_id)` (re)starts the sidecar. For a host running vllm as a *separate* container (e.g. via the docker Engine API) so it can be stopped/started without a restart policy. | `None` ⇒ the vllm backend's own built-in pkill/no-op seams (same-container or unmanaged deployment). |
| `set_llm_server_resolver` | `(resolver: Callable[[dict], Any] \| None) -> None` | The per-dispatch LLM server resolver — called once per node-job by `node_executor.execute_node`; its return is threaded into the node as `run(llm_server=...)`, parallel to `model_handle`. Lets a host select across a box's (possibly multiple) vllm/ollama servers, and across boxes via the heartbeat URL advertisement. Pass `None` to clear. | `None` ⇒ nodes get `llm_server=None` and self-resolve from local env. |
| `register_ingest_task` | `(name: str, callable_: Callable[[str], dict]) -> None` | Registers a periodic ingest callable under `name`; the claim worker runs it for an `ingest_jobs` row with that `task_name`. Also the valid `task_name` set `node_queue.enqueue_ingest_job` validates against. | No ingest tasks registered. |
| `set_ingest_schedule` | `(schedule: list[ScheduleEntry]) -> None` | Sets the scheduler's periodic schedule; the `Ticker` fires it, the boot-kick enqueues the non-freshness entries. | Empty ⇒ the ticker has nothing to fire. |
| `register_pool_handler` | `(name: str, callable_: Callable[..., dict]) -> None` | Registers a shared-GPU-pool handler under `name` (deployed on a GPU box); a pooled worker resolves a claimed task's `handler` to it and runs `fn(*, inputs, output_dir, params) -> dict`. Op code lives here; data lives on shared NFS. | Empty on a submit-only app. |
| `register_broker_handler` | `(key: str, callable_: Callable[[job, cancel], dict \| None]) -> None` | Registers a `broker_service` worker-runtime handler under `key` (matched against a granted job's `payload['handler']` else its `resource`); `cancel` is a `threading.Event` set when the broker revokes permission mid-run. See [broker.md](broker.md). | Empty on a submit-only app. |

## The ai_leads-byte-compat default philosophy

`queue_workflows` was extracted from `ai_leads` (its "Phase 6"), which remains
the origin and first consumer, with ~35 sibling projects sharing this one
source. That history explains a pattern visible throughout `config.py`: every
env-var-name field defaults to the name `ai_leads` already renders into its
`.env` — `db_url_env` → `AI_LEADS_DB_URL`, `host_label_env` →
`AI_LEADS_HOST_LABEL`, `host_priority_env` → `AI_LEADS_GPU_CONSUMER_PRIORITY`,
`container_prefix` → `"ai_leads-"`, `ollama_url_env` / `vllm_url_env` →
`AI_LEADS_OLLAMA_URL` / `AI_LEADS_VLLM_URL`. These are **configurable
defaults, not couplings** — the engine reads an env *name* off `EngineConfig`
and a host renames it with `configure(...)`, never by touching engine code.
When adding a new tunable, follow the same shape: read an env name off
`EngineConfig`, default it to the `ai_leads` name.

**The one deliberate exception, and it's BREAKING (v1.0.0):** `db_backend`
now defaults to `"sqlite"`, not `"pg"`. Every prior version defaulted to `pg`;
v1.0.0 switches the default to the friendliest zero-config option for a
reusable library — a daemon-less local file needs no server at all. A
Postgres consumer (`ai_leads` and its siblings) **must opt in**, either with

```python
queue_workflows.configure(db_backend="pg")
```

or by exporting the env knob before the process starts:

```bash
export QUEUE_WORKFLOWS_DB_BACKEND=pg
```

The env-var form matters because it also reaches the **standalone console
scripts** (`queue-orchestrator`, `queue-claim-worker`, `queue-scheduler`,
`queue-worker-control`, `queue-broker`) which have no host `configure()` call
to pass `db_backend=` into — they read `QUEUE_WORKFLOWS_DB_BACKEND` directly
(`config._default_db_backend()`). It also accepts `--db-backend pg` as a CLI
flag on those scripts. Without either, a pg consumer's `AI_LEADS_DB_URL`
value gets read as a *SQLite file path* instead of a Postgres DSN — a subtle,
silent misconfiguration, which is why `_default_db_backend()` validates and
normalizes the raw env value (`"Sqlite"`, `"mongo"`, etc.) and raises loudly
on an unknown name rather than silently mis-routing. Every other default
stays byte-compatible.

## Env knobs

Read directly from the process environment (not renameable via `configure()`
unless noted):

| env var | read by | purpose |
|---|---|---|
| `QUEUE_WORKFLOWS_DB_BACKEND` | `config._default_db_backend()` | Selects the storage engine (`sqlite`/`pg`/`redis`/`mongodb`, plus aliases `postgres`/`postgresql`/`mongo`) before any `configure()` call runs — reaches the standalone console scripts. See above. |
| `QUEUE_WORKFLOWS_PROJECT` | `config._default_project()` | Default tenant `project` tag for entrypoints that hand-roll their own `configure()` and never pass `project=` explicitly. |
| `QUEUE_WORKFLOWS_REDIS_URL` | `backends._url_for` (name overridable via `EngineConfig.redis_url_env`) | The redis DSN, read only when `db_backend="redis"` (or `gpu_pool_backend="redis"`). |
| `QUEUE_WORKFLOWS_MONGO_URL` | `backends._url_for` (name overridable via `EngineConfig.mongo_url_env`) | The mongodb DSN, read only when `db_backend="mongodb"`. |
| `QUEUE_WORKFLOWS_GPU_POOL_URL` | shared GPU pool store lookup (name overridable via `configure(gpu_pool_url_env=...)`) | DSN for the shared GPU pool `StorageBackend`, independent of the app's own `db_backend`. |
| `AI_LEADS_DB_URL` | default value of `db_url_env` | The queue DSN, unless a host renamed `db_url_env`. |
| `AI_LEADS_HOST_LABEL` | default value of `host_label_env` | This host's identity string. |
| `AI_LEADS_GPU_CONSUMER_PRIORITY` | default value of `host_priority_env` | GPU claim-priority tiebreak. |
| `AI_LEADS_OLLAMA_URL` / `AI_LEADS_VLLM_URL` | default value of `ollama_url_env` / `vllm_url_env` | Per-machine LLM server root URLs (deployment topology, set per host — e.g. by ansible). See [gpu_and_llm.md](gpu_and_llm.md). |

## The `project` tenant tag vs `db_namespace`

These two knobs look similar — both scope multi-tenant data on a shared
store — but they are **inverses**, and picking the wrong one silently
mis-isolates or mis-pools your data:

- **`project`** (migration 0017, relational backends only) **pools** tenants
  *onto* one shared Postgres/SQLite queue. There is still exactly one `cpu`
  queue and one `gpu` queue in the database; every queue row (`workflow_runs`,
  `workflow_node_jobs`, `ingest_jobs`, `worker_heartbeats`) carries a `project`
  column, and a client's claim SQL filters on `project = <this client's
  project>` by **exact match**. The default `""` is the single-tenant
  sentinel — every row is `''` and the filter `project=''` matches them all,
  so a single-Postgres-per-project deployment is byte-compatible with zero
  host wiring. This is the shape the [broker](broker.md) uses to run one
  shared control-plane database across every project on the fleet.
- **`db_namespace`** (the `StorageBackend` SPI — redis/mongodb, plus a
  `namespace` column for pg/sqlite SPI rows) **isolates** tenants on a shared
  redis/mongodb server: every key/collection is scoped by it, so two apps
  pointed at one server literally cannot see or claim each other's jobs. See
  [storage_backends.md](storage_backends.md) for the anti-leakage guarantee
  this is part of.

In short: `project` says "we're deliberately sharing one queue, tag your
rows"; `db_namespace` says "we're sharing one server, but must never see each
other's data." Set `project` via `configure(project="ai_leads")` or the
`QUEUE_WORKFLOWS_PROJECT` env var; set `db_namespace` via
`configure(db_namespace=...)`.

## Minimal standalone vs. full host wiring

Minimal — enough to run the engine against a reachable database with no host
code at all (mirrors `tests/test_standalone_import.py`):

```python
import queue_workflows

queue_workflows.configure()   # every default applies; sqlite, no host wiring
# ... register a node module resolver / workflow provider before dispatching
# any real DAG work, or run purely ingest jobs via register_ingest_task.
```

Full host wiring (mirrors the module docstring in `queue_workflows/__init__.py`,
what `ai_leads` does at process startup, before launching a worker /
scheduler / orchestrator):

```python
import queue_workflows
from queue_workflows import model_registry
from queue_workflows.model_registry import ModelSpec

queue_workflows.configure(
    db_url_env="AI_LEADS_DB_URL",
    db_backend="pg",                                   # opt in — v1.0.0 default is sqlite
    video_model_ids=frozenset({"wan_i2v", "ltx_flf"}),
    node_module_package="workflows.nodes",
    container_prefix="ai_leads-",
)
queue_workflows.set_workflow_provider(load_workflow, pipeline_schema)
queue_workflows.set_builtin_model_registrar(register_builtin_models)
queue_workflows.register_ingest_task("run_fetch_all", run_fetch_all)
queue_workflows.set_ingest_schedule([ScheduleEntry(...), ...])

queue_workflows.claim_worker.main(["--queue", "gpu"])
```

## See also

- [architecture.md](architecture.md) — how the process roles (orchestrator,
  claim worker, scheduler) consume these hooks at runtime.
- [gpu_and_llm.md](gpu_and_llm.md) — the LLM/GPU-specific hooks
  (`set_vllm_lifecycle`, `set_llm_server_resolver`, `set_llm_servers_available`,
  `gpu_pool_*`) in depth.
- [broker.md](broker.md) — `register_broker_handler` and the shared-broker
  `project` tenant model.
- [storage_backends.md](storage_backends.md) — `db_backend`, `db_namespace`,
  and the `StorageBackend` SPI's anti-leakage guarantee.
- [schema.md](schema.md) — the migration chain the `project` column and
  `worker_controls` table belong to.
