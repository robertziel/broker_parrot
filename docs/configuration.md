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
| `db_url_env` | `"QUEUE_WORKFLOWS_DB_URL"` | Env var name holding the queue DSN (Postgres DSN for `db_backend="pg"`, a filesystem path for `"sqlite"`). |
| `metrics_db_url_env` | `None` | Env var name holding the DSN hw-metrics telemetry publishes to / reads from — the shared "broker" Postgres, so every project shows the same fleet-wide hardware view. `None` falls back to `db_url_env` (a project whose queue DB already *is* the broker needs no extra wiring). Always a pg DSN regardless of `db_backend` — hw-metrics is NOTIFY-only and Postgres-only. |
| `video_model_ids` | `frozenset()` | GPU model ids on the tight per-job video-render wall-clock budget (`claim_worker.budget_for`). |
| `node_module_package` | `""` | Dotted package the default node-module resolver imports under (e.g. `"workflows.nodes"`). Empty ⇒ a stored `node_module` value is imported as a fully-qualified module name. |
| `host_label_env` | `"QUEUE_WORKFLOWS_HOST_LABEL"` | Env var name holding this host's identity string (used as `claimed_by` / heartbeat key). |
| `host_priority_env` | `"QUEUE_WORKFLOWS_GPU_CONSUMER_PRIORITY"` | Env var name holding this host's GPU claim-priority tiebreak. |
| `container_prefix` | `""` (attribution off) | cgroup-attribution container-name prefix — `cgroup_attribution.py` sums the CPU/RAM slice owned by containers whose name starts with this. Empty ⇒ no attribution; a host sets its own prefix. |
| `project` | `""` (or `QUEUE_WORKFLOWS_PROJECT`) | Tenant tag on every queue row (migration 0017) so a shared broker Postgres can pool multiple projects — see [The `project` tenant tag](#the-project-tenant-tag) below. |
| `ingest_queues` | `{"fetch", "load"}` | The valid ingest-family queue names (host-side validation since migration 0008 dropped the DB `CHECK`). **Rejects reuse of the reserved DAG queue names `cpu`/`gpu`/`download`** — raises `ValueError` naming the offending set. |
| `ingest_default_budget_s` | `3600` | Wall-clock budget (seconds) `claim_worker.budget_for` applies to ingest queues other than the built-in `fetch`/`load`. |
| `cpu_worker_threads` | `1` | *(lane width — see also the per-job `cpu_slots` column, migration 0024, for how much of a lane ONE job takes.)* Node-jobs ONE **cpu** claim-worker process runs at a time. `1` = the serial worker (no pool is created). `N > 1` gives the cpu lane an in-process pool of N long-lived worker threads. Env twin `QUEUE_WORKFLOWS_CPU_WORKER_THREADS` (wins), CLI `--threads N`. gpu/ingest are unaffected. **For CPU-bound pure-Python nodes the GIL serialises threads — prefer N processes** (`deploy/run_claim_lane.sh`); raise this for IO-bound cpu nodes. Budget ≈ 4–5 DB connections per thread. See [deployment](deployment.md) §2.5. |
| `download_max_concurrent` | `1` | **The download lane's width** (migration 0025): how many `download` node-jobs may be RUNNING at once, counted **fleet-wide per project** (no per-host scope — the shared uplink is the protected resource). The cap binds in the CLAIM (`claim_next_download_job`), serialized by a (queue, project)-keyed advisory lock, and a dead downloader's slot frees via the ordinary lease reclaim. Floored at 1. Env twin `QUEUE_WORKFLOWS_DOWNLOAD_MAX_CONCURRENT` (wins). |
| `gpu_box_max_concurrent` | `{}` | **A per-BOX ceiling on concurrent GPU node-jobs** — `{"box-b": 3, "box-c": 3}`. Everything else that sets a GPU box's concurrency describes the LLM *server* (the operator's PAR, what the running vLLM's KV cache holds, what the serving kind can absorb); this describes the *machine* — its power envelope, its unified memory, the thermal latch a flood of concurrent work trips. Keyed by the **physical box id** (`gpu_box_id` / `gpu_model_lease.default_box_id()`, e.g. `box-b`) — the same name `avoid_box`/`force_box` place against, **not** the per-project lane label (`box-b-gpu`), because every project's lane on a machine shares one card. Applied as the last word in `ClaimWorker._pool_parallelism()`, the one function every consumer of "how many can this box run" goes through, so it reaches the heartbeat's advertised `concurrency`, the DB-side box-slot arbiter's `max_running`, the per-kind widths, the inline admission gate and the pool feeder's budget alike. **Lowers only** — a box whose vLLM holds 2 still runs 2 under a ceiling of 3. A box not named here has **no** ceiling (byte-identical to before). A value that is not a positive int is **ignored, never obeyed**: a 0 would be a GPU lane that claims nothing, silently, forever. Env twin `QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT` (wins **per entry**). |
| `db_backend` | `"sqlite"` (or `QUEUE_WORKFLOWS_DB_BACKEND`) | Which storage engine the store resolves to: `"sqlite"` or `"pg"` (both via the dialect seam; aliases `postgres`/`postgresql` → `pg`). Any other value raises `ValueError`. See [env-var names](#env-var-names--queue_workflows_-only) below for the one breaking exception. |
| `cancel_orphan_queued_jobs` | `False` | When `True`, `NodePool` periodically flips `queued` node-jobs whose parent run is already `cancelled`/`failed` to `cancelled`, cleaning up queue gauges. Default preserves pre-0.4 behaviour byte-for-byte. |
| `gpu_requires_local_device` | `True` | When `False`, a GPU worker with no local CUDA device does **not** self-park as blind — it's treated as a remote/thin client whose GPU nodes call an external inference server over HTTP and never touch `torch.cuda`, so the box-blind guard doesn't apply. |
| `disable_gpu_worker_hw_sampler` | `False` | When `True`, a GPU worker does not start the engine's flocked host hw-metrics sampler — for a host that publishes hw-metrics from its own (richer/different) sampler and would otherwise double-`NOTIFY` `hw_metrics`. |
| `vlm_pool_node_modules` | `frozenset()` | Node modules that are genuine VLM-facade (HTTP call to the per-host vLLM/ollama server) and therefore safe to run PAR-concurrently in the GPU pool lane. Non-empty ⇒ every *other* no-model GPU job routes to the conc-1 inline lane instead. |
| `gpu_self_load_node_modules` | `frozenset()` | GPU node modules that intentionally run without a cache-managed `model` and are exempt from the required-model guard (`dispatcher._assert_gpu_nodes_declare_model`) — in-process self-loaders not yet migrated onto the warm `ModelCache`. |

Two related `EngineConfig` fields hold env-var *names* but are **not** yet
exposed as `configure()` keywords: `ollama_url_env`
(`"QUEUE_WORKFLOWS_OLLAMA_URL"`) and `vllm_url_env` (`"QUEUE_WORKFLOWS_VLLM_URL"`) name the
env vars the LLM backend factory reads for per-machine server root URLs — see
[gpu_and_llm.md](gpu_and_llm.md). Rename these by setting the env var itself,
or by mutating `get_config()` directly.

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

## Env-var names — `QUEUE_WORKFLOWS_*` only

Every runtime knob is canonically named `QUEUE_WORKFLOWS_<KNOB>` (`db_url_env`
→ `QUEUE_WORKFLOWS_DB_URL`, `host_label_env` → `QUEUE_WORKFLOWS_HOST_LABEL`,
`host_priority_env` → `QUEUE_WORKFLOWS_GPU_CONSUMER_PRIORITY`, `ollama_url_env`
/ `vllm_url_env` → `QUEUE_WORKFLOWS_OLLAMA_URL` / `QUEUE_WORKFLOWS_VLLM_URL`,
plus the tuning knobs listed throughout these docs).

**No legacy fallback.** The engine was extracted from an earlier host
application (now retired) whose env file used a different prefix. Those
names are no longer read: if a retired `<LEGACY>_<KNOB>` twin is still set,
`queue_workflows/envcompat.py` raises a `RuntimeError` naming its replacement
`QUEUE_WORKFLOWS_<KNOB>`, so an un-migrated deploy fails at startup instead of
silently losing its setting (`tests/test_env_compat.py`, which also
source-scans the engine for any legacy-prefixed name). `container_prefix` has no
default (`""` = cgroup attribution off); a host sets it with
`configure(container_prefix=...)`.

These are **configurable defaults, not couplings** — the engine reads an env
*name* off `EngineConfig` and a host renames it with `configure(...)`, never by
touching engine code. When adding a new tunable, read the env through
`envcompat.env_get` with the canonical name.

**The one deliberate exception, and it's BREAKING (v1.0.0):** `db_backend`
now defaults to `"sqlite"`, not `"pg"`. Every prior version defaulted to `pg`;
v1.0.0 switches the default to the friendliest zero-config option for a
reusable library — a daemon-less local file needs no server at all. A
Postgres consumer **must opt in**, either with

```python
queue_workflows.configure(db_backend="pg")
```

or by exporting the env knob before the process starts:

```bash
export QUEUE_WORKFLOWS_DB_BACKEND=pg
```

The env-var form matters because it also reaches the **standalone console
scripts** (`queue-orchestrator`, `queue-claim-worker`, `queue-scheduler`,
`queue-worker-control`, `queue-hw-watch`) which have no host `configure()` call
to pass `db_backend=` into — they read `QUEUE_WORKFLOWS_DB_BACKEND` directly
(`config._default_db_backend()`); `queue-hw-watch` also takes `--db-backend pg`. Without either, a pg consumer's `QUEUE_WORKFLOWS_DB_URL`
value gets read as a *SQLite file path* instead of a Postgres DSN — a subtle,
silent misconfiguration, which is why `_default_db_backend()` validates and
normalizes the raw env value through `config.canonical_db_backend` — the same
validator `configure()` and `--db-backend` use — and raises loudly on an
unknown name rather than silently mis-routing. Every other default
stays byte-compatible.

## Env knobs

Read directly from the process environment (not renameable via `configure()`
unless noted):

| env var | read by | purpose |
|---|---|---|
| `QUEUE_WORKFLOWS_DB_BACKEND` | `config._default_db_backend()` | Selects the storage engine (`sqlite`/`pg`, plus aliases `postgres`/`postgresql`) before any `configure()` call runs — reaches the standalone console scripts. See above. |
| `QUEUE_WORKFLOWS_PROJECT` | `config._default_project()` | Default tenant `project` tag for entrypoints that hand-roll their own `configure()` and never pass `project=` explicitly. |
| `QUEUE_WORKFLOWS_DB_URL` | default value of `db_url_env` | The queue DSN, unless a host renamed `db_url_env`. |
| `QUEUE_WORKFLOWS_HOST_LABEL` | default value of `host_label_env` | This host's identity string. |
| `QUEUE_WORKFLOWS_GPU_CONSUMER_PRIORITY` | default value of `host_priority_env` | GPU claim-priority tiebreak. |
| `QUEUE_WORKFLOWS_OLLAMA_URL` / `QUEUE_WORKFLOWS_VLLM_URL` | default value of `ollama_url_env` / `vllm_url_env` | Per-machine LLM server root URLs (deployment topology, set per host — e.g. by ansible). See [gpu_and_llm.md](gpu_and_llm.md). |
| `QUEUE_WORKFLOWS_DISABLE_LLM_CONTEXT_SWITCH` | `llm_context_lever.disabled()` | Kill switch for the whole context-window mechanism (0028): no box changes its window, every advertised ceiling reads "what I serve now", and no context switch is ever requested. Sized jobs still route only to a box whose window fits. |
| `QUEUE_WORKFLOWS_OLLAMA_CONTAINER` / `QUEUE_WORKFLOWS_VLLM_CONTAINER` | `llm_context_lever.container_name()` | The container each LLM server runs in on this box, for the built-in window lever (defaults `broker-ollama` / `vllm`). No such container ⇒ no lever ⇒ the box advertises no room to grow. |
| `QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB` | `llm_context.reserve_bytes()` | Memory kept free **beyond** the LLM's weights and cache when computing how big a window this box could serve (default 8192 MB). Raise it on a box that also renders or trains. |
| `QUEUE_WORKFLOWS_VLLM_MAX_CONTEXT` | `claim_worker._vllm_model_max_context()` | The largest window this box's vLLM **model** supports. vLLM does not publish it (`/v1/models` reports the window it was started with), and starting it beyond the model's own maximum makes vLLM exit at startup — so without this a vLLM box advertises no room to grow. ollama needs no equivalent: it reports the model maximum itself. |
| `QUEUE_WORKFLOWS_ROGUE_LLM_GUARD` / `QUEUE_WORKFLOWS_DISABLE_ROGUE_LLM_GUARD` | `rogue_llm.sweep()` | One LLM daemon per box: a second running `ollama/ollama` (or `vllm/vllm-openai`) container beside the engine's own is stopped on sight by whichever engine process holds the docker socket. `=report` only names it; the `DISABLE` twin turns the guard off. A box whose configured daemon is not running is never policed. |
| `QUEUE_WORKFLOWS_GPU_CONTAINER_ALLOW` | `rogue_llm.sweep()` | Opt-in second rule of the same guard: comma-separated container-name prefixes and/or compose project names that may use this box's GPU (the engine's own ollama/vLLM containers always may). Any other RUNNING container with GPU access (a device request, the nvidia runtime, a `/dev/dri`/`/dev/kfd`/`/dev/nvidia*` device) is stopped and pinned to `restart=no` when dockerd would bring it back by itself (`always`/`unless-stopped`/`on-failure`), and named once but left alone when its policy is `no` (a run launched by hand). Unset = off, no container is inspected. `QUEUE_WORKFLOWS_ROGUE_LLM_GUARD=report` applies. |
| `QUEUE_WORKFLOWS_ROGUE_GPU_RUNS` | `rogue_llm.sweep()` | `report` (default) or `stop` for hand-launched GPU RUNS — a container with GPU access, outside the allow-list, whose restart policy is already `no`. The default names it once and leaves it: presence is not the offence, memory is, and the memory sentinel stops it if the box runs short (`rogue_llm.stop_gpu_outsiders`, after the broker has given back its own servers). `stop` removes it on sight. GPU SERVICES are always stopped on sight — they come back at boot by themselves. |
| `QUEUE_WORKFLOWS_VLLM_MIN_UTILIZATION` | `claim_worker._vllm_min_utilization()` | The smallest `--gpu-memory-utilization` worth starting vLLM at (default 0.30). The engine sizes vLLM's pool to the box's free memory at start (never above the operator's value); below this it does not start vLLM on that box at all. |
| `QUEUE_WORKFLOWS_VLLM_MIN_CONCURRENCY` | `claim_worker._vllm_min_concurrency()` | Fewest concurrent requests that make vLLM worth starting where ollama can serve instead (default 2). vLLM's advantage over ollama is batching; when the memory left on a box pays for fewer than this many requests of the job's window (counted from the model's remembered profile, migration 0029) the start is refused and the box keeps serving ollama. A box with no ollama URL of its own is unaffected, as is a model with no profile yet. |
| `QUEUE_WORKFLOWS_LLM_DANGER_FLOOR_MB` | `claim_worker._danger_floor_bytes()` | How little free memory means the box is in trouble, whatever anyone planned (default 12288 = 12 GB). Below it, a vLLM still WARMING is stopped and its switch abandoned, and a loaded vLLM loses its exemption from a start-verdict. An absolute floor on purpose — never a multiple of `QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB`, which is a planning margin: a box holding a legitimate 80 GB pool sits well under 2x a 32 GB reserve and is not dying. |
| `QUEUE_WORKFLOWS_LLM_KILL_FLOOR_MB` | `memory_sentinel.kill_floor_bytes()` | The last-resort line (default 512 MB of `MemAvailable`): under it the gpu lane sheds its LLM servers AT ONCE, ignoring the 60 s cooldown the danger floor observes. The danger floor is the line that saves a box; this is the line for a box that fell through it. |
| `QUEUE_WORKFLOWS_DISABLE_MEMORY_SENTINEL` | `memory_sentinel.disabled()` | Turns off the 1 Hz memory sentinel thread on the gpu lane. The 10 s lease tick still sheds; only the cadence is lost. |
| `QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT` | `claim_worker.resolve_box_max_concurrent()` | Env twin of `configure(gpu_box_max_concurrent=...)`, carrying the whole mapping in one variable so a single rendered `.env` fits the fleet: `box-b=3,box-c=3`. A bare integer (`3`) means "this box", for a machine whose compose file is its own; a named entry beats the bare one. Precedence is **per entry**, not per map — an env naming `box-c` overrides `box-c` and leaves `box-b`'s configured ceiling standing, because an operator adding one machine must never silently *remove* another's ceiling. An entry that carries an `=` must name a box (`=3` is a typo, not "every box"); an unreadable or non-positive entry is dropped with a warning and costs only that entry. |
| `QUEUE_WORKFLOWS_DOCKER_SOCK` | `llm_backends.docker_api.default_socket()` | The docker socket the vLLM model switch and the window lever talk to (default `/var/run/docker.sock`). Not bind-mounted into the worker ⇒ both levers are inert, which is the safe direction. |

## The `project` tenant tag

**`project`** (migration 0017) **pools** tenants *onto* one shared
Postgres/SQLite queue. There is still exactly one `cpu` queue and one `gpu`
queue in the database; every queue row (`workflow_runs`, `workflow_node_jobs`,
`ingest_jobs`, `worker_heartbeats`) carries a `project` column, and a client's
claim SQL filters on `project = <this client's project>` by **exact match**.
The default `""` is the single-tenant sentinel — every row is `''` and the
filter `project=''` matches them all, so a single-Postgres-per-project
deployment is byte-compatible with zero host wiring. This is the shape the
[broker](broker.md) uses to run one shared control-plane database across every
project on the fleet. Set it via `configure(project="alpha")` or the
`QUEUE_WORKFLOWS_PROJECT` env var.

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
what a host application does at process startup, before launching a worker /
scheduler / orchestrator):

```python
import queue_workflows
from queue_workflows import model_registry
from queue_workflows.model_registry import ModelSpec

queue_workflows.configure(
    db_url_env="MYAPP_DB_URL",
    db_backend="pg",                                   # opt in — v1.0.0 default is sqlite
    video_model_ids=frozenset({"video-model-b", "video-model-c"}),
    node_module_package="myapp.nodes",
    container_prefix="myapp-",
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
  (`set_vllm_lifecycle`, `set_llm_server_resolver`, `set_llm_servers_available`) in depth.
- [broker.md](broker.md) — the shared-broker `project` tenant model.
- [storage_backends.md](storage_backends.md) — `db_backend`: sqlite vs pg.
- [schema.md](schema.md) — the migration chain the `project` column and
  `worker_controls` table belong to.
