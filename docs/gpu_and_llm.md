# 🎮 GPU model management & LLM backends

*How the GPU claim worker keeps one model warm and how per-machine ollama/vLLM servers are wired.*

This doc covers the GPU in one place, since the warm cache and the LLM servers sit on the same physical resource — the GPU — and interact (a diffusion model and a co-tenant LLM server compete for the same VRAM on a box).

## Contents

1. [The GPU warm-model cache & routing](#1-the-gpu-warm-model-cache--routing)
2. [LLM backends: per-machine ollama / vLLM + idle supervisor](#2-llm-backends-per-machine-ollama--vllm--idle-supervisor)

---

## 1. The GPU warm-model cache & routing

### Why: reload cost dominates a short GPU job

A GPU node (diffusion, super-resolution, detection, …) declares `gpu: true` + `model: "<id>"` in its workflow JSON. Loading weights onto the GPU can take longer than running the job itself, so the whole point of a **long-lived GPU claim worker** (one process, concurrency-1) is to keep a model resident across consecutive same-model jobs and pay the load cost only on an actual swap.

### The two modules

- **`queue_workflows/model_cache.py`** — the cache *logic*, deliberately DB-decoupled. `ModelCache` never imports psycopg; every side effect is an injected callback:
  - `publish_current_model(model_id | None)` — advertise the loaded model (see below).
  - `register_builtins()` — the empty-registry re-registration fallback (below).
- **`queue_workflows/gpu_model_cache.py`** — wires the **one process-wide instance**. `gpu_model_cache()` lazily constructs a single `ModelCache` per process and injects `_publish_current_model`, which upserts `worker_heartbeats.current_model` (queue `'gpu'`) via `node_queue.upsert_worker_heartbeat`. This is late-bound (the shim re-reads the module-level function on every call) so a test that monkeypatches `_publish_current_model` is still observed.

A GPU worker is one process holding one model by contract, so one cache per process is correct — there is no cross-process sharing to reconcile.

### `require_model(model_id)` — the swap decision

```python
handle = gpu_model_cache().require_model("sdxl-base")
```

- Same model already loaded → return the cached handle immediately (no I/O).
- Different model (or none) loaded → `drop_cache()` (drop the ref, `gc.collect()`, `torch.cuda.empty_cache()`, `malloc_trim(0)`), publish `current_model = NULL` **mid-swap** (so the claim's affinity tiebreak doesn't try to pin a same-model job here while a load is in flight), resolve the id via `model_registry.get()`, call the spec's `loader()`, then publish the new `current_model`.
- If the registry is empty when resolved (a prefork `worker_process_init` race with the first task on a freshly-forked child), it re-runs the injected/configured builtin registrar and retries once.

`mark_busy()` / `mark_idle()` bracket each job (mirrors the watchdog bracket elsewhere in the engine) — `active` stays >0 while a job runs so the idle reaper never unloads mid-inference, and `mark_idle()` restarts the idle clock from the *end* of the job.

### `model_registry.py` + `ModelSpec`

```python
@dataclass
class ModelSpec:
    id: str
    loader: ModelLoader            # zero-arg factory -> handle
    unloader: ModelUnloader | None = None
    est_vram_gb: float = 0.0       # scheduler hint; see capacity-aware assignment below
```

`MODELS: dict[str, ModelSpec]` is populated by the **host**, never by the engine — the engine only owns the registry as a *target*. The host registers its specs via the `builtin_model_registrar` hook:

```python
queue_workflows.set_builtin_model_registrar(register_builtin_models)
```

This is called once at GPU-worker startup and again by `ModelCache`'s empty-registry fallback. `model_registry.py` stays cheap to import — no torch/diffusers at module load; loaders defer heavy imports until called. See [configuration](configuration.md) for the full hook-setter list.

### Warm-model affinity routing (the claim `ORDER BY`)

The GPU claim SQL (`node_queue._CLAIM_SQL`, see [architecture § the queue mechanism](architecture.md#3-the-queue-mechanism)) adds two tiebreaks after the capability gate:

1. **Affinity** — `required_model IS NOT DISTINCT FROM current_model` sorts first, so a worker with a model already warm preferentially claims a job needing that same model instead of forcing a cold worker to reload.
2. **`host_priority`** direction term — flips the creation-order walk (oldest-first for `host_priority >= 0`, newest-first for `< 0`) so a fleet can bias which host drains a queue first; set per-host via the `QUEUE_WORKFLOWS_GPU_CONSUMER_PRIORITY` env knob (`config.host_priority_env`).

Both tiebreaks are built only from validated ints/fixed SQL fragments — never caller strings — matching the engine-wide claim-SQL safety rule.

### Capacity-aware GPU model assignment (migration 0015)

Before 0015, *any* GPU worker could claim *any* GPU job as long as the model id was known — a model whose weights don't fit a given machine's VRAM was claimed anyway and OOM'd (or failed) at load, and if **no** machine in the fleet was big enough the node just sat `queued` forever with no visible reason. 0015 fixes both:

- **`worker_heartbeats.vram_total_mb`** — the machine's total GPU VRAM (MB), sampled by the worker each heartbeat (`ClaimWorker._vram_total_mb`, probed once and cached — a stable hardware property).
- **`worker_heartbeats.fits_models`** — `text[]` of registered model ids whose `est_vram_gb` fits this machine, computed **worker-side** via `model_registry.fits_within(vram_total_mb)` (the orchestrator holds no model registry, so the fit decision is pushed to the worker and advertised as plain data). A model with `est_vram_gb <= 0` (unset/informational) is treated as fitting everywhere; `vram_total_mb is None` (unknown capacity) advertises *all* known ids so a cold worker never wedges the queue.
- **`workflow_node_jobs.unassignable_at` / `unassignable_reason`** — the **red flag**. The node stays `queued` (not failed — the condition is transient: a bigger machine can come online and clear it). `node_queue.flag_unassignable_gpu_jobs()`, driven by `NodePool._sweep_unassignable_jobs` on an interval (default 15 s, `_unassignable_interval_s`), flags a queued `gpu` job with a `required_model` iff **no fresh GPU heartbeat** (within the stale-worker window, default 30 s) advertises that model in `fits_models`. It clears the flag once a live worker fits it, or once the job leaves `queued`.
  - **Liveness guard**: flagging requires at least one *fresh* GPU heartbeat to exist — if the whole GPU fleet is momentarily down (a deploy, all-workers bounce), that's a liveness gap, not a capacity verdict, so the sweep is a no-op rather than red-flagging every job.
  - Project-scoped (migration 0017): the sweep only judges a project's own jobs against that project's own fleet.
  - A newly-flagged row emits one `unassignable` `workflow_node_events` row (joins the event vocabulary added by 0011/0015 — see [architecture § durable node-event history](architecture.md)).

The claim capability gate (`node_queue._capability_clause`) itself still gates only on `known_models`/`required_model IS NULL`; `fits_models` is advertised data the fleet sweep reads, not (today) a second claim-time filter — a worker whose `fits_models` excludes a model simply never advertises it as "known" for the affinity sort in the first place when the caller passes `fits_models` as `known_models`.

### The idle-unload decision

```python
def gpu_should_unload(handle_present: bool, active: int, idle_s: float, ttl_s: float) -> bool:
    if ttl_s <= 0:
        return False
    return handle_present and active <= 0 and idle_s >= ttl_s
```

Pure function, unit-testable with no real waiting. `ModelCache.ensure_idle_reaper()` arms a daemon thread the first time `require_model` is called (idempotent; no-op if `QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER` is set or `idle_ttl_s <= 0`), polling on `clamp(idle_ttl_s / 5, 5s, 60s)`. Default TTL is 60 s (`QUEUE_WORKFLOWS_GPU_MODEL_IDLE_TTL_S`) — short, so a shared GPU frees VRAM quickly for a co-tenant vLLM/ollama server between diffusion bursts (see [§3](#2-llm-backends-per-machine-ollama--vllm--idle-supervisor)). Each tick calls `reap_idle_once()`, which re-checks the same predicate under the lock and, if true, drops the cache and republishes `current_model = NULL`.

This is the same shape (`now_fn`/injectable clock, pure decision function, daemon-thread reaper) as the LLM idle supervisor in §2 and the watchdogs — a deliberate house style so the three GPU-idle mechanisms in this codebase read alike.

### GPU health sampling

Per-container GPU%/RAM sampling that backs the `GpuHealthWatchdog`'s wedge detection (health-driven, not wall-clock) is covered in **[watchdogs.md](watchdogs.md)** — not duplicated here. That watchdog is a *liveness* guard (is the job making progress); this section is a *capacity/residency* concern (which model is warm, does it fit).

---

## 2. LLM backends: per-machine ollama / vLLM + idle supervisor

### Why: a co-tenant VLM server shares the GPU with the DAG worker

A workflow node often needs a language/vision-language model (e.g. captioning or reasoning over an image) rather than an in-process diffusion model. Which *kind* of server a machine runs for that — **ollama** (an externally-managed long-lived daemon) or **vLLM** (an OpenAI-compatible server the engine can start/stop to reclaim VRAM) — is per-machine, operator-set state. `queue_workflows/llm_backends/` is the polymorphic seam: a factory reads that config, builds the matching backend, and every node drives it through one surface — never branching on server type itself.

The package is deliberately **OpenAI-chat-shape biased and vendor-coupled to exactly ollama + vLLM** (not a general LLM abstraction) — both speak the same `POST /…/chat` shape, so the host node is written once. It owns **request accounting** and the **lifecycle decision**; it does **not** make the HTTP call (the host node POSTs to `backend.chat_url` itself, bracketed by `mark_request_start`/`mark_request_end`) and it does **not** know about your container runtime. It imports nothing but stdlib + `queue_workflows.*`.

### The pieces (`queue_workflows/llm_backends/`)

| File | Role |
|---|---|
| `base.py` | `LLMBackend` (ABC) — concrete, shared request accounting; abstract identity (`server_type`, `chat_url`, `health_url`) + lifecycle (`ensure_ready`, `is_running`, `stop_server`). |
| `ollama.py` | `OllamaBackend` — every lifecycle method is a truthful no-op; the daemon self-manages idle via `OLLAMA_KEEP_ALIVE`. |
| `vllm.py` | `VLLMBackend` (+ `VLLMState`) — a real lifecycle state machine over fully-injected I/O (kill/ensure-up/health/served-model probes; httpx lazy-imported). |
| `supervisor.py` | `LLMSupervisor` (+ pure `vllm_should_stop`) — the idle-VRAM daemon for vLLM. |
| `factory.py` | `BackendFactory` / `get_backend(host_label, queue)` — the per-`(host, queue)` owner of the live backend + its supervisor, synced to the DB config. |

`LLMBackend`'s concrete half (`mark_request_start`/`mark_request_end`/`inflight`/`idle_seconds`) is the direct analog of `ModelCache`'s `active`/`last_used` busy-counter — RLock-guarded because the supervisor reads it off-thread, `now_fn`-injectable for deterministic tests.

- **`OllamaBackend`** — `chat_url` → `{base}/api/chat`, `health_url` → `{base}/api/tags`. `ensure_ready` is a no-op (the daemon cold-loads/hot-swaps on the first request itself), `is_running()` always `True` (no I/O — that's the host node's call to make), `stop_server()` always `False` (the engine never manages the daemon, so the idle supervisor's stop attempt is a truthful no-op).
- **`VLLMBackend`** — `chat_url` → `{base}/v1/chat/completions`, `health_url` → `{base}/health`. State machine `DEAD → LOADING → SERVING`, plus `RELOADING`/`SLEEPING_L2`/`UNSUPPORTED_SLEEP` for a model switch. vLLM pins one model in VRAM for its whole process lifetime with **no built-in idle unload** and **no hot model swap** — the intended fast path is vLLM **Sleep-Mode L2** (offload weights to host RAM, reload the new model into the same process), but `_sleep_l2_reload` is a deliberate stub (`raise NotImplementedError`); `ensure_ready` catches that once, sets a **sticky** `_sleep_unsupported` flag, and falls back permanently to the slow path (stop → cold-start the new model). After bring-up, the served model id is reconciled against a `/v1/models` probe and a **mismatch raises `RuntimeError`** (fail loud) — because the default `ensure_up_fn` is a no-op that relies on docker `restart: unless-stopped` bringing the sidecar back up on its **baked** `--model` flag, which may not be the newly-requested one; a consumer should soft-degrade (e.g. fall back to ollama) rather than silently POST to the wrong model.
- **`LLMSupervisor`** — mirrors `ModelCache`'s idle reaper exactly: pure decision `vllm_should_stop(running, inflight, idle_s, ttl_s)`, poll cadence `clamp(idle_ttl_s/5, 5s, 60s)`, gated by `QUEUE_WORKFLOWS_DISABLE_LLM_SUPERVISOR`. **Inert** for any non-`"vllm"` `server_type` and for `idle_ttl_s <= 0` — ollama is never touched. It only decides *when*; the actual kill is `backend.stop_server()`.

### `BackendFactory` — identity preservation + freshness

`get_backend(host_label, queue)` must hand back the **same instance** across calls as long as the config snapshot is unchanged — the backend's request counters and the vLLM state machine live *on* that instance, so a spurious rebuild would silently reset the idle clock and the served-model belief. The snapshot is `(server_type, parallelism, vllm_idle_ttl_s)`; on a genuine change, the old entry is torn down (`supervisor.stop()` then `backend.shutdown()`) and a new one built.

Two ways a re-read is triggered:

- a **TTL** throttle (default 10 s, `DEFAULT_CONFIG_TTL_S`) — the dropped-NOTIFY fallback;
- a **LISTEN** invalidator thread (`start()`/`stop()`, gated by `QUEUE_WORKFLOWS_DISABLE_LLM_CONFIG_LISTENER`) on the `worker_llm_config_changed` channel — instant refresh the moment an operator edits the config.

Server-root URLs are **not** part of the snapshot — they're deployment topology, resolved once from env at build time (`config.ollama_url_env` = `QUEUE_WORKFLOWS_OLLAMA_URL`, `config.vllm_url_env` = `QUEUE_WORKFLOWS_VLLM_URL`; a redeploy, not a runtime reconfig, is how a URL changes). vLLM's stop/start seams (`cfg.vllm_stop_fn` / `cfg.vllm_start_fn`, wired via `set_vllm_lifecycle(stop_fn, start_fn)`) let a host that runs vLLM as a **separate container** (e.g. via the docker Engine API over a unix socket) have the in-worker supervisor control the sibling sidecar without relying on a docker restart policy; `None` (default) falls back to the backend's own `pkill`/no-op seams. See [configuration](configuration.md) for the hook-setter list.

### Per-machine server config: DB owns *which*, env owns *where* (migration 0013)

`worker_controls` (already the ON/OFF control-plane row, migration 0012 — see [worker_control.md](worker_control.md)) gains three operator-set columns:

| Column | Meaning | Default |
|---|---|---|
| `llm_server_type` | `'ollama'` \| `'vllm'` | `'ollama'` |
| `llm_parallelism` | the **sidecar's** concurrent-request capacity (ollama `OLLAMA_NUM_PARALLEL` / vLLM `--max-num-seqs`) — **not** the claim-worker concurrency, which stays 1 by contract | `1` |
| `vllm_idle_ttl_s` | seconds of zero requests before the supervisor SIGTERMs the vLLM sidecar; ignored for ollama | `60` |

A row trigger fires `pg_notify('worker_llm_config_changed', '<host>|<queue>')` — a **distinct** channel (`|` separator) from the 0012 `worker_control` channel (`:` separator), so an LLM-config edit is never mistaken for an ON/OFF change by the hard-stop `WorkerControlWatcher`; the trigger stays quiet on an UPDATE that touches none of the three LLM columns. Read via `worker_control.llm_config_for(host, queue)`, written via `worker_control.set_llm_config(...)` (a partial `COALESCE` upsert that never touches `desired_state`).

The **root URL** is deliberately *not* in this table — it's per-host deployment topology (env, or the per-box YAML below), while *which server type* + its tunables is centrally operator-visible/editable state (DB). This split is why the factory's identity snapshot excludes the URL.

### Per-box topology: which address each box dials (`llm_topology_path`)

One env holding the URL is fine when every box talks to the same server, but a real fleet wants a **different address per box** — each keyed by its own `host_label`.

Set `configure(llm_topology_path="…/llm_topology.yml")` and the backend factory resolves a box's LLM server ROOT URL **by its `host_label`** from that YAML, in preference to the env — so the engine hands each box the address of *its own* server. The file is deployment topology (real hostnames/IPs): keep the live one gitignored and commit a `.example` (see [`llm_topology.example.yml`](llm_topology.example.yml)).

```yaml
# llm_topology.yml — host_label → LLM server ROOT URL that box dispatches to
boxes:
  box-a-gpu:  http://host.docker.internal:11434  # serves its own model
  box-b-gpu:  http://host.docker.internal:11434  # serves its own model
  box-c-gpu:  http://host.docker.internal:11434  # serves its own model
  default:    http://host.docker.internal:11434  # any box → itself
```

- **The rule: every GPU box serves its OWN model and dials itself** (`host.docker.internal`, the docker host gateway to the box's own server). Do **not** point one box at another's server — a single ollama serialises requests per model, so N boxes sharing one server queue behind each other while the borrowers' GPUs sit idle *and* still hold a claim slot showing `running`. Pointing several boxes at one server adds *clients*, not capacity. (This is the failure `llm_probe` was written to surface — see [Observed capability](#observed-capability-migration-0014).) If a box has no GPU, don't give it a `-gpu` lane at all.
- Keyed by the box's own `host_label` (the value it exports as the engine's host-label env, e.g. `QUEUE_WORKFLOWS_HOST_LABEL`) — so a per-lane label like `box-b-gpu` is the key, not the bare hostname.
- Server-type-agnostic — the same entry feeds both the ollama and vLLM resolvers; the DB still decides which type runs there.
- **Opt-in + byte-compatible**: unset `llm_topology_path` (default), a missing file, or an unmatched box ⇒ the factory uses the env + `host.docker.internal` default — which is already "serve yourself", so the safe default matches the rule. A broken file never breaks dispatch (it logs + falls back).
- The YAML is cached by `(path, mtime)`, so an operator edit needs no restart to be *re-read* — but `BackendFactory` caches the built backend per `(host, queue)` and only invalidates on the DB LLM-config NOTIFY, so a **running worker keeps its current URL until that entry is rebuilt**. Restart the worker when you want a topology edit to take effect immediately.

### Observed capability (migration 0014)

`worker_heartbeats.llm_servers_available text[]` (default `{ollama}`) is the **observed** counterpart — which server types *this machine can actually run* — re-advertised every heartbeat. A queue UI reads it to **gate** its per-machine server-type control: disable the vLLM toggle on a host with no sidecar for it, so an operator can't route VLM calls at a non-existent endpoint (vLLM, the CUDA `vllm/vllm-openai` sidecar, runs only on NVIDIA boxes, never on a ROCm box).

**"Observed" means *probed*, not *declared* (`llm_probe`).** The column DEFAULTs to `{ollama}` and originally the worker just re-published a static config value that *also* defaulted to `["ollama"]` — so the field was a guess dressed as a fact. A GPU box running **no** LLM server still advertised `{ollama}`, showed as an ollama server in any UI that reads the column, and (via a borrowing topology entry) dispatched every call to another box while its own GPU sat idle — invisibly, because nothing ever asked the endpoint. The GPU worker now wires `HeartbeatEmitter(llm_servers_fn=…)` to `llm_probe.probe_llm_servers(base_url)`, which GETs the box's resolved endpoint and advertises the server that **actually answers** — `[]` when none does. An empty list is the whole point: it makes "this box serves no model" a *visible* state, and the worker logs an ERROR (can't serve) or WARNING (dispatching to a remote box) at startup so the wiring is diagnosable from the log. `is_local_endpoint()` distinguishes serving-yourself from borrowing; the probe is stdlib-only (psycopg stays the one hard dep), HTTP-injected for tests, and never raises. cpu/ingest workers leave `llm_servers_fn` unset and keep publishing the config value unchanged.

Both 0013 and 0014 are default-safe: a DB predating them (or an accessor call before they're applied) returns the all-defaults config / the `{ollama}` baseline, so the engine and any other consumer runs unchanged.

### Per-dispatch server resolution

`set_llm_server_resolver(resolver)` lets a host resolve the exact server (type + address + port) **per node-job** at dispatch time; `node_executor` threads the result in as `run(llm_server=...)`, so a node body never hardcodes a URL — it always asks for the backend the resolver decided on for that job.

### The context window (migration 0028)

**Why it is a routing fact and not a request parameter.** ollama does not refuse a prompt longer than the window its runner was loaded with — it **truncates it and answers HTTP 200**. Measured on 0.33.2: a ~12,000-token prompt at an 8192 window came back with `prompt_tokens` 4098, no error, no warning, a confident answer to half a document. Nothing downstream can detect that, so the window has to be decided **before the claim**: a node-job declares `context_length`, and the claim gives that row only to a box whose server serves at least that much. A box that cannot state its window claims only unsized rows.

**What each box answers with.** Two heartbeat columns, recomputed every ~10 s by the GPU worker:

| column | what it is | where it comes from |
|---|---|---|
| `llm_context_length` | the window this box serves **now** | ollama: the value its container is CONFIGURED with (`OLLAMA_CONTEXT_LENGTH`), clamped to the model's own maximum — *not* a loaded runner's, because another client can load one natively at a bigger window and our very next `/v1` call reloads it back. vLLM: `max_model_len` from `/v1/models`. |
| `llm_context_max` | the biggest it could **switch** to | The memory the box has free (`MemAvailable`) plus what the current runner would release, minus an operator reserve, minus the weights, divided by what one token of window actually costs — capped by the model's maximum. vLLM instead: its KV pool is sized by `gpu_memory_utilization`, so a bigger window costs no new memory and only has to fit the pool it already owns. |

A box serving two kinds advertises the **smaller** window (a sized job may resolve to either) and the **larger** ceiling (that is what a switch could reach). A box that cannot grow — no lever, the kill switch, or a resize that already failed here — advertises a ceiling equal to what it serves, so the fleet does not wait for a switch that cannot happen.

**What a token of window costs — measured, not assumed.** The textbook f16 KV formula is wrong in both directions on real models, so the engine prefers the OBSERVED cost of a loaded runner (`(runner_bytes − weights) / window`, an average that bounds the marginal cost from above) and keeps the formula only as the fallback, with a ×1.25 safety factor:

| model | observed | the formula says |
|---|---|---|
| a standard transformer (qwen2.5:0.5b) | 14–18 KB/token at the margin | 12,288 B — 15–30 % **low** (compute buffers grow with the window too) |
| a hybrid (full attention every 4th layer) | ~3.7 KB/token | 64 KiB — ~18× **high** |

**Growing into demand.** `node_queue.context_switch_need` is the demand read every box runs against the same heartbeats, so they elect the same box without talking to each other: a resize happens only when **nobody** is already serving that window (a box that can take the job will, within seconds), the job has waited (below ~30 s "nobody serves it" is as likely to be heartbeat lag), and of the boxes that could reach it this is the **tightest fit** — so the roomiest box stays free for the job that needs the room. Placement (0020) and tenancy bind exactly as they do in the claim, and a parked or stale box neither counts nor masks the demand.

**Performing it.** A resize reuses the `ServerSwitch` (`docs/watchdogs.md`, §"one server at a time"): the box drains, blocks claims, and comes back when its server ANSWERS at the new window — with two differences. Nothing is unloaded (the lever replaces the container in one operation), and a failed resize quarantines *resizing that kind*, never the kind itself, so a box whose window cannot be changed can still switch to that server normally. If the resize leaves the server not answering at all, the operator's own window is restored — the one configuration the box is known to run.

The lever itself recreates the box's container with one value swapped (`OLLAMA_CONTEXT_LENGTH` / `--max-model-len`), from the container's **own** inspect output, carrying its network endpoints over explicitly: docker's create body does not inherit them, and a `broker-ollama` that comes back without its aliases is a box whose LLM service is gone while the container looks healthy. The operator's original value is remembered in a container label, so "no queued job needs a big window" restores it rather than freezing the last override forever. A deployment whose server is not a local container wires `set_llm_context_lever(fn)` instead; a box with neither lever nor docker socket simply keeps the window it has.

**When a server is started anyway** — a kind switch, or a cold vLLM start — it comes up at the highest window the queue needs and can reach, rather than coming up wrong and being restarted a minute later. No sized work queued ⇒ its configured default, exactly as before.

**A job nobody can serve** is red-flagged by the existing `unassignable` sweep (0015), and the reason names the window, the largest any box could reach, and the fact that the answer was taken against each box's free memory. Model and window are judged **on the same box**: one box having the model and another the window must never pass a job nobody can run.

**Knobs**: `QUEUE_WORKFLOWS_DISABLE_LLM_CONTEXT_SWITCH` (off switch), `QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB` (default 8192), `QUEUE_WORKFLOWS_OLLAMA_CONTAINER` / `QUEUE_WORKFLOWS_VLLM_CONTAINER`, `QUEUE_WORKFLOWS_VLLM_MAX_CONTEXT` (a vLLM box cannot grow without it — vLLM does not publish its model's maximum, and exceeding it makes the server exit at startup).

### Operational notes

- One machine runs **one** server type at a time — that is what the `llm_server_type` toggle means.
- **…and it is the only kind offered while vLLM serves.** ollama's daemon stays up after a switch (only its models unload), so a box used to advertise both kinds — and an ollama-only job then loaded a second copy beside vLLM's pool, until the box was out of memory. While vLLM serves, the heartbeat advertises `{vllm}` alone.
- **No load into memory the box does not have.** ollama loads on the first request, so the claim refuses a job whose model is not resident here unless weights + cache fit in `MemAvailable` minus the reserve (`QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB`, 8 GB). vLLM's `--gpu-memory-utilization` — a fraction of the WHOLE machine on a unified-memory box — is lowered to what is free at start (the operator's value is remembered and never exceeded), and vLLM is not started at all below `QUEUE_WORKFLOWS_VLLM_MIN_UTILIZATION` (0.30). Raise the reserve on a box that runs wide cpu lanes or other GPU tenants.
- **…and one daemon of it.** The residency enforcer only sees the servers it was configured with; a second `ollama/ollama` (or `vllm/vllm-openai`) container on another port is invisible to it, and its model is a second resident copy (e.g. a benchmark's ollama beside the box's own: two copies of one model in one box's memory). `rogue_llm.sweep` therefore stops such a container on sight, every tick, from whichever engine process holds the docker socket — the gpu claim worker's heartbeat, and the orchestrator's sweep loop. Run benchmarks against the box's own daemon or on another box; `QUEUE_WORKFLOWS_ROGUE_LLM_GUARD=report` if you only want it named.
- **…and, on a box that is the broker's, nothing else on the GPU.** Opt-in: with `QUEUE_WORKFLOWS_GPU_CONTAINER_ALLOW=<name prefixes / compose projects>` set, the same sweep stops any other RUNNING container with GPU access that dockerd would bring back by itself (`always`/`unless-stopped`/`on-failure`) and pins it to `restart=no` (such a container can take the LLM server's host port after a reboot, and its memory is invisible to admission), while a hand-launched run (`restart=no`: a training job, a benchmark) is named once and left to share the box — presence is not the offence, MEMORY is. When the box crosses its danger floor the sentinel sheds the broker's own servers first, re-reads, and only if it is still short stops every un-named GPU container (`rogue_llm.stop_gpu_outsiders`). `QUEUE_WORKFLOWS_ROGUE_GPU_RUNS=stop` removes runs on sight instead. The engine's own containers are always allowed; unset, the rule is off. A parked gpu worker sweeps too.
- **vLLM sized to what it needs; concurrency counted from what fits (0029).** The pool (`--gpu-memory-utilization`, a fraction of the WHOLE machine on unified memory) used to be sized to what was free. Now the worker reads, once per start, what vLLM itself printed — `Model loading took … GiB`, `Available KV cache memory`, `GPU KV cache size: … tokens`, `Maximum concurrency for … tokens per request` — caps the box's PAR at that concurrency (a cache that holds fewer full-window requests than the box's PAR would otherwise feed it), and remembers the numbers in `llm_model_catalog.profile` under the command line's fingerprint. The next start is planned (`llm_context.vllm_plan`): pool = weights + fixed overhead + N × window × per-token (+5 %), capped by what is free minus the reserve and by the operator's value; N lowered to what fits and passed as `--max-num-seqs`; a box that cannot hold one request of the window refuses, naming the numbers. A model's first start on a box keeps the old rule and becomes its profile. The startup lines are read by TIME (the half hour from the container's `StartedAt`), not by `tail=`, because a serving vLLM's request log buries them. A server already RUNNING whose startup was observed complete is exempt from a start-verdict — it is judged by whether the box is in danger (`MemAvailable` under the absolute `QUEUE_WORKFLOWS_LLM_DANGER_FLOOR_MB`, 12 GB — never a multiple of the reserve, which is a planning margin), not by whether a start it is not making would fit; one still LOADING into memory the box lacks is stopped at any reading.
- vLLM cold start is real (roughly 1–4 min for a 7B AWQ on a modern GPU); the supervisor's idle-stop trades that cold start for reclaimed VRAM. ollama is always up but serializes by default (raise `OLLAMA_NUM_PARALLEL` for concurrency).
- vLLM has no hot model swap — a model switch is a server restart; ollama hot-loads a new tag on the running daemon.
- A vLLM server needs its model **fully present** in that host's own model cache before it can serve; an incomplete download makes the loader block (looks like a hang).
- **While a server WARMS the box keeps being watched.** The pool is decided once per switch (re-deciding against the memory the start itself is consuming would recreate it in a loop), but the load is still watched: a crossing of the danger floor while it warms stops the loading server and abandons the switch — nothing is recomputed or recreated.
- **A box out of memory sheds — it is never parked.** The gpu lane's `MemorySentinel` reads `MemAvailable` once a second; under the danger floor (12 GB) it stops the vLLM container (serving or loading), unloads every model ollama holds and abandons a warm in flight, at most once per 60 s; under the kill floor (512 MB) it does so at once. The lane stays ON so the next start is planned against what is then free and everything that fits keeps flowing. Parking is an operator lever and nothing in the engine writes `worker_controls` — a test forbids it.
- **When RAM has been back for a minute, vLLM may come back.** A hold the plan placed over memory (every such refusal is tagged `memory: `) is lifted by the sentinel once `MemAvailable` has covered what a worthwhile vLLM needs here — weights + overhead + the minimum concurrency's worth of cache (+5 %) + the reserve, or the minimum pool + the reserve without a profile — for 60 s continuously; the demand logic then requests the switch and the plan sizes the start to what is free. A dip restarts the count; holds for other reasons (a server that never became ready) are left alone.
- **A load killed mid-way raises the bar.** A vLLM load can take many minutes, so a kill is evidence, not just a loss: each one adds 25 % (capped at 1.5x) to the headroom the next start — and the memory-recovery lift — demands over the bare need, until a start completes and clears the history. Kills report what they cost ("4.7 min into the load").

### Coexisting with the diffusion / model-cache lane (§1)

A diffusion (or SR/CV) model **never** goes through ollama or vLLM — those only serve language/vision-language models. A model-backed GPU node runs in-process in the claim worker's *inline* lane (concurrency-1, the warm `ModelCache` from §1); the LLM server, if any, runs *beside* it on the same physical GPU, fed only by a separate *pool* lane of VLM-only jobs (`require_model IS NULL`). The two lanes share one GPU, so a machine's total concurrent node-jobs is capped at `PAR` (the sidecar's `llm_parallelism`), and while an inline diffusion job runs it occupies one of those `PAR` slots (`PAR − 1` budgeted to the pool feeder). Running a resident LLM server *and* a diffusion model on one GPU competes for VRAM — this is precisely why the vLLM supervisor idle-stops the server (freeing VRAM for diffusion between VLM bursts) and why the diffusion side's own idle TTL (§1, default 60 s) is short.

---

## See also

- [architecture.md](architecture.md) — the claim SQL, lease/reclaim, and the process roles the GPU worker plays.
- [watchdogs.md](watchdogs.md) — the wall-clock, stall, and GPU-health watchdogs that police a *running* GPU job (a liveness concern, distinct from this doc's residency/capacity concerns).
- [worker_control.md](worker_control.md) — the operator ON/OFF plane that `worker_controls` (this doc's migration 0013 columns live on the same table) also implements.
- [configuration.md](configuration.md) — every hook-setter (`set_builtin_model_registrar`, `set_llm_servers_available`, `set_vllm_lifecycle`, `set_llm_server_resolver`) and env knob named above.
