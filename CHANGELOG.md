# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.6] — 2026-07-19

### Added — ComfyUI as a first-class queue citizen
- **Built-in `comfyui` render node** (`queue_workflows/builtin_nodes/comfyui.py`)
  — one node_job = one whole ComfyUI render: stage inputs, POST the API-format
  graph, poll to completion, collect the output. The reusable submit client +
  box-acquisition live in `queue_workflows/comfyui.py` (`ensure_comfyui_box`
  clears rival server kinds off the card, optionally boots the server via the
  host-wired `set_comfyui_lifecycle`, then ready-probes).
- **Pool-lane fixes for ComfyUI boxes**: a box serving ComfyUI (no LLM server)
  can claim no-model ComfyUI render jobs, and never defers them to an
  LLM-capacity peer (the fill-before-spill gate is LLM-work-only — deferring a
  render to a box without ComfyUI starved it forever).

### Added — per-box supervisor (`box_agent`)
- **`queue_workflows/box_agent.py` + `queue-box-agent` console script** — ONE
  agent per physical box that treats `worker_controls` as the single source of
  truth and reconciles the box's actually-running worker lanes to match it
  every 30 s: start a desired-on lane that isn't running, stop a desired-off
  one, restart one that died. Lanes are declared in a JSON manifest (argv +
  env per lane; an env value of `null` REMOVES the inherited key, so a lane on
  a pinned baked engine never silently inherits the agent's own `PYTHONPATH`
  tree). Control rows are fetched per-lane-DSN FAIL-OPEN (an unreachable
  control DB defaults its lanes ON), tolerate a pre-0019 schema (no `project`
  column), and SIGTERM/SIGINT forward to every lane so claim workers drain
  gracefully when the container stops. Ends the "parked in compose but the DB
  says on" split-brain: whether a lane runs is decided in exactly one place.

### Added — operator control
- **`stop_policy="drain"`** — parking a worker now has a keep-the-last-job
  mode: the control watcher FLAGS the worker (process lives, watcher stays
  armed so an escalation to `hard` still lands), and the claim loop parks at
  its next iteration boundary — between jobs, never mid-job — via the existing
  park-until-ON gate. The in-flight job and the box's inference server are
  untouched (the server-stop moved inside the `hard` handler). The GPU pool
  feeder also stops claiming while draining. Proven end-to-end on live
  LISTEN/NOTIFY: a gate-held in-flight job completes with zero requeues, new
  work stays unclaimed while parked, and the ON flip resumes claiming
  (`tests/test_drain_end_to_end.py`).

### Added — reliability
- **Over-claim reaper** — the runtime safety-fallback for the per-box slot
  arbiter: if a box is ever RUNNING more gpu jobs than its advertised
  `worker_heartbeats.concurrency` (a bypassed claim path, a counter bug), a
  NodePool sweep re-queues the NEWEST over-capacity job(s) resume-style (keep
  the oldest running; no `watchdog_retries` bump; CAS so a just-finished job is
  never yanked; a terminal parent cancels instead). Interval-gated (10 s), ON
  by default, `QUEUE_WORKFLOWS_DISABLE_OVER_CLAIM_REAPER` to disable.

### Added — telemetry
- **Fan speed in the `hw_watch` flight recorder** — every hwmon `fan*_input`
  tachometer (chassis/CPU/GPU) plus the GPU's own `fan.speed` %; `deep_sample`
  derives `fan_rpm_max` / `gpu_fan_pct` graph-ready. `None` ⇒ no fan sensor
  (distinct from a powered-off box, which writes no row at all).
- **CPU/SoC temperature on ARM boards** — the CPU temp probe falls back to the
  `acpitz` thermal zone on SoCs whose hwmon exposes no coretemp/k10temp analog.

### Fixed
- **A no-model ComfyUI render is its own box-lease server kind.** The
  cross-project box lease keyed EVERY no-model job to the shared
  `__llm_server__` slot, so a ComfyUI render and an ollama dispatch on one box
  were treated as the same "model" and co-resided on the card. A
  ComfyUI-serving worker's no-model job now keys to a distinct
  `COMFYUI_SERVER_SLOT`, so the two kinds mutually exclude (while two ComfyUI
  renders still share); `_model_still_resident` probes ComfyUI residency when
  a URL is configured, else releases at job end via the active-jobs guard.

## [1.0.5] — 2026-07-18

### Added
- **CPU hardware stats in the `hw_watch` flight recorder** — `deep_sample()`
  gains a `cpu` block (stdlib `/proc` + `/sys`, no `psutil`): utilisation %,
  mean core frequency, package temperature (coretemp / k10temp / zenpower /
  cpu_thermal), and the summed core thermal-throttle counter (the CPU analogue
  of the GPU violation counters). On a unified-memory SoC the CPU shares the
  die and power budget with the GPU, so its thermals belong in the same
  box-health record. Best-effort per sub-probe — a missing sensor yields
  `None`, never a raise.

### Fixed
- **Per-node-job box placement (`avoid_box` / `force_box`) now reaches the
  enqueued node-jobs.** The dispatcher's `_nodes_of` dropped the placement
  fields when expanding a run, so migration 0020's constraints never made it
  onto the rows; they are now threaded through.

### Docs
- README: a screenshot of an example operator front-end built directly on the
  engine's telemetry (the `hw_watch_samples` flight recorder + `worker_heartbeats`
  + `worker_controls`) — per-box GPU/CPU temperature, power, clock, and throttle.

## [1.0.4] — 2026-07-17

### Added — one model per physical GPU box (cross-project arbitration)
- **Run-path box lease** (`gpu_model_lease`) — every GPU worker acquires the
  physical box for its effective model before executing; a live peer holding a
  DIFFERENT model makes the job spill (re-queue to a free/matching box), while
  the same model is shared. Coordinates across projects and across uids (root
  containers + non-root host processes) through one box-local flock file, made
  world-writable so mixed-uid participants can all join.
- **Per-box slot arbiter** — `claim_next_gpu_job(max_running=…)` refuses an
  over-claim in the claim SQL itself, so per-box concurrency is DB-enforced
  rather than living in per-lane in-process counters.
- **Residency enforcer + last-defence load gate** (`model_residency`,
  `ModelCache.pre_load_check`) — a box holds at most one distinct model / one
  server KIND (ollama / vLLM / ComfyUI / in-process); a second is evicted via
  its own lever (or the load is refused) and reported in the daemon log.
- **Idle external-LLM unload** — after a box does no GPU work for
  `QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S` (default 300 s) its GPU worker unloads
  the resident LLM to free unified memory; reloads on demand.
- **Box-addressed worker-control rows** — `get_worker_control` / `llm_config_for`
  fall back to a `(box, queue)` row so one control governs every lane on a box.

### Added — per-node-job physical-box placement (migration 0020)
- `workflow_node_jobs.avoid_box` / `force_box` (`text[]`, NULL = unconstrained)
  let a queued job be pinned to or excluded from physical boxes by name; the
  cpu+gpu claim SQL ANDs the constraint in so an ineligible box never claims.
  Threaded through `enqueue_node_job`, the dispatcher, and the worker's box
  identity. See `docs/box_placement.md`.

### Added — hardware flight recorder (migration 0021)
- **`hw_watch`** — opt-in (`QUEUE_WORKFLOWS_HW_WATCH=1`) two-tier persisted
  hardware telemetry (`detail` 2 s / 1 h + `history` 60 s / 24 h): per-GPU
  temperature, power draw (average + instantaneous), SM clock, utilisation,
  throttle-reason mask, and the nvidia-smi **Violation** counters (accumulated
  per-cause throttle time), every thermal zone, hwmon temps, meminfo, load and
  disk. Best-effort writes (own connection, swallow-on-failure) driven by the
  existing sampler thread or the standalone `queue-hw-watch` CLI; pruned on a
  NodePool sweep. `throttle_index()` collapses the mask to a 0–100 severity;
  `power_brake()` flags the power-delivery echo. See `docs/hw_watch.md`.

### Reliability
- **Fast zombie/ghost reclaim** — worker-boot self-reclaim, orchestrator
  flagged-worker fast-reclaim (~30 s), and a **frozen-lease sweep** that
  re-queues a `running` row whose lease stopped renewing (~35 s) instead of
  waiting out the full lease — catching the fast-restart case heartbeat-keyed
  paths can't see. Orphan-cancel of terminal-run queued jobs now defaults on.
- **Anti-starvation break** for the fill-before-spill VLM-pool gate — a
  draining top box never lets claimable work age past
  `QUEUE_WORKFLOWS_VLM_POOL_MAX_DEFER_AGE_S` (default 20 s), so a busy/blind
  top box can't starve the fleet (deadlock-proof).
- Box lease is held for a job's whole lifetime (counter-based), flicker-proof
  against a transient residency blip.

## [1.0.3] — 2026-07-15

### License
- **Relicensed to the GNU Affero General Public License v3.0 or later**
  (`AGPL-3.0-or-later`) — was PolyForm Noncommercial 1.0.0 in 1.0.2, MIT before
  that. Matching [Open-Meteo](https://open-meteo.com/)'s licensing: a real
  open-source (OSI-approved) copyleft license, so **commercial use is now
  allowed**, on the condition that derivative works stay AGPL and
  source-available — including when offered over a network (AGPL §13). Prior
  releases remain available under the terms they shipped with (≤ 1.0.1 MIT,
  1.0.2 PolyForm Noncommercial).

### Added — per-box LLM serving
- **Per-box LLM-server topology** (`configure(llm_topology_path=…)`) — an
  optional YAML maps each worker box (by its `host_label`) to the LLM server ROOT
  URL it dispatches to, so a fleet can give every GPU box its own local server
  instead of pointing them all at one env-configured URL. The backend factory
  prefers it over `ollama_url_env` / `vllm_url_env`, keyed by the box's label;
  `resolve_base_url()` exposes a box's URL without building a backend. Opt-in +
  byte-compatible (unset / missing file / unmatched box ⇒ the env + localhost
  default). See `docs/gpu_and_llm.md`.
- **Observed LLM-server capability** (`queue_workflows.llm_probe`) — the GPU
  heartbeat now PROBES its resolved endpoint and advertises the server type that
  actually answers, publishing `worker_heartbeats.llm_servers_available = []`
  when none does, instead of a static default that could report a server the box
  wasn't running.
- **ONLY-GPU claim gate** — a GPU worker whose LLM nodes dispatch to an external
  server now verifies that server is running the model on the GPU
  (`probe_gpu_placement` reads ollama `/api/ps`). If it has fallen back to CPU (a
  lost GPU device, or a model too large for VRAM), the no-model pool lane STOPS
  claiming LLM jobs so they route to a GPU-backed box, and resumes automatically
  once the server is back on the GPU. The only other skips remain insufficient
  VRAM (capacity gate) and the operator OFF toggle.
- **GPU toggle governs the inference server**
  (`set_inference_server_lifecycle(start_fn, stop_fn)`) — turning a box's GPU
  worker OFF now also stops the machine's LLM server (freeing VRAM), and turning
  it ON starts it. GPU-lane-only, best-effort, and a no-op unless wired.

### Changed
- **One-model-per-GPU-box lease wired into `ModelCache`** — the arbitration
  primitives added in 1.0.2 are now active on the warm-model load path (they were
  opt-in primitives only). Still inert without a configured lease store, so the
  default behavior is unchanged.
- New optional extra **`topology`** (PyYAML) — required only to parse the per-box
  topology YAML; psycopg remains the sole hard runtime dependency.

## [1.0.2] — 2026-07-13

### License
- **Relicensed to PolyForm Noncommercial 1.0.0** (was MIT). Free for any
  noncommercial purpose; commercial use requires a license from the author.
  Versions ≤ 1.0.1 were published under MIT and remain available under those
  terms.

### Added
- **`queue-worker-supervisor`** — a per-host daemon that closes the dead-worker
  loop: the orchestrator's detector (0009 `last_flagged_dead_at`) flags a wedged
  worker but deliberately never kills it; this optional supervisor reads the
  flag for the `host_label`s its box owns (a `label:container` map via `--map` /
  `QUEUE_WORKFLOWS_SUPERVISOR_MAP`) and `docker restart`s the local container.
  Report-only without a map; per-`(host, queue)` cooldown; injectable bounce
  (`set_worker_bounce`) for systemd/k8s hosts. Plus
  `node_queue.flagged_dead_workers()`, the read side of the flag.
- **One-model-per-GPU-box arbitration primitives**
  (`queue_workflows.gpu_model_lease`) — a box-wide model lease so N GPU workers
  (one per project) sharing one card can't warm different models concurrently:
  pure `decide()` (empty grants / same model shares / different model denies
  unless the holder's lease expired), a flock'd `FileLeaseStore` shared across
  containers, `set_gpu_lease_store()` for custom stores, and
  `QUEUE_WORKFLOWS_GPU_{BOX_ID,MODEL_LEASE_DIR,MODEL_LEASE_TTL_S}` knobs.
  **Opt-in and inert by default** (no store ⇒ every load grants, byte-identical
  behavior); not yet wired into the runtime load path.

## [1.0.1] — 2026-07-12

### Changed
- **Runtime env knobs are canonically `QUEUE_WORKFLOWS_*`.** Every knob
  (`QUEUE_WORKFLOWS_DB_URL`, `QUEUE_WORKFLOWS_HOST_LABEL`,
  `QUEUE_WORKFLOWS_GPU_CONSUMER_PRIORITY`, `QUEUE_WORKFLOWS_OLLAMA_URL` /
  `_VLLM_URL`, and the tuning knobs) now reads through a single compat helper
  (`queue_workflows/envcompat.py`) whose lookup order is **canonical →
  legacy → default**. The pre-1.0 `AI_LEADS_*` spellings keep working as a
  silent fallback, so an existing deploy upgrades with **zero `.env` changes**;
  the canonical name wins when both are set. A source-scan test forbids any new
  direct legacy read. `EngineConfig`'s env-name field defaults are the
  canonical spellings; `container_prefix` stays the value `"ai_leads-"` (a
  cgroup-attribution string, not an env name — override per host).

### Added
- **Connection-pool sizing seam.** `QUEUE_WORKFLOWS_DB_POOL_MIN` (floor,
  default 1) joins `QUEUE_WORKFLOWS_DB_POOL_MAX` (cap, default 10); the floor
  is clamped into `[0, max]` so no env combination can crash pool
  construction. The cap is the connection-budget lever for an N-process claim
  lane (`lane_processes × max` bounds the Postgres backend count).
- **`queue-broker-web` operator panel — visual refresh.** A warm, tokenized
  design language (amber primary, ink text, carded tables, bordered pill
  status badges), a `created` column, timestamp rendering robust across the
  SQLite (TEXT/ISO-8601) and Postgres (`timestamptz`) backends (a missing
  value renders `—`, never a blank cell), and status-tinted queue counts.

### Docs
- Ten worked operational use-case scenarios under `docs/use_cases/` (box
  power-loss and the front-of-queue requeue + zombie kill signal, boot/rejoin,
  operator stop, wedged-GPU recovery, DAGs, ingest, multi-project broker,
  warm-model affinity, run-next, human-in-the-loop).
- `README` gains an "Architecture at a glance" section (Mermaid system / job
  lifecycle / durable-outbox diagrams) and a screenshot of the operator panel.
- Host-neutral reaudit of the docs set (generic example project names; the
  `AI_LEADS_*` origin story reframed as the legacy-fallback contract).

## [1.0.0] — 2026-07-11

Initial public release: **broker_parrot** (import package `queue_workflows`) —
a standalone Postgres/SQLite-as-queue workflow engine. `SELECT … FOR UPDATE
SKIP LOCKED` claim loop woken by `LISTEN`, lease reclaim, DAG dispatcher with a
durable outbox, GPU warm-model cache, watchdog stack, durable per-node event
log, operator worker ON/OFF control plane, multi-tenant `project` scoping, and
pluggable storage backends (SQLite default, Postgres, plus the opt-in
Redis/MongoDB flat-queue SPI). Postgres via `psycopg` 3 is the only hard
runtime dependency for the `pg` backend; the SQLite default runs daemon-less.

The `queue-broker-web` control plane ships hardened: fail-closed non-loopback
bind (`QUEUE_WORKFLOWS_BROKER_WEB_TOKEN` must be a real secret — set, not a
placeholder, ≥ 16 chars — or the service refuses to start; loopback stays
open), timing-safe bearer comparison, a 1 MiB request-body cap (`413`), a
bounded request thread pool (`QUEUE_WORKFLOWS_BROKER_WEB_MAX_WORKERS`, default
32), and sanitized error responses.
