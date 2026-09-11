# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.0] — 2026-09-11

A feature release on from 1.1.0. Migrations 0025–0027 are additive (one dropped
`CHECK`, nullable columns, a new table) and new features default to off or to the
old behaviour — **except** two deliberate changes: what a process does when the
*database* is the problem, and the env names the reference lane script reads. Read
*Changed — an unreachable database is waited for* and *Changed — the reference lane
supervisor's knobs* before upgrading a supervised fleet. Versions
1.2.0–1.3.2 were internal builds that were never published; everything they
contained is listed here.

### Added — the `download` node-job queue (migration 0025)

- **A third first-class node queue, `download`** — the network-WAIT lane. A
  pipeline schema marks a node `download: true` and the dispatcher routes it to
  `queue='download'` (full DAG citizenship: `depends_on` edges, `force_box`
  placement, lease/watchdog/outbox; `download`+`gpu` on one node is a loud
  expansion error). Downloads no longer burn a cpu/gpu claim slot on
  wall-clock.
- **One downloader at a time, engine-enforced**: the download claim
  (`claim_next_download_job`) carries a **fleet-wide per-project width cap**
  (`download_max_concurrent`, default **1**, env
  `QUEUE_WORKFLOWS_DOWNLOAD_MAX_CONCURRENT`) — counted with no per-host scope
  (any box's running download occupies the lane; the shared uplink is the
  protected resource) and serialized by a (queue, project)-keyed advisory
  lock so two simultaneous claimants can't both take the last slot. A dead
  downloader's slot frees via the ordinary lease reclaim.
- **"One file is never downloaded twice"** — opt-in per-node `dedup_key`
  (`workflow_node_jobs.dedup_key`, literal or `$ref` resolved at enqueue):
  when a node-job with the same `(project, dedup_key)` has already completed,
  `execute_node` finalizes the new job as success (the twin's
  `context_delta` + a `_deduped_from` marker, dispatch event included)
  without invoking the body. The engine compares strings only — a host whose
  outputs are box-local encodes the location in the key.
- Migration `0025` drops the `workflow_node_jobs` cpu/gpu queue `CHECK`
  (matched by definition, not name — the 0008 "allow-list moves host-side"
  move; sqlite twin = a one-time transactional table rebuild) and adds
  `dedup_key` + its partial index. Download workers gate on schema ≥ 25.

### Added — a model catalog, and an ordered array of acceptable models (migrations 0026, 0027)

- **`llm_model_catalog`** — which `(box, server kind)` can actually serve which
  model. OBSERVED, never declared: the claim worker's GPU heartbeat probe asks its
  box's servers (ollama `/api/tags` — everything installed, since ollama swaps on
  demand; vLLM `/v1/models` — the one pinned model) and publishes the answer. It
  runs in the claim worker because that is the process that can reach the server.
  An empty catalog means "we do not know", and the matcher treats it as
  cannot-serve. A model a probe stops listing is retired — scoped to the
  `(host, kind)` pairs that observation covered, so a transient probe failure never
  blanks a box.
- **`required_models`** — a node-job may name an ORDERED array of acceptable
  `{model, server_types}` entries (`workflow_node_jobs.required_models` JSONB +
  `required_model_ids`, its flat claim index; schemas declare it per node). The
  order is a priority: the claim picks the first entry this box can serve and
  stamps it on **`resolved_model`**. The scalar `required_model` is unchanged and
  dual-written as the array's head, so a worker on an older engine still claims a
  new row correctly — no flag day.
- **The claim advertises what the box HOLDS, not what the registry declares.** A
  GPU worker narrows `model_registry.known_ids()` by what its own servers just
  reported, so a job for a model no box has pulled is no longer claimable
  anywhere. Only ids whose `ModelSpec` declares `served_by=(…)` are narrowed — an
  in-process model can never appear in a server's listing, so an unmarked spec is
  treated as in-process (the safe default). An empty or non-intersecting
  observation leaves the declared list untouched.
- **Which BYTES ran (migration 0027)** — `llm_model_catalog.digest` keeps each
  tag's server-reported content digest, and the claim worker stamps
  `workflow_node_jobs.resolved_artifact`: the immutable artifact tag
  (`<role>-v<N>-<precision>-<server>`) whose digest matches the role tag's on that
  box, or vLLM's served id when that is itself an artifact tag.
  `ModelSpec.server_models` declares artifact options per kind as a tie-break
  only. NULL when nothing matches — an absent answer, never a guessed one.
- `flag_unassignable_gpu_jobs` says WHICH problem a stuck job has: "no live GPU
  machine SERVES model X" (installed nowhere) versus the VRAM verdict.

### Added — vLLM brought up on the REQUESTED model

- The default `ensure_up_fn` is no longer a no-op: it inspects the vLLM sidecar
  over the docker Engine API (stdlib, unix socket) and, when it serves a
  different model, recreates it from its own inspect output with only `--model`
  swapped — image, env, ports, binds, device requests and restart policy survive
  verbatim. An already-correct sidecar is left alone; a missing one raises so the
  caller can soft-degrade. The container name and socket are overridable
  (`QUEUE_WORKFLOWS_VLLM_CONTAINER`, `QUEUE_WORKFLOWS_DOCKER_SOCK`), and
  `EngineConfig.vllm_start_fn` still replaces the whole mechanism.

### Added — LAN broadcast of hardware metrics (opt-in)

- `configure(broadcast_hw_stats=True)` or `QUEUE_WORKFLOWS_BROADCAST_HW_STATS=1`:
  the hw-metrics sampler also sends each sample as a UDP broadcast on the local
  network (`hw_broadcast_port`, default 48291) beside the Postgres NOTIFY — one
  shared envelope builder, no extra sampling, the default-route interface only
  unless asked, and no network error can reach the sampler thread. **Off by
  default.** `hw_metrics` no longer imports the database at module scope, so a
  broadcast-only host needs no psycopg; `deploy/hw-broadcast.sh` runs a
  standalone sampler for a host with no claim worker (ROCm or NVIDIA GPU
  passthrough). See `docs/hw_lan_broadcast.md`.

### Changed — an unreachable database is waited for, not crash-looped on

- `db.get_pool` installs the pool only after a successful open. A timed-out open
  used to leave a permanently-closed pool installed, so every later call raised
  `PoolClosed` however healthy the database became; now it raises
  `db.DbUnreachable` and the next call builds a fresh pool.
- The claim worker, orchestrator and scheduler **wait in process** for an
  unreachable database (`db.wait_for_db`: capped 1→30 s backoff, one
  `waiting_db target=host:port/db` line a minute, interruptible by SIGTERM)
  instead of exiting into a supervisor restart loop. A waiting worker holds no
  lease and claims nothing. **Behaviour change** — set
  `QUEUE_WORKFLOWS_DB_WAIT_MAX_S` to bound the wait and get exit-and-restart back.
- A database that can never work **as configured** (bad credentials, no such
  database, a malformed or unset DSN — `db.DbMisconfigured`) exits with code **2**,
  distinct from the watchdog codes 75–79, naming `host:port/dbname` and never the
  password. `db.mask_dsn` / `db.scrub_secrets` also redact libpq's own echoes of a
  malformed DSN.

### Changed — the reference lane supervisor's knobs

- `deploy/run_claim_lane.sh` reads its per-lane process counts from
  `QUEUE_WORKFLOWS_CPU_LANE_WORKERS` (default 30), `QUEUE_WORKFLOWS_GPU_LANE_WORKERS`
  (default 1) and `QUEUE_WORKFLOWS_LANE_WORKERS` (any other queue, default 1) — the
  engine's own env namespace. If you run this script, set these names.

### Changed — broker panel UX pass (Chromium-audited)

- **The v2 operator panel** (`queue-broker-web`, `GET /`) got a measured UX/a11y
  pass, canon intact (stdlib-only, server-rendered, still zero `<script>`):
  - `?project=` now scopes **all three** sections — the workers table (and
    `GET /api/workers` / `/api/snapshot`) previously ignored the filter and
    showed every tenant's fleet on a filtered view
    (`orchestrator.list_workers(project=…)`).
  - Filter chips come from a new `orchestrator.list_projects()` DISTINCT view
    instead of scanning the newest-500 jobs — a project whose only job is old
    keeps its chip (it was an unreachable filter), and the panel renders ~5×
    faster (5.78 → 1.19 ms at 523 jobs; the scan was 79 % of render time,
    paid per viewer per refresh).
  - `?refresh=` cadence control (5s default · 30s · off, clamped, malformed
    falls back): the WCAG 2.2.1 "user can stop the timed reload" affordance,
    still no-JS — `0` renders no meta-refresh tag at all.
  - Tables scroll inside their own `overflow-x:auto` wrapper — at 390 px the
    page itself side-scrolled (scrollWidth 664), pushing status/time columns
    off-screen.
  - WCAG AA contrast: all six job chips + three worker chips sat at
    2.88–3.77:1; text-role tokens darkened hue-for-hue (every pair now
    ≥ 4.62:1), enforced by a test that parses the live stylesheet.
  - Semantics: `scope="col"` on every `th`, visually-hidden table captions,
    `aria-current="page"` on the active project chip.
  - Scannability: the 8-char job id keeps the full id in `title`; liveness
    columns render relative ages ("3s/5m/2h/4d ago", absolute UTC in `title`);
    the capped jobs table says "newest 50 of N"; empty states point at
    `POST /api/submit` / `/api/ask`.

### Fixed

- **A server switch no longer holds a box out of service for a start that
  cannot work.** The host's start lever may report what it did
  (`server_switch.StartOutcome`, a plain `{"status", "reason"}` dict, or `None` —
  "no verdict", which keeps the old behaviour). A PERMANENT failure (no such
  container, an OCI error, no lever wired) aborts on the same tick and
  QUARANTINES that kind with a doubling, persisted backoff; a start that is never
  acknowledged gives up after `start_ack_timeout_s` (300 s) instead of the 1800 s
  warm window; and the whole switch budget belongs to the REQUEST, so a restart
  mid-switch inherits what is left instead of re-arming a full window. A completed
  switch clears the kind's quarantine.
- **A drain-park no longer fights the job it keeps.** The park watchdog holds off
  the LLM eviction while this worker's own pool-lane job is still in flight
  (bounded by the GPU job budget) instead of forcing a cold reload on every call
  of the job it promised to keep; a peer's remote traffic still gets nothing. Its
  repeated verdict is logged once a minute, not every tick.
- **The spill storm.** A GPU worker denied the card by its own box's lease pauses
  its GPU claiming (1 s doubling to 30 s, under the lease TTL) instead of
  re-claiming the row it just re-queued to the front, a dozen times a second.
- **A server-served model no longer enters the in-process load gate** — which
  evicted the very server copy the node was about to call, then refused the
  load — and residency comparisons use canonical tags (`model` ≡ `model:latest`)
  everywhere.
- `register_worker`'s UPSERT used a bare `state` inside ON CONFLICT DO UPDATE —
  AmbiguousColumn on Postgres, silently resolved to the target on SQLite.
- **Oversized `POST` bodies (413) killed clients with EPIPE**: the broker web
  service answered 413 and closed while the client was still mid-`send()`, so
  urllib-style clients raised a broken pipe instead of reading the response
  (the intermittent `test_oversized_body_rejected_413` failure). The handler
  now discards the rejected body in bounded 64 KiB chunks (8 MiB drain cap —
  the OOM guard holds) and drops the connection so the remainder can't be
  parsed as a next keep-alive request.
- **Schema `cpu_slots` never reached pipeline-expanded rows** (0024
  regression): the `_nodes_of` whitelist silently dropped the key, so every
  pipeline-schema node enqueued as an ordinary 1-slot job regardless of its
  declared width. `cpu_slots` (and the new `download`/`dedup_key`) are now
  threaded through explicitly, with a regression test.
- The vLLM sidecar recreate no longer times out mid-cycle (socket 180 s, stop
  grace 10 s) and leaves the box with no sidecar at all.
- The wheel builds again: a redundant `force-include` of the migrations made
  hatchling refuse to add the same files twice.

### Security & hardening — full-system audit pass

- **Probe URL scheme guard**: the four `llm_probe` fetch helpers now refuse
  non-`http(s)` URLs (`file://` etc.). Base URLs come from operator config and
  the DB (`worker_controls`, 0013), so a DB writer must not be able to turn an
  LLM probe into a local file read. GET-style helpers treat a refused scheme as
  unreachable (logged); JSON/POST helpers raise `ValueError`.
- **GPU lease dir is sticky**: the cross-uid `FileLeaseStore` directory is now
  `0o1777` (like `/tmp`), not bare `0o777` — root containers and host processes
  still share it, but one uid can no longer unlink/squat another's lease file.
- Latent-NameError annotations fixed (`claim_worker.predict_default_kind`'s
  `Iterable`, `vllm._docker_api`'s `Any`) and pinned by a hints-resolve test;
  4 shadowed local `import json` removed (node_queue); `queue-broker --status`'s
  count closure default-binds its loop variable; the two rc-handled
  `subprocess.run` calls declare `check=False` explicitly.
- **`ingest_store` SPI seam covered hermetically**: 5 stub-backend tests pin the
  priority inversion (ingest ASC ⇄ SPI DESC), sparse-payload re-inflation
  defaults, terminal delegation, the reclaim projection, and the snapshot shape —
  previously only exercised by live-redis tests that skip without a server.
- Audit clean bills: immutability guards verified on every status transition
  (no violation), pip-audit clean, no committed secrets, claim path
  index-backed at 50k rows (21–37 ms), 59× bandit B608 confirmed
  false-positive per the validated-fragments house rule.

### Notes

- `deploy/k8s/llm-servers.yaml` pins `ollama/ollama:0.33.2` instead of `:latest`:
  ollama gates model support on its own version, so a floating tag silently falls
  behind the models it is asked to serve.
- Docs: `schema.md` is current through 0027; `watchdogs.md` documents exit code 2;
  new `docs/hw_lan_broadcast.md`.

## [1.1.0] — 2026-08-04

A feature release, 110 commits on from 1.0.6. Everything here is **additive with
safe defaults** — migrations 0022–0024 add nullable columns, the two new cpu
knobs default to the historical behaviour, and an existing deployment upgrades
without a config change.

### Added — one LLM server at a time, provably

- **Drain-based server switch** (`queue_workflows/server_switch.py`): a box that
  changes serving kind (ollama ↔ vLLM) goes OUT OF SERVICE for the switch, waits
  until the outgoing server is *observed gone*, and only then starts the
  incoming one. Starting vLLM is a **handover**, not an addition — two resident
  copies of a 34 GB model is how a unified-memory box reaches OOM.
- **Per-kind slot caps** — concurrency follows the SERVING KIND, not one global
  PAR: an ollama-resolved job caps at 1 (ollama answers one request at a time),
  vLLM at PAR (it batches). Every GPU row now stamps a serving kind
  (**migration 0023**, `workflow_node_jobs.resolved_server`).
- **Per-node LLM server preference** (**migration 0022**,
  `workflow_node_jobs.server_types`) — an ordered list, e.g.
  `["vllm", "ollama"]`. The row is claimable only by a worker whose served kinds
  intersect it, and a row *with* a preference is pool-eligible even when it
  names a `required_model` (an external server holds the weights).
- **vLLM lifecycle**: start a stopped vLLM when queue depth justifies it, stop
  it after an idle TTL observed **at the server**, and an amortisation gate so a
  single job never pays a multi-minute cold start.

### Added — the parked box releases the card

- A parked GPU worker now frees the box's LLM memory, and the **park watchdog**
  kills any broker-managed LLM server on sight while parked — a serving vLLM is
  stopped mid-request included, since remote traffic must not keep a parked box
  alive. Silenced by `QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER`.
- A **self-managing GPU node** (one that loads its own weights) occupies a slot
  of its own instead of borrowing the LLM server's, so it genuinely excludes an
  LLM dispatch from the same card — and reclaims that card's resident LLM
  memory, because a keep-alive is not a lease.

### Added — a wider cpu lane, and jobs that need more of it

- **`cpu_worker_threads`** — one cpu claim-worker process may run N node-jobs
  concurrently, on a pool of N long-lived worker threads (`--threads N`, env
  `QUEUE_WORKFLOWS_CPU_WORKER_THREADS`, or `configure(cpu_worker_threads=N)`).
  **Default `1` creates no pool and is byte-identical to the historical serial
  worker.** Deliberately NOT a `ThreadPoolExecutor`: that starts its threads
  lazily *inside* `submit`, after enqueuing the work, so a refused thread raises
  while the body still runs — and a slot freed on that signal breaks the
  ceiling. For CPU-bound pure-Python nodes the GIL serialises threads, so N
  processes remain the recommendation; the knob is for IO-bound cpu nodes.
- **`cpu_slots`** (**migration 0024**) — how many slots of a widened lane ONE
  job occupies, for a node whose body runs its own threads. Enforced in the
  CLAIM (`LEAST(COALESCE(cpu_slots,1), lane_width) <= free_slots`), so the
  reservation binds before the body starts. A job wider than the lane runs
  alone rather than being excluded forever; a serial worker ignores the column
  entirely. `NULL` = 1 = an ordinary job.
- A watchdog trip in a widened lane releases the sibling jobs **resume-style**
  (front of queue, no `watchdog_retries` bump) before the process exits, scoped
  to the claims that process actually holds — never host-wide, which would yank
  a sibling *process's* rows and cascade.

### Fixed

- The VLM pool's defer view now defines "pool work" exactly as the claim does,
  so a box can no longer defer work no peer will take.
- The double-load watchdog counts VRAM **copies**, not model ids.
- A remote-served node no longer demands an in-process model cache.
- The GPU heartbeat advertises the **enforced** slot budget, not the raw PAR.
- Warm-model affinity is scoped to opted-in rows; the lane split no longer leaks.
- `resolve_for_job` probes availability instead of assuming it.

### Notes

- Docs: `deployment.md` gains §2.5 (threads vs processes, the GIL trade-off,
  connection budget, the heartbeat-starvation hazard, and `cpu_slots`);
  `watchdogs.md` gains §7 (trip blast radius in a widened lane); `schema.md`'s
  migration table is current through 0024.
- `__version__` was stale at `0.5.0` and now tracks the packaged version.

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
