# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
