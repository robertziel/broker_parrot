<div align="center">

# 🦜 broker_parrot

_A self-hosted job fleet on the database you already run — Postgres or SQLite._

### Turn the machines you already own into one self-healing job fleet.

> A workflow engine that uses your database as the queue and message bus — dispatching DAG and periodic jobs across worker processes, keeping GPU models warm, and recovering from crashes on its own.

Orchestrate work across a handful of heterogeneous CPU/GPU boxes with nothing but a database you already run. No Celery, no Redis to babysit, no cluster scheduler. Insert a row and the work is enqueued; a dead worker's lease lapses and its job re-runs somewhere healthy.

[![License: AGPL v3](https://img.shields.io/badge/license-AGPL%20v3%20or%20later-blue.svg)](LICENSE)
[![Free for open source](https://img.shields.io/badge/free-open%20source%20%26%20personal-brightgreen.svg)](#️-license--free-for-open-use-commercial-licensing-available)
[![Commercial license](https://img.shields.io/badge/commercial%20license-hello%40robertz.co-orange.svg)](mailto:hello@robertz.co)
[![Agent friendly](https://img.shields.io/badge/🤖%20agent-friendly-8A2BE2.svg)](#-agent-friendly--just-type-it)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![SQLite](https://img.shields.io/badge/sqlite-default-003B57.svg)](#installation)
[![Postgres](https://img.shields.io/badge/postgres-14%2B-336791.svg)](#installation)

**[Quickstart](#-quickstart--60-seconds-zero-servers)** ·
**[Agent setup](#-agent-friendly--just-type-it)** ·
**[Docs](#-documentation)** ·
**[License](#️-license--free-for-open-use-commercial-licensing-available)**

_Free for personal, research & open-source use (AGPL). Commercial license? **[hello@robertz.co](mailto:hello@robertz.co)**_

</div>

**broker_parrot** is a small, self-hosted workflow engine where **the database _is_ the message bus**. Inserting a row enqueues work; a trigger fires `LISTEN/NOTIFY` inside the writer's own transaction, so there's never a "queued but never woken" gap. Workers claim jobs with `SELECT … FOR UPDATE SKIP LOCKED`, renew a lease while they run, and a dead or wedged worker's job is automatically re-queued onto a healthy peer. It runs DAG node-jobs and periodic background jobs across boxes you already own, lets you flip any machine's worker **ON/OFF** on demand, and keeps a GPU model resident across same-model jobs.

As of **v1.0.0 the default backend is SQLite** — a daemon-less local file, zero server to stand up — so you can `import`, `configure()`, and run with nothing else installed. Point it at **Postgres** for a real fleet with one line.

---

## ⚡ Quickstart — 60 seconds, zero servers

The whole engine in one file. No database server, no broker, no YAML — SQLite is the default backend, and every line below is the public API:

```bash
pip install "queue_workflows @ git+https://github.com/robertziel/broker_parrot"
```

```python
# quickstart.py — run it with:  python quickstart.py
import types, uuid

import queue_workflows
from queue_workflows import db, dispatcher, node_queue, run_store
from queue_workflows.claim_worker import ClaimWorker

queue_workflows.configure()                     # SQLite file — nothing to install or run

# A "node" is any module with a run() function. Register one inline:
hello = types.ModuleType("hello")
hello.run = lambda **_: {"greeting": "hello, world"}
queue_workflows.set_node_resolver(lambda name: hello)

# One workflow -> one pipeline -> one node.
queue_workflows.set_workflow_provider(
    lambda n: {"name": "hi", "mode": "node",
               "steps": [{"id": "s", "kind": "pipeline", "pipeline": "hi"}]},
    lambda n: {"name": "hi", "nodes": [{"id": "hello", "node": "hello"}]},
)

db.bootstrap()                                  # applies the migration chain (idempotent)

run_id = str(uuid.uuid4())
run_store.insert_run(run_id=run_id, workflow_name="hi")   # inserting a row IS enqueuing
dispatcher.start_run(run_id)                              # fan out the ready nodes

ClaimWorker(queue="cpu", host="laptop").run_once()        # claim + execute the node
print(node_queue.list_jobs_for_run(run_id)[0]["status"])  # -> completed
```

That's the whole loop: **a row went in, a worker claimed it with `SKIP LOCKED`, the node ran, the terminal state and its dispatch event committed in one transaction.** Growing up from here is configuration, not rearchitecture:

1. **Real project layout** — nodes as modules, pipelines/workflows as JSON files → [`docs/setup.md`](docs/setup.md)
2. **A real fleet** — `configure(db_backend="pg")` + one process per role (`queue-orchestrator`, `queue-claim-worker --queue cpu|gpu`, `queue-scheduler`) → [`docs/deployment.md`](docs/deployment.md)
3. **GPU boxes** — warm-model cache, affinity routing, watchdogs, per-box arbitration → [`docs/gpu_and_llm.md`](docs/gpu_and_llm.md)

---

## 🤖 Agent-friendly — just type it

This repo is written to be **driven by a coding agent** (Claude Code, Cursor, Codex, …): the docs are the spec, the test suite is the behavioral contract, all host wiring funnels through one `configure()` seam, and the safe-by-default SQLite backend means an agent can verify its own work end-to-end without touching your infrastructure. **You describe the jobs; the agent wires the engine.**

Copy-paste this into your agent, edit the two bracketed lines, and let it work:

```text
Set up broker_parrot (python package queue_workflows,
https://github.com/robertziel/broker_parrot) as the job queue for this project.

My jobs: [e.g. "fetch RSS feeds every 15 min, then summarize each new article
with an LLM on my GPU box, then publish the summary — fetch/publish are CPU,
summarize is GPU"].
My database: [e.g. "the Postgres this app already uses, DSN in env MYAPP_DB_URL"
— or "just SQLite for now"].

Steps:
1. pip install "queue_workflows @ git+https://github.com/robertziel/broker_parrot"
2. Read docs/setup.md and docs/configuration.md for the wiring pattern.
3. Create a nodes/ package (one run() module per job), pipelines/*.schema.json
   (the DAG: ids, depends_on, gpu flags), definitions/*.json (the workflow),
   and an engine.py that calls queue_workflows.configure(...),
   set_workflow_provider(...), and db.bootstrap().
4. For periodic work use register_ingest_task + set_ingest_schedule
   (docs/configuration.md) instead of cron.
5. Prove it: enqueue one run with run_store.insert_run + dispatcher.start_run,
   execute it with ClaimWorker(queue="cpu").run_once(), and show me the
   completed status. Use the SQLite default for this proof even if production
   is Postgres.
6. Give me the production launch commands (queue-orchestrator,
   queue-claim-worker per queue, queue-scheduler) from docs/deployment.md.
```

Every claim in that prompt is backed by a doc the agent can read: the engine never imports your app (the seam is `configure()` + hooks), the suite refuses to run against a non-`_test` database, and one end-to-end round-trip — enqueue → claim → execute → completed — is exactly what the `run_once()` API is for.

---

> **There's no bundled dashboard — the engine emits what a dashboard needs.** Live per-host CPU/GPU/RAM over `pg_notify('hw_metrics', …)`, `worker_heartbeats`, the `node_queue.*_snapshot()` read models, and the `worker_controls` ON/OFF toggles are all there. Bring your own front-end — it's a great task to hand a coding agent. A newer **broker web service + operator panel** ([`docs/broker.md`](docs/broker.md), [screenshot](#-one-broker-for-many-projects)) also ships as a pure-stdlib, server-rendered option.

Here's an example of a richer front-end built directly on that telemetry — a fleet **hardware + queue** operator panel. It reads the engine's `hw_watch_samples` flight recorder (migration 0021), `worker_heartbeats`, and `worker_controls` straight from the DB and draws a 1-hour trail per box: GPU **and** CPU temperature, power, **clock speed**, and throttle, plus worker ON/OFF control. Below, `box-a` runs uncapped and is hitting its power-brake (⚡), while `box-b`'s GPU clock is pinned flat at 2100 MHz and stays stable — the kind of operational story the engine's raw telemetry makes visible:

![A fleet hardware + queue operator panel built on the engine's telemetry — per-box GPU/CPU temperature, power, clock, and throttle over the last hour, with worker ON/OFF control](docs/images/flight-deck-panel.png)

---

## Table of Contents

- [Quickstart — 60 seconds, zero servers](#-quickstart--60-seconds-zero-servers)
- [Agent-friendly — just type it](#-agent-friendly--just-type-it)
- [Why broker_parrot?](#-why-broker_parrot)
- [How it compares (vs. Triton Inference Server)](#-how-it-compares-vs-triton-inference-server)
- [Highlights](#-highlights)
- [Installation](#installation)
- [Core concepts](#-core-concepts)
- [Architecture at a glance](#-architecture-at-a-glance)
- [Setting up a real project](#-setting-up-a-real-project)
- [Turning workers on/off](#️-turning-workers-onoff--the-operator-control-plane)
- [Host-defined queues + ingest jobs](#️-host-defined-queues--parametrised-ingest-jobs)
- [One broker for many projects](#-one-broker-for-many-projects)
- [Pluggable storage backends](#️-pluggable-storage-backends)
- [GPU models & LLM backends](#-gpu-models--llm-backends)
- [Migrations](#️-migrations)
- [Tests](#-tests)
- [Documentation](#-documentation)
- [Background](#-background)
- [Contributing](#-contributing)
- [License — free for open use, commercial licensing available](#️-license--free-for-open-use-commercial-licensing-available)

---

## ✨ Why broker_parrot?

The pitch in one line: **the database you already run is the most durable thing you own — so let it _be_ the queue.** No second broker to operate, no scheduler cluster. You `INSERT` a row inside your own transaction and the work is enqueued; a crashed worker's lease lapses and the row is re-queued. Purpose-built for a **small, self-hosted, heterogeneous CPU/GPU fleet you already own** — not a 1,000-node cloud.

**Reach for it when** you have a handful of self-hosted boxes, want GPU-aware scheduling (warm models, per-box ON/OFF) and crash-safe recovery, and would rather not stand up a broker or a workflow platform just to move jobs between machines.

**Look elsewhere when** you need a hosted UI, multi-region durability, or versioned-workflow replay at large scale — that's a different class of tool.

---

## 🆚 How it compares (vs. Triton Inference Server)

People often ask how this relates to [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html). **They solve different problems and compose rather than compete.** Triton is a *model server* — it loads models into one process and answers inference requests synchronously, optimized for raw throughput. broker_parrot is a *durable job orchestrator* — it routes async work across a fleet of boxes with crash recovery, using the database as the bus. Triton makes one model fast on one node; broker_parrot reliably moves a unit of work to whichever node holds the right warm model and survives failure. You can even run **Triton (or vLLM/ollama) _as a node body inside_ a broker_parrot job** — the engine schedules and recovers the work, Triton serves the tensors.

| Dimension | **broker_parrot** | **Triton Inference Server** |
|---|---|---|
| **Category** | Durable async workflow / job-queue engine | Real-time model inference server |
| **Core interaction** | `INSERT` a row → job runs later, durably | HTTP/gRPC request → synchronous response |
| **Latency profile** | Async, seconds–hours; survives restarts | Sub-second request/response |
| **Unit of work** | Arbitrary node body / ingest task (`run(...)`); a model call is just one kind | A model inference (tensors in → tensors out) |
| **Multi-step / pipelines** | First-class **DAG** dispatch with a durable outbox, deps, skip-if, fan-out | Model **ensembles** + Business Logic Scripting |
| **Throughput tricks** | None by design — concurrency-1 per worker by contract | **Dynamic batching**, concurrent model execution |
| **Model lifecycle** | Warm `ModelCache`, idle-unload, **warm-model affinity routing** across the fleet | Model repository: load/unload, versioning |
| **Framework backends** | Host-agnostic — you bring `run(...)`; no framework coupling | TensorRT, PyTorch, ONNX, OpenVINO, vLLM, Python |
| **Fleet & scheduling** | DB `FOR UPDATE SKIP LOCKED` claim + `LISTEN/NOTIFY`, lease + reclaim across N boxes | Per-node server; scaling delegated to **Kubernetes** |
| **Crash recovery** | Lease lapse → re-queue onto a healthy peer; wall-clock / stall / GPU-health watchdogs re-queue-and-retry | K8s restarts the pod; no built-in job re-queue |
| **Operator control** | Per-`(host, queue)` **ON/OFF** control plane (hard-stop frees VRAM) | No fleet ON/OFF; managed by K8s |
| **State & durability** | Everything in the DB — jobs, leases, append-only event log, heartbeats | Stateless server; metrics only |
| **Background / periodic work** | First-class **ingest jobs** + scheduler ticker | Out of scope — request-driven only |
| **Dependencies** | Just a **database** (SQLite default, or Postgres via psycopg 3; optional redis/mongo) | NVIDIA runtime / CUDA; typically GPU + K8s for scale |

**Where they overlap:** both keep models warm and care about GPU efficiency. **Where they don't:** Triton has no durable queue, no DAG, no cross-host crash-recovery, no operator ON/OFF, and no periodic work; broker_parrot has no dynamic batching, no native framework backends, and isn't a low-latency request server. Use Triton to make an inference *fast*; use broker_parrot to make the *work* reliable across the fleet.

---

## 🌟 Highlights

- 🔒 **Exactly-once claims** — `SELECT … FOR UPDATE SKIP LOCKED` over a database queue, woken instantly by `LISTEN node_job_ready` so workers never poll-spin or double-run a job.
- ❤️‍🩹 **Self-healing leases** — a live worker renews its lease as it runs; a crashed or wedged worker's lease lapses and the orchestrator re-queues the row onto a healthy peer.
- 🔗 **DAG dispatch with a durable outbox** — a node's terminal status and its dispatch event are written in one transaction, then drained to fan out the next ready nodes — no lost edges, no double fan-out.
- 🔥 **GPU warm-model cache** — keeps a single model resident across consecutive same-model jobs and only drops/reloads on a real swap, so the expensive load happens once.
- ⚡ **"Run next" priority flag** — flag a queued node so the next worker asking for its queue claims it first — jump an urgent job to the head with one call.
- 🟢 **Operator ON/OFF control** — hard-stop or park any `(host, queue)` worker on demand; a hard stop requeues the in-flight job to a free peer and frees its RAM/VRAM, with no restart.
- 📊 **Per-host telemetry** — live CPU/GPU/RAM and capacity stream over `pg_notify('hw_metrics', …)` plus `worker_heartbeats`, ready to drive a dashboard.
- ⏰ **DB-native scheduler** — a built-in ticker enqueues recurring background jobs at scheduled minutes (with optional hour windows) — no cron, no external beat process.
- 🗄️ **Pluggable store** — SQLite (default), Postgres, Redis, or MongoDB, behind one durable-queue contract.

---

## Installation

Requires **Python 3.10+**. As of **v1.0.0 the default backend is SQLite** — a daemon-less local file, zero server to run — so the only hard runtime dependency is `psycopg` (used by the SQLite *and* Postgres paths). For a Postgres deployment (**14+**), opt in with `configure(db_backend="pg")` or `export QUEUE_WORKFLOWS_DB_BACKEND=pg`.

Not on PyPI yet — install straight from GitHub (the distribution/import name in code is `queue_workflows`):

```bash
pip install "queue_workflows @ git+https://github.com/robertziel/broker_parrot"
```

Optional extras:

```bash
# hw_metrics CPU/RAM probe (GPU probe shells out, no extra dep)
pip install "queue_workflows[metrics] @ git+https://github.com/robertziel/broker_parrot"

# alternative storage backends
pip install "queue_workflows[redis]   @ git+https://github.com/robertziel/broker_parrot"
pip install "queue_workflows[mongodb] @ git+https://github.com/robertziel/broker_parrot"   # needs a replica set
```

---

## 🧩 Core concepts

Three ideas carry the whole design. The full treatment is in [`docs/architecture.md`](docs/architecture.md).

**The database is the bus.** `INSERT`ing a row *is* enqueuing the work. A trigger fires `pg_notify('node_job_ready', …)` **inside the writer's transaction**, so a listening worker wakes the instant the row is visible — no separate publish step, no lost-wakeup window.

**Three process roles, one database.** All three run as ordinary processes against the same DB:

- **Orchestrator** — the only process that bootstraps migrations. It runs the DAG dispatch loop, drains the dispatch-event outbox, sweeps for lapsed leases and dead workers, and resumes parked input nodes. No node bodies run here.
- **Claim worker** — **one process is one worker, concurrency-1 by contract.** It `LISTEN`s, greedily drains its queue on each wake, renews its lease while a job runs, and writes the terminal status + outbox event in one transaction. `cpu`/`gpu` draw DAG node-jobs; ingest-family queues draw standalone ingest jobs.
- **Scheduler** — a DB-native ticker that sleeps to the next scheduled minute and enqueues periodic `ingest_jobs` rows.

**Leases make it self-healing.** A live worker renews `lease_expires_at` (~every 10 s), so lease length is independent of job duration. A dead or wedged worker stops renewing; its lease lapses; the orchestrator's reclaim sweep flips the row back to `queued` (re-firing the NOTIFY). Layered on top are wall-clock, no-progress, and GPU-health watchdogs — each re-queues-and-retries rather than failing — plus an out-of-process dead-worker detector for hardware hangs. See [`docs/watchdogs.md`](docs/watchdogs.md).

---

## 🏗 Architecture at a glance

**The system.** Your app inserts rows inside its own transactions; the trigger's `NOTIFY` wakes workers the instant the row commits. Three kinds of engine processes — orchestrator, claim workers, scheduler — coordinate through nothing but the database:

```mermaid
flowchart LR
    subgraph app["Your application"]
        A["INSERT run / job<br/>(inside your own txn)"]
    end
    subgraph db["The database — SQLite or Postgres"]
        Q[("workflow_runs<br/>workflow_node_jobs<br/>ingest_jobs")]
        HB[("worker_heartbeats<br/>worker_controls<br/>node-event log")]
    end
    subgraph fleet["Engine processes — spread across N boxes"]
        O["Orchestrator<br/>DAG dispatch · outbox drain<br/>lease reclaim · dead-worker sweep"]
        W1["Claim worker — cpu<br/>LISTEN → SKIP LOCKED claim"]
        W2["Claim worker — gpu<br/>+ warm ModelCache"]
        S["Scheduler<br/>periodic ingest ticker"]
    end
    A --> Q
    Q -- "NOTIFY fires inside<br/>the writer's txn" --> W1
    Q -- "instant wake,<br/>no polling" --> W2
    S --> Q
    O <--> Q
    W1 <--> Q
    W2 <--> Q
    W1 --> HB
    W2 --> HB
    O --> HB
```

**A job's life.** Every transition is a guarded `UPDATE`; crash recovery is just a lapsed lease:

```mermaid
stateDiagram-v2
    [*] --> queued: INSERT - the NOTIFY rides the txn
    queued --> running: worker claims - FOR UPDATE SKIP LOCKED
    running --> completed: run() returns
    running --> failed: raises, or watchdog retries exhausted
    running --> cancelled: run cancelled
    running --> queued: lease lapses - worker died, reclaim sweep requeues
    running --> queued: watchdog trip - requeue and retry on a healthy peer
    completed --> [*]
    failed --> [*]
    cancelled --> [*]
```

**The durable outbox.** A worker never calls the dispatcher — it writes the terminal status *and* a dispatch event in one transaction, and the orchestrator fans out from there. Fan-out is retryable and survives any crash between the two steps:

```mermaid
sequenceDiagram
    participant W as Claim worker
    participant DB as Database
    participant O as Orchestrator
    W->>DB: UPDATE job to completed + INSERT dispatch_event (ONE txn)
    O->>DB: drain the outbox
    O->>DB: enqueue downstream nodes whose deps are all met
    DB-->>W: NOTIFY node_job_ready → next claim
```

The full treatment — including the watchdog stack and the host-agnostic hook seam — is in [`docs/architecture.md`](docs/architecture.md) and [`docs/watchdogs.md`](docs/watchdogs.md).

---

## 🛠 Setting up a real project

The production-shaped walkthrough lives in **[`docs/setup.md`](docs/setup.md)** — a real node module, the pipeline + workflow JSON, the `configure()` wiring, the three processes, and kicking off a run. Every snippet there is the public API, end to end. (Prefer to delegate? The [agent prompt](#-agent-friendly--just-type-it) above walks a coding agent through exactly that page.)

```text
your-app/
├── nodes/            # one run() module per job — no base class, no decorator
├── pipelines/        # *.schema.json — the DAG: ids, depends_on, gpu flags
├── definitions/      # *.json — workflows stringing pipeline steps together
└── engine.py         # configure() + set_workflow_provider() + db.bootstrap()
```

---

## 🎚️ Turning workers on/off — the operator control plane

Each machine runs **one worker per queue** under a `host_label` (a box can run both a `cpu` and a `gpu` worker). An operator can flip any one **ON or OFF independently**, and the state is just **a row in `worker_controls`** — so the engine's helper, the `queue-worker-control` CLI, *or any app sharing the database* can set it. A trigger fires `pg_notify('worker_control', '<host>:<queue>')` so the worker reacts **immediately**, no polling lag.

Turning a worker **OFF is a hard stop**: it **requeues the in-flight job** (resume-style, redistributed to a healthy peer — *not* failed), frees RAM/VRAM, and the worker comes back **parked** (idle, not claiming) until turned back ON.

```bash
# CLI — defaults --host to the local hostname
queue-worker-control --queue gpu --off                 # hard-stop this box's gpu worker
queue-worker-control --queue gpu --on  --host host-a   # bring it back (resumes in place, no restart)
```

```sql
-- …or a plain SQL write from any consumer sharing the DB. The trigger wakes the worker.
INSERT INTO worker_controls (host_label, queue, desired_state, stop_policy, requested_by)
VALUES ('host-a', 'gpu', 'off', 'hard', 'ops')
ON CONFLICT (host_label, queue) DO UPDATE
  SET desired_state = EXCLUDED.desired_state, updated_at = now();
```

A worker **absent** from `worker_controls` is treated as **ON** (default-on), and the accessors no-op cleanly on a database that predates the feature — so adding the engine changes nothing until you opt in. Why a process exit (and a requeue, not a cancel), plus the extensible stop-policy seam: [`docs/worker_control.md`](docs/worker_control.md).

---

## 🏷️ Host-defined queues + parametrised ingest jobs

Two job families share the engine: DAG **node-jobs** (`workflow_node_jobs`, the reserved `cpu`/`gpu` queues, fanned out by the dispatcher) and standalone **ingest jobs** (`ingest_jobs`, **your own** queue names, no DAG). The ingest path isn't limited to one app's vocabulary — a second consumer can route its **own** queues and carry **per-job arguments**, enqueued **atomically with its own domain row** (the `NOTIFY` rides the caller's transaction):

```python
queue_workflows.configure(
    db_backend="pg",
    db_url_env="MY_DB_URL",
    ingest_queues=frozenset({"ingest", "hydro", "hydraulic", "corrdiff"}),  # NOT cpu/gpu (reserved for the DAG path)
    ingest_default_budget_s=3600,                                           # watchdog budget for these queues
)
queue_workflows.register_ingest_task("run_scenario", run_scenario)         # fn(reason) OR fn(reason, args) -> dict

# Parametrised + atomic with your own write — one transaction, no dual-write:
with my_pool.connection() as conn:
    my_create_scenario(conn, scenario_id)
    node_queue.enqueue_ingest_job(
        task_name="run_scenario", queue="hydraulic",
        args={"scenario_id": scenario_id}, conn=conn,
    )
```

A registered callable may be `fn(reason)` or `fn(reason, args)`. Ingest workers emit `worker_heartbeats`, so the snapshot reflects live workers per queue. `cpu`/`gpu` stay **reserved** for the DAG path.

---

## 🏢 One broker for many projects

Instead of running **one database per app**, pool **several apps onto one shared "broker" database** and tell them apart with a `project` tag (migration `0017`). Each client claims **only** rows whose `project` matches its own — exact match, in the same `SKIP LOCKED` statement — so two projects never see each other's work, yet one operator dashboard sees the whole fleet.

```python
# app A and app B share one BROKER_DSN, distinct tags
queue_workflows.configure(project="forecast", db_backend="pg", db_url_env="BROKER_DSN")
queue_workflows.configure(project="render3d", db_backend="pg", db_url_env="BROKER_DSN")
```

Set it once per process with `configure(project=...)`, or export `QUEUE_WORKFLOWS_PROJECT=<name>` (the env knob also reaches console tooling that hand-rolls its own `configure`). Default `""` keeps single-tenant deploys byte-identical. This is the **inverse** of `db_namespace` (redis/mongo), which *isolates* tenants who can't see each other; `project` *pools* them into one queue you filter.

The `queue-broker` console script bootstraps the shared schema and inspects the consolidated queue. A newer **broker service** inverts the model further — a pull→grant control plane with a web panel where the broker arbitrates a shared CPU/GPU across projects and can revoke a job at will. Full treatment: [`docs/broker.md`](docs/broker.md).

The bundled `queue-broker-web` operator panel (pure stdlib, server-rendered, no JS) over that shared queue:

![queue-broker-web — the shared CPU/GPU queue, per-project tabs, and the worker fleet](docs/images/broker-web-panel.png)

---

## 🗄️ Pluggable storage backends

The database is the engine — but the **queue store is selectable**: `configure(db_backend="sqlite" | "pg" | "redis" | "mongodb")`. **SQLite (default)** and **Postgres** are the two *relational* backends: they run the **full DAG engine** through a dialect seam. **Redis** and **MongoDB** are opt-in providers that reproduce the *same durable-queue contract* — **claim exactly-once, lease/reclaim, idempotent terminals, an atomic outbox, and per-namespace tenant isolation** — via a `StorageBackend` SPI (the redis/pymongo drivers import lazily, so a sqlite/pg deploy needs neither).

```python
import queue_workflows
queue_workflows.configure(db_backend="redis")        # or "mongodb" / "pg" / "sqlite" (default)

from queue_workflows.backends import get_backend
be = get_backend()                                    # bound to your configured namespace
job_id = be.enqueue("cpu", {"task": "render"})
job    = be.claim("cpu", worker="box-1", lease_s=30)
be.complete_with_event(job["id"], "completed", result={"ok": True})   # go terminal + append event, atomically
```

The port is deliberately **non-leaky** (no method takes or returns a driver handle) and each backend is **namespace-bound** so two tenants on one server stay isolated. The SPI is additive: selecting redis/mongo does not (yet) re-home the DAG orchestrator/worker off the relational store. Contract, capability matrix, and caveats: [`docs/storage_backends.md`](docs/storage_backends.md).

---

## 🤖 GPU models & LLM backends

A GPU claim worker owns a process-wide **warm `ModelCache`**: it keeps one model resident across consecutive same-model jobs, drops/reloads only on a real swap, and publishes `current_model` to its heartbeat so the claim `ORDER BY` can route a matching job to the box that's already warm (**affinity routing**). A capacity-aware assignment pass (migration `0015`) fits models to VRAM and flags what can't fit rather than thrashing.

GPU nodes often need a co-tenant **ollama / vLLM** server next to the worker. Which *kind* of server a box runs is per-machine, operator-set state (on `worker_controls`, migration `0013`); each host advertises which server types it can actually run in its heartbeat (`0014`). A built-in **idle supervisor** can stop a vLLM sidecar to reclaim VRAM and restart it on demand — the host teaches it how via `set_vllm_lifecycle(stop_fn, start_fn)`. Nodes never branch on server type; the engine resolves the exact endpoint per dispatch and threads it in as `run(llm_server=...)`.

Apps can also share a **GPU pool** — a namespaced `StorageBackend` queue addressed independently of the main store — so pooled workers across apps claim self-contained GPU tasks (code on the box, data on shared NFS). Full design: [`docs/gpu_and_llm.md`](docs/gpu_and_llm.md).

---

## 🗄️ Migrations

The engine **owns its schema**. Migrations ship as package data at `queue_workflows/migrations/NNNN_*.sql` (with a parallel `migrations_sqlite/` chain), and `queue_workflows.db.bootstrap()` applies them against a version ledger (`queue_schema_version`) — idempotent, safe on every boot. Only the orchestrator bootstraps (advisory-locked, so concurrent boots are safe); claim workers `db.wait_for_schema(min_version)` and block until the schema is ready.

A host with its **own domain tables** runs a *second* chain alongside the engine's, with its own version table:

```python
import queue_workflows

# 1. engine chain → queue_schema_version (queue tables)
queue_workflows.db.bootstrap()

# 2. your app's chain → its own ledger, fully independent of the engine's
queue_workflows.db.bootstrap(
    migrations_dir="myapp/migrations",
    version_table="myapp_schema_version",
)
```

The chain and every table are documented in [`docs/schema.md`](docs/schema.md).

---

## 🧪 Tests

```bash
pip install -e '.[test]'

QUEUE_WORKFLOWS_TEST_DB_URL=postgresql://user:pw@host:port/queue_workflows_test \
  python -m pytest
```

The suite **forces a `*_test` DB** and applies the engine migration chain — it refuses to run against a non-`_test` database, so you can't point it at anything precious. The multi-backend contract suite additionally reads `QUEUE_WORKFLOWS_TEST_REDIS_URL` / `QUEUE_WORKFLOWS_TEST_MONGO_URL`; each backend **skips** if its server is unset or unreachable. See `tests/conftest.py` and [`docs/deployment.md`](docs/deployment.md).

---

## 📚 Documentation

The [`docs/`](docs/) set:

- [`setup.md`](docs/setup.md) — the full project walkthrough: a real node, the pipeline + workflow JSON, `configure()` wiring, the three processes, and kicking off a run.
- [`architecture.md`](docs/architecture.md) — the mental model: the DB as the bus, the three process roles, the claim mechanism, DAG dispatch + the durable outbox, and the host-agnostic seam.
- [`configuration.md`](docs/configuration.md) — the complete host-wiring reference: `configure()`, every `set_*` / `register_*` hook, the env knobs, and the default philosophy.
- [`schema.md`](docs/schema.md) — the database schema and the full migration chain (Postgres and SQLite), plus the idempotency contracts.
- [`storage_backends.md`](docs/storage_backends.md) — the `db_backend` seam and the `StorageBackend` SPI (`sqlite` / `pg` / `redis` / `mongodb`).
- [`watchdogs.md`](docs/watchdogs.md) — the liveness model: leases, reclaim, the three watchdogs, the state watchers, and the out-of-process dead-worker detector.
- [`worker_control.md`](docs/worker_control.md) — the operator ON/OFF control plane and the extensible stop-policy seam.
- [`broker.md`](docs/broker.md) — the shared multi-project broker (the `project` tag) and the v2 pull→grant broker service + operator panel.
- [`gpu_and_llm.md`](docs/gpu_and_llm.md) — the warm-model cache, capacity-aware assignment, the shared GPU pool, and the ollama / vLLM backends.
- [`deployment.md`](docs/deployment.md) — running the engine in production: console scripts, the one-container-N-processes lane, migrations at deploy, and the fleet cutover runbook.
- [`use_cases/`](docs/use_cases/README.md) — ten worked operational scenarios: a box powering off mid-job (front-of-queue requeue + the zombie kill signal), boot/rejoin, operator stop, wedged-GPU recovery, DAGs, periodic ingest, the multi-project broker, warm-model affinity, run-next priority, and human-in-the-loop input.

---

## 📦 Background

The engine was extracted from a larger self-hosted stack into this standalone, open-source package so it can be reused and shared on its own. It runs a real multi-box CPU/GPU fleet in production every day — the reliability layers (leases, watchdogs, the over-claim reaper, the box-agent supervisor) each exist because something actually failed that way first.

---

## 🤝 Contributing

Contributions are welcome — bug reports, storage-backend providers, docs fixes, and rough edges all count. The house rules are short:

- **Tests are the spec.** The suite is written test-first; a behavior change comes with the test that pins it (`tests/test_invariant_*.py` for engine guarantees).
- **Stay host-agnostic.** The engine imports nothing from any host app — new tunables go through `configure()` / `envcompat` with a safe default (two guard tests enforce this).
- **SQLite + Postgres both green.** `QUEUE_WORKFLOWS_TEST_SQLITE=1 python -m pytest` needs no servers at all.

If broker_parrot saved you from standing up a broker, a ⭐ helps others find it.

---

## ⚖️ License — free for open use, commercial licensing available

**Free for personal, research, and open-source use** under the
**GNU Affero General Public License v3.0 or later** ([`LICENSE`](LICENSE))
© Robert Zieliński.

- 🆓 **Personal projects, research, homelabs, open source** — use it, modify it,
  ship it. The AGPL asks one thing back: derivative works stay source-available
  under the same license (AGPL §13 extends that to modified versions run as a
  network service). Simply *running* an unmodified copy behind your own app
  triggers no obligation.
- 💼 **Commercial product where the AGPL's copyleft doesn't fit?** A commercial
  license without the AGPL obligations is available — **let's talk:
  [hello@robertz.co](mailto:hello@robertz.co)**.

Previous releases stay under the terms they shipped with: versions ≤ 1.0.1 under
MIT, version 1.0.2 under PolyForm Noncommercial 1.0.0.
