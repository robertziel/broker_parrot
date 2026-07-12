# 🛑 Worker ON/OFF control

*The operator control plane that hard-stops or parks a `(host, queue)` worker via one Postgres table — migration 0012.*

## Desired vs observed state

Two tables, deliberately kept apart:

| table | what it is | who writes it | lifecycle |
|---|---|---|---|
| `worker_heartbeats` | **observed** state | the live worker, every ~10 s | ages out silently — a stopped worker just stops writing |
| `worker_controls` | **desired** state | an operator, a host UI, or the CLI | persists until explicitly changed |

An OFF row must survive precisely the window where the worker isn't beating — that's the whole point of an OFF switch. If desired state lived on `worker_heartbeats` it would either get clobbered by the next heartbeat upsert or vanish along with the row it's trying to suppress. Splitting the tables also means an operator's control write and the worker's heartbeat upsert never race each other.

`worker_controls` is keyed `(host_label, queue, project)` — the same identity `worker_heartbeats` and the claim's `claimed_by`/`queue` use, plus the migration-0019 tenant tag (see [schema](schema.md)). A host can run several workers under one `host_label` (a box runs a `cpu` and a `gpu` worker), so control is per-queue: turning off "host-a gpu" never touches "host-a cpu". And on a shared broker, `host_label` alone isn't unique — two projects can each run a worker on the same machine + queue — so the tenant term keeps an OFF for project A from also parking project B's worker there.

```
worker_controls(
  host_label      text,
  queue           text,             -- 'cpu' | 'gpu' | <ingest queue>
  project         text,             -- tenant tag, default '' (migration 0019)
  desired_state   text,             -- 'on' | 'off'          (CHECK)
  stop_policy     text,             -- 'hard' (default); free-form, validated in Python
  requested_by    text,
  updated_at      timestamptz,
  llm_server_type text,             -- 'ollama' | 'vllm'     (migration 0013)
  llm_parallelism integer,
  vllm_idle_ttl_s integer,
  PRIMARY KEY (host_label, queue, project)
)
```

## The NOTIFY trigger

A row trigger fires `pg_notify('worker_control', '<host>:<queue>')` on every INSERT/UPDATE, riding the writer's own transaction — there's no "row written, no wake" window. This is the same shape as the `node_job_ready` / `ingest_job_ready` triggers from migrations 0006/0007: because the wake is DB-native, **any** process that can write a row wakes the worker, with zero app-side NOTIFY code. A host Rails app sharing the Postgres instance can flip a worker off with a raw `INSERT … ON CONFLICT`.

The NOTIFY payload is deliberately just `host:queue` — no project segment, even after 0019 added the tenant column. Both the watcher and the boot park-gate re-read their own `(host, queue, project)` row on any wake rather than trusting the payload, so adding a tenant term would buy nothing; it would only break the pinned two-field payload for any external listener. The cost on a shared broker is a spurious wake when another tenant's row changes — the re-read is correct either way.

A **second**, separate trigger (migration 0013) fires `pg_notify('worker_llm_config_changed', '<host>|<queue>')` — note the `|` separator, not `:` — when an LLM-config column changes. It's a distinct channel on purpose: an LLM-config edit (switching `ollama`↔`vllm`, changing parallelism) must never look like an ON/OFF change to the `WorkerControlWatcher`. See [gpu_and_llm.md](gpu_and_llm.md).

## Enforcement — `WorkerControlWatcher`

`queue_workflows.worker_control.WorkerControlWatcher` is a daemon thread the claim worker starts once it's past its boot park-gate (mirrors `JobStatusWatcher`'s shape): a dedicated connection `LISTEN`s `worker_control`, with a `WORKER_CONTROL_POLL_S` (default 5 s, env `QUEUE_WORKFLOWS_WORKER_CONTROL_POLL_S`) safety poll behind it to catch a dropped NOTIFY or a row written before the worker booted.

On seeing `desired_state = 'off'` it looks up the row's `stop_policy` in the `STOP_POLICIES` registry and dispatches to the handler:

```python
STOP_POLICIES: dict[str, Callable[..., None]] = {
    "hard": _apply_hard_stop,
}
```

Only `"hard"` exists today. `"drain"` (finish the current job, then park) and `"pause"` (stop claiming, keep the model warm) are **reserved names** — the column exists to receive them later. That's exactly why `stop_policy` is free-form `TEXT` in the schema rather than a `CHECK`-constrained enum: a new policy is a new Python handler registered in `STOP_POLICIES`, no migration required. `set_worker_control` validates a requested policy against the registry *before* writing (fail-before-write); if a row somehow carries an unregistered policy, the watcher logs an error and falls back to `hard` rather than doing nothing.

## Hard stop = process exit (`os._exit(79)`)

The node body runs **inline on the worker's main thread** — nothing wraps it in a subprocess or a preemptible thread. That means a watcher thread can't cooperatively interrupt in-flight work, and a wedged CUDA kernel won't honor a cancel flag. Killing the **process** is the only thing that reliably stops the work and frees VRAM — the OS tears down the CUDA context on exit. This is the same lever every in-engine watchdog already pulls (see [watchdogs.md](watchdogs.md)); exit code `79` is the control plane's slot in that map: `75` budget · `76` stall · `77` job-status · `78` gpu-health · `79` worker-control.

`_apply_hard_stop` runs in this order:

1. **Re-queue the in-flight job first** — `worker.requeue_inflight_for_control()` flips the running row back to `queued`, resume-style, so the work redistributes to a healthy peer (or back to this same worker once it's re-enabled) immediately. Critically, this does **not** bump `watchdog_retries` — an operator turning a worker off is not a node fault, and the re-queue must not eat into the node's retry budget the way a watchdog trip does.
2. Best-effort logging of what happened (host, queue, jobs re-queued).
3. `on_exit(EXIT_CONTROL_HARD_STOP)` — `os._exit(79)` by default (injectable in tests).

The re-queue is wrapped so a failure there can't block the exit: if it raises, the exception is swallowed and the process still exits — the lease-reclaim sweep is the safety net that recovers any row the re-queue couldn't flip.

## Restart / park-on-boot

`os._exit` is a **non-zero** exit specifically so the container's `restart: on-failure` policy brings it back. On boot, before doing anything else, the claim worker calls `_park_until_enabled()`: it re-reads `worker_controls` and, if still OFF, parks — sits idle, does not claim, does not heartbeat (so it ages out of the capacity gauge) — while it `LISTEN`s the same `worker_control` channel plus a safety poll for the eventual ON. There is no in-process running→parked transition: a running worker that gets turned off always hard-exits, and parking only ever happens at this one boot gate.

```
operator/CLI/host UI ── INSERT/UPDATE worker_controls(off) ──► trigger ── NOTIFY 'worker_control' ──┐
                                                                                                      ▼
running worker: WorkerControlWatcher sees OFF ──► STOP_POLICIES['hard']
   1. requeue in-flight job(s), no watchdog_retries bump
   2. os._exit(79)                                                                                   │
                                                                                                      ▼
supervisor (restart: on-failure) restarts the container
   boot: _park_until_enabled() re-reads the row → still OFF → PARK (idle, no claim, no heartbeat)
                                                                                                      ▲
operator sets desired_state='on' ──► NOTIFY ──► parked loop resumes IN PLACE (no restart needed) ────┘
```

## Backward compatibility

`get_worker_control` never raises on a partially-migrated DB, and it never silently drops an OFF:

- **No row for this `(host, queue, project)`** ⇒ `None` ⇒ treated as ON. `desired_state_for` is the single decision point both the park-gate and the watcher consult: it returns `'off'` **only** when an explicit OFF row exists, otherwise `'on'`.
- **`UndefinedTable`** (a DB that predates migration 0012) ⇒ swallowed ⇒ `None` ⇒ ON. This is what lets the engine run unchanged on a pre-0012 database — claim workers gate startup on schema version 6/8 (the lease/ingest migrations), **not** 12, so a worker never blocks waiting for a control plane it doesn't need yet.
- **`UndefinedColumn`** (0012 applied, 0019 not) ⇒ retry the pre-0019 2-column lookup rather than falling through to `None` — an operator's OFF must not silently flip back ON just because the tenant column hasn't landed yet.

## The CLI

```
queue-worker-control --queue=gpu --off [--host HOST] [--policy hard] [--requested-by WHO] [--project PROJECT]
queue-worker-control --queue=gpu --on  [--host HOST]                 [--requested-by WHO] [--project PROJECT]
```

- `--queue` is required: `cpu` | `gpu` | any configured ingest queue.
- `--host` defaults to `QUEUE_WORKFLOWS_HOST_LABEL` (or the host-label env configured via `configure(host_label_env=...)`), falling back to `socket.gethostname()`.
- `--project` defaults to this process's configured `project` (`configure(project=...)` / `QUEUE_WORKFLOWS_PROJECT`); on a shared broker this is what keeps an OFF scoped to the right tenant.
- `--policy` defaults to `hard` (the only implemented policy today).

Equivalent Python:

```python
from queue_workflows import worker_control

worker_control.disable_worker("host-a", "gpu")      # hard stop + stay off
worker_control.enable_worker("host-a", "gpu")        # resume
worker_control.get_worker_control("host-a", "gpu")   # row dict or None
worker_control.desired_state_for("host-a", "gpu")    # 'on' | 'off' (no row / pre-0012 ⇒ 'on')
```

Or write the row directly over SQL from any DB consumer — the trigger does the rest:

```sql
INSERT INTO worker_controls (host_label, queue, project, desired_state, stop_policy)
VALUES ('host-a', 'gpu', '', 'off', 'hard')
ON CONFLICT (host_label, queue, project) DO UPDATE
    SET desired_state = EXCLUDED.desired_state, updated_at = now();
```

## Env knobs

- `QUEUE_WORKFLOWS_WORKER_CONTROL_POLL_S` — safety-poll cadence behind the `LISTEN` wake (default 5.0 s).
- `QUEUE_WORKFLOWS_DISABLE_WORKER_CONTROL` — keeps the watcher inert (tests).

## Related config on the same table

Migration 0013 added **per-machine LLM server config** (`llm_server_type`, `llm_parallelism`, `vllm_idle_ttl_s`) to `worker_controls` — same `(host_label, queue, project)` key, same operator/Rails write path, but a *soft* config change (never touches `desired_state`/`stop_policy`, never stops a running worker) delivered over its own NOTIFY channel. See [gpu_and_llm.md](gpu_and_llm.md).

Migration 0019 added the `project` column and re-keyed the primary key to `(host_label, queue, project)` for multi-tenant broker deploys — see [schema](schema.md) for the full migration chain and the tenant-pooling model.

## Residual gap

A hang that holds the GIL freezes the watcher thread along with everything else, so the in-process `os._exit` can't fire. The backstop for that case is a host-local agent with a docker socket that force-kills the container when it observes `worker_controls.desired_state = 'off'` going unacknowledged for too long — the engine itself has no cross-host kill capability. That agent is future hardening, not part of this milestone.
