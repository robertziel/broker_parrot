# Watchdogs — the liveness model 🐕

*How a wedged or dead worker gets recovered without a human paging in, from a live lease renewal up through an orchestrator-side dead-process detector.*

A claim worker is **concurrency-1** by contract: one process, one job at a time (a GPU worker additionally holds one warm model slot). That structural fact cuts both ways — it's why the engine can run a warm-model cache at all, but it also means a single wedged job takes the *entire* worker offline until something frees it. This doc covers every layer that "something": lease + reclaim (the sole recovery path for an orphaned row), three in-process daemon watchdogs (each a distinct hard-exit code), the one policy point they all funnel through, two state-watcher threads that aren't about liveness but share the same hard-exit shape, and a last-resort orchestrator-side detector for the case where the process itself is too wedged to save itself.

All watchdog and state-watcher classes live in `queue_workflows/claim_worker.py`; the physical GPU/RAM samplers live in `queue_workflows/gpu_health.py`; the reclaim sweeps and dead-worker detector live in `queue_workflows/node_queue.py`, driven from `NodePool._tick` in `queue_workflows/node_pool.py`.

## 1. Lease + reclaim — the baseline recovery path

Every claimed row (`workflow_node_jobs` or `ingest_jobs`) carries a `lease_expires_at`. A live worker pushes it forward via `LeaseRenewer`, a daemon thread that runs every `LEASE_RENEW_INTERVAL_S` (**10 s**) for the duration of the job, scoped to `id AND claimed_by` so it can never extend a lease a reclaim already handed to another worker. The lease itself is `LEASE_S` = `node_queue.DEFAULT_LEASE_S` (**600 s**).

Because the lease is renewed on a fixed cadence independent of the job's own duration, **lease length has nothing to do with how long a job is allowed to run** — a healthy 3-hour GPU render renews its lease ~1080 times and never comes close to lapsing. The lease exists to answer one question only: *is the worker that claimed this row still alive?*

A worker that dies or wedges stops renewing. Once `lease_expires_at` passes, the orchestrator's reclaim sweep (`node_queue.reclaim_expired_leases` / `reclaim_expired_ingest_leases`, run every tick from `NodePool._tick`) flips the row back to `queued`, clears the lease bookkeeping, and — for a DAG node-job whose parent run is still `running` — bumps it to the front of the queue (`priority = LEAST(priority, 10)`) so recovered work isn't stuck behind newer work. The flip re-fires the `node_job_ready` / `ingest_job_ready` NOTIFY (migration 0006/0007), so an idle peer wakes and re-claims at once rather than waiting on the 1 s safety poll.

A DAG node-job whose parent run has already gone `cancelled`/`failed`/`completed` is instead flipped to `cancelled` — re-queuing it would create a ghost the claim SQL will never pick up (its parent no longer matches "running"), leaving a phantom `+1 queued` in the gauges forever.

**This sweep is the sole recovery path for an orphaned `running` row.** Nothing else notices a row whose owning process silently vanished (crash, OOM-kill, host power loss) — every watchdog below assumes the worker process is *itself* the one detecting the problem. If the process never gets to run its own trip logic, lease-reclaim is what eventually frees the row (worst case: after the lease lapses, ~600 s).

## 2. The three daemon watchdogs

Every claimed job is bracketed by up to three daemon-thread watchdogs, each hard-exiting the process with a **distinct** exit code so the cause is legible straight from the exit status — no log-diving needed to tell a stall from a budget trip from an operator stop.

### `Watchdog` — wall-clock budget (exit `75`)

The simplest guard: a fixed deadline set at `start()` (`time.monotonic() + budget_s`), polled every `poll_s` (default 1 s). Trips the instant `elapsed >= budget_s`. Applied to **CPU + ingest jobs only** — GPU jobs are policed by health, not elapsed time (below). `budget_for(job)`:

| Job | Budget |
|---|---|
| GPU job, `required_model` ∈ `config.video_model_ids` | `VIDEO_BUDGET_S` = 1800 s |
| GPU job, any other model | `GPU_DEFAULT_BUDGET_S` = 8100 s (2.25 h) — computed for logging/display, but a GPU job is never bracketed by this class (see below) |
| `fetch` ingest sweep | `FETCH_BUDGET_S` = 7200 s (2 h) |
| `load` ingest sweep | `LOAD_BUDGET_S` = 3600 s (1 h) |
| host-defined (G1) ingest queue | `config.ingest_default_budget_s` |
| `__input__*` park node | `INPUT_BUDGET_S` = 120 s |
| any other CPU node | `CPU_BUDGET_S` = 2100 s (35 min) |

A GPU job still computes a `budget_for` value, but the `Watchdog` class itself is only *constructed* for the CPU/ingest path; a GPU job is bracketed by `GpuHealthWatchdog` instead, never by a wall-clock cap.

### `StallWatchdog` — no-progress deadline (exit `76`)

**Opt-in**, defense-in-depth for a non-video GPU node whose `run(...)` declares a `status_callback` parameter (`ClaimWorker._node_reports_progress`) — a node that never reports progress can't be told apart from a hung one, so it's left to whatever else polices it (for GPU, that's `GpuHealthWatchdog`). A **video** model steps too slowly per beat-segment to fit a 120 s window, so video nodes are excluded too.

It is **inert until the first `beat()`** — `_deadline` starts `None` and the poll loop only checks it once armed. The executor calls `beat()` once immediately after the model load completes, which is what arms the watchdog; a multi-minute cold load (observed ~6 min) therefore sits entirely outside the no-progress window. After that, each diffusion step beats again (`STALL_TIMEOUT_S` = **120.0 s**, chosen to be far larger than the observed ~12 s inter-step gap, polled every `STALL_POLL_S` = 5.0 s).

**A no-beat timeout alone is only a suspicion, never a verdict.** When the deadline lapses, `StallWatchdog` does not trip outright — it runs a short confirmation window (`STALL_CONFIRM_SAMPLES` = 3 reads, `STALL_CONFIRM_POLL_S` = 1.0 s apart) using the *same* `gpu_health` samplers and the *same* "GPU idle AND RAM static" predicate `GpuHealthWatchdog` uses:

- GPU busy at any sample (`max sm% > GPU_IDLE_PCT`, default 5) → the node is doing a legitimately slow step → **re-arm, don't kill**.
- RAM moved beyond `GPU_HEALTH_RAM_DELTA_MB` (default 5120 MB / 5 GiB) → the node is loading weights / staging / preparing → **re-arm, don't kill**.
- Only when GPU stayed idle **and** RAM stayed static across the whole confirmation window is it genuinely doing nothing → **trip**.

A multi-GB model load moves RAM far past the 5 GiB delta, so a cold or lazy load can never be confirmed as wedged — this is the fix for the false positive that motivated the watchdog ("should not kill if the GPU model is being loaded or preparing to start; only if it does nothing"). An unconfirmed suspicion emits a best-effort `stall_suspected` node event and re-arms rather than latching, so a transient slow patch followed by resumed progress fully recovers with no trip at all.

### `GpuHealthWatchdog` — the GPU guard, health-driven not wall-clock (exit `78`)

This is the primary guard for **every** GPU job — it replaces the old fixed wall-clock cap for GPU entirely (there is *no* `Watchdog` bracketing a GPU job). Every `GPU_HEALTH_INTERVAL_S` (default **300 s**, `AI_LEADS_GPU_HEALTH_INTERVAL_S`) it checkpoints the window: the peak per-container GPU utilization seen (`gpu_util_pct()`, sampled every `STALL_POLL_S` and remembered as a running max) and the change in this container's RAM since the last checkpoint (`container_ram_mb()`). It **trips iff, over the whole window, the GPU stayed idle (`max_util <= GPU_IDLE_PCT`, default 5%) *and* RAM was static (`|Δram| <= GPU_HEALTH_RAM_DELTA_MB`, default 5120 MB)** — no GPU work and no memory movement together mean wedged. A busy GPU at any point, *or* a RAM move past the delta (staging, decode, model swap), resets the window and the job keeps running no matter how long it takes.

It **arms at job start**, not on first beat — with a generous `GPU_HEALTH_LOAD_GRACE_S` (default **1200 s** / 20 min) first window. That's deliberately different from `StallWatchdog`'s inert-until-beat design, and it's safe for the same reason the trip rule is safe: a healthy model load *moves RAM* (can't trip the RAM-static half) and a healthy render *keeps the GPU busy* (can't trip the GPU-idle half), so being armed during load poses no false-positive risk — only a load that is *genuinely* hung (idle GPU and static RAM for the full 20 minutes) trips. The first `beat()` — the executor's post-load pulse, then any per-step progress beats — collapses the window down to the normal `interval_s` cadence.

**Why this replaced the old fixed GPU wall-clock cap.** The prior design bracketed GPU jobs with the same `Watchdog` used for CPU: a fixed budget (2.25 h generic / 30 min video). That failed in both directions on the hardware this engine targets:

- **False negative** — a Blackwell qwen inference hang sits model-resident at 0% GPU indefinitely; the wall-clock cap let it camp the full 8100 s (2.25 hours) of dead GPU before tripping, because "elapsed time" told it nothing about whether the job was actually doing anything.
- **False positive** — a long-but-healthy render (a big video job, a slow-but-legitimate diffusion run) could exceed the fixed budget purely by taking a while, and got killed for being slow rather than for being stuck.

Policing *health* instead of *time* fixes both: a truly wedged job (idle GPU, static RAM) trips in one `interval_s` window regardless of how young the job is; a healthy job — however long — is never killed for elapsed time alone.

### The per-container samplers (`gpu_health.py`)

Both `StallWatchdog`'s confirmation and `GpuHealthWatchdog`'s checkpoint read the same two samplers, deliberately scoped to *this* container rather than the box:

- **`gpu_util_pct()`** — tries `nvidia-smi pmon -c 1 -s u` first, which the NVIDIA driver renders in the caller's own PID namespace: called from inside the GPU worker container it returns rows for *only this container's* processes, so a co-tenant ollama sidecar sharing the same physical GPU is invisible and can't mask this container's idle state (or vice versa — verified: neither container's pmon lists the other's PID). A row's `sm%` of `-` (N/A, common when a process isn't actively issuing SM work) counts as 0; the max across rows is the busy signal. Falls back to the box-level probe (`hw_metrics._gpu_probe`) when pmon is unavailable (ROCm, or no `nvidia-smi`) — coarser attribution, but acceptable on the ROCm overflow host, which has no GPU sidecar to be confused with anyway.
- **`container_ram_mb()`** — reads cgroup v2 `memory.current` from the container's own namespaced cgroup root (`/sys/fs/cgroup/memory.current`), which needs no host cgroup mount; falls back to `/proc/self/status` VmRSS if that file is absent. Returns `None` only when neither is readable, and a `None` reading is treated as "no RAM signal this checkpoint" — never grounds for a trip on its own.

## 3. The single policy point: `_watchdog_trip`

All three daemon watchdogs — and only them — funnel their trip through one function, `claim_worker._watchdog_trip`, so the requeue-vs-fail decision lives in exactly one place.

**The policy is re-queue-and-retry, not fail-on-first-trip** (migration `0010` added `workflow_node_jobs.watchdog_retries`):

- **DAG node-job** (`workflow_node_jobs`): read the row's current `watchdog_retries`. If it's below `AI_LEADS_WATCHDOG_MAX_RETRIES` (default **3**, read live at trip-time via `_watchdog_max_retries()` so an env override needs no restart), the trip is treated as a transient wedge — `_requeue_job_and_exit` flips the row `running → queued`, clears the lease, bumps priority to the front (`LEAST(priority, 10)` — the identical mechanic `reclaim_expired_leases` uses), and increments `watchdog_retries`. **It writes no dispatch event** — the parent *run* stays `running`; only this one node re-runs. The status flip's `node_job_ready` NOTIFY wakes an idle peer to re-claim immediately.
- Only once `watchdog_retries` reaches the cap does it fall back to `_fail_job_and_exit`: mark the row `failed` **and** write the `failed` dispatch-event row in **one transaction** — the same outbox-atomicity contract the rest of the engine uses, coded in exactly this one place so no caller can accidentally split the mark from the event.
- **Ingest job** (`ingest_jobs`): always `_fail_job_and_exit` — there's no parent run to keep alive and no `watchdog_retries` column; `ingest_jobs` has its own `reclaim_expired_ingest_leases` re-queue path for lease-based recovery.

Either exit path calls `_clear_busy_ghost(host_label, queue)` first: because `os._exit` skips `_run_node`'s `finally`, `ModelCache.mark_idle` and the heartbeat refresh never run on a hard exit, so the worker's last `worker_heartbeats` row would otherwise keep advertising a `current_model` after the process is gone. `_clear_busy_ghost` nulls `current_model` and ages `last_seen` so a dead worker's phantom "busy" status drops out of any gauge immediately rather than waiting up to 30 s to age out. It's best-effort and swallows all errors — a hung DB write must never block the hard-exit itself.

Both requeue and fail are logged with the retry count so an operator can see, per trip, exactly which attempt a node is on and whether it was retried or finally failed. **Every trip is also recorded as a `workflow_node_events` row** (`stall_trip` / `gpu_health_trip` / `budget_trip`, keyed by `_trip_event_type(label)`) regardless of the requeue-vs-fail outcome, plus a matching `requeued` or `failed` event from whichever exit path runs — the append-only per-attempt forensic log this engine keeps alongside the mutable `workflow_node_jobs` row.

**Why re-queuing a `running` row is safe (no double-run).** The re-queue clears `claimed_by` and flips `status='queued'`, then the tripping worker hard-exits. A fresh claim is the same CAS-guarded `queued → running` UPDATE any claim uses, so at most one worker wins it. And if another process still held a stale reference to the row, `JobStatusWatcher` (below) is exactly the mechanism that self-kills it the moment `claimed_by` no longer matches.

## 4. Two state-watcher threads (not watchdogs, same hard-exit shape)

These aren't liveness/health guards — they hard-exit for a *correctness* reason, not because anything is wedged — but they share the daemon-thread-with-a-distinct-exit-code shape, so they're covered here for completeness.

### `JobStatusWatcher` (exit `77`)

Polls `workflow_node_jobs` every `poll_s` (default 2.0 s) and hard-exits the instant this worker's claimed row is no longer `claimed_by` it — i.e. some other actor (a lease reclaim, an operator reassignment) took the row out from under it. Scoped to `claimed_by` specifically (not bare `status`) so the worker's *own* terminal mark — `mark_completed`/`mark_failed` leave `claimed_by` intact — never self-trips it; only an external hand-off, which clears or changes `claimed_by`, does. This is precisely what makes re-queuing a still-`running` row (§3) safe: instead of racing the new claimant to a double-run, the displaced worker notices and kills itself. `os._exit` is the only reliable way to abandon a node body stuck deep in a CUDA kernel — there's no cooperative unwind to ask for.

(A related but distinct mechanism, `cancel_watcher.py`'s `_start_run_cancel_watcher`, polls a run's status and sets a cooperative `cancel_event` a node body can check mid-execution to unwind early on a user-initiated cancel. It has no exit code of its own — it's a signal a well-behaved node *opts into* reading, not a hard-exit guard.)

### `WorkerControlWatcher` (exit `79`)

Enforces the operator ON/OFF control plane (migration `0012`) — `LISTEN worker_control` plus a periodic safety poll catches both a dropped NOTIFY and a control row written before this worker booted. When an operator sets `(host_label, queue)` to `desired_state='off'`, it dispatches the row's `stop_policy` (today only `"hard"` exists) — hard stop re-queues the in-flight job resume-style (**no** `watchdog_retries` bump — an operator-requested stop isn't a fault) and then `os._exit(79)`. Full design, the `STOP_POLICIES` registry, and the boot-time park gate live in [`worker_control.md`](worker_control.md).

## 5. Last resort: the orchestrator-side dead-worker detector

Every watchdog above is an **in-process daemon thread**. That's a shared, fatal assumption: it only works if the worker's Python interpreter is still schedulable enough to run those threads and act on a trip. Two ways that assumption breaks:

- **The trip signal itself can be unobservable from inside.** On a ROCm box, `gpu_util_pct()` has no per-process `pmon` path and falls back to the *box-level* probe — which can read non-idle (driver noise, a co-tenant process, contention) even while *this specific* render is wedged. A hung render also holds its weights resident, so container RAM reads static. "GPU idle AND RAM static" then never both hold, so `GpuHealthWatchdog` never trips — not because the design is wrong, but because the box-level fallback can't see what the per-container path could.
- **A GIL-holding hang can stop the daemon threads from running at all.** If the wedge sits inside a call that holds the GIL, no Python thread in the process — including the watchdog — gets scheduled, so there's no trip logic to run in the first place.

The orchestrator (`NodePool`) is a **separate process** — GIL-independent of any wedged worker — so it can observe what the worker itself cannot. `NodePool._tick` runs `_sweep_dead_workers` (interval-gated to every 5 s so the ~0.5 s dispatch tick doesn't hammer the detector query), which calls `node_queue.flag_stale_workers_holding_running_jobs`:

- Finds every `worker_heartbeats` row whose `last_seen` is older than `stale_after_s` (default **30 s** = `node_queue.STALE_WORKER_AFTER_S`, 3× the 10 s heartbeat cadence — the same freshness window a queue-liveness gauge uses to call a row stale) **while it still owns ≥ 1 `running` job** — the join is `j.claimed_by = wh.host_label AND j.queue = wh.queue` (plus a `project` match), so a host running both a CPU and a GPU worker under one host label doesn't get a healthy sibling flagged by a wedged one.
- Stamps `worker_heartbeats.last_flagged_dead_at = now()` (migration `0009`) on the matching rows and returns them so the caller logs one `DEAD WORKER:` **ERROR** per flagged worker, naming the host/queue/project, how long the heartbeat has been stale, and how many running jobs it still owns.
- Is **idempotent**: only rows whose `last_flagged_dead_at` is `NULL` or itself older than the threshold get (re)flagged, so the frequent tick doesn't relog the same dead worker every pass — it flags once and stays quiet until a fresh heartbeat clears the flag (`upsert_worker_heartbeat`), then flags again if it goes stale a second time.

**The recovery split is deliberate.** The *job* is already recovered by the ordinary lease-reclaim sweep (§1) — that doesn't need this detector at all, since a lease-reclaim only cares that renewal stopped, not why. What this detector adds is flagging the dead **process** so a host-supervisor can bounce it: the orchestrator has no docker socket and is frequently on a different host than the worker, so it cannot safely cross-host-kill a container — it can only surface a durable, queryable signal (`last_flagged_dead_at` + the ERROR log line) for something with that authority to act on.

## 6. Exit-code summary

| Code | Source | Trigger | Outcome |
|---|---|---|---|
| **75** | `Watchdog` | wall-clock `elapsed >= budget_for(job)` (CPU + ingest jobs only) | `_watchdog_trip` → re-queue+retry under the cap, else fail |
| **76** | `StallWatchdog` | no `beat()` for `STALL_TIMEOUT_S` (120 s) *and* confirmed GPU-idle + RAM-static (opt-in, non-video GPU nodes with `status_callback`) | `_watchdog_trip` → re-queue+retry under the cap, else fail |
| **77** | `JobStatusWatcher` | this worker's claimed row's `claimed_by` no longer matches it (reassigned/reclaimed out from under it) | self-kill only — no requeue/fail (another actor already changed the row) |
| **78** | `GpuHealthWatchdog` | GPU idle (`<= GPU_IDLE_PCT`) *and* RAM static (`<= GPU_HEALTH_RAM_DELTA_MB`) across a full `GPU_HEALTH_INTERVAL_S` (300 s) window (every GPU job) | `_watchdog_trip` → re-queue+retry under the cap, else fail |
| **79** | `WorkerControlWatcher` | operator sets `worker_controls.desired_state = 'off'` for this `(host, queue)` | hard stop: resume-style re-queue (no `watchdog_retries` bump), then exit — see [`worker_control.md`](worker_control.md) |

All five exits are `os._exit` (or an injected `on_exit` in tests) — a hard process kill, not a Python-level exception, because the only thing guaranteed to free a worker stuck deep in a CUDA kernel or a blocked driver call is terminating the process outright. systemd (or the container supervisor) restarts the process; on boot it re-reads any operator control state and either resumes claiming or parks, per [`worker_control.md`](worker_control.md).

## Related docs

- [`worker_control.md`](worker_control.md) — the operator ON/OFF control plane that owns exit `79` and the `"hard"` stop policy.
- [`architecture.md`](architecture.md) — the three process roles, the outbox pattern `_fail_job_and_exit` writes into, and how the dispatch loop consumes a node's terminal state.
- [`schema.md`](schema.md) — `workflow_node_jobs.watchdog_retries` (migration 0010), `workflow_node_events` (migration 0011), `worker_heartbeats.last_flagged_dead_at` (migration 0009), and the full migration chain (currently through 0019).
- [`gpu_and_llm.md`](gpu_and_llm.md) — the warm `ModelCache`, the GPU worker's inline + pool lanes, and how `current_model` feeds both claim affinity and the busy-ghost cleanup this doc's watchdogs perform on exit.
- [`configuration.md`](configuration.md) — every `AI_LEADS_*` env knob named above (`AI_LEADS_WATCHDOG_MAX_RETRIES`, `AI_LEADS_GPU_HEALTH_INTERVAL_S`, `AI_LEADS_STALE_WORKER_AFTER_S`, …) and their defaults in one place.
- [`deployment.md`](deployment.md) — the host-supervisor hook that consumes a `DEAD WORKER:` flag and bounces the container.
