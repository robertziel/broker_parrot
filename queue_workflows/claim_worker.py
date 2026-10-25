"""Postgres-as-queue claim worker — the live job runtime.

A per-worker loop: a Postgres ``SELECT … FOR UPDATE SKIP LOCKED`` claim woken
by ``LISTEN node_job_ready`` (migration 0006) with a 1 s safety poll. One
process per worker (``queue-claim-worker --queue={cpu,gpu,fetch,load}`` or the
host's launcher); the dispatcher INSERTs job rows and the migration-0006/0007
trigger NOTIFYs this loop.

Loop shape (one process == one worker, concurrency-1 structural):

    LISTEN node_job_ready
    while running:
        claimed = claim_next_{cpu,gpu}_job(host, host_priority, current_model,
                                           lease_s)
        if claimed:
            run it through execute_node under a lease-renewer + a per-job
            watchdog, then re-loop immediately (drain greedily)
        else:
            block on the notify with a 1 s timeout, then re-loop

Daemon threads bracket each claimed job:

  * :class:`LeaseRenewer` — every ~10 s pushes ``lease_expires_at`` out so a
    long job keeps its lease; scoped to ``id AND claimed_by`` so it can't
    extend a lease a reclaim already handed to another worker.
  * :class:`Watchdog` — a wall-clock budget. On trip it RE-QUEUES the node for a
    retry on a fresh worker (DAG jobs; run stays alive) under a per-job cap, then
    fails the run once the cap is reached — and HARD-exits the process either way.
    Applied to CPU + ingest jobs only (ingest trips always mark failed — no run).
  * :class:`GpuHealthWatchdog` — the GPU guard, HEALTH-driven not wall-clock:
    it replaces the fixed budget for GPU jobs and kills ONLY a truly-wedged
    worker (no per-container GPU work AND static container RAM over a 5-min
    window). A busy GPU or a > 5 GB RAM move keeps the job alive no matter how
    long it runs — there is NO fixed time cap for GPU renders. It arms AT job
    start with a generous load-grace first window (so a load-phase hang, and a
    GPU node that never beats, are both bounded), then falls to the 5-min
    cadence on the first beat.
  * :class:`StallWatchdog` — a tight no-progress deadline kept for non-video
    GPU nodes (defense in depth: catches a fast 0%-GPU hang in ~2 min). Its trip
    is GATED on the physical signal: a no-beat timeout only SUSPECTS a stall, then
    it confirms GPU-idle AND RAM-static before killing, so a loading / preparing /
    slow-but-working node is never a false positive.

GPU jobs are policed by HEALTH, never by elapsed time — a long-but-healthy
render is never killed; a wedged one is caught by the health watchdog (and, for
non-video, the stall watchdog). A trip RE-QUEUES the node for a retry on a fresh
worker (the run stays ``running``) up to ``QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES`` times,
then fails the run — so a single transient wedge no longer kills a whole workflow.
On a hard-exit the trip path also clears this worker's ``current_model`` busy-ghost
so a killed worker doesn't keep inflating the GPU-busy gauge.
"""

from __future__ import annotations

from collections.abc import Iterable

import inspect
import logging
import os

from queue_workflows.envcompat import env_get
import socket
import threading
import time
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from queue_workflows import ingest_store, node_executor, node_queue, worker_control
from queue_workflows.config import get_config
from queue_workflows.db import connection, db_url

log = logging.getLogger(__name__)

#: The box-lease sentinel a switching worker holds. DISTINCT from every real model id
#: and from LLM_SERVER_SLOT/COMFYUI_SERVER_SLOT on purpose: a distinct value means a
#: peer asking for ANY model is denied (spills to another box) while the switch runs.
SWITCH_SLOT = "__server_switch__"


class _ClusterHold:
    """One CLUSTER RESERVATION this lane is holding (``ClaimWorker._begin_cluster_hold``): the
    job it is for, the quiesce that keeps the broker's servers down, and the watch that ticks it."""

    __slots__ = ("job_id", "purpose", "quiesce", "stop", "thread")

    def __init__(self, *, job_id: str, purpose: str, quiesce: Any) -> None:
        self.job_id = job_id
        self.purpose = purpose
        self.quiesce = quiesce
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

#: How long a gpu worker stops claiming after its own box denied it the card, and the
#: ceiling that hold doubles up to. A denied acquire re-queues the job to the FRONT
#: (it is a resume, not a fault), so without a brake the very next claim returns the
#: same row and is denied again: dozens of spills of ONE job within seconds were seen
#: after a lane restart left a dead process's SWITCH_SLOT holder in the box lease. The
#: ceiling stays at or under the lease TTL so a box whose stale holder simply expired
#: resumes promptly rather than sitting out a brake that outlives the cause.
SPILL_PAUSE_S = 1.0
SPILL_PAUSE_MAX_S = 30.0
#: How often the same spill verdict is restated at WARNING (a different job is always
#: reported).
SPILL_REPORT_EVERY_S = 60.0


def _switch_timeouts() -> tuple[float, float]:
    """The switch phase budgets, env-overridable per deployment. One box loads
    its vLLM weights over a network mount (many minutes); a box with slower storage raises
    QUEUE_WORKFLOWS_SWITCH_WARM_TIMEOUT_S rather than editing a constant. Malformed
    values fall through to the defaults."""
    from queue_workflows import server_switch as _ss

    def _read(name: str, default: float) -> float:
        raw = (env_get(name, "") or "").strip()
        if raw:
            try:
                return max(30.0, float(raw))
            except ValueError:
                pass
        return default

    return (
        _read("QUEUE_WORKFLOWS_SWITCH_DRAIN_TIMEOUT_S", _ss.DEFAULT_DRAIN_TIMEOUT_S),
        _read("QUEUE_WORKFLOWS_SWITCH_WARM_TIMEOUT_S", _ss.DEFAULT_WARM_TIMEOUT_S),
    )

#: Consecutive missed probes before a server kind leaves the advertised set. At the
#: 10 s heartbeat cadence, 3 misses ≈ 30 s.
_SERVED_DEBOUNCE_MISSES = 3

#: How long a sized job must have been queued before this box will restart its server
#: for it. Below this, "no box serves that window" is as likely to be heartbeat lag as
#: a real gap — and the restart costs the box its whole out-of-service window.
CONTEXT_SWITCH_MIN_AGE_S = 30.0


#: The smallest ``--gpu-memory-utilization`` worth starting vLLM at. Below this the pool
#: cannot hold the weights and a useful cache, so the box does not start vLLM at all —
#: the memory gate that keeps a box from filling its memory to the last byte.
DEFAULT_VLLM_MIN_UTILIZATION = 0.30


def _vllm_min_utilization() -> float:
    raw = (env_get("QUEUE_WORKFLOWS_VLLM_MIN_UTILIZATION", "") or "").strip()
    try:
        value = float(raw) if raw else DEFAULT_VLLM_MIN_UTILIZATION
    except ValueError:
        return DEFAULT_VLLM_MIN_UTILIZATION
    return value if 0.0 < value <= 1.0 else DEFAULT_VLLM_MIN_UTILIZATION


#: Fewer concurrent requests than this and vLLM is not worth starting where ollama can
#: serve instead: vLLM's advantage over ollama is BATCHING, and a pool that pays for one
#: request buys nothing ollama does not already give — at a fraction of the memory, with
#: no switch, no weight reload and no warm window.
DEFAULT_VLLM_MIN_CONCURRENCY = 2


#: How little free memory means the box is in trouble, whatever anyone planned
#: (``QUEUE_WORKFLOWS_LLM_DANGER_FLOOR_MB``). 12 GB: a box can lose sshd at
#: ~8 GB MemAvailable, and a box at 12 GB is one model load from that.
DEFAULT_DANGER_FLOOR_MB = 12288


def _danger_floor_bytes() -> int:
    raw = (env_get("QUEUE_WORKFLOWS_LLM_DANGER_FLOOR_MB", "") or "").strip()
    try:
        mb = int(raw) if raw else DEFAULT_DANGER_FLOOR_MB
    except ValueError:
        mb = DEFAULT_DANGER_FLOOR_MB
    return max(0, mb) * 1024 * 1024


def _vllm_min_concurrency() -> int:
    raw = (env_get("QUEUE_WORKFLOWS_VLLM_MIN_CONCURRENCY", "") or "").strip()
    try:
        value = int(raw) if raw else DEFAULT_VLLM_MIN_CONCURRENCY
    except ValueError:
        return DEFAULT_VLLM_MIN_CONCURRENCY
    return max(1, value)


def _vllm_model_max_context() -> int | None:
    """The largest window this box's vLLM model supports
    (``QUEUE_WORKFLOWS_VLLM_MAX_CONTEXT``), or ``None``.

    vLLM does not publish it — ``/v1/models`` reports the window it was STARTED with,
    which is exactly the number we would be trying to raise. And starting it beyond the
    model's own maximum is not a soft failure: vLLM exits at startup, leaving the box
    with no server until the switch aborts. So without an operator statement a vLLM box
    advertises no room to grow, and only its model swap changes its window."""
    raw = (env_get("QUEUE_WORKFLOWS_VLLM_MAX_CONTEXT", "") or "").strip()
    try:
        value = int(raw) if raw else 0
    except ValueError:
        return None
    return value if value > 0 else None


# ── wall-clock budgets ─────────────────────────────────────────────────────
GPU_DEFAULT_BUDGET_S = 8100      # 2.25 h hard — default GPU job
VIDEO_BUDGET_S = 1800            # 30 min — video render (config.video_model_ids)
CPU_BUDGET_S = 2100              # 35 min hard — default CPU job
INPUT_BUDGET_S = 120             # input (park-and-return) node
DOWNLOAD_BUDGET_S = 3600         # 1 h — network-bound download node (a big
                                 # dataset/tile warm on a slow feed; a trip
                                 # re-queues + retries like any node job)
# Ingest: the PG run_fetch_all / run_load_all run the whole per-(source,scope)
# sweep INLINE, so the budget must cover the slowest full sweep.
FETCH_BUDGET_S = 2 * 3600        # 2 h — full network sweep across all sources
LOAD_BUDGET_S = 3600             # 1 h — full landing-zone → Postgres sweep

# Lease length + renewal cadence. The lease is renewed every
# ``LEASE_RENEW_INTERVAL_S`` while a job runs, so its absolute length is
# independent of how long the job takes.
LEASE_S = node_queue.DEFAULT_LEASE_S          # 600 s
LEASE_RENEW_INTERVAL_S = 10.0
NOTIFY_POLL_TIMEOUT_S = 1.0                   # safety poll for a dropped NOTIFY

# No-progress (stall) deadline for a GPU node. The wall-clock budget above only
# catches a job that runs too LONG; this catches one that makes NO PROGRESS — a
# an LLM inference hang sits model-resident at 0 % GPU and would camp
# the full 8100 s budget. A node beats this deadline once per diffusion step
# (via ``status_callback``); the executor beats once more right after the model
# load completes — which ARMS the watchdog. It is inert before that first beat,
# so a multi-minute cold model load (observed ~6 min) can't false-trip it. The
# window only spans the gap between two diffusion steps (~12 s), so it can be
# tight. CRITICAL (Part A): no beat for this long once armed is only a SUSPECTED
# stall, not a verdict — the watchdog then CONFIRMS against the physical GPU/RAM
# signal (GPU idle AND RAM static) and trips ONLY if the worker is genuinely
# doing nothing; a loading / preparing / slow-but-working node (busy GPU or moving
# RAM) is re-armed, never killed. A confirmed wedge re-queues the node + hard-exits
# so the lease reclaim retries it on a healthy host (run stays alive, under a cap).
STALL_TIMEOUT_S = 120.0                        # ≫ inter-step gap (~12 s); load is excluded
STALL_POLL_S = 5.0

# A CLUSTER RESERVATION's lease, deliberately shorter than LEASE_S (see the LeaseRenewer in
# _run_pool_node). What releases a hold whose owner dies? Answered where it actually lives: nothing enforces a hold whose lane is gone, yet the row
# reads 'running' to llm-gateway until its lease lapses. The reservation's liveness IS its
# renewal, every LEASE_RENEW_INTERVAL_S, so a long lease buys nothing but a long lie. 60 s is
# 6x the renew interval — the same margin every other lease runs on — and the orchestrator's
# 5 s reclaim sweep then re-queues a dead holder's row within ~a minute instead of ~ten.
CLUSTER_RESERVATION_LEASE_S = 60

# How long a cluster hold lets work ALREADY IN FLIGHT on this lane finish before it takes the
# card anyway (ParkQuiesce's hold-off ceiling, in the 'cluster-hold' voice). llm-gateway waits
# ~60 s for the RAM before it gives up, so the drain sits comfortably
# inside that window: a job that ends sooner is never disturbed, and one that never ends is
# not a reason to starve the cluster the box was reserved for.
CLUSTER_HOLD_DRAIN_MAX_S = 30.0

# How long a CLUSTER RESERVATION waits for the box lease before it gives the box back. It does
# NOT spill behind the card it was sent to take: the occupant is usually the very LLM model its
# own hold is unloading, and each re-queue-and-release re-opened the box to LLM claims —
# a rapid HOLDING/RELEASED flap. So it waits, ticking
# its hold (which is what clears the card), and only spills if something it cannot clear — a
# render from another project — is still there at the ceiling.
CLUSTER_RESERVATION_ACQUIRE_MAX_S = 120.0
CLUSTER_RESERVATION_ACQUIRE_POLL_S = 2.0

# The POOL lane's no-progress guard (an LLM / VLM facade node waiting on the box's
# ollama / vLLM). OPT-IN PER NODE, and the node names its own window: a module-level
# ``POOL_STALL_TIMEOUT_S`` on a node whose ``run`` declares ``status_callback``. It is
# not armed merely because ``status_callback`` is declared — ComfyUI renders run on
# this lane and beat once per sampler step, minutes apart on a video model, and a
# fixed window would kill them. The floor stops a typo'd tiny window from killing
# every healthy job in the gap between two requests (a prompt prefill + a model
# swap); the kill switch disarms it fleet-wide without a redeploy.
POOL_STALL_MIN_TIMEOUT_S = 30.0
_POOL_STALL_DISABLE_ENV = "QUEUE_WORKFLOWS_DISABLE_POOL_STALL_WATCHDOG"


# ── GPU health watchdog thresholds (HEALTH-driven, NOT wall-clock) ──────────
# The GPU health watchdog replaces the fixed wall-clock budget for GPU jobs: it
# never kills a job just because time passed. Instead, every
# ``GPU_HEALTH_INTERVAL_S`` it asks "is this worker actually using the GPU, or
# moving memory?" — and only kills a truly-wedged worker (no GPU work AND static
# RAM over the whole window). A busy GPU or a > RAM_DELTA memory move means the
# job is healthy and is NEVER policed by time. Every threshold is env-overridable.
#
#   * GPU_HEALTH_INTERVAL_S  — checkpoint cadence; the window the trip rule spans.
#   * GPU_IDLE_PCT           — per-container GPU util at/under which GPU counts as idle.
#   * GPU_HEALTH_RAM_DELTA_MB — |RAM change| over the window above which the job
#                              counts as doing work (e.g. staging / decode), so it
#                              is NOT killed even if GPU util reads idle.
# GPU util is sampled on the same short ``STALL_POLL_S`` cadence and the MAX over
# the window is the busy signal (a single denoise burst keeps the window healthy).


def _env_float(name: str, default: float) -> float:
    try:
        return float((env_get(name) or "").strip() or default)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float((env_get(name) or "").strip() or default))
    except (TypeError, ValueError):
        return int(default)


GPU_HEALTH_INTERVAL_S = _env_float("QUEUE_WORKFLOWS_GPU_HEALTH_INTERVAL_S", 300.0)
GPU_IDLE_PCT = _env_int("QUEUE_WORKFLOWS_GPU_IDLE_PCT", 5)
GPU_HEALTH_RAM_DELTA_MB = _env_int("QUEUE_WORKFLOWS_GPU_HEALTH_RAM_DELTA_MB", 5120)  # 5 GiB
# First-window length when the watchdog arms AT job start (before any beat).
# Generous (20 min) so a healthy-but-slow cold model load — which moves RAM as
# weights load and so can't trip the "GPU idle AND RAM static" rule anyway — is
# never policed tightly; it only bounds a load that is GENUINELY hung (idle GPU
# AND static RAM) past this grace. The first beat (executor post-load / per
# step) collapses the window to the normal GPU_HEALTH_INTERVAL_S cadence.
GPU_HEALTH_LOAD_GRACE_S = _env_float("QUEUE_WORKFLOWS_GPU_HEALTH_LOAD_GRACE_S", 1200.0)  # 20 min

# ── watchdog re-queue-with-cap policy ───────────────────────────────────────
# A watchdog trip RE-QUEUES the node for a retry on a fresh worker (flip
# running→queued, clear the lease, bump priority to the front, increment the
# per-job ``watchdog_retries`` counter — see node_queue.requeue_job_for_retry)
# rather than failing the whole run. A node that wedges EVERY time would loop
# forever, so the retry is CAPPED: once a job's ``watchdog_retries`` reaches this
# many, the trip falls back to the old mark-failed path (→ run fails). The cap is
# read at trip-time (not captured at import) so an ops env-override takes effect
# without a process restart, and so tests can flip it per-case.
WATCHDOG_MAX_RETRIES = _env_int("QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES", 3)


def _watchdog_max_retries() -> int:
    """The watchdog re-queue cap, read live from ``QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES``
    (default 3). Read at trip-time so an env-override needs no restart."""
    return _env_int("QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES", WATCHDOG_MAX_RETRIES)


# ── cpu lane width ──────────────────────────────────────────────────────────
#: How many node-jobs ONE cpu worker process runs at a time. 1 = the historical
#: concurrency-1 worker (claim, run inline, loop) and the default.
DEFAULT_CPU_WORKER_THREADS = 1


def resolve_cpu_worker_threads() -> int:
    """The effective cpu lane width, resolved at CALL time so a host's
    ``configure()`` (which runs after import) is honoured: the
    ``QUEUE_WORKFLOWS_CPU_WORKER_THREADS`` env knob wins (operator override,
    read via envcompat), then ``EngineConfig.cpu_worker_threads``,
    then :data:`DEFAULT_CPU_WORKER_THREADS`.

    Always ``>= 1``: a malformed or sub-1 value falls through to the floor rather
    than producing a lane that claims nothing — an unparseable knob must degrade
    to the safe default, never to a silently idle worker.

    Mirrors :func:`queue_workflows.llm_idle_unload.resolve_llm_idle_ttl_s`."""
    raw = env_get("QUEUE_WORKFLOWS_CPU_WORKER_THREADS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    try:
        val = getattr(get_config(), "cpu_worker_threads", None)
        if val is not None:
            return max(1, int(val))
    except Exception:
        pass
    return DEFAULT_CPU_WORKER_THREADS


#: The env twin of ``EngineConfig.gpu_box_max_concurrent``. Carries the whole
#: mapping in one variable — ``"box-b=3,box-c=3"`` — so a single rendered
#: ``.env`` fits the fleet; a bare integer (``"3"``) means "whichever box reads
#: this", for a machine whose compose file is its own.
BOX_MAX_CONCURRENT_ENV = "QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT"

#: Key for the bare-integer entry in a parsed ceiling map: "this box, whichever
#: it is". Not a legal box id (``default_box_id()`` never returns empty), so it
#: can never collide with a named one.
_THIS_BOX = ""


def _parse_box_ceilings(raw: str) -> dict[str, int]:
    """Parse ``"box-b=3,box-c=3"`` (or a bare ``"3"``) into ``{box: ceiling}``.

    ONE BAD ENTRY COSTS ONLY THAT ENTRY. A fleet-wide variable is edited by hand,
    and a typo in one machine's entry must not silently lift the ceiling off every
    other machine — so each entry is parsed independently and an unreadable or
    non-positive one is dropped (see :func:`resolve_box_max_concurrent` for why a
    0 is never obeyed)."""
    out: dict[str, int] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            # An entry that carries an '=' is NAMING a box, so it must name one:
            # a bare '=3' names nothing and is a typo, not "every box". The bare
            # fleet default has its own unambiguous spelling (no '=' at all).
            box, _, value = entry.partition("=")
            box = box.strip()
            if not box:
                log.warning(
                    "[claim-worker:gpu] %s: ignoring %r — an entry with '=' must "
                    "name a box (write a bare N for 'this box')",
                    BOX_MAX_CONCURRENT_ENV, entry,
                )
                continue
        else:
            box, value = _THIS_BOX, entry
        try:
            n = int(value.strip())
        except ValueError:
            log.warning(
                "[claim-worker:gpu] %s: ignoring unreadable ceiling %r "
                "(expected 'box=N' or a bare N)", BOX_MAX_CONCURRENT_ENV, entry,
            )
            continue
        if n <= 0:
            log.warning(
                "[claim-worker:gpu] %s: ignoring ceiling %r — a ceiling of %d is a "
                "GPU lane that claims nothing; the box keeps its computed "
                "concurrency", BOX_MAX_CONCURRENT_ENV, entry, n,
            )
            continue
        out[box.strip()] = n
    return out


def resolve_box_max_concurrent(box: str) -> int | None:
    """This BOX's operator ceiling on concurrent GPU node-jobs, or ``None`` for
    "no ceiling" — resolved at CALL time so a host's ``configure()`` (which runs
    after import) and an operator's live env are both honoured. Precedence matches
    every other knob in this engine (``cpu_worker_threads``,
    ``download_max_concurrent``): :data:`BOX_MAX_CONCURRENT_ENV` wins (read
    via envcompat), then
    ``EngineConfig.gpu_box_max_concurrent``.

    Precedence is PER ENTRY, not per map: an env that names box-c overrides
    box-c and leaves box-b's configured ceiling standing. Whole-map replacement
    would mean an operator adding one machine to the variable silently REMOVES
    every other machine's ceiling — and silently raising a ceiling is the one
    direction this knob must never move by accident.

    ``box`` is the PHYSICAL box identity (``gpu_model_lease.default_box_id()``,
    e.g. ``box-b``) — the name ``avoid_box``/``force_box`` place against, NOT the
    per-project lane label (``box-b-gpu``): every project's lane on a machine
    shares one card, so the ceiling is one per machine, not one per lane.

    A box nobody named returns ``None`` and keeps today's behaviour exactly.

    NONSENSE IS IGNORED, NEVER OBEYED. 0, a negative, an unparseable value — all
    return ``None`` rather than a ceiling. A ceiling of 0 would be a GPU lane that
    claims nothing, silently, forever, on a machine the operator believes is
    working; a typo in an env var must not be able to do that (the standing rule
    is NEVER PARK — a box is shed or capped, never idled by accident)."""
    raw = env_get(BOX_MAX_CONCURRENT_ENV)
    if raw and raw.strip():
        ceilings = _parse_box_ceilings(raw)
        # A named entry is the more specific statement; the bare integer is the
        # fleet default carried in the same variable.
        hit = ceilings.get(box, ceilings.get(_THIS_BOX))
        if hit is not None:
            return hit
    try:
        configured = getattr(get_config(), "gpu_box_max_concurrent", None) or {}
        val = configured.get(box)
        if val is None:
            return None
        n = int(val)
    except Exception:
        # A host that wired a nonsense value (a string, a None, a broken mapping)
        # gets today's behaviour, not a parked lane.
        return None
    return n if n > 0 else None


# Worker capacity heartbeat refresh cadence. A host's queue gauge may treat a row
# whose ``last_seen`` is older than 30 s as stale, so a 10 s refresh keeps us
# comfortably inside that window.
HEARTBEAT_INTERVAL_S = 10.0

# Lowest engine migration version each queue family's claim loop needs before
# polling (the ``queue_schema_version`` ledger). cpu/gpu draw from
# ``workflow_node_jobs`` and need the migration-0006 lease columns; ingest queues
# draw from ``ingest_jobs`` and need migration 0008 (the ``args`` column + the
# relaxed queue CHECK that lets host-defined queue names exist — without it a
# custom-queue row can't be inserted). A claim worker WAITS for its version
# before polling — the orchestrator owns the migration run (``db.bootstrap``
# serializes concurrent bootstraps with an advisory lock), the workers wait for it.
_NODE_REQUIRED_VERSION = 6
_INGEST_REQUIRED_VERSION = 8
# The gpu lane needs 0028: its claim names ``context_length`` on EVERY claim (that is
# how a box that cannot state its window is kept off sized rows), so a gpu worker on an
# older schema would fail every claim instead of taking work. It waits for the
# orchestrator's migration run, exactly as the download lane waits for 0025.
_GPU_REQUIRED_VERSION = 29
# The download lane needs the 0025 CHECK drop: before it, a download row cannot
# even be INSERTed (the cpu/gpu CHECK rejects it), so a download worker on an
# older schema would poll a queue that can never hold work.
_DOWNLOAD_REQUIRED_VERSION = 25


def _host_label() -> str:
    return env_get(get_config().host_label_env, "").strip() or socket.gethostname()


def _host_priority() -> int:
    """The claim's cross-host tiebreaker (``config.host_priority_env``):
    high-priority hosts head, overflow hosts tail. See ``node_queue._host_dir``."""
    try:
        return int(env_get(get_config().host_priority_env, "0"))
    except (TypeError, ValueError):
        return 0


def predict_default_kind(
    warm_servers: Iterable[str] | None, host_label: str, queue: str,
) -> str:
    """The serving kind an UNDECLARED gpu row will actually land on — the claim's
    ``default_server`` prediction (the law: a row's stamp is never NULL).

    Mirrors ``llm_backends.factory.resolve_for_job``'s no-preference default so the
    kind the arbiter COUNTS is the kind dispatch will USE: a warm vLLM is the box's
    active server and wins the default (anything else feeds a second
    model copy); otherwise the box's configured default via the factory's TTL
    cache. The rule holds: ``warm_servers`` is state the caller already probed
    for the affinity sort, and the cache means no per-claim DB read. NEVER raises —
    a control-plane hiccup degrades to the engine default rather than parking the
    claim loop. A wrong prediction is not a lost job: the dispatch-time re-stamp corrects the
    row when dispatch lands elsewhere."""
    try:
        if worker_control.SERVER_TYPE_VLLM in set(warm_servers or ()):
            return worker_control.SERVER_TYPE_VLLM
        from queue_workflows.llm_backends import factory
        return (factory.cached_default_kind(host_label, queue)
                or worker_control.DEFAULT_LLM_SERVER_TYPE)
    except Exception:
        return worker_control.DEFAULT_LLM_SERVER_TYPE


def budget_for(job: dict) -> int:
    """Wall-clock budget (s) for a claimed job:

      * GPU job whose ``required_model`` is in the host's video set → 1800 s;
      * any other GPU job → 8100 s;
      * a download node-job (migration 0025) → 3600 s;
      * built-in fetch / load ingest jobs → their generous sweep budgets;
      * any other (host-defined) ingest queue → ``config.ingest_default_budget_s``;
      * an input node (``node_module`` starts ``__input__``) → 120 s;
      * any other CPU job → 2100 s.
    """
    queue = job.get("queue")
    if queue == "gpu":
        if (job.get("required_model") or "") in get_config().video_model_ids:
            return VIDEO_BUDGET_S
        return GPU_DEFAULT_BUDGET_S
    if queue == "download":
        return DOWNLOAD_BUDGET_S
    if queue == "fetch":
        return FETCH_BUDGET_S
    if queue == "load":
        return LOAD_BUDGET_S
    if queue not in _NODE_QUEUES:
        # host-defined ingest queue — configurable default budget.
        return get_config().ingest_default_budget_s
    if (job.get("node_module") or "").startswith("__input__"):
        return INPUT_BUDGET_S
    return CPU_BUDGET_S


# ── lease renewal ──────────────────────────────────────────────────────────


_LEASE_TABLES = frozenset({"workflow_node_jobs", "ingest_jobs"})


class LeaseRenewer:
    """Daemon thread that pushes ``lease_expires_at`` out every ``interval_s``
    while a job runs. Scoped to ``id AND claimed_by`` so a reclaim that handed
    the row to another worker is NOT clobbered.

    ``table`` selects which lease table to renew: ``workflow_node_jobs``
    (cpu/gpu) or ``ingest_jobs`` (fetch/load). The table name is validated
    against a fixed allowlist (never interpolated from caller data)."""

    def __init__(
        self, *, job_id: str, claimed_by: str,
        lease_s: int = LEASE_S, interval_s: float = LEASE_RENEW_INTERVAL_S,
        table: str = "workflow_node_jobs",
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"lease table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._claimed_by = claimed_by
        self._lease_s = int(lease_s)
        self._interval_s = float(interval_s)
        self._table = table
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"lease-renew-{self._job_id[:8]}",
        )
        self._thread.start()

    def _renew_once(self) -> bool:
        try:
            # Ingest leases go through the ingest_store seam; it runs the
            # identical ``ingest_jobs`` UPDATE this used to run inline. DAG node-job leases
            # (``workflow_node_jobs``) stay on the direct-PG path, unchanged.
            if self._table == "ingest_jobs":
                return ingest_store.renew_ingest_lease(
                    self._job_id, self._claimed_by, lease_s=self._lease_s,
                )
            with connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self._table} "
                    "SET lease_expires_at = now() + make_interval(secs => %s) "
                    "WHERE id = %s AND claimed_by = %s AND status = 'running'",
                    (self._lease_s, self._job_id, self._claimed_by),
                )
                return (cur.rowcount or 0) > 0
        except Exception:
            log.exception("[lease-renew] %s renew failed; will retry", self._job_id)
            return False

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._renew_once()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── watchdog ─────────────────────────────────────────────────────────────


def _clear_busy_ghost(host_label: str | None, queue: str | None) -> None:
    """Best-effort: clear THIS worker's ``current_model`` busy-ghost before a hard
    exit (Part C). ``os._exit`` skips ``_run_node``'s ``finally`` — so
    ``ModelCache.mark_idle`` and the heartbeat refresh never run, and the worker's
    last ``worker_heartbeats`` row keeps advertising a ``current_model`` even
    though the process is dead, inflating a host's "N/M GPU busy" gauge (e.g.
    "3/2 GPU busy" after a kill). Null it out + age ``last_seen`` past the gauge's
    30 s window so the dead worker drops out at once. Swallows everything — a hung
    DB write must NEVER block the hard-exit. No-op when the worker identity wasn't
    threaded through (e.g. a unit test constructing a watchdog with just a
    ``job_id``), or for a CPU/ingest worker that never advertised a model."""
    if not host_label or not queue:
        return
    try:
        node_queue.clear_worker_current_model(host_label, queue)
    except Exception:
        log.exception(
            "[watchdog] could not clear current_model busy-ghost for %s/%s "
            "before exit (gauge will age it out within ~30s)", host_label, queue,
        )


def _trip_event_type(label: str) -> str:
    """Map a watchdog's label tag to its node-event type (migration 0011)."""
    lbl = (label or "").lower()
    if "stall" in lbl:
        return "stall_trip"
    if "gpu" in lbl:
        return "gpu_health_trip"
    return "budget_trip"


def _emit_node_event(
    job_id: str,
    event_type: str,
    *,
    host_label: str | None = None,
    queue: str | None = None,
    error: str | None = None,
    detail: dict | None = None,
    row: dict | None = None,
) -> None:
    """Best-effort node event from a claim / watchdog site that holds a job_id
    (and usually the row already). Fetches the row for run_id / node_id /
    attempt / model when not supplied; NEVER raises (forensic only — the
    underlying node_queue.record_node_event swallows too). Skips ingest jobs and
    vanished rows (no DAG run_id ⇒ nothing to attach the event to)."""
    try:
        r = row if row is not None else node_queue.get_node_job(job_id)
        if not r or not r.get("run_id"):
            return
        node_queue.record_node_event(
            run_id=r["run_id"], node_id=r["node_id"], job_id=job_id,
            attempt=int(r.get("watchdog_retries") or 0), event_type=event_type,
            host_label=host_label, queue=queue, model=r.get("required_model"),
            error=error, detail=detail,
        )
    except Exception:
        log.exception(
            "[node-event] emit %s for %s failed (ignored)", event_type, job_id,
        )


def _fail_job_and_exit(
    *, job_id: str, table: str, error: str,
    on_exit: Callable[[int], None], exit_code: int,
    host_label: str | None = None, queue: str | None = None,
) -> None:
    """Mark a doomed job ``failed`` (+ the dispatch-event outbox row for DAG
    node-jobs) then call ``on_exit``. Shared by :class:`Watchdog` (budget) and
    :class:`StallWatchdog` (no-progress) so the outbox-atomicity contract — the
    terminal mark and the ``failed`` event in ONE txn — is written in exactly
    one place. A mark failure is swallowed + logged: the hard-exit must still
    happen so the lease can expire and a reclaim re-queue the row.

    Before exiting it also clears this worker's ``current_model`` busy-ghost
    (:func:`_clear_busy_ghost`) so the dead worker doesn't keep inflating the
    GPU-busy gauge (Part C)."""
    try:
        if table == "ingest_jobs":
            # Ingest job: no parent run, no dispatch-event outbox — just mark
            # the row failed so a reclaim/operator sees it. Routed through the
            # ingest_store seam, like every ingest call site.
            ingest_store.mark_ingest_failed(job_id, error=error)
        else:
            # DAG node-job: mark failed + write the dispatch-event row in ONE
            # txn so the run fails through to downstream nodes even though we're
            # about to hard-exit.
            with connection() as conn, conn.cursor() as cur:
                row = node_queue.mark_failed_in_txn(cur, job_id, error=error)
                if row is not None:
                    node_queue.enqueue_dispatch_event_in_txn(
                        cur, row["run_id"], row["node_id"], "failed",
                    )
            if row is not None:
                _emit_node_event(
                    job_id, "failed", host_label=host_label, queue=queue,
                    error=error, row=row,
                )
    except Exception:
        log.exception("[watchdog] %s could not mark failed before exit", job_id)
    _clear_busy_ghost(host_label, queue)
    on_exit(exit_code)


def _requeue_job_and_exit(
    *, job_id: str, table: str, error: str,
    on_exit: Callable[[int], None], exit_code: int,
    host_label: str | None = None, queue: str | None = None,
) -> None:
    """Re-queue a watchdog-tripped DAG node-job for a RETRY on a fresh worker,
    then call ``on_exit`` — the re-queue twin of :func:`_fail_job_and_exit`.

    Instead of failing the run, flip the row ``running`` → ``queued`` (clearing
    the lease, bumping priority to the front, incrementing ``watchdog_retries``)
    via :func:`node_queue.requeue_job_for_retry`. Writes NO dispatch event — the
    run stays ``running``; only this node re-runs. The ``node_job_ready`` NOTIFY
    fired by the status flip wakes an idle worker to re-claim it.

    Same exit/atomicity discipline as :func:`_fail_job_and_exit`: a re-queue
    failure is swallowed + logged, but the hard-exit STILL happens — leaving the
    row ``running`` with a lease that the orchestrator's ``reclaim_expired_leases``
    will eventually re-queue anyway, so the node is never silently stranded.

    SAFETY (no double-run): the re-queue sets ``status='queued'`` and clears
    ``claimed_by``, then this process hard-exits. The fresh re-claim is the same
    CAS-guarded ``queued → running`` UPDATE as any claim; and any OTHER worker
    that still thought it held this row self-exits the instant its
    :class:`JobStatusWatcher` sees ``claimed_by`` no longer equals it. So at most
    one worker ever runs the re-queued node, and node bodies are idempotent on
    their out_dir.

    Only meaningful for ``workflow_node_jobs`` — ``requeue_job_for_retry`` targets
    that table. (Ingest jobs never reach here: they take the fail path, which is
    correct — an ingest job has no DAG run to keep alive, and ``ingest_jobs`` has
    its own ``reclaim_expired_ingest_leases`` re-queue.)"""
    try:
        _requeued = node_queue.requeue_job_for_retry(job_id)
        if _requeued is not None:
            _emit_node_event(
                job_id, "requeued", host_label=host_label, queue=queue,
                error=error, row=_requeued,
                detail={"retry": int(_requeued.get("watchdog_retries") or 0),
                        "cap": _watchdog_max_retries()},
            )
    except Exception:
        log.exception(
            "[watchdog] %s could not re-queue before exit (lease reclaim will "
            "recover it)", job_id,
        )
    _clear_busy_ghost(host_label, queue)
    on_exit(exit_code)


def _watchdog_trip(
    *, job_id: str, table: str, error: str, label: str,
    on_exit: Callable[[int], None], exit_code: int,
    host_label: str | None = None, queue: str | None = None,
) -> None:
    """Decide a watchdog trip's outcome: RE-QUEUE the node for a retry (under the
    cap) or FAIL the run (cap reached), then hard-exit. Shared by all three
    watchdogs so the policy lives in exactly one place.

    ``host_label`` / ``queue`` identify the tripping worker so both outcome paths
    can clear its ``current_model`` busy-ghost before the hard-exit (Part C).

    Policy:

      * DAG node-job (``workflow_node_jobs``): read the row's current
        ``watchdog_retries`` (migration 0010). If it's ``< QUEUE_WORKFLOWS_WATCHDOG_MAX_
        RETRIES`` (default 3) the trip is treated as a TRANSIENT wedge —
        :func:`_requeue_job_and_exit` re-queues the node + retries it on a fresh
        worker, and the RUN stays alive. Once the counter has reached the cap the
        node is wedging persistently, so we fall back to :func:`_fail_job_and_exit`
        (mark failed + ``failed`` dispatch event → the run fails) rather than loop
        forever.
      * Ingest job (``ingest_jobs``): always :func:`_fail_job_and_exit` — there's
        no DAG run to keep alive and no ``watchdog_retries`` column; the row is
        marked failed and the ingest lease-reclaim handles re-queue separately.

    Logs the chosen path + the attempt count clearly so an operator can see, per
    trip, whether the node was retried (and which attempt) or finally failed.
    ``label`` is the tripping watchdog's tag (e.g. ``stall-watchdog``) for the log
    line."""
    if table == "ingest_jobs":
        log.error(
            "[%s] %s tripped on an ingest job → marking failed (ingest jobs "
            "have no run to keep alive; lease-reclaim re-queues separately)",
            label, job_id,
        )
        _fail_job_and_exit(
            job_id=job_id, table=table, error=error,
            on_exit=on_exit, exit_code=exit_code,
            host_label=host_label, queue=queue,
        )
        return

    cap = _watchdog_max_retries()
    # Read the current re-queue count to decide retry vs fail. A read failure
    # (or a vanished row) is treated as 0 prior retries so we PREFER the
    # non-destructive re-queue — never fail a run just because the count read
    # blipped.
    try:
        row = node_queue.get_node_job(job_id)
        retries = int((row or {}).get("watchdog_retries") or 0)
    except Exception:
        log.exception(
            "[%s] %s could not read watchdog_retries; assuming 0 (re-queue)",
            label, job_id,
        )
        row = None
        retries = 0

    # Forensic: record the trip itself (stall / gpu-health / budget) regardless
    # of the requeue-vs-fail outcome below — the matching requeued / failed
    # event is emitted by the chosen exit path.
    _emit_node_event(
        job_id, _trip_event_type(label), host_label=host_label, queue=queue,
        error=error, row=row,
    )

    if retries < cap:
        attempt = retries + 1
        log.warning(
            "[%s] %s tripped (%s) — RE-QUEUEING for retry %d/%d on a fresh "
            "worker; the run stays alive",
            label, job_id, error, attempt, cap,
        )
        _requeue_job_and_exit(
            job_id=job_id, table=table, error=error,
            on_exit=on_exit, exit_code=exit_code,
            host_label=host_label, queue=queue,
        )
    else:
        log.error(
            "[%s] %s tripped (%s) — watchdog retry cap reached (%d/%d); "
            "FAILING the run",
            label, job_id, error, retries, cap,
        )
        _fail_job_and_exit(
            job_id=job_id, table=table, error=error,
            on_exit=on_exit, exit_code=exit_code,
            host_label=host_label, queue=queue,
        )


class Watchdog:
    """Daemon thread enforcing a wall-clock budget on a single running job. On
    trip: route through :func:`_watchdog_trip` — for a DAG node-job, RE-QUEUE it
    for a retry on a fresh worker under the cap (``QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES``,
    default 3), then mark failed once the cap is reached; for an ingest job, mark
    failed — then call ``on_exit`` (default: hard ``os._exit``). A hard exit kills
    exactly the over-budget job.

    Applied to CPU + ingest jobs only — GPU jobs are policed by health
    (:class:`GpuHealthWatchdog`), not by a wall-clock cap, so a long-but-healthy
    render is never killed for elapsed time."""

    def __init__(
        self, *, job_id: str, budget_s: float,
        on_exit: Callable[[int], None] | None = None,
        poll_s: float = 1.0, exit_code: int = 75,
        table: str = "workflow_node_jobs",
        host_label: str | None = None, queue: str | None = None,
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"watchdog table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._budget_s = float(budget_s)
        self._on_exit = on_exit or os._exit
        self._poll_s = float(poll_s)
        self._exit_code = int(exit_code)
        self._table = table
        self._host_label = host_label
        self._queue = queue
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._deadline = time.monotonic() + self._budget_s
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"watchdog-{self._job_id[:8]}",
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if time.monotonic() >= self._deadline:
                self._trip()
                return
            # Wake on stop OR at the next poll boundary, whichever first.
            self._stop.wait(self._poll_s)

    def _trip(self) -> None:
        err = (
            f"exceeded wall-clock budget ({int(self._budget_s)}s) — "
            f"watchdog hard-stopped the worker"
        )
        # Re-queue-with-cap (DAG node-jobs) for uniformity with the no-progress /
        # health watchdogs: a CPU over-budget MAY be a genuine runaway, but the
        # cap bounds it either way — at most QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES retries
        # then the run fails. Ingest over-budget still marks failed (no run to
        # keep alive) — _watchdog_trip branches on the table.
        _watchdog_trip(
            job_id=self._job_id, table=self._table, error=err, label="watchdog",
            on_exit=self._on_exit, exit_code=self._exit_code,
            host_label=self._host_label, queue=self._queue,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── stall (no-progress) watchdog ───────────────────────────────────────────


#: How many GPU/RAM samples the StallWatchdog takes to CONFIRM a worker is truly
#: wedged before it trips on a no-beat timeout (Part A — gate the trip on the
#: physical signal). A few samples spaced ``STALL_CONFIRM_POLL_S`` apart so a
#: single noisy reading can't flip the verdict either way.
STALL_CONFIRM_SAMPLES = _env_int("QUEUE_WORKFLOWS_STALL_CONFIRM_SAMPLES", 3)
STALL_CONFIRM_POLL_S = _env_float("QUEUE_WORKFLOWS_STALL_CONFIRM_POLL_S", 1.0)


class StallWatchdog:
    """Daemon thread enforcing a NO-PROGRESS deadline on one running job, GATED on
    the physical GPU/RAM signal so a LOADING / PREPARING / slow-but-working node
    is never killed.

    Where :class:`Watchdog` is a fixed wall-clock budget (catches a job running
    too *long*), this catches a job making *no progress*: it arms a short
    deadline that every :meth:`beat` pushes out. A GPU node beats once per
    diffusion step (threaded in as ``status_callback``); the executor also beats
    once when the model finishes loading, so the cold-load phase gets its own
    fresh window.

    **The no-beat timeout is necessary but NOT sufficient to trip** (the fix for
    the user's false-positive: "should not kill if the GPU model is being loaded
    or preparing to start; only if it does nothing"). When ``stall_timeout_s``
    elapses with no beat, the watchdog does NOT trip outright — beat-absence alone
    can't tell a wedged node from one in a legitimately slow step, mid-load, or
    preparing. Instead it CONFIRMS with the SAME ``gpu_health`` samplers + the SAME
    "GPU idle AND RAM static" predicate :class:`GpuHealthWatchdog` uses, over a
    short confirmation window (``confirm_samples`` reads, ``confirm_poll_s``
    apart):

      * GPU busy at ANY sample (``max sm% > idle_pct``) ⇒ the node IS doing GPU
        work (a slow step) ⇒ NOT wedged — reset the deadline, log, don't trip.
      * RAM moving (``|Δ| > ram_delta_mb`` across the window) ⇒ the node is
        loading weights / staging / preparing ⇒ NOT wedged — reset, log, don't
        trip. (A multi-GB model load moves RAM far beyond the 5 GB delta, so a
        cold/lazy load can NEVER false-trip.)
      * ONLY when GPU stayed idle AND RAM stayed static across the whole
        confirmation window — genuinely "doing nothing" — does it trip.

    On a confirmed trip :func:`_watchdog_trip` RE-QUEUES the node for a retry on a
    fresh worker (run stays alive) under the cap (``QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES``,
    default 3), then marks it failed once the cap is reached, hard-exiting in
    either case. ``beat`` is tolerant of extra args so it can be wired straight in
    as a node ``status_callback``.

    GPU/RAM samplers are injected (``gpu_sampler`` / ``ram_sampler``), defaulting
    to :mod:`queue_workflows.gpu_health` — the exact same samplers
    :class:`GpuHealthWatchdog` uses, so the two watchdogs' "is this worker idle?"
    verdict is consistent. Tests feed fakes + a tight confirmation window.
    ``host_label`` / ``queue`` are threaded so the trip path can clear this
    worker's ``current_model`` busy-ghost on hard-exit (Part C).

    ``confirm_idle=False`` is the POOL lane's mode: the beat gap alone trips and the
    samplers are never read. A pool job's GPU work happens in the ollama / vLLM
    container, so this container reads idle + static whether the generation is
    streaming or wedged — the confirmation would always say "wedged" and adds nothing
    but a delay. There, the node's beat (one per streamed chunk) IS the progress
    signal."""

    def __init__(
        self, *, job_id: str, stall_timeout_s: float = STALL_TIMEOUT_S,
        on_exit: Callable[[int], None] | None = None,
        poll_s: float = STALL_POLL_S, exit_code: int = 76,
        table: str = "workflow_node_jobs",
        idle_pct: int = GPU_IDLE_PCT,
        ram_delta_mb: int = GPU_HEALTH_RAM_DELTA_MB,
        confirm_samples: int = STALL_CONFIRM_SAMPLES,
        confirm_poll_s: float = STALL_CONFIRM_POLL_S,
        gpu_sampler: Callable[[], int] | None = None,
        ram_sampler: Callable[[], int | None] | None = None,
        host_label: str | None = None,
        queue: str | None = None,
        confirm_idle: bool = True,
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"stall-watchdog table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._stall_timeout_s = float(stall_timeout_s)
        self._on_exit = on_exit or os._exit
        self._poll_s = float(poll_s)
        self._exit_code = int(exit_code)
        self._table = table
        self._idle_pct = int(idle_pct)
        self._ram_delta_mb = int(ram_delta_mb)
        self._confirm_samples = max(1, int(confirm_samples))
        self._confirm_poll_s = float(confirm_poll_s)
        self._confirm_idle = bool(confirm_idle)
        if self._confirm_idle and (gpu_sampler is None or ram_sampler is None):
            from queue_workflows import gpu_health
            gpu_sampler = gpu_sampler or gpu_health.gpu_util_pct
            ram_sampler = ram_sampler or gpu_health.container_ram_mb
        self._gpu_sampler = gpu_sampler
        self._ram_sampler = ram_sampler
        self._host_label = host_label
        self._queue = queue
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # None ⇒ not yet armed: the watchdog stays inert until the first beat
        # (the executor's after-model-load beat), so a long cold load is never
        # policed by the no-progress deadline.
        self._deadline: float | None = None

    def beat(self, *args: Any, **kwargs: Any) -> None:
        """Record progress: arm (on the first call) / push the no-progress
        deadline out by one window. Thread-safe; ignores any args so it doubles
        as a node ``status_callback``."""
        with self._lock:
            self._deadline = time.monotonic() + self._stall_timeout_s

    def _rearm(self) -> None:
        """Open a fresh no-progress window (used after a no-beat timeout that the
        physical-signal confirmation cleared as 'busy/loading, not wedged')."""
        with self._lock:
            self._deadline = time.monotonic() + self._stall_timeout_s

    def start(self) -> None:
        # Note: NO initial beat — the watchdog arms on the first beat, not at
        # start, so the model-load phase before that beat is never policed here.
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"stall-watchdog-{self._job_id[:8]}",
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                deadline = self._deadline
            if deadline is not None and time.monotonic() >= deadline:
                # No beat for the window — but that ALONE doesn't mean wedged.
                # Confirm against the physical GPU/RAM signal before tripping, unless
                # this container has no physical signal to offer (the pool lane).
                if not self._confirm_idle or self._confirm_wedged():
                    self._trip()
                    return
                # Busy / loading / preparing — a false alarm. Re-arm the window
                # and keep watching; do NOT kill a working node.
                self._rearm()
            # Wake on stop OR at the next poll boundary, whichever first.
            self._stop.wait(self._poll_s)

    def _sample_gpu(self) -> int:
        try:
            return int(self._gpu_sampler() or 0)
        except Exception:
            log.exception("[stall-watchdog] %s gpu sample failed; treating as 0", self._job_id)
            return 0

    def _sample_ram(self) -> int | None:
        try:
            v = self._ram_sampler()
            return None if v is None else int(v)
        except Exception:
            log.exception("[stall-watchdog] %s ram sample failed; treating as None", self._job_id)
            return None

    def _confirm_wedged(self) -> bool:
        """The no-beat timeout fired — is the worker TRULY doing nothing, or just
        loading / preparing / in a slow step? Sample the physical signal over a
        short window and apply the SAME "GPU idle AND RAM static" rule the
        :class:`GpuHealthWatchdog` uses.

        Returns ``True`` (⇒ trip) ONLY when, across the whole confirmation window,
        the MAX GPU util stayed ``<= idle_pct`` AND RAM never moved more than
        ``ram_delta_mb`` from its first reading. A busy GPU at any sample (slow
        step) or a RAM move beyond the delta (weights loading / staging /
        preparing) ⇒ returns ``False`` (NOT wedged) and the caller re-arms.

        A genuinely-loading model moves RAM by GBs >> the delta, so a cold/lazy
        load can never be confirmed as wedged here — exactly the false positive
        the user reported. Interruptible: bails to ``False`` if stopped mid-window
        (the node finished / was reclaimed)."""
        ram_anchor = self._sample_ram()
        max_util = self._sample_gpu()
        ram_now = ram_anchor
        for i in range(self._confirm_samples - 1):
            if self._stop.wait(self._confirm_poll_s):
                return False  # stopped → don't trip
            util = self._sample_gpu()
            if util > max_util:
                max_util = util
            ram_now = self._sample_ram()
        gpu_idle = max_util <= self._idle_pct
        ram_moved = (
            ram_anchor is not None
            and ram_now is not None
            and abs(ram_now - ram_anchor) > self._ram_delta_mb
        )
        if gpu_idle and not ram_moved:
            log.warning(
                "[stall-watchdog] %s no beat for %ds AND confirmed idle "
                "(max sm%% %d <= %d, RAM anchor=%sMB now=%sMB static) — WEDGED, "
                "tripping",
                self._job_id, int(self._stall_timeout_s), max_util,
                self._idle_pct, ram_anchor, ram_now,
            )
            return True
        log.info(
            "[stall-watchdog] %s no beat for %ds BUT GPU/RAM active "
            "(max sm%% %d, RAM anchor=%sMB now=%sMB) — loading/preparing/slow "
            "step, NOT killing; resetting the window",
            self._job_id, int(self._stall_timeout_s), max_util, ram_anchor, ram_now,
        )
        # Highest-value invisible signal today: the stall was SUSPECTED but the
        # worker is healthy (loading / slow step), so we did NOT kill it. Record
        # it so the "why did this node look stuck but wasn't" story is in-app.
        _emit_node_event(
            self._job_id, "stall_suspected",
            host_label=self._host_label, queue=self._queue,
            error=(f"no beat for {int(self._stall_timeout_s)}s but GPU/RAM "
                   f"active — not killing"),
            detail={"max_sm_pct": max_util, "ram_anchor_mb": ram_anchor,
                    "ram_now_mb": ram_now},
        )
        return False

    def _trip(self) -> None:
        if self._confirm_idle:
            err = (
                f"no progress for {int(self._stall_timeout_s)}s (no GPU step beat) "
                f"AND confirmed GPU-idle + static-RAM — stall watchdog hard-stopped "
                f"a wedged worker"
            )
        else:
            err = (
                f"no progress beat for {int(self._stall_timeout_s)}s from a pool-lane "
                f"node (its declared POOL_STALL_TIMEOUT_S) — stall watchdog "
                f"hard-stopped the worker so the stuck request actually ends"
            )
        # Transient-wedge detector (a long multi-step node that wedges once):
        # re-queue + retry on a fresh worker under the cap, then fail.
        _watchdog_trip(
            job_id=self._job_id, table=self._table, error=err,
            label="stall-watchdog",
            on_exit=self._on_exit, exit_code=self._exit_code,
            host_label=self._host_label, queue=self._queue,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── GPU health watchdog (HEALTH-driven, replaces the wall-clock cap for GPU) ─


class GpuHealthWatchdog:
    """Daemon thread that hard-stops a GPU job ONLY when it's truly wedged —
    health-driven, NOT wall-clock.

    Where :class:`Watchdog` kills a GPU job purely because its time budget
    elapsed (health-blind), this never kills for elapsed time. Every
    ``interval_s`` (default 300 s) it evaluates the window:

      * the MAX per-container GPU utilization seen since the last checkpoint
        (sampled every ``poll_s`` s and remembered), and
      * the change in this container's RAM since the last checkpoint.

    It TRIPS iff over the whole window the GPU stayed idle
    (``max_util <= idle_pct``) AND RAM was static (``|Δram| <= ram_delta_mb``)
    — i.e. no GPU work AND no memory movement ⇒ wedged. If the GPU was busy at
    any point, OR RAM moved more than the delta (staging / decode / model
    swap), the job is HEALTHY: the window resets and it keeps running, no matter
    how long. On trip it routes through :func:`_watchdog_trip` (the same policy as
    the other watchdogs): RE-QUEUE the node for a retry on a fresh worker under
    the cap (``QUEUE_WORKFLOWS_WATCHDOG_MAX_RETRIES``, default 3) so the run stays alive,
    then mark failed once the cap is reached — hard-exiting in either case.

    Arming: the watchdog arms AT job start (:meth:`start`) with a generous
    ``load_grace_s`` (20 min) FIRST window — NOT inert-until-first-beat. That is
    safe precisely because the trip rule is "GPU idle AND RAM static": a healthy
    model load MOVES RAM (weights, multiple GB) so it can't trip even while armed
    during load, and a healthy render keeps the GPU busy so it can't trip
    regardless. Only a GENUINELY hung load (idle GPU AND static RAM past the
    grace) trips. This closes the gap the removed wall-clock cap left: a hang
    DURING load (before any beat), and a GPU node with no ``required_model`` /
    no ``status_callback`` that never beats at all, are now both bounded. The
    first :meth:`beat` (the executor's post-model-load beat, then per-step node
    progress beats) collapses the window to the normal ``interval_s`` (5-min)
    cadence. A node progress ``beat`` also resets the window as an extra liveness
    signal. ``beat`` tolerates any args so it doubles as a node
    ``status_callback``.

    GPU/RAM samplers are injected (``gpu_sampler`` / ``ram_sampler``) so tests
    feed fakes instead of shelling out to ``nvidia-smi``; production defaults to
    :mod:`queue_workflows.gpu_health` (per-container pmon ``sm%`` + cgroup RAM).
    """

    def __init__(
        self, *, job_id: str,
        interval_s: float = GPU_HEALTH_INTERVAL_S,
        load_grace_s: float = GPU_HEALTH_LOAD_GRACE_S,
        idle_pct: int = GPU_IDLE_PCT,
        ram_delta_mb: int = GPU_HEALTH_RAM_DELTA_MB,
        poll_s: float = STALL_POLL_S,
        gpu_sampler: Callable[[], int] | None = None,
        ram_sampler: Callable[[], int | None] | None = None,
        on_exit: Callable[[int], None] | None = None,
        exit_code: int = 78,
        table: str = "workflow_node_jobs",
        host_label: str | None = None, queue: str | None = None,
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"gpu-health table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._interval_s = float(interval_s)
        self._load_grace_s = float(load_grace_s)
        self._idle_pct = int(idle_pct)
        self._ram_delta_mb = int(ram_delta_mb)
        self._poll_s = float(poll_s)
        if gpu_sampler is None or ram_sampler is None:
            from queue_workflows import gpu_health
            gpu_sampler = gpu_sampler or gpu_health.gpu_util_pct
            ram_sampler = ram_sampler or gpu_health.container_ram_mb
        self._gpu_sampler = gpu_sampler
        self._ram_sampler = ram_sampler
        self._on_exit = on_exit or os._exit
        self._exit_code = int(exit_code)
        self._table = table
        self._host_label = host_label
        self._queue = queue
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Window state. ``_armed`` is set True in ``start()`` with a generous
        # ``load_grace_s`` first checkpoint (NOT inert-until-beat) so a load-phase
        # hang and a never-beating node are both bounded; the first beat collapses
        # the window to ``interval_s``. ``_max_util`` accumulates the peak GPU util
        # seen this window; ``_ram_anchor`` is the RAM at the window's start that
        # the |Δ| is measured against.
        self._armed = False
        self._max_util = 0
        self._ram_anchor: int | None = None
        self._next_checkpoint = 0.0

    # ── window management ────────────────────────────────────────────────

    def _reset_window(self, window_s: float | None = None) -> None:
        """Open a fresh window: clear the peak-util accumulator, re-anchor RAM,
        and push the next checkpoint out by ``window_s`` (default: the normal
        ``interval_s`` cadence; ``start()`` passes the generous ``load_grace_s``
        for the first window). Caller holds the lock."""
        self._max_util = 0
        self._ram_anchor = self._sample_ram()
        self._next_checkpoint = time.monotonic() + (
            self._interval_s if window_s is None else window_s
        )

    def beat(self, *args: Any, **kwargs: Any) -> None:
        """Record progress: arm (idempotent — already armed at start) + reset the
        window to the NORMAL ``interval_s`` cadence. Wired in as the node
        ``status_callback`` (extra liveness) and pulsed once by the executor right
        after the model load — that post-load beat collapses the generous
        load-grace first window opened by ``start()`` down to ``interval_s``.
        Thread-safe; ignores any args so it doubles as a ``status_callback``."""
        with self._lock:
            self._armed = True
            self._reset_window()

    def _sample_gpu(self) -> int:
        try:
            return int(self._gpu_sampler() or 0)
        except Exception:
            log.exception("[gpu-health] %s gpu sample failed; treating as 0", self._job_id)
            return 0

    def _sample_ram(self) -> int | None:
        try:
            v = self._ram_sampler()
            return None if v is None else int(v)
        except Exception:
            log.exception("[gpu-health] %s ram sample failed; treating as None", self._job_id)
            return None

    def start(self) -> None:
        # Arm AT start with a generous load-grace first window (NOT inert-until-
        # beat). Safe because the trip rule is "GPU idle AND RAM static": a
        # healthy model load moves RAM and a healthy render keeps the GPU busy, so
        # neither trips even while armed during load — only a genuinely hung load
        # (idle GPU AND static RAM past load_grace_s) does. This bounds both a
        # load-phase hang and a GPU node that never beats. The first beat (post-
        # load) collapses the window to the normal interval_s cadence.
        with self._lock:
            self._armed = True
            self._reset_window(window_s=self._load_grace_s)
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"gpu-health-{self._job_id[:8]}",
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                armed = self._armed
                if armed:
                    util = self._sample_gpu()
                    if util > self._max_util:
                        self._max_util = util
                    due = time.monotonic() >= self._next_checkpoint
                    if due and self._evaluate_window_locked():
                        # tripped → _trip() already exited; bail just in case the
                        # injected on_exit didn't terminate (tests).
                        return
            self._stop.wait(self._poll_s)

    def _evaluate_window_locked(self) -> bool:
        """At a checkpoint: TRIP iff GPU idle AND RAM static over the window.
        Otherwise reset the window (healthy) and continue. Returns True iff it
        tripped. Caller holds the lock."""
        ram_now = self._sample_ram()
        gpu_idle = self._max_util <= self._idle_pct
        # RAM is "moved" (⇒ healthy) only when we have BOTH anchor and current
        # readings and the |Δ| exceeds the threshold. A missing reading is
        # treated as "no RAM movement detected" so a flaky RAM probe can't, on
        # its own, keep a wedged + GPU-idle worker alive forever — but note GPU
        # idle is still required to trip, so a busy GPU always survives.
        ram_moved = (
            self._ram_anchor is not None
            and ram_now is not None
            and abs(ram_now - self._ram_anchor) > self._ram_delta_mb
        )
        if gpu_idle and not ram_moved:
            self._trip(max_util=self._max_util, ram_anchor=self._ram_anchor, ram_now=ram_now)
            return True
        # Healthy this window — re-anchor and re-arm the next checkpoint.
        self._reset_window()
        return False

    def _trip(self, *, max_util: int, ram_anchor: int | None, ram_now: int | None) -> None:
        err = (
            f"no GPU activity (max sm% {max_util} <= {self._idle_pct}) and static "
            f"RAM (anchor={ram_anchor}MB now={ram_now}MB, |Δ| <= {self._ram_delta_mb}MB) "
            f"for {int(self._interval_s)}s — health watchdog hard-stopped a wedged worker"
        )
        # Transient-wedge detector (a wedged GPU render): re-queue + retry on a
        # fresh worker under the cap, then fail.
        _watchdog_trip(
            job_id=self._job_id, table=self._table, error=err,
            label="gpu-health",
            on_exit=self._on_exit, exit_code=self._exit_code,
            host_label=self._host_label, queue=self._queue,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── job-status watcher (abandon a job taken from us) ───────────────────────


class JobStatusWatcher:
    """Daemon thread that HARD-EXITS the worker the instant its claimed job is
    no longer ``claimed_by`` this host — i.e. another actor re-queued it
    (orchestrator restart-resume / lease reclaim) or reassigned it to a peer.
    Polls ``workflow_node_jobs`` every ``poll_s``; on a miss it calls
    ``on_exit`` (default ``os._exit``). systemd then restarts the worker and it
    claims fresh work.

    This is what makes re-queuing a *running* job SAFE: the displaced worker
    kills itself instead of racing the new claimant to a double-run. ``os._exit``
    is the only way to abandon a node body wedged deep in a CUDA kernel.

    Scoped to ``claimed_by`` (not bare ``status``) on purpose: the worker's OWN
    terminal mark — ``mark_completed`` / ``mark_failed`` set ``status`` but leave
    ``claimed_by`` intact — must NOT trip it; only an external hand-off (which
    clears or changes ``claimed_by``) does."""

    def __init__(
        self, *, job_id: str, claimed_by: str,
        on_exit: Callable[[int], None] | None = None,
        poll_s: float = 2.0, exit_code: int = 77,
    ) -> None:
        self._job_id = job_id
        self._claimed_by = claimed_by
        self._on_exit = on_exit or os._exit
        self._poll_s = float(poll_s)
        self._exit_code = int(exit_code)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"job-status-{self._job_id[:8]}",
        )
        self._thread.start()

    def _loop(self) -> None:
        # Wait first, then check — gives the claim a moment to settle and wakes
        # instantly on stop().
        while not self._stop.wait(self._poll_s):
            try:
                row = node_queue.get_node_job(self._job_id)
            except Exception:
                log.exception("[job-status-watcher] %s poll failed; retrying",
                              self._job_id)
                continue
            if row is None or row.get("claimed_by") != self._claimed_by:
                log.warning(
                    "[job-status-watcher] %s no longer ours "
                    "(status=%s claimed_by=%s) — hard-exiting so a fresh worker "
                    "resumes it",
                    self._job_id,
                    None if row is None else row.get("status"),
                    None if row is None else row.get("claimed_by"),
                )
                if row is not None:
                    _emit_node_event(
                        self._job_id, "reassigned",
                        host_label=self._claimed_by, row=row,
                        detail={"new_claimed_by": row.get("claimed_by"),
                                "status": row.get("status")},
                    )
                self._on_exit(self._exit_code)
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── worker capacity heartbeat ────────────────────────────────────────────────


class HeartbeatEmitter:
    """Daemon thread that keeps this worker's ``worker_heartbeats`` row fresh so
    a host's queue gauge sees the fleet's real capacity.

    Claim-worker shape:

      * ``concurrency`` is **1** for cpu/ingest — a claim worker is one
        poll-claim loop. For **GPU** it is ``max(1, PAR)``: the machine's GPU job
        concurrency, i.e. its VLM pool size (``worker_controls.llm_parallelism``),
        read live each tick via the ``concurrency_fn`` seam so a UI PAR change is
        reflected without a restart. So a PAR-4 vLLM machine reads GPU x/4 in the
        consumer pill; an ollama machine (PAR 1) reads x/1. (NOT 1+PAR — the
        inline diffusion job occupies one of those PAR slots, not an extra one.)
      * the **GPU** heartbeat reports ``current_model`` read from the worker's
        :class:`ModelCache` on every tick — the gauge's busy signal. The
        **CPU** heartbeat reports ``current_model=None``.
      * ingest workers (fetch/load + any host-defined queue) also heartbeat —
        migration 0008 dropped the cpu/gpu-only CHECK — reporting
        ``current_model=None`` so a host's queue gauge shows their liveness.
      * honours ``QUEUE_WORKFLOWS_DISABLE_WORKER_HEARTBEAT`` (tests).

    Upserts once on :meth:`start` (so the row exists immediately) then refreshes
    every ``interval_s`` until :meth:`stop`."""

    def __init__(
        self, *, queue: str, host_label: str, model_cache: Any = None,
        interval_s: float = HEARTBEAT_INTERVAL_S,
        concurrency_fn: Callable[[], int] | None = None,
        llm_servers_fn: Callable[[], list[str]] | None = None,
        on_tick: Callable[[], None] | None = None,
        occupant_fn: Callable[[], str | None] | None = None,
        known_models_fn: Callable[[list[str]], list[str]] | None = None,
        llm_context_fn: Callable[[], tuple[int | None, int | None]] | None = None,
        residents_fn: Callable[[], set[str] | None] | None = None,
    ) -> None:
        # Narrows the DECLARED registry to what this box was OBSERVED to hold. The
        # gpu worker supplies one; default is identity, so a standalone emitter (and
        # every cpu/ingest worker, which observes no models) behaves exactly as before.
        self._known_models_fn = known_models_fn or (lambda declared: list(declared))
        # What this box's LLM SERVERS were observed to hold this tick, or None for "not
        # observed". Only the gpu worker supplies one; unset means current_model behaves
        # exactly as it always did.
        self._residents_fn = residents_fn
        # Extra per-tick hook (gpu worker: renew/release the box-wide model lease
        # by observed residency). Best-effort; a failure never stops the heartbeat.
        self._on_tick = on_tick
        self._queue = queue
        self._host_label = host_label
        self._model_cache = model_cache
        self._interval_s = float(interval_s)
        # Live concurrency provider, read each tick. GPU supplies one returning
        # max(1, PAR) (its VLM pool size = the machine's GPU job concurrency);
        # cpu/ingest pass None ⇒ the historical constant 1.
        self._concurrency_fn = concurrency_fn
        # OBSERVED llm capability provider, read each tick. The GPU worker wires
        # this to a live endpoint probe (queue_workflows.llm_probe) so a box
        # advertises the server it ACTUALLY answers on — [] when none does —
        # instead of the static config guess that defaulted to ["ollama"] even on
        # a box with no server (the incident this seam fixes). None ⇒ the historical
        # behaviour: publish config.llm_servers_available unchanged, so cpu/ingest
        # workers and other consumers are byte-compatible.
        self._llm_servers_fn = llm_servers_fn
        # What holds the CARD when no in-process model does — the box lease's
        # occupant (gpu worker wires this to its held slot). Without it a
        # self-managing render heartbeats current_model=NULL and the panel shows
        # an idle box doing minutes of GPU work; with the old fallback it showed
        # the LLM's model instead — reporting ollama as loaded while a render runs.
        self._occupant_fn = occupant_fn
        # The CONTEXT WINDOW pair (migration 0028): what this box's LLM server serves
        # now, and the biggest it could switch to within its free memory. The GPU
        # worker wires it; every other lane publishes (None, None) — a cpu box has no
        # window, and a sized job must never be routed to one.
        self._llm_context_fn = llm_context_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Cached total-VRAM probe (stable hardware property; sampled once).
        self._vram_probed = False
        self._vram_total: int | None = None

    @property
    def _enabled(self) -> bool:
        """Heartbeat runs for every queue family — cpu/gpu node workers AND the
        ingest workers (migration 0008 dropped the cpu/gpu-only CHECK) — unless
        the test opt-out env is set."""
        return not bool(env_get("QUEUE_WORKFLOWS_DISABLE_WORKER_HEARTBEAT"))

    def _current_model(self) -> str | None:
        """The MODEL on this box's card: the warm-model slot, read live each tick, or
        the box lease's occupant when that occupant IS a model. NULL for CPU, for a
        genuinely idle GPU, and for every SENTINEL occupant.

        A field called ``current_model`` carries a model id or nothing. The LLM slot was
        excluded here from the start — "the server's OWN model reporting already covers
        it, and 'LLM server' beside 'chat-model-a:latest' would double-report one
        occupant" — and that reason holds for every sentinel: a ComfyUI render and a
        self-managing node are OCCUPANCY, not models. Publishing one made an operator
        panel show "MODEL GPU job: render_report" beside the real model and warn
        of two residents contending for VRAM; what the box is DOING is already on the
        card as its running jobs.
        """
        if self._queue != "gpu":
            return None
        model = getattr(self._model_cache, "current_model", None) \
            if self._model_cache is not None else None
        if model and self._box_still_holds(model):
            return model
        occupant = self._occupant_fn() if self._occupant_fn is not None else None
        from queue_workflows.gpu_model_lease import is_sentinel_slot

        if occupant and not is_sentinel_slot(occupant) \
                and self._box_still_holds(str(occupant)):
            return str(occupant)
        return None

    def _box_still_holds(self, model: str) -> bool:
        """Is the warm slot still RESIDENCY, or did it outlive the model?

        For a model the worker loads itself (``ModelSpec.served_by`` empty) the cache IS
        the observation — it holds the weights in this process, and no server can vouch
        for it. For a model served by ollama or vLLM the cache is only a claim: the five
        minute keep-alive, or another lane loading something else, evicts the model while
        the slot stands until the cache's own idle TTL. A box that kept advertising it
        made an operator panel warn of two models contending for one GPU.

        Ignorance is never evidence of absence: no seam wired, a registry or probe that
        raises, or an observation that has not happened yet all leave the old answer."""
        if self._residents_fn is None:
            return True
        from queue_workflows import model_registry
        from queue_workflows.model_catalog import canonical_tag

        try:
            want = canonical_tag(str(model))
            served = {canonical_tag(str(i)) for i in model_registry.server_served_ids()}
            if want not in served:
                return True                      # in-process: the cache is the proof
            residents = self._residents_fn()
            if residents is None:
                return True                      # not observed — not "absent"
            # A role tag is SERVED BY an artifact (0027: `<role>-v<N>-<precision>-<server>`),
            # and a box can hold both spellings. Either vouches for the role.
            from queue_workflows.model_catalog import artifact_role

            return any(canonical_tag(str(r)) == want or artifact_role(str(r)) == want
                       for r in residents)
        except Exception:
            log.debug("[claim-worker:%s] residency check failed for %s — publishing it "
                      "as before", self._queue, model, exc_info=True)
            return True

    def _concurrency(self) -> int:
        """The advertised capacity for this tick — the historical constant 1
        unless a caller supplies a ``concurrency_fn`` (GPU does: ``max(1, PAR)``).
        A read failure falls back to 1 so a transient DB blip never zeroes the
        gauge nor crashes the heartbeat."""
        if self._concurrency_fn is None:
            return 1
        try:
            return max(1, int(self._concurrency_fn()))
        except Exception:
            log.exception(
                "[claim-worker:%s] concurrency_fn read failed; advertising 1",
                self._queue,
            )
            return 1

    def _vram_total_mb(self) -> int | None:
        """This machine's total GPU VRAM (MB), GPU queue only — the capacity the
        fit decision is measured against (migration 0015). Sampled once and
        cached: VRAM total is a stable hardware property, so we don't re-shell
        ``nvidia-smi`` / ``rocm-smi`` every 10 s tick. ``None`` for cpu/ingest or
        when no GPU is detected. Best-effort: a probe failure caches ``None`` for
        this process (re-probed only on a new worker)."""
        if self._queue != "gpu":
            return None
        if not self._vram_probed:
            try:
                from queue_workflows import hw_metrics
                self._vram_total = hw_metrics.total_vram_mb()
            except Exception:
                log.exception("[claim-worker:gpu] total VRAM probe failed; unknown")
                self._vram_total = None
            self._vram_probed = True
        return self._vram_total

    def _fits_models(self, known: list[str], vram_total_mb: int | None) -> list[str]:
        """The subset of ``known`` whose est_vram_gb fits this machine's VRAM —
        the worker's advertised ``fits_models`` (GPU queue only). Computed from
        the registry the worker holds (the orchestrator has none), so the fleet
        fit decision can be pure SQL over the heartbeat. ``[]`` for cpu/ingest."""
        if self._queue != "gpu":
            return []
        from queue_workflows import model_registry
        fits = model_registry.fits_within(vram_total_mb)
        # Intersect with the ids actually advertised as known this tick (defends
        # against a registry/known drift), preserving sorted order.
        known_set = set(known)
        return [m for m in fits if m in known_set]

    def emit_once(self) -> None:
        """Upsert this worker's row once (concurrency = 1 for cpu/ingest, or
        ``max(1, PAR)`` for GPU via ``concurrency_fn``; GPU also carries its live
        ``current_model``, the diffusion busy signal, plus its ``vram_total_mb``
        capacity + ``fits_models`` — the registered models its VRAM can hold —
        for capacity-aware assignment, migration 0015). Failures are swallowed +
        logged — a transient DB blip must never crash a worker mid-job."""
        try:
            from queue_workflows import model_registry
            known = model_registry.known_ids()
            vram_total = self._vram_total_mb()
            # PROBE BEFORE READING. The servers probe WRITES
            # the state the concurrency read derives from (_box_served feeds
            # _gpu_slot_budget), and kwargs evaluate left-to-right — so putting the
            # probe after the read published tick N-1's concurrency with tick N's
            # served list. That skew is what the panel showed as "NODE JOBS 3/1
            # while vLLM was serving". One row must describe one observation.
            servers = self._llm_servers()
            # ...and narrow AFTER it, for the same reason. The probe is what observes
            # this box's models, so narrowing before it would advertise tick N-1's
            # observation next to tick N's server list — the identical skew the comment
            # above records, one field over.
            known = self._known_models_fn(known)
            concurrency = self._concurrency()
            # ...and the window AFTER the servers probe too, for the same reason: the
            # probe is what establishes which kinds this box serves, and the window is
            # read per serving kind. One row must describe one observation.
            kwargs: dict[str, Any] = {}
            if self._llm_context_fn is not None:
                # NAMED ONLY BY A LANE THAT HAS A WINDOW. A cpu/ingest worker of this
                # engine can start against a database the orchestrator has not migrated
                # to 0028 yet, and a heartbeat naming columns that do not exist would
                # flag the whole lane dead over a field it never uses.
                kwargs["llm_context_length"], kwargs["llm_context_max"] = \
                    self._llm_context()
            node_queue.upsert_worker_heartbeat(
                host_label=self._host_label,
                queue=self._queue,
                concurrency=concurrency,
                current_model=self._current_model(),
                known_models=known,
                llm_servers_available=servers,
                vram_total_mb=vram_total,
                fits_models=self._fits_models(known, vram_total),
                **kwargs,
            )
        except Exception:
            log.exception("[claim-worker:%s] heartbeat upsert failed", self._queue)

    def _llm_context(self) -> tuple[int | None, int | None]:
        """``(serves_now, could_switch_to)`` in tokens, or ``(None, None)``.

        Unknown is a first-class answer and the SAFE one: the claim refuses a sized job
        to a box that cannot state its window (silent truncation is the alternative),
        and the fleet sweep tags a job no box can reach. So a probe that fails publishes
        unknown rather than a guess — and never breaks the heartbeat."""
        if self._llm_context_fn is None:
            return None, None
        try:
            serving, ceiling = self._llm_context_fn()
        except Exception:
            log.exception("[claim-worker:%s] llm context probe failed — publishing "
                          "unknown", self._queue)
            return None, None
        return serving, ceiling

    def _llm_servers(self) -> list[str]:
        """The OBSERVED llm capability to advertise this tick. Prefer the injected
        live probe (the GPU worker's endpoint check); fall back to the static
        ``config.llm_servers_available`` when no probe is wired (cpu/ingest workers,
        other consumers). A probe that raises must never break the heartbeat, so it
        degrades to the config value rather than propagating."""
        if self._llm_servers_fn is not None:
            try:
                return list(self._llm_servers_fn())
            except Exception:
                log.exception(
                    "[claim-worker:%s] llm probe failed — falling back to config",
                    self._queue,
                )
        return list(get_config().llm_servers_available)

    def _loop(self) -> None:
        # First tick already fired in start(); refresh until stopped.
        while not self._stop.wait(self._interval_s):
            self.emit_once()
            if self._on_tick is not None:
                try:
                    self._on_tick()
                except Exception:
                    log.exception("[heartbeat:%s] on_tick failed (ignored)", self._queue)

    def start(self) -> None:
        if not self._enabled:
            return
        self.emit_once()  # row exists immediately, before the first sleep
        if self._on_tick is not None:
            try:
                self._on_tick()
            except Exception:
                log.exception("[heartbeat:%s] on_tick failed (ignored)", self._queue)
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"worker-heartbeat-{self._queue}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── the claim loop ─────────────────────────────────────────────────────────


#: Queues whose claim worker draws DAG node-jobs from ``workflow_node_jobs``.
#: ``download`` (migration 0025) is the width-capped network-WAIT lane — node
#: family (lease/watchdog/outbox, ``node_job_ready`` wake), never ingest.
_NODE_QUEUES = frozenset({"cpu", "gpu", "download"})
#: Default ingest queues. The LIVE ingest set is
#: ``config.ingest_queues`` (host-configurable); any queue NOT in
#: ``_NODE_QUEUES`` is treated as ingest-family (draws from ``ingest_jobs``).
_DEFAULT_INGEST_QUEUES = frozenset({"fetch", "load"})


def _probe_gpu_usable() -> bool | None:
    """Tri-state CUDA-usability probe for the GPU-blind claim gate.

    * ``True``  — a CUDA device is usable right now.
    * ``False`` — torch imports but reports NO usable device (incl. a probe that
      raises, e.g. ``Can't initialize NVML``). THIS is the post-recreate
      blind condition: the worker image always ships torch, the HOST GPU is
      healthy, but the container's device cgroup was not injected.
    * ``None``  — torch is not importable at all. NOT the blind condition (a GPU
      worker image always has torch) — most likely a non-GPU / test environment.
      The caller must NOT refuse on ``None`` (that would false-park a box that
      merely lacks torch, and would break every torch-less GPU unit test).

    Best-effort: never raises. Module-level so a test can monkeypatch it."""
    try:
        import torch
    except Exception:
        return None
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


#: Opt OUT of reclaiming the card's LLM RAM when a non-LLM occupant takes the
#: box. For a box whose ollama is deliberately shared with something outside the
#: engine's accounting; everywhere else the reclaim is the point.
_DISABLE_CARD_RECLAIM_ENV = "QUEUE_WORKFLOWS_DISABLE_GPU_CARD_RECLAIM"


class _CpuLanePool:
    """N long-lived worker threads pulling from a plain queue — the cpu lane's
    executor.

    NOT a :class:`ThreadPoolExecutor`, and the difference is the whole point.
    The executor starts its worker threads LAZILY inside ``submit``, and it does
    so AFTER it has already put the work item on its queue::

        self._work_queue.put(w)      # the body will run
        self._adjust_thread_count()  # ← `RuntimeError: can't start new thread`

    So when the OS refuses a thread — precisely the failure a widened lane makes
    likelier, and the one the deploy docs' M×P×N composition invites — ``submit``
    raises *even though an existing worker will still run the body*. A claim path
    that infers "not submitted, give the slot back" from that exception frees a
    slot its job still occupies: the count under-reads, the ceiling is breached
    (N+1 ``running`` rows, the KILL line), and the drained event can be set while
    a body is executing, so a drain parks a still-working worker.

    Owning the threads moves every failure to ONE unambiguous place. :meth:`start`
    creates all of them before any work exists, so a refusal there has enqueued
    nothing; afterwards ``submit`` is a plain ``Queue.put`` on an unbounded queue,
    which cannot fail after enqueuing. That is what makes the caller's
    "not submitted ⇒ release the slot" inference true rather than merely usual."""

    def __init__(self, width: int) -> None:
        self._width = max(1, int(width))
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._closed = False

    def start(self) -> None:
        """Create every worker thread up front. If the OS refuses one, the ones
        already started are stopped and the error propagates — the pool is never
        left half-built, and no work has been accepted."""
        try:
            for i in range(self._width):
                t = threading.Thread(
                    target=self._loop, daemon=True, name=f"cpu-lane-{i}",
                )
                t.start()
                self._threads.append(t)
        except BaseException:
            self.shutdown(wait=False)
            raise

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:                  # shutdown sentinel
                return
            fn, args = item
            try:
                fn(*args)
            except BaseException:
                # The submitted callable already funnels its own failures; this
                # is the last net so a worker thread can never die and silently
                # narrow the lane for the rest of the process's life.
                log.exception("[claim-worker:cpu] lane worker swallowed a fatal")

    def submit(self, fn: Callable[..., Any], *args: Any) -> None:
        """Hand work to the lane. Cannot fail AFTER enqueuing — see the class
        docstring; that property is load-bearing for the caller's slot rule.

        Refuses outright once shut down, which is the other half of that rule.
        A bare ``put`` onto a queue whose workers have all consumed their
        sentinels would look like success and never run — the caller would mark
        the slot submitted, never release it, and both drain and shutdown would
        wait on the drained event forever. Raising keeps the caller's inference
        honest (``ThreadPoolExecutor`` raises here too)."""
        if self._closed:
            raise RuntimeError("cpu lane pool is shut down")
        self._q.put((fn, args))

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        """Stop the workers. ``wait`` joins them (bounded by ``timeout`` when
        given); the threads are daemons, so a detached shutdown cannot wedge
        interpreter exit.

        One sentinel per started thread, and the queue is FIFO — so sentinels
        land BEHIND any pending work and every enqueued job is still picked up
        before its worker exits. Idempotent, and safe before :meth:`start`."""
        self._closed = True
        for _ in self._threads:
            self._q.put(None)
        if wait:
            for t in self._threads:
                t.join(timeout=timeout)
        self._threads = []


class ClaimWorker:
    """One Postgres-as-queue worker. ``queue`` ∈ {cpu, gpu, download} ∪
    ``config.ingest_queues`` (default fetch/load).

    A GPU worker holds a single :class:`ModelCache` (concurrency-1 structural)
    and runs DAG node-jobs. A ``download`` worker (migration 0025) is the
    node-family network-WAIT lane: concurrency-1 like gpu, no model cache; the
    fleet-wide width cap lives in its CLAIM (``claim_next_download_job``), so
    running N download workers is safe — at most ``download_max_concurrent``
    of them hold a job at once. A fetch/load worker runs standalone ingest
    jobs from ``ingest_jobs`` — no model cache, no run-cancel watcher (there's
    no parent run), no ``$from`` resolution. ``run_once`` claims + executes one
    ready job; ``run_forever`` is the LISTEN/poll loop. The queue families
    share the lease / reclaim / watchdog machinery."""

    def __init__(
        self, *, queue: str, host: str | None = None,
        host_priority: int | None = None, lease_s: int = LEASE_S,
        model_cache: Any = None, threads: int | None = None,
    ) -> None:
        valid = _NODE_QUEUES | get_config().ingest_queues
        if queue not in valid:
            raise ValueError(f"queue must be in {sorted(valid)}, got {queue!r}")
        self.queue = queue
        self.host = host or _host_label()
        self.host_priority = (
            host_priority if host_priority is not None else _host_priority()
        )
        self.lease_s = int(lease_s)
        # GPU worker holds the process-wide warm-model cache; a unit test may
        # inject a fake. CPU + ingest workers have no cache.
        if model_cache is None and queue == "gpu":
            from queue_workflows.gpu_model_cache import gpu_model_cache
            model_cache = gpu_model_cache()
        self.model_cache = model_cache
        self._stop = threading.Event()
        # ── cpu lane width ────────────────────────────────────────────────
        # How many node-jobs THIS process runs at once. CPU only: gpu already
        # owns a two-lane PAR budget (inline diffusion + the VLM pool) that this
        # must not perturb, and ingest is out of scope for the knob — both are
        # pinned to 1. ``threads=`` beats the env/config resolution so a test (or
        # a host that computes its own width) can be explicit.
        if queue == "cpu":
            self.threads = max(
                1, int(threads) if threads is not None
                else resolve_cpu_worker_threads(),
            )
        else:
            self.threads = 1
        #: The cpu lane's executor — created lazily on the FIRST threaded claim,
        #: so a ``threads=1`` worker never builds one (the serial deployment
        #: carries no thread machinery at all).
        self._cpu_pool: _CpuLanePool | None = None
        #: The slots THIS process holds: ``reservation token -> (job_id,
        #: started_at)``, with ``job_id`` ``None`` until the claim lands.
        #:
        #: Keyed by the RESERVATION, not by the job id, and the distinction is
        #: load-bearing: a row this worker is still running can be re-queued out
        #: from under it (lapsed lease, a corrective requeue, the over-claim
        #: reaper) and then re-claimed BY THIS SAME PROCESS. An id-keyed map
        #: overwrites the first entry, so two live bodies share one slot — the
        #: count under-reads (an extra claim is admitted) and the map can empty
        #: while a body still runs, which sets the drained event and lets a drain
        #: park a worker that is still working.
        #:
        #: The ``started_at`` is the claim's own timestamp, and it is what makes
        #: the trip sweep honest (see :meth:`_trip_exit`): an id alone goes stale
        #: the moment the row is reclaimed and re-claimed elsewhere, and under a
        #: shared ``host_label`` a stale id would still match a sibling PROCESS's
        #: live row. A re-claim stamps a new one, so a stale entry is a no-op.
        self._cpu_inflight: dict[str, tuple[str | None, Any, int]] = {}
        #: Signalled whenever the in-flight set empties, so drain/shutdown can
        #: wait for the lane to quiesce without polling the DB.
        self._cpu_drained = threading.Event()
        self._cpu_drained.set()
        # GPU two-lane pool (NEW, additive): a PAR-sized ThreadPoolExecutor +
        # feeder thread for no-model VLM jobs, started in run_forever (gpu only).
        # None for cpu/ingest and until run_forever arms it.
        self._pool: ThreadPoolExecutor | None = None
        # ONE SERVER AT A TIME: the drain-based switch state machine. While a switch
        # is in flight this box is OUT OF SERVICE, because any claim it takes would
        # reload the server being replaced (see queue_workflows.server_switch).
        from queue_workflows.server_switch import ServerSwitch
        # Persist in the BOX-LOCAL lease dir (never NFS — same rule as the flock
        # lease) so a worker restart mid-switch re-enters the out-of-service window
        # instead of forgetting it and claiming while the incoming server warms.
        _lease_dir = env_get("QUEUE_WORKFLOWS_GPU_MODEL_LEASE_DIR", "").strip()
        _persist = (
            os.path.join(_lease_dir,
                         f"server_switch_{getattr(self, 'host', '?')}.json")
            if _lease_dir else None
        )
        _drain_s, _warm_s = _switch_timeouts()
        self._switch = ServerSwitch(label=getattr(self, "host", "?"),
                                    persist_path=_persist,
                                    drain_timeout_s=_drain_s,
                                    warm_timeout_s=_warm_s)
        # CAPABILITY GATE on resume: a ROCm box with no vLLM (its
        # topology has no distinct vllm URL) resumed a persisted vllm switch and
        # sat OUT OF SERVICE warming a server that can never exist.
        if (self._switch.target == "vllm"
                and self.queue == "gpu" and not self._box_has_vllm_endpoint()):
            self._switch.abandon("this box has no distinct vLLM endpoint")
        self._pool_max = 0
        self._pool_feeder: threading.Thread | None = None
        self._pool_inflight = 0
        self._pool_lock = threading.Lock()
        # True while the INLINE lane is executing a (model-backed diffusion) job.
        # The gpu VLM-pool feeder subtracts this from PAR so a machine's TOTAL
        # concurrent node-jobs (1 inline diffusion + pool VLM) never exceeds PAR —
        # the diffusion occupies one of the PAR slots. Plain bool: written by the
        # inline loop, read by the feeder thread (GIL-atomic; best-effort cap).
        self._inline_running = False
        # ONLY-GPU gate: True while this box's LLM server is serving on the GPU (or
        # is cold/unknown — optimistic). Set False
        # by the heartbeat probe when the server fell back to CPU (NVML/cgroup loss,
        # or a model too big for VRAM); the pool lane then SKIPS claiming no-model
        # LLM-dispatch jobs so they route to a GPU-backed box, and resumes when the
        # server returns to GPU. Defaults True so a cold worker is never blocked
        # before its first probe. Only meaningful for a gpu worker. See llm_probe.
        self._llm_gpu_ok = True
        self._llm_idle_reaper: Any = None
        self._residency_enforcer: Any = None
        # ONE-MODEL-PER-BOX lease (gpu_model_lease): a flock'd per-box file every
        # worker container on this physical GPU — regardless of project or DB —
        # coordinates through. This worker holds it for its currently-RESIDENT
        # model (renewed by the heartbeat tick, released when the model unloads);
        # the pre-execution gate SPILLS a job whose model would be a SECOND on the
        # box. Lazy (gpu only); inert until QUEUE_WORKFLOWS_GPU_MODEL_LEASE_DIR set.
        self._box_lease: Any = None
        self._box_held_model: str | None = None
        #: This worker's physical box name (placement, migration 0020) — lazily
        #: resolved once via :meth:`_physical_box`.
        self._box_name: str | None = None
        #: How many jobs are executing on this worker RIGHT NOW under the box hold.
        #: The box stays claimed while this is > 0 REGARDLESS of a residency-probe
        #: flicker (a long ollama job's model can read 'not resident' for a beat) —
        #: else the tick would release mid-job and never re-acquire (acquire runs
        #: only at claim), leaving the box unprotected. A counter, not a bool, so PAR
        #: pool jobs (all same-model by the lease) release the box only when the LAST
        #: one finishes. Falls to 0 on job end → the tick then frees the box once the
        #: warm model actually unloads.
        self._box_active_jobs: int = 0
        # Cached total-VRAM probe for the capacity claim gate (migration 0015) —
        # stable hardware property, sampled once per worker process.
        self._vram_probed_claim = False
        self._vram_total_claim: int | None = None
        # Worker capacity heartbeat — a no-op for fetch/load. cpu/ingest advertise
        # 1 (one poll-claim loop). GPU advertises max(1, PAR): the machine's GPU
        # job concurrency = its VLM pool size (worker_controls.llm_parallelism),
        # read live each tick so a UI PAR change is reflected without a restart.
        # A PAR-4 vLLM machine therefore reads GPU x/4 in the consumer pill; an
        # ollama machine (PAR 1) reads x/1. (NOT 1+PAR — the inline diffusion job
        # occupies one of those PAR slots, it is not an extra one.)
        self.heartbeat = HeartbeatEmitter(
            queue=self.queue, host_label=self.host,
            model_cache=self.model_cache,
            # Advertise what this box HOLDS, not what the software knows about. The
            # registry is populated identically on every box at import, so an
            # un-narrowed heartbeat says "box-b knows chat-expert-a" on a box that has
            # never pulled it — and `fits_models` is derived from this list, so the
            # dishonesty propagates into every capacity view.
            known_models_fn=self._known_models,
            # The heartbeat must advertise the SAME number the claim enforces —
            # _gpu_slot_budget (PAR capped by what the serving kind can absorb),
            # not the raw operator PAR. Advertising 3 while the claim takes 1
            # makes the panel read "1/3" and makes every capacity view (the
            # fill-before-spill defer heuristic reads `concurrency`) think this
            # box has spare slots it will never actually use.
            # gpu: the machine's PAR budget. cpu: the lane width, but ONLY when
            # widened — a threads=1 worker keeps the historical constant 1 so the
            # default deployment's heartbeat is byte-identical. Display-only:
            # the cpu claim is SKIP LOCKED and is never gated by this number.
            concurrency_fn=(self._gpu_slot_budget if self.queue == "gpu"
                            else self._cpu_lane_width if self.threads > 1
                            else None),
            # GPU worker: advertise the llm server it ACTUALLY answers on, probed
            # live from the box's resolved endpoint, so a box with no server shows
            # [] (no "OLLAMA · ON" chip) instead of the static ["ollama"] guess that
            # let a server-less box masquerade as capable. cpu/ingest keep the
            # config default (llm_servers_fn=None).
            llm_servers_fn=(self._probe_llm_servers if self.queue == "gpu" else None),
            # gpu: renew/release the box-wide model lease by observed residency.
            on_tick=(self._box_lease_tick if self.queue == "gpu" else None),
            # gpu: when no in-process model is loaded, the gauge shows the box
            # occupant instead ("GPU job: render_scene") — a self-managing
            # render used to heartbeat NULL and read as an idle box.
            occupant_fn=((lambda: self._box_held_model) if self.queue == "gpu" else None),
            # gpu: the context window this box serves and the one it could switch to
            # (migration 0028) — the claim filter for sized jobs and the input to the
            # fleet's unassignable sweep. Other lanes publish nothing: a cpu box has no
            # window, and a sized job must never route to one.
            llm_context_fn=(self._observe_llm_context if self.queue == "gpu" else None),
            residents_fn=(self._observed_residents if self.queue == "gpu" else None),
        )
        # This host's hw-metrics sampler — started in ``run_forever`` ONLY when
        # ``queue == 'gpu'`` (one gpu container per host ⇒ one sampler per host).
        self._hw_sampler: Any = None
        # Operator ON/OFF control watcher — started in ``run_forever`` after the
        # boot park-gate (None until then; stopped in the finally).
        self._control_watcher: Any = None
        # DRAIN flag (stop_policy="drain"): set by the control watcher; the claim
        # loop parks at its next iteration boundary — after the in-flight job —
        # instead of claiming again. The in-flight job is never touched.
        self._drain_evt = threading.Event()

    @property
    def drain_requested(self) -> bool:
        """True once an operator OFF with ``stop_policy='drain'`` has been seen and
        the worker hasn't parked-and-resumed yet."""
        return self._drain_evt.is_set()

    def request_drain(self) -> None:
        """Flag this worker to park at the next claim-loop boundary (the control
        watcher's drain handler). Idempotent; never interrupts the in-flight job."""
        self._drain_evt.set()

    def _drain_park_if_requested(self) -> bool:
        """Claim-loop boundary hook: when a drain is pending, park via the existing
        park-until-ON gate (LISTEN + safety poll), then clear the flag and resume
        claiming. Returns True iff it parked. Unlike the boot park, the heartbeat
        keeps beating here — the panel shows the box parked from worker_controls,
        and the kept in-flight job stays attributable to a live worker."""
        if not self._drain_evt.is_set():
            return False
        # A widened cpu lane returns from ``run_once`` the moment a job is
        # SUBMITTED, so "the boundary" is no longer "the body finished" — wait
        # for the lane to quiesce, or a drain would park a worker with N jobs
        # still running on it. Unbounded, exactly like the serial case (there,
        # ``run_once`` simply didn't return until the body did).
        if self.threads > 1:
            self._wait_for_inflight()
        log.warning(
            "[claim-worker:%s] drain boundary reached for host=%s — PARKED "
            "(in-flight work finished; not claiming) until turned back ON",
            self.queue, self.host,
        )
        self._park_until_enabled()
        self._drain_evt.clear()
        return True

    @property
    def _is_ingest(self) -> bool:
        # __init__ validated queue ∈ _NODE_QUEUES ∪ config.ingest_queues, so any
        # non-node queue is a (host-configured) ingest queue.
        return self.queue not in _NODE_QUEUES

    # ── claim ────────────────────────────────────────────────────────────

    def _publish_model_catalog(self, advertised, base_url, factory) -> None:
        """Publish WHICH MODELS this box's servers can serve (migration 0026).

        Rides the existing GPU heartbeat probe rather than adding a second loop: the probe
        already resolved this box's per-kind endpoints and just decided which ones answer,
        so asking each of them for its model list is one more HTTP call on a path that is
        already awake. A separate poller would drift out of step with `advertised` and
        publish capability for a server the heartbeat had already written off.

        Runs HERE, in the claim worker, and that is not incidental: in a common topology,
        the claim container resolves the LLM endpoint while the orchestrator container on
        the SAME box cannot (no host-gateway), so publishing from there would write an
        EMPTY catalog for a healthy box — and `best_match` treats unknown as cannot-serve,
        so an empty catalog does not degrade, it stops every LLM claim on that box.

        Best-effort by construction: any failure is swallowed. A heartbeat that dies
        because a model list was slow is worse than a catalog that is one tick stale.
        """
        if not advertised:
            return
        try:
            from queue_workflows import llm_probe, node_queue

            urls = {}
            for kind in advertised:
                url = factory.resolve_base_url(kind) or base_url
                if url:
                    urls[kind] = url
            entries = llm_probe.observe_box_models(self.host, urls_by_kind=urls)
            if entries:
                node_queue.upsert_model_catalog(entries)
                # ...and keep it for OUR OWN claim gate. This box's live probe is a
                # better answer than the table it just wrote — same data, no read-back,
                # no staleness window. `_known_models()` narrows the declared registry
                # with it; see model_catalog.narrow_to_observed for why an empty
                # observation must leave the declaration alone.
                self._observed_model_ids = sorted({e.model_id for e in entries})
                # ...and the FULL entries, digests included. The id list above feeds the
                # claim gate's narrowing; the entries feed per-claim ARTIFACT resolution
                # (which immutable tag the role currently serves on THIS box) — that
                # needs the digests, and re-probing per claim would put an HTTP call in
                # the claim hot path for data this probe just fetched.
                self._observed_entries = list(entries)
        except Exception:
            log.debug("[model-catalog] publish skipped", exc_info=True)

    def _stamp_resolved_artifact(self, job) -> None:
        """Record WHICH BYTES this claim will run — the artifact tag behind the role.

        ``resolved_model`` answers "which role" and is deliberately a stable tag
        (promotions re-point it), so it can never answer "which weights". A role tag can
        point at the wrong build while every job row truthfully records only the role name. This stamp is the per-job antidote:
        the immutable artifact tag, derived from THIS box's own digest observation at the
        moment of the claim (migration 0027).

        Best-effort and absent-not-guessed: no observation, no digest, or no matching
        artifact tag leaves the column NULL. A wrong stamp here would be worse than none —
        it would carry exactly the false authority the role name already has. Failures are
        swallowed: a claim that runs the job beats a claim that dies labelling it.
        """
        try:
            role = job.get("resolved_model") or job.get("required_model")
            if not role:
                return
            entries = getattr(self, "_observed_entries", None)
            if not entries:
                return
            from queue_workflows import model_registry, node_queue
            from queue_workflows.model_catalog import resolve_artifact

            server = job.get("resolved_server") or None
            options = (model_registry.server_model_options(
                str(role).split(":")[0]).get(server or "", ()) if server else ())
            artifact = resolve_artifact(str(role), server, entries, options)
            if not artifact:
                return
            node_queue.set_resolved_artifact(job.get("id"), artifact)
            job["resolved_artifact"] = artifact
        except Exception:
            log.debug("[artifact-stamp] skipped", exc_info=True)

    def _known_models(self, declared) -> list[str]:
        """What this worker advertises as claimable — DECLARED, narrowed by OBSERVED.

        ``model_registry.known_ids()`` is a process-level dict every box populates
        identically at import, so on its own it answers "which models does this software
        know about", not "which weights are on this disk". Before this, an expert pinned
        with ``force_box`` was the only thing keeping its rows off a box that had never
        pulled the tag. Narrowing here is what makes that pin redundant rather than
        load-bearing.

        Best-effort and fail-open: no observation yet ⇒ the declared list, unchanged.
        """
        try:
            from queue_workflows import model_registry
            from queue_workflows.model_catalog import narrow_to_observed

            # ONLY server-served ids are narrowed. The observation comes from ollama
            # /api/tags + vLLM /v1/models, so an in-process model can never appear in it
            # and filtering against it would delete that model from every box at once.
            narrowed = narrow_to_observed(
                declared, getattr(self, "_observed_model_ids", ()) or (),
                narrowable=model_registry.server_served_ids())
        except Exception:
            log.debug("[model-catalog] narrowing skipped", exc_info=True)
            narrowed = list(declared or [])
        # ...and then by what this box could LOAD right now (see _narrow_by_memory).
        return self._narrow_by_memory(narrowed)

    def _probe_llm_servers(self) -> list[str]:
        """OBSERVED, GPU-verified llm capability for THIS gpu box's heartbeat.

        Resolve the box's endpoint (per-box topology → env → default), probe what
        actually answers, AND check the server is serving on the GPU — then set the
        ONLY-GPU claim gate ``self._llm_gpu_ok`` and return the advertised server list.

        * server answers, model on GPU (or cold/vllm) ⇒ advertise it, gate OPEN.
        * server answers but serving on CPU (NVML/cgroup loss, or model over VRAM) ⇒
          advertise ``[]`` (NOT a usable GPU server — no "OLLAMA · ON" chip) and CLOSE
          the gate: the pool lane stops claiming so jobs route to a GPU-backed box.
          Reopens automatically on the next probe once the server is back on GPU.
        * no server at all ⇒ advertise ``[]``, gate CLOSED.

        Never raises — the emitter falls back to config on any exception."""
        from queue_workflows import llm_probe
        from queue_workflows.llm_backends import factory as _llm_factory

        base_url = _llm_factory.resolve_base_url()
        servers = llm_probe.probe_llm_servers(base_url)
        # CO-RESIDENCY (migration 0022): a box may run ollama AND vLLM at once, on
        # different ports. The default endpoint above finds at most one of them, so
        # also probe every OTHER kind whose resolved URL differs, and advertise the
        # union. Before 0022 both kinds resolved to the same URL, so this loop adds
        # nothing and the advertised list is byte-identical.
        for kind in (worker_control.SERVER_TYPE_OLLAMA, worker_control.SERVER_TYPE_VLLM):
            if kind in servers:
                continue
            other = _llm_factory.resolve_base_url(kind)
            if not other or other == base_url:
                continue
            if llm_probe.probe_llm_servers(other) == [kind]:
                servers = servers + [kind]
        # vLLM is CUDA-only (a live vLLM is GPU by construction); only ollama can
        # silently fall back to CPU, so only ollama is placement-probed.
        placement = (
            llm_probe.probe_gpu_placement(base_url)
            if servers == [llm_probe.OLLAMA]
            else llm_probe.GPU if servers else llm_probe.UNKNOWN
        )
        gpu_ok = llm_probe.gpu_usable(servers, placement)
        self._llm_gpu_ok = gpu_ok
        # Advertise only a GPU-usable server; a CPU-bound one reads as "no server".
        observed = servers if gpu_ok else []
        # DEBOUNCE. One timed-out probe used to collapse the
        # set for a tick — and everything hangs off it: _gpu_slot_budget dropped
        # 3→1, the DB-side arbiter refused every new claim against jobs admitted at
        # 3, and peers read the flap through the defer ranking. A kind JOINS on
        # first sighting (a server that answers is up); it LEAVES only after
        # _SERVED_DEBOUNCE_MISSES consecutive missed probes (a server that missed
        # once is usually just busy). The same rule guards RESIDENCY
        # (RESIDENT_ABSENT_MISSES), for the same reason: neither collector can tell a
        # timed-out endpoint from an idle one. This comment used to cite
        # _refresh_box_llm_state as already debouncing the warm snapshot — it did not,
        # it only kept the last answer on a RAISE the collector never performs, and a
        # later guard was built on that misreading.
        # The ONLY-GPU claim gate (_llm_gpu_ok) stays UNdebounced —
        # a CPU-fallback must stop LLM claims immediately.
        misses = getattr(self, "_served_miss_counts", None)
        if misses is None:
            misses = self._served_miss_counts = {}
        stable = set(getattr(self, "_box_served", ()) or ())
        for kind in observed:
            misses[kind] = 0
        stable |= set(observed)
        for kind in list(stable):
            if kind not in observed:
                n = misses.get(kind, 0) + 1
                misses[kind] = n
                if n >= _SERVED_DEBOUNCE_MISSES:
                    stable.discard(kind)
                    misses.pop(kind, None)
        advertised = [k for k in (worker_control.SERVER_TYPE_OLLAMA,
                                  worker_control.SERVER_TYPE_VLLM) if k in stable]
        # ONE SERVING KIND. The switch to vLLM unloads ollama's MODELS, not its daemon,
        # so after a switch both servers answer their probes and this box used to
        # advertise BOTH. An ollama-only job then passed the preference gate, landed
        # here, and its first request loaded a second copy of the same model
        # into the idle daemon beside vLLM's pool — the box hit full memory and froze.
        # vLLM pins its weights for its whole lifetime, so while it
        # serves it is the only kind offered; ollama is still in the debounced set and
        # returns the moment vLLM is gone.
        if worker_control.SERVER_TYPE_VLLM in advertised:
            advertised = [worker_control.SERVER_TYPE_VLLM]
        # Feed the 0022 claim-side preference gate from the SAME observation the
        # heartbeat advertises, so "what this box says it serves" and "what it will
        # claim" can never disagree. Empty ⇒ gate off (claim-any), by design.
        self._box_served = list(advertised)
        self._publish_model_catalog(advertised, base_url, _llm_factory)

        if not servers:
            log.error("%s", llm_probe.describe_endpoint(self.host, base_url, servers))
        elif placement == llm_probe.CPU:
            # Server up, but serving on CPU. Loud + actionable.
            log.error(
                "[llm] %s: LLM server at %s is serving on CPU, not GPU "
                "(NVML/cgroup GPU loss, or model too big for VRAM). REFUSING to "
                "claim GPU LLM jobs (policy: only GPU) until it returns to GPU — "
                "restart the LLM server to recover its GPU.",
                self.host, base_url,
            )
        elif not llm_probe.is_local_endpoint(base_url):
            log.warning("%s", llm_probe.describe_endpoint(self.host, base_url, servers))
        else:
            log.info("%s", llm_probe.describe_endpoint(self.host, base_url, servers))
        return advertised

    def _gpu_capacity_for_claim(self) -> tuple[int | None, list[str], bool]:
        """``(vram_total_mb, fits_models, has_models)`` for the GPU claim gate
        (migration 0015). ``vram_total_mb`` is this machine's total GPU VRAM,
        probed once and cached (stable hardware property); ``fits_models`` is the
        registered ids whose est_vram_gb fits it (the full known set when VRAM is
        unknown — claim-any-capable); ``has_models`` is whether the registry is
        populated at all. The registry is re-read each claim (it can grow as the
        host registers models) but the VRAM probe is not re-shelled.

        ``has_models`` matters for the empty-registry case: a COLD worker that
        hasn't registered its models yet has ``fits == []`` for the same reason a
        too-small machine does, but the two must behave differently — a cold
        worker falls back to claim-any (it must not wedge the queue), a too-small
        machine claims nothing. The caller branches on ``has_models`` to tell them
        apart."""
        from queue_workflows import hw_metrics, model_registry
        if not self._vram_probed_claim:
            try:
                self._vram_total_claim = hw_metrics.total_vram_mb()
            except Exception:
                log.exception("[claim-worker:gpu] total VRAM probe failed; unknown")
                self._vram_total_claim = None
            self._vram_probed_claim = True
        vram_total = self._vram_total_claim
        known = model_registry.known_ids()
        known_set = set(known)
        fits = [m for m in model_registry.fits_within(vram_total) if m in known_set]
        return vram_total, fits, bool(known)

    def _refresh_box_llm_state(self) -> None:
        """Re-observe this box's LLM state — which server kinds answer, and which
        (kind, model) pairs are already warm. Migration 0022's affinity inputs.

        Runs on the HEARTBEAT thread (its ~10 s tick), never on the claim path.
        Probing is 1-3 HTTP calls with real timeouts; doing it inline would make
        every claim pay for it and, with a PAR-wide pool feeder, serialise the lane
        behind a dead endpoint's timeout. The heartbeat is already this box's
        observer, so it owns the observation and the claim just reads the result.

        Uses the SAME collector as the residency enforcer and the load gate
        (``model_residency.probe_box_residents``) so all three see the box identically.

        That collector NEVER RAISES — "a dead endpoint contributes nothing", and each leg
        swallows a timeout to an empty list — so the ``except`` below is not the guard it
        looks like: a box whose ollama times out reports the same nothing as a box whose
        ollama is merely idle. Presence is therefore trusted immediately and ABSENCE is
        debounced: a model already seen must be missing from
        :data:`RESIDENT_ABSENT_MISSES` consecutive snapshots before it is dropped. Per
        MODEL, not per snapshot, because a box running two kinds where only one answers
        returns a non-empty list, which no emptiness test can catch. Same rule, same
        reason, as ``_probe_llm_servers``' served-set debounce."""
        servers: list[str] = []
        models: list[str] = []
        held: set[str] = set()
        try:
            from queue_workflows import llm_probe, model_residency

            for r in model_residency.probe_box_residents(
                inprocess_model=getattr(self.model_cache, "current_model", None),
            ):
                if r.server and r.server not in servers:
                    servers.append(r.server)
                if r.model and r.model not in models:
                    models.append(r.model)
                # What a SERVER was seen holding — the heartbeat's evidence that its warm
                # slot is still residency. The in-process leg is excluded deliberately:
                # it IS that slot, echoed back, and a claim cannot be its own witness.
                if r.model and r.server != llm_probe.INPROCESS:
                    held.add(str(r.model))
        except Exception:
            log.debug(
                "[claim-worker:gpu] warm-residency probe failed; keeping the last "
                "snapshot", exc_info=True,
            )
            return
        self._box_warm = (servers, models)
        self._ctx_residents = self._debounce_residents(held)

    #: How many consecutive snapshots a model must be MISSING before the box stops
    #: vouching for it. The collector cannot tell "idle" from "did not answer", and
    #: ollama's own keep-alive is five minutes, so three ~10 s ticks expires a genuine
    #: eviction inside 30 s while no single timed-out probe can blank a live model.
    RESIDENT_ABSENT_MISSES = 3

    def _debounce_residents(self, seen: set[str]) -> set[str]:
        """This tick's residents, with absence held for ``RESIDENT_ABSENT_MISSES``."""
        misses: dict[str, int] = self.__dict__.setdefault("_ctx_resident_misses", {})
        previous: set[str] = getattr(self, "_ctx_residents", None) or set()
        for model in seen:
            misses.pop(model, None)
        held = set(seen)
        for model in previous - seen:
            n = misses.get(model, 0) + 1
            if n < self.RESIDENT_ABSENT_MISSES:
                misses[model] = n
                held.add(model)          # not yet evidence of absence
            else:
                misses.pop(model, None)
        return held

    # ── the context window (migration 0028) ──────────────────────────────────

    #: The last observed ``(serves_now, could_switch_to)``. Class-level so the claim
    #: path can read it before the first heartbeat tick has run.
    _llm_context_state: tuple[int | None, int | None] = (None, None)
    #: The last model seen loaded on this box's ollama — what the ceiling is computed
    #: for once the runner has been unloaded (an idle box reports no runner at all).
    _ctx_last_model: str | None = None

    @property
    def _ctx_model_info(self) -> dict:
        cache = self.__dict__.get("_ctx_model_info_cache")
        if cache is None:
            cache = self.__dict__["_ctx_model_info_cache"] = {}
        return cache

    @property
    def _ctx_last_window(self) -> dict:
        """The last window each kind was OBSERVED serving. ollama only: a stopped vLLM
        means the server is gone, and remembering its window would route sized jobs at
        nothing (see :meth:`_llm_context_for_kind`)."""
        cache = self.__dict__.get("_ctx_last_window_cache")
        if cache is None:
            cache = self.__dict__["_ctx_last_window_cache"] = {}
        return cache

    @property
    def _ctx_per_token_seen(self) -> dict:
        """The cache cost per token OBSERVED on a loaded runner, per model — what the
        admission gate estimates a load by once that runner is gone."""
        cache = self.__dict__.get("_ctx_per_token_cache")
        if cache is None:
            cache = self.__dict__["_ctx_per_token_cache"] = {}
        return cache

    @property
    def _ctx_ceiling_by_kind(self) -> dict:
        """Per-kind ceilings, kept across a server going DOWN: a stopped vLLM reports
        nothing, and the window to start it back at has to come from somewhere."""
        cache = self.__dict__.get("_ctx_ceiling_cache")
        if cache is None:
            cache = self.__dict__["_ctx_ceiling_cache"] = {}
        return cache

    def _observe_llm_context(self) -> tuple[int | None, int | None]:
        """``(serves_now, could_switch_to)`` for this box, in tokens — the pair the
        heartbeat publishes and the claim filters on.

        Read per SERVING KIND and combined conservatively: the window is the SMALLEST
        any serving kind offers (a sized job may resolve to either, and promising the
        larger is how a prompt gets silently truncated), the ceiling the largest any
        could reach. Runs on the heartbeat thread — never on the claim path, which reads
        the cached pair.

        Best-effort in the honest direction: anything it cannot establish is ``None``,
        and a kind it cannot GROW (no lever, the kill switch, a resize quarantine after
        a failed attempt) has its ceiling pinned to what it serves — the fleet must not
        wait for a switch that will not happen."""
        state: tuple[int | None, int | None] = (None, None)
        try:
            served = [k for k in self._served_llm_types() if k]
            currents: list[int | None] = []
            ceilings: list[int] = []
            for kind in served:
                current, ceiling = self._llm_context_for_kind(kind)
                currents.append(current)
                if ceiling is not None:
                    ceilings.append(int(ceiling))
                    self._ctx_ceiling_by_kind[kind] = int(ceiling)
            if served:
                serving = (None if not currents or any(c is None for c in currents)
                           else min(int(c) for c in currents))
                ceiling = max(ceilings) if ceilings else None
                if serving is not None and ceiling is not None:
                    ceiling = max(ceiling, serving)
                state = (serving, ceiling)
                self._observe_admission(served, serving)
            if worker_control.SERVER_TYPE_VLLM in served:
                # The cap it sets applies only while vLLM serves (_pool_parallelism), so
                # nothing is reset when it leaves — and a stale cap on a vLLM that comes
                # back before its first observation errs on the safe side.
                self._observe_vllm_profile()
        except Exception:
            log.exception("[claim-worker:gpu] context-window observation failed")
            state = (None, None)
        self._llm_context_state = state
        return state

    #: How many heartbeat ticks a serving vLLM's log is re-read for its startup lines
    #: before this start is given up on (a TTY container, a rotated log, an older vLLM).
    _VLLM_PROFILE_ATTEMPTS = 30

    def _observe_vllm_profile(self) -> None:
        """ONCE per start of the vLLM serving here: read what it holds and what it cost.
        The concurrency its cache really holds (``GPU KV cache size`` ÷ the window — the
        metrics gauge first, the log if not) caps this box's PAR from now on; the start's
        numbers (weights, cache, tokens, pool) are remembered in the catalog under the
        command line's fingerprint, so the NEXT start is sized to what N requests need
        rather than to what is free. Runs on the heartbeat thread; never raises."""
        from queue_workflows import llm_context, llm_context_lever

        kind = worker_control.SERVER_TYPE_VLLM
        info = llm_context_lever.describe(kind)
        if not info.exists or not info.running:
            return
        key = (info.container_id, info.started_at)
        if key == self.__dict__.get("_vllm_observed_key"):
            return
        attempts = self.__dict__.get("_vllm_profile_attempts") or (None, 0)
        attempts = (key, attempts[1] + 1) if attempts[0] == key else (key, 1)
        self.__dict__["_vllm_profile_attempts"] = attempts

        from queue_workflows import llm_probe
        from queue_workflows.llm_backends import factory as _llm_factory

        window = info.configured or self._ctx_last_window.get(kind) or self._vllm_model_max_context()
        capacity = llm_probe.vllm_kv_capacity_tokens(_llm_factory.resolve_base_url(kind))
        profile = llm_context_lever.observe_vllm(total_bytes=llm_context.mem_total_bytes())
        if profile is not None and capacity is None:
            capacity = profile.kv_tokens
        concurrency = llm_context.vllm_concurrency(kv_tokens=capacity, window=window)
        if concurrency is not None:
            self.__dict__["_vllm_concurrency"] = concurrency
        if profile is None:
            if attempts[1] >= self._VLLM_PROFILE_ATTEMPTS:
                log.warning("[vllm-plan] %s: vLLM's startup lines were not found in its log "
                            "after %d reads — no profile of %s here; the next start is sized "
                            "to what is free, as before", self.host, attempts[1],
                            info.served_name or info.name)
                self.__dict__["_vllm_observed_key"] = key
            return
        if info.served_name:
            node_queue.record_llm_profile(self.host, kind, info.served_name,
                                          profile.as_record(fingerprint=info.fingerprint))
        self.__dict__["_vllm_observed_key"] = key
        # Its pool is taken and its cache allocated: from here the memory it holds is a
        # fact about the box, not a start to be judged (see _vllm_pool_committed).
        self.__dict__["_vllm_committed_key"] = key
        self._clear_startup_losses()
        par = self._pool_parallelism_configured()
        log.warning(
            "[vllm-plan] %s: vLLM %s at pool %.2f — weights %.1f GB, cache %.1f GB = %s tokens "
            "(%.1f KB/token): %d concurrent %s-token request(s) fit (--max-num-seqs %s, "
            "PAR %d → this box runs %d); profile remembered — the next start is sized to "
            "what it needs",
            self.host, info.served_name or info.name, profile.utilization,
            profile.weights_bytes / 2**30, profile.kv_bytes / 2**30, f"{profile.kv_tokens:,}",
            profile.per_token_bytes / 1024, int(concurrency or 0),
            f"{int(window):,}" if window else "?", info.max_num_seqs_now, par,
            self._pool_parallelism(),
        )

    def _observed_residents(self) -> set[str] | None:
        """What this box's SERVERS were last seen holding, or ``None`` for "not observed".

        ONE TICK BEHIND, deliberately. The emitter runs `emit_once()` and then its
        `on_tick` hook, and the snapshot is taken in that hook because the ROGUE-CONTAINER
        SWEEP must run before any residency probe — a box must be judged after the guard
        has cleaned it, or an outsider's model counts as a resident. Moving the probe
        earlier to publish a freshly-loaded model a tick sooner would invert that, so the
        lag stands: a model loaded since the last snapshot is published as unknown for one
        ~10 s tick, which is the safe direction and costs only the panel.

        No probe of its own — this is `_refresh_box_llm_state`'s snapshot, taken on the
        heartbeat thread each tick with `model_residency.probe_box_residents`, the same
        collector the residency enforcer and the load gate read "so all three see the box
        identically". That collector does NOT keep the last answer when a probe fails —
        it never raises, and every leg swallows a timeout to an empty list, which is
        indistinguishable from an idle box. Absence is therefore debounced HERE, by
        :meth:`_debounce_residents`: a model already seen survives
        :data:`RESIDENT_ABSENT_MISSES` consecutive misses before the box stops vouching
        for it, so no single flaky tick can blank a live model."""
        return getattr(self, "_ctx_residents", None)

    def _llm_context_for_kind(self, kind: str) -> tuple[int | None, int | None]:
        """One serving kind's ``(serves_now, could_switch_to)``."""
        from queue_workflows import llm_context, llm_context_lever, llm_probe
        from queue_workflows.llm_backends import factory as _f

        url = _f.resolve_base_url(kind)
        # A box can only advertise room it could actually use: the kill switch, a
        # missing lever, or a resize that already failed here all mean "what I serve is
        # what I reach" — the fleet must not tag nothing and wait for a switch that
        # cannot happen.
        can_grow = (not llm_context_lever.disabled()
                    and llm_context_lever.available(kind)
                    and not self._switch.resize_quarantined(kind))
        if kind == worker_control.SERVER_TYPE_VLLM:
            current = llm_probe.vllm_max_model_len(url)
            if not can_grow:
                return current, current
            ceiling = llm_context.vllm_ceiling(
                current=current, model_max=_vllm_model_max_context(),
                kv_capacity_tokens=llm_probe.vllm_kv_capacity_tokens(url))
            return current, ceiling

        # ollama. The CONFIGURED value wins over any loaded runner: /v1 loads at the
        # configured default, and a runner another client loaded natively at a bigger
        # window is undone by our very next call.
        runners = llm_probe.ollama_runners(url)
        loaded = runners[0] if runners else None
        if loaded and loaded.get("name"):
            self._ctx_last_model = str(loaded["name"])
        configured = llm_context_lever.describe(kind).configured
        current = configured
        if current is None and loaded:
            current = loaded.get("context_length")
        if current is None:
            # IDLED OUT, NOT GONE. ollama's daemon is always up; only the model unloads
            # (5 min here), and the next request reloads it at the same window. A box
            # with no lever to ask then has no runner to read either — and publishing
            # nothing means it can neither claim a sized job nor escape being told it is
            # unassignable, until some unsized job happens to warm it again. The window
            # belongs to the DAEMON, so the last one OBSERVED stands until something
            # contradicts it. Never invented: unknown stays unknown until first sight —
            # and "seen" survives a restart, because the memory this box wrote to its own
            # heartbeat is the same observation, one process ago.
            current = self._ctx_last_window.get(kind) or self._seeded_window(kind)
        elif current:
            self._ctx_last_window[kind] = int(current)
        model = (str(loaded.get("name")) if loaded else None) or self._ctx_last_model
        model_max = (llm_context.model_info_max_context(
            self._ollama_model_info(url, model)) if model else None)
        if current is not None and model_max is not None:
            # ollama clamps a bigger request to the model's own maximum, so a box
            # configured beyond it does NOT serve what the number says.
            current = min(int(current), int(model_max))
        if not can_grow or model is None:
            return current, current
        sizes = self.__dict__.get("_ctx_model_sizes") or llm_probe.ollama_model_sizes(url)
        weights = sizes.get(model)
        runner_bytes = int(loaded.get("size") or 0) if loaded else 0
        per_token = llm_context.kv_bytes_per_token_observed(
            runner_bytes=runner_bytes, weights_bytes=weights,
            context_length=(loaded or {}).get("context_length"),
        ) or llm_context.kv_bytes_per_token_formula(
            self._ollama_model_info(url, model))
        ceiling = llm_context.ollama_ceiling(
            current=current, model_max=model_max, per_token=per_token,
            weights_bytes=weights, runner_bytes=runner_bytes,
            available_bytes=llm_context.mem_available_bytes(),
            reserve_bytes=llm_context.reserve_bytes(),
        )
        return current, ceiling

    def _seeded_window(self, kind: str) -> int | None:
        """This box's own last published window, read back ONCE per process.

        Only for ollama, and only as the memory of last resort: a fresh worker on a box
        with no lever, whose model is not loaded right now, would otherwise have nothing
        to say about a daemon whose window has not changed. Read once — the heartbeat
        asks every 10 s and this answer cannot change while the process runs (a live
        observation replaces it in the memory it seeds)."""
        if kind != worker_control.SERVER_TYPE_OLLAMA:
            return None                      # a stopped vLLM serves nothing to remember
        if self.__dict__.get("_ctx_seed_done"):
            return None
        self.__dict__["_ctx_seed_done"] = True
        try:
            # OLLAMA'S OWN publications only. The row also carries what a vLLM on this
            # box served, and that number says nothing about a daemon whose default is
            # chosen from VRAM — adopting it is how a box advertises a window it has
            # never observed, and a sized job sent there is silently truncated.
            seen = node_queue.last_published_context(self.host, self.queue,
                                                     served_by=kind)
        except Exception:
            log.debug("[llm-context] %s: window read-back failed", self.host,
                      exc_info=True)
            return None
        if seen:
            self._ctx_last_window[kind] = int(seen)
            log.info("[llm-context] %s: no runner loaded — resuming the %d-token window "
                     "this box last published (it belongs to the daemon, not the "
                     "runner)", self.host, int(seen))
        return seen

    def _ollama_model_info(self, url: str, model: str | None) -> dict:
        """``/api/show`` model_info, cached for the process: it describes the WEIGHTS,
        which do not change under a tag without a re-pull, and the heartbeat asks every
        tick."""
        if not model:
            return {}
        cache = self._ctx_model_info
        if model not in cache:
            from queue_workflows import llm_probe
            cache[model] = dict(llm_probe.ollama_model_info(url, model) or {})
        return cache[model]

    def _observe_admission(self, served: list[str], serving: int | None) -> None:
        """What this box could LOAD right now — the inputs :meth:`_narrow_by_memory`
        judges a claim by. Runs on the heartbeat, never on the claim path.

        Residents (a loaded ollama runner, vLLM's pinned model) always fit; anything
        else would have to be loaded, and ollama loads on the FIRST REQUEST, so the claim
        is the engine's only hand on it. The sizes are the box's own inventory of weights
        on disk (ollama ``/api/tags``, asked whichever kind serves — the daemon answers
        either way); the cache is what one token of window costs here, times the window
        the box serves."""
        from queue_workflows import llm_context, llm_probe
        from queue_workflows.llm_backends import factory as _f

        resident: set[str] = set()
        sizes: dict[str, int] = {}
        kv_bytes = 0
        try:
            ollama_url = _f.resolve_base_url(worker_control.SERVER_TYPE_OLLAMA)
            sizes = dict(llm_probe.ollama_model_sizes(ollama_url) or {})
            runners = llm_probe.ollama_runners(ollama_url) if ollama_url else []
            resident |= {str(r.get("name")) for r in runners if r.get("name")}
            if worker_control.SERVER_TYPE_VLLM in served:
                resident |= {str(m) for m in llm_probe.vllm_served_models(
                    _f.resolve_base_url(worker_control.SERVER_TYPE_VLLM)) or []}
            loaded = runners[0] if runners else None
            model = (str(loaded.get("name")) if loaded else None) or self._ctx_last_model
            per_token = None
            if loaded:
                per_token = llm_context.kv_bytes_per_token_observed(
                    runner_bytes=int(loaded.get("size") or 0),
                    weights_bytes=sizes.get(str(loaded.get("name"))),
                    context_length=loaded.get("context_length"))
                if per_token:
                    # The measured cost of THIS model on THIS daemon — remembered, so
                    # the estimate does not collapse to a formula the moment it unloads.
                    self._ctx_per_token_seen[str(loaded.get("name"))] = int(per_token)
            if per_token is None and model:
                per_token = self._ctx_per_token_seen.get(model)
            if per_token is None and model:
                # ADMISSION, not the ceiling. The formula's x4 slots x1.25 safety is the
                # conservative direction for "how big could the window grow" — and the
                # wrong one here: it refused a ~16 GB model into ~4x that much free room.
                # One slot must fit; the reserve covers the rest.
                per_token = llm_context.kv_bytes_per_token_formula(
                    self._ollama_model_info(ollama_url, model), parallel=1, safety=1.0)
            kv_bytes = int(per_token or 0) * int(serving or 0)
        except Exception:
            log.debug("[llm-memory] %s: admission observation incomplete", self.host,
                      exc_info=True)
        self.__dict__["_ctx_model_sizes"] = sizes
        self.__dict__["_ctx_admission"] = {
            "resident": resident, "sizes": sizes, "kv": kv_bytes,
            "available": llm_context.mem_available_bytes(),
            "reserve": llm_context.reserve_bytes(),
        }

    def _narrow_by_memory(self, ids: list[str]) -> list[str]:
        """Drop the models this box could not LOAD right now — the memory gate before
        the load, applied where the engine has a hand on it: the claim.

        Example: vLLM held most of a box's memory, little was free, and a claim for a
        job needing a model not resident here pulled that model's weights into the rest.
        A model already resident always passes; one whose size the box does not know
        passes too (a box that cannot describe itself must not wedge the queue); a KNOWN
        misfit is refused and the job waits for a box that can hold it — or is tagged
        unassignable if none can, because the same set is what the heartbeat publishes as
        ``fits_models``. Before the first observation nothing is narrowed."""
        adm = self.__dict__.get("_ctx_admission")
        if not adm:
            return list(ids)
        try:
            from queue_workflows import llm_context
            from queue_workflows.model_catalog import canonical_tag

            resident = {canonical_tag(m) for m in adm["resident"]}
            sizes = {canonical_tag(k): v for k, v in adm["sizes"].items()}
            kept: list[str] = []
            refused: list[tuple[str, int]] = []
            for mid in ids:
                tag = canonical_tag(mid)
                if tag in resident:
                    kept.append(mid)
                    continue
                fits = llm_context.load_fits(
                    weights_bytes=sizes.get(tag), kv_bytes=adm["kv"],
                    available_bytes=adm["available"], reserve_bytes=adm["reserve"])
                if fits is False:
                    refused.append((mid, int(sizes.get(tag) or 0)))
                else:
                    kept.append(mid)
            names = {m for m, _ in refused}
            if names and names != self.__dict__.get("_ctx_refused_last"):
                log.warning(
                    "[llm-memory] %s: NOT offering to load %s — %.1f GB available, %.0f GB "
                    "reserved for everything else, cache %.1f GB at this box's window; "
                    "the job waits for a box that can hold it",
                    self.host,
                    ", ".join(f"{m} ({b / 2**30:.1f} GB)" for m, b in refused),
                    (adm["available"] or 0) / 2**30, (adm["reserve"] or 0) / 2**30,
                    (adm["kv"] or 0) / 2**30,
                )
            self.__dict__["_ctx_refused_last"] = names
            return kept
        except Exception:
            log.debug("[llm-memory] %s: admission narrowing skipped", self.host,
                      exc_info=True)
            return list(ids)

    def _vllm_start_plan(self) -> "Any":
        """How to start vLLM here — see :func:`llm_context.vllm_plan`. With a remembered
        profile of the model this container serves (the catalog, migration 0029, keyed by
        the command line's fingerprint), the pool is what N requests of the window NEED and
        N is what the box can afford; without one, the memory gate's rule: the configured
        pool lowered to what is free, refused below the minimum. Never raises."""
        from queue_workflows import llm_context, llm_context_lever

        kind = worker_control.SERVER_TYPE_VLLM
        if llm_context_lever.disabled() or not llm_context_lever.available(kind):
            return llm_context.VllmPlan(None, None, None, "")
        info = llm_context_lever.describe(kind)
        if not info.exists:
            return llm_context.VllmPlan(None, None, None, "")
        if (info.running and self._vllm_pool_committed(info)
                and not self._box_memory_in_danger()):
            # THE MEMORY GATE GOVERNS STARTS. This server is already running and its
            # startup is COMPLETE — its pool taken, its cache allocated. Judging it
            # against a fresh MemAvailable reading judges it against its OWN pool:
            # A server can allocate most of the box's memory as cache after a long load; a
            # worker that rolls a minute later reads little free, plans a start that could
            # not fit and stops the server that had just finished. A verdict about a
            # start that is not happening must not kill one that already happened —
            # unless the box is in real danger, where the LLM server is the biggest
            # thing the engine can give back.
            return llm_context.VllmPlan(None, None, None, "")
        window = (self._switch.target_context or info.configured
                  or self._vllm_model_max_context())
        plan = llm_context.vllm_plan(
            total_bytes=llm_context.mem_total_bytes(),
            available_bytes=llm_context.mem_available_bytes(),
            reserve_bytes=llm_context.reserve_bytes(),
            operator_utilization=info.utilization,
            operator_max_seqs=info.max_num_seqs or self._pool_parallelism_configured(),
            window=window, profile=self._vllm_remembered_profile(info),
            min_utilization=_vllm_min_utilization())
        plan = self._yield_to_ollama_if_thin(plan)
        plan = self._refuse_a_start_that_would_be_wasted(plan)
        if plan.reason:
            log.info("[vllm-plan] %s: %s", self.host, plan.reason)
        return plan

    def _refuse_a_start_that_would_be_wasted(self, plan: "Any") -> "Any":
        """A start is worth its 6-11 minutes only if the box is likely to still hold it
        at the end. With a clean history the plan's own arithmetic settles that; after a
        load was killed mid-way, the same conditions are known NOT to be enough, so the
        start waits until there is :meth:`_vllm_startup_headroom` more than the bare
        need — the same bar the memory-recovery lift uses."""
        from queue_workflows import llm_context

        if plan.refusal is not None or plan.utilization is None and plan.max_num_seqs is None:
            if plan.refusal is not None:
                return plan
        losses = int(self.__dict__.get("_vllm_startup_losses") or 0)
        if not losses:
            return plan
        available = llm_context.mem_available_bytes()
        need = self._vllm_memory_need_bytes()
        if available is None or need is None or int(available) >= need:
            return plan
        return llm_context.VllmPlan(None, None, (
            f"{llm_context.MEMORY_REFUSAL}a vLLM load was killed mid-load {losses} time(s) "
            f"on this box since one completed, so a start needs "
            f"{need / 2**30:.1f} GB free ({self._vllm_startup_headroom():.2f}x the bare "
            f"need) to be worth its minutes — {int(available) / 2**30:.1f} GB now"),
            "not worth the load time yet")

    def _yield_to_ollama_if_thin(self, plan: "Any") -> "Any":
        """A plan that can afford fewer than :func:`_vllm_min_concurrency` requests is
        no reason to start vLLM on a box that also has ollama: batching is the whole
        advantage, and one slot of it costs a pool ollama does not need. On a box with
        no ollama of its own there is no alternative, so the thin plan stands."""
        from queue_workflows import llm_context

        minimum = _vllm_min_concurrency()
        if (plan.refusal is not None or plan.max_num_seqs is None
                or plan.max_num_seqs >= minimum or not self._box_has_ollama_endpoint()):
            return plan
        return llm_context.VllmPlan(None, None, (
            f"{llm_context.MEMORY_REFUSAL}vLLM would hold this box for {plan.max_num_seqs} concurrent request(s), "
            f"below the {minimum} it takes to be worth its pool "
            f"(QUEUE_WORKFLOWS_VLLM_MIN_CONCURRENCY) — ollama serves this box without "
            f"one ({plan.reason})"), "one request is not worth a vLLM")

    def _box_has_ollama_endpoint(self) -> bool:
        """Can this box serve ollama itself? The same discriminator
        :meth:`_box_has_vllm_endpoint` uses, the other way round: an ollama URL that is
        real and DISTINCT from the vLLM one. A box whose topology resolves both kinds to
        one URL has no ollama of its own to fall back to."""
        try:
            from queue_workflows.llm_backends import factory as _f
            o, v = _f.resolve_base_url("ollama"), _f.resolve_base_url("vllm")
            return bool(o) and o != v
        except Exception:
            return False

    def _box_memory_in_danger(self) -> bool:
        """Is this box close enough to running out that the LLM server should go? Only
        a KNOWN shortage counts: an unreadable meminfo is not danger.

        The line is an ABSOLUTE floor (:func:`_danger_floor_bytes`), deliberately not a
        multiple of the reserve. The reserve is a PLANNING margin — how much the next
        start must leave behind — and raising it on a large unified-memory box would
        move a reserve-scaled danger line so high that a box holding a legitimate large
        pool (with ample free memory) would be declared dying, every tick. Danger is a different
        quantity: how little is left before ``fork()`` fails and the kernel refuses GPU
        allocations. A box can die at a MemAvailable of several GB — MemAvailable
        counts reclaimable page cache a CUDA allocation cannot take, which is why the
        floor sits above zero and not at the reserve."""
        from queue_workflows import llm_context

        available = llm_context.mem_available_bytes()
        if available is None:
            return False
        return available < _danger_floor_bytes()

    def _abandon_warm_if_box_is_dying(self) -> bool:
        """While a vLLM WARMS, keep watching the BOX. Returns True when the warm was
        given up (the caller does nothing else this tick).

        The pool is decided ONCE per switch on purpose — re-deciding it against the
        memory the start itself is consuming is what made a box recreate its container
        at ever-lower utilizations and kill its own load. But "do not re-plan" was read as
        "do not look": a first start of a model had no profile, used the configured
        utilization which fitted what was then free, loaded beside the box's other
        tenants, and drained MemAvailable until sshd could not fork and the gpu heartbeat
        went minutes stale — with nothing watching.

        Deciding the pool and watching the box are different questions. This one keeps
        being asked, and only a KNOWN crossing of the danger line answers it: a load
        LOWERS MemAvailable by design, so "less than when we decided" means nothing,
        and an unreadable meminfo is not danger. Nothing is recomputed and nothing is
        recreated here — the server is given back and the switch abandoned."""
        from queue_workflows import server_switch as _sw

        if self._switch.state != _sw.WARMING:
            return False
        if self._switch.target != worker_control.SERVER_TYPE_VLLM:
            return False
        if not self._box_memory_in_danger():
            return False
        from queue_workflows import llm_context

        available = llm_context.mem_available_bytes() or 0
        why = (f"the box has {available / 2**30:.1f} GB left while vLLM warms — under the "
               f"{_danger_floor_bytes() / 2**30:.0f} GB floor "
               f"(QUEUE_WORKFLOWS_LLM_DANGER_FLOOR_MB). A start that is taking the box "
               f"down is not a start worth finishing")
        started = self.__dict__.pop("_warm_started_at", None)
        cost = (f"{(time.monotonic() - started) / 60.0:.1f} min into the load"
                if started else "mid-load")
        log.error("[llm-memory] %s: ABANDONING the vLLM warm at %s — %s", self.host, cost, why)
        self._stop_loading_vllm(why)
        self._switch.abandon(why)
        self._note_startup_loss(f"{why} — {cost}")
        return True

    def _vllm_pool_committed(self, info: "Any") -> bool:
        """Has THIS start of this container finished taking its memory? True once its
        startup was observed complete (:meth:`_observe_vllm_profile` — the cache
        allocated, in the server's own words). Keyed on the (container, start) pair, so
        a restart at a bigger pool is a new start with its memory still ahead of it."""
        return (self.__dict__.get("_vllm_committed_key")
                == (info.container_id, info.started_at))

    def _vllm_remembered_profile(self, info: "Any") -> "Any":
        """The catalog's profile of the model this box's vLLM serves — only when it was
        taken on the SAME command line (minus the engine's own flags); a model re-quantised
        or re-flagged is a different profile, and none is better than a wrong one."""
        from queue_workflows import llm_context

        if not getattr(info, "served_name", None):
            return None
        record = node_queue.llm_profile(self.host, worker_control.SERVER_TYPE_VLLM,
                                        info.served_name)
        if not record or str(record.get("fingerprint") or "") != str(info.fingerprint or ""):
            return None
        return llm_context.VllmProfile.from_record(record)

    def _vllm_start_utilization(self) -> tuple[float | None, str | None]:
        """``(pool, refusal)`` of :meth:`_vllm_start_plan` — kept for callers of the
        pool-only shape."""
        plan = self._vllm_start_plan()
        return plan.utilization, plan.refusal

    def _vllm_start_decision(self) -> "Any":
        """:meth:`_vllm_start_plan`, decided ONCE per switch and reused on the retries.
        The switch re-issues START every 30 s while the server warms, and a fresh reading
        then sees MemAvailable already lowered by the very vLLM that is loading — a box
        recreated its container at ever-lower utilizations, killing the load each
        time. A retry is the same start; its verdict was taken when the switch entered
        WARMING."""
        key = self._switch.requested_wall
        cached = self.__dict__.get("_vllm_decision")
        if cached is not None and cached[0] == key:
            return cached[1]
        decision = self._vllm_start_plan()
        self.__dict__["_vllm_decision"] = (key, decision)
        return decision

    def _stop_loading_vllm(self, why: str, *, what: str = "still loading") -> bool:
        """Stop this box's vLLM container while it is RUNNING — after the memory gate
        refused its pool, or because the box is out of memory (:meth:`_shed_llm_if_box_is_dying`).
        Best-effort: no lever, no container, or a stop that fails leaves a loud log
        line, never an exception into the caller. Returns True when a stop was issued."""
        from queue_workflows import llm_context_lever
        try:
            if not llm_context_lever.describe(worker_control.SERVER_TYPE_VLLM).running:
                return False
            stop = get_config().vllm_stop_fn
            if stop is None:
                log.error("[llm-memory] %s: a vLLM is %s in memory the box does not have "
                          "and no stop lever is wired (set_vllm_lifecycle) — stop it by "
                          "hand: %s", self.host, what, why)
                return False
            stop()
            log.error("[llm-memory] %s: STOPPED a vLLM that was %s — %s", self.host, what, why)
            return True
        except Exception:
            log.exception("[llm-memory] %s: could not stop the vLLM", self.host)
            return False

    def _serving_context(self) -> int | None:
        """The window this box serves, as last observed — a plain read for the claim
        path (the observation itself is the heartbeat's work)."""
        return (getattr(self, "_llm_context_state", None) or (None, None))[0]

    def _served_llm_types(self) -> list[str]:
        """The LLM server kinds THIS box can serve — the claim-side gate for a job's
        ``server_types`` preference (migration 0022).

        Read from the heartbeat's last PROBED capability, so it reflects what
        actually answers rather than what an operator hoped for. Empty ⇒ the gate is
        off and every row is claimable: the same cold-worker escape hatch
        ``known_models`` uses, because a worker that cannot yet describe itself must
        never wedge the queue."""
        return list(getattr(self, "_box_served", ()) or ())

    def _warm_state(self) -> tuple[list[str], list[str]]:
        """The affinity inputs — ``(warm_server_kinds, warm_model_ids)`` — as last
        observed by :meth:`_refresh_box_llm_state`. A plain read: no I/O, no lock,
        safe to call on every claim. Empty ⇒ every row scores 0 and the sort falls
        back to its pre-0022 order."""
        warm = getattr(self, "_box_warm", None)
        if not warm:
            return [], []
        return list(warm[0]), list(warm[1])

    #: What each LLM server kind can usefully do AT ONCE. ollama SERIALISES per
    #: model — a second concurrent request queues behind the first, so extra node-job
    #: slots buy nothing and merely hold work hostage on one box. vLLM does
    #: continuous batching, so concurrency is real there. ``None`` = "no ceiling of
    #: its own, use the operator's PAR".
    _KIND_CONCURRENCY: dict[str, int | None] = {"ollama": 1, "vllm": None}

    def _box_has_vllm_endpoint(self) -> bool:
        """Can this box plausibly serve vLLM AT ALL? The discriminator is the same
        one probe_box_residents uses: a vllm URL DISTINCT from ollama's. A bare
        topology entry resolves both kinds to one URL — that box has no vLLM, and
        no amount of queue pressure may make the demand-starter chase one (else a box
        holds a doomed switch cycle against thousands of pending rows)."""
        try:
            from queue_workflows.llm_backends import factory as _f
            o, v = _f.resolve_base_url("ollama"), _f.resolve_base_url("vllm")
            return bool(v) and v != o
        except Exception:
            return False

    # ``_default_llm_kind`` was removed as dead code:
    # it fed ``claim_next_gpu_job(default_server=...)``, which the claim SQL bound
    # and never read, at the cost of an UNCACHED ``worker_controls`` SELECT on
    # every single claim. The law gave the parameter its reader — an
    # UNDECLARED row's stamp IS this prediction (the old "unconstrained rows are
    # not LLM-served" premise was false: a real LLM-calling node had exactly
    # that row shape) — so it is back, via ``predict_default_kind`` below,
    # which keeps the actual lesson: no per-claim DB read (a warm hint the worker
    # already holds, else the factory's TTL cache) and no exception path into the
    # claim loop.

    def _kind_caps(self) -> dict[str, int]:
        """PER-KIND concurrent-job caps for the claim arbiter (migration 0023).
        ollama SERIALISES per model — its cap is 1 no matter how many slots the box
        advertises; vLLM batches, so it gets the operator PAR. This is what stops
        3 ollama-resolved jobs queuing inside the daemon while vLLM work runs
        3-wide on the same box (ollama allows 1)."""
        par = max(1, self._pool_parallelism())
        return {"ollama": 1, "vllm": par}

    def _gpu_slot_budget(self) -> int:
        """This box's TOTAL concurrent GPU job budget — the same number the heartbeat
        advertises as ``concurrency`` and the DB-side box-slot arbiter enforces at
        claim time (``claim_next_gpu_job(max_running=...)``).

        PAR is the operator's CEILING; the SERVING KIND sets the real one. Raising PAR
        to 3 on a box serving ollama gave it three node-job slots for a server that
        answers one request at a time (NODE JOBS 3/1, three
        llm_chat jobs queued behind one another inside ollama). The budget is
        therefore the operator's PAR capped by what the kinds actually serving here
        can absorb: an ollama-only box is 1, as it was before vLLM existed; a box
        serving vLLM gets the full PAR.

        Falls back to PAR when nothing is observed yet — a cold box must not clamp
        itself to 1 forever if it is about to serve vLLM."""
        par = max(1, self._pool_parallelism())
        serving = [k for k in (self._served_llm_types() or []) if k]
        if not serving:
            return par
        caps = [self._KIND_CONCURRENCY.get(k) for k in serving]
        if any(c is None for c in caps):
            return par                      # at least one batching server → full PAR
        return max(1, min(par, max(int(c) for c in caps if c is not None)))

    def _gpu_active_count(self) -> int:
        """GPU jobs in flight on this box right now: the VLM pool's inflight count
        plus the inline diffusion lane (0/1). >0 means the GPU IS in use, so the
        idle reaper must not unload. Best-effort read (a stale value only shifts an
        unload by one poll)."""
        return int(self._pool_inflight) + (1 if self._inline_running else 0)

    def _claim(self, *, free_slots: int | None = None) -> dict | None:
        """Claim the next job for the INLINE lane.

        ``free_slots`` (cpu, widened lane only) is how much capacity the caller
        has reserved; the claim then returns only a row whose ``cpu_slots`` fits
        it (migration 0024). ``None`` ⇒ no width term — the serial path, and gpu.

        GPU: the inline lane runs warm-model diffusion ONLY — it claims with
        ``require_model=True`` so it never grabs a no-model VLM job (those are
        the pool lane's, claimed via :meth:`_claim_pool`). This keeps the two
        lanes' claim sets disjoint (no stealing, no over-claim). On a host with
        the VLM pool disabled (PAR clamps to >= 1, but no no-model jobs exist)
        this is a pure narrowing — diffusion behaviour is byte-identical."""
        if self.queue == "gpu" and self._switch.blocks_claims():
            # Out of service for the duration of a server switch. Taking work here is
            # exactly what reloaded the outgoing server and produced the two-resident
            # state; a peer takes it instead.
            return None
        if self.queue == "gpu" and self._spill_paused():
            # Our own box just denied us the card. Claiming again right now returns the
            # row we just spilled (it goes back to the FRONT) and spills it again —
            # see SPILL_PAUSE_S.
            return None
        if self._is_ingest:
            return ingest_store.claim_next_ingest_job(
                self.queue, host=self.host, lease_s=self.lease_s,
            )
        if self.queue == "download":
            # The width cap (default 1, fleet-wide per project) lives INSIDE
            # the claim — resolved there too (env → config → 1), so this
            # worker needs no knob plumbing of its own.
            return node_queue.claim_next_download_job(
                0, host=self.host, lease_s=self.lease_s,
                host_priority=self.host_priority,
                box=self._physical_box(),
            )
        if self.queue == "gpu":
            current_model = getattr(self.model_cache, "current_model", None)
            vram_total, fits, has_models = self._gpu_capacity_for_claim()
            # Capacity gate (migration 0015): the inline lane claims a model-job
            # ONLY if this machine's VRAM can hold the model. ``fits`` is the
            # capacity-filtered known set. When the registry IS populated and VRAM
            # is KNOWN but nothing fits, this machine can run NO model — return
            # None rather than fall through to claim_next_gpu_job's empty-known
            # claim-any (which would let a too-small box grab a model it can't
            # load). A COLD worker (registry not yet populated ⇒ ``has_models``
            # False) or UNKNOWN VRAM keeps the claim-any-capable fallback so it
            # never wedges the queue.
            if has_models and vram_total is not None and not fits:
                return None
            warm_servers, warm_models = self._warm_state()
            job = node_queue.claim_next_gpu_job(
                0, current_model,
                host=self.host, lease_s=self.lease_s,
                host_priority=self.host_priority,
                known_models=self._known_models(fits),
                require_model=True,
                pool_modules=get_config().vlm_pool_node_modules,
                max_running=self._gpu_slot_budget(),
                box=self._physical_box(),
                served_types=self._served_llm_types(),
                warm_servers=warm_servers,
                warm_models=warm_models,
                kind_caps=self._kind_caps(),
                default_server=predict_default_kind(
                    warm_servers, self.host, self.queue),
                serving_context=self._serving_context(),
            )
            if job:
                self._stamp_resolved_artifact(job)
            return job
        return node_queue.claim_next_cpu_job(
            0, host=self.host, lease_s=self.lease_s,
            host_priority=self.host_priority,
            box=self._physical_box(),
            # Width fit only when the lane is widened AND capacity was reserved;
            # a serial worker passes neither, so its statement is byte-identical
            # to pre-0024 and claims any weight (it runs one job at a time).
            lane_width=self.threads if free_slots is not None else None,
            free_slots=free_slots,
        )

    def _pool_gpu_serving_ok(self) -> bool:
        """Is a GPU serving backend up for the pool lane on this box?

        The pool lane's no-model jobs POST to a per-host server. Classically that's the
        LLM (``_llm_gpu_ok``, set by the heartbeat probe). But a ComfyUI render box has
        NO LLM — its comfyui_render jobs POST to ComfyUI — so a reachable ComfyUI is an
        equally-valid backend. Open the gate when EITHER is up; else the render-only box
        would skip claiming comfyui_render forever."""
        if self._llm_gpu_ok:
            return True
        from queue_workflows import llm_probe
        from queue_workflows.config import get_config
        from queue_workflows.envcompat import env_get
        url = (env_get("QUEUE_WORKFLOWS_COMFYUI_URL") or get_config().comfyui_url or "").strip()
        return bool(url) and llm_probe.comfyui_reachable(url)

    def _pool_should_defer(self, par: int) -> bool:
        """Fill-before-spill, LLM boxes only. The consolidation heuristic
        (:func:`node_queue.vlm_pool_should_defer`) exists so light VLM load packs onto
        one LLM box. A box whose pool gate opened via ComfyUI (``_llm_gpu_ok`` False)
        serves box-placed RENDER jobs — a 'fresher LLM peer' can never claim those, so
        deferring to it starves the queue (observed: unified-memory GPU boxes idle on queued comfyui jobs
        while an LLM box's heartbeat kept them deferring). Such a box never defers."""
        if not self._llm_gpu_ok:
            return False
        try:
            return bool(node_queue.vlm_pool_should_defer(
                self.host, par, box=self._physical_box(),
            ))
        except Exception:
            log.exception(
                "[claim-worker:gpu:pool] defer check failed; claiming (spread fallback)"
            )
            return False

    def _claim_pool(self) -> dict | None:
        """Claim the next job for the POOL lane (gpu only): no-model GPU jobs
        (``require_model=False``) — VLM facade work that POSTs to a per-host
        vLLM server which batches up to PAR requests on the GPU. These load NO
        warm model in-worker, so they run in a PAR-sized concurrent pool instead
        of the concurrency-1 inline lane. Disjoint from :meth:`_claim`'s
        ``require_model=True`` set so the lanes never steal each other's rows.

        ONLY-GPU gate: no-model jobs are LLM-dispatch (they POST to this box's LLM
        server). If that server fell back to CPU (``self._llm_gpu_ok`` False, set by
        the heartbeat probe), SKIP claiming — running the job here would run it on
        CPU. The row stays queued for a GPU-backed box, and claiming resumes on its
        own once the server is back on GPU. The two other skips (insufficient VRAM,
        operator OFF) are enforced elsewhere (capacity gate / worker-control park).

        THE CLUSTER RESERVATION (``node_queue.CLUSTER_SERVER_KIND``) is the one claim a lane
        out of service for LLM work still makes. A switch blocks claims because a claim would
        reload the outgoing server, and the ONLY-GPU gate because the job would be served on
        CPU; a reservation loads nothing, is served by no broker server, and holding it ENDS
        the switch (:meth:`_begin_cluster_hold`). So both narrow the claim to reservations —
        and only when one is queued for this box, a read that fails to "none" (the old
        refusal); otherwise they refuse exactly as before."""
        from queue_workflows import model_registry
        reservations_only = self._switch.blocks_claims()
        if self._spill_paused():
            return None
        if not reservations_only and not self._pool_gpu_serving_ok():
            reservations_only = True
        # THE HOLD GATES THE CLAIM. While this lane holds its box for the cluster, every LLM
        # pool row it could claim is served by this box's own card — the card the hold just
        # gave away. Unloading after the fact is a race the hold loses: a held lane can claim
        # llm_chat a second after HOLDING, and the model is resident again for long enough
        # that llm-gateway gives up waiting for RAM at 60 s. Narrow to reservations, which need no server of ours; with one already
        # held nothing is queued for this box, so the claim below is skipped entirely.
        if not reservations_only and self._cluster_hold_reason() is not None:
            reservations_only = True
        if reservations_only and not self._reservation_queued_here():
            return None
        warm_servers, warm_models = self._warm_state()
        job = node_queue.claim_next_gpu_job(
            0, None,
            host=self.host, lease_s=self.lease_s,
            host_priority=self.host_priority,
            known_models=self._known_models(model_registry.known_ids()),
            require_model=False,
            pool_modules=get_config().vlm_pool_node_modules,
            max_running=self._gpu_slot_budget(),
            box=self._physical_box(),
            served_types=self._served_llm_types(),
            warm_servers=warm_servers,
            warm_models=warm_models,
            kind_caps=self._kind_caps(),
            default_server=predict_default_kind(
                warm_servers, self.host, self.queue),
            # 0028: a sized row is claimable only where its window is served. A box
            # that cannot state one (None) claims only unsized rows — the alternative
            # is ollama truncating the prompt and answering 200.
            serving_context=self._serving_context(),
            reservations_only=reservations_only,
        )
        return job

    def _pool_parallelism(self) -> int:
        """PAR for the VLM pool lane: the operator's per-``(host, gpu)`` LLM
        ``parallelism`` (:meth:`_pool_parallelism_configured`), CAPPED by what the vLLM
        serving here can actually hold — the concurrency its KV cache reported at its
        start (:meth:`_observe_vllm_profile`; e.g. 8 slots fed to a cache that holds
        under 3 full-window requests get the rest preempted inside the server). Only while
        vLLM serves; an ollama box is capped to 1 by :meth:`_gpu_slot_budget` as
        before.

        ...and then capped once more by the operator's PER-BOX ceiling
        (:meth:`_box_ceiling`) — see there for why this method is where it lands."""
        par = self._pool_parallelism_configured()
        seen = self.__dict__.get("_vllm_concurrency")
        if seen and worker_control.SERVER_TYPE_VLLM in (self._served_llm_types() or []):
            par = max(1, min(par, int(seen)))
        return self._box_ceiling(par)

    def _box_ceiling(self, par: int) -> int:
        """THE LAST WORD on how many GPU node-jobs this BOX runs at once: ``par``
        lowered to the operator's per-box ceiling (``gpu_box_max_concurrent`` /
        ``QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT``), or ``par`` unchanged where no
        ceiling names this box.

        WHY HERE, AND ONLY HERE. Every input to ``par`` describes the LLM SERVER —
        the operator's PAR, what the vLLM KV cache holds, what the serving kind can
        absorb. None of them describes the MACHINE: its power envelope, its unified
        memory, the thermal latch a flood of concurrent work trips. :meth:`_pool_parallelism` is the ONE function every consumer of
        "how many can this box run" already goes through — :meth:`_gpu_slot_budget`
        (→ the heartbeat's advertised ``concurrency`` and the DB-side box-slot
        arbiter's ``max_running``), :meth:`_kind_caps`, the inline ``run_once``
        admission gate, and the pool feeder's budget + executor size — so the
        ceiling binds once, here, and no call site can escape it by being written
        later. Sprinkling ``min(...)`` at those call sites is how a ceiling becomes
        advice.

        LOWERS ONLY. ``min`` both ways round: a box whose vLLM cache holds 2 still
        runs 2 under a ceiling of 3, because the ceiling is what the operator is
        willing to ALLOW, not a floor the server must reach. And the result is
        always ``>= 1`` (both operands are), so this can never park a lane."""
        ceiling = resolve_box_max_concurrent(self._physical_box())
        return par if ceiling is None else min(par, ceiling)

    def _pool_parallelism_configured(self) -> int:
        """The operator's PAR: the per-``(host, gpu)`` LLM ``parallelism`` (vLLM
        ``--max-num-seqs`` / ollama ``OLLAMA_NUM_PARALLEL``), clamped to ``>= 1``. Read
        live (cheap DB read, default-safe on a pre-0013 schema) so an operator's UI
        change to parallelism takes effect without a restart — the feeder re-reads it
        each cycle to gate submissions."""
        try:
            par = worker_control.llm_config_for(self.host, "gpu").parallelism
        except Exception:
            log.exception(
                "[claim-worker:gpu] llm_config_for parallelism read failed; "
                "defaulting pool PAR to 1",
            )
            return 1
        return max(1, int(par or 1))

    def _pool_budget(self, par: int) -> int:
        """The VLM pool's in-flight ceiling THIS cycle: ``PAR`` minus the inline
        diffusion slot when it's busy. A GPU machine's capacity is PAR TOTAL
        node-jobs; the concurrency-1 inline diffusion lane (when running) takes
        one of those slots, so the pool may keep at most ``PAR - 1`` VLM node-jobs
        in flight alongside it — keeping the machine's total ≤ PAR. Idle inline ⇒
        the full PAR. Clamped ≥ 0 (PAR=1 + a diffusion ⇒ 0 VLM slots)."""
        return max(0, par - (1 if self._inline_running else 0))

    # ── execute one ───────────────────────────────────────────────────────

    def run_once(self) -> bool:
        """Claim the next ready job and run it to a terminal state. Returns
        ``True`` if a job was claimed + executed, ``False`` when the queue had
        nothing claimable (caller should block on NOTIFY).

        GPU per-machine cap (both lanes, one budget). A GPU worker runs two
        claim lanes in ONE host process — the inline diffusion lane (here) and
        the PAR-sized VLM pool feeder. A machine's TOTAL concurrent node-jobs
        (inline + pool) must never exceed PAR (``worker_controls.llm_parallelism``).
        The pool feeder already enforces its half (``_pool_budget = PAR -
        inline_running`` — it yields a slot to a running inline job). This is the
        mirror gate the inline lane was MISSING: before claiming, it reserves its
        slot under ``_pool_lock`` gated on ``_pool_inflight < PAR``. So a machine
        already full of pool work claims NO inline job — it returns ``False`` and
        the queued job spills to an idle peer instead of stacking PAR+1 here (the
        ``2/1`` over-claim that starved an idle box). One gpu worker per host ⇒
        the in-process lock is the whole coordination surface."""
        if self.queue == "gpu":
            par = self._pool_parallelism()
            with self._pool_lock:
                # The inline lane contributes one slot; admit it only if the
                # machine's total (pool already in flight + this inline) stays
                # <= PAR. Reserve _inline_running under the SAME lock the feeder
                # reads so the two lanes can't both slip past a PAR=1 budget.
                if self._pool_inflight >= par:
                    return False
                self._inline_running = True
            try:
                job = self._claim()
                if job is None:
                    return False
                return self._run_node(job)
            finally:
                self._inline_running = False
        # CPU with a widened lane (``cpu_worker_threads > 1``): reserve a slot,
        # claim, hand the body to the executor and return immediately so the
        # caller loops back and fills the remaining slots. Same discipline as the
        # VLM feeder — the slot is taken BEFORE the claim, so a fast-returning
        # claim can never race past the ceiling.
        if self.threads > 1 and not self._is_ingest:
            return self._claim_threaded()
        # CPU / ingest: a single serial poll-claim loop, no pool lane — claim
        # then run, byte-identical to before.
        job = self._claim()
        if job is None:
            return False
        if self._is_ingest:
            return self._run_ingest(job)
        # The inline lane holds the warm-model slot for this job's duration; the
        # flag is harmless for cpu (no feeder reads it).
        self._inline_running = True
        try:
            return self._run_node(job)
        finally:
            self._inline_running = False

    # ── the threaded cpu lane ───────────────────────────────────────────────

    def _claim_threaded(self) -> bool:
        """One cycle of the widened cpu lane: reserve → claim → submit.

        Returns ``True`` when a job was claimed (the caller re-loops to fill the
        next slot) and ``False`` when the lane is full **or** the queue is empty
        — both mean "stop claiming and go idle on the NOTIFY". A full lane
        returning ``False`` is deliberate: the 1 s safety poll re-drives this the
        moment a slot frees, and NOT claiming leaves the row for an idle peer
        instead of stacking it behind N busy threads here.

        The slot is reserved under the lock BEFORE the claim and released on
        every failure path, so the in-flight count can never exceed
        :attr:`threads` and a claim that returns nothing can never leak a slot."""
        held = self._reserve_slot()
        if held is None:
            return False
        reserved, free = held
        submitted = False
        try:
            job = None
            try:
                # The claim is told exactly what is held, so a row it returns
                # always fits the reservation (migration 0024's fit test).
                job = self._claim(free_slots=free)
            except Exception:
                log.exception("[claim-worker:cpu] threaded claim failed")
            if job is None:
                return False
            job_id = job["id"]
            # Shrink the pessimistic hold to what this job actually needs. The
            # width is clamped to the lane: a job wider than the whole worker
            # occupies it entirely rather than being unclaimable forever — the
            # same clamp the claim SQL applies, so the two can never disagree.
            # Clamped at BOTH ends. The upper clamp is the lane width (a job
            # wider than the worker occupies it entirely rather than being
            # unclaimable forever); the lower clamp of 1 is what keeps the sum an
            # actual ceiling. ``or 1`` maps NULL/0 to 1 but would pass a NEGATIVE
            # straight through, and the weights are SUMMED — one out-of-band row
            # (raw SQL, a future writer that skips the API validation) would then
            # let the lane admit far more claims than it has pool threads, and
            # rows claimed past that point sit ``running`` with no LeaseRenewer
            # yet: their leases lapse, reclaim hands them to a peer, and the job
            # double-runs. Cheap to make structurally impossible here.
            weight = max(1, min(int(job.get("cpu_slots") or 1), self.threads))
            self._set_slot_weight(reserved, job_id, job.get("started_at"), weight)
            try:
                self._ensure_cpu_pool().submit(
                    self._run_node_and_release, job, reserved,
                )
                submitted = True
            except Exception:
                # Building the pool was refused (the OS would not give us a
                # thread), or it is already shut down. Either way NOTHING was
                # enqueued — that is exactly the property :class:`_CpuLanePool`
                # exists to guarantee — so the row is left to the lease reclaim
                # and the ``finally`` gives the slot back.
                log.exception("[claim-worker:cpu] submit failed for %s", job_id)
                return False
            return True
        finally:
            # ONE release rule, covering every exit: nothing claimed, the claim
            # raised (``BaseException`` included — a KeyboardInterrupt in an
            # embedded host), the submit failed. A leaked slot is not a merely
            # narrower lane: it leaves ``_cpu_drained`` cleared forever, and both
            # drain and shutdown wait on that event unbounded — a hang.
            #
            # Skipped once ``submit`` succeeded: from that instant the pool
            # worker owns this slot and frees it in its own ``finally``, so
            # releasing here too would drop the count while the body runs and
            # admit an extra claim (N+1).
            if not submitted:
                self._release_slot(reserved)

    def _reserve_slot(self) -> tuple[str, int] | None:
        """Take capacity and return ``(token, slots_held)``, or ``None`` when the
        lane is full.

        The token is a UUID and it IS the slot's identity for the slot's whole
        life — the claimed job id is recorded *against* it, never used as the
        key. A positional label (the map's size) let two concurrent reservers
        compute the SAME token, so the second insert silently overwrote the
        first, the count under-read by one, and an extra job was admitted past
        the ceiling — N+1, the thing this whole reserve-before-claim dance exists
        to prevent.

        WEIGHT (migration 0024). A job's width is on the ROW, so it cannot be
        known until the claim returns — yet capacity must be held BEFORE the
        claim or the reservation is worthless. So this reserves ALL currently
        free capacity, pessimistically, and :meth:`_claim_threaded` shrinks it to
        the row's real ``cpu_slots`` the moment it knows. The claim SQL is given
        the same number as ``free_slots``, so it can only return a row that fits
        what is held — over-commit is impossible, and the surplus is given back
        within microseconds."""
        with self._pool_lock:
            free = self.threads - self._inflight_weight_locked()
            if free < 1:
                return None
            reserved = str(uuid.uuid4())
            self._cpu_inflight[reserved] = (None, None, free)
            self._cpu_drained.clear()
            return reserved, free

    def _inflight_weight_locked(self) -> int:
        """Slots currently held. Caller holds ``_pool_lock``."""
        return sum(w for (_jid, _started, w) in self._cpu_inflight.values())

    def inflight_weight(self) -> int:
        """Slots currently held by this process (threaded lane only) — the sum of
        every in-flight job's width, not a count of jobs."""
        with self._pool_lock:
            return self._inflight_weight_locked()

    def _set_slot_weight(self, token: str, job_id: str, started_at: Any,
                         weight: int) -> None:
        """Record the claimed job against its reservation and shrink the hold to
        the job's real width, releasing the pessimistic surplus."""
        with self._pool_lock:
            if token in self._cpu_inflight:
                self._cpu_inflight[token] = (job_id, started_at, weight)

    def _cpu_lane_width(self) -> int:
        """This worker's advertised capacity when the cpu lane is widened — the
        thread count, i.e. the number of node-jobs it really can hold. Wired as
        the heartbeat's ``concurrency_fn`` only at ``threads > 1``, so a default
        worker keeps publishing the historical constant 1.

        DISPLAY ONLY. The cpu claim is ``SKIP LOCKED`` and is never gated by this
        number (unlike the gpu budget, which the fill-before-spill heuristic
        genuinely reads) — it exists so the panel reads ``CPU x/N`` instead of
        ``x/1`` while N jobs run."""
        return self.threads

    def _ensure_cpu_pool(self) -> "_CpuLanePool":
        """The cpu lane's worker pool, built on first use. Lazy so a
        ``threads=1`` worker never creates one; the width is fixed for the
        process's life (unlike the GPU pool's live-PAR resize, an operator
        changing the cpu width restarts the worker).

        Built under the lock: an unguarded check-then-assign lets concurrent
        first callers construct SEPARATE pools, and the losers are orphaned
        holding submitted work that :meth:`shutdown_threads` can never see. Only
        one thread claims today, so this is unreachable — which is exactly the
        argument for closing it while it is still cheap.

        Raises if the OS refuses a worker thread — and does so having enqueued
        NOTHING, which is what makes the caller's "not submitted ⇒ release the
        slot" rule sound. See :class:`_CpuLanePool`."""
        with self._pool_lock:
            if self._cpu_pool is None:
                pool = _CpuLanePool(width=self.threads)
                pool.start()
                self._cpu_pool = pool
            return self._cpu_pool

    def _release_slot(self, token: str) -> None:
        """Give one slot back — by its reservation TOKEN, so releasing is exactly
        as unambiguous as reserving was — and, when the lane empties, wake
        anything waiting on quiescence (drain-park, shutdown)."""
        with self._pool_lock:
            self._cpu_inflight.pop(token, None)
            if not self._cpu_inflight:
                self._cpu_drained.set()

    def _run_node_and_release(self, job: dict, token: str) -> None:
        """Pool-worker entry: run one cpu node-job, then free the slot it was
        submitted under. The release is in a ``finally`` — a stray exception here
        must never leak a permanent slot, or the lane would narrow by one for the
        rest of the process's life."""
        try:
            self._run_node(job)
        except Exception:
            log.exception(
                "[claim-worker:cpu] %s raised in threaded execution", job.get("id"),
            )
        finally:
            self._release_slot(token)

    def inflight_job_ids(self) -> set[str]:
        """The node-job ids this PROCESS currently holds (threaded lane only).
        Real claims only — a slot reserved but not yet claimed is excluded."""
        return set(self._inflight_claims())

    def _inflight_claims(self) -> dict[str, Any]:
        """``job_id -> started_at`` for the real claims this process holds
        (unclaimed reservations excluded). Collapsing on job id is right HERE —
        its only consumer is the trip sweep, which releases a row once."""
        with self._pool_lock:
            return {jid: started
                    for jid, started, _weight in self._cpu_inflight.values()
                    if jid is not None}

    def _wait_for_inflight(self, timeout: float | None = None) -> bool:
        """Block until the threaded lane is empty. ``True`` if it drained."""
        return self._cpu_drained.wait(timeout=timeout)

    def shutdown_threads(self, timeout: float | None = None) -> None:
        """Quiesce the cpu lane: stop accepting work, let the in-flight jobs
        finish, drop the executor.

        In-flight work is WAITED for, not cancelled — ``run_claim_lane.sh``
        forwards SIGTERM and relies on jobs finishing inside the compose
        ``stop_grace_period``; dropping them would strand N rows ``running``
        until their leases lapsed. ``run_forever``'s teardown passes no timeout,
        so that wait is unbounded and every job finishes.

        ``timeout`` bounds the WHOLE call, not just the first wait: when it
        lapses with work still running we detach (``wait=False``) rather than
        blocking again inside ``shutdown``, which would make the parameter
        decorative. The still-running jobs keep their daemon threads and their
        rows fall to lease reclaim, which is the normal fate of work in a process
        that is going away. Safe to call when the lane was never armed."""
        if self._cpu_pool is None:
            return
        drained = self._wait_for_inflight(timeout)
        if not drained:
            log.warning(
                "[claim-worker:cpu] shutdown timed out with %d job(s) still in "
                "flight; detaching (their rows fall to lease reclaim)",
                len(self.inflight_job_ids()),
            )
        # Swap under the SAME lock ``_ensure_cpu_pool`` builds under, so a
        # concurrent first-use can never hand work to the pool being retired.
        # The shutdown itself runs OUTSIDE the lock — it joins workers, and a
        # worker's own ``_release_slot`` takes this lock.
        with self._pool_lock:
            pool, self._cpu_pool = self._cpu_pool, None
        if pool is None:
            return
        pool.shutdown(wait=drained)

    def _trip_exit(
        self, job_id: str, *, on_exit: Callable[[int], None] | None = None,
    ) -> Callable[[int], None] | None:
        """The ``on_exit`` a watchdog gets in threaded mode — or ``None`` at
        ``threads=1``, where the stock ``os._exit`` is already right.

        A trip hard-exits the PROCESS, because a Python thread cannot be killed
        and a wedged body must actually stop. With N threads that also kills the
        N-1 healthy siblings, so they are released FIRST: back to ``queued``,
        front of the line, and with NO ``watchdog_retries`` bump — being on a box
        whose neighbour wedged is not a fault of theirs, and burning their retry
        cap for it would eventually fail perfectly good runs.

        SCOPED TO THIS PROCESS'S OWN CLAIMS — by id AND by the claim's
        ``started_at``, never host-wide. The recommended lane
        runs N processes that SHARE one ``host_label``, so
        ``requeue_running_for_worker(host, queue)`` — right for the operator OFF,
        which turns the whole lane off — would here yank a healthy sibling
        PROCESS's rows; each of those would then self-exit on its
        :class:`JobStatusWatcher` and one wedge would cascade into killing the
        whole box's lane. The ``started_at`` term closes the same hole for a
        STALE id: a row that left this process (lease lapse, corrective requeue)
        and was re-claimed by such a sibling carries a NEW claim timestamp, so
        the sweep passes over it instead of stealing it — the sibling could not
        even detect the theft, since its ``JobStatusWatcher`` compares against
        the shared label."""
        if self.threads <= 1:
            return None
        exit_fn = on_exit or os._exit

        def _sweep_then_exit(code: int) -> None:
            # BaseException everywhere below, not Exception: the courtesy release
            # must never cost us the EXIT. Leaving a wedged body alive is far
            # worse than leaving a sibling row to its lease reclaim, which
            # recovers it anyway. That covers reading the map too, not just the
            # per-sibling write — a guarantee should be structural, not a matter
            # of which line happens to be inside the try.
            try:
                claims = list(self._inflight_claims().items())
            except BaseException:
                log.exception(
                    "[claim-worker:cpu] could not read in-flight claims before "
                    "exit (lease reclaim will recover them)",
                )
                claims = []
            for sibling, started_at in claims:
                if sibling == job_id:
                    continue
                try:
                    node_queue.requeue_running_job(
                        sibling, host_label=self.host, queue=self.queue,
                        started_at=started_at,
                    )
                except BaseException:
                    log.exception(
                        "[claim-worker:cpu] could not release sibling %s before "
                        "exit (lease reclaim will recover it)", sibling,
                    )
            exit_fn(code)

        return _sweep_then_exit

    @staticmethod
    def _node_reports_progress(module_name: str) -> bool:
        """True iff the node's ``run(...)`` declares a ``status_callback`` param
        — its opt-in to no-progress policing. A node that doesn't report
        progress can't be told apart from a hung one, so the StallWatchdog is
        NOT armed for it (it's left to the wall-clock :class:`Watchdog`)."""
        try:
            mod = get_config().resolve_node_module(module_name)
            return "status_callback" in inspect.signature(mod.run).parameters
        except Exception:
            return False

    @staticmethod
    def _pool_stall_timeout_s(module_name: str) -> float | None:
        """The no-progress window a POOL-lane node opted into, or ``None``.

        Armed only when the node module declares BOTH a positive numeric
        ``POOL_STALL_TIMEOUT_S`` and a ``status_callback`` parameter — a window the
        node cannot beat would kill every job it runs. Below
        :data:`POOL_STALL_MIN_TIMEOUT_S` the window is raised to the floor;
        ``QUEUE_WORKFLOWS_DISABLE_POOL_STALL_WATCHDOG`` disarms it everywhere."""
        from queue_workflows.config import _env_flag

        if _env_flag(_POOL_STALL_DISABLE_ENV):
            return None
        try:
            mod = get_config().resolve_node_module(module_name)
            raw = getattr(mod, "POOL_STALL_TIMEOUT_S", None)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
                return None
            if "status_callback" not in inspect.signature(mod.run).parameters:
                return None
        except Exception:
            return None
        return max(float(raw), float(POOL_STALL_MIN_TIMEOUT_S))

    def _run_node(self, job: dict) -> bool:
        """Execute a claimed DAG node-job (cpu/gpu) under a cancel-watcher +
        lease-renewer + a guard set that depends on the queue:

          * CPU jobs keep the wall-clock :class:`Watchdog` (their budgets are
            fine — the user's concern is GPU renders camping forever).
          * GPU jobs get the HEALTH-driven :class:`GpuHealthWatchdog` instead of
            a wall-clock cap (video + non-video): it only kills a worker that's
            truly wedged (no GPU work AND static RAM over a 5-min window), never
            just because time passed.
          * non-video GPU jobs that report per-step progress ALSO keep the tight
            :class:`StallWatchdog` (defense in depth — it catches a fast 0%-GPU
            hang in ~2 min, well before the health watchdog's first checkpoint).
        """
        job_id = job["id"]
        # ONE-MODEL-PER-BOX gate (gpu, real node): spill if the card already holds
        # another project's model. __input__ park nodes touch no GPU — skip them.
        if self.queue == "gpu" \
                and not (job.get("node_module") or "").startswith("__input__") \
                and not self._acquire_box_or_spill(job):
            return False
        log.info(
            "[claim-worker:%s] claimed %s (node=%s model=%s)",
            self.queue, job_id, job.get("node_id"), job.get("required_model"),
        )
        _emit_node_event(
            job_id, "claimed", host_label=self.host, queue=self.queue, row=job,
        )

        # Input/await nodes carry a sentinel module name (``__input__<widget>``)
        # and execute NO node module — they park the run for user input. Mirror
        # the completed/failed OUTBOX pattern in node_executor: in one txn, mark
        # the job awaiting_input + enqueue an ``awaiting_input`` dispatch event.
        # NodePool._drain_dispatch_events (which holds the workflow loader) then
        # calls dispatcher.on_node_awaiting_input to build the input_spec.
        # Calling the dispatcher directly here would bypass the durable outbox
        # AND require the loader in every claim-worker process. Without this
        # guard the worker hands the sentinel to execute_node →
        # ``import workflows.nodes.__input__*`` → ModuleNotFoundError.
        if (job.get("node_module") or "").startswith("__input__"):
            with connection() as conn, conn.cursor() as cur:
                node_queue.mark_awaiting_input_in_txn(cur, job_id)
                node_queue.enqueue_dispatch_event_in_txn(
                    cur, job["run_id"], job["node_id"], "awaiting_input",
                )
            return True

        # A cancel-watcher feeds cooperative node bodies the run-cancel signal.
        from queue_workflows.cancel_watcher import _start_run_cancel_watcher
        cancel_event = threading.Event()
        cancel_thread = _start_run_cancel_watcher(
            job["run_id"], cancel_event, interval_s=5.0,
        )

        renewer = LeaseRenewer(
            job_id=job_id, claimed_by=self.host, lease_s=self.lease_s,
        )
        is_gpu = self.queue == "gpu"
        # Wall-clock budget: CPU jobs only. GPU jobs are policed by health, NOT
        # elapsed time — a long-but-healthy render (busy GPU / moving RAM) must
        # never be killed for running too long, so GPU gets NO fixed cap here.
        # In a widened cpu lane every hard exit takes the healthy siblings with
        # it, so both watchdogs get an ``on_exit`` that releases this process's
        # OTHER in-flight rows first. ``None`` at threads=1 ⇒ the stock os._exit.
        trip_exit = self._trip_exit(job_id)
        watchdog: Watchdog | None = None if is_gpu else Watchdog(
            job_id=job_id, budget_s=budget_for(job),
            host_label=self.host, queue=self.queue,
            **({"on_exit": trip_exit} if trip_exit else {}),
        )
        # Abandon-watcher: hard-exit if this job is re-queued / reassigned out
        # from under us (restart-resume, lease reclaim) so it's never double-run.
        status_watcher = JobStatusWatcher(
            job_id=job_id, claimed_by=self.host,
            **({"on_exit": trip_exit} if trip_exit else {}),
        )
        renewer.start()
        if watchdog is not None:
            watchdog.start()
        status_watcher.start()

        # GPU guards.
        #
        #  * GpuHealthWatchdog — the universal GPU guard (video + non-video),
        #    replacing the removed wall-clock cap. Arms AT job start with a
        #    generous load-grace first window (so a load-phase hang AND a GPU node
        #    that never beats — no required_model / no status_callback, e.g.
        #    a detector, scene_build — are both bounded), then the post-load /
        #    per-step beats collapse it to the 5-min cadence. Kills only a wedged
        #    worker (no GPU work AND static RAM over the window); a busy GPU or a
        #    > 5 GB RAM move keeps it alive forever. Video is the whole point —
        #    a hung wan/ltx render now gets caught on health instead of camping
        #    a (formerly 1800 s) budget that punished healthy long renders.
        #  * StallWatchdog — kept for NON-VIDEO GPU as defense in depth: it stays
        #    inert until the first (post-load) beat so a multi-minute cold load is
        #    never policed, then its tight 120 s no-progress window catches a fast
        #    0%-GPU hang minutes before the health watchdog's first checkpoint.
        #    EXCLUDED for video: a video backend steps slowly (minutes/beat) so
        #    120 s would false-trip a healthy render — health is its only guard.
        is_video = (job.get("required_model") or "") in get_config().video_model_ids
        reports_progress = is_gpu and self._node_reports_progress(job["node_module"])
        stall: StallWatchdog | None = None
        gpu_health: GpuHealthWatchdog | None = None
        beats: list[Callable[..., None]] = []
        if is_gpu:
            gpu_health = GpuHealthWatchdog(
                job_id=job_id, host_label=self.host, queue=self.queue,
            )
            gpu_health.start()
            beats.append(gpu_health.beat)
        if reports_progress and not is_video:
            stall = StallWatchdog(
                job_id=job_id, host_label=self.host, queue=self.queue,
            )
            stall.start()
            beats.append(stall.beat)

        # Thread ONE status_callback to the node that fans a beat out to every
        # armed GPU guard, so each reported step (and the executor's post-load
        # beat) arms + resets all of them at once. ``None`` when no guard wants
        # beats (CPU, or a GPU node that doesn't report progress) — but note the
        # GpuHealthWatchdog still needs the post-load arming beat, so a GPU job
        # always gets a callback whenever a health watchdog is armed.
        status_callback: Callable[..., None] | None = (
            (lambda *a, **k: [b() for b in beats]) if beats else None
        )
        # GPU busy-bracket: hold ``ModelCache._active`` > 0 for the job's
        # lifetime so the cache's idle reaper can't unload the warm model
        # mid-inference. ``mark_idle`` is in the finally so it ALWAYS releases.
        busy = (
            self.queue == "gpu"
            and self.model_cache is not None
            and hasattr(self.model_cache, "mark_busy")
            and hasattr(self.model_cache, "mark_idle")
        )
        if busy:
            self.model_cache.mark_busy()
        try:
            node_executor.execute_node(
                job, model_cache=self.model_cache, cancel_event=cancel_event,
                status_callback=status_callback,
            )
        finally:
            if busy:
                self.model_cache.mark_idle()
            if is_gpu:
                self._box_job_done()      # drop the box hold for THIS job (warm-hold stays)
            status_watcher.stop()
            if stall is not None:
                stall.stop()
            if gpu_health is not None:
                gpu_health.stop()
            if watchdog is not None:
                watchdog.stop()
            renewer.stop()
            cancel_event.set()
            cancel_thread.join(timeout=2.0)
        return True

    def _run_pool_node(self, job: dict) -> bool:
        """Execute a no-model GPU job (VLM — HTTP to a per-host vLLM server) in
        the PAR-sized pool lane. A DELIBERATELY lighter sibling of
        :meth:`_run_node` — it reuses the same helpers but OMITS the in-worker
        diffusion guards, because a VLM job loads no warm model and does its GPU
        work inside the vLLM server (HTTP-bound here):

          * NO ``model_cache.mark_busy/idle`` — there is no warm-model slot to
            protect from the idle reaper; these jobs carry ``required_model=None``.
          * NO :class:`GpuHealthWatchdog`, and NO physically-confirmed
            :class:`StallWatchdog` — those police an in-worker diffusion hang (idle
            GPU + static RAM in THIS container); an HTTP-bound request does its GPU
            work in the server's container, and the pool runs PAR jobs at once, so
            a single-job GPU-health verdict would be meaningless.
          * EXCEPT an opted-in no-progress guard: a node that declares
            ``POOL_STALL_TIMEOUT_S`` (and ``status_callback``) gets a
            :class:`StallWatchdog` with ``confirm_idle=False`` — its beats (one per
            streamed chunk) are the progress signal, and silence past its own
            window trips through the ordinary :func:`_watchdog_trip` policy. Without
            it a request that never answers held this lane for ever: the lease
            renewer keeps renewing while the process lives, so reclaim never fires.
            A node that did not opt in is untouched (``status_callback=None``).
            The trip hard-exits the PROCESS, so any sibling pool job dies with it
            and is recovered resume-style by the orchestrator's frozen-lease sweep —
            the same blast radius the JobStatusWatcher's exit has always had here.

        KEPT (the per-job correctness machinery, identical to the inline lane):
        the ``__input__`` park-and-return outbox early-return, the run-cancel
        watcher, a per-job :class:`LeaseRenewer`, and a :class:`JobStatusWatcher`
        so a reclaimed/reassigned row hard-exits instead of double-running. Runs
        the node through the SAME ``node_executor.execute_node`` path (no
        model_cache → the executor invokes the node directly). Returns ``True``
        once the row reached a terminal state.

        Runs on a pool worker thread; never touches the inline lane's
        ``self.model_cache``. ``os._exit`` from the JobStatusWatcher still kills
        the whole process (the safe abandon path) — acceptable: the inline
        diffusion job, if any, is re-queued by lease reclaim exactly as a
        watchdog hard-exit would do."""
        # THE CLUSTER RESERVATION (node_queue.CLUSTER_SERVER_KIND): this box is being held
        # for a server the engine does not run. When a cluster starts, classic vLLM is shut
        # down first to free RAM. The hold begins BEFORE the box
        # lease is asked for — a switch of our own holds SWITCH_SLOT and would deny it, and the
        # hold ends that switch first — and its watch stops the classic servers at once.
        hold = (self._begin_cluster_hold(job) if node_queue.is_cluster_reservation(job)
                else None)
        # ONE-MODEL-PER-BOX gate: spill this no-model LLM job if the card already
        # holds another project's model (e.g. a diffusion render). Same gate as the
        # inline lane — both funnel through _acquire_box_or_spill.
        #
        # A RESERVATION WAITS INSTEAD OF SPILLING: the occupant is usually the LLM model
        # its own hold is unloading, and a spill hands the box straight back.
        if hold is not None:
            if not self._acquire_box_for_reservation(job, hold):
                # Only now, after the ceiling: give the box back and re-queue, still pinned
                # here (a reservation queued for this box still refuses every start meanwhile).
                self._end_cluster_hold(hold)
                try:
                    node_queue.requeue_job_for_retry(job["id"])
                except Exception:
                    log.exception("[cluster-hold] %s: re-queue failed for %s",
                                  self.host, job.get("id"))
                return False
        elif not self._acquire_box_or_spill(job):
            return False
        if hold is not None:
            self._start_cluster_hold_watch(hold)
        job_id = job["id"]
        log.info(
            "[claim-worker:gpu:pool] claimed %s (node=%s model=%s)",
            job_id, job.get("node_id"), job.get("required_model"),
        )
        _emit_node_event(
            job_id, "claimed", host_label=self.host, queue=self.queue, row=job,
        )

        # Input/await nodes: mark awaiting_input + enqueue the dispatch event in
        # one txn (the durable outbox), exactly as the inline lane — never hand a
        # ``__input__`` sentinel to execute_node. (A VLM lane shouldn't normally
        # see one, but the guard keeps the two lanes' contract identical.)
        if (job.get("node_module") or "").startswith("__input__"):
            self._box_job_done()          # balance the acquire above (park touches no GPU)
            if hold is not None:
                self._end_cluster_hold(hold)
            with connection() as conn, conn.cursor() as cur:
                node_queue.mark_awaiting_input_in_txn(cur, job_id)
                node_queue.enqueue_dispatch_event_in_txn(
                    cur, job["run_id"], job["node_id"], "awaiting_input",
                )
            return True

        from queue_workflows.cancel_watcher import _start_run_cancel_watcher
        cancel_event = threading.Event()
        cancel_thread = _start_run_cancel_watcher(
            job["run_id"], cancel_event, interval_s=5.0,
        )
        # Visible to the lane, not just to this thread: a cluster hold starting on another pool
        # thread has to be able to tell this job to stop calling the card's server.
        self._register_pool_job(job, cancel_event)
        renewer = LeaseRenewer(
            job_id=job_id, claimed_by=self.host,
            # A RESERVATION HOLDS ITS BOX ON A SHORT LEASE — see CLUSTER_RESERVATION_LEASE_S.
            # The lease is how long the fleet keeps believing a lane that stopped renewing,
            # and for a reservation that belief is "this box is held", which llm-gateway acts on.
            lease_s=(CLUSTER_RESERVATION_LEASE_S if hold is not None else self.lease_s),
        )
        status_watcher = JobStatusWatcher(job_id=job_id, claimed_by=self.host)
        renewer.start()
        status_watcher.start()
        stall_s = self._pool_stall_timeout_s(job.get("node_module") or "")
        stall: StallWatchdog | None = None
        if stall_s is not None:
            # Inert until the node's FIRST beat — the first streamed token — so a cold
            # server-side model load and the prompt prefill are never timed.
            stall = StallWatchdog(
                job_id=job_id, stall_timeout_s=stall_s, confirm_idle=False,
                host_label=self.host, queue=self.queue,
            )
            stall.start()
            log.info(
                "[claim-worker:gpu:pool] %s: no-progress guard armed (%.0fs, opted in "
                "by %s.POOL_STALL_TIMEOUT_S)", job_id, stall_s, job.get("node_module"),
            )
        try:
            # No model_cache (no warm model); the status_callback is the opted-in
            # guard's beat, or None — the executor invokes the node directly.
            node_executor.execute_node(
                job, model_cache=None, cancel_event=cancel_event,
                status_callback=stall.beat if stall is not None else None,
            )
        finally:
            self._unregister_pool_job(job)
            self._box_job_done()          # drop this pool job's box hold
            if hold is not None:
                # The reservation's job ended (the LB cancelled its run, or it failed):
                # normal service resumes.
                self._end_cluster_hold(hold)
            if stall is not None:
                stall.stop()
            status_watcher.stop()
            renewer.stop()
            cancel_event.set()
            cancel_thread.join(timeout=2.0)
        return True

    # ── GPU VLM pool lane (feeder + executor pool) ──────────────────────────

    def _pool_run_and_release(self, job: dict) -> None:
        """Pool-worker entry: run one no-model GPU job then release its in-flight
        slot. Always decrements the in-flight counter (even on a node exception
        — :meth:`_run_pool_node` already finalises the row through the executor's
        outbox; a stray error here must NOT leak a permanent slot or the feeder
        would stop claiming)."""
        try:
            self._run_pool_node(job)
        except Exception:
            log.exception(
                "[claim-worker:gpu:pool] %s raised in pool execution",
                job.get("id"),
            )
        finally:
            with self._pool_lock:
                self._pool_inflight -= 1

    def _ensure_pool(self, par: int) -> ThreadPoolExecutor:
        """Return the pool executor sized for ``par``, recreating it if PAR grew
        since last cycle (an operator raised ``--max-num-seqs`` in the UI). A
        shrink keeps the larger pool but the feeder GATES submissions on the
        current PAR, so excess threads simply sit idle — cheaper than tearing a
        live pool down mid-flight. Caller is the single feeder thread, so no lock
        is needed around the executor handle itself."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=par, thread_name_prefix="gpu-vlm-pool",
            )
            self._pool_max = par
        elif par > self._pool_max:
            # Grow: drain-free swap — the old pool's in-flight jobs keep running
            # on their threads; new submissions go to the larger pool.
            old = self._pool
            self._pool = ThreadPoolExecutor(
                max_workers=par, thread_name_prefix="gpu-vlm-pool",
            )
            self._pool_max = par
            old.shutdown(wait=False)
        return self._pool

    def _pool_feeder_loop(self) -> None:
        """The NEW pool lane's feeder (gpu only): keep up to PAR no-model VLM
        jobs in flight, claiming them via :meth:`_claim_pool` and running each on
        the :class:`ThreadPoolExecutor`. Mirrors the inline loop's
        LISTEN(``node_job_ready``)+safety-poll shape on its OWN autocommit
        connection (never shares the inline loop's), so the two lanes wake
        independently.

        Per cycle:

          1. re-read PAR live (cheap, default-safe) so a UI change takes effect;
          2. while in-flight < PAR, claim a no-model job and ``submit`` it
             (in-flight bumped BEFORE submit so a fast-returning claim can't race
             past PAR — the hard no-over-claim guarantee);
          3. when the queue is drained (claim returns None) or PAR is full,
             block on the wake NOTIFY with the safety-poll timeout, then re-loop.

        Stops cleanly when ``self._stop`` is set (the run_forever finally also
        shuts the pool down).

        PG-reconnect: the LISTEN connection is opened via
        :func:`db.listen_with_reconnect`, so a PG bounce no longer kills the
        feeder loop — the helper reopens the connection with exponential
        backoff and re-runs the body."""
        def _body(listen_conn):
            while not self._stop.is_set():
                par = self._pool_parallelism()
                pool = self._ensure_pool(par)
                claimed_any = False
                while not self._stop.is_set():
                    # Drain (stop_policy="drain"): the pool lane must also stop
                    # claiming; in-flight pool jobs finish on their own threads.
                    # The inline loop owns the actual park-until-ON.
                    if self.drain_requested:
                        break
                    # FILL-BEFORE-SPILL gate (additive): if a fresh
                    # higher-ranked vLLM/ollama peer still has free VLM
                    # capacity, DEFER this cycle — let that machine fill its
                    # PAR slots first so light load consolidates onto one box
                    # (others stay free for diffusion / idle-unload). We do
                    # NOT claim-then-release: just don't claim. Failure-safe
                    # — a query blip falls through to today's spread behaviour
                    # (claim now), never crashes the feeder. The single-box /
                    # no-fresh-higher-peer case returns False ⇒ byte-identical
                    # to today. Checked BEFORE reserving a slot so a deferral
                    # leaves the in-flight counter untouched.
                    # Rank SELF by the same capped, published number peers see —
                    # ranking by raw PAR while advertising the capped budget made an
                    # ollama box compare itself at 3 against peers' 1.
                    if self._pool_should_defer(self._gpu_slot_budget()):
                        break
                    with self._pool_lock:
                        # Budget = PAR minus the inline diffusion slot when
                        # busy, so the machine's TOTAL node-jobs (inline + pool)
                        # stays ≤ PAR (the diffusion takes one of the PAR slots).
                        if self._pool_inflight >= self._pool_budget(par):
                            break
                        # Reserve the slot BEFORE the claim/submit so a
                        # concurrent decrement can't let us exceed the budget.
                        self._pool_inflight += 1
                    try:
                        job = self._claim_pool()
                    except Exception:
                        log.exception(
                            "[claim-worker:gpu:pool] claim failed"
                        )
                        job = None
                    if job is None:
                        # Nothing to run — release the reserved slot.
                        with self._pool_lock:
                            self._pool_inflight -= 1
                        break
                    claimed_any = True
                    try:
                        pool.submit(self._pool_run_and_release, job)
                    except Exception:
                        # Submit failed (pool shutting down) — release the
                        # slot and finalise the row's fate to lease reclaim.
                        log.exception(
                            "[claim-worker:gpu:pool] submit failed for %s",
                            job.get("id"),
                        )
                        with self._pool_lock:
                            self._pool_inflight -= 1
                        break
                if self._stop.is_set():
                    break
                # Drained or PAR-full: block on the wake NOTIFY (short
                # safety-poll timeout) before the next cycle. A claim this
                # cycle ⇒ loop straight back to drain greedily.
                if not claimed_any:
                    for _ in listen_conn.notifies(
                        timeout=NOTIFY_POLL_TIMEOUT_S, stop_after=1,
                    ):
                        break

        try:
            from queue_workflows.db import listen_with_reconnect
            listen_with_reconnect(self._wake_channel, self._stop, _body)
        except Exception:
            log.exception(
                "[claim-worker:gpu:pool] feeder loop crashed (pool lane down; "
                "inline diffusion lane unaffected)"
            )

    def _start_pool_lane(self) -> None:
        """Arm the GPU VLM pool lane (gpu only): create the PAR-sized executor +
        start the feeder thread. Idempotent-safe (only called once from
        run_forever after the park gate)."""
        if self.queue != "gpu":
            return
        self._ensure_pool(self._pool_parallelism())
        self._pool_feeder = threading.Thread(
            target=self._pool_feeder_loop, daemon=True,
            name="gpu-vlm-feeder",
        )
        self._pool_feeder.start()

    def _stop_pool_lane(self) -> None:
        """Tear the GPU VLM pool lane down: join the feeder, shut the executor
        down (cancel queued, let in-flight drain briefly). Safe to call when the
        lane was never armed (cpu/ingest, or stopped while parked)."""
        if self._pool_feeder is not None:
            self._pool_feeder.join(timeout=2.0)
            self._pool_feeder = None
        if self._pool is not None:
            # cancel_futures drops not-yet-started submissions; in-flight VLM
            # requests finish on their own threads (they're HTTP-bound + short).
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _run_ingest(self, job: dict) -> bool:
        """Execute a claimed ingest job (fetch/load) under a lease-renewer +
        watchdog. No cancel-watcher (no parent run) and no model cache."""
        from queue_workflows import ingest_executor
        job_id = job["id"]
        log.info(
            "[claim-worker:%s] claimed ingest %s (task=%s reason=%s)",
            self.queue, job_id, job.get("task_name"), job.get("reason"),
        )
        renewer = LeaseRenewer(
            job_id=job_id, claimed_by=self.host, lease_s=self.lease_s,
            table="ingest_jobs",
        )
        watchdog = Watchdog(
            job_id=job_id, budget_s=budget_for(job), table="ingest_jobs",
            host_label=self.host, queue=self.queue,
        )
        renewer.start()
        watchdog.start()
        try:
            ingest_executor.execute_ingest_job(job)
        finally:
            watchdog.stop()
            renewer.stop()
        return True

    # ── operator control (ON/OFF) ────────────────────────────────────────────

    def _physical_box(self) -> str:
        """This worker's PHYSICAL box name for placement (migration 0020) — the
        cross-project identity ``QUEUE_WORKFLOWS_GPU_BOX_ID`` / ``config.gpu_box_id``
        (else the machine hostname), the SAME value the box lease keys on and NOT
        the per-project ``host_label``. Cached (it can't change within a process)."""
        if self._box_name is None:
            from queue_workflows.gpu_model_lease import default_box_id
            self._box_name = default_box_id()
        return self._box_name

    def _reclaim_predecessor_zombies(self) -> None:
        """Boot-time zombie kill: re-queue any ``running`` row still claimed by THIS
        worker's identity whose heartbeat is stale/absent — i.e. my dead
        predecessor's in-flight job after a restart/recreate.

        Without this, a recreated worker sat idle next to its predecessor's zombie
        for up to the whole 600 s lease (the box's ONE gpu slot camped by a corpse).
        The staleness predicate makes it multi-process-lane safe: a live sibling
        sharing this label keeps the heartbeat fresh, so its job is never yanked
        (see :func:`node_queue.requeue_running_if_worker_stale`). Runs BEFORE the
        heartbeat starts — our own first beat would otherwise mark the label fresh
        and hide the predecessor's corpse. Best-effort: recovery must never block
        worker startup (the lease-reclaim remains the backstop)."""
        try:
            n = node_queue.requeue_running_if_worker_stale(self.host, self.queue)
            if n:
                log.warning(
                    "[claim-worker:%s] boot: re-queued %d zombie job(s) my dead "
                    "predecessor (host=%s) still held — freed the slot(s) "
                    "immediately instead of waiting out the lease",
                    self.queue, n, self.host,
                )
        except Exception:
            log.exception(
                "[claim-worker:%s] boot zombie-reclaim failed (lease-reclaim "
                "remains the backstop)", self.queue,
            )

    def requeue_inflight_for_control(self) -> int:
        """Re-queue any job this worker is currently running back to the queue and
        clear its GPU busy-ghost — the DB cleanup a control HARD stop does just
        before the process exits.

        ``os._exit`` skips ``_run_node``'s ``finally`` (exactly like a watchdog
        trip), so this mirrors the trip path's pre-exit bookkeeping: release the
        in-flight row (resume-style, NO ``watchdog_retries`` increment — an operator
        turning a machine off is redistribution, not a node failure) so it
        redistributes at once, and null this worker's ``current_model`` busy-ghost
        so the GPU-busy gauge drops it immediately. Returns the rows re-queued."""
        n = node_queue.requeue_running_for_worker(self.host, self.queue)
        _clear_busy_ghost(self.host, self.queue)
        return n

    # ── the loop ───────────────────────────────────────────────────────────

    @property
    def _wake_channel(self) -> str:
        """The LISTEN channel this worker wakes on. Ingest workers (fetch/load)
        wake on ``ingest_job_ready`` (migration 0007); node workers (cpu/gpu)
        wake on ``node_job_ready`` (migration 0006). Both NOTIFYs carry the
        queue name as payload."""
        return "ingest_job_ready" if self._is_ingest else "node_job_ready"

    @property
    def _min_schema_version(self) -> int:
        """The migration this queue's claim path depends on: ``download`` needs
        the 0025 CHECK drop (its rows can't exist before it), ``gpu`` the 0028 window
        columns its claim filters on, ingest 0008 (host-defined queue names +
        ``args``), cpu the 0006 lease columns."""
        if self.queue == "download":
            return _DOWNLOAD_REQUIRED_VERSION
        if self.queue == "gpu":
            return _GPU_REQUIRED_VERSION
        return (
            _NODE_REQUIRED_VERSION if self.queue in _NODE_QUEUES
            else _INGEST_REQUIRED_VERSION
        )

    def await_schema(self) -> None:
        """Block until the migrations this queue's claim loop depends on are
        applied. The orchestrator owns the migration run (``db.bootstrap``,
        serialized by an advisory lock on Postgres); a claim worker WAITS for the schema
        before it starts polling."""
        from queue_workflows import db
        # WAIT FOR THE DATABASE FIRST. `wait_for_schema` needs a connection, so an
        # unreachable DB used to surface here as a 15 s pool timeout and a process
        # exit — the crash-loop that ran hundreds of restarts deep against a Postgres
        # whose docker network endpoint had gone, while nothing durable recorded it.
        # Waiting in process holds no lease and claims nothing; a MISCONFIGURED DSN
        # still raises (a restart cannot fix a wrong password) and `main` exits 2.
        if not db.wait_for_db(stop_event=getattr(self, "_stop", None)):
            # Asked to stop (SIGTERM), or QUEUE_WORKFLOWS_DB_WAIT_MAX_S elapsed and the
            # deployment wants exit-and-restart. Either way the DB is still unreachable,
            # so falling through to wait_for_schema would just re-enter the pool timeout
            # and surface as a traceback instead of the clean stop that was asked for.
            raise db.DbUnreachable(
                "gave up waiting for the database (stop requested or "
                "QUEUE_WORKFLOWS_DB_WAIT_MAX_S elapsed)")
        min_version = self._min_schema_version
        log.info(
            "[claim-worker:%s] waiting for queue_schema_version >= %d",
            self.queue, min_version,
        )
        db.wait_for_schema(min_version)

    def _park_until_enabled(self) -> bool:
        """Boot-time gate: while this worker's control row is OFF, do NOT claim and
        do NOT advertise capacity — sit idle until an operator turns it back ON.

        Entered from :meth:`run_forever` before the claim loop; returns when the
        control state is ON (or the worker is stopped). The WORKER's own RAM is
        free here (a freshly (re)started process holds no model) but the BOX's is
        not: sidecar LLM servers outlive the worker, and a
        ``restart: unless-stopped`` sidecar even resurrects itself mid-park
        (a parked box's vllm can crash-loop and re-load its weights hours
        later with nothing able to stop it). So PARK RELEASES THE CARD: a parked gpu worker ticks a
        :class:`~queue_workflows.llm_idle_unload.ParkQuiesce` every wait
        iteration — an idle vLLM is stopped (a real ``docker stop``, which the
        restart policy respects), idle ollama models are unloaded, and a
        resurrected server is stopped again — while a server a PEER project is
        actively using is left alone (the guards are activity-based). A parked
        worker does NOT heartbeat, so it ages out of the fresh-heartbeat window
        within ~30 s and an OFF machine correctly shows zero capacity. LISTENs
        ``worker_control`` for an instant wake on the ON flip, with the same
        safety-poll timeout the watcher uses.

        This is the ONLY place a worker parks: a RUNNING worker that's turned OFF
        hard-exits (the WorkerControlWatcher) and the supervisor restarts it back
        through this gate — there is no in-process running→parked transition.

        Returns ``True`` if the worker should proceed to its claim loop (control
        state is ON — never parked, or parked then turned back ON); ``False`` if it
        was stopped while parked (e.g. SIGTERM during park), so run_forever exits
        without starting the heartbeat / claim loop."""
        import psycopg

        if (
            worker_control.desired_state_for(self.host, self.queue)
            != worker_control.STATE_OFF
        ):
            return True
        log.warning(
            "[claim-worker:%s] control state OFF for host=%s — PARKED (not "
            "claiming, not advertising capacity; releasing the box's idle LLM "
            "RAM) until turned back ON",
            self.queue, self.host,
        )
        poll_s = worker_control._worker_control_poll_s()
        try:
            # Guarded like the arm sites: releasing the card is best-effort and
            # must never take down the park gate itself (an
            # ImportError here would crash a PARKING worker).
            quiesce_tick = self._build_park_quiesce()
        except Exception:
            log.exception("[claim-worker:%s] park quiesce arm failed (parking "
                          "without card release)", self.queue)
            quiesce_tick = None
        # PG-reconnect: park loop is long-lived (can be parked for hours), so a
        # PG bounce mid-park must NOT exit the worker — the helper reconnects
        # and the body resumes waiting for the ON flip.
        from queue_workflows.db import listen_with_reconnect
        result = {"on": False}
        def _body(listen_conn):
            while not self._stop.is_set():
                if (
                    worker_control.desired_state_for(self.host, self.queue)
                    != worker_control.STATE_OFF
                ):
                    log.info(
                        "[claim-worker:%s] control state ON for host=%s — resuming",
                        self.queue, self.host,
                    )
                    result["on"] = True
                    return
                if quiesce_tick is not None:
                    quiesce_tick()
                for _ in listen_conn.notifies(timeout=poll_s, stop_after=1):
                    break
        listen_with_reconnect(worker_control.NOTIFY_CHANNEL, self._stop, _body)
        return result["on"]

    def _build_park_quiesce(self):
        """The parked gpu worker's card-release tick, or ``None`` when this worker
        must not touch the box's LLM servers (non-gpu lane, or the shared reaper
        kill-switch). Wiring only — every probe/lever resolves lazily at TICK
        time, so building the gate costs nothing and follows topology edits."""
        from queue_workflows import llm_idle_unload

        if self.queue != "gpu" or env_get(llm_idle_unload._DISABLE_ENV):
            return None

        quiesce = llm_idle_unload.ParkQuiesce(
            **self._llm_quiesce_levers(state="parked"),
            voice="park",
            label=self.host,
            # A drain-park KEEPS the job it already had (that is what drain means), so
            # the watchdog must not evict the model that job is streaming from — see
            # ParkQuiesce's "THE ONE EXEMPTION". Only OUR work exempts the box; a peer's
            # remote traffic still gets nothing.
            #
            # The POOL lane only, deliberately, NOT `_gpu_active_count`: that also
            # counts the inline diffusion lane, and a render does not touch the LLM at
            # all. Exempting on one would let a peer's traffic keep a parked box's vLLM
            # busy — the precise failure the no-remote-exemption rule was written to end.
            in_flight_fn=lambda: self._pool_inflight,
        )
        last_sweep = [0.0]

        def tick() -> None:
            # A parked box is the one most likely to be borrowed for something else,
            # and still the box that serves the broker's models the moment it is
            # enabled: the rogue guard runs here too, at the lease tick's 10 s.
            now = time.monotonic()
            if now - last_sweep[0] >= 10.0:
                last_sweep[0] = now
                from queue_workflows import rogue_llm

                rogue_llm.sweep(host=self.host)
            # A PARK ENDS A SWITCH. Lanes WARMING a vLLM when they are parked: the watchdog stopped the server once it answered
            # /metrics, and 20 s later the switch — still WARMING, re-issuing its start every
            # 30 s from the heartbeat thread a drain-park keeps — started the same container
            # again. The switch ends here, with the
            # server it was loading. A lane that restarted parked resumed its persisted
            # switch from disk; that one ends here too, before any heartbeat could drive it.
            self._abandon_switch("the box is PARKED — a parked box serves nobody")
            quiesce.tick()

        return tick

    def _llm_quiesce_levers(self, *, state: str) -> dict:
        """The probes and levers a :class:`~queue_workflows.llm_idle_unload.ParkQuiesce` works
        this box's broker LLM servers with — for the park gate (``state='parked'``) and for a
        cluster reservation's hold (``state='held'``). vLLM is judged by its CONTAINER'S state
        as well as by ``/metrics``: a vLLM still loading answers no probe. Every probe and
        lever resolves at TICK time."""
        from queue_workflows import llm_probe
        from queue_workflows.llm_backends import factory as _llm_factory

        def _stop_vllm() -> None:
            stop = get_config().vllm_stop_fn
            if stop is None:
                log.warning(
                    "[llm-idle] %s: %s with an idle vLLM but no stop hook is "
                    "wired — set_vllm_lifecycle(start_fn, stop_fn) so a %s "
                    "box can release its VRAM", self.host, state, state,
                )
                return
            stop()

        return dict(
            vllm_activity_fn=lambda: llm_probe.vllm_activity(
                _llm_factory.resolve_base_url("vllm")),
            vllm_stop_fn=_stop_vllm,
            vllm_running_fn=self._vllm_container_running,
            ollama_models_fn=lambda: llm_probe.loaded_models_info(
                _llm_factory.resolve_base_url()),
            ollama_unload_fn=lambda models: llm_probe.unload_ollama_models(
                _llm_factory.resolve_base_url(), models),
        )

    def _vllm_container_running(self) -> bool:
        """Is this box's broker ``vllm`` container running — serving OR still loading? Read
        fresh from docker every time: the describe cache holds 20 s, and a watchdog that kills
        on sight cannot wait that long. ``False`` when docker cannot say (no socket, no such
        container), so a box without the lever is judged by ``/metrics`` alone."""
        from queue_workflows import llm_context_lever

        return bool(llm_context_lever.describe(worker_control.SERVER_TYPE_VLLM,
                                               ttl_s=0.0).running)

    def _abandon_switch(self, why: str, *, stop_loading: bool = True) -> bool:
        """End the server switch in flight because this box must start nothing (``why``).
        Returns True when there was one.

        With it go the box lease the switch holds for its whole duration (every peer is
        denied while it holds ``SWITCH_SLOT`` — and so would be our own reservation's
        acquire) and, unless ``stop_loading`` is False, the vLLM it started and is still
        loading: the one thing a switch leaves behind that takes memory, and the server this
        box was just told not to run."""
        from queue_workflows import server_switch as _sw

        switch = getattr(self, "_switch", None)
        if switch is None or switch.state == _sw.IDLE:
            return False
        target, state = switch.target, switch.state
        switch.abandon(why)
        self.__dict__.pop("_vllm_decision", None)
        self.__dict__.pop("_warm_started_at", None)
        try:
            lease = self._get_box_lease()
            if lease is not None:
                lease.release()
        except Exception:
            log.debug("[server-switch] %s: box-lease release failed (TTL will expire it)",
                      self.host, exc_info=True)
        if (stop_loading and target == worker_control.SERVER_TYPE_VLLM
                and state in (_sw.SWITCHING, _sw.WARMING)):
            try:
                if self._vllm_container_running():
                    stop = get_config().vllm_stop_fn
                    if stop is None:
                        log.error("[server-switch] %s: the abandoned switch left a vLLM "
                                  "loading and no stop lever is wired "
                                  "(set_vllm_lifecycle) — %s", self.host, why)
                    else:
                        stop()
                        log.warning("[server-switch] %s: STOPPED the vLLM the abandoned "
                                    "switch was loading — %s", self.host, why)
            except Exception:
                log.exception("[server-switch] %s: could not stop the vLLM the abandoned "
                              "switch was loading", self.host)
        return True

    def _cluster_hold_reason(self) -> str | None:
        """Why this lane is HOLDING its box for a cluster reservation right now, or ``None``.

        THE CARD IS THE SCOPE. A hold means the whole card belongs to a server the engine
        does not run, so it governs both things this lane could do with that card: START a
        server on it (:meth:`_llm_starts_forbidden`) and CLAIM work that would occupy it
        (:meth:`_claim_pool`). The holds are this process's own, so this is an in-memory
        read — no DB, safe to ask on the claim path."""
        holds = self.__dict__.get("_cluster_holds") or {}
        if not holds:
            return None
        return f"the box is HELD for cluster reservation {next(iter(holds))[:8]}"

    def _pool_jobs_lock(self) -> threading.Lock:
        lock = self.__dict__.get("_pool_jobs_lock_obj")
        if lock is None:
            lock = self.__dict__.setdefault("_pool_jobs_lock_obj", threading.Lock())
        return lock

    def _register_pool_job(self, job: dict, cancel_event: threading.Event) -> None:
        """Record a pool job in flight on this lane together with its COOPERATIVE cancel seam —
        the same ``threading.Event`` the run-cancel watcher feeds.

        WHY THE LANE NEEDS A REGISTRY. ``_pool_inflight`` is a count, and a count cannot be
        asked "which of you is talking to the card?". A cluster hold has to answer exactly
        that: the reservation's own body uses no LLM, but its siblings on the pool lane do, and
        they are what reload the weights the hold unloads."""
        with self._pool_jobs_lock():
            self.__dict__.setdefault("_pool_jobs", {})[str(job.get("id") or "")] = (
                job, cancel_event)

    def _unregister_pool_job(self, job: dict) -> None:
        """Drop a pool job from the registry — called on EVERY exit path of its runner."""
        with self._pool_jobs_lock():
            (self.__dict__.get("_pool_jobs") or {}).pop(str(job.get("id") or ""), None)

    def _cancel_inflight_llm_pool_jobs(self, why: str, *, keep: str = "") -> list[str]:
        """Signal cancel to every LLM-serving pool job in flight on this lane, and say so.

        THE REASON: a row already ``cancelled`` can still have its process issue a fresh
        ``/v1/chat/completions`` tens of seconds later, running for minutes — reloading the
        weights the hold unloads every 5 s. Two things
        were slow: the run-cancel watcher only polls every 5 s, and nothing told it at all
        (the run was not cancelled — the BOX was taken). Signalling here is immediate.

        COOPERATIVE, and only that. The event is the contract the engine owns; whether a node
        checks it between its own LLM calls is the node's half (see ``execute_node``'s
        ``cancel_event`` kwarg). Work that does not serve the LLM is never signalled — a render
        does not compete for the card's weights — and ``keep`` spares the reservation's own
        body, which IS the hold."""
        jobs = dict(self.__dict__.get("_pool_jobs") or {})
        cancelled: list[str] = []
        for job_id, (row, event) in jobs.items():
            if job_id == keep or not self._job_serves_llm(row):
                continue
            try:
                event.set()
            except Exception:                       # pragma: no cover - a set() that raises
                log.debug("[cluster-hold] %s: could not signal %s", self.host, job_id,
                          exc_info=True)
                continue
            cancelled.append(job_id)
        if cancelled:
            log.warning(
                "[cluster-hold] %s: CANCELLING %d LLM job(s) still in flight on this lane "
                "(%s) — %s. A job that keeps calling this box's server reloads the weights "
                "the hold just unloaded, at the request boundary, every time.",
                self.host, len(cancelled), ", ".join(j[:8] for j in cancelled), why)
        return cancelled

    def _pool_jobs_beside_the_reservation(self) -> int:
        """Pool jobs RUNNING on this lane other than a reservation's own body — read off the
        registry of jobs actually in flight, never off ``_pool_inflight``.

        The reservation runs ON the pool lane, so it must never count as work draining off its
        own card. And the slot counter is not a count of jobs: the feeder reserves a slot BEFORE
        it asks for a job, and a held box keeps asking (the hold refuses, the slot goes back) —
        so every hold tick would count a job that did not exist and skip the eviction,
        indefinitely."""
        holds = set(self.__dict__.get("_cluster_holds") or {})
        with self._pool_jobs_lock():
            return sum(1 for job_id in (self.__dict__.get("_pool_jobs") or {}) if job_id not in holds)

    def _llm_starts_forbidden(self) -> str | None:
        """Why this box must START no broker LLM server right now, or ``None``.

        One rule, asked by every start path — a switch request, the switch's own tick and
        START, the idle stopper's demand start, the boot-time server start:

          * a CLUSTER RESERVATION is held on this lane: the box, and its memory, belong to a
            server the engine does not run;
          * one is QUEUED for this box, or still RUNNING under this lane's name (a lane that
            restarted before re-claiming it): the box is about to be given up, and minutes of
            weight loading would take the memory the cluster needs;
          * the operator has PARKED this lane: a parked box serves nobody.

        The holds are this process's own; the rest are reads that fail OPEN, as the idle
        stopper's park gate always has — a control-plane blip must not wedge every start on
        the fleet."""
        held = self._cluster_hold_reason()
        if held is not None:
            return held
        try:
            if (worker_control.desired_state_for(self.host, self.queue)
                    == worker_control.STATE_OFF):
                return "the box is PARKED"
        except Exception:
            log.debug("[claim-worker:gpu] control state unreadable — treated as ON",
                      exc_info=True)
        row = self._reservation_for_this_box()
        if row:
            return (f"cluster reservation {str(row.get('id') or '?')[:8]} is "
                    f"{row.get('status') or '?'} for this box")
        return None

    def _reservation_for_this_box(self) -> dict | None:
        """:func:`node_queue.cluster_reservation_for` for this lane and box; ``None`` on any
        failure (a worker that cannot name its box reserves nothing)."""
        try:
            return node_queue.cluster_reservation_for(
                host=self.host, box=self._physical_box(), queue=self.queue)
        except Exception:
            log.debug("[claim-worker:gpu] cluster-reservation read failed — none assumed",
                      exc_info=True)
            return None

    def _reservation_queued_here(self) -> bool:
        """Is a cluster reservation QUEUED for this box — the one claim a lane out of service
        for LLM work still makes (:meth:`_claim_pool`)? Any failure is "no": the old refusal."""
        row = self._reservation_for_this_box()
        return bool(row) and str(row.get("status") or "") == "queued"

    #: How often a held box re-checks that the broker's own servers are still down: the park
    #: gate's cadence. The first check runs the moment the hold begins.
    CLUSTER_HOLD_POLL_S = 5.0

    def _cluster_hold_lock(self) -> threading.Lock:
        lock = self.__dict__.get("_cluster_hold_lock_obj")
        if lock is None:
            lock = self.__dict__.setdefault("_cluster_hold_lock_obj", threading.Lock())
        return lock

    def _begin_cluster_hold(self, job: dict) -> _ClusterHold:
        """THE CLUSTER RESERVATION: this lane has claimed a row that holds the box for a
        server the engine does not run (``node_queue.CLUSTER_SERVER_KIND``). Register the hold
        — from here every start path refuses (:meth:`_llm_starts_forbidden`) — and end any
        switch in flight NOW, releasing the box lease it holds so the reservation's own
        acquire is not denied by it. :meth:`_start_cluster_hold_watch` then stops whatever
        runs, at once."""
        from queue_workflows import llm_idle_unload

        job_id = str(job.get("id") or "")
        inputs = job.get("inputs")
        if isinstance(inputs, str):
            import json

            try:
                inputs = json.loads(inputs)
            except ValueError:
                inputs = None
        purpose = str(inputs.get("purpose") or "") if isinstance(inputs, dict) else ""
        disabled = env_get(llm_idle_unload._DISABLE_ENV)
        # THE WORK ALREADY IN FLIGHT DRAINS, UNDER A CEILING. The reservation's own body uses
        # no LLM, but the pool lane may have claimed an llm_chat a moment before the hold
        # began and that job is streaming
        # from this card. Evicting under it does not free the card anyway — ollama applies
        # keep_alive:0 at the REQUEST BOUNDARY and the job's next call cold-reloads the same
        # weights (ParkQuiesce's request-boundary lesson). So: let it finish, say so on
        # every tick, and take the card regardless after CLUSTER_HOLD_DRAIN_MAX_S, because a
        # request that never ends must not starve the cluster. The drain is guaranteed to
        # terminate: no new work can arrive while the hold gates the claim (:meth:`_claim_pool`).
        quiesce = None if disabled else llm_idle_unload.ParkQuiesce(
            **self._llm_quiesce_levers(state="held"), voice="cluster-hold", label=self.host,
            in_flight_fn=self._pool_jobs_beside_the_reservation,
            hold_off_max_s=CLUSTER_HOLD_DRAIN_MAX_S,
            now_fn=self._spill_clock)
        hold = _ClusterHold(job_id=job_id, purpose=purpose, quiesce=quiesce)
        with self._cluster_hold_lock():
            self.__dict__.setdefault("_cluster_holds", {})[job_id] = hold
        log.warning(
            "[cluster-hold] %s: HOLDING this box for cluster reservation %s (%s) — the "
            "broker's own vLLM is stopped and ollama's models unloaded, and neither starts "
            "again until the job ends%s", self.host, job_id[:8], purpose or "no purpose given",
            "" if quiesce is not None else
            f" (NOTHING will be stopped: {llm_idle_unload._DISABLE_ENV} is set)")
        self._abandon_switch(f"the box is HELD for cluster reservation {job_id[:8]}",
                             stop_loading=False)
        # The card is ours now; whoever else on this lane is still talking to its server must
        # stop, or the next request reloads what the first tick unloads.
        self._cancel_inflight_llm_pool_jobs(
            f"the box is HELD for cluster reservation {job_id[:8]}", keep=job_id)
        return hold

    def _start_cluster_hold_watch(self, hold: _ClusterHold) -> None:
        """Tick :meth:`_cluster_hold_tick` at once — shut down classic vllm first —
        and then every :data:`CLUSTER_HOLD_POLL_S` until the hold ends, on a
        thread of its own: the job's body blocks its pool thread for as long as it holds."""
        def _watch() -> None:
            while True:
                try:
                    self._cluster_hold_tick(hold)
                except Exception:
                    log.exception("[cluster-hold] %s: hold tick failed (retried)", self.host)
                if hold.stop.wait(self.CLUSTER_HOLD_POLL_S):
                    return

        hold.thread = threading.Thread(target=_watch, daemon=True,
                                       name=f"cluster-hold[{hold.job_id[:8]}]")
        hold.thread.start()

    def _cluster_hold_tick(self, hold: _ClusterHold | None = None) -> None:
        """One pass over a held box: no switch in flight, the broker's ``vllm`` container down
        (by container state — a vLLM still loading answers no probe), every ollama model
        unloaded. A resurrection, engine-driven or out of band, is met on the next pass. No
        exemption: the job being held uses no LLM."""
        holds = self.__dict__.get("_cluster_holds") or {}
        if hold is None:
            hold = next(iter(holds.values()), None)
        if hold is None or hold.job_id not in holds:
            return
        self._abandon_switch(f"the box is HELD for cluster reservation {hold.job_id[:8]}")
        if hold.quiesce is not None:
            hold.quiesce.tick()

    def _end_cluster_hold(self, hold: _ClusterHold) -> None:
        """The reservation's job ended (the LB cancelled its run; it failed or spilled): stop
        the watch and drop the hold. Normal service resumes — the box claims what fits, and
        the idle stopper may start a vLLM on demand again."""
        with self._cluster_hold_lock():
            removed = (self.__dict__.get("_cluster_holds") or {}).pop(hold.job_id, None)
        hold.stop.set()
        thread = hold.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if removed is not None:
            log.warning("[cluster-hold] %s: RELEASED cluster reservation %s — normal service "
                        "resumes", self.host, hold.job_id[:8])

    def _refuse_blind_gpu(self) -> bool:
        """True iff this is a GPU worker whose container cannot use the GPU
        (``torch.cuda.is_available()`` is False) — the post-recreate race
        where the HOST GPU is healthy but the device cgroup was not injected.

        Such a worker MUST NOT claim GPU jobs: it would fast-fail every one with
        ``cudaErrorNoDevice`` ~1 s after claiming — a silent black hole that
        poisons every run a lane fans onto it (a GPU-blind box can fail
        every lane it draws for an hour while sibling lanes on healthy hosts complete). ``run_forever`` calls this right after the
        park-gate and exits BEFORE ``heartbeat.start`` when it's True, so the
        worker also ages out of the capacity gauge. Recovery is a worker restart,
        which re-probes. cpu/ingest workers never probe and never refuse."""
        if self.queue != "gpu":
            return False
        if not get_config().gpu_requires_local_device:
            # REMOTE/THIN gpu worker: its gpu nodes are HTTP calls to an external
            # inference server (never torch.cuda), so it legitimately has no local
            # device and the GPU-blind guard does not apply. Never refuse.
            return False
        usable = _probe_gpu_usable()
        if usable is None or usable:
            # None ⇒ can't judge (no torch — NOT the blind condition); True ⇒ the
            # GPU is fine. Either way, don't refuse. Only a definitive False (torch
            # present but no usable device) is the GPU-blind black hole.
            return False
        log.error(
            "[claim-worker:gpu] host=%s: torch.cuda is unavailable in this "
            "container (the host GPU may be healthy but the device cgroup was "
            "not injected — container recreate race). REFUSING to claim GPU jobs so "
            "this worker does not fast-fail every job it draws. Restart this "
            "worker to re-probe the GPU.",
            self.host,
        )
        return True

    def _start_inference_server(self) -> None:
        """Start this machine's LLM server when this GPU worker enters its claiming
        state (boot-ON, or unparked from OFF→ON) — the host-wired
        ``EngineConfig.inference_server_start_fn``, the ON twin of the worker-control
        OFF stop. GPU-lane only, best-effort (a failure never blocks claiming;
        idempotent — starting an already-running server is a no-op). Unset ⇒ no-op."""
        if self.queue != "gpu":
            return
        fn = get_config().inference_server_start_fn
        if fn is None:
            return
        why = self._llm_starts_forbidden()
        if why:
            # A lane that restarts under a cluster reservation gets here BEFORE it re-claims
            # the row its predecessor held; the box is not the broker's to serve from.
            log.warning("[claim-worker:gpu] %s: NOT starting the inference server — %s",
                        self.host, why)
            return
        try:
            fn()
            log.info("[claim-worker:gpu] GPU ON — started the inference server")
        except Exception:
            log.exception(
                "[claim-worker:gpu] inference_server_start_fn failed (ignored — "
                "claiming proceeds; the server may already be up)",
            )

    def _get_box_lease(self):
        if self._box_lease is None:
            from queue_workflows.gpu_model_lease import build_lease
            self._box_lease = build_lease()
        return self._box_lease

    def _box_serves_comfyui(self) -> bool:
        """Does THIS worker put ComfyUI on its box? Then a no-model job here is a
        ComfyUI render (ComfyUI's own server kind), NOT an ollama LLM dispatch — so it
        must key to a DISTINCT box-lease slot, else a render and an ollama job on one
        card wrongly share the LLM slot (the cross-project over-claim). Signalled by a
        wired ComfyUI lifecycle (the usual case — a host-managed ComfyUI leaves the URL
        unset but still wires the lifecycle) or a configured ComfyUI URL."""
        from queue_workflows.envcompat import env_get
        cfg = get_config()
        url = (env_get("QUEUE_WORKFLOWS_COMFYUI_URL") or cfg.comfyui_url or "").strip()
        return bool(url) or cfg.comfyui_start_fn is not None

    def _effective_gpu_model(self, job: dict) -> str:
        """WHAT OCCUPIES THE CARD while this job runs — not necessarily a model.

        Resolution order, most specific first:

        0. A CLUSTER RESERVATION (``node_queue.CLUSTER_SERVER_KIND``) holds the card for a
           server the engine does not run: a slot of its own
           (:data:`~queue_workflows.gpu_model_lease.CLUSTER_RESERVATION_SLOT`), neither the
           LLM server's nor ComfyUI's, whatever else the row carries.
        1. ``required_model`` — a diffusion / ModelCache job IS its model.
        2. A SELF-MANAGING node (its module is registered in
           ``gpu_self_load_node_modules``): it puts its own thing on the card and
           wants no server at all, so it occupies a slot of its OWN
           (``__gpu__:<module>``).
        3. The JOB's own server kind: a row that names an LLM server
           (``server_types`` / the claim's ``resolved_server``) is the LLM
           server's occupant, whatever else the box can serve.
        4. Otherwise the BOX's server-kind slot — ComfyUI's when this worker
           serves ComfyUI, else the ollama LLM server's.

        Step 3 was missing and unified-memory GPU boxes exposed it: they serve ComfyUI (a configured
        URL) AND the LLM lanes, which step 4 alone never anticipated. Every no-model
        job there took the ComfyUI slot — ``system.vllm_lease`` and every LLM dispatch
        included — so the panel showed "ComfyUI render" beside vLLM's real model and
        warned of two residents while ComfyUI's queue was empty; worse, an LLM dispatch
        and a real render then shared ONE slot and the lease granted both, putting two
        models on one card.

        Step 2 is the one that was missing, and its absence was not cosmetic. A
        self-managing job fell through to :data:`LLM_SERVER_SLOT`, so it SHARED
        the lease with ollama dispatches instead of excluding them: the lease
        granted both and two unrelated workloads ran on one card, which is the
        exact thing "one model per physical box" exists to prevent. It also made
        every panel report the LLM model as "in use" while a Blender render held
        the GPU.

        Self-managing modules get a slot EACH rather than one shared key: two
        renders of the same node share a card fine, a Blender render and a
        video-model-a render must not.
        """
        from queue_workflows.gpu_model_lease import (
            CLUSTER_RESERVATION_SLOT, COMFYUI_SERVER_SLOT, LLM_SERVER_SLOT, self_managed_slot,
        )
        if node_queue.is_cluster_reservation(job):
            return CLUSTER_RESERVATION_SLOT
        required = job.get("required_model")
        if required:
            return required
        module = (job.get("node_module") or "").strip()
        if module and module in self._self_managing_modules():
            return self_managed_slot(module)
        if self._job_serves_llm(job):
            return LLM_SERVER_SLOT
        return COMFYUI_SERVER_SLOT if self._box_serves_comfyui() else LLM_SERVER_SLOT

    @staticmethod
    def _job_serves_llm(job: dict) -> bool:
        """Does THIS row go to an LLM server? The kind the claim stamped
        (``resolved_server``, 0023) if it has one, else the kinds it asked for
        (``server_types``, 0022). Only the kinds the engine knows as LLM servers
        count — an unrecognised one says nothing about the card."""
        kinds = [str(job.get("resolved_server") or "").strip().lower()]
        kinds += [str(k or "").strip().lower() for k in (job.get("server_types") or [])]
        return any(k in (worker_control.SERVER_TYPE_OLLAMA, worker_control.SERVER_TYPE_VLLM)
                   for k in kinds)

    def _self_managing_modules(self) -> frozenset[str]:
        """Modules that declare they manage their own GPU work.

        The SAME registry the run-expansion guard reads
        (``_assert_gpu_nodes_declare_model``): declaring a module there is
        already the statement "this gpu node has no model on purpose". Reusing it
        means new self-managing work needs no second registration and no new
        sentinel — the failure mode a hand-written slot list guarantees is that
        the next one is forgotten and silently borrows the LLM's.
        """
        cfg = get_config()
        return frozenset(getattr(cfg, "gpu_self_load_node_modules", frozenset()) or ())

    def _release_llm_ram_unless_llm_owns_the_card(self, slot: str) -> None:
        """The card has ONE occupant. If it is not the LLM server, the LLM
        server's weights are not entitled to the card — unload them.

        The lease alone could not achieve this, because a resident model is not
        the same thing as a held lease: ollama's keep-alive holds its model's
        weights AFTER its job finished and its lease was released. So a Blender
        render would acquire the box legitimately, and run alongside those weights on
        the card it had just been granted exclusive use of. Exclusivity that only
        covers the bookkeeping and not the VRAM is not exclusivity.

        The same lever the park watchdog uses, for the same reason — the daemon
        survives, only the weights go. Best-effort and silent on failure: a
        render must not fail because a probe was unreachable, and the next
        acquire tries again.
        """
        from queue_workflows.gpu_model_lease import describe_slot, is_llm_slot

        if is_llm_slot(slot) or env_get(_DISABLE_CARD_RECLAIM_ENV):
            return
        try:
            from queue_workflows import llm_probe
            from queue_workflows.llm_backends import factory as _llm_factory

            base = _llm_factory.resolve_base_url()
            names = sorted({str(m.get("name"))
                            for m in (llm_probe.loaded_models_info(base) or [])
                            if m.get("name")})
            if not names:
                return
            log.info(
                "[gpu-lease] %s: %s took the card but ollama still held %s — "
                "unloading; one occupant per card, and a keep-alive is not one",
                self.host, describe_slot(slot), ", ".join(names),
            )
            llm_probe.unload_ollama_models(base, names)
        except Exception:
            log.debug("[gpu-lease] %s: could not reclaim LLM RAM (retried on the "
                      "next acquire)", self.host, exc_info=True)

    def _spill_clock(self) -> float:
        """Monotonic now, injectable for tests (the bare workers claim-path tests build
        skip ``__init__``, so every spill field is read with a default)."""
        fn = getattr(self, "_spill_now", None)
        return fn() if fn is not None else time.monotonic()

    def _spill_paused(self) -> bool:
        """Is this worker briefly holding off gpu claims after its own box denied it?"""
        return self._spill_clock() < getattr(self, "_spill_pause_until", 0.0)

    def _acquire_box_or_spill(self, job: dict) -> bool:
        """ONE-MODEL-PER-BOX admission, across ALL projects on this physical GPU.
        Acquire the box-wide model lease for this job's effective model BEFORE it
        runs. Granted ⇒ run (and hold the lease for that model until it unloads).
        DENIED (a live worker on this box — any project, any DB — holds a DIFFERENT
        model) ⇒ re-queue the job so it SPILLS to a box that holds its model or is
        free, and DON'T execute, so a second model never loads on the card.

        Inert (always grants) until the shared lease dir is configured — byte-compat."""
        model = self._effective_gpu_model(job)
        if not self._get_box_lease().acquire(model):
            held = self._get_box_lease().current().model
            now = self._spill_clock()
            streak = getattr(self, "_spill_streak", 0)
            pause = min(SPILL_PAUSE_S * (2 ** min(streak, 8)), SPILL_PAUSE_MAX_S)
            self._spill_streak = streak + 1
            # THE BRAKE. The re-queue puts this row at the FRONT, so without a pause
            # this worker simply claims it again on the next loop and is denied again
            # — a 12 Hz claim/requeue storm against the queue until the holder's TTL
            # expires. Pausing THIS box's gpu claiming lets a free/matching peer take
            # the row, which is what the spill is for.
            self._spill_pause_until = now + pause
            job_id = job.get("id")
            last = getattr(self, "_spill_said", None)
            if last is None or last[0] != job_id or now - last[1] >= SPILL_REPORT_EVERY_S:
                self._spill_said = (job_id, now)
                log.warning(
                    "[claim-worker:gpu] host=%s: box already holds %r — SPILLING job "
                    "%s (node=%s, model=%s) to a free/matching box (one model per GPU "
                    "box across all projects); pausing gpu claims here for %.0fs",
                    self.host, held, job_id, job.get("node_id"), model, pause,
                )
            else:
                log.debug(
                    "[claim-worker:gpu] host=%s: still denied by %r — spilling %s "
                    "again (pause %.0fs)", self.host, held, job_id, pause,
                )
            try:
                node_queue.requeue_job_for_retry(job["id"])
            except Exception:
                log.exception("[claim-worker:gpu] spill re-queue failed for %s", job.get("id"))
            return False
        self._box_acquire_granted(model)
        return True

    def _box_acquire_granted(self, model: str) -> None:
        """The bookkeeping every GRANTED box acquire does, whichever path asked for it.

        Granted means the box is ours, so whatever was denying us is gone: clear the brake, or
        a box that spilled once would stay slowed down after it got the card."""
        self._spill_streak = 0
        self._spill_pause_until = 0.0
        self._box_held_model = model
        self._box_active_jobs += 1     # hold the box for this job's whole lifetime
        self._release_llm_ram_unless_llm_owns_the_card(model)

    def _acquire_box_for_reservation(self, job: dict, hold: _ClusterHold) -> bool:
        """Take the card for a CLUSTER RESERVATION — WAITING for the occupant its own hold is
        clearing, instead of spilling behind it.

        THE FLAP. The lease was held by the resident
        ``chat-model-a:latest``, so the reservation was DENIED, spilled, and ended its hold —
        ``normal service resumes`` — which re-opened the box to LLM claims. It then re-claimed
        and repeated: HOLDING/RELEASED cycles every few seconds, the spill brake backing off 1 s →
        30 s, while the card never actually changed hands.

        The reservation is the decision that this box belongs to the cluster, and
        the thing denying it is usually the very model the hold is unloading. So it ticks the
        hold — that is what frees the card — and asks again, until
        :data:`CLUSTER_RESERVATION_ACQUIRE_MAX_S`. Only something the hold CANNOT clear
        (another project's render) is still there at the ceiling, and that is a real spill."""
        model = self._effective_gpu_model(job)
        deadline = self._spill_clock() + CLUSTER_RESERVATION_ACQUIRE_MAX_S
        waited = False
        while True:
            lease = self._get_box_lease()
            if lease is None or lease.acquire(model):
                if waited:
                    log.warning("[cluster-hold] %s: the card came free — the reservation has "
                                "the box", self.host)
                self._box_acquire_granted(model)
                return True
            if self._spill_clock() >= deadline:
                held = None
                try:
                    held = lease.current().model
                except Exception:                   # pragma: no cover - a probe that raises
                    log.debug("[cluster-hold] %s: lease read failed", self.host, exc_info=True)
                log.error(
                    "[cluster-hold] %s: the card is STILL held by %r after %.0fs — the "
                    "reservation gives the box back and spills; this is an occupant the hold "
                    "cannot clear (another project's work), not the broker's own LLM.",
                    self.host, held, CLUSTER_RESERVATION_ACQUIRE_MAX_S)
                return False
            if not waited:
                waited = True
                log.warning(
                    "[cluster-hold] %s: the card is still occupied — WAITING for it (up to "
                    "%.0fs) rather than handing the box back; the hold is what clears it.",
                    self.host, CLUSTER_RESERVATION_ACQUIRE_MAX_S)
            self._cluster_hold_tick(hold)
            if hold.stop.wait(CLUSTER_RESERVATION_ACQUIRE_POLL_S):
                return False                        # the hold ended under us

    def _box_job_done(self) -> None:
        """A job that held the box has finished — decrement the active count (floor 0).
        Called from the node-execution ``finally`` on EVERY path. The box is NOT
        released here: while a model stays warm the box is kept for it (affinity, no
        thrash), and the tick frees it once the model actually unloads. Only the
        transition to 0 active jobs re-enables that residency-driven release."""
        if self._box_active_jobs > 0:
            self._box_active_jobs -= 1

    def _model_still_resident(self, held: str) -> bool:
        """Is ``held`` still on the card? For the LLM slot: any model loaded on the
        box's ollama. For the ComfyUI slot: ComfyUI reports a model resident (needs a
        configured URL to probe — an unset URL, the host-managed-ComfyUI case, reads
        not-resident, so the box releases at job end via the active-jobs guard rather
        than warm-holding on a card it can't see). For a diffusion model: the warm
        ModelCache still holds it."""
        from queue_workflows.gpu_model_lease import COMFYUI_SERVER_SLOT, LLM_SERVER_SLOT
        if held == LLM_SERVER_SLOT:
            from queue_workflows import llm_probe
            from queue_workflows.llm_backends import factory as _llm_factory
            return bool(llm_probe.loaded_models(_llm_factory.resolve_base_url("ollama")))
        if held == COMFYUI_SERVER_SLOT:
            from queue_workflows import llm_probe
            from queue_workflows.envcompat import env_get
            url = (env_get("QUEUE_WORKFLOWS_COMFYUI_URL") or get_config().comfyui_url or "").strip()
            return bool(url) and llm_probe.comfyui_loaded(url)
        return getattr(self.model_cache, "current_model", None) == held

    #: After shedding, how long before a still-dying box is shed again. Memory comes
    #: back over seconds (a docker stop, a model unload); a second stop racing the
    #: first helps nobody. The KILL floor ignores this.
    SHED_COOLDOWN_S = 60.0

    def _shed_llm_if_box_is_dying(self, available: int | None = None) -> bool:
        """THE RULE: a box out of memory gives its LLM servers back and is
        NEVER parked. Returns True when something was shed this call.

        Two lines on MemAvailable: the danger floor (12 GB) sheds at most once per
        :data:`SHED_COOLDOWN_S`; the kill floor (512 MB) sheds
        at once, cooldown or not. Shedding is: stop the vLLM container whether it is
        serving or still loading, unload every model ollama holds, and abandon a warm
        in flight so the tick does not go on to probe for a server it just gave back.
        The lane stays ON — the next start is planned against what is then free
        (:meth:`_vllm_start_plan`), and every job that fits keeps flowing. Nothing here
        writes worker_controls, and nothing may."""
        from queue_workflows import llm_context, memory_sentinel

        if available is None:
            available = llm_context.mem_available_bytes()
        if available is None:
            return False                        # a KNOWN shortage only
        available = int(available)
        kill = available < memory_sentinel.kill_floor_bytes()
        if not kill and available >= _danger_floor_bytes():
            return False
        now = time.monotonic()
        last = self.__dict__.get("_shed_at")
        if not kill and last is not None and now - last < self.SHED_COOLDOWN_S:
            return False
        self.__dict__["_shed_at"] = now
        why = (f"{available / 2**30:.1f} GB available — "
               + (f"under the {memory_sentinel.kill_floor_bytes() / 2**20:.0f} MB KILL floor"
                  if kill else f"under the {_danger_floor_bytes() / 2**30:.0f} GB danger floor")
               + "; the box keeps its lane and starts again sized to what is free")
        log.error("[llm-memory] %s: SHEDDING the LLM servers%s — %s", self.host,
                  " NOW" if kill else "", why)
        shed = False
        from queue_workflows import server_switch as _sw

        if self._switch.state == _sw.WARMING:
            started = self.__dict__.get("_warm_started_at")
            cost = f"{(now - started) / 60.0:.1f} min into the load" if started else "mid-load"
            self._switch.abandon(f"{why} ({cost})")
            self.__dict__.pop("_vllm_decision", None)
            self.__dict__.pop("_warm_started_at", None)
            self._note_startup_loss(f"{why} — {cost}")
        if self._stop_loading_vllm(why, what="serving or loading"):
            shed = True
        try:
            from queue_workflows import llm_probe
            from queue_workflows.llm_backends import factory as _llm_factory

            base = _llm_factory.resolve_base_url("ollama")
            names = sorted({str(m.get("name")) for m in (llm_probe.loaded_models_info(base) or [])
                            if m.get("name")})
            if names:
                llm_probe.unload_ollama_models(base, names)
                log.error("[llm-memory] %s: UNLOADED %s from ollama — %s", self.host,
                          ", ".join(names), why)
                shed = True
        except Exception:
            log.exception("[llm-memory] %s: could not unload ollama's models", self.host)
        self._stop_gpu_outsiders_if_still_short()
        return shed

    def _stop_gpu_outsiders_if_still_short(self) -> list[str]:
        """THE ESCALATION, in the order etiquette demands. The broker has just given
        back its OWN servers; if the box is STILL below the danger floor, something else
        is taking it down and "whose work is it" stops being the question — every GPU
        container the allow-list does not name is stopped (:func:`rogue_llm.stop_gpu_outsiders`).

        Presence on the box was never the offence — memory is: GPU work outside the
        engine is stopped only by the RAM guard, never for merely existing. So this runs ONLY here, after our own memory went back and did not
        save the box."""
        from queue_workflows import llm_context, rogue_llm

        available = llm_context.mem_available_bytes()
        if available is None or int(available) >= _danger_floor_bytes():
            return []
        why = f"{int(available) / 2**30:.1f} GB available"
        stopped = rogue_llm.stop_gpu_outsiders(host=self.host, why=why)
        if stopped:
            log.error("[llm-memory] %s: the box is STILL short (%s) after the broker gave "
                      "back its own LLM servers — stopped the GPU work outside the "
                      "allow-list: %s", self.host, why, ", ".join(stopped))
        return stopped

    #: A vLLM load can cost many minutes. A start KILLED mid-load is that
    #: time wasted, and evidence: the box could not hold this server while everything
    #: else was happening. Each loss raises the headroom the next start demands over the
    #: bare need, so a retry that would only just fit waits instead of burning the time
    #: again. Cleared the moment a start completes (its cache allocated) — the newer,
    #: better evidence.
    STARTUP_LOSS_HEADROOM = 0.25
    STARTUP_LOSS_HEADROOM_MAX = 1.5

    #: How long MemAvailable must cover a worthwhile vLLM before a memory hold lifts.
    MEMORY_RECOVERY_S = 60.0
    #: How often the need is recomputed (a catalog read); the comparison is per second.
    MEMORY_NEED_TTL_S = 60.0

    def _memory_tick(self, available: int | None) -> None:
        """What the sentinel does with one reading, once a second: shed if the box is
        dying, and notice when it has recovered."""
        self._shed_llm_if_box_is_dying(available)
        self._note_memory_recovery(available)

    def _mark_warm_clock(self) -> None:
        """When did the server now WARMING begin? Only so a kill can say what it cost —
        "4.7 min into the load" is the number that tells an operator whether the guard
        is saving time or wasting it."""
        from queue_workflows import server_switch as _sw

        if self._switch.state == _sw.WARMING:
            self.__dict__.setdefault("_warm_started_at", time.monotonic())
        else:
            self.__dict__.pop("_warm_started_at", None)

    def _vllm_startup_headroom(self) -> float:
        """How much MORE than the bare need a start must see, given how many loads this
        box has already lost mid-way. ``1.0`` with a clean history."""
        losses = int(self.__dict__.get("_vllm_startup_losses") or 0)
        return min(1.0 + self.STARTUP_LOSS_HEADROOM * losses, self.STARTUP_LOSS_HEADROOM_MAX)

    def _note_startup_loss(self, why: str) -> None:
        """A vLLM load was killed before it finished. Count it and say what it cost."""
        losses = int(self.__dict__.get("_vllm_startup_losses") or 0) + 1
        self.__dict__["_vllm_startup_losses"] = losses
        self.__dict__.pop("_vllm_need_cache", None)
        self.__dict__.pop("_vllm_decision", None)
        log.error("[llm-memory] %s: that vLLM load is time wasted (%d lost start(s) since "
                  "one last completed) — the next start needs %.0f%% more than the bare "
                  "minimum before it is worth trying: %s", self.host, losses,
                  (self._vllm_startup_headroom() - 1.0) * 100, why)

    def _clear_startup_losses(self) -> None:
        if self.__dict__.get("_vllm_startup_losses"):
            log.info("[llm-memory] %s: a vLLM start completed — the lost-start history "
                     "is cleared", self.host)
        self.__dict__["_vllm_startup_losses"] = 0
        self.__dict__.pop("_vllm_need_cache", None)

    def _vllm_memory_need_bytes(self) -> int | None:
        """What a vLLM WORTH STARTING needs on this box, plus the reserve: with the
        remembered profile, weights + overhead + the minimum concurrency's worth of
        cache (+ the plan's margin); without one, the minimum pool. ``None`` when the
        box cannot say. Cached for :data:`MEMORY_NEED_TTL_S` — one catalog read a
        minute, one comparison a second."""
        from queue_workflows import llm_context, llm_context_lever

        now = time.monotonic()
        cached = self.__dict__.get("_vllm_need_cache")
        if cached is not None and now - cached[0] < self.MEMORY_NEED_TTL_S:
            return cached[1]
        need: int | None = None
        try:
            total = llm_context.mem_total_bytes()
            reserve = llm_context.reserve_bytes()
            info = llm_context_lever.describe(worker_control.SERVER_TYPE_VLLM)
            profile = self._vllm_remembered_profile(info) if info.exists else None
            if total:
                if profile is not None and profile.per_token_bytes > 0:
                    window = (info.configured or self._vllm_model_max_context()
                              or profile.window)
                    per_seq = int(window) * profile.per_token_bytes
                    wanted = max(1, _vllm_min_concurrency())
                    need = int((profile.weights_bytes + profile.overhead_bytes
                                + wanted * per_seq) * (1.0 + llm_context.VLLM_PLAN_MARGIN)) + reserve
                else:
                    need = int(_vllm_min_utilization() * int(total)) + reserve
            if need is not None:
                need = int(need * self._vllm_startup_headroom())
        except Exception:
            log.debug("[memory-sentinel] %s: could not compute the vLLM need", self.host,
                      exc_info=True)
        self.__dict__["_vllm_need_cache"] = (now, need)
        return need

    def _note_memory_recovery(self, available: int | None) -> None:
        """THE RULE: if RAM frees for more than a minute and the box can switch from
        ollama to vLLM, do it. A hold the plan placed over MEMORY is lifted
        once MemAvailable has covered what a worthwhile vLLM needs for
        :data:`MEMORY_RECOVERY_S` continuously; the demand logic then requests the
        switch as it always does and the plan sizes the start to what is free. A dip
        starts the count again. A hold for any other reason is left alone."""
        from queue_workflows import llm_context

        kind = worker_control.SERVER_TYPE_VLLM
        reason = self._switch.quarantine_reason(kind)
        if available is None or reason is None or llm_context.MEMORY_REFUSAL not in reason:
            self.__dict__["_mem_ok_since"] = None
            return
        need = self._vllm_memory_need_bytes()
        if need is None or int(available) < need:
            self.__dict__["_mem_ok_since"] = None
            return
        now = time.monotonic()
        since = self.__dict__.get("_mem_ok_since")
        if since is None:
            self.__dict__["_mem_ok_since"] = now
            return
        if now - since < self.MEMORY_RECOVERY_S:
            return
        self.__dict__["_mem_ok_since"] = None
        log.warning("[memory-sentinel] %s: %.1f GB available for %.0f s — more than the "
                    "%.1f GB a vLLM needs here; lifting the memory hold so the switch may "
                    "ask for it again (%s)", self.host, int(available) / 2**30,
                    self.MEMORY_RECOVERY_S, need / 2**30, reason[:120])
        self._switch.clear_quarantine(kind)

    def _arm_memory_sentinel(self) -> None:
        """Start the 1 Hz memory sentinel (gpu lane only): every second, MemAvailable is
        read and handed to :meth:`_memory_tick` — shed if dying, recover if it can.
        Env-gated; never breaks worker startup."""
        from queue_workflows import llm_context, memory_sentinel

        if memory_sentinel.disabled():
            log.info("[memory-sentinel] %s: disabled (%s)", self.host, memory_sentinel.DISABLE_ENV)
            return
        try:
            self._memory_sentinel = memory_sentinel.MemorySentinel(
                read_fn=llm_context.mem_available_bytes,
                act_fn=self._memory_tick, label=self.host).start()
            log.info("[memory-sentinel] %s: watching MemAvailable every %.0fs — danger "
                     "floor %.0f GB, kill floor %.0f MB", self.host,
                     memory_sentinel.INTERVAL_S, _danger_floor_bytes() / 2**30,
                     memory_sentinel.kill_floor_bytes() / 2**20)
        except Exception:
            log.exception("[memory-sentinel] %s: could not start (ignored)", self.host)

    def _box_lease_tick(self) -> None:
        """Heartbeat hook (gpu only): refresh the box's observed LLM state for the
        0022 affinity sort, then keep holding the box lease while our model is
        RESIDENT (renew), and RELEASE it once the model unloads — so the box frees
        for a peer's model only when this one is actually gone. Best-effort."""
        # Piggy-backed on the existing gpu heartbeat hook rather than adding a second
        # daemon: both want the same ~10 s cadence and the same residency probe.
        try:
            # ONE LLM DAEMON PER BOX, enforced where a second one can be SEEN — the
            # docker socket — and before the residency probe below looks, so it sees a
            # box already cleaned. A no-socket box answers instantly (see rogue_llm).
            from queue_workflows import rogue_llm
            rogue_llm.sweep(host=self.host)
        except Exception:
            log.debug("[rogue-llm] %s: sweep failed (ignored)", self.host, exc_info=True)
        self._refresh_box_llm_state()
        # The sentinel does this every second; the tick repeats it so a box whose
        # sentinel is disabled (or not yet armed) still sheds within ~10 s.
        self._shed_llm_if_box_is_dying()
        self._drive_server_switch()
        try:
            # 0028: only once the switch above has settled — a box already out of
            # service must not be asked to go out of service for a window as well.
            self._maybe_request_context_switch()
        except Exception:
            log.exception("[llm-context] %s: context-switch check failed (ignored)",
                          self.host)
        if self._switch.blocks_claims():
            # While a switch is in flight the box lease belongs
            # to the SWITCH (SWITCH_SLOT, acquired in _drive_server_switch). This
            # method's held-model renew ran one line later and — renew() being
            # acquire() on the single-slot store — CLOBBERED it, so peers were
            # denied on only 1 of 6 switch ticks and kept reloading the outgoing
            # server (the original failure, back through a side door). The
            # held model is being drained/unloaded by the switch anyway; its lease
            # bookkeeping resumes when the switch resolves.
            return
        held = self._box_held_model
        if not held:
            return
        try:
            # Keep holding while a job is ACTIVE (robust to a residency-probe flicker
            # — a mid-job blip must never drop the box) OR while the model stays warm
            # after the job (affinity). Release only when NO job runs AND the model has
            # actually left the card, so the box frees for a peer's model exactly then.
            if self._box_active_jobs > 0 or self._model_still_resident(held):
                self._get_box_lease().renew(held)
            else:
                self._get_box_lease().release()
                self._box_held_model = None
        except Exception:
            log.exception("[claim-worker:gpu] box-lease tick failed (ignored)")

    def _arm_llm_idle_reaper(self) -> None:
        """Arm the external-LLM-server idle reaper (gpu only): once the box's ollama
        model has seen NO use — from ANY client, observed via ``/api/ps``
        ``expires_at`` advances — for the TTL (``llm_server_idle_ttl_s``, default
        5 min), unload it to free RAM. Reads this box's endpoint live via the
        backend factory; a vllm box has nothing on ``/api/ps`` so the reaper no-ops
        there (the LLMSupervisor handles vllm idle). Best-effort + env-gated; never
        breaks worker startup."""
        try:
            from queue_workflows import llm_idle_unload, llm_probe
            from queue_workflows.llm_backends import factory as _llm_factory

            self._llm_idle_reaper = llm_idle_unload.ExternalModelIdleReaper(
                active_fn=self._gpu_active_count,
                loaded_models_info_fn=lambda: llm_probe.loaded_models_info(
                    _llm_factory.resolve_base_url()),
                unload_fn=lambda models: llm_probe.unload_ollama_models(
                    _llm_factory.resolve_base_url(), models),
                label=self.host,
            )
            self._llm_idle_reaper.ensure_started()
        except Exception:
            log.exception(
                "[claim-worker:gpu] LLM idle-unload reaper arm failed (ignored)"
            )
        # The vLLM twin. Same TTL, same server-observed idleness — but a whole
        # SERVER stop rather than a model unload, because vLLM holds its weights for
        # its process lifetime and there is nothing smaller to free.
        #
        # NOT LLMSupervisor: its clock is the backend's in-process request counter,
        # which only ticks when a caller routes through the backend object. Our nodes
        # are handed a URL and POST it themselves — as do host-internal callers and
        # other projects sharing the box — so that clock reads "idle" while the
        # server is flat out. Arming it would SIGTERM vLLM mid-generation. Watch the
        # server instead (llm_probe.vllm_activity), exactly as the ollama reaper does.
        try:
            from queue_workflows import llm_idle_unload, llm_probe
            from queue_workflows.llm_backends import factory as _llm_factory

            def _stop_vllm() -> None:
                stop = get_config().vllm_stop_fn
                if stop is None:
                    log.warning(
                        "[llm-idle] %s: vLLM is idle but no stop hook is wired — "
                        "set_vllm_lifecycle(start_fn, stop_fn) so its VRAM can be "
                        "reclaimed", self.host,
                    )
                    return
                stop()

            def _start_vllm() -> None:
                # Was a bare unload+start, which returned while vLLM still had ~8 min
                # of loading ahead and the box kept claiming — reloading ollama and
                # ending with both resident. Now it opens a SWITCH: the box goes out
                # of service until vLLM actually answers.
                if not self._box_has_vllm_endpoint():
                    return          # capability gate: no distinct vLLM URL, no chase
                # R4 (0028): it is being started anyway, so start it at the window the
                # queue needs — else a sized job waits out this cold start and then a
                # second one for the resize. None ⇒ its configured default, as before.
                self.request_server_switch("vllm", self._queued_context_for("vllm"))

            def _pending_vllm() -> int:
                """Jobs queued for vLLM that THIS box could serve — by its placement too:
                a row pinned to another box, or avoiding this one, can never run here and
                must not start a server here. The same threshold the dispatch-time
                amortisation gate uses, so "is a cold start worth it?" has one answer
                wherever it is asked."""
                from queue_workflows import node_queue as _nq
                return _nq.count_pending_for_server(None, "vllm", any_model=True,
                                                    box=self._physical_box())

            self._vllm_idle_stopper = llm_idle_unload.VLLMIdleStopper(
                active_fn=self._gpu_active_count,
                activity_fn=lambda: llm_probe.vllm_activity(
                    _llm_factory.resolve_base_url("vllm")),
                stop_fn=_stop_vllm,
                # The START half. Without it the stopper is a one-way door: stopped
                # ⇒ unadvertised ⇒ no job can prefer it ⇒ never used ⇒ stays stopped.
                # Without it vLLM can sit exited with 0 requests while thousands of jobs
                # preferring it queue behind ollama.
                start_fn=_start_vllm,
                pending_fn=_pending_vllm,
                # THE RESOLVER, not its value: the topology file's vllm_min_queue is
                # operator policy documented to need no restart, and reading it once
                # here froze a box under a threshold the operator had already lowered
                # (a queued vLLM job could sit unassignable for hours).
                min_queue=_llm_factory.resolve_cold_start_min_queue,
                label=self.host,
                # PARK RELEASES THE CARD: a worker whose control
                # row has gone OFF (drain pending, or racing its own hard stop)
                # must not ignite a multi-minute vLLM start the parked box can
                # never use — nor may a box HELD for a cluster reservation, or one a
                # reservation is queued for. One rule for every start path
                # (_llm_starts_forbidden), failing open on an unreadable read: a
                # control-plane blip degrades to the pre-gate behaviour.
                may_start_fn=lambda: self._llm_starts_forbidden() is None,
            )
            self._vllm_idle_stopper.ensure_started()
        except Exception:
            log.exception(
                "[claim-worker:gpu] vLLM idle-stopper arm failed (ignored)"
            )

    def _drive_server_switch(self) -> None:
        """One tick of the switch machine, with the observations it needs and the
        actions it asks for. Best-effort: a probe that fails simply produces no
        transition this tick, and the machine's own timeouts guarantee the box cannot
        stay out of service forever."""
        from queue_workflows import server_switch as _sw
        if self._switch.state == _sw.IDLE:
            self._mark_warm_clock()          # an abandoned warm leaves no clock running
            return
        # A BOX THAT MUST START NOTHING ENDS ITS SWITCH — parked, held for a cluster
        # reservation, or one queued for it (_llm_starts_forbidden). First, before this tick
        # re-acquires the switch's box lease or, WARMING, re-issues START on its 30 s retry:
        # that retry, ticked by the heartbeat a drain-park keeps, is how a PARKED box
        # could start its vLLM again.
        why = self._llm_starts_forbidden()
        if why:
            self._abandon_switch(why)
            self._mark_warm_clock()
            return
        # HOLD THE BOX for the whole switch (the cross-process half). The
        # switch state is per-process, but the VRAM it protects is per-BOX — and the
        # flock box lease is the one primitive every project on the box already
        # honours. Holding a DISTINCT sentinel means every peer worker asking for
        # any model (or the LLM slot) is DENIED and SPILLS to another box for the
        # duration, instead of reloading the server being replaced. Best-effort:
        # a box with no lease dir configured simply keeps the per-process guarantee.
        try:
            lease = self._get_box_lease()
            if lease is not None:
                if not lease.acquire(SWITCH_SLOT):
                    lease.renew(SWITCH_SLOT)
        except Exception:
            log.debug("[server-switch] %s: box-lease hold failed (per-process "
                      "guarantee only this tick)", self.host, exc_info=True)
        if self._abandon_warm_if_box_is_dying():
            return
        self._mark_warm_clock()
        try:
            from queue_workflows import llm_probe, server_switch
            from queue_workflows.llm_backends import factory as _f

            target = self._switch.target
            target_context = self._switch.target_context
            resize = self._switch.resize
            ollama_url = _f.resolve_base_url("ollama")
            vllm_url = _f.resolve_base_url("vllm")
            # "outgoing" is whatever is NOT the target — and on a RESIZE there is no
            # outgoing server at all: the lever replaces the one being resized.
            if resize:
                outgoing_loaded = False
                # ANSWERING and serving the window. A container configured for it that
                # never came up is the failure this switch has to survive, so the
                # configured value alone must never read as ready.
                incoming_ready = (
                    llm_probe.probe_llm_servers(_f.resolve_base_url(target)) == [target]
                    and self._serves_window(target, target_context))
            elif target == "vllm":
                outgoing_loaded = bool(llm_probe.loaded_models(ollama_url))
                incoming_ready = (
                    llm_probe.probe_llm_servers(vllm_url) == [llm_probe.VLLM]
                    and self._serves_window(target, target_context))
            else:
                outgoing_loaded = bool(llm_probe.vllm_served_models(vllm_url))
                incoming_ready = (
                    llm_probe.probe_llm_servers(ollama_url) == [llm_probe.OLLAMA]
                    and self._serves_window(target, target_context))

            actions = self._switch.tick(
                in_flight=self._gpu_active_count(),
                outgoing_loaded=outgoing_loaded,
                incoming_ready=incoming_ready,
            )
            if any(a in (server_switch.DONE, server_switch.ABORT) for a in actions):
                try:
                    lease = self._get_box_lease()
                    if lease is not None:
                        lease.release()
                except Exception:
                    log.debug("[server-switch] %s: box-lease release failed "
                              "(TTL will expire it)", self.host, exc_info=True)
            followups: list[str] = []
            for action in actions:
                if action == server_switch.UNLOAD_OUTGOING:
                    if target == "vllm":
                        llm_probe.unload_ollama_models(
                            ollama_url, llm_probe.loaded_models(ollama_url))
                    else:
                        stop = get_config().vllm_stop_fn
                        if stop is not None:
                            stop()
                elif action == server_switch.START_INCOMING:
                    if self._cluster_hold_reason() is not None:
                        # A reservation was claimed between this tick's check and its
                        # action (the pool lane runs beside the heartbeat): the start it
                        # asks for is exactly the one the hold forbids.
                        self._abandon_switch("the box was HELD for a cluster reservation "
                                             "mid-tick")
                        break
                    # THE SEAM. Whatever the lever tells us goes straight back into the
                    # machine, in THIS tick — a definitive failure (no such container,
                    # an OCI error, no lever at all) then ends the out-of-service window
                    # now instead of at the warm timeout 1800 s later. A lever that
                    # returns None says nothing, which is UNKNOWN and behaves exactly as
                    # before, so an un-upgraded host is unaffected.
                    plan = self._vllm_start_decision() if target == "vllm" else None
                    utilization = plan.utilization if plan is not None else None
                    max_seqs = plan.max_num_seqs if plan is not None else None
                    refusal = plan.refusal if plan is not None else None
                    if refusal is not None:
                        # THE MEMORY GATE. No start into memory the box does not have:
                        # a vLLM pool is taken at start, and on a unified-memory box it
                        # is taken from the same RAM the OS and every other container
                        # live in. The switch is told the start CANNOT succeed, so it
                        # aborts now and quarantines vLLM here instead of warming for
                        # half an hour — and the demand goes to a box that can.
                        log.error("[llm-memory] %s: NOT starting vLLM — %s",
                                  self.host, refusal)
                        # A vLLM ALREADY LOADING into that memory is the very thing the
                        # verdict says cannot fit — a start issued before the verdict (an
                        # earlier engine, a hand start). Otherwise the refusal aborts
                        # the switch and leaves one loading toward the pool it had just
                        # refused, with ollama's model beside it. A vLLM that is SERVING
                        # never reaches this branch — the switch completes on it instead.
                        self._stop_loading_vllm(refusal)
                        outcome: Any = server_switch.StartOutcome.permanent(refusal)
                        followups.extend(self._switch.report_start_outcome(outcome))
                    elif target_context is not None or utilization is not None:
                        # THE WINDOW (0028), the POOL and the CONCURRENCY (a plan that
                        # names a concurrency always names the pool) are set by recreating the
                        # server's container with the values it reads at start — which
                        # also STARTS it, so this replaces the plain start lever rather
                        # than following it. Used for a resize, for a kind switch that
                        # named a window, and for a vLLM whose operator pool would not
                        # fit in what the box has free; an ollama target with no window
                        # needs nothing at all.
                        from queue_workflows import llm_context_lever
                        try:
                            outcome = llm_context_lever.set_context(
                                target, target_context, utilization=utilization,
                                max_num_seqs=max_seqs)
                        except Exception as exc:
                            outcome = server_switch.StartOutcome.transient(
                                f"{type(exc).__name__}: {exc}")
                        followups.extend(self._switch.report_start_outcome(outcome))
                    elif target == "vllm":
                        start = get_config().vllm_start_fn
                        if start is None:
                            outcome = server_switch.StartOutcome.permanent(
                                "no vllm start lever wired "
                                "(set_vllm_lifecycle(stop_fn, start_fn))")
                        else:
                            try:
                                outcome = start(None)
                            except Exception as exc:
                                # A raising lever performed no start. Transient, so a
                                # flaky socket retries; the ack budget bounds it.
                                outcome = server_switch.StartOutcome.transient(
                                    f"{type(exc).__name__}: {exc}")
                        followups.extend(self._switch.report_start_outcome(outcome))
                    # switching TO ollama needs no start: it is an always-on daemon.
            if server_switch.ABORT in followups:
                # The switch resolved inside this tick, so the box lease it was holding
                # must be released here too — the release above only saw `actions`.
                try:
                    lease = self._get_box_lease()
                    if lease is not None:
                        lease.release()
                except Exception:
                    log.debug("[server-switch] %s: box-lease release failed "
                              "(TTL will expire it)", self.host, exc_info=True)
            if (resize and not incoming_ready
                    and server_switch.ABORT in list(actions) + followups):
                # A GIVEN-UP RESIZE THAT LEFT NOTHING SERVING. The container was
                # recreated for a window its server could not honour (vLLM refuses a
                # max_model_len past the model's own and exits; ollama can fail to load
                # at all), so aborting alone would leave the box with NO LLM until an
                # operator noticed. Put the operator's own value back — the one
                # configuration this box is known to be able to run.
                self._restore_llm_context(target)
        except Exception:
            log.exception("[server-switch] %s: tick failed (will retry)", self.host)
        finally:
            # AFTER the transitions: the tick that ENTERS warming starts the clock, so a
            # kill can say what the load had already cost.
            self._mark_warm_clock()

    def _serves_window(self, kind: str | None, target_context: int | None) -> bool:
        """Is ``kind`` serving a window at least ``target_context`` wide? True when no
        window was asked for — then only the server answering matters.

        A window the box cannot READ reads as not-yet: a switch must not complete on an
        assumption, because completing means resuming claims for jobs that need it."""
        if not kind or target_context is None:
            return True
        current, _ceiling = self._llm_context_for_kind(kind)
        return current is not None and int(current) >= int(target_context)

    def _restore_llm_context(self, kind: str | None) -> None:
        """Put ``kind``'s server back to the window the operator configured."""
        if not kind:
            return
        from queue_workflows import llm_context_lever
        try:
            outcome = llm_context_lever.set_context(kind, None)
        except Exception:
            log.exception("[llm-context] %s: could not restore %s's own window",
                          self.host, kind)
            return
        log.error(
            "[llm-context] %s: the resize of %s failed and left it not answering — "
            "restored the configured window (%s)", self.host, kind,
            (outcome or {}).get("reason") or (outcome or {}).get("status"),
        )

    def request_server_switch(self, target: str, context_length: int | None = None,
                              *, resize: bool = False) -> bool:
        """Ask this box to move to ``target``. Returns True iff a switch started.

        ``context_length`` is the window the incoming server should be STARTED at
        (0028); ``resize=True`` keeps the kind and changes only that window.

        Refused while this box must start nothing (:meth:`_llm_starts_forbidden`): parked,
        held for a cluster reservation, or one is queued for it."""
        why = self._llm_starts_forbidden()
        if why:
            log.info("[server-switch] %s: NOT %s %s — %s", self.host,
                     "resizing" if resize else "switching to", target, why)
            return False
        return self._switch.request(target, context_length, resize=resize)

    def _maybe_request_context_switch(self) -> bool:
        """Open a RESIZE when the queue needs a window this box could serve but does
        not. Runs on the heartbeat tick; returns True iff a switch started.

        Every condition here is a way of NOT restarting a healthy server: the box must
        have a lever, its last resize must not have failed, the kill switch must be off,
        no switch may be in flight, and the demand read
        (:func:`node_queue.context_switch_need`) must say this box is the one that
        should grow — nobody is serving that window, the job has waited, and of the
        boxes that could reach it this is the tightest fit."""
        from queue_workflows import llm_context, llm_context_lever, server_switch as _sw

        if self.queue != "gpu" or self._switch.state != _sw.IDLE:
            return False
        if llm_context_lever.disabled():
            return False
        serving, ceiling = getattr(self, "_llm_context_state", (None, None))
        if ceiling is None or (serving is not None and int(ceiling) <= int(serving)):
            return False
        # The kind to resize: the one whose ceiling this is. With two kinds co-resident
        # (rare — the switch exists to keep it to one) the roomier one is the one the
        # ceiling was advertised from.
        kinds = [k for k in self._served_llm_types() if k]
        kind = max(kinds, key=lambda k: self._ctx_ceiling_by_kind.get(k, 0),
                   default=None)
        if not kind or not llm_context_lever.available(kind):
            return False
        if self._switch.resize_quarantined(kind):
            return False
        if self._llm_starts_forbidden():
            return False                # a resize restarts a server; this box starts none
        try:
            need = node_queue.context_switch_need(
                host=self.host, serving=serving, ceiling=ceiling,
                box=self._physical_box(), min_age_s=CONTEXT_SWITCH_MIN_AGE_S,
            )
        except Exception:
            log.exception("[llm-context] %s: demand read failed (no switch)", self.host)
            return False
        target = llm_context.switch_target([need], current=serving, ceiling=ceiling)
        if not target:
            return False
        log.warning(
            "[llm-context] %s: queued work needs a %s-token window and this box serves "
            "%s — resizing %s (it goes out of service until the server answers there)",
            self.host, need, serving, kind,
        )
        return self.request_server_switch(kind, target, resize=True)

    def _queued_context_for(self, kind: str) -> int | None:
        """The window to START ``kind`` at: the highest queued need this box could
        reach on it, or ``None`` for "whatever it is configured for" — the operator's
        default, which is what an un-sized queue should get (R4).

        Its ceiling comes from the last time that kind was observed serving here: a
        stopped server reports nothing, and guessing would risk starting it at a window
        it cannot honour."""
        from queue_workflows import llm_context, llm_context_lever

        if llm_context_lever.disabled() or not llm_context_lever.available(kind):
            return None
        ceiling = self._ctx_ceiling_by_kind.get(kind)
        if not ceiling:
            return None
        try:
            need = node_queue.context_switch_need(
                host=self.host, serving=None, ceiling=ceiling,
                box=self._physical_box(), min_age_s=0.0, elect=False,
            )
        except Exception:
            log.exception("[llm-context] %s: demand read failed (default window)",
                          self.host)
            return None
        return llm_context.switch_target([need], current=None, ceiling=ceiling)

    def _handover_to_vllm(self) -> bool:
        """Hand the box OVER to vLLM: unload ollama FIRST, then start. Returns True
        iff a start was issued.

        ONE SERVER AT A TIME. Two servers each hold their own full copy of the
        weights, so starting vLLM while ollama is still resident can take a
        unified-memory box to within a few GB of OOM. The start is therefore a HANDOVER, not an addition: free
        the outgoing server before the incoming one allocates, in that order, every
        time.

        Uses the residency enforcer's own eviction path
        (:func:`model_residency.clear_box_for`) so the box is cleared exactly the way
        the watchdog would clear it — one definition of "make room". If the clear
        fails we do NOT start: better a slow queue on ollama than two copies on one
        card."""
        start = get_config().vllm_start_fn
        if start is None:
            return False
        from queue_workflows import llm_probe, model_residency
        from queue_workflows.llm_backends import factory as _f
        try:
            ollama_url = _f.resolve_base_url("ollama")
            model_residency.clear_box_for(
                llm_probe.VLLM, None, self._collect_box_residents(),
                unload_ollama=lambda models: llm_probe.unload_ollama_models(
                    ollama_url, models),
                stop_vllm=lambda: None,          # we are STARTING it, not stopping
                label=self.host,
            )
        except Exception:
            log.exception(
                "[llm-idle] %s: could not free the box for vLLM — NOT starting it "
                "(two resident copies would risk an OOM)", self.host,
            )
            return False
        start(None)
        return True

    def _collect_box_residents(self) -> list:
        """Everything resident on THIS box's LLM servers right now — the shared
        probe both the residency enforcer and the ModelCache load gate use, so all
        three see the box identically."""
        from queue_workflows import model_residency
        return model_residency.probe_box_residents()

    def _stop_rogue_vllm(self) -> None:
        """Hard-kill lever for a vLLM that lost the residency decision: the
        host-wired sidecar stop (``set_vllm_lifecycle``). Without one there is no
        portable way to stop an external vLLM — say so loudly instead of pretending."""
        fn = get_config().vllm_stop_fn
        if fn is not None:
            fn()
            return
        log.error(
            "[model-residency] %s: a vLLM lost the one-model-per-box decision but "
            "no stop hook is wired — wire set_vllm_lifecycle(stop_fn, start_fn) so "
            "the enforcer can actually kill it.", self.host,
        )

    def _free_box_comfyui(self) -> None:
        """Evict lever for a ComfyUI that lost the one-server-per-box decision: POST
        ``/free`` so it drops its models + releases VRAM. No-op (with a note) when no
        ComfyUI URL is configured for this box — then ComfyUI isn't part of the box's
        residency picture and nothing on it could have been decided against."""
        from queue_workflows import llm_probe
        url = (env_get("QUEUE_WORKFLOWS_COMFYUI_URL") or get_config().comfyui_url or "").strip()
        if not url:
            log.error(
                "[model-residency] %s: a ComfyUI lost the one-server-per-box decision "
                "but no QUEUE_WORKFLOWS_COMFYUI_URL is set — cannot free it.", self.host,
            )
            return
        llm_probe.comfyui_free(url)

    def _arm_model_residency_enforcer(self) -> None:
        """Arm the ONE-MODEL-PER-BOX enforcer (gpu only): poll this box's LLM
        servers and, whenever more than ``QUEUE_WORKFLOWS_BOX_MODEL_CAP`` (default 1)
        distinct models are resident, hard-kill the extras (ollama ``keep_alive:0``;
        vLLM sidecar stop) and raise/log ``ModelResidencyViolation`` — the invariant
        the incident history shows config alone cannot hold. Best-effort arming;
        never breaks worker startup."""
        try:
            from queue_workflows import llm_probe, model_residency
            from queue_workflows.llm_backends import factory as _llm_factory

            self._residency_enforcer = model_residency.ModelResidencyEnforcer(
                collect_fn=self._collect_box_residents,
                unload_ollama_fn=lambda models: llm_probe.unload_ollama_models(
                    _llm_factory.resolve_base_url("ollama"), models),
                stop_vllm_fn=self._stop_rogue_vllm,
                free_comfyui_fn=self._free_box_comfyui,
                # A server still loading has no traffic and would lose every keeper
                # vote — which is how the incoming vLLM was killed on every switch.
                protected_server_fn=self._switch.protected_kind,
                desired_server_fn=lambda: worker_control.llm_config_for(
                    self.host, "gpu").server_type,
                label=self.host,
            )
            self._residency_enforcer.ensure_started()
        except Exception:
            log.exception(
                "[claim-worker:gpu] model-residency enforcer arm failed (ignored)"
            )

    def run_forever(self) -> None:
        """Block on ``LISTEN <wake_channel>`` and drain the queue greedily on
        each wake (and on the 1 s safety poll for a dropped NOTIFY). Uses a
        dedicated autocommit connection for the LISTEN so it never holds a
        pooled connection across the blocking wait."""
        import psycopg
        # Gate on schema readiness FIRST so the loop never polls a table the
        # orchestrator's bootstrap hasn't created yet.
        self.await_schema()
        # Host hw-metrics sampler: start it BEFORE the park gate so a PARKED gpu
        # worker KEEPS streaming this box's telemetry (CPU/GPU/RAM). Otherwise
        # turning the gpu worker OFF also kills the host's telemetry and the queue
        # gauge FREEZES at its last (busy) sample — it looks "still GPU busy" even
        # though the GPU just went idle. The sampler is host-level, independent of
        # whether this worker is claiming; one per host (flock).
        if self.queue == "gpu" and not get_config().disable_gpu_worker_hw_sampler:
            from queue_workflows import hw_metrics
            self._hw_sampler = hw_metrics.start_hw_metrics_sampler_flocked()
        # Boot-gate: if an operator has this worker turned OFF, PARK (no claim, no
        # heartbeat) until it's turned back ON. A fresh process holds no model so
        # RAM is already free; parking keeps us idle + out of the capacity gauge.
        # Returns False only if stopped while parked ⇒ exit without starting up.
        if not self._park_until_enabled():
            return
        # GPU-blind self-park: a container that can't use a CUDA device (the box-c
        # recreate race — host GPU healthy, device cgroup missing) must NOT claim.
        # It would fast-fail every GPU job ~1 s after claiming (cudaErrorNoDevice)
        # — a silent black hole. Refuse BEFORE heartbeat.start so it also ages out
        # of the capacity gauge; recovery is a worker restart, which re-probes.
        if self._refuse_blind_gpu():
            return
        log.info(
            "[claim-worker:%s] starting (host=%s priority=%d lease=%ds channel=%s)",
            self.queue, self.host, self.host_priority, self.lease_s,
            self._wake_channel,
        )
        # GPU ON: the toggle governs the machine's inference server too — bring it up
        # now that this gpu worker is (past the park gate and) about to claim. The
        # OFF twin (worker_control._stop_inference_server) stopped it. No-op unless a
        # host wired the lifecycle; idempotent if the server is already running.
        self._start_inference_server()
        # Boot zombie-kill: free my dead predecessor's slot(s) NOW — must run
        # BEFORE heartbeat.start() (our first beat marks this label fresh, which
        # would hide the predecessor's corpse from the staleness predicate).
        self._reclaim_predecessor_zombies()
        # Advertise capacity (no-op for fetch/load) + keep last_seen fresh.
        self.heartbeat.start()
        # (hw-metrics sampler already started above, before the park gate, so it
        # survives an OFF→park cycle.)
        # GPU only: arm the LLM-backend factory's config-change LISTEN invalidator
        # so an operator's ollama↔vllm / tunable edit (worker_controls, 0013) is
        # picked up instantly by the co-tenant VLM backend. Cheap, isolated, and
        # env-gated (QUEUE_WORKFLOWS_DISABLE_LLM_CONFIG_LISTENER) so it's inert in tests.
        # Wrapped because the LLM backend is an OPTIONAL subsystem (a host may run
        # no VLM at all) — its arming must never take down the claim worker.
        if self.queue == "gpu":
            try:
                from queue_workflows.llm_backends import factory as _llm_factory
                _llm_factory.start()
            except Exception:
                log.exception(
                    "[claim-worker:gpu] LLM backend factory start failed (ignored)"
                )
            self._arm_llm_idle_reaper()
            self._arm_memory_sentinel()
            self._arm_model_residency_enforcer()
        # Operator control watcher: HARD-stop this worker the instant it's turned
        # OFF (re-queue in-flight + os._exit to free RAM; the supervisor restart
        # re-enters the park gate above). Honours QUEUE_WORKFLOWS_DISABLE_WORKER_CONTROL.
        self._control_watcher = worker_control.WorkerControlWatcher(worker=self)
        self._control_watcher.start()
        # GPU VLM pool lane (NEW, additive): start the PAR-sized executor + feeder
        # thread that drains no-model GPU jobs concurrently (HTTP to the per-host
        # vLLM server, which batches up to PAR on the GPU). No-op for cpu/ingest.
        # The inline loop below now claims require_model=True (diffusion) only;
        # the two lanes' claim sets are disjoint so neither steals the other's.
        self._start_pool_lane()
        # PG-reconnect: the central claim loop is the highest-impact LISTEN
        # site — a PG bounce mid-run_forever used to exit(1) the worker
        # process and the documented "PG-restart strands workers" pattern
        # would leave it Exited until ops re-bounced the container. The
        # helper now reconnects with backoff and resumes claim_once.
        from queue_workflows.db import listen_with_reconnect
        def _body(listen_conn):
            while not self._stop.is_set():
                # Drain-park boundary (stop_policy="drain"): the watcher flags us;
                # we park HERE — between jobs, never mid-job — until turned ON.
                self._drain_park_if_requested()
                # Drain greedily until the queue is empty.
                try:
                    while not self._stop.is_set() and not self.drain_requested \
                            and self.run_once():
                        pass
                except Exception:
                    log.exception("[claim-worker:%s] run_once failed", self.queue)
                if self._stop.is_set():
                    break
                if self.drain_requested:
                    continue                 # park at the top of the loop, not idle-wait
                # Idle: block on the wake NOTIFY with a safety-poll timeout.
                for _ in listen_conn.notifies(
                    timeout=NOTIFY_POLL_TIMEOUT_S, stop_after=1,
                ):
                    break
        try:
            listen_with_reconnect(self._wake_channel, self._stop, _body)
        finally:
            # Stop the GPU VLM pool lane FIRST so the feeder stops claiming new
            # work before the rest of the teardown (no-op for cpu/ingest).
            self._stop_pool_lane()
            # Quiesce a widened cpu lane: in-flight jobs are WAITED for so a
            # SIGTERM drains their leases inside the stop_grace_period rather
            # than stranding N running rows (no-op at threads=1).
            self.shutdown_threads()
            if self._control_watcher is not None:
                self._control_watcher.stop()
                self._control_watcher = None
            # Stop the LLM-backend factory (invalidator thread + release any cached
            # backends → frees a vllm sidecar's VRAM). GPU only / best-effort, the
            # mirror of the gpu-gated start() above.
            if self.queue == "gpu":
                try:
                    from queue_workflows.llm_backends import factory as _llm_factory
                    _llm_factory.stop()
                except Exception:
                    log.exception(
                        "[claim-worker:gpu] LLM backend factory stop failed (ignored)"
                    )
            self.heartbeat.stop()
            # Stop the sampler iff THIS process won the flock.
            if self._hw_sampler is not None:
                self._hw_sampler.stop()
                self._hw_sampler.join(timeout=2.0)
                self._hw_sampler = None

    def stop(self) -> None:
        self._stop.set()


# ── entrypoint ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse
    import signal

    from queue_workflows import db

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="queue-claim-worker")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--lease-seconds", type=int, default=LEASE_S)
    parser.add_argument(
        "--threads", type=int, default=None,
        help="cpu queue only: node-jobs to run concurrently in THIS process "
             "(default 1 = the serial worker; env QUEUE_WORKFLOWS_CPU_WORKER_THREADS). "
             "For CPU-BOUND nodes the GIL serialises threads — prefer N processes "
             "(deploy/run_claim_lane.sh); raise this for IO-BOUND cpu nodes.",
    )
    args = parser.parse_args(argv)

    # Custom ingest queue names are accepted when the host configured them
    # (queue_workflows.configure(ingest_queues=...)) before calling main(); the
    # bare console script keeps the cpu/gpu/fetch/load default set.
    valid = _NODE_QUEUES | get_config().ingest_queues
    if args.queue not in valid:
        parser.error(
            f"--queue must be in {sorted(valid)} "
            "(call queue_workflows.configure(ingest_queues=...) for custom names)"
        )

    # Argument validation BEFORE any host work — an invalid combination must not
    # first run the model registrar.
    if args.threads is not None and args.queue != "cpu":
        parser.error("--threads applies to --queue=cpu only (gpu owns its own "
                     "PAR budget; ingest queues are single-job by contract)")

    if args.queue == "gpu":
        # Register the model registry up front so the first claim's
        # require_model resolves. The registrar is the engine's configured hook
        # (config.builtin_model_registrar) — a no-op unless a host wired one.
        get_config().builtin_model_registrar()

    worker = ClaimWorker(
        queue=args.queue, lease_s=args.lease_seconds, threads=args.threads,
    )

    def _handler(signum, _frame):
        log.info("[claim-worker:%s] signal %s; stopping", args.queue, signum)
        worker.stop()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    try:
        worker.run_forever()
    except db.DbUnreachable as exc:
        # Not a misconfiguration: the database may well come back. Exit cleanly (no
        # traceback) so the supervisor restarts us, which is what asking for a bounded
        # wait means.
        log.warning("[claim-worker:%s] %s", args.queue, exc)
        return 1
    except db.DbMisconfigured as exc:
        # DISTINCT from every other exit: a restart cannot fix a wrong DSN, so say so
        # once and stop rather than joining the supervisor's restart loop. 2 is the
        # usage-error code — "fix your configuration", next to the watchdog codes
        # 75-79 in docs/watchdogs.md.
        log.error("[claim-worker:%s] %s", args.queue, exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
