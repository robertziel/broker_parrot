"""Node-per-job queue — DB helpers.

Thin SQL layer on top of the ``workflow_node_jobs`` table. All queries
use the psycopg connection pool established in :mod:`queue_workflows.db`
so they interoperate with the rest of the engine.

This module is pure plumbing — no subprocess spawning, no torch. Used
by:

- the dispatcher (to enqueue newly-satisfied nodes),
- the CPU pool (to claim + finalise rows),
- the GPU pool (to claim + finalise rows, with a "prefer current
  model" order),
- a host's queue-snapshot view via a summary query.

The ingest task set (``run_fetch_all`` / etc.) is HOST-CONFIGURABLE:
``enqueue_ingest_job`` validates ``task_name`` against the host's
registered ``config.ingest_task_map`` (an empty map by default), not a
hard-coded host-domain frozenset. ``INGEST_QUEUES`` (``fetch``/``load``)
stays an engine constant — it matches the migration-0007 queue CHECK, which is
generic.
"""

from __future__ import annotations

import json
import logging
import os

from queue_workflows.envcompat import env_get
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from queue_workflows.db import connection

log = logging.getLogger(__name__)

#: The serving KIND stamped on an in-process model row (required_model, no
#: server_types): the job loads weights into the claiming process itself — a
#: serving kind of its own, width 1 like any non-batching kind. Same literal the
#: residency layer probes (``llm_probe.INPROCESS``); kept as a literal because
#: importing llm_probe here would weigh down every consumer of this leaf-ish
#: module — a drift-guard test asserts the two stay equal.
INPROCESS_SERVER_KIND = "inprocess"

#: Prediction fallback for an undeclared row when the caller supplies no
#: ``default_server`` (old client code on a new wheel). Mirrors
#: ``worker_control.DEFAULT_LLM_SERVER_TYPE`` — a literal because worker_control
#: imports THIS module (cycle); its own drift-guard test pins the pair.
_DEFAULT_SERVER_KIND = "ollama"


# ── Types ────────────────────────────────────────────────────────────────


QueueName = str  # 'cpu' | 'gpu' | 'download' — soft type; validated HOST-SIDE
# (enqueue_node_job); migration 0025 dropped the DB CHECK, 0008-style.
NodeStatus = str  # queued|running|completed|failed|cancelled|awaiting_input


# ── Helpers ──────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_json(value: Any) -> Any:
    """psycopg Jsonb shortcut — kept tiny so callers don't import psycopg.types."""
    from psycopg.types.json import Jsonb
    return Jsonb(value or {})


def _project(project: str | None) -> str:
    """Resolve the tenant tag for an enqueue/claim (migration 0017).

    ``None`` ⇒ this client's configured ``config.project`` (the common path: a
    per-project client sets it once at startup and every enqueue/claim inherits
    it). An explicit string overrides it (the dispatcher passes the parent run's
    project; tests pin it). Default ``""`` is the single-tenant sentinel — see
    :attr:`queue_workflows.config.EngineConfig.project`."""
    if project is not None:
        return project
    from queue_workflows.config import get_config
    return get_config().project or ""


# ── Enqueue ──────────────────────────────────────────────────────────────


#: The declared server kind of a CLUSTER RESERVATION: a gpu node-job that holds one box for a
#: server the engine did NOT start and does not run — e.g. an external multi-GPU vLLM spanning several
#: boxes, in containers of its own. No box serves (or advertises) this kind and none ever will: the row
#: asks for the BOX, not for a server. Everywhere a row's ``server_types`` is read as "which LLM
#: server must serve me", a reservation is therefore the exception:
#:
#:   * the claim's server-preference gate passes it (:func:`claim_next_gpu_job`). It is
#:     otherwise an ordinary pool-lane row, so ``force_box`` pins it, ``gpu_slots='all'``
#:     weighs it as the whole box, and the run-next flag makes a busy box drain for it;
#:   * a lane out of service for LLM work (a switch in flight, no GPU LLM server) still claims
#:     it (``reservations_only``);
#:   * the stranded sweep never flags it (:func:`flag_server_stranded_jobs`);
#:   * the executor resolves no LLM server for it (``node_executor.execute_node``);
#:   * the lane holding one keeps the broker's own LLM servers down for as long as it runs, and
#:     a box one is queued for starts none (``claim_worker.ClaimWorker._llm_starts_forbidden``,
#:     :func:`cluster_reservation_for`).
CLUSTER_SERVER_KIND = "cluster"


def is_cluster_reservation(row: Mapping[str, Any] | None) -> bool:
    """Does this node-job row declare :data:`CLUSTER_SERVER_KIND`? Reads a claimed row, a
    fetched row, or a ``server_types`` still in its JSON-text form."""
    if not isinstance(row, Mapping):
        return False
    kinds = row.get("server_types")
    if isinstance(kinds, str):
        try:
            kinds = json.loads(kinds)
        except ValueError:
            kinds = [kinds]
    if not isinstance(kinds, (list, tuple)):
        return False
    return any(str(k or "").strip().lower() == CLUSTER_SERVER_KIND for k in kinds)


def normalise_gpu_slots(value) -> str | None:
    """The stored form of a ``gpu_slots`` declaration, or ``ValueError``.

    ``None``/``1`` → ``None`` (an ordinary row; NULL so "1" and "unset" cannot drift),
    an integer >= 2 → its decimal text, ``"all"``/``"half"`` (any case, padded) → the
    word. Everything else — 0, negatives, floats, other words — is refused HERE, before
    the write, because the claim SQL trusts the column's shape the way 0024 trusts
    ``cpu_slots``: the only route to nonsense is out-of-band.
    """
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"gpu_slots must be an int >= 1, 'half' or 'all', got {value!r}")
    if isinstance(value, str):
        word = value.strip().lower()
        if word in ("all", "half"):
            return word
        if not word.isdigit():
            raise ValueError(f"gpu_slots must be an int >= 1, 'half' or 'all', got {value!r}")
        n = int(word)
    elif isinstance(value, int):
        n = value
    else:
        raise ValueError(f"gpu_slots must be an int >= 1, 'half' or 'all', got {value!r}")
    if n < 1:
        raise ValueError(f"gpu_slots must be >= 1, got {value!r}")
    return None if n == 1 else str(n)


def enqueue_node_job(
    *,
    run_id: str,
    node_id: str,
    node_module: str,
    queue: QueueName,
    required_model: str | None = None,
    required_models: Sequence[dict[str, Any]] | None = None,
    inputs: dict[str, Any] | None = None,
    priority: int = 100,
    pipeline_name: str | None = None,
    project: str | None = None,
    avoid_box: Iterable[str] | None = None,
    force_box: Iterable[str] | None = None,
    server_types: Iterable[str] | None = None,
    cpu_slots: int | None = None,
    dedup_key: str | None = None,
    context_length: int | None = None,
    gpu_slots: int | str | None = None,
) -> str:
    """Insert a fresh ``queued`` node-job row. Returns the row id.

    ``context_length`` (migration 0028) is the LLM context window this job's requests
    need, in tokens (prompt + completion). ``None`` ⇒ stored NULL = no requirement,
    byte-identical to pre-0028. A sized row is only claimed by a box whose LLM server
    serves at least that window (``claim_next_gpu_job(serving_context=…)``) — the
    alternative is ollama silently truncating the prompt. gpu rows only; a positive
    ``int`` or ``None``, anything else is a ``ValueError`` before the INSERT.

    ``gpu_slots`` (migration 0030) is how much of the box's GPU budget this row occupies
    at the box-slot arbiter: ``None``/``1`` = an ordinary job (stored NULL, so the two
    spellings can never drift), an integer >= 2, ``"half"`` or ``"all"`` — the last two
    resolved against the CLAIMING box's own ``max_running``, since a whole box is a
    different number on a PAR-2 and a PAR-4 machine. gpu rows only; anything else is a
    ``ValueError`` before the INSERT. It is the ``cpu_slots`` shape on the GPU lane, for
    a row that legitimately owns the whole server (a gateway's lease on the box's vLLM)
    and must not have the fleet co-scheduled into it.

    ``dedup_key`` (migration 0025) is an opt-in, opaque OUTPUT-IDENTITY string
    — "this job produces X". When a node-job with the same (project,
    dedup_key) has already COMPLETED, ``execute_node`` finalizes this one as
    completed without invoking the body (see
    :func:`find_completed_duplicate`): one file is never downloaded twice.
    ``None``/empty ⇒ stored NULL = no dedup, byte-identical to pre-0025. The
    engine only compares strings, so a host whose outputs are box-local must
    encode the location in the key (or omit it).

    ``cpu_slots`` (migration 0024) is how many slots of a WIDENED cpu lane this
    one job occupies — the weight for a node whose body runs its own threads.
    ``None``/``1`` ⇒ stored NULL = an ordinary 1-slot job, byte-identical to
    pre-0024. A serial worker (``cpu_worker_threads=1``, the default) ignores it
    entirely: one job at a time already owns the whole process. See
    :func:`claim_next_cpu_job` for the fit test.

    ``server_types`` (migration 0022) is an ORDERED LLM-server preference —
    ``['vllm', 'ollama']`` reads "serve me on vLLM, fall back to ollama". Empty /
    ``None`` ⇒ stored NULL = unconstrained. Two effects, both in the claim: the row
    is only claimable by a worker whose served kinds INTERSECT the list, and a row
    with a preference is POOL-eligible (an EXTERNAL server holds the model) even
    when it names a ``required_model``. See :func:`claim_next_gpu_job`.

    ``avoid_box`` / ``force_box`` (migration 0020) are optional PHYSICAL-BOX
    placement constraints, by BOX NAME (``gpu_model_lease.default_box_id()`` —
    NOT ``host_label``): the job may run ONLY on a box in ``force_box`` (when set)
    and NEVER on a box in ``avoid_box``. Empty / ``None`` ⇒ stored NULL =
    unconstrained (every box eligible). The claim SQL enforces both.

    ``project`` (migration 0017) is the tenant tag stamped on the row; ``None``
    ⇒ this client's ``config.project``. The dispatcher passes the parent run's
    project so the node inherits it. See :func:`_project`.

    Invariants:
        - queue in {'cpu', 'gpu', 'download'} — validated HERE (host-side;
          migration 0025 dropped the DB CHECK, the 0008 move).
        - only queue='gpu' MAY carry required_model (cpu/download tasks don't
          use the model cache).
        - (run_id, node_id) UNIQUE — one row per DAG cell.

    ``download`` (migration 0025) is the network-WAIT lane: full DAG
    citizenship (deps, placement, lease/watchdog), but its claim enforces a
    fleet-wide per-project width cap — see :func:`claim_next_download_job`.

    Raises ``ValueError`` before hitting the DB on a bad queue name
    or a non-gpu row with a stray model.
    """
    if queue not in ("cpu", "gpu", "download"):
        raise ValueError(
            f"queue must be 'cpu', 'gpu' or 'download', got {queue!r}")
    if queue != "gpu" and required_model:
        raise ValueError(f"{queue} node-job must not set required_model")
    _context = validate_context_length(context_length)
    if queue != "gpu" and _context is not None:
        # No LLM serves a cpu/download row, so a size there is a schema mistake — and
        # silently dropping it would hide the mistake.
        raise ValueError(
            f"{queue} node-job must not set context_length (only gpu rows are LLM-served)")
    # Reject a 0/negative width at the API edge: the fit test would be
    # meaningless and a 0-weight row would let a lane admit unboundedly many.
    if cpu_slots is not None and int(cpu_slots) < 1:
        raise ValueError(f"cpu_slots must be >= 1, got {cpu_slots!r}")
    # 1 IS the default — store it as NULL so the two spellings can never drift.
    _slots = int(cpu_slots) if cpu_slots is not None and int(cpu_slots) > 1 else None
    _gslots = normalise_gpu_slots(gpu_slots)
    if queue != "gpu" and _gslots is not None:
        # Only gpu rows pass the box-slot arbiter, so a weight elsewhere is a schema
        # mistake — and silently dropping it would hide the mistake.
        raise ValueError(f"{queue} node-job must not set gpu_slots (only gpu rows are weighed)")

    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    # Empty ⇒ NULL (unconstrained); a stray empty force_box must never strand the row.
    _avoid = [b for b in (avoid_box or []) if b] or None
    _force = [b for b in (force_box or []) if b] or None
    # Same rule for the server preference — an empty list must mean "any server",
    # never "no server", which would make the row permanently unclaimable. ORDER IS
    # SIGNIFICANT (it IS the preference), so this filters without sorting.
    _servers = [s for s in (server_types or []) if s] or None
    # DUAL-WRITE (0026). `required_model` is set to the array's FIRST model so a worker
    # that has not been bumped yet still claims this row correctly — that is what makes
    # the rollout flag-day-free. `required_models` carries the full ordered structure and
    # `required_model_ids` is its flat claim index; all three are written HERE, together,
    # so they cannot drift apart at any other call site.
    _models = [e for e in (required_models or [])
               if isinstance(e, dict) and str(e.get("model") or "").strip()] or None
    _model_ids = [str(e["model"]).strip() for e in _models] if _models else None
    if _models and not required_model:
        required_model = _model_ids[0]
    # Empty/whitespace ⇒ NULL: a blank key must mean "no dedup", never a
    # universal twin every other blank-keyed job would match.
    _dedup = (str(dedup_key).strip() if dedup_key is not None else "") or None

    row_id = str(uuid.uuid4())
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO workflow_node_jobs
                (id, run_id, pipeline_name, node_id, node_module,
                 queue, required_model, project,
                 status, priority, inputs, context_delta,
                 avoid_box, force_box, server_types, cpu_slots, dedup_key,
                 required_models, required_model_ids, context_length,
                 gpu_slots, created_at)
            VALUES
                (%s, %s, %s, %s, %s,
                 %s, %s, %s,
                 'queued', %s, %s, '{}'::jsonb,
                 %s, %s, %s, %s, %s,
                 %s, %s, %s,
                 %s, now())
            """,
            (
                row_id, run_id, pipeline_name, node_id, node_module,
                queue, required_model, _project(project),
                priority, _as_json(inputs or {}),
                _d.array_literal(_avoid) if _avoid else None,
                _d.array_literal(_force) if _force else None,
                _d.array_literal(_servers) if _servers else None,
                _slots,
                _dedup,
                _as_json(_models) if _models else None,
                _d.array_literal(_model_ids) if _model_ids else None,
                _context,
                _gslots,
            ),
        )
    return row_id


def validate_context_length(value: Any) -> int | None:
    """``None`` or a positive ``int`` — the only values a ``context_length`` may take.

    Strict at this API edge on purpose: a bool, a float or a numeric STRING is refused
    rather than coerced, because a context requirement that silently became "no
    requirement" (or the wrong number) is exactly the truncation it exists to stop. The
    dispatcher's schema layer, which reads JSON, does its own lenient coercion first."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"context_length must be a positive integer (tokens) or None, got {value!r}")
    return int(value)


def insert_skipped_job(
    *,
    run_id: str,
    node_id: str,
    pipeline_name: str | None = None,
    project: str | None = None,
) -> str:
    """Insert a status='skipped' marker row for a node whose
    ``skip_if`` evaluated to true at ready-check time. No worker
    touches it; it exists so dependents see a satisfied predecessor
    via ``_find_ready_nodes``.

    queue='cpu' + required_model NULL satisfies the
    ``workflow_node_jobs_required_model_check`` CHECK; node_module
    stays empty (no Python import) — the row is purely bookkeeping.
    """
    row_id = str(uuid.uuid4())
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO workflow_node_jobs
                (id, run_id, pipeline_name, node_id, node_module,
                 queue, required_model, project,
                 status, priority, inputs, context_delta,
                 created_at, finished_at)
            VALUES
                (%s, %s, %s, %s, '',
                 'cpu', NULL, %s,
                 'skipped', 100, '{}'::jsonb, '{}'::jsonb,
                 now(), now())
            """,
            (row_id, run_id, pipeline_name, node_id, _project(project)),
        )
    return row_id


def get_node_job(job_id: str) -> dict[str, Any] | None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM workflow_node_jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def list_jobs_for_run(run_id: str) -> list[dict[str, Any]]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM workflow_node_jobs "
            "WHERE run_id = %s ORDER BY created_at ASC",
            (run_id,),
        )
        return list(cur.fetchall())


def prioritize_node_job(job_id: str) -> dict[str, Any] | None:
    """Flag a QUEUED node job to "run next" — set ``is_priority`` so the next
    worker asking for a node in its queue + capability claims it before older +
    default-priority peers (``is_priority`` sorts FIRST in the claim ORDER BY,
    ahead of the priority band and the GPU warm-model affinity tiebreak).

    A no-op on a non-queued row — a job that's already running/terminal can't be
    re-ordered. Returns the updated row, or ``None`` if nothing was queued under
    that id (so the caller can tell the flag didn't take).
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET is_priority = TRUE "
            "WHERE id = %s AND status = 'queued' "
            "RETURNING *",
            (job_id,),
        )
        return cur.fetchone()


# ── Claim / update ───────────────────────────────────────────────────────
#
# The live claim path for the Postgres-as-queue backend. These ``claim_next_*``
# helpers are the ``SELECT … FOR UPDATE SKIP LOCKED`` queue: a single atomic
# ``queued → running`` UPDATE that also stamps a lease and folds in the
# run-cancel guard. The ``claim_worker`` loops call them on every wake.
#
# Default lease: 600 s. Independent of job duration — a live worker
# renews its lease (~every 10 s) while running, so a multi-hour job
# keeps its lease via renewal; only a dead/wedged worker lets it
# lapse, at which point ``reclaim_expired_leases`` re-queues the row.
DEFAULT_LEASE_S = 600

# Staleness threshold for the orchestrator-side dead-worker detector
# (:func:`flag_stale_workers_holding_running_jobs`). A live claim worker
# refreshes its ``worker_heartbeats`` row every ``HEARTBEAT_INTERVAL_S`` (10 s);
# 3× that (30 s) is the same window a host's queue gauge uses to call a row stale,
# so a heartbeat older than this is a worker that stopped beating — wedged or
# dead. Env-overridable for ops tuning.
STALE_WORKER_AFTER_S = 30

# ANTI-STARVATION for the fill-before-spill pool gate: a box may DEFER a no-model
# claim to let a higher-ranked peer fill first (bin-packing) — but ONLY while the work
# is actually draining. If any claimable no-model job has been queued longer than this,
# deferral is OFF for everyone and the job is claimed NOW, no matter the ranking. This
# is the liveness guarantee that makes the gate deadlock-proof: the DB-only capacity view
# is BLIND to cross-project box occupancy (a top-ranked box whose card is held for
# another project's diffusion model can only SPILL the LLM work, yet still reads as
# "free capacity"), so without a starvation break the lower boxes would defer forever
# while the work bounces on the occupied box. Short by design — long enough for a healthy
# top box to drain a cycle, short enough that a stuck one never starves the fleet.
# Env-overridable (``QUEUE_WORKFLOWS_VLM_POOL_MAX_DEFER_AGE_S``).
VLM_POOL_MAX_DEFER_AGE_S = int(
    env_get("QUEUE_WORKFLOWS_VLM_POOL_MAX_DEFER_AGE_S", "20") or "20"
)


def _stale_worker_after_s() -> int:
    raw = (env_get("QUEUE_WORKFLOWS_STALE_WORKER_AFTER_S", "") or "").strip()
    if not raw:
        return STALE_WORKER_AFTER_S
    try:
        return max(1, int(float(raw)))
    except (TypeError, ValueError):
        return STALE_WORKER_AFTER_S


def _host_dir(host_priority: int) -> int:
    """Direction sign for the ``created_at`` tiebreak, derived from the
    claiming worker's ``host_priority``.

    The baseline (``host_priority >= 0``) walks the band **oldest-first**
    → claims the **head** of the queue (``+1``). Only an explicit overflow
    host (``host_priority < 0``) sorts last: it walks newest-first → yields
    the head to the priority hosts and claims the **tail** (``-1``).

    Returns ``+1`` or ``-1``; the value is multiplied into
    ``EXTRACT(EPOCH FROM created_at)`` in the ORDER BY.
    """
    return -1 if host_priority < 0 else 1


# The claim is one statement. The SKIP-LOCKED subselect picks the next
# claimable row; the outer UPDATE stamps running + lease + claimed_by.
# ``{order}`` is the only interpolation and is built from validated ints
# / fixed fragments below — never from caller-supplied strings.
#
# Run-cancel guard: the subselect additionally requires the parent run is
# NOT terminal-cancelled/failed, so a worker can't even claim a job whose
# run was cancelled out from under it.
#
# _RUN_CLAIMABLE_GUARD is THE single definition of "this job's parent run may
# still execute" — shared by the claim below and by every "is there claimable
# work?" view (``vlm_pool_should_defer``'s anti-starvation check). The two must
# never disagree: when the starvation view briefly required ``status =
# 'running'`` while the claim accepted anything not cancelled/failed,
# trigger-expanded runs stuck at ``'queued'`` aged invisibly and the whole
# fleet deferred a claimable queue into a standstill.
_RUN_CLAIMABLE_GUARD = "status NOT IN ('cancelled', 'failed')"

#: "the caller said nothing", distinct from "the caller said None" — which for the 0028
#: window columns is the difference between leaving a value alone and CLEARING it.
_UNSET: Any = object()

_CLAIM_SQL = """
UPDATE workflow_node_jobs AS j
SET status = 'running',
    started_at = now(),
    worker_lane = %(worker_lane)s,
    claimed_by = %(host)s,
    resolved_server = {stamp},
    resolved_model = {model_stamp},
    lease_expires_at = {lease_expr}
WHERE j.id = (
    SELECT c.id FROM workflow_node_jobs c
    WHERE c.queue = %(queue)s
      AND c.status = 'queued'
      AND c.project = %(project)s
      AND EXISTS (
          SELECT 1 FROM workflow_runs r
          WHERE r.id = c.run_id
            AND r.{run_guard}
      )
      {placement}
      {capability}
    ORDER BY {order}
    {skip_locked}
    LIMIT 1
)
RETURNING *
"""


def _placement_terms(alias: str = "c") -> str:
    """The box-placement WHERE fragment (migration 0020) — shared by the cpu + gpu
    claims (alias ``c``) AND by every "is there claimable work for this box?" view
    (the VLM-pool anti-starvation check, alias ``sj``), so eligibility can never be
    judged two different ways. Reads the ``%(box)s`` param (the claiming worker's
    ``default_box_id()``):

      * ``avoid_box`` — excluded when this box is listed;
      * ``force_box`` — required to be listed when the array is set.

    NULL arrays ⇒ no constraint (the ``IS NULL`` short-circuit), so an unconstrained
    row is byte-identical to pre-0020. A ``NULL`` box param (a worker that resolved no
    box) fails BOTH inner tests, so it claims ONLY unconstrained rows — the safe
    default (it can't verify a constraint it can't name). Dialect-portable
    (pg ``= ANY``, sqlite ``json_each``)."""
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    in_avoid = _d.array_contains_value(f"{alias}.avoid_box", "%(box)s")
    in_force = _d.array_contains_value(f"{alias}.force_box", "%(box)s")
    return (
        f"AND ({alias}.avoid_box IS NULL OR NOT ({in_avoid})) "
        f"AND ({alias}.force_box IS NULL OR ({in_force}))"
    )


def _affinity_term() -> str:
    """Warm-model affinity tiebreak (GPU): rows whose ``required_model`` equals
    the worker's currently-loaded model jump to the head of their priority band
    so consecutive same-model jobs don't reload. Null-safe (NULL=NULL matches)
    via the dialect (``IS NOT DISTINCT FROM`` on pg, ``IS`` on sqlite)."""
    from queue_workflows.dialect import get_dialect
    return (
        f"({get_dialect().not_distinct_from('c.required_model', '%(current_model)s')})"
        " DESC"
    )


def count_pending_for_server(
    required_model: str | None, server_type: str, *, project: str | None = None,
    any_model: bool = False, box: Any = _UNSET,
) -> int:
    """How many QUEUED node-jobs are waiting for ``server_type`` to serve
    ``required_model`` — the amortisation input for a cold start (migration 0022).

    Starting a vLLM costs minutes of weight loading, so it is only worth paying when
    enough work is queued behind it to spread the cost. This is that count: rows
    still ``queued``, naming this model, whose ``server_types`` preference includes
    this kind, in this tenant. Rows whose parent run is already terminal are excluded
    by the same guard the claim uses — a queue full of ghosts must not justify a
    cold start.

    Best-effort: any failure returns 0, which is the CONSERVATIVE direction (do not
    pay the cold start), so a DB blip can never trigger an expensive server start.

    ``box`` (placement, migration 0020) — the physical box asking. Passed, only rows that box
    could CLAIM count: a row pinned elsewhere (``force_box``) or avoiding it (``avoid_box``)
    can never run there, so it must not start a server there — the vLLM demand starter on
    every box used to count a lease pinned to another. ``None`` is a box that resolved no
    name, and like the claim it counts only unplaced rows. Not passed, placement is not
    consulted: the dispatch-time amortisation read asks about the queue, not a box."""
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    # ``any_model`` answers the BOX-level question ("is any vLLM work waiting?"),
    # which is what the lifecycle manager needs; the default answers the per-job one.
    model_term = (
        "" if any_model
        else "AND " + _d.not_distinct_from("c.required_model", "%(model)s")
    )
    placement = "" if box is _UNSET else _placement_terms("c")
    sql = f"""
        SELECT COUNT(*) AS n FROM workflow_node_jobs c
         WHERE c.status = 'queued'
           AND c.project = %(project)s
           {model_term}
           AND c.server_types IS NOT NULL
           AND {_d.arrays_overlap('c.server_types', '%(kinds)s')}
           AND EXISTS (SELECT 1 FROM workflow_runs r
                        WHERE r.id = c.run_id AND r.{_RUN_CLAIMABLE_GUARD})
           {placement}
    """
    params = {
        "project": _project(project),
        "model": required_model,
        "kinds": _d.array_param([server_type]),
        "box": None if box is _UNSET else (box or None),
    }
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return int((row or {}).get("n") or 0)
    except Exception:
        log.debug("[node-queue] pending-for-server count failed; assuming 0",
                  exc_info=True)
        return 0


def cluster_reservation_for(
    *, host: str | None, box: str | None, queue: str = "gpu", project: str | None = None,
) -> dict[str, Any] | None:
    """The CLUSTER RESERVATION (:data:`CLUSTER_SERVER_KIND`) this box answers to right now, or
    ``None``: one QUEUED that this box could claim (the claim's own placement, tenant and run
    guard), or one RUNNING under this lane's name (``claimed_by = host``). A queued one first.

    The read behind "may this box start an LLM server?"
    (``claim_worker.ClaimWorker._llm_starts_forbidden``): a box that is about to be given to an
    external server must not spend minutes loading one of its own into the memory that server
    needs, and a lane that restarted under a held reservation must not start one before it
    re-claims it. Also how a lane out of service for LLM work knows there is a reservation
    worth claiming (``_claim_pool``).

    Best-effort: any failure answers ``None``, the behaviour before reservations existed — the
    same fail-open direction as the park gate's control-plane read."""
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    sql = f"""
        SELECT c.id, c.run_id, c.status, c.claimed_by, c.created_at
          FROM workflow_node_jobs c
         WHERE c.queue = %(queue)s
           AND c.project = %(project)s
           AND c.server_types IS NOT NULL
           AND {_d.arrays_overlap('c.server_types', '%(reservation_kinds)s')}
           AND EXISTS (SELECT 1 FROM workflow_runs r
                        WHERE r.id = c.run_id AND r.{_RUN_CLAIMABLE_GUARD})
           AND ((c.status = 'queued' {_placement_terms('c')})
                OR (c.status = 'running' AND c.claimed_by = %(host)s))
         ORDER BY CASE WHEN c.status = 'queued' THEN 0 ELSE 1 END, c.created_at
         LIMIT 1
    """
    params = {
        "queue": queue,
        "project": _project(project),
        "host": host,
        "box": box or None,
        "reservation_kinds": _d.array_param([CLUSTER_SERVER_KIND]),
    }
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return dict(row) if row else None
    except Exception:
        log.debug("[node-queue] cluster-reservation read failed; assuming none",
                  exc_info=True)
        return None


def context_switch_need(
    *, host: str, serving: int | None, ceiling: int | None,
    box: str | None = None, project: str | None = None,
    stale_s: float | None = None, min_age_s: float = 30.0, elect: bool = True,
) -> int | None:
    """The context window THIS box should restart its LLM server at, or ``None``
    (migration 0028) — the demand half of the context switch.

    Restarting the server costs the box its whole out-of-service window, so a queued
    job asking for more than ``serving`` is not on its own a reason. Three more must
    hold, and each one is a failure this read exists to avoid:

      * NO live box of this tenant is ALREADY serving that window. One that is will
        claim the job within seconds; growing a second box for it is pure loss.
      * the job has waited ``min_age_s`` — below that, "nobody serves it" is as likely
        to be heartbeat lag as a real gap.
      * of the boxes that COULD reach it, this one is the right one: the TIGHTEST fit
        (smallest sufficient ``llm_context_max``, ties by name). Every box runs the
        same read against the same heartbeats, so they elect the same box without
        talking to each other — and the roomiest box stays free for the job that needs
        the room.

    A box that is PARKED (worker_controls OFF) or stale serves nothing, so it is
    neither a candidate nor allowed to mask the demand — else an operator switching the
    big box off would strand every sized job behind its last heartbeat.

    ``elect=False`` drops only the tightest-fit election, for the case where the server
    is being STARTED anyway (a kind switch, migration 0022): picking the window it comes
    up at costs nothing, so there is no box to elect — the alternative is starting it
    wrong and restarting it a minute later.

    Placement (0020) and tenancy are honoured exactly as the claim honours them: a job
    that could never run here must never restart this box's server. Best-effort — any
    failure answers ``None``, the conservative direction.
    """
    if ceiling is None:
        return None                     # cannot grow ⇒ nothing to ask for
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    election = """
           AND NOT EXISTS (
                 SELECT 1 FROM live l2
                  WHERE l2.llm_context_max >= c.context_length
                    AND (l2.llm_context_max < %(ceiling)s
                         OR (l2.llm_context_max = %(ceiling)s
                             AND l2.host_label < %(host)s)))
    """ if elect else ""
    window = float(stale_s) if stale_s is not None else float(_stale_worker_after_s())
    sql = f"""
        WITH live AS (
            SELECT w.host_label, w.llm_context_length, w.llm_context_max
              FROM worker_heartbeats w
             WHERE w.queue = 'gpu'
               AND w.project = %(project)s
               AND w.last_seen > now() - make_interval(secs => %(window)s)
               AND NOT EXISTS (
                     SELECT 1 FROM worker_controls wc
                      WHERE wc.host_label = w.host_label
                        AND wc.queue = 'gpu'
                        AND wc.project = %(project)s
                        AND wc.desired_state = 'off')
        )
        SELECT max(c.context_length) AS n
          FROM workflow_node_jobs c
         WHERE c.queue = 'gpu'
           AND c.status = 'queued'
           AND c.project = %(project)s
           AND c.context_length IS NOT NULL
           AND c.context_length > %(serving)s
           AND c.context_length <= %(ceiling)s
           AND c.created_at <= now() - make_interval(secs => %(min_age)s)
           {_placement_terms('c')}
           AND EXISTS (SELECT 1 FROM workflow_runs r
                        WHERE r.id = c.run_id AND r.{_RUN_CLAIMABLE_GUARD})
           AND NOT EXISTS (
                 SELECT 1 FROM live l
                  WHERE l.llm_context_length >= c.context_length)
           {election}
    """
    params = {
        "project": _project(project),
        "window": window,
        # NULL "I cannot state my window" means every sized job is beyond it.
        "serving": int(serving or 0),
        "ceiling": int(ceiling),
        "min_age": float(min_age_s),
        "host": host,
        "box": box,
    }
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    except Exception:
        log.debug("[node-queue] context-need read failed; asking for no switch",
                  exc_info=True)
        return None
    value = (row or {}).get("n")
    return int(value) if value else None


def _pool_row_term(alias: str = "c") -> str:
    """"This row is POOL-lane work" — the ONE definition, shared by the claim SQL and
    by every view that reasons about pool capacity or starvation.

    A row is pool work when it loads no in-process model: it names none, OR it is
    remote-served (``server_types`` set, migration 0022 — an external server holds
    the model). Before this was shared, the defer view still tested only
    ``required_model IS NULL``, so every remote-served job was invisible to it: peers
    looked idle (capacity count missed their running jobs) AND the anti-starvation
    break never armed (aging jobs weren't counted), so a box deferred forever while a
    2579-deep queue sat unclaimed. That is the defer-standstill class,
    and the cure is the same one: define claimability ONCE."""
    return f"({alias}.required_model IS NULL OR {alias}.server_types IS NOT NULL)"


def _warm_affinity_term() -> str:
    """Warm **(server, model)** affinity tiebreak (GPU, migration 0022) — the
    "switch servers and weights as rarely as possible" sort.

    Scores each candidate row against what is ALREADY loaded on this box and sorts
    the score DESC, so the claim prefers work that costs nothing to start:

    ======  ======================================  ==========================
    score   condition                               cost avoided
    ======  ======================================  ==========================
    3       server kind up **and** model resident   nothing to do
    2       server kind up, different model         a server start
    1       model resident, different server kind   a weight load
    0       neither                                 start **and** load
    ======  ======================================  ==========================

    The server term is weighted ×2 because starting a server (a vLLM cold start is
    minutes) dominates a weight load (ollama swaps a model in seconds).

    Complements — does not replace — :func:`_affinity_term`, which matches the
    worker's own IN-PROCESS ``current_model``. This one reads EXTERNAL servers, so
    a thin HTTP client with no in-process model still gets affinity.

    Both params degrade to empty lists, in which case every row scores 0 and the
    term is a no-op tiebreak — never a filter, so a cold box still claims."""
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    server_hit = _d.arrays_overlap("c.server_types", "%(warm_servers)s")
    model_hit = _d.value_in_param_array("c.required_model", "%(warm_models)s")
    # SCOPED TO REMOTE-SERVED ROWS ONLY. The whole score is gated on
    # ``server_types IS NOT NULL`` — including the MODEL half — so a row that
    # predates 0022 always scores 0 and this term cannot reorder it. Without that
    # gate the model half fired on every legacy row and, because the term sorts
    # ABOVE ``priority ASC``, warm-model affinity silently outranked the operator's
    # priority bands for work that never opted in. In-process warm-model affinity is
    # already owned by :func:`_affinity_term` (which matches the worker's own
    # ``current_model``), so nothing is lost by scoping this one.
    return (
        "(CASE WHEN c.server_types IS NULL THEN 0 ELSE "
        f"((CASE WHEN {server_hit} THEN 2 ELSE 0 END)"
        f" + (CASE WHEN c.required_model IS NOT NULL AND {model_hit} THEN 1 ELSE 0 END))"
        " END) DESC"
    )


def _host_dir_term() -> str:
    """host_priority tiebreak: multiply the creation epoch by ±1 so a high-
    priority host takes the oldest (head) and an overflow host the newest (tail).
    See :func:`_host_dir`. ``EXTRACT(EPOCH …)`` on pg / ``strftime`` on sqlite."""
    from queue_workflows.dialect import get_dialect
    return f"({get_dialect().creation_order('c')} * %(host_dir)s) ASC"


def claim_next_cpu_job(
    worker_lane: int = 0,
    *,
    host: str | None = None,
    lease_s: int = DEFAULT_LEASE_S,
    host_priority: int = 0,
    project: str | None = None,
    box: str | None = None,
    lane_width: int | None = None,
    free_slots: int | None = None,
) -> dict[str, Any] | None:
    """Atomically grab the next queued CPU job (the CPU claim worker's
    claim).

    ``lane_width`` / ``free_slots`` (migration 0024) are the CPU-WIDTH fit test,
    used only by a WIDENED lane (``cpu_worker_threads > 1``). ``lane_width`` is
    the worker's total slots, ``free_slots`` how many are free right now; a row is
    claimable when ``LEAST(cpu_slots, lane_width) <= free_slots``. Both ``None``
    (a serial worker, and every pre-0024 caller) ⇒ **no width term at all**: one
    job at a time already owns the process, so every weight is claimable there.

    The ``LEAST`` clamp is what stops a job wider than any worker in the fleet
    from being excluded forever — a ``cpu_slots=9`` row on a 3-wide lane can
    never "fit" arithmetically, so it is measured as 3 and simply runs alone. The
    alternative (excluding it) strands work no operator can see a reason for.

    STARVATION, NAMED: this is a FILTER, not a reservation. It has no aging and
    no priority override, so it silently outranks the ``ORDER BY`` — while the
    lane stays busier than ``threads - cpu_slots`` a wide row waits behind
    narrower ones no matter how high its ``is_priority``/``priority``. Give a
    latency-sensitive wide node its own lane rather than relying on priority.

    Stamps the lease (``claimed_by=host``,
    ``lease_expires_at=now()+lease_s``) and applies the run-cancel guard.
    CPU has no model cache, so there's no warm-affinity term; ordering
    is ``is_priority DESC`` (the operator "run next" flag jumps the queue),
    then ``priority ASC``, then the ``host_priority``-directed creation
    tiebreak.

    ``box`` (migration 0020) is the claiming worker's PHYSICAL box name; the claim
    skips rows this box is ineligible for (``avoid_box`` / ``force_box``). ``None``
    ⇒ claim only unconstrained rows."""
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    order = f"c.is_priority DESC, c.priority ASC, {_host_dir_term()}"
    # CPU WIDTH fit (migration 0024) — only for a widened lane. Absent params ⇒
    # empty fragment ⇒ the statement is byte-identical to pre-0024, which is what
    # keeps every serial worker unaffected.
    width = ""
    if lane_width is not None and free_slots is not None:
        # GREATEST(..., 1) floors the weight the same way the worker does: a
        # negative ``cpu_slots`` (only reachable out-of-band — the API raises on
        # < 1) would otherwise satisfy this test unconditionally AND drag the
        # worker's running total down, so the lane would admit more claims than
        # it has threads. Both clamps live on both sides so the SQL and the
        # in-process accounting can never disagree about a row's weight.
        width = ("AND LEAST(GREATEST(COALESCE(c.cpu_slots, 1), 1), "
                 "%(lane_width)s) <= %(free_slots)s ")
    # CPU jobs carry no required_model, so no capability gate applies.
    sql = _CLAIM_SQL.format(
        order=order, capability=width, placement=_placement_terms(),
        run_guard=_RUN_CLAIMABLE_GUARD,
        lease_expr=_d.future_seconds("%(lease_s)s"), skip_locked=_d.skip_locked,
        stamp="NULL",   # cpu rows have no LLM serving kind (migration 0023)
        model_stamp="NULL",   # ...and therefore no resolved model either (0026)
    )
    params = {
        "worker_lane": worker_lane,
        "host": host,
        "lease_s": int(lease_s),
        "queue": "cpu",
        "project": _project(project),
        "host_dir": _host_dir(host_priority),
        "box": box or None,
    }
    if width:
        params["lane_width"] = int(lane_width)
        params["free_slots"] = int(free_slots)
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def resolve_download_max_concurrent() -> int:
    """The effective download-lane width, resolved at CALL time so a host's
    ``configure()`` (which runs after import) is honoured: the
    ``QUEUE_WORKFLOWS_DOWNLOAD_MAX_CONCURRENT`` env knob wins (operator
    override, read via envcompat), then
    ``EngineConfig.download_max_concurrent``, then the engine default **1**.

    Always ``>= 1``: a malformed or sub-1 value falls through to the floor —
    a 0 cap would be a lane that admits nothing, silently, forever.

    Mirrors :func:`queue_workflows.claim_worker.resolve_cpu_worker_threads`."""
    raw = env_get("QUEUE_WORKFLOWS_DOWNLOAD_MAX_CONCURRENT")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    try:
        from queue_workflows.config import get_config
        val = getattr(get_config(), "download_max_concurrent", None)
        if val is not None:
            return max(1, int(val))
    except Exception:
        pass
    return 1


def claim_next_download_job(
    worker_lane: int = 0,
    *,
    host: str | None = None,
    lease_s: int = DEFAULT_LEASE_S,
    host_priority: int = 0,
    project: str | None = None,
    box: str | None = None,
    max_concurrent: int | None = None,
) -> dict[str, Any] | None:
    """Atomically grab the next queued DOWNLOAD job — with THE LANE CAP
    (migration 0025).

    The download lane exists for two guarantees at once: network WAIT never
    occupies a cpu/gpu claim slot, and the shared uplink is never saturated —
    at most ``max_concurrent`` download rows (engine default **1**) may be
    ``running`` at a time, counted **fleet-wide per project**, NOT per host.
    Contrast the gpu box-slot arbiter (``max_running``), which scopes its count
    to ``claimed_by = host``: here ANY box's running download occupies the
    lane, because the resource being protected (the uplink) is shared by all
    of them. Cross-project downloads don't contend in this count — tenant
    isolation (0017) wins over uplink pessimism, and a single-tenant deploy
    (project ``''``) gets a true global cap anyway.

    ALWAYS ON: ``max_concurrent=None`` resolves the knob
    (:func:`resolve_download_max_concurrent`) rather than skipping the gate,
    so no call path can forget the lane's whole point. An explicit value wins
    (tests; a host that computes its own width).

    RACE-SAFETY: the cap is a COUNT guard, so two
    concurrent claimants can both read "cap-1 running" under MVCC and both
    take the last slot. The (queue, project)-keyed advisory xact lock
    (:meth:`~queue_workflows.dialect.Dialect.queue_claim_serializer`)
    serializes them — keyed WITHOUT the host, unlike the box serializer,
    precisely because the contenders are on different boxes. SQLite's single
    writer needs none.

    RECOVERY: a dead downloader's slot frees when its row leaves ``running`` —
    the ordinary lease reclaim (:func:`reclaim_expired_leases`), no
    lane-specific machinery. A zombie row therefore holds the lane until the
    reclaim sweep lands, the same accepted semantics as the box-slot arbiter.

    Ordering, placement and the run-cancel guard are the cpu claim's:
    ``is_priority DESC, priority ASC``, the ``host_priority``-directed
    creation tiebreak; ``avoid_box``/``force_box`` honoured; no serving-kind
    stamp (``resolved_server`` stays NULL — nothing serves an LLM here)."""
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    cap = (resolve_download_max_concurrent()
           if max_concurrent is None else max(1, int(max_concurrent)))
    order = f"c.is_priority DESC, c.priority ASC, {_host_dir_term()}"
    # THE LANE CAP — fleet-wide per project: no claimed_by/host term on
    # purpose (see the docstring). Counted over the same statement that flips
    # the row, so the guard and the claim can never disagree.
    capability = (
        "AND (SELECT COUNT(*) FROM workflow_node_jobs r "
        "WHERE r.status = 'running' AND r.queue = %(queue)s "
        "AND r.project = %(project)s) < %(max_concurrent)s "
    )
    sql = _CLAIM_SQL.format(
        order=order, capability=capability, placement=_placement_terms(),
        run_guard=_RUN_CLAIMABLE_GUARD,
        lease_expr=_d.future_seconds("%(lease_s)s"), skip_locked=_d.skip_locked,
        stamp="NULL",   # download rows have no LLM serving kind (0023)
        model_stamp="NULL",   # ...nor a resolved model (0026)
    )
    params = {
        "worker_lane": worker_lane,
        "host": host,
        "lease_s": int(lease_s),
        "queue": "download",
        "project": _project(project),
        "host_dir": _host_dir(host_priority),
        "box": box or None,
        "max_concurrent": cap,
    }
    with connection() as conn, conn.cursor() as cur:
        # Serialize claims fleet-wide per (queue, project) — any
        # count-based arbiter needs the lock, or two claimants both read
        # "cap-1 running" and both take the last slot.
        ser = _d.queue_claim_serializer()
        if ser:
            cur.execute(ser, params)
        cur.execute(sql, params)
        return cur.fetchone()


def find_completed_duplicate(
    *,
    dedup_key: str | None,
    project: str | None = None,
    exclude_id: str | None = None,
) -> dict[str, Any] | None:
    """The most recent COMPLETED node-job with this (project, ``dedup_key``) —
    the "one file is never downloaded twice" lookup (migration 0025).

    ``execute_node`` calls this before invoking a keyed job's body; a hit means
    the OUTPUT already exists, so the new job finalizes as completed with the
    twin's ``context_delta`` instead of re-doing the work. Only ``completed``
    counts — a failed/cancelled twin proves nothing about the output, and a
    ``running`` twin is not yet proof (under the download lane's cap-1 the
    same-key jobs serialize, so by the time the second executes the first IS
    terminal; with a wider cap two same-key jobs may both run — dedup is a
    short-circuit, not a lock, and the body's own idempotency is the second
    layer).

    Scoped to one tenant (0017): a ``project`` of ``None`` resolves to this
    client's configured project, mirroring every claim. ``exclude_id`` guards
    against self-matching (the caller's own row can never be its twin)."""
    key = (str(dedup_key).strip() if dedup_key is not None else "") or None
    if key is None:
        return None
    sql = (
        "SELECT * FROM workflow_node_jobs "
        "WHERE dedup_key = %(key)s AND project = %(project)s "
        "AND status = 'completed'"
    )
    params: dict[str, Any] = {"key": key, "project": _project(project)}
    if exclude_id:
        sql += " AND id != %(exclude_id)s"
        params["exclude_id"] = exclude_id
    sql += " ORDER BY finished_at DESC LIMIT 1"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def claim_next_gpu_job(
    worker_lane: int = 0,
    current_model: str | None = None,
    *,
    host: str | None = None,
    lease_s: int = DEFAULT_LEASE_S,
    host_priority: int = 0,
    known_models: Iterable[str] | None = None,
    require_model: bool | None = None,
    pool_modules: Iterable[str] | None = None,
    project: str | None = None,
    max_running: int | None = None,
    box: str | None = None,
    served_types: Iterable[str] | None = None,
    warm_servers: Iterable[str] | None = None,
    warm_models: Iterable[str] | None = None,
    kind_caps: Mapping[str, int] | None = None,
    default_server: str | None = None,
    serving_context: int | None = None,
    reservations_only: bool = False,
) -> dict[str, Any] | None:
    """Atomically grab the next queued GPU job (the GPU claim worker's
    claim).

    ``reservations_only`` — claim nothing but a CLUSTER RESERVATION
    (:data:`CLUSTER_SERVER_KIND`). A lane that is out of service for LLM work (a server
    switch in flight, no GPU LLM server answering) passes it: a reservation loads nothing,
    is served by no broker server, and holding it ends the switch. A reservation is also
    exempt from the ``served_types`` gate below — it asks for the BOX, and names a server
    no box runs — while placement, the weight and the run-next reservation bind it as they
    bind every row.

    ``serving_context`` (migration 0028) — the context window, in tokens, this box's
    LLM server serves right now. A row with a ``context_length`` is claimable only when
    it fits (``context_length <= serving_context``); ``None`` ("I cannot state my
    window") claims only unconstrained rows. Unconstrained rows — every row written
    before 0028 — are unaffected either way. The alternative to refusing here is ollama
    truncating the prompt to fit and answering 200 as if nothing happened.

    ``kind_caps`` (migration 0023; LAW since) — PER-KIND slot
    arbitration. The claim computes each candidate row's RESOLVED serving kind
    (first ``server_types`` element this box serves → first preference →
    ``'inprocess'`` for an undeclared model row → ``default_server``), STAMPS it
    on the row — NEVER NULL — and refuses the claim while this host already runs
    ``kind_caps[kind]`` jobs of that kind. ollama SERIALISES per model, so its
    cap is 1 however many total slots the box advertises; vLLM batches, so it
    gets PAR; every OTHER kind — comfyui, inprocess, custom, undeclared — falls
    to the width-1 default (the rule: only vLLM batches — ollama, comfyui and
    any custom server take 1 at a time). A running row stamped NULL (a pre-law
    wheel's claim) counts against EVERY candidate's kind: unknown is not free.
    ``kind_caps=None`` applies the engine defaults ({'vllm': max_running-or-1});
    the explicit empty dict ``{}`` is the documented opt-out, restoring the
    legacy single-number arbiter AND the legacy NULL stamp byte-for-byte.

    ``default_server`` — the caller's prediction of the kind an UNDECLARED
    no-model row will be served on: pass what dispatch's active-server-wins
    default will actually do (a warm vLLM ⇒ ``'vllm'``, else the box's cached
    configured default). ``None`` falls back to the engine default (ollama) —
    the conservative width-1 kind, never NULL.

    ``served_types`` (migration 0022) — the LLM server kinds THIS worker's box can
    actually serve (its probed ``llm_servers_available``). A row is claimable iff
    its ``server_types`` is NULL (unconstrained) or INTERSECTS this list, so an
    ollama-only box never grabs a vllm-only job. ``None`` / empty ⇒ the gate is
    off (claim-any), the same cold-worker escape hatch ``known_models`` uses: a
    worker that has not advertised yet must never wedge the queue.

    ``warm_servers`` / ``warm_models`` — what is ALREADY loaded on this box right
    now (from ``model_residency.probe_box_residents``). They drive the warm
    (server, model) affinity sort: a job whose server kind is already up scores
    above one that would need a server start, and within that a job whose model is
    already resident scores above one that would need a weight load. Purely an
    ORDER BY — never a filter, so nothing becomes unclaimable because a box is cold.

    ``box`` (migration 0020) is the claiming worker's PHYSICAL box name; the claim
    skips rows this box is ineligible for (``avoid_box`` / ``force_box``). ``None``
    ⇒ claim only unconstrained rows.

    ``max_running`` — THE BOX-SLOT ARBITER: when set, the claim is refused while
    this ``host`` already has that many ``running`` gpu rows (same project). The
    single centralized per-box concurrency gate every claim path funnels through;
    ``None`` keeps the legacy unbounded claim.

    Same lease + run-cancel guard as :func:`claim_next_cpu_job`, plus:

    * **Capability gate** — only claim a job whose ``required_model`` this
      worker can serve (in ``known_models``) or that needs no model. Restores
      the gate the old Celery ``_gpu_should_accept`` enforced (reject+requeue),
      as a cleaner claim-time filter so an incapable worker never grabs the row
      — a capable peer does. With no ``known_models`` (worker hasn't advertised
      its registry yet) it falls back to claim-any so a cold worker can't wedge
      the queue.
    * **Warm-model affinity tiebreak** — rows whose ``required_model`` matches
      the worker's ``current_model`` (``IS NOT DISTINCT FROM``) sort ahead of
      the integer ``priority`` band (only the operator "run next" ``is_priority``
      flag outranks affinity), so consecutive same-model jobs don't reload — a
      warm-model job preempts a colder one even from a worse priority band.
      ``host_priority`` then breaks the creation-order tie exactly as on CPU.
    * **Model-presence lane filter** (``require_model``) — splits the GPU queue
      into two disjoint claim sets so a two-lane GPU worker (inline warm-model
      diffusion lane + a PAR-sized no-model VLM pool lane) never over-claims or
      steals the other lane's rows. ``None`` (default) keeps the existing
      claim-any behaviour; ``True`` adds ``AND c.required_model IS NOT NULL``
      (model-backed diffusion jobs only); ``False`` adds
      ``AND c.required_model IS NULL`` (no-model GPU jobs — VLM/HTTP — only).
      Orthogonal to and ANDed with the capability gate above.
    * **VLM-pool eligibility** (``pool_modules``) — when non-empty, only these
      node modules are genuine VLM-facade (pool-safe). The POOL lane then claims
      a no-model job ONLY if its ``node_module`` is in ``pool_modules``; the
      INLINE lane additionally claims no-model jobs whose module is NOT in the
      set, so heavy in-process GPU work (erasers, detectors, builders) runs on
      the conc-1 serial lane instead of PAR-concurrently in the pool. Empty /
      unset ⇒ legacy split (every no-model GPU job is pool-eligible)."""
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    served = [s for s in (served_types or []) if s]
    order = (
        # The operator "run next" flag jumps the whole queue — ahead of the
        # warm-model affinity tiebreak too (a flagged cold-model node preempts a
        # warm one; the reload is the accepted cost of "run this next").
        f"c.is_priority DESC, "
        f"{_affinity_term()}, "
        f"{_warm_affinity_term()}, "
        f"c.priority ASC, "
        f"{_host_dir_term()}"
    )
    known = [m for m in (known_models or []) if m]
    pool = [m for m in (pool_modules or []) if m]

    def _eligibility(a: str) -> list[str]:
        """Every term that decides whether THIS claim may take a row EXCEPT what is already
        running here (the box-slot weight and the per-kind caps, appended below). Rendered
        per ALIAS because it is asked twice: of the candidate (``c``), and of any run-next
        wide row that would reserve the box (``q`` — THE RESERVATION below). A reservation
        is therefore judged by exactly the claim's own terms, and can never fire for a row
        this box could not take."""
        terms: list[str] = []
        if served:
            # SERVER-PREFERENCE gate (0022): unconstrained rows always pass; a row that
            # names a preference needs this box to serve at least one of those kinds —
            # except a CLUSTER RESERVATION, which asks for the box and names a server no
            # box runs (CLUSTER_SERVER_KIND). Placement and weight still bind it.
            terms.append(
                f"AND ({a}.server_types IS NULL OR "
                + _d.arrays_overlap(f"{a}.server_types", "%(served_types)s")
                + " OR " + _d.arrays_overlap(f"{a}.server_types", "%(reservation_kinds)s")
                + ")"
            )
        if known:
            # Capability gate over BOTH job shapes (0026). The scalar path is unchanged; the
            # array path passes when ANY acceptable model is one this box serves — which is
            # the entire point of the array, and would be lost if only the dual-written head
            # (`required_model`) were tested. `required_model_ids` is the flat denormalised
            # index so this stays one cheap `arrays_overlap`, identical on both dialects.
            terms.append(
                f"AND ({a}.required_model IS NULL "
                "OR " + _d.value_in_param_array(f"{a}.required_model", "%(known_models)s")
                + f" OR ({a}.required_model_ids IS NOT NULL AND "
                + _d.arrays_overlap(f"{a}.required_model_ids", "%(known_models)s") + "))"
            )
        # CONTEXT GATE (0028). A sized row needs a box whose served window covers it; a box
        # that cannot state its window takes only rows that name none. Evaluated per
        # candidate inside the claim, so a row too big for THIS box is skipped, not blocking
        # the rows queued behind it.
        if serving_context is None:
            terms.append(f"AND {a}.context_length IS NULL")
        else:
            terms.append(
                f"AND ({a}.context_length IS NULL OR {a}.context_length <= %(serving_context)s)")
        # REMOTE-SERVED (0022): a row naming ``server_types`` is executed by an EXTERNAL
        # server, so it loads NO model into this process — it belongs in the PAR-wide
        # pool even though it names a ``required_model``. Before 0022 the two lanes were
        # split purely on ``required_model IS NULL``, which forced every model-naming LLM
        # job onto the concurrency-1 inline lane meant for in-process weights. NULL
        # ``server_types`` reduces to the old predicate exactly, so existing rows are
        # byte-identical.
        _remote = f"{a}.server_types IS NOT NULL"
        if require_model is True:
            if pool:
                # Inline lane: in-process model jobs PLUS no-model jobs whose module is
                # NOT VLM-pool-eligible — heavy in-process GPU work (erasers,
                # detectors, builders) runs conc-1 here, never concurrently in the
                # pool. A remote-served row is excluded on both counts.
                terms.append(
                    f"AND NOT ({_remote}) "
                    f"AND ({a}.required_model IS NOT NULL "
                    "OR NOT (" + _d.value_in_param_array(f"{a}.node_module", "%(pool_modules)s")
                    + "))"
                )
            else:
                terms.append(
                    f"AND NOT ({_remote}) AND {a}.required_model IS NOT NULL"
                )
        elif require_model is False:
            if pool:
                # Pool lane: a no-model job whose module is VLM-pool-eligible, OR any
                # remote-served row.
                #
                # The ``OR remote`` sits OUTSIDE the pool_modules test on purpose. The
                # inline lane refuses every remote-served row, so if the pool lane also
                # required pool membership, a remote-served row whose module is not
                # listed would be claimable by NEITHER lane — queued forever, with no
                # error and no unassignable stamp. ``pool_modules`` exists to identify
                # work that loads nothing in-process; ``server_types IS NOT NULL`` states
                # that directly, so it needs no second opinion.
                terms.append(
                    f"AND (({a}.required_model IS NULL AND "
                    + _d.value_in_param_array(f"{a}.node_module", "%(pool_modules)s")
                    + f") OR {_remote})"
                )
            else:
                terms.append(f"AND ({a}.required_model IS NULL OR {_remote})")
        return terms

    capability_terms = _eligibility("c")
    if reservations_only:
        # A lane out of service for LLM work still takes a CLUSTER RESERVATION — and only
        # that (see the docstring and claim_worker._claim_pool).
        capability_terms.append(
            "AND c.server_types IS NOT NULL AND "
            + _d.arrays_overlap("c.server_types", "%(reservation_kinds)s"))

    def _running_here(alias: str) -> str:
        """The this-host running-rows scope EVERY arbiter counts over. The box-slot
        gate and the per-kind width gate must mean the same population, or two
        arbiters can disagree their way past each other."""
        return (
            f"FROM workflow_node_jobs {alias} "
            f"WHERE {alias}.status = 'running' AND {alias}.queue = %(queue)s "
            f"AND {alias}.claimed_by = %(host)s AND {alias}.project = %(project)s"
        )

    if max_running is not None and host:
        # THE BOX-SLOT ARBITER — the one centralized per-box concurrency gate.
        # Refuse the claim inside the claim statement itself when this box already
        # has >= max_running rows running (same queue + project). Every claim path
        # funnels here — inline lane, pool lane, a second process — so process-local
        # counters can no longer disagree their way into a 2/1 over-claim, and a
        # zombie row (a thread that died without finalizing) keeps its slot
        # occupied until the lease reclaim frees it instead of being stacked on.
        # WEIGHTED (migration 0030): every row used to count as 1, so a row that owns the
        # whole server (a gateway's lease on the box's vLLM) left PAR-1 slots open and the
        # fleet co-scheduled into the very server the lease was taken to hold. ``all`` and
        # ``half`` resolve against ``max_running`` — already a parameter of this statement,
        # so a whole box costs no lookup — and an integer is measured as LEAST(n, budget),
        # exactly as cpu_slots is, so an over-wide row runs alone instead of never fitting.
        # NULL weighs 1: every pre-0030 row is admitted precisely as before.
        # ``COUNT(*) < budget`` and ``SUM(1) + 1 <= budget`` are the same test; the second
        # is the one that generalises.
        def _gpu_weight(alias: str) -> str:
            n = _d.least(_d.greatest(f"CAST({alias}.gpu_slots AS INTEGER)", "1"),
                         "%(max_running)s")
            return (f"CASE WHEN {alias}.gpu_slots IS NULL THEN 1 "
                    f"WHEN {alias}.gpu_slots = 'all' THEN %(max_running)s "
                    f"WHEN {alias}.gpu_slots = 'half' THEN ((%(max_running)s + 1) / 2) "
                    f"ELSE {n} END")

        capability_terms.append(
            f"AND ((SELECT COALESCE(SUM({_gpu_weight('r')}), 0) {_running_here('r')}) "
            f"+ {_gpu_weight('c')} <= %(max_running)s)"
        )
        # THE RESERVATION — the run-next flag made real for a row wider than one slot.
        #
        # The weight above is a FILTER, and a filter outranks the ORDER BY: while anything
        # runs here, a row weighing the whole box does not fit and a weight-1 row does, so
        # every slot that frees is refilled by narrower work before the box is ever empty —
        # and the flagged row, sorted FIRST, is filtered out of every one of those claims.
        # Observed in practice: a gateway lease (gpu_slots='all', flagged) sat queued while
        # its warm vLLM box took narrower jobs into every slot it freed, until the gateway
        # gave up. (claim_next_cpu_job names the same starvation for cpu_slots.)
        #
        # So while a queued RUN-NEXT row that weighs more than one slot HERE is claimable in
        # every respect but the rows already running — the same run guard, tenant, placement
        # and _eligibility terms as this claim, rendered for it — only such a row may be
        # claimed. The box drains; the flagged row fits; it takes the box.
        #
        # Bounded by construction:
        #   * only the flag reserves: an unflagged wide row keeps the 0030 behaviour, and a
        #     row weighing 1 here (every NULL row; 'all' on a one-wide box) reserves nothing;
        #   * a row this claim could never take reserves nothing, so a reservation cannot
        #     outlive the row's claimability here;
        #   * it lifts on the very next claim once the row is claimed — here or on a peer —
        #     or cancelled.
        # Its cost is its purpose: while the box drains for the row that asked for all of it,
        # no new narrower work is admitted. Lane-local: the inline lane (require_model=True)
        # never takes a remote-served row, so it does not reserve for one.
        def _run_next_wide(alias: str) -> str:
            return (f"({alias}.is_priority AND {alias}.gpu_slots IS NOT NULL "
                    f"AND {_gpu_weight(alias)} > 1)")

        capability_terms.append(
            f"AND ({_run_next_wide('c')} OR NOT EXISTS ("
            "SELECT 1 FROM workflow_node_jobs q "
            "WHERE q.queue = %(queue)s AND q.status = 'queued' AND q.project = %(project)s "
            "AND EXISTS (SELECT 1 FROM workflow_runs qr "
            f"WHERE qr.id = q.run_id AND qr.{_RUN_CLAIMABLE_GUARD}) "
            f"{_placement_terms('q')} "
            + " ".join(_eligibility("q"))
            + f" AND {_run_next_wide('q')}))"
        )
    # The row's RESOLVED serving kind, computable inside SQL for both dialects and
    # NEVER NULL under the law: declared preference this box serves →
    # declared first preference (cold worker) → 'inprocess' for a model row with no
    # preference (it loads weights in this process — a serving kind of its own) →
    # the caller's box-default prediction. Rendered per-ALIAS because it appears in
    # two scopes: the outer UPDATE's SET (alias j — the stamp) and the inner
    # candidate guard (alias c — the per-kind count).
    #
    # The previous build stamped NULL for any undeclared row and EXEMPTED it from
    # the caps ("not LLM-served work"). That premise was false — the dispatch
    # layer's own docs (llm_backends/factory.py) name an LLM-using node
    # whose rows never carry server_types — and the
    # exemption admitted a ComfyUI render beside an ollama LLM job on one
    # unified-memory GPU box. Undeclared work IS served by something; predict it,
    # stamp it, and let the dispatch-time re-stamp correct the label if dispatch lands
    # elsewhere.
    def _model_stamp(alias: str) -> str:
        """WHICH model this row will actually run (migration 0026).

        The array's ORDER is a PRIORITY, so this is `first_overlap`: the FIRST entry in
        `required_model_ids` that this box can serve. Without it the array degrades into an
        unordered SET — the row still claims, nothing looks broken, and the fleet quietly
        runs the fallback while the preferred model sits loaded and idle.

        Falls back to the scalar `required_model` so a single-model row (every row written
        before 0026) is stamped identically, and ONE column answers "what actually ran" for
        both shapes. NULL for an unconstrained row, which names no model by definition."""
        if not known:
            return f"{alias}.required_model"
        return (
            "COALESCE("
            + _d.first_overlap(f"{alias}.required_model_ids", "%(known_models)s")
            + f", {alias}.required_model)"
        )

    def _stamp(alias: str) -> str:
        return (
            "COALESCE("
            + (_d.first_overlap(f"{alias}.server_types", "%(served_types)s")
               if served else "NULL")
            + f", {_d.array_first(alias + '.server_types')}"
            + f", CASE WHEN {alias}.required_model IS NOT NULL "
              "THEN %(inprocess_kind)s ELSE %(default_server)s END)"
        )
    if kind_caps is None:
        # ON BY DEFAULT: a host-scoped claim gets the law without
        # asking — off-by-default is how the NULL hole shipped. The one batching
        # kind gets the box budget as its PAR; everything else falls to the
        # width-1 default below. ``kind_caps={}`` (explicitly empty) is the
        # documented opt-out back to the legacy single-number arbiter.
        caps = {"vllm": max(1, int(max_running)) if max_running is not None else 1}
    else:
        caps = {k: int(v) for k, v in kind_caps.items() if v}
    law_on = bool(caps) and bool(host)
    if law_on:
        stamp_expr = _stamp("c")
        cap_case = "CASE " + " ".join(
            f"WHEN {stamp_expr} = %(cap_kind_{i})s THEN %(cap_val_{i})s"
            for i in range(len(caps))
        ) + " ELSE %(cap_default)s END"
        # A legacy NULL running row (stamped by a pre-law wheel) counts against
        # EVERY candidate's kind: the engine cannot know what that job put on the
        # card, and (a 07-19 wheel beside a 07-28 arbiter) is what
        # trusting the unknown looks like. Against a width-1 kind it blocks;
        # against a batching kind it occupies one slot, not the whole box.
        capability_terms.append(
            f"AND (SELECT COUNT(*) {_running_here('rk')} "
            f"AND (rk.resolved_server = {stamp_expr} "
            "OR rk.resolved_server IS NULL)) "
            f"< {cap_case}"
        )
    capability = " ".join(capability_terms)
    sql = _CLAIM_SQL.format(
        order=order, capability=capability, placement=_placement_terms(),
        run_guard=_RUN_CLAIMABLE_GUARD,
        lease_expr=_d.future_seconds("%(lease_s)s"), skip_locked=_d.skip_locked,
        # The opt-out opts out of stamping too — byte-compat with the legacy claim.
        stamp=_stamp("j") if law_on else "NULL",
        # Stamped even when the per-kind law is off: "which model ran" is a fact about the
        # row, not part of the arbiter, and a NULL here would blind the panel and the
        # warm-affinity sort for every deployment that opts out of the caps.
        model_stamp=_model_stamp("j"),
    )
    params = {
        "worker_lane": worker_lane,
        "host": host,
        "lease_s": int(lease_s),
        "queue": "gpu",
        "project": _project(project),
        "current_model": current_model,
        "host_dir": _host_dir(host_priority),
        "box": box or None,
        # Warm (server, model) affinity inputs — ALWAYS bound (the ORDER BY term is
        # unconditional), empty ⇒ every row scores 0 and the tiebreak is inert.
        "warm_servers": _d.array_param([s for s in (warm_servers or []) if s]),
        "warm_models": _d.array_param([m for m in (warm_models or []) if m]),
        # The cluster reservation's kind — read by the served gate (for the candidate AND the
        # run-next reservation's own eligibility) and by ``reservations_only``.
        "reservation_kinds": _d.array_param([CLUSTER_SERVER_KIND]),
    }
    if served:
        params["served_types"] = _d.array_param(served)
    if serving_context is not None:
        params["serving_context"] = int(serving_context)
    if law_on:
        for i, (k, v) in enumerate(caps.items()):
            params[f"cap_kind_{i}"] = k
            params[f"cap_val_{i}"] = int(v)
        # The TIGHTEST cap. This is the
        # budget for a row whose resolved kind is not in ``caps`` at all: ollama,
        # comfyui, 'inprocess', any custom server, any typo, a kind this build has
        # never heard of. The rule: only a kind EXPLICITLY
        # capped wider (vLLM = PAR) batches; everything else serialises at 1.
        # (Was ``min(caps.values())`` — correct while ollama:1 was always present,
        # but the engine-default caps are now {'vllm': PAR}, whose min would hand
        # the least-known kind the loosest budget again.)
        params["cap_default"] = 1
        params["inprocess_kind"] = INPROCESS_SERVER_KIND
        params["default_server"] = default_server or _DEFAULT_SERVER_KIND
    if known:
        params["known_models"] = _d.array_param(known)
    if pool:
        params["pool_modules"] = _d.array_param(pool)
    if max_running is not None and host:
        params["max_running"] = int(max_running)
    # EITHER arbiter needs the lock. Both the box-slot gate
    # (max_running) and the per-kind gate (kind_caps) decide by COUNTING running
    # rows, so both lose the same race: two claimants read "cap-1 running" and both
    # take the last slot. The per-kind gate shipped without the serialisation the
    # box gate has always had, which made ollama's cap of 1 breakable by exactly
    # the concurrency it exists to prevent.
    arbitrated = bool(host) and (max_running is not None or law_on)
    with connection() as conn, conn.cursor() as cur:
        if arbitrated:
            # Serialize claims per box (pg advisory xact lock; sqlite's single
            # writer needs none) so two concurrent claimants can't both read
            # "N-1 running" and both take the last slot.
            ser = _d.box_claim_serializer()
            if ser:
                cur.execute(ser, params)
        cur.execute(sql, params)
        return cur.fetchone()


def restamp_resolved_server(job_id: str, server_type: str) -> None:
    """Correct a running row's ``resolved_server`` to the kind ACTUALLY serving it.

    The claim stamps the kind it counted the job against, and the per-kind arbiter
    counts running rows BY that stamp (:func:`claim_next_gpu_job`). When the stamped
    server is dead at dispatch time the job is served on a live kind instead — the
    right call, a job should not die because a sidecar went down — but the stale
    stamp then makes every LATER claim count this job against a server that is not
    running it. Both directions are wrong: the dead kind looks busy (its cap is
    consumed by nothing) and the serving kind looks free (it can be over-claimed).

    The arbiter re-reads the DB on every claim rather than caching, so correcting
    the row is sufficient — no coordination, no invalidation. Best-effort by
    design: this rides a dispatch path that must not fail for a bookkeeping write,
    and a missed correction degrades to exactly the pre-fix behaviour.
    """
    if not job_id or not server_type:
        return
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE workflow_node_jobs SET resolved_server = %(kind)s "
                "WHERE id = %(id)s AND status = 'running'",
                {"kind": str(server_type), "id": str(job_id)},
            )
    except Exception:
        log.debug("[node-queue] re-stamp of job %s to %s failed", job_id,
                  server_type, exc_info=True)


def vlm_pool_should_defer(
    host_label: str, par: int, *, stale_s: int = STALE_WORKER_AFTER_S,
    project: str | None = None, box: str | None = None,
) -> bool:
    """FILL-BEFORE-SPILL gate for the no-model GPU (VLM) pool lane.

    A vLLM/ollama machine's pool feeder calls this BEFORE each no-model claim to
    decide whether to *defer* this cycle. The fleet ranks GPU machines by
    advertised PAR — ``(worker_heartbeats.concurrency DESC, host_label ASC)`` —
    so a high-`--max-num-seqs` vLLM box ranks first and a PAR-1 ollama box ranks
    last. This machine ``M = (host_label, par)`` defers IFF some FRESH gpu worker
    ``R`` ranked STRICTLY ABOVE it still has free VLM capacity:

      * ranked above:  ``R.concurrency > M.par
                         OR (R.concurrency = M.par AND R.host_label < M.host)``
      * free capacity: ``(running no-model gpu jobs claimed_by R) < R.concurrency``

    The effect is bin-packing: VLM jobs fill the highest-ranked box's PAR slots
    first and spill to the next box only when the higher one is full — under
    light load they consolidate onto one machine (freeing the others for
    diffusion / idle-unload), under heavy load they still spill (no throughput
    loss). Invariants (all enforced by the EXISTS query, no Python branching):

      * **Top-ranked never defers.** The max-PAR / earliest-host machine has no
        ``R`` above it → the EXISTS is empty → it always fills first. No global
        starvation.
      * **Single box / no fresh higher peer → never defers** (returns ``False``)
        ⇒ behaviour byte-identical to today. SAFE default that protects
        single-box fleets and other library consumers.
      * **Stale peers don't count.** ``R`` must be fresh
        (``last_seen > now() - stale_s``); a dead top box can't block everyone.
      * **Pool lane only.** Capacity counts running gpu rows that are POOL work by
        :func:`_pool_row_term` — no model, OR remote-served (``server_types`` set,
        migration 0022). The inline diffusion lane (an in-process warm-model job) is
        neither counted nor affected. Using the claim's own definition is
        load-bearing: when this view tested only ``required_model IS NULL`` it could
        not see remote-served work at all, so peers read as idle and the
        anti-starvation break never armed.
      * **Run next is never packed.** A queued pool row this box could claim that carries
        the RUN-NEXT flag (``is_priority``) breaks deferral at once, age 0 — packing is a
        placement preference, and the flag says "run this next". Before, a gateway lease
        pinned to a lower-ranked box waited out the whole starvation window
        (:data:`VLM_POOL_MAX_DEFER_AGE_S`) for a peer that could never take it.

    A single cheap query: ``worker_heartbeats`` (fresh gpu rows ranked above M)
    LEFT JOINed to a per-``claimed_by`` COUNT of its running no-model gpu jobs,
    short-circuited by ``EXISTS``. ``stale_s`` mirrors the heartbeat freshness
    window (default :data:`STALE_WORKER_AFTER_S` = 30 s).

    ``box`` is this machine's PHYSICAL box id (``default_box_id()``), threaded
    into the anti-starvation view so "claimable" means claimable BY THIS BOX —
    see below."""
    # ONE definition of pool-lane membership, shared with the claim SQL — see
    # _pool_row_term. Bound here so the f-string below interpolates them.
    pool_running = _pool_row_term("workflow_node_jobs")
    pool_aging = _pool_row_term("sj")
    sql = f"""
        SELECT (
          EXISTS (
            SELECT 1
              FROM worker_heartbeats r
              LEFT JOIN (
                  SELECT claimed_by, COUNT(*) AS running_no_model
                    FROM workflow_node_jobs
                   WHERE queue = 'gpu'
                     AND status = 'running'
                     AND {pool_running}
                     AND project = %(project)s
                   GROUP BY claimed_by
              ) j ON j.claimed_by = r.host_label
             WHERE r.queue = 'gpu'
               AND r.project = %(project)s
               AND r.last_seen > now() - make_interval(secs => %(stale_s)s)
               -- ranked STRICTLY above M = (host_label, par)
               AND (
                    r.concurrency > %(par)s
                 OR (r.concurrency = %(par)s AND r.host_label < %(host)s)
               )
               -- R still has free VLM capacity
               AND COALESCE(j.running_no_model, 0) < r.concurrency
          )
          -- ANTI-STARVATION: never defer while a no-model job THIS BOX COULD CLAIM is
          -- AGING. If the top box were draining, nothing would age past the window; a
          -- job older than it means the higher peer ISN'T taking the work (its card is
          -- held for another project's model, or the job's box placement excludes it
          -- outright) — so claim NOW instead of deferring forever. Liveness the DB-only
          -- capacity view can't give (it's blind to the box lease AND to placement).
          -- "Claimable" is defined ONCE, by the claim SQL's own fragments: the same
          -- parent-run guard (a stuck-'queued' run is claimable —) and the
          -- same placement terms keyed on MY box (a job that avoids me isn't starving
          -- for me; a job that avoids the TOP box must still break MY deferral).
          -- RUN NEXT breaks it at age 0. Packing is a placement preference and the flag says
          -- run this next: a flagged row this box could claim must not sit out the window for
          -- a peer that merely READS free. Observed in practice: most gateway fallback
          -- leases pinned to a box ranked below a vLLM-serving box were claimed only after the
          -- full deferral window, while the top-ranked box never showed that wait.
          AND NOT EXISTS (
            SELECT 1
              FROM workflow_node_jobs sj
              JOIN workflow_runs sr ON sr.id::text = sj.run_id
             WHERE sj.queue = 'gpu'
               AND sj.status = 'queued'
               AND {pool_aging}
               AND sj.project = %(project)s
               AND sr.{_RUN_CLAIMABLE_GUARD}
               {_placement_terms("sj")}
               AND (sj.created_at < now() - make_interval(secs => %(starve_s)s)
                    OR sj.is_priority)
          )
        ) AS should_defer
    """
    params = {
        "host": host_label,
        "par": int(par),
        "stale_s": max(1, int(stale_s)),
        "starve_s": max(1, int(VLM_POOL_MAX_DEFER_AGE_S)),
        "project": _project(project),
        "box": box or None,
    }
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return bool(cur.fetchone()["should_defer"])


# ── Ingest queue ────────────────────────────────────────────────────────────
#
# The periodic ingest work rides a DEDICATED ``ingest_jobs`` table (migration
# 0007), not ``workflow_node_jobs``: a periodic ingest unit is a standalone
# callable with no parent run, so it can't satisfy the NOT NULL ``run_id`` FK
# nor the claim SQL's run-cancel join. These helpers mirror the cpu/gpu
# claim/lease shape (DRY at the SQL level) minus the run-existence guard —
# there's no run.

#: Default ingest queues. The LIVE allow-list is
#: ``config.ingest_queues`` (host-configurable) — see :func:`_ingest_queues`.
#: Migration 0008 dropped the DB CHECK that pinned ``ingest_jobs.queue`` to these.
INGEST_QUEUES: frozenset[str] = frozenset({"fetch", "load"})


def _ingest_queues() -> frozenset[str]:
    """The valid ingest queue set — host-configurable via
    ``queue_workflows.configure(ingest_queues=...)`` (default {'fetch','load'}).
    Migration 0008 moved this allow-list from a DB CHECK to here (mirrors the
    task_name gate in :func:`_ingest_tasks`)."""
    from queue_workflows.config import get_config
    return get_config().ingest_queues


def _ingest_tasks() -> frozenset[str]:
    """The valid ingest ``task_name`` set — the keys of the host-registered
    ``config.ingest_task_map``. Empty until a host registers tasks
    via ``queue_workflows.register_ingest_task``."""
    from queue_workflows.config import get_config
    return frozenset(get_config().ingest_task_map.keys())


def enqueue_ingest_job(
    *, task_name: str, queue: str, reason: str = "tick", priority: int = 100,
    args: dict[str, Any] | None = None, conn: Any = None,
    project: str | None = None,
) -> str:
    """Insert a fresh ``queued`` ingest-job row. Returns the row id.

    Raises ``ValueError`` before touching the DB on an unknown queue or
    task_name (must be a registered ingest task), matching
    :func:`enqueue_node_job`'s fail-before-write contract.

    ``args`` (migration 0008) is an optional JSON-able dict of per-job
    arguments — persisted to the ``args`` column and handed to the registered
    callable — so a host can enqueue a *parametrised* ingest task (e.g.
    ``run_scenario`` with a scenario id), not only parameterless periodic
    sweeps. Defaults to ``{}``.

    ``conn`` is an optional host psycopg connection. When given, the INSERT runs
    on it so the **caller controls the transaction** — the job row and the
    host's own domain row (e.g. ``scenario_runs``) commit atomically, and the
    ``ingest_job_ready`` NOTIFY rides the same txn. When ``None``, a pooled
    connection is borrowed and autocommits on success.
    """
    iq = _ingest_queues()
    if queue not in iq:
        raise ValueError(f"ingest queue must be in {sorted(iq)}, got {queue!r}")
    tasks = _ingest_tasks()
    if task_name not in tasks:
        raise ValueError(
            f"task_name must be a registered ingest task {sorted(tasks)}, got "
            f"{task_name!r} (register via queue_workflows.register_ingest_task)"
        )
    row_id = str(uuid.uuid4())
    sql = """
        INSERT INTO ingest_jobs
            (id, task_name, queue, reason, args, project, status, priority, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, 'queued', %s, now())
    """
    params = (
        row_id, task_name, queue, reason, _as_json(args or {}),
        _project(project), priority,
    )
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    else:
        with connection() as own, own.cursor() as cur:
            cur.execute(sql, params)
    return row_id


def get_ingest_job(job_id: str) -> dict[str, Any] | None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM ingest_jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def claim_next_ingest_job(
    queue: str, *, host: str | None = None, lease_s: int = DEFAULT_LEASE_S,
    project: str | None = None,
) -> dict[str, Any] | None:
    """Atomically grab the next queued ingest job on ``queue``.

    Mirrors :func:`claim_next_cpu_job` — a single ``SELECT … FOR UPDATE
    SKIP LOCKED`` claim stamping the lease — but WITHOUT the run-cancel join
    (an ingest job has no parent run). Ordered ``priority ASC`` then FIFO
    on creation. Project-scoped (migration 0017): only rows whose ``project``
    matches this client's (``None`` ⇒ ``config.project``). Returns the claimed
    row, or ``None`` when the queue had nothing claimable.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_jobs AS j
            SET status = 'running',
                started_at = now(),
                claimed_by = %(host)s,
                lease_expires_at = now() + make_interval(secs => %(lease_s)s)
            WHERE j.id = (
                SELECT c.id FROM ingest_jobs c
                WHERE c.queue = %(queue)s
                  AND c.status = 'queued'
                  AND c.project = %(project)s
                ORDER BY c.priority ASC, c.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
            """,
            {
                "host": host, "lease_s": int(lease_s), "queue": queue,
                "project": _project(project),
            },
        )
        return cur.fetchone()


def mark_ingest_completed(
    job_id: str, *, result: dict[str, Any] | None = None, seconds: float | None = None,
) -> dict[str, Any] | None:
    """Transition an ingest row to ``completed`` (idempotent — returns
    ``None`` if already terminal). Same WHERE-narrowed contract as
    :func:`mark_completed`."""
    json.dumps(result or {})
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_jobs
            SET status = 'completed', finished_at = now(),
                result = %s, seconds = %s
            WHERE id = %s
              AND status NOT IN ('completed', 'failed', 'cancelled')
            RETURNING *
            """,
            (_as_json(result or {}), seconds, job_id),
        )
        return cur.fetchone()


def mark_ingest_failed(
    job_id: str, *, error: str, seconds: float | None = None,
) -> dict[str, Any] | None:
    """Transition an ingest row to ``failed`` (idempotent)."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_jobs
            SET status = 'failed', finished_at = now(),
                error = %s, seconds = %s
            WHERE id = %s
              AND status NOT IN ('completed', 'failed', 'cancelled')
            RETURNING *
            """,
            (error[:8000] if error else error, seconds, job_id),
        )
        return cur.fetchone()


def reclaim_expired_ingest_leases() -> list[dict[str, Any]]:
    """Re-queue ``running`` ingest rows whose lease has lapsed.

    The ingest-table twin of :func:`reclaim_expired_leases`: a live claim
    worker renews its lease while running, so a lapsed lease means the
    owner died/wedged. Flip back to ``queued``, clear the lease, and bump
    priority to the front — the status flip fires the ``ingest_job_ready``
    NOTIFY so an idle worker picks it straight back up. Returns the
    reclaimed rows' id / task_name / queue.

    INTENTIONALLY broker-wide (NOT project-scoped, unlike the other 0017 sweeps):
    lease-reclaim is a pure status-driven recovery (running→queued by lapsed
    lease) that reads no project state and leaves each row's ``project`` tag
    intact — so a re-queued row is still claimed only by its OWN project's worker
    (exact-match claim). Running it broker-wide makes any orchestrator a recovery
    backstop for any project's orphaned lease, which is strictly safer on a shared
    broker than requiring each project's orchestrator to be up. Concurrent reclaims
    are idempotent (WHERE status='running').
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_jobs
            SET status = 'queued',
                started_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = LEAST(priority, 10)
            WHERE status = 'running'
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at < now()
            RETURNING id, task_name, queue
            """,
        )
        return list(cur.fetchall())


def reclaim_expired_leases() -> list[dict[str, Any]]:
    """Re-queue ``running`` rows whose lease has lapsed — UNLESS the
    parent run is no longer active.

    INTENTIONALLY broker-wide (NOT project-scoped, unlike the other 0017 sweeps):
    a pure status-driven recovery that leaves each row's ``project`` tag intact,
    so a re-queued row is still claimed only by its OWN project's worker. Running
    it broker-wide makes any orchestrator a recovery backstop for any project's
    orphaned lease on a shared broker — strictly safer than per-project-only
    reclaim. Idempotent under concurrency (the CASE-on-target ``WHERE
    j.status='running'``).

    A live worker renews its lease while running, so a lapsed lease
    means the owner died or wedged. Two outcomes depending on the
    parent run's status:

      * ``running`` parent — flip the row to ``queued``, clear the
        lease bookkeeping, and put the recovered work at the FRONT:
        ``is_priority = TRUE`` (sorts ahead of the band AND the GPU
        warm-model affinity) plus the band bump ``LEAST(priority, 10)``
        for backends/orderings that only read the band. Work that
        already burned wall-clock on a lost machine is the next thing a
        healthy peer claims. The status flip fires the
        ``node_job_ready`` NOTIFY (migration 0006) so an idle worker
        picks it straight back up.
      * ``cancelled`` / ``failed`` / ``completed`` parent — flip the
        row to ``cancelled`` instead. The claim SQL filters out jobs
        whose parent is non-running, so re-queuing here would create
        a ghost: the row sits in ``queued`` forever (no worker will
        ever claim it) while the queue popover keeps reporting "+1
        queued." Regression: a cancelled run left a GPU render node
        orphaned in ``queued`` after a worker restart killed the
        in-flight render. Pinned
        by ``test_lease_reclaim_cancels_running_row_on_cancelled_parent``.

    Only touches rows that carry a lease (``lease_expires_at IS NOT NULL``).

    Returns the reclaimed rows' ``id`` / ``run_id`` / ``node_id`` so the
    caller can re-dispatch or log them."""
    from queue_workflows.dialect import get_dialect
    ret = get_dialect().qualify_returning("j", ("id", "run_id", "node_id"))
    with connection() as conn, conn.cursor() as cur:
        # Single UPDATE branches on the parent run's status. The
        # CASE-on-target keeps this atomic — no race window between an
        # is-parent-active check and the lease flip.
        cur.execute(
            """
            UPDATE workflow_node_jobs AS j
            SET status = CASE WHEN r.status = 'running'
                              THEN 'queued'
                              ELSE 'cancelled'
                         END,
                started_at = CASE WHEN r.status = 'running'
                                  THEN NULL
                                  ELSE j.started_at
                             END,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = CASE WHEN r.status = 'running'
                                THEN LEAST(j.priority, 10)
                                ELSE j.priority
                           END,
                is_priority = CASE WHEN r.status = 'running'
                                   THEN TRUE
                                   ELSE j.is_priority
                              END,
                finished_at = CASE WHEN r.status = 'running'
                                   THEN j.finished_at
                                   ELSE now()
                              END
            FROM workflow_runs r
            WHERE r.id::text = j.run_id
              AND j.status = 'running'
              AND j.lease_expires_at IS NOT NULL
              AND j.lease_expires_at < now()
            RETURNING j.id, j.run_id, j.node_id
            """.replace("j.id, j.run_id, j.node_id", ret),
        )
        return list(cur.fetchall())


def requeue_job_for_retry_in_txn(cur, job_id: str) -> dict[str, Any] | None:
    """Re-queue ONE ``running`` node-job for a watchdog retry — same-cursor
    variant of :func:`requeue_job_for_retry`.

    The watchdog-trip twin of the lease-reclaim mechanic, scoped to a single
    row by id. In one statement it:

      * flips ``running`` → ``queued`` (which fires the migration-0006
        ``node_job_ready`` NOTIFY so an idle worker re-claims it at once);
      * clears the lease bookkeeping (``started_at``/``claimed_by``/
        ``lease_expires_at`` → NULL) so a fresh worker can stamp its own;
      * bumps ``priority`` to the FRONT with ``LEAST(priority, 10)`` — the EXACT
        mechanic :func:`reclaim_expired_leases` uses — so the retry runs
        promptly ahead of newer work;
      * increments ``watchdog_retries`` (migration 0010), the per-job re-queue
        counter the trip site reads to enforce the retry cap.

    CAS-guarded + idempotent like the other transitions: the WHERE narrows to
    ``status = 'running'``, so a row that's already terminal (completed / failed
    / cancelled / skipped) or already ``queued``/``awaiting_input`` is left
    untouched and the function returns ``None``. Writes NO dispatch event — the
    run stays ``running``; only the node re-runs.

    Returns the updated row (with the incremented ``watchdog_retries``), or
    ``None`` on no-match."""
    cur.execute(
        """
        UPDATE workflow_node_jobs
        SET status = 'queued',
            started_at = NULL,
            claimed_by = NULL,
            lease_expires_at = NULL,
            priority = LEAST(priority, 10),
            watchdog_retries = watchdog_retries + 1
        WHERE id = %s
          AND status = 'running'
        RETURNING *
        """,
        (job_id,),
    )
    return cur.fetchone()


def requeue_job_for_retry(job_id: str) -> dict[str, Any] | None:
    """Re-queue a watchdog-tripped node-job so a fresh worker retries it,
    instead of failing the whole run.

    Opens its own connection and delegates to
    :func:`requeue_job_for_retry_in_txn` (one ``UPDATE … RETURNING`` ⇒ one
    transaction). See that function for the full re-queue mechanic + CAS /
    idempotency contract. Returns the updated row, or ``None`` when the row was
    not ``running`` (already terminal / already re-queued)."""
    with connection() as conn, conn.cursor() as cur:
        return requeue_job_for_retry_in_txn(cur, job_id)


def flag_stale_workers_holding_running_jobs(
    *, stale_after_s: int | None = None, project: str | None = None,
) -> list[dict[str, Any]]:
    """Detect + flag workers that have gone SILENT while still owning a
    ``running`` job — the GPU-hardware-hang recovery the lease-reclaim alone
    can't reach.

    THE GAP. A GPU hardware-hang (e.g. a ROCr "GPU Hang" HW exception on a
    video-model-b render) can leave the claim-worker PROCESS wedged: a torch/HIP call
    blocked in the dead GPU context, with the worker's in-process
    :class:`~queue_workflows.claim_worker.GpuHealthWatchdog` unable to make its
    trip — either because the trip signal is unobservable from inside (on a ROCm
    box the box-level GPU probe can read non-idle even while THIS render is
    wedged, while the hung render holds its weights resident so RAM is static, so
    "GPU idle AND RAM static" is never both-true) or, on a GIL-holding hang,
    because the daemon thread can't run at all. Either way the worker stops
    refreshing its ``worker_heartbeats`` row — it silently quits claiming
    overflow work. The lease-reclaim re-queues the JOB onto a healthy host
    (good), but nothing flags the dead PROCESS so it can be bounced.

    THE DETECTOR. The orchestrator (:class:`~queue_workflows.node_pool.NodePool`)
    runs in a SEPARATE process, GIL-independent of any wedged worker, so it CAN
    observe the frozen heartbeat. This finds every ``worker_heartbeats`` row
    whose ``last_seen`` is older than ``stale_after_s`` (default 30 s = 3× the
    10 s heartbeat cadence) THAT still owns ≥ 1 ``running``
    ``workflow_node_jobs`` row. The job→worker join is BOTH
    ``j.claimed_by = wh.host_label`` AND ``j.queue = wh.queue``: the claim stamps
    ``claimed_by`` with the worker's host label (the value ``worker_heartbeats``
    is keyed on), and the queue match attributes the job to the right worker
    PROCESS on a host that runs several (e.g. host-c runs a cpu AND a gpu worker
    under one ``host_label`` — a wedged gpu worker must not flag the healthy cpu
    worker's row, and vice-versa). It stamps ``last_flagged_dead_at = now()`` on
    the matching rows and RETURNS them so the caller logs a clear, actionable
    line for an operator / host-supervisor to bounce the container.

    IDEMPOTENT. Only rows whose ``last_flagged_dead_at`` is NULL or itself older
    than ``stale_after_s`` are (re)flagged, so the 0.5 s orchestrator tick
    doesn't relog every pass — it flags once, then stays quiet until the worker
    recovers (a fresh heartbeat clears the flag via
    :func:`upsert_worker_heartbeat`) and goes stale again.

    SAFE / NON-DESTRUCTIVE. This does NOT touch the job rows (the lease-reclaim
    owns re-queuing) and does NOT kill anything — a cross-host container kill
    isn't feasible from the orchestrator (no docker socket, different host). It
    surfaces a durable, queryable signal; the host-supervisor hook acts on it.

    Returns the flagged rows' ``host_label`` / ``queue`` / ``last_seen`` /
    ``running_jobs`` (the count of running jobs that worker still owns)."""
    threshold = (
        _stale_worker_after_s() if stale_after_s is None else max(1, int(stale_after_s))
    )
    proj = _project(project)
    from queue_workflows.dialect import is_sqlite
    stale_cte = """
            WITH stale AS (
                SELECT wh.host_label, wh.queue, wh.project, wh.last_seen,
                       COUNT(j.id) AS running_jobs
                  FROM worker_heartbeats wh
                  JOIN workflow_node_jobs j
                    ON j.claimed_by = wh.host_label
                   AND j.queue = wh.queue
                   AND j.project = wh.project
                   AND j.status = 'running'
                 WHERE wh.project = %(project)s
                   AND wh.last_seen < now() - make_interval(secs => %(thr)s)
                   AND (
                        wh.last_flagged_dead_at IS NULL
                     OR wh.last_flagged_dead_at < now() - make_interval(secs => %(thr)s)
                   )
                 GROUP BY wh.host_label, wh.queue, wh.project, wh.last_seen
            )"""
    params = {"thr": threshold, "project": proj}
    with connection() as conn, conn.cursor() as cur:
        if is_sqlite():
            # SQLite RETURNING can't return CTE columns → SELECT the stale set,
            # flag each row, then return the SELECTed rows.
            cur.execute(
                stale_cte
                + " SELECT host_label, queue, project, last_seen, running_jobs FROM stale",
                params,
            )
            rows = list(cur.fetchall())
            for r in rows:
                cur.execute(
                    "UPDATE worker_heartbeats SET last_flagged_dead_at = now() "
                    "WHERE host_label = %s AND queue = %s AND project = %s",
                    (r["host_label"], r["queue"], r["project"]),
                )
            return rows
        cur.execute(
            stale_cte
            + """
            UPDATE worker_heartbeats AS wh
               SET last_flagged_dead_at = now()
              FROM stale
             WHERE wh.host_label = stale.host_label
               AND wh.queue = stale.queue
               AND wh.project = stale.project
            RETURNING wh.host_label, wh.queue, wh.project, stale.last_seen,
                      stale.running_jobs
            """,
            params,
        )
        return list(cur.fetchall())


def reclaim_all_running_for_resume(
    *, project: str | None = None,
) -> list[dict[str, Any]]:
    """Re-queue ``running`` node jobs whose CLAIMING WORKER is gone — the
    orchestrator-boot recovery hook, not the lease-expiry path
    (:func:`reclaim_expired_leases`).

    Project-scoped (migration 0017): only THIS orchestrator's project's jobs
    (``None`` ⇒ ``config.project``). The outer ``UPDATE`` filters
    ``j.project = %(project)s`` — NOT just the heartbeat sub-join — so on a
    shared broker project A's restart can't clear project B's ``claimed_by``
    (which would trip B's live worker's ``JobStatusWatcher`` and kill B's render,
    e.g. a GIL-stalled long CUDA job whose heartbeat froze >30 s while still
    rendering). Default ``""`` matches every job (single-tenant byte-compatible).

    Scoped to jobs whose ``claimed_by`` host has **no fresh heartbeat** on the
    job's queue (last beat older than ``STALE_WORKER_AFTER_S``, or no row at
    all). Such a row is genuinely orphaned — its worker died across the restart
    — so flip it back to ``queued`` at once (clearing the lease bookkeeping,
    jumping the queue via ``LEAST(priority, 10)``) instead of idling out its
    up-to-600 s lease. The status flip fires the ``node_job_ready`` NOTIFY so a
    fresh worker grabs it immediately.

    Why the heartbeat scope (was: re-queue EVERY running row): the original
    assumed a force-recreate had bounced the WHOLE fleet. But the orchestrator/
    dispatcher container restarts independently of the GPU claim workers
    (deploys, hot-fixes, a single-container bounce). Re-queuing *all* running
    rows then yanked HEALTHY in-flight jobs on workers that never restarted
    (e.g. a GPU box mid-diffusion) — clearing ``claimed_by`` trips that live
    worker's :class:`JobStatusWatcher`, which hard-exits it (symptom: a GPU box
    stops taking GPU tasks after an unrelated dispatcher restart). A
    live worker beats every ``HEARTBEAT_INTERVAL_S`` (10 s), so a fresh
    heartbeat reliably means "still running its job — leave it alone".

    Correctness doesn't hinge on perfect timing: any orphan this skips (e.g. a
    full-fleet restart where the new workers beat before this runs) is still
    caught by :func:`reclaim_expired_leases` once the dead lease lapses. So this
    is a fast-path optimisation that is now also SAFE for partial restarts.

    Returns the reclaimed rows' ``id`` / ``run_id`` / ``node_id``."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_node_jobs AS j
            SET status = 'queued',
                started_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = LEAST(j.priority, 10)
            WHERE j.status = 'running'
              AND j.project = %(project)s
              AND NOT EXISTS (
                SELECT 1 FROM worker_heartbeats h
                WHERE h.host_label = j.claimed_by
                  AND h.queue = j.queue
                  AND h.project = j.project
                  AND h.last_seen > now() - make_interval(secs => %(stale_s)s)
              )
            RETURNING id, run_id, node_id
            """,
            {"stale_s": float(STALE_WORKER_AFTER_S), "project": _project(project)},
        )
        return list(cur.fetchall())


def requeue_running_for_worker(
    host_label: str, queue: str, *, project: str | None = None,
) -> int:
    """Re-queue every ``running`` row a SPECIFIC worker still owns — the
    per-``(host_label, queue, project)``-scoped twin of
    :func:`reclaim_all_running_for_resume`.

    Project-scoped (migration 0017): ``project`` defaults to this worker's
    ``config.project``. On a shared broker ``host_label`` is no longer globally
    unique (two projects' workers can share a machine + queue), so without the
    project term an operator hard-stop of project A's worker on ``host-a/gpu``
    would also yank project B's running job there. The hard-stop runs inside the
    worker's own ``WorkerControlWatcher``, so the default ``config.project`` is
    exactly that worker's tenant.

    Used when an operator turns a machine's cpu/gpu (or ingest) worker OFF via the
    ``worker_controls`` hard-stop: the in-flight job is released back to the queue
    so a healthy peer (or this worker once it's turned back ON) picks it up — the
    point is to stop the WORKER, not to fail the WORK.

    RESUME-STYLE, not a watchdog retry: flips ``running`` → ``queued``, clears the
    lease bookkeeping, and puts the released work at the FRONT (``is_priority =
    TRUE`` on node jobs, plus the band bump ``LEAST(priority, 10)``) WITHOUT
    incrementing ``watchdog_retries`` (turning a machine off is an operational
    redistribution, not a node failure — it must not burn the retry cap). The
    status flip fires the migration-0006/0007 ``node_job_ready`` /
    ``ingest_job_ready`` NOTIFY so an idle worker re-claims at once.

    Targets ``ingest_jobs`` when ``queue`` is a host-configured ingest queue
    (``config.ingest_queues``), else ``workflow_node_jobs`` (cpu/gpu) — matching
    the table that worker draws from. The table is chosen by this fixed branch
    (never interpolated from caller data); ``host_label`` / ``queue`` are bound
    params. Returns the number of rows re-queued.

    SAFETY (no double-run): clearing ``claimed_by`` is exactly what trips a
    still-running worker's :class:`~queue_workflows.claim_worker.JobStatusWatcher`
    (it hard-exits the instant its row is no longer claimed-by-it) — the same
    guarantee :func:`reclaim_all_running_for_resume` relies on, so the row is never
    run twice across the hand-off."""
    table = "ingest_jobs" if queue in _ingest_queues() else "workflow_node_jobs"
    # Front-of-queue flag lives on node jobs only — ``ingest_jobs`` has no
    # ``is_priority`` column, so its requeue stays band-only.
    front = "" if table == "ingest_jobs" else ", is_priority = TRUE"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {table}
            SET status = 'queued',
                started_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = LEAST(priority, 10){front}
            WHERE status = 'running'
              AND claimed_by = %s
              AND queue = %s
              AND project = %s
            """,
            (host_label, queue, _project(project)),
        )
        return cur.rowcount or 0


def requeue_running_job(
    job_id: str, *, host_label: str, queue: str, project: str | None = None,
    started_at: Any = None,
) -> int:
    """Re-queue ONE specific ``running`` row this worker owns — the per-job twin
    of :func:`requeue_running_for_worker`, with the identical resume-style
    semantics (``running`` → ``queued``, lease cleared, front of the line, and NO
    ``watchdog_retries`` bump).

    WHY A PER-ID VARIANT EXISTS. ``requeue_running_for_worker`` releases every
    row for a ``(host_label, queue, project)``, which is exactly right for its
    caller — an operator turning that whole lane OFF. It is exactly WRONG for a
    threaded cpu worker's watchdog trip: the recommended deployment runs N
    claim-worker PROCESSES that share ONE ``host_label``
    (``deploy/run_claim_lane.sh``), so a host-wide release would yank a healthy
    sibling PROCESS's rows out from under it. That process would then self-exit
    the moment its :class:`~queue_workflows.claim_worker.JobStatusWatcher` saw
    the row reassigned — turning one wedged job into a whole box's lane going
    down. A tripping worker may only release the ids it actually holds.

    ``started_at`` PINS THE CLAIM, not just the row, and is what makes the
    scoping actually hold. ``claimed_by``/``queue``/``project`` are not enough:
    under the shared-``host_label`` deployment above, a row that left the caller
    (lease lapse, a host-wide corrective requeue, the over-claim reaper) and was
    then re-claimed by a SIBLING PROCESS still matches every one of those terms,
    so an id-only release would steal a live row — and that sibling could not
    even notice, since its ``JobStatusWatcher`` compares against the same shared
    label. A re-claim stamps a fresh ``started_at``, so passing the value seen at
    claim time turns a stale id into a genuine no-op. ``None`` (the default)
    keeps the looser id-only match for callers that have no claim timestamp.

    The claim check is done in PYTHON, not as an ``AND started_at = %s`` term,
    and that is deliberate: on the sqlite backend the column is stored as TEXT
    with MILLISECOND precision while ``db._adapt_dt`` renders a bound datetime to
    WHOLE SECONDS, so that equality can never match — it would silently turn
    every sweep into a no-op there. Comparing the two Python values sidesteps the
    dialect entirely (a value read back compares equal to the one the claim
    returned).

    The ``FOR UPDATE`` makes the read+write atomic against a concurrent re-claim
    **on Postgres**; the sqlite dialect strips the clause (``db._FORUPDATE_RE``)
    and relies on its process-wide connection lock plus WAL conflict detection
    instead. That is the same guarantee the rest of the engine's claim path runs
    on, not a weaker one invented here.

    Returns rows affected (0 or 1)."""
    table = "ingest_jobs" if queue in _ingest_queues() else "workflow_node_jobs"
    front = "" if table == "ingest_jobs" else ", is_priority = TRUE"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT started_at FROM {table}
            WHERE id = %s
              AND status = 'running'
              AND claimed_by = %s
              AND queue = %s
              AND project = %s
            FOR UPDATE
            """,
            (job_id, host_label, queue, _project(project)),
        )
        row = cur.fetchone()
        if row is None:
            return 0
        if started_at is not None and row["started_at"] != started_at:
            # Same id, DIFFERENT claim — the row left this caller and someone
            # else (a sibling process under the shared host_label) owns it now.
            return 0
        cur.execute(
            f"""
            UPDATE {table}
            SET status = 'queued',
                started_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = LEAST(priority, 10){front}
            WHERE id = %s
              AND status = 'running'
              AND claimed_by = %s
              AND queue = %s
              AND project = %s
            """,
            (job_id, host_label, queue, _project(project)),
        )
        return cur.rowcount or 0


def requeue_running_if_worker_stale(
    host_label: str, queue: str, *,
    project: str | None = None, stale_after_s: int | None = None,
) -> int:
    """FAST zombie reclaim: re-queue a worker's ``running`` rows NOW — but only
    while that worker's heartbeat is STALE (or absent), i.e. nobody with this
    ``(host_label, queue, project)`` identity is alive.

    This is the conditional twin of :func:`requeue_running_for_worker`, and the
    fast path the 600 s lease-reclaim deliberately isn't: the lease is long so a
    LIVE worker's renewal hiccup never gets it reclaimed, which meant a DEAD
    worker's job camped the box's slot for up to the whole lease. The heartbeat
    is the fast liveness signal (10 s cadence, stale at
    :data:`STALE_WORKER_AFTER_S` = 30 s), so keying the reclaim on it frees the
    slot in seconds-to-~30 s instead.

    SAFE FOR MULTI-PROCESS LANES (30 cpu procs share one label): any live sibling
    keeps the heartbeat fresh, and a claim can only happen AFTER the claimer's
    first beat (``run_forever`` starts the heartbeat before the claim loop), so
    "claimed row + stale heartbeat" can never describe a live job. The staleness
    check rides IN the UPDATE statement (one atomic snapshot, no check-then-act
    race), and double-run is prevented by the standard guarantee: clearing
    ``claimed_by`` trips a still-alive claimant's JobStatusWatcher.

    RESUME-STYLE like the operator-stop requeue: front-of-queue, NO
    ``watchdog_retries`` bump — a dead process is an operational loss, not a node
    fault. Returns the rows re-queued (0 when the worker is alive)."""
    threshold = STALE_WORKER_AFTER_S if stale_after_s is None else int(stale_after_s)
    table = "ingest_jobs" if queue in _ingest_queues() else "workflow_node_jobs"
    front = "" if table == "ingest_jobs" else ", is_priority = TRUE"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {table} AS t
            SET status = 'queued',
                started_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = LEAST(priority, 10){front}
            WHERE t.status = 'running'
              AND t.claimed_by = %(host)s
              AND t.queue = %(queue)s
              AND t.project = %(project)s
              AND NOT EXISTS (
                    SELECT 1 FROM worker_heartbeats wh
                     WHERE wh.host_label = %(host)s
                       AND wh.queue = %(queue)s
                       AND wh.project = %(project)s
                       AND wh.last_seen > now() - make_interval(secs => %(stale_s)s)
              )
            """,
            {
                "host": host_label, "queue": queue,
                "project": _project(project), "stale_s": max(1, threshold),
            },
        )
        return cur.rowcount or 0


def detect_frozen_leases(
    prev_sample: dict[str, tuple[Any, float]],
    rows: list[tuple[str, Any]],
    *, now: float, frozen_after_s: float,
) -> tuple[list[tuple[str, Any]], dict[str, tuple[Any, float]]]:
    """PURE frozen-lease detector — the per-JOB liveness check heartbeats can't give.

    A live claimant's :class:`~queue_workflows.claim_worker.LeaseRenewer` advances
    ``lease_expires_at`` every ~10 s; a dead one's value FREEZES. This folds the
    current ``running`` rows (``[(job_id, lease_expires_at), …]``) into the previous
    sample (``{job_id: (lease_value, frozen_since_ts)}``) and returns

      * ``to_reclaim`` — ``[(job_id, frozen_lease_value), …]`` whose value has not
        moved for ``frozen_after_s`` (pass the value to the CAS requeue so an
        in-between renewal aborts the reclaim), and
      * the new sample (changed/new values restart the clock; rows no longer
        ``running`` drop out).

    WHY (the gap this closes): a FAST worker restart (< heartbeat staleness) hides
    the corpse — the old process died between beats and the new one starts beating
    the SAME label within the window, so every heartbeat-keyed reclaim sees a live
    label while the predecessor's job rots until the full lease lapses. The lease
    renewal is per-JOB and unambiguous. Pure + injectable-clock ⇒ unit-testable with
    no waiting; the orchestrator wires it (``NodePool._sweep_frozen_leases``)."""
    new_sample: dict[str, tuple[Any, float]] = {}
    to_reclaim: list[tuple[str, Any]] = []
    for job_id, lease in rows:
        prev = prev_sample.get(job_id)
        since = prev[1] if (prev is not None and prev[0] == lease) else now
        new_sample[job_id] = (lease, since)
        if now - since >= frozen_after_s:
            to_reclaim.append((job_id, lease))
    return to_reclaim, new_sample


def requeue_running_if_lease_frozen(job_id: str, expected_lease_expires_at: Any) -> int:
    """CAS reclaim of ONE frozen-lease zombie: re-queue the ``running`` row — but
    only if ``lease_expires_at`` STILL equals the frozen value the detector
    observed. A renewal in between (the claimant is alive after all) changes the
    value, the CAS misses, and this is a no-op — so a live job can never be yanked.

    Resume-style like every dead-process reclaim (front-of-queue, NO
    ``watchdog_retries`` bump), with the :func:`reclaim_expired_leases` parent-run
    CASE: a terminal parent flips the row to ``cancelled`` instead of re-queuing a
    ghost. Double-run across the hand-off is covered by the JobStatusWatcher
    (clearing ``claimed_by`` self-kills a somehow-still-alive claimant)."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_node_jobs AS j
            SET status = CASE WHEN r.status = 'running' THEN 'queued' ELSE 'cancelled' END,
                started_at = CASE WHEN r.status = 'running' THEN NULL ELSE j.started_at END,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = CASE WHEN r.status = 'running' THEN LEAST(j.priority, 10) ELSE j.priority END,
                is_priority = CASE WHEN r.status = 'running' THEN TRUE ELSE j.is_priority END,
                finished_at = CASE WHEN r.status = 'running' THEN j.finished_at ELSE now() END
            FROM workflow_runs r
            WHERE r.id::text = j.run_id
              AND j.id = %(id)s
              AND j.status = 'running'
              AND j.lease_expires_at = %(lease)s
            """,
            {"id": job_id, "lease": expected_lease_expires_at},
        )
        return cur.rowcount or 0


def running_lease_snapshot(*, queue: str | None = None) -> list[tuple[str, Any]]:
    """The frozen-lease detector's input: every ``running`` row's
    ``(id, lease_expires_at)`` — broker-wide (like :func:`reclaim_expired_leases`:
    a pure status-driven recovery that leaves ``project`` tags intact, so any
    orchestrator backstops any tenant's zombies). ``queue`` narrows when given."""
    q = "SELECT id, lease_expires_at FROM workflow_node_jobs WHERE status = 'running'"
    args: tuple = ()
    if queue:
        q += " AND queue = %s"
        args = (queue,)
    with connection() as conn, conn.cursor() as cur:
        cur.execute(q, args)
        return [(r["id"], r["lease_expires_at"]) for r in cur.fetchall()]


# ── Over-claim reaper (runtime fallback for the per-box slot arbiter) ────────
#
# The slot arbiter (``claim_next_gpu_job(max_running=…)``) prevents an over-claim
# AT THE CLAIM. This is the runtime backstop for when that prevention is bypassed
# — a claim path that forgets ``max_running``, a counter bug, a race a future
# lane opens: if a box is RUNNING more jobs than its advertised concurrency,
# re-queue the NEWEST over-capacity job(s) so the box converges to capacity
# instead of two jobs contending on one GPU until a lease lapses. Per
# ``(host, queue, project)`` — the same grouping the arbiter uses; a cross-DB
# over-claim (two projects on one physical box) is the box lease's domain, not
# visible from one DB.


def pick_over_claim_victims(
    rows: list[tuple[str, Any, Any]], capacity: int, *,
    lease_tie_s: float = 30.0,
) -> list[str]:
    """PURE: given a box's running ``(job_id, started_at, lease_epoch_s)``
    rows and its ``capacity``, return the ids to re-queue.

    Victim choice is by LEASE freshness first, job age second: the lease
    renewal is the only signal that separates a live claimant (renews every
    ~10 s, lease tracking ``now + lease_s``) from a zombie row (a dead
    predecessor's claim whose lease FROZE). The old keep-oldest-``started_at``
    rule reaped a LIVE render while keeping the zombie whenever a worker
    restart left its predecessor's older row on the slot — and the reaper's
    10 s cadence always beats the ~35 s frozen-lease sweep that would have
    cleaned that zombie first (observed live).

    Raw freshness alone would be wrong the other way: two genuinely LIVE
    claimants renew on offset cadences, so their leases differ by jitter and
    the victim would be arbitrary. Rows whose lease is within ``lease_tie_s``
    (~3× the renewal cadence) of the freshest lease are therefore one LIVE
    bucket — kept oldest-``started_at``-first, the most-progress heuristic —
    while rows lagging further (frozen ⇒ decaying) sort behind it, stalest
    last. A ``None`` lease on a running row is broken state a live claimant
    can't have: reaped first (the requeue is the safe repair). ``None``
    ``started_at`` sorts oldest (kept); id breaks remaining ties."""
    cap = max(int(capacity), 0)
    fresh = [r[2] for r in rows if r[2] is not None]
    live_floor = (max(fresh) - float(lease_tie_s)) if fresh else None

    def _key(r: tuple[str, Any, Any]) -> tuple:
        lease, started = r[2], r[1]
        in_live = lease is not None and live_floor is not None and lease >= live_floor
        return (
            0 if in_live else 1,
            # live bucket: oldest started_at first (None oldest)
            # stale bucket: freshest lease first, so the stalest fill the tail
            (started is not None, started if started is not None else 0)
            if in_live else (lease is None, -(lease or 0)),
            r[0],
        )

    ordered = sorted(rows, key=_key)
    return [r[0] for r in ordered[cap:]]


def requeue_over_claimed_running(job_id: str) -> int:
    """Re-queue ONE over-capacity running job, resume-style (front-of-queue, NO
    ``watchdog_retries`` bump — an over-claim isn't the job's fault). CAS on
    ``status = 'running'`` so a job that just finished is never yanked; a
    terminal parent flips it to ``cancelled`` instead of re-queuing a ghost (the
    :func:`requeue_running_if_lease_frozen` pattern). Double-run across the
    hand-off is covered by the JobStatusWatcher (clearing ``claimed_by``
    self-kills a somehow-still-alive claimant)."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_node_jobs AS j
            SET status = CASE WHEN r.status = 'running' THEN 'queued' ELSE 'cancelled' END,
                started_at = CASE WHEN r.status = 'running' THEN NULL ELSE j.started_at END,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = CASE WHEN r.status = 'running' THEN LEAST(j.priority, 10) ELSE j.priority END,
                is_priority = CASE WHEN r.status = 'running' THEN TRUE ELSE j.is_priority END,
                finished_at = CASE WHEN r.status = 'running' THEN j.finished_at ELSE now() END
            FROM workflow_runs r
            WHERE r.id::text = j.run_id
              AND j.id = %(id)s
              AND j.status = 'running'
            """,
            {"id": job_id},
        )
        return cur.rowcount or 0


def reap_over_claimed_boxes(*, queue: str = "gpu") -> list[dict[str, Any]]:
    """Runtime over-claim backstop: for every ``(claimed_by, project)`` on
    ``queue`` running MORE jobs than that box advertises
    (``worker_heartbeats.concurrency``), re-queue the stalest-leased
    over-capacity job(s) (see :func:`pick_over_claim_victims` — a frozen lease
    marks the zombie; the freshest-renewing rows are the live ones to keep).
    Skips a box with no heartbeat (capacity unknown — the lease /
    dead-worker sweeps own the vanished-worker case). Returns one dict per
    re-queued victim for the caller to log; a requeue that races a terminal
    loses the CAS and is simply not counted."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, claimed_by, project, started_at, lease_expires_at "
            "FROM workflow_node_jobs "
            "WHERE status = 'running' AND queue = %s AND claimed_by IS NOT NULL",
            (queue,),
        )
        running = cur.fetchall()
        cur.execute(
            "SELECT host_label, project, concurrency FROM worker_heartbeats "
            "WHERE queue = %s",
            (queue,),
        )
        caps = {(r["host_label"], r["project"]): r["concurrency"] for r in cur.fetchall()}

    def _lease_epoch(v: Any) -> float | None:
        # pg hands back a datetime, sqlite an ISO text — the picker buckets by
        # SECONDS relative to the freshest lease, so normalize to epoch floats.
        if v is None:
            return None
        if hasattr(v, "timestamp"):
            return v.timestamp()
        try:
            return datetime.fromisoformat(str(v)).timestamp()
        except ValueError:
            return None

    groups: dict[tuple[str, str], list[tuple[str, Any, Any]]] = {}
    for r in running:
        groups.setdefault((r["claimed_by"], r["project"]), []).append(
            (r["id"], r["started_at"], _lease_epoch(r["lease_expires_at"])))

    reaped: list[dict[str, Any]] = []
    for (host, project), rows in groups.items():
        cap = caps.get((host, project))
        if cap is None or len(rows) <= cap:
            continue
        for job_id in pick_over_claim_victims(rows, cap):
            if requeue_over_claimed_running(job_id):
                reaped.append({"job_id": job_id, "host": host, "project": project,
                               "running": len(rows), "capacity": cap})
    return reaped


# ── Worker capacity heartbeat (DRY upsert) ──────────────────────────────────
#
# Single home for the ``worker_heartbeats`` INSERT … ON CONFLICT so the two
# emitters (the claim worker's ``HeartbeatEmitter`` for capacity + the GPU
# model cache's ``_publish_current_model`` for the model slot) share ONE
# statement rather than each carrying its own copy.


def upsert_worker_heartbeat(
    *,
    host_label: str,
    queue: str,
    concurrency: int,
    current_model: str | None = None,
    known_models: Iterable[str] | None = None,
    llm_servers_available: Iterable[str] | None = None,
    vram_total_mb: int | None = None,
    fits_models: Iterable[str] | None = None,
    update_current_model: bool = True,
    project: str | None = None,
    llm_context_length: Any = _UNSET,
    llm_context_max: Any = _UNSET,
) -> None:
    """Upsert this worker's ``(host_label, queue, project)`` capacity row,
    refreshing ``last_seen`` to ``now()``.

    ``llm_context_length`` / ``llm_context_max`` (migration 0028) — the context window
    this box's LLM server serves NOW (observed), and the highest it could switch to
    (model limit ∧ free memory, or the current window when the box has no working
    lever to restart its server bigger). The orchestrator tags a sized job no live
    box's max reaches; see :func:`flag_unassignable_gpu_jobs`.

    Passing ``None`` CLEARS them — a box whose server went away must not leave a window
    behind for the claim to route sized jobs into. NOT passing them at all leaves the
    columns untouched and keeps them out of the statement entirely, which is what makes
    a worker of this engine safe against a database the orchestrator has not migrated
    to 0028 yet: a lane with no window (cpu, ingest, download) never names them.

    ``project`` (migration 0017) is the worker's tenant tag; ``None`` ⇒
    ``config.project``. It's part of the worker identity so two projects' clients
    on the SAME machine + queue (one shared broker) don't clobber each other's
    heartbeat (the PK is ``(host_label, queue, project)``).

    Any queue family may upsert (migration 0008 dropped the cpu/gpu-only
    CHECK), so ingest workers heartbeat too. The GPU heartbeat passes
    ``current_model`` (the gauge's busy signal); CPU + ingest heartbeats leave
    it NULL. ``known_models`` is the capability list advertised for affinity
    routing; ``None`` is normalised to an empty array so the column is never
    left stale.

    ``llm_servers_available`` is the OBSERVED LLM-server capability (migration
    0014) — which server types this host can actually run (e.g. ``['ollama']`` on
    an AMD box, ``['ollama', 'vllm']`` on an NVIDIA host with the vllm sidecar).
    ``None`` is normalised to the ``['ollama']`` baseline so the column is never
    left stale and a caller that doesn't care (cpu/ingest workers, other consumer
    projects) keeps the safe default.

    ``update_current_model`` controls whether the ON CONFLICT path
    overwrites ``current_model``.

    A live refresh ALSO clears ``last_flagged_dead_at`` (migration 0009): if the
    orchestrator's stale-worker detector had flagged this worker as wedged, a
    fresh heartbeat means it (or its replacement after a bounce) is alive again,
    so the dead-flag is reset — a future hang then re-flags cleanly instead of
    staying latched from the previous incident.
    """
    known = list(known_models) if known_models is not None else []
    llm_servers = (
        list(llm_servers_available) if llm_servers_available is not None else ["ollama"]
    )
    # Capacity advertisement (migration 0015). ``fits_models`` is the worker-
    # computed subset of known ids whose est_vram_gb fits this machine, used by
    # the claim gate and the fleet unassignable sweep; ``None`` ⇒ empty array
    # (advertise no capacity claim) so a non-GPU / pre-capacity worker leaves the
    # column at its '{}' default rather than stale.
    fits = list(fits_models) if fits_models is not None else []
    vram = int(vram_total_mb) if vram_total_mb is not None else None
    model_set = (
        "current_model = EXCLUDED.current_model," if update_current_model else ""
    )
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    # The 0028 columns appear in the statement ONLY when the caller named them, so a
    # worker that has no window to publish runs the pre-0028 statement verbatim.
    ctx_cols, ctx_sets, ctx_params = "", "", ()
    if llm_context_length is not _UNSET or llm_context_max is not _UNSET:
        ctx_cols = ", llm_context_length, llm_context_max"
        ctx_sets = ("llm_context_length = EXCLUDED.llm_context_length,\n"
                    "                    llm_context_max    = EXCLUDED.llm_context_max,")
        ctx_params = (
            None if llm_context_length in (None, _UNSET) else int(llm_context_length),
            None if llm_context_max in (None, _UNSET) else int(llm_context_max),
        )
    ctx_values = ", %s, %s" if ctx_cols else ""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO worker_heartbeats
                (host_label, queue, project, concurrency, last_seen,
                 current_model, known_models, llm_servers_available,
                 vram_total_mb, fits_models{ctx_cols})
            VALUES (%s, %s, %s, %s, now(), %s, %s, %s, %s, %s{ctx_values})
            ON CONFLICT (host_label, queue, project) DO UPDATE
                SET concurrency   = EXCLUDED.concurrency,
                    {model_set}
                    known_models  = EXCLUDED.known_models,
                    llm_servers_available = EXCLUDED.llm_servers_available,
                    vram_total_mb = EXCLUDED.vram_total_mb,
                    fits_models   = EXCLUDED.fits_models,
                    {ctx_sets}
                    last_seen     = EXCLUDED.last_seen,
                    last_flagged_dead_at = NULL
            """,
            (host_label, queue, _project(project), int(concurrency), current_model,
             _d.array_literal(known), _d.array_literal(llm_servers), vram,
             _d.array_literal(fits), *ctx_params),
        )


def _as_server_list(value: Any) -> list[str]:
    """The serving kinds of one heartbeat row, as a list of strings.

    BOTH backends already hand this column back decoded — pg as a ``text[]``, sqlite
    through ``db._JSON_ARRAY_COLS`` — so the list branch is the live one. The string
    branch is belt-and-braces for a caller that reads the column raw; it used the wrong
    module name and could only ever have raised, which the caller would have swallowed
    into "this box has no remembered window"."""
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    return []


def last_published_context(
    host_label: str, queue: str = "gpu", project: str | None = None,
    served_by: str | None = None,
) -> int | None:
    """The context window THIS worker last published for itself, or ``None``.

    The restart half of the idle-box trap (0028). A box with no lever to ask remembers
    the window it observed, and a restart erases that memory — so a box restarted while
    its model happens to be unloaded (five minutes of quiet is enough) is windowless
    again, and stays that way until some unsized job warms it. Its own last heartbeat is
    the record of what it saw; the first live observation overwrites it.

    Its OWN row only — another box's window says nothing about this daemon. And with
    ``served_by``, only a publication made while THAT kind was serving: the number a box
    published while its vLLM ran is vLLM's, and adopting it as ollama's is how a box
    advertises a window it has never observed (ollama picks its own default from VRAM).
    With two kinds serving, the published number is the smallest either offered, so it
    cannot overstate the kind that asks for it.

    Best-effort: this runs on the heartbeat path, where a DB blip must never do more than
    leave the box saying "I do not know"."""
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT llm_context_length, llm_servers_available FROM worker_heartbeats "
                "WHERE host_label = %s AND queue = %s AND project = %s",
                (host_label, queue, _project(project)),
            )
            row = cur.fetchone()
            if row is not None and served_by is not None:
                # Filtered HERE, not in SQL: the column is a text[] on pg and a JSON
                # string on sqlite, and one Python membership test reads the same on both.
                served = row["llm_servers_available"] if hasattr(row, "keys") else row[1]  # noqa: E501
                if served_by not in _as_server_list(served):
                    return None
    except Exception:
        log.debug("[node-queue] last-published-context read failed; unknown",
                  exc_info=True)
        return None
    value = (row or {}).get("llm_context_length")
    return int(value) if value else None


def clear_worker_current_model(
    host_label: str, queue: str, *, mark_stale: bool = True,
    project: str | None = None,
) -> dict[str, Any] | None:
    """Clear a worker's ``current_model`` (its GPU busy signal) and, by default,
    age its ``last_seen`` out of the live window — the "don't leave a busy ghost"
    fix for a HARD-exiting worker.

    A watchdog trip exits via ``os._exit``, which SKIPS the ``_run_node``
    ``finally`` (so ``ModelCache.mark_idle`` and the heartbeat refresh never run).
    The worker's last-written ``worker_heartbeats`` row therefore keeps advertising
    a ``current_model`` even though the process is gone — inflating a host's "N/M GPU
    busy" gauge (e.g. "3/2 GPU busy" after a kill). Before
    hard-exiting, the trip path calls this to null out ``current_model`` and, when
    ``mark_stale`` is set, push ``last_seen`` ~10× the heartbeat cadence into the
    past so the gauge — which counts only rows fresh within 30 s — drops the dead
    worker immediately rather than waiting up to 30 s for the heartbeat to age out.
    (A replacement worker's fresh heartbeat re-establishes the row normally.)

    Best-effort + idempotent: scoped by the ``(host_label, queue)`` primary key,
    a no-op (returns ``None``) when the row is absent. The caller swallows any
    error — the hard-exit must happen regardless. Returns the updated row, or
    ``None`` on no-match."""
    # 100 s ≈ 10× the 10 s heartbeat cadence, comfortably past the 30 s gauge
    # window so the dead worker drops out at once.
    stale_secs = 100 if mark_stale else 0
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE worker_heartbeats
            SET current_model = NULL,
                last_seen = CASE WHEN %s > 0
                                 THEN now() - make_interval(secs => %s)
                                 ELSE last_seen END
            WHERE host_label = %s AND queue = %s AND project = %s
            RETURNING *
            """,
            (stale_secs, stale_secs, host_label, queue, _project(project)),
        )
        return cur.fetchone()


def mark_completed_in_txn(
    cur,
    job_id: str,
    *,
    context_delta: dict[str, Any],
    seconds: float,
    vm_rss_mb_peak: int | None = None,
) -> dict[str, Any] | None:
    """Same as :func:`mark_completed` but runs on a caller-supplied
    cursor — lets the worker write the dispatch-event row in the same
    transaction (outbox atomicity contract)."""
    json.dumps(context_delta or {})
    cur.execute(
        """
        UPDATE workflow_node_jobs
        SET status = 'completed',
            finished_at = now(),
            context_delta = %s,
            seconds = %s,
            vm_rss_mb_peak = %s,
            -- Stamp the executing machine on the terminal row so per-host error /
            -- log queries work off workflow_node_jobs too (claimed_by is the real
            -- host identity; host_label was left NULL in practice). COALESCE keeps
            -- any value a worker already set.
            host_label = COALESCE(host_label, claimed_by)
        WHERE id = %s
          AND status NOT IN ('completed', 'failed', 'cancelled')
        RETURNING *
        """,
        (_as_json(context_delta), seconds, vm_rss_mb_peak, job_id),
    )
    return cur.fetchone()


def mark_completed(
    job_id: str,
    *,
    context_delta: dict[str, Any],
    seconds: float,
    vm_rss_mb_peak: int | None = None,
) -> dict[str, Any] | None:
    """Transition a row from a non-terminal state to ``completed``.

    Returns the updated row, or ``None`` when the row was already
    in a terminal state. The WHERE clause is the load-bearing piece —
    without it, a stray second call would silently overwrite a
    freshly-finalised ``context_delta`` with whatever the second call
    computed (often ``{}``).

    The allowlist is "any non-terminal state": ``queued``, ``running``,
    ``awaiting_input``."""
    with connection() as conn, conn.cursor() as cur:
        return mark_completed_in_txn(
            cur, job_id, context_delta=context_delta,
            seconds=seconds, vm_rss_mb_peak=vm_rss_mb_peak,
        )


def mark_failed_in_txn(
    cur,
    job_id: str,
    *,
    error: str,
    seconds: float | None = None,
) -> dict[str, Any] | None:
    """Same-txn variant of :func:`mark_failed`."""
    cur.execute(
        """
        UPDATE workflow_node_jobs
        SET status = 'failed',
            finished_at = now(),
            error = %s,
            seconds = %s,
            -- Stamp the failing machine on the terminal row (claimed_by = the real
            -- host) so "which machine failed?" is answerable from workflow_node_jobs,
            -- not just the events table. COALESCE keeps an already-set value.
            host_label = COALESCE(host_label, claimed_by)
        WHERE id = %s
          AND status NOT IN ('completed', 'failed', 'cancelled')
        RETURNING *
        """,
        (error[:8000] if error else error, seconds, job_id),
    )
    return cur.fetchone()


def mark_failed(
    job_id: str, *, error: str, seconds: float | None = None,
) -> dict[str, Any] | None:
    """Transition a row from a non-terminal state to ``failed``.

    Returns the updated row, or ``None`` when the row was already
    in a terminal state. Same idempotency reasoning as
    :func:`mark_completed`."""
    with connection() as conn, conn.cursor() as cur:
        return mark_failed_in_txn(
            cur, job_id, error=error, seconds=seconds,
        )


def mark_awaiting_input_in_txn(cur, job_id: str) -> dict[str, Any] | None:
    """Same-txn variant of :func:`mark_awaiting_input`."""
    cur.execute(
        "UPDATE workflow_node_jobs "
        "SET status = 'awaiting_input' "
        "WHERE id = %s "
        "  AND status NOT IN ('completed', 'failed', 'cancelled') "
        "RETURNING *",
        (job_id,),
    )
    return cur.fetchone()


def mark_awaiting_input(job_id: str) -> dict[str, Any] | None:
    with connection() as conn, conn.cursor() as cur:
        return mark_awaiting_input_in_txn(cur, job_id)


def set_input_spec(run_id: str, node_id: str, spec: dict | None) -> None:
    """Persist a per-job ``input_spec`` so the frontend can render
    N awaiting inputs side-by-side. Idempotent — re-calling with the
    same spec is a no-op write."""
    payload = json.dumps(spec) if spec is not None else None
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET input_spec = %s::jsonb "
            "WHERE run_id = %s AND node_id = %s "
            "  AND status = 'awaiting_input'",
            (payload, run_id, node_id),
        )


# ── Dispatch-event outbox ─────────────────────────────────────────────────


def enqueue_dispatch_event_in_txn(
    cur, run_id: str, node_id: str, kind: str,
) -> int:
    """Insert a dispatch event in the caller's transaction. ``kind``
    is one of 'completed' / 'failed' / 'awaiting_input' — DB CHECK
    enforces the set."""
    cur.execute(
        """
        INSERT INTO workflow_dispatch_events (run_id, node_id, kind)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (run_id, node_id, kind),
    )
    row = cur.fetchone()
    return int(row["id"])


def list_unprocessed_dispatch_events(
    *, limit: int = 50,
) -> list[dict[str, Any]]:
    """Return up-to-``limit`` rows where ``processed_at IS NULL``.

    Read-only inspection helper for tests, the snapshot endpoint, and
    the ``DispatchDriver.drain_until_quiescent`` no-progress probe.
    The drain path lives in :meth:`NodePool._drain_dispatch_events`."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, run_id, node_id, kind, attempts, error
              FROM workflow_dispatch_events
             WHERE processed_at IS NULL
             ORDER BY created_at ASC
             LIMIT %s
            """,
            (int(limit),),
        )
        return list(cur.fetchall())


def mark_dispatch_event_processed(event_id: int) -> None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_dispatch_events "
            "SET processed_at = now() WHERE id = %s",
            (event_id,),
        )


def record_dispatch_event_failure(event_id: int, error: str) -> None:
    """Increment ``attempts`` and record the error text without
    setting ``processed_at`` so the next tick retries."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_dispatch_events "
            "SET attempts = attempts + 1, error = %s "
            "WHERE id = %s",
            (error[:8000] if error else error, event_id),
        )


def count_unprocessed_dispatch_events() -> int:
    """Cheap COUNT for the queue snapshot endpoint + startup health log."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM workflow_dispatch_events "
            "WHERE processed_at IS NULL"
        )
        return int(cur.fetchone()["n"])


# ── Node-event history (per-node, per-attempt) ─────────────────────────────
#
# Append-only log (migration 0011). ``record_node_event_in_txn`` rides the SAME
# transaction as the terminal / requeue state change (outbox-atomicity, exactly
# like ``enqueue_dispatch_event_in_txn``). ``record_node_event`` opens its OWN
# connection and is BEST-EFFORT — it swallows every error so an event-write blip
# can never fail the load-bearing claim / terminal / watchdog path.

#: The migration-0011 CHECK set (+ ``unassignable`` from 0015), mirrored here so a
#: bad type fails loudly in Python (before the surrounding txn aborts on the DB CHECK).
#: MUST stay in sync with the ``workflow_node_events_type_check`` DB CHECK — drift
#: silently drops events (``record_node_event`` swallows the ValueError best-effort);
#: guarded by ``tests/test_invariant_node_event_types.py``.
NODE_EVENT_TYPES = frozenset({
    "claimed", "model_load_start", "model_load_done", "progress_beat",
    "stall_suspected", "stall_trip", "gpu_health_trip", "budget_trip",
    "unassignable",  # 0015 — the capacity sweep's red-flag (node_pool._sweep_unassignable_jobs)
    "requeued", "reassigned", "lease_renew", "completed", "failed",
    "cancelled", "error",
})


def record_node_event_in_txn(
    cur,
    *,
    run_id: str,
    node_id: str,
    event_type: str,
    job_id: str | None = None,
    attempt: int = 0,
    host_label: str | None = None,
    queue: str | None = None,
    model: str | None = None,
    elapsed_s: float | None = None,
    error: str | None = None,
    detail: dict[str, Any] | None = None,
    project: str | None = None,
) -> int:
    """Append one node event in the caller's transaction (outbox-atomicity).

    Used by the terminal (completed/failed/cancelled) + requeue paths so the
    event lands atomically with the state change — mirroring
    :func:`enqueue_dispatch_event_in_txn`. ``event_type`` must be in
    :data:`NODE_EVENT_TYPES` (validated here so a bad type raises in Python
    rather than aborting the surrounding txn on the DB CHECK). ``attempt`` is
    the node-job's ``watchdog_retries`` at emit time — the cross-attempt key."""
    if event_type not in NODE_EVENT_TYPES:
        raise ValueError(f"unknown node event_type: {event_type!r}")
    cur.execute(
        """
        INSERT INTO workflow_node_events
            (run_id, node_id, job_id, attempt, event_type,
             host_label, queue, model, elapsed_s, error, detail, project)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            run_id, node_id, job_id, int(attempt or 0), event_type,
            host_label, queue, model, elapsed_s,
            (error[:8000] if error else error),
            _as_json(detail or {}), _project(project),
        ),
    )
    return int(cur.fetchone()["id"])


def record_node_event(
    *,
    run_id: str,
    node_id: str,
    event_type: str,
    job_id: str | None = None,
    attempt: int = 0,
    host_label: str | None = None,
    queue: str | None = None,
    model: str | None = None,
    elapsed_s: float | None = None,
    error: str | None = None,
    detail: dict[str, Any] | None = None,
    project: str | None = None,
) -> int | None:
    """Best-effort append of one node event on its OWN connection.

    For the non-terminal emit sites (claim, model-load, watchdog
    suspected/trip, reassign, lease-renew) that must NOT widen or fail the
    load-bearing path. Swallows EVERY exception (logs once) and returns ``None``
    on failure — an event-history blip can never take down a node run. The
    terminal / requeue sites use :func:`record_node_event_in_txn` instead
    (atomic with the state change)."""
    try:
        with connection() as conn, conn.cursor() as cur:
            return record_node_event_in_txn(
                cur, run_id=run_id, node_id=node_id, event_type=event_type,
                job_id=job_id, attempt=attempt, host_label=host_label,
                queue=queue, model=model, elapsed_s=elapsed_s, error=error,
                detail=detail, project=project,
            )
    except Exception:  # noqa: BLE001 — best-effort; never propagate
        log.exception(
            "[node-event] failed to record %s for run=%s node=%s (ignored)",
            event_type, run_id, node_id,
        )
        return None


def prune_node_events(older_than_days: int = 30) -> int:
    """Delete node events older than ``older_than_days`` (append-only growth
    control; ``ON DELETE CASCADE`` already covers run-delete / purge). Called
    from an interval-gated NodePool sweep. Returns rows deleted."""
    days = max(int(older_than_days), 1)
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM workflow_node_events "
            "WHERE created_at < now() - make_interval(days => %s)",
            (days,),
        )
        return cur.rowcount or 0


def flag_server_stranded_jobs(
    *, stale_s: int | None = None, project: str | None = None,
) -> list[dict[str, Any]]:
    """SERVER-KIND twin of :func:`flag_unassignable_gpu_jobs`.

    A queued gpu job whose ``server_types`` (migration 0022) intersects NO fresh gpu
    worker's ``llm_servers_available`` can be claimed by nobody — a ``['vllm']``-only
    row on a fleet where no vLLM answers sits queued FOREVER, with no error, no flag
    and no log. That silence is the same failure mode the 0015 capacity sweep exists
    to name, so it gets the same treatment: stamp ``unassignable_at`` /
    ``unassignable_reason``, clear the stamp the moment a matching server appears or
    the row leaves ``queued``, and return only the NEWLY flagged rows so the caller
    emits one ``unassignable`` event each.

    Same liveness guard as the capacity sweep: with ZERO fresh gpu heartbeats this is
    a fleet-down condition, not a routing verdict — no-op. A worker advertising an
    EMPTY server list still counts as fresh (it is alive, just serverless), because
    the claim-side gate treats an empty advertisement as claim-any.

    A CLUSTER RESERVATION (:data:`CLUSTER_SERVER_KIND`) is never stranded by this test: no
    worker serves 'cluster' and none ever will — the row asks for a box, not a server — so it
    is never flagged, and a flag an older sweep left on one is cleared."""
    window = int(stale_s) if stale_s is not None else _stale_worker_after_s()
    proj = _project(project)
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    overlap = _d.arrays_overlap("j.server_types", "lg.llm_servers_available")
    # A worker with an EMPTY advertisement claims ANY row (the claim gate is off for
    # it), so its presence means the row is NOT stranded.
    lg_empty = _d.array_is_empty("lg.llm_servers_available")
    reservation = _d.arrays_overlap("j.server_types", "%(reservation_kinds)s")
    kinds = _d.array_param([CLUSTER_SERVER_KIND])
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            WITH live_gpu AS (
                SELECT llm_servers_available FROM worker_heartbeats
                WHERE queue = 'gpu'
                  AND project = %(project)s
                  AND last_seen > now() - make_interval(secs => %(window)s)
            )
            UPDATE workflow_node_jobs AS j
            SET unassignable_at = NULL, unassignable_reason = NULL
            WHERE j.unassignable_at IS NOT NULL
              AND j.unassignable_reason LIKE 'no live worker serves%%'
              AND j.project = %(project)s
              AND j.server_types IS NOT NULL
              AND (
                    j.status <> 'queued'
                 OR EXISTS (SELECT 1 FROM live_gpu lg WHERE {overlap})
                 OR EXISTS (SELECT 1 FROM live_gpu lg
                            WHERE {lg_empty})
                 OR {reservation}
              )
            """,
            {"window": window, "project": proj, "reservation_kinds": kinds},
        )
        cur.execute(
            f"""
            UPDATE workflow_node_jobs AS j
            SET unassignable_at = now(),
                unassignable_reason = 'no live worker serves any of: ' ||
                    {_d.array_to_text('j.server_types')}
            WHERE j.status = 'queued'
              AND j.queue = 'gpu'
              AND j.project = %(project)s
              AND j.server_types IS NOT NULL
              AND j.unassignable_at IS NULL
              AND NOT ({reservation})
              AND EXISTS (
                    SELECT 1 FROM worker_heartbeats h
                    WHERE h.queue = 'gpu' AND h.project = %(project)s
                      AND h.last_seen > now() - make_interval(secs => %(window)s)
              )
              AND NOT EXISTS (
                    SELECT 1 FROM worker_heartbeats lg
                    WHERE lg.queue = 'gpu' AND lg.project = %(project)s
                      AND lg.last_seen > now() - make_interval(secs => %(window)s)
                      AND ({overlap} OR {lg_empty})
              )
            RETURNING *
            """,
            {"window": window, "project": proj, "reservation_kinds": kinds},
        )
        return list(cur.fetchall())


def flag_unassignable_gpu_jobs(
    *, stale_s: int | None = None, project: str | None = None,
) -> list[dict[str, Any]]:
    """Capacity-aware fleet sweep (migration 0015): red-flag queued GPU model-jobs
    that **no live machine can hold**, and clear the flag when they become
    assignable again. Returns the rows NEWLY flagged this call (the NULL→now
    transition) so the caller emits one ``unassignable`` event each.

    A queued ``gpu`` job with a ``required_model`` is *unassignable* iff NO fresh
    GPU worker advertises that model in its ``fits_models`` (the worker-computed
    set of ids whose ``est_vram_gb`` fits its VRAM). The decision is pure SQL over
    ``worker_heartbeats`` — the worker pushed the fit computation into the
    heartbeat precisely so the orchestrator (which holds no model registry) can
    decide here.

    CRITICAL liveness guard: flagging requires at least ONE fresh GPU heartbeat.
    "No machine has enough VRAM" is a CAPACITY verdict, not a liveness one — if
    the whole GPU fleet is momentarily down (a deploy / all-workers bounce) we
    must NOT red-flag every job as unassignable (that's the dead-worker sweep's
    concern). With zero fresh GPU workers the sweep is a no-op.

    Idempotent: the flag UPDATE is guarded ``unassignable_at IS NULL`` so a job
    already flagged is not re-returned (no duplicate event) until it clears and
    re-flags. The clear path un-flags rows that became fittable OR left ``queued``
    (claimed / cancelled), so a stale red flag never lingers.

    ``stale_s`` overrides the freshness window (default ``_stale_worker_after_s``,
    30 s = 3× the 10 s heartbeat).

    CONTEXT (migration 0028). A queued gpu job with a ``context_length`` is ALSO
    unassignable when no live box can reach that window — judged against each box's
    ``llm_context_max``, the highest window it could SWITCH to within the memory it has
    free (the worker recomputes it every heartbeat), not the window it serves now: a box
    that can grow into the job will switch, so the job is not stuck. Model and window
    are judged ON THE SAME BOX — one live box must both hold the model and reach the
    window — so one box having the model and another the window cannot pass a job
    nobody can run, and a flag can never be cleared by one half and re-set by the other
    (a fresh event every sweep). The reason names which half is missing.
    """
    window = int(stale_s) if stale_s is not None else _stale_worker_after_s()
    proj = _project(project)
    from queue_workflows.dialect import get_dialect
    # Array-column membership: pg ``= ANY(fits_models)`` ; sqlite json_each. The pg
    # rendering equals the literal below, so ``.replace`` is a no-op on pg.
    _contains = get_dialect().array_contains_value("lg.fits_models", "j.required_model")
    _ANY = "j.required_model = ANY(lg.fits_models)"
    # Possession vs capacity are DIFFERENT questions and now have different answers:
    # `fits_models` is (what the box HOLDS ∩ what its VRAM fits), so a model missing from it
    # may be missing for either reason. `known_models` isolates possession.
    _knows = get_dialect().array_contains_value("lg.known_models", "j.required_model")
    _ANY_KNOWN = "j.required_model = ANY(lg.known_models)"
    # ONE box serves the job: its model fits there AND its window is reachable there.
    _serves_here = (
        "(j.required_model IS NULL OR j.required_model = ANY(lg.fits_models)) "
        "AND (j.context_length IS NULL OR lg.llm_context_max >= j.context_length)"
    )
    with connection() as conn, conn.cursor() as cur:
        # CLEAR first: un-flag any flagged job that some live box can now serve (model
        # AND window, on that one box), or that has left the queued state (claimed /
        # cancelled / terminal) — so a red flag never outlives the condition that set
        # it. Project-scoped (migration 0017): a per-project orchestrator only judges
        # its OWN project's jobs against its OWN project's fleet — another project's
        # worker fitting the model must NOT mask this project's stuck job (exact-match
        # claim means that worker can never take it).
        cur.execute(
            f"""
            WITH live_gpu AS (
                SELECT fits_models, llm_context_max FROM worker_heartbeats
                WHERE queue = 'gpu'
                  AND project = %(project)s
                  AND last_seen > now() - make_interval(secs => %(window)s)
            )
            UPDATE workflow_node_jobs AS j
            SET unassignable_at = NULL, unassignable_reason = NULL
            WHERE j.unassignable_at IS NOT NULL
              AND (j.unassignable_reason IS NULL
                   OR j.unassignable_reason NOT LIKE 'no live worker serves%%')
              AND j.project = %(project)s
              AND (
                    j.status <> 'queued'
                 OR EXISTS (
                        SELECT 1 FROM live_gpu lg
                        WHERE {_serves_here}
                    )
              )
            """.replace(_ANY, _contains),
            {"window": window, "project": proj},
        )
        # FLAG: queued gpu jobs — model-bound or sized — that no fresh GPU worker can
        # serve. Guarded on at least one fresh GPU heartbeat existing (liveness vs
        # capacity).
        cur.execute(
            f"""
            WITH live_gpu AS (
                SELECT fits_models, known_models, vram_total_mb, llm_context_max
                FROM worker_heartbeats
                WHERE queue = 'gpu'
                  AND project = %(project)s
                  AND last_seen > now() - make_interval(secs => %(window)s)
            ),
            fleet AS (
                SELECT count(*) AS n, max(vram_total_mb) AS max_vram,
                       max(llm_context_max) AS max_ctx
                FROM live_gpu
            )
            UPDATE workflow_node_jobs AS j
            SET unassignable_at = now(),
                -- SAY WHICH PROBLEM IT IS. This always blamed VRAM, and once the claim
                -- gate started advertising OBSERVED capability that became wrong far more
                -- often than right: a model installed on no box at all was reported as
                -- "none of N live GPU machine(s) has enough VRAM (max …MB)" — a capacity
                -- verdict for a model whose problem was that nobody had it.
                -- An operator reading that goes looking for a bigger box.
                unassignable_reason = CASE
                    -- THE WINDOW (0028) first: when no box can reach it, that is the
                    -- whole story whatever the model, and the operator's question is
                    -- "how big could any box go?" — so say it, and say that the answer
                    -- already accounts for each box's free memory.
                    WHEN j.context_length IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM live_gpu lg
                        WHERE lg.llm_context_max >= j.context_length
                    ) THEN
                        'no live GPU machine can serve a ' || j.context_length::text
                        || '-token context — none of ' || (SELECT n FROM fleet)::text
                        || ' live GPU machine(s) reaches it (largest reachable: '
                        || COALESCE((SELECT max_ctx FROM fleet)::text, 'unknown')
                        || ' tokens, checked against each box''s free memory)'
                    WHEN j.required_model IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM live_gpu lg
                        WHERE j.required_model = ANY(lg.known_models)
                    ) THEN
                        'no live GPU machine SERVES model ' || j.required_model
                        || ' — none of ' || (SELECT n FROM fleet)::text
                        || ' live GPU machine(s) advertises it (weights not installed, or '
                        || 'its server does not report the tag)'
                    WHEN j.required_model IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM live_gpu lg
                        WHERE j.required_model = ANY(lg.fits_models)
                    ) THEN
                        'no live GPU machine can hold model ' || j.required_model
                        || ' — none of ' || (SELECT n FROM fleet)::text
                        || ' live GPU machine(s) has enough VRAM (max '
                        || COALESCE((SELECT max_vram FROM fleet)::text, 'unknown')
                        || 'MB)'
                    ELSE
                        -- Each half is available somewhere, but never on one box.
                        'no single live GPU machine both holds model ' || j.required_model
                        || ' and reaches a ' || j.context_length::text
                        || '-token context — none of ' || (SELECT n FROM fleet)::text
                        || ' live GPU machine(s) has both'
                END
            WHERE j.queue = 'gpu'
              AND j.project = %(project)s
              AND j.status = 'queued'
              AND (j.required_model IS NOT NULL OR j.context_length IS NOT NULL)
              AND j.unassignable_at IS NULL
              AND (SELECT n FROM fleet) > 0
              AND NOT EXISTS (
                    SELECT 1 FROM live_gpu lg
                    WHERE {_serves_here}
              )
            RETURNING id, run_id, node_id, required_model, context_length,
                      unassignable_reason
            """.replace(_ANY_KNOWN, _knows).replace(_ANY, _contains),
            {"window": window, "project": proj},
        )
        return list(cur.fetchall())


def cancel_queued_jobs_for_run(run_id: str) -> int:
    """Flip all ``queued`` jobs for a run to ``cancelled``. Running jobs
    are left alone — workers notice the run's cancel flag between jobs.
    Returns the number of rows touched.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET status = 'cancelled', finished_at = now() "
            "WHERE run_id = %s AND status = 'queued'",
            (run_id,),
        )
        return cur.rowcount or 0


def cancel_siblings_after_failure(run_id: str) -> int:
    """When a node fails, cancel any still-queued jobs for the run —
    downstream work is moot. Returns rows touched.
    """
    return cancel_queued_jobs_for_run(run_id)


def delete_non_terminal_jobs_for_run(run_id: str) -> list[str]:
    """Restart primitive: delete every job for ``run_id`` whose status is NOT
    ``completed`` / ``skipped``. Returns the list of deleted ``node_id``s so
    the host can cascade into its own artefacts (on-disk node dirs, input
    submissions). Idempotent — a second call returns ``[]``.

    The engine's ``dispatcher._find_ready_nodes`` treats surviving
    ``completed`` / ``skipped`` rows as cursors when re-expanding the DAG: only
    nodes WITHOUT a row whose deps are completed/skipped get enqueued. So a
    follow-up ``dispatcher.start_run`` after this call resumes the run from
    exactly the deleted set (plus any downstream that never got enqueued
    because of the original failure) — the host's "retry whole run" button no
    longer re-does the completed prefix.

    CALLER POLICY: this primitive INCLUDES ``running`` rows in the deletion
    set. If the caller's retry contract needs to refuse while a worker is
    mid-flight (the typical case — the cancel-watcher polls at 5 s, so a
    just-terminated run can briefly still have ``running`` children winding
    down), it must check ``WHERE status='running'`` itself BEFORE calling this.
    The engine's lease-reclaim sweep eventually re-queues a stranded
    ``running`` row, so the worst case here is a single straggler producing
    output to a path the caller then `rm -rf`'s — recoverable, but worth a
    409 from the caller.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM workflow_node_jobs "
            "WHERE run_id = %s "
            "  AND status NOT IN ('completed', 'skipped') "
            "RETURNING node_id",
            (run_id,),
        )
        return [r["node_id"] for r in cur.fetchall()]


def cancel_orphaned_queued_jobs(*, project: str | None = None) -> int:
    """Flip every ``queued`` job whose parent run is already terminal
    (``cancelled`` / ``failed``) to ``cancelled``. Returns the number of rows
    touched.

    The host's cancel handler typically only updates ``workflow_runs.status``;
    it does NOT cascade into ``workflow_node_jobs``. The claim SQL's run-cancel
    guard prevents such jobs from ever running, but they linger in ``queued``
    forever — operator-facing queue gauges then misleadingly suggest a worker
    stall. This sweep is the cleanup. Only ``status='queued'`` rows are touched
    so we don't race the cancel-watcher's cooperative ``running`` cancel.

    Project-scoped (migration 0017): only this orchestrator's project's jobs
    (``None`` ⇒ ``config.project``). The flip is correctness-neutral (the parent
    run is already terminal, so the job could never run), but a per-project
    orchestrator must not write another tenant's rows — kept consistent with the
    other sweeps. Default ``""`` matches all (single-tenant byte-compatible).

    Idempotent: a second call after the first returns 0.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_node_jobs AS j
               SET status = 'cancelled', finished_at = now()
              FROM workflow_runs r
             WHERE j.run_id = r.id
               AND j.project = %s
               AND j.status = 'queued'
               AND r.status IN ('cancelled', 'failed')
            """,
            (_project(project),),
        )
        return cur.rowcount or 0


def set_resolved_inputs(job_id: str, resolved_inputs: dict[str, Any]) -> None:
    """Write the execution-time snapshot of resolved inputs into the
    ``resolved_inputs`` column. Called by the worker just before
    invoking the node module so the snapshot reflects exactly what
    the node received — useful for forensics when the upstream
    context changes between enqueue and execute.

    Pre-validates JSON serialisation so a bad input fails before any
    state mutation, matching :func:`mark_completed`'s contract."""
    json.dumps(resolved_inputs or {})
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET resolved_inputs = %s WHERE id = %s",
            (_as_json(resolved_inputs or {}), job_id),
        )


# ── Queue snapshot ───────────────────────────────────────────────────────


def list_projects() -> list[str]:
    """Distinct ``project`` tenant tags seen across the queue records (migration
    0017) — the option list for a multi-tenant queue/fleet view's project filter.

    Union over runs, node-jobs, ingest jobs, and worker heartbeats; sorted. The
    single-tenant sentinel ``''`` is included when present. Plain SQL — works on
    both the Postgres and SQLite backends."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT project FROM workflow_runs "
            "UNION SELECT project FROM workflow_node_jobs "
            "UNION SELECT project FROM ingest_jobs "
            "UNION SELECT project FROM worker_heartbeats "
            "ORDER BY project"
        )
        return [r["project"] for r in cur.fetchall()]


def snapshot(*, project: str | None = None) -> dict[str, Any]:
    """Return counts and running/queued rows per queue, for the
    queue-indicator UI. Keeps the payload small — at most 50 rows per
    section.

    ``project`` (migration 0017) filters to one tenant; ``None`` (default) is
    the broker-wide view across all projects — byte-compatible with the pre-0017
    single-tenant deploy (every row is ``''``). Each returned row carries its
    ``project`` (``SELECT *``) so a multi-tenant caller can group client-side."""
    pred = "" if project is None else " WHERE project = %(project)s"
    pf = {} if project is None else {"project": project}
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT queue, status, COUNT(*) AS n
            FROM workflow_node_jobs{pred}
            GROUP BY queue, status
            """,
            pf,
        )
        counts: dict[tuple[str, str], int] = {}
        for row in cur.fetchall():
            counts[(row["queue"], row["status"])] = int(row["n"])

        def _top(queue: str, status: str, limit: int = 50) -> list[dict[str, Any]]:
            cur.execute(
                "SELECT * FROM workflow_node_jobs "
                "WHERE queue = %(queue)s AND status = %(status)s"
                + (" AND project = %(project)s" if project is not None else "")
                + " ORDER BY created_at ASC LIMIT %(limit)s",
                {"queue": queue, "status": status, "limit": limit, **pf},
            )
            return list(cur.fetchall())

        return {
            "cpu": {
                "running": _top("cpu", "running"),
                "queued":  _top("cpu", "queued"),
            },
            "gpu": {
                "running": _top("gpu", "running"),
                "queued":  _top("gpu", "queued"),
            },
            "download": {
                "running": _top("download", "running"),
                "queued":  _top("download", "queued"),
            },
            "counts": {
                f"{q}_{s}": n for (q, s), n in counts.items()
            },
        }


def ingest_snapshot(*, project: str | None = None) -> dict[str, Any]:
    """Per-queue depth + live-worker counts for the INGEST path — the
    ``ingest_jobs`` twin of :func:`snapshot` (which covers only the cpu/gpu DAG
    queues). ``queues[q]`` carries the status counts plus ``workers`` = the
    number of ``worker_heartbeats`` rows on that queue still fresh (< 30 s,
    matching the claim worker's 10 s refresh). A host maps queued+running →
    "messages" and ``workers`` → "consumers" for its queue-indicator UI.

    NB: ``worker_heartbeats`` is keyed ``(host_label, queue, project)``, so
    ``workers`` counts live worker *hosts* per queue, not processes — enough to
    drive the "no consumer → starvation" warning.

    ``project`` (migration 0017) filters to one tenant; ``None`` (default) is the
    broker-wide view (byte-compatible with single-tenant)."""
    jpred = "" if project is None else " WHERE project = %(project)s"
    hpred = "" if project is None else " AND project = %(project)s"
    pf = {} if project is None else {"project": project}
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT queue, status, COUNT(*) AS n FROM ingest_jobs{jpred} "
            "GROUP BY queue, status",
            pf,
        )
        counts: dict[str, dict[str, int]] = {}
        for row in cur.fetchall():
            counts.setdefault(row["queue"], {})[row["status"]] = int(row["n"])
        cur.execute(
            "SELECT queue, COUNT(*) AS n FROM worker_heartbeats "
            "WHERE last_seen > now() - make_interval(secs => 30)" + hpred
            + " GROUP BY queue",
            pf,
        )
        workers = {row["queue"]: int(row["n"]) for row in cur.fetchall()}

    queues: dict[str, dict[str, int]] = {}
    for q in set(counts) | set(workers):
        st = counts.get(q, {})
        queues[q] = {
            "queued": st.get("queued", 0),
            "running": st.get("running", 0),
            "completed": st.get("completed", 0),
            "failed": st.get("failed", 0),
            "workers": workers.get(q, 0),
        }
    return {"queues": queues}


def set_resolved_artifact(job_id, artifact: str) -> None:
    """Stamp the ARTIFACT tag a freshly-claimed job will actually run (migration 0027).

    Called by the claim worker immediately after its claim returns, from the worker's own
    live digest observation — see ``ClaimWorker._stamp_resolved_artifact`` for the why.
    Guarded to running rows so a delayed call cannot label a row some other worker has
    since reclaimed; the claim's own UPDATE set status='running' + claimed_by, so between
    the claim and this statement the row is exclusively ours.
    """
    if not job_id or not artifact:
        return
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET resolved_artifact = %s "
            "WHERE id = %s AND status = 'running'",
            (str(artifact), job_id),
        )
        conn.commit()


def upsert_model_catalog(entries) -> int:
    """Publish this box's OBSERVED serving capability (migration 0026). Returns rows written.

    Upsert, never delete-then-insert: a transient probe failure must not blank a box's
    capability and strand every job that needs it. Losing a row costs a missed claim;
    blanking the table costs the whole fleet.

    RETIRE WHAT VANISHED, but only within the (host, server kind) pairs THIS observation
    actually covered. That distinction is the whole safety argument:

      * a kind the probe did not report on is left completely alone — that is the
        transient-failure case above, and it is why this is not a wholesale sync;
      * a kind the probe DID report on is authoritative for that kind, so a model it no
        longer lists is genuinely gone and gets `enabled = FALSE`.

    Without the second half the catalog only ever grew. Observed in practice:
    `chat-expert-b` and `chat-expert-a` were removed from box-a's ollama store and the
    catalog went on advertising both for `box-a-gpu/ollama` indefinitely — the exact
    "declared, not observed" failure
    this table exists to end, reintroduced by its own writer. The heartbeat's `known_models`
    was correct the whole time, which is what made the disagreement visible.

    KNOWN LIMITATION, stated rather than papered over: retirement writes `enabled`, which
    migration 0026 also documents as the OPERATOR's park lever. So a retired row is
    currently indistinguishable from a parked one. Separating them needs a column
    (`retired_at`) and therefore a migration; it has not been done because
    :func:`fetch_model_catalog` still has no non-test caller — the claim gate reads the
    heartbeat's `known_models`, not this table — so the conflict is unreachable today, and
    a schema change to shared infra for an unread table is the larger risk. Wire a
    reader and this needs fixing first.

    (An operator park being cleared by the next observation is NOT new here: the upsert has
    always set `enabled = TRUE` on conflict. This change adds the opposite direction.)
    """
    rows = list(entries or [])
    if not rows:
        return 0
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()

    covered: dict[tuple[str, str], set[str]] = {}
    for e in rows:
        covered.setdefault((e.host_label, e.server_type), set()).add(e.model_id)

    with connection() as conn, conn.cursor() as cur:
        for e in rows:
            cur.execute(
                "INSERT INTO llm_model_catalog "
                "(host_label, server_type, model_id, enabled, observed_at, digest) "
                f"VALUES (%s, %s, %s, TRUE, {_d.now}, %s) "
                "ON CONFLICT (host_label, server_type, model_id) DO UPDATE SET "
                # digest updates with the observation — a promotion re-points the role
                # tag to new bytes and the very next probe must say so, or artifact
                # resolution keeps answering for the previous version.
                f"observed_at = {_d.now}, enabled = TRUE, digest = EXCLUDED.digest",
                (e.host_label, e.server_type, e.model_id,
                 str(getattr(e, "digest", "") or "")),
            )
        for (host, kind), models in covered.items():
            placeholders = ", ".join(["%s"] * len(models))
            cur.execute(
                "UPDATE llm_model_catalog SET enabled = FALSE "
                "WHERE host_label = %s AND server_type = %s AND enabled "
                f"AND model_id NOT IN ({placeholders})",
                (host, kind, *sorted(models)),
            )
        conn.commit()
    return len(rows)


def record_llm_profile(host_label: str, server_type: str, model_id: str, profile: Mapping[str, Any]) -> None:
    """Remember what ``model_id`` COST on ``host_label``'s ``server_type`` (migration
    0029): the numbers its server printed at its latest complete start, keyed with the
    fingerprint of the command line they belong to. A model the catalog has not observed
    yet gets its row; the observation upsert leaves the profile alone. Best-effort — a
    failure here must never reach the heartbeat that called it."""
    import json as _json

    payload = _json.dumps(dict(profile), sort_keys=True)
    from queue_workflows.dialect import get_dialect
    _d = get_dialect()
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE llm_model_catalog SET profile = %s "
                "WHERE host_label = %s AND server_type = %s AND model_id = %s",
                (payload, host_label, server_type, model_id),
            )
            if not cur.rowcount:
                cur.execute(
                    "INSERT INTO llm_model_catalog "
                    "(host_label, server_type, model_id, enabled, observed_at, digest, profile) "
                    f"VALUES (%s, %s, %s, TRUE, {_d.now}, '', %s)",
                    (host_label, server_type, model_id, payload),
                )
            conn.commit()
    except Exception:
        log.warning("[model-catalog] could not record the profile of %s on %s/%s",
                    model_id, host_label, server_type, exc_info=True)


def llm_profile(host_label: str, server_type: str, model_id: str) -> dict | None:
    """The remembered profile of ``model_id`` on that box and server, or ``None``
    (unknown, or a catalog that cannot be read — the conservative direction)."""
    import json as _json

    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT profile FROM llm_model_catalog "
                "WHERE host_label = %s AND server_type = %s AND model_id = %s",
                (host_label, server_type, model_id),
            )
            row = cur.fetchone()
    except Exception:
        log.debug("[model-catalog] profile read failed", exc_info=True)
        return None
    if not row:
        return None
    value = row["profile"] if isinstance(row, Mapping) else row[0]
    if isinstance(value, (str, bytes)):
        try:
            value = _json.loads(value)
        except ValueError:
            return None
    return dict(value) if isinstance(value, Mapping) else None


def fetch_model_catalog(*, stale_after_s: float | None = None) -> list:
    """Every enabled catalog row, optionally excluding ones not observed recently.

    `stale_after_s=None` returns all enabled rows — the conservative default, because a
    box that is merely quiet still holds its models."""
    from queue_workflows.model_catalog import CatalogEntry

    sql = ("SELECT host_label, server_type, model_id, digest FROM llm_model_catalog "
           "WHERE enabled")
    params: dict = {}
    if stale_after_s is not None:
        from queue_workflows.dialect import get_dialect
        _d = get_dialect()
        sql += f" AND observed_at >= {_d.past_seconds('%(stale)s')}"
        params["stale"] = float(stale_after_s)
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return [CatalogEntry(r["host_label"], r["server_type"], r["model_id"],
                                 digest=str(r.get("digest") or ""))
                    for r in (cur.fetchall() or [])]
    except Exception:
        # A catalog read that raises must not break a claim; an empty answer is the
        # conservative direction (see model_catalog.best_match).
        log.debug("[model-catalog] fetch failed", exc_info=True)
        return []


def fleet_snapshot(
    *, stale_after_s: float = 30.0, project: str | None = None,
) -> list[dict[str, Any]]:
    """Read-only per-``(host_label, queue)`` fleet capacity view — the observed
    ``worker_heartbeats`` rows, the telemetry read model a fleet view / operator
    control plane consumes.

    Unlike :func:`snapshot` / :func:`ingest_snapshot` (which only *count*
    heartbeats or join them internally for claim decisions), this returns the
    per-worker rows with their advertised capability (``current_model``,
    ``known_models``, ``llm_servers_available`` [0014], ``vram_total_mb`` /
    ``fits_models`` [0015]). It deliberately surfaces **stale and dead-flagged**
    workers too — that's the point of an observability read — so it returns ALL
    rows ordered by ``(queue, host_label)``, each augmented with two derived
    flags rather than filtering:

      * ``fresh``        — ``last_seen`` within ``stale_after_s`` (default 30 s,
                           matching the claim worker's 10 s refresh × 3);
      * ``flagged_dead`` — the orchestrator's stale-worker detector stamped
                           ``last_flagged_dead_at`` (migration 0009) and no fresh
                           heartbeat has cleared it.

    ``project`` (migration 0017) filters to one tenant's workers; ``None``
    (default) returns the whole fleet across all projects (each row carries its
    ``project`` via ``SELECT *``).

    Pure read; no host coupling. Returns ``[]`` on an empty fleet.
    """
    pred = "" if project is None else " WHERE project = %(project)s"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT *,
                   (last_seen > now() - make_interval(secs => %(stale)s)) AS fresh,
                   (last_flagged_dead_at IS NOT NULL)               AS flagged_dead
            FROM worker_heartbeats{pred}
            ORDER BY queue, host_label
            """,
            {"stale": float(stale_after_s),
             **({} if project is None else {"project": project})},
        )
        return [dict(row) for row in cur.fetchall()]


def recent_jobs(
    *, project: str | None = None, status: str | None = None,
    min_retries: int = 0, limit: int = 40,
) -> list[dict[str, Any]]:
    """Read-only recent-activity feed across BOTH job families — DAG node-jobs
    (``workflow_node_jobs``) and standalone ingest jobs (``ingest_jobs``) — unified
    into one list, newest-first by ``COALESCE(finished_at, started_at, created_at)``.
    The read model a job-dashboard-style "recent activity / retries / dead" view consumes.

    Each row carries a ``kind`` (``'node'`` | ``'ingest'``) and a ``name`` (the
    node id / task name) plus the common lifecycle columns — ``queue``, ``status``,
    ``project``, ``worker`` (``claimed_by``), ``created_at`` / ``started_at`` /
    ``finished_at``, ``seconds``, ``retries`` (``watchdog_retries`` for node-jobs, 0
    for ingest), and ``error``.

    ``project`` (migration 0017) filters to one tenant; ``status`` filters to one
    state (``'failed'`` ⇒ a dead-jobs view, ``'running'`` ⇒ busy); ``min_retries``
    ≥ 1 ⇒ a retries view (node-jobs whose ``watchdog_retries`` reached it; ingest
    jobs have no retry counter, so they drop out). All ``None``/0 ⇒ the broker-wide
    feed. No backend-specific time funcs, so it is portable across the pg / sqlite
    dialects. Returns ``[]`` when empty.
    """
    common: list[str] = []
    params: dict[str, Any] = {"limit": int(limit)}
    if project is not None:
        common.append("project = %(project)s")
        params["project"] = project
    if status is not None:
        common.append("status = %(status)s")
        params["status"] = status
    node_clauses, ingest_clauses = list(common), list(common)
    if min_retries > 0:
        node_clauses.append("watchdog_retries >= %(min_retries)s")
        params["min_retries"] = int(min_retries)
        ingest_clauses.append("1 = 0")  # ingest jobs carry no retry counter
    nwhere = (" WHERE " + " AND ".join(node_clauses)) if node_clauses else ""
    iwhere = (" WHERE " + " AND ".join(ingest_clauses)) if ingest_clauses else ""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, 'node' AS kind, node_id AS name, queue, status, project,
                   claimed_by AS worker, created_at, started_at, finished_at,
                   seconds, watchdog_retries AS retries, error,
                   COALESCE(finished_at, started_at, created_at) AS recency
            FROM workflow_node_jobs{nwhere}
            UNION ALL
            SELECT id, 'ingest' AS kind, task_name AS name, queue, status, project,
                   claimed_by AS worker, created_at, started_at, finished_at,
                   seconds, 0 AS retries, error,
                   COALESCE(finished_at, started_at, created_at) AS recency
            FROM ingest_jobs{iwhere}
            ORDER BY recency DESC
            LIMIT %(limit)s
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


def list_node_events(job_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
    """Read-only per-attempt event timeline for a node-job — the durable
    ``workflow_node_events`` log (migration 0011), the forensic history a
    job-detail view renders: ``claimed`` → ``model_load_*`` → ``progress_beat`` →
    ``stall_*`` / ``gpu_health_trip`` / ``budget_trip`` → ``requeued`` → terminal,
    oldest-first, each row carrying ``attempt`` (the ``watchdog_retries`` at emit,
    tying one node's tries together), ``host_label``, ``elapsed_s``, ``error``, and
    a free-form ``detail``. Returns ``[]`` for an ingest job (no per-attempt log)
    or an unknown id."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, attempt, event_type, host_label, queue, model, "
            "elapsed_s, error, detail, created_at "
            "FROM workflow_node_events WHERE job_id = %(job_id)s "
            "ORDER BY created_at ASC, id ASC LIMIT %(limit)s",
            {"job_id": job_id, "limit": int(limit)},
        )
        return [dict(row) for row in cur.fetchall()]


#: Node-event types that count as a queue ERROR by default (failures the operator cares about).
#: Callers can widen to the watchdog trips: ``stall_trip``, ``gpu_health_trip``, ``budget_trip``.
ERROR_EVENT_KINDS: tuple[str, ...] = ("failed", "error")


def error_snapshot(
    *,
    project: str | None = None,
    limit: int = 200,
    kinds: tuple[str, ...] = ERROR_EVENT_KINDS,
    host_label: str | None = None,
) -> list[dict[str, Any]]:
    """Read-only queue ERROR LOG — recent ``workflow_node_events`` failure rows; the error twin of
    :func:`fleet_snapshot` (which surfaces worker capacity). Returns the most-recent failure events
    newest-first, each carrying run_id / node_id / job_id / host_label / queue / model / error /
    attempt / detail / project / created_at — what an operator error console (a dashboard's
    Errors view) consumes for ANY project.

    ``kinds`` are the event types treated as errors (default :data:`ERROR_EVENT_KINDS`).
    ``project`` (migration 0018) scopes to one tenant; ``None`` returns ALL projects' errors.
    ``host_label`` filters one machine; ``limit`` caps the newest rows. Pure read; no host
    coupling; backend-agnostic SQL; ``[]`` when none."""
    kinds = tuple(kinds) or ERROR_EVENT_KINDS
    placeholders = ", ".join(["%s"] * len(kinds))
    where = [f"event_type IN ({placeholders})"]
    args: list[Any] = list(kinds)
    if project is not None:
        where.append("project = %s")
        args.append(project)
    if host_label is not None:
        where.append("host_label = %s")
        args.append(host_label)
    args.append(int(limit))
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, run_id, node_id, job_id, attempt, event_type,
                   host_label, queue, model, elapsed_s, error, detail, project, created_at
            FROM workflow_node_events
            WHERE {" AND ".join(where)}
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            args,
        )
        return [dict(row) for row in cur.fetchall()]
