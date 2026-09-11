"""ONE SERVER AT A TIME — the drain-based switch state machine.

WHY THIS EXISTS. Moving a GPU box from one LLM server kind to another is not
"unload the old one, start the new one". Each server holds its OWN full copy of the
weights, and the incoming one needs MINUTES to load them (~8 for a 35 GB FP8 model on
a unified-memory GPU box). A box that keeps claiming work during those minutes immediately reloads the
server it just unloaded, so the switch ends with BOTH resident.

That is not theoretical. On box-a:

  * unload ollama, start vLLM, keep claiming
  * ollama reloads within seconds to serve the next job
  * vLLM finishes loading ⇒ both resident, ~3 GB from OOM on a box
    with an OOM death already in its history
  * the residency watchdog resolves the violation by evicting the server with no
    traffic — which is always the INCOMING one, because it only just booted
  * so vLLM was killed on every single attempt, forever

The missing piece is an **out-of-service window**, not an ordering. Doing it by hand
worked: drain the worker, wait for in-flight to reach zero, unload the outgoing
server, start the incoming one, wait until it is ready, resume. This module is that
sequence as a state machine.

    IDLE ─request─> DRAINING ─in-flight 0─> SWITCHING ─outgoing gone─> WARMING
      ^                            │              │                       │
      └──────────────── ready / timeout (ABORT) ──┴───────────────────────┘

The SWITCHING step is not ceremony: the incoming server is started only once the
outgoing one is OBSERVED gone. Issuing both in one breath meant a silently-failed
unload (the ollama lever reports what it actually freed, and it may free nothing)
still started the incoming server — the two-copies-on-one-card bug wearing a
state machine as a hat.

TWO GUARANTEES, and they pull against each other:

  * **Never two loaded kinds.** Claims are blocked from the moment a switch is
    requested until the incoming server is READY — the whole window in which a claim
    could reload the outgoing server. And while WARMING, the incoming kind is
    *protected*: the residency vote and the idle stopper must not evict a server for
    being idle when it has not had the chance to be busy yet.
  * **Never wedged.** A box that can neither claim nor finish a switch is worse than
    one that never switched: it silently stops doing any work. Every phase is
    therefore timeout-bounded and ABORTs back to IDLE, resuming service with whatever
    is actually up — worst case ``drain_timeout + unload_timeout + warm_timeout`` out
    of service, and that whole budget belongs to the REQUEST: it is wall-stamped and
    carried in the record, so a restart mid-switch inherits what is LEFT of it instead
    of re-arming a fresh window. A failed switch costs a slow queue; a wedged box costs
    everything.

KNOWN LIMITS — stated because an independent audit found them and they are real:

  * **Per PROCESS, not per BOX.** The state lives in memory on one worker. Several
    workers of DIFFERENT projects share a physical box (ai_leads and a second consumer both
    run gpu workers on the unified-memory GPU boxes), and a peer keeps claiming through this box's
    switch — reloading the very server being replaced. A worker restart mid-switch
    likewise forgets it and resumes claiming. The engine already has the cross-process
    primitive for this — the flock box lease in ``gpu_model_lease`` — and the switch
    should acquire it; that is deliberately NOT done here, to keep this change to one
    reviewable unit. Until then the guarantee holds for a box whose gpu work is
    claimed by ONE worker.
  * **Only the caller's tick advances it.** The timeout escapes live inside
    :meth:`tick`, so a caller that stops ticking (its heartbeat disabled) leaves the
    box blocked — except for the absolute deadline and the 2x-budget escape, which
    :meth:`blocks_claims` evaluates on the claim path itself.

FIXED after an outage that cost two GPU boxes ~50 % of three hours: a
start that CANNOT succeed is now an observation. The caller reports what its lever did
through :meth:`report_start_outcome`; a ``PERMANENT`` verdict aborts on that same tick
and QUARANTINES the kind (doubling backoff, persisted), so the demand starter cannot
re-request it into another doomed window. Previously a 404 from the start lever was
indistinguishable from "still loading", and the docstring said so: *"No backoff between
failed switches. A target that never becomes ready is re-requested after the
requester's own cooldown, so a permanently-broken server yields a mostly-out-of-service
box rather than one that gives up on it."* That is what happened, live.

DESIGN. Pure decision + injected clock, the same idiom as ``model_cache`` and
``llm_idle_unload``: :meth:`ServerSwitch.tick` takes OBSERVATIONS and returns ACTIONS
for the caller to perform. It does no I/O, so the whole machine is exercised on a
virtual clock with no sleeping, no network and no DB — and the caller stays free to
decide how "unload" and "start" are actually done on its deployment.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)

# ── states ───────────────────────────────────────────────────────────────────

#: No switch in progress; the box serves normally.
IDLE = "idle"
#: A switch was requested; claims are blocked while in-flight work finishes.
DRAINING = "draining"
#: Drained; the outgoing server has been told to unload and we are waiting to OBSERVE
#: it gone. The incoming one is not started until then — see the class docstring.
SWITCHING = "switching"
#: The incoming server has been started and is loading its weights.
WARMING = "warming"

# ── actions the caller must perform ──────────────────────────────────────────

#: Free the outgoing server's memory (unload its model / stop it).
UNLOAD_OUTGOING = "unload_outgoing"
#: Start the incoming server.
START_INCOMING = "start_incoming"
#: The switch completed — the incoming server is serving.
DONE = "done"
#: The switch gave up; the box returns to service with whatever is up.
ABORT = "abort"

#: Defaults. The drain window only has to outlast the longest in-flight LLM call;
#: the warm window has to outlast a cold weight load — and MUST have real headroom
#: over the slowest observed load, because completion is event-driven (the switch
#: closes the moment the server answers) while the timeout only bounds the FAILURE
#: case. 900 s left 18 s of margin against an observed 887 s load from network
#: storage — one slow storage day from aborting every switch. 1800 s costs nothing when the server
#: comes up (nothing waits on the deadline) and only stretches the give-up point
#: for a genuinely broken start, which abort+backoff already handles.
#: Operator overrides: QUEUE_WORKFLOWS_SWITCH_DRAIN_TIMEOUT_S /
#: QUEUE_WORKFLOWS_SWITCH_WARM_TIMEOUT_S (read where the worker constructs its
#: switch — see claim_worker._switch_timeouts).
#: DRAINING must outlast the LONGEST legitimate in-flight job — a chat-model-a
#: reasoning call runs 5-20 minutes, and a 300 s drain aborted four switches in a
#: row against ONE healthy long job (observed live). A draining box is not wedged:
#: in-flight work is progressing and the job's own watchdog (budget 3600 s) bounds
#: a truly stuck one, so the drain ceiling matches that bound.
DEFAULT_DRAIN_TIMEOUT_S = 3600.0
#: SWITCHING (waiting to OBSERVE the outgoing server release) keeps its own short
#: bound — an unload is seconds, not minutes, and a lever that did nothing for this
#: long will not start working.
DEFAULT_UNLOAD_TIMEOUT_S = 300.0
DEFAULT_WARM_TIMEOUT_S = 1800.0
#: How often an unperformed/failed action is re-issued. Actions are emitted once per
#: tick and the caller may fail to perform one (a host hook that raises, a container
#: that vanished); without a retry the box would sit out the whole window having done
#: nothing. Every action we emit is idempotent, so re-issuing is safe.
DEFAULT_RETRY_EVERY_S = 30.0

#: How long a start may stay UNACKNOWLEDGED (the lever reported a transient failure and
#: has not yet reported a success) before the switch gives up. Distinct from the warm
#: budget, which times a start that DID happen and is loading weights: a lever that keeps
#: answering "connection reset" has not started anything, and waiting out a 1800 s weight
#: load for a load that never began is exactly the outage described above. Generous enough for
#: a host lever that itself blocks ~120 s per attempt.
DEFAULT_START_ACK_TIMEOUT_S = 300.0
#: How long a kind stays QUARANTINED after a definitive start failure, doubling per
#: consecutive failure up to the cap. Without it the demand starter simply re-requests
#: after its own cooldown and the box re-enters the doomed window — observed live, twice
#: an hour, for three hours.
DEFAULT_QUARANTINE_S = 900.0
DEFAULT_QUARANTINE_MAX_S = 14400.0

# ── the outcome of asking the caller to START the incoming server ────────────

#: The lever started the container.
STARTED = "started"
#: The lever found it already running (docker's 304) — success for our purposes.
ALREADY_RUNNING = "already_running"
#: The attempt failed in a way that may succeed later (timeout, reset, 5xx).
TRANSIENT = "transient"
#: The attempt CANNOT succeed as configured (no such container, an OCI/mount error, no
#: docker socket, no lever wired). Retrying is pure loss.
PERMANENT = "permanent"
#: The lever told us nothing — every host hook returns ``None`` today. Treated as
#: acknowledged so this whole mechanism is byte-compatible with an un-upgraded host.
UNKNOWN = "unknown"

_STATUSES = frozenset({STARTED, ALREADY_RUNNING, TRANSIENT, PERMANENT, UNKNOWN})
#: Statuses under which a start is considered to be genuinely in flight.
_ACKNOWLEDGED = frozenset({STARTED, ALREADY_RUNNING, UNKNOWN})


@dataclass(frozen=True)
class StartOutcome:
    """What the caller's start lever actually did — the observation the machine was
    missing.

    ``tick`` accepts three observations (in-flight, outgoing loaded, incoming ready), and
    a start that FAILED looks identical to one still loading: not ready yet. So a box
    whose vLLM container did not exist retried the start every 30 s for the whole warm
    window, aborted, and was re-requested. This value object is the missing fourth
    observation, reported through :meth:`ServerSwitch.report_start_outcome`.

    The host lever may return one of these, a plain ``{"status": ..., "reason": ...}``
    dict (so a host file can be deployed against an OLDER engine wheel without importing
    anything), a bool, or ``None``. :meth:`coerce` normalises all of them and never
    raises — a lever's return value must never be able to break the switch.
    """

    status: str
    reason: str = ""

    @property
    def acknowledged(self) -> bool:
        """Is a start plausibly in flight? False only when the lever told us it failed."""
        return self.status in _ACKNOWLEDGED

    @property
    def fatal(self) -> bool:
        return self.status == PERMANENT

    @classmethod
    def started(cls, reason: str = "") -> "StartOutcome":
        return cls(STARTED, reason)

    @classmethod
    def already_running(cls, reason: str = "") -> "StartOutcome":
        return cls(ALREADY_RUNNING, reason)

    @classmethod
    def transient(cls, reason: str = "") -> "StartOutcome":
        return cls(TRANSIENT, reason)

    @classmethod
    def permanent(cls, reason: str = "") -> "StartOutcome":
        return cls(PERMANENT, reason)

    @classmethod
    def unknown(cls, reason: str = "") -> "StartOutcome":
        return cls(UNKNOWN, reason)

    @classmethod
    def coerce(cls, value: Any) -> "StartOutcome":
        """Normalise whatever a host lever returned. Never raises."""
        try:
            if isinstance(value, StartOutcome):
                return value
            if value is None:
                return cls.unknown()
            if isinstance(value, bool):
                return cls.started() if value else cls.transient()
            raw: Any = None
            if isinstance(value, dict):
                raw = value.get("status")
                reason = str(value.get("reason") or "")
            else:
                raw = getattr(value, "status", None)
                reason = str(getattr(value, "reason", "") or "")
            status = str(raw or "").strip().lower()
            if status in _STATUSES:
                return cls(status, reason)
            return cls.unknown(reason)
        except Exception:      # a host object with a hostile __getattr__
            return cls.unknown()


class ServerSwitch:
    """The per-box switch state. Thread-safe; all I/O is the caller's.

    Typical use from a worker::

        if switch.blocks_claims():
            return None                     # out of service for this switch
        for action in switch.tick(
            in_flight=self._gpu_active_count(),
            outgoing_loaded=ollama_has_a_model,
            incoming_ready=vllm_answers,
        ):
            ...                             # perform it
    """

    def __init__(
        self,
        *,
        now_fn: Callable[[], float] = time.monotonic,
        drain_timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
        unload_timeout_s: float = DEFAULT_UNLOAD_TIMEOUT_S,
        warm_timeout_s: float = DEFAULT_WARM_TIMEOUT_S,
        retry_every_s: float = DEFAULT_RETRY_EVERY_S,
        start_ack_timeout_s: float = DEFAULT_START_ACK_TIMEOUT_S,
        quarantine_s: float = DEFAULT_QUARANTINE_S,
        quarantine_max_s: float = DEFAULT_QUARANTINE_MAX_S,
        label: str = "?",
        persist_path: str | None = None,
        wall_clock_fn: Callable[[], float] = time.time,
    ) -> None:
        self._now = now_fn
        self._drain_timeout_s = float(drain_timeout_s)
        self._unload_timeout_s = float(unload_timeout_s)
        self._warm_timeout_s = float(warm_timeout_s)
        self._start_ack_timeout_s = float(start_ack_timeout_s)
        self._quarantine_s = float(quarantine_s)
        self._quarantine_max_s = float(quarantine_max_s)
        self._label = label
        self._lock = threading.RLock()
        self._retry_every_s = float(retry_every_s)
        self._state = IDLE
        self._target: str | None = None
        self._since = 0.0
        self._last_try = 0.0
        # ── the start observation ────────────────────────────────────────────
        # A start is "acknowledged" unless the lever explicitly told us it failed.
        # Optimistic on purpose: every host lever returns None today, and treating
        # silence as failure would cut short every healthy weight load.
        self._start_ack = True
        self._start_unack_since: float | None = None
        self._last_start: dict | None = None
        # ── the deadline, owned by the REQUEST not by the process ────────────
        # Wall-stamped at request time and carried in the record, so a restart
        # INHERITS the remaining budget instead of re-arming a full window.
        self._requested_wall: float | None = None
        # kind -> {until_wall, failures, reason, since_wall}
        self._quarantine: dict[str, dict] = {}
        # ── persistence (analysis #1, restart half) ──────────────────────────
        # The switch state lived only in this process, so a worker restart
        # mid-switch FORGOT it: the fresh process resumed claiming while the
        # incoming server warmed — the exact race the machine exists to close.
        # A small JSON file (in the box-local lease dir, never NFS) survives the
        # restart; on construction a fresh, in-flight record re-enters DRAINING —
        # the safest re-entry, since every later phase is re-derived from
        # observations anyway. Wall-clock (not monotonic) because it must be
        # meaningful across processes; staleness is bounded by the total switch
        # budget so a record from a long-dead switch is ignored and removed.
        self._persist_path = persist_path
        self._wall = wall_clock_fn
        self._resume_persisted()

    # ── read model ───────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def target(self) -> str | None:
        """The kind being switched TO, or ``None`` when idle."""
        with self._lock:
            return self._target

    @property
    def last_start(self) -> dict | None:
        """The last start outcome reported, as a plain dict (for logs/operators)."""
        with self._lock:
            return dict(self._last_start) if self._last_start else None

    def total_budget_s(self) -> float:
        """The worst-case out-of-service window for ONE switch, all phases summed."""
        return self._budget_total()

    def remaining_budget_s(self) -> float:
        """How much of THIS switch's original budget is left. Measured from the wall
        time of the request, so it is the same number before and after a restart."""
        with self._lock:
            if self._state == IDLE:
                return 0.0
            return self._remaining_locked()

    def start_in_flight(self) -> bool:
        """Is a start of the incoming server genuinely in flight right now?"""
        with self._lock:
            if self._state != WARMING:
                return False
            if self._start_ack:
                return True
            since = self._start_unack_since
            return since is not None and (
                self._now() - since < self._start_ack_timeout_s)

    def quarantined(self, kind: str) -> bool:
        """Is this kind currently held back after a definitive start failure?"""
        with self._lock:
            return self._quarantined_locked(kind)

    def quarantine_remaining_s(self, kind: str) -> float:
        with self._lock:
            self._prune_quarantine_locked()
            entry = self._quarantine.get(kind)
            if not entry:
                return 0.0
            return max(0.0, float(entry.get("until_wall") or 0.0) - self._wall())

    def clear_quarantine(self, kind: str) -> None:
        """Drop the hold on a kind. Called when that kind is OBSERVED serving — an
        operator who repairs and starts the container by hand must not wait out a
        quarantine that the observation has already disproved."""
        with self._lock:
            if self._quarantine.pop(kind, None) is not None:
                log.info("[server-switch] %s: %s is serving again — quarantine cleared",
                         self._label, kind)
                self._persist()

    def blocks_claims(self) -> bool:
        """Is this box out of service right now?

        True for the WHOLE switch — from the request until the incoming server is
        ready. This is the guarantee: a claim taken mid-switch is what reloads the
        outgoing server and produces the two-resident state.

        DEADLINE-AWARE (analysis #6 / audit D4): the timeout escapes used to live
        only inside :meth:`tick`, so a caller that stopped ticking (its heartbeat
        disabled — config-reachable) left the box blocked forever. Now this read
        itself self-aborts once DOUBLE the current phase's budget has passed with no
        tick-driven resolution: the claim path calls it every loop, so even a
        tick-less box unblocks. Double, so the normal tick-driven abort always wins
        when ticks are flowing."""
        with self._lock:
            if self._state == IDLE:
                return False
            # THE ABSOLUTE DEADLINE, from the wall time of the original request. A
            # restart used to re-enter DRAINING with fresh phase clocks, so a switch was
            # still blocking claims more than three hours after it was requested, across
            # a reboot and a manual restart. The budget belongs to the request.
            if self._remaining_locked() <= 0:
                log.warning(
                    "[server-switch] %s: the switch to %s outlived its whole budget "
                    "(%.0fs from the original request) — aborting and resuming service",
                    self._label, self._target, self._budget_total(),
                )
                self._reset()
                return False
            budget = (self._warm_timeout_s if self._state == WARMING
                      else self._unload_timeout_s if self._state == SWITCHING
                      else self._drain_timeout_s)
            if self._now() - self._since >= budget * 2:
                log.warning(
                    "[server-switch] %s: %s exceeded 2x its budget with no tick "
                    "resolving it (ticks stalled?) — self-aborting from the claim "
                    "path to put the box back to work", self._label, self._state,
                )
                self._reset()
                return False
            return True

    def protected_kind(self) -> str | None:
        """The kind that must NOT be evicted right now, or ``None``.

        A server that is still loading has no traffic, so any "evict the idle one"
        vote picks it — which is precisely how the incoming server was killed on
        every attempt. Callers (the residency enforcer, the idle stopper) consult
        this before acting."""
        with self._lock:
            return self._target if self._state == WARMING else None

    # ── transitions ──────────────────────────────────────────────────────────

    def request(self, target: str) -> bool:
        """Ask to switch this box to ``target``. Returns True iff a switch started.

        Refused (False) when one is already in progress — including a request for the
        same target, so a caller polling "should I be on vLLM?" every tick starts one
        switch, not one per tick."""
        if not target:
            return False
        with self._lock:
            if self._state != IDLE:
                return False
            self._prune_quarantine_locked()
            if self._quarantined_locked(target):
                entry = self._quarantine[target]
                log.info(
                    "[server-switch] %s: NOT switching to %s — it is quarantined for "
                    "another %.0fs after %d failed start(s): %s",
                    self._label, target,
                    max(0.0, float(entry.get("until_wall") or 0.0) - self._wall()),
                    int(entry.get("failures") or 0), entry.get("reason") or "?",
                )
                return False
            self._state = DRAINING
            self._target = target
            self._since = self._now()
            self._requested_wall = self._wall()
            self._start_ack = True
            self._start_unack_since = None
            self._last_start = None
            self._persist()
        log.info(
            "[server-switch] %s: switching to %s — DRAINING (claims blocked until it "
            "is ready)", self._label, target,
        )
        return True

    def tick(
        self, *, in_flight: int, outgoing_loaded: bool, incoming_ready: bool,
    ) -> list[str]:
        """Advance the machine against fresh observations; return actions to perform.

        ``in_flight`` — jobs still running on this box; ``outgoing_loaded`` — whether
        the server being replaced still holds memory; ``incoming_ready`` — whether the
        new one is answering.
        """
        with self._lock:
            if self._state == IDLE:
                return []
            now = self._now()
            waited = now - self._since
            # The absolute deadline first: no phase may extend the window past the
            # budget the ORIGINAL request was given (see blocks_claims).
            if self._remaining_locked() <= 0:
                log.warning(
                    "[server-switch] %s: the switch to %s outlived its whole budget "
                    "(%.0fs from the original request) — ABORTING and resuming "
                    "service", self._label, self._target, self._budget_total(),
                )
                return self._abort()

            if self._state == DRAINING:
                if int(in_flight) > 0:
                    if waited >= self._drain_timeout_s:
                        # THE KILL LINE. Work that will not drain must not take the
                        # box out of service permanently — going back to serving on
                        # the old server is strictly better than serving nothing.
                        log.warning(
                            "[server-switch] %s: drain timed out after %.0fs with %d "
                            "job(s) still in flight — ABORTING the switch to %s and "
                            "resuming service", self._label, waited, int(in_flight),
                            self._target,
                        )
                        return self._abort()
                    return []
                self._state = SWITCHING
                self._since = now
                self._last_try = now      # the transition itself emits the first try
                log.info("[server-switch] %s: drained — freeing the outgoing server",
                         self._label)
                return [UNLOAD_OUTGOING] if outgoing_loaded else []

            if self._state == SWITCHING:
                if outgoing_loaded:
                    # NOT started yet, on purpose. The incoming server is only
                    # started once the outgoing one is OBSERVED gone — a "start
                    # anyway" is how the box ended up holding two 34 GB copies. A
                    # silently-failed unload therefore stalls here and aborts on the
                    # timeout, which costs a slow queue instead of an OOM.
                    if waited >= self._unload_timeout_s:
                        log.warning(
                            "[server-switch] %s: the outgoing server did not release "
                            "within %.0fs — ABORTING rather than starting %s on top "
                            "of it", self._label, waited, self._target,
                        )
                        return self._abort()
                    if now - self._last_try >= self._retry_every_s:
                        self._last_try = now
                        return [UNLOAD_OUTGOING]      # retry the free
                    return []
                self._state = WARMING
                self._since = now
                self._last_try = now
                self._start_ack = True          # optimistic until a lever says otherwise
                self._start_unack_since = None
                self._persist()
                log.info("[server-switch] %s: outgoing server is gone — starting %s "
                         "(WARMING)", self._label, self._target)
                return [START_INCOMING]

            # WARMING
            if incoming_ready:
                target = self._target
                # SUCCESS CLEARS THE HOLD. The entry survives its own expiry on purpose
                # (it carries `failures`, so a permanently-broken target escalates), but
                # a completed switch is proof the kind is healthy — without this, every
                # slow-but-successful load would keep doubling the next hold until the
                # box refused vLLM for hours.
                if target:
                    self._quarantine.pop(target, None)
                self._reset()
                log.info(
                    "[server-switch] %s: %s is ready after %.0fs — switch COMPLETE, "
                    "resuming service", self._label, target, waited,
                )
                return [DONE]
            if waited >= self._warm_timeout_s:
                log.warning(
                    "[server-switch] %s: %s did not become ready within %.0fs — "
                    "ABORTING and resuming service on whatever is up",
                    self._label, self._target, waited,
                )
                return self._abort_and_quarantine(
                    f"not ready within {waited:.0f}s")
            # THE ACK BUDGET. A lever that keeps reporting a transient failure has
            # started NOTHING, so the warm budget — which times a weight load that is
            # actually happening — is the wrong bound. Give up much sooner.
            if not self._start_ack and self._start_unack_since is not None:
                unacked = now - self._start_unack_since
                if unacked >= self._start_ack_timeout_s:
                    reason = (self._last_start or {}).get("reason") or "?"
                    log.warning(
                        "[server-switch] %s: the start of %s was never acknowledged in "
                        "%.0fs (last: %s) — ABORTING and resuming service",
                        self._label, self._target, unacked, reason,
                    )
                    return self._abort_and_quarantine(reason)
            if outgoing_loaded:
                # The outgoing server CAME BACK mid-warm (the switch is per-process;
                # a peer worker on the same box can reload it — see KNOWN LIMITS).
                # Re-issuing START now would be the two-copies bug through the retry
                # door: the round-2 audit fuzzed 1,486 violations, every one here.
                # Hold the retry; the warm timeout still bounds the wait, and the
                # start resumes the moment the outgoing is observed gone again.
                return []
            if now - self._last_try >= self._retry_every_s:
                # A start that FAILED (the host hook raised, the container was gone)
                # would otherwise leave the box with NO server at all for the whole
                # warm window — the actions are emitted once and nobody retries them.
                # Re-issuing is safe because starting an already-running server is a
                # no-op success on every backend we drive.
                self._last_try = now
                return [START_INCOMING]
            return []

    def report_start_outcome(self, outcome: Any) -> list[str]:
        """Tell the machine what the START it just asked for actually did.

        THE SEAM the outage in the module docstring needed. The caller performs ``START_INCOMING``
        and immediately reports the result here — in the SAME tick, so a definitive
        failure ends the out-of-service window at once instead of at the warm timeout
        ten ticks later. Returns any further actions to perform (``[ABORT]`` when the
        switch is given up).

        Accepts anything :meth:`StartOutcome.coerce` understands, so a caller that does
        not know about outcomes (every host today, returning ``None``) keeps exactly
        today's behaviour."""
        result = StartOutcome.coerce(outcome)
        with self._lock:
            if self._state == IDLE:
                return []
            self._last_start = {"status": result.status, "reason": result.reason,
                                "wall": self._wall()}
            if result.fatal:
                log.error(
                    "[server-switch] %s: the start of %s CANNOT succeed (%s) — "
                    "ABORTING now and quarantining it; the box goes back to work",
                    self._label, self._target, result.reason or "no reason given",
                )
                return self._abort_and_quarantine(result.reason)
            if result.acknowledged:
                self._start_ack = True
                self._start_unack_since = None
            elif self._start_ack:
                self._start_ack = False
                self._start_unack_since = self._now()
                log.warning(
                    "[server-switch] %s: the start of %s failed transiently (%s) — "
                    "retrying, but only for %.0fs",
                    self._label, self._target, result.reason or "?",
                    self._start_ack_timeout_s,
                )
            self._persist()
            return []

    def abandon(self, reason: str) -> None:
        """Cancel an in-flight switch and return to service, clearing persistence.

        Exists for the CAPABILITY case (found live): a persisted switch to a kind
        this box cannot serve — box-c (ROCm) resumed a vllm switch after a
        restart and sat out of service warming a server that can never exist. The
        resume path cannot know capabilities; the worker can, and calls this."""
        with self._lock:
            if self._state == IDLE:
                return
            target, state = self._target, self._state
            self._reset()
        log.warning("[server-switch] %s: ABANDONING the switch to %s (was %s): %s",
                    self._label, target, state, reason)

    # ── internals (caller holds the lock) ────────────────────────────────────

    def _abort(self) -> list[str]:
        self._reset()
        return [ABORT]

    def _abort_and_quarantine(self, reason: str) -> list[str]:
        """Give up on this switch AND hold the target kind back for a while. Called
        wherever the evidence says the target cannot come up: a definitive lever
        failure, an unacknowledged start, a warm window that expired."""
        target = self._target
        if target:
            self._quarantine_locked(target, reason)
        return self._abort()

    def _reset(self) -> None:
        self._state = IDLE
        self._target = None
        self._since = 0.0
        self._last_try = 0.0
        self._requested_wall = None
        self._start_ack = True
        self._start_unack_since = None
        # NOT the quarantine: it must outlive the switch it came from (and the
        # process), or the demand starter simply asks again.
        self._persist()

    # ── quarantine ───────────────────────────────────────────────────────────

    def _quarantine_locked(self, kind: str, reason: str) -> None:
        entry = self._quarantine.get(kind) or {}
        failures = int(entry.get("failures") or 0)
        hold = min(self._quarantine_s * (2 ** min(failures, 8)),
                   self._quarantine_max_s)
        now_wall = self._wall()
        self._quarantine[kind] = {
            "until_wall": now_wall + hold,
            "failures": failures + 1,
            "reason": str(reason or "")[:300],
            "since_wall": now_wall,
        }
        log.warning(
            "[server-switch] %s: %s is quarantined for %.0fs after %d failed "
            "start(s): %s", self._label, kind, hold, failures + 1, reason or "?",
        )

    def _quarantined_locked(self, kind: str) -> bool:
        entry = self._quarantine.get(kind)
        return bool(entry) and float(entry.get("until_wall") or 0.0) > self._wall()

    def _prune_quarantine_locked(self) -> None:
        """Forget an entry only once it has been EXPIRED for a long time.

        An expired entry is deliberately kept: it carries ``failures``, and dropping it
        on expiry would reset the backoff, so a permanently-broken target would be
        retried every ``quarantine_s`` for ever instead of escalating. It is dropped
        after a full cap-length of not being re-tried — by then the box has had every
        chance to prove the kind healthy."""
        cutoff = self._wall() - self._quarantine_max_s
        for kind in [k for k, v in self._quarantine.items()
                     if float(v.get("until_wall") or 0.0) <= cutoff]:
            self._quarantine.pop(kind, None)

    # ── persistence ──────────────────────────────────────────────────────────

    def _budget_total(self) -> float:
        """The worst case out of service for one switch: each phase's own bound."""
        return self._drain_timeout_s + self._unload_timeout_s + self._warm_timeout_s

    def _remaining_locked(self) -> float:
        """Budget left for the CURRENT switch, from the wall time of its request."""
        if self._requested_wall is None:
            # NOT `if not …`: a wall clock reading exactly 0.0 is falsy, and that would
            # silently disarm the deadline for the life of the switch.
            return self._budget_total()
        return self._budget_total() - (self._wall() - self._requested_wall)

    def _persist(self) -> None:
        """Write the record — or remove it when there is nothing left to remember.

        Record v2 keeps two independent things: the in-flight switch (with the wall
        time of the REQUEST, so a restart inherits the remaining budget) and the
        quarantine (which must survive both the abort and the process)."""
        if not self._persist_path:
            return
        self._prune_quarantine_locked()
        if self._state == IDLE and not self._quarantine:
            self._unpersist()
            return
        rec: dict = {"v": 2, "quarantine": self._quarantine}
        if self._state != IDLE:
            rec["switch"] = {
                "target": self._target,
                "requested_wall": (self._requested_wall
                                   if self._requested_wall is not None
                                   else self._wall()),
                "phase": self._state,
                "start_acknowledged": self._start_ack,
                "last_start": self._last_start,
            }
        try:
            tmp = f"{self._persist_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            os.replace(tmp, self._persist_path)
        except Exception:
            log.debug("[server-switch] %s: persist failed (non-fatal)", self._label,
                      exc_info=True)

    def _unpersist(self) -> None:
        if not self._persist_path:
            return
        try:
            os.unlink(self._persist_path)
        except FileNotFoundError:
            pass
        except Exception:
            log.debug("[server-switch] %s: unpersist failed (non-fatal)", self._label,
                      exc_info=True)

    def _resume_persisted(self) -> None:
        if not self._persist_path:
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
            if not isinstance(rec, dict):
                self._unpersist()
                return
            if int(rec.get("v") or 0) >= 2:
                self._quarantine = {
                    str(k): dict(v) for k, v in (rec.get("quarantine") or {}).items()
                    if isinstance(v, dict)
                }
                self._prune_quarantine_locked()
                sw_rec = rec.get("switch") or {}
                target = str(sw_rec.get("target") or "")
                raw_wall = sw_rec.get("requested_wall")
                requested_wall = None if raw_wall is None else float(raw_wall)
            else:
                # v1: {"target": ..., "wall": ...} — the shape older engines persisted.
                target = str(rec.get("target") or "")
                raw_wall = rec.get("wall")
                requested_wall = None if raw_wall is None else float(raw_wall)
            age = (self._wall() - requested_wall
                   if requested_wall is not None else None)
            if target and self._quarantined_locked(target):
                log.warning(
                    "[server-switch] %s: dropping a persisted switch to %s — that "
                    "kind is quarantined", self._label, target,
                )
            elif target and age is not None and 0 <= age < self._budget_total():
                self._state = DRAINING
                self._target = target
                self._since = self._now()
                self._requested_wall = requested_wall
                log.warning(
                    "[server-switch] %s: resuming an in-flight switch to %s from a "
                    "previous process (age %.0fs, %.0fs of its budget left) — claims "
                    "stay blocked until it completes or that budget runs out",
                    self._label, target, age, self._remaining_locked(),
                )
            self._persist()
        except FileNotFoundError:
            pass
        except Exception:
            log.debug("[server-switch] %s: resume failed (ignored)", self._label,
                      exc_info=True)
            self._unpersist()


__all__ = [
    "ServerSwitch", "StartOutcome",
    "IDLE", "DRAINING", "SWITCHING", "WARMING",
    "UNLOAD_OUTGOING", "START_INCOMING", "DONE", "ABORT",
    "STARTED", "ALREADY_RUNNING", "TRANSIENT", "PERMANENT", "UNKNOWN",
    "DEFAULT_DRAIN_TIMEOUT_S", "DEFAULT_WARM_TIMEOUT_S", "DEFAULT_RETRY_EVERY_S",
    "DEFAULT_START_ACK_TIMEOUT_S", "DEFAULT_QUARANTINE_S",
    "DEFAULT_QUARANTINE_MAX_S",
]
