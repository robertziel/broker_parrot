"""ONE SERVER AT A TIME — the drain-based switch state machine.

WHY THIS EXISTS. Moving a GPU box from one LLM server kind to another is not
"unload the old one, start the new one". Each server holds its OWN full copy of the
weights, and the incoming one needs MINUTES to load them (~8 for a 35 GB FP8 model on
a unified-memory GPU box). A box that keeps claiming work during those minutes immediately reloads the
server it just unloaded, so the switch ends with BOTH resident.

That is not theoretical. On box-a:

  * unload ollama, start vLLM, keep claiming
  * ollama reloads within seconds to serve the next job
  * vLLM finishes loading ⇒ 117.7/121 GB, both resident, ~3 GB from OOM on a box
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
    is actually up — worst case ``drain_timeout + drain_timeout + warm_timeout`` out
    of service (DRAINING and SWITCHING each get the drain budget; 1500 s at the
    defaults). A failed switch costs a slow queue; a wedged box costs everything.

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
    box blocked. No independent watchdog exists.
  * **No backoff between failed switches.** A target that never becomes ready is
    re-requested after the requester's own cooldown, so a permanently-broken server
    yields a mostly-out-of-service box rather than one that gives up on it.

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
from typing import Callable

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
#: case. 900 s left 18 s of margin against box-b's 887 s NFS-mounted load — one
#: slow NFS day from aborting every switch. 1800 s costs nothing when the server
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
        label: str = "?",
        persist_path: str | None = None,
        wall_clock_fn: Callable[[], float] = time.time,
    ) -> None:
        self._now = now_fn
        self._drain_timeout_s = float(drain_timeout_s)
        self._unload_timeout_s = float(unload_timeout_s)
        self._warm_timeout_s = float(warm_timeout_s)
        self._label = label
        self._lock = threading.RLock()
        self._retry_every_s = float(retry_every_s)
        self._state = IDLE
        self._target: str | None = None
        self._since = 0.0
        self._last_try = 0.0
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
            self._state = DRAINING
            self._target = target
            self._since = self._now()
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
                log.info("[server-switch] %s: outgoing server is gone — starting %s "
                         "(WARMING)", self._label, self._target)
                return [START_INCOMING]

            # WARMING
            if incoming_ready:
                target = self._target
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
                return self._abort()
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

    def _reset(self) -> None:
        self._state = IDLE
        self._target = None
        self._since = 0.0
        self._last_try = 0.0
        self._unpersist()

    # ── persistence ──────────────────────────────────────────────────────────

    def _budget_total(self) -> float:
        return self._drain_timeout_s * 2 + self._warm_timeout_s

    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            tmp = f"{self._persist_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"target": self._target, "wall": self._wall()}, fh)
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
            target = str(rec.get("target") or "")
            age = self._wall() - float(rec.get("wall") or 0)
            if target and 0 <= age < self._budget_total():
                self._state = DRAINING
                self._target = target
                self._since = self._now()
                log.warning(
                    "[server-switch] %s: resuming an in-flight switch to %s from a "
                    "previous process (age %.0fs) — claims stay blocked until it "
                    "completes or times out", self._label, target, age,
                )
            else:
                self._unpersist()
        except FileNotFoundError:
            pass
        except Exception:
            log.debug("[server-switch] %s: resume failed (ignored)", self._label,
                      exc_info=True)
            self._unpersist()


__all__ = [
    "ServerSwitch",
    "IDLE", "DRAINING", "SWITCHING", "WARMING",
    "UNLOAD_OUTGOING", "START_INCOMING", "DONE", "ABORT",
    "DEFAULT_DRAIN_TIMEOUT_S", "DEFAULT_WARM_TIMEOUT_S", "DEFAULT_RETRY_EVERY_S",
]
