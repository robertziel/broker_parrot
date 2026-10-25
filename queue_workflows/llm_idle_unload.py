"""Free an EXTERNAL LLM server's model from RAM after the GPU box goes idle.

WHY THIS EXISTS. The engine already reaps its own idle GPU memory in two places:
:mod:`queue_workflows.model_cache` unloads a warm IN-PROCESS model, and the vLLM
:class:`~queue_workflows.llm_backends.supervisor.LLMSupervisor` stops an idle vLLM
sidecar. But a box that serves its LLM via **ollama** is left to ollama's own
``OLLAMA_KEEP_ALIVE`` timer — engine-uncontrolled — so a model can sit resident in
VRAM/RAM long after the box stops doing GPU work (and even while the GPU worker is
PARKED). This reaper closes that gap: when the GPU worker has done no GPU work for
``ttl_s`` (default 5 minutes), it asks the box's ollama what it is holding and
unloads it (a ``keep_alive: 0`` request), giving the RAM back.

WHAT COUNTS AS "USED". The idle clock is NOT the worker's claim activity — that
proved wrong both ways on a live box: memoized queue jobs reset a
claim-based clock without ever touching the model (so it never unloaded), while
host-internal LLM calls (a node's own HTTP client, an app-side captioner) were
invisible to it (so it could unload a model another client was actively using).
The one signal every client shares is the SERVER's: ollama bumps each loaded
model's ``/api/ps`` ``expires_at`` on every request, so the reaper watches that
value between ticks — an advance = the model was used, a frozen value ages.
First sighting of a loaded model stamps "used now" (it was just loaded; worst
case an unknown-history model gets one extra TTL of grace). An unparseable
expiry degrades to that sighting clock. ``active_fn`` (running node-jobs)
remains the in-flight guard so a claimed-but-not-yet-dispatched job is never
yanked out from under.

MIRRORS ``model_cache.py`` on purpose so the two read the same:

  * the pure decision :func:`should_unload_external_model` is the analog of
    ``gpu_should_unload`` (loaded + nothing running + idle ≥ ttl; ttl<=0 disables);
  * :meth:`ExternalModelIdleReaper.reap_idle_once` is one tick of the reaper;
  * :meth:`_loop` polls on the same ``max(5, min(60, ttl/5))`` cadence;
  * :meth:`ensure_started` gates on ttl>0 + the ``QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER``
    kill-switch (shared with ModelCache — one switch silences both idle reapers).

DECOUPLING. Everything is injected — ``active_fn`` (the GPU worker's busy
signal), ``loaded_models_info_fn`` / ``unload_fn`` (the ollama HTTP, wired to
:mod:`queue_workflows.llm_probe`), and ``now_fn`` / ``sleep_fn`` — so the reaper
runs on a virtual clock against a fake ollama with no real waiting and no
network. It imports nothing from the backend modules. Best-effort throughout: a
tick that can't reach ollama unloads nothing and never raises, exactly like
ModelCache's.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from queue_workflows.envcompat import env_get

log = logging.getLogger(__name__)

#: THE idle-model constant (seconds): how long an LLM-server model may stay
#: loaded while UNUSED before it is unloaded; <=0 disables. 300 (5 min) — long
#: enough to keep a hot model through a lull between jobs, short enough that a
#: quiet or PARKED box frees its RAM promptly. Resolution order (see
#: :func:`resolve_llm_idle_ttl_s`): env ``QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S``
#: → ``configure(llm_server_idle_ttl_s=…)`` → this default.
DEFAULT_LLM_IDLE_TTL_S = 300.0


def resolve_llm_idle_ttl_s() -> float:
    """The effective idle TTL, resolved at CALL time (not import time, so a
    host's ``configure()`` — which runs after import — is honoured): the
    ``QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S`` env knob wins (operator override,
    read via envcompat), then ``EngineConfig.llm_server_idle_ttl_s``,
    then :data:`DEFAULT_LLM_IDLE_TTL_S`. A malformed env value falls through."""
    raw = env_get("QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    try:
        from queue_workflows.config import get_config
        val = getattr(get_config(), "llm_server_idle_ttl_s", None)
        if val is not None:
            return float(val)
    except Exception:
        pass
    return DEFAULT_LLM_IDLE_TTL_S

#: Poll-cadence bounds (mirror ModelCache._idle_reaper_loop's clamp).
_POLL_FLOOR_S = 5.0
_POLL_CEIL_S = 60.0

#: Shared with ModelCache — one env kill-switch silences BOTH idle reapers (tests).
_DISABLE_ENV = "QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER"


def should_unload_external_model(
    model_loaded: bool, active: int, idle_s: float, ttl_s: float,
) -> bool:
    """Pure decision: unload only when a model IS loaded on the server, no GPU job
    is running (``active <= 0``), and the model has gone UNUSED — no observed
    server-side use — for at least ``ttl_s``. ``ttl_s <= 0`` disables idle unload
    entirely. Same shape as :func:`queue_workflows.model_cache.gpu_should_unload`."""
    if ttl_s <= 0:
        return False
    return bool(model_loaded) and active <= 0 and idle_s >= ttl_s


class ExternalModelIdleReaper:
    """Daemon that unloads an idle ollama box's model. All I/O injected (see module
    docstring). One tick = :meth:`reap_idle_once`; :meth:`ensure_started` arms the
    background poller once.

    ``loaded_models_info_fn`` returns ``[{"name": str, "mru": float}]`` — the
    :func:`queue_workflows.llm_probe.loaded_models_info` shape, ``mru`` being the
    parsed ``expires_at``. The reaper keeps a per-model ``(mru, stamped_at)``
    memory across ticks: a HIGHER ``mru`` than last seen means some client used
    the model since the previous tick (any client — queue node, host-internal
    HTTP, an app captioner), which re-stamps the use clock."""

    def __init__(
        self,
        *,
        active_fn: Callable[[], int],
        loaded_models_info_fn: Callable[[], list[dict]],
        unload_fn: Callable[[list[str]], object],
        ttl_s: float | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        label: str = "?",
    ) -> None:
        self._active_fn = active_fn
        self._loaded_models_info_fn = loaded_models_info_fn
        self._unload_fn = unload_fn
        self._ttl_s = resolve_llm_idle_ttl_s() if ttl_s is None else float(ttl_s)
        self._now = now_fn
        self._sleep = sleep_fn
        self._label = label
        self._started = False
        # model name -> (last seen expires_at ("mru"), when the use was stamped)
        self._seen: dict[str, tuple[float, float]] = {}

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def reap_idle_once(self) -> list[str]:
        """Unload the box's LLM-server model(s) iff loaded, no GPU job runs, and no
        model has seen a server-side use for ≥ TTL. Returns the unloaded ids ([] if
        it did nothing).

        The cheap LOCAL checks (ttl / active) run FIRST so the ollama query only
        fires on a quiet box; the query itself is one ``/api/ps`` per tick (the
        heartbeat probe already polls it far more often). Never raises."""
        try:
            if self._ttl_s <= 0:
                return []
            if self._active_fn() > 0:
                return []
            infos = list(self._loaded_models_info_fn() or [])
            now = self._now()
            names = {str(m.get("name")) for m in infos if m.get("name")}
            if not names:
                self._seen.clear()
                return []
            # Observe use: a new model, or an expires_at that ADVANCED since the
            # last tick, means the server was hit — stamp "used now".
            for m in infos:
                name = str(m.get("name") or "")
                if not name:
                    continue
                mru = float(m.get("mru") or 0.0)
                prev = self._seen.get(name)
                if prev is None or mru > prev[0]:
                    self._seen[name] = (mru, now)
            self._seen = {n: v for n, v in self._seen.items() if n in names}
            idle_s = now - max(stamped for (_, stamped) in self._seen.values())
            if not should_unload_external_model(True, 0, idle_s, self._ttl_s):
                return []
            models = sorted(names)
            log.info(
                "[llm-idle] %s: no observed use for %.0fs ≥ TTL %.0fs — unloading "
                "%s to free RAM",
                self._label, idle_s, self._ttl_s, ", ".join(models),
            )
            self._unload_fn(models)
            self._seen.clear()
            return models
        except Exception:
            log.exception("[llm-idle] %s reaper tick failed (ignored)", self._label)
            return []

    def ensure_started(self) -> None:
        """Arm the poller once. No-op when disabled (ttl<=0) or under the env
        kill-switch."""
        if self._started or self._ttl_s <= 0 or env_get(_DISABLE_ENV):
            return
        self._started = True
        threading.Thread(
            target=self._loop, name=f"llm-idle-reaper-{self._label}", daemon=True,
        ).start()
        log.info("[llm-idle] %s idle-unload reaper armed (TTL=%.0fs)", self._label, self._ttl_s)

    def _loop(self) -> None:
        poll = max(_POLL_FLOOR_S, min(_POLL_CEIL_S, self._ttl_s / 5.0))
        while True:
            self._sleep(poll)
            self.reap_idle_once()   # already swallows its own exceptions


# ── vLLM: the same idea, a different server-side signal ──────────────────────


def should_stop_idle_vllm(
    *, served: bool, running: int, active: int, idle_s: float, ttl_s: float,
) -> bool:
    """Pure decision: stop a vLLM sidecar only when it IS serving, has nothing in
    flight AT THE SERVER (``running <= 0``), no GPU job is claimed on this worker
    (``active <= 0``), and it has gone UNUSED for ``ttl_s``. ``ttl_s <= 0`` disables.

    Deliberately the same shape as :func:`should_unload_external_model` — the two
    idle paths must read identically or they will drift apart."""
    if ttl_s <= 0:
        return False
    return bool(served) and running <= 0 and active <= 0 and idle_s >= ttl_s


def should_start_vllm(*, served: bool, pending: int, min_queue: int) -> bool:
    """Pure decision: start a stopped vLLM when enough work is waiting to amortise
    the load. The mirror of :func:`should_stop_idle_vllm`, sharing its threshold with
    the dispatch-time amortisation gate so "is a cold start worth it?" has ONE answer.

    Never starts one that is already serving. ``min_queue <= 0`` means "start on any
    demand at all" — but never on zero, or an idle fleet would hold the VRAM forever
    for work that does not exist."""
    if served:
        return False
    return int(pending) > 0 and int(pending) >= int(min_queue)


class VLLMIdleStopper:
    """Daemon that stops an idle vLLM sidecar, freeing its whole VRAM footprint.

    WHY NOT ``LLMSupervisor``. That one exists and does the same job, but its idle
    clock is the BACKEND's in-process counter (``mark_request_start`` /
    ``mark_request_end``). It only ticks when the caller routes through the backend
    object — and on this fleet the nodes are handed a URL and POST it themselves, as
    do host-internal callers and other projects sharing the box. That clock therefore
    reads "idle" while the server is flat out, and arming it would SIGTERM vLLM
    mid-generation.

    So this watches the SERVER, exactly as the ollama reaper above does (and for the
    same reason, learned live — see the module docstring). vLLM's
    Prometheus endpoint gives two signals: ``vllm:num_requests_running`` (in flight
    right now) and ``vllm:request_success_total`` (monotonic completions). Requests
    running, or a completion count that ADVANCED since the previous tick, means used
    — by ANY client, which is the property the in-process counter cannot have.

    ``activity_fn`` returns ``{"served": bool, "running": int, "finished": int}``.
    First sighting stamps "used now": a server we have never observed was at worst
    just started, so it always gets a full TTL of grace. All I/O injected; best-effort
    throughout — a tick that cannot reach vLLM stops nothing and never raises."""

    def __init__(
        self,
        *,
        active_fn: Callable[[], int],
        activity_fn: Callable[[], dict],
        stop_fn: Callable[[], object],
        start_fn: Callable[[], object] | None = None,
        pending_fn: Callable[[], int] | None = None,
        min_queue: int | Callable[[], int] = 0,
        restart_cooldown_s: float = 120.0,
        ttl_s: float | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        label: str = "vllm",
        may_start_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._active_fn = active_fn
        self._activity_fn = activity_fn
        self._stop_fn = stop_fn
        self._start_fn = start_fn
        self._pending_fn = pending_fn
        #: PARK RELEASES THE CARD (design rule): a worker whose
        #: control state is OFF must not ignite a multi-minute server start the
        #: parked box can never use — the demand it would be answering is some
        #: OTHER box's business. ``None`` (every pre-rule caller) = always may.
        self._may_start_fn = may_start_fn
        #: How many waiting jobs justify a cold start — a NUMBER or a reader of one.
        #: A reader, for the deployments where it is operator policy: the topology
        #: file's ``vllm_min_queue`` is documented as taking effect without a restart,
        #: and it did on the claim side while this side had frozen the value at arm
        #: time. A box armed under an old hold of 1000000 then ignored a queue the
        #: operator had explicitly opened the gate for, silently, indefinitely.
        self._min_queue_fn: Callable[[], int] = (
            min_queue if callable(min_queue) else (lambda value=int(min_queue): value))
        self._restart_cooldown_s = float(restart_cooldown_s)
        self._last_action_at: float | None = None
        #: Consecutive starts that never produced a serving server — each doubles the
        #: cooldown (capped at 2^5 = 32x). Without it a permanently-broken target was
        #: re-requested every cooldown, taking the box out of service most of the
        #: time. Reset the moment the server is actually observed
        #: serving.
        self._failed_starts = 0
        self._ttl_s = resolve_llm_idle_ttl_s() if ttl_s is None else float(ttl_s)
        self._now = now_fn
        self._sleep = sleep_fn
        self._label = label
        self._started = False
        # (last finished count, when we last saw it advance)
        self._seen: tuple[int, float] | None = None

    def stop_idle_once(self) -> bool:
        """One tick. Returns True iff it stopped the server."""
        try:
            act = self._activity_fn() or {}
            served = bool(act.get("served"))
            running = int(act.get("running") or 0)
            finished = int(act.get("finished") or 0)
        except Exception:
            log.debug("[llm-idle] %s: vLLM activity probe failed", self._label,
                      exc_info=True)
            return False

        now = self._now()
        if not served:
            # Not serving — nothing to stop, and forget the history so a restarted
            # server gets a fresh full TTL rather than inheriting a stale clock.
            self._seen = None
            return False

        prev = self._seen
        if prev is None or finished > prev[0] or running > 0:
            # First sighting, an observed completion, or work in flight — all mean
            # "used now". A serving server also proves the last start WORKED, so the
            # failure backoff resets.
            self._failed_starts = 0
            self._seen = (finished, now)
            return False

        idle_s = now - prev[1]
        try:
            active = int(self._active_fn() or 0)
        except Exception:
            active = 0
        if not should_stop_idle_vllm(
            served=served, running=running, active=active,
            idle_s=idle_s, ttl_s=self._ttl_s,
        ):
            return False
        # STICKY HYSTERESIS (asymmetric on purpose — a design rule).
        # The START side refuses to pay a cold start for fewer than min_queue jobs;
        # the STOP side refuses while ANY job is waiting — zero, not min_queue.
        # Stopping under a 1..min_queue-1 backlog handed the very next job to
        # ollama, which reloaded the same weights this box just held warm, and the
        # re-deepening queue then paid the vLLM cold start AGAIN (many minutes on an
        # NFS-mounted load). A warm server with work behind it is
        # never torn down; it stops only once the queue it serves is EMPTY.
        # Deliberately independent of min_queue, so min_queue=0 ("always pay the
        # start") cannot also unstick the stop.
        if self._pending_fn is not None:
            try:
                raw = self._pending_fn()
                if raw is None or raw == "":
                    # UNREADABLE IS NOT ZERO. A probe that raises
                    # already keeps the server (below); a probe that returns None
                    # knows exactly as little, and used to coerce to 0 and STOP.
                    # Same ignorance must take the same branch, and the branch is
                    # the cheap one.
                    raise ValueError("pending probe returned no count")
                if int(raw) > 0:
                    log.debug(
                        "[llm-idle] %s: idle past TTL but %d job(s) waiting — "
                        "keeping the warm server", self._label, int(raw))
                    return False
            except Exception:
                # FAIL SAFE, not open: stopping is the EXPENSIVE action
                # (a long reload, e.g. over NFS); keeping a warm server costs
                # RAM for one more tick. An unreadable queue must never trigger the
                # expensive action — same direction as the start side.
                log.debug("[llm-idle] %s: pending count unreadable — KEEPING the "
                          "warm server", self._label, exc_info=True)
                return False
        try:
            self._stop_fn()
        except Exception:
            log.exception("[llm-idle] %s: vLLM stop failed (will retry next tick)",
                          self._label)
            return False
        log.info(
            "[llm-idle] %s: vLLM unused for %.0fs (TTL=%.0fs) — stopped, VRAM freed",
            self._label, idle_s, self._ttl_s,
        )
        self._seen = None
        self._last_action_at = now
        return True

    def start_if_wanted_once(self) -> bool:
        """One tick of the START side. Returns True iff it asked for a start.

        Guarded by a COOLDOWN so the pair cannot oscillate: a box that just idled out
        must not be restarted by the very next tick (the queue behind it did not
        change — only our own action did), and a vLLM takes minutes to boot, during
        which every tick still reads ``served=False`` and would otherwise fire another
        start.

        Best-effort: no ``start_fn`` wired, no ``pending_fn``, or a probe that raises
        all mean "do nothing". An unreadable queue must never trigger a
        multi-minute server start."""
        if self._start_fn is None or self._pending_fn is None:
            return False
        if self._may_start_fn is not None:
            try:
                if not self._may_start_fn():
                    log.debug(
                        "[llm-idle] %s: start gate closed (worker OFF/parked) — "
                        "not starting", self._label)
                    return False
            except Exception:
                # FAIL OPEN: the gate is a control-plane read; a DB blip must
                # degrade to the pre-gate behaviour (start on demand), not wedge
                # every start on the fleet behind an unreadable table.
                log.debug("[llm-idle] %s: start gate unreadable — treating as "
                          "open", self._label, exc_info=True)
        now = self._now()
        cooldown = self._restart_cooldown_s * (2 ** min(self._failed_starts, 5))
        if (self._last_action_at is not None
                and now - self._last_action_at < cooldown):
            return False
        try:
            act = self._activity_fn() or {}
            served = bool(act.get("served"))
        except Exception:
            return False
        if served:
            return False
        try:
            pending = int(self._pending_fn() or 0)
        except Exception:
            log.debug("[llm-idle] %s: pending probe failed; not starting",
                      self._label, exc_info=True)
            return False
        try:
            min_queue = int(self._min_queue_fn())
        except Exception:
            # Same direction as every other unknown on this side: a start is minutes
            # of weight loading, so ignorance must not trigger it.
            log.debug("[llm-idle] %s: start threshold unreadable; not starting",
                      self._label, exc_info=True)
            return False
        if not should_start_vllm(
            served=served, pending=pending, min_queue=min_queue,
        ):
            return False
        try:
            self._start_fn()
        except Exception:
            log.exception("[llm-idle] %s: vLLM start failed", self._label)
            return False
        log.info(
            "[llm-idle] %s: %d job(s) waiting (>= %d) — starting vLLM",
            self._label, pending, min_queue,
        )
        self._last_action_at = now
        self._failed_starts += 1      # provisional; reset when it is seen serving
        return True

    def tick(self) -> None:
        """One full lifecycle tick: stop an idle server, or start a stopped one the
        queue is waiting on. Never both — a stop returns before the start is
        considered."""
        if self.stop_idle_once():
            return
        self.start_if_wanted_once()

    def ensure_started(self) -> None:
        if self._started or self._ttl_s <= 0 or env_get(_DISABLE_ENV):
            return
        self._started = True
        threading.Thread(
            target=self._loop, name=f"vllm-idle-stopper-{self._label}", daemon=True,
        ).start()
        log.info("[llm-idle] %s vLLM idle-stopper armed (TTL=%.0fs)",
                 self._label, self._ttl_s)

    def _loop(self) -> None:
        poll = max(_POLL_FLOOR_S, min(_POLL_CEIL_S, self._ttl_s / 5.0))
        while True:
            self._sleep(poll)
            self.tick()


# ── THE PARK WATCHDOG (the kill rule) ────────────────────────────


class ParkQuiesce:
    """Tick-driven KILL of a parked box's broker-managed LLM servers.

    The rule: a parked box runs no engine-managed LLM server.
    Unconditional — no idle TTL, no activity exemption.

    History, because the softer versions both failed live:

    * No quiesce at all: a parked box's vllm sidecar
      crash-looped (``restart: unless-stopped`` resurrects every clean
      self-exit) and re-loaded a large model hours later with nothing able to stop it —
      the only stop mechanism (the idle stopper) lives in the gpu claim
      worker, which was parked.
    * Idle-reaper composition, activity-guarded for peers: defeated
      by REMOTE traffic — a peer box POSTing chat completions over
      the fast iface kept a parked box's vllm "busy" at high GPU (several
      concurrent requests), so the busy-exemption meant anyone outside the box could keep
      it un-parked forever. An out-of-band ``docker run`` had also recreated
      the container mid-park (fresh CreatedAt, RestartCount reset — the
      engine's start lever can only START an existing container).

    Parked now means: NO LLM service on this box, full stop. A serving vLLM is
    stopped ON SIGHT — mid-request included; the caller gets a failed call and
    connection-refused thereafter, which is exactly the pressure that fails it
    over to an ON box. Every loaded ollama MODEL is unloaded on sight (the
    daemon itself survives — shared box infra with ~no resident cost). Every
    resurrection, engine-driven or out-of-band, is re-killed on the next tick.
    The vllm stop lever is a real ``docker stop``, which ``unless-stopped``
    respects, so a kill sticks until something outside re-creates or re-starts
    the container — and then it is killed again.

    Each half is independently best-effort and swallows its own errors, so
    :meth:`tick` never raises — it lives on the park gate's wait loop.

    THE ONE EXEMPTION — our OWN in-flight work. ``stop_policy='drain'``
    (what the panel's park lever writes) means "finish what you are holding, claim
    nothing new", so a parked worker legitimately keeps running the job it already had.
    The kill rule then told ollama to unload the very model that job was streaming from —
    a ``keep_alive:0`` post every tick, one identical log line each time.

    What that actually cost: ollama does NOT refuse the
    unload. It applies the zero keep-alive at the REQUEST BOUNDARY — the runner exits
    when the in-flight call ends (``Stopping...`` means "expiry scheduled, runner
    draining", not "refused") — and the job's NEXT call then COLD-RELOADS the whole
    model. The model-reload rate after a park rose several-fold over the rate before
    it. The park was not
    releasing the card, it was making the kept job thrash it.

    The kill rule is about OTHER people's traffic — an exemption a peer could hold open for
    ever — and that stays dead. A job THIS worker is running is the opposite: the park
    policy itself chose to preserve it, and it ends. So the watchdog holds off while
    ``in_flight_fn`` reports work and kills on sight the instant it does not — which is
    immediately under ``hard`` park, since that re-queues the job first. A caller that
    passes no seam gets the kill rule verbatim, and an unreadable probe falls back to killing.

    The seam must be scoped to the lane that uses the LLM (the pool lane), NOT to every
    GPU job: an inline diffusion render does not touch the LLM, and exempting on one
    would let a peer's traffic keep a parked box's vLLM busy — the activity-guard failure,
    through a side door.

    BY CONTAINER STATE. ``vllm_activity_fn`` reads ``/metrics``, and a vLLM
    still LOADING its weights answers nothing: a lane parked in the middle of a vLLM
    start goes on loading the whole model, invisible to a probe-only watchdog,
    until the server binds its port. ``vllm_running_fn`` — the container's own
    state, where the caller can read it — closes that: a container that runs but does not
    answer yet is stopped too. An unreadable container state falls back to the probe alone,
    exactly as before.

    THE SECOND VOICE — a CLUSTER RESERVATION's hold (``voice='cluster-hold'``). A lane that
    holds its box for a server the engine does not run keeps the broker's own servers down
    for the reason a park does. The reservation's OWN body uses no LLM, but the pool lane
    can have claimed an llm_chat moments before the hold began, and that job streams from
    this very card — so the same exemption applies, with a much shorter ceiling
    (``claim_worker.CLUSTER_HOLD_DRAIN_MAX_S``): the in-flight job drains, and then the
    cluster gets its box whether or not the job ended. The caller counts the reservation's
    own pool slot OUT of ``in_flight_fn``, or the hold would wait for itself.
    """

    #: How each caller's lines read: ``(log tag, the box's state, why it serves nothing)``.
    _VOICES = {
        "park": ("park-watchdog", "PARKED", "a parked box serves nobody"),
        "cluster-hold": ("cluster-hold", "HELD for a cluster reservation",
                         "the external cluster needs the whole box"),
    }

    #: How often a repeated, unchanged verdict is restated at INFO. The watchdog ticks
    #: on the park loop (~5 s); narrating every tick is what buried the real lines.
    DEFAULT_REPORT_EVERY_S = 60.0

    #: The CEILING on the exemption above, and it is load-bearing. The exemption is
    #: only as bounded as the job is — and a pool-lane job has no wall-clock watchdog
    #: (``_run_pool_node`` arms none) while its lease renewer keeps renewing for as
    #: long as the process lives, so lease-reclaim cannot fire either. One wedged HTTP
    #: call would therefore pin the counter at 1 and hold the card SILENTLY for ever,
    #: with the operator looking at PARKED. After this much CONTINUOUS holding off,
    #: the kill rule wins and says why. The default is the GPU job budget: we wait no longer
    #: than a GPU job may legitimately run.
    DEFAULT_HOLD_OFF_MAX_S = 8100.0

    def __init__(
        self,
        *,
        vllm_activity_fn: Callable[[], dict],
        vllm_stop_fn: Callable[[], object],
        ollama_models_fn: Callable[[], list[dict]],
        ollama_unload_fn: Callable[[list[str]], object],
        in_flight_fn: Callable[[], int] | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        report_every_s: float = DEFAULT_REPORT_EVERY_S,
        hold_off_max_s: float = DEFAULT_HOLD_OFF_MAX_S,
        label: str = "?",
        vllm_running_fn: Callable[[], bool] | None = None,
        voice: str = "park",
    ) -> None:
        self._vllm_activity_fn = vllm_activity_fn
        self._vllm_stop_fn = vllm_stop_fn
        #: The vLLM CONTAINER's own state (running = serving OR still loading); ``None`` =
        #: the caller cannot read it, and the watchdog decides on ``/metrics`` alone.
        self._vllm_running_fn = vllm_running_fn
        self._tag, self._state, self._reason = self._VOICES.get(voice, self._VOICES["park"])
        self._ollama_models_fn = ollama_models_fn
        self._ollama_unload_fn = ollama_unload_fn
        self._in_flight_fn = in_flight_fn
        self._now = now_fn
        self._report_every_s = float(report_every_s)
        self._hold_off_max_s = float(hold_off_max_s)
        self._label = label
        self._said: dict[str, tuple[str, float]] = {}
        #: When the CURRENT continuous stretch of holding off began. Reset the moment
        #: the box drains, so a box that drains and is parked again gets a fresh grace
        #: — the ceiling is about one stuck job, not a lifetime budget.
        self._holding_since: float | None = None

    def _say(self, key: str, detail: str, msg: str, *args,
             level: int = logging.INFO) -> None:
        """Log at ``level`` when the verdict CHANGES or the report window has elapsed,
        at DEBUG otherwise. A change is never rate-limited away."""
        now = self._now()
        prev = self._said.get(key)
        if prev is not None and prev[0] == detail and now - prev[1] < self._report_every_s:
            log.debug(msg, *args)
            return
        self._said[key] = (detail, now)
        log.log(level, msg, *args)

    def _own_work_in_flight(self) -> int:
        if self._in_flight_fn is None:
            return 0
        try:
            return max(0, int(self._in_flight_fn() or 0))
        except Exception:
            # Fail TOWARD the kill rule: if we cannot tell whether the work is ours, a
            # parked box still serves nobody.
            log.debug("[park-watchdog] %s: in-flight probe failed — assuming none",
                      self._label, exc_info=True)
            return 0

    def tick(self) -> tuple[bool, list[str]]:
        """One watchdog pass: ``(vllm_stopped, ollama_models_unloaded)``."""
        in_flight = self._own_work_in_flight()
        now = self._now()
        if in_flight > 0:
            if self._holding_since is None:
                self._holding_since = now
            held_for = now - self._holding_since
            if held_for < self._hold_off_max_s:
                self._say(
                    "draining", str(in_flight),
                    "[%s] %s: %s and draining — %d job(s) still in "
                    "flight on this worker; holding off the LLM eviction until they "
                    "finish (force-kill the job to release the card now)",
                    self._tag, self._label, self._state, in_flight,
                )
                return False, []
            self._say(
                "holdoff-expired", str(in_flight),
                "[%s] %s: %d job(s) have held the card for %.0fs since this box was %s "
                "(ceiling %.0fs) — releasing it anyway; %s",
                self._tag, self._label, in_flight, held_for, self._state,
                self._hold_off_max_s, self._reason,
                level=logging.WARNING,
            )
        else:
            self._holding_since = None
        stopped = False
        served, requests = False, "?"
        try:
            act = self._vllm_activity_fn() or {}
            served = bool(act.get("served"))
            requests = act.get("running", "?")
        except Exception:
            log.debug("[%s] %s: vLLM activity probe failed", self._tag, self._label,
                      exc_info=True)
        loading = False
        if not served and self._vllm_running_fn is not None:
            try:
                loading = bool(self._vllm_running_fn())
            except Exception:
                # No readable container state: exactly as good as the probe alone, never
                # worse.
                log.debug("[%s] %s: vLLM container state unreadable (probe only)",
                          self._tag, self._label, exc_info=True)
        if served or loading:
            try:
                # KILL FIRST, narrate after: the log line's value
                # formatting must never be able to skip the stop.
                self._vllm_stop_fn()
                stopped = True
                if served:
                    self._say(
                        "vllm", str(requests),
                        "[%s] %s: box is %s but vLLM was serving "
                        "(%s request(s) in flight) — stopped it NOW; %s",
                        self._tag, self._label, self._state, requests, self._reason,
                    )
                else:
                    self._say(
                        "vllm-loading", "loading",
                        "[%s] %s: box is %s but its vLLM container was running and not "
                        "answering yet (still loading) — stopped it NOW, by container "
                        "state; %s", self._tag, self._label, self._state, self._reason,
                    )
            except Exception:
                log.debug("[%s] %s: vllm half failed (retried next tick)", self._tag,
                          self._label, exc_info=True)
        unloaded: list[str] = []
        try:
            infos = self._ollama_models_fn() or []
            names = sorted({str(m.get("name")) for m in infos if m.get("name")})
            if names:
                self._say(
                    "ollama", ",".join(names),
                    "[%s] %s: box is %s but ollama holds %s — "
                    "unloading NOW; %s",
                    self._tag, self._label, self._state, ", ".join(names), self._reason,
                )
                self._ollama_unload_fn(names)
                unloaded = names
        except Exception:
            log.debug("[%s] %s: ollama half failed (retried next tick)", self._tag,
                      self._label, exc_info=True)
        return stopped, unloaded


__all__ = [
    "should_unload_external_model",
    "ExternalModelIdleReaper",
    "should_stop_idle_vllm",
    "should_start_vllm",
    "VLLMIdleStopper",
    "ParkQuiesce",
    "DEFAULT_LLM_IDLE_TTL_S",
    "resolve_llm_idle_ttl_s",
]
