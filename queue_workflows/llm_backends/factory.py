"""The per-``(host_label, queue)`` LLM-backend FACTORY — the live owner of the
concrete :class:`LLMBackend` singleton AND its :class:`LLMSupervisor`, kept in
sync with the operator's DB config.

WHY THIS EXISTS. The backend modules (``base`` / ``ollama`` / ``vllm``) and the
idle :class:`~queue_workflows.llm_backends.supervisor.LLMSupervisor` are
deliberately DB-free, I/O-injected, unit-testable parts. SOMETHING has to read a
machine's ``worker_controls`` LLM config (migration 0013, via
:func:`queue_workflows.worker_control.llm_config_for`), pick the matching concrete
backend, hand it the right base URL, and — for vllm — arm a supervisor to free its
VRAM after an idle window. That is THIS factory. The (later-built) VLM node calls
:func:`get_backend(host, queue)` at request time and brackets its HTTP POST on the
returned backend; the supervisor reads that SAME instance off-thread to decide when
to SIGTERM the sidecar. So the factory's contract is two-fold:

  * **Identity preservation** — within one ``(host, queue)``, hand back the SAME
    backend instance across calls as long as the config SNAPSHOT
    ``(server_type, parallelism, vllm_idle_ttl_s)`` is unchanged. The backend's
    request counters (``inflight`` / ``idle_seconds``) and the vllm state machine
    (``_served_model`` / the sticky Sleep-L2 flag) live ON that instance; rebuild it
    spuriously and you reset all of that, so the supervisor and the request bracket
    would lose their read model. Only an ACTUAL config change rebuilds.
  * **Freshness** — when the operator flips a machine ollama↔vllm or retunes it, the
    next request must observe the change. A 0013 NOTIFY on
    :data:`~queue_workflows.worker_control.LLM_CONFIG_NOTIFY_CHANNEL` invalidates the
    cache instantly (the LISTEN invalidator thread below); a TTL re-read is the
    dropped-NOTIFY fallback.

DESIGN — a process-singleton with injected side effects, the same idiom as
:mod:`queue_workflows.model_cache` / :mod:`queue_workflows.gpu_model_cache`:

  * ``build_backend_fn`` constructs the concrete backend — default builds a real
    :class:`~queue_workflows.llm_backends.ollama.OllamaBackend` /
    :class:`~queue_workflows.llm_backends.vllm.VLLMBackend` with their default I/O
    seams (httpx lazy, never called at construction — the factory does no health
    probe on build). A test injects a fake.
  * ``supervisor_factory`` builds the :class:`LLMSupervisor` for a vllm backend —
    default ``LLMSupervisor(backend=backend)``; ``.start()`` is idempotent and
    env-gated (``AI_LEADS_DISABLE_LLM_SUPERVISOR``), so it is INERT in tests.
  * the URL per server type is resolved from the env NAMES on
    :class:`~queue_workflows.config.EngineConfig` (``ollama_url_env`` /
    ``vllm_url_env``) — the DB owns WHICH server + tunables, the URL is deployment
    topology and stays in env. Constructor ``ollama_url`` / ``vllm_url`` kwargs
    override env (tests).

CONCURRENCY. ``get_backend`` runs on the node thread; the LISTEN invalidator calls
``invalidate`` from its own daemon thread; a rebuild calls ``shutdown`` / supervisor
``stop`` which may block briefly. The cache dict + per-key flags are guarded by a
single reentrant :class:`~threading.RLock` (reentrant so ``get_backend`` may call a
locked helper without self-deadlock). We hold the lock across the read-modify of a
cache entry; the DB read + backend build run UNDER the lock too — they are cheap
(one SELECT, no I/O on construction), and holding the lock keeps two racing
``get_backend`` calls from each building a backend for the same key.

The daemon-thread plumbing (``start`` / ``stop`` / ``threading.Event`` /
autocommit-connect LISTEN loop / ``_enabled`` env gate) MIRRORS
:class:`queue_workflows.worker_control.WorkerControlWatcher`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any, Callable

from queue_workflows import worker_control
from queue_workflows.config import get_config
from queue_workflows.db import db_url
from queue_workflows.envcompat import env_get
from queue_workflows.llm_backends.base import LLMBackend
from queue_workflows.llm_backends.ollama import OllamaBackend
from queue_workflows.llm_backends.supervisor import LLMSupervisor
from queue_workflows.llm_backends.vllm import VLLMBackend

log = logging.getLogger(__name__)


# ── env defaults + the listener kill-switch ─────────────────────────────────

#: Fallback base URLs when the env NAME on the config (ollama_url_env /
#: vllm_url_env) is unset — the localhost/docker-host defaults documented on
#: ``EngineConfig``.
DEFAULT_OLLAMA_URL = "http://host.docker.internal:11434"
DEFAULT_VLLM_URL = "http://host.docker.internal:8000"

#: Env kill-switch for the LISTEN invalidator thread. DISTINCT from the
#: supervisor's ``AI_LEADS_DISABLE_LLM_SUPERVISOR`` — a host may want the idle
#: supervisor inert (e.g. a test) while still keeping config-change LISTEN, or
#: vice-versa, so the two gates are independent. Added to the engine conftest's
#: ``setdefault`` block so tests don't spawn the thread.
DISABLE_LISTENER_ENV = "AI_LEADS_DISABLE_LLM_CONFIG_LISTENER"

#: Default TTL (s) between DB re-reads of a key's config when no NOTIFY fired —
#: the dropped-NOTIFY safety net behind the instant LISTEN invalidation.
DEFAULT_CONFIG_TTL_S = 10.0


@dataclass(frozen=True)
class _Snapshot:
    """The config fields that DECIDE the backend's identity. Two reads with the
    same snapshot ⇒ the SAME backend is kept (no rebuild). The base URL is NOT in
    the snapshot: it is env-derived deployment topology, fixed for a process'
    lifetime, so a change to it isn't a runtime reconfiguration the factory tracks
    (a redeploy restarts the process)."""

    server_type: str
    parallelism: int
    vllm_idle_ttl_s: int

    @classmethod
    def from_config(
        cls, cfg: worker_control.LLMConfig, *, server_type: str | None = None,
    ) -> "_Snapshot":
        """``server_type`` overrides the row's kind for a per-job resolution
        (migration 0022); ``None`` keeps the box default."""
        return cls(
            server_type=server_type or cfg.server_type,
            parallelism=int(cfg.parallelism),
            vllm_idle_ttl_s=int(cfg.vllm_idle_ttl_s),
        )


@dataclass
class _Entry:
    """One cached ``(host, queue)`` slot: the live snapshot it was built for, the
    backend instance whose identity we preserve, its supervisor (vllm only; ``None``
    for ollama), the monotonic time of the last DB check (for the TTL throttle), and
    a force-rebuild flag the NOTIFY invalidator sets."""

    snapshot: _Snapshot
    backend: LLMBackend
    supervisor: Any  # LLMSupervisor | None
    last_check: float
    invalid: bool = False


def _default_build_backend(
    server_type: str, base_url: str, parallelism: int, idle_ttl_s: float,
) -> LLMBackend:
    """Build the real concrete backend for ``server_type`` with its DEFAULT I/O
    seams. ``VLLMBackend``'s httpx-backed health/served seams are lazy (imported
    inside their bodies and only CALLED on ``ensure_ready`` / ``is_running``), and
    the factory never calls those at construction — so building one here does no
    network I/O and keeps the host-import guard green."""
    if server_type == worker_control.SERVER_TYPE_VLLM:
        # Thread the host's vllm lifecycle hooks (ai_leads wires docker-over-UDS
        # stop/start so the in-worker supervisor can control the SIBLING sidecar
        # container). Each is None unless a host wired it via set_vllm_lifecycle;
        # the backend treats None as "use my built-in default" (``kill_fn or
        # _default_kill_fn``), so an unconfigured deployment is unchanged.
        cfg = get_config()
        return VLLMBackend(
            base_url=base_url, parallelism=parallelism, idle_ttl_s=idle_ttl_s,
            kill_fn=cfg.vllm_stop_fn,
            ensure_up_fn=cfg.vllm_start_fn,
        )
    # Anything else resolves to ollama (the default-safe server type). ollama's
    # idle is owned by its own daemon's KEEP_ALIVE, so idle_ttl_s is forced to 0
    # by the caller and the supervisor stays inert regardless.
    return OllamaBackend(
        base_url=base_url, parallelism=parallelism, idle_ttl_s=idle_ttl_s,
    )


class BackendFactory:
    """The per-``(host, queue)`` owner of the live :class:`LLMBackend` + its
    :class:`LLMSupervisor`, kept in sync with the DB config (migration 0013).

    See the module docstring for the identity-preservation + freshness contract.
    All side effects are injected so a test drives fakes with no network / no DB
    threads:

      * ``build_backend_fn(server_type, base_url, parallelism, idle_ttl_s) ->
        LLMBackend`` — construct the concrete backend (default: real ollama/vllm).
      * ``supervisor_factory(backend) -> LLMSupervisor`` — build the idle supervisor
        for a vllm backend (default: ``LLMSupervisor(backend=backend)``).
      * ``now_fn() -> float`` — the TTL clock (default ``time.monotonic``); a test
        feeds a mutable cell to cross the TTL deterministically.
      * ``ollama_url`` / ``vllm_url`` — override the env URL resolution (tests).
    """

    def __init__(
        self,
        *,
        ttl_s: float = DEFAULT_CONFIG_TTL_S,
        now_fn: Callable[[], float] = time.monotonic,
        build_backend_fn: Callable[..., LLMBackend] | None = None,
        supervisor_factory: Callable[[Any], Any] | None = None,
        ollama_url: str | None = None,
        vllm_url: str | None = None,
    ) -> None:
        self._ttl_s = float(ttl_s)
        self._now_fn = now_fn
        self._build_backend = build_backend_fn or _default_build_backend
        self._supervisor_factory = supervisor_factory or (
            lambda backend: LLMSupervisor(backend=backend)
        )
        # None ⇒ resolve from env at use-time; a non-None override is used verbatim
        # (after the same normalization the env path gets).
        self._ollama_url_override = ollama_url
        self._vllm_url_override = vllm_url

        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str], _Entry] = {}

        # LISTEN invalidator plumbing (mirrors WorkerControlWatcher).
        self._listener_stop = threading.Event()
        self._listener_thread: threading.Thread | None = None

    # ── URL resolution ───────────────────────────────────────────────────────

    @staticmethod
    def _topology_url(server_type: str | None = None) -> str | None:
        """The per-box LLM server ROOT URL from the deployment topology YAML
        (:mod:`queue_workflows.llm_backends.topology`), keyed by THIS box's
        ``host_label`` — the value it exports as the engine's host-label env. Returns
        ``None`` when no ``llm_topology_path`` is configured, no file/entry matches,
        or the label env is unset — so the caller falls back to the env + localhost
        default (the byte-compatible path for a consumer that never sets a topology).

        Server-type-agnostic: the topology supplies the ADDRESS a box dispatches to;
        the DB (worker_controls) still owns WHICH server type runs there. So the same
        entry feeds both the ollama and vllm resolvers below."""
        cfg = get_config()
        path = cfg.llm_topology_path
        if not path:
            return None
        host_label = env_get(cfg.host_label_env, "").strip()
        if not host_label:
            return None
        from queue_workflows.llm_backends import topology

        if server_type:
            # Per-kind address (0022): two servers can be co-resident on one box and
            # they answer on different ports. Falls back INSIDE resolve_for to the
            # entry's shared ``url``, so every pre-0022 topology is unchanged.
            return topology.resolve_for(path, host_label, server_type) or None
        entry = topology.resolve(path, host_label)
        return (entry or {}).get("url") or None

    def _ollama_base_url(self) -> str:
        if self._ollama_url_override is not None:
            return self._normalize_ollama(self._ollama_url_override)
        topo = self._topology_url(worker_control.SERVER_TYPE_OLLAMA)
        if topo:
            return self._normalize_ollama(topo)
        raw = env_get(get_config().ollama_url_env, "").strip()
        return self._normalize_ollama(raw or DEFAULT_OLLAMA_URL)

    def _vllm_base_url(self) -> str:
        if self._vllm_url_override is not None:
            return self._normalize_vllm(self._vllm_url_override)
        topo = self._topology_url(worker_control.SERVER_TYPE_VLLM)
        if topo:
            return self._normalize_vllm(topo)
        raw = env_get(get_config().vllm_url_env, "").strip()
        return self._normalize_vllm(raw or DEFAULT_VLLM_URL)

    @staticmethod
    def _normalize_ollama(url: str) -> str:
        """Ollama's chat endpoint is built off the ROOT (``/api/chat``), so we only
        strip a trailing slash; nothing else to peel."""
        return url.rstrip("/")

    @staticmethod
    def _normalize_vllm(url: str) -> str:
        """Normalize the vllm URL to the server ROOT. ``VLLMBackend`` appends
        ``/v1/chat/completions`` itself, but the deployed ai_leads env value is the
        OpenAI base (``http://host:8000/v1``) — so strip a trailing ``/v1`` (after
        any trailing slash) to avoid a doubled ``/v1/v1``. Idempotent for a value
        already at the root."""
        u = url.rstrip("/")
        if u.endswith("/v1"):
            u = u[: -len("/v1")]
        return u.rstrip("/")

    def _base_url_for(self, server_type: str) -> str:
        if server_type == worker_control.SERVER_TYPE_VLLM:
            return self._vllm_base_url()
        return self._ollama_base_url()

    def resolve_base_url(self, server_type: str = worker_control.SERVER_TYPE_OLLAMA) -> str:
        """The LLM server ROOT URL THIS box would dispatch to for ``server_type`` —
        the per-box topology entry (keyed by this box's host_label) if set, else the
        URL env, else the localhost/docker-host default. Public so a caller can PROBE
        the endpoint (queue_workflows.llm_probe) without building a backend or
        touching the DB — the heartbeat's observed-capability check uses it."""
        return self._base_url_for(server_type)

    # ── the read-through cache ─────────────────────────────────────────────────

    def get_backend(
        self, host_label: str, queue: str, server_type: str | None = None,
    ) -> LLMBackend:
        """Return the live backend for ``(host_label, queue, server_type)``,
        building/rebuilding it only when the DB config snapshot actually changed.

        ``server_type`` (migration 0022) selects the KIND for THIS request, so a box
        running both servers keeps both backends alive side by side — each with its
        own request counters and (for vllm) its own state machine. ``None`` keeps the
        pre-0022 behaviour: the kind comes from the box's ``worker_controls`` row,
        which is now its DEFAULT rather than a switch.

        The cache key includes the resolved kind precisely so asking for ollama does
        not tear down a live vllm backend (and vice versa) — under co-residency that
        would reset the other server's accounting mid-flight.

        Fast path: a cached entry whose ``last_check`` is within ``ttl_s`` and not
        flagged invalid is returned WITHOUT touching the DB — the TTL throttle that
        keeps a hot request loop from SELECTing every call. Otherwise re-read
        ``llm_config_for``, refresh ``last_check``, clear the invalid flag, and:
        same snapshot ⇒ return the cached instance unchanged (identity preserved);
        changed snapshot ⇒ rebuild (shutdown the old + stop its supervisor, build
        the new + arm a supervisor for vllm)."""
        with self._lock:
            # The DB read is cheap and default-safe; do it first so the cache key can
            # carry the RESOLVED kind (an explicit server_type overrides the row).
            cfg = worker_control.llm_config_for(host_label, queue)
            resolved = server_type or cfg.server_type
            key = (host_label, queue, resolved)
            entry = self._cache.get(key)
            now = self._now_fn()
            if (
                entry is not None
                and not entry.invalid
                and (now - entry.last_check) < self._ttl_s
            ):
                # Within the TTL and not invalidated — the throttle hit.
                return entry.backend

            snapshot = _Snapshot.from_config(cfg, server_type=resolved)

            if entry is not None and entry.snapshot == snapshot:
                # Re-read confirmed nothing changed — refresh the check clock + drop
                # the invalid flag, but KEEP the same instance (its counters / vllm
                # state survive). This is the identity-preservation core.
                entry.last_check = now
                entry.invalid = False
                return entry.backend

            # Snapshot changed (or first build for this key) — rebuild.
            if entry is not None:
                self._teardown_entry(entry)
            new_entry = self._build_entry(snapshot, now)
            self._cache[key] = new_entry
            return new_entry.backend

    def invalidate(self, host_label: str, queue: str) -> None:
        """Force the next :meth:`get_backend` for this key to re-read the DB,
        regardless of the TTL. Set by the LISTEN invalidator on a NOTIFY so an
        operator's config edit takes effect at once (the TTL is only the fallback).
        A no-op for a key we've never built — there's nothing cached to refresh,
        and the first ``get_backend`` will read the DB anyway.

        Invalidates EVERY server kind cached for this ``(host, queue)`` — one config
        edit can change the box default, the parallelism or the idle TTL, any of
        which applies to whichever kinds are live."""
        with self._lock:
            for key, entry in self._cache.items():
                if key[0] == host_label and key[1] == queue:
                    entry.invalid = True

    def _build_entry(self, snapshot: _Snapshot, now: float) -> _Entry:
        """Construct a backend (+ a vllm supervisor) for ``snapshot``. Caller holds
        the lock. idle_ttl mapping: ollama → 0 (its daemon self-manages idle, and
        the supervisor is inert for it anyway); vllm → the configured
        ``vllm_idle_ttl_s``. Both get the configured ``parallelism``."""
        base_url = self._base_url_for(snapshot.server_type)
        is_vllm = snapshot.server_type == worker_control.SERVER_TYPE_VLLM
        idle_ttl_s = float(snapshot.vllm_idle_ttl_s) if is_vllm else 0.0
        backend = self._build_backend(
            snapshot.server_type, base_url, snapshot.parallelism, idle_ttl_s,
        )
        # NO in-process idle supervisor — DELIBERATELY (found live, the "unstable
        # server" oscillation). LLMSupervisor's idle clock is the backend's
        # in-process request bracket, and on this fleet the nodes are handed a URL
        # and POST it themselves — so the clock NEVER ticks, reads "idle >= TTL"
        # every 300 s while the server is flat out, and killed the sidecar
        # MID-REQUEST every few minutes, all night (three severed chats per kill in
        # the log). vLLM idle lifecycle has exactly ONE owner now: the
        # server-observed, queue-aware VLLMIdleStopper the gpu worker arms
        # (llm_idle_unload) — it watches /metrics, refuses to stop while >= min_queue
        # jobs wait, and backs off on failed starts. Two reapers with different
        # clocks was the instability.
        supervisor = None
        return _Entry(
            snapshot=snapshot, backend=backend, supervisor=supervisor,
            last_check=now, invalid=False,
        )

    def _teardown_entry(self, entry: _Entry) -> None:
        """Release a replaced/dropped entry: stop its supervisor (if any), then
        ``shutdown`` the backend (frees a vllm sidecar's VRAM; a no-op for ollama).
        Best-effort — a flaky stop/shutdown must not block the rebuild or the
        process teardown. Caller holds the lock; supervisor.stop() joins its own
        daemon (bounded 2s), which doesn't re-enter the factory, so no deadlock."""
        if entry.supervisor is not None:
            try:
                entry.supervisor.stop()
            except Exception:
                log.exception("[BackendFactory] supervisor stop failed (ignored)")
        try:
            entry.backend.shutdown()
        except Exception:
            log.exception("[BackendFactory] backend shutdown failed (ignored)")

    # ── LISTEN invalidator (mirrors WorkerControlWatcher) ──────────────────────

    @property
    def _listener_enabled(self) -> bool:
        """The LISTEN invalidator runs unless its dedicated kill-switch is set
        (tests set it in conftest). DISTINCT from the supervisor's gate."""
        return not bool(env_get(DISABLE_LISTENER_ENV))

    def start(self) -> None:
        """Spawn the daemon that LISTENs
        :data:`~queue_workflows.worker_control.LLM_CONFIG_NOTIFY_CHANNEL` and, on each
        ``"host|queue"`` payload, invalidates that key so the next request re-reads.
        Idempotent (a second call while running is a no-op) and env-gated — a no-op
        when :attr:`_listener_enabled` is False, exactly like
        ``WorkerControlWatcher.start`` / ``LLMSupervisor.start``."""
        if not self._listener_enabled or self._listener_thread is not None:
            return
        self._listener_stop.clear()
        self._listener_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="llm-config-invalidator",
        )
        self._listener_thread.start()
        log.info("[BackendFactory] LLM-config invalidator armed")

    def _listen_loop(self) -> None:
        """Autocommit-connect, ``LISTEN`` the config channel, and invalidate the
        keyed entry on each NOTIFY. MIRRORS ``WorkerControlWatcher._loop``: a
        ``notifies(timeout=…, stop_after=1)`` wait so ``stop()`` is observed within
        one poll; the whole loop swallows + logs on crash (a dropped LISTEN must not
        take down the worker — the TTL re-read still keeps configs fresh).

        PG-reconnect: a PG bounce mid-watch used to crash the daemon (only the
        TTL re-read kept configs fresh from then on); the helper now reconnects
        so instant config invalidation survives a PG restart."""
        from queue_workflows.db import listen_with_reconnect

        def _body(listen_conn):
            while not self._listener_stop.is_set():
                for notify in listen_conn.notifies(
                    timeout=self._ttl_s, stop_after=1,
                ):
                    self._handle_notify(notify.payload)
                if self._listener_stop.is_set():
                    return
        try:
            listen_with_reconnect(
                worker_control.LLM_CONFIG_NOTIFY_CHANNEL,
                self._listener_stop, _body,
            )
        except Exception:
            log.exception("[BackendFactory] config-invalidator loop crashed")

    def _handle_notify(self, payload: str | None) -> None:
        """Parse a ``"host|queue"`` NOTIFY payload (the 0013 trigger's format) and
        invalidate that key. A malformed/empty payload is ignored — the TTL re-read
        is the backstop, so a stray NOTIFY can't crash the daemon."""
        if not payload or "|" not in payload:
            return
        host_label, _, queue = payload.partition("|")
        if host_label and queue:
            self.invalidate(host_label, queue)

    def stop(self) -> None:
        """Stop the invalidator daemon AND release every cached backend (shutdown +
        supervisor stop). Called on worker teardown. Idempotent + safe when never
        started; joins the daemon bounded (mirrors ``WorkerControlWatcher.stop``)."""
        self._listener_stop.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=2.0)
            self._listener_thread = None
        with self._lock:
            for entry in self._cache.values():
                self._teardown_entry(entry)
            self._cache.clear()


# ── module singleton + thin functions (model_cache idiom) ────────────────────

_DEFAULT = BackendFactory()


def get_backend(
    host_label: str, queue: str, server_type: str | None = None,
) -> LLMBackend:
    """Process-wide :func:`BackendFactory.get_backend` — the node's request-time
    entry point."""
    return _DEFAULT.get_backend(host_label, queue, server_type)


#: Server kinds whose START is expensive enough to need amortising. vLLM reloads
#: its weights on every start (~7 min for a 35 GB model on unified-memory GPU box); ollama swaps a
#: model in seconds and is never gated.
COLD_START_KINDS = frozenset({worker_control.SERVER_TYPE_VLLM})

#: Default queue depth that justifies paying a cold start. Deployment policy, so it
#: is overridable per box from the topology YAML (``settings.vllm_min_queue``) and
#: by env; see :func:`resolve_cold_start_min_queue`.
DEFAULT_COLD_START_MIN_QUEUE = 5


def resolve_cold_start_min_queue() -> int:
    """How many waiting jobs justify a cold start, resolved at CALL time so a host's
    ``configure()`` and an operator's topology edit are both honoured.

    Order: env ``QUEUE_WORKFLOWS_VLLM_MIN_QUEUE`` (operator override) → the topology
    YAML's ``settings.vllm_min_queue`` (deployment policy, editable without a
    redeploy — the file is re-read on mtime change) → :data:`DEFAULT_COLD_START_MIN_QUEUE`.
    A malformed value falls through rather than breaking dispatch. ``0`` disables the
    gate (always pay the cold start)."""
    raw = env_get("QUEUE_WORKFLOWS_VLLM_MIN_QUEUE")
    if raw:
        try:
            return max(0, int(str(raw).strip()))
        except ValueError:
            pass
    try:
        cfg = get_config()
        path = cfg.llm_topology_path
        if path:
            from queue_workflows.llm_backends import topology
            val = topology.setting(path, "vllm_min_queue")
            if val is not None:
                return max(0, int(val))
    except Exception:
        log.debug("[BackendFactory] topology vllm_min_queue unreadable", exc_info=True)
    return DEFAULT_COLD_START_MIN_QUEUE


def choose_server_type(
    preference: Iterable[str] | None,
    available: Iterable[str] | None,
    *,
    warm: Iterable[str] | None = None,
    pending: int = 0,
    min_queue: int = 0,
) -> str | None:
    """PURE. The FIRST kind in ``preference`` this box can serve AND is worth using.

    This is the one place the ordering is interpreted — the claim SQL only checks
    that the sets INTERSECT, deliberately, so the ordering decision stays here in
    python where it is unit-testable and cannot be spelled two different ways.

    THE AMORTISATION GATE. A preferred kind in :data:`COLD_START_KINDS` is skipped
    when it is neither already ``warm`` nor backed by at least ``min_queue`` waiting
    jobs — one job must not wait ~7 min for a server that the next preference can
    serve now. It is skipped only if a LATER preference is actually available:
    gating away the sole option would strand the job, so a slow start beats no start.
    ``min_queue=0`` (the default, and every pre-rule caller) disables the gate.

    * empty / ``None`` preference ⇒ ``None`` (caller uses the box default);
    * empty / ``None`` ``available`` ⇒ the first PREFERRED kind — a box that has not
      advertised its capability yet must not be forced onto its default, or a
      not-yet-probed worker would silently ignore every preference;
    * no overlap ⇒ ``None``. The claim gate should have prevented this row reaching
      this box at all, so falling back to the box default is the safe read."""
    prefs = [p for p in (preference or []) if p]
    if not prefs:
        return None
    have = {a for a in (available or []) if a}
    if not have:
        return prefs[0]
    hot = {w for w in (warm or []) if w}
    eligible = [p for p in prefs if p in have]
    for i, p in enumerate(eligible):
        if (
            min_queue > 0
            and p in COLD_START_KINDS
            and p not in hot
            and int(pending) < int(min_queue)
            and i + 1 < len(eligible)          # only skip when there IS a fallback
        ):
            log.info(
                "[llm] skipping a %s cold start — %d job(s) waiting, %d needed; "
                "using %s instead", p, int(pending), int(min_queue), eligible[i + 1],
            )
            continue
        return p
    return eligible[0] if eligible else None


#: Availability-probe cache TTL (s). resolve_for_job runs PER DISPATCH, and a
#: PAR-wide pool dispatches concurrently — probing two endpoints every time was pure
#: cost (analysis #4) and hammered the servers it was checking. 5 s is far fresher
#: than the failure modes it guards against (a server start takes minutes, a stop is
#: followed by minutes of not-running) while collapsing per-dispatch probes to ~one
#: per server per window.
_PROBE_TTL_S = 5.0
#: Grace window during which a kind that WAS answering keeps counting as alive
#: through failed probes (found live: one 5 s cache slot of "vllm dead" routed a
#: dispatch to ollama, which loaded a second an LLM copy, which fed the eviction
#: loop). 3x the TTL mirrors the heartbeat set's 3-miss debounce. A kind that was
#: never seen alive gets NO grace — a cold server must not soak up dispatches.
_PROBE_GRACE_S = 15.0
_probe_cache: dict[str, tuple[float, bool]] = {}
_probe_last_alive: dict[str, float] = {}
_probe_cache_lock = threading.Lock()


_default_kind_cache: dict[tuple[str, str], tuple[float, str]] = {}


def cached_default_kind(host_label: str, queue: str) -> str:
    """The box-default server kind with a short TTL (round-3 audit, LOW): the
    liveness fix added an UNCACHED worker_controls read per no-preference dispatch —
    50 dispatches, 50 DB reads — undoing the probe cache's point on that path. Same
    TTL as the probe; a default flip still lands within one window (and the panel
    lever's NOTIFY invalidates the backend factory anyway)."""
    now = time.monotonic()
    hit = _default_kind_cache.get((host_label, queue))
    if hit is not None and (now - hit[0]) < _PROBE_TTL_S:
        return hit[1]
    kind = worker_control.llm_config_for(host_label, queue).server_type
    _default_kind_cache[(host_label, queue)] = (now, kind)
    return kind


def _probe_available(kinds: Iterable[str]) -> list[str]:
    """Which of ``kinds`` actually ANSWER on this box right now (TTL-cached).

    A cheap liveness probe per kind (llm_probe returns [] for anything that does not
    respond), used when a caller cannot supply an availability list. A kind whose URL
    is unset or unreachable is simply absent, so the preference falls through to the
    next one instead of dispatching at a dead port."""
    from queue_workflows import llm_probe

    out: list[str] = []
    now = time.monotonic()
    for kind in kinds:
        with _probe_cache_lock:
            hit = _probe_cache.get(kind)
            if hit is not None and (now - hit[0]) < _PROBE_TTL_S:
                if hit[1]:
                    out.append(kind)
                continue
        alive = False
        try:
            url = _DEFAULT.resolve_base_url(kind)
            alive = bool(url) and llm_probe.probe_llm_servers(url) == [kind]
        except Exception:
            log.debug("[llm] availability probe for %s failed", kind, exc_info=True)
        with _probe_cache_lock:
            if alive:
                _probe_last_alive[kind] = now
            elif (now - _probe_last_alive.get(kind, -1e9)) < _PROBE_GRACE_S:
                # Debounce: it answered moments ago; treat the blip as busy, not
                # dead, so one timed-out GET can't reroute dispatches and seed a
                # second resident copy.
                alive = True
            _probe_cache[kind] = (now, alive)
        if alive:
            out.append(kind)
    return out


def _reset_probe_cache_for_tests() -> None:
    with _probe_cache_lock:
        _probe_cache.clear()
        _probe_last_alive.clear()
    _default_kind_cache.clear()


def resolve_for_job(
    job: Mapping[str, Any] | None,
    host_label: str,
    queue: str,
    *,
    available: Iterable[str] | None = None,
) -> LLMBackend:
    """The backend that should serve THIS node-job — the host-facing entry point
    behind :func:`queue_workflows.set_llm_server_resolver`.

    Reads the job's ``server_types`` preference (migration 0022), picks the first
    kind this box can serve (:func:`choose_server_type`), and returns that backend.
    A job with no preference — every job before 0022 — resolves to the box default
    exactly as it did, so a host that adopts this helper changes no behaviour until
    it starts tagging jobs."""
    prefs = (job or {}).get("server_types") if isinstance(job, Mapping) else None
    prefs = [p for p in (prefs or []) if p]
    # THE CLAIM ALREADY DECIDED (migration 0023): the arbiter counted this job
    # against the kind stamped at claim time, so serving it on any other kind would
    # break the per-kind caps it just enforced. Honour the stamp when that server
    # answers; fall through to a fresh resolution only when it does not.
    stamped = (job or {}).get("resolved_server") if isinstance(job, Mapping) else None
    job_id = (job or {}).get("id") if isinstance(job, Mapping) else None

    def _serve(kind: str | None):
        """Return the backend for ``kind`` — and, when that is NOT the kind the
        claim counted this job against, correct the row first (AUDIT DELTA D4).

        The per-kind arbiter counts running rows by ``resolved_server``. Leaving a
        stale stamp behind makes the dead kind look busy (its cap consumed by a job
        it is not running) and the serving kind look free (open to an over-claim) —
        for as long as this job runs. The arbiter re-reads the DB each claim, so
        correcting the row is the whole fix. ``kind is None`` means "the box
        default", which the stamp cannot contradict, so it is left alone."""
        if kind and stamped and kind != stamped and job_id:
            try:
                from queue_workflows import node_queue
                node_queue.restamp_resolved_server(str(job_id), kind)
                log.warning(
                    "[llm] %s/%s: job %s was claimed as '%s' but that server is not "
                    "answering — serving on '%s' and re-stamping the row so the "
                    "per-kind arbiter counts it where it actually runs",
                    host_label, queue, job_id, stamped, kind)
            except Exception:
                log.debug("[llm] re-stamp failed", exc_info=True)
        return _DEFAULT.get_backend(host_label, queue, kind)

    if stamped and stamped in _probe_available([stamped]):
        return _serve(stamped)
    if not prefs:
        # THE DEFAULT PATH MUST NOT DISPATCH BLIND (analysis #2). The box default
        # (worker_controls.llm_server_type — also what the panel's "default →" lever
        # writes) used to be dispatched at without a liveness check, so flipping the
        # default to a server that was not running routed every no-preference job at
        # a dead port — the same failure class as the 33-job Connection-refused
        # incident, resurfaced via config. Probe the default; if it is down but the
        # other kind answers, serve there. If NOTHING answers, fall through to the
        # default anyway — there is no better choice, and the error should name the
        # configured server.
        live = _probe_available([worker_control.SERVER_TYPE_OLLAMA,
                                 worker_control.SERVER_TYPE_VLLM])
        if worker_control.SERVER_TYPE_VLLM in live:
            # THE ACTIVE SERVER WINS THE DEFAULT (found live, box-b at 105/121 GB):
            # a no-preference LLM caller (compose_region_report — an LLM user with
            # no model field, so no server_types ever gets stamped) resolved to the
            # box default = ollama, whose daemon ALWAYS answers — and its internal
            # an LLM calls loaded a second 29 GB copy beside the actively-serving
            # vLLM, keep-alive-warm for the node's whole 28-minute run. A live
            # vLLM IS the box's active server (it only answers when it holds the
            # weights), and it serves the fleet's one local model — so the default
            # path must route there rather than force a second serving copy into
            # existence. ollama remains the default the moment vLLM is not live.
            return _serve(worker_control.SERVER_TYPE_VLLM)
        default_kind = cached_default_kind(host_label, queue)
        if default_kind in live or not live:
            # Pass the RESOLVED kind, not None: with never-NULL stamps (the
            # law) a claim can predict 'vllm' (warm at claim time) and
            # land here because vLLM died in between — _serve(None) skipped the D4
            # re-stamp on exactly that path, leaving the arbiter counting a vllm
            # job that the default kind is serving.
            return _serve(default_kind)
        log.warning(
            "[llm] %s/%s: the box default server '%s' is not answering — serving "
            "this no-preference job on '%s' instead", host_label, queue,
            default_kind, live[0],
        )
        return _serve(live[0])

    if available is None:
        # PROBE the box rather than treating "caller didn't say" as "unknown".
        #
        # choose_server_type reads an EMPTY availability as "this box has not
        # advertised yet — honour the preference blindly", which is right for a cold
        # worker but catastrophic here: with available=None every job resolved to the
        # FIRST preference (vllm) whether or not vLLM was running, so once the idle
        # stopper stopped it, 33 live jobs died on `[Errno 111] Connection refused`
        # in ten minutes. The caller is a per-dispatch hook that has no capability
        # list to pass, so the default must be to look, not to guess.
        #
        # Probe EVERY kind, not just the preferred ones (round-3 audit, MED):
        # probing only the prefs could not tell "box unprobed" from "all prefs
        # dead" — both read as available=[], and the blind-honour rule then
        # dispatched a ['vllm']-only job at a dead vLLM while a healthy ollama sat
        # right there. With all kinds probed, a live fallback makes the no-overlap
        # case real: the violation warning fires and the job serves on the live
        # default instead of a dead port.
        all_kinds = list(dict.fromkeys(
            [*prefs, worker_control.SERVER_TYPE_OLLAMA, worker_control.SERVER_TYPE_VLLM]))
        available = _probe_available(all_kinds)

    # AMORTISATION INPUTS. Only gather them when a cold-start-expensive kind is
    # actually preferred AND a cheaper fallback exists — otherwise the answer cannot
    # change and the probe/DB read would be pure cost on every dispatch.
    warm: list[str] = []
    pending = 0
    min_queue = 0
    if len(prefs) > 1 and any(p in COLD_START_KINDS for p in prefs):
        min_queue = resolve_cold_start_min_queue()
        if min_queue > 0:
            model = (job or {}).get("required_model") if isinstance(job, Mapping) else None
            for kind in prefs:
                if kind not in COLD_START_KINDS:
                    continue
                try:
                    from queue_workflows import llm_probe
                    url = _DEFAULT.resolve_base_url(kind)
                    # "Warm" = nothing to amortise. With a model named: the server
                    # already SERVES that model. With NO model named (a stamped
                    # LLM caller whose node carries no model field — e.g. a
                    # report composer): the server being LIVE is the whole test —
                    # its weights are resident, and amortisation-gating the row
                    # onto a cheaper kind would load a SECOND copy of the very
                    # weights it is serving (the 105/121 GB incident, through the
                    # preference door). Mirrors the no-preference path's
                    # ACTIVE-SERVER-WINS rule above.
                    if url:
                        served_here = llm_probe.vllm_served_models(url)
                        if (model in served_here) if model else bool(served_here):
                            warm.append(kind)
                except Exception:
                    log.debug("[llm] warm probe for %s failed", kind, exc_info=True)
                if kind not in warm:
                    try:
                        from queue_workflows import node_queue
                        pending = max(pending, node_queue.count_pending_for_server(
                            model, kind))
                    except Exception:
                        log.debug("[llm] pending count for %s failed", kind,
                                  exc_info=True)

    chosen = choose_server_type(
        prefs, available, warm=warm, pending=pending, min_queue=min_queue)
    if chosen is None:
        # The job named servers and NONE of them is live here (the claim gate is
        # deliberately claim-any for unprobed workers, so this row can legally land
        # on a box that serves none of its preference). Running it on the box
        # default keeps liveness — the model is the same — but it IS a violated
        # preference and must be loud, not silent (analysis #7).
        log.warning(
            "[llm] %s/%s: job %s prefers %s but none is live on this box — running "
            "on the box default instead", host_label, queue,
            job_id, prefs,
        )
    return _serve(chosen)


def invalidate(host_label: str, queue: str) -> None:
    """Process-wide :func:`BackendFactory.invalidate`."""
    _DEFAULT.invalidate(host_label, queue)


def resolve_base_url(server_type: str = worker_control.SERVER_TYPE_OLLAMA) -> str:
    """Process-wide :func:`BackendFactory.resolve_base_url` — this box's LLM ROOT URL
    for ``server_type`` (topology → env → localhost default)."""
    return _DEFAULT.resolve_base_url(server_type)


def start() -> None:
    """Start the process-wide factory's LISTEN invalidator (env-gated, idempotent).
    The gpu claim worker calls this once its heartbeat is up."""
    _DEFAULT.start()


def stop() -> None:
    """Stop the process-wide factory's invalidator + release all cached backends.
    The gpu claim worker calls this on teardown."""
    _DEFAULT.stop()


def reset_default_for_tests() -> None:
    """TEST-ONLY. Stop + replace the process-wide :data:`_DEFAULT` so a test's
    singleton state (cached backends, the running invalidator) doesn't leak into the
    next — the analog of ``gpu_model_cache._reset_gpu_model_cache_for_tests``."""
    global _DEFAULT
    try:
        _DEFAULT.stop()
    except Exception:
        log.exception("[BackendFactory] reset_default_for_tests: stop failed")
    _DEFAULT = BackendFactory()


__all__ = [
    "BackendFactory", "get_backend", "choose_server_type", "resolve_for_job",
]
