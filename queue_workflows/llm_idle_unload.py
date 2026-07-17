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

MIRRORS ``model_cache.py`` on purpose so the two read the same:

  * the pure decision :func:`should_unload_external_model` is the analog of
    ``gpu_should_unload`` (loaded + nothing running + idle ≥ ttl; ttl<=0 disables);
  * :meth:`ExternalModelIdleReaper.reap_idle_once` is one tick of the reaper;
  * :meth:`_loop` polls on the same ``max(5, min(60, ttl/5))`` cadence;
  * :meth:`ensure_started` gates on ttl>0 + the ``QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER``
    kill-switch (shared with ModelCache — one switch silences both idle reapers).

DECOUPLING. Everything is injected — ``active_fn`` / ``idle_seconds_fn`` (the GPU
worker's busy + quiet signals), ``loaded_models_fn`` / ``unload_fn`` (the ollama
HTTP, wired to :mod:`queue_workflows.llm_probe`), and ``now_fn`` / ``sleep_fn`` — so
the reaper runs on a virtual clock against a fake ollama with no real waiting and no
network. It imports nothing from the backend modules. Best-effort throughout: a tick
that can't reach ollama unloads nothing and never raises, exactly like ModelCache's.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from queue_workflows.envcompat import env_get

log = logging.getLogger(__name__)

#: Default idle TTL (s) before an idle box's LLM-server model is unloaded; <=0
#: disables. Default 300 (5 min) — long enough to keep a hot model through a lull
#: between jobs, short enough that a quiet or PARKED box frees its RAM promptly.
DEFAULT_LLM_IDLE_TTL_S = float(env_get("QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S", "300"))

#: Poll-cadence bounds (mirror ModelCache._idle_reaper_loop's clamp).
_POLL_FLOOR_S = 5.0
_POLL_CEIL_S = 60.0

#: Shared with ModelCache — one env kill-switch silences BOTH idle reapers (tests).
_DISABLE_ENV = "QUEUE_WORKFLOWS_DISABLE_GPU_IDLE_REAPER"


def should_unload_external_model(
    model_loaded: bool, active: int, idle_s: float, ttl_s: float,
) -> bool:
    """Pure decision: unload only when a model IS loaded on the server, no GPU job
    is running (``active <= 0``), and the box has been idle at least ``ttl_s``.
    ``ttl_s <= 0`` disables idle unload entirely. Same shape as
    :func:`queue_workflows.model_cache.gpu_should_unload`."""
    if ttl_s <= 0:
        return False
    return bool(model_loaded) and active <= 0 and idle_s >= ttl_s


class ExternalModelIdleReaper:
    """Daemon that unloads an idle ollama box's model. All I/O injected (see module
    docstring). One tick = :meth:`reap_idle_once`; :meth:`ensure_started` arms the
    background poller once."""

    def __init__(
        self,
        *,
        active_fn: Callable[[], int],
        idle_seconds_fn: Callable[[], float],
        loaded_models_fn: Callable[[], list[str]],
        unload_fn: Callable[[list[str]], object],
        ttl_s: float | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        label: str = "?",
    ) -> None:
        self._active_fn = active_fn
        self._idle_seconds_fn = idle_seconds_fn
        self._loaded_models_fn = loaded_models_fn
        self._unload_fn = unload_fn
        self._ttl_s = DEFAULT_LLM_IDLE_TTL_S if ttl_s is None else float(ttl_s)
        self._now = now_fn
        self._sleep = sleep_fn
        self._label = label
        self._started = False

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def reap_idle_once(self) -> list[str]:
        """Unload the box's LLM-server model iff it is loaded, no GPU job runs, and
        the box has been idle ≥ TTL. Returns the unloaded ids ([] if it did nothing).

        The cheap LOCAL checks (ttl / active / idle) run FIRST so the ollama query
        (``loaded_models_fn``) only fires when the box is actually idle past TTL — a
        quiet steady state costs no HTTP. Never raises."""
        try:
            if self._ttl_s <= 0:
                return []
            active = self._active_fn()
            idle_s = self._idle_seconds_fn()
            if active > 0 or idle_s < self._ttl_s:
                return []
            models = list(self._loaded_models_fn() or [])
            if not should_unload_external_model(bool(models), active, idle_s, self._ttl_s):
                return []
            log.info(
                "[llm-idle] %s idle %.0fs ≥ TTL %.0fs — unloading %s to free RAM",
                self._label, idle_s, self._ttl_s, ", ".join(models),
            )
            self._unload_fn(models)
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


__all__ = [
    "should_unload_external_model",
    "ExternalModelIdleReaper",
    "DEFAULT_LLM_IDLE_TTL_S",
]
