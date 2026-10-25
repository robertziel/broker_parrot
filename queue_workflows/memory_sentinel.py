"""THE MEMORY SENTINEL — a box that is running out of memory gives its LLM server back
NOW, every second, and is never parked.

THE DESIGN RULE: check RAM every second; once it falls below the kill floor, stop the
server immediately — but don't park the box, so the server can start again. A new
server start adjusts to the RAM that is available. Never park.

WHY A THREAD OF ITS OWN. The gpu lane's memory guards ran on the box-lease tick — every
~10 s, and only on a START. A large co-tenant training run landing on a unified-memory
box beside a serving vLLM and ollama's resident model can drive MemAvailable to near
zero inside a minute, until sshd can no longer fork; the only remedy then applied was
to park the lane — the one thing this design forbids, because a parked lane stays out of service until
someone returns it. A box in danger needs the biggest thing the engine controls given
back within a second, and then LEFT ON: the next start is planned against what is
actually free (:func:`llm_context.vllm_plan`), and everything that fits keeps flowing.

TWO LINES, both on ``MemAvailable`` (what the kernel can actually hand out — ``MemFree``
sits near zero on any busy Linux box because the page cache takes what nobody is using,
so a MemFree line would trip on a healthy box reading a model off NFS):

  * the DANGER floor (``QUEUE_WORKFLOWS_LLM_DANGER_FLOOR_MB``, 12 GB): shed the LLM
    servers, at most once per cooldown — memory comes back over seconds, and a second
    ``docker stop`` racing the first helps nobody;
  * the KILL floor (``QUEUE_WORKFLOWS_LLM_KILL_FLOOR_MB``, 512 MB): shed NOW,
    cooldown or not. This is the last resort; a box can already be unreachable with a
    few GB still in MemAvailable, so the danger floor is the line
    that saves a box and this one is the line for a box that fell through it.

The sentinel decides nothing about starts — it only reads one number and calls one
action. Deciding what to start, and how big, stays with the plan.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from queue_workflows.config import _env_flag
from queue_workflows.envcompat import env_get

log = logging.getLogger(__name__)

DISABLE_ENV = "QUEUE_WORKFLOWS_DISABLE_MEMORY_SENTINEL"
KILL_FLOOR_ENV = "QUEUE_WORKFLOWS_LLM_KILL_FLOOR_MB"
#: The operator's number: below half a gigabyte available, stop the server at once.
DEFAULT_KILL_FLOOR_MB = 512
#: Once a second — the design cadence. A /proc/meminfo read is microseconds.
INTERVAL_S = 1.0


def disabled() -> bool:
    return _env_flag(DISABLE_ENV)


def kill_floor_bytes() -> int:
    raw = (env_get(KILL_FLOOR_ENV, "") or "").strip()
    try:
        mb = int(raw) if raw else DEFAULT_KILL_FLOOR_MB
    except ValueError:
        mb = DEFAULT_KILL_FLOOR_MB
    return max(0, mb) * 1024 * 1024


class MemorySentinel:
    """A daemon thread that reads ``MemAvailable`` every ``interval_s`` and hands the
    number to ``act_fn(available_bytes)`` — the worker's shedding logic. Injectable
    ``read_fn`` / ``sleep_fn`` / ``now_fn`` so the cadence is tested with a virtual clock
    and no real waiting; an ``act_fn`` that raises is logged and never stops the loop
    (the whole point is to keep looking)."""

    def __init__(
        self, *, read_fn: Callable[[], int | None], act_fn: Callable[[int], None],
        interval_s: float = INTERVAL_S, sleep_fn: Callable[[float], None] = time.sleep,
        label: str = "?",
    ) -> None:
        self._read = read_fn
        self._act = act_fn
        self._interval = max(0.05, float(interval_s))
        self._sleep = sleep_fn
        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ticks = 0
        self.failures = 0

    def tick(self) -> None:
        """One read, one call. Public so a test (or a caller with its own clock) can
        drive it without the thread."""
        self.ticks += 1
        try:
            available = self._read()
        except Exception:
            self.failures += 1
            log.debug("[memory-sentinel] %s: meminfo read failed", self._label, exc_info=True)
            return
        if available is None:
            return                              # an unreadable meminfo is not danger
        try:
            self._act(int(available))
        except Exception:
            self.failures += 1
            log.exception("[memory-sentinel] %s: shedding raised (the sentinel keeps "
                          "watching)", self._label)

    def run(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self._sleep(self._interval)

    def start(self) -> "MemorySentinel":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self.run, name=f"memory-sentinel[{self._label}]", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


__all__ = ["DEFAULT_KILL_FLOOR_MB", "DISABLE_ENV", "INTERVAL_S", "KILL_FLOOR_ENV",
           "MemorySentinel", "disabled", "kill_floor_bytes"]
