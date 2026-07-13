"""Per-host dead-worker supervisor — the automated consumer of the engine's
dead-worker flag (migration 0009 ``worker_heartbeats.last_flagged_dead_at``).

THE GAP IT CLOSES. The orchestrator FLAGS a wedged worker — a stale heartbeat
while it still owns a running job — but deliberately never kills it: an engine
can't safely cross-host-kill a container, and a wedged process (GIL/hardware
hang) won't exit, so a docker ``restart:`` policy (which fires on *exit*, not on
an unhealthy healthcheck) never bounces it. The flag is a durable, queryable
signal with, until now, no automated consumer.

This daemon is that consumer, on the HOST side. It runs on a box, reads the flag
for the ``host_label``\\ s THAT BOX owns (its label→container map), and bounces the
local container. A fresh heartbeat from the restarted worker then clears the flag
(``upsert_worker_heartbeat`` zeroes ``last_flagged_dead_at``), so the loop is
self-terminating: bounce → worker beats → flag clears → nothing more to do.

SAFE BY DEFAULT. With no label→container map it runs **report-only** — it logs
what it *would* bounce and touches nothing. Ownership is exactly the map's keys,
so a box only ever restarts its own containers (never cross-host). A per-``(host,
queue)`` **cooldown** stops a restart storm when a bounce doesn't help. The bounce
action is injectable (:func:`queue_workflows.set_worker_bounce`) so a host that
manages workers via systemd / k8s / a custom API plugs in its own restart.

Pure-logic seam: :func:`select_bounces` decides *what* to bounce from plain data
(flagged rows + map + clock + cooldown state) with no I/O, so it's unit-tested
with a virtual clock. The daemon wires it to the DB read + the bounce action.

Console entry: ``queue-worker-supervisor`` (see pyproject). One per box.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

from queue_workflows import node_queue
from queue_workflows.config import get_config
from queue_workflows.envcompat import env_get

log = logging.getLogger(__name__)

#: env: ``label:container,label2:container2`` — which flagged host_labels this box
#: owns and the local container to restart for each. Absent ⇒ report-only.
_MAP_ENV = "QUEUE_WORKFLOWS_SUPERVISOR_MAP"
DEFAULT_POLL_S = 10.0
#: don't re-bounce the same (host_label, queue) within this window — give the last
#: restart time to take (a fresh heartbeat clears the flag) before trying again.
DEFAULT_COOLDOWN_S = 300.0


def parse_map(spec: str | None) -> dict[str, str]:
    """``"a:c1, b:c2"`` → ``{"a": "c1", "b": "c2"}``. Blank/garbage entries are
    skipped; a missing spec is an empty map (report-only)."""
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, container = part.split(":", 1)
        label, container = label.strip(), container.strip()
        if label and container:
            out[label] = container
    return out


def select_bounces(
    flagged: list[dict[str, Any]],
    label_map: dict[str, str],
    now: float,
    cooldown_s: float,
    last_bounced: dict[tuple[str, str], float],
) -> list[dict[str, str]]:
    """PURE decision: from the flagged rows, pick the ones to bounce right now.

    A row is picked iff (a) its ``host_label`` is in ``label_map`` (this box owns
    it) and (b) it wasn't bounced within ``cooldown_s``. No I/O, no clock — ``now``
    and ``last_bounced`` are passed in, so a virtual clock drives the tests."""
    picks: list[dict[str, str]] = []
    for r in flagged:
        hl = r.get("host_label")
        container = label_map.get(hl) if hl else None
        if not container:
            continue  # not ours / unmapped — never cross-host, never guess
        key = (hl, r.get("queue"))
        last = last_bounced.get(key)
        if last is not None and (now - last) < cooldown_s:
            continue  # within cooldown — let the previous bounce take effect first
        picks.append({"host_label": hl, "queue": r.get("queue"), "container": container})
    return picks


def _default_bounce(
    host_label: str, queue: str, container: str,
    *, which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
) -> bool:
    """Built-in bounce: ``docker restart <container>``. ``which``/``run`` are
    injectable so a test asserts the argv without a real docker. Best-effort —
    any failure logs and returns False (the next tick retries after cooldown)."""
    docker = which("docker")
    if docker is None:
        log.warning("[worker-supervisor] docker not found; cannot bounce %s", container)
        return False
    try:
        proc = run([docker, "restart", container], capture_output=True, timeout=60)
        ok = getattr(proc, "returncode", 1) == 0
        if not ok:
            log.error("[worker-supervisor] docker restart %s failed (rc=%s)", container,
                      getattr(proc, "returncode", "?"))
        return ok
    except OSError:
        log.exception("[worker-supervisor] docker restart %s raised", container)
        return False


class WorkerSupervisor:
    """The per-host daemon loop. Injectable seams (``list_fn``/``bounce_fn``/
    ``now_fn``/``sleep_fn``) make :meth:`tick` deterministic in tests."""

    def __init__(
        self, *,
        label_map: dict[str, str] | None = None,
        poll_s: float = DEFAULT_POLL_S,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        within_s: int = 1800,
        bounce_fn: Callable[[str, str, str], bool] | None = None,
        list_fn: Callable[[], list[dict[str, Any]]] | None = None,
        now_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.label_map = label_map if label_map is not None else parse_map(env_get(_MAP_ENV))
        self.poll_s = float(poll_s)
        self.cooldown_s = float(cooldown_s)
        self.within_s = int(within_s)
        self._bounce = bounce_fn or get_config().worker_bounce_fn or _default_bounce
        self._list = list_fn or (lambda: node_queue.flagged_dead_workers(within_s=self.within_s))
        self._now = now_fn or time.monotonic
        self._sleep = sleep_fn or time.sleep
        self._last: dict[tuple[str, str], float] = {}
        self._stop = threading.Event()

    def tick(self) -> list[dict[str, str]]:
        """One pass: read the flag, bounce the owned+eligible workers, record the
        cooldown. Best-effort — a DB blip is swallowed and retried next tick.
        Returns the picks acted on (empty in report-only mode)."""
        try:
            flagged = self._list()
        except Exception:
            log.exception("[worker-supervisor] flag read failed; retrying next tick")
            return []
        if not flagged:
            return []
        if not self.label_map:
            log.warning("[worker-supervisor] %d worker(s) flagged dead but no %s map set "
                        "(report-only): %s", len(flagged), _MAP_ENV,
                        ", ".join(f'{r.get("host_label")}/{r.get("queue")}' for r in flagged))
            return []
        now = self._now()
        picks = select_bounces(flagged, self.label_map, now, self.cooldown_s, self._last)
        for p in picks:
            ok = self._bounce(p["host_label"], p["queue"], p["container"])
            self._last[(p["host_label"], p["queue"])] = now
            log.warning("[worker-supervisor] %s dead worker %s/%s → docker restart %s",
                        "bounced" if ok else "FAILED to bounce",
                        p["host_label"], p["queue"], p["container"])
        return picks

    def run_forever(self) -> None:
        log.info("[worker-supervisor] up; owns %s; poll %.0fs cooldown %.0fs%s",
                 sorted(self.label_map) or "(nothing — report-only)",
                 self.poll_s, self.cooldown_s, "" if self.label_map else "")
        while not self._stop.is_set():
            self.tick()
            self._sleep(self.poll_s)

    def stop(self) -> None:
        self._stop.set()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="queue-worker-supervisor",
        description="Per-host daemon: restart a worker the orchestrator flagged dead.",
    )
    ap.add_argument("--map", default=None,
                    help="label:container,label2:container2 (default: $%s)" % _MAP_ENV)
    ap.add_argument("--poll", type=float, default=DEFAULT_POLL_S)
    ap.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S)
    ap.add_argument("--db-backend", default=None, help="pg | sqlite (default: sqlite)")
    ap.add_argument("--once", action="store_true", help="run one tick and exit (for cron/testing)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import queue_workflows
    if args.db_backend:
        queue_workflows.configure(db_backend=args.db_backend)
    from queue_workflows import db
    # the supervisor doesn't own migrations (only the orchestrator bootstraps). Wait
    # briefly for schema 9 (last_flagged_dead_at), but NON-fatally: a tick is
    # best-effort, so if the orchestrator isn't up yet we start anyway and the loop
    # begins working the moment the schema appears (a missing table is swallowed).
    try:
        db.wait_for_schema(9, timeout_s=30.0)
    except TimeoutError:
        log.warning("[worker-supervisor] schema 9 not ready (orchestrator not up?); "
                    "starting anyway — ticks are best-effort until it appears")

    label_map = parse_map(args.map) if args.map is not None else None
    sup = WorkerSupervisor(label_map=label_map, poll_s=args.poll, cooldown_s=args.cooldown)
    if args.once:
        sup.tick()
        return 0
    try:
        sup.run_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
