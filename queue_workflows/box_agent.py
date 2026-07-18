"""Per-box fleet supervisor — the SINGLE source of truth for what runs on a box.

The fleet used to park a lane THREE unreconciled ways: commented-out in the box's
compose (host-level, invisible to the DB and the panel), an ``off`` row in
``worker_controls`` (the desired-state the panel reads), or simply not beating
(``worker_heartbeats``). They drifted apart — a render worker sat compose-disabled
while ``worker_controls`` still said ``on`` and the panel showed the box up.

This module is the fix: **one agent per box, ``worker_controls`` is the source of
truth, and the agent continuously reconciles the box's actual running lanes to match
it** — start a lane that should be on and isn't, stop one that should be off, restart
one that died. "Compose-disabled" stops existing as a hidden state: every lane the
box CAN run is declared in the agent's manifest, and whether it runs is decided only
by ``worker_controls``.

The pure core (:func:`reconcile`, :func:`desired_from_controls`) is unit-tested on
plain data; :class:`BoxAgent` is a thin, seam-injected shell over it so the spawn/kill
side is testable with a fake process. The DB-reading, subprocess-spawning runner
(:func:`run_forever`) is the only part that touches the outside world.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from queue_workflows.envcompat import env_get

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Lane:
    """One launchable worker the box CAN run. ``key`` is a stable id
    (``"<project>:<queue>"``); ``argv``/``env`` are the launch spec the runner uses
    to spawn it (unused by the pure core)."""

    key: str
    project: str
    queue: str
    argv: tuple[str, ...] = ()
    #: value None ⇒ REMOVE the key from the child env (see :func:`_lane_env`)
    env: Mapping[str, str | None] = field(default_factory=dict)


def reconcile(
    lanes: Sequence[Lane], desired: Mapping[str, bool], running: set[str],
) -> tuple[list[Lane], list[Lane]]:
    """PURE. Given the box's ``lanes``, the ``desired`` on/off per lane key, and which
    lane keys are currently ``running``, return ``(to_start, to_stop)``: a lane
    desired-on but not running ⇒ start; desired-off but running ⇒ stop. A lane absent
    from ``desired`` defaults to ON (the engine's ``worker_controls`` contract:
    absent = ON). Order follows ``lanes`` for determinism."""
    to_start: list[Lane] = []
    to_stop: list[Lane] = []
    for lane in lanes:
        want_on = desired.get(lane.key, True)
        is_running = lane.key in running
        if want_on and not is_running:
            to_start.append(lane)
        elif not want_on and is_running:
            to_stop.append(lane)
    return to_start, to_stop


def _host_matches(row_host: str, box: str) -> bool:
    """A ``worker_controls`` row governs this box if its ``host_label`` is the bare
    box name (``box-a``) or one of the box's lane labels (``box-a-gpu``). The
    ``box + "-"`` guard stops ``box-a`` from matching ``box-a2``."""
    return row_host == box or row_host.startswith(box + "-")


def desired_from_controls(
    rows: Iterable[tuple[str, str, str, str]], *, host: str, lanes: Sequence[Lane],
) -> dict[str, bool]:
    """Map ``worker_controls`` rows ``(host_label, queue, project, desired_state)`` to
    a per-lane on/off dict for this ``host``. Every lane defaults ON; an ``off`` row
    whose host matches the box and whose ``queue`` matches the lane parks it. A row
    with an empty ``project`` is box-wide (parks every lane of that queue); a
    project-scoped row parks only that project's lane."""
    desired = {lane.key: True for lane in lanes}
    for row_host, queue, project, state in rows:
        if state != "off" or not _host_matches(row_host, host):
            continue
        for lane in lanes:
            if lane.queue == queue and (not project or project == lane.project):
                desired[lane.key] = False
    return desired


class BoxAgent:
    """Holds the box's lane manifest + live worker handles and reconciles them to
    ``worker_controls`` on every :meth:`tick`. Seams (``fetch_controls`` / ``spawn_fn``
    / ``kill_fn``) are injected so the loop is testable without a DB or real
    processes. ``spawn_fn(lane)`` returns a handle exposing ``poll()`` (Popen
    contract: ``None`` ⇒ still running); ``kill_fn(lane, handle)`` stops it."""

    def __init__(
        self,
        *,
        host: str,
        lanes: Sequence[Lane],
        fetch_controls: Callable[[], Iterable[tuple[str, str, str, str]]],
        spawn_fn: Callable[[Lane], Any],
        kill_fn: Callable[[Lane, Any], None],
    ):
        self.host = host
        self.lanes = list(lanes)
        self.fetch_controls = fetch_controls
        self.spawn_fn = spawn_fn
        self.kill_fn = kill_fn
        self.procs: dict[str, Any] = {}
        self._lane_by_key = {lane.key: lane for lane in self.lanes}

    def _running(self) -> set[str]:
        """Live lane keys, pruning any handle whose process has exited (so a crashed
        lane reads as not-running and gets restarted next reconcile)."""
        dead = [key for key, proc in self.procs.items() if proc.poll() is not None]
        for key in dead:
            self.procs.pop(key, None)
            log.warning("[box-agent] lane %s exited — will restart", key)
        return set(self.procs)

    def tick(self) -> None:
        """One reconcile pass: read desired (worker_controls), diff against actual,
        start/stop to converge. Best-effort per lane — one lane's spawn/kill failure
        never aborts the pass."""
        desired = desired_from_controls(
            self.fetch_controls(), host=self.host, lanes=self.lanes)
        to_start, to_stop = reconcile(self.lanes, desired, self._running())
        for lane in to_start:
            try:
                self.procs[lane.key] = self.spawn_fn(lane)
                log.info("[box-agent] started lane %s", lane.key)
            except Exception:
                log.exception("[box-agent] failed to start %s", lane.key)
        for lane in to_stop:
            proc = self.procs.pop(lane.key, None)
            if proc is None:
                continue
            try:
                self.kill_fn(lane, proc)
                log.info("[box-agent] stopped lane %s (parked off)", lane.key)
            except Exception:
                log.exception("[box-agent] failed to stop %s", lane.key)


# ── the runner: manifest + real subprocess + multi-DB control fetch + loop ────
#
# The MANIFEST is the box's declarative replacement for its pile of compose services:
# a JSON file listing every lane the box CAN run and how to launch it. Whether a lane
# runs is decided ONLY by worker_controls (fetched per lane's control DB), so parking
# is one source of truth. One agent process per box (one container, boot-persistent).


def load_manifest(path: str) -> tuple[list[Lane], dict[str, str | None]]:
    """Parse a box manifest ``{"lanes": [{key, project, queue, argv, env,
    control_dsn}, ...]}`` into ``(lanes, control_dsn_by_key)``. ``control_dsn`` is the
    project DB whose ``worker_controls`` governs that lane."""
    with open(path) as fh:
        raw = json.load(fh)
    lanes: list[Lane] = []
    control_dsn: dict[str, str | None] = {}
    for entry in raw.get("lanes", []):
        lanes.append(Lane(
            key=entry["key"], project=entry.get("project", ""), queue=entry["queue"],
            argv=tuple(entry.get("argv", ())), env=dict(entry.get("env") or {}),
        ))
        control_dsn[entry["key"]] = entry.get("control_dsn")
    return lanes, control_dsn


def _read_worker_controls(conn: Any, cur: Any) -> list[tuple[str, str, str, str]]:
    """SELECT the control rows, tolerating a pre-0019 schema (no ``project`` column)
    — a host DB on an older migration line must still be able to park its lanes.
    The failed first SELECT aborts the txn, so roll back before the retry."""
    try:
        cur.execute("SELECT host_label, queue, COALESCE(project, ''), "
                    "desired_state FROM worker_controls")
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        cur.execute("SELECT host_label, queue, '', desired_state FROM worker_controls")
    return [(h, q, p, s) for h, q, p, s in cur.fetchall()]


def fetch_controls_from_dbs(
    control_dsn: Mapping[str, str | None],
) -> list[tuple[str, str, str, str]]:
    """Read ``worker_controls`` from each DISTINCT control DB and merge the rows.
    FAIL-OPEN: a DB that's unreachable (or predates migration 0012) contributes no
    rows, so its lanes default ON — a control-DB blip must never silently park a
    healthy box."""
    import psycopg

    rows: list[tuple[str, str, str, str]] = []
    for dsn in {d for d in control_dsn.values() if d}:
        try:
            with psycopg.connect(dsn, connect_timeout=4) as conn, conn.cursor() as cur:
                rows.extend(_read_worker_controls(conn, cur))
        except Exception:
            log.warning("[box-agent] worker_controls unreadable for a DB "
                        "(its lanes default ON)", exc_info=True)
    return rows


def _lane_env(lane: Lane) -> dict[str, str]:
    """The child environment: the agent's own env overlaid with the lane's. A manifest
    env value of null/None REMOVES the key — e.g. the agent itself runs the engine off
    a PYTHONPATH tree, but a lane running a host app's baked engine install must not
    inherit it (a silent engine swap under a pinned deployment)."""
    env: dict[str, Any] = {**os.environ, **dict(lane.env)}
    return {k: v for k, v in env.items() if v is not None}


def _spawn(lane: Lane) -> subprocess.Popen:
    return subprocess.Popen(list(lane.argv), env=_lane_env(lane))


def _kill(lane: Lane, proc: subprocess.Popen, *, grace_s: float = 30.0) -> None:
    """Graceful stop: SIGTERM (claim workers drain their in-flight job + release the
    box lease), then SIGKILL if it overstays."""
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except Exception:
        proc.kill()


def _shutdown(agent: BoxAgent, *, kill_fn: Callable[[Lane, Any], None] = _kill) -> None:
    """Stop every running lane gracefully, then exit. The agent is the container's
    PID 1: ``docker stop`` TERMs only US, so we must forward the stop to the lanes —
    otherwise claim workers get no drain and are KILLed with the container at the end
    of the grace period."""
    log.info("[box-agent] shutdown — stopping %d running lane(s)", len(agent.procs))
    for key, proc in list(agent.procs.items()):
        lane = agent._lane_by_key.get(key)
        try:
            kill_fn(lane, proc)
        except Exception:
            log.exception("[box-agent] shutdown stop failed for %s", key)
        agent.procs.pop(key, None)
    raise SystemExit(0)


def run_forever(manifest_path: str | None = None, *, interval_s: float = 30.0) -> None:
    """Load the box manifest and reconcile it to ``worker_controls`` forever (default
    every 30 s). The box label is ``QUEUE_WORKFLOWS_GPU_BOX_ID`` (the same identity the
    box lease uses) or the hostname. Resilient: a tick failure logs and the agent
    stays up. SIGTERM/SIGINT forward the stop to every lane (graceful drain)."""
    import signal

    manifest_path = manifest_path or env_get("QUEUE_WORKFLOWS_BOX_MANIFEST")
    if not manifest_path:
        raise SystemExit("box-agent: set QUEUE_WORKFLOWS_BOX_MANIFEST to a manifest path")
    host = (env_get("QUEUE_WORKFLOWS_GPU_BOX_ID") or socket.gethostname()).strip()
    lanes, control_dsn = load_manifest(manifest_path)
    agent = BoxAgent(
        host=host, lanes=lanes,
        fetch_controls=lambda: fetch_controls_from_dbs(control_dsn),
        spawn_fn=_spawn, kill_fn=_kill,
    )
    signal.signal(signal.SIGTERM, lambda *_: _shutdown(agent))
    signal.signal(signal.SIGINT, lambda *_: _shutdown(agent))
    log.info("[box-agent] box=%s reconciling %d lane(s) every %ss (source of truth: "
             "worker_controls)", host, len(lanes), interval_s)
    while True:
        try:
            agent.tick()
        except Exception:
            log.exception("[box-agent] reconcile tick failed (agent stays up)")
        _time.sleep(interval_s)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Per-box fleet supervisor.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_forever(args.manifest, interval_s=args.interval)


if __name__ == "__main__":  # pragma: no cover
    main()
