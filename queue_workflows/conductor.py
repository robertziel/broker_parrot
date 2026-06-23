"""``queue-conductor`` — the operator-facing READ side of the GO-half conductor.

A single-DB fleet capacity view: it renders :func:`queue_workflows.node_queue.fleet_snapshot`
(the observed ``worker_heartbeats`` rows — capacity, ``current_model``, fresh /
dead) for whatever database the engine's ``db_url_env`` points at, exactly like
every other console script (``queue-orchestrator`` / ``queue-claim-worker`` /
``queue-worker-control``). The operator supplies the DSN via env; there are **no
stored fleet credentials** and **no networked service** here.

Scope, on purpose:

  * READ-ONLY. Worker ON/OFF control already lives in ``queue-worker-control``
    (``worker_control.set_worker_control``) — this module does not duplicate it.
  * SINGLE-DB. It is the building block of the plan's Phase-1 "Option A" (an
    operator runs the view per app DB). The **networked, multi-DB daemon + web
    UI** that would aggregate ~35 app DBs — and the inference proxy — are a
    separate, human-gated build (they touch production credentials / shared
    infra and aren't testable against one local Postgres). See
    ``worklog/conductor-client-split.md``.

Usage::

    queue-conductor                 # table of every reporting worker
    queue-conductor --queue gpu     # filter to one queue
    queue-conductor --json          # machine-readable (for piping)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from queue_workflows import node_queue

log = logging.getLogger("queue_workflows.conductor")


def render_fleet(rows: list[dict[str, Any]], *, as_json: bool = False) -> str:
    """Render the ``fleet_snapshot`` rows for an operator.

    ``as_json`` ⇒ a JSON array (datetimes stringified) for piping. Otherwise a
    compact fixed-width table whose STATUS column collapses the two derived flags
    to a single token: ``DEAD`` (``flagged_dead``) outranks ``stale`` (not
    ``fresh``) outranks ``ok``.
    """
    if as_json:
        return json.dumps(rows, default=str, indent=2)
    if not rows:
        return "fleet: no workers reporting"

    header = (
        f"{'QUEUE':<6} {'HOST':<24} {'MODEL':<18} {'STATUS':<6} "
        f"{'VRAM_MB':>8}  SERVERS"
    )
    lines = [header]
    for r in rows:
        if r.get("flagged_dead"):
            status = "DEAD"
        elif not r.get("fresh"):
            status = "stale"
        else:
            status = "ok"
        model = r.get("current_model") or "-"
        vram = r.get("vram_total_mb")
        vram_s = str(vram) if vram is not None else "-"
        servers = ",".join(r.get("llm_servers_available") or [])
        lines.append(
            f"{r['queue']:<6} {r['host_label']:<24} {model:<18} {status:<6} "
            f"{vram_s:>8}  {servers}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``queue-conductor [--queue Q] [--stale-after S] [--json]`` — print the
    observed fleet capacity view for the configured database. Read-only; control
    lives in ``queue-worker-control``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="queue-conductor",
        description=(
            "Read-only fleet capacity view (worker_heartbeats) for the "
            "configured DB. Worker ON/OFF control lives in queue-worker-control."
        ),
    )
    parser.add_argument(
        "--queue", default=None,
        help="filter to one queue (cpu | gpu | <ingest queue>)",
    )
    parser.add_argument(
        "--stale-after", type=float, default=30.0,
        help="seconds since last_seen before a worker is 'stale' (default 30)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit a JSON array instead of a table",
    )
    args = parser.parse_args(argv)

    rows = node_queue.fleet_snapshot(stale_after_s=args.stale_after)
    if args.queue:
        rows = [r for r in rows if r["queue"] == args.queue]
    print(render_fleet(rows, as_json=args.json))
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Console entry point (``queue-conductor``): run :func:`main`, then release
    the connection pool so this short-lived process exits promptly and cleanly —
    otherwise the pool's background threads are only reaped at interpreter
    shutdown, printing noisy ``couldn't stop thread`` warnings. Programmatic /
    in-process callers use :func:`main` directly so their pool isn't torn down."""
    from queue_workflows import db

    try:
        return main(argv)
    finally:
        db.close_pool()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
