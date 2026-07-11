"""Worker runtime behind the client permission gate (v2 control plane).

The runnable loop a client project's worker container executes. It does NOT claim
work or run its own LLM servers — it **asks the broker for permission**, runs the
granted job's handler **only while permitted**, and stops the instant the broker
revokes (kills) the grant. This is the runnable form of
:mod:`queue_workflows.broker_service.permission`.

A handler is ``fn(job, cancel) -> dict | None``: it does the work and returns a
JSON-able result; ``cancel`` is a :class:`threading.Event` set when the broker
withdraws permission mid-run (a long handler should poll ``cancel.is_set()`` /
``cancel.wait(...)`` and bail). Handlers are resolved by ``job.payload['handler']``
else the job's ``resource``; register them with
``queue_workflows.register_broker_handler(key, fn)`` (or pass a ``handlers`` dict).

A background watcher renews the grant lease (keeping a long job alive) AND detects
revocation — so a wedged/preempted job is stopped without the worker having to
guess. On revoke mid-run the worker does **not** finish: the broker already owns
the job (re-queued for the next grant).
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any, Callable

from queue_workflows.broker_service import orchestrator as _o
from queue_workflows.broker_service import permission as _p
from queue_workflows.broker_service.schema import RESOURCES, ensure_schema
from queue_workflows.config import get_config

Handler = Callable[[dict, threading.Event], Any]


def _resolve_handler(job: dict, handlers: dict[str, Handler]) -> Handler:
    payload = job.get("payload")
    key = payload.get("handler") if isinstance(payload, dict) else None
    key = key or job["resource"]
    fn = handlers.get(key)
    if fn is None:
        raise LookupError(f"no broker handler registered for {key!r}")
    return fn


def run_once(
    worker_id: str,
    *,
    project: str,
    resource: str,
    handlers: dict[str, Handler] | None = None,
    lease_s: float = 30.0,
    capacity: int | None = None,
    poll_s: float = 1.0,
    on_grant: Callable[[dict], None] | None = None,
) -> str | None:
    """One iteration: ask the broker for permission; if granted, run the job's
    handler under a revocation watcher and report the outcome. Returns the job id
    handled, or ``None`` if the broker denied permission / there was no work.

    ``on_grant`` is a test seam invoked once the job is running (before the handler)."""
    handlers = handlers if handlers is not None else get_config().broker_handlers
    job = _p.ask_to_run(
        worker_id, project=project, resource=resource, lease_s=lease_s, capacity=capacity
    )
    if job is None:
        return None
    job_id = job["job_id"]

    cancel = threading.Event()
    stop = threading.Event()

    def _watch() -> None:
        # Renew the grant lease (keeps a long job alive) AND observe revocation.
        while not stop.is_set():
            if not _p.keep_permission(job_id, worker_id, lease_s=lease_s):
                cancel.set()
                return
            stop.wait(poll_s)

    watcher = threading.Thread(target=_watch, name=f"broker-watch-{job_id[:8]}", daemon=True)
    watcher.start()
    try:
        fn = _resolve_handler(job, handlers)
        _o.begin_job(job_id, worker_id)  # granted → running (reflect in the panel)
        if on_grant is not None:
            on_grant(job)
        result = fn(job, cancel)
    except Exception as exc:  # noqa: BLE001 — a handler failure fails the job, never crashes
        stop.set()
        watcher.join(timeout=poll_s * 2 + 1)
        _p.abort(job_id, worker_id, error=str(exc))
        return job_id
    stop.set()
    watcher.join(timeout=poll_s * 2 + 1)

    if cancel.is_set() or not _o.has_permission(job_id, worker_id):
        # Broker revoked/reassigned mid-run — it owns the job now; do not finish.
        return job_id
    _p.finish(job_id, worker_id, result=result if isinstance(result, dict) else None)
    return job_id


def run_forever(
    worker_id: str,
    *,
    project: str,
    resource: str,
    handlers: dict[str, Handler] | None = None,
    lease_s: float = 30.0,
    capacity: int | None = None,
    poll_s: float = 1.0,
    idle_sleep_s: float = 1.0,
    stop_event: threading.Event | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Loop :func:`run_once` forever (until ``stop_event`` is set), sleeping
    ``idle_sleep_s`` between empty polls (broker denied / no work)."""
    stop_event = stop_event if stop_event is not None else threading.Event()
    while not stop_event.is_set():
        handled = run_once(
            worker_id, project=project, resource=resource, handlers=handlers,
            lease_s=lease_s, capacity=capacity, poll_s=poll_s,
        )
        if handled is None:
            sleep_fn(idle_sleep_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="queue-broker-worker",
        description="Run a worker behind the broker permission gate (v2 control plane).",
    )
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--project", default="")
    ap.add_argument("--resource", required=True, choices=sorted(RESOURCES))
    ap.add_argument("--lease-s", type=float, default=30.0)
    ap.add_argument("--capacity", type=int, default=None)
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--db-backend", default=None)
    ap.add_argument("--db-url-env", default=None)
    args = ap.parse_args(argv)

    import queue_workflows
    kw: dict[str, Any] = {}
    if args.db_backend:
        kw["db_backend"] = args.db_backend
    if args.db_url_env:
        kw["db_url_env"] = args.db_url_env
    if kw:
        queue_workflows.configure(**kw)
    ensure_schema()
    handlers = get_config().broker_handlers
    if not handlers:
        print("warning: no broker handlers registered — register_broker_handler(key, fn) first")
    print(f"broker worker {args.worker_id} · project={args.project or '(default)'} · resource={args.resource}")
    run_forever(
        args.worker_id, project=args.project, resource=args.resource,
        handlers=handlers, lease_s=args.lease_s, capacity=args.capacity, poll_s=args.poll_s,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
