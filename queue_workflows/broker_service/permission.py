"""Client-side permission gate — the thin layer a worker container runs behind.

A client project's worker does NOT claim work or manage LLM servers itself. It
asks the broker for permission and runs only while permitted, ready to be killed
the moment the broker withdraws that permission. These four calls are the whole
client-facing surface of the v2 control model; they wrap the broker-side
orchestration primitives from the *worker's* point of view:

* :func:`ask_to_run`     — register + heartbeat + ask the broker "may I run
  something?" (one call). Returns the granted job or ``None`` (denied / no work).
* :func:`keep_permission` — renew the lease AND confirm the broker hasn't revoked
  the grant. Returns False → the worker must **stop immediately** (it was killed
  / reassigned). Call this between units of cooperative work.
* :func:`finish` / :func:`abort` — report the terminal outcome.

The actual node body + the LLM calls (to the broker-managed ollama/vLLM) run
between :func:`ask_to_run` and :func:`finish`, bracketed by :func:`keep_permission`
checks — but that runtime loop, the broker web service, and the k8s-managed LLM
servers are later passes; this module is the permission contract they build on.
"""

from __future__ import annotations

from typing import Any

from queue_workflows.broker_service import orchestrator as _o


def ask_to_run(
    worker_id: str,
    *,
    project: str,
    resource: str,
    lease_s: float = 30.0,
    capacity: int | None = None,
) -> dict[str, Any] | None:
    """Register the worker as waiting and ask the broker to grant it the next
    job. Returns the granted job dict, or ``None`` if the broker denied
    permission (shared capacity full) or there is no eligible work."""
    _o.register_worker(worker_id, project=project, resource=resource)
    return _o.grant_next(worker_id, lease_s=lease_s, capacity=capacity)


def keep_permission(job_id: str, worker_id: str, *, lease_s: float = 30.0) -> bool:
    """Renew the grant lease and confirm the worker still holds it. Returns False
    the moment the broker has revoked/reassigned the grant — the worker must stop."""
    _o.renew_grant(job_id, worker_id, lease_s=lease_s)
    return _o.has_permission(job_id, worker_id)


def finish(job_id: str, worker_id: str, *, result: Any = None) -> dict[str, Any] | None:
    """Report success. Marks the job running-then-done and frees the worker."""
    _o.begin_job(job_id, worker_id)
    return _o.complete_job(job_id, worker_id, result=result)


def abort(job_id: str, worker_id: str, *, error: str | None = None) -> dict[str, Any] | None:
    """Report failure of the granted job."""
    _o.begin_job(job_id, worker_id)
    return _o.fail_job(job_id, worker_id, error=error)
