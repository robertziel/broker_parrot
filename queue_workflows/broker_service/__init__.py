"""``queue_workflows.broker_service`` — the v2 broker orchestration core.

The clean-slate control plane that inverts the engine's autonomous-worker model:
a SHARED, project-labelled CPU/GPU queue that the **broker orchestrates** —
granting a worker permission to run the next job, arbitrating a shared core/GPU
across projects, killing (revoking) a job at will, and health-checking workers to
reclaim a dead one's job for the next grant.

This is the software core only. Deliberately outside the 0001–0017 migration
chain (fresh schema via :func:`ensure_schema`, no ledger). It coexists with the
legacy engine so the existing suite stays green; the broker web service + panel,
the Kubernetes-managed ollama/vLLM servers, removing per-project LLM servers, and
retiring the old engine are subsequent passes.

Public surface:

* schema         — :func:`ensure_schema`, :data:`RESOURCES`
* broker side    — :func:`register_worker`, :func:`worker_heartbeat`,
  :func:`submit_job`, :func:`grant_next`, :func:`has_permission`,
  :func:`begin_job`, :func:`renew_grant`, :func:`complete_job`, :func:`fail_job`,
  :func:`revoke`, :func:`sweep_unhealthy`, :func:`get_job`, :func:`get_worker`
* client gate    — :func:`ask_to_run`, :func:`keep_permission`, :func:`finish`,
  :func:`abort`
"""

from __future__ import annotations

from queue_workflows.broker_service.orchestrator import (
    begin_job,
    complete_job,
    fail_job,
    get_job,
    get_worker,
    grant_next,
    has_permission,
    list_jobs,
    list_workers,
    queue_counts,
    register_worker,
    renew_grant,
    revoke,
    submit_job,
    sweep_unhealthy,
    worker_heartbeat,
)
from queue_workflows.broker_service.permission import (
    abort,
    ask_to_run,
    finish,
    keep_permission,
)
from queue_workflows.broker_service.schema import RESOURCES, ensure_schema

__all__ = [
    "ensure_schema",
    "RESOURCES",
    "register_worker",
    "worker_heartbeat",
    "submit_job",
    "grant_next",
    "has_permission",
    "begin_job",
    "renew_grant",
    "complete_job",
    "fail_job",
    "revoke",
    "sweep_unhealthy",
    "get_job",
    "get_worker",
    "list_jobs",
    "list_workers",
    "queue_counts",
    "ask_to_run",
    "keep_permission",
    "finish",
    "abort",
]
