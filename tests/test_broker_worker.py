"""Worker runtime behind the broker permission gate.

Drives ``broker_service.worker.run_once`` on the hermetic SQLite store: the worker
asks the broker, runs the granted job's handler only while permitted, and stops
(without finishing) the instant the broker revokes the grant.
"""

from __future__ import annotations

import threading

import pytest

from queue_workflows import broker_service as bs
from queue_workflows.broker_service import worker as w
from queue_workflows import db


@pytest.fixture(autouse=True)
def _fresh():
    bs.ensure_schema()
    with db.connection() as conn, conn.cursor() as cur:
        for tbl in ("bw_jobs", "bw_workers"):
            cur.execute(f"DELETE FROM {tbl}")
        conn.commit()
    yield


def test_grant_run_finish():
    jid = bs.submit_job(project="A", resource="cpu", payload={"n": 3})
    seen = {}

    def handler(job, cancel):
        seen["payload"] = job["payload"]
        return {"double": job["payload"]["n"] * 2}

    out = w.run_once("wA", project="A", resource="cpu", handlers={"cpu": handler})
    assert out == jid
    assert seen["payload"] == {"n": 3}
    job = bs.get_job(jid)
    assert job["status"] == "done" and job["result"] == {"double": 6}


def test_no_work_returns_none():
    assert w.run_once("wA", project="A", resource="cpu", handlers={"cpu": lambda j, c: {}}) is None


def test_denied_capacity_returns_none():
    bs.submit_job(project="A", resource="cpu")
    out = w.run_once("wA", project="A", resource="cpu", handlers={"cpu": lambda j, c: {}}, capacity=0)
    assert out is None
    # the job is untouched (still queued) because permission was denied.
    jobs = bs.list_jobs(project="A")
    assert jobs and jobs[0]["status"] == "queued"


def test_handler_exception_marks_failed():
    jid = bs.submit_job(project="A", resource="cpu")

    def boom(job, cancel):
        raise RuntimeError("kaboom")

    out = w.run_once("wA", project="A", resource="cpu", handlers={"cpu": boom})
    assert out == jid
    job = bs.get_job(jid)
    assert job["status"] == "failed" and "kaboom" in (job["error"] or "")


def test_missing_handler_marks_failed_not_crash():
    jid = bs.submit_job(project="A", resource="cpu")
    out = w.run_once("wA", project="A", resource="cpu", handlers={})  # no 'cpu' handler
    assert out == jid
    assert bs.get_job(jid)["status"] == "failed"


def test_handler_resolved_by_payload_key():
    jid = bs.submit_job(project="A", resource="cpu", payload={"handler": "special", "x": 1})
    ran = {}

    def special(job, cancel):
        ran["ok"] = True
        return {}

    w.run_once("wA", project="A", resource="cpu", handlers={"special": special})
    assert ran.get("ok") is True
    assert bs.get_job(jid)["status"] == "done"


def test_revoke_midrun_does_not_finish_deterministic():
    jid = bs.submit_job(project="A", resource="cpu")

    def handler(job, cancel):
        # the broker kills the grant while the handler runs
        bs.revoke(job["job_id"], requeue=True, reason="preempt")
        return {"should_not": "apply"}

    out = w.run_once("wA", project="A", resource="cpu", handlers={"cpu": handler}, poll_s=5)
    assert out == jid
    # worker must NOT have finished it; the broker re-queued it for the next grant.
    assert bs.get_job(jid)["status"] == "queued"


def test_watcher_cancels_handler_on_external_revoke():
    jid = bs.submit_job(project="A", resource="cpu")
    started = threading.Event()

    def handler(job, cancel):
        started.set()
        cancel.wait(timeout=5)  # block until the broker revokes (watcher sets cancel)
        return {"reached": "end"}

    holder = {}

    def run():
        holder["out"] = w.run_once(
            "wA", project="A", resource="cpu", handlers={"cpu": handler}, poll_s=0.05
        )

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(3), "handler never started running"
    bs.revoke(jid, requeue=True, reason="operator")  # broker kills mid-run
    thread.join(timeout=5)
    assert not thread.is_alive(), "run_once did not return after the broker revoked"
    assert holder["out"] == jid
    assert bs.get_job(jid)["status"] == "queued"  # revoked+requeued, not finished


def test_register_broker_handler_config_hook():
    import queue_workflows
    from queue_workflows.config import get_config

    got = {}
    queue_workflows.register_broker_handler("cpu", lambda job, cancel: got.setdefault("ran", True) or {})
    assert "cpu" in get_config().broker_handlers
    jid = bs.submit_job(project="A", resource="cpu")
    # handlers=None ⇒ run_once falls back to the config registry.
    w.run_once("wA", project="A", resource="cpu")
    assert got.get("ran") is True
    assert bs.get_job(jid)["status"] == "done"
