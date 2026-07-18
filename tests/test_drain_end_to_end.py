"""END-TO-END proof of stop_policy="drain" — a REAL run_forever worker.

The unit contracts (test_worker_control_watcher) pin the handler/boundary pieces;
this proves the integrated behavior the operator asked for, against a live worker
thread and the real DB:

  1. a job is IN FLIGHT when the operator parks the box with drain
  2. that job COMPLETES normally (kept — never re-queued, never killed)
  3. while parked, NEW queued work is NOT claimed (excluded from assignment)
  4. flipping back ON resumes claiming and the new work completes

PG-only: run_forever's LISTEN loops need real NOTIFY (the sqlite shim covers the
claim SQL, not the wake channels).
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types
import uuid

import pytest

import queue_workflows
from queue_workflows import dispatcher, node_queue, run_store, worker_control
from queue_workflows.claim_worker import ClaimWorker

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("QUEUE_WORKFLOWS_TEST_SQLITE")),
    reason="drain e2e needs real LISTEN/NOTIFY (pg only)",
)

_HOST = "drain-e2e-cpu"


def _install_fake_node(name: str, run_fn):
    mod = types.ModuleType(f"qwf_drain_nodes.{name}")
    mod.run = run_fn
    sys.modules[f"qwf_drain_nodes.{name}"] = mod


def _wire_workflow(node_name: str):
    workflows = {
        "_drain_wf": {
            "name": "_drain_wf",
            "steps": [{"id": "p", "kind": "pipeline", "pipeline": "_drain_pipe"}],
        }
    }
    pipelines = {
        "_drain_pipe": {
            "name": "_drain_pipe",
            "nodes": [{"id": "n1", "node": node_name}],
        }
    }
    queue_workflows.set_workflow_provider(
        lambda n: workflows[n], lambda n: pipelines[n],
    )


def _start_run() -> str:
    run_id = str(uuid.uuid4())
    run_store.insert_run(run_id=run_id, workflow_name="_drain_wf",
                         out_dir=None, status="queued", mode="node")
    assert dispatcher.start_run(run_id) == 1
    return run_id


def _job_of(run_id: str) -> dict:
    return node_queue.list_jobs_for_run(run_id)[0]


def _wait(pred, timeout_s: float, what: str):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.25)
    raise AssertionError(f"timed out waiting for: {what}")


def test_drain_end_to_end_keeps_inflight_blocks_new_claims_and_resumes(monkeypatch):
    # fast watcher/park polling so the test converges in seconds
    monkeypatch.setenv("QUEUE_WORKFLOWS_WORKER_CONTROL_POLL_S", "0.5")
    monkeypatch.delenv("QUEUE_WORKFLOWS_DISABLE_WORKER_CONTROL", raising=False)
    monkeypatch.delenv("AI_LEADS_DISABLE_WORKER_CONTROL", raising=False)

    queue_workflows.set_node_module_package("qwf_drain_nodes")
    started = threading.Event()   # the in-flight node signals it's executing
    gate = threading.Event()      # released by the test → the node finishes

    def gated_run(inputs: dict, out=None):
        started.set()
        assert gate.wait(timeout=30), "test gate never released"
        return {"context_delta": {"held": True}}

    _install_fake_node("gated_node", gated_run)
    _wire_workflow("gated_node")

    run_a = _start_run()
    worker = ClaimWorker(queue="cpu", host=_HOST)
    t = threading.Thread(target=worker.run_forever, daemon=True, name="drain-e2e")
    t.start()
    try:
        # 1) job A is in flight
        _wait(started.is_set, 15, "job A to start executing")

        # 2) operator parks with DRAIN while A runs → worker flagged, A untouched
        worker_control.set_worker_control(
            host_label=_HOST, queue="cpu", desired_state="off", stop_policy="drain",
        )
        _wait(lambda: worker.drain_requested, 10, "watcher to flag the drain")
        row = _job_of(run_a)
        assert row["status"] == "running" and row["claimed_by"] == _HOST

        # 3) release the gate → A COMPLETES (the kept last job)
        gate.set()
        _wait(lambda: _job_of(run_a)["status"] == "completed", 15,
              "in-flight job A to complete during drain")
        assert (_job_of(run_a).get("watchdog_retries") or 0) == 0   # never re-queued

        # 4) parked: new work must NOT be claimed
        gate.set()  # any later node would finish instantly
        run_b = _start_run()
        time.sleep(3.5)  # several claim-loop safety polls
        assert _job_of(run_b)["status"] == "queued", "parked worker claimed a job!"

        # 5) back ON → resumes and drains the queue
        worker_control.enable_worker(_HOST, "cpu")
        _wait(lambda: _job_of(run_b)["status"] == "completed", 20,
              "resumed worker to run job B")
        assert not worker.drain_requested   # boundary cleared the flag
    finally:
        worker.stop()
        t.join(timeout=10)
