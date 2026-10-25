"""Resume is for ORPHANS. A run whose work is still queued was never orphaned.

The startup hook flips every ``running`` run of the project back
to ``queued`` so a worker death cannot strand it. But it asked only about the RUN, never
about its work — so it also re-queued runs whose node-jobs were sitting right there,
queued and claimable. Two consequences, both observed:

  * A run created by an external client — broker_parrot's own queue is a public
    interface, and a load balancer inserts a ``running`` run plus one gpu job for
    its lease, deliberately ``running`` so the dispatcher never tries to expand a
    workflow name it does not have — was flipped to ``queued`` by an orchestrator
    restart. The dispatcher then tried to expand it every tick: ``KeyError: no workflow
    named 'lb_lease'``, an ERROR and a traceback twice a second, for ever.
  * For a run the host CAN load, the re-expansion is worse than noise: the DAG's initial
    nodes are enqueued a second time alongside the ones already queued.

The fix is the hook's own docstring, enforced: "a row is running here only because a
worker died mid-execution without marking its node terminal". A live node-job is proof
that did not happen — the lease reclaim and the dispatch outbox are already driving it.
"""
from __future__ import annotations

import uuid

from queue_workflows import node_queue, run_store
from queue_workflows.db import connection
from tests._helpers import make_run


def _job(run_id: str, *, status: str = "queued") -> str:
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id=f"n-{uuid.uuid4().hex[:8]}", node_module="m",
        queue="cpu",
    )
    if status != "queued":
        with connection() as c, c.cursor() as cur:
            cur.execute("UPDATE workflow_node_jobs SET status = %s WHERE id = %s",
                        (status, job_id))
    return job_id


def test_an_orphaned_run_is_still_resumed():
    """The case the hook exists for: the worker died and left nothing behind."""
    run_id = make_run(status="running")
    assert run_store.reenqueue_running_for_resume() >= 1
    assert run_store.get_run(run_id)["status"] == "queued"


def test_a_run_whose_work_is_still_queued_is_left_alone():
    """Nothing is orphaned — a worker will claim that job. Re-queueing the run makes the
    dispatcher expand it a SECOND time, on top of the job already waiting."""
    run_id = make_run(status="running")
    _job(run_id)
    run_store.reenqueue_running_for_resume()
    run = run_store.get_run(run_id)
    assert run["status"] == "running"
    assert run["resume_count"] == 0, "it was never resumed, so nothing to count"


def test_a_run_whose_work_is_running_is_left_alone():
    run_id = make_run(status="running")
    _job(run_id, status="running")
    run_store.reenqueue_running_for_resume()
    assert run_store.get_run(run_id)["status"] == "running"


def test_a_run_parked_on_an_input_is_left_alone():
    """``awaiting_input`` is live work too — it is waiting for a person, not for a
    worker that died."""
    run_id = make_run(status="running")
    _job(run_id, status="awaiting_input")
    run_store.reenqueue_running_for_resume()
    assert run_store.get_run(run_id)["status"] == "running"


def test_a_run_whose_every_job_is_terminal_IS_resumed():
    """The subtle orphan: the jobs finished but nothing advanced the run (the dispatch
    event never landed). That is exactly what resume is for."""
    run_id = make_run(status="running")
    _job(run_id, status="completed")
    _job(run_id, status="failed")
    assert run_store.reenqueue_running_for_resume() >= 1
    assert run_store.get_run(run_id)["status"] == "queued"


def test_a_foreign_run_the_host_cannot_expand_stays_out_of_the_dispatchers_way():
    """The live case, end to end: an external client owns the run AND its job, and names
    a workflow this host has never heard of. Leaving it `running` is what keeps the
    dispatcher from trying to expand it twice a second for ever."""
    run_id = str(uuid.uuid4())
    run_store.insert_run(run_id=run_id, workflow_name="lb_lease", out_dir=None,
                         status="running", mode="node")
    _job(run_id)
    run_store.reenqueue_running_for_resume()
    assert run_store.get_run(run_id)["status"] == "running"
