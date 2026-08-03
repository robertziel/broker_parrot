"""INVARIANT — ``dispatcher.start_run`` owns the queued→running flip.

A host may expand a run itself at trigger time (``insert_run`` + ``start_run``
inside an ingest task); the orchestrator's dispatch loop then finds nothing NEW
to enqueue for that run. The flip must therefore ride EVERY expansion path,
keyed on "node-jobs exist for the run", never on "THIS caller enqueued new
rows": before this invariant, trigger-expanded runs sat ``'queued'`` forever —
still claimable (the claim guard only excludes cancelled/failed) yet invisible
to every consumer keyed on ``status='running'``, which disarmed the VLM-pool
anti-starvation break and deferred a full render queue into a standstill.
"""
from __future__ import annotations

import queue_workflows
from queue_workflows import dispatcher, node_queue, run_store
from tests._helpers import make_run

_WF = {
    "name": "_flip_wf", "mode": "node",
    "steps": [{"id": "a", "kind": "input", "widget": "confirm"}],
}


def _queued_run() -> str:
    queue_workflows.set_workflow_provider(lambda name: _WF, lambda name: {})
    return make_run(status="queued", workflow_name="_flip_wf")


def test_start_run_flips_a_freshly_expanded_run_to_running():
    """The trigger path: insert_run + start_run with NO orchestrator involved
    must leave the run 'running', not 'queued'."""
    run_id = _queued_run()
    n = dispatcher.start_run(run_id)
    assert n == 1
    assert run_store.get_run(run_id)["status"] == "running"


def test_start_run_flips_an_ALREADY_expanded_stuck_run():
    """The self-heal path: a run whose jobs already exist (a pre-fix stuck row,
    or the dispatch loop re-visiting a trigger-expanded run) enqueues nothing
    new — the flip must fire anyway, keyed on existing jobs."""
    run_id = _queued_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="__input__confirm", queue="cpu",
    )
    n = dispatcher.start_run(run_id)
    assert n == 0                     # nothing new — idempotent expansion
    assert run_store.get_run(run_id)["status"] == "running"


def test_start_run_is_idempotent_on_the_flip():
    run_id = _queued_run()
    dispatcher.start_run(run_id)
    assert dispatcher.start_run(run_id) == 0
    assert run_store.get_run(run_id)["status"] == "running"


def test_an_empty_expansion_leaves_the_run_queued():
    """No jobs enqueued AND none existing ⇒ no flip — a run with nothing to do
    stays 'queued' for the phantom-run reconciler to own, exactly as today."""
    queue_workflows.set_workflow_provider(
        lambda name: {"name": "_empty_wf", "mode": "node", "steps": []},
        lambda name: {},
    )
    run_id = make_run(status="queued", workflow_name="_empty_wf")
    assert dispatcher.start_run(run_id) == 0
    assert run_store.get_run(run_id)["status"] == "queued"


def test_a_terminal_run_is_never_revived_by_the_flip():
    """The guard is ``status='queued'`` — a cancelled/completed run with
    leftover job rows must not be flipped back to life."""
    run_id = _queued_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="__input__confirm", queue="cpu",
    )
    run_store.update_run(run_id, status="cancelled")
    run_store.mark_run_running_if_expanded(run_id)
    assert run_store.get_run(run_id)["status"] == "cancelled"
