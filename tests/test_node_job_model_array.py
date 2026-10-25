"""A node-job may name an ORDERED ARRAY of acceptable models — and the scalar still works.

Both shapes must survive: `required_model` is what every pre-0026 row carries and the
single-model case is the common one. These tests pin the two properties that make the
rollout safe:

  * DUAL-WRITE — an array job also sets the scalar to the array's head, so a worker that
    has not been bumped yet claims it correctly instead of ignoring it forever.
  * The capability gate spans BOTH — a box that can serve only the array's SECOND entry
    must still claim, which is the entire point of the array and is exactly what a
    scalar-only gate would silently lose.
"""
from __future__ import annotations

import pytest

from queue_workflows import node_queue
from tests._helpers import make_run


def _row(job_id: str) -> dict:
    from queue_workflows.db import connection
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM workflow_node_jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def _arr(value):
    """Array columns come back as a python list on pg and as JSON TEXT on sqlite.
    These tests assert BEHAVIOUR, not storage, so they must not care which backend ran —
    an assertion that only holds on one dialect is a test that silently covers half the
    fleet."""
    import json
    if value is None or isinstance(value, list):
        return value
    return json.loads(value)


def test_an_array_job_dual_writes_the_scalar_head_for_un_bumped_workers():
    run = make_run()
    job = node_queue.enqueue_node_job(
        run_id=run, node_id="n1", node_module="llm_chat", queue="gpu",
        required_models=[{"model": "chat-expert-a", "server_types": ["vllm"]},
                         {"model": "chat-model-a:latest", "server_types": ["ollama"]}],
    )
    row = _row(job)
    assert row["required_model"] == "chat-expert-a", "the scalar must be the array's head"
    assert _arr(row["required_model_ids"]) == ["chat-expert-a", "chat-model-a:latest"]


def test_a_scalar_job_writes_no_array_and_is_unchanged():
    run = make_run()
    job = node_queue.enqueue_node_job(
        run_id=run, node_id="n1", node_module="llm_chat", queue="gpu",
        required_model="chat-model-a:latest",
    )
    row = _row(job)
    assert row["required_model"] == "chat-model-a:latest"
    assert row["required_models"] is None
    assert row["required_model_ids"] is None


def test_a_box_serving_only_the_SECOND_entry_can_still_claim_it():
    """The whole point of the array. A scalar-only capability gate loses this."""
    run = make_run()
    node_queue.enqueue_node_job(
        run_id=run, node_id="n1", node_module="llm_chat", queue="gpu",
        required_models=[{"model": "chat-expert-a", "server_types": ["vllm"]},
                         {"model": "chat-model-a:latest", "server_types": ["ollama"]}],
    )
    claimed = node_queue.claim_next_gpu_job(
        host="box-b", known_models=["chat-model-a:latest"],   # does NOT have chat-expert-a
    )
    assert claimed is not None, "a box serving the second acceptable model must claim"
    assert claimed["node_id"] == "n1"


def test_a_box_serving_NONE_of_the_acceptable_models_does_not_claim():
    run = make_run()
    node_queue.enqueue_node_job(
        run_id=run, node_id="n1", node_module="llm_chat", queue="gpu",
        required_models=[{"model": "chat-expert-a", "server_types": ["vllm"]}],
    )
    assert node_queue.claim_next_gpu_job(host="box-c", known_models=["something-else"]) is None


def test_malformed_entries_never_reach_the_row():
    run = make_run()
    job = node_queue.enqueue_node_job(
        run_id=run, node_id="n1", node_module="llm_chat", queue="gpu",
        required_models=[{"server_types": ["vllm"]},          # no model
                         {"model": "  ", "server_types": []},  # blank
                         {"model": "chat-model-a:latest"}],
    )
    row = _row(job)
    assert _arr(row["required_model_ids"]) == ["chat-model-a:latest"]
    assert row["required_model"] == "chat-model-a:latest"


def test_the_dispatcher_actually_threads_required_models_from_the_SCHEMA():
    """The WIRING test, asserted on the row rather than on source text.

    A `required_models` a schema declares but the dispatcher never reads is a DARK SEAM:
    it validates, it documents, and it never reaches a row. The failure is invisible —
    every job silently falls back to the scalar path and nothing says why."""
    from queue_workflows import dispatcher

    run = make_run()
    # `gpu: true` + a model is what the schema bridge stamps for an LLM node; a cpu row
    # carrying a required_model violates the table's own CHECK.
    node = {"id": "n-wired", "node": "llm_chat", "gpu": True,
            "model": "chat-expert-a", "server_types": ["vllm", "ollama"],
            "required_models": [{"model": "chat-expert-a", "server_types": ["vllm"]},
                                {"model": "chat-model-a:latest", "server_types": ["ollama"]}]}
    job_id = dispatcher._enqueue(run, node, {"priority": 100, "project": None})
    row = _row(job_id)
    assert _arr(row["required_model_ids"]) == ["chat-expert-a", "chat-model-a:latest"], (
        "the schema's required_models never reached the row — dark seam")
    assert row["required_model"] == "chat-expert-a"


def test_required_models_survives_the_SCHEMA_to_NODE_bridge():
    """The seam a first wiring test missed by building the node dict BY HAND.

    `_nodes_of` is an explicit per-field copy, not a passthrough: a schema key absent from
    that list is dropped silently. The failure mode: llm_chat declares `required_models`, the
    orchestrator loads the schema WITH it, and every enqueued row still has NULL. Testing
    `_enqueue` with a hand-built node proved only that `_enqueue` works; it could not see
    the bridge that feeds it.
    """
    from queue_workflows import dispatcher

    schema = {"name": "p", "nodes": [{
        "id": "n", "node": "llm_chat", "queue": "gpu", "gpu": True,
        "model": "chat-expert-a", "server_types": ["vllm"],
        "required_models": [{"model": "chat-expert-a", "server_types": ["vllm"]},
                            {"model": "chat-model-a:latest", "server_types": ["ollama"]}],
    }]}
    wf = {"name": "w", "steps": [{"id": "s", "kind": "pipeline", "pipeline": "p"}]}

    import queue_workflows
    queue_workflows.set_workflow_provider(lambda _n: wf, lambda _n: schema)
    try:
        nodes = dispatcher._nodes_of(wf)
    finally:
        pass
    node = next(n for n in nodes if n.get("kind") == "node")
    assert node.get("required_models"), (
        "required_models was dropped by the schema→node bridge — the field never reaches "
        "_enqueue, so every row lands with NULL while every unit test passes")
    assert [e["model"] for e in node["required_models"]] == ["chat-expert-a", "chat-model-a:latest"]
