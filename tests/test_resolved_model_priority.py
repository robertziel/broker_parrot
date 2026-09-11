"""The ORDER in `required_models` is a PRIORITY, and the claim must honour it.

"Highest in order has highest priority of use if a slot is available" — so a box that can
serve BOTH the first and the second acceptable model must take the FIRST, and only fall
through when it genuinely cannot serve the earlier ones. Without this the array degrades
into an unordered set: the row still claims, so nothing looks broken, and the fleet quietly
runs the fallback model while the preferred one sits loaded and idle.
"""
from __future__ import annotations

from queue_workflows import node_queue
from tests._helpers import make_run


def _row(job_id: str) -> dict:
    from queue_workflows.db import connection
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM workflow_node_jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def _enqueue(models):
    run = make_run()
    return node_queue.enqueue_node_job(
        run_id=run, node_id="n1", node_module="llm_chat", queue="gpu",
        required_models=models)


def test_a_box_serving_BOTH_takes_the_FIRST_in_order():
    _enqueue([{"model": "chat-expert-a", "server_types": ["ollama"]},
              {"model": "chat-model-a:latest", "server_types": ["ollama"]}])
    claimed = node_queue.claim_next_gpu_job(
        host="box-a", known_models=["chat-model-a:latest", "chat-expert-a"],
        served_types=["ollama"])
    assert claimed is not None
    assert _row(claimed["id"])["resolved_model"] == "chat-expert-a", (
        "the array's order is a PRIORITY; a box able to serve the first must take it")


def test_a_box_serving_only_the_SECOND_falls_through_to_it():
    _enqueue([{"model": "chat-expert-a", "server_types": ["ollama"]},
              {"model": "chat-model-a:latest", "server_types": ["ollama"]}])
    claimed = node_queue.claim_next_gpu_job(
        host="box-b", known_models=["chat-model-a:latest"], served_types=["ollama"])
    assert claimed is not None
    assert _row(claimed["id"])["resolved_model"] == "chat-model-a:latest"


def test_a_single_model_row_stamps_that_model():
    """The scalar path keeps working and is stamped the same way, so one column answers
    'what actually ran' for both shapes."""
    run = make_run()
    node_queue.enqueue_node_job(run_id=run, node_id="n1", node_module="llm_chat",
                                queue="gpu", required_model="chat-model-a:latest")
    claimed = node_queue.claim_next_gpu_job(
        host="box-c", known_models=["chat-model-a:latest"], served_types=["ollama"])
    assert _row(claimed["id"])["resolved_model"] == "chat-model-a:latest"


def test_an_unconstrained_row_stamps_no_model():
    run = make_run()
    node_queue.enqueue_node_job(run_id=run, node_id="n1", node_module="comfyui_video",
                                queue="gpu")
    claimed = node_queue.claim_next_gpu_job(host="box-d", known_models=["x"])
    assert claimed is not None
    assert _row(claimed["id"])["resolved_model"] is None
