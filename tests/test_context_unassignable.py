"""The tag: a sized job that NO live box can serve says so, and stops saying so when one can.

The rule: a node_job whose context matches no box gets a tag, and the tagging always
checks against available RAM whether the context can be increased. So the verdict is taken against each box's ``llm_context_max`` — the
highest window it could SWITCH to, which the worker recomputes from its free memory on
every heartbeat — not against the window it happens to serve right now. A box that can
grow into the job keeps it untagged (it will switch); only a job beyond every reachable
window is tagged. The tag is the existing unassignable stamp the panel already shows,
cleared the moment a box can reach the job (memory freed, a bigger box came up) or the
job leaves the queue.

Model and window are judged ON THE SAME BOX: a job is assignable only if one live box
both holds its model and reaches its window. Judged separately, one box having the model
and another the window would pass a job nobody can run — and a flag the model path
clears while the context path re-sets it would emit a fresh event every sweep.
"""
from __future__ import annotations

import uuid

from queue_workflows import node_queue, run_store
from queue_workflows.db import connection


def _box(host: str, *, current: int | None, reachable: int | None,
         fits: list[str] | None = None, project: str | None = None) -> None:
    node_queue.upsert_worker_heartbeat(
        host_label=host, queue="gpu", concurrency=1,
        known_models=fits or [], fits_models=fits or [],
        llm_context_length=current, llm_context_max=reachable, project=project,
    )


def _job(context_length: int | None, *, model: str | None = None,
         project: str | None = None) -> str:
    run_id = str(uuid.uuid4())
    run_store.insert_run(run_id=run_id, workflow_name="_ctx_tag", out_dir=None,
                         status="running", mode="node", project=project)
    return node_queue.enqueue_node_job(
        run_id=run_id, node_id=f"n-{uuid.uuid4().hex[:8]}", node_module="llm_node",
        queue="gpu", required_model=model, context_length=context_length,
        project=project,
    )


def _row(job_id: str) -> dict:
    return node_queue.get_node_job(job_id)


def test_a_job_beyond_every_reachable_window_is_tagged_and_says_why():
    _box("box-a-gpu", current=32768, reachable=65536)
    job = _job(131072)

    flagged = node_queue.flag_unassignable_gpu_jobs()

    assert job in [r["id"] for r in flagged]
    reason = _row(job)["unassignable_reason"]
    assert "131072-token context" in reason
    assert "largest reachable: 65536" in reason
    assert "free memory" in reason, "the reason must say the RAM check was made"


def test_a_box_that_can_switch_up_to_the_job_keeps_it_untagged():
    """Serving 32k now does not matter — the box can reach 131k, so it will switch."""
    _box("box-a-gpu", current=32768, reachable=131072)
    job = _job(100000)
    assert node_queue.flag_unassignable_gpu_jobs() == []
    assert _row(job)["unassignable_at"] is None


def test_a_box_that_reaches_exactly_the_window_serves_it():
    _box("box-a-gpu", current=32768, reachable=131072)
    job = _job(131072)
    assert node_queue.flag_unassignable_gpu_jobs() == []
    assert _row(job)["unassignable_at"] is None


def test_when_both_halves_are_missing_the_window_is_named_first():
    """No box holds the model AND no box reaches the window: freeing memory is what the
    operator can do on the spot, and the model verdict would hide that the window is
    also out of reach."""
    _box("box-a-gpu", current=32768, reachable=65536, fits=["other-model"])
    job = _job(131072, model="chat-model-a")
    node_queue.flag_unassignable_gpu_jobs()
    assert "131072-token context" in _row(job)["unassignable_reason"]


def test_freeing_memory_lifts_the_tag_on_the_next_sweep():
    """The ceiling moves with the box's free RAM every heartbeat — so the tag does."""
    _box("box-a-gpu", current=32768, reachable=65536)
    job = _job(131072)
    node_queue.flag_unassignable_gpu_jobs()
    assert _row(job)["unassignable_at"] is not None

    _box("box-a-gpu", current=32768, reachable=131072)     # memory freed
    node_queue.flag_unassignable_gpu_jobs()
    assert _row(job)["unassignable_at"] is None


def test_a_bigger_box_coming_up_lifts_the_tag():
    _box("box-a-gpu", current=32768, reachable=65536)
    job = _job(200000)
    node_queue.flag_unassignable_gpu_jobs()
    _box("box-b-gpu", current=262144, reachable=262144)
    node_queue.flag_unassignable_gpu_jobs()
    assert _row(job)["unassignable_at"] is None


def test_a_box_that_cannot_state_its_window_cannot_serve_a_sized_job():
    _box("box-a-gpu", current=None, reachable=None)
    job = _job(8192)
    node_queue.flag_unassignable_gpu_jobs()
    assert "largest reachable: unknown" in _row(job)["unassignable_reason"]


def test_with_no_live_gpu_box_there_is_no_verdict():
    _box("box-a-gpu", current=32768, reachable=65536)
    with connection() as c, c.cursor() as cur:
        cur.execute("UPDATE worker_heartbeats SET last_seen = now() - "
                    "make_interval(secs=>600) WHERE host_label='box-a-gpu'")
    job = _job(131072)
    assert node_queue.flag_unassignable_gpu_jobs() == []
    assert _row(job)["unassignable_at"] is None


def test_the_tag_is_set_once_not_once_per_sweep():
    """The churn trap: a job whose MODEL fits box-a but whose window does not must stay
    flagged across sweeps — never cleared by the model path and re-flagged by the
    context path (a fresh 'unassignable' event every 15 s)."""
    _box("box-a-gpu", current=32768, reachable=65536, fits=["chat-model-a"])
    job = _job(131072, model="chat-model-a")
    first = node_queue.flag_unassignable_gpu_jobs()
    stamped = _row(job)["unassignable_at"]
    second = node_queue.flag_unassignable_gpu_jobs()
    assert job in [r["id"] for r in first]
    assert job not in [r["id"] for r in second]
    assert _row(job)["unassignable_at"] == stamped


def test_model_and_window_must_be_on_the_same_box():
    _box("box-a-gpu", current=32768, reachable=32768, fits=["chat-model-a"])
    _box("box-b-gpu", current=262144, reachable=262144, fits=["other-model"])
    job = _job(100000, model="chat-model-a")
    node_queue.flag_unassignable_gpu_jobs()
    reason = _row(job)["unassignable_reason"]
    assert reason is not None, "no single box can run it"
    assert "both holds model chat-model-a and reaches a 100000-token context" in reason


def test_the_verdict_names_the_model_when_the_model_is_the_problem():
    """A sized job whose window some box reaches but whose model no box holds keeps the
    pre-0028 SERVES verdict — the window is not what is missing."""
    _box("box-a-gpu", current=262144, reachable=262144, fits=["other-model"])
    job = _job(8192, model="chat-model-a")
    node_queue.flag_unassignable_gpu_jobs()
    assert "SERVES model chat-model-a" in _row(job)["unassignable_reason"]


def test_an_unsized_job_is_judged_exactly_as_before():
    _box("box-a-gpu", current=None, reachable=None, fits=["chat-model-a"])
    free = _job(None)
    modelled = _job(None, model="chat-model-a")
    assert node_queue.flag_unassignable_gpu_jobs() == []
    assert _row(free)["unassignable_at"] is None
    assert _row(modelled)["unassignable_at"] is None


def test_another_projects_big_box_does_not_mask_the_tag():
    _box("box-a-gpu", current=32768, reachable=65536, project="alpha")
    _box("box-z-gpu", current=262144, reachable=262144, project="beta")
    job = _job(131072, project="alpha")
    node_queue.flag_unassignable_gpu_jobs(project="alpha")
    assert _row(job)["unassignable_at"] is not None, (
        "beta's box can never claim alpha's job (exact-match tenancy)")


def test_the_tag_clears_when_the_job_leaves_the_queue():
    _box("box-a-gpu", current=32768, reachable=65536)
    job = _job(131072)
    node_queue.flag_unassignable_gpu_jobs()
    with connection() as c, c.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET status='cancelled' WHERE id=%s", (job,))
    node_queue.flag_unassignable_gpu_jobs()
    assert _row(job)["unassignable_at"] is None
