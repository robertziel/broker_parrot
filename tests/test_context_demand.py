"""Which box grows, and when — the demand read behind a context switch.

Restarting a box's LLM server costs the box its whole out-of-service window, so "a job
needs a bigger window" is not on its own a reason to do it. Three conditions are:

  * NOBODY is serving that window right now (a box that can already take the job will,
    and its claim is seconds away — growing a second box for it is pure loss);
  * the job has been waiting long enough that this is not just heartbeat lag;
  * and of the boxes that COULD grow into it, this one is the right one — the tightest
    fit, so the box with the most headroom stays free for the job that needs it.

Every box runs the same read against the same heartbeats, so they elect the same one
without talking to each other. A box that is parked, stale or beyond its own ceiling
is not a candidate — and must not mask the demand either, which is the failure that
would leave a sized job queued for ever while a big box sat switched off.
"""
from __future__ import annotations

import uuid

from queue_workflows import node_queue, run_store, worker_control
from queue_workflows.db import connection


def _box(host: str, *, serving: int | None, ceiling: int | None,
         project: str | None = None) -> None:
    node_queue.upsert_worker_heartbeat(
        host_label=host, queue="gpu", concurrency=1,
        llm_context_length=serving, llm_context_max=ceiling, project=project,
    )


def _job(context_length: int | None, *, project: str | None = None,
         avoid_box=None, force_box=None, age_s: float = 600.0,
         run_status: str = "running") -> str:
    run_id = str(uuid.uuid4())
    run_store.insert_run(run_id=run_id, workflow_name="_ctx_demand", out_dir=None,
                         status=run_status, mode="node", project=project)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id=f"n-{uuid.uuid4().hex[:8]}", node_module="llm_node",
        queue="gpu", context_length=context_length, project=project,
        avoid_box=avoid_box, force_box=force_box,
    )
    if age_s:
        with connection() as c, c.cursor() as cur:
            cur.execute("UPDATE workflow_node_jobs SET created_at = now() - "
                        "make_interval(secs => %(age)s) WHERE id = %(id)s",
                        {"age": float(age_s), "id": job_id})
    return job_id


def _need(host: str = "box-a-gpu", *, serving: int | None = 32768,
          ceiling: int | None = 262144, **kw):
    return node_queue.context_switch_need(
        host=host, serving=serving, ceiling=ceiling, **kw)


# ── the basic question ───────────────────────────────────────────────────────


def test_a_waiting_job_nobody_can_serve_is_what_this_box_should_grow_to():
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _job(131072)
    assert _need() == 131072


def test_the_highest_need_wins_so_one_restart_covers_the_others():
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _job(65536)
    _job(131072)
    assert _need() == 131072


def test_a_window_this_box_already_serves_asks_for_nothing():
    _box("box-a-gpu", serving=131072, ceiling=262144)
    _job(131072)
    assert _need(serving=131072) is None


def test_a_need_beyond_this_boxs_own_ceiling_is_not_its_problem():
    """That job belongs to a bigger box, or to the unassignable tag."""
    _box("box-a-gpu", serving=32768, ceiling=65536)
    _job(131072)
    assert _need(ceiling=65536) is None


def test_a_box_that_cannot_state_a_ceiling_never_grows():
    _box("box-a-gpu", serving=None, ceiling=None)
    _job(131072)
    assert _need(serving=None, ceiling=None) is None


def test_no_sized_work_means_no_switch():
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _job(None)
    assert _need() is None


# ── somebody else is already serving it ──────────────────────────────────────


def test_a_live_box_already_serving_that_window_settles_it():
    """box-a would otherwise be the one elected to grow (box-b has MORE room, so it is
    not the tighter fit) — being already served is what stops it."""
    _box("box-a-gpu", serving=32768, ceiling=131072)
    _box("box-b-gpu", serving=131072, ceiling=262144)
    _job(131072)
    assert _need("box-a-gpu", ceiling=131072) is None, (
        "box-b's claim is seconds away; restarting box-a is loss")


def test_a_parked_box_serves_nothing_and_must_not_mask_the_demand():
    """The failure this prevents: an operator switches the big box OFF and every sized
    job queues for ever behind a heartbeat that still advertises its window."""
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _box("box-b-gpu", serving=131072, ceiling=131072)
    worker_control.disable_worker("box-b-gpu", "gpu")
    _job(131072)
    assert _need() == 131072


def test_a_box_that_stopped_beating_stops_counting():
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _box("box-b-gpu", serving=131072, ceiling=131072)
    with connection() as c, c.cursor() as cur:
        cur.execute("UPDATE worker_heartbeats SET last_seen = now() - "
                    "make_interval(secs => 600) WHERE host_label = 'box-b-gpu'")
    _job(131072)
    assert _need() == 131072


# ── which box grows ──────────────────────────────────────────────────────────


def test_the_tightest_fit_grows_so_the_roomiest_box_stays_free():
    """box-b can just about hold it; box-a could hold far more. Growing box-a would
    spend the one box that can serve the NEXT, bigger job."""
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _box("box-b-gpu", serving=32768, ceiling=131072)
    _job(131072)
    assert _need("box-a-gpu", ceiling=262144) is None
    assert _need("box-b-gpu", ceiling=131072) == 131072


def test_boxes_of_equal_room_pick_one_by_name_not_both():
    _box("box-a-gpu", serving=32768, ceiling=131072)
    _box("box-b-gpu", serving=32768, ceiling=131072)
    _job(131072)
    assert _need("box-a-gpu", ceiling=131072) == 131072
    assert _need("box-b-gpu", ceiling=131072) is None


def test_a_server_being_started_anyway_picks_its_window_without_an_election():
    """Nothing is being restarted for the job — the box is coming up regardless — so
    the only question is which window it should come up at (R4)."""
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _box("box-b-gpu", serving=32768, ceiling=131072)
    _job(131072)
    assert _need("box-a-gpu", ceiling=262144) is None
    assert _need("box-a-gpu", serving=None, ceiling=262144, elect=False) == 131072


def test_a_peer_too_small_to_ever_serve_it_does_not_stop_this_box():
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _box("box-small-gpu", serving=8192, ceiling=8192)
    _job(131072)
    assert _need() == 131072


# ── the rest of the queue's rules still apply ────────────────────────────────


def test_a_job_still_settling_is_left_to_a_box_that_can_already_take_it():
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _job(131072, age_s=0)
    assert _need(min_age_s=30.0) is None
    assert _need(min_age_s=0.0) == 131072


def test_a_job_that_avoids_this_box_never_makes_it_restart():
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _job(131072, avoid_box=["box-a"])
    assert _need(box="box-a") is None
    assert _need(box="box-b") == 131072


def test_a_job_pinned_elsewhere_never_makes_it_restart():
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _job(131072, force_box=["box-b"])
    assert _need(box="box-a") is None


def test_another_tenants_queue_is_not_this_ones():
    _box("box-a-gpu", serving=32768, ceiling=262144, project="alpha")
    _job(131072, project="beta")
    assert _need(project="alpha") is None
    assert _need(project="beta") == 131072


def test_a_job_whose_run_is_already_dead_justifies_nothing():
    _box("box-a-gpu", serving=32768, ceiling=262144)
    _job(131072, run_status="cancelled")
    assert _need() is None


def test_a_broken_read_asks_for_no_switch(monkeypatch):
    """Best-effort, in the conservative direction: a DB blip must never restart a
    box's LLM server."""
    def boom(*_a, **_k):
        raise RuntimeError("db gone")

    monkeypatch.setattr(node_queue, "connection", boom)
    assert _need() is None
