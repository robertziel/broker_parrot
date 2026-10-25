"""``gpu_slots`` — how much of the box's GPU budget one node-job occupies (migration 0030).

The box-slot arbiter (test_gpu_box_slot_gate.py) counts every running row as 1 against the
worker's ``max_running``. A row that legitimately owns the whole server — a gateway's lease on
the box's vLLM — therefore leaves PAR-1 slots open, and the fleet co-schedules its own
``llm_chat`` work into the very server the lease was taken to hold (fleet jobs running on a leased box while the gateway's own
turns queue behind them).

``gpu_slots`` is the row's WEIGHT, the exact shape ``cpu_slots`` (0024) gave the cpu lane:
NULL = 1, enforced at the claim, ``>= 1`` held at the API edge, an over-wide row measured as the
whole box rather than excluded forever. ``all`` and ``half`` resolve against the claiming box's
own ``max_running`` — which is already a parameter of the claim statement, so they cost no
lookup. Every pre-0030 row weighs 1 and is claimed exactly as before.
"""
from __future__ import annotations

import pytest

from queue_workflows import node_queue
from tests._helpers import make_run


def _enq(run, node="n", **kw):
    return node_queue.enqueue_node_job(run_id=run, node_id=node, node_module="x", queue="gpu",
                                       priority=100, **kw)


def _claim(host="box-a-gpu", par=2):
    return node_queue.claim_next_gpu_job(0, host=host, max_running=par, kind_caps={})


# ── the API edge ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("given,stored", [
    (None, None), (1, None), ("1", None),        # 1 IS the default — stored as NULL, like cpu_slots
    (2, "2"), (3, "3"), ("4", "4"),
    ("half", "half"), ("all", "all"), ("ALL", "all"), (" Half ", "half"),
])
def test_the_weight_is_stored_as_declared_with_1_as_null(given, stored):
    run = make_run()
    jid = _enq(run, gpu_slots=given)
    assert node_queue.get_node_job(jid)["gpu_slots"] == stored


@pytest.mark.parametrize("bad", [0, -1, "0", "-2", "foo", "", 1.5, "1.5", "both"])
def test_a_nonsense_weight_is_refused_at_the_edge(bad):
    run = make_run()
    with pytest.raises(ValueError):
        _enq(run, gpu_slots=bad)


def test_a_weight_on_a_non_gpu_row_is_a_schema_mistake():
    """Only gpu rows pass the box-slot arbiter, so a weight elsewhere is a mistake — and
    silently dropping it would hide the mistake (the context_length rule, same reason)."""
    run = make_run()
    with pytest.raises(ValueError):
        node_queue.enqueue_node_job(run_id=run, node_id="c", node_module="x", queue="cpu",
                                    priority=100, gpu_slots="all")


# ── the arbiter ───────────────────────────────────────────────────────────────

def test_all_holds_the_whole_box():
    run = make_run()
    _enq(run, "lease", gpu_slots="all")
    _enq(run, "other")
    first = _claim(par=2)
    assert first is not None and first["node_id"] == "lease"
    assert _claim(par=2) is None, "an 'all' row weighs the whole budget: nothing else fits"


def test_all_waits_behind_ordinary_work_and_never_strands():
    """The KILL line. 1 running + an 'all' candidate = PAR+1 > PAR, so it waits — and the
    moment the box empties it claims. Busy is not excluded."""
    run = make_run()
    _enq(run, "ordinary")
    _enq(run, "lease", gpu_slots="all")
    first = _claim(par=2)
    assert first["node_id"] == "ordinary"
    assert _claim(par=2) is None, "the 'all' row does not fit beside running work"
    node_queue.mark_completed(first["id"], context_delta={}, seconds=1.0)
    got = _claim(par=2)
    assert got is not None and got["node_id"] == "lease"


def test_half_on_a_three_wide_box_weighs_two():
    run = make_run()
    _enq(run, "half", gpu_slots="half")
    _enq(run, "a")
    _enq(run, "b")
    assert _claim(par=3)["node_id"] == "half"       # (3+1)/2 = 2 of 3
    assert _claim(par=3) is not None                # 2 + 1 = 3, fits
    assert _claim(par=3) is None                    # 3 + 1 > 3


def test_an_integer_wider_than_the_box_is_measured_as_the_box():
    """LEAST(weight, max_running), exactly as cpu_slots does — a 9-wide row on a 2-wide box
    simply runs alone rather than waiting for a fit that can never arrive."""
    run = make_run()
    _enq(run, "wide", gpu_slots=9)
    _enq(run, "other")
    assert _claim(par=2)["node_id"] == "wide"
    assert _claim(par=2) is None


def test_ordinary_rows_are_unchanged():
    run = make_run()
    for i in range(3):
        _enq(run, f"n{i}")
    assert _claim(par=2) is not None
    assert _claim(par=2) is not None
    assert _claim(par=2) is None


def test_max_running_none_keeps_the_old_unbounded_behaviour():
    """Byte-compat: a caller that passes no budget never had an arbiter, and still has none."""
    run = make_run()
    _enq(run, "lease", gpu_slots="all")
    _enq(run, "other")
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", kind_caps={}) is not None
    assert node_queue.claim_next_gpu_job(0, host="box-a-gpu", kind_caps={}) is not None


def test_a_full_peer_does_not_block_another_box():
    run = make_run()
    _enq(run, "lease", gpu_slots="all")
    _enq(run, "other")
    assert _claim("box-a-gpu")["node_id"] == "lease"
    assert _claim("box-b-gpu") is not None, "box-a is full; box-b takes the spilled row"


# ── run next: a flagged wide row RESERVES the box ─────────────────────────────
#
# The weight is a FILTER, not a reservation. While anything runs, an 'all' row does not fit
# and a weight-1 row does, so every slot that frees is refilled by narrower work before the
# box is ever empty — and the flagged row, sorted FIRST, is filtered out of every one of those
# claims (the starvation claim_next_cpu_job's docstring names for cpu_slots). Without a
# reservation, a gateway lease (gpu_slots='all', is_priority) queued for a warm vLLM box waits
# while that box takes fleet jobs into every slot it frees, until the gateway gives up. A RUN-NEXT row that weighs more than one slot therefore reserves: while it
# is claimable here in every respect but the rows already running, nothing narrower is
# admitted, the box drains, and the flagged row takes it.

def _flag(job_id: str) -> str:
    assert node_queue.prioritize_node_job(job_id) is not None
    return job_id


def _lease(run, **kw) -> str:
    """A gateway lease as the gateway writes one: band 0, the whole box, flagged run-next."""
    return _flag(node_queue.enqueue_node_job(
        run_id=run, node_id="lease", node_module="system.vllm_lease", queue="gpu",
        priority=0, gpu_slots="all", **kw))


def test_a_run_next_wide_row_is_not_starved_by_refills():
    """THE KILL LINE. Two rows run on a three-wide box; a flagged 'all' row queues behind them
    and more ordinary work arrives. None of it is admitted, the box drains, and the flagged row
    takes the whole box."""
    run = make_run()
    for i in range(2):
        _enq(run, f"busy{i}")
    running = [_claim(par=3), _claim(par=3)]
    assert all(r is not None for r in running)
    lease = _lease(run)
    for i in range(3):
        _enq(run, f"refill{i}")
    assert _claim(par=3) is None, "a refill took the slot the flagged row is waiting for"
    node_queue.mark_completed(running[0]["id"], context_delta={}, seconds=1.0)
    assert _claim(par=3) is None, "one row still runs: the box is not empty yet"
    node_queue.mark_completed(running[1]["id"], context_delta={}, seconds=1.0)
    got = _claim(par=3)
    assert got is not None and got["id"] == lease
    assert _claim(par=3) is None, "and while it holds the box nothing else fits"


def test_the_reservation_lifts_the_moment_the_wide_row_lands_elsewhere():
    """Two boxes may both be eligible. Each stops admitting narrower work; the one that is
    empty claims the flagged row; the other resumes on its very next claim."""
    run = make_run()
    _enq(run, "busy")
    assert _claim("box-a-gpu", par=3) is not None
    lease = _lease(run)
    _enq(run, "ordinary")
    assert _claim("box-a-gpu", par=3) is None, "box-a refilled instead of reserving"
    got = _claim("box-b-gpu", par=3)
    assert got is not None and got["id"] == lease
    nxt = _claim("box-a-gpu", par=3)
    assert nxt is not None and nxt["node_id"] == "ordinary"


def test_the_gateway_lease_shape_reserves_a_warm_vllm_box():
    """The production shape, end to end: the pool lane's own arguments (serving kind, served
    window, per-kind caps, physical box) against a lease written exactly as the gateway writes it
    (server_types=['vllm'], context_length=262144). A fleet row arriving while the box drains
    is not admitted beside it."""
    run = make_run()
    kw = dict(host="box-c-gpu", max_running=3, box="box-c", require_model=False,
              served_types=["vllm"], serving_context=262144,
              kind_caps={"ollama": 1, "vllm": 3})
    _enq(run, "fleet0", server_types=["vllm"])
    running = node_queue.claim_next_gpu_job(0, **kw)
    assert running is not None and running["resolved_server"] == "vllm"
    lease = _lease(run, server_types=["vllm"], context_length=262144)
    _enq(run, "fleet1", server_types=["vllm"])
    assert node_queue.claim_next_gpu_job(0, **kw) is None, "fleet1 admitted beside a waiting lease"
    node_queue.mark_completed(running["id"], context_delta={}, seconds=1.0)
    got = node_queue.claim_next_gpu_job(0, **kw)
    assert got is not None and got["id"] == lease


def test_an_unflagged_wide_row_reserves_nothing():
    """The reservation belongs to the RUN-NEXT flag, not to the weight: an unflagged 'all' row
    keeps exactly the 0030 behaviour — it waits for an empty box while ordinary work is still
    admitted beside the rows that run."""
    run = make_run()
    _enq(run, "busy")
    assert _claim(par=3) is not None
    _enq(run, "wide", gpu_slots="all")
    _enq(run, "ordinary")
    got = _claim(par=3)
    assert got is not None and got["node_id"] == "ordinary"


@pytest.mark.parametrize("shape", [
    "pinned_elsewhere", "another_server", "window_too_big", "run_cancelled", "another_project",
])
def test_a_reservation_counts_only_a_row_this_box_could_claim(shape):
    """THE OVER-REACH GUARD. A reservation for a row this box can never take would stop the
    box for good, so it is judged by the claim's own terms — placement, the serving kind, the
    served window, the run guard, the tenant. A flagged wide row that fails any of them here
    leaves the box working."""
    run = make_run()
    kw = dict(host="box-a-gpu", max_running=3, kind_caps={}, box="box-a",
              served_types=["ollama"], serving_context=8192)
    _enq(run, "busy")
    assert node_queue.claim_next_gpu_job(0, **kw) is not None
    lease_run = make_run(status="cancelled") if shape == "run_cancelled" else run
    _lease(lease_run, **{
        "pinned_elsewhere": dict(force_box=["box-b"]),
        "another_server": dict(server_types=["vllm"]),
        "window_too_big": dict(context_length=262144),
        "run_cancelled": {},
        "another_project": dict(project="someone-else"),
    }[shape])
    _enq(run, "ordinary")
    got = node_queue.claim_next_gpu_job(0, **kw)
    assert got is not None and got["node_id"] == "ordinary"
