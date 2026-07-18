"""Per-node-job PHYSICAL-BOX placement — ``avoid_box`` / ``force_box`` (migration 0020).

A queued node-job can pin or exclude the box(es) that may execute it, by BOX NAME (the
value a worker agrees on via ``gpu_model_lease.default_box_id()`` /
``QUEUE_WORKFLOWS_GPU_BOX_ID`` — not the per-project host_label). The claim SQL enforces
it: a box claims a row iff it is NOT in ``avoid_box`` AND (when ``force_box`` is set) IS
in ``force_box``. Both NULL ⇒ unconstrained (every box eligible), so all existing rows +
consumers are byte-identical.

These pin: the round-trip, the two-directional filter on the real claim path (cpu AND
gpu), empty-list normalisation, the combined case, a box-unaware worker, and the
dispatcher reading the flags off a node spec.
"""
from __future__ import annotations

from queue_workflows import dispatcher, node_queue
from tests._helpers import make_run


def _queue(node_id: str, *, queue: str = "gpu", avoid=None, force=None,
           required_model=None) -> str:
    run_id = make_run()
    return node_queue.enqueue_node_job(
        run_id=run_id, node_id=node_id, node_module="x", queue=queue,
        required_model=required_model, avoid_box=avoid, force_box=force,
    )


def _claim(box: str | None, *, queue: str = "gpu") -> dict | None:
    if queue == "gpu":
        return node_queue.claim_next_gpu_job(0, None, host=f"{box}-gpu", box=box)
    return node_queue.claim_next_cpu_job(0, host=f"{box}-cpu", box=box)


# ── round-trip ───────────────────────────────────────────────────────────────


def test_flags_round_trip_onto_the_row():
    jid = _queue("n", avoid=["box-a"], force=["box-b", "box-c"])
    row = node_queue.get_node_job(jid)
    assert sorted(row["avoid_box"]) == ["box-a"]
    assert sorted(row["force_box"]) == ["box-b", "box-c"]


def test_unconstrained_by_default_is_null():
    row = node_queue.get_node_job(_queue("n"))
    assert row["avoid_box"] is None and row["force_box"] is None


def test_empty_lists_normalise_to_null_not_an_impossible_row():
    # [] must mean "no constraint", NEVER "no box allowed" (which would strand it).
    row = node_queue.get_node_job(_queue("n", avoid=[], force=[]))
    assert row["avoid_box"] is None and row["force_box"] is None


# ── avoid_box ────────────────────────────────────────────────────────────────


def test_avoid_box_blocks_the_named_box_only():
    jid = _queue("n", avoid=["box-a"])
    assert _claim("box-a") is None                 # excluded
    got = _claim("box-b")
    assert got is not None and got["id"] == jid       # a peer takes it


def test_avoid_box_with_several_names():
    _queue("n", avoid=["box-a", "box-b"])
    assert _claim("box-a") is None
    assert _claim("box-b") is None
    assert _claim("box-c") is not None               # the only one left


# ── force_box ────────────────────────────────────────────────────────────────


def test_force_box_pins_to_the_named_box_only():
    jid = _queue("n", force=["box-c"])
    assert _claim("box-a") is None
    assert _claim("box-b") is None
    got = _claim("box-c")
    assert got is not None and got["id"] == jid


# ── combined ─────────────────────────────────────────────────────────────────


def test_force_and_avoid_intersect():
    # allowed = force ∩ not-avoid → only box-b may run it
    _queue("n", force=["box-b", "box-c"], avoid=["box-c"])
    assert _claim("box-c") is None                   # forced but also avoided
    assert _claim("box-a") is None                  # not forced
    assert _claim("box-b") is not None


# ── both lanes ───────────────────────────────────────────────────────────────


def test_cpu_lane_honours_placement_too():
    jid = _queue("n", queue="cpu", force=["box-c"])
    assert _claim("box-a", queue="cpu") is None
    got = _claim("box-c", queue="cpu")
    assert got is not None and got["id"] == jid


# ── a box-unaware worker skips constrained jobs, still claims free ones ───────


def test_box_unaware_worker_skips_constrained_but_claims_unconstrained():
    constrained = _queue("c", force=["box-c"])
    free = _queue("f")
    got = _claim(None)                                # worker didn't resolve a box
    assert got is not None and got["id"] == free      # only the unconstrained one
    assert node_queue.get_node_job(constrained)["status"] == "queued"


# ── dispatcher threads the flags off the node spec ───────────────────────────


def test_dispatcher_reads_avoid_and_force_off_the_node_spec(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        node_queue, "enqueue_node_job",
        lambda **kw: seen.update(kw) or "jid",
    )
    run_id = make_run()
    run = {"id": run_id, "priority": 100, "project": ""}
    node = {"id": "render", "gpu": True, "model": "sdxl",
            "avoid_box": ["box-a"], "force_box": ["box-c"]}
    dispatcher._enqueue(run_id, node, run)
    assert seen["avoid_box"] == ["box-a"] and seen["force_box"] == ["box-c"]


# ── _nodes_of carries the flags from the pipeline schema node ─────────────────


def test_nodes_of_threads_avoid_and_force_from_the_schema(monkeypatch):
    # The bug this pins: _nodes_of builds each node dict from a WHITELIST of
    # schema keys, so placement injected by a schema provider (e.g. a host's
    # default avoid_box) was silently dropped before _enqueue's
    # node.get("avoid_box") — every node_job got NULL despite the schema.
    schema = {
        "nodes": [
            {"id": "prep", "gpu": False, "inputs": []},
            {"id": "render", "gpu": True, "model": "sdxl", "depends_on": ["prep"],
             "inputs": [], "avoid_box": ["box-a", "box-b"], "force_box": ["box-c"]},
        ],
    }
    monkeypatch.setattr(dispatcher, "_pipeline_schema", lambda name: schema)
    workflow = {
        "name": "wf", "mode": "node",
        "steps": [{"id": "s", "kind": "pipeline", "pipeline": "p"}],
    }
    nodes = dispatcher._nodes_of(workflow)
    render = next(n for n in nodes if n["id"] == "s/render")
    assert render["avoid_box"] == ["box-a", "box-b"]
    assert render["force_box"] == ["box-c"]
    prep = next(n for n in nodes if n["id"] == "s/prep")
    assert prep["avoid_box"] is None and prep["force_box"] is None
