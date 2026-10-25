"""Migration 0028 — the context window a node-job NEEDS, and the one a box SERVES.

WHY. An LLM job's prompt has to fit the server's context window, and when it does not
the failure is SILENT: ollama truncates the prompt to fit and answers HTTP 200.
Measured on ollama 0.33.2 — a ~12k-token prompt to a runner with an 8192 context came
back 200 with ``usage.prompt_tokens`` = 4098 and no error anywhere. The model answered
a question it never fully saw.

So a node-job may now DECLARE the window it needs (``context_length``, tokens; NULL =
no requirement), each GPU box ADVERTISES the window its LLM server serves right now
(``llm_context_length``) and the highest one it could switch to within its RAM and the
model's limit (``llm_context_max``), and the claim refuses a job a box cannot serve —
before the prompt is ever sent, instead of after it was cut in half.

This module is the data plumbing: the columns, the enqueue validation, the claim gate,
the heartbeat fields, and the dispatcher carrying the schema key to the row. Choosing
and applying a bigger window is tested in test_context_switch.py; tagging a job no box
can serve, in test_context_unassignable.py.
"""
from __future__ import annotations

import sys
import types
import uuid

import pytest

import queue_workflows
from queue_workflows import dispatcher, node_queue, run_store
from queue_workflows.db import connection
from tests._helpers import make_run


def _enqueue(run_id: str, *, node_id: str = "n", queue: str = "gpu", **kw) -> str:
    return node_queue.enqueue_node_job(
        run_id=run_id, node_id=node_id, node_module="llm_node", queue=queue, **kw,
    )


# ── the column ───────────────────────────────────────────────────────────────


def test_a_declared_context_length_is_stored_on_the_row():
    job_id = _enqueue(make_run(), context_length=65536)
    assert node_queue.get_node_job(job_id)["context_length"] == 65536


def test_no_declaration_means_no_requirement():
    job_id = _enqueue(make_run())
    assert node_queue.get_node_job(job_id)["context_length"] is None


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "65536", "lots"])
def test_a_malformed_context_length_is_refused_before_the_insert(bad):
    run_id = make_run()
    with pytest.raises(ValueError, match="context_length"):
        _enqueue(run_id, context_length=bad)
    assert node_queue.list_jobs_for_run(run_id) == [], "nothing was written"


@pytest.mark.parametrize("queue", ["cpu", "download"])
def test_only_a_gpu_job_can_need_an_llm_context(queue):
    """A cpu or download row is served by no LLM: a context requirement there is a
    schema mistake, and silently ignoring it would hide the mistake."""
    with pytest.raises(ValueError, match="context_length"):
        _enqueue(make_run(), queue=queue, context_length=8192)


# ── the claim gate ───────────────────────────────────────────────────────────


def _claim(serving_context):
    return node_queue.claim_next_gpu_job(
        0, None, host="box-a-gpu", serving_context=serving_context)


def test_an_unconstrained_row_is_claimable_by_any_box():
    job_id = _enqueue(make_run())
    assert _claim(None)["id"] == job_id


@pytest.mark.parametrize(("serving", "claimable"), [
    (None, False),        # a box that cannot state its window never takes a sized job
    (16384, False),       # too small — the prompt would be truncated
    (32768, True),        # exactly enough
    (262144, True),       # more than enough
])
def test_a_sized_row_is_claimed_only_by_a_box_whose_window_covers_it(serving, claimable):
    job_id = _enqueue(make_run(), context_length=32768)
    claimed = _claim(serving)
    if claimable:
        assert claimed is not None and claimed["id"] == job_id
    else:
        assert claimed is None
        assert node_queue.get_node_job(job_id)["status"] == "queued"


def test_a_row_this_box_cannot_serve_does_not_block_the_ones_behind_it():
    run_id = make_run()
    big = _enqueue(run_id, node_id="big", context_length=131072, priority=1)
    small = _enqueue(run_id, node_id="small", context_length=8192, priority=50)
    free = _enqueue(run_id, node_id="free", priority=90)

    claimed = []
    for _ in range(3):
        job = _claim(32768)
        if job is None:
            break
        claimed.append(job["id"])
        # finish it: the per-kind width-1 cap otherwise (rightly) holds the next claim
        node_queue.mark_completed(job["id"], context_delta={}, seconds=0.1)
    assert sorted(claimed) == sorted([small, free])
    assert node_queue.get_node_job(big)["status"] == "queued", (
        "left for a box with a big enough window (or a switch)")


def test_the_claim_keeps_its_old_behaviour_for_callers_that_do_not_pass_a_window():
    """Every pre-0028 row is unconstrained, so a caller that never learned about the
    window claims exactly what it always did."""
    job_id = _enqueue(make_run())
    assert node_queue.claim_next_gpu_job(0, None, host="h")["id"] == job_id


# ── the heartbeat ────────────────────────────────────────────────────────────


def _heartbeat_row(host_label: str, queue: str = "gpu") -> dict:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT llm_context_length, llm_context_max FROM worker_heartbeats "
            "WHERE host_label = %s AND queue = %s", (host_label, queue),
        )
        return cur.fetchone()


def test_the_heartbeat_advertises_the_served_window_and_the_reachable_one():
    node_queue.upsert_worker_heartbeat(
        host_label="box-a-gpu", queue="gpu", concurrency=1,
        llm_context_length=32768, llm_context_max=131072,
    )
    assert dict(_heartbeat_row("box-a-gpu")) == {
        "llm_context_length": 32768, "llm_context_max": 131072}


def test_a_later_heartbeat_overwrites_the_window_it_no_longer_serves():
    kw = dict(host_label="box-b-gpu", queue="gpu", concurrency=1)
    node_queue.upsert_worker_heartbeat(**kw, llm_context_length=32768,
                                       llm_context_max=65536)
    node_queue.upsert_worker_heartbeat(**kw, llm_context_length=65536,
                                       llm_context_max=65536)
    assert _heartbeat_row("box-b-gpu")["llm_context_length"] == 65536


def test_a_worker_that_serves_no_llm_advertises_no_window():
    node_queue.upsert_worker_heartbeat(host_label="box-c-cpu", queue="cpu",
                                       concurrency=4)
    assert dict(_heartbeat_row("box-c-cpu", "cpu")) == {
        "llm_context_length": None, "llm_context_max": None}


def test_a_heartbeat_that_names_no_window_does_not_touch_the_0028_columns():
    """DEPLOY ORDER. A worker of this engine can start against a database the
    orchestrator has not migrated yet, and a heartbeat naming columns that do not
    exist would kill it — a whole lane flagged DEAD over a field it has no use for.
    So a lane with no window (cpu, ingest, download) leaves them alone entirely."""
    kw = dict(host_label="box-d-gpu", queue="gpu", concurrency=1)
    node_queue.upsert_worker_heartbeat(**kw, llm_context_length=32768,
                                       llm_context_max=131072)
    node_queue.upsert_worker_heartbeat(**kw)          # a caller that says nothing
    assert dict(_heartbeat_row("box-d-gpu")) == {
        "llm_context_length": 32768, "llm_context_max": 131072}


def test_a_box_that_has_lost_its_window_clears_it_rather_than_leaving_it_stale():
    """The other half: a gpu worker whose server went away publishes None, and that
    MUST overwrite. A stale window is a lie the claim would route sized jobs into."""
    kw = dict(host_label="box-e-gpu", queue="gpu", concurrency=1)
    node_queue.upsert_worker_heartbeat(**kw, llm_context_length=32768,
                                       llm_context_max=131072)
    node_queue.upsert_worker_heartbeat(**kw, llm_context_length=None,
                                       llm_context_max=None)
    assert dict(_heartbeat_row("box-e-gpu")) == {
        "llm_context_length": None, "llm_context_max": None}


def test_a_box_can_read_back_the_window_it_last_published():
    """THE RESTART HALF of the idle-box trap. A box with no lever remembers the window
    it observed — in memory, which a restart erases. Restart it while its model happens
    to be unloaded (5 min of quiet is enough) and it is windowless again until something
    warms the model. Its own last heartbeat is the record of what it saw."""
    node_queue.upsert_worker_heartbeat(
        host_label="box-restart-gpu", queue="gpu", concurrency=1,
        llm_context_length=32768, llm_context_max=65536)
    assert node_queue.last_published_context("box-restart-gpu", "gpu") == 32768


def test_the_read_back_is_this_boxs_own_and_nobody_elses():
    node_queue.upsert_worker_heartbeat(
        host_label="box-other-gpu", queue="gpu", concurrency=1,
        llm_context_length=262144, llm_context_max=262144)
    assert node_queue.last_published_context("box-quiet-gpu", "gpu") is None
    assert node_queue.last_published_context("box-other-gpu", "cpu") is None


def test_a_box_that_never_published_one_reads_back_nothing():
    node_queue.upsert_worker_heartbeat(host_label="box-blank-gpu", queue="gpu",
                                       concurrency=1)
    assert node_queue.last_published_context("box-blank-gpu", "gpu") is None


def test_an_unreadable_read_back_is_simply_unknown(monkeypatch):
    """Best-effort: this runs on the heartbeat path, where a DB blip must never do more
    than leave the box saying 'I do not know'."""
    def boom(*_a, **_k):
        raise RuntimeError("db gone")

    monkeypatch.setattr(node_queue, "connection", boom)
    assert node_queue.last_published_context("box-restart-gpu", "gpu") is None


def test_a_gpu_worker_waits_for_the_migration_its_claim_depends_on():
    """The gpu claim SQL names ``context_length`` on every claim (that is how an
    un-upgraded box is kept off sized rows), so a gpu worker on a pre-0028 database
    would fail every claim. It waits for the orchestrator instead — the rule the
    download lane already follows for 0025."""
    from queue_workflows import claim_worker

    class _W(claim_worker.ClaimWorker):
        def __init__(self, queue):
            self.queue = queue

    assert _W("gpu")._min_schema_version >= 28
    assert _W("cpu")._min_schema_version == claim_worker._NODE_REQUIRED_VERSION
    assert _W("download")._min_schema_version == 25


# ── the dispatcher carries the schema key to the row ─────────────────────────


def _expand(node: dict, *, context: dict | None = None) -> dict:
    # An LLM facade is a pool module: it names no in-process model (the server holds
    # the weights), which the dispatcher otherwise refuses for a gpu node.
    queue_workflows.configure(vlm_pool_node_modules=frozenset({"llm_node"}))
    queue_workflows.set_node_module_package("qwf_ctx_nodes")
    mod = types.ModuleType("qwf_ctx_nodes.llm_node")
    mod.run = lambda inputs, out=None: {"context_delta": {}}
    sys.modules["qwf_ctx_nodes.llm_node"] = mod
    workflows = {"_ctx_wf": {"name": "_ctx_wf",
                             "steps": [{"id": "p", "kind": "pipeline",
                                        "pipeline": "_ctx_pipe"}]}}
    pipelines = {"_ctx_pipe": {"name": "_ctx_pipe",
                               "nodes": [{"node": "llm_node", "gpu": True, **node}]}}
    queue_workflows.set_workflow_provider(
        lambda n: workflows[n], lambda n: pipelines[n])
    run_id = str(uuid.uuid4())
    run_store.insert_run(run_id=run_id, workflow_name="_ctx_wf", out_dir=None,
                         status="queued", mode="node", context=context or {})
    assert dispatcher.start_run(run_id) == 1
    (job,) = node_queue.list_jobs_for_run(run_id)
    return job


def test_a_literal_schema_context_length_reaches_the_row():
    assert _expand({"id": "chat", "context_length": 65536})["context_length"] == 65536


def test_a_context_length_can_come_from_the_run():
    """The caller knows how long ITS prompt is — the schema can defer to it."""
    job = _expand({"id": "chat", "context_length": {"$from": "context_length"}},
                  context={"context_length": 131072})
    assert job["context_length"] == 131072


def test_a_ref_that_resolves_to_nothing_leaves_the_row_unconstrained():
    """A host resolver's optional form (e.g. a coalesce with a null fallback) resolves
    to None for a run that gave no size — that is 'no requirement', not an error."""
    job = _expand({"id": "chat", "context_length": {"$value": None}})
    assert job["context_length"] is None


def test_a_schema_without_the_key_is_byte_identical_to_before():
    assert _expand({"id": "chat"})["context_length"] is None


def test_a_malformed_schema_context_length_fails_the_expansion_loudly():
    """A typo must not become 'no requirement' — that is the silent truncation this
    column exists to prevent."""
    with pytest.raises(ValueError, match="context_length"):
        _expand({"id": "chat", "context_length": "a lot"})
