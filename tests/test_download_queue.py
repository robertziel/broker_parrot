"""The ``download`` node-job queue — a third first-class lane (migration 0025).

Downloads are WAIT (network I/O), not compute: they must not occupy a cpu/gpu
claim slot, and a shared uplink must not be saturated by N of them running at
once. So ``download`` is a NODE queue (full DAG citizenship — ``depends_on``
edges, ``force_box`` placement, the lease/watchdog/outbox machinery) whose
claim carries a **fleet-wide per-project width cap**, default **1**: while
``download_max_concurrent`` rows are ``running``, every further claim — any
host — returns None.

Contracts pinned here:

- enqueue accepts ``queue='download'``; the allow-list stays HOST-SIDE
  (unknown names still raise — migration 0025 dropped the DB CHECK, 0008-style);
- ``required_model`` remains gpu-only (a download row may not carry one);
- claim sets are disjoint (cpu claims never see download rows and vice versa);
- THE CAP: claim-side, counted over ``running`` download rows in the SAME
  project — host-independent (unlike the gpu box-slot arbiter), per-project
  scoped (tenant isolation, 0017), freed by terminals AND by the lease reclaim
  (dead-downloader recovery), race-safe on pg via an advisory-lock serializer;
- schema routing: ``download: true`` on a node ⇒ queue='download';
  ``download``+``gpu`` on one node is a contradiction ⇒ loud ValueError;
- worker plumbing: ``ClaimWorker(queue='download')`` is node-family
  (concurrency-1, ``node_job_ready`` wake, schema gate ≥ 25) and executes a
  download node end-to-end via ``run_once``.
"""

from __future__ import annotations

import threading
import uuid

import pytest

import queue_workflows
from queue_workflows import claim_worker, dispatcher, node_queue, run_store
from queue_workflows.db import connection
from tests._helpers import make_run


def _backdate_lease(job_id: str, *, seconds_ago: float = 30.0) -> None:
    """Force a row ``running`` with an ALREADY-EXPIRED lease, dialect-safely.

    Uses ``past_seconds`` with a POSITIVE param — a negative interval builds an
    invalid sqlite modifier (``'+-30 seconds'`` ⇒ NULL), so this never passes
    negatives through the dialect seam."""
    from queue_workflows.dialect import get_dialect
    d = get_dialect()
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            f"SET status='running', started_at={d.now}, claimed_by='dl-host-x', "
            f"    lease_expires_at = {d.past_seconds('%(ago)s')} "
            "WHERE id=%(id)s",
            {"ago": float(seconds_ago), "id": job_id},
        )


def _enqueue_download(run_id: str, node_id: str = "dl", **kw) -> str:
    return node_queue.enqueue_node_job(
        run_id=run_id, node_id=node_id, node_module="_dl_node",
        queue="download", **kw,
    )


# ── enqueue ────────────────────────────────────────────────────────────────


def test_enqueue_download_job_succeeds():
    run_id = make_run()
    job_id = _enqueue_download(run_id)
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
    assert row["queue"] == "download"
    assert row["required_model"] is None


def test_enqueue_download_with_model_rejected():
    run_id = make_run()
    with pytest.raises(ValueError, match="required_model"):
        node_queue.enqueue_node_job(
            run_id=run_id, node_id="dl", node_module="x",
            queue="download", required_model="some-model",
        )


def test_enqueue_unknown_queue_still_rejected():
    # 0025 drops the DB CHECK; the allow-list is host-side and must still bite.
    run_id = make_run()
    with pytest.raises(ValueError, match="queue must be"):
        node_queue.enqueue_node_job(
            run_id=run_id, node_id="n", node_module="x", queue="tpu",
        )


# ── claim-set disjointness ─────────────────────────────────────────────────


def test_download_claim_ignores_cpu_and_gpu_rows():
    run_id = make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="c", node_module="x", queue="cpu")
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="x", queue="gpu")
    assert node_queue.claim_next_download_job(host="dl-1") is None


def test_cpu_claim_ignores_download_rows():
    run_id = make_run()
    _enqueue_download(run_id)
    assert node_queue.claim_next_cpu_job(host="cpu-1") is None


def test_download_claim_stamps_lease_and_host():
    run_id = make_run()
    job_id = _enqueue_download(run_id)
    row = node_queue.claim_next_download_job(host="dl-1", lease_s=600)
    assert row is not None and row["id"] == job_id
    assert row["status"] == "running"
    assert row["claimed_by"] == "dl-1"
    assert row["lease_expires_at"] is not None
    assert row["resolved_server"] is None  # no LLM serving kind on this lane


# ── THE CAP — one downloader at a time (default), fleet-wide per project ───


def test_second_claim_refused_while_one_running_default_cap():
    run_id = make_run()
    _enqueue_download(run_id, node_id="dl1")
    _enqueue_download(run_id, node_id="dl2")
    first = node_queue.claim_next_download_job(host="dl-host-a")
    assert first is not None
    # A DIFFERENT host: the cap is fleet-wide, not per-box.
    assert node_queue.claim_next_download_job(host="dl-host-b") is None


def test_cap_2_admits_two_then_refuses_third():
    run_id = make_run()
    for i in range(3):
        _enqueue_download(run_id, node_id=f"dl{i}")
    a = node_queue.claim_next_download_job(host="a", max_concurrent=2)
    b = node_queue.claim_next_download_job(host="b", max_concurrent=2)
    assert a is not None and b is not None
    assert node_queue.claim_next_download_job(host="c", max_concurrent=2) is None


def test_cap_scope_is_per_project():
    # Tenant isolation (0017): project A's running download must not block B.
    run_a = make_run()
    run_b = make_run()
    with connection() as c, c.cursor() as cur:
        cur.execute("UPDATE workflow_runs SET project='proj-a' WHERE id=%s", (run_a,))
        cur.execute("UPDATE workflow_runs SET project='proj-b' WHERE id=%s", (run_b,))
    _enqueue_download(run_a, project="proj-a")
    _enqueue_download(run_b, project="proj-b")
    a = node_queue.claim_next_download_job(host="a", project="proj-a")
    assert a is not None
    b = node_queue.claim_next_download_job(host="b", project="proj-b")
    assert b is not None, "a running download in proj-a must not block proj-b"


def test_slot_frees_on_completion():
    run_id = make_run()
    _enqueue_download(run_id, node_id="dl1")
    _enqueue_download(run_id, node_id="dl2")
    first = node_queue.claim_next_download_job(host="a")
    assert node_queue.claim_next_download_job(host="b") is None
    node_queue.mark_completed(first["id"], context_delta={}, seconds=0.1)
    second = node_queue.claim_next_download_job(host="b")
    assert second is not None and second["id"] != first["id"]


def test_slot_frees_on_lease_reclaim():
    """Dead-downloader recovery: the lease reclaim IS the slot release."""
    run_id = make_run()
    dead = _enqueue_download(run_id, node_id="dl1")
    _backdate_lease(dead, seconds_ago=30)
    # While the corpse still reads 'running', the lane is occupied…
    _enqueue_download(run_id, node_id="dl2")
    assert node_queue.claim_next_download_job(host="b") is None
    # …until the sweep re-queues it; then the lane is claimable again.
    reclaimed = node_queue.reclaim_expired_leases()
    assert any(r["id"] == dead for r in reclaimed)
    row = node_queue.claim_next_download_job(host="b")
    assert row is not None
    assert row["id"] == dead, "reclaimed work re-claims FIRST (is_priority)"


def test_claim_order_is_priority_then_fifo():
    run_id = make_run()
    first = _enqueue_download(run_id, node_id="dl1")
    jumped = _enqueue_download(run_id, node_id="dl2")
    node_queue.prioritize_node_job(jumped)
    row = node_queue.claim_next_download_job(host="a")
    assert row["id"] == jumped
    node_queue.mark_completed(jumped, context_delta={}, seconds=0.1)
    row2 = node_queue.claim_next_download_job(host="a")
    assert row2["id"] == first


def test_box_placement_respected():
    run_id = make_run()
    _enqueue_download(run_id, node_id="dl1", force_box=["box-b"])
    assert node_queue.claim_next_download_job(host="a", box="box-a") is None
    row = node_queue.claim_next_download_job(host="b", box="box-b")
    assert row is not None


def test_run_cancel_guard_applies():
    run_id = make_run(status="cancelled")
    _enqueue_download(run_id)
    assert node_queue.claim_next_download_job(host="a") is None


# ── the knob ───────────────────────────────────────────────────────────────


def test_resolve_download_max_concurrent_env_beats_config(monkeypatch):
    queue_workflows.configure(download_max_concurrent=3)
    assert node_queue.resolve_download_max_concurrent() == 3
    monkeypatch.setenv("QUEUE_WORKFLOWS_DOWNLOAD_MAX_CONCURRENT", "2")
    assert node_queue.resolve_download_max_concurrent() == 2
    # Malformed / sub-1 degrade to the floor, never to a dead lane.
    monkeypatch.setenv("QUEUE_WORKFLOWS_DOWNLOAD_MAX_CONCURRENT", "0")
    assert node_queue.resolve_download_max_concurrent() == 1
    monkeypatch.setenv("QUEUE_WORKFLOWS_DOWNLOAD_MAX_CONCURRENT", "nope")
    assert node_queue.resolve_download_max_concurrent() == 3


def test_configure_floors_download_max_concurrent():
    queue_workflows.configure(download_max_concurrent=0)
    assert queue_workflows.get_config().download_max_concurrent == 1


def test_configure_rejects_download_as_ingest_queue():
    with pytest.raises(ValueError, match="reserved"):
        queue_workflows.configure(ingest_queues=frozenset({"download"}))


# ── dispatcher routing ─────────────────────────────────────────────────────


def test_queue_of_download_flag():
    assert dispatcher._queue_of({"id": "n", "download": True}) == "download"
    assert dispatcher._queue_of({"id": "n"}) == "cpu"
    assert dispatcher._queue_of({"id": "n", "gpu": True}) == "gpu"


def test_queue_of_download_plus_gpu_raises():
    with pytest.raises(ValueError, match="download"):
        dispatcher._queue_of({"id": "n", "download": True, "gpu": True})


def test_start_run_routes_download_node():
    import sys
    import types

    queue_workflows.set_node_module_package("qwf_dl_nodes")
    for name in ("dl_node", "compute_node"):
        mod = types.ModuleType(f"qwf_dl_nodes.{name}")
        mod.run = lambda inputs, out=None: {"context_delta": {}}
        sys.modules[f"qwf_dl_nodes.{name}"] = mod

    workflows = {
        "_dl_wf": {"name": "_dl_wf",
                   "steps": [{"id": "p", "kind": "pipeline",
                              "pipeline": "_dl_pipe"}]},
    }
    pipelines = {
        "_dl_pipe": {"name": "_dl_pipe",
                     "nodes": [
                         {"id": "fetch", "node": "dl_node", "download": True},
                         {"id": "crunch", "node": "compute_node",
                          "depends_on": ["fetch"]},
                     ]},
    }
    queue_workflows.set_workflow_provider(
        lambda n: workflows[n], lambda n: pipelines[n])

    run_id = str(uuid.uuid4())
    run_store.insert_run(run_id=run_id, workflow_name="_dl_wf",
                         out_dir=None, status="queued", mode="node")
    assert dispatcher.start_run(run_id) == 1
    # Node ids are step-prefixed on expansion (``<step>/<node>``).
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert jobs["p/fetch"]["queue"] == "download", (
        "the _nodes_of whitelist must thread `download` through — a key not "
        "copied there is silently dropped and the node routes to cpu"
    )


def test_pipeline_schema_cpu_slots_reaches_the_row():
    """Regression for the 0024 whitelist drop (found while adding `download`):
    a schema's ``cpu_slots`` was not copied by ``_nodes_of``, so every
    pipeline-expanded row stamped NULL and a wide node reserved one slot."""
    import sys
    import types

    queue_workflows.set_node_module_package("qwf_dl_nodes")
    mod = types.ModuleType("qwf_dl_nodes.wide_node")
    mod.run = lambda inputs, out=None: {"context_delta": {}}
    sys.modules["qwf_dl_nodes.wide_node"] = mod

    workflows = {"_w_wf": {"name": "_w_wf",
                           "steps": [{"id": "p", "kind": "pipeline",
                                      "pipeline": "_w_pipe"}]}}
    pipelines = {"_w_pipe": {"name": "_w_pipe",
                             "nodes": [{"id": "wide", "node": "wide_node",
                                        "cpu_slots": 5}]}}
    queue_workflows.set_workflow_provider(
        lambda n: workflows[n], lambda n: pipelines[n])

    run_id = str(uuid.uuid4())
    run_store.insert_run(run_id=run_id, workflow_name="_w_wf",
                         out_dir=None, status="queued", mode="node")
    assert dispatcher.start_run(run_id) == 1
    (job,) = node_queue.list_jobs_for_run(run_id)
    assert job["cpu_slots"] == 5


# ── worker plumbing ────────────────────────────────────────────────────────


def test_budget_for_download():
    assert claim_worker.budget_for({"queue": "download"}) \
        == claim_worker.DOWNLOAD_BUDGET_S


def test_claim_worker_accepts_download_queue():
    w = claim_worker.ClaimWorker(queue="download", host="dl-1")
    assert w._is_ingest is False
    assert w.threads == 1                       # concurrency-1, like gpu/ingest
    assert w._wake_channel == "node_job_ready"  # node family wake
    assert w.model_cache is None                # no warm-model machinery


def test_download_schema_gate_is_0025():
    assert claim_worker.ClaimWorker(
        queue="download", host="dl-1")._min_schema_version == 25
    assert claim_worker.ClaimWorker(
        queue="cpu", host="c")._min_schema_version == 6


def test_run_once_executes_download_node_end_to_end():
    import sys
    import types

    queue_workflows.set_node_module_package("qwf_dl_nodes")
    ran: list[str] = []

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None,
            cancel_event=None):
        ran.append("dl")
        return {"context_delta": {"downloaded": True}}

    mod = types.ModuleType("qwf_dl_nodes._dl_body")
    mod.run = run
    sys.modules["qwf_dl_nodes._dl_body"] = mod

    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="dl", node_module="_dl_body", queue="download",
    )
    worker = claim_worker.ClaimWorker(queue="download", host="dl-e2e")
    assert worker.run_once() is True
    assert ran == ["dl"]
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "completed"
    assert row["context_delta"] == {"downloaded": True}
    assert row["claimed_by"] == "dl-e2e"


# ── dedup — "one file is never downloaded twice" (migration 0025) ──────────


def _install_body(name: str, ran: list):
    import sys
    import types

    queue_workflows.set_node_module_package("qwf_dl_nodes")

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None,
            cancel_event=None):
        ran.append(name)
        return {"context_delta": {"fetched_by": name}}

    mod = types.ModuleType(f"qwf_dl_nodes.{name}")
    mod.run = run
    sys.modules[f"qwf_dl_nodes.{name}"] = mod


def test_enqueue_stores_dedup_key():
    run_id = make_run()
    job_id = _enqueue_download(run_id, dedup_key="dataset|v1|2024-01-01|12z")
    assert node_queue.get_node_job(job_id)["dedup_key"] == "dataset|v1|2024-01-01|12z"
    # Blank ⇒ NULL — a whitespace key must never become a universal twin.
    job2 = _enqueue_download(run_id, node_id="dl2", dedup_key="   ")
    assert node_queue.get_node_job(job2)["dedup_key"] is None


def test_dedup_second_job_completes_without_running_body():
    ran: list[str] = []
    _install_body("_dl_first", ran)
    _install_body("_dl_second", ran)
    key = "release|2024-01-01|12z"

    run_a = make_run()
    a = node_queue.enqueue_node_job(
        run_id=run_a, node_id="dl", node_module="_dl_first",
        queue="download", dedup_key=key)
    worker = claim_worker.ClaimWorker(queue="download", host="dl-1")
    assert worker.run_once() is True
    assert ran == ["_dl_first"]
    assert node_queue.get_node_job(a)["status"] == "completed"

    # A SECOND pipeline's downloader for the same output: completes as
    # success, body NOT invoked, twin's context_delta + provenance marker.
    run_b = make_run()
    b = node_queue.enqueue_node_job(
        run_id=run_b, node_id="dl", node_module="_dl_second",
        queue="download", dedup_key=key)
    assert worker.run_once() is True
    assert ran == ["_dl_first"], "the duplicate's body must NOT run"
    row = node_queue.get_node_job(b)
    assert row["status"] == "completed"
    assert row["context_delta"]["fetched_by"] == "_dl_first"
    assert row["context_delta"]["_deduped_from"] == a

    # The outbox contract holds for the deduped job too: its terminal wrote a
    # 'completed' dispatch event, so downstream fan-out still fires.
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT kind FROM workflow_dispatch_events "
            "WHERE run_id = %s AND node_id = %s",
            (run_b, "dl"),
        )
        kinds = [r["kind"] for r in cur.fetchall()]
    assert kinds == ["completed"]


def test_dedup_ignores_failed_twin():
    ran: list[str] = []
    _install_body("_dl_after_fail", ran)
    key = "release|failed-first"
    run_a = make_run()
    a = node_queue.enqueue_node_job(
        run_id=run_a, node_id="dl", node_module="x",
        queue="download", dedup_key=key)
    claimed = node_queue.claim_next_download_job(host="a")
    assert claimed["id"] == a
    node_queue.mark_failed(a, error="boom", seconds=0.1)

    run_b = make_run()
    node_queue.enqueue_node_job(
        run_id=run_b, node_id="dl", node_module="_dl_after_fail",
        queue="download", dedup_key=key)
    worker = claim_worker.ClaimWorker(queue="download", host="dl-1")
    assert worker.run_once() is True
    assert ran == ["_dl_after_fail"], (
        "a failed twin proves nothing — the body must run")


def test_dedup_is_project_scoped():
    ran: list[str] = []
    _install_body("_dl_proj_b", ran)
    key = "release|shared-key"
    run_a = make_run()
    with connection() as c, c.cursor() as cur:
        cur.execute("UPDATE workflow_runs SET project='proj-a' WHERE id=%s",
                    (run_a,))
    a = node_queue.enqueue_node_job(
        run_id=run_a, node_id="dl", node_module="x",
        queue="download", dedup_key=key, project="proj-a")
    node_queue.claim_next_download_job(host="a", project="proj-a")
    node_queue.mark_completed(a, context_delta={}, seconds=0.1)

    # Same key, DIFFERENT tenant: no dedup across the 0017 boundary.
    assert node_queue.find_completed_duplicate(
        dedup_key=key, project="proj-b") is None
    assert node_queue.find_completed_duplicate(
        dedup_key=key, project="proj-a") is not None


def test_keyless_jobs_never_dedup():
    ran: list[str] = []
    _install_body("_dl_k1", ran)
    _install_body("_dl_k2", ran)
    run_a = make_run()
    node_queue.enqueue_node_job(
        run_id=run_a, node_id="d1", node_module="_dl_k1", queue="download")
    node_queue.enqueue_node_job(
        run_id=run_a, node_id="d2", node_module="_dl_k2", queue="download")
    worker = claim_worker.ClaimWorker(queue="download", host="dl-1")
    assert worker.run_once() is True
    assert worker.run_once() is True
    assert sorted(ran) == ["_dl_k1", "_dl_k2"], (
        "NULL dedup_key means no dedup — both bodies run")


def test_dispatcher_threads_dedup_key_from_schema():
    import sys
    import types

    queue_workflows.set_node_module_package("qwf_dl_nodes")
    mod = types.ModuleType("qwf_dl_nodes.keyed_dl")
    mod.run = lambda inputs, out=None: {"context_delta": {}}
    sys.modules["qwf_dl_nodes.keyed_dl"] = mod

    workflows = {"_k_wf": {"name": "_k_wf",
                           "steps": [{"id": "p", "kind": "pipeline",
                                      "pipeline": "_k_pipe"}]}}
    pipelines = {"_k_pipe": {"name": "_k_pipe",
                             "nodes": [{"id": "dl", "node": "keyed_dl",
                                        "download": True,
                                        "dedup_key": "static|key|v1"}]}}
    queue_workflows.set_workflow_provider(
        lambda n: workflows[n], lambda n: pipelines[n])

    run_id = str(uuid.uuid4())
    run_store.insert_run(run_id=run_id, workflow_name="_k_wf",
                         out_dir=None, status="queued", mode="node")
    assert dispatcher.start_run(run_id) == 1
    (job,) = node_queue.list_jobs_for_run(run_id)
    assert job["queue"] == "download"
    assert job["dedup_key"] == "static|key|v1"


# ── snapshot ───────────────────────────────────────────────────────────────


def test_snapshot_includes_download_section():
    run_id = make_run()
    _enqueue_download(run_id)
    snap = node_queue.snapshot()
    assert "download" in snap
    assert len(snap["download"]["queued"]) == 1
    assert snap["counts"].get("download_queued") == 1


# ── pg race-safety — the per-kind serializer lesson, applied to this arbiter ─


@pytest.mark.pg_only
def test_concurrent_claims_cannot_exceed_cap():
    """Two SIMULTANEOUS claimants, cap 1 ⇒ exactly one wins, every round.

    Without the (queue, project)-keyed advisory-lock serializer both MVCC
    snapshots read "0 running" and both take a row — the same race the box-slot
    arbiter documents (test_audit_deltas_multi_server). 10 rounds keeps the runtime tiny while making a
    silent regression overwhelmingly unlikely to pass."""
    run_id = make_run()
    for i in range(2):
        _enqueue_download(run_id, node_id=f"dl{i}")

    for round_no in range(10):
        results: list[dict | None] = [None, None]
        barrier = threading.Barrier(2)

        def _claim(slot: int) -> None:
            barrier.wait()
            results[slot] = node_queue.claim_next_download_job(
                host=f"racer-{slot}")

        threads = [threading.Thread(target=_claim, args=(i,)) for i in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        winners = [r for r in results if r is not None]
        assert len(winners) == 1, (
            f"round {round_no}: cap-1 admitted {len(winners)} concurrent claims"
        )
        # Reset: put the winner back so the next round starts from 0 running.
        with connection() as c, c.cursor() as cur:
            cur.execute(
                "UPDATE workflow_node_jobs SET status='queued', claimed_by=NULL, "
                "lease_expires_at=NULL, started_at=NULL, is_priority=FALSE "
                "WHERE id=%s",
                (winners[0]["id"],),
            )
