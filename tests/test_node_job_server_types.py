"""Per-node-job LLM SERVER PREFERENCE — ``server_types`` (migration 0022).

A node-job may name an ORDERED preference of LLM server kinds
(``['vllm', 'ollama']`` = "serve me on vLLM, fall back to ollama"). NULL ⇒
unconstrained, so every existing row and consumer is byte-identical.

Three behaviours are pinned here:

* **round-trip** — the array stores and reads back, empty normalises to NULL
  (an empty list must never mean "no server allowed", which would strand the row);
* **the claim FILTER** — a worker claims a row iff the row is unconstrained OR the
  row's preference intersects the kinds this worker can actually serve. So a
  ``['vllm']``-only job is never grabbed by an ollama-only box;
* **the LANE SPLIT** — ``server_types IS NOT NULL`` means "an EXTERNAL server runs
  this model", so the job is POOL-eligible (PAR-wide) even though it names a
  ``required_model``. That keeps warm-model affinity (the model id is still there)
  while escaping the concurrency-1 inline lane, which exists for jobs that load a
  model INTO the worker process.
"""
from __future__ import annotations

from queue_workflows import dispatcher, node_queue
from tests._helpers import make_run

POOL = ("llm_chat",)


def _queue(node_id: str, *, queue: str = "gpu", server_types=None,
           required_model=None, node_module: str = "llm_chat") -> str:
    run_id = make_run()
    return node_queue.enqueue_node_job(
        run_id=run_id, node_id=node_id, node_module=node_module, queue=queue,
        required_model=required_model, server_types=server_types,
    )


# ── round-trip ───────────────────────────────────────────────────────────────


def test_server_types_round_trip_onto_the_row():
    row = node_queue.get_node_job(_queue("n", server_types=["vllm", "ollama"]))
    assert row["server_types"] == ["vllm", "ollama"], "order is the PREFERENCE — preserve it"


def test_unconstrained_by_default_is_null():
    assert node_queue.get_node_job(_queue("n"))["server_types"] is None


def test_empty_list_normalises_to_null_not_an_unservable_row():
    assert node_queue.get_node_job(_queue("n", server_types=[]))["server_types"] is None


# ── the claim filter ─────────────────────────────────────────────────────────


def test_ollama_box_skips_a_vllm_only_job():
    _queue("n", server_types=["vllm"], required_model="chat-model-a:latest")
    got = node_queue.claim_next_gpu_job(
        0, None, host="ollama-box-gpu", served_types=["ollama"],
        require_model=False, pool_modules=POOL,
    )
    assert got is None, "an ollama-only box must never claim a vllm-only job"


def test_vllm_box_claims_a_vllm_only_job():
    jid = _queue("n", server_types=["vllm"], required_model="chat-model-a:latest")
    got = node_queue.claim_next_gpu_job(
        0, None, host="vllm-box-gpu", served_types=["ollama", "vllm"],
        require_model=False, pool_modules=POOL,
    )
    assert got is not None and got["id"] == jid


def test_ordered_preference_falls_back_to_ollama():
    # ['vllm','ollama'] on a box that only serves ollama STILL claims — the
    # preference is ordered, not exclusive. This is the safety property that lets
    # us ship the preference before vLLM exists anywhere.
    jid = _queue("n", server_types=["vllm", "ollama"], required_model="chat-model-a:latest")
    got = node_queue.claim_next_gpu_job(
        0, None, host="ollama-box-gpu", served_types=["ollama"],
        require_model=False, pool_modules=POOL,
    )
    assert got is not None and got["id"] == jid


def test_unconstrained_row_is_claimed_by_any_box():
    jid = _queue("n", node_module="llm_chat")
    got = node_queue.claim_next_gpu_job(
        0, None, host="any-gpu", served_types=["ollama"],
        require_model=False, pool_modules=POOL,
    )
    assert got is not None and got["id"] == jid


def test_worker_that_advertises_nothing_still_claims_unconstrained_rows():
    # A cold worker must never wedge the queue (same rule as the known_models gate).
    jid = _queue("n")
    got = node_queue.claim_next_gpu_job(
        0, None, host="cold-gpu", served_types=None,
        require_model=False, pool_modules=POOL,
    )
    assert got is not None and got["id"] == jid


# ── the lane split ───────────────────────────────────────────────────────────


def test_server_typed_job_with_a_model_is_POOL_eligible():
    """The core fix: a remote-served job keeps ``required_model`` (so affinity and
    the VRAM/lease machinery still see it) but runs in the PAR-wide pool lane."""
    jid = _queue("n", server_types=["ollama"], required_model="chat-model-a:latest")
    got = node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", served_types=["ollama"],
        require_model=False, pool_modules=POOL,
    )
    assert got is not None and got["id"] == jid


def test_server_typed_job_is_NOT_claimed_by_the_inline_lane():
    """Disjoint claim sets — the inline lane must not steal a pool row, or the two
    lanes over-claim the box."""
    _queue("n", server_types=["ollama"], required_model="chat-model-a:latest")
    got = node_queue.claim_next_gpu_job(
        0, "chat-model-a:latest", host="h-gpu", served_types=["ollama"],
        require_model=True, pool_modules=POOL, known_models=["chat-model-a:latest"],
    )
    assert got is None


def test_model_job_WITHOUT_server_types_still_runs_INLINE_unchanged():
    """Byte-compat: the in-process diffusion path is untouched."""
    jid = _queue("n", required_model="sdxl", node_module="render")
    got = node_queue.claim_next_gpu_job(
        0, "sdxl", host="h-gpu", require_model=True, pool_modules=POOL,
        known_models=["sdxl"],
    )
    assert got is not None and got["id"] == jid


def test_no_model_no_server_types_pool_job_unchanged():
    """Byte-compat for the classic no-model VLM pool job."""
    jid = _queue("n", node_module="llm_chat")
    got = node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=False, pool_modules=POOL,
    )
    assert got is not None and got["id"] == jid


# ── dispatcher threading ─────────────────────────────────────────────────────


def test_dispatcher_reads_server_types_off_the_node_spec(monkeypatch):
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        return "jid"

    monkeypatch.setattr(node_queue, "enqueue_node_job", _fake)
    run_id = make_run()
    dispatcher._enqueue(
        run_id,
        {"id": "n", "node": "llm_chat", "queue": "gpu",
         "required_model": "chat-model-a:latest",
         "server_types": ["vllm", "ollama"]},
        {"id": run_id, "priority": 100, "project": "", "context": {}},
    )
    assert seen["server_types"] == ["vllm", "ollama"]


def test_remote_served_row_is_claimable_even_if_its_module_is_NOT_pool_listed():
    """REGRESSION. The inline lane refuses every remote-served row, so if
    the pool lane ALSO demanded pool_modules membership, a remote-served row whose
    module is not listed would be claimable by NEITHER lane — queued forever, silently.

    A real shape: a node like `plan_gen_segments` carries chat-model-a (so it gets a
    server preference stamped) but is not in vlm_pool_node_modules."""
    jid = _queue("n", server_types=["ollama"], required_model="chat-model-a:latest",
                 node_module="plan_gen_segments")          # deliberately NOT in POOL

    inline = node_queue.claim_next_gpu_job(
        0, "chat-model-a:latest", host="h-gpu", served_types=["ollama"],
        require_model=True, pool_modules=POOL, known_models=["chat-model-a:latest"],
    )
    assert inline is None, "inline must still refuse a remote-served row"

    pool = node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", served_types=["ollama"],
        require_model=False, pool_modules=POOL,
    )
    assert pool is not None and pool["id"] == jid, "…so the POOL lane must take it"


def test_no_model_job_outside_pool_modules_is_still_inline_only():
    """The counterpart: pool_modules still gates NO-MODEL jobs, so heavy in-process
    GPU work keeps running serially on the inline lane."""
    jid = _queue("n", node_module="heavy_job")           # no model, not pool-listed
    pool = node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=False, pool_modules=POOL,
    )
    assert pool is None
    inline = node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=True, pool_modules=POOL,
    )
    assert inline is not None and inline["id"] == jid


def test_remote_served_job_executes_without_a_model_cache():
    """REGRESSION (found LIVE on the first real job after deploy). A remote-served
    row keeps its ``required_model`` — affinity, the VRAM gate and the box lease all
    read it — but it is claimed by the PAR-wide POOL lane, which supplies NO
    ModelCache. `execute_node` used to demand a cache for any row naming a model, so
    every chat-model-a job failed instantly with "no model_cache was supplied".

    The model lives in the external server, not in this process; there is nothing to
    load here."""
    from queue_workflows import node_executor

    run_id = make_run()
    jid = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="fake_llm", queue="gpu",
        required_model="chat-model-a:latest", server_types=["ollama"],
    )
    job = node_queue.get_node_job(jid)

    called = {}

    def _fake_invoke(**kw):
        called["handle"] = kw.get("handle")
        called["llm_server"] = kw.get("llm_server")
        return {"context_delta": {"ok": True}}

    import queue_workflows.node_executor as ne
    orig = ne._invoke
    ne._invoke = _fake_invoke
    try:
        node_executor.execute_node(job, model_cache=None)   # the pool lane's call shape
    finally:
        ne._invoke = orig

    row = node_queue.get_node_job(jid)
    assert row["status"] == "completed", f"expected completed, got {row['status']}: {row.get('error')}"
    assert called["handle"] is None, "no in-process model handle for a remote-served node"


def test_in_process_model_job_still_REQUIRES_a_cache():
    """The other side of the line — a genuine in-process model job must still fail
    loudly without a cache, or a misconfigured diffusion worker would run modelless."""
    from queue_workflows import node_executor

    jid = node_queue.enqueue_node_job(
        run_id=make_run(), node_id="n", node_module="render", queue="gpu",
        required_model="sdxl",                      # no server_types ⇒ in-process
    )
    node_executor.execute_node(node_queue.get_node_job(jid), model_cache=None)
    row = node_queue.get_node_job(jid)
    assert row["status"] == "failed"
    assert "no model_cache was supplied" in (row.get("error") or "")
