"""Fault-path deltas in the multi-server-type (ollama/vLLM) per-node-job LLM routing.

Each case below is a decision that was correct in the common case but not under a
fault. They share one
shape: a decision that is CORRECT in the common case but takes the LOOSE / STALE /
DIVERGENT branch under a fault. This module pins the tight branch for each.

  * **D1** the per-box claim SERIALIZER only fired for the box-slot arbiter, so two
    concurrent claimants could both read "cap-1 running" for a KIND and both take
    the last per-kind slot — the very over-claim the arbiter exists to stop.
  * **D2** ``default_server`` was threaded through the whole claim path and bound
    into every statement while NOTHING read it — and producing it cost a
    ``worker_controls`` SELECT on every single claim.
  * **D3** a row resolving to an UNKNOWN kind was capped at ``max(caps)`` — the
    loosest slot budget on the box. An unrecognised server is exactly what we know
    least about; it must get the tightest.
  * **D4** the claim STAMPS the kind it counted a job against; if that server is
    dead at dispatch the job is served elsewhere — silently making the arbiter's
    accounting a lie for as long as it runs.
  * **D5** a ``pending_fn`` returning ``None`` (an unreadable queue) still stopped
    the server, while a pending_fn that RAISED kept it. Same ignorance, opposite —
    and far more expensive — action.
"""
from __future__ import annotations

import pytest

from queue_workflows import node_queue
from tests._helpers import make_run


# ── D1 — the serializer must cover per-KIND caps, not just the box arbiter ────


def _spy_serializer(monkeypatch):
    """Record whether the claim took the per-box serialisation lock."""
    from queue_workflows import dialect as _dialect

    calls: list[str] = []
    real = _dialect.get_dialect()


    class _Wrapped:
        def __getattr__(self, name):
            return getattr(real, name)

        def box_claim_serializer(self):
            calls.append("serializer")
            return real.box_claim_serializer()

    monkeypatch.setattr("queue_workflows.dialect.get_dialect", lambda: _Wrapped())
    return calls


def test_per_kind_caps_take_the_claim_serializer_too(monkeypatch):
    """Without this, two claimants racing for ollama's LAST (only) slot both read
    "0 running" and both claim — the per-kind cap of 1 becomes 2. The box-slot
    arbiter has always serialised; the per-kind arbiter counts the same way and
    needs the same lock."""
    calls = _spy_serializer(monkeypatch)
    node_queue.claim_next_gpu_job(
        0, None, host="box-a-gpu", max_running=None,
        kind_caps={"ollama": 1, "vllm": 3}, served_types=["ollama"],
        require_model=False, pool_modules=("pooled_llm",),
    )
    assert calls, "a per-kind-capped claim must serialise like the box arbiter does"


def test_an_unarbitrated_claim_still_takes_no_lock(monkeypatch):
    """The lock is not free — a claim with neither arbiter must not pay for it.
    Since the law, ``kind_caps=None`` means "engine defaults" (the law
    is ON), so the unarbitrated case is the explicit opt-out ``{}``."""
    calls = _spy_serializer(monkeypatch)
    node_queue.claim_next_gpu_job(
        0, None, host="box-a-gpu", max_running=None, kind_caps={},
        require_model=False, pool_modules=("pooled_llm",),
    )
    assert not calls


# ── D2 — the parameter that died as dead code, reinstated WITH a reader ──────
#
# D2's finding stands: a parameter nothing reads, fed by an uncached DB SELECT per
# claim, is waste. The law gave ``default_server`` its reader — an
# UNDECLARED row's stamp IS the caller's prediction — so the parameter is back,
# and what these tests now pin is the part of D2 that must never come back: the
# per-claim DB read, and an exception path into the claim loop.


def test_default_server_is_back_and_gpu_only():
    """The GPU claim reads it (test_per_kind_arbiter proves the SQL does); cpu
    rows have no serving kind, so the parameter stays off that signature."""
    import inspect

    assert "default_server" in inspect.signature(
        node_queue.claim_next_gpu_job).parameters
    assert "default_server" not in inspect.signature(
        node_queue.claim_next_cpu_job).parameters, (
        "cpu rows have no serving kind — the parameter is gpu-only")


def test_the_worker_passes_the_prediction_but_keeps_the_db_off_the_hot_path():
    """Parsed, not grepped: the gpu claim call sites must pass the prediction, and
    the producer must be the warm-hint/TTL-cache path — ``_default_llm_kind`` (the
    uncached worker_controls SELECT per claim) stays dead."""
    import ast
    import inspect

    from queue_workflows import claim_worker

    tree = ast.parse(inspect.getsource(claim_worker))
    passed = [kw.arg for node in ast.walk(tree) if isinstance(node, ast.Call)
              for kw in node.keywords if kw.arg == "default_server"]
    assert passed, "the gpu claim call sites must pass default_server now"
    assert not hasattr(claim_worker.ClaimWorker, "_default_llm_kind"), (
        "the per-claim worker_controls read must stay dead")


def test_the_prediction_mirrors_the_active_server_default(monkeypatch):
    """The claim must count an undeclared job where dispatch will PUT it: a warm
    vLLM is the box's active server (6931668 — the active server wins the
    no-preference default), else the TTL-cached configured default."""
    from queue_workflows import claim_worker
    from queue_workflows.llm_backends import factory

    monkeypatch.setattr(factory, "cached_default_kind", lambda h, q: "ollama")
    assert claim_worker.predict_default_kind(["vllm"], "h", "gpu") == "vllm"
    assert claim_worker.predict_default_kind([], "h", "gpu") == "ollama"
    assert claim_worker.predict_default_kind(["comfyui"], "h", "gpu") == "ollama", (
        "a warm non-LLM resident is not an LLM default")


def test_the_prediction_never_raises_into_the_claim_loop(monkeypatch):
    """A control-plane read failure must degrade to the engine default — a claim
    loop that dies because worker_controls hiccuped is a parked box."""
    from queue_workflows import claim_worker
    from queue_workflows.llm_backends import factory

    def _boom(h, q):
        raise RuntimeError("worker_controls unreadable")

    monkeypatch.setattr(factory, "cached_default_kind", _boom)
    assert claim_worker.predict_default_kind(None, "h", "gpu") == "ollama"


# ── D3 — an unknown kind gets the TIGHTEST cap, not the loosest ──────────────


def test_an_unknown_server_kind_gets_the_tightest_cap():
    """A row resolving to a kind we have no cap for is the case we understand
    LEAST — it must inherit the most conservative budget on the box, not the most
    permissive. With max() a typo'd or newly-introduced server_types value silently
    bought vLLM's PAR-wide slot budget."""
    run = make_run()
    for i in range(3):
        node_queue.enqueue_node_job(
            run_id=run, node_id=f"n{i}", node_module="pooled_llm", queue="gpu",
            required_model="m", server_types=["exotic"],
        )
    caps = {"ollama": 1, "vllm": 3}
    first = node_queue.claim_next_gpu_job(
        0, None, host="box-a-gpu", kind_caps=caps, served_types=["exotic"],
        require_model=False, pool_modules=("pooled_llm",),
    )
    assert first is not None and first["resolved_server"] == "exotic"
    second = node_queue.claim_next_gpu_job(
        0, None, host="box-a-gpu", kind_caps=caps, served_types=["exotic"],
        require_model=False, pool_modules=("pooled_llm",),
    )
    assert second is None, (
        "an unknown kind must be capped at min(caps)=1, not max(caps)=3")


# ── D4 — dispatch must not silently diverge from what the claim counted ──────


def test_serving_a_job_on_a_kind_the_claim_did_not_count_RESTAMPS_the_row(monkeypatch):
    """The claim stamps resolved_server and the arbiter counts running rows BY that
    stamp. If the stamped server is dead at dispatch we serve elsewhere — correct,
    the job should not die for it — but leaving the stale stamp makes every later
    claim count this job against a kind that is not running it. The arbiter reads
    the DB fresh on each claim, so correcting the stamp is enough to keep its view
    true."""
    from queue_workflows.llm_backends import factory

    run = make_run()
    jid = node_queue.enqueue_node_job(
        run_id=run, node_id="n", node_module="pooled_llm", queue="gpu",
        required_model="m", server_types=["vllm", "ollama"],
    )
    node_queue.claim_next_gpu_job(
        0, None, host="box-a-gpu", served_types=["vllm"],
        require_model=False, pool_modules=("pooled_llm",),
    )
    assert node_queue.get_node_job(jid)["resolved_server"] == "vllm"

    job = dict(node_queue.get_node_job(jid))
    factory._reset_probe_cache_for_tests()
    # vLLM died between claim and dispatch; only ollama answers now. monkeypatch
    # (not a bare assignment) so the stub cannot leak into another module's run.
    monkeypatch.setattr(factory, "_probe_available", lambda kinds: ["ollama"])
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: object())
    try:
        factory.resolve_for_job(job, "box-a-gpu", "gpu")
    finally:
        factory._reset_probe_cache_for_tests()
    assert node_queue.get_node_job(jid)["resolved_server"] == "ollama", (
        "the row must be re-stamped to the kind actually serving it")


def test_the_no_preference_default_path_restamps_a_stale_prediction(monkeypatch):
    """With never-NULL stamps the old premise — "the box default
    cannot contradict the stamp" — is gone: a claim can predict 'vllm' (warm at
    claim time) and dispatch can land on the configured default because vLLM died
    in between. ``_serve(None)`` skipped the D4 re-stamp on exactly that path,
    leaving the arbiter counting a vllm job that ollama is serving."""
    from queue_workflows.llm_backends import factory

    run = make_run()
    jid = node_queue.enqueue_node_job(
        run_id=run, node_id="np", node_module="pooled_llm", queue="gpu")
    node_queue.claim_next_gpu_job(
        0, None, host="box-a-gpu", require_model=False,
        pool_modules=("pooled_llm",), default_server="vllm")
    assert node_queue.get_node_job(jid)["resolved_server"] == "vllm"

    job = dict(node_queue.get_node_job(jid))
    factory._reset_probe_cache_for_tests()
    monkeypatch.setattr(factory, "_probe_available", lambda kinds: ["ollama"])
    monkeypatch.setattr(factory, "cached_default_kind", lambda h, q: "ollama")
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: object())
    try:
        factory.resolve_for_job(job, "box-a-gpu", "gpu")
    finally:
        factory._reset_probe_cache_for_tests()
    assert node_queue.get_node_job(jid)["resolved_server"] == "ollama", (
        "the default path must re-stamp like every other divergent dispatch")


# ── D5 — an unreadable queue must never trigger the expensive action ─────────


@pytest.mark.parametrize("pending", [None, "", "not-a-number"])
def test_an_unreadable_pending_count_KEEPS_the_warm_server(pending):
    """A pending_fn that RAISES already keeps the server (stopping costs a
    long reload from network storage; keeping costs RAM for one tick). A pending_fn that
    returns None/garbage is the SAME ignorance and must take the same branch — it
    used to coerce to 0 and stop."""
    stops = _run_idle_stopper(pending_fn=lambda: pending)
    assert not stops, f"pending_fn()->{pending!r} is unreadable, not zero"


def test_a_genuine_zero_pending_still_stops():
    """The fail-safe must not become a never-stop: a queue we CAN read, that says
    nothing is waiting, still releases the VRAM."""
    assert _run_idle_stopper(pending_fn=lambda: 0) == [1]


def test_a_raising_pending_fn_keeps_it_too_unchanged():
    """The branch this aligns with — pinned so the two stay together."""
    def _boom():
        raise RuntimeError("db down")
    assert _run_idle_stopper(pending_fn=_boom) == []


def _run_idle_stopper(*, pending_fn):
    """Drive a serving, idle-past-TTL vLLM to the stop decision. The first tick
    primes the seen-counter; the clock then jumps past the TTL with the counter
    unchanged, which is exactly 'served, nothing running, nobody finished'."""
    from queue_workflows import llm_idle_unload

    stops: list[int] = []
    t = [0.0]
    st = llm_idle_unload.VLLMIdleStopper(
        active_fn=lambda: 0,
        activity_fn=lambda: {"served": True, "running": 0, "finished": 7},
        stop_fn=lambda: stops.append(1),
        pending_fn=pending_fn,
        min_queue=5,
        ttl_s=300.0,
        now_fn=lambda: t[0],
    )
    st.stop_idle_once()          # prime: first sighting
    t[0] = 1_000.0               # idle well past the TTL, counter unmoved
    st.stop_idle_once()
    return stops
