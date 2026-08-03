"""DEEP AUDIT — the full node_job → claim-eligibility and queue-ORDER matrix.

WHY THIS EXISTS. A node_job's fate is decided by SEVEN independent attributes that
all meet in one SQL statement: ``queue``, ``required_model``, ``server_types``,
``node_module`` (vs the pool allow-list), ``avoid_box``/``force_box``, ``priority``,
``is_priority`` — crossed with the claiming worker's own ``served_types``,
``known_models``, ``box``, ``current_model`` and warm state, and the parent run's
status. The existing suites each pin ONE of those axes. Nothing pinned the
INTERACTIONS, and that gap is not theoretical: migration 0022 shipped two live bugs
in one afternoon, both in cells no test covered —

  * a remote-served row whose module was not pool-listed was claimable by NEITHER
    lane (the two lanes' predicates were each individually correct);
  * warm affinity leaked onto rows that never opted in, silently outranking the
    operator's priority bands.

So this module is deliberately COMBINATORIAL rather than example-based. Two halves:

  1. **ELIGIBILITY** — for every meaningful row shape, assert exactly WHICH lane(s)
     can claim it, and assert the union is never empty for a row that any worker
     should be able to run. "Claimable by nobody" is the failure mode that has no
     error message and no alert; it is what this half exists to make impossible.
  2. **ORDER** — the claim ORDER BY is a 5-level sort. Each level is pinned against
     the level below it, so a future reordering cannot silently change precedence.
"""
from __future__ import annotations

import itertools

import pytest

from queue_workflows import node_queue
from tests._helpers import make_run

POOL = ("pooled_llm", "pooled_vlm")
BOX = "box-a"
HOST = "box-a-gpu"


def _q(**kw):
    """Enqueue one gpu node-job; every axis defaulted to 'unconstrained'."""
    kw.setdefault("node_module", "pooled_llm")
    return node_queue.enqueue_node_job(
        run_id=make_run(), node_id=f"n-{kw.get('node_module')}-{id(kw)}",
        queue="gpu", **kw,
    )


def _inline(**kw):
    kw.setdefault("served_types", ["ollama"])
    kw.setdefault("known_models", None)
    return node_queue.claim_next_gpu_job(
        0, kw.pop("current_model", None), host=HOST, box=kw.pop("box", BOX),
        require_model=True, pool_modules=POOL, **kw,
    )


def _pool(**kw):
    kw.setdefault("served_types", ["ollama"])
    return node_queue.claim_next_gpu_job(
        0, None, host=HOST, box=kw.pop("box", BOX),
        require_model=False, pool_modules=POOL, **kw,
    )


# ═══ HALF 1 — ELIGIBILITY ════════════════════════════════════════════════════
#
# The core invariant: INLINE and POOL are DISJOINT (never both — that would be a
# double-claim) and their UNION covers every runnable row (never neither — that is
# the silent-stranding bug).

_MODEL = [None, "chat-model-a:latest"]
_SERVERS = [None, ["ollama"], ["vllm", "ollama"]]
_MODULE = ["pooled_llm", "heavy_inproc"]


@pytest.mark.parametrize("model,servers,module", itertools.product(_MODEL, _SERVERS, _MODULE))
def test_every_row_shape_is_claimable_by_exactly_one_lane(model, servers, module):
    """The 2x3x2 core matrix. For EVERY shape: exactly one lane claims it.

    A row claimable by BOTH lanes would let one box run it twice (the lanes are
    separate claim calls). A row claimable by NEITHER sits queued forever with no
    error — the failure mode that cost 8 live jobs."""
    jid = _q(required_model=model, server_types=servers, node_module=module)

    got_inline = _inline(known_models=[model] if model else None)
    inline_took = got_inline is not None and got_inline["id"] == jid
    if inline_took:                       # put it back so the pool lane sees a fair queue
        node_queue.requeue_job_for_retry(jid)

    got_pool = _pool()
    pool_took = got_pool is not None and got_pool["id"] == jid

    assert inline_took != pool_took, (
        f"shape(model={model}, server_types={servers}, module={module}) — "
        f"inline={inline_took} pool={pool_took}; must be exactly one "
        f"({'BOTH lanes = double-claim' if inline_took and pool_took else 'NEITHER = silently stranded'})"
    )


@pytest.mark.parametrize("servers", [["ollama"], ["vllm", "ollama"]])
def test_remote_served_always_lands_in_the_pool_whatever_the_module(servers):
    """server_types is the authority on 'loads nothing in-process'; the pool
    allow-list must not get a second vote on it."""
    jid = _q(required_model="chat-model-a:latest", server_types=servers,
             node_module="module_nobody_allow_listed")
    got = _pool()
    assert got is not None and got["id"] == jid


def test_in_process_model_row_never_reaches_the_pool():
    """The converse: an in-process model job in the PAR-wide pool would put two
    models on one card."""
    _q(required_model="sdxl", node_module="pooled_llm")
    assert _pool() is None


# ── the worker-side axes ─────────────────────────────────────────────────────


@pytest.mark.parametrize("served,prefs,claimable", [
    # NOTE: prefs=None is deliberately absent — an unconstrained MODEL row is an
    # INLINE row, so asking the pool lane about it tests the lane split, not the
    # preference gate. Its eligibility is pinned by the core matrix above.
    (["ollama"],          ["ollama"],         True),
    (["ollama"],          ["vllm", "ollama"], True),   # ordered fallback
    (["ollama"],          ["vllm"],           False),  # no overlap
    (["ollama", "vllm"],  ["vllm"],           True),
    ([],                  ["vllm"],           True),   # cold worker ⇒ gate OFF
    (None,                ["vllm"],           True),
])
def test_server_preference_gate(served, prefs, claimable):
    jid = _q(required_model="chat-model-a:latest", server_types=prefs)
    got = _pool(served_types=served)
    assert (got is not None and got["id"] == jid) is claimable


@pytest.mark.parametrize("avoid,force,box,ok", [
    (None,      None,      BOX,     True),
    ([BOX],     None,      BOX,     False),
    (["other"], None,      BOX,     True),
    (None,      [BOX],     BOX,     True),
    (None,      ["other"], BOX,     False),
    ([BOX],     [BOX],     BOX,     False),   # avoid wins over force
    (None,      [BOX],     None,    False),   # box-unaware worker skips constrained
    (None,      None,      None,    True),
])
def test_placement_still_composes_with_a_server_preference(avoid, force, box, ok):
    """0020 placement and 0022 preference are ANDed; neither may swallow the other."""
    jid = _q(required_model="chat-model-a:latest", server_types=["ollama"],
             avoid_box=avoid, force_box=force)
    got = _pool(box=box)
    assert (got is not None and got["id"] == jid) is ok


def test_a_row_whose_run_is_cancelled_is_never_claimable_by_either_lane():
    from queue_workflows import run_store
    run_id = make_run()
    jid = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="pooled_llm", queue="gpu",
        required_model="chat-model-a:latest", server_types=["ollama"],
    )
    run_store.update_run(run_id, status="cancelled")
    assert _pool() is None
    assert _inline(known_models=["chat-model-a:latest"]) is None
    assert node_queue.get_node_job(jid)["status"] == "queued"


def test_box_slot_arbiter_still_caps_a_remote_served_lane():
    """max_running must bound the pool lane too, or per-job routing would let the
    box over-claim past its advertised concurrency."""
    for _ in range(3):
        _q(required_model="chat-model-a:latest", server_types=["ollama"])
    first = _pool(max_running=1)
    assert first is not None
    assert _pool(max_running=1) is None, "the box already has 1 running"


# ═══ HALF 2 — QUEUE ORDER ════════════════════════════════════════════════════
#
# The sort is: is_priority DESC → in-process affinity → warm(server,model) →
# priority ASC → host-direction/created. Each test pins ONE boundary by making the
# LOWER level argue for the opposite row.


def _two(a_kw, b_kw):
    return _q(**a_kw), _q(**b_kw)


def test_run_next_flag_outranks_everything_below_it():
    warm = _q(required_model="chat-model-a:latest", server_types=["ollama"], priority=1)
    urgent = _q(required_model="cold", server_types=["vllm", "ollama"], priority=200)
    node_queue.prioritize_node_job(urgent)
    got = _pool(warm_servers=["ollama"], warm_models=["chat-model-a:latest"])
    assert got["id"] == urgent
    assert got["id"] != warm


def test_warm_affinity_outranks_priority_band_for_opted_in_rows():
    _q(required_model="cold", server_types=["ollama"], priority=1)
    warm = _q(required_model="chat-model-a:latest", server_types=["ollama"], priority=200)
    got = _pool(warm_servers=["ollama"], warm_models=["chat-model-a:latest"])
    assert got["id"] == warm


def test_warm_affinity_NEVER_outranks_priority_for_legacy_rows():
    """The regression, pinned. A row with no server_types must sort on
    priority alone however warm the box is."""
    urgent = _q(required_model="chat-model-a:latest", node_module="heavy_inproc", priority=1)
    _q(required_model="chat-model-a:latest", node_module="heavy_inproc", priority=200)
    got = _inline(known_models=["chat-model-a:latest"],
                  warm_servers=["ollama"], warm_models=["chat-model-a:latest"])
    assert got["id"] == urgent


def test_server_match_outranks_model_match():
    """A server cold-start (vLLM: minutes) costs more than a weight swap (seconds),
    so the row whose SERVER is already up wins."""
    model_only = _q(required_model="chat-model-a:latest", server_types=["vllm"])
    server_only = _q(required_model="cold", server_types=["ollama"])
    got = _pool(served_types=["ollama", "vllm"],
                warm_servers=["ollama"], warm_models=["chat-model-a:latest"])
    assert got["id"] == server_only
    assert got["id"] != model_only


def test_priority_band_decides_when_nothing_is_warm():
    hi = _q(required_model="a", server_types=["ollama"], priority=5)
    _q(required_model="b", server_types=["ollama"], priority=200)
    got = _pool(warm_servers=[], warm_models=[])
    assert got["id"] == hi


def test_creation_order_breaks_a_full_tie():
    first = _q(required_model="a", server_types=["ollama"], priority=100)
    _q(required_model="a", server_types=["ollama"], priority=100)
    got = _pool(warm_servers=[], warm_models=[])
    assert got["id"] == first


def test_affinity_is_never_a_filter_only_a_sort():
    """The whole queue is unwarm and unmatched — it must still drain."""
    jid = _q(required_model="nothing-warm", server_types=["vllm", "ollama"])
    got = _pool(warm_servers=["something-else"], warm_models=["other-model"])
    assert got is not None and got["id"] == jid


# ═══ HALF 3 — THE DEFER VIEW MUST AGREE WITH THE CLAIM ═══════════════════════
#
# Found LIVE: a box deferred forever while 2579 remote-served jobs sat queued and
# vLLM served zero requests. The claim SQL knew they were pool work; the defer view
# still tested `required_model IS NULL`, so it saw none of them — peers read as idle
# AND the anti-starvation break never armed. Same class as the
# defer-standstill: two places defining "claimable" differently.


def _peer(host, *, par=4, project=""):
    """A fresher, higher-ranked peer so the defer heuristic has someone to defer to."""
    node_queue.upsert_worker_heartbeat(
        host_label=host, queue="gpu", concurrency=par, project=project)


def test_an_aging_REMOTE_SERVED_job_breaks_the_deferral():
    """The live bug, pinned. Before the fix this returned True forever."""
    import time
    _peer("aaa-top-gpu", par=8)
    jid = _q(required_model="chat-model-a:latest", server_types=["vllm", "ollama"])
    # Backdate past the anti-starvation window.
    node_queue.backdate_job_created_at(jid, seconds=120) if hasattr(
        node_queue, "backdate_job_created_at") else None
    from queue_workflows.db import connection
    from queue_workflows.dialect import get_dialect
    with connection() as c, c.cursor() as cur:
        cur.execute(
            f"UPDATE workflow_node_jobs SET created_at = {get_dialect().past_seconds('%(s)s')} "
            f"WHERE id = %(id)s", {"s": 120, "id": jid})
    assert node_queue.vlm_pool_should_defer("zzz-me-gpu", 1, box=BOX) is False


def test_a_running_REMOTE_SERVED_job_counts_against_a_peers_capacity():
    """A peer busy with remote-served work is NOT free capacity to defer to."""
    _peer("aaa-top-gpu", par=1)
    jid = _q(required_model="chat-model-a:latest", server_types=["ollama"])
    node_queue.claim_next_gpu_job(
        0, None, host="aaa-top-gpu", box="aaa-top", require_model=False,
        pool_modules=POOL, served_types=["ollama"])
    # The top peer's only slot is occupied by a remote-served job → nothing to
    # defer to → claim here instead.
    assert node_queue.vlm_pool_should_defer("zzz-me-gpu", 1, box=BOX) is False
