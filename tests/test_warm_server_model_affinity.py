"""Warm **(server, model)** affinity — "switch servers and weights as rarely as
possible" (migration 0022).

The GPU claim sorts candidate rows by what is ALREADY loaded on the claiming box,
so the queue naturally drains in an order that avoids restarts:

    3 = server kind up AND model resident   (nothing to do)
    2 = server kind up, different model     (avoids a server start)
    1 = model resident, different server    (avoids a weight load)
    0 = neither

The server term outweighs the model term because a server cold start (vLLM: minutes)
dominates a weight swap (ollama: seconds).

This is a TIEBREAK, never a filter — a cold box with nothing warm must still claim,
or the fleet deadlocks the first time every box is empty.
"""
from __future__ import annotations

from queue_workflows import node_queue
from tests._helpers import make_run

POOL = ("llm_chat",)


def _queue(node_id: str, *, server_types=None, required_model=None, priority=100) -> str:
    return node_queue.enqueue_node_job(
        run_id=make_run(), node_id=node_id, node_module="llm_chat", queue="gpu",
        required_model=required_model, server_types=server_types, priority=priority,
    )


def _claim(*, warm_servers=None, warm_models=None):
    return node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=False, pool_modules=POOL,
        served_types=["ollama", "vllm"],
        warm_servers=warm_servers, warm_models=warm_models,
    )


def test_both_warm_beats_server_only():
    cold = _queue("cold", server_types=["vllm"], required_model="other")
    hot = _queue("hot", server_types=["vllm"], required_model="chat-model-a:latest")
    got = _claim(warm_servers=["vllm"], warm_models=["chat-model-a:latest"])
    assert got["id"] == hot, "the row needing neither a start nor a load must go first"
    assert got["id"] != cold


def test_server_match_beats_model_match():
    # A weight load is cheaper than a server start, so the row whose SERVER is
    # already up wins even though the other row's model is resident.
    model_only = _queue("model-only", server_types=["vllm"], required_model="chat-model-a:latest")
    server_only = _queue("server-only", server_types=["ollama"], required_model="other")
    got = _claim(warm_servers=["ollama"], warm_models=["chat-model-a:latest"])
    assert got["id"] == server_only
    assert got["id"] != model_only


def test_warm_row_preempts_a_colder_row_queued_first():
    _queue("first-but-cold", server_types=["vllm"], required_model="other")
    warm = _queue("second-but-warm", server_types=["ollama"], required_model="chat-model-a:latest")
    got = _claim(warm_servers=["ollama"], warm_models=["chat-model-a:latest"])
    assert got["id"] == warm, "affinity outranks creation order within a priority band"


def test_affinity_is_a_tiebreak_not_a_filter_on_a_cold_box():
    jid = _queue("only", server_types=["ollama"], required_model="chat-model-a:latest")
    got = _claim(warm_servers=[], warm_models=[])
    assert got is not None and got["id"] == jid


def test_nothing_warm_passed_at_all_still_claims():
    jid = _queue("only", server_types=["ollama"], required_model="chat-model-a:latest")
    got = _claim()
    assert got is not None and got["id"] == jid


def test_operator_run_next_still_outranks_affinity():
    """`is_priority` is the operator's "run this now" escape hatch — it must keep
    beating affinity, accepting the switch cost on purpose."""
    warm = _queue("warm", server_types=["ollama"], required_model="chat-model-a:latest")
    urgent = _queue("urgent", server_types=["vllm"], required_model="other")
    node_queue.prioritize_node_job(urgent)
    got = _claim(warm_servers=["ollama"], warm_models=["chat-model-a:latest"])
    assert got["id"] == urgent
    assert got["id"] != warm


def test_unconstrained_rows_score_zero_and_do_not_crash_the_sort():
    jid = _queue("plain")
    got = _claim(warm_servers=["ollama"], warm_models=["chat-model-a:latest"])
    assert got is not None and got["id"] == jid


def test_legacy_rows_are_NOT_reordered_by_warm_affinity():
    """REGRESSION (audit D2). The warm term sorts ABOVE `priority ASC`, so if it
    scored rows that never opted in, warm-model affinity would silently outrank the
    operator's priority bands for every pre-0022 row in production.

    A row with `server_types IS NULL` must score 0 no matter what is warm — strict
    priority order, byte-identical to before the migration."""
    low = node_queue.enqueue_node_job(
        run_id=make_run(), node_id="urgent", node_module="llm_chat", queue="gpu",
        required_model="other", priority=10,          # better band, model NOT warm
    )
    node_queue.enqueue_node_job(
        run_id=make_run(), node_id="warm-but-later", node_module="llm_chat", queue="gpu",
        required_model="chat-model-a:latest", priority=100,   # worse band, model IS warm
    )
    # Legacy model-rows run on the INLINE lane — exactly where the regression bit.
    got = node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=True, pool_modules=POOL,
        served_types=["ollama"],
        warm_servers=["ollama"], warm_models=["chat-model-a:latest"],
    )
    assert got is not None
    assert got["id"] == low, "priority must still win for rows with no server_types"


def test_opted_in_rows_ARE_still_reordered():
    """The other side of the same line — the feature must still work where asked."""
    node_queue.enqueue_node_job(
        run_id=make_run(), node_id="cold", node_module="llm_chat", queue="gpu",
        required_model="other", server_types=["ollama"], priority=10,
    )
    warm = node_queue.enqueue_node_job(
        run_id=make_run(), node_id="warm", node_module="llm_chat", queue="gpu",
        required_model="chat-model-a:latest", server_types=["ollama"], priority=100,
    )
    got = _claim(warm_servers=["ollama"], warm_models=["chat-model-a:latest"])
    assert got["id"] == warm
