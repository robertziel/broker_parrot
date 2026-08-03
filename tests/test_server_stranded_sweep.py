"""A row no live server can satisfy must be FLAGGED, not silent (analysis #3).

A ``['vllm']``-only job on a fleet where no vLLM answers is claimable by nobody — it
used to sit ``queued`` forever with no error, no flag and no log. Same failure mode
the 0015 capacity sweep names for VRAM, so it gets the same treatment: the
``unassignable`` stamp, cleared the moment the condition lifts.
"""
from __future__ import annotations

from queue_workflows import node_queue
from tests._helpers import make_run


def _hb(host, servers):
    node_queue.upsert_worker_heartbeat(
        host_label=host, queue="gpu", concurrency=1, llm_servers_available=servers)


def _job(servers, model="chat-model-a:latest"):
    return node_queue.enqueue_node_job(
        run_id=make_run(), node_id=f"n{id(servers)}", node_module="llm_chat",
        queue="gpu", required_model=model, server_types=servers)


def test_a_vllm_only_row_on_an_ollama_fleet_is_flagged():
    _hb("a-gpu", ["ollama"])
    jid = _job(["vllm"])
    newly = node_queue.flag_server_stranded_jobs()
    assert [r["id"] for r in newly] == [jid]
    row = node_queue.get_node_job(jid)
    assert row["unassignable_at"] is not None
    assert "vllm" in (row["unassignable_reason"] or "")


def test_an_ordered_preference_with_a_live_fallback_is_NOT_flagged():
    """['vllm','ollama'] on an ollama fleet is claimable — that is the whole point
    of the ordered fallback."""
    _hb("a-gpu", ["ollama"])
    _job(["vllm", "ollama"])
    assert node_queue.flag_server_stranded_jobs() == []


def test_the_flag_clears_when_a_matching_server_appears():
    _hb("a-gpu", ["ollama"])
    jid = _job(["vllm"])
    node_queue.flag_server_stranded_jobs()
    _hb("b-gpu", ["ollama", "vllm"])            # a vLLM comes up somewhere
    node_queue.flag_server_stranded_jobs()
    assert node_queue.get_node_job(jid)["unassignable_at"] is None


def test_zero_fresh_workers_is_fleet_down_not_a_routing_verdict():
    jid = _job(["vllm"])
    assert node_queue.flag_server_stranded_jobs() == []
    assert node_queue.get_node_job(jid)["unassignable_at"] is None


def test_already_flagged_rows_are_not_re_returned():
    """One event per episode, not one per sweep tick."""
    _hb("a-gpu", ["ollama"])
    _job(["vllm"])
    assert len(node_queue.flag_server_stranded_jobs()) == 1
    assert node_queue.flag_server_stranded_jobs() == []


def test_unconstrained_rows_are_never_flagged_by_this_sweep():
    _hb("a-gpu", ["ollama"])
    node_queue.enqueue_node_job(
        run_id=make_run(), node_id="plain", node_module="llm_chat", queue="gpu")
    assert node_queue.flag_server_stranded_jobs() == []


def test_the_two_sweeps_do_not_clear_each_others_flags():
    """ROUND-3 MED: the 0015 capacity CLEAR un-flagged stranded rows whose model
    happened to FIT a live worker, and the pair thrashed — 5 events in 5 tick-pairs
    against the one-event-per-episode contract. Each sweep now owns its reason."""
    _hb("a-gpu", ["ollama"])
    jid = _job(["vllm"])
    assert len(node_queue.flag_server_stranded_jobs()) == 1
    for _ in range(4):
        node_queue.flag_unassignable_gpu_jobs()      # the capacity sweep runs first…
        newly = node_queue.flag_server_stranded_jobs()
        assert newly == [], "the stranded flag must be STABLE across tick pairs"
    assert node_queue.get_node_job(jid)["unassignable_at"] is not None


def test_a_serverless_advertising_worker_unstrands_the_row():
    """ROUND-3 LOW: a fresh worker advertising [] has the claim gate OFF and WILL
    claim the row — flagging it 'stranded' while a worker is about to run it lies."""
    node_queue.upsert_worker_heartbeat(
        host_label="cold-gpu", queue="gpu", concurrency=1, llm_servers_available=[])
    _job(["vllm"])
    assert node_queue.flag_server_stranded_jobs() == []
