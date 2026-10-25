"""A box that serves vLLM offers vLLM — not ollama as well. That is how it got two copies.

THE FAILURE. The switch to vLLM unloads ollama's MODELS,
not its daemon, so after the switch both servers answered their probes and the box
advertised ``{ollama, vllm}``. A job carrying ``server_types=['ollama']`` then passed the
preference gate, landed on the box, and its first request loaded a second full copy of
the same model into the idle ollama beside vLLM's pool: all memory committed, no ssh, the
panel's last reading frozen before the hang. The residency enforcer is an after-the-fact evictor;
by the time it could vote, the memory was gone.

vLLM pins its weights for its whole lifetime, so while it serves it is the ONLY kind this
box offers. ollama comes back the moment vLLM is gone — it never left the debounced set.
"""
from __future__ import annotations

import pytest

from queue_workflows import claim_worker, llm_probe, node_queue
from tests._helpers import make_run


def _worker():
    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-c-gpu"
            self.queue = "gpu"
    w = _W()
    w._pool_parallelism = lambda: 3
    return w


def _probe(w, monkeypatch, answers):
    from queue_workflows.llm_backends import factory as f
    monkeypatch.setattr(f, "resolve_base_url", lambda kind="ollama": f"http://{kind}:1")
    monkeypatch.setattr(llm_probe, "probe_llm_servers",
                        lambda url, **k: answers.get(url.split("//")[1].split(":")[0], []))
    monkeypatch.setattr(llm_probe, "probe_gpu_placement", lambda url, **k: llm_probe.GPU)
    monkeypatch.setattr(llm_probe, "describe_endpoint", lambda *a, **k: "x")
    return w._probe_llm_servers()


def test_with_vllm_serving_only_vllm_is_offered(monkeypatch):
    w = _worker()
    assert _probe(w, monkeypatch, {"ollama": ["ollama"], "vllm": ["vllm"]}) == ["vllm"]
    assert w._served_llm_types() == ["vllm"]


def test_ollama_alone_is_offered_exactly_as_before(monkeypatch):
    w = _worker()
    assert _probe(w, monkeypatch, {"ollama": ["ollama"], "vllm": []}) == ["ollama"]


def test_ollama_returns_once_vllm_has_really_gone(monkeypatch):
    """vLLM keeps its exclusivity through the debounce window — a busy vLLM that misses
    one probe must not let an ollama job in beside it — and ollama, which never left
    the debounced set, is offered again the moment vLLM is dropped from it."""
    w = _worker()
    _probe(w, monkeypatch, {"ollama": ["ollama"], "vllm": ["vllm"]})
    assert _probe(w, monkeypatch, {"ollama": ["ollama"], "vllm": []}) == ["vllm"]
    for _ in range(claim_worker._SERVED_DEBOUNCE_MISSES - 1):
        _probe(w, monkeypatch, {"ollama": ["ollama"], "vllm": []})
    assert w._served_llm_types() == ["ollama"]


def test_the_slot_budget_is_vllms_while_it_serves(monkeypatch):
    """Advertising only vLLM must not shrink the box to ollama's width-1."""
    w = _worker()
    _probe(w, monkeypatch, {"ollama": ["ollama"], "vllm": ["vllm"]})
    assert w._gpu_slot_budget() == 3


def test_an_ollama_only_job_cannot_be_claimed_by_a_box_serving_vllm():
    """The gate that let the storm judge onto box-c, with the box's advertisement
    fixed: {ollama} ∩ {vllm} is empty, so the row waits for a box that serves ollama."""
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="judge", node_module="llm.llm_storm_judge", queue="gpu",
        server_types=["ollama"])
    assert node_queue.claim_next_gpu_job(0, None, host="box-c-gpu",
                                         served_types=["vllm"]) is None
    claimed = node_queue.claim_next_gpu_job(0, None, host="box-a-gpu",
                                            served_types=["ollama"])
    assert claimed and claimed["id"] == job_id
