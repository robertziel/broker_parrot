"""A thin GPU worker must not run LLM work on CPU — only on GPU.

The complement to ``test_gpu_blind_refuse``: that guards a torch-blind WORKER
container; this guards the case where the worker is fine but its external ollama
SERVER fell back to CPU (an NVML/cgroup GPU loss, or the model didn't fit VRAM).
ollama still ANSWERS, so the existence probe says "ollama" — but ``ollama ps`` reads
"100% CPU", and every LLM-dispatch job the box claims runs at CPU speed.

Policy: ONLY use a box on GPU. The pool lane (no-model
LLM-dispatch jobs) SKIPS claiming while the server is on CPU, and resumes when it
returns to GPU — a dynamic, self-recovering gate (no restart needed). The only
other skips are insufficient VRAM (routed to a fitting box) and an operator OFF
toggle (the worker-control park path); a CPU fallback folds into "can't serve on
GPU", so it is skipped the same way.
"""

from __future__ import annotations

from queue_workflows import claim_worker, llm_probe, model_registry


class _FakeCache:
    current_model = None


def _gpu_worker(host="box-a-gpu"):
    return claim_worker.ClaimWorker(queue="gpu", host=host, model_cache=_FakeCache())


# ── the claim gate: _claim_pool skips when the server is on CPU ───────────────


def test_pool_claim_skipped_when_llm_on_cpu(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        claim_worker.node_queue, "claim_next_gpu_job",
        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or {"id": "x"},
    )
    monkeypatch.setattr(model_registry, "known_ids", lambda: [])
    w = _gpu_worker()
    w._llm_gpu_ok = False  # server is serving on CPU
    assert w._claim_pool() is None
    assert calls["n"] == 0  # must NOT even reach the DB claim


def test_pool_claim_proceeds_when_llm_on_gpu(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        claim_worker.node_queue, "claim_next_gpu_job",
        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or {"id": "job1"},
    )
    monkeypatch.setattr(model_registry, "known_ids", lambda: [])
    w = _gpu_worker()
    w._llm_gpu_ok = True
    assert w._claim_pool() == {"id": "job1"}
    assert calls["n"] == 1


def test_llm_gpu_ok_defaults_true_so_startup_is_not_blocked():
    # Before the first heartbeat probe, the worker must be free to claim (a cold
    # ollama has no loaded model to judge; it will load on GPU if it can).
    assert _gpu_worker()._llm_gpu_ok is True


# ── the heartbeat probe drives the flag + honest advertisement ───────────────


def _wire_probe(monkeypatch, *, servers, placement):
    from queue_workflows.llm_backends import factory
    monkeypatch.setattr(factory, "resolve_base_url", lambda *a, **k: "http://box:11434")
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda *a, **k: list(servers))
    monkeypatch.setattr(llm_probe, "probe_gpu_placement", lambda *a, **k: placement)


def test_probe_sets_gpu_ok_false_and_advertises_empty_on_cpu(monkeypatch):
    _wire_probe(monkeypatch, servers=["ollama"], placement="cpu")
    w = _gpu_worker()
    advertised = w._probe_llm_servers()
    assert w._llm_gpu_ok is False
    # a CPU-bound server is NOT a usable GPU LLM server → advertise nothing, so the
    # panel shows no "OLLAMA · ON" chip and the box reads as not-serving.
    assert advertised == []


def test_probe_sets_gpu_ok_true_and_advertises_server_on_gpu(monkeypatch):
    _wire_probe(monkeypatch, servers=["ollama"], placement="gpu")
    w = _gpu_worker()
    assert w._probe_llm_servers() == ["ollama"]
    assert w._llm_gpu_ok is True


def test_probe_optimistic_on_unknown_cold_server(monkeypatch):
    # Nothing loaded yet → placement unknown → still usable (claim, load on GPU).
    _wire_probe(monkeypatch, servers=["ollama"], placement="unknown")
    w = _gpu_worker()
    assert w._probe_llm_servers() == ["ollama"]
    assert w._llm_gpu_ok is True


def test_probe_gpu_ok_false_when_no_server_at_all(monkeypatch):
    _wire_probe(monkeypatch, servers=[], placement="unknown")
    w = _gpu_worker()
    assert w._probe_llm_servers() == []
    assert w._llm_gpu_ok is False


def test_probe_vllm_is_gpu_without_placement_call(monkeypatch):
    # vLLM is CUDA-only: a live vLLM is GPU by construction, no /api/ps probe.
    from queue_workflows.llm_backends import factory
    monkeypatch.setattr(factory, "resolve_base_url", lambda *a, **k: "http://box:8000")
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda *a, **k: ["vllm"])

    def _should_not_be_called(*a, **k):
        raise AssertionError("placement probe must not run for vllm")

    monkeypatch.setattr(llm_probe, "probe_gpu_placement", _should_not_be_called)
    w = _gpu_worker()
    assert w._probe_llm_servers() == ["vllm"]
    assert w._llm_gpu_ok is True
