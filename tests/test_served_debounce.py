"""The advertised server set must not flap on one missed probe.

Everything hangs off `_box_served`: `_gpu_slot_budget` (ollama→1, vllm→PAR), the
DB-side box-slot arbiter, the published heartbeat `concurrency`, and — through it —
the fill-before-spill defer ranking every peer reads. One timed-out vLLM GET used to
collapse the set for a tick: budget 3→1 against 3 admitted jobs (the panel's
"NODE JOBS ⚠ 3/1"), the arbiter refusing every new claim, peers mis-ranking the box.

Rule: a kind JOINS on first sighting; it LEAVES only after N consecutive misses.
And the heartbeat row must pair the concurrency with the SAME tick's served list.
"""
from __future__ import annotations

from queue_workflows import claim_worker, llm_probe


def _worker():
    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-a-gpu"
            self.queue = "gpu"
    w = _W()
    w._pool_parallelism = lambda: 3
    return w


def _drive(w, monkeypatch, answers):
    """One probe tick where `answers` maps url-kind → what probe_llm_servers says."""
    from queue_workflows.llm_backends import factory as f
    monkeypatch.setattr(f, "resolve_base_url",
                        lambda kind="ollama": f"http://{kind}:1")
    monkeypatch.setattr(llm_probe, "probe_llm_servers",
                        lambda url, **k: answers.get(url.split("//")[1].split(":")[0], []))
    monkeypatch.setattr(llm_probe, "probe_gpu_placement", lambda url, **k: llm_probe.GPU)
    monkeypatch.setattr(llm_probe, "describe_endpoint", lambda *a, **k: "x")
    return w._probe_llm_servers()


def test_one_missed_probe_does_not_drop_the_kind(monkeypatch):
    w = _worker()
    _drive(w, monkeypatch, {"ollama": ["ollama"], "vllm": ["vllm"]})
    # ONE SERVING KIND: while vLLM serves it is the only kind offered —
    # ollama's always-up daemon stays in the DEBOUNCED set but is not advertised, or an
    # ollama-only job lands here and loads a second copy beside vLLM's pool.
    assert w._box_served == ["vllm"]
    assert w._gpu_slot_budget() == 3

    _drive(w, monkeypatch, {"ollama": ["ollama"], "vllm": []})      # miss 1
    assert w._box_served == ["vllm"], "one miss must not collapse the set"
    assert w._gpu_slot_budget() == 3, "…nor the budget the arbiter enforces"

    _drive(w, monkeypatch, {"ollama": ["ollama"], "vllm": []})      # miss 2
    assert w._box_served == ["vllm"]


def test_sustained_misses_do_drop_it():
    """Debounce, not denial — a genuinely stopped server must still leave."""
    import pytest
    w = _worker()
    mp = pytest.MonkeyPatch()
    try:
        _drive(w, mp, {"ollama": ["ollama"], "vllm": ["vllm"]})
        for _ in range(claim_worker._SERVED_DEBOUNCE_MISSES):
            _drive(w, mp, {"ollama": ["ollama"], "vllm": []})
        assert w._box_served == ["ollama"]
        assert w._gpu_slot_budget() == 1
    finally:
        mp.undo()


def test_a_kind_rejoins_on_first_sighting(monkeypatch):
    w = _worker()
    _drive(w, monkeypatch, {"ollama": ["ollama"], "vllm": ["vllm"]})
    for _ in range(claim_worker._SERVED_DEBOUNCE_MISSES):
        _drive(w, monkeypatch, {"ollama": ["ollama"], "vllm": []})
    assert w._box_served == ["ollama"]
    _drive(w, monkeypatch, {"ollama": ["ollama"], "vllm": ["vllm"]})
    assert w._box_served == ["vllm"], (
        "a server that answers is up — and vLLM, once up, is the only kind offered")
    assert w._gpu_slot_budget() == 3


def test_heartbeat_row_pairs_concurrency_with_the_SAME_ticks_served_list(monkeypatch):
    """kwargs evaluate left-to-right, and concurrency was computed
    BEFORE the servers probe that feeds it — every published row paired tick N-1's
    concurrency with tick N's served list. That two-tick skew is the exact
    'adv=1 served={ollama,vllm}' row an operator panel would show."""
    from queue_workflows import node_queue

    state = {"served": ["ollama"]}

    def _servers():
        state["served"] = ["ollama", "vllm"]          # the probe changes the world…
        return list(state["served"])

    def _concurrency():
        return 3 if "vllm" in state["served"] else 1  # …the budget derives from it

    captured = {}
    monkeypatch.setattr(node_queue, "upsert_worker_heartbeat",
                        lambda **kw: captured.update(kw))
    emitter = claim_worker.HeartbeatEmitter(
        host_label="box-a-gpu", queue="gpu",
        concurrency_fn=_concurrency, llm_servers_fn=_servers,
    )
    emitter.emit_once()
    assert captured["llm_servers_available"] == ["ollama", "vllm"]
    assert captured["concurrency"] == 3, (
        "the row must describe ONE observation — concurrency computed after the probe")
