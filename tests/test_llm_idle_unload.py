"""Unload the EXTERNAL LLM server's model after the GPU box has been idle a while.

The engine's ``ModelCache`` reaper (model_cache.py) unloads a warm IN-PROCESS model
on idle, and the vLLM ``LLMSupervisor`` stops the vLLM sidecar on idle — but an
ollama box's model is left to ollama's own ``OLLAMA_KEEP_ALIVE`` timer, engine-
uncontrolled, so it can sit in VRAM/RAM long after the box goes quiet. This reaper
closes that: when the GPU worker has done no GPU work for ``ttl_s`` (default 5 min),
it asks the box's ollama what it's holding and unloads it, giving the RAM back.

Shape mirrors ``model_cache.gpu_should_unload`` / ``reap_idle_once`` on purpose:
a pure decision + a one-tick reaper + a clamped-cadence daemon, all I/O injected so
the whole thing runs on a virtual clock with a fake ollama.
"""

from __future__ import annotations

import pytest

from queue_workflows import llm_idle_unload, llm_probe


# ── the pure decision ─────────────────────────────────────────────────────────


def test_should_unload_only_when_loaded_quiet_and_idle_past_ttl():
    assert llm_idle_unload.should_unload_external_model(True, 0, 300.0, 300.0) is True
    assert llm_idle_unload.should_unload_external_model(True, 0, 999.0, 300.0) is True


@pytest.mark.parametrize("loaded,active,idle,ttl", [
    (False, 0, 999.0, 300.0),   # nothing loaded → nothing to free
    (True, 1, 999.0, 300.0),    # a job is running → the GPU IS in use
    (True, 0, 120.0, 300.0),    # idle, but not long enough yet
    (True, 0, 999.0, 0.0),      # ttl<=0 disables the reaper entirely
    (True, 0, 999.0, -1.0),
])
def test_should_not_unload(loaded, active, idle, ttl):
    assert llm_idle_unload.should_unload_external_model(loaded, active, idle, ttl) is False


def test_default_ttl_is_five_minutes(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S", raising=False)
    import importlib
    m = importlib.reload(llm_idle_unload)
    assert m.DEFAULT_LLM_IDLE_TTL_S == 300.0


# ── the reaper tick ───────────────────────────────────────────────────────────


def _reaper(*, active=0, idle=999.0, loaded=("llama3",), ttl=300.0, unloaded=None):
    calls = {"loaded_q": 0, "unload": []}

    def loaded_models_fn():
        calls["loaded_q"] += 1
        return list(loaded)

    def unload_fn(models):
        calls["unload"].append(list(models))
        if unloaded is not None:
            unloaded.extend(models)

    r = llm_idle_unload.ExternalModelIdleReaper(
        active_fn=lambda: active,
        idle_seconds_fn=lambda: idle,
        loaded_models_fn=loaded_models_fn,
        unload_fn=unload_fn,
        ttl_s=ttl,
        label="box-a-gpu",
    )
    return r, calls


def test_reap_unloads_when_idle_and_loaded():
    r, calls = _reaper(active=0, idle=400.0, loaded=("llama3", "qwen"))
    assert r.reap_idle_once() == ["llama3", "qwen"]
    assert calls["unload"] == [["llama3", "qwen"]]


def test_reap_skips_ollama_query_while_busy():
    # active>0 is a cheap LOCAL check — must short-circuit BEFORE any HTTP to ollama.
    r, calls = _reaper(active=2, idle=999.0)
    assert r.reap_idle_once() == []
    assert calls["loaded_q"] == 0


def test_reap_skips_ollama_query_before_ttl():
    r, calls = _reaper(active=0, idle=60.0, ttl=300.0)
    assert r.reap_idle_once() == []
    assert calls["loaded_q"] == 0


def test_reap_queries_but_no_model_loaded_is_a_noop():
    r, calls = _reaper(active=0, idle=999.0, loaded=())
    assert r.reap_idle_once() == []
    assert calls["loaded_q"] == 1 and calls["unload"] == []


def test_reap_disabled_when_ttl_not_positive():
    r, calls = _reaper(idle=999.0, ttl=0.0)
    assert r.reap_idle_once() == []
    assert calls["loaded_q"] == 0


def test_reap_never_raises_on_a_broken_ollama():
    def boom():
        raise OSError("ollama down")

    r = llm_idle_unload.ExternalModelIdleReaper(
        active_fn=lambda: 0, idle_seconds_fn=lambda: 999.0,
        loaded_models_fn=boom, unload_fn=lambda m: None, ttl_s=300.0, label="x",
    )
    assert r.reap_idle_once() == []          # swallowed, no crash


def test_reap_never_raises_when_unload_fails():
    def boom(models):
        raise OSError("unload failed")

    r = llm_idle_unload.ExternalModelIdleReaper(
        active_fn=lambda: 0, idle_seconds_fn=lambda: 999.0,
        loaded_models_fn=lambda: ["llama3"], unload_fn=boom, ttl_s=300.0, label="x",
    )
    assert r.reap_idle_once() == []


# ── ollama probe helpers (llm_probe) ─────────────────────────────────────────


def test_loaded_models_reads_api_ps():
    ps = {"models": [{"name": "llama3:latest"}, {"name": "qwen:7b"}]}
    got = llm_probe.loaded_models("http://box:11434", get_json_fn=lambda u, t: ps)
    assert got == ["llama3:latest", "qwen:7b"]


def test_loaded_models_empty_on_nothing_or_unreachable():
    assert llm_probe.loaded_models("http://box:11434", get_json_fn=lambda u, t: {"models": []}) == []
    def boom(u, t):
        raise OSError("down")
    assert llm_probe.loaded_models("http://box:11434", get_json_fn=boom) == []
    assert llm_probe.loaded_models("", get_json_fn=lambda u, t: {"models": [{"name": "x"}]}) == []


def test_unload_posts_keep_alive_zero_per_model():
    posts = []

    def post(url, payload, timeout_s):
        posts.append((url, payload))
        return 200

    done = llm_probe.unload_ollama_models("http://box:11434", ["llama3", "qwen"], post_fn=post)
    assert done == ["llama3", "qwen"]
    assert posts[0] == ("http://box:11434/api/generate", {"model": "llama3", "keep_alive": 0})
    assert posts[1][1]["model"] == "qwen" and posts[1][1]["keep_alive"] == 0


def test_unload_is_best_effort_per_model():
    def post(url, payload, timeout_s):
        if payload["model"] == "bad":
            raise OSError("boom")
        return 200

    done = llm_probe.unload_ollama_models("http://box:11434", ["bad", "good"], post_fn=post)
    assert done == ["good"]      # the failure didn't stop the next unload
