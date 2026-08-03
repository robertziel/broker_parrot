"""Unload the EXTERNAL LLM server's model after 5 minutes WITHOUT A USE.

The engine's ``ModelCache`` reaper (model_cache.py) unloads a warm IN-PROCESS model
on idle, and the vLLM ``LLMSupervisor`` stops the vLLM sidecar on idle — but an
ollama box's model is left to ollama's own ``OLLAMA_KEEP_ALIVE`` timer, engine-
uncontrolled, so it can sit in VRAM/RAM long after its last use. This reaper closes
that: once no client has used the model for ``ttl_s`` (default 5 min), it unloads
it, giving the RAM back.

THE IDLE SIGNAL (redesign). "Used" is observed at the SERVER: ollama
bumps each loaded model's ``/api/ps`` ``expires_at`` on every request, so an
ADVANCE between reaper ticks = a use (by ANY client — a queue node, a host's own
HTTP call, an app captioner), and a frozen value ages toward the TTL. The old
claim-based clock was wrong both ways on a live box: memoized queue claims reset
it without touching the model (never unloaded), while host-internal LLM calls
were invisible to it (could unload a model mid-use-pattern).

Shape mirrors ``model_cache.gpu_should_unload`` / ``reap_idle_once`` on purpose:
a pure decision + a one-tick reaper + a clamped-cadence daemon, all I/O injected so
the whole thing runs on a virtual clock with a fake ollama.
"""

from __future__ import annotations

import pytest

import queue_workflows
from queue_workflows import llm_idle_unload, llm_probe


# ── the pure decision ─────────────────────────────────────────────────────────


def test_should_unload_only_when_loaded_quiet_and_idle_past_ttl():
    assert llm_idle_unload.should_unload_external_model(True, 0, 300.0, 300.0) is True
    assert llm_idle_unload.should_unload_external_model(True, 0, 999.0, 300.0) is True


@pytest.mark.parametrize("loaded,active,idle,ttl", [
    (False, 0, 999.0, 300.0),   # nothing loaded → nothing to free
    (True, 1, 999.0, 300.0),    # a job is running → the GPU IS in use
    (True, 0, 120.0, 300.0),    # unused, but not long enough yet
    (True, 0, 999.0, 0.0),      # ttl<=0 disables the reaper entirely
    (True, 0, 999.0, -1.0),
])
def test_should_not_unload(loaded, active, idle, ttl):
    assert llm_idle_unload.should_unload_external_model(loaded, active, idle, ttl) is False


# ── THE config constant + its resolution order ───────────────────────────────


def test_default_ttl_is_five_minutes():
    assert llm_idle_unload.DEFAULT_LLM_IDLE_TTL_S == 300.0


def test_ttl_resolves_from_configure(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S", raising=False)
    monkeypatch.delenv("AI_LEADS_LLM_SERVER_IDLE_TTL_S", raising=False)
    queue_workflows.configure(llm_server_idle_ttl_s=120)
    assert llm_idle_unload.resolve_llm_idle_ttl_s() == 120.0
    # …and a reaper armed AFTER configure() picks it up (call-time, not import-time)
    r = llm_idle_unload.ExternalModelIdleReaper(
        active_fn=lambda: 0, loaded_models_info_fn=lambda: [],
        unload_fn=lambda m: None, label="x",
    )
    assert r.ttl_s == 120.0


def test_ttl_env_knob_overrides_configure(monkeypatch):
    queue_workflows.configure(llm_server_idle_ttl_s=120)
    monkeypatch.setenv("QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S", "45")
    assert llm_idle_unload.resolve_llm_idle_ttl_s() == 45.0


def test_ttl_defaults_when_nothing_set(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S", raising=False)
    monkeypatch.delenv("AI_LEADS_LLM_SERVER_IDLE_TTL_S", raising=False)
    assert llm_idle_unload.resolve_llm_idle_ttl_s() == 300.0


def test_ttl_malformed_env_falls_through(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S", "soon")
    queue_workflows.configure(llm_server_idle_ttl_s=90)
    assert llm_idle_unload.resolve_llm_idle_ttl_s() == 90.0


# ── the reaper tick (virtual clock, fake ollama) ─────────────────────────────


class _Box:
    """A fake ollama + virtual clock: ``use(model)`` bumps that model's
    ``expires_at`` the way any real request does."""

    def __init__(self, *models: str, active: int = 0, ttl: float = 300.0):
        self.now = 1000.0
        self.active = active
        self.models = {m: 5000.0 for m in models}   # name -> expires_at ("mru")
        self.unloaded: list[list[str]] = []
        self.ps_calls = 0
        self.reaper = llm_idle_unload.ExternalModelIdleReaper(
            active_fn=lambda: self.active,
            loaded_models_info_fn=self._info,
            unload_fn=self._unload,
            ttl_s=ttl,
            now_fn=lambda: self.now,
            label="box-a-gpu",
        )

    def _info(self):
        self.ps_calls += 1
        return [{"name": n, "mru": e} for n, e in self.models.items()]

    def _unload(self, ms):
        self.unloaded.append(list(ms))
        for m in ms:
            self.models.pop(m, None)

    def use(self, model: str):
        self.models[model] = self.models[model] + 100.0   # server bumps expires_at

    def tick(self, advance_s: float = 0.0):
        self.now += advance_s
        return self.reaper.reap_idle_once()


def test_first_sighting_starts_the_clock_and_unloads_after_ttl():
    box = _Box("an LLM")
    assert box.tick() == []                 # sighting stamps "used now"
    assert box.tick(advance_s=299.0) == []  # not unused long enough yet
    assert box.tick(advance_s=1.0) == ["an LLM"]
    assert box.unloaded == [["an LLM"]]


def test_any_clients_use_resets_the_clock():
    """An expires_at ADVANCE between ticks = some client used the model — the
    unload must move out by a fresh TTL. This is the one signal every client
    shares (queue nodes, host-internal HTTP, app captioners)."""
    box = _Box("an LLM")
    box.tick()
    box.tick(advance_s=250.0)
    box.use("an LLM")                          # a use at t+250…
    assert box.tick(advance_s=50.0) == []    # …observed at t+300 → clock restamped
    assert box.tick(advance_s=299.0) == []   # still inside the fresh TTL
    assert box.tick(advance_s=1.0) == ["an LLM"]   # TTL after the observed use


def test_a_reload_after_unload_restarts_the_clock():
    box = _Box("an LLM")
    box.tick()
    assert box.tick(advance_s=300.0) == ["an LLM"]
    box.models["an LLM"] = 9000.0              # someone loads it again
    assert box.tick(advance_s=10.0) == []    # fresh sighting → fresh clock
    assert box.tick(advance_s=300.0) == ["an LLM"]


def test_active_job_short_circuits_before_any_http():
    box = _Box("an LLM", active=2)
    assert box.tick(advance_s=999.0) == []
    assert box.ps_calls == 0


def test_no_model_loaded_is_a_noop_and_clears_state():
    box = _Box()
    assert box.tick(advance_s=999.0) == []
    assert box.unloaded == []


def test_disabled_when_ttl_not_positive():
    box = _Box("an LLM", ttl=0.0)
    assert box.tick(advance_s=999.0) == []
    assert box.ps_calls == 0


def test_unparseable_expiry_degrades_to_the_sighting_clock():
    """mru 0.0 (unparseable expires_at) can never advance — the reaper still
    unloads TTL after the first sighting rather than never."""
    box = _Box("an LLM")
    box.models["an LLM"] = 0.0
    box.tick()
    assert box.tick(advance_s=300.0) == ["an LLM"]


def test_reap_never_raises_on_a_broken_ollama():
    def boom():
        raise OSError("ollama down")

    r = llm_idle_unload.ExternalModelIdleReaper(
        active_fn=lambda: 0, loaded_models_info_fn=boom,
        unload_fn=lambda m: None, ttl_s=300.0, label="x",
    )
    assert r.reap_idle_once() == []          # swallowed, no crash


def test_reap_never_raises_when_unload_fails():
    def boom(models):
        raise OSError("unload failed")

    now = [1000.0]
    r = llm_idle_unload.ExternalModelIdleReaper(
        active_fn=lambda: 0,
        loaded_models_info_fn=lambda: [{"name": "llama3", "mru": 5.0}],
        unload_fn=boom, ttl_s=300.0, now_fn=lambda: now[0], label="x",
    )
    r.reap_idle_once()
    now[0] += 300.0
    assert r.reap_idle_once() == []


# ── ollama probe helpers (llm_probe) ─────────────────────────────────────────


def test_loaded_models_reads_api_ps():
    ps = {"models": [{"name": "llama3:latest"}, {"name": "chat-model-c"}]}
    got = llm_probe.loaded_models("http://box:11434", get_json_fn=lambda u, t: ps)
    assert got == ["llama3:latest", "chat-model-c"]


def test_loaded_models_empty_on_nothing_or_unreachable():
    assert llm_probe.loaded_models("http://box:11434", get_json_fn=lambda u, t: {"models": []}) == []
    def boom(u, t):
        raise OSError("down")
    assert llm_probe.loaded_models("http://box:11434", get_json_fn=boom) == []
    assert llm_probe.loaded_models("", get_json_fn=lambda u, t: {"models": [{"name": "x"}]}) == []


def test_loaded_models_info_carries_the_expiry_as_mru():
    ps = {"models": [{"name": "chat-model-c", "expires_at": "2026-07-19T21:00:00.123456789Z"}]}
    got = llm_probe.loaded_models_info("http://box:11434", get_json_fn=lambda u, t: ps)
    assert got[0]["name"] == "chat-model-c" and got[0]["mru"] > 0


def test_unload_posts_keep_alive_zero_per_model():
    posts = []

    def post(url, payload, timeout_s):
        posts.append((url, payload))
        return 200

    done = llm_probe.unload_ollama_models("http://box:11434", ["llama3", "an LLM"], post_fn=post)
    assert done == ["llama3", "an LLM"]
    assert posts[0] == ("http://box:11434/api/generate", {"model": "llama3", "keep_alive": 0})
    assert posts[1][1]["model"] == "an LLM" and posts[1][1]["keep_alive"] == 0


def test_unload_is_best_effort_per_model():
    def post(url, payload, timeout_s):
        if payload["model"] == "bad":
            raise OSError("boom")
        return 200

    done = llm_probe.unload_ollama_models("http://box:11434", ["bad", "good"], post_fn=post)
    assert done == ["good"]      # the failure didn't stop the next unload
