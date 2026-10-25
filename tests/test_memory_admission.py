"""No model is loaded into a box that cannot hold it — decided BEFORE the load.

Two loads exist on a GPU box, and both used to happen blind:

  * ollama loads a model on the FIRST REQUEST for it. The engine's only hand on that is
    the claim: a job whose model is not resident here must not be claimed here unless the
    weights plus the cache fit in the memory the box has free, after a reserve for
    everything else. Example: a vLLM pool held most of a box, ~17 GB was free, and a
    claim for an ollama job pulled 17.4 GB of weights into the rest. Refused now.
  * vLLM takes ``--gpu-memory-utilization`` of "GPU memory" at start — and on a
    unified-memory box that is the whole machine, so 0.75 is 91 GB of 121 before the OS,
    the claim lanes and every other container get a byte. The pool is now sized to what
    is free (never above the configured value), and a box that cannot fit even the
    minimum pool does not start vLLM at all.

The arithmetic follows the 0015 convention on unknowns: a size or a free-memory reading
the box cannot produce lets the job through (a box that cannot describe itself must not
wedge the queue); a KNOWN misfit is refused.
"""
from __future__ import annotations

import pytest

from queue_workflows import claim_worker, llm_context, llm_context_lever, llm_probe
from queue_workflows import server_switch
from queue_workflows.llm_backends import factory as _factory
from queue_workflows.llm_backends import server_context as sc

GiB = 2**30


# ── the arithmetic (pure) ────────────────────────────────────────────────────


def test_a_model_fits_when_weights_and_cache_clear_the_reserve():
    assert llm_context.load_fits(weights_bytes=17 * GiB, kv_bytes=1 * GiB,
                                 available_bytes=30 * GiB, reserve_bytes=8 * GiB) is True


def test_a_model_does_not_fit_when_the_reserve_would_be_eaten():
    """Example numbers: 17.4 GB of weights into ~17 GB free beside a 91 GB vLLM pool."""
    assert llm_context.load_fits(weights_bytes=int(17.4 * GiB), kv_bytes=1 * GiB,
                                 available_bytes=17 * GiB, reserve_bytes=8 * GiB) is False


def test_an_unknown_size_or_reading_cannot_refuse():
    assert llm_context.load_fits(weights_bytes=None, kv_bytes=0,
                                 available_bytes=1 * GiB, reserve_bytes=0) is None
    assert llm_context.load_fits(weights_bytes=17 * GiB, kv_bytes=0,
                                 available_bytes=None, reserve_bytes=0) is None


def test_vllm_pool_is_lowered_to_what_the_box_has_free():
    """121 GB box, 60 GB free after other tenants, 8 GB reserve: 0.75 would be 91 GB of
    memory the box does not have. (60-8)/121 = 0.42."""
    util = llm_context.vllm_utilization(configured=0.75, total_bytes=121 * GiB,
                                        available_bytes=60 * GiB, reserve_bytes=8 * GiB)
    assert util == pytest.approx(0.42, abs=0.005)


def test_vllm_pool_is_never_raised_above_the_operators_value():
    util = llm_context.vllm_utilization(configured=0.75, total_bytes=121 * GiB,
                                        available_bytes=115 * GiB, reserve_bytes=8 * GiB)
    assert util == pytest.approx(0.75)


def test_vllm_pool_without_a_flag_starts_from_vllms_own_default():
    util = llm_context.vllm_utilization(configured=None, total_bytes=100 * GiB,
                                        available_bytes=100 * GiB, reserve_bytes=0)
    assert util == pytest.approx(llm_context.DEFAULT_VLLM_UTILIZATION)


def test_vllm_pool_is_unknown_without_the_readings():
    assert llm_context.vllm_utilization(configured=0.75, total_bytes=None,
                                        available_bytes=10 * GiB, reserve_bytes=0) is None


# ── the vLLM start argv ──────────────────────────────────────────────────────


def test_the_utilization_flag_is_read_and_rewritten_in_place():
    argv = ["--model", "m", "--gpu-memory-utilization", "0.75", "--max-num-seqs", "8"]
    assert sc.argv_utilization(argv) == pytest.approx(0.75)
    assert sc.argv_with_utilization(argv, "0.42") == [
        "--model", "m", "--gpu-memory-utilization", "0.42", "--max-num-seqs", "8"]
    assert sc.argv_utilization(["--model", "m"]) is None
    assert sc.argv_with_utilization(["--model", "m"], "0.42") == [
        "--model", "m", "--gpu-memory-utilization", "0.42"]
    assert sc.argv_with_utilization(["--gpu-memory-utilization=0.9"], "0.42") == [
        "--gpu-memory-utilization=0.42"]


VLLM = {
    "Id": "abc", "Config": {"Image": "vllm/vllm-openai:latest",
                            "Cmd": ["--model", "m", "--gpu-memory-utilization", "0.75",
                                    "--max-model-len", "262144"],
                            "Env": [], "Labels": {}},
    "HostConfig": {"NetworkMode": "host"}, "NetworkSettings": {"Networks": {}},
}


def test_a_memory_only_change_recreates_and_remembers_the_operators_value():
    plan = sc.plan_context(VLLM, "vllm", None, name="vllm", utilization=0.42)
    assert plan is not None
    assert sc.argv_utilization(plan.payload["Cmd"]) == pytest.approx(0.42)
    assert plan.payload["Labels"][sc.UTILIZATION_LABEL] == "0.75"
    assert sc.DEFAULT_LABEL not in plan.payload["Labels"], "the window was not touched"


def test_the_remembered_value_is_the_one_future_starts_size_from():
    """Else a pool lowered once could only ever go down: the next start would read the
    lowered flag as 'the operator's value'."""
    lowered = {**VLLM, "Config": {**VLLM["Config"],
                                  "Cmd": ["--model", "m", "--gpu-memory-utilization", "0.42"],
                                  "Labels": {sc.UTILIZATION_LABEL: "0.75"}}}
    assert sc.operator_utilization(lowered) == pytest.approx(0.75)
    assert sc.operator_utilization(VLLM) == pytest.approx(0.75)
    plan = sc.plan_context(lowered, "vllm", None, name="vllm", utilization=0.75)
    assert sc.argv_utilization(plan.payload["Cmd"]) == pytest.approx(0.75)
    assert plan.payload["Labels"][sc.UTILIZATION_LABEL] == "0.75", "still remembered"


def test_the_same_utilization_is_not_a_recreate():
    assert sc.plan_context(VLLM, "vllm", None, name="vllm", utilization=0.75) is None
    assert sc.plan_context(VLLM, "vllm", None, name="vllm", utilization=0.752) is None


def test_window_and_pool_can_change_in_one_recreate():
    plan = sc.plan_context(VLLM, "vllm", 131072, name="vllm", utilization=0.42)
    assert sc.argv_context(plan.payload["Cmd"]) == 131072
    assert sc.argv_utilization(plan.payload["Cmd"]) == pytest.approx(0.42)


def test_ollama_ignores_a_utilization_it_has_no_flag_for():
    ollama = {"Id": "x", "Config": {"Image": "ollama/ollama", "Cmd": ["serve"], "Env": [],
                                    "Labels": {}}, "HostConfig": {"NetworkMode": "bridge"},
              "NetworkSettings": {"Networks": {}}}
    assert sc.plan_context(ollama, "ollama", None, name="broker-ollama", utilization=0.4) is None


# ── the worker: claim-time admission ─────────────────────────────────────────

PS_LOADED = [{"name": "chat-model-a:latest", "size": 18_714_627_931,
              "size_vram": 18_714_627_931, "context_length": 32768}]
SIZES = {"chat-model-a:latest": 17_741_872_154, "chat-expert-a:latest": 16_800_000_000,
         "tiny:latest": 400_000_000}
INFO = {"general.architecture": "qwen35", "qwen35.context_length": 262144,
        "qwen35.block_count": 36, "qwen35.attention.head_count_kv": 8,
        "qwen35.attention.head_count": 32, "qwen35.embedding_length": 4096,
        "qwen35.full_attention_interval": 4}


def _worker(served=("ollama",)):
    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-c-gpu"
            self.queue = "gpu"
            self._box_served = list(served)
            self._switch = server_switch.ServerSwitch(label="box-c-gpu")
    w = _W()
    w._physical_box = lambda: "box-c"
    return w


@pytest.fixture(autouse=True)
def _box(monkeypatch):
    llm_context_lever.reset_cache_for_tests()
    monkeypatch.setattr(_factory, "resolve_base_url", lambda kind="ollama": f"http://{kind}:1")
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: list(PS_LOADED))
    monkeypatch.setattr(llm_probe, "ollama_model_sizes", lambda url, **k: dict(SIZES))
    monkeypatch.setattr(llm_probe, "ollama_model_info", lambda url, m, **k: dict(INFO))
    monkeypatch.setattr(llm_probe, "vllm_max_model_len", lambda url, **k: 262144)
    monkeypatch.setattr(llm_probe, "vllm_served_models", lambda url, **k: ["chat-model-a:latest"])
    monkeypatch.setattr(llm_probe, "vllm_kv_capacity_tokens", lambda url, **k: None)
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k:
                        llm_context_lever.ServerContainer(kind=kind, name=kind, exists=False))
    monkeypatch.setattr(llm_context_lever, "available", lambda kind, **k: False)
    monkeypatch.setattr(llm_context, "mem_total_bytes", lambda *a, **k: 121 * GiB)
    yield
    llm_context_lever.reset_cache_for_tests()


def test_a_resident_model_always_passes_admission(monkeypatch):
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 2 * GiB)
    w = _worker()
    w._observe_llm_context()
    assert w._known_models(["chat-model-a:latest", "chat-expert-a:latest"]) == ["chat-model-a:latest"]


def test_a_model_that_would_not_fit_is_not_offered_to_the_claim(monkeypatch):
    """The storm-judge case: not resident here, 16.8 GB of weights, 17 GB free after a
    91 GB vLLM pool. It waits for a box that can hold it."""
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 17 * GiB)
    w = _worker()
    w._observe_llm_context()
    assert "chat-expert-a:latest" not in w._known_models(["chat-model-a:latest", "chat-expert-a:latest"])


def test_a_model_that_fits_is_offered(monkeypatch):
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 60 * GiB)
    w = _worker()
    w._observe_llm_context()
    assert w._known_models(["chat-model-a:latest", "chat-expert-a:latest", "tiny:latest"]) == [
        "chat-model-a:latest", "chat-expert-a:latest", "tiny:latest"]


def test_admission_follows_free_memory_from_tick_to_tick(monkeypatch):
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 17 * GiB)
    w = _worker()
    w._observe_llm_context()
    assert "tiny:latest" in w._known_models(["tiny:latest"])
    assert "chat-expert-a:latest" not in w._known_models(["chat-expert-a:latest"])
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 60 * GiB)
    w._observe_llm_context()
    assert "chat-expert-a:latest" in w._known_models(["chat-expert-a:latest"])


def test_admission_does_not_use_the_ceilings_pessimistic_cache_estimate(monkeypatch):
    """One observed refusal: 'NOT offering to load … chat-model-a:latest (16.5 GB) — 75.6 GB available'.
    No runner was loaded, so the cache was estimated by the formula the CEILING uses —
    x4 slots x1.25 safety, ~18x the measured cost of this hybrid model — and 16.5 GB of
    weights were refused into 67 GB of room. That direction is right for "how big could
    the window grow" and wrong here: admission needs ONE slot to fit, and the reserve
    covers the rest."""
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [])
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 60 * GiB)
    w = _worker()
    w._ctx_last_model = "chat-model-a:latest"          # the box's usual model, unloaded now
    w._ctx_last_window["ollama"] = 262144
    w._observe_llm_context()
    assert "chat-model-a:latest" in w._known_models(["chat-model-a:latest"])


def test_the_cache_cost_seen_on_a_runner_is_remembered_after_it_unloads(monkeypatch):
    """The measured cost of THIS model on THIS daemon is the best estimate there is —
    the formula, even at one slot, is still ~4x over for a hybrid model."""
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 30 * GiB)
    w = _worker()
    w._observe_llm_context()                       # runner loaded: cost observed
    monkeypatch.setattr(llm_probe, "ollama_runners", lambda url, **k: [])
    w._observe_llm_context()                       # unloaded: remembered, not re-guessed
    kv = w.__dict__["_ctx_admission"]["kv"]
    assert 0 < kv < int(1.1 * GiB), f"observed ~0.97 GB at 32768, formula would say more: {kv/GiB:.2f}"


def test_a_size_the_box_does_not_know_lets_the_job_through(monkeypatch):
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 1 * GiB)
    w = _worker()
    w._observe_llm_context()
    assert "mystery:latest" in w._known_models(["mystery:latest"])


def test_before_the_first_observation_nothing_is_narrowed():
    w = _worker()
    assert w._known_models(["chat-expert-a:latest"]) == ["chat-expert-a:latest"]


def test_a_vllm_box_offers_only_the_model_its_server_pins(monkeypatch):
    """Anything else would have to load beside the pool — refused by size when the size
    is known, and by exclusivity of kind either way."""
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 17 * GiB)
    w = _worker(served=("vllm",))
    w._observe_llm_context()
    assert w._known_models(["chat-model-a:latest", "chat-expert-a:latest"]) == ["chat-model-a:latest"]


# ── the worker: the vLLM start ───────────────────────────────────────────────


def _vllm_worker(monkeypatch, *, available_gib, configured=0.75):
    monkeypatch.setattr(llm_context_lever, "available", lambda kind, **k: True)
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k:
                        llm_context_lever.ServerContainer(
                            kind=kind, name="vllm", exists=True, running=False,
                            configured=262144, utilization=configured))
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: available_gib * GiB)
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: [])
    monkeypatch.setattr(llm_probe, "loaded_models", lambda url, **k: [])
    monkeypatch.setattr(llm_probe, "vllm_served_models", lambda url, **k: [])
    w = _worker()
    w._get_box_lease = lambda: None
    w._gpu_active_count = lambda: 0
    return w


def test_vllm_starts_with_a_pool_sized_to_what_is_free(monkeypatch):
    calls: list = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: calls.append((kind, n, k.get("utilization"))) or
                        {"status": "started"})
    w = _vllm_worker(monkeypatch, available_gib=60)
    w._switch.request("vllm")
    w._drive_server_switch()
    w._drive_server_switch()
    assert calls == [("vllm", None, pytest.approx(0.42, abs=0.005))]


def test_a_pool_that_already_fits_starts_the_operators_way(monkeypatch):
    from queue_workflows import config as _config

    started: list = []
    cfg = _config.get_config()
    cfg.vllm_start_fn = lambda model: started.append(model) or None
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda *a, **k: pytest.fail("no recreate needed"))
    try:
        w = _vllm_worker(monkeypatch, available_gib=115)
        w._switch.request("vllm")
        w._drive_server_switch()
        w._drive_server_switch()
        assert started == [None]
    finally:
        cfg.vllm_start_fn = None


def test_the_pool_is_decided_once_per_switch_not_on_every_retry(monkeypatch):
    """One start sequence: the first start fitted (114 GB free, plain docker start) and
    vLLM began loading; the switch re-issues START every 30 s while warming, and the retry
    re-read MemAvailable — now lower BECAUSE of that load — decided 0.68, and RECREATED the
    container, killing the load. Then 0.63. Then 0.58. A retry is the same start; its
    memory decision was made when the switch entered WARMING and must not be remade against
    the memory the start itself is consuming."""
    from queue_workflows import config as _config

    started: list = []
    recreated: list = []
    cfg = _config.get_config()
    cfg.vllm_start_fn = lambda model: started.append(model) or None
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: recreated.append(k.get("utilization")) or
                        {"status": "started"})
    try:
        w = _vllm_worker(monkeypatch, available_gib=115)
        w._switch.request("vllm")
        w._drive_server_switch()
        w._drive_server_switch()                      # first START: plain, pool fits
        assert started == [None] and recreated == []
        # the load eats memory; the switch retries the start 30 s later
        monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 60 * GiB)
        w._switch._last_try -= 999                    # make the retry due
        w._drive_server_switch()
        assert recreated == [], "a retry must not recreate the container it is warming"
        assert started == [None, None], "the retry is the same plain start"
    finally:
        cfg.vllm_start_fn = None


def test_a_new_switch_decides_afresh(monkeypatch):
    calls: list = []
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda kind, n, **k: calls.append(k.get("utilization")) or
                        {"status": "started"})
    w = _vllm_worker(monkeypatch, available_gib=60)
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert calls == [pytest.approx(0.42, abs=0.005)]
    w._switch.abandon("test")
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 90 * GiB)
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert calls[-1] == pytest.approx(0.67, abs=0.005), "the next switch sees today's memory"


def test_a_refused_pool_stops_a_vllm_that_is_already_loading(monkeypatch):
    """One earlier restart: the previous engine had docker-started vLLM before the roll; the new
    one then refused the pool ("37.2 GB available") and aborted — leaving that container
    loading toward the 91 GB it had just refused, with ollama holding 17 GB beside it. A
    refusal is a verdict about the box's memory; a vLLM still loading into it is the very
    thing the verdict says cannot fit. (A vLLM already SERVING never reaches this branch —
    the switch completes on it instead.)"""
    from queue_workflows import config as _config

    stopped: list = []
    cfg = _config.get_config()
    cfg.vllm_stop_fn = lambda: stopped.append(True) or True
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda *a, **k: pytest.fail("must not start"))
    try:
        w = _vllm_worker(monkeypatch, available_gib=20)
        monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k:
                            llm_context_lever.ServerContainer(
                                kind=kind, name="vllm", exists=True, running=True,
                                configured=262144, utilization=0.75))
        w._switch.request("vllm")
        w._drive_server_switch()
        w._drive_server_switch()
        assert stopped == [True], "the loading vLLM must be stopped, not left to finish"
        assert w._switch.quarantined("vllm") is True
    finally:
        cfg.vllm_stop_fn = None


def test_a_refusal_with_nothing_running_stops_nothing(monkeypatch):
    from queue_workflows import config as _config

    cfg = _config.get_config()
    cfg.vllm_stop_fn = lambda: pytest.fail("nothing to stop")
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda *a, **k: pytest.fail("must not start"))
    try:
        w = _vllm_worker(monkeypatch, available_gib=20)      # describe(): running=False
        w._switch.request("vllm")
        w._drive_server_switch()
        w._drive_server_switch()
        assert w._switch.quarantined("vllm") is True
    finally:
        cfg.vllm_stop_fn = None


def test_a_box_that_cannot_fit_even_the_minimum_pool_does_not_start_vllm(monkeypatch):
    """The gate: no start into memory the box does not have. The
    switch is told the start CANNOT succeed, so it aborts now and quarantines vLLM here
    instead of warming for 30 minutes."""
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda *a, **k: pytest.fail("must not start"))
    w = _vllm_worker(monkeypatch, available_gib=20)
    w._switch.request("vllm")
    w._drive_server_switch()
    w._drive_server_switch()
    assert w._switch.state == server_switch.IDLE
    assert w._switch.quarantined("vllm") is True
    assert "GB" in (w._switch.last_start or {}).get("reason", "")
