"""The context-window domain: what a bigger window costs, and how big a box can go.

Pure functions, no I/O — the worker feeds them observations (ollama /api/ps + /api/show
+ /api/tags, vLLM /v1/models + /metrics, /proc/meminfo) and publishes the answers in its
heartbeat. Every number the defaults rest on was MEASURED on ollama 0.33.2:

  * qwen2.5:0.5b (a standard transformer) on CPU — runner size at three contexts:
        4096 → 484,221,909 B   16384 → 701,811,915 B   32768 → 936,692,939 B
    weights (/api/tags size) 397,821,319 B. Marginal cost 14–18 KB/token against the
    f16 KV formula's 12,288 B — the formula runs 15–30 % LOW (compute buffers grow with
    the context too), hence the formula's safety factor.
  * chat-model-a (hybrid: full attention every 4th layer) — runner 18,714,627,931 B at a
    262,144 context, weights 17,741,872,154 B ⇒ ~3.7 KB/token actually allocated. The
    formula says 64 KiB/token, ~18× too much. So an OBSERVED cost from a loaded runner
    always wins, and the formula is only the fallback for a model nobody has loaded —
    where overestimating means refusing a switch that would have fit, never promising
    one that will not.
  * asking for num_ctx=65536 on a 32768-max model was silently capped at 32768 — so the
    model's own maximum bounds every ceiling.
"""
from __future__ import annotations

import pytest

from queue_workflows import llm_context as lc

MiB = 1024 * 1024
GiB = 1024 * MiB

QWEN25_INFO = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 24,
    "qwen2.attention.head_count": 14,
    "qwen2.attention.head_count_kv": 2,
    "qwen2.attention.key_length": 64,
    "qwen2.attention.value_length": 64,
    "qwen2.embedding_length": 896,
    "qwen2.context_length": 32768,
}
QWEN35_INFO = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 65,
    "qwen35.attention.head_count": 24,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.full_attention_interval": 4,
    "qwen35.context_length": 262144,
}


# ── reading the model ────────────────────────────────────────────────────────


def test_the_model_max_comes_from_model_info():
    assert lc.model_info_max_context(QWEN25_INFO) == 32768
    assert lc.model_info_max_context(QWEN35_INFO) == 262144
    assert lc.model_info_max_context({}) is None
    assert lc.model_info_max_context({"general.architecture": "x"}) is None


def test_the_formula_matches_the_measured_standard_transformer_within_its_safety():
    """24 layers × 2 KV heads × (64+64) × 2 B = 12,288 B/token for ONE slot; the
    measured marginal cost was 14,336–17,707 B. With the 1.25 safety and one slot the
    estimate must cover the measured slope."""
    one_slot = lc.kv_bytes_per_token_formula(QWEN25_INFO, parallel=1)
    assert one_slot == int(12288 * lc.FORMULA_SAFETY)
    assert one_slot >= 14336


def test_the_formula_counts_only_the_full_attention_layers_of_a_hybrid_model():
    per_slot_raw = (65 // 4) * 4 * (256 + 256) * 2
    assert lc.kv_bytes_per_token_formula(QWEN35_INFO, parallel=1) == \
        int(per_slot_raw * lc.FORMULA_SAFETY)


def test_the_formula_assumes_ollamas_parallel_slots_unless_told():
    assert lc.kv_bytes_per_token_formula(QWEN25_INFO) == \
        lc.FORMULA_DEFAULT_PARALLEL * lc.kv_bytes_per_token_formula(QWEN25_INFO, parallel=1)


def test_the_formula_falls_back_to_embedding_over_heads_for_the_head_width():
    info = {k: v for k, v in QWEN25_INFO.items() if "key_length" not in k
            and "value_length" not in k}
    # 896 / 14 = 64 — the same as the explicit fields
    assert lc.kv_bytes_per_token_formula(info, parallel=1) == \
        lc.kv_bytes_per_token_formula(QWEN25_INFO, parallel=1)


@pytest.mark.parametrize("info", [
    {},
    {"general.architecture": "qwen2"},
    {**QWEN25_INFO, "qwen2.attention.head_count_kv": 0},
    {**QWEN25_INFO, "qwen2.block_count": "many"},
])
def test_an_unreadable_model_has_no_formula_cost(info):
    assert lc.kv_bytes_per_token_formula(info) is None


def test_a_per_layer_kv_head_list_is_summed_over_its_attention_layers():
    """Some architectures publish head_count_kv per layer; a 0 marks a layer with no
    KV cache (a recurrent / SSM layer)."""
    info = {**QWEN25_INFO, "qwen2.block_count": 4,
            "qwen2.attention.head_count_kv": [2, 0, 2, 0]}
    assert lc.kv_bytes_per_token_formula(info, parallel=1) == \
        int(2 * 2 * (64 + 64) * 2 * lc.FORMULA_SAFETY)


def test_the_observed_cost_is_the_runner_minus_its_weights_per_token():
    # chat-model-a, measured: (18,714,627,931 − 17,741,872,154) / 262,144
    assert lc.kv_bytes_per_token_observed(
        runner_bytes=18_714_627_931, weights_bytes=17_741_872_154,
        context_length=262_144) == 3711


def test_the_observed_average_bounds_the_measured_marginal_cost_from_above():
    """At 16384 the average (runner − weights) / ctx must be >= the measured marginal
    slope from 16384 to 32768 (14,336 B/token) — extrapolating UP with it is safe."""
    avg = lc.kv_bytes_per_token_observed(
        runner_bytes=701_811_915, weights_bytes=397_821_319, context_length=16384)
    assert avg >= 14336


@pytest.mark.parametrize("kw", [
    dict(runner_bytes=None, weights_bytes=1, context_length=1),
    dict(runner_bytes=1, weights_bytes=None, context_length=1),
    dict(runner_bytes=100, weights_bytes=200, context_length=4096),   # nonsense
    dict(runner_bytes=100, weights_bytes=50, context_length=0),
])
def test_an_observation_that_cannot_be_trusted_is_none(kw):
    assert lc.kv_bytes_per_token_observed(**kw) is None


# ── how big a box could go ───────────────────────────────────────────────────


def _ollama(**kw):
    base = dict(current=32768, model_max=262144, per_token=16 * 1024,
                weights_bytes=17 * GiB, runner_bytes=18 * GiB,
                available_bytes=60 * GiB, reserve_bytes=8 * GiB,
                granule=lc.CONTEXT_GRANULE)
    base.update(kw)
    return lc.ollama_ceiling(**base)


def test_a_loaded_ollama_box_can_grow_into_its_free_memory_plus_the_runner_it_frees():
    # budget = 60 + 18 − 8 = 70 GiB; minus 17 GiB weights = 53 GiB for the cache
    expected = (53 * GiB // (16 * 1024)) // 1024 * 1024
    assert _ollama(model_max=10_000_000) == expected


def test_the_model_maximum_caps_the_ceiling():
    assert _ollama() == 262144


def test_an_unloaded_model_must_fit_its_weights_too():
    ceiling = _ollama(runner_bytes=0, available_bytes=20 * GiB, model_max=10_000_000)
    # budget 20 − 8 = 12 GiB < 17 GiB of weights: it cannot even load — no promise
    # beyond what it serves now
    assert ceiling == 32768


def test_a_ceiling_never_drops_below_the_window_already_served():
    assert _ollama(available_bytes=0, runner_bytes=0) == 32768


def test_an_overestimated_cost_cannot_shrink_what_a_box_demonstrably_serves():
    """The formula fallback on a loaded hybrid model overestimates ~18×: here the
    arithmetic says 131072 tokens fit, but the box is SERVING 262144 right now."""
    assert _ollama(current=262144, per_token=16 * 1024, weights_bytes=17 * GiB,
                   runner_bytes=18 * GiB, available_bytes=1 * GiB,
                   reserve_bytes=0) == 262144


@pytest.mark.parametrize("missing", ["per_token", "weights_bytes", "available_bytes"])
def test_without_the_numbers_to_prove_a_bigger_window_the_ceiling_is_the_current_one(missing):
    assert _ollama(**{missing: None}) == 32768


def test_a_box_that_knows_nothing_advertises_nothing():
    assert lc.ollama_ceiling(current=None, model_max=None, per_token=None,
                             weights_bytes=None, runner_bytes=0, available_bytes=None,
                             reserve_bytes=0, granule=1024) is None


def test_the_ceiling_is_a_whole_granule():
    ceiling = _ollama(model_max=10_000_000, available_bytes=61 * GiB + 12345)
    assert ceiling % lc.CONTEXT_GRANULE == 0


def test_a_vllm_ceiling_is_its_model_max_bounded_by_the_kv_pool_it_already_owns():
    """vLLM's KV pool is sized by gpu_memory_utilization, not by max_model_len, so a
    bigger window costs no new memory — it only has to fit the pool (/metrics)."""
    assert lc.vllm_ceiling(current=32768, model_max=262144,
                           kv_capacity_tokens=100_000, granule=1024) == 99328
    assert lc.vllm_ceiling(current=32768, model_max=65536,
                           kv_capacity_tokens=100_000, granule=1024) == 65536


@pytest.mark.parametrize("kw", [
    dict(model_max=None, kv_capacity_tokens=100_000),
    dict(model_max=262144, kv_capacity_tokens=None),
])
def test_a_vllm_box_that_cannot_prove_more_stays_where_it_is(kw):
    assert lc.vllm_ceiling(current=32768, granule=1024, **kw) == 32768


# ── what to switch to ────────────────────────────────────────────────────────


def test_the_target_is_the_highest_queued_need_this_box_can_reach():
    assert lc.switch_target([40000, 70000, 300000], current=32768, ceiling=131072) == 70656


def test_no_switch_when_every_queued_need_already_fits():
    assert lc.switch_target([8192, 32768], current=32768, ceiling=131072) is None


def test_no_switch_when_no_queued_job_defines_a_context():
    """'If none defines a context, use the current default' — nothing to do."""
    assert lc.switch_target([], current=32768, ceiling=131072) is None
    assert lc.switch_target([None, None], current=32768, ceiling=131072) is None


def test_no_switch_for_needs_beyond_this_box_they_belong_to_another_box_or_the_tag():
    assert lc.switch_target([300000], current=32768, ceiling=131072) is None


def test_a_box_with_an_unknown_window_switches_up_to_the_need_it_can_reach():
    assert lc.switch_target([50000], current=None, ceiling=131072) == 50176


def test_a_box_without_a_ceiling_never_switches():
    assert lc.switch_target([50000], current=32768, ceiling=None) is None


def test_the_target_is_rounded_up_to_a_granule_but_never_past_the_ceiling():
    assert lc.switch_target([130000], current=32768, ceiling=131072) == 130048
    # 130100 rounds up to 131072, which is past a 130500 ceiling: capped there
    assert lc.switch_target([130100], current=32768, ceiling=130500) == 130500


# ── the host's free memory ───────────────────────────────────────────────────


def test_mem_available_is_read_from_meminfo(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       127940000 kB\nMemFree:         1000 kB\n"
                       "MemAvailable:   84000000 kB\n")
    assert lc.mem_available_bytes(str(meminfo)) == 84_000_000 * 1024


def test_an_unreadable_meminfo_is_unknown(tmp_path):
    assert lc.mem_available_bytes(str(tmp_path / "absent")) is None
    bad = tmp_path / "bad"
    bad.write_text("nothing useful here\n")
    assert lc.mem_available_bytes(str(bad)) is None


def test_the_reserve_is_an_operator_knob(monkeypatch):
    assert lc.reserve_bytes() == lc.DEFAULT_RESERVE_MB * MiB
    monkeypatch.setenv("QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB", "2048")
    assert lc.reserve_bytes() == 2048 * MiB
    monkeypatch.setenv("QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB", "lots")
    assert lc.reserve_bytes() == lc.DEFAULT_RESERVE_MB * MiB


def test_the_box_total_is_read_from_meminfo(tmp_path):
    """On a unified-memory box MemTotal is also the "GPU memory" vLLM takes its pool
    fraction of — the denominator of vllm_utilization."""
    from queue_workflows import llm_context as _ctx

    p = tmp_path / "meminfo"
    p.write_text("MemTotal:       131072000 kB\nMemFree:        1234 kB\nMemAvailable:   5678 kB\n")
    assert _ctx.mem_total_bytes(str(p)) == 131072000 * 1024
    assert _ctx.mem_total_bytes(str(tmp_path / "missing")) is None
