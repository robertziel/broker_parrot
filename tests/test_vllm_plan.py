"""vLLM sized to what it NEEDS, and concurrency counted from what FITS.

THE FACT this is built on. At startup vLLM states, in its own log, exactly what the
broker needs to know about THIS model on THIS box:

    Model loading took 18.74 GiB memory and 315.5 seconds
    Available KV cache memory: 42.23 GiB
    GPU KV cache size: 684,056 tokens
    Maximum concurrency for 262,144 tokens per request: 2.61x

(One start at ``--gpu-memory-utilization 0.42``, ``--max-num-seqs 8``.) Read back:
the weights cost 18.7 GB, the cache costs 66 KB per token, and at that pool the box
holds 2.6 full-window requests — while the broker fed it 8 slots and the pool was sized
to "whatever is free", 91 GB when free. Two errors in opposite directions, from the same
missing number.

The plan: with a profile of this model, the pool is what N requests of the window need
(weights + the start's fixed overhead + N × window × per-token) and nothing more; when
the box cannot afford N, N is what the box CAN afford, and vLLM is started with that
``--max-num-seqs`` so it never admits more than it can hold. Without a profile — the
first start of a model on a box — the memory gate's default rule applies, and the
start's own log becomes the profile for the next one. Either way the box advertises the
concurrency its cache really holds, not the operator's ceiling.
"""
from __future__ import annotations

import pytest

from queue_workflows import llm_context
from queue_workflows.llm_backends import server_context as sc

GiB = 2**30

VLLM_LOG = """\
(APIServer pid=1) INFO 01-01 00:21:21 [api_utils.py:273] non-default args: {'model_tag': '/models/base'}
(EngineCore pid=170) INFO 01-01 00:23:48 [default_loader.py:397] Loading weights took 125.57 seconds
(EngineCore pid=170) INFO 01-01 00:23:49 [gpu_model_runner.py:5187] Model loading took 18.74 GiB memory and 126.992779 seconds
(EngineCore pid=170) INFO 01-01 00:24:50 [gpu_worker.py:480] Available KV cache memory: 42.23 GiB
(EngineCore pid=170) INFO 01-01 00:24:50 [gpu_worker.py:495] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.4200 is equivalent to --gpu-memory-utilization=0.4189 with the old behavior.
(EngineCore pid=170) INFO 01-01 00:24:50 [kv_cache_utils.py:1744] GPU KV cache size: 684,056 tokens
(EngineCore pid=170) INFO 01-01 00:24:50 [kv_cache_utils.py:1745] Maximum concurrency for 262,144 tokens per request: 2.61x
(APIServer pid=1) INFO 01-01 00:27:01 [launcher.py:36] Route: /v1/chat/completions, Methods: POST
"""


# ── the profile: what one start said ─────────────────────────────────────────


def test_the_startup_log_is_the_profile():
    p = llm_context.parse_vllm_startup(VLLM_LOG, total_bytes=121 * GiB)
    assert p is not None
    assert p.weights_bytes == pytest.approx(18.74 * GiB, rel=1e-3)
    assert p.kv_bytes == pytest.approx(42.23 * GiB, rel=1e-3)
    assert p.kv_tokens == 684_056
    assert p.window == 262_144
    assert p.utilization == pytest.approx(0.42)
    assert p.per_token_bytes == pytest.approx(42.23 * GiB / 684_056, rel=1e-3)   # ~66 KB
    assert p.concurrency == pytest.approx(2.61, abs=0.01)


def test_the_overhead_is_what_the_pool_held_beyond_weights_and_cache():
    """The pool was 0.42 × 121 GB = 50.8 GB; weights 18.74 + cache 42.23 = 61 GB — MORE
    than the pool. vLLM's accounting on a unified-memory box is not the pool's
    arithmetic; the overhead is clamped at zero, never negative."""
    p = llm_context.parse_vllm_startup(VLLM_LOG, total_bytes=121 * GiB)
    assert p.overhead_bytes == 0
    tight = VLLM_LOG.replace("Available KV cache memory: 42.23 GiB",
                              "Available KV cache memory: 27.85 GiB")
    p2 = llm_context.parse_vllm_startup(tight, total_bytes=121 * GiB)
    assert p2.overhead_bytes == pytest.approx((0.42 * 121 - 18.74 - 27.85) * GiB, rel=1e-2)


def test_only_a_complete_start_is_a_profile():
    """A container's log keeps every start; the last one may still be loading. The lines
    of ONE start are taken together — the latest complete one — never mixed."""
    half = VLLM_LOG + "(EngineCore pid=9) INFO Model loading took 30.00 GiB memory and 1 seconds\n"
    p = llm_context.parse_vllm_startup(half, total_bytes=121 * GiB)
    assert p.weights_bytes == pytest.approx(18.74 * GiB, rel=1e-3), "the complete start"
    assert llm_context.parse_vllm_startup("nothing here", total_bytes=121 * GiB) is None
    assert llm_context.parse_vllm_startup(
        "Model loading took 18.74 GiB memory\n", total_bytes=121 * GiB) is None


def test_the_utilization_comes_from_the_log_or_the_caller():
    old = "\n".join(l for l in VLLM_LOG.splitlines() if "equivalent to" not in l)
    assert llm_context.parse_vllm_startup(old, total_bytes=121 * GiB, utilization=0.5).utilization == 0.5
    assert llm_context.parse_vllm_startup(old, total_bytes=121 * GiB) is None, \
        "no pool fraction anywhere: the overhead cannot be derived, so no profile"


def test_a_profile_round_trips_through_its_record():
    p = llm_context.parse_vllm_startup(VLLM_LOG, total_bytes=121 * GiB)
    rec = p.as_record(fingerprint="abc123")
    assert rec["fingerprint"] == "abc123" and rec["kv_tokens"] == 684_056
    back = llm_context.VllmProfile.from_record(rec)
    assert back == p
    assert llm_context.VllmProfile.from_record({"kv_tokens": 5}) is None
    assert llm_context.VllmProfile.from_record(None) is None


# ── the concurrency the cache really holds ───────────────────────────────────


def test_concurrency_is_the_cache_divided_by_the_window_floored_at_one():
    assert llm_context.vllm_concurrency(kv_tokens=684_056, window=262_144) == 2
    assert llm_context.vllm_concurrency(kv_tokens=684_056, window=32_768) == 20
    assert llm_context.vllm_concurrency(kv_tokens=100, window=262_144) == 1
    assert llm_context.vllm_concurrency(kv_tokens=None, window=262_144) is None
    assert llm_context.vllm_concurrency(kv_tokens=684_056, window=None) is None


# ── the plan ─────────────────────────────────────────────────────────────────

PROFILE = llm_context.parse_vllm_startup(VLLM_LOG, total_bytes=121 * GiB)


def _plan(*, available_gib, profile=PROFILE, operator_util=0.75, max_seqs=8, window=262_144,
          reserve_gib=8, total_gib=121):
    return llm_context.vllm_plan(
        total_bytes=total_gib * GiB, available_bytes=available_gib * GiB,
        reserve_bytes=reserve_gib * GiB, operator_utilization=operator_util,
        operator_max_seqs=max_seqs, window=window, profile=profile,
        min_utilization=0.30)


def test_with_a_profile_the_pool_is_what_n_requests_need_not_what_is_free():
    """110 GB free. The default rule takes the operator's 0.75 = 91 GB. The need for
    8 × 262k tokens at 66 KB/token is 8 × 17.3 GB + 18.7 GB weights = 157 GB — MORE than
    the box; so N is what fits: (cap/1.05 − 18.74) / 17.3 = 4 requests, pool = the cap."""
    plan = _plan(available_gib=110)
    assert plan.refusal is None
    assert plan.max_num_seqs == 4
    # 4 requests: (18.74 + 4 × 16.18) × 1.05 = 87.6 GiB → 0.73 of 121 — under the 0.75
    # the operator allows, because the 5th request would not fit and nothing else is owed.
    assert plan.utilization == pytest.approx(0.73, abs=0.011)
    assert plan.utilization < 0.75
    assert "4" in plan.reason and "lowered" in plan.reason


def test_a_small_window_needs_a_small_pool_even_on_an_empty_box():
    """At 32k tokens a request costs 2.1 GB; 8 of them plus weights is 36 GB — take that
    (+5 %), not the 91 GB the operator allows. This is the memory nobody else lost."""
    plan = _plan(available_gib=110, window=32_768)
    assert plan.max_num_seqs == 8
    need = (18.74 + 8 * 32_768 * PROFILE.per_token_bytes / GiB) * 1.05
    assert plan.utilization == pytest.approx(need / 121, abs=0.011)
    assert plan.utilization < 0.40


def test_the_pool_is_never_above_the_operators_value_or_the_box(monkeypatch):
    plan = _plan(available_gib=110, window=32_768, operator_util=0.25)
    assert plan.utilization <= 0.25
    assert plan.max_num_seqs == 4, "at 0.25 (30 GB) only 4 requests fit beside the weights"
    tight = _plan(available_gib=40, window=32_768)
    assert tight.utilization <= (40 - 8) / 121 + 1e-9
    assert tight.max_num_seqs == 5, "(32/1.05 − 18.74) / 2.02 GiB per request"


def test_a_box_that_cannot_hold_one_request_refuses_with_the_numbers():
    plan = _plan(available_gib=40)                       # 32 GB usable, one 262k request = 36 GB
    assert plan.utilization is None and plan.max_num_seqs is None
    assert "one 262,144-token request" in plan.refusal
    assert "18.7 GB" in plan.refusal and "16.2 GB" in plan.refusal   # weights, one request's cache


def test_the_margin_is_part_of_what_one_request_needs():
    """44 GB free → 36 GB usable. One request is 34.9 GB of weights + cache; with the
    5 % headroom vLLM's own accounting needs, 36.7 GB — and that is the number that
    decides. A start that "just fits" without the margin is the one that fails at the
    KV-cache allocation after five minutes of loading."""
    assert _plan(available_gib=44).refusal is not None
    assert _plan(available_gib=46).refusal is None


def test_the_pool_fraction_never_exceeds_what_is_free_even_after_rounding_up():
    """40 GB free → 32 GB usable = 0.2645 of the box; six 31k-token requests plus the
    weights come to 31.7 GB = 0.2622, which rounds UP to 0.27 — 32.7 GB the box does not
    have. The fraction is capped at what is free (0.26); the margin absorbs the loss."""
    plan = _plan(available_gib=40, window=31_000)
    assert plan.max_num_seqs == 6
    assert plan.utilization == pytest.approx(0.26)


def test_without_a_profile_the_default_rule_applies():
    """The first start of a model on a box: no numbers, so the pool is the operator's
    value lowered to what is free, concurrency untouched — and that start's log is the
    profile for the next."""
    plan = _plan(available_gib=60, profile=None)
    assert plan.utilization == pytest.approx(0.42, abs=0.005)
    assert plan.max_num_seqs is None and plan.refusal is None
    fits = _plan(available_gib=115, profile=None)
    assert fits.utilization is None and fits.max_num_seqs is None, "as configured"
    refused = _plan(available_gib=20, profile=None)
    assert refused.refusal is not None and "0.30" in refused.refusal


def test_no_readings_no_plan():
    plan = llm_context.vllm_plan(total_bytes=None, available_bytes=None, reserve_bytes=0,
                                 operator_utilization=0.75, operator_max_seqs=8,
                                 window=262_144, profile=PROFILE, min_utilization=0.3)
    assert plan == llm_context.VllmPlan(None, None, None, plan.reason)


def test_an_unknown_window_cannot_count_requests_so_only_the_pool_is_sized():
    plan = _plan(available_gib=60, window=None)
    assert plan.max_num_seqs is None
    assert plan.utilization == pytest.approx(0.42, abs=0.005)


def test_an_operator_without_max_num_seqs_is_planned_at_the_lanes_par():
    plan = _plan(available_gib=110, window=32_768, max_seqs=3)
    assert plan.max_num_seqs == 3


# ── the argv: --max-num-seqs, the fingerprint, the names ─────────────────────

ARGV = ["--model", "/models/base.chat-model-a-dense-nvfp4-vllm", "--served-model-name",
        "chat-model-a:latest", "--max-model-len", "262144", "--max-num-seqs", "8",
        "--gpu-memory-utilization", "0.42", "--reasoning-parser", "qwen3"]


def test_max_num_seqs_is_read_and_rewritten_like_the_pool():
    assert sc.argv_max_seqs(ARGV) == 8
    assert sc.argv_max_seqs(["--model", "m"]) is None
    assert sc.argv_with_max_seqs(ARGV, "4")[7] == "4"
    assert sc.argv_with_max_seqs(["--model", "m"], "4") == ["--model", "m", "--max-num-seqs", "4"]
    assert sc.argv_with_max_seqs(["--max-num-seqs=8"], "4") == ["--max-num-seqs=4"]


def test_the_fingerprint_ignores_only_what_the_engine_itself_changes():
    """A profile belongs to a model AND the flags that shape its memory (dtype, kv cache
    dtype, batched tokens…). The three flags the engine manages must not invalidate it —
    they are exactly what changes between starts."""
    base = sc.argv_fingerprint(ARGV)
    assert base == sc.argv_fingerprint(sc.argv_with_max_seqs(ARGV, "2"))
    assert base == sc.argv_fingerprint(sc.argv_with_utilization(ARGV, "0.9"))
    assert base == sc.argv_fingerprint(sc.argv_with_context(ARGV, "8192"))
    assert base != sc.argv_fingerprint(ARGV + ["--kv-cache-dtype", "fp8"])
    assert base != sc.argv_fingerprint(sc._argv_with_flag(ARGV, "--model", "/models/other"))
    assert sc.argv_fingerprint(None) == sc.argv_fingerprint([])


def test_the_served_name_is_what_the_catalog_keys_on():
    assert sc.argv_served_name(ARGV) == "chat-model-a:latest"
    assert sc.argv_served_name(["--model", "/models/x"]) == "/models/x", "vLLM serves the path"
    assert sc.argv_served_name([]) is None


VLLM = {
    "Id": "abc", "Config": {"Image": "vllm/vllm-openai:latest", "Cmd": list(ARGV),
                            "Env": [], "Labels": {}},
    "HostConfig": {"NetworkMode": "host"}, "NetworkSettings": {"Networks": {}},
    "State": {"Running": True, "StartedAt": "2024-01-01T00:00:00Z"},
}


def test_a_concurrency_change_recreates_and_remembers_the_operators_value():
    plan = sc.plan_context(VLLM, "vllm", None, name="vllm", max_num_seqs=4)
    assert plan is not None
    assert sc.argv_max_seqs(plan.payload["Cmd"]) == 4
    assert plan.payload["Labels"][sc.MAX_SEQS_LABEL] == "8"
    assert sc.UTILIZATION_LABEL not in plan.payload["Labels"], "the pool was not touched"
    lowered = {**VLLM, "Config": {**VLLM["Config"], "Cmd": plan.payload["Cmd"],
                                  "Labels": plan.payload["Labels"]}}
    assert sc.operator_max_seqs(lowered) == 8
    assert sc.operator_max_seqs(VLLM) == 8
    assert sc.plan_context(lowered, "vllm", None, name="vllm", max_num_seqs=4) is None
    back = sc.plan_context(lowered, "vllm", None, name="vllm", max_num_seqs=8)
    assert sc.argv_max_seqs(back.payload["Cmd"]) == 8
    assert back.payload["Labels"][sc.MAX_SEQS_LABEL] == "8", "still remembered"


def test_pool_window_and_concurrency_can_change_in_one_recreate():
    plan = sc.plan_context(VLLM, "vllm", 131072, name="vllm", utilization=0.3, max_num_seqs=2)
    argv = plan.payload["Cmd"]
    assert (sc.argv_context(argv), sc.argv_utilization(argv), sc.argv_max_seqs(argv)) == \
        (131072, pytest.approx(0.3), 2)


def test_ollama_has_no_such_flag():
    ollama = {"Id": "x", "Config": {"Image": "ollama/ollama", "Cmd": ["serve"], "Env": [],
                                    "Labels": {}}, "HostConfig": {"NetworkMode": "bridge"},
              "NetworkSettings": {"Networks": {}}}
    assert sc.plan_context(ollama, "ollama", None, name="ollama", max_num_seqs=4) is None
