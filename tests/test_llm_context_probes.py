"""The observations the context arithmetic runs on — what each server says about its window.

Same contract as every probe in ``llm_probe``: an injectable fetcher, and NEVER raise —
an unreachable or unparseable endpoint answers "unknown", which the arithmetic treats as
"cannot prove a bigger window", never as a guess. The ollama payload shapes below are the
ones ollama 0.33.2 returns (/api/ps carries ``context_length`` per
loaded runner; /api/tags ``size`` per installed tag; /api/show ``model_info``); the vLLM
ones follow its OpenAI model card (``max_model_len``) and Prometheus ``cache_config_info``.
"""
from __future__ import annotations

from queue_workflows import llm_probe

PS = {"models": [{
    "name": "chat-model-a:latest", "size": 18714627931, "size_vram": 18714627931,
    "context_length": 262144, "expires_at": "2024-01-01T00:00:00.000000000Z",
}]}
TAGS = {"models": [
    {"name": "chat-model-a:latest", "size": 17741872154, "digest": "abc"},
    {"name": "chat-expert-b:latest", "size": 16837912464, "digest": "def"},
]}


def _json(payloads: dict):
    def get(url, timeout_s):
        for suffix, body in payloads.items():
            if url.endswith(suffix):
                return body
        raise OSError(f"unexpected {url}")
    return get


def _boom(*_a, **_k):
    raise OSError("connection refused")


# ── ollama ───────────────────────────────────────────────────────────────────


def test_ollama_runners_report_size_and_window_per_loaded_model():
    assert llm_probe.ollama_runners("http://box:11434", get_json_fn=_json({"/api/ps": PS})) == [
        {"name": "chat-model-a:latest", "size": 18714627931, "size_vram": 18714627931,
         "context_length": 262144}]


def test_a_runner_without_a_window_field_reports_none_for_it():
    ps = {"models": [{"name": "old:latest", "size": 10, "size_vram": 10}]}
    (row,) = llm_probe.ollama_runners("http://box", get_json_fn=_json({"/api/ps": ps}))
    assert row["context_length"] is None


def test_no_runner_and_no_server_are_both_an_empty_list():
    assert llm_probe.ollama_runners("http://box", get_json_fn=_json({"/api/ps": {"models": []}})) == []
    assert llm_probe.ollama_runners("http://box", get_json_fn=_boom) == []
    assert llm_probe.ollama_runners(None) == []


def test_ollama_model_sizes_are_the_weights_per_installed_tag():
    assert llm_probe.ollama_model_sizes("http://box", get_json_fn=_json({"/api/tags": TAGS})) == {
        "chat-model-a:latest": 17741872154, "chat-expert-b:latest": 16837912464}
    assert llm_probe.ollama_model_sizes("http://box", get_json_fn=_boom) == {}


def test_ollama_model_info_is_posted_for_one_model():
    seen = {}

    def post(url, payload, timeout_s):
        seen.update(url=url, payload=payload)
        return {"model_info": {"general.architecture": "qwen35",
                               "qwen35.context_length": 262144}}

    info = llm_probe.ollama_model_info("http://box:11434/", "chat-model-a:latest",
                                       post_json_fn=post)
    assert info == {"general.architecture": "qwen35", "qwen35.context_length": 262144}
    assert seen == {"url": "http://box:11434/api/show",
                    "payload": {"model": "chat-model-a:latest"}}


def test_ollama_model_info_is_empty_on_any_failure():
    assert llm_probe.ollama_model_info("http://box", "m", post_json_fn=_boom) == {}
    assert llm_probe.ollama_model_info("http://box", "m",
                                       post_json_fn=lambda *a: {"error": "no"}) == {}
    assert llm_probe.ollama_model_info(None, "m") == {}
    assert llm_probe.ollama_model_info("http://box", "") == {}


# ── vLLM ─────────────────────────────────────────────────────────────────────


def test_vllm_reports_the_window_it_was_started_with():
    models = {"object": "list", "data": [
        {"id": "chat-model-a", "object": "model", "max_model_len": 32768}]}
    assert llm_probe.vllm_max_model_len(
        "http://box:8000", get_json_fn=_json({"/v1/models": models})) == 32768


def test_a_vllm_without_the_field_or_unreachable_is_unknown():
    no_field = {"data": [{"id": "chat-model-a"}]}
    assert llm_probe.vllm_max_model_len("http://box", get_json_fn=_json({"/v1/models": no_field})) is None
    assert llm_probe.vllm_max_model_len("http://box", get_json_fn=_boom) is None
    assert llm_probe.vllm_max_model_len(None) is None


METRICS = """\
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",gpu_memory_utilization="0.85",num_cpu_blocks="0",num_gpu_blocks="6144"} 1.0
vllm:num_requests_running{model_name="chat-model-a"} 0.0
"""


def test_vllm_kv_capacity_is_its_gpu_blocks_times_their_size():
    assert llm_probe.vllm_kv_capacity_tokens(
        "http://box:8000", get_text_fn=lambda url, t: METRICS) == 6144 * 16


def test_vllm_kv_capacity_is_unknown_without_the_gauge():
    assert llm_probe.vllm_kv_capacity_tokens(
        "http://box", get_text_fn=lambda url, t: "vllm:num_requests_running 0\n") is None
    assert llm_probe.vllm_kv_capacity_tokens("http://box", get_text_fn=lambda url, t: "") is None
    assert llm_probe.vllm_kv_capacity_tokens("http://box", get_text_fn=_boom) is None
    assert llm_probe.vllm_kv_capacity_tokens(None) is None
