"""A box must advertise the LLM server it ACTUALLY serves — not one it might have.

WHY THIS EXISTS (the incident it pins). ``worker_heartbeats.llm_servers_available``
(migration 0014) is documented as OBSERVED capability, but nothing ever observed
anything: the column DEFAULTs to ``{ollama}`` and the worker published a static
``EngineConfig.llm_servers_available`` that also defaults to ``["ollama"]``. So a GPU
box running NO llm server at all still advertised "ollama", the operator panel drew an
"OLLAMA · ON" chip for it, and the box quietly dispatched every LLM call to ANOTHER
box's server over the network — its own GPU pinned at 0% while it held a GPU claim slot
for hours. Nothing in the system could show that, because nothing asked.

So: probe the endpoint, publish what answers (``[]`` when nothing does), and record
WHICH endpoint the box dials so "this box is borrowing that box's GPU" is visible
rather than inferred.
"""

from __future__ import annotations

import pytest

from queue_workflows import llm_probe


# ── the probe: what actually answers ─────────────────────────────────────────


def _fake_get(responses: dict[str, int | None]):
    """Injectable HTTP seam: url -> status (or None for 'no answer')."""
    calls: list[str] = []

    def get(url: str, timeout_s: float) -> int | None:
        calls.append(url)
        return responses.get(url)

    get.calls = calls  # type: ignore[attr-defined]
    return get


def test_ollama_answers_is_advertised_as_ollama():
    get = _fake_get({"http://box:11434/api/tags": 200})
    assert llm_probe.probe_llm_servers("http://box:11434", get_fn=get) == ["ollama"]


def test_vllm_answers_only_openai_route_is_advertised_as_vllm():
    # vLLM serves /v1/models but NOT ollama's /api/tags — that's the discriminator.
    get = _fake_get({"http://box:8000/api/tags": 404, "http://box:8000/v1/models": 200})
    assert llm_probe.probe_llm_servers("http://box:8000", get_fn=get) == ["vllm"]


def test_ollama_wins_when_both_routes_answer():
    """ollama ALSO serves the OpenAI-compatible /v1/models, so /api/tags is checked
    first — otherwise every ollama box would be mislabelled a vllm box."""
    get = _fake_get({"http://box:11434/api/tags": 200, "http://box:11434/v1/models": 200})
    assert llm_probe.probe_llm_servers("http://box:11434", get_fn=get) == ["ollama"]


def test_nothing_answers_advertises_EMPTY_not_ollama():
    """THE BUG. No server ⇒ advertise nothing. The old code advertised ['ollama']."""
    get = _fake_get({})  # every GET returns None (connection refused / no route)
    assert llm_probe.probe_llm_servers("http://box:11434", get_fn=get) == []


def test_probe_never_raises_on_a_broken_endpoint():
    def boom(url: str, timeout_s: float):
        raise OSError("network is down")

    assert llm_probe.probe_llm_servers("http://box:11434", get_fn=boom) == []


def test_empty_base_url_probes_nothing():
    get = _fake_get({"http://box:11434/api/tags": 200})
    assert llm_probe.probe_llm_servers("", get_fn=get) == []
    assert llm_probe.probe_llm_servers(None, get_fn=get) == []
    assert get.calls == []  # type: ignore[attr-defined]


# ── ComfyUI: a THIRD server kind that can hold the box (diffusion) ────────────
#
# broker_parrot must arbitrate every serving path on a box, not just the two LLM
# servers. ComfyUI has no "current checkpoint" API, but its /system_stats reports the
# torch allocator's VRAM: a loaded diffusion model shows GBs of torch_vram in use,
# where a bare ComfyUI (CUDA context only) sits near zero. That is the honest
# residency signal, and /free is the honest evict lever.


def _fake_get_json(url_to_payload: dict):
    calls: list[str] = []

    def get_json(url: str, timeout_s: float):
        calls.append(url)
        if url not in url_to_payload:
            raise OSError("connection refused")
        val = url_to_payload[url]
        if isinstance(val, Exception):
            raise val
        return val

    get_json.calls = calls  # type: ignore[attr-defined]
    return get_json


def _stats(torch_used_bytes: int):
    """A /system_stats payload where the cuda device's torch allocator holds
    ``torch_used_bytes`` (total - free)."""
    total = 128 << 30
    return {"devices": [{
        "name": "cuda:0", "type": "cuda", "index": 0,
        "vram_total": total, "vram_free": total - torch_used_bytes,
        "torch_vram_total": total, "torch_vram_free": total - torch_used_bytes,
    }]}


def test_comfyui_with_a_model_loaded_is_resident():
    # 20 GiB of torch VRAM in use ⇒ a diffusion model is resident.
    gj = _fake_get_json({"http://box:8188/system_stats": _stats(20 << 30)})
    assert llm_probe.comfyui_loaded("http://box:8188", get_json_fn=gj) is True


def test_comfyui_up_but_no_model_is_not_resident():
    # Bare ComfyUI: only the CUDA context (~hundreds of MB) — below the model floor.
    gj = _fake_get_json({"http://box:8188/system_stats": _stats(256 << 20)})
    assert llm_probe.comfyui_loaded("http://box:8188", get_json_fn=gj) is False


def test_comfyui_unreachable_is_not_resident():
    gj = _fake_get_json({})  # nothing answers
    assert llm_probe.comfyui_loaded("http://box:8188", get_json_fn=gj) is False


def test_comfyui_probe_never_raises():
    gj = _fake_get_json({"http://box:8188/system_stats": OSError("boom")})
    assert llm_probe.comfyui_loaded("http://box:8188", get_json_fn=gj) is False
    assert llm_probe.comfyui_loaded("", get_json_fn=gj) is False
    assert llm_probe.comfyui_loaded(None, get_json_fn=gj) is False


def test_comfyui_free_posts_the_unload_payload():
    posted: list[tuple] = []

    def post(url, payload, timeout_s):
        posted.append((url, payload))
        return 200

    assert llm_probe.comfyui_free("http://box:8188", post_fn=post) is True
    assert posted == [("http://box:8188/free",
                       {"unload_models": True, "free_memory": True})]


def test_comfyui_free_never_raises_and_reports_failure():
    def boom(url, payload, timeout_s):
        raise OSError("down")

    assert llm_probe.comfyui_free("http://box:8188", post_fn=boom) is False
    assert llm_probe.comfyui_free("", post_fn=boom) is False


# ── locality: is this box serving itself, or borrowing another box's GPU? ─────


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:11434",
    "http://localhost:11434",
    "http://host.docker.internal:11434",  # the docker host gateway == THIS box
    "http://[::1]:11434",
])
def test_local_endpoints_are_local(url):
    assert llm_probe.is_local_endpoint(url) is True


@pytest.mark.parametrize("url", [
    "http://box-b-fast:11434",
    "http://10.0.0.2:11434",
    "http://10.0.0.3:11434",
])
def test_remote_endpoints_are_not_local(url):
    """A GPU box pointing here is dispatching to ANOTHER machine's GPU: its own stays
    idle while it still occupies a claim slot. That must be visible, not silent."""
    assert llm_probe.is_local_endpoint(url) is False


def test_unparseable_endpoint_is_not_local():
    assert llm_probe.is_local_endpoint("") is False
    assert llm_probe.is_local_endpoint(None) is False


# ── GPU placement: is the server ACTUALLY on the GPU, or fell back to CPU? ────
#
# THE SECOND INCIDENT. A box's ollama can silently serve a model on CPU — it lost
# the GPU to a cgroup event (NVML "Unknown Error"), or the model didn't fit VRAM so
# ollama offloaded it. `ollama ps` then reads "100% CPU". The endpoint still ANSWERS,
# so the existence probe above says "ollama" and the box keeps claiming GPU jobs it
# runs at CPU speed. The policy: ONLY use a box on GPU. A model not FULLY on GPU
# (size_vram < size) means either a fault or "not enough VRAM" — both ⇒ skip.


def _fake_ps(payload):
    """Injectable JSON seam: returns `payload` for /api/ps (or raises for None)."""
    def get_json(url, timeout_s):
        if payload is None:
            raise OSError("unreachable")
        return payload
    return get_json


def test_placement_gpu_when_model_fully_in_vram():
    ps = _fake_ps({"models": [{"name": "llama3", "size": 30_000, "size_vram": 30_000}]})
    assert llm_probe.probe_gpu_placement("http://box:11434", get_json_fn=ps) == "gpu"


def test_placement_cpu_when_size_vram_is_zero():
    """The CPU-fallback state: model loaded, size_vram=0 → 100% CPU."""
    ps = _fake_ps({"models": [{"name": "llama3", "size": 30_000, "size_vram": 0}]})
    assert llm_probe.probe_gpu_placement("http://box:11434", get_json_fn=ps) == "cpu"


def test_placement_cpu_when_only_partially_offloaded():
    """Partial offload is NOT 'only GPU' — some layers run on CPU. Treated as cpu
    (= not enough VRAM to fit fully ⇒ skip, route to a box that fits)."""
    ps = _fake_ps({"models": [{"name": "llama3", "size": 30_000, "size_vram": 18_000}]})
    assert llm_probe.probe_gpu_placement("http://box:11434", get_json_fn=ps) == "cpu"


def test_placement_unknown_when_no_model_loaded():
    """Cold server — nothing loaded, so placement can't be known yet. The box is
    allowed to claim (it'll load on GPU if it can); a later probe catches a fallback."""
    ps = _fake_ps({"models": []})
    assert llm_probe.probe_gpu_placement("http://box:11434", get_json_fn=ps) == "unknown"


def test_placement_unknown_on_unreachable_or_bad_payload():
    assert llm_probe.probe_gpu_placement("http://box:11434", get_json_fn=_fake_ps(None)) == "unknown"
    assert llm_probe.probe_gpu_placement("http://box:11434", get_json_fn=_fake_ps("nonsense")) == "unknown"
    assert llm_probe.probe_gpu_placement("", get_json_fn=_fake_ps({"models": []})) == "unknown"


def test_placement_never_raises():
    def boom(url, timeout_s):
        raise RuntimeError("kaboom")
    assert llm_probe.probe_gpu_placement("http://box:11434", get_json_fn=boom) == "unknown"


# ── gpu_usable: the one predicate the claim gate reads ───────────────────────


def test_gpu_usable_true_on_gpu_or_unknown():
    assert llm_probe.gpu_usable(["ollama"], "gpu") is True
    assert llm_probe.gpu_usable(["ollama"], "unknown") is True   # cold ⇒ optimistic
    assert llm_probe.gpu_usable(["vllm"], "gpu") is True


def test_gpu_usable_false_on_cpu_or_no_server():
    assert llm_probe.gpu_usable(["ollama"], "cpu") is False      # the fault we skip
    assert llm_probe.gpu_usable([], "unknown") is False          # no server at all
    assert llm_probe.gpu_usable([], "gpu") is False
