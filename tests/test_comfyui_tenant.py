"""ComfyUI as a FIRST-CLASS GPU box tenant + a reusable submit client.

Today ComfyUI is evict-only: it can be probed (`comfyui_loaded`) and freed
(`comfyui_free`), and it always LOSES the one-server-per-box arbitration. This adds
the winning half — when a node_job REQUESTS ComfyUI, ComfyUI takes the box:

  * `acquire_box_for_comfyui()` — clear every rival serving kind off the card
    (ollama keep_alive:0 / vLLM stop / in-process unload) via `clear_box_for`, then
    START the box's ComfyUI (host-wired lifecycle), then wait until it answers.
  * `submit_workflow()` — the reusable CLIENT: POST a ComfyUI graph to `/prompt`,
    poll `/history`, return the output files. What a render node calls.
  * `set_comfyui_lifecycle()` — wire the host's start/stop levers (mirror
    `set_inference_server_lifecycle`).
"""
from __future__ import annotations

import pytest

import queue_workflows
from queue_workflows import comfyui, config, llm_probe
from queue_workflows.model_residency import ModelAlreadyLoadedError, Resident


@pytest.fixture(autouse=True)
def _reset():
    cfg = config.get_config()
    cfg.comfyui_start_fn = None
    cfg.comfyui_stop_fn = None
    yield
    cfg.comfyui_start_fn = None
    cfg.comfyui_stop_fn = None


# ── the lifecycle setter wires both hooks ────────────────────────────────────


def test_set_comfyui_lifecycle_wires_both():
    starts, stops = [], []
    queue_workflows.set_comfyui_lifecycle(
        start_fn=lambda: starts.append(1), stop_fn=lambda: stops.append(1),
    )
    cfg = config.get_config()
    cfg.comfyui_start_fn()
    cfg.comfyui_stop_fn()
    assert starts == [1] and stops == [1]


def test_set_comfyui_lifecycle_none_is_noop():
    queue_workflows.set_comfyui_lifecycle(start_fn=None, stop_fn=None)
    cfg = config.get_config()
    assert cfg.comfyui_start_fn is None and cfg.comfyui_stop_fn is None


# ── acquire: ComfyUI WINS the box (evict rivals, then start, then ready) ──────


def test_acquire_evicts_every_rival_kind_then_starts_and_readies():
    fired = {"ollama": None, "vllm": 0, "inproc": 0, "start": 0}

    def unload_ollama(models):
        fired["ollama"] = models

    residents = [
        Resident(server=llm_probe.OLLAMA, model="qwen3:8b", mru=5.0),
        Resident(server=llm_probe.INPROCESS, model="wan-s2v-14b"),
    ]
    evicted = comfyui.acquire_box_for_comfyui(
        residents,
        start_fn=lambda: fired.__setitem__("start", fired["start"] + 1),
        ready_probe=lambda: True,
        unload_ollama=unload_ollama,
        stop_vllm=lambda: fired.__setitem__("vllm", fired["vllm"] + 1),
        unload_inprocess=lambda: fired.__setitem__("inproc", fired["inproc"] + 1),
        label="box-a",
    )
    # every foreign kind's lever fired; the ollama model name was passed through
    assert fired["ollama"] == ["qwen3:8b"]
    assert fired["inproc"] == 1
    assert fired["start"] == 1
    # returns what it evicted (both rivals)
    assert {(r.server, r.model) for r in evicted} == {
        (llm_probe.OLLAMA, "qwen3:8b"),
        (llm_probe.INPROCESS, "wan-s2v-14b"),
    }


def test_acquire_on_a_box_already_only_comfyui_is_a_noop_but_still_starts():
    """No rivals → nothing evicted, but we still ensure ComfyUI is up (idempotent)."""
    starts = []
    residents = [Resident(server=llm_probe.COMFYUI, model=llm_probe.COMFYUI)]
    evicted = comfyui.acquire_box_for_comfyui(
        residents, start_fn=lambda: starts.append(1), ready_probe=lambda: True,
    )
    assert evicted == []
    assert starts == [1]


def test_acquire_raises_if_a_rival_survives_eviction():
    """A rival that won't die must fail the acquire (reuse clear_box_for's reprobe guard)."""
    residents = [Resident(server=llm_probe.OLLAMA, model="qwen3:8b")]
    with pytest.raises(ModelAlreadyLoadedError):
        comfyui.acquire_box_for_comfyui(
            residents,
            start_fn=lambda: None,
            unload_ollama=lambda models: None,        # pretend it fails to unload
            reprobe=lambda: residents,                # still there
            ready_probe=lambda: True,
            label="box-a",
        )


def test_acquire_waits_for_ready_then_succeeds():
    calls = {"n": 0}

    def ready():
        calls["n"] += 1
        return calls["n"] >= 3           # not ready first two polls, ready on the 3rd

    slept = []
    comfyui.acquire_box_for_comfyui(
        [], start_fn=lambda: None, ready_probe=ready,
        poll_s=0.5, ready_timeout_s=10, sleep_fn=lambda s: slept.append(s),
    )
    assert calls["n"] == 3
    assert slept == [0.5, 0.5]           # slept between the two not-ready polls


def test_acquire_times_out_if_never_ready():
    with pytest.raises(comfyui.ComfyUINotReady):
        comfyui.acquire_box_for_comfyui(
            [], start_fn=lambda: None, ready_probe=lambda: False,
            poll_s=1, ready_timeout_s=3, sleep_fn=lambda s: None,
        )


def test_acquire_start_failure_propagates():
    def boom():
        raise RuntimeError("docker start vg-comfy failed")

    with pytest.raises(RuntimeError):
        comfyui.acquire_box_for_comfyui([], start_fn=boom, ready_probe=lambda: True)


# ── submit_workflow: the reusable CLIENT (POST /prompt, poll /history) ────────


def test_submit_workflow_posts_graph_and_returns_outputs():
    graph = {"3": {"class_type": "KSampler", "inputs": {}}}
    posted = {}

    def submit_fn(url, payload, timeout_s):
        posted["url"] = url
        posted["payload"] = payload
        return {"prompt_id": "pid-123"}

    def history_fn(url, timeout_s):
        assert url.endswith("/history/pid-123")
        return {
            "pid-123": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {"9": {"gifs": [{"filename": "vid_00001_.mp4",
                                            "subfolder": "", "type": "output"}]}},
            }
        }

    out = comfyui.submit_workflow(
        "http://box-a:8188", graph,
        submit_fn=submit_fn, history_fn=history_fn, sleep_fn=lambda s: None,
    )
    assert posted["url"].endswith("/prompt")
    assert posted["payload"]["prompt"] == graph
    assert out == [{"filename": "vid_00001_.mp4", "subfolder": "", "type": "output"}]


def test_submit_workflow_raises_on_validation_error():
    def submit_fn(url, payload, timeout_s):
        # ComfyUI 400: prompt_outputs_failed_validation → no prompt_id
        return {"error": {"type": "prompt_outputs_failed_validation"},
                "node_errors": {"37": {}}}

    with pytest.raises(comfyui.ComfyUIError):
        comfyui.submit_workflow(
            "http://box-a:8188", {}, submit_fn=submit_fn,
            history_fn=lambda u, t: {}, sleep_fn=lambda s: None,
        )


def test_submit_workflow_raises_when_run_errors():
    def history_fn(url, timeout_s):
        return {"pid-1": {"status": {"status_str": "error", "completed": False}}}

    with pytest.raises(comfyui.ComfyUIError):
        comfyui.submit_workflow(
            "http://box-a:8188", {},
            submit_fn=lambda u, p, t: {"prompt_id": "pid-1"},
            history_fn=history_fn, sleep_fn=lambda s: None,
        )


def test_submit_workflow_polls_until_complete():
    seq = [
        {},                                                   # not in history yet
        {"pid-1": {"status": {"completed": False}}},          # queued/running
        {"pid-1": {"status": {"status_str": "success", "completed": True},
                   "outputs": {"9": {"images": [{"filename": "img.png",
                                                 "subfolder": "", "type": "output"}]}}}},
    ]
    calls = {"n": 0}

    def history_fn(url, timeout_s):
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    out = comfyui.submit_workflow(
        "http://box-a:8188", {},
        submit_fn=lambda u, p, t: {"prompt_id": "pid-1"},
        history_fn=history_fn, poll_s=0.1, timeout_s=100, sleep_fn=lambda s: None,
    )
    assert out == [{"filename": "img.png", "subfolder": "", "type": "output"}]
    assert calls["n"] >= 3


def test_submit_workflow_times_out():
    with pytest.raises(comfyui.ComfyUITimeout):
        comfyui.submit_workflow(
            "http://box-a:8188", {},
            submit_fn=lambda u, p, t: {"prompt_id": "pid-1"},
            history_fn=lambda u, t: {},          # never completes
            poll_s=1, timeout_s=3, sleep_fn=lambda s: None,
        )


# ── comfyui_reachable: "is the server up?" (independent of a loaded model) ────


def test_comfyui_reachable_true_when_system_stats_answers():
    assert llm_probe.comfyui_reachable(
        "http://box-a:8188", get_json_fn=lambda u, t: {"system": {"comfyui_version": "0.26"}}
    )


def test_comfyui_reachable_false_when_unreachable():
    def boom(url, t):
        raise OSError("connection refused")

    assert llm_probe.comfyui_reachable("http://box-a:8188", get_json_fn=boom) is False
    assert llm_probe.comfyui_reachable("") is False


# ── ensure_comfyui_box: wire the levers from live config + probe ─────────────


def test_ensure_comfyui_box_wires_config_start_and_probe(monkeypatch):
    """The render-node convenience: it needs only inprocess_model + unload_inprocess;
    it pulls the ComfyUI URL, start lever and residents from config/probe."""
    started = []
    queue_workflows.set_comfyui_lifecycle(start_fn=lambda: started.append(1), stop_fn=None)
    residents = [Resident(server=llm_probe.OLLAMA, model="qwen3:8b", mru=1.0)]
    unloaded = []

    evicted = comfyui.ensure_comfyui_box(
        comfyui_url="http://box-a:8188",
        residents_fn=lambda: residents,
        ready_probe=lambda: True,
        unload_ollama=lambda models: unloaded.append(models),
        stop_vllm=lambda: None,
        label="box-a",
    )
    assert started == [1]                       # config start lever fired
    assert unloaded == [["qwen3:8b"]]           # rival ollama evicted
    assert {(r.server, r.model) for r in evicted} == {(llm_probe.OLLAMA, "qwen3:8b")}
