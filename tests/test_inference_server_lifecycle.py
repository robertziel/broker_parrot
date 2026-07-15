"""The GPU ON/OFF toggle also turns the machine's INFERENCE SERVER on/off.

Operator policy: toggling a box's GPU worker OFF should also stop its LLM
server (free the GPU's VRAM), and toggling it ON should start the server — so "GPU
OFF" means the machine's GPU is actually released, not just the worker parked while
ollama keeps a model resident.

The engine exposes a host-wired lifecycle (``set_inference_server_lifecycle`` →
``EngineConfig.inference_server_{start,stop}_fn``); a host wires it to docker-over-UDS
control of its ``broker-ollama`` (or vllm) container. The worker-control OFF path
stops the server before the hard-exit; the worker start path (after the ON park-gate)
starts it. Both are GPU-only, best-effort (a failure never breaks stop/claim), and
default-unset ⇒ no-op (byte-compatible for every consumer that never wires them).
"""

from __future__ import annotations

import pytest

import queue_workflows
from queue_workflows import claim_worker, config, worker_control


@pytest.fixture(autouse=True)
def _reset():
    cfg = config.get_config()
    cfg.inference_server_start_fn = None
    cfg.inference_server_stop_fn = None
    yield
    cfg.inference_server_start_fn = None
    cfg.inference_server_stop_fn = None


# ── the setter wires both hooks ──────────────────────────────────────────────


def test_set_inference_server_lifecycle_wires_both():
    starts, stops = [], []
    queue_workflows.set_inference_server_lifecycle(
        start_fn=lambda: starts.append(1), stop_fn=lambda: stops.append(1),
    )
    cfg = config.get_config()
    cfg.inference_server_start_fn()
    cfg.inference_server_stop_fn()
    assert starts == [1] and stops == [1]


# ── STOP on GPU OFF (worker-control) ─────────────────────────────────────────


class _FakeWorker:
    def __init__(self, queue="gpu"):
        self.queue = queue
        self.host = "box-a-gpu"
        self.requeued = 0

    def requeue_inflight_for_control(self):
        self.requeued += 1
        return 0


def test_gpu_off_stops_the_inference_server(monkeypatch):
    stopped = []
    config.get_config().inference_server_stop_fn = lambda: stopped.append(1)
    # OFF row for a gpu worker; hard policy. Assert the server was stopped, before exit.
    monkeypatch.setattr(
        worker_control, "get_worker_control",
        lambda h, q, **k: {"desired_state": "off", "stop_policy": "hard"},
    )
    exits = []
    w = _FakeWorker(queue="gpu")
    watcher = worker_control.WorkerControlWatcher(
        worker=w, on_exit=lambda code: exits.append(code),
    )
    assert watcher.check_once() is True
    assert stopped == [1]       # server stopped
    assert exits == [worker_control.EXIT_CONTROL_HARD_STOP]  # then hard-exit


def test_cpu_off_does_not_stop_the_inference_server(monkeypatch):
    """A CPU/download lane turning OFF must NOT stop the machine's ollama — only the
    GPU toggle governs the inference server."""
    stopped = []
    config.get_config().inference_server_stop_fn = lambda: stopped.append(1)
    monkeypatch.setattr(
        worker_control, "get_worker_control",
        lambda h, q, **k: {"desired_state": "off", "stop_policy": "hard"},
    )
    w = _FakeWorker(queue="cpu")
    worker_control.WorkerControlWatcher(
        worker=w, on_exit=lambda code: None,
    ).check_once()
    assert stopped == []


def test_off_stop_hook_failure_never_blocks_the_hard_exit(monkeypatch):
    def boom():
        raise RuntimeError("docker socket down")

    config.get_config().inference_server_stop_fn = boom
    monkeypatch.setattr(
        worker_control, "get_worker_control",
        lambda h, q, **k: {"desired_state": "off", "stop_policy": "hard"},
    )
    exits = []
    worker_control.WorkerControlWatcher(
        worker=_FakeWorker("gpu"), on_exit=lambda code: exits.append(code),
    ).check_once()
    assert exits == [worker_control.EXIT_CONTROL_HARD_STOP]  # exit still happened


def test_no_hook_wired_is_a_noop(monkeypatch):
    monkeypatch.setattr(
        worker_control, "get_worker_control",
        lambda h, q, **k: {"desired_state": "off", "stop_policy": "hard"},
    )
    exits = []
    # inference_server_stop_fn is None (fixture) → nothing to call, exit as normal
    worker_control.WorkerControlWatcher(
        worker=_FakeWorker("gpu"), on_exit=lambda code: exits.append(code),
    ).check_once()
    assert exits == [worker_control.EXIT_CONTROL_HARD_STOP]


# ── START on GPU ON (worker enters claiming) ─────────────────────────────────


class _Cache:
    current_model = None


def _gpu_worker():
    return claim_worker.ClaimWorker(queue="gpu", host="box-a-gpu", model_cache=_Cache())


def test_gpu_worker_start_hook_starts_the_server():
    started = []
    config.get_config().inference_server_start_fn = lambda: started.append(1)
    _gpu_worker()._start_inference_server()
    assert started == [1]


def test_cpu_worker_never_starts_the_server():
    started = []
    config.get_config().inference_server_start_fn = lambda: started.append(1)
    claim_worker.ClaimWorker(queue="cpu", host="box-a-cpu")._start_inference_server()
    assert started == []


def test_start_hook_failure_never_crashes_the_worker():
    def boom():
        raise RuntimeError("docker socket down")

    config.get_config().inference_server_start_fn = boom
    _gpu_worker()._start_inference_server()  # must not raise
