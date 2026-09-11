"""Bringing the vLLM sidecar up on a REQUESTED model — the Phase-6 seam, finally built.

`vllm.py` has documented this hole since it was written:

    ensure_up_fn(model_id) -> None — Default is a NO-OP: today the bring-up is owned by
    docker `restart: unless-stopped` + a (Phase-6) entrypoint that reads the desired model
    from config.

Consequence, measured on a live fleet: a box whose `worker_controls.
llm_server_type` says `vllm` served whatever model was BAKED into its compose file, and a
freshly quantized model sitting on that box's own disk could not be reached at all. The
state machine already handles a switch (RELOADING → stop → cold start → health poll); it
just had nothing to call.

The domain here is pure: given the sidecar's CURRENT argv and a wanted model, say what the
argv should become. Docker is injected, so these run with no daemon.
"""
from __future__ import annotations

import pytest

from queue_workflows.llm_backends.vllm_sidecar import (
    SidecarSpec,
    desired_argv,
    ensure_sidecar,
)


# ── the pure part ────────────────────────────────────────────────────
def test_the_model_flag_is_REPLACED_not_appended():
    argv = ["--model", "org/old-35b", "--served-model-name", "chat-model-a", "--port", "8000"]
    assert desired_argv(argv, "/models/new-nvfp4") == [
        "--model", "/models/new-nvfp4", "--served-model-name", "chat-model-a", "--port", "8000"]


def test_a_sidecar_with_no_model_flag_GAINS_one():
    assert desired_argv(["--port", "8000"], "m") == ["--port", "8000", "--model", "m"]


def test_the_equals_form_is_handled_too():
    assert desired_argv(["--model=old", "--port", "8000"], "new") == [
        "--model=new", "--port", "8000"]


def test_everything_else_is_left_exactly_alone():
    """The sidecar's tuning — memory fraction, reasoning parser, served names — is the
    operator's. A bring-up that rewrote it would silently change how the box serves."""
    argv = ["--model", "old", "--gpu-memory-utilization", "0.75",
            "--reasoning-parser", "qwen3", "--served-model-name", "chat-model-a", "chat-model-a:latest"]
    out = desired_argv(argv, "new")
    assert out[1] == "new"
    assert out[2:] == argv[2:]


# ── the decision ─────────────────────────────────────────────────────
def _spec(model: str, running: bool = True) -> SidecarSpec:
    return SidecarSpec(name="vllm", running=running,
                       argv=["--model", model, "--port", "8000"], image="vllm/vllm-openai")


def test_a_sidecar_already_serving_the_wanted_model_is_LEFT_ALONE():
    """A needless recreate costs minutes of weight loading — the exact cost the
    vllm_min_queue gate exists to amortise."""
    calls = []
    ensure_sidecar("m1", inspect_fn=lambda: _spec("m1"),
                   recreate_fn=lambda spec, argv: calls.append(argv))
    assert calls == []


def test_a_sidecar_serving_a_DIFFERENT_model_is_recreated_with_the_new_argv():
    calls = []
    ensure_sidecar("m2", inspect_fn=lambda: _spec("m1"),
                   recreate_fn=lambda spec, argv: calls.append(argv))
    assert calls == [["--model", "m2", "--port", "8000"]]


def test_a_STOPPED_sidecar_is_started_even_when_its_baked_model_already_matches():
    calls = []
    ensure_sidecar("m1", inspect_fn=lambda: _spec("m1", running=False),
                   recreate_fn=lambda spec, argv: calls.append(argv))
    assert calls == [["--model", "m1", "--port", "8000"]]


def test_no_sidecar_at_all_RAISES_rather_than_pretending_it_started_one():
    """Silence here would let ensure_ready poll health for 90 s and then blame the model.
    The caller can soft-degrade to ollama, but only if it is told."""
    with pytest.raises(RuntimeError, match="no vLLM sidecar"):
        ensure_sidecar("m", inspect_fn=lambda: None, recreate_fn=lambda *_: None)


def test_a_recreate_that_fails_propagates():
    def boom(spec, argv):
        raise OSError("docker daemon unreachable")
    with pytest.raises(OSError):
        ensure_sidecar("m2", inspect_fn=lambda: _spec("m1"), recreate_fn=boom)


# ── the recreate must PRESERVE everything but the command ────────────
def test_recreate_payload_keeps_the_containers_own_config_and_swaps_only_Cmd():
    """The model comes from the NODE-JOB (ensure_ready(model_id) → ensure_up_fn), so the
    recreate must carry it in the container's Cmd — not through an env var a compose file
    would have to interpolate. Everything else (image, env, ports, volumes, devices,
    restart policy) is the operator's and must survive untouched, or a model switch
    silently becomes a redeploy."""
    from queue_workflows.llm_backends.vllm_sidecar import recreate_payload

    inspected = {
        "Config": {"Image": "vllm/vllm-openai:latest",
                   "Cmd": ["--model", "old", "--gpu-memory-utilization", "0.75"],
                   "Env": ["NVIDIA_VISIBLE_DEVICES=all"],
                   "Labels": {"com.docker.compose.service": "vllm"}},
        "HostConfig": {"PortBindings": {"8000/tcp": [{"HostPort": "8000"}]},
                       "Binds": ["/srv/models:/models:ro"],
                       "RestartPolicy": {"Name": "unless-stopped"},
                       "DeviceRequests": [{"Count": -1, "Capabilities": [["gpu"]]}]},
    }
    payload = recreate_payload(inspected, ["--model", "/models/new-nvfp4",
                                           "--gpu-memory-utilization", "0.75"])

    assert payload["Cmd"] == ["--model", "/models/new-nvfp4",
                              "--gpu-memory-utilization", "0.75"]
    assert payload["Image"] == "vllm/vllm-openai:latest"
    assert payload["Env"] == ["NVIDIA_VISIBLE_DEVICES=all"]
    assert payload["HostConfig"]["Binds"] == ["/srv/models:/models:ro"]
    assert payload["HostConfig"]["RestartPolicy"] == {"Name": "unless-stopped"}
    assert payload["HostConfig"]["DeviceRequests"][0]["Capabilities"] == [["gpu"]]
    assert payload["HostConfig"]["PortBindings"]["8000/tcp"][0]["HostPort"] == "8000"


def test_recreate_payload_needs_no_env_var_to_carry_the_model():
    """Guards the mistake this replaced: passing the model through the ENVIRONMENT means a
    compose file must interpolate it, which makes the switch depend on a file the engine
    does not own. The model belongs in Cmd, where the node-job put it."""
    from queue_workflows.llm_backends.vllm_sidecar import recreate_payload

    payload = recreate_payload(
        {"Config": {"Image": "i", "Cmd": ["--model", "old"], "Env": ["A=1"]},
         "HostConfig": {}},
        ["--model", "new"])
    assert not any("VLLM_MODEL" in e for e in payload["Env"])
    assert payload["Cmd"] == ["--model", "new"]


def test_the_socket_timeout_outlasts_a_full_stop_remove_create_start_cycle():
    """A recreate is stop + remove + create + start. The first version used a 30 s socket
    timeout against a 30 s stop grace period, and on a live box the call timed out
    MID-CYCLE — container removed, replacement not created, sidecar down. The socket must
    outlast the whole cycle; the stop grace is what should be short."""
    import re

    from queue_workflows.llm_backends import vllm

    src = (__import__("pathlib").Path(vllm.__file__)).read_text()
    sock = int(re.search(r"self\.sock\.settimeout\((\d+)\)", src).group(1))
    stop = int(re.search(r"/stop\?t=(\d+)", src).group(1))
    assert sock >= 120, f"socket timeout {sock}s is too short for a recreate cycle"
    assert stop <= 15, f"stop grace {stop}s keeps the box dark longer than needed"
    assert sock > stop * 4
