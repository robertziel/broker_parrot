"""Bring the vLLM sidecar up on a REQUESTED model — the Phase-6 seam :mod:`vllm` documents.

WHY THIS EXISTS. :class:`~queue_workflows.llm_backends.vllm.VLLMBackend` has always known
how to switch models — ``ensure_ready`` runs a full RELOADING → stop → cold start → health
poll state machine — but its ``ensure_up_fn`` default was a NO-OP, so the "cold start" leg
relied on docker ``restart: unless-stopped`` bringing the container back on its **baked**
``--model`` flag. The backend even says so, and fails loud on the resulting mismatch:

    the default ``ensure_up_fn`` is a no-op that relies on docker ``restart:
    unless-stopped`` bringing the sidecar back up on its baked ``--model`` flag, which may
    not be the newly-requested one

Measured consequence on a live fleet: a box whose ``worker_controls.
llm_server_type`` said ``vllm`` served whatever model was baked into a hand-maintained
compose file, and a freshly quantized model sitting on that box's own disk was unreachable
— not because anything was broken, but because nothing could ask for it.

DESIGN. The decision is a PURE function of (what is running, what is wanted); docker is an
injected seam, so this whole module is unit-testable with no daemon. That is the same shape
:mod:`model_catalog` and :mod:`model_cache` use, and it is why the tests for it run in 0.1s.

WHERE THE MODEL COMES FROM. The NODE-JOB, and nowhere else. ``ensure_ready(model_id)`` is
called with the row's ``required_model`` and hands it to ``ensure_up_fn(model_id)``; this
module puts it in the container's ``Cmd``. An earlier draft of this recreated the sidecar
through ``docker compose``, which re-reads the compose FILE and therefore ignored the argv
computed here — so the model had to be smuggled through an environment variable the file
would have to interpolate. That made a model switch depend on a file the engine does not
own, for no reason: the container's own config is right there, and only ``Cmd`` needs to
change.

WHAT IT DELIBERATELY DOES NOT DO. It rewrites the ``--model`` flag and nothing else. The
sidecar's tuning — ``--gpu-memory-utilization``, ``--reasoning-parser``,
``--served-model-name`` — is the operator's, and a bring-up that rewrote it would silently
change how the box serves while appearing to only switch models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

_MODEL_FLAG = "--model"


@dataclass(frozen=True)
class SidecarSpec:
    """What the sidecar container currently is."""

    name: str
    running: bool
    argv: list[str]
    image: str


def current_model(argv: list[str]) -> str | None:
    """The model this argv selects, in either CLI form."""
    for i, tok in enumerate(argv or []):
        if tok == _MODEL_FLAG and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(f"{_MODEL_FLAG}="):
            return tok.split("=", 1)[1]
    return None


def desired_argv(argv: list[str], model_id: str) -> list[str]:
    """``argv`` with ``--model`` set to ``model_id``; everything else untouched.

    Both CLI forms are handled because a hand-written compose uses whichever it likes, and
    guessing wrong would APPEND a second ``--model`` — vLLM would then take one of them and
    the served id would disagree with the request, which the backend correctly treats as a
    fatal mismatch.
    """
    out = list(argv or [])
    for i, tok in enumerate(out):
        if tok == _MODEL_FLAG and i + 1 < len(out):
            out[i + 1] = model_id
            return out
        if tok.startswith(f"{_MODEL_FLAG}="):
            out[i] = f"{_MODEL_FLAG}={model_id}"
            return out
    return out + [_MODEL_FLAG, model_id]


def recreate_payload(inspected: dict, argv: list[str]) -> dict:
    """A ``POST /containers/create`` body: this container, with a new ``Cmd``.

    Built from the container's OWN inspect output so image, env, ports, binds, devices and
    restart policy all survive verbatim. A recreate that rebuilt those from assumptions
    would turn a model switch into a silent redeploy — losing a GPU device request or a
    bind mount is the kind of thing that shows up as "the model is slow" three hours later.
    """
    config = dict(inspected.get("Config") or {})
    host = dict(inspected.get("HostConfig") or {})
    payload = {
        "Image": config.get("Image"),
        "Cmd": list(argv),
        "Env": list(config.get("Env") or []),
        "Labels": dict(config.get("Labels") or {}),
        "ExposedPorts": dict(config.get("ExposedPorts") or {}),
        "Entrypoint": config.get("Entrypoint"),
        "WorkingDir": config.get("WorkingDir") or "",
        "HostConfig": host,
    }
    # Drop keys docker rejects as null on create rather than sending them empty.
    return {k: v for k, v in payload.items() if v not in (None, {}, "")} | {"Cmd": list(argv)}


def ensure_sidecar(
    model_id: str,
    *,
    inspect_fn: Callable[[], SidecarSpec | None],
    recreate_fn: Callable[[SidecarSpec, list[str]], None],
) -> bool:
    """Make the sidecar be up and serving ``model_id``. Returns True iff it acted.

    Three cases, and the middle one is the whole point of the "already correct" check: a
    needless recreate costs minutes of weight loading, which is exactly the cost the
    ``vllm_min_queue`` amortisation gate exists to avoid paying casually.

      * running AND already serving ``model_id``   ⇒ leave it alone
      * running on a DIFFERENT model               ⇒ recreate with the new argv
      * not running                                ⇒ start it (even if the baked model
                                                     matches — a stopped server serves
                                                     nothing)

    No sidecar at all RAISES rather than returning quietly: silence would leave
    ``ensure_ready`` polling health for its full timeout and then blaming the model. The
    caller can soft-degrade to ollama, but only if it is told.
    """
    spec = inspect_fn()
    if spec is None:
        raise RuntimeError(
            "no vLLM sidecar found to bring up — cannot serve "
            f"{model_id!r}; wire EngineConfig.vllm_start_fn or deploy the sidecar")

    if spec.running and current_model(spec.argv) == model_id:
        log.debug("[vllm-sidecar] already serving %s", model_id)
        return False

    argv = desired_argv(spec.argv, model_id)
    log.info("[vllm-sidecar] %s → serving %s (was %s)",
             "starting" if not spec.running else "recreating",
             model_id, current_model(spec.argv))
    recreate_fn(spec, argv)
    return True
