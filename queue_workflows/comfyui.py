"""ComfyUI as a FIRST-CLASS GPU box tenant, plus a reusable submit client.

Elsewhere in the engine ComfyUI is evict-only: :mod:`queue_workflows.llm_probe`
can `comfyui_loaded` (probe) and `comfyui_free` (evict), and the residency arbiter
treats a `comfyui` resident as a KIND that always LOSES the one-server-per-box rule.
This module adds the winning half — when a node_job REQUESTS ComfyUI, ComfyUI takes
the box — and the client used to actually render through it. Both halves live here so
the broker (box side) and any client (render node) share one implementation.

Box side:
    :func:`acquire_box_for_comfyui` — evict every rival serving kind off the card
    (ollama `keep_alive:0`, vLLM stop, in-process unload) via
    :func:`queue_workflows.model_residency.clear_box_for`, then START the box's
    ComfyUI (host-wired lifecycle), then wait until it answers. The inverse of the
    evict-only loser: the previous model is unloaded from the other servers the
    moment ComfyUI is requested.

Client side:
    :func:`submit_workflow` — POST a ComfyUI prompt graph to `/prompt`, poll
    `/history/<id>`, return the output files. What a render node calls.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from queue_workflows import llm_probe
from queue_workflows.model_residency import Resident, clear_box_for

log = logging.getLogger(__name__)

COMFYUI = llm_probe.COMFYUI

_DEFAULT_HTTP_TIMEOUT_S = 30.0


class ComfyUIError(RuntimeError):
    """ComfyUI rejected the prompt (validation) or the run itself errored."""


class ComfyUITimeout(TimeoutError):
    """A submitted ComfyUI prompt did not complete within the deadline."""


class ComfyUINotReady(RuntimeError):
    """The box's ComfyUI did not come up / answer within the ready deadline."""


# ── BOX side: make ComfyUI win the card ──────────────────────────────────────


def acquire_box_for_comfyui(
    residents: list[Resident],
    *,
    start_fn: Callable[[], None] = lambda: None,
    ready_probe: Callable[[], bool] = lambda: True,
    unload_ollama: Callable[[list[str]], object] = lambda models: None,
    stop_vllm: Callable[[], object] = lambda: None,
    free_comfyui: Callable[[], object] = lambda: None,
    unload_inprocess: Callable[[], object] = lambda: None,
    reprobe: Callable[[], list[Resident]] | None = None,
    ready_timeout_s: float = 120.0,
    poll_s: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    label: str = "?",
) -> list[Resident]:
    """Give the box to ComfyUI: clear rivals, start it, wait until it answers.

    1. :func:`clear_box_for` with ``incoming_kind=comfyui`` evicts every resident of a
       DIFFERENT serving kind (ollama / vLLM / in-process diffusion) via its own lever
       — "unload the previous model from the other servers once ComfyUI is requested."
       If ``reprobe`` is given and a rival survives, it raises
       :class:`~queue_workflows.model_residency.ModelAlreadyLoadedError`.
    2. ``start_fn()`` brings up (or no-ops if already up) the box's ComfyUI. Failures
       propagate — a box that can't start its renderer must fail the acquire.
    3. Poll ``ready_probe`` until True or raise :class:`ComfyUINotReady` after
       ``ready_timeout_s``.

    Returns the residents it evicted (possibly empty). Idempotent: a box already
    holding only ComfyUI evicts nothing but still ensures the server is up."""
    evicted = clear_box_for(
        COMFYUI,
        COMFYUI,
        residents,
        unload_ollama=unload_ollama,
        stop_vllm=stop_vllm,
        free_comfyui=free_comfyui,
        unload_inprocess=unload_inprocess,
        reprobe=reprobe,
        label=label,
    )

    start_fn()  # bring up the box's ComfyUI (fail-loud: the card is ours or nothing)

    if ready_probe():
        return evicted
    max_polls = max(1, int(ready_timeout_s / poll_s)) if poll_s > 0 else 1
    for _ in range(max_polls):
        sleep_fn(poll_s)
        if ready_probe():
            return evicted
    raise ComfyUINotReady(
        f"[{label}] ComfyUI did not become ready within {ready_timeout_s}s after start"
    )


def ensure_comfyui_box(
    *,
    comfyui_url: str | None = None,
    inprocess_model: str | None = None,
    unload_inprocess: Callable[[], object] = lambda: None,
    residents_fn: Callable[[], list[Resident]] | None = None,
    start_fn: Callable[[], None] | None = None,
    ready_probe: Callable[[], bool] | None = None,
    unload_ollama: Callable[[list[str]], object] | None = None,
    stop_vllm: Callable[[], object] | None = None,
    ready_timeout_s: float = 120.0,
    poll_s: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    label: str = "?",
) -> list[Resident]:
    """Acquire the box for ComfyUI, wiring every lever from live config + probes.

    The convenience a render node calls: it supplies only its own ``inprocess_model``
    (the worker's warm model, so it's evicted too) and ``unload_inprocess`` callback;
    the ComfyUI URL, start lever, ollama/vLLM evict levers, resident probe, and ready
    gate all resolve from :mod:`queue_workflows.config` + :mod:`queue_workflows.llm_probe`
    + :func:`queue_workflows.model_residency.probe_box_residents` — the same wiring the
    residency enforcer uses. Any of them can be overridden (tests inject them)."""
    from queue_workflows import llm_probe as _lp
    from queue_workflows import model_residency as _mr
    from queue_workflows.config import get_config
    from queue_workflows.envcompat import env_get

    cfg = get_config()
    url = (comfyui_url or env_get("QUEUE_WORKFLOWS_COMFYUI_URL") or cfg.comfyui_url or "").strip()

    if residents_fn is None:
        def residents_fn():  # noqa: E306
            return _mr.probe_box_residents(inprocess_model=inprocess_model)
    if start_fn is None:
        start_fn = cfg.comfyui_start_fn or (lambda: None)
    if ready_probe is None:
        def ready_probe():  # noqa: E306
            return _lp.comfyui_reachable(url)
    if unload_ollama is None:
        def unload_ollama(models):  # noqa: E306
            from queue_workflows.llm_backends import factory as _f
            return _lp.unload_ollama_models(_f.resolve_base_url("ollama"), models)
    if stop_vllm is None:
        stop_vllm = cfg.vllm_stop_fn or (lambda: None)

    return acquire_box_for_comfyui(
        residents_fn(),
        start_fn=start_fn,
        ready_probe=ready_probe,
        unload_ollama=unload_ollama,
        stop_vllm=stop_vllm,
        free_comfyui=lambda: _lp.comfyui_free(url),
        unload_inprocess=unload_inprocess,
        ready_timeout_s=ready_timeout_s,
        poll_s=poll_s,
        sleep_fn=sleep_fn,
        label=label,
    )


# ── CLIENT side: submit a graph and collect the outputs ──────────────────────


def _collect_outputs(outputs: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten a ComfyUI history ``outputs`` map to a list of file descriptors.

    ComfyUI groups outputs per node under ``images`` / ``gifs`` / ``videos`` — each a
    list of ``{filename, subfolder, type}``. Order-stable by node id then position."""
    files: list[dict[str, str]] = []
    for _node_id, node_out in sorted((outputs or {}).items()):
        for key in ("images", "gifs", "videos"):
            for f in node_out.get(key, []) or []:
                files.append(
                    {
                        "filename": f.get("filename", ""),
                        "subfolder": f.get("subfolder", ""),
                        "type": f.get("type", "output"),
                    }
                )
    return files


def submit_workflow(
    base_url: str,
    graph: dict[str, Any],
    *,
    client_id: str = "broker-comfyui",
    poll_s: float = 2.0,
    timeout_s: float = 1800.0,
    http_timeout_s: float = _DEFAULT_HTTP_TIMEOUT_S,
    submit_fn: Callable[[str, dict, float], dict] | None = None,
    history_fn: Callable[[str, float], dict] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[dict[str, str]]:
    """Render a ComfyUI graph and return its output files.

    POSTs ``{"prompt": graph, "client_id": ...}`` to ``<base_url>/prompt``; a response
    with no ``prompt_id`` is a validation failure (:class:`ComfyUIError`, carrying
    ``node_errors``). Then polls ``<base_url>/history/<id>`` every ``poll_s`` until the
    run completes (returns :func:`_collect_outputs`), errors (:class:`ComfyUIError`),
    or ``timeout_s`` elapses (:class:`ComfyUITimeout`).

    ``submit_fn(url, payload, timeout_s) -> dict`` and ``history_fn(url, timeout_s) ->
    dict`` default to stdlib-``urllib`` implementations; both are injectable so the
    caller can supply a client and tests can drive it without a live server."""
    base = str(base_url).rstrip("/")
    submit = submit_fn or _default_submit
    history = history_fn or _default_history

    resp = submit(f"{base}/prompt", {"prompt": graph, "client_id": client_id}, http_timeout_s)
    pid = (resp or {}).get("prompt_id")
    if not pid:
        raise ComfyUIError(
            f"ComfyUI rejected the prompt at {base}: error={(resp or {}).get('error')} "
            f"node_errors={(resp or {}).get('node_errors')}"
        )

    max_polls = max(1, int(timeout_s / poll_s)) if poll_s > 0 else 1
    for _ in range(max_polls + 1):
        hist = history(f"{base}/history/{pid}", http_timeout_s) or {}
        rec = hist.get(pid)
        if rec:
            status = rec.get("status", {}) or {}
            if status.get("status_str") == "error":
                raise ComfyUIError(f"ComfyUI run {pid} errored: {status}")
            if status.get("completed") or status.get("status_str") == "success":
                return _collect_outputs(rec.get("outputs", {}))
        sleep_fn(poll_s)
    raise ComfyUITimeout(
        f"ComfyUI prompt {pid} did not complete within {timeout_s}s at {base}"
    )


# ── default stdlib-urllib transport (injected in tests) ──────────────────────


def _default_submit(url: str, payload: dict, timeout_s: float) -> dict:
    import urllib.request

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:  # 400 validation carries a JSON body
        try:
            return json.loads(exc.read() or b"{}")
        except Exception:
            return {}


def _default_history(url: str, timeout_s: float) -> dict:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
            return json.loads(resp.read() or b"{}")
    except Exception:
        return {}
