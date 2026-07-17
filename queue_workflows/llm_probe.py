"""ASK the box's LLM server what it is, instead of assuming.

WHY THIS EXISTS. ``worker_heartbeats.llm_servers_available`` (migration 0014) is
*documented* as OBSERVED capability — "which LLM servers can THIS machine actually
run" — but nothing observed anything. The DB column DEFAULTs to ``{ollama}`` and the
worker published :attr:`~queue_workflows.config.EngineConfig.llm_servers_available`,
whose default is also ``["ollama"]``. Both halves guessed, and they guessed the same
way, so the guess looked like a fact.

The failure that exposed it: a GPU box that ran **no LLM server at all** still
advertised ``{ollama}``; the operator panel drew it an "OLLAMA · ON" chip; and its
per-box topology entry quietly pointed at ANOTHER machine's server, so every LLM call
crossed the network. Its own GPU sat at 0% while it held a GPU claim slot for hours,
and no signal in the system contradicted the picture — because nothing ever asked.

Two honest answers replace the guess:

* :func:`probe_llm_servers` — GET the endpoint and report what ANSWERS. Nothing there
  ⇒ ``[]``, not ``["ollama"]``. An empty list is the whole point: it makes "this box
  serves no model" a visible state instead of an invisible one.
* :func:`is_local_endpoint` — does this box dial ITSELF, or somebody else's GPU? A
  worker that borrows a peer's server is a legitimate deployment (a client box), but
  it must be *chosen*, not stumbled into, so the caller can say so out loud.

DESIGN. No new dependency: psycopg stays the engine's only hard runtime dep, so the
default fetch is ``urllib`` from the stdlib (never httpx/requests). HTTP is an injected
seam (``get_fn``) so tests run with a virtual network and no server. Nothing here
raises: a probe is diagnostic, and a diagnostic that can crash a heartbeat is worse
than no diagnostic at all.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

#: The serving-path KINDS the engine arbitrates on a box. ``ollama``/``vllm`` are LLM
#: servers (mirrors ``worker_control.SERVER_TYPE_*``); ``comfyui`` is a diffusion
#: server; ``inprocess`` is a model loaded in the worker's own process (the native
#: ``ModelCache`` path, e.g. sdxl). One box may hold at most one kind at a time.
OLLAMA = "ollama"
VLLM = "vllm"
COMFYUI = "comfyui"
INPROCESS = "inprocess"

#: Torch-allocator VRAM (bytes) above which a ComfyUI is judged to HOLD a model. A bare
#: ComfyUI keeps only its CUDA context (hundreds of MB); a loaded diffusion model is
#: several GB — so a 1 GiB floor cleanly separates "serving a model" from "just up".
COMFYUI_MODEL_VRAM_FLOOR = 1 << 30

#: Default probe timeout. Deliberately short — this runs on the heartbeat path, and a
#: slow/dead LLM server must never stall the worker's liveness signal. A server that
#: cannot answer a trivial GET in this long is not one we want to advertise anyway.
DEFAULT_TIMEOUT_S = 2.0

#: Hostnames that mean "the machine this worker runs on". ``host.docker.internal`` is
#: the docker host gateway — from inside a container that IS this box, which is exactly
#: how a GPU worker reaches an ollama running on its own host.
_LOCAL_HOSTS = frozenset({
    "127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0", "host.docker.internal",
})


def _default_get(url: str, timeout_s: float) -> int | None:
    """GET ``url``, return its HTTP status, or ``None`` when it cannot be reached.

    stdlib only (see module docstring). A 4xx/5xx is still an ANSWER — something is
    listening and speaking HTTP — so it comes back as a status, not ``None``; only a
    connection failure / timeout / DNS miss is ``None``."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except Exception:
        return None


def _default_get_json(url: str, timeout_s: float):
    """GET ``url`` and parse JSON. stdlib only. Raises on any failure — the caller
    (``probe_gpu_placement``) treats any exception as ``"unknown"``."""
    import json
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def probe_llm_servers(
    base_url: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    get_fn=None,
) -> list[str]:
    """The LLM server types that actually ANSWER at ``base_url``.

    Returns ``["ollama"]``, ``["vllm"]``, or ``[]`` — and ``[]`` is a real, meaningful
    answer meaning *this box serves no model*, which is what the old static default
    could never say.

    ``/api/tags`` is checked FIRST because it is ollama-specific, while ollama *also*
    serves the OpenAI-compatible ``/v1/models``: probe them the other way round and
    every ollama box gets mislabelled a vllm box.

    Never raises — a broken probe degrades to ``[]`` (see module docstring)."""
    if not base_url:
        return []
    get = get_fn or _default_get
    root = str(base_url).rstrip("/")
    try:
        if get(f"{root}/api/tags", timeout_s) == 200:
            return [OLLAMA]
        if get(f"{root}/v1/models", timeout_s) == 200:
            return [VLLM]
    except Exception:
        log.debug("[llm-probe] %s did not answer — advertising no server", root)
        return []
    return []


#: Placement verdicts from :func:`probe_gpu_placement`.
GPU = "gpu"
CPU = "cpu"
UNKNOWN = "unknown"


def probe_gpu_placement(
    base_url: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    get_json_fn=None,
) -> str:
    """Is the ollama at ``base_url`` serving its loaded model FULLY on the GPU?

    * ``"gpu"``  — every loaded model has ``size_vram >= size`` (all weights in VRAM).
    * ``"cpu"``  — some loaded model has ``size_vram < size``: a full CPU fallback
      (``size_vram == 0``, the NVML-loss case) OR a partial offload (too big for VRAM).
      Both violate "only GPU", and both resolve to *skip this box*.
    * ``"unknown"`` — nothing loaded yet (cold), the endpoint is unreachable, or the
      payload is unparseable. The box is NOT skipped on unknown: a cold server will
      load on GPU if it can, and the *next* probe catches a fallback once a model is up.

    ollama-specific (reads ``/api/ps``). vLLM is CUDA-only, so a live vLLM is GPU by
    construction — callers pass ``"gpu"`` for it without probing. Never raises."""
    if not base_url:
        return UNKNOWN
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/api/ps", timeout_s)
    except Exception:
        return UNKNOWN
    if not isinstance(data, dict):
        return UNKNOWN
    models = data.get("models") or []
    if not models:
        return UNKNOWN
    for m in models:
        try:
            size = int(m.get("size") or 0)
            vram = int(m.get("size_vram") or 0)
        except (TypeError, ValueError):
            continue
        if size > 0 and vram < size:
            return CPU
    return GPU


def loaded_models(
    base_url: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    get_json_fn=None,
) -> list[str]:
    """The model ids ollama currently holds resident at ``base_url`` (``/api/ps``),
    or ``[]`` when nothing is loaded / the endpoint is unreachable / not ollama.
    Never raises — the idle reaper treats any failure as "nothing to unload"."""
    if not base_url:
        return []
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/api/ps", timeout_s)
        models = data.get("models") if isinstance(data, dict) else None
        return [str(m["name"]) for m in (models or []) if isinstance(m, dict) and m.get("name")]
    except Exception:
        return []


def _parse_rfc3339(ts: str) -> float:
    """Best-effort RFC3339 → epoch seconds (0.0 on any failure). ollama emits
    nanosecond fractions which ``fromisoformat`` rejects, so the fraction is
    trimmed to microseconds first."""
    import re
    from datetime import datetime

    try:
        ts = re.sub(r"\.(\d{6})\d+", r".\1", str(ts)).replace("Z", "+00:00")
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


def loaded_models_info(
    base_url: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    get_json_fn=None,
) -> list[dict]:
    """Like :func:`loaded_models` but with a recency key per model:
    ``[{"name": ..., "mru": <epoch-ish float>}]``. ``expires_at`` (last use +
    keep_alive) is the recency proxy — a later expiry means a more recent use — so
    the residency enforcer can keep the most-recently-used model deterministically.
    Never raises; ``[]`` on any failure."""
    if not base_url:
        return []
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/api/ps", timeout_s)
        models = data.get("models") if isinstance(data, dict) else None
        return [
            {"name": str(m["name"]), "mru": _parse_rfc3339(m.get("expires_at", ""))}
            for m in (models or []) if isinstance(m, dict) and m.get("name")
        ]
    except Exception:
        return []


def vllm_served_models(
    base_url: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    get_json_fn=None,
) -> list[str]:
    """The model id(s) a vLLM at ``base_url`` is serving (OpenAI ``/v1/models``).
    A vLLM pins its model for the server's lifetime, so a non-empty answer means
    that model is resident in VRAM right now. Never raises; ``[]`` on any failure
    (including "that URL is not a vLLM")."""
    if not base_url:
        return []
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/v1/models", timeout_s)
        rows = data.get("data") if isinstance(data, dict) else None
        return [str(m["id"]) for m in (rows or []) if isinstance(m, dict) and m.get("id")]
    except Exception:
        return []


def _default_post_json(url: str, payload, timeout_s: float) -> int:
    """POST ``payload`` as JSON, return the HTTP status. stdlib only."""
    import json
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        return int(resp.status)


def unload_ollama_models(
    base_url: str | None,
    models,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    post_fn=None,
) -> list[str]:
    """Unload each of ``models`` from the ollama at ``base_url`` — a ``keep_alive: 0``
    request drops it from VRAM immediately (the documented ollama unload). Returns the
    ids actually asked to unload; per-model best-effort, so one failure never blocks
    the rest and the function never raises."""
    if not base_url:
        return []
    post = post_fn or _default_post_json
    root = str(base_url).rstrip("/")
    done: list[str] = []
    for m in models:
        try:
            post(f"{root}/api/generate", {"model": m, "keep_alive": 0}, timeout_s)
            done.append(m)
        except Exception:
            log.debug("[llm-probe] unload of %s at %s failed (ignored)", m, root)
    return done


def comfyui_loaded(
    base_url: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    get_json_fn=None,
    vram_floor: int = COMFYUI_MODEL_VRAM_FLOOR,
) -> bool:
    """Is the ComfyUI at ``base_url`` currently holding a model in VRAM?

    ComfyUI exposes no "current checkpoint" endpoint, but ``/system_stats`` reports the
    torch allocator per device (``torch_vram_total`` − ``torch_vram_free`` = what
    ComfyUI itself holds). A loaded diffusion model is GBs; a bare, model-less ComfyUI
    is only its CUDA context. So "torch VRAM in use ≥ ``vram_floor``" is the residency
    signal — it counts ComfyUI's OWN allocation, not co-tenants', so a neighbouring
    ollama does not make it read loaded. ``True`` only when ComfyUI answers AND holds a
    model. Never raises (``False`` on any failure / unreachable / not-ComfyUI)."""
    if not base_url:
        return False
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/system_stats", timeout_s)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    used = 0
    for dev in data.get("devices") or []:
        if not isinstance(dev, dict):
            continue
        try:
            total = int(dev.get("torch_vram_total") or 0)
            free = int(dev.get("torch_vram_free") or 0)
        except (TypeError, ValueError):
            continue
        if total > 0 and free >= 0:
            used += max(0, total - free)
    return used >= int(vram_floor)


def comfyui_free(
    base_url: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    post_fn=None,
) -> bool:
    """Evict lever for ComfyUI: POST ``/free`` with ``unload_models`` + ``free_memory``
    so ComfyUI drops its loaded models and releases VRAM — the ComfyUI equivalent of
    ollama's ``keep_alive: 0``. ``True`` iff the server accepted it (2xx). Never raises
    (``False`` on any failure), so a dead ComfyUI can't wedge the caller."""
    if not base_url:
        return False
    post = post_fn or _default_post_json
    try:
        status = post(
            f"{str(base_url).rstrip('/')}/free",
            {"unload_models": True, "free_memory": True},
            timeout_s,
        )
        return 200 <= int(status) < 300
    except Exception:
        log.debug("[llm-probe] comfyui /free at %s failed (ignored)", base_url)
        return False


def gpu_usable(servers: list[str], placement: str) -> bool:
    """The one predicate the GPU claim gate reads: may this box take GPU LLM work?

    ``True`` iff it has a live server AND that server is not running a model on CPU.
    ``"unknown"`` placement (cold server, or vLLM) is usable — optimistic, because a
    healthy box loads on GPU and a real fallback is caught on the next probe. ``"cpu"``
    (fault or over-capacity) and *no server at all* are the two not-usable states —
    exactly the "skip the box" cases the policy allows (an OFF toggle is handled
    separately by the worker-control park path)."""
    return bool(servers) and placement != CPU


def is_local_endpoint(base_url: str | None) -> bool:
    """``True`` when ``base_url`` points at the machine this worker runs on.

    False for a peer's address — which means this box dispatches its LLM work to
    ANOTHER machine's GPU while its own stays idle. That is a valid client-box
    deployment, but the caller should say so out loud (see
    :func:`describe_endpoint`) rather than let it look like local GPU work."""
    if not base_url:
        return False
    try:
        host = urlsplit(str(base_url)).hostname
    except Exception:
        return False
    return bool(host) and host.lower() in _LOCAL_HOSTS


def describe_endpoint(host_label: str, base_url: str | None, servers: list[str]) -> str:
    """A one-line, operator-readable verdict on this box's LLM wiring — the sentence
    that would have made the original incident obvious on day one.

    The caller logs it at the severity implied by the verdict (see
    :func:`endpoint_is_healthy`)."""
    if not base_url:
        return (
            f"[llm] {host_label}: NO LLM endpoint resolved — this worker cannot run "
            f"LLM/VLM nodes. Set a topology entry (llm_topology_path) or the URL env."
        )
    if not servers:
        return (
            f"[llm] {host_label}: NO LLM SERVER ANSWERS at {base_url} — advertising "
            f"no server. Any LLM/VLM node claimed here will fail or hang. Start a "
            f"server on this box, or point its topology entry at one that answers."
        )
    where = "its OWN" if is_local_endpoint(base_url) else "a REMOTE box's"
    return f"[llm] {host_label}: serving {servers[0]} from {where} server at {base_url}"


def endpoint_is_healthy(base_url: str | None, servers: list[str]) -> bool:
    """``True`` only when an endpoint resolved AND something actually answered on it.

    A GPU worker for which this is ``False`` is a worker that will claim GPU jobs it
    cannot execute — the caller should log :func:`describe_endpoint` as an ERROR."""
    return bool(base_url) and bool(servers)


__all__ = [
    "OLLAMA",
    "VLLM",
    "COMFYUI",
    "INPROCESS",
    "COMFYUI_MODEL_VRAM_FLOOR",
    "GPU",
    "CPU",
    "UNKNOWN",
    "DEFAULT_TIMEOUT_S",
    "probe_llm_servers",
    "probe_gpu_placement",
    "gpu_usable",
    "loaded_models",
    "loaded_models_info",
    "vllm_served_models",
    "unload_ollama_models",
    "comfyui_loaded",
    "comfyui_free",
    "is_local_endpoint",
    "describe_endpoint",
    "endpoint_is_healthy",
]
