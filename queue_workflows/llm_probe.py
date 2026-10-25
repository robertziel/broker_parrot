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
import socket
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

#: ...and THE SAME GATEWAY IN ITS NUMERIC SPELLING, which the static set above cannot
#: name: it differs per box and per network (172.17.0.1 on the default bridge, 172.18.0.1
#: and up on a compose network). A topology entry or an OLLAMA_URL carries the number far
#: more often than the ``host.docker.internal`` alias, and then a box's OWN ollama reads as
#: somebody else's. Observed in production: a lane HOLDING its card for a multi-GPU cluster
#: narrated "serving ollama from a REMOTE box's server at http://172.17.0.1:11434" every
#: 10 s — about the very card the cluster was waiting for. Locality answers "is this MY
#: card's server?", and a card is not identified by a hostname spelling.
#:
#: Resolved once and cached: a container's routing table is a boot-time property, and this
#: is read on probe paths.
_gateway_cache: frozenset[str] | None = None


def _read_default_gateway() -> str | None:
    """This container's default-route gateway as a dotted quad — the docker host — or
    ``None`` where there is none to read (``/proc/net/route`` is Linux-only, and a
    host-network process may have no default route). Reads the same table
    :func:`queue_workflows.hw_lan._default_interface` does, one column over."""
    try:
        with open("/proc/net/route") as fh:
            for line in fh.readlines()[1:]:
                parts = line.split()
                if len(parts) > 2 and parts[1] == "00000000" and parts[2] != "00000000":
                    return socket.inet_ntoa(int(parts[2], 16).to_bytes(4, "little"))
    except (OSError, ValueError):
        pass
    return None


def _host_gateways() -> frozenset[str]:
    """The cached gateway set. Never raises; empty where nothing could be read — which is
    exactly the behaviour from before gateways were consulted at all."""
    global _gateway_cache
    if _gateway_cache is None:
        try:
            gateway = _read_default_gateway()
        except Exception:      # pragma: no cover - a probe path must never raise here
            gateway = None
        _gateway_cache = frozenset({gateway} if gateway else ())
    return _gateway_cache


def reset_locality_cache_for_tests() -> None:
    """TEST-ONLY: drop the cached gateway so a test can inject its own."""
    global _gateway_cache
    _gateway_cache = None


#: The only URL schemes a probe will fetch. Base URLs come from operator config —
#: the topology file, env, and the DB (``worker_controls``, 0013) — so any process
#: that can write those must still not be able to point a probe at ``file://`` (a
#: local file read into probe results) or another non-HTTP scheme.
_ALLOWED_URL_SCHEMES = ("http", "https")


def _refuse_non_http(url: str) -> bool:
    """True (and a WARNING log) when ``url`` is not plain http(s) — callers treat
    it as unreachable / raise, per their own contract."""
    import urllib.parse

    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme in _ALLOWED_URL_SCHEMES:
        return False
    log.warning("[llm-probe] refusing non-HTTP url scheme %r (%s)", scheme, url)
    return True


def _default_get(url: str, timeout_s: float) -> int | None:
    """GET ``url``, return its HTTP status, or ``None`` when it cannot be reached.

    stdlib only (see module docstring). A 4xx/5xx is still an ANSWER — something is
    listening and speaking HTTP — so it comes back as a status, not ``None``; only a
    connection failure / timeout / DNS miss — or a refused non-HTTP scheme — is
    ``None``."""
    import urllib.error
    import urllib.request

    if _refuse_non_http(url):
        return None
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

    if _refuse_non_http(url):
        raise ValueError(f"refusing non-HTTP url scheme: {url!r}")
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


def vllm_activity(
    base_url: str | None, *, timeout_s: float = DEFAULT_TIMEOUT_S, get_text_fn=None,
) -> dict:
    """vLLM's SERVER-SIDE use signals, from its Prometheus endpoint.

    Returns ``{"served": bool, "running": int, "finished": int}``:

      * ``running``  — ``vllm:num_requests_running``, in flight RIGHT NOW;
      * ``finished`` — ``vllm:request_success_total``, monotonic completions.

    Together they answer "has anyone used this server since I last looked?" for ANY
    client, which is the property an in-process request counter cannot have (the
    node POSTs the URL directly, as do host-internal callers and other projects on
    the box). This is vLLM's analog of ollama's ``/api/ps`` ``expires_at``; see
    :mod:`queue_workflows.llm_idle_unload`.

    Metric names are matched by PREFIX because vLLM labels them per model
    (``vllm:num_requests_running{model_name="..."}``) and a box may serve several —
    the values are summed. Never raises: an unreachable or unparseable endpoint
    returns ``served=False``, which the idle stopper treats as "do not act"."""
    out = {"served": False, "running": 0, "finished": 0}
    if not base_url:
        return out
    root = str(base_url).rstrip("/")
    try:
        text = (get_text_fn or _default_get_text)(f"{root}/metrics", timeout_s)
    except Exception:
        log.debug("[llm-probe] %s/metrics did not answer", root)
        return out
    if not text:
        return out
    running = 0.0
    finished = 0.0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        if line.startswith("vllm:num_requests_running"):
            running += value
        elif line.startswith("vllm:request_success_total"):
            finished += value
    out["served"] = True
    out["running"] = int(running)
    out["finished"] = int(finished)
    return out


def _default_get_text(url: str, timeout_s: float) -> str:
    """GET a URL and return its body, or "" on any failure. stdlib-only (urllib) —
    :mod:`llm_probe` must never grow an HTTP dependency."""
    import urllib.request

    if _refuse_non_http(url):
        return ""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            if getattr(resp, "status", 200) != 200:
                return ""
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return ""


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


def ollama_available_models(
    base_url: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    get_json_fn=None,
) -> list[str]:
    """Every model INSTALLED on an ollama at ``base_url`` (``/api/tags``).

    Deliberately a different question from :func:`vllm_served_models`, and both answers
    are right: ollama swaps models on demand, so everything installed is servable; a vLLM
    pins ONE model for its process lifetime, so only that one is. Conflating them would
    advertise capability a vLLM cannot honour without a restart, and the claim gate would
    then send it work it must refuse.

    Never raises; ``[]`` on any failure — absence is a real state (see the module
    docstring), not a reason to guess."""
    if not base_url:
        return []
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/api/tags", timeout_s)
        rows = data.get("models") if isinstance(data, dict) else None
        return [str(m["name"]) for m in (rows or [])
                if isinstance(m, dict) and m.get("name")]
    except Exception:
        return []


def ollama_available_models_with_digests(
    base_url: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    get_json_fn=None,
) -> list[tuple[str, str]]:
    """``(name, digest)`` for every model an ollama at ``base_url`` holds.

    The digest is what ties a stable ROLE tag to the immutable ARTIFACT tag behind it:
    ollama reports one content hash per tag, and two tags with one hash are one set of
    bytes. Kept as a SECOND function rather than widening the return of
    :func:`ollama_available_models` because that one's ``list[str]`` shape is consumed by
    the observed-narrowing path in several projects — changing it is a flag-day; adding
    beside it is not.

    Same failure contract: never raises, ``[]`` on anything — absence is a real state.
    """
    if not base_url:
        return []
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/api/tags", timeout_s)
        rows = data.get("models") if isinstance(data, dict) else None
        return [(str(m["name"]), str(m.get("digest") or ""))
                for m in (rows or []) if isinstance(m, dict) and m.get("name")]
    except Exception:
        return []


def observe_box_models(
    host_label: str,
    *,
    # ── WHERE THIS MUST RUN ──────────────────────────────────────────────────────
    # In the CLAIM WORKER, not the orchestrator. Observed in a real deployment:
    # the claim container resolves `host.docker.internal` and saw 4 models; the
    # orchestrator container on the SAME box got
    #     URLError [Errno -2] Name or service not known
    # because it is not launched with extra_hosts host-gateway. Publishing from there
    # would write an EMPTY catalog for a box that is serving perfectly well — and since
    # `best_match` treats unknown as cannot-serve (deliberately, the conservative
    # direction), an empty catalog does not degrade gracefully: it stops every LLM claim
    # on that box. The conservative default is only safe if the observer is the process
    # that can actually see the server.
    urls_by_kind: dict,
    ollama_fn=None,
    vllm_fn=None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list:
    """What THIS box's servers can actually serve, as catalog entries.

    One entry per (kind, model). A kind with no URL contributes nothing rather than an
    empty claim, and a box whose servers all answer nothing publishes NOTHING — which is
    the visible-absence property this module exists to preserve."""
    from queue_workflows.model_catalog import CatalogEntry

    # ollama contributes (name, digest) pairs — the digest is what lets the catalog tie a
    # role tag to the artifact behind it (model_catalog.resolve_artifact). vLLM reports no
    # digest, so its entries carry "" and resolution falls back to the served id itself
    # being an artifact tag. Injected fns may return either shape; normalise here.
    ask = {"ollama": ollama_fn or ollama_available_models_with_digests,
           "vllm": vllm_fn or vllm_served_models}
    out: list = []
    for kind, url in (urls_by_kind or {}).items():
        if not url or kind not in ask:
            continue
        for item in ask[kind](url, timeout_s=timeout_s):
            if isinstance(item, tuple):
                model_id, digest = item
            else:
                model_id, digest = item, ""
            entry = CatalogEntry(host_label, kind, str(model_id), digest=str(digest))
            if entry not in out:
                out.append(entry)
    return out


# ── context-window observations (migration 0028) ─────────────────────────────
#
# What each server says about the window it serves and what a bigger one would cost.
# The arithmetic lives in :mod:`queue_workflows.llm_context`; these only observe, and an
# unknown stays unknown — the arithmetic then refuses to promise a bigger window.


def _int_or_none(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def ollama_runners(
    base_url: str | None, *, timeout_s: float = DEFAULT_TIMEOUT_S, get_json_fn=None,
) -> list[dict]:
    """Every runner ollama holds right now (``/api/ps``), with what it occupies and the
    window it was loaded at: ``[{"name", "size", "size_vram", "context_length"}]``.

    ``context_length`` is the runner's ACTUAL window — ollama reports the num_ctx it
    loaded with (clamped to the model's maximum), which is exactly what a prompt has to
    fit. ``[]`` when nothing is loaded or the endpoint does not answer. Never raises."""
    if not base_url:
        return []
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/api/ps", timeout_s)
        rows = data.get("models") if isinstance(data, dict) else None
        return [
            {"name": str(m["name"]), "size": _int_or_none(m.get("size")),
             "size_vram": _int_or_none(m.get("size_vram")),
             "context_length": _int_or_none(m.get("context_length"))}
            for m in (rows or []) if isinstance(m, dict) and m.get("name")
        ]
    except Exception:
        return []


def ollama_model_sizes(
    base_url: str | None, *, timeout_s: float = DEFAULT_TIMEOUT_S, get_json_fn=None,
) -> dict[str, int]:
    """``{tag: bytes}`` for every model installed on the ollama at ``base_url``
    (``/api/tags`` ``size``) — the weights a runner of that model has to hold before any
    cache. ``{}`` on any failure. Never raises."""
    if not base_url:
        return {}
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/api/tags", timeout_s)
        rows = data.get("models") if isinstance(data, dict) else None
        out: dict[str, int] = {}
        for m in rows or []:
            if isinstance(m, dict) and m.get("name"):
                size = _int_or_none(m.get("size"))
                if size:
                    out[str(m["name"])] = size
        return out
    except Exception:
        return {}


def _default_post_json_body(url: str, payload, timeout_s: float):
    """POST ``payload`` as JSON and parse the JSON reply. stdlib only; raises on any
    failure (callers here swallow it)."""
    import json
    import urllib.request

    if _refuse_non_http(url):
        raise ValueError(f"refusing non-HTTP url scheme: {url!r}")
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def ollama_model_info(
    base_url: str | None, model: str, *, timeout_s: float = DEFAULT_TIMEOUT_S,
    post_json_fn=None,
) -> dict:
    """``model_info`` for one installed model (``POST /api/show``) — its architecture,
    layer and head counts, and its own maximum window. Reading it loads nothing. ``{}``
    on any failure. Never raises."""
    if not base_url or not model:
        return {}
    post = post_json_fn or _default_post_json_body
    try:
        data = post(f"{str(base_url).rstrip('/')}/api/show", {"model": model}, timeout_s)
        info = data.get("model_info") if isinstance(data, dict) else None
        return dict(info) if isinstance(info, dict) else {}
    except Exception:
        return {}


def vllm_max_model_len(
    base_url: str | None, *, timeout_s: float = DEFAULT_TIMEOUT_S, get_json_fn=None,
) -> int | None:
    """The window the vLLM at ``base_url`` was started with — its model card's
    ``max_model_len`` (``/v1/models``). A request longer than it is refused by the
    server, so this is the window the claim has to respect. ``None`` when unknown."""
    if not base_url:
        return None
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/v1/models", timeout_s)
        rows = data.get("data") if isinstance(data, dict) else None
        for m in rows or []:
            if isinstance(m, dict):
                value = _int_or_none(m.get("max_model_len"))
                if value:
                    return value
    except Exception:
        return None
    return None


def vllm_kv_capacity_tokens(
    base_url: str | None, *, timeout_s: float = DEFAULT_TIMEOUT_S, get_text_fn=None,
) -> int | None:
    """How many tokens the vLLM's KV pool holds — ``num_gpu_blocks × block_size`` from
    its ``vllm:cache_config_info`` gauge (``/metrics``). The pool is sized by
    ``gpu_memory_utilization``, not by ``max_model_len``, so this bounds how big a window
    the server could restart at without claiming any new memory. ``None`` when unknown."""
    if not base_url:
        return None
    import re

    try:
        text = (get_text_fn or _default_get_text)(
            f"{str(base_url).rstrip('/')}/metrics", timeout_s)
    except Exception:
        return None
    for line in (text or "").splitlines():
        if not line.startswith("vllm:cache_config_info{"):
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
        blocks = _int_or_none(labels.get("num_gpu_blocks"))
        size = _int_or_none(labels.get("block_size"))
        if blocks and size:
            return blocks * size
    return None


def _default_post_json(url: str, payload, timeout_s: float) -> int:
    """POST ``payload`` as JSON, return the HTTP status. stdlib only."""
    import json
    import urllib.request

    if _refuse_non_http(url):
        raise ValueError(f"refusing non-HTTP url scheme: {url!r}")
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


def comfyui_reachable(
    base_url: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    get_json_fn=None,
) -> bool:
    """Is the ComfyUI at ``base_url`` UP and answering (regardless of a loaded model)?

    Distinct from :func:`comfyui_loaded` (which asks "is it holding a model"): right
    after a fresh start ComfyUI holds no model but IS ready to accept a prompt. Readiness
    = ``/system_stats`` returns a JSON object. Used as the ``acquire_box_for_comfyui``
    ready gate. Never raises (``False`` on any failure / unreachable)."""
    if not base_url:
        return False
    get_json = get_json_fn or _default_get_json
    try:
        data = get_json(f"{str(base_url).rstrip('/')}/system_stats", timeout_s)
    except Exception:
        return False
    return isinstance(data, dict) and bool(data)


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

    THE SCOPE IS THE CARD, not a URL: loopback, the docker host gateway by name, and
    that same gateway by NUMBER (:func:`_host_gateways`) all reach the server on this
    box's own card, so all three are local.

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
    return bool(host) and (host.lower() in _LOCAL_HOSTS or host in _host_gateways())


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
    "vllm_activity",
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
