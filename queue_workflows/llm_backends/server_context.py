"""The context window a server is CONFIGURED to serve, and the change that moves it.

WHY A RECREATE. Neither server can be told to change its window while it runs. ollama
reads ``OLLAMA_CONTEXT_LENGTH`` from its environment when the daemon starts and applies it
to every model it loads for an OpenAI-style ``/v1`` request — measured on 0.33.2, a
per-request ``num_ctx`` is accepted only by the native ``/api/chat`` route and is undone
by the very next ``/v1`` call, so the environment is the only lever that holds. vLLM takes
``--max-model-len`` on its command line and fixes it for the process lifetime. So changing
a box's window means replacing the container — the same operation the vLLM model switch
already performs (:mod:`vllm_sidecar`), with a different value swapped.

WHAT THIS MODULE IS. The DECISION, as pure functions of (what the container is, what we
want): which value to set, what to remember, and the exact ``POST /containers/create``
body. No docker, no network — :mod:`queue_workflows.llm_context_lever` does the I/O. Same
idiom as ``vllm_sidecar``: the tests for it need no daemon.

THE REDEPLOY HAZARD. Docker's create body does not inherit the container's network
endpoints. A payload that forgets them brings the ``ollama`` container back on its network
WITHOUT its aliases: every client that dials it by name gets NXDOMAIN while the container
looks perfectly healthy — the box's whole LLM service, lost to a window change. So the
endpoints are carried over explicitly (aliases, IPAM, links), one at create and the rest
connected after it, and everything else — image, entrypoint, env, binds, devices, GPU
requests, ports, restart policy — is copied from the container's own inspect output.

REMEMBERING THE CONFIGURED VALUE. The first override records what was configured in a
container LABEL. Without it, "no queued job needs a bigger window" could only mean "keep
the last override forever": the box would stay at 131072 tokens — and pay for that cache
on every model load — because one job once asked for it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

log = logging.getLogger(__name__)

OLLAMA = "ollama"
VLLM = "vllm"

#: Where ollama reads its default window (applies to every ``/v1`` load).
CONTEXT_ENV = "OLLAMA_CONTEXT_LENGTH"
#: Where vLLM takes its window.
CONTEXT_FLAG = "--max-model-len"
#: Where the value the OPERATOR configured is remembered across overrides. Empty string
#: means "the operator set nothing", which restores by REMOVING the setting.
DEFAULT_LABEL = "queue_workflows.llm_context.default"
#: vLLM's memory pool, as a fraction of "GPU memory" — on a unified-memory box, of the
#: whole machine. The engine may LOWER it to what the box has free at start; the
#: operator's own value is remembered here so a lowered pool can come back up.
UTILIZATION_FLAG = "--gpu-memory-utilization"
UTILIZATION_LABEL = "queue_workflows.llm_context.utilization"
#: vLLM's concurrency, and where the operator's own value is remembered once the engine
#: has lowered it to what the box's cache really holds.
MAX_SEQS_FLAG = "--max-num-seqs"
MAX_SEQS_LABEL = "queue_workflows.llm_context.max_num_seqs"
#: The flags the ENGINE changes between starts; everything else on the command line
#: shapes the model's memory and is part of a profile's identity.
_ENGINE_FLAGS = (CONTEXT_FLAG, UTILIZATION_FLAG, MAX_SEQS_FLAG)

#: Endpoint keys ``POST /containers/create`` (and ``/networks/{id}/connect``) accept.
#: Everything else in an inspected endpoint is runtime state (IPs, ids, gateways) that
#: docker assigns itself — sending it back is at best ignored, at worst a conflict.
_ENDPOINT_KEYS = ("IPAMConfig", "Links", "Aliases", "DriverOpts", "MacAddress")

#: Network modes that own the container's whole network namespace: docker REFUSES a
#: create that both shares one of these and names endpoints.
_EXCLUSIVE_MODES = ("host", "none")


def _int_or_none(value: Any) -> int | None:
    try:
        out = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


# ── reading the container ────────────────────────────────────────────────────


def env_context(env: Iterable[str] | None) -> int | None:
    """The window in an inspected ``Config.Env`` list, or ``None``."""
    for item in env or []:
        name, sep, value = str(item).partition("=")
        if sep and name == CONTEXT_ENV:
            return _int_or_none(value)
    return None


def argv_context(argv: Iterable[str] | None) -> int | None:
    """The window on an inspected ``Config.Cmd``, in either CLI form."""
    tokens = list(argv or [])
    for i, tok in enumerate(tokens):
        if tok == CONTEXT_FLAG and i + 1 < len(tokens):
            return _int_or_none(tokens[i + 1])
        if str(tok).startswith(f"{CONTEXT_FLAG}="):
            return _int_or_none(str(tok).split("=", 1)[1])
    return None


def configured_context(inspected: Mapping[str, Any] | None, kind: str) -> int | None:
    """What window this container is configured for — ``None`` when it says nothing
    (then the server's own default applies, which only the server can report)."""
    config = (inspected or {}).get("Config") or {}
    if kind == VLLM:
        return argv_context(config.get("Cmd"))
    return env_context(config.get("Env"))


def recorded_default(inspected: Mapping[str, Any] | None) -> str | None:
    """The operator's original value as remembered by a previous override (``""`` = they
    set none), or ``None`` when this container was never overridden."""
    labels = ((inspected or {}).get("Config") or {}).get("Labels") or {}
    value = labels.get(DEFAULT_LABEL)
    return None if value is None else str(value)


# ── writing the container ────────────────────────────────────────────────────


def env_with_context(env: Iterable[str] | None, value: str | None) -> list[str]:
    """``env`` with :data:`CONTEXT_ENV` set to ``value`` — or removed when ``None``.
    Every other variable keeps its place: the operator's ``OLLAMA_KEEP_ALIVE`` and
    ``OLLAMA_NUM_PARALLEL`` are load-bearing on this fleet."""
    out = [str(e) for e in (env or [])
           if str(e).partition("=")[0] != CONTEXT_ENV]
    if value is not None:
        out.append(f"{CONTEXT_ENV}={value}")
    return out


def _argv_flag(argv: Iterable[str] | None, flag: str) -> str | None:
    tokens = list(argv or [])
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return str(tokens[i + 1])
        if str(tok).startswith(f"{flag}="):
            return str(tok).split("=", 1)[1]
    return None


def _argv_with_flag(argv: Iterable[str] | None, flag: str, value: str | None) -> list[str]:
    """``argv`` with ``flag`` set to ``value`` in whichever CLI form it already uses — or
    removed when ``None``. Guessing the form wrong would APPEND a second flag, and vLLM
    would then take one of them while we believe the other."""
    tokens = [str(t) for t in (argv or [])]
    out: list[str] = []
    replaced = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == flag and i + 1 < len(tokens):
            if value is not None:
                out.extend([flag, value])
                replaced = True
            i += 2
            continue
        if tok.startswith(f"{flag}="):
            if value is not None:
                out.append(f"{flag}={value}")
                replaced = True
            i += 1
            continue
        out.append(tok)
        i += 1
    if value is not None and not replaced:
        out.extend([flag, value])
    return out


def argv_utilization(argv: Iterable[str] | None) -> float | None:
    """The ``--gpu-memory-utilization`` on an inspected ``Config.Cmd``, or ``None``."""
    raw = _argv_flag(argv, UTILIZATION_FLAG)
    try:
        value = float(raw) if raw is not None else None
    except ValueError:
        return None
    return value if value is not None and 0.0 < value <= 1.0 else None


def argv_with_utilization(argv: Iterable[str] | None, value: str | None) -> list[str]:
    return _argv_with_flag(argv, UTILIZATION_FLAG, value)


def operator_utilization(inspected: Mapping[str, Any] | None) -> float | None:
    """The pool fraction the OPERATOR chose: the remembered original when the engine has
    lowered it, else what the command line says. This is what every start is sized
    FROM — sizing from a lowered flag could only ever go down."""
    labels = ((inspected or {}).get("Config") or {}).get("Labels") or {}
    raw = labels.get(UTILIZATION_LABEL)
    if raw not in (None, ""):
        try:
            return float(raw)
        except ValueError:
            pass
    return argv_utilization(((inspected or {}).get("Config") or {}).get("Cmd"))


def argv_max_seqs(argv: Iterable[str] | None) -> int | None:
    """The ``--max-num-seqs`` on an inspected ``Config.Cmd``, or ``None``."""
    return _int_or_none(_argv_flag(argv, MAX_SEQS_FLAG))


def argv_with_max_seqs(argv: Iterable[str] | None, value: str | None) -> list[str]:
    return _argv_with_flag(argv, MAX_SEQS_FLAG, value)


def operator_max_seqs(inspected: Mapping[str, Any] | None) -> int | None:
    """The concurrency the OPERATOR chose: the remembered original when the engine has
    lowered it, else what the command line says."""
    labels = ((inspected or {}).get("Config") or {}).get("Labels") or {}
    remembered = _int_or_none(labels.get(MAX_SEQS_LABEL))
    if remembered is not None:
        return remembered
    return argv_max_seqs(((inspected or {}).get("Config") or {}).get("Cmd"))


def argv_served_name(argv: Iterable[str] | None) -> str | None:
    """The name vLLM serves the model under — ``--served-model-name``, else the
    ``--model`` path itself, which is what vLLM reports on ``/v1/models`` then. This is
    the catalog's ``model_id`` for the box's vLLM rows."""
    return _argv_flag(argv, "--served-model-name") or _argv_flag(argv, "--model")


def argv_fingerprint(argv: Iterable[str] | None) -> str:
    """The identity of a memory profile: the command line MINUS the flags the engine
    itself manages (window, pool, concurrency). The same model with a different dtype,
    KV-cache dtype or batch size is a different profile."""
    import hashlib

    kept: list[str] = []
    items = list(argv or [])
    i = 0
    while i < len(items):
        item = str(items[i])
        if item in _ENGINE_FLAGS:
            i += 2
            continue
        if any(item.startswith(f"{flag}=") for flag in _ENGINE_FLAGS):
            i += 1
            continue
        kept.append(item)
        i += 1
    return hashlib.sha1(" ".join(kept).encode("utf-8")).hexdigest()[:16]


def argv_with_context(argv: Iterable[str] | None, value: str | None) -> list[str]:
    """``argv`` with :data:`CONTEXT_FLAG` set to ``value`` — or removed when ``None``.

    Both CLI forms are handled for the reason ``vllm_sidecar.desired_argv`` gives about
    ``--model``: guessing wrong APPENDS a second flag, and vLLM would then take one of
    them while we believe the other."""
    return _argv_with_flag(argv, CONTEXT_FLAG, value)


def _endpoint(config: Mapping[str, Any], drop: Iterable[str]) -> dict:
    """One inspected endpoint reduced to what create/connect accept, with the old
    container's own ID stripped from the aliases (docker adds the new container's
    itself; re-sending the old one names the replacement after a container that no
    longer exists).

    The NAME is not stripped: on a user-defined network an alias equal to the container
    name is what some compose files write by hand, and dropping it would be the very
    NXDOMAIN this function exists to prevent."""
    dropped = {d for d in drop if d}
    out: dict = {}
    for key in _ENDPOINT_KEYS:
        value = config.get(key)
        if key == "Aliases" and value:
            value = [a for a in value if a not in dropped]
        if value:
            out[key] = value
    return out


def networking(inspected: Mapping[str, Any] | None, *, name: str | None = None) -> tuple[
        dict | None, list[tuple[str, dict]]]:
    """``(endpoint_for_create, [(network, endpoint), …])`` — the container's network
    identity, split the way the docker API takes it: one endpoint may ride the create,
    the rest must be connected afterwards.

    ``(None, [])`` for a container sharing the host (or another container's) namespace."""
    inspected = inspected or {}
    host = inspected.get("HostConfig") or {}
    mode = str(host.get("NetworkMode") or "")
    if mode in _EXCLUSIVE_MODES or mode.startswith("container:"):
        return None, []
    networks = ((inspected.get("NetworkSettings") or {}).get("Networks") or {})
    if not networks:
        return None, []
    drop = [str(inspected.get("Id") or "")[:12]]
    ordered = list(networks)
    first = mode if mode in networks else ordered[0]
    create = {first: _endpoint(networks.get(first) or {}, drop)}
    extras = [(n, _endpoint(networks.get(n) or {}, drop))
              for n in ordered if n != first]
    return create, extras


def recreate_payload(
    inspected: Mapping[str, Any],
    argv: list[str] | None = None,
    *,
    env: list[str] | None = None,
    labels: Mapping[str, str] | None = None,
    name: str | None = None,
) -> dict:
    """A ``POST /containers/create`` body: this container, with ``argv`` / ``env`` /
    ``labels`` overridden where given and everything else carried over verbatim.

    Built from the container's OWN inspect output so image, entrypoint, env, ports,
    binds, devices, restart policy AND network identity survive — a recreate that
    rebuilt those from assumptions would turn a window (or model) change into a silent
    redeploy, and losing a bind mount or a network alias is the kind of thing that shows
    up as "the model is slow" three hours later.
    """
    config = dict(inspected.get("Config") or {})
    host = dict(inspected.get("HostConfig") or {})
    cmd = list(argv) if argv is not None else list(config.get("Cmd") or [])
    payload = {
        "Image": config.get("Image"),
        "Cmd": cmd,
        "Env": list(env if env is not None else (config.get("Env") or [])),
        "Labels": dict(labels if labels is not None else (config.get("Labels") or {})),
        "ExposedPorts": dict(config.get("ExposedPorts") or {}),
        "Entrypoint": config.get("Entrypoint"),
        "WorkingDir": config.get("WorkingDir") or "",
        "HostConfig": host,
    }
    # Drop keys docker rejects as null on create rather than sending them empty — but
    # never a value the CALLER set: an empty env or label map is a deliberate "this
    # container has none", and silently dropping it would leave the reader unable to
    # tell an override from an omission.
    payload = {k: v for k, v in payload.items() if v not in (None, {}, "")}
    payload["Cmd"] = cmd
    if env is not None:
        payload["Env"] = list(env)
    if labels is not None:
        payload["Labels"] = dict(labels)
    create_endpoint, _extras = networking(inspected, name=name)
    if create_endpoint is not None:
        payload["NetworkingConfig"] = {"EndpointsConfig": create_endpoint}
    return payload


@dataclass(frozen=True)
class ContextPlan:
    """One window change, ready to perform: the create body, the networks to connect
    after it, and what the window was / becomes (for the log line that explains it)."""

    payload: dict
    connect: list[tuple[str, dict]] = field(default_factory=list)
    previous: int | None = None
    target: int | None = None


def plan_context(
    inspected: Mapping[str, Any] | None, kind: str, context_length: int | None, *,
    name: str | None = None, utilization: float | None = None,
    max_num_seqs: int | None = None,
) -> ContextPlan | None:
    """How to make this container serve ``context_length`` — and, for vLLM, run its
    pool at ``utilization`` — or ``None`` for "nothing to do", which is the common case
    and the one worth getting right: a needless recreate costs the box its LLM service
    for a restart and a weight reload.

    ``context_length=None`` means RESTORE the operator's own window (and forget the
    override); a container nobody overrode is then already correct. ``utilization`` is
    applied as given (the caller sizes it from the box's free memory, never above the
    operator's value) and the operator's own value is remembered the first time it is
    changed, so a lowered pool can come back up on a later start.
    """
    if not inspected:
        return None
    config = inspected.get("Config") or {}
    current = configured_context(inspected, kind)
    recorded = recorded_default(inspected)
    labels = dict(config.get("Labels") or {})

    # ── the window ─────────────────────────────────────────────────────────
    window_change = False
    target: int | None = current
    value: str | None = None
    if context_length is None:
        if recorded is not None:                 # overridden before: restore it
            target = _int_or_none(recorded)
            if target != current:
                window_change = True
                labels.pop(DEFAULT_LABEL, None)
                value = None if target is None else str(target)
    else:
        target = int(context_length)
        if target != current:
            window_change = True
            if recorded is None:
                # FIRST override: remember what the operator had, so it can be given back.
                labels[DEFAULT_LABEL] = "" if current is None else str(current)
            value = str(target)

    # ── the pool (vLLM only) ───────────────────────────────────────────────
    util_change = False
    util_value: str | None = None
    if kind == VLLM and utilization is not None:
        now_util = argv_utilization(config.get("Cmd"))
        if now_util is None or abs(now_util - float(utilization)) >= 0.005:
            util_change = True
            util_value = f"{float(utilization):.2f}"
            if UTILIZATION_LABEL not in labels:
                labels[UTILIZATION_LABEL] = "" if now_util is None else f"{now_util:g}"

    # ── the concurrency (vLLM only) ────────────────────────────────────────
    seqs_change = False
    seqs_value: str | None = None
    if kind == VLLM and max_num_seqs is not None:
        now_seqs = argv_max_seqs(config.get("Cmd"))
        if now_seqs != int(max_num_seqs):
            seqs_change = True
            seqs_value = str(int(max_num_seqs))
            if MAX_SEQS_LABEL not in labels:
                labels[MAX_SEQS_LABEL] = "" if now_seqs is None else str(now_seqs)

    if not window_change and not util_change and not seqs_change:
        return None
    if kind == VLLM:
        argv = list(config.get("Cmd") or [])
        if window_change:
            argv = argv_with_context(argv, value)
        if util_change:
            argv = argv_with_utilization(argv, util_value)
        if seqs_change:
            argv = argv_with_max_seqs(argv, seqs_value)
        env = None
    else:
        argv = None
        env = env_with_context(config.get("Env"), value)
    payload = recreate_payload(inspected, argv, env=env, labels=labels, name=name)
    _create, extras = networking(inspected, name=name)
    return ContextPlan(payload=payload, connect=extras, previous=current, target=target)


__all__ = [
    "OLLAMA", "VLLM", "CONTEXT_ENV", "CONTEXT_FLAG", "DEFAULT_LABEL",
    "ContextPlan", "env_context", "argv_context", "configured_context",
    "recorded_default", "env_with_context", "argv_with_context", "networking",
    "recreate_payload", "plan_context", "UTILIZATION_FLAG", "UTILIZATION_LABEL",
    "argv_utilization", "argv_with_utilization", "operator_utilization",
]
