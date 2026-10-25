"""Changing a box's LLM context window — the I/O half, and what it costs to get wrong.

:mod:`queue_workflows.llm_backends.server_context` decides WHAT a window change is; this
performs it, and answers the two questions the rest of the engine asks before deciding
anything:

  * :func:`available` — can this box's window be changed AT ALL? A box with no lever
    cannot grow, so its advertised ceiling is the window it already serves (see
    ``llm_context.ollama_ceiling``) and no switch is ever requested for it. Saying "yes"
    when there is no lever is what would take a box out of service for a switch that
    could never happen.
  * :func:`describe` — what window is the container configured for? For ollama this is
    the AUTHORITATIVE answer for ``/v1`` traffic: a loaded runner may sit at a window some
    other client asked for natively, which the next ``/v1`` call undoes. Cached, because
    the heartbeat asks every ~10 s and a docker inspect per tick is not free.

THE DEFAULT IS THE BOX'S OWN CONTAINER. ``broker-ollama`` / ``vllm`` over the docker
socket, the same seam the vLLM model switch already uses — so a fleet that already lets
its worker drive its sidecar gets context switching with no new wiring, and one that does
not (no socket mounted, a server on another host, k8s) reports "no lever" and simply keeps
its current window. A host with a stranger deployment wires
:func:`queue_workflows.set_llm_context_lever` and this module never touches docker.

VERDICTS, NOT EXCEPTIONS. Every call answers with a ``{"status": …, "reason": …}`` dict in
the vocabulary of :class:`~queue_workflows.server_switch.StartOutcome`, because the switch
acts on it in the same tick: ``permanent`` (no such container, a refused create, no
socket, disabled) ends the out-of-service window NOW instead of at the warm timeout half
an hour later; ``transient`` (a busy or slow daemon) is worth retrying.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from queue_workflows.config import _env_flag, get_config
from queue_workflows.envcompat import env_get
from queue_workflows.llm_backends import docker_api, server_context

log = logging.getLogger(__name__)

OLLAMA = server_context.OLLAMA
VLLM = server_context.VLLM

#: Kill switch. With it set, no box changes its window, every ceiling reads "what I serve
#: now", and no context switch is ever requested — the pre-0028 behaviour exactly.
DISABLE_ENV = "QUEUE_WORKFLOWS_DISABLE_LLM_CONTEXT_SWITCH"

#: The container each kind lives in by default. ``broker-ollama`` is the default
#: name; a deployment that names them otherwise sets these.
_CONTAINER_ENV = {OLLAMA: "QUEUE_WORKFLOWS_OLLAMA_CONTAINER",
                  VLLM: "QUEUE_WORKFLOWS_VLLM_CONTAINER"}
_DEFAULT_CONTAINER = {OLLAMA: "broker-ollama", VLLM: "vllm"}

#: How long an inspect answer is reused. Long enough that a 10 s heartbeat does not
#: inspect every tick, short enough that an operator's own recreate is noticed quickly.
DESCRIBE_TTL_S = 20.0

#: HTTP codes that mean "retrying cannot help as configured": no such container (404), a
#: refused create (400), a conflict (409), a daemon that does not implement it (501).
#: Everything else — a timeout, a reset, a 5xx — may well succeed next tick.
_PERMANENT_HTTP = (400, 403, 404, 409, 501)

_lock = threading.Lock()
_describe_cache: dict[str, tuple[float, "ServerContainer"]] = {}


@dataclass(frozen=True)
class ServerContainer:
    """What this box's container for one server kind is, as far as the lever can see."""

    kind: str
    name: str
    exists: bool
    running: bool = False
    #: The window it is configured for; ``None`` = it says nothing and the server's own
    #: default applies (only the server can report that).
    configured: int | None = None
    #: What the operator had configured before the first override (``None`` = never
    #: overridden), so "no job needs a bigger window" can give it back.
    default: int | None = None
    overridden: bool = False
    #: vLLM only: the pool fraction the OPERATOR chose (the remembered original when the
    #: engine has lowered it), which every start is sized from.
    utilization: float | None = None
    #: vLLM only: what the command line says NOW — the pool this start ran at.
    utilization_now: float | None = None
    #: vLLM only: the concurrency the OPERATOR chose (remembered original), and now.
    max_num_seqs: int | None = None
    max_num_seqs_now: int | None = None
    #: vLLM only: the name it serves the model under (the catalog's ``model_id``), and
    #: the identity of its memory profile (the command line minus the engine's flags).
    served_name: str | None = None
    fingerprint: str | None = None
    #: Which container, and which start of it — an observation is made once per start.
    container_id: str | None = None
    started_at: str | None = None


def container_name(kind: str) -> str:
    return (env_get(_CONTAINER_ENV.get(kind, "")) or "").strip() or \
        _DEFAULT_CONTAINER.get(kind, kind)


def disabled() -> bool:
    return _env_flag(DISABLE_ENV)


def _outcome(status: str, reason: str = "") -> dict:
    return {"status": status, "reason": reason}


def _classify_http(code: int | None, detail: str) -> dict:
    if code is not None and code in _PERMANENT_HTTP:
        return _outcome("permanent", detail)
    return _outcome("transient", detail)


def _classify_exception(exc: BaseException, what: str) -> dict:
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        return _outcome("permanent",
                        f"no usable docker socket at {docker_api.default_socket()} "
                        f"({type(exc).__name__}) — {what} cannot change a container's "
                        f"window; bind-mount it on the worker to enable it")
    return _outcome("transient", f"{type(exc).__name__}: {exc}")


def reset_cache_for_tests() -> None:
    with _lock:
        _describe_cache.clear()


def _invalidate(kind: str) -> None:
    with _lock:
        _describe_cache.pop(kind, None)


def describe(
    kind: str, *, api_fn: Callable[..., tuple[int, Any]] | None = None,
    ttl_s: float = DESCRIBE_TTL_S, now_fn: Callable[[], float] = time.monotonic,
) -> ServerContainer:
    """What this box's ``kind`` container is configured for. NEVER raises: an
    unreachable daemon or a missing container answers ``exists=False``, which every
    caller reads as "no lever here" rather than as a failure."""
    name = container_name(kind)
    now = now_fn()
    with _lock:
        hit = _describe_cache.get(kind)
        if hit is not None and now - hit[0] < ttl_s and hit[1].name == name:
            return hit[1]
    api = api_fn or docker_api.call
    try:
        status, data = api("GET", f"/containers/{name}/json")
    except Exception as exc:
        log.debug("[llm-context] %s: inspect failed (%s)", name, exc)
        info = ServerContainer(kind=kind, name=name, exists=False)
    else:
        if status >= 400 or not isinstance(data, dict):
            info = ServerContainer(kind=kind, name=name, exists=False)
        else:
            recorded = server_context.recorded_default(data)
            argv = (data.get("Config") or {}).get("Cmd")
            state = data.get("State") or {}
            info = ServerContainer(
                kind=kind, name=name, exists=True,
                running=bool(state.get("Running")),
                configured=server_context.configured_context(data, kind),
                default=(None if not recorded else
                         server_context._int_or_none(recorded)),
                overridden=recorded is not None,
                utilization=(server_context.operator_utilization(data)
                             if kind == VLLM else None),
                utilization_now=(server_context.argv_utilization(argv)
                                 if kind == VLLM else None),
                max_num_seqs=(server_context.operator_max_seqs(data)
                              if kind == VLLM else None),
                max_num_seqs_now=(server_context.argv_max_seqs(argv)
                                  if kind == VLLM else None),
                served_name=(server_context.argv_served_name(argv)
                             if kind == VLLM else None),
                fingerprint=(server_context.argv_fingerprint(argv)
                             if kind == VLLM else None),
                container_id=(str(data.get("Id")) if data.get("Id") else None),
                started_at=(str(state.get("StartedAt")) if state.get("StartedAt") else None),
            )
    with _lock:
        _describe_cache[kind] = (now, info)
    return info


def available(kind: str, *, api_fn: Callable[..., tuple[int, Any]] | None = None) -> bool:
    """Can this box's ``kind`` window be changed? A host-wired lever says yes without any
    probing; otherwise there must be a container here to recreate."""
    if disabled():
        return False
    if get_config().llm_context_lever_fn is not None:
        return True
    return describe(kind, api_fn=api_fn).exists


#: The fallback read when a container cannot say when it started.
STARTUP_LOG_TAIL = 4000
#: How long after a start its four startup lines can still appear. A weight load over
#: NFS can take several minutes, the CUDA-graph and fp4 tuning passes minutes
#: more; half an hour covers it and excludes the request log that follows.
STARTUP_WINDOW_S = 1800


def _epoch(stamp: str | None) -> int | None:
    """A docker RFC3339 timestamp as unix seconds. ``None`` for the zero value docker
    writes for "never started", or anything unparseable."""
    text = str(stamp or "").strip()
    if not text or text.startswith("0001-01-01"):
        return None
    import datetime as _dt

    head = text.rstrip("Z")
    if "." in head:                                   # docker's nanoseconds
        base, frac = head.split(".", 1)
        head = f"{base}.{frac[:6]}"
    try:
        return int(_dt.datetime.fromisoformat(head).replace(
            tzinfo=_dt.timezone.utc).timestamp())
    except ValueError:
        return None


def vllm_startup_log(
    *, raw_fn: Callable[..., tuple[int, bytes]] | None = None, tail: int = STARTUP_LOG_TAIL,
    started_at: str | None = None,
) -> str | None:
    """The STARTUP window of this box's vLLM container log, demultiplexed — the half
    hour from ``started_at``, because the API server logs every request afterwards and a
    ``tail=`` read never reaches back past a busy hour (a vLLM serving for hours behind a
    load balancer has its four startup lines far out of tail reach). A
    container that cannot say when it started falls back to the tail. ``None`` when
    there is no socket, no container, or the daemon refuses."""
    name = container_name(VLLM)
    started = _epoch(started_at)
    if started is not None:
        # since is exclusive on some daemons; a second early costs nothing.
        window = f"since={started - 1}&until={started + STARTUP_WINDOW_S}"
    else:
        window = f"tail={int(tail)}"
    try:
        status, raw = (raw_fn or docker_api.call_raw)(
            "GET", f"/containers/{name}/logs?stdout=1&stderr=1&{window}")
    except Exception as exc:
        log.debug("[llm-context] %s: log read failed (%s)", name, exc)
        return None
    if status >= 400 or not isinstance(raw, (bytes, bytearray)):
        return None
    return docker_api.demux_stream(bytes(raw))


def observe_vllm(
    *, total_bytes: int | None, api_fn: Callable[..., tuple[int, Any]] | None = None,
    raw_fn: Callable[..., tuple[int, bytes]] | None = None,
):
    """The memory profile of this box's vLLM as its latest complete start reported it
    (see :func:`llm_context.parse_vllm_startup`), or ``None``. The pool fraction that
    start ran at comes from the log itself, else from the container's command line."""
    from queue_workflows import llm_context

    info = describe(VLLM, api_fn=api_fn)
    if not info.exists:
        return None
    text = vllm_startup_log(raw_fn=raw_fn, started_at=info.started_at)
    if not text:
        return None
    return llm_context.parse_vllm_startup(text, total_bytes=total_bytes,
                                          utilization=info.utilization_now)


def set_context(
    kind: str, context_length: int | None, *,
    api_fn: Callable[..., tuple[int, Any]] | None = None,
    utilization: float | None = None,
    max_num_seqs: int | None = None,
) -> dict:
    """Make this box's ``kind`` server serve ``context_length`` tokens — or, with
    ``None``, go back to the window the operator configured. For vLLM, ``utilization``
    additionally sets the memory pool it starts with (sized by the caller to what the
    box has free — see ``llm_context.vllm_utilization``).

    Returns a ``StartOutcome``-shaped dict; never raises. ``already_running`` means the
    server is already configured for that window and nothing was touched."""
    if disabled():
        return _outcome("permanent", f"LLM context switching is disabled ({DISABLE_ENV})")
    host_lever = get_config().llm_context_lever_fn
    if host_lever is not None:
        try:
            result = host_lever(kind, context_length)
        except Exception as exc:
            return _outcome("transient", f"{type(exc).__name__}: {exc}")
        if isinstance(result, dict):
            return dict(result)
        from queue_workflows.server_switch import StartOutcome

        coerced = StartOutcome.coerce(result)
        return _outcome(coerced.status, coerced.reason)
    return _docker_set_context(kind, context_length, api_fn or docker_api.call,
                               utilization=utilization, max_num_seqs=max_num_seqs)


#: Docker's remove returns before the old endpoint's port binding is fully torn down, so
#: the replacement's start can lose the race for the same port ("driver failed programming
#: external connectivity"). Seen on every recreate on box-c. A moment later it is free.
START_RACE_RETRIES = 3
START_RACE_WAIT_S = 2.0


def _network_mode(inspected: Mapping[str, Any]) -> str:
    return str(((inspected or {}).get("HostConfig") or {}).get("NetworkMode") or "")


def _detached(inspected: Mapping[str, Any]) -> bool:
    """Running with no network endpoint at all. Host networking and ``container:<id>``
    sharing have no endpoint of their own to miss; every other mode does."""
    mode = _network_mode(inspected)
    if mode in ("host", "none") or mode.startswith("container:"):
        return False
    networks = ((inspected or {}).get("NetworkSettings") or {}).get("Networks")
    return not networks


def _stop_quietly(api: Callable[..., tuple[int, Any]], name: str) -> None:
    try:
        api("POST", f"/containers/{name}/stop?t=5")
    except Exception as exc:                          # best effort — it is already broken
        log.debug("[llm-context] %s: stop after a failed start: %s", name, exc)


def _classify_start_failure(api: Callable[..., tuple[int, Any]], name: str, status: int,
                            body: Any) -> dict:
    """A start that still fails after the port-race retries. The daemon can lose the
    port and still leave the process running with no endpoint (see :func:`_detached`),
    so the container is stopped before the failure is reported. A port ANOTHER
    container holds ("port is already allocated") will not free itself — that is
    permanent, and the reason names the port so the operator knows what to move."""
    _stop_quietly(api, name)
    text = str(body)
    if "connectivity" in text.lower() or "already allocated" in text.lower():
        import re

        port = re.search(r"Bind for [^:]+:(\d+)", text)
        return _outcome("permanent",
                        f"could not start {name!r}: host port "
                        f"{port.group(1) if port else '?'} is held by another container "
                        f"({text[:160]}) — free the port or move one of them")
    return _classify_http(status, f"could not start {name!r}: HTTP {status} {body!r}"[:300])


def _start_with_port_race_retry(api: Callable[..., tuple[int, Any]], name: str) -> tuple[int, Any]:
    status, body = api("POST", f"/containers/{name}/start")
    for _ in range(START_RACE_RETRIES):
        if not (status >= 400 and "connectivity" in str(body).lower()):
            break
        time.sleep(START_RACE_WAIT_S)
        status, body = api("POST", f"/containers/{name}/start")
    return status, body


def _docker_set_context(
    kind: str, context_length: int | None, api: Callable[..., tuple[int, Any]], *,
    utilization: float | None = None, max_num_seqs: int | None = None,
) -> dict:
    """The built-in lever: recreate this box's container with one value swapped.

    The ORDER is the name: stop, remove, create, (re)connect the remaining networks,
    start. The container name is the identity every client and every topology entry
    dials, so the replacement must take it back — which it can only do once the old one
    is gone."""
    name = container_name(kind)
    try:
        status, data = api("GET", f"/containers/{name}/json")
    except Exception as exc:
        return _classify_exception(exc, "the LLM context lever")
    if status == 404 or not isinstance(data, dict):
        return _outcome("permanent",
                        f"no container named {name!r} on this box — the LLM context "
                        f"lever has nothing to recreate (set "
                        f"{_CONTAINER_ENV.get(kind, '')} if it is named differently)")
    if status >= 400:
        return _classify_http(status, f"inspect {name!r}: HTTP {status}")

    plan = server_context.plan_context(data, kind, context_length, name=name,
                                       utilization=utilization, max_num_seqs=max_num_seqs)
    if plan is None:
        if bool((data.get("State") or {}).get("Running")):
            if _detached(data):
                # Running, configured — and attached to nothing: unreachable on every
                # address the topology dials. The daemon leaves a container like this
                # behind when a start loses its port race (box-c: vLLM logged
                # 'Application startup complete' with `Networks: {}`). Reported as
                # already_running, the switch would wait its whole warm window for it.
                _stop_quietly(api, name)
                return _outcome("permanent",
                                f"{name} was running without a network endpoint "
                                f"(NetworkMode {_network_mode(data)!r}, attached to "
                                f"nothing) — stopped it; the next start recreates it")
            return _outcome("already_running",
                            f"{name} is already configured for "
                            f"{context_length if context_length is not None else 'its default'} "
                            f"context")
        # Configured right, but STOPPED — the seeded sidecar, or one an earlier verdict
        # stopped. "Already configured" used to be reported as already_running here, and
        # nothing started: the switch warmed for a server that did not exist.
        try:
            status, body = _start_with_port_race_retry(api, name)
        except Exception as exc:
            return _classify_exception(exc, "the LLM context lever")
        if status >= 400 and status != 304:
            return _classify_start_failure(api, name, status, body)
        return _outcome("started", f"{name} started as configured")

    log.warning(
        "[llm-context] %s: recreating for a %s-token window (was %s)%s — the server "
        "restarts and reloads its model", name,
        plan.target if plan.target is not None else "default",
        plan.previous if plan.previous is not None else "its default",
        (f" and a memory pool of {utilization:.2f}" if utilization is not None else "")
        + (f" and {max_num_seqs} concurrent requests" if max_num_seqs is not None else ""),
    )
    try:
        # t=10, not the default 30: the switch already drained this box, so there is
        # nothing to wait for, and every second here is a second it serves nothing.
        api("POST", f"/containers/{name}/stop?t=10")
        status, body = api("DELETE", f"/containers/{name}?force=1")
        if status >= 400 and status != 404:
            return _classify_http(status, f"could not remove {name!r}: HTTP {status} {body!r}"[:300])
        status, body = api("POST", f"/containers/create?name={name}", plan.payload)
        if status >= 400:
            return _classify_http(
                status, f"could not create {name!r}: HTTP {status} {body!r}"[:300])
        for network, endpoint in plan.connect:
            # EVERY network, or the box loses half its reachability silently: create
            # takes one endpoint, the rest are connects.
            status, body = api("POST", f"/networks/{network}/connect",
                               {"Container": name, "EndpointConfig": endpoint})
            if status >= 400:
                log.error("[llm-context] %s: could not reconnect network %s: HTTP %s %r",
                          name, network, status, body)
        status, body = _start_with_port_race_retry(api, name)
        if status >= 400 and status != 304:
            return _classify_start_failure(api, name, status, body)
    except Exception as exc:
        return _classify_exception(exc, "the LLM context lever")
    finally:
        _invalidate(kind)
    return _outcome("started",
                    f"{name} recreated for a "
                    f"{plan.target if plan.target is not None else 'default'}-token window")


__all__ = ["STARTUP_WINDOW_S", 
    "OLLAMA", "VLLM", "DISABLE_ENV", "DESCRIBE_TTL_S", "ServerContainer",
    "container_name", "disabled", "describe", "available", "set_context",
    "reset_cache_for_tests",
]
