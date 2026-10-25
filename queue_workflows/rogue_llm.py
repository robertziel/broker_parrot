"""ONE LLM DAEMON PER BOX — a second ollama / vLLM container is stopped on sight.

WHY THIS EXISTS. The engine's one-model-per-box rule is enforced by
:mod:`model_residency`, which probes the servers it was CONFIGURED with. A daemon on
another port is invisible to it, and it happens: a benchmark can start a second
``ollama/ollama`` container beside the box's own ollama (the production model volume
mounted read-only) and load the same model into it. Two resident copies push a
unified-memory box toward an OOM death while every probe the engine owns says "one model,
all is well" — and a container stopped by hand can be back within a minute.

WHERE A SECOND DAEMON CAN BE SEEN is the docker socket, so that is where the rule is
enforced — every tick, the way :class:`~queue_workflows.llm_idle_unload.ParkQuiesce`
re-kills a resurrected server on a parked box. Whichever engine process on a box holds
the socket does it (typically the claim worker on a GPU box, the orchestrator on a
control box); the others find no socket and stay quiet.

THE RULE, precisely: a box that runs the engine's OWN ollama (or vLLM) container — the
one :func:`llm_context_lever.container_name` names — may run no other RUNNING container
whose image is that server's. Deliberately narrow:

  * a box whose configured daemon is not running is not policed at all (a dev box with
    an unrelated ollama is left alone; "no second daemon beside ours", not "no ollama
    anywhere");
  * images match on the REPOSITORY (``ollama/ollama``, ``vllm/vllm-openai``), never a
    substring — ``llm-gateway:latest`` is the proxy every client dials, not a server;
  * ``QUEUE_WORKFLOWS_ROGUE_LLM_GUARD=report`` names the rogue and leaves it;
    ``QUEUE_WORKFLOWS_DISABLE_ROGUE_LLM_GUARD`` turns the guard off.

THE SECOND RULE — GPU SERVICES (opt-in, ``QUEUE_WORKFLOWS_GPU_CONTAINER_ALLOW``). After
a reboot, a sibling project's web container (``restart: unless-stopped``, ``--gpus all``)
can come up before vLLM and take host port 8000 — the port every client dials vLLM on.
The switch recreates vLLM, its start loses the port, and the box serves nothing for the
whole warm window; a worker beside it holds GPU memory the admission gate can only read
as "less available". The rule for a GPU box is that nothing uses the GPU there except
what the broker arbitrates, so: a RUNNING container with GPU access (a device request, the nvidia
runtime, or a ``/dev/dri`` / ``/dev/kfd`` / ``/dev/nvidia*`` device) whose name does not
start with an allowed prefix and whose compose project is not an allowed name is

  * STOPPED, and its restart policy set to ``no``, when dockerd would bring it back on
    its own (``always`` / ``unless-stopped`` / ``on-failure``) — that is a SERVICE, and
    "never starts" means the boot path too;
  * NAMED ONCE and left alone when its policy is already ``no`` — a run somebody
    launched by hand (a training job, a benchmark) that dockerd will never resurrect.
    PRESENCE IS NOT THE OFFENCE; MEMORY IS — a hand-launched run is never stopped for
    being there, only by the RAM guard once memory runs short. Such a run shares the box until the box runs short, and then the memory sentinel
    takes it — see :func:`stop_gpu_outsiders`, called from the shed path once the broker
    has given back its OWN servers and the box is still in danger.
    ``QUEUE_WORKFLOWS_ROGUE_GPU_RUNS=stop`` opts a deployment into stopping them on
    sight instead.

A SERVICE is treated differently because it is different: it came back by ITSELF at
boot, which is how a sibling project's web container took vLLM's port 8000 and would
have taken it again every reboot.

The engine's own containers (:func:`llm_context_lever.container_name`) are always
allowed. Unset, the rule is off and no container is inspected. ``=report`` applies here
too.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from queue_workflows.config import _env_flag
from queue_workflows.envcompat import env_get

log = logging.getLogger(__name__)

DISABLE_ENV = "QUEUE_WORKFLOWS_DISABLE_ROGUE_LLM_GUARD"
#: ``stop`` (default) or ``report``.
MODE_ENV = "QUEUE_WORKFLOWS_ROGUE_LLM_GUARD"

#: Image repositories that ARE an LLM daemon, by kind. Matched on the repository path
#: (registry prefix stripped, tag/digest stripped), so ``registry.local/ollama/ollama``
#: counts and ``llm-gateway`` does not.
_REPOS = {
    "ollama": ("ollama/ollama", "ollama"),
    "vllm": ("vllm/vllm-openai", "vllm-openai"),
}

#: Comma-separated container-NAME PREFIXES and/or compose PROJECT names that may use
#: this box's GPU. Unset or empty = the GPU-service rule is off.
ALLOW_ENV = "QUEUE_WORKFLOWS_GPU_CONTAINER_ALLOW"

#: ``report`` (default — name it once, leave it; the memory sentinel takes it if the box
#: runs short) or ``stop`` (on sight) for hand-launched GPU RUNS specifically. Services
#: are governed by :data:`MODE_ENV` alone.
RUNS_ENV = "QUEUE_WORKFLOWS_ROGUE_GPU_RUNS"

#: Restart policies under which dockerd brings a container back by itself — what makes
#: a GPU container a SERVICE rather than a run.
RESTARTING_POLICIES = frozenset({"always", "unless-stopped", "on-failure"})
_GPU_DEVICE_PREFIXES = ("/dev/dri", "/dev/kfd", "/dev/nvidia")
_PROJECT_LABEL = "com.docker.compose.project"

#: After a socket failure, how long before trying again. The gpu heartbeat ticks every
#: 10 s and most claim containers have no socket: one failed connect a minute, not six.
NO_SOCKET_RETRY_S = 60.0

_lock = threading.Lock()
_no_socket_until = 0.0
#: Per container Id: ``(uses_gpu, restart_policy)`` — HostConfig does not change while a
#: container runs, so one inspect per Id, forgotten when the Id leaves the listing.
_gpu_facts: dict[str, tuple[bool, str]] = {}
#: Ids of GPU runs (restart=no) already named, so the WARNING is once, not every tick.
_named_runs: set[str] = set()


@dataclass(frozen=True)
class Rogue:
    name: str
    kind: str
    image: str
    #: The engine's own container of that kind, which this one runs beside.
    ours: str


@dataclass(frozen=True)
class RogueGpu:
    name: str
    image: str
    #: docker restart policy — a member of :data:`RESTARTING_POLICIES` makes it a service.
    restart: str
    #: compose project, ``""`` when not compose-managed.
    project: str

    @property
    def is_service(self) -> bool:
        return self.restart in RESTARTING_POLICIES


def allowed_prefixes() -> tuple[str, ...]:
    raw = env_get(ALLOW_ENV) or ""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def uses_gpu(inspected: Mapping[str, Any]) -> bool:
    """Pure. ``inspected`` is docker's ``GET /containers/{id}/json``: a device request
    (``--gpus``), the nvidia runtime, or a GPU device node passed through."""
    host = (inspected or {}).get("HostConfig") or {}
    if host.get("DeviceRequests"):
        return True
    if str(host.get("Runtime") or "").lower() == "nvidia":
        return True
    for device in host.get("Devices") or []:
        if str((device or {}).get("PathOnHost") or "").startswith(_GPU_DEVICE_PREFIXES):
            return True
    return False


def restart_policy(inspected: Mapping[str, Any]) -> str:
    host = (inspected or {}).get("HostConfig") or {}
    return str((host.get("RestartPolicy") or {}).get("Name") or "no").lower() or "no"


def is_allowed(name: str, project: str, *, allow: Iterable[str], ours: Iterable[str]) -> bool:
    if name in set(ours):
        return True
    return any(name.startswith(prefix) or (project and project == prefix)
               for prefix in allow)


def find_rogue_gpu(
    containers: Iterable[dict], inspect_fn: Callable[[str], Mapping[str, Any] | None], *,
    allow: Iterable[str], ours: Iterable[str],
) -> list[RogueGpu]:
    """Every RUNNING container with GPU access that is neither ours nor allowed —
    services and runs alike (see :attr:`RogueGpu.is_service`). ``inspect_fn(id)``
    answers the container's inspect, or ``None`` when it is already gone."""
    allow = tuple(allow)
    rogues: list[RogueGpu] = []
    for c in containers:
        if str(c.get("State") or "").lower() != "running":
            continue
        name = _name(c)
        project = str((c.get("Labels") or {}).get(_PROJECT_LABEL) or "")
        if is_allowed(name, project, allow=allow, ours=ours):
            continue
        inspected = inspect_fn(str(c.get("Id") or ""))
        if not inspected or not uses_gpu(inspected):
            continue
        rogues.append(RogueGpu(name=name, image=str(c.get("Image") or ""),
                               restart=restart_policy(inspected), project=project))
    return rogues


def image_kind(image: str) -> str | None:
    """``'ollama'`` / ``'vllm'`` for a server image, else ``None``. Repository only."""
    ref = str(image or "").strip()
    if not ref:
        return None
    ref = ref.split("@", 1)[0]                       # drop a digest
    last = ref.rsplit("/", 1)[-1]
    if ":" in last:                                  # drop a tag (never a registry port)
        ref = ref[: len(ref) - len(last)] + last.split(":", 1)[0]
    parts = ref.split("/")
    for kind, repos in _REPOS.items():
        for repo in repos:
            want = repo.split("/")
            if parts[-len(want):] == want:
                return kind
    return None


def _name(container: dict) -> str:
    names = container.get("Names") or []
    return str(names[0]).lstrip("/") if names else ""


def find_rogues(containers: Iterable[dict], *, ollama_name: str, vllm_name: str) -> list[Rogue]:
    """Pure. ``containers`` is docker's ``GET /containers/json``."""
    running = [c for c in containers if str(c.get("State") or "").lower() == "running"]
    ours = {"ollama": ollama_name, "vllm": vllm_name}
    present = {
        kind for kind, name in ours.items()
        if any(_name(c) == name and image_kind(str(c.get("Image") or "")) == kind
               for c in running)
    }
    rogues: list[Rogue] = []
    for c in running:
        kind = image_kind(str(c.get("Image") or ""))
        if kind is None or kind not in present:
            continue
        name = _name(c)
        if name == ours[kind]:
            continue
        rogues.append(Rogue(name=name, kind=kind, image=str(c.get("Image") or ""),
                            ours=ours[kind]))
    return rogues


def reset_for_tests() -> None:
    global _no_socket_until
    with _lock:
        _no_socket_until = 0.0
        _gpu_facts.clear()
        _named_runs.clear()


def _remembered_inspect(api: Callable[..., tuple[int, Any]], listing: list[dict]):
    """An ``inspect_fn`` for :func:`find_rogue_gpu` backed by :data:`_gpu_facts`: one
    real inspect per container Id, the facts kept while the Id is still listed."""
    live = {str(c.get("Id") or "") for c in listing}
    with _lock:
        for gone in [cid for cid in _gpu_facts if cid not in live]:
            _gpu_facts.pop(gone, None)
            _named_runs.discard(gone)

    def inspect(cid: str) -> Mapping[str, Any] | None:
        with _lock:
            facts = _gpu_facts.get(cid)
        if facts is None:
            try:
                status, data = api("GET", f"/containers/{cid}/json")
            except Exception as exc:
                log.debug("[rogue-gpu] inspect %s failed (%s)", cid, exc)
                return None
            if status >= 400 or not isinstance(data, dict):
                return None
            facts = (uses_gpu(data), restart_policy(data))
            with _lock:
                _gpu_facts[cid] = facts
        # Re-shaped so find_rogue_gpu reads it like a real inspect.
        return {"Id": cid, "HostConfig": {
            "DeviceRequests": [{"Capabilities": [["gpu"]]}] if facts[0] else [],
            "RestartPolicy": {"Name": facts[1]}}}

    return inspect


def stop_gpu_outsiders(
    *, api_fn: Callable[..., tuple[int, Any]] | None = None, host: str = "?", why: str = "",
) -> list[str]:
    """THE RAM GUARD'S LAST MOVE. Stop every running container with GPU access that the
    allow-list does not name — run and service alike, whatever the report modes say,
    and never the engine's own ollama/vLLM (the sentinel sheds those by their own path,
    which knows how to stop a server properly).

    Called only when the box is STILL below its danger floor after the broker has given
    back its own servers: at that point something else is taking the box down, and
    "whose work is it" stops being the question. Returns the names stopped; never
    raises. With no allow-list the engine does not know what belongs here and stops
    nothing."""
    allow = allowed_prefixes()
    if not allow:
        return []
    from queue_workflows import llm_context_lever
    from queue_workflows.llm_backends import docker_api

    api = api_fn or docker_api.call
    try:
        status, listing = api("GET", "/containers/json")
    except Exception as exc:
        log.warning("[rogue-gpu] %s: container listing failed while the box was short "
                    "of memory (%s)", host, exc)
        return []
    if status >= 400 or not isinstance(listing, list):
        return []
    ours = (llm_context_lever.container_name("ollama"), llm_context_lever.container_name("vllm"))
    stopped: list[str] = []
    for rogue in find_rogue_gpu(listing, _remembered_inspect(api, listing),
                                allow=allow, ours=ours):
        try:
            status, body = api("POST", f"/containers/{rogue.name}/stop?t=10")
        except Exception as exc:
            log.warning("[rogue-gpu] %s: could not stop %r: %s", host, rogue.name, exc)
            continue
        if status >= 400 and status != 304:
            log.warning("[rogue-gpu] %s: could not stop %r: HTTP %s %r", host, rogue.name,
                        status, body)
            continue
        log.error(
            "[rogue-gpu] %s: STOPPED %r (%s, restart=%s) — this box is out of memory (%s) "
            "and the broker has already given back its own LLM servers; %s does not name "
            "this container, so it is what is left to give. Add it to %s if it must "
            "survive a squeeze.",
            host, rogue.name, rogue.image, rogue.restart, why, ALLOW_ENV, ALLOW_ENV,
        )
        stopped.append(rogue.name)
    return stopped


def _sweep_gpu(
    api: Callable[..., tuple[int, Any]], listing: list[dict], *, allow: tuple[str, ...],
    ours: Iterable[str], host: str, report_only: bool,
) -> list[str]:
    acted: list[str] = []
    ids = {_name(c): str(c.get("Id") or "") for c in listing}
    for rogue in find_rogue_gpu(listing, _remembered_inspect(api, listing),
                                allow=allow, ours=ours):
        where = f"{rogue.name!r} ({rogue.image}, restart={rogue.restart}" + (
            f", compose project {rogue.project!r})" if rogue.project else ")")
        stop_runs = (env_get(RUNS_ENV) or "").strip().lower() == "stop"
        if not rogue.is_service and not stop_runs:
            cid = ids.get(rogue.name, "")
            with _lock:
                first = cid not in _named_runs
                _named_runs.add(cid)
            if first:
                log.warning(
                    "[rogue-gpu] %s: %s is using this box's GPU outside the broker's "
                    "arbitration — a run launched by hand (restart=no), left to run: "
                    "presence is not the offence, MEMORY is. The admission gate accounts "
                    "for what it holds, and if this box runs short the memory sentinel "
                    "stops it. Add it to %s to allow it, or set %s=stop to have it "
                    "stopped on sight.",
                    host, where, ALLOW_ENV, RUNS_ENV,
                )
            continue
        if report_only:
            log.error(
                "[rogue-gpu] %s: %s is a GPU SERVICE outside the broker's arbitration on "
                "a box that serves the broker's models — its memory and ports are "
                "invisible to admission. NOT stopping it (%s=report).",
                host, where, MODE_ENV,
            )
            acted.append(rogue.name)
            continue
        try:
            if rogue.is_service:
                # A service comes back at boot unless the policy goes with the stop; a
                # run is already `no` and its owner's container is not ours to rewrite.
                status, body = api("POST", f"/containers/{rogue.name}/update",
                                   {"RestartPolicy": {"Name": "no"}})
                if status >= 400:
                    log.warning("[rogue-gpu] %s: could not pin %r to restart=no: HTTP %s %r",
                                host, rogue.name, status, body)
            status, body = api("POST", f"/containers/{rogue.name}/stop?t=10")
        except Exception as exc:
            log.warning("[rogue-gpu] %s: could not stop %r: %s", host, rogue.name, exc)
            continue
        if status >= 400 and status != 304:
            log.warning("[rogue-gpu] %s: could not stop %r: HTTP %s %r", host,
                        rogue.name, status, body)
            continue
        log.error(
            "[rogue-gpu] %s: STOPPED %s — GPU work outside the broker's arbitration "
            "on a box that serves the broker's models: its memory and ports are invisible "
            "to admission (a sibling project's web app took vLLM's port 8000 this way). "
            "Its restart policy is now 'no', so dockerd will not bring it back at boot, "
            "and whatever `docker compose up`s it again finds it stopped within a tick. "
            "Add its name prefix or compose project to %s to allow it; %s=report only "
            "names it.",
            host, where, ALLOW_ENV, MODE_ENV,
        )
        acted.append(rogue.name)
    return acted


def sweep(
    *, api_fn: Callable[..., tuple[int, Any]] | None = None, host: str = "?",
    now_fn: Callable[[], float] = time.monotonic,
) -> list[str]:
    """One tick: list the box's containers, stop (or report) every rogue. Never raises;
    returns the names acted on. A box with no docker socket answers ``[]`` and is not
    asked again for :data:`NO_SOCKET_RETRY_S`."""
    global _no_socket_until
    if _env_flag(DISABLE_ENV):
        return []
    now = now_fn()
    with _lock:
        if now < _no_socket_until:
            return []
    from queue_workflows import llm_context_lever
    from queue_workflows.llm_backends import docker_api

    api = api_fn or docker_api.call
    try:
        status, listing = api("GET", "/containers/json")
    except (FileNotFoundError, PermissionError):
        with _lock:
            _no_socket_until = now + NO_SOCKET_RETRY_S
        return []
    except Exception as exc:
        log.debug("[rogue-llm] %s: container listing failed (%s)", host, exc)
        return []
    if status >= 400 or not isinstance(listing, list):
        log.debug("[rogue-llm] %s: container listing answered %s", host, status)
        return []

    ollama_name = llm_context_lever.container_name("ollama")
    vllm_name = llm_context_lever.container_name("vllm")
    rogues = find_rogues(listing, ollama_name=ollama_name, vllm_name=vllm_name)
    report_only = (env_get(MODE_ENV) or "").strip().lower() == "report"
    acted: list[str] = []
    allow = allowed_prefixes()
    if allow:
        acted.extend(_sweep_gpu(api, listing, allow=allow, ours=(ollama_name, vllm_name),
                                host=host, report_only=report_only))
    for rogue in rogues:
        if report_only:
            log.error(
                "[rogue-llm] %s: a second %s daemon %r (%s) is running beside %r — one "
                "LLM server per box, and its model is a second resident copy the engine "
                "cannot see. NOT stopping it (%s=report).",
                host, rogue.kind, rogue.name, rogue.image, rogue.ours, MODE_ENV,
            )
            acted.append(rogue.name)
            continue
        try:
            status, body = api("POST", f"/containers/{rogue.name}/stop?t=5")
        except Exception as exc:
            log.warning("[rogue-llm] %s: could not stop %r: %s", host, rogue.name, exc)
            continue
        if status >= 400 and status != 304:
            log.warning("[rogue-llm] %s: could not stop %r: HTTP %s %r", host,
                        rogue.name, status, body)
            continue
        log.error(
            "[rogue-llm] %s: STOPPED a second %s daemon %r (%s) running beside %r — one "
            "LLM server per box: its model was a second resident copy the engine cannot "
            "see, which is how this box runs out of memory. Whatever started it will find "
            "it stopped again every tick; run benchmarks against %r or on another box. "
            "%s=report only names it; %s turns this guard off.",
            host, rogue.kind, rogue.name, rogue.image, rogue.ours, rogue.ours, MODE_ENV,
            DISABLE_ENV,
        )
        acted.append(rogue.name)
    return acted


__all__ = ["ALLOW_ENV", "RUNS_ENV", "stop_gpu_outsiders", "DISABLE_ENV", "MODE_ENV", "NO_SOCKET_RETRY_S", "RESTARTING_POLICIES",
           "Rogue", "RogueGpu", "allowed_prefixes", "find_rogue_gpu", "find_rogues",
           "image_kind", "is_allowed", "restart_policy", "reset_for_tests", "sweep",
           "uses_gpu"]
