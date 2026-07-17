"""ONE model per GPU box — the engine-owned residency invariant, hard-enforced.

WHY THIS EXISTS (the recurring failure it ends). A GPU box can hold a model three
ways: the ollama daemon, a vLLM sidecar, and the in-process warm ``ModelCache``.
Only the third was ever arbitrated (``gpu_model_lease``); the two SERVERS were
governed by deployment config — ``OLLAMA_MAX_LOADED_MODELS``, ``KEEP_ALIVE``,
compose healthchecks, host crons — and deployment config drifts. Worse, multi-tenant
pooling (migration 0017) puts several PROJECTS on one box, each free to load its own
model through its own path, with nothing at box level able to see the total. The
observed result, repeatedly: a 124 GB unified-memory box at 5 GB free with two ~30 GB
models resident, discovered by a human reading a dashboard.

THE INVARIANT. Across a box's LLM servers, at most ``cap`` (default 1) DISTINCT
models may be resident. The enforcer polls, and on violation it does two things,
always, in this order:

  1. **hard-kills the extras** — ollama models beyond the keeper get an immediate
     ``keep_alive: 0`` unload; a losing vLLM sidecar is stopped;
  2. **raises** :class:`ModelResidencyViolation` — the violation is never silent.
     The daemon loop catches it and logs ``log.exception`` (ERROR + traceback), so
     each occurrence is loud in the worker log; a direct ``enforce_once()`` caller
     gets the raise itself.

KEEPER SELECTION (pure, deterministic — N enforcers on one box converge):
the operator's desired ``llm_server_type`` (worker_controls, 0013) wins across
server types; within a server, the most-recently-used model wins (ollama's
``expires_at`` is the recency proxy: last use + keep_alive). Everything else is
evicted.

DECOUPLING. Collector and killers are injected (wired to
:mod:`queue_workflows.llm_probe` + the vLLM stop hook by the GPU worker), the loop
mirrors the idle reapers (clamped cadence, injectable sleep), and a broken collector
degrades to "saw nothing" — the enforcer can never take down the claim worker.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from queue_workflows import llm_probe
from queue_workflows.envcompat import env_get

log = logging.getLogger(__name__)

#: Max DISTINCT models resident across a box's LLM servers. Default 1 — the whole
#: point. Raise only for a box deliberately serving several small models.
DEFAULT_BOX_MODEL_CAP = int(env_get("QUEUE_WORKFLOWS_BOX_MODEL_CAP", "1"))

#: Enforcement poll cadence (s).
DEFAULT_POLL_S = float(env_get("QUEUE_WORKFLOWS_MODEL_RESIDENCY_POLL_S", "30"))

#: Env kill-switch (tests / opt-out).
_DISABLE_ENV = "QUEUE_WORKFLOWS_DISABLE_MODEL_RESIDENCY_ENFORCER"


class ModelResidencyViolation(RuntimeError):
    """Raised — after the hard-kill — whenever a box held more models than the cap.

    Deliberately an exception, not a return flag: the operator asked for this state
    to be an ERROR every time it occurs, and an exception is the one shape that is
    loud in a log (``log.exception`` in the daemon), fatal in a script, and
    assertable in a test."""


class ModelAlreadyLoadedError(ModelResidencyViolation):
    """Raised by the LAST-DEFENCE load gate (:func:`assert_can_load`) when a loader
    is about to pull a model into VRAM but the box already holds a DIFFERENT one.

    A subclass of :class:`ModelResidencyViolation` so one ``except`` catches both the
    after-the-fact enforcer and this before-the-fact gate. Where the enforcer EVICTS a
    second model that slipped in, this REFUSES the load so the second model never
    forms — the final backstop for a model the cooperative lease never saw (an ollama
    daemon, or a project that doesn't share the lease file)."""


def assert_can_load(model_id: str, residents: list["Resident"], *, label: str = "?") -> None:
    """Last line of defence at model-load time: raise
    :class:`ModelAlreadyLoadedError` if the box already holds any model OTHER than
    ``model_id``. Reloading / re-serving the SAME model the box already holds is
    fine (one model, no violation). ``residents`` is the box's OBSERVED residency
    (ollama ``/api/ps`` + vLLM ``/v1/models`` — see :func:`probe_box_residents`), so
    this catches a rogue model no cooperative lease tracks."""
    foreign = sorted({r.model for r in residents if r.model != model_id})
    if not foreign:
        return
    where = ", ".join(
        sorted({f"{r.model}[{r.server}]" for r in residents if r.model != model_id})
    )
    raise ModelAlreadyLoadedError(
        f"MODEL LOAD BLOCKED on {label}: refusing to load {model_id!r} — the box "
        f"already holds {where}. One model per box: unload it first, or route this "
        f"job to the box that already holds {model_id!r} (warm-model affinity). "
        f"This is the last-defence gate; a second resident model was about to form."
    )


def held_server_types(residents: list["Resident"]) -> list[str]:
    """The distinct serving KINDS resident on the box right now — the answer to "what
    server type is loaded here" (``["ollama"]``, ``["comfyui", "inprocess"]``, ``[]``
    for idle). More than one kind is the cross-project violation this module ends."""
    return sorted({r.server for r in residents})


def describe_box_residency(residents: list["Resident"], *, label: str = "?") -> str:
    """One operator-readable line naming the server kind(s) + model(s) a box holds — so
    "box-c is running ollama:qwen" (or the two-kind violation) is a visible fact,
    logged at claim/heartbeat, not something a human infers from a VRAM dashboard."""
    if not residents:
        return f"[box] {label}: idle — no model resident"
    what = ", ".join(sorted({f"{r.server}:{r.model}" for r in residents}))
    tag = " ⚠ MULTIPLE SERVER KINDS on one box" if len(held_server_types(residents)) > 1 else ""
    return f"[box] {label}: held by {what}{tag}"


def clear_box_for(
    incoming_kind: str,
    incoming_model: str | None,
    residents: list["Resident"],
    *,
    unload_ollama: Callable[[list[str]], object] = lambda models: None,
    stop_vllm: Callable[[], object] = lambda: None,
    free_comfyui: Callable[[], object] = lambda: None,
    unload_inprocess: Callable[[], object] = lambda: None,
    reprobe: Callable[[], list["Resident"]] | None = None,
    label: str = "?",
) -> list["Resident"]:
    """Make the box hold ONLY ``(incoming_kind, incoming_model)`` before a load: evict
    every resident of a DIFFERENT serving kind — a second serving path can never
    coexist — using that kind's own lever (ollama ``keep_alive:0``, vLLM stop, ComfyUI
    ``/free``, in-process unload). Within the SAME kind, a different model is evicted
    too when ``incoming_model`` is concrete; pass ``incoming_model=None`` for an
    LLM-server-slot job that only needs the diffusion kinds cleared (its own server's
    models are the residency enforcer's job, not this gate's).

    This is the proactive half of one-model-per-box: instead of REFUSING when the box
    is dirty and waiting for an idle-unload, it CLEARS the box, then loads. Each lever
    is best-effort (one dead lever never spares the rest). If ``reprobe`` is given, the
    box is re-observed after eviction and :class:`ModelAlreadyLoadedError` is raised
    when a foreign resident STILL stands — an un-evictable model must fail the load, not
    silently become the second one. Returns the residents it evicted."""
    def _foreign(rs: list["Resident"]) -> list["Resident"]:
        return [
            r for r in rs
            if not (r.server == incoming_kind
                    and (incoming_model is None or r.model == incoming_model))
        ]

    foreign = _foreign(residents)
    if not foreign:
        return []

    levers = (
        (llm_probe.OLLAMA,
         lambda: unload_ollama(sorted({r.model for r in foreign if r.server == llm_probe.OLLAMA}))),
        (llm_probe.VLLM, stop_vllm),
        (llm_probe.COMFYUI, free_comfyui),
        (llm_probe.INPROCESS, unload_inprocess),
    )
    for kind, fire in levers:
        if any(r.server == kind for r in foreign):
            try:
                fire()
            except Exception:
                log.exception("[box-clear] %s: %s evict lever failed", label, kind)

    if reprobe is not None:
        try:
            fresh = list(reprobe() or [])
        except Exception:
            fresh = []
        still = _foreign(fresh)
        if still:
            where = ", ".join(sorted({f"{r.model}[{r.server}]" for r in still}))
            raise ModelAlreadyLoadedError(
                f"MODEL LOAD BLOCKED on {label}: cleared the box for {incoming_model!r} "
                f"({incoming_kind}) but {where} is STILL resident after eviction — "
                f"refusing to form a 2nd model. That server's evict lever did not free it."
            )
    return foreign


def probe_box_residents(*, inprocess_model: str | None = None) -> list["Resident"]:
    """The box's OBSERVED residents right now, across EVERY serving kind — as
    :class:`Resident` rows: each ollama-loaded model (with its recency key), a genuine
    vLLM's pinned model, a ComfyUI that holds a diffusion model, and — when the caller
    passes it — the worker's own in-process model. Resolves this box's server URLs via
    the backend factory / config and probes them (:mod:`queue_workflows.llm_probe`).

    * The vLLM leg counts only when its URL is distinct from ollama's AND answers as a
      vLLM (ollama serves ``/v1/models`` too, so a blind probe double-counts one daemon).
    * The ComfyUI leg fires only when a ComfyUI URL is configured
      (``QUEUE_WORKFLOWS_COMFYUI_URL`` / ``config.comfyui_url``) AND it holds a model.
    * ``inprocess_model`` is passed by the GPU worker (its ``ModelCache.current_model``)
      because no probe can observe another process's in-process weights — this leg makes
      THIS worker's own in-process model visible to the report; cross-project in-process
      collisions are the box lease's job, not the probe's.

    Never raises — a dead endpoint contributes nothing. Shared by the residency enforcer,
    the load gate, and the report so all see the box the same way."""
    from queue_workflows.config import get_config
    from queue_workflows.llm_backends import factory as _llm_factory

    residents: list[Resident] = []
    ollama_url = _llm_factory.resolve_base_url("ollama")
    for m in llm_probe.loaded_models_info(ollama_url):
        residents.append(Resident(server=llm_probe.OLLAMA, model=m["name"], mru=m["mru"]))
    vllm_url = _llm_factory.resolve_base_url("vllm")
    if vllm_url and vllm_url != ollama_url \
            and llm_probe.probe_llm_servers(vllm_url) == [llm_probe.VLLM]:
        for mid in llm_probe.vllm_served_models(vllm_url):
            residents.append(Resident(server=llm_probe.VLLM, model=mid, mru=0.0))
    comfyui_url = (env_get("QUEUE_WORKFLOWS_COMFYUI_URL") or get_config().comfyui_url or "").strip()
    if comfyui_url and llm_probe.comfyui_loaded(comfyui_url):
        # ComfyUI exposes no current-checkpoint API, so the model is a kind-sentinel;
        # the KIND (comfyui) is what the one-per-box rule arbitrates on.
        residents.append(Resident(server=llm_probe.COMFYUI, model=llm_probe.COMFYUI))
    if inprocess_model:
        residents.append(Resident(server=llm_probe.INPROCESS, model=inprocess_model))
    return residents


@dataclass(frozen=True)
class Resident:
    """One resident model on one serving path. ``mru`` is a monotonic-ish recency
    key (bigger = more recently used); cross-server comparison only matters when no
    desired server type is known."""

    server: str          # "ollama" | "vllm"
    model: str
    mru: float = 0.0


def decide_evictions(
    residents: list[Resident],
    *,
    cap: int = 1,
    desired_server: str | None = None,
) -> list[Resident]:
    """The pure decision: which residents must die so distinct models ≤ ``cap``.

    Keeper ranking: the desired server type's residents first (operator-set,
    worker_controls 0013), then most-recently-used. The top ``cap`` MODELS survive;
    every resident of any other model is evicted. Deterministic and
    order-independent, so multiple enforcers on one box pick the same keeper."""
    models: dict[str, list[Resident]] = {}
    for r in residents:
        models.setdefault(r.model, []).append(r)
    if len(models) <= cap:
        return []

    def rank(item: tuple[str, list[Resident]]):
        _, rows = item
        on_desired = any(r.server == desired_server for r in rows) if desired_server else False
        return (1 if on_desired else 0, max(r.mru for r in rows))

    keep = {m for m, _ in sorted(models.items(), key=rank, reverse=True)[:cap]}
    return [r for r in residents if r.model not in keep]


class ModelResidencyEnforcer:
    """Poll a box's LLM servers and hard-enforce the one-model cap.

    ``collect_fn() -> list[Resident]`` enumerates what is resident right now;
    ``unload_ollama_fn(models)`` hard-unloads ollama models (``keep_alive: 0``);
    ``stop_vllm_fn()`` stops the vLLM sidecar; ``desired_server_fn()`` names the
    operator's server type for the keeper rule. All injected; every failure path is
    contained (see :meth:`enforce_once` / :meth:`tick`)."""

    def __init__(
        self,
        *,
        collect_fn: Callable[[], list[Resident]],
        unload_ollama_fn: Callable[[list[str]], object],
        stop_vllm_fn: Callable[[], object],
        free_comfyui_fn: Callable[[], object] = lambda: None,
        desired_server_fn: Callable[[], str | None] = lambda: None,
        cap: int | None = None,
        poll_s: float | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        label: str = "?",
    ) -> None:
        self._collect = collect_fn
        self._unload_ollama = unload_ollama_fn
        self._stop_vllm = stop_vllm_fn
        self._free_comfyui = free_comfyui_fn
        self._desired_server = desired_server_fn
        self._cap = DEFAULT_BOX_MODEL_CAP if cap is None else int(cap)
        self._poll_s = DEFAULT_POLL_S if poll_s is None else float(poll_s)
        self._sleep = sleep_fn
        self._label = label
        self._started = False
        self._last_reported: tuple | None = None

    def _report_residency(self, residents: list[Resident]) -> None:
        """INFORM what server kind + model holds this box, but only when it CHANGES —
        so the daemon log carries an operator-readable, up-to-date 'box-c held by
        ollama:qwen' (or a two-kind violation) without per-poll spam. This is the
        engine answering "what server type is loaded here" out loud."""
        key = tuple(sorted({(r.server, r.model) for r in residents}))
        if key == self._last_reported:
            return
        self._last_reported = key
        log.info("%s", describe_box_residency(residents, label=self._label))

    def enforce_once(self) -> list[Resident]:
        """One enforcement pass. Clean box ⇒ ``[]``. Violation ⇒ hard-kill the
        extras, then RAISE :class:`ModelResidencyViolation` (naming box, residents,
        keeper) — the kill always runs first so the raise can never preempt it. A
        collector failure is 'saw nothing' (no kill, no raise): the enforcer must
        never invent a violation out of a dead endpoint."""
        try:
            residents = list(self._collect() or [])
        except Exception:
            log.warning("[model-residency] %s collector failed (skipping pass)", self._label)
            return []
        self._report_residency(residents)
        try:
            desired = self._desired_server()
        except Exception:
            desired = None
        evict = decide_evictions(residents, cap=self._cap, desired_server=desired)
        if not evict:
            return []
        # Hard-kill FIRST — per-path best-effort so one dead lever doesn't spare the rest.
        ollama_models = sorted({r.model for r in evict if r.server == "ollama"})
        if ollama_models:
            try:
                self._unload_ollama(ollama_models)
            except Exception:
                log.exception("[model-residency] %s ollama unload failed", self._label)
        if any(r.server == "vllm" for r in evict):
            try:
                self._stop_vllm()
            except Exception:
                log.exception("[model-residency] %s vllm stop failed", self._label)
        if any(r.server == llm_probe.COMFYUI for r in evict):
            try:
                self._free_comfyui()
            except Exception:
                log.exception("[model-residency] %s comfyui free failed", self._label)
        kept = sorted({r.model for r in residents} - {r.model for r in evict})
        raise ModelResidencyViolation(
            f"MODEL RESIDENCY VIOLATION on {self._label}: "
            f"{len({r.model for r in residents})} models resident "
            f"({', '.join(sorted({f'{r.model}[{r.server}]' for r in residents}))}) "
            f"exceeds cap {self._cap} — hard-killed "
            f"{', '.join(sorted({r.model for r in evict}))}; kept {', '.join(kept)}. "
            f"A box must serve ONE model; find who loaded the extra."
        )

    def tick(self) -> None:
        """One daemon-loop iteration: enforce, and turn a violation into a loud
        ERROR (with traceback) instead of thread death — the invariant keeps being
        enforced on the next tick."""
        try:
            self.enforce_once()
        except ModelResidencyViolation:
            log.exception("[model-residency] %s violation (extras hard-killed)", self._label)
        except Exception:
            log.exception("[model-residency] %s enforcement pass failed", self._label)

    def ensure_started(self) -> None:
        """Arm the poller once. No-op under the env kill-switch or a non-positive
        cadence."""
        if self._started or self._poll_s <= 0 or env_get(_DISABLE_ENV):
            return
        self._started = True
        threading.Thread(
            target=self._loop, name=f"model-residency-{self._label}", daemon=True,
        ).start()
        log.info(
            "[model-residency] %s enforcer armed (cap=%d, poll=%.0fs)",
            self._label, self._cap, self._poll_s,
        )

    def _loop(self) -> None:
        while True:
            self._sleep(self._poll_s)
            self.tick()


__all__ = [
    "Resident",
    "ModelResidencyViolation",
    "ModelAlreadyLoadedError",
    "assert_can_load",
    "clear_box_for",
    "held_server_types",
    "describe_box_residency",
    "probe_box_residents",
    "ModelResidencyEnforcer",
    "decide_evictions",
    "DEFAULT_BOX_MODEL_CAP",
    "DEFAULT_POLL_S",
]
