"""One model per GPU box — cross-process, cross-project model arbitration.

THE INVARIANT. A physical GPU box should hold exactly ONE model resident at a
time, **whatever its type**: the engine's warm :class:`~queue_workflows.model_cache.\
ModelCache` holds one model per worker *process*, and an LLM server (ollama/vLLM)
holds its own — they share the same physical VRAM. A box running N GPU workers
(one per project) therefore has N independent caches and **no coordination**: two
projects can warm two different models on one card and contend, or OOM it. The
per-process "one model" rule guarantees nothing at the box level.

This module is the arbiter. A GPU worker ACQUIRES the box's model lease before it
loads:

  * no model held          → **grant**
  * the SAME model held    → **grant** (one copy in VRAM; holders share it — two
                             workers warm on the same model is NOT a violation)
  * a DIFFERENT model held → **DENY**, unless every holder's lease has EXPIRED
                             (a dead worker must not hold VRAM hostage forever)

Keyed by the **physical box** (``config.gpu_box_id``, default the machine
hostname) — deliberately NOT ``host_label``, which differs per project on the
same machine (``box-a`` vs ``box-a-gpu``) and so could never unify them.

SAFE BY DEFAULT. With no store configured the lease is a **no-op that grants
everything** — byte-identical to today's behavior. Enforcement is opt-in via a
store every worker on the box can see: :class:`FileLeaseStore` (each container
mounts one host directory) or a host-provided store
(:func:`queue_workflows.set_gpu_lease_store`).

The decision is a PURE function (:func:`decide`) over plain state + a clock, so
every branch is unit-tested with a virtual clock and no I/O.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

#: How long an acquired lease stays valid without a renew. A holder renews while it
#: keeps the model warm; if its process dies the lease lapses and the next worker
#: may preempt the box (mirrors the job-lease model).
DEFAULT_TTL_S = 120.0


#: The lease key for a NO-MODEL LLM-dispatch job. A box's LLM server (ollama)
#: holds ONE model, so every such job shares this one slot; it is a DIFFERENT key
#: from any diffusion model id, so the two mutually exclude on the box (one model
#: per card across all projects).
LLM_SERVER_SLOT = "__llm_server__"

#: The lease key for a NO-MODEL ComfyUI render job. A ComfyUI render puts ComfyUI's
#: OWN model on the card — a different SERVER KIND from an ollama dispatch — so it
#: must NOT share :data:`LLM_SERVER_SLOT`; a DISTINCT key makes a ComfyUI render and
#: an ollama job mutually exclude on one box (while two ComfyUI renders still share).
#: Used when the worker serves ComfyUI (a wired ComfyUI lifecycle or a configured
#: ComfyUI URL) and the job carries no ``required_model``.
COMFYUI_SERVER_SLOT = "__comfyui_server__"

#: The lease key for a CLUSTER RESERVATION (``node_queue.CLUSTER_SERVER_KIND``): the card is
#: held for a server the engine did not start and cannot see — e.g. an external multi-GPU vLLM spanning
#: several boxes — so it is an occupant of its own. It is neither the LLM server (the hold stops that
#: one) nor a ComfyUI render (such a box may serve both, and a reservation keyed there would share
#: the card with a render), and while it is held every other project's occupant is denied and
#: spills.
CLUSTER_RESERVATION_SLOT = "__cluster_reservation__"

#: Prefix for SELF-MANAGING GPU work — a node that puts its OWN thing on the card
#: and needs no server at all (Blender/EEVEE, a bespoke CUDA kernel, a rasteriser).
#:
#: These are the jobs the lease had no vocabulary for. Its subject was "which
#: MODEL is on the card", so a job that is not a model had to borrow a key that
#: is: :data:`LLM_SERVER_SLOT` unless the box happened to serve ComfyUI. Two real
#: consequences, and the second is the serious one:
#:
#:   * every panel and heartbeat reported the box's LLM model as "in use" while
#:     something entirely different held the card;
#:   * sharing :data:`LLM_SERVER_SLOT` meant such a job did NOT exclude an ollama
#:     dispatch — the lease granted both, and two unrelated workloads landed on
#:     one card. The invariant the lease exists to enforce was off precisely for
#:     the class of job that fits none of its categories.
#:
#: Per MODULE, not one shared key: two renders of the same node share a card
#: happily (same occupant), while a Blender render and a video-model-a render must not.
SELF_MANAGED_SLOT_PREFIX = "__gpu__:"


def self_managed_slot(module: str) -> str:
    """The lease key for self-managing GPU work from ``module``."""
    return f"{SELF_MANAGED_SLOT_PREFIX}{module}"


def describe_slot(slot: str | None) -> str:
    """A slot rendered for humans — panels, heartbeats, log lines.

    The sentinels are internal keys; printing them raw is what made a panel say
    a model was loaded when none was. Anything that shows a lease holder should
    come through here.
    """
    if not slot:
        return "idle"
    if slot == LLM_SERVER_SLOT:
        return "LLM server"
    if slot == COMFYUI_SERVER_SLOT:
        return "ComfyUI render"
    if slot == CLUSTER_RESERVATION_SLOT:
        return "cluster reservation"
    if slot.startswith(SELF_MANAGED_SLOT_PREFIX):
        return f"GPU job: {slot[len(SELF_MANAGED_SLOT_PREFIX):]}"
    return slot                      # a real model id — already human-readable


def is_sentinel_slot(slot: str | None) -> bool:
    """Is this lease key a SENTINEL — an occupant that is not a model?

    Every sentinel in this module is written ``__name__`` (:data:`LLM_SERVER_SLOT`,
    :data:`COMFYUI_SERVER_SLOT`, ``claim_worker.SWITCH_SLOT``) or ``__gpu__:<module>``
    (:func:`self_managed_slot`), and a model id never looks like either — so the
    CONVENTION is the test, and a sentinel added later is covered without editing a
    list that someone would forget. ``tests/test_gpu_box_model_lease_gate.py`` pins the
    named ones to the convention so a rename that breaks it fails loudly.

    Only a non-sentinel belongs in a field called ``current_model``."""
    if not slot:
        return False
    text = str(slot)
    if text.startswith(SELF_MANAGED_SLOT_PREFIX):
        return True
    return text.startswith("__") and text.endswith("__")


def is_llm_slot(slot: str | None) -> bool:
    """Whether ``slot`` is the box's LLM server. The negation is what tells a box
    it may release its LLM RAM: something else owns the card now."""
    return slot == LLM_SERVER_SLOT


class LeaseContended(RuntimeError):
    """The lease file stayed flock'd by another process past the bounded wait.
    Treated exactly like a denial by every caller — deny/spill, never block."""


class ModelLeaseDenied(RuntimeError):
    """Raised when a worker may not load a model because a LIVE holder on this box
    has a different one resident. The caller must not load — let the job be
    re-queued (warm-model affinity will route it to the box that already holds it,
    or it runs here once the incumbent unloads)."""


@dataclass(frozen=True)
class LeaseState:
    """The box's model slot: which model is resident and who is holding it."""
    model: str | None = None
    holders: dict[str, float] = field(default_factory=dict)  # holder id -> expires_at


def decide(state: LeaseState, model_id: str, holder: str, now: float,
           ttl_s: float = DEFAULT_TTL_S) -> tuple[bool, LeaseState]:
    """PURE: may ``holder`` load ``model_id`` on this box? Returns
    ``(granted, new_state)``. Expired holders are dropped (a dead process can't
    be holding VRAM). See the module docstring for the three cases."""
    live = {h: e for h, e in state.holders.items() if e > now}
    others = {h: e for h, e in live.items() if h != holder}
    if state.model == model_id:
        granted = True          # same model already resident — share the one copy
    elif not others:
        granted = True          # nobody live holds a different model — take the box
    else:
        granted = False         # a LIVE holder has a DIFFERENT model resident
    if not granted:
        return False, LeaseState(state.model, live)
    holders = dict(live) if state.model == model_id else {}
    holders[holder] = now + ttl_s
    return True, LeaseState(model_id, holders)


def release_state(state: LeaseState, holder: str) -> LeaseState:
    """PURE: drop ``holder``. The box's model slot clears once nobody holds it."""
    holders = {h: e for h, e in state.holders.items() if h != holder}
    return LeaseState(state.model if holders else None, holders)


# ── stores ───────────────────────────────────────────────────────────────────


class NullLeaseStore:
    """The default: no shared store ⇒ grant everything. Behavior is byte-identical
    to an engine with no arbitration (this module is strictly opt-in)."""

    def read(self, box_id: str) -> LeaseState:
        return LeaseState()

    def update(self, box_id: str, fn: Callable[[LeaseState], tuple[Any, LeaseState]]) -> Any:
        result, _new = fn(LeaseState())
        return result


class FileLeaseStore:
    """A shared-file store: JSON under ``dir``, guarded by an exclusive ``flock`` so
    the read-modify-write is atomic across PROCESSES (and across containers, when
    each mounts the same host directory — that's what makes it a *box*-wide lease).

    The participants deliberately span uids: root worker containers and non-root
    HOST processes (e.g. an editable-install fleet run outside Docker) all
    coordinate on one file. So every artifact is made world-writable (file ``0666``, dir ``0777``,
    chmod best-effort past the umask) — whoever creates it first must never lock the
    others out (the observed failure: a root container created the file ``0644`` and
    the host worker hit ``EACCES``, silently unable to participate). It's a
    coordination lockfile, not data — the classic shared-lock permission."""

    def __init__(self, directory: str) -> None:
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)
        try:
            # 0o1777, not 0o777: cross-uid writability is the requirement (root
            # containers + host procs share this dir), but the STICKY bit — like
            # /tmp's — stops one uid unlinking/squatting another's lease file.
            os.chmod(self.dir, 0o1777)  # nosec B103 — deliberate: shared cross-uid dir, sticky like /tmp
        except OSError:
            pass  # not the owner — fine, as long as we can write into it

    def _path(self, box_id: str) -> str:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in box_id)
        return os.path.join(self.dir, f"gpu_model_lease_{safe}.json")

    def read(self, box_id: str) -> LeaseState:
        try:
            with open(self._path(box_id)) as fh:
                raw = json.load(fh)
            return LeaseState(raw.get("model"), dict(raw.get("holders") or {}))
        except (FileNotFoundError, ValueError):
            return LeaseState()

    def update(self, box_id: str, fn: Callable[[LeaseState], tuple[Any, LeaseState]]) -> Any:
        """Atomic read-modify-write under an exclusive lock."""
        import fcntl
        path = self._path(box_id)
        created = not os.path.exists(path)
        with open(path, "a+") as fh:
            if created:
                try:
                    os.chmod(path, 0o666)  # nosec B103 — deliberate: cross-uid flock target (root containers + host procs)
                except OSError:
                    pass
            # NON-BLOCKING with a bounded retry — a plain
            # LOCK_EX blocks FOREVER when a crashed/stuck peer keeps the fd open,
            # and this lock is taken on the HEARTBEAT thread (the switch hold, the
            # lease tick) — a wedge there freezes heartbeats indefinitely. A lease that
            # cannot be locked within ~2 s is treated as CONTENDED: acquire/renew
            # read that as DENIED (the caller spills — its documented behaviour),
            # release leaves the TTL to expire the entry. Liveness beats fairness
            # here; nothing may ever block a worker thread on another process's fd.
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise LeaseContended(
                            f"lease file {path} locked by another process for >2s")
                    time.sleep(0.05)
            try:
                fh.seek(0)
                body = fh.read()
                try:
                    raw = json.loads(body) if body.strip() else {}
                except ValueError:
                    raw = {}
                state = LeaseState(raw.get("model"), dict(raw.get("holders") or {}))
                result, new = fn(state)
                fh.seek(0)
                fh.truncate()
                json.dump({"model": new.model, "holders": new.holders}, fh)
                fh.flush()
                os.fsync(fh.fileno())
                return result
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# ── the lease ────────────────────────────────────────────────────────────────


def default_box_id() -> str:
    """The PHYSICAL box identity every worker on this machine agrees on — the env
    knob / config if set, else the machine hostname. Never ``host_label`` (which is
    per-project)."""
    from queue_workflows.config import get_config
    from queue_workflows.envcompat import env_get
    return ((env_get("QUEUE_WORKFLOWS_GPU_BOX_ID") or get_config().gpu_box_id or "").strip()
            or socket.gethostname())


class ModelLease:
    """The box's one-model slot. Injectable ``store``/``now_fn`` keep it testable."""

    def __init__(
        self, *,
        box_id: str | None = None,
        store: Any | None = None,
        holder: str | None = None,
        ttl_s: float = DEFAULT_TTL_S,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.box_id = box_id or default_box_id()
        self.store = store if store is not None else NullLeaseStore()
        self.holder = holder or f"{socket.gethostname()}:{os.getpid()}"
        self.ttl_s = float(ttl_s)
        self._now = now_fn or time.time
        self._lock = threading.Lock()

    @property
    def enforcing(self) -> bool:
        return not isinstance(self.store, NullLeaseStore)

    def current(self) -> LeaseState:
        return self.store.read(self.box_id)

    def acquire(self, model_id: str) -> bool:
        """Claim the box for ``model_id``. True ⇒ safe to load. False ⇒ a live
        holder has a DIFFERENT model resident; the caller MUST NOT load."""
        now = self._now()

        def txn(state: LeaseState) -> tuple[bool, LeaseState]:
            return decide(state, model_id, self.holder, now, self.ttl_s)

        with self._lock:
            try:
                granted = bool(self.store.update(self.box_id, txn))
            except LeaseContended:
                # A file we cannot lock within the bound is a box we cannot claim:
                # DENY (the caller spills), never block a worker thread.
                log.warning("[gpu-model-lease] box %s lease file contended — "
                            "treating as DENIED for %r", self.box_id, model_id)
                return False
        if not granted:
            held = self.current().model
            log.warning("[gpu-model-lease] box %s DENIED %r — %r is resident (holder alive)",
                        self.box_id, model_id, held)
        return granted

    def acquire_or_raise(self, model_id: str) -> None:
        if not self.acquire(model_id):
            held = self.current().model
            raise ModelLeaseDenied(
                f"cannot load {model_id!r} on GPU box {self.box_id!r}: {held!r} is "
                f"resident and held by a live worker (one model per GPU box)")

    def renew(self, model_id: str) -> bool:
        """Extend this holder's lease while it keeps the model warm."""
        return self.acquire(model_id)

    def release(self) -> None:
        """Drop this holder. The slot clears when the last holder leaves."""
        def txn(state: LeaseState) -> tuple[None, LeaseState]:
            return None, release_state(state, self.holder)

        with self._lock:
            try:
                self.store.update(self.box_id, txn)
            except LeaseContended:
                log.warning("[gpu-model-lease] box %s contended on release — the "
                            "TTL will expire this holder", self.box_id)


def build_lease() -> ModelLease:
    """Build the process lease from config: a host-injected store wins, else a
    :class:`FileLeaseStore` when ``gpu_model_lease_dir`` is set, else the no-op
    (unenforced) default."""
    from queue_workflows.config import get_config
    from queue_workflows.envcompat import env_get
    cfg = get_config()
    store = cfg.gpu_lease_store
    lease_dir = (env_get("QUEUE_WORKFLOWS_GPU_MODEL_LEASE_DIR") or cfg.gpu_model_lease_dir or "").strip()
    if store is None and lease_dir:
        store = FileLeaseStore(lease_dir)
    ttl = env_get("QUEUE_WORKFLOWS_GPU_MODEL_LEASE_TTL_S")
    ttl_s = float(ttl) if ttl else float(cfg.gpu_model_lease_ttl_s or DEFAULT_TTL_S)
    return ModelLease(store=store, ttl_s=ttl_s)
