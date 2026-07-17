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
            os.chmod(self.dir, 0o777)
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
                    os.chmod(path, 0o666)   # cross-uid: root containers + host procs
                except OSError:
                    pass
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
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
            granted = bool(self.store.update(self.box_id, txn))
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
            self.store.update(self.box_id, txn)


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
