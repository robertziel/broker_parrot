"""Which (box, server kind) can actually serve which MODEL — the fleet's capability answer.

WHY THIS EXISTS. ``worker_heartbeats.llm_servers_available`` (migration 0014) says which
server KINDS a box runs, and :mod:`llm_probe` learned — the hard way — to observe that
instead of defaulting it. But nothing answers the next question: *which models can those
servers serve?* The claim path approximates it with ``known_models``, a per-worker config
list, so the fleet has no queryable answer and a job naming a model no box holds waits
forever with nothing saying why.

This module is that answer, deliberately built on :mod:`llm_probe`'s lesson: the catalog is
OBSERVED, and **absence is a real state**. An empty catalog means "we do not know", and the
matcher treats that as "cannot serve" — the conservative direction. A gate that read unknown
as *yes* would recreate exactly the bug that had a GPU box advertising ``{ollama}`` while
running no server at all.

── The two job shapes, and why BOTH stay ────────────────────────────
``required_model`` (scalar) is what every row written before this carries, and it is not
deprecated: a single-model job is the common case and must keep working untouched.
``required_models`` (array) adds the ability to say "any of these, in this order, on these
server kinds":

    required_models: [
      {"model": "chat-expert-a",       "server_types": ["vllm"]},
      {"model": "chat-model-a:latest", "server_types": ["vllm", "ollama"]},
    ]

:func:`acceptable_models` flattens either shape into ONE ordered list of ``(model, server)``
pairs, so every caller downstream — claim SQL, dispatch, the arbiter — reasons about a
single representation. That flattening is the DRY seam: the scalar/array distinction exists
at the edge and nowhere else.

DESIGN. DB-FREE and I/O-INJECTED, matching :mod:`model_cache`: nothing here imports psycopg,
the row source is a ``fetch`` callback, and the whole domain is unit-testable with no
database. The engine wires the real fetch in :mod:`node_queue`.
"""
from __future__ import annotations

import logging
import threading
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

log = logging.getLogger(__name__)

#: How long a fetched catalog is trusted before the next read. Short, because a model
#: appearing on a box must become claimable without a worker restart — the catalog is a
#: cache of an observation, never a configuration.
DEFAULT_TTL_S = 30.0


def canonical_tag(model_id: str) -> str:
    """``chat-expert-b`` and ``chat-expert-b:latest`` are the SAME weights — collapse them.

    ollama resolves a bare name to ``:latest``, so the two spellings name one model, and
    the fleet uses both: a schema's ``model:`` field carries the bare tag (that is what a
    caller types and what ``required_model`` is stamped with), while ``/api/tags`` reports
    the resolved one. Comparing them literally is not a cosmetic bug — it is the difference
    between "this box holds the expert" and "this box holds nothing", and the direction it
    fails in is a lane that no box will claim.
    """
    tag = str(model_id or "").strip()
    return tag[: -len(":latest")] if tag.endswith(":latest") else tag


def narrow_to_observed(
    declared: Iterable[str], observed: Iterable[str] | None,
    *, narrowable: Iterable[str] | None = None,
) -> list[str]:
    """The models a worker should advertise: what it DECLARES, kept only if it HAS it.

    ``declared`` is :func:`model_registry.known_ids` — a process-level registry every box
    populates identically at import, so it answers "which models does this software know
    about", never "which weights are on this disk". ``observed`` is what this box's own
    servers just reported to the heartbeat probe. Intersecting them turns the claim gate
    from a declaration into a fact.

    ``narrowable`` is WHICH ids the observation is evidence about — normally
    :func:`model_registry.server_served_ids`. Anything outside it passes through
    untouched, because the probe reads ollama ``/api/tags`` and vLLM ``/v1/models`` and a
    model the worker loads IN-PROCESS can never appear there. Narrowing those away once
    deleted `tts-model-v1` from every box at once and made every job needing it
    unclaimable fleet-wide — silently, and with the perverse property that the lane
    recovered only while a box's ollama was BROKEN (no observation ⇒ fail-open ⇒ the full
    registry came back). ``None`` narrows everything, which is the unit-test shape; real
    callers pass the set.

    Three asymmetries, all in the same direction — never advertise LESS than the truth:

      * An EMPTY or missing observation returns ``declared`` untouched. A cold worker, a
        probe that has not run, a transient endpoint failure — treating any of those as
        "this box serves nothing" would wedge the queue on a box that is merely early.
      * An observation that intersects NOTHING also returns ``declared``. Returning ``[]``
        would be read downstream as "no capability list", which turns the claim gate OFF
        (``node_queue`` gates on ``if known:``) and is strictly WIDER than not narrowing
        at all — a box would claim work for models it has never held.
      * Comparison is on :func:`canonical_tag`, and the DECLARED spelling comes back — the
        registry's ids are what the claim SQL matches ``required_model`` against.
    """
    declared_ids = [str(m).strip() for m in (declared or []) if str(m).strip()]
    observed_canon = {canonical_tag(m) for m in (observed or []) if str(m or "").strip()}
    if not observed_canon:
        return declared_ids

    subject = None if narrowable is None else {str(m).strip() for m in narrowable}
    kept = [m for m in declared_ids
            if (subject is not None and m not in subject)
            or canonical_tag(m) in observed_canon]
    # Never hand the caller an empty list while the box declares something: downstream an
    # empty `known_models` means "no gate", not "no capability".
    return kept if kept or not declared_ids else declared_ids


@dataclass(frozen=True)
class CatalogEntry:
    """One observed capability: this box, this server kind, can serve this model.

    ``digest`` is the server's own content hash for the tag (ollama ``/api/tags``
    reports one per model; vLLM reports none, so it stays ``""``). Two tags with one
    digest are ONE set of bytes — which is how a stable ROLE tag is matched to the
    immutable ARTIFACT tag it currently serves, per box, from observation. An empty
    digest is an honest "the server did not say", never a wildcard: nothing matches it.
    """

    host_label: str
    server_type: str
    model_id: str
    enabled: bool = True
    digest: str = ""


class ModelCatalog:
    """Read-mostly view of the fleet's serving capability.

    ``fetch`` returns the current rows; the result is cached for ``ttl_s`` so the claim
    path does not re-query per candidate row. ``ttl_s=0`` disables caching, which is what
    the tests use to prove the catalog is refreshed rather than frozen at construction.
    """

    def __init__(self, fetch: Callable[[], Iterable[CatalogEntry]],
                 ttl_s: float = DEFAULT_TTL_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._fetch = fetch
        self._ttl_s = float(ttl_s)
        self._clock = clock
        self._lock = threading.RLock()
        self._rows: list[CatalogEntry] = []
        self._fetched_at: float | None = None

    # ── reads ────────────────────────────────────────────────────────
    def _current(self) -> list[CatalogEntry]:
        with self._lock:
            now = self._clock()
            fresh = (self._fetched_at is not None
                     and self._ttl_s > 0
                     and (now - self._fetched_at) < self._ttl_s)
            if not fresh:
                try:
                    self._rows = [r for r in self._fetch() if r.enabled]
                except Exception:
                    # A catalog read that raises must not break a claim. Keep the last
                    # good rows; an empty start stays empty, which is the safe direction.
                    log.debug("[model-catalog] fetch failed; keeping last rows",
                              exc_info=True)
                self._fetched_at = now
            return list(self._rows)

    def models_for(self, host_label: str, server_type: str) -> list[str]:
        return sorted({r.model_id for r in self._current()
                       if r.host_label == host_label and r.server_type == server_type})

    def can_serve(self, host_label: str, server_type: str, model_id: str) -> bool:
        return any(r.host_label == host_label and r.server_type == server_type
                   and r.model_id == model_id for r in self._current())

    def kinds_for(self, host_label: str, model_id: str) -> list[str]:
        """Which server kinds this box holds this model on. Sorted, so two workers
        resolving the same "any kind" pair make the SAME choice."""
        return sorted({r.server_type for r in self._current()
                       if r.host_label == host_label and r.model_id == model_id})

    def boxes_serving(self, model_id: str) -> list[str]:
        """Every box that can serve this model, on any kind — "where can this run?"."""
        return sorted({r.host_label for r in self._current() if r.model_id == model_id})


# ── the job side ─────────────────────────────────────────────────────
def acceptable_models(
    *,
    required_model: str | None,
    required_models: Sequence[dict[str, Any]] | None,
    server_types: Sequence[str] | None = None,
) -> list[tuple[str, str | None]]:
    """Flatten either job shape into ONE ordered list of ``(model, server_type)`` pairs.

    Order is the preference: the first pair a box can serve wins. A pair whose server is
    ``None`` means "any kind" — the scalar path with no stated preference, preserved so an
    old row behaves exactly as it did before this module existed.

    Returns ``[]`` for a job that names no model — unconstrained, NOT unsatisfiable. Those
    two must not be conflated: an unconstrained row is claimable by anyone.
    """
    out: list[tuple[str, str | None]] = []

    def add(model: Any, kinds: Any) -> None:
        model_id = str(model or "").strip()
        if not model_id:
            return                              # a ghost pair is worse than a dropped one
        kind_list = [str(k).strip() for k in (kinds or []) if str(k).strip()]
        if not kind_list:
            out.append((model_id, None))
            return
        for kind in kind_list:
            if (model_id, kind) not in out:
                out.append((model_id, kind))

    if required_models:
        for entry in required_models:
            if not isinstance(entry, dict):
                continue
            add(entry.get("model"), entry.get("server_types"))
        return out

    if required_model:
        add(required_model, server_types)
    return out


def best_match(
    catalog: ModelCatalog, host_label: str,
    pairs: Sequence[tuple[str, str | None]],
) -> tuple[str | None, str | None] | None:
    """The first acceptable pair this box can actually serve.

    Returns ``(None, None)`` for an unconstrained job (it matches any box and names no
    model), and ``None`` when the box can serve none of the pairs — the caller must treat
    those differently, which is why they are different values rather than both falsy.
    """
    if not pairs:
        return (None, None)

    for model_id, kind in pairs:
        if kind is not None:
            if catalog.can_serve(host_label, kind, model_id):
                return (model_id, kind)
            continue
        # "any kind": ask the catalog which kinds this box holds it on. `kinds_for` sorts,
        # so two workers resolving the same pair pick the same server.
        for kind_held in catalog.kinds_for(host_label, model_id):
            return (model_id, kind_held)
    return None


# Artifact-tag grammar — a host-side naming convention:
# `<role>-v<N>-<precision>-<server>`. The engine does not invent
# names; it only recognises the shape so an OBSERVED tag can be tied back to the role it
# serves. Kept permissive on precision (any [a-z0-9]+ run) and strict on the two ends the
# convention is strict about: the version segment and the server suffix.
_ARTIFACT_RE = re.compile(r"^(?P<role>.+)-v\d+-[a-z0-9_]+-(?P<server>[a-z0-9]+)$")


def artifact_role(tag: str) -> str | None:
    """The role an ARTIFACT tag names, or None for a tag that is not one.

    `chat-expert-a-v6-q4km-ollama` → `chat-expert-a`; `chat-expert-a` → None (a role
    tag is not an artifact tag — that distinction is the whole convention)."""
    m = _ARTIFACT_RE.match(canonical_tag(tag))
    return m.group("role") if m else None


def resolve_artifact(
    role_id: str,
    server_type: str | None,
    entries: Iterable[CatalogEntry],
    options: Iterable[str] = (),
) -> str | None:
    """WHICH artifact tag the role currently serves on this box — observed, not declared.

    The question this answers existed because of an eight-day incident: `chat-expert-a`
    resolved to a GGUF built from a void merge (the untuned base) and every job row
    truthfully recorded the role name. A promotion is re-pointing a role tag, so the role
    name can never answer "which bytes"; the artifact tag can, and this derives it from
    the same observation the claim gate already trusts.

    Resolution order, most evidence first:

      1. DIGEST MATCH (ollama): the artifact-shaped tag whose digest equals the role
         tag's digest in ``entries``. Two tags, one digest, one set of bytes — this is
         the by-bytes answer and survives any promotion without redeclaration.
      2. DIRECT ARTIFACT (vLLM): a vLLM serves exactly one model and reports its real
         name; when the observed/served id is ITSELF an artifact tag of this role, that
         id is the answer with no digest needed.
      3. Nothing — ``None``. An absent answer, never a guessed one: stamping the
         DECLARED option without observational support would recreate the exact
         "a name is an assertion nobody verifies" failure this exists to end.

    ``options`` (the spec's declared ``server_models`` for this kind) acts as a
    PREFERENCE among multiple digest-equal artifact tags, not as a fallback: with zero
    observed candidates it contributes nothing.
    """
    role = canonical_tag(role_id)
    if not role:
        return None
    rows = [e for e in entries
            if (server_type is None or e.server_type == server_type) and e.enabled]

    # the role tag's own digest on this box, if its server reported one
    role_digest = next((e.digest for e in rows
                        if canonical_tag(e.model_id) == role and e.digest), "")

    candidates: list[str] = []
    for e in rows:
        tag = canonical_tag(e.model_id)
        if tag == role or artifact_role(tag) != role:
            continue
        if role_digest and e.digest == role_digest:
            candidates.append(tag)              # 1. digest-verified alias
        elif not role_digest and not e.digest:
            candidates.append(tag)              # 2. digest-less server naming the artifact

    if not candidates:
        return None
    wanted = [canonical_tag(o) for o in (options or ())]
    for o in wanted:
        if o in candidates:
            return o
    return sorted(candidates)[0]
