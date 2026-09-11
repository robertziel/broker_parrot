"""GPU-worker model registry.

A ``ModelSpec`` names a model loadable by the long-lived GPU worker.
Nodes with ``gpu: true`` in the workflow JSON declare ``model: "<id>"``
pointing here. The GPU worker resolves the id → loader and caches the
result across consecutive jobs so same-model back-to-back work skips
reload.

This is the registration *target*: the engine owns the registry; the host
registers ITS ModelSpecs INTO it (via the injected builtin-model registrar,
plan §2c). Keep this module cheap to import — no torch, no diffusers at module
load. Loader functions defer heavy imports until called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ModelHandle = Any  # whatever the loader returns — a diffusers pipe, a bespoke object, etc.
ModelLoader = Callable[[], ModelHandle]
ModelUnloader = Callable[[ModelHandle], None]


@dataclass
class ModelSpec:
    """One entry in the model registry.

    - ``id``           — stable string node JSONs point at.
    - ``loader``       — zero-arg factory returning the loaded handle.
    - ``unloader``     — optional; defaults to the GPU-worker's generic
                          drop-ref + gc + empty_cache + malloc_trim path.
    - ``est_vram_gb``  — scheduler hint (today informational only;
                          future: admission control).
    - ``served_by``    — the LLM server kinds that serve this id (``('ollama',
                          'vllm')``), or EMPTY for a model the worker loads
                          IN-PROCESS. See :func:`server_served_ids`.
    - ``server_models`` — the ARTIFACT options behind this id, per server kind:
                          ``{"ollama": ("chat-expert-a-v6-q4km-ollama",),
                          "vllm": ("chat-expert-a-v6-bf16-vllm",)}``. The id above is a
                          ROLE tag — stable across promotions, what traffic dials and
                          what the claim SQL keys on; each artifact tag is immutable
                          (one tag, one set of bytes) and names its precision and
                          serving stack. Declaring the options here lets the panel
                          show what a role CAN resolve to and lets
                          :func:`model_catalog.resolve_artifact` prefer a declared
                          option when observation offers several digest-equal tags.
                          OPTIONAL, and deliberately not required to be current: the
                          per-box truth comes from digest observation, so a stale
                          declaration degrades to "no preference", never to a wrong
                          stamp.
    """
    id: str
    loader: ModelLoader
    unloader: ModelUnloader | None = None
    est_vram_gb: float = 0.0
    served_by: tuple[str, ...] = ()
    server_models: dict[str, tuple[str, ...]] | None = None


# Populated at import by registry consumers. Keep the dict in one
# place so tests can swap entries via ``MODELS[id] = fake_spec``.
MODELS: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    """Register a model spec. Replaces any existing entry for the id."""
    MODELS[spec.id] = spec


def get(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError as e:
        raise KeyError(
            f"unknown model id {model_id!r}. "
            f"Known: {sorted(MODELS)!r}. "
            f"Register via model_registry.register(ModelSpec(...))."
        ) from e


def known_ids() -> list[str]:
    return sorted(MODELS)


def server_served_ids() -> set[str]:
    """The registered ids an LLM SERVER serves — the only ones a probe can observe.

    The claim worker narrows its advertised capability with what its box's servers
    actually reported (``model_catalog.narrow_to_observed``). That observation comes from
    ollama ``/api/tags`` and vLLM ``/v1/models``, so a model the worker loads IN-PROCESS
    can never appear in it — narrowing against it would delete that model from every box
    permanently, and its node-jobs would be claimable by nobody. Measured on a live fleet:
    `tts-model-v1` (a TTS engine, loaded in-process via the warm ModelCache) vanished from
    every box's `known_models` the moment narrowing shipped.

    The default is therefore the SAFE direction: a spec that does not declare
    ``served_by`` is treated as in-process and is never narrowed away. Only ids that
    explicitly say a server serves them are subject to the observation — because those
    are the only ones an empty observation is evidence ABOUT.
    """
    return {spec.id for spec in MODELS.values() if spec.served_by}


def fits_within(vram_total_mb: int | None, *, headroom: float = 1.0) -> list[str]:
    """The registered model ids whose ``est_vram_gb`` fits a machine with
    ``vram_total_mb`` of total GPU VRAM — sorted, the worker's advertised
    ``fits_models``.

    A model fits iff ``est_vram_gb * 1024 * headroom <= vram_total_mb``.
    ``headroom`` (>= 1.0) reserves margin over the bare estimate for runtime
    overshoot (activations, fragmentation); default 1.0 = the raw estimate.

    Semantics of the edges:
      * ``vram_total_mb is None`` (capacity unknown — no GPU probe / cold worker)
        ⇒ return ALL known ids. The claim gate must NOT wedge the queue on a
        worker that simply hasn't measured its VRAM yet; the existing capability
        gate already falls back to claim-any on an empty/uninformative set, and
        a "fits everything" advertisement keeps that behaviour.
      * A model with ``est_vram_gb <= 0`` (unset / informational) fits anywhere —
        it carries no capacity claim, so it is never filtered out.
    """
    if vram_total_mb is None:
        return known_ids()
    cap = int(vram_total_mb)
    hr = max(1.0, float(headroom))
    out = [
        mid for mid, spec in MODELS.items()
        if (spec.est_vram_gb or 0.0) <= 0.0
        or float(spec.est_vram_gb) * 1024.0 * hr <= cap
    ]
    return sorted(out)


def clear_for_tests() -> None:
    """TEST-ONLY. Drops the registry so a test can populate it freshly."""
    MODELS.clear()


def server_model_options(model_id: str) -> dict[str, tuple[str, ...]]:
    """The declared artifact options per server kind for ``model_id``, or ``{}``.

    Empty for an unregistered id or one declared without options — callers treat that as
    "no preference" and fall back to observation (digest match / artifact-tag grammar),
    never as "cannot resolve".
    """
    spec = MODELS.get(model_id)
    return dict(spec.server_models or {}) if spec else {}
