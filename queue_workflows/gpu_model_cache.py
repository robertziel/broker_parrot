"""Process-wide GPU warm-model cache + its ``current_model`` advertise.

The live PG claim worker owns the GPU warm slot here. A GPU worker is one
process holding one model (concurrency-1 by contract), so one cache per
process is right.

The cache logic itself lives in :mod:`model_cache` (decoupled from the DB).
This module wires the single process-wide instance and injects the "publish my
current_model to ``worker_heartbeats``" side effect — the gauge's GPU busy
signal + the dispatcher's affinity-routing input. The injection stays a
late-bound shim so a test that monkeypatches :func:`_publish_current_model` is
still observed by the cache.
"""

from __future__ import annotations

import logging
import os
import time

from queue_workflows.envcompat import env_get
import socket

from queue_workflows.model_cache import ModelCache

log = logging.getLogger(__name__)


# Lazily-constructed so importing this module doesn't read the idle-TTL env
# before tests / compose can set it. The singleton is the GPU worker
# process's single warm slot.
_GPU_MODEL_CACHE: ModelCache | None = None


def gpu_model_cache() -> ModelCache:
    """Return the process-wide warm-model cache, constructing it on first
    use. The advertise-side effect is wired via a late-bound shim that
    re-reads :func:`_publish_current_model` off this module on every call —
    so a test that monkeypatches it (incl. the mid-load
    ``current_model=NULL`` publish) is still observed by the cache. The
    cache itself never imports psycopg."""
    global _GPU_MODEL_CACHE
    if _GPU_MODEL_CACHE is None:
        # The box-wide model lease (gpu_model_lease). build_lease() returns a no-op
        # lease unless a store is configured, so an un-opted-in deploy is unchanged;
        # once a store IS set, this one process-wide cache is what claims the card.
        from queue_workflows.gpu_model_lease import build_lease

        _GPU_MODEL_CACHE = ModelCache(
            publish_current_model=lambda m: _publish_current_model(m),
            lease=build_lease(),
            pre_load_check=_box_load_gate,
        )
    return _GPU_MODEL_CACHE


#: Settle the post-eviction verify: an evicted server (ollama ``keep_alive:0``) usually
#: drops the model by the time its POST returns, but the box is RE-PROBED up to
#: ``_GATE_SETTLE_TRIES`` times (sleeping ``_GATE_SETTLE_SLEEP_S`` between) so a
#: slightly-slow unload never spuriously refuses an otherwise-clean load. Returns the
#: moment the box is clean. Injectable (``_GATE_SLEEP``) so tests run with a virtual clock.
_GATE_SETTLE_SLEEP_S = 0.5
_GATE_SETTLE_TRIES = 3
_GATE_SLEEP = time.sleep


def _box_load_gate(model_id: str) -> None:
    """LAST-DEFENCE gate wired into the warm cache — CLEAR-THEN-LOAD, not refuse-and-wait.

    A ``ModelCache`` load is always IN-PROCESS (native diffusion, e.g. sdxl), so
    before the weights hit VRAM this makes the box hold ONLY this model: it evicts every
    OTHER serving kind via its own lever — a resident ollama model gets ``keep_alive:0``,
    a losing vLLM its stop hook, a ComfyUI a ``/free`` — then re-observes and REFUSES
    (``ModelAlreadyLoadedError``) only if a foreign model SURVIVED eviction (an
    un-evictable second model must fail the load, never silently coexist). Every
    probe/evict is best-effort: a dead endpoint degrades to 'saw nothing' and can't wedge
    the load. Logs which kind(s) it cleared so "what server type held the box" is visible."""
    from queue_workflows import llm_probe, model_registry, model_residency
    from queue_workflows.config import get_config
    from queue_workflows.envcompat import env_get
    from queue_workflows.llm_backends import factory as _llm_factory

    # A SERVER-SERVED MODEL NEVER LOADS IN THIS PROCESS — so this gate must not run for it.
    #
    # `ModelSpec.served_by` marks ids an ollama/vLLM serves (the sentinel specs a project
    # registers for its LLM tags: their "loader" returns the id and puts nothing in VRAM).
    # Running the clear-then-load gate for one is actively harmful in two ways, both measured
    # on a live fleet:
    #
    #   1. It EVICTS the very server-resident copy the node is about to call. The node then
    #      dials ollama for a model this gate just unloaded — the "one flag, two meanings"
    #      failure where an LLM CLIENT is treated as an LLM LOADER.
    #   2. It then refuses the load, because the surviving resident is reported by the server
    #      under its full tag (`chat-expert-a:latest`) while the load asks for the bare id
    #      (`chat-expert-a`) — the same model, compared unequal:
    #
    #        MODEL LOAD BLOCKED on box-c-gpu: cleared the box for 'chat-expert-a' (inprocess)
    #        but chat-expert-a:latest[ollama] is STILL resident after eviction
    #
    # 482 `chat-model-a:latest` failures on that fleet share this shape. The residency comparison
    # is canonicalised below as defence in depth, but the real fix is not entering the gate:
    # nothing is about to occupy VRAM here, so there is no second model to prevent.
    _spec = model_registry.MODELS.get(model_id)
    if _spec is not None and getattr(_spec, "served_by", ()):
        log.debug("[box] %s is server-served (%s) — no in-process load, gate skipped",
                  model_id, ",".join(_spec.served_by))
        return

    cfg = get_config()
    label = env_get(cfg.host_label_env, "") or "this-box"
    comfyui_url = (env_get("QUEUE_WORKFLOWS_COMFYUI_URL") or cfg.comfyui_url or "").strip()

    def _reprobe() -> list:
        # Poll until the box is clean (only this in-process model, or empty) or the
        # grace runs out — absorbs a slow ollama unload without a spurious refusal.
        residents = model_residency.probe_box_residents()
        for _ in range(_GATE_SETTLE_TRIES - 1):
            # Canonical tags — see the note in assert_can_load. `chat-expert-a:latest` from an
            # ollama probe is the same model as a load of `chat-expert-a`.
            from queue_workflows.model_catalog import canonical_tag
            _w = canonical_tag(model_id)
            if not [r for r in residents
                    if not (r.server == llm_probe.INPROCESS
                            and canonical_tag(r.model) == _w)]:
                return residents
            _GATE_SLEEP(_GATE_SETTLE_SLEEP_S)
            residents = model_residency.probe_box_residents()
        return residents

    def _stop_vllm() -> None:
        fn = cfg.vllm_stop_fn
        if fn is not None:
            fn()
        else:
            log.error(
                "[box] %s: a vLLM must be evicted to load %s but no stop hook is wired "
                "(set_vllm_lifecycle) — it may survive as a 2nd model.", label, model_id,
            )

    evicted = model_residency.clear_box_for(
        llm_probe.INPROCESS, model_id, model_residency.probe_box_residents(),
        unload_ollama=lambda models: llm_probe.unload_ollama_models(
            _llm_factory.resolve_base_url("ollama"), models),
        stop_vllm=_stop_vllm,
        free_comfyui=lambda: llm_probe.comfyui_free(comfyui_url),
        reprobe=_reprobe,
        label=label,
    )
    if evicted:
        log.warning(
            "[box] %s: cleared %s to load %s:%s — one model per box",
            label,
            ", ".join(sorted({f"{r.server}:{r.model}" for r in evicted})),
            llm_probe.INPROCESS, model_id,
        )


def _reset_gpu_model_cache_for_tests() -> None:
    """TEST-ONLY. Drop the process-wide cache so the next
    :func:`gpu_model_cache` builds a fresh one — keeps the warm-slot state
    (current_model, active count, idle TTL) from leaking across tests."""
    global _GPU_MODEL_CACHE
    _GPU_MODEL_CACHE = None


def _publish_current_model(model_id: str | None) -> None:
    """Update ``worker_heartbeats.current_model`` for THIS GPU worker so
    the dispatcher's affinity routing + the queue gauge can see what's
    loaded. Called by ``ModelCache.require_model`` — once with ``None``
    mid-swap, then with the new model_id after the loader returns.

    No-op when ``AI_LEADS_DISABLE_WORKER_HEARTBEAT`` is set (tests).
    ``current_model`` is GPU-only by design, and the GPU cache is only ever
    constructed by a gpu-queue worker, so this always upserts the ``gpu``
    row. Failures are swallowed: a transient DB blip should not crash a
    worker that already has the model loaded successfully.
    """
    if env_get("QUEUE_WORKFLOWS_DISABLE_WORKER_HEARTBEAT"):
        return
    from queue_workflows.config import get_config
    host = (
        env_get(get_config().host_label_env, "").strip()
        or socket.gethostname()
    )
    try:
        from queue_workflows import model_registry, node_queue
        node_queue.upsert_worker_heartbeat(
            host_label=host, queue="gpu",
            concurrency=1,
            current_model=model_id,
            known_models=model_registry.known_ids(),
        )
    except Exception:
        log.exception(
            "[worker_heartbeat] current_model upsert failed (%s)", model_id,
        )
