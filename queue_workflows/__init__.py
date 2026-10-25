"""``queue_workflows`` — a standalone Postgres-as-queue workflow engine.

A host-agnostic engine (extracted from an earlier, now-retired app): a ``SELECT … FOR
UPDATE SKIP LOCKED`` claim loop woken by ``LISTEN``, lease reclaim, a DAG
dispatcher with an outbox, a GPU warm-model cache, periodic ingest work, and
per-host hw-metrics telemetry. Postgres is the only hard dependency.

Usage — a host wires the engine once at startup, then launches a worker /
scheduler / orchestrator::

    import queue_workflows
    from queue_workflows import model_registry
    from queue_workflows.model_registry import ModelSpec

    queue_workflows.configure(
        db_url_env="QUEUE_WORKFLOWS_DB_URL",
        video_model_ids=frozenset({"video-model-b", "video-model-c"}),
        node_module_package="workflows.nodes",
        container_prefix="myapp-",
    )
    queue_workflows.set_workflow_provider(load_workflow, pipeline_schema)
    queue_workflows.set_builtin_model_registrar(register_builtin_models)
    queue_workflows.register_ingest_task("run_fetch_all", run_fetch_all)
    queue_workflows.set_ingest_schedule([ScheduleEntry(...), ...])

    queue_workflows.claim_worker.main(["--queue", "gpu"])

Every hook has a safe default, so ``import queue_workflows`` +
``configure(...)`` + a reachable Postgres is enough to run the engine
standalone (no host wiring required). The package imports NOTHING from any
host application (``workflows.*``) — enforced by ``tests/test_no_host_import.py``.
"""

from __future__ import annotations

from typing import Any, Callable

from queue_workflows.config import EngineConfig, get_config

__version__ = "1.5.0"

__all__ = [
    "configure",
    "set_node_module_package",
    "set_node_resolver",
    "set_builtin_model_registrar",
    "set_workflow_provider",
    "set_invoke_context",
    "set_vllm_lifecycle",
    "set_llm_context_lever",
    "set_inference_server_lifecycle",
    "set_gpu_lease_store",
    "set_llm_server_resolver",
    "set_llm_servers_available",
    "register_ingest_task",
    "set_ingest_schedule",
    "get_config",
    "EngineConfig",
    "__version__",
]


# ── configuration (host calls these once at startup, before launching) ──


def configure(
    *,
    db_url_env: str | None = None,
    metrics_db_url_env: str | None = None,
    video_model_ids: frozenset[str] | None = None,
    node_module_package: str | None = None,
    host_label_env: str | None = None,
    host_priority_env: str | None = None,
    llm_topology_path: str | None = None,
    container_prefix: str | None = None,
    project: str | None = None,
    ingest_queues: frozenset[str] | None = None,
    ingest_default_budget_s: int | None = None,
    db_backend: str | None = None,
    cancel_orphan_queued_jobs: bool | None = None,
    gpu_requires_local_device: bool | None = None,
    disable_gpu_worker_hw_sampler: bool | None = None,
    broadcast_hw_stats: bool | None = None,
    hw_broadcast_port: int | None = None,
    vlm_pool_node_modules: frozenset[str] | None = None,
    gpu_self_load_node_modules: frozenset[str] | None = None,
    llm_server_idle_ttl_s: float | None = None,
    cpu_worker_threads: int | None = None,
    download_max_concurrent: int | None = None,
    gpu_box_max_concurrent: dict[str, int] | None = None,
) -> EngineConfig:
    """Set engine configuration values. Only the passed keyword args are
    mutated; the rest keep their defaults. Returns
    the live :class:`EngineConfig` for chaining/inspection.

    Safe to call once at host startup before launching a worker / scheduler /
    orchestrator. Idempotent and additive — call again to adjust a subset.
    """
    cfg = get_config()
    with cfg._lock:
        if db_url_env is not None:
            cfg.db_url_env = db_url_env
        if metrics_db_url_env is not None:
            cfg.metrics_db_url_env = metrics_db_url_env
        if video_model_ids is not None:
            cfg.video_model_ids = frozenset(video_model_ids)
        if node_module_package is not None:
            cfg.node_module_package = node_module_package
        if host_label_env is not None:
            cfg.host_label_env = host_label_env
        if host_priority_env is not None:
            cfg.host_priority_env = host_priority_env
        if llm_topology_path is not None:
            cfg.llm_topology_path = llm_topology_path
        if container_prefix is not None:
            cfg.container_prefix = container_prefix
        if project is not None:
            cfg.project = str(project)
        if ingest_queues is not None:
            iq = frozenset(ingest_queues)
            reserved = iq & {"cpu", "gpu", "download"}
            if reserved:
                raise ValueError(
                    f"ingest_queues must not reuse the reserved DAG queue names "
                    f"{sorted(reserved)} (cpu/gpu/download draw from "
                    "workflow_node_jobs); use distinct names for non-DAG "
                    "ingest queues."
                )
            cfg.ingest_queues = iq
        if ingest_default_budget_s is not None:
            cfg.ingest_default_budget_s = int(ingest_default_budget_s)
        if db_backend is not None:
            # "sqlite" or "pg" (aliases folded); anything else raises here, at
            # startup, rather than mis-routing the engine's store.
            from queue_workflows.config import canonical_db_backend

            cfg.db_backend = canonical_db_backend(db_backend)
        if cancel_orphan_queued_jobs is not None:
            cfg.cancel_orphan_queued_jobs = bool(cancel_orphan_queued_jobs)
        if gpu_requires_local_device is not None:
            cfg.gpu_requires_local_device = bool(gpu_requires_local_device)
        if disable_gpu_worker_hw_sampler is not None:
            cfg.disable_gpu_worker_hw_sampler = bool(disable_gpu_worker_hw_sampler)
        if broadcast_hw_stats is not None:
            cfg.broadcast_hw_stats = bool(broadcast_hw_stats)
        if hw_broadcast_port is not None:
            cfg.hw_broadcast_port = int(hw_broadcast_port)
        if vlm_pool_node_modules is not None:
            cfg.vlm_pool_node_modules = frozenset(vlm_pool_node_modules)
        if gpu_self_load_node_modules is not None:
            cfg.gpu_self_load_node_modules = frozenset(gpu_self_load_node_modules)
        if llm_server_idle_ttl_s is not None:
            cfg.llm_server_idle_ttl_s = float(llm_server_idle_ttl_s)
        if cpu_worker_threads is not None:
            # Floored here as well as at the read site: a host that configures 0
            # gets the serial worker, never a lane that claims nothing.
            cfg.cpu_worker_threads = max(1, int(cpu_worker_threads))
        if download_max_concurrent is not None:
            # Same double floor as cpu_worker_threads: a 0 cap would be a
            # download lane that admits nothing, silently, forever.
            cfg.download_max_concurrent = max(1, int(download_max_concurrent))
        if gpu_box_max_concurrent is not None:
            # Stored verbatim, including entries that make no sense: the read site
            # (claim_worker.resolve_box_max_concurrent) is where a non-positive or
            # unreadable value is IGNORED, because the env spelling must obey the
            # identical law and there is one implementation of it. Floor-here would
            # silently turn a host's 0 into 1 — a ceiling the host never asked for.
            cfg.gpu_box_max_concurrent = dict(gpu_box_max_concurrent)
    return cfg


# ── node registration (the node-module resolver) ──


def set_node_module_package(package: str) -> None:
    """Set the dotted package the node-module resolver imports under (e.g.
    ``"workflows.nodes"``). A stored ``node_module`` value of ``"smoke"`` then
    imports ``workflows.nodes.smoke``. Empty string ⇒ the stored value is
    imported as a fully-qualified module name."""
    cfg = get_config()
    with cfg._lock:
        cfg.node_module_package = package


def set_node_resolver(resolver: Callable[[str], Any]) -> None:
    """Inject a fully custom node-module resolver (``Callable[[str], module]``)
    — overrides :func:`set_node_module_package`. The module must expose a
    ``run(...)`` callable."""
    cfg = get_config()
    with cfg._lock:
        cfg.node_resolver = resolver


# ── model registration (GPU warm-cache registry) ──


def set_builtin_model_registrar(registrar: Callable[[], None]) -> None:
    """Set the idempotent builtin-model registrar — the empty-registry
    re-registration fallback (``model_cache``) + the once-at-startup
    registration (``claim_worker`` / ``orchestrator``). The host wires its
    ``register_builtin_models`` here; it should register ``ModelSpec``s into
    ``queue_workflows.model_registry``."""
    cfg = get_config()
    with cfg._lock:
        cfg.builtin_model_registrar = registrar


# ── workflow/pipeline provider (DAG source) ──


def set_workflow_provider(
    load_workflow: Callable[[str], dict],
    pipeline_schema: Callable[[str], dict],
    *,
    resolve_ref: Callable[[Any, dict], Any] | None = None,
) -> None:
    """Inject the DAG definition source the dispatcher reads from:

    - ``load_workflow(name) -> dict``    — the workflow definition.
    - ``pipeline_schema(name) -> dict``  — the pipeline schema (owns the node
                                           DAG).
    - ``resolve_ref`` (optional)         — a custom ref resolver; defaults to
                                           the engine's
                                           :func:`queue_workflows.refs.resolve_ref`.
    """
    cfg = get_config()
    with cfg._lock:
        cfg.workflow_loader = load_workflow
        cfg.pipeline_schema_loader = pipeline_schema
        if resolve_ref is not None:
            cfg.resolve_ref = resolve_ref


# ── per-node invoke wrapper (host setup/teardown around each node run) ──


def set_invoke_context(factory: Callable[[dict, dict], Any]) -> None:
    """Set the per-node invoke wrapper — a ``Callable[[job, run], ContextManager]``
    whose context manager brackets each node invoke: ``__enter__`` does host setup
    (e.g. pin a run-context ContextVar + capture a live mock flag) and yields a
    ``finalize(context_delta) -> context_delta`` callable applied ONLY on success
    (e.g. stamp a per-node ``_mocked`` marker); ``__exit__`` does teardown on every
    exit path. Default unset ⇒ the engine runs nodes directly (no wrapping). See
    :class:`queue_workflows.config.EngineConfig.invoke_context`."""
    cfg = get_config()
    with cfg._lock:
        cfg.invoke_context = factory


def set_llm_servers_available(servers: list[str]) -> None:
    """Declare which LLM server types THIS host can actually run — published in
    the worker heartbeat (migration 0014) so the queue UI gates its per-machine
    server-type control. ``["ollama"]`` is the universal baseline; an NVIDIA host
    with the vllm sidecar rendered passes ``["ollama", "vllm"]``. See
    :attr:`queue_workflows.config.EngineConfig.llm_servers_available`."""
    cfg = get_config()
    with cfg._lock:
        cfg.llm_servers_available = list(servers)


def set_vllm_lifecycle(
    stop_fn: Callable[[], bool],
    start_fn: Callable[[str], None],
) -> None:
    """Wire the vllm-sidecar stop/start the idle supervisor + model-switch drive.

    ``stop_fn() -> bool`` frees the sidecar's VRAM (True iff it stopped one);
    ``start_fn(model_id) -> None`` (re)starts it serving ``model_id``. A host
    that runs vllm as a SEPARATE container wires these so the in-worker supervisor
    can control the SIBLING sidecar without a docker restart policy (e.g. →
    docker Engine API over the unix socket). Threaded into ``VLLMBackend`` by the
    backend factory. See
    :attr:`queue_workflows.config.EngineConfig.vllm_stop_fn`."""
    cfg = get_config()
    with cfg._lock:
        cfg.vllm_stop_fn = stop_fn
        cfg.vllm_start_fn = start_fn


def set_llm_context_lever(fn: Callable[[str, int | None], Any] | None) -> None:
    """Wire how THIS box changes the context window its LLM server serves.

    ``fn(server_kind, context_length | None)`` — make the box's ``'ollama'`` /
    ``'vllm'`` server serve that many tokens, or the operator's own value when ``None``.
    May return a verdict (a ``StartOutcome``, a ``{"status": …, "reason": …}`` dict, a
    bool, ``None``), which the switch acts on in the same tick.

    Only needed for a deployment whose server is NOT a local container: by default the
    engine recreates the box's own ``broker-ollama`` / ``vllm`` over the docker socket
    (:mod:`queue_workflows.llm_context_lever`), and a box with no socket simply keeps the
    window it serves. ``None`` restores that default. See
    :attr:`queue_workflows.config.EngineConfig.llm_context_lever_fn`."""
    cfg = get_config()
    with cfg._lock:
        cfg.llm_context_lever_fn = fn


def set_inference_server_lifecycle(
    start_fn: Callable[[], None] | None,
    stop_fn: Callable[[], None] | None,
) -> None:
    """Wire the GPU ON/OFF toggle to this machine's inference SERVER on/off.

    Turning the box's GPU worker OFF (worker_controls, migration 0012) stops the
    server to free the GPU's VRAM; turning it ON starts it. ``start_fn() -> None``
    starts this machine's LLM server, ``stop_fn() -> None`` stops it — the host wires
    them to whatever it runs (e.g. ``docker start/stop ollama`` over the UDS),
    so the engine stays server-type-agnostic. Either ``None`` ⇒ that direction is a
    no-op, so a deployment that never wires them keeps today's behaviour (the GPU
    toggle parks only the worker). The worker invokes them GPU-lane-only and
    best-effort. See
    :attr:`queue_workflows.config.EngineConfig.inference_server_start_fn`."""
    cfg = get_config()
    with cfg._lock:
        cfg.inference_server_start_fn = start_fn
        cfg.inference_server_stop_fn = stop_fn


def set_comfyui_lifecycle(
    start_fn: Callable[[], None] | None,
    stop_fn: Callable[[], None] | None,
) -> None:
    """Wire the host's ComfyUI START/STOP levers so ComfyUI can be a first-class box
    tenant (not just an evict-only loser).

    ``start_fn() -> None`` brings up this box's ComfyUI server (e.g. ``docker start
    comfyui`` over the UDS); ``stop_fn() -> None`` takes it down. When a node_job
    requests comfyui, :func:`queue_workflows.comfyui.acquire_box_for_comfyui` clears
    the box of rival serving kinds and calls ``start_fn`` to ensure ComfyUI holds the
    card. Either ``None`` ⇒ that direction is a no-op, so a deployment that manages
    ComfyUI externally (an always-up container) keeps today's behaviour. See
    :attr:`queue_workflows.config.EngineConfig.comfyui_start_fn`."""
    cfg = get_config()
    with cfg._lock:
        cfg.comfyui_start_fn = start_fn
        cfg.comfyui_stop_fn = stop_fn


def set_gpu_lease_store(store: Any | None) -> None:
    """Wire a cross-box GPU model-lease store (one-model-per-GPU-box arbitration,
    :mod:`queue_workflows.gpu_model_lease`). ``store`` implements ``read(box_id)``
    + ``update(box_id, fn)`` (e.g. a DB-backed store shared across boxes). ``None``
    (default) ⇒ the file store when ``gpu_model_lease_dir`` is set, else a no-op
    that grants every load. See
    :attr:`queue_workflows.config.EngineConfig.gpu_lease_store`."""
    cfg = get_config()
    with cfg._lock:
        cfg.gpu_lease_store = store


def set_llm_server_resolver(
    resolver: Callable[[dict], Any] | None,
) -> None:
    """Wire the per-DISPATCH LLM server resolver.

    ``resolver(job) -> server`` is called once per node-job by
    ``node_executor.execute_node``; its return is threaded into the node as
    ``run(llm_server=...)`` (parallel to ``model_handle``). This is how
    broker_parrot hands each run node-job the exact vllm/ollama server (TYPE +
    ADDRESS + PORT) to call, instead of every node hardcoding a URL. The resolver
    body owns selection across the box's (possibly MULTIPLE) servers — and across
    boxes if it consults the DB heartbeat url advertisement. Pass ``None`` to
    clear. See :attr:`queue_workflows.config.EngineConfig.llm_server_resolver`."""
    cfg = get_config()
    with cfg._lock:
        cfg.llm_server_resolver = resolver


# ── ingest task + schedule registration (periodic work) ──


def register_ingest_task(name: str, callable_: Callable[[str], dict]) -> None:
    """Register a periodic ingest callable under ``name``. The claim worker
    runs it when an ``ingest_jobs`` row with that ``task_name`` is claimed; the
    callable takes the ``reason`` string and returns a JSON-able result dict.
    The registered names are also the valid ``task_name`` set
    ``node_queue.enqueue_ingest_job`` validates against."""
    cfg = get_config()
    with cfg._lock:
        cfg.ingest_task_map[name] = callable_


def set_ingest_schedule(schedule: list) -> None:
    """Set the scheduler's periodic schedule (a list of
    ``queue_workflows.scheduler.ScheduleEntry``). The ``Ticker`` fires it; the
    boot-kick enqueues the non-freshness entries."""
    cfg = get_config()
    with cfg._lock:
        cfg.ingest_schedule = list(schedule)
