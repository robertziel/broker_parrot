"""Engine configuration + the dependency-inversion seams.

``queue_workflows`` is a host-agnostic Postgres-as-queue engine. Everything
that used to couple it to a specific application is now an
*injected* hook held on a process-wide :class:`EngineConfig` singleton. A
host wires those hooks once at startup (``queue_workflows.configure(...)`` +
the ``set_*`` / ``register_*`` helpers) before launching a claim worker /
scheduler / orchestrator.

Every hook has a **safe default** so ``import queue_workflows`` +
``configure()`` + a reachable Postgres is enough to run the engine
standalone (no host wiring required):

  * ``db_url_env``           — env var holding the DSN (default
                               ``QUEUE_WORKFLOWS_DB_URL``; a host may pass
                               its own).
  * ``video_model_ids``      — GPU models on the tight render budget (empty).
  * ``node_module_package``  — dotted package the node-module resolver imports
                               under (empty → the stored ``node_module`` value
                               is imported as a fully-qualified module).
  * ``container_prefix``     — cgroup-attribution container-name prefix
                               (default ``""`` = attribution off).
  * the workflow provider    — ``load_workflow`` / ``pipeline_schema`` /
                               ``resolve_ref`` the dispatcher reads the DAG
                               from (defaults raise / use the built-in
                               ``refs.resolve_ref``).
  * the builtin-model registrar — the empty-registry re-registration fallback
                               (default no-op).
  * the ingest dispatch map + schedule — periodic-work callables + cron
                               (default empty).

This module imports NOTHING from the engine's other modules (it's a leaf), so
any engine module can ``from queue_workflows import config`` without a cycle.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

# ── env-var NAMES (configurable; defaults are QUEUE_WORKFLOWS_*) ─────────────
#
# The engine reads these env *names* off the config so a host can rename them
# without touching engine code. The DEFAULTS are the canonical
# ``QUEUE_WORKFLOWS_*`` names.


# Engine BUILT-IN pool-lane node modules (shipped in queue_workflows.builtin_nodes):
# always exempt from the gpu required-model guard and always resolvable, in every
# project, regardless of host configuration. See EngineConfig.effective_vlm_pool_node_modules
# and resolve_node_module's builtin fallback.
BUILTIN_POOL_NODE_MODULES: frozenset[str] = frozenset({"comfyui"})

#: The ONLY engine backends — the two relational stores behind the dialect seam
#: (``dialect.py``) — and the aliases each accepts. Every other name is
#: rejected, never mapped to one of these.
_DB_BACKEND_ALIASES = {
    "sqlite": "sqlite", "pg": "pg", "postgres": "pg", "postgresql": "pg",
}


def canonical_db_backend(name: str, *, source: str = "db_backend") -> str:
    """Normalize a ``db_backend`` spelling (case/whitespace-insensitive, aliases
    folded) to ``"sqlite"`` or ``"pg"``, or raise ``ValueError`` naming the
    valid values. The ONE validator ``configure()``, the
    ``QUEUE_WORKFLOWS_DB_BACKEND`` env default and the ``--db-backend`` CLI
    flags share, so a typo or a removed backend fails loudly at startup instead
    of mis-routing to a store the operator never chose. ``source`` names the
    knob in the error."""
    norm = _DB_BACKEND_ALIASES.get(str(name).strip().lower())
    if norm is None:
        raise ValueError(
            f"{source}={name!r} is not a supported backend "
            f"(valid: {sorted(_DB_BACKEND_ALIASES)}; the engine runs only on "
            "sqlite or Postgres)"
        )
    return norm


def _default_db_backend() -> str:
    """Default ``db_backend``, read from ``QUEUE_WORKFLOWS_DB_BACKEND`` (the env
    knob that reaches the standalone console scripts) and validated through
    :func:`canonical_db_backend`. Unset ⇒ ``"sqlite"`` (the v1.0.0 default)."""
    raw = os.environ.get("QUEUE_WORKFLOWS_DB_BACKEND")
    if not raw:
        return "sqlite"
    return canonical_db_backend(raw, source="QUEUE_WORKFLOWS_DB_BACKEND")


def _env_flag(name: str) -> bool:
    """Truthiness for a ``QUEUE_WORKFLOWS_*`` boolean knob, read through
    ``envcompat`` (which refuses a retired legacy twin). Anything
    in ``{1,true,yes,on}`` (case-insensitive) is on; unset or anything else is
    off, so a typo fails safe rather than silently enabling a feature."""
    from queue_workflows.envcompat import env_get
    raw = env_get(name)
    return bool(raw) and raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_broadcast_hw_stats() -> bool:
    """Default ``broadcast_hw_stats``, from ``QUEUE_WORKFLOWS_BROADCAST_HW_STATS``
    — the env knob that reaches the standalone console scripts, which have no
    host ``configure()``. Unset ⇒ ``False`` (off, the safe default: a host must
    opt in before its telemetry goes on the wire)."""
    return _env_flag("QUEUE_WORKFLOWS_BROADCAST_HW_STATS")


def _default_hw_broadcast_port() -> int:
    """Default ``hw_broadcast_port``, from ``QUEUE_WORKFLOWS_HW_BROADCAST_PORT``.
    Unset or unparseable ⇒ :data:`queue_workflows.hw_lan.DEFAULT_PORT`."""
    from queue_workflows.envcompat import env_get
    raw = env_get("QUEUE_WORKFLOWS_HW_BROADCAST_PORT")
    try:
        return int(raw) if raw else 48291
    except (TypeError, ValueError):
        return 48291


def _default_project() -> str:
    """Default ``project``, read from ``QUEUE_WORKFLOWS_PROJECT`` (the env knob that
    reaches entrypoints which don't go through a host ``configure(project=...)`` —
    e.g. standalone scripts that hand-roll their own ``configure``). Unset ⇒ ``""``
    (the single-tenant default). Mirrors ``QUEUE_WORKFLOWS_DB_BACKEND``."""
    return os.environ.get("QUEUE_WORKFLOWS_PROJECT", "")


@dataclass
class EngineConfig:
    """Process-wide engine configuration. One instance lives in this module
    (:data:`_CONFIG`); access it via :func:`get_config`. Mutated only by the
    public ``configure`` / ``set_*`` / ``register_*`` helpers in
    ``queue_workflows.__init__`` — never directly by engine modules at import
    time (so a host can configure AFTER import)."""

    # ── env-var names ────────────────────────────────────────────────────────
    db_url_env: str = "QUEUE_WORKFLOWS_DB_URL"
    #: env var holding the DSN that hw-metrics telemetry is published to + read
    #: from — the shared "broker" Postgres, so every project shows the SAME
    #: fleet-wide hardware view instead of each sampling its own DB. ``None`` ⇒
    #: fall back to :attr:`db_url_env` (so a project whose queue DB already IS the
    #: broker needs no extra wiring; a project on its own queue DB sets this to the
    #: broker DSN env). hw-metrics is NOTIFY-only + Postgres-only, so this is
    #: always a pg DSN regardless of :attr:`db_backend`. See ``hw_metrics`` /
    #: ``hw_feed``.
    metrics_db_url_env: str | None = None
    host_label_env: str = "QUEUE_WORKFLOWS_HOST_LABEL"
    host_priority_env: str = "QUEUE_WORKFLOWS_GPU_CONSUMER_PRIORITY"
    #: env vars holding the per-machine LLM server ROOT URLs the backend factory
    #: (``queue_workflows.llm_backends.factory``) reads. The DB (worker_controls,
    #: migration 0013) owns WHICH server type a machine runs + its tunables; the
    #: URL is deployment topology (set per host by ansible), so it stays in env.
    #: Names default to the ``QUEUE_WORKFLOWS_*`` vars; values fall back to the
    #: localhost defaults below when the env is unset.
    ollama_url_env: str = "QUEUE_WORKFLOWS_OLLAMA_URL"
    vllm_url_env: str = "QUEUE_WORKFLOWS_VLLM_URL"
    #: OPTIONAL path to a per-box LLM-server TOPOLOGY YAML
    #: (:mod:`queue_workflows.llm_backends.topology`). When set, the backend factory
    #: resolves a box's LLM server ROOT URL by its ``host_label`` FROM THIS FILE
    #: (best route per box — loopback on the server host, the fast interconnect from
    #: a client host) IN PREFERENCE TO ``ollama_url_env`` / ``vllm_url_env``. The DB
    #: still owns WHICH server type each box runs. ``None`` (default) ⇒ the env path
    #: only, so every existing consumer is byte-compatible. See
    #: ``docs/llm_topology.example.yml``.
    llm_topology_path: str | None = None

    # ── value config ──────────────────────────────────────────────────────────
    #: GPU model ids on the tight per-job video render budget (claim_worker).
    video_model_ids: frozenset[str] = frozenset()
    #: Node modules that are genuine VLM-facade (HTTP to the per-host vLLM/ollama
    #: server) and therefore SAFE to run PAR-concurrently in the GPU pool lane.
    #: When non-empty, every OTHER no-model GPU job (heavy in-process work —
    #: erasers, detectors, builders) is routed to the conc-1 inline lane instead,
    #: so a PAR>1 box never runs several heavy in-process GPU loads at once.
    #: Empty (default) ⇒ legacy behaviour: every no-model GPU job is pool-eligible
    #: (keeps other consumers byte-identical).
    vlm_pool_node_modules: frozenset[str] = frozenset()
    #: GPU node modules that intentionally run WITHOUT a cache-managed ``model``
    #: and are therefore EXEMPT from the required-model guard
    #: (:func:`dispatcher._assert_gpu_nodes_declare_model`). Genuine VLM-facade
    #: nodes already live in ``vlm_pool_node_modules``; this set is for in-process
    #: self-loaders not yet migrated onto the warm cache (they manage their own
    #: load/unload). A ``gpu:true`` node whose module is in NEITHER set MUST
    #: declare a ``model`` — else the run is rejected at expansion (no more silent
    #: ``required_model=NULL`` → wrong lane / no warm cache / blank heartbeat).
    gpu_self_load_node_modules: frozenset[str] = frozenset()
    #: Dotted package the node-module resolver imports under (e.g.
    #: ``"workflows.nodes"``). Empty ⇒ the stored ``node_module`` is treated as
    #: a fully-qualified importable module name.
    node_module_package: str = ""
    #: cgroup-attribution container-name prefix (hw_metrics per-container slice).
    #: Empty (the default) ⇒ attribution is off.
    container_prefix: str = ""
    #: Tenant identity of THIS client on a SHARED broker Postgres (migration
    #: 0017). One broker DB holds one cpu + one gpu (+ ingest) queue across all
    #: projects; every queue record carries ``project`` and a client enqueues +
    #: claims ONLY rows whose ``project`` matches this value. Default ``""`` is the
    #: single-tenant sentinel: every row is ``''`` and the claim filter
    #: ``project=''`` matches them all, so a single-Postgres-per-project deploy is
    #: byte-compatible with no host wiring. Set via ``configure(project="app-a")`` OR by exporting
    #: ``QUEUE_WORKFLOWS_PROJECT`` (the env knob, for entrypoints that hand-roll
    #: their own ``configure`` and never pass ``project`` — e.g. standalone scripts).
    project: str = field(default_factory=_default_project)
    #: OBSERVED LLM-server capability this worker advertises in its heartbeat
    #: (migration 0014) — which server types this HOST can actually run. The host
    #: sets it once at startup (e.g. from a vllm-sidecar-rendered env), and
    #: the heartbeat emitter publishes it so the queue UI can gate its per-machine
    #: server-type control (an AMD box that can't run the CUDA vllm sidecar
    #: advertises just ``["ollama"]`` → the UI disables vllm there). Default
    #: ``["ollama"]`` (the universal baseline) keeps every other consumer unchanged.
    llm_servers_available: list[str] = field(default_factory=lambda: ["ollama"])

    # ── storage backend selection (sqlite or Postgres) ─────────────────────────
    #: Which relational engine store the dialect seam resolves to:
    #: ``"sqlite"`` (**default** — a daemon-less local file, the friendliest
    #: zero-config default for a reusable library) or ``"pg"`` (Postgres). Both
    #: run the full DAG engine; any other value is rejected by
    #: :func:`canonical_db_backend`.
    #:
    #: **BREAKING (v1.0.0):** the default was ``"pg"`` through v0.x. It is now
    #: ``"sqlite"`` — the ONE intentionally non-byte-compat
    #: default. A Postgres consumer MUST opt in, either with
    #: ``configure(db_backend="pg")`` OR by exporting ``QUEUE_WORKFLOWS_DB_BACKEND=pg``
    #: (the env knob, read here) — the latter is the "one line at startup" that
    #: also reaches the standalone console scripts (``queue-hw-watch``, the
    #: dashboards, …) which have no host ``configure()`` call. Without it, an
    #: ``QUEUE_WORKFLOWS_DB_URL`` pg DSN is read as a SQLite path. Every other default
    #: stays byte-compat.
    db_backend: str = field(default_factory=_default_db_backend)

    # ── node-module resolver (overrides node_module_package when set) ──────────
    #: ``Callable[[str], module]`` — resolve a stored ``node_module`` string to
    #: an imported module exposing ``run(...)``. Default builds from
    #: ``node_module_package`` (see :meth:`resolve_node_module`).
    node_resolver: Callable[[str], Any] | None = None

    # ── builtin-model registrar (empty-registry fallback hook) ─────────────────
    #: ``Callable[[], None]`` — idempotently register the host's ModelSpecs into
    #: the engine ``model_registry``. The empty-registry re-registration
    #: fallback in ``model_cache`` calls it; ``node_pool`` / ``claim_worker``
    #: call it once at startup. Default no-op (standalone engine has no models).
    builtin_model_registrar: Callable[[], None] = lambda: None

    # ── workflow / pipeline provider (DAG source) ──────────────────────────────
    #: ``Callable[[str], dict]`` — load a workflow definition by name.
    workflow_loader: Callable[[str], dict] | None = None
    #: ``Callable[[str], dict]`` — load a pipeline schema by name.
    pipeline_schema_loader: Callable[[str], dict] | None = None
    #: ``Callable[[Any, dict], Any]`` — resolve a ``$from``/``$value``/``$filter``
    #: ref against a context. Defaults to the built-in :func:`refs.resolve_ref`
    #: (wired lazily in ``get_resolve_ref`` to keep this module a leaf).
    resolve_ref: Callable[[Any, dict], Any] | None = None

    # ── per-node invoke wrapper (host setup/teardown around each node run) ──────
    #: ``Callable[[dict, dict], ContextManager[Callable[[dict], dict] | None]]`` —
    #: given ``(job, run)``, returns a context manager wrapping the node invoke.
    #: ``__enter__`` does host setup (e.g. pin a run-context ContextVar, capture a
    #: live flag) and yields a ``finalize(context_delta) -> context_delta`` callable
    #: that ``execute_node`` applies ONLY on success (e.g. stamp a per-node marker);
    #: ``__exit__`` does teardown on EVERY exit path (success / failure / skip).
    #: Default ``None`` ⇒ no wrapping (the engine runs the node directly). Lets a
    #: host thread per-node execution state (e.g. a smoke/mock ``_mocked`` stamp)
    #: without forking ``execute_node``.
    invoke_context: Callable[[dict, dict], Any] | None = None

    # ── vllm sidecar lifecycle (host-provided; idle supervisor + model switch) ─
    #: ``Callable[[], bool]`` — stop the vllm sidecar to free VRAM, returning True
    #: iff it stopped one. The :class:`~queue_workflows.llm_backends.supervisor.\
    #: LLMSupervisor` calls this (via the backend's ``stop_server``) on idle.
    #: ``Callable[[str], None]`` — (re)start the sidecar serving ``model_id``; the
    #: backend's ``ensure_ready`` calls it on a cold start / respawn. Default
    #: ``None`` ⇒ the vllm backend's built-in pkill / no-op seams (a same-container
    #: or unmanaged deployment). A host that runs vllm as a SEPARATE container
    #: wires these (e.g. → docker Engine API over the UDS) so the in-worker
    #: supervisor can stop/start the SIBLING sidecar WITHOUT a docker restart
    #: policy (which would re-trigger the NFS cold-start boot race). Threaded into
    #: ``VLLMBackend`` by the backend factory; ``None`` passes through to the
    #: backend's own default (``kill_fn or _default_kill_fn``).
    vllm_stop_fn: Callable[[], bool] | None = None
    #: Start the box's vLLM server. MAY return what it did — a
    #: ``server_switch.StartOutcome``, a plain ``{"status": ..., "reason": ...}`` dict
    #: (so a host file can report outcomes while still deployed against an older engine
    #: wheel), a bool, or ``None``. ``None`` means "no verdict" and behaves exactly as
    #: before; a ``permanent`` verdict ends the switch's out-of-service window on the
    #: spot instead of at the warm timeout. See ServerSwitch.report_start_outcome.
    vllm_start_fn: Callable[..., Any] | None = None
    #: Serve a different CONTEXT WINDOW on this box (migration 0028).
    #: ``fn(server_kind, context_length | None) -> verdict`` — restart/reconfigure the
    #: box's ``'ollama'`` / ``'vllm'`` server so it serves ``context_length`` tokens, or
    #: the operator's own value when ``None``. MAY return what it did, in the same
    #: vocabulary as ``vllm_start_fn`` (a ``StartOutcome``, a ``{"status", "reason"}``
    #: dict, a bool, ``None``). Default ``None`` ⇒ the engine's built-in lever recreates
    #: the box's own container over the docker socket
    #: (:mod:`queue_workflows.llm_context_lever`); a box with neither lever nor socket
    #: simply advertises the window it already serves and is never asked to switch. Wire
    #: this when the server is not a local container (another host, k8s). See
    #: :func:`queue_workflows.set_llm_context_lever`.
    llm_context_lever_fn: Callable[[str, int | None], Any] | None = None

    # ── inference-server ON/OFF, tied to the GPU worker toggle (host-provided) ─
    #: The operator GPU ON/OFF toggle (worker_controls, migration 0012) governs the
    #: machine's inference SERVER too, not just the worker process: turning the GPU
    #: worker OFF stops the box's LLM server (frees the GPU's VRAM), turning it ON
    #: starts it. These are the host-wired levers. ``stop_fn()`` stops this machine's
    #: server (e.g. ``docker stop ollama`` over the UDS); ``start_fn()`` starts
    #: it. SERVER-TYPE-AGNOSTIC — the host controls whatever it runs (ollama / vllm).
    #: Distinct from ``vllm_{stop,start}_fn`` above (those drive the vllm backend's
    #: IDLE reap + model switch; these track the operator DESIRED state). Both default
    #: ``None`` ⇒ no-op, so a deployment that never wires them is unchanged — the GPU
    #: toggle keeps parking only the worker, exactly as before. The worker calls them
    #: GPU-lane-only + best-effort (a failure never blocks the stop/claim path). See
    #: :func:`queue_workflows.set_inference_server_lifecycle`.
    inference_server_start_fn: Callable[[], None] | None = None
    inference_server_stop_fn: Callable[[], None] | None = None

    # ── ComfyUI box lifecycle (host-provided) ─────────────────────────────────
    #: Bring the box's ComfyUI server up / down. Unlike the evict-only ``comfyui_free``
    #: lever (which just drops its models), these START/STOP the server itself, so
    #: ComfyUI can be a FIRST-CLASS box tenant: when a node_job requests comfyui,
    #: :func:`queue_workflows.comfyui.acquire_box_for_comfyui` clears every rival
    #: serving kind off the card and then calls ``comfyui_start_fn()`` to ensure
    #: ComfyUI holds the box. The host wires these to whatever it runs (e.g.
    #: ``docker start/stop comfyui`` over the UDS). Both default ``None`` ⇒ no-op
    #: (a deployment that never wires them keeps ComfyUI as an already-running,
    #: externally-managed server). See :func:`queue_workflows.set_comfyui_lifecycle`.
    comfyui_start_fn: Callable[[], None] | None = None
    comfyui_stop_fn: Callable[[], None] | None = None

    # ── one-model-per-GPU-box arbitration (opt-in) ────────────────────────────
    #: The PHYSICAL box identity every GPU worker on a machine agrees on, so the
    #: model lease (:mod:`queue_workflows.gpu_model_lease`) unifies per-project
    #: workers that carry DIFFERENT ``host_label``s (``box-a`` vs ``box-a-gpu``).
    #: Empty ⇒ the machine hostname. Env ``QUEUE_WORKFLOWS_GPU_BOX_ID``.
    gpu_box_id: str = ""
    #: PER-BOX CEILING on concurrent GPU node-jobs — ``{box_id: max_concurrent}``,
    #: e.g. ``{"box-b": 3, "box-c": 3}``. The box's computed concurrency (operator
    #: PAR, capped by the vLLM KV cache and by the serving kind) describes the LLM
    #: SERVER; this describes the MACHINE — its power envelope, its unified memory,
    #: the thermal latch a flood of concurrent work trips. Keyed by the PHYSICAL box
    #: identity (:data:`gpu_box_id` / ``gpu_model_lease.default_box_id()``, e.g.
    #: ``box-b``), the same name ``avoid_box``/``force_box`` place against — NOT the
    #: per-project lane label (``box-b-gpu``), because every project's lane on a
    #: machine shares one card.
    #:
    #: Applied as the LAST word in ``claim_worker.ClaimWorker._pool_parallelism()``,
    #: so it can only LOWER the computed number, never raise it. A box not named
    #: here has NO ceiling (today's behaviour, byte-identical). A value that is not
    #: a positive int is IGNORED, never obeyed — a 0 would be a GPU lane that claims
    #: nothing, silently, forever.
    #:
    #: Env twin ``QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT`` (wins) carries the same
    #: mapping as a comma-separated list — ``"box-b=3,box-c=3"`` — so ONE rendered
    #: ``.env`` fits the whole fleet; a bare integer (``"3"``) means "this box",
    #: for a machine whose compose file is its own. Comma+``=`` is the shell-safe
    #: shape for the engine's first mapping knob (every other knob is a scalar).
    gpu_box_max_concurrent: dict[str, int] = field(default_factory=dict)
    #: Directory for the file-backed box model lease (each GPU container mounts the
    #: SAME host dir → a box-wide flock). Unset ⇒ the lease is a no-op that grants
    #: every load (byte-identical to no arbitration). Env
    #: ``QUEUE_WORKFLOWS_GPU_MODEL_LEASE_DIR``.
    gpu_model_lease_dir: str = ""
    #: Lease TTL seconds — a holder renews while it keeps the model warm; a dead
    #: holder's lease lapses so it can't hold VRAM hostage. Env
    #: ``QUEUE_WORKFLOWS_GPU_MODEL_LEASE_TTL_S``.
    gpu_model_lease_ttl_s: float = 120.0
    #: THE idle-model rule, in seconds: an EXTERNAL LLM server's model (ollama)
    #: may stay loaded this long while UNUSED before the gpu worker's idle
    #: reaper unloads it (``llm_idle_unload``); ``<= 0`` disables. "Unused" is
    #: measured at the SERVER (``/api/ps`` ``expires_at`` advances on every
    #: client's use), so host-internal LLM calls count too. Deploy config should
    #: derive the server-side ``OLLAMA_KEEP_ALIVE`` from this same constant so
    #: ollama enforces the identical rule natively. Env twin
    #: ``QUEUE_WORKFLOWS_LLM_SERVER_IDLE_TTL_S`` overrides (envcompat).
    llm_server_idle_ttl_s: float = 300.0
    #: A host-provided lease store (overrides ``gpu_model_lease_dir``) — anything
    #: with ``read(box_id)`` + ``update(box_id, fn)``, e.g. a shared-DB-backed store
    #: shared across boxes. ``None`` ⇒ the file store (if a dir is set) else no-op.
    gpu_lease_store: Any | None = None
    #: ComfyUI base URL on THIS box, if it runs one (a diffusion server that can hold the
    #: GPU alongside/instead of an LLM). Set ⇒ the box-residency probe includes ComfyUI
    #: (``/system_stats`` residency, ``/free`` evict) so the one-server-per-box rule
    #: arbitrates it too. Empty ⇒ ComfyUI is not part of this box's residency picture.
    #: Env ``QUEUE_WORKFLOWS_COMFYUI_URL``.
    comfyui_url: str = ""

    # ── per-dispatch LLM server resolver (host-provided) ──────────────────────
    #: ``Callable[[dict], Any]`` — given a node-job dict (id, node_module, queue,
    #: required_model, host_label, …), return the LLM server the node should call:
    #: a host-defined object carrying the server's TYPE + ADDRESS + PORT (e.g. an
    #: :class:`~queue_workflows.llm_backends.base.LLMBackend` / its ``chat_url``,
    #: or a small ``{"type","url","port"}`` dict). ``node_executor`` resolves this
    #: PER JOB at dispatch and threads it into ``run(llm_server=...)`` — parallel
    #: to ``model_handle`` — so broker_parrot hands each run node-job the exact
    #: server endpoint to use. The resolver body owns selection across the box's
    #: (possibly MULTIPLE) vllm/ollama servers, and — if it consults the DB
    #: heartbeat url advertisement — across boxes. Default ``None`` ⇒ nodes get
    #: ``llm_server=None`` and keep self-resolving from local env (unchanged).
    llm_server_resolver: Callable[[dict], Any] | None = None

    # ── orphan-cancel sweep (ON by default) ──
    #: When True, the :class:`NodePool` periodically flips ``queued`` jobs whose
    #: parent run is already ``cancelled`` / ``failed`` to ``cancelled``. The
    #: host's cancel handler is usually a single ``UPDATE workflow_runs SET
    #: status='cancelled'`` and does not cascade into ``workflow_node_jobs``; the
    #: claim SQL refuses such jobs (run-cancel guard), but they linger in
    #: ``queued`` forever — polluting queue gauges AND (worse) feeding every
    #: queue-depth-driven policy a lie (a "full" gpu queue can be
    #: mostly ghosts). Default ``True``: ghosts now cannot
    #: accumulate. The flip is semantically safe — a job of a terminal run can
    #: never be claimed anyway — and hosts that want the old bookkeeping keep it
    #: via ``configure(cancel_orphan_queued_jobs=False)``.
    cancel_orphan_queued_jobs: bool = True

    #: When ``False``, a gpu worker whose container has NO local CUDA device does
    #: NOT self-park: it is a REMOTE/THIN client (its gpu nodes call an external
    #: inference server over HTTP — e.g. lemonade — and never touch torch.cuda),
    #: so the ``_refuse_blind_gpu`` box-c-blind guard (which assumes the worker
    #: SHOULD own a local device) does not apply. Default ``True`` preserves the
    #: existing fail-closed behaviour for real-passthrough hosts.
    #: Set via ``configure(gpu_requires_local_device=False)``.
    gpu_requires_local_device: bool = True
    #: When ``True``, a gpu worker does NOT start the engine's flocked host
    #: hw-metrics sampler — for a host that publishes hw-metrics from its OWN
    #: sampler (a richer/different payload) and would otherwise double-NOTIFY
    #: ``hw_metrics`` with two shapes. Default ``False`` keeps the engine sampler.
    #: Set via ``configure(disable_gpu_worker_hw_sampler=True)``.
    disable_gpu_worker_hw_sampler: bool = False

    # ── LAN broadcast of hw-metrics (opt-in) ─────────────────────────────────
    #: When ``True``, the host's hw-metrics sampler ALSO emits each sample as a
    #: UDP broadcast on the local network, alongside the Postgres NOTIFY it
    #: already fires. This exists so a LAN appliance (a display panel, say) can
    #: consume telemetry with zero configuration — no DSN, no credentials, no
    #: sidecar re-probing a box the engine already probes — and keeps working
    #: across DHCP changes on either side. It adds no sampling: the very sample
    #: handed to the NOTIFY path is what goes on the wire, in the same envelope.
    #: Default ``False`` — a host must opt in before its telemetry is broadcast.
    #: Set via ``configure(broadcast_hw_stats=True)`` or
    #: ``QUEUE_WORKFLOWS_BROADCAST_HW_STATS=1``. See :mod:`queue_workflows.hw_lan`.
    broadcast_hw_stats: bool = field(default_factory=_default_broadcast_hw_stats)

    #: UDP port the above broadcasts to. Consumers must agree with it.
    #: ``QUEUE_WORKFLOWS_HW_BROADCAST_PORT`` overrides; default 48291.
    hw_broadcast_port: int = field(default_factory=_default_hw_broadcast_port)

    # ── cpu lane width (opt-in in-process concurrency) ────────────────────────
    #: How many node-jobs ONE cpu claim-worker process may run at a time. ``1``
    #: (the default) is the historical concurrency-1 worker: claim, run the body
    #: inline, loop — no executor is created at all. ``N > 1`` gives the cpu lane
    #: an in-process pool of N long-lived worker
    #: threads (``claim_worker._CpuLanePool``) fed by a reserve-slot-then-claim
    #: loop. It deliberately does NOT use a ``ThreadPoolExecutor``: that starts
    #: its threads lazily inside ``submit``, *after* enqueuing the work, so a
    #: refused thread raises while the body still runs — and a slot freed on
    #: that signal breaks the ceiling.
    #:
    #: WHICH TO USE: for a CPU-BOUND pure-Python node the GIL serialises threads,
    #: so N PROCESSES (``deploy/run_claim_lane.sh``) remain the only real
    #: parallelism and stay the recommended default. Raise this instead when the
    #: cpu nodes are IO-BOUND (HTTP, DB waits, an ffmpeg/subprocess that releases
    #: the GIL) or when one container running one process is simply the deploy
    #: you want. Env twin ``QUEUE_WORKFLOWS_CPU_WORKER_THREADS`` overrides
    #: (envcompat); gpu/ingest queues are unaffected — gpu owns its own two-lane
    #: PAR budget.
    cpu_worker_threads: int = 1

    # ── download lane width (fleet-wide per-project cap; migration 0025) ──────
    #: How many ``download`` node-jobs may be RUNNING at once — counted across
    #: the WHOLE fleet for this project, not per host. Downloads are network
    #: WAIT, and the uplink they contend for is shared by every box, so the cap
    #: is claim-side and host-independent (``node_queue.claim_next_download_job``
    #: refuses while the count is at the cap; an advisory lock serializes the
    #: racing claimants). Default **1** = one downloader at a time — the lane's
    #: whole point. Floored at 1 (a 0 cap would be a permanently dead lane).
    #: Env twin ``QUEUE_WORKFLOWS_DOWNLOAD_MAX_CONCURRENT`` overrides (envcompat).
    download_max_concurrent: int = 1

    # ── ingest queue names + budget (host-configurable) ────────────────────
    #: Ingest-family queue names. Migration 0008 dropped the fetch/load DB CHECK;
    #: the host validates ``queue`` against THIS set before enqueue (mirroring the
    #: task_name gate). Default {'fetch','load'}; a
    #: different project (e.g. a forecast service) sets {'ingest','etl','sim','reports'}.
    ingest_queues: frozenset[str] = frozenset({"fetch", "load"})
    #: Wall-clock budget (s) the claim worker applies to ingest queues OTHER than
    #: the built-in fetch/load (``claim_worker.budget_for``). Host-tunable.
    ingest_default_budget_s: int = 3600

    # ── ingest task seam (periodic work) ───────────────────────────────────────
    #: ``task_name -> Callable[[str], dict]`` — the periodic ingest callables the
    #: claim worker runs. The callable takes the ``reason`` string and returns a
    #: JSON-able result dict. Empty ⇒ no ingest work registered.
    ingest_task_map: dict[str, Callable[[str], dict]] = field(default_factory=dict)
    #: The scheduler's periodic schedule (list of ``ScheduleEntry``). Empty ⇒ the
    #: ticker has nothing to fire. Typed ``Any`` here to keep config a leaf (the
    #: ``ScheduleEntry`` type lives in ``scheduler``).
    ingest_schedule: list[Any] = field(default_factory=list)

    # ── lock so configure() from a host thread is safe ─────────────────────────
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ── derived accessors ──────────────────────────────────────────────────────

    @property
    def effective_vlm_pool_node_modules(self) -> frozenset[str]:
        """The host-configured pool modules UNIONED with the engine's built-ins.

        Built-in queue citizens (the ``comfyui`` render node) must pass the gpu
        no-model guard and route to the pool lane for EVERY project, including one
        that sets its own ``vlm_pool_node_modules`` — union, never replace."""
        return frozenset(self.vlm_pool_node_modules) | BUILTIN_POOL_NODE_MODULES

    def resolve_node_module(self, node_module: str):
        """Import + return the module for a stored ``node_module`` value.

        Honours an injected :attr:`node_resolver` first; otherwise builds the
        dotted name from :attr:`node_module_package` (``"<pkg>.<node_module>"``
        when a package is set, else ``node_module`` verbatim) and imports it.
        When that import fails and the name matches an engine BUILT-IN node
        (``queue_workflows.builtin_nodes.<name>``), the builtin is the fallback —
        host nodes always win; built-ins make capabilities like the ComfyUI render
        job work with zero host glue.
        """
        if self.node_resolver is not None:
            return self.node_resolver(node_module)
        import importlib

        dotted = (
            f"{self.node_module_package}.{node_module}"
            if self.node_module_package
            else node_module
        )
        try:
            return importlib.import_module(dotted)
        except ImportError:
            if node_module in BUILTIN_POOL_NODE_MODULES:
                return importlib.import_module(f"queue_workflows.builtin_nodes.{node_module}")
            raise

    def resolve_llm_server(self, job: dict) -> Any:
        """Resolve the LLM server endpoint for ``job`` via the injected
        :attr:`llm_server_resolver`, or ``None`` when no host resolver is wired
        (nodes then keep self-resolving from local env — unchanged behaviour).
        Called once per dispatch by :func:`node_executor.execute_node`; a
        resolver exception propagates to fail the node honestly."""
        if self.llm_server_resolver is None:
            return None
        return self.llm_server_resolver(job)

    def get_resolve_ref(self) -> Callable[[Any, dict], Any]:
        """Return the ref resolver, defaulting to the engine's own
        :func:`queue_workflows.refs.resolve_ref` (imported lazily so this
        config module stays a leaf with no engine-internal imports)."""
        if self.resolve_ref is not None:
            return self.resolve_ref
        from queue_workflows.refs import resolve_ref as _builtin_resolve_ref

        return _builtin_resolve_ref


# Process-wide singleton.
_CONFIG = EngineConfig()


def get_config() -> EngineConfig:
    """Return the process-wide :class:`EngineConfig`."""
    return _CONFIG


def reset_for_tests() -> None:
    """TEST-ONLY. Restore the config to its all-default state so a test that
    mutated a hook doesn't leak into the next."""
    global _CONFIG
    _CONFIG = EngineConfig()
