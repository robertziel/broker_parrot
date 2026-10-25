"""The LLM context window — what a bigger one costs, and how big a GPU box can go.

WHY THIS EXISTS. A node-job may declare the window its requests need (migration 0028,
``workflow_node_jobs.context_length``), because the alternative is silent: ollama
truncates an over-long prompt to fit its runner's window and answers HTTP 200. For the
queue to route such a job, every GPU box has to answer two questions in its heartbeat —
what window its LLM server serves NOW (``llm_context_length``, observed), and the
highest one it could switch to (``llm_context_max``) — and "could" has to mean "fits in
the memory this box actually has", checked against the box's free RAM every heartbeat.
This module is that arithmetic: pure functions, no I/O, the same idiom as
``model_catalog`` / ``model_cache``. The worker feeds it observations.

WHAT A BIGGER WINDOW COSTS. Measured on ollama 0.33.2:

  * a standard transformer (qwen2.5:0.5b) — runner size 484 / 702 / 937 MB at a
    4096 / 16384 / 32768 context over 398 MB of weights: 14–18 KB per token at the
    margin, where the textbook f16 KV formula says 12,288 B. The formula runs 15–30 %
    low (compute buffers grow with the context too), hence :data:`FORMULA_SAFETY`.
  * a hybrid model (chat-model-a: full attention every 4th layer) — 18.71 GB runner at a
    262,144 context over 17.74 GB of weights: ~3.7 KB per token actually allocated,
    where the formula says 64 KiB. ~18× too much.

So the OBSERVED cost of a loaded runner — (runner − weights) / window, an average that
bounds the marginal cost from above — always wins, and the formula is only the fallback
for a model no runner currently holds. Its error runs one way: it may refuse a switch
that would have fit, never promise one that will not.

HOW BIG A BOX CAN GO. ollama allocates the whole cache when it loads a runner, so a
bigger window means reloading: the budget is the memory the box has free NOW plus what
the current runner releases, minus an operator reserve; the weights come off the top and
the rest, divided by the per-token cost, is the window. The model's own maximum caps it
— ollama silently clamps a larger ``num_ctx`` to it (a 65536 request on a 32768 model
loaded 32768). vLLM is different: its KV pool is sized by ``gpu_memory_utilization``,
not by ``max_model_len``, so a bigger window costs no new memory at all — it only has to
fit the pool the server already owns (``/metrics`` reports it) and the model's maximum.

A ceiling is never below the window already served: a box that cannot prove more still
serves what it serves.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from queue_workflows.envcompat import env_get

import re
from dataclasses import dataclass

#: Round every ceiling DOWN and every target UP to this many tokens — keeps server
#: configs tidy and a ceiling honest (a partial granule is not promised).
CONTEXT_GRANULE = 1024

#: Headroom on the textbook KV formula; see the module docstring for the measurement.
FORMULA_SAFETY = 1.25

#: Parallel slots the formula assumes when the server's setting is unknown. ollama
#: allocates the cache per slot and auto-selects up to 4; assuming the maximum keeps the
#: fallback in its conservative direction.
FORMULA_DEFAULT_PARALLEL = 4

#: Memory kept free for everything else on the box, on top of the LLM's weights + cache.
#: ``QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB`` overrides.
DEFAULT_RESERVE_MB = 8192

_MiB = 1024 * 1024


# ── reading the model ────────────────────────────────────────────────────────


def _arch(model_info: Mapping[str, Any]) -> str | None:
    arch = model_info.get("general.architecture") if model_info else None
    return str(arch) if arch else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = int(value)
    return value if value > 0 else None


def model_info_max_context(model_info: Mapping[str, Any]) -> int | None:
    """The model's own maximum window, from ollama ``/api/show`` ``model_info``."""
    arch = _arch(model_info)
    if not arch:
        return None
    return _positive_int(model_info.get(f"{arch}.context_length"))


def kv_bytes_per_token_formula(
    model_info: Mapping[str, Any], *,
    parallel: int = FORMULA_DEFAULT_PARALLEL,
    bytes_per_element: float = 2.0,
    safety: float = FORMULA_SAFETY,
) -> int | None:
    """The textbook KV-cache cost of one more token of window, from ``model_info``.

    Counts only the layers that keep a cache: every ``full_attention_interval``-th layer
    of a hybrid model, or the layers whose per-layer ``head_count_kv`` is non-zero when
    the model publishes a list. Head width from ``key_length`` / ``value_length``, or
    ``embedding_length / head_count`` when those are absent. ``None`` when the fields
    are missing or nonsensical — an unknown cost proves nothing."""
    arch = _arch(model_info)
    if not arch:
        return None
    blocks = _positive_int(model_info.get(f"{arch}.block_count"))
    raw_kv = model_info.get(f"{arch}.attention.head_count_kv")
    if blocks is None or raw_kv is None:
        return None
    if isinstance(raw_kv, (list, tuple)):
        kv_heads_total = sum(int(h) for h in raw_kv
                             if isinstance(h, (int, float)) and not isinstance(h, bool)
                             and h > 0)
    else:
        kv_heads = _positive_int(raw_kv)
        if kv_heads is None:
            return None
        interval = _positive_int(model_info.get(f"{arch}.full_attention_interval"))
        layers = blocks // interval if interval else blocks
        kv_heads_total = layers * kv_heads
    if kv_heads_total <= 0:
        return None
    key = _positive_int(model_info.get(f"{arch}.attention.key_length"))
    value = _positive_int(model_info.get(f"{arch}.attention.value_length"))
    if key is None or value is None:
        emb = _positive_int(model_info.get(f"{arch}.embedding_length"))
        heads = _positive_int(model_info.get(f"{arch}.attention.head_count"))
        if emb is None or heads is None:
            return None
        key = key or emb // heads
        value = value or emb // heads
    per_slot = kv_heads_total * (key + value) * bytes_per_element
    return int(per_slot * max(1, int(parallel)) * safety)


def kv_bytes_per_token_observed(
    *, runner_bytes: int | None, weights_bytes: int | None, context_length: int | None,
) -> int | None:
    """What ONE token of window actually cost a loaded runner: ``(runner − weights) /
    window``, rounded up. An average including the fixed buffers, so it bounds the
    marginal cost from above — safe to extrapolate UP with. ``None`` for an observation
    that does not add up (a runner smaller than its own weights, no window)."""
    if runner_bytes is None or weights_bytes is None or not context_length:
        return None
    cache = int(runner_bytes) - int(weights_bytes)
    if cache <= 0 or int(context_length) <= 0:
        return None
    return -(-cache // int(context_length))


# ── how big a box could go ───────────────────────────────────────────────────


def _floor_granule(tokens: int, granule: int) -> int:
    return (int(tokens) // granule) * granule if granule > 0 else int(tokens)


def ollama_ceiling(
    *,
    current: int | None,
    model_max: int | None,
    per_token: int | None,
    weights_bytes: int | None,
    runner_bytes: int,
    available_bytes: int | None,
    reserve_bytes: int,
    granule: int = CONTEXT_GRANULE,
) -> int | None:
    """The highest window an ollama box could reload its model at, within its memory.

    ``runner_bytes`` — what the currently loaded runner of THIS model holds (0 when it is
    not loaded): a reload releases it, so it counts toward the budget. Without the
    numbers to prove a bigger window, the answer is the window already served."""
    if per_token is None or weights_bytes is None or available_bytes is None:
        return current
    budget = int(available_bytes) + int(runner_bytes or 0) - int(reserve_bytes)
    cache_bytes = budget - int(weights_bytes)
    if cache_bytes <= 0 or per_token <= 0:
        return current
    tokens = cache_bytes // int(per_token)
    if model_max is not None and tokens >= int(model_max):
        ceiling = int(model_max)          # the model's own limit — ollama clamps to it
    else:
        ceiling = _floor_granule(tokens, granule)
    # Never below what it serves: an overestimated cost (the formula fallback on a
    # loaded model) must not make a box advertise less than it is demonstrably doing.
    return max(ceiling, int(current or 0)) or current


def vllm_ceiling(
    *, current: int | None, model_max: int | None, kv_capacity_tokens: int | None,
    granule: int = CONTEXT_GRANULE,
) -> int | None:
    """The highest ``max_model_len`` a vLLM box could restart at: its model's maximum,
    bounded by the KV pool it already owns. Either unknown ⇒ the window it serves."""
    if model_max is None or kv_capacity_tokens is None:
        return current
    tokens = min(int(model_max), int(kv_capacity_tokens))
    ceiling = tokens if tokens == int(model_max) else _floor_granule(tokens, granule)
    return max(ceiling, int(current or 0)) or current


# ── what to switch to ────────────────────────────────────────────────────────


def switch_target(
    needs: Iterable[int | None], *, current: int | None, ceiling: int | None,
    granule: int = CONTEXT_GRANULE,
) -> int | None:
    """The window to restart this box's server at, or ``None`` for "no switch".

    The highest queued need this box can reach (``<= ceiling``) that it does not
    already serve (``> current``), rounded up to a granule but never past the ceiling.
    Needs beyond the ceiling belong to another box — or, if none can, to the
    orchestrator's unassignable tag. No queued need at all means "keep the current
    default"."""
    if ceiling is None:
        return None
    floor = int(current or 0)
    reachable = [int(n) for n in needs
                 if n is not None and floor < int(n) <= int(ceiling)]
    if not reachable:
        return None
    target = int(math.ceil(max(reachable) / granule) * granule) if granule > 0 \
        else max(reachable)
    return min(target, int(ceiling))


# ── the box ──────────────────────────────────────────────────────────────────


def mem_available_bytes(meminfo_path: str = "/proc/meminfo") -> int | None:
    """``MemAvailable`` in bytes — the kernel's own estimate of what can be allocated
    without swapping. The pool that matters on a unified-memory box, where the LLM's
    cache lives in system RAM. ``None`` when it cannot be read."""
    try:
        with open(meminfo_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


#: vLLM's own default for ``--gpu-memory-utilization`` when the flag is absent.
DEFAULT_VLLM_UTILIZATION = 0.90


def mem_total_bytes(meminfo_path: str = "/proc/meminfo") -> int | None:
    """``MemTotal`` in bytes — on a unified-memory box, also the "GPU memory" vLLM
    takes its ``--gpu-memory-utilization`` fraction of. ``None`` when unreadable."""
    try:
        with open(meminfo_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def load_fits(
    *, weights_bytes: int | None, kv_bytes: int | None, available_bytes: int | None,
    reserve_bytes: int,
) -> bool | None:
    """Would loading a model of ``weights_bytes`` plus its ``kv_bytes`` of cache leave
    ``reserve_bytes`` of the box's free memory untouched? ``None`` when the box cannot
    say — a size or a reading it does not have. The 0015 convention on unknowns: only a
    KNOWN misfit is refused, because a box that cannot describe itself must not wedge
    the queue; the dangerous case is the one with numbers, and it has them."""
    if weights_bytes is None or available_bytes is None:
        return None
    need = int(weights_bytes) + int(kv_bytes or 0)
    return need <= int(available_bytes) - int(reserve_bytes or 0)


def vllm_utilization(
    *, configured: float | None, total_bytes: int | None, available_bytes: int | None,
    reserve_bytes: int, granule: float = 0.01,
) -> float | None:
    """The ``--gpu-memory-utilization`` to START vLLM with: the operator's value, lowered
    to what the box actually has free.

    On a unified-memory box "GPU memory" is the whole machine, so 0.75 of it is a pool
    the OS, the claim lanes and every other container then have to live around. Never
    ABOVE the configured value (that is the operator's ceiling); ``None`` when the
    readings are missing, meaning "start as configured"."""
    if total_bytes is None or available_bytes is None or int(total_bytes) <= 0:
        return None
    fitted = (int(available_bytes) - int(reserve_bytes or 0)) / int(total_bytes)
    fitted = math.floor(max(fitted, 0.0) / granule) * granule
    base = float(configured) if configured is not None else DEFAULT_VLLM_UTILIZATION
    return round(min(base, fitted), 2)


# ── vLLM sized to what it needs ──────────────────────────────────────────────
#
# At startup vLLM states, in its own log, what the broker needs to know about THIS model
# on THIS box. For example, at --gpu-memory-utilization 0.42, --max-num-seqs 8:
#
#     Model loading took 18.74 GiB memory and 315.5 seconds
#     Available KV cache memory: 42.23 GiB
#     GPU KV cache size: 684,056 tokens
#     Maximum concurrency for 262,144 tokens per request: 2.61x
#
# The weights cost 18.7 GB, the cache 66 KB per token, and at that pool the box holds
# 2.6 full-window requests — while the broker fed it 8 slots, and a pool sized to "what
# is free" would have taken most of the box's memory. From these four lines a plan sizes the pool to what
# N requests of the window NEED, and lowers N to what the box can afford.

#: Every refusal that is about MEMORY starts with this, so the memory sentinel can tell a
#: hold it may lift (RAM came back) from one it may not (a server that never came up).
MEMORY_REFUSAL = "memory: "

#: Headroom over the computed need: vLLM's own accounting shifts a little between
#: starts (CUDA-graph capture, the profiling pass) and the pool is a 0.01 fraction.
VLLM_PLAN_MARGIN = 0.05

_RE_WEIGHTS = re.compile(r"Model loading took ([\d.]+) GiB memory")
_RE_KV_BYTES = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_RE_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_RE_CONCURRENCY = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_RE_UTILIZATION = re.compile(r"--gpu-memory-utilization=([\d.]+) is equivalent")


@dataclass(frozen=True)
class VllmProfile:
    """What ONE start of vLLM cost on this box, as vLLM itself reported it."""

    weights_bytes: int
    kv_bytes: int
    kv_tokens: int
    window: int
    #: The pool fraction that start ran at, and the box it ran on — so the fixed
    #: overhead (what the pool held beyond weights and cache) can be derived.
    utilization: float
    total_bytes: int

    @property
    def per_token_bytes(self) -> float:
        return self.kv_bytes / self.kv_tokens if self.kv_tokens else 0.0

    @property
    def pool_bytes(self) -> int:
        return int(self.utilization * self.total_bytes)

    @property
    def overhead_bytes(self) -> int:
        """The start's fixed cost beyond weights and cache (activations, CUDA graphs,
        non-torch allocations). vLLM's accounting on a unified-memory box does not
        always add up to the pool; never negative."""
        return max(0, self.pool_bytes - self.weights_bytes - self.kv_bytes)

    @property
    def concurrency(self) -> float:
        return self.kv_tokens / self.window if self.window else 0.0

    def as_record(self, *, fingerprint: str | None = None) -> dict:
        return {"weights_bytes": int(self.weights_bytes), "kv_bytes": int(self.kv_bytes),
                "kv_tokens": int(self.kv_tokens), "window": int(self.window),
                "utilization": float(self.utilization), "total_bytes": int(self.total_bytes),
                "fingerprint": fingerprint or ""}

    @classmethod
    def from_record(cls, record: Mapping[str, Any] | None) -> "VllmProfile | None":
        if not isinstance(record, Mapping):
            return None
        try:
            return cls(weights_bytes=int(record["weights_bytes"]), kv_bytes=int(record["kv_bytes"]),
                       kv_tokens=int(record["kv_tokens"]), window=int(record["window"]),
                       utilization=float(record["utilization"]),
                       total_bytes=int(record["total_bytes"]))
        except (KeyError, TypeError, ValueError):
            return None


def parse_vllm_startup(
    text: str | None, *, total_bytes: int | None, utilization: float | None = None,
) -> VllmProfile | None:
    """The profile of the LATEST COMPLETE start in a vLLM container's log. A log keeps
    every start of the container and the last may still be loading, so the four lines
    of one start are taken together — split at each "Model loading took" — never mixed
    across starts. The pool fraction is the one vLLM prints beside its CUDA-graph note,
    else the caller's; without either the overhead cannot be derived and there is no
    profile."""
    if not text or not total_bytes:
        return None
    starts = _RE_WEIGHTS.split(text)
    # split() yields [prefix, weights1, rest1, weights2, rest2, …]
    for i in range(len(starts) - 1, 0, -2):
        weights, rest = starts[i - 1], starts[i]
        kv = _RE_KV_BYTES.search(rest)
        tokens = _RE_KV_TOKENS.search(rest)
        conc = _RE_CONCURRENCY.search(rest)
        if not (kv and tokens and conc):
            continue
        util_m = _RE_UTILIZATION.search(rest)
        util = float(util_m.group(1)) if util_m else utilization
        if util is None:
            return None
        return VllmProfile(
            weights_bytes=int(float(weights) * 2**30), kv_bytes=int(float(kv.group(1)) * 2**30),
            kv_tokens=int(tokens.group(1).replace(",", "")),
            window=int(conc.group(1).replace(",", "")), utilization=float(util),
            total_bytes=int(total_bytes))
    return None


def vllm_concurrency(*, kv_tokens: int | None, window: int | None) -> int | None:
    """How many full-window requests the cache holds at once, floored at one — the
    honest concurrency of a vLLM box, whatever ``--max-num-seqs`` says."""
    if not kv_tokens or not window:
        return None
    return max(1, int(kv_tokens) // int(window))


@dataclass(frozen=True)
class VllmPlan:
    """How to start vLLM here: the pool (``None`` = as configured), the concurrency
    (``None`` = as configured), or the refusal that says it must not start."""

    utilization: float | None
    max_num_seqs: int | None
    refusal: str | None
    reason: str = ""


def _gb(n: float) -> str:
    return f"{n / 2**30:.1f} GB"


def vllm_plan(
    *, total_bytes: int | None, available_bytes: int | None, reserve_bytes: int,
    operator_utilization: float | None, operator_max_seqs: int | None, window: int | None,
    profile: VllmProfile | None, min_utilization: float, granule: float = 0.01,
    margin: float = VLLM_PLAN_MARGIN,
) -> VllmPlan:
    """The start plan for vLLM on this box.

    WITH a profile of this model: the pool is what ``operator_max_seqs`` requests of
    ``window`` tokens NEED — weights + the start's fixed overhead + N × window ×
    per-token — plus ``margin``, and not a byte of the operator's ceiling beyond that;
    when the usable memory (what is free minus the reserve, capped by the operator's
    pool) cannot hold that many, N is what it CAN hold, so vLLM is started with a
    ``--max-num-seqs`` it can honour instead of preempting. A box that cannot hold ONE
    request of the window refuses, with the numbers.

    WITHOUT one (the first start of a model on a box): the operator's pool lowered to
    what is free (:func:`vllm_utilization`), refused below ``min_utilization``,
    concurrency untouched — that start's own log is the profile for the next."""
    if total_bytes is None or available_bytes is None or int(total_bytes) <= 0:
        return VllmPlan(None, None, None, "no memory readings — starting as configured")
    total = int(total_bytes)
    op_util = (float(operator_utilization) if operator_utilization is not None
               else DEFAULT_VLLM_UTILIZATION)
    fitted = vllm_utilization(configured=op_util, total_bytes=total,
                              available_bytes=available_bytes, reserve_bytes=reserve_bytes)
    usable_profile = profile is not None and profile.per_token_bytes > 0
    if not usable_profile or not window:
        # today's rule, plus "size only the pool" when the window is unknown — or when
        # the profile cannot price a token (a zero-token cache is no measurement)
        if fitted is None:
            return VllmPlan(None, None, None, "no memory readings — starting as configured")
        if fitted < float(min_utilization):
            return VllmPlan(None, None, (
                f"{MEMORY_REFUSAL}vLLM cannot fit on this box: {_gb(available_bytes)} available of "
                f"{_gb(total)} with {_gb(reserve_bytes)} reserved for everything else — a "
                f"pool of {fitted:.2f} is below the {float(min_utilization):.2f} minimum "
                f"(QUEUE_WORKFLOWS_VLLM_MIN_UTILIZATION)"), "no profile of this model here")
        if fitted + 0.005 >= op_util:
            return VllmPlan(None, None, None, "the operator's pool fits as it is (no profile)")
        return VllmPlan(fitted, None, None,
                        f"pool lowered to {fitted:.2f} for what is free (no profile of this "
                        f"model here yet — this start's log becomes one)")

    assert profile is not None
    budget = int(available_bytes) - int(reserve_bytes or 0)
    cap = min(budget, int(op_util * total))
    fixed = profile.weights_bytes + profile.overhead_bytes
    per_seq = int(window) * profile.per_token_bytes
    one = (fixed + per_seq) * (1.0 + margin)
    if one > cap:
        return VllmPlan(None, None, (
            f"{MEMORY_REFUSAL}vLLM cannot hold one {int(window):,}-token request on this box: weights "
            f"{_gb(profile.weights_bytes)} + overhead {_gb(profile.overhead_bytes)} + cache "
            f"{_gb(per_seq)} (+{margin:.0%}) = {_gb(one)}, {_gb(cap)} usable "
            f"({_gb(available_bytes)} available, {_gb(reserve_bytes)} reserved, operator "
            f"pool {op_util:.2f} = {_gb(op_util * total)}) — lower the window or free memory"),
            "one request does not fit")
    n_max = max(1, int(operator_max_seqs or 1))
    n = min(n_max, int((cap / (1.0 + margin) - fixed) // per_seq))
    n = max(1, n)
    pool = (fixed + n * per_seq) * (1.0 + margin)
    util = math.ceil(pool / total / granule) * granule
    util = round(min(util, op_util, fitted if fitted is not None else op_util), 2)
    reason = (f"sized to need: {n} × {int(window):,} tokens × "
              f"{profile.per_token_bytes / 1024:.1f} KB + weights {_gb(profile.weights_bytes)} "
              f"+ overhead {_gb(profile.overhead_bytes)} (+{margin:.0%}) = {_gb(pool)} → pool "
              f"{util:.2f} of {_gb(total)} ({_gb(cap)} usable)")
    if n < n_max:
        reason += f"; concurrency lowered from {n_max} to {n} — that is what fits"
    return VllmPlan(util, n, None, reason)


def reserve_bytes() -> int:
    """The memory to keep free beyond the LLM (``QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB``,
    default :data:`DEFAULT_RESERVE_MB`). A malformed value falls back to the default."""
    raw = env_get("QUEUE_WORKFLOWS_LLM_CONTEXT_RESERVE_MB")
    try:
        mb = int(raw) if raw else DEFAULT_RESERVE_MB
    except (TypeError, ValueError):
        mb = DEFAULT_RESERVE_MB
    return max(0, mb) * _MiB


__all__ = [
    "CONTEXT_GRANULE", "FORMULA_SAFETY", "FORMULA_DEFAULT_PARALLEL", "DEFAULT_RESERVE_MB",
    "model_info_max_context", "kv_bytes_per_token_formula", "kv_bytes_per_token_observed",
    "ollama_ceiling", "vllm_ceiling", "switch_target", "mem_available_bytes",
    "mem_total_bytes", "load_fits", "vllm_utilization", "DEFAULT_VLLM_UTILIZATION",
    "reserve_bytes",
]
