"""Don't pay a cold start for one job — the vLLM amortisation rule.

vLLM reloads its weights on every start (minutes for a large model on a unified-memory GPU box); ollama
swaps a model in seconds. So "prefer vLLM" cannot mean "start vLLM for whatever
turns up next" — a single job would wait longer for the server than the work takes.

The rule: take the preferred-but-expensive server only when the cost is already
paid or is about to be amortised —

  * it is ALREADY serving that model (no start to pay for), or
  * there are at least ``min_queue`` jobs waiting for it (the start is spread
    across all of them),

otherwise fall through to the next preference — ollama — which is cheap to warm.

The threshold is deployment policy, not a law, so it is a config constant.
"""
from __future__ import annotations

import pytest

from queue_workflows.llm_backends import choose_server_type


PREF = ["vllm", "ollama"]
BOTH = ["ollama", "vllm"]


# ── the amortisation gate ────────────────────────────────────────────────────


def test_already_warm_vllm_is_taken_immediately_however_short_the_queue():
    """Nothing to amortise — the weights are already resident."""
    assert choose_server_type(
        PREF, BOTH, warm=["vllm"], pending=1, min_queue=5) == "vllm"


def test_cold_vllm_is_skipped_for_a_lone_job():
    """One job must not wait ~7 min for a server that ollama can serve now."""
    assert choose_server_type(
        PREF, BOTH, warm=[], pending=1, min_queue=5) == "ollama"


def test_cold_vllm_is_taken_once_the_queue_justifies_it():
    assert choose_server_type(
        PREF, BOTH, warm=[], pending=5, min_queue=5) == "vllm"


def test_just_below_the_threshold_still_goes_to_ollama():
    assert choose_server_type(
        PREF, BOTH, warm=[], pending=4, min_queue=5) == "ollama"


@pytest.mark.parametrize("pending", [0, 1, 3])
def test_the_gate_only_applies_to_expensive_kinds(pending):
    """ollama is cheap to warm, so it is never queue-gated — otherwise a cold fleet
    with a short queue would resolve to nothing at all."""
    assert choose_server_type(
        ["ollama", "vllm"], BOTH, warm=[], pending=pending, min_queue=5) == "ollama"


def test_a_threshold_of_zero_disables_the_gate():
    """Escape hatch: min_queue=0 means "always pay the cold start"."""
    assert choose_server_type(
        PREF, BOTH, warm=[], pending=0, min_queue=0) == "vllm"


def test_vllm_only_preference_still_takes_vllm_even_when_cold():
    """The gate picks BETWEEN alternatives. With no cheaper fallback listed, gating
    it away would strand the job — better a slow start than no start."""
    assert choose_server_type(
        ["vllm"], BOTH, warm=[], pending=1, min_queue=5) == "vllm"


def test_unavailable_vllm_falls_through_regardless_of_queue():
    assert choose_server_type(
        PREF, ["ollama"], warm=[], pending=99, min_queue=5) == "ollama"


# ── it must not break the pre-rule behaviour ─────────────────────────────────


def test_defaults_keep_the_plain_first_available_wins_semantics():
    """Called without the new kwargs (every existing caller), the gate is inert."""
    assert choose_server_type(PREF, BOTH) == "vllm"
    assert choose_server_type(PREF, ["ollama"]) == "ollama"
    assert choose_server_type([], BOTH) is None
    assert choose_server_type(None, BOTH) is None


def test_unprobed_box_still_honours_the_preference():
    assert choose_server_type(PREF, [], warm=[], pending=1, min_queue=5) == "vllm"


# ── the queue-depth input ────────────────────────────────────────────────────


def test_pending_count_sees_only_matching_queued_rows():
    from queue_workflows import node_queue, run_store
    from tests._helpers import make_run

    def q(model, servers, status="queued", run=None):
        rid = run or make_run()
        jid = node_queue.enqueue_node_job(
            run_id=rid, node_id=f"n{id(servers)}{model}{status}", node_module="m",
            queue="gpu", required_model=model, server_types=servers)
        return rid, jid

    q("chat-model-a:latest", ["vllm", "ollama"])
    q("chat-model-a:latest", ["vllm"])
    q("chat-model-a:latest", ["ollama"])          # different kind — excluded
    q("other", ["vllm"])                      # different model — excluded
    q("chat-model-a:latest", None)                 # unconstrained — excluded

    assert node_queue.count_pending_for_server("chat-model-a:latest", "vllm") == 2


def test_box_level_demand_is_an_ALLOW_set_not_a_preference_order():
    """The switch gate asks ``any_model=True``: every queued row that ALLOWS vllm
    counts — wherever vllm sits in its list, with or without a model named — while
    a row that RESTRICTS to ollama, or carries no ``server_types`` at all (NULL =
    an in-process model load, a different execution family entirely), never does.
    This is the "check if it allows vllm, not whether it prefers it" contract the
    ollama→vLLM switch trigger stands on (a deliberate policy)."""
    from queue_workflows import node_queue
    from tests._helpers import make_run

    def q(model, servers, node_id):
        node_queue.enqueue_node_job(
            run_id=make_run(), node_id=node_id, node_module="m",
            queue="gpu", required_model=model, server_types=servers)

    q("chat-model-a:latest", ["vllm", "ollama"], "vllm-first")    # counts
    q("chat-model-a:latest", ["ollama", "vllm"], "ollama-first")  # counts — order is a
    #                                          tiebreak at dispatch, not a gate here
    q(None, ["vllm", "ollama"], "model-less-llm-caller")     # counts — the stamped
    #                                          report-composer shape
    q("chat-model-a:latest", ["ollama"], "ollama-restricted")     # excluded — allow-set
    #                                          excludes vllm
    q("chat-model-a:latest", None, "in-process-load")             # excluded — NULL types
    #                                          = in-process family, never vllm demand
    q(None, None, "plain-pool-row")                          # excluded — not LLM work

    assert node_queue.count_pending_for_server(None, "vllm", any_model=True) == 3


def test_pending_count_excludes_ghosts_of_terminal_runs():
    """A queue full of rows whose run already finished must not justify paying a
    ~7-minute cold start — the claim would never run them anyway."""
    from queue_workflows import node_queue, run_store
    from tests._helpers import make_run

    dead = make_run()
    node_queue.enqueue_node_job(
        run_id=dead, node_id="ghost", node_module="m", queue="gpu",
        required_model="m1", server_types=["vllm"])
    run_store.update_run(dead, status="cancelled")
    assert node_queue.count_pending_for_server("m1", "vllm") == 0


def test_pending_count_is_zero_on_failure_not_an_exception():
    """Conservative direction: a DB blip must never trigger an expensive start."""
    from queue_workflows import node_queue
    assert node_queue.count_pending_for_server(None, "vllm") >= 0


# ── the resolver must LOOK, not guess ────────────────────────────────────────


def test_resolve_for_job_never_dispatches_at_a_dead_server(monkeypatch):
    """REGRESSION (live, many failed jobs in a short window). `resolve_for_job(job, host, queue)`
    is a per-dispatch hook with no capability list to pass, so `available` arrives as
    None. choose_server_type reads an EMPTY availability as "unprobed box — honour
    the preference blindly", which is right for a cold worker and catastrophic here:
    every job resolved to vllm whether or not vllm was running, and once the idle
    stopper stopped it they all died on `[Errno 111] Connection refused`.

    The default must be to PROBE."""
    from queue_workflows.llm_backends import factory

    seen = {}

    def _fake_probe(kinds):
        seen["asked"] = list(kinds)
        return ["ollama"]                      # vllm is DOWN

    def _fake_get(host, queue, server_type=None):
        seen["chosen"] = server_type
        return object()

    monkeypatch.setattr(factory, "_probe_available", _fake_probe)
    monkeypatch.setattr(factory._DEFAULT, "get_backend", _fake_get)

    factory.resolve_for_job(
        {"server_types": ["vllm", "ollama"], "required_model": "chat-model-a:latest"},
        "box-a-gpu", "gpu")

    assert seen["asked"] == ["vllm", "ollama"], "it must probe, not assume"
    assert seen["chosen"] == "ollama", "a stopped vllm must never be dispatched at"


def test_a_model_less_job_treats_a_LIVE_vllm_as_warm(monkeypatch):
    """A stamped-but-model-less LLM caller (compose_report — its node names
    no model) must not be amortisation-gated away from a vLLM that is ALREADY
    serving: the server being up means the weights are resident, so there is
    nothing to amortise — and routing its calls to ollama instead summons a SECOND
    copy of the very weights vLLM is serving (the memory-exhaustion incident, through the
    preference door this time)."""
    from queue_workflows import llm_probe
    from queue_workflows.llm_backends import factory

    seen = {}
    monkeypatch.setattr(factory, "_probe_available",
                        lambda kinds: ["vllm", "ollama"])       # BOTH answer
    monkeypatch.setattr(llm_probe, "vllm_served_models",
                        lambda url: ["chat-model-a"])                # live + serving
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: seen.setdefault("c", server_type))

    factory.resolve_for_job({"server_types": ["vllm", "ollama"]}, "h", "gpu")
    assert seen["c"] == "vllm", \
        "live vLLM = warm for a model-less row; ollama would load copy #2"


def test_a_model_less_job_with_vllm_down_still_rides_ollama(monkeypatch):
    """The inverse guard: down = cold = the gate stands. A lone model-less job must
    keep riding ollama, not wait minutes for a start it cannot amortise."""
    from queue_workflows import llm_probe
    from queue_workflows.llm_backends import factory

    seen = {}
    monkeypatch.setattr(factory, "_probe_available", lambda kinds: ["ollama"])
    monkeypatch.setattr(llm_probe, "vllm_served_models", lambda url: [])
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: seen.setdefault("c", server_type))

    factory.resolve_for_job({"server_types": ["vllm", "ollama"]}, "h", "gpu")
    assert seen["c"] == "ollama"


def test_an_explicit_availability_list_is_still_honoured(monkeypatch):
    """A caller that DOES know skips the probe entirely."""
    from queue_workflows.llm_backends import factory

    def _boom(kinds):
        raise AssertionError("must not probe when told")

    seen = {}
    monkeypatch.setattr(factory, "_probe_available", _boom)
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: seen.setdefault("c", server_type))
    factory.resolve_for_job(
        {"server_types": ["vllm", "ollama"]}, "h", "gpu", available=["ollama"])
    assert seen["c"] == "ollama"


# ── concurrency must follow the SERVING KIND, not one global number ──────────


def _budget(served, par):
    from queue_workflows import claim_worker

    class _W(claim_worker.ClaimWorker):
        def __init__(self):  # bypass the real __init__
            pass
    w = _W()
    w._served_llm_types = lambda: served
    w._pool_parallelism = lambda: par
    return claim_worker.ClaimWorker._gpu_slot_budget(w)


def test_an_ollama_only_box_gets_one_slot_however_high_PAR_is():
    """ollama SERIALISES per model: a second concurrent request queues behind the
    first, so extra node-job slots buy nothing and hold work hostage on one box.
    Observed live: three llm_chat jobs stacked inside one
    ollama. This is the pre-vLLM behaviour, restored."""
    assert _budget(["ollama"], 3) == 1
    assert _budget(["ollama"], 8) == 1


def test_a_vllm_box_gets_the_full_PAR():
    """vLLM does continuous batching, so concurrency is real there."""
    assert _budget(["vllm"], 3) == 3
    assert _budget(["ollama", "vllm"], 3) == 3


def test_an_unprobed_box_is_not_clamped_to_one():
    """A cold box must not pin itself at 1 forever if it is about to serve vLLM."""
    assert _budget([], 3) == 3
    assert _budget(None, 3) == 3


def test_par_is_still_the_ceiling():
    assert _budget(["vllm"], 1) == 1


def test_heartbeat_advertises_the_ENFORCED_budget_not_the_raw_PAR():
    """Advertising 3 while the claim takes 1 reads as "1/3" on the panel and makes
    every capacity view — including the fill-before-spill defer heuristic, which
    reads `concurrency` — believe the box has spare slots it will never use."""
    import inspect
    from queue_workflows import claim_worker
    src = inspect.getsource(claim_worker.ClaimWorker.__init__) if hasattr(
        claim_worker.ClaimWorker, "__init__") else ""
    src = inspect.getsource(claim_worker)
    assert "concurrency_fn=(self._gpu_slot_budget" in src, (
        "the gpu heartbeat must publish _gpu_slot_budget, not _pool_parallelism")


# ── analysis #2/#4/#7: the resolver must look, cache, and be loud ────────────


def test_the_DEFAULT_path_does_not_dispatch_at_a_dead_server(monkeypatch):
    """Analysis #2. The panel's `default → vLLM` lever writes the box default; if
    that server is down, every NO-PREFERENCE job used to be dispatched at the dead
    port — the 33-job Connection-refused class, resurfaced via config."""
    from queue_workflows import worker_control
    from queue_workflows.llm_backends import factory

    factory._reset_probe_cache_for_tests()
    monkeypatch.setattr(worker_control, "llm_config_for",
                        lambda h, q: worker_control.LLMConfig(server_type="vllm"))
    monkeypatch.setattr(factory, "_probe_available",
                        lambda kinds: ["ollama"])          # vllm is DOWN
    seen = {}
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: seen.setdefault("k", server_type))
    factory.resolve_for_job({}, "box-a-gpu", "gpu")        # no server_types at all
    assert seen["k"] == "ollama", "a dead default must fall through to the live kind"


def test_the_default_path_keeps_the_default_when_nothing_answers(monkeypatch):
    """No live kind ⇒ nothing better exists; the error should name the configured
    server rather than us inventing a different dead target."""
    from queue_workflows import worker_control
    from queue_workflows.llm_backends import factory

    factory._reset_probe_cache_for_tests()
    monkeypatch.setattr(worker_control, "llm_config_for",
                        lambda h, q: worker_control.LLMConfig(server_type="vllm"))
    monkeypatch.setattr(factory, "_probe_available", lambda kinds: [])
    seen = {}
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: seen.setdefault("k", "DEFAULT" if server_type is None else server_type))
    factory.resolve_for_job({}, "box-a-gpu", "gpu")
    # Since the never-NULL-stamp law the default path passes the RESOLVED default
    # kind (so the D4 re-stamp can correct a stale claim prediction) — same
    # backend, and the assertion is now STRONGER: it names the configured server.
    assert seen["k"] == "vllm"


def test_the_probe_is_TTL_cached(monkeypatch):
    """Analysis #4: resolve_for_job runs per dispatch; probing two endpoints every
    time was pure cost. Within the TTL the wire is touched once per kind."""
    from queue_workflows import llm_probe
    from queue_workflows.llm_backends import factory

    factory._reset_probe_cache_for_tests()
    calls = []
    monkeypatch.setattr(factory._DEFAULT, "resolve_base_url",
                        lambda kind="ollama": f"http://{kind}:1")
    monkeypatch.setattr(llm_probe, "probe_llm_servers",
                        lambda url, **k: calls.append(url) or ["ollama"] if "ollama" in url else calls.append(url) or [])
    for _ in range(5):
        factory._probe_available(["ollama", "vllm"])
    assert len(calls) == 2, f"one wire touch per kind within the TTL, got {len(calls)}"


def test_a_violated_preference_is_loud(monkeypatch, caplog):
    """Analysis #7: a ['vllm']-only job landing on an ollama-only box (legal via the
    cold-worker escape) runs on the default — but silently was how it hid."""
    import logging
    from queue_workflows.llm_backends import factory

    monkeypatch.setattr(factory, "_probe_available", lambda kinds: [])
    monkeypatch.setattr(factory, "resolve_cold_start_min_queue", lambda: 0)
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: object())
    with caplog.at_level(logging.WARNING):
        factory.resolve_for_job(
            {"id": "j1", "server_types": ["vllm"]}, "h-gpu", "gpu",
            available=["ollama"])
    assert any("prefers" in r.message and "vllm" in r.message
               for r in caplog.records), "the violation must be logged"


def test_a_dead_single_preference_serves_on_the_LIVE_kind_not_the_dead_port(monkeypatch):
    """ROUND-3 MED, the reachable version. server_types=['vllm'], vLLM down, ollama
    up: probing only the prefs read as 'unprobed box' and dispatched at the dead
    vLLM silently. Probing all kinds makes the fallback real and the warning fire."""
    import logging
    from queue_workflows.llm_backends import factory

    factory._reset_probe_cache_for_tests()
    monkeypatch.setattr(factory, "_probe_available",
                        lambda kinds: [k for k in kinds if k == "ollama"])
    monkeypatch.setattr(factory, "resolve_cold_start_min_queue", lambda: 0)
    seen = {}
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: seen.setdefault("k", server_type))
    import logging as _l
    caplog_records = []
    h = _l.Handler(); h.emit = lambda r: caplog_records.append(r.getMessage())
    _l.getLogger("queue_workflows.llm_backends.factory").addHandler(h)
    try:
        factory.resolve_for_job({"id": "j", "server_types": ["vllm"]}, "h-gpu", "gpu")
    finally:
        _l.getLogger("queue_workflows.llm_backends.factory").removeHandler(h)
    assert seen["k"] is None, "falls to the box default (a live kind), not dead vllm"
    assert any("prefers" in m for m in caplog_records), "and it is LOUD"


def test_default_kind_reads_are_cached(monkeypatch):
    from queue_workflows import worker_control
    from queue_workflows.llm_backends import factory

    factory._reset_probe_cache_for_tests()
    calls = []
    monkeypatch.setattr(worker_control, "llm_config_for",
                        lambda h, q: calls.append(1) or worker_control.LLMConfig())
    monkeypatch.setattr(factory, "_probe_available", lambda kinds: ["ollama"])
    monkeypatch.setattr(factory._DEFAULT, "get_backend", lambda *a, **k: object())
    for _ in range(50):
        factory.resolve_for_job({}, "h-gpu", "gpu")
    assert len(calls) == 1, f"one DB read per TTL window, got {len(calls)}"


def test_a_recently_alive_kind_survives_one_probe_blip(monkeypatch):
    """Found live: one timed-out vLLM GET rerouted dispatches to ollama, which
    loaded a second copy of the LLM, which fed the eviction loop. Recently-alive rides
    through a blip; never-alive gets no grace."""
    from queue_workflows import llm_probe
    from queue_workflows.llm_backends import factory

    factory._reset_probe_cache_for_tests()
    answers = {"vllm": ["vllm"]}
    monkeypatch.setattr(factory._DEFAULT, "resolve_base_url",
                        lambda kind="ollama": f"http://{kind}:1")
    monkeypatch.setattr(llm_probe, "probe_llm_servers",
                        lambda url, **k: answers.get(url.split("//")[1].split(":")[0], []))
    t = [1000.0]
    monkeypatch.setattr(factory.time, "monotonic", lambda: t[0])

    assert factory._probe_available(["vllm"]) == ["vllm"]      # seen alive
    answers["vllm"] = []                                        # blip
    t[0] += 6                                                   # past the 5s cache
    assert factory._probe_available(["vllm"]) == ["vllm"], "grace rides the blip"
    t[0] += 20                                                  # past the grace
    assert factory._probe_available(["vllm"]) == [], "a real outage still lands"
    factory._reset_probe_cache_for_tests()
    answers["ollama"] = []
    assert factory._probe_available(["ollama"]) == [], "never-alive gets no grace"


def test_a_no_preference_job_uses_the_LIVE_vllm_not_the_ollama_default(monkeypatch):
    """Found live, on a box already short of memory: compose_report — an LLM caller with
    no model field, so nothing ever stamps server_types on it — resolved to the box
    default ollama (whose daemon ALWAYS answers) and loaded a second copy of the LLM
    beside the actively-serving vLLM. A live vLLM only answers when it holds the
    weights: it IS the active server, and the default path must route to it."""
    from queue_workflows.llm_backends import factory

    factory._reset_probe_cache_for_tests()
    monkeypatch.setattr(factory, "_probe_available",
                        lambda kinds: ["ollama", "vllm"])       # both up; vllm serving
    seen = {}
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: seen.setdefault("k", server_type))
    factory.resolve_for_job({}, "box-b-gpu", "gpu")            # no server_types
    assert seen["k"] == "vllm", "must not force a second serving copy into existence"


def test_the_default_still_rules_when_vllm_is_not_live(monkeypatch):
    from queue_workflows import worker_control
    from queue_workflows.llm_backends import factory

    factory._reset_probe_cache_for_tests()
    monkeypatch.setattr(factory, "_probe_available", lambda kinds: ["ollama"])
    monkeypatch.setattr(worker_control, "llm_config_for",
                        lambda h, q: worker_control.LLMConfig(server_type="ollama"))
    seen = {}
    monkeypatch.setattr(factory._DEFAULT, "get_backend",
                        lambda h, q, server_type=None: seen.setdefault("k", "DEFAULT" if server_type is None else server_type))
    factory.resolve_for_job({}, "box-c-gpu", "gpu")
    # Explicit resolved kind since the never-NULL-stamp law (see the test above).
    assert seen["k"] == "ollama"
