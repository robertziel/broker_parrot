"""The gpu heartbeat publishes what is true of the box NOW — or says it does not know.

Two fields could be false in a way the fleet then acted on. Both are observations again.

**The window.** A box remembers the context window it last published, so a restart while
its model happens to be unloaded does not leave it windowless (the 0028 idle-box trap).
The memory was read back per ``(host, queue)`` and adopted as OLLAMA's window — but the
number may have been vLLM's, from a box that has since stopped it. ollama 0.33 chooses
its own default from VRAM ("4k/32k/256k based on VRAM"), so a borrowed number can be
wrong by 64×, and a sized job sent to that box is silently truncated — exactly what 0028
exists to prevent. A remembered window is now adopted only when the same kind was serving
when it was published.

**The model.** ``current_model`` is the worker's warm-model slot. For a model served by a
SERVER (``ModelSpec.served_by`` non-empty — the LLM facade) that slot is not residency:
ollama's five-minute keep-alive, or another lane's load, evicts the model while the slot
stands. The box then advertises a resident it does not hold and the panel warns of two
models contending for one GPU. A facade model is now published only while the box's
servers are observed to still hold it; an in-process model is published as before,
because there the cache IS the observation.
"""
from __future__ import annotations

import pytest

from queue_workflows import claim_worker, model_registry, node_queue, worker_control

OLLAMA = worker_control.SERVER_TYPE_OLLAMA
VLLM = worker_control.SERVER_TYPE_VLLM


# ── 1. the remembered window belongs to the kind that published it ───────────


def _worker(monkeypatch, published):
    """A bare gpu worker whose only DB read is the heartbeat read-back."""

    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"

    w = _W()
    seen: list = []

    def _read(host, queue="gpu", project=None, served_by=None):
        seen.append(served_by)
        window, servers = published
        if served_by is not None and served_by not in servers:
            return None
        return window

    monkeypatch.setattr(claim_worker.node_queue, "last_published_context", _read)
    return w, seen


def test_a_window_published_while_only_vllm_served_is_not_ollamas(monkeypatch):
    """The shape: a box served vLLM at 262144, its vLLM stopped, its worker
    restarted, no ollama runner loaded. It has never observed its own ollama — and
    ollama's default is chosen from VRAM, not from what vLLM was doing."""
    w, _seen = _worker(monkeypatch, (262_144, [VLLM]))
    assert w._seeded_window(OLLAMA) is None


def test_a_window_published_while_ollama_served_is_still_remembered(monkeypatch):
    """The 0028 idle-box trap stays fixed: this IS ollama's own observation, one
    process ago."""
    w, _seen = _worker(monkeypatch, (32_768, [OLLAMA]))
    assert w._seeded_window(OLLAMA) == 32_768
    assert w._ctx_last_window[OLLAMA] == 32_768


def test_a_window_published_while_both_served_is_remembered(monkeypatch):
    """With two kinds serving, the published number is the SMALLEST either offered —
    so ollama's real window is at least that. Adopting it cannot overstate."""
    w, _seen = _worker(monkeypatch, (8_192, [OLLAMA, VLLM]))
    assert w._seeded_window(OLLAMA) == 8_192


def test_the_read_back_asks_for_ollamas_own_publications(monkeypatch):
    """The filter is the QUERY's job, not a second guess in the caller."""
    w, seen = _worker(monkeypatch, (262_144, [VLLM]))
    w._seeded_window(OLLAMA)
    assert seen == [OLLAMA]


def test_it_is_still_read_back_once_per_process(monkeypatch):
    w, seen = _worker(monkeypatch, (16_384, [OLLAMA]))
    assert w._seeded_window(OLLAMA) == 16_384
    assert w._seeded_window(OLLAMA) is None, "asked once; the answer cannot change"
    assert len(seen) == 1


def test_a_stopped_vllm_still_remembers_nothing(monkeypatch):
    """Unchanged: a vLLM that is not running serves nothing to remember."""
    w, seen = _worker(monkeypatch, (262_144, [VLLM]))
    assert w._seeded_window(VLLM) is None
    assert seen == [], "the vLLM path never reads the row at all"


def test_a_read_that_fails_leaves_the_box_saying_unknown(monkeypatch):
    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"

    w = _W()

    def _boom(*a, **k):
        raise RuntimeError("db blip")

    monkeypatch.setattr(claim_worker.node_queue, "last_published_context", _boom)
    assert w._seeded_window(OLLAMA) is None


# ── 2. current_model is residency, not a slot that outlived it ───────────────


@pytest.fixture
def capture(monkeypatch):
    box = {"kwargs": None}
    monkeypatch.setattr(claim_worker.node_queue, "upsert_worker_heartbeat",
                        lambda **kw: box.__setitem__("kwargs", kw))
    monkeypatch.setattr(model_registry, "known_ids", lambda: [])
    monkeypatch.setattr(claim_worker, "get_config",
                        lambda: type("C", (), {"llm_servers_available": [], "project": ""})())
    return box


class _Cache:
    def __init__(self, model):
        self.current_model = model


def _emitter(monkeypatch, *, model, residents, served_by=(OLLAMA,)):
    """A gpu heartbeat whose cache holds ``model`` and whose box holds ``residents``."""
    monkeypatch.setattr(model_registry, "server_served_ids",
                        lambda: {"chat-model-a"} if served_by else set())
    em = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-b-gpu", model_cache=_Cache(model),
        residents_fn=(None if residents is _UNSET else (lambda: residents)))
    monkeypatch.setattr(em, "_vram_total_mb", lambda: None)
    return em


_UNSET = object()


def test_a_facade_model_the_box_no_longer_holds_is_not_published(monkeypatch, capture):
    """ollama's keep-alive expired; the slot stands; the model is gone. Publishing it
    made the panel warn of two models contending for one GPU."""
    em = _emitter(monkeypatch, model="chat-model-a", residents=[])
    em.emit_once()
    assert capture["kwargs"]["current_model"] is None


def test_a_facade_model_the_box_still_holds_is_published(monkeypatch, capture):
    em = _emitter(monkeypatch, model="chat-model-a", residents=["chat-model-a:latest"])
    em.emit_once()
    assert capture["kwargs"]["current_model"] == "chat-model-a"


def test_the_comparison_is_canonical_not_a_string_match(monkeypatch, capture):
    """``chat-model-a`` and ``chat-model-a:latest`` are one model — the engine's own
    rule (``model_catalog.canonical_tag``), which the claim already keys on."""
    em = _emitter(monkeypatch, model="Chat-Model-A:latest", residents=["chat-model-a"])
    em.emit_once()
    assert capture["kwargs"]["current_model"] == "Chat-Model-A:latest"


def test_an_in_process_model_needs_no_server_to_vouch_for_it(monkeypatch, capture):
    """A model the worker loads itself: the cache holds the weights, so the cache IS
    the observation. A render's model must not be blanked because no LLM server lists
    it."""
    em = _emitter(monkeypatch, model="chat-model-a", residents=[], served_by=())
    em.emit_once()
    assert capture["kwargs"]["current_model"] == "chat-model-a"


def test_no_observation_at_all_leaves_the_old_behaviour(monkeypatch, capture):
    """Ignorance is not evidence of absence: a box that could not probe publishes what
    it always did rather than blanking a live model."""
    em = _emitter(monkeypatch, model="chat-model-a", residents=_UNSET)
    em.emit_once()
    assert capture["kwargs"]["current_model"] == "chat-model-a"


def test_a_probe_that_raises_leaves_the_old_behaviour(monkeypatch, capture):
    def _boom():
        raise RuntimeError("docker socket hiccup")

    monkeypatch.setattr(model_registry, "server_served_ids", lambda: {"chat-model-a"})
    em = claim_worker.HeartbeatEmitter(queue="gpu", host_label="box-b-gpu",
                                       model_cache=_Cache("chat-model-a"),
                                       residents_fn=_boom)
    monkeypatch.setattr(em, "_vram_total_mb", lambda: None)
    em.emit_once()
    assert capture["kwargs"]["current_model"] == "chat-model-a"


def test_the_lease_OCCUPANT_is_judged_by_residency_too(monkeypatch, capture):
    """This path, unguarded, defeats the whole guard.

    A real gpu worker ALWAYS wires ``occupant_fn``, and a facade LLM job holds the box
    lease under the model id rather than the LLM slot — so the occupant fallback
    republished exactly the stale model ``_box_still_holds`` had just refused, and the
    renewal that keeps the lease alive asks the same warm slot. The occupant is a claim
    about the same card, so it answers to the same evidence."""
    monkeypatch.setattr(model_registry, "server_served_ids", lambda: {"chat-model-a"})
    em = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-b-gpu", model_cache=_Cache(None),
        occupant_fn=lambda: "chat-model-a", residents_fn=lambda: [])
    monkeypatch.setattr(em, "_vram_total_mb", lambda: None)
    em.emit_once()
    assert capture["kwargs"]["current_model"] is None


def test_an_occupant_the_box_DOES_hold_is_still_reported(monkeypatch, capture):
    monkeypatch.setattr(model_registry, "server_served_ids", lambda: {"chat-model-a"})
    em = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-b-gpu", model_cache=_Cache(None),
        occupant_fn=lambda: "chat-model-a", residents_fn=lambda: ["chat-model-a:latest"])
    monkeypatch.setattr(em, "_vram_total_mb", lambda: None)
    em.emit_once()
    assert capture["kwargs"]["current_model"] == "chat-model-a"


def test_an_in_process_occupant_needs_no_server(monkeypatch, capture):
    """A render's own model holds the lease and no LLM server will ever list it."""
    monkeypatch.setattr(model_registry, "server_served_ids", lambda: set())
    em = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-b-gpu", model_cache=_Cache(None),
        occupant_fn=lambda: "wan-s2v", residents_fn=lambda: [])
    monkeypatch.setattr(em, "_vram_total_mb", lambda: None)
    em.emit_once()
    assert capture["kwargs"]["current_model"] == "wan-s2v"


def test_an_observation_that_has_not_happened_yet_is_not_absence(monkeypatch, capture):
    """A documented shape that is easy to leave untested: the seam is WIRED but the probe
    has not succeeded yet, so it returns None — which must read as "I do not know",
    never as "the box holds nothing"."""
    em = _emitter(monkeypatch, model="chat-model-a", residents=None)
    em.emit_once()
    assert capture["kwargs"]["current_model"] == "chat-model-a"


def test_an_artifact_tag_vouches_for_the_role_it_serves(monkeypatch, capture):
    """0027: a role tag (`chat-expert-a`) is served by an immutable artifact
    (`chat-expert-a-v1-q4-ollama`), and this box has BOTH installed. The slot names the
    role; `/api/ps` reports the artifact. Blanking the role because the spellings differ
    is the same mistake canonical_tag exists to prevent, one layer out."""
    monkeypatch.setattr(model_registry, "server_served_ids", lambda: {"chat-expert-a"})
    em = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-b-gpu", model_cache=_Cache("chat-expert-a"),
        residents_fn=lambda: ["chat-expert-a-v1-q4-ollama:latest"])
    monkeypatch.setattr(em, "_vram_total_mb", lambda: None)
    em.emit_once()
    assert capture["kwargs"]["current_model"] == "chat-expert-a"


# ── the observation is the one the worker already makes ──────────────────────


def test_residency_comes_from_the_shared_collector_not_a_new_probe(monkeypatch):
    """DRY, and the reason: `_refresh_box_llm_state` already runs `probe_box_residents`
    on the heartbeat thread each tick, and is what the residency enforcer and the load
    gate read. A parallel observation of the same endpoints cost a second GET of
    /v1/models per tick. (An earlier version of this docstring claimed the collector
    keeps its last answer when a probe fails; it does not — see
    `test_a_timed_out_endpoint_does_not_erase_what_the_box_holds`, which is why absence
    is debounced rather than trusted.)"""
    from queue_workflows import model_residency

    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
            self.model_cache = _Cache("chat-model-a")

    w = _W()
    assert w._observed_residents() is None, "nothing observed yet ⇒ unknown"
    R = model_residency.Resident
    monkeypatch.setattr(model_residency, "probe_box_residents", lambda **k: [
        R(server="ollama", model="held-by-ollama:latest", mru=1.0),
        R(server="vllm", model="held-by-vllm", mru=0.0),
        R(server="inprocess", model="chat-model-a"),
    ])
    w._refresh_box_llm_state()
    seen = w._observed_residents()
    assert seen == {"held-by-ollama:latest", "held-by-vllm"}
    assert "chat-model-a" not in seen, (
        "the in-process leg is the worker's own warm slot — the very claim under test. "
        "Letting it vouch for itself makes the guard unfalsifiable.")


def test_a_failed_probe_keeps_the_last_residency(monkeypatch):
    from queue_workflows import model_residency

    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
            self.model_cache = _Cache(None)

    w = _W()
    R = model_residency.Resident
    monkeypatch.setattr(model_residency, "probe_box_residents",
                        lambda **k: [R(server="ollama", model="m:latest", mru=1.0)])
    w._refresh_box_llm_state()
    assert w._observed_residents() == {"m:latest"}

    def _boom(**k):
        raise RuntimeError("endpoint timed out")

    monkeypatch.setattr(model_residency, "probe_box_residents", _boom)
    w._refresh_box_llm_state()
    assert w._observed_residents() == {"m:latest"}, (
        "one flaky probe must not un-warm the box — the rule this file already "
        "applies to the served-set and the warm snapshot")


# ── 3. the observation the gpu worker hands over ─────────────────────────────


def test_the_heartbeat_is_wired_to_it():
    """Only the gpu lane: a cpu/ingest worker observes no models and must keep the
    emitter's default behaviour byte-for-byte."""
    import inspect

    src = inspect.getsource(claim_worker.ClaimWorker)
    line = next(l for l in src.splitlines() if "residents_fn=" in l)
    assert "_observed_residents" in line and 'self.queue == "gpu"' in line, line


# ── 4. the read-back, against a real database ────────────────────────────────
#
# Every test above stubs `last_published_context` out, so neutering the
# `served_by` filter — or the decoder it leans on — would change nothing there. These go
# through the real column on whichever backend the suite is running.


def _publish(host, *, window, servers):
    node_queue.upsert_worker_heartbeat(
        host_label=host, queue="gpu", concurrency=1, current_model=None,
        known_models=[], llm_servers_available=list(servers),
        vram_total_mb=None, fits_models=[], llm_context_length=window,
        llm_context_max=window)


def test_a_real_row_published_by_vllm_alone_is_not_ollamas_window():
    _publish("box-vllm-only-gpu", window=262_144, servers=["vllm"])
    assert node_queue.last_published_context("box-vllm-only-gpu", "gpu") == 262_144
    assert node_queue.last_published_context(
        "box-vllm-only-gpu", "gpu", served_by=OLLAMA) is None
    assert node_queue.last_published_context(
        "box-vllm-only-gpu", "gpu", served_by=VLLM) == 262_144


def test_a_real_row_published_while_ollama_served_is_returned():
    _publish("box-ollama-gpu", window=32_768, servers=["ollama"])
    assert node_queue.last_published_context(
        "box-ollama-gpu", "gpu", served_by=OLLAMA) == 32_768


def test_a_real_row_with_both_kinds_answers_for_either():
    _publish("box-both-gpu", window=8_192, servers=["ollama", "vllm"])
    for kind in (OLLAMA, VLLM):
        assert node_queue.last_published_context("box-both-gpu", "gpu", served_by=kind) == 8_192


def test_a_box_that_never_published_a_window_says_nothing():
    _publish("box-blank-gpu", window=None, servers=["ollama"])
    assert node_queue.last_published_context(
        "box-blank-gpu", "gpu", served_by=OLLAMA) is None


def test_the_decoder_reads_what_either_backend_stores():
    """`llm_servers_available` comes back decoded from both backends; the string branch
    is defensive. It named the wrong module and could only raise — and the caller
    swallows exceptions into "no remembered window", so the 0028 trap would have
    reopened silently on any caller that read the column raw."""
    assert node_queue._as_server_list(["ollama", "vllm"]) == ["ollama", "vllm"]
    assert node_queue._as_server_list(("ollama",)) == ["ollama"]
    assert node_queue._as_server_list('["ollama","vllm"]') == ["ollama", "vllm"]
    assert node_queue._as_server_list(b'["ollama"]') == ["ollama"]
    assert node_queue._as_server_list("{ollama,vllm}") == []      # a raw pg literal
    assert node_queue._as_server_list("not json") == []
    assert node_queue._as_server_list(None) == []
    assert node_queue._as_server_list([]) == []
    assert node_queue._as_server_list(["ollama", None, ""]) == ["ollama"]


def test_the_window_observation_makes_no_extra_call_for_residency(monkeypatch):
    """Residency must add "no new probe" — and an earlier version did add one: the vLLM
    branch called `vllm_served_models` beside `vllm_max_model_len`, and both GET
    /v1/models with their own urlopen — a second request every tick on every vLLM box,
    and double the stall on a sick endpoint. Residency no longer rides this path at all.
    """
    from queue_workflows import llm_context_lever, llm_probe, server_switch
    from queue_workflows.llm_backends import factory as _f

    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
            self._switch = server_switch.ServerSwitch(label="box-b-gpu")

    w = _W()
    monkeypatch.setattr(_f, "resolve_base_url", lambda kind="ollama": f"http://{kind}:1")
    monkeypatch.setattr(llm_context_lever, "disabled", lambda: True)   # can_grow False
    calls: list[str] = []
    for name in ("vllm_served_models", "vllm_max_model_len", "vllm_kv_capacity_tokens"):
        monkeypatch.setattr(llm_probe, name,
                            (lambda n: lambda url, **k: calls.append(n) or None)(name))
    w._llm_context_for_kind(VLLM)
    assert "vllm_served_models" not in calls, (
        f"the window observation fetched the served list again: {calls}")


# ── 5. a probe that cannot answer, as it really fails ────────────────────────
#
# Asserting keep-last by patching `probe_box_residents` to RAISE tests nothing real:
# the real collector never raises: "Never raises — a dead endpoint contributes
# nothing", and every leg swallows to `[]`. So a timed-out /api/ps produced an EMPTY set,
# not an absent one, and the guard read that as "the box holds nothing". These drive the
# failure the way production meets it — the transport refusing to answer.


def _worker_with_real_collector(monkeypatch, cache_model="chat-model-a"):
    from queue_workflows.llm_backends import factory as _f

    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
            self.model_cache = _Cache(cache_model)

    monkeypatch.setattr(_f, "resolve_base_url",
                        lambda kind="ollama": {"ollama": "http://o:11434",
                                               "vllm": "http://v:8000"}.get(kind))
    return _W()


def test_a_timed_out_endpoint_does_not_erase_what_the_box_holds(monkeypatch):
    """THE defect: the collector swallows a dead endpoint to an
    empty list, so `except: return` is unreachable and the snapshot became `set()`."""
    from queue_workflows import llm_probe

    w = _worker_with_real_collector(monkeypatch)
    monkeypatch.setattr(llm_probe, "_default_get_json",
                        lambda url, t: {"models": [{"name": "chat-model-a:latest",
                                                    "size": 1, "size_vram": 1}]}
                        if "/api/ps" in url else {})
    w._refresh_box_llm_state()
    assert w._observed_residents() == {"chat-model-a:latest"}

    def _timeout(url, t):
        raise TimeoutError("endpoint did not answer")

    monkeypatch.setattr(llm_probe, "_default_get_json", _timeout)
    for _ in range(claim_worker.ClaimWorker.RESIDENT_ABSENT_MISSES - 1):
        w._refresh_box_llm_state()
        assert w._observed_residents() == {"chat-model-a:latest"}, \
            "one missed probe is not an eviction"


def test_a_model_really_gone_does_expire(monkeypatch):
    """The other direction, which is the whole point of the guard: ollama's keep-alive
    expires, the box answers and lists nothing, and after the debounce the slot goes."""
    from queue_workflows import llm_probe

    w = _worker_with_real_collector(monkeypatch)
    monkeypatch.setattr(llm_probe, "_default_get_json",
                        lambda url, t: {"models": [{"name": "chat-model-a:latest",
                                                    "size": 1, "size_vram": 1}]}
                        if "/api/ps" in url else {})
    w._refresh_box_llm_state()
    monkeypatch.setattr(llm_probe, "_default_get_json",
                        lambda url, t: {"models": []} if "/api/ps" in url else {})
    for _ in range(claim_worker.ClaimWorker.RESIDENT_ABSENT_MISSES):
        w._refresh_box_llm_state()
    assert w._observed_residents() == set(), "an answering, empty box IS an eviction"


def test_one_kinds_silence_does_not_drop_the_other_kinds_model(monkeypatch):
    """A second defect: a box running BOTH kinds where only ollama times out
    returns a NON-empty list (vLLM's model), so no emptiness test can catch it — the
    snapshot is simply replaced and ollama's resident vanishes."""
    from queue_workflows import llm_probe

    w = _worker_with_real_collector(monkeypatch)

    def _both(url, t):
        if "/api/ps" in url:
            return {"models": [{"name": "chat-model-a:latest", "size": 1, "size_vram": 1}]}
        if "/v1/models" in url:
            return {"data": [{"id": "served-by-vllm", "max_model_len": 4096}]}
        return {}
    monkeypatch.setattr(llm_probe, "_default_get_json", _both)
    monkeypatch.setattr(llm_probe, "probe_llm_servers", lambda url, **k: [llm_probe.VLLM])
    monkeypatch.setattr(llm_probe, "vllm_activity", lambda url, **k: {"running": 0})
    w._refresh_box_llm_state()
    assert w._observed_residents() == {"chat-model-a:latest", "served-by-vllm"}

    def _ollama_silent(url, t):
        if "/api/ps" in url:
            raise TimeoutError("ollama did not answer")
        return _both(url, t)
    monkeypatch.setattr(llm_probe, "_default_get_json", _ollama_silent)
    w._refresh_box_llm_state()
    assert "chat-model-a:latest" in w._observed_residents(), \
        "vLLM answering must not evict the model ollama simply did not report"


def test_residency_is_observed_after_the_rogue_sweep_even_though_that_costs_a_tick():
    """TWO contracts meet here and the safety one wins.

    Taking the snapshot before the row is written is tempting, so a
    freshly-loaded model is not published as NULL for one ~10 s tick. But
    `test_rogue_llm.py::test_the_gpu_heartbeat_sweeps_before_it_looks_at_residency` holds
    that the residency probe must see a box the GUARD HAS ALREADY CLEANED — otherwise an
    outsider container's model is counted as a resident of ours. Moving the probe into
    `emit_once` put it before the sweep and broke that.

    So residency stays in `on_tick`, one tick behind, and the cost is bounded: a model
    loaded since the last snapshot reads as unknown, which publishes it (ignorance is not
    absence) rather than blanking it."""
    import inspect

    tick = inspect.getsource(claim_worker.ClaimWorker._box_lease_tick)
    code = "\n".join(l.split("#", 1)[0] for l in tick.splitlines())
    assert "_refresh_box_llm_state()" in code
    assert code.index("rogue_llm.sweep(") < code.index("_refresh_box_llm_state()")
    assert "_refresh_box_llm_state()" not in inspect.getsource(
        claim_worker.ClaimWorker._observe_llm_context)


def test_a_latest_spelled_slot_is_matched(monkeypatch, capture):
    """The surviving mutant: nothing exercised `canonical_tag` on the WANT side, though
    `:latest` is the spelling the fleet's own /api/tags reports."""
    monkeypatch.setattr(model_registry, "server_served_ids", lambda: {"chat-model-a:latest"})
    em = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-b-gpu", model_cache=_Cache("chat-model-a:latest"),
        residents_fn=lambda: ["chat-model-a"])
    monkeypatch.setattr(em, "_vram_total_mb", lambda: None)
    em.emit_once()
    assert capture["kwargs"]["current_model"] == "chat-model-a:latest"


# ── 6. the guard must be able to say NO for a :latest-spelled slot ───────────
#
# A `:latest` test can be VACUOUS: one that asserts the model IS published,
# and dropping `canonical_tag` produces that too — `want not in served` then falls
# through the IN-PROCESS exemption, so the guard is skipped rather than satisfied.
# Nothing distinguished "published because residency vouched" from "published because
# the guard never ran". These test the direction that can tell them apart.


def test_a_latest_spelled_slot_can_still_be_REFUSED(monkeypatch, capture):
    """Slot `chat-model-a:latest`, registry says `chat-model-a`, the box holds nothing.
    Without canonicalisation on the WANT side the slot reads as in-process and is
    published regardless — the guard silently never runs."""
    monkeypatch.setattr(model_registry, "server_served_ids", lambda: {"chat-model-a"})
    em = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-b-gpu", model_cache=_Cache("chat-model-a:latest"),
        residents_fn=lambda: [])
    monkeypatch.setattr(em, "_vram_total_mb", lambda: None)
    em.emit_once()
    assert capture["kwargs"]["current_model"] is None


def test_a_latest_spelled_REGISTRY_id_can_still_refuse(monkeypatch, capture):
    """The mirror: the registry carries the `:latest` spelling and the slot does not."""
    monkeypatch.setattr(model_registry, "server_served_ids", lambda: {"chat-model-a:latest"})
    em = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-b-gpu", model_cache=_Cache("chat-model-a"),
        residents_fn=lambda: [])
    monkeypatch.setattr(em, "_vram_total_mb", lambda: None)
    em.emit_once()
    assert capture["kwargs"]["current_model"] is None


def test_the_absence_window_is_three_snapshots(monkeypatch):
    """The VALUE, not just the name. Every other test loops `range(...)` over the
    constant, so it proves itself: raising it to 50 would leave a dead box vouching for
    a model for eight minutes and nothing would notice."""
    assert claim_worker.ClaimWorker.RESIDENT_ABSENT_MISSES == 3


def test_the_misses_must_be_CONSECUTIVE(monkeypatch):
    """"Consecutive" is the whole premise. If a sighting does not reset the counter the
    rule degrades to CUMULATIVE, and an endpoint that misses one probe a minute blanks a
    live model on a single miss once it has banked three."""
    from queue_workflows import llm_probe

    w = _worker_with_real_collector(monkeypatch)
    held = {"models": [{"name": "chat-model-a:latest", "size": 1, "size_vram": 1}]}

    def _timeout(url, t):
        raise TimeoutError("endpoint did not answer")

    for _ in range(claim_worker.ClaimWorker.RESIDENT_ABSENT_MISSES * 3):
        monkeypatch.setattr(llm_probe, "_default_get_json",
                            lambda url, t: held if "/api/ps" in url else {})
        w._refresh_box_llm_state()                       # seen
        monkeypatch.setattr(llm_probe, "_default_get_json", _timeout)
        w._refresh_box_llm_state()                       # one miss
        assert w._observed_residents() == {"chat-model-a:latest"}, \
            "a single miss blanked a model seen one tick ago — the counter never reset"


def test_a_collector_that_RAISES_keeps_the_warm_snapshot(monkeypatch):
    """The `except` branch is not reachable by a probe timeout, but it still protects
    `_box_warm` (the claim's warm-model inputs) if the collector itself blows up —
    an import error, a config fault. Unpinned, deleting it changed no test."""
    from queue_workflows import model_residency

    w = _worker_with_real_collector(monkeypatch)
    R = model_residency.Resident
    monkeypatch.setattr(model_residency, "probe_box_residents",
                        lambda **k: [R(server="ollama", model="m:latest", mru=1.0)])
    w._refresh_box_llm_state()
    assert w._box_warm == (["ollama"], ["m:latest"])

    def _boom(**k):
        raise RuntimeError("collector blew up")

    monkeypatch.setattr(model_residency, "probe_box_residents", _boom)
    w._refresh_box_llm_state()
    assert w._box_warm == (["ollama"], ["m:latest"]), "the warm snapshot must survive"
