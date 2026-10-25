"""Don't spend ten minutes loading a server that is going to be killed anyway.

The rule: tests must guard the scenario of a model server loading into too little
available RAM and getting killed midway, so no time is wasted loading a server that
will be killed anyway.

A vLLM load can cost many minutes (weights over a network mount, then the KV allocation,
then CUDA-graph and fp4 tuning). Three things have to be true for that time to be well
spent, and each is a section here:

  1. IT MUST NOT START AT ALL when what is free cannot hold it — the pre-check, which
     refuses with the arithmetic instead of finding out at minute nine.
  2. IT MUST BE KILLED EARLY when the box turns against it mid-load — the sentinel,
     every second, rather than at the 1800 s warm timeout.
  3. IT MUST NOT IMMEDIATELY TRY AGAIN into the same conditions. A kill is evidence:
     the box could not hold this server WHILE OTHER THINGS WERE HAPPENING, and a retry
     that only just fits buys another ten wasted minutes. Each mid-load kill raises the
     headroom the next start (and the memory-recovery lift) demands, until a start
     actually completes.
"""
from __future__ import annotations

import pytest

from queue_workflows import claim_worker, llm_context, llm_context_lever, llm_probe, node_queue
from queue_workflows import server_switch
from queue_workflows.config import get_config
from queue_workflows.llm_backends import factory as _factory

GiB = 2**30
PROFILE = {"weights_bytes": int(18.74 * GiB), "kv_bytes": int(42.23 * GiB), "kv_tokens": 684_056,
           "window": 262_144, "utilization": 0.42, "total_bytes": 121 * GiB, "fingerprint": "fp"}


def _container(*, running=False):
    return llm_context_lever.ServerContainer(
        kind="vllm", name="vllm", exists=True, running=running, configured=262144,
        utilization=0.75, utilization_now=0.42, max_num_seqs=8, max_num_seqs_now=8,
        served_name="chat-model-a:latest", fingerprint="fp", container_id="c1",
        started_at="2024-01-01T00:00:00Z")


@pytest.fixture(autouse=True)
def _box(monkeypatch):
    llm_context_lever.reset_cache_for_tests()
    monkeypatch.setattr(_factory, "resolve_base_url", lambda kind="ollama": f"http://{kind}:1")
    for name, value in (("ollama_runners", []), ("loaded_models", []), ("loaded_models_info", []),
                        ("probe_llm_servers", []), ("vllm_served_models", ["chat-model-a:latest"])):
        monkeypatch.setattr(llm_probe, name, lambda url, _v=value, **k: list(_v))
    monkeypatch.setattr(llm_probe, "ollama_model_sizes", lambda url, **k: {})
    monkeypatch.setattr(llm_probe, "ollama_model_info", lambda url, m, **k: {})
    monkeypatch.setattr(llm_probe, "vllm_max_model_len", lambda url, **k: 262144)
    monkeypatch.setattr(llm_probe, "vllm_kv_capacity_tokens", lambda url, **k: None)
    monkeypatch.setattr(llm_probe, "unload_ollama_models", lambda url, names, **k: None)
    monkeypatch.setattr(llm_context_lever, "available", lambda kind, **k: True)
    monkeypatch.setattr(llm_context_lever, "observe_vllm", lambda **k: None)
    monkeypatch.setattr(llm_context, "mem_total_bytes", lambda *a, **k: 121 * GiB)
    cfg = get_config()
    cfg.vllm_start_fn = lambda model: None
    yield
    cfg.vllm_stop_fn = None
    cfg.vllm_start_fn = None
    llm_context_lever.reset_cache_for_tests()


def _worker(monkeypatch, *, available_gib, running=False):
    class _W(claim_worker.ClaimWorker):
        def __init__(self):
            self.host = "box-b-gpu"
            self.queue = "gpu"
            self._box_served = ["ollama"]
            self._switch = server_switch.ServerSwitch(label="box-b-gpu")
    w = _W()
    w._physical_box = lambda: "box-b"
    w._pool_parallelism_configured = lambda: 8
    w._get_box_lease = lambda: None
    w._gpu_active_count = lambda: 0
    w._box_held_model = None
    w._pool_inflight = 0
    w._inline_running = False
    monkeypatch.setattr(w, "_box_has_ollama_endpoint", lambda: False)
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: _container(running=running))
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: available_gib * GiB)
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", PROFILE)
    now = [1000.0]
    monkeypatch.setattr(claim_worker.time, "monotonic", lambda: now[0])
    return w, now


# ── 1. it must not start at all ──────────────────────────────────────────────


def test_a_start_that_cannot_fit_is_refused_before_a_single_byte_is_loaded(monkeypatch):
    """The whole point of the pre-check: the arithmetic is known BEFORE the ten
    minutes, so the answer comes now and names the numbers."""
    w, _now = _worker(monkeypatch, available_gib=40)
    monkeypatch.setattr(llm_context_lever, "set_context",
                        lambda *a, **k: pytest.fail("must not start what cannot fit"))
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    reason = w._switch.quarantine_reason("vllm") or ""
    assert "cannot hold one" in reason and "GB" in reason
    assert w._switch.quarantined("vllm") is True


# ── 2. it must be killed early, not at the warm timeout ──────────────────────


def test_a_load_the_box_turns_against_is_killed_within_a_second_not_in_half_an_hour(monkeypatch, caplog):
    """The sentinel reads every second; the warm budget is 1800 s. Ten minutes of
    loading is bad, twenty-nine is worse."""
    stopped: list = []
    get_config().vllm_stop_fn = lambda: stopped.append(True) or True
    w, now = _worker(monkeypatch, available_gib=110)
    monkeypatch.setattr(llm_context_lever, "set_context", lambda *a, **k: {"status": "started"})
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert w._switch.state == "warming"
    monkeypatch.setattr(llm_context_lever, "describe", lambda kind, **k: _container(running=True))
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: 6 * GiB)
    now[0] += 120                                     # two minutes into the load
    with caplog.at_level("ERROR", logger="queue_workflows.claim_worker"):
        w._memory_tick(6 * GiB)
    assert stopped == [True]
    assert w._switch.state == "idle"
    assert any("2.0 min into the load" in r.getMessage() for r in caplog.records), \
        "say what the kill cost, or nobody can weigh it"


# ── 3. it must not walk straight back into the same conditions ───────────────


def test_a_kill_raises_the_bar_for_the_next_start(monkeypatch):
    """72 GB free is enough for a first attempt (2 requests) — and exactly the
    condition that just failed. After one mid-load kill the next start demands
    STARTUP_LOSS_HEADROOM more than the bare need, so a retry that would only just
    fit waits instead of burning another ten minutes."""
    w, now = _worker(monkeypatch, available_gib=72)
    first = w._vllm_start_plan()
    assert first.refusal is None, "with a clean history this start is allowed"
    w._note_startup_loss("the box ran out")
    w.__dict__.pop("_vllm_decision", None)
    w.__dict__.pop("_vllm_need_cache", None)
    again = w._vllm_start_plan()
    assert again.refusal is not None
    assert "killed mid-load" in again.refusal and "1 time" in again.refusal


def test_the_bar_rises_with_each_loss_and_is_capped(monkeypatch):
    w, _now = _worker(monkeypatch, available_gib=110)
    base = w._vllm_startup_headroom()
    assert base == 1.0
    w._note_startup_loss("x")
    one = w._vllm_startup_headroom()
    w._note_startup_loss("x")
    two = w._vllm_startup_headroom()
    for _ in range(10):
        w._note_startup_loss("x")
    assert 1.0 < one < two <= w._vllm_startup_headroom()
    assert w._vllm_startup_headroom() == claim_worker.ClaimWorker.STARTUP_LOSS_HEADROOM_MAX


def test_a_completed_start_forgives_the_history(monkeypatch):
    """The losses say "this box could not hold it while other things were happening".
    A start that reached its cache says the opposite, and is the newer evidence."""
    w, _now = _worker(monkeypatch, available_gib=110, running=True)
    w._box_served = ["vllm"]
    w._note_startup_loss("x"); w._note_startup_loss("x")
    assert w._vllm_startup_headroom() > 1.0
    monkeypatch.setattr(llm_context_lever, "observe_vllm",
                        lambda **k: llm_context.VllmProfile.from_record(PROFILE))
    monkeypatch.setattr(llm_probe, "vllm_kv_capacity_tokens", lambda url, **k: 684_056)
    w._observe_llm_context()                            # the start completed
    assert w._vllm_startup_headroom() == 1.0


def test_the_recovery_lift_demands_the_same_raised_bar(monkeypatch):
    """Otherwise the lift is a back door into exactly the retry the bar exists to
    prevent: memory returns to 'just enough', the hold lifts, ten more minutes burn."""
    w, now = _worker(monkeypatch, available_gib=40)
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert w._switch.quarantined("vllm") is True
    w._note_startup_loss("killed mid-load")
    w.__dict__.pop("_vllm_need_cache", None)
    need = w._vllm_memory_need_bytes()
    for _ in range(70):
        now[0] += 1
        w._memory_tick(need - 1)
    assert w._switch.quarantined("vllm") is True, "just under the raised bar: keep waiting"
    for _ in range(70):
        now[0] += 1
        w._memory_tick(need + 1)
    assert w._switch.quarantined("vllm") is False


def test_the_need_grows_by_exactly_the_headroom(monkeypatch):
    w, _now = _worker(monkeypatch, available_gib=110)
    plain = w._vllm_memory_need_bytes()
    w._note_startup_loss("x")
    w.__dict__.pop("_vllm_need_cache", None)
    raised = w._vllm_memory_need_bytes()
    assert raised == pytest.approx(plain * w._vllm_startup_headroom(), rel=1e-6)


def test_a_shed_that_kills_no_warm_is_not_counted_as_a_lost_start(monkeypatch):
    """Shedding a SERVING vLLM (or an idle box's models) costs no load time — only a
    warm in flight is a wasted start."""
    get_config().vllm_stop_fn = lambda: True
    w, _now = _worker(monkeypatch, available_gib=6, running=True)
    assert w._switch.state == "idle"
    w._shed_llm_if_box_is_dying()
    assert w._vllm_startup_headroom() == 1.0


def test_enough_headroom_after_a_loss_lets_the_start_through(monkeypatch):
    """The bar is a bar, not a ban: with the raised need actually covered, the start
    goes ahead — and a box whose memory cannot be read is not held back by it."""
    w, _now = _worker(monkeypatch, available_gib=110)
    w._note_startup_loss("x")
    plan = w._vllm_start_plan()
    assert plan.refusal is None and plan.max_num_seqs, "110 GB clears 1.25x the need"
    w.__dict__.pop("_vllm_decision", None)
    monkeypatch.setattr(llm_context, "mem_available_bytes", lambda *a, **k: None)
    w.__dict__.pop("_vllm_need_cache", None)
    assert w._vllm_start_plan().refusal is None, "an unreadable box is not judged"


def test_the_warm_clock_resets_between_loads(monkeypatch):
    """A clock left running from a previous warm would misreport the next kill's cost
    (and a stale one across an idle period would report minutes that never happened)."""
    w, now = _worker(monkeypatch, available_gib=110)
    monkeypatch.setattr(llm_context_lever, "set_context", lambda *a, **k: {"status": "started"})
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert w._switch.state == "warming"
    first = w.__dict__["_warm_started_at"]
    now[0] += 300
    w._switch.abandon("test")
    w._drive_server_switch()                         # back to idle: the clock goes
    assert "_warm_started_at" not in w.__dict__
    w._switch.request("vllm")
    w._drive_server_switch(); w._drive_server_switch()
    assert w.__dict__["_warm_started_at"] > first, "a new load, a new clock"
