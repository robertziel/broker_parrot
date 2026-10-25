"""A PER-BOX CEILING on how many GPU jobs one machine runs at once.

THE PROBLEM. A GPU box's concurrency is computed, not declared: the operator's
``worker_controls.llm_parallelism`` (PAR) capped by what the running vLLM's KV
cache can actually hold, capped again by what the serving kind can absorb. Every
one of those inputs describes the LLM SERVER. None of them describes the BOX —
its power envelope, its unified memory, the thermal latch a flood of concurrent
work trips. On some machines a modest fixed concurrency is the sweet spot; the
computed number can go higher and the box pays for it.

So the operator needs a last word that is about the MACHINE:
``gpu_box_max_concurrent`` — a mapping of box name to ceiling, e.g.
``{"box-b": 3, "box-c": 3}``.

THE FOUR PROPERTIES THIS FILE FREEZES.

1. **It is a CEILING, never a floor.** It may only lower the computed number. A
   box whose vLLM cache holds 2 requests still runs 2 with a ceiling of 3 — the
   ceiling is the operator's opinion about the machine, not a promise the server
   can keep.
2. **It is keyed by the PHYSICAL BOX** (``gpu_model_lease.default_box_id()``, e.g.
   ``box-b``), the same identity ``avoid_box``/``force_box`` place against — NOT
   the per-project lane label (``box-b-gpu``). Two projects' workers share one
   card; a ceiling keyed per lane label would be N ceilings on one machine.
3. **It is applied in ONE place** — ``ClaimWorker._pool_parallelism()``, the
   single function every consumer of "how many can this box run" already goes
   through (``_gpu_slot_budget`` → the heartbeat and the DB-side box-slot
   arbiter, ``_kind_caps``, the inline ``run_once`` gate, the pool feeder's
   budget and executor size). A ceiling sprinkled at call sites is a ceiling that
   one new call site silently escapes.
4. **A nonsense value is IGNORED, never obeyed.** 0, a negative, ``"three"``,
   junk — the box keeps today's behaviour. The standing design rule is NEVER
   PARK: a ceiling of 0 would be a GPU lane that claims nothing, silently,
   forever, and a typo must not be able to do that to a machine.

A box nobody names is untouched — no ceiling, byte-identical to pre-knob.
"""

from __future__ import annotations

import pytest

from queue_workflows import claim_worker, node_queue, worker_control
import queue_workflows
from queue_workflows.db import connection


class _Cache:
    current_model = None


def _worker(monkeypatch, *, box="box-b", host="box-b-gpu", par=8):
    """A gpu ClaimWorker on physical box ``box``, lane-labelled ``host``, whose
    operator PAR reads ``par``. The box id is set BEFORE construction because
    ``_physical_box()`` caches its answer for the life of the process."""
    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_BOX_ID", box)
    monkeypatch.setattr(
        worker_control, "llm_config_for",
        lambda h, q: worker_control.LLMConfig(parallelism=par),
    )
    return claim_worker.ClaimWorker(queue="gpu", host=host, model_cache=_Cache())


# ── the ceiling lowers ───────────────────────────────────────────────────────


def test_the_ceiling_lowers_a_higher_computed_concurrency(monkeypatch):
    """The whole point: box-b computes 8 from its server config and the operator
    says this machine runs 3. The last word is 3."""
    w = _worker(monkeypatch, box="box-b", par=8)
    assert w._pool_parallelism() == 8, "no ceiling configured ⇒ the computed PAR"
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 3, "box-c": 3})
    assert w._pool_parallelism() == 3


def test_the_ceiling_never_raises_a_lower_one(monkeypatch):
    """A box whose vLLM cache holds only 2 full-window requests must still run 2
    under a ceiling of 3. The ceiling is a maximum the operator is willing to
    allow, never a minimum the box must reach — raising it here would hand the
    server more work than its KV cache can hold, which is the preemption storm
    ``_vllm_concurrency`` exists to prevent."""
    w = _worker(monkeypatch, box="box-b", par=8)
    w.__dict__["_vllm_concurrency"] = 2
    w._box_served = [worker_control.SERVER_TYPE_VLLM]
    assert w._pool_parallelism() == 2
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 3})
    assert w._pool_parallelism() == 2


def test_an_operator_par_below_the_ceiling_is_untouched(monkeypatch):
    """Same law from the other input: a PAR-1 box under a ceiling of 3 stays 1."""
    w = _worker(monkeypatch, box="box-b", par=1)
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 3})
    assert w._pool_parallelism() == 1


def test_a_box_nobody_named_keeps_todays_behaviour(monkeypatch):
    """A fleet is not uniform. A ceiling naming box-b/box-c must leave
    every other machine exactly as it was — this knob is opt-in per box, and an
    unnamed box has NO ceiling (not a default one)."""
    w = _worker(monkeypatch, box="box-a", host="box-a-gpu", par=8)
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 3, "box-c": 3})
    assert w._pool_parallelism() == 8
    assert w._gpu_slot_budget() == 8


def test_the_ceiling_is_keyed_by_box_not_by_lane_label(monkeypatch):
    """``box-b`` is the machine; ``box-b-gpu`` is one project's lane on it. The
    ceiling protects the CARD, which every project's lane shares, so it keys on
    the box identity ``force_box``/``avoid_box`` already place against. A map
    written against the lane label names nothing and must not bind."""
    w = _worker(monkeypatch, box="box-b", host="box-b-gpu", par=8)
    queue_workflows.configure(gpu_box_max_concurrent={"box-b-gpu": 3})
    assert w._pool_parallelism() == 8, "the lane label is not the box identity"
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 3})
    assert w._pool_parallelism() == 3


# ── the two spellings ────────────────────────────────────────────────────────


def test_the_env_spelling_and_the_configure_spelling_agree(monkeypatch):
    """The fleet sets env vars in compose; a host library is wired through
    ``configure()``. Both must reach the same ceiling or the deploy and the code
    disagree about what a machine is allowed to do."""
    w = _worker(monkeypatch, box="box-b", par=8)
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 3, "box-c": 3})
    via_configure = w._pool_parallelism()

    queue_workflows.configure(gpu_box_max_concurrent={})
    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT", "box-b=3,box-c=3")
    via_env = w._pool_parallelism()

    assert via_configure == via_env == 3


def test_the_canonical_env_spelling_resolves(monkeypatch):
    """Every engine knob is read through ``envcompat`` under its canonical
    ``QUEUE_WORKFLOWS_*`` name."""
    w = _worker(monkeypatch, box="box-b", par=8)
    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT", "box-b=3")
    assert w._pool_parallelism() == 3


def test_the_env_spelling_wins_over_configure(monkeypatch):
    """Same precedence as every other knob in this engine (``cpu_worker_threads``,
    ``download_max_concurrent``): the operator's env override beats what the host
    code compiled in, because the operator is the one standing next to the box."""
    w = _worker(monkeypatch, box="box-b", par=8)
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 6})
    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT", "box-b=2")
    assert w._pool_parallelism() == 2


def test_a_bare_integer_caps_the_box_that_reads_it(monkeypatch):
    """A box whose compose file is its own gets to say just ``3`` — the value
    means "this machine", whichever one reads it. Without this, the natural thing
    an operator types into one box's env would parse as nothing and be ignored."""
    w = _worker(monkeypatch, box="box-b", par=8)
    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT", "3")
    assert w._pool_parallelism() == 3


def test_a_named_entry_beats_the_bare_default_for_that_box(monkeypatch):
    """One fleet-wide variable can carry both: a floor-plan default plus the
    machines that differ. The named entry is the more specific statement."""
    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT", "2,box-b=3")
    box_b = _worker(monkeypatch, box="box-b", par=8)
    assert box_b._pool_parallelism() == 3


# ── nonsense is ignored, never obeyed ────────────────────────────────────────


@pytest.mark.parametrize("raw", ["box-b=0", "box-b=-1", "box-b=three",
                                 "box-b=", "=3", "box-b", "", "   ", "0", "-4",
                                 "box-b=3=4", ",,,"])
def test_a_malformed_or_non_positive_ceiling_is_ignored(monkeypatch, raw):
    """NEVER PARK. A ceiling of 0 (or a value that cannot be read as a positive
    int) is not a ceiling — it is a GPU lane that claims nothing, silently,
    forever, on a machine the operator believes is working. A typo in an env var
    must not be able to do that, so an unusable value leaves the box exactly as
    it was.

    ``"=3"`` is in this list deliberately: an entry carrying an ``=`` is NAMING a
    box, so it must name one. The fleet-wide default has its own unambiguous
    spelling (a bare integer), and reading a half-typed ``box-b=3`` as "cap every
    machine in the fleet" is not a guess this knob gets to make."""
    w = _worker(monkeypatch, box="box-b", par=8)
    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT", raw)
    assert w._pool_parallelism() == 8
    assert w._gpu_slot_budget() >= 1


def test_a_malformed_configure_value_is_ignored_too(monkeypatch):
    """The same law on the ``configure()`` side — a host that computes a 0 from
    its own config must not park the machine either."""
    w = _worker(monkeypatch, box="box-b", par=8)
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 0, "box-c": "x"})
    assert w._pool_parallelism() == 8


def test_one_bad_entry_does_not_void_the_good_ones(monkeypatch):
    """A fleet-wide variable is edited by hand. One unreadable entry must cost
    only that entry — the other machines keep the ceiling the operator set."""
    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_BOX_MAX_CONCURRENT",
                       "box-b=oops,box-c=3")
    box_b = _worker(monkeypatch, box="box-b", par=8)
    assert box_b._pool_parallelism() == 8
    box_c = _worker(monkeypatch, box="box-c", host="box-c-gpu", par=8)
    assert box_c._pool_parallelism() == 3


# ── every consumer sees the capped number (the "one place" proof) ────────────


def test_the_heartbeat_advertises_the_capped_number(monkeypatch):
    """The panel reads ``worker_heartbeats.concurrency``. If the box advertised 8
    while claiming at most 3, every capacity view in the fleet — the panel's
    "N at a time", the fill-before-spill defer heuristic that ranks peers by this
    column — would believe in five slots that do not exist. The number a box
    publishes must be the number it will actually run."""
    w = _worker(monkeypatch, box="box-b", par=8)
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 3})
    assert w._gpu_slot_budget() == 3

    # ...and the value that lands in the row, through the real emitter.
    w.heartbeat._llm_servers_fn = lambda: []
    w.heartbeat._llm_context_fn = None
    w.heartbeat._residents_fn = None
    w.heartbeat._on_tick = None
    w.heartbeat.emit_once()
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT concurrency FROM worker_heartbeats "
            "WHERE host_label = %s AND queue = 'gpu'",
            ("box-b-gpu",),
        )
        assert cur.fetchone()["concurrency"] == 3


def test_the_claim_arbiter_is_given_the_capped_number(monkeypatch):
    """The DB-side box-slot arbiter (``claim_next_gpu_job(max_running=...)``) is
    where "how many at once" is actually ENFORCED — in-process counters drift,
    the claim statement does not. It must be handed the capped number, or the
    ceiling is advice the claim path ignores."""
    w = _worker(monkeypatch, box="box-b", par=8)
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 3})
    seen: dict[str, object] = {}

    def spy(*a, **kw):
        seen.update(kw)
        return None

    monkeypatch.setattr(node_queue, "claim_next_gpu_job", spy)
    w._claim()
    assert seen["max_running"] == 3
    # The per-kind widths are derived from the same number, so vLLM's batching
    # lane cannot be handed more than the machine is allowed to run either.
    assert seen["kind_caps"][worker_control.SERVER_TYPE_VLLM] == 3


def test_the_inline_lane_admits_at_most_the_capped_number(monkeypatch):
    """``run_once`` reserves the inline diffusion slot only while the machine's
    total in-flight stays under the budget. With a ceiling of 3, a box already
    running 3 pool jobs claims nothing inline — it spills to an idle peer instead
    of becoming the fourth concurrent job on a card rated for three."""
    w = _worker(monkeypatch, box="box-b", par=8)
    queue_workflows.configure(gpu_box_max_concurrent={"box-b": 3})
    monkeypatch.setattr(node_queue, "claim_next_gpu_job",
                        lambda *a, **kw: pytest.fail("claimed past the ceiling"))
    w._pool_inflight = 3
    assert w.run_once() is False
