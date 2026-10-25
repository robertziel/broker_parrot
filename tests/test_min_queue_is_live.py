"""An operator's threshold edit must reach the thing that acts on it.

THE FAILURE. The topology file's ``vllm_min_queue`` is documented — in
:func:`resolve_cold_start_min_queue`, in the file's own comments, and in the operator's
mental model — as re-read on change, "no worker restart needed". The claim-side gate does
re-read it. The START side did not: the GPU worker resolved the number ONCE while arming
its idle stopper and handed the stopper an int.

The cost: a box armed with ``vllm_min_queue: 1000000`` (an earlier hold, set while no
box had a startable vLLM). Later the operator lowered it to 0 — "the first queued job
that accepts vLLM starts it" — and later still exactly such a job was queued. Every
tick after that compared 1 against the stale 1000000 and did nothing, silently: the decision returns before it logs. The job sat UNASSIGNABLE with
"no live worker serves any of: vllm" while the one mechanism that would have started a
vLLM had been switched on hours earlier.

So the threshold is read where it is USED, and a caller may pass a callable.
"""
from __future__ import annotations

from queue_workflows import llm_idle_unload as idle


def _stopper(min_queue, *, pending, started: list):
    return idle.VLLMIdleStopper(
        active_fn=lambda: 0,
        activity_fn=lambda: {"served": False, "running": 0, "finished": 0},
        stop_fn=lambda: None,
        start_fn=lambda: started.append(True),
        pending_fn=lambda: pending,
        min_queue=min_queue,
        ttl_s=300.0,
    )


def test_a_threshold_lowered_after_arming_takes_effect_without_a_restart():
    live = {"value": 1000000}
    started: list = []
    stopper = _stopper(lambda: live["value"], pending=1, started=started)

    assert stopper.start_if_wanted_once() is False, "1 job does not justify a start yet"
    live["value"] = 0                      # the operator edits the topology file
    assert stopper.start_if_wanted_once() is True
    assert started == [True]


def test_a_raised_threshold_takes_effect_the_same_way():
    live = {"value": 0}
    started: list = []
    stopper = _stopper(lambda: live["value"], pending=1, started=started)
    live["value"] = 5
    assert stopper.start_if_wanted_once() is False
    assert started == []


def test_a_plain_number_still_works_exactly_as_before():
    started: list = []
    assert _stopper(5, pending=1, started=started).start_if_wanted_once() is False
    assert _stopper(0, pending=1, started=started).start_if_wanted_once() is True


def test_an_unreadable_threshold_holds_the_expensive_action_back():
    """Same direction as every other unknown here: a start is minutes of weight
    loading, so ignorance must not trigger it."""
    def boom():
        raise RuntimeError("topology file vanished")

    started: list = []
    assert _stopper(boom, pending=1, started=started).start_if_wanted_once() is False
    assert started == []


def test_the_worker_hands_over_the_LIVE_reader_not_a_number():
    """The defect was not in the stopper — it was in what the worker passed it. A
    source check, because the wiring is what was wrong and nothing else asserts it."""
    import inspect

    from queue_workflows import claim_worker

    src = inspect.getsource(claim_worker.ClaimWorker._arm_llm_idle_reaper)
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    assert "min_queue=_llm_factory.resolve_cold_start_min_queue," in code.replace(" ", "") \
        or "min_queue=lambda:" in code.replace(" ", ""), (
        "the worker must pass the RESOLVER, not its value at arm time — an operator's "
        "topology edit is documented to need no restart")
