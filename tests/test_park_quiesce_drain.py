"""A DRAIN-park must not fight the job it is deliberately keeping.

Observed live, over about twenty minutes. An operator parked both GPU boxes from the
panel. The panel's park lever writes ``stop_policy='drain'``, whose whole meaning is
"finish what you are holding, claim nothing new" — so a 31-minute LLM job kept
running, as designed. Meanwhile the park watchdog
(rule #3: a parked box serves nobody, kill on sight) told ollama to unload the model
that job was streaming from — every 5 seconds, 265 times:

    [park-watchdog] box-a-gpu: box is PARKED but ollama holds chat-model-a:latest —
    unloading NOW; a parked box serves nobody

ollama does not refuse those. It applies the zero keep-alive at the REQUEST BOUNDARY,
so the runner exits when the in-flight call ends and the job's NEXT call cold-reloads
34 GiB (``Stopping...`` means "expiry scheduled", not "refused"). Measured: one box went
from 6 model reloads in 3 h 21 m before the park to 7 in the 30 minutes after it; the
other from 5 in 3 h 12 m to 10 in 28 minutes. The park was not releasing the card — it
was making the job it had promised to keep thrash it, while the panel said PARKED and the
log filled with an identical line at 0.2 Hz. Two engine policies, each correct alone,
cancelling each other out.

The resolution, pinned here: **rule #3 is about OTHER people's traffic, not our own
drained job.** Rule #2 died because a peer POSTing completions over the fast interface
could keep a parked box serving for ever — an exemption anyone outside the box could
hold open. A job THIS worker is still running is the opposite: the park policy itself
chose to preserve it, it is bounded (it ends, and its watchdog bounds it), and evicting
its model mid-stream would fail work the operator asked to finish. So the watchdog holds
off exactly while this worker has in-flight work, and kills on sight the moment it does
not — which is also the moment ``hard`` park reaches, since that policy re-queues the job
first and therefore has no in-flight work at all.
"""
from __future__ import annotations

import logging

from queue_workflows import llm_idle_unload as idle


class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def _watchdog(activity, infos, in_flight=None, clock=None, hold_off_max_s=None):
    """A park watchdog over mutable one-element cells, so a test can change what the
    box looks like between ticks."""
    stopped, unloaded = [], []
    kw = {}
    if in_flight is not None:
        kw["in_flight_fn"] = lambda: in_flight[0]
    if clock is not None:
        kw["now_fn"] = clock
    if hold_off_max_s is not None:
        kw["hold_off_max_s"] = hold_off_max_s
    q = idle.ParkQuiesce(
        vllm_activity_fn=lambda: activity[0],
        vllm_stop_fn=lambda: stopped.append(True),
        ollama_models_fn=lambda: infos[0],
        ollama_unload_fn=lambda models: unloaded.append(list(models)),
        label="box-a-gpu",
        **kw,
    )
    return q, stopped, unloaded


_SERVING = {"served": True, "running": 9}
_QUIET = {"served": False, "running": 0}
_LOADED = [{"name": "chat-model-a:latest"}]


# ── the fix: our own drained job is not "someone else's traffic" ─────────────


def test_a_drained_job_in_flight_holds_off_the_ollama_eviction():
    """THE INCIDENT. The model the running job is streaming from must not be pulled
    out from under it 265 times."""
    q, stopped, unloaded = _watchdog([_QUIET], [_LOADED], in_flight=[1])

    assert q.tick() == (False, [])
    assert unloaded == []


def test_a_drained_job_in_flight_holds_off_the_vllm_stop_too():
    q, stopped, unloaded = _watchdog([_SERVING], [[]], in_flight=[1])

    assert q.tick() == (False, [])
    assert stopped == []


def test_the_kill_resumes_THE_MOMENT_the_drained_job_finishes():
    """The hold-off is bounded by the job, not by a timer: no job, no exemption."""
    in_flight = [1]
    q, stopped, unloaded = _watchdog([_SERVING], [_LOADED], in_flight=in_flight)
    q.tick()
    assert (stopped, unloaded) == ([], [])

    in_flight[0] = 0

    assert q.tick() == (True, ["chat-model-a:latest"])
    assert stopped == [True]
    assert unloaded == [["chat-model-a:latest"]]


# ── rule #3 is preserved exactly where it was won ────────────────────────────


def test_a_PEERS_remote_traffic_still_gets_no_exemption():
    """Rule #2 died here: 9 concurrent completions from another box kept a parked box
    serving at 96 % GPU for ever. That must stay dead — those requests are not ours."""
    q, stopped, unloaded = _watchdog([_SERVING], [_LOADED], in_flight=[0])

    assert q.tick() == (True, ["chat-model-a:latest"])


def test_a_resurrected_server_is_re_killed_while_nothing_is_in_flight():
    in_flight = [0]
    q, stopped, unloaded = _watchdog([_SERVING], [_LOADED], in_flight=in_flight)
    for _ in range(3):
        q.tick()
    assert stopped == [True, True, True]


def test_an_unreadable_in_flight_probe_falls_back_to_KILLING():
    """Fail toward rule #3: if we cannot tell whether we hold work, a parked box still
    serves nobody."""
    def _boom():
        raise RuntimeError("no")

    q = idle.ParkQuiesce(
        vllm_activity_fn=lambda: _SERVING,
        vllm_stop_fn=lambda: None,
        ollama_models_fn=lambda: _LOADED,
        ollama_unload_fn=lambda models: None,
        in_flight_fn=_boom,
        label="box-a-gpu",
    )

    assert q.tick() == (True, ["chat-model-a:latest"])


def test_without_the_in_flight_seam_the_behaviour_is_unchanged():
    """Byte-compat: a caller that does not pass the seam gets rule #3 verbatim."""
    q, stopped, unloaded = _watchdog([_SERVING], [_LOADED])

    assert q.tick() == (True, ["chat-model-a:latest"])


# ── the 0.2 Hz identical log line ────────────────────────────────────────────


def test_the_unload_line_is_not_repeated_on_every_tick(caplog):
    """265 identical lines in 22 minutes buried everything else on the box."""
    clock = _Clock()
    q, _, _ = _watchdog([_QUIET], [_LOADED], in_flight=[0], clock=clock)

    with caplog.at_level(logging.INFO, logger="queue_workflows.llm_idle_unload"):
        for _ in range(20):
            q.tick()
            clock.advance(5)

    said = [r for r in caplog.records if "ollama holds" in r.getMessage()]
    assert len(said) == 2, f"expected the line once per report window, got {len(said)}"


def test_the_hold_off_is_reported_but_not_on_every_tick(caplog):
    clock = _Clock()
    q, _, _ = _watchdog([_QUIET], [_LOADED], in_flight=[1], clock=clock)

    with caplog.at_level(logging.INFO, logger="queue_workflows.llm_idle_unload"):
        for _ in range(20):
            q.tick()
            clock.advance(5)

    said = [r for r in caplog.records if "draining" in r.getMessage()]
    assert 1 <= len(said) <= 2
    assert "1 job" in said[0].getMessage()


def test_a_CHANGE_in_what_the_box_holds_is_always_reported(caplog):
    """Rate-limiting must never hide new information."""
    clock = _Clock()
    infos = [_LOADED]
    q, _, _ = _watchdog([_QUIET], infos, in_flight=[0], clock=clock)

    with caplog.at_level(logging.INFO, logger="queue_workflows.llm_idle_unload"):
        q.tick()
        clock.advance(5)
        infos[0] = [{"name": "other-model:latest"}]
        q.tick()

    said = [r.getMessage() for r in caplog.records if "ollama holds" in r.getMessage()]
    assert len(said) == 2
    assert "other-model:latest" in said[1]


# ── the hold-off must itself be bounded ──────────────────────────────────────


def test_an_ENDLESS_job_cannot_suppress_the_park_for_ever(caplog):
    """RED TEAM. The exemption is only as bounded as the job is, and a pool-lane job
    has no wall-clock watchdog: `_run_pool_node` arms none, and the lease renewer keeps
    renewing while the process lives, so lease-reclaim cannot fire either. One wedged
    HTTP call would pin the counter at 1 and hold the card SILENTLY for ever — the
    operator sees PARKED and the card never comes back. So the hold-off has its own
    ceiling, after which rule #3 wins and says why."""
    clock = _Clock()
    q, stopped, unloaded = _watchdog([_SERVING], [_LOADED], in_flight=[1], clock=clock,
                                     hold_off_max_s=600.0)

    with caplog.at_level(logging.WARNING, logger="queue_workflows.llm_idle_unload"):
        assert q.tick() == (False, [])
        clock.advance(601)
        result = q.tick()

    assert result == (True, ["chat-model-a:latest"])
    assert any("held the card" in r.getMessage() for r in caplog.records)


def test_the_hold_off_ceiling_measures_CONTINUOUS_holding_not_total():
    """A box that drains, serves again, and is parked again later starts a fresh
    grace — the ceiling is about ONE stuck job, not a lifetime budget."""
    clock = _Clock()
    in_flight = [1]
    q, _, _ = _watchdog([_QUIET], [_LOADED], in_flight=in_flight, clock=clock,
                        hold_off_max_s=600.0)
    q.tick()
    clock.advance(500)

    in_flight[0] = 0                      # drained: the grace is spent and reset
    q.tick()
    in_flight[0] = 1
    q.tick()
    clock.advance(500)                    # 1000 s total, but only 500 continuous

    assert q.tick() == (False, [])


def test_the_default_ceiling_is_the_gpu_job_budget():
    """We hold off no longer than a GPU job may legitimately run."""
    from queue_workflows import claim_worker, llm_idle_unload as idle

    assert idle.ParkQuiesce.DEFAULT_HOLD_OFF_MAX_S == claim_worker.GPU_DEFAULT_BUDGET_S


# ── the seam must be the LLM lane, not every gpu job ─────────────────────────


def test_the_worker_exempts_on_the_POOL_lane_only_not_on_a_diffusion_render():
    """RED TEAM. `_gpu_active_count` counts the inline diffusion lane too. A render
    does not use the LLM at all, so exempting on it would let a PEER's traffic keep a
    parked box's vLLM at 96 % — exactly the rule #2 failure that rule #3 replaced."""
    import inspect
    from queue_workflows import claim_worker

    src = inspect.getsource(claim_worker.ClaimWorker._park_quiesce_tick) \
        if hasattr(claim_worker.ClaimWorker, "_park_quiesce_tick") else ""
    if not src:
        src = inspect.getsource(claim_worker.ClaimWorker)
    i = src.index("ParkQuiesce(")
    depth, j = 0, src.index("(", i)
    for j in range(src.index("(", i), len(src)):        # the call, exactly
        depth += (src[j] == "(") - (src[j] == ")")
        if depth == 0:
            break
    call = src[i:j + 1]
    # CODE only: the comment above the seam names `_gpu_active_count` to say why it is
    # NOT used, and a scan that cannot tell prose from code would read that as the bug.
    wiring = "\n".join(ln.split("#", 1)[0] for ln in call.splitlines())

    assert "in_flight_fn" in wiring
    assert "_gpu_active_count" not in wiring, (
        "the exemption must not count the inline diffusion lane")
    assert "_pool_inflight" in wiring
