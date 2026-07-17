"""ONE model per GPU box — enforced by the engine, not by config.

WHY. A box can hold a model three ways (ollama daemon, vLLM sidecar, in-process
ModelCache), and only the third was arbitrated (gpu_model_lease). Deploy-level
guards (OLLAMA_MAX_LOADED_MODELS, KEEP_ALIVE, crons) drift — and a second project
pooled onto the same box (migration 0017) can load a second model with nothing in
the engine able to see it, let alone stop it. Result: a 124 GB box at 5 GB free with
two ~30 GB models resident. Point fixes kept treating symptoms; this module owns the
invariant: the box's LLM SERVERS hold at most ``cap`` (default 1) distinct models.
More than that is a VIOLATION — the enforcer hard-kills the extras (ollama
``keep_alive:0`` per model; vLLM sidecar stop) AND raises ``ModelResidencyViolation``
so it is loud, every time, wherever it happens.

Pure decision + injected collectors/killers, so the whole thing runs against fakes.
"""

from __future__ import annotations

import logging

import pytest

from queue_workflows import model_residency
from queue_workflows.model_residency import (
    ModelResidencyViolation,
    Resident,
    decide_evictions,
)


def _r(server, model, mru=0.0):
    return Resident(server=server, model=model, mru=mru)


# ── the pure decision ─────────────────────────────────────────────────────────


def test_no_eviction_at_or_under_cap():
    assert decide_evictions([], cap=1) == []
    assert decide_evictions([_r("ollama", "llama3")], cap=1) == []
    two = [_r("ollama", "a"), _r("ollama", "b")]
    assert decide_evictions(two, cap=2) == []


def test_two_ollama_models_keep_most_recently_used():
    old, new = _r("ollama", "old-model", mru=100.0), _r("ollama", "new-model", mru=200.0)
    assert decide_evictions([old, new], cap=1) == [old]
    assert decide_evictions([new, old], cap=1) == [old]     # order-independent


def test_cross_server_desired_type_wins():
    # Operator desired llm_server_type (worker_controls 0013) decides the keeper
    # when both server types hold a model — regardless of recency.
    ol, vl = _r("ollama", "llama3", mru=999.0), _r("vllm", "vlm-x", mru=1.0)
    assert decide_evictions([ol, vl], cap=1, desired_server="vllm") == [ol]
    assert decide_evictions([ol, vl], cap=1, desired_server="ollama") == [vl]


def test_cross_server_without_desired_falls_back_to_mru():
    ol, vl = _r("ollama", "llama3", mru=999.0), _r("vllm", "vlm-x", mru=1.0)
    assert decide_evictions([ol, vl], cap=1) == [vl]


def test_same_model_on_one_server_multiple_rows_is_not_a_violation():
    rows = [_r("ollama", "llama3", 1.0), _r("ollama", "llama3", 2.0)]
    assert decide_evictions(rows, cap=1) == []


def test_three_models_evicts_all_but_keeper():
    a, b, c = _r("ollama", "a", 3.0), _r("ollama", "b", 2.0), _r("vllm", "c", 1.0)
    out = decide_evictions([a, b, c], cap=1, desired_server="ollama")
    assert a not in out and set(out) == {b, c}


# ── the enforcer: hard-kill + RAISE ──────────────────────────────────────────


def _enforcer(residents, *, cap=1, desired="ollama", collect_error=None):
    kills = {"ollama": [], "vllm": 0}

    def collect():
        if collect_error:
            raise collect_error
        return list(residents)

    return model_residency.ModelResidencyEnforcer(
        collect_fn=collect,
        unload_ollama_fn=lambda models: kills["ollama"].append(list(models)),
        stop_vllm_fn=lambda: kills.__setitem__("vllm", kills["vllm"] + 1),
        desired_server_fn=lambda: desired,
        cap=cap,
        label="box-a-gpu",
    ), kills


def test_violation_kills_extras_and_raises():
    old, new = _r("ollama", "old", 1.0), _r("ollama", "new", 2.0)
    enf, kills = _enforcer([old, new])
    with pytest.raises(ModelResidencyViolation) as e:
        enf.enforce_once()
    assert kills["ollama"] == [["old"]]          # extra hard-killed
    assert "old" in str(e.value) and "new" in str(e.value) and "box-a-gpu" in str(e.value)


def test_violation_across_servers_stops_vllm_when_ollama_desired():
    ol, vl = _r("ollama", "llama3", 2.0), _r("vllm", "vlm-x", 1.0)
    enf, kills = _enforcer([ol, vl], desired="ollama")
    with pytest.raises(ModelResidencyViolation):
        enf.enforce_once()
    assert kills["vllm"] == 1 and kills["ollama"] == []


def test_violation_evicts_ollama_when_vllm_desired():
    ol, vl = _r("ollama", "llama3", 2.0), _r("vllm", "vlm-x", 1.0)
    enf, kills = _enforcer([ol, vl], desired="vllm")
    with pytest.raises(ModelResidencyViolation):
        enf.enforce_once()
    assert kills["ollama"] == [["llama3"]] and kills["vllm"] == 0


def test_clean_box_neither_kills_nor_raises():
    enf, kills = _enforcer([_r("ollama", "llama3")])
    assert enf.enforce_once() == []
    assert kills["ollama"] == [] and kills["vllm"] == 0


def test_collector_failure_never_breaks_the_worker():
    enf, kills = _enforcer([], collect_error=OSError("server down"))
    assert enf.enforce_once() == []              # no raise, no kill
    assert kills["ollama"] == [] and kills["vllm"] == 0


def test_kill_failure_still_raises_the_violation():
    old, new = _r("ollama", "old", 1.0), _r("ollama", "new", 2.0)

    def boom(models):
        raise OSError("unload failed")

    enf = model_residency.ModelResidencyEnforcer(
        collect_fn=lambda: [old, new], unload_ollama_fn=boom,
        stop_vllm_fn=lambda: None, desired_server_fn=lambda: "ollama",
        cap=1, label="x",
    )
    with pytest.raises(ModelResidencyViolation):
        enf.enforce_once()


def test_daemon_tick_logs_the_violation_instead_of_dying(caplog):
    old, new = _r("ollama", "old", 1.0), _r("ollama", "new", 2.0)
    enf, kills = _enforcer([old, new])
    with caplog.at_level(logging.ERROR):
        enf.tick()                               # the loop body: catch + ERROR log
    assert kills["ollama"] == [["old"]]
    assert any("MODEL RESIDENCY VIOLATION" in r.getMessage() or
               "ModelResidencyViolation" in (r.exc_text or "") for r in caplog.records)


def test_cap_env_default_is_one(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_BOX_MODEL_CAP", raising=False)
    import importlib
    m = importlib.reload(model_residency)
    assert m.DEFAULT_BOX_MODEL_CAP == 1


# ── llm_probe collectors the enforcer wires ──────────────────────────────────


def test_loaded_models_info_parses_names_and_recency():
    from queue_workflows import llm_probe
    ps = {"models": [
        {"name": "a", "expires_at": "2026-07-16T11:00:00.123456789Z"},
        {"name": "b", "expires_at": "2026-07-16T12:00:00Z"},
    ]}
    got = llm_probe.loaded_models_info("http://box:11434", get_json_fn=lambda u, t: ps)
    assert [m["name"] for m in got] == ["a", "b"]
    assert got[1]["mru"] > got[0]["mru"] > 0     # later expiry ⇒ more recently used


def test_loaded_models_info_never_raises():
    from queue_workflows import llm_probe
    def boom(u, t):
        raise OSError("down")
    assert llm_probe.loaded_models_info("http://box:11434", get_json_fn=boom) == []
    assert llm_probe.loaded_models_info("", get_json_fn=lambda u, t: {}) == []


def test_vllm_served_models_reads_openai_route():
    from queue_workflows import llm_probe
    data = {"data": [{"id": "vlm-x"}]}
    got = llm_probe.vllm_served_models("http://box:8000", get_json_fn=lambda u, t: data)
    assert got == ["vlm-x"]
    def boom(u, t):
        raise OSError("down")
    assert llm_probe.vllm_served_models("http://box:8000", get_json_fn=boom) == []


# ── LAST DEFENCE LINE: assert_can_load (raise if the box already holds a model) ─
#
# The residency ENFORCER above evicts an already-loaded second model after the
# fact. assert_can_load is the gate BEFORE the fact: called at the moment a loader
# is about to pull weights into VRAM, it refuses if the box already holds a
# DIFFERENT model — the last backstop that stops the two-models-resident state from
# ever forming, even for a model the cooperative lease never saw (ollama; a project
# that doesn't share the lease file).


def test_assert_can_load_ok_on_empty_box():
    model_residency.assert_can_load("llama3", [], label="box-a-gpu")  # no raise


def test_assert_can_load_ok_when_same_model_already_resident():
    # Re-serving / reloading the one model the box already holds is not a violation.
    model_residency.assert_can_load(
        "llama3", [_r("ollama", "llama3", 5.0), _r("ollama", "llama3", 9.0)],
        label="box-a-gpu",
    )


def test_assert_can_load_raises_on_a_foreign_model():
    with pytest.raises(model_residency.ModelAlreadyLoadedError) as e:
        model_residency.assert_can_load(
            "sdxl", [_r("ollama", "llama3", 5.0)], label="box-a-gpu",
        )
    msg = str(e.value)
    assert "sdxl" in msg and "llama3" in msg and "box-a-gpu" in msg


def test_assert_can_load_raises_when_any_resident_is_foreign():
    with pytest.raises(model_residency.ModelAlreadyLoadedError):
        model_residency.assert_can_load(
            "llama3",
            [_r("ollama", "llama3", 5.0), _r("vllm", "other", 1.0)],  # one foreign
            label="box-a-gpu",
        )


def test_model_already_loaded_error_is_a_residency_violation():
    # So a single `except ModelResidencyViolation` catches both the enforcer's and
    # the load-gate's refusals.
    assert issubclass(model_residency.ModelAlreadyLoadedError,
                      model_residency.ModelResidencyViolation)
