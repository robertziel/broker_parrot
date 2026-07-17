"""A BOX-ADDRESSED control row: ``(<box>, queue)`` governs ``<box>-<lane>`` workers.

THE INCIDENT THIS PINS. Deployments label per-lane workers ``<box>-<lane>``
(``box-c-gpu``), but an operator UI parks a *machine*: it wrote
``worker_controls (host_label='box-c', queue='gpu', desired_state='off')`` — keyed by
the bare box name. The worker's control lookup matched only its exact label, found no
row, and treated absent-as-ON — so the panel showed PARKED while the worker kept
claiming GPU jobs. The UI's *read* side mapped the row to the box card by substring,
which is exactly what made the lie invisible.

So the engine now honors the box address: when no exact ``(host_label, queue)`` row
exists, a row keyed by the label with ONE trailing ``-<segment>`` stripped
(``box-c-gpu`` → ``box-c``) applies. An exact row always wins — the box row is a
fallback, not an override — and a worker with no ``-`` in its label is unaffected.
``queue`` still filters as before, so ``(box-c, cpu)`` never parks ``box-c-gpu``.
Same rule for the LLM config lookup (the same UI writes those rows the same way).
"""

from __future__ import annotations

from queue_workflows import worker_control
from queue_workflows.worker_control import (
    EXIT_CONTROL_HARD_STOP,
    WorkerControlWatcher,
    desired_state_for,
    get_worker_control,
    set_llm_config,
    set_worker_control,
)


# ── get_worker_control / desired_state_for ────────────────────────────────────


def test_box_row_governs_lane_labelled_worker():
    set_worker_control("box-c", "gpu", desired_state="off")
    row = get_worker_control("box-c-gpu", "gpu")
    assert row is not None and row["desired_state"] == "off"
    assert desired_state_for("box-c-gpu", "gpu") == "off"


def test_exact_row_always_wins_over_box_row():
    set_worker_control("box-c", "gpu", desired_state="off")      # box says OFF...
    set_worker_control("box-c-gpu", "gpu", desired_state="on")   # ...lane says ON
    assert desired_state_for("box-c-gpu", "gpu") == "on"


def test_only_one_trailing_segment_is_stripped():
    # 'a-b-gpu' falls back to 'a-b' — never all the way to 'a'.
    set_worker_control("a", "gpu", desired_state="off")
    assert get_worker_control("a-b-gpu", "gpu") is None
    assert desired_state_for("a-b-gpu", "gpu") == "on"


def test_queue_still_filters_the_box_row():
    set_worker_control("box-c", "cpu", desired_state="off")
    assert desired_state_for("box-c-gpu", "gpu") == "on"


def test_dashless_label_has_no_box_fallback():
    assert get_worker_control("solo", "gpu") is None
    assert desired_state_for("solo", "gpu") == "on"


# ── the watcher trips on a box row (the end-to-end park path) ────────────────


class _FakeWorker:
    queue = "gpu"
    host = "box-c-gpu"

    def requeue_inflight_for_control(self):
        return 0


def test_watcher_hard_stops_on_box_addressed_off_row():
    set_worker_control("box-c", "gpu", desired_state="off", stop_policy="hard")
    exits: list[int] = []
    w = WorkerControlWatcher(worker=_FakeWorker(), on_exit=exits.append)
    assert w.check_once() is True
    assert exits == [EXIT_CONTROL_HARD_STOP]


# ── llm_config_for honors the box address the same way ───────────────────────


def test_llm_config_box_row_fallback():
    set_llm_config("box-c", "gpu", server_type="vllm", parallelism=4)
    cfg = worker_control.llm_config_for("box-c-gpu", "gpu")
    assert cfg.server_type == "vllm"
    assert cfg.parallelism == 4


def test_llm_config_exact_row_wins():
    set_llm_config("box-c", "gpu", server_type="vllm", parallelism=4)
    set_llm_config("box-c-gpu", "gpu", server_type="ollama", parallelism=1)
    cfg = worker_control.llm_config_for("box-c-gpu", "gpu")
    assert cfg.server_type == "ollama"
    assert cfg.parallelism == 1
