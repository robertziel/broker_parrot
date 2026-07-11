"""``worker_controls`` is project-keyed (migration 0019).

0017 pooled the queue onto one shared broker and re-keyed ``worker_heartbeats``
to ``(host_label, queue, project)`` — but left ``worker_controls`` on the old
2-col PK. On a shared broker ``host_label`` is not globally unique: two projects
run a worker on the SAME machine + queue (live: ``host-b`` runs both ai_leads'
and project-b's gpu worker). Sharing one control row means

  * an operator OFF for project A hard-stops project B's worker, and
  * an LLM-config write for A silently reconfigures B's LLM server.

These tests pin the control plane to the same identity as the heartbeat it
controls, and pin the single-tenant (``project=''``) default as unchanged.
"""

from __future__ import annotations

import pytest

import queue_workflows
from queue_workflows import worker_control
from queue_workflows.config import get_config


@pytest.fixture
def _two_projects():
    """Restore the ambient project after each test — ``configure`` is global."""
    original = get_config().project
    yield
    queue_workflows.configure(project=original)


# ── ON/OFF isolation ─────────────────────────────────────────────────────────


def test_off_for_one_project_does_not_park_the_other():
    """The finding this migration exists for."""
    worker_control.set_worker_control(
        "host-b", "gpu", desired_state="off", project="ai_leads",
    )
    worker_control.set_worker_control(
        "host-b", "gpu", desired_state="on", project="project-b",
    )

    assert worker_control.desired_state_for("host-b", "gpu", project="ai_leads") == "off"
    assert worker_control.desired_state_for("host-b", "gpu", project="project-b") == "on"


def test_two_projects_hold_distinct_rows_on_one_host_and_queue():
    worker_control.set_worker_control(
        "host-b", "gpu", desired_state="off", requested_by="a", project="ai_leads",
    )
    worker_control.set_worker_control(
        "host-b", "gpu", desired_state="off", requested_by="b", project="project-b",
    )
    a = worker_control.get_worker_control("host-b", "gpu", project="ai_leads")
    b = worker_control.get_worker_control("host-b", "gpu", project="project-b")
    assert a["requested_by"] == "a"
    assert b["requested_by"] == "b"


def test_upsert_is_idempotent_within_a_project():
    for state in ("off", "on", "off"):
        worker_control.set_worker_control(
            "h", "cpu", desired_state=state, project="p",
        )
    assert worker_control.desired_state_for("h", "cpu", project="p") == "off"


# ── LLM-config isolation (the columns that live on this table) ───────────────


def test_llm_config_for_one_project_does_not_reconfigure_the_other():
    worker_control.set_llm_config(
        "host-b", "gpu", server_type="vllm", parallelism=128, project="ai_leads",
    )
    worker_control.set_llm_config(
        "host-b", "gpu", server_type="ollama", parallelism=1, project="project-b",
    )

    a = worker_control.llm_config_for("host-b", "gpu", project="ai_leads")
    b = worker_control.llm_config_for("host-b", "gpu", project="project-b")
    assert (a.server_type, a.parallelism) == ("vllm", 128)
    assert (b.server_type, b.parallelism) == ("ollama", 1)


def test_llm_config_partial_update_stays_within_its_project():
    worker_control.set_llm_config("h", "gpu", server_type="vllm", parallelism=64, project="a")
    worker_control.set_llm_config("h", "gpu", server_type="ollama", project="b")
    # A partial write to 'b' must not touch 'a'.
    worker_control.set_llm_config("h", "gpu", parallelism=4, project="b")

    a = worker_control.llm_config_for("h", "gpu", project="a")
    b = worker_control.llm_config_for("h", "gpu", project="b")
    assert (a.server_type, a.parallelism) == ("vllm", 64)
    assert (b.server_type, b.parallelism) == ("ollama", 4)


def test_llm_config_write_does_not_disturb_desired_state():
    """Soft config change: must never park a running worker (0013 contract)."""
    worker_control.set_worker_control("h", "gpu", desired_state="on", project="p")
    worker_control.set_llm_config("h", "gpu", server_type="vllm", project="p")
    assert worker_control.desired_state_for("h", "gpu", project="p") == "on"


# ── single-tenant default is unchanged ───────────────────────────────────────


def test_project_defaults_to_the_ambient_config_project(_two_projects):
    queue_workflows.configure(project="alpha")
    worker_control.set_worker_control("h", "cpu", desired_state="off")
    # Written under 'alpha' ⇒ invisible to another tenant, visible to alpha.
    assert worker_control.desired_state_for("h", "cpu") == "off"
    assert worker_control.desired_state_for("h", "cpu", project="beta") == "on"


def test_single_tenant_sentinel_is_the_empty_string(_two_projects):
    queue_workflows.configure(project="")
    worker_control.set_worker_control("h", "cpu", desired_state="off")
    row = worker_control.get_worker_control("h", "cpu", project="")
    assert row is not None and row["project"] == ""


def test_missing_row_is_default_on():
    assert worker_control.desired_state_for("nobody", "gpu", project="p") == "on"
    assert worker_control.llm_config_for("nobody", "gpu", project="p") == worker_control.LLMConfig()
