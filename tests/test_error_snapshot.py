"""``node_queue.error_snapshot`` — the project-scoped queue ERROR LOG (the read twin of
``fleet_snapshot``), plus the migration-0018 ``project`` stamp on ``workflow_node_events``."""

from __future__ import annotations

from queue_workflows import node_queue
from tests._helpers import make_run


def _ev(run_id, *, event_type, project=None, error=None, node="n", host="host-a"):
    eid = node_queue.record_node_event(
        run_id=run_id, node_id=node, event_type=event_type,
        project=project, error=error, host_label=host,
    )
    assert eid is not None, f"record_node_event swallowed a failure for {event_type!r}"
    return eid


def test_error_snapshot_filters_by_project_and_kind():
    run = make_run(workflow_name="_err1")
    _ev(run, event_type="failed", project="ai_leads", error="boom A")
    _ev(run, event_type="error", project="ai_leads", error="boom B")
    _ev(run, event_type="completed", project="ai_leads")            # not an error -> excluded
    _ev(run, event_type="failed", project="project-b", error="other-proj")

    ai = node_queue.error_snapshot(project="ai_leads")
    assert {e["error"] for e in ai} == {"boom A", "boom B"}
    assert all(e["project"] == "ai_leads" for e in ai)
    assert all(e["event_type"] in ("failed", "error") for e in ai)

    allp = {e["error"] for e in node_queue.error_snapshot()}        # all projects
    assert {"boom A", "boom B", "other-proj"} <= allp

    p3 = node_queue.error_snapshot(project="project-b")             # isolation
    assert {e["error"] for e in p3} == {"other-proj"}


def test_error_snapshot_newest_first_and_limit():
    run = make_run(workflow_name="_err2")
    for i in range(5):
        _ev(run, event_type="failed", project="p", error=f"e{i}")
    rows = node_queue.error_snapshot(project="p", limit=3)
    assert [r["error"] for r in rows] == ["e4", "e3", "e2"]         # newest-first, capped


def test_error_snapshot_kinds_widens_to_trips():
    run = make_run(workflow_name="_err3")
    _ev(run, event_type="failed", project="p", error="fail")
    _ev(run, event_type="stall_trip", project="p", error="stalled")
    assert {e["error"] for e in node_queue.error_snapshot(project="p")} == {"fail"}
    widened = node_queue.error_snapshot(project="p", kinds=("failed", "stall_trip"))
    assert {e["error"] for e in widened} == {"fail", "stalled"}


def test_record_node_event_defaults_project_to_config():
    # project=None -> _project(None) -> config.project (default ''), visible under project=''
    run = make_run(workflow_name="_err4")
    _ev(run, event_type="failed", error="defproj")
    assert "defproj" in {r["error"] for r in node_queue.error_snapshot(project="")}


def test_error_snapshot_host_filter():
    run = make_run(workflow_name="_err5")
    _ev(run, event_type="failed", project="p", error="on-host-b", host="host-b")
    _ev(run, event_type="failed", project="p", error="on-host-a", host="host-a")
    rows = node_queue.error_snapshot(project="p", host_label="host-b")
    assert {e["error"] for e in rows} == {"on-host-b"}
