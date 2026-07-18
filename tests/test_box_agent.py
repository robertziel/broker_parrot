"""Per-box fleet supervisor — ONE source of truth for what runs on a box.

The fleet's bug (2026-07-18): a box's lanes could be parked THREE independent ways —
commented-out in the box's compose (host-level, invisible to everything), an OFF row
in ``worker_controls`` (the DB desired-state the panel reads), or simply not beating
(``worker_heartbeats``). Nothing reconciled them, so a render worker sat compose-
disabled while ``worker_controls`` still said ``on`` and the panel showed the box up.

The box-agent collapses that to one rule: **``worker_controls`` is the source of
truth, and the agent makes the box's actual running lanes MATCH it** — start a lane
that should be on and isn't, stop one that should be off, restart one that died.
These tests pin the reconcile contract; the runner (subprocess spawn/kill + the 30 s
loop) is a thin shell over this pure core.
"""
from __future__ import annotations

from queue_workflows import box_agent
from queue_workflows.box_agent import Lane


def _lanes():
    return [
        Lane(key="app_a:gpu", project="app_a", queue="gpu"),
        Lane(key="app_b:gpu", project="app_b", queue="gpu"),
        Lane(key="app_c:cpu", project="app_c", queue="cpu"),
    ]


# ── the pure reconcile: actual → desired ──────────────────────────────────────


def test_reconcile_starts_a_desired_on_lane_that_is_not_running():
    lanes = _lanes()
    start, stop = box_agent.reconcile(lanes, desired={}, running=set())
    assert {l.key for l in start} == {l.key for l in lanes}   # absent desired ⇒ ON
    assert stop == []


def test_reconcile_stops_a_desired_off_lane_that_is_running():
    lanes = _lanes()
    desired = {"app_b:gpu": False}
    running = {"app_a:gpu", "app_b:gpu", "app_c:cpu"}
    start, stop = box_agent.reconcile(lanes, desired=desired, running=running)
    assert start == []
    assert [l.key for l in stop] == ["app_b:gpu"]         # only the OFF one


def test_reconcile_is_a_noop_when_actual_already_matches_desired():
    lanes = _lanes()
    desired = {"app_b:gpu": False}
    running = {"app_a:gpu", "app_c:cpu"}               # off lane already down
    start, stop = box_agent.reconcile(lanes, desired=desired, running=running)
    assert start == [] and stop == []


def test_reconcile_absent_control_row_means_ON_matching_engine_default():
    # the engine treats a worker absent from worker_controls as ON; the agent must
    # agree, else a box with no control rows would never start anything.
    lanes = _lanes()
    start, _ = box_agent.reconcile(lanes, desired={"app_a:gpu": True}, running=set())
    assert {l.key for l in start} == {l.key for l in lanes}


# ── mapping worker_controls rows → per-lane desired on/off ─────────────────────


def test_desired_from_controls_off_row_parks_the_matching_lane():
    lanes = _lanes()
    rows = [("box-a-gpu", "gpu", "app_b", "off")]         # (host_label,queue,project,state)
    desired = box_agent.desired_from_controls(rows, host="box-a", lanes=lanes)
    assert desired["app_b:gpu"] is False
    assert desired["app_a:gpu"] is True                  # unaffected lane stays on


def test_desired_from_controls_matches_bare_box_or_lane_label():
    # worker_controls may be keyed by the bare box ('box-a') OR the lane label
    # ('box-a-gpu'); a control on either must govern the box's lanes of that queue.
    lanes = _lanes()
    bare = box_agent.desired_from_controls([("box-a", "gpu", "", "off")], host="box-a", lanes=lanes)
    assert bare["app_a:gpu"] is False and bare["app_b:gpu"] is False   # all gpu lanes
    assert bare["app_c:cpu"] is True                                          # cpu untouched


def test_desired_from_controls_defaults_to_on_with_no_rows():
    lanes = _lanes()
    desired = box_agent.desired_from_controls([], host="box-a", lanes=lanes)
    assert all(desired[l.key] for l in lanes)


# ── the agent tick: reconcile + restart the dead ──────────────────────────────


class _FakeProc:
    def __init__(self):
        self._alive = True

    def poll(self):
        return None if self._alive else 1        # Popen contract: None = running

    def die(self):
        self._alive = False


def _agent(lanes, controls):
    started, stopped = [], []

    def spawn(lane):
        started.append(lane.key)
        return _FakeProc()

    def kill(lane, proc):
        stopped.append(lane.key)

    a = box_agent.BoxAgent(
        host="box-a", lanes=lanes,
        fetch_controls=lambda: controls, spawn_fn=spawn, kill_fn=kill,
    )
    return a, started, stopped


def test_agent_first_tick_starts_every_on_lane():
    lanes = _lanes()
    a, started, _ = _agent(lanes, controls=[])
    a.tick()
    assert set(started) == {l.key for l in lanes}


def test_agent_restarts_a_lane_whose_process_died():
    lanes = _lanes()
    a, started, _ = _agent(lanes, controls=[])
    a.tick()                                     # start all
    started.clear()
    a.procs["app_a:gpu"].die()              # its worker crashed
    a.tick()                                     # reconcile: dead ⇒ not running ⇒ restart
    assert started == ["app_a:gpu"]         # only the dead one


def test_agent_stops_a_lane_that_gets_parked_off():
    lanes = _lanes()
    a, started, stopped = _agent(lanes, controls=[])
    a.tick()
    a.controls_source = [("box-a", "gpu", "app_b", "off")]  # operator parks it
    a.fetch_controls = lambda: a.controls_source
    a.tick()
    assert stopped == ["app_b:gpu"]
    assert "app_b:gpu" not in a.procs        # no longer tracked as running


# ── manifest parse (the declarative replacement for the box's compose pile) ───


def test_load_manifest_parses_lanes_and_control_dsns(tmp_path):
    import json
    p = tmp_path / "box.json"
    p.write_text(json.dumps({"lanes": [
        {"key": "app_a:gpu", "project": "app_a", "queue": "gpu",
         "argv": ["python", "-m", "queue_workflows.claim_worker", "--queue=gpu"],
         "env": {"QUEUE_WORKFLOWS_DB_URL": "postgresql://x/lm"},
         "control_dsn": "postgresql://x/lm"},
        {"key": "app_b:gpu", "project": "app_b", "queue": "gpu",
         "argv": ["python", "-m", "gen_workflows.run_worker", "--queue", "gpu"],
         "control_dsn": "postgresql://x/broker"},
    ]}))
    lanes, dsns = box_agent.load_manifest(str(p))
    assert [l.key for l in lanes] == ["app_a:gpu", "app_b:gpu"]
    assert lanes[0].argv[-1] == "--queue=gpu"
    assert lanes[0].env["QUEUE_WORKFLOWS_DB_URL"].endswith("/lm")
    assert dsns["app_b:gpu"].endswith("/broker")


def test_load_manifest_preserves_env_null_as_none(tmp_path):
    # a null env value in the manifest means "REMOVE this key from the inherited
    # environment" — needed so a lane running a host app's baked engine install
    # doesn't inherit the agent's own PYTHONPATH (the NFS engine tree).
    import json
    p = tmp_path / "box.json"
    p.write_text(json.dumps({"lanes": [
        {"key": "a:gpu", "queue": "gpu", "argv": ["x"], "env": {"PYTHONPATH": None}},
    ]}))
    lanes, _ = box_agent.load_manifest(str(p))
    assert lanes[0].env == {"PYTHONPATH": None}


def test_lane_env_null_removes_the_inherited_key(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/nfs/engine-tree")
    monkeypatch.setenv("KEEP_ME", "yes")
    lane = Lane(key="a:gpu", project="a", queue="gpu",
                env={"PYTHONPATH": None, "EXTRA": "1"})
    env = box_agent._lane_env(lane)
    assert "PYTHONPATH" not in env           # null ⇒ removed, baked install stays clean
    assert env["KEEP_ME"] == "yes"           # untouched inherited keys survive
    assert env["EXTRA"] == "1"               # lane-set keys land


# ── worker_controls read tolerates a pre-0019 host DB (no project column) ─────


class _FakeCur:
    def __init__(self, fail_first):
        self.fail_first, self.calls, self._rows = fail_first, [], []

    def execute(self, sql):
        self.calls.append(sql)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError('column "project" does not exist')
        self._rows = [("box-a-gpu", "gpu", "", "off")]

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


def test_read_worker_controls_falls_back_when_project_column_is_missing():
    conn, cur = _FakeConn(), _FakeCur(fail_first=True)
    rows = box_agent._read_worker_controls(conn, cur)
    assert rows == [("box-a-gpu", "gpu", "", "off")]
    assert conn.rollbacks == 1               # aborted txn reset before the retry
    assert "COALESCE(project" in cur.calls[0] and "project" not in cur.calls[1]


def test_read_worker_controls_modern_schema_single_query():
    conn, cur = _FakeConn(), _FakeCur(fail_first=False)
    rows = box_agent._read_worker_controls(conn, cur)
    assert rows == [("box-a-gpu", "gpu", "", "off")]
    assert len(cur.calls) == 1 and conn.rollbacks == 0


def test_shutdown_stops_every_lane_then_exits():
    # docker stop TERMs the agent (container PID 1); it must forward the stop to
    # every lane so claim workers drain their in-flight job instead of being
    # KILLed with the container at the end of the grace period.
    lanes = _lanes()
    a, _started, stopped = _agent(lanes, controls=[])
    a.tick()
    import pytest
    with pytest.raises(SystemExit):
        box_agent._shutdown(a, kill_fn=a.kill_fn)
    assert set(stopped) == {l.key for l in lanes}   # every running lane stopped
    assert a.procs == {}                            # nothing left tracked
