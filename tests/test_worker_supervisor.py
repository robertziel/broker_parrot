"""The per-host dead-worker supervisor — the automated consumer of the 0009
``last_flagged_dead_at`` flag. Pure decision logic runs on a virtual clock with
no I/O; the read query runs against the real test DB.
"""
from __future__ import annotations

import queue_workflows
from queue_workflows import node_queue, worker_supervisor as ws
from queue_workflows.db import connection
from queue_workflows.dialect import get_dialect


# ── map parsing ──────────────────────────────────────────────────────────────

def test_parse_map():
    assert ws.parse_map("a:c1, b:c2") == {"a": "c1", "b": "c2"}
    assert ws.parse_map("") == {} and ws.parse_map(None) == {}
    assert ws.parse_map("bad,also-bad,x:y") == {"x": "y"}   # garbage entries skipped


# ── select_bounces: the pure decision ───────────────────────────────────────

def _flag(hl, q="gpu"):
    return {"host_label": hl, "queue": q}


def test_only_owned_labels_are_bounced():
    flagged = [_flag("box-a-gpu"), _flag("box-b-gpu"), _flag("mystery")]
    picks = ws.select_bounces(flagged, {"box-a-gpu": "cont-a", "box-b-gpu": "cont-b"},
                              now=100.0, cooldown_s=300.0, last_bounced={})
    got = {p["host_label"]: p["container"] for p in picks}
    assert got == {"box-a-gpu": "cont-a", "box-b-gpu": "cont-b"}   # 'mystery' unmapped → skipped


def test_never_bounces_unmapped_when_map_empty():
    assert ws.select_bounces([_flag("box-a-gpu")], {}, now=1.0, cooldown_s=1.0, last_bounced={}) == []


def test_cooldown_suppresses_repeat_bounce():
    lm, last = {"box-a-gpu": "c"}, {("box-a-gpu", "gpu"): 100.0}
    # 200s later, still within the 300s cooldown → no bounce
    assert ws.select_bounces([_flag("box-a-gpu")], lm, 300.0, 300.0, last) == []
    # 400s later → past cooldown → bounce again
    assert len(ws.select_bounces([_flag("box-a-gpu")], lm, 500.0, 300.0, last)) == 1


# ── the tick loop (fakes for DB + bounce) ────────────────────────────────────

def test_tick_bounces_and_records_cooldown():
    calls = []
    sup = ws.WorkerSupervisor(
        label_map={"box-a-gpu": "worker-cont-1"}, cooldown_s=300.0,
        list_fn=lambda: [_flag("box-a-gpu")],
        bounce_fn=lambda hl, q, c: calls.append((hl, q, c)) or True,
        now_fn=lambda: 1000.0,
    )
    picks = sup.tick()
    assert calls == [("box-a-gpu", "gpu", "worker-cont-1")]
    assert picks[0]["container"] == "worker-cont-1"
    # a second tick at the same time is suppressed by the recorded cooldown
    assert sup.tick() == []
    assert len(calls) == 1


def test_tick_report_only_without_map():
    calls = []
    sup = ws.WorkerSupervisor(label_map={}, list_fn=lambda: [_flag("box-a-gpu")],
                              bounce_fn=lambda *a: calls.append(a) or True)
    assert sup.tick() == [] and calls == []      # flagged but no map → touches nothing


def test_tick_swallows_db_errors():
    def boom():
        raise RuntimeError("db down")
    sup = ws.WorkerSupervisor(label_map={"x": "c"}, list_fn=boom, bounce_fn=lambda *a: True)
    assert sup.tick() == []                       # best-effort: no crash


def test_injected_config_hook_is_used(monkeypatch):
    seen = []
    queue_workflows.set_worker_bounce(lambda hl, q, c: seen.append(hl) or True)
    try:
        sup = ws.WorkerSupervisor(label_map={"box-a-gpu": "c"}, list_fn=lambda: [_flag("box-a-gpu")],
                                  now_fn=lambda: 0.0)
        sup.tick()
        assert seen == ["box-a-gpu"]
    finally:
        queue_workflows.set_worker_bounce(None)


# ── default bounce builds the right docker argv (no real docker) ─────────────

def test_default_bounce_runs_docker_restart():
    argv = {}
    class P:
        returncode = 0
    def fake_run(cmd, **kw):
        argv["cmd"] = cmd
        return P()
    ok = ws._default_bounce("box-a-gpu", "gpu", "worker-cont-1",
                            which=lambda _: "/usr/bin/docker", run=fake_run)
    assert ok is True
    assert argv["cmd"] == ["/usr/bin/docker", "restart", "worker-cont-1"]


def test_default_bounce_false_when_docker_missing():
    assert ws._default_bounce("h", "gpu", "c", which=lambda _: None) is False


# ── the read query against the real test DB ──────────────────────────────────

def test_flagged_dead_workers_returns_only_still_stale_flagged():
    # a fresh, unflagged worker
    node_queue.upsert_worker_heartbeat(host_label="alive", queue="gpu", concurrency=1)
    # a worker with a stale last_seen AND a recent dead flag → should be returned
    node_queue.upsert_worker_heartbeat(host_label="wedged", queue="gpu", concurrency=1)
    with connection() as conn, conn.cursor() as cur:
        d = get_dialect()
        cur.execute(
            "UPDATE worker_heartbeats SET last_seen = " + d.past_seconds("300")
            + ", last_flagged_dead_at = " + d.past_seconds("60")
            + " WHERE host_label = 'wedged'")
        conn.commit()
    rows = node_queue.flagged_dead_workers(within_s=1800)
    labels = {r["host_label"] for r in rows}
    assert "wedged" in labels          # flagged + still stale
    assert "alive" not in labels       # never flagged, beating


def test_flagged_dead_workers_excludes_recovered_worker():
    # flagged in the past, but has since beaten (fresh last_seen) → NOT returned
    node_queue.upsert_worker_heartbeat(host_label="recovered", queue="gpu", concurrency=1)
    with connection() as conn, conn.cursor() as cur:
        d = get_dialect()
        cur.execute("UPDATE worker_heartbeats SET last_flagged_dead_at = "
                    + d.past_seconds("60") + " WHERE host_label = 'recovered'")
        conn.commit()
    rows = node_queue.flagged_dead_workers(within_s=1800)
    assert "recovered" not in {r["host_label"] for r in rows}   # fresh heartbeat = all-clear
