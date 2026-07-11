"""Broker web service + panel — real HTTP against an in-thread server.

The v2 broker exposes its orchestration core over HTTP: a worker/broker JSON API
(``/api/ask`` etc.) and a read-only operator panel (``GET /``). These drive a real
``ThreadingHTTPServer`` on an ephemeral port (the shared SQLite connection is
``check_same_thread=False``, so handler threads share the store safely) — no
Postgres server needed. NB: never hold a ``db.connection()`` while making an HTTP
call here (the request thread needs the same RLock — that would deadlock); assert
via HTTP responses / separate reads instead.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from queue_workflows import broker_service as bs
from queue_workflows import db
from queue_workflows.broker_service import web as _web
from queue_workflows.broker_service.web import BrokerWebHandler


@pytest.fixture()
def base_url():
    bs.ensure_schema()
    with db.connection() as conn, conn.cursor() as cur:
        for tbl in ("bw_jobs", "bw_workers"):
            cur.execute(f"DELETE FROM {tbl}")
        conn.commit()
    # Exercise the whole suite through the bounded pool the service actually uses.
    httpd = _web._BoundedThreadingHTTPServer(("127.0.0.1", 0), BrokerWebHandler, max_workers=4)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post(base, path, obj):
    req = urllib.request.Request(
        base + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read() or b"{}")


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return r.status, r.read().decode()


def test_healthz(base_url):
    status, body = _get(base_url, "/healthz")
    assert status == 200 and body.strip() == "ok"


def test_submit_ask_finish_flow(base_url):
    _, sub = _post(base_url, "/api/submit", {"project": "A", "resource": "cpu", "payload": {"n": 7}})
    job_id = sub["job_id"]
    _, ask = _post(base_url, "/api/ask", {"worker_id": "wA", "project": "A", "resource": "cpu"})
    assert ask["granted"] is True
    assert ask["job"]["job_id"] == job_id and ask["job"]["payload"] == {"n": 7}
    _, fin = _post(base_url, "/api/finish", {"job_id": job_id, "worker_id": "wA", "result": {"ok": True}})
    assert fin["job"]["status"] == "done"


def test_ask_denied_when_capacity_full(base_url):
    _post(base_url, "/api/submit", {"project": "A", "resource": "cpu"})
    _, ask = _post(base_url, "/api/ask", {"worker_id": "wA", "project": "A", "resource": "cpu", "capacity": 0})
    assert ask["granted"] is False and ask["job"] is None


def test_revoke_requeues(base_url):
    _, sub = _post(base_url, "/api/submit", {"project": "A", "resource": "cpu"})
    job_id = sub["job_id"]
    _post(base_url, "/api/ask", {"worker_id": "wA", "project": "A", "resource": "cpu"})
    _, rev = _post(base_url, "/api/revoke", {"job_id": job_id, "requeue": True, "reason": "operator"})
    assert rev["job"]["status"] == "queued"
    _, jobs = _post(base_url, "/api/ask", {"worker_id": "wA2", "project": "A", "resource": "cpu"})
    assert jobs["job"]["job_id"] == job_id  # requeued job is grantable again


def test_heartbeat_and_renew(base_url):
    _, sub = _post(base_url, "/api/submit", {"project": "A", "resource": "gpu"})
    _post(base_url, "/api/ask", {"worker_id": "wA", "project": "A", "resource": "gpu"})
    _, hb = _post(base_url, "/api/heartbeat", {"worker_id": "wA"})
    assert hb["ok"] is True
    _, rn = _post(base_url, "/api/renew", {"job_id": sub["job_id"], "worker_id": "wA"})
    assert rn["permitted"] is True


def test_missing_field_is_400(base_url):
    req = urllib.request.Request(
        base_url + "/api/ask", data=json.dumps({"worker_id": "wA"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400


def test_panel_renders_the_queue(base_url):
    _post(base_url, "/api/submit", {"project": "Alpha", "resource": "cpu", "payload": {"k": 1}})
    _post(base_url, "/api/ask", {"worker_id": "worker-1", "project": "Alpha", "resource": "cpu"})
    status, html_body = _get(base_url, "/")
    assert status == 200
    low = html_body.lower()
    assert "<!doctype html>" in low
    assert "broker" in low  # title/header
    assert "Alpha" in html_body  # project label present
    assert "worker-1" in html_body  # the registered worker is shown
    assert "granted" in html_body  # the granted job's status
    assert "<script" not in low  # no-JS invariant (⇒ no console errors)


def test_favicon_served_and_public(base_url):
    # The brand mark is a static SVG served pre-auth (browsers don't attach the
    # bearer token to a favicon fetch), cacheable, and linked from the panel head.
    req = urllib.request.Request(base_url + "/favicon.svg")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
        assert r.headers["Content-Type"] == "image/svg+xml"
        assert "max-age" in (r.headers.get("Cache-Control") or "")
        body = r.read().decode()
    assert body.startswith("<svg") and "</svg>" in body
    # The panel inlines the same mark and links the favicon.
    _, html_body = _get(base_url, "/")
    assert '<link rel="icon" type="image/svg+xml" href="/favicon.svg">' in html_body
    assert "<svg" in html_body and "🦜" not in html_body  # vector logo replaced the emoji


def test_favicon_open_even_with_token(base_url, monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_BROKER_WEB_TOKEN", "s3cret")
    with urllib.request.urlopen(base_url + "/favicon.svg", timeout=5) as r:
        assert r.status == 200  # favicon never needs auth (like /healthz)


def test_panel_project_filter(base_url):
    _post(base_url, "/api/submit", {"project": "Alpha", "resource": "cpu"})
    _post(base_url, "/api/submit", {"project": "Beta", "resource": "cpu"})
    _, filtered = _get(base_url, "/?project=Alpha")
    assert "Alpha" in filtered
    # Beta's job should be filtered out of the jobs table.
    _, all_jobs = _get(base_url, "/")
    assert "Beta" in all_jobs  # unfiltered shows both


def test_capacity_non_int_is_400(base_url):
    _post(base_url, "/api/submit", {"project": "A", "resource": "cpu"})
    req = urllib.request.Request(
        base_url + "/api/ask",
        data=json.dumps({"worker_id": "wA", "project": "A", "resource": "cpu", "capacity": "abc"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400  # malformed capacity is a 400, not a 500


def test_no_auth_required_by_default(base_url):
    # token unset (default) ⇒ open, back-compat.
    assert _get(base_url, "/")[0] == 200
    assert _post(base_url, "/api/submit", {"project": "A", "resource": "cpu"})[0] == 200


def test_auth_required_when_token_set(base_url, monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_BROKER_WEB_TOKEN", "s3cret")
    # no Authorization ⇒ 401
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base_url, "/")
    assert exc.value.code == 401
    # wrong token ⇒ 401
    bad = urllib.request.Request(base_url + "/", headers={"Authorization": "Bearer nope"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(bad, timeout=5)
    assert exc.value.code == 401
    # correct token ⇒ 200 (panel served)
    ok = urllib.request.Request(base_url + "/", headers={"Authorization": "Bearer s3cret"})
    with urllib.request.urlopen(ok, timeout=5) as r:
        assert r.status == 200
    # a POST also requires the token
    unauth_post = urllib.request.Request(
        base_url + "/api/submit", data=json.dumps({"project": "A", "resource": "cpu"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(unauth_post, timeout=5)
    assert exc.value.code == 401


def test_healthz_open_even_with_token(base_url, monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_BROKER_WEB_TOKEN", "s3cret")
    assert _get(base_url, "/healthz") == (200, "ok")  # health probes never need auth


# ── hardening: fail-closed public bind (F1/F2), body cap (F4), sanitized 500 (F5) ──


def test_public_bind_refused_without_token(monkeypatch):
    # A non-loopback bind with no token would be a wide-open control plane: refuse.
    monkeypatch.delenv("QUEUE_WORKFLOWS_BROKER_WEB_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        _web._check_bind_security("0.0.0.0")


def test_public_bind_refused_with_placeholder_token(monkeypatch):
    # The shipped k8s Secret ships "REPLACE_ME"; applied unedited it must NOT pass.
    monkeypatch.setenv("QUEUE_WORKFLOWS_BROKER_WEB_TOKEN", "REPLACE_ME")
    with pytest.raises(RuntimeError):
        _web._check_bind_security("0.0.0.0")


def test_public_bind_refused_with_short_token(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_BROKER_WEB_TOKEN", "s3cret")  # too short to be a real secret
    with pytest.raises(RuntimeError):
        _web._check_bind_security("0.0.0.0")


def test_public_bind_allowed_with_strong_token(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_BROKER_WEB_TOKEN", "x" * _web._MIN_TOKEN_LEN)
    _web._check_bind_security("0.0.0.0")  # no raise ⇒ a strong token unlocks a public bind


def test_loopback_bind_allowed_without_token(monkeypatch):
    # Loopback stays open with no token — back-compat for the default single-box deploy.
    monkeypatch.delenv("QUEUE_WORKFLOWS_BROKER_WEB_TOKEN", raising=False)
    _web._check_bind_security("127.0.0.1")  # no raise


def test_oversized_body_rejected_413(base_url):
    big = json.dumps(
        {"project": "A", "resource": "cpu", "payload": "x" * (_web._MAX_BODY_BYTES + 1)}
    ).encode()
    req = urllib.request.Request(
        base_url + "/api/submit", data=big,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 413  # capped before the body is read into memory


def test_internal_error_is_sanitized_500(base_url, monkeypatch):
    # A backend exception must not echo its message (table/column/DSN details) to the client.
    from queue_workflows.broker_service import orchestrator as _orch

    def boom(*_a, **_k):
        raise RuntimeError("secret detail: bw_jobs internal leak")

    monkeypatch.setattr(_orch, "submit_job", boom)
    req = urllib.request.Request(
        base_url + "/api/submit", data=json.dumps({"project": "A", "resource": "cpu"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 500
    body = exc.value.read().decode()
    assert "secret detail" not in body and "bw_jobs" not in body  # not leaked
    assert "internal" in body.lower()  # generic message instead


# ── bounded worker pool (F4, thread-exhaustion half) ─────────────────────────


def test_max_workers_env_override(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_BROKER_WEB_MAX_WORKERS", "7")
    assert _web._max_workers() == 7
    # malformed / non-positive ⇒ safe default, never a crash or an unbounded pool.
    monkeypatch.setenv("QUEUE_WORKFLOWS_BROKER_WEB_MAX_WORKERS", "nope")
    assert _web._max_workers() == _web._DEFAULT_MAX_WORKERS
    monkeypatch.setenv("QUEUE_WORKFLOWS_BROKER_WEB_MAX_WORKERS", "0")
    assert _web._max_workers() == _web._DEFAULT_MAX_WORKERS


def test_bounded_pool_caps_concurrency_and_serves(base_url):
    # The fixture server was built with max_workers=4; concurrency is bounded there,
    # yet many overlapping requests all still complete correctly.
    results: list[int] = []

    def hit():
        results.append(_get(base_url, "/healthz")[0])

    ts = [threading.Thread(target=hit) for _ in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=5)
    assert results == [200] * 16  # all served through the 4-worker pool


def test_bounded_server_pool_is_capped_and_shuts_down():
    bs.ensure_schema()
    httpd = _web._BoundedThreadingHTTPServer(("127.0.0.1", 0), BrokerWebHandler, max_workers=3)
    try:
        assert httpd._pool._max_workers == 3  # bounded, not thread-per-connection
    finally:
        httpd.server_close()
    assert httpd._pool._shutdown  # server_close() tears the pool down
