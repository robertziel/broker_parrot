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
import re
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


# ── panel UX pass: chips, filters, responsive,
#    refresh control, contrast, semantics, scannability ──────────────────────


def test_panel_project_chips_survive_high_volume(base_url):
    """Filter chips must come from a DISTINCT-project view, not the newest-500 jobs:
    a project whose only job is old still gets a chip (its jobs still count in the
    KPIs, so a missing chip is an unreachable filter) — and the capped jobs table
    says how much it is hiding instead of silently truncating."""
    from queue_workflows.broker_service import orchestrator as _orch

    _orch.submit_job(project="zz-ancient", resource="cpu")
    for _ in range(501):
        _orch.submit_job(project="alpha", resource="cpu")
    _, html_body = _get(base_url, "/")
    assert 'href="/?project=zz-ancient"' in html_body  # chip survives 501 newer jobs
    assert "newest 50 of 502" in html_body  # the cap is visible, not silent


def test_panel_workers_respect_project_filter(base_url):
    """``?project=`` scopes the whole panel. The workers table (and /api/workers)
    used to ignore it — a filtered view silently showed every tenant's fleet."""
    _post(base_url, "/api/ask", {"worker_id": "w-alpha", "project": "Alpha", "resource": "cpu"})
    _post(base_url, "/api/ask", {"worker_id": "w-beta", "project": "Beta", "resource": "cpu"})
    _, filtered = _get(base_url, "/?project=Alpha")
    assert "w-alpha" in filtered and "w-beta" not in filtered
    _, unfiltered = _get(base_url, "/")
    assert "w-beta" in unfiltered
    _, jw = _get(base_url, "/api/workers?project=Alpha")
    workers = json.loads(jw)["workers"]
    assert workers and all(w["project"] == "Alpha" for w in workers)


def test_panel_tables_scroll_inside_wrapper(base_url):
    """Wide tables scroll inside their own container, never the page — at 390px the
    document was side-scrolling (scrollWidth 664 vs clientWidth 390), pushing the
    status/worker columns off-screen."""
    _post(base_url, "/api/submit", {"project": "A", "resource": "cpu"})
    _post(base_url, "/api/ask", {"worker_id": "w1", "project": "A", "resource": "cpu"})
    _, html_body = _get(base_url, "/")
    assert html_body.count('<div class="tablewrap">') == 2  # jobs + workers
    assert "overflow-x:auto" in html_body


def test_panel_refresh_interval_control(base_url):
    """WCAG 2.2.1/2.2.4: a self-refreshing page must let the operator slow or stop
    it. ``?refresh=`` is the no-JS control — clamped int, malformed falls back to
    the default, 0 pauses (no meta tag at all)."""
    _, default = _get(base_url, "/")
    assert '<meta http-equiv="refresh" content="5">' in default
    _, slow = _get(base_url, "/?refresh=30")
    assert '<meta http-equiv="refresh" content="30">' in slow
    _, paused = _get(base_url, "/?refresh=0")
    assert 'http-equiv="refresh"' not in paused and "paused" in paused
    _, junk = _get(base_url, "/?refresh=abc")
    assert '<meta http-equiv="refresh" content="5">' in junk  # malformed -> default, never a 500
    _, huge = _get(base_url, "/?refresh=999999")
    assert '<meta http-equiv="refresh" content="3600">' in huge  # clamped ceiling
    # the two knobs propagate through each other's links
    _post(base_url, "/api/submit", {"project": "Alpha", "resource": "cpu"})
    _, both = _get(base_url, "/?project=Alpha&refresh=30")
    assert 'href="/?refresh=30"' in both  # the "all" chip keeps the cadence
    assert 'href="/?project=Alpha&amp;refresh=30"' in both  # the project chip keeps both


def test_panel_contrast_meets_aa():
    """Every chip text colour must hit WCAG AA 4.5:1 on its own background — parsed
    from the REAL ``_CSS`` (tokens resolved through ``var()``), so a future palette
    tweak that regresses readability fails here instead of in an operator's squint.
    Baseline audit: all six job chips + all three worker chips sat at 2.88–3.77:1."""
    css = _web._CSS
    tokens = dict(re.findall(r"--([a-z-]+):(#[0-9a-fA-F]{3,6})", css))

    def resolve(v: str) -> str:
        m = re.fullmatch(r"var\(--([a-z-]+)\)", v.strip())
        return tokens[m.group(1)] if m else v.strip()

    def lum(h: str) -> float:
        h = h.lstrip("#")
        h = "".join(c * 2 for c in h) if len(h) == 3 else h
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
        lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
        return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

    def ratio(fg: str, bg: str) -> float:
        hi, lo = sorted((lum(fg), lum(bg)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    failures = []
    for cls in ("s-queued", "s-granted", "s-running", "s-done", "s-failed", "s-killed",
                "w-waiting", "w-running", "w-dead"):
        body = re.search(r"\." + cls + r"\{([^}]*)\}", css).group(1)
        bg = resolve(re.search(r"background:([^;}]+)", body).group(1))
        fg = resolve(re.search(r"(?<!-)color:([^;}]+)", body).group(1))
        if ratio(fg, bg) < 4.5:
            failures.append(f"{cls}: {fg} on {bg} = {ratio(fg, bg):.2f}:1")
    # status/muted text that sits directly on the white card (th, KPI counts)
    for name in ("muted", "success", "danger", "pending", "ink-soft"):
        if ratio(tokens[name], tokens["card"]) < 4.5:
            failures.append(f"--{name} on --card = {ratio(tokens[name], tokens['card']):.2f}:1")
    assert not failures, "WCAG AA contrast failures:\n" + "\n".join(failures)


def test_panel_a11y_semantics(base_url):
    """Screen-reader basics on the panel: every ``th`` carries ``scope=col``, each
    table is named by a caption, and the active project chip is marked
    ``aria-current`` (colour alone must not carry the selection)."""
    _post(base_url, "/api/ask", {"worker_id": "w1", "project": "Alpha", "resource": "cpu"})
    _post(base_url, "/api/submit", {"project": "Alpha", "resource": "cpu"})
    _, html_body = _get(base_url, "/?project=Alpha")
    assert 'aria-current="page"' in html_body
    assert html_body.count("<caption") == 2
    assert not re.search(r"<th(?![a-z])(?![^>]*scope=)", html_body)  # (?![a-z]) keeps <thead> out


def test_panel_full_ids_and_relative_age(base_url):
    """The 8-char job id keeps the FULL id in ``title`` (hover/copy affordance), and
    liveness columns read as relative ages ("3s ago") with the absolute timestamp
    preserved in ``title`` — an operator eyeballs staleness, not wall-clock math."""
    _, sub = _post(base_url, "/api/submit", {"project": "Alpha", "resource": "cpu"})
    _post(base_url, "/api/ask", {"worker_id": "w-age", "project": "Alpha", "resource": "cpu"})
    _, html_body = _get(base_url, "/")
    assert f'title="{sub["job_id"]}"' in html_body
    assert "ago</td>" in html_body
    assert re.search(r'title="20\d\d-\d\d-\d\d [0-9:]{8} UTC"', html_body)


def test_panel_empty_state_hints(base_url):
    """First-run empty states point at the calls that populate them instead of
    dead-ending ("No jobs on the queue.")."""
    _, html_body = _get(base_url, "/")
    assert "POST /api/submit" in html_body
    assert "/api/ask" in html_body
