"""Broker web service + operator panel (v2 control plane).

Pure-stdlib ``http.server`` over :mod:`queue_workflows.broker_service` — the
operator's "the broker must be a web service with a panel" surface. Two faces on
one server (matching the ``conductor/web.py`` house style: stdlib only,
server-rendered, no JS, ``Cache-Control: no-store``, ``ThreadingHTTPServer``):

* **worker / broker JSON API** — ``POST /api/{submit,ask,heartbeat,renew,finish,
  abort,revoke}``: the endpoints a client-library worker calls to be granted work
  and report outcomes, plus the operator's ``revoke`` (kill). This is the network
  form of the pull→grant control model.
* **read-only operator panel** — ``GET /``: the shared, project-labelled CPU/GPU
  queue, the worker fleet + liveness, and a per-resource KPI strip, with a
  ``?project=`` filter. Also ``GET /api/{jobs,workers,snapshot}`` (JSON),
  ``GET /healthz``, and ``GET /favicon.svg`` (the brand mark, served pre-auth).

**Auth is opt-in but fail-closed for a public bind.** Set
``QUEUE_WORKFLOWS_BROKER_WEB_TOKEN`` to require ``Authorization: Bearer <token>``
on every route except ``/healthz`` + ``/favicon.svg`` (compared in constant time).
A loopback bind (``127.0.0.1``/``::1``) stays open with no token for the default
single-box deploy; binding a **non-loopback** host (e.g. ``0.0.0.0`` in k8s)
**requires** a non-placeholder token ≥ :data:`_MIN_TOKEN_LEN` chars — otherwise
:func:`run` refuses to start rather than expose an unauthenticated control plane.
A single shared token is the minimal gate; per-worker identity / RBAC / TLS are
later passes. The SQLite store is ``check_same_thread=False`` so the threading
server shares it safely.
"""

from __future__ import annotations

import argparse
import hmac
import html
import json
import logging
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

log = logging.getLogger(__name__)

from queue_workflows.broker_service import orchestrator as _o
from queue_workflows.broker_service import permission as _p
from queue_workflows.broker_service.schema import RESOURCES, ensure_schema

_JOB_STATUSES = ("queued", "granted", "running", "done", "failed", "killed")

#: The broker_parrot brand mark — a parrot MANAGING DATA: perched on a database
#: cylinder (the queue store), tail sweeping down it, hooked beak turned toward the
#: queue rows it dispatches (two in flight, one completed-green). Slate rounded-square
#: badge on a 32×32 grid so it stays crisp from a header logo down to a 16px
#: browser-tab favicon. One string, two uses: inlined in the panel header and served
#: verbatim at ``GET /favicon.svg`` (self-contained, ``xmlns`` set). The same art
#: ships as ``docs/images/logo.svg`` (the README hero).
_BRAND_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" role="img" aria-label="broker_parrot">'
    '<rect width="32" height="32" rx="7" fill="#0f172a"/>'
    '<path d="M4.6 20.8 L4.6 25.4 C4.6 26.9 7.6 27.9 10 27.9 C12.4 27.9 15.4 26.9 15.4 25.4 L15.4 20.8 Z" fill="#334155"/>'
    '<ellipse cx="10" cy="20.8" rx="5.4" ry="2.2" fill="#475569"/>'
    '<rect x="19.6" y="19.6" width="8.4" height="2.5" rx="1.25" fill="#38bdf8"/>'
    '<rect x="19.6" y="23.2" width="6.2" height="2.5" rx="1.25" fill="#38bdf8" opacity="0.72"/>'
    '<rect x="19.6" y="26.8" width="4.2" height="2.5" rx="1.25" fill="#22c55e"/>'
    '<path d="M8.6 21 C7.2 15 8.8 8.3 13.3 6.3 C17.3 4.6 21.3 6.6 22.1 9.8 C22.8 12.9 21.3 15.4 18.3 16.9 L13.6 21 Z" fill="#22c55e"/>'
    '<path d="M11.4 13.4 C9.9 16.5 10.6 19.6 12.8 21 C15 19.5 15.7 16.2 14.5 13.2 Z" fill="#0d9488"/>'
    '<path d="M9.2 20.8 C7.2 23.6 6.5 26.1 7.2 27.6 C9.2 26.6 10.7 24.5 11.3 21.6 Z" fill="#f43f5e"/>'
    '<circle cx="17.2" cy="10.1" r="2.7" fill="#f8fafc"/>'
    '<circle cx="17.9" cy="9.9" r="1.15" fill="#0f172a"/>'
    '<path d="M19.4 8.9 C23.4 8.4 25.5 10.3 24.5 12.5 C23.8 14 22 14.6 20.7 13.8 C21.6 12.8 21.3 11.1 19.2 10.4 Z" fill="#fbbf24"/>'
    '<path d="M20.9 13.7 C22.1 14.4 23.5 14.4 24.3 13.7 C23.7 15.3 21.7 15.8 20.3 14.8 Z" fill="#f59e0b"/>'
    '</svg>'
)

#: When set, every route except ``/healthz`` + ``/favicon.svg`` requires
#: ``Authorization: Bearer <token>``. Unset ⇒ open (back-compat / trusted loopback
#: only; a public bind refuses to start without a strong token — see
#: :func:`_check_bind_security`).
_TOKEN_ENV = "QUEUE_WORKFLOWS_BROKER_WEB_TOKEN"

#: Minimum length for a token to gate a public (non-loopback) bind. Short enough to
#: not annoy, long enough that a shipped placeholder / a hand-typed word is rejected.
_MIN_TOKEN_LEN = 16

#: Placeholders that must never be accepted as a real token on a public bind — the
#: manifests ship these verbatim, so an unedited apply must fail closed. Matched
#: case-insensitively; the length gate already catches most, this makes intent explicit.
_PLACEHOLDER_TOKENS = frozenset({"replace_me", "changeme", "change_me", "changme", "secret", "token"})

#: Hosts treated as loopback — a bind to one of these stays open with no token.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})

#: Max request body we read into memory. A bounded cap turns a hostile / buggy
#: Content-Length into a 413 instead of an OOM (the panel/API payloads are tiny).
_MAX_BODY_BYTES = 1_048_576  # 1 MiB

#: Ceiling on concurrent request-handling threads. The stdlib ThreadingMixIn spawns
#: one thread per connection with no cap; a bounded pool is the thread-exhaustion
#: guard. The API/panel is I/O-light, so a modest default serves a fleet fine.
_MAX_WORKERS_ENV = "QUEUE_WORKFLOWS_BROKER_WEB_MAX_WORKERS"
_DEFAULT_MAX_WORKERS = 32


def _max_workers() -> int:
    """Configured request-handler pool size — env override, else the default. A
    malformed / non-positive value falls back to the default (never unbounded)."""
    try:
        n = int(os.environ.get(_MAX_WORKERS_ENV) or _DEFAULT_MAX_WORKERS)
    except ValueError:
        return _DEFAULT_MAX_WORKERS
    return n if n > 0 else _DEFAULT_MAX_WORKERS


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """A ``ThreadingHTTPServer`` whose requests run on a **bounded** thread pool.

    ``ThreadingMixIn`` mints one thread per connection with no ceiling, so a flood
    of connections becomes a flood of threads (resource-exhaustion DoS). Overriding
    :meth:`process_request` to submit onto a fixed :class:`ThreadPoolExecutor` caps
    concurrency at ``max_workers``; excess connections queue on the pool (and the
    socket backlog) instead of each spawning a thread. Handler cleanup reuses the
    mixin's own :meth:`process_request_thread`."""

    daemon_threads = True
    block_on_close = False  # the pool owns thread lifetime; don't also join in the mixin

    def __init__(self, *args: Any, max_workers: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or _max_workers(), thread_name_prefix="broker-web"
        )

    def process_request(self, request: Any, client_address: Any) -> None:  # type: ignore[override]
        self._pool.submit(self.process_request_thread, request, client_address)

    def server_close(self) -> None:
        super().server_close()
        self._pool.shutdown(wait=False)


def _valid_public_token(token: str | None) -> bool:
    """True iff ``token`` is strong enough to gate a public bind: present, not a
    known placeholder, and at least :data:`_MIN_TOKEN_LEN` chars."""
    if not token:
        return False
    if token.strip().lower() in _PLACEHOLDER_TOKENS:
        return False
    return len(token) >= _MIN_TOKEN_LEN


def _check_bind_security(host: str) -> None:
    """Fail closed before binding a **non-loopback** host without a strong token.

    Loopback binds stay open (back-compat single-box default). A public bind with
    no / placeholder / short token raises :class:`RuntimeError` so an operator can't
    accidentally stand up an unauthenticated control plane (whose ``/api/revoke``
    can kill any job)."""
    if host in _LOOPBACK_HOSTS:
        return
    if not _valid_public_token(os.environ.get(_TOKEN_ENV)):
        raise RuntimeError(
            f"refusing to bind broker web on non-loopback host {host!r} without a strong "
            f"{_TOKEN_ENV} (unset, a shipped placeholder, or < {_MIN_TOKEN_LEN} chars). "
            "Set a random secret, or bind 127.0.0.1 and front it with an authenticated proxy."
        )

#: The panel's design tokens — warm amber primary on a soft neutral page, ink
#: text, hairline borders, pill badges, carded tables with a tinted row hover.
#: Pure CSS on the handler's own class names; light theme (dashboards embedding
#: this can theme around it).
_CSS = """
:root{color-scheme:light;
--primary:#f7a825;--primary-dark:#e0941a;--primary-soft:#fdebcb;--primary-tint:#fff6e7;
--ink:#313131;--ink-soft:#4d4d4d;--muted:#7e7e7e;
--border:#dadada;--border-soft:#ececec;--page:#f9f9f9;--card:#fff;
--success:#1f9d6b;--success-bg:#d8f0e6;--danger:#d64545;--danger-bg:#fbe0e0;
--pending:#b9740c;--pending-bg:#fdebcb;
--radius:12px;--radius-sm:8px;--radius-pill:100px;--topbar-h:64px;
--shadow-card:0 1px 2px rgba(49,49,49,.05),0 1px 3px rgba(49,49,49,.06);
--ring:0 0 0 3px rgba(247,168,37,.25)}
*{box-sizing:border-box}
a:focus-visible{outline:none;box-shadow:var(--ring);border-radius:var(--radius-sm)}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
     color:var(--ink);background:var(--page);font-variant-numeric:tabular-nums}
header{min-height:var(--topbar-h);padding:12px 24px;background:var(--card);
       border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.logo{flex:none;line-height:0}
.logo svg{display:block;width:32px;height:32px;border-radius:var(--radius-sm)}
h1{font-size:19px;margin:0;font-weight:600;letter-spacing:.1px}
h1 .v{color:var(--muted);font-weight:500;font-size:13px}
.sub{color:var(--muted);font-size:12.5px;margin-left:auto}
main{padding:22px 24px;max-width:1120px;margin:0 auto}
section{margin-bottom:28px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin:0 0 10px;font-weight:600}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 10px}
.filters a{text-decoration:none;font-size:13px;font-weight:600;padding:5px 16px;border-radius:var(--radius-pill);
           border:1px solid var(--border);color:var(--ink-soft);background:var(--card);
           transition:all .18s ease;letter-spacing:.1px}
.filters a:hover{background:var(--primary-tint);border-color:var(--primary);color:var(--ink)}
.filters a.on{background:var(--primary);color:var(--ink);border-color:var(--primary);font-weight:700}
.kpis{display:flex;gap:14px;flex-wrap:wrap}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
     box-shadow:var(--shadow-card);padding:14px 16px;min-width:160px}
.kpi .r{font-weight:700;text-transform:uppercase;font-size:12px;letter-spacing:.5px;color:var(--ink-soft);margin-bottom:8px}
.kpi .row{display:flex;justify-content:space-between;font-size:12.5px;color:var(--muted);padding:1px 0}
.kpi .row b{color:var(--ink)}
.kpi .row b.n-done{color:var(--success)}.kpi .row b.n-failed{color:var(--danger)}
.kpi .row b.n-running{color:var(--pending)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--border);
      border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow-card)}
th,td{text-align:left;padding:12px;font-size:13.5px}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.5px;
   border-bottom:1px solid var(--border);background:var(--card);white-space:nowrap}
td{border-bottom:1px solid var(--border-soft);color:var(--ink)}
tbody tr{transition:background-color .12s ease}
tbody tr:hover{background:var(--primary-tint)}
tr:last-child td{border-bottom:none}
td.mono,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.chip{display:inline-flex;align-items:center;padding:3px 12px;border-radius:var(--radius-pill);
      font-size:12px;font-weight:600;line-height:1.5;border:1px solid transparent;white-space:nowrap;letter-spacing:.1px}
.s-queued{background:#eee;color:var(--muted);border-color:var(--border)}
.s-granted{background:var(--primary-tint);color:var(--pending);border-color:rgba(247,168,37,.35)}
.s-running{background:var(--pending-bg);color:var(--pending);border-color:rgba(247,168,37,.35)}
.s-done{background:var(--success-bg);color:var(--success);border-color:rgba(31,157,107,.2)}
.s-failed{background:var(--danger-bg);color:var(--danger);border-color:rgba(214,69,69,.2)}
.s-killed{background:#eee;color:var(--danger);border-color:rgba(214,69,69,.2)}
.w-waiting{background:#eee;color:var(--muted);border-color:var(--border)}
.w-running{background:var(--success-bg);color:var(--success);border-color:rgba(31,157,107,.2)}
.w-dead{background:var(--danger-bg);color:var(--danger);border-color:rgba(214,69,69,.2)}
.empty{color:var(--muted);font-size:13px;padding:16px;background:var(--card);
       border:1px dashed var(--border);border-radius:var(--radius)}
"""


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _fmt_ts(v: Any) -> str:
    """Render a timestamp cell across both backends: a datetime formats directly;
    the sqlite TEXT / ISO-8601 form is trimmed to the same ``YYYY-MM-DD HH:MM:SS``
    shape (no microseconds/offset noise); missing → an em-dash, never a blank cell."""
    if v is None or v == "":
        return "—"
    try:
        return v.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:
        s = str(v).replace("T", " ")
        return _esc(s[:19]) if len(s) >= 19 else _esc(s)


def _status_chip(status: str) -> str:
    return f'<span class="chip s-{_esc(status)}">{_esc(status)}</span>'


def _worker_chip(state: str) -> str:
    return f'<span class="chip w-{_esc(state)}">{_esc(state)}</span>'


def render_panel(project: str | None = None) -> str:
    """Server-render the read-only operator panel for the shared queue + fleet."""
    counts = _o.queue_counts(project=project)
    workers = _o.list_workers()
    jobs = _o.list_jobs(project=project, limit=50)
    all_projects = sorted(
        {j["project"] for j in _o.list_jobs(limit=500)} | {w["project"] for w in workers}
    )

    # counts[resource][status] -> n
    by_res: dict[str, dict[str, int]] = {r: {s: 0 for s in _JOB_STATUSES} for r in sorted(RESOURCES)}
    for c in counts:
        by_res.setdefault(c["resource"], {s: 0 for s in _JOB_STATUSES})[c["status"]] = c["n"]

    kpis = "".join(
        '<div class="kpi"><div class="r">{r}</div>{rows}</div>'.format(
            r=_esc(res),
            rows="".join(
                # n-<status> tints the count with its status colour when non-zero
                f'<div class="row"><span>{s}</span>'
                f'<b class="{"n-" + s if by_res[res].get(s, 0) else ""}">{by_res[res].get(s, 0)}</b></div>'
                for s in _JOB_STATUSES
            ),
        )
        for res in by_res
    )

    def _chip_link(label: str, value: str | None, on: bool) -> str:
        href = "/" if value is None else "/?project=" + urllib.parse.quote(value)
        return f'<a href="{href}" class="{"on" if on else ""}">{_esc(label)}</a>'

    filters = _chip_link("all", None, project is None) + "".join(
        _chip_link(p, p, project == p) for p in all_projects
    )

    if workers:
        worker_rows = "".join(
            "<tr><td class='mono'>{w}</td><td>{p}</td><td>{r}</td><td>{st}</td><td class='mono'>{ls}</td></tr>".format(
                w=_esc(x["worker_id"]), p=_esc(x["project"]), r=_esc(x["resource"]),
                st=_worker_chip(x["state"]), ls=_fmt_ts(x["last_seen"]),
            )
            for x in workers
        )
        workers_html = (
            "<table><thead><tr><th>worker</th><th>project</th><th>resource</th>"
            f"<th>state</th><th>last seen</th></tr></thead><tbody>{worker_rows}</tbody></table>"
        )
    else:
        workers_html = '<div class="empty">No workers registered.</div>'

    if jobs:
        job_rows = "".join(
            "<tr><td class='mono'>{id}</td><td>{p}</td><td>{r}</td><td>{st}</td>"
            "<td class='mono'>{pri}</td><td class='mono'>{w}</td>"
            "<td class='mono'>{c}</td><td class='mono'>{u}</td></tr>".format(
                id=_esc(j["job_id"][:8]), p=_esc(j["project"]), r=_esc(j["resource"]),
                st=_status_chip(j["status"]), pri=_esc(j["priority"]),
                w=_esc(j["granted_worker"] or "—"),
                c=_fmt_ts(j.get("created_at")), u=_fmt_ts(j.get("updated_at")),
            )
            for j in jobs
        )
        jobs_html = (
            "<table><thead><tr><th>job</th><th>project</th><th>resource</th><th>status</th>"
            f"<th>prio</th><th>worker</th><th>created</th><th>updated</th></tr></thead><tbody>{job_rows}</tbody></table>"
        )
    else:
        jobs_html = '<div class="empty">No jobs on the queue.</div>'

    scope = f"project: {_esc(project)}" if project else "all projects"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<title>broker_parrot — broker</title><style>{_CSS}</style></head>
<body>
<header>
  <span class="logo">{_BRAND_SVG}</span>
  <h1>broker <span class="v">— shared CPU/GPU queue · orchestrated</span></h1>
  <span class="sub">{scope} · auto-refresh 5s</span>
</header>
<main>
  <section>
    <h2>Queue</h2>
    <div class="kpis">{kpis}</div>
  </section>
  <section>
    <div class="filters">{filters}</div>
    <h2>Jobs</h2>
    {jobs_html}
  </section>
  <section>
    <h2>Workers</h2>
    {workers_html}
  </section>
</main>
</body></html>"""


class BrokerWebHandler(BaseHTTPRequestHandler):
    """Routes the broker JSON API + the operator panel. One handler per request
    thread; all state lives in the shared relational store."""

    server_version = "broker_parrot-web"

    def log_message(self, *_a: Any) -> None:  # quiet by default
        pass

    # ── response helpers ────────────────────────────────────────────────────
    def _send(self, status: int, ctype: str, body: str, cache: str | None = None) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # Live data is never cached (matches the no-store house style); a caller may
        # pass ``cache`` for an immutable static asset like the favicon.
        self.send_header("Cache-Control", cache or "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, obj: Any) -> None:
        self._send(status, "application/json", json.dumps(obj, default=str))

    def _content_length(self) -> int:
        """Parse Content-Length defensively (a malformed header ⇒ 0, never raises)."""
        try:
            return max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            return 0

    def _read_json(self) -> dict[str, Any]:
        n = self._content_length()
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def _authorized(self) -> bool:
        """True unless a shared token is configured and the request doesn't present it.

        The bearer value is compared with :func:`hmac.compare_digest` so a wrong
        token can't be recovered byte-by-byte via response-timing."""
        token = os.environ.get(_TOKEN_ENV)
        if not token:
            return True
        presented = self.headers.get("Authorization", "")
        return hmac.compare_digest(presented, f"Bearer {token}")

    def _unauthorized(self) -> None:
        data = json.dumps({"error": "unauthorized"}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("WWW-Authenticate", 'Bearer realm="broker"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ── routing ─────────────────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        project = (q.get("project") or [None])[0]
        if u.path == "/healthz":
            return self._send(200, "text/plain; charset=utf-8", "ok")
        if u.path == "/favicon.svg":
            # Public static asset — served before the auth gate (the browser doesn't
            # attach the bearer token to a favicon fetch, and the mark isn't sensitive).
            return self._send(200, "image/svg+xml", _BRAND_SVG, cache="public, max-age=86400")
        if not self._authorized():
            return self._unauthorized()
        if u.path == "/api/jobs":
            return self._json(200, {"jobs": _o.list_jobs(project=project)})
        if u.path == "/api/workers":
            return self._json(200, {"workers": _o.list_workers()})
        if u.path == "/api/snapshot":
            return self._json(200, {
                "counts": _o.queue_counts(project=project),
                "workers": _o.list_workers(),
                "jobs": _o.list_jobs(project=project),
            })
        if u.path == "/":
            return self._send(200, "text/html; charset=utf-8", render_panel(project))
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        if not self._authorized():
            return self._unauthorized()
        if self._content_length() > _MAX_BODY_BYTES:
            # Reject oversized bodies before reading them into memory (DoS guard).
            return self._json(413, {"error": "request body too large"})
        b = self._read_json()
        try:
            if u.path == "/api/submit":
                job_id = _o.submit_job(
                    project=b["project"], resource=b["resource"],
                    priority=int(b.get("priority", 100)), payload=b.get("payload"),
                )
                return self._json(200, {"job_id": job_id})
            if u.path == "/api/ask":
                cap = b.get("capacity")
                job = _p.ask_to_run(
                    b["worker_id"], project=b["project"], resource=b["resource"],
                    lease_s=float(b.get("lease_s", 30.0)),
                    capacity=int(cap) if cap is not None else None,
                )
                return self._json(200, {"granted": job is not None, "job": job})
            if u.path == "/api/heartbeat":
                return self._json(200, {"ok": _o.worker_heartbeat(b["worker_id"])})
            if u.path == "/api/renew":
                permitted = _p.keep_permission(
                    b["job_id"], b["worker_id"], lease_s=float(b.get("lease_s", 30.0))
                )
                return self._json(200, {"permitted": permitted})
            if u.path == "/api/finish":
                return self._json(200, {"job": _p.finish(b["job_id"], b["worker_id"], result=b.get("result"))})
            if u.path == "/api/abort":
                return self._json(200, {"job": _p.abort(b["job_id"], b["worker_id"], error=b.get("error"))})
            if u.path == "/api/revoke":
                job = _o.revoke(b["job_id"], requeue=bool(b.get("requeue", True)), reason=b.get("reason"))
                return self._json(200, {"job": job})
        except KeyError as exc:
            return self._json(400, {"error": f"missing field {exc}"})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        except Exception:  # noqa: BLE001 — surface as 500, never crash the server
            # Log the detail server-side; return a generic message so a backend
            # exception can't leak table/column/DSN internals to the caller.
            log.exception("[broker-web] unhandled error on POST %s", u.path)
            return self._json(500, {"error": "internal error"})
        return self._json(404, {"error": "not found"})


def run(host: str = "127.0.0.1", port: int = 8787) -> None:
    """Serve the broker web service until interrupted.

    Refuses to start on a non-loopback ``host`` without a strong
    ``QUEUE_WORKFLOWS_BROKER_WEB_TOKEN`` (see :func:`_check_bind_security`)."""
    _check_bind_security(host)
    httpd = _BoundedThreadingHTTPServer((host, port), BrokerWebHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="queue-broker-web",
        description="Broker web service + read-only operator panel over the shared queue.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--db-backend", default=None, help="pg | sqlite | redis | mongodb (default: sqlite)")
    ap.add_argument("--db-url-env", default=None, help="env var holding the DSN / sqlite path")
    args = ap.parse_args(argv)

    import queue_workflows
    kw: dict[str, Any] = {}
    if args.db_backend:
        kw["db_backend"] = args.db_backend
    if args.db_url_env:
        kw["db_url_env"] = args.db_url_env
    if kw:
        queue_workflows.configure(**kw)
    try:
        _check_bind_security(args.host)  # fail closed with a clean message, not a traceback
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    ensure_schema()
    print(f"broker web service on http://{args.host}:{args.port}  (panel: /, api: /api/*)")
    run(args.host, args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
