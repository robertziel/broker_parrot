"""INVARIANT: an unreachable database is something to WAIT for, not to die of.

A control-host reboot once brought a project's Postgres container back with no docker
network endpoint (``NetworkSettings.Networks == {}`` — the third occurrence of that
specific failure). Every worker of that project then crash-looped — hundreds of restarts
each — and their heartbeat rows went weeks stale, so nothing in any database or panel
showed the fleet had lost a whole project's workers. Another project's worker had been
in the same loop for months.

There are two defects, and the second is what made it a LOOP:

1. ``get_pool`` assigned the module-global ``_pool`` BEFORE calling
   ``open(wait=True, timeout=15)``. psycopg_pool CLOSES a pool whose ``wait`` times
   out, and a closed pool cannot be reopened — so the first slow start left a dead
   object installed as the process-wide pool, and every later call raised
   ``PoolClosed: the pool 'pool-1' is already closed`` no matter how healthy the
   database had become. Both symptoms are in the live logs, from the same containers:

       psycopg_pool.PoolTimeout: pool initialization incomplete after 15.0 sec
       psycopg_pool.PoolClosed: the pool 'pool-1' is already closed

2. Nothing distinguished "the DB is not up YET" from "this worker is misconfigured",
   so the only recovery was to exit and let docker restart the container, which
   re-entered the same 15-second timeout for ever. `RestartCount` was the sole record,
   and no reader looks at that.

So: a transient failure is waited out IN PROCESS with a capped backoff and a line at
most once a minute; a genuine misconfiguration (bad credentials, no such database, a
malformed DSN) still exits, distinctly, because a restart cannot fix it. A DNS failure
is TRANSIENT — that is exactly what the network-detach looks like from inside the
container, and retrying is what recovers when someone re-attaches the endpoint.
"""
from __future__ import annotations

import logging

import pytest

from queue_workflows import db


class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


class _Sleeper:
    """Records sleeps and advances the virtual clock instead of waiting."""

    def __init__(self, clock): self.clock, self.slept = clock, []

    def __call__(self, s):
        self.slept.append(round(s, 3))
        self.clock.advance(s)


# ── (1) the closed pool that could never reopen ──────────────────────────────


def test_a_failed_open_leaves_NO_pool_behind(monkeypatch):
    """THE LOOP. A pool whose open() timed out is closed and unusable; installing it
    as the process-wide pool poisoned every later call."""
    class _DeadPool:
        def __init__(self, *a, **k): pass
        def open(self, **k): raise db.PoolTimeout("pool initialization incomplete")
        def close(self): pass

    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "_pool_factory", lambda *a, **k: _DeadPool())

    with pytest.raises(db.DbUnreachable):
        db.get_pool()

    assert db._pool is None, "a dead pool must not be installed process-wide"


def test_the_next_call_can_still_build_a_HEALTHY_pool(monkeypatch):
    """Recovery must not need a process restart once the DB comes back."""
    class _DeadPool:
        def __init__(self, *a, **k): pass
        def open(self, **k): raise db.PoolTimeout("nope")
        def close(self): pass

    class _LivePool:
        def __init__(self, *a, **k): pass
        def open(self, **k): return None
        def close(self): pass

    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "_pool_factory", lambda *a, **k: _DeadPool())
    with pytest.raises(db.DbUnreachable):
        db.get_pool()

    monkeypatch.setattr(db, "_pool_factory", lambda *a, **k: _LivePool())

    assert isinstance(db.get_pool(), _LivePool)


# ── classification: what is worth waiting for ────────────────────────────────


@pytest.mark.parametrize("err,transient", [
    ('connection failed: failed to resolve host "postgres": Temporary failure in '
     'name resolution', True),           # THE network-detach symptom, from the live log
    ('connection to server at "192.0.2.10", port 6543 failed: Connection refused',
     True),
    ('connection to server at "192.0.2.10", port 6543 failed: server closed the '
     'connection unexpectedly', True),   # also from the live log
    ("the database system is starting up", True),          # post-power-cut WAL replay
    ("pool initialization incomplete after 15.0 sec", True),
])
def test_a_recoverable_failure_is_TRANSIENT(err, transient):
    assert db.classify_db_error(RuntimeError(err)).transient is transient


@pytest.mark.parametrize("err", [
    'password authentication failed for user "ai_leads"',
    'database "nope" does not exist',
    "invalid connection option",
])
def test_a_misconfiguration_is_NOT_transient(err):
    assert db.classify_db_error(RuntimeError(err)).transient is False


def test_the_target_is_reported_WITHOUT_the_password():
    """The wait line names the DSN so an operator knows which database to fix — and it
    must never be the place a password leaks into a log."""
    masked = db.mask_dsn("postgresql://ai_leads:hunter2@192.0.2.10:6543/ai_leads")

    assert masked == "192.0.2.10:6543/ai_leads"
    assert "hunter2" not in masked


def test_masking_survives_a_keyword_dsn_and_junk():
    assert "secret" not in db.mask_dsn("host=h port=5432 dbname=d password=secret")
    assert db.mask_dsn("") == "?"


# ── (2) wait, with a bound and a bounded voice ───────────────────────────────


def test_a_transient_failure_is_retried_with_a_CAPPED_backoff():
    clock = _Clock()
    sleeper = _Sleeper(clock)
    attempts = {"n": 0}

    def _probe():
        attempts["n"] += 1
        return attempts["n"] > 6

    assert db.wait_for_db(
        probe_fn=lambda: db.Reachability(ok=_probe(), transient=True,
                                         target="h:5432/d", error="refused"),
        now_fn=clock, sleep_fn=sleeper) is True
    assert sleeper.slept == [1, 2, 4, 8, 16, 30]


def test_the_waiting_line_is_logged_at_most_once_a_minute(caplog):
    """A 15-second crash-loop wrote nothing durable; a wait must not write a wall of
    lines instead."""
    clock = _Clock()
    sleeper = _Sleeper(clock)
    n = {"i": 0}

    def _probe():
        n["i"] += 1
        return db.Reachability(ok=n["i"] > 40, transient=True,
                               target="192.0.2.10:6543/ai_leads", error="refused")

    with caplog.at_level(logging.WARNING, logger="queue_workflows.db"):
        db.wait_for_db(probe_fn=_probe, now_fn=clock, sleep_fn=sleeper)

    said = [r.getMessage() for r in caplog.records if "waiting_db" in r.getMessage()]
    elapsed_min = (clock() - 1000.0) / 60.0
    assert 1 <= len(said) <= elapsed_min + 2
    assert "192.0.2.10:6543/ai_leads" in said[0]


def test_a_misconfiguration_raises_at_once_and_never_sleeps():
    """A restart cannot fix a wrong password; looping on one only hides it."""
    clock = _Clock()
    sleeper = _Sleeper(clock)

    with pytest.raises(db.DbMisconfigured):
        db.wait_for_db(
            probe_fn=lambda: db.Reachability(
                ok=False, transient=False, target="h:5432/d",
                error='password authentication failed for user "ai_leads"'),
            now_fn=clock, sleep_fn=sleeper)

    assert sleeper.slept == []


def test_a_stop_event_ends_the_wait_promptly():
    """SIGTERM during a 30 s backoff must not be ignored for 30 s."""
    import threading
    clock = _Clock()
    stop = threading.Event()
    calls = {"n": 0}

    def _probe():
        calls["n"] += 1
        if calls["n"] == 2:
            stop.set()
        return db.Reachability(ok=False, transient=True, target="h:5432/d",
                               error="refused")

    assert db.wait_for_db(probe_fn=_probe, now_fn=clock,
                          sleep_fn=lambda s: clock.advance(s),
                          stop_event=stop) is False


def test_an_operator_can_restore_the_old_exit_and_restart_behaviour():
    """``QUEUE_WORKFLOWS_DB_WAIT_MAX_S`` is the escape hatch back to letting the
    supervisor restart the process."""
    clock = _Clock()
    sleeper = _Sleeper(clock)

    assert db.wait_for_db(
        probe_fn=lambda: db.Reachability(ok=False, transient=True,
                                         target="h:5432/d", error="refused"),
        now_fn=clock, sleep_fn=sleeper, max_wait_s=10.0) is False


def test_waiting_returns_immediately_when_the_database_is_already_up():
    clock = _Clock()
    sleeper = _Sleeper(clock)

    assert db.wait_for_db(
        probe_fn=lambda: db.Reachability(ok=True, transient=False,
                                         target="h:5432/d", error=""),
        now_fn=clock, sleep_fn=sleeper) is True
    assert sleeper.slept == []


# ── credentials must never reach a log (audit delta D3) ─────────────────────


def test_a_MALFORMED_dsn_error_does_not_echo_the_password():
    """libpq quotes the connection string back inside its own parse error, so the
    error text is not safe just because we never formatted the DSN into it."""
    bad = "postgres//ai_leads:sup3rs3cr3t@192.0.2.10/ai_leads"
    exc = RuntimeError(f'missing "=" after "{bad}" in connection info string')

    r = db.classify_db_error(exc, target="?", dsn=bad)

    assert r.transient is False
    assert "sup3rs3cr3t" not in r.error


def test_the_raised_misconfiguration_carries_no_password():
    bad = "postgresql://ai_leads:sup3rs3cr3t@h:5432/d"
    with pytest.raises(db.DbMisconfigured) as caught:
        db.wait_for_db(
            probe_fn=lambda: db.classify_db_error(
                RuntimeError(f'missing "=" after "{bad}" in connection info string'),
                target="?", dsn=bad),
            now_fn=_Clock(), sleep_fn=lambda s: None)

    assert "sup3rs3cr3t" not in str(caught.value)


@pytest.mark.parametrize("text,secret", [
    ("connection failed for postgresql://u:PWSECRET@h/db", "PWSECRET"),
    ("host=h password=PWSECRET dbname=d", "PWSECRET"),
    ('missing "=" after "postgres//u:PWSECRET@h/db" in connection info string',
     "PWSECRET"),
])
def test_scrubbing_catches_every_password_shape_without_a_dsn(text, secret):
    """The shapes a pattern alone can anchor on. The previous version of this test
    asserted `... or "***" in out`, which passes whenever ANY redaction appears
    anywhere — so it could not fail, and a real leak shipped past it twice."""
    assert secret not in db.scrub_secrets(text)


@pytest.mark.parametrize("dsn,error", [
    # libpq quoting back only the FRAGMENT it choked on — no user:pass@ shape for a
    # pattern to anchor on, and not the whole DSN either. All four observed live.
    ("postgresql://qw:p@ssPWSECRET@127.0.0.1:55499/d",
     "failed to resolve host 'ssPWSECRET@127.0.0.1': [Errno -2] Name or service not known"),
    ("postgresql://qw:a bPWSECRET@127.0.0.1:55499/d",
     'unexpected spaces found in "a bPWSECRET", use percent-encoded spaces (%20) instead'),
    ("postgresql://qw:a%PWSECRET@127.0.0.1:55499/d",
     'invalid percent-encoded token: "a%PWSECRET"'),
    ("postgresql://qw:a@b@PWSECRET@127.0.0.1:55499/d",
     "failed to resolve host 'b@PWSECRET@127.0.0.1'"),
    ("postgresql://qw:PWSECRET/x@127.0.0.1:55499/d",
     'missing "=" after "postgresql://qw:PWSECRET/x@127.0.0.1:55499/d" in connection info'),
])
def test_a_partial_echo_of_the_password_is_still_scrubbed(dsn, error):
    assert "PWSECRET" not in db.scrub_secrets(error, dsn=dsn)


def test_an_ordinary_error_is_left_readable():
    """Over-redaction is the acceptable failure, but it must not be the NORMAL one."""
    err = 'connection to server at "192.0.2.10", port 6543 failed: Connection refused'

    out = db.scrub_secrets(err, dsn="postgresql://ai_leads:hunter2@192.0.2.10:6543/d")

    assert out == err


@pytest.mark.parametrize("dsn,expected", [
    ("postgresql://u:p@[::1]:5432/d", "::1:5432/d"),          # IPv6 stays readable
    ("host=/var/run/postgresql port=5432 dbname=d", "/var/run/postgresql:5432/d"),
    ("postgresql://u:p@10.0.0.1:5432/d", "10.0.0.1:5432/d"),
])
def test_legitimate_hosts_are_not_thrown_away_as_mis_parses(dsn, expected):
    assert db.mask_dsn(dsn) == expected


@pytest.mark.parametrize("err", [
    'unexpected spaces found in "a b", use percent-encoded spaces (%20) instead',
    'invalid percent-encoded token: "a%z"',
    'invalid URI query parameter: "bogus"',
    "unterminated quoted string in connection info string",
])
def test_a_libpq_PARSE_complaint_is_fatal_not_something_to_wait_out(err):
    """These strings never become valid on their own, and the per-minute wait line is
    the most likely place to quote the malformed token back."""
    assert db.classify_db_error(RuntimeError(err)).transient is False


def test_an_UNSET_dsn_env_is_a_misconfiguration_not_an_outage(monkeypatch):
    """Otherwise `db_url()`'s bare RuntimeError escapes `main`'s handler: traceback and
    exit 1, instead of the distinct 'fix your configuration' exit 2."""
    monkeypatch.setattr(db, "_engine_is_sqlite", lambda: False)
    monkeypatch.setattr(db, "db_url", lambda: (_ for _ in ()).throw(
        RuntimeError("QUEUE_WORKFLOWS_DB_URL is not set")))

    assert db.probe().transient is False

    with pytest.raises(db.DbMisconfigured):
        db.wait_for_db(now_fn=_Clock(), sleep_fn=lambda s: None)


# ── the masked target is a promise, not a hope (audit round 2) ──────────────


@pytest.mark.parametrize("dsn,secret", [
    ("postgresql://ai_leads:Xk9/2Lm@192.0.2.10:5432/ai_leads", "Xk9/2Lm"),
    ("postgresql://ai_leads:hunter2@192.0.2.10:6543/ai_leads", "hunter2"),
    ("postgresql://u:a/b/c@h:5432/d", "a/b/c"),
    ("host=h port=5432 dbname=d password=sec/ret", "sec/ret"),
    ("postgres//u:pw@h/db", "pw"),
    ("postgresql://u:" + "z" * 200 + "@h:5432/d", "z" * 200),
])
def test_mask_dsn_never_emits_the_password(dsn, secret):
    """An unencoded `/` in a URI password makes libpq mis-read the authority —
    `host='u' port='Xk9' dbname='2Lm@h:5432/d'` — and a naive reassembly printed the
    credentials back out of the ONE helper whose contract is that it never does."""
    assert secret not in db.mask_dsn(dsn)


def test_a_mis_parsed_dsn_yields_no_detail_rather_than_a_guess():
    assert db.mask_dsn("postgresql://ai_leads:Xk9/2Lm@192.0.2.10:5432/ai_leads") == "?"


def test_the_waiting_line_itself_carries_no_password(caplog):
    """The target reaches a log every minute for as long as the outage lasts, so it is
    the highest-volume place a leak could land."""
    dsn = "postgresql://qw:pa/ss@127.0.0.1:55499/nope"
    clock = _Clock()

    with caplog.at_level(logging.WARNING, logger="queue_workflows.db"):
        db.wait_for_db(
            probe_fn=lambda: db.Reachability(ok=False, transient=True,
                                             target=db.mask_dsn(dsn),
                                             error=db.scrub_secrets("refused", dsn=dsn)),
            now_fn=clock, sleep_fn=_Sleeper(clock), max_wait_s=200.0)

    assert not any("pa/ss" in r.getMessage() for r in caplog.records)


def test_scrub_secrets_handles_a_slash_bearing_password():
    out = db.scrub_secrets("invalid dsn: postgresql://ai_leads:Xk9/2Lm@h:5432/db")
    assert "Xk9/2Lm" not in out
    assert "***:***@" in out


def test_a_bounded_wait_that_gives_up_is_a_clean_stop_not_a_traceback():
    """`wait_for_db` returning False used to fall through into `wait_for_schema`, which
    re-entered the pool timeout and surfaced as exit 1 WITH a traceback — the opposite
    of the clean exit-and-restart the knob asks for."""
    import queue_workflows.claim_worker as cw

    w = cw.ClaimWorker.__new__(cw.ClaimWorker)
    w.queue = "cpu"                       # `_min_schema_version` derives from this
    w._stop = None
    import queue_workflows.db as _db
    orig = _db.wait_for_db
    try:
        _db.wait_for_db = lambda **k: False
        with pytest.raises(_db.DbUnreachable):
            w.await_schema()
    finally:
        _db.wait_for_db = orig


# ── a wait must always be interruptible (audit round 3, B1) ────────────────


def test_every_entrypoint_passes_its_stop_event_into_the_wait():
    """B1. The orchestrator called `wait_for_db()` with no stop_event while its signal
    handler dutifully set one, so SIGTERM (and Ctrl-C) during an unbounded wait were
    logged and ignored — `docker stop`, `systemctl stop` and a reboot all hung to the
    grace period and ended in SIGKILL. A wait that cannot be interrupted is the wedge
    this work exists to remove.

    This scans the PACKAGE SOURCE rather than a list of modules. Two previous versions
    of this guard named their subjects by hand: the first covered two of three call
    sites while the commit said "every", and the second added the third and compared a
    hardcoded triple against itself — a tautology that a real fourth module walked
    straight past with the suite green. A guard for "nobody may do X" has to go looking
    for X.
    """
    import pathlib
    import re

    import queue_workflows

    root = pathlib.Path(queue_workflows.__file__).resolve().parent
    call = re.compile(r"wait_for_db\s*\(([^)]*)\)", re.S)
    offenders = []
    seen = 0
    for path in sorted(root.rglob("*.py")):
        if path.name == "db.py":
            continue                      # where it is DEFINED, not called
        text = path.read_text(encoding="utf-8")
        for match in call.finditer(text):
            args = match.group(1)
            if "def wait_for_db" in text[max(0, match.start() - 40):match.start()]:
                continue
            seen += 1
            if "stop_event" not in args:
                line = text[:match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(root.parent)}:{line}")

    assert seen >= 3, (
        f"expected to find the known wait_for_db call sites, found {seen} — the scan "
        f"has stopped seeing them, which makes it vacuous")
    assert not offenders, (
        "these wait_for_db calls cannot be interrupted by a signal: "
        + ", ".join(offenders))


# ── round-4 follow-ups ──────────────────────────────────────────────────────


def test_a_signal_during_the_backoff_is_not_slept_through():
    """A bare `time.sleep` resumes after the handler returns (PEP 475), so a SIGTERM
    landing early in a 30 s backoff still cost the full 30 s — past `docker stop`'s
    10 s grace, so the process died by SIGKILL anyway."""
    import threading

    stop = threading.Event()
    waited: list[float] = []

    class _Event:
        def is_set(self): return stop.is_set()
        def wait(self, t):
            waited.append(t)
            stop.set()          # the signal lands DURING the wait
            return True

    assert db.wait_for_db(
        probe_fn=lambda: db.Reachability(ok=False, transient=True, target="h:5432/d",
                                         error="refused"),
        now_fn=_Clock(), stop_event=_Event()) is False
    assert waited, "the backoff must be an interruptible wait, not a bare sleep"


_ORDINARY_ERRORS = [
    'connection to server at "127.0.0.1", port 5432 failed: Connection refused',
    'FATAL:  password authentication failed for user "qw"',
    'FATAL:  database "nosuchdb" does not exist',
    "failed to resolve host 'db.internal'",
]


@pytest.mark.parametrize("pw", ["e", "on", "1", "ab", "5432", "port", "host", "user",
                                "exist"])
@pytest.mark.parametrize("text", _ORDINARY_ERRORS)
def test_a_SHORT_password_does_not_shred_an_ordinary_message(text, pw):
    """`pw='e'` once turned "connection failed" into "conn***ction fail***d", and a
    later fix still let `pw='5432'` hide the port — a claim an audit measured as
    false. A short secret is a word or a number the message may legitimately
    use, so it is redacted only in CREDENTIAL context."""
    assert db.scrub_secrets(text, dsn=f"postgresql://u:{pw}@127.0.0.1:5432/d") == text


def test_a_SHORT_password_is_still_redacted_where_it_stands_alone():
    out = db.scrub_secrets("failed to resolve host 'ab@127.0.0.1'",
                           dsn="postgresql://u:ab@127.0.0.1:5432/d")
    assert "'***@127.0.0.1'" in out


def test_the_pool_error_is_scrubbed_too(monkeypatch):
    """`get_pool` built its message from the raw driver exception — new code on a path
    whose whole premise is that driver text is untrusted."""
    class _DeadPool:
        def __init__(self, *a, **k): pass
        def open(self, **k):
            raise db.PoolTimeout('cannot connect to "postgresql://u:PWSECRET@h/d"')
        def close(self): pass

    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "_engine_is_sqlite", lambda: False)
    monkeypatch.setattr(db, "db_url", lambda: "postgresql://u:PWSECRET@h:5432/d")
    monkeypatch.setattr(db, "_pool_factory", lambda *a, **k: _DeadPool())

    with pytest.raises(db.DbUnreachable) as caught:
        db.get_pool()

    assert "PWSECRET" not in str(caught.value)


@pytest.mark.parametrize("pw,text", [
    ("ab", "failed to resolve host 'ab@127.0.0.1'"),
    ("e", "failed to resolve host 'e@127.0.0.1'"),
    ("a%z", 'invalid percent-encoded token: "a%z"'),
    ("5432", "failed to resolve host '5432@127.0.0.1'"),
])
def test_a_SHORT_password_IS_redacted_in_credential_context(pw, text):
    """The other half: leaving a fragment echo readable to protect the prose would be
    the wrong trade."""
    out = db.scrub_secrets(text, dsn=f"postgresql://u:{pw}@127.0.0.1:5432/d")
    assert "***" in out
    assert f"'{pw}@" not in out and f'"{pw}"' not in out


def test_a_password_in_the_URI_QUERY_STRING_is_found_too():
    """`_password_of` read only the userinfo, so `?password=…` — a shape libpq accepts
    — was never redacted. Pre-existing (the parent leaks it with a traceback), found by
    the round-5 audit."""
    dsn = "postgresql://u@127.0.0.1:5432/d?password=QSZQXSECRET&sslmode=require"

    out = db.scrub_secrets('invalid percent-encoded token: "QSZQXSECRET"', dsn=dsn)

    assert "QSZQXSECRET" not in out
