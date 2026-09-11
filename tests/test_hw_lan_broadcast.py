"""Contract for the opt-in LAN broadcast of hw-metrics samples.

The engine already samples this host's CPU/RAM/GPU every few seconds and fires a
Postgres ``NOTIFY hw_metrics``. This adds a SECOND publication path for that same
sample -- a UDP broadcast on the local network -- so a LAN listener (e.g. a
display appliance) can consume it with zero configuration and without a bespoke
sidecar re-sampling the same box.

The guarantees pinned here:

* it is **off by default** -- no socket, no datagram, no behaviour change;
* both switches work: ``configure(...)`` and the ``QUEUE_WORKFLOWS_*`` env knob;
* the datagram carries the **same envelope** as the NOTIFY, from one shared
  builder -- if the two ever diverge, ``test_lan_payload_is_the_notify_envelope``
  fails;
* it can never break a worker: every network error is swallowed.
"""

from __future__ import annotations

import json
import socket

import pytest

from queue_workflows import config as config_mod
from queue_workflows import hw_lan, hw_metrics


def _reset_config_preserving_db() -> None:
    """Re-read the config defaults without losing the DB wiring.

    ``reset_for_tests()`` restores *every* field, including ``db_backend``,
    which conftest's between-test truncate depends on. These tests need the
    env-defaulted factories to re-run, so reset and then put the DB fields
    back.
    """
    cfg = config_mod.get_config()
    backend, url_env = cfg.db_backend, cfg.db_url_env
    config_mod.reset_for_tests()
    import queue_workflows
    queue_workflows.configure(db_backend=backend, db_url_env=url_env)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Both spellings cleared so a stray export can't mask a default."""
    for name in ("QUEUE_WORKFLOWS_BROADCAST_HW_STATS", "AI_LEADS_BROADCAST_HW_STATS",
                 "QUEUE_WORKFLOWS_HW_BROADCAST_PORT", "AI_LEADS_HW_BROADCAST_PORT"):
        monkeypatch.delenv(name, raising=False)
    _reset_config_preserving_db()
    yield
    _reset_config_preserving_db()


class _FakeSocket:
    """Stands in for a UDP socket; records what would have gone on the wire."""

    def __init__(self, *_a, **_kw):
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.opts: list[tuple] = []
        self.closed = False

    def setsockopt(self, *args):
        self.opts.append(args)

    def setblocking(self, _flag):
        pass

    def sendto(self, data, addr):
        self.sent.append((data, addr))

    def close(self):
        self.closed = True


@pytest.fixture
def fake_socket(monkeypatch):
    made: list[_FakeSocket] = []

    def _factory(*a, **kw):
        sock = _FakeSocket(*a, **kw)
        made.append(sock)
        return sock

    monkeypatch.setattr(hw_lan.socket, "socket", _factory)
    monkeypatch.setattr(hw_lan, "discover_targets", lambda *a, **kw: ("192.0.2.255",))
    return made


# ── 1. off by default ────────────────────────────────────────────────────────

def test_flag_is_false_by_default():
    assert config_mod.get_config().broadcast_hw_stats is False
    assert hw_lan.enabled() is False


def test_disabled_publishes_nothing_and_opens_no_socket(fake_socket):
    hw_lan.LanBroadcaster().publish({"cpu_percent": 1.0})
    assert fake_socket == [], "a disabled broadcaster must not open a socket"


# ── 2. both switches ─────────────────────────────────────────────────────────

def test_configure_enables_the_flag():
    import queue_workflows
    queue_workflows.configure(broadcast_hw_stats=True)
    assert config_mod.get_config().broadcast_hw_stats is True
    assert hw_lan.enabled() is True


@pytest.mark.parametrize("var", ["QUEUE_WORKFLOWS_BROADCAST_HW_STATS",
                                 "AI_LEADS_BROADCAST_HW_STATS"])
def test_env_knob_enables_the_flag_in_either_spelling(monkeypatch, var):
    monkeypatch.setenv(var, "1")
    _reset_config_preserving_db()
    assert config_mod.get_config().broadcast_hw_stats is True


@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("yes", True),
                                          ("0", False), ("false", False), ("", False)])
def test_env_knob_truthiness(monkeypatch, raw, expected):
    monkeypatch.setenv("QUEUE_WORKFLOWS_BROADCAST_HW_STATS", raw)
    _reset_config_preserving_db()
    assert config_mod.get_config().broadcast_hw_stats is expected


def test_port_is_configurable_with_a_sane_default(monkeypatch):
    assert config_mod.get_config().hw_broadcast_port == hw_lan.DEFAULT_PORT
    monkeypatch.setenv("QUEUE_WORKFLOWS_HW_BROADCAST_PORT", "40000")
    _reset_config_preserving_db()
    assert config_mod.get_config().hw_broadcast_port == 40000


# ── 3. it actually emits, and DRY with the NOTIFY path ───────────────────────

def test_enabled_sends_one_datagram_per_target(monkeypatch, fake_socket):
    import queue_workflows
    queue_workflows.configure(broadcast_hw_stats=True)
    monkeypatch.setattr(hw_lan, "discover_targets",
                        lambda *a, **kw: ("192.0.2.255", "255.255.255.255"))

    hw_lan.LanBroadcaster().publish({"cpu_percent": 12.5})

    assert len(fake_socket) == 1, "one socket, reused"
    sent = fake_socket[0].sent
    assert [addr for _data, addr in sent] == [
        ("192.0.2.255", hw_lan.DEFAULT_PORT),
        ("255.255.255.255", hw_lan.DEFAULT_PORT),
    ]
    assert socket.SO_BROADCAST in [o[1] for o in fake_socket[0].opts]


def test_lan_payload_is_the_notify_envelope(fake_socket):
    """DRY guard: the LAN datagram and the Postgres NOTIFY must be the SAME
    envelope from the SAME builder. If someone re-implements one of them, the
    shapes drift silently -- this test is what stops that."""
    import queue_workflows
    queue_workflows.configure(broadcast_hw_stats=True)
    sample = {"cpu_percent": 42.0, "ram_percent": 7.5, "gpus": []}

    hw_lan.LanBroadcaster().publish(sample)
    on_the_wire = json.loads(fake_socket[0].sent[0][0].decode())

    expected = json.loads(hw_metrics.build_envelope(sample))
    on_the_wire.pop("sampled_at", None)
    expected.pop("sampled_at", None)
    assert on_the_wire == expected
    assert on_the_wire["host"], "envelope must identify the source host"
    assert on_the_wire["cpu_percent"] == 42.0


# ── 4. never breaks the worker ───────────────────────────────────────────────

def test_send_failure_is_swallowed(monkeypatch, fake_socket):
    import queue_workflows
    queue_workflows.configure(broadcast_hw_stats=True)

    def _boom(*_a, **_kw):
        raise OSError("network is down")

    hw_lan.LanBroadcaster().publish({"cpu_percent": 1.0})   # prime the socket
    monkeypatch.setattr(fake_socket[0], "sendto", _boom)
    hw_lan.LanBroadcaster().publish({"cpu_percent": 1.0})   # must not raise


def test_socket_construction_failure_is_swallowed(monkeypatch):
    import queue_workflows
    queue_workflows.configure(broadcast_hw_stats=True)

    def _boom(*_a, **_kw):
        raise OSError("no sockets for you")

    monkeypatch.setattr(hw_lan.socket, "socket", _boom)
    hw_lan.LanBroadcaster().publish({"cpu_percent": 1.0})   # must not raise


# ── 5. target discovery ──────────────────────────────────────────────────────

def test_discover_targets_always_includes_global_broadcast():
    assert "255.255.255.255" in hw_lan.discover_targets()


def test_discover_targets_prefers_the_default_route_interface(monkeypatch):
    monkeypatch.setattr(hw_lan, "_default_interface", lambda: "eth0")
    monkeypatch.setattr(hw_lan, "_interfaces", lambda: ["eth0", "vpn0"])
    monkeypatch.setattr(hw_lan, "_broadcast_address",
                        lambda name: {"eth0": "10.0.0.255", "vpn0": "10.9.9.255"}.get(name))
    assert hw_lan.discover_targets() == ("10.0.0.255", "255.255.255.255")
    assert hw_lan.discover_targets(all_interfaces=True) == (
        "10.0.0.255", "10.9.9.255", "255.255.255.255")


# ── 6. the sampler wiring ────────────────────────────────────────────────────

def test_sampler_publishes_to_lan_only_when_enabled(monkeypatch):
    """The existing sampler gains the second path -- it must not gain a second
    *sample*: publish is fed the very dict handed to the NOTIFY path."""
    published: list[dict] = []
    monkeypatch.setattr(hw_metrics, "_broadcast", lambda s: None)
    monkeypatch.setattr(hw_metrics, "_lan_publish", lambda s: published.append(s))

    sample = {"cpu_percent": 3.0}
    monkeypatch.setattr(hw_metrics, "_build_sample", lambda *a, **kw: sample)

    hw_metrics.publish_sample(sample)
    assert published == [sample], "sampler must forward the SAME sample object"


# ── 7. LAN-only hosts must not need Postgres ─────────────────────────────────

def test_lan_broadcast_does_not_require_psycopg():
    """A box that only broadcasts on the LAN should not need a Postgres driver.

    ``hw_metrics`` used to ``from queue_workflows.db import connection`` at
    module scope, which imports ``psycopg`` — so a slim LAN-only deployment
    crashed on import for a dependency it never uses. The DB connection is
    needed by exactly one function (the NOTIFY path); importing it there keeps
    the broadcast path dependency-free.
    """
    import subprocess
    import sys as _sys

    program = (
        "import sys\n"
        # Block psycopg exactly as a slim host would. This MUST use find_spec:
        # find_module was removed in 3.12, so a legacy blocker silently does
        # nothing and the test would pass vacuously.
        "class _Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'psycopg':\n"
        "            raise ImportError('psycopg blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        # Prove the blocker actually bites before trusting the rest.
        "try:\n"
        "    import psycopg\n"
        "    raise SystemExit('BLOCKER-INEFFECTIVE')\n"
        "except ImportError:\n"
        "    pass\n"
        "from queue_workflows import hw_lan, hw_metrics\n"
        "env = hw_metrics.build_envelope({'cpu_percent': 1.0})\n"
        "assert 'host' in env\n"
        "assert hw_lan.enabled() is False\n"
        "print('OK')\n"
    )
    out = subprocess.run([_sys.executable, "-c", program],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, (
        f"LAN path still needs psycopg:\nSTDOUT{out.stdout}\nSTDERR{out.stderr}")
    assert "OK" in out.stdout
