"""LAN broadcast of hardware-metrics samples — the second publication path.

The engine already samples this host's CPU / RAM / GPU on a fixed cadence and
publishes each sample as a Postgres ``NOTIFY hw_metrics`` (see
:mod:`queue_workflows.hw_metrics`). That path serves consumers that can reach the
broker database. This module adds a second, *optional* path for the very same
sample: a UDP broadcast on the local network.

**Why a second path rather than a sidecar.** A LAN appliance — a display panel,
say — should not need Postgres credentials, a DSN, or a bespoke container that
re-probes a box the engine is already probing every few seconds. Broadcasting
lets such a listener consume telemetry with *zero* configuration: it needs no
address for this host, keeps working when either side's DHCP lease changes, and
one datagram serves any number of listeners. The cost is that broadcast does not
cross subnets — but "the local network" is precisely the broadcast domain, which
is the intent.

**Off by default.** ``EngineConfig.broadcast_hw_stats`` is ``False``; nothing
here runs, and no socket is created, until a host opts in with
``configure(broadcast_hw_stats=True)`` or ``QUEUE_WORKFLOWS_BROADCAST_HW_STATS=1``.

**Never fatal.** This is telemetry. Every failure — no socket, no route, a
downed interface — is logged and swallowed, because a claim worker must never
die for want of a status datagram.

This module is a leaf: it imports only ``config`` and ``envcompat``, never
``hw_metrics``, so the sampler depends on the transport and not the reverse.
"""

from __future__ import annotations

import fcntl
import logging
import socket
import struct
from typing import Any

from queue_workflows.config import get_config

log = logging.getLogger(__name__)

#: Unprivileged, unassigned-by-IANA. Consumers must agree with this.
DEFAULT_PORT = 48291

#: linux/sockios.h — ask the kernel for an interface's broadcast address.
_SIOCGIFBRDADDR = 0x8919

#: Interfaces that are never the LAN a listener sits on: loopback, container
#: bridges, VPN tunnels. Broadcasting telemetry at VPN peers is noise at best.
_VIRTUAL_PREFIXES = ("lo", "docker", "veth", "br-", "virbr", "tailscale",
                     "tun", "tap", "wg")


def enabled() -> bool:
    """True when this host opted into LAN broadcast."""
    return bool(get_config().broadcast_hw_stats)


def port() -> int:
    return int(get_config().hw_broadcast_port or DEFAULT_PORT)


# ── interface discovery ──────────────────────────────────────────────────────


def _default_interface() -> str | None:
    """The interface carrying the default route, from ``/proc/net/route``."""
    try:
        with open("/proc/net/route") as fh:
            for line in fh.readlines()[1:]:
                parts = line.split()
                if len(parts) > 2 and parts[1] == "00000000":
                    return parts[0]
    except OSError:
        pass
    return None


def _interfaces() -> list[str]:
    """Real, non-virtual interface names from ``/proc/net/dev``."""
    names: list[str] = []
    try:
        with open("/proc/net/dev") as fh:
            for line in fh.readlines()[2:]:
                name = line.split(":", 1)[0].strip()
                if name and not name.startswith(_VIRTUAL_PREFIXES):
                    names.append(name)
    except OSError:
        pass
    return names


def _broadcast_address(name: str) -> str | None:
    """This interface's broadcast address, or ``None`` if it has none."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = fcntl.ioctl(probe.fileno(), _SIOCGIFBRDADDR,
                             struct.pack("256s", name[:15].encode()))
        addr = socket.inet_ntoa(packed[20:24])
    except OSError:
        return None
    finally:
        probe.close()
    return addr if addr and addr != "0.0.0.0" else None


def discover_targets(all_interfaces: bool = False) -> tuple[str, ...]:
    """Addresses to send each sample to.

    Defaults to the default-route interface only — a box with VPN or secondary
    subnets should not spray telemetry at every peer on them; that is opt-in via
    ``all_interfaces``. ``255.255.255.255`` is always appended because some
    stacks deliver the limited broadcast when they drop the subnet-directed one,
    and vice versa; consumers de-duplicate on the envelope instead.
    """
    names = _interfaces()
    if not all_interfaces:
        primary = _default_interface()
        names = [primary] if primary in names else names
    found: list[str] = []
    for name in names:
        addr = _broadcast_address(name)
        if addr and addr not in found:
            found.append(addr)
    if "255.255.255.255" not in found:
        found.append("255.255.255.255")
    return tuple(found)


# ── the broadcaster ──────────────────────────────────────────────────────────


class LanBroadcaster:
    """Sends an already-built envelope to the LAN. Holds one socket.

    Construction is cheap and side-effect free: the socket is opened on first
    successful ``publish`` and only when the flag is on, so a disabled engine
    never touches the network stack.
    """

    def __init__(self, targets: tuple[str, ...] | None = None,
                 all_interfaces: bool = False) -> None:
        self._targets = targets
        self._all_interfaces = all_interfaces
        self._sock: Any = None

    @property
    def targets(self) -> tuple[str, ...]:
        if self._targets is None:
            self._targets = discover_targets(self._all_interfaces)
        return self._targets

    def _socket(self) -> Any:
        if self._sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Non-blocking: a full send buffer must drop this sample rather
            # than stall the sampler thread — the next one supersedes it.
            sock.setblocking(False)
            self._sock = sock
        return self._sock

    def publish(self, sample: dict[str, Any]) -> bool:
        """Broadcast one sample. Returns True if at least one datagram left.

        Takes the raw sample and wraps it with the shared envelope builder, so
        this path and the NOTIFY path can never drift apart.
        """
        if not enabled():
            return False
        try:
            from queue_workflows.hw_metrics import build_envelope
            payload = build_envelope(sample).encode()
            sock = self._socket()
        except Exception:
            log.exception("[hw_lan] could not prepare broadcast")
            return False

        target_port = port()
        sent = 0
        for addr in self.targets:
            try:
                sock.sendto(payload, (addr, target_port))
                sent += 1
            except Exception:
                # One dead interface must not stop the others, and no network
                # error may ever reach the sampler thread.
                log.debug("[hw_lan] send to %s failed", addr, exc_info=True)
        return sent > 0

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
