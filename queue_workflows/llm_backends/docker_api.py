"""One request to the docker Engine API over its unix socket. stdlib only.

Talking to the API directly — not the CLI, not compose — is what lets the engine change a
sibling container's configuration from inside its own container: compose re-reads its
file and would ignore a computed argv, and the worker image ships no docker binary. psycopg
stays the engine's single hard dependency, so this is ``http.client`` with the TCP connect
swapped for an AF_UNIX connect, and nothing else.

Callers: the vLLM model switch (:mod:`vllm_sidecar`, via :mod:`vllm`) and the LLM context
lever (:mod:`queue_workflows.llm_context_lever`).
"""
from __future__ import annotations

import json
from typing import Any

#: The socket, where a deployment bind-mounts it into the worker container.
DEFAULT_SOCK = "/var/run/docker.sock"

#: 30 s was not enough: a recreate is stop(t=30) + remove + create + start, and the stop
#: alone can use the full grace period. The call then timed out MID-CYCLE on a live box —
#: container removed, replacement not yet created — which is the worst place to give up.
#: Generous, because the alternative to waiting is a half-recreated container.
DEFAULT_TIMEOUT_S = 180


def default_socket() -> str:
    """``QUEUE_WORKFLOWS_DOCKER_SOCK`` or the usual path."""
    from queue_workflows.envcompat import env_get

    return (env_get("QUEUE_WORKFLOWS_DOCKER_SOCK") or DEFAULT_SOCK).strip() or DEFAULT_SOCK


def call(
    method: str, path: str, body: dict | None = None, *,
    sock_path: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[int, Any]:
    """``(status, parsed_body)``. Raises whatever the socket raises — a caller that must
    not fail (every caller here) classifies the failure itself."""
    import http.client
    import socket as _socket

    sock = sock_path or default_socket()

    class _UDSConnection(http.client.HTTPConnection):
        def connect(self):  # noqa: D102
            self.sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            self.sock.settimeout(timeout_s)
            self.sock.connect(sock)

    conn = _UDSConnection("localhost")
    try:
        payload = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=payload,
                     headers={"Content-Type": "application/json"} if payload else {})
        resp = conn.getresponse()
        raw = resp.read()
        try:
            return resp.status, (json.loads(raw) if raw else None)
        except ValueError:
            return resp.status, raw.decode("utf-8", "replace")
    finally:
        conn.close()


def call_raw(
    method: str, path: str, *, sock_path: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[int, bytes]:
    """``(status, raw_bytes)`` — for endpoints that do not answer JSON (``/logs``)."""
    import http.client
    import socket as _socket

    sock = sock_path or default_socket()

    class _UDSConnection(http.client.HTTPConnection):
        def connect(self):  # noqa: D102
            self.sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            self.sock.settimeout(timeout_s)
            self.sock.connect(sock)

    conn = _UDSConnection("localhost")
    try:
        conn.request(method, path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def demux_stream(raw: bytes) -> str:
    """The text of a docker log stream. A container without a TTY answers ``/logs`` as
    frames — an 8-byte header (stream type, three zero bytes, a big-endian length) before
    each chunk — which are stripped here; a TTY container's log is plain text and is
    returned as it is. A cut trailing frame yields what it has."""
    if not raw:
        return ""
    if len(raw) < 8 or raw[0] not in (0, 1, 2) or raw[1:4] != b"\x00\x00\x00":
        return raw.decode("utf-8", "replace")
    out: list[bytes] = []
    i = 0
    while i + 8 <= len(raw):
        if raw[i] not in (0, 1, 2) or raw[i + 1:i + 4] != b"\x00\x00\x00":
            out.append(raw[i:])                 # not a header after all: keep the rest
            break
        size = int.from_bytes(raw[i + 4:i + 8], "big")
        out.append(raw[i + 8:i + 8 + size])
        i += 8 + size
    return b"".join(out).decode("utf-8", "replace")


__all__ = ["DEFAULT_SOCK", "DEFAULT_TIMEOUT_S", "default_socket", "call", "call_raw",
           "demux_stream"]
