"""Per-box LLM-server TOPOLOGY — a small YAML that maps each worker box (by its
``host_label``) to the LLM server ROOT URL that box should DISPATCH to.

WHY THIS EXISTS. The backend :mod:`~queue_workflows.llm_backends.factory` resolves a
machine's LLM server *type* from the DB (``worker_controls``, migration 0013) but
its ROOT *URL* from a single per-process env (``ollama_url_env`` / ``vllm_url_env``).
One env is fine when every box talks to the same server — but a real fleet wants
DIFFERENT addresses per box: the GPU host runs ollama locally
(``http://127.0.0.1:11434``) while a CPU/client box must reach it across the fast
interconnect (``http://box-b:11434``). This module lets a deployment express
that as a small YAML; broker_parrot then hands each box the address that is BEST
*for that box*, keyed by the box's own ``host_label`` — "broker_parrot knows what
to send to a given box as its LLM address".

The file is DEPLOYMENT topology (real hostnames / IPs), so a project keeps the live
one gitignored and commits a ``.example``. Absent file / absent entry ⇒ ``None`` and
the factory falls back to the env + localhost default exactly as before: the whole
mechanism is OPT-IN and byte-compatible for any consumer that never sets
``EngineConfig.llm_topology_path``.

Parsing needs **PyYAML**, which is an OPTIONAL extra (``pip install
'queue_workflows[topology]'``) — psycopg remains the engine's only hard runtime
dependency. Without it this module logs once and resolves ``None``, so a consumer
that sets a topology path but never installed the extra degrades to the env default
instead of raising on the node thread.

Format (``url`` may be a bare string or a ``{url: …}`` mapping so future per-box
fields can be added without a schema break)::

    boxes:
      box-a-gpu:  http://127.0.0.1:11434       # runs its own ollama locally
      box-b-gpu:  http://127.0.0.1:11434       # runs its own ollama locally
      box-c-gpu:  http://box-b:11434           # no local server → reach box-b's
      default:    http://127.0.0.1:11434       # any other box → its own local server

A ``default:`` entry is the catch-all for a box with no explicit row.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

log = logging.getLogger(__name__)

# path -> (mtime, {host_label: {"url": str}}). Cached by (path, mtime) so a hot
# request loop (get_backend runs per node-job) never re-reads/re-parses the file
# unless it actually changed on disk — an operator edit is picked up on the next
# call without a restart, but a steady state costs one stat().
_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}
_DOC_CACHE: dict[str, tuple[float, dict]] = {}

#: Per-kind URL keys an entry may carry (migration 0022 — co-resident servers).
#: Kept as a literal rather than importing ``worker_control.VALID_SERVER_TYPES`` so
#: this module stays a leaf with no engine imports (same discipline as config.py).
_SERVER_KINDS = ("ollama", "vllm")


def resolve(path: str, host_label: str) -> dict[str, Any] | None:
    """Return the topology entry for ``host_label`` — a dict with at least
    ``"url"`` (the LLM server ROOT URL that box dispatches to) — falling back to a
    ``default:`` entry, or ``None`` when the file/entry is absent.

    Never raises: a missing file, unreadable YAML, or a malformed ``boxes:`` block
    resolves to ``None`` (logged once per mtime) so a broken topology file can
    never take down dispatch — the factory then uses its env/localhost default."""
    boxes = _load(path)
    if not boxes:
        return None
    return boxes.get(host_label) or boxes.get("default")


def resolve_for(path: str, host_label: str, server_type: str) -> str | None:
    """The ROOT URL this box dispatches to **for one server KIND**.

    Two servers can now be co-resident on a box (migration 0022), and they answer
    on different ports — so a single per-box ``url`` is no longer enough. An entry
    may name a URL per kind::

        boxes:
          box-a-gpu:
            ollama: http://127.0.0.1:11434
            vllm:   http://127.0.0.1:8000

    BACK-COMPAT is the point of this function: a bare-string or ``{url: …}`` entry
    (every topology written before 0022) still answers for **both** kinds, so an
    existing deployment behaves exactly as before until it opts in per kind.
    Resolution order is kind-specific key → the shared ``url`` → ``None`` (caller
    falls back to the env / localhost default)."""
    entry = resolve(path, host_label)
    if not entry:
        return None
    specific = str(entry.get(server_type) or "").strip()
    if specific:
        return specific
    return str(entry.get("url") or "").strip() or None


def setting(path: str, key: str, default: Any = None) -> Any:
    """Read one value from the topology file's ``settings:`` block.

    The file already carries per-box deployment topology; operator POLICY that
    varies the same way belongs beside it rather than in a second file or a redeploy.
    Re-read on mtime change like everything else here, so an edit takes effect
    without restarting a worker::

        settings:
          vllm_min_queue: 5      # jobs that justify a vLLM cold start

    Never raises — a missing file, missing block or missing key returns ``default``."""
    try:
        doc = _load_doc(path)
    except Exception:
        return default
    block = doc.get("settings") if isinstance(doc, dict) else None
    if not isinstance(block, dict):
        return default
    return block.get(key, default)


def _load(path: str) -> dict[str, dict[str, Any]]:
    """Parse + cache the topology YAML at ``path`` into ``{host_label: {"url": …}}``,
    keyed by mtime. Returns ``{}`` for any absent/broken file."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    with _LOCK:
        cached = _CACHE.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
    boxes = _parse(path)
    with _LOCK:
        _CACHE[path] = (mtime, boxes)
    return boxes


def _load_doc(path: str) -> dict:
    """The whole parsed YAML document, mtime-cached (the boxes cache holds only the
    boxes mapping, and ``settings`` lives outside it)."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    with _LOCK:
        cached = _DOC_CACHE.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
    except Exception:
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    with _LOCK:
        _DOC_CACHE[path] = (mtime, doc)
    return doc


def _parse(path: str) -> dict[str, dict[str, Any]]:
    # PyYAML is an OPTIONAL extra — psycopg stays the engine's only hard runtime
    # dependency — so the import rides INSIDE the guard like every other failure
    # mode here. A deployment that sets ``llm_topology_path`` without installing it
    # degrades to the env/localhost default rather than raising ModuleNotFoundError
    # on the node thread (this module never breaks dispatch).
    try:
        import yaml
    except ImportError:
        log.warning(
            "[llm-topology] PyYAML is not installed — ignoring %s and using the "
            "env/default LLM URL. Install the optional extra to enable per-box "
            "topology: pip install 'queue_workflows[topology]'",
            path,
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
    except Exception:
        log.exception("[llm-topology] failed to read %s — using env/default", path)
        return {}
    raw = doc.get("boxes") if isinstance(doc, dict) else None
    if not isinstance(raw, dict):
        log.warning(
            "[llm-topology] %s has no 'boxes:' mapping — ignoring (env/default)", path
        )
        return {}
    out: dict[str, dict[str, Any]] = {}
    for label, val in raw.items():
        # A value is either a bare URL string, a {url: …} mapping (both apply to
        # EVERY server kind), or a per-kind mapping ({ollama: …, vllm: …}) added in
        # 0022 for co-resident servers. A per-kind mapping needs no shared ``url``.
        if isinstance(val, str) and val.strip():
            out[str(label)] = {"url": val.strip()}
        elif isinstance(val, dict) and str(val.get("url") or "").strip():
            out[str(label)] = {**val, "url": str(val["url"]).strip()}
        elif isinstance(val, dict) and any(
            str(val.get(k) or "").strip() for k in _SERVER_KINDS
        ):
            out[str(label)] = {
                **val,
                **{k: str(val[k]).strip() for k in _SERVER_KINDS if val.get(k)},
            }
    return out


def _reset_cache_for_tests() -> None:
    """TEST-ONLY: drop the mtime cache so a test can rewrite a topology file at the
    same path and observe the new content."""
    with _LOCK:
        _CACHE.clear()
        _DOC_CACHE.clear()


__all__ = ["resolve", "resolve_for", "setting"]
