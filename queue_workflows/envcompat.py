"""Env-var name compatibility — canonical ``QUEUE_WORKFLOWS_*``, legacy ``AI_LEADS_*``.

The engine was extracted from a host application and its runtime knobs
originally kept that host's env-var names so the first deploy needed zero
``.env`` changes. As a standalone library the canonical names are now
``QUEUE_WORKFLOWS_*``; the legacy spellings keep working as a silent fallback.

:func:`env_get` is the ONE lookup path every engine module uses (a source-scan
test, ``tests/test_env_compat.py``, forbids direct ``os.environ.get("AI_LEADS_…")``
reads). Resolution order for a name carrying either known prefix:

    1. ``QUEUE_WORKFLOWS_<suffix>``  (canonical — always wins)
    2. ``AI_LEADS_<suffix>``         (legacy fallback)
    3. the caller's ``default``

A name with neither prefix (a host-configured custom name, e.g.
``configure(db_url_env="MYAPP_DB_URL")``) is read verbatim — no twin lookup.

This module is a **leaf** (imports nothing from the engine) so any module,
including ``config``, can use it without cycles.
"""
from __future__ import annotations

import os

CANONICAL_PREFIX = "QUEUE_WORKFLOWS_"
LEGACY_PREFIX = "AI_LEADS_"


def env_get(name: str, default: str | None = None) -> str | None:
    """``os.environ.get`` with the canonical→legacy twin fallback (see module
    docstring). Accepts either spelling — or any custom name, read verbatim."""
    suffix = None
    for prefix in (CANONICAL_PREFIX, LEGACY_PREFIX):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            break
    if suffix is None:  # custom host-configured name — no twins
        return os.environ.get(name, default)
    for candidate in (CANONICAL_PREFIX + suffix, LEGACY_PREFIX + suffix):
        value = os.environ.get(candidate)
        if value is not None:
            return value
    return default
