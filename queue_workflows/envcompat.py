"""Env-var lookup — every engine knob is ``QUEUE_WORKFLOWS_*``.

:func:`env_get` is the ONE lookup path every engine module uses. The engine
once also read an ``AI_LEADS_*`` twin of each knob as a silent fallback; that
fallback is gone. A legacy spelling still set in the environment is an ERROR
naming its replacement, so an un-migrated deploy fails at startup instead of
silently dropping its setting (``tests/test_env_compat.py``).

A name without the canonical prefix (a host-configured custom name, e.g.
``configure(db_url_env="MYAPP_DB_URL")``) is read verbatim.

This module is a **leaf** (imports nothing from the engine) so any module,
including ``config``, can use it without cycles.
"""
from __future__ import annotations

import os

CANONICAL_PREFIX = "QUEUE_WORKFLOWS_"
LEGACY_PREFIX = "AI_LEADS_"


def env_get(name: str, default: str | None = None) -> str | None:
    """``os.environ.get`` that refuses the retired ``AI_LEADS_*`` twin of a
    ``QUEUE_WORKFLOWS_*`` knob (see module docstring)."""
    if name.startswith(CANONICAL_PREFIX):
        legacy = LEGACY_PREFIX + name[len(CANONICAL_PREFIX):]
        if legacy in os.environ:
            raise RuntimeError(
                f"{legacy} is no longer read: rename it to {name} "
                f"(the AI_LEADS_* env names were retired)"
            )
    return os.environ.get(name, default)
