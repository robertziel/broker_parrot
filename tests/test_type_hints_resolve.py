"""Latent-NameError guard: public/typed functions' annotations must RESOLVE.

``from __future__ import annotations`` makes annotations lazy strings, so a
missing import (``Iterable``, ``Any``) never crashes at import time — it crashes
the first time anything introspects the hints (``typing.get_type_hints``:
dataclasses, pydantic-ish tooling, doc generators). These two regressed exactly
that way (ruff F821); resolving the hints here keeps them honest.
"""

from __future__ import annotations

import typing


def test_claim_worker_predict_default_kind_hints_resolve():
    from queue_workflows import claim_worker

    hints = typing.get_type_hints(claim_worker.predict_default_kind)
    assert "warm_servers" in hints and "return" in hints


def test_vllm_docker_api_hints_resolve():
    from queue_workflows.llm_backends import vllm

    hints = typing.get_type_hints(vllm._docker_api)
    assert "return" in hints
