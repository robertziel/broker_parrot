"""The model CATALOG — which (box, server kind) can actually serve which model.

WHY THIS DOMAIN EXISTS. ``worker_heartbeats.llm_servers_available`` (0014) answers
"which server KINDS does this box run" and :mod:`llm_probe` observes it honestly. But
nothing anywhere answers the next question: *which MODELS can that server serve?* The
claim path approximates it with ``known_models``, a per-worker config list — so the
fleet has no queryable, observed answer, and a job naming a model no box holds simply
waits forever with nothing saying why.

The catalog is that answer, and it is deliberately the same shape as the lesson
``llm_probe`` records in its own docstring: OBSERVED, not declared. A box publishes what
its servers actually report; absence is a real state, not a default.

DESIGN (mirrors :mod:`model_cache`): the domain is DB-FREE and I/O-INJECTED. Nothing
here imports psycopg; every side effect is a callback. That is what lets these tests run
with no database at all.
"""
from __future__ import annotations

import pytest

from queue_workflows.model_catalog import (
    CatalogEntry,
    ModelCatalog,
    acceptable_models,
    best_match,
)


# ── the catalog: who can serve what ──────────────────────────────────
def test_a_box_that_serves_nothing_is_an_EMPTY_answer_not_a_default():
    """The failure llm_probe was built for: a box advertising a capability it lacks."""
    cat = ModelCatalog(fetch=lambda: [])
    assert cat.models_for("box-a-gpu", "vllm") == []
    assert cat.can_serve("box-a-gpu", "vllm", "chat-model-a:latest") is False


def test_can_serve_is_per_box_AND_per_server_kind():
    cat = ModelCatalog(fetch=lambda: [
        CatalogEntry("box-a-gpu", "vllm", "chat-model-a:latest"),
        CatalogEntry("box-a-gpu", "ollama", "chat-expert-b:latest"),
        CatalogEntry("box-b-gpu", "ollama", "chat-model-a:latest"),
    ])
    # same box, different kind
    assert cat.can_serve("box-a-gpu", "vllm", "chat-model-a:latest") is True
    assert cat.can_serve("box-a-gpu", "ollama", "chat-model-a:latest") is False
    # same model+kind, different box
    assert cat.can_serve("box-b-gpu", "ollama", "chat-model-a:latest") is True
    assert cat.can_serve("box-c-gpu", "ollama", "chat-model-a:latest") is False


def test_disabled_entries_are_invisible_so_an_operator_can_park_a_model():
    cat = ModelCatalog(fetch=lambda: [
        CatalogEntry("box-a-gpu", "vllm", "chat-model-a:latest", enabled=False),
    ])
    assert cat.can_serve("box-a-gpu", "vllm", "chat-model-a:latest") is False
    assert cat.models_for("box-a-gpu", "vllm") == []


def test_boxes_serving_answers_where_can_this_run():
    """The scheduler question, which today has no answer at all."""
    cat = ModelCatalog(fetch=lambda: [
        CatalogEntry("box-a-gpu", "vllm", "chat-model-a:latest"),
        CatalogEntry("box-b-gpu", "ollama", "chat-model-a:latest"),
        CatalogEntry("box-c-gpu", "ollama", "chat-expert-b:latest"),
    ])
    assert cat.boxes_serving("chat-model-a:latest") == ["box-a-gpu", "box-b-gpu"]
    assert cat.boxes_serving("nothing-holds-this") == []


def test_the_catalog_is_refreshed_not_frozen_at_construction():
    """A box that installs a model mid-flight must become claimable without a restart."""
    rows = []
    cat = ModelCatalog(fetch=lambda: list(rows), ttl_s=0)
    assert cat.can_serve("box-a-gpu", "ollama", "chat-expert-a") is False
    rows.append(CatalogEntry("box-a-gpu", "ollama", "chat-expert-a"))
    assert cat.can_serve("box-a-gpu", "ollama", "chat-expert-a") is True


# ── the job side: an ORDERED list of acceptable (model, server) pairs ─
def test_a_single_model_job_still_works_unchanged():
    """The scalar path is not deprecated — both shapes must be expressible, because
    every row written before this change carries only `required_model`."""
    assert acceptable_models(required_model="chat-model-a:latest", required_models=None) == [
        ("chat-model-a:latest", None),
    ]


def test_a_scalar_model_with_server_preference_keeps_that_preference():
    assert acceptable_models(
        required_model="chat-model-a:latest", required_models=None,
        server_types=["vllm", "ollama"],
    ) == [("chat-model-a:latest", "vllm"), ("chat-model-a:latest", "ollama")]


def test_an_array_job_expands_in_declared_order():
    """Order IS the preference — first acceptable pair a box can serve wins."""
    assert acceptable_models(
        required_model=None,
        required_models=[
            {"model": "chat-expert-a", "server_types": ["vllm"]},
            {"model": "chat-model-a:latest", "server_types": ["vllm", "ollama"]},
        ],
    ) == [("chat-expert-a", "vllm"),
          ("chat-model-a:latest", "vllm"), ("chat-model-a:latest", "ollama")]


def test_the_array_wins_when_both_are_present_and_the_scalar_is_its_first_entry():
    """Dual-write contract: a writer sets BOTH so old readers keep working. The scalar
    must be the array's head, and the array is what a new reader honours."""
    pairs = acceptable_models(
        required_model="chat-expert-a",
        required_models=[{"model": "chat-expert-a", "server_types": ["vllm"]},
                         {"model": "chat-model-a:latest", "server_types": ["ollama"]}],
    )
    assert pairs[0] == ("chat-expert-a", "vllm")
    assert ("chat-model-a:latest", "ollama") in pairs


def test_a_job_naming_no_model_is_unconstrained_not_unsatisfiable():
    assert acceptable_models(required_model=None, required_models=None) == []


# ── matching a job to a box ──────────────────────────────────────────
def _cat():
    return ModelCatalog(fetch=lambda: [
        CatalogEntry("box-a-gpu", "vllm", "chat-model-a:latest"),
        CatalogEntry("box-a-gpu", "ollama", "chat-expert-b:latest"),
        CatalogEntry("box-b-gpu", "ollama", "chat-model-a:latest"),
    ])


def test_best_match_returns_the_FIRST_acceptable_pair_this_box_can_serve():
    pairs = acceptable_models(
        required_model=None,
        required_models=[{"model": "chat-expert-a", "server_types": ["vllm"]},
                         {"model": "chat-model-a:latest", "server_types": ["vllm", "ollama"]}],
    )
    # box-a has no chat-expert-a, so the second entry wins — on vllm, its first preference.
    assert best_match(_cat(), "box-a-gpu", pairs) == ("chat-model-a:latest", "vllm")
    # box-b serves the same model only on ollama.
    assert best_match(_cat(), "box-b-gpu", pairs) == ("chat-model-a:latest", "ollama")


def test_kinds_for_is_sorted_so_two_workers_resolve_any_kind_identically():
    cat = ModelCatalog(fetch=lambda: [
        CatalogEntry("box-a-gpu", "vllm", "chat-model-a:latest"),
        CatalogEntry("box-a-gpu", "ollama", "chat-model-a:latest"),
    ])
    assert cat.kinds_for("box-a-gpu", "chat-model-a:latest") == ["ollama", "vllm"]
    # an "any kind" pair therefore resolves deterministically
    assert best_match(cat, "box-a-gpu", [("chat-model-a:latest", None)]) == \
        ("chat-model-a:latest", "ollama")


def test_best_match_is_None_when_the_box_can_serve_none_of_them():
    pairs = acceptable_models(required_model="chat-expert-a", required_models=None,
                              server_types=["vllm"])
    assert best_match(_cat(), "box-c-gpu", pairs) is None


def test_an_unconstrained_job_matches_any_box_without_naming_a_model():
    assert best_match(_cat(), "box-c-gpu", []) == (None, None)


def test_a_cold_box_with_an_EMPTY_catalog_does_not_silently_claim_everything():
    """The conservative direction. An empty catalog means 'we do not know', and a claim
    gate that treats unknown as 'yes' recreates the bug llm_probe exists to kill."""
    empty = ModelCatalog(fetch=lambda: [])
    pairs = acceptable_models(required_model="chat-model-a:latest", required_models=None)
    assert best_match(empty, "box-a-gpu", pairs) is None


@pytest.mark.parametrize("bad", [
    [{"server_types": ["vllm"]}],          # no model
    [{"model": "", "server_types": []}],   # blank model
])
def test_malformed_entries_are_dropped_rather_than_producing_a_ghost_pair(bad):
    assert acceptable_models(required_model=None, required_models=bad) == []


def test_the_package_version_matches_pyproject():
    """A wheel whose FILENAME says 1.2.0 while the package reports 1.1.0 defeats every
    version check downstream — and that is exactly what shipped on the first build here,
    because pyproject.toml was bumped and the module literal was not."""
    import pathlib
    import re

    import queue_workflows

    root = pathlib.Path(queue_workflows.__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert queue_workflows.__version__ == declared, (
        f"__init__.py says {queue_workflows.__version__}, pyproject says {declared}")


# ── narrowing a DECLARED registry to what the box actually HAS ────────
from queue_workflows.model_catalog import canonical_tag, narrow_to_observed  # noqa: E402


def test_the_bare_tag_and_latest_are_the_same_model():
    """The trap this whole helper exists for. A host's expert config declares `chat-expert-b`;
    ollama's /api/tags reports `chat-expert-b:latest`. A literal intersection drops the
    expert entirely and its lane becomes unclaimable on EVERY box."""
    assert canonical_tag("chat-expert-b:latest") == canonical_tag("chat-expert-b") == "chat-expert-b"
    assert narrow_to_observed(["chat-expert-b"], ["chat-expert-b:latest"]) == ["chat-expert-b"]
    assert narrow_to_observed(["chat-expert-b:latest"], ["chat-expert-b"]) == ["chat-expert-b:latest"]


def test_the_declared_spelling_is_what_comes_back():
    """`required_model` is stamped with the registry's spelling and the claim SQL
    compares literally — returning the observed spelling swaps one mismatch for another."""
    assert narrow_to_observed(["chat-expert-b"], ["chat-expert-b:latest"]) == ["chat-expert-b"]


def test_a_declared_model_the_box_does_not_hold_is_dropped():
    """The point: `known_ids()` is identical on every box, so without this a box that
    never pulled the expert still advertises it and can claim its rows."""
    assert narrow_to_observed(
        ["chat-model-a:latest", "chat-expert-b", "chat-expert-a"],
        ["chat-model-a:latest", "chat-expert-b:latest"],
    ) == ["chat-model-a:latest", "chat-expert-b"]


def test_NO_observation_keeps_the_full_declared_list_rather_than_wedging():
    """The asymmetry, both spellings of 'nothing observed'. A cold worker or a probe
    that has not run must not read as 'this box can serve nothing'."""
    for nothing in (None, [], ["", "  "]):
        assert narrow_to_observed(["a", "b"], nothing) == ["a", "b"]


def test_an_observation_that_matches_nothing_falls_back_to_DECLARED():
    """This asserted `== []` until a review showed the fallback was backwards.

    Downstream (`node_queue`) gates on `if known:` — an EMPTY capability list means "no
    gate", i.e. claim-ANY, which is strictly WIDER than not narrowing at all. So the
    "honest" empty result was the one direction that could let a box claim work for a
    model it had never held. Falling back to `declared` is at least as tight as the
    pre-change behaviour, which is the bound that matters."""
    assert narrow_to_observed(["chat-expert-b"], ["some-other-model"]) == ["chat-expert-b"]


def test_the_filter_still_FILTERS_when_something_does_match():
    """The other half of the pair above: the fallback must not have turned narrowing into
    a pass-through, or the capability gate is decorative."""
    assert narrow_to_observed(["chat-expert-b", "chat-expert-a"], ["chat-expert-b"]) == ["chat-expert-b"]


def test_declared_order_is_preserved_so_advertisement_is_stable():
    assert narrow_to_observed(["c", "a", "b"], ["a", "b", "c"]) == ["c", "a", "b"]


# ── the catalog must RETIRE what vanished ────────────────────────────
def _cat_rows(host=None):
    from queue_workflows import node_queue
    return sorted((e.host_label, e.server_type, e.model_id)
                  for e in node_queue.fetch_model_catalog()
                  if host is None or e.host_label == host)


def test_a_model_removed_from_a_box_stops_being_advertised():
    """The bug this table exists to prevent, reintroduced by its own writer.

    `chat-expert-b`/`chat-expert-a` were deleted from box-a's ollama store and the catalog kept
    advertising both for box-a-gpu/ollama forever, because the publisher only ever upserted.
    A catalog that only grows is a DECLARATION, which is the thing it replaced.
    """
    from queue_workflows import node_queue
    from queue_workflows.model_catalog import CatalogEntry

    node_queue.upsert_model_catalog([
        CatalogEntry("retire-box", "ollama", "keep-me"),
        CatalogEntry("retire-box", "ollama", "delete-me"),
    ])
    assert _cat_rows("retire-box") == [
        ("retire-box", "ollama", "delete-me"), ("retire-box", "ollama", "keep-me")]

    # The box now reports only one of them.
    node_queue.upsert_model_catalog([CatalogEntry("retire-box", "ollama", "keep-me")])
    assert _cat_rows("retire-box") == [("retire-box", "ollama", "keep-me")]


def test_an_UNREPORTED_server_kind_is_left_completely_alone():
    """The safety half. A probe that covered ollama says NOTHING about vllm, so a vllm row
    must survive — this is the transient-failure case the upsert-never-delete rule protects,
    and retiring across kinds would resurrect it."""
    from queue_workflows import node_queue
    from queue_workflows.model_catalog import CatalogEntry

    node_queue.upsert_model_catalog([
        CatalogEntry("two-kind-box", "ollama", "m1"),
        CatalogEntry("two-kind-box", "vllm", "m2"),
    ])
    # A round that only observed ollama.
    node_queue.upsert_model_catalog([CatalogEntry("two-kind-box", "ollama", "m1")])
    assert _cat_rows("two-kind-box") == [
        ("two-kind-box", "ollama", "m1"), ("two-kind-box", "vllm", "m2")]


def test_an_EMPTY_observation_still_blanks_nothing():
    """Unchanged: no rows in, no rows touched."""
    from queue_workflows import node_queue
    from queue_workflows.model_catalog import CatalogEntry

    node_queue.upsert_model_catalog([CatalogEntry("quiet-box", "ollama", "still-here")])
    assert node_queue.upsert_model_catalog([]) == 0
    assert _cat_rows("quiet-box") == [("quiet-box", "ollama", "still-here")]
