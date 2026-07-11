"""Reference-resolution mini-language (carved from ai_leads ``registry.py``).

The dispatcher resolves a node's ``inputs`` (and ``skip_if`` predicates,
``choose_one``/``pick_perspective`` sources) against the run context using a
tiny ``$value`` / ``$from`` / ``$filter`` / ``$eq`` / ``$ne`` mini-language.
That resolver is generic DAG plumbing — it knows nothing about the host's
domain — so it lives in the engine as a pure leaf module (no imports beyond
the stdlib). Both the engine's dispatcher and the host's workflow loader use
it; the host can also inject its own via ``config.resolve_ref`` if it needs a
superset.
"""

from __future__ import annotations

from typing import Any


def resolve_ref(value: Any, context: dict) -> Any:
    """Resolve ``{"$value": X}`` / ``{"$from": "path"}`` / ``{"$filter": {...}}``
    / ``{"$from": "...", "$eq": X}`` / ``{"$from": "...", "$ne": X}``
    entries against the workflow context.

    Supported forms::

        {"$value": 42}                       → 42
        {"$from": "parcel.lat"}              → context["parcel"]["lat"]
        {"$from": "svinfer.primary_file"}    → context["svinfer"]["primary_file"]
        {"$from": "svinfer.files",           → filter array items
         "$filter": {"kind:eq": "image",
                     "rel_path:matches": "annotated_pano_\\d+"}}
        {"$from": "turnout.path",            → True if context["turnout"]["path"] == "rotate"
         "$eq": "rotate"}
        {"$from": "turnout.path",            → True if context["turnout"]["path"] != "direct"
         "$ne": "direct"}
        "literal string"                      → "literal string"  (as-is)

    The ``$eq`` / ``$ne`` forms back the dispatcher's ``skip_if`` field:
    a step with ``"skip_if": {"$from": "turnout.path", "$ne": "rotate"}``
    is skipped on every branch except the one where the user picked
    ``"rotate"``.

    Objects without $value/$from/$filter are treated as literal dicts.
    """
    if not isinstance(value, dict):
        return value
    if "$value" in value:
        return value["$value"]
    if "$coalesce" in value:
        return _coalesce(value, context)
    if "$from" in value:
        result = _dig(context, value["$from"])
        if "$filter" in value:
            if not isinstance(result, list):
                raise TypeError(
                    f"$filter on non-list at {value['$from']!r}: {type(result).__name__}"
                )
            return [item for item in result if _match(item, value["$filter"])]
        if "$eq" in value:
            return result == value["$eq"]
        if "$ne" in value:
            return result != value["$ne"]
        return result
    # Plain dict → recurse so nested $from work as object values
    return {k: resolve_ref(v, context) for k, v in value.items()}


def _coalesce(value: dict, context: dict) -> Any:
    """Resolve ``{"$coalesce": [ref1, ref2, …], "$default": X?}`` to the FIRST
    candidate that resolves to a non-None value — SKIPPING candidates whose ref
    is missing from the context (a ``KeyError`` from ``$from``).

    This is the lane-CONVERGENCE primitive: N mutually-exclusive lanes feed one
    shared downstream step (``depends_on`` all of them; the unchosen lanes are
    ``skip_if``-skipped and write no context), and the step reads
    ``{"$coalesce": [{"$from": "lane_a.out"}, {"$from": "lane_b.out"}]}`` to pick
    whichever lane actually ran — without the duplicated-tail workaround. A
    skipped lane's ``$from`` raises ``KeyError`` (the step isn't in the context);
    ``$coalesce`` swallows that and tries the next candidate.

    All candidates missing/None → return ``$default`` if given, else raise
    ``KeyError`` (a genuinely unresolved ref is still an error)."""
    candidates = value["$coalesce"]
    if not isinstance(candidates, list):
        raise TypeError(
            f"$coalesce expects a list, got {type(candidates).__name__}"
        )
    for cand in candidates:
        try:
            resolved = resolve_ref(cand, context)
        except KeyError:
            continue
        if resolved is not None:
            return resolved
    if "$default" in value:
        return value["$default"]
    raise KeyError(
        f"$coalesce: no candidate resolved (and no $default): {candidates!r}"
    )


def _dig(obj: Any, dotted_path: str) -> Any:
    """Walk a dotted path like ``svinfer.summary.top_score``.

    Each path segment looks up either a dict key or a dataclass attr.
    Empty/missing segments raise KeyError.
    """
    cur = obj
    for part in dotted_path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"{dotted_path}: missing segment {part!r}")
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                raise KeyError(f"{dotted_path}: missing attr {part!r}")
            cur = getattr(cur, part)
    return cur


def _match(item: Any, filter_spec: dict) -> bool:
    """Apply one filter spec against an item.

    Filter keys are ``field:op`` pairs. Supported ops: ``eq``, ``matches`` (regex).
    """
    import re
    for key, expected in filter_spec.items():
        if ":" not in key:
            raise ValueError(f"filter key missing op: {key!r}")
        field, op = key.split(":", 1)
        actual = item.get(field) if isinstance(item, dict) else getattr(item, field, None)
        if op == "eq":
            if actual != expected:
                return False
        elif op == "matches":
            if actual is None or not re.search(expected, str(actual)):
                return False
        else:
            raise ValueError(f"unsupported filter op: {op!r}")
    return True


__all__ = ["resolve_ref"]
