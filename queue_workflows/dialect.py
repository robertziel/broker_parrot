"""SQL dialect seam — the one place Postgres vs SQLite divergence lives.

``queue_workflows`` is a Postgres-as-queue engine whose hot paths use
Postgres-only SQL (``FOR UPDATE SKIP LOCKED``, ``make_interval``, ``EXTRACT
(EPOCH …)``, ``= ANY(…::text[])``, ``LEAST``). To let the SAME engine run on a
single SQLite file (a local, RAM/disk-light deploy with no daemon) WITHOUT
forking every module, the divergent fragments are produced by a process-wide
:class:`Dialect` selected from ``config.db_backend``:

* ``db_backend="pg"``            → :class:`PgDialect` — emits **exactly** the SQL
  the engine has always used, so the live Postgres deploy is byte-identical.
* ``db_backend="sqlite"`` (default, v1.0.0) → :class:`SqliteDialect` — the SQLite renderings.

These are the only two engine backends: ``configure()`` and the
``QUEUE_WORKFLOWS_DB_BACKEND`` env default reject every other value
(:func:`queue_workflows.config.canonical_db_backend`), so ``db_backend`` is
always exactly ``"pg"`` or ``"sqlite"`` by the time a dialect is chosen.

DESIGN: the dialect returns SQL *fragments* in **pyformat** placeholders
(``%s`` / ``%(name)s``) — the SAME paramstyle the engine writes everywhere — so a
caller can splice a fragment into a pyformat query and the sqlite connection's
string-literal-aware translator (:mod:`queue_workflows.db`) converts the whole
statement to qmark/named at execute time. The one exception is :meth:`epoch`,
whose SQLite rendering contains a literal ``'%s'`` inside a string (``strftime``)
— safe precisely because that translator skips string-literal content.

This module imports nothing from other engine modules (leaf, like ``config``).
"""

from __future__ import annotations

from typing import Any


class Dialect:
    """Base dialect. Defaults are the Postgres renderings; :class:`SqliteDialect`
    overrides the ones that differ. Subclasses are stateless singletons."""

    name = "base"

    # ── current time (UTC) ────────────────────────────────────────────────
    @property
    def now(self) -> str:
        """SQL expression for 'current timestamp, UTC'."""
        return "now()"

    # ── interval arithmetic (the fragment embeds a pyformat placeholder) ───
    def future_seconds(self, secs_param: str) -> str:
        """``now() + <secs_param> seconds``. ``secs_param`` is a pyformat
        placeholder (e.g. ``%(lease_s)s``) or a literal int expression."""
        return f"now() + make_interval(secs => {secs_param})"

    def past_seconds(self, secs_param: str) -> str:
        """``now() - <secs_param> seconds``."""
        return f"now() - make_interval(secs => {secs_param})"

    def past_days(self, days_param: str) -> str:
        """``now() - <days_param> days``."""
        return f"now() - make_interval(days => {days_param})"

    # ── ordering / numeric helpers ────────────────────────────────────────
    def epoch(self, col: str) -> str:
        """Seconds-since-epoch of a timestamp column, as a number."""
        return f"EXTRACT(EPOCH FROM {col})"

    def creation_order(self, alias: str) -> str:
        """A monotonic-with-creation numeric expression for the ``host_priority``
        -directed FIFO tiebreak. pg uses the (sub-second) creation epoch; SQLite
        uses the implicit ``rowid`` (strictly increasing with INSERT, so it never
        ties — unlike a whole-/milli-second timestamp for rapid inserts)."""
        return f"EXTRACT(EPOCH FROM {alias}.created_at)"

    def least(self, *exprs: str) -> str:
        """Scalar minimum of N expressions (``LEAST`` on pg)."""
        return f"LEAST({', '.join(exprs)})"

    def greatest(self, *exprs: str) -> str:
        """Scalar maximum of N expressions (``GREATEST`` on pg)."""
        return f"GREATEST({', '.join(exprs)})"

    # ── claim concurrency ─────────────────────────────────────────────────
    @property
    def skip_locked(self) -> str:
        """Row-level skip-locked clause for the claim subselect."""
        return "FOR UPDATE SKIP LOCKED"

    def box_claim_serializer(self) -> str | None:
        """Statement that serializes CLAIMS per box inside the claim transaction —
        run before the slot-gated claim so two concurrent claimants can't both read
        "N-1 running" and both take the box's last slot. pg: an advisory xact lock
        keyed on (host, queue, project), auto-released at commit. ``None`` ⇒ the
        store needs no extra serialization (SQLite's single writer already
        serializes every claim)."""
        return (
            "SELECT pg_advisory_xact_lock(hashtext("
            "%(host)s || '|' || %(queue)s || '|' || %(project)s))"
        )

    def queue_claim_serializer(self) -> str | None:
        """Statement that serializes CLAIMS fleet-wide per (queue, project) —
        the download lane's width cap counts running rows across ALL hosts, so
        (unlike :meth:`box_claim_serializer`) the key must NOT include the
        host: two claimants on different boxes must contend for the SAME lock.
        pg: an advisory xact lock, auto-released at commit; the 'queue-wide|'
        prefix keeps its hash domain distinct from the box serializer's
        'host|queue|project' strings. ``None`` ⇒ the store needs no extra
        serialization (SQLite's single writer already serializes every
        claim)."""
        return (
            "SELECT pg_advisory_xact_lock(hashtext("
            "'queue-wide|' || %(queue)s || '|' || %(project)s))"
        )

    def not_distinct_from(self, a: str, b: str) -> str:
        """Null-safe equality (``NULL`` equals ``NULL``) — the warm-model
        affinity tiebreak. pg: ``a IS NOT DISTINCT FROM b``."""
        return f"{a} IS NOT DISTINCT FROM {b}"

    # ── arrays (text[] on pg; JSON text on sqlite) ────────────────────────
    def array_contains_value(self, array_col: str, value_expr: str) -> str:
        """True iff the array COLUMN ``array_col`` contains ``value_expr``
        (e.g. ``j.required_model = ANY(lg.fits_models)``)."""
        return f"{value_expr} = ANY({array_col})"

    def value_in_param_array(self, value_expr: str, array_param: str) -> str:
        """True iff ``value_expr`` is in the bound list parameter ``array_param``
        (e.g. capability gate ``c.required_model = ANY(%(known)s::text[])``).
        Pair with :meth:`array_param` to encode the bound value."""
        return f"{value_expr} = ANY({array_param}::text[])"

    def arrays_overlap(self, array_col: str, array_param: str) -> str:
        """True iff the array COLUMN ``array_col`` shares at least one element with
        the bound list parameter ``array_param`` — the per-node-job LLM server
        preference gate (migration 0022): a worker claims a row iff the kinds it
        serves INTERSECT the row's preference. pg has a native operator.

        Also accepts a second array COLUMN as ``array_param`` (both sides rendered
        as columns) — the stranded-row sweep joins ``server_types`` against
        ``llm_servers_available``."""
        if array_param.startswith("%"):
            # Any bound parameter (named %(x)s OR positional %s) needs the cast; a
            # column reference never starts with '%'. The old '%(' test silently
            # dropped the cast for positional params.
            return f"{array_col} && {array_param}::text[]"
        return f"{array_col} && {array_param}"

    def array_to_text(self, array_col: str) -> str:
        """Render an array column as human-readable text (for a reason message)."""
        return f"array_to_string({array_col}, ',')"

    def array_is_empty(self, array_col: str) -> str:
        """True iff the array column holds ZERO elements (NOT NULL, just empty)."""
        return f"cardinality({array_col}) = 0"

    def first_overlap(self, array_col: str, array_param: str) -> str:
        """The FIRST element of the ordered array COLUMN that appears in the bound
        list parameter — the per-job server preference resolved in SQL, so the claim
        can stamp the kind the arbiter just counted (migration 0023). NULL when
        nothing overlaps."""
        return (f"(SELECT s.k FROM unnest({array_col}) WITH ORDINALITY AS s(k, ord) "
                f"WHERE s.k = ANY({array_param}::text[]) ORDER BY s.ord LIMIT 1)")

    def array_first(self, array_col: str) -> str:
        """The first element of the array column (its top preference)."""
        return f"({array_col})[1]"

    def array_param(self, values: Any) -> Any:
        """Encode a python list for binding as an array parameter. pg binds a
        list directly (psycopg adapts to ``text[]``)."""
        return list(values)

    def array_literal(self, values: list[str]) -> Any:
        """Encode a python list for storing INTO an array column. pg binds a
        list directly."""
        return list(values)

    # ── RETURNING ─────────────────────────────────────────────────────────
    def qualify_returning(self, alias: str, cols: tuple[str, ...]) -> str:
        """Render a ``RETURNING`` column list for target-table columns. pg keeps
        the ``alias.`` qualifier (needed to disambiguate when an ``UPDATE … FROM``
        join brings in a same-named column); SQLite RETURNING cannot alias-qualify
        (and only sees the target table), so it drops the qualifier."""
        return ", ".join(f"{alias}.{c}" for c in cols)

    # ── schema introspection ──────────────────────────────────────────────
    def table_exists(self, table_param: str) -> str:
        """SQL returning a non-null value iff a table named by ``table_param``
        (a pyformat placeholder bound to the bare table name) exists."""
        return f"to_regclass('public.' || {table_param})"


class PgDialect(Dialect):
    name = "pg"


class SqliteDialect(Dialect):
    name = "sqlite"

    def box_claim_serializer(self) -> str | None:
        # SQLite has ONE writer at a time — every claim is already serialized by
        # the store itself, so no extra per-box lock statement is needed.
        return None

    def queue_claim_serializer(self) -> str | None:
        # Same single-writer argument as box_claim_serializer: the claim
        # UPDATE (count guard included) is one atomic statement under SQLite's
        # write lock, so no fleet-wide lock statement is needed either.
        return None

    @property
    def now(self) -> str:
        # ISO-8601 UTC to the second — lexically comparable + matches the TEXT
        # timestamps the migrations store. SQLite has no tz type; everything is
        # UTC by convention (datetime('now') is already UTC).
        return "datetime('now')"

    def future_seconds(self, secs_param: str) -> str:
        # datetime('now', '+<n> seconds'); the modifier is built by concat so the
        # placeholder stays a real bound param.
        return f"datetime('now', ('+' || {secs_param} || ' seconds'))"

    def past_seconds(self, secs_param: str) -> str:
        return f"datetime('now', ('-' || {secs_param} || ' seconds'))"

    def past_days(self, days_param: str) -> str:
        return f"datetime('now', ('-' || {days_param} || ' days'))"

    def epoch(self, col: str) -> str:
        # strftime('%s', …) → unix seconds as text; cast to REAL for arithmetic.
        # The literal '%s' lives inside a string → the db.py translator skips it.
        return f"CAST(strftime('%s', {col}) AS REAL)"

    def creation_order(self, alias: str) -> str:
        # rowid is monotonic with INSERT and never ties (a whole-/milli-second
        # timestamp ties for rapid inserts; rowid does not).
        return f"{alias}.rowid"

    def least(self, *exprs: str) -> str:
        # SQLite overloads MIN/MAX as scalar funcs with N args.
        return f"MIN({', '.join(exprs)})"

    def greatest(self, *exprs: str) -> str:
        return f"MAX({', '.join(exprs)})"

    @property
    def skip_locked(self) -> str:
        # SQLite serializes writers (WAL + busy_timeout), so the single-statement
        # UPDATE…WHERE id=(SELECT … LIMIT 1) claim is already atomic; there is no
        # row-level skip-locked clause.
        return ""

    def not_distinct_from(self, a: str, b: str) -> str:
        # SQLite's IS / IS NOT are already null-safe.
        return f"{a} IS {b}"

    def array_contains_value(self, array_col: str, value_expr: str) -> str:
        return (
            f"EXISTS (SELECT 1 FROM json_each({array_col}) "
            f"WHERE value = {value_expr})"
        )

    def value_in_param_array(self, value_expr: str, array_param: str) -> str:
        # array_param is bound as a JSON-text list (see array_param()).
        return f"{value_expr} IN (SELECT value FROM json_each({array_param}))"

    def arrays_overlap(self, array_col: str, array_param: str) -> str:
        # Both sides are JSON-text lists on sqlite (columns or bound params alike);
        # the overlap is a correlated EXISTS over the column's elements filtered by
        # the other side's elements.
        return (
            f"EXISTS (SELECT 1 FROM json_each({array_col}) "
            f"WHERE value IN (SELECT value FROM json_each({array_param})))"
        )

    def array_to_text(self, array_col: str) -> str:
        # JSON text is already readable enough for a reason message.
        return array_col

    def array_is_empty(self, array_col: str) -> str:
        return f"json_array_length({array_col}) = 0"

    def first_overlap(self, array_col: str, array_param: str) -> str:
        return (f"(SELECT j1.value FROM json_each({array_col}) j1 "
                f"WHERE j1.value IN (SELECT value FROM json_each({array_param})) "
                f"ORDER BY j1.key LIMIT 1)")

    def array_first(self, array_col: str) -> str:
        return f"(SELECT value FROM json_each({array_col}) ORDER BY key LIMIT 1)"

    def array_param(self, values: Any) -> Any:
        import json
        return json.dumps(list(values))

    def array_literal(self, values: list[str]) -> Any:
        import json
        return json.dumps(list(values))

    def qualify_returning(self, alias: str, cols: tuple[str, ...]) -> str:
        return ", ".join(cols)

    def table_exists(self, table_param: str) -> str:
        return (
            f"(SELECT name FROM sqlite_master "
            f"WHERE type = 'table' AND name = {table_param})"
        )


_PG = PgDialect()
_SQLITE = SqliteDialect()


def get_dialect() -> Dialect:
    """Return the process-wide dialect for the engine's relational store, chosen
    from ``config.db_backend``. ``sqlite`` → :class:`SqliteDialect`; ``pg`` →
    :class:`PgDialect` (the only other value the validator admits)."""
    from queue_workflows.config import get_config
    return _SQLITE if get_config().db_backend == "sqlite" else _PG


def is_sqlite() -> bool:
    """True when the engine's relational store is SQLite."""
    from queue_workflows.config import get_config
    return get_config().db_backend == "sqlite"
