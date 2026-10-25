# 🗄️ Storage backends

*The `db_backend` seam: `sqlite` or `pg`. Both run the full DAG engine.*

## The selection

```python
import queue_workflows
queue_workflows.configure(db_backend="sqlite")   # "sqlite" (default) | "pg"
```

or the env knob (reaches the standalone console scripts too, which have no host
`configure()` call):

```bash
export QUEUE_WORKFLOWS_DB_BACKEND=pg
```

`db_backend` defaults to **`"sqlite"`** as of v1.0.0 — a daemon-less local file,
the friendliest zero-config default for a reusable library. A Postgres consumer
must opt in with `configure(db_backend="pg")` or the env var above; otherwise a
`QUEUE_WORKFLOWS_DB_URL` Postgres DSN gets read as a SQLite path. Aliases
`postgres`/`postgresql` → `pg` are accepted (case- and whitespace-insensitive);
any other name raises `ValueError` at `configure()` time, at env read, or from
`queue-hw-watch --db-backend` — one validator,
`queue_workflows.config.canonical_db_backend`, never a substitute store. See
[configuration.md](configuration.md) for the full `EngineConfig` reference.

## One engine, two dialects

Both backends run the *entire* engine — orchestrator, claim loop,
lease/reclaim, dispatch outbox, watchdogs, worker control — through one dialect
seam, `queue_workflows/dialect.py`. `PgDialect` emits exactly the SQL the engine
has always used (`FOR UPDATE SKIP LOCKED`, `make_interval`, `pg_notify`);
`SqliteDialect` renders the same statements for a single SQLite file, with
`queue_workflows.db` translating pyformat placeholders to qmark/named at execute
time. Selecting between them changes *where the DAG runs*, not *what runs it*.
See [architecture.md](architecture.md) and [schema.md](schema.md) for the
relational schema the two share (`migrations/` and its `migrations_sqlite/` twin).

| | `sqlite` | `pg` |
|---|---|---|
| Server process | none (a file) | Postgres 14+ |
| Worker wake | 1 s safety poll | `LISTEN`/`NOTIFY` (trigger in the writer's txn) + 1 s poll |
| Claim exclusion | WAL + `busy_timeout`; serialized writers | `FOR UPDATE SKIP LOCKED` |
| Multi-box fleet / shared broker (`project`) | single box | yes |
| Fits | local dev, a single-box fleet | a second machine and up |

Move to Postgres when you add a second machine.
