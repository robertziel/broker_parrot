# Setting up a project on broker_parrot — the full walkthrough

This is the **detailed, production-shaped setup**: a real node module, the pipeline +
workflow JSON that references it, the host wiring (`configure()` + seams), the three
processes, and kicking off a run. If you just want to see the engine work in 60
seconds with zero servers, start with the [README quickstart](../README.md#-quickstart--60-seconds-zero-servers)
— or hand your coding agent the [agent prompt](../README.md#-agent-friendly--just-type-it)
and let it do this whole page for you.

Related references: [`configuration.md`](configuration.md) (every hook and env knob),
[`architecture.md`](architecture.md) (the mental model), [`deployment.md`](deployment.md)
(production process layout).


Here's the whole thing end-to-end: a trivial node, a pipeline that references it, a workflow that runs that pipeline, the host wiring, and a worker that executes it. Nothing here is pseudo-code — every call is part of the public API.

## 1. Define a node — one `run()` function

A node is a module exposing a single `run(...)`. The engine introspects the signature and **auto-wires well-known kwargs** by name — `out` (the run's output dir), `status_callback`, `cancel_event`, `inputs`, `model_handle` — so you only declare the ones you use. Return a JSON-able dict; its keys become the node's `context_delta` for downstream `$from` refs.

```python
# myapp/nodes/greet.py
from pathlib import Path
from typing import Any


def run(out: Path, name: str = "world", status_callback: Any = None) -> dict:
    """Trivial CPU node: write a file, return a tiny summary."""
    if status_callback:
        status_callback(0.5, "greeting")

    out.mkdir(parents=True, exist_ok=True)
    (out / "hello.txt").write_text(f"hello, {name}!\n")

    return {"primary_file": f"{out.name}/hello.txt", "greeted": name}
```

No base class, no decorator, no registration call — discovery is by dotted module name (configured below).

## 2. Reference it from a pipeline + workflow

Two small JSON files. The **pipeline schema owns the DAG** (`nodes` with `id` / `node` / `depends_on` / `gpu` / `inputs` / `outputs`); the **workflow** strings one or more pipeline steps together and maps run-level context into them.

<details>
<summary><code>myapp/pipelines/greet.schema.json</code> — the DAG (one node, CPU)</summary>

```json
{
  "name": "greet",
  "display_name": "greet · hello world",
  "requires_gpu": false,
  "inputs": {
    "type": "object",
    "properties": {
      "name": { "type": "string", "default": "world", "maxLength": 64 }
    }
  },
  "outputs": {
    "primary_file": { "type": "file", "mime": "text/plain" },
    "summary_keys": ["greeted"]
  },
  "nodes": [
    {
      "id": "greet",
      "node": "greet",
      "depends_on": [],
      "inputs":  [{ "name": "name", "from": "pipeline.name" }],
      "outputs": [{ "name": "hello.txt", "kind": "text" }],
      "gpu": false
    }
  ]
}
```

`"node": "greet"` resolves to the `myapp.nodes.greet` module via the `node_module_package` prefix set in step 3. `"gpu": false` routes the node-job to the `cpu` queue (set `true` for `gpu`).

</details>

<details>
<summary><code>myapp/definitions/greet.json</code> — the workflow (one pipeline step)</summary>

```json
{
  "name": "greet",
  "display_name": "greet — hello world",
  "mode": "node",
  "steps": [
    {
      "id": "greet",
      "kind": "pipeline",
      "pipeline": "greet",
      "inputs": { "name": { "$from": "parcel.label" } }
    }
  ]
}
```

The `$from` ref pulls `name` out of the run's `context` at execute time — late resolution: workers re-resolve refs when they pick up the job, not when it's enqueued.

</details>

## 3. Wire the host (`configure` + seams) and bootstrap

One startup module does the wiring. `configure()` only mutates the keys you pass; `db.bootstrap()` applies the engine's migration chain idempotently. The full hook reference is [`docs/configuration.md`](docs/configuration.md).

```python
# myapp/engine.py
import json
from pathlib import Path

import queue_workflows
from queue_workflows import db

DEFS = Path(__file__).parent / "definitions"
PIPES = Path(__file__).parent / "pipelines"


def _load_workflow(name: str) -> dict:
    return json.loads((DEFS / f"{name}.json").read_text())


def _pipeline_schema(name: str) -> dict:
    return json.loads((PIPES / f"{name}.schema.json").read_text())


def init() -> None:
    # 1. configure the engine (every key is optional)
    queue_workflows.configure(
        db_backend="pg",                    # v1.0.0: default is now "sqlite" — opt in for Postgres
        db_url_env="MYAPP_DB_URL",          # env var holding the DSN
        node_module_package="myapp.nodes",  # "greet" -> myapp.nodes.greet
        container_prefix="myapp-",          # cgroup attribution for hw_metrics
    )

    # 2. tell the dispatcher where the DAG definitions live
    queue_workflows.set_workflow_provider(_load_workflow, _pipeline_schema)

    # 3. apply the engine's migration chain (idempotent)
    db.bootstrap()
```

> 💡 The DSN lives in the `MYAPP_DB_URL` **environment variable**, not in code — `configure(db_url_env=...)` only names the variable to read. Keep secrets in your secrets store.

## 4. Launch the processes (console scripts)

Independent processes run against the one database. You run a **claim worker per queue per host**; the **orchestrator** drives DAG fan-out + lease reclaim; the **scheduler** fires periodic ingest jobs (skip it if you have none). Each process calls `myapp.engine.init()` at startup, then hands off. See [`docs/deployment.md`](docs/deployment.md).

```bash
queue-orchestrator                 # DAG dispatch loop + dead-worker lease reclaim (bootstraps migrations)
queue-claim-worker --queue cpu     # claims & runs cpu node-jobs (our greet node)
queue-claim-worker --queue gpu     # add one per GPU host for gpu-routed nodes
queue-scheduler                    # optional — only if you registered ingest tasks
```

Or call the entry points directly from your own bootstrap:

```python
from myapp.engine import init
import queue_workflows

init()
queue_workflows.claim_worker.main(["--queue", "cpu"])
```

## 5. Kick off a run

Enqueuing work is inserting a `workflow_runs` row and expanding its DAG — `start_run()` enqueues every node whose `depends_on` is empty, the `NOTIFY` rides the transaction, and a listening worker wakes immediately.

```python
import uuid
from myapp.engine import init
from queue_workflows import run_store, dispatcher

init()

run_id = str(uuid.uuid4())
run_store.insert_run(
    run_id=run_id,
    workflow_name="greet",
    context={"parcel": {"label": "Ada"}},   # feeds the $from ref → name="Ada"
)
dispatcher.start_run(run_id)                 # fan out the initial ready nodes
```

The `cpu` worker from step 4 claims the `greet` node, runs it, writes `hello.txt`, and flips the run to `completed`. That's the whole loop — **inserting a row is enqueuing the work.**
