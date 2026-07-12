# Use case 5 — run a multi-step DAG pipeline

**Scenario.** A run needs several steps with dependencies — fetch, then two
transforms in parallel, then a merge — with some steps on `cpu` and some on
`gpu`, and the whole thing must survive any process dying between steps.

## Declare the DAG, insert a run

The pipeline schema owns the `nodes` list (`id`, `node`, `depends_on`, `gpu`,
`inputs`, `outputs`); the workflow references it. Enqueuing is one insert +
one call:

```python
run_store.insert_run(run_id=rid, workflow_name="my_pipeline", context={...})
dispatcher.start_run(rid)     # enqueues every node whose depends_on is empty
```

## Fan-out is durable: the outbox pattern

The worker→dispatcher handoff is where naive queue systems lose edges. Here a
worker finalizing a node writes the terminal status **and** a
`workflow_dispatch_events` row in **one transaction** — then the orchestrator
drains that outbox and enqueues every downstream node whose deps are all
`completed`/`skipped`:

- crash *before* the txn: the node re-runs (lease reclaim);
- crash *after* the txn: the event is durable; fan-out happens on the next
  drain tick;
- a failing dispatch callback is retried next tick and poison-flagged after
  repeated failures — never silently dropped.

There is **no** synchronous coupling: a worker never imports or calls the
dispatcher.

## Conditionals, refs, priorities

- **`skip_if`** — a node can declare a condition; the dispatcher inserts a
  `skipped` marker instead of enqueuing, and downstream deps treat `skipped`
  as satisfied.
- **`$from` refs** resolve **late** — when a worker picks the job up, against
  the run's *current* context, so upstream `context_delta` keys written after
  enqueue are visible.
- **Queue routing** — `"gpu": true` routes a node to the `gpu` queue; every
  node is claimed independently, so cpu and gpu stages interleave across the
  fleet.
- Any node in the DAG inherits every resilience property of
  [use cases 1–4](01_box_power_loss.md): a lost box re-runs *only that node*,
  at the front of the queue, and the run continues.

See the end-to-end worked example in the [README](../../README.md#-example-implementation)
and the mental model in [`../architecture.md`](../architecture.md).
