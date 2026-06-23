# `gpu_pool` — the shared GPU fleet

`gpu_pool` lets **several apps share one pool of GPU machines** without any box touching another app's
database. It is **additive and opt-in**: an app keeps its own Postgres for run/DAG state and only
reaches for the pool when it wants cross-machine GPU sharing.

## Why a separate primitive (not a DAG node-job)

A DAG node-job (`workflow_node_jobs`) is bound to its app's Postgres — its run row, dispatch outbox,
leases, node-events — and to a local filesystem `out_dir`. A pooled worker on a *different* box can't
reach any of that. So the pool trades the DB-bound node-job for a **self-contained `PoolTask`**:

```
PoolTask = { model, handler, inputs, output_dir, params }
```

`inputs` / `output_dir` are **references into shared NFS**, so a pooled worker reads inputs and writes
outputs on the shared filesystem and **never opens an app database**. The op *code* lives on each GPU
box (a registered handler); the *data* lives on NFS.

## The shared store

The pool is a **`StorageBackend`** (the same SPI as the pluggable DB backends — see
[`storage_backends.md`](storage_backends.md)), addressed **independently** of any app's `db_backend`:

```python
queue_workflows.configure(
    gpu_pool_backend="redis",                       # default
    gpu_pool_url_env="QUEUE_WORKFLOWS_GPU_POOL_URL", # env holds the DSN
    gpu_pool_namespace="gpu_pool",                   # every app + box sharing a fleet uses the SAME value
)
```

An app keeps `db_backend="pg"` for its own DAG while pooled GPU workers across apps claim from this one
shared store. The namespace isolates one fleet's tasks from another on a shared server.

## Capability routing — by queue name

A `PoolTask` is enqueued onto a **capability queue**: a model id, or a box-class name like `gpu:box-a` /
`gpu:box-b`. A pooled worker serves an **ordered set** of queues, and that order *is* the routing
policy:

- **its warm-model queue first** ⇒ affinity (consecutive same-model tasks don't reload the model);
- **then box-class queues** ⇒ keep work on machines that can actually run it.

> Routing is coarser than the DAG GPU claim on purpose: the pool does **not** do the within-queue
> warm-model sort or the VRAM capacity-fit gate the DAG path has — the operator hand-partitions by
> choosing queue names. (If a fleet needs finer routing, that belongs in `gpu_pool`, not a central
> scheduler.)

## API

**Submitter (any app — needs no handlers registered):**

```python
from queue_workflows import gpu_pool

task_id = gpu_pool.submit_pool_task(
    queue="gpu:box-a", handler="upscale", model="sr",
    inputs={"src": "/nfs/in/img.png"}, output_dir="/nfs/out/job123",
    params={"scale": 4}, priority=0,
)
result = gpu_pool.await_pool_result(task_id, timeout_s=600)   # raises PoolTaskFailed / TimeoutError
```

**Worker (on each GPU box — registers the op code):**

```python
def upscale(*, inputs, output_dir, params):
    ...                      # read inputs from NFS, write to output_dir
    return {"out": f"{output_dir}/result.png"}

queue_workflows.register_pool_handler("upscale", upscale)

# claim → run handler → atomic-outbox terminal; returns "completed" / "failed" / None (queues empty)
gpu_pool.run_pool_worker_once(queues=["sr", "gpu:box-a"], worker="box-a-1")
```

The orchestrator (or a sweep) calls `gpu_pool.reclaim_expired_pool_leases()` to re-queue tasks whose
worker died mid-run — the same lease/reclaim guarantee as the DAG path.

## Guarantees

- **Exactly-once terminal via an atomic outbox** — the worker writes the terminal status + the result
  event together, so a crash after the work is done doesn't drop the result.
- **Lease + reclaim** — a dead worker's task is re-queued, not lost.
- **Namespace isolation** — two fleets on one shared server can't see each other's tasks.
- **No app-DB coupling** — pooled workers touch only the shared pool store + NFS.

Submit-only apps need no handlers; a box that only runs work needs only its handlers + the worker loop.
