# Per-node-job box placement — `avoid_box` / `force_box`

*Migration 0020. Additive + opt-in — every existing node-job is unconstrained and
byte-identical to before.*

A queued **node-job** can pin or exclude the **physical box(es)** that may execute it.
Two optional lists, both NULL by default:

| Flag | Meaning | Example |
|---|---|---|
| `avoid_box` | the job **must not** run on any listed box | `avoid_box: [box-a]` → only `box-b`/`box-c` run it |
| `force_box` | the job may run **only** on a listed box (hard pin) | `force_box: [box-c]` → nowhere but `box-c` |

A box is eligible iff it is **not** in `avoid_box` **and** (when `force_box` is set) **is**
in `force_box`. Set both to intersect them (`force_box: [box-b, box-c]` +
`avoid_box: [box-c]` → only `box-b`). An empty list means *no constraint* — it is
normalised to NULL, so a stray `force_box: []` can never strand a job.

## What "box" means

The name is the **physical box identity** — the value a worker resolves via
`gpu_model_lease.default_box_id()`: `QUEUE_WORKFLOWS_GPU_BOX_ID` → `config.gpu_box_id` →
the machine hostname. It is the **same identity the one-model-per-box lease keys on**,
and deliberately **not** the per-project `host_label` (`box-b-gpu`), so one constraint
covers every project/lane on that machine.

> **Requirement:** for placement to bind, workers on the box must agree on the name — set
> `QUEUE_WORKFLOWS_GPU_BOX_ID` (the fleet already does this for the box lease:
> `box-a` / `box-b` / `box-c`). A worker that resolves **no** box name claims **only
> unconstrained** jobs — the safe default (it can't verify a constraint it can't name).

## How it's enforced

It's a **claim-time filter**, not a post-claim reject: the box-placement predicate is
ANDed into the `SELECT … FOR UPDATE SKIP LOCKED` claim (both the cpu and gpu lanes), so an
ineligible worker never grabs the row — an eligible peer does. Nothing re-queues, nothing
spills. Dialect-portable (Postgres `= ANY(col)`, SQLite `json_each`).

## Setting it

**In a workflow / pipeline node spec** — the dispatcher threads the two keys straight
through when it fans a run out into node-jobs:

```json
{
  "id": "render_seg0",
  "gpu": true,
  "model": "sdxl",
  "avoid_box": ["box-a"]
}
```

**Directly, when a host enqueues its own node-job:**

```python
from queue_workflows import node_queue

node_queue.enqueue_node_job(
    run_id=run_id, node_id="score", node_module="x", queue="gpu",
    force_box=["box-c"],           # only box-c may run it
    # avoid_box=["box-a"],        # …and/or keep it off box-a
)
```

Both default to `None` (unconstrained). Reading a job back
(`node_queue.get_node_job(id)`) returns `avoid_box` / `force_box` as `list[str]` or
`None`.

## Interplay with the other GPU gates

Placement is **orthogonal** and composes with them — all are ANDed at claim time:

- **one-model-per-box lease** — placement decides *which boxes may claim*; the lease still
  decides *which model a box holds*. A job forced onto `box-c` still spills there if
  `box-c` currently holds a different model.
- **capability / VRAM gate** — a `force_box` that can't serve the job's `required_model`
  means the job simply never runs (no eligible+capable box). That's the caller's
  responsibility; the engine won't silently relax the pin.
- **fill-before-spill, warm-model affinity, box-slot arbiter** — unchanged; they operate
  within the set of boxes placement already allows.

## Operational notes

- Purely additive: no default behaviour changes, no config flag to flip.
- The claim reads the columns every claim, so a value edit on a **still-queued** row takes
  effect on the next claim with no restart.
- A contradictory constraint (`force_box` ∩ `avoid_box` = ∅, or a `force_box` naming no
  live/capable box) leaves the job `queued` — visible in the queue gauge, not lost.
