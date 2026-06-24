# Running the full DAG engine on Redis / MongoDB — feasibility & staged design

> **Stage-1 deliverable** (a feasibility map, not a build). It answers the
> decisive question — *can the relational DAG engine run on the flat-queue
> StorageBackend SPI, and by what design?* — so the multi-week build can be
> funded (or not) with eyes open. Produced by a parallel subsystem analysis
> (dispatcher / node_queue / node_executor / run_store / refs / node_pool / the
> SPI baseline); see `worklog/redis-mongo-engine-rehome.md`.

## TL;DR verdict — **feasible but LARGE, and the framing is a category error**

"Re-home onto the flat-queue SPI" understates the work by an order of magnitude.
The SPI (`backends/base.py`) is **one flat queue**: a `Job` has no
`run_id`/`node_id`/DAG, the claim is a single precomputed priority score, and
`docs/storage_backends.md` itself says selecting redis/mongo does **not** re-home
the engine. The DAG engine is the opposite: a **multi-entity relational store** —
`workflow_runs ⋈ workflow_node_jobs ⋈ workflow_dispatch_events ⋈
worker_heartbeats ⋈ worker_controls`, with **cross-entity atomic transitions
everywhere** (the claim guards on parent-run status; reclaim branches
CASE-on-run; cancel-orphaned anti-joins run status; stuck-detection is
NOT-EXISTS; unassignable does fleet aggregation). You cannot *extend* the flat
SPI to host this — you must reintroduce a secondary index on `run_id`, multi-key
atomic transactions, GROUP-BY aggregation, anti-joins, and a dynamic composite
claim. **That is growing a SECOND, run-aware DAG-storage SPI alongside the flat
one** — a new store, not a re-home. (Why a *second* SPI and not just adding
`run_id` to the flat one: the flat `StorageBackend` is a **stable contract used
standalone by the GPU pool** (`gpu_pool_backend`), so its `Job` can't grow DAG
fields without breaking that consumer — the DAG store must be a distinct interface.)

Per backend:

| Backend | Verdict | Why |
|---|---|---|
| **MongoDB** | **Feasible but large** | A relational-lite store on documents (`runs` + `node_jobs` collections, secondary-indexed by `run_id`). The dispatcher's DAG-walk genuinely maps clean to `find({run_id})`. **Replica set is the FLOOR, not optional** — the atomic outbox, the change-stream wake, *and* every cross-entity sweep are multi-doc transactions, which need a RS. A single-node RS clears all of it. |
| **Redis** | **Likely not worth it** | Pure KV: re-home = reimplementing a relational engine in **Lua on a single non-clustered instance**. Cross-key atomicity is Lua-on-one-instance only → Redis **Cluster is out** (keys must share a hash slot) → a **SPOF with no horizontal scale**, which *defeats the multi-host fleet that is the engine's whole reason to exist*. A rewrite, not a re-home. |

**Confidence:** high on the shape (grounded in the actual SQL — the claim's
run-status guard, the CASE-on-run reclaim, the NOT-EXISTS stuck detector, the
multi-doc outbox contract). The 80/20: the dispatcher (the *clean* part) is ~44
operations that map to a document read; the *expensive* 20% — the cross-entity
atomic claim/lease/reclaim/sweep layer — is 12 hard-blocker ops with no flat-SPI
analog.

## First, the load-bearing question: what use case justifies it?

**The engine ALREADY runs the multi-host fleet on PostgreSQL — and now on SQLite
too** (both relational; both do JOIN / GROUP-BY / anti-join / atomic multi-row
without a rewrite). So before funding a multi-week Mongo port, name the concrete
need it serves that pg/sqlite don't — this is the **load-bearing funding
decision**, and the analysis could not find it stated. Candidates that *might*
justify it:

- an **edge / no-Postgres** deployment that already runs Mongo;
- **"one system"** consolidation (an app already all-in on Mongo/Redis for its
  primary store, wanting the queue there too);
- workflow **definitions already living in Mongo**.

If none of these is real, this is **exploratory R&D with uncertain payoff** — fund
it as such, or don't. **The flat ingest / GPU-pool queue on redis/mongo already
works and needs none of this.** Recommend the operator answer "*who needs the DAG
engine on Mongo, and why not pg?*" **before** the thin slice, not after.

## The hard blockers (12 cross-entity / relational ops with no flat-SPI analog)

1. **Atomic outbox** (`node_executor`): terminal job-status **and** the
   `workflow_dispatch_events` row in ONE transaction (both-or-neither). → Mongo
   multi-doc txn (**RS-only**); Redis Lua spanning run+job keys (single instance).
2. **`reclaim_expired_leases`**: `UPDATE … SET status = CASE WHEN r.status='running'
   THEN 'queued' ELSE 'cancelled' END FROM workflow_runs r` — reads parent-run
   status and branches the job write atomically.
3. **`cancel_orphaned_queued_jobs`** / **`cancel_queued_for_run`**: anti-join on
   run status (`r.status IN ('cancelled','failed')`).
4. **`list_stuck_node_run_ids`**: `NOT EXISTS` (a run with no live jobs) — a
   cross-entity anti-join; needs a `has_live_jobs(run_id)` primitive.
5. **`FOR UPDATE SKIP LOCKED`** (the claim, the outbox drain, the
   `InputListener`): pessimistic concurrent-claim with no KV/document equivalent —
   two orchestrators/listeners would double-process without a bespoke atomic claim.
6. **Dynamic composite claim** — *the real 80/20 gate, not the dispatcher.*
   Warm-model affinity (`required_model IS NOT DISTINCT FROM current_model`, where
   `current_model` varies **per worker per claim**) + `host_priority` direction +
   the run-not-cancelled guard — cannot be reduced to one precomputed sort score.
   Mongo's `find_one_and_update` **cannot conditional-sort by a per-worker
   value**, so the doc's "pre-flag affinity pass then sort" really means a
   **multi-doc transaction (find candidates → claim)** on the **hottest path**
   (every claim) — adding latency + complexity, and **exactly-once under
   concurrency is NOT guaranteed by `find_one_and_update` alone** here. This is
   the single operation most likely to make the port "feel like rebuilding SQL";
   it must be probed FIRST (see the thin slice).
7. **Fleet aggregations**: `snapshot`/`ingest_snapshot`/`fleet_snapshot` (GROUP BY
   status), `flag_stale_workers_holding_running_jobs` (heartbeats ⋈ jobs + COUNT),
   `flag_unassignable_gpu_jobs` (fleet VRAM aggregation + array membership with
   NULL-matches-NULL).
8. **Two opposing access patterns one layout can't serve**: the DAG-walk wants
   per-run enumeration (one doc/hash per run) while the claim wants a cross-run
   fleet-wide priority scan (next gpu job across ALL runs) → forces TWO
   synchronized structures = hand-rolled relational secondary indexing.

The dispatcher's ~44 "maps-clean" ops (deps-satisfied walk, `skip_if`, `$from`
context merge, cascading skips) are genuinely pure-Python over a list of a run's
jobs — **but every one silently depends on run-scoped enumeration
(`list_jobs(run_id)`) the flat SPI does not provide.**

## What a real port needs (a new run-aware DAG-store SPI)

- A **Job/run model**: `run_id`, `node_id`, `upstream_node_ids`, `context_delta`
  + a **first-class secondary index on `run_id`** (without it nothing run-scoped
  is queryable).
- **`list_jobs(run_id, status?)`** — the single method that unblocks the entire
  dispatcher.
- A **run-store facet** the flat SPI entirely lacks: `insert_run`/`get_run`/
  `update_run`/`delete_run(+cascade)`/`claim_next_queued`/`has_live_jobs`/
  `reenqueue_running_for_resume`.
- **Cross-entity atomic transitions** that read parent-run status and branch the
  job write in one unit (reclaim CASE-on-run, cancel-orphaned).
- The **dynamic composite claim** (affinity + host_priority + run-guard).
- The **fleet aggregations** (snapshot / stale-worker / unassignable).

## Staged plan (Mongo-first; gate honestly at Stage 0)

0. **DESIGN (make-or-break):** commit a per-backend layout on paper. Mongo = a
   `runs` collection + a `node_jobs` collection indexed by `run_id` (relational-
   lite). Redis = per-run hash + per-queue claim ZSET + run-index sets. **This is
   where you decide "extend an SPI" vs "rebuild a relational store" — gate here.**
1. Extend the Job/run model + add `list_jobs(run_id)` on **Mongo**.
2. **Port the dispatcher** (pure-Python walk) onto `list_jobs` — prove `$from`
   cross-node resolution, deps-satisfied, `skip_if` on a document model. (No new
   atomicity; the strongest maps-clean claim.)
3. Build the **atomic outbox** on a **single-node RS**; extend the contract suite
   with run/node shape (both-or-neither + no-double-on-redelivery).
4. Add the **run-store facet** (`claim_next_queued`, `has_live_jobs`, resume).
5. The **cross-entity atomic sweeps** (reclaim CASE-on-run, cancel-orphaned) via
   Mongo session txn.
6. The **dynamic composite claim** (affinity + host_priority).
7. **Fleet aggregations** + capacity sweeps.
8. **ONLY after Mongo validates the thesis end-to-end, evaluate Redis** — expect
   to reimplement stages 3/5/6 in single-instance Lua and to accept a SPOF + no
   Cluster. **If Mongo (the easy target) felt like rebuilding SQL, KILL Redis
   here** rather than fund a Lua relational engine.

## The thin first slice (the decisive probe — ~2–4 focused days, behind a human gate)

> **Corrected after audit (do NOT probe the dispatcher first).** The dispatcher
> walk is the *trivially feasible* 20% — a per-run document read + pure-Python
> logic; proving it validates almost nothing and risks a false "go" that commits
> funding before the hard 80% surfaces. **Probe the HARD blockers first.**

**Mongo-only, on a single-node replica set, probe the cross-entity ATOMIC layer
under concurrency — in this order:**

1. **The dynamic composite claim (blocker #6) under contention** — `N` concurrent
   "workers" claiming GPU node-jobs that carry `required_model`, each worker with
   a *different* `current_model`. Assert **exactly-once** (no job claimed twice,
   none lost) AND that warm-model affinity + `host_priority` actually order the
   picks. This is the operation `find_one_and_update` can't conditional-sort —
   the make-or-break. If this needs an ugly multi-doc-txn candidates-then-claim
   with races, that's the KILL signal.
2. **Lease-reclaim CASE-on-run (blocker #2) + the atomic outbox (blocker #1)** —
   force a lapsed lease whose parent run is `running` (→ re-queue) vs `cancelled`
   (→ cancel) in one atomic unit, and finalize a node writing terminal-status +
   dispatch-event both-or-neither; assert no-double-on-redelivery.
3. **THEN** the easy part — one linear 2-node DAG (A → B, B's input a `$from` ref
   to A) reusing `dispatcher.py` **unchanged** over the new store, to confirm the
   walk rides the same layout.

**Out of slice:** capacity/fleet aggregations, watchdog retries, Redis, standalone
(non-RS) Mongo.

**Why this order:** it spends the probe on the **single highest-uncertainty
question** — *can Mongo do the per-worker dynamic claim + the cross-entity
CASE-on-run sweep atomically and exactly-once, or does it degrade into painful
multi-doc-txn races?* — not on the dispatcher walk (a foregone "yes"). **If steps
1–2 are clean, the document-model thesis is validated and the rest is "large but
known." If they fight you, KILL before funding the expensive 80%.**

## MongoDB replica-set requirement (a hard floor)

A replica set is the **floor**, not polish, for a *correct* Mongo re-home. The
atomic outbox (`complete_with_event`/`fail_with_event`) is a multi-doc
transaction and the wake is a change stream — both replica-set-only; a standalone
`mongod` fails loudly on connect. Every cross-entity sweep that atomically reads
parent-run status and branches the job write (reclaim CASE-on-run,
cancel-orphaned) is **also** RS-gated. A single-node RS (`rs.initiate()` on one
`mongod`) fully satisfies all of it and is fine for dev / small deploys — but
"standalone Mongo" as a deployment target is a hard no. **This environment
currently has a standalone `mongod` (no RS) → the Mongo path is untestable here
until a single-node RS is stood up.**

## Recommendation (the human gate)

This is a **multi-week, architecture-level build** — outside a single autonomous
unit, which is why it stops here at the design. Recommended order:

1. **MongoDB, behind the gate:** if you want it, fund the **thin first slice**
   first (2–4 days, single-node RS) — it cheaply validates or kills the document-
   model thesis before the expensive 80%.
2. **Redis: reconsider.** The honest finding is that a Redis full-engine re-home
   is a *rewrite* (a relational engine in Lua) on a **single non-clustered
   instance** — a SPOF with no horizontal scale, which contradicts the multi-host
   fleet the engine exists to serve. Recommend **KILL the Redis full-engine
   re-home** unless a specific single-box, low-scale use case justifies it; keep
   Redis for the **flat ingest / GPU-pool queue** (where it already works well).

   *(Rebuttal to the obvious counter-argument — "Redis hash-tags `{run_id}`
   co-locate a run's keys on one slot, so Cluster IS safe":* true for per-RUN
   atomicity, but (a) the cross-RUN fleet-wide claim scan still spans slots, and
   (b) the flat SPI `Job` has **no `run_id`** — using hash-tags means redesigning
   the key schema into a run-aware store anyway. Hash-tags don't rescue the flat
   design; you've built the new store either way. The KILL stands.*)*

The flat-queue SPI on redis/mongo (ingest + GPU pool) is unaffected and remains
the supported, tested multi-backend story today.
