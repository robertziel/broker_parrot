# Use cases

Concrete operational scenarios and how the engine handles each one, end to end.
Every file describes **actual engine behavior** — the mechanisms named
(tables, modules, exit codes, timings) are the ones in the code, not aspirations.

| # | Use case | One line |
|---|----------|----------|
| 1 | [A box powers off mid-job](01_box_power_loss.md) | Lease lapses → job requeued **at the front of the queue** → healthy peer claims it next; if the box comes back, its zombie worker gets a kill signal. |
| 2 | [A box boots (or reboots)](02_box_boot_and_rejoin.md) | Schema gate → park check → heartbeat → claim; a rejoining zombie self-kills instead of double-running. |
| 3 | [Operator stops / parks a worker](03_operator_stop_and_park.md) | One SQL row hard-stops a `(host, queue)` worker, requeues its job fault-free, frees VRAM; the worker parks on boot while OFF. |
| 4 | [A wedged GPU job recovers itself](04_wedged_gpu_recovery.md) | Budget / stall / GPU-health watchdogs re-queue-and-retry with distinct exit codes; a hardware hang is caught out-of-process. |
| 5 | [Run a multi-step DAG pipeline](05_dag_pipeline.md) | Declarative fan-out with a durable outbox — no lost edges, no double fan-out, retryable dispatch. |
| 6 | [Periodic background work](06_periodic_ingest.md) | The DB-native scheduler + your own queue names and per-job args, enqueued atomically with your own writes. |
| 7 | [Many projects, one broker](07_multi_project_broker.md) | Pool N apps onto one shared DB with the `project` tag; exact-match claiming keeps them isolated. |
| 8 | [Keep GPU models warm across jobs](08_warm_model_affinity.md) | The warm `ModelCache` + heartbeat-advertised `current_model` route same-model jobs to the already-warm box. |
| 9 | [Jump an urgent job to the front](09_urgent_job_run_next.md) | `prioritize_node_job(job_id)` — `is_priority` sorts first in the claim `ORDER BY`. |
| 10 | [Pause a workflow for human input](10_human_in_the_loop.md) | A node parks its run as `awaiting_input`; a submission resumes it — durable across restarts. |

**Reading order.** 1–4 are the resilience story (what the engine does when
machines fail or operators intervene); 5–10 are the feature story (what you
build on it). If you read only one, read
[01 — a box powers off mid-job](01_box_power_loss.md): it exercises the whole
liveness model.
