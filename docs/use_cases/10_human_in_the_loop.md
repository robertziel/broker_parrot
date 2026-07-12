# Use case 10 — pause a workflow for human input

**Scenario.** Mid-pipeline, a step needs a human decision — approve generated
copy, pick one of N rendered variants — and the wait may be minutes or days.
Holding a worker for that is unacceptable; the run must park durably and
resume the moment input arrives.

## Park: `awaiting_input`

A node signals it needs input; the engine marks the node **and its run**
`awaiting_input` (with an `input_spec` describing what's being asked), writes
the dispatch event through the same durable outbox as any terminal status, and
**releases the worker** — nothing is blocked, nothing is polled by the node.

The park survives anything: it's plain rows. Orchestrator restarts, worker
reboots, a week of waiting — the run just sits there, visible in any snapshot
as `awaiting_input`.

## Resume: one row in `workflow_input_submissions`

Your UI (or CLI, or any DB writer) answers by inserting a submission row keyed
to the run. The orchestrator's `InputListener` polls `workflow_input_submissions`,
claims the submission, merges the answer into the run context, and re-enqueues
the parked node — which now sees the human's answer through the normal
late-resolving `$from` refs.

```
node: needs input ──> run + node = awaiting_input      (outbox event, worker freed)
human: INSERT INTO workflow_input_submissions (…)      (any DB writer is a UI)
orchestrator: InputListener claims it ──> node requeued ──> pipeline continues
```

Claiming a submission uses the same guarded-update idempotency as everything
else, so double-submits and racing listeners are safe; the listener's reclaim
is covered by the engine's invariant tests.

This is the pattern for **any** external-event wait, not just humans: a
payment webhook, an upstream batch landing, a manual QA gate — anything that
can write one row can resume a workflow.
