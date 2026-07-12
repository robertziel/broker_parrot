# Use case 9 — jump an urgent job to the front of the queue

**Scenario.** A queue has depth — tens of queued node jobs — and something
urgent lands: a customer-facing render, a fix-verification job. It must be
**the next thing claimed**, without re-prioritizing everything else.

## `prioritize_node_job` — the binary "run next" flag

```python
node_queue.prioritize_node_job(job_id)   # UPDATE … SET is_priority = TRUE WHERE status='queued'
```

`is_priority DESC` is the **first** term of the claim `ORDER BY` — ahead of
the integer `priority` band and ahead of GPU warm-model affinity. The next
worker asking for that queue claims the flagged job, even at the cost of a
model swap.

Properties worth knowing:

- **No-op unless queued.** A running/terminal job can't be re-ordered; the
  call returns the updated row or `None`.
- **Binary, not a ladder.** Two flagged jobs tie and fall through to the band
  and the normal tiebreaks — the flag is "front of the line", not a second
  priority system.
- **Shared with the resilience path.** Machine-loss requeues
  ([use case 1](01_box_power_loss.md)) and operator hard-stops
  ([03](03_operator_stop_and_park.md)) set the same flag — "urgent" and
  "already burned wall-clock" are deliberately the same lane.
- **Bands still exist.** For coarse steering use the integer `priority`
  column (lower = sooner) at enqueue time; the flag is for *after* the job is
  already queued and the world changed.
