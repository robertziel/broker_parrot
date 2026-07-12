# Use case 8 — keep GPU models warm across jobs

**Scenario.** Your GPU jobs use a handful of large models. Loading one takes
minutes; the inference takes seconds-to-minutes. Naive scheduling reloads
models constantly — the fleet spends its life in `load_state_dict`.

## The warm cache + affinity routing

- **One process, one warm model.** A GPU claim worker owns a process-wide
  `ModelCache`: consecutive same-model jobs reuse the resident model; only a
  real switch drops and reloads. An idle TTL unloads after a quiet period.
- **The fleet knows who's warm.** Each GPU worker publishes `current_model`
  in its heartbeat. The claim `ORDER BY` adds an affinity tiebreak —
  `required_model IS NOT DISTINCT FROM current_model` sorts first — so a job
  needing model M gravitates to the box already holding M.
- **Capacity-aware assignment** (migration `0015`) fits models to VRAM and
  flags jobs whose model cannot fit anywhere (`unassignable`) instead of
  letting them thrash the queue.

## Ordering, precisely

```
ORDER BY is_priority DESC,          -- machine-loss requeues + operator "run next"
         priority ASC,              -- the band
         affinity (warm model), host_priority, created_at …
```

Affinity is a **tiebreak, not a law**: a front-of-queue job (see
[use case 1](01_box_power_loss.md) / [09](09_urgent_job_run_next.md)) may
force a model swap on a healthy box — deliberately, because the box with the
warm copy may be the one that just died.

## LLM sidecars

GPU nodes often call a co-tenant ollama/vLLM server. Which *type* a box runs
is operator-set state (`worker_controls`, migration `0013`); each host
advertises what it *can* run in its heartbeat (`0014`); nodes never branch on
server type — the engine resolves the endpoint per dispatch and passes it as
`run(llm_server=...)`. An idle supervisor can stop a vLLM sidecar to reclaim
VRAM and restart it on demand.

Full treatment: [`../gpu_and_llm.md`](../gpu_and_llm.md).
