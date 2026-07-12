# Use case 7 — many projects, one broker database

**Scenario.** You run several independent apps on one physical fleet. Each has
its own queue workload; you want **one** shared queue database (one thing to
operate, one place to observe the whole fleet) with hard isolation between the
projects' work.

## The `project` tag (migration 0017)

Every queue row carries a `project` TEXT tag, and **claiming is exact-match**
— the filter lives in the same `FOR UPDATE SKIP LOCKED` statement, so a worker
configured for one project can never claim another's row:

```python
# each app, same BROKER_DSN, its own tag
queue_workflows.configure(project="alpha", db_backend="pg", db_url_env="BROKER_DSN")
```

`QUEUE_WORKFLOWS_PROJECT=<name>` sets the same knob for console scripts.
Default `""` keeps single-tenant deploys byte-identical. Heartbeats key by
`(host_label, queue, project)`, and the operator control plane is
project-keyed too (migration `0019`) — parking alpha's gpu worker on a box
leaves beta's untouched.

## Adoption gotcha — drain first, or backfill

Migration `0017` backfills existing rows to `project=''`. Because claiming is
exact-match, the instant a running deploy switches to
`configure(project="alpha")` it stops seeing those `''` rows. Adopt a project
name on a **drained** queue, or backfill in the same maintenance window
(`UPDATE workflow_node_jobs SET project='alpha' WHERE project='' …`).

## Scaling up: the pull→grant broker service

Tag-pooling keeps each client autonomous (it claims for itself). The
**broker service** goes further: workers *ask permission*
(`POST /api/ask`) and the broker grants work, arbitrates shared CPU/GPU
capacity across projects, and can revoke a grant at will — with a bundled
operator panel (`queue-broker-web`):

![queue-broker-web panel](../images/broker-web-panel.png)

The full model, API, and cutover checklist:
[`../broker.md`](../broker.md) and [`../deployment.md`](../deployment.md).
