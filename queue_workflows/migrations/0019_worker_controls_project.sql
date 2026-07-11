-- queue_workflows 0019 — project (tenant) identity on the operator control plane.
--
-- WHY. 0017 pooled the QUEUE onto one shared broker and re-keyed
-- ``worker_heartbeats`` to ``(host_label, queue, project)`` — but it left
-- ``worker_controls`` on the old 2-col PK ``(host_label, queue)``. On a shared
-- broker ``host_label`` is no longer globally unique: two projects can run a
-- worker on the SAME machine + queue (live example: ``host-b`` runs both
-- ai_leads' and project-b's gpu worker). With a 2-col PK those two workers share
-- ONE control row, so:
--
--   * an operator ON/OFF for project A's worker also hard-stops project B's
--     worker on that machine (wrong tenant parked), and
--   * an LLM-config write for A (``llm_server_type`` / ``llm_parallelism`` /
--     ``vllm_idle_ttl_s``, migration 0013) silently RECONFIGURES B's LLM server.
--
-- The control plane must be keyed exactly like the heartbeat it controls. This
-- migration is the control-plane twin of 0017's heartbeat re-key.
--
-- DESIGN — mirrors 0017 exactly. ``DEFAULT ''`` keeps a single-tenant deploy
-- byte-compatible: every existing row backfills to the sentinel and the
-- 3-col lookup ``(host, queue, '')`` matches them, so today's behaviour is
-- unchanged with zero host wiring.
--
-- Additive + idempotent (``IF NOT EXISTS`` / drop-then-add the PK) so re-running
-- on an already-migrated DB is a safe no-op.

-- ── tenant tag on the control row ──────────────────────────────────────────
ALTER TABLE worker_controls
    ADD COLUMN IF NOT EXISTS project TEXT NOT NULL DEFAULT '';

-- ── control-row identity now includes project ──────────────────────────────
-- BREAKING for raw-SQL writers: the 2-col unique constraint is gone. Any
-- consumer upserting with its own ``INSERT … ON CONFLICT (host_label, queue)``
-- must move to ``worker_control.set_worker_control`` / ``set_llm_config``
-- (3-col ON CONFLICT) or it errors with "no unique or exclusion constraint
-- matching the ON CONFLICT specification".
ALTER TABLE worker_controls DROP CONSTRAINT IF EXISTS worker_controls_pkey;
ALTER TABLE worker_controls
    ADD CONSTRAINT worker_controls_pkey PRIMARY KEY (host_label, queue, project);

-- ── NOTIFY payloads are DELIBERATELY unchanged ─────────────────────────────
-- The 0012 ``worker_control`` and 0013 ``worker_llm_config_changed`` payloads
-- stay ``host:queue`` / ``host|queue``. Adding a tenant segment would break the
-- pinned payload contract (and any external listener that unpacks exactly two
-- fields) while buying the engine nothing: both the WorkerControlWatcher and the
-- LLM backend factory ignore the payload — they treat the NOTIFY as a bare wake
-- and then RE-READ their own project's row. The cost on a shared broker is a
-- spurious wake when another tenant writes its control row; the re-read is
-- correct either way. If a dashboard ever needs tenant routing, add a NEW
-- channel rather than re-shaping this one.
