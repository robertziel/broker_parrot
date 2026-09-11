-- 0026 — the model CATALOG, and an ARRAY of acceptable models per node-job.
--
-- WHY. Two questions had no answer in this schema:
--
--   1. "Which MODELS can a given box's given server actually serve?"
--      0014's `worker_heartbeats.llm_servers_available` answers only which server
--      KINDS a box runs. The claim path approximated the rest with a per-worker
--      `known_models` config list, so the fleet had no queryable, observed answer
--      and a job naming a model nobody holds waited forever with nothing saying why.
--
--   2. "Can this job run on EITHER of these models, on these servers?"
--      `required_model` is one string. A caller could say "chat-model-a:latest" but not
--      "the fine-tune if it is loaded, else the general model" — nor pin a model to a
--      server kind independently of the job's overall server preference.
--
-- BOTH SHAPES STAY. `required_model` is not deprecated and not backfilled away: every
-- row written before this carries it, the single-model case is the common one, and
-- `acceptable_models()` in queue_workflows/model_catalog.py flattens either shape into
-- one ordered list of (model, server) pairs. The scalar/array distinction therefore
-- exists at the edge and nowhere else.
--
-- Writers DUAL-WRITE: set `required_model` to the array's FIRST model so an old reader
-- (an un-bumped worker mid-rollout) still claims correctly, and `required_models` for a
-- new one. That is what makes this a no-flag-day migration.

CREATE TABLE IF NOT EXISTS llm_model_catalog (
    host_label   TEXT        NOT NULL,
    server_type  TEXT        NOT NULL,
    model_id     TEXT        NOT NULL,
    -- Operator lever: park a model on one box without deleting the observation.
    enabled      BOOLEAN     NOT NULL DEFAULT TRUE,
    -- OBSERVED, not declared (the llm_probe lesson): when did a server last report it.
    observed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (host_label, server_type, model_id)
);

-- The claim asks "can THIS box serve any of these models" once per candidate row.
CREATE INDEX IF NOT EXISTS llm_model_catalog_lookup
    ON llm_model_catalog (host_label, server_type)
 WHERE enabled;

-- "Where can this model run" — the scheduler/diagnostic direction.
CREATE INDEX IF NOT EXISTS llm_model_catalog_by_model
    ON llm_model_catalog (model_id)
 WHERE enabled;

-- The ordered array of acceptable {model, server_types} entries, as JSONB so an entry
-- can gain fields (a box list, a min VRAM) without another migration. NULL = use the
-- scalar `required_model`, which is exactly how every existing row behaves.
ALTER TABLE workflow_node_jobs
    ADD COLUMN IF NOT EXISTS required_models JSONB;

-- The FLAT list of model ids from `required_models`, denormalised purely so the claim
-- gate can use the same cheap `arrays_overlap` it already uses for `server_types`.
-- Matching inside JSONB would need per-dialect JSON extraction in the hot claim path;
-- this keeps the SQL identical on pg and sqlite. The JSONB stays the source of truth for
-- ORDER and per-entry server pinning — this column is an index, not a second opinion,
-- and both are written together by the same enqueue call.
ALTER TABLE workflow_node_jobs
    ADD COLUMN IF NOT EXISTS required_model_ids TEXT[];

-- Which model actually got chosen at claim time. `resolved_server` (0023) already
-- records the KIND; without the model beside it, a row naming three acceptable models
-- gives no way to tell which one ran — and the warm-affinity sort needs the pair.
ALTER TABLE workflow_node_jobs
    ADD COLUMN IF NOT EXISTS resolved_model TEXT;
