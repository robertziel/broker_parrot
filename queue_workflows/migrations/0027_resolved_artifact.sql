-- 0027 — the ARTIFACT behind the role tag: digest on the catalog, artifact on the row.
--
-- WHY. Serving traffic dials ROLE tags (`chat-expert-a`) on purpose — they are stable across
-- promotions, the claim SQL / warm-affinity / GPU lease all key on them, and re-pointing
-- one is what a promotion IS. But that means `resolved_model` answers "which role" and
-- nothing anywhere answers "which BYTES": a role tag like `chat-expert-a` can resolve to a
-- GGUF built from a bad merge (the untuned base) for days, while every job row
-- truthfully says `chat-expert-a` the whole time. A name is an assertion nobody verifies.
--
-- The artifact tag (`chat-expert-a-v2-q4km-ollama`) is the checkable name: immutable, one tag
-- one set of bytes, `-<server>` suffix stating the serving stack. This migration gives the
-- fleet the two columns needed to OBSERVE it rather than declare it:
--
--   * `llm_model_catalog.digest` — what the box's server reported for each tag (ollama
--     /api/tags carries a content digest per model). Two tags with one digest are one set
--     of bytes, which is how a role tag is matched to the artifact it currently serves —
--     per box, from observation, promotion-proof.
--
--   * `workflow_node_jobs.resolved_artifact` — stamped by the claim worker from ITS OWN
--     live observation right after a claim: the artifact tag whose digest equals the
--     resolved role tag's digest on THIS box. NULL when the box's server reports no
--     digests (vLLM) and the served id itself is not an artifact tag, or when nothing
--     matches — an absent answer, never a guessed one.
--
-- The role/artifact split itself is the host project's naming convention
-- (`<role>-v<N>-<precision>-<server>`); the engine stores and stamps, it does
-- not name.

ALTER TABLE llm_model_catalog
    ADD COLUMN IF NOT EXISTS digest TEXT NOT NULL DEFAULT '';

ALTER TABLE workflow_node_jobs
    ADD COLUMN IF NOT EXISTS resolved_artifact TEXT;
