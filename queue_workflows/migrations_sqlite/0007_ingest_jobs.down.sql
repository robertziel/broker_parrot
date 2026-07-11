-- Revert 0007: drop indexes and table.
DROP INDEX IF EXISTS ingest_jobs_lease_idx;
DROP INDEX IF EXISTS ingest_jobs_claim_idx;
DROP TABLE IF EXISTS ingest_jobs;
