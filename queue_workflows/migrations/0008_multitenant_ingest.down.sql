-- Reverse 0008.
--
-- NB: re-adding the CHECKs assumes only fetch/load (ingest_jobs) and cpu/gpu
-- (worker_heartbeats) rows remain — a host that used custom queue names must
-- drain/delete those rows before downgrading or the ADD CONSTRAINT fails.
-- Dropping `args` is lossy.

ALTER TABLE worker_heartbeats
    ADD CONSTRAINT worker_heartbeats_queue_check CHECK (queue IN ('cpu', 'gpu'));

ALTER TABLE ingest_jobs
    ADD CONSTRAINT ingest_jobs_queue_check CHECK (queue IN ('fetch', 'load'));

ALTER TABLE ingest_jobs
    DROP COLUMN IF EXISTS args;
