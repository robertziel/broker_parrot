-- 0020 — per-node-job PHYSICAL-BOX placement constraints.
--
-- Lets a queued node-job pin or exclude the physical box(es) that may execute it,
-- by BOX NAME (the value a worker agrees on via ``QUEUE_WORKFLOWS_GPU_BOX_ID`` /
-- ``gpu_model_lease.default_box_id()`` — NOT the per-project ``host_label``). Two
-- optional arrays, both NULL by default (⇒ unconstrained, every box eligible, so
-- every existing row + consumer is byte-identical):
--
--   * avoid_box — the job MUST NOT run on any listed box (e.g. keep video renders
--                 off the box-a control hub: ``avoid_box := {box-a}``);
--   * force_box — the job may run ONLY on a listed box (hard pin, e.g. a node that
--                 needs data staged on one machine: ``force_box := {box-c}``).
--
-- The claim SQL ANDs both into its WHERE (a box claims iff it is NOT in avoid_box
-- AND, when force_box is set, IS in force_box), so an ineligible worker never grabs
-- the row — an eligible peer does. text[] mirrors ``worker_heartbeats.known_models``
-- so the sqlite dialect stores/reads it as JSON via the same seam.
ALTER TABLE workflow_node_jobs ADD COLUMN IF NOT EXISTS avoid_box text[];
ALTER TABLE workflow_node_jobs ADD COLUMN IF NOT EXISTS force_box text[];
