-- Revert 0019 — drop the tenant tag from the operator control plane.
--
-- DESTRUCTIVE (deliberately, and unavoidably): the 2-col PK cannot be restored
-- while two projects hold a control row for the same (host_label, queue). Those
-- rows only exist on a shared broker that has ALREADY cut over, and a downgrade
-- is precisely the statement "we are no longer multi-tenant here", so the
-- non-default tenants' rows are dropped. Control rows are cheap operator state
-- (desired_state + LLM config), NOT queue work — nothing in flight is lost, and
-- a worker with no control row defaults to ON (``desired_state_for``).
--
-- The single-tenant sentinel ('') rows are kept — they are the rows a
-- pre-0019 deploy would have had.

DELETE FROM worker_controls WHERE project <> '';

-- The 0012/0013 NOTIFY trigger functions are untouched by 0019, so there is
-- nothing to restore here.
ALTER TABLE worker_controls DROP CONSTRAINT IF EXISTS worker_controls_pkey;
ALTER TABLE worker_controls DROP COLUMN IF EXISTS project;
ALTER TABLE worker_controls
    ADD CONSTRAINT worker_controls_pkey PRIMARY KEY (host_label, queue);
