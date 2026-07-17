"""One-model-per-GPU-box arbitration — the pure decision, the file store, and the
ModelLease wrapper. Enforces the invariant across processes/projects on one box.
"""
from __future__ import annotations

import queue_workflows
from queue_workflows import gpu_model_lease as gml
from queue_workflows.gpu_model_lease import (
    FileLeaseStore, LeaseState, ModelLease, ModelLeaseDenied, NullLeaseStore, decide, release_state,
)


# ── decide(): the pure three-case rule ───────────────────────────────────────

def test_empty_box_grants():
    ok, st = decide(LeaseState(), "m1", "w1", now=100.0, ttl_s=60)
    assert ok and st.model == "m1" and st.holders == {"w1": 160.0}


def test_same_model_grants_and_shares():
    st0 = LeaseState("m1", {"w1": 160.0})
    ok, st = decide(st0, "m1", "w2", now=100.0, ttl_s=60)
    assert ok and st.model == "m1"
    assert set(st.holders) == {"w1", "w2"}          # both hold the ONE copy


def test_different_model_denied_while_holder_live():
    st0 = LeaseState("m1", {"w1": 160.0})
    ok, st = decide(st0, "m2", "w2", now=100.0, ttl_s=60)
    assert not ok and st.model == "m1"              # m1 stays; w2 must not load m2


def test_different_model_granted_once_holder_expired():
    st0 = LeaseState("m1", {"w1": 90.0})            # w1's lease already lapsed
    ok, st = decide(st0, "m2", "w2", now=100.0, ttl_s=60)
    assert ok and st.model == "m2" and set(st.holders) == {"w2"}   # dead holder evicted


def test_holder_reloading_same_slot_is_not_blocked_by_itself():
    st0 = LeaseState("m1", {"w1": 160.0})
    ok, st = decide(st0, "m2", "w1", now=100.0, ttl_s=60)   # w1 itself swaps model
    assert ok and st.model == "m2" and set(st.holders) == {"w1"}


def test_release_clears_slot_when_last_holder_leaves():
    st = release_state(LeaseState("m1", {"w1": 160.0}), "w1")
    assert st.model is None and st.holders == {}
    st2 = release_state(LeaseState("m1", {"w1": 160.0, "w2": 160.0}), "w1")
    assert st2.model == "m1" and set(st2.holders) == {"w2"}   # w2 still holds it


# ── NullLeaseStore: the safe default grants everything ───────────────────────

def test_null_store_lease_is_noop_grant():
    lease = ModelLease(box_id="box", store=NullLeaseStore(), holder="w1")
    assert lease.enforcing is False
    assert lease.acquire("anything") is True
    assert lease.acquire("something-else") is True   # never denies — no-op


# ── FileLeaseStore: cross-process box-wide enforcement ───────────────────────

def test_file_store_enforces_one_model_across_holders(tmp_path):
    store = FileLeaseStore(str(tmp_path))
    t = [1000.0]
    a = ModelLease(box_id="box-a", store=store, holder="proj-a:1", ttl_s=60, now_fn=lambda: t[0])
    b = ModelLease(box_id="box-a", store=store, holder="proj-b:2", ttl_s=60, now_fn=lambda: t[0])
    assert a.acquire("qwen") is True            # first worker takes the box
    assert b.acquire("qwen") is True            # same model → shared, fine
    assert b.acquire("flux") is False           # DIFFERENT model while A live → denied
    assert store.read("box-a").model == "qwen"
    # A dies (stops renewing); 61 s later B may preempt
    t[0] += 61
    assert b.acquire("flux") is True
    assert store.read("box-a").model == "flux"


def test_file_store_release_frees_the_box(tmp_path):
    store = FileLeaseStore(str(tmp_path))
    a = ModelLease(box_id="box-a", store=store, holder="w1", ttl_s=60, now_fn=lambda: 1.0)
    b = ModelLease(box_id="box-a", store=store, holder="w2", ttl_s=60, now_fn=lambda: 1.0)
    assert a.acquire("qwen") and not b.acquire("flux")
    a.release()                                  # A unloads
    assert b.acquire("flux") is True             # box is free now
    assert store.read("box-a").model == "flux"


def test_acquire_or_raise(tmp_path):
    store = FileLeaseStore(str(tmp_path))
    a = ModelLease(box_id="b", store=store, holder="w1", now_fn=lambda: 1.0)
    b = ModelLease(box_id="b", store=store, holder="w2", now_fn=lambda: 1.0)
    a.acquire_or_raise("qwen")
    try:
        b.acquire_or_raise("flux")
        assert False, "expected denial"
    except ModelLeaseDenied as exc:
        assert "qwen" in str(exc) and "flux" in str(exc)


# ── config wiring: safe default + opt-in ─────────────────────────────────────

def test_build_lease_defaults_to_noop(monkeypatch):
    for k in ("QUEUE_WORKFLOWS_GPU_MODEL_LEASE_DIR", "QUEUE_WORKFLOWS_GPU_BOX_ID"):
        monkeypatch.delenv(k, raising=False)
    queue_workflows.set_gpu_lease_store(None)
    lease = gml.build_lease()
    assert lease.enforcing is False              # no store configured ⇒ no-op

def test_build_lease_uses_dir_env(monkeypatch, tmp_path):
    queue_workflows.set_gpu_lease_store(None)
    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_MODEL_LEASE_DIR", str(tmp_path))
    lease = gml.build_lease()
    assert lease.enforcing is True and isinstance(lease.store, FileLeaseStore)


def test_box_id_prefers_env_over_hostname(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_GPU_BOX_ID", "box-a-physical")
    assert gml.default_box_id() == "box-a-physical"


def test_injected_store_hook_wins(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_GPU_MODEL_LEASE_DIR", raising=False)
    sentinel = NullLeaseStore()
    queue_workflows.set_gpu_lease_store(sentinel)
    try:
        assert gml.build_lease().store is sentinel
    finally:
        queue_workflows.set_gpu_lease_store(None)


def test_file_store_artifacts_are_cross_uid_writable(tmp_path):
    """The lease file coordinates ROOT containers AND host (non-root) processes on one
    box — whoever creates it first must not lock the others out (the observed failure:
    a root container created it 0644, then a non-root host worker hit EACCES and could
    not participate at all). So the store chmods the lock file 0666 and the lease dir
    0777 (a coordination lockfile, not data — the classic shared-lock permission)."""
    import os
    import stat

    d = str(tmp_path / "lease")
    store = gml.FileLeaseStore(d)
    store.update("box-a", lambda st: (True, st))
    dmode = stat.S_IMODE(os.stat(d).st_mode)
    fmode = stat.S_IMODE(os.stat(os.path.join(d, "gpu_model_lease_box-a.json")).st_mode)
    assert fmode == 0o666, f"lease file must be world-writable, got {oct(fmode)}"
    assert dmode == 0o777, f"lease dir must be world-writable, got {oct(dmode)}"
