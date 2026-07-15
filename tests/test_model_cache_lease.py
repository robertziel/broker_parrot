"""ModelCache honours the one-model-per-GPU-box lease.

The warm cache guarantees one model per *process*; the box lease
(:mod:`queue_workflows.gpu_model_lease`) guarantees one per *box* — the thing
that actually protects VRAM when N workers (one per project) share a card.

Contract:
  * with NO lease (default) the cache behaves exactly as before — nothing may
    regress for a deploy that hasn't opted in;
  * with a lease, a load ACQUIRES the box first; a denial raises
    ``ModelLeaseDenied`` and the cache does NOT load (and does not clobber the
    model it already holds);
  * re-serving the SAME model (a cache hit) never re-acquires — no lease traffic
    on the hot path;
  * dropping/unloading RELEASES the box so the next worker can take it.
"""
from __future__ import annotations

import pytest

from queue_workflows import model_registry
from queue_workflows.gpu_model_lease import (
    FileLeaseStore, ModelLease, ModelLeaseDenied,
)
from queue_workflows.model_cache import ModelCache


@pytest.fixture(autouse=True)
def _models():
    """Two cheap fake models in the registry."""
    saved = dict(model_registry.MODELS)
    model_registry.MODELS.clear()
    for mid in ("m1", "m2"):
        model_registry.register(model_registry.ModelSpec(id=mid, loader=lambda mid=mid: f"handle::{mid}"))
    yield
    model_registry.MODELS.clear()
    model_registry.MODELS.update(saved)


def _lease(tmp_path, holder, now=lambda: 1000.0):
    return ModelLease(box_id="box-a", store=FileLeaseStore(str(tmp_path)),
                      holder=holder, ttl_s=60, now_fn=now)


# ── default: no lease ⇒ unchanged behavior ──────────────────────────────────

def test_no_lease_loads_freely():
    c = ModelCache()
    assert c.require_model("m1") == "handle::m1"
    assert c.require_model("m2") == "handle::m2"        # swaps freely, as before
    assert c.current_model == "m2"


# ── with a lease: the box gates the load ────────────────────────────────────

def test_lease_is_acquired_before_loading(tmp_path):
    lease = _lease(tmp_path, "w1")
    c = ModelCache(lease=lease)
    assert c.require_model("m1") == "handle::m1"
    assert lease.current().model == "m1"                # the box now records m1
    assert "w1" in lease.current().holders


def test_denied_load_raises_and_does_not_clobber(tmp_path):
    store = FileLeaseStore(str(tmp_path))
    other = ModelLease(box_id="box-a", store=store, holder="other", ttl_s=60, now_fn=lambda: 1000.0)
    assert other.acquire("m2")                          # a LIVE peer holds m2 on this box

    mine = ModelLease(box_id="box-a", store=store, holder="w1", ttl_s=60, now_fn=lambda: 1000.0)
    c = ModelCache(lease=mine)
    with pytest.raises(ModelLeaseDenied):
        c.require_model("m1")                           # different model → denied
    assert c.current_model is None                      # nothing loaded
    assert other.current().model == "m2"                # peer's model untouched


def test_same_model_as_peer_is_allowed(tmp_path):
    store = FileLeaseStore(str(tmp_path))
    peer = ModelLease(box_id="box-a", store=store, holder="peer", ttl_s=60, now_fn=lambda: 1000.0)
    peer.acquire("m1")
    c = ModelCache(lease=_lease(tmp_path, "w1"))
    assert c.require_model("m1") == "handle::m1"        # same model → shares the one copy
    assert c.current_model == "m1"


def test_cache_hit_does_not_reacquire(tmp_path):
    lease = _lease(tmp_path, "w1")
    calls = []
    orig = lease.acquire
    lease.acquire = lambda m: (calls.append(m), orig(m))[1]      # type: ignore[method-assign]
    c = ModelCache(lease=lease)
    c.require_model("m1")
    c.require_model("m1")                                # cache hit
    c.require_model("m1")
    assert calls == ["m1"]                               # acquired ONCE, not per call


def test_drop_cache_releases_the_box(tmp_path):
    store = FileLeaseStore(str(tmp_path))
    c = ModelCache(lease=ModelLease(box_id="box-a", store=store, holder="w1",
                                    ttl_s=60, now_fn=lambda: 1000.0))
    c.require_model("m1")
    assert store.read("box-a").model == "m1"
    c.drop_cache()                                       # unload / idle reaper
    assert store.read("box-a").model is None             # box freed for the next worker
    # and now a peer wanting a DIFFERENT model may take it
    peer = ModelLease(box_id="box-a", store=store, holder="peer", ttl_s=60, now_fn=lambda: 1000.0)
    assert peer.acquire("m2") is True
