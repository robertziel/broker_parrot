"""ModelCache's LAST-DEFENCE load gate (`pre_load_check`).

The gate is the final backstop before weights hit VRAM: the GPU worker wires it to
`model_residency.assert_can_load`, which raises when the box already holds a different
model — even one the cooperative box lease never saw (an ollama daemon, or a project
that doesn't share the lease file). Here we pin the ModelCache side of the seam: the
gate runs before ANY state change, a refusal loads nothing and leaves the cache
intact, a same-model cache hit never gates, and unset ⇒ byte-identical to before.
"""
from __future__ import annotations

import pytest

from queue_workflows import model_registry
from queue_workflows.model_cache import ModelCache
from queue_workflows.model_residency import ModelAlreadyLoadedError, Resident, assert_can_load


@pytest.fixture(autouse=True)
def _models():
    saved = dict(model_registry.MODELS)
    model_registry.MODELS.clear()
    loads: list[str] = []
    for mid in ("m1", "m2"):
        def _loader(mid=mid):
            loads.append(mid)
            return f"handle::{mid}"
        model_registry.register(model_registry.ModelSpec(id=mid, loader=_loader))
    _models.loads = loads
    yield
    model_registry.MODELS.clear()
    model_registry.MODELS.update(saved)


def test_no_gate_loads_freely():
    c = ModelCache()
    assert c.require_model("m1") == "handle::m1"
    assert c.require_model("m2") == "handle::m2"
    assert c.current_model == "m2"


def test_gate_is_called_with_the_model_id_before_the_loader():
    seen = []
    c = ModelCache(pre_load_check=lambda mid: seen.append(mid))
    c.require_model("m1")
    assert seen == ["m1"]
    assert _models.loads == ["m1"]          # loader ran after the gate passed


def test_gate_raise_blocks_the_load_and_leaves_cache_untouched():
    def gate(mid):
        raise ModelAlreadyLoadedError("box already holds another model")

    c = ModelCache(pre_load_check=gate)
    with pytest.raises(ModelAlreadyLoadedError):
        c.require_model("m1")
    assert _models.loads == []              # loader NEVER called
    assert c.current_model is None          # cache unchanged — nothing loaded
    assert c.current_handle is None


def test_gate_does_not_clobber_a_model_already_held():
    # Hold m1, then a gated attempt at m2 is refused: m1 must survive intact.
    calls = {"n": 0}

    def gate(mid):
        calls["n"] += 1
        if mid == "m2":
            raise ModelAlreadyLoadedError("refused")

    c = ModelCache(pre_load_check=gate)
    assert c.require_model("m1") == "handle::m1"
    with pytest.raises(ModelAlreadyLoadedError):
        c.require_model("m2")
    assert c.current_model == "m1"          # still serving m1
    assert c.require_model("m1") == "handle::m1"   # and it's a hot cache hit


def test_same_model_cache_hit_never_calls_the_gate():
    seen = []
    c = ModelCache(pre_load_check=lambda mid: seen.append(mid))
    c.require_model("m1")
    c.require_model("m1")                    # cache hit
    c.require_model("m1")
    assert seen == ["m1"]                    # gated only on the ONE real load


def test_end_to_end_with_the_real_assert_can_load():
    # The exact production wiring: the gate is assert_can_load bound to a live
    # box-residents probe. A foreign resident ⇒ the load is refused.
    residents = [Resident(server="ollama", model="llama3", mru=1.0)]
    c = ModelCache(
        pre_load_check=lambda mid: assert_can_load(mid, residents, label="box-a-gpu"),
    )
    with pytest.raises(ModelAlreadyLoadedError) as e:
        c.require_model("m1")               # m1 != the resident llama3 → blocked
    assert "m1" in str(e.value) and "llama3" in str(e.value)
    assert _models.loads == []
