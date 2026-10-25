"""Two servers holding the SAME model is two copies of VRAM — the guard must see it.

The failure: ollama and vLLM both resident on ``chat-model-a:latest``, the box a few GB
from OOM. The
residency enforcer waved it through because it counted distinct MODEL IDS, and both
copies share one id.

The cap is about VRAM, and each server loads its own weights, so the unit is the
``(server, model)`` COPY.
"""
from __future__ import annotations

from queue_workflows.model_residency import Resident, decide_evictions


def _r(server, model, mru=0.0):
    return Resident(server=server, model=model, mru=mru)


def test_the_same_model_on_two_servers_is_a_violation():
    """THE bug. Two copies of the weights, not one — one of them has to go."""
    residents = [_r("ollama", "chat-model-a:latest", 10), _r("vllm", "chat-model-a:latest", 5)]
    evicted = decide_evictions(residents, cap=1)
    assert len(evicted) == 1, "one of the two copies must be evicted"
    assert len({(r.server, r.model) for r in residents}
               - {(r.server, r.model) for r in evicted}) == 1


def test_the_desired_server_keeps_its_copy():
    residents = [_r("ollama", "chat-model-a:latest", 10), _r("vllm", "chat-model-a:latest", 5)]
    evicted = decide_evictions(residents, cap=1, desired_server="vllm")
    assert [(r.server) for r in evicted] == ["ollama"]


def test_most_recently_used_wins_when_no_desired_server():
    residents = [_r("ollama", "m", 1), _r("vllm", "m", 99)]
    evicted = decide_evictions(residents, cap=1)
    assert [r.server for r in evicted] == ["ollama"]


def test_one_copy_is_never_evicted():
    assert decide_evictions([_r("vllm", "chat-model-a:latest")], cap=1) == []


def test_a_cap_of_two_permits_deliberate_co_residency():
    """Raising the cap is how an operator opts INTO two copies — it must then be
    allowed, or the knob would be a lie."""
    residents = [_r("ollama", "chat-model-a:latest"), _r("vllm", "chat-model-a:latest")]
    assert decide_evictions(residents, cap=2) == []


def test_two_models_on_one_server_is_still_a_violation():
    """The pre-existing behaviour must survive the regrouping."""
    residents = [_r("ollama", "a", 1), _r("ollama", "b", 2)]
    evicted = decide_evictions(residents, cap=1)
    assert [r.model for r in evicted] == ["a"]


def test_duplicate_sightings_of_one_copy_are_not_two_copies():
    """The same (server, model) reported twice by two probes is ONE copy in VRAM."""
    residents = [_r("vllm", "m", 1), _r("vllm", "m", 2)]
    assert decide_evictions(residents, cap=1) == []


def test_a_BUSY_server_is_never_the_victim():
    """Evicting a server with requests in flight kills live work to satisfy a
    bookkeeping cap. And desired_server is only the box DEFAULT now that routing is
    per node-job, so it must not outvote observed reality: a vLLM serving three
    requests beats an idle ollama that merely happens to be configured as default."""
    residents = [
        Resident(server="ollama", model="chat-model-a:latest", mru=99),
        Resident(server="vllm", model="chat-model-a:latest", mru=0.0, busy=True),
    ]
    evicted = decide_evictions(residents, cap=1, desired_server="ollama")
    assert [r.server for r in evicted] == ["ollama"], (
        "the busy vLLM must survive even though ollama is the default AND newer")


def test_with_neither_busy_COST_outranks_the_default():
    """CONTRACT FLIP: this test used to assert desired_server=ollama
    evicts the vLLM copy of the same model — which is precisely the eviction loop
    that kept killing a slow-to-restore server to save a fast-to-restore copy. Cost decides."""
    residents = [
        Resident(server="ollama", model="m", mru=1),
        Resident(server="vllm", model="m", mru=99),
    ]
    evicted = decide_evictions(residents, cap=1, desired_server="ollama")
    assert [r.server for r in evicted] == ["ollama"]


def test_a_WARMING_server_is_never_evicted():
    """A server still loading its weights has no traffic, so both the busy vote AND
    the mru vote pick it as the victim — which is exactly how the incoming vLLM was
    killed on every switch attempt. While a switch is warming, its target is
    untouchable, outranking even a busy incumbent."""
    residents = [
        Resident(server="ollama", model="chat-model-a:latest", mru=99, busy=True),
        Resident(server="vllm", model="chat-model-a:latest", mru=0.0, busy=False),
    ]
    evicted = decide_evictions(
        residents, cap=1, desired_server="ollama", protected_server="vllm")
    assert [r.server for r in evicted] == ["ollama"], (
        "the warming vLLM must survive a busy, default, newer ollama")


def test_protection_is_scoped_to_the_switch_target():
    residents = [
        Resident(server="ollama", model="a", mru=1),
        Resident(server="vllm", model="b", mru=2),
    ]
    evicted = decide_evictions(residents, cap=1, protected_server="ollama")
    assert [r.server for r in evicted] == ["vllm"]


def test_same_model_on_both_kinds_ALWAYS_evicts_the_ollama_copy():
    """THE EVICTION LOOP. One stray ollama-resolved job put an LLM on
    ollama beside a serving vLLM; the (busy, desired, mru) vote — ollama never busy,
    desired default = ollama — then killed the slow-to-restore vLLM to save the
    seconds-to-restore ollama copy. Every time. Cost decides: same model on both ⇒
    the ollama copy is the victim, whatever busy/desired/mru say."""
    residents = [
        Resident(server="ollama", model="chat-model-a:latest", mru=99),
        Resident(server="vllm", model="chat-model-a:latest", mru=0.0, busy=False),
    ]
    evicted = decide_evictions(residents, cap=1, desired_server="ollama")
    assert [(r.server) for r in evicted] == ["ollama"]


def test_the_cost_rule_does_not_touch_distinct_models():
    residents = [
        Resident(server="ollama", model="a", mru=99, busy=True),
        Resident(server="vllm", model="b", mru=1),
    ]
    evicted = decide_evictions(residents, cap=1)
    assert [(r.server) for r in evicted] == ["vllm"], "busy still wins across models"


def test_the_switch_shield_outranks_the_cost_rule():
    """AUDIT (post-approve delta): a warming OLLAMA switch target must survive even
    a same-model dup — the shield exists precisely because a warming server has no
    traffic to defend itself with. Cost picks among the unprotected only."""
    residents = [
        Resident(server="ollama", model="m", mru=1),
        Resident(server="vllm", model="m", mru=99, busy=True),
    ]
    evicted = decide_evictions(residents, cap=1, protected_server="ollama")
    assert [r.server for r in evicted] == ["vllm"]
