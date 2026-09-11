"""EVERY GPU row has a serving kind — the closed NULL hole in the per-kind arbiter.

History. Migration 0023 added per-kind slot caps ("ollama allows 1!! vllm allows 3!!")
but exempted any row with no ``server_types``: stamped NULL, it bypassed the caps
entirely — rationalised as "not LLM-served work". On the operator's panel
showed where that leads: box-a ran ``comfyui_video`` BESIDE ``compose_report``
(96 % GPU) — BOTH rows undeclared, both exempt, and the second one is an
ollama LLM user by the dispatch layer's own documentation (``llm_backends/factory.py``
names compose_report as *"an LLM user with no model field, so no server_types
ever gets stamped"*). Undeclared ≠ not LLM-served: 1530 rows of another LLM-calling node
sat in the same class in one 2-day window.

THE LAW (operator decision — "it's not vllm: ollama, comfyui and any
custom should allow 1 only"):

* A GPU row's resolved kind is **NEVER NULL**: declared preference → ``'inprocess'``
  for an in-process model row → the box-default prediction (what dispatch's
  active-server-wins default will actually do).
* Width is per KIND: ``kind_caps[kind]`` (vLLM = PAR), and **defaults to 1** for
  every other kind — ollama, comfyui, any custom, any typo (D3's tightest-budget
  rule, now literal).
* A legacy NULL running row (old wheel) **counts against every candidate's kind** —
  unknown is not free.
* The law is ON whenever the claim is host-scoped: ``kind_caps=None`` means "engine
  defaults", the explicit empty dict ``{}`` is the documented opt-out.

Deliberately NOT in the law: cross-KIND admission exclusion. The two-lane worker
contract (inline warm-model diffusion beside a PAR VLM pool — ai_leads' shipped
topology) requires an ``inprocess`` job and pool LLM work to coexist on one box;
cross-kind stacking is arbitrated at EXECUTION by the box model lease and the
residency enforcer, which see what is actually resident rather than a prediction.
"""
from __future__ import annotations

from queue_workflows import llm_probe, node_queue
from queue_workflows.db import connection
from tests._helpers import make_run

POOL = ("llm_chat",)
CAPS = {"ollama": 1, "vllm": 3}


def _q(prefs, model="chat-model-a:latest", module="llm_chat"):
    return node_queue.enqueue_node_job(
        run_id=make_run(), node_id=f"n{id(prefs)}{id(model)}", node_module=module,
        queue="gpu", required_model=model, server_types=prefs)


def _plain(node_id, module="llm_chat"):
    """An UNDECLARED no-model gpu row — the ``compose_report`` /
    ``comfyui_video`` class this law exists for."""
    return node_queue.enqueue_node_job(
        run_id=make_run(), node_id=node_id, node_module=module, queue="gpu")


def _claim(served, **kw):
    kw.setdefault("kind_caps", CAPS)
    return node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=False, pool_modules=POOL,
        served_types=served, **kw)


# ── width within one kind (the surviving 0023 behaviour) ──────────────────────

def test_ollama_resolved_jobs_cap_at_ONE_even_with_three_box_slots():
    for _ in range(3):
        _q(["ollama"])
    assert _claim(["ollama", "vllm"]) is not None
    assert _claim(["ollama", "vllm"]) is None, (
        "ollama serialises — a second concurrent ollama-resolved job buys nothing")


def test_vllm_resolved_jobs_reach_the_full_PAR():
    for _ in range(4):
        _q(["vllm", "ollama"])
    got = [j for j in (_claim(["ollama", "vllm"]) for _ in range(4)) if j]
    assert len(got) == 3, "vLLM batches — PAR-wide is correct"
    assert all(j["resolved_server"] == "vllm" for j in got)


def test_the_kinds_do_not_starve_each_other():
    """1 ollama + 3 vllm still run TOGETHER — the caps are per kind, not shared.
    KEPT under the law on purpose: cross-kind stacking is the box
    lease's and the residency enforcer's call at execution time (they see what is
    actually resident); admission only bounds each kind's own width. This is also
    the two-lane worker contract — killing it would serialise ai_leads' inline
    diffusion beside its VLM pool fleet-wide."""
    _q(["ollama"]); _q(["ollama"])
    for _ in range(3):
        _q(["vllm"])
    got = [j for j in (_claim(["ollama", "vllm"]) for _ in range(6)) if j]
    kinds = sorted(j["resolved_server"] for j in got)
    assert kinds == ["ollama", "vllm", "vllm", "vllm"]


def test_the_stamp_is_the_first_SERVED_preference():
    _q(["vllm", "ollama"])
    got = _claim(["ollama"])            # box serves only ollama → fallback wins
    assert got["resolved_server"] == "ollama"


def test_a_refused_row_spills_to_another_box():
    """Width refusal here is spill-there, never starvation: the arbiter is per host."""
    _q(["ollama"]); _q(["ollama"])
    assert _claim(["ollama", "vllm"])["resolved_server"] == "ollama"
    assert _claim(["ollama", "vllm"]) is None
    other = node_queue.claim_next_gpu_job(
        0, None, host="h2-gpu", require_model=False, pool_modules=POOL,
        served_types=["ollama", "vllm"], kind_caps=CAPS)
    assert other is not None and other["resolved_server"] == "ollama"


# ── undeclared rows: the closed NULL hole ─────────────────────────────────────

def test_an_undeclared_no_model_row_is_stamped_the_default_kind():
    """FLIPS ``test_an_unconstrained_row_is_NOT_dragged_under_the_llm_caps``. The old
    exemption reasoned "a no-model pool job is not LLM-served"; the dispatch layer's
    own docs name compose_report as an LLM user with exactly this row shape.
    Undeclared work IS served by something — count it where it will run."""
    _plain("plain")
    got = _claim(["ollama", "vllm"], default_server="ollama")
    assert got["resolved_server"] == "ollama"


def test_undeclared_rows_are_capped_not_exempt():
    """FLIPS ``test_unconstrained_rows_run_beside_a_full_ollama_cap`` — THE SCREENSHOT
    TEST. box-a: comfyui_video (undeclared → predicted ollama) ran beside
    compose_report (undeclared ollama user). Same predicted kind, width 1 ⇒
    the second claim must be refused."""
    _plain("comfyui_video_like", module="llm_chat")
    assert _claim(["ollama", "vllm"], default_server="ollama") is not None
    _plain("compose_report_like", module="llm_chat")
    assert _claim(["ollama", "vllm"], default_server="ollama") is None, (
        "the pair: two undeclared rows must not run beside each other")


def test_default_server_falls_back_to_ollama_when_caller_gives_none():
    """A caller that supplies no prediction (old client code on a new wheel) must
    still stamp SOMETHING — the engine-wide default, never NULL."""
    _plain("plain3")
    got = _claim(["ollama", "vllm"])            # no default_server kwarg at all
    assert got["resolved_server"] == "ollama"


def test_the_default_prediction_follows_the_caller():
    """A box whose active server is vLLM predicts vllm for undeclared rows — the
    claim counts the job where dispatch (active-server-wins, 6931668) will actually
    put it, and vllm width lets the pool keep its PAR."""
    _plain("plain4")
    got = _claim(["ollama", "vllm"], default_server="vllm")
    assert got["resolved_server"] == "vllm"


# ── in-process model rows join the law ────────────────────────────────────────

def test_an_inprocess_model_row_is_stamped_inprocess():
    """A required_model row with no server_types loads weights IN THIS PROCESS
    (tts-model). That is a serving kind of its own — llm_probe.INPROCESS — not a
    bystander: its width is 1 and a legacy NULL row blocks it like any other."""
    _q(None, model="tts-model-v1")
    got = node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=True,
        known_models=["tts-model-v1"], kind_caps=CAPS)
    assert got is not None
    assert got["resolved_server"] == llm_probe.INPROCESS


def test_inprocess_width_is_one():
    _q(None, model="tts-model-v1")
    _q(None, model="tts-model-v1")
    assert node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=True,
        known_models=["tts-model-v1"], kind_caps=CAPS) is not None
    assert node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=True,
        known_models=["tts-model-v1"], kind_caps=CAPS) is None, (
        "two in-process loads on one box is the OOM shape")


def test_inline_and_pool_lanes_stay_cotenant():
    """The two-lane contract holds: a running in-process diffusion job does NOT
    block pool LLM work on the same box at admission — that co-tenancy is ai_leads'
    shipped topology, and execution-time stacking belongs to the lease/residency
    layer, which sees actual residents instead of predictions."""
    _q(None, model="tts-model-v1")
    assert node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=True,
        known_models=["tts-model-v1"], kind_caps=CAPS) is not None
    _q(["ollama"])
    got = _claim(["ollama", "vllm"])
    assert got is not None and got["resolved_server"] == "ollama"


def test_a_remote_served_model_row_keeps_its_declared_kind():
    """server_types beats required_model in the stamp: a model-naming row served by
    an EXTERNAL server (0022 remote-served) is that server's work, not inprocess."""
    _q(["vllm"], model="chat-model-a:latest")
    got = _claim(["ollama", "vllm"])
    assert got["resolved_server"] == "vllm"


def test_the_inprocess_kind_literal_matches_llm_probe():
    """Drift guard: the SQL stamps the same literal the residency layer probes."""
    assert node_queue.INPROCESS_SERVER_KIND == llm_probe.INPROCESS


def test_the_fallback_default_literal_matches_worker_control():
    """Drift guard the constant's comment promises (audit LOW #1): node_queue keeps
    its own literal because worker_control imports node_queue (cycle), so the pair
    must be pinned EQUAL here — behavioural tests pin each side to 'ollama'
    separately, but only this names the pairing."""
    from queue_workflows import worker_control
    assert node_queue._DEFAULT_SERVER_KIND == worker_control.DEFAULT_LLM_SERVER_TYPE


# ── legacy rows and escape hatches ────────────────────────────────────────────

def test_a_legacy_null_running_row_counts_against_every_kind():
    """A running row stamped NULL can only come from an old wheel that predates the
    stamp. The engine cannot know what that job put on the card — and (a
    07-19 wheel admitted a render beside an LLM job) is what trusting the unknown
    looks like. Unknown ≠ free: it occupies one slot of WHATEVER kind asks next."""
    _q(["ollama"])
    got = _claim(["ollama", "vllm"])
    assert got is not None
    with connection() as c, c.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET resolved_server = NULL "
                    "WHERE id = %s", (got["id"],))
    _q(["ollama"])
    assert _claim(["ollama", "vllm"]) is None, "NULL consumed ollama's one slot"
    _q(None, model="tts-model-v1")
    assert node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=True,
        known_models=["tts-model-v1"], kind_caps=CAPS) is None, (
        "NULL blocks the width-1 inprocess kind too")


def test_a_legacy_null_row_consumes_one_vllm_slot_not_all():
    """Against a BATCHING kind the unknown row is one occupant, not a wall — vLLM
    still reaches PAR-1. Blanket blocking would freeze a box for a whole legacy
    job's runtime for no physical reason."""
    _q(["vllm"])
    got = _claim(["ollama", "vllm"])
    assert got is not None
    with connection() as c, c.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET resolved_server = NULL "
                    "WHERE id = %s", (got["id"],))
    _q(["vllm"]); _q(["vllm"]); _q(["vllm"])
    admitted = [j for j in (_claim(["ollama", "vllm"]) for _ in range(3)) if j]
    assert len(admitted) == 2, "NULL + 2 live = PAR(3); the 4th must wait"


def test_an_unknown_custom_kind_gets_width_one():
    """D3 chose the TIGHTEST budget for a kind the build has never heard of; the law
    keeps that: any custom server kind serialises unless the operator caps it wider."""
    _q(["tgi"]); _q(["tgi"])
    a = _claim(["tgi"])
    assert a is not None and a["resolved_server"] == "tgi"
    assert _claim(["tgi"]) is None


def test_the_law_is_on_by_default_when_host_scoped():
    """kind_caps=None must mean "engine defaults", NOT "off" — an old caller on a new
    wheel gets the law without asking. Off-by-default is how the last hole shipped."""
    _plain("d1"); _plain("d2")
    a = node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=False, pool_modules=POOL,
        max_running=3)
    assert a is not None and a["resolved_server"] == "ollama"
    assert node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=False, pool_modules=POOL,
        max_running=3) is None, "undeclared width is 1 under the default law"


def test_default_law_gives_vllm_the_box_budget():
    """Under engine defaults the one batching kind gets max_running as its PAR."""
    for _ in range(3):
        _q(["vllm"])
    got = [node_queue.claim_next_gpu_job(
        0, None, host="h-gpu", require_model=False, pool_modules=POOL,
        served_types=["vllm"], max_running=2) for _ in range(3)]
    assert len([j for j in got if j]) == 2


def test_empty_kind_caps_is_the_documented_opt_out():
    """``kind_caps={}`` — explicitly empty — restores the legacy arbiter AND the
    legacy NULL stamp, byte-compat for a caller with its own arbitration."""
    _q(["ollama"]); _q(["ollama"])
    a = _claim(["ollama", "vllm"], kind_caps={}, max_running=2)
    b = _claim(["ollama", "vllm"], kind_caps={}, max_running=2)
    assert a is not None and b is not None, "opt-out keeps the pre-law behaviour"
    assert a["resolved_server"] is None and b["resolved_server"] is None


def test_cpu_claims_are_untouched_by_the_law():
    for n in ("c1", "c2"):
        node_queue.enqueue_node_job(
            run_id=make_run(), node_id=n, node_module="anything", queue="cpu")
    a = node_queue.claim_next_cpu_job(0, host="h-cpu")
    b = node_queue.claim_next_cpu_job(0, host="h-cpu")
    assert a is not None and b is not None
