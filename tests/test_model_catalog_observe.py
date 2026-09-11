"""Observing which models each box's servers can actually serve, and publishing it.

The catalog is only worth having if it is OBSERVED — that is `llm_probe`'s whole lesson,
recorded in its own docstring after a GPU box advertised `{ollama}` while running no LLM
server at all. These tests pin the observation, including the part that is easy to get
wrong: the two server kinds answer a DIFFERENT question and both answers are correct.

  * ollama  — `/api/tags` lists every INSTALLED model. ollama swaps on demand, so all of
              them are servable.
  * vLLM    — `/v1/models` lists the ONE model it is currently serving. A vLLM pins its
              model for the process lifetime, so that is the only one it can serve.

Conflating them ("vLLM has these models available") would advertise capability a vLLM
cannot honour without a restart, and the claim gate would send it work it must refuse.
"""
from __future__ import annotations

from queue_workflows.llm_probe import ollama_available_models, observe_box_models
from queue_workflows.model_catalog import CatalogEntry


def test_ollama_reports_every_installed_model_because_it_swaps_on_demand():
    payload = {"models": [{"name": "chat-model-a:latest"}, {"name": "chat-expert-b:latest"}]}
    assert ollama_available_models("http://box:11434",
                                   get_json_fn=lambda *_a, **_k: payload) == [
        "chat-model-a:latest", "chat-expert-b:latest"]


def test_a_dead_or_absent_ollama_is_an_EMPTY_list_never_a_guess():
    def boom(*_a, **_k):
        raise OSError("connection refused")
    assert ollama_available_models("http://box:11434", get_json_fn=boom) == []
    assert ollama_available_models(None) == []


def test_observe_returns_one_entry_per_model_per_kind_tagged_with_the_box():
    entries = observe_box_models(
        "box-a-gpu",
        urls_by_kind={"ollama": "http://box:11434", "vllm": "http://box:8000"},
        ollama_fn=lambda url, **_k: ["chat-model-a:latest", "chat-expert-b:latest"],
        vllm_fn=lambda url, **_k: ["chat-model-a:latest"],
    )
    assert set(entries) == {
        CatalogEntry("box-a-gpu", "ollama", "chat-model-a:latest"),
        CatalogEntry("box-a-gpu", "ollama", "chat-expert-b:latest"),
        CatalogEntry("box-a-gpu", "vllm", "chat-model-a:latest"),
    }


def test_a_kind_with_no_url_contributes_nothing_rather_than_an_empty_claim():
    entries = observe_box_models(
        "box-c-gpu",
        urls_by_kind={"ollama": "http://box:11434", "vllm": None},
        ollama_fn=lambda url, **_k: ["chat-model-a:latest"],
        vllm_fn=lambda url, **_k: ["should-not-be-called"],
    )
    assert entries == [CatalogEntry("box-c-gpu", "ollama", "chat-model-a:latest")]


def test_a_box_whose_servers_all_answer_nothing_publishes_NOTHING():
    """The state llm_probe exists to make visible: absence, not a default."""
    assert observe_box_models(
        "box-b-gpu",
        urls_by_kind={"ollama": "http://box:11434", "vllm": "http://box:8000"},
        ollama_fn=lambda url, **_k: [],
        vllm_fn=lambda url, **_k: [],
    ) == []


# ── it must ride the heartbeat, and never break it ───────────────────
def test_the_publisher_is_best_effort_and_cannot_break_a_heartbeat():
    """A heartbeat that dies because a model list was slow is worse than a catalog one
    tick stale. The publish is wrapped, so a raising probe is swallowed."""
    from queue_workflows.claim_worker import ClaimWorker

    w = ClaimWorker.__new__(ClaimWorker)
    w.host = "box-a"

    class _Factory:
        @staticmethod
        def resolve_base_url(kind=None):
            raise OSError("endpoint gone")

    w._publish_model_catalog(["ollama"], "http://x", _Factory)   # must not raise


def test_nothing_is_published_when_the_box_advertises_no_server():
    """Absence stays absence: a box with no working server must not write rows claiming
    it can serve anything."""
    from queue_workflows.claim_worker import ClaimWorker

    w = ClaimWorker.__new__(ClaimWorker)
    w.host = "box-a"
    called = []

    class _Factory:
        @staticmethod
        def resolve_base_url(kind=None):
            called.append(kind)
            return "http://x"

    w._publish_model_catalog([], "http://x", _Factory)
    assert called == []
