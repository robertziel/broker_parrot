"""Migration 0029: the catalog remembers what a model costs on a box (its vLLM profile)."""
from __future__ import annotations

from queue_workflows import node_queue
from queue_workflows.model_catalog import CatalogEntry

PROFILE = {"weights_bytes": 20_122_000_000, "kv_bytes": 45_343_000_000, "kv_tokens": 684_056,
           "window": 262_144, "utilization": 0.42, "total_bytes": 129_922_000_000,
           "fingerprint": "abc123def4567890"}


def test_a_profile_is_recorded_and_read_back_for_its_box_and_model():
    node_queue.upsert_model_catalog([CatalogEntry("box-b-gpu", "vllm", "chat-model-a:latest")])
    assert node_queue.llm_profile("box-b-gpu", "vllm", "chat-model-a:latest") is None
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", PROFILE)
    assert node_queue.llm_profile("box-b-gpu", "vllm", "chat-model-a:latest") == PROFILE
    assert node_queue.llm_profile("box-c-gpu", "vllm", "chat-model-a:latest") is None


def test_the_observation_upsert_does_not_erase_a_profile():
    """The heartbeat probe re-upserts every catalog row it sees, every tick."""
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", PROFILE)
    node_queue.upsert_model_catalog([CatalogEntry("box-b-gpu", "vllm", "chat-model-a:latest",
                                                  digest="sha256:new")])
    assert node_queue.llm_profile("box-b-gpu", "vllm", "chat-model-a:latest") == PROFILE


def test_a_profile_for_a_model_the_catalog_has_not_seen_creates_its_row():
    node_queue.record_llm_profile("box-b-gpu", "vllm", "brand-new:latest", PROFILE)
    assert node_queue.llm_profile("box-b-gpu", "vllm", "brand-new:latest") == PROFILE
    rows = {(e.host_label, e.server_type, e.model_id) for e in node_queue.fetch_model_catalog()}
    assert ("box-b-gpu", "vllm", "brand-new:latest") in rows


def test_a_new_observation_replaces_the_old_profile():
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", PROFILE)
    newer = {**PROFILE, "kv_tokens": 900_000, "fingerprint": "ffff000011112222"}
    node_queue.record_llm_profile("box-b-gpu", "vllm", "chat-model-a:latest", newer)
    assert node_queue.llm_profile("box-b-gpu", "vllm", "chat-model-a:latest") == newer


def test_the_gpu_lane_waits_for_the_profile_column(monkeypatch):
    from queue_workflows import claim_worker, db

    seen = {}
    monkeypatch.setattr(db, "wait_for_schema", lambda min_version, **kw: seen.setdefault("v", min_version))
    claim_worker.ClaimWorker(queue="gpu", host="h").await_schema()
    assert seen["v"] == 29
