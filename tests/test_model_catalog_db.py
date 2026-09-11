"""The catalog's DB seam: publish observed capability, read it back."""
from __future__ import annotations

import pytest

from queue_workflows import node_queue
from queue_workflows.model_catalog import CatalogEntry, ModelCatalog


def test_publish_then_read_round_trips():
    node_queue.upsert_model_catalog([
        CatalogEntry("box-a-gpu", "vllm", "chat-model-a:latest"),
        CatalogEntry("box-b-gpu", "ollama", "chat-model-a:latest"),
    ])
    rows = node_queue.fetch_model_catalog()
    assert CatalogEntry("box-a-gpu", "vllm", "chat-model-a:latest") in rows
    assert CatalogEntry("box-b-gpu", "ollama", "chat-model-a:latest") in rows


def test_republishing_is_an_UPSERT_not_a_duplicate():
    for _ in range(3):
        node_queue.upsert_model_catalog([CatalogEntry("b", "ollama", "m")])
    rows = [r for r in node_queue.fetch_model_catalog() if r.host_label == "b"]
    assert len(rows) == 1


def test_publishing_NOTHING_does_not_blank_a_box():
    """A transient probe failure must not strand every job that needs that box."""
    node_queue.upsert_model_catalog([CatalogEntry("b2", "ollama", "m")])
    assert node_queue.upsert_model_catalog([]) == 0
    assert any(r.host_label == "b2" for r in node_queue.fetch_model_catalog())


def test_the_process_catalog_reads_through_the_db():
    node_queue.upsert_model_catalog([CatalogEntry("b3", "vllm", "chat-expert-a")])
    cat = ModelCatalog(fetch=node_queue.fetch_model_catalog, ttl_s=0)
    assert cat.can_serve("b3", "vllm", "chat-expert-a") is True
    assert cat.can_serve("b3", "ollama", "chat-expert-a") is False
