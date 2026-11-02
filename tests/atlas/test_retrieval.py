"""Hybrid retrieval pipeline over the example corpus (offline mock embeddings) and its fallbacks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nimlab.atlas import Atlas, AtlasConfig
from nimlab.atlas.evaluation import load_qa, retrieval_metrics
from nimlab.atlas.retrieval import Hit, merge_hits
from nimlab.core import NIMAuthError, NIMError

from .conftest import QA_FILE, FakeClient

_, QA_ITEMS = load_qa(QA_FILE)


@pytest.mark.parametrize("item", QA_ITEMS, ids=[i.id for i in QA_ITEMS])
async def test_relevant_file_is_in_the_top_three(indexed: Atlas, item):
    sources = await indexed.retrieve(item.question, k=3)
    found = {s.source for s in sources}
    assert found & set(item.sources), f"{item.question!r} -> {sorted(found)}"


async def test_retrieval_quality_over_the_whole_qa_set(indexed: Atlas):
    recalls, rr = [], []
    for item in QA_ITEMS:
        sources = await indexed.retrieve(item.question, k=6)
        first, recall = retrieval_metrics(item.sources, sources, [6])
        recalls.append(recall[6])
        rr.append(1 / first if first else 0.0)
    assert sum(recalls) / len(recalls) >= 0.9
    assert sum(rr) / len(rr) >= 0.8


@pytest.mark.parametrize(
    ("question", "filename"),
    [
        ("how many missed heartbeats before a robot is lost", "architecture.md"),
        ("what is the API key header", "api_guide.md"),
        ("how long is a rotation of on-call", "onboarding.md"),
        ("battery above 55 degrees thermal event", "incident_runbook.md"),
        ("define takt", "glossary.md"),
    ],
)
async def test_top_hit_is_the_expected_document(indexed: Atlas, question: str, filename: str):
    sources = await indexed.retrieve(question, k=4)
    assert sources[0].source == filename


async def test_sources_are_numbered_and_carry_all_stage_scores(indexed: Atlas):
    sources = await indexed.retrieve("webhook signature HMAC", k=5)
    assert [s.n for s in sources] == list(range(1, len(sources) + 1))
    top = sources[0]
    assert top.source == "api_guide.md" and top.heading_path[-1] == "Webhooks"
    assert (
        top.dense_score is not None and top.bm25_score is not None and top.rerank_score is not None
    )
    assert "HMAC-SHA256" in top.text
    assert top.label.startswith("api_guide.md > Hive API Guide > Webhooks")
    assert top.path.endswith("api_guide.md")


async def test_k_and_context_budget_are_respected(indexed: Atlas):
    assert len(await indexed.retrieve("robots", k=2)) == 2
    tight = await indexed.retriever.retrieve("robots and tasks", "default", k=6, context_chars=900)
    assert 1 <= len(tight.hits) < 6
    assert sum(len(h.context_text) for h in tight.hits) <= 900
    tiny = await indexed.retriever.retrieve("robots", "default", k=3, context_chars=200)
    assert len(tiny.hits) >= 1 and len(tiny.hits[0].context_text) <= 200


async def test_empty_or_unknown_collection_returns_nothing(atlas: Atlas):
    assert await atlas.retrieve("anything", "nope") == []


async def test_reranker_can_be_switched_off(db_path: Path, corpus_dir: Path):
    client = FakeClient()
    a = Atlas(client, db_path)
    await a.ingest(corpus_dir)
    await a.retrieve("api key header", use_rerank=False)
    assert client.rerank_calls == 0
    sources = await a.retrieve("api key header")
    assert client.rerank_calls == 1 and sources[0].rerank_score is not None
    a.close()


async def test_dense_failure_degrades_to_bm25_only(db_path: Path, corpus_dir: Path):
    client = FakeClient()
    a = Atlas(client, db_path)
    await a.ingest(corpus_dir)
    client.embed_error = NIMError("embeddings unavailable")
    result = await a.retriever.retrieve("what is the API key header", "default", k=3)
    assert result.hits and result.hits[0].chunk.doc_name == "api_guide.md"
    assert any("BM25 only" in w for w in result.warnings)
    assert all(h.dense is None for h in result.hits)
    a.close()


async def test_auth_errors_are_not_swallowed(db_path: Path, corpus_dir: Path):
    client = FakeClient()
    a = Atlas(client, db_path)
    await a.ingest(corpus_dir)
    client.embed_error = NIMAuthError("bad key")
    with pytest.raises(NIMAuthError):
        await a.retrieve("anything")
    client.embed_error = None
    client.rerank_error = NIMAuthError("bad key")
    with pytest.raises(NIMAuthError):
        await a.retrieve("anything")
    a.close()


async def test_rerank_failure_keeps_the_fused_order(db_path: Path, corpus_dir: Path):
    client = FakeClient()
    a = Atlas(client, db_path)
    await a.ingest(corpus_dir)
    client.rerank_error = NIMError("reranker down")
    result = await a.retriever.retrieve("rate limit requests per minute", "default", k=3)
    assert result.hits and any("reranker unavailable" in w for w in result.warnings)
    assert all(h.rerank is None for h in result.hits)
    a.close()


async def _near_duplicate_atlas(client, db_path: Path, **cfg) -> Atlas:
    a = Atlas(client, db_path, config=AtlasConfig(**cfg))
    await a.ingest_bytes(
        [
            (
                "a1.md",
                b"# One\n\nBattery swap procedure: remove the battery pack, "
                b"insert a fresh battery pack, confirm the battery light.",
            ),
            (
                "a2.md",
                b"# Two\n\nBattery swap procedure: remove the battery pack, "
                b"insert a fresh battery pack, confirm the green battery light.",
            ),
            (
                "b.md",
                b"# Three\n\nCharging docks are located at the end of each aisle; "
                b"robots dock automatically when the battery is low.",
            ),
        ]
    )
    return a


async def test_mmr_replaces_a_near_duplicate_with_a_different_passage(mock_client, db_path: Path):
    a = await _near_duplicate_atlas(mock_client, db_path, mmr_lambda=0.3)
    query = "battery swap procedure battery pack"
    plain = await a.retriever.retrieve(query, "default", k=2, use_mmr=False)
    diverse = await a.retriever.retrieve(query, "default", k=2, use_mmr=True)
    assert {h.chunk.doc_name for h in plain.hits} == {"a1.md", "a2.md"}
    assert "b.md" in {h.chunk.doc_name for h in diverse.hits}
    assert diverse.hits[0].chunk.doc_name in {"a1.md", "a2.md"}  # most relevant item still first
    a.close()


async def test_mmr_lambda_one_keeps_relevance_order(mock_client, db_path: Path):
    a = await _near_duplicate_atlas(mock_client, db_path, mmr_lambda=1.0)
    plain = await a.retriever.retrieve("battery swap procedure", "default", k=3, use_mmr=False)
    mmr = await a.retriever.retrieve("battery swap procedure", "default", k=3, use_mmr=True)
    assert [h.id for h in plain.hits] == [h.id for h in mmr.hits]
    a.close()


async def test_hyde_text_is_embedded_as_a_passage_and_blended(db_path: Path, corpus_dir: Path):
    client = FakeClient()
    a = Atlas(client, db_path)
    await a.ingest(corpus_dir / "glossary.md")
    client.embed_calls.clear()
    plain = await a.retriever.query_vector("tote weight")
    blended = await a.retriever.query_vector(
        "tote weight", hyde_text="A tote holds up to 25 kg of goods"
    )
    assert ("passage", 1) in client.embed_calls and ("query", 1) in client.embed_calls
    assert np.linalg.norm(blended) == pytest.approx(1.0, abs=1e-5)
    assert not np.allclose(plain, blended)
    a.close()


async def test_merge_hits_uses_rank_agreement_across_lists(indexed: Atlas):
    result = await indexed.retriever.retrieve("heartbeat lost robot", "default", k=4)
    other = await indexed.retriever.retrieve("aisle lock epoch", "default", k=4)
    merged = merge_hits([result.hits, other.hits])
    ids = [h.id for h in merged]
    assert len(ids) == len(set(ids))
    assert set(ids) == {h.id for h in result.hits} | {h.id for h in other.hits}
    shared = {h.id for h in result.hits} & {h.id for h in other.hits}
    if shared:
        assert ids[0] in shared  # agreement across lists wins
    assert [h.score for h in merged] == sorted((h.score for h in merged), reverse=True)
    assert isinstance(merged[0], Hit) and merged[0].fused == merged[0].score
    assert len(merge_hits([result.hits, other.hits], limit=3)) == 3


async def test_sources_are_json_serialisable(indexed: Atlas):
    sources = await indexed.retrieve("what is a tote", k=2)
    payload = json.loads(json.dumps([s.model_dump(mode="json") for s in sources]))
    assert payload[0]["n"] == 1 and "label" in payload[0]
