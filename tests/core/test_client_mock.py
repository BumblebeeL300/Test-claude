"""NIMClient on the offline mock backend: every public method works without network."""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from nimlab.core import (
    ChatResult,
    NIMClient,
    NIMConfig,
    NIMJSONError,
    RerankResult,
    cosine,
    default_registry,
    run_agent,
    tool,
)
from nimlab.core.mock import JSON_SCHEMA_CLOSE, JSON_SCHEMA_OPEN

CHAT = "nvidia/nemotron-3-super-120b-a12b"
USER = [{"role": "user", "content": "Compare Redis and Memcached for session storage"}]


class Severity(str, Enum):
    LOW = "low"
    HIGH = "high"


class Finding(BaseModel):
    title: str
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    tags: list[str]


class Review(BaseModel):
    summary: str
    verdict: Literal["approve", "reject"]
    score: int = Field(ge=1, le=10)
    findings: list[Finding]
    follow_up: Finding | None = None


async def test_chat(mock_client: NIMClient) -> None:
    r = await mock_client.chat(CHAT, USER)
    assert isinstance(r, ChatResult)
    assert r.model == CHAT
    assert "redis" in r.content.lower()
    assert r.reasoning is None and r.tool_calls == []
    assert r.finish_reason == "stop"
    assert r.usage.prompt_tokens > 0 and r.usage.total_tokens == (
        r.usage.prompt_tokens + r.usage.completion_tokens
    )
    assert r.latency_s >= 0
    assert (await mock_client.chat(CHAT, USER)).content == r.content  # deterministic


async def test_chat_reasoning_model_strips_think(mock_client: NIMClient) -> None:
    r = await mock_client.chat("nvidia/nemotron-3-ultra-550b-a55b", USER)
    assert "<think>" not in r.content and "</think>" not in r.content
    assert r.reasoning and "redis" in r.reasoning.lower()
    assert r.content


async def test_auto_mock_without_key() -> None:
    async with NIMClient() as client:  # no env key -> mock
        assert client.config.mock
        assert (await client.chat(CHAT, USER)).content


async def test_chat_json_nested_schema(mock_client: NIMClient) -> None:
    review = await mock_client.chat_json(CHAT, USER, Review)
    assert isinstance(review, Review)
    assert review.verdict in {"approve", "reject"} and 1 <= review.score <= 10
    assert len(review.findings) >= 1
    assert all(isinstance(f.severity, Severity) and 0 <= f.confidence <= 1 for f in review.findings)
    assert review.follow_up is not None  # optional fields are exercised too


async def test_chat_json_with_reasoning_model(mock_client: NIMClient) -> None:
    review = await mock_client.chat_json("nvidia/nemotron-3-ultra-550b-a55b", USER, Review)
    assert isinstance(review, Review)


async def test_chat_json_forwards_kwargs_and_merges_system_prompt(mock_client: NIMClient) -> None:
    msgs = [{"role": "system", "content": "You are a strict reviewer."}, *USER]
    out = await mock_client.chat_json(CHAT, msgs, Review, temperature=0.0, max_tokens=64)
    assert isinstance(out, Review)  # JSON is never truncated by max_tokens
    assert msgs[0]["content"] == "You are a strict reviewer."  # caller's list untouched


def test_schema_prompt_is_merged_into_first_system_message() -> None:
    merged = NIMClient._with_schema_prompt(
        [{"role": "system", "content": "Be strict."}, {"role": "user", "content": "hi"}], Review
    )
    assert [m["role"] for m in merged] == ["system", "user"]
    assert merged[0]["content"].startswith("Be strict.")
    assert JSON_SCHEMA_OPEN in merged[0]["content"] and JSON_SCHEMA_CLOSE in merged[0]["content"]
    assert '"findings"' in merged[0]["content"]
    inserted = NIMClient._with_schema_prompt([{"role": "user", "content": "hi"}], Review)
    assert [m["role"] for m in inserted] == ["system", "user"]


async def test_chat_json_retries_then_raises(
    mock_client: NIMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[dict[str, str]]] = []

    async def bad_chat(model: str, messages: list[dict[str, str]], **kw: object) -> ChatResult:
        calls.append(messages)
        return ChatResult(model=model, content="definitely not json")

    monkeypatch.setattr(mock_client, "chat", bad_chat)
    with pytest.raises(NIMJSONError) as info:
        await mock_client.chat_json(CHAT, USER, Review, retries=2)
    assert len(calls) == 3
    assert info.value.raw == "definitely not json"
    assert "not parseable JSON" in calls[1][-1]["content"]  # error fed back for the re-prompt


async def test_stream_chat_yields_word_chunks(mock_client: NIMClient) -> None:
    chunks = [c async for c in mock_client.stream_chat(CHAT, USER)]
    full = (await mock_client.chat(CHAT, USER)).content
    assert len(chunks) > 5
    assert "".join(chunks) == full


async def test_stream_chat_filters_think(mock_client: NIMClient) -> None:
    text = "".join([c async for c in mock_client.stream_chat("nvidia/nemotron-3-ultra-550b-a55b", USER)])
    assert "<think>" not in text and "think>" not in text
    assert text == (await mock_client.chat("nvidia/nemotron-3-ultra-550b-a55b", USER)).content


async def test_embed(mock_client: NIMClient) -> None:
    model = default_registry().pick("embed")
    docs = [
        "Redis is an in-memory data store used for caching",
        "Memcached is a simple distributed memory cache",
        "Tropical rainforests host remarkable biodiversity",
    ]
    vecs = await mock_client.embed(model, docs)
    q = (await mock_client.embed(model, ["in-memory caching store"], input_type="query"))[0]
    assert len(vecs) == 3 and all(len(v) == 1024 for v in vecs)
    ranked = sorted(range(3), key=lambda i: cosine(q, vecs[i]), reverse=True)
    assert ranked[0] == 0 and ranked[-1] == 2


async def test_embed_large_batch_and_empty(mock_client: NIMClient) -> None:
    vecs = await mock_client.embed("m", [f"text {i}" for i in range(100)])
    assert len(vecs) == 100
    assert await mock_client.embed("m", []) == []


async def test_rerank(mock_client: NIMClient) -> None:
    passages = ["bananas are yellow", "redis caching strategy guide", "caching with memcached"]
    results = await mock_client.rerank("m", "redis caching", passages)
    assert all(isinstance(r, RerankResult) for r in results)
    assert [r.index for r in results] == [1, 2, 0]
    assert results[0].text == passages[1]
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    top1 = await mock_client.rerank("m", "redis caching", passages, top_n=1)
    assert len(top1) == 1 and top1[0].index == 1
    assert await mock_client.rerank("m", "q", []) == []


async def test_list_models_mock(mock_client: NIMClient) -> None:
    ids = await mock_client.list_models()
    assert "nvidia/nemotron-3-super-120b-a12b" in ids and ids == sorted(ids)


async def test_usage_tracking(mock_client: NIMClient) -> None:
    await mock_client.chat(CHAT, USER)
    await mock_client.chat(CHAT, USER)
    await mock_client.embed("emb", ["a", "b"])
    await mock_client.rerank("rr", "a", ["a b"])
    s = mock_client.usage.summary()
    assert s["models"][CHAT]["calls"] == 2
    assert s["models"][CHAT]["tokens_in"] > 0 and s["models"][CHAT]["tokens_out"] > 0
    assert s["models"]["emb"]["calls"] == 1 and s["models"]["rr"]["calls"] == 1
    assert s["total"]["calls"] == 4 and s["total"]["errors"] == 0
    assert mock_client.usage.rich_table() is not None


async def test_tool_loop_end_to_end(mock_client: NIMClient) -> None:
    seen: list[dict[str, object]] = []

    @tool
    def lookup(query: str, limit: int = 3) -> dict[str, object]:
        """Search the knowledge base.

        Args:
            query: What to search for.
            limit: Maximum number of hits.
        """
        seen.append({"query": query, "limit": limit})
        return {"hits": [f"doc about {query}"] * limit}

    run = await run_agent(
        mock_client,
        CHAT,
        [{"role": "user", "content": "Find docs about vector databases"}],
        [lookup],
    )
    assert len(seen) == 1 and seen[0]["query"] == "Find docs about vector databases"
    assert run.final.startswith("Based on the tool results")
    assert "vector databases" in run.final
    assert [m["role"] for m in run.messages] == ["user", "assistant", "tool", "assistant"]
    assert run.messages[1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert run.messages[2]["tool_call_id"] == run.messages[1]["tool_calls"][0]["id"]
    assert not run.truncated and len(run.steps) == 2


class _CountingSemaphore:
    """Wraps a real semaphore and records the peak number of concurrent holders."""

    def __init__(self, inner: asyncio.Semaphore) -> None:
        self.inner = inner
        self.current = 0
        self.peak = 0

    async def __aenter__(self) -> None:
        await self.inner.acquire()
        self.current += 1
        self.peak = max(self.peak, self.current)

    async def __aexit__(self, *exc: object) -> None:
        self.current -= 1
        self.inner.release()


async def test_concurrent_calls_respect_max_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMLAB_MOCK_LATENCY", "0.01")
    async with NIMClient(NIMConfig(mock=True, max_concurrency=2)) as client:
        counting = _CountingSemaphore(asyncio.Semaphore(2))
        monkeypatch.setattr(client, "_semaphore", lambda: counting)
        results = await asyncio.gather(*(client.chat(CHAT, USER) for _ in range(12)))
    assert len(results) == 12
    assert counting.peak == 2 and counting.current == 0


def test_client_can_be_reused_across_event_loops() -> None:
    client = NIMClient(NIMConfig(mock=True))

    async def once() -> str:
        return (await client.chat(CHAT, USER)).content

    assert asyncio.run(once()) == asyncio.run(once())


async def test_mock_latency_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMLAB_MOCK_LATENCY", "0.02")
    async with NIMClient(NIMConfig(mock=True)) as client:
        r = await client.chat(CHAT, USER)
    assert r.latency_s >= 0.015


async def test_response_json_roundtrip(mock_client: NIMClient) -> None:
    r = await mock_client.chat(CHAT, USER)
    assert json.loads(r.model_dump_json())["content"] == r.content
