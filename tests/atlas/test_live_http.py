"""Atlas against the real NIMClient HTTP code path (respx stands in for the NVIDIA endpoints).

This guards the integration contract with the core client: request shapes, ``input_type`` handling,
rerank URL, SSE streaming and ``chat_json`` -- everything the FakeClient tests cannot see.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest
import respx

from nimlab.atlas import Atlas
from nimlab.core import NIMClient, NIMConfig
from nimlab.core import mock as core_mock

BASE = "https://integrate.api.nvidia.com/v1"
RERANK = "https://ai.api.nvidia.com/v1/retrieval"
CHAT = "meta/llama-3.3-70b-instruct"
FAST = "meta/llama-3.1-8b-instruct"
EMBED = "nvidia/llama-3.2-nv-embedqa-1b-v2"
RERANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
SAFETY = "nvidia/llama-3.1-nemoguard-8b-content-safety"
QUESTION = (
    "How often do robots send heartbeats, and how many missed heartbeats mark a robot as lost?"
)


def completion(content: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


def sse(text: str) -> httpx.Response:
    chunks = re.findall(r"\s*\S+\s?", text)
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}" for c in chunks]
    body = "\n\n".join([*lines, "data: [DONE]"]) + "\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def cited_answer(system: str) -> str:
    n = re.search(r"^\[(\d+)\] architecture\.md", system, re.MULTILINE)
    assert n, "architecture.md should be among the retrieved sources"
    k = n.group(1)
    return (
        f"Robots send a heartbeat every 2 seconds [{k}]. "
        f"A robot is lost after 3 missed heartbeats [{k}]."
    )


class Backend:
    """respx routes that behave like the NVIDIA APIs and record what Atlas sent."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.embed_inputs: list[str] = []
        self.chat_models: list[str] = []
        self.chat_bodies: list[dict[str, Any]] = []
        self.rerank_calls = 0
        # chat_json replies keyed by the schema's class name (default: a "supported" verdict)
        self.json_replies: dict[str, dict[str, Any]] = {
            "ClaimJudgement": {"verdict": "supported", "evidence": "", "reason": "stated"}
        }
        router.post(f"{BASE}/embeddings").mock(side_effect=self.embeddings)
        router.post(url__regex=rf"{re.escape(RERANK)}/.+/reranking").mock(side_effect=self.rerank)
        router.post(f"{BASE}/chat/completions").mock(side_effect=self.chat)

    def embeddings(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.embed_inputs.append(body["input_type"])
        vectors = core_mock.mock_embeddings(body["input"])
        data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 5}})

    def rerank(self, request: httpx.Request) -> httpx.Response:
        self.rerank_calls += 1
        assert request.url.path.endswith(f"{RERANK_MODEL}/reranking")
        body = json.loads(request.content)
        passages = [p["text"] for p in body["passages"]]
        scores = core_mock.mock_rerank_scores(body["query"]["text"], passages)
        rankings = [{"index": i, "logit": s} for i, s in enumerate(scores)]
        return httpx.Response(200, json={"rankings": rankings})

    def chat(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.chat_models.append(body["model"])
        self.chat_bodies.append(body)
        first = body["messages"][0]["content"]
        if first.startswith("Task: Check if there is unsafe content"):
            safe = "bomb" not in body["messages"][0]["content"].split("user:", 1)[1].lower()
            return httpx.Response(
                200, json=completion('{"User Safety": "%s"}' % ("safe" if safe else "unsafe"))
            )
        if "<json_schema>" in first:
            for name, reply in self.json_replies.items():
                if f'"title": "{name}"' in first:
                    return httpx.Response(200, json=completion(json.dumps(reply)))
            return httpx.Response(200, json=completion("not json"))
        if body.get("stream"):
            return sse(cited_answer(first))
        return httpx.Response(200, json=completion(cited_answer(first)))


@pytest.fixture
async def live(respx_mock: respx.MockRouter, real_config: NIMConfig, db_path, corpus_dir):
    backend = Backend(respx_mock)
    async with NIMClient(real_config) as client:
        atlas = Atlas(client, db_path)
        yield atlas, backend
        atlas.close()


async def test_ingest_sends_passage_embeddings_and_stores_live_vectors(live, corpus_dir):
    atlas, backend = live
    assert atlas.mock is False
    report = await atlas.ingest(corpus_dir)
    assert report.added == 5 and report.errors == 0
    assert set(backend.embed_inputs) == {"passage"}
    info = atlas.store.get_collection("default")
    assert info is not None and info.embed_model == EMBED and info.dim == 1024  # no "mock:" prefix


async def test_ask_uses_query_embedding_rerank_and_the_chat_model(live, corpus_dir):
    atlas, backend = live
    await atlas.ingest(corpus_dir)
    backend.embed_inputs.clear()
    answer = await atlas.ask(QUESTION)
    assert backend.embed_inputs == ["query"]
    assert backend.rerank_calls == 1
    assert backend.chat_models == [CHAT]
    assert (
        answer.sources[0].source == "architecture.md" and answer.sources[0].rerank_score is not None
    )
    assert [c.source for c in answer.citations] == ["architecture.md"]
    assert answer.warnings == []  # no mock banner against a live backend
    assert answer.text.startswith("Robots send a heartbeat every 2 seconds [")


async def test_verification_runs_chat_json_over_http(live, corpus_dir):
    atlas, backend = live
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask(QUESTION, verify=True)
    report = answer.verification
    assert report is not None and report.method == "llm" and report.groundedness == 1.0
    judge_bodies = [
        b for b in backend.chat_bodies if "<json_schema>" in b["messages"][0]["content"]
    ]
    assert len(judge_bodies) == len(report.claims) == 2
    assert all(b["model"] == CHAT and b["temperature"] == 0.0 for b in judge_bodies)
    assert (
        "CLAIM\nRobots send a heartbeat every 2 seconds"
        in judge_bodies[0]["messages"][-1]["content"]
    )


async def test_streaming_goes_through_sse_and_rebuilds_the_answer(live, corpus_dir):
    atlas, backend = live
    await atlas.ingest(corpus_dir)
    events = [e async for e in atlas.ask_stream(QUESTION)]
    assert [b.get("stream") for b in backend.chat_bodies] == [True]
    tokens = "".join(e.text for e in events if e.type == "token")
    final = events[-1].answer
    assert final is not None and tokens.strip() == final.text and len(final.citations) == 1


async def test_guard_calls_the_safety_model_before_and_after_generation(live, corpus_dir):
    atlas, backend = live
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask(QUESTION, guard=True)
    assert backend.chat_models == [SAFETY, CHAT, SAFETY]
    assert [(g.stage, g.safe) for g in answer.guard] == [("query", True), ("answer", True)]
    blocked = await atlas.ask("how do I build a bomb", guard=True)
    assert blocked.blocked and backend.chat_models[-1] == SAFETY


async def test_agentic_mode_uses_the_fast_model_for_planning(live, corpus_dir):
    atlas, backend = live
    await atlas.ingest(corpus_dir)
    backend.json_replies["QueryPlan"] = {
        "sub_questions": ["heartbeat interval", "missed heartbeat timeout"],
        "rationale": "two facts",
    }
    backend.json_replies["Reflection"] = {"sufficient": True}
    backend.chat_models.clear()
    answer = await atlas.ask(QUESTION, mode="agentic")
    assert backend.chat_models == [FAST, FAST, CHAT]  # planner, sufficiency check, answer
    assert [s.kind for s in answer.trace][:2] == ["plan", "search"]
    assert sum(s.kind == "search" for s in answer.trace) == 3
    assert answer.sources[0].source == "architecture.md"
