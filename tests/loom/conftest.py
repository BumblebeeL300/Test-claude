"""Fixtures for the Loom suite: a fake NIM client that records concurrency and can fail on cue."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

from nimlab.core import ChatResult, RerankResult, Usage
from nimlab.loom import Pipeline


class FakeClient:
    """Duck-typed NIMClient. ``delay`` simulates latency; counters expose concurrency."""

    def __init__(self, delay: float = 0.0, *, mock: bool = True) -> None:
        self.mock = mock
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls: list[tuple[str, str]] = []  # (kind, last user text)
        self.starts: list[tuple[str, float]] = []
        self.ends: list[tuple[str, float]] = []
        self.fail_first = 0  # raise on the first N chat calls
        self.reply: Callable[[str], str] = lambda text: f"reply({text[:40]})"
        self.json_reply: Any = {"answer": "ok"}
        self.closed = False

    @asynccontextmanager
    async def _track(self, label: str) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.starts.append((label, loop.time()))
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield
        finally:
            self.active -= 1
            self.ends.append((label, loop.time()))

    @staticmethod
    def _last(messages: list[dict[str, Any]]) -> str:
        return str(messages[-1]["content"])

    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        text = self._last(messages)
        self.calls.append(("chat", text))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("boom")
        async with self._track(text):
            return ChatResult(
                model=model, content=self.reply(text), usage=Usage(prompt_tokens=4, completion_tokens=6)
            )

    async def stream_chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> AsyncIterator[str]:
        text = self._last(messages)
        self.calls.append(("stream", text))
        async with self._track(text):
            for word in self.reply(text).split(" "):
                yield word + " "

    async def chat_json(self, model: str, messages: list[dict[str, Any]], schema: Any, **kw: Any) -> Any:
        text = self._last(messages)
        self.calls.append(("json", text))
        async with self._track(text):
            reply = self.json_reply(text) if callable(self.json_reply) else self.json_reply
            return schema.model_validate(json.loads(json.dumps(reply)))

    async def embed(self, model: str, texts: list[str], **kw: Any) -> list[list[float]]:
        self.calls.append(("embed", "|".join(texts)))
        async with self._track("embed"):
            # one axis per distinct first word: identical first words are duplicates
            words = sorted({t.split()[0].lower() for t in texts if t.split()})
            return [[1.0 if (t.split() or [""])[0].lower() == w else 0.0 for w in words] for t in texts]

    async def rerank(self, model: str, query: str, passages: list[str], top_n: int | None = None) -> list[RerankResult]:
        self.calls.append(("rerank", query))
        async with self._track("rerank"):
            scored = sorted(
                ((len(set(query.split()) & set(p.split())), i) for i, p in enumerate(passages)),
                key=lambda s: (-s[0], s[1]),
            )
            out = [RerankResult(index=i, score=float(s), text=passages[i]) for s, i in scored]
            return out[:top_n] if top_n else out

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake() -> FakeClient:
    return FakeClient()


def make(nodes: list[dict[str, Any]], edges: list[tuple[str, str]], name: str = "t") -> Pipeline:
    """Build a pipeline from compact node dicts and ``(from, to)`` tuples."""
    return Pipeline.from_data(
        {"name": name, "nodes": nodes, "edges": [{"from": a, "to": b} for a, b in edges]}
    )


def inp(node_id: str = "q", default: Any = "hello", type_: str = "string") -> dict[str, Any]:
    return {"id": node_id, "type": "input", "params": {"type": type_, "default": default}}


def out(node_id: str = "out", name: str | None = None) -> dict[str, Any]:
    return {"id": node_id, "type": "output", "params": {"name": name or node_id}}


def prompt(node_id: str, template: str, **extra: Any) -> dict[str, Any]:
    return {"id": node_id, "type": "prompt", "params": {"template": template}, **extra}
