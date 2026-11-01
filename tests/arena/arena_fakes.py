"""Test doubles for the Arena suite: a scripted LLM client and small suite builders."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel

from nimlab.arena.suites import Suite
from nimlab.core.results import ChatResult, Usage

Answer = str | Callable[[str, str], str]


def make_suite(
    name: str, items: list[tuple[str, str, float]], *, category: str = "math_reasoning", **kw: Any
) -> Suite:
    """A numeric-graded suite from ``(item_id, prompt, expected)`` tuples."""
    return Suite.model_validate(
        {
            "name": name,
            "category": category,
            "grader": {"type": "numeric", "expected": 0},
            "items": [
                {"id": i, "prompt": p, "grader": {"type": "numeric", "expected": e}}
                for i, p, e in items
            ],
            **kw,
        }
    )


class ScriptedClient:
    """LLM double. ``answers`` maps ``prompt -> text`` or is ``fn(model, prompt) -> text``.

    Records every call, simulates time-to-first-token / per-chunk delays, raises scripted
    exceptions (``fail_first[(model, prompt)] = [exc, exc]`` fail that many times, then succeed)
    and tracks the maximum number of concurrent requests.
    """

    mock = False

    def __init__(
        self,
        answers: dict[str, str] | Callable[[str, str], str] | None = None,
        *,
        ttft: float = 0.0,
        chunk_delay: float = 0.0,
        chunks: int = 3,
        fail_first: dict[tuple[str, str], list[Exception]] | None = None,
        judge: Callable[[str, list[dict[str, Any]], type[BaseModel]], BaseModel] | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        streaming: bool = True,
    ) -> None:
        self.answers = answers or {}
        self.ttft = ttft
        self.chunk_delay = chunk_delay
        self.chunks = chunks
        self.fail_first = {k: list(v) for k, v in (fail_first or {}).items()}
        self.judge = judge
        self.embed_fn = embed_fn
        self.calls: list[tuple[str, str]] = []
        self.chat_json_calls: list[tuple[str, list[dict[str, Any]], str]] = []
        self.embed_calls: list[tuple[str, int, str]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        if not streaming:
            self.stream_chat = None  # type: ignore[assignment,method-assign]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _prompt(messages: list[dict[str, Any]]) -> str:
        return next(m["content"] for m in reversed(messages) if m["role"] == "user")

    def _text(self, model: str, prompt: str) -> str:
        if callable(self.answers):
            return self.answers(model, prompt)
        return self.answers.get(prompt, "I do not know")

    def _maybe_fail(self, model: str, prompt: str) -> None:
        queue = self.fail_first.get((model, prompt))
        if queue:
            raise queue.pop(0)

    # ------------------------------------------------------------------ client API
    async def stream_chat(
        self, model: str, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        prompt = self._prompt(messages)
        self.calls.append((model, prompt))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            self._maybe_fail(model, prompt)
            await asyncio.sleep(self.ttft)
            text = self._text(model, prompt)
            step = max(1, len(text) // self.chunks)
            for i in range(0, len(text), step):
                if i:
                    await asyncio.sleep(self.chunk_delay)
                yield text[i : i + step]
        finally:
            self.in_flight -= 1

    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        prompt = self._prompt(messages)
        self.calls.append((model, prompt))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        t0 = time.perf_counter()
        try:
            self._maybe_fail(model, prompt)
            await asyncio.sleep(self.ttft)
            text = self._text(model, prompt)
        finally:
            self.in_flight -= 1
        return ChatResult(
            model=model,
            content=text,
            usage=Usage(prompt_tokens=7, completion_tokens=11),
            latency_s=time.perf_counter() - t0,
        )

    async def chat_json(
        self, model: str, messages: list[dict[str, Any]], schema: type[BaseModel], **kw: Any
    ) -> BaseModel:
        self.chat_json_calls.append((model, messages, schema.__name__))
        if self.judge is None:
            raise AssertionError("no judge scripted")
        result = self.judge(model, messages, schema)
        if isinstance(result, Exception):
            raise result
        return result

    async def embed(self, model: str, texts: list[str], **kw: Any) -> list[list[float]]:
        self.embed_calls.append((model, len(texts), kw.get("input_type", "passage")))
        if self.embed_fn is None:
            raise AssertionError("no embed_fn scripted")
        return self.embed_fn(texts)
