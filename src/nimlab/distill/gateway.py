"""A thin wrapper around the client: bounded concurrency plus usage accounting for one run."""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypeVar

from pydantic import BaseModel

from nimlab.core import ChatResult
from nimlab.core.mock import estimate_tokens

from .costs import Tally

__all__ = ["Gateway"]

T = TypeVar("T", bound=BaseModel)


def _tokens_of(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(str(m.get("content") or "")) for m in messages)


class Gateway:
    """Every model call of a run goes through here.

    ``concurrency`` bounds in-flight requests; ``tally`` records calls and tokens per model. Token
    counts of JSON calls and embeddings are estimates (characters / 4); chat calls use the exact
    usage the API reports.
    """

    def __init__(self, client: Any, tally: Tally, concurrency: int = 8) -> None:
        self.client = client
        self.tally = tally
        self._sem = asyncio.Semaphore(max(1, concurrency))

    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        async with self._sem:
            result: ChatResult = await self.client.chat(model, messages, **kw)
        usage = result.usage
        self.tally.add(
            model,
            usage.prompt_tokens or _tokens_of(messages),
            usage.completion_tokens or estimate_tokens(result.content),
        )
        return result

    async def chat_json(
        self, model: str, messages: list[dict[str, Any]], schema: type[T], **kw: Any
    ) -> T:
        async with self._sem:
            value: T = await self.client.chat_json(model, messages, schema, **kw)
        self.tally.add(
            model,
            _tokens_of(messages),
            estimate_tokens(json.dumps(value.model_dump(), default=str)),
        )
        return value

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        async with self._sem:
            vectors: list[list[float]] = await self.client.embed(model, texts)
        self.tally.add(model, sum(estimate_tokens(t) for t in texts), 0)
        return vectors
