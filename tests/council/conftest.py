"""Fixtures for the Council tests: a scripted, duck-typed client plus small helpers.

``make_client(...)`` builds a :class:`FakeClient` whose behaviour is described by plain functions:

* ``chat(model, messages, kind) -> str | Exception | ChatResult``
* ``json(model, messages, schema, kind) -> BaseModel | dict | Exception``
* ``embed(text) -> list[float]``

``kind`` classifies the prompt (``"proposer"``, ``"moa"``, ``"critique"``, ...) so scripts do not
have to parse prompts themselves.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from nimlab.core.errors import NIMError
from nimlab.core.results import ChatResult, Usage
from nimlab.core.usage import UsageTracker
from nimlab.council import prompts

REWARD_MODEL = "nvidia/llama-3.1-nemotron-70b-reward"
EMBED_MODEL = "nvidia/embed-test"

_KINDS = {
    prompts.DEFAULT_SYSTEM: "proposer",
    prompts.MOA_AGGREGATE_SYSTEM: "moa",
    prompts.DEBATE_CRITIQUE_SYSTEM: "debate_critique",
    prompts.DEBATE_REVISE_SYSTEM: "debate_revise",
    prompts.DEBATE_JUDGE_SYSTEM: "judge",
    prompts.RUBRIC_SYSTEM: "rubric",
    prompts.DRAFT_SYSTEM: "draft",
    prompts.CRITIC_SYSTEM: "critic",
    prompts.REVISER_SYSTEM: "reviser",
}


def kind_of(messages: list[dict[str, Any]]) -> str:
    """Classify a council prompt by its system message (reward calls have none)."""
    if messages and messages[0].get("role") == "system":
        return _KINDS.get(messages[0]["content"], "other")
    return "reward"


@dataclass
class Call:
    api: str  # chat | json | embed
    model: str
    kind: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    texts: list[str] = field(default_factory=list)

    @property
    def user(self) -> str:
        return next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), "")


class FakeClient:
    """A scripted stand-in for :class:`nimlab.core.NIMClient` (no network, no mock backend)."""

    def __init__(
        self,
        *,
        chat: Callable[[str, list[dict[str, Any]], str], Any] | None = None,
        json: Callable[[str, list[dict[str, Any]], type[BaseModel], str], Any] | None = None,
        embed: Callable[[str], list[float]] | None = None,
        mock: bool = False,
        delay: float = 0.0,
    ) -> None:
        self._chat = chat or (lambda model, messages, kind: f"{kind} answer from {model}")
        self._json = json
        self._embed = embed or _keyword_embedding
        self.mock = mock
        self.config = SimpleNamespace(mock=mock)
        self.usage = UsageTracker()
        self.delay = delay
        self.calls: list[Call] = []
        self.in_flight = 0
        self.max_in_flight = 0

    # -- helpers for assertions
    def calls_of(self, api: str | None = None, kind: str | None = None) -> list[Call]:
        return [
            c
            for c in self.calls
            if (api is None or c.api == api) and (kind is None or c.kind == kind)
        ]

    async def _enter(self) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self.delay)

    # -- NIMClient surface
    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        kind = kind_of(messages)
        self.calls.append(Call("chat", model, kind, messages, kw))
        await self._enter()
        try:
            out = self._chat(model, messages, kind)
            if isinstance(out, Exception):
                self.usage.record(model, None, 0.0, error=True)
                raise out
            if isinstance(out, ChatResult):
                self.usage.record(model, out.usage, out.latency_s)
                return out
            text = str(out)
            usage = Usage(prompt_tokens=10, completion_tokens=max(1, len(text.split())))
            self.usage.record(model, usage, 0.01)
            return ChatResult(model=model, content=text, usage=usage, latency_s=0.01)
        finally:
            self.in_flight -= 1

    async def chat_json(
        self, model: str, messages: list[dict[str, Any]], schema: type[BaseModel], **kw: Any
    ) -> BaseModel:
        kind = kind_of(messages)
        self.calls.append(Call("json", model, kind, messages, kw))
        await self._enter()
        try:
            if self._json is None:
                raise AssertionError(f"unexpected chat_json call ({kind})")
            out = self._json(model, messages, schema, kind)
            if isinstance(out, Exception):
                self.usage.record(model, None, 0.0, error=True)
                raise out
            self.usage.record(model, Usage(prompt_tokens=20, completion_tokens=5), 0.01)
            return out if isinstance(out, schema) else schema.model_validate(out)
        finally:
            self.in_flight -= 1

    async def embed(self, model: str, texts: list[str], *, input_type: str = "passage") -> Any:
        self.calls.append(
            Call("embed", model, "embed", kwargs={"input_type": input_type}, texts=list(texts))
        )
        await self._enter()
        try:
            self.usage.record(model, Usage(prompt_tokens=sum(len(t.split()) for t in texts)), 0.01)
            return [self._embed(t) for t in texts]
        finally:
            self.in_flight -= 1


def _keyword_embedding(text: str) -> list[float]:
    """One-hot-ish topic vectors: texts mentioning ALPHA / BETA / GAMMA point in different directions."""
    low = text.lower()
    vec = [float("alpha" in low), float("beta" in low), float("gamma" in low), 0.05]
    return vec


@pytest.fixture
def make_client() -> Callable[..., FakeClient]:
    return FakeClient


@pytest.fixture
def kinds() -> Callable[[list[dict[str, Any]]], str]:
    return kind_of


@pytest.fixture
def nim_error() -> Callable[[str], NIMError]:
    return lambda msg="boom": NIMError(msg)


PANEL = ["org-a/model-a", "org-b/model-b", "org-c/model-c"]


@pytest.fixture
def panel() -> list[str]:
    return list(PANEL)
