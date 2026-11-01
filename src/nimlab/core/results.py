"""Pydantic result types returned by :class:`nimlab.core.client.NIMClient`."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Usage(BaseModel):
    """Token accounting for one call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @model_validator(mode="after")
    def _fill_total(self) -> Usage:
        if not self.total_tokens:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self


class ToolCall(BaseModel):
    """A function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatResult(BaseModel):
    """Result of :meth:`NIMClient.chat`.

    ``content`` never contains ``<think>`` blocks; those are moved to ``reasoning``.
    """

    model: str
    content: str
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0

    def to_message(self) -> dict[str, Any]:
        """Assistant message (OpenAI format) ready to append to a conversation."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        return msg


class RerankResult(BaseModel):
    """One reranked passage. ``index`` refers to the position in the input list."""

    index: int
    score: float
    text: str
