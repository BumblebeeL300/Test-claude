"""Test doubles for Forge: a scripted, duck-typed stand-in for ``NIMClient``."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from nimlab.core import ChatResult, ToolCall, Usage, UsageTracker

_ids = itertools.count(1)


def tc(name: str, **arguments: Any) -> ToolCall:
    """A tool call as the core client would parse it."""
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)


def calls(*tool_calls: ToolCall, content: str = "") -> ChatResult:
    """An assistant turn that requests tools."""
    return ChatResult(
        model="fake",
        content=content,
        tool_calls=list(tool_calls),
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=100, completion_tokens=20),
    )


def say(text: str) -> ChatResult:
    """A plain assistant answer (ends a tool loop)."""
    return ChatResult(
        model="fake",
        content=text,
        finish_reason="stop",
        usage=Usage(prompt_tokens=80, completion_tokens=max(1, len(text) // 4)),
    )


@dataclass
class FakeCall:
    """One recorded call to the fake client."""

    kind: str  # "chat" or "chat_json"
    index: int
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    schema: type[BaseModel] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def last_user(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return str(msg.get("content") or "")
        return ""

    @property
    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self.tools or []]


class FakeClient:
    """Plays back a script. Duck-typed like ``NIMClient`` (``chat``, ``chat_json``, ``usage``).

    ``chat`` and ``json`` are sequences whose items are consumed in call order. A chat item is a
    ``ChatResult``, a ``str`` (plain answer), an ``Exception`` (raised) or a callable
    ``(FakeCall) -> item``. A JSON item is a ``dict`` / ``BaseModel`` / ``Exception`` / callable.
    A callable ``chat`` / ``json`` script handles *every* call instead of a sequence.
    Running past the end of a script records the call in ``unexpected`` and raises.
    """

    mock = False

    def __init__(
        self,
        chat: list[Any] | Callable[[FakeCall], Any] | None = None,
        json: list[Any] | Callable[[FakeCall], Any] | None = None,
    ) -> None:
        self.usage = UsageTracker()
        self._chat = chat if callable(chat) else list(chat or [])
        self._json = json if callable(json) else list(json or [])
        self._chat_pos = 0
        self._json_pos = 0
        self.calls: list[FakeCall] = []
        self.unexpected: list[FakeCall] = []

    # ------------------------------------------------------------------ introspection

    def of_kind(self, kind: str) -> list[FakeCall]:
        return [c for c in self.calls if c.kind == kind]

    def with_model(self, model: str) -> list[FakeCall]:
        return [c for c in self.calls if c.model == model]

    # ------------------------------------------------------------------ NIMClient surface

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        call = FakeCall(
            "chat",
            len(self.of_kind("chat")),
            model,
            copy.deepcopy(messages),
            copy.deepcopy(tools),
            kwargs={"temperature": temperature, "max_tokens": max_tokens},
        )
        self.calls.append(call)
        item = self._next("chat", call)
        if isinstance(item, Exception):
            self.usage.record(model, None, 0.0, error=True)
            raise item
        result = say(item) if isinstance(item, str) else item
        result = result.model_copy(update={"model": model})
        self.usage.record(model, result.usage, 0.01)
        return result

    async def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        *,
        retries: int = 2,
        **kw: Any,
    ) -> Any:
        call = FakeCall(
            "chat_json",
            len(self.of_kind("chat_json")),
            model,
            copy.deepcopy(messages),
            schema=schema,
            kwargs=kw,
        )
        self.calls.append(call)
        item = self._next("json", call)
        if isinstance(item, Exception):
            self.usage.record(model, None, 0.0, error=True)
            raise item
        self.usage.record(model, Usage(prompt_tokens=200, completion_tokens=60), 0.01)
        if isinstance(item, BaseModel):
            return item
        return schema.model_validate(item)

    # ------------------------------------------------------------------ internals

    def _next(self, which: str, call: FakeCall) -> Any:
        script = self._chat if which == "chat" else self._json
        if callable(script):
            item = script(call)
        else:
            pos = self._chat_pos if which == "chat" else self._json_pos
            if pos >= len(script):
                self.unexpected.append(call)
                raise AssertionError(
                    f"unexpected extra {call.kind} call #{call.index} ({call.model})"
                )
            item = script[pos]
            if which == "chat":
                self._chat_pos += 1
            else:
                self._json_pos += 1
        if callable(item) and not isinstance(item, (BaseModel, Exception)):
            item = item(call)
        return item
