"""Thin helpers around :mod:`nimlab.core` shared by the Arena modules."""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from nimlab.core import NIMClient, NIMConfig, Role, default_registry
from nimlab.core.errors import NIMAuthError, NIMError, NIMRateLimitError, NIMResponseError
from nimlab.core.registry import ModelRegistry
from nimlab.core.results import ChatResult

__all__ = [
    "LLMClient",
    "default_models",
    "estimate_tokens",
    "is_transient",
    "make_client",
    "message_text",
    "pick",
    "time_scale_of",
]


@runtime_checkable
class LLMClient(Protocol):
    """The slice of :class:`nimlab.core.NIMClient` that Arena relies on (also met by test fakes)."""

    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult: ...

    def stream_chat(
        self, model: str, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]: ...

    async def chat_json(
        self, model: str, messages: list[dict[str, Any]], schema: type[BaseModel], **kw: Any
    ) -> BaseModel: ...

    async def embed(self, model: str, texts: list[str], **kw: Any) -> list[list[float]]: ...


def estimate_tokens(text: str) -> int:
    """Rough token count (about four characters per token), same rule as the core mock."""
    return max(1, math.ceil(len(text) / 4)) if text else 0


def message_text(message: dict[str, Any]) -> str:
    """Plain text of a chat message whose content is a string or a list of content parts."""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and p.get("type", "text") == "text"
        )
    return str(content)


def time_scale_of(client: Any) -> float:
    """Wall-clock seconds per reported second (``< 1`` only for the offline simulator)."""
    scale = getattr(client, "time_scale", 1.0)
    return float(scale) if isinstance(scale, (int, float)) and scale > 0 else 1.0


def is_transient(exc: BaseException) -> bool:
    """Whether retrying the call could help (rate limits, timeouts, 5xx, dropped connections)."""
    import asyncio

    import httpx

    if isinstance(exc, NIMAuthError):
        return False
    if isinstance(exc, NIMRateLimitError):
        return True
    if isinstance(exc, NIMResponseError):
        status = exc.status_code
        return status is None or status in (408, 409, 425, 429) or status >= 500
    if isinstance(exc, NIMError):  # timeouts and connection errors
        return True
    return isinstance(exc, (asyncio.TimeoutError, httpx.TransportError, ConnectionError))


def pick(registry: ModelRegistry | None, role: Role) -> str:
    return (registry or default_registry()).pick(role)


def default_models(registry: ModelRegistry | None = None, limit: int = 4) -> list[str]:
    """A diverse default line-up: the registry's chat, reasoning, coder and fast picks."""
    reg = registry or default_registry()
    out: list[str] = []
    for role in (Role.CHAT, Role.REASONING, Role.CODER, Role.FAST):
        try:
            model = reg.pick(role)
        except KeyError:
            continue
        if model not in out:
            out.append(model)
    return out[:limit]


def make_client(
    *, mock: bool | None = None, config: NIMConfig | None = None, suites: Any = None
) -> Any:
    """A client for Arena: the offline simulator in mock mode, else the real :class:`NIMClient`.

    Mock mode is on when ``mock=True``, ``NIMLAB_MOCK`` is truthy, or no ``NVIDIA_API_KEY`` is set.
    """
    if config is None:
        config = NIMConfig.from_env() if mock is None else NIMConfig.from_env(mock=mock)
    elif mock is not None and config.mock != mock:
        config = config.model_copy(update={"mock": mock})
    if config.mock:
        from .simulate import SimulatedClient

        return SimulatedClient(suites=suites, config=config)
    return NIMClient(config)


_WORD = re.compile(r"[a-z0-9]+")


def word_set(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))
