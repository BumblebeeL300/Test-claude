"""Small helpers around :mod:`nimlab.core` shared by the Sentinel modules."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from nimlab.core import NIMClient, NIMConfig, Role, default_registry
from nimlab.core.errors import NIMAuthError, NIMError, NIMRateLimitError, NIMResponseError
from nimlab.core.registry import ModelRegistry

__all__ = [
    "default_models",
    "guard_model",
    "is_mock",
    "is_simulated",
    "is_transient",
    "make_client",
    "message_text",
]


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


def is_mock(client: Any) -> bool:
    flag = getattr(client, "mock", None)
    if isinstance(flag, bool):
        return flag
    return bool(getattr(getattr(client, "config", None), "mock", False))


def is_simulated(client: Any) -> bool:
    """True for :class:`~nimlab.sentinel.simulate.SimulatedClient` (it can play NemoGuard too)."""
    return bool(getattr(client, "simulated", False))


def is_transient(exc: BaseException) -> bool:
    """Whether retrying the call could help (rate limits, timeouts, 5xx, dropped connections)."""
    if isinstance(exc, NIMAuthError):
        return False
    if isinstance(exc, NIMRateLimitError):
        return True
    if isinstance(exc, NIMResponseError):
        status = exc.status_code
        return status is None or status in (408, 409, 425, 429) or status >= 500
    if isinstance(exc, NIMError):
        return True
    return isinstance(
        exc, (asyncio.TimeoutError, TimeoutError, httpx.TransportError, ConnectionError)
    )


def guard_model(registry: ModelRegistry | None = None) -> str:
    """The registry's NemoGuard content-safety model."""
    return (registry or default_registry()).pick(Role.SAFETY)


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


def make_client(*, mock: bool | None = None, config: NIMConfig | None = None) -> Any:
    """The offline simulator in mock mode, else the real :class:`NIMClient`.

    Mock mode is on when ``mock=True``, ``NIMLAB_MOCK`` is truthy, or no ``NVIDIA_API_KEY`` is set.
    """
    if config is None:
        config = NIMConfig.from_env() if mock is None else NIMConfig.from_env(mock=mock)
    elif mock is not None and config.mock != mock:
        config = config.model_copy(update={"mock": mock})
    if config.mock:
        from .simulate import SimulatedClient

        return SimulatedClient(config=config)
    return NIMClient(config)
