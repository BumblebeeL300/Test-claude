"""Runtime objects shared by the engine and the node implementations."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schema import Node

__all__ = ["NodeContext", "NodeResult", "RunEnv", "estimate_tokens"]

Emit = Callable[[dict[str, Any]], None]


def estimate_tokens(text: str) -> int:
    """Rough token count (4 characters per token) for calls that do not report usage."""
    return max(1, -(-len(text) // 4)) if text else 0


@dataclass
class NodeResult:
    """What a node returns: values by output port (absent port = nothing emitted there)."""

    outputs: dict[str, Any]
    tokens: int = 0
    model: str | None = None


@dataclass
class RunEnv:
    """Everything a run shares: the client, pipeline inputs, switches and the event sink."""

    client: Any
    input_values: Mapping[str, Any] = field(default_factory=dict)
    emit: Emit | None = None
    stream: bool = False
    http_allow: Sequence[str] = ()
    http_client: Any | None = None
    atlas_db: str | os.PathLike[str] | None = None
    t0: float = field(default_factory=time.monotonic)

    def event(self, type_: str, **data: Any) -> None:
        if self.emit is not None:
            data["type"] = type_
            data["t"] = int((time.monotonic() - self.t0) * 1000)
            self.emit(data)


class NodeContext:
    """One node execution: its params, resolved inputs and helpers."""

    def __init__(
        self,
        node: Node,
        params: dict[str, Any],
        inputs: dict[str, Any],
        env: RunEnv,
        *,
        quiet: bool = False,
    ) -> None:
        self.node = node
        self.params = params
        self.inputs = inputs
        self.env = env
        self._quiet = quiet
        self._buf: list[str] = []
        self._buf_chars = 0
        self._last_flush = 0.0

    @property
    def client(self) -> Any:
        return self.env.client

    @property
    def streaming(self) -> bool:
        return self.env.stream and not self._quiet and self.env.emit is not None

    def scope(self) -> dict[str, Any]:
        """Template/expression variables: the node's defaults overlaid with its inputs."""
        return {**self.node.defaults, **self.inputs}

    async def token(self, piece: str) -> None:
        """Queue streamed text; ``node.token`` events are coalesced (about 25 per second)."""
        if not self.streaming or not piece:
            return
        self._buf.append(piece)
        self._buf_chars += len(piece)
        now = time.monotonic()
        if self._buf_chars >= 64 or now - self._last_flush >= 0.04:
            self.flush_tokens()
            await asyncio.sleep(0)

    def flush_tokens(self) -> None:
        if self._buf:
            self.env.event("node.token", node=self.node.id, text="".join(self._buf))
            self._buf.clear()
            self._buf_chars = 0
        self._last_flush = time.monotonic()
