"""The per-run context a mode talks to: model calls, events and simulation in one place."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from pydantic import BaseModel

from nimlab.core import Usage
from nimlab.core.mock import estimate_tokens, mock_stream_chunks

from .images import PrismImage, content_parts

__all__ = ["Observer", "Session"]

T = TypeVar("T", bound=BaseModel)
Observer = Callable[[str, dict[str, Any]], None]
Sim = Any  # a value, or a zero-argument callable producing it


def _resolve(sim: Sim) -> Any:
    return sim() if callable(sim) else sim


class Session:
    """Wraps the client for one run.

    Modes call :meth:`text` / :meth:`json` once, passing both the real prompt and a ``sim`` value
    (or factory). In simulation the model is never called, but the surrounding flow (stages, token
    events, refine loops) is identical, so the mock path exercises the same code as the real one.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        simulated: bool,
        observer: Observer | None = None,
        sim_delay: float = 0.0,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        self.client = client
        self.model = model
        self.simulated = simulated
        self.observer = observer
        self.sim_delay = max(0.0, sim_delay)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.calls = 0
        self.current_stage = ""

    # ------------------------------------------------------------------ events
    def emit(self, type_: str, data: dict[str, Any] | None = None) -> None:
        if self.observer is not None:
            self.observer(type_, data or {})

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        previous, self.current_stage = self.current_stage, name
        t0 = time.monotonic()
        self.emit("stage.start", {"stage": name})
        try:
            yield
        finally:
            self.emit("stage.end", {"stage": name, "seconds": round(time.monotonic() - t0, 3)})
            self.current_stage = previous

    def _token(self, text: str) -> None:
        self.emit("token", {"text": text, "stage": self.current_stage})

    # ------------------------------------------------------------------ simulation
    def _record(self, prompt: str, reply: str, t0: float) -> None:
        self.calls += 1
        usage = getattr(self.client, "usage", None)
        if usage is not None:
            usage.record(
                self.model,
                Usage(
                    prompt_tokens=estimate_tokens(prompt) + 800,  # a nominal image cost
                    completion_tokens=estimate_tokens(reply),
                ),
                max(time.monotonic() - t0, 0.001),
            )

    async def _pause(self, factor: float = 1.0) -> None:
        if self.sim_delay > 0:
            await asyncio.sleep(self.sim_delay * factor)

    # ------------------------------------------------------------------ model calls
    @staticmethod
    def messages(system: str, prompt: str, images: list[PrismImage]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content_parts(prompt, images)},
        ]

    async def text(
        self,
        prompt: str,
        images: list[PrismImage],
        *,
        system: str,
        sim: Sim,
        max_tokens: int | None = None,
    ) -> str:
        """Free text. Streamed (as ``token`` events) whenever someone is observing the run."""
        t0 = time.monotonic()
        if self.simulated:
            text = str(_resolve(sim))
            await self._pause(0.4)
            for chunk in mock_stream_chunks(text):
                self._token(chunk)
                await self._pause(0.004)
            self._record(prompt, text, t0)
            return text
        msgs = self.messages(system, prompt, images)
        kw = {"temperature": self.temperature, "max_tokens": max_tokens or self.max_tokens}
        self.calls += 1
        if self.observer is None:
            result = await self.client.chat(self.model, msgs, **kw)
            return str(result.content)
        parts: list[str] = []
        async for piece in self.client.stream_chat(self.model, msgs, **kw):
            parts.append(piece)
            self._token(piece)
        return "".join(parts)

    async def json(
        self,
        prompt: str,
        images: list[PrismImage],
        schema: type[T],
        *,
        system: str,
        sim: Sim,
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> T:
        """A validated ``schema`` instance (``chat_json`` re-prompts on invalid output)."""
        t0 = time.monotonic()
        if self.simulated:
            await self._pause(0.8)
            value = schema.model_validate(_resolve(sim))
            self._record(prompt, value.model_dump_json(), t0)
            return value
        self.calls += 1
        return await self.client.chat_json(
            self.model,
            self.messages(system, prompt, images),
            schema,
            retries=retries,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
