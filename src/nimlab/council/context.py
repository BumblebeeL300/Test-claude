"""The :class:`RunContext` strategies use to talk to models.

Every model call a strategy makes goes through the context so that it is (a) recorded as a
:class:`~nimlab.council.records.TraceStep` with timing / tokens / reasoning, (b) announced to
observers (the live terminal UI) and (c) failure-tolerant: ``try_*`` helpers record the error in the
trace and return ``None`` instead of aborting the whole council because one model had a bad minute.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from nimlab.core.errors import NIMError, NIMResponseError
from nimlab.core.logging import get_logger
from nimlab.core.registry import ModelRegistry
from nimlab.core.results import ChatResult

from .config import CouncilConfig
from .consensus import build_report, lexical_vectors
from .errors import AllModelsFailedError
from .events import (
    CallFinished,
    CallStarted,
    ConsensusComputed,
    CouncilEvent,
    Observer,
    PhaseStarted,
)
from .records import ConsensusReport, StepType, Trace, TraceStep
from .selection import ResolvedRoles, is_reasoning_model

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

log = get_logger("nimlab.council")

SUMMARY_CHARS = 160
PROMPT_CHARS = 20_000


@dataclass
class Answer:
    """One agent's answer at some point in a strategy (``label`` is anonymous: ``Agent 2``)."""

    label: str
    model: str
    text: str
    step_index: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def agent_labels(count: int, prefix: str = "Agent") -> list[str]:
    return [f"{prefix} {i + 1}" for i in range(count)]


def clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def summarise_prompt(messages: Sequence[dict[str, Any]], limit: int = SUMMARY_CHARS) -> str:
    """Short one-line description of what was asked (the last user message)."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return clip(re.sub(r"\s+", " ", _text(msg)), limit)
    return ""


def render_prompt(messages: Sequence[dict[str, Any]], limit: int = PROMPT_CHARS) -> str:
    text = "\n\n".join(f"[{m.get('role', '?')}]\n{_text(m)}" for m in messages)
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def _text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


class RunContext:
    """Per-run state and helpers handed to a :class:`~nimlab.council.strategies.base.Strategy`."""

    def __init__(
        self,
        *,
        client: Any,
        question: str,
        config: CouncilConfig,
        roles: ResolvedRoles,
        trace: Trace,
        observers: Sequence[Observer] = (),
        registry: ModelRegistry | None = None,
    ) -> None:
        self.client = client
        self.question = question
        self.config = config
        self.roles = roles
        self.trace = trace
        self.registry = registry
        self._observers = list(observers)
        self._t0 = time.perf_counter()
        self._phase = ""
        self._sem = asyncio.Semaphore(config.max_parallel) if config.max_parallel else None
        self._live: dict[str, set[int]] = {}
        self._overlapped: set[int] = set()
        self._usage_before: dict[int, tuple[int, int]] = {}

    # ------------------------------------------------------------------ basics
    @property
    def models(self) -> list[str]:
        return self.roles.models

    @property
    def mock(self) -> bool:
        return bool(getattr(self.client, "mock", False))

    def now(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def phase(self) -> str:
        return self._phase

    def set_phase(self, name: str) -> None:
        self._phase = name
        self.emit(PhaseStarted(name))

    def emit(self, event: CouncilEvent) -> None:
        for observer in self._observers:
            try:
                observer(event)
            except Exception:  # an observer must never break a run
                log.exception("council observer failed on %s", type(event).__name__)

    def max_tokens_for(self, model: str, requested: int | None = None) -> int:
        """Token budget; reasoning models get a multiple so ``<think>`` does not eat the answer."""
        base = requested or self.config.max_tokens
        if self.registry is not None and is_reasoning_model(model, self.registry):
            return int(base * self.config.reasoning_token_multiplier)
        return base

    # ------------------------------------------------------------------ trace plumbing
    def _tokens(self, model: str) -> tuple[int, int]:
        """Cumulative (in, out) tokens the client has recorded for ``model``."""
        usage = getattr(self.client, "usage", None)
        if usage is None:
            return (0, 0)
        try:
            stats = usage.summary()["models"].get(model, {})
            return (int(stats.get("tokens_in", 0)), int(stats.get("tokens_out", 0)))
        except Exception:
            return (0, 0)

    def _begin(
        self,
        type_: StepType,
        model: str,
        label: str,
        messages: Sequence[dict[str, Any]] | None,
        temperature: float | None,
        meta: dict[str, Any] | None,
    ) -> TraceStep:
        step = TraceStep(
            index=len(self.trace.steps),
            type=type_,
            model=model,
            label=label,
            phase=self._phase,
            temperature=temperature,
            started_s=round(self.now(), 6),
            meta=dict(meta or {}),
        )
        if messages is not None:
            step.prompt_summary = summarise_prompt(messages)
            if self.config.record_prompts:
                step.prompt = render_prompt(messages)
        self.trace.steps.append(step)
        live = self._live.setdefault(model, set())
        if live:  # concurrent calls to one model make usage-delta attribution ambiguous
            self._overlapped.update(live)
            self._overlapped.add(step.index)
        live.add(step.index)
        self._usage_before[step.index] = self._tokens(model)
        self.emit(CallStarted(step.index, model, label, self._phase, type_))
        return step

    def _end(
        self,
        step: TraceStep,
        *,
        output: str = "",
        reasoning: str | None = None,
        latency_s: float | None = None,
        usage: Any = None,
        error: str | None = None,
    ) -> None:
        step.duration_s = round(self.now() - step.started_s, 6)
        step.latency_s = round(latency_s if latency_s is not None else step.duration_s, 6)
        step.output = output
        step.reasoning = reasoning or None
        step.status = "error" if error else "ok"
        step.error = error
        if usage is not None:
            step.prompt_tokens = int(usage.prompt_tokens)
            step.completion_tokens = int(usage.completion_tokens)
            step.total_tokens = int(usage.total_tokens)
        elif not error and step.index not in self._overlapped:
            before_in, before_out = self._usage_before.get(step.index, (0, 0))
            now_in, now_out = self._tokens(step.model)
            if (now_in, now_out) != (before_in, before_out):
                step.prompt_tokens = now_in - before_in
                step.completion_tokens = now_out - before_out
                step.total_tokens = step.prompt_tokens + step.completion_tokens
        self._live.get(step.model, set()).discard(step.index)
        self._usage_before.pop(step.index, None)
        self.emit(
            CallFinished(
                step.index,
                step.status == "ok",
                step.duration_s,
                step.total_tokens,
                error,
            )
        )

    async def _guarded(self, coro: Awaitable[T]) -> T:
        if self._sem is None:
            return await coro
        async with self._sem:
            return await coro

    # ------------------------------------------------------------------ model calls
    async def _chat_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        type: StepType,
        label: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> tuple[ChatResult, TraceStep]:
        temp = self.config.temperature if temperature is None else temperature
        step = self._begin(type, model, label, messages, temp, meta)
        try:
            result: ChatResult = await self._guarded(
                self.client.chat(
                    model,
                    messages,
                    temperature=temp,
                    max_tokens=self.max_tokens_for(model, max_tokens),
                )
            )
        except asyncio.CancelledError:
            self._end(step, error="cancelled")
            raise
        except Exception as exc:
            self._end(step, error=_describe(exc))
            raise
        if not result.content.strip():
            # e.g. a reasoning model that spent its whole budget thinking: useless as an answer
            self._end(
                step,
                reasoning=result.reasoning,
                latency_s=result.latency_s or None,
                usage=result.usage,
                error=f"empty response (finish_reason={result.finish_reason})",
            )
            raise NIMResponseError("model returned an empty response", model=model)
        self._end(
            step,
            output=result.content,
            reasoning=result.reasoning,
            latency_s=result.latency_s or None,
            usage=result.usage,
        )
        return result, step

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        type: StepType,
        **kw: Any,
    ) -> ChatResult:
        """Chat completion recorded in the trace. Raises :class:`NIMError` on failure."""
        result, _ = await self._chat_call(model, messages, type=type, **kw)
        return result

    async def try_chat(
        self, model: str, messages: list[dict[str, Any]], **kw: Any
    ) -> ChatResult | None:
        """Like :meth:`chat`, but a :class:`NIMError` yields ``None`` (error stays in the trace)."""
        try:
            return await self.chat(model, messages, **kw)
        except NIMError:
            return None

    async def ask(
        self, model: str, messages: list[dict[str, Any]], *, label: str, **kw: Any
    ) -> Answer | None:
        """Fan-out friendly chat: an :class:`Answer` (with its trace index) or ``None`` on error."""
        try:
            result, step = await self._chat_call(model, messages, label=label, **kw)
        except NIMError:
            return None
        return Answer(label=label, model=model, text=result.content.strip(), step_index=step.index)

    async def chat_first(
        self, models: Sequence[str], messages: list[dict[str, Any]], **kw: Any
    ) -> ChatResult:
        """Try ``models`` in order until one answers (de-duplicated)."""
        errors: list[str] = []
        for model in dict.fromkeys(models):
            try:
                return await self.chat(model, messages, **kw)
            except NIMError as exc:
                errors.append(f"{model}: {exc}")
        raise AllModelsFailedError("; ".join(errors) or "no models to call", trace=self.trace)

    async def json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[M],
        *,
        type: StepType,
        label: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> M:
        """Structured output via ``client.chat_json``; recorded in the trace. Raises on failure."""
        temp = 0.0 if temperature is None else temperature
        step = self._begin(type, model, label, messages, temp, meta)
        started = time.perf_counter()
        try:
            parsed: M = await self._guarded(
                self.client.chat_json(
                    model,
                    messages,
                    schema,
                    temperature=temp,
                    max_tokens=self.max_tokens_for(model, max_tokens),
                )
            )
        except asyncio.CancelledError:
            self._end(step, error="cancelled")
            raise
        except Exception as exc:
            self._end(step, error=_describe(exc))
            raise
        self._end(
            step,
            output=parsed.model_dump_json(indent=2),
            latency_s=time.perf_counter() - started,
        )
        return parsed

    async def try_json(
        self, model: str, messages: list[dict[str, Any]], schema: type[M], **kw: Any
    ) -> M | None:
        try:
            return await self.json(model, messages, schema, **kw)
        except NIMError:
            return None

    async def json_first(
        self, models: Sequence[str], messages: list[dict[str, Any]], schema: type[M], **kw: Any
    ) -> M:
        errors: list[str] = []
        for model in dict.fromkeys(models):
            try:
                return await self.json(model, messages, schema, **kw)
            except NIMError as exc:
                errors.append(f"{model}: {exc}")
        raise AllModelsFailedError("; ".join(errors) or "no models to call", trace=self.trace)

    # ------------------------------------------------------------------ concurrency
    async def gather(self, *coros: Coroutine[Any, Any, T]) -> list[T]:
        """``asyncio.gather`` that cancels siblings if one raises (so no orphaned requests)."""
        tasks = [asyncio.ensure_future(c) for c in coros]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def require(self, items: Sequence[Any], what: str) -> None:
        """Raise :class:`AllModelsFailedError` if ``items`` (the survivors of a stage) is empty."""
        if not items:
            raise AllModelsFailedError(f"every model failed during: {what}", trace=self.trace)

    # ------------------------------------------------------------------ consensus
    async def consensus(self, phase: str, answers: Sequence[Answer]) -> ConsensusReport | None:
        """Embed ``answers`` and record pairwise agreement; ``None`` for fewer than two answers.

        Falls back to hashed bag-of-words vectors when no embedding model is configured or the
        embedding call fails, so consensus-driven logic (early stopping, outliers) always works.
        """
        usable = [a for a in answers if a.text.strip()]
        if len(usable) < 2:
            return None
        texts = [a.text for a in usable]
        embed_model = self.roles.embed
        step_model = embed_model or "lexical"
        step = self._begin(StepType.CONSENSUS, step_model, phase, None, None, {"n": len(texts)})
        method = "lexical"
        vectors: Any = None
        if embed_model:
            try:
                vectors = await self._guarded(
                    self.client.embed(embed_model, texts, input_type="passage")
                )
                if len(vectors) != len(texts):
                    raise ValueError("embedding count mismatch")
                method = "embedding"
            except asyncio.CancelledError:
                self._end(step, error="cancelled")
                raise
            except (NIMError, ValueError) as exc:
                log.warning("embedding failed (%s); using lexical consensus", exc)
                step.meta["fallback"] = _describe(exc)
        if vectors is None:
            vectors = lexical_vectors(texts)
        report = build_report(
            vectors,
            labels=[a.label for a in usable],
            models=[a.model for a in usable],
            phase=phase,
            method=method,
            embed_model=embed_model if method == "embedding" else None,
            outlier_z=self.config.outlier_z,
            outlier_margin=self.config.outlier_margin,
        )
        outliers = ", ".join(report.outlier_labels) or "none"
        self._end(
            step,
            output=(
                f"agreement {report.agreement:.3f} across {report.size} answers "
                f"({method}); outliers: {outliers}"
            ),
        )
        self.trace.consensus.append(report)
        self.emit(ConsensusComputed(phase, report.agreement, tuple(report.outlier_labels)))
        return report


def _describe(exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    return clip(text, 500)
