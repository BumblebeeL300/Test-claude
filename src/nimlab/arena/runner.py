"""Concurrent, resumable benchmark runner: models x suites x items."""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from nimlab.core.errors import NIMAuthError
from nimlab.core.text import strip_think

from .graders import GradeContext, GraderError, get_grader
from .llm import estimate_tokens, is_transient, message_text, time_scale_of
from .results import ItemResult, RunResult, RunStore, SuiteInfo, utcnow
from .suites import EvalItem, Suite

__all__ = ["Completion", "ProgressFn", "RunSettings", "Runner", "build_messages"]

T = TypeVar("T")
ProgressFn = Callable[[int, int, ItemResult], None]


@dataclass
class RunSettings:
    """Knobs of a benchmark run."""

    concurrency: int = 4  # items in flight at once, across all models
    per_model_concurrency: int | None = None  # extra cap per model (None = no extra cap)
    max_retries: int = 2  # additional attempts for transient errors
    backoff_s: float = 0.5  # first backoff; doubles each retry, with jitter
    item_timeout_s: float = 120.0  # wall-clock limit for one model call
    judge_model: str | None = None
    save_interval_s: float = 1.0  # minimum time between checkpoint writes
    stream: bool = True  # stream to measure time-to-first-token (falls back to non-streaming)

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")


@dataclass
class Completion:
    """A model reply plus its performance measurements (in reported seconds)."""

    text: str
    latency_s: float
    ttft_s: float | None
    tokens_in: int
    tokens_out: int
    tokens_per_s: float | None
    estimated: bool


def build_messages(suite: Suite, item: EvalItem) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = suite.system_for(item)
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": item.prompt})
    return messages


class Runner:
    """Runs every (model, suite, item) combination with bounded concurrency.

    Each model call is streamed to measure time-to-first-token and decode throughput, retried on
    transient errors, graded by the item's grader and checkpointed to the :class:`RunStore`
    (when one is given) so an interrupted run can be resumed with ``resume=``.
    """

    def __init__(
        self,
        client: Any,
        *,
        settings: RunSettings | None = None,
        store: RunStore | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or RunSettings()
        self.store = store
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._last_save = 0.0

    # ------------------------------------------------------------------ public
    async def run(
        self,
        models: list[str],
        suites: list[Suite],
        *,
        resume: RunResult | None = None,
        on_progress: ProgressFn | None = None,
    ) -> RunResult:
        models = list(dict.fromkeys(models))
        if not models:
            raise ValueError("at least one model is required")
        if not suites:
            raise ValueError("at least one suite is required")
        if hasattr(self.client, "register_suites"):
            self.client.register_suites(suites)

        run = resume or (self.store.new_run() if self.store else RunResult.new())
        run.mock = bool(getattr(self.client, "mock", False))
        run.judge_model = self.settings.judge_model or run.judge_model
        run.models = list(dict.fromkeys([*run.models, *models]))
        run.status = "running"
        run.config.update(
            concurrency=self.settings.concurrency,
            max_retries=self.settings.max_retries,
            streaming=self.settings.stream,
        )
        for s in suites:
            run.suites[s.name] = SuiteInfo(
                name=s.name,
                title=s.title,
                category=s.category,
                open_ended=s.open_ended,
                n_items=len(s.items),
            )

        done = {r.key: r for r in run.items if r.scored}
        work: list[tuple[str, Suite, EvalItem, str]] = []
        for suite in suites:
            for item in suite.items:
                digest = suite.item_hash(item)
                for model in models:
                    prev = done.get((model, suite.name, item.id))
                    if prev is None or prev.item_hash != digest:
                        work.append((model, suite, item, digest))

        total = len(work)
        counter = 0
        position = {r.key: i for i, r in enumerate(run.items)}
        global_sem = asyncio.Semaphore(self.settings.concurrency)
        model_sems = {
            m: asyncio.Semaphore(self.settings.per_model_concurrency)
            for m in models
            if self.settings.per_model_concurrency
        }

        async def one(model: str, suite: Suite, item: EvalItem, digest: str) -> None:
            nonlocal counter
            async with contextlib.AsyncExitStack() as stack:
                await stack.enter_async_context(global_sem)
                if model in model_sems:
                    await stack.enter_async_context(model_sems[model])
                result = await self.evaluate(model, suite, item, digest)
            slot = position.get(result.key)
            if slot is None:  # first result for this key (replacing keeps the list order stable)
                position[result.key] = len(run.items)
                run.items.append(result)
            else:
                run.items[slot] = result
            counter += 1
            self._checkpoint(run)
            if on_progress:
                on_progress(counter, total, result)

        tasks = [asyncio.ensure_future(one(*w)) for w in work]
        try:
            await _gather_fail_fast(tasks)
            run.status = "complete" if not any(not r.scored for r in run.items) else "partial"
        except BaseException:
            run.status = "partial"
            raise
        finally:
            if self.store:
                self.store.save(run)
        return run

    async def evaluate(
        self, model: str, suite: Suite, item: EvalItem, digest: str | None = None
    ) -> ItemResult:
        """Query ``model`` on one item and grade the answer. Never raises except on auth errors."""
        spec = suite.grader_for(item)
        result = ItemResult(
            model=model,
            suite=suite.name,
            item_id=item.id,
            item_hash=digest or suite.item_hash(item),
            weight=item.weight,
            grader=spec.type,
        )
        messages = build_messages(suite, item)
        try:
            completion, attempts = await self._retrying(
                lambda: self._complete(
                    model, messages, suite.temperature, suite.max_tokens_for(item)
                )
            )
        except NIMAuthError:
            raise
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.attempts = self.settings.max_retries + 1 if is_transient(exc) else 1
            result.finished_at = utcnow().isoformat(timespec="seconds")
            return result

        result.attempts = attempts
        result.output = completion.text
        result.latency_s = completion.latency_s
        result.ttft_s = completion.ttft_s
        result.tokens_in = completion.tokens_in
        result.tokens_out = completion.tokens_out
        result.tokens_per_s = completion.tokens_per_s
        result.usage_estimated = completion.estimated

        ctx = GradeContext(
            output=completion.text,
            params=spec.params,
            prompt=item.prompt,
            reference=item.reference,
            item_id=item.id,
            suite=suite.name,
            client=self.client,
            judge_model=self.settings.judge_model,
        )
        try:
            grade, _ = await self._retrying(lambda: get_grader(spec.type).grade(ctx))
        except NIMAuthError:
            raise
        except Exception as exc:
            result.error = f"grader {spec.type}: {exc}"
        else:
            result.score = grade.score
            result.passed = grade.passed
            result.detail = grade.detail
        result.finished_at = utcnow().isoformat(timespec="seconds")
        return result

    # ------------------------------------------------------------------ internals
    async def _retrying(self, make_call: Callable[[], Awaitable[T]]) -> tuple[T, int]:
        attempts = 0
        while True:
            attempts += 1
            try:
                return await make_call(), attempts
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                cause = exc.__cause__ if isinstance(exc, GraderError) and exc.__cause__ else exc
                if attempts > self.settings.max_retries or not is_transient(cause):
                    raise
                delay = self.settings.backoff_s * (2 ** (attempts - 1))
                await self._sleep(delay * (0.5 + random.random() * 0.5))

    async def _complete(
        self, model: str, messages: list[dict[str, Any]], temperature: float, max_tokens: int
    ) -> Completion:
        client = self.client
        scale = time_scale_of(client)
        tokens_in = sum(estimate_tokens(message_text(m)) for m in messages)
        t0 = time.perf_counter()
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}

        if self.settings.stream and getattr(client, "stream_chat", None) is not None:
            parts: list[str] = []
            first: float | None = None

            async def consume() -> None:
                nonlocal first
                async for chunk in client.stream_chat(model, messages, **kwargs):
                    if chunk:
                        if first is None:
                            first = time.perf_counter() - t0
                        parts.append(chunk)

            await asyncio.wait_for(consume(), timeout=self.settings.item_timeout_s * scale)
            total = (time.perf_counter() - t0) / scale
            raw = "".join(parts)
            text = strip_think(raw)[0]
            tokens_out = estimate_tokens(raw)
            ttft = first / scale if first is not None else None
            decode = max(total - (ttft or 0.0), total * 0.1)
            tps = tokens_out / decode if tokens_out and decode > 0 else None
            return Completion(text.strip(), total, ttft, tokens_in, tokens_out, tps, True)

        reply = await asyncio.wait_for(
            client.chat(model, messages, **kwargs), timeout=self.settings.item_timeout_s * scale
        )
        total = (time.perf_counter() - t0) / scale
        text = strip_think(reply.content or "")[0].strip()
        usage = getattr(reply, "usage", None)
        out_tokens = getattr(usage, "completion_tokens", 0) or estimate_tokens(text)
        in_tokens = getattr(usage, "prompt_tokens", 0) or tokens_in
        estimated = not (usage and getattr(usage, "completion_tokens", 0))
        tps = out_tokens / total if out_tokens and total > 0 else None
        return Completion(text, total, None, in_tokens, out_tokens, tps, estimated)

    def _checkpoint(self, run: RunResult) -> None:
        if not self.store:
            return
        now = time.monotonic()
        if now - self._last_save >= self.settings.save_interval_s:
            self._last_save = now
            self.store.save(run)


async def _gather_fail_fast(tasks: Sequence[asyncio.Future[Any]]) -> None:
    """Await all tasks; on the first exception cancel the rest and re-raise (py3.10-safe)."""
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
