"""The :class:`Council` facade: pick a strategy, wire up models, run, return result + trace."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from nimlab.core.errors import NIMError
from nimlab.core.registry import ModelRegistry, default_registry

from . import strategies as _builtin  # noqa: F401  (registers the built-in strategies)
from .config import CouncilConfig
from .context import RunContext
from .errors import CouncilError
from .events import Observer, RunFinished, RunStarted
from .records import CouncilResult, ModelUsage, Trace
from .selection import resolve
from .strategies.base import Strategy, get_strategy, list_strategies


class Council:
    """Orchestrates several NIM models to answer one question better than any of them alone.

    >>> council = Council(client)                         # doctest: +SKIP
    >>> result = await council.run("Why is the sky blue?", strategy="debate", rounds=2)
    >>> result.answer, result.trace.save("trace.json")    # doctest: +SKIP

    Parameters
    ----------
    client:
        A :class:`~nimlab.core.NIMClient` (or duck-typed equivalent). If omitted, ``run`` opens a
        default client (mock mode automatically when ``NVIDIA_API_KEY`` is unset).
    registry:
        Model registry used for default panel / aggregator / judge / reward / embedding models.
    config:
        Base :class:`CouncilConfig`; per-call keyword overrides are merged on top.
    observers:
        Callables receiving :mod:`~nimlab.council.events` (e.g. :class:`CouncilLive`).
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        registry: ModelRegistry | None = None,
        config: CouncilConfig | None = None,
        observers: Sequence[Observer] = (),
        **config_overrides: Any,
    ) -> None:
        self.client = client
        self.registry = registry or default_registry()
        base = config or CouncilConfig()
        self.config = base.merged(**config_overrides) if config_overrides else base
        self.observers: list[Observer] = list(observers)

    # ------------------------------------------------------------------ discovery
    @staticmethod
    def strategies() -> list[Strategy]:
        """All registered strategies, built-ins first."""
        return list_strategies()

    # ------------------------------------------------------------------ running
    async def run(
        self,
        question: str,
        strategy: str | Strategy = "moa",
        *,
        models: Sequence[str] | None = None,
        observers: Sequence[Observer] = (),
        live: bool = False,
        **overrides: Any,
    ) -> CouncilResult:
        """Run ``strategy`` on ``question``.

        ``overrides`` are :class:`CouncilConfig` fields (``rounds=``, ``layers=``, ``n=``,
        ``aggregator=``, ``judge=``, ``max_iterations=``, ...). ``live=True`` shows the rich live
        view while the council works. Raises :class:`CouncilError` (with the partial ``.trace``)
        if a stage fails entirely.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        strat = get_strategy(strategy) if isinstance(strategy, str) else strategy
        cfg = self.config.merged(models=list(models) if models else None, **overrides)
        all_observers = [*self.observers, *observers]

        if live:
            from .live import CouncilLive

            with CouncilLive() as view:
                return await self._run_with_client(question, strat, cfg, [*all_observers, view])
        return await self._run_with_client(question, strat, cfg, all_observers)

    def run_sync(self, question: str, strategy: str | Strategy = "moa", **kw: Any) -> CouncilResult:
        """Blocking convenience wrapper around :meth:`run` for scripts and notebooks."""
        return asyncio.run(self.run(question, strategy, **kw))

    async def _run_with_client(
        self, question: str, strat: Strategy, cfg: CouncilConfig, observers: list[Observer]
    ) -> CouncilResult:
        if self.client is not None:
            return await self._execute(self.client, question, strat, cfg, observers)
        from nimlab.core.client import NIMClient

        async with NIMClient() as client:
            return await self._execute(client, question, strat, cfg, observers)

    async def _execute(
        self,
        client: Any,
        question: str,
        strat: Strategy,
        cfg: CouncilConfig,
        observers: list[Observer],
    ) -> CouncilResult:
        roles = resolve(cfg, self.registry)
        trace = Trace(
            question=question,
            strategy=strat.name,
            models=list(roles.models),
            config={
                **cfg.model_dump(mode="json", exclude_none=True),
                "resolved": {
                    "aggregator": roles.aggregator,
                    "judge": roles.judge,
                    "drafter": roles.drafter,
                    "critic": roles.critic,
                    "reviser": roles.reviser,
                    "reward": roles.reward,
                    "embed": roles.embed,
                },
            },
            mock=bool(getattr(client, "mock", False)),
            started_at=datetime.now(timezone.utc),
        )
        ctx = RunContext(
            client=client,
            question=question,
            config=cfg,
            roles=roles,
            trace=trace,
            observers=observers,
            registry=self.registry,
        )
        before = _usage_models(client)
        started = time.perf_counter()
        ctx.emit(RunStarted(strat.name, question, tuple(roles.models)))
        error: CouncilError | None = None
        outcome = None
        try:
            outcome = await strat.run(ctx)
        except CouncilError as exc:
            error = exc
        except NIMError as exc:
            error = CouncilError(f"{type(exc).__name__}: {exc}")
            error.__cause__ = exc
        except Exception as exc:  # a bug in a (custom) strategy: keep the trace, re-raise as is
            trace.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            trace.wall_time_s = round(time.perf_counter() - started, 6)
            trace.usage = _usage_delta(trace, before, _usage_models(client))
            if outcome is not None:
                trace.final_answer = outcome.answer
                trace.rationale = outcome.rationale
                trace.candidates = outcome.candidates
                trace.meta.update(outcome.meta)
            if error is not None:
                trace.error = str(error)
            ctx.emit(RunFinished(trace, trace.error))
        if error is not None:
            error.trace = trace
            raise error
        assert outcome is not None
        return CouncilResult(
            question=question,
            strategy=strat.name,
            answer=outcome.answer,
            rationale=outcome.rationale,
            candidates=outcome.candidates,
            consensus=outcome.consensus or trace.final_consensus,
            trace=trace,
        )


# --------------------------------------------------------------------------- usage accounting
def _usage_models(client: Any) -> dict[str, dict[str, Any]] | None:
    usage = getattr(client, "usage", None)
    if usage is None:
        return None
    try:
        return dict(usage.summary()["models"])
    except Exception:
        return None


def _usage_delta(
    trace: Trace, before: dict[str, dict[str, Any]] | None, after: dict[str, dict[str, Any]] | None
) -> list[ModelUsage]:
    """Per-model usage for this run: the client tracker's delta, else derived from the steps."""
    if before is None or after is None:
        return trace.usage_from_steps()
    rows: list[ModelUsage] = []
    for model, now in sorted(after.items()):
        prev = before.get(model, {})

        def d(key: str, now: dict[str, Any] = now, prev: dict[str, Any] = prev) -> float:
            return float(now.get(key, 0)) - float(prev.get(key, 0))

        calls = int(d("calls"))
        if calls <= 0:
            continue
        tin, tout = int(d("tokens_in")), int(d("tokens_out"))
        rows.append(
            ModelUsage(
                model=model,
                calls=calls,
                errors=int(d("errors")),
                prompt_tokens=tin,
                completion_tokens=tout,
                total_tokens=tin + tout,
                latency_s=round(d("latency_s"), 6),
            )
        )
    return rows or trace.usage_from_steps()
