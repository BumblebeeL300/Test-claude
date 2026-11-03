"""Council runs: stage/model events, consensus and the final answer.

Events: ``stage.start`` / ``stage.end`` (per Council phase), ``model.start`` / ``model.end`` (with a
reply preview), ``consensus`` (agreement per stage) and finally ``run.done`` with the result.
"""

from __future__ import annotations

import time
from typing import Any

from nimlab.council import Council
from nimlab.council.errors import UnknownStrategyError
from nimlab.council.events import (
    CallFinished,
    CallStarted,
    ConsensusComputed,
    CouncilEvent,
    PhaseStarted,
)
from nimlab.council.strategies import get_strategy, list_strategies

from .runs import Run
from .simulate import current_step
from .state import StudioState

__all__ = ["start_council", "strategy_names"]

PREVIEW_CHARS = 280


def strategy_names() -> list[str]:
    return [s.name for s in list_strategies()]


def validate_strategy(name: str) -> str:
    try:
        return get_strategy(name).name
    except UnknownStrategyError as exc:
        raise ValueError(
            f"unknown strategy {name!r}; choose one of {', '.join(strategy_names())}"
        ) from exc


def _preview(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1].rstrip() + "…"


class _Observer:
    def __init__(self, run: Run, client: Any, strategy: str) -> None:
        self.run = run
        self.client = client
        self.strategy = strategy
        self.stage: str | None = None
        self.stage_started = time.monotonic()
        self.steps: dict[int, dict[str, Any]] = {}
        self.skipped: set[int] = set()

    def _end_stage(self) -> None:
        if self.stage is not None:
            self.run.emit(
                "stage.end",
                {"stage": self.stage, "seconds": round(time.monotonic() - self.stage_started, 3)},
                self.stage,
            )

    def close(self) -> None:
        self._end_stage()
        self.stage = None

    def __call__(self, ev: CouncilEvent) -> None:
        if isinstance(ev, PhaseStarted):
            self._end_stage()
            self.stage = ev.phase
            self.stage_started = time.monotonic()
            self.run.emit("stage.start", {"stage": ev.phase, "strategy": self.strategy}, ev.phase)
        elif isinstance(ev, CallStarted):
            if getattr(ev.step_type, "value", ev.step_type) == "consensus":
                self.skipped.add(ev.step_index)  # embedding pass: reported by `consensus` instead
                return
            current_step.set(ev.step_index)  # lets the client tag the reply it produces
            self.steps[ev.step_index] = {"model": ev.model, "label": ev.label, "stage": ev.phase}
            self.run.emit(
                "model.start",
                {
                    "step": ev.step_index,
                    "model": ev.model,
                    "label": ev.label,
                    "stage": ev.phase,
                    "kind": getattr(ev.step_type, "value", str(ev.step_type)),
                },
                ev.phase,
            )
        elif isinstance(ev, CallFinished):
            if ev.step_index in self.skipped:
                self.skipped.discard(ev.step_index)
                return
            info = self.steps.pop(ev.step_index, {})
            preview = self.client.previews.pop(ev.step_index, "") if ev.ok else ""
            self.run.emit(
                "model.end",
                {
                    "step": ev.step_index,
                    "model": info.get("model"),
                    "label": info.get("label"),
                    "ok": ev.ok,
                    "duration_s": round(ev.duration_s, 3),
                    "tokens": ev.total_tokens,
                    "preview": _preview(preview),
                    "error": ev.error,
                },
                info.get("stage"),
            )
        elif isinstance(ev, ConsensusComputed):
            self.run.emit(
                "consensus",
                {
                    "stage": ev.phase,
                    "agreement": round(ev.agreement, 4),
                    "outliers": list(ev.outliers),
                },
                ev.phase,
            )


def start_council(
    state: StudioState, question: str, strategy: str, models: list[str] | None
) -> Run:
    strategy = validate_strategy(strategy)
    simulated = state.mock

    async def work(run: Run) -> dict[str, Any]:
        client = state.studio_client(council=True)
        observer = _Observer(run, client, strategy)
        council = Council(client, observers=[observer])
        try:
            result = await council.run(question, strategy=strategy, models=models or None)
        finally:
            observer.close()
        trace = result.trace
        consensus = result.consensus
        return {
            "question": question,
            "strategy": result.strategy,
            "answer": result.answer,
            "rationale": result.rationale,
            "agreement": round(consensus.agreement, 4) if consensus else None,
            "consensus": consensus.model_dump(mode="json") if consensus else None,
            "candidates": [
                {
                    "label": c.label,
                    "model": c.model,
                    "text": c.text,
                    "score": c.score,
                    "score_source": c.score_source,
                    "selected": c.selected,
                    "outlier": c.outlier,
                }
                for c in result.candidates
            ],
            "models": trace.all_models,
            "usage": [u.model_dump(mode="json") for u in trace.usage],
            "total_tokens": trace.total_tokens,
            "wall_time_s": trace.wall_time_s,
            "steps": len(trace.steps),
            "simulated": simulated,
        }

    return state.runs.start("council", question.strip()[:80], work, simulated=simulated)
