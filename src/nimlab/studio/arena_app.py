"""Arena: saved-run leaderboards, routing decisions and benchmark runs.

Run events: ``plan`` (what will be evaluated), ``item`` (each finished model x item), then
``run.done`` with the leaderboard.
"""

from __future__ import annotations

from typing import Any

from nimlab.arena import Arena, Leaderboard, RunStore, SmartRouter
from nimlab.arena.llm import default_models
from nimlab.arena.results import ItemResult
from nimlab.arena.simulate import SimulatedClient
from nimlab.arena.suites import list_suites
from nimlab.core import default_registry, model_scope

from .runs import Run
from .scope import scoped_work
from .state import StudioState

__all__ = ["leaderboard_summary", "list_runs", "route", "start_run", "suite_names"]

MAX_LISTED_RUNS = 20


def suite_names() -> list[str]:
    return sorted(list_suites())


def _store(state: StudioState) -> RunStore:
    return RunStore(state.settings.arena_runs_dir)


def leaderboard_summary(board: Leaderboard, *, top: int | None = None) -> dict[str, Any]:
    rows = board.standings if top is None else board.standings[:top]
    return {
        "run_id": board.run_id,
        "created_at": board.created_at,
        "mock": board.mock,
        "judge_model": board.judge_model,
        "suites": board.suites,
        "elo_source": board.elo_source,
        "standings": [
            {
                "rank": s.rank,
                "model": s.model,
                "overall": s.overall,
                "pass_rate": s.pass_rate,
                "elo": s.elo,
                "elo_lo": s.elo_lo,
                "elo_hi": s.elo_hi,
                "latency_p50_s": s.latency_p50_s,
                "ttft_p50_s": s.ttft_p50_s,
                "tokens_per_s": s.tokens_per_s,
                "cost_tier": s.cost_tier,
                "n_scored": s.n_scored,
                "n_errors": s.n_errors,
                "pareto": s.pareto,
                "suite_scores": s.suite_scores,
            }
            for s in rows
        ],
    }


def list_runs(state: StudioState) -> list[dict[str, Any]]:
    """Newest first: run metadata plus the leaderboard of each saved run."""
    store = _store(state)
    out: list[dict[str, Any]] = []
    for run_id in reversed(store.list_ids()[-MAX_LISTED_RUNS:]):
        try:
            run = store.load(run_id)
            board = Leaderboard.from_run(run, registry=default_registry(), n_boot=0)
        except (OSError, ValueError):
            continue
        summary = leaderboard_summary(board)
        out.append(
            {
                **summary,
                "status": run.status,
                "models": run.models,
                "n_items": len(run.items),
                "updated_at": run.updated_at,
            }
        )
    return out


async def route(state: StudioState, prompt: str) -> dict[str, Any]:
    client = state.client()
    router = SmartRouter(client=client, runs_dir=state.settings.arena_runs_dir)
    decision = await router.route(prompt)
    return decision.model_dump(mode="json")


def start_run(
    state: StudioState,
    suites: list[str] | str | None,
    models: list[str] | None,
    overrides: dict[str, str] | None = None,
) -> Run:
    available = set(suite_names())
    names = (
        ["all"]
        if suites in (None, "", "all", ["all"])
        else ([suites] if isinstance(suites, str) else list(suites))
    )
    unknown = [n for n in names if n != "all" and n not in available]
    if unknown:
        raise ValueError(
            f"unknown suite(s): {', '.join(unknown)}; available: {', '.join(sorted(available))}"
        )
    simulated = state.mock
    with model_scope("arena", overrides) as planning:
        chosen = [m.strip() for m in (models or []) if m and m.strip()] or default_models(
            default_registry()
        )

    async def work(run: Run) -> dict[str, Any]:
        base = state.client()
        client: Any
        if simulated:
            client = SimulatedClient(config=state.base_config().model_copy(update={"mock": True}))
            client.usage = state.usage
        else:
            client = base
        arena = Arena(client=client, runs_dir=state.settings.arena_runs_dir)
        resolved = arena.resolve_suites("all" if names == ["all"] else names)
        run.emit(
            "plan",
            {
                "models": chosen,
                "suites": [s.name for s in resolved],
                "items": sum(len(s.items) for s in resolved),
                "total_calls": sum(len(s.items) for s in resolved) * len(chosen),
            },
        )

        def on_progress(done: int, total_: int, item: ItemResult) -> None:
            run.emit(
                "item",
                {
                    "done": done,
                    "total": total_,
                    "model": item.model,
                    "suite": item.suite,
                    "item": item.item_id,
                    "score": round(item.score, 4),
                    "passed": item.passed,
                    "latency_s": round(item.latency_s, 3),
                    "error": item.error,
                },
            )

        result = await arena.run(chosen, resolved, on_progress=on_progress)
        board = Leaderboard.from_run(result, registry=default_registry(), n_boot=100)
        return {
            **leaderboard_summary(board),
            "status": result.status,
            "models": result.models,
            "n_items": len(result.items),
            "simulated": simulated,
        }

    return state.runs.start(
        "arena",
        f"{len(chosen)} models on {', '.join(names)[:48]}",
        scoped_work("arena", overrides, work, planning),
        simulated=simulated,
    )
