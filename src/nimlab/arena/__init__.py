"""Arena: benchmark NVIDIA NIM models, rank them with Elo, and route prompts to the best one.

Public API::

    from nimlab.arena import Arena, SmartRouter

    async with Arena() as arena:          # offline simulation unless NVIDIA_API_KEY is set
        run = await arena.run(suites="all")
        board = arena.leaderboard(run)
        decision = await arena.router().route("Write a function that merges intervals")

The OpenAI-compatible routing proxy lives in :mod:`nimlab.arena.proxy` (``create_app``) and the
Typer CLI in :mod:`nimlab.arena.cli` (mounted as ``nimlab arena``).
"""

from __future__ import annotations

from typing import Any

from .arena import Arena
from .battle import BattleEngine, PairwiseVerdict
from .elo import EloRating, Outcome, bootstrap_elo, fit_bradley_terry
from .graders import GradeContext, Grader, GradeResult, available_graders, register_grader
from .leaderboard import Leaderboard, ModelStanding
from .results import BattleRecord, ItemResult, RunResult, RunStore
from .router import RoutingDecision, SmartRouter
from .runner import Runner, RunSettings
from .suites import EvalItem, Suite, load_suite, load_suites

__all__ = [
    "Arena",
    "BattleEngine",
    "BattleRecord",
    "EloRating",
    "EvalItem",
    "GradeContext",
    "GradeResult",
    "Grader",
    "ItemResult",
    "Leaderboard",
    "ModelStanding",
    "Outcome",
    "PairwiseVerdict",
    "RoutingDecision",
    "RunResult",
    "RunSettings",
    "RunStore",
    "Runner",
    "SmartRouter",
    "Suite",
    "available_graders",
    "bootstrap_elo",
    "create_app",
    "fit_bradley_terry",
    "load_suite",
    "load_suites",
    "register_grader",
]


def __getattr__(name: str) -> Any:
    # FastAPI is only imported when the proxy is actually used
    if name == "create_app":
        from .proxy import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
