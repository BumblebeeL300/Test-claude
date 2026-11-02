"""Council: multi-model deliberation over NVIDIA NIM.

from nimlab.council import Council

async with NIMClient() as client:
    result = await Council(client).run("Is P = NP?", strategy="debate", rounds=2)
    print(result.answer)
    result.save_json("trace.json"); result.save_html("trace.html")
"""

from __future__ import annotations

from .config import CouncilConfig
from .consensus import agreement_score, build_report, cosine_matrix, find_outliers
from .council import Council
from .errors import (
    AllModelsFailedError,
    CouncilError,
    NotEnoughModelsError,
    UnknownStrategyError,
)
from .events import (
    CallFinished,
    CallStarted,
    ConsensusComputed,
    CouncilEvent,
    PhaseStarted,
    RunFinished,
    RunStarted,
)
from .live import CouncilLive
from .records import (
    Candidate,
    ConsensusReport,
    CouncilResult,
    ModelUsage,
    StepType,
    Trace,
    TraceStep,
)
from .report import render_html, save_html
from .reward import parse_reward
from .schemas import Critique, Issue, JudgeVerdict, RubricScore
from .strategies import (
    BaseStrategy,
    Strategy,
    StrategyOutcome,
    get_strategy,
    list_strategies,
    register_strategy,
)

__all__ = [
    "AllModelsFailedError",
    "BaseStrategy",
    "CallFinished",
    "CallStarted",
    "Candidate",
    "ConsensusComputed",
    "ConsensusReport",
    "Council",
    "CouncilConfig",
    "CouncilError",
    "CouncilEvent",
    "CouncilLive",
    "CouncilResult",
    "Critique",
    "Issue",
    "JudgeVerdict",
    "ModelUsage",
    "NotEnoughModelsError",
    "PhaseStarted",
    "RubricScore",
    "RunFinished",
    "RunStarted",
    "StepType",
    "Strategy",
    "StrategyOutcome",
    "Trace",
    "TraceStep",
    "UnknownStrategyError",
    "agreement_score",
    "build_report",
    "cosine_matrix",
    "find_outliers",
    "get_strategy",
    "list_strategies",
    "parse_reward",
    "register_strategy",
    "render_html",
    "save_html",
]
