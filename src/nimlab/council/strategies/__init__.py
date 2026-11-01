"""Built-in council strategies. Importing this package registers them."""

from __future__ import annotations

from .base import (
    BaseStrategy,
    Strategy,
    StrategyOutcome,
    get_strategy,
    list_strategies,
    register_strategy,
    unregister_strategy,
)

from .moa import MixtureOfAgents  # isort: skip
from .debate import Debate  # isort: skip
from .best_of_n import BestOfN  # isort: skip
from .critic import CriticLoop  # isort: skip

__all__ = [
    "BaseStrategy",
    "BestOfN",
    "CriticLoop",
    "Debate",
    "MixtureOfAgents",
    "Strategy",
    "StrategyOutcome",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "unregister_strategy",
]
