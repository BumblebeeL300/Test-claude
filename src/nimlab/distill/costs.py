"""Token usage tallies and a rough cost estimate.

Prices are *estimates* derived from the registry's ``cost_tier`` (1 = cheapest ... 5 = priciest),
not billing data. Hosted NIM endpoints may be free or priced differently; use the figure to compare
runs, not to reconcile an invoice.
"""

from __future__ import annotations

from typing import Any

from nimlab.core import ModelRegistry, default_registry

__all__ = ["PRICE_PER_MTOK", "Tally", "estimate_cost"]

#: Assumed USD per one million tokens (prompt and completion priced alike) by cost tier.
PRICE_PER_MTOK: dict[int, float] = {1: 0.10, 2: 0.30, 3: 0.60, 4: 1.20, 5: 3.00}
DEFAULT_TIER = 3


class Tally:
    """Per-model call/token counters for one run (embeddings count as prompt tokens)."""

    def __init__(self, initial: dict[str, dict[str, int]] | None = None) -> None:
        self.models: dict[str, dict[str, int]] = {k: dict(v) for k, v in (initial or {}).items()}

    def add(
        self, model: str, prompt_tokens: int, completion_tokens: int = 0, calls: int = 1
    ) -> None:
        row = self.models.setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        )
        row["calls"] += calls
        row["prompt_tokens"] += prompt_tokens
        row["completion_tokens"] += completion_tokens

    @property
    def total_tokens(self) -> int:
        return sum(r["prompt_tokens"] + r["completion_tokens"] for r in self.models.values())

    @property
    def total_calls(self) -> int:
        return sum(r["calls"] for r in self.models.values())

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.models.items()}


def estimate_cost(usage: dict[str, dict[str, int]], registry: ModelRegistry | None = None) -> float:
    """Estimated USD for a ``Tally.as_dict()`` (unknown models are priced as tier 3)."""
    reg = registry or default_registry()
    total = 0.0
    for model, row in usage.items():
        info = reg.find(model)
        tier = info.cost_tier if info is not None else DEFAULT_TIER
        tokens = row.get("prompt_tokens", 0) + row.get("completion_tokens", 0)
        total += tokens / 1_000_000 * PRICE_PER_MTOK.get(tier, PRICE_PER_MTOK[DEFAULT_TIER])
    return round(total, 6)


def summarize(
    usage: dict[str, dict[str, int]], registry: ModelRegistry | None = None
) -> dict[str, Any]:
    """``{"models": {...}, "total_tokens", "total_calls", "cost_usd"}`` for reports."""
    tally = Tally(usage)
    return {
        "models": tally.as_dict(),
        "total_tokens": tally.total_tokens,
        "total_calls": tally.total_calls,
        "cost_usd": estimate_cost(usage, registry),
    }
