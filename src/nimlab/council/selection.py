"""Choosing which models sit on the panel and which one plays each special role."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from nimlab.core.registry import ModelRegistry, Role, default_registry

from .config import CouncilConfig
from .errors import CouncilError

DEFAULT_PANEL_SIZE = 3


def family(model_id: str) -> str:
    """Model family proxy: the organisation prefix (``meta/llama-...`` -> ``meta``)."""
    return model_id.split("/", 1)[0].lower() if "/" in model_id else model_id.lower()


def _ids(items: Iterable[Any]) -> list[str]:
    return [i if isinstance(i, str) else str(i.id) for i in items]


def _safe_pick(registry: ModelRegistry, role: Role) -> str | None:
    try:
        return registry.pick(role)
    except (KeyError, ValueError):
        return None


def diverse_subset(candidates: Iterable[str], count: int) -> list[str]:
    """First ``count`` unique candidates, preferring one model per family before repeating one."""
    ordered = list(dict.fromkeys(candidates))
    chosen: list[str] = []
    seen_families: set[str] = set()
    for model in ordered:
        if len(chosen) >= count:
            break
        if family(model) not in seen_families:
            chosen.append(model)
            seen_families.add(family(model))
    for model in ordered:
        if len(chosen) >= count:
            break
        if model not in chosen:
            chosen.append(model)
    return chosen


def default_panel(
    registry: ModelRegistry | None = None, count: int = DEFAULT_PANEL_SIZE
) -> list[str]:
    """A diverse default panel: the REASONING pick, the CHAT pick, then other strong families.

    Filler models are ranked by closeness to mid cost tier 3 (strong but not the very largest and
    slowest), NVIDIA's own models first on ties, then alphabetically.
    """
    reg = registry or default_registry()
    pool: list[str] = []
    for role in (Role.REASONING, Role.CHAT):
        picked = _safe_pick(reg, role)
        if picked:
            pool.append(picked)

    def rank(info: Any) -> tuple[int, bool, str]:
        return (abs(int(getattr(info, "cost_tier", 3)) - 3), family(info.id) != "nvidia", info.id)

    fillers = {m.id: m for m in [*reg.by_role(Role.CHAT), *reg.by_role(Role.REASONING)]}
    pool += [m.id for m in sorted(fillers.values(), key=rank)]
    pool += _ids(reg.by_role(Role.FAST))
    panel = diverse_subset(pool, count)
    if not panel:
        raise CouncilError("the model registry has no CHAT/REASONING models to form a panel")
    return panel


@dataclass(frozen=True)
class ResolvedRoles:
    """Concrete model ids for every role a strategy may need."""

    models: list[str]
    aggregator: str
    judge: str
    drafter: str
    critic: str
    reviser: str
    reward: str | None
    embed: str | None


def resolve(
    config: CouncilConfig,
    registry: ModelRegistry | None = None,
    *,
    models: list[str] | None = None,
) -> ResolvedRoles:
    """Fill in every unset model from the registry.

    When the caller supplied the panel explicitly (``models`` argument or ``config.models``) the
    aggregator/judge default to the first panel member, so no model outside the requested set is
    called unexpectedly. With the default panel they use the registry's CHAT / REASONING picks.
    """
    reg = registry or default_registry()
    explicit = models or config.models
    panel = list(explicit) if explicit else default_panel(reg)
    if not panel:
        raise CouncilError("no models available for the council")

    if explicit:
        aggregator = config.aggregator or panel[0]
        judge = config.judge or panel[0]
    else:
        aggregator = config.aggregator or _safe_pick(reg, Role.CHAT) or panel[0]
        judge = config.judge or _safe_pick(reg, Role.REASONING) or panel[0]

    drafter = config.drafter or panel[0]
    critic = config.critic or (panel[1] if len(panel) > 1 else panel[0])
    reviser = config.reviser or drafter
    return ResolvedRoles(
        models=panel,
        aggregator=aggregator,
        judge=judge,
        drafter=drafter,
        critic=critic,
        reviser=reviser,
        reward=config.reward_model or _safe_pick(reg, Role.REWARD),
        embed=config.embed_model or _safe_pick(reg, Role.EMBED),
    )


def is_reasoning_model(model: str, registry: ModelRegistry | None = None) -> bool:
    """True if the registry lists ``model`` as a REASONING model (these need larger budgets)."""
    reg = registry or default_registry()
    info = reg.find(model)
    return bool(info and Role.REASONING in info.roles)
