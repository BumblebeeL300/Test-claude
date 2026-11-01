"""Model selection for Forge's three roles (planner, coder, reviewer)."""

from __future__ import annotations

from nimlab.core import ModelRegistry, Role, default_registry

from .models import ModelAssignment

__all__ = ["model_family", "resolve_models"]


def model_family(model_id: str) -> str:
    """Vendor prefix of a NIM model id (``qwen/qwen2.5-coder`` -> ``qwen``)."""
    return model_id.split("/", 1)[0].lower() if "/" in model_id else model_id.lower()


def resolve_models(
    planner: str | None = None,
    coder: str | None = None,
    reviewer: str | None = None,
    registry: ModelRegistry | None = None,
) -> ModelAssignment:
    """Pick one model per role.

    * planner: ``Role.REASONING`` (root-cause analysis also uses it)
    * coder: ``Role.CODER``
    * reviewer: a ``CHAT`` model from a *different family* than the coder where one exists, so the
      reviewer does not share the coder's blind spots.

    Explicit arguments always win.
    """
    reg = registry or default_registry()
    chosen_planner = planner or reg.pick(Role.REASONING)
    chosen_coder = coder or reg.pick(Role.CODER)
    if reviewer:
        chosen_reviewer = reviewer
    else:
        candidates: list[str] = []
        for role in (Role.CHAT, Role.FAST, Role.REASONING):
            candidates.append(reg.pick(role))
            candidates.extend(m.id for m in reg.by_role(role))
        coder_family = model_family(chosen_coder)
        chosen_reviewer = next(
            (c for c in candidates if model_family(c) != coder_family), reg.pick(Role.CHAT)
        )
    return ModelAssignment(planner=chosen_planner, coder=chosen_coder, reviewer=chosen_reviewer)
