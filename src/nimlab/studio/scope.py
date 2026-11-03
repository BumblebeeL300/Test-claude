"""Per-app / per-request model overrides for Studio runs.

Every run-starting endpoint accepts an optional ``model_overrides`` object (``{role: model}``).
A run executes inside :func:`nimlab.core.model_scope`, so ``default_registry().pick(role)`` honours,
highest precedence first: this request's overrides, the app's saved overrides (the patch bay in the
app's header), the global per-role overrides (Settings -> Models), ``NIMLAB_MODEL_<ROLE>``,
``nimlab.toml``, live discovery and finally the curated default. The result carries a
``model_info`` block listing which model served which role, including automatic substitutions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from nimlab.core import ModelRegistry, Role, model_scope

from .runs import Run

__all__ = [
    "APP_ROLES",
    "clean_overrides",
    "model_info",
    "scoped_work",
]

#: Studio app id -> the roles that app uses, with the label shown on its patch chips.
APP_ROLES: dict[str, list[dict[str, str]]] = {
    "forge": [
        {"role": "reasoning", "label": "PLANNER"},
        {"role": "coder", "label": "CODER"},
        {"role": "chat", "label": "REVIEWER"},
    ],
    "council": [
        {"role": "chat", "label": "PANEL"},
        {"role": "reasoning", "label": "JUDGE"},
    ],
    "atlas": [
        {"role": "embed", "label": "EMBED"},
        {"role": "rerank", "label": "RERANK"},
        {"role": "chat", "label": "ANSWER"},
    ],
    "arena": [
        {"role": "chat", "label": "CHAT"},
        {"role": "reasoning", "label": "REASONING"},
        {"role": "coder", "label": "CODER"},
        {"role": "fast", "label": "FAST"},
    ],
    "prism": [{"role": "vision", "label": "VISION"}],
    "sentinel": [
        {"role": "chat", "label": "TARGET"},
        {"role": "safety", "label": "GUARD"},
    ],
    "loom": [
        {"role": "chat", "label": "CHAT"},
        {"role": "fast", "label": "FAST"},
        {"role": "reasoning", "label": "REASON"},
        {"role": "embed", "label": "EMBED"},
    ],
    "distill": [
        {"role": "chat", "label": "GENERATE"},
        {"role": "fast", "label": "WEAK"},
        {"role": "reward", "label": "REWARD"},
        {"role": "embed", "label": "EMBED"},
    ],
}


def clean_overrides(raw: Mapping[str, Any] | None) -> dict[str, str]:
    """Validated ``{role: model}``: unknown roles raise ``ValueError``; blanks / ``auto`` drop."""
    out: dict[str, str] = {}
    for key, value in (raw or {}).items():
        try:
            role = Role(str(key).strip().lower())
        except ValueError:
            valid = ", ".join(r.value for r in Role)
            raise ValueError(f"unknown role {key!r}; valid roles: {valid}") from None
        if not isinstance(value, str):
            raise ValueError(f"model for role {key!r} must be a string")
        model = value.strip()
        if not model or model.lower() == "auto":
            continue
        if len(model) > 200 or any(ch.isspace() for ch in model):
            raise ValueError(f"invalid model id for role {key!r}")
        out[role.value] = model
    return out


def model_info(
    view: ModelRegistry,
    overrides: Mapping[str, str] | None = None,
    earlier: ModelRegistry | None = None,
) -> dict[str, Any]:
    """What a finished run used: roles picked inside the scope, request overrides, fallbacks.

    ``earlier`` is the scope used while validating / planning the request (before the run task).
    """
    picks = [*(earlier.picks or [] if earlier else []), *(view.picks or [])]
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for p in picks:
        key = (p["role"], p["model"], p["source"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    picks = unique
    return {
        "picked": [p for p in picks if p.get("source") != "fallback"],
        "substitutions": [p for p in picks if p.get("source") == "fallback"],
        "overrides": dict(overrides or {}),
    }


def scoped_work(
    app: str,
    overrides: Mapping[str, str] | None,
    work: Callable[[Run], Awaitable[dict[str, Any]]],
    earlier: ModelRegistry | None = None,
) -> Callable[[Run], Awaitable[dict[str, Any]]]:
    """Run ``work`` inside the model scope of ``app`` and attach ``model_info`` to its result."""
    chosen = dict(overrides or {})

    async def runner(run: Run) -> dict[str, Any]:
        with model_scope(app, chosen) as view:
            result = await work(run)
        if isinstance(result, dict):
            result["model_info"] = model_info(view, chosen, earlier)
        return result

    return runner
