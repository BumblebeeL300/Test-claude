"""Port data types: compatibility between jacks, runtime coercion and short previews."""

from __future__ import annotations

import json
from typing import Any

from .errors import InputError
from .template import stringify

__all__ = ["TYPES", "coerce", "compat", "preview", "type_of"]

#: text, number, bool, json (any structured value), list, vectors (embeddings), any
TYPES = ("text", "number", "bool", "json", "list", "vectors", "any")

# (source, target) -> "warn" | "error"; absent pairs are fine.
_LOSSY: dict[tuple[str, str], str] = {
    ("json", "text"): "warn",
    ("list", "text"): "warn",
    ("text", "number"): "warn",
    ("text", "bool"): "warn",
    ("text", "json"): "warn",
    ("text", "list"): "warn",
    ("json", "list"): "warn",
    ("number", "bool"): "warn",
    ("bool", "number"): "warn",
    ("json", "number"): "warn",
    ("json", "bool"): "warn",
    ("list", "number"): "error",
    ("list", "bool"): "error",
    ("number", "list"): "error",
    ("bool", "list"): "error",
    ("number", "json"): "warn",
    ("bool", "json"): "warn",
    ("vectors", "text"): "error",
    ("vectors", "number"): "error",
    ("vectors", "bool"): "error",
    ("text", "vectors"): "error",
    ("number", "vectors"): "error",
    ("bool", "vectors"): "error",
    ("json", "vectors"): "error",
}


def compat(source: str, target: str) -> str:
    """``"ok"``, ``"warn"`` (coerced at run time) or ``"error"`` for a cable source -> target."""
    if source == target or "any" in (source, target):
        return "ok"
    if target == "text" and source in ("number", "bool"):
        return "ok"
    if source == "list" and target == "json":
        return "ok"
    if source == "vectors" and target in ("list", "json"):
        return "ok"
    return _LOSSY.get((source, target), "error")


def type_of(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "json"
    return "any"


def coerce(value: Any, target: str) -> Any:  # noqa: C901
    """Convert ``value`` to the port type ``target`` (raises :class:`InputError`)."""
    if target in ("any", "vectors") or value is None:
        return value
    have = type_of(value)
    if have == target:
        return value
    try:
        if target == "text":
            return stringify(value)
        if target == "number":
            if isinstance(value, bool):
                return int(value)
            number = float(str(value).strip())
            return int(number) if number.is_integer() else number
        if target == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "y", "on")
            return bool(value)
        if target == "list":
            if isinstance(value, str):
                text = value.strip()
                if text.startswith("["):
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return parsed
                return [ln for ln in text.splitlines() if ln.strip()]
            if isinstance(value, dict):
                return list(value.values())
            return [value]
        if target == "json":
            if isinstance(value, str):
                return json.loads(value)
            return value
    except (ValueError, TypeError) as exc:
        raise InputError(f"cannot use {have} value as {target}: {exc}") from exc
    return value


def preview(value: Any, limit: int = 240) -> str:
    """One compact line for the node LCD / event log."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        text = " | ".join(value)
    elif isinstance(value, list) and value and isinstance(value[0], list):
        text = f"{len(value)} vectors x {len(value[0])} dims"
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
