"""``{{var}}`` templates: interpolation with dotted paths and a few filters. No code execution.

``{{name}}``, ``{{plan.steps.0}}``, ``{{items | bullets}}``. Anything between ``{{`` and ``}}`` that
does not look like a variable path (for example the JSON example ``{{"a": 1}}``) is left as is.

Filters: ``json`` ``upper`` ``lower`` ``title`` ``trim`` ``len`` ``lines`` ``bullets`` ``numbered``
``first`` ``last`` ``join`` (comma separated).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .errors import TemplateError

__all__ = ["FILTERS", "render", "stringify", "variables"]

_SLOT = re.compile(r"\{\{\s*([A-Za-z_][\w]*(?:\.[\w]+)*)\s*((?:\|\s*[A-Za-z_]\w*\s*)*)\}\}")


def stringify(value: Any) -> str:
    """Text form of a value: strings as is, lists of scalars one per line, the rest as JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
        return "\n".join(str(v) for v in value)
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [ln for ln in value.splitlines() if ln.strip()]
    if isinstance(value, dict):
        return list(value.values())
    return [value]


def _first(value: Any) -> Any:
    items = _as_list(value)
    return items[0] if items else ""


def _last(value: Any) -> Any:
    items = _as_list(value)
    return items[-1] if items else ""


FILTERS: dict[str, Callable[[Any], Any]] = {
    "json": lambda v: json.dumps(v, ensure_ascii=False, indent=2, default=str),
    "upper": lambda v: stringify(v).upper(),
    "lower": lambda v: stringify(v).lower(),
    "title": lambda v: stringify(v).title(),
    "trim": lambda v: stringify(v).strip(),
    "len": lambda v: len(v) if isinstance(v, (str, list, dict)) else len(stringify(v)),
    "lines": lambda v: "\n".join(_as_list(v) if not isinstance(v, str) else v.splitlines()),
    "bullets": lambda v: "\n".join(f"- {stringify(x)}" for x in _as_list(v)),
    "numbered": lambda v: "\n".join(f"{i}. {stringify(x)}" for i, x in enumerate(_as_list(v), 1)),
    "first": _first,
    "last": _last,
    "join": lambda v: ", ".join(stringify(x) for x in _as_list(v)),
}


def variables(text: str) -> list[str]:
    """Base variable names used in ``text``, in first-use order."""
    out: list[str] = []
    for m in _SLOT.finditer(text or ""):
        base = m.group(1).split(".", 1)[0]
        if base not in out:
            out.append(base)
    return out


def paths(text: str) -> list[str]:
    """Every full variable path (``plan.steps.0``) used in ``text``."""
    return [m.group(1) for m in _SLOT.finditer(text or "")]


def filters_used(text: str) -> list[str]:
    """Filter names used in ``text`` (to report unknown ones before running)."""
    out: list[str] = []
    for m in _SLOT.finditer(text or ""):
        out.extend(re.findall(r"\|\s*([A-Za-z_]\w*)", m.group(2)))
    return out


def _lookup(path: str, scope: Mapping[str, Any]) -> Any:
    head, *rest = path.split(".")
    if head not in scope:
        raise TemplateError(f"unknown variable {head!r}")
    value = scope[head]
    for seg in rest:
        if isinstance(value, dict) and seg in value:
            value = value[seg]
        elif isinstance(value, list) and seg.isdigit() and int(seg) < len(value):
            value = value[int(seg)]
        else:
            raise TemplateError(f"cannot read {seg!r} in {head}.{'.'.join(rest)}")
    return value


def render(text: str, scope: Mapping[str, Any]) -> str:
    """Interpolate ``text``. Raises :class:`TemplateError` for unknown variables or paths."""

    def sub(m: re.Match[str]) -> str:
        value = _lookup(m.group(1), scope)
        for name in re.findall(r"\|\s*([A-Za-z_]\w*)", m.group(2)):
            fn = FILTERS.get(name)
            if fn is None:
                raise TemplateError(f"unknown filter {name!r}")
            value = fn(value)
        return stringify(value)

    return _SLOT.sub(sub, text or "")
