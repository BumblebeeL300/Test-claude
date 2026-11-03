"""``python_fn`` nodes call only functions registered here. Arbitrary code is never accepted.

>>> from nimlab.loom import register_function
>>> register_function("shout", lambda value, **_: str(value).upper() + "!")

A function receives the node's ``value`` input as its first argument and the node's ``args``
object as keyword arguments, and may be sync or async. It must return JSON-compatible data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

__all__ = ["builtin_functions", "get_function", "list_functions", "register_function"]

_FUNCS: dict[str, tuple[Callable[..., Any], str]] = {}
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,47}$")


def register_function(
    name: str, fn: Callable[..., Any], *, doc: str = "", replace: bool = False
) -> Callable[..., Any]:
    if not _NAME.match(name):
        raise ValueError(f"invalid function name {name!r}")
    if name in _FUNCS and not replace:
        raise ValueError(f"function {name!r} is already registered")
    summary = doc or next(iter((fn.__doc__ or "").strip().splitlines()), "")
    _FUNCS[name] = (fn, summary)
    return fn


def get_function(name: str) -> Callable[..., Any] | None:
    entry = _FUNCS.get(name)
    return entry[0] if entry else None


def list_functions() -> dict[str, str]:
    return {name: doc for name, (_, doc) in sorted(_FUNCS.items())}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        return [ln for ln in value.splitlines() if ln.strip()]
    return [value]


def _flatten(value: Any, **_: Any) -> list[Any]:
    """Flatten a list of lists by one level."""
    out: list[Any] = []
    for item in _as_list(value):
        out.extend(item if isinstance(item, list) else [item])
    return out


def _pluck(value: Any, field: str = "", **_: Any) -> list[Any]:
    """Take ``args.field`` from each object in a list."""
    return [item.get(field) if isinstance(item, dict) else item for item in _as_list(value)]


def _flatten_field(value: Any, field: str = "", **_: Any) -> list[Any]:
    """Take ``args.field`` (a list) from each object and concatenate them."""
    return _flatten(_pluck(value, field=field))


def _get_field(value: Any, field: str = "", **_: Any) -> Any:
    """Read ``args.field`` from an object (dotted paths allowed: ``a.b``)."""
    cur = value
    for seg in [p for p in str(field).split(".") if p]:
        if isinstance(cur, dict):
            cur = cur.get(seg)
        elif isinstance(cur, list) and seg.isdigit() and int(seg) < len(cur):
            cur = cur[int(seg)]
        else:
            return None
    return cur


def _unique(value: Any, **_: Any) -> list[Any]:
    """Drop exact duplicates, keeping order."""
    seen: set[str] = set()
    out: list[Any] = []
    for item in _as_list(value):
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _sort(value: Any, reverse: bool = False, **_: Any) -> list[Any]:
    """Sort a list of scalars."""
    return sorted(_as_list(value), key=lambda v: (str(type(v)), v), reverse=bool(reverse))


def _top_k(value: Any, k: int = 3, **_: Any) -> list[Any]:
    """Keep the first ``args.k`` items."""
    return _as_list(value)[: max(int(k), 0)]


def _count(value: Any, **_: Any) -> int:
    """Number of items (or characters of a string)."""
    return len(value) if isinstance(value, (str, list, dict)) else 0


def _join_lines(value: Any, **_: Any) -> str:
    """Join items with newlines."""
    return "\n".join(str(v) for v in _as_list(value))


def _split_lines(value: Any, **_: Any) -> list[str]:
    """Split text into non-empty lines."""
    return [ln.strip() for ln in str(value or "").splitlines() if ln.strip()]


def _sentences(value: Any, **_: Any) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(value or "")) if s.strip()]


def _word_count(value: Any, **_: Any) -> int:
    """Number of words in the text."""
    return len(re.findall(r"\w+", str(value or "")))


def _truncate(value: Any, chars: int = 200, **_: Any) -> str:
    """Cut text to ``args.chars`` characters."""
    text = str(value or "")
    return text if len(text) <= int(chars) else text[: int(chars)].rstrip() + "…"


def _to_json(value: Any, **_: Any) -> str:
    """Serialise to a JSON string."""
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _from_json(value: Any, **_: Any) -> Any:
    """Parse a JSON string."""
    return json.loads(value) if isinstance(value, str) else value


def _lower(value: Any, **_: Any) -> str:
    """Lowercase text."""
    return str(value or "").lower()


def _numbered(value: Any, **_: Any) -> str:
    """Number the items 1., 2., ..."""
    return "\n".join(f"{i}. {v}" for i, v in enumerate(_as_list(value), 1))


def builtin_functions() -> dict[str, Callable[..., Any]]:
    return {
        "flatten": _flatten, "pluck": _pluck, "flatten_field": _flatten_field, "get_field": _get_field,
        "unique": _unique, "sort": _sort, "top_k": _top_k, "count": _count,
        "join_lines": _join_lines, "split_lines": _split_lines, "sentences": _sentences,
        "word_count": _word_count, "truncate": _truncate, "to_json": _to_json,
        "from_json": _from_json, "lower": _lower, "numbered": _numbered,
    }  # fmt: skip


for _name, _fn in builtin_functions().items():
    register_function(_name, _fn)
