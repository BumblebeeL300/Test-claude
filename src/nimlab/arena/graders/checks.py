"""Programmatic instruction-following checks (word count, casing, bullets, ...)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import GradeContext, Grader, GradeResult, register_grader
from .basic import parse_flags

__all__ = ["CHECKS", "count_words", "run_check"]

CheckFn = Callable[[str, dict[str, Any]], tuple[bool, str]]
CHECKS: dict[str, CheckFn] = {}
_REQUIRED: dict[str, tuple[str, ...]] = {}  # check name -> keys of which at least one is needed
_BULLET = re.compile(r"^\s*(?:[-*•])\s+\S")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+\S")


def check(name: str, *, needs_any: tuple[str, ...] = ()) -> Callable[[CheckFn], CheckFn]:
    def deco(fn: CheckFn) -> CheckFn:
        CHECKS[name] = fn
        _REQUIRED[name] = needs_any
        return fn

    return deco


# ------------------------------------------------------------------------------ text metrics


def count_words(text: str) -> int:
    """Whitespace-separated tokens that contain at least one word character."""
    return sum(1 for tok in text.split() if re.search(r"\w", tok))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+", text.strip())
    return [p for p in parts if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def nonempty_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def _range(value: int, p: dict[str, Any], label: str) -> tuple[bool, str]:
    exactly, lo, hi = p.get("exactly"), p.get("min"), p.get("max")
    if exactly is not None:
        return value == exactly, f"{label}={value} (need exactly {exactly})"
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    want = f"{lo if lo is not None else '-inf'}..{hi if hi is not None else 'inf'}"
    return ok, f"{label}={value} (need {want})"


def _terms(p: dict[str, Any]) -> list[str]:
    terms = list(p.get("texts") or [])
    if p.get("text") is not None:
        terms.append(p["text"])
    return [str(t) for t in terms]


# ------------------------------------------------------------------------------ the checks


@check("word_count", needs_any=("exactly", "min", "max"))
def _word_count(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    return _range(count_words(text), p, "words")


@check("char_count", needs_any=("exactly", "min", "max"))
def _char_count(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    return _range(len(text.strip()), p, "chars")


@check("sentence_count", needs_any=("exactly", "min", "max"))
def _sentence_count(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    return _range(len(split_sentences(text)), p, "sentences")


@check("paragraph_count", needs_any=("exactly", "min", "max"))
def _paragraph_count(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    return _range(len(split_paragraphs(text)), p, "paragraphs")


@check("line_count", needs_any=("exactly", "min", "max"))
def _line_count(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    return _range(len(nonempty_lines(text)), p, "lines")


@check("bullet_count", needs_any=("exactly", "min", "max"))
def _bullet_count(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    return _range(sum(1 for ln in text.splitlines() if _BULLET.match(ln)), p, "bullets")


@check("only_bullets")
def _only_bullets(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    lines = nonempty_lines(text)
    bad = [ln for ln in lines if not _BULLET.match(ln)]
    return bool(lines) and not bad, f"{len(bad)} non-bullet line(s)"


@check("numbered_list", needs_any=("exactly", "min", "max"))
def _numbered_list(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    nums = [int(m.group(1)) for ln in text.splitlines() if (m := _NUMBERED.match(ln))]
    ok, msg = _range(len(nums), p, "numbered items")
    if p.get("sequential", True) and nums != list(range(1, len(nums) + 1)):
        return False, f"{msg}; numbering is not 1..n"
    return ok, msg


@check("all_caps")
def _all_caps(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters), "all letters uppercase"


@check("all_lowercase")
def _all_lowercase(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters), "all letters lowercase"


@check("includes", needs_any=("text", "texts"))
def _includes(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    flags = 0 if p.get("case_sensitive") else re.IGNORECASE
    missing = [t for t in _terms(p) if not re.search(re.escape(t), text, flags)]
    return not missing, f"missing {missing!r}" if missing else "all terms present"


@check("excludes", needs_any=("text", "texts"))
def _excludes(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    flags = 0 if p.get("case_sensitive") else re.IGNORECASE
    found = [t for t in _terms(p) if re.search(re.escape(t), text, flags)]
    return not found, f"found forbidden {found!r}" if found else "no forbidden terms"


@check("starts_with", needs_any=("text",))
def _starts_with(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    want = str(p["text"])
    got = text.lstrip()
    ok = (
        got.startswith(want)
        if p.get("case_sensitive", True)
        else got.lower().startswith(want.lower())
    )
    return ok, f"starts with {want!r}"


@check("ends_with", needs_any=("text",))
def _ends_with(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    want = str(p["text"])
    got = text.rstrip()
    ok = got.endswith(want) if p.get("case_sensitive", True) else got.lower().endswith(want.lower())
    return ok, f"ends with {want!r}"


@check("regex", needs_any=("pattern",))
def _regex(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    found = re.search(p["pattern"], text.strip(), parse_flags(p.get("flags", "")))
    ok = (found is None) if p.get("negate") else (found is not None)
    return ok, f"regex {p['pattern']!r} {'negated ' if p.get('negate') else ''}satisfied"


@check("no_commas")
def _no_commas(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    return "," not in text, f"{text.count(',')} comma(s)"


@check("in_order", needs_any=("words",))
def _in_order(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    pos = 0
    low = text.lower()
    for w in p["words"]:
        m = re.search(rf"\b{re.escape(str(w).lower())}\b", low[pos:])
        if not m:
            return False, f"{w!r} not found in order"
        pos += m.end()
    return True, "words appear in order"


@check("max_words_per_paragraph", needs_any=("max",))
def _max_words_per_paragraph(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    longest = max((count_words(par) for par in split_paragraphs(text)), default=0)
    return longest <= p["max"], f"longest paragraph has {longest} words (max {p['max']})"


@check("json_parses")
def _json_parses(text: str, p: dict[str, Any]) -> tuple[bool, str]:
    import json

    try:
        json.loads(text.strip())
    except ValueError as exc:
        return False, f"not valid JSON: {exc}"
    return True, "valid JSON"


def run_check(spec: dict[str, Any], text: str) -> tuple[bool, str]:
    """Run one check spec (``{"type": "word_count", "max": 50}``) against ``text``."""
    params = {k: v for k, v in spec.items() if k != "type"}
    return CHECKS[spec["type"]](text, params)


# ------------------------------------------------------------------------------ the grader


class ChecksParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checks: list[dict[str, Any]] = Field(min_length=1)

    @field_validator("checks")
    @classmethod
    def _validate(cls, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for spec in specs:
            kind = spec.get("type")
            if kind not in CHECKS:
                raise ValueError(f"unknown check type {kind!r}; available: {sorted(CHECKS)}")
            needs = _REQUIRED[kind]
            if needs and not any(spec.get(k) is not None for k in needs):
                raise ValueError(f"check {kind!r} needs one of {list(needs)}")
            if "pattern" in spec:
                try:
                    re.compile(spec["pattern"])
                except re.error as exc:
                    raise ValueError(f"invalid regex in check {kind!r}: {exc}") from exc
        return specs


@register_grader
class ChecksGrader(Grader):
    """All programmatic checks must pass; the score is the fraction that did."""

    name = "checks"
    description = "Programmatic checks: word count, casing, bullets, includes/excludes, ..."
    Params = ChecksParams

    async def run(self, params: ChecksParams, ctx: GradeContext) -> GradeResult:
        results = [(spec["type"], *run_check(spec, ctx.output)) for spec in params.checks]
        passed = [r for r in results if r[1]]
        failed = [f"{kind}: {msg}" for kind, ok, msg in results if not ok]
        return GradeResult(
            score=len(passed) / len(results),
            passed=not failed,
            detail=f"{len(passed)}/{len(results)} checks passed"
            + (" | failed: " + "; ".join(failed) if failed else ""),
            meta={"checks": [{"type": k, "ok": ok, "msg": m} for k, ok, m in results]},
        )
