"""Deterministic string / number graders: exact, numeric, regex, contains_all."""

from __future__ import annotations

import math
import re
import string
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import GradeContext, Grader, GradeResult, register_grader

__all__ = ["extract_number", "find_numbers", "normalize_text"]

_FLAG_BITS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}
_LEAD_IN = re.compile(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*", re.IGNORECASE)
_STRIP_CHARS = string.punctuation + "\u201c\u201d\u2018\u2019 \t\r\n"


def parse_flags(flags: str) -> int:
    bits = 0
    for ch in flags.lower():
        if ch not in _FLAG_BITS:
            raise ValueError(f"unknown regex flag {ch!r}; use any of 'imsx'")
        bits |= _FLAG_BITS[ch]
    return bits


def normalize_text(text: str, *, case_sensitive: bool = False) -> str:
    """Lowercase (unless asked not to), drop markdown emphasis, lead-ins and edge punctuation."""
    t = text.replace("**", "").replace("__", "").strip()
    t = _LEAD_IN.sub("", t)
    t = t.strip(_STRIP_CHARS)
    t = re.sub(r"\s+", " ", t)
    return t if case_sensitive else t.casefold()


# --------------------------------------------------------------------------------- exact


class ExactParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: str | list[str]
    case_sensitive: bool = False
    mode: Literal["equals", "contains"] = "equals"

    @property
    def accepted(self) -> list[str]:
        return [self.expected] if isinstance(self.expected, str) else list(self.expected)

    @field_validator("expected")
    @classmethod
    def _non_empty(cls, v: str | list[str]) -> str | list[str]:
        if (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v):
            raise ValueError("expected must not be empty")
        return v


@register_grader
class ExactGrader(Grader):
    """Normalised equality with any accepted answer (or a whole-word match in ``contains`` mode)."""

    name = "exact"
    description = "Output equals (or contains as a whole word) one of the accepted answers."
    Params = ExactParams

    async def run(self, params: ExactParams, ctx: GradeContext) -> GradeResult:
        cs = params.case_sensitive
        got = normalize_text(ctx.output, case_sensitive=cs)
        for want in params.accepted:
            norm = normalize_text(want, case_sensitive=cs)
            if params.mode == "equals":
                hit = got == norm
            else:
                flags = 0 if cs else re.IGNORECASE
                hit = re.search(rf"(?<!\w){re.escape(norm)}(?!\w)", got, flags) is not None
            if hit:
                return GradeResult(
                    score=1.0, passed=True, detail=f"matched {want!r}", extracted=got
                )
        return GradeResult(
            score=0.0,
            passed=False,
            detail=f"expected one of {params.accepted!r}, got {got[:80]!r}",
            extracted=got[:200],
        )


# --------------------------------------------------------------------------------- numeric

_NUMBER_RE = re.compile(
    r"""
    (?P<frac>(?<![\w.])(?P<fn>\d+)\s*/\s*(?P<fd>\d+)(?![\w.]))
    |
    (?:(?<![\w.])(?P<sign>[-+\u2212])\s?)?
    \$?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    (?:[eE](?P<exp>[-+]?\d+))?
    """,
    re.VERBOSE,
)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_RE = re.compile(
    r"^[ \t>#\-]*(?:the\s+)?(?:final\s+)?answer\b[ \t]*(?:is\b|:|=|-)?[ \t]*(?P<rest>.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def find_numbers(text: str) -> list[float]:
    """All numbers in ``text`` (handles ``1,234.5``, ``-3``, ``$12``, ``1e3`` and ``3/4``)."""
    out: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        if m.group("frac"):
            denom = int(m.group("fd"))
            if denom:
                out.append(int(m.group("fn")) / denom)
            continue
        value = float(m.group("num").replace(",", ""))
        if m.group("exp"):
            value *= 10 ** int(m.group("exp"))
        if m.group("sign") in ("-", "\u2212"):
            value = -value
        out.append(value)
    return out


def extract_number(text: str, mode: str = "auto") -> float | None:
    """Pick the model's final numeric answer.

    ``auto``: the last ``\\boxed{}``, else the last ``Answer: ...`` line, else the last number in
    the text. ``first`` / ``last`` just take the first / last number found.
    """
    clean = text.replace("**", "").replace("__", "")
    if mode == "first":
        nums = find_numbers(clean)
        return nums[0] if nums else None
    if mode == "auto":
        for rx in (_BOXED_RE, _ANSWER_RE):
            matches = list(rx.finditer(clean))
            for m in reversed(matches):
                segment = m.group(1) if rx is _BOXED_RE else m.group("rest")
                nums = find_numbers(segment)
                if nums:
                    return nums[0]
    nums = find_numbers(clean)
    return nums[-1] if nums else None


class NumericParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: float
    rel_tol: float = Field(default=1e-6, ge=0)
    abs_tol: float = Field(default=1e-9, ge=0)
    extract: Literal["auto", "first", "last"] = "auto"


@register_grader
class NumericGrader(Grader):
    """Extracts the final number from the output and compares with tolerance."""

    name = "numeric"
    description = "Final number in the output is within rel/abs tolerance of the expected value."
    Params = NumericParams

    async def run(self, params: NumericParams, ctx: GradeContext) -> GradeResult:
        value = extract_number(ctx.output, params.extract)
        if value is None:
            return GradeResult(score=0.0, passed=False, detail="no number found in output")
        ok = math.isclose(value, params.expected, rel_tol=params.rel_tol, abs_tol=params.abs_tol)
        detail = f"extracted {value:g}, expected {params.expected:g}"
        return GradeResult(
            score=1.0 if ok else 0.0, passed=ok, detail=detail, extracted=f"{value:g}"
        )


# --------------------------------------------------------------------------------- regex


class RegexParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str
    flags: str = "i"
    mode: Literal["search", "match", "fullmatch"] = "search"
    negate: bool = False

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return v

    @field_validator("flags")
    @classmethod
    def _flags_ok(cls, v: str) -> str:
        parse_flags(v)
        return v


@register_grader
class RegexGrader(Grader):
    """Passes when the pattern matches (or, with ``negate``, does not match) the output."""

    name = "regex"
    description = "Output matches a regular expression."
    Params = RegexParams

    async def run(self, params: RegexParams, ctx: GradeContext) -> GradeResult:
        rx = re.compile(params.pattern, parse_flags(params.flags))
        m = getattr(rx, params.mode)(ctx.output.strip())
        ok = (m is None) if params.negate else (m is not None)
        detail = f"pattern {params.pattern!r} {'matched' if m else 'did not match'}"
        return GradeResult(
            score=1.0 if ok else 0.0,
            passed=ok,
            detail=detail,
            extracted=m.group(0)[:200] if m else None,
        )


# --------------------------------------------------------------------------------- contains_all


class ContainsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    any_of: list[list[str]] = Field(default_factory=list)
    case_sensitive: bool = False
    whole_word: bool = False

    @field_validator("any_of")
    @classmethod
    def _groups_non_empty(cls, v: list[list[str]]) -> list[list[str]]:
        if any(not g for g in v):
            raise ValueError("every any_of group needs at least one alternative")
        return v

    def model_post_init(self, context: Any, /) -> None:
        if not (self.required or self.forbidden or self.any_of):
            raise ValueError("contains_all needs at least one of required / forbidden / any_of")


@register_grader
class ContainsAllGrader(Grader):
    """Every required term present, no forbidden term present, one of each ``any_of`` group.

    The score is the fraction of satisfied conditions; ``passed`` requires all of them.
    """

    name = "contains_all"
    description = "Required terms present and forbidden terms absent."
    Params = ContainsParams

    async def run(self, params: ContainsParams, ctx: GradeContext) -> GradeResult:
        flags = 0 if params.case_sensitive else re.IGNORECASE

        def has(term: str) -> bool:
            pat = re.escape(term)
            if params.whole_word:
                pat = rf"(?<!\w){pat}(?!\w)"
            return re.search(pat, ctx.output, flags) is not None

        checks: list[tuple[str, bool]] = []
        checks += [(f"has {t!r}", has(t)) for t in params.required]
        checks += [(f"lacks {t!r}", not has(t)) for t in params.forbidden]
        checks += [(f"has one of {g!r}", any(has(t) for t in g)) for g in params.any_of]
        ok = sum(1 for _, passed in checks if passed)
        failed = [label for label, passed in checks if not passed]
        return GradeResult(
            score=ok / len(checks),
            passed=not failed,
            detail="all conditions met" if not failed else "failed: " + "; ".join(failed),
        )
