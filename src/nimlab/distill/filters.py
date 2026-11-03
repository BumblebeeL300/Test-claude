"""Quality filters. Each check returns ``(reason, detail)`` for a failing sample or ``None``."""

from __future__ import annotations

import re

from .models import Sample
from .recipes import FilterConfig, RecipeKind
from .textutil import content_words, looks_english, repeat_ratio

__all__ = [
    "apply_filters",
    "check_grounding",
    "check_language",
    "check_length",
    "check_refusal",
    "check_repetition",
    "check_score",
    "is_refusal",
]

Failure = tuple[str, str]

_REFUSAL_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi(?:'m| am) sorry,? (?:but )?i (?:can(?:'|no)t|cannot|am unable|won't)",
        r"\bi (?:can(?:'|no)t|cannot|am unable to|won't be able to) (?:help|assist|comply|provide)",
        r"\bas an ai (?:language )?model\b",
        r"\bi(?:'m| am) (?:not able|unable) to (?:help|assist|provide|answer|comply)",
        r"\bi (?:do not|don't) know\b",
        r"\bi apologi[sz]e,? but\b",
    )
]
#: A refusal is a short reply; long answers may legitimately quote such phrases.
_REFUSAL_MAX_CHARS = 400


def is_refusal(text: str) -> bool:
    """True for short replies that decline to answer ("I'm sorry, but I can't ...")."""
    flat = " ".join(text.split())
    return len(flat) <= _REFUSAL_MAX_CHARS and any(rx.search(flat) for rx in _REFUSAL_RES)


def check_refusal(text: str) -> Failure | None:
    return ("refusal", "response declines or gives a non-answer") if is_refusal(text) else None


def check_length(text: str, cfg: FilterConfig, *, prompt: bool = False) -> Failure | None:
    n = len(text.strip())
    if prompt:
        if n < cfg.min_prompt_chars:
            return "short_prompt", f"{n} chars < {cfg.min_prompt_chars}"
        return None
    if n < cfg.min_chars:
        return "too_short", f"{n} chars < {cfg.min_chars}"
    if n > cfg.max_chars:
        return "too_long", f"{n} chars > {cfg.max_chars}"
    return None


def check_language(text: str, cfg: FilterConfig) -> Failure | None:
    if cfg.language == "en" and not looks_english(text):
        return "language", "does not look like English"
    return None


def check_repetition(text: str, cfg: FilterConfig) -> Failure | None:
    if cfg.max_repeat_ratio is None:
        return None
    ratio = repeat_ratio(text)
    if ratio > cfg.max_repeat_ratio:
        return "repetition", f"{ratio:.0%} of 3-grams repeat"
    return None


def check_score(sample: Sample, cfg: FilterConfig) -> Failure | None:
    if sample.score is None or sample.score_source is None:
        return "unscored", "no reward or judge score"
    floor = cfg.min_reward if sample.score_source == "reward" else cfg.min_judge
    if floor is not None and sample.score < floor:
        return "low_score", f"{sample.score_source} {sample.score:.2f} < {floor:g}"
    return None


def check_grounding(sample: Sample, cfg: FilterConfig) -> Failure | None:
    if cfg.min_grounding is None or not sample.context:
        return None
    answer = set(content_words(sample.response))
    if not answer:
        return "ungrounded", "empty answer"
    support = len(answer & set(content_words(sample.context))) / len(answer)
    if support < cfg.min_grounding:
        return "ungrounded", f"{support:.0%} of answer words occur in the excerpt"
    return None


def apply_filters(sample: Sample, cfg: FilterConfig, kind: RecipeKind) -> Failure | None:
    """The first failing check for ``sample`` (checks run cheapest and most specific first)."""
    checks: list[Failure | None] = [
        check_length(sample.prompt, cfg, prompt=True),
        check_refusal(sample.response),
        check_length(sample.response, cfg),
        check_language(sample.response, cfg),
        check_repetition(sample.response, cfg),
        check_score(sample, cfg),
    ]
    for failure in checks:
        if failure is not None:
            return failure
    if kind == "preference_pairs":
        if not sample.rejected or sample.rejected_score is None or sample.score is None:
            return "unscored", "missing rejected response"
        margin = sample.score - sample.rejected_score
        if margin < cfg.min_margin:
            return "low_margin", f"margin {margin:.2f} < {cfg.min_margin:g}"
    if kind == "qa_from_docs":
        return check_grounding(sample, cfg)
    if kind == "eval_set" and cfg.min_difficulty is not None:
        floor = cfg.min_difficulty
        if sample.difficulty is None or sample.difficulty < floor:
            return "too_easy", f"difficulty {sample.difficulty} < {floor:g}"
    return None
