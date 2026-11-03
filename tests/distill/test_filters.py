"""Each quality filter, plus the combined ``apply_filters`` order."""

from __future__ import annotations

from typing import Any

import pytest

from nimlab.distill.filters import (
    apply_filters,
    check_grounding,
    check_language,
    check_length,
    check_refusal,
    check_repetition,
    check_score,
    is_refusal,
)
from nimlab.distill.models import Sample
from nimlab.distill.recipes import FilterConfig

GOOD = (
    "Start with a clear goal for the project and write it down so that everyone on the team "
    "can check later decisions against it. Then split the work into small steps."
)


def sample(**kw: Any) -> Sample:
    base: dict[str, Any] = {
        "id": "s0",
        "seed_idx": 0,
        "prompt": "How should a team plan a project?",
        "response": GOOD,
        "model": "m",
        "score": 1.0,
        "score_source": "reward",
    }
    return Sample(**{**base, **kw})


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with that request.",
        "As an AI language model, I cannot answer this.",
        "I am unable to assist with that.",
        "I don't know based on the provided text.",
        "I apologize, but that is outside what I can do.",
    ],
)
def test_refusals_are_detected(text: str) -> None:
    assert is_refusal(text)
    assert check_refusal(text) is not None


def test_long_answers_quoting_refusal_phrases_are_not_refusals() -> None:
    long_text = "Explain the phrase 'as an AI language model' in a chat log. " + GOOD * 4
    assert not is_refusal(long_text)
    assert check_refusal(GOOD) is None


def test_length_bounds() -> None:
    cfg = FilterConfig(min_chars=20, max_chars=100, min_prompt_chars=10)
    assert check_length("short", cfg) == ("too_short", "5 chars < 20")
    assert check_length("x" * 101, cfg) is not None
    assert check_length("x" * 100, cfg) is None
    assert check_length("hi", cfg, prompt=True)[0] == "short_prompt"  # type: ignore[index]
    assert check_length("long enough prompt", cfg, prompt=True) is None


def test_language_heuristic() -> None:
    cfg = FilterConfig()
    assert check_language(GOOD, cfg) is None
    spanish = "Esta es una guia practica sobre el tema. Es importante definir objetivos claros."
    assert check_language(spanish, cfg) is not None
    assert check_language("你好，这是一个中文句子。", cfg) is not None
    assert check_language(spanish, FilterConfig(language=None)) is None
    assert check_language("Yes.", cfg) is None  # too short to judge


def test_repetition_filter() -> None:
    cfg = FilterConfig(max_repeat_ratio=0.3)
    loop = "the same words again and again " * 12
    assert check_repetition(loop, cfg) is not None
    assert check_repetition(GOOD, cfg) is None
    assert check_repetition(loop, FilterConfig(max_repeat_ratio=None)) is None


def test_score_threshold_depends_on_source() -> None:
    cfg = FilterConfig(min_reward=-5.0, min_judge=6.0)
    assert check_score(sample(score=-4.0), cfg) is None
    assert check_score(sample(score=-6.0), cfg)[0] == "low_score"  # type: ignore[index]
    assert check_score(sample(score=5.0, score_source="judge"), cfg)[0] == "low_score"  # type: ignore[index]
    assert check_score(sample(score=7.0, score_source="judge"), cfg) is None
    assert check_score(sample(score=None, score_source=None), cfg)[0] == "unscored"  # type: ignore[index]
    assert check_score(sample(score=-99.0), FilterConfig(min_reward=None)) is None


def test_grounding_filter() -> None:
    cfg = FilterConfig(min_grounding=0.6)
    ctx = "Exponential backoff doubles the waiting time after every failed attempt."
    grounded = sample(response="Backoff doubles the waiting time after each failed attempt.", context=ctx)
    assert check_grounding(grounded, cfg) is None
    invented = sample(response="Quantum entanglement powers distributed ledgers.", context=ctx)
    assert check_grounding(invented, cfg)[0] == "ungrounded"  # type: ignore[index]
    assert check_grounding(invented, FilterConfig()) is None
    assert check_grounding(sample(response="x", context=""), cfg) is None


def test_apply_filters_reports_first_failure_and_kind_rules() -> None:
    cfg = FilterConfig(min_chars=20, min_margin=1.0, min_difficulty=0.4, min_grounding=0.5)
    assert apply_filters(sample(), cfg, "instruction_diversity") is None
    assert apply_filters(sample(prompt="hi"), cfg, "instruction_diversity")[0] == "short_prompt"  # type: ignore[index]
    refusal = sample(response="I'm sorry, but I can't help with that request.")
    assert apply_filters(refusal, cfg, "instruction_diversity")[0] == "refusal"  # type: ignore[index]
    pair = sample(rejected="meh", rejected_score=0.5)
    assert apply_filters(pair, cfg, "preference_pairs")[0] == "low_margin"  # type: ignore[index]
    pair_ok = sample(rejected="meh", rejected_score=-3.0)
    assert apply_filters(pair_ok, cfg, "preference_pairs") is None
    assert apply_filters(sample(), cfg, "preference_pairs")[0] == "unscored"  # type: ignore[index]
    assert apply_filters(sample(difficulty=0.1), cfg, "eval_set")[0] == "too_easy"  # type: ignore[index]
    assert apply_filters(sample(difficulty=0.9), cfg, "eval_set") is None
    qa = sample(response="Totally unrelated words about nothing", context="backoff doubles delay")
    assert apply_filters(qa, cfg, "qa_from_docs")[0] == "ungrounded"  # type: ignore[index]
