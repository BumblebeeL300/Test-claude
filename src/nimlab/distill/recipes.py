"""Recipes: declarative JSON descriptions of a dataset, validated with pydantic.

A recipe says *what* to build (its ``kind`` selects the pipeline shape) and *how strict* to be (the
score, length, language, dedup and decontamination settings). Four recipes ship in
``nimlab/distill/recipes/*.json``; any JSON file with the same schema works
(``distill run my.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

__all__ = [
    "BUNDLED_DIR",
    "DecontamConfig",
    "DedupConfig",
    "FilterConfig",
    "Recipe",
    "RecipeError",
    "RecipeKind",
    "get_recipe",
    "list_recipes",
]

BUNDLED_DIR = Path(__file__).parent / "recipes"

RecipeKind = Literal["qa_from_docs", "instruction_diversity", "preference_pairs", "eval_set"]
Role_ = Literal["chat", "reasoning"]


class RecipeError(ValueError):
    """A recipe is missing, unreadable or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FilterConfig(_Strict):
    """Quality filters. ``None`` switches a check off."""

    #: Minimum reward-model score (Nemotron reward is unbounded; good answers sit around 0).
    min_reward: float | None = -8.0
    #: Minimum LLM-judge score (0-10), used only for samples the reward model could not score.
    min_judge: float | None = 6.0
    min_chars: int = Field(default=40, ge=0)
    max_chars: int = Field(default=6000, ge=1)
    min_prompt_chars: int = Field(default=12, ge=0)
    drop_refusals: bool = True
    #: ``"en"`` drops text that does not look English; ``None`` disables the check.
    language: Literal["en"] | None = "en"
    #: Drop responses whose word 3-grams repeat more than this share.
    max_repeat_ratio: float | None = Field(default=0.4, ge=0.0, le=1.0)
    #: Preference pairs: minimum reward gap between chosen and rejected.
    min_margin: float = Field(default=1.0, ge=0.0)
    #: Grounded QA: minimum share of answer words that appear in the source excerpt.
    min_grounding: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Eval sets: minimum difficulty estimate (0 = trivial, 1 = very hard).
    min_difficulty: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bounds(self) -> FilterConfig:
        if self.min_chars >= self.max_chars:
            raise ValueError("min_chars must be smaller than max_chars")
        return self


class DedupConfig(_Strict):
    exact: bool = True
    near: bool = True
    #: Cosine similarity at or above which two samples count as near-duplicates.
    threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    #: Which text is compared.
    field: Literal["prompt", "response", "both"] = "prompt"


class DecontamConfig(_Strict):
    enabled: bool = True
    #: Word n-gram length.
    ngram: int = Field(default=8, ge=1, le=50)
    #: A sample is contaminated when this share of its n-grams occurs in the eval set.
    min_overlap: float = Field(default=0.5, gt=0.0, le=1.0)
    check: Literal["prompt", "both"] = "prompt"


class Recipe(_Strict):
    name: str
    kind: RecipeKind
    description: str = ""
    #: Model role that expands topics and writes prompts.
    seed_role: Role_ = "chat"
    #: Model role used for answers when ``--models`` is not given.
    generator_role: Role_ = "chat"
    #: Sampling temperatures, cycled over the samples so the dataset mixes tight and loose answers.
    temperatures: list[float] = Field(default_factory=lambda: [0.3, 0.7, 1.0], min_length=1)
    max_tokens: int = Field(default=700, ge=16, le=8192)
    system_prompt: str | None = None
    #: Topic-tree shape for kinds that expand topics.
    domains: list[str] = Field(default_factory=list)
    topics_per_domain: int = Field(default=4, ge=1, le=50)
    subtopics_per_topic: int = Field(default=4, ge=1, le=50)
    #: Document chunking for ``qa_from_docs``.
    chunk_chars: int = Field(default=900, ge=200, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    #: Number of generations compared per prompt (``preference_pairs`` uses 2).
    responses_per_prompt: int = Field(default=1, ge=1, le=2)
    #: Default dataset format (``openai-chat``, ``alpaca``, ``sharegpt`` or ``dpo``).
    default_format: Literal["openai-chat", "alpaca", "sharegpt", "dpo"] = "openai-chat"
    #: Add a ``metadata`` object (id, score, model, topic...) to every exported record.
    include_meta: bool = False
    filters: FilterConfig = Field(default_factory=FilterConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    decontaminate: DecontamConfig = Field(default_factory=DecontamConfig)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("name may only contain letters, digits, '_' and '-'")
        return v

    @field_validator("temperatures")
    @classmethod
    def _temps(cls, v: list[float]) -> list[float]:
        if any(t < 0 or t > 2 for t in v):
            raise ValueError("temperatures must be between 0 and 2")
        return v

    @model_validator(mode="after")
    def _kind_rules(self) -> Recipe:
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("chunk_overlap must be smaller than chunk_chars")
        if self.kind == "preference_pairs":
            self.responses_per_prompt = 2
            if self.default_format == "openai-chat":
                self.default_format = "dpo"
        return self

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        p = Path(path)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RecipeError(f"cannot read recipe {p}: {exc}") from exc
        except ValueError as exc:
            raise RecipeError(f"{p} is not valid JSON: {exc}") from exc
        return cls.parse(raw, source=str(p))

    @classmethod
    def parse(cls, raw: Any, *, source: str = "recipe") -> Recipe:
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(x) for x in e['loc']) or '(root)'}: {e['msg']}"
                for e in exc.errors()
            )
            raise RecipeError(f"invalid {source}: {problems}") from exc


def list_recipes(extra_dirs: list[Path] | None = None) -> dict[str, Recipe]:
    """Bundled recipes (plus any ``*.json`` in ``extra_dirs``), keyed by name."""
    found: dict[str, Recipe] = {}
    for directory in [BUNDLED_DIR, *(extra_dirs or [])]:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            recipe = Recipe.load(path)
            found[recipe.name] = recipe
    return found


def get_recipe(ref: str | Path | Recipe, extra_dirs: list[Path] | None = None) -> Recipe:
    """A recipe from a name (bundled), a path to a JSON file, or an existing :class:`Recipe`."""
    if isinstance(ref, Recipe):
        return ref
    text = str(ref)
    path = Path(text)
    if text.endswith(".json") or path.is_file():
        return Recipe.load(path)
    recipes = list_recipes(extra_dirs)
    if text not in recipes:
        raise RecipeError(f"unknown recipe {text!r}; available: {', '.join(sorted(recipes))}")
    return recipes[text]
