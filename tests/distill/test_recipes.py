"""Recipe validation and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nimlab.distill import Recipe, RecipeError, get_recipe, list_recipes
from nimlab.distill.recipes import BUNDLED_DIR, FilterConfig

KINDS = {"qa_from_docs", "instruction_diversity", "preference_pairs", "eval_set"}


def test_four_bundled_recipes_cover_every_kind() -> None:
    recipes = list_recipes()
    assert {r.kind for r in recipes.values()} == KINDS
    assert set(recipes) == KINDS  # bundled names equal their kinds
    for path in BUNDLED_DIR.glob("*.json"):
        assert Recipe.load(path).name == path.stem


def test_preference_recipe_is_forced_to_pairs_and_dpo() -> None:
    rec = Recipe.parse({"name": "p", "kind": "preference_pairs"})
    assert rec.responses_per_prompt == 2
    assert rec.default_format == "dpo"


def test_get_recipe_by_name_path_and_instance(tmp_path: Path) -> None:
    rec = get_recipe("eval_set")
    assert rec.kind == "eval_set"
    assert get_recipe(rec) is rec
    custom = tmp_path / "mine.json"
    custom.write_text(json.dumps({"name": "mine", "kind": "instruction_diversity"}))
    assert get_recipe(custom).name == "mine"
    assert get_recipe(str(custom)).name == "mine"


def test_list_recipes_includes_extra_dirs(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(json.dumps({"name": "extra", "kind": "eval_set"}))
    assert "extra" in list_recipes([tmp_path])


def test_unknown_recipe_lists_alternatives() -> None:
    with pytest.raises(RecipeError, match="available: .*eval_set"):
        get_recipe("nope")


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"name": "bad name!"}, "name"),
        ({"name": ""}, "name"),
        ({"kind": "mystery"}, "kind"),
        ({"temperatures": []}, "temperatures"),
        ({"temperatures": [0.5, 3.0]}, "between 0 and 2"),
        ({"unknown_field": 1}, "unknown_field"),
        ({"chunk_chars": 300, "chunk_overlap": 400}, "chunk_overlap"),
        ({"max_tokens": 4}, "max_tokens"),
        ({"dedup": {"threshold": 1.5}}, "threshold"),
        ({"dedup": {"threshold": 0}}, "threshold"),
        ({"filters": {"min_chars": 500, "max_chars": 100}}, "min_chars"),
        ({"filters": {"language": "fr"}}, "language"),
        ({"decontaminate": {"min_overlap": 0}}, "min_overlap"),
        ({"default_format": "csv"}, "default_format"),
    ],
)
def test_invalid_recipes_are_rejected(patch: dict[str, Any], message: str) -> None:
    base: dict[str, Any] = {"name": "ok", "kind": "instruction_diversity"}
    with pytest.raises(RecipeError, match=message):
        Recipe.parse({**base, **patch})


def test_load_reports_unreadable_and_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="cannot read"):
        Recipe.load(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(RecipeError, match="not valid JSON"):
        Recipe.load(broken)


def test_defaults_are_sensible() -> None:
    f = FilterConfig()
    assert f.min_chars < f.max_chars
    assert f.language == "en" and f.drop_refusals
    rec = Recipe.parse({"name": "d", "kind": "eval_set"})
    assert rec.dedup.near and rec.dedup.exact and 0 < rec.dedup.threshold <= 1
