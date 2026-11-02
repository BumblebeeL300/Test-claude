"""Suite schema validation and the bundled suites."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nimlab.arena.graders import GradeContext, get_grader
from nimlab.arena.suites import (
    BUNDLED_DIR,
    CATEGORIES,
    GraderSpec,
    Suite,
    SuiteError,
    list_suites,
    load_suite,
    load_suites,
    resolve_suites,
)

EXPECTED = {
    "math_reasoning": ("math_reasoning", {"numeric"}),
    "coding": ("coding", {"code_exec"}),
    "factual_qa": ("factual_qa", {"exact", "regex", "contains_all"}),
    "instruction_following": ("instruction_following", {"checks"}),
    "json_compliance": ("json_compliance", {"json_schema"}),
    "summarization": ("summarization", {"llm_judge"}),
}
REPO = Path(__file__).resolve().parents[2]


def minimal(**over: object) -> dict:
    base = {
        "name": "t",
        "grader": {"type": "exact", "expected": "x"},
        "items": [{"id": "a", "prompt": "p"}],
    }
    base.update(over)
    return base


def test_all_six_bundled_suites_load() -> None:
    suites = load_suites()
    assert set(suites) == set(EXPECTED)
    for name, (category, graders) in EXPECTED.items():
        suite = suites[name]
        assert suite.category == category
        assert set(suite.grader_types()) == graders
        assert 8 <= len(suite.items) <= 12, f"{name} has {len(suite.items)} items"
        assert len({i.id for i in suite.items}) == len(suite.items)


def test_bundled_files_are_plain_json_and_valid() -> None:
    files = sorted(BUNDLED_DIR.glob("*.json"))
    assert len(files) == 6
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        Suite.model_validate(raw)
        assert all(i["prompt"].strip() for i in raw["items"])


def test_summarization_is_open_ended_and_has_battle_rubric() -> None:
    s = load_suites()["summarization"]
    assert s.open_ended and s.battle_rubric
    assert all(i.reference for i in s.items)


@pytest.mark.parametrize("name", sorted(set(EXPECTED) - {"summarization"}))  # judge-graded
async def test_reference_answers_pass_their_graders(name: str) -> None:
    """Every deterministic item's reference must score 1.0; a non-answer must not pass."""
    suite = load_suites()[name]
    for item in suite.items:
        spec = suite.grader_for(item)
        assert item.reference, f"{name}/{item.id} lacks a reference"
        ctx = GradeContext(output=item.reference, params=spec.params, prompt=item.prompt)
        good = await get_grader(spec.type).grade(ctx)
        assert good.passed and good.score == 1.0, f"{name}/{item.id}: {good.detail}"
        bad = await get_grader(spec.type).grade(
            GradeContext(output="I have no idea.", params=spec.params, prompt=item.prompt)
        )
        assert not bad.passed, f"{name}/{item.id} accepts a non-answer"


def test_suite_categories_are_known() -> None:
    assert {s.category for s in load_suites().values()} <= set(CATEGORIES)


def test_example_suite_loads() -> None:
    suite = load_suite(REPO / "examples" / "arena" / "custom_suite.json")
    assert suite.name == "support_triage" and len(suite.items) >= 4


# ------------------------------------------------------------------ validation errors


def test_valid_minimal_suite() -> None:
    suite = Suite.model_validate(minimal())
    assert suite.grader_for(suite.items[0]).type == "exact"
    assert suite.category == "general_chat"


def test_grader_spec_accepts_flat_and_nested_forms() -> None:
    flat = GraderSpec.model_validate({"type": "numeric", "expected": 3})
    nested = GraderSpec.model_validate({"type": "numeric", "params": {"expected": 3}})
    assert flat.params == nested.params == {"expected": 3}
    assert flat.flat() == {"type": "numeric", "expected": 3}


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"items": []}, "at least 1"),
        ({"name": "bad name!"}, "may only contain"),
        ({"category": "poetry"}, "unknown category"),
        ({"items": [{"id": "a", "prompt": "p"}, {"id": "a", "prompt": "q"}]}, "duplicate item id"),
        ({"grader": None, "items": [{"id": "a", "prompt": "p"}]}, "no grader"),
        ({"grader": {"type": "telepathy"}}, "unknown grader"),
        ({"grader": {"type": "numeric"}}, "expected"),
        ({"grader": {"type": "regex", "pattern": "("}}, "invalid regex"),
        ({"grader": {"type": "regex", "pattern": "a", "flags": "z"}}, "unknown regex flag"),
        ({"grader": {"type": "checks", "checks": [{"type": "nope"}]}}, "unknown check"),
        ({"grader": {"type": "checks", "checks": [{"type": "word_count"}]}}, "needs one of"),
        ({"grader": {"type": "code_exec", "tests": ["assert ("]}}, "not valid Python"),
        ({"grader": {"type": "json_schema", "schema": {"type": "blob"}}}, "unknown type"),
        ({"items": [{"id": "bad id", "prompt": "p"}]}, "item id"),
        ({"items": [{"id": "a", "prompt": ""}]}, "at least 1"),
        ({"unexpected": 1}, "Extra inputs"),
    ],
)
def test_invalid_suites_are_rejected(override: dict, fragment: str) -> None:
    with pytest.raises(ValidationError) as err:
        Suite.model_validate(minimal(**override))
    assert fragment in str(err.value)


def test_load_suite_error_messages(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="not found"):
        load_suite(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SuiteError, match="invalid JSON"):
        load_suite(bad)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(minimal(category="nope")), encoding="utf-8")
    with pytest.raises(SuiteError, match="unknown category"):
        load_suite(invalid)


def test_load_suites_rejects_duplicate_names(tmp_path: Path) -> None:
    for fname in ("a.json", "b.json"):
        (tmp_path / fname).write_text(json.dumps(minimal()), encoding="utf-8")
    with pytest.raises(SuiteError, match="duplicate suite name"):
        load_suites(tmp_path)
    with pytest.raises(SuiteError, match="directory not found"):
        load_suites(tmp_path / "nope")


# ------------------------------------------------------------------ resolution & helpers


def test_resolve_suites_variants(tmp_path: Path) -> None:
    assert len(resolve_suites("all")) == 6
    assert [s.name for s in resolve_suites("coding,factual_qa")] == ["coding", "factual_qa"]
    assert [s.name for s in resolve_suites(["coding"])] == ["coding"]
    custom = tmp_path / "mine.json"
    custom.write_text(json.dumps(minimal(name="mine")), encoding="utf-8")
    assert [s.name for s in resolve_suites(str(custom))] == ["mine"]
    assert [s.name for s in resolve_suites("all", extra_dirs=[tmp_path])][-1] == "mine"
    with pytest.raises(SuiteError, match="unknown suite 'nope'"):
        resolve_suites("nope")


def test_limit_and_item_hash() -> None:
    suite = list_suites()["math_reasoning"]
    assert len(resolve_suites("math_reasoning", limit=3)[0].items) == 3
    assert suite.limited(None) is suite
    item = suite.items[0]
    h1 = suite.item_hash(item)
    changed = suite.model_copy(
        update={"items": [item.model_copy(update={"prompt": item.prompt + "!"}), *suite.items[1:]]}
    )
    assert changed.item_hash(changed.items[0]) != h1
    assert suite.item_hash(item) == h1
