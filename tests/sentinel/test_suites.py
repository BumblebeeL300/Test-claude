"""Every bundled suite validates, and the suite schema rejects unsafe or malformed probes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nimlab.sentinel.models import ATTACK_CATEGORIES, CATEGORIES, Probe, ProbeSuite
from nimlab.sentinel.suites import (
    BUNDLED_DIR,
    SuiteError,
    list_suites,
    load_suite,
    resolve_suites,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "sentinel" / "custom_suite.json"


def test_one_bundled_suite_per_category() -> None:
    suites = list_suites()
    assert {s.category for s in suites.values()} == set(CATEGORIES)
    assert len(suites) == len(CATEGORIES)


@pytest.mark.parametrize("path", sorted(BUNDLED_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_bundled_suite_file_validates(path: Path) -> None:
    suite = load_suite(path)
    assert suite.name == path.stem
    assert 8 <= len(suite.probes) <= 10
    ids = [p.id for p in suite.probes]
    assert len(set(ids)) == len(ids)


def test_category_contracts() -> None:
    for suite in list_suites().values():
        for p in suite.probes:
            if suite.category in ("prompt_injection", "jailbreak"):
                assert p.success_marker and p.success_marker.startswith("CANARY-")
            if suite.category == "system_prompt_extraction":
                assert p.canary and p.system and p.canary in p.system
                assert p.canary.lower() not in p.user.lower()
            if suite.category == "pii_leakage":
                assert p.secrets and any(s in (p.system or "") for s in p.secrets)
            if suite.category in ("harmful_refusal", "jailbreak"):
                assert "[REDACTED" in p.user
            if suite.category == "over_refusal":
                assert p.benign


def test_probes_stay_placeholder_only() -> None:
    """No operational content: harmful requests are placeholders and prompts stay short."""
    for suite in list_suites().values():
        for p in suite.probes:
            assert len(p.user) < 900 and len(p.system or "") < 900
            if suite.category == "harmful_refusal":
                assert p.user.count("[REDACTED HARMFUL REQUEST:") == 1


def test_attack_categories_exclude_over_refusal() -> None:
    assert "over_refusal" not in ATTACK_CATEGORIES and len(ATTACK_CATEGORIES) == 6


def test_validation_errors() -> None:
    def build(category: str, **probe: object) -> ProbeSuite:
        return ProbeSuite(name="x", category=category, probes=[Probe(id="p1", user="hi", **probe)])  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="unknown category"):
        build("nonsense")
    with pytest.raises(ValidationError, match="success_marker"):
        build("prompt_injection")
    with pytest.raises(ValidationError, match="must have expect"):
        build("harmful_refusal", expect="answer")
    with pytest.raises(ValidationError, match=r"\[REDACTED"):
        build("harmful_refusal", expect="refuse")
    with pytest.raises(ValidationError, match="canary"):
        build("system_prompt_extraction")
    with pytest.raises(ValidationError, match="synthetic secrets"):
        build("pii_leakage")
    with pytest.raises(ValidationError, match="duplicate"):
        ProbeSuite(
            name="x",
            category="over_refusal",
            probes=[
                Probe(id="a", user="q", expect="answer"),
                Probe(id="a", user="r", expect="answer"),
            ],
        )
    with pytest.raises(ValidationError, match="at least one"):
        ProbeSuite(name="x", category="over_refusal", probes=[])
    with pytest.raises(ValidationError, match="canary must not appear"):
        ProbeSuite(
            name="x",
            category="system_prompt_extraction",
            probes=[
                Probe(
                    id="a", user="say CANARY-1234", system="code CANARY-1234", canary="CANARY-1234"
                )
            ],
        )


def test_resolve_suites_variants(tmp_path: Path) -> None:
    assert len(resolve_suites("all")) == len(CATEGORIES)
    assert [s.name for s in resolve_suites("jailbreak,pii_leakage")] == ["jailbreak", "pii_leakage"]
    assert len(resolve_suites(["jailbreak", "jailbreak"])) == 1
    limited = resolve_suites("all", limit=2)
    assert all(len(s.probes) == 2 for s in limited)
    with pytest.raises(SuiteError, match="unknown suite"):
        resolve_suites("nope")


def test_custom_suite_file_and_dir(tmp_path: Path) -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    custom = tmp_path / "mine.json"
    custom.write_text(json.dumps(data), encoding="utf-8")
    assert resolve_suites(str(custom))[0].name == data["name"]
    assert data["name"] in list_suites([tmp_path])
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SuiteError, match="cannot read suite"):
        list_suites([tmp_path])
    with pytest.raises(SuiteError, match="not found"):
        load_suite(tmp_path / "missing.json")
    with pytest.raises(SuiteError, match="directory not found"):
        list_suites([tmp_path / "nope"])
