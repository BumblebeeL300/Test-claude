from __future__ import annotations

import pytest

from nimlab.loom import TemplateError
from nimlab.loom.template import render, variables


def test_variables_in_order_without_duplicates() -> None:
    assert variables("{{b}} then {{ a.x }} and {{b | upper}}") == ["b", "a"]


def test_renders_paths_and_lists() -> None:
    scope = {"plan": {"steps": ["one", "two"], "n": 2}, "name": "Ada"}
    assert render("Hi {{name}}: {{plan.steps.1}} / {{plan.n}}", scope) == "Hi Ada: two / 2"


def test_filters() -> None:
    scope = {"items": ["x", "y"], "t": "  Hello  ", "d": {"a": 1}}
    assert render("{{items | bullets}}", scope) == "- x\n- y"
    assert render("{{items | numbered}}", scope) == "1. x\n2. y"
    assert render("{{t | trim | upper}}", scope) == "HELLO"
    assert render("{{items | join}}", scope) == "x, y"
    assert render("{{items | len}}", scope) == "2"
    assert render("{{items | first}}", scope) == "x"
    assert '"a": 1' in render("{{d | json}}", scope)


def test_non_variable_braces_are_left_alone() -> None:
    assert render('JSON like {{"a": 1}} and {single}', {}) == 'JSON like {{"a": 1}} and {single}'


@pytest.mark.parametrize(
    "text, message",
    [
        ("{{missing}}", "unknown variable 'missing'"),
        ("{{a.b}}", "cannot read 'b'"),
        ("{{a | nope}}", "unknown filter"),
    ],
)
def test_errors(text, message) -> None:
    with pytest.raises(TemplateError, match=message):
        render(text, {"a": {"c": 1}})
