from __future__ import annotations

import inspect

import pytest

from nimlab.loom import ExprError, evaluate_expression
from nimlab.loom import expr as expr_module
from nimlab.loom.expr import check, names

V = {"value": {"category": "billing", "score": 7, "tags": ["a", "b"]}, "n": 5, "text": "Refund please"}


@pytest.mark.parametrize(
    "source, expected",
    [
        ("1 + 2 * 3", 7),
        ("(1 + 2) * 3", 9),
        ("10 // 3", 3),
        ("10 % 4", 2),
        ("7 / 2", 3.5),
        ("-n", -5),
        ("n > 3", True),
        ("1 < n <= 5", True),
        ("1 < n < 5", False),
        ("n == 5 and not False", True),
        ("False or n", 5),
        ("n if n > 9 else 'small'", "small"),
        ("value['category'] == 'billing'", True),
        ("value['score'] >= 7 and 'a' in value['tags']", True),
        ("'z' not in value['tags']", True),
        ("value['tags'][-1]", "b"),
        ("value['tags'][0:1]", ["a"]),
        ("text[:6]", "Refund"),
        ("'Refund' in text", True),
        ("[1, 2] + [3]", [1, 2, 3]),
        ("{'a': 1}['a']", 1),
        ("None is None", True),
        ("true and not false", True),
        ("'ab' * 3", "ababab"),
    ],
)
def test_evaluates(source, expected) -> None:
    assert evaluate_expression(source, V) == expected


def test_short_circuit_skips_the_unsafe_side() -> None:
    assert evaluate_expression("n > 3 or missing", V) is True  # `missing` is never evaluated
    assert evaluate_expression("n > 9 and missing", V) is False


@pytest.mark.parametrize(
    "source, hint",
    [
        ("value.keys", "attribute access"),
        ("text.upper", "attribute access"),
        ("().__class__", "attribute access"),
        ("value.__class__.__mro__", "attribute access"),
        ("len(text)", "function calls"),
        ("__import__('os')", "function calls"),
        ("print(1)", "function calls"),
        ("(lambda: 1)()", "lambdas|function calls"),
        ("[x for x in text]", "comprehensions"),
        ("{x: 1 for x in text}", "comprehensions"),
        ("f'{n}'", "f-strings"),
        ("n ** 2", "unsupported operator"),
        ("2 ** 99999", "unsupported operator"),
        ("(y := 1)", "assignment"),
        ("*value", "starred|syntax"),
        ("__builtins__", "not allowed"),
        ("n; 1", "syntax"),
        ("", "empty"),
        ("n +", "syntax error"),
        ("1 << 3", "unsupported operator"),
        ("n & 1", "unsupported operator"),
        ("~n", "unsupported operator"),
        ("await n", "await|syntax"),
        ("b'bytes'", "unsupported literal"),
    ],
)
def test_rejects(source, hint) -> None:
    with pytest.raises(ExprError, match=hint):
        evaluate_expression(source, V)
    with pytest.raises(ExprError):
        check(source)


def test_runtime_errors_are_expr_errors() -> None:
    with pytest.raises(ExprError, match="unknown variable 'nope'"):
        evaluate_expression("nope", V)
    with pytest.raises(ExprError, match="cannot read"):
        evaluate_expression("value['missing']", V)
    with pytest.raises(ExprError, match="cannot compare"):
        evaluate_expression("text < 3", V)
    with pytest.raises(ExprError, match="cannot index"):
        evaluate_expression("n[0]", V)
    with pytest.raises(ExprError):
        evaluate_expression("1 / 0", V)


def test_size_limits() -> None:
    with pytest.raises(ExprError, match="too large"):
        evaluate_expression("'x' * 100000", V)
    with pytest.raises(ExprError, match="too long"):
        check("1 + " * 200 + "1")
    with pytest.raises(ExprError, match="too complex"):
        check("+".join(["n"] * 90))


def test_names_lists_free_variables_in_order() -> None:
    assert names("b > a and a != c") == ["b", "a", "c"]
    assert names("value['x'] == true") == ["value"]


def test_module_never_calls_eval_or_exec() -> None:
    source = inspect.getsource(expr_module)
    for forbidden in ("eval(", "exec(", "compile(", "__import__("):
        assert forbidden not in source
