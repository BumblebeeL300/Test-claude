"""A safe mini expression language for branch conditions.

Parsed with :mod:`ast` and walked against a strict whitelist. There is no ``eval``/``exec``:
the evaluator is a small interpreter over JSON-like values.

Allowed: literals (numbers, strings, ``True``/``False``/``None``), names (the node's input
variables), ``and`` / ``or`` / ``not``, comparisons (``== != < <= > >= in not in is is not``),
``+ - * / // %``, unary ``-``/``+``, ``a if cond else b``, list / tuple / dict literals and
subscripts / slices (``value["kind"]``, ``items[0]``, ``text[:5]``).

Rejected with a clear :class:`~nimlab.loom.errors.ExprError`: attribute access (``a.b``), calls
(``f(x)``), lambdas, comprehensions, f-strings, ``**``, assignment expressions and anything else.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from typing import Any

from .errors import ExprError

__all__ = ["check", "evaluate", "names"]

MAX_LENGTH = 500
MAX_NODES = 200
MAX_SEQUENCE = 10_000

_BIN: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_CMP: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_CONSTANTS: dict[str, Any] = {"true": True, "false": False, "null": None}
_HINTS: dict[type[ast.AST], str] = {
    ast.Attribute: "attribute access is not allowed; use subscripts like value['key']",
    ast.Call: "function calls are not allowed",
    ast.Lambda: "lambdas are not allowed",
    ast.ListComp: "comprehensions are not allowed",
    ast.SetComp: "comprehensions are not allowed",
    ast.DictComp: "comprehensions are not allowed",
    ast.GeneratorExp: "comprehensions are not allowed",
    ast.JoinedStr: "f-strings are not allowed",
    ast.NamedExpr: "assignment expressions are not allowed",
    ast.Await: "await is not allowed",
    ast.Starred: "starred expressions are not allowed",
}


def _parse(source: str) -> ast.expr:
    if not isinstance(source, str) or not source.strip():
        raise ExprError("empty expression")
    if len(source) > MAX_LENGTH:
        raise ExprError(f"expression is too long ({len(source)} > {MAX_LENGTH} characters)")
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"syntax error: {exc.msg} (column {exc.offset or 0})") from exc
    for count, node in enumerate(ast.walk(tree), 1):
        if count > MAX_NODES:
            raise ExprError("expression is too complex")
        _validate(node)
    return tree.body


def _validate(node: ast.AST) -> None:
    allowed = (
        ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.IfExp,
        ast.Constant, ast.Name, ast.Subscript, ast.Slice, ast.List, ast.Tuple, ast.Dict,
        ast.Load, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
        *_BIN, *_CMP,
    )  # fmt: skip
    if isinstance(node, allowed):
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (str, int, float, bool, type(None))
        ):
            raise ExprError(f"unsupported literal {node.value!r}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ExprError(f"name {node.id!r} is not allowed")
        return
    hint = _HINTS.get(type(node))
    if hint:
        raise ExprError(hint)
    if isinstance(node, (ast.operator, ast.unaryop)):
        raise ExprError(f"unsupported operator {type(node).__name__}")
    raise ExprError(f"unsupported syntax: {type(node).__name__}")


def check(source: str) -> None:
    """Raise :class:`ExprError` unless ``source`` is a valid expression."""
    _parse(source)


def names(source: str) -> list[str]:
    """Free variable names used by the expression, in order of first use."""
    found = [n for n in ast.walk(_parse(source)) if isinstance(n, ast.Name)]
    found.sort(key=lambda n: (n.lineno, n.col_offset))
    out: list[str] = []
    for n in found:
        if n.id not in out and n.id not in _CONSTANTS:
            out.append(n.id)
    return out


def evaluate(source: str, variables: Mapping[str, Any] | None = None) -> Any:
    """Evaluate ``source`` with ``variables``. Raises :class:`ExprError` on any problem."""
    return _Eval(variables or {}).visit(_parse(source))


class _Eval:
    def __init__(self, variables: Mapping[str, Any]) -> None:
        self.vars = variables

    def visit(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            if node.id not in self.vars:
                raise ExprError(f"unknown variable {node.id!r}")
            return self.vars[node.id]
        if isinstance(node, ast.BoolOp):
            result: Any = None
            for i, value in enumerate(node.values):
                result = self.visit(value)
                last = i == len(node.values) - 1
                if not last and bool(result) == isinstance(node.op, ast.Or):
                    return result
            return result
        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.Not):
                return not v
            try:
                return -v if isinstance(node.op, ast.USub) else +v
            except TypeError as exc:
                raise ExprError(f"cannot apply unary operator to {_t(v)}") from exc
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.Compare):
            left = self.visit(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = self.visit(comparator)
                try:
                    ok = _CMP[type(op)](left, right)
                except TypeError as exc:
                    raise ExprError(f"cannot compare {_t(left)} with {_t(right)}") from exc
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.visit(node.body if self.visit(node.test) else node.orelse)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.visit(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            if any(k is None for k in node.keys):
                raise ExprError("dict unpacking is not allowed")
            return {
                self._hashable(self.visit(k)): self.visit(v)
                for k, v in zip(node.keys, node.values, strict=True)
                if k is not None
            }
        if isinstance(node, ast.Subscript):
            return self._subscript(node)
        raise ExprError(f"unsupported syntax: {type(node).__name__}")

    @staticmethod
    def _hashable(key: Any) -> Any:
        if isinstance(key, (list, dict)):
            raise ExprError("dict keys must be scalars")
        return key

    def _binop(self, node: ast.BinOp) -> Any:
        a, b = self.visit(node.left), self.visit(node.right)
        op = type(node.op)
        if op not in _BIN:
            raise ExprError("unsupported operator")
        if op is ast.Mult:
            seq, times = (a, b) if isinstance(a, (str, list)) else (b, a)
            too_big = (
                isinstance(seq, (str, list))
                and isinstance(times, int)
                and len(seq) * max(times, 0) > MAX_SEQUENCE
            )
            if too_big:
                raise ExprError("result of '*' is too large")
        try:
            return _BIN[op](a, b)
        except (TypeError, ZeroDivisionError, ValueError) as exc:
            raise ExprError(f"cannot apply operator to {_t(a)} and {_t(b)}: {exc}") from exc

    def _subscript(self, node: ast.Subscript) -> Any:
        target = self.visit(node.value)
        sl = node.slice
        if isinstance(sl, ast.Slice):
            parts = [
                self.visit(p) if p is not None else None for p in (sl.lower, sl.upper, sl.step)
            ]
            key: Any = slice(*parts)
        else:
            key = self.visit(sl)
        if not isinstance(target, (str, list, dict)):
            raise ExprError(f"cannot index into {_t(target)}")
        try:
            return target[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExprError(f"cannot read {key!r} from {_t(target)}") from exc


def _t(value: Any) -> str:
    return type(value).__name__ if value is not None else "null"
