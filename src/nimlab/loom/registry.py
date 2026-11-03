"""The pluggable node-type registry.

A :class:`NodeType` describes one kind of module: its parameter schema (JSON Schema properties,
used for validation and the Studio inspector), its input/output jacks and the async function that
runs it. Register your own with :func:`register_node_type`.

Input jacks come from three places: ``inputs`` (fixed), ``input_vars`` (the ``{{variables}}`` of
template params, or the identifiers of expression params) and an optional ``ports_hook`` for
types whose jacks depend on their params in other ways (``input``, ``map``).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import expr as _expr
from . import template as _tpl
from .errors import ExprError, PipelineError

if TYPE_CHECKING:
    from .context import NodeContext, NodeResult

__all__ = [
    "NodeType",
    "Port",
    "VarSource",
    "apply_defaults",
    "get_node_type",
    "node_types",
    "ports_for",
    "register_node_type",
    "unregister_node_type",
]

Runner = Callable[["NodeContext"], Awaitable["NodeResult"]]
PortsHook = Callable[[dict[str, Any]], tuple[list["Port"], list["Port"]]]
_PATH = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")


@dataclass(frozen=True)
class Port:
    """One jack. ``multi`` jacks accept several cables and receive a list (in cable order)."""

    name: str
    type: str = "any"
    required: bool = True
    multi: bool = False
    doc: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.type}
        if not self.required:
            out["required"] = False
        if self.multi:
            out["multi"] = True
        if self.doc:
            out["doc"] = self.doc
        return out


@dataclass(frozen=True)
class VarSource:
    """A param whose text defines input jacks: ``kind`` is ``template`` or ``expr``."""

    path: str
    kind: str = "template"


@dataclass
class NodeType:
    name: str
    title: str
    category: str
    description: str
    run: Runner
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    inputs: tuple[Port, ...] = ()
    input_vars: tuple[VarSource, ...] = ()
    input_exclude: tuple[str, ...] = ()
    outputs: tuple[Port, ...] = ()
    #: run as soon as one input is live (merge/join) instead of skipping on a dead input
    any_input: bool = False
    #: cache results on disk by default
    cacheable: bool = False
    glyph: str = "•"
    ports_hook: PortsHook | None = None

    def describe(self) -> dict[str, Any]:
        """JSON-able description for the Studio palette and inspector."""
        ins, outs = self.static_ports()
        return {
            "type": self.name,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "glyph": self.glyph,
            "params": {
                "type": "object",
                "properties": self.params,
                "required": list(self.required),
            },
            "inputs": [p.to_dict() for p in ins],
            "input_vars": [{"path": v.path, "kind": v.kind} for v in self.input_vars],
            "input_exclude": list(self.input_exclude),
            "outputs": [p.to_dict() for p in outs],
            "any_input": self.any_input,
            "cacheable": self.cacheable,
        }

    def static_ports(self) -> tuple[list[Port], list[Port]]:
        if self.ports_hook:
            return self.ports_hook({})
        return list(self.inputs), list(self.outputs)


_REGISTRY: dict[str, NodeType] = {}


def register_node_type(node_type: NodeType, *, replace: bool = False) -> NodeType:
    if node_type.name in _REGISTRY and not replace:
        raise ValueError(f"node type {node_type.name!r} is already registered")
    _REGISTRY[node_type.name] = node_type
    return node_type


def unregister_node_type(name: str) -> None:
    _REGISTRY.pop(name, None)


def get_node_type(name: str) -> NodeType:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise PipelineError(f"unknown node type {name!r} (known: {known})") from None


def node_types() -> list[NodeType]:
    return list(_REGISTRY.values())


def _dig(params: dict[str, Any], path: str) -> Any:
    cur: Any = params
    for seg in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(seg)
    return cur


def ports_for(node_type: NodeType, params: dict[str, Any]) -> tuple[list[Port], list[Port]]:
    """Concrete ``(inputs, outputs)`` jacks of a node with these params."""
    if node_type.ports_hook:
        ins, outs = node_type.ports_hook(params)
    else:
        ins, outs = list(node_type.inputs), list(node_type.outputs)
    taken = {p.name for p in ins}
    for src in node_type.input_vars:
        text = _dig(params, src.path)
        if not isinstance(text, str):
            continue
        if src.kind == "template":
            found = _tpl.variables(text)
        else:
            try:
                found = _expr.names(text)
            except ExprError:
                found = []
        for name in found:
            if name not in taken and name not in node_type.input_exclude:
                taken.add(name)
                ins.append(Port(name, "any"))
    return ins, outs


def apply_defaults(node_type: NodeType, params: dict[str, Any]) -> dict[str, Any]:
    """``params`` with the schema's defaults filled in (does not mutate the input)."""
    out = dict(params)
    for key, spec in node_type.params.items():
        if key not in out and "default" in spec:
            default = spec["default"]
            out[key] = (
                list(default)
                if isinstance(default, list)
                else (dict(default) if isinstance(default, dict) else default)
            )
    return out
