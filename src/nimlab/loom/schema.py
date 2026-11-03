"""The pipeline document: versioned, pydantic-validated JSON.

.. code-block:: json

    {"version": 1, "name": "demo", "title": "Demo", "description": "",
     "nodes": [{"id": "q", "type": "input", "params": {"type": "string"}, "pos": {"x": 0, "y": 0}},
               {"id": "a", "type": "prompt", "params": {"template": "Answer: {{q}}"}},
               {"id": "out", "type": "output"}],
     "edges": [{"from": "q.value", "to": "a.q"}, {"from": "a.text", "to": "out.value"}]}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import CycleError, PipelineError

__all__ = ["SCHEMA_VERSION", "Edge", "Node", "Pipeline", "Position", "find_cycle", "topo_levels"]

SCHEMA_VERSION = 1
_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,47}$")
_PORT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,47}$")


class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0


class Node(BaseModel):
    """One module on the rack."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    label: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    #: values for input ports that have no cable (``{"tone": "friendly"}``)
    defaults: dict[str, Any] = Field(default_factory=dict)
    retries: int = Field(default=0, ge=0, le=10)
    timeout: float | None = Field(default=None, gt=0, le=3600)
    #: ``None`` = the node type's default (model nodes cache, pure nodes do not)
    cache: bool | None = None
    pos: Position | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(
                f"invalid node id {v!r}: use letters, digits, '_' or '-' (max 48), "
                "starting with a letter or '_'"
            )
        return v


class Edge(BaseModel):
    """A patch cable: ``"node.port"`` -> ``"node.port"``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str = Field(alias="from")
    target: str = Field(alias="to")

    @field_validator("source", "target")
    @classmethod
    def _check_ref(cls, v: str) -> str:
        node, dot, port = v.partition(".")
        if not dot or not _ID.match(node) or not _PORT.match(port):
            raise ValueError(f"invalid endpoint {v!r}: expected 'node.port'")
        return v

    @property
    def src_node(self) -> str:
        return self.source.partition(".")[0]

    @property
    def src_port(self) -> str:
        return self.source.partition(".")[2]

    @property
    def dst_node(self) -> str:
        return self.target.partition(".")[0]

    @property
    def dst_port(self) -> str:
        return self.target.partition(".")[2]


class Pipeline(BaseModel):
    """A DAG of nodes joined by cables."""

    model_config = ConfigDict(extra="forbid")

    version: int = SCHEMA_VERSION
    name: str = "untitled"
    title: str = ""
    description: str = ""
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v < 1 or v > SCHEMA_VERSION:
            raise ValueError(
                f"unsupported pipeline schema version {v} "
                f"(this Loom reads versions 1..{SCHEMA_VERSION})"
            )
        return v

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not v.strip() or len(v) > 80:
            raise ValueError("name must be 1..80 characters")
        return v

    @model_validator(mode="after")
    def _check_graph(self) -> Pipeline:
        seen: set[str] = set()
        for node in self.nodes:
            if node.id in seen:
                raise ValueError(f"duplicate node id {node.id!r}")
            seen.add(node.id)
        for edge in self.edges:
            for end in (edge.src_node, edge.dst_node):
                if end not in seen:
                    raise ValueError(
                        f"edge {edge.source} -> {edge.target} refers to unknown node {end!r}"
                    )
        return self

    # ------------------------------------------------------------------ helpers
    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.dst_node == node_id]

    def outgoing(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.src_node == node_id]

    def topological_order(self) -> list[str]:
        """Node ids with every node after its dependencies. Raises :class:`CycleError`."""
        return [nid for level in topo_levels(self) for nid in level]

    # ------------------------------------------------------------------ io
    @classmethod
    def from_data(cls, data: Any) -> Pipeline:
        try:
            return cls.model_validate(data)
        except ValueError as exc:
            raise PipelineError(explain_error(exc)) from exc

    @classmethod
    def from_json(cls, text: str) -> Pipeline:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PipelineError(f"invalid JSON: {exc}") from exc
        return cls.from_data(data)

    @classmethod
    def load(cls, path: str | Path) -> Pipeline:
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelineError(f"cannot read {p}: {exc}") from exc
        return cls.from_json(text)

    def to_data(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_data(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json() + "\n", encoding="utf-8")
        return p


def explain_error(exc: ValueError) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        parts = []
        for err in errors()[:6]:
            loc = ".".join(str(p) for p in err.get("loc", ()))
            msg = str(err.get("msg", "")).removeprefix("Value error, ")
            parts.append(f"{loc}: {msg}" if loc else msg)
        return "invalid pipeline: " + "; ".join(parts)
    return f"invalid pipeline: {exc}"


def find_cycle(pipeline: Pipeline) -> list[str] | None:
    """One cycle as ``[a, b, c, a]`` or ``None``. Deterministic (follows node order)."""
    adj: dict[str, list[str]] = {n.id: [] for n in pipeline.nodes}
    for e in pipeline.edges:
        if e.dst_node not in adj[e.src_node]:
            adj[e.src_node].append(e.dst_node)
    state: dict[str, int] = {}
    stack: list[str] = []

    def dfs(start: str) -> list[str] | None:
        work = [(start, iter(adj[start]))]
        state[start] = 1
        stack.append(start)
        while work:
            node, it = work[-1]
            for nxt in it:
                if state.get(nxt) == 1:
                    return [*stack[stack.index(nxt) :], nxt]
                if nxt not in state:
                    state[nxt] = 1
                    stack.append(nxt)
                    work.append((nxt, iter(adj[nxt])))
                    break
            else:
                state[node] = 2
                stack.pop()
                work.pop()
        return None

    for n in pipeline.nodes:
        if n.id not in state:
            found = dfs(n.id)
            if found:
                return found
    return None


def topo_levels(pipeline: Pipeline) -> list[list[str]]:
    """Nodes grouped into dependency levels (all nodes in a level can run in parallel)."""
    cycle = find_cycle(pipeline)
    if cycle:
        raise CycleError(cycle)
    deps: dict[str, set[str]] = {n.id: set() for n in pipeline.nodes}
    for e in pipeline.edges:
        deps[e.dst_node].add(e.src_node)
    levels: list[list[str]] = []
    done: set[str] = set()
    remaining = [n.id for n in pipeline.nodes]
    while remaining:
        level = [nid for nid in remaining if deps[nid] <= done]
        levels.append(level)
        done.update(level)
        remaining = [nid for nid in remaining if nid not in done]
    return levels
