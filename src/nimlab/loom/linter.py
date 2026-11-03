"""Static validation: everything that can be known without calling a model.

Codes (severity): ``schema`` (e) ``cycle`` (e) ``unknown-type`` (e) ``param`` (e/w)
``expr`` (e) ``template`` (e) ``unknown-port`` (e) ``multiple-edges`` (e) ``unbound`` (e)
``type-mismatch`` (e/w) ``path-type`` (w) ``unreachable`` (w) ``no-output`` (e)
``duplicate-name`` (e) ``unknown-function`` (e) ``http-disabled`` (w) ``unused-default`` (w)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import expr as _expr
from . import template as _tpl
from .errors import ExprError
from .functions import get_function
from .jsonschema_lite import validate
from .nodes import MAP_STEP_FORBIDDEN, http_allowlist
from .registry import NodeType, get_node_type, node_types, ports_for
from .schema import Edge, Node, Pipeline, find_cycle
from .datatypes import compat

__all__ = ["Issue", "LintReport", "lint"]


@dataclass(frozen=True)
class Issue:
    severity: str  # error | warning
    code: str
    message: str
    node: str | None = None
    port: str | None = None

    def format(self) -> str:
        where = f"{self.node}" + (f".{self.port}" if self.port else "") if self.node else "pipeline"
        return f"{where}: {self.message} [{self.code}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "node": self.node,
            "port": self.port,
        }


@dataclass
class LintReport:
    issues: list[Issue] = field(default_factory=list)
    #: concrete jacks per node, for editors: ``{id: {"inputs": [...], "outputs": [...]}}``
    ports: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
            "ports": self.ports,
        }


class _Linter:
    def __init__(self, pipeline: Pipeline, http_allow: list[str]) -> None:
        self.p = pipeline
        self.issues: list[Issue] = []
        self.ports: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self.http_allow = http_allow
        self.types: dict[str, NodeType] = {}

    def add(self, sev: str, code: str, msg: str, node: str | None = None, port: str | None = None) -> None:
        self.issues.append(Issue(sev, code, msg, node, port))

    # ------------------------------------------------------------------ params
    def check_params(self, node_id: str, ntype: NodeType, params: dict[str, Any], prefix: str = "") -> None:
        for key in ntype.required:
            if key not in params or params[key] in (None, ""):
                self.add("error", "param", f"{prefix}missing required parameter {key!r}", node_id)
        for key, value in params.items():
            spec = ntype.params.get(key)
            if spec is None:
                self.add("warning", "param", f"{prefix}unknown parameter {key!r} for {ntype.name}", node_id)
                continue
            for problem in validate(value, {k: v for k, v in spec.items() if k != "default"}, key):
                self.add("error", "param", f"{prefix}{problem}", node_id)
        # text params: template syntax and expressions
        for src in ntype.input_vars:
            text = params
            for seg in src.path.split("."):
                text = text.get(seg) if isinstance(text, dict) else None  # type: ignore[assignment]
            if not isinstance(text, str):
                continue
            if src.kind == "expr":
                try:
                    _expr.check(text)
                except ExprError as exc:
                    self.add("error", "expr", f"{prefix}{src.path}: {exc}", node_id)
            else:
                for name in _tpl.filters_used(text):
                    if name not in _tpl.FILTERS:
                        self.add("error", "template", f"{prefix}unknown filter {name!r}", node_id)
        if ntype.name == "python_fn" and params.get("fn") and get_function(str(params["fn"])) is None:
            self.add("error", "unknown-function", f"{prefix}function {params['fn']!r} is not registered", node_id)
        if ntype.name == "http_get" and not self.http_allow:
            self.add(
                "warning", "http-disabled",
                "http_get is disabled (set NIMLAB_LOOM_HTTP_ALLOW to allow hosts)", node_id,
            )  # fmt: skip
        if ntype.name == "map":
            self.check_map_step(node_id, params)

    def check_map_step(self, node_id: str, params: dict[str, Any]) -> None:
        step = params.get("step")
        if not isinstance(step, dict):
            return
        try:
            st = get_node_type(str(step.get("type") or ""))
        except Exception:
            self.add("error", "unknown-type", f"map step: unknown node type {step.get('type')!r}", node_id)
            return
        if st.name in MAP_STEP_FORBIDDEN:
            self.add("error", "param", f"map step cannot be a {st.name!r} node", node_id)
            return
        self.check_params(node_id, st, dict(step.get("params") or {}), prefix="step: ")

    # ------------------------------------------------------------------ main passes
    def run(self) -> LintReport:
        cycle = find_cycle(self.p)
        if cycle:
            self.add("error", "cycle", "cycle: " + " -> ".join(cycle), cycle[0])
        node_ports: dict[str, tuple[list[Any], list[Any]]] = {}
        for node in self.p.nodes:
            try:
                ntype = get_node_type(node.type)
            except Exception:
                known = ", ".join(sorted(t.name for t in node_types()))
                self.add("error", "unknown-type", f"unknown node type {node.type!r} (known: {known})", node.id)
                continue
            self.types[node.id] = ntype
            self.check_params(node.id, ntype, node.params)
            ins, outs = ports_for(ntype, node.params)
            node_ports[node.id] = (ins, outs)
            self.ports[node.id] = {"inputs": [p.to_dict() for p in ins], "outputs": [p.to_dict() for p in outs]}
            self.check_defaults(node, ins)
        for edge in self.p.edges:
            self.check_edge(edge, node_ports)
        for node in self.p.nodes:
            if node.id in node_ports:
                self.check_inputs(node, node_ports[node.id][0])
        self.check_names()
        self.check_reachability()
        return LintReport(self.issues, self.ports)

    def check_defaults(self, node: Node, ins: list[Any]) -> None:
        names = {p.name for p in ins}
        for key in node.defaults:
            if key not in names:
                self.add("warning", "unused-default", f"default for unknown input {key!r}", node.id, key)

    def check_edge(self, edge: Edge, node_ports: dict[str, tuple[list[Any], list[Any]]]) -> None:
        if edge.src_node not in node_ports or edge.dst_node not in node_ports:
            return
        outs = {p.name: p for p in node_ports[edge.src_node][1]}
        ins = {p.name: p for p in node_ports[edge.dst_node][0]}
        ok = True
        if edge.src_port not in outs:
            self.add("error", "unknown-port", f"cable from unknown output {edge.source!r}", edge.src_node, edge.src_port)
            ok = False
        if edge.dst_port not in ins:
            self.add("error", "unknown-port", f"cable into unknown input {edge.target!r}", edge.dst_node, edge.dst_port)
            ok = False
        if not ok:
            return
        src, dst = outs[edge.src_port], ins[edge.dst_port]
        verdict = compat(src.type, dst.type)
        if verdict != "ok":
            self.add(
                "error" if verdict == "error" else "warning",
                "type-mismatch",
                f"{edge.source} ({src.type}) -> {edge.target} ({dst.type})"
                + (": incompatible types" if verdict == "error" else ": converted at run time"),
                edge.dst_node,
                edge.dst_port,
            )

    def check_inputs(self, node: Node, ins: list[Any]) -> None:
        ntype = self.types[node.id]
        for port in ins:
            edges = [e for e in self.p.incoming(node.id) if e.dst_port == port.name]
            if len(edges) > 1 and not port.multi:
                self.add("error", "multiple-edges", f"input {port.name!r} has {len(edges)} cables; only one is allowed", node.id, port.name)
            if not edges and port.name not in node.defaults and port.required:
                self.add("error", "unbound", f"unknown variable: input {port.name!r} has no cable and no default", node.id, port.name)
        # dotted paths ({{plan.items}}) need a structured source
        for src in ntype.input_vars:
            text: Any = node.params
            for seg in src.path.split("."):
                text = text.get(seg) if isinstance(text, dict) else None
            if src.kind != "template" or not isinstance(text, str):
                continue
            for path in _tpl.paths(text):
                head, _, rest = path.partition(".")
                if not rest:
                    continue
                for e in self.p.incoming(node.id):
                    if e.dst_port != head or e.src_node not in self.ports:
                        continue
                    src_t = next((o["type"] for o in self.ports[e.src_node]["outputs"] if o["name"] == e.src_port), "any")
                    if src_t in ("text", "number", "bool", "vectors"):
                        self.add("warning", "path-type", f"{{{{{path}}}}}: {e.source} is {src_t}, not structured", node.id, head)

    def check_names(self) -> None:
        for kind in ("input", "output"):
            seen: dict[str, str] = {}
            for n in self.p.nodes:
                if n.type != kind:
                    continue
                name = str(n.params.get("name") or n.id)
                if name in seen:
                    self.add("error", "duplicate-name", f"{kind} name {name!r} is also used by {seen[name]!r}", n.id)
                seen[name] = n.id

    def check_reachability(self) -> None:
        outputs = [n.id for n in self.p.nodes if n.type == "output"]
        if not outputs:
            self.add("error", "no-output", "the pipeline has no output node")
            return
        back: dict[str, list[str]] = {n.id: [] for n in self.p.nodes}
        for e in self.p.edges:
            back[e.dst_node].append(e.src_node)
        seen = set(outputs)
        queue = deque(outputs)
        while queue:
            for prev in back[queue.popleft()]:
                if prev not in seen:
                    seen.add(prev)
                    queue.append(prev)
        for n in self.p.nodes:
            if n.id not in seen:
                self.add("warning", "unreachable", "does not contribute to any output", n.id)


def lint(
    pipeline: Pipeline | Mapping[str, Any], *, http_allow: list[str] | None = None
) -> LintReport:
    """Lint a pipeline (or its raw dict). Never raises for bad input; reports ``schema`` errors."""
    if not isinstance(pipeline, Pipeline):
        try:
            pipeline = Pipeline.model_validate(dict(pipeline))
        except ValueError as exc:
            from .schema import explain_error

            return LintReport([Issue("error", "schema", explain_error(exc))])
    return _Linter(pipeline, http_allowlist(http_allow)).run()
