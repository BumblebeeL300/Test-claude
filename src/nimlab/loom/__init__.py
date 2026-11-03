"""Loom: visual, patchable pipelines for NVIDIA NIM models.

>>> from nimlab.loom import Pipeline, run_pipeline            # doctest: +SKIP
>>> pipe = Pipeline.load("research-brief.json")               # doctest: +SKIP
>>> result = await run_pipeline(pipe, {"topic": "fusion power"}, mock=True)   # doctest: +SKIP
>>> result.outputs["brief"]                                   # doctest: +SKIP

Pipelines are JSON DAGs of typed nodes joined by cables. See ``docs/loom.md``.
"""

from __future__ import annotations

from . import nodes as _nodes  # noqa: F401  (registers the built-in node types)
from .cache import NodeCache
from .engine import NodeReport, RunResult, run_pipeline
from .errors import (
    CycleError,
    ExprError,
    InputError,
    LoomError,
    NodeFailed,
    NodeTimeout,
    PipelineError,
    TemplateError,
)
from .expr import evaluate as evaluate_expression
from .export import to_mermaid, to_python
from .functions import list_functions, register_function
from .library import list_pipelines, load_pipeline, new_pipeline
from .linter import Issue, LintReport, lint
from .registry import NodeType, Port, VarSource, node_types, register_node_type
from .schema import SCHEMA_VERSION, Edge, Node, Pipeline, Position

__all__ = [
    "SCHEMA_VERSION",
    "CycleError",
    "Edge",
    "ExprError",
    "InputError",
    "Issue",
    "LintReport",
    "LoomError",
    "Node",
    "NodeCache",
    "NodeFailed",
    "NodeReport",
    "NodeTimeout",
    "NodeType",
    "Pipeline",
    "PipelineError",
    "Port",
    "Position",
    "RunResult",
    "TemplateError",
    "VarSource",
    "evaluate_expression",
    "lint",
    "list_functions",
    "list_pipelines",
    "load_pipeline",
    "new_pipeline",
    "node_types",
    "register_function",
    "register_node_type",
    "run_pipeline",
    "to_mermaid",
    "to_python",
]
