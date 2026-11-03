"""Exception types for Loom. Everything derives from :class:`LoomError`."""

from __future__ import annotations

__all__ = [
    "CycleError",
    "ExprError",
    "InputError",
    "LoomError",
    "NodeFailed",
    "NodeTimeout",
    "PipelineError",
    "TemplateError",
]


class LoomError(Exception):
    """Base class for every error Loom raises on purpose."""


class PipelineError(LoomError):
    """The pipeline document is malformed (schema, unknown node type, dangling edge...)."""


class CycleError(PipelineError):
    """The graph has a cycle. ``cycle`` lists the node ids along it (first id repeated last)."""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__("pipeline has a cycle: " + " -> ".join(cycle))


class ExprError(LoomError):
    """A branch/condition expression is invalid or cannot be evaluated."""


class TemplateError(LoomError):
    """A ``{{var}}`` template cannot be rendered (unknown variable, bad path...)."""


class InputError(LoomError):
    """A pipeline input is missing or has the wrong type."""


class NodeFailed(LoomError):
    """A node raised while running (after its retries)."""

    def __init__(self, node: str, message: str) -> None:
        self.node = node
        super().__init__(f"node {node!r} failed: {message}")


class NodeTimeout(LoomError):
    """A node exceeded its timeout."""
