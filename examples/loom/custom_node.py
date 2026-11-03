"""Extend Loom: register a safe function for ``python_fn`` and a brand-new node type.

``python_fn`` nodes can only call registered functions; a node type adds a whole new module (with
its own jacks and parameter schema) to the palette, the CLI and the Studio editor.
"""

from __future__ import annotations

import asyncio
import os

from nimlab.loom import (
    NodeType,
    Pipeline,
    Port,
    register_function,
    register_node_type,
    run_pipeline,
)
from nimlab.loom.context import NodeContext, NodeResult

os.environ.setdefault("NIMLAB_MOCK", "1")

# 1. a function that python_fn nodes may call: value in, JSON-compatible value out
register_function("title_case", lambda value, **_: str(value).title(), doc="Title Case the text")


# 2. a new module: reverses the words of its input
async def run_reverse(ctx: NodeContext) -> NodeResult:
    words = str(ctx.inputs["text"]).split()
    sep = str(ctx.params.get("sep", " "))
    return NodeResult({"text": sep.join(reversed(words))})


register_node_type(
    NodeType(
        name="reverse_words",
        title="Reverse words",
        category="data",
        description="Reverse the order of the words.",
        run=run_reverse,
        params={"sep": {"type": "string", "title": "Separator", "default": " "}},
        inputs=(Port("text", "text"),),
        outputs=(Port("text", "text"),),
        glyph="↺",
    )
)

pipeline = Pipeline.from_data(
    {
        "name": "custom-demo",
        "nodes": [
            {
                "id": "text",
                "type": "input",
                "params": {"type": "string", "default": "patch the bay"},
            },
            {"id": "rev", "type": "reverse_words", "params": {"sep": " / "}},
            {"id": "tc", "type": "python_fn", "params": {"fn": "title_case"}},
            {"id": "out", "type": "output", "params": {"name": "result"}},
        ],
        "edges": [
            {"from": "text.value", "to": "rev.text"},
            {"from": "rev.text", "to": "tc.value"},
            {"from": "tc.result", "to": "out.value"},
        ],
    }
)

result = asyncio.run(run_pipeline(pipeline, mock=True))
print(result.outputs["result"])  # Bay / The / Patch
