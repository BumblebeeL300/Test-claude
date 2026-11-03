"""Loom quickstart: build a pipeline in code, lint it, run it offline and watch the events.

python examples/loom/quickstart.py            # mock mode, no API key needed
NVIDIA_API_KEY=nvapi-... python examples/loom/quickstart.py   # real NIM models
"""

from __future__ import annotations

import asyncio
import os

from nimlab.loom import Pipeline, lint, run_pipeline, to_mermaid

os.environ.setdefault("NIMLAB_MOCK", "1")  # remove this line to use a real key if you have one

pipeline = Pipeline.from_data(
    {
        "name": "pun-factory",
        "title": "Pun factory",
        "nodes": [
            {"id": "topic", "type": "input", "params": {"type": "string", "default": "coffee"}},
            {
                "id": "ideas",
                "type": "json",
                "params": {
                    "template": "Give three short pun setups about {{topic}}.",
                    "schema": {
                        "type": "object",
                        "properties": {"setups": {"type": "array", "items": {"type": "string"}}},
                        "required": ["setups"],
                    },
                },
            },
            {
                "id": "setups",
                "type": "python_fn",
                "params": {"fn": "get_field", "args": {"field": "setups"}},
            },
            {
                "id": "punch",
                "type": "map",
                "params": {
                    "step": {
                        "type": "prompt",
                        "params": {"template": "Finish this pun in one line: {{item}}"},
                    },
                    "concurrency": 3,
                },
            },
            {"id": "best", "type": "output", "params": {"name": "puns"}},
        ],
        "edges": [
            {"from": "topic.value", "to": "ideas.topic"},
            {"from": "ideas.data", "to": "setups.value"},
            {"from": "setups.result", "to": "punch.items"},
            {"from": "punch.results", "to": "best.value"},
        ],
    }
)


async def main() -> None:
    report = lint(pipeline)
    print("lint:", "clean" if not report.issues else [i.format() for i in report.issues])

    def show(event: dict) -> None:
        if event["type"] in ("node.start", "node.end", "node.skipped"):
            print(f"  {event['type']:<13} {event['node']}")

    result = await run_pipeline(pipeline, {"topic": "tea"}, on_event=show, cache=False)
    print("ok:", result.ok, "| tokens:", result.tokens, "| seconds:", round(result.elapsed_s, 3))
    print("puns:", result.outputs["puns"])
    print()
    print(to_mermaid(pipeline))


asyncio.run(main())
