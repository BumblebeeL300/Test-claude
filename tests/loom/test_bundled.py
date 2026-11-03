from __future__ import annotations

from typing import Any

import pytest

from nimlab.loom import list_pipelines, load_pipeline, run_pipeline
from nimlab.loom.library import bundled_dir

BUNDLED = [i.id for i in list_pipelines(include_user=False)]


def test_the_four_example_pipelines_ship() -> None:
    assert {"research-brief", "support-triage", "doc-to-quiz", "translate-backcheck"} <= set(
        BUNDLED
    )
    assert all(p.suffix == ".json" for p in bundled_dir().iterdir() if p.is_file())


@pytest.mark.parametrize("name", BUNDLED)
async def test_runs_end_to_end_in_mock_mode(name: str) -> None:
    events: list[dict[str, Any]] = []
    result = await run_pipeline(name, mock=True, on_event=events.append)
    assert result.ok, result.error
    assert result.outputs and all(v not in (None, "", []) for v in result.outputs.values())
    assert events[0]["type"] == "run.plan" and events[-1]["type"] == "run.done"
    assert result.tokens > 0
    done = {e["node"] for e in events if e["type"] in ("node.end", "node.skipped")}
    assert done == {n.id for n in load_pipeline(name).nodes}  # every node ended or was skipped


async def test_research_brief_fans_out_with_map() -> None:
    result = await run_pipeline("research-brief", {"topic": "tidal power"}, mock=True)
    assert len(result.outputs["questions"]) >= 3
    assert result.nodes["answers"].status == "done" and len(
        result.nodes["answers"].outputs["results"]
    ) == len(result.outputs["questions"])


async def test_support_triage_runs_exactly_one_responder() -> None:
    result = await run_pipeline("support-triage", mock=True)
    ran = [
        n
        for n in ("billing_reply", "tech_reply", "general_reply")
        if result.nodes[n].status == "done"
    ]
    skipped = [
        n
        for n in ("billing_reply", "tech_reply", "general_reply")
        if result.nodes[n].status == "skipped"
    ]
    assert len(ran) == 1 and len(skipped) == 2
    assert result.outputs["reply"] == result.nodes[ran[0]].outputs["text"]


async def test_doc_to_quiz_dedupes() -> None:
    result = await run_pipeline("doc-to-quiz", mock=True)
    produced = sum(len(r["questions"]) for r in result.nodes["generate"].outputs["results"])
    assert result.outputs["question_count"] <= produced
    assert result.outputs["quiz"].startswith("1. ")


async def test_translate_backcheck_takes_one_arm() -> None:
    result = await run_pipeline("translate-backcheck", {"language": "German"}, mock=True)
    arms = {result.nodes[n].status for n in ("keep", "fix")}
    assert arms == {"done", "skipped"}
    assert 1 <= result.outputs["report"]["score"] <= 10
