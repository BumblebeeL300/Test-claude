from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nimlab.core.errors import NIMResponseError
from nimlab.council import Council, JudgeVerdict, StepType, Trace, render_html, save_html
from nimlab.council.report import build_view, fmt_seconds, model_hues, short_model

Q = "Which is best: ALPHA, BETA or GAMMA?"


def chat(model: str, messages: list[dict], kind: str) -> str:
    return {"org-a/model-a": "ALPHA", "org-b/model-b": "BETA", "org-c/model-c": "GAMMA"}.get(
        model, "ALPHA beta"
    ) + f" ({kind})"


def judge(model, messages, schema, kind):
    return JudgeVerdict(selected=1, final_answer="", rationale="because", confidence=0.7)


@pytest.fixture
async def debate_trace(make_client, panel) -> Trace:
    client = make_client(chat=chat, json=judge)
    result = await Council(client).run(Q, "debate", models=panel, rounds=1, early_stop=False)
    return result.trace


async def test_trace_records_every_step_with_details(debate_trace: Trace, panel) -> None:
    trace = debate_trace
    assert trace.question == Q and trace.strategy == "debate" and trace.models == panel
    assert [s.index for s in trace.steps] == list(range(len(trace.steps)))
    assert {s.type for s in trace.steps} >= {
        StepType.PROPOSE,
        StepType.CRITIQUE,
        StepType.REVISE,
        StepType.CONSENSUS,
        StepType.JUDGE,
    }
    for step in trace.steps:
        assert step.status == "ok" and step.phase
        assert step.duration_s >= 0 and step.ended_s >= step.started_s
        if step.type != StepType.CONSENSUS:
            assert step.prompt_summary and step.prompt and step.output
    first = trace.steps[0]
    assert first.model == panel[0] and first.prompt_tokens == 10 and first.total_tokens > 10
    # started_s is non-decreasing in launch order
    starts = [s.started_s for s in trace.steps]
    assert starts == sorted(starts)


async def test_json_steps_get_token_counts_from_usage_delta(make_client, panel) -> None:
    client = make_client(chat=chat, json=judge)
    result = await Council(client).run(Q, "debate", models=panel, rounds=0)
    judge_step = result.trace.steps_of(StepType.JUDGE)[0]
    assert judge_step.prompt_tokens == 20 and judge_step.completion_tokens == 5
    assert judge_step.output.startswith("{")  # structured output is recorded as JSON
    consensus_step = result.trace.steps_of(StepType.CONSENSUS)[0]
    assert consensus_step.total_tokens and consensus_step.model  # embedding usage is attributed


async def test_overlapping_json_calls_leave_tokens_unknown(make_client, panel) -> None:
    from nimlab.council import RubricScore

    client = make_client(
        json=lambda m, msgs, s, k: RubricScore(score=5, rationale=""), mock=True, delay=0.01
    )
    result = await Council(client).run(Q, "best-of-n", models=["org/solo"], n=3)
    scores = result.trace.steps_of(StepType.SCORE)
    assert len(scores) == 3
    assert all(s.total_tokens is None for s in scores)  # three concurrent calls to one model


async def test_usage_rows_reflect_this_run_only(make_client, panel) -> None:
    client = make_client(chat=chat, json=judge)
    council = Council(client)
    first = await council.run(Q, "debate", models=panel, rounds=0)
    second = await council.run(Q, "debate", models=panel, rounds=0)
    assert [u.calls for u in first.usage] == [u.calls for u in second.usage]
    assert sum(u.calls for u in first.usage) == len(client.calls) // 2
    assert first.trace.total_tokens == second.trace.total_tokens > 0


async def test_failed_steps_are_recorded_with_errors(make_client, panel) -> None:
    def flaky(model, messages, kind):
        if model == panel[1]:
            return NIMResponseError("HTTP 500 upstream", status_code=500, model=model)
        return chat(model, messages, kind)

    result = await Council(make_client(chat=flaky)).run(Q, "moa", models=panel, layers=0)
    bad = [s for s in result.trace.steps if s.status == "error"]
    assert bad and "HTTP 500" in bad[0].error
    assert result.trace.usage_from_steps()[0].model  # derived rows work too
    assert sum(u.errors for u in result.trace.usage_from_steps()) == len(bad)


# ----------------------------------------------------------------------------- JSON


async def test_trace_json_round_trip(debate_trace: Trace, tmp_path: Path) -> None:
    text = debate_trace.to_json()
    restored = Trace.from_json(text)
    assert restored == debate_trace
    assert json.loads(text)["schema_version"] == 1

    path = debate_trace.save(tmp_path / "nested" / "trace.json")
    assert path.exists()
    assert Trace.load(path) == debate_trace


async def test_trace_json_is_plain_data(debate_trace: Trace) -> None:
    data = json.loads(debate_trace.to_json())
    assert {"question", "strategy", "steps", "consensus", "usage", "final_answer", "config"} <= set(
        data
    )
    assert data["steps"][0]["type"] == "propose"
    assert data["config"]["resolved"]["embed"]
    assert data["consensus"][0]["matrix"]


def test_trace_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        Trace.from_json('{"not": "a trace"}')


# ----------------------------------------------------------------------------- HTML


async def test_html_report_contains_expected_elements(debate_trace: Trace, panel) -> None:
    html = render_html(debate_trace)
    assert html.startswith("<!doctype html>")
    assert "<title>Council · debate" in html
    assert debate_trace.question in html
    assert html.count('<details class="step') == len(debate_trace.steps)
    for model in panel:  # per-model chips with colours
        assert short_model(model) in html
    assert html.count("--h:") >= len(panel)
    assert "prefers-color-scheme: dark" in html
    assert 'class="heat"' in html  # consensus heatmap table
    assert 'class="timeline' in html or "timeline" in html
    assert 'class="bar' in html
    assert "Usage by model" in html  # usage table
    assert "because" in html  # judge rationale
    assert "Expand all" in html and "<script>" in html
    assert 'id="step-0"' in html and 'href="#step-0"' in html
    for rep in debate_trace.consensus:
        assert rep.phase in html


async def test_html_is_self_contained(debate_trace: Trace) -> None:
    html = render_html(debate_trace)
    assert "<link" not in html
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)
    assert not re.search(r"@import|url\(", html)
    assert "<script src" not in html
    assert not re.search(r"\{\{|\{%", html)  # no unrendered template syntax


async def test_html_escapes_untrusted_content(make_client, panel) -> None:
    evil = '<script>alert("xss")</script> & <img src=x onerror=1>'
    client = make_client(chat=lambda m, msgs, k: evil)
    result = await Council(client).run("<b>q</b>", "moa", models=panel, layers=0)
    html = result.html()
    assert evil not in html
    assert "&lt;script&gt;alert" in html
    assert "<b>q</b>" not in html and "&lt;b&gt;q&lt;/b&gt;" in html


async def test_html_for_failed_run_and_empty_trace(make_client, panel) -> None:
    from nimlab.council.errors import AllModelsFailedError

    client = make_client(chat=lambda m, msgs, k: NIMResponseError("down", model=m))
    with pytest.raises(AllModelsFailedError) as info:
        await Council(client).run(Q, "moa", models=panel)
    html = render_html(info.value.trace)
    assert "Run failed" in html and "failed" in html

    bare = Trace(question="q", strategy="moa")
    assert "No answer was produced" in render_html(bare)


async def test_html_shows_candidates_scores_and_outliers(make_client, panel) -> None:
    from nimlab.council import RubricScore

    texts = {panel[0]: "ALPHA one", panel[1]: "ALPHA two", panel[2]: "BETA odd one"}
    client = make_client(
        chat=lambda m, msgs, k: texts[m],
        json=lambda m, msgs, s, k: RubricScore(score=6.5, rationale=""),
        mock=True,
    )
    result = await Council(client, outlier_margin=0.05).run(Q, "best-of-n", models=panel, n=3)
    html = result.html()
    assert "Candidate 1" in html and "6.50" in html and ">selected<" in html
    assert "outlier" in html


async def test_save_html_and_result_helpers(debate_trace: Trace, tmp_path: Path) -> None:
    out = save_html(debate_trace, tmp_path / "x" / "report.html")
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


async def test_build_view_groups_steps_by_phase(debate_trace: Trace) -> None:
    view = build_view(debate_trace)
    names = [p["name"] for p in view["phases"]]
    assert names[0].startswith("Round 0") and names[-1] == "Judge"
    assert sum(len(p["steps"]) for p in view["phases"]) == len(debate_trace.steps)
    lanes = view["timeline"]["lanes"]
    assert {lane["model"] for lane in lanes} >= set(debate_trace.models)
    for lane in lanes:
        for bar in lane["bars"]:
            assert 0 <= bar["left"] <= 100 and 0 < bar["width"] <= 100
    assert len(view["consensus"]) == len(debate_trace.consensus)
    assert all(0.0 <= float(c["agreement"]) <= 1.0 for c in view["consensus"])


def test_formatting_helpers() -> None:
    assert fmt_seconds(0.0123) == "12 ms"
    assert fmt_seconds(2.5) == "2.50 s"
    assert fmt_seconds(125) == "2m 5s"
    assert short_model("nvidia/nemotron-3-super-120b-a12b") == "llama-3.3-70b-instruct"
    assert short_model("plain") == "plain"
    hues = model_hues(["a", "b", "c"])
    assert len(set(hues.values())) == 3 and all(0 <= h < 360 for h in hues.values())
