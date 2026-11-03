"""The simulation layer: project templates, scripted Forge client and composed text."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nimlab.core import NIMClient, NIMConfig, UsageTracker
from nimlab.studio import simulate as sim
from nimlab.studio.sim_extra import generic_template, prompt_words
from nimlab.studio.sim_projects import _TEXTKIT, all_templates, pick_template

GENERIC_PROMPTS = [
    "a weather forecast formatter",
    "Build me a haiku syllable counter!",
    "def class import",
    "x",
    '"quoted" \\ back\\slash """ prompt about pizza',
]
ALL_TEMPLATES = [*all_templates(), _TEXTKIT, *(generic_template(p) for p in GENERIC_PROMPTS)]


def run_pytest(tmp: Path) -> str:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=line"],
        cwd=tmp,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return out.stdout.strip().splitlines()[-1]


class TestTemplates:
    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.key + "-" + t.module)
    def test_exactly_one_failing_test_then_green(self, template, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / template.test_module).write_text(template.tests)
        (tmp_path / template.source_file).write_text(template.buggy_source)
        assert run_pytest(tmp_path).startswith("1 failed"), template.key
        assert template.patch.startswith("<<<<<<< SEARCH\n") and template.patch.endswith(
            ">>>>>>> REPLACE\n"
        )
        (tmp_path / template.source_file).write_text(template.source)
        green = run_pytest(tmp_path)
        assert "passed" in green and "failed" not in green, green

    @pytest.mark.parametrize(
        ("prompt", "key"),
        [
            ("Write FizzBuzz", "fizzbuzz"),
            ("a todo app", "todo"),
            ("URL slug generator", "slugify"),
            ("an LRU cache", "lru"),
            ("count word frequency", "wordstats"),
            ("a calculator", "calculator"),
            ("a URL shortener with an HTTP API", "shortener"),
            ("parse a markdown table", "mdtable"),
            ("unit converter", "units"),
            ("password generator", "password"),
            ("text adventure mini game", "adventure"),
            ("CSV stats tool", "csvstats"),
            ("a notes CLI", "notes"),
            ("something else entirely", "generic"),
        ],
    )
    def test_keyword_selection(self, prompt: str, key: str) -> None:
        assert pick_template(prompt).key == key

    def test_different_prompts_make_different_projects(self) -> None:
        prompts = [
            "todo list",
            "url shortener api",
            "markdown table parser",
            "LRU cache",
            "unit converter",
            "password generator",
            "text adventure game",
            "csv stats",
            "weather forecast formatter",
            "haiku syllable counter",
        ]
        picked = [pick_template(p) for p in prompts]
        assert len({t.source for t in picked}) == len(prompts)
        assert len({t.source_file for t in picked}) == len(prompts)

    def test_generic_module_is_named_after_the_prompt(self) -> None:
        t = generic_template("a weather forecast formatter")
        assert t.source_file == "weather_forecast_formatter_kit.py"
        assert "def count_weather(" in t.source and "def count_forecast(" in t.source
        assert prompt_words("def class import") == ["def", "import"]
        assert prompt_words("!!") == ["item"]
        assert generic_template("haiku syllable counter").source_file.startswith("haiku_syllable")


class TestText:
    def test_topic_extraction(self) -> None:
        assert (
            sim.topic_of("Is a monorepo right for a small team?")
            == "a monorepo right for a small team"
        )
        assert sim.topic_of("Explain TCP slow start") == "TCP slow start"
        assert sim.topic_of("?") == "this question"

    def test_models_write_differently_but_deterministically(self) -> None:
        msgs = [
            {"role": "system", "content": "You are one expert member of a council of AI models"},
            {"role": "user", "content": "How should we version an API?"},
        ]
        a = sim.compose_council_text("meta/llama-3.3-70b-instruct", msgs)
        b = sim.compose_council_text("deepseek-ai/deepseek-r1", msgs)
        assert a != b and a == sim.compose_council_text("meta/llama-3.3-70b-instruct", msgs)
        assert "version an API" in a and len(a.split()) > 40

    def test_critique_and_aggregate_shapes(self) -> None:
        crit = [
            {"role": "system", "content": "You are a rigorous reviewer in a panel debate"},
            {
                "role": "user",
                "content": "Question:\nQ?\n\nPeer answers to review:\n\n### Response 1\nA\n\n### Response 2\nB",
            },
        ]
        text = sim.compose_council_text("m", crit)
        assert "Response 1" in text and "Response 2" in text and "Overall position:" in text
        agg = [
            {"role": "system", "content": "You have been provided with a set of responses"},
            {
                "role": "user",
                "content": "Question:\nWhy?\n\nResponses from other models:\n\n### Response 1\nA",
            },
        ]
        assert sim.compose_council_text("m", agg).startswith(
            "Pulling the strongest threads together"
        )

    def test_atlas_answers_are_extractive_and_cited(self) -> None:
        system = (
            "You are Atlas ...\nSOURCES\n[1] api.md\n<<<\nThe API allows 100 requests per minute per key. "
            "Bursts above that receive HTTP 429.\n>>>\n\n[2] ops.md\n<<<\nDeploys happen on Tuesday mornings.\n>>>"
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": "What is the API rate limit?"},
        ]
        text = sim.compose_atlas_text(msgs)
        assert "100 requests per minute" in text and text.rstrip().endswith("[1]")
        assert "[2]" not in text
        nothing = [
            {
                "role": "system",
                "content": "You are Atlas ...\nSOURCES\n(no sources were retrieved)",
            },
            {"role": "user", "content": "x"},
        ]
        assert "do not cover" in sim.compose_atlas_text(nothing)


class TestStudioClient:
    async def test_passthrough_when_not_simulating(self) -> None:
        base = NIMClient(NIMConfig(mock=True))
        client = sim.StudioClient(base, simulate=False)
        res = await client.chat("meta/llama-3.3-70b-instruct", [{"role": "user", "content": "hi"}])
        assert res.content and client.mock is True
        assert client.usage is base.usage

    async def test_simulates_council_prompts_and_records_usage(self) -> None:
        base = NIMClient(NIMConfig(mock=True))
        client = sim.StudioClient(base, simulate=True, delay=0, council=True)
        token = sim.current_step.set(7)
        try:
            res = await client.chat(
                "qwen/qwen2.5-coder-32b-instruct",
                [
                    {"role": "system", "content": "You are one expert member of a council"},
                    {"role": "user", "content": "Why is the sky blue?"},
                ],
            )
        finally:
            sim.current_step.reset(token)
        assert "sky is blue" in res.content or "sky" in res.content
        assert client.previews[7] == res.content
        assert base.usage.summary()["total"]["calls"] == 1

    async def test_json_and_tool_calls_pass_through(self) -> None:
        base = NIMClient(NIMConfig(mock=True))
        client = sim.StudioClient(base, simulate=True, delay=0, council=True)
        res = await client.chat(
            "m", [{"role": "user", "content": "x"}], response_format={"type": "json_object"}
        )
        assert res.content  # core mock output, untouched
        # the reward-model shaped call ([user, assistant]) is never simulated
        res = await client.chat(
            "m", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )
        assert "sky" not in res.content

    async def test_streaming(self) -> None:
        base = NIMClient(NIMConfig(mock=True))
        client = sim.StudioClient(base, simulate=True, delay=0)
        system = "You are Atlas ...\nSOURCES\n[1] a.md\n<<<\nThe limit is 100 requests per minute for every key.\n>>>"
        pieces = [
            p
            async for p in client.stream_chat(
                "m", [{"role": "system", "content": system}, {"role": "user", "content": "limit?"}]
            )
        ]
        assert len(pieces) >= 1 and "100 requests" in "".join(pieces)


class TestSimForgeClient:
    async def test_scripted_turns(self) -> None:
        usage = UsageTracker()
        client = sim.SimForgeClient("fizzbuzz", usage=usage, delay=0)
        sysmsg = {"role": "system", "content": "s"}
        first = await client.chat(
            "coder", [sysmsg, {"role": "user", "content": "implement"}], tools=[{}]
        )
        assert [c.name for c in first.tool_calls] == ["list_dir"]
        assert first.usage.prompt_tokens > 0 and usage.summary()["total"]["calls"] == 1
        # after four tool rounds the implement turn ends in plain text
        msgs = [sysmsg, {"role": "user", "content": "implement"}] + [
            {"role": "assistant", "content": ""}
        ] * 5
        last = await client.chat("coder", msgs, tools=[{}])
        assert last.tool_calls == [] and "Implemented" in last.content
        # a second user turn is the fix round
        fix = await client.chat(
            "coder", [*msgs, {"role": "user", "content": "fix round 1"}], tools=[{}]
        )
        assert [c.name for c in fix.tool_calls] == ["read_file"]

    async def test_unknown_schema_is_an_error(self) -> None:
        from pydantic import BaseModel

        class Other(BaseModel):
            x: int = 1

        with pytest.raises(ValueError, match="no script"):
            await sim.SimForgeClient("x", delay=0).chat_json("m", [], Other)

    def test_not_flagged_as_mock(self) -> None:
        assert sim.SimForgeClient("x").mock is False


class TestPromptDrivenPlan:
    async def test_plan_and_review_reference_the_prompt(self) -> None:
        from nimlab.forge.models import Plan, Review

        prompt = "Build a password generator for my team"
        client = sim.SimForgeClient(prompt, delay=0)
        plan = await client.chat_json("m", [], Plan)
        assert "password generator for my team" in plan.goal
        assert "password generator for my team" in plan.steps[0].description
        review = await client.chat_json("m", [], Review)
        assert "password generator for my team" in review.summary
