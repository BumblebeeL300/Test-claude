"""Every grader, including the sandbox (timeout, env scrubbing, no network)."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from nimlab.arena.graders import (
    GradeContext,
    Grader,
    GradeResult,
    JudgeVerdict,
    available_graders,
    get_grader,
    register_grader,
    run_python_tests,
    validate_json_schema,
)
from nimlab.arena.graders import base as base_module
from nimlab.arena.graders.base import GraderError
from nimlab.arena.graders.basic import extract_number, find_numbers, normalize_text
from nimlab.arena.graders.checks import CHECKS, count_words, run_check
from nimlab.arena.graders.sandbox import extract_code, scrubbed_env
from nimlab.core.errors import NIMResponseError


async def grade(kind: str, output: str, **params: Any) -> GradeResult:
    return await get_grader(kind).grade(GradeContext(output=output, params=params))


# ------------------------------------------------------------------ registry


def test_registry_has_all_builtin_graders_and_aliases() -> None:
    assert set(available_graders()) >= {
        "exact", "numeric", "regex", "contains_all", "checks", "json_schema", "code_exec", "llm_judge",
    }  # fmt: skip
    assert get_grader("contains-all").name == "contains_all"
    assert get_grader("numeric-tolerance").name == "numeric"
    assert get_grader("programmatic_checks").name == "checks"
    with pytest.raises(ValueError, match="unknown grader"):
        get_grader("nope")


async def test_custom_grader_can_be_registered() -> None:
    class ShoutParams(BaseModel):
        word: str

    @register_grader
    class ShoutGrader(Grader):
        name = "shout_test"
        Params = ShoutParams

        async def run(self, params: ShoutParams, ctx: GradeContext) -> GradeResult:
            ok = params.word.upper() in ctx.output
            return GradeResult(score=float(ok), passed=ok)

    try:
        assert "shout_test" in available_graders()
        assert (await grade("shout_test", "HELLO there", word="hello")).passed
        assert not (await grade("shout_test", "hello", word="hello")).passed
    finally:
        base_module._REGISTRY.pop("shout_test", None)


# ------------------------------------------------------------------ exact


async def test_exact_normalises_case_punctuation_and_lead_ins() -> None:
    for out in (
        "Canberra",
        " canberra. ",
        "**Canberra**",
        "The answer is: Canberra!",
        '"Canberra"',
    ):
        assert (await grade("exact", out, expected="Canberra")).passed, out
    assert not (await grade("exact", "Sydney", expected="Canberra")).passed
    assert not (await grade("exact", "It is Canberra", expected="Canberra")).passed


async def test_exact_accepts_lists_case_sensitive_and_contains_mode() -> None:
    assert (await grade("exact", "Fe", expected=["iron", "Fe"], case_sensitive=True)).passed
    assert not (await grade("exact", "fe", expected="Fe", case_sensitive=True)).passed
    assert (
        await grade("exact", "It is Canberra, of course", expected="Canberra", mode="contains")
    ).passed
    assert not (await grade("exact", "Canberras", expected="Canberra", mode="contains")).passed
    with pytest.raises(ValidationError):
        get_grader("exact").validate_params({"expected": ""})


def test_normalize_text() -> None:
    assert normalize_text("  Hello,   World!! ") == "hello, world"
    assert normalize_text("Final answer: 42.") == "42"


# ------------------------------------------------------------------ numeric


@pytest.mark.parametrize(
    ("text", "mode", "want"),
    [
        ("so x = 4\nAnswer: 150", "auto", 150.0),
        ("Answer: $2,315.25", "auto", 2315.25),
        ("**Answer:** 17 minutes", "auto", 17.0),
        ("working 3*4=12 ... the final answer is -7.5.", "auto", -7.5),
        (r"Thus \boxed{42}.", "auto", 42.0),
        ("It is 3/4 of the total", "auto", 0.75),
        ("1.5e3 units", "auto", 1500.0),
        ("first 10 then 20 then 30", "first", 10.0),
        ("first 10 then 20 then 30", "last", 30.0),
        ("no digits here", "auto", None),
        ("range 2-3", "first", 2.0),
    ],
)
def test_extract_number(text: str, mode: str, want: float | None) -> None:
    assert extract_number(text, mode) == want


def test_find_numbers_handles_thousands_and_minus() -> None:
    assert find_numbers("1,234,567 and −5 and 0.25") == [1234567.0, -5.0, 0.25]


async def test_numeric_tolerance() -> None:
    assert (
        await grade("numeric", "Answer: 0.1389", expected=5 / 36, abs_tol=0.0006, rel_tol=0)
    ).passed
    assert not (
        await grade("numeric", "Answer: 0.14", expected=5 / 36, abs_tol=0.0006, rel_tol=0)
    ).passed
    assert (await grade("numeric", "Answer: 100.4", expected=100, rel_tol=0.01)).passed
    miss = await grade("numeric", "Answer: 41", expected=42)
    assert not miss.passed and miss.score == 0 and "expected 42" in miss.detail
    assert "no number" in (await grade("numeric", "dunno", expected=1)).detail


# ------------------------------------------------------------------ regex / contains_all


async def test_regex_grader_modes_flags_and_negate() -> None:
    assert (await grade("regex", "The planet JUPITER", pattern=r"\bjupiter\b")).passed
    assert not (await grade("regex", "jupiter", pattern=r"\bJupiter\b", flags="")).passed
    assert (await grade("regex", "yes-yes", pattern=r"yes(-yes)+", mode="fullmatch")).passed
    assert not (await grade("regex", "yes-yes!", pattern=r"yes(-yes)+", mode="fullmatch")).passed
    assert (await grade("regex", "clean text", pattern=r"password", negate=True)).passed
    assert (await grade("regex", "line1\nline2", pattern=r"line1.line2", flags="is")).passed


async def test_contains_all_scores_fraction_of_conditions() -> None:
    r = await grade(
        "contains_all", "harvest and wind", required=["harvest", "wind"], forbidden=["leaf"]
    )
    assert r.passed and r.score == 1.0
    partial = await grade(
        "contains_all", "harvest leaf", required=["harvest", "wind"], forbidden=["leaf"]
    )
    assert not partial.passed and partial.score == pytest.approx(1 / 3)
    assert "wind" in partial.detail and "leaf" in partial.detail
    assert (await grade("contains_all", "I like cats", any_of=[["dog", "cat"]])).passed
    assert not (
        await grade("contains_all", "concatenate", required=["cat"], whole_word=True)
    ).passed
    assert not (
        await grade("contains_all", "Cat", required=["Cat"], case_sensitive=True, forbidden=["Cat"])
    ).passed
    with pytest.raises(ValidationError):
        get_grader("contains_all").validate_params({})


# ------------------------------------------------------------------ programmatic checks


def test_count_words_ignores_pure_punctuation() -> None:
    assert count_words("one - two -- three!") == 3


@pytest.mark.parametrize(
    ("spec", "text", "ok"),
    [
        ({"type": "word_count", "exactly": 3}, "a b c", True),
        ({"type": "word_count", "exactly": 3}, "a b", False),
        ({"type": "word_count", "min": 2, "max": 3}, "a b c d", False),
        ({"type": "char_count", "max": 5}, "abcdef", False),
        ({"type": "sentence_count", "exactly": 2}, "One. Two!", True),
        ({"type": "sentence_count", "exactly": 2}, "One.", False),
        ({"type": "paragraph_count", "exactly": 2}, "a\n\nb", True),
        ({"type": "line_count", "exactly": 3}, "a\nb\n\nc", True),
        ({"type": "bullet_count", "exactly": 2}, "- a\n* b\ntext", True),
        ({"type": "only_bullets"}, "- a\n- b", True),
        ({"type": "only_bullets"}, "- a\nplain", False),
        ({"type": "numbered_list", "exactly": 3}, "1. a\n2. b\n3. c", True),
        ({"type": "numbered_list", "exactly": 3}, "1. a\n3. b\n4. c", False),
        ({"type": "all_caps"}, "HELLO, WORLD 1", True),
        ({"type": "all_caps"}, "Hello", False),
        ({"type": "all_caps"}, "123", False),
        ({"type": "all_lowercase"}, "hello world", True),
        ({"type": "all_lowercase"}, "Hello", False),
        ({"type": "includes", "texts": ["a", "B"]}, "xa xb", True),
        ({"type": "includes", "text": "B", "case_sensitive": True}, "b", False),
        ({"type": "excludes", "texts": ["bad"]}, "so BAD", False),
        ({"type": "excludes", "texts": ["bad"]}, "good", True),
        ({"type": "starts_with", "text": "Morning"}, "  Morning sun", True),
        ({"type": "ends_with", "text": "?"}, "Really?  \n", True),
        ({"type": "regex", "pattern": r"^<<.+>>"}, "<<T>>\nbody", True),
        ({"type": "regex", "pattern": "x", "negate": True}, "abc", True),
        ({"type": "no_commas"}, "a, b", False),
        ({"type": "in_order", "words": ["quiet", "river"]}, "The quiet river", True),
        ({"type": "in_order", "words": ["quiet", "river"]}, "river then quiet", False),
        ({"type": "max_words_per_paragraph", "max": 3}, "a b c\n\nd e f g", False),
        ({"type": "json_parses"}, '{"a": 1}', True),
        ({"type": "json_parses"}, "{a: 1}", False),
    ],
)
def test_individual_checks(spec: dict, text: str, ok: bool) -> None:
    assert run_check(spec, text)[0] is ok


def test_every_check_type_is_covered_by_the_table_above() -> None:
    assert set(CHECKS) == {
        "word_count", "char_count", "sentence_count", "paragraph_count", "line_count",
        "bullet_count", "only_bullets", "numbered_list", "all_caps", "all_lowercase", "includes",
        "excludes", "starts_with", "ends_with", "regex", "no_commas", "in_order",
        "max_words_per_paragraph", "json_parses",
    }  # fmt: skip


async def test_checks_grader_partial_credit() -> None:
    checks = [{"type": "all_caps"}, {"type": "word_count", "max": 3}, {"type": "no_commas"}]
    full = await grade("checks", "HI THERE", checks=checks)
    assert full.passed and full.score == 1.0
    part = await grade("checks", "Hi there, ok", checks=checks)
    assert not part.passed and part.score == pytest.approx(1 / 3)
    assert "failed" in part.detail and len(part.meta["checks"]) == 3


# ------------------------------------------------------------------ json schema

SCHEMA = {
    "type": "object",
    "required": ["name", "age"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
}


async def test_json_schema_grader_accepts_valid_and_fenced_json() -> None:
    assert (await grade("json_schema", '{"name": "Ann", "age": 3}', schema=SCHEMA)).passed
    fenced = 'Sure!\n```json\n{"name": "Ann", "age": 3, "tags": ["x"]}\n```'
    assert (await grade("json_schema", fenced, schema=SCHEMA)).passed
    assert (
        await grade("json_schema", '<think>hmm</think>{"name": "Ann", "age": 0}', schema=SCHEMA)
    ).passed


@pytest.mark.parametrize(
    ("output", "fragment"),
    [
        ('{"name": "Ann"}', "missing required property 'age'"),
        ('{"name": "Ann", "age": "3"}', "expected integer"),
        ('{"name": "Ann", "age": -1}', "below minimum"),
        ('{"name": "", "age": 1}', "shorter than"),
        ('{"name": "A", "age": 1, "extra": 1}', "unexpected property 'extra'"),
        ('{"name": "A", "age": 1, "tags": ["a", "b", "c"]}', "at most 2"),
        ("not json at all", "no valid JSON"),
    ],
)
async def test_json_schema_grader_rejects_with_reason(output: str, fragment: str) -> None:
    result = await grade("json_schema", output, schema=SCHEMA)
    assert not result.passed and result.score == 0.0
    assert fragment in result.detail


async def test_json_schema_strict_rejects_surrounding_prose() -> None:
    body = '{"name": "Ann", "age": 3}'
    assert (await grade("json_schema", body, schema=SCHEMA, strict=True)).passed
    assert (await grade("json_schema", f"```json\n{body}\n```", schema=SCHEMA, strict=True)).passed
    assert not (
        await grade("json_schema", f"Here you go: {body}", schema=SCHEMA, strict=True)
    ).passed


def test_validator_supports_combinators_refs_and_formats() -> None:
    schema = {
        "$defs": {"id": {"type": "string", "pattern": "^[a-z]{3}$"}},
        "type": "object",
        "properties": {
            "id": {"$ref": "#/$defs/id"},
            "when": {"type": "string", "format": "date"},
            "val": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
            "opt": {"type": ["string", "null"]},
            "n": {"type": "number", "exclusiveMinimum": 0, "multipleOf": 0.5},
            "pair": {
                "type": "array",
                "prefixItems": [{"const": 1}, {"enum": ["a", "b"]}],
                "items": False,
            },
        },
    }
    ok = {"id": "abc", "when": "2025-02-28", "val": 3, "opt": None, "n": 1.5, "pair": [1, "b"]}
    assert validate_json_schema(ok, schema) == []
    bad = {"id": "ABC", "when": "2025-02-30", "val": 1.5, "opt": 1, "n": 0.7, "pair": [2, "c", 3]}
    errors = " | ".join(validate_json_schema(bad, schema))
    for fragment in (
        "pattern",
        "valid date",
        "oneOf",
        "string|null",
        "multiple of",
        "constant",
        "not one of",
    ):
        assert fragment in errors, fragment
    assert validate_json_schema(True, {"type": "integer"}) != []  # bool is not an integer
    assert validate_json_schema(3.0, {"type": "integer"}) == []


def test_validator_matches_reference_jsonschema_library() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    from nimlab.arena.suites import load_suites

    suite = load_suites()["json_compliance"]
    for item in suite.items:
        schema = suite.grader_for(item).params["schema"]
        good = json.loads(item.reference)
        assert validate_json_schema(good, schema) == []
        jsonschema.validate(good, schema)
        mutants: list[Any] = [None, [], {}, "x", 3, {"zzz": 1}]
        if isinstance(good, dict) and good:
            mutants.append({k: v for k, v in list(good.items())[1:]})
            mutants.append({**good, "unexpected_key": 1})
        for mutant in mutants:
            mine = validate_json_schema(mutant, schema) == []
            ref = jsonschema.Draft202012Validator(schema).is_valid(mutant)
            assert mine == ref, f"{item.id}: {mutant!r}: mine={mine} reference={ref}"


# ------------------------------------------------------------------ code_exec sandbox

TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]


async def test_code_exec_all_pass_extracts_fenced_python() -> None:
    out = "Here:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    r = await grade("code_exec", out, tests=TESTS, entrypoint="add")
    assert r.passed and r.score == 1.0 and "3/3" in r.detail


async def test_code_exec_partial_credit_and_failure_detail() -> None:
    r = await grade("code_exec", "def add(a, b):\n    return abs(a) + abs(b)", tests=TESTS)
    assert not r.passed and r.score == pytest.approx(2 / 3)
    assert "first failure" in r.detail and "assertion failed" in r.detail


async def test_code_exec_reports_syntax_and_runtime_errors() -> None:
    syntax = await grade("code_exec", "def add(a, b) return a + b", tests=TESTS)
    assert syntax.score == 0 and "failed to load" in syntax.detail
    boom = await grade("code_exec", "def add(a, b):\n    raise RuntimeError('boom')", tests=TESTS)
    assert boom.score == 0 and "boom" in json.dumps(boom.meta["tests"])
    assert (await grade("code_exec", "", tests=TESTS)).score == 0
    missing = await grade("code_exec", "x = 1", tests=TESTS)
    assert missing.score == 0 and "NameError" in json.dumps(missing.meta["tests"])


async def test_code_exec_wall_clock_timeout_kills_runaway_code() -> None:
    start = time.monotonic()
    res = await run_python_tests("while True:\n    pass\n", ["assert True"], timeout_s=1.0)
    assert res.timed_out and not res.ok
    assert time.monotonic() - start < 8
    assert "timed out" in res.summary()


async def test_code_exec_per_test_timeout_keeps_other_tests() -> None:
    code = "def f(n):\n    while n == 99:\n        pass\n    return n\n"
    res = await run_python_tests(
        code,
        ["assert f(1) == 1", "assert f(99) == 99", "assert f(2) == 2"],
        timeout_s=15,
        test_timeout_s=0.5,
    )
    assert not res.timed_out
    assert (res.passed, res.total) == (2, 3)
    assert res.details[1]["error"] == "timeout"


async def test_code_exec_sleep_beyond_limit_is_killed() -> None:
    res = await run_python_tests("import time\ntime.sleep(30)\n", ["assert True"], timeout_s=1.0)
    assert res.timed_out


async def test_sandbox_scrubs_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-super-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("NIMLAB_TEST_SECRET", "hunter2")
    code = "import os\nSEEN = dict(os.environ)\n"
    tests = [
        "assert 'NVIDIA_API_KEY' not in SEEN",
        "assert 'AWS_SECRET_ACCESS_KEY' not in SEEN",
        "assert 'NIMLAB_TEST_SECRET' not in SEEN",
        "assert not any('secret' in v.lower() or 'hunter2' in v for v in SEEN.values())",
        "assert set(SEEN) <= {'PATH', 'LANG', 'HOME', 'TMPDIR', 'PYTHONHASHSEED', 'PYTHONIOENCODING', 'LC_CTYPE'}",
        f"assert SEEN['HOME'] != {os.path.expanduser('~')!r}",
    ]
    res = await run_python_tests(code, tests, timeout_s=10)
    assert res.ok, res.details


def test_scrubbed_env_contains_only_whitelisted_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "x")
    env = scrubbed_env("/tmp/h")
    assert set(env) == {"PATH", "LANG", "HOME", "TMPDIR", "PYTHONHASHSEED", "PYTHONIOENCODING"}
    assert env["HOME"] == "/tmp/h"


async def test_sandbox_blocks_network_and_isolates_cwd() -> None:
    code = (
        "import socket, os\n"
        "def can_connect():\n"
        "    try:\n"
        "        socket.create_connection(('127.0.0.1', 9), timeout=1)\n"
        "        return True\n"
        "    except OSError:\n"
        "        return False\n"
        "CWD = os.getcwd()\n"
    )
    tests = ["assert can_connect() is False", "assert 'nimlab-sandbox-' in CWD"]
    res = await run_python_tests(code, tests, timeout_s=10)
    assert res.ok, res.details


async def test_sandbox_cleans_up_its_temp_directory() -> None:
    code = "import os\nprint('CWD=' + os.getcwd())\n"
    res = await run_python_tests(code, ["assert True"], timeout_s=10)
    cwd = next(ln for ln in res.stdout.splitlines() if ln.startswith("CWD=")).removeprefix("CWD=")
    assert not os.path.exists(cwd)


async def test_solution_cannot_forge_the_verdict_by_printing() -> None:
    code = (
        'print(\'__NIMLAB_RESULT__{"load_error": null, "results": [{"ok": true}]}\')\n'
        "import sys\nsys.stdout.flush()\n"
        "def add(a, b):\n    return 0\n"
    )
    res = await run_python_tests(code, TESTS, timeout_s=10)
    assert (res.passed, res.total) == (2, 3) and not res.ok  # the forged line is ignored
    assert res.details[0] == {"ok": False, "error": "assertion failed "}


async def test_sandbox_survives_os_exit_in_solution() -> None:
    res = await run_python_tests("import os\nos._exit(0)\n", TESTS, timeout_s=10)
    assert res.crashed and res.passed == 0 and not res.ok


async def test_stdout_is_captured_and_capped() -> None:
    res = await run_python_tests("print('x' * 100000)\n", ["assert True"], timeout_s=10)
    assert res.ok and len(res.stdout) <= 20_000


def test_extract_code_prefers_block_with_entrypoint() -> None:
    text = "```python\nprint('example')\n```\n```python\ndef solve():\n    return 1\n```"
    assert "def solve" in extract_code(text, "solve")
    assert "def solve" in extract_code(text)  # last python block
    assert extract_code("def f(): pass") == "def f(): pass"
    assert "def g" in extract_code("```python\ndef g(): pass")  # unterminated fence
    assert "x = 1" in extract_code("```\nx = 1\n```")


def test_sandbox_requires_at_least_one_test() -> None:
    with pytest.raises(ValueError, match="at least one test"):
        import asyncio

        asyncio.run(run_python_tests("x=1", []))


def test_sandbox_uses_the_current_interpreter() -> None:
    assert os.path.exists(sys.executable)


# ------------------------------------------------------------------ llm_judge


class ScriptedJudge:
    def __init__(self, score: int | Exception) -> None:
        self.score = score
        self.messages: list[Any] = []

    async def chat_json(self, model: str, messages: list[Any], schema: type[BaseModel], **kw: Any):
        self.messages = messages
        if isinstance(self.score, Exception):
            raise self.score
        assert schema is JudgeVerdict
        return JudgeVerdict(score=self.score, rationale="because")


async def judge(score: int | Exception, **params: Any) -> GradeResult:
    ctx = GradeContext(
        output="A summary.",
        params={"rubric": "Be faithful.", **params},
        prompt="Summarize X",
        reference="Gold summary",
        client=ScriptedJudge(score),
        judge_model="judge-model",
    )
    result = await get_grader("llm_judge").grade(ctx)
    result.meta["messages"] = ctx.client.messages
    return result


async def test_llm_judge_normalises_scores_to_unit_interval() -> None:
    assert (await judge(10)).score == pytest.approx(1.0)
    assert (await judge(1)).score == pytest.approx(0.0)
    mid = await judge(7)
    assert mid.score == pytest.approx(6 / 9) and not mid.passed  # below the 0.7 threshold
    assert (await judge(8)).passed and (await judge(7, pass_threshold=0.6)).passed
    assert "7/10" in mid.detail and mid.meta["judge_model"] == "judge-model"


async def test_llm_judge_prompt_contains_rubric_reference_and_candidate() -> None:
    r = await judge(9)
    user = r.meta["messages"][1]["content"]
    for part in ("Summarize X", "Be faithful.", "Gold summary", "A summary."):
        assert part in user
    without = await judge(9, use_reference=False)
    assert "Gold summary" not in without.meta["messages"][1]["content"]


async def test_llm_judge_errors_become_grader_errors() -> None:
    with pytest.raises(GraderError, match="judge call failed"):
        await judge(NIMResponseError("boom"))
    with pytest.raises(GraderError, match="needs an LLM client"):
        await get_grader("llm_judge").grade(GradeContext(output="x", params={"rubric": "rubric"}))


def test_judge_verdict_schema_enforces_range() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(score=11, rationale="x")
    with pytest.raises(ValidationError):
        JudgeVerdict(score=0, rationale="x")
