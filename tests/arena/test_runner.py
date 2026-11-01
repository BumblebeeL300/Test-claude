"""The runner, driven by a scripted fake client with known answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from arena_fakes import ScriptedClient, make_suite
from pydantic import BaseModel

from nimlab.arena.graders import JudgeVerdict
from nimlab.arena.results import RunResult, RunStore
from nimlab.arena.runner import Runner, RunSettings, build_messages
from nimlab.arena.suites import Suite
from nimlab.core.errors import NIMAuthError, NIMRateLimitError, NIMResponseError

PROMPTS = {"What is 2+2?": "Answer: 4", "3 squared?": "Answer: 9", "4 squared?": "Answer: 16"}
FAST = RunSettings(concurrency=4, backoff_s=0.0)


def runner(client: Any, store: RunStore | None = None, **kw: Any) -> Runner:
    settings = RunSettings(**{"concurrency": 4, "backoff_s": 0.0, **kw})
    return Runner(client, settings=settings, store=store)


async def test_scores_follow_known_correct_and_incorrect_answers(small_suite: Suite) -> None:
    def answers(model: str, prompt: str) -> str:
        good = PROMPTS[prompt]
        return good if model == "smart" else "Answer: 0"

    client = ScriptedClient(answers)
    run = await runner(client).run(["smart", "dumb"], [small_suite])
    assert run.status == "complete"
    assert len(run.items) == 6
    for r in run.items:
        assert r.score == (1.0 if r.model == "smart" else 0.0)
        assert r.passed is (r.model == "smart") and r.error is None
        assert r.grader == "numeric" and r.item_hash and r.finished_at
    assert run.models == ["smart", "dumb"]
    assert run.suites["small"].category == "math_reasoning" and run.suites["small"].n_items == 3


async def test_ttft_is_measured_from_the_first_streamed_chunk(small_suite: Suite) -> None:
    client = ScriptedClient(PROMPTS, ttft=0.15, chunk_delay=0.05, chunks=3)
    run = await runner(client).run(["m"], [small_suite])
    for r in run.items:
        assert r.ttft_s is not None
        assert 0.14 <= r.ttft_s < 0.15 + 0.25, r.ttft_s
        assert r.latency_s >= r.ttft_s + 0.09  # two more chunks, 50 ms apart
        assert r.tokens_out > 0 and r.tokens_in > 0 and r.usage_estimated
        assert r.tokens_per_s and r.tokens_per_s > 0


async def test_ttft_ignores_empty_chunks_and_strips_think_blocks() -> None:
    suite = make_suite("one", [("q", "Q?", 5)])

    class Chunky(ScriptedClient):
        async def stream_chat(self, model, messages, **kw):  # type: ignore[no-untyped-def]
            import asyncio

            yield ""
            await asyncio.sleep(0.1)
            yield "<think>pondering</think>"
            yield "Answer: 5"

    run = await runner(Chunky()).run(["m"], [suite])
    (r,) = run.items
    assert r.output == "Answer: 5" and r.score == 1.0
    assert r.ttft_s is not None and r.ttft_s >= 0.09  # the empty first chunk does not count


async def test_non_streaming_fallback_uses_api_usage(small_suite: Suite) -> None:
    client = ScriptedClient(PROMPTS, streaming=False)
    run = await runner(client).run(["m"], [small_suite])
    assert all(r.ttft_s is None and r.score == 1.0 for r in run.items)
    assert all((r.tokens_in, r.tokens_out, r.usage_estimated) == (7, 11, False) for r in run.items)
    run2 = await runner(ScriptedClient(PROMPTS), stream=False).run(["m"], [small_suite])
    assert all(r.ttft_s is None for r in run2.items)


async def test_system_prompt_and_item_overrides_reach_the_model() -> None:
    suite = Suite.model_validate(
        {
            "name": "sys",
            "system": "Be terse.",
            "grader": {"type": "numeric", "expected": 1},
            "items": [
                {"id": "a", "prompt": "p1"},
                {"id": "b", "prompt": "p2", "system": "Be loud."},
            ],
        }
    )
    assert build_messages(suite, suite.items[0]) == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "p1"},
    ]
    assert build_messages(suite, suite.items[1])[0]["content"] == "Be loud."
    bare = suite.model_copy(update={"system": None})
    assert build_messages(bare, bare.items[0]) == [{"role": "user", "content": "p1"}]


async def test_transient_errors_are_retried_then_succeed(small_suite: Suite) -> None:
    key = ("m", "What is 2+2?")
    client = ScriptedClient(
        PROMPTS,
        fail_first={
            key: [NIMRateLimitError("429"), NIMResponseError("bad gateway", status_code=502)]
        },
    )
    run = await runner(client, max_retries=2).run(["m"], [small_suite])
    flaky = next(r for r in run.items if r.item_id == "two-plus-two")
    assert flaky.score == 1.0 and flaky.attempts == 3 and flaky.error is None
    assert client.calls.count(key) == 3


async def test_retries_are_bounded_and_failures_are_recorded_not_scored(small_suite: Suite) -> None:
    key = ("m", "What is 2+2?")
    client = ScriptedClient(PROMPTS, fail_first={key: [NIMRateLimitError("429")] * 10})
    run = await runner(client, max_retries=2).run(["m"], [small_suite])
    assert client.calls.count(key) == 3  # 1 try + 2 retries
    bad = next(r for r in run.items if r.item_id == "two-plus-two")
    assert bad.error and "NIMRateLimitError" in bad.error and not bad.scored
    assert run.status == "partial"
    # the other items are unaffected
    assert sum(1 for r in run.items if r.scored and r.score == 1.0) == 2


async def test_non_transient_errors_are_not_retried(small_suite: Suite) -> None:
    key = ("m", "What is 2+2?")
    client = ScriptedClient(
        PROMPTS, fail_first={key: [NIMResponseError("bad request", status_code=400)] * 5}
    )
    run = await runner(client, max_retries=3).run(["m"], [small_suite])
    assert client.calls.count(key) == 1
    assert next(r for r in run.items if r.item_id == "two-plus-two").error


async def test_auth_errors_abort_the_run_but_checkpoint(small_suite: Suite, runs_dir: Path) -> None:
    client = ScriptedClient(PROMPTS, fail_first={("m", "3 squared?"): [NIMAuthError("bad key")]})
    store = RunStore(runs_dir)
    with pytest.raises(NIMAuthError):
        await runner(client, store, concurrency=1).run(["m"], [small_suite])
    (saved_id,) = store.list_ids()
    assert store.load(saved_id).status == "partial"


async def test_concurrency_is_bounded(small_suite: Suite) -> None:
    suite = make_suite("many", [(f"i{n}", f"q{n}", n) for n in range(12)])
    client = ScriptedClient(lambda m, p: f"Answer: {p[1:]}", ttft=0.02)
    await runner(client, concurrency=3).run(["a", "b"], [suite])
    assert client.max_in_flight == 3
    client2 = ScriptedClient(lambda m, p: "Answer: 1", ttft=0.02)
    await runner(client2, concurrency=8, per_model_concurrency=2).run(["a", "b"], [suite])
    assert client2.max_in_flight <= 4  # 2 models x 2 each
    with pytest.raises(ValueError, match="concurrency"):
        RunSettings(concurrency=0)


async def test_run_validates_inputs(small_suite: Suite) -> None:
    with pytest.raises(ValueError, match="model"):
        await runner(ScriptedClient()).run([], [small_suite])
    with pytest.raises(ValueError, match="suite"):
        await runner(ScriptedClient()).run(["m"], [])


async def test_results_are_persisted_as_json_and_reloadable(
    small_suite: Suite, runs_dir: Path
) -> None:
    store = RunStore(runs_dir)
    run = await runner(ScriptedClient(PROMPTS), store).run(["m1", "m2"], [small_suite])
    path = store.path_for(run.run_id)
    assert path.parent == runs_dir and path.suffix == ".json"
    assert len(run.run_id) >= 16 and run.run_id.endswith("Z")  # UTC timestamp
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["status"] == "complete" and len(raw["items"]) == 6
    assert RunStore(runs_dir).load(run.run_id).model_dump() == run.model_dump()
    assert RunStore(runs_dir).load("latest").run_id == run.run_id
    assert RunStore(runs_dir).load(str(path)).run_id == run.run_id
    assert not list(runs_dir.glob(".*tmp"))  # atomic writes leave no temp files


async def test_resume_only_queries_missing_and_errored_items(
    small_suite: Suite, runs_dir: Path
) -> None:
    store = RunStore(runs_dir)
    key = ("m", "4 squared?")
    flaky = ScriptedClient(PROMPTS, fail_first={key: [NIMRateLimitError("x")] * 10})
    first = await runner(flaky, store, max_retries=0).run(["m"], [small_suite])
    assert first.status == "partial" and sum(1 for r in first.items if r.scored) == 2

    healthy = ScriptedClient(PROMPTS)
    second = await runner(healthy, store).run(["m"], [small_suite], resume=store.load(first.run_id))
    assert healthy.calls == [key]  # only the errored item was re-queried
    assert second.run_id == first.run_id and second.status == "complete"
    assert len(second.items) == 3 and all(r.score == 1.0 for r in second.items)

    # a model added on resume is evaluated on everything, existing ones are untouched
    third_client = ScriptedClient(PROMPTS)
    third = await runner(third_client, store).run(["m", "n"], [small_suite], resume=second)
    assert {m for m, _ in third_client.calls} == {"n"} and len(third_client.calls) == 3
    assert len(third.items) == 6


async def test_resume_reruns_items_whose_definition_changed(
    small_suite: Suite, runs_dir: Path
) -> None:
    store = RunStore(runs_dir)
    first = await runner(ScriptedClient(PROMPTS), store).run(["m"], [small_suite])
    edited = small_suite.model_copy(
        update={
            "items": [
                small_suite.items[0].model_copy(update={"prompt": "What is 1+3?"}),
                *small_suite.items[1:],
            ]
        }
    )
    client = ScriptedClient({**PROMPTS, "What is 1+3?": "Answer: 4"})
    run = await runner(client, store).run(["m"], [edited], resume=first)
    assert client.calls == [("m", "What is 1+3?")]
    assert run.index()[("m", "small", "two-plus-two")].score == 1.0


async def test_progress_callback_reports_every_item(small_suite: Suite) -> None:
    seen: list[tuple[int, int, str]] = []
    await runner(ScriptedClient(PROMPTS)).run(
        ["a", "b"], [small_suite], on_progress=lambda d, t, r: seen.append((d, t, r.item_id))
    )
    assert [d for d, _, _ in seen] == [1, 2, 3, 4, 5, 6] and {t for _, t, _ in seen} == {6}


# ------------------------------------------------------------------ llm-judge integration


def judge_suite() -> Suite:
    return Suite.model_validate(
        {
            "name": "sum",
            "category": "summarization",
            "open_ended": True,
            "grader": {"type": "llm_judge", "rubric": "faithful and short"},
            "items": [{"id": "a", "prompt": "Summarize: cats purr", "reference": "Cats purr."}],
        }
    )


async def test_llm_judge_items_use_the_configured_judge_model() -> None:
    def judge(model: str, messages: list[Any], schema: type[BaseModel]) -> BaseModel:
        return JudgeVerdict(
            score=10 if "Cats purr" in messages[1]["content"] else 1, rationale="ok"
        )

    client = ScriptedClient(lambda m, p: "Cats purr.", judge=judge)
    run = await runner(client, judge_model="judge-x").run(["m"], [judge_suite()])
    (r,) = run.items
    assert r.score == 1.0 and r.passed and "10/10" in r.detail
    assert client.chat_json_calls[0][0] == "judge-x"
    assert run.judge_model == "judge-x"


async def test_judge_failures_mark_the_item_errored_not_wrong() -> None:
    client = ScriptedClient(
        lambda m, p: "x", judge=lambda m, msgs, s: NIMResponseError("judge down", status_code=400)
    )
    run = await runner(client, judge_model="j").run(["m"], [judge_suite()])
    (r,) = run.items
    assert r.error and "judge" in r.error and not r.scored
