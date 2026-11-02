"""The offline simulator that makes mock-mode results meaningful."""

from __future__ import annotations

import asyncio
import time

import httpx

from nimlab.arena.battle import PairwiseVerdict, build_pairwise_messages
from nimlab.arena.graders import JudgeVerdict
from nimlab.arena.graders.judge import build_judge_messages
from nimlab.arena.llm import default_models, is_transient, make_client
from nimlab.arena.router import CategoryChoice
from nimlab.arena.runner import Runner, RunSettings
from nimlab.arena.simulate import SimulatedClient, model_profile
from nimlab.arena.suites import load_suites
from nimlab.core import NIMClient, NIMConfig
from nimlab.core.errors import NIMAuthError, NIMError, NIMRateLimitError, NIMResponseError

BIG = "meta/llama-3.1-405b-instruct"
SMALL = "meta/llama-3.2-1b-instruct"
CODER = "qwen/qwen2.5-coder-32b-instruct"
REASONER = "deepseek-ai/deepseek-r1"


def test_profiles_are_deterministic_and_reflect_model_size_and_role() -> None:
    assert model_profile(BIG) == model_profile(BIG)
    big, small = model_profile(BIG), model_profile(SMALL)
    assert (
        big.skill > small.skill
        and big.ttft_s > small.ttft_s
        and big.tokens_per_s < small.tokens_per_s
    )
    coder = model_profile(CODER)
    assert coder.p_correct("coding") > coder.p_correct("math_reasoning") + 0.15
    assert model_profile(REASONER).p_correct("math_reasoning") > model_profile(REASONER).p_correct(
        "coding"
    )
    assert model_profile(REASONER).ttft_s > 2 * model_profile("meta/llama-3.3-70b-instruct").ttft_s
    unknown = model_profile("acme/mystery-model")
    assert 1 <= unknown.tier <= 5 and 0.05 <= unknown.p_correct("coding") <= 0.98


async def test_answers_are_deterministic_and_come_from_the_suites() -> None:
    suite = load_suites()["math_reasoning"]
    item = suite.items[0]
    client = SimulatedClient(time_scale=0.0)
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": item.prompt}]
    a = await client.chat(BIG, msgs)
    b = await client.chat(BIG, msgs)
    assert a.content == b.content and a.usage.completion_tokens > 0
    assert a.content in (item.reference, "Quick estimate.\nAnswer: 228")  # right or plainly wrong
    streamed = "".join([c async for c in client.stream_chat(BIG, msgs)])
    assert streamed == a.content


async def test_stronger_models_score_higher_on_every_deterministic_suite() -> None:
    suites = [s for n, s in load_suites().items() if n != "summarization"]
    runner = Runner(SimulatedClient(time_scale=0.0), settings=RunSettings(concurrency=16))
    run = await runner.run([BIG, SMALL], suites)
    assert all(r.scored for r in run.items)

    def mean(model: str) -> float:
        rs = [r.score for r in run.items if r.model == model]
        return sum(rs) / len(rs)

    assert mean(BIG) > mean(SMALL) + 0.15
    coding = {m: sum(r.score for r in run.for_model(m, "coding")) for m in (BIG, SMALL)}
    assert coding[BIG] >= coding[SMALL]


async def test_specialists_win_their_own_category() -> None:
    runner = Runner(SimulatedClient(time_scale=0.0), settings=RunSettings(concurrency=16))
    suites = [load_suites()["coding"], load_suites()["math_reasoning"]]
    run = await runner.run([CODER, "meta/llama-3.3-70b-instruct"], suites)

    def score(model: str, suite: str) -> float:
        rs = run.for_model(model, suite)
        return sum(r.score for r in rs) / len(rs)

    assert score(CODER, "coding") > score("meta/llama-3.3-70b-instruct", "coding")


async def test_wrong_answers_are_plausible_and_fail_their_graders() -> None:
    client = SimulatedClient(time_scale=0.0)
    runner = Runner(client, settings=RunSettings(concurrency=16))
    run = await runner.run([SMALL], list(load_suites().values()))
    wrong = [r for r in run.items if not r.passed]
    assert wrong, "a 1B model should miss something"
    assert all(r.output.strip() for r in run.items)
    coding_wrong = [r for r in wrong if r.suite == "coding"]
    assert all("```python" in r.output or r.score > 0 for r in coding_wrong)


async def test_virtual_time_is_scaled_and_reported_unscaled() -> None:
    suite = load_suites()["factual_qa"].limited(2)
    client = SimulatedClient(time_scale=0.04)
    assert client.time_scale == 0.04
    start = time.perf_counter()
    run = await Runner(client, settings=RunSettings(concurrency=4)).run([BIG], [suite])
    wall = time.perf_counter() - start
    prof = model_profile(BIG)
    for r in run.items:
        assert r.ttft_s is not None and prof.ttft_s * 0.8 < r.ttft_s < prof.ttft_s * 3
        assert r.latency_s >= r.ttft_s
    assert wall < 2.0  # real time is ~25x shorter than the reported (virtual) time


async def test_ttft_and_throughput_differ_between_models() -> None:
    suite = load_suites()["instruction_following"].limited(4)
    run = await Runner(SimulatedClient(), settings=RunSettings(concurrency=8)).run(
        [BIG, SMALL], [suite]
    )

    def med(model: str, attr: str) -> float:
        vals = sorted(getattr(r, attr) for r in run.for_model(model))
        return vals[len(vals) // 2]

    assert med(BIG, "ttft_s") > med(SMALL, "ttft_s")
    assert med(BIG, "latency_s") > med(SMALL, "latency_s")


async def test_judge_scores_by_overlap_with_the_reference() -> None:
    client = SimulatedClient(time_scale=0.0)
    item = load_suites()["summarization"].items[0]
    ref = item.reference or ""

    async def score(answer: str) -> int:
        msgs = build_judge_messages(prompt=item.prompt, response=answer, rubric="r", reference=ref)
        verdict = await client.chat_json("judge", msgs, JudgeVerdict)
        assert 1 <= verdict.score <= 10 and verdict.rationale
        return verdict.score

    assert await score(ref) >= 9
    assert await score("Bananas are yellow and grow in bunches.") <= 3
    assert await score(" ".join(ref.split()[: len(ref.split()) // 2])) < await score(ref)


async def test_pairwise_judge_prefers_the_closer_answer_and_has_mild_position_bias() -> None:
    client = SimulatedClient(time_scale=0.0)
    item = load_suites()["summarization"].items[0]
    good, bad = item.reference or "", "Something about turbines? Not sure."

    async def pick(a: str, b: str) -> str:
        msgs = build_pairwise_messages(prompt=item.prompt, answer_a=a, answer_b=b, reference=good)
        return (await client.chat_json("judge", msgs, PairwiseVerdict)).winner

    assert await pick(good, bad) == "A" and await pick(bad, good) == "B"
    assert await pick(good, good) in ("A", "tie")  # identical answers: bias towards slot A or a tie


async def test_category_classifier_and_unknown_schemas_delegate_sensibly() -> None:
    client = SimulatedClient(time_scale=0.0)
    choice = await client.chat_json(
        "fast",
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "Write a python function"},
        ],
        CategoryChoice,
    )
    assert choice.category == "coding"

    class Other(JudgeVerdict):  # not recognised by name: falls through to the core mock backend
        pass

    got = await client.chat_json("m", [{"role": "user", "content": "whatever"}], Other)
    assert 1 <= got.score <= 10


async def test_unknown_prompts_fall_back_to_the_core_mock_and_embeddings_work() -> None:
    client = SimulatedClient(time_scale=0.0)
    r = await client.chat("some/model", [{"role": "user", "content": "Tell me about volcanoes"}])
    assert r.content and "volcanoes" in r.content.lower()
    pieces = [
        c
        async for c in client.stream_chat(
            "some/model", [{"role": "user", "content": "Tell me about volcanoes"}]
        )
    ]
    assert "".join(pieces)
    vecs = await client.embed("embed-model", ["hello world", "hello there"], input_type="query")
    assert len(vecs) == 2 and len(vecs[0]) == len(vecs[1]) > 100
    assert "meta/llama-3.3-70b-instruct" in await client.list_models()
    assert client.mock is True


async def test_custom_suites_can_be_registered_with_the_simulator() -> None:
    from arena_fakes import make_suite

    suite = make_suite("mine", [("q", "What is the secret number?", 42)])
    client = SimulatedClient(time_scale=0.0)
    suite.items[0].reference = "Answer: 42"
    client.register_suites([suite])
    outs = {
        (await client.chat(m, [{"role": "user", "content": "What is the secret number?"}])).content
        for m in (BIG, SMALL, CODER, REASONER)
    }
    assert outs <= {"Answer: 42", "Quick estimate.\nAnswer: 66"}
    assert "Answer: 42" in outs


async def test_usage_is_tracked() -> None:
    client = SimulatedClient(time_scale=0.0)
    item = load_suites()["factual_qa"].items[0]
    await client.chat(BIG, [{"role": "user", "content": item.prompt}])
    assert client.usage.summary()["models"][BIG]["calls"] == 1


# ------------------------------------------------------------------ llm helpers


def test_make_client_chooses_the_simulator_in_mock_mode_and_the_real_client_otherwise() -> None:
    assert isinstance(make_client(mock=True), SimulatedClient)
    real = make_client(mock=False, config=NIMConfig(api_key="nvapi-x", mock=False))
    assert isinstance(real, NIMClient) and not real.mock
    assert isinstance(make_client(), SimulatedClient)  # no API key in the test environment
    forced = make_client(config=NIMConfig(api_key="nvapi-x", mock=False), mock=True)
    assert isinstance(forced, SimulatedClient)


def test_default_models_are_distinct_registry_picks() -> None:
    models = default_models()
    assert len(models) == len(set(models)) == 4 and "deepseek-ai/deepseek-r1" in models
    assert len(default_models(limit=2)) == 2


def test_transient_error_classification() -> None:
    assert is_transient(NIMRateLimitError("429"))
    assert is_transient(NIMResponseError("502", status_code=502))
    assert is_transient(NIMResponseError("408", status_code=408))
    assert is_transient(NIMError("connection reset"))
    assert is_transient(TimeoutError()) and is_transient(asyncio.TimeoutError())
    assert not is_transient(NIMAuthError("401", status_code=401))
    assert not is_transient(NIMResponseError("400", status_code=400))
    assert not is_transient(ValueError("nope"))
    assert is_transient(httpx.ConnectError("x"))


async def test_items_without_a_reference_still_get_correct_answers_for_simple_graders() -> None:
    from nimlab.arena.suites import Suite

    suite = Suite.model_validate(
        {
            "name": "noref",
            "items": [
                {
                    "id": "n",
                    "prompt": "Pick a number",
                    "grader": {"type": "numeric", "expected": 7},
                },
                {
                    "id": "e",
                    "prompt": "Pick a word",
                    "grader": {"type": "exact", "expected": ["alpha", "beta"]},
                },
                {
                    "id": "c",
                    "prompt": "Name two fruits",
                    "grader": {
                        "type": "contains_all",
                        "required": ["apple", "pear"],
                        "any_of": [["red", "green"]],
                    },
                },
            ],
        }
    )
    run = await Runner(SimulatedClient(time_scale=0.0)).run(
        [BIG, "meta/llama-3.1-70b-instruct", CODER, REASONER], [suite]
    )
    passed = [r for r in run.items if r.passed]
    assert passed, "at least some strong models must answer correctly from the grader parameters"
    outputs = {r.item_id: r.output for r in passed}
    assert set(outputs) <= {"n", "e", "c"}
    for r in passed:
        assert r.score == 1.0
    assert all(r.output for r in run.items)
