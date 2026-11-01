"""SmartRouter: classification (kNN + keywords + LLM), utility-based selection and explanations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from arena_fakes import ScriptedClient
from pydantic import BaseModel

from nimlab.arena.leaderboard import Leaderboard
from nimlab.arena.results import ItemResult, RunResult, RunStore, SuiteInfo
from nimlab.arena.router import (
    CATEGORY_ROLES,
    PROFILES,
    CategoryChoice,
    SmartRouter,
    keyword_scores,
    load_exemplars,
    prompt_from_messages,
)
from nimlab.arena.suites import CATEGORIES
from nimlab.core import Role, default_registry
from nimlab.core.errors import NIMError
from nimlab.core.mock import mock_embeddings

CODER = "qwen/qwen2.5-coder-32b-instruct"  # tier 2
REASONER = "deepseek-ai/deepseek-r1"  # tier 4
FASTM = "meta/llama-3.1-8b-instruct"  # tier 1
BIG = "meta/llama-3.1-405b-instruct"  # tier 5

# model -> {suite: (score, latency_s)}; suites map to categories below
TABLE = {
    CODER: {"code": (0.95, 1.5), "math": (0.40, 1.5), "facts": (0.70, 1.0)},
    REASONER: {"code": (0.80, 6.0), "math": (0.97, 6.0), "facts": (0.85, 4.0)},
    FASTM: {"code": (0.55, 0.4), "math": (0.50, 0.4), "facts": (0.60, 0.3)},
    BIG: {"code": (0.85, 5.0), "math": (0.90, 5.0), "facts": (0.92, 3.0)},
}
CATS = {"code": "coding", "math": "math_reasoning", "facts": "factual_qa"}


def make_run(table: dict = TABLE) -> RunResult:
    run = RunResult.new(models=list(table))
    run.suites = {s: SuiteInfo(name=s, category=c, n_items=2) for s, c in CATS.items()}
    for model, suites in table.items():
        for suite, (score, lat) in suites.items():
            for i in range(2):
                run.items.append(
                    ItemResult(
                        model=model, suite=suite, item_id=f"i{i}", score=score, latency_s=lat
                    )
                )
    return run


@pytest.fixture
def lb() -> Leaderboard:
    return Leaderboard.from_run(make_run(), n_boot=0)


def router(lb: Leaderboard | None, **kw: Any) -> SmartRouter:
    return (
        SmartRouter(lb, **kw)
        if lb
        else SmartRouter(runs_dir=kw.pop("runs_dir", "/nonexistent"), **kw)
    )


# ------------------------------------------------------------------ classification


@pytest.mark.parametrize(
    ("prompt", "category"),
    [
        ("Write a Python function that merges two sorted lists", "coding"),
        ("Why does my SQL query return duplicate rows after the join? Here is the code", "coding"),
        ("```js\nconsole.log(1)\n``` fix this bug", "coding"),
        ("What is 17% of 340 and then add 12?", "math_reasoning"),
        ("Solve the equation 5x - 3 = 22", "math_reasoning"),
        (
            "A train travels 120 km in 90 minutes. What is its average speed in km/h?",
            "math_reasoning",
        ),
        ("What is the capital of Peru?", "factual_qa"),
        ("Who invented the telephone?", "factual_qa"),
        (
            "Write exactly three sentences about dogs, all in lowercase, without any commas",
            "instruction_following",
        ),
        (
            "Answer in no more than 12 words and end with the phrase That is all",
            "instruction_following",
        ),
        ("Return a JSON object with keys name and age for this person", "json_compliance"),
        ("Output valid JSON matching this schema with the fields id and label", "json_compliance"),
        ("Summarize the following article in two sentences", "summarization"),
        ("tl;dr of this email thread please", "summarization"),
        ("Hi there, how are you today?", "general_chat"),
        ("Can you recommend a good book for a long train ride?", "general_chat"),
    ],
)
async def test_knn_classification_with_local_embeddings(prompt: str, category: str) -> None:
    cls = await SmartRouter(runs_dir="/nonexistent").classify(prompt)
    assert cls.category == category, (cls.method, cls.scores, cls.notes)


async def test_knn_is_used_when_confident_and_exposes_neighbours() -> None:
    cls = await SmartRouter(runs_dir="/nonexistent").classify(
        "Write a Python function that reverses a string"
    )
    assert cls.method == "knn" and cls.confidence >= 0.5
    assert cls.neighbors and cls.neighbors[0].category == "coding"
    assert 0 < cls.neighbors[0].similarity <= 1
    assert abs(sum(cls.scores.values()) - 1.0) < 1e-3


@pytest.mark.parametrize(
    ("prompt", "category"),
    [
        ("python function", "coding"),
        ("calculate 12 * 7", "math_reasoning"),
        ("exactly 5 words lowercase", "instruction_following"),
        ("json with keys", "json_compliance"),
        ("summarize this", "summarization"),
        ("hello", "general_chat"),
    ],
)
def test_keyword_heuristic(prompt: str, category: str) -> None:
    scores = keyword_scores(prompt)
    assert max(scores, key=lambda c: scores[c]) == category
    assert SmartRouter(runs_dir="/nonexistent").classify_keywords(prompt).category == category


def test_keywords_default_to_general_chat_without_signal() -> None:
    cls = SmartRouter(runs_dir="/nonexistent").classify_keywords("zzz qqq www")
    assert cls.category == "general_chat" and cls.method == "default"
    assert sum(keyword_scores("zzz").values()) == 0
    assert (
        keyword_scores("lorem ipsum " * 100)["summarization"] > 0
    )  # long text leans to summarising


async def test_classifier_falls_back_to_keywords_when_knn_is_unsure() -> None:
    r = SmartRouter(runs_dir="/nonexistent", min_similarity=0.99)  # kNN can never be confident
    cls = await r.classify("Summarize the following passage in two sentences")
    assert cls.method == "keywords" and cls.category == "summarization"
    assert any("kNN unsure" in n for n in cls.notes)
    kw_only = await SmartRouter(runs_dir="/nonexistent", classifier="keywords").classify(
        "python function"
    )
    assert kw_only.method == "keywords"
    knn_only = await SmartRouter(runs_dir="/nonexistent", classifier="knn").classify(
        "python function"
    )
    assert knn_only.method in ("knn", "default")


async def test_embedding_failures_fall_back_to_keywords_and_back_off() -> None:
    def broken(texts: list[str]) -> list[list[float]]:
        raise NIMError("embedding endpoint down")

    client = ScriptedClient(embed_fn=broken)
    r = SmartRouter(client=client, runs_dir="/nonexistent")
    cls = await r.classify("Write a python function to sort a list")
    assert cls.method == "keywords" and cls.category == "coding"
    assert any("embedding unavailable" in n for n in cls.notes)
    await r.classify("another prompt")
    assert len(client.embed_calls) == 1  # no hammering of a failing endpoint within the back-off


async def test_client_embeddings_are_used_with_query_and_passage_types() -> None:
    def tiny(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            low = t.lower()
            out.append([float("python" in low or "code" in low), float("capital" in low), 0.01])
        return out

    client = ScriptedClient(embed_fn=tiny)
    r = SmartRouter(client=client, runs_dir="/nonexistent", min_similarity=0.0)
    cls = await r.classify("Write python code")
    assert cls.category == "coding" and cls.method == "knn"
    kinds = [(n, t) for _, n, t in client.embed_calls]
    assert kinds[0] == (len(load_exemplars()), "passage") and kinds[1] == (1, "query")
    await r.classify("another")
    assert len(client.embed_calls) == 3  # exemplar matrix is cached


async def test_llm_classifier_via_fast_model() -> None:
    seen: list[tuple[str, str]] = []

    def judge(model: str, messages: list[Any], schema: type[BaseModel]) -> BaseModel:
        seen.append((model, schema.__name__))
        return CategoryChoice(category="json_compliance", reason="asks for structured output")

    client = ScriptedClient(judge=judge, embed_fn=lambda t: [[1.0, 0.0] for _ in t])
    r = SmartRouter(client=client, runs_dir="/nonexistent", classifier="llm")
    cls = await r.classify("anything at all")
    assert cls.method == "llm" and cls.category == "json_compliance"
    assert seen == [(default_registry().pick(Role.FAST), "CategoryChoice")]
    assert "structured output" in cls.notes[0]
    # use_llm: only consulted when the cheaper classifiers are unsure
    queries = {"zzz qqq", "Summarize this article"}

    def orthogonal(texts: list[str]) -> list[list[float]]:  # queries are far from every exemplar
        return [[0.0, 1.0] if t in queries else [1.0, 0.0] for t in texts]

    client2 = ScriptedClient(judge=judge, embed_fn=orthogonal)
    r2 = SmartRouter(client=client2, runs_dir="/nonexistent", use_llm=True)
    assert (await r2.classify("zzz qqq")).method == "llm"  # kNN and keywords both unsure
    assert (await r2.classify("Summarize this article")).method == "keywords"  # keywords are sure
    assert len(client2.chat_json_calls) == 1


async def test_llm_classifier_errors_degrade_gracefully() -> None:
    client = ScriptedClient(judge=lambda *a: NIMError("down"), embed_fn=mock_embeddings)  # type: ignore[arg-type,return-value]
    r = SmartRouter(client=client, runs_dir="/nonexistent", classifier="llm")
    cls = await r.classify("Write a python function to sort")
    assert cls.category == "coding"  # fell back to kNN/keywords


def test_exemplars_cover_every_category_and_are_unique() -> None:
    ex = load_exemplars()
    cats = {c for c, _ in ex}
    assert cats == set(CATEGORIES)
    assert len({t for _, t in ex}) == len(ex)
    for cat in CATEGORIES:
        assert sum(1 for c, _ in ex if c == cat) >= 10


def test_prompt_from_messages_extracts_last_user_text_and_images() -> None:
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        },
    ]
    assert prompt_from_messages(msgs) == ("describe", True)
    assert prompt_from_messages(msgs[:3]) == ("first", False)
    assert prompt_from_messages([]) == ("", False)


# ------------------------------------------------------------------ utility selection


async def test_coding_prompts_go_to_the_coding_specialist(lb: Leaderboard) -> None:
    d = await router(lb).route("Write a Python function that merges overlapping intervals")
    assert d.category == "coding" and d.model == CODER and d.source == "leaderboard"
    assert d.run_id == lb.run_id and d.candidates[0].model == CODER


async def test_math_prompts_go_to_the_reasoner_under_the_quality_profile(lb: Leaderboard) -> None:
    d = await router(lb, profile="quality").route("Solve 3x + 7 = 28 and show the steps")
    assert d.category == "math_reasoning" and d.model == REASONER


async def test_latency_weight_flips_the_choice_to_the_faster_model(lb: Leaderboard) -> None:
    prompt = "Solve the equation 5x - 3 = 22"
    slow = await router(lb, profile="quality").route(prompt)
    fast = await router(lb, profile="speed").route(prompt)
    assert slow.model == REASONER
    assert fast.model != REASONER and fast.model in (BIG, CODER, FASTM)
    custom = await router(lb, latency_weight=1.0, cost_weight=0.0).route(prompt)
    assert custom.model == FASTM  # a huge latency penalty makes the 0.4s model win


async def test_cost_weight_flips_the_choice_to_the_cheaper_model(lb: Leaderboard) -> None:
    prompt = "What is the capital of Peru?"
    free = await router(lb, latency_weight=0.0, cost_weight=0.0).route(prompt)
    thrifty = await router(lb, latency_weight=0.0, cost_weight=0.2).route(prompt)
    assert free.model == BIG  # 0.92 on factual QA
    assert thrifty.model != BIG  # tier 5 costs 1.0 utility
    economy = await router(lb, profile="economy").route(prompt)
    assert economy.model in (REASONER, CODER, FASTM)


async def test_utility_arithmetic_is_exact_and_candidates_are_sorted(lb: Leaderboard) -> None:
    d = await router(lb, latency_weight=0.05, cost_weight=0.1).route(
        "Write a Python function", category="coding"
    )
    assert (d.latency_weight, d.cost_weight, d.profile) == (0.05, 0.1, "balanced")
    reg = default_registry()
    for c in d.candidates:
        q = lb.category_quality(c.model, "coding")
        lat = lb.category_latency(c.model, "coding")
        want = q - 0.05 * lat - 0.1 * reg.get(c.model).cost_tier
        assert c.utility == pytest.approx(want, abs=2e-4)
        assert c.quality_term == pytest.approx(q, abs=1e-3)
        assert c.latency_penalty == pytest.approx(0.05 * lat, abs=1e-3)
        assert c.cost_penalty == pytest.approx(0.1 * reg.get(c.model).cost_tier)
    utilities = [c.utility for c in d.candidates]
    assert utilities == sorted(utilities, reverse=True) and d.model == d.candidates[0].model


async def test_decision_is_explainable(lb: Leaderboard) -> None:
    d = await router(lb).route("Write a Python function that parses a CSV file")
    text = " ".join(d.reasons)
    assert "coding" in text and "kNN" in text  # why this category
    assert "utility = quality" in text and "latency" in text and "cost_tier" in text
    assert d.model in text and "Runner-up" in text and lb.run_id in text
    assert d.explain().startswith(f"Routed to {d.model}")
    assert d.classification and d.classification.neighbors
    json.loads(d.model_dump_json())  # fully serialisable


async def test_general_chat_uses_overall_quality(lb: Leaderboard) -> None:
    d = await router(lb, profile="quality").route("hello, how are you doing today?")
    assert d.category == "general_chat"
    best_overall = lb.standings[0].model
    assert d.model == best_overall


async def test_models_without_category_data_use_their_overall_score() -> None:
    table = {**TABLE, "x/newcomer": {"math": (1.0, 0.5)}}
    lb2 = Leaderboard.from_run(make_run(table), n_boot=0)
    d = await router(lb2, profile="quality").route(
        "def f(): pass  # fix this python bug", category="coding"
    )
    newcomer = next(c for c in d.candidates if c.model == "x/newcomer")
    assert newcomer.quality_source == "overall" and newcomer.quality == pytest.approx(1.0)
    assert any("lack coding data" in r for r in d.reasons)


async def test_categories_without_suites_fall_back_to_overall_quality(lb: Leaderboard) -> None:
    d = await router(lb, profile="quality").route("Summarize this article in two sentences")
    assert d.category == "summarization" and d.source == "leaderboard"
    assert all(c.quality_source == "overall" for c in d.candidates)


async def test_allow_list_restricts_candidates(lb: Leaderboard) -> None:
    d = await router(lb, allow=[FASTM, BIG]).route("Write a python function", category="coding")
    assert {c.model for c in d.candidates} == {FASTM, BIG}
    d2 = await router(lb).route("Write a python function", category="coding", allow=[FASTM])
    assert d2.model == FASTM


async def test_explicit_category_and_profile_overrides(lb: Leaderboard) -> None:
    r = router(lb)
    d = await r.route("hello", category="coding", profile="quality")
    assert d.classifier == "override" and d.category == "coding" and d.profile == "quality"
    with pytest.raises(ValueError, match="unknown category"):
        await r.route("x", category="poetry")
    with pytest.raises(ValueError, match="unknown profile"):
        await r.route("x", profile="warp")
    assert set(PROFILES) == {"quality", "balanced", "speed", "economy"}


async def test_chat_messages_are_accepted_and_images_route_to_vision(lb: Leaderboard) -> None:
    msgs = [{"role": "user", "content": "Write a Python function that sorts a list"}]
    assert (await router(lb).route(msgs)).category == "coding"
    with_image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        }
    ]
    d = await router(lb).route(with_image)
    assert d.category == "vision" and d.classifier == "vision" and d.source == "registry"
    assert d.model == default_registry().pick(Role.VISION)


async def test_registry_fallback_without_leaderboard_data() -> None:
    reg = default_registry()
    r = SmartRouter(runs_dir="/nonexistent")
    assert r.leaderboard is None
    for category, role in CATEGORY_ROLES.items():
        d = await r.route("anything", category=category)
        assert d.source == "registry" and d.model == reg.pick(role) and d.run_id is None
        assert "registry default" in " ".join(d.reasons)
        assert d.candidates and d.candidates[0].utility is None
    coding = await r.route("Write a python function to sort", category="coding")
    assert coding.model == reg.pick(Role.CODER)


async def test_registry_fallback_honours_allow_and_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NIMLAB_MODEL_CODER", "my/custom-coder")
    d = await SmartRouter(runs_dir="/nonexistent").route("x", category="coding")
    assert d.model == "my/custom-coder"
    d2 = await SmartRouter(runs_dir="/nonexistent", allow=[FASTM]).route("x", category="coding")
    assert d2.model == FASTM


# ------------------------------------------------------------------ latest leaderboard data


def test_router_loads_the_latest_runs_and_reloads_when_they_change(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    r = SmartRouter(runs_dir=tmp_path)
    assert r.leaderboard is None
    old = make_run()
    old.run_id = "20250101T000000Z"
    store.save(old)
    assert r.leaderboard is not None and r.leaderboard.run_id.startswith("merged-20250101")
    assert {s.model for s in r.leaderboard.standings} == set(TABLE)

    # a newer run overrides older results for the same (model, suite, item)
    newer = make_run({CODER: {"code": (0.1, 1.0)}})
    newer.run_id = "20260101T000000Z"
    store.save(newer)
    d = r.route_sync("Write a Python function", category="coding", profile="quality")
    assert d.run_id == "merged-20260101T000000Z"
    assert r.leaderboard.get(CODER).suite_scores["code"] == pytest.approx(0.1)  # type: ignore[union-attr]
    assert r.leaderboard.get(REASONER).suite_scores["math"] == pytest.approx(0.97)  # type: ignore[union-attr]


def test_set_leaderboard_disables_reloading(lb: Leaderboard, tmp_path: Path) -> None:
    r = SmartRouter(runs_dir=tmp_path)
    r.set_leaderboard(lb)
    RunStore(tmp_path).save(make_run({FASTM: {"code": (0.2, 1.0)}}))
    assert r.leaderboard is lb


def test_route_sync_wrapper(lb: Leaderboard) -> None:
    assert router(lb).route_sync("Write a Python function to sort").category == "coding"


def test_exemplar_vectors_are_unit_length_and_dimension_consistent() -> None:
    m = np.asarray(mock_embeddings([t for _, t in load_exemplars()]))
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
