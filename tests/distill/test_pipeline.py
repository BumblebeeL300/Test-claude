"""End-to-end runs on the simulator: every recipe, resume, events, checkpoints, cost."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from distill_helpers import DOCS, run_recipe

from nimlab.core import NIMClient, NIMConfig
from nimlab.distill import DistillError, Distiller, RunOptions
from nimlab.distill.gateway import Gateway
from nimlab.distill.costs import Tally, estimate_cost, summarize
from nimlab.distill.models import SAMPLE_STAGES, STAGES
from nimlab.distill.sim import SimClient
from nimlab.distill.store import RunStore, StoreError

RECIPES = ["qa_from_docs", "instruction_diversity", "preference_pairs", "eval_set"]


@pytest.mark.parametrize("recipe", RECIPES)
async def test_full_mock_run_produces_a_dataset_and_visible_drops(
    distiller: Distiller, recipe: str, tmp_path: Path
) -> None:
    result = await run_recipe(distiller, recipe, n=120, out=tmp_path / "out")
    assert result.meta.status == "done" and result.meta.simulated
    assert result.kept, "the dataset must not be empty"
    assert result.dropped, "filters and dedup must visibly do something"
    assert len(result.kept) + len(result.dropped) == 120
    funnel = result.report.funnel
    assert [f.stage for f in funnel] == list(SAMPLE_STAGES)
    assert funnel[0].in_ == 120
    outs = [f.out for f in funnel]
    assert outs == sorted(outs, reverse=True) and outs[-1] == len(result.kept)
    for prev, row in zip(funnel, funnel[1:], strict=False):
        assert row.in_ == prev.out
    assert sum(result.report.drop_reasons.values()) == len(result.dropped)
    filtered = {r for f in funnel for r in f.reasons}
    assert filtered & {"refusal", "too_short", "language", "repetition", "low_score", "low_margin"}
    assert result.report.histograms and result.report.diversity.n == len(result.kept)
    assert result.meta.cost_usd > 0 and result.meta.usage
    names = {p.name for p in result.files}
    assert {"README.md", "quality_report.html"} <= names
    data = next(p for p in result.files if p.suffix == ".jsonl")
    assert len(data.read_text().splitlines()) == len(result.kept)


async def test_recipe_specific_behaviour(distiller: Distiller) -> None:
    qa = await run_recipe(distiller, "qa_from_docs", n=60)
    assert all(s.context and s.source for s in qa.samples)
    assert {s.source.split(" #")[0] for s in qa.samples} == {
        "greenhouse_notes.md",
        "incident_handbook.md",
    }

    pref = await run_recipe(distiller, "preference_pairs", n=60)
    assert all(s.rejected and s.model != s.rejected_model for s in pref.kept)
    assert all(s.score - s.rejected_score >= 0.5 for s in pref.kept)  # type: ignore[operator]

    ev = await run_recipe(distiller, "eval_set", n=60)
    assert all(s.difficulty is not None and 0.3 <= s.difficulty <= 1 for s in ev.kept)
    assert all("weak_response" in s.extra for s in ev.samples)


async def test_qa_falls_back_to_the_builtin_corpus_only_when_simulated(
    distiller: Distiller,
) -> None:
    builtin = await distiller.run("qa_from_docs", n=20)
    assert {s.source.split(" #")[0] for s in builtin.samples} == {
        "retries.md",
        "solar.md",
        "sourdough.md",
    }
    result = await distiller.run("qa_from_docs", n=30, docs=DOCS)
    assert {s.source.split(" #")[0] for s in result.samples} == {
        "greenhouse_notes.md",
        "incident_handbook.md",
    }
    with pytest.raises(DistillError, match="does not exist"):
        await distiller.run("qa_from_docs", n=5, docs=DOCS / "nope")


async def test_quality_filters_and_threshold_overrides(distiller: Distiller) -> None:
    strict = await distiller.run(
        "instruction_diversity",
        n=80,
        options={"filters": {"min_reward": 5.0, "min_judge": 9.9}},
    )
    assert len(strict.kept) < 10  # nothing scores above +5, so almost everything is dropped
    assert "low_score" in strict.report.drop_reasons
    loose = await distiller.run("instruction_diversity", n=80, options={"dedup_threshold": 0.5})
    assert loose.report.drop_reasons.get("duplicate_near", 0) > 0


async def test_decontamination_drops_samples_that_match_the_eval_set(distiller: Distiller) -> None:
    base = await distiller.run("instruction_diversity", n=60)
    leaked = [s.prompt for s in base.kept[:4]]
    assert not base.report.drop_reasons.get("contaminated")
    eval_path = distiller.store.root.parent / "eval.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text("\n".join(json.dumps({"question": q}) for q in leaked))
    result = await distiller.run("instruction_diversity", n=60, eval_set=eval_path)
    contaminated = [s for s in result.samples if s.reason == "contaminated"]
    assert len(contaminated) >= 4
    assert not {s.prompt for s in result.kept} & set(leaked)
    assert result.meta.eval_set == str(eval_path)


async def test_dedup_stage_removes_duplicates_and_reports_clusters(
    make_distiller: Callable[..., Distiller], events: list[tuple[str, dict[str, Any]]]
) -> None:
    result = await make_distiller().run("qa_from_docs", n=90, docs=DOCS)
    kinds = {e["kind"] for t, e in events if t == "dedup.cluster"}
    assert "exact" in kinds
    exact = [s for s in result.samples if s.reason == "duplicate_exact"]
    assert exact and all(s.cluster for s in exact)
    survivors = [s.prompt.lower() for s in result.kept]
    assert len(survivors) == len(set(survivors))


async def test_events_follow_the_documented_protocol(
    make_distiller: Callable[..., Distiller], events: list[tuple[str, dict[str, Any]]]
) -> None:
    result = await make_distiller().run("instruction_diversity", n=60)
    types = [t for t, _ in events]
    assert types[0] == "run.config"
    starts = [d["stage"] for t, d in events if t == "stage.start"]
    ends = [d["stage"] for t, d in events if t == "stage.end"]
    assert starts == ends == list(STAGES)
    assert types.count("sample.generated") == 60
    assert types.count("sample.scored") == 60
    assert types.count("funnel") == len(STAGES)
    dropped = [d for t, d in events if t == "sample.dropped"]
    assert len(dropped) == len(result.dropped)
    assert all(d["reason"] and d["stage"] in SAMPLE_STAGES for d in dropped)
    scored = next(d for t, d in events if t == "sample.scored")
    assert isinstance(scored["score"], float) and scored["source"] in ("reward", "judge")
    last_funnel = [d for t, d in events if t == "funnel"][-1]
    assert last_funnel["kept"] == len(result.kept) and last_funnel["stages"][0]["in"] == 60
    # per stage ordering: stage.start precedes the events of that stage
    assert types.index("stage.start") < types.index("sample.generated") < types.index("sample.scored")


async def test_runs_are_deterministic_and_seed_dependent(runs_dir: Path) -> None:
    async def kept(seed: int, run_id: str) -> list[str]:
        d = Distiller(mock=True, runs_dir=runs_dir)
        r = await d.run("instruction_diversity", n=50, options={"seed": seed}, run_id=run_id)
        return [f"{s.id}:{s.prompt}:{s.score}" for s in r.kept]

    assert await kept(0, "a") == await kept(0, "b")
    assert await kept(0, "c") != await kept(1, "d")


# ------------------------------------------------------------------ checkpoints and resume


async def test_checkpoint_files_exist_after_each_stage(
    distiller: Distiller, runs_dir: Path
) -> None:
    partial = await distiller.run("instruction_diversity", n=40, run_id="p", stop_after="generate")
    assert partial.meta.status == "interrupted"
    assert partial.meta.stages_done == ["seed", "generate"]
    run_dir = runs_dir / "p"
    assert (run_dir / "run.json").is_file() and (run_dir / "seeds.json").is_file()
    rows = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 40 and all(r["score"] is None for r in rows)
    assert not (run_dir / "quality.json").exists()


async def test_resume_after_a_clean_stop_equals_an_uninterrupted_run(runs_dir: Path) -> None:
    straight = await Distiller(mock=True, runs_dir=runs_dir).run(
        "instruction_diversity", n=50, run_id="straight"
    )
    d = Distiller(mock=True, runs_dir=runs_dir)
    stopped = await d.run("instruction_diversity", n=50, run_id="stopped", stop_after="score")
    assert len(stopped.kept) == 50  # nothing is filtered yet
    resumed = await Distiller(mock=True, runs_dir=runs_dir).resume("stopped")
    assert resumed.meta.status == "done"
    assert [s.model_dump() for s in resumed.samples] == [s.model_dump() for s in straight.samples]
    assert resumed.meta.stages_done == list(STAGES)


@pytest.mark.parametrize("recipe", ["instruction_diversity", "preference_pairs", "qa_from_docs"])
async def test_resume_after_a_crash_mid_generation(runs_dir: Path, recipe: str) -> None:
    options = {"domains": ["cooking"], "checkpoint_every": 5}
    kw: dict[str, Any] = {"docs": DOCS} if recipe == "qa_from_docs" else {}
    straight = await Distiller(mock=True, runs_dir=runs_dir).run(
        recipe, n=40, run_id="straight", options=options, **kw
    )
    crashing = Distiller(SimClient(fail_after=30), runs_dir=runs_dir)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await crashing.run(recipe, n=40, run_id="crash", options=options, **kw)
    meta = RunStore(runs_dir).load_meta("crash")
    assert meta.status == "failed" and "simulated crash" in (meta.error or "")
    saved = RunStore(runs_dir).load_samples("crash")
    seeds_done = "seed" in meta.stages_done
    if seeds_done:
        assert 0 < len(saved) < 40 or recipe == "qa_from_docs"
    resumed = await Distiller(mock=True, runs_dir=runs_dir).resume("crash")
    assert resumed.meta.status == "done" and resumed.meta.error is None
    assert [s.model_dump() for s in resumed.samples] == [s.model_dump() for s in straight.samples]


async def test_resume_skips_finished_work(runs_dir: Path) -> None:
    done = await Distiller(mock=True, runs_dir=runs_dir).run(
        "instruction_diversity", n=30, run_id="done"
    )
    calls: list[str] = []
    again = Distiller(mock=True, runs_dir=runs_dir, on_event=lambda t, d: calls.append(t))
    result = await again.resume("done")
    assert "stage.start" not in calls and len(result.kept) == len(done.kept)


async def test_resume_errors(runs_dir: Path) -> None:
    d = Distiller(mock=True, runs_dir=runs_dir)
    with pytest.raises(StoreError, match="no such run"):
        await d.resume("ghost")
    with pytest.raises(StoreError, match="invalid run id"):
        await d.resume("../etc")
    await d.run("eval_set", n=10, run_id="dup")
    with pytest.raises(DistillError, match="already exists"):
        await d.run("eval_set", n=10, run_id="dup")


async def test_cancellation_leaves_a_resumable_checkpoint(runs_dir: Path) -> None:
    started = asyncio.Event()

    def hook(type_: str, data: dict[str, Any]) -> None:
        if type_ == "stage.start" and data["stage"] == "generate":
            started.set()

    slow = Distiller(SimClient(delay=0.01), runs_dir=runs_dir, on_event=hook)
    task = asyncio.create_task(slow.run("instruction_diversity", n=60, run_id="c", options={"domains": ["cooking"]}))
    await asyncio.wait_for(started.wait(), 5)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert RunStore(runs_dir).load_meta("c").status == "interrupted"
    resumed = await Distiller(mock=True, runs_dir=runs_dir).resume("c")
    assert resumed.meta.status == "done" and resumed.kept


# ------------------------------------------------------------------ validation and plumbing


async def test_plan_validation(distiller: Distiller) -> None:
    with pytest.raises(DistillError, match="dpo format needs"):
        await distiller.run("instruction_diversity", n=5, fmt="dpo")
    with pytest.raises(DistillError, match="unknown format"):
        await distiller.run("eval_set", n=5, fmt="parquet")
    topped_up = distiller.plan("preference_pairs", models=["only/one"])[2]
    assert topped_up[0] == "only/one" and len(topped_up) == 2  # a second model is added
    with pytest.raises(DistillError, match="n must"):
        await distiller.run("eval_set", n=0)
    with pytest.raises(DistillError, match="invalid options"):
        await distiller.run("eval_set", n=5, options={"nonsense": 1})
    with pytest.raises(DistillError, match="unknown stage"):
        await distiller.run("eval_set", n=5, stop_after="nope")
    with pytest.raises(DistillError, match="eval set"):
        await distiller.run("eval_set", n=5, eval_set="/no/such/file.jsonl")


async def test_default_and_custom_model_panels(distiller: Distiller) -> None:
    _, _, panel, fmt = distiller.plan("instruction_diversity")
    assert len(panel) == 2 and fmt == "openai-chat"
    assert distiller.plan("eval_set")[2] == ["deepseek-ai/deepseek-r1"]
    assert distiller.plan("preference_pairs")[3] == "dpo"
    _, _, panel, _ = distiller.plan("preference_pairs", models=["a/x", "b/y", "a/x"])
    assert panel == ["a/x", "b/y"]
    result = await distiller.run("instruction_diversity", n=40, models=["a/x", "b/y", "c/z"])
    assert {s.model for s in result.samples} == {"a/x", "b/y", "c/z"}


async def test_models_and_temperatures_are_mixed(distiller: Distiller) -> None:
    result = await distiller.run("instruction_diversity", n=60)
    assert len({s.model for s in result.samples}) == 2
    assert {s.temperature for s in result.samples} == {0.4, 0.7, 1.0}


async def test_error_samples_are_dropped_not_fatal(runs_dir: Path) -> None:
    from nimlab.core import NIMResponseError

    class Flaky(SimClient):
        async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
            if messages[-1]["role"] == "user" and "number 7" in messages[-1]["content"]:
                raise NIMResponseError("upstream 500", status_code=500)
            return await super().chat(model, messages, **kw)

    result = await Distiller(Flaky(), runs_dir=runs_dir).run("instruction_diversity", n=30)
    assert result.meta.status == "done"
    assert all(s.reason == "error" for s in result.dropped if s.dropped_at == "generate")


async def test_gateway_bounds_concurrency_and_tallies_usage() -> None:
    live = peak = 0

    class Counting(SimClient):
        async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.005)
            try:
                return await super().chat(model, messages, **kw)
            finally:
                live -= 1

    from nimlab.distill.prompts import answer_messages

    tally = Tally()
    gw = Gateway(Counting(), tally, concurrency=3)
    await asyncio.gather(*(gw.chat("m/x", answer_messages(f"Explain {i}")) for i in range(20)))
    assert peak == 3
    assert tally.models["m/x"]["calls"] == 20 and tally.total_tokens > 0


def test_cost_estimate_uses_cost_tiers() -> None:
    cheap = {"meta/llama-3.1-8b-instruct": {"calls": 1, "prompt_tokens": 1_000_000, "completion_tokens": 0}}
    pricey = {"meta/llama-3.1-405b-instruct": {"calls": 1, "prompt_tokens": 1_000_000, "completion_tokens": 0}}
    assert estimate_cost(pricey) > estimate_cost(cheap) > 0
    assert estimate_cost({"unknown/model": {"prompt_tokens": 1_000_000}}) == 0.6
    s = summarize(cheap)
    assert s["total_tokens"] == 1_000_000 and s["total_calls"] == 1


async def test_export_other_formats_after_the_run(distiller: Distiller, tmp_path: Path) -> None:
    result = await distiller.run("preference_pairs", n=50)
    for fmt in ("dpo", "openai-chat", "alpaca", "sharegpt"):
        path = distiller.export(result.run_id, fmt, tmp_path / f"x.{fmt}")
        assert len(path.read_text().splitlines()) == len(result.kept)
    plain = await distiller.run("instruction_diversity", n=30)
    with pytest.raises(DistillError, match="dpo"):
        distiller.export(plain.run_id, "dpo")
    meta_path = distiller.export(plain.run_id, "openai-chat", tmp_path / "m.jsonl", include_meta=True)
    assert "metadata" in json.loads(meta_path.read_text().splitlines()[0])
    assert distiller.load(plain.run_id).meta.id == plain.run_id
    assert [m.id for m in distiller.list_runs()][:2] == sorted(
        [plain.run_id, result.run_id], reverse=True
    ) or len(distiller.list_runs()) == 2


async def test_real_mode_without_key_falls_back_to_the_simulator(runs_dir: Path) -> None:
    client = NIMClient(NIMConfig(mock=True))
    d = Distiller(client, runs_dir=runs_dir)
    result = await d.run("instruction_diversity", n=20)
    assert result.meta.simulated and client.usage.total_calls > 0  # usage flows to the shared tracker
    assert d.uses_simulator()
    await client.aclose()


def test_run_options_validation() -> None:
    assert RunOptions().checkpoint_every == 25
    with pytest.raises(ValueError):
        RunOptions(concurrency=0)
    with pytest.raises(ValueError):
        RunOptions(bogus=1)  # type: ignore[call-arg]
