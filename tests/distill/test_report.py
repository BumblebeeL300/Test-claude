"""The quality report, the HTML page and the dataset card."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from distill_helpers import run_recipe

from nimlab.distill import Distiller, DistillResult
from nimlab.distill.card import render_card
from nimlab.distill.models import Sample
from nimlab.distill.quality import build_report, compute_funnel, histogram
from nimlab.distill.report import histogram_svg, render_report_html, render_terminal


@pytest.fixture
async def result(distiller: Distiller) -> DistillResult:
    return await run_recipe(distiller, "instruction_diversity", n=100)


def make(i: int, **kw: object) -> Sample:
    base: dict[str, object] = {
        "id": f"s{i}",
        "seed_idx": i,
        "prompt": f"prompt {i}",
        "response": "answer",
        "model": "m",
    }
    return Sample(**{**base, **kw})  # type: ignore[arg-type]


def test_funnel_counts_per_stage() -> None:
    samples = [
        make(0),
        make(1, dropped_at="generate", reason="error"),
        make(2, dropped_at="filter", reason="refusal"),
        make(3, dropped_at="filter", reason="refusal"),
        make(4, dropped_at="dedup", reason="duplicate_exact"),
    ]
    rows = compute_funnel(samples, 5)
    assert [(r.stage, r.in_, r.out, r.dropped) for r in rows] == [
        ("generate", 5, 4, 1),
        ("score", 4, 4, 0),
        ("filter", 4, 2, 2),
        ("dedup", 2, 1, 1),
        ("decontaminate", 1, 1, 0),
    ]
    assert rows[2].reasons == {"refusal": 2}


def test_histogram_bins_cover_all_values() -> None:
    h = histogram([-10.0, -5.0, 0.0, 0.0, 3.0], [0.0, 3.0], "reward", bins=4)
    assert h is not None and sum(h.all) == 5 and sum(h.kept) == 2 and len(h.edges) == 5
    judge = histogram([1.0, 9.5], [9.5], "judge", bins=10)
    assert judge is not None and judge.edges[0] == 0.0 and judge.edges[-1] == 10.0
    assert histogram([], [], "reward") is None
    flat = histogram([2.0, 2.0], [2.0], "reward", bins=3)
    assert flat is not None and sum(flat.all) == 2
    assert "<svg" in histogram_svg(flat, "t")


async def test_report_contents(result: DistillResult) -> None:
    r = result.report
    assert r.n_requested == 100 and r.n_final == len(result.kept)
    assert r.retained == pytest.approx(len(result.kept) / 100)
    assert 0 < (r.diversity.mean_pairwise_distance or 0) <= 1
    assert 0 < r.diversity.distinct_1 <= r.diversity.distinct_3 <= 1
    assert {m.model for m in r.models} == {s.model for s in result.samples}
    assert r.top and r.top[0].score is not None and r.bottom[0].score is not None
    assert r.top[0].score >= r.bottom[0].score  # type: ignore[operator]
    assert r.usage["total_calls"] > 0 and r.usage["cost_usd"] > 0
    assert r.dominant_source in ("reward", "judge")
    assert sum(r.histograms[r.dominant_source or "reward"].all) > 0


async def test_report_survives_a_round_trip_through_disk(
    distiller: Distiller, result: DistillResult
) -> None:
    again = await distiller.report(result.run_id)
    assert again.model_dump() == result.report.model_dump()
    rebuilt = build_report(result.meta, result.samples, 100)
    assert rebuilt.diversity.mean_pairwise_distance is None  # no embeddings passed
    assert [f.out for f in rebuilt.funnel] == [f.out for f in result.report.funnel]


async def test_html_report_is_self_contained_and_labels_simulation(
    result: DistillResult, tmp_path: Path
) -> None:
    html = render_report_html(result.report, result.meta)
    assert html.startswith("<!doctype html>") and "</html>" in html
    assert "Simulated data" in html
    for section in ("Pipeline funnel", "Reward scores", "Generator models", "Diversity", "Usage"):
        assert section in html
    assert "<svg" in html and "prefers-color-scheme: dark" in html
    assert not re.search(r"(?:src|href)=[\"']https?://", html)  # no external assets
    for row in result.report.funnel:
        assert row.label in html
    html_file = next(p for p in result.files if p.name == "quality_report.html")
    assert html_file.read_text() == html


async def test_html_report_escapes_sample_text(distiller: Distiller) -> None:
    result = await distiller.run("instruction_diversity", n=30)
    evil = result.report.model_copy(
        update={
            "top": [
                result.report.top[0].model_copy(
                    update={"prompt": "<script>alert(1)</script>", "response": "<b>x</b>"}
                )
            ]
        }
    )
    html = render_report_html(evil, result.meta)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "<b>x</b>" not in html


async def test_dataset_card_has_provenance(result: DistillResult) -> None:
    card = render_card(result.meta, result.report, "data.jsonl")
    meta = result.meta
    assert card.startswith("---\nlicense: TODO")
    assert "SIMULATED DATA" in card and "  - simulated" in card
    for model in [*meta.models, meta.reward_model, meta.judge_model, meta.embed_model]:
        assert model in card
    assert "`instruction_diversity`" in card
    for heading in ("## Provenance", "## Pipeline and filters", "## Counts", "## Licence"):
        assert heading in card
    assert f"{result.report.n_final} (from 100 candidates" in card
    assert "| Generated | 100 |" in card
    for reason in result.report.drop_reasons:
        assert f"`{reason}`" in card
    readme = next(p for p in result.files if p.name == "README.md")
    assert "Dataset card" in readme.read_text()


async def test_card_mentions_recipe_specific_settings(distiller: Distiller) -> None:
    pref = await distiller.run("preference_pairs", n=40)
    card = render_card(pref.meta, pref.report, "x.jsonl")
    assert "Preference margin" in card and '"prompt": "...", "chosen"' in card
    ev = await distiller.run("eval_set", n=40)
    card = render_card(ev.meta, ev.report, "x.jsonl")
    assert "Weak model" in card and "Difficulty" in card


async def test_terminal_table(result: DistillResult) -> None:
    table = render_terminal(result.report)
    assert table.row_count == 5 and "SIMULATED" in str(table.title)
