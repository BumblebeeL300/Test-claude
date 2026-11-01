"""Leaderboard aggregation, the Pareto frontier, the HTML report and the terminal table."""

from __future__ import annotations

import io
import re
from html.parser import HTMLParser

import pytest
from rich.console import Console

from nimlab.arena.battle import position_bias
from nimlab.arena.leaderboard import Leaderboard, pareto_frontier
from nimlab.arena.report import (
    bucket,
    fmt_seconds,
    render_elo_table,
    render_html,
    render_terminal,
    write_html,
)
from nimlab.arena.results import BattleRecord, ItemResult, RunResult, SuiteInfo

BIG = "meta/llama-3.1-405b-instruct"  # cost tier 5
MID = "meta/llama-3.3-70b-instruct"  # cost tier 2
SMALL = "meta/llama-3.1-8b-instruct"  # cost tier 1

# model -> {suite: (score, latency)}
DATA = {
    BIG: {"math": (0.9, 4.0), "code": (0.8, 5.0)},
    MID: {"math": (0.7, 1.0), "code": (0.7, 1.2)},
    SMALL: {"math": (0.3, 0.4), "code": (0.5, 0.5)},
}


def build_run(n_items: int = 4, *, errors: bool = False) -> RunResult:
    run = RunResult.new(models=list(DATA))
    run.suites = {
        "math": SuiteInfo(name="math", category="math_reasoning", n_items=n_items),
        "code": SuiteInfo(name="code", category="coding", n_items=n_items),
    }
    for model, suites in DATA.items():
        for suite, (score, latency) in suites.items():
            for i in range(n_items):
                run.items.append(
                    ItemResult(
                        model=model,
                        suite=suite,
                        item_id=f"i{i}",
                        score=score,
                        passed=score > 0.5,
                        latency_s=latency + 0.01 * i,
                        ttft_s=latency / 2,
                        tokens_out=100,
                        tokens_per_s=50.0 + i,
                        output="x",
                    )
                )
    if errors:
        run.items.append(ItemResult(model=SMALL, suite="math", item_id="boom", error="Timeout"))
    return run


def board(**kw) -> Leaderboard:
    return Leaderboard.from_run(build_run(**kw.pop("run_kw", {})), n_boot=40, **kw)


# ------------------------------------------------------------------ aggregation


def test_ranking_scores_and_speed_stats() -> None:
    lb = board()
    assert [s.model for s in lb.standings] == [BIG, MID, SMALL]
    assert [s.rank for s in lb.standings] == [1, 2, 3]
    top = lb.standings[0]
    assert top.overall == pytest.approx((0.9 + 0.8) / 2)  # macro-average over suites
    assert top.suite_scores == {"code": pytest.approx(0.8), "math": pytest.approx(0.9)}
    assert top.latency_p50_s == pytest.approx(4.5, abs=0.5)
    assert top.ttft_p50_s is not None and top.tokens_per_s == pytest.approx(51.5)
    assert top.cost_tier == 5 and lb.get(SMALL).cost_tier == 1  # type: ignore[union-attr]
    assert top.pass_rate == 1.0 and top.n_errors == 0 and top.tokens_out == 800
    assert lb.suite_categories == {"code": "coding", "math": "math_reasoning"}


def test_item_weights_and_unknown_models() -> None:
    run = RunResult.new(models=["x/unknown"])
    run.items = [
        ItemResult(model="x/unknown", suite="s", item_id="a", score=1.0, weight=3.0),
        ItemResult(model="x/unknown", suite="s", item_id="b", score=0.0, weight=1.0),
    ]
    lb = Leaderboard.from_run(run, n_boot=0)
    assert lb.standings[0].overall == pytest.approx(0.75)
    assert lb.standings[0].cost_tier is None


def test_errored_items_are_excluded_from_scores_but_counted() -> None:
    lb = board(run_kw={"errors": True})
    small = lb.get(SMALL)
    assert small is not None and small.n_errors == 1
    assert small.suite_scores["math"] == pytest.approx(0.3)  # the errored item is not a zero
    assert small.n_scored == 8


def test_model_with_no_scored_items_ranks_last() -> None:
    run = build_run()
    run.models.append("dead/model")
    run.items.append(ItemResult(model="dead/model", suite="math", item_id="x", error="auth"))
    lb = Leaderboard.from_run(run, n_boot=0)
    assert lb.standings[-1].model == "dead/model" and lb.standings[-1].overall is None


def test_category_quality_and_latency_helpers() -> None:
    lb = board()
    assert lb.category_quality(BIG, "coding") == pytest.approx(0.8)
    assert lb.category_quality(BIG, "general_chat") == pytest.approx(0.85)
    assert lb.category_quality(BIG, "summarization") is None
    assert lb.category_quality("nobody", "coding") is None
    assert lb.category_latency(MID, "math_reasoning") == pytest.approx(1.015, abs=0.02)
    assert lb.suites_in("coding") == ["code"]
    assert lb.best_for_suite("math") == BIG and lb.best_for_suite("nope") is None
    assert set(lb.covered_categories()) == {"coding", "math_reasoning"}


# ------------------------------------------------------------------ Pareto frontier


def test_pareto_frontier_basic_and_ties() -> None:
    pts = [("a", 1.0, 0.5), ("b", 2.0, 0.9), ("c", 3.0, 0.7), ("d", 0.5, 0.4), ("e", 1.0, 0.5)]
    # c is dominated by b (faster and better); a and e are identical so neither dominates the other
    assert pareto_frontier(pts) == ["a", "b", "d", "e"]
    assert pareto_frontier([("x", None, 0.5), ("y", 1.0, None)]) == []
    assert pareto_frontier([("only", 1.0, 0.1)]) == ["only"]


def test_leaderboard_flags_pareto_models() -> None:
    lb = board()
    flags = {s.model: s.pareto for s in lb.standings}
    # BIG is best, SMALL is fastest, MID (1.0s, 0.75) is not dominated by either
    assert flags == {BIG: True, MID: True, SMALL: True}
    run = build_run()
    for r in run.items:
        if r.model == MID:
            r.score, r.latency_s = 0.2, 9.0  # slow and bad: dominated
    lb2 = Leaderboard.from_run(run, n_boot=0)
    assert {s.model: s.pareto for s in lb2.standings}[MID] is False


# ------------------------------------------------------------------ Elo sources


def test_elo_from_scores_by_default_and_from_battles_when_present() -> None:
    lb = board()
    assert lb.elo_source == "scores" and lb.elo_games > 0 and lb.n_boot == 40
    elo = {s.model: s.elo for s in lb.standings}
    assert elo[BIG] > elo[MID] > elo[SMALL]
    for s in lb.standings:
        assert s.elo_lo <= s.elo <= s.elo_hi  # type: ignore[operator]

    run = build_run()
    run.battles = [
        BattleRecord(suite="s", item_id=str(i), model_a=SMALL, model_b=BIG, score_a=1.0)
        for i in range(8)
    ]
    lb2 = Leaderboard.from_run(run, n_boot=30)
    assert lb2.elo_source == "battles" and lb2.elo_games == 8
    elo2 = {s.model: s.elo for s in lb2.standings}
    assert elo2[SMALL] > elo2[BIG]  # the battles say the opposite of the scores
    assert elo2[MID] is None  # MID never battled
    assert lb2.position_bias == position_bias([]) is None
    assert Leaderboard.from_run(run, n_boot=0, elo_source="scores").elo_source == "scores"


def test_single_model_run_has_no_elo() -> None:
    run = RunResult.new(models=["solo"])
    run.items = [ItemResult(model="solo", suite="s", item_id="a", score=1.0, latency_s=1.0)]
    lb = Leaderboard.from_run(run, n_boot=10)
    assert lb.elo_source == "none" and lb.standings[0].elo is None


# ------------------------------------------------------------------ HTML report


class Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.ids: list[str] = []
        self.attrs: list[tuple[str, str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        for k, v in attrs:
            self.attrs.append((tag, k, v))
            if k == "id" and v:
                self.ids.append(v)


def parse(page: str) -> Collector:
    parser = Collector()
    parser.feed(page)
    return parser


def test_html_report_has_all_sections() -> None:
    page = render_html(board())
    parsed = parse(page)
    assert page.startswith("<!doctype html>") and "</html>" in page
    assert {"ranking", "heatmap", "elo", "frontier"} <= set(parsed.ids)
    assert parsed.tags.count("svg") >= 4  # elo chart + scatter + two legend swatches
    assert "Overall ranking" in page and "Scores by suite" in page
    assert "Elo ratings with confidence intervals" in page and "Quality vs latency" in page
    for model in ("llama-3.1-405b-instruct", "llama-3.3-70b-instruct", "llama-3.1-8b-instruct"):
        assert model in page
    # per-suite heatmap: one cell per model x suite with a bucket class
    assert len(re.findall(r'<td class="c h\d', page)) == 6
    assert "90.0" in page and "30.0" in page
    assert "Pareto" in page  # badge for frontier models
    assert "$$$$$" in page  # cost tier of the 405B
    assert "[" in page and "Elo (95% CI)" in page


def test_html_report_is_self_contained_and_theme_aware() -> None:
    page = render_html(board())
    assert not re.findall(r'(?:src|href)\s*=\s*["\']https?:', page)
    assert "url(http" not in page and "@import" not in page and "<link" not in page
    assert "https://" not in page.replace("https://www.w3.org", "")
    assert "prefers-color-scheme: dark" in page
    assert 'name="viewport"' in page and "max-width: 560px" in page
    assert "<script src" not in page
    assert '[data-theme="dark"]' in page and "color-scheme" in page
    assert page.count("<style>") == 1


def test_html_report_escapes_untrusted_model_names() -> None:
    run = build_run()
    evil = 'evil/<script>alert("x")</script>&"'
    for r in run.items:
        if r.model == SMALL:
            r.model = evil
    run.models = [evil if m == SMALL else m for m in run.models]
    page = render_html(Leaderboard.from_run(run, n_boot=20))
    assert '<script>alert("x")</script>' not in page
    assert "&lt;script&gt;" in page or "&lt;script" in page
    assert page.count("<script") == 1  # only our tooltip script


def test_html_report_marks_simulated_runs_and_handles_degenerate_input(tmp_path) -> None:
    run = build_run()
    run.mock = True
    page = render_html(Leaderboard.from_run(run, n_boot=10))
    assert "Simulated data" in page
    assert "Simulated data" not in render_html(board())

    solo = RunResult.new(models=["solo/model"])
    solo.items = [ItemResult(model="solo/model", suite="s", item_id="a", score=1.0, latency_s=0.2)]
    page = render_html(Leaderboard.from_run(solo, n_boot=10))
    assert "No pairwise data" in page and "solo/model" in page

    empty = RunResult.new(models=[])
    page = render_html(Leaderboard.from_run(empty, n_boot=0))
    assert "No pairwise data" in page and "No latency data" in page

    out = write_html(board(), tmp_path / "nested" / "lb.html", title="My <board>")
    text = out.read_text(encoding="utf-8")
    assert "<title>My &lt;board&gt;</title>" in text


def test_elo_chart_draws_error_bars_for_every_rated_model() -> None:
    page = render_html(board())
    chart = page.split('id="elo"')[1].split("</section>")[0]
    assert chart.count('class="dot"') == 3
    assert chart.count('class="ci"') == 3 * 3  # bar plus two caps per model
    assert "95% confidence" in chart


def test_scatter_highlights_frontier_models() -> None:
    page = render_html(board())
    chart = page.split('id="frontier"')[1].split("</section>")[0]
    assert 'class="front"' in chart and chart.count('class="dot"') >= 3
    assert "log scale" in chart  # latencies span 0.5s..5s (>8x)


def test_formatters() -> None:
    assert (
        fmt_seconds(None) == "–" and fmt_seconds(0.0123) == "12 ms" and fmt_seconds(2.5) == "2.50 s"
    )
    assert bucket(0.0) == 0 and bucket(1.0) == 8 and bucket(0.5) == 4


# ------------------------------------------------------------------ terminal


def render_to_text(table) -> str:
    buf = io.StringIO()
    Console(file=buf, width=200, color_system=None, force_terminal=False).print(table)
    return buf.getvalue()


def test_terminal_table_lists_models_scores_and_markers() -> None:
    text = render_to_text(render_terminal(board()))
    for model in (BIG, MID, SMALL):
        assert model in text
    assert "math" in text and "code" in text
    assert "90.0" in text and "$$$$$" in text and "*" in text
    assert "Arena leaderboard" in text and "SIMULATED" not in text


def test_terminal_elo_table_shows_ci_and_battle_record() -> None:
    run = build_run()
    run.battles = [
        BattleRecord(suite="s", item_id="1", model_a=BIG, model_b=MID, score_a=1.0),
        BattleRecord(suite="s", item_id="2", model_a=BIG, model_b=MID, score_a=0.5),
    ]
    lb = Leaderboard.from_run(run, n_boot=20)
    text = render_to_text(render_elo_table(lb, run))
    assert "judge battles" in text and "W-T-L" in text and "1-1-0" in text and BIG in text
