"""Leaderboard reports: a self-contained HTML page (inline CSS + SVG) and a rich terminal table."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.table import Table
from rich.text import Text

from .leaderboard import Leaderboard, ModelStanding

__all__ = ["render_elo_table", "render_html", "render_terminal", "write_html"]

TEMPLATE_DIR = Path(__file__).parent / "templates"
_N_BUCKETS = 9
_DASH = chr(0x2013)  # en dash for "no value"


# ------------------------------------------------------------------------------ formatting


def short_name(model: str) -> str:
    return model.split("/", 1)[-1]


def vendor_of(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else ""


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return _DASH
    if value < 1:
        return f"{value * 1000:.0f} ms"
    return f"{value:.2f} s"


def fmt_pct(value: float | None) -> str:
    return _DASH if value is None else f"{value * 100:.1f}"


def bucket(value: float) -> int:
    return max(0, min(_N_BUCKETS - 1, int(value * _N_BUCKETS)))


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def nice_step(span: float, target_ticks: int = 6) -> float:
    raw = span / max(1, target_ticks)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * mag:
            return mult * mag
    return 10 * mag


def _ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    step = nice_step(hi - lo, target)
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


# ------------------------------------------------------------------------------ SVG charts


def _elo_bounds(s: ModelStanding) -> tuple[float, float, float]:
    """``(elo, lo, hi)`` of a rated model; the interval collapses to the point if unknown."""
    elo = s.elo if s.elo is not None else 0.0
    return elo, s.elo_lo if s.elo_lo is not None else elo, s.elo_hi if s.elo_hi is not None else elo


def elo_svg(lb: Leaderboard) -> str:
    rated = sorted((s for s in lb.standings if s.elo is not None), key=lambda s: -(s.elo or 0.0))
    if not rated:
        return '<p class="muted">No pairwise data: run at least two models to compute Elo.</p>'
    lo = min(_elo_bounds(s)[1] for s in rated)
    hi = max(_elo_bounds(s)[2] for s in rated)
    pad = max(15.0, (hi - lo) * 0.06)
    lo, hi = lo - pad, hi + pad
    label_w = min(250, max(110, int(max(len(short_name(s.model)) for s in rated) * 6.6) + 12))
    right = 150
    width, row_h, top, bottom = 760, 34, 14, 34
    height = top + row_h * len(rated) + bottom
    plot_l, plot_r = label_w, width - right

    def x(v: float) -> float:
        return plot_l + (v - lo) / (hi - lo) * (plot_r - plot_l)

    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Elo ratings with 95% confidence intervals for {len(rated)} models">'
    ]
    for t in _ticks(lo, hi):
        out.append(
            f'<line class="grid" x1="{x(t):.1f}" y1="{top}" x2="{x(t):.1f}" y2="{height - bottom}"/>'
        )
        out.append(
            f'<text class="t-muted" x="{x(t):.1f}" y="{height - bottom + 18}" text-anchor="middle">{t:g}</text>'
        )
    out.append(
        f'<line class="axis" x1="{plot_l}" y1="{height - bottom}" x2="{plot_r}" y2="{height - bottom}"/>'
    )
    for i, s in enumerate(rated):
        cy = top + row_h * i + row_h / 2
        elo = s.elo or 0.0
        a = s.elo_lo if s.elo_lo is not None else elo
        b = s.elo_hi if s.elo_hi is not None else elo
        tip = f"{s.model}: Elo {elo:.0f} (95% CI {a:.0f} to {b:.0f}), {s.elo_games} games"
        out.append(
            f'<text class="t-strong" x="{plot_l - 10}" y="{cy + 4:.1f}" text-anchor="end">'
            f"{_esc(_clip(short_name(s.model), 36))}</text>"
        )
        out.append(
            f'<line class="ci" x1="{x(a):.1f}" y1="{cy:.1f}" x2="{x(b):.1f}" y2="{cy:.1f}"/>'
        )
        for edge in (a, b):
            out.append(
                f'<line class="ci" x1="{x(edge):.1f}" y1="{cy - 5:.1f}" x2="{x(edge):.1f}" y2="{cy + 5:.1f}"/>'
            )
        out.append(
            f'<circle class="hit" cx="{x(elo):.1f}" cy="{cy:.1f}" r="14" tabindex="0" data-tip '
            f'data-v="{elo:.0f} Elo" data-l="{_esc(tip)}"/>'
            f'<circle class="dot" cx="{x(elo):.1f}" cy="{cy:.1f}" r="5"/>'
        )
        out.append(
            f'<text x="{x(b) + 12:.1f}" y="{cy + 4:.1f}"><tspan class="t-strong">{elo:.0f}</tspan>'
            f'<tspan class="t-muted"> [{a:.0f}, {b:.0f}]</tspan></text>'
        )
    out.append(
        f'<text class="t-muted" x="{(plot_l + plot_r) / 2:.0f}" y="{height - 2}" text-anchor="middle">Elo</text>'
    )
    out.append("</svg>")
    return "".join(out)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def scatter_svg(lb: Leaderboard) -> tuple[str, bool]:
    pts = [s for s in lb.standings if s.overall is not None and s.latency_p50_s]
    if not pts:
        return '<p class="muted">No latency data to plot.</p>', False
    lats = [s.latency_p50_s or 0.0 for s in pts]
    log_x = len(pts) > 1 and max(lats) / max(min(lats), 1e-9) > 8
    width, height = 760, 380
    ml, mr, mt, mb = 56, 24, 16, 44
    ys = [s.overall or 0.0 for s in pts]
    y_lo = max(0.0, math.floor((min(ys) - 0.05) * 20) / 20)
    y_hi = min(1.0, math.ceil((max(ys) + 0.05) * 20) / 20)
    if y_hi - y_lo < 0.1:
        y_lo, y_hi = max(0.0, y_lo - 0.05), min(1.0, y_hi + 0.05)

    def fx(v: float) -> float:
        return math.log10(max(v, 1e-9)) if log_x else v

    x_lo, x_hi = fx(min(lats)), fx(max(lats))
    if x_hi - x_lo < 1e-9:
        x_lo, x_hi = x_lo - 0.5, x_hi + 0.5
    span = x_hi - x_lo
    x_lo, x_hi = x_lo - span * 0.08, x_hi + span * 0.12

    def px(v: float) -> float:
        return ml + (fx(v) - x_lo) / (x_hi - x_lo) * (width - ml - mr)

    def py(v: float) -> float:
        return mt + (1 - (v - y_lo) / (y_hi - y_lo)) * (height - mt - mb)

    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Quality versus median latency for {len(pts)} models; '
        f'{sum(1 for s in pts if s.pareto)} on the Pareto frontier">'
    ]
    y_step = nice_step(y_hi - y_lo, 5)
    yv = math.ceil(y_lo / y_step) * y_step
    while yv <= y_hi + 1e-9:
        out.append(
            f'<line class="grid" x1="{ml}" y1="{py(yv):.1f}" x2="{width - mr}" y2="{py(yv):.1f}"/>'
        )
        out.append(
            f'<text class="t-muted" x="{ml - 8}" y="{py(yv) + 4:.1f}" text-anchor="end">{yv * 100:.0f}</text>'
        )
        yv += y_step
    if log_x:
        decade = math.floor(x_lo)
        xt = []
        while decade <= x_hi:
            for m in (1, 2, 5):
                v = m * 10**decade
                if x_lo <= math.log10(v) <= x_hi:
                    xt.append(v)
            decade += 1
    else:
        xt = _ticks(x_lo, x_hi, 6)
    for v in xt:
        out.append(
            f'<line class="grid" x1="{px(v):.1f}" y1="{mt}" x2="{px(v):.1f}" y2="{height - mb}"/>'
        )
        out.append(
            f'<text class="t-muted" x="{px(v):.1f}" y="{height - mb + 18}" text-anchor="middle">{fmt_seconds(v)}</text>'
        )
    out.append(
        f'<line class="axis" x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}"/>'
    )
    out.append(
        f'<text class="t-muted" x="{(ml + width - mr) / 2:.0f}" y="{height - 4}" text-anchor="middle">'
        f"Median latency per item{' (log scale)' if log_x else ''}</text>"
    )
    out.append(
        f'<text class="t-muted" transform="translate(14 {(mt + height - mb) / 2:.0f}) rotate(-90)" '
        f'text-anchor="middle">Overall score</text>'
    )

    front = sorted((s for s in pts if s.pareto), key=lambda s: s.latency_p50_s or 0.0)
    if len(front) > 1:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{px(s.latency_p50_s or 0):.1f},{py(s.overall or 0):.1f}"
            for i, s in enumerate(front)
        )
        out.append(f'<path class="front" d="{path}"/>')

    placed: list[Box] = [
        (
            px(p.latency_p50_s or 0) - 8,
            py(p.overall or 0) - 8,
            px(p.latency_p50_s or 0) + 8,
            py(p.overall or 0) + 8,
        )
        for p in pts
    ]
    for s in sorted(
        pts, key=lambda s: not s.pareto
    ):  # frontier labels get first pick of the space  # frontier points drawn last (on top)
        cx, cy = px(s.latency_p50_s or 0), py(s.overall or 0)
        name = _clip(short_name(s.model), 28)
        w = len(name) * 6.6
        label_xy = _place_label(cx, cy, w, placed, (ml, mt, width - mr, height - mb))
        tip = (
            f"{s.model}: score {fmt_pct(s.overall)}, latency {fmt_seconds(s.latency_p50_s)}"
            f"{', Pareto-optimal' if s.pareto else ''}"
        )
        cls = "dot" if s.pareto else "dot off"
        out.append(
            f'<text class="halo {"t-strong" if s.pareto else "t-muted"}" x="{label_xy[0]:.1f}" y="{label_xy[1]:.1f}" '
            f'text-anchor="{label_xy[2]}">{_esc(name)}</text>'
        )
        out.append(
            f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="14" tabindex="0" data-tip '
            f'data-v="{fmt_pct(s.overall)} score" data-l="{_esc(tip)}"/>'
            f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{6 if s.pareto else 5}"/>'
        )
    out.append("</svg>")
    return "".join(out), log_x


Box = tuple[float, float, float, float]


def _overlap(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _place_label(
    cx: float, cy: float, w: float, placed: list[Box], bounds: Box
) -> tuple[float, float, str]:
    """Pick a spot for a point label that avoids other labels and dots (falls back to least overlap)."""
    left, top, right, bottom = bounds
    h = 13.0
    gap = 10.0
    # (dx, dy of the text baseline anchor, text-anchor); preference order
    spots = [
        (gap, 4.0, "start"),
        (gap, -gap, "start"),
        (gap, gap + h, "start"),
        (-gap, 4.0, "end"),
        (-gap, -gap, "end"),
        (-gap, gap + h, "end"),
        (0.0, -gap - 4, "middle"),
        (0.0, gap + h + 2, "middle"),
    ]
    best: tuple[float, float, float, str, Box] | None = None
    for dx, dy, anchor in spots:
        x, y = cx + dx, cy + dy
        x0 = x if anchor == "start" else x - w if anchor == "end" else x - w / 2
        box: Box = (x0, y - h + 2, x0 + w, y + 3)
        cost = sum(_overlap(box, p) for p in placed)
        if box[0] < left or box[2] > right + 8 or box[1] < top - 4 or box[3] > bottom:
            cost += 1000.0  # outside the plot area
        if best is None or cost < best[0]:
            best = (cost, x, y, anchor, box)
        if cost == 0:
            break
    assert best is not None
    placed.append(best[4])
    return best[1], best[2], best[3]


# ------------------------------------------------------------------------------ HTML


def _cost_html(tier: int | None) -> tuple[str, str]:
    if tier is None:
        return '<span class="muted">?</span>', "cost tier unknown"
    return (
        f"<b>{'$' * tier}</b><span>{'$' * (5 - tier)}</span>",
        f"cost tier {tier} of 5 (1 = cheapest)",
    )


def build_context(lb: Leaderboard, *, title: str | None = None) -> dict[str, Any]:
    standings = lb.standings
    top = next((s for s in standings if s.overall is not None), None)
    timed = [s for s in standings if s.latency_p50_s]
    fastest = min(timed, key=lambda s: s.latency_p50_s or 0.0) if timed else None
    cheap = min(
        (s for s in standings if s.cost_tier and s.overall is not None),
        key=lambda s: (s.cost_tier or 9, -(s.overall or 0)),
        default=None,
    )
    n_items = sum(s.n_scored for s in standings)
    kpis = [
        {
            "label": "Top model",
            "value": short_name(top.model) if top else "n/a",
            "sub": f"overall {fmt_pct(top.overall)}" if top else "",
        },
        {
            "label": "Fastest (median latency)",
            "value": short_name(fastest.model) if fastest else "n/a",
            "sub": fmt_seconds(fastest.latency_p50_s) if fastest else "",
        },
        {
            "label": "Best score at lowest cost tier",
            "value": short_name(cheap.model) if cheap else "n/a",
            "sub": f"tier {cheap.cost_tier}, overall {fmt_pct(cheap.overall)}" if cheap else "",
        },
        {
            "label": "Pareto-optimal models",
            "value": str(sum(1 for s in standings if s.pareto)),
            "sub": f"of {len(standings)} · {n_items} scored answers",
        },
    ]

    rows = []
    for s in standings:
        cost, cost_title = _cost_html(s.cost_tier)
        if s.elo is not None and s.elo_lo is not None and s.elo_hi is not None:
            elo_text = f"{s.elo:.0f} [{s.elo_lo:.0f}, {s.elo_hi:.0f}]"
        elif s.elo is not None:
            elo_text = f"{s.elo:.0f}"
        else:
            elo_text = _DASH
        rows.append(
            {
                "rank": s.rank,
                "model": s.model,
                "short": short_name(s.model),
                "vendor": vendor_of(s.model),
                "pareto": s.pareto,
                "overall": s.overall,
                "overall_pct": round((s.overall or 0) * 100, 1),
                "overall_text": fmt_pct(s.overall),
                "elo_text": elo_text,
                "latency": fmt_seconds(s.latency_p50_s),
                "ttft": fmt_seconds(s.ttft_p50_s),
                "tps": f"{s.tokens_per_s:.0f}" if s.tokens_per_s else _DASH,
                "cost": cost,
                "cost_title": cost_title,
                "errors": s.n_errors,
            }
        )

    suites = [
        {"name": n, "label": n.replace("_", " "), "category": lb.suite_categories.get(n, "")}
        for n in lb.suites
    ]
    best = {n: lb.best_for_suite(n) for n in lb.suites}
    heat = []
    for s in standings:
        cells = []
        for n in lb.suites:
            v = s.suite_scores.get(n)
            cells.append(
                {
                    "value": v,
                    "suite": n.replace("_", " "),
                    "text": fmt_pct(v),
                    "bucket": bucket(v) if v is not None else 0,
                    "best": v is not None and best.get(n) == s.model and len(standings) > 1,
                }
            )
        heat.append({"model": s.model, "short": short_name(s.model), "cells": cells})

    scatter, log_x = scatter_svg(lb)
    if lb.elo_source == "battles":
        elo_note = (
            "Bradley-Terry ratings fitted to judge-decided pairwise battles (each answer pair "
            "judged in both positions to cancel position bias). Dots are the maximum-likelihood "
            "rating, bars the 95% bootstrap interval."
        )
        if lb.position_bias is not None:
            elo_note += (
                f" Judge picked slot A in {lb.position_bias * 100:.0f}% of decisive verdicts."
            )
    elif lb.elo_source == "scores":
        elo_note = (
            "Bradley-Terry ratings fitted to per-item head-to-head comparisons of the graded "
            "scores (higher score wins, near-equal scores tie). Dots are the maximum-likelihood "
            "rating, bars the 95% bootstrap interval. Run `nimlab arena battle` for judge-based Elo."
        )
    else:
        elo_note = "Not enough data to rate models."
    return {
        "title": title or "NIM Arena leaderboard",
        "run_id": lb.run_id,
        "created": lb.created_at.replace("T", " ") if lb.created_at else "",
        "n_models": len(standings),
        "n_suites": len(lb.suites),
        "judge_model": lb.judge_model,
        "mock": lb.mock,
        "kpis": kpis,
        "rows": rows,
        "suites": suites,
        "heat": heat,
        "elo_note": elo_note,
        "elo_svg": elo_svg(lb),
        "scatter_svg": scatter,
        "log_x": log_x,
        "elo_games": lb.elo_games,
        "n_boot": lb.n_boot,
    }


def render_html(lb: Leaderboard, *, title: str | None = None) -> str:
    """Self-contained HTML (inline CSS/SVG/JS, no external assets, light/dark aware)."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("leaderboard.html.j2").render(**build_context(lb, title=title))


def write_html(lb: Leaderboard, path: str | Path, *, title: str | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(lb, title=title), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------ terminal


def _score_style(v: float | None) -> str:
    if v is None:
        return "dim"
    return "green" if v >= 0.8 else "yellow" if v >= 0.5 else "red"


_SHORT_SUITE = {
    "math_reasoning": "math",
    "coding": "code",
    "factual_qa": "facts",
    "instruction_following": "instr",
    "json_compliance": "json",
    "summarization": "summ",
}


def render_terminal(lb: Leaderboard) -> Table:
    """A rich table: rank, model, overall, per-suite scores, Elo, speed and cost."""
    title = f"Arena leaderboard \u2022 run {lb.run_id}" + (
        " \u2022 SIMULATED (mock)" if lb.mock else ""
    )
    table = Table(
        title=title,
        title_justify="left",
        header_style="bold",
        caption="* = on the quality/latency Pareto frontier \u2022 scores 0-100 \u2022 latency in seconds",
        caption_justify="left",
    )
    table.add_column("#", justify="right")
    table.add_column("Model", overflow="fold", min_width=22)
    table.add_column("Overall", justify="right", no_wrap=True, min_width=7)
    for s in lb.suites:
        table.add_column(_SHORT_SUITE.get(s, s[:8]), justify="right", no_wrap=True)
    table.add_column("Elo\u00b1CI", justify="right", no_wrap=True)
    table.add_column("p50", justify="right", no_wrap=True)
    table.add_column("TTFT", justify="right", no_wrap=True)
    table.add_column("tok/s", justify="right", no_wrap=True)
    table.add_column("$", justify="center", no_wrap=True)
    for st in lb.standings:
        table.add_row(*_terminal_row(st, lb))
    return table


def _terminal_row(st: ModelStanding, lb: Leaderboard) -> list[Any]:
    name = Text(st.model)
    if st.pareto:
        name.append(" *", style="bold cyan")
    cells: list[Any] = [str(st.rank), name]
    cells.append(Text(fmt_pct(st.overall), style=f"bold {_score_style(st.overall)}"))
    for s in lb.suites:
        v = st.suite_scores.get(s)
        cells.append(Text(fmt_pct(v), style=_score_style(v)))
    if st.elo is not None and st.elo_lo is not None and st.elo_hi is not None:
        cells.append(f"{st.elo:.0f} ±{(st.elo_hi - st.elo_lo) / 2:.0f}")
    else:
        cells.append(_DASH)
    cells += [
        fmt_seconds(st.latency_p50_s),
        fmt_seconds(st.ttft_p50_s),
        f"{st.tokens_per_s:.0f}" if st.tokens_per_s else _DASH,
        ("$" * st.cost_tier) if st.cost_tier else "?",
    ]
    return cells


def render_elo_table(lb: Leaderboard, run: Any = None) -> Table:
    """Rich table of Elo ratings with confidence intervals (and battle record if ``run`` given)."""
    source = {"battles": "judge battles", "scores": "per-item scores", "none": "no data"}[
        lb.elo_source
    ]
    table = Table(
        title=f"Elo (Bradley-Terry, 95% bootstrap CI) • {source} • {lb.elo_games} games",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("#", justify="right")
    table.add_column("Model", overflow="fold")
    table.add_column("Elo", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Games", justify="right")
    record: dict[str, list[int]] = {}
    if run is not None and getattr(run, "battles", None):
        for b in run.battles:
            for m, s in ((b.model_a, b.score_a), (b.model_b, 1.0 - b.score_a)):
                w = record.setdefault(m, [0, 0, 0])
                w[0 if s > 0.5 else 2 if s < 0.5 else 1] += 1
        table.add_column("W-T-L", justify="right")
    rated = sorted((s for s in lb.standings if s.elo is not None), key=lambda s: -(s.elo or 0))
    for i, s in enumerate(rated, 1):
        row = [
            str(i),
            s.model,
            f"{s.elo:.0f}",
            f"{s.elo_lo:.0f} to {s.elo_hi:.0f}" if s.elo_lo is not None else _DASH,
            str(s.elo_games),
        ]
        if record:
            w = record.get(s.model, [0, 0, 0])
            row.append(f"{w[0]}-{w[1]}-{w[2]}")
        table.add_row(*row)
    if lb.position_bias is not None:
        table.caption = (
            f"Judge picked slot A in {lb.position_bias * 100:.0f}% of decisive verdicts "
            "(50% = no position bias; each pair is judged in both orders)."
        )
    return table
