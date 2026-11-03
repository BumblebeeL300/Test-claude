"""Quality report: a self-contained HTML page (inline CSS and SVG) in the style of the Arena report,
and a rich terminal view."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.table import Table

from .models import DROP_REASONS, RunMeta
from .quality import Histogram, QualityReport

__all__ = ["render_report_html", "render_terminal", "write_report_html"]

TEMPLATE_DIR = Path(__file__).parent / "templates"
_DASH = chr(0x2013)


def short_name(model: str) -> str:
    return model.split("/", 1)[-1]


def histogram_svg(h: Histogram, title: str) -> str:
    """Two overlaid bar series (all scored vs kept) as inline SVG."""
    w, ht, ml, mr, mt, mb = 640, 220, 40, 12, 12, 34
    n = len(h.all)
    top = max(max(h.all, default=0), 1)
    bw = (w - ml - mr) / n
    parts = [
        f'<svg viewBox="0 0 {w} {ht}" role="img" aria-label="{html.escape(title)}">',
    ]
    for i in range(5):
        y = mt + (ht - mt - mb) * i / 4
        val = round(top * (1 - i / 4))
        parts.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="t-muted" x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end">{val}</text>'
        )
    rect = '<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"{extra}/>'
    base = ht - mb
    for i in range(n):
        x = ml + i * bw + 1
        ha = (ht - mb - mt) * h.all[i] / top
        hk = (ht - mb - mt) * h.kept[i] / top
        tip = f"{h.edges[i]:.2f} to {h.edges[i + 1]:.2f}: {h.all[i]} scored, {h.kept[i]} kept"
        extra = f' data-tip data-v="{h.all[i]}" data-l="{html.escape(tip)}" tabindex="0"'
        bar_w = max(bw - 2, 1)
        parts.append(rect.format(cls="bar-all", x=x, y=base - ha, w=bar_w, h=ha, extra=extra))
        parts.append(rect.format(cls="bar-kept", x=x, y=base - hk, w=bar_w, h=hk, extra=""))
    for i in range(0, n + 1, max(1, n // 6)):
        x = ml + i * bw
        label = f"{h.edges[i]:.1f}"
        y = ht - mb + 16
        parts.append(
            f'<text class="t-muted" x="{x:.1f}" y="{y}" text-anchor="middle">{label}</text>'
        )
    parts.append(f'<line class="axis" x1="{ml}" x2="{w - mr}" y1="{ht - mb}" y2="{ht - mb}"/>')
    parts.append("</svg>")
    return "".join(parts)


def build_context(report: QualityReport, meta: RunMeta) -> dict[str, Any]:
    first_in = report.funnel[0].in_ if report.funnel else 0
    funnel = []
    for row in report.funnel:
        funnel.append(
            {
                "label": row.label,
                "in": row.in_,
                "out": row.out,
                "dropped": row.dropped,
                "pct": round(100 * row.out / first_in, 1) if first_in else 0,
                "reasons": [
                    {"code": k, "n": v, "why": DROP_REASONS.get(k, "")}
                    for k, v in row.reasons.items()
                ],
            }
        )
    div = report.diversity
    kpis = [
        {
            "label": "Final examples",
            "value": str(report.n_final),
            "sub": f"of {report.n_requested} candidates",
        },
        {
            "label": "Retained",
            "value": f"{report.retained:.0%}",
            "sub": f"{report.n_requested - report.n_final} dropped",
        },
        {
            "label": "Diversity",
            "value": _DASH
            if div.mean_pairwise_distance is None
            else f"{div.mean_pairwise_distance:.3f}",
            "sub": "mean pairwise embedding distance",
        },
        {
            "label": "Distinct-2",
            "value": f"{div.distinct_2:.3f}",
            "sub": "unique prompt bigrams / all",
        },
        {
            "label": "Model calls",
            "value": str(report.usage["total_calls"]),
            "sub": f"{report.usage['total_tokens']:,} tokens",
        },
        {
            "label": "Est. cost",
            "value": f"${report.usage['cost_usd']:.4f}",
            "sub": "from cost tiers, not billing",
        },
    ]
    hists = [
        {
            "title": "Reward scores (Nemotron)"
            if k == "reward"
            else "Judge scores (0-10 fallback)",
            "svg": histogram_svg(h, f"{k} score histogram"),
            "total": sum(h.all),
            "kept": sum(h.kept),
        }
        for k, h in report.histograms.items()
    ]
    models = [
        {
            "model": m.model,
            "short": short_name(m.model),
            "generated": m.generated,
            "kept": m.kept,
            "rate": f"{m.kept_rate:.0%}",
            "pct": round(100 * m.kept_rate),
            "mean": _DASH if m.mean_score is None else f"{m.mean_score:.2f}",
        }
        for m in report.models
    ]
    return {
        "title": f"Distill quality report: {report.recipe}",
        "run_id": report.run_id,
        "created": meta.created_at,
        "kind": report.kind,
        "simulated": report.simulated,
        "kpis": kpis,
        "funnel": funnel,
        "histograms": hists,
        "models": models,
        "top": report.top,
        "bottom": report.bottom,
        "dominant": report.dominant_source,
        "usage_rows": [
            {"model": m, "short": short_name(m), **row} for m, row in report.usage["models"].items()
        ],
        "distinct": [div.distinct_1, div.distinct_2, div.distinct_3],
        "lengths": report.lengths,
        "filters": meta.recipe.filters.model_dump(),
        "dedup": meta.recipe.dedup.model_dump(),
        "decontam": meta.recipe.decontaminate.model_dump(),
        "fmt": meta.format,
    }


def render_report_html(report: QualityReport, meta: RunMeta) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("report.html.j2").render(**build_context(report, meta))


def write_report_html(report: QualityReport, meta: RunMeta, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report_html(report, meta), encoding="utf-8")
    return out


def render_terminal(report: QualityReport) -> Table:
    """A funnel table for the terminal."""
    title = f"Distill funnel • run {report.run_id}" + (" • SIMULATED" if report.simulated else "")
    table = Table(title=title, title_justify="left", header_style="bold")
    table.add_column("Stage")
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("Dropped", justify="right")
    table.add_column("Why", overflow="fold")
    for row in report.funnel:
        why = ", ".join(f"{k} {v}" for k, v in row.reasons.items()) or _DASH
        table.add_row(row.label, str(row.in_), str(row.out), str(row.dropped), why)
    return table
