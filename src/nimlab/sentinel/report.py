"""Scorecard reports: self-contained HTML (inline CSS, SVG radar per model) and a terminal table."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from rich.table import Table
from rich.text import Text

from .models import ATTACK_CATEGORIES, CATEGORY_LABELS, ProbeResult, Scorecard, SentinelRun
from .scoring import CAP_ASR, FALSE_REFUSAL_PENALTY, GRADE_BANDS, UNSAFE_PENALTY, WEIGHTS

__all__ = ["radar_axes", "radar_svg", "render_html", "render_terminal", "write_html"]

TEMPLATE_DIR = Path(__file__).parent / "templates"
_DASH = chr(0x2013)
MAX_BREACHES = 12
_AXIS_SHORT = {
    "prompt_injection": "Injection",
    "jailbreak": "Jailbreak",
    "system_prompt_extraction": "Extraction",
    "pii_leakage": "PII",
    "harmful_refusal": "Harmful",
    "hallucinated_citation": "Citations",
    "helpfulness": "Helpful",
    "safe_output": "Safe output",
}


def short_name(model: str) -> str:
    return model.split("/", 1)[-1]


def fmt_pct(value: float | None) -> str:
    return _DASH if value is None else f"{value * 100:.0f}%"


def bucket(rate: float | None) -> int:
    """0 (clean) .. 8 (always breached) heat bucket for an attack success rate."""
    return 0 if rate is None else max(0, min(8, math.ceil(rate * 8)))


def radar_axes(card: Scorecard) -> list[tuple[str, str, float]]:
    """``(key, label, value 0..1)`` axes with data; 1 = fully resists / fully helpful."""
    axes: list[tuple[str, str, float]] = []
    for cat in ATTACK_CATEGORIES:
        s = card.categories.get(cat)
        if s is not None and s.resistance is not None:
            axes.append((cat, _AXIS_SHORT[cat], s.resistance))
    if card.false_refusal_rate is not None:
        axes.append(("helpfulness", _AXIS_SHORT["helpfulness"], 1 - card.false_refusal_rate))
    if card.unsafe_output_rate is not None:
        axes.append(("safe_output", _AXIS_SHORT["safe_output"], 1 - card.unsafe_output_rate))
    return axes


def radar_svg(card: Scorecard, size: int = 320) -> str:
    """Inline SVG radar chart: the outer ring is a perfect score on every axis."""
    axes = radar_axes(card)
    if len(axes) < 3:
        return ""
    cx = cy = size / 2
    radius = size / 2 - 58
    n = len(axes)

    def pt(i: int, frac: float) -> tuple[float, float]:
        ang = -math.pi / 2 + 2 * math.pi * i / n
        return cx + radius * frac * math.cos(ang), cy + radius * frac * math.sin(ang)

    parts = [
        f'<svg class="radar" viewBox="0 0 {size} {size}" role="img" '
        f'aria-label="Radar chart of {short_name(card.model)}: '
        + "; ".join(f"{label} {v * 100:.0f}%" for _, label, v in axes)
        + '">'
    ]
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, ring) for i in range(n)))
        parts.append(f'<polygon class="ring{" outer" if ring == 1.0 else ""}" points="{pts}"/>')
    for i in range(n):
        x, y = pt(i, 1.0)
        parts.append(f'<line class="spoke" x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}"/>')
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, v) for i, (_, _, v) in enumerate(axes)))
    parts.append(f'<polygon class="shape" points="{poly}"/>')
    for i, (_, label, v) in enumerate(axes):
        x, y = pt(i, v)
        parts.append(
            f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="3.5">'
            f"<title>{label}: {v * 100:.0f}%</title></circle>"
        )
        lx, ly = pt(i, 1.17)
        anchor = "middle" if abs(lx - cx) < 8 else ("start" if lx > cx else "end")
        parts.append(
            f'<text class="axis" x="{lx:.1f}" y="{ly + 4:.1f}" '
            f'text-anchor="{anchor}">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _model_context(card: Scorecard, results: list[ProbeResult]) -> dict[str, Any]:
    mine = [r for r in results if r.model == card.model]
    cats = []
    for cat, s in card.categories.items():
        benign = cat == "over_refusal"
        cats.append(
            {
                "key": cat,
                "label": CATEGORY_LABELS.get(cat, cat),
                "benign": benign,
                "n": s.n,
                "bad": s.breached,
                "rate": fmt_pct(s.rate),
                "bucket": bucket(s.rate),
                "weight": WEIGHTS.get(cat),
            }
        )
    breaches = [r for r in mine if r.breached or r.false_refusal][:MAX_BREACHES]
    return {
        "name": card.model,
        "short": short_name(card.model),
        "grade": card.grade,
        "score": f"{card.score:.1f}",
        "integrity": f"{card.integrity:.0f}",
        "frr": fmt_pct(card.false_refusal_rate),
        "uor": fmt_pct(card.unsafe_output_rate),
        "capped": card.capped,
        "notes": card.notes,
        "cats": cats,
        "radar": Markup(radar_svg(card)),
        "breaches": [
            {
                "id": r.probe_id,
                "suite": r.suite,
                "title": r.title,
                "outcome": r.outcome,
                "text": " ".join(r.response.split())[:260],
            }
            for r in breaches
        ],
        "n_breaches": sum(1 for r in mine if r.breached or r.false_refusal),
    }


def build_context(run: SentinelRun, *, title: str | None = None) -> dict[str, Any]:
    cards = sorted(run.scorecards, key=lambda c: (-c.score, c.model))
    models = [_model_context(c, run.results) for c in cards]
    cat_keys = [
        c for c in (*ATTACK_CATEGORIES, "over_refusal") if any(c in k.categories for k in cards)
    ]
    matrix = []
    for card, m in zip(cards, models, strict=True):
        row = []
        for key in cat_keys:
            s = card.categories.get(key)
            row.append(
                {"rate": fmt_pct(s.rate if s else None), "bucket": bucket(s.rate if s else None)}
            )
        matrix.append({"name": m["short"], "full": m["name"], "grade": m["grade"], "cells": row})
    try:
        stamp = datetime.fromisoformat(run.created_at).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        stamp = run.created_at
    return {
        "title": title or "Sentinel scorecard",
        "run": run,
        "stamp": stamp,
        "models": models,
        "cat_labels": [CATEGORY_LABELS.get(k, k) for k in cat_keys],
        "matrix": matrix,
        "n_probes": len({(r.suite, r.probe_id) for r in run.results}),
        "n_calls": len(run.results),
        "best": models[0] if models else None,
        "weights": [(CATEGORY_LABELS[k], w) for k, w in WEIGHTS.items()],
        "bands": GRADE_BANDS,
        "fr_penalty": int(FALSE_REFUSAL_PENALTY),
        "unsafe_penalty": int(UNSAFE_PENALTY),
        "cap_pct": int(CAP_ASR * 100),
    }


def render_html(run: SentinelRun, *, title: str | None = None) -> str:
    """Self-contained HTML (inline CSS/SVG, no external assets, light/dark aware)."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("scorecard.html.j2").render(**build_context(run, title=title))


def write_html(run: SentinelRun, path: str | Path, *, title: str | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(run, title=title), encoding="utf-8")
    return out


_AXIS_TABLE = {
    "prompt_injection": "Inject",
    "jailbreak": "Jailbrk",
    "system_prompt_extraction": "Extract",
    "pii_leakage": "PII",
    "harmful_refusal": "Harmful",
    "hallucinated_citation": "Cites",
    "over_refusal": "FalseRef",
}
_GRADE_STYLE = {"A": "bold green", "B": "green", "C": "yellow", "D": "dark_orange", "F": "bold red"}


def render_terminal(run: SentinelRun) -> Table:
    """A rich table: one row per model, attack success rate per category, then the grade."""
    cards = sorted(run.scorecards, key=lambda c: (-c.score, c.model))
    cats = [
        c for c in (*ATTACK_CATEGORIES, "over_refusal") if any(c in k.categories for k in cards)
    ]
    caption = "attack success rates, lower is better"
    if run.simulated:
        caption += "  |  SIMULATED results (offline demo behaviour, not real models)"
    table = Table(title=f"Sentinel {run.run_id}", caption=caption, show_lines=False)
    table.add_column("Model", no_wrap=True)
    table.add_column("Grade", justify="center")
    table.add_column("Score", justify="right")
    for c in cats:
        table.add_column(_AXIS_TABLE[c], justify="right")
    table.add_column("Unsafe", justify="right")
    for card in cards:
        row: list[str | Text] = [
            short_name(card.model),
            Text(card.grade + ("*" if card.capped else ""), style=_GRADE_STYLE.get(card.grade, "")),
            f"{card.score:.1f}",
        ]
        for c in cats:
            s = card.categories.get(c)
            row.append(fmt_pct(s.rate if s else None))
        row.append(fmt_pct(card.unsafe_output_rate))
        table.add_row(*row)
    return table
