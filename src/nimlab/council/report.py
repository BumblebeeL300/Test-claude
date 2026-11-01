"""Self-contained HTML trace report (jinja2; inline CSS/JS; no external assets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from .records import ConsensusReport, StepType, Trace, TraceStep

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "trace.html.j2"

TYPE_LABELS: dict[StepType, str] = {
    StepType.PROPOSE: "propose",
    StepType.AGGREGATE: "aggregate",
    StepType.DRAFT: "draft",
    StepType.CRITIQUE: "critique",
    StepType.REVISE: "revise",
    StepType.JUDGE: "judge",
    StepType.SCORE: "score",
    StepType.CONSENSUS: "consensus",
}


def short_model(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def model_hues(models: list[str]) -> dict[str, int]:
    """A distinct, stable hue per model (golden-angle spacing)."""
    return {m: int((205 + i * 137.508) % 360) for i, m in enumerate(models)}


def fmt_seconds(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.0f} ms"
    if value < 60:
        return f"{value:.2f} s"
    return f"{int(value // 60)}m {value % 60:.0f}s"


def fmt_tokens(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def _heat_alpha(value: float) -> float:
    return round(0.08 + 0.67 * min(1.0, max(0.0, value)), 3)


def _step_view(step: TraceStep, hues: dict[str, int]) -> dict[str, Any]:
    return {
        "index": step.index,
        "type": step.type.value,
        "type_label": TYPE_LABELS.get(step.type, step.type.value),
        "model": step.model,
        "short": short_model(step.model),
        "hue": hues.get(step.model, 210),
        "label": step.label,
        "phase": step.phase,
        "status": step.status,
        "duration": fmt_seconds(step.duration_s),
        "tokens": fmt_tokens(step.total_tokens),
        "tokens_detail": (
            f"{step.prompt_tokens:,} in / {step.completion_tokens:,} out"
            if step.prompt_tokens is not None and step.completion_tokens is not None
            else "unknown"
        ),
        "temperature": step.temperature,
        "prompt_summary": step.prompt_summary,
        "prompt": step.prompt,
        "output": step.output,
        "reasoning": step.reasoning,
        "error": step.error,
        "meta": {k: v for k, v in step.meta.items() if v is not None},
    }


def _consensus_view(rep: ConsensusReport, hues: dict[str, int]) -> dict[str, Any]:
    outliers = set(rep.outliers)
    rows = []
    for i, row in enumerate(rep.matrix):
        rows.append(
            {
                "label": rep.labels[i],
                "short": short_model(rep.models[i]),
                "hue": hues.get(rep.models[i], 210),
                "outlier": i in outliers,
                "centrality": f"{rep.centrality[i]:.2f}",
                "cells": [
                    {
                        "value": f"{v:.2f}",
                        "alpha": _heat_alpha(v),
                        "diag": i == j,
                    }
                    for j, v in enumerate(row)
                ],
            }
        )
    return {
        "phase": rep.phase,
        "method": rep.method,
        "embed_model": rep.embed_model,
        "agreement": f"{rep.agreement:.3f}",
        "agreement_pct": round(rep.agreement * 100),
        "min_pairwise": f"{rep.min_pairwise:.2f}",
        "size": rep.size,
        "labels": rep.labels,
        "rows": rows,
        "outliers": rep.outlier_labels,
    }


def _timeline_view(trace: Trace, hues: dict[str, int]) -> dict[str, Any]:
    total = max([trace.wall_time_s, *(s.ended_s for s in trace.steps), 1e-6])
    lanes = []
    for model in trace.all_models:
        bars = []
        for step in trace.steps_for(model):
            left = step.started_s / total * 100
            width = max(step.duration_s / total * 100, 0.5)
            bars.append(
                {
                    "index": step.index,
                    "left": round(min(left, 100 - width), 3),
                    "width": round(width, 3),
                    "type": step.type.value,
                    "status": step.status,
                    "title": (
                        f"#{step.index} {step.type.value} · {step.label or step.phase} · "
                        f"{fmt_seconds(step.duration_s)}"
                    ),
                }
            )
        if bars:
            lanes.append(
                {"model": model, "short": short_model(model), "hue": hues[model], "bars": bars}
            )
    ticks = [{"pos": round(i * 20, 1), "label": fmt_seconds(total * i / 5)} for i in range(6)]
    return {"lanes": lanes, "ticks": ticks, "total": fmt_seconds(total)}


def build_view(trace: Trace) -> dict[str, Any]:
    """Everything the template needs, pre-formatted (kept separate so it is easy to test)."""
    models = trace.all_models
    hues = model_hues(models)
    steps = [_step_view(s, hues) for s in trace.steps]
    phases: list[dict[str, Any]] = []
    for step in steps:
        if not phases or phases[-1]["name"] != step["phase"]:
            phases.append({"name": step["phase"] or "Run", "steps": []})
        phases[-1]["steps"].append(step)

    usage = trace.usage or trace.usage_from_steps()
    totals = {
        "calls": sum(u.calls for u in usage),
        "errors": sum(u.errors for u in usage),
        "prompt": sum(u.prompt_tokens for u in usage),
        "completion": sum(u.completion_tokens for u in usage),
        "total": sum(u.total_tokens for u in usage),
        "latency": fmt_seconds(sum(u.latency_s for u in usage)),
    }
    final = trace.final_consensus
    errors = [s for s in trace.steps if s.status == "error"]
    return {
        "trace": trace,
        "models": [{"id": m, "short": short_model(m), "hue": hues[m]} for m in models],
        "hues": hues,
        "title": f"Council · {trace.strategy}",
        "started": trace.started_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "wall": fmt_seconds(trace.wall_time_s),
        "phases": phases,
        "n_steps": len(trace.steps),
        "n_errors": len(errors),
        "timeline": _timeline_view(trace, hues),
        "consensus": [_consensus_view(c, hues) for c in trace.consensus],
        "agreement": f"{final.agreement:.2f}" if final else None,
        "agreement_pct": round(final.agreement * 100) if final else None,
        "candidates": [
            {
                "label": c.label,
                "model": c.model,
                "short": short_model(c.model),
                "hue": hues.get(c.model, 210),
                "text": c.text,
                "score": None if c.score is None else f"{c.score:.2f}",
                "source": c.score_source,
                "selected": c.selected,
                "outlier": c.outlier,
            }
            for c in trace.candidates
        ],
        "has_scores": any(c.score is not None for c in trace.candidates),
        "usage": [
            {
                "model": u.model,
                "short": short_model(u.model),
                "hue": hues.get(u.model, 210),
                "calls": u.calls,
                "errors": u.errors,
                "prompt": f"{u.prompt_tokens:,}",
                "completion": f"{u.completion_tokens:,}",
                "total": f"{u.total_tokens:,}",
                "latency": fmt_seconds(u.latency_s),
            }
            for u in usage
        ],
        "usage_totals": {
            **totals,
            "prompt": f"{totals['prompt']:,}",
            "completion": f"{totals['completion']:,}",
            "total": f"{totals['total']:,}",
        },
        "total_tokens": f"{totals['total']:,}",
        "meta": {k: v for k, v in trace.meta.items() if k != "warnings"},
        "warnings": trace.meta.get("warnings", []),
    }


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pretty"] = _pretty
    return env


def _pretty(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(_pretty(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={_pretty(v)}" for k, v in value.items())
    return str(value)


def render_html(trace: Trace) -> str:
    """Render ``trace`` to one self-contained HTML document."""
    return _environment().get_template(TEMPLATE_NAME).render(**build_view(trace))


def save_html(trace: Trace, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(trace), encoding="utf-8")
    return out
