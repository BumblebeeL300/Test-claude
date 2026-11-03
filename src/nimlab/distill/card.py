"""The dataset card: a README with provenance, filters, counts and a licence placeholder."""

from __future__ import annotations

from .models import DROP_REASONS, RunMeta
from .quality import QualityReport

__all__ = ["render_card"]

_SCHEMAS = {
    "openai-chat": '{"messages": [{"role": "user", "content": "..."}, '
    '{"role": "assistant", "content": "..."}]}',
    "alpaca": '{"instruction": "...", "input": "...", "output": "..."}',
    "sharegpt": '{"conversations": [{"from": "human", "value": "..."}, '
    '{"from": "gpt", "value": "..."}]}',
    "dpo": '{"prompt": "...", "chosen": "...", "rejected": "..."}',
}


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_card(meta: RunMeta, report: QualityReport, data_file: str) -> str:
    """Markdown dataset card for a finished run."""
    rec = meta.recipe
    f, d, dc = rec.filters, rec.dedup, rec.decontaminate
    lines: list[str] = [
        "---",
        "license: TODO",
        "task_categories:",
        "  - text-generation",
        "language:",
        "  - en",
        "tags:",
        "  - synthetic",
        "  - nimlab-distill",
        *(["  - simulated"] if meta.simulated else []),
        f"size_categories: {_size_bucket(report.n_final)}",
        "---",
        "",
        f"# Dataset card: {rec.name}",
        "",
    ]
    if meta.simulated:
        lines += [
            "> **SIMULATED DATA.** This dataset was produced by the NIM Lab Distill offline "
            "simulator (no model was called). The text is synthetic filler written from templates "
            "and the scores are heuristics. Use it to test pipelines, never to train or "
            "evaluate a model.",
            "",
        ]
    lines += [
        rec.description or f"Synthetic `{rec.kind}` dataset built with NIM Lab Distill.",
        "",
        "## Overview",
        "",
        f"- **Examples:** {report.n_final} (from {report.n_requested} candidates, "
        f"{report.retained:.0%} retained)",
        f"- **Format:** `{meta.format}` in `{data_file}`",
        f"- **Run:** `{meta.id}` created {meta.created_at}, nimlab {meta.version or 'unknown'}",
        "",
        "## Provenance",
        "",
        f"- **Recipe:** `{rec.name}` (kind `{rec.kind}`)",
        "- **Generator models:** " + ", ".join(f"`{m}`" for m in meta.models),
        f"- **Seed / topic model:** `{meta.seed_model}`",
        f"- **Reward model:** `{meta.reward_model}` (LLM-judge fallback: `{meta.judge_model}`)",
        f"- **Embedding model (dedup):** `{meta.embed_model}`",
    ]
    if meta.weak_model:
        lines.append(f"- **Weak model (difficulty estimate):** `{meta.weak_model}`")
    if meta.docs:
        lines.append(f"- **Source documents:** `{meta.docs}`")
    if meta.eval_set:
        lines.append(f"- **Decontamination eval set:** `{meta.eval_set}`")
    lines += [
        f"- **Sampling temperatures:** {', '.join(str(t) for t in rec.temperatures)}",
        f"- **Random seed:** {meta.options.seed}",
        "",
        "## Pipeline and filters",
        "",
        "| Stage | Setting |",
        "|---|---|",
        f"| Score | reward >= {f.min_reward}, judge >= {f.min_judge} (fallback scale 0-10) |",
        f"| Length | response {f.min_chars}-{f.max_chars} chars, "
        f"prompt >= {f.min_prompt_chars} chars |",
        f"| Refusals | {'dropped' if f.drop_refusals else 'kept'} |",
        f"| Language | {f.language or 'not checked'} |",
        f"| Repetition | 3-gram repeat ratio <= {f.max_repeat_ratio} |",
    ]
    if rec.kind == "preference_pairs":
        lines.append(f"| Preference margin | chosen - rejected >= {f.min_margin} |")
    if f.min_grounding is not None:
        lines.append(f"| Grounding | >= {f.min_grounding:.0%} of answer words in the excerpt |")
    if f.min_difficulty is not None:
        lines.append(f"| Difficulty | >= {f.min_difficulty} |")
    lines += [
        f"| Dedup | exact={d.exact}, near={d.near} (cosine >= {d.threshold}, on {d.field}) |",
        f"| Decontamination | {'on' if dc.enabled else 'off'}, {dc.ngram}-gram overlap >= "
        f"{dc.min_overlap:.0%} |",
        "",
        "## Counts",
        "",
        "| Stage | In | Out | Dropped | Why |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report.funnel:
        why = ", ".join(f"{k} {v}" for k, v in row.reasons.items()) or "-"
        lines.append(f"| {row.label} | {row.in_} | {row.out} | {row.dropped} | {why} |")
    if report.drop_reasons:
        lines += ["", "Drop reasons:", ""]
        lines += [
            f"- `{k}` ({DROP_REASONS.get(k, '')}): {v}" for k, v in report.drop_reasons.items()
        ]
    lines += ["", "## Quality", ""]
    div = report.diversity
    lines += [
        f"- Score source: {report.dominant_source or 'n/a'}",
        f"- Mean pairwise embedding distance: {_fmt(div.mean_pairwise_distance, 3)}",
        f"- Distinct-1 / 2 / 3 (prompts): {div.distinct_1:.3f} / {div.distinct_2:.3f} / "
        f"{div.distinct_3:.3f}",
        "",
        "## Format",
        "",
        f"One JSON object per line (`{meta.format}`):",
        "",
        "```json",
        _SCHEMAS[meta.format],
        "```",
        "",
        "## Usage and cost",
        "",
        f"- Model calls: {report.usage['total_calls']}, tokens: {report.usage['total_tokens']}",
        f"- Estimated cost: ${report.usage['cost_usd']:.4f} (estimate from model cost tiers, "
        "not billing data)",
        "",
        "## Licence",
        "",
        "TODO: choose a licence. Check the terms of every generator model (outputs may carry "
        "restrictions) and of any source documents before you redistribute this dataset.",
        "",
        "## Limitations",
        "",
        "Synthetic data inherits the errors and biases of the models that wrote and scored it. "
        "Reward and judge scores are proxies for quality, not ground truth.",
        "",
    ]
    return "\n".join(lines)


def _size_bucket(n: int) -> str:
    for limit, name in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if n < limit:
            return name
    return "100K<n<1M"
