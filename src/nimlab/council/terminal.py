"""Rich renderables for finished runs (summary, consensus heat-table, step list, strategies)."""

from __future__ import annotations

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .live import short_model
from .records import ConsensusReport, CouncilResult, Trace
from .strategies.base import Strategy


def _heat_style(value: float) -> str:
    if value >= 0.9:
        return "black on green3"
    if value >= 0.75:
        return "black on chartreuse3"
    if value >= 0.55:
        return "black on yellow3"
    if value >= 0.35:
        return "black on dark_orange3"
    return "white on red3"


def consensus_table(report: ConsensusReport) -> Table:
    """Pairwise similarity matrix as a colour-coded table (green = agree, red = diverge)."""
    table = Table(
        title=f"Consensus · {report.phase}  (agreement {report.agreement:.2f}, {report.method})",
        title_justify="left",
    )
    table.add_column("")
    outliers = set(report.outliers)
    for i, label in enumerate(report.labels):
        table.add_column(label + ("*" if i in outliers else ""), justify="center")
    table.add_column("mean", justify="center", style="dim")
    for i, row in enumerate(report.matrix):
        name = f"{report.labels[i]} · {short_model(report.models[i])}"
        cells = [
            Text(f"{v:.2f}", style="dim" if i == j else _heat_style(v)) for j, v in enumerate(row)
        ]
        table.add_row(
            Text(name, style="red" if i in outliers else ""), *cells, f"{report.centrality[i]:.2f}"
        )
    if outliers:
        table.caption = f"* outlier: {', '.join(report.outlier_labels)}"
    return table


def summary_table(trace: Trace) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("strategy", f"[bold cyan]{trace.strategy}[/]" + (" (mock)" if trace.mock else ""))
    table.add_row("models", ", ".join(short_model(m) for m in trace.models) or "-")
    table.add_row(
        "steps", f"{len(trace.steps)}  ({sum(s.status == 'error' for s in trace.steps)} failed)"
    )
    table.add_row("wall time", f"{trace.wall_time_s:.2f}s")
    table.add_row("tokens", f"{trace.total_tokens:,}")
    final = trace.final_consensus
    if final is not None:
        outliers = ", ".join(final.outlier_labels) or "none"
        table.add_row("agreement", f"{final.agreement:.2f}  (outliers: {outliers})")
    for key in ("stop_reason", "rounds_run", "layers_run", "iterations", "scorer", "winner"):
        if key in trace.meta:
            table.add_row(key.replace("_", " "), str(trace.meta[key]))
    return table


def steps_table(trace: Trace) -> Table:
    table = Table(title="Steps", title_justify="left")
    for col, just in (("#", "right"), ("Phase", "left"), ("Type", "left"), ("Model", "left"),
                      ("Label", "left"), ("Time", "right"), ("Tokens", "right")):  # fmt: skip
        table.add_column(col, justify=just)  # type: ignore[arg-type]
    for s in trace.steps:
        table.add_row(
            str(s.index),
            s.phase,
            s.type.value,
            short_model(s.model),
            s.label,
            f"{s.duration_s:.2f}s",
            "-" if s.total_tokens is None else f"{s.total_tokens:,}",
            style="red" if s.status == "error" else None,
        )
    return table


def result_panel(result: CouncilResult) -> RenderableType:
    parts: list[RenderableType] = [Text(result.answer or "(no answer)")]
    if result.rationale:
        parts.append(Text(""))
        parts.append(Text.assemble(("Rationale: ", "bold dim"), (result.rationale, "dim")))
    return Panel(
        Group(*parts),
        title=f"[bold]Council answer[/] · {result.strategy}",
        title_align="left",
        border_style="green",
        padding=(1, 2),
    )


def strategies_table(strategies: list[Strategy]) -> Table:
    table = Table(title="Council strategies", title_justify="left", show_lines=True)
    table.add_column("Name", style="bold cyan", no_wrap=True)
    table.add_column("Description", ratio=3)
    table.add_column("Options", style="dim", ratio=1)
    table.add_column("Min models", justify="right", no_wrap=True)
    for s in strategies:
        aliases = getattr(s, "aliases", ())
        name = s.name + (f"\n[dim]({', '.join(aliases)})[/]" if aliases else "")
        options = ", ".join(f"--{o.replace('_', '-')}" for o in getattr(s, "options", ()))
        table.add_row(name, s.description, options, str(getattr(s, "min_models", 1)))
    return table


def print_result(console: Console, result: CouncilResult, *, verbose: bool = False) -> None:
    console.print(result_panel(result))
    console.print(summary_table(result.trace))
    if result.consensus is not None:
        console.print(consensus_table(result.consensus))
    if verbose:
        console.print(steps_table(result.trace))
    for warning in result.trace.meta.get("warnings", []):
        console.print(f"[yellow]warning:[/] {warning}")
