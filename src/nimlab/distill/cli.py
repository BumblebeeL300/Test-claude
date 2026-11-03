"""``nimlab distill ...``: recipes, run, resume, report, inspect and dedup."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from nimlab.core import NIMAuthError, NIMError, get_console

from .distiller import Distiller, DistillError, DistillResult
from .export import FORMATS, ExportError, read_dataset
from .models import RunOptions
from .quality import QualityReport
from .recipes import RecipeError, list_recipes
from .report import render_terminal, write_report_html
from .store import StoreError
from .textutil import distinct_n

app = typer.Typer(
    help="Distill: build instruction-tuning, preference and eval datasets from NIM models "
    "(generate, reward-score, filter, dedup, decontaminate, export).",
    no_args_is_help=True,
    add_completion=False,
)

MockOpt = Annotated[
    bool, typer.Option("--mock", help="Use the offline simulator (automatic without an API key).")
]
RunsDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--runs-dir",
        help="Where runs are checkpointed (default: .nimlab/distill/runs).",
        envvar="NIMLAB_DISTILL_RUNS",
    ),
]
_DASH = chr(0x2013)


def _fail(message: str) -> typer.Exit:
    get_console().print(f"[red]error:[/red] {message}")
    return typer.Exit(1)


def _split(value: str | None) -> list[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


def _run(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except NIMAuthError as exc:
        raise _fail(f"{exc}. Check NVIDIA_API_KEY, or use --mock.") from exc
    except (DistillError, RecipeError, StoreError, ExportError, FileNotFoundError, NIMError) as exc:
        raise _fail(str(exc)) from exc


@app.callback()
def _root(
    ctx: typer.Context,
    mock: Annotated[
        bool, typer.Option("--mock", help="Force the offline simulator for all commands.")
    ] = False,
    runs_dir: Annotated[
        Path | None, typer.Option("--runs-dir", envvar="NIMLAB_DISTILL_RUNS")
    ] = None,
) -> None:
    ctx.obj = {"mock": mock, "runs_dir": runs_dir}


def _distiller(
    ctx: typer.Context, mock: bool, runs_dir: Path | None, progress: _Progress | None = None
) -> Distiller:
    base: dict[str, Any] = ctx.obj or {}
    return Distiller(
        mock=True if (mock or base.get("mock")) else None,
        runs_dir=runs_dir or base.get("runs_dir"),
        on_event=progress.handle if progress else None,
    )


class _Progress:
    """Turns pipeline events into a progress bar and one line per finished stage."""

    def __init__(self) -> None:
        self.console = get_console()
        self.bar = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self.task: Any = None

    def __enter__(self) -> _Progress:
        self.bar.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.bar.stop()

    def handle(self, type_: str, data: dict[str, Any]) -> None:
        if type_ == "stage.start":
            if self.task is not None:
                self.bar.remove_task(self.task)
            self.task = self.bar.add_task(data["stage"], total=data.get("total") or None)
        elif type_ in ("sample.generated", "sample.scored") and self.task is not None:
            self.bar.update(self.task, completed=data.get("done", 0), total=data.get("total"))
        elif type_ == "stage.end":
            self.console.print(f"  [green]ok[/green] {data['stage']:<14} {data['seconds']:.2f}s")


def _banner(result: DistillResult) -> None:
    if result.meta.simulated:
        get_console().print(
            "[yellow]Simulated:[/yellow] no model was called. Text, scores and costs come from the "
            "offline simulator (set NVIDIA_API_KEY to use real NIM models)."
        )


def _print_result(result: DistillResult) -> None:
    console = get_console()
    _banner(result)
    console.print(render_terminal(result.report))
    meta = result.meta
    console.print(
        f"run [bold]{meta.id}[/bold] ({meta.status}) • {len(result.kept)} examples kept of "
        f"{meta.n} • est. cost ${meta.cost_usd:.4f}"
    )
    for path in result.files:
        console.print(f"  [dim]->[/dim] {path}")
    if meta.status != "done":
        console.print(f"[yellow]Not finished.[/yellow] Continue with: distill resume {meta.id}")


# --------------------------------------------------------------------------------- recipes


@app.command()
def recipes(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    show: Annotated[str | None, typer.Option("--show", help="Print one recipe as JSON.")] = None,
) -> None:
    """List the bundled recipes."""
    console = get_console()
    try:
        available = list_recipes()
    except RecipeError as exc:
        raise _fail(str(exc)) from exc
    if show:
        if show not in available:
            raise _fail(f"unknown recipe {show!r}; available: {', '.join(sorted(available))}")
        console.print_json(available[show].model_dump_json())
        return
    if as_json:
        console.print_json(json.dumps({k: v.model_dump() for k, v in available.items()}))
        return
    table = Table(title="Distill recipes", title_justify="left", header_style="bold")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Format")
    table.add_column("Description", overflow="fold")
    for rec in available.values():
        table.add_row(rec.name, rec.kind, rec.default_format, rec.description)
    console.print(table)


# --------------------------------------------------------------------------------- run


@app.command()
def run(
    ctx: typer.Context,
    recipe: Annotated[str, typer.Argument(help="Recipe name or path to a recipe JSON file.")],
    n: Annotated[int, typer.Option("--n", min=1, help="Candidate samples to generate.")] = 200,
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated generator models.")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output directory.")] = None,
    fmt: Annotated[
        str | None,
        typer.Option("--format", help=f"Export format: {', '.join(FORMATS)} (default: recipe)."),
    ] = None,
    docs: Annotated[Path | None, typer.Option("--docs", help="Documents for qa_from_docs.")] = None,
    eval_set: Annotated[
        Path | None, typer.Option("--eval-set", help="Eval set to decontaminate against.")
    ] = None,
    domain: Annotated[
        list[str] | None, typer.Option("--domain", help="Topic-tree domain (repeatable).")
    ] = None,
    threshold: Annotated[
        float | None, typer.Option("--threshold", min=0.01, max=1.0, help="Near-dup cosine.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the simulator.")] = 0,
    concurrency: Annotated[int | None, typer.Option("--concurrency", min=1, max=64)] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help="Name the run.")] = None,
    mock: MockOpt = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Generate a dataset from RECIPE."""
    options = RunOptions(
        seed=seed,
        domains=domain or None,
        concurrency=concurrency,
        dedup_threshold=threshold,
    )
    with _Progress() as progress:
        distiller = _distiller(ctx, mock, runs_dir, progress)
        result = _run(
            distiller.run(
                recipe,
                n=n,
                models=_split(models) or None,
                docs=docs,
                eval_set=eval_set,
                out=out,
                fmt=fmt,
                options=options,
                run_id=run_id,
            )
        )
    _print_result(result)


@app.command()
def resume(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(metavar="RUN", help="Run id (see `distill report`).")],
    out: Annotated[Path | None, typer.Option("--out", help="Output directory.")] = None,
    mock: MockOpt = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Continue an interrupted run from its checkpoint."""
    with _Progress() as progress:
        distiller = _distiller(ctx, mock, runs_dir, progress)
        result = _run(distiller.resume(run_id, out=out))
    _print_result(result)


# --------------------------------------------------------------------------------- report


def _print_report(report: QualityReport) -> None:
    console = get_console()
    console.print(render_terminal(report))
    div = report.diversity
    dist = _DASH if div.mean_pairwise_distance is None else f"{div.mean_pairwise_distance:.3f}"
    console.print(
        f"kept {report.n_final}/{report.n_requested} • mean pairwise distance {dist} • "
        f"distinct-1/2/3 {div.distinct_1:.3f}/{div.distinct_2:.3f}/{div.distinct_3:.3f}"
    )
    models = Table(title="Models", title_justify="left", header_style="bold")
    for col in ("Model", "Generated", "Kept", "Mean score"):
        models.add_column(col, justify="left" if col == "Model" else "right")
    for m in report.models:
        score = _DASH if m.mean_score is None else f"{m.mean_score:.2f}"
        models.add_row(m.model, str(m.generated), str(m.kept), score)
    console.print(models)
    for title, views in (("Top samples", report.top[:3]), ("Lowest kept", report.bottom[:3])):
        for v in views:
            console.print(
                Panel(
                    f"{v.response[:300]}",
                    title=f"{title} • {v.score_source} {v.score} • {v.prompt[:70]}",
                    title_align="left",
                )
            )


@app.command()
def report(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(metavar="RUN")],
    html_out: Annotated[
        Path | None, typer.Option("--html-out", help="Write the HTML quality report here.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Show the quality report of a run (funnel, scores, diversity); optionally write HTML."""
    distiller = _distiller(ctx, False, runs_dir)
    try:
        result = distiller.load(run_id)
    except (StoreError, DistillError) as exc:
        raise _fail(str(exc)) from exc
    if as_json:
        get_console().print_json(result.report.model_dump_json(by_alias=True))
    else:
        if result.meta.simulated:
            get_console().print("[yellow]Simulated run:[/yellow] scores are not real.")
        _print_report(result.report)
    if html_out is not None:
        path = write_report_html(result.report, result.meta, html_out)
        get_console().print(f"[dim]HTML report -> {path}[/dim]")


# --------------------------------------------------------------------------------- inspect


@app.command()
def inspect(
    file: Annotated[Path, typer.Argument(help="A JSONL dataset in any supported format.")],
    sample: Annotated[int, typer.Option("--sample", min=0, help="Examples to show.")] = 5,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Summarize a dataset file and print random examples."""
    console = get_console()
    try:
        records = read_dataset(file)
    except ExportError as exc:
        raise _fail(str(exc)) from exc
    if not records:
        raise _fail(f"{file} contains no records")
    prompts = [r.prompt for r in records]
    resp_len = sorted(len(r.response) for r in records)
    pairs = sum(1 for r in records if r.rejected is not None)
    console.print(
        f"[bold]{file}[/bold]: {len(records)} records"
        + (f", {pairs} preference pairs" if pairs else "")
        + f" • response chars median {resp_len[len(resp_len) // 2]}, "
        f"min {resp_len[0]}, max {resp_len[-1]} • distinct-2 (prompts) "
        f"{distinct_n(prompts, 2):.3f} • unique prompts {len(set(prompts))}"
    )
    rng = random.Random(seed)
    for i, rec in enumerate(rng.sample(records, min(sample, len(records))), 1):
        body = f"[bold]prompt[/bold]\n{rec.prompt}\n\n[bold]response[/bold]\n{rec.response}"
        if rec.rejected is not None:
            body += f"\n\n[bold]rejected[/bold]\n{rec.rejected}"
        meta = rec.meta
        sub = f"score {meta.get('score')} ({meta.get('score_source')})" if meta.get("score") else ""
        console.print(Panel(body, title=f"example {i}", subtitle=sub, title_align="left"))


# --------------------------------------------------------------------------------- dedup


@app.command()
def dedup(
    ctx: typer.Context,
    file: Annotated[Path, typer.Argument(help="A JSONL dataset in any supported format.")],
    threshold: Annotated[
        float, typer.Option("--threshold", min=0.01, max=1.0, help="Cosine similarity cut-off.")
    ] = 0.92,
    field: Annotated[str, typer.Option("--field", help="prompt, response or both.")] = "prompt",
    out: Annotated[
        Path | None, typer.Option("--out", help="Output (default: FILE.dedup.jsonl).")
    ] = None,
    mock: MockOpt = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Remove exact and near-duplicate rows from FILE (embedding cosine >= threshold)."""
    distiller = _distiller(ctx, mock, runs_dir)
    rows, kept, target = _run(distiller.dedup_file(file, threshold=threshold, field=field, out=out))
    console = get_console()
    if distiller._force_mock or (
        distiller._base is not None and getattr(distiller._base, "mock", False)
    ):
        console.print("[yellow]Mock embeddings:[/yellow] similarity is lexical, not semantic.")
    console.print(f"{rows} rows -> {kept} kept ({rows - kept} duplicates) • wrote {target}")


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
