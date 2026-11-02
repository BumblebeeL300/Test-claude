"""``nimlab arena ...``: benchmark, battle, report, route and serve."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from nimlab.core import NIMAuthError, NIMError, get_console

from .arena import Arena
from .leaderboard import Leaderboard
from .report import render_elo_table, render_terminal, write_html
from .results import ItemResult
from .router import PROFILES, RoutingDecision
from .suites import SuiteError, list_suites

app = typer.Typer(
    help="Arena: benchmark NIM models on eval suites, rank them with Elo, and route prompts "
    "to the best model (including an OpenAI-compatible routing proxy).",
    no_args_is_help=True,
    add_completion=False,
)

MockOpt = Annotated[
    bool,
    typer.Option(
        "--mock", help="Use the offline simulator (automatic when NVIDIA_API_KEY is unset)."
    ),
]
RunsDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--runs-dir",
        help="Where run files live (default: .nimlab/arena/runs).",
        envvar="NIMLAB_ARENA_RUNS",
    ),
]
SuiteDirOpt = Annotated[
    list[Path] | None,
    typer.Option("--suites-dir", help="Extra directory of suite JSON files (repeatable)."),
]


_DASH = chr(0x2013)


def _split(value: str | None) -> list[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


def _fail(message: str) -> typer.Exit:
    get_console().print(f"[red]error:[/red] {message}")
    return typer.Exit(1)


@app.callback()
def _root(
    ctx: typer.Context,
    mock: Annotated[
        bool, typer.Option("--mock", help="Force the offline simulator for all commands.")
    ] = False,
    runs_dir: Annotated[Path | None, typer.Option("--runs-dir", envvar="NIMLAB_ARENA_RUNS")] = None,
) -> None:
    ctx.obj = {"mock": mock, "runs_dir": runs_dir}


def _opts(ctx: typer.Context, mock: bool, runs_dir: Path | None) -> tuple[bool | None, Path | None]:
    base: dict[str, Any] = ctx.obj or {}
    return (True if (mock or base.get("mock")) else None), (runs_dir or base.get("runs_dir"))


def _run_async(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except NIMAuthError as exc:
        raise _fail(f"{exc}. Check NVIDIA_API_KEY, or use --mock.") from exc
    except (SuiteError, FileNotFoundError, ValueError, NIMError) as exc:
        raise _fail(str(exc)) from exc


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=get_console(),
        transient=True,
    )


def _banner(arena: Arena) -> None:
    if arena.mock:
        get_console().print(
            "[yellow]Mock mode:[/yellow] responses, latencies and judgments are simulated offline "
            "(set NVIDIA_API_KEY to benchmark real models)."
        )


# --------------------------------------------------------------------------------- suites


@app.command()
def suites(
    suites_dir: SuiteDirOpt = None,
    show: Annotated[str | None, typer.Option("--show", help="List the items of one suite.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """List the bundled (and custom) eval suites."""
    console = get_console()
    try:
        available = list_suites(list(suites_dir or []))
    except SuiteError as exc:
        raise _fail(str(exc)) from exc
    if show:
        suite = available.get(show)
        if suite is None:
            raise _fail(f"unknown suite {show!r}; available: {', '.join(sorted(available))}")
        table = Table(title=f"{suite.name}: {suite.description}", title_justify="left")
        table.add_column("Item")
        table.add_column("Grader")
        table.add_column("Prompt", overflow="fold")
        for item in suite.items:
            prompt = item.prompt.replace("\n", " ")
            table.add_row(
                item.id,
                suite.grader_for(item).type,
                prompt[:110] + ("..." if len(prompt) > 110 else ""),
            )
        console.print(table)
        return
    if as_json:
        console.print_json(
            json.dumps(
                [
                    {
                        "name": s.name,
                        "category": s.category,
                        "items": len(s.items),
                        "graders": s.grader_types(),
                        "open_ended": s.open_ended,
                    }
                    for s in available.values()
                ]
            )
        )
        return
    table = Table(title="Arena eval suites", title_justify="left", header_style="bold")
    for col in ("Suite", "Category", "Items", "Graders", "Battles", "Description"):
        table.add_column(col, overflow="fold")
    for s in available.values():
        table.add_row(
            s.name,
            s.category,
            str(len(s.items)),
            ", ".join(s.grader_types()),
            "yes" if s.open_ended else "",
            s.description,
        )
    console.print(table)


# --------------------------------------------------------------------------------- run


@app.command()
def run(
    ctx: typer.Context,
    suite: Annotated[
        str,
        typer.Option("--suite", "-s", help="'all', a suite name, a comma list, or a suite file."),
    ] = "all",
    models: Annotated[
        str | None,
        typer.Option("--models", "-m", help="Comma-separated model ids (default: registry picks)."),
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", min=1, help="Items in flight at once.")
    ] = 4,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Only the first N items of each suite.")
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Run id or file to continue.")
    ] = None,
    judge_model: Annotated[
        str | None, typer.Option("--judge-model", help="Judge for llm_judge suites.")
    ] = None,
    html_out: Annotated[
        Path | None, typer.Option("--html-out", help="Also write the HTML leaderboard.")
    ] = None,
    suites_dir: SuiteDirOpt = None,
    mock: MockOpt = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Benchmark models on suites (concurrently), save the run, print the leaderboard."""
    use_mock, root = _opts(ctx, mock, runs_dir)
    console = get_console()

    async def go() -> tuple[Arena, Leaderboard, Any]:
        async with Arena(
            mock=use_mock,
            runs_dir=root,
            concurrency=concurrency,
            judge_model=judge_model,
            suites_dirs=list(suites_dir or []),
        ) as arena:
            _banner(arena)
            wanted = _split(models) or None
            with _progress() as progress:
                task = progress.add_task("benchmarking", total=None)

                def tick(done: int, total: int, _: ItemResult) -> None:
                    progress.update(task, completed=done, total=total)

                result = await arena.run(
                    wanted, suite, limit=limit, resume=resume, on_progress=tick
                )
            board = Leaderboard.from_run(result, registry=arena.registry)
            return arena, board, result

    arena, board, result = _run_async(go())
    console.print(render_terminal(board))
    errors = sum(1 for r in result.items if not r.scored)
    if errors:
        console.print(
            f"[yellow]{errors} item(s) errored (API or grader failures); "
            f"rerun with --resume {result.run_id}[/yellow]"
        )
    if arena.store:
        console.print(f"Saved [bold]{arena.store.path_for(result.run_id)}[/bold]")
    if html_out:
        console.print(f"Wrote [bold]{write_html(board, html_out)}[/bold]")


# --------------------------------------------------------------------------------- battle


@app.command()
def battle(
    ctx: typer.Context,
    models: Annotated[
        str, typer.Option("--models", "-m", help="Comma-separated model ids (at least two).")
    ],
    suite: Annotated[
        str, typer.Option("--suite", "-s", help="Suite to battle on.")
    ] = "summarization",
    judge_model: Annotated[str | None, typer.Option("--judge-model")] = None,
    max_items: Annotated[int | None, typer.Option("--max-items")] = None,
    max_pairs: Annotated[
        int | None, typer.Option("--max-pairs", help="Sample at most N model pairs.")
    ] = None,
    from_run: Annotated[
        str | None, typer.Option("--from-run", help="Reuse the answers of this run.")
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", min=1)] = 4,
    html_out: Annotated[Path | None, typer.Option("--html-out")] = None,
    suites_dir: SuiteDirOpt = None,
    mock: MockOpt = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Pairwise LLM-judged battles (position-swapped) and Bradley-Terry Elo with CIs."""
    use_mock, root = _opts(ctx, mock, runs_dir)
    console = get_console()
    names = _split(models)

    async def go() -> tuple[Arena, Leaderboard, Any]:
        async with Arena(
            mock=use_mock,
            runs_dir=root,
            concurrency=concurrency,
            judge_model=judge_model,
            suites_dirs=list(suites_dir or []),
        ) as arena:
            _banner(arena)
            if arena.judge_model in names:
                console.print(
                    f"[yellow]Note:[/yellow] judge {arena.judge_model} is also a contestant; "
                    "judges tend to favour their own answers. Consider --judge-model."
                )
            with _progress() as progress:
                task = progress.add_task("battling", total=None)

                def tick(done: int, total: int, _: Any) -> None:
                    progress.update(task, completed=done, total=total)

                result = await arena.battle(
                    names,
                    suite,
                    run=from_run,
                    max_items=max_items,
                    max_pairs=max_pairs,
                    concurrency=concurrency,
                    on_progress=tick,
                    on_battle=tick,
                )
            board = Leaderboard.from_run(result, registry=arena.registry, elo_source="battles")
            return arena, board, result

    arena, board, result = _run_async(go())
    console.print(render_elo_table(board, result))
    failures = result.config.get("battle_failures") or []
    if failures:
        console.print(f"[yellow]{len(failures)} battle(s) failed and were skipped.[/yellow]")
    if arena.store:
        console.print(f"Saved [bold]{arena.store.path_for(result.run_id)}[/bold]")
    if html_out:
        console.print(f"Wrote [bold]{write_html(board, html_out)}[/bold]")


# --------------------------------------------------------------------------------- report


@app.command()
def report(
    ctx: typer.Context,
    run_ref: Annotated[
        str, typer.Argument(metavar="[RUN]", help="Run id, file path, or 'latest'.")
    ] = "latest",
    html_out: Annotated[
        Path | None, typer.Option("--html-out", help="Write a self-contained HTML leaderboard.")
    ] = None,
    boot: Annotated[
        int, typer.Option("--bootstrap", min=0, help="Bootstrap resamples for Elo CIs.")
    ] = 200,
    mock: MockOpt = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Show a saved run as a terminal leaderboard (and optionally an HTML report)."""
    from .results import RunStore

    _, root = _opts(ctx, mock, runs_dir)
    console = get_console()
    try:
        saved = RunStore(root).load(run_ref)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    board = Leaderboard.from_run(saved, n_boot=boot)
    console.print(render_terminal(board))
    console.print(render_elo_table(board, saved))
    if html_out:
        console.print(f"Wrote [bold]{write_html(board, html_out)}[/bold]")


# --------------------------------------------------------------------------------- route


def _print_decision(decision: RoutingDecision) -> None:
    console = get_console()
    console.print(
        Panel.fit(
            f"[bold]{decision.model}[/bold]\n"
            f"category [cyan]{decision.category}[/cyan] via {decision.classifier} "
            f"(confidence {decision.confidence:.0%}) • source {decision.source} • "
            f"profile {decision.profile}",
            title="Routing decision",
        )
    )
    for reason in decision.reasons:
        console.print(f"  • {reason}")
    if decision.candidates:
        table = Table(title="Candidates", title_justify="left", header_style="bold")
        for col in ("Model", "Quality", "Latency", "Cost", "Utility"):
            table.add_column(col, justify="right" if col != "Model" else "left", overflow="fold")
        for c in decision.candidates[:8]:
            table.add_row(
                ("→ " if c.model == decision.model else "  ") + c.model,
                f"{c.quality:.3f}" if c.quality is not None else _DASH,
                f"{c.latency_s:.2f}s" if c.latency_s is not None else _DASH,
                str(c.cost_tier) if c.cost_tier is not None else "?",
                f"{c.utility:.3f}" if c.utility is not None else _DASH,
            )
        console.print(table)


@app.command()
def route(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="The prompt to route.")],
    profile: Annotated[
        str, typer.Option("--profile", help=f"One of: {', '.join(PROFILES)}.")
    ] = "balanced",
    latency_weight: Annotated[
        float | None, typer.Option("--latency-weight", help="Utility penalty per second.")
    ] = None,
    cost_weight: Annotated[
        float | None, typer.Option("--cost-weight", help="Utility penalty per cost tier.")
    ] = None,
    category: Annotated[str | None, typer.Option("--category", help="Skip classification.")] = None,
    llm: Annotated[
        bool, typer.Option("--llm", help="Use the FAST model to classify when unsure.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    mock: MockOpt = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Explain which model Arena would route PROMPT to, and why."""
    use_mock, root = _opts(ctx, mock, runs_dir)

    async def go() -> RoutingDecision:
        async with Arena(mock=use_mock, runs_dir=root) as arena:
            router = arena.router(
                profile=profile, latency_weight=latency_weight, cost_weight=cost_weight, use_llm=llm
            )
            return await router.route(prompt, category=category)

    decision = _run_async(go())
    if as_json:
        get_console().print_json(decision.model_dump_json(exclude={"classification"}))
    else:
        _print_decision(decision)


# --------------------------------------------------------------------------------- serve


@app.command()
def serve(
    ctx: typer.Context,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p")] = 8000,
    mock: MockOpt = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Serve the OpenAI-compatible routing proxy (model 'nimlab/auto' routes automatically)."""
    import uvicorn

    from .proxy import create_app

    use_mock, root = _opts(ctx, mock, runs_dir)
    console = get_console()
    proxy = create_app(runs_dir=root, mock=use_mock)
    if getattr(proxy.state.client, "mock", False):
        console.print("[yellow]Mock mode:[/yellow] the proxy answers from the offline simulator.")
    console.print(f"Routing proxy on http://{host}:{port}/v1  (model: nimlab/auto)")
    uvicorn.run(proxy, host=host, port=port, log_level="info")
