"""``nimlab council ...`` commands."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from nimlab.core.logging import get_console

from . import strategies as _builtin  # noqa: F401  (registers the built-in strategies)
from .config import CouncilConfig
from .council import Council
from .errors import CouncilError
from .live import CouncilLive
from .records import CouncilResult, Trace
from .report import save_html
from .strategies.base import get_strategy, list_strategies
from .terminal import print_result, strategies_table

app = typer.Typer(
    help="Council: multi-model deliberation (mixture-of-agents, debate, best-of-n, critic loop).",
    no_args_is_help=True,
    add_completion=False,
)


def _split_models(value: str | None) -> list[str] | None:
    if not value:
        return None
    models = [m.strip() for m in value.split(",") if m.strip()]
    return models or None


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question for the council."),
    strategy: str = typer.Option(
        "moa", "--strategy", "-s", help="moa | debate | best-of-n | critic"
    ),
    models: str | None = typer.Option(
        None, "--models", "-m", help="Comma-separated model ids (default: registry panel)."
    ),
    rounds: int | None = typer.Option(
        None, "--rounds", "-r", min=0, help="Debate rounds (default 2)."
    ),
    layers: int | None = typer.Option(
        None, "--layers", "-l", min=0, help="MoA aggregation layers (default 2)."
    ),
    n: int | None = typer.Option(None, "--n", min=1, help="Best-of-N candidates (default 4)."),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", min=0, help="Critic loop iterations (default 3)."
    ),
    aggregator: str | None = typer.Option(None, "--aggregator", help="MoA final aggregator model."),
    judge: str | None = typer.Option(None, "--judge", help="Debate / rubric judge model."),
    critic: str | None = typer.Option(None, "--critic", help="Critic-loop reviewer model."),
    reward_model: str | None = typer.Option(None, "--reward-model", help="Best-of-N reward model."),
    embed_model: str | None = typer.Option(
        None, "--embed-model", help="Embedding model for consensus."
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", min=0.0, max=1.0, help="Debate convergence threshold (default 0.9)."
    ),
    temperature: float | None = typer.Option(None, "--temperature", min=0.0, max=2.0),
    max_tokens: int | None = typer.Option(None, "--max-tokens", min=16),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write the full trace as JSON."),
    html_out: Path | None = typer.Option(None, "--html-out", help="Write the HTML trace report."),
    mock: bool = typer.Option(False, "--mock", help="Offline mock mode (same as NIMLAB_MOCK=1)."),
    live: bool | None = typer.Option(
        None, "--live/--no-live", help="Live view (default: only on a terminal)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Also print every step."),
) -> None:
    """Ask the council a question."""
    console = get_console()
    try:
        get_strategy(strategy)
    except CouncilError as exc:
        raise typer.BadParameter(str(exc), param_hint="--strategy") from exc

    overrides: dict[str, Any] = {
        "rounds": rounds,
        "layers": layers,
        "n": n,
        "max_iterations": max_iterations,
        "aggregator": aggregator,
        "judge": judge,
        "critic": critic,
        "reward_model": reward_model,
        "embed_model": embed_model,
        "convergence_threshold": threshold,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    previous = os.environ.get("NIMLAB_MOCK")
    if mock:
        os.environ["NIMLAB_MOCK"] = "1"
    try:
        result, error, usage = asyncio.run(
            _ask(question, strategy, _split_models(models), overrides, live, console)
        )
    finally:
        if mock:
            if previous is None:
                os.environ.pop("NIMLAB_MOCK", None)
            else:
                os.environ["NIMLAB_MOCK"] = previous

    trace: Trace | None = result.trace if result else (error.trace if error else None)
    if trace is not None:
        if json_out:
            trace.save(json_out)
            console.print(f"[dim]trace JSON  -> {json_out}[/]")
        if html_out:
            save_html(trace, html_out)
            console.print(f"[dim]trace HTML  -> {html_out}[/]")
    if error is not None:
        console.print(f"[bold red]council failed:[/] {error}")
        raise typer.Exit(1)
    assert result is not None
    print_result(console, result, verbose=verbose)
    if usage is not None:
        console.print(usage)


async def _ask(
    question: str,
    strategy: str,
    models: list[str] | None,
    overrides: dict[str, Any],
    live: bool | None,
    console,
) -> tuple[CouncilResult | None, CouncilError | None, Table | None]:
    from nimlab.core.client import NIMClient

    use_live = console.is_terminal if live is None else live
    async with NIMClient() as client:
        council = Council(client, config=CouncilConfig())
        try:
            if use_live:
                with CouncilLive(console) as view:
                    result = await council.run(
                        question, strategy, models=models, observers=[view], **overrides
                    )
            else:
                result = await council.run(question, strategy, models=models, **overrides)
        except CouncilError as exc:
            return None, exc, None
        return result, None, client.usage.rich_table("Token usage")


@app.command()
def strategies() -> None:
    """List the available deliberation strategies."""
    get_console().print(strategies_table(list_strategies()))


@app.command()
def render(
    trace_file: Path = typer.Argument(..., exists=True, dir_okay=False, help="A trace JSON file."),
    html_out: Path = typer.Option(
        Path("council-trace.html"), "--html-out", "-o", help="Output HTML file."
    ),
) -> None:
    """Render a saved trace JSON file as the HTML report."""
    console = get_console()
    try:
        trace = Trace.load(trace_file)
    except ValueError as exc:
        console.print(f"[bold red]not a valid council trace:[/] {exc}")
        raise typer.Exit(1) from exc
    save_html(trace, html_out)
    table = Table.grid(padding=(0, 2))
    table.add_row("[dim]trace[/]", f"{trace.strategy} · {len(trace.steps)} steps")
    table.add_row("[dim]html[/]", str(html_out))
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
