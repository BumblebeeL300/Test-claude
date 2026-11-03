"""``nimlab loom ...``: list, run, lint, graph, export and scaffold pipelines."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from nimlab.core.logging import get_console

from .cache import NodeCache, default_cache_dir
from .engine import run_pipeline
from .errors import LoomError
from .export import to_mermaid, to_python
from .library import list_pipelines, load_pipeline, new_pipeline, slugify
from .linter import lint as lint_pipeline
from .schema import Pipeline
from .template import stringify

app = typer.Typer(
    help="Loom: visual, patchable pipelines of NIM models (JSON DAGs).",
    no_args_is_help=True,
    add_completion=False,
)


def _fail(message: str, code: int = 2) -> typer.Exit:
    Console(stderr=True).print(f"[red]error:[/red] {escape(message)}")
    return typer.Exit(code)


def _load(ref: str) -> Pipeline:
    try:
        return load_pipeline(ref)
    except LoomError as exc:
        raise _fail(str(exc)) from exc


def _parse_sets(pairs: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise _fail(f"--set expects key=value, got {pair!r}")
        if raw.startswith("@"):
            try:
                raw = Path(raw[1:]).read_text(encoding="utf-8")
            except OSError as exc:
                raise _fail(f"cannot read {raw[1:]}: {exc}") from exc
        values[key.strip()] = raw
    return values


@app.command("list")
def list_cmd(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """List the bundled pipelines (and any saved in ~/NIMLab/loom)."""
    items = list_pipelines()
    if as_json:
        typer.echo(json.dumps([i.to_dict() for i in items], indent=2))
        return
    table = Table(title="Loom pipelines", show_lines=False)
    for col in ("id", "nodes", "source", "description"):
        table.add_column(col)
    for i in items:
        table.add_row(i.id, str(i.nodes), i.source, i.description)
    get_console().print(table)
    get_console().print("[dim]Run one with: nimlab loom run <id> --mock[/dim]")


@app.command()
def run(
    pipeline: str = typer.Argument(..., help="A pipeline JSON file or a bundled/saved name."),
    sets: list[str] = typer.Option(
        [], "--set", "-s", help="Pipeline input as key=value (value=@file reads a file)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the result as JSON only."),
    mock: bool = typer.Option(False, "--mock", help="Offline mock mode (same as NIMLAB_MOCK=1)."),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Use the on-disk node cache."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="No per-node progress."),
    allow_http: list[str] = typer.Option([], "--allow-http", help="Host http_get nodes may call."),
) -> None:
    """Run a pipeline and print its outputs."""
    pipe = _load(pipeline)
    inputs = _parse_sets(sets)
    err = Console(stderr=True)

    def on_event(ev: dict[str, Any]) -> None:
        if as_json or quiet:
            return
        kind, node = ev["type"], ev.get("node")
        if kind == "node.end":
            tag = "cached" if ev["cached"] else f"{ev['latency_ms']} ms, {ev['tokens']} tok"
            err.print(f"[green]ok[/green]      {node} [dim]({tag})[/dim]")
        elif kind == "node.skipped":
            err.print(f"[dim]skipped {node}: {escape(ev['reason'])}[/dim]")
        elif kind == "node.retry":
            err.print(
                f"[yellow]retry[/yellow]   {node} (attempt {ev['attempt']}): {escape(ev['error'])}"
            )
        elif kind == "node.error":
            err.print(f"[red]failed[/red]  {node}: {escape(ev['error'])}")

    previous = os.environ.get("NIMLAB_MOCK")
    if mock:
        os.environ["NIMLAB_MOCK"] = "1"
    try:
        result = asyncio.run(
            run_pipeline(
                pipe,
                inputs,
                on_event=on_event,
                cache=cache,
                stream=False,
                http_allow=allow_http,
                mock=True if mock else None,
            )
        )
    except LoomError as exc:
        raise _fail(str(exc)) from exc
    finally:
        if mock:
            if previous is None:
                os.environ.pop("NIMLAB_MOCK", None)
            else:
                os.environ["NIMLAB_MOCK"] = previous

    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        console = get_console()
        for name, value in result.outputs.items():
            console.print(Panel(escape(stringify(value)), title=name, title_align="left"))
        console.print(
            f"[dim]{len(result.nodes)} nodes, {result.tokens} tokens, "
            f"{result.elapsed_s:.2f}s, {result.cache_hits} cache hit(s)[/dim]"
        )
    if not result.ok:
        raise typer.Exit(1)


@app.command("lint")
def lint_cmd(
    file: str = typer.Argument(..., help="A pipeline JSON file or a bundled/saved name."),
    strict: bool = typer.Option(False, "--strict", help="Warnings fail too."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Check a pipeline: unknown variables, type mismatches, unreachable nodes and more."""
    path = Path(file)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _fail(f"cannot read {file}: {exc}") from exc
        report = lint_pipeline(data)
    else:
        report = lint_pipeline(_load(file))
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        console = get_console()
        for issue in report.issues:
            color = "red" if issue.severity == "error" else "yellow"
            console.print(f"[{color}]{issue.severity:<7}[/{color}] {escape(issue.format())}")
        if not report.issues:
            console.print("[green]clean:[/green] no problems found")
        else:
            console.print(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    if report.errors or (strict and report.warnings):
        raise typer.Exit(1)


@app.command()
def graph(
    file: str = typer.Argument(..., help="A pipeline JSON file or a bundled/saved name."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write to a file."),
) -> None:
    """Print the pipeline as a mermaid flowchart."""
    text = to_mermaid(_load(file))
    if output:
        output.write_text(text, encoding="utf-8")
        get_console().print(f"[dim]wrote {output}[/dim]")
    else:
        typer.echo(text, nl=False)


@app.command()
def export(
    file: str = typer.Argument(..., help="A pipeline JSON file or a bundled/saved name."),
    output: Path = typer.Option(..., "--output", "-o", help="Python script to write."),
) -> None:
    """Export a standalone Python script that uses nimlab.core."""
    pipe = _load(file)
    try:
        script = to_python(pipe)
    except (LoomError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    output.write_text(script, encoding="utf-8")
    output.chmod(output.stat().st_mode | 0o111)
    get_console().print(f"[dim]wrote {output}  (run: python {output} --mock)[/dim]")


@app.command()
def new(
    name: str = typer.Argument(..., help="Pipeline name."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="File (default: ./<name>.json)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
) -> None:
    """Scaffold a starter pipeline (input -> prompt -> output)."""
    target = output or Path(f"{slugify(name)}.json")
    if target.exists() and not force:
        raise _fail(f"{target} already exists (use --force)")
    new_pipeline(name).save(target)
    console = get_console()
    console.print(f"[green]created[/green] {target}")
    console.print(f"[dim]next: nimlab loom lint {target} && nimlab loom run {target} --mock[/dim]")


@app.command("cache-clear")
def cache_clear() -> None:
    """Delete the on-disk node cache."""
    n = NodeCache().clear()
    get_console().print(f"removed {n} cached result(s) from {default_cache_dir()}")


if __name__ == "__main__":
    app()
