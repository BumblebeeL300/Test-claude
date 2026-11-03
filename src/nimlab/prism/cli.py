"""``nimlab prism ...`` commands (also runnable standalone as ``python -m nimlab.prism.cli``)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape
from rich.table import Table

from nimlab.core import NIMError
from nimlab.core.logging import get_console

from .errors import PrismError
from .modes import list_modes
from .prism import BatchItem, Prism, PrismResult, write_jsonl
from .schemas import schema_names

app = typer.Typer(
    help="Prism: a vision workbench on NIM vision models (describe, ocr, extract, ui2code, ...).",
    no_args_is_help=True,
    add_completion=False,
)

IMG = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Image file.")
MOCK = typer.Option(False, "--mock", help="Offline mock mode (same as NIMLAB_MOCK=1).")
MODEL = typer.Option(None, "--model", "-m", help="Vision model id (default: registry vision pick).")
AS_JSON = typer.Option(False, "--json", help="Print the full result as JSON.")


@contextmanager
def _mock_env(mock: bool) -> Iterator[None]:
    previous = os.environ.get("NIMLAB_MOCK")
    if mock:
        os.environ["NIMLAB_MOCK"] = "1"
    try:
        yield
    finally:
        if mock:
            if previous is None:
                os.environ.pop("NIMLAB_MOCK", None)
            else:
                os.environ["NIMLAB_MOCK"] = previous


def _execute(mock: bool, job: Callable[[Prism], Awaitable[Any]], model: str | None = None) -> Any:
    """Run ``job(prism)`` on a fresh event loop; turn Prism/NIM errors into a clean exit."""
    console = get_console()

    async def go() -> Any:
        async with Prism(model=model) as prism:
            return await job(prism)

    with _mock_env(mock):
        try:
            return asyncio.run(go())
        except (PrismError, NIMError) as exc:
            console.print(f"[bold red]prism failed:[/] {escape(str(exc))}")
            raise typer.Exit(1) from exc


def _banner(result: PrismResult) -> None:
    if result.simulated:
        get_console().print(
            "[yellow]SIMULATED[/] offline mock output (no NVIDIA API key). "
            "Add a key for real model results.",
            highlight=False,
        )
    for note in result.notes[1 if result.simulated else 0 :]:
        get_console().print(f"[dim]note: {escape(note)}[/]")


def _emit(result: PrismResult, as_json: bool, out: Path | None = None) -> None:
    """Print the result (or its JSON) and optionally save the primary text to ``out``."""
    console = get_console()
    _banner(result)
    if out is not None:
        out.write_text(result.text + ("" if result.text.endswith("\n") else "\n"), encoding="utf-8")
        console.print(f"[dim]wrote {out}[/]")
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    elif out is None:
        typer.echo(result.text)


@app.command()
def modes() -> None:
    """List the modes and the built-in extraction schemas."""
    console = get_console()
    table = Table(title="Prism modes", show_lines=False)
    for col in ("mode", "images", "options", "what it does"):
        table.add_column(col)
    for m in list_modes():
        opts = ", ".join(f"--{o.name}" for o in m.options) or "-"
        table.add_row(m.name, str(m.max_images), opts, m.description)
    console.print(table)
    console.print(f"[dim]extract schemas: {', '.join(schema_names())}[/]")


@app.command()
def describe(
    image: Path = IMG,
    detail: str = typer.Option("rich", "--detail", help="brief | rich"),
    model: str | None = MODEL,
    as_json: bool = AS_JSON,
    mock: bool = MOCK,
) -> None:
    """Describe an image and write alt text for it."""
    result = _execute(mock, lambda p: p.describe(image, detail=detail), model)
    _emit(result, as_json)


@app.command()
def ocr(
    image: Path = IMG,
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the text to this file."),
    model: str | None = MODEL,
    as_json: bool = AS_JSON,
    mock: bool = MOCK,
) -> None:
    """Extract text from an image, keeping its layout."""
    result = _execute(mock, lambda p: p.ocr(image), model)
    _emit(result, as_json, out)


@app.command()
def extract(
    image: Path = IMG,
    schema: str = typer.Option(
        "receipt",
        "--schema",
        "-s",
        help="receipt | invoice | business_card | chart_data | path to a JSON Schema file",
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the JSON to this file."),
    model: str | None = MODEL,
    as_json: bool = AS_JSON,
    mock: bool = MOCK,
) -> None:
    """Extract structured data into a built-in or custom JSON Schema."""
    result = _execute(mock, lambda p: p.extract(image, schema=schema), model)
    _emit(result, as_json, out)
    if not result.output["valid"]:
        get_console().print("[bold red]output does not match the schema:[/]")
        for err in result.output["errors"]:
            get_console().print(f"  - {escape(err)}")
        raise typer.Exit(2)


@app.command()
def ui2code(
    image: Path = IMG,
    out: Path = typer.Option(Path("out.html"), "--out", "-o", help="Where to write the HTML."),
    refine: int = typer.Option(0, "--refine", "-r", min=0, max=5, help="Critique-and-fix rounds."),
    model: str | None = MODEL,
    as_json: bool = AS_JSON,
    mock: bool = MOCK,
) -> None:
    """Turn a screenshot into one self-contained HTML/CSS file."""
    result = _execute(mock, lambda p: p.ui2code(image, refine=refine), model)
    out.write_text(result.output["html"], encoding="utf-8")
    _banner(result)
    console = get_console()
    for c in result.output["critiques"]:
        console.print(
            f"[dim]round {c['round']}: score {c['score']:.2f}, {len(c['issues'])} issue(s)[/]"
        )
    for problem in result.output["problems"]:
        console.print(f"[yellow]html: {escape(problem)}[/]")
    console.print(f"wrote {out} ({result.output['drafts']} draft(s))")
    if as_json:
        typer.echo(result.model_dump_json(indent=2))


@app.command()
def compare(
    a: Path = typer.Argument(..., exists=True, dir_okay=False, help="First image."),
    b: Path = typer.Argument(..., exists=True, dir_okay=False, help="Second image."),
    model: str | None = MODEL,
    as_json: bool = AS_JSON,
    mock: bool = MOCK,
) -> None:
    """Diff two images and list the differences."""
    result = _execute(mock, lambda p: p.compare(a, b), model)
    _emit(result, as_json)


@app.command()
def chart(
    image: Path = IMG,
    csv: Path | None = typer.Option(None, "--csv", help="Write the data table as CSV."),
    model: str | None = MODEL,
    as_json: bool = AS_JSON,
    mock: bool = MOCK,
) -> None:
    """Read a chart back into a data table and a summary."""
    result = _execute(mock, lambda p: p.chart(image), model)
    _banner(result)
    if csv is not None:
        csv.write_text(result.output["csv"], encoding="utf-8")
        get_console().print(f"[dim]wrote {csv}[/]")
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(result.text)


@app.command()
def ask(
    image: Path = IMG,
    question: str = typer.Argument(..., help="What do you want to know about the image?"),
    model: str | None = MODEL,
    as_json: bool = AS_JSON,
    mock: bool = MOCK,
) -> None:
    """Ask a free-form question about an image."""
    result = _execute(mock, lambda p: p.ask(image, question), model)
    _emit(result, as_json)


@app.command()
def batch(
    directory: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of images."),
    mode: str = typer.Option("ocr", "--mode", help="Any single-image mode."),
    out: Path = typer.Option(Path("results.jsonl"), "--out", "-o", help="JSONL results file."),
    concurrency: int = typer.Option(4, "--concurrency", "-c", min=1, max=32),
    recursive: bool = typer.Option(False, "--recursive", "-R", help="Include subfolders."),
    question: str | None = typer.Option(None, "--question", "-q", help="For --mode ask."),
    schema: str | None = typer.Option(None, "--schema", "-s", help="For --mode extract."),
    refine: int | None = typer.Option(None, "--refine", help="For --mode ui2code."),
    model: str | None = MODEL,
    mock: bool = MOCK,
) -> None:
    """Run one mode over every image in a folder, concurrently, and write JSONL."""
    options: dict[str, Any] = {
        k: v
        for k, v in {"question": question, "schema": schema, "refine": refine}.items()
        if v is not None
    }
    items: list[BatchItem] = _execute(
        mock,
        lambda p: p.batch(directory, mode, concurrency=concurrency, recursive=recursive, **options),
        model,
    )
    write_jsonl(items, out)
    console = get_console()
    failed = [i for i in items if not i.ok]
    if any(i.result and i.result.simulated for i in items):
        console.print("[yellow]SIMULATED[/] offline mock output (no NVIDIA API key).")
    for item in failed:
        console.print(f"[red]failed[/] {escape(item.name)}: {escape(item.error or '')}")
    console.print(f"{len(items) - len(failed)}/{len(items)} succeeded -> {out}")
    if failed:
        raise typer.Exit(1)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
