"""``sentinel ...`` (also ``nimlab sentinel ...``): suites, scan, report, shield and redact."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from nimlab.core import NIMAuthError, NIMError, get_console

from .lab import Sentinel
from .llm import default_models, make_client
from .models import CATEGORY_LABELS, ProbeResult
from .pii import PII_KINDS, Redactor
from .report import render_terminal, write_html
from .runner import ScanSettings
from .shield import Action, Shield, ShieldConfig
from .store import RunStore
from .suites import SuiteError, list_suites

app = typer.Typer(
    help="Sentinel: probe NIM models for prompt injection, jailbreaks, leaks and over-refusal, "
    "score them, and screen live traffic with Shield. Defensive use only.",
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
        help="Where run files live (default: .nimlab/sentinel/runs).",
        envvar="NIMLAB_SENTINEL_RUNS",
    ),
]
SuiteDirOpt = Annotated[
    list[Path] | None,
    typer.Option("--suites-dir", help="Extra directory of probe-suite JSON files (repeatable)."),
]


def _split(value: str | None) -> list[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


def _fail(message: str) -> typer.Exit:
    get_console().print(f"[red]error:[/red] {message}")
    return typer.Exit(1)


@app.callback()
def _root(
    ctx: typer.Context,
    mock: Annotated[bool, typer.Option("--mock", help="Force the offline simulator.")] = False,
    runs_dir: Annotated[
        Path | None, typer.Option("--runs-dir", envvar="NIMLAB_SENTINEL_RUNS")
    ] = None,
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


def _banner(simulated: bool) -> None:
    if simulated:
        get_console().print(
            "[yellow]Mock mode:[/yellow] model behaviour and judgments are [bold]simulated[/bold] "
            "offline (set NVIDIA_API_KEY to scan real models)."
        )


# --------------------------------------------------------------------------------- suites


@app.command()
def suites(
    suites_dir: SuiteDirOpt = None,
    show: Annotated[
        str | None, typer.Option("--show", help="List the probes of one suite.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """List the bundled (and custom) probe suites."""
    console = get_console()
    try:
        available = list_suites(list(suites_dir or []))
    except SuiteError as exc:
        raise _fail(str(exc)) from exc
    if show:
        if show not in available:
            raise _fail(f"unknown suite {show!r}; available: {', '.join(available)}")
        suite = available[show]
        if as_json:
            typer.echo(suite.model_dump_json(indent=2))
            return
        table = Table(title=f"{suite.name}: {suite.title}")
        table.add_column("Probe")
        table.add_column("Title")
        table.add_column("Expect")
        table.add_column("Prompt")
        for p in suite.probes:
            table.add_row(p.id, p.title, p.expect, " ".join(p.user.split())[:70])
        console.print(table)
        return
    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "name": s.name,
                        "category": s.category,
                        "title": s.title,
                        "probes": len(s.probes),
                    }
                    for s in available.values()
                ],
                indent=2,
            )
        )
        return
    table = Table(title="Sentinel probe suites")
    for col in ("Suite", "Category", "Probes", "Title"):
        table.add_column(col)
    for s in available.values():
        table.add_row(s.name, CATEGORY_LABELS[s.category], str(len(s.probes)), s.title)
    console.print(table)
    console.print(
        f"{sum(len(s.probes) for s in available.values())} probes in {len(available)} suites."
    )


# --------------------------------------------------------------------------------- scan


@app.command()
def scan(
    ctx: typer.Context,
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated model ids.")
    ] = None,
    suite: Annotated[
        str, typer.Option("--suite", help="'all', suite names or JSON files, comma-separated.")
    ] = "all",
    suites_dir: SuiteDirOpt = None,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Probes per suite (smoke test).")
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, max=64)] = 6,
    retries: Annotated[int, typer.Option("--retries", min=0, max=8)] = 2,
    guard: Annotated[
        bool | None,
        typer.Option(
            "--guard/--no-guard", help="Judge replies with NemoGuard (default: automatic)."
        ),
    ] = None,
    llm_judge: Annotated[
        str | None, typer.Option("--llm-judge", help="Model for an LLM refusal judge (optional).")
    ] = None,
    shield: Annotated[
        bool, typer.Option("--shield", help="Route every call through Shield (block mode).")
    ] = False,
    html_out: Annotated[
        Path | None, typer.Option("--html-out", help="Write a self-contained HTML scorecard.")
    ] = None,
    mock: MockOpt = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Probe models with the attack suites and print a scorecard per model."""
    use_mock, root = _opts(ctx, mock, runs_dir)
    console = get_console()

    async def go() -> Any:
        client = make_client(mock=use_mock)
        try:
            lab = Sentinel(
                client,
                runs_dir=root,
                suites_dirs=list(suites_dir or []),
                settings=ScanSettings(
                    concurrency=concurrency,
                    retries=retries,
                    guard=guard,
                    llm_judge_model=llm_judge,
                    shield=ShieldConfig() if shield else None,
                ),
            )
            chosen = _split(models) or default_models(lab.registry)
            resolved = lab.resolve_suites(_split(suite) or "all", limit)
            total = len(chosen) * sum(len(s.probes) for s in resolved)
            _banner(lab.mock)
            with Progress(
                TextColumn("[bold]scanning"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as bar:
                task = bar.add_task("scan", total=total)

                def tick(_r: ProbeResult, _done: int, _total: int) -> None:
                    bar.advance(task)

                run = await lab.scan(chosen, resolved, on_result=tick)
            return lab, run
        finally:
            if hasattr(client, "aclose"):
                await client.aclose()

    lab, run = _run_async(go())
    console.print(render_terminal(run))
    for card in run.scorecards:
        for note in card.notes:
            console.print(f"[dim]{card.model}: {note}[/dim]")
    if lab.store is not None:
        console.print(f"Saved run [bold]{run.run_id}[/bold] to {lab.store.root}")
    if html_out:
        console.print(f"Wrote [bold]{write_html(run, html_out)}[/bold]")


# --------------------------------------------------------------------------------- report


@app.command()
def report(
    ctx: typer.Context,
    run_ref: Annotated[
        str, typer.Argument(metavar="[RUN]", help="Run id, file path, or 'latest'.")
    ] = "latest",
    html_out: Annotated[
        Path | None, typer.Option("--html-out", help="Write a self-contained HTML scorecard.")
    ] = None,
    mock: MockOpt = False,
    runs_dir: RunsDirOpt = None,
) -> None:
    """Show a saved run as a terminal scorecard (and optionally an HTML report)."""
    _, root = _opts(ctx, mock, runs_dir)
    console = get_console()
    try:
        saved = RunStore(root).load(run_ref)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    _banner(saved.simulated)
    console.print(render_terminal(saved))
    if html_out:
        console.print(f"Wrote [bold]{write_html(saved, html_out)}[/bold]")


# --------------------------------------------------------------------------------- shield


@app.command()
def shield(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="The user prompt to screen.")],
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    system: Annotated[str | None, typer.Option("--system", help="Optional system prompt.")] = None,
    canary: Annotated[
        list[str] | None, typer.Option("--canary", help="String that must not appear in the reply.")
    ] = None,
    injection: Annotated[
        Action, typer.Option(help="Action for the injection heuristics.")
    ] = Action.BLOCK,
    pii_in: Annotated[Action, typer.Option(help="Action for PII in the prompt.")] = Action.ANNOTATE,
    guard_in: Annotated[
        Action, typer.Option(help="Action for NemoGuard on the prompt.")
    ] = Action.BLOCK,
    guard_out: Annotated[
        Action, typer.Option(help="Action for NemoGuard on the reply.")
    ] = Action.BLOCK,
    leak: Annotated[
        Action, typer.Option(help="Action for canary / system-prompt leaks.")
    ] = Action.REDACT,
    pii_out: Annotated[Action, typer.Option(help="Action for PII in the reply.")] = Action.REDACT,
    mock: MockOpt = False,
) -> None:
    """Run a prompt through Shield and show every check (and the screened reply)."""
    use_mock, _ = _opts(ctx, mock, None)
    console = get_console()
    cfg = ShieldConfig(
        input_injection=injection,
        input_pii=pii_in,
        input_guard=guard_in,
        output_guard=guard_out,
        output_leak=leak,
        output_pii=pii_out,
        canaries=list(canary or []),
    )

    async def go() -> Any:
        client = make_client(mock=use_mock)
        try:
            _banner(bool(getattr(client, "mock", False)))
            chosen = model or default_models()[0]
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ]
            async with Shield(client, cfg) as sh:
                return chosen, await sh.chat(chosen, messages)
        finally:
            if hasattr(client, "aclose"):
                await client.aclose()

    chosen, result = _run_async(go())
    table = Table(title=f"Shield checks ({chosen})")
    for col in ("Stage", "Check", "Action", "Result", "Applied", "Detail"):
        table.add_column(col)
    for c in result.checks:
        state = (
            "[dim]skipped[/dim]"
            if c.skipped
            else ("[red]triggered[/red]" if c.triggered else "[green]ok[/green]")
        )
        table.add_row(c.stage, c.name, c.action.value, state, c.applied, c.detail)
    console.print(table)
    if result.blocked:
        console.print(Panel(result.content, title="[red]BLOCKED[/red]", border_style="red"))
    else:
        console.print(Panel(result.content, title="reply", border_style="green"))
    if result.redactions:
        console.print(f"[yellow]{result.redactions} redaction(s) applied to the reply[/yellow]")


# --------------------------------------------------------------------------------- redact


@app.command()
def redact(
    source: Annotated[
        str, typer.Argument(metavar="FILE|-", help="File to redact, or '-' for stdin.")
    ],
    restore: Annotated[
        bool, typer.Option("--restore", help="Put the original values back (needs --map).")
    ] = False,
    map_path: Annotated[
        Path | None,
        typer.Option(
            "--map", help="Token mapping JSON: written when redacting, read with --restore."
        ),
    ] = None,
    style: Annotated[
        str, typer.Option("--style", help="token (reversible), label or mask.")
    ] = "token",
    kinds: Annotated[
        str | None, typer.Option("--kinds", help=f"Comma list of: {', '.join(PII_KINDS)}.")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write the result here.")] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print matches as JSON, not text.")
    ] = False,
) -> None:
    """Redact PII (emails, phones, cards, IBANs, IPs, API keys) from a file or stdin.

    With the default `token` style the output holds tokens such as [EMAIL_1]; save the mapping with
    --map and run `redact --restore --map FILE` later to undo it. The mapping file contains the
    original values: treat it as sensitive.
    """
    try:
        text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _fail(f"cannot read {source}: {exc}") from exc
    if restore:
        if map_path is None:
            raise _fail("--restore needs --map FILE (the mapping written when redacting)")
        try:
            mapping = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _fail(f"cannot read mapping {map_path}: {exc}") from exc
        if not isinstance(mapping, dict):
            raise _fail("mapping file must be a JSON object of token -> original")
        result_text = Redactor().restore(text, {str(k): str(v) for k, v in mapping.items()})
        _emit(result_text, out)
        return
    try:
        redactor = Redactor(style=style, kinds=_split(kinds) or None)  # type: ignore[arg-type]
        result = redactor.redact(text)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    if map_path is not None:
        if style != "token":
            raise _fail("--map only makes sense with --style token (the others are not reversible)")
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(
            json.dumps(redactor.mapping, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.chmod(map_path, 0o600)
    if as_json:
        payload = {
            "text": result.text,
            "counts": result.counts,
            "matches": [
                {"kind": m.kind, "start": m.start, "end": m.end, "token": t}
                for m, t in zip(result.matches, result.tokens, strict=True)
            ],
        }
        _emit(json.dumps(payload, indent=2, ensure_ascii=False), out)
    else:
        _emit(result.text, out)
    summary = ", ".join(f"{n} {k}" for k, n in sorted(result.counts.items())) or "nothing found"
    typer.echo(f"redacted: {summary}", err=True)


def _emit(text: str, out: Path | None) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
