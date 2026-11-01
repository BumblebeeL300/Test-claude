"""``nimlab forge`` command line interface."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

from nimlab.core import NIMClient, NIMConfig, NIMError, get_console

from .display import ForgeDisplay, print_report
from .forge import Forge
from .models import REPORT_JSON, ForgeConfig, ForgeReport

app = typer.Typer(
    help=(
        "Forge: an autonomous, test-driven coding agent. It plans (reasoning model), writes code "
        "and tests (coder model), runs pytest, debugs failures, has a second model review the "
        "diff, and writes a report to <workspace>/.forge/."
    ),
    no_args_is_help=True,
    add_completion=False,
)

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def _read_task(task: str | None, task_file: Path | None) -> str:
    if task_file is not None:
        try:
            return task_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise typer.BadParameter(f"cannot read --task-file: {exc}") from exc
    if task == "-":
        return sys.stdin.read().strip()
    return (task or "").strip()


def _exit_code(report: ForgeReport) -> int:
    if report.status == "error":
        return EXIT_USAGE if not report.mock else EXIT_FAILED
    if report.status == "failed" and not report.mock:
        return EXIT_FAILED
    return EXIT_OK


async def _run(
    task: str,
    workspace: Path,
    forge_config: ForgeConfig,
    *,
    mock: bool,
    quiet: bool,
    as_json: bool,
    show_diff: bool,
) -> int:
    console = get_console()
    config = NIMConfig.from_env(mock=True) if mock else NIMConfig.from_env()
    silent = quiet or as_json
    async with NIMClient(config) as client:
        forge = Forge(client, forge_config)
        if config.mock and not silent:
            console.print(
                "[yellow]MOCK MODE[/]: offline, placeholder model output. "
                "Set NVIDIA_API_KEY for real runs."
            )
        display = ForgeDisplay(console, quiet=silent)
        try:
            with display:
                report = await forge.run(task, workspace, on_event=display)
        except (ValueError, NIMError, OSError) as exc:
            console.print(f"[red]forge failed:[/] {exc}")
            return EXIT_USAGE
        if as_json:
            typer.echo(report.to_json())
        elif not quiet:
            print_report(console, report, show_diff=show_diff)
            console.print(client.usage.rich_table("NIM usage"))
            if not report.dry_run:
                console.print(
                    f"[dim]Report: {Path(report.workspace) / '.forge' / 'forge_report.md'}[/]"
                )
    return _exit_code(report)


@app.command("run")
def run_cmd(
    task: str | None = typer.Argument(
        None, help="What to build, in plain language. Use '-' to read it from stdin."
    ),
    workspace: Path = typer.Option(
        Path("."), "--workspace", "-w", help="Directory Forge works in (created if missing)."
    ),
    task_file: Path | None = typer.Option(
        None,
        "--task-file",
        "-f",
        help="Read the task description from a file (e.g. examples/forge/*.md).",
    ),
    max_iters: int = typer.Option(6, "--max-iters", min=0, help="Max debug/fix rounds."),
    planner: str | None = typer.Option(
        None, "--planner", help="Reasoning model for planning and root-cause analysis."
    ),
    coder: str | None = typer.Option(
        None, "--coder", help="Coding model that writes code and tests."
    ),
    reviewer: str | None = typer.Option(
        None, "--reviewer", help="Model that reviews the diff (default: a different family)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Explore and plan only; write nothing."),
    rollback_on_fail: bool = typer.Option(
        False, "--rollback-on-fail", help="Restore the workspace if the final tests do not pass."
    ),
    timeout: float = typer.Option(
        90.0, "--timeout", min=1.0, help="Per-command timeout in seconds."
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Force offline mock mode (no network, placeholder output)."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="No live display or report; only the exit code."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON on stdout."),
    show_diff: bool = typer.Option(False, "--diff", help="Also print the unified diff."),
) -> None:
    """Run the Forge pipeline on TASK inside --workspace."""
    text = _read_task(task, task_file)
    if not text:
        typer.echo("error: provide a TASK argument or --task-file", err=True)
        raise typer.Exit(EXIT_USAGE)
    cfg = ForgeConfig(
        max_iters=max_iters,
        dry_run=dry_run,
        rollback_on_fail=rollback_on_fail,
        planner=planner,
        coder=coder,
        reviewer=reviewer,
        command_timeout_s=timeout,
    )
    try:
        code = asyncio.run(
            _run(text, workspace, cfg, mock=mock, quiet=quiet, as_json=as_json, show_diff=show_diff)
        )
    except KeyboardInterrupt:
        typer.echo("interrupted", err=True)
        raise typer.Exit(130) from None
    raise typer.Exit(code)


@app.command("inspect")
def inspect_cmd(
    directory: Path = typer.Argument(
        Path("."), help="Workspace that contains a .forge/ directory."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the raw JSON report."),
    markdown: bool = typer.Option(False, "--markdown", "--md", help="Print the Markdown report."),
    show_diff: bool = typer.Option(False, "--diff", help="Include the unified diff."),
) -> None:
    """Show the last Forge report stored in DIRECTORY/.forge/."""
    console = get_console()
    try:
        report = ForgeReport.load(directory)
    except FileNotFoundError:
        console.print(
            f"[red]No Forge report in[/] {directory} [dim](expected .forge/{REPORT_JSON})[/]"
        )
        raise typer.Exit(EXIT_FAILED) from None
    except ValueError as exc:
        console.print(f"[red]Unreadable Forge report:[/] {exc}")
        raise typer.Exit(EXIT_FAILED) from None
    if as_json:
        typer.echo(report.to_json())
    elif markdown:
        typer.echo(report.to_markdown())
    else:
        console.print(f"[bold]Task:[/] {report.task.strip()}")
        print_report(console, report, show_diff=show_diff)


if __name__ == "__main__":  # pragma: no cover
    app()
