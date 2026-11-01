"""Root ``nimlab`` command line: built-ins (doctor, models) plus auto-discovered app plugins.

Plugin contract: ``nimlab.<name>.cli`` exposes ``app`` (a ``typer.Typer``) for each of
``council``, ``atlas``, ``forge``, ``arena``. Plugin CLI modules must NOT import ``nimlab.cli``
(it imports them while it is being initialised).
"""

from __future__ import annotations

import asyncio
import importlib
import os
import platform
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Annotated, Optional

import httpx
import typer
from rich.markup import escape
from rich.table import Table

from . import __version__
from .core.client import NIMClient
from .core.config import NIMConfig, quiet_config
from .core.errors import NIMError
from .core.logging import configure_logging, get_console
from .core.registry import Role, default_registry

PLUGIN_NAMES: tuple[str, ...] = ("council", "atlas", "forge", "arena")

app = typer.Typer(
    name="nimlab",
    help="NIM Lab: council, atlas, forge and arena on NVIDIA NIM. Use --mock to run offline.",
    no_args_is_help=True,
    add_completion=False,
)

PLUGINS_LOADED: list[str] = []


def register_plugins(
    target: typer.Typer,
    names: Sequence[str] = PLUGIN_NAMES,
    *,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> list[str]:
    """Attach ``nimlab.<name>.cli.app`` for every plugin that exists; returns loaded names.

    A missing ``nimlab.<name>`` / ``nimlab.<name>.cli`` module is skipped silently; any other
    import failure (including a missing third-party dependency *inside* an existing plugin) is
    re-raised so real bugs are not hidden.
    """
    loaded: list[str] = []
    for name in names:
        module_name = f"nimlab.{name}.cli"
        try:
            module = importer(module_name)
        except ModuleNotFoundError as exc:
            if exc.name in (f"nimlab.{name}", module_name):
                continue
            raise
        sub = getattr(module, "app", None)
        if not isinstance(sub, typer.Typer):
            continue
        if not any(g.name == name for g in target.registered_groups):
            target.add_typer(sub, name=name)
        loaded.append(name)
    return loaded


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"nimlab {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    mock: Annotated[
        bool, typer.Option("--mock", help="Use the offline mock backend (sets NIMLAB_MOCK=1).")
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    configure_logging()
    if mock:
        os.environ["NIMLAB_MOCK"] = "1"


def probe_connectivity(config: NIMConfig, timeout: float = 8.0) -> tuple[bool, str]:
    """One quick GET {base_url}/models (no retries). Returns (ok, human-readable detail)."""
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    try:
        resp = httpx.get(f"{config.base_url}/models", headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        return False, f"unreachable ({type(exc).__name__}: {str(exc)[:100]})"
    if resp.status_code == 200:
        try:
            count = len(resp.json()["data"])
        except (ValueError, KeyError, TypeError):
            return True, "reachable (HTTP 200)"
        return True, f"reachable, {count} models listed"
    if resp.status_code in (401, 403):
        return False, f"reachable but authentication failed (HTTP {resp.status_code})"
    return False, f"HTTP {resp.status_code}"


@app.command()
def doctor(
    offline: Annotated[
        bool, typer.Option("--offline", help="Skip the network connectivity check.")
    ] = False,
) -> None:
    """Show configuration, mock mode, API key status and connectivity."""
    console = get_console()
    with quiet_config():
        config = NIMConfig.from_env()

    if config.mock:
        reason = {
            "no_api_key": "no NVIDIA_API_KEY set",
            "explicit": "NIMLAB_MOCK / --mock",
        }.get(config.mock_reason or "", "enabled")
        mock_text = f"[yellow]ON[/yellow] ({reason}): offline deterministic backend"
    else:
        mock_text = "[green]off[/green]: real NIM calls"

    table = Table(title="nimlab doctor", title_justify="left", show_header=False)
    table.add_column("Check", style="bold")
    table.add_column("Value", overflow="fold")
    table.add_row("python", f"{platform.python_version()} ({platform.python_implementation()})")
    table.add_row("nimlab", __version__)
    table.add_row(
        "NVIDIA_API_KEY", "[green]set[/green]" if config.api_key else "[red]not set[/red]"
    )
    table.add_row("mock mode", mock_text)
    table.add_row("base_url", config.base_url)
    table.add_row("rerank_base_url", config.rerank_base_url)
    table.add_row(
        "limits",
        f"timeout={config.timeout:g}s retries={config.max_retries} "
        f"concurrency={config.max_concurrency}",
    )
    loaded = ", ".join(PLUGINS_LOADED) or "none"
    missing = ", ".join(n for n in PLUGIN_NAMES if n not in PLUGINS_LOADED) or "none"
    table.add_row("apps loaded", loaded)
    table.add_row("apps not installed", missing)

    healthy = True
    if offline:
        table.add_row("connectivity", "[dim]skipped (--offline)[/dim]")
    else:
        ok, detail = probe_connectivity(config)
        if ok:
            table.add_row("connectivity", f"[green]OK[/green]: {escape(detail)}")
        elif config.mock:
            table.add_row("connectivity", f"[yellow]unavailable[/yellow]: {escape(detail)}")
        else:
            healthy = False
            table.add_row("connectivity", f"[red]FAILED[/red]: {escape(detail)}")
    console.print(table)
    if not healthy:
        raise typer.Exit(code=1)


@app.command()
def models(
    role: Annotated[
        Optional[Role],  # noqa: UP045
        typer.Option("--role", "-r", case_sensitive=False, help="Only models with this role."),
    ] = None,
    live: Annotated[
        bool,
        typer.Option("--live", help="Query the endpoint's /models and mark which are available."),
    ] = False,
) -> None:
    """List the curated NIM model catalog."""
    console = get_console()
    registry = default_registry()
    entries = (
        registry.by_role(role)
        if role
        else sorted(registry.all(), key=lambda m: (m.cost_tier, m.id))
    )

    live_ids: set[str] | None = None
    failed = False
    config: NIMConfig | None = None
    if live:
        config = NIMConfig.from_env()
        client_config = config

        async def fetch() -> list[str]:
            async with NIMClient(client_config) as client:
                return await client.list_models()

        try:
            live_ids = set(asyncio.run(fetch()))
        except NIMError as exc:
            console.print(f"[red]Could not fetch live models:[/red] {escape(str(exc))}")
            failed = True

    defaults: dict[str, list[str]] = {}
    for r in Role:
        try:
            defaults.setdefault(registry.pick(r), []).append(r.value)
        except KeyError:
            continue

    table = Table(title="NIM models", title_justify="left")
    table.add_column("Model", no_wrap=True)
    table.add_column("Roles")
    table.add_column("Context", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Default for")
    if live_ids is not None:
        table.add_column("Live", justify="center")
    table.add_column("Notes", overflow="fold")
    for m in entries:
        row = [
            m.id,
            ", ".join(sorted(r.value for r in m.roles)),
            f"{m.context_window:,}",
            str(m.cost_tier),
            ", ".join(defaults.get(m.id, [])),
        ]
        if live_ids is not None:
            row.append("[green]yes[/green]" if m.id in live_ids else "[red]no[/red]")
        row.append(escape(m.notes))
        table.add_row(*row)
    console.print(table)

    if live_ids is not None:
        if config is not None and config.mock:
            console.print(
                "[dim]mock mode: 'live' reflects the curated catalog, not the network.[/dim]"
            )
        extra = sorted(live_ids - set(registry.ids()))
        if extra:
            shown = ", ".join(extra[:20]) + (" ..." if len(extra) > 20 else "")
            console.print(f"{len(extra)} live model(s) not in the catalog: {escape(shown)}")
    if failed:
        raise typer.Exit(code=1)


PLUGINS_LOADED.extend(register_plugins(app))


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
