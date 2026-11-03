"""Root ``nimlab`` command line: built-ins (doctor, models, update, uninstall) plus
auto-discovered app plugins.

Plugin contract: ``nimlab.<name>.cli`` exposes ``app`` (a ``typer.Typer``) for each of
``council``, ``atlas``, ``forge``, ``arena``, ``prism``, ``sentinel``, ``loom``,
``distill`` and ``studio``. Plugin CLI modules must NOT import
``nimlab.cli``
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
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .core.client import NIMClient
from .core.config import NIMConfig, quiet_config
from .core.logging import configure_logging, get_console
from .core.registry import UNLISTED_ROLES, ModelRegistry, Role, default_registry

PLUGIN_NAMES: tuple[str, ...] = (
    "council",
    "atlas",
    "forge",
    "arena",
    "prism",
    "sentinel",
    "loom",
    "distill",
    "studio",
)

app = typer.Typer(
    name="nimlab",
    help=(
        "NIM Lab: council, atlas, forge, arena, prism, sentinel, loom and distill on NVIDIA NIM. "
        "Use --mock to run offline."
    ),
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
    ctx: typer.Context,
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
        # Scope the override to this invocation: put the environment back when the command ends,
        # so running the CLI in-process (tests, notebooks) cannot leak mock mode into later work.
        previous = os.environ.get("NIMLAB_MOCK")
        os.environ["NIMLAB_MOCK"] = "1"

        def _restore() -> None:
            if previous is None:
                os.environ.pop("NIMLAB_MOCK", None)
            else:
                os.environ["NIMLAB_MOCK"] = previous

        ctx.call_on_close(_restore)


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
        typer.Option(
            "--live",
            help="Query the endpoint's /models: show each role's chosen model, flag curated "
            "ids that are missing and suggest replacements.",
        ),
    ] = False,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh", help="Refetch /models now (ignore the 24 h cache); implies --live."
        ),
    ] = False,
) -> None:
    """List the curated NIM model catalog (and, with --live, what your endpoint serves)."""
    console = get_console()
    live = live or refresh
    config: NIMConfig | None = None
    if live:
        config = NIMConfig.from_env()
        registry = ModelRegistry(live=not config.mock, nim_config=config)
    else:
        registry = default_registry()
    entries = (
        registry.by_role(role)
        if role
        else sorted(registry.all(), key=lambda m: (m.deprecated, m.cost_tier, m.id))
    )

    live_ids: set[str] | None = None
    failed = False
    snapshot_note = ""
    if live and config is not None:
        if config.mock:
            client_config = config

            async def fetch() -> list[str]:
                async with NIMClient(client_config) as client:
                    return await client.list_models()

            live_ids = set(asyncio.run(fetch()))
        else:
            snap = registry.refresh_catalog(config=config) if refresh else registry.snapshot()
            if snap is None:
                err = registry._live_catalog().last_error or "no response"
                console.print(f"[red]Could not fetch live models:[/red] {escape(err)}")
                failed = True
            else:
                live_ids = set(snap.ids)
                age = int(snap.age_s() // 60)
                snapshot_note = (
                    f"{len(live_ids)} models listed by {snap.base_url} "
                    f"({snap.source}, {age} min old; cache: {registry._live_catalog().path})"
                )

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
        row.append(escape(("[deprecated] " if m.deprecated else "") + m.notes))
        table.add_row(*row)
    console.print(table)

    if live_ids is not None:
        if config is not None and config.mock:
            console.print(
                "[dim]mock mode: 'live' reflects the curated catalog, not the network.[/dim]"
            )
        elif snapshot_note:
            console.print(f"[dim]{escape(snapshot_note)}[/dim]")
        extra = sorted(live_ids - set(registry.ids()))
        if extra:
            shown = ", ".join(extra[:20]) + (" ..." if len(extra) > 20 else "")
            console.print(f"{len(extra)} live model(s) not in the catalog: {escape(shown)}")
        if config is not None and not config.mock:
            _print_role_picks(console, registry, live_ids, role)
    if failed:
        raise typer.Exit(code=1)


def _print_role_picks(
    console: Console, registry: ModelRegistry, live_ids: set[str], only: Role | None
) -> None:
    """Chosen model per role, plus curated ids missing from the live catalog with replacements."""
    picks = Table(title="Chosen model per role", title_justify="left")
    picks.add_column("Role")
    picks.add_column("Model", no_wrap=True)
    picks.add_column("Source")
    picks.add_column("Status", overflow="fold")
    stale = {s["role"]: s for s in registry.stale_defaults()}
    for r in Role:
        if only is not None and r != only:
            continue
        info = registry.describe(r)
        status = "[green]ok[/green]"
        pinned = info["source"] in ("env", "toml", "studio", "app", "request")
        if pinned:
            status = "pinned by you" + (
                "" if info["model"] in live_ids else " [red](not in the live list)[/red]"
            )
        elif r.value in stale:
            s = stale[r.value]
            status = (
                f"[yellow]default {escape(s['default'])} unavailable; auto-substituted[/yellow]"
            )
        elif r in UNLISTED_ROLES and info["model"] not in live_ids:
            status = "[dim]not listed by /models (retrieval API); checked at call time[/dim]"
        elif info["model"] not in live_ids:
            status = "[red]not in the live list[/red]"
        picks.add_row(r.value, info["model"], info["source"], status)
    console.print(picks)
    missing = [i for i in registry.missing_from(live_ids) if r_in_scope(registry, i, only)]
    if missing:
        console.print("[bold]Curated ids missing from the live catalog[/bold]")
        for mid in missing:
            repl = registry.replacement_for(mid)
            console.print(
                f"  [red]{escape(mid)}[/red]"
                + (f"  ->  suggested: [green]{escape(repl)}[/green]" if repl else "")
            )


def r_in_scope(registry: ModelRegistry, model_id: str, only: Role | None) -> bool:
    info = registry.find(model_id)
    if info is None or (only is not None and only not in info.roles):
        return False
    # the rerank API is not listed by /models: absence there is not a signal
    return not (info.roles <= UNLISTED_ROLES)


@app.command()
def update(
    check: Annotated[bool, typer.Option("--check", help="Only check; do not update.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Update without asking.")] = False,
) -> None:
    """Check for a newer NIM Lab and (for git installs) update it."""
    from .studio.updater import apply_update, check_for_updates

    console = get_console()
    st = check_for_updates()
    console.print(f"nimlab {escape(st.version)}: {escape(st.message)}")
    for line in st.commits:
        console.print(f"  [dim]{escape(line)}[/dim]")
    if st.instructions:
        console.print(escape(st.instructions))
    if st.state in ("offline", "error", "diverged") or (
        st.kind == "git" and st.dirty and st.behind
    ):
        raise typer.Exit(code=1)
    if check or not st.can_update:
        return
    if not yes and not typer.confirm("Pull the update and reinstall now?", default=True):
        return
    result = apply_update()
    for step in result["steps"]:
        mark = "[green]ok[/green]" if step["ok"] else "[red]failed[/red]"
        console.print(f"  {mark} {escape(step['step'])}")
        if not step["ok"] and step.get("output"):
            console.print(escape(step["output"]))
    console.print(escape(result["message"]))
    if not result["ok"]:
        raise typer.Exit(code=1)


@app.command()
def uninstall(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be removed; change nothing.")
    ] = False,
    remove_projects: Annotated[
        bool,
        typer.Option(
            "--remove-projects/--keep-projects",
            help="Also delete Forge projects and saved Loom pipelines (default: keep).",
        ),
    ] = False,
    remove_atlas: Annotated[
        bool, typer.Option("--remove-atlas/--keep-atlas", help="Also delete the Atlas database.")
    ] = False,
    remove_key: Annotated[
        bool, typer.Option("--remove-key/--keep-key", help="Delete the stored API key.")
    ] = True,
    remove_app: Annotated[
        bool,
        typer.Option(
            "--remove-app/--keep-app", help="Also uninstall the nimlab package / .app bundle."
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Remove NIM Lab's key, settings, history and caches (projects are kept unless asked)."""
    from .studio import uninstall as un
    from .studio.settings import StudioSettings

    console = get_console()
    opts = un.UninstallOptions(
        remove_key=remove_key,
        remove_data=True,
        remove_projects=remove_projects,
        remove_atlas=remove_atlas,
        remove_app=remove_app,
        dry_run=dry_run,
    )
    home = StudioSettings.from_env().home
    plan = un.plan(opts, home=home)
    console.print(f"NIM Lab folder: {escape(plan['home'])}   config: {escape(plan['config_dir'])}")
    for item in plan["items"]:
        mark = "[red]remove[/red]" if item["selected"] else "[green]keep  [/green]"
        console.print(
            f"  {mark} {escape(item['label'])}: {item['count']} item(s), "
            f"{un.human_size(item['bytes'])}"
        )
        for path in item["paths"][:6]:
            console.print(f"         [dim]{escape(path)}[/dim]")
    for line in plan["outside"]:
        console.print(f"  [yellow]note[/yellow] {escape(line)}")
    if remove_app:
        console.print(escape(plan["install"]["instructions"]))
    if dry_run:
        console.print("[dim]dry run: nothing was changed.[/dim]")
        return
    if not yes and typer.prompt("Type UNINSTALL to confirm", default="") != "UNINSTALL":
        console.print("Cancelled.")
        raise typer.Exit(code=1)
    result = un.execute(opts, home=home)
    for path in result["removed"]:
        console.print(f"  removed {escape(path)}")
    for line in result["refused"]:
        console.print(f"  [yellow]refused[/yellow] {escape(line)}")
    for line in result["errors"]:
        console.print(f"  [red]error[/red] {escape(line)}")
    if remove_app:
        info = un.detect_install()
        cmd = un.schedule_app_removal(info)
        console.print(
            "The application will be removed when this command exits."
            if cmd
            else escape(info.instructions)
        )
    if result["errors"]:
        raise typer.Exit(code=1)


PLUGINS_LOADED.extend(register_plugins(app))


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
