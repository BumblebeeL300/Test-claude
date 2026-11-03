"""``nimlab studio``: open NIM Lab Studio (native window, or your browser)."""

from __future__ import annotations

from typing import Annotated

import typer

app = typer.Typer(
    name="studio",
    help="NIM Lab Studio: the desktop app (native window with pywebview, otherwise your browser).",
    invoke_without_command=True,
    add_completion=False,
)


@app.callback()
def studio(
    ctx: typer.Context,
    browser: Annotated[
        bool, typer.Option("--browser", help="Open the default browser instead of a native window.")
    ] = False,
    port: Annotated[
        int,
        typer.Option("--port", min=0, max=65535, help="Port on 127.0.0.1 (0 = pick a free one)."),
    ] = 0,
    dev: Annotated[
        bool, typer.Option("--dev", help="Developer mode: web inspector and server logging.")
    ] = False,
) -> None:
    """Start the Studio server on 127.0.0.1 and open it."""
    if ctx.invoked_subcommand is not None:
        return
    from .desktop import launch

    raise typer.Exit(code=launch(browser=browser, port=port, dev=dev))
