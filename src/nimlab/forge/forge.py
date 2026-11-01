"""Public entry point: :class:`Forge`."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nimlab.core import ModelRegistry

from .context import RunContext
from .events import EventBus, EventCallback
from .models import ForgeConfig, ForgeReport
from .orchestrator import Orchestrator
from .progress import PhaseTimer, StallDetector, UsageProbe
from .roles import resolve_models
from .sandbox import Sandbox, SandboxLimits
from .tools import WorkspaceTools

__all__ = ["Forge"]


def _expand(workspace: str | Path) -> tuple[Path, Path]:
    """``(path, display_path)``: the expanded path and its absolute form if it exists."""
    path = Path(workspace).expanduser()
    return path, (path.resolve() if path.exists() else path)


@contextmanager
def _sandbox_root(workspace: Path, *, dry_run: bool) -> Iterator[Path]:
    """The directory the sandbox is rooted in.

    A real run creates a missing workspace. A dry run never writes, so for a workspace that does
    not exist yet it plans against an empty throw-away directory instead.
    """
    if workspace.is_dir():
        yield workspace
    elif workspace.exists():
        raise NotADirectoryError(f"workspace is not a directory: {workspace}")
    elif dry_run:
        with tempfile.TemporaryDirectory(prefix="forge-dryrun-") as tmp:
            yield Path(tmp)
    else:
        workspace.mkdir(parents=True, exist_ok=True)
        yield workspace


class Forge:
    """An autonomous, test-driven coding agent over NVIDIA NIM.

    Example::

        async with NIMClient() as client:
            report = await Forge(client).run("Write fizzbuzz with tests", "./work")
            print(report.status, report.summary)

    ``client`` only needs ``chat``, ``chat_json`` and (optionally) a ``usage`` tracker, so any
    duck-typed stand-in works in tests. The client's lifecycle is the caller's responsibility.
    """

    def __init__(
        self,
        client: Any,
        config: ForgeConfig | None = None,
        *,
        registry: ModelRegistry | None = None,
        on_event: EventCallback | None = None,
        **overrides: Any,
    ) -> None:
        self.client = client
        self.config = (config or ForgeConfig()).model_copy(update=overrides)
        self.registry = registry
        self._listeners: list[EventCallback] = [on_event] if on_event else []

    def subscribe(self, callback: EventCallback) -> None:
        """Receive every :class:`~nimlab.forge.events.ForgeEvent` of every future run."""
        self._listeners.append(callback)

    async def run(
        self,
        task: str,
        workspace: str | Path,
        *,
        max_iters: int | None = None,
        planner: str | None = None,
        coder: str | None = None,
        reviewer: str | None = None,
        dry_run: bool | None = None,
        rollback_on_fail: bool | None = None,
        on_event: EventCallback | None = None,
    ) -> ForgeReport:
        """Plan, implement, verify, review and report on ``task`` inside ``workspace``.

        Keyword arguments override the instance's :class:`ForgeConfig` for this run. Ordinary
        failures (tests that never pass, flaky model output, a revoked API key) do not raise: they
        come back as a report whose ``status`` is ``"failed"`` or ``"error"``.

        Raises:
            ValueError: ``task`` is empty.
            OSError: the workspace exists but is not a directory, or cannot be created.
        """
        if not task or not task.strip():
            raise ValueError("task must not be empty")
        overrides = {
            k: v
            for k, v in {
                "max_iters": max_iters,
                "planner": planner,
                "coder": coder,
                "reviewer": reviewer,
                "dry_run": dry_run,
                "rollback_on_fail": rollback_on_fail,
            }.items()
            if v is not None
        }
        cfg = self.config.model_copy(update=overrides)
        requested, display_path = _expand(workspace)

        bus = EventBus()
        for listener in (*self._listeners, *([on_event] if on_event else [])):
            bus.subscribe(listener)
        models = resolve_models(cfg.planner, cfg.coder, cfg.reviewer, self.registry)
        probe = UsageProbe(self.client)

        with _sandbox_root(requested, dry_run=cfg.dry_run) as root:
            sandbox = Sandbox(
                root,
                SandboxLimits(
                    command_timeout_s=cfg.command_timeout_s, max_output_bytes=cfg.max_output_bytes
                ),
                read_only=cfg.dry_run,
            )
            tools = WorkspaceTools(sandbox, bus)
            ctx = RunContext(
                task=task.strip(),
                workspace=display_path,
                config=cfg,
                models=models,
                client=self.client,
                sandbox=sandbox,
                tools=tools,
                bus=bus,
                timer=PhaseTimer(probe),
                probe=probe,
                detector=StallDetector(threshold=cfg.stall_threshold),
                mock=bool(getattr(self.client, "mock", False)),
                started_at=time.time(),
                started_monotonic=time.monotonic(),
            )
            bus.emit(
                "run_start",
                "init",
                task.strip().splitlines()[0][:120],
                workspace=str(ctx.workspace),
                dry_run=cfg.dry_run,
                mock=ctx.mock,
                models=models.model_dump(),
            )
            report = await Orchestrator(ctx).run()
            bus.emit("run_end", "report", report.status, status=report.status)
        return report
