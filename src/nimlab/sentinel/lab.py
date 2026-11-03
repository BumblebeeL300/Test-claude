"""The :class:`Sentinel` facade: scan models, build scorecards and reports, create Shields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nimlab.core import NIMConfig
from nimlab.core.registry import ModelRegistry, default_registry

from .llm import default_models, is_mock, make_client
from .models import ProbeSuite, SentinelRun
from .report import render_html, write_html
from .runner import ResultFn, Runner, ScanSettings, StartFn
from .shield import Shield, ShieldConfig
from .store import RunStore
from .suites import list_suites, resolve_suites

__all__ = ["Sentinel"]


class Sentinel:
    """Safety and robustness lab for NIM models.

    Example::

        async with Sentinel() as lab:                     # simulated offline without a key
            run = await lab.scan(["nvidia/nemotron-3-super-120b-a12b"], suites="all")
            print(run.scorecards[0].grade)
            lab.write_html(run, "scorecard.html")

    Pass ``client=`` to reuse an existing client (it is not closed on exit).
    """

    def __init__(
        self,
        client: Any = None,
        *,
        mock: bool | None = None,
        config: NIMConfig | None = None,
        registry: ModelRegistry | None = None,
        runs_dir: str | Path | None = None,
        settings: ScanSettings | None = None,
        suites_dirs: list[str | Path] | None = None,
        persist: bool = True,
    ) -> None:
        self.registry = registry or default_registry()
        self._owns_client = client is None
        self.client = client if client is not None else make_client(mock=mock, config=config)
        self.store = RunStore(runs_dir) if persist else None
        self.settings = settings or ScanSettings()
        self.suites_dirs = suites_dirs or []

    async def __aenter__(self) -> Sentinel:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and hasattr(self.client, "aclose"):
            await self.client.aclose()

    @property
    def mock(self) -> bool:
        return is_mock(self.client)

    def suites(self) -> dict[str, ProbeSuite]:
        return list_suites(self.suites_dirs)

    def resolve_suites(
        self, spec: str | list[str] | None = "all", limit: int | None = None
    ) -> list[ProbeSuite]:
        return resolve_suites(spec, extra_dirs=self.suites_dirs, limit=limit)

    def default_models(self) -> list[str]:
        return default_models(self.registry)

    async def scan(
        self,
        models: list[str] | None = None,
        suites: str | list[str] | list[ProbeSuite] | None = "all",
        *,
        limit: int | None = None,
        on_start: StartFn | None = None,
        on_result: ResultFn | None = None,
    ) -> SentinelRun:
        """Probe ``models`` with ``suites`` (names, ``"all"`` or suite objects) and score them."""
        if isinstance(suites, list) and suites and all(isinstance(s, ProbeSuite) for s in suites):
            resolved = [s for s in suites if isinstance(s, ProbeSuite)]
        else:
            names = (
                [s for s in suites if isinstance(s, str)] if isinstance(suites, list) else suites
            )
            resolved = self.resolve_suites(names, limit)
        runner = Runner(self.client, settings=self.settings, store=self.store)
        return await runner.run(
            models or self.default_models(), resolved, on_start=on_start, on_result=on_result
        )

    def load_run(self, ref: str | Path = "latest") -> SentinelRun:
        return (self.store or RunStore()).load(ref)

    def shield(self, config: ShieldConfig | None = None, **overrides: Any) -> Shield:
        """A :class:`Shield` sharing this lab's client (closing it leaves the client open)."""
        return Shield(self.client, config, **overrides)

    @staticmethod
    def html(run: SentinelRun, title: str | None = None) -> str:
        return render_html(run, title=title)

    @staticmethod
    def write_html(run: SentinelRun, path: str | Path, title: str | None = None) -> Path:
        return write_html(run, path, title=title)
