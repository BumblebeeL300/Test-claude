"""The :class:`Arena` facade: benchmark runs, battles, leaderboards, reports and routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nimlab.core import NIMConfig, Role, default_registry
from nimlab.core.registry import ModelRegistry

from .battle import BattleEngine
from .leaderboard import Leaderboard
from .llm import default_models, make_client
from .report import render_html
from .results import BattleRecord, RunResult, RunStore
from .router import SmartRouter
from .runner import ProgressFn, Runner, RunSettings
from .suites import Suite, list_suites, resolve_suites

__all__ = ["Arena"]


class Arena:
    """Benchmark NIM models, rank them, and route prompts to the best one.

    Example::

        async with Arena() as arena:                     # mock mode without NVIDIA_API_KEY
            run = await arena.run(["meta/llama-3.3-70b-instruct"], suites="all")
            print(arena.leaderboard(run).standings[0])
            decision = await arena.router().route("Prove that sqrt(2) is irrational")

    Pass ``client=`` to reuse an existing :class:`nimlab.core.NIMClient` (or a test double); the
    arena only closes clients it created itself.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        mock: bool | None = None,
        config: NIMConfig | None = None,
        registry: ModelRegistry | None = None,
        runs_dir: str | Path | None = None,
        judge_model: str | None = None,
        concurrency: int = 4,
        suites_dirs: list[str | Path] | None = None,
        persist: bool = True,
        settings: RunSettings | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self._owns_client = client is None
        self.client = client if client is not None else make_client(mock=mock, config=config)
        self.store = RunStore(runs_dir) if persist else None
        self.suites_dirs = suites_dirs or []
        self.settings = settings or RunSettings(concurrency=concurrency)
        if judge_model:
            self.settings.judge_model = judge_model
        self.judge_model = self.settings.judge_model or self.registry.pick(Role.CHAT)
        self.settings.judge_model = self.judge_model

    # ------------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> Arena:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and hasattr(self.client, "aclose"):
            await self.client.aclose()

    @property
    def mock(self) -> bool:
        return bool(getattr(self.client, "mock", False))

    # ------------------------------------------------------------------ suites
    def suites(self) -> dict[str, Suite]:
        """Bundled suites plus any found in ``suites_dirs``."""
        return list_suites(self.suites_dirs)

    def resolve_suites(self, spec: str | list[str] | None = "all", limit: int | None = None) -> list[Suite]:
        return resolve_suites(spec, extra_dirs=self.suites_dirs, limit=limit)

    # ------------------------------------------------------------------ runs
    def _runner(self, concurrency: int | None) -> Runner:
        settings = RunSettings(**{**vars(self.settings)})
        if concurrency:
            settings.concurrency = concurrency
        return Runner(self.client, settings=settings, store=self.store)

    def load_run(self, ref: str | Path = "latest") -> RunResult:
        return (self.store or RunStore()).load(ref)

    async def run(
        self,
        models: list[str] | None = None,
        suites: str | list[str] | list[Suite] | None = "all",
        *,
        concurrency: int | None = None,
        resume: str | Path | RunResult | None = None,
        limit: int | None = None,
        on_progress: ProgressFn | None = None,
    ) -> RunResult:
        """Evaluate ``models`` on ``suites`` concurrently and persist the run.

        ``suites`` is ``"all"``, a comma list / list of names or suite-file paths, or ready-made
        :class:`Suite` objects. ``resume`` continues an earlier run (id, path or object): only
        missing or errored items are re-queried.
        """
        if isinstance(suites, list) and suites and isinstance(suites[0], Suite):
            suite_list = [s.limited(limit) for s in suites]  # type: ignore[union-attr]
        else:
            suite_list = self.resolve_suites(suites, limit)  # type: ignore[arg-type]
        previous = self._as_run(resume)
        return await self._runner(concurrency).run(
            models or default_models(self.registry),
            suite_list,
            resume=previous,
            on_progress=on_progress,
        )

    def _as_run(self, ref: str | Path | RunResult | None) -> RunResult | None:
        if ref is None or isinstance(ref, RunResult):
            return ref
        return self.load_run(ref)

    # ------------------------------------------------------------------ battles
    async def battle(
        self,
        models: list[str],
        suite: str | Suite = "summarization",
        *,
        run: str | Path | RunResult | None = None,
        max_items: int | None = None,
        max_pairs: int | None = None,
        concurrency: int | None = None,
        on_progress: ProgressFn | None = None,
        on_battle: Any = None,
    ) -> RunResult:
        """Pairwise-judge every pair of ``models`` on ``suite`` (each pair in both slot orders).

        Missing answers are generated first (reusing ``run`` if given); the battle records are
        stored in the run file and feed the Elo ratings of :meth:`leaderboard`.
        """
        if len(set(models)) < 2:
            raise ValueError("a battle needs at least two different models")
        if isinstance(suite, Suite):
            the_suite = suite.limited(max_items)
        else:
            the_suite = self.resolve_suites(suite, max_items)[0]
        current = await self.run(
            models,
            [the_suite],
            concurrency=concurrency,
            resume=run,
            on_progress=on_progress,
        )
        engine = BattleEngine(
            self.client,
            self.judge_model,
            concurrency=concurrency or self.settings.concurrency,
            max_retries=self.settings.max_retries,
            backoff_s=self.settings.backoff_s,
        )
        records = await engine.run(
            current, the_suite, models, max_items=max_items, max_pairs=max_pairs, on_progress=on_battle
        )
        fresh = {(r.suite, r.item_id, frozenset((r.model_a, r.model_b))) for r in records}
        kept: list[BattleRecord] = [
            b
            for b in current.battles
            if (b.suite, b.item_id, frozenset((b.model_a, b.model_b))) not in fresh
        ]
        order = {it.id: i for i, it in enumerate(the_suite.items)}
        records.sort(key=lambda r: (order.get(r.item_id, 0), r.model_a, r.model_b))
        current.battles = [*kept, *records]
        current.judge_model = self.judge_model
        current.config["battle_failures"] = engine.failures[:20]
        if self.store:
            self.store.save(current)
        return current

    # ------------------------------------------------------------------ reporting
    def leaderboard(
        self, run: str | Path | RunResult | None = None, *, n_boot: int = 200, seed: int = 0
    ) -> Leaderboard:
        """Leaderboard of ``run`` (default: the latest saved run)."""
        result = self._as_run(run if run is not None else "latest")
        assert result is not None
        return Leaderboard.from_run(result, registry=self.registry, n_boot=n_boot, seed=seed)

    def report(
        self,
        run: str | Path | RunResult | None = None,
        *,
        html_out: str | Path | None = None,
        title: str | None = None,
    ) -> str:
        """Render the HTML leaderboard (optionally writing it to ``html_out``) and return it."""
        page = render_html(self.leaderboard(run), title=title)
        if html_out is not None:
            out = Path(html_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(page, encoding="utf-8")
        return page

    # ------------------------------------------------------------------ routing
    def router(self, **kw: Any) -> SmartRouter:
        """A :class:`SmartRouter` backed by this arena's client, registry and run store."""
        kw.setdefault("client", self.client)
        kw.setdefault("registry", self.registry)
        if self.store is not None:
            kw.setdefault("runs_dir", self.store.root)
        return SmartRouter(**kw)
