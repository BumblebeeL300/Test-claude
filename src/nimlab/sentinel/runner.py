"""The scan runner: models x suites, concurrent, with retries, judging and persistence."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nimlab.core import NIMAuthError

from .judges import Judge, JudgeContext, build_judges
from .llm import guard_model as default_guard_model
from .llm import is_mock, is_simulated, is_transient
from .models import JudgeVerdict, Probe, ProbeResult, ProbeSuite, SentinelRun
from .scoring import build_scorecards, classify
from .shield import Shield, ShieldConfig
from .store import RunStore, new_run_id, utcnow

__all__ = ["Runner", "ScanSettings"]

MAX_STORED_CHARS = 4000
StartFn = Callable[[str, ProbeSuite, Probe], None]
ResultFn = Callable[[ProbeResult, int, int], None]


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass
class ScanSettings:
    concurrency: int = 6
    #: Extra attempts after a transient failure (rate limit, timeout, 5xx).
    retries: int = 2
    backoff_s: float = 0.5
    temperature: float = 0.0
    max_tokens: int = 512
    #: Run the NemoGuard judge. ``None`` = automatic (real clients and the simulator).
    guard: bool | None = None
    guard_model: str | None = None
    #: Model for the optional LLM refusal judge (heuristics only when ``None``).
    llm_judge_model: str | None = None
    #: Route every call through a :class:`Shield` with this config (measures its effectiveness).
    shield: ShieldConfig | None = None
    judges: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Runner:
    """Run probe suites against models and judge every reply.

    ``on_start(model, suite, probe)`` fires when a probe is dispatched and
    ``on_result(result, done, total)`` when it has been judged.
    """

    def __init__(
        self,
        client: Any,
        *,
        settings: ScanSettings | None = None,
        store: RunStore | None = None,
        judges: list[Judge] | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or ScanSettings()
        self.store = store
        use_guard = (
            self.settings.guard
            if self.settings.guard is not None
            else (is_simulated(client) or not is_mock(client))
        )
        self.guard_model = (
            (self.settings.guard_model or default_guard_model()) if use_guard else None
        )
        ctx = JudgeContext(client, self.guard_model, self.settings.llm_judge_model)
        self.judges = judges if judges is not None else build_judges(ctx, self.settings.judges)
        self._shield: Shield | None = None
        if self.settings.shield is not None:
            cfg = self.settings.shield
            if cfg.guard_model is None and self.guard_model:
                cfg = cfg.model_copy(update={"guard_model": self.guard_model})
            self._shield = Shield(client, cfg)

    # ------------------------------------------------------------------ one probe
    async def _call(self, model: str, probe: Probe) -> tuple[str, bool, float]:
        """``(reply text, blocked by shield, latency)`` with retries on transient errors."""
        s = self.settings
        kw = {"temperature": s.temperature, "max_tokens": s.max_tokens}
        t0 = time.perf_counter()
        if self._shield is not None:
            res = await self._shield.chat(model, probe.messages(), **kw)
            return res.content, res.blocked, res.latency_s or time.perf_counter() - t0
        res = await self.client.chat(model, probe.messages(), **kw)
        return res.content, False, getattr(res, "latency_s", 0.0) or time.perf_counter() - t0

    async def run_probe(self, model: str, suite: ProbeSuite, probe: Probe) -> ProbeResult:
        s = self.settings
        out = ProbeResult(
            model=model,
            suite=suite.name,
            category=suite.category,
            probe_id=probe.id,
            title=probe.title,
            benign=probe.benign,
            shielded=self._shield is not None,
        )
        blocked = False
        for attempt in range(1, s.retries + 2):
            out.attempts = attempt
            try:
                out.response, blocked, out.latency_s = await self._call(model, probe)
                out.error = None
                break
            except NIMAuthError:
                raise
            except Exception as exc:
                out.error = f"{type(exc).__name__}: {exc}"[:300]
                if attempt > s.retries or not is_transient(exc):
                    break
                await _sleep(s.backoff_s * 2 ** (attempt - 1))
        if out.error is not None:
            return out

        verdicts: dict[str, JudgeVerdict] = {}
        if blocked:
            out.outcome = "false_refusal" if probe.benign else "blocked"
            out.false_refusal = probe.benign
            out.verdicts = [JudgeVerdict(judge="shield", flagged=True, detail="blocked by Shield")]
        else:
            for judge in self.judges:
                if not judge.applies(probe):
                    continue
                try:
                    verdicts[judge.name] = await judge.judge(probe, out.response)
                except NIMAuthError:
                    raise
                except Exception as exc:
                    verdicts[judge.name] = JudgeVerdict(
                        judge=judge.name, skipped=True, detail=f"judge failed: {exc}"[:200]
                    )
            out.verdicts = list(verdicts.values())
            out.outcome, out.breached, out.false_refusal, out.unsafe = classify(
                suite.category, probe.benign, verdicts
            )
        out.response = out.response[:MAX_STORED_CHARS]
        return out

    # ------------------------------------------------------------------ whole scan
    async def run(
        self,
        models: list[str],
        suites: list[ProbeSuite],
        *,
        run_id: str | None = None,
        on_start: StartFn | None = None,
        on_result: ResultFn | None = None,
    ) -> SentinelRun:
        if not models:
            raise ValueError("no models to scan")
        if not suites:
            raise ValueError("no suites to run")
        run = SentinelRun(
            run_id=run_id or new_run_id(),
            created_at=utcnow().isoformat(timespec="seconds"),
            models=list(models),
            suites=[s.name for s in suites],
            simulated=is_mock(self.client),
            shield=self._shield is not None,
            config={
                "concurrency": self.settings.concurrency,
                "retries": self.settings.retries,
                "guard_model": self.guard_model,
                "judges": [j.name for j in self.judges],
                "llm_judge_model": self.settings.llm_judge_model,
                "shield": self.settings.shield.model_dump(mode="json")
                if self.settings.shield
                else None,
            },
        )
        # probe-major order so a live view lights up all models of one probe together
        jobs = [(m, s, p) for s in suites for p in s.probes for m in models]
        total = len(jobs)
        sem = asyncio.Semaphore(max(1, self.settings.concurrency))
        done = 0

        async def one(model: str, suite: ProbeSuite, probe: Probe) -> None:
            nonlocal done
            async with sem:
                if on_start:
                    on_start(model, suite, probe)
                result = await self.run_probe(model, suite, probe)
            run.results.append(result)
            done += 1
            if on_result:
                on_result(result, done, total)

        tasks = [asyncio.ensure_future(one(*job)) for job in jobs]
        try:
            await asyncio.gather(*tasks)
            run.status = "complete"
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            run.status = "partial"
            raise
        finally:
            order = {(m, s.name, p.id): i for i, (m, s, p) in enumerate(jobs)}
            run.results.sort(key=lambda r: order.get((r.model, r.suite, r.probe_id), 0))
            run.scorecards = build_scorecards(run.models, run.results)
            if self.store is not None:
                self.store.save(run)
        return run
