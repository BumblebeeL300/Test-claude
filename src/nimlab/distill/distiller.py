"""The Distill pipeline: seed -> generate -> score -> filter -> dedup -> decontaminate -> finalize.

Every stage is idempotent over the sample table in ``samples.jsonl``, so a crashed or interrupted
run resumes exactly where it stopped, and a resumed run produces the same dataset as an
uninterrupted one (the simulator is a pure function of its inputs).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import ValidationError

from nimlab import __version__
from nimlab.atlas.chunking import chunk_document
from nimlab.atlas.loaders import discover_files, load_file
from nimlab.atlas.models import AtlasError
from nimlab.core import NIMAuthError, NIMClient, NIMError, Role, default_registry
from nimlab.core.registry import ModelRegistry

from .card import render_card
from .corpus import BUILTIN_DOCS
from .costs import Tally, estimate_cost
from .decontam import EvalIndex
from .dedup import cluster_members, exact_duplicates, greedy_clusters
from .export import FORMATS, ExportError, from_json, read_texts, sample_to_record, write_jsonl
from .filters import apply_filters
from .gateway import Gateway
from .models import STAGES, ExportFormat, RunMeta, RunOptions, Sample, Seed, utc_now
from .prompts import (
    InstructionList,
    QAPair,
    QuestionList,
    SubtopicList,
    TopicList,
    answer_messages,
    hard_questions_messages,
    instructions_messages,
    qa_messages,
    subtopics_messages,
    topics_messages,
)
from .quality import QualityReport, build_report
from .recipes import Recipe, get_recipe
from .report import write_report_html
from .scoring import Scorer
from .sim import SimClient
from .store import RunStore, new_run_id
from .textutil import normalize

__all__ = ["DistillError", "DistillResult", "Distiller"]

EventSink = Callable[[str, dict[str, Any]], None]
PREVIEW = 200


class DistillError(RuntimeError):
    """A run cannot start or continue (bad input, no usable seeds, wrong mode on resume...)."""


def _clip(text: str, n: int = PREVIEW) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: n - 1].rstrip() + "…"


def comparable_score(s: Sample) -> float:
    """Priority used to pick the best of a group (judge 0-10 is mapped under the reward scale)."""
    if s.score is None:
        return -1e9
    return s.score if s.score_source == "reward" else s.score - 10.0


@dataclass
class DistillResult:
    """What a finished (or stopped) run hands back."""

    meta: RunMeta
    report: QualityReport
    samples: list[Sample]
    directory: Path
    files: list[Path] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.meta.id

    @property
    def kept(self) -> list[Sample]:
        return [s for s in self.samples if s.alive]

    @property
    def dropped(self) -> list[Sample]:
        return [s for s in self.samples if not s.alive]

    def summary(self) -> dict[str, Any]:
        """A JSON-friendly digest (the Studio ``run.done`` payload)."""
        r = self.report
        return {
            "run_id": self.meta.id,
            "recipe": self.meta.recipe.name,
            "kind": self.meta.recipe.kind,
            "simulated": self.meta.simulated,
            "status": self.meta.status,
            "format": self.meta.format,
            "n_requested": r.n_requested,
            "n_final": r.n_final,
            "funnel": [f.model_dump(by_alias=True) for f in r.funnel],
            "drop_reasons": r.drop_reasons,
            "histograms": {k: v.model_dump() for k, v in r.histograms.items()},
            "diversity": r.diversity.model_dump(),
            "models": [m.model_dump() for m in r.models],
            "usage": r.usage,
            "cost_usd": self.meta.cost_usd,
            "files": [str(p) for p in self.files],
        }


class _Run:
    """Mutable state of one executing run."""

    def __init__(self, meta: RunMeta, gateway: Gateway, tally: Tally, scorer: Scorer) -> None:
        self.meta = meta
        self.gw = gateway
        self.tally = tally
        self.scorer = scorer
        self.seeds: list[Seed] = []
        self.samples: dict[int, Sample] = {}
        self.eval_texts: list[str] = []
        self.vectors: dict[str, list[float]] = {}

    def table(self) -> list[Sample]:
        return [self.samples[i] for i in sorted(self.samples)]


class Distiller:
    """Build synthetic datasets from NIM models.

    ``Distiller()`` uses a real client when ``NVIDIA_API_KEY`` is set and the deterministic
    simulator otherwise; ``Distiller(mock=True)`` always simulates. Pass ``client=`` to share a
    client (Studio does). ``on_event(type, data)`` receives pipeline events.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        runs_dir: str | Path | None = None,
        registry: ModelRegistry | None = None,
        mock: bool | None = None,
        on_event: EventSink | None = None,
    ) -> None:
        self.store = RunStore(runs_dir)
        self.registry = registry or default_registry()
        self.on_event = on_event
        self._force_mock = bool(mock)
        self._base = client
        self._owns_base = False

    # ------------------------------------------------------------------ plumbing
    def emit(self, type_: str, data: dict[str, Any] | None = None) -> None:
        if self.on_event is not None:
            self.on_event(type_, data or {})

    async def aclose(self) -> None:
        if self._owns_base and self._base is not None:
            await self._base.aclose()
            self._base = None

    async def __aenter__(self) -> Distiller:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _client_for(
        self, opts: RunOptions, *, require_simulated: bool | None = None
    ) -> tuple[Any, bool]:
        """``(client, simulated)`` for a run."""
        if require_simulated is False and self._force_mock:
            raise DistillError("this run used real models; it cannot be resumed with mock=True")
        if self._force_mock or require_simulated:
            base_usage = getattr(self._base, "usage", None)
            return SimClient(
                seed=opts.seed, delay=opts.sim_delay, registry=self.registry, usage=base_usage
            ), True
        if self._base is None:
            self._base = NIMClient()
            self._owns_base = True
        if getattr(self._base, "simulated", False):
            return self._base, True
        if getattr(self._base, "mock", False):
            if require_simulated is False:
                raise DistillError(
                    "this run used real models; set NVIDIA_API_KEY to resume it (no key found)"
                )
            usage = getattr(self._base, "usage", None)
            return SimClient(
                seed=opts.seed, delay=opts.sim_delay, registry=self.registry, usage=usage
            ), True
        return self._base, False

    # ------------------------------------------------------------------ public API
    def plan(
        self,
        recipe: str | Path | Recipe,
        *,
        n: int = 200,
        models: list[str] | None = None,
        fmt: str | None = None,
        options: RunOptions | dict[str, Any] | None = None,
    ) -> tuple[Recipe, RunOptions, list[str], str]:
        """Validate run settings without side effects: ``(recipe, options, models, format)``."""
        if n < 1:
            raise DistillError("n must be at least 1")
        try:
            opts = (
                options
                if isinstance(options, RunOptions)
                else RunOptions.model_validate(options or {})
            )
        except ValidationError as exc:
            raise DistillError(f"invalid options: {exc.errors()[0]['msg']}") from exc
        try:
            rec = self._with_overrides(get_recipe(recipe), opts)
        except ValidationError as exc:
            raise DistillError(f"invalid option override: {exc.errors()[0]['msg']}") from exc
        fmt = fmt or rec.default_format
        if fmt not in FORMATS:
            raise DistillError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")
        if fmt == "dpo" and rec.kind != "preference_pairs":
            raise DistillError("the dpo format needs the preference_pairs recipe (chosen/rejected)")
        return rec, opts, self._panel(rec, models), fmt

    async def run(
        self,
        recipe: str | Path | Recipe,
        *,
        n: int = 200,
        models: list[str] | None = None,
        docs: str | Path | None = None,
        eval_set: str | Path | None = None,
        out: str | Path | None = None,
        fmt: str | None = None,
        options: RunOptions | dict[str, Any] | None = None,
        run_id: str | None = None,
        stop_after: str | None = None,
    ) -> DistillResult:
        """Create a run and execute it. ``n`` is the number of candidate samples to generate."""
        rec, opts, panel, fmt = self.plan(recipe, n=n, models=models, fmt=fmt, options=options)
        if stop_after is not None and stop_after not in STAGES:
            raise DistillError(f"unknown stage {stop_after!r}")
        _client, simulated = self._client_for(opts)
        docs_path = str(docs or opts.docs or "") or None
        eval_path = str(eval_set or opts.eval_set or "") or None
        if rec.kind == "qa_from_docs" and docs_path is None and not simulated:
            raise DistillError("qa_from_docs needs --docs DIR (a folder or file of documents)")
        if docs_path is not None and not await asyncio.to_thread(Path(docs_path).exists):
            raise DistillError(f"docs path does not exist: {docs_path}")
        eval_texts: list[str] = []
        if eval_path is not None:
            try:
                eval_texts = read_texts(eval_path)
            except (ExportError, ValueError) as exc:
                raise DistillError(f"cannot read the eval set: {exc}") from exc

        meta = RunMeta(
            id=run_id or new_run_id(rec.name),
            recipe=rec,
            n=n,
            models=panel,
            seed_model=opts.seed_model or self.registry.pick(Role(rec.seed_role)),
            reward_model=opts.reward_model or self.registry.pick(Role.REWARD),
            judge_model=opts.judge_model or self.registry.pick(Role.CHAT),
            embed_model=opts.embed_model or self.registry.pick(Role.EMBED),
            weak_model=(opts.weak_model or self.registry.pick(Role.FAST))
            if rec.kind == "eval_set"
            else None,
            docs=docs_path,
            eval_set=eval_path,
            format=cast(ExportFormat, fmt),
            out=str(out) if out else None,
            options=opts,
            simulated=simulated,
            version=__version__,
        )
        if self.store.exists(meta.id):
            raise DistillError(f"run {meta.id} already exists; use resume")
        self.store.save_meta(meta)
        if eval_texts:
            self.store.save_eval_texts(meta.id, eval_texts)
        return await self._execute(meta, stop_after=stop_after)

    async def resume(
        self, run_id: str, *, out: str | Path | None = None, stop_after: str | None = None
    ) -> DistillResult:
        """Continue an interrupted or failed run from its checkpoint."""
        meta = self.store.load_meta(run_id)
        if out is not None:
            meta.out = str(out)
        meta.error = None
        return await self._execute(meta, stop_after=stop_after, resuming=True)

    def uses_simulator(self, options: RunOptions | None = None) -> bool:
        """Would a run now use the offline simulator (mock mode or no API key)?"""
        return self._client_for(options or RunOptions())[1]

    def list_runs(self) -> list[RunMeta]:
        return self.store.list_runs()

    def load(self, run_id: str) -> DistillResult:
        """A finished (or partial) run from disk, without running anything."""
        meta = self.store.load_meta(run_id)
        samples = self.store.load_samples(run_id)
        seeds = self.store.load_seeds(run_id)
        report = self._report_from_disk(meta, samples, len(seeds))
        return DistillResult(meta, report, samples, self.store.path(run_id))

    def export(
        self,
        run_id: str,
        fmt: str | None = None,
        path: str | Path | None = None,
        *,
        include_meta: bool | None = None,
    ) -> Path:
        """Write the kept samples of a run as JSONL in ``fmt`` and return the file path."""
        meta = self.store.load_meta(run_id)
        fmt = fmt or meta.format
        if fmt not in FORMATS:
            raise DistillError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")
        if fmt == "dpo" and meta.recipe.kind != "preference_pairs":
            raise DistillError("the dpo format needs a preference_pairs run (chosen/rejected)")
        kept = [s for s in self.store.load_samples(run_id) if s.alive]
        target = (
            Path(path)
            if path
            else self.store.export_dir(run_id) / f"{meta.recipe.name}.{fmt}.jsonl"
        )
        meta_flag = meta.options.include_meta
        flag = (
            include_meta
            if include_meta is not None
            else (meta_flag if meta_flag is not None else meta.recipe.include_meta)
        )
        write_jsonl(
            (sample_to_record(s, meta.recipe.system_prompt) for s in kept),
            target,
            fmt,
            include_meta=flag,
        )
        return target

    async def dedup_file(
        self,
        path: str | Path,
        *,
        threshold: float = 0.92,
        field: str = "prompt",
        out: str | Path | None = None,
        options: RunOptions | None = None,
    ) -> tuple[int, int, Path]:
        """Near-duplicate removal for any JSONL dataset. Returns ``(rows, kept, output path)``.

        Rows keep their original JSON; the first row of each cluster wins. Embeddings come from the
        embedding model of the registry (``Role.EMBED``).
        """
        if field not in ("prompt", "response", "both"):
            raise DistillError("field must be prompt, response or both")
        source = Path(path)
        try:
            text = await asyncio.to_thread(source.read_text, encoding="utf-8")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            records = [from_json(json.loads(ln)) for ln in lines]
        except (OSError, ValueError, KeyError, ExportError) as exc:
            raise DistillError(f"cannot read {source}: {exc}") from exc
        opts = options or RunOptions()
        client, _ = self._client_for(opts)
        gateway = Gateway(client, Tally(), opts.concurrency or 8)
        texts = [
            r.response
            if field == "response"
            else f"{r.prompt}\n{r.response}"
            if field == "both"
            else r.prompt
            for r in records
        ]
        model = opts.embed_model or self.registry.pick(Role.EMBED)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 32):
            vectors.extend(await gateway.embed(model, texts[i : i + 32]))
        keys = [normalize(t) for t in texts]
        exact = exact_duplicates(keys)
        keep = [i for i in range(len(records)) if i not in exact]
        reps = greedy_clusters(np.array([vectors[i] for i in keep]), threshold) if keep else []
        kept_rows = sorted({keep[r] for r in reps})
        target = Path(out) if out else source.with_name(f"{source.stem}.dedup{source.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines[i] + "\n" for i in kept_rows), encoding="utf-8")
        return len(records), len(kept_rows), target

    async def report(self, run_id: str) -> QualityReport:
        return self.load(run_id).report

    # ------------------------------------------------------------------ setup helpers
    @staticmethod
    def _with_overrides(rec: Recipe, opts: RunOptions) -> Recipe:
        update: dict[str, Any] = {}
        if opts.filters:
            update["filters"] = rec.filters.model_copy(update=opts.filters)
        if opts.dedup_threshold is not None:
            update["dedup"] = rec.dedup.model_copy(update={"threshold": opts.dedup_threshold})
        if opts.max_tokens is not None:
            update["max_tokens"] = opts.max_tokens
        if opts.domains:
            update["domains"] = opts.domains
        if not update:
            return rec
        merged = rec.model_dump()
        for key, value in update.items():
            merged[key] = value.model_dump() if hasattr(value, "model_dump") else value
        return Recipe.model_validate(merged)

    def _panel(self, rec: Recipe, models: list[str] | None) -> list[str]:
        if models:
            panel = list(dict.fromkeys(m.strip() for m in models if m.strip()))
        elif rec.kind == "eval_set":
            panel = [self.registry.pick(Role.REASONING)]
        else:
            panel = list(
                dict.fromkeys(
                    [self.registry.pick(Role(rec.generator_role)), self.registry.pick(Role.FAST)]
                )
            )
        if rec.kind == "preference_pairs" and len(panel) < 2:
            others = [m.id for m in self.registry.by_role(Role.CHAT) if m.id not in panel]
            if len(panel) + len(others) < 2:
                raise DistillError(
                    "preference_pairs needs at least two different models (--models a,b)"
                )
            panel += others[: 2 - len(panel)]
        if not panel:
            raise DistillError("no generator models given")
        return panel

    # ------------------------------------------------------------------ execution
    async def _execute(
        self, meta: RunMeta, *, stop_after: str | None = None, resuming: bool = False
    ) -> DistillResult:
        client, _simulated = self._client_for(
            meta.options, require_simulated=meta.simulated if resuming else None
        )
        tally = Tally(meta.usage)
        gateway = Gateway(client, tally, meta.options.concurrency or 8)
        scorer = Scorer(gateway, meta.reward_model, meta.judge_model)
        state = _Run(meta, gateway, tally, scorer)
        state.seeds = self.store.load_seeds(meta.id)
        for s in self.store.load_samples(meta.id):
            state.samples[s.seed_idx] = s
        state.eval_texts = self.store.load_eval_texts(meta.id)
        state.vectors = self.store.load_vectors(meta.id)
        meta.status = "running"
        self.store.save_meta(meta)
        self.emit(
            "run.config",
            {
                "run_id": meta.id,
                "recipe": meta.recipe.name,
                "kind": meta.recipe.kind,
                "n": meta.n,
                "models": meta.models,
                "simulated": meta.simulated,
                "resumed": resuming,
                "stages_done": list(meta.stages_done),
            },
        )
        handlers: dict[str, Callable[[_Run], Awaitable[None]]] = {
            "seed": self._stage_seed,
            "generate": self._stage_generate,
            "score": self._stage_score,
            "filter": self._stage_filter,
            "dedup": self._stage_dedup,
            "decontaminate": self._stage_decontaminate,
            "finalize": self._stage_finalize,
        }
        try:
            for stage in STAGES:
                if stage in meta.stages_done:
                    continue
                started = time.monotonic()
                self.emit("stage.start", {"stage": stage, "total": len(state.seeds) or meta.n})
                await handlers[stage](state)
                self._checkpoint(state, stage)
                self.emit(
                    "stage.end",
                    {
                        "stage": stage,
                        "seconds": round(time.monotonic() - started, 3),
                        "alive": sum(1 for s in state.samples.values() if s.alive),
                    },
                )
                self._emit_funnel(state)
                if stop_after == stage and stage != STAGES[-1]:
                    meta.status = "interrupted"
                    self.store.save_meta(meta)
                    return self._result(state)
            meta.status = "done"
            meta.finished_at = utc_now()
            self.store.save_meta(meta)
        except asyncio.CancelledError:
            self._interrupt(state, "interrupted", None)
            raise
        except Exception as exc:
            self._interrupt(state, "failed", f"{type(exc).__name__}: {exc}")
            raise
        return self._result(state)

    def _interrupt(self, state: _Run, status: str, error: str | None) -> None:
        meta = state.meta
        meta.status = status  # type: ignore[assignment]
        meta.error = error
        meta.usage = state.tally.as_dict()
        meta.cost_usd = estimate_cost(meta.usage, self.registry)
        self.store.save_samples(meta.id, state.table())
        self.store.save_meta(meta)

    def _checkpoint(self, state: _Run, stage: str) -> None:
        meta = state.meta
        meta.stages_done.append(stage)
        meta.usage = state.tally.as_dict()
        meta.cost_usd = estimate_cost(meta.usage, self.registry)
        self.store.save_samples(meta.id, state.table())
        self.store.save_meta(meta)

    def _partial_save(self, state: _Run) -> None:
        state.meta.usage = state.tally.as_dict()
        self.store.save_samples(state.meta.id, state.table())
        self.store.save_meta(state.meta)

    def _emit_funnel(self, state: _Run) -> None:
        from .quality import compute_funnel

        rows = compute_funnel(state.table(), len(state.seeds))
        self.emit(
            "funnel",
            {
                "stages": [r.model_dump(by_alias=True) for r in rows],
                "kept": sum(1 for s in state.samples.values() if s.alive),
                "total": len(state.seeds),
            },
        )

    def _report_from_disk(
        self, meta: RunMeta, samples: list[Sample], n_seeds: int
    ) -> QualityReport:
        cached = self.store.load_json(meta.id, "quality.json")
        if cached is not None and "finalize" in meta.stages_done:
            return QualityReport.model_validate(cached)
        return build_report(meta, samples, n_seeds, self.store.load_vectors(meta.id))

    def _result(self, state: _Run) -> DistillResult:
        meta = state.meta
        samples = state.table()
        report = self._report_from_disk(meta, samples, len(state.seeds))
        files = [Path(f) for f in meta.files]
        return DistillResult(meta, report, samples, self.store.path(meta.id), files)

    # ------------------------------------------------------------------ utilities
    @staticmethod
    async def _gather(coros: Iterable[Awaitable[Any]]) -> list[Any]:
        """Run awaitables concurrently; on the first failure cancel the rest and re-raise."""
        tasks = [asyncio.ensure_future(c) for c in coros]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    @staticmethod
    async def _try(coro: Awaitable[Any]) -> Any:
        """Await ``coro``; model errors (other than auth) become ``None``."""
        try:
            return await coro
        except NIMAuthError:
            raise
        except NIMError:
            return None

    # ------------------------------------------------------------------ stage: seed
    async def _stage_seed(self, st: _Run) -> None:
        meta, rec = st.meta, st.meta.recipe
        if rec.kind == "qa_from_docs":
            seeds = self._seeds_from_docs(st)
        else:
            seeds = await self._seeds_from_tree(st)
        if not seeds:
            raise DistillError(
                "the seed stage produced nothing (no documents, or every model call failed)"
            )
        st.seeds = seeds
        self.store.save_seeds(meta.id, seeds)
        self.emit("seeds", {"count": len(seeds), "topics": len({s.topic for s in seeds})})

    def _seeds_from_docs(self, st: _Run) -> list[Seed]:
        meta, rec = st.meta, st.meta.recipe
        chunks: list[tuple[str, str, str]] = []  # (source, title, text)
        if meta.docs:
            for path, name in discover_files(Path(meta.docs)):
                try:
                    doc = load_file(path, name=name)
                except AtlasError:
                    self.emit("warning", {"message": f"skipped unreadable document {name}"})
                    continue
                for ch in chunk_document(
                    doc, max_chars=rec.chunk_chars, overlap=rec.chunk_overlap, min_chars=120
                ):
                    chunks.append((f"{name} #{ch.ord}", doc.title, ch.text))
        else:
            from nimlab.atlas.loaders import load_bytes

            for name, text in BUILTIN_DOCS.items():
                doc = load_bytes(text.encode("utf-8"), name=name)
                for ch in chunk_document(
                    doc, max_chars=rec.chunk_chars, overlap=rec.chunk_overlap, min_chars=120
                ):
                    chunks.append((f"{name} #{ch.ord}", doc.title, ch.text))
        if not chunks:
            raise DistillError(f"no readable text found under {meta.docs}")
        total = len(chunks)
        seeds = []
        for i in range(meta.n):
            idx = (i * total) // meta.n if meta.n <= total else i % total
            source, title, text = chunks[idx]
            seeds.append(
                Seed(
                    idx=i,
                    topic=title,
                    subtopic=source,
                    context=text,
                    source=source,
                    variant=i // total if meta.n > total else 0,
                )
            )
        return seeds

    async def _seeds_from_tree(self, st: _Run) -> list[Seed]:
        meta, rec = st.meta, st.meta.recipe
        domains = rec.domains or ["general knowledge"]
        model = meta.seed_model

        async def topics_of(domain: str) -> list[tuple[str, str]]:
            r = await self._try(
                st.gw.chat_json(
                    model,
                    topics_messages(domain, rec.topics_per_domain),
                    TopicList,
                    temperature=0.8,
                )
            )
            return [(domain, t) for t in (r.topics if r else [])[: rec.topics_per_domain]]

        topic_pairs = [
            p for batch in await self._gather(topics_of(d) for d in domains) for p in batch
        ]

        async def subtopics_of(domain: str, topic: str) -> list[tuple[str, str, str]]:
            r = await self._try(
                st.gw.chat_json(
                    model,
                    subtopics_messages(topic, rec.subtopics_per_topic),
                    SubtopicList,
                    temperature=0.8,
                )
            )
            return [
                (domain, topic, s) for s in (r.subtopics if r else [])[: rec.subtopics_per_topic]
            ]

        leaves = [
            leaf
            for batch in await self._gather(subtopics_of(d, t) for d, t in topic_pairs)
            for leaf in batch
        ]
        if not leaves:
            return []
        per_leaf = math.ceil(meta.n / len(leaves))

        async def prompts_of(topic: str, sub: str) -> list[str]:
            if rec.kind == "eval_set":
                q = await self._try(
                    st.gw.chat_json(
                        model,
                        hard_questions_messages(topic, sub, per_leaf),
                        QuestionList,
                        temperature=0.8,
                    )
                )
                return list(q.questions) if q else []
            i = await self._try(
                st.gw.chat_json(
                    model,
                    instructions_messages(topic, sub, per_leaf),
                    InstructionList,
                    temperature=0.9,
                )
            )
            return list(i.instructions) if i else []

        lists = await self._gather(prompts_of(t, s) for _d, t, s in leaves)
        seeds: list[Seed] = []
        depth = max((len(x) for x in lists), default=0)
        for level in range(depth):  # round-robin over leaves: neighbouring samples differ in topic
            for (domain, topic, sub), prompts in zip(leaves, lists, strict=True):
                if level < len(prompts) and len(seeds) < meta.n:
                    seeds.append(
                        Seed(
                            idx=len(seeds),
                            topic=f"{domain} / {topic}",
                            subtopic=sub,
                            prompt=prompts[level],
                        )
                    )
        return seeds

    # ------------------------------------------------------------------ stage: generate
    async def _stage_generate(self, st: _Run) -> None:
        todo = [s for s in st.seeds if s.idx not in st.samples]
        total = len(st.seeds)
        done = len(st.samples)
        every = st.meta.options.checkpoint_every

        async def one(seed: Seed) -> None:
            nonlocal done
            sample = await self._generate_one(st, seed)
            st.samples[seed.idx] = sample
            done += 1
            if sample.alive:
                self.emit(
                    "sample.generated",
                    {
                        "id": sample.id,
                        "prompt": _clip(sample.prompt),
                        "model": sample.model,
                        "temperature": sample.temperature,
                        "topic": sample.topic,
                        "done": done,
                        "total": total,
                    },
                )
            else:
                self._emit_dropped(sample, "generate")
            if done % every == 0:
                self._partial_save(st)

        await self._gather(one(s) for s in todo)

    async def _generate_one(self, st: _Run, seed: Seed) -> Sample:
        meta, rec = st.meta, st.meta.recipe
        k = len(meta.models)
        model = meta.models[seed.idx % k]
        temp = rec.temperatures[(seed.idx // k) % len(rec.temperatures)]
        base = Sample(
            id=f"s{seed.idx:05d}",
            seed_idx=seed.idx,
            prompt=seed.prompt,
            response="",
            model=model,
            temperature=temp,
            topic=seed.topic,
            subtopic=seed.subtopic,
            source=seed.source,
            context=seed.context,
        )
        try:
            if rec.kind == "qa_from_docs":
                qa = await st.gw.chat_json(
                    model,
                    qa_messages(seed.context, seed.variant, seed.source),
                    QAPair,
                    temperature=temp,
                )
                base.prompt, base.response = qa.question.strip(), qa.answer.strip()
            elif rec.kind == "preference_pairs":
                model_b = meta.models[(seed.idx + 1) % k]
                temp_b = rec.temperatures[(seed.idx // k + 1) % len(rec.temperatures)]
                msgs = answer_messages(seed.prompt, rec.system_prompt)
                ra, rb = await asyncio.gather(
                    st.gw.chat(model, msgs, temperature=temp, max_tokens=rec.max_tokens),
                    st.gw.chat(model_b, msgs, temperature=temp_b, max_tokens=rec.max_tokens),
                )
                base.response, base.rejected, base.rejected_model = (
                    ra.content.strip(),
                    rb.content.strip(),
                    model_b,
                )
            elif rec.kind == "eval_set":
                weak = meta.weak_model or meta.models[0]
                ref, wk = await asyncio.gather(
                    st.gw.chat(
                        model,
                        answer_messages(seed.prompt, reference=True),
                        temperature=temp,
                        max_tokens=rec.max_tokens,
                    ),
                    st.gw.chat(
                        weak,
                        answer_messages(seed.prompt, rec.system_prompt),
                        temperature=0.7,
                        max_tokens=rec.max_tokens,
                    ),
                )
                base.response = ref.content.strip()
                base.extra = {"weak_model": weak, "weak_response": wk.content.strip()}
            else:
                r = await st.gw.chat(
                    model,
                    answer_messages(seed.prompt, rec.system_prompt),
                    temperature=temp,
                    max_tokens=rec.max_tokens,
                )
                base.response = r.content.strip()
        except NIMAuthError:
            raise
        except NIMError as exc:
            base.drop("generate", "error", _clip(str(exc), 160))
        else:
            base.tokens = max(1, (len(base.prompt) + len(base.response)) // 4)
        return base

    def _emit_dropped(self, s: Sample, stage: str) -> None:
        self.emit(
            "sample.dropped",
            {
                "id": s.id,
                "stage": stage,
                "reason": s.reason,
                "detail": s.detail,
                "prompt": _clip(s.prompt, 120),
                "score": s.score,
                "score_source": s.score_source,
            },
        )

    # ------------------------------------------------------------------ stage: score
    async def _stage_score(self, st: _Run) -> None:
        kind = st.meta.recipe.kind
        todo = [s for s in st.table() if s.alive and s.score is None]
        total = len(st.samples)
        done = total - len(todo)
        every = st.meta.options.checkpoint_every

        async def one(s: Sample) -> None:
            nonlocal done
            ok = await self._score_one(st, s, kind)
            done += 1
            if ok:
                self.emit(
                    "sample.scored",
                    {
                        "id": s.id,
                        "score": round(s.score, 4) if s.score is not None else None,
                        "source": s.score_source,
                        "model": s.model,
                        "difficulty": s.difficulty,
                        "done": done,
                        "total": total,
                    },
                )
            else:
                self._emit_dropped(s, "score")
            if done % every == 0:
                self._partial_save(st)

        await self._gather(one(s) for s in todo)

    async def _score_one(self, st: _Run, s: Sample, kind: str) -> bool:
        scorer = st.scorer
        if kind == "preference_pairs" and s.rejected is not None:
            pair = await scorer.score_pair(s.prompt, s.response, s.rejected)
            if pair is None:
                s.drop("score", "unscored", "reward model and judge both failed")
                return False
            a, b = pair
            if b.value > a.value:  # the second answer wins: swap so `response` is the chosen one
                s.response, s.rejected = s.rejected, s.response
                s.model, s.rejected_model = s.rejected_model or s.model, s.model
                a, b = b, a
            s.score, s.rejected_score, s.score_source = a.value, b.value, a.source
            return True
        if kind == "eval_set":
            weak = str(s.extra.get("weak_response", ""))
            pair = await scorer.score_pair(s.prompt, s.response, weak)
            if pair is None:
                s.drop("score", "unscored", "reward model and judge both failed")
                return False
            ref, wk = pair
            scale = 3.0 if ref.source == "reward" else 1.5
            s.score, s.score_source = ref.value, ref.source
            s.extra["weak_score"] = round(wk.value, 4)
            s.difficulty = round(1.0 / (1.0 + math.exp(-(ref.value - wk.value) / scale)), 4)
            return True
        score = await scorer.score(s.prompt, s.response)
        if score is None:
            s.drop("score", "unscored", "reward model and judge both failed")
            return False
        s.score, s.score_source = score.value, score.source
        return True

    # ------------------------------------------------------------------ stage: filter
    async def _stage_filter(self, st: _Run) -> None:
        cfg, kind = st.meta.recipe.filters, st.meta.recipe.kind
        for s in st.table():
            if not s.alive:
                continue
            failure = apply_filters(s, cfg, kind)
            if failure is not None:
                s.drop("filter", failure[0], failure[1])
                self._emit_dropped(s, "filter")

    # ------------------------------------------------------------------ stage: dedup
    @staticmethod
    def _dedup_text(s: Sample, field_: str) -> str:
        if field_ == "response":
            return s.response
        if field_ == "both":
            return f"{s.prompt}\n{s.response}"
        return s.prompt

    async def _vectors_for(self, st: _Run, samples: list[Sample], field_: str) -> np.ndarray:
        """Embeddings keyed by sample id and text field (cached in ``vectors.npz``)."""

        def key(s: Sample) -> str:
            return f"{field_}:{s.id}"

        missing = [s for s in samples if key(s) not in st.vectors]
        batch = 32
        chunks = [missing[i : i + batch] for i in range(0, len(missing), batch)]

        async def embed(group: list[Sample]) -> list[list[float]]:
            return await st.gw.embed(
                st.meta.embed_model, [self._dedup_text(s, field_) for s in group]
            )

        for group, vecs in zip(chunks, await self._gather(embed(g) for g in chunks), strict=True):
            for s, v in zip(group, vecs, strict=True):
                st.vectors[key(s)] = [float(x) for x in v]
        if missing:
            self.store.save_vectors(st.meta.id, st.vectors)
        return np.array([st.vectors[key(s)] for s in samples], dtype=np.float64)

    async def _stage_dedup(self, st: _Run) -> None:
        cfg = st.meta.recipe.dedup
        cluster_no = 0

        def alive() -> list[Sample]:
            return [s for s in st.table() if s.alive]

        if cfg.exact:
            pool = alive()
            keys = [normalize(self._dedup_text(s, cfg.field)) for s in pool]
            dups = exact_duplicates(keys, [comparable_score(s) for s in pool])
            groups: dict[int, list[int]] = {}
            for dup, kept in dups.items():
                groups.setdefault(kept, [kept]).append(dup)
            for kept, members in sorted(groups.items()):
                cluster_no += 1
                for i in members:
                    pool[i].cluster = cluster_no
                for i in members:
                    if i != kept:
                        pool[i].drop("dedup", "duplicate_exact", f"same text as {pool[kept].id}")
                        self._emit_dropped(pool[i], "dedup")
                self.emit(
                    "dedup.cluster",
                    {
                        "cluster": cluster_no,
                        "kind": "exact",
                        "size": len(members),
                        "kept": pool[kept].id,
                        "members": [pool[i].id for i in sorted(members)],
                    },
                )
        if cfg.near:
            pool = alive()
            if len(pool) > 1:
                vectors = await self._vectors_for(st, pool, cfg.field)
                reps = greedy_clusters(vectors, cfg.threshold, [comparable_score(s) for s in pool])
                for rep, members in sorted(cluster_members(reps).items()):
                    if len(members) < 2:
                        continue
                    cluster_no += 1
                    for i in members:
                        pool[i].cluster = cluster_no
                        if i != rep:
                            pool[i].drop(
                                "dedup",
                                "duplicate_near",
                                f"cosine >= {cfg.threshold:g} to {pool[rep].id}",
                            )
                            self._emit_dropped(pool[i], "dedup")
                    self.emit(
                        "dedup.cluster",
                        {
                            "cluster": cluster_no,
                            "kind": "near",
                            "size": len(members),
                            "kept": pool[rep].id,
                            "members": [pool[i].id for i in members],
                        },
                    )

    # ------------------------------------------------------------------ stage: decontaminate
    async def _stage_decontaminate(self, st: _Run) -> None:
        cfg = st.meta.recipe.decontaminate
        if not cfg.enabled or not st.eval_texts:
            return
        index = EvalIndex(st.eval_texts, cfg.ngram)
        for s in st.table():
            if not s.alive:
                continue
            text = s.prompt if cfg.check == "prompt" else f"{s.prompt}\n{s.response}"
            overlap = index.overlap(text)
            if overlap >= cfg.min_overlap:
                s.drop("decontaminate", "contaminated", f"{overlap:.0%} of n-grams in the eval set")
                self._emit_dropped(s, "decontaminate")

    # ------------------------------------------------------------------ stage: finalize
    async def _stage_finalize(self, st: _Run) -> None:
        meta = st.meta
        kept = [s for s in st.table() if s.alive]
        if len(kept) > 1:
            vectors = await self._vectors_for(st, kept, "prompt")
            by_id = {s.id: vectors[i].tolist() for i, s in enumerate(kept)}
        else:
            by_id = {}
        # Embedding the kept samples may have added usage: refresh it before building the report.
        meta.usage = st.tally.as_dict()
        meta.cost_usd = estimate_cost(meta.usage, self.registry)
        report = build_report(meta, st.table(), len(st.seeds), by_id)
        self.store.save_json(meta.id, "quality.json", report.model_dump(by_alias=True))
        self.store.save_samples(meta.id, st.table())
        meta.stages_done.append("finalize")  # so the cached report is used while exporting
        out_dir = Path(meta.out) / meta.id if meta.out else self.store.export_dir(meta.id)
        files = self._write_outputs(meta, out_dir)
        meta.files = [str(p.resolve()) for p in files]
        meta.stages_done.remove("finalize")
        self.emit("export", {"files": meta.files, "kept": len(kept)})

    def _write_outputs(self, meta: RunMeta, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        data_path = out_dir / f"{meta.recipe.name}.{meta.format}.jsonl"
        self.export(meta.id, meta.format, data_path)
        report = self._report_from_disk(
            meta, self.store.load_samples(meta.id), len(self.store.load_seeds(meta.id))
        )
        html_path = write_report_html(report, meta, out_dir / "quality_report.html")
        card_path = out_dir / "README.md"
        card_path.write_text(render_card(meta, report, data_path.name), encoding="utf-8")
        return [data_path, card_path, html_path]
