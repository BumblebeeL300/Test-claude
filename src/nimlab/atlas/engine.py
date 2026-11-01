"""The :class:`Atlas` facade: ingestion, retrieval and (agentic, verified) question answering."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nimlab.core import ModelRegistry, NIMAuthError, NIMClient, NIMConfig, NIMError, strip_think

from .agent import AgenticRetriever, AgentState
from .answering import build_answer_messages, looks_like_abstention, parse_citations
from .chunking import chunk_document
from .compat import is_mock, usage_totals
from .config import DEFAULT_COLLECTION, DEFAULT_DB_PATH, AtlasConfig, ModelSet
from .guard import SafetyGuard, refusal_text
from .loaders import discover_files, load_bytes, media_type_for
from .models import (
    Answer,
    AtlasError,
    AtlasEvent,
    Chunk,
    CollectionInfo,
    DocRecord,
    FileReport,
    GuardVerdict,
    IngestReport,
    LoadedDocument,
    LoaderError,
    Mode,
    SourceRef,
    TraceStep,
    VerificationReport,
)
from .retrieval import Retriever, sources_from_hits
from .store import AtlasStore, NewChunk, normalize_rows
from .verify import Verifier, build_revision_messages

MOCK_WARNING = (
    "Mock mode: answers are synthetic placeholders. Retrieval, ranking and groundedness checks "
    "are real, offline computations."
)


@dataclass
class _Prepared:
    """A document that has been parsed and chunked and is waiting for embeddings."""

    doc: LoadedDocument
    key: str
    existing: DocRecord | None
    chunks: list[Chunk]
    duplicates: int
    reused: dict[str, np.ndarray]
    todo: list[Chunk]


def _pieces(text: str) -> list[str]:
    """Word-sized pieces (whitespace kept) so a buffered answer can still be streamed."""
    return re.findall(r"\s*\S+\s?|\s+", text) or ([text] if text else [])


class Atlas:
    """Agentic, citation-verified RAG over a local SQLite knowledge base.

    >>> atlas = Atlas(client, ".nimlab/atlas.db")           # doctest: +SKIP
    >>> await atlas.ingest("docs/")                         # doctest: +SKIP
    >>> answer = await atlas.ask("How do I rotate keys?", verify=True)  # doctest: +SKIP
    """

    def __init__(
        self,
        client: Any,
        db_path: str | Path = DEFAULT_DB_PATH,
        *,
        config: AtlasConfig | None = None,
        registry: ModelRegistry | None = None,
        store: AtlasStore | None = None,
    ) -> None:
        self.client = client
        self.config = config or AtlasConfig()
        self.models = ModelSet.resolve(self.config, registry)
        self.store = store or AtlasStore(db_path, k1=self.config.bm25_k1, b=self.config.bm25_b)
        self.store.configure_bm25(self.config.bm25_k1, self.config.bm25_b)
        self.mock = is_mock(client)
        self.retriever = Retriever(self.store, client, self.config, self.models)
        self.agent = AgenticRetriever(
            client, self.retriever, self.models, self.config, offline=self.mock
        )
        self._owned_client: NIMClient | None = None

    # ------------------------------------------------------------------ construction
    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        db_path: str | Path = DEFAULT_DB_PATH,
        *,
        config: AtlasConfig | None = None,
        nim_config: NIMConfig | None = None,
        mock: bool | None = None,
        registry: ModelRegistry | None = None,
    ) -> AsyncIterator[Atlas]:
        """Create an :class:`Atlas` that owns its NIM client and store and closes both on exit.

        ``mock=True`` forces the offline backend; ``None`` follows ``NIMLAB_MOCK`` / the API key.
        """
        if nim_config is None:
            nim_config = NIMConfig.from_env(**({"mock": True} if mock else {}))
        async with NIMClient(nim_config) as client:
            atlas = cls(client, db_path, config=config, registry=registry)
            try:
                yield atlas
            finally:
                atlas.close()

    def close(self) -> None:
        self.store.close()

    @property
    def embed_tag(self) -> str:
        """Identifies the embedding space; mock and live vectors are never mixed."""
        return f"mock:{self.models.embed}" if self.mock else self.models.embed

    # ------------------------------------------------------------------ ingestion
    async def ingest(
        self,
        path: str | Path,
        collection: str = DEFAULT_COLLECTION,
        *,
        force: bool = False,
        prune: bool = False,
        recursive: bool = True,
        on_file: Callable[[FileReport], None] | None = None,
    ) -> IngestReport:
        """Ingest a file or directory (``.md .txt .html .pdf``) into ``collection``.

        Unchanged files (same content hash and chunking settings) are skipped; changed files are
        re-chunked and only their *new* chunks are embedded. ``prune=True`` also removes documents
        that were ingested from this directory but no longer exist on disk.
        """
        root = Path(path).expanduser()
        found = discover_files(root, recursive=recursive)
        sources: list[tuple[str, str, bytes | Exception]] = []
        for file_path, name in found:
            key = str(file_path.resolve())
            try:
                sources.append((key, name, file_path.read_bytes()))
            except OSError as exc:
                sources.append((key, name, LoaderError(f"cannot read {file_path}: {exc}")))
        report = await self._ingest_sources(collection, sources, force=force, on_file=on_file)
        if prune and root.is_dir():
            prefix = str(root.resolve()).rstrip("/") + "/"
            alive = {key for key, _, _ in sources}
            for doc in self.store.list_documents(collection):
                if doc.path.startswith(prefix) and doc.path not in alive:
                    self.store.delete_document(collection, doc.path)
                    report.removed.append(doc.name)
        return report

    async def ingest_bytes(
        self,
        files: Sequence[tuple[str, bytes]],
        collection: str = DEFAULT_COLLECTION,
        *,
        force: bool = False,
    ) -> IngestReport:
        """Ingest uploaded ``(filename, bytes)`` pairs (stored under ``upload://<filename>``)."""
        sources: list[tuple[str, str, bytes | Exception]] = []
        for name, data in files:
            clean = Path(name.replace("\\", "/")).name or "upload"
            sources.append((f"upload://{clean}", clean, data))
        return await self._ingest_sources(collection, sources, force=force)

    async def _ingest_sources(
        self,
        collection: str,
        sources: list[tuple[str, str, bytes | Exception]],
        *,
        force: bool,
        on_file: Callable[[FileReport], None] | None = None,
    ) -> IngestReport:
        t0 = time.perf_counter()
        if not collection.strip():
            raise AtlasError("collection name must not be empty")
        self.store.ensure_collection(collection, self.embed_tag)
        cfg = self.config
        sig = cfg.chunk_signature(self.embed_tag)
        known = self.store.known_chunk_hashes(collection)
        seen_run: dict[str, str] = {}
        reports: dict[str, FileReport] = {}
        prepared: list[_Prepared] = []

        def finish(rep: FileReport) -> None:
            reports[rep.path] = rep
            if on_file is not None:
                on_file(rep)

        # phase 1: parse + chunk + dedupe (no network)
        for key, name, data in sources:
            if isinstance(data, Exception):
                finish(FileReport(path=key, name=name, status="error", message=str(data)))
                continue
            if media_type_for(name) is None:
                finish(FileReport(path=key, name=name, status="skipped", message="unsupported type"))
                continue
            import hashlib

            raw_hash = hashlib.sha256(data).hexdigest()
            existing = self.store.get_document(collection, key)
            if (
                existing
                and not force
                and existing.content_hash == raw_hash
                and existing.chunk_sig == sig
            ):
                finish(
                    FileReport(
                        path=key,
                        name=name,
                        status="unchanged",
                        chunks=existing.n_chunks,
                        message="content hash unchanged",
                    )
                )
                continue
            try:
                doc = load_bytes(data, name=name, path=key)
                chunks = chunk_document(
                    doc,
                    max_chars=cfg.chunk_max_chars,
                    overlap=cfg.chunk_overlap,
                    min_chars=cfg.chunk_min_chars,
                )
            except AtlasError as exc:
                finish(FileReport(path=key, name=name, status="error", message=str(exc)))
                continue
            if not chunks:
                finish(FileReport(path=key, name=name, status="error", message="no text found"))
                continue
            kept: list[Chunk] = []
            dups = 0
            local: set[str] = set()
            for chunk in chunks:
                h = chunk.hash
                owner = known.get(h)
                if h in local or (owner is not None and owner != key) or h in seen_run:
                    dups += 1
                    continue
                local.add(h)
                kept.append(chunk)
            if not kept:
                if existing:
                    self.store.delete_document(collection, key)
                finish(
                    FileReport(
                        path=key,
                        name=name,
                        status="skipped",
                        duplicates=dups,
                        message="every chunk duplicates content that is already indexed",
                    )
                )
                continue
            for chunk in kept:
                seen_run[chunk.hash] = key
            old = self.store.document_vectors(collection, key) if existing else {}
            prepared.append(
                _Prepared(
                    doc=doc,
                    key=key,
                    existing=existing,
                    chunks=kept,
                    duplicates=dups,
                    reused={h: v for h, v in old.items()},
                    todo=[c for c in kept if c.hash not in old],
                )
            )

        # phase 2: embed everything that needs it, in concurrent batches
        vectors, failed = await self._embed_chunks(prepared)

        # phase 3: write documents
        for prep in prepared:
            if prep.key in failed:
                finish(
                    FileReport(
                        path=prep.key, name=prep.doc.name, status="error", message=failed[prep.key]
                    )
                )
                continue
            new_chunks = [
                NewChunk(
                    c, prep.reused[c.hash] if c.hash in prep.reused else vectors[(prep.key, c.hash)]
                )
                for c in prep.chunks
            ]
            stored = self.store.replace_document(collection, prep.doc, new_chunks, sig)
            finish(
                FileReport(
                    path=prep.key,
                    name=prep.doc.name,
                    status="updated" if prep.existing else "added",
                    chunks=stored,
                    duplicates=prep.duplicates,
                    embedded=len(prep.todo),
                    reused=len(prep.chunks) - len(prep.todo),
                )
            )
        ordered = [reports[key] for key, _, _ in sources if key in reports]
        return IngestReport(collection=collection, files=ordered, elapsed_s=time.perf_counter() - t0)

    async def _embed_chunks(
        self, prepared: list[_Prepared]
    ) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, str]]:
        """Embed all pending chunks. Returns ``({(doc_key, chunk_hash): vector}, {doc_key: error})``."""
        items = [(p.key, c) for p in prepared for c in p.todo]
        vectors: dict[tuple[str, str], np.ndarray] = {}
        failed: dict[str, str] = {}
        if not items:
            return vectors, failed
        size = self.config.embed_batch_size
        batches = [items[i : i + size] for i in range(0, len(items), size)]

        async def run(batch: list[tuple[str, Chunk]]) -> None:
            texts = [c.indexed_text for _, c in batch]
            try:
                raw = await self.client.embed(self.models.embed, texts, input_type="passage")
            except NIMAuthError:
                raise
            except NIMError as exc:
                for key, _ in batch:
                    failed.setdefault(key, f"embedding failed: {exc}")
                return
            if len(raw) != len(batch):
                for key, _ in batch:
                    failed.setdefault(key, "embedding service returned the wrong number of vectors")
                return
            mat = normalize_rows(np.asarray(raw, dtype=np.float32))
            for (key, chunk), vec in zip(batch, mat, strict=True):
                vectors[(key, chunk.hash)] = vec

        await asyncio.gather(*(run(b) for b in batches))
        return vectors, failed

    async def reindex(self, collection: str = DEFAULT_COLLECTION) -> int:
        """Re-embed every stored chunk with the current embedding model. Returns chunk count."""
        info = self.store.get_collection(collection)
        if info is None:
            raise AtlasError(f"unknown collection {collection!r}")
        rows = self.store.iter_chunk_texts(collection)
        size = self.config.embed_batch_size
        new: dict[int, np.ndarray] = {}
        for i in range(0, len(rows), size):
            batch = rows[i : i + size]
            raw = await self.client.embed(
                self.models.embed, [t for _, t in batch], input_type="passage"
            )
            mat = normalize_rows(np.asarray(raw, dtype=np.float32))
            for (cid, _), vec in zip(batch, mat, strict=True):
                new[cid] = vec
        if new:
            dim = int(next(iter(new.values())).shape[0])
            self.store.update_vectors(collection, new)
            self.store.set_embedding(collection, self.embed_tag, dim)
        else:
            self.store.set_embedding(collection, self.embed_tag, info.dim or 0)
        return len(new)

    # ------------------------------------------------------------------ inspection
    def collections(self) -> list[CollectionInfo]:
        return self.store.collections()

    def documents(self, collection: str | None = None) -> list[DocRecord]:
        return self.store.list_documents(collection)

    def stats(self) -> dict[str, Any]:
        data = self.store.stats()
        data["mock"] = self.mock
        data["models"] = {
            "chat": self.models.chat,
            "fast": self.models.fast,
            "embed": self.models.embed,
            "rerank": self.models.rerank,
            "safety": self.models.safety,
        }
        return data

    def delete_collection(self, collection: str) -> bool:
        return self.store.delete_collection(collection)

    def delete_document(self, collection: str, path_or_name: str) -> bool:
        for doc in self.store.list_documents(collection):
            if path_or_name in (doc.path, doc.name):
                return self.store.delete_document(collection, doc.path)
        return False

    # ------------------------------------------------------------------ retrieval
    async def retrieve(
        self,
        query: str,
        collection: str = DEFAULT_COLLECTION,
        *,
        k: int | None = None,
        use_rerank: bool | None = None,
        use_mmr: bool | None = None,
        hyde_text: str | None = None,
    ) -> list[SourceRef]:
        """Hybrid retrieval only (no generation). Returns numbered :class:`SourceRef` objects."""
        result = await self.retriever.retrieve(
            query, collection, k=k, use_rerank=use_rerank, use_mmr=use_mmr, hyde_text=hyde_text
        )
        return sources_from_hits(result.hits)

    # ------------------------------------------------------------------ answering
    async def ask(
        self,
        question: str,
        collection: str = DEFAULT_COLLECTION,
        *,
        mode: Mode = "simple",
        k: int | None = None,
        verify: bool = False,
        revise: bool = False,
        guard: bool = False,
        hyde: bool | None = None,
    ) -> Answer:
        """Answer ``question`` from ``collection`` with inline ``[n]`` citations.

        ``mode="agentic"`` plans sub-questions, searches iteratively and synthesises. ``verify``
        checks every claim against its cited passages; ``revise`` additionally rewrites the answer
        when claims are unsupported. ``guard`` screens the question and the answer for safety.
        """
        final: Answer | None = None
        async for event in self.ask_stream(
            question,
            collection,
            mode=mode,
            k=k,
            verify=verify,
            revise=revise,
            guard=guard,
            hyde=hyde,
            stream_tokens=False,
        ):
            if event.type == "final":
                final = event.answer
        assert final is not None
        return final

    async def ask_stream(
        self,
        question: str,
        collection: str = DEFAULT_COLLECTION,
        *,
        mode: Mode = "simple",
        k: int | None = None,
        verify: bool = False,
        revise: bool = False,
        guard: bool = False,
        hyde: bool | None = None,
        stream_tokens: bool = True,
    ) -> AsyncIterator[AtlasEvent]:
        """Like :meth:`ask` but yields :class:`AtlasEvent` objects as the pipeline progresses.

        Event order: ``trace`` steps, one ``sources`` event, ``token`` events (when
        ``stream_tokens``), more ``trace`` steps for verification, then exactly one ``final``.
        With ``guard=True`` the answer is generated in full and screened *before* its tokens are
        released, so unsafe text is never streamed.
        """
        question = question.strip()
        if not question:
            raise AtlasError("question must not be empty")
        if mode not in ("simple", "agentic"):
            raise AtlasError(f"unknown mode {mode!r}; use 'simple' or 'agentic'")
        cfg = self.config
        k = k or cfg.k
        verify = verify or revise
        t0 = time.perf_counter()
        calls0, tokens0 = usage_totals(self.client)
        trace: list[TraceStep] = []
        warnings: list[str] = [MOCK_WARNING] if self.mock else []
        verdicts: list[GuardVerdict] = []

        def step(kind: str, message: str, **data: Any) -> AtlasEvent:
            s = TraceStep(kind=kind, message=message, data=data)
            trace.append(s)
            return AtlasEvent(type="trace", step=s)

        def finish(
            text: str,
            *,
            sources: Sequence[SourceRef] = (),
            citations: Sequence[SourceRef] = (),
            verification: VerificationReport | None = None,
            blocked: bool = False,
            original_text: str | None = None,
        ) -> AtlasEvent:
            calls1, tokens1 = usage_totals(self.client)
            answer = Answer(
                question=question,
                text=text,
                mode=mode,
                collection=collection,
                citations=list(citations),
                sources=list(sources),
                verification=verification,
                guard=verdicts,
                blocked=blocked,
                original_text=original_text,
                trace=trace,
                warnings=list(dict.fromkeys(warnings)),
                model=self.models.chat,
                latency_s=round(time.perf_counter() - t0, 4),
                usage={"calls": calls1 - calls0, "tokens": tokens1 - tokens0},
            )
            return AtlasEvent(type="final", answer=answer)

        guard_obj = (
            SafetyGuard(self.client, self.models.safety, fail_open=cfg.guard_fail_open)
            if guard
            else None
        )

        # 1) screen the question
        if guard_obj is not None:
            verdict = await guard_obj.check("query", question)
            verdicts.append(verdict)
            if verdict.skipped:
                msg = "Safety screening skipped (mock mode)"
            elif verdict.uncertain:
                msg = "Safety screening inconclusive"
            else:
                msg = "Question passed safety screening" if verdict.safe else "Question blocked"
            yield step("guard", msg, safe=verdict.safe, categories=verdict.categories)
            if not verdict.safe:
                warnings.append("The question was blocked by the safety guard.")
                yield finish(refusal_text("query", verdict.categories), blocked=True)
                return

        # 2) is there anything to search?
        info = self.store.get_collection(collection)
        if info is None or info.chunks == 0:
            warnings.append(f"Collection {collection!r} is empty; ingest documents first.")
            yield finish(f"No documents are indexed in collection '{collection}' yet.")
            return

        # 3) retrieve
        use_hyde = cfg.hyde if hyde is None else hyde
        sub_questions: list[str] = []
        if mode == "simple":
            hyde_text: str | None = None
            if use_hyde:
                hyde_text = await self.agent.hyde(question)
                if hyde_text:
                    yield step("hyde", "Generated a hypothetical answer for embedding (HyDE)")
            result = await self.retriever.retrieve(
                question, collection, k=k, hyde_text=hyde_text
            )
            hits = result.hits
            warnings.extend(result.warnings)
            yield step(
                "retrieve",
                f"Retrieved {len(hits)} passages from {info.chunks} chunks",
                **result.stats,
            )
        else:
            state = AgentState()
            async for trace_step in self.agent.explore(
                question, collection, k=k, use_hyde=use_hyde, state=state
            ):
                trace.append(trace_step)
                yield AtlasEvent(type="trace", step=trace_step)
            hits = state.hits
            warnings.extend(state.warnings)
            sub_questions = [q for q in state.sub_questions if q.lower() != question.lower()]
        sources = sources_from_hits(hits)
        yield AtlasEvent(type="sources", sources=sources)
        if not sources:
            warnings.append("Nothing relevant was retrieved.")
            yield finish("I could not find anything relevant in the indexed documents.")
            return

        # 4) generate the grounded answer
        messages = build_answer_messages(
            question, sources, sub_questions=sub_questions if len(sub_questions) > 1 else None
        )
        yield step("generate", f"Answering with {self.models.chat} from {len(sources)} passages")
        gen_kw = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
        streamed = stream_tokens and guard_obj is None
        if streamed:
            parts: list[str] = []
            async for token in self.client.stream_chat(self.models.chat, messages, **gen_kw):
                parts.append(token)
                yield AtlasEvent(type="token", text=token)
            raw = "".join(parts)
        else:
            reply = await self.client.chat(self.models.chat, messages, **gen_kw)
            raw = reply.content
            if reply.finish_reason == "length":
                warnings.append("The answer was cut off by the token limit.")
        text, _ = strip_think(raw)
        parsed = parse_citations(text.strip(), len(sources))
        text, cited = parsed.text, parsed.cited
        if parsed.invalid:
            warnings.append(f"Removed citations to non-existent sources: {parsed.invalid}")
        if not cited and not looks_like_abstention(text):
            warnings.append("The answer contains no inline citations.")

        # 5) screen the draft before releasing it
        if guard_obj is not None:
            verdict = await guard_obj.check("answer", question, text)
            verdicts.append(verdict)
            yield step(
                "guard",
                "Answer passed safety screening" if verdict.safe else "Answer blocked",
                safe=verdict.safe,
                categories=verdict.categories,
            )
            if not verdict.safe:
                warnings.append("The generated answer was withheld by the safety guard.")
                yield finish(refusal_text("answer", verdict.categories), sources=sources, blocked=True)
                return
            if stream_tokens:
                for piece in _pieces(text):
                    yield AtlasEvent(type="token", text=piece)

        # 6) verify (and optionally revise)
        report: VerificationReport | None = None
        original_text: str | None = None
        if verify:
            verifier = Verifier(
                self.client,
                self.models.chat,
                concurrency=cfg.verify_concurrency,
                max_claims=cfg.max_claims,
                lexical_only=self.mock,
            )
            yield step("verify", "Verifying claims against cited sources")
            report = await verifier.verify(text, sources)
            yield step(
                "verify",
                _verification_summary(report),
                groundedness=report.groundedness,
                supported=report.supported,
                partial=report.partial,
                unsupported=report.unsupported,
            )
            if revise and report.unsupported:
                for _ in range(cfg.revise_rounds):
                    if not report.unsupported:
                        break
                    yield step("revise", f"Revising {report.unsupported} unsupported claim(s)")
                    revised = await self._revise(question, text, report, sources, verifier)
                    if revised is None:
                        yield step("revise", "Revision was not an improvement; kept the original")
                        break
                    original_text = original_text or text
                    text, cited, report = revised
                    yield step("revise", _verification_summary(report), revised=True)
                if report.revised and guard_obj is not None:
                    verdict = await guard_obj.check("answer", question, text)
                    verdicts.append(verdict)
                    if not verdict.safe:
                        warnings.append("The revised answer was withheld by the safety guard.")
                        yield finish(
                            refusal_text("answer", verdict.categories), sources=sources, blocked=True
                        )
                        return

        by_n = {s.n: s for s in sources}
        yield finish(
            text,
            sources=sources,
            citations=[by_n[n] for n in cited if n in by_n],
            verification=report,
            original_text=original_text,
        )

    async def _revise(
        self,
        question: str,
        draft: str,
        report: VerificationReport,
        sources: list[SourceRef],
        verifier: Verifier,
    ) -> tuple[str, list[int], VerificationReport] | None:
        """One revision attempt; returns ``None`` unless groundedness strictly improves."""
        reply = await self.client.chat(
            self.models.chat,
            build_revision_messages(question, draft, report, sources),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
        )
        revised_text, _ = strip_think(reply.content)
        parsed = parse_citations(revised_text.strip(), len(sources))
        if not parsed.text:
            return None
        new_report = await verifier.verify(parsed.text, sources)
        old_score = report.groundedness or 0.0
        new_score = new_report.groundedness if new_report.groundedness is not None else 1.0
        if new_score <= old_score and new_report.unsupported >= report.unsupported:
            return None
        new_report.revised = True
        new_report.groundedness_before_revision = report.groundedness
        return parsed.text, parsed.cited, new_report


def _verification_summary(report: VerificationReport) -> str:
    if report.groundedness is None:
        return "No checkable claims in the answer"
    return (
        f"Groundedness {report.groundedness:.0%}: {report.supported} supported, "
        f"{report.partial} partial, {report.unsupported} unsupported"
    )
