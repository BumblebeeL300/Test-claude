"""Atlas: collections, ingestion (uploads or a path) and cited answers as a run.

Ask events: ``trace`` (retrieval/verification steps), ``token`` (answer text), ``sources`` (the
retrieved passages), then ``run.done`` with the answer, citations and groundedness.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from nimlab.atlas.config import DEFAULT_COLLECTION
from nimlab.atlas.loaders import SUPPORTED_EXTENSIONS, media_type_for
from nimlab.atlas.models import FileReport, IngestReport, SourceRef, StoreError

from .runs import Run
from .state import StudioState

__all__ = [
    "MAX_UPLOAD_BYTES",
    "check_collection",
    "collections",
    "ingest_path",
    "ingest_uploads",
    "start_ask",
]

MAX_UPLOAD_BYTES = 60 * 1024 * 1024
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def check_collection(name: str | None) -> str:
    name = (name or DEFAULT_COLLECTION).strip()
    if not _COLLECTION_RE.match(name):
        raise ValueError(
            "collection names use letters, digits, '.', '_' or '-' (max 64 characters)"
        )
    return name


def collections(state: StudioState) -> list[dict[str, Any]]:
    out = []
    for c in state.atlas().collections():
        docs = state.atlas().documents(c.name)
        out.append(
            {
                "name": c.name,
                "documents": c.documents,
                "chunks": c.chunks,
                "embed_model": c.embed_model,
                "dim": c.dim,
                "created_at": c.created_at,
                "files": [
                    {
                        "name": d.name,
                        "title": d.title,
                        "type": d.media_type,
                        "chunks": d.n_chunks,
                        "size": d.size,
                    }
                    for d in docs
                ],
            }
        )
    return out


def _report(rep: IngestReport) -> dict[str, Any]:
    data = rep.model_dump(mode="json")
    data["ok"] = rep.errors == 0
    return data


def _needs_reindex(exc: StoreError) -> bool:
    text = str(exc)
    return "reindex" in text or "re-embed" in text


async def _with_reindex(state: StudioState, collection: str, call: Any) -> Any:
    """Run ``call()``; if another backend embedded the collection, re-embed once and retry."""
    try:
        return await call()
    except StoreError as exc:
        if not _needs_reindex(exc):
            raise
        await state.atlas().reindex(collection)
        return await call()


async def ingest_uploads(
    state: StudioState, files: list[tuple[str, bytes]], collection: str, force: bool = False
) -> dict[str, Any]:
    accepted = [
        (Path(n.replace("\\", "/")).name or "upload", d) for n, d in files if media_type_for(n)
    ]
    rejected = [Path(n.replace("\\", "/")).name for n, _ in files if not media_type_for(n)]
    if not accepted:
        raise ValueError(
            f"unsupported file type; supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    report = await _with_reindex(
        state, collection, lambda: state.atlas().ingest_bytes(accepted, collection, force=force)
    )
    for name in rejected:
        report.files.append(
            FileReport(
                path=f"upload://{name}", name=name, status="skipped", message="unsupported type"
            )
        )
    return _report(report)


def _existing_path(path: str) -> Path:
    target = Path(os.path.expanduser(path))
    if not target.exists():
        raise FileNotFoundError(f"no such file or folder: {target}")
    return target


async def ingest_path(
    state: StudioState, path: str, collection: str, force: bool = False
) -> dict[str, Any]:
    target = await asyncio.to_thread(_existing_path, path)
    report = await _with_reindex(
        state, collection, lambda: state.atlas().ingest(target, collection, force=force)
    )
    return _report(report)


def _source(s: SourceRef, *, full: bool) -> dict[str, Any]:
    data = s.model_dump(mode="json")
    if not full and len(data.get("text", "")) > 600:
        data["text"] = data["text"][:600].rstrip() + "…"
    return data


def start_ask(state: StudioState, question: str, collection: str, mode: str, verify: bool) -> Run:
    simulated = state.mock

    async def work(run: Run) -> dict[str, Any]:
        atlas = state.atlas()
        final = None
        for attempt in (1, 2):
            stream = atlas.ask_stream(
                question, collection, mode=mode, verify=verify, stream_tokens=True
            )
            try:
                async for ev in stream:
                    if ev.type == "token":
                        run.emit("token", {"text": ev.text})
                    elif ev.type == "trace" and ev.step is not None:
                        run.emit(
                            "trace",
                            {
                                "kind": ev.step.kind,
                                "message": ev.step.message,
                                "data": ev.step.data,
                            },
                        )
                    elif ev.type == "sources" and ev.sources is not None:
                        run.emit(
                            "sources", {"sources": [_source(s, full=False) for s in ev.sources]}
                        )
                    elif ev.type == "final":
                        final = ev.answer
                break
            except StoreError as exc:
                # The collection was embedded by another backend (e.g. mock, then a real key).
                if attempt == 2 or not _needs_reindex(exc):
                    raise
                run.emit("log", {"level": "info", "message": "Re-embedding the collection..."})
                await atlas.reindex(collection)
            finally:
                await stream.aclose()
        assert final is not None, "Atlas finished without an answer"
        ver = final.verification
        return {
            "question": final.question,
            "answer": final.text,
            "mode": final.mode,
            "collection": final.collection,
            "citations": [_source(s, full=True) for s in final.citations],
            "sources": [_source(s, full=False) for s in final.sources],
            "groundedness": final.groundedness,
            "verification": ver.model_dump(mode="json") if ver else None,
            "blocked": final.blocked,
            "warnings": final.warnings,
            "model": final.model,
            "latency_s": round(final.latency_s, 3),
            "usage": final.usage,
            "simulated": simulated,
        }

    title = question.strip()[:80]
    return state.runs.start(
        "atlas", title, work, simulated=simulated, meta={"collection": collection}
    )
