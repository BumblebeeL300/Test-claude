"""Fixtures for the Atlas suite: example corpus paths, Atlas instances and a scripted fake client.

The shared ``tests/conftest.py`` already provides ``mock_client`` (offline NIMClient), hermetic env
and an empty temp cwd, so everything here must use absolute paths into the repository.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from nimlab.atlas import Atlas
from nimlab.core import ChatResult, NIMClient, NIMJSONError, RerankResult, Usage, UsageTracker
from nimlab.core import mock as core_mock

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "examples" / "atlas_corpus"
QA_FILE = REPO / "examples" / "atlas" / "qa.json"


class FakeClient:
    """Scripted stand-in for :class:`NIMClient`.

    Embeddings and reranking reuse the deterministic mock backend (so retrieval is real) while chat
    and ``chat_json`` replies are scripted by the test. ``mock=False`` makes Atlas believe it talks
    to a live endpoint, which switches on LLM planning, LLM verification and the safety guard.
    """

    def __init__(
        self,
        chat: str | list[str] | Callable[[str, list[dict[str, Any]]], str] | None = None,
        json: dict[str, Any] | None = None,
        *,
        mock: bool = False,
    ) -> None:
        self.mock = mock
        self.config = SimpleNamespace(mock=mock)
        self.usage = UsageTracker()
        self._chat = chat if chat is not None else "Scripted answer."
        self._json = json or {}
        self.chat_calls: list[dict[str, Any]] = []
        self.json_calls: list[dict[str, Any]] = []
        self.embed_calls: list[tuple[str, int]] = []
        self.rerank_calls = 0
        self.embed_error: Exception | None = None
        self.rerank_error: Exception | None = None

    # -- chat
    def _next_text(self, model: str, messages: list[dict[str, Any]]) -> str:
        script = self._chat
        if callable(script):
            return script(model, messages)
        if isinstance(script, list):
            return script.pop(0) if len(script) > 1 else script[0]
        return script

    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        self.chat_calls.append({"model": model, "messages": messages, **kw})
        text = self._next_text(model, messages)
        self.usage.record(model, Usage(prompt_tokens=10, completion_tokens=5), 0.0)
        return ChatResult(model=model, content=text, finish_reason=kw.get("finish", "stop"))

    async def stream_chat(
        self, model: str, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        self.chat_calls.append({"model": model, "messages": messages, "stream": True, **kw})
        text = self._next_text(model, messages)
        self.usage.record(model, Usage(prompt_tokens=10, completion_tokens=5), 0.0)
        for piece in re.findall(r"\s*\S+\s?", text):
            yield piece

    async def chat_json(
        self, model: str, messages: list[dict[str, Any]], schema: type[BaseModel], **kw: Any
    ) -> Any:
        self.json_calls.append({"model": model, "messages": messages, "schema": schema.__name__})
        self.usage.record(model, Usage(prompt_tokens=10, completion_tokens=5), 0.0)
        handler = self._json.get(schema.__name__)
        if handler is None:
            raise NIMJSONError(f"no script for {schema.__name__}", raw="", model=model)
        value = handler(messages) if callable(handler) else handler
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, schema) else schema.model_validate(value)

    # -- retrieval models (real offline maths)
    async def embed(self, model: str, texts: list[str], *, input_type: str = "passage") -> Any:
        self.embed_calls.append((input_type, len(texts)))
        if self.embed_error is not None:
            raise self.embed_error
        self.usage.record(model, Usage(prompt_tokens=len(texts)), 0.0)
        return core_mock.mock_embeddings(texts)

    async def rerank(
        self, model: str, query: str, passages: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        self.rerank_calls += 1
        if self.rerank_error is not None:
            raise self.rerank_error
        scores = core_mock.mock_rerank_scores(query, passages)
        order = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
        return [RerankResult(index=i, score=scores[i], text=passages[i]) for i in order[:top_n]]


@pytest.fixture
def corpus_dir() -> Path:
    return CORPUS


@pytest.fixture
def qa_file() -> Path:
    return QA_FILE


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "atlas.db"


@pytest.fixture
def fake_client() -> type[FakeClient]:
    """The :class:`FakeClient` class; instantiate with scripts inside the test."""
    return FakeClient


@pytest.fixture
async def atlas(mock_client: NIMClient, db_path: Path) -> AsyncIterator[Atlas]:
    instance = Atlas(mock_client, db_path)
    yield instance
    instance.close()


@pytest.fixture
async def indexed(atlas: Atlas, corpus_dir: Path) -> Atlas:
    """An Atlas (offline mock backend) with the example corpus ingested into ``default``."""
    report = await atlas.ingest(corpus_dir)
    assert report.errors == 0
    return atlas


@pytest.fixture
def make_pdf() -> Callable[[list[list[str]]], bytes]:
    """Build a tiny text PDF: one list of lines per page."""

    def build(pages: list[list[str]]) -> bytes:
        n = len(pages)
        page_ids = [4 + 2 * i for i in range(n)]
        content_ids = [5 + 2 * i for i in range(n)]
        objs: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
        objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        for i, lines in enumerate(pages):
            objs.append(
                (
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Contents {content_ids[i]} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
                ).encode()
            )
            ops = ["BT /F1 12 Tf 14 TL 72 720 Td"]
            for j, line in enumerate(lines):
                esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                ops.append(f"({esc}) Tj" + (" T*" if j < len(lines) - 1 else ""))
            ops.append("ET")
            stream = "\n".join(ops).encode("latin-1")
            objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        out = bytearray(b"%PDF-1.4\n")
        offsets: list[int] = []
        for i, body in enumerate(objs, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode()
        out += (
            f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
        ).encode()
        return bytes(out)

    return build
