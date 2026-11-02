"""Typed data structures shared across Atlas.

Internal pipeline objects are plain dataclasses; anything that crosses an API boundary (HTTP,
CLI ``--json``, eval reports) is a pydantic model so it validates and serialises cleanly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

Mode = Literal["simple", "agentic"]
Verdict = Literal["supported", "partial", "unsupported"]
MediaType = Literal["markdown", "text", "html", "pdf"]
JudgeMethod = Literal["llm", "lexical", "lexical-fallback"]
Stage = Literal["query", "answer"]

HEADING_SEP = " > "

_WS_RE = re.compile(r"\s+")


class AtlasError(Exception):
    """Base class for Atlas-specific errors (bad input, store problems, ...)."""


class StoreError(AtlasError):
    """The SQLite store is inconsistent with the request (dimension / embedding mismatch...)."""


class LoaderError(AtlasError):
    """A file could not be read or parsed."""


def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return _WS_RE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    """Stable hash of whitespace-normalised text (used for chunk de-duplication)."""
    return hashlib.sha256(normalize_ws(text).encode("utf-8")).hexdigest()


def format_location(name: str, heading_path: tuple[str, ...] | list[str], page: int | None) -> str:
    """Human-readable source label: ``file.md > Heading > Sub (p. 3)``."""
    parts = [name, *heading_path]
    label = HEADING_SEP.join(parts)
    return f"{label} (p. {page})" if page is not None else label


# --------------------------------------------------------------------------- ingestion types


@dataclass(frozen=True)
class Section:
    """A contiguous run of document text with its heading breadcrumb (and PDF page)."""

    text: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None


@dataclass
class LoadedDocument:
    """A parsed file, ready to be chunked."""

    path: str  # unique key within a collection (absolute path, or ``upload://name``)
    name: str  # display name (relative to the ingest root)
    title: str
    media_type: MediaType
    content_hash: str  # sha256 of the raw bytes
    size: int
    sections: list[Section]


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit produced by the chunker."""

    text: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    ord: int = 0
    section_start: bool = field(default=False, compare=False)

    @property
    def indexed_text(self) -> str:
        """Text that is embedded / BM25-indexed: the heading breadcrumb plus the body."""
        return indexed_text(self.heading_path, self.text)

    @property
    def hash(self) -> str:
        return content_hash(self.text)


def indexed_text(heading_path: tuple[str, ...], text: str) -> str:
    if not heading_path:
        return text
    return f"{HEADING_SEP.join(heading_path)}\n{text}"


@dataclass
class StoredChunk:
    """A chunk row loaded from the store (joined with its document)."""

    id: int
    collection: str
    doc_id: int
    doc_path: str
    doc_name: str
    doc_title: str
    ord: int
    text: str
    heading_path: tuple[str, ...]
    page: int | None

    @property
    def indexed_text(self) -> str:
        return indexed_text(self.heading_path, self.text)

    @property
    def location(self) -> str:
        return format_location(self.doc_name, self.heading_path, self.page)


@dataclass
class DocRecord:
    id: int
    collection: str
    path: str
    name: str
    title: str
    media_type: str
    content_hash: str
    chunk_sig: str
    size: int
    n_chunks: int
    ingested_at: float


@dataclass
class CollectionInfo:
    name: str
    documents: int
    chunks: int
    embed_model: str | None
    dim: int | None
    created_at: float


# --------------------------------------------------------------------------- API types


class FileReport(BaseModel):
    """Outcome of ingesting one file."""

    path: str
    name: str
    status: Literal["added", "updated", "unchanged", "skipped", "error"]
    chunks: int = 0
    duplicates: int = 0
    embedded: int = 0
    reused: int = 0
    message: str = ""


class IngestReport(BaseModel):
    collection: str
    files: list[FileReport] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    elapsed_s: float = 0.0

    def _count(self, status: str) -> int:
        return sum(1 for f in self.files if f.status == status)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def added(self) -> int:
        return self._count("added")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def updated(self) -> int:
        return self._count("updated")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unchanged(self) -> int:
        return self._count("unchanged")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def errors(self) -> int:
        return self._count("error")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chunks(self) -> int:
        return sum(f.chunks for f in self.files)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duplicates(self) -> int:
        return sum(f.duplicates for f in self.files)


class SourceRef(BaseModel):
    """One numbered context passage handed to the model (and shown to the user)."""

    n: int = Field(description="1-based citation number used in the answer text")
    chunk_id: int
    source: str = Field(description="Document display name")
    path: str
    title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    page: int | None = None
    text: str
    score: float = 0.0
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        return format_location(self.source, self.heading_path, self.page)


class TraceStep(BaseModel):
    """One step of the pipeline, surfaced for transparency (agent plan, searches, ...)."""

    kind: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class ClaimCheck(BaseModel):
    """Verification outcome for one claim (sentence) of the answer."""

    id: int
    text: str
    cited: list[int] = Field(default_factory=list)
    verdict: Verdict
    evidence: str = ""
    reason: str = ""
    evidence_found: bool | None = None
    uncited: bool = False
    method: JudgeMethod = "llm"


class VerificationReport(BaseModel):
    claims: list[ClaimCheck] = Field(default_factory=list)
    groundedness: float | None = Field(
        default=None,
        description="supported=1, partial=0.5, unsupported=0, averaged; None if no claims",
    )
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    method: str = "llm"
    truncated: bool = False
    revised: bool = False
    groundedness_before_revision: float | None = None

    @property
    def unsupported_claims(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.verdict == "unsupported"]

    @property
    def weak_claims(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.verdict != "supported"]


class GuardVerdict(BaseModel):
    """Result of screening text with the safety model."""

    stage: Stage
    safe: bool
    categories: list[str] = Field(default_factory=list)
    uncertain: bool = False
    skipped: bool = False
    raw: str = ""


class Answer(BaseModel):
    """Everything Atlas knows about one answered question."""

    question: str
    text: str
    mode: Mode = "simple"
    collection: str = "default"
    citations: list[SourceRef] = Field(
        default_factory=list, description="Sources actually cited in the answer text"
    )
    sources: list[SourceRef] = Field(
        default_factory=list, description="All context passages the model saw, in citation order"
    )
    verification: VerificationReport | None = None
    guard: list[GuardVerdict] = Field(default_factory=list)
    blocked: bool = False
    original_text: str | None = Field(
        default=None, description="Answer before auto-revision, when it was revised"
    )
    trace: list[TraceStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model: str = ""
    latency_s: float = 0.0
    usage: dict[str, Any] = Field(default_factory=dict)

    @property
    def groundedness(self) -> float | None:
        return self.verification.groundedness if self.verification else None


class AtlasEvent(BaseModel):
    """One event of a streamed ``ask``: ``trace`` | ``sources`` | ``token`` | ``final``."""

    type: Literal["trace", "sources", "token", "final"]
    text: str = ""
    step: TraceStep | None = None
    sources: list[SourceRef] | None = None
    answer: Answer | None = None
