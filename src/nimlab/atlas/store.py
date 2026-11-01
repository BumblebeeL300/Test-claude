"""SQLite persistence plus the in-memory dense (numpy) and BM25 indexes built from it.

Tables: ``collections`` (namespaces + the embedding model that fills them), ``documents`` (one row
per source file, with its content hash for incremental ingest) and ``chunks`` (text, heading path,
page and the float32 vector as a BLOB). Chunk text is unique per collection (content-hash dedup).

Retrieval is brute force: all vectors of a collection are loaded into one L2-normalised float32
matrix (cosine == dot product) next to a BM25 index over the same chunks. Both are cached per
collection and rebuilt lazily after any write.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25Index
from .models import (
    Chunk,
    CollectionInfo,
    DocRecord,
    LoadedDocument,
    StoredChunk,
    StoreError,
)
from .tokens import tokenize

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS collections (
    name TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    embed_model TEXT,
    dim INTEGER
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    media_type TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    chunk_sig TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    n_chunks INTEGER NOT NULL DEFAULT 0,
    ingested_at REAL NOT NULL,
    UNIQUE (collection, path)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    heading_path TEXT NOT NULL DEFAULT '[]',
    page INTEGER,
    vector BLOB NOT NULL,
    UNIQUE (collection, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);
"""


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows (zero rows stay zero); returns float32."""
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1.0, norms)


def vector_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)


@dataclass
class NewChunk:
    """A chunk ready to be written, with its (normalised) vector."""

    chunk: Chunk
    vector: np.ndarray

    @property
    def hash(self) -> str:
        return self.chunk.hash


class CollectionIndex:
    """Immutable in-memory snapshot of one collection: chunks, dense matrix and BM25 index."""

    def __init__(
        self,
        chunks: list[StoredChunk],
        matrix: np.ndarray,
        bm25: BM25Index,
    ) -> None:
        self.chunks = chunks
        self.matrix = matrix  # (n, dim) float32, rows L2-normalised
        self.bm25 = bm25
        self.id_to_pos = {c.id: i for i, c in enumerate(chunks)}

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.ndim == 2 else 0

    def dense_top(self, query_vec: np.ndarray, n: int) -> list[tuple[int, float]]:
        """Top-``n`` ``(position, cosine)`` with positive similarity, best first."""
        if not len(self.chunks):
            return []
        q = normalize_rows(query_vec)[0]
        if q.shape[0] != self.dim:
            raise StoreError(
                f"query vector has dimension {q.shape[0]} but the collection was embedded with "
                f"dimension {self.dim}; re-embed the collection with the current model"
            )
        sims = self.matrix @ q
        n = min(n, sims.shape[0])
        if n <= 0:
            return []
        top = np.argpartition(-sims, n - 1)[:n] if n < sims.shape[0] else np.arange(sims.shape[0])
        order = top[np.lexsort((top, -sims[top]))]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0.0]

    def bm25_top(self, query: str, n: int) -> list[tuple[int, float]]:
        return self.bm25.search(tokenize(query), n)


class AtlasStore:
    """Thread-safe SQLite store. ``path=":memory:"`` gives a throwaway database."""

    def __init__(self, path: str | Path = ".nimlab/atlas.db", *, k1: float = 1.5, b: float = 0.75):
        self.path = str(path)
        self.k1 = k1
        self.b = b
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        self._indexes: dict[str, CollectionIndex] = {}
        self._init_schema()

    # ------------------------------------------------------------------ plumbing
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) > SCHEMA_VERSION:
                raise StoreError(
                    f"database schema v{row['value']} is newer than this Atlas (v{SCHEMA_VERSION})"
                )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._indexes.clear()
            self._conn.close()

    def __enter__(self) -> AtlasStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _invalidate(self, collection: str | None = None) -> None:
        if collection is None:
            self._indexes.clear()
        else:
            self._indexes.pop(collection, None)

    # ------------------------------------------------------------------ collections
    def ensure_collection(self, name: str, embed_model: str | None = None) -> CollectionInfo:
        """Create the collection if needed; refuse to mix embedding models in one collection."""
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM collections WHERE name=?", (name,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO collections(name, created_at, embed_model) VALUES (?,?,?)",
                    (name, time.time(), embed_model),
                )
            else:
                n_chunks = conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE collection=?", (name,)
                ).fetchone()[0]
                if row["embed_model"] and embed_model and row["embed_model"] != embed_model:
                    if n_chunks:
                        raise StoreError(
                            f"collection {name!r} was embedded with {row['embed_model']!r}, not "
                            f"{embed_model!r}; run reindex (or use a new collection)"
                        )
                if embed_model and (not row["embed_model"] or not n_chunks):
                    conn.execute(
                        "UPDATE collections SET embed_model=?, dim=CASE WHEN ?=0 THEN NULL "
                        "ELSE dim END WHERE name=?",
                        (embed_model, n_chunks, name),
                    )
        info = self.get_collection(name)
        assert info is not None
        return info

    def get_collection(self, name: str) -> CollectionInfo | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM documents d WHERE d.collection=c.name) AS docs,"
                " (SELECT COUNT(*) FROM chunks k WHERE k.collection=c.name) AS chunks "
                "FROM collections c WHERE c.name=?",
                (name,),
            ).fetchone()
        return self._collection_from_row(row) if row else None

    def collections(self) -> list[CollectionInfo]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM documents d WHERE d.collection=c.name) AS docs,"
                " (SELECT COUNT(*) FROM chunks k WHERE k.collection=c.name) AS chunks "
                "FROM collections c ORDER BY c.name"
            ).fetchall()
        return [self._collection_from_row(r) for r in rows]

    @staticmethod
    def _collection_from_row(row: sqlite3.Row) -> CollectionInfo:
        return CollectionInfo(
            name=row["name"],
            documents=row["docs"],
            chunks=row["chunks"],
            embed_model=row["embed_model"],
            dim=row["dim"],
            created_at=row["created_at"],
        )

    def delete_collection(self, name: str) -> bool:
        with self._tx() as conn:
            conn.execute("DELETE FROM chunks WHERE collection=?", (name,))
            conn.execute("DELETE FROM documents WHERE collection=?", (name,))
            cur = conn.execute("DELETE FROM collections WHERE name=?", (name,))
        self._invalidate(name)
        return cur.rowcount > 0

    def set_embedding(self, collection: str, embed_model: str, dim: int) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE collections SET embed_model=?, dim=? WHERE name=?",
                (embed_model, dim, collection),
            )
        self._invalidate(collection)

    # ------------------------------------------------------------------ documents
    @staticmethod
    def _doc_from_row(row: sqlite3.Row) -> DocRecord:
        return DocRecord(
            id=row["id"],
            collection=row["collection"],
            path=row["path"],
            name=row["name"],
            title=row["title"],
            media_type=row["media_type"],
            content_hash=row["content_hash"],
            chunk_sig=row["chunk_sig"],
            size=row["size"],
            n_chunks=row["n_chunks"],
            ingested_at=row["ingested_at"],
        )

    def get_document(self, collection: str, path: str) -> DocRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE collection=? AND path=?", (collection, path)
            ).fetchone()
        return self._doc_from_row(row) if row else None

    def list_documents(self, collection: str | None = None) -> list[DocRecord]:
        with self._lock:
            if collection is None:
                rows = self._conn.execute("SELECT * FROM documents ORDER BY collection, name")
            else:
                rows = self._conn.execute(
                    "SELECT * FROM documents WHERE collection=? ORDER BY name", (collection,)
                )
            return [self._doc_from_row(r) for r in rows.fetchall()]

    def delete_document(self, collection: str, path: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute(
                "DELETE FROM documents WHERE collection=? AND path=?", (collection, path)
            )
        self._invalidate(collection)
        return cur.rowcount > 0

    def known_chunk_hashes(self, collection: str) -> dict[str, str]:
        """``{chunk_hash: document_path}`` for every chunk currently stored in the collection."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT k.content_hash AS h, d.path AS p FROM chunks k "
                "JOIN documents d ON d.id = k.doc_id WHERE k.collection=?",
                (collection,),
            ).fetchall()
        return {r["h"]: r["p"] for r in rows}

    def document_vectors(self, collection: str, path: str) -> dict[str, np.ndarray]:
        """``{chunk_hash: vector}`` of a stored document, used to avoid re-embedding."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT k.content_hash AS h, k.vector AS v FROM chunks k "
                "JOIN documents d ON d.id = k.doc_id WHERE d.collection=? AND d.path=?",
                (collection, path),
            ).fetchall()
        return {r["h"]: blob_to_vector(r["v"]) for r in rows}

    def replace_document(
        self,
        collection: str,
        doc: LoadedDocument,
        chunks: Sequence[NewChunk],
        chunk_sig: str,
    ) -> int:
        """Atomically replace a document and its chunks. Returns the number of chunks stored."""
        if not chunks:
            raise StoreError("refusing to store a document without chunks")
        dim = int(chunks[0].vector.shape[0])
        if any(int(c.vector.shape[0]) != dim for c in chunks):
            raise StoreError("chunk vectors have inconsistent dimensions")
        with self._tx() as conn:
            col = conn.execute(
                "SELECT dim FROM collections WHERE name=?", (collection,)
            ).fetchone()
            if col is None:
                raise StoreError(f"unknown collection {collection!r}")
            # Count chunks that survive this replacement to decide whether the dim is binding.
            others = conn.execute(
                "SELECT COUNT(*) FROM chunks k JOIN documents d ON d.id=k.doc_id "
                "WHERE k.collection=? AND d.path<>?",
                (collection, doc.path),
            ).fetchone()[0]
            if col["dim"] and others and col["dim"] != dim:
                raise StoreError(
                    f"embedding dimension {dim} does not match collection {collection!r} "
                    f"(dimension {col['dim']}); re-embed the collection with the current model"
                )
            conn.execute(
                "DELETE FROM documents WHERE collection=? AND path=?", (collection, doc.path)
            )
            cur = conn.execute(
                "INSERT INTO documents(collection, path, name, title, media_type, content_hash, "
                "chunk_sig, size, n_chunks, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    collection,
                    doc.path,
                    doc.name,
                    doc.title,
                    doc.media_type,
                    doc.content_hash,
                    chunk_sig,
                    doc.size,
                    0,
                    time.time(),
                ),
            )
            doc_id = cur.lastrowid
            stored = 0
            for nc in chunks:
                ins = conn.execute(
                    "INSERT OR IGNORE INTO chunks(collection, doc_id, ord, text, content_hash, "
                    "heading_path, page, vector) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        collection,
                        doc_id,
                        nc.chunk.ord,
                        nc.chunk.text,
                        nc.hash,
                        json.dumps(list(nc.chunk.heading_path)),
                        nc.chunk.page,
                        vector_to_blob(nc.vector),
                    ),
                )
                stored += ins.rowcount
            conn.execute("UPDATE documents SET n_chunks=? WHERE id=?", (stored, doc_id))
            conn.execute("UPDATE collections SET dim=? WHERE name=?", (dim, collection))
        self._invalidate(collection)
        return stored

    # ------------------------------------------------------------------ chunk access
    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> StoredChunk:
        return StoredChunk(
            id=row["id"],
            collection=row["collection"],
            doc_id=row["doc_id"],
            doc_path=row["path"],
            doc_name=row["name"],
            doc_title=row["title"],
            ord=row["ord"],
            text=row["text"],
            heading_path=tuple(json.loads(row["heading_path"])),
            page=row["page"],
        )

    _CHUNK_COLS = (
        "k.id, k.collection, k.doc_id, k.ord, k.text, k.heading_path, k.page, "
        "d.path, d.name, d.title"
    )
    _CHUNK_FROM = "FROM chunks k JOIN documents d ON d.id = k.doc_id"

    def get_chunks(self, ids: Sequence[int]) -> list[StoredChunk]:
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._CHUNK_COLS} {self._CHUNK_FROM} WHERE k.id IN ({marks})",
                tuple(ids),
            ).fetchall()
        by_id = {r["id"]: self._chunk_from_row(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def iter_chunk_texts(self, collection: str) -> list[tuple[int, str]]:
        """``(chunk_id, indexed_text)`` for every chunk, for re-embedding."""
        chunks = self._load_chunks(collection)
        return [(c.id, c.indexed_text) for c in chunks]

    def update_vectors(self, collection: str, vectors: dict[int, np.ndarray]) -> None:
        with self._tx() as conn:
            conn.executemany(
                "UPDATE chunks SET vector=? WHERE id=? AND collection=?",
                [(vector_to_blob(v), cid, collection) for cid, v in vectors.items()],
            )
        self._invalidate(collection)

    def _load_chunks(self, collection: str) -> list[StoredChunk]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._CHUNK_COLS} {self._CHUNK_FROM} WHERE k.collection=? ORDER BY k.id",
                (collection,),
            ).fetchall()
        return [self._chunk_from_row(r) for r in rows]

    # ------------------------------------------------------------------ indexes
    def index(self, collection: str) -> CollectionIndex:
        """Cached dense + BM25 snapshot of a collection (rebuilt after writes)."""
        with self._lock:
            cached = self._indexes.get(collection)
            if cached is not None:
                return cached
            rows = self._conn.execute(
                f"SELECT {self._CHUNK_COLS}, k.vector {self._CHUNK_FROM} "
                "WHERE k.collection=? ORDER BY k.id",
                (collection,),
            ).fetchall()
            chunks = [self._chunk_from_row(r) for r in rows]
            if rows:
                matrix = normalize_rows(np.vstack([blob_to_vector(r["vector"]) for r in rows]))
            else:
                matrix = np.zeros((0, 0), dtype=np.float32)
            bm25 = BM25Index.build([tokenize(c.indexed_text) for c in chunks], self.k1, self.b)
            built = CollectionIndex(chunks, matrix, bm25)
            self._indexes[collection] = built
            return built

    def configure_bm25(self, k1: float, b: float) -> None:
        """Change BM25 parameters; cached indexes are rebuilt on next use."""
        if (k1, b) != (self.k1, self.b):
            self.k1, self.b = k1, b
            self._invalidate()

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, object]:
        with self._lock:
            size = 0
            if self.path != ":memory:":
                try:
                    size = Path(self.path).stat().st_size
                except OSError:
                    size = 0
        cols = self.collections()
        return {
            "db_path": self.path,
            "db_bytes": size,
            "schema_version": SCHEMA_VERSION,
            "collections": [
                {
                    "name": c.name,
                    "documents": c.documents,
                    "chunks": c.chunks,
                    "embed_model": c.embed_model,
                    "dim": c.dim,
                }
                for c in cols
            ],
            "documents": sum(c.documents for c in cols),
            "chunks": sum(c.chunks for c in cols),
        }
