"""SQLite store: round trips, dedup, collections, dimension guards and the in-memory indexes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nimlab.atlas.models import Chunk, LoadedDocument, Section, StoreError
from nimlab.atlas.store import AtlasStore, NewChunk, normalize_rows


def make_doc(path: str = "/docs/a.md", name: str = "a.md", sha: str = "h1") -> LoadedDocument:
    return LoadedDocument(
        path=path,
        name=name,
        title="Doc A",
        media_type="markdown",
        content_hash=sha,
        size=123,
        sections=[Section("x")],
    )


def unit(*values: float) -> np.ndarray:
    return normalize_rows(np.array(values, dtype=np.float32))[0]


def new_chunks(*specs: tuple[str, tuple[str, ...], int | None, np.ndarray]) -> list[NewChunk]:
    return [
        NewChunk(Chunk(text, path, page, ord=i), vec)
        for i, (text, path, page, vec) in enumerate(specs)
    ]


@pytest.fixture
def store(tmp_path: Path):
    s = AtlasStore(tmp_path / "t.db")
    yield s
    s.close()


def test_round_trip_preserves_text_metadata_and_float32_vectors(store: AtlasStore):
    store.ensure_collection("c", "emb-1")
    vec = unit(0.1, 0.2, 0.3, 0.4)
    chunks = new_chunks(("Hello world", ("Guide", "Intro"), 3, vec))
    assert store.replace_document("c", make_doc(), chunks, "sig") == 1
    (stored,) = store.get_chunks([store.index("c").chunks[0].id])
    assert stored.text == "Hello world"
    assert stored.heading_path == ("Guide", "Intro") and stored.page == 3
    assert stored.doc_name == "a.md" and stored.doc_title == "Doc A"
    assert stored.location == "a.md > Guide > Intro (p. 3)"
    saved = store.document_vectors("c", "/docs/a.md")[chunks[0].hash]
    assert saved.dtype == np.float32
    assert np.array_equal(saved, vec.astype(np.float32))


def test_data_persists_across_reopen(tmp_path: Path):
    path = tmp_path / "p.db"
    with AtlasStore(path) as s:
        s.ensure_collection("c", "m")
        s.replace_document("c", make_doc(), new_chunks(("persist me", (), None, unit(1, 0))), "sig")
    with AtlasStore(path) as s2:
        assert s2.get_collection("c").chunks == 1  # type: ignore[union-attr]
        assert s2.index("c").chunks[0].text == "persist me"


def test_replace_document_swaps_chunks_atomically(store: AtlasStore):
    store.ensure_collection("c", "m")
    store.replace_document(
        "c",
        make_doc(),
        new_chunks(("old one", (), None, unit(1, 0)), ("old two", (), None, unit(0, 1))),
        "s",
    )
    store.replace_document(
        "c", make_doc(sha="h2"), new_chunks(("new only", (), None, unit(1, 1))), "s"
    )
    assert [c.text for c in store.index("c").chunks] == ["new only"]
    (doc,) = store.list_documents("c")
    assert doc.content_hash == "h2" and doc.n_chunks == 1


def test_replace_rolls_back_on_failure(store: AtlasStore):
    store.ensure_collection("c", "m")
    store.replace_document("c", make_doc(), new_chunks(("keep me", (), None, unit(1, 0))), "s")
    bad = new_chunks(("x", (), None, unit(1, 0)), ("y", (), None, unit(1, 0, 0)))  # mixed dims
    with pytest.raises(StoreError, match="inconsistent"):
        store.replace_document("c", make_doc(sha="h2"), bad, "s")
    assert [c.text for c in store.index("c").chunks] == ["keep me"]
    assert store.get_document("c", "/docs/a.md").content_hash == "h1"  # type: ignore[union-attr]


def test_duplicate_chunk_text_is_stored_once_per_collection(store: AtlasStore):
    store.ensure_collection("c", "m")
    store.ensure_collection("d", "m")
    dup = new_chunks(("same text", (), None, unit(1, 0)), ("same   text", (), None, unit(1, 0)))
    assert store.replace_document("c", make_doc(), dup, "s") == 1  # whitespace-normalised hash
    assert (
        store.replace_document("d", make_doc(), dup[:1], "s") == 1
    )  # other collection is separate
    assert store.known_chunk_hashes("c") == {dup[0].hash: "/docs/a.md"}


def test_collections_are_isolated_and_listed_with_counts(store: AtlasStore):
    for name in ("alpha", "beta"):
        store.ensure_collection(name, "m")
    store.replace_document("alpha", make_doc(), new_chunks(("a1", (), None, unit(1, 0))), "s")
    store.replace_document(
        "beta",
        make_doc(path="/b.md", name="b.md"),
        new_chunks(("b1", (), None, unit(0, 1)), ("b2", (), None, unit(1, 1))),
        "s",
    )
    info = {c.name: c for c in store.collections()}
    assert (info["alpha"].documents, info["alpha"].chunks) == (1, 1)
    assert (info["beta"].documents, info["beta"].chunks, info["beta"].dim) == (1, 2, 2)
    assert [c.text for c in store.index("alpha").chunks] == ["a1"]
    assert store.get_collection("missing") is None


def test_delete_document_and_collection_cascade(store: AtlasStore):
    store.ensure_collection("c", "m")
    store.replace_document("c", make_doc(), new_chunks(("t1", (), None, unit(1, 0))), "s")
    assert store.delete_document("c", "/docs/a.md") is True
    assert store.delete_document("c", "/docs/a.md") is False
    assert len(store.index("c")) == 0
    store.replace_document("c", make_doc(), new_chunks(("t2", (), None, unit(1, 0))), "s")
    assert store.delete_collection("c") is True
    assert store.collections() == [] and store.delete_collection("c") is False


def test_embedding_model_and_dimension_are_enforced(store: AtlasStore):
    store.ensure_collection("c", "model-a")
    store.replace_document("c", make_doc(), new_chunks(("t", (), None, unit(1, 0))), "s")
    with pytest.raises(StoreError, match="model-b"):
        store.ensure_collection("c", "model-b")
    store.ensure_collection("c", "model-a")  # same model is fine
    with pytest.raises(StoreError, match="dimension"):
        store.replace_document(
            "c",
            make_doc(path="/other.md", name="o.md"),
            new_chunks(("u", (), None, unit(1, 0, 0))),
            "s",
        )


def test_an_empty_collection_may_switch_embedding_model(store: AtlasStore):
    store.ensure_collection("c", "model-a")
    info = store.ensure_collection("c", "model-b")
    assert info.embed_model == "model-b"


def test_dense_top_ranks_by_cosine_and_drops_non_positive(store: AtlasStore):
    store.ensure_collection("c", "m")
    store.replace_document(
        "c",
        make_doc(),
        new_chunks(
            ("east", (), None, unit(1, 0)),
            ("north-east", (), None, unit(1, 1)),
            ("north", (), None, unit(0, 1)),
            ("west", (), None, unit(-1, 0)),
        ),
        "s",
    )
    idx = store.index("c")
    hits = idx.dense_top(np.array([1.0, 0.2], dtype=np.float32), 10)
    assert [idx.chunks[p].text for p, _ in hits] == ["east", "north-east", "north"]
    assert hits[0][1] == pytest.approx(float(unit(1, 0.2) @ unit(1, 0)))
    assert len(idx.dense_top(np.array([1.0, 0.0]), 2)) == 2
    with pytest.raises(StoreError, match="dimension"):
        idx.dense_top(np.array([1.0, 0.0, 0.0]), 3)


def test_bm25_index_uses_heading_path_text(store: AtlasStore):
    store.ensure_collection("c", "m")
    store.replace_document(
        "c",
        make_doc(),
        new_chunks(
            ("Restart the pods.", ("Runbook", "Allocator"), None, unit(1, 0)),
            ("Unrelated text about colours.", ("Style",), None, unit(0, 1)),
        ),
        "s",
    )
    idx = store.index("c")
    top = idx.bm25_top("allocator restart", 5)
    assert idx.chunks[top[0][0]].heading_path == ("Runbook", "Allocator")


def test_index_cache_is_reused_and_invalidated_by_writes(store: AtlasStore):
    store.ensure_collection("c", "m")
    store.replace_document("c", make_doc(), new_chunks(("one", (), None, unit(1, 0))), "s")
    first = store.index("c")
    assert store.index("c") is first
    store.replace_document(
        "c", make_doc(path="/b.md", name="b.md"), new_chunks(("two", (), None, unit(0, 1))), "s"
    )
    second = store.index("c")
    assert second is not first and len(second) == 2


def test_update_vectors_and_set_embedding(store: AtlasStore):
    store.ensure_collection("c", "m")
    store.replace_document("c", make_doc(), new_chunks(("t", (), None, unit(1, 0))), "s")
    ((cid, text),) = store.iter_chunk_texts("c")
    store.update_vectors("c", {cid: unit(0, 0, 1)})
    store.set_embedding("c", "m2", 3)
    assert store.index("c").dim == 3
    info = store.get_collection("c")
    assert info is not None and (info.embed_model, info.dim) == ("m2", 3)
    assert text == "t"


def test_stats_and_memory_database(tmp_path: Path):
    with AtlasStore(":memory:") as mem:
        mem.ensure_collection("c", "m")
        assert mem.stats()["collections"][0]["name"] == "c"
    with AtlasStore(tmp_path / "nested" / "dir" / "x.db") as s:
        assert (tmp_path / "nested" / "dir" / "x.db").exists()
        assert s.stats()["db_bytes"] > 0


def test_normalize_rows_handles_zero_vectors():
    out = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out[0].tolist() == pytest.approx([0.6, 0.8])
    assert out[1].tolist() == [0.0, 0.0]
