"""Ingestion: incremental re-ingest, dedup, pruning, uploads, failures and reindexing."""

from __future__ import annotations

from pathlib import Path

import pytest

from nimlab.atlas import Atlas, AtlasConfig
from nimlab.atlas.models import StoreError
from nimlab.core import NIMError

from .conftest import FakeClient


def embed_calls(atlas: Atlas) -> int:
    stats = atlas.client.usage.summary()["models"].get(atlas.models.embed, {})
    return int(stats.get("calls", 0))


async def test_ingest_corpus_adds_every_file_with_metadata(atlas: Atlas, corpus_dir: Path):
    report = await atlas.ingest(corpus_dir, "handbook")
    assert report.added == 5 and report.errors == 0 and report.chunks > 20
    assert {f.name for f in report.files} == {
        "api_guide.md",
        "architecture.md",
        "glossary.md",
        "incident_runbook.md",
        "onboarding.md",
    }
    index = atlas.store.index("handbook")
    paths = {tuple(c.heading_path) for c in index.chunks}
    assert ("Incident Runbook", "Common incidents", "Map desync") in paths
    assert index.dim == 1024
    info = atlas.store.get_collection("handbook")
    assert info is not None and info.embed_model.startswith("mock:")  # type: ignore[union-attr]


async def test_reingest_of_unchanged_files_does_nothing(atlas: Atlas, corpus_dir: Path):
    await atlas.ingest(corpus_dir)
    calls = embed_calls(atlas)
    again = await atlas.ingest(corpus_dir)
    assert again.unchanged == 5 and again.added == 0 and again.updated == 0
    assert embed_calls(atlas) == calls  # no embedding requests at all


async def test_changed_file_is_updated_and_only_new_chunks_are_embedded(
    atlas: Atlas, tmp_path: Path
):
    docs = tmp_path / "docs"
    docs.mkdir()
    body = "\n\n".join(
        f"## Section {i}\n\n" + f"Unique paragraph number {i} about robots. " * 15 for i in range(6)
    )
    (docs / "guide.md").write_text(f"# Guide\n\n{body}\n")
    (docs / "other.md").write_text("# Other\n\nSomething else entirely.\n")
    first = await atlas.ingest(docs)
    assert first.added == 2
    before = embed_calls(atlas)
    (docs / "guide.md").write_text(f"# Guide\n\n{body}\n\n## Extra\n\nA brand new paragraph.\n")
    second = await atlas.ingest(docs)
    by_name = {f.name: f for f in second.files}
    assert by_name["other.md"].status == "unchanged"
    guide = by_name["guide.md"]
    assert guide.status == "updated" and guide.embedded >= 1 and guide.reused >= 1
    assert guide.embedded < guide.chunks  # unchanged chunks keep their vectors
    assert embed_calls(atlas) == before + 1
    texts = [c.text for c in atlas.store.index("default").chunks]
    assert any("brand new paragraph" in t for t in texts)


async def test_content_hash_dedup_across_files(atlas: Atlas, tmp_path: Path):
    shared = "This paragraph is boilerplate repeated verbatim in two documents. " * 4
    unique = "Unique content that only exists in document B and nowhere else at all. " * 4
    (tmp_path / "a.md").write_text(f"# A\n\n{shared}\n")
    (tmp_path / "b.md").write_text(f"# B\n\n{shared}\n\n## Extra\n\n{unique}\n")
    report = await atlas.ingest(tmp_path)
    by = {f.name: f for f in report.files}
    assert by["a.md"].status == "added" and by["a.md"].duplicates == 0
    assert by["b.md"].duplicates == 1 and by["b.md"].chunks == 1
    texts = [c.text for c in atlas.store.index("default").chunks]
    assert sum(shared.strip() in t for t in texts) == 1


async def test_file_made_only_of_duplicates_is_skipped(atlas: Atlas, tmp_path: Path):
    (tmp_path / "a.md").write_text("# T\n\nExactly the same words in both files.\n")
    (tmp_path / "b.md").write_text("# T\n\nExactly the same words in both files.\n")
    report = await atlas.ingest(tmp_path)
    assert [f.status for f in report.files] == ["added", "skipped"]


async def test_duplicates_inside_one_file_are_collapsed(atlas: Atlas, tmp_path: Path):
    para = "Repeated paragraph that appears several times in a single document for testing."
    (tmp_path / "a.md").write_text("\n\n".join(["# T", para, "## Two", para]) + "\n")
    report = await atlas.ingest(tmp_path)
    assert report.files[0].chunks == 1 and report.duplicates >= 0
    assert len(atlas.store.index("default")) == 1


async def test_prune_removes_documents_whose_files_disappeared(atlas: Atlas, tmp_path: Path):
    (tmp_path / "keep.md").write_text("# Keep\n\nThis stays.\n")
    (tmp_path / "gone.md").write_text("# Gone\n\nThis disappears.\n")
    await atlas.ingest(tmp_path)
    (tmp_path / "gone.md").unlink()
    report = await atlas.ingest(tmp_path, prune=True)
    assert report.removed == ["gone.md"]
    assert [d.name for d in atlas.documents("default")] == ["keep.md"]


async def test_unsupported_empty_and_corrupt_files_do_not_abort_the_run(
    atlas: Atlas, tmp_path: Path, make_pdf
):
    (tmp_path / "good.md").write_text("# Good\n\nReal content here.\n")
    (tmp_path / "empty.md").write_text("   \n")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "scan.pdf").write_bytes(make_pdf([[]]))
    (tmp_path / "pic.png").write_bytes(b"png")  # ignored by discovery
    report = await atlas.ingest(tmp_path)
    status = {f.name: f.status for f in report.files}
    assert status == {
        "broken.pdf": "error",
        "empty.md": "error",
        "good.md": "added",
        "scan.pdf": "error",
    }
    assert report.errors == 3 and len(atlas.store.index("default")) == 1


async def test_pdf_ingest_keeps_page_numbers(atlas: Atlas, tmp_path: Path, make_pdf):
    (tmp_path / "manual.pdf").write_bytes(
        make_pdf([["Page one talks about batteries."], ["Page two talks about docking stations."]])
    )
    await atlas.ingest(tmp_path)
    pages = sorted(c.page for c in atlas.store.index("default").chunks)
    assert pages == [1, 2]
    hit = (await atlas.retrieve("docking stations", k=1))[0]
    assert hit.page == 2 and "(p. 2)" in hit.label


async def test_ingest_single_file_and_missing_path(atlas: Atlas, corpus_dir: Path):
    report = await atlas.ingest(corpus_dir / "glossary.md")
    assert [f.name for f in report.files] == ["glossary.md"]
    with pytest.raises(Exception, match="does not exist"):
        await atlas.ingest(corpus_dir / "missing.md")


async def test_ingest_bytes_stores_uploads_under_a_virtual_path(atlas: Atlas):
    report = await atlas.ingest_bytes(
        [("../../evil/notes.md", b"# N\n\nUploaded note about docks.\n")], "up"
    )
    assert report.files[0].name == "notes.md"  # directory components are dropped
    assert atlas.documents("up")[0].path == "upload://notes.md"
    again = await atlas.ingest_bytes([("notes.md", b"# N\n\nUploaded note about docks.\n")], "up")
    assert again.unchanged == 1
    changed = await atlas.ingest_bytes([("notes.md", b"# N\n\nChanged note about docks.\n")], "up")
    assert changed.updated == 1


async def test_force_reingests_unchanged_files_without_reembedding(atlas: Atlas, corpus_dir: Path):
    await atlas.ingest(corpus_dir)
    calls = embed_calls(atlas)
    report = await atlas.ingest(corpus_dir, force=True)
    assert report.updated == 5 and all(f.embedded == 0 for f in report.files)
    assert embed_calls(atlas) == calls


async def test_changing_chunk_settings_triggers_reingest(
    mock_client, db_path: Path, corpus_dir: Path
):
    a = Atlas(mock_client, db_path)
    await a.ingest(corpus_dir)
    n_before = len(a.store.index("default"))
    a.close()
    b = Atlas(
        mock_client,
        db_path,
        config=AtlasConfig(chunk_max_chars=500, chunk_overlap=50, chunk_min_chars=100),
    )
    report = await b.ingest(corpus_dir)
    assert report.updated == 5
    assert len(b.store.index("default")) > n_before
    b.close()


async def test_collections_are_separate_namespaces(atlas: Atlas, corpus_dir: Path):
    await atlas.ingest(corpus_dir / "glossary.md", "one")
    await atlas.ingest(corpus_dir / "api_guide.md", "two")
    assert {c.name: c.documents for c in atlas.collections()} == {"one": 1, "two": 1}
    assert (await atlas.retrieve("rate limit", "one"))[0].source == "glossary.md"
    assert (await atlas.retrieve("rate limit", "two"))[0].source == "api_guide.md"
    assert atlas.delete_collection("one") and [c.name for c in atlas.collections()] == ["two"]
    assert atlas.delete_document("two", "api_guide.md") and atlas.documents("two") == []


async def test_embedding_failure_marks_files_as_errors_not_crashes(db_path: Path, tmp_path: Path):
    client = FakeClient()
    client.embed_error = NIMError("embedding endpoint down")
    atlas = Atlas(client, db_path)
    (tmp_path / "a.md").write_text("# A\n\nSome content.\n")
    report = await atlas.ingest(tmp_path)
    assert report.errors == 1 and "embedding failed" in report.files[0].message
    assert atlas.documents("default") == []
    atlas.close()


async def test_mixing_mock_and_live_vectors_in_one_collection_is_refused(
    mock_client, db_path: Path, corpus_dir: Path
):
    a = Atlas(mock_client, db_path)
    await a.ingest(corpus_dir / "glossary.md")
    a.close()
    live = Atlas(FakeClient(), db_path)
    with pytest.raises(StoreError, match="reindex"):
        await live.ingest(corpus_dir / "api_guide.md")
    live.close()


async def test_reindex_reembeds_with_the_current_backend(
    mock_client, db_path: Path, corpus_dir: Path
):
    a = Atlas(mock_client, db_path)
    await a.ingest(corpus_dir / "glossary.md")
    a.close()
    live = Atlas(FakeClient(), db_path)
    n = await live.reindex("default")
    assert n == len(live.store.index("default")) > 0
    info = live.store.get_collection("default")
    assert info is not None and not info.embed_model.startswith("mock:")  # type: ignore[union-attr]
    report = await live.ingest(corpus_dir / "api_guide.md")  # now allowed
    assert report.added == 1
    live.close()


async def test_progress_callback_sees_every_file(atlas: Atlas, corpus_dir: Path):
    seen: list[str] = []
    await atlas.ingest(corpus_dir, on_file=lambda r: seen.append(r.name))
    assert sorted(seen) == [
        "api_guide.md",
        "architecture.md",
        "glossary.md",
        "incident_runbook.md",
        "onboarding.md",
    ]


async def test_empty_collection_name_is_rejected(atlas: Atlas, corpus_dir: Path):
    with pytest.raises(Exception, match="collection name"):
        await atlas.ingest(corpus_dir, " ")
