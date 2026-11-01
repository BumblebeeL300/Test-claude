"""Command line interface via Typer's CliRunner (always with --mock, fully offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nimlab.atlas.cli import app

runner = CliRunner()
HEARTBEAT_Q = "How many missed heartbeats mark a robot as lost?"


@pytest.fixture
def db(tmp_path: Path) -> str:
    return str(tmp_path / "cli.db")


@pytest.fixture
def ingested(db: str, corpus_dir: Path) -> str:
    result = runner.invoke(app, ["ingest", str(corpus_dir), "--db", db, "--mock"])
    assert result.exit_code == 0, result.output
    return db


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "ask", "serve", "stats", "eval", "delete", "reindex"):
        assert command in result.output


def test_ingest_then_reingest_is_incremental(db: str, corpus_dir: Path):
    first = runner.invoke(app, ["ingest", str(corpus_dir), "--db", db, "--mock", "-c", "handbook"])
    assert first.exit_code == 0, first.output
    assert (
        "5 added" in first.output
        and "architecture.md" in first.output
        and "handbook" in first.output
    )
    second = runner.invoke(app, ["ingest", str(corpus_dir), "--db", db, "--mock", "-c", "handbook"])
    assert "5 unchanged" in second.output and "0 added" in second.output
    forced = runner.invoke(
        app, ["ingest", str(corpus_dir), "--db", db, "--mock", "-c", "handbook", "--force"]
    )
    assert "5 updated" in forced.output


def test_ingest_options_and_errors(db: str, tmp_path: Path, corpus_dir: Path):
    assert (
        runner.invoke(app, ["ingest", str(tmp_path / "nope"), "--db", db, "--mock"]).exit_code == 1
    )
    bad = runner.invoke(
        app,
        [
            "ingest",
            str(corpus_dir),
            "--db",
            db,
            "--mock",
            "--chunk-size",
            "300",
            "--overlap",
            "300",
        ],
    )
    assert bad.exit_code == 1 and "chunk_overlap" in bad.output
    (tmp_path / "bad.pdf").write_bytes(b"nope")
    failing = runner.invoke(app, ["ingest", str(tmp_path / "bad.pdf"), "--db", db, "--mock"])
    assert failing.exit_code == 1 and "error" in failing.output


def test_ingest_prune(db: str, tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nalpha content here")
    (docs / "b.md").write_text("# B\n\nbeta content here")
    runner.invoke(app, ["ingest", str(docs), "--db", db, "--mock"])
    (docs / "b.md").unlink()
    result = runner.invoke(app, ["ingest", str(docs), "--db", db, "--mock", "--prune"])
    assert "1 removed" in result.output


def test_stats_table_and_json(ingested: str):
    table = runner.invoke(app, ["stats", "--db", ingested, "--docs"])
    assert table.exit_code == 0 and "default" in table.output and "5 documents" in table.output
    assert "glossary.md" in table.output and "mock:" in table.output
    data = json.loads(runner.invoke(app, ["stats", "--db", ingested, "--json"]).output)
    assert data["documents"] == 5 and data["collections"][0]["dim"] == 1024
    assert runner.invoke(app, ["stats", "--db", ingested + ".missing"]).exit_code == 1


def test_ask_prints_answer_sources_and_verification(ingested: str):
    result = runner.invoke(
        app, ["ask", HEARTBEAT_Q, "--db", ingested, "--mock", "--verify", "--trace", "-k", "4"]
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Answer (simple)" in out and "Sources" in out and "architecture.md" in out
    assert "Groundedness:" in out and "Claims" in out and "retrieve:" in out and "Mock mode" in out


def test_ask_json_output_is_valid_and_complete(ingested: str):
    result = runner.invoke(
        app,
        ["ask", HEARTBEAT_Q, "--db", ingested, "--mock", "--json", "--verify", "--mode", "agentic"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert (
        data["mode"] == "agentic" and data["sources"] and data["verification"]["claims"] is not None
    )
    assert data["trace"][0]["kind"] == "plan"


def test_ask_flags_guard_revise_hyde_and_ablations(ingested: str):
    result = runner.invoke(
        app,
        [
            "ask",
            HEARTBEAT_Q,
            "--db",
            ingested,
            "--mock",
            "--guard",
            "--revise",
            "--hyde",
            "--no-rerank",
            "--no-mmr",
            "--no-sources",
            "--usage",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Sources" not in result.output.split("Answer", 1)[1].split("Groundedness")[0]
    assert "NIM usage" in result.output


def test_ask_on_an_empty_database_explains_itself(db: str):
    result = runner.invoke(app, ["ask", "anything", "--db", db, "--mock"])
    assert result.exit_code == 0 and "No documents are indexed" in result.output


def test_ask_rejects_bad_options(ingested: str):
    assert (
        runner.invoke(app, ["ask", "q", "--db", ingested, "--mock", "--mode", "turbo"]).exit_code
        == 2
    )
    assert runner.invoke(app, ["ask", "q", "--db", ingested, "--mock", "-k", "0"]).exit_code == 2


def test_eval_reports_recall_mrr_and_groundedness(ingested: str, qa_file: Path, tmp_path: Path):
    out = tmp_path / "report.json"
    result = runner.invoke(
        app, ["eval", str(qa_file), "--db", ingested, "--mock", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    for text in (
        "recall@1",
        "recall@3",
        "recall@6",
        "MRR",
        "hit rate",
        "mean groundedness",
        "Mossgrove handbook QA",
    ):
        assert text in result.output
    report = json.loads(out.read_text())
    assert (
        report["n_questions"] == 11 and report["recall_at_k"]["6"] >= 0.9 and report["mrr"] >= 0.8
    )


def test_eval_retrieval_only_and_recall_gate(ingested: str, qa_file: Path):
    ok = runner.invoke(
        app,
        ["eval", str(qa_file), "--db", ingested, "--mock", "--no-answer", "--min-recall", "0.8"],
    )
    assert ok.exit_code == 0 and "mean groundedness" not in ok.output
    gated = runner.invoke(
        app,
        ["eval", str(qa_file), "--db", ingested, "--mock", "--no-answer", "--min-recall", "1.01"],
    )
    assert gated.exit_code == 1 and "recall@6" in gated.output
    assert (
        runner.invoke(app, ["eval", str(qa_file) + ".nope", "--db", ingested, "--mock"]).exit_code
        == 1
    )


def test_delete_requires_confirmation(ingested: str):
    declined = runner.invoke(app, ["delete", "default", "--db", ingested], input="n\n")
    assert declined.exit_code != 0
    assert "default" in runner.invoke(app, ["stats", "--db", ingested]).output
    done = runner.invoke(app, ["delete", "default", "--db", ingested, "--yes"])
    assert done.exit_code == 0 and "Deleted" in done.output
    assert runner.invoke(app, ["delete", "default", "--db", ingested, "--yes"]).exit_code == 1


def test_reindex_command(ingested: str):
    result = runner.invoke(app, ["reindex", "--db", ingested, "--mock"])
    assert result.exit_code == 0 and "Re-embedded" in result.output
    assert runner.invoke(app, ["reindex", "-c", "ghost", "--db", ingested, "--mock"]).exit_code == 1


def test_serve_builds_the_app_and_starts_uvicorn(db: str, monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}
    monkeypatch.setattr("uvicorn.run", lambda web, **kw: captured.update(app=web, **kw))
    result = runner.invoke(
        app, ["serve", "--db", db, "--mock", "--port", "9123", "--host", "0.0.0.0"]
    )
    assert result.exit_code == 0 and "9123" in result.output
    assert captured["port"] == 9123 and captured["host"] == "0.0.0.0"
    assert captured["app"].title == "Atlas"


def test_atlas_is_mounted_under_the_root_cli(corpus_dir: Path, db: str):
    from nimlab.cli import app as root

    assert "atlas" in runner.invoke(root, ["--help"]).output
    result = runner.invoke(
        root, ["--mock", "atlas", "ingest", str(corpus_dir / "glossary.md"), "--db", db]
    )
    assert result.exit_code == 0 and "1 added" in result.output
    asked = runner.invoke(root, ["--mock", "atlas", "ask", "What is a tote?", "--db", db])
    assert asked.exit_code == 0 and "glossary.md" in asked.output
