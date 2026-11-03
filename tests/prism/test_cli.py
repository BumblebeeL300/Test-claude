"""``prism`` CLI through CliRunner with --mock."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from nimlab.prism.cli import app
from nimlab.prism.schemas import BUILTIN_SCHEMAS

runner = CliRunner()


def invoke(*args: str, code: int = 0) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == code, result.output
    return result.output


def test_no_args_shows_help() -> None:
    out = runner.invoke(app, []).output
    for cmd in ("describe", "ocr", "extract", "ui2code", "compare", "chart", "ask", "batch"):
        assert cmd in out


def test_modes_lists_everything() -> None:
    out = invoke("modes")
    assert "ui2code" in out and "business_card" in out


def test_describe(sample_ui: Path) -> None:
    out = invoke("describe", str(sample_ui), "--mock")
    assert "SIMULATED" in out and "Alt text:" in out and "256 by 160" in out


def test_describe_json(sample_ui: Path) -> None:
    out = invoke("describe", str(sample_ui), "--mock", "--json")
    payload = json.loads(out[out.index("{") :])
    assert payload["simulated"] is True and payload["output"]["alt_text"]


def test_ocr_to_file(sample_ui: Path, tmp_path: Path) -> None:
    dest = tmp_path / "text.txt"
    invoke("ocr", str(sample_ui), "--mock", "-o", str(dest))
    assert "TOTAL" in dest.read_text()


def test_extract_builtin_and_file_schema(sample_ui: Path, tmp_path: Path) -> None:
    out = invoke("extract", str(sample_ui), "--schema", "invoice", "--mock")
    data = json.loads(out[out.index("{") :])
    assert list(Draft202012Validator(BUILTIN_SCHEMAS["invoice"].schema).iter_errors(data)) == []

    schema = tmp_path / "s.json"
    schema.write_text(
        json.dumps(
            {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
        )
    )
    dest = tmp_path / "x.json"
    invoke("extract", str(sample_ui), "-s", str(schema), "--mock", "-o", str(dest))
    assert json.loads(dest.read_text())["title"]


def test_extract_bad_schema_is_a_clean_error(sample_ui: Path) -> None:
    out = invoke("extract", str(sample_ui), "--schema", "nope", "--mock", code=1)
    assert "prism failed" in out and "unknown schema" in out


def test_ui2code_writes_valid_html(sample_ui: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out.html"
    out = invoke("ui2code", str(sample_ui), "-o", str(dest), "--refine", "2", "--mock")
    html = dest.read_text()
    assert html.startswith("<!doctype html>") and "draft 3" in html
    assert "round 1" in out and "round 2" in out and "3 draft(s)" in out


def test_ui2code_rejects_too_many_rounds(sample_ui: Path) -> None:
    assert runner.invoke(app, ["ui2code", str(sample_ui), "--refine", "9", "--mock"]).exit_code == 2


def test_compare(sample_ui: Path, sample_chart: Path) -> None:
    out = invoke("compare", str(sample_ui), str(sample_chart), "--mock")
    assert "dimensions" in out and "palette" in out


def test_chart_csv(sample_chart: Path, tmp_path: Path) -> None:
    dest = tmp_path / "chart.csv"
    out = invoke("chart", str(sample_chart), "--csv", str(dest), "--mock")
    rows = list(csv.reader(dest.open()))
    assert rows[0][0] == "Period" and len(rows) > 3 and "wrote" in out


def test_ask(sample_ui: Path) -> None:
    out = invoke("ask", str(sample_ui), "What colors are used?", "--mock")
    assert "navy" in out


def test_batch_to_jsonl(tmp_path: Path, png_file: Callable[..., Path]) -> None:
    for name in ("one.png", "two.png"):
        png_file(name, width=12, height=9)
    folder = tmp_path
    dest = tmp_path / "results.jsonl"
    out = invoke("batch", str(folder), "--mode", "ocr", "--out", str(dest), "--mock")
    rows = [json.loads(line) for line in dest.read_text().splitlines()]
    assert {r["name"] for r in rows} == {"one.png", "two.png"} and all(r["ok"] for r in rows)
    assert "2/2 succeeded" in out and "SIMULATED" in out


def test_batch_reports_failures_with_exit_1(tmp_path: Path, png_file: Callable[..., Path]) -> None:
    png_file("ok.png")
    (tmp_path / "bad.png").write_bytes(b"junk")
    dest = tmp_path / "r.jsonl"
    out = invoke("batch", str(tmp_path), "--out", str(dest), "--mock", code=1)
    assert "failed bad.png" in out and "1/2 succeeded" in out
    assert len(dest.read_text().splitlines()) == 2


def test_batch_with_mode_options(tmp_path: Path, png_file: Callable[..., Path]) -> None:
    png_file("q.png")
    dest = tmp_path / "r.jsonl"
    invoke("batch", str(tmp_path), "--mode", "ask", "-q", "Size?", "--out", str(dest), "--mock")
    assert "Size?" in dest.read_text()


def test_unsupported_file_is_a_clean_error(tmp_path: Path) -> None:
    bad = tmp_path / "doc.png"
    bad.write_bytes(b"%PDF-1.4")
    out = invoke("describe", str(bad), "--mock", code=1)
    assert "unsupported image type" in out


def test_missing_image_is_rejected_by_typer(tmp_path: Path) -> None:
    assert runner.invoke(app, ["describe", str(tmp_path / "none.png"), "--mock"]).exit_code == 2


def test_mock_flag_does_not_leak_into_the_environment(
    sample_ui: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    monkeypatch.delenv("NIMLAB_MOCK", raising=False)
    invoke("describe", str(sample_ui), "--mock")
    assert "NIMLAB_MOCK" not in os.environ
