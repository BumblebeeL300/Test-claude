"""Loaders: markdown, text, HTML (stdlib parser) and page-aware PDF."""

from __future__ import annotations

from pathlib import Path

import pytest

from nimlab.atlas.loaders import (
    discover_files,
    load_bytes,
    load_file,
    media_type_for,
    parse_html,
    parse_text,
    reflow_pdf_text,
)
from nimlab.atlas.models import LoaderError

HTML = """<!doctype html><html><head><title>Robot FAQ</title>
<style>body {color: red}</style><script>var secret = 1;</script></head>
<body><nav>Home | About</nav>
<h1>Fleet</h1><p>Hello <b>world</b> &amp; friends.</p>
<h2>Charging</h2><ul><li>Dock at 20%</li><li>Swap at 5%</li></ul>
<pre>kubectl get pods
  -n hive</pre>
<table><tr><th>Zone</th><th>Speed</th></tr><tr><td>Pick</td><td>0.6</td></tr></table>
<h2>Safety</h2><p>Press the e-stop.</p>
<noscript>enable js</noscript></body></html>"""


def test_html_headings_become_sections_and_tags_are_stripped():
    title, sections = parse_html(HTML)
    assert title == "Robot FAQ"
    paths = [s.heading_path for s in sections]
    assert paths == [("Fleet",), ("Fleet", "Charging"), ("Fleet", "Safety")]
    body = " ".join(s.text for s in sections)
    assert "secret" not in body and "color: red" not in body and "enable js" not in body
    assert "Home | About" not in body  # <nav> dropped
    assert "Hello world & friends." in sections[0].text


def test_html_lists_pre_and_tables_keep_structure():
    _, sections = parse_html(HTML)
    charging = sections[1].text
    assert "- Dock at 20%\n- Swap at 5%" in charging
    assert "```\nkubectl get pods\n  -n hive\n```" in charging
    assert "Zone | Speed" in charging and "Pick | 0.6" in charging


def test_html_without_title_uses_first_h1_and_handles_malformed_markup():
    title, sections = parse_html("<h1>Only</h1><p>unclosed <b>bold<p>next")
    assert title == "Only"
    assert "unclosed" in sections[0].text and "next" in sections[0].text


def test_html_missing_head_close_does_not_swallow_body():
    _, sections = parse_html("<html><head><title>T</title><body><p>visible text</p></body></html>")
    assert sections and "visible text" in sections[0].text


def test_plain_text_is_one_section():
    title, sections = parse_text("line one\r\n\r\nline two\r\n")
    assert title is None
    assert len(sections) == 1 and sections[0].text == "line one\n\nline two"
    assert parse_text("   \n")[1] == []


def test_load_bytes_dispatches_on_extension_and_hashes_raw_bytes():
    doc = load_bytes(b"# Title\n\nHello there.\n", name="note.md")
    assert doc.media_type == "markdown" and doc.title == "Title"
    assert len(doc.content_hash) == 64
    assert doc.content_hash == load_bytes(b"# Title\n\nHello there.\n", name="x.md").content_hash
    assert load_bytes(b"plain", name="a.txt").title == "a"
    assert load_bytes(b"<h1>H</h1><p>x</p>", name="a.HTML").media_type == "html"


def test_load_bytes_handles_bom_and_bad_utf8():
    doc = load_bytes("# Hi\n\ncaf\xe9\n".encode("utf-8-sig"), name="a.md")
    assert doc.sections[0].text == "caf\xe9"
    doc = load_bytes(b"ok \xff\xfe bytes", name="b.txt")
    assert "ok" in doc.sections[0].text


@pytest.mark.parametrize("name", ["image.png", "archive.zip", "noext"])
def test_unsupported_types_raise(name: str):
    assert media_type_for(name) is None
    with pytest.raises(LoaderError, match="unsupported"):
        load_bytes(b"x", name=name)


def test_empty_document_raises():
    with pytest.raises(LoaderError, match="empty"):
        load_bytes(b"  \n\n", name="a.md")


def test_pdf_pages_become_page_aware_sections(make_pdf):
    pdf = make_pdf([["Alpha page content about batteries."], ["Beta page content about docks."]])
    doc = load_bytes(pdf, name="manual.pdf")
    assert doc.media_type == "pdf"
    assert [s.page for s in doc.sections] == [1, 2]
    assert "batteries" in doc.sections[0].text and "docks" in doc.sections[1].text


def test_pdf_blank_pages_are_skipped_and_empty_pdf_errors(make_pdf):
    doc = load_bytes(make_pdf([["First page text."], [], ["Third page text."]]), name="a.pdf")
    assert [s.page for s in doc.sections] == [1, 3]
    with pytest.raises(LoaderError, match="no extractable text"):
        load_bytes(make_pdf([[], []]), name="blank.pdf")


def test_corrupt_pdf_raises_loader_error():
    with pytest.raises(LoaderError):
        load_bytes(b"%PDF-1.4 this is not really a pdf", name="bad.pdf")


def test_reflow_joins_wrapped_lines_and_dehyphenates():
    text = (
        "The fleet manager allocates\ntasks to every idle ro-\nbot nearby.\n\n"
        "- first item\n- second item"
    )
    out = reflow_pdf_text(text)
    assert "allocates tasks to every idle robot nearby." in out
    assert "- first item" in out.split("\n\n") and "- second item" in out.split("\n\n")


def test_discover_files_is_sorted_recursive_and_skips_hidden(tmp_path: Path):
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "skip.png").write_bytes(b"x")
    (tmp_path / ".hidden.md").write_text("h")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.md").write_text("g")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.html").write_text("<p>c</p>")
    names = [name for _, name in discover_files(tmp_path)]
    assert names == ["a.txt", "b.md", "sub/c.html"]
    assert [n for _, n in discover_files(tmp_path, recursive=False)] == ["a.txt", "b.md"]
    assert discover_files(tmp_path / "b.md") == [(tmp_path / "b.md", "b.md")]
    assert discover_files(tmp_path / "skip.png") == []
    with pytest.raises(LoaderError):
        discover_files(tmp_path / "missing")


def test_load_file_uses_resolved_path_as_key(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text("# D\n\ntext here\n")
    doc = load_file(f, name="docs/doc.md")
    assert doc.path == str(f.resolve()) and doc.name == "docs/doc.md"
    with pytest.raises(LoaderError, match="cannot read"):
        load_file(tmp_path / "nope.md")
