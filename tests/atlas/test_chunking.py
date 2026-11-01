"""Structure-aware chunking: headings, overlap, code blocks, merging, pages."""

from __future__ import annotations

from itertools import pairwise

import pytest

from nimlab.atlas.chunking import chunk_sections, split_blocks, split_long
from nimlab.atlas.loaders import parse_markdown
from nimlab.atlas.models import Section

LOREM = (
    "Robots carry totes between stations. Each tote holds twenty five kilograms. "
    "Fleet software assigns tasks every quarter second. Batteries are swapped at the dock. "
)


def chunks_of(md: str, **kw: int):
    _, sections = parse_markdown(md)
    return chunk_sections(sections, **kw)


def test_heading_path_is_kept_as_metadata():
    md = "# Guide\n\nIntro text.\n\n## Setup\n\nInstall it.\n\n### Details\n\nMore details here.\n"
    _, sections = parse_markdown(md)
    assert [s.heading_path for s in sections] == [
        ("Guide",),
        ("Guide", "Setup"),
        ("Guide", "Setup", "Details"),
    ]
    chunks = chunk_sections(sections, max_chars=1000, overlap=0, min_chars=0)
    assert [c.heading_path for c in chunks] == [s.heading_path for s in sections]
    assert chunks[2].indexed_text.startswith("Guide > Setup > Details\n")


def test_sibling_headings_replace_each_other_in_the_path():
    md = "# A\n\n## One\n\nalpha\n\n## Two\n\nbeta\n\n# B\n\ngamma\n"
    _, sections = parse_markdown(md)
    assert [s.heading_path for s in sections] == [("A", "One"), ("A", "Two"), ("B",)]


def test_hash_lines_inside_code_fences_are_not_headings():
    md = "# Title\n\n```bash\n# just a comment\nls -la\n```\n\nAfter the code.\n"
    _, sections = parse_markdown(md)
    assert len(sections) == 1
    assert "# just a comment" in sections[0].text
    assert sections[0].heading_path == ("Title",)


def test_setext_headings_and_front_matter_title():
    md = "---\ntitle: My Doc\n---\nSection\n=======\n\nBody text.\n"
    title, sections = parse_markdown(md)
    assert title == "My Doc"
    assert sections[0].heading_path == ("Section",)


def test_atx_heading_cleanup_keeps_trailing_hash_of_c_sharp():
    _, sections = parse_markdown("## Learning C#\n\ntext\n\n## **Bold** `code` ##\n\nmore\n")
    assert sections[0].heading_path == ("Learning C#",)
    assert sections[1].heading_path == ("Bold code",)


def test_chunks_never_exceed_max_chars():
    md = "# T\n\n" + "\n\n".join(LOREM * 3 for _ in range(12))
    chunks = chunks_of(md, max_chars=400, overlap=60, min_chars=0)
    assert len(chunks) > 5
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.heading_path == ("T",) for c in chunks)


def test_overlap_shares_text_between_consecutive_chunks_of_a_section():
    md = "# T\n\n" + "\n\n".join(f"Paragraph {i}. {LOREM}" for i in range(8))
    chunks = chunks_of(md, max_chars=420, overlap=100, min_chars=0)
    assert len(chunks) >= 3
    for prev, nxt in pairwise(chunks):
        head = nxt.text.split("\n\n")[0]
        assert 0 < len(head) <= 100
        assert head in prev.text  # the carried-over tail really came from the previous chunk
        assert prev.text.endswith(head)


def test_overlap_zero_means_no_shared_text():
    md = "# T\n\n" + "\n\n".join(f"Paragraph {i}. {LOREM}" for i in range(8))
    chunks = chunks_of(md, max_chars=420, overlap=0, min_chars=0)
    joined = "\n\n".join(c.text for c in chunks)
    assert joined.count("Paragraph 3.") == 1


def test_overlap_does_not_cross_section_boundaries():
    md = f"# T\n\n## A\n\n{LOREM * 3}\n\n## B\n\n{LOREM * 3}\n"
    chunks = chunks_of(md, max_chars=300, overlap=80, min_chars=0)
    first_b = next(c for c in chunks if c.heading_path[-1] == "B")
    last_a = [c for c in chunks if c.heading_path[-1] == "A"][-1]
    assert not first_b.text.startswith(last_a.text[-40:])


def test_code_block_stays_in_one_chunk_when_it_fits():
    code = "```python\nprint('a')\nprint('b')\n```"
    md = f"# T\n\nIntro paragraph.\n\n{code}\n\nOutro paragraph.\n"
    chunks = chunks_of(md, max_chars=500, overlap=0, min_chars=0)
    assert len(chunks) == 1
    assert code in chunks[0].text


def test_oversized_code_block_is_split_with_balanced_fences():
    body = "\n".join(f"line_{i} = {i}" for i in range(60))
    blocks = split_long(f"```python\n{body}\n```", 200)
    assert len(blocks) > 3
    for block in blocks:
        assert len(block) <= 200
        assert block.startswith("```python\n") and block.endswith("\n```")
    assert "line_0 = 0" in blocks[0] and "line_59 = 59" in blocks[-1]


def test_lists_and_tables_stay_together():
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    items = "- one\n- two\n- three"
    assert split_blocks(f"{table}\n\n{items}") == [table, items]


def test_blank_lines_inside_fences_do_not_split_blocks():
    assert split_blocks("```\na\n\nb\n```\n\nafter") == ["```\na\n\nb\n```", "after"]


def test_long_sentence_without_breaks_is_hard_split():
    pieces = split_long("word " * 100, 120)
    assert all(len(p) <= 120 for p in pieces)
    assert " ".join(pieces).split() == ["word"] * 100


def test_unbreakable_token_is_sliced():
    pieces = split_long("x" * 500, 100)
    assert [len(p) for p in pieces] == [100] * 5


def test_small_chunks_are_merged_and_headings_restated():
    md = "# T\n\nIntro line.\n\n## Setup\n\nShort setup text.\n\n### Deep\n\nTiny.\n"
    chunks = chunks_of(md, max_chars=300, overlap=0, min_chars=80)
    assert len(chunks) == 1
    text = chunks[0].text
    assert "## Setup" in text and "### Deep" in text
    assert chunks[0].heading_path == ("T",)  # common prefix of the merged sections


def test_merge_disabled_with_min_chars_zero():
    md = "# T\n\nIntro line.\n\n## Setup\n\nShort setup text.\n"
    assert len(chunks_of(md, max_chars=300, overlap=0, min_chars=0)) == 2


def test_merge_does_not_exceed_max_chars():
    md = "# T\n\nTiny.\n\n## S\n\n" + "x " * 140
    chunks = chunks_of(md, max_chars=300, overlap=0, min_chars=100)
    assert all(len(c.text) <= 300 for c in chunks)


def test_pages_are_preserved_and_never_merged_across_pages():
    sections = [
        Section("Page one text.", (), 1),
        Section("Page two text.", (), 2),
        Section("Page three text.", (), 3),
    ]
    chunks = chunk_sections(sections, max_chars=500, overlap=0, min_chars=200)
    assert [c.page for c in chunks] == [1, 2, 3]


def test_ord_is_sequential_and_empty_input_gives_no_chunks():
    chunks = chunks_of("# T\n\n" + "\n\n".join(LOREM for _ in range(6)), max_chars=300, overlap=0)
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    assert chunk_sections([]) == []
    assert chunk_sections([Section("   \n  ")]) == []


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (100, 100), (100, -1), (100, 150)])
def test_invalid_parameters_raise(max_chars: int, overlap: int):
    with pytest.raises(ValueError):
        chunk_sections([Section("text")], max_chars=max_chars, overlap=overlap)


def test_content_hash_ignores_whitespace_differences():
    a = chunks_of("# T\n\nHello   world\n")[0]
    b = chunks_of("# T\n\nHello world\n")[0]
    assert a.hash == b.hash
    assert a.hash != chunks_of("# T\n\nHello there\n")[0].hash
