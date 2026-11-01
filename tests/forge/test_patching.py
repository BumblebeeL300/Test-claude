"""apply_patch: SEARCH/REPLACE edit-block parsing and application, including the edge cases."""

from __future__ import annotations

import pytest

from nimlab.forge import PatchError, apply_patch_text
from nimlab.forge.patching import EditBlock, apply_edit_blocks, parse_edit_blocks


def block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


SOURCE = '''\
import math


def area(r):
    return 3.14 * r * r


def perimeter(r):
    return 2 * 3.14 * r
'''


class TestParsing:
    def test_single_block(self) -> None:
        blocks = parse_edit_blocks(block("a = 1", "a = 2"))
        assert blocks == [EditBlock(("a = 1",), ("a = 2",))]

    def test_multiple_blocks_and_noise_between(self) -> None:
        patch = "Here is the fix:\n```python\n" + block("x", "y") + "\nsome prose\n" + block("p", "q") + "```\n"
        blocks = parse_edit_blocks(patch)
        assert [b.search for b in blocks] == [("x",), ("p",)]

    def test_marker_length_and_indentation_are_tolerated(self) -> None:
        patch = "<<<<< SEARCH\nold\n=====\nnew\n>>>>>>>>> REPLACE"
        assert parse_edit_blocks(patch) == [EditBlock(("old",), ("new",))]

    def test_crlf_patch_text(self) -> None:
        patch = "<<<<<<< SEARCH\r\nold\r\n=======\r\nnew\r\n>>>>>>> REPLACE\r\n"
        assert parse_edit_blocks(patch) == [EditBlock(("old",), ("new",))]

    def test_multiline_blocks_keep_blank_lines(self) -> None:
        blocks = parse_edit_blocks(block("a\n\nb", "c\n\n\nd"))
        assert blocks[0].search == ("a", "", "b")
        assert blocks[0].replace == ("c", "", "", "d")

    @pytest.mark.parametrize(
        ("patch", "needle"),
        [
            ("", "empty"),
            ("   \n  ", "empty"),
            ("just some text", "no SEARCH/REPLACE"),
            ("<<<<<<< SEARCH\nold\nnew\n>>>>>>> REPLACE", "divider"),
            ("<<<<<<< SEARCH\nold\n=======\nnew\n", "closing"),
            ("<<<<<<< SEARCH\nold\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE", "second SEARCH"),
        ],
    )
    def test_malformed_patches_explain_the_format(self, patch: str, needle: str) -> None:
        with pytest.raises(PatchError, match=needle):
            parse_edit_blocks(patch)

    def test_non_string_patch(self) -> None:
        with pytest.raises(PatchError):
            parse_edit_blocks(None)  # type: ignore[arg-type]


class TestApplying:
    def test_exact_replacement(self) -> None:
        res = apply_patch_text(SOURCE, block("    return 3.14 * r * r", "    return math.pi * r * r"))
        assert "math.pi * r * r" in res.text
        assert "2 * 3.14 * r" in res.text  # the other function is untouched
        assert res.applied == 1 and res.tiers == ["exact"]
        assert (res.added, res.removed) == (1, 1)

    def test_multiple_blocks_apply_in_order(self) -> None:
        patch = block("def area(r):", "def circle_area(r):") + block("def perimeter(r):", "def circle_perimeter(r):")
        res = apply_patch_text(SOURCE, patch)
        assert "def circle_area(r):" in res.text and "def circle_perimeter(r):" in res.text
        assert res.applied == 2

    def test_second_block_can_target_text_created_by_the_first(self) -> None:
        patch = block("def area(r):", "def surface(r):") + block("def surface(r):", "def surface_area(r):")
        assert "def surface_area(r):" in apply_patch_text(SOURCE, patch).text

    def test_delete_with_empty_replace(self) -> None:
        patch = "<<<<<<< SEARCH\nimport math\n\n\n=======\n>>>>>>> REPLACE\n"
        res = apply_patch_text(SOURCE, patch)
        assert not res.text.startswith("import math")
        assert res.text.startswith("def area")

    def test_insert_by_extending_an_anchor_line(self) -> None:
        res = apply_patch_text(SOURCE, block("import math", "import math\nimport sys"))
        assert res.text.startswith("import math\nimport sys\n")

    def test_not_found_includes_closest_region_hint(self) -> None:
        with pytest.raises(PatchError, match="not found") as exc:
            apply_patch_text(SOURCE, block("def area(radius):\n    return 3.14 * radius", "x"))
        assert "Closest region" in str(exc.value)
        assert "def area(r):" in str(exc.value)

    def test_not_found_without_any_similar_text(self) -> None:
        with pytest.raises(PatchError, match="not found"):
            apply_patch_text(SOURCE, block("completely unrelated content here", "x"))

    def test_ambiguous_match_rejected_with_line_numbers(self) -> None:
        text = "x = 1\ny = 2\nx = 1\n"
        with pytest.raises(PatchError, match="ambiguous") as exc:
            apply_patch_text(text, block("x = 1", "x = 9"))
        assert "1, 3" in str(exc.value)

    def test_ambiguity_resolved_by_more_context(self) -> None:
        text = "x = 1\ny = 2\nx = 1\nz = 3\n"
        res = apply_patch_text(text, block("x = 1\nz = 3", "x = 9\nz = 3"))
        assert res.text == "x = 1\ny = 2\nx = 9\nz = 3\n"

    def test_partial_line_text_does_not_match(self) -> None:
        # Line-based matching: a substring of a line is not a match.
        with pytest.raises(PatchError, match="not found"):
            apply_patch_text("value = 10\n", block("value = 1", "value = 2"))

    def test_atomic_when_a_later_block_fails(self) -> None:
        patch = block("def area(r):", "def A(r):") + block("this does not exist", "x")
        with pytest.raises(PatchError, match=r"block 2/2"):
            apply_patch_text(SOURCE, patch)
        # apply_patch_text is pure: the caller still has the untouched original
        assert "def area(r):" in SOURCE

    def test_trailing_whitespace_tolerated(self) -> None:
        text = "a = 1   \nb = 2\n"
        res = apply_patch_text(text, block("a = 1", "a = 10"))
        assert res.tiers == ["rstrip"]
        assert res.text == "a = 10\nb = 2\n"

    def test_indentation_mismatch_is_reindented(self) -> None:
        text = "class A:\n    def f(self):\n        return 1\n"
        # model forgot the class indentation
        patch = block("def f(self):\n    return 1", "def f(self):\n    return 2")
        res = apply_patch_text(text, patch)
        assert res.tiers == ["indent-insensitive"]
        assert res.text == "class A:\n    def f(self):\n        return 2\n"

    def test_over_indented_search_is_dedented(self) -> None:
        text = "def f():\n    return 1\n"
        patch = block("        def f():\n            return 1", "        def f():\n            return 2")
        assert apply_patch_text(text, patch).text == "def f():\n    return 2\n"

    def test_leading_and_trailing_blank_lines_in_search_are_forgiven(self) -> None:
        text = "a = 1\nb = 2\n"
        patch = "<<<<<<< SEARCH\n\na = 1\n\n=======\na = 5\n>>>>>>> REPLACE\n"
        assert apply_patch_text(text, patch).text == "a = 5\nb = 2\n"

    def test_crlf_files_keep_crlf(self) -> None:
        text = "a = 1\r\nb = 2\r\n"
        res = apply_patch_text(text, block("b = 2", "b = 3"))
        assert res.text == "a = 1\r\nb = 3\r\n"
        assert "\n" not in res.text.replace("\r\n", "")

    def test_missing_trailing_newline_is_preserved(self) -> None:
        res = apply_patch_text("a = 1\nb = 2", block("b = 2", "b = 3"))
        assert res.text == "a = 1\nb = 3"

    def test_trailing_newline_is_preserved(self) -> None:
        res = apply_patch_text("a = 1\nb = 2\n", block("a = 1", "a = 0"))
        assert res.text.endswith("b = 2\n")

    def test_empty_search_creates_content_in_empty_file(self) -> None:
        patch = "<<<<<<< SEARCH\n=======\nprint('hello')\n>>>>>>> REPLACE\n"
        assert apply_patch_text("", patch).text == "print('hello')\n"

    def test_empty_search_rejected_for_existing_content(self) -> None:
        patch = "<<<<<<< SEARCH\n=======\nprint('hello')\n>>>>>>> REPLACE\n"
        with pytest.raises(PatchError, match="empty SEARCH"):
            apply_patch_text("x = 1\n", patch)

    def test_whitespace_only_search_rejected_for_existing_content(self) -> None:
        with pytest.raises(PatchError, match="empty SEARCH"):
            apply_patch_text("x = 1\n", block("   ", "y"))

    def test_noop_patch_returns_identical_text(self) -> None:
        res = apply_patch_text(SOURCE, block("import math", "import math"))
        assert res.text == SOURCE and (res.added, res.removed) == (0, 0)

    def test_unicode_content(self) -> None:
        res = apply_patch_text("s = 'héllo wörld ✓'\n", block("s = 'héllo wörld ✓'", "s = 'ünïcode ✓✓'"))
        assert res.text == "s = 'ünïcode ✓✓'\n"

    def test_lines_that_look_like_markers_inside_python_strings(self) -> None:
        text = 'banner = "======="\nx = 1\n'
        # The divider must be the first bare '=======' line after SEARCH; quoted text is a normal line.
        res = apply_patch_text(text, block('banner = "======="', 'banner = "-------"'))
        assert res.text == 'banner = "-------"\nx = 1\n'

    def test_apply_edit_blocks_direct(self) -> None:
        res = apply_edit_blocks("a\nb\nc\n", [EditBlock(("b",), ("B1", "B2"))])
        assert res.text == "a\nB1\nB2\nc\n"
        assert (res.added, res.removed) == (2, 1)
