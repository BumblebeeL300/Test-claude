from __future__ import annotations

import itertools

import numpy as np
import pytest

from nimlab.core.text import (
    JSONExtractionError,
    ThinkStreamFilter,
    chunk_text,
    cosine,
    extract_json,
    strip_think,
)


class TestStripThink:
    def test_basic(self) -> None:
        assert strip_think("<think>reasoning here</think>\n\nThe answer.") == (
            "The answer.",
            "reasoning here",
        )

    def test_no_think_is_unchanged(self) -> None:
        text = "  plain answer\n"
        assert strip_think(text) == (text, None)

    def test_empty(self) -> None:
        assert strip_think("") == ("", None)

    def test_multiple_blocks(self) -> None:
        content, reasoning = strip_think("<think>a</think>X<think>b</think>Y")
        assert content == "XY"
        assert reasoning == "a\n\nb"

    def test_unterminated_block_is_all_reasoning(self) -> None:
        assert strip_think("<think>cut off by max_tokens") == ("", "cut off by max_tokens")

    def test_text_before_unterminated_block_is_kept(self) -> None:
        assert strip_think("Answer. <think>and then") == ("Answer.", "and then")

    def test_lone_closing_tag(self) -> None:
        assert strip_think("pre-filled reasoning</think>\nfinal") == (
            "final",
            "pre-filled reasoning",
        )

    def test_case_insensitive_and_multiline(self) -> None:
        content, reasoning = strip_think("<THINK>\nline1\nline2\n</Think>ok")
        assert (content, reasoning) == ("ok", "line1\nline2")

    def test_empty_block_gives_none_reasoning(self) -> None:
        assert strip_think("<think>  </think>hello") == ("hello", None)


class TestThinkStreamFilter:
    @pytest.mark.parametrize("size", [1, 2, 3, 5, 100])
    def test_matches_strip_think_for_any_chunking(self, size: int) -> None:
        text = "Intro. <think>secret\nthoughts</think>\n\nAnswer part one. <think>more</think>Tail."
        f = ThinkStreamFilter()
        out = "".join(f.feed(text[i : i + size]) for i in range(0, len(text), size)) + f.flush()
        assert out == "Intro. Answer part one. Tail."
        assert "secret" not in out

    def test_partial_open_tag_at_end_is_emitted_on_flush(self) -> None:
        f = ThinkStreamFilter()
        assert f.feed("a <thi") == "a "
        assert f.flush() == "<thi"

    def test_unterminated_think_is_dropped(self) -> None:
        f = ThinkStreamFilter()
        assert f.feed("<think>never closed") == ""
        assert f.flush() == ""

    def test_text_without_tags_passes_through(self) -> None:
        f = ThinkStreamFilter()
        assert f.feed("hello ") + f.feed("world") + f.flush() == "hello world"


class TestExtractJson:
    def test_plain_object(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_plain_array(self) -> None:
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_scalar_whole_text(self) -> None:
        assert extract_json("42") == 42

    def test_fenced_json(self) -> None:
        assert extract_json('Here you go:\n```json\n{"a": [1, 2]}\n```\nHope it helps') == {
            "a": [1, 2]
        }

    def test_fenced_without_language(self) -> None:
        assert extract_json('```\n{"x": true}\n```') == {"x": True}

    def test_leading_and_trailing_prose(self) -> None:
        assert extract_json('Sure! The result is {"ok": true, "n": 3}. Anything else?') == {
            "ok": True,
            "n": 3,
        }

    def test_think_block_with_braces_is_ignored(self) -> None:
        text = '<think>maybe {"wrong": 1} is it?</think>\n{"right": 2}'
        assert extract_json(text) == {"right": 2}

    def test_trailing_commas_are_repaired(self) -> None:
        assert extract_json('{"a": [1, 2,], "b": {"c": 1,},}') == {"a": [1, 2], "b": {"c": 1}}

    def test_braces_inside_strings(self) -> None:
        assert extract_json('prefix {"s": "a } b { c", "n": 1} suffix') == {
            "s": "a } b { c",
            "n": 1,
        }

    def test_nested(self) -> None:
        assert extract_json('x {"a": {"b": [{"c": 1}]}} y') == {"a": {"b": [{"c": 1}]}}

    def test_first_json_wins(self) -> None:
        assert extract_json('{"first": 1} and {"second": 2}') == {"first": 1}

    def test_unparseable_candidate_then_valid(self) -> None:
        assert extract_json('use {braces} like so: {"a": 1}') == {"a": 1}

    @pytest.mark.parametrize("text", ["", "   ", "no json here", "<think>only thoughts</think>"])
    def test_raises_when_nothing_found(self, text: str) -> None:
        with pytest.raises(JSONExtractionError):
            extract_json(text)

    def test_is_value_error(self) -> None:
        assert issubclass(JSONExtractionError, ValueError)


class TestChunkText:
    def test_empty(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("  \n\n  ") == []

    def test_short_text_single_chunk(self) -> None:
        assert chunk_text("Hello world.") == ["Hello world."]

    def test_respects_max_chars(self) -> None:
        text = " ".join(f"Sentence number {i} is here." for i in range(200))
        chunks = chunk_text(text, max_chars=300, overlap=50)
        assert len(chunks) > 5
        assert all(len(c) <= 300 for c in chunks)

    def test_paragraphs_kept_whole_when_they_fit(self) -> None:
        paras = ["Para one is short.", "Para two is also short.", "Para three."]
        chunks = chunk_text("\n\n".join(paras), max_chars=25, overlap=0)
        assert chunks == ["Para one is short.", "Para two is also short.", "Para three."]

    def test_paragraphs_merge_when_they_fit(self) -> None:
        chunks = chunk_text("aaa\n\nbbb\n\nccc", max_chars=100, overlap=0)
        assert chunks == ["aaa\n\nbbb\n\nccc"]

    def test_splits_on_sentence_boundaries(self) -> None:
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunk_text(text, max_chars=30, overlap=0)
        assert chunks == ["First sentence here.", "Second sentence here.", "Third sentence here."]

    def test_overlap_repeats_tail_of_previous_chunk(self) -> None:
        text = " ".join(f"Sentence number {i} is here." for i in range(40))
        chunks = chunk_text(text, max_chars=200, overlap=60)
        for prev, nxt in itertools.pairwise(chunks):
            tail_words = prev.split()[-3:]
            assert " ".join(tail_words) in nxt

    def test_overlap_zero_has_no_repeats(self) -> None:
        sentences = [f"Unique sentence {i}." for i in range(30)]
        chunks = chunk_text(" ".join(sentences), max_chars=100, overlap=0)
        joined = " ".join(chunks)
        for s in sentences:
            assert joined.count(s) == 1

    def test_long_unbroken_word_is_hard_split(self) -> None:
        chunks = chunk_text("x" * 1000, max_chars=100, overlap=10)
        assert all(len(c) <= 100 for c in chunks)
        assert "".join(chunks).count("x") >= 1000

    def test_no_content_lost(self) -> None:
        words = [f"w{i}" for i in range(500)]
        chunks = chunk_text(" ".join(words), max_chars=120, overlap=20)
        seen = {w for c in chunks for w in c.split()}
        assert seen == set(words)

    def test_defaults(self) -> None:
        text = ("Lorem ipsum dolor sit amet. " * 200).strip()
        chunks = chunk_text(text)
        assert all(len(c) <= 1200 for c in chunks)
        assert len(chunks) >= 4

    @pytest.mark.parametrize(
        ("max_chars", "overlap"), [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)]
    )
    def test_invalid_parameters(self, max_chars: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            chunk_text("text", max_chars=max_chars, overlap=overlap)


class TestCosine:
    def test_identical(self) -> None:
        assert cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite(self) -> None:
        assert cosine([1, 1], [-1, -1]) == pytest.approx(-1.0)

    def test_scale_invariant(self) -> None:
        assert cosine([1, 2], [10, 20]) == pytest.approx(1.0)

    def test_zero_vector_is_zero(self) -> None:
        assert cosine([0, 0], [1, 2]) == 0.0

    def test_numpy_input(self) -> None:
        assert cosine(np.array([1.0, 0.0]), np.array([1.0, 1.0])) == pytest.approx(2**-0.5)

    def test_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError):
            cosine([1, 2], [1, 2, 3])
