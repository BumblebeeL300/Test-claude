"""BM25 correctness on a toy corpus, plus tokenizer behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nimlab.atlas.bm25 import BM25Index
from nimlab.atlas.tokens import stem, tokenize

DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps all day long in the sun",
    "quick quick quick foxes are quick",
    "a completely unrelated sentence about batteries and chargers",
]


def build(docs=DOCS, **kw) -> BM25Index:
    return BM25Index.build([d.split() for d in docs], **kw)


def test_idf_matches_the_lucene_formula():
    index = build()
    n, df = 4, 2  # "lazy" occurs in docs 0 and 1
    assert index.idf("lazy") == pytest.approx(math.log(1 + (n - df + 0.5) / (df + 0.5)))
    assert index.idf("batteries") > index.idf("lazy") > 0
    assert index.idf("never-seen") == 0.0


def test_score_matches_hand_computation():
    index = build(k1=1.2, b=0.75)
    lengths = [len(d.split()) for d in DOCS]
    avgdl = sum(lengths) / len(lengths)
    tf = 1  # "fox" appears once in doc 0
    idf = math.log(1 + (4 - 1 + 0.5) / (1 + 0.5))
    expected = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * lengths[0] / avgdl))
    assert index.scores(["fox"])[0] == pytest.approx(expected, rel=1e-5)
    assert index.scores(["fox"])[1] == 0.0


def test_ranking_prefers_term_frequency_then_shorter_documents():
    index = build()
    ranked = [i for i, _ in index.search(["quick"], 10)]
    assert ranked[0] == 2  # four occurrences beat one
    assert ranked == [2, 0]
    ranked = [i for i, _ in index.search(["lazy", "dog"], 10)]
    assert set(ranked) == {0, 1}
    assert ranked[0] == 0  # shorter document wins at equal tf


def test_rare_terms_outweigh_common_terms():
    index = build()
    scores = index.scores(["the", "batteries"])
    assert int(np.argmax(scores)) == 3


def test_b_zero_disables_length_normalisation_and_k1_zero_ignores_tf():
    docs = ["cat", "cat filler filler filler filler filler", "dog"]
    flat = BM25Index.build([d.split() for d in docs], b=0.0)
    s = flat.scores(["cat"])
    assert s[0] == pytest.approx(s[1])
    normed = BM25Index.build([d.split() for d in docs], b=1.0)
    s = normed.scores(["cat"])
    assert s[0] > s[1]
    binary = BM25Index.build([["cat"] * 5, ["cat"], ["dog"]], k1=0.0)
    s = binary.scores(["cat"])
    assert s[0] == pytest.approx(s[1])


def test_repeated_query_terms_count_once():
    index = build()
    assert np.allclose(index.scores(["fox", "fox", "fox"]), index.scores(["fox"]))


def test_unseen_terms_empty_query_and_empty_index():
    index = build()
    assert index.search(["zzz"], 5) == []
    assert index.search([], 5) == []
    assert index.search(["fox"], 0) == []
    empty = BM25Index.build([])
    assert empty.search(["fox"], 5) == [] and len(empty) == 0


def test_ties_break_by_document_position_and_k_truncates():
    index = BM25Index.build([["same", "words"], ["same", "words"], ["same", "words"]])
    assert [i for i, _ in index.search(["same"], 10)] == [0, 1, 2]
    assert [i for i, _ in index.search(["same"], 2)] == [0, 1]


def test_scores_are_positive_and_sorted_descending():
    index = build()
    hits = index.search(["quick", "fox", "dog"], 10)
    scores = [s for _, s in hits]
    assert all(s > 0 for s in scores) and scores == sorted(scores, reverse=True)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        BM25Index(k1=-1)
    with pytest.raises(ValueError):
        BM25Index(b=1.5)


def test_tokenizer_lowercases_drops_stopwords_and_splits_punctuation():
    assert tokenize("The API-key, v2.5!") == ["api", "key", "v2", "5"]
    assert tokenize("The API", stopwords=False) == ["the", "api"]
    assert tokenize("Tokens", stemming=False) == ["tokens"]
    assert tokenize("naive caf\xe9", stemming=False) == ["naive", "caf\xe9"]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("tokens", "token"),
        ("rotating", "rotated"),
        ("rotating", "rotate"),
        ("policies", "policy"),
        ("heartbeats", "heartbeat"),
    ],
)
def test_stemmer_conflates_inflections(a: str, b: str):
    assert stem(a) == stem(b)


def test_stemmer_leaves_short_words_numbers_and_ss_words_alone():
    assert stem("429") == "429"
    assert stem("bus") == "bus"
    assert stem("access") == "access"
    assert stem("is") == "is"
