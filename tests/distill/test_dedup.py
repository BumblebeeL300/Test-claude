"""Exact and near-duplicate detection on crafted vectors, plus decontamination."""

from __future__ import annotations

import numpy as np
import pytest

from nimlab.distill.decontam import EvalIndex
from nimlab.distill.dedup import (
    cluster_members,
    exact_duplicates,
    greedy_clusters,
    mean_pairwise_distance,
    normalize_rows,
)
from nimlab.distill.textutil import distinct_n, exact_key, normalize


def test_exact_key_ignores_case_punctuation_and_spacing() -> None:
    assert exact_key("Hello,  World!") == exact_key("hello world")
    assert normalize("A--B  c") == "a b c"
    assert exact_key("hello world") != exact_key("hello there")


def test_exact_duplicates_keep_the_highest_priority_member() -> None:
    keys = ["a", "b", "a", "c", "b", "a"]
    assert exact_duplicates(keys) == {2: 0, 4: 1, 5: 0}  # first occurrence wins by default
    dups = exact_duplicates(keys, priority=[0.1, 0.5, 0.9, 0.0, 0.2, 0.3])
    assert dups == {0: 2, 5: 2, 4: 1}  # the best-scored copy survives


def test_greedy_clusters_on_crafted_vectors() -> None:
    v = np.array(
        [
            [1.0, 0.0, 0.0],  # 0
            [0.99, 0.05, 0.0],  # 1  ~ 0
            [0.0, 1.0, 0.0],  # 2
            [0.02, 0.98, 0.0],  # 3  ~ 2
            [0.0, 0.0, 1.0],  # 4  alone
            [0.7, 0.7, 0.0],  # 5  between 0 and 2: similar to neither at 0.95
        ]
    )
    reps = greedy_clusters(v, 0.95)
    assert reps == [0, 0, 2, 2, 4, 5]
    assert cluster_members(reps) == {0: [0, 1], 2: [2, 3], 4: [4], 5: [5]}


def test_threshold_controls_cluster_size() -> None:
    v = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])  # cos(0,1)=0.8, cos(1,2)=0.6
    assert greedy_clusters(v, 0.9) == [0, 1, 2]
    assert greedy_clusters(v, 0.8) == [0, 0, 2]  # a similarity equal to the threshold counts
    # Only representatives are compared: row 2 resembles member 1 (0.6) but not representative 0.
    assert greedy_clusters(v, 0.5) == [0, 0, 2]


def test_priority_decides_who_represents_a_cluster() -> None:
    v = np.array([[1.0, 0.0], [0.99, 0.1], [0.98, 0.15]])
    assert greedy_clusters(v, 0.95) == [0, 0, 0]
    assert greedy_clusters(v, 0.95, priority=[0.0, 5.0, 1.0]) == [1, 1, 1]


def test_vectors_are_normalised_and_inputs_validated() -> None:
    assert np.allclose(np.linalg.norm(normalize_rows(np.array([[3.0, 4.0]])), axis=1), 1.0)
    assert normalize_rows(np.zeros((1, 3))).tolist() == [[0.0, 0.0, 0.0]]  # zero row stays zero
    with pytest.raises(ValueError, match="threshold"):
        greedy_clusters(np.eye(2), 0.0)
    with pytest.raises(ValueError, match="priority"):
        greedy_clusters(np.eye(2), 0.9, priority=[1.0])
    with pytest.raises(ValueError, match="2-D"):
        normalize_rows(np.zeros(3))
    assert greedy_clusters(np.zeros((0, 3)), 0.9) == []


def test_mean_pairwise_distance() -> None:
    assert mean_pairwise_distance(np.array([[1.0, 0.0], [1.0, 0.0]])) == pytest.approx(0.0)
    assert mean_pairwise_distance(np.eye(3)) == pytest.approx(1.0)
    assert mean_pairwise_distance(np.eye(1)) == 0.0
    big = np.random.default_rng(1).normal(size=(60, 8))
    assert mean_pairwise_distance(big, max_items=20, seed=3) == mean_pairwise_distance(
        big, max_items=20, seed=3
    )


def test_distinct_n() -> None:
    assert distinct_n(["a b c", "a b c"], 2) == pytest.approx(0.5)
    assert distinct_n(["a b c d"], 1) == 1.0
    assert distinct_n([], 2) == 0.0


# ------------------------------------------------------------------ decontamination

EVAL = [
    "What is the capital city of Australia and when did it become the capital?",
    "Prove that the square root of two is irrational.",
]


def test_decontamination_flags_verbatim_and_embedded_eval_questions() -> None:
    index = EvalIndex(EVAL, n=5)
    assert index.size == 2
    verbatim = "What is the capital city of Australia and when did it become the capital?"
    assert index.overlap(verbatim) == 1.0
    assert index.contaminated(verbatim, 0.5)
    wrapped = "Please answer: " + verbatim.lower() + " Keep it short."
    assert index.contaminated(wrapped, 0.5)
    assert 0 < index.overlap(wrapped) < 1


def test_decontamination_leaves_unrelated_text_alone() -> None:
    index = EvalIndex(EVAL, n=5)
    assert index.overlap("Explain how to bake sourdough bread at home today") == 0.0
    assert not index.contaminated("Explain how to bake sourdough bread at home today", 0.1)


def test_short_texts_only_match_as_a_whole_and_empty_index_never_matches() -> None:
    index = EvalIndex(EVAL, n=8)
    assert index.overlap("Prove that the square root of two is irrational.") == 1.0
    assert index.overlap("square root of two") == 0.0
    assert not EvalIndex([], n=8).contaminated("anything at all", 0.1)


def test_ngram_size_changes_sensitivity() -> None:
    text = "the quick brown fox jumps over the lazy dog near the river bank"
    probe = "a quick brown fox jumps over a sleepy cat"
    assert EvalIndex([text], n=3).overlap(probe) > 0
    assert EvalIndex([text], n=8).overlap(probe) == 0
