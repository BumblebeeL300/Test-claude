"""Reciprocal Rank Fusion, MMR and context packing."""

from __future__ import annotations

import numpy as np
import pytest

from nimlab.atlas.fusion import (
    minmax,
    mmr_select,
    pack_context,
    reciprocal_rank_fusion,
    truncate_at_boundary,
)


def test_rrf_math_is_exact():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_order_rewards_agreement():
    ranked = [item for item, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]])]
    assert ranked == ["b", "c", "a", "d"]


def test_rrf_weights_scale_each_ranking():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=10, weights=[2.0, 0.5]))
    assert fused["a"] == pytest.approx(2.0 / 11)
    assert fused["b"] == pytest.approx(0.5 / 11)
    ranked = [i for i, _ in reciprocal_rank_fusion([["a"], ["b"]], weights=[0.0, 1.0])]
    assert ranked[0] == "b"


def test_rrf_ties_keep_first_seen_order_and_duplicates_count_once():
    ranked = [i for i, _ in reciprocal_rank_fusion([["x", "y"], ["y", "x"]])]
    assert ranked == ["x", "y"]
    fused = dict(reciprocal_rank_fusion([["a", "a", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)


def test_rrf_edge_cases():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1, 2])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_minmax_scales_and_handles_constants():
    assert minmax([1, 2, 3]).tolist() == [0.0, 0.5, 1.0]
    assert minmax([5, 5]).tolist() == [1.0, 1.0]
    assert minmax([]).size == 0


def test_mmr_with_lambda_one_is_pure_relevance():
    rel = np.array([0.9, 0.8, 0.7, 0.1])
    vecs = np.eye(4)
    assert mmr_select(rel, vecs, 3, lambda_=1.0) == [0, 1, 2]


def test_mmr_penalises_near_duplicates():
    rel = np.array([1.0, 0.99, 0.6])
    vecs = np.array([[1.0, 0.0], [0.999, 0.04], [0.0, 1.0]])  # items 0 and 1 are near copies
    assert mmr_select(rel, vecs, 2, lambda_=1.0) == [0, 1]
    assert mmr_select(rel, vecs, 2, lambda_=0.5) == [0, 2]


def test_mmr_lambda_zero_maximises_diversity_after_first_pick():
    rel = np.array([1.0, 0.9, 0.8])
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    picked = mmr_select(rel, vecs, 2, lambda_=0.0)
    assert picked[0] == 0 and picked[1] == 2


def test_mmr_returns_each_candidate_once_and_clamps_k():
    rel = np.array([0.5, 0.4, 0.3])
    picked = mmr_select(rel, np.random.default_rng(0).normal(size=(3, 5)), 10, 0.7)
    assert sorted(picked) == [0, 1, 2]
    assert mmr_select(rel, np.eye(3), 0) == []


def test_mmr_validates_inputs():
    with pytest.raises(ValueError):
        mmr_select([1.0], np.eye(2), 1)
    with pytest.raises(ValueError):
        mmr_select([1.0], np.eye(1), 1, lambda_=1.5)


def test_pack_context_respects_budget_and_skips_oversized_items():
    texts = ["a" * 40, "b" * 70, "c" * 30, "d" * 20]
    packed = pack_context(texts, budget_chars=100, max_items=10)
    assert [i for i, _ in packed] == [0, 2, 3]
    assert sum(len(t) for _, t in packed) <= 100


def test_pack_context_limits_item_count():
    assert [i for i, _ in pack_context(["x"] * 10, 1000, 3)] == [0, 1, 2]


def test_pack_context_truncates_a_too_large_first_item():
    text = "First sentence here. Second sentence follows. " * 20
    ((index, packed),) = pack_context([text], budget_chars=120, max_items=5)
    assert index == 0 and len(packed) <= 120 and packed.endswith("...")


def test_truncate_prefers_sentence_boundaries():
    out = truncate_at_boundary("One two three four five. Six seven eight nine ten eleven.", 30)
    assert out == "One two three four five. ..."
    assert truncate_at_boundary("short", 100) == "short"
    assert len(truncate_at_boundary("w" * 50, 20)) <= 20
