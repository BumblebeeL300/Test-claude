from __future__ import annotations

import numpy as np
import pytest

from nimlab.council.consensus import (
    agreement_score,
    build_report,
    centrality,
    cosine_matrix,
    find_outliers,
    lexical_vectors,
)


def test_cosine_matrix_identical_orthogonal_and_opposite() -> None:
    m = cosine_matrix([[1, 0], [1, 0], [0, 1], [-1, 0]])
    assert m.shape == (4, 4)
    assert m[0, 1] == pytest.approx(1.0)
    assert m[0, 2] == pytest.approx(0.0)
    assert m[0, 3] == pytest.approx(-1.0)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 1.0)


def test_cosine_matrix_is_scale_invariant() -> None:
    m = cosine_matrix([[1, 2, 3], [10, 20, 30]])
    assert m[0, 1] == pytest.approx(1.0)


def test_cosine_matrix_zero_vector_has_zero_similarity() -> None:
    m = cosine_matrix([[0, 0], [1, 1]])
    assert m[0, 1] == 0.0
    assert m[0, 0] == 1.0  # diagonal is pinned to 1


def test_cosine_matrix_empty_and_bad_shape() -> None:
    assert cosine_matrix([]).shape == (0, 0)
    with pytest.raises(ValueError):
        cosine_matrix([[[1, 2]], [[3, 4]]])


def test_agreement_is_mean_off_diagonal_clipped() -> None:
    m = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.4], [0.6, 0.4, 1.0]])
    assert agreement_score(m) == pytest.approx((0.8 + 0.6 + 0.4) / 3)
    assert agreement_score(np.array([[1.0]])) == 1.0
    assert agreement_score(np.array([[1.0, -0.5], [-0.5, 1.0]])) == 0.0  # clipped, never negative


def test_centrality_is_mean_similarity_to_others() -> None:
    m = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
    assert centrality(m) == pytest.approx([0.5, 0.5, 0.1])
    assert centrality(np.array([[1.0]])) == pytest.approx([1.0])


def test_find_outliers_flags_the_odd_one_out() -> None:
    # three answers that agree and one that does not
    vectors = [[1, 0.02], [1, 0.05], [1, 0.0], [0, 1]]
    report = build_report(vectors, labels=list("ABCD"), models=list("wxyz"), phase="p")
    assert report.outliers == [3]
    assert report.outlier_labels == ["D"]
    assert report.medoid in (0, 1, 2)
    assert report.agreement < 0.8


def test_find_outliers_none_when_everyone_agrees() -> None:
    vectors = [[1, 0.01], [1, 0.02], [1, 0.0], [1, 0.03]]
    report = build_report(vectors, labels=list("ABCD"), models=list("wxyz"), phase="p")
    assert report.outliers == []
    assert report.agreement > 0.99


def test_find_outliers_needs_three_items_and_respects_margin() -> None:
    assert find_outliers(np.array([0.9, 0.1]), min_items=3) == []
    # a spread smaller than the absolute margin is not flagged even though z is high
    cent = np.array([0.90, 0.90, 0.90, 0.85])
    assert find_outliers(cent, z=0.5, margin=0.1) == []
    assert find_outliers(cent, z=0.5, margin=0.01) == [3]


def test_build_report_shape_rounding_and_roundtrip() -> None:
    vectors = [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    report = build_report(
        vectors,
        labels=["A1", "A2", "A3"],
        models=["m/a", "m/b", "m/c"],
        phase="Round 1",
        embed_model="emb",
    )
    assert report.size == 3 and report.phase == "Round 1" and report.embed_model == "emb"
    assert len(report.matrix) == 3 and all(len(r) == 3 for r in report.matrix)
    assert report.matrix[0][1] == pytest.approx(1 / np.sqrt(2), abs=1e-5)
    assert report.min_pairwise == pytest.approx(0.0)
    assert 0.0 <= report.agreement <= 1.0
    assert type(report).model_validate_json(report.model_dump_json()) == report


def test_build_report_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        build_report([[1, 0]], labels=["a", "b"], models=["m"], phase="p")


def test_build_report_single_answer() -> None:
    report = build_report([[1.0, 0.0]], labels=["only"], models=["m"], phase="p")
    assert report.agreement == 1.0
    assert report.outliers == []
    assert report.medoid == 0


def test_lexical_vectors_shared_words_score_higher() -> None:
    v = lexical_vectors(
        [
            "the sky looks blue because of rayleigh scattering of sunlight",
            "rayleigh scattering of sunlight makes the sky blue",
            "bananas are rich in potassium and grow in tropical climates",
        ]
    )
    m = cosine_matrix(v)
    assert m[0, 1] > 0.5
    assert m[0, 2] < 0.1
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_lexical_vectors_deterministic_and_handles_empty() -> None:
    a = lexical_vectors(["same text here", ""])
    b = lexical_vectors(["same text here", ""])
    assert np.array_equal(a, b)
    assert np.linalg.norm(a[1]) == 0.0
    assert cosine_matrix(a)[0, 1] == 0.0
