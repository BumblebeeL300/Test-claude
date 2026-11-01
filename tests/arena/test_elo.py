"""Bradley-Terry fitting and bootstrap intervals on synthetic data."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nimlab.arena.elo import (
    Outcome,
    bootstrap_elo,
    elo_scale,
    expected_score,
    fit_bradley_terry,
    outcomes_from_scores,
)

TRUE = {"a": 1250.0, "b": 1100.0, "c": 1000.0, "d": 900.0, "e": 750.0}


def simulate(n_games: int, seed: int = 0, truth: dict[str, float] = TRUE) -> list[Outcome]:
    rng = np.random.default_rng(seed)
    names = list(truth)
    out = []
    for _ in range(n_games):
        i, j = rng.choice(len(names), 2, replace=False)
        a, b = names[i], names[j]
        won = rng.random() < expected_score(truth[a], truth[b])
        out.append(Outcome(a, b, 1.0 if won else 0.0))
    return out


def test_expected_score_is_the_elo_logistic() -> None:
    assert expected_score(1000, 1000) == pytest.approx(0.5)
    assert expected_score(1400, 1000) == pytest.approx(10 / 11)  # 400 points = 10:1 odds
    assert expected_score(1000, 1400) == pytest.approx(1 / 11)


def test_fit_recovers_the_true_ordering_and_scale() -> None:
    fit = fit_bradley_terry(simulate(2500))
    ranked = sorted(fit.elo, key=lambda m: -fit.elo[m])
    assert ranked == ["a", "b", "c", "d", "e"]
    assert fit.converged and fit.iterations < 2000
    # ratings are identified up to a shift: compare differences to the truth
    mean_true = np.mean(list(TRUE.values()))
    for m, true in TRUE.items():
        assert fit.elo[m] - 1000 == pytest.approx(true - mean_true, abs=40)
    assert np.mean(list(fit.elo.values())) == pytest.approx(1000, abs=1.0)  # centred at the base
    assert fit.games.sum() == 2 * 2500


def test_fit_matches_pairwise_win_probabilities() -> None:
    fit = fit_bradley_terry(simulate(4000, seed=3))
    p = fit.strengths
    idx = {m: i for i, m in enumerate(fit.models)}
    est = p[idx["a"]] / (p[idx["a"]] + p[idx["e"]])
    assert est == pytest.approx(expected_score(TRUE["a"], TRUE["e"]), abs=0.04)


def test_fractional_scores_and_ties_are_supported() -> None:
    ties = [Outcome("x", "y", 0.5)] * 20
    fit = fit_bradley_terry(ties)
    assert fit.elo["x"] == pytest.approx(fit.elo["y"], abs=1e-6)
    mixed = [Outcome("x", "y", 0.75)] * 40
    f2 = fit_bradley_terry(mixed)
    assert f2.elo["x"] > f2.elo["y"]
    # 0.75 win share => 400 * log10(3) elo points apart (less due to the weak prior)
    assert 0 < f2.elo["x"] - f2.elo["y"] <= 400 * math.log10(3) + 1


def test_undefeated_models_get_finite_ratings() -> None:
    outcomes = (
        [Outcome("w", "l", 1.0)] * 10
        + [Outcome("w", "m", 1.0)] * 10
        + [Outcome("m", "l", 1.0)] * 10
    )
    fit = fit_bradley_terry(outcomes)
    assert all(math.isfinite(v) for v in fit.elo.values())
    assert fit.elo["w"] > fit.elo["m"] > fit.elo["l"]


def test_unplayed_models_stay_at_the_base_rating() -> None:
    fit = fit_bradley_terry([Outcome("a", "b", 1.0)] * 5, models=["a", "b", "ghost"])
    assert fit.elo["ghost"] == pytest.approx(1000.0)
    assert fit.games[fit.models.index("ghost")] == 0


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="no models"):
        fit_bradley_terry([])
    with pytest.raises(ValueError, match="scores must lie"):
        fit_bradley_terry([Outcome("a", "b", 1.5)])
    # self-play and models outside the requested list are ignored
    fit = fit_bradley_terry([Outcome("a", "a", 1.0), Outcome("a", "z", 1.0)], models=["a", "b"])
    assert fit.games.sum() == 0


def test_elo_scale_is_400_per_decade_of_strength() -> None:
    assert list(elo_scale(np.array([1.0, 10.0, 0.1]))) == pytest.approx([1000, 1400, 600])


def test_bootstrap_confidence_intervals_bracket_the_estimate_and_the_truth() -> None:
    outcomes = simulate(1500, seed=5)
    ratings = bootstrap_elo(outcomes, n_boot=150, seed=1)
    shift = np.mean(list(TRUE.values())) - 1000
    covered = 0
    for m, r in ratings.items():
        assert r.lo <= r.elo <= r.hi
        assert r.games > 0 and r.half_width > 5
        covered += r.lo - 15 <= TRUE[m] - shift <= r.hi + 15
    assert covered >= 4  # a 95% interval should cover (nearly) every true rating


def test_bootstrap_intervals_shrink_with_more_games() -> None:
    small = bootstrap_elo(simulate(200, seed=2), n_boot=100, seed=0)
    large = bootstrap_elo(simulate(3000, seed=2), n_boot=100, seed=0)
    assert np.mean([r.half_width for r in large.values()]) < 0.6 * np.mean(
        [r.half_width for r in small.values()]
    )


def test_bootstrap_is_deterministic_for_a_seed_and_varies_across_seeds() -> None:
    outcomes = simulate(300, seed=7)
    a = bootstrap_elo(outcomes, n_boot=50, seed=11)
    b = bootstrap_elo(outcomes, n_boot=50, seed=11)
    c = bootstrap_elo(outcomes, n_boot=50, seed=12)
    assert a == b
    assert a["a"].lo != c["a"].lo


def test_bootstrap_without_resamples_or_games() -> None:
    point = bootstrap_elo(simulate(100), n_boot=0)
    assert all(r.lo == r.hi == r.elo for r in point.values())
    empty = bootstrap_elo([], models=["a", "b"], n_boot=10)
    assert empty["a"].elo == 1000 and empty["a"].games == 0


def test_two_decisive_models_have_wide_intervals_with_few_games() -> None:
    r = bootstrap_elo([Outcome("a", "b", 1.0)] * 3 + [Outcome("a", "b", 0.0)], n_boot=100)
    assert r["a"].elo > r["b"].elo
    assert r["a"].half_width > 20


def test_outcomes_from_scores_pairs_models_per_item_with_ties() -> None:
    per_item = {
        ("s", "i1"): {"a": 1.0, "b": 0.0, "c": 1.0},
        ("s", "i2"): {"a": 0.5, "b": 0.52},  # within the tie margin
        ("s", "i3"): {"a": 0.2},  # a single model plays nobody
    }
    out = outcomes_from_scores(per_item)
    assert len(out) == 3 + 1
    scores = {(o.a, o.b): o.score_a for o in out[:3]}
    assert scores == {("a", "b"): 1.0, ("a", "c"): 0.5, ("b", "c"): 0.0}
    assert out[3] == Outcome("a", "b", 0.5)


def test_elo_from_graded_scores_orders_models_by_skill() -> None:
    rng = np.random.default_rng(0)
    skill = {"strong": 0.9, "mid": 0.6, "weak": 0.3}
    per_item = {
        ("s", str(i)): {m: float(rng.random() < p) for m, p in skill.items()} for i in range(120)
    }
    fit = fit_bradley_terry(outcomes_from_scores(per_item))
    assert fit.elo["strong"] > fit.elo["mid"] > fit.elo["weak"]
