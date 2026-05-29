"""Fast, pure-numpy invariants the scoreboard must satisfy.

These encode the properties whose absence made the old benchmark unsound. They
run in milliseconds (no PolicyEngine, no H5) so the loop can keep them green on
every iteration. The expensive end-to-end recovery check lives in
``test_ecps_recovery.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mp_rebuild import (
    CalibrationProblem,
    compare_problems,
    fit,
    score,
    split_targets,
)


def _random_problem(seed: int, n_targets: int = 20, n_records: int = 200):
    """A feasible problem: target = M @ w_true for a strictly positive w_true."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n_targets, n_records))
    w_true = rng.uniform(0.5, 5.0, size=n_records)
    target = matrix @ w_true
    return CalibrationProblem(matrix=matrix, target=target), w_true


def test_score_is_the_problem_loss():
    """Scoring must be literally the same function fitting minimises."""
    problem, w_true = _random_problem(0)
    rng = np.random.default_rng(1)
    for _ in range(5):
        w = rng.uniform(0.1, 3.0, size=problem.n_records)
        assert score(problem, w) == problem.loss(w)


@pytest.mark.parametrize("seed", range(8))
def test_fit_never_increases_loss(seed):
    """THE recovery invariant, unit level.

    The operator must never make a dataset worse on the scored loss. The old
    harness violated this (eCPS 0.166 -> 0.544). Here it holds by construction.
    """
    problem, _ = _random_problem(seed)
    rng = np.random.default_rng(100 + seed)
    init = rng.uniform(0.1, 5.0, size=problem.n_records)
    initial_loss = score(problem, init)
    fitted = fit(problem, init)
    final_loss = score(problem, fitted)
    assert final_loss <= initial_loss + 1e-9


def test_fit_from_optimum_stays_put():
    """Refitting an already-optimal dataset must not degrade it.

    This is the toy analogue of "refit eCPS recovers ~0.166". Starting at a
    near-zero-loss solution, the operator must keep the loss near zero, not
    blow it up.
    """
    problem, w_true = _random_problem(7)
    assert score(problem, w_true) < 1e-12  # w_true is exact by construction
    fitted = fit(problem, w_true)
    assert score(problem, fitted) <= 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_fit_actually_reduces_loss_from_bad_start(seed):
    """Sanity: the operator is not a no-op; it makes a bad start meaningfully better."""
    problem, _ = _random_problem(seed)
    rng = np.random.default_rng(500 + seed)
    init = rng.uniform(5.0, 20.0, size=problem.n_records)  # deliberately off
    initial_loss = score(problem, init)
    final_loss = score(problem, fit(problem, init))
    assert final_loss < 0.5 * initial_loss


def test_fit_keeps_weights_nonnegative_by_default():
    problem, _ = _random_problem(3)
    rng = np.random.default_rng(9)
    init = rng.uniform(0.1, 5.0, size=problem.n_records)
    fitted = fit(problem, init)
    assert np.all(fitted >= 0.0)


def test_compare_is_symmetric():
    """Swapping candidate/baseline swaps the reported scores. No one-sided refit."""
    prob_a, w_a = _random_problem(11)
    prob_b, w_b = _random_problem(12)
    rng = np.random.default_rng(13)
    init_a = rng.uniform(1.0, 6.0, size=prob_a.n_records)
    init_b = rng.uniform(1.0, 6.0, size=prob_b.n_records)

    forward = compare_problems(prob_a, prob_b, init_a, init_b)
    swapped = compare_problems(prob_b, prob_a, init_b, init_a)

    assert forward["candidate_loss"] == pytest.approx(swapped["baseline_loss"])
    assert forward["baseline_loss"] == pytest.approx(swapped["candidate_loss"])
    # Both sides are actually fit (initial losses recorded, finals are better).
    assert forward["candidate_loss"] <= forward["candidate_loss_initial"] + 1e-9
    assert forward["baseline_loss"] <= forward["baseline_loss_initial"] + 1e-9


def test_split_targets_is_a_clean_partition():
    problem, _ = _random_problem(4, n_targets=50)
    train, holdout = split_targets(problem, holdout_fraction=0.3, seed=42)
    assert train.n_targets + holdout.n_targets == problem.n_targets
    assert holdout.n_targets == 15  # round(0.3 * 50)
    # Same record space on both sides.
    assert train.n_records == holdout.n_records == problem.n_records


def test_split_targets_is_deterministic_in_seed():
    problem, _ = _random_problem(5, n_targets=40)
    a1, b1 = split_targets(problem, 0.25, seed=7)
    a2, b2 = split_targets(problem, 0.25, seed=7)
    assert np.array_equal(a1.target, a2.target)
    assert np.array_equal(b1.target, b2.target)
