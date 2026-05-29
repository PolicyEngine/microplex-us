"""Fast unit tests for the symmetric comparison logic (no PolicyEngine)."""

from __future__ import annotations

import numpy as np
import pytest

from mp_rebuild.compare import align_common, symmetric_holdout_compare
from mp_rebuild.pe_native import RawPeNative, build_scaled_problem
from mp_rebuild.scoreboard import score


def _raw(names, targets, is_national, matrix, weights):
    return RawPeNative(
        raw_matrix=np.asarray(matrix, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float64),
        names=list(names),
        is_national=np.asarray(is_national, dtype=bool),
        weights=np.asarray(weights, dtype=np.float64),
    )


def test_align_common_intersects_and_shares_scaling():
    names_a = ["nation/x1", "nation/x2", "state/s1", "state/s2", "state/s3"]
    names_b = ["nation/x1", "state/s1", "state/s2", "nation/zz"]
    ta = [100.0, 200.0, 50.0, 60.0, 70.0]
    tb = [100.0, 50.0, 60.0, 999.0]  # shared names carry identical target values
    rng = np.random.default_rng(0)
    raw_a = _raw(names_a, ta, [True, True, False, False, False],
                 rng.normal(size=(30, 5)), rng.uniform(1, 5, 30))
    raw_b = _raw(names_b, tb, [True, False, False, True],
                 rng.normal(size=(20, 4)), rng.uniform(1, 5, 20))

    prob_a, prob_b, common, tdis = align_common(raw_a, raw_b)

    assert common == ["nation/x1", "state/s1", "state/s2"]
    assert tdis == 0.0  # shared targets agree
    assert prob_a.n_targets == prob_b.n_targets == 3
    # Identical scaling => identical scaled target vectors on both sides.
    assert np.allclose(prob_a.target, prob_b.target)
    assert prob_a.reduction == "sum"


def test_identical_datasets_compare_equal():
    """Symmetry: comparing a dataset against an identical copy yields equal losses."""
    names = ["nation/x1", "state/s1", "state/s2", "state/s3"]
    t = [100.0, 50.0, 60.0, 70.0]
    isnat = [True, False, False, False]
    rng = np.random.default_rng(1)
    m = rng.normal(size=(25, 4)).astype(np.float32)
    w = rng.uniform(1, 5, 25)
    raw = _raw(names, t, isnat, m, w)
    raw_copy = _raw(list(names), list(t), list(isnat), m.copy(), w.copy())

    res = symmetric_holdout_compare(
        raw, raw_copy, holdout_fraction=0.5, seed=0, max_steps=200
    )
    assert res.candidate_holdout_loss == pytest.approx(res.baseline_holdout_loss, abs=1e-9)
    assert res.candidate_full_refit_loss == pytest.approx(res.baseline_full_refit_loss, abs=1e-9)
    assert "PARITY" in res.verdict()
    assert res.n_holdout_targets == 2


def test_pe_native_scaling_matches_reference_loss():
    """The scaled `sum` form must equal the canonical per-target-weighted PE-native loss.

    Independent reimplementation of the reference metric (mirrors
    pe_native_scores.compute / _extract_batch_matrix): rel_err =
    ((estimate - target) + 1)/(target + 1), weighted by per_target_weight with
    national/state balancing. Locks the (target-1) shift + inv_mean scaling.
    """
    rng = np.random.default_rng(7)
    n_t, n_h = 40, 60
    is_national = np.array([True] * 12 + [False] * 28)
    targets = rng.uniform(10.0, 1000.0, size=n_t)
    raw = rng.normal(size=(n_h, n_t)).astype(np.float32)  # households x targets
    w = rng.uniform(1.0, 5.0, size=n_h)

    n_nat = int(is_national.sum())
    n_state = int((~is_national).sum())
    norm = np.where(is_national, 1.0 / n_nat, 1.0 / n_state)
    per_target_weight = (1.0 / np.mean(norm)) * norm / n_t
    estimate = raw.astype(np.float64).T @ w
    rel_err = ((estimate - targets) + 1.0) / (targets + 1.0)
    loss_ref = float(np.sum(per_target_weight * rel_err**2))

    prob = build_scaled_problem(raw, targets, is_national, [f"t{i}" for i in range(n_t)])
    assert score(prob, w) == pytest.approx(loss_ref, rel=1e-4)


def test_align_common_rejects_duplicate_names():
    rng = np.random.default_rng(0)
    raw = RawPeNative(
        raw_matrix=rng.normal(size=(5, 3)).astype(np.float32),
        targets=np.array([100.0, 50.0, 60.0]),
        names=["nation/x", "state/s", "state/s"],  # duplicate
        is_national=np.array([True, False, False]),
        weights=rng.uniform(1, 5, 5),
    )
    with pytest.raises(ValueError, match="unique"):
        align_common(raw, raw)


def test_holdout_split_is_disjoint_and_shared():
    names = ["nation/x1", "state/s1", "state/s2", "state/s3", "state/s4"]
    t = [100.0, 50.0, 60.0, 70.0, 80.0]
    isnat = [True, False, False, False, False]
    rng = np.random.default_rng(2)
    raw = _raw(names, t, isnat, rng.normal(size=(15, 5)), rng.uniform(1, 5, 15))
    res = symmetric_holdout_compare(raw, raw, holdout_fraction=0.4, seed=3, max_steps=50)
    assert res.n_train_targets + res.n_holdout_targets == res.n_common_targets == 5
