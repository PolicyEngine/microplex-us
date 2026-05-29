"""Fast unit tests for the symmetric comparison logic (no PolicyEngine)."""

from __future__ import annotations

import numpy as np
import pytest

from mp_rebuild.compare import align_common, symmetric_holdout_compare
from mp_rebuild.pe_native import RawPeNative


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


def test_holdout_split_is_disjoint_and_shared():
    names = ["nation/x1", "state/s1", "state/s2", "state/s3", "state/s4"]
    t = [100.0, 50.0, 60.0, 70.0, 80.0]
    isnat = [True, False, False, False, False]
    rng = np.random.default_rng(2)
    raw = _raw(names, t, isnat, rng.normal(size=(15, 5)), rng.uniform(1, 5, 15))
    res = symmetric_holdout_compare(raw, raw, holdout_fraction=0.4, seed=3, max_steps=50)
    assert res.n_train_targets + res.n_holdout_targets == res.n_common_targets == 5
