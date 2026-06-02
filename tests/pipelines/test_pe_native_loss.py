"""Tests for robust PE-native loss helpers."""

from __future__ import annotations

import numpy as np
import pytest

from microplex_us.pipelines.pe_native_loss import (
    build_pe_native_loss_arrays,
    pe_native_huber_loss,
    pe_native_huber_loss_terms,
)
from microplex_us.pipelines.pe_native_optimization import (
    _project_to_simplex,
    optimize_pe_native_loss_weights,
)


def test_bucketed_loss_downweights_tiny_baseline_outlier() -> None:
    targets = np.asarray([10.0, 1_000_000.0])
    names = [
        "nation/irs/estate losses/total/AGI in 100k-200k/taxable/All",
        "nation/irs/adjusted gross income/total/AGI in 500k-1m/taxable/All",
    ]
    loss_arrays = build_pe_native_loss_arrays(names, targets)

    estimate = np.asarray([410.0, 1_100_000.0])
    terms = pe_native_huber_loss_terms(estimate, loss_arrays)

    assert loss_arrays.target_weight[1] > loss_arrays.target_weight[0]
    assert terms[0] / terms.sum() < 0.05
    assert terms[1] > terms[0]


def test_robust_pe_native_optimizer_uses_huber_objective() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    target = np.asarray([1.0, 1.0])
    loss_arrays = build_pe_native_loss_arrays(
        [
            "nation/irs/example income/total/AGI in 0_1/taxable/All",
            "nation/irs/example income/total/AGI in 1_2/taxable/All",
        ],
        target,
    )
    initial_weights = np.asarray([0.0, 2.0])

    optimized, summary = optimize_pe_native_loss_weights(
        scaled_matrix=matrix,
        scaled_target=target,
        initial_weights=initial_weights,
        loss_arrays=loss_arrays,
        max_iter=100,
        tol=1e-10,
    )

    assert summary["optimized_loss"] < summary["initial_loss"]
    assert pe_native_huber_loss(matrix.T @ optimized, loss_arrays) == pytest.approx(
        summary["optimized_loss"]
    )
    assert optimized == pytest.approx(np.asarray([1.0, 1.0]), abs=1e-3)


def test_simplex_projection_preserves_large_population_total_exactly() -> None:
    values = np.asarray([50_000_000.0, 50_000_000.0, 53_768_767.0])
    target_total = values.sum() - 1_000.0

    projected = _project_to_simplex(values, target_total)

    assert projected.sum() == pytest.approx(target_total, abs=1e-6)
