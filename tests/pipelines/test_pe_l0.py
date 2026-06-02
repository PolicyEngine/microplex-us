"""Tests for the PolicyEngine L0 calibrator adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from microplex.calibration import LinearConstraint

from microplex_us.pipelines.pe_l0 import PolicyEngineL0Calibrator


def _install_fake_policyengine_l0(weights: np.ndarray):
    calls: dict[str, object] = {}

    def fake_fit_l0_weights(**kwargs):
        calls.update(kwargs)
        return np.asarray(weights, dtype=float)

    return calls, fake_fit_l0_weights


def test_policyengine_l0_calibrator_supports_explicit_linear_constraints():
    calls, fake_fit_l0_weights = _install_fake_policyengine_l0(np.array([1.0, 2.0]))
    data = pd.DataFrame({"weight": [1.0, 1.0]})
    constraints = (
        LinearConstraint("row1", np.array([1.0, 0.0]), 1.0),
        LinearConstraint("row2", np.array([0.0, 1.0]), 2.0),
    )

    calibrator = PolicyEngineL0Calibrator(
        lambda_l0=1e-4,
        lambda_l2=1e-12,
        beta=0.35,
        learning_rate=0.15,
        epochs=25,
        tol=1e-6,
        device="cpu",
        fit_l0_weights_fn=fake_fit_l0_weights,
    )
    result = calibrator.fit_transform(
        data,
        {},
        weight_col="weight",
        linear_constraints=constraints,
    )
    validation = calibrator.validate(result)

    assert result["weight"].tolist() == [1.0, 2.0]
    assert calls["X_sparse"].shape == (2, 2)
    assert calls["target_names"] == ["row1", "row2"]
    assert calls["targets"].tolist() == [1.0, 2.0]
    assert calls["initial_weights"].tolist() == [1.0, 1.0]
    assert validation["converged"] is True
    assert validation["max_error"] < 1e-9
    assert validation["sparsity"] == 0.0


def test_policyengine_l0_calibrator_reports_sparsity():
    _, fake_fit_l0_weights = _install_fake_policyengine_l0(np.array([0.0, 3.0, 0.0]))
    data = pd.DataFrame({"weight": [1.0, 1.0, 1.0]})
    constraints = (LinearConstraint("row", np.array([0.0, 1.0, 0.0]), 3.0),)

    calibrator = PolicyEngineL0Calibrator(
        epochs=5,
        tol=1e-6,
        fit_l0_weights_fn=fake_fit_l0_weights,
    )
    calibrator.fit(
        data,
        {},
        weight_col="weight",
        linear_constraints=constraints,
    )

    assert calibrator.get_sparsity() == 2 / 3


def test_policyengine_l0_lambda_zero_uses_dense_no_gate_path(monkeypatch):
    calls, fake_fit_l0_weights = _install_fake_policyengine_l0(np.array([99.0, 99.0]))
    data = pd.DataFrame({"weight": [1.0, 1.0]})
    constraints = (
        LinearConstraint("row1", np.array([1.0, 0.0]), 2.0),
        LinearConstraint("row2", np.array([0.0, 1.0]), 3.0),
    )

    calibrator = PolicyEngineL0Calibrator(
        lambda_l0=0.0,
        lambda_l2=0.0,
        epochs=100,
        tol=1e-10,
        fit_l0_weights_fn=fake_fit_l0_weights,
    )
    result = calibrator.fit_transform(
        data,
        {},
        weight_col="weight",
        linear_constraints=constraints,
    )
    validation = calibrator.validate(result)

    assert calls == {}
    assert calibrator.effective_backend_ == "dense_projected_gradient"
    assert validation["backend"] == "dense_projected_gradient"
    assert validation["uses_gates"] is False
    assert validation["loss_history"][0]["iteration"] == 0
    assert (
        validation["loss_history"][-1]["objective_loss"]
        < validation["loss_history"][0]["objective_loss"]
    )
    assert result["weight"].to_numpy(dtype=float) == pytest.approx(
        [2.0, 3.0],
        rel=1e-5,
    )


def test_policyengine_l0_requires_explicit_fit_function_for_nonzero_l0():
    data = pd.DataFrame({"weight": [1.0]})
    constraints = (LinearConstraint("row", np.array([1.0]), 1.0),)

    calibrator = PolicyEngineL0Calibrator(lambda_l0=1e-4, epochs=1)

    with pytest.raises(RuntimeError, match="no longer loads policyengine-us-data"):
        calibrator.fit(
            data,
            {},
            weight_col="weight",
            linear_constraints=constraints,
        )
