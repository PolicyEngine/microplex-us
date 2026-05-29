"""Pipeline-level test: `calibration_backend="microcalibrate"` dispatches to
`MicrocalibrateAdapter` and round-trips one calibration call inside the
USMicroplexPipeline context.

This is the final link between the adapter and the production pipeline:
the backend string needs to be valid in `USMicroplexBuildConfig`, and
`_build_weight_calibrator` must return an adapter instance that
satisfies the same `fit_transform` / `validate` contract the rest of
`calibrate_policyengine_tables` expects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from microplex.calibration import LinearConstraint

from microplex_us.calibration import MicrocalibrateAdapter
from microplex_us.pipelines.us import USMicroplexBuildConfig, USMicroplexPipeline


def _toy_households(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "household_id": np.arange(n),
            "household_weight": np.ones(n, dtype=float),
            "income": rng.normal(80_000, 40_000, n).clip(0, None),
        }
    )


def test_backend_string_resolves_to_adapter() -> None:
    cfg = USMicroplexBuildConfig(calibration_backend="microcalibrate")
    pipeline = USMicroplexPipeline(cfg)
    calibrator = pipeline._build_weight_calibrator()
    assert isinstance(calibrator, MicrocalibrateAdapter)


def test_backend_dispatch_fit_transform_end_to_end() -> None:
    """Full path: pipeline config → dispatch → fit_transform → validate."""
    cfg = USMicroplexBuildConfig(
        calibration_backend="microcalibrate",
        calibration_max_iter=200,
    )
    pipeline = USMicroplexPipeline(cfg)
    calibrator = pipeline._build_weight_calibrator()

    data = _toy_households(n=200, seed=1)
    # Constraint: weighted count of households with income > 80k should be 1.4x current.
    mask = (data["income"] > 80_000).to_numpy(dtype=float)
    target = 1.4 * float(mask.sum())
    constraint = LinearConstraint(
        name="above_80k", coefficients=mask, target=target
    )

    result = calibrator.fit_transform(
        data,
        marginal_targets={},
        weight_col="household_weight",
        linear_constraints=(constraint,),
    )

    assert len(result) == len(data)
    assert "household_weight" in result.columns
    assert (result["household_weight"] >= 0).all()

    validation = calibrator.validate(result)
    assert set(validation) == {"converged", "max_error", "sparsity", "linear_errors"}
    assert "above_80k" in validation["linear_errors"]


def test_invalid_backend_still_raises() -> None:
    """Regression test: unknown backend strings surface a clear error."""
    # The Literal type is only checked by static tools; runtime dispatch
    # raises a ValueError, which we want to preserve.
    # Construct the dataclass bypassing the Literal constraint.
    bad_cfg = USMicroplexBuildConfig()
    object.__setattr__(bad_cfg, "calibration_backend", "no_such_backend")
    pipeline = USMicroplexPipeline(bad_cfg)
    with pytest.raises(ValueError, match="Unsupported calibration backend"):
        pipeline._build_weight_calibrator()


def test_pe_l0_deferred_stage_disables_sparsity_penalty() -> None:
    """Stages ≥2 must refine weights without re-sparsifying.

    v10 ran three L0 stages with `lambda_l0=1e-4` each, warm-starting
    stages 2/3 from stage 1's already-sparse weights. Loss compounded
    pruning down to 1,511 active households — unusable. Stages 2+ now
    drop the sparsity penalty so they only reduce residual error.
    """
    cfg = USMicroplexBuildConfig(calibration_backend="pe_l0")
    pipeline = USMicroplexPipeline(cfg)

    stage1 = pipeline._build_weight_calibrator(stage_index=1)
    stage2 = pipeline._build_weight_calibrator(stage_index=2)
    stage3 = pipeline._build_weight_calibrator(stage_index=3)

    assert stage1.lambda_l0 == pytest.approx(1e-4)
    assert stage2.lambda_l0 == 0.0
    assert stage3.lambda_l0 == 0.0


def test_hardconcrete_deferred_stage_disables_sparsity_penalty() -> None:
    cfg = USMicroplexBuildConfig(calibration_backend="hardconcrete")
    pipeline = USMicroplexPipeline(cfg)
    stage1 = pipeline._build_weight_calibrator(stage_index=1)
    stage2 = pipeline._build_weight_calibrator(stage_index=2)
    assert stage1.lambda_l0 == pytest.approx(1e-4)
    assert isinstance(stage2, MicrocalibrateAdapter)
    assert stage2.config.regularize_with_l0 is False
