"""Pin the zi_qrf donor-imputer backend behavior before v8 relies on it.

v7 (2026-04-18) used `donor_imputer_backend="qrf"` which bypasses the
zero-classifier gate (see `USMicroplexPipeline._build_donor_imputer`:
`zero_inflated_vars` is populated only when `backend == "zi_qrf"`). With
an empty whitelist, every QRF predict runs over all 3.37M rows even on
columns that are 99%+ zero, which is the main reason donor integration
took hours per source on v7.

v8 flips `--donor-imputer-backend zi_qrf`. These tests pin the three
guarantees v8 relies on:

1. The factory (`_build_donor_imputer`) populates `zero_inflated_vars`
   from the `VariableSupportFamily.SUPPORT_SENSITIVE` variables
   when `backend == "zi_qrf"`, and leaves it empty otherwise.
2. `ColumnwiseQRFDonorImputer.fit` trains a `RandomForestClassifier`
   zero-gate on each whitelisted column whose observed zero fraction
   crosses the threshold, and does not train one on dense columns.
3. `ColumnwiseQRFDonorImputer.generate` skips QRF `.predict` on rows
   the zero-gate sent to zero — i.e. the QRF is invoked on a strict
   subset, which is the wall-clock win.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("quantile_forest")

from microplex_us.pipelines.us import (
    ColumnwiseQRFDonorImputer,
    USMicroplexBuildConfig,
    USMicroplexPipeline,
)


def _tiny_problem(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Two-column donor frame: one heavy-zero target, one dense target."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, size=n).astype(float)
    is_female = rng.integers(0, 2, size=n).astype(float)
    # 97 % zero — only a handful of positive values, like SSI or TANF.
    heavy_zero = np.where(rng.random(n) > 0.97, rng.exponential(500, n), 0.0)
    # Dense — every row has a positive draw, like age or weight.
    dense = rng.normal(40_000, 10_000, size=n).clip(0, None)
    return pd.DataFrame(
        {
            "age": age,
            "is_female": is_female,
            "tanf_reported": heavy_zero,
            "employment_income": dense,
        }
    )


class TestImputerFit:
    """Whitelisted + heavy-zero → RF classifier gate; otherwise no gate."""

    def test_zi_whitelist_produces_zero_classifier(self) -> None:
        data = _tiny_problem()
        imputer = ColumnwiseQRFDonorImputer(
            condition_vars=["age", "is_female"],
            target_vars=["tanf_reported", "employment_income"],
            n_estimators=25,
            zero_inflated_vars={"tanf_reported"},
            zero_threshold=0.05,
        )
        imputer.fit(data)
        assert "tanf_reported" in imputer._zero_models, (
            "Heavy-zero column in whitelist must get a zero-gate classifier; "
            "this is the optimization v8 depends on."
        )
        assert "employment_income" not in imputer._zero_models, (
            "Dense column must not get a zero-gate classifier."
        )

    def test_empty_whitelist_means_no_gates(self) -> None:
        """v7 configuration: backend='qrf' → no gates ever fitted."""
        data = _tiny_problem()
        imputer = ColumnwiseQRFDonorImputer(
            condition_vars=["age", "is_female"],
            target_vars=["tanf_reported", "employment_income"],
            n_estimators=25,
            zero_inflated_vars=set(),
            zero_threshold=0.05,
        )
        imputer.fit(data)
        assert imputer._zero_models == {}


class TestImputerGenerateSkipsPredict:
    """With a zero-gate, the QRF's .predict runs on a strict subset."""

    def test_generate_calls_qrf_only_on_predicted_positive_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _tiny_problem(n=800, seed=1)
        imputer = ColumnwiseQRFDonorImputer(
            condition_vars=["age", "is_female"],
            target_vars=["tanf_reported"],
            n_estimators=25,
            zero_inflated_vars={"tanf_reported"},
            zero_threshold=0.05,
        )
        imputer.fit(data)

        qrf_model = imputer._models["tanf_reported"]
        call_input_sizes: list[int] = []
        original_predict = qrf_model.predict

        def spy_predict(x_values: np.ndarray, **kwargs):
            call_input_sizes.append(len(x_values))
            return original_predict(x_values, **kwargs)

        monkeypatch.setattr(qrf_model, "predict", spy_predict)

        # Generate on 10k conditioning rows (much larger than training).
        rng = np.random.default_rng(42)
        n_generate = 10_000
        conditions = pd.DataFrame(
            {
                "age": rng.integers(18, 80, size=n_generate).astype(float),
                "is_female": rng.integers(0, 2, size=n_generate).astype(float),
            }
        )
        synthetic = imputer.generate(conditions, seed=42)

        assert len(call_input_sizes) == 1, call_input_sizes
        predict_rows = call_input_sizes[0]
        # Heavy-zero base rate is ~3 %; ZI-predicted-positive fraction
        # should be well below 50 % on unseen data, and definitely
        # below n_generate.
        assert predict_rows < n_generate, (
            f"QRF predict was called on all {n_generate} rows — the "
            f"zero-gate isn't skipping any. call_input_sizes={call_input_sizes}"
        )
        assert predict_rows < n_generate * 0.5, (
            f"QRF predict got {predict_rows}/{n_generate} rows; the gate "
            "is barely cutting the wall, not matching the 3 % training base rate."
        )
        # Non-predicted rows must be exactly zero (not NaN, not drawn).
        zero_mass = float((synthetic["tanf_reported"] == 0).mean())
        assert zero_mass > 0.5, (
            f"Synthetic zero mass = {zero_mass:.3f}; gate should leave "
            "more than half of rows at exactly 0."
        )


class TestBuildDonorImputerFactory:
    """The pipeline factory wires zero_inflated_vars only when backend='zi_qrf'."""

    def _factory(self, backend: str) -> ColumnwiseQRFDonorImputer:
        config = USMicroplexBuildConfig(
            donor_imputer_backend=backend,
            donor_imputer_qrf_n_estimators=25,
        )
        pipeline = USMicroplexPipeline(config=config)
        # Variables chosen to span support families:
        #   qualified_dividend_income, taxable_interest_income → SUPPORT_SENSITIVE
        #   age → BOUNDED_INTEGER
        # These are all real PolicyEngine-US variable names with explicit
        # semantic specs in microplex_us.variables.
        target_vars = (
            "qualified_dividend_income",
            "taxable_interest_income",
            "age",
        )
        return pipeline._build_donor_imputer(
            condition_vars=["is_female", "cps_race"],
            target_vars=target_vars,
        )

    def test_zi_qrf_backend_populates_whitelist(self) -> None:
        imputer = self._factory("zi_qrf")
        assert isinstance(imputer, ColumnwiseQRFDonorImputer)
        assert "qualified_dividend_income" in imputer.zero_inflated_vars
        assert "taxable_interest_income" in imputer.zero_inflated_vars
        assert "age" not in imputer.zero_inflated_vars

    def test_qrf_backend_leaves_whitelist_empty(self) -> None:
        """v7 semantics: pre-v8 default leaves optimization inactive."""
        imputer = self._factory("qrf")
        assert isinstance(imputer, ColumnwiseQRFDonorImputer)
        assert imputer.zero_inflated_vars == set()
