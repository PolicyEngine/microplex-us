"""Regime-aware donor imputer integration for v9.

v7 had a `y > 0` bug that dropped negative training rows — fixed
minimally in v8 (commit 8c88277) by relabelling the gate to `y != 0`.
v8's fix makes the QRF see both signs, but it fits ONE QRF over mixed
positive and negative training rows, which allows predictions to land
in the interior band (``max(train_negatives)``, ``min(train_positives)``)
— a region no real record occupies.

v9 upgrades to `microimpute.models.ZeroInflatedImputer`, which at fit
time auto-detects the three-sign regime per target and routes
predictions through separate positive and negative QRFs. The
interior-band gap becomes a structural guarantee, not a statistical
averaging hope.

Downstream integration lives under a new `--donor-imputer-backend
regime_aware` option; the existing `qrf` and `zi_qrf` backends stay
unchanged for regression comparison.

Tests pin:

1. The new backend value resolves through the factory to a donor
   imputer that uses ZeroInflatedImputer internally.
2. On a three-sign training fixture, predictions preserve negatives
   (as v8's `y != 0` fix already does).
3. On the same fixture, predictions NEVER land in the interior band
   between the positive and negative training regimes — the upgrade
   v9 provides over v8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("quantile_forest")
pytest.importorskip("microimpute")
pytest.importorskip("microimpute.models.zero_inflated")


def _three_sign_frame_with_gap(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Fixture with a hard gap between positive and negative training values.

    Positives live in [100, ∞), negatives in (-∞, -100], zeros exactly
    at 0. Any prediction that lands in (-100, 100) excluding zero is
    an "interior-band violation" — the test metric for the tripartite
    advantage.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, size=n).astype(float)
    is_female = rng.integers(0, 2, size=n).astype(float)

    # Three-way regime assignment driven by (age, is_female).
    logit_pos = -0.3 + 0.04 * (age - 50)
    logit_neg = 0.3 - 0.04 * (age - 50)
    logit_zero = 0.2 * (1 - is_female)
    logits = np.stack([logit_neg, logit_zero, logit_pos], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    u = rng.random(n)
    cum = np.cumsum(probs, axis=1)
    regime_idx = (cum >= u[:, None]).argmax(axis=1)

    y = np.zeros(n)
    pos_mask = regime_idx == 2
    neg_mask = regime_idx == 0
    y[pos_mask] = 100.0 + rng.exponential(250, size=pos_mask.sum())
    y[neg_mask] = -(100.0 + rng.exponential(250, size=neg_mask.sum()))

    return pd.DataFrame(
        {
            "age": age,
            "is_female": is_female,
            "short_term_capital_gains": y,
        }
    )


def _count_interior_violations(
    predictions: np.ndarray, band: float = 100.0, atol: float = 1e-6
) -> int:
    """Count predictions in the (-band, band) interior, excluding exact zero."""
    interior = (np.abs(predictions) < band) & (np.abs(predictions) > atol)
    return int(interior.sum())


class TestRegimeAwareDonorImputerClassExists:
    """The new donor imputer must be importable from microplex_us.pipelines.us."""

    def test_importable_from_us_module(self) -> None:
        from microplex_us.pipelines.us import RegimeAwareDonorImputer

        assert RegimeAwareDonorImputer is not None


class TestRegimeAwareBackendFactory:
    """`_build_donor_imputer(backend='regime_aware')` returns the new class."""

    def test_factory_dispatches_to_regime_aware(self) -> None:
        from microplex_us.pipelines.us import (
            RegimeAwareDonorImputer,
            USMicroplexBuildConfig,
            USMicroplexPipeline,
        )

        config = USMicroplexBuildConfig(
            donor_imputer_backend="regime_aware",
            donor_imputer_qrf_n_estimators=25,
        )
        pipeline = USMicroplexPipeline(config=config)
        imputer = pipeline._build_donor_imputer(
            condition_vars=["is_female", "cps_race"],
            target_vars=("qualified_dividend_income", "age"),
        )
        assert isinstance(imputer, RegimeAwareDonorImputer)


class TestRegimeAwareFitGenerate:
    """Fit/generate contract and tripartite-specific guarantees."""

    def test_multi_target_fit_uses_one_chained_zero_inflated_imputer(self) -> None:
        from microplex_us.pipelines.us import RegimeAwareDonorImputer

        rng = np.random.default_rng(20260606)
        n = 300
        age = rng.integers(18, 80, size=n).astype(float)
        first = rng.normal(loc=age * 400.0, scale=2_000.0, size=n)
        second = 0.75 * first + rng.normal(scale=250.0, size=n)
        train = pd.DataFrame(
            {
                "age": age,
                "first_income_leaf": first,
                "second_income_leaf": second,
            }
        )

        imputer = RegimeAwareDonorImputer(
            condition_vars=["age"],
            target_vars=["first_income_leaf", "second_income_leaf"],
            n_estimators=25,
        )
        imputer.fit(train)

        first_fitted = imputer._fitted["first_income_leaf"]
        second_fitted = imputer._fitted["second_income_leaf"]
        assert first_fitted is second_fitted

        second_bundle = second_fitted._per_variable["second_income_leaf"]
        assert second_bundle["predictors"] == ["age", "first_income_leaf"]

    def test_target_predictor_overlap_is_owned_by_sequential_chain(self) -> None:
        from microplex_us.pipelines.us import RegimeAwareDonorImputer

        rng = np.random.default_rng(2026060601)
        n = 300
        age = rng.integers(18, 80, size=n).astype(float)
        first = rng.normal(loc=age * 300.0, scale=1_000.0, size=n)
        second = 0.5 * first + rng.normal(scale=250.0, size=n)
        train = pd.DataFrame(
            {
                "age": age,
                "first_income_leaf": first,
                "second_income_leaf": second,
            }
        )

        imputer = RegimeAwareDonorImputer(
            condition_vars=["age", "first_income_leaf"],
            target_vars=["first_income_leaf", "second_income_leaf"],
            n_estimators=25,
        )
        imputer.fit(train)

        fitted = imputer._fitted["first_income_leaf"]
        first_bundle = fitted._per_variable["first_income_leaf"]
        second_bundle = fitted._per_variable["second_income_leaf"]
        assert first_bundle["predictors"] == ["age"]
        assert second_bundle["predictors"] == ["age", "first_income_leaf"]

        conditions = pd.DataFrame({"age": [25.0, 45.0, 65.0]})
        synthetic = imputer.generate(conditions, seed=20260606)
        assert list(synthetic.columns) == [
            "age",
            "first_income_leaf",
            "second_income_leaf",
        ]
        assert (
            synthetic[["first_income_leaf", "second_income_leaf"]].notna().all().all()
        )

    def test_duplicate_input_columns_are_collapsed_before_microimpute(self) -> None:
        from microplex_us.pipelines.us import RegimeAwareDonorImputer

        rng = np.random.default_rng(2026060602)
        n = 300
        age = rng.integers(18, 80, size=n).astype(float)
        first = rng.normal(loc=age * 300.0, scale=1_000.0, size=n)
        second = 0.5 * first + rng.normal(scale=250.0, size=n)
        train = pd.DataFrame(
            np.column_stack([age, first, first, second]),
            columns=[
                "age",
                "first_income_leaf",
                "first_income_leaf",
                "second_income_leaf",
            ],
        )
        assert not train.columns.is_unique

        imputer = RegimeAwareDonorImputer(
            condition_vars=["age", "first_income_leaf"],
            target_vars=[
                "first_income_leaf",
                "first_income_leaf",
                "second_income_leaf",
            ],
            n_estimators=25,
        )
        imputer.fit(train)

        assert imputer._fitted_columns == (
            "first_income_leaf",
            "second_income_leaf",
        )
        fitted = imputer._fitted["first_income_leaf"]
        first_bundle = fitted._per_variable["first_income_leaf"]
        second_bundle = fitted._per_variable["second_income_leaf"]
        assert first_bundle["predictors"] == ["age"]
        assert second_bundle["predictors"] == ["age", "first_income_leaf"]

        conditions = pd.DataFrame(
            np.column_stack([[25.0, 45.0, 65.0], [26.0, 46.0, 66.0]]),
            columns=["age", "age"],
        )
        synthetic = imputer.generate(conditions, seed=20260606)
        assert list(synthetic.columns) == [
            "age",
            "first_income_leaf",
            "second_income_leaf",
        ]
        assert synthetic.columns.is_unique
        assert (
            synthetic[["first_income_leaf", "second_income_leaf"]].notna().all().all()
        )

    def _fit_generate(
        self, n_train: int = 1500, n_gen: int = 2000, seed: int = 0
    ) -> np.ndarray:
        from microplex_us.pipelines.us import RegimeAwareDonorImputer

        train = _three_sign_frame_with_gap(n=n_train, seed=seed)
        # Precondition: fixture genuinely three-sign.
        y = train["short_term_capital_gains"].to_numpy()
        assert (y > 100).sum() > 100
        assert (y < -100).sum() > 100
        assert (y == 0).sum() > 100

        imputer = RegimeAwareDonorImputer(
            condition_vars=["age", "is_female"],
            target_vars=["short_term_capital_gains"],
            n_estimators=25,
        )
        imputer.fit(train)

        rng = np.random.default_rng(42)
        conditions = pd.DataFrame(
            {
                "age": rng.integers(18, 80, size=n_gen).astype(float),
                "is_female": rng.integers(0, 2, size=n_gen).astype(float),
            }
        )
        synthetic = imputer.generate(conditions, seed=42)
        return synthetic["short_term_capital_gains"].to_numpy()

    def test_generates_negative_predictions(self) -> None:
        """Drop-negatives bug must not recur under regime-aware path."""
        synth_y = self._fit_generate()
        n_neg = int((synth_y < 0).sum())
        assert n_neg > 0, (
            "Regime-aware donor imputer produced no negatives on a "
            "three-sign training fixture — regression."
        )
        assert n_neg / len(synth_y) > 0.05

    def test_generates_positive_predictions(self) -> None:
        synth_y = self._fit_generate()
        n_pos = int((synth_y > 0).sum())
        assert n_pos / len(synth_y) > 0.05

    def test_generates_zero_predictions(self) -> None:
        synth_y = self._fit_generate()
        n_zero = int((np.abs(synth_y) < 1e-6).sum())
        assert n_zero > 0, "Gate must emit some exact zeros."

    def test_no_interior_band_violations(self) -> None:
        """Core v9 advantage over v8.

        v8's `y != 0` fix keeps negatives but fits ONE QRF over mixed
        pos+neg training rows, so predictions can interpolate into the
        (-100, 100) interior band. v9's regime-aware path fits
        separate positive and negative QRFs and routes through a
        three-way gate, so the interior is empty by construction.
        """
        synth_y = self._fit_generate()
        violations = _count_interior_violations(synth_y, band=100.0)
        assert violations == 0, (
            f"Regime-aware imputer produced {violations} predictions in "
            f"the (-100, 100) interior band, which should be empty by "
            f"construction. Sample offenders: "
            f"{sorted(synth_y[(np.abs(synth_y) < 100) & (np.abs(synth_y) > 1e-6)][:10])}"
        )

    def test_same_seed_repeats_identically(self) -> None:
        from microplex_us.pipelines.us import RegimeAwareDonorImputer

        train = _three_sign_frame_with_gap(n=1200, seed=3)
        conditions = train[["age", "is_female"]].head(300).reset_index(drop=True)
        imputer = RegimeAwareDonorImputer(
            condition_vars=["age", "is_female"],
            target_vars=["short_term_capital_gains"],
            n_estimators=25,
        )
        imputer.fit(train)

        first = imputer.generate(conditions, seed=123)[
            "short_term_capital_gains"
        ].to_numpy()
        second = imputer.generate(conditions, seed=123)[
            "short_term_capital_gains"
        ].to_numpy()
        third = imputer.generate(conditions, seed=999)[
            "short_term_capital_gains"
        ].to_numpy()

        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, third)

    def test_same_seed_repeats_identically_for_multiple_targets(self) -> None:
        from microplex_us.pipelines.us import RegimeAwareDonorImputer

        train = _three_sign_frame_with_gap(n=1200, seed=4)
        train["rental_income"] = -0.5 * train["short_term_capital_gains"]
        conditions = train[["age", "is_female"]].head(300).reset_index(drop=True)
        imputer = RegimeAwareDonorImputer(
            condition_vars=["age", "is_female"],
            target_vars=["short_term_capital_gains", "rental_income"],
            n_estimators=25,
        )
        imputer.fit(train)

        first = imputer.generate(conditions, seed=456)
        second = imputer.generate(conditions, seed=456)
        third = imputer.generate(conditions, seed=654)

        for column in ("short_term_capital_gains", "rental_income"):
            np.testing.assert_array_equal(
                first[column].to_numpy(), second[column].to_numpy()
            )
            assert not np.array_equal(
                first[column].to_numpy(), third[column].to_numpy()
            )
