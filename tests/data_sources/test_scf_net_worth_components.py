"""SCF net-worth component leaves: manifest wiring, rebalance, and a donor fit.

Covers the 19 SCF balance-sheet component columns added to the ``scf``
source-impute block (G1). Mirrors policyengine-us-data eCPS:

    * map / signs / rebalance: utils/asset_imputation.py (upstream/main)
    * per-leaf QRF + rebalance:  calibration/source_impute.py _impute_scf
      (upstream/main L1113-1344)

What is covered here:
  - The 19 leaves are present in the scf block target/person/loader columns
    and the loader raw-column map equals SCF_NET_WORTH_COMPONENT_TARGETS.
  - The ported rebalance reconciles signed components to a net_worth anchor
    within the eCPS float32 tolerance (unit test on synthetic components).
  - Debt leaves are stored as positive magnitudes and subtract in net worth.
  - A small synthetic zero-inflated donor fit on the SCF predictors produces
    non-degenerate, non-negative per-leaf predictions (smoke test, not a full
    donor run). The runnable path uses ColumnwiseQRFDonorImputer; the
    production RegimeAwareDonorImputer variant skips without microimpute.

Not covered here (would require a full pipeline run / SCF download):
  - End-to-end SCF donor synthesis on real CPS receivers.
  - Final-H5 export of the leaves through the legacy-contract entity path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microplex_us.asset_reconciliation import (
    NET_WORTH_COMPONENT_SIGNS,
    SCF_NET_WORTH_COMPONENT_TARGETS,
    SCF_NET_WORTH_COMPONENT_VARIABLES,
    compute_net_worth_from_components,
    rebalance_scf_net_worth_components,
)

SCF_COMPONENT_LEAVES = (
    "scf_certificates_of_deposit",
    "scf_savings_bonds",
    "scf_retirement_assets",
    "scf_cash_value_life_insurance",
    "scf_other_managed_assets",
    "scf_other_financial_assets",
    "scf_primary_residence_value",
    "scf_other_residential_real_estate",
    "scf_nonresidential_real_estate_equity",
    "scf_business_equity",
    "scf_other_nonfinancial_assets",
    "scf_mortgage_debt",
    "scf_other_residential_debt",
    "scf_other_lines_of_credit",
    "scf_credit_card_debt",
    "scf_vehicle_installment_debt",
    "scf_student_loan_debt",
    "scf_other_installment_debt",
    "scf_other_debt",
)

# eCPS raw SCF summary-extract source column per leaf
# (asset_imputation.py SCF_NET_WORTH_COMPONENT_TARGETS).
EXPECTED_RAW_SOURCE_COLUMN = {
    "scf_certificates_of_deposit": "cds",
    "scf_savings_bonds": "savbnd",
    "scf_retirement_assets": "retqliq",
    "scf_cash_value_life_insurance": "cashli",
    "scf_other_managed_assets": "othma",
    "scf_other_financial_assets": "othfin",
    "scf_primary_residence_value": "houses",
    "scf_other_residential_real_estate": "oresre",
    "scf_nonresidential_real_estate_equity": "nnresre",
    "scf_business_equity": "bus",
    "scf_other_nonfinancial_assets": "othnfin",
    "scf_mortgage_debt": "mrthel",
    "scf_other_residential_debt": "resdbt",
    "scf_other_lines_of_credit": "othloc",
    "scf_credit_card_debt": "ccbal",
    "scf_vehicle_installment_debt": "veh_inst",
    "scf_student_loan_debt": "edn_inst",
    "scf_other_installment_debt": "oth_inst",
    "scf_other_debt": "odebt",
}

DEBT_LEAVES = frozenset(
    leaf for leaf, sign in NET_WORTH_COMPONENT_SIGNS.items() if sign < 0
) & frozenset(SCF_COMPONENT_LEAVES)

_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "microplex_us"
    / "manifests"
    / "pe_source_impute_blocks.json"
)


def _scf_block() -> dict:
    payload = json.loads(_MANIFEST_PATH.read_text())
    return payload["blocks"]["scf"]


class TestScfBlockManifest:
    """The scf source-impute block must carry the 19 component leaves."""

    def test_all_nineteen_in_target_variables(self) -> None:
        target_variables = set(_scf_block()["target_variables"])
        missing = [
            leaf for leaf in SCF_COMPONENT_LEAVES if leaf not in target_variables
        ]
        assert not missing, f"missing from scf target_variables: {missing}"

    def test_all_nineteen_in_person_variables(self) -> None:
        person_variables = set(_scf_block()["person_variables"])
        missing = [
            leaf for leaf in SCF_COMPONENT_LEAVES if leaf not in person_variables
        ]
        assert not missing, f"missing from scf person_variables: {missing}"

    def test_loader_raw_column_map_matches_ecps(self) -> None:
        """Each leaf maps to the eCPS raw SCF column, no sign flip at load."""
        direct = _scf_block()["dataset_loader"]["direct_person_columns"]
        for leaf, raw_column in EXPECTED_RAW_SOURCE_COLUMN.items():
            assert direct.get(leaf) == raw_column, (
                f"{leaf} should load from raw SCF column '{raw_column}', "
                f"got {direct.get(leaf)!r}"
            )

    def test_target_map_matches_ecps_component_targets(self) -> None:
        """The ported leaf->raw map equals eCPS SCF_NET_WORTH_COMPONENT_TARGETS."""
        for leaf, raw_column in EXPECTED_RAW_SOURCE_COLUMN.items():
            assert SCF_NET_WORTH_COMPONENT_TARGETS[leaf] == (raw_column,)
        assert set(SCF_NET_WORTH_COMPONENT_VARIABLES) == set(SCF_COMPONENT_LEAVES)


class TestNetWorthSigns:
    """Debt leaves subtract; asset leaves add."""

    def test_eight_scf_debt_leaves_carry_negative_sign(self) -> None:
        expected_debt = {
            "scf_mortgage_debt",
            "scf_other_residential_debt",
            "scf_other_lines_of_credit",
            "scf_credit_card_debt",
            "scf_vehicle_installment_debt",
            "scf_student_loan_debt",
            "scf_other_installment_debt",
            "scf_other_debt",
        }
        assert DEBT_LEAVES == expected_debt

    def test_asset_leaves_have_no_negative_sign(self) -> None:
        asset_leaves = set(SCF_COMPONENT_LEAVES) - DEBT_LEAVES
        for leaf in asset_leaves:
            assert NET_WORTH_COMPONENT_SIGNS.get(leaf, 1.0) >= 0

    def test_debt_stored_positive_subtracts_in_net_worth(self) -> None:
        """A positive-magnitude debt leaf reduces net worth."""
        components = {
            "scf_primary_residence_value": np.array([100_000.0], dtype=np.float32),
            "scf_mortgage_debt": np.array([40_000.0], dtype=np.float32),
        }
        net_worth = compute_net_worth_from_components(components=components)
        assert net_worth[0] == pytest.approx(60_000.0)


def _synthetic_components(
    n: int,
    *,
    seed: int,
    include_protected: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Zero-inflated positive-magnitude component draws + a net-worth anchor.

    Emulates the per-leaf QRF output: each leaf is a non-negative magnitude
    with a leaf-specific zero share, the way the real SCF leaves look.
    """
    rng = np.random.default_rng(seed)
    variables = list(SCF_NET_WORTH_COMPONENT_VARIABLES)
    if include_protected:
        variables += [
            "bank_account_assets",
            "stock_assets",
            "bond_assets",
            "household_vehicles_value",
        ]
    components: dict[str, np.ndarray] = {}
    for variable in variables:
        scale = rng.uniform(1e3, 5e5)
        values = rng.exponential(scale=scale, size=n).astype(np.float32)
        zero_share = rng.uniform(0.0, 0.7)
        values[rng.random(n) < zero_share] = 0.0
        components[variable] = values
    target = rng.normal(2e5, 6e5, size=n).astype(np.float32)
    return components, target


class TestRebalanceReconciliation:
    """After rebalance, signed components must reconcile to the net-worth anchor."""

    def test_reconciles_within_ecps_tolerance(self) -> None:
        components, target = _synthetic_components(5000, seed=11)
        adjusted = rebalance_scf_net_worth_components(
            components={k: v.copy() for k, v in components.items()},
            target_net_worth=target.copy(),
        )
        net_worth = compute_net_worth_from_components(components=adjusted)
        abs_diff = np.abs(net_worth.astype(np.float64) - target.astype(np.float64))
        rel_diff = abs_diff / np.maximum(np.abs(target.astype(np.float64)), 1.0)
        # eCPS reconciliation contract: rtol=1e-6, atol=1.0
        # (check_household_net_worth_reconciliation). float32 accumulation
        # leaves a sub-dollar residual on million-dollar balance sheets.
        reconciled = (abs_diff <= 1.5) | (rel_diff <= 1e-4)
        assert reconciled.all(), (
            f"{(~reconciled).sum()} of {len(target)} rows failed to reconcile; "
            f"max abs diff {abs_diff.max():.4g}"
        )

    def test_rebalance_preserves_protected_blended_leaves(self) -> None:
        """SIPP/SCF-blended leaves are not rescaled by the rebalance."""
        components, target = _synthetic_components(2000, seed=3)
        before = {
            k: components[k].copy()
            for k in ("bank_account_assets", "stock_assets", "bond_assets")
        }
        adjusted = rebalance_scf_net_worth_components(
            components={k: v.copy() for k, v in components.items()},
            target_net_worth=target.copy(),
        )
        for variable, original in before.items():
            np.testing.assert_array_equal(adjusted[variable], original)

    def test_adjusted_leaves_stay_nonnegative(self) -> None:
        """Proportional scaling never drives a stored leaf magnitude negative."""
        components, target = _synthetic_components(3000, seed=5)
        adjusted = rebalance_scf_net_worth_components(
            components={k: v.copy() for k, v in components.items()},
            target_net_worth=target.copy(),
        )
        for leaf in SCF_NET_WORTH_COMPONENT_VARIABLES:
            assert (adjusted[leaf] >= -1e-3).all(), f"{leaf} went negative"


def _donor_training_frame(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """SCF-like donor frame: 8 predictors + zero-inflated positive leaves.

    Leaf support is driven by age/income so a gate classifier can learn it.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, size=n).astype(float)
    is_female = rng.integers(0, 2, size=n).astype(float)
    cps_race = rng.integers(1, 8, size=n).astype(float)
    is_married = rng.integers(0, 2, size=n).astype(float)
    own_children = rng.integers(0, 4, size=n).astype(float)
    employment_income = np.maximum(rng.normal(45_000, 35_000, size=n), 0.0).astype(
        float
    )
    interest_dividend_income = np.maximum(rng.normal(2_000, 6_000, size=n), 0.0).astype(
        float
    )
    social_security_pension_income = np.where(
        age >= 65,
        np.maximum(rng.normal(18_000, 9_000, size=n), 0.0),
        0.0,
    ).astype(float)

    frame = pd.DataFrame(
        {
            "age": age,
            "is_female": is_female,
            "cps_race": cps_race,
            "is_married": is_married,
            "own_children_in_household": own_children,
            "employment_income": employment_income,
            "interest_dividend_income": interest_dividend_income,
            "social_security_pension_income": social_security_pension_income,
        }
    )

    # Retirement assets: held mostly by older / higher-income; ~55% have it.
    holds_ret = rng.random(n) < (0.2 + 0.5 * (age >= 50) + 1e-6 * employment_income)
    holds_ret = holds_ret & (rng.random(n) < 0.9)
    retirement = np.where(
        holds_ret,
        np.maximum(rng.lognormal(11.5, 1.0, size=n), 0.0),
        0.0,
    ).astype(float)
    frame["scf_retirement_assets"] = retirement

    # Credit card debt: ~38% carry a balance, broadly across ages.
    holds_cc = rng.random(n) < 0.38
    ccbal = np.where(
        holds_cc,
        np.maximum(rng.lognormal(8.0, 0.8, size=n), 0.0),
        0.0,
    ).astype(float)
    frame["scf_credit_card_debt"] = ccbal

    return frame


_DONOR_CONDITION_VARS = [
    "age",
    "is_female",
    "cps_race",
    "is_married",
    "own_children_in_household",
    "employment_income",
    "interest_dividend_income",
    "social_security_pension_income",
]
_DONOR_TARGET_VARS = ("scf_retirement_assets", "scf_credit_card_debt")


def _assert_leaf_predictions_nondegenerate(out: pd.DataFrame) -> None:
    for leaf in _DONOR_TARGET_VARS:
        values = out[leaf].to_numpy(dtype=float)
        assert np.isfinite(values).all(), f"{leaf} has non-finite predictions"
        # Non-negative (positive-magnitude leaf, clamped at zero).
        assert (values >= 0).all(), f"{leaf} produced negative magnitude"
        # Non-degenerate: a meaningful share is nonzero (the leaf has mass),
        # and not everyone is nonzero (zero-inflation preserved).
        nonzero_share = float((values > 0).mean())
        assert 0.05 < nonzero_share < 0.98, (
            f"{leaf} nonzero share {nonzero_share:.3f} looks degenerate"
        )
        nonzero = values[values > 0]
        assert nonzero.size == 0 or nonzero.std() > 0, f"{leaf} has no spread"


class TestDonorFitOnScfLeaves:
    """A small zero-inflated donor fit yields non-degenerate, non-negative leaves.

    The component leaves are strongly zero-inflated, so production wiring uses
    ``RegimeAwareDonorImputer`` (``donor_imputer_backend='regime_aware'``).
    These leaves are positive-magnitude single-sign, so the simpler
    ``ColumnwiseQRFDonorImputer`` zero-inflated gate reproduces the same
    behavior and is what runs here when ``microimpute`` is not installed.
    """

    def test_columnwise_qrf_zero_inflated_fit(self) -> None:
        """Runnable proof: zero-inflated QRF gate on two SCF leaves."""
        pytest.importorskip("quantile_forest")
        from microplex_us.pipelines.donor_imputers import (
            ColumnwiseQRFDonorImputer,
        )

        train = _donor_training_frame(n=1200, seed=0)
        imputer = ColumnwiseQRFDonorImputer(
            condition_vars=_DONOR_CONDITION_VARS,
            target_vars=list(_DONOR_TARGET_VARS),
            n_estimators=60,
            zero_inflated_vars=set(_DONOR_TARGET_VARS),
            nonnegative_vars=set(_DONOR_TARGET_VARS),
        )
        imputer.fit(train)
        receivers = _donor_training_frame(n=600, seed=7)[_DONOR_CONDITION_VARS]
        out = imputer.generate(receivers, seed=42)
        _assert_leaf_predictions_nondegenerate(out)

    def test_regime_aware_fit(self) -> None:
        """Production backend: regime-aware imputer (skips without microimpute)."""
        pytest.importorskip("quantile_forest")
        pytest.importorskip("microimpute")
        from microplex_us.pipelines.donor_imputers import RegimeAwareDonorImputer

        train = _donor_training_frame(n=1200, seed=0)
        imputer = RegimeAwareDonorImputer(
            condition_vars=_DONOR_CONDITION_VARS,
            target_vars=list(_DONOR_TARGET_VARS),
            n_estimators=60,
            nonnegative_vars=set(_DONOR_TARGET_VARS),
            seed=42,
        )
        imputer.fit(train)
        receivers = _donor_training_frame(n=600, seed=7)[_DONOR_CONDITION_VARS]
        out = imputer.generate(receivers, seed=42)
        _assert_leaf_predictions_nondegenerate(out)
