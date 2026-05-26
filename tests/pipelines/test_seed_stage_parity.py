"""Tests for seed/source-impute parity auditing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import pandas as pd
import pytest

from microplex_us.pipelines.seed_stage_parity import (
    SeedStageBooleanLandingFeatureSpec,
    SeedStageCategoricalLandingFeatureSpec,
    SeedStageFocusVariableSpec,
    _normalize_seed_ids_for_policyengine_support,
    _seed_tax_unit_support_payload,
    build_us_seed_stage_parity_audit,
    write_us_seed_stage_parity_audit,
)


def _write_period_dataset(path, data: dict[str, list | tuple], *, period: int = 2024) -> None:
    with h5py.File(path, "w") as handle:
        for variable, values in data.items():
            group = handle.create_group(variable)
            group.create_dataset(str(period), data=values)


def test_build_us_seed_stage_parity_audit_projects_reference_and_profiles_positive_rows(
    tmp_path,
) -> None:
    seed_path = tmp_path / "seed.parquet"
    reference_path = tmp_path / "reference.h5"

    pd.DataFrame(
        {
            "person_id": ["1", "2", "3"],
            "household_id": ["10", "10", "20"],
            "tax_unit_id": ["100", "100", "200"],
            "hh_weight": [10.0, 10.0, 20.0],
            "self_employment_income": [0.0, 50.0, 100.0],
            "employment_income": [10.0, 20.0, 30.0],
            "has_esi": [False, True, True],
            "has_marketplace_health_coverage": [False, True, False],
            "age": [5, 42, 67],
            "state_fips": [1, 1, 2],
            "employment_status": ["not_working", "self_employed", "self_employed"],
        }
    ).to_parquet(seed_path, index=False)

    _write_period_dataset(
        reference_path,
        {
            "household_id": [10, 20],
            "household_weight": [20.0, 40.0],
            "person_id": [101, 102, 103],
            "person_household_id": [10, 10, 20],
            "tax_unit_id": [100, 200],
            "person_tax_unit_id": [100, 100, 200],
            "self_employment_income_before_lsr": [0.0, 50.0, 100.0],
            "employment_income_before_lsr": [10.0, 20.0, 30.0],
            "has_esi": [False, True, True],
            "has_marketplace_health_coverage": [False, True, False],
            "age": [5, 42, 67],
            "state_fips": [1, 2],
        },
    )

    audit = build_us_seed_stage_parity_audit(
        seed_path,
        reference_path,
        focus_variables=(
            SeedStageFocusVariableSpec(
                "self_employment_income",
                "self_employment_income",
                "self_employment_income_before_lsr",
                value_kind="numeric",
            ),
            "missing_metric",
        ),
        boolean_landing_features=(
            SeedStageBooleanLandingFeatureSpec("has_esi", "has_esi"),
            SeedStageBooleanLandingFeatureSpec(
                "has_marketplace_health_coverage",
                "has_marketplace_health_coverage",
            ),
        ),
        categorical_landing_features=(
            SeedStageCategoricalLandingFeatureSpec(
                "age_bin",
                "age",
                "age",
                transform="age_bin",
            ),
            SeedStageCategoricalLandingFeatureSpec(
                "state_fips",
                "state_fips",
                "state_fips",
            ),
        ),
        candidate_only_landing_features=(
            SeedStageCategoricalLandingFeatureSpec(
                "employment_status",
                "employment_status",
            ),
        ),
    )

    assert audit["comparisonStage"] == "seed_source_impute"
    assert audit["weightScale"]["reference_to_seed_weight_scale"] == pytest.approx(2.0)
    assert audit["seedStructure"]["tax_unit_id_count"] == 2
    assert audit["seedStructure"]["weighted_mean_rows_per_household"] == pytest.approx(
        4.0 / 3.0
    )
    assert audit["referenceStructure"]["weighted_mean_rows_per_household"] == pytest.approx(
        4.0 / 3.0
    )

    self_employment = audit["focusVariables"]["self_employment_income"]
    assert self_employment["comparison"]["type"] == "numeric"
    assert self_employment["comparison"]["weighted_sum_ratio"] == pytest.approx(0.5)
    assert self_employment["comparison"]["reference_scaled_weighted_sum_ratio"] == pytest.approx(
        1.0
    )
    assert self_employment["positiveSupport"]["seed_positive_weight_share"] == pytest.approx(
        0.75
    )

    marketplace = self_employment["positiveBooleanProfiles"][
        "has_marketplace_health_coverage"
    ]
    assert marketplace["seed_positive_share"] == pytest.approx(1.0 / 3.0)
    assert marketplace["reference_positive_share"] == pytest.approx(1.0 / 3.0)
    assert marketplace["share_delta"] == pytest.approx(0.0)

    age_bin = self_employment["positiveCategoricalProfiles"]["age_bin"]
    assert age_bin["seed"]["top_values"][0]["value"] == "65-69"
    assert age_bin["seed"]["top_values"][0]["weighted_share"] == pytest.approx(2.0 / 3.0)

    state_fips = self_employment["positiveCategoricalProfiles"]["state_fips"]
    assert state_fips["reference_present"] is True
    assert state_fips["reference"]["top_values"][0]["value"] == "2"

    employment_status = self_employment["positiveCandidateOnlyProfiles"][
        "employment_status"
    ]
    assert employment_status["seed"]["top_values"][0]["value"] == "self_employed"
    assert employment_status["seed"]["top_values"][0]["weighted_share"] == pytest.approx(1.0)

    missing = audit["focusVariables"]["missing_metric"]
    assert missing["seed_present"] is False
    assert missing["reference_present"] is False


def test_write_us_seed_stage_parity_audit_persists_json(tmp_path) -> None:
    seed_path = tmp_path / "seed.parquet"
    reference_path = tmp_path / "reference.h5"
    output_path = tmp_path / "audit.json"

    pd.DataFrame(
        {
            "person_id": ["1"],
            "household_id": ["10"],
            "hh_weight": [10.0],
            "health_savings_account_ald": [25.0],
        }
    ).to_parquet(seed_path, index=False)
    _write_period_dataset(
        reference_path,
        {
            "household_id": [10],
            "household_weight": [10.0],
            "person_id": [101],
            "person_household_id": [10],
            "tax_unit_id": [100],
            "person_tax_unit_id": [100],
            "health_savings_account_ald": [25.0],
        },
    )

    written = write_us_seed_stage_parity_audit(
        seed_path,
        reference_path,
        output_path,
        focus_variables=("health_savings_account_ald",),
        boolean_landing_features=(),
        categorical_landing_features=(),
        candidate_only_landing_features=(),
    )

    payload = json.loads(written.read_text())
    assert written == output_path.resolve()
    assert payload["focusVariables"]["health_savings_account_ald"]["comparison"]["type"] == (
        "numeric"
    )


def test_build_us_seed_stage_parity_audit_uses_household_weights_not_row_order(tmp_path) -> None:
    reference_path = tmp_path / "reference.h5"
    seed_a_path = tmp_path / "seed_a.parquet"
    seed_b_path = tmp_path / "seed_b.parquet"

    rows = pd.DataFrame(
        {
            "person_id": ["1", "2", "3"],
            "household_id": ["10", "10", "20"],
            "weight": [100.0, 1.0, 200.0],
            "taxable_interest_income": [1.0, 0.0, 2.0],
        }
    )
    rows.to_parquet(seed_a_path, index=False)
    rows.iloc[[1, 0, 2]].reset_index(drop=True).to_parquet(seed_b_path, index=False)

    _write_period_dataset(
        reference_path,
        {
            "household_id": [10, 20],
            "household_weight": [5.0, 10.0],
            "person_id": [101, 102, 103],
            "person_household_id": [10, 10, 20],
            "taxable_interest_income": [1.0, 0.0, 2.0],
        },
    )

    audit_a = build_us_seed_stage_parity_audit(
        seed_a_path,
        reference_path,
        focus_variables=("taxable_interest_income",),
        boolean_landing_features=(),
        categorical_landing_features=(),
        candidate_only_landing_features=(),
    )
    audit_b = build_us_seed_stage_parity_audit(
        seed_b_path,
        reference_path,
        focus_variables=("taxable_interest_income",),
        boolean_landing_features=(),
        categorical_landing_features=(),
        candidate_only_landing_features=(),
    )

    assert audit_a["seedStructure"]["weighted_mean_rows_per_household"] == pytest.approx(
        audit_b["seedStructure"]["weighted_mean_rows_per_household"]
    )


def test_build_us_seed_stage_parity_audit_marks_zero_reference_numeric_ratios(tmp_path) -> None:
    seed_path = tmp_path / "seed.parquet"
    reference_path = tmp_path / "reference.h5"

    pd.DataFrame(
        {
            "person_id": ["1"],
            "household_id": ["10"],
            "hh_weight": [10.0],
            "taxable_interest_income": [25.0],
        }
    ).to_parquet(seed_path, index=False)
    _write_period_dataset(
        reference_path,
        {
            "household_id": [10],
            "household_weight": [10.0],
            "person_id": [101],
            "person_household_id": [10],
            "taxable_interest_income": [0.0],
        },
    )

    audit = build_us_seed_stage_parity_audit(
        seed_path,
        reference_path,
        focus_variables=("taxable_interest_income",),
        boolean_landing_features=(),
        categorical_landing_features=(),
        candidate_only_landing_features=(),
    )

    comparison = audit["focusVariables"]["taxable_interest_income"]["comparison"]
    assert comparison["weighted_sum_ratio_defined"] is False
    assert comparison["weighted_sum_ratio_case"] == "candidate_nonzero_reference_zero"
    assert comparison["reference_scaled_weighted_sum_ratio_defined"] is False
    assert comparison["weighted_positive_share_ratio_defined"] is False


def test_seed_stage_module_imports_without_duckdb(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import builtins
import sys

real_import = builtins.__import__

def hooked(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "duckdb":
        raise ModuleNotFoundError("No module named 'duckdb'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = hooked

import microplex_us.pipelines.seed_stage_parity as mod
assert callable(mod.build_us_seed_stage_parity_audit)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_seed_tax_unit_support_payload_sorts_largest_gaps_first() -> None:
    payload = _seed_tax_unit_support_payload(
        seed_path=Path("/tmp/seed.parquet"),
        reference_path=Path("/tmp/reference.h5"),
        period=2024,
        support_audit={
            "candidate": {
                "filing_status_weighted_counts": {
                    "SINGLE": {"weighted_count": 120.0},
                    "JOINT": {"weighted_count": 80.0},
                    "SEPARATE": {"weighted_count": 5.0},
                    "HEAD_OF_HOUSEHOLD": {"weighted_count": 30.0},
                    "SURVIVING_SPOUSE": {"weighted_count": 2.0},
                },
                "mfs_high_agi_support": [
                    {
                        "agi_bin": "75k_to_100k",
                        "weighted_count": 10.0,
                        "weighted_agi": 850000.0,
                    },
                    {
                        "agi_bin": "500k_plus",
                        "weighted_count": 0.0,
                        "weighted_agi": 0.0,
                    },
                ],
            },
            "baseline": {
                "filing_status_weighted_counts": {
                    "SINGLE": {"weighted_count": 100.0},
                    "JOINT": {"weighted_count": 60.0},
                    "SEPARATE": {"weighted_count": 15.0},
                    "HEAD_OF_HOUSEHOLD": {"weighted_count": 20.0},
                    "SURVIVING_SPOUSE": {"weighted_count": 3.0},
                },
                "mfs_high_agi_support": [
                    {
                        "agi_bin": "75k_to_100k",
                        "weighted_count": 20.0,
                        "weighted_agi": 1700000.0,
                    },
                    {
                        "agi_bin": "500k_plus",
                        "weighted_count": 4.0,
                        "weighted_agi": 4000000.0,
                    },
                ],
            },
            "comparisons": {
                "filing_status_weighted_delta": [
                    {"filing_status": "SINGLE", "weighted_count_delta": 12.0},
                    {"filing_status": "JOINT", "weighted_count_delta": 20.0},
                    {"filing_status": "SEPARATE", "weighted_count_delta": -10.0},
                    {"filing_status": "HEAD_OF_HOUSEHOLD", "weighted_count_delta": 10.0},
                    {"filing_status": "SURVIVING_SPOUSE", "weighted_count_delta": -1.0},
                ],
                "mfs_high_agi_delta": [
                    {
                        "agi_bin": "75k_to_100k",
                        "weighted_count_delta": -10.0,
                        "weighted_agi_delta": -850000.0,
                    },
                    {
                        "agi_bin": "500k_plus",
                        "weighted_count_delta": -4.0,
                        "weighted_agi_delta": -4000000.0,
                    },
                ],
            },
        },
    )

    assert payload["comparisonStage"] == "seed_tax_unit_support"
    filing_rows = payload["comparisons"]["filing_status_weighted_delta"]
    mfs_rows = payload["comparisons"]["mfs_high_agi_delta"]

    assert abs(filing_rows[0]["weighted_count_delta"]) >= abs(
        filing_rows[1]["weighted_count_delta"]
    )
    assert mfs_rows[0]["agi_bin"] == "500k_plus"
    assert payload["verdictHints"]["largestFilingStatusGap"] == filing_rows[0]["filing_status"]
    assert payload["verdictHints"]["largestMFSAgiGap"] == "500k_plus"


def test_normalize_seed_ids_for_policyengine_support_factorizes_string_ids() -> None:
    normalized = _normalize_seed_ids_for_policyengine_support(
        pd.DataFrame(
            {
                "person_id": ["14:1", "14:2", "22:1"],
                "household_id": ["14", "14", "22"],
                "tax_unit_id": ["14", "14", "22"],
            }
        )
    )

    assert normalized["person_id"].dtype.kind in {"i", "u"}
    assert normalized["household_id"].dtype.kind in {"i", "u"}
    assert normalized["tax_unit_id"].dtype.kind in {"i", "u"}
    assert normalized["household_id"].tolist() == [0, 0, 1]
