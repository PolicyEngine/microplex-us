"""Tests for lightweight PE-US H5 readiness audits."""

from __future__ import annotations

import json

import h5py
import numpy as np

from microplex_us.pipelines.pe_us_dataset_readiness import (
    DEFAULT_EXPECTED_MATERIALIZED_VARIABLES,
    build_policyengine_us_dataset_readiness_audit,
    write_policyengine_us_dataset_readiness_audit,
)


def test_build_policyengine_us_dataset_readiness_audit_passes_complete_artifact(
    tmp_path,
):
    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    dataset_path = artifact_dir / "policyengine_us.h5"
    _write_dataset(dataset_path)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rows": {"calibrated": 2},
                "weights": {"total": 3.0},
                "artifacts": {
                    "policyengine_dataset": "policyengine_us.h5",
                    "source_spine_composition": "source_spine_composition.json",
                },
            }
        )
    )
    (artifact_dir / "source_spine_composition.json").write_text(
        json.dumps(
            {
                "household_count": 2,
                "nonzero_household_count": 2,
                "total_active_weight": 3.0,
                "effective_sample_size": 1.8,
                "groups": [
                    {
                        "spine": "cps_asec",
                        "household_count": 1,
                        "nonzero_household_count": 1,
                        "total_active_weight": 2.0,
                        "total_source_weight": 2.0,
                    },
                    {
                        "spine": "acs_pums",
                        "household_count": 1,
                        "nonzero_household_count": 1,
                        "total_active_weight": 1.0,
                        "total_source_weight": 5.0,
                    },
                ],
            }
        )
    )

    audit = build_policyengine_us_dataset_readiness_audit(artifact_dir, period=2024)

    assert audit["valid"] is True
    assert audit["entityCounts"] == {
        "household": 2,
        "person": 3,
        "tax_unit": 2,
        "spm_unit": 2,
    }
    assert audit["variableSummaries"]["state_fips"]["entity"] == "household"
    assert audit["variableSummaries"]["spm_unit_spm_threshold"]["positiveShare"] == 1.0
    assert audit["sourceSpineComposition"]["groups"][1]["spine"] == "acs_pums"
    assert audit["issues"] == []


def test_build_policyengine_us_dataset_readiness_audit_reports_missing_outputs(
    tmp_path,
):
    dataset_path = tmp_path / "policyengine_us.h5"
    _write_dataset(dataset_path, omit=("snap", "county_fips"))

    audit = build_policyengine_us_dataset_readiness_audit(
        dataset_path,
        expected_spines=(),
    )
    issues_by_variable = {
        issue.get("variable"): issue for issue in audit["issues"] if issue.get("variable")
    }

    assert audit["valid"] is False
    assert issues_by_variable["county_fips"]["severity"] == "error"
    assert issues_by_variable["snap"]["severity"] == "error"


def test_write_policyengine_us_dataset_readiness_audit_writes_sidecar(tmp_path):
    dataset_path = tmp_path / "policyengine_us.h5"
    _write_dataset(dataset_path)

    output_path = write_policyengine_us_dataset_readiness_audit(
        dataset_path,
        expected_spines=(),
    )

    assert output_path == tmp_path / "policyengine_us_readiness.json"
    payload = json.loads(output_path.read_text())
    assert payload["valid"] is True
    assert payload["expectedMaterializedVariables"] == list(
        DEFAULT_EXPECTED_MATERIALIZED_VARIABLES
    )


def _write_dataset(path, *, omit=()):
    omit = set(omit)
    arrays = {
        "household_id": np.array([1, 2]),
        "household_weight": np.array([2.0, 1.0]),
        "person_id": np.array([10, 11, 20]),
        "person_household_id": np.array([1, 1, 2]),
        "tax_unit_id": np.array([100, 200]),
        "person_tax_unit_id": np.array([100, 100, 200]),
        "spm_unit_id": np.array([500, 600]),
        "person_spm_unit_id": np.array([500, 500, 600]),
        "state_fips": np.array([6, 36]),
        "county_fips": np.array([b"06001", b"36061"]),
        "congressional_district_geoid": np.array([605, 3610]),
        "spm_unit_spm_threshold": np.array([30_000.0, 36_000.0]),
        "spm_unit_tenure_type": np.array([b"OWN_WITH_MORTGAGE", b"RENT"]),
        "income_tax": np.array([100.0, 200.0]),
        "income_tax_positive": np.array([100.0, 200.0]),
        "eitc": np.array([0.0, 50.0]),
        "ctc": np.array([1_000.0, 0.0]),
        "refundable_ctc": np.array([400.0, 0.0]),
        "non_refundable_ctc": np.array([600.0, 0.0]),
        "snap": np.array([10.0, 0.0]),
        "ssi": np.array([0.0, 100.0, 0.0]),
        "tanf": np.array([0.0, 0.0]),
        "medicaid": np.array([1.0, 0.0, 1.0]),
        "aca_ptc": np.array([0.0, 75.0]),
    }
    with h5py.File(path, "w") as handle:
        for variable, values in arrays.items():
            if variable in omit:
                continue
            group = handle.create_group(variable)
            group.create_dataset("2024", data=values)
