"""Tests for mp-300k artifact quality gates."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from microplex_us.pipelines.mp300k_artifact_gates import (
    main,
    write_mp300k_artifact_gate_report,
)
from microplex_us.policyengine.us import write_policyengine_us_time_period_dataset


def _write_minimal_policyengine_dataset(path: Path, *, period: int = 2024) -> Path:
    arrays = {
        "household_id": {str(period): np.asarray([1, 2])},
        "household_weight": {str(period): np.asarray([10.0, 20.0])},
        "person_id": {str(period): np.asarray([1, 2, 3])},
        "person_household_id": {str(period): np.asarray([1, 1, 2])},
        "tax_unit_id": {str(period): np.asarray([10, 20])},
        "person_tax_unit_id": {str(period): np.asarray([10, 10, 20])},
        "spm_unit_id": {str(period): np.asarray([100, 200])},
        "person_spm_unit_id": {str(period): np.asarray([100, 100, 200])},
        "family_id": {str(period): np.asarray([1000, 2000])},
        "person_family_id": {str(period): np.asarray([1000, 1000, 2000])},
        "marital_unit_id": {str(period): np.asarray([10000, 10001, 20000])},
        "person_marital_unit_id": {str(period): np.asarray([10000, 10001, 20000])},
    }
    return write_policyengine_us_time_period_dataset(arrays, path)


def _write_incomplete_policyengine_dataset(path: Path, *, period: int = 2024) -> Path:
    _write_minimal_policyengine_dataset(path, period=period)
    with h5py.File(path, "a") as handle:
        del handle["person_household_id"]
    return path


def _add_period_dataset(
    path: Path,
    variable: str,
    values: list[object] | np.ndarray,
    *,
    period: int = 2024,
) -> None:
    with h5py.File(path, "a") as handle:
        if variable in handle:
            del handle[variable]
        group = handle.create_group(variable)
        group.create_dataset(str(period), data=np.asarray(values))


def _write_artifact_manifest(
    artifact_dir: Path,
    *,
    baseline_dataset: Path | None = None,
    source_weight_diagnostics: bool = True,
) -> None:
    artifacts = {"policyengine_dataset": "candidate.h5"}
    if source_weight_diagnostics:
        (artifact_dir / "source_weight_diagnostics.json").write_text(
            json.dumps(_source_weight_diagnostics_payload())
        )
        artifacts["source_weight_diagnostics"] = "source_weight_diagnostics.json"
    manifest = {
        "created_at": "2026-05-27T00:00:00+00:00",
        "config": {
            "policyengine_baseline_dataset": str(baseline_dataset)
            if baseline_dataset is not None
            else None,
            "policyengine_dataset_year": 2024,
        },
        "artifacts": artifacts,
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))


def _write_benchmark_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "period": 2024,
                "target_profile": "pe_native_broad",
                "baseline_dataset": {
                    "path": "/tmp/enhanced_cps_2024.h5",
                    "sha256": "a" * 64,
                },
                "policyengine_us_data": {
                    "repo": "PolicyEngine/policyengine-us-data",
                    "commit": "b" * 40,
                },
                "policyengine_us": {"version": "1.587.0"},
                "target_db": {
                    "path": "/tmp/policyengine_targets.db",
                    "sha256": "c" * 64,
                },
            }
        )
    )


def _arch_coverage_payload(
    *,
    profile_name: str = "pe_native_broad_source_backed",
    period: int = 2024,
    target_cell_count: int = 183,
    uncovered_cell_count: int = 0,
) -> dict[str, object]:
    covered_cell_count = target_cell_count - uncovered_cell_count
    return {
        "profile_name": profile_name,
        "period": period,
        "target_cell_count": target_cell_count,
        "covered_cell_count": covered_cell_count,
        "uncovered_cell_count": uncovered_cell_count,
        "coverage_rate": (
            covered_cell_count / target_cell_count if target_cell_count else 0.0
        ),
    }


def _source_weight_diagnostics_payload(
    *,
    puf_support_share: float = 0.05,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "summary": {
            "max_source_household_weight_share": 0.85,
            "puf_support_household_weight_share": puf_support_share,
        },
        "sources": [
            {
                "source_name": "cps_asec",
                "source_class": "base",
                "household_weight_share": 0.85,
            },
            {
                "source_name": "irs_soi_puf_support_clone",
                "source_class": "puf_support",
                "household_weight_share": puf_support_share,
            },
            {
                "source_name": "forbes_fixed_spine",
                "source_class": "fixed_spine",
                "household_weight_share": 0.10,
            },
        ],
    }


def _sound_ecps_comparison_payload(
    *,
    candidate_loss: float = 0.12,
    baseline_loss: float = 0.20,
) -> dict[str, object]:
    fit_config = {
        "lambda_l0": 0.0,
        "lambda_l2": 0.0,
        "use_gates": False,
        "epochs": 2000,
    }
    protected_family_losses = {
        family: {"candidate_loss": 0.01, "baseline_loss": 0.01}
        for family in (
            "ssi",
            "snap",
            "wages",
            "self_employment_income",
            "capital_gains",
            "interest",
            "dividends",
            "retirement_income",
            "disability",
            "household_net_income",
        )
    }
    family_breakdown = [
        {
            "family": family,
            "candidate_loss_contribution": 0.01,
            "baseline_loss_contribution": 0.01,
        }
        for family in (
            "state_agi_distribution",
            "state_age_distribution",
            "national_ssa",
            "national_irs_other",
            "state_aca_spending",
        )
    ]
    return {
        "summary": {
            "candidate_enhanced_cps_native_loss": candidate_loss,
            "baseline_enhanced_cps_native_loss": baseline_loss,
            "enhanced_cps_native_loss_delta": candidate_loss - baseline_loss,
            "candidate_beats_baseline": candidate_loss < baseline_loss,
            "n_targets_kept": 150,
            "candidate_household_count": 41_314,
            "baseline_household_count": 41_314,
            "candidate_refit_config": fit_config,
            "baseline_refit_config": fit_config,
            "refit_objective_matches_scoring": True,
            "ecps_refit_recovery_passed": True,
            "holdout_target_fraction": 0.2,
            "protected_family_losses": protected_family_losses,
        },
        "score": {"family_breakdown": family_breakdown},
    }


def test_write_mp300k_artifact_gate_report_passes_with_all_evidence(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    candidate_dataset = _write_minimal_policyengine_dataset(
        artifact_dir / "candidate.h5"
    )
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={
            "candidate_seconds": 11.0,
            "baseline_seconds": 10.0,
        },
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
    )

    record = json.loads(report_path.read_text())
    manifest = json.loads((artifact_dir / "manifest.json").read_text())

    assert record["summary"]["status"] == "passed"
    assert record["gates"]["candidate_artifact"]["status"] == "pass"
    assert record["gates"]["compatibility"]["metrics"]["household_count"] == 2
    assert record["gates"]["compatibility"]["metrics"]["person_count"] == 3
    assert record["gates"]["column_contract"]["status"] == "pass"
    assert record["gates"]["export_support"]["status"] == "pass"
    assert record["gates"]["export_lineage"]["status"] == "pass"
    assert record["gates"]["artifact_size"]["status"] == "pass"
    assert record["gates"]["ecps_comparison"]["status"] == "pass"
    assert record["gates"]["arch_target_coverage"]["status"] == "pass"
    assert record["gates"]["runtime"]["status"] == "pass"
    assert record["gates"]["runtime"]["metrics"]["runtime_ratio"] == 1.1
    assert record["gates"]["source_weight_diagnostics"]["status"] == "pass"
    assert (
        record["gates"]["source_weight_diagnostics"]["metrics"][
            "puf_support_household_weight_share"
        ]
        == 0.05
    )
    assert record["gates"]["benchmark_manifest"]["status"] == "pass"
    assert record["candidate_dataset"]["path"] == str(candidate_dataset.resolve())
    assert (
        manifest["artifacts"]["mp300k_artifact_gates"] == "mp300k_artifact_gates.json"
    )
    assert manifest["mp300k_artifact_gates"]["status"] == "passed"


def test_benchmark_manifest_gate_requires_pinned_release_evidence(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    benchmark_manifest.write_text(json.dumps({"schema_version": 1, "frozen": True}))
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    benchmark_gate = record["gates"]["benchmark_manifest"]

    assert record["summary"]["status"] == "failed"
    assert benchmark_gate["status"] == "fail"
    assert benchmark_gate["details"]["missing_evidence"] == [
        "baseline_dataset.path",
        "baseline_dataset.sha256",
        "policyengine_us_data.commit",
        "policyengine_us.version",
        "target_db.path",
        "target_db.sha256",
    ]


def test_column_contract_gate_rejects_missing_ecps_contract_column(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _add_period_dataset(baseline_dataset, "age", [34, 12, 45])
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    column_gate = record["gates"]["column_contract"]

    assert record["summary"]["status"] == "failed"
    assert column_gate["status"] == "fail"
    assert column_gate["metrics"]["missing_contract_column_count"] == 1
    assert column_gate["details"]["missing_contract_columns"] == ["age"]


def test_export_support_gate_rejects_ecps_populated_numeric_filler(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    candidate_dataset = _write_minimal_policyengine_dataset(
        artifact_dir / "candidate.h5"
    )
    _add_period_dataset(candidate_dataset, "hourly_wage", [0.0, 0.0, 0.0])
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _add_period_dataset(baseline_dataset, "hourly_wage", [0.0, 25.0, 0.0])
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    support_gate = record["gates"]["export_support"]

    assert record["summary"]["status"] == "failed"
    assert support_gate["status"] == "fail"
    assert support_gate["metrics"]["unsupported_populated_export_column_count"] == 1
    assert support_gate["details"]["issues"][0]["column"] == "hourly_wage"
    assert support_gate["details"]["issues"][0]["requirement"] == "numeric_positive"


def test_export_support_gate_requires_signed_self_employment_support(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    candidate_dataset = _write_minimal_policyengine_dataset(
        artifact_dir / "candidate.h5"
    )
    _add_period_dataset(
        candidate_dataset,
        "self_employment_income_before_lsr",
        [0.0, 5_000.0, 0.0],
    )
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _add_period_dataset(
        baseline_dataset,
        "self_employment_income_before_lsr",
        [-2_000.0, 5_000.0, 0.0],
    )
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    support_gate = record["gates"]["export_support"]

    assert record["summary"]["status"] == "failed"
    assert support_gate["status"] == "fail"
    assert support_gate["metrics"]["unsupported_populated_export_column_count"] == 1
    assert support_gate["details"]["issues"][0]["column"] == (
        "self_employment_income_before_lsr"
    )
    assert support_gate["details"]["issues"][0]["requirement"] == "numeric_signed"


def test_export_support_gate_rejects_ecps_varied_categorical_filler(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    candidate_dataset = _write_minimal_policyengine_dataset(
        artifact_dir / "candidate.h5"
    )
    _add_period_dataset(candidate_dataset, "is_tipped_occupation", [False, False])
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _add_period_dataset(baseline_dataset, "is_tipped_occupation", [False, True])
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    support_gate = record["gates"]["export_support"]

    assert record["summary"]["status"] == "failed"
    assert support_gate["status"] == "fail"
    assert support_gate["details"]["issues"][0]["column"] == "is_tipped_occupation"
    assert (
        support_gate["details"]["issues"][0]["requirement"] == "categorical_variation"
    )


def test_export_support_gate_ignores_ecps_filler_columns(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    candidate_dataset = _write_minimal_policyengine_dataset(
        artifact_dir / "candidate.h5"
    )
    _add_period_dataset(candidate_dataset, "second_home_mortgage_interest", [0.0, 0.0])
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _add_period_dataset(baseline_dataset, "second_home_mortgage_interest", [0.0, 0.0])
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    support_gate = record["gates"]["export_support"]

    assert record["summary"]["status"] == "passed"
    assert support_gate["status"] == "pass"
    assert support_gate["metrics"]["ecps_filler_export_column_count"] == 1


def test_export_lineage_gate_rejects_ecps_populated_default_only_column(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    candidate_dataset = _write_minimal_policyengine_dataset(
        artifact_dir / "candidate.h5"
    )
    _add_period_dataset(
        candidate_dataset,
        "is_wic_at_nutritional_risk",
        [False, True],
    )
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _add_period_dataset(
        baseline_dataset,
        "is_wic_at_nutritional_risk",
        [False, True],
    )
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    support_gate = record["gates"]["export_support"]
    lineage_gate = record["gates"]["export_lineage"]

    assert record["summary"]["status"] == "failed"
    assert support_gate["status"] == "pass"
    assert lineage_gate["status"] == "fail"
    assert lineage_gate["details"]["issues"] == [
        {
            "column": "is_wic_at_nutritional_risk",
            "ecps_support_requirement": "categorical_variation",
            "export_path_status": "default_only",
            "issue": "ecps_populated_export_has_no_source_lineage",
        }
    ]


def test_column_contract_gate_rejects_extra_candidate_columns(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    candidate_dataset = _write_minimal_policyengine_dataset(
        artifact_dir / "candidate.h5"
    )
    _add_period_dataset(candidate_dataset, "filing_status", [1, 2])
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    column_gate = record["gates"]["column_contract"]

    assert record["summary"]["status"] == "failed"
    assert column_gate["status"] == "fail"
    assert column_gate["metrics"]["extra_candidate_column_count"] == 1
    assert column_gate["details"]["extra_candidate_columns"] == ["filing_status"]


def test_column_contract_gate_rejects_renamed_candidate_columns(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    candidate_dataset = _write_minimal_policyengine_dataset(
        artifact_dir / "candidate.h5"
    )
    _add_period_dataset(
        candidate_dataset,
        "medicare_part_b_premiums_reported",
        [0.0, 0.0, 0.0],
    )
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _add_period_dataset(baseline_dataset, "medicare_part_b_premiums", [0.0, 0.0, 0.0])
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    column_gate = record["gates"]["column_contract"]

    assert record["summary"]["status"] == "failed"
    assert column_gate["status"] == "fail"
    assert column_gate["details"]["missing_contract_columns"] == [
        "medicare_part_b_premiums"
    ]
    assert column_gate["details"]["extra_candidate_columns"] == [
        "medicare_part_b_premiums_reported"
    ]


def test_column_contract_gate_excludes_computed_baseline_outputs(
    tmp_path,
):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _add_period_dataset(
        baseline_dataset,
        "self_employed_health_insurance_ald",
        [0.0, 0.0],
    )
    _add_period_dataset(
        baseline_dataset,
        "self_employed_pension_contribution_ald",
        [0.0, 0.0],
    )
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    column_gate = record["gates"]["column_contract"]

    assert record["summary"]["status"] == "passed"
    assert column_gate["status"] == "pass"
    assert column_gate["metrics"]["excluded_baseline_computed_column_count"] == 2
    assert column_gate["details"]["excluded_baseline_computed_columns"] == [
        "self_employed_health_insurance_ald",
        "self_employed_pension_contribution_ald",
    ]


def test_column_contract_gate_rejects_missing_legacy_baseline_columns(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _add_period_dataset(baseline_dataset, "taxpayer_id_type", [1, 1, 1])
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    column_gate = record["gates"]["column_contract"]

    assert record["summary"]["status"] == "failed"
    assert column_gate["status"] == "fail"
    assert column_gate["details"]["missing_contract_columns"] == ["taxpayer_id_type"]


def test_source_weight_diagnostics_gate_rejects_missing_sidecar(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(
        artifact_dir,
        baseline_dataset=baseline_dataset,
        source_weight_diagnostics=False,
    )

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    source_gate = record["gates"]["source_weight_diagnostics"]

    assert record["summary"]["status"] == "incomplete"
    assert source_gate["status"] == "unmeasured"
    assert "source_weight_diagnostics" in record["summary"]["unmeasured_required_gates"]


def test_source_weight_diagnostics_gate_rejects_puf_support_dominance(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        source_weight_diagnostics_payload=_source_weight_diagnostics_payload(
            puf_support_share=0.40
        ),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    source_gate = record["gates"]["source_weight_diagnostics"]

    assert record["summary"]["status"] == "failed"
    assert source_gate["status"] == "fail"
    assert source_gate["metrics"]["puf_support_household_weight_share"] == 0.40
    assert source_gate["details"]["failures"] == [
        "support_household_weight_share",
        "puf_support_household_weight_share",
    ]


def test_arch_target_coverage_gate_rejects_uncovered_source_backed_cells(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(uncovered_cell_count=1),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    coverage_gate = record["gates"]["arch_target_coverage"]

    assert record["summary"]["status"] == "failed"
    assert coverage_gate["status"] == "fail"
    assert coverage_gate["details"]["failures"] == [
        "uncovered_cell_count",
        "covered_cell_count",
        "coverage_rate",
    ]


def test_benchmark_manifest_gate_rejects_dirty_us_data_pin(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    payload = json.loads(benchmark_manifest.read_text())
    payload["policyengine_us_data"]["dirty"] = True
    benchmark_manifest.write_text(json.dumps(payload))
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    benchmark_gate = record["gates"]["benchmark_manifest"]

    assert record["summary"]["status"] == "failed"
    assert benchmark_gate["status"] == "fail"
    assert benchmark_gate["details"]["missing_evidence"] == [
        "policyengine_us_data.clean"
    ]


def test_write_mp300k_artifact_gate_report_fails_missing_structural_array(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_incomplete_policyengine_dataset(artifact_dir / "candidate.h5")
    _write_artifact_manifest(artifact_dir)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())

    assert record["summary"]["status"] == "failed"
    assert record["gates"]["candidate_artifact"]["status"] == "pass"
    assert record["gates"]["artifact_size"]["status"] == "unmeasured"
    assert record["gates"]["compatibility"]["status"] == "fail"
    assert record["gates"]["compatibility"]["details"]["missing_arrays"] == [
        "person_household_id"
    ]
    assert record["gates"]["ecps_comparison"]["status"] == "unmeasured"


def test_write_mp300k_artifact_gate_report_fails_invalid_entity_join(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    with h5py.File(artifact_dir / "candidate.h5", "a") as handle:
        handle["person_household_id"]["2024"][2] = 999
    _write_artifact_manifest(artifact_dir)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())

    assert record["summary"]["status"] == "failed"
    assert record["gates"]["compatibility"]["status"] == "fail"
    assert record["gates"]["compatibility"]["details"][
        "invalid_person_entity_links"
    ] == {"person_household_id": [999]}


def test_write_mp300k_artifact_gate_report_fails_source_diagnostic_variable(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    with h5py.File(artifact_dir / "candidate.h5", "a") as handle:
        diagnostic = handle.create_group("ssi_reported")
        diagnostic.create_dataset("2024", data=np.asarray([1.0, 0.0, 0.0]))
    _write_artifact_manifest(artifact_dir)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())

    assert record["summary"]["status"] == "failed"
    assert record["gates"]["compatibility"]["status"] == "fail"
    assert record["gates"]["compatibility"]["details"][
        "forbidden_source_diagnostic_variables"
    ] == ["ssi_reported"]


def test_write_mp300k_artifact_gate_report_fails_nonfinite_numeric_value(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    with h5py.File(artifact_dir / "candidate.h5", "a") as handle:
        income = handle.create_group("employment_income")
        income.create_dataset("2024", data=np.asarray([1.0, np.nan, 3.0]))
    _write_artifact_manifest(artifact_dir)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())

    assert record["summary"]["status"] == "failed"
    assert record["gates"]["compatibility"]["status"] == "fail"
    assert record["gates"]["compatibility"]["details"]["nonfinite_numeric_arrays"] == {
        "employment_income": 1
    }


def test_write_mp300k_artifact_gate_report_reports_missing_candidate(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_artifact_manifest(artifact_dir)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())

    assert record["summary"]["status"] == "failed"
    assert record["candidate_dataset"]["exists"] is False
    assert record["gates"]["candidate_artifact"]["status"] == "fail"


def test_main_writes_artifact_gate_report_from_payload_files(tmp_path, capsys):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)
    ecps_comparison_path = tmp_path / "ecps_comparison.json"
    ecps_comparison_path.write_text(
        json.dumps(_sound_ecps_comparison_payload(candidate_loss=0.10))
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps({"runtime_ratio": 1.2, "runtime_ratio_threshold": 1.25})
    )
    arch_coverage_path = tmp_path / "arch_coverage.json"
    arch_coverage_path.write_text(json.dumps(_arch_coverage_payload()))
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)

    exit_code = main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--ecps-comparison-json",
            str(ecps_comparison_path),
            "--runtime-smoke-json",
            str(runtime_path),
            "--arch-coverage-json",
            str(arch_coverage_path),
            "--benchmark-manifest",
            str(benchmark_manifest),
        ]
    )

    printed_path = Path(capsys.readouterr().out.strip())
    record = json.loads(printed_path.read_text())

    assert exit_code == 0
    assert printed_path == artifact_dir / "mp300k_artifact_gates.json"
    assert record["summary"]["status"] == "passed"


def test_ecps_comparison_can_become_nonblocking(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        runtime_smoke_payload={
            "runtime_ratio": 1.0,
            "runtime_ratio_threshold": 1.25,
        },
        arch_coverage_payload=_arch_coverage_payload(),
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        require_ecps_comparison=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())

    assert record["summary"]["status"] == "passed"
    assert "ecps_comparison" not in record["required_gates"]
    assert record["gates"]["ecps_comparison"]["status"] == "unmeasured"
    assert record["summary"]["unmeasured_optional_gates"] == ["ecps_comparison"]


def test_runtime_gate_accepts_repeated_loader_smoke_payload(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(candidate_loss=0.10),
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={
            "median_runtime_ratio": 1.19,
            "candidate": {"median_elapsed_seconds": 0.137},
            "baseline": {"median_elapsed_seconds": 0.115},
        },
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())

    assert record["summary"]["status"] == "passed"
    assert record["gates"]["runtime"]["status"] == "pass"
    assert record["gates"]["runtime"]["metrics"]["runtime_ratio"] == 1.19
    assert record["gates"]["runtime"]["metrics"]["candidate_seconds"] == 0.137
    assert record["gates"]["runtime"]["metrics"]["baseline_seconds"] == 0.115


def test_ecps_comparison_accepts_existing_broad_loss_array_payload(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=[
            {
                "broad_loss": {
                    "candidate_enhanced_cps_native_loss": 0.25,
                    "baseline_enhanced_cps_native_loss": 0.20,
                    "enhanced_cps_native_loss_delta": 0.05,
                    "candidate_beats_baseline": False,
                }
            }
        ],
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())

    assert record["summary"]["status"] == "failed"
    assert record["gates"]["ecps_comparison"]["status"] == "fail"
    assert (
        record["gates"]["ecps_comparison"]["metrics"][
            "candidate_enhanced_cps_native_loss"
        ]
        == 0.25
    )


def test_ecps_comparison_rejects_one_sided_unmatched_refit_win(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload={
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.09,
                "baseline_enhanced_cps_native_loss": 0.16,
                "enhanced_cps_native_loss_delta": -0.07,
                "candidate_beats_baseline": True,
                "candidate_household_count": 120_000,
                "baseline_household_count": 41_314,
                "score_candidate_only": True,
            }
        },
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    ecps_gate = record["gates"]["ecps_comparison"]

    assert record["summary"]["status"] == "failed"
    assert ecps_gate["status"] == "fail"
    assert "matched_household_count" in ecps_gate["summary"]
    assert ecps_gate["details"]["score_candidate_only"] is True


def test_ecps_comparison_rejects_protected_family_regression(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)
    payload = _sound_ecps_comparison_payload(candidate_loss=0.10)
    payload["summary"]["protected_family_losses"]["ssi"] = {
        "candidate_loss": 0.0301,
        "baseline_loss": 0.02,
    }

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=payload,
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    ecps_gate = record["gates"]["ecps_comparison"]

    assert record["summary"]["status"] == "failed"
    assert ecps_gate["status"] == "fail"
    assert "protected_family_floors" in ecps_gate["summary"]
    assert ecps_gate["details"]["protected_family_floor"]["regressions"] == [
        {
            "family": "ssi",
            "candidate_loss": 0.0301,
            "baseline_loss": 0.02,
            "loss_delta": pytest.approx(0.0101),
            "allowed_delta": 0.005,
        }
    ]


def test_ecps_comparison_rejects_core_benchmark_family_regression(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)
    payload = _sound_ecps_comparison_payload(candidate_loss=0.10)
    payload["score"]["family_breakdown"][0] = {
        "family": "state_agi_distribution",
        "candidate_loss_contribution": 0.0601,
        "baseline_loss_contribution": 0.05,
    }

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=payload,
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    ecps_gate = record["gates"]["ecps_comparison"]

    assert record["summary"]["status"] == "failed"
    assert ecps_gate["status"] == "fail"
    assert "core_benchmark_family_floors" in ecps_gate["summary"]
    assert ecps_gate["details"]["core_benchmark_family_floor"]["regressions"] == [
        {
            "family": "state_agi_distribution",
            "candidate_loss": 0.0601,
            "baseline_loss": 0.05,
            "loss_delta": pytest.approx(0.0101),
            "allowed_delta": 0.005,
        }
    ]


def test_ecps_comparison_rejects_missing_ecps_refit_recovery(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)
    payload = _sound_ecps_comparison_payload(candidate_loss=0.10)
    payload["summary"]["ecps_refit_recovery_passed"] = False

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=payload,
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    ecps_gate = record["gates"]["ecps_comparison"]

    assert record["summary"]["status"] == "failed"
    assert ecps_gate["status"] == "fail"
    assert "ecps_refit_recovery" in ecps_gate["summary"]
    assert ecps_gate["details"]["ecps_refit_recovery_passed"] is False


def test_ecps_comparison_requires_measured_refit_objective_identity(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)
    payload = _sound_ecps_comparison_payload(candidate_loss=0.10)
    del payload["summary"]["refit_objective_matches_scoring"]

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=payload,
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    ecps_gate = record["gates"]["ecps_comparison"]

    assert record["summary"]["status"] == "failed"
    assert ecps_gate["status"] == "fail"
    assert "refit_objective_matches_scoring" in ecps_gate["summary"]
    assert ecps_gate["details"]["refit_objective_matches_scoring"] is None


def test_runtime_gate_ignores_contradictory_producer_verdict(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload={
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.1,
                "baseline_enhanced_cps_native_loss": 0.2,
            }
        },
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={
            "runtime_ratio": 10.0,
            "runtime_ratio_threshold": 100.0,
            "passes_runtime_gate": True,
        },
        benchmark_manifest_path=benchmark_manifest,
        runtime_ratio_threshold=1.25,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    runtime_gate = record["gates"]["runtime"]

    assert record["summary"]["status"] == "failed"
    assert runtime_gate["status"] == "fail"
    assert runtime_gate["metrics"]["runtime_ratio_threshold"] == 1.25
    assert runtime_gate["details"]["reported_runtime_ratio_threshold"] == 100.0
    assert runtime_gate["details"]["enforced_runtime_ratio_threshold"] == 1.25
    assert runtime_gate["details"]["reported_passes_runtime_gate"] is True
    assert runtime_gate["details"]["computed_passes_runtime_gate"] is False


def test_ecps_gate_derives_verdict_from_losses_not_producer_flag(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _write_benchmark_manifest(benchmark_manifest)
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload={
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.3,
                "baseline_enhanced_cps_native_loss": 0.2,
                "enhanced_cps_native_loss_delta": -0.1,
                "candidate_beats_baseline": True,
            }
        },
        arch_coverage_payload=_arch_coverage_payload(),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )

    record = json.loads(report_path.read_text())
    ecps_gate = record["gates"]["ecps_comparison"]

    assert record["summary"]["status"] == "failed"
    assert ecps_gate["status"] == "fail"
    assert ecps_gate["metrics"]["enhanced_cps_native_loss_delta"] == pytest.approx(0.1)
    assert ecps_gate["details"]["reported_candidate_beats_baseline"] is True
    assert ecps_gate["details"]["computed_candidate_beats_baseline"] is False
