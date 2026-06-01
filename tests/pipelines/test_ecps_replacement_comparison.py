"""Tests for sound Microplex-vs-eCPS replacement comparisons."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from microplex_us.pipelines import ecps_replacement_comparison as ecps
from microplex_us.pipelines.mp300k_artifact_gates import (
    write_mp300k_artifact_gate_report,
)
from microplex_us.policyengine.us import write_policyengine_us_time_period_dataset

_TARGET_NAMES = [
    "nation/irs/ssi",
    "nation/irs/snap",
    "nation/irs/employment_income",
    "nation/irs/self_employment_income",
    "nation/irs/capital_gains",
    "nation/irs/taxable_interest_income",
    "nation/irs/dividend_income",
    "nation/irs/pension_income",
    "nation/irs/disability_income",
    "nation/irs/household_net_income",
]


def _write_minimal_policyengine_dataset(
    path: Path,
    *,
    weights: tuple[float, float] = (0.5, 0.5),
    period: int = 2024,
) -> Path:
    arrays = {
        "household_id": {str(period): np.asarray([1, 2])},
        "household_weight": {str(period): np.asarray(weights, dtype=np.float32)},
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


def _read_weights(path: Path, *, period: int = 2024) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["household_weight"][str(period)], dtype=np.float64)


def _fake_loss_inputs(input_dataset_path: str | Path, **_kwargs) -> dict[str, object]:
    path = Path(input_dataset_path)
    if path.name.startswith("candidate"):
        matrix = np.tile(np.asarray([[1.0], [0.0]]), (1, len(_TARGET_NAMES)))
    else:
        matrix = np.tile(np.asarray([[0.5], [0.5]]), (1, len(_TARGET_NAMES)))
    return {
        "scaled_matrix": matrix,
        "scaled_target": np.ones(len(_TARGET_NAMES), dtype=np.float64),
        "initial_weights": _read_weights(path),
        "unscaled_target": np.ones(len(_TARGET_NAMES), dtype=np.float64),
        "scaling": np.ones(len(_TARGET_NAMES), dtype=np.float64),
        "metadata": {
            "target_names": list(_TARGET_NAMES),
            "n_targets_kept": len(_TARGET_NAMES),
        },
    }


def _fake_pe_native_scores(**kwargs) -> dict[str, object]:
    candidate_path = Path(kwargs["candidate_dataset_path"])
    baseline_path = Path(kwargs["baseline_dataset_path"])
    candidate_inputs = _fake_loss_inputs(candidate_path)
    baseline_inputs = _fake_loss_inputs(baseline_path)
    candidate_loss = ecps._objective(
        np.asarray(candidate_inputs["scaled_matrix"]),
        np.asarray(candidate_inputs["scaled_target"]),
        _read_weights(candidate_path),
    )
    baseline_loss = ecps._objective(
        np.asarray(baseline_inputs["scaled_matrix"]),
        np.asarray(baseline_inputs["scaled_target"]),
        _read_weights(baseline_path),
    )
    return {
        "metric": "enhanced_cps_native_loss",
        "period": 2024,
        "summary": {
            "candidate_enhanced_cps_native_loss": candidate_loss,
            "baseline_enhanced_cps_native_loss": baseline_loss,
            "enhanced_cps_native_loss_delta": candidate_loss - baseline_loss,
            "candidate_beats_baseline": candidate_loss < baseline_loss,
            "n_targets_kept": len(_TARGET_NAMES),
        },
        "broad_loss": {},
        "family_breakdown": [
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
        ],
    }


def test_target_value_diagnostics_falls_back_for_zero_scaling():
    loss_inputs = {
        "scaled_matrix": np.asarray([[2.0, 3.0], [0.0, 0.0]]),
        "scaled_target": np.asarray([1.0, 2.0]),
        "unscaled_target": np.asarray([10.0, 20.0]),
        "scaling": np.asarray([0.0, 0.5]),
    }

    diagnostics = ecps._target_value_diagnostics(
        loss_inputs,
        np.asarray([1.0, 0.0]),
    )

    assert diagnostics["value_scale"].tolist() == ["scaled", "native"]
    assert diagnostics["target"].tolist() == [1.0, 20.0]
    assert diagnostics["estimate"].tolist() == [2.0, 6.0]


def _fake_support_audit(**_kwargs) -> dict[str, object]:
    return {
        "metric": "enhanced_cps_support_audit",
        "comparisons": {
            "critical_input_support": [
                {
                    "variable": "medicare_part_b_premiums",
                    "candidate_stored": True,
                    "baseline_stored": False,
                    "weighted_nonzero_delta": 100.0,
                }
            ],
            "filing_status_weighted_delta": [
                {
                    "filing_status": "HEAD_OF_HOUSEHOLD",
                    "weighted_count_delta": 50.0,
                }
            ],
            "hoh_agi_delta": [
                {
                    "agi_bin": "500k_to_1m",
                    "weighted_count_delta": 40.0,
                }
            ],
            "ssi_by_age_delta": [
                {
                    "age_bucket": "65_plus",
                    "weighted_recipient_delta": 30.0,
                }
            ],
            "medicare_part_b_premiums_by_age_delta": [
                {
                    "age_bucket": "10_to_19",
                    "weighted_positive_delta": 20.0,
                }
            ],
            "state_aca_ptc_spending_top_gaps": [
                {
                    "state": "CA",
                    "weighted_aca_ptc_delta": -10.0,
                }
            ],
        },
    }


def test_protected_family_losses_match_pe_native_labels_with_spaces():
    target_names = [
        "nation/bea/wages and salaries",
        "nation/irs/capital gains gross/total/AGI in -inf-inf/taxable/All",
        "nation/census/household net income",
    ]
    candidate_inputs = {
        "scaled_matrix": np.asarray([[2.0, 2.0, 2.0]]),
        "scaled_target": np.ones(len(target_names), dtype=np.float64),
    }
    baseline_inputs = {
        "scaled_matrix": np.asarray([[1.0, 1.0, 1.0]]),
        "scaled_target": np.ones(len(target_names), dtype=np.float64),
    }

    rows = ecps._protected_family_losses(
        target_names=target_names,
        candidate_inputs=candidate_inputs,
        baseline_inputs=baseline_inputs,
        candidate_weights=np.asarray([1.0]),
        baseline_weights=np.asarray([1.0]),
    )

    assert rows["wages"]["n_targets"] == 1
    assert rows["capital_gains"]["n_targets"] == 1
    assert rows["household_net_income"]["n_targets"] == 1
    assert rows["wages"]["candidate_loss"] == pytest.approx(1.0)
    assert rows["wages"]["baseline_loss"] == pytest.approx(0.0)
    assert rows["capital_gains"]["loss_delta"] == pytest.approx(1.0)


def test_target_loss_diagnostics_family_breakdown_uses_total_loss_scale():
    target_names = [
        "nation/irs/capital gains gross/total/AGI in 1m-inf/taxable/All",
        "state/census/age/CA/65",
    ]
    candidate_inputs = {
        "scaled_matrix": np.asarray([[2.0, 3.0]]),
        "scaled_target": np.ones(len(target_names), dtype=np.float64),
    }
    baseline_inputs = {
        "scaled_matrix": np.asarray([[1.0, 1.0]]),
        "scaled_target": np.ones(len(target_names), dtype=np.float64),
    }

    diagnostics = ecps._target_loss_diagnostics(
        target_names=target_names,
        candidate_inputs=candidate_inputs,
        baseline_inputs=baseline_inputs,
        candidate_weights=np.asarray([1.0]),
        baseline_weights=np.asarray([1.0]),
        holdout_mask=np.asarray([False, True]),
        top_k=2,
    )

    assert diagnostics["summary"]["candidate_loss"] == pytest.approx(5.0)
    breakdown = {row["family"]: row for row in diagnostics["family_breakdown"]}
    assert breakdown["national_irs_other"][
        "candidate_loss_contribution"
    ] == pytest.approx(1.0)
    assert breakdown["state_age_distribution"][
        "candidate_loss_contribution"
    ] == pytest.approx(4.0)
    assert sum(
        row["candidate_loss_contribution"] for row in diagnostics["family_breakdown"]
    ) == pytest.approx(diagnostics["summary"]["candidate_loss"])


def _artifact_manifest(artifact_dir: Path, baseline_dataset: Path) -> None:
    (artifact_dir / "source_weight_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "source_name": "cps_asec",
                        "source_class": "base",
                        "household_weight_share": 0.95,
                    },
                    {
                        "source_name": "irs_soi_puf_support_clone",
                        "source_class": "puf_support",
                        "household_weight_share": 0.05,
                    },
                ],
            }
        )
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "policyengine_baseline_dataset": str(baseline_dataset),
                    "policyengine_dataset_year": 2024,
                },
                "artifacts": {
                    "policyengine_dataset": "candidate.h5",
                    "source_weight_diagnostics": "source_weight_diagnostics.json",
                },
            }
        )
    )


def _benchmark_manifest(path: Path) -> None:
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


def _arch_coverage_payload() -> dict[str, object]:
    return {
        "profile_name": "pe_native_broad_source_backed",
        "period": 2024,
        "target_cell_count": 183,
        "covered_cell_count": 183,
        "uncovered_cell_count": 0,
        "coverage_rate": 1.0,
    }


def test_sound_ecps_replacement_comparison_satisfies_gate_contract(
    monkeypatch,
    tmp_path,
):
    candidate = _write_minimal_policyengine_dataset(tmp_path / "candidate.h5")
    baseline = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    output_dir = tmp_path / "comparison"
    monkeypatch.setattr(ecps, "_extract_pe_native_loss_inputs", _fake_loss_inputs)
    monkeypatch.setattr(ecps, "compute_us_pe_native_scores", _fake_pe_native_scores)
    monkeypatch.setattr(ecps, "compute_us_pe_native_support_audit", _fake_support_audit)

    payload = ecps.build_sound_ecps_replacement_comparison(
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
        output_dir=output_dir,
        optimizer_max_iter=50,
    )

    summary = payload["summary"]
    assert summary["candidate_household_count"] == 2
    assert summary["baseline_household_count"] == 2
    assert payload["matched_datasets"]["sample_method"] == "uniform"
    assert summary["symmetric_refit"] is True
    assert summary["score_candidate_only"] is False
    assert summary["refit_objective_matches_scoring"] is True
    assert summary["ecps_refit_recovery_passed"] is True
    assert (
        summary["candidate_enhanced_cps_native_loss"]
        < summary["baseline_enhanced_cps_native_loss"]
    )
    assert summary["holdout_targets"] > 0
    assert set(summary["protected_family_losses"]) == {
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
    }
    assert summary["protected_family_losses"]["wages"]["n_targets"] == 1
    target_diagnostics = payload["target_diagnostics"]
    assert target_diagnostics["summary"]["n_targets"] == len(_TARGET_NAMES)
    assert target_diagnostics["summary"]["candidate_wins"] + target_diagnostics[
        "summary"
    ]["baseline_wins"] + target_diagnostics["summary"]["ties"] == len(_TARGET_NAMES)
    assert target_diagnostics["summary"]["train_targets"] > 0
    assert target_diagnostics["summary"]["holdout_targets"] > 0
    assert target_diagnostics["top_regressions"]
    assert target_diagnostics["top_improvements"]
    assert len(target_diagnostics["targets"]) == len(_TARGET_NAMES)
    first_target = target_diagnostics["targets"][0]
    assert first_target["value_scale"] == "native"
    assert first_target["target_value"] == pytest.approx(1.0)
    assert "candidate_estimate" in first_target
    assert "baseline_estimate" in first_target
    assert "candidate_relative_error" in first_target
    assert "baseline_relative_error" in first_target
    assert {row["split"] for row in target_diagnostics["targets"]} == {
        "train",
        "holdout",
    }
    assert target_diagnostics["family_breakdown"]
    support_summary = payload["summary"]["support_audit"]
    assert support_summary["top_filing_status_gaps"][0]["filing_status"] == (
        "HEAD_OF_HOUSEHOLD"
    )
    assert support_summary["top_hoh_agi_gaps"][0]["agi_bin"] == "500k_to_1m"
    assert support_summary["top_ssi_by_age_gaps"][0]["age_bucket"] == "65_plus"
    assert (
        support_summary["top_medicare_part_b_by_age_gaps"][0]["age_bucket"]
        == "10_to_19"
    )
    assert support_summary["top_aca_ptc_spending_gaps"][0]["state"] == "CA"
    structure = payload["entity_structure"]["candidate_matched"]
    assert structure["household_count"] == 2
    assert structure["person_count"] == 3
    assert structure["tax_unit_count"] == 2
    assert structure["tax_unit"]["singleton_unit_count"] == 1
    assert structure["tax_unit"]["singleton_unit_share"] == pytest.approx(0.5)
    assert structure["tax_unit"]["duplicate_unit_id_count"] == 0
    assert structure["tax_unit"]["missing_referenced_unit_count"] == 0
    assert structure["tax_unit"]["cross_household_unit_count"] == 0
    assert structure["spm_unit_count"] == 2
    assert structure["family_count"] == 2
    assert structure["marital_unit_count"] == 3
    assert structure["marital_unit"]["singleton_unit_share"] == pytest.approx(1.0)
    assert payload["entity_structure"]["baseline_refit"]["household_count"] == 2
    candidate_curve = payload["candidate_refit"]["loss_curve"]
    baseline_curve = payload["baseline_refit"]["loss_curve"]
    assert candidate_curve[0]["iteration"] == 0
    assert baseline_curve[0]["iteration"] == 0
    assert candidate_curve[0]["full_loss"] == pytest.approx(
        payload["candidate_refit"]["initial_full_loss"]
    )
    assert candidate_curve[-1]["full_loss"] == pytest.approx(
        payload["candidate_refit"]["optimized_full_loss"]
    )
    assert baseline_curve[-1]["holdout_loss"] == pytest.approx(
        payload["baseline_refit"]["optimized_holdout_loss"]
    )

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    shutil.copy2(candidate, artifact_dir / "candidate.h5")
    _artifact_manifest(artifact_dir, baseline)
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    _benchmark_manifest(benchmark_manifest)
    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload=payload,
        runtime_smoke_payload={"runtime_ratio": 1.0},
        arch_coverage_payload=_arch_coverage_payload(),
        benchmark_manifest_path=benchmark_manifest,
        compute_native_scores=False,
        update_manifest=False,
    )
    gate_report = json.loads(report_path.read_text())

    assert gate_report["summary"]["status"] == "passed"
    assert gate_report["gates"]["ecps_comparison"]["status"] == "pass"


def test_sound_ecps_replacement_comparison_writes_target_diagnostics_sidecar(
    monkeypatch,
    tmp_path,
):
    candidate = _write_minimal_policyengine_dataset(tmp_path / "candidate.h5")
    baseline = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    output_dir = tmp_path / "comparison"
    output_path = output_dir / "comparison.json"
    monkeypatch.setattr(ecps, "_extract_pe_native_loss_inputs", _fake_loss_inputs)
    monkeypatch.setattr(ecps, "compute_us_pe_native_scores", _fake_pe_native_scores)
    monkeypatch.setattr(ecps, "compute_us_pe_native_support_audit", _fake_support_audit)

    written = ecps.write_sound_ecps_replacement_comparison(
        output_path,
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
        output_dir=output_dir,
        optimizer_max_iter=50,
        target_diagnostics_top_k=3,
    )

    payload = json.loads(written.read_text())
    diagnostics_path = output_dir / "target_loss_diagnostics.json"
    diagnostics_payload = json.loads(diagnostics_path.read_text())
    descriptor = payload["artifacts"]["target_loss_diagnostics"]
    support_descriptor = payload["artifacts"]["support_audit"]

    assert descriptor["path"] == str(diagnostics_path.resolve())
    assert descriptor["size_bytes"] == diagnostics_path.stat().st_size
    assert payload["target_diagnostics"] == diagnostics_payload
    support_path = output_dir / "support_audit.json"
    assert support_descriptor["path"] == str(support_path.resolve())
    assert json.loads(support_path.read_text()) == payload["support_audit"]
    assert diagnostics_payload["summary"]["top_k"] == 3
    assert len(diagnostics_payload["top_regressions"]) == 3
    assert len(diagnostics_payload["top_improvements"]) == 3


def test_sound_ecps_replacement_comparison_flags_score_mismatch(
    monkeypatch,
    tmp_path,
):
    candidate = _write_minimal_policyengine_dataset(tmp_path / "candidate.h5")
    baseline = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    monkeypatch.setattr(ecps, "_extract_pe_native_loss_inputs", _fake_loss_inputs)
    monkeypatch.setattr(ecps, "compute_us_pe_native_support_audit", _fake_support_audit)

    def mismatched_scores(**kwargs):
        payload = _fake_pe_native_scores(**kwargs)
        payload["summary"]["candidate_enhanced_cps_native_loss"] += 0.1
        return payload

    monkeypatch.setattr(ecps, "compute_us_pe_native_scores", mismatched_scores)

    payload = ecps.build_sound_ecps_replacement_comparison(
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
        output_dir=tmp_path / "comparison",
        optimizer_max_iter=50,
    )

    assert payload["summary"]["refit_objective_matches_scoring"] is False
    assert payload["summary"]["candidate_score_abs_error"] == pytest.approx(0.1)


def test_sound_ecps_replacement_comparison_refuses_stale_matched_files(
    monkeypatch,
    tmp_path,
):
    candidate = _write_minimal_policyengine_dataset(tmp_path / "candidate.h5")
    baseline = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    output_dir = tmp_path / "comparison"
    output_dir.mkdir()
    (output_dir / "candidate_matched.h5").write_bytes(b"stale")
    monkeypatch.setattr(ecps, "_extract_pe_native_loss_inputs", _fake_loss_inputs)
    monkeypatch.setattr(ecps, "compute_us_pe_native_scores", _fake_pe_native_scores)

    with pytest.raises(FileExistsError):
        ecps.build_sound_ecps_replacement_comparison(
            candidate_dataset_path=candidate,
            baseline_dataset_path=baseline,
            output_dir=output_dir,
            optimizer_max_iter=50,
        )


def test_ecps_replacement_comparison_module_cli_help_runs():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "microplex_us.pipelines.ecps_replacement_comparison",
            "--help",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    assert "Build a sound Microplex-vs-eCPS replacement comparison payload" in (
        completed.stdout
    )
