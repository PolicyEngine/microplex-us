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
from microplex_us.pipelines.pe_native_loss import build_pe_native_loss_arrays
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
    "state/CA/adjusted_gross_income/amount/0_1",
    "state/census/age/CA/65",
    "nation/ssa/retirement",
    "nation/irs/aca_spending/CA",
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
        "social_security_retirement": {str(period): np.asarray([1.0, 0.0, 0.0])},
        "social_security_disability": {str(period): np.asarray([0.0, 1.0, 0.0])},
        "employment_income_before_lsr": {str(period): np.asarray([100.0, 0.0, 200.0])},
    }
    return write_policyengine_us_time_period_dataset(arrays, path)


def _read_weights(path: Path, *, period: int = 2024) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["household_weight"][str(period)], dtype=np.float64)


def _write_clean_git_repo(path: Path) -> Path:
    path.mkdir()
    (path / "README.md").write_text("pinned scorer repo\n")
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Microplex Tests",
            "-c",
            "user.email=microplex-tests@example.com",
            "commit",
            "-m",
            "Initial scorer pin",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return path


def _fake_loss_inputs(input_dataset_path: str | Path, **_kwargs) -> dict[str, object]:
    path = Path(input_dataset_path)
    if path.name.startswith("candidate"):
        matrix = np.tile(np.asarray([[1.0], [0.0]]), (1, len(_TARGET_NAMES)))
    else:
        matrix = np.tile(np.asarray([[0.9], [0.0]]), (1, len(_TARGET_NAMES)))
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


def test_robust_loss_terms_match_objective() -> None:
    target_names = [
        "nation/irs/example income/total/AGI in 0_1/taxable/All",
        "nation/irs/example count/count/AGI in 0_1/taxable/All",
    ]
    targets = np.asarray([100.0, 10.0])
    loss_arrays = build_pe_native_loss_arrays(target_names, targets)
    matrix = np.asarray([[90.0, 20.0], [5.0, 0.0]])
    weights = np.asarray([1.0, 1.0])
    loss_inputs = {
        "scaled_matrix": matrix,
        "scaled_target": loss_arrays.objective_target,
        "unscaled_target": targets,
        "loss_denominator": loss_arrays.denominator,
        "loss_target_weight": loss_arrays.target_weight,
        "loss_bucket": loss_arrays.bucket_keys,
        "loss_unit": loss_arrays.unit_keys,
        "loss_scope": loss_arrays.scope_keys,
        "loss_family": loss_arrays.family_keys,
        "loss_epsilon": loss_arrays.epsilon,
        "metadata": {
            **loss_arrays.metadata(),
            "target_names": target_names,
        },
    }

    assert ecps._loss_terms(loss_inputs, weights).sum() == pytest.approx(
        ecps._objective(
            matrix,
            loss_arrays.objective_target,
            weights,
            loss_arrays=loss_arrays,
        )
    )


def test_comparison_bad_targets_exclude_dataset_derived_source_counts() -> None:
    bad_targets = ecps._comparison_bad_targets()

    assert len(bad_targets) == len(set(bad_targets))
    assert set(ecps._ENHANCED_CPS_BAD_TARGETS).issubset(bad_targets)
    assert {
        "nation/source/household_count",
        "nation/source/cps_household_count",
        "nation/source/puf_clone_household_count",
    }.issubset(bad_targets)


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


def _benchmark_manifest(
    path: Path,
    *,
    certificate: dict[str, object] | None = None,
) -> None:
    if certificate is not None:
        baseline_dataset = dict(certificate["baseline_dataset"])
        target_db = dict(certificate["target_db"])
        policyengine_us_data = dict(certificate["policyengine_us_data"])
    else:
        baseline_dataset = {
            "path": "/tmp/enhanced_cps_2024.h5",
            "sha256": "a" * 64,
        }
        target_db = {
            "path": "/tmp/policyengine_targets.db",
            "sha256": "c" * 64,
        }
        policyengine_us_data = {
            "repo": "PolicyEngine/policyengine-us-data",
            "commit": "b" * 40,
        }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "period": 2024,
                "target_profile": "pe_native_broad",
                "baseline_dataset": baseline_dataset,
                "policyengine_us_data": policyengine_us_data,
                "policyengine_us": {"version": "1.587.0"},
                "target_db": target_db,
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
    targets_db = tmp_path / "policyengine_targets.db"
    targets_db.write_bytes(b"pinned target database")
    scorer_repo = _write_clean_git_repo(tmp_path / "policyengine-us-data")
    output_dir = tmp_path / "comparison"
    monkeypatch.setattr(ecps, "_extract_pe_native_loss_inputs", _fake_loss_inputs)
    monkeypatch.setattr(ecps, "compute_us_pe_native_scores", _fake_pe_native_scores)
    monkeypatch.setattr(ecps, "compute_us_pe_native_support_audit", _fake_support_audit)

    payload = ecps.build_sound_ecps_replacement_comparison(
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
        output_dir=output_dir,
        optimizer_max_iter=50,
        policyengine_targets_db_path=targets_db,
        policyengine_us_data_repo=scorer_repo,
    )

    summary = payload["summary"]
    certificate = payload["frozen_ecps_baseline_certificate"]
    assert certificate["baseline_dataset"]["sha256"]
    assert certificate["target_db"]["sha256"]
    assert certificate["policyengine_us_data"]["commit"]
    assert (
        certificate["baseline_metrics"]["baseline_enhanced_cps_native_loss"]
        == (summary["baseline_enhanced_cps_native_loss"])
    )
    assert (
        certificate["baseline_metrics"]["baseline_holdout_loss"]
        == (summary["baseline_holdout_loss"])
    )
    assert summary["candidate_household_count"] == 2
    assert summary["baseline_household_count"] == 2
    assert payload["matched_datasets"]["sample_method"] == "uniform"
    assert summary["symmetric_refit"] is True
    assert summary["score_candidate_only"] is False
    assert summary["score_source"] == "refit_loss_matrix"
    assert summary["exact_rescore_status"] == "skipped"
    assert summary["refit_objective_matches_scoring"] is True
    assert summary["ecps_refit_recovery_passed"] is True
    assert summary["ecps_refit_effective_passed"] is True
    assert summary["baseline_sanity"]["mode"] == "msre"
    assert summary["baseline_sanity"]["status"] == "passed"
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
    _benchmark_manifest(benchmark_manifest, certificate=certificate)
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

    assert gate_report["summary"]["status"] == "failed"
    assert gate_report["summary"]["failed_required_gates"] == ["column_contract"]
    assert gate_report["gates"]["ecps_comparison"]["status"] == "pass"


def test_sound_ecps_replacement_comparison_skips_exact_rescore_by_default(
    monkeypatch,
    tmp_path,
):
    candidate = _write_minimal_policyengine_dataset(tmp_path / "candidate.h5")
    baseline = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    monkeypatch.setattr(ecps, "_extract_pe_native_loss_inputs", _fake_loss_inputs)
    monkeypatch.setattr(ecps, "compute_us_pe_native_support_audit", _fake_support_audit)

    def fail_exact_rescore(**_kwargs):
        raise AssertionError("exact PE-native rescore should be opt-in")

    monkeypatch.setattr(ecps, "compute_us_pe_native_scores", fail_exact_rescore)

    payload = ecps.build_sound_ecps_replacement_comparison(
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
        output_dir=tmp_path / "comparison",
        optimizer_max_iter=50,
    )

    assert payload["summary"]["score_source"] == "refit_loss_matrix"
    assert payload["summary"]["exact_rescore_requested"] is False
    assert payload["summary"]["exact_rescore_status"] == "skipped"
    assert payload["score"]["score_source"] == "refit_loss_matrix"


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
        exact_rescore=True,
    )

    assert payload["summary"]["refit_objective_matches_scoring"] is False
    assert payload["summary"]["candidate_score_abs_error"] == pytest.approx(0.1)


def test_sound_ecps_replacement_comparison_forwards_targets_db(
    monkeypatch,
    tmp_path,
):
    candidate = _write_minimal_policyengine_dataset(tmp_path / "candidate.h5")
    baseline = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    targets_db = tmp_path / "policy_data.db"
    targets_db.write_bytes(b"sqlite placeholder")
    captured_loss_calls: list[dict[str, object]] = []
    captured_score_call: dict[str, object] = {}

    def fake_loss_inputs_with_capture(**kwargs):
        captured_loss_calls.append(kwargs)
        return _fake_loss_inputs(kwargs["input_dataset_path"])

    def fake_scores_with_capture(**kwargs):
        captured_score_call.update(kwargs)
        return _fake_pe_native_scores(**kwargs)

    monkeypatch.setattr(
        ecps,
        "_extract_pe_native_loss_inputs",
        fake_loss_inputs_with_capture,
    )
    monkeypatch.setattr(ecps, "compute_us_pe_native_scores", fake_scores_with_capture)
    monkeypatch.setattr(ecps, "compute_us_pe_native_support_audit", _fake_support_audit)

    payload = ecps.build_sound_ecps_replacement_comparison(
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
        output_dir=tmp_path / "comparison",
        optimizer_max_iter=50,
        exact_rescore=True,
        policyengine_targets_db_path=targets_db,
    )

    assert len(captured_loss_calls) == 2
    assert {
        Path(call["policyengine_targets_db_path"]) for call in captured_loss_calls
    } == {targets_db.resolve()}
    assert Path(captured_score_call["policyengine_targets_db_path"]) == (
        targets_db.resolve()
    )
    assert payload["summary"]["policyengine_targets_db"]["path"] == str(
        targets_db.resolve()
    )


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
    assert "--policyengine-targets-db" in completed.stdout


def test_assert_refit_effective_passes_when_loss_drops():
    ecps._assert_refit_effective(
        "candidate",
        {"initial_full_loss": 0.05, "optimized_full_loss": 0.03},
        1e-9,
    )


def test_assert_refit_effective_raises_on_no_op_refit():
    # optimized == initial: the refit never reweighted (the d2d621b failure mode).
    with pytest.raises(ecps.ComparisonGateError) as excinfo:
        ecps._assert_refit_effective(
            "candidate",
            {"initial_full_loss": 0.042815, "optimized_full_loss": 0.042815},
            1e-9,
        )
    assert "no-op" in str(excinfo.value)


def test_assert_refit_effective_passes_when_loss_moves_but_rises():
    # The refit minimizes the train objective; on an already-well-calibrated
    # baseline the full-set loss can tick up from the held-out split even though
    # the refit genuinely ran. Only a frozen no-movement refit is a failure.
    ecps._assert_refit_effective(
        "baseline",
        {"initial_full_loss": 0.0243817, "optimized_full_loss": 0.0266164},
        1e-9,
    )


def test_assert_baseline_sane_passes_on_clean_baseline():
    ecps._assert_baseline_sane({"baseline_unweighted_msre": 0.17}, 2.0)


def test_assert_baseline_sane_raises_on_mis_scored_surface():
    # MSRE 8.17 is the soi-total-se mis-scored surface (eCPS +615% unemployment).
    with pytest.raises(ecps.ComparisonGateError) as excinfo:
        ecps._assert_baseline_sane({"baseline_unweighted_msre": 8.166}, 2.0)
    assert "anomalously" in str(excinfo.value)


def test_assert_baseline_sane_skips_when_msre_absent():
    # Exact-rescore path has no unweighted MSRE; the gate must no-op, not crash.
    summary = ecps._assert_baseline_sane({}, 2.0)
    assert summary["status"] == "skipped"


def test_assert_production_baseline_content_sane_passes_on_required_columns(
    tmp_path,
):
    baseline = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")

    summary = ecps._assert_production_baseline_content_sane(baseline, period=2024)

    assert summary["mode"] == "content"
    assert summary["status"] == "passed"
    assert (
        summary["required_nonzero_columns"]["social_security_retirement"][
            "nonzero_count"
        ]
        == 1
    )


def test_assert_production_baseline_content_sane_raises_on_missing_column(
    tmp_path,
):
    baseline = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    with h5py.File(baseline, "a") as handle:
        del handle["social_security_retirement"]

    with pytest.raises(ecps.ComparisonGateError) as excinfo:
        ecps._assert_production_baseline_content_sane(baseline, period=2024)

    assert "missing social_security_retirement/2024" in str(excinfo.value)


def test_sound_ecps_replacement_comparison_can_use_content_baseline_sanity(
    monkeypatch,
    tmp_path,
):
    candidate = _write_minimal_policyengine_dataset(tmp_path / "candidate.h5")
    baseline = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    monkeypatch.setattr(ecps, "_extract_pe_native_loss_inputs", _fake_loss_inputs)
    monkeypatch.setattr(ecps, "compute_us_pe_native_support_audit", _fake_support_audit)

    payload = ecps.build_sound_ecps_replacement_comparison(
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
        output_dir=tmp_path / "comparison",
        optimizer_max_iter=50,
        baseline_sanity_mode="content",
    )

    assert payload["summary"]["baseline_sanity"]["mode"] == "content"
    assert payload["summary"]["baseline_sanity"]["status"] == "passed"
