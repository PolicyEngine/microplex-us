"""Tests for packaging mp-300k gate inputs."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np

from microplex_us.pipelines.mp300k_artifact_gates import (
    write_mp300k_artifact_gate_report,
)
from microplex_us.pipelines.mp300k_gate_inputs import (
    main,
    package_mp300k_gate_inputs,
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


def _write_manifest(
    artifact_dir: Path,
    *,
    candidate_path: str = "policyengine_us.h5",
    baseline_path: str = "baseline/enhanced_cps_2024.h5",
) -> None:
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-05-27T00:00:00+00:00",
                "config": {
                    "policyengine_dataset_year": 2024,
                    "policyengine_baseline_dataset": baseline_path,
                },
                "artifacts": {"policyengine_dataset": candidate_path},
            }
        )
    )


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


def _arch_coverage_payload() -> dict[str, object]:
    return {
        "profile_name": "pe_native_broad_source_backed",
        "period": 2024,
        "target_cell_count": 183,
        "covered_cell_count": 183,
        "uncovered_cell_count": 0,
        "coverage_rate": 1.0,
    }


def _archive_manifest(archive_path: Path) -> dict:
    with tarfile.open(archive_path) as archive:
        manifest = archive.extractfile("artifact/manifest.json")
        assert manifest is not None
        return json.loads(manifest.read())


def _sound_ecps_comparison_payload() -> dict[str, object]:
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
            "candidate_enhanced_cps_native_loss": 0.1,
            "baseline_enhanced_cps_native_loss": 0.2,
            "candidate_household_count": 2,
            "baseline_household_count": 2,
            "candidate_refit_config": fit_config,
            "baseline_refit_config": fit_config,
            "refit_objective_matches_scoring": True,
            "ecps_refit_recovery_passed": True,
            "holdout_target_fraction": 0.2,
            "protected_family_losses": protected_family_losses,
        },
        "score": {"family_breakdown": family_breakdown},
    }


def test_package_mp300k_gate_inputs_rewrites_external_candidate(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_manifest(artifact_dir)
    external_candidate = tmp_path / "external" / "pe_l0_candidate.h5"
    external_candidate.parent.mkdir()
    external_candidate.write_bytes(b"candidate")
    baseline_dataset = artifact_dir / "baseline" / "enhanced_cps_2024.h5"
    baseline_dataset.parent.mkdir()
    baseline_dataset.write_bytes(b"baseline")
    ecps_comparison = tmp_path / "scores.json"
    ecps_comparison.write_text(json.dumps([{"broad_loss": {}}]))
    arch_coverage = tmp_path / "arch_coverage.json"
    arch_coverage.write_text(json.dumps(_arch_coverage_payload()))
    runtime_smoke = tmp_path / "runtime.json"
    runtime_smoke.write_text(json.dumps({"runtime_ratio": 1.0}))
    benchmark_manifest = tmp_path / "benchmark.json"
    _write_benchmark_manifest(benchmark_manifest)

    metadata = package_mp300k_gate_inputs(
        artifact_dir,
        tmp_path / "gate-inputs",
        candidate_dataset_path=external_candidate,
        ecps_comparison_path=ecps_comparison,
        arch_coverage_path=arch_coverage,
        runtime_smoke_path=runtime_smoke,
        benchmark_manifest_path=benchmark_manifest,
    )

    output_dir = tmp_path / "gate-inputs"
    archive_path = output_dir / "artifact.tar.gz"
    manifest = _archive_manifest(archive_path)

    assert archive_path.exists()
    assert (output_dir / "ecps_comparison.json").exists()
    assert (output_dir / "arch_coverage.json").exists()
    assert (output_dir / "runtime_smoke.json").exists()
    assert (output_dir / "benchmark_manifest.json").exists()
    assert (output_dir / "gate_inputs.json").exists()
    assert manifest["artifacts"]["policyengine_dataset"] == "pe_l0_candidate.h5"
    assert (
        manifest["config"]["policyengine_baseline_dataset"]
        == "baseline/enhanced_cps_2024.h5"
    )
    assert manifest["mp300k_gate_inputs"]["source_candidate_dataset"] == str(
        external_candidate.resolve()
    )
    assert manifest["mp300k_gate_inputs"]["source_baseline_dataset"] == str(
        baseline_dataset.resolve()
    )
    assert metadata["artifact_archive"]["path"] == str(archive_path.resolve())
    assert metadata["workflow_call"]["with"]["gate_inputs_artifact"] == "gate-inputs"


def test_main_packages_gate_inputs(tmp_path, capsys):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_manifest(artifact_dir)
    (artifact_dir / "policyengine_us.h5").write_bytes(b"candidate")
    baseline_dataset = artifact_dir / "baseline" / "enhanced_cps_2024.h5"
    baseline_dataset.parent.mkdir()
    baseline_dataset.write_bytes(b"baseline")
    output_dir = tmp_path / "gate-inputs"

    exit_code = main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    printed_path = Path(capsys.readouterr().out.strip())

    assert exit_code == 0
    assert printed_path == output_dir / "gate_inputs.json"
    assert (output_dir / "artifact.tar.gz").exists()


def test_packaged_inputs_run_gates_from_clean_extract(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_manifest(
        artifact_dir,
        candidate_path="../candidate.h5",
        baseline_path="../baseline.h5",
    )
    _write_minimal_policyengine_dataset(tmp_path / "candidate.h5")
    _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark.json"
    _write_benchmark_manifest(benchmark_manifest)
    arch_coverage = tmp_path / "arch_coverage.json"
    arch_coverage.write_text(json.dumps(_arch_coverage_payload()))
    output_dir = tmp_path / "gate-inputs"

    package_mp300k_gate_inputs(
        artifact_dir,
        output_dir,
        arch_coverage_path=arch_coverage,
        benchmark_manifest_path=benchmark_manifest,
    )

    packaged_manifest = _archive_manifest(output_dir / "artifact.tar.gz")
    assert packaged_manifest["artifacts"]["policyengine_dataset"] == "candidate.h5"
    assert (
        packaged_manifest["config"]["policyengine_baseline_dataset"]
        == "baseline/baseline.h5"
    )
    extract_root = tmp_path / "extract"
    with tarfile.open(output_dir / "artifact.tar.gz") as archive:
        archive.extractall(extract_root, filter="data")
    packaged_artifact_dir = next(
        path.parent for path in extract_root.rglob("manifest.json")
    )

    report_path = write_mp300k_artifact_gate_report(
        packaged_artifact_dir,
        ecps_comparison_payload=_sound_ecps_comparison_payload(),
        arch_coverage_payload=json.loads((output_dir / "arch_coverage.json").read_text()),
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=output_dir / "benchmark_manifest.json",
        compute_native_scores=False,
        update_manifest=False,
    )

    report = json.loads(report_path.read_text())

    assert report["summary"]["status"] == "passed"
    assert report["candidate_dataset"]["path"].startswith(str(packaged_artifact_dir))
    assert report["baseline_dataset"]["path"].startswith(str(packaged_artifact_dir))
    assert report["gates"]["artifact_size"]["status"] == "pass"
