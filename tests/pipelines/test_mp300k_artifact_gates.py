"""Tests for mp-300k artifact quality gates."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

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
    }
    return write_policyengine_us_time_period_dataset(arrays, path)


def _write_incomplete_policyengine_dataset(path: Path, *, period: int = 2024) -> Path:
    with h5py.File(path, "w") as handle:
        household_id = handle.create_group("household_id")
        household_id.create_dataset(str(period), data=[1, 2])
        household_weight = handle.create_group("household_weight")
        household_weight.create_dataset(str(period), data=[10.0, 20.0])
        person_id = handle.create_group("person_id")
        person_id.create_dataset(str(period), data=[1, 2, 3])
    return path


def _write_artifact_manifest(
    artifact_dir: Path,
    *,
    baseline_dataset: Path | None = None,
) -> None:
    manifest = {
        "created_at": "2026-05-27T00:00:00+00:00",
        "config": {
            "policyengine_baseline_dataset": str(baseline_dataset)
            if baseline_dataset is not None
            else None,
            "policyengine_dataset_year": 2024,
        },
        "artifacts": {"policyengine_dataset": "candidate.h5"},
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))


def test_write_mp300k_artifact_gate_report_passes_with_all_evidence(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    candidate_dataset = _write_minimal_policyengine_dataset(
        artifact_dir / "candidate.h5"
    )
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    benchmark_manifest.write_text(json.dumps({"schema_version": 1, "frozen": True}))
    _write_artifact_manifest(artifact_dir, baseline_dataset=baseline_dataset)

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        ecps_comparison_payload={
            "metric": "enhanced_cps_native_loss",
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.12,
                "baseline_enhanced_cps_native_loss": 0.20,
                "enhanced_cps_native_loss_delta": -0.08,
                "candidate_beats_baseline": True,
                "n_targets_kept": 150,
            },
        },
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
    assert record["gates"]["artifact_size"]["status"] == "pass"
    assert record["gates"]["ecps_comparison"]["status"] == "pass"
    assert record["gates"]["runtime"]["status"] == "pass"
    assert record["gates"]["runtime"]["metrics"]["runtime_ratio"] == 1.1
    assert record["gates"]["benchmark_manifest"]["status"] == "pass"
    assert record["candidate_dataset"]["path"] == str(candidate_dataset.resolve())
    assert (
        manifest["artifacts"]["mp300k_artifact_gates"] == "mp300k_artifact_gates.json"
    )
    assert manifest["mp300k_artifact_gates"]["status"] == "passed"


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
        json.dumps(
            {
                "summary": {
                    "candidate_enhanced_cps_native_loss": 0.10,
                    "baseline_enhanced_cps_native_loss": 0.20,
                    "enhanced_cps_native_loss_delta": -0.10,
                }
            }
        )
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps({"runtime_ratio": 1.2, "runtime_ratio_threshold": 1.25})
    )
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    benchmark_manifest.write_text(json.dumps({"schema_version": 1}))

    exit_code = main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--ecps-comparison-json",
            str(ecps_comparison_path),
            "--runtime-smoke-json",
            str(runtime_path),
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
    benchmark_manifest.write_text(json.dumps({"schema_version": 1}))

    report_path = write_mp300k_artifact_gate_report(
        artifact_dir,
        runtime_smoke_payload={
            "runtime_ratio": 1.0,
            "runtime_ratio_threshold": 1.25,
        },
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


def test_ecps_comparison_accepts_existing_broad_loss_array_payload(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_minimal_policyengine_dataset(artifact_dir / "candidate.h5")
    baseline_dataset = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    benchmark_manifest.write_text(json.dumps({"schema_version": 1}))
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
