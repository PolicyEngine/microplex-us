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
    }
    return write_policyengine_us_time_period_dataset(arrays, path)


def _write_manifest(artifact_dir: Path) -> None:
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-05-27T00:00:00+00:00",
                "config": {
                    "policyengine_dataset_year": 2024,
                    "policyengine_baseline_dataset": "baseline/enhanced_cps_2024.h5",
                },
                "artifacts": {"policyengine_dataset": "policyengine_us.h5"},
            }
        )
    )


def _archive_manifest(archive_path: Path) -> dict:
    with tarfile.open(archive_path) as archive:
        manifest = archive.extractfile("artifact/manifest.json")
        assert manifest is not None
        return json.loads(manifest.read())


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
    runtime_smoke = tmp_path / "runtime.json"
    runtime_smoke.write_text(json.dumps({"runtime_ratio": 1.0}))
    benchmark_manifest = tmp_path / "benchmark.json"
    benchmark_manifest.write_text(json.dumps({"schema_version": 1}))

    metadata = package_mp300k_gate_inputs(
        artifact_dir,
        tmp_path / "gate-inputs",
        candidate_dataset_path=external_candidate,
        ecps_comparison_path=ecps_comparison,
        runtime_smoke_path=runtime_smoke,
        benchmark_manifest_path=benchmark_manifest,
    )

    output_dir = tmp_path / "gate-inputs"
    archive_path = output_dir / "artifact.tar.gz"
    manifest = _archive_manifest(archive_path)

    assert archive_path.exists()
    assert (output_dir / "ecps_comparison.json").exists()
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
    _write_manifest(artifact_dir)
    external_candidate = _write_minimal_policyengine_dataset(tmp_path / "candidate.h5")
    external_baseline = _write_minimal_policyengine_dataset(tmp_path / "baseline.h5")
    benchmark_manifest = tmp_path / "benchmark.json"
    benchmark_manifest.write_text(json.dumps({"schema_version": 1}))
    output_dir = tmp_path / "gate-inputs"

    package_mp300k_gate_inputs(
        artifact_dir,
        output_dir,
        candidate_dataset_path=external_candidate,
        baseline_dataset_path=external_baseline,
        benchmark_manifest_path=benchmark_manifest,
    )

    extract_root = tmp_path / "extract"
    with tarfile.open(output_dir / "artifact.tar.gz") as archive:
        archive.extractall(extract_root, filter="data")
    packaged_artifact_dir = next(
        path.parent for path in extract_root.rglob("manifest.json")
    )

    report_path = write_mp300k_artifact_gate_report(
        packaged_artifact_dir,
        ecps_comparison_payload={
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.1,
                "baseline_enhanced_cps_native_loss": 0.2,
            }
        },
        runtime_smoke_payload={"runtime_ratio": 1.0},
        benchmark_manifest_path=output_dir / "benchmark_manifest.json",
        compute_native_scores=False,
        update_manifest=False,
    )

    report = json.loads(report_path.read_text())

    assert report["summary"]["status"] == "passed"
    assert report["baseline_dataset"]["path"].startswith(str(packaged_artifact_dir))
    assert report["gates"]["artifact_size"]["status"] == "pass"
