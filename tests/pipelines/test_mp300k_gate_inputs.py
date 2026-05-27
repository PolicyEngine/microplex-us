"""Tests for packaging mp-300k gate inputs."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from microplex_us.pipelines.mp300k_gate_inputs import (
    main,
    package_mp300k_gate_inputs,
)


def _write_manifest(artifact_dir: Path) -> None:
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-05-27T00:00:00+00:00",
                "config": {"policyengine_dataset_year": 2024},
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
    assert manifest["mp300k_gate_inputs"]["source_candidate_dataset"] == str(
        external_candidate.resolve()
    )
    assert metadata["artifact_archive"]["path"] == str(archive_path.resolve())
    assert metadata["workflow_call"]["with"]["gate_inputs_artifact"] == "gate-inputs"


def test_main_packages_gate_inputs(tmp_path, capsys):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_manifest(artifact_dir)
    (artifact_dir / "policyengine_us.h5").write_bytes(b"candidate")
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
