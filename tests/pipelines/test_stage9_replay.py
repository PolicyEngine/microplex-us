"""Tests for safe Stage 9 validation replay."""

import json

import pytest

from microplex_us.pipelines.stage9_replay import (
    main,
    replay_us_stage9_validation_benchmarking,
)


def _write_stage8_bundle(tmp_path, *, stage8_status: str = "complete"):
    artifact_dir = tmp_path / "bundle"
    manifest_dir = artifact_dir / "stage_artifacts" / "manifests"
    manifest_dir.mkdir(parents=True)
    dataset_path = artifact_dir / "policyengine_us.h5"
    dataset_path.write_bytes(b"h5-placeholder")
    stage8_manifest = {
        "stageId": "08_dataset_assembly",
        "lifecycleStatus": stage8_status,
        "outputs": {
            "policyengine_dataset": {
                "path": "policyengine_us.h5",
                "exists": True,
            }
        },
    }
    (manifest_dir / "08_dataset_assembly.json").write_text(json.dumps(stage8_manifest))
    manifest = {
        "config": {"policyengine_dataset_year": 2024},
        "artifacts": {"policyengine_dataset": "policyengine_us.h5"},
        "stage_output_manifests": {
            "08_dataset_assembly": (
                "stage_artifacts/manifests/08_dataset_assembly.json"
            )
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    return artifact_dir


def test_stage9_replay_writes_new_evidence_without_mutating_source_bundle(tmp_path):
    artifact_dir = _write_stage8_bundle(tmp_path)
    original_manifest = (artifact_dir / "manifest.json").read_text()

    result = replay_us_stage9_validation_benchmarking(
        artifact_dir,
        run_id="unit-replay",
        precomputed_policyengine_native_scores={
            "summary": {"enhanced_cps_native_loss_delta": -0.1}
        },
    )

    assert result.output_dir == (
        artifact_dir
        / "stage_artifacts"
        / "09_validation_benchmarking"
        / "replays"
        / "unit-replay"
    )
    assert result.validation_evidence.exists()
    assert result.policyengine_native_scores is not None
    assert result.policyengine_native_scores.exists()
    assert (artifact_dir / "manifest.json").read_text() == original_manifest

    evidence = json.loads(result.validation_evidence.read_text())
    assert evidence["stageId"] == "09_validation_benchmarking"
    assert evidence["evidence"][0]["key"] == "policyengine_native_scores"


def test_stage9_replay_rejects_incomplete_stage8(tmp_path):
    artifact_dir = _write_stage8_bundle(tmp_path, stage8_status="running")

    with pytest.raises(ValueError, match="Stage 8 must be complete"):
        replay_us_stage9_validation_benchmarking(
            artifact_dir,
            precomputed_policyengine_native_scores={"summary": {"loss": 1.0}},
        )


def test_stage9_replay_rejects_stage8_dataset_path_mismatch(tmp_path):
    artifact_dir = _write_stage8_bundle(tmp_path)
    stage8_manifest_path = (
        artifact_dir / "stage_artifacts" / "manifests" / "08_dataset_assembly.json"
    )
    stage8_manifest = json.loads(stage8_manifest_path.read_text())
    stage8_manifest["outputs"]["policyengine_dataset"]["path"] = (
        "other/policyengine_us.h5"
    )
    stage8_manifest_path.write_text(json.dumps(stage8_manifest))

    with pytest.raises(ValueError, match="does not match"):
        replay_us_stage9_validation_benchmarking(
            artifact_dir,
            precomputed_policyengine_native_scores={"summary": {"loss": 1.0}},
        )


def test_stage9_replay_cli_smoke(tmp_path, capsys):
    artifact_dir = _write_stage8_bundle(tmp_path)
    payload_path = tmp_path / "native_scores.json"
    payload_path.write_text(json.dumps({"summary": {"loss": 1.0}}))

    assert (
        main(
            [
                str(artifact_dir),
                "--run-id",
                "cli-replay",
                "--precomputed-policyengine-native-scores",
                str(payload_path),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out.strip()
    assert output.endswith("evidence_manifest.json")
