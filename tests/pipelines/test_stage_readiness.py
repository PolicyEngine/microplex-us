"""Tests for US conditional-readiness reports."""

import json

import pytest

from microplex_us.pipelines.stage_artifacts import build_us_stage_artifact_inventory
from microplex_us.pipelines.stage_readiness import (
    build_us_conditional_readiness_report,
    build_us_stage_reuse_key,
    load_us_conditional_readiness_report,
    write_us_conditional_readiness_report,
)


def test_build_us_stage_reuse_key_ignores_checkpoint_output_paths(tmp_path):
    (tmp_path / "synthetic_data.parquet").write_text("synthetic")
    base_manifest = {
        "config": {
            "n_synthetic": 10,
            "calibration_backend": "none",
            "pipeline_checkpoint_save_post_microsim_path": "/tmp/a",
        },
        "rows": {"synthetic": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"synthetic_data": "synthetic_data.parquet"},
    }
    changed_output_path_manifest = {
        **base_manifest,
        "config": {
            **base_manifest["config"],
            "pipeline_checkpoint_save_post_microsim_path": "/tmp/b",
        },
    }

    inventory = build_us_stage_artifact_inventory(
        tmp_path,
        manifest_payload=base_manifest,
        max_hash_bytes=None,
    )

    assert build_us_stage_reuse_key(
        "05_donor_integration_synthesis",
        base_manifest,
        inventory,
    ) == build_us_stage_reuse_key(
        "05_donor_integration_synthesis",
        changed_output_path_manifest,
        inventory,
    )


def test_build_us_stage_reuse_key_uses_stage_scoped_config(tmp_path):
    (tmp_path / "synthetic_data.parquet").write_text("synthetic")
    base_manifest = {
        "config": {
            "n_synthetic": 10,
            "synthesis_backend": "bootstrap",
            "policyengine_dataset_year": 2024,
        },
        "rows": {"synthetic": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"synthetic_data": "synthetic_data.parquet"},
    }
    changed_stage8_config = {
        **base_manifest,
        "config": {
            **base_manifest["config"],
            "policyengine_dataset_year": 2025,
        },
    }
    changed_stage5_config = {
        **base_manifest,
        "config": {
            **base_manifest["config"],
            "n_synthetic": 20,
        },
    }
    inventory = build_us_stage_artifact_inventory(
        tmp_path,
        manifest_payload=base_manifest,
        max_hash_bytes=None,
    )

    base_key = build_us_stage_reuse_key(
        "05_donor_integration_synthesis",
        base_manifest,
        inventory,
    )
    assert base_key == build_us_stage_reuse_key(
        "05_donor_integration_synthesis",
        changed_stage8_config,
        inventory,
    )
    assert base_key != build_us_stage_reuse_key(
        "05_donor_integration_synthesis",
        changed_stage5_config,
        inventory,
    )


def test_conditional_readiness_reports_config_mismatch_as_rerun(tmp_path):
    (tmp_path / "synthetic_data.parquet").write_text("synthetic")
    manifest = {
        "config": {"n_synthetic": 10, "calibration_backend": "none"},
        "rows": {"synthetic": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"synthetic_data": "synthetic_data.parquet"},
    }

    report = build_us_conditional_readiness_report(
        tmp_path,
        manifest_payload=manifest,
        requested_config={"n_synthetic": 20, "calibration_backend": "none"},
    )

    stages = {stage["stageId"]: stage for stage in report["stages"]}
    assert stages["05_donor_integration_synthesis"]["compatibility"] == "mismatch"
    assert stages["05_donor_integration_synthesis"]["readiness"] == "must_rerun"
    assert stages["05_donor_integration_synthesis"]["reason"] == (
        "Requested configuration does not match this stage's saved run inputs."
    )
    assert stages["08_dataset_assembly"]["compatibility"] == "match"


def test_conditional_readiness_reports_manual_replay_without_requested_config(tmp_path):
    (tmp_path / "synthetic_data.parquet").write_text("synthetic")
    manifest = {
        "config": {"n_synthetic": 10, "calibration_backend": "none"},
        "rows": {"synthetic": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"synthetic_data": "synthetic_data.parquet"},
    }

    report = build_us_conditional_readiness_report(
        tmp_path,
        manifest_payload=manifest,
    )

    stages = {stage["stageId"]: stage for stage in report["stages"]}
    assert stages["05_donor_integration_synthesis"]["compatibility"] == (
        "not_evaluated"
    )
    assert stages["05_donor_integration_synthesis"]["readiness"] == "manual_replay"
    assert stages["05_donor_integration_synthesis"]["reloadableArtifacts"] == [
        "05_donor_integration_synthesis.synthetic_data"
    ]


def test_conditional_readiness_reports_missing_required_artifacts_as_rerun(tmp_path):
    manifest = {
        "config": {"n_synthetic": 10, "calibration_backend": "none"},
        "rows": {"synthetic": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"synthetic_data": "synthetic_data.parquet"},
    }

    report = build_us_conditional_readiness_report(
        tmp_path,
        manifest_payload=manifest,
    )

    stages = {stage["stageId"]: stage for stage in report["stages"]}
    assert stages["05_donor_integration_synthesis"]["readiness"] == "must_rerun"
    assert "05_donor_integration_synthesis.synthetic_data" in stages[
        "05_donor_integration_synthesis"
    ]["missingArtifacts"]


def test_conditional_readiness_reports_stage9_from_stage8_dataset(tmp_path):
    (tmp_path / "policyengine_us.h5").write_text("dataset")
    manifest = {
        "config": {"calibration_backend": "none"},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"policyengine_dataset": "policyengine_us.h5"},
    }

    report = build_us_conditional_readiness_report(
        tmp_path,
        manifest_payload=manifest,
    )

    stages = {stage["stageId"]: stage for stage in report["stages"]}
    assert stages["09_validation_benchmarking"]["status"] == "deferred"
    assert stages["09_validation_benchmarking"]["readiness"] == (
        "post_artifact_evidence"
    )


def test_write_and_load_us_conditional_readiness_report(tmp_path):
    (tmp_path / "policyengine_us.h5").write_text("dataset")
    manifest = {
        "config": {"calibration_backend": "none"},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"policyengine_dataset": "policyengine_us.h5"},
    }

    path = write_us_conditional_readiness_report(
        tmp_path,
        tmp_path / "stage_artifacts" / "conditional_readiness.json",
        manifest_payload=manifest,
    )
    loaded = load_us_conditional_readiness_report(path)

    assert loaded["schemaVersion"] == 1
    assert loaded["generatedAt"] is None
    assert loaded["stages"][0]["stageId"] == "01_run_profile"


def test_load_us_conditional_readiness_report_rejects_unknown_schema(tmp_path):
    path = tmp_path / "conditional_readiness.json"
    path.write_text(json.dumps({"schemaVersion": 99}))

    with pytest.raises(RuntimeError, match="Unsupported US conditional-readiness"):
        load_us_conditional_readiness_report(path)
