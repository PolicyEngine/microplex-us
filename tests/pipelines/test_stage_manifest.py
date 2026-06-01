"""Tests for US stage manifests and reusable stage artifacts."""

import json

import pandas as pd
import pytest

from microplex_us.pipelines.stage_manifest import (
    build_us_stage_manifest,
    load_us_policyengine_entity_stage_artifact,
    load_us_stage_manifest,
    resolve_us_stage_artifact_path,
    stage_summary_for_data_flow_snapshot,
    write_us_policyengine_entity_stage_artifact,
    write_us_stage_manifest,
)
from microplex_us.policyengine import PolicyEngineUSEntityTableBundle


def test_build_us_stage_manifest_reports_nine_stage_statuses(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    scaffold_seed_path = (
        tmp_path / "stage_artifacts" / "04_seed_scaffold" / "scaffold_seed_data.parquet"
    )
    scaffold_seed_path.parent.mkdir(parents=True)
    scaffold_seed_path.write_text("scaffold")
    (tmp_path / "seed_data.parquet").write_text("seed")
    (tmp_path / "synthetic_data.parquet").write_text("synthetic")
    (tmp_path / "calibrated_data.parquet").write_text("calibrated")
    (tmp_path / "targets.json").write_text("{}")
    (tmp_path / "policyengine_us.h5").write_text("dataset")
    source_plan_path = tmp_path / "stage_artifacts" / "03_source_planning"
    source_plan_path.mkdir(parents=True)
    (source_plan_path / "source_plan.json").write_text("{}")
    entity_path = tmp_path / "stage_artifacts" / "06_policyengine_entities"
    entity_path.mkdir(parents=True)
    (entity_path / "metadata.json").write_text("{}")
    calibration_path = tmp_path / "stage_artifacts" / "07_calibration"
    calibration_path.mkdir(parents=True)
    (calibration_path / "calibration_summary.json").write_text("{}")
    (tmp_path / "stage_manifest.json").write_text("{}")
    (tmp_path / "data_flow_snapshot.json").write_text("{}")
    (tmp_path / "stage_artifacts" / "artifact_inventory.json").write_text("{}")
    (tmp_path / "stage_artifacts" / "conditional_readiness.json").write_text("{}")
    manifest = {
        "created_at": "2026-05-28T00:00:00+00:00",
        "config": {"calibration_backend": "entropy"},
        "rows": {"seed": 1, "synthetic": 1, "calibrated": 1},
        "synthesis": {
            "source_names": ["cps_asec_2023"],
            "scaffold_source": "cps_asec_2023",
            "backend": "seed",
            "donor_integrated_variables": [],
        },
        "calibration": {"backend": "policyengine_db_entropy"},
        "artifacts": {
            "scaffold_seed_data": (
                "stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet"
            ),
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "calibrated_data": "calibrated_data.parquet",
            "targets": "targets.json",
            "source_plan": "stage_artifacts/03_source_planning/source_plan.json",
            "policyengine_entity_tables": (
                "stage_artifacts/06_policyengine_entities/metadata.json"
            ),
            "calibration_summary": (
                "stage_artifacts/07_calibration/calibration_summary.json"
            ),
            "policyengine_dataset": "policyengine_us.h5",
            "stage_manifest": "stage_manifest.json",
            "data_flow_snapshot": "data_flow_snapshot.json",
            "artifact_inventory": "stage_artifacts/artifact_inventory.json",
            "conditional_readiness": "stage_artifacts/conditional_readiness.json",
        },
    }

    payload = build_us_stage_manifest(tmp_path, manifest_payload=manifest)

    assert payload["schemaVersion"] == 3
    assert payload["generatedAt"] == "2026-05-28T00:00:00+00:00"
    assert [stage["id"] for stage in payload["stages"]] == [
        "01_run_profile",
        "02_source_loading",
        "03_source_planning",
        "04_seed_scaffold",
        "05_donor_integration_synthesis",
        "06_policyengine_entities",
        "07_calibration",
        "08_dataset_assembly",
        "09_validation_benchmarking",
    ]
    statuses = {stage["id"]: stage["status"] for stage in payload["stages"]}
    assert statuses["01_run_profile"] == "ready"
    assert statuses["02_source_loading"] == "metadata_only"
    assert statuses["03_source_planning"] == "ready"
    assert statuses["04_seed_scaffold"] == "ready"
    assert statuses["05_donor_integration_synthesis"] == "ready"
    assert statuses["06_policyengine_entities"] == "ready"
    assert statuses["07_calibration"] == "ready"
    assert statuses["08_dataset_assembly"] == "ready"
    assert statuses["09_validation_benchmarking"] == "deferred"
    stage5_artifacts = {
        artifact["key"]: artifact
        for stage in payload["stages"]
        if stage["id"] == "05_donor_integration_synthesis"
        for artifact in stage["artifacts"]
    }
    assert stage5_artifacts["synthetic_data"]["format"] == "parquet_dataframe"
    assert stage5_artifacts["synthetic_data"]["hash_mode"] == "file_sha256"


def test_load_us_stage_manifest_accepts_v1_v2_and_v3(tmp_path):
    v1_path = tmp_path / "stage_manifest_v1.json"
    v1_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "contractVersion": "us-runtime-stages-v1",
                "generatedAt": None,
                "pipeline": "us_microplex",
                "artifactRoot": ".",
                "manifest": "manifest.json",
                "stages": [],
            }
        )
    )
    v2_path = tmp_path / "stage_manifest_v2.json"
    v2_path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "contractVersion": "us-runtime-stages-v2",
                "generatedAt": None,
                "pipeline": "us_microplex",
                "artifactRoot": ".",
                "manifest": "manifest.json",
                "stages": [],
            }
        )
    )
    v3_path = tmp_path / "stage_manifest_v3.json"
    v3_path.write_text(
        json.dumps(
            {
                "schemaVersion": 3,
                "contractVersion": "us-runtime-stages-v2",
                "generatedAt": None,
                "pipeline": "us_microplex",
                "artifactRoot": ".",
                "manifest": "manifest.json",
                "stages": [],
            }
        )
    )

    assert load_us_stage_manifest(v1_path)["schemaVersion"] == 1
    assert load_us_stage_manifest(v2_path)["schemaVersion"] == 2
    assert load_us_stage_manifest(v3_path)["schemaVersion"] == 3


def test_build_us_stage_manifest_keeps_empty_validation_index_deferred(tmp_path):
    (tmp_path / "policyengine_us.h5").write_text("dataset")
    (tmp_path / "stage_manifest.json").write_text("{}")
    (tmp_path / "data_flow_snapshot.json").write_text("{}")
    (tmp_path / "stage_artifacts" / "artifact_inventory.json").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (tmp_path / "stage_artifacts" / "artifact_inventory.json").write_text("{}")
    (tmp_path / "stage_artifacts" / "conditional_readiness.json").write_text("{}")
    evidence_path = (
        tmp_path
        / "stage_artifacts"
        / "09_validation_benchmarking"
        / "evidence_manifest.json"
    )
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(
        json.dumps(
            {
                "formatVersion": 1,
                "stageId": "09_validation_benchmarking",
                "evidence": [],
                "summaries": {},
            }
        )
    )
    manifest = {
        "config": {"calibration_backend": "entropy"},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {
            "policyengine_dataset": "policyengine_us.h5",
            "stage_manifest": "stage_manifest.json",
            "data_flow_snapshot": "data_flow_snapshot.json",
            "artifact_inventory": "stage_artifacts/artifact_inventory.json",
            "conditional_readiness": "stage_artifacts/conditional_readiness.json",
            "validation_evidence": (
                "stage_artifacts/09_validation_benchmarking/evidence_manifest.json"
            ),
        },
    }

    payload = build_us_stage_manifest(tmp_path, manifest_payload=manifest)

    statuses = {stage["id"]: stage["status"] for stage in payload["stages"]}
    assert statuses["09_validation_benchmarking"] == "deferred"


def test_build_us_stage_manifest_requires_validation_evidence_for_stage9_ready(
    tmp_path,
):
    (tmp_path / "policyengine_us.h5").write_text("dataset")
    (tmp_path / "policyengine_native_scores.json").write_text("{}")
    manifest = {
        "config": {"calibration_backend": "entropy"},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {
            "policyengine_dataset": "policyengine_us.h5",
            "policyengine_native_scores": "policyengine_native_scores.json",
        },
    }

    payload = build_us_stage_manifest(tmp_path, manifest_payload=manifest)

    statuses = {stage["id"]: stage["status"] for stage in payload["stages"]}
    assert statuses["09_validation_benchmarking"] == "incomplete"


def test_stage_summary_omits_unreferenced_path_hints(tmp_path):
    manifest = {
        "config": {"calibration_backend": "entropy"},
        "rows": {"seed": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {},
    }

    payload = build_us_stage_manifest(tmp_path, manifest_payload=manifest)
    summaries = stage_summary_for_data_flow_snapshot(payload)

    outputs = {stage["id"]: stage["outputs"] for stage in summaries}
    assert outputs["04_seed_scaffold"] == []
    assert outputs["05_donor_integration_synthesis"] == []


def test_build_us_stage_manifest_reports_incomplete_referenced_artifacts(tmp_path):
    manifest = {
        "created_at": "2026-05-28T00:00:00+00:00",
        "config": {"calibration_backend": "entropy"},
        "rows": {"seed": 1, "synthetic": 1},
        "synthesis": {
            "source_names": ["cps_asec_2023"],
            "scaffold_source": "cps_asec_2023",
            "backend": "seed",
        },
        "artifacts": {
            "scaffold_seed_data": (
                "stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet"
            ),
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "policyengine_harness": "policyengine_harness.json",
        },
    }

    payload = build_us_stage_manifest(tmp_path, manifest_payload=manifest)

    statuses = {stage["id"]: stage["status"] for stage in payload["stages"]}
    assert statuses["04_seed_scaffold"] == "incomplete"
    assert statuses["05_donor_integration_synthesis"] == "incomplete"
    assert statuses["09_validation_benchmarking"] == "incomplete"


def test_write_us_stage_manifest_and_resolve_artifact_path(tmp_path):
    (tmp_path / "policyengine_us.h5").write_text("dataset")
    manifest = {
        "config": {"calibration_backend": "none"},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"policyengine_dataset": "policyengine_us.h5"},
    }

    manifest_path = write_us_stage_manifest(
        tmp_path,
        tmp_path / "stage_manifest.json",
        manifest_payload=manifest,
    )
    loaded = json.loads(manifest_path.read_text())
    dataset_path = resolve_us_stage_artifact_path(
        tmp_path,
        loaded,
        "08_dataset_assembly",
        "policyengine_dataset",
    )

    assert dataset_path == tmp_path / "policyengine_us.h5"
    assert (
        stage_summary_for_data_flow_snapshot(loaded)[7]["id"] == "08_dataset_assembly"
    )


def test_policyengine_entity_stage_artifact_round_trips_partial_bundle(tmp_path):
    pytest.importorskip("pyarrow")

    bundle = PolicyEngineUSEntityTableBundle(
        households=pd.DataFrame(
            {"household_id": [1, 2], "household_weight": [1.0, 2.0]}
        ),
        persons=pd.DataFrame(
            {"person_id": [10, 20], "household_id": [1, 2], "age": [40, 50]}
        ),
        tax_units=None,
        spm_units=None,
        families=None,
        marital_units=None,
    )

    manifest_path = write_us_policyengine_entity_stage_artifact(bundle, tmp_path)
    loaded, metadata = load_us_policyengine_entity_stage_artifact(manifest_path)

    assert manifest_path == (
        tmp_path / "stage_artifacts" / "06_policyengine_entities" / "metadata.json"
    )
    assert metadata["stageId"] == "06_policyengine_entities"
    assert metadata["stage"] == "post_microsim"
    pd.testing.assert_frame_equal(loaded.households, bundle.households)
    pd.testing.assert_frame_equal(loaded.persons, bundle.persons)
    assert loaded.tax_units is None


def test_policyengine_entity_stage_artifact_does_not_replace_run_root(tmp_path):
    pytest.importorskip("pyarrow")

    (tmp_path / "manifest.json").write_text("{}")
    bundle = PolicyEngineUSEntityTableBundle(
        households=pd.DataFrame({"household_id": [1], "household_weight": [1.0]}),
        persons=pd.DataFrame({"person_id": [10], "household_id": [1]}),
        tax_units=None,
        spm_units=None,
        families=None,
        marital_units=None,
    )

    write_us_policyengine_entity_stage_artifact(bundle, tmp_path)

    assert (tmp_path / "manifest.json").exists()
    assert (
        tmp_path / "stage_artifacts" / "06_policyengine_entities" / "metadata.json"
    ).exists()
