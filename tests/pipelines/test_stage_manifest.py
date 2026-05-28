"""Tests for US stage manifests and reusable stage artifacts."""

import json

import pandas as pd
import pytest

from microplex_us.pipelines.stage_manifest import (
    build_us_stage_manifest,
    load_us_policyengine_entity_stage_artifact,
    resolve_us_stage_artifact_path,
    stage_summary_for_data_flow_snapshot,
    write_us_policyengine_entity_stage_artifact,
    write_us_stage_manifest,
)
from microplex_us.policyengine import PolicyEngineUSEntityTableBundle


def test_build_us_stage_manifest_reports_nine_stage_statuses(tmp_path):
    (tmp_path / "seed_data.parquet").write_text("seed")
    (tmp_path / "synthetic_data.parquet").write_text("synthetic")
    (tmp_path / "calibrated_data.parquet").write_text("calibrated")
    (tmp_path / "targets.json").write_text("{}")
    (tmp_path / "policyengine_us.h5").write_text("dataset")
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
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "calibrated_data": "calibrated_data.parquet",
            "targets": "targets.json",
            "policyengine_dataset": "policyengine_us.h5",
        },
    }

    payload = build_us_stage_manifest(tmp_path, manifest_payload=manifest)

    assert payload["schemaVersion"] == 1
    assert payload["generatedAt"] == "2026-05-28T00:00:00+00:00"
    assert [stage["id"] for stage in payload["stages"]] == [
        "01_run_profile",
        "02_source_loading",
        "03_source_planning",
        "04_seed_and_donors",
        "05_synthesis",
        "06_policyengine_entities",
        "07_calibration",
        "08_dataset_assembly",
        "09_validation_benchmarking",
    ]
    statuses = {stage["id"]: stage["status"] for stage in payload["stages"]}
    assert statuses["08_dataset_assembly"] == "ready"
    assert statuses["09_validation_benchmarking"] == "deferred"


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
    assert stage_summary_for_data_flow_snapshot(loaded)[7]["id"] == "08_dataset_assembly"


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

    assert metadata["stageId"] == "06_policyengine_entities"
    assert metadata["stage"] == "post_microsim"
    pd.testing.assert_frame_equal(loaded.households, bundle.households)
    pd.testing.assert_frame_equal(loaded.persons, bundle.persons)
    assert loaded.tax_units is None
