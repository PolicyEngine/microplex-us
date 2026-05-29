"""Tests for US stage artifact inventory helpers."""

import json

import pytest

from microplex_us.pipelines.stage_artifacts import (
    build_us_stage_artifact_inventory,
    load_us_stage_artifact_inventory,
    resolve_us_stage_artifact_from_inventory,
    write_us_stage_artifact_inventory,
)


def test_build_us_stage_artifact_inventory_hashes_files_and_directories(tmp_path):
    (tmp_path / "seed_data.parquet").write_text("seed")
    (tmp_path / "synthetic_data.parquet").write_text("synthetic")
    source_plan = tmp_path / "stage_artifacts" / "03_source_planning" / "source_plan.json"
    source_plan.parent.mkdir(parents=True)
    source_plan.write_text("{}")
    entity_dir = tmp_path / "stage_artifacts" / "06_policyengine_entities"
    entity_dir.mkdir(parents=True)
    (entity_dir / "metadata.json").write_text("{}")
    (entity_dir / "households.parquet").write_text("households")
    manifest = {
        "config": {"calibration_backend": "none"},
        "rows": {"seed": 1, "synthetic": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "source_plan": "stage_artifacts/03_source_planning/source_plan.json",
            "policyengine_entity_tables": (
                "stage_artifacts/06_policyengine_entities/metadata.json"
            ),
        },
    }

    inventory = build_us_stage_artifact_inventory(
        tmp_path,
        manifest_payload=manifest,
        max_hash_bytes=None,
    )

    records = {
        (record["stageId"], record["key"]): record
        for record in inventory["artifacts"]
    }
    assert records[("05_donor_integration_synthesis", "synthetic_data")][
        "classification"
    ] == "manual_replay"
    assert records[("05_donor_integration_synthesis", "synthetic_data")][
        "hashStatus"
    ] == "hashed"
    assert records[("05_donor_integration_synthesis", "synthetic_data")][
        "contentHash"
    ]
    assert records[("03_source_planning", "source_plan")]["classification"] == (
        "diagnostic_only"
    )
    entity_record = records[("06_policyengine_entities", "policyengine_entity_tables")]
    assert entity_record["classification"] == "manual_resume"
    assert entity_record["fileCount"] == 2
    assert entity_record["hashStatus"] == "hashed"


def test_build_us_stage_artifact_inventory_classifies_missing_and_contract_only(
    tmp_path,
):
    manifest = {
        "config": {"calibration_backend": "none"},
        "rows": {"seed": 1, "synthetic": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
        },
    }

    inventory = build_us_stage_artifact_inventory(tmp_path, manifest_payload=manifest)

    records = {
        (record["stageId"], record["key"]): record
        for record in inventory["artifacts"]
    }
    assert records[("05_donor_integration_synthesis", "synthetic_data")][
        "classification"
    ] == "missing_required"
    assert records[("05_donor_integration_synthesis", "synthesizer")][
        "classification"
    ] == "contract_only"


def test_build_us_stage_artifact_inventory_skips_large_file_hashes(tmp_path):
    (tmp_path / "synthetic_data.parquet").write_text("synthetic")
    manifest = {
        "config": {"calibration_backend": "none"},
        "rows": {"synthetic": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"synthetic_data": "synthetic_data.parquet"},
    }

    inventory = build_us_stage_artifact_inventory(
        tmp_path,
        manifest_payload=manifest,
        max_hash_bytes=3,
    )

    record = next(
        record
        for record in inventory["artifacts"]
        if record["key"] == "synthetic_data"
    )
    assert record["hashStatus"] == "too_large"
    assert record["contentHash"] is None


def test_write_load_and_resolve_us_stage_artifact_inventory(tmp_path):
    (tmp_path / "policyengine_us.h5").write_text("dataset")
    manifest = {
        "config": {"calibration_backend": "none"},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"policyengine_dataset": "policyengine_us.h5"},
    }

    path = write_us_stage_artifact_inventory(
        tmp_path,
        tmp_path / "stage_artifacts" / "artifact_inventory.json",
        manifest_payload=manifest,
    )
    loaded = load_us_stage_artifact_inventory(path)
    dataset_path = resolve_us_stage_artifact_from_inventory(
        tmp_path,
        loaded,
        "08_dataset_assembly",
        "policyengine_dataset",
    )

    assert loaded["schemaVersion"] == 1
    assert dataset_path == tmp_path / "policyengine_us.h5"


def test_load_us_stage_artifact_inventory_rejects_unknown_schema(tmp_path):
    path = tmp_path / "artifact_inventory.json"
    path.write_text(json.dumps({"schemaVersion": 99}))

    with pytest.raises(RuntimeError, match="Unsupported US stage artifact inventory"):
        load_us_stage_artifact_inventory(path)
