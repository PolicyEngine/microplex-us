"""Tests for US stage artifact inventory helpers."""

import json

import pandas as pd
import pytest

from microplex_us.pipelines.stage_artifacts import (
    build_us_stage_artifact_inventory,
    load_us_calibrated_stage_artifacts,
    load_us_candidate_calibration_replay_artifacts,
    load_us_candidate_stage_artifacts,
    load_us_dataset_assembly_artifacts,
    load_us_policyengine_entity_stage_artifacts,
    load_us_seed_scaffold_stage_artifacts,
    load_us_stage_artifact_inventory,
    load_us_stage_json_artifact,
    resolve_us_stage_artifact_from_inventory,
    resolve_us_stage_artifact_path_checked,
    write_us_stage_artifact_inventory,
)
from microplex_us.pipelines.stage_manifest import (
    write_us_policyengine_entity_stage_artifact,
)
from microplex_us.policyengine import PolicyEngineUSEntityTableBundle


def test_build_us_stage_artifact_inventory_hashes_files_and_directories(tmp_path):
    (tmp_path / "seed_data.parquet").write_text("seed")
    (tmp_path / "synthetic_data.parquet").write_text("synthetic")
    source_plan = (
        tmp_path / "stage_artifacts" / "03_source_planning" / "source_plan.json"
    )
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
            "pre_calibration_policyengine_entity_tables": (
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
        (record["stageId"], record["key"]): record for record in inventory["artifacts"]
    }
    assert (
        records[("05_donor_integration_synthesis", "synthetic_data")]["classification"]
        == "manual_replay"
    )
    assert (
        records[("05_donor_integration_synthesis", "synthetic_data")]["hashStatus"]
        == "hashed"
    )
    assert records[("05_donor_integration_synthesis", "synthetic_data")]["contentHash"]
    assert records[("03_source_planning", "source_plan")]["classification"] == (
        "diagnostic_only"
    )
    entity_record = records[
        ("06_policyengine_entities", "pre_calibration_policyengine_entity_tables")
    ]
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
        (record["stageId"], record["key"]): record for record in inventory["artifacts"]
    }
    assert (
        records[("05_donor_integration_synthesis", "synthetic_data")]["classification"]
        == "missing_required"
    )
    assert (
        records[("05_donor_integration_synthesis", "synthesizer")]["classification"]
        == "contract_only"
    )


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
        record for record in inventory["artifacts"] if record["key"] == "synthetic_data"
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


def test_load_us_candidate_stage_artifacts_reads_stage5_boundary(tmp_path):
    pytest.importorskip("pyarrow")
    seed = pd.DataFrame({"person_id": [1], "income": [20]})
    synthetic = pd.DataFrame({"person_id": [1, 2], "income": [20, 30]})
    seed.to_parquet(tmp_path / "seed_data.parquet", index=False)
    synthetic.to_parquet(tmp_path / "synthetic_data.parquet", index=False)
    manifest = {
        "config": {"calibration_backend": "none"},
        "rows": {"seed": 1, "synthetic": 2},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
        },
    }

    loaded = load_us_candidate_stage_artifacts(tmp_path, manifest_payload=manifest)

    pd.testing.assert_frame_equal(loaded.seed_data, seed)
    pd.testing.assert_frame_equal(loaded.synthetic_data, synthetic)
    assert (
        loaded.artifact_paths["synthetic_data"] == tmp_path / "synthetic_data.parquet"
    )


def test_load_us_seed_scaffold_stage_artifacts_reads_stage4_boundary(tmp_path):
    pytest.importorskip("pyarrow")
    scaffold = pd.DataFrame({"person_id": [1], "income": [10]})
    scaffold_path = (
        tmp_path / "stage_artifacts" / "04_seed_scaffold" / "scaffold_seed_data.parquet"
    )
    scaffold_path.parent.mkdir(parents=True)
    scaffold.to_parquet(scaffold_path, index=False)
    manifest = {
        "config": {"calibration_backend": "none"},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {
            "scaffold_seed_data": (
                "stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet"
            ),
        },
    }

    loaded = load_us_seed_scaffold_stage_artifacts(tmp_path, manifest_payload=manifest)

    pd.testing.assert_frame_equal(loaded.scaffold_seed_data, scaffold)
    assert loaded.artifact_paths["scaffold_seed_data"] == scaffold_path


def test_load_us_candidate_calibration_replay_artifacts_combines_boundaries(
    tmp_path,
):
    pytest.importorskip("pyarrow")
    scaffold = pd.DataFrame({"person_id": [1], "income": [10]})
    seed = pd.DataFrame({"person_id": [1], "income": [20]})
    synthetic = pd.DataFrame({"person_id": [1, 2], "income": [20, 30]})
    scaffold_path = (
        tmp_path / "stage_artifacts" / "04_seed_scaffold" / "scaffold_seed_data.parquet"
    )
    scaffold_path.parent.mkdir(parents=True)
    scaffold.to_parquet(scaffold_path, index=False)
    seed.to_parquet(tmp_path / "seed_data.parquet", index=False)
    synthetic.to_parquet(tmp_path / "synthetic_data.parquet", index=False)
    (tmp_path / "targets.json").write_text(
        json.dumps({"marginal": {"age": {"20": 1.0}}, "continuous": {"income": 1.0}})
    )
    manifest = {
        "config": {"calibration_backend": "none"},
        "rows": {"seed": 1, "synthetic": 2},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {
            "scaffold_seed_data": (
                "stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet"
            ),
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "targets": "targets.json",
        },
    }

    loaded = load_us_candidate_calibration_replay_artifacts(
        tmp_path,
        manifest_payload=manifest,
    )

    pd.testing.assert_frame_equal(loaded.candidate.synthetic_data, synthetic)
    assert loaded.seed_scaffold is not None
    pd.testing.assert_frame_equal(loaded.seed_scaffold.scaffold_seed_data, scaffold)
    assert loaded.targets.continuous == {"income": 1.0}
    assert loaded.artifact_paths["targets"] == tmp_path / "targets.json"


def test_load_us_policyengine_entity_stage_artifacts_reads_checkpoint(tmp_path):
    pytest.importorskip("pyarrow")
    bundle = PolicyEngineUSEntityTableBundle(
        households=pd.DataFrame({"household_id": [1], "household_weight": [1.0]}),
        persons=pd.DataFrame({"person_id": [10], "household_id": [1]}),
        tax_units=None,
        spm_units=None,
        families=None,
        marital_units=None,
    )
    write_us_policyengine_entity_stage_artifact(bundle, tmp_path)
    manifest = {
        "config": {"calibration_backend": "none"},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {
            "pre_calibration_policyengine_entity_tables": (
                "stage_artifacts/06_policyengine_entities/metadata.json"
            ),
        },
    }

    loaded = load_us_policyengine_entity_stage_artifacts(
        tmp_path,
        manifest_payload=manifest,
    )

    assert loaded.metadata["stageId"] == "06_policyengine_entities"
    pd.testing.assert_frame_equal(loaded.bundle.households, bundle.households)


def test_load_us_calibrated_stage_artifacts_reads_stage7_outputs(tmp_path):
    pytest.importorskip("pyarrow")
    calibrated = pd.DataFrame({"person_id": [1], "weight": [2.0]})
    calibrated.to_parquet(tmp_path / "calibrated_data.parquet", index=False)
    (tmp_path / "targets.json").write_text(
        json.dumps({"marginal": {}, "continuous": {"income": 1.0}})
    )
    summary_path = tmp_path / "stage_artifacts" / "07_calibration"
    summary_path.mkdir(parents=True)
    (summary_path / "calibration_summary.json").write_text(
        json.dumps({"backend": "none", "converged": True})
    )
    manifest = {
        "config": {"calibration_backend": "none"},
        "rows": {"calibrated": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {"backend": "none"},
        "artifacts": {
            "calibrated_data": "calibrated_data.parquet",
            "targets": "targets.json",
            "calibration_summary": (
                "stage_artifacts/07_calibration/calibration_summary.json"
            ),
        },
    }

    loaded = load_us_calibrated_stage_artifacts(tmp_path, manifest_payload=manifest)

    pd.testing.assert_frame_equal(loaded.calibrated_data, calibrated)
    assert loaded.targets.continuous == {"income": 1.0}
    assert loaded.calibration_summary["converged"] is True


def test_load_us_dataset_assembly_artifacts_resolves_stage8_paths(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "stage_manifest.json").write_text("{}")
    (tmp_path / "data_flow_snapshot.json").write_text("{}")
    (tmp_path / "policyengine_us.h5").write_text("dataset")
    stage_artifacts = tmp_path / "stage_artifacts"
    stage_artifacts.mkdir()
    (stage_artifacts / "artifact_inventory.json").write_text("{}")
    (stage_artifacts / "conditional_readiness.json").write_text("{}")
    manifest = {
        "config": {"calibration_backend": "none"},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {
            "policyengine_dataset": "policyengine_us.h5",
            "stage_manifest": "stage_manifest.json",
            "data_flow_snapshot": "data_flow_snapshot.json",
            "artifact_inventory": "stage_artifacts/artifact_inventory.json",
            "conditional_readiness": "stage_artifacts/conditional_readiness.json",
        },
    }

    loaded = load_us_dataset_assembly_artifacts(tmp_path, manifest_payload=manifest)

    assert loaded.policyengine_dataset == tmp_path / "policyengine_us.h5"
    assert loaded.stage_manifest == tmp_path / "stage_manifest.json"
    assert loaded.data_flow_snapshot == tmp_path / "data_flow_snapshot.json"
    assert loaded.artifact_inventory == stage_artifacts / "artifact_inventory.json"
    assert (
        loaded.conditional_readiness == stage_artifacts / "conditional_readiness.json"
    )


def test_stage_artifact_checked_resolver_enforces_format_and_existence(tmp_path):
    (tmp_path / "synthetic_data.parquet").write_text("synthetic")
    manifest = {
        "config": {"calibration_backend": "none"},
        "rows": {"synthetic": 1},
        "synthesis": {"source_names": ["source"], "scaffold_source": "source"},
        "calibration": {},
        "artifacts": {"synthetic_data": "synthetic_data.parquet"},
    }

    with pytest.raises(ValueError, match="expected 'json'"):
        resolve_us_stage_artifact_path_checked(
            tmp_path,
            "05_donor_integration_synthesis",
            "synthetic_data",
            manifest_payload=manifest,
            expected_format="json",
        )

    with pytest.raises(FileNotFoundError, match="Stage artifact not found"):
        load_us_stage_json_artifact(
            tmp_path,
            "03_source_planning",
            "source_plan",
            manifest_payload={
                **manifest,
                "artifacts": {"source_plan": "missing.json"},
            },
        )
