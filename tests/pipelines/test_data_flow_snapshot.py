"""Tests for the canonical US data-flow snapshot."""

import json

from microplex_us.pipelines.data_flow_snapshot import (
    build_us_microplex_data_flow_snapshot,
    write_us_microplex_data_flow_snapshot,
)


def test_build_us_microplex_data_flow_snapshot_reads_manifest_runtime_mix(tmp_path):
    artifact_dir = tmp_path / "run-1"
    artifact_dir.mkdir()
    (artifact_dir / "policyengine_us.h5").write_text("dataset")
    (artifact_dir / "policyengine_harness.json").write_text("{}")
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-05T00:00:00+00:00",
                "config": {
                    "n_synthetic": 5000,
                    "policyengine_direct_override_variables": [],
                },
                "rows": {
                    "seed": 3000,
                    "synthetic": 5000,
                    "calibrated": 5000,
                },
                "synthesis": {
                    "backend": "synthesizer",
                    "source_names": ["cps_asec_2023", "irs_soi_puf"],
                    "scaffold_source": "cps_asec_2023",
                    "condition_vars": ["age", "state_fips"],
                    "target_vars": ["income", "employment_income"],
                    "donor_integrated_variables": [
                        "employment_income",
                        "qualified_dividend_income",
                        "non_qualified_dividend_income",
                    ],
                    "donor_authoritative_override_variables": ["employment_income"],
                    "state_program_support_proxies": {
                        "available": ["ssi"],
                        "missing": ["snap"],
                    },
                },
                "calibration": {
                    "backend": "ipf",
                    "n_loaded_targets": 100,
                    "n_supported_targets": 90,
                    "converged": True,
                },
                "artifacts": {
                    "policyengine_dataset": "policyengine_us.h5",
                    "policyengine_harness": "policyengine_harness.json",
                },
                "policyengine_harness": {
                    "mean_abs_relative_error_delta": -0.2,
                    "target_win_rate": 0.4,
                },
            }
        )
    )

    snapshot = build_us_microplex_data_flow_snapshot(artifact_dir)

    assert snapshot["schemaVersion"] == 1
    assert snapshot["coverageMode"] == "artifact_frozen"
    assert snapshot["runtime"]["scaffoldSource"] == "cps_asec_2023"
    assert snapshot["runtime"]["nSynthetic"] == 5000
    assert snapshot["sharedCoverage"]["sourceNames"] == [
        "cps_asec_2023",
        "irs_soi_puf",
    ]
    assert snapshot["sources"][0]["name"] == "cps_asec_2023"
    assert snapshot["sources"][1]["manifestBacked"] is True
    assert any(
        block["restoreFrame"] == "restore_dividend_components_from_composition"
        for block in snapshot["donorBlocks"]
    )
    assert any(
        highlight["variableName"] == "employment_income"
        and highlight["hasDonorTransform"] is True
        for highlight in snapshot["semanticHighlights"]
    )
    assert [stage["id"] for stage in snapshot["stages"]] == [
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
    assert snapshot["stages"][8]["status"] == "ready"


def test_build_us_microplex_data_flow_snapshot_resolves_cps_parquet_source_exactly(
    tmp_path,
):
    artifact_dir = tmp_path / "run-2"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-05T00:00:00+00:00",
                "config": {"n_synthetic": 1000},
                "rows": {"seed": 1000, "synthetic": 1000, "calibrated": 1000},
                "synthesis": {
                    "backend": "seed",
                    "source_names": ["cps_asec_parquet"],
                    "scaffold_source": "cps_asec_parquet",
                    "condition_vars": [],
                    "target_vars": [],
                    "donor_integrated_variables": [],
                    "state_program_support_proxies": {"available": [], "missing": []},
                },
                "calibration": {},
                "artifacts": {},
            }
        )
    )

    snapshot = build_us_microplex_data_flow_snapshot(artifact_dir)

    assert snapshot["sources"][0]["name"] == "cps_asec_parquet"
    assert "split household/person parquet files" in snapshot["sources"][0]["notes"][0]


def test_build_us_microplex_data_flow_snapshot_prefers_saved_sidecar_but_can_refresh(
    tmp_path,
):
    artifact_dir = tmp_path / "run-3"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-05T00:00:00+00:00",
                "config": {"n_synthetic": 1000},
                "rows": {"seed": 1000, "synthetic": 1000, "calibrated": 1000},
                "synthesis": {
                    "backend": "seed",
                    "source_names": ["cps_asec_parquet"],
                    "scaffold_source": "cps_asec_parquet",
                    "condition_vars": [],
                    "target_vars": [],
                    "donor_integrated_variables": [],
                    "state_program_support_proxies": {"available": [], "missing": []},
                },
                "calibration": {},
                "artifacts": {},
            }
        )
    )
    (artifact_dir / "data_flow_snapshot.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "generatedAt": "2000-01-01T00:00:00Z",
                "coverageMode": "stale",
                "runtime": {"scaffoldSource": "stale_source"},
            }
        )
    )

    saved_snapshot = build_us_microplex_data_flow_snapshot(artifact_dir)
    fresh_snapshot = build_us_microplex_data_flow_snapshot(
        artifact_dir,
        prefer_saved=False,
    )

    assert saved_snapshot["coverageMode"] == "stale"
    assert saved_snapshot["runtime"]["scaffoldSource"] == "stale_source"
    assert fresh_snapshot["coverageMode"] == "artifact_frozen"
    assert fresh_snapshot["runtime"]["scaffoldSource"] == "cps_asec_parquet"


def test_write_us_microplex_data_flow_snapshot_ignores_stale_stage_manifest(
    tmp_path,
):
    artifact_dir = tmp_path / "run-4"
    artifact_dir.mkdir()
    (artifact_dir / "policyengine_us.h5").write_text("dataset")
    (artifact_dir / "stage_manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "contractVersion": "stale",
                "generatedAt": "2000-01-01T00:00:00Z",
                "pipeline": "us_microplex",
                "artifactRoot": ".",
                "manifest": "manifest.json",
                "stages": [{"id": "stale_stage"}],
            }
        )
    )
    manifest = {
        "created_at": "2026-04-05T00:00:00+00:00",
        "config": {"n_synthetic": 1000},
        "rows": {"seed": 1000, "synthetic": 1000, "calibrated": 1000},
        "synthesis": {
            "backend": "seed",
            "source_names": ["cps_asec_parquet"],
            "scaffold_source": "cps_asec_parquet",
            "condition_vars": [],
            "target_vars": [],
            "donor_integrated_variables": [],
            "state_program_support_proxies": {"available": [], "missing": []},
        },
        "calibration": {},
        "artifacts": {
            "policyengine_dataset": "policyengine_us.h5",
            "stage_manifest": "stage_manifest.json",
            "data_flow_snapshot": "data_flow_snapshot.json",
        },
    }

    write_us_microplex_data_flow_snapshot(
        artifact_dir,
        artifact_dir / "data_flow_snapshot.json",
        manifest_payload=manifest,
        assume_existing_stage_artifact_keys=("stage_manifest",),
    )

    snapshot = json.loads((artifact_dir / "data_flow_snapshot.json").read_text())
    assert snapshot["stages"][0]["id"] == "01_run_profile"
    assert snapshot["stages"][7]["status"] == "ready"
