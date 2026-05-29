"""Tests for pipeline artifact persistence."""

import json
import sqlite3
from pathlib import Path

import h5py
import pandas as pd
from microplex.core import EntityType
from microplex.targets import StaticTargetProvider, TargetQuery, TargetSet, TargetSpec

from microplex_us.pipelines.artifacts import (
    replay_us_microplex_policyengine_stage_from_artifact,
    save_us_microplex_artifacts,
)
from microplex_us.pipelines.registry import load_us_microplex_run_registry
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexTargets,
)
from microplex_us.policyengine import (
    PolicyEngineUSEntityTableBundle,
    PolicyEngineUSHarnessSlice,
    build_policyengine_us_time_period_arrays,
    compute_policyengine_us_definition_hash,
    write_policyengine_us_time_period_dataset,
)


def test_replay_policyengine_stage_from_artifact_uses_saved_synthetic(
    tmp_path,
    monkeypatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    config = USMicroplexBuildConfig(
        policyengine_targets_db=str(tmp_path / "policy_data.db"),
        calibration_backend="entropy",
    )
    seed_data = pd.DataFrame(
        {
            "person_id": [1],
            "household_id": [10],
            "weight": [10.0],
        }
    )
    scaffold_seed_data = pd.DataFrame(
        {
            "person_id": [1],
            "household_id": [10],
            "weight": [5.0],
        }
    )
    synthetic_data = pd.DataFrame(
        {
            "person_id": [2],
            "household_id": [20],
            "weight": [20.0],
        }
    )
    stale_calibrated_data = pd.DataFrame(
        {
            "person_id": [3],
            "household_id": [30],
            "weight": [999.0],
        }
    )
    scaffold_seed_path = (
        artifact_dir
        / "stage_artifacts"
        / "04_seed_scaffold"
        / "scaffold_seed_data.parquet"
    )
    scaffold_seed_path.parent.mkdir(parents=True)
    scaffold_seed_data.to_parquet(scaffold_seed_path, index=False)
    seed_data.to_parquet(artifact_dir / "seed_data.parquet", index=False)
    synthetic_data.to_parquet(artifact_dir / "synthetic_data.parquet", index=False)
    stale_calibrated_data.to_parquet(
        artifact_dir / "calibrated_data.parquet",
        index=False,
    )
    (artifact_dir / "targets.json").write_text(
        json.dumps({"marginal": {}, "continuous": {}})
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "artifacts": {
                    "scaffold_seed_data": (
                        "stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet"
                    ),
                    "seed_data": "seed_data.parquet",
                    "synthetic_data": "synthetic_data.parquet",
                    "calibrated_data": "calibrated_data.parquet",
                    "targets": "targets.json",
                },
                "synthesis": {"source_names": ["test_source"]},
            }
        )
    )

    captured: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config

        def build_policyengine_entity_tables(self, frame):
            captured["table_input"] = frame.copy()
            return "synthetic_tables"

        def calibrate_policyengine_tables(self, tables):
            captured["tables"] = tables
            calibrated = captured["table_input"].copy()
            calibrated["weight"] = calibrated["weight"] * 2.0
            return "policyengine_tables", calibrated, {"backend": "policyengine_db_none"}

    monkeypatch.setattr(
        "microplex_us.pipelines.artifacts.USMicroplexPipeline",
        FakePipeline,
    )

    result = replay_us_microplex_policyengine_stage_from_artifact(
        artifact_dir,
        config_overrides={"calibration_backend": "none"},
    )

    assert captured["config"].calibration_backend == "none"
    assert captured["tables"] == "synthetic_tables"
    pd.testing.assert_frame_equal(captured["table_input"], synthetic_data)
    pd.testing.assert_frame_equal(result.seed_data, seed_data)
    pd.testing.assert_frame_equal(result.synthetic_data, synthetic_data)
    assert result.calibrated_data["person_id"].tolist() == [2]
    assert result.calibrated_data["weight"].tolist() == [40.0]
    assert result.policyengine_tables == "policyengine_tables"
    pd.testing.assert_frame_equal(result.scaffold_seed_data, scaffold_seed_data)
    assert result.calibration_summary == {"backend": "policyengine_db_none"}
    assert result.synthesis_metadata["policyengine_stage_replay"][
        "config_override_keys"
    ] == ["calibration_backend"]


def _write_baseline_dataset(
    path: Path,
    tables: PolicyEngineUSEntityTableBundle,
) -> Path:
    arrays = build_policyengine_us_time_period_arrays(
        tables,
        period=2024,
        household_variable_map={"state_fips": "state_fips", "snap": "snap"},
        person_variable_map={"age": "age", "income": "employment_income"},
        tax_unit_variable_map={"filing_status": "filing_status"},
    )
    write_policyengine_us_time_period_dataset(arrays, path)
    return path


def _create_policyengine_targets_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE strata (
            stratum_id INTEGER PRIMARY KEY,
            definition_hash TEXT,
            parent_stratum_id INTEGER
        );

        CREATE TABLE stratum_constraints (
            stratum_id INTEGER NOT NULL,
            constraint_variable TEXT NOT NULL,
            operation TEXT NOT NULL,
            value TEXT NOT NULL
        );

        CREATE TABLE targets (
            target_id INTEGER PRIMARY KEY,
            variable TEXT NOT NULL,
            period INTEGER NOT NULL,
            stratum_id INTEGER NOT NULL,
            reform_id INTEGER NOT NULL DEFAULT 0,
            value REAL,
            active BOOLEAN NOT NULL DEFAULT 1,
            tolerance REAL,
            source TEXT,
            notes TEXT
        );

        CREATE VIEW target_overview AS
        SELECT
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.period,
            t.active,
            'state' AS geo_level,
            '06' AS geographic_id,
            'household_count' AS domain_variable
        FROM targets AS t;
        """
    )
    conn.execute(
        """
        INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
        VALUES (?, ?, NULL)
        """,
        (1, compute_policyengine_us_definition_hash(())),
    )
    conn.executemany(
        """
        INSERT INTO targets (
            target_id,
            variable,
            period,
            stratum_id,
            reform_id,
            value,
            active,
            tolerance,
            source,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, "household_count", 2024, 1, 0, 3.0, 1, None, "test", "count"),
        ],
    )
    conn.commit()
    conn.close()


class TestSaveUSMicroplexArtifacts:
    """Test saving pipeline artifacts."""

    def test_writes_expected_files(self, tmp_path):
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            scaffold_seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [1.0, 1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [0.5, 1.5]}),
            targets=USMicroplexTargets(
                marginal={"state": {"CA": 2.0}},
                continuous={"income": 30.0},
            ),
            calibration_summary={"max_error": 0.01, "mean_error": 0.005},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {"household_id": [1, 2], "household_weight": [0.5, 1.5]}
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [10, 11],
                        "household_id": [1, 2],
                        "tax_unit_id": [101, 102],
                        "spm_unit_id": [201, 202],
                        "family_id": [301, 302],
                        "marital_unit_id": [401, 402],
                        "age": [35, 62],
                        "taxable_interest_income": [125.0, 250.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["SINGLE", "JOINT"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )

        paths = save_us_microplex_artifacts(result, tmp_path)

        assert paths.output_dir == tmp_path
        assert paths.scaffold_seed_data is not None
        assert paths.scaffold_seed_data.exists()
        assert paths.seed_data.exists()
        assert paths.synthetic_data.exists()
        assert paths.calibrated_data.exists()
        assert paths.targets.exists()
        assert paths.manifest.exists()
        assert paths.synthesizer is None
        assert paths.policyengine_dataset is not None
        assert paths.policyengine_dataset.exists()
        assert paths.stage_manifest is not None
        assert paths.stage_manifest.exists()
        assert paths.source_plan is not None
        assert paths.source_plan.exists()
        assert paths.policyengine_entity_tables is not None
        assert paths.policyengine_entity_tables.exists()
        assert paths.calibration_summary is not None
        assert paths.calibration_summary.exists()
        assert paths.validation_evidence is not None
        assert paths.validation_evidence.exists()

        manifest = json.loads(paths.manifest.read_text())
        assert manifest["rows"]["synthetic"] == 2
        assert manifest["weights"]["nonzero"] == 2
        assert manifest["config"]["synthesis_backend"] == "bootstrap"
        assert (
            manifest["artifacts"]["scaffold_seed_data"]
            == "stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet"
        )
        assert manifest["artifacts"]["policyengine_dataset"] == "policyengine_us.h5"
        assert manifest["artifacts"]["stage_manifest"] == "stage_manifest.json"
        assert (
            manifest["artifacts"]["policyengine_entity_tables"]
            == "stage_artifacts/06_policyengine_entities/metadata.json"
        )

        with h5py.File(paths.policyengine_dataset, "r") as handle:
            assert "household_id" in handle
            assert "person_household_id" in handle
            assert "tax_unit_id" in handle
            assert "taxable_interest_income" in handle
            assert "filing_status" in handle

    def test_writes_model_when_present(self, tmp_path):
        class FakeSynthesizer:
            def __init__(self):
                self.saved_path = None

            def save(self, path):
                self.saved_path = path
                path.write_text("model")

        fake = FakeSynthesizer()
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=1,
                synthesis_backend="synthesizer",
                calibration_backend="entropy",
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0], "weight": [1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0], "weight": [1.0]}),
            targets=USMicroplexTargets(marginal={}, continuous={"income": 10.0}),
            calibration_summary={"max_error": 0.0, "mean_error": 0.0},
            synthesis_metadata={"backend": "synthesizer"},
            synthesizer=fake,
            policyengine_tables=None,
        )

        paths = save_us_microplex_artifacts(result, tmp_path)

        assert paths.synthesizer is not None
        assert paths.synthesizer.exists()
        assert fake.saved_path == paths.synthesizer

    def test_writes_data_flow_snapshot_before_manifest_validation(self, tmp_path):
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=1,
                synthesis_backend="seed",
                calibration_backend="entropy",
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0], "weight": [1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0], "weight": [1.0]}),
            targets=USMicroplexTargets(marginal={}, continuous={"income": 10.0}),
            calibration_summary={"max_error": 0.0, "mean_error": 0.0},
            synthesis_metadata={
                "backend": "seed",
                "source_names": ["cps_asec_parquet"],
                "scaffold_source": "cps_asec_parquet",
                "condition_vars": [],
                "target_vars": [],
                "donor_integrated_variables": [],
                "state_program_support_proxies": {
                    "available": [],
                    "missing": [],
                },
            },
            synthesizer=None,
            policyengine_tables=None,
        )

        paths = save_us_microplex_artifacts(result, tmp_path)

        assert paths.data_flow_snapshot is not None
        assert paths.data_flow_snapshot.exists()
        assert paths.stage_manifest is not None
        assert paths.stage_manifest.exists()
        manifest = json.loads(paths.manifest.read_text())
        assert manifest["artifacts"]["data_flow_snapshot"] == "data_flow_snapshot.json"
        assert manifest["artifacts"]["stage_manifest"] == "stage_manifest.json"
        snapshot = json.loads(paths.data_flow_snapshot.read_text())
        assert snapshot["runtime"]["scaffoldSource"] == "cps_asec_parquet"
        assert len(snapshot["stages"]) == 9

    def test_writes_child_tax_unit_agi_drift_summary(self, tmp_path):
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="seed",
                calibration_backend="entropy",
                policyengine_dataset_year=2024,
            ),
            seed_data=pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "tax_unit_id": [10, 10],
                    "age": [40.0, 10.0],
                    "is_tax_unit_dependent": [0, 1],
                    "employment_income": [30_000.0, 0.0],
                    "wage_income": [28_000.0, 0.0],
                    "taxable_interest_income": [100.0, 0.0],
                }
            ),
            synthetic_data=pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "tax_unit_id": [10, 10],
                    "age": [40.0, 10.0],
                    "is_tax_unit_dependent": [0, 1],
                    "employment_income": [30_000.0, 0.0],
                    "wage_income": [28_000.0, 0.0],
                    "taxable_interest_income": [100.0, 0.0],
                    "weight": [1.0, 1.0],
                }
            ),
            calibrated_data=pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "tax_unit_id": [10, 10],
                    "age": [40.0, 10.0],
                    "is_tax_unit_dependent": [0, 1],
                    "employment_income": [30_000.0, 0.0],
                    "wage_income": [28_000.0, 0.0],
                    "taxable_interest_income": [100.0, 0.0],
                    "weight": [1.0, 1.0],
                }
            ),
            targets=USMicroplexTargets(marginal={}, continuous={"income": 10.0}),
            calibration_summary={"max_error": 0.01, "mean_error": 0.005},
            synthesis_metadata={"backend": "seed"},
            synthesizer=None,
            policyengine_tables=PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {
                        "household_id": [1],
                        "household_weight": [2.0],
                        "state_fips": [6],
                        "snap": [100.0],
                    }
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [1, 2],
                        "household_id": [1, 1],
                        "tax_unit_id": [10, 10],
                        "spm_unit_id": [20, 20],
                        "family_id": [30, 30],
                        "marital_unit_id": [40, 40],
                        "age": [40.0, 10.0],
                        "income": [30_000.0, 0.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [10],
                        "household_id": [1],
                        "filing_status": ["JOINT"],
                    }
                ),
                spm_units=pd.DataFrame({"spm_unit_id": [20], "household_id": [1]}),
                families=pd.DataFrame({"family_id": [30], "household_id": [1]}),
                marital_units=pd.DataFrame({"marital_unit_id": [40], "household_id": [1]}),
            ),
        )
        baseline_dataset = _write_baseline_dataset(
            tmp_path / "baseline.h5",
            result.policyengine_tables,
        )
        provider = StaticTargetProvider(
            TargetSet(
                [
                    TargetSpec(
                        name="snap_total",
                        entity=EntityType.HOUSEHOLD,
                        value=100.0,
                        period=2024,
                        measure="snap",
                        aggregation="sum",
                    ),
                ]
            )
        )

        paths = save_us_microplex_artifacts(
            result,
            tmp_path / "bundle",
            policyengine_target_provider=provider,
            policyengine_baseline_dataset=baseline_dataset,
            policyengine_harness_slices=(
                PolicyEngineUSHarnessSlice(
                    name="snap",
                    description="SNAP parity",
                    query=TargetQuery(period=2024, names=("snap_total",)),
                ),
            ),
            policyengine_harness_metadata={"baseline_dataset": baseline_dataset.name},
            enable_child_tax_unit_agi_drift=True,
        )

        assert paths.child_tax_unit_agi_drift is not None
        assert paths.child_tax_unit_agi_drift.exists()
        manifest = json.loads(paths.manifest.read_text())
        assert (
            manifest["artifacts"]["child_tax_unit_agi_drift"]
            == "child_tax_unit_agi_drift.json"
        )
        assert "child_tax_unit_agi_drift" in manifest.get("diagnostics", {})
        registry_entries = load_us_microplex_run_registry(
            paths.run_registry or tmp_path / "run_registry.jsonl"
        )
        assert registry_entries[-1].metadata.get("child_tax_unit_agi_drift") is not None

    def test_writes_policyengine_harness_when_baseline_and_targets_are_provided(
        self, tmp_path
    ):
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                policyengine_dataset_year=2024,
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [1.0, 1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [0.5, 1.5]}),
            targets=USMicroplexTargets(
                marginal={"state": {"CA": 2.0}},
                continuous={"income": 30.0},
            ),
            calibration_summary={"max_error": 0.01, "mean_error": 0.005},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "household_weight": [2.0, 1.0],
                        "state_fips": [6, 36],
                        "snap": [100.0, 50.0],
                    }
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [10, 11, 20],
                        "household_id": [1, 1, 2],
                        "tax_unit_id": [101, 101, 102],
                        "spm_unit_id": [201, 201, 202],
                        "family_id": [301, 301, 302],
                        "marital_unit_id": [401, 401, 402],
                        "age": [40.0, 10.0, 30.0],
                        "income": [30_000.0, 0.0, 20_000.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["JOINT", "SINGLE"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )
        baseline_dataset = _write_baseline_dataset(
            tmp_path / "baseline.h5",
            PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "household_weight": [1.0, 1.0],
                        "state_fips": [6, 36],
                        "snap": [75.0, 50.0],
                    }
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [10, 11, 20],
                        "household_id": [1, 1, 2],
                        "tax_unit_id": [101, 101, 102],
                        "spm_unit_id": [201, 201, 202],
                        "family_id": [301, 301, 302],
                        "marital_unit_id": [401, 401, 402],
                        "age": [40.0, 10.0, 30.0],
                        "income": [30_000.0, 0.0, 20_000.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["JOINT", "SINGLE"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )
        provider = StaticTargetProvider(
            TargetSet(
                [
                    TargetSpec(
                        name="household_count",
                        entity=EntityType.HOUSEHOLD,
                        value=3.0,
                        period=2024,
                        aggregation="count",
                    ),
                ]
            )
        )

        paths = save_us_microplex_artifacts(
            result,
            tmp_path / "bundle",
            policyengine_target_provider=provider,
            policyengine_baseline_dataset=baseline_dataset,
            policyengine_harness_slices=(
                PolicyEngineUSHarnessSlice(
                    name="household_count",
                    description="Household count parity",
                    query=TargetQuery(period=2024, names=("household_count",)),
                ),
            ),
            policyengine_harness_metadata={"baseline_dataset": baseline_dataset.name},
        )

        assert paths.policyengine_harness is not None
        assert paths.policyengine_harness.exists()

        manifest = json.loads(paths.manifest.read_text())
        assert manifest["artifacts"]["policyengine_harness"] == "policyengine_harness.json"
        assert manifest["policyengine_harness"]["slice_win_rate"] == 1.0
        assert manifest["policyengine_harness"]["target_win_rate"] == 1.0
        assert manifest["policyengine_harness"]["candidate_composite_parity_loss"] is not None

        harness_payload = json.loads(paths.policyengine_harness.read_text())
        assert harness_payload["metadata"]["baseline_dataset"] == "baseline.h5"
        assert harness_payload["metadata"]["policyengine_us_runtime_version"] is not None
        assert harness_payload["summary"]["slice_win_rate"] == 1.0
        assert harness_payload["summary"]["candidate_composite_parity_loss"] is not None

    def test_can_defer_policyengine_harness_generation(self, monkeypatch, tmp_path):
        baseline_dataset = _write_baseline_dataset(
            tmp_path / "baseline.h5",
            PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "household_weight": [1.0, 1.0],
                        "state_fips": [6, 36],
                        "snap": [75.0, 50.0],
                    }
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [10, 11, 20],
                        "household_id": [1, 1, 2],
                        "tax_unit_id": [101, 101, 102],
                        "spm_unit_id": [201, 201, 202],
                        "family_id": [301, 301, 302],
                        "marital_unit_id": [401, 401, 402],
                        "age": [40.0, 10.0, 30.0],
                        "income": [30_000.0, 0.0, 20_000.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["JOINT", "SINGLE"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                policyengine_dataset_year=2024,
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [1.0, 1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [0.5, 1.5]}),
            targets=USMicroplexTargets(
                marginal={"state": {"CA": 2.0}},
                continuous={"income": 30.0},
            ),
            calibration_summary={"max_error": 0.01, "mean_error": 0.005},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "household_weight": [2.0, 1.0],
                        "state_fips": [6, 36],
                        "snap": [100.0, 50.0],
                    }
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [10, 11, 20],
                        "household_id": [1, 1, 2],
                        "tax_unit_id": [101, 101, 102],
                        "spm_unit_id": [201, 201, 202],
                        "family_id": [301, 301, 302],
                        "marital_unit_id": [401, 401, 402],
                        "age": [40.0, 10.0, 30.0],
                        "income": [30_000.0, 0.0, 20_000.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["JOINT", "SINGLE"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )
        provider = StaticTargetProvider(
            TargetSet(
                [
                    TargetSpec(
                        name="snap_total",
                        entity=EntityType.HOUSEHOLD,
                        value=250.0,
                        period=2024,
                        measure="snap",
                        aggregation="sum",
                    ),
                ]
            )
        )

        monkeypatch.setattr(
            "microplex_us.pipelines.artifacts.evaluate_policyengine_us_harness",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("harness evaluation should be deferred")
            ),
        )

        paths = save_us_microplex_artifacts(
            result,
            tmp_path / "bundle",
            policyengine_target_provider=provider,
            policyengine_baseline_dataset=baseline_dataset,
            policyengine_harness_slices=(
                PolicyEngineUSHarnessSlice(
                    name="snap",
                    description="SNAP parity",
                    query=TargetQuery(period=2024, names=("snap_total",)),
                ),
            ),
            defer_policyengine_harness=True,
            defer_policyengine_native_score=True,
        )

        manifest = json.loads(paths.manifest.read_text())
        assert paths.policyengine_harness is None
        assert manifest["artifacts"]["policyengine_harness"] is None
        assert "policyengine_harness" not in manifest

    def test_writes_policyengine_harness_from_build_config_defaults(self, tmp_path):
        targets_db = tmp_path / "policyengine_targets.db"
        _create_policyengine_targets_db(targets_db)
        baseline_dataset = _write_baseline_dataset(
            tmp_path / "baseline.h5",
            PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "household_weight": [1.0, 1.0],
                        "state_fips": [6, 36],
                        "snap": [75.0, 50.0],
                    }
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [10, 11, 20],
                        "household_id": [1, 1, 2],
                        "tax_unit_id": [101, 101, 102],
                        "spm_unit_id": [201, 201, 202],
                        "family_id": [301, 301, 302],
                        "marital_unit_id": [401, 401, 402],
                        "age": [40.0, 10.0, 30.0],
                        "income": [30_000.0, 0.0, 20_000.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["JOINT", "SINGLE"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                policyengine_dataset_year=2024,
                policyengine_targets_db=str(targets_db),
                policyengine_baseline_dataset=str(baseline_dataset),
                policyengine_target_variables=("household_count",),
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [1.0, 1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [0.5, 1.5]}),
            targets=USMicroplexTargets(
                marginal={"state": {"CA": 2.0}},
                continuous={"income": 30.0},
            ),
            calibration_summary={"max_error": 0.01, "mean_error": 0.005},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "household_weight": [2.0, 1.0],
                        "state_fips": [6, 36],
                        "snap": [100.0, 50.0],
                    }
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [10, 11, 20],
                        "household_id": [1, 1, 2],
                        "tax_unit_id": [101, 101, 102],
                        "spm_unit_id": [201, 201, 202],
                        "family_id": [301, 301, 302],
                        "marital_unit_id": [401, 401, 402],
                        "age": [40.0, 10.0, 30.0],
                        "income": [30_000.0, 0.0, 20_000.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["JOINT", "SINGLE"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )

        paths = save_us_microplex_artifacts(
            result,
            tmp_path / "bundle",
            defer_policyengine_native_score=True,
        )

        assert paths.policyengine_harness is not None
        assert paths.policyengine_harness.exists()
        assert paths.run_registry is not None
        assert paths.run_registry.exists()

        manifest = json.loads(paths.manifest.read_text())
        assert manifest["policyengine_harness"]["slice_win_rate"] == 1.0
        assert manifest["policyengine_harness"]["target_win_rate"] == 1.0
        assert manifest["policyengine_harness"]["candidate_composite_parity_loss"] is not None
        assert manifest["policyengine_harness"]["parity_scorecard"]["overall"][
            "candidate_beats_baseline"
        ] is True
        assert manifest["run_registry"]["artifact_id"] == "bundle"
        assert manifest["run_registry"]["improved_candidate_frontier"] is True
        assert manifest["run_registry"]["improved_composite_frontier"] is True
        assert (
            manifest["run_registry"]["default_frontier_metric"]
            == "candidate_composite_parity_loss"
        )

        harness_payload = json.loads(paths.policyengine_harness.read_text())
        assert harness_payload["metadata"]["baseline_dataset"] == "baseline.h5"
        assert harness_payload["metadata"]["targets_db"] == "policyengine_targets.db"
        assert harness_payload["metadata"]["harness_suite"] == "policyengine_us_all_targets"
        assert harness_payload["metadata"]["harness_slice_names"] == ["all_targets"]
        assert harness_payload["metadata"]["target_variables"] == ["household_count"]
        assert harness_payload["metadata"]["policyengine_us_runtime_version"] is not None
        assert [slice_payload["name"] for slice_payload in harness_payload["slices"]] == [
            "all_targets",
        ]
        registry_entries = load_us_microplex_run_registry(paths.run_registry)
        assert len(registry_entries) == 1
        assert registry_entries[0].artifact_id == "bundle"
        assert registry_entries[0].policyengine_us_runtime_version is not None
        assert registry_entries[0].supported_target_rate == 1.0
        assert registry_entries[0].candidate_composite_parity_loss is not None
        assert registry_entries[0].tag_summaries["all_targets"]["target_win_rate"] == 1.0

    def test_writes_policyengine_native_scores_when_available(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(
            "microplex_us.pipelines.artifacts.compute_us_pe_native_scores",
            lambda **_kwargs: {
                "metric": "enhanced_cps_native_loss",
                "summary": {
                    "candidate_enhanced_cps_native_loss": 0.25,
                    "baseline_enhanced_cps_native_loss": 0.5,
                    "enhanced_cps_native_loss_delta": -0.25,
                    "candidate_beats_baseline": True,
                    "candidate_unweighted_msre": 0.3,
                    "baseline_unweighted_msre": 0.6,
                    "unweighted_msre_delta": -0.3,
                    "n_targets_total": 2863,
                    "n_targets_kept": 2853,
                    "n_targets_zero_dropped": 10,
                    "n_targets_bad_dropped": 10,
                    "n_national_targets": 2000,
                    "n_state_targets": 853,
                },
            },
        )

        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                policyengine_dataset_year=2024,
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [1.0, 1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [0.5, 1.5]}),
            targets=USMicroplexTargets(marginal={}, continuous={"income": 30.0}),
            calibration_summary={"max_error": 0.01, "mean_error": 0.005},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "household_weight": [0.5, 1.5],
                        "state_fips": [6, 48],
                        "snap": [0.0, 50.0],
                    }
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [10, 11],
                        "household_id": [1, 2],
                        "tax_unit_id": [101, 102],
                        "spm_unit_id": [201, 202],
                        "family_id": [301, 302],
                        "marital_unit_id": [401, 402],
                        "age": [35.0, 62.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["SINGLE", "JOINT"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )

        baseline_dataset = tmp_path / "baseline.h5"
        baseline_dataset.write_text("baseline")

        paths = save_us_microplex_artifacts(
            result,
            tmp_path / "bundle-native",
            policyengine_baseline_dataset=baseline_dataset,
        )

        assert paths.policyengine_native_scores is not None
        assert paths.policyengine_native_scores.exists()

        manifest = json.loads(paths.manifest.read_text())
        assert (
            manifest["artifacts"]["policyengine_native_scores"]
            == "policyengine_native_scores.json"
        )
        assert paths.run_registry is not None
        assert paths.run_registry.exists()
        assert (
            manifest["policyengine_native_scores"]["candidate_enhanced_cps_native_loss"]
            == 0.25
        )
        assert manifest["policyengine_native_scores"]["candidate_beats_baseline"] is True
        assert (
            manifest["run_registry"]["default_frontier_metric"]
            == "enhanced_cps_native_loss_delta"
        )

        registry_entries = load_us_microplex_run_registry(paths.run_registry)
        assert len(registry_entries) == 1
        assert registry_entries[0].candidate_beats_baseline_native_loss is True

    def test_uses_precomputed_policyengine_native_scores_without_recomputing(
        self, monkeypatch, tmp_path
    ) -> None:
        def _boom(**_kwargs):
            raise AssertionError("native scorer should not be called")

        monkeypatch.setattr(
            "microplex_us.pipelines.artifacts.compute_us_pe_native_scores",
            _boom,
        )

        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                policyengine_dataset_year=2024,
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [1.0, 1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [0.5, 1.5]}),
            targets=USMicroplexTargets(marginal={}, continuous={"income": 30.0}),
            calibration_summary={"max_error": 0.01, "mean_error": 0.005},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "household_weight": [0.5, 1.5],
                        "state_fips": [6, 48],
                        "snap": [0.0, 50.0],
                    }
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [10, 11],
                        "household_id": [1, 2],
                        "tax_unit_id": [101, 102],
                        "spm_unit_id": [201, 202],
                        "family_id": [301, 302],
                        "marital_unit_id": [401, 402],
                        "age": [35.0, 62.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["SINGLE", "JOINT"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )

        payload = {
            "metric": "enhanced_cps_native_loss",
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.25,
                "baseline_enhanced_cps_native_loss": 0.5,
                "enhanced_cps_native_loss_delta": -0.25,
                "candidate_beats_baseline": True,
                "candidate_unweighted_msre": 0.3,
                "baseline_unweighted_msre": 0.6,
                "unweighted_msre_delta": -0.3,
                "n_targets_total": 2863,
                "n_targets_kept": 2853,
                "n_targets_zero_dropped": 10,
                "n_targets_bad_dropped": 10,
                "n_national_targets": 2000,
                "n_state_targets": 853,
            },
        }

        paths = save_us_microplex_artifacts(
            result,
            tmp_path / "bundle-native-precomputed",
            precomputed_policyengine_native_scores=payload,
        )

        assert paths.policyengine_native_scores is not None
        assert json.loads(paths.policyengine_native_scores.read_text()) == payload
