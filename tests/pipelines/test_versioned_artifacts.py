"""Tests for versioned US microplex artifact saving and frontier lookup."""

import json
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceDescriptor,
    StaticSourceProvider,
    TimeStructure,
)

from microplex_us.pipelines import (
    build_and_save_versioned_us_microplex,
    build_and_save_versioned_us_microplex_from_data_dir,
    build_and_save_versioned_us_microplex_from_source_provider,
    build_and_save_versioned_us_microplex_from_source_providers,
    compare_us_microplex_target_delta_rows,
    list_us_microplex_target_delta_rows,
    rebuild_us_microplex_run_index,
    resolve_us_microplex_frontier_artifact_dir,
    resolve_us_microplex_run_index_path,
    save_versioned_us_microplex_artifacts,
    select_us_microplex_frontier_entry,
    select_us_microplex_frontier_index_row,
)
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexTargets,
)
from microplex_us.policyengine import (
    PolicyEngineUSEntityTableBundle,
    build_policyengine_us_time_period_arrays,
    compute_policyengine_us_definition_hash,
    write_policyengine_us_time_period_dataset,
)


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
            CASE
                WHEN t.variable = 'snap' THEN 'state'
                ELSE 'district'
            END AS geo_level,
            CASE
                WHEN t.variable = 'snap' THEN '06'
                ELSE '0601'
            END AS geographic_id,
            CASE
                WHEN t.variable = 'snap' THEN 'snap'
                WHEN t.variable = 'household_count' THEN 'snap'
                ELSE NULL
            END AS domain_variable
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
            (2, "snap", 2024, 1, 0, 250.0, 1, None, "test", "snap"),
        ],
    )
    conn.commit()
    conn.close()


def _make_result(
    *,
    targets_db: Path,
    baseline_dataset: Path,
    snap_values: tuple[float, float],
) -> USMicroplexBuildResult:
    return USMicroplexBuildResult(
        config=USMicroplexBuildConfig(
            n_synthetic=2,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            policyengine_dataset_year=2024,
            policyengine_targets_db=str(targets_db),
            policyengine_baseline_dataset=str(baseline_dataset),
            policyengine_target_variables=("snap", "household_count"),
        ),
        seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
        synthetic_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [1.0, 1.0]}),
        calibrated_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [0.5, 1.5]}),
        targets=USMicroplexTargets(
            marginal={"state": {"CA": 2.0}},
            continuous={"income": 30.0},
        ),
        calibration_summary={"max_error": 0.01, "mean_error": 0.005},
        synthesis_metadata={"backend": "bootstrap", "source_names": ["cps", "puf"]},
        synthesizer=None,
        policyengine_tables=PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "household_weight": [2.0, 1.0],
                    "state_fips": [6, 36],
                    "snap": list(snap_values),
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
            spm_units=pd.DataFrame({"spm_unit_id": [201, 202], "household_id": [1, 2]}),
            families=pd.DataFrame({"family_id": [301, 302], "household_id": [1, 2]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [401, 402], "household_id": [1, 2]}
            ),
        ),
    )


def test_save_versioned_us_microplex_artifacts_accepts_path_config_values(tmp_path):
    targets_db = tmp_path / "policy_data.db"
    baseline_dataset = tmp_path / "baseline.h5"
    _create_policyengine_targets_db(targets_db)
    _write_baseline_dataset(
        baseline_dataset,
        _make_result(
            targets_db=targets_db,
            baseline_dataset=baseline_dataset,
            snap_values=(100.0, 50.0),
        ).policyengine_tables,
    )

    result = _make_result(
        targets_db=targets_db,
        baseline_dataset=baseline_dataset,
        snap_values=(100.0, 50.0),
    )
    result.config = USMicroplexBuildConfig(
        n_synthetic=2,
        synthesis_backend="bootstrap",
        calibration_backend="entropy",
        policyengine_dataset_year=2024,
        policyengine_targets_db=targets_db,
        policyengine_baseline_dataset=baseline_dataset,
        policyengine_target_variables=("snap", "household_count"),
    )

    artifact_paths = save_versioned_us_microplex_artifacts(
        result, tmp_path / "artifacts"
    )
    manifest = json.loads(artifact_paths.manifest.read_text())

    assert manifest["config"]["policyengine_targets_db"] == str(targets_db)
    assert manifest["config"]["policyengine_baseline_dataset"] == str(baseline_dataset)


def test_save_versioned_us_microplex_artifacts_accepts_stage_runtime_writer_keyword(
    tmp_path,
):
    targets_db = tmp_path / "policy_data.db"
    baseline_dataset = tmp_path / "baseline.h5"
    _create_policyengine_targets_db(targets_db)
    _write_baseline_dataset(
        baseline_dataset,
        _make_result(
            targets_db=targets_db,
            baseline_dataset=baseline_dataset,
            snap_values=(100.0, 50.0),
        ).policyengine_tables,
    )

    artifact_paths = save_versioned_us_microplex_artifacts(
        _make_result(
            targets_db=targets_db,
            baseline_dataset=baseline_dataset,
            snap_values=(100.0, 50.0),
        ),
        tmp_path / "artifacts",
        stage_runtime_writer=None,
    )

    assert artifact_paths.policyengine_dataset.exists()
    assert artifact_paths.manifest.exists()


def _make_source_provider(
    *,
    name: str,
    households: pd.DataFrame,
    persons: pd.DataFrame,
) -> StaticSourceProvider:
    provider_households = households.rename(
        columns={
            "household_id": "hh_id",
            "hh_weight": "household_weight",
        }
    )
    provider_persons = persons.rename(
        columns={
            "person_id": "person_key",
            "household_id": "hh_id",
        }
    )
    frame = ObservationFrame(
        source=SourceDescriptor(
            name=name,
            shareability=Shareability.PUBLIC,
            time_structure=TimeStructure.REPEATED_CROSS_SECTION,
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column="hh_id",
                    weight_column="household_weight",
                    variable_names=tuple(
                        column
                        for column in provider_households.columns
                        if column != "hh_id"
                    ),
                ),
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column="person_key",
                    variable_names=tuple(
                        column
                        for column in provider_persons.columns
                        if column not in {"person_key", "hh_id"}
                    ),
                ),
            ),
        ),
        tables={
            EntityType.HOUSEHOLD: provider_households,
            EntityType.PERSON: provider_persons,
        },
        relationships=(
            EntityRelationship(
                parent_entity=EntityType.HOUSEHOLD,
                child_entity=EntityType.PERSON,
                parent_key="hh_id",
                child_key="hh_id",
                cardinality=RelationshipCardinality.ONE_TO_MANY,
            ),
        ),
    )
    return StaticSourceProvider(frame)


def test_save_versioned_us_microplex_artifacts_uses_explicit_version(tmp_path):
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
            spm_units=pd.DataFrame({"spm_unit_id": [201, 202], "household_id": [1, 2]}),
            families=pd.DataFrame({"family_id": [301, 302], "household_id": [1, 2]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [401, 402], "household_id": [1, 2]}
            ),
        ),
    )
    result = _make_result(
        targets_db=targets_db,
        baseline_dataset=baseline_dataset,
        snap_values=(100.0, 50.0),
    )

    paths = save_versioned_us_microplex_artifacts(
        result,
        tmp_path / "builds",
        version_id="run-1",
    )

    assert paths.version_id == "run-1"
    assert paths.output_dir == tmp_path / "builds" / "run-1"
    assert paths.run_registry == tmp_path / "builds" / "run_registry.jsonl"
    assert paths.run_index_db == tmp_path / "builds" / "run_index.duckdb"
    assert paths.stage_manifest == paths.output_dir / "stage_manifest.json"
    assert paths.artifact_inventory == (
        paths.output_dir / "stage_artifacts" / "artifact_inventory.json"
    )
    assert paths.conditional_readiness == (
        paths.output_dir / "stage_artifacts" / "conditional_readiness.json"
    )
    assert paths.source_plan == (
        paths.output_dir / "stage_artifacts" / "03_source_planning" / "source_plan.json"
    )
    assert paths.policyengine_entity_tables == (
        paths.output_dir
        / "stage_artifacts"
        / "07_calibration"
        / "policyengine_entity_tables"
        / "metadata.json"
    )
    assert paths.calibration_summary == (
        paths.output_dir
        / "stage_artifacts"
        / "07_calibration"
        / "calibration_summary.json"
    )
    assert paths.validation_evidence == (
        paths.output_dir
        / "stage_artifacts"
        / "09_validation_benchmarking"
        / "evidence_manifest.json"
    )
    assert paths.source_weight_diagnostics == (
        paths.output_dir / "source_weight_diagnostics.json"
    )
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["run_registry"]["artifact_id"] == "run-1"
    assert manifest["run_index"]["artifact_id"] == "run-1"

    with duckdb.connect(str(paths.run_index_db), read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM slice_metrics").fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM target_metrics").fetchone()[0] == 2


def test_frontier_helpers_select_best_versioned_run(tmp_path):
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
            spm_units=pd.DataFrame({"spm_unit_id": [201, 202], "household_id": [1, 2]}),
            families=pd.DataFrame({"family_id": [301, 302], "household_id": [1, 2]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [401, 402], "household_id": [1, 2]}
            ),
        ),
    )

    root = tmp_path / "builds"
    better_result = _make_result(
        targets_db=targets_db,
        baseline_dataset=baseline_dataset,
        snap_values=(100.0, 50.0),
    )
    worse_result = _make_result(
        targets_db=targets_db,
        baseline_dataset=baseline_dataset,
        snap_values=(60.0, 50.0),
    )
    save_versioned_us_microplex_artifacts(better_result, root, version_id="run-1")
    save_versioned_us_microplex_artifacts(worse_result, root, version_id="run-2")

    frontier_entry = select_us_microplex_frontier_entry(root)
    frontier_dir = resolve_us_microplex_frontier_artifact_dir(root)

    assert frontier_entry is not None
    assert frontier_entry.artifact_id == "run-1"
    assert frontier_dir == root / "run-1"
    assert frontier_entry.candidate_composite_parity_loss is not None

    indexed_frontier = select_us_microplex_frontier_index_row(root)

    assert indexed_frontier is not None
    assert indexed_frontier["artifact_id"] == "run-1"


def test_build_and_save_versioned_us_microplex_returns_frontier_gap(tmp_path):
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
            spm_units=pd.DataFrame({"spm_unit_id": [201, 202], "household_id": [1, 2]}),
            families=pd.DataFrame({"family_id": [301, 302], "household_id": [1, 2]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [401, 402], "household_id": [1, 2]}
            ),
        ),
    )
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "state_fips": [6, 36, 48],
            "hh_weight": [100.0, 150.0, 200.0],
            "tenure": [1, 2, 1],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [10, 11, 12, 13, 14, 15],
            "household_id": [1, 1, 2, 2, 3, 3],
            "age": [34, 12, 47, 43, 68, 30],
            "sex": [1, 2, 2, 1, 1, 2],
            "education": [3, 1, 4, 4, 2, 4],
            "employment_status": [1, 0, 1, 1, 2, 1],
            "income": [55_000.0, 0.0, 72_000.0, 40_000.0, 18_000.0, 65_000.0],
        }
    )

    saved = build_and_save_versioned_us_microplex(
        persons,
        households,
        tmp_path / "builds",
        version_id="run-build",
        config=USMicroplexBuildConfig(
            n_synthetic=6,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            random_seed=7,
            policyengine_dataset_year=2024,
            policyengine_targets_db=str(targets_db),
            policyengine_baseline_dataset=str(baseline_dataset),
            policyengine_target_variables=("snap", "household_count"),
        ),
    )

    assert saved.artifact_paths.version_id == "run-build"
    assert saved.build_result.policyengine_tables is not None
    assert saved.current_entry is not None
    assert saved.current_entry.artifact_id == "run-build"
    assert saved.frontier_entry is not None
    assert saved.frontier_entry.artifact_id == "run-build"
    assert saved.frontier_delta == 0.0
    assert saved.frontier_entry.candidate_composite_parity_loss is not None


def test_build_and_save_versioned_us_microplex_from_source_provider(tmp_path):
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
            spm_units=pd.DataFrame({"spm_unit_id": [201, 202], "household_id": [1, 2]}),
            families=pd.DataFrame({"family_id": [301, 302], "household_id": [1, 2]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [401, 402], "household_id": [1, 2]}
            ),
        ),
    )
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "state_fips": [6, 36, 48],
            "hh_weight": [100.0, 150.0, 200.0],
            "tenure": [1, 2, 1],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [10, 11, 12, 13, 14, 15],
            "household_id": [1, 1, 2, 2, 3, 3],
            "age": [34, 12, 47, 43, 68, 30],
            "sex": [1, 2, 2, 1, 1, 2],
            "education": [3, 1, 4, 4, 2, 4],
            "employment_status": [1, 0, 1, 1, 2, 1],
            "income": [55_000.0, 0.0, 72_000.0, 40_000.0, 18_000.0, 65_000.0],
        }
    )
    provider = _make_source_provider(
        name="test_cps", households=households, persons=persons
    )

    saved = build_and_save_versioned_us_microplex_from_source_provider(
        provider,
        tmp_path / "provider-builds",
        version_id="provider-run",
        config=USMicroplexBuildConfig(
            n_synthetic=6,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            random_seed=7,
            policyengine_dataset_year=2024,
            policyengine_targets_db=str(targets_db),
            policyengine_baseline_dataset=str(baseline_dataset),
            policyengine_target_variables=("snap", "household_count"),
        ),
    )

    assert saved.artifact_paths.version_id == "provider-run"
    assert saved.build_result.source_frame is not None
    assert saved.build_result.source_frame.source.name == "test_cps"
    assert saved.current_entry is not None
    assert saved.frontier_delta == 0.0


def test_build_and_save_versioned_us_microplex_from_source_providers(tmp_path):
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
            spm_units=pd.DataFrame({"spm_unit_id": [201, 202], "household_id": [1, 2]}),
            families=pd.DataFrame({"family_id": [301, 302], "household_id": [1, 2]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [401, 402], "household_id": [1, 2]}
            ),
        ),
    )
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "state_fips": [6, 36, 48],
            "hh_weight": [100.0, 150.0, 200.0],
            "tenure": [1, 2, 1],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [10, 11, 12, 13, 14, 15],
            "household_id": [1, 1, 2, 2, 3, 3],
            "age": [34, 12, 47, 43, 68, 30],
            "sex": [1, 2, 2, 1, 1, 2],
            "education": [3, 1, 4, 4, 2, 4],
            "employment_status": [1, 0, 1, 1, 2, 1],
            "income": [55_000.0, 0.0, 72_000.0, 40_000.0, 18_000.0, 65_000.0],
        }
    )
    providers = [
        _make_source_provider(name="test_cps", households=households, persons=persons),
        _make_source_provider(name="test_puf", households=households, persons=persons),
    ]

    saved = build_and_save_versioned_us_microplex_from_source_providers(
        providers,
        tmp_path / "multisource-builds",
        version_id="multisource-run",
        config=USMicroplexBuildConfig(
            n_synthetic=6,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            random_seed=7,
            policyengine_dataset_year=2024,
            policyengine_targets_db=str(targets_db),
            policyengine_baseline_dataset=str(baseline_dataset),
            policyengine_target_variables=("snap", "household_count"),
        ),
    )

    assert saved.artifact_paths.version_id == "multisource-run"
    assert saved.build_result.fusion_plan is not None
    assert saved.build_result.fusion_plan.source_names == ("test_cps", "test_puf")
    assert saved.current_entry is not None
    assert saved.frontier_delta == 0.0
    manifest = json.loads(saved.artifact_paths.manifest.read_text())
    stage_output_manifests = manifest["stage_output_manifests"]
    assert tuple(stage_output_manifests) == (
        "01_run_profile",
        "02_source_loading",
        "03_source_planning",
        "04_seed_scaffold",
        "05_donor_integration_synthesis",
        "06_policyengine_entities",
        "07_calibration",
        "08_dataset_assembly",
        "09_validation_benchmarking",
    )
    for stage_id, manifest_path in stage_output_manifests.items():
        stage_manifest = json.loads(
            (saved.artifact_paths.output_dir / manifest_path).read_text()
        )
        assert stage_manifest["lifecycleStatus"] in {"complete", "deferred"}
        assert stage_manifest["events"]


def test_build_and_save_versioned_us_microplex_from_data_dir(tmp_path):
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
            spm_units=pd.DataFrame({"spm_unit_id": [201, 202], "household_id": [1, 2]}),
            families=pd.DataFrame({"family_id": [301, 302], "household_id": [1, 2]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [401, 402], "household_id": [1, 2]}
            ),
        ),
    )
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "state_fips": [6, 36, 48],
            "household_weight": [100.0, 150.0, 200.0],
            "tenure": [1, 2, 1],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [10, 11, 12, 13, 14, 15],
            "household_id": [1, 1, 2, 2, 3, 3],
            "age": [34, 12, 47, 43, 68, 30],
            "sex": [1, 2, 2, 1, 1, 2],
            "education": [3, 1, 4, 4, 2, 4],
            "employment_status": [1, 0, 1, 1, 2, 1],
            "income": [55_000.0, 0.0, 72_000.0, 40_000.0, 18_000.0, 65_000.0],
        }
    )
    data_dir = tmp_path / "cps_parquet"
    data_dir.mkdir()
    households.to_parquet(data_dir / "cps_asec_households.parquet", index=False)
    persons.to_parquet(data_dir / "cps_asec_persons.parquet", index=False)

    saved = build_and_save_versioned_us_microplex_from_data_dir(
        data_dir,
        tmp_path / "data-dir-builds",
        version_id="data-dir-run",
        config=USMicroplexBuildConfig(
            n_synthetic=6,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            random_seed=7,
            policyengine_dataset_year=2024,
            policyengine_targets_db=str(targets_db),
            policyengine_baseline_dataset=str(baseline_dataset),
            policyengine_target_variables=("snap", "household_count"),
        ),
    )

    assert saved.artifact_paths.version_id == "data-dir-run"
    assert saved.build_result.source_frame is not None
    assert saved.build_result.source_frame.source.name == "cps_asec_parquet"
    assert saved.current_entry is not None
    assert saved.frontier_delta == 0.0


def test_run_index_target_deltas_are_queryable(tmp_path):
    targets_db = tmp_path / "policyengine_targets.db"
    _create_policyengine_targets_db(targets_db)
    baseline_seed_result = _make_result(
        targets_db=targets_db,
        baseline_dataset=tmp_path / "baseline.h5",
        snap_values=(75.0, 50.0),
    )
    baseline_dataset = _write_baseline_dataset(
        tmp_path / "baseline.h5",
        baseline_seed_result.policyengine_tables,
    )
    root = tmp_path / "builds"
    save_versioned_us_microplex_artifacts(
        _make_result(
            targets_db=targets_db,
            baseline_dataset=baseline_dataset,
            snap_values=(100.0, 50.0),
        ),
        root,
        version_id="run-1",
    )

    rows = list_us_microplex_target_delta_rows(
        root,
        artifact_id="run-1",
    )

    assert len(rows) == 2
    assert {row["target_value"] for row in rows} == {3.0, 250.0}
    assert {row["domain_variable"] for row in rows} == {"snap"}
    assert all(row["artifact_id"] == "run-1" for row in rows)


def test_run_index_target_deltas_are_comparable_across_runs(tmp_path):
    targets_db = tmp_path / "policyengine_targets.db"
    _create_policyengine_targets_db(targets_db)
    baseline_seed_result = _make_result(
        targets_db=targets_db,
        baseline_dataset=tmp_path / "baseline.h5",
        snap_values=(75.0, 50.0),
    )
    baseline_dataset = _write_baseline_dataset(
        tmp_path / "baseline.h5",
        baseline_seed_result.policyengine_tables,
    )
    root = tmp_path / "builds"
    save_versioned_us_microplex_artifacts(
        _make_result(
            targets_db=targets_db,
            baseline_dataset=baseline_dataset,
            snap_values=(100.0, 50.0),
        ),
        root,
        version_id="run-1",
    )
    save_versioned_us_microplex_artifacts(
        _make_result(
            targets_db=targets_db,
            baseline_dataset=baseline_dataset,
            snap_values=(140.0, 50.0),
        ),
        root,
        version_id="run-2",
    )

    rows = compare_us_microplex_target_delta_rows(
        root,
        artifact_id="run-2",
        baseline_artifact_id="run-1",
    )

    assert len(rows) == 2
    assert {row["artifact_id"] for row in rows} == {"run-2"}
    assert {row["baseline_artifact_id"] for row in rows} == {"run-1"}
    assert {row["domain_variable"] for row in rows} == {"snap"}


def test_run_index_can_be_rebuilt_from_registry(tmp_path):
    targets_db = tmp_path / "policyengine_targets.db"
    _create_policyengine_targets_db(targets_db)
    baseline_seed_result = _make_result(
        targets_db=targets_db,
        baseline_dataset=tmp_path / "baseline.h5",
        snap_values=(75.0, 50.0),
    )
    baseline_dataset = _write_baseline_dataset(
        tmp_path / "baseline.h5",
        baseline_seed_result.policyengine_tables,
    )
    root = tmp_path / "builds"
    save_versioned_us_microplex_artifacts(
        _make_result(
            targets_db=targets_db,
            baseline_dataset=baseline_dataset,
            snap_values=(100.0, 50.0),
        ),
        root,
        version_id="run-1",
    )
    save_versioned_us_microplex_artifacts(
        _make_result(
            targets_db=targets_db,
            baseline_dataset=baseline_dataset,
            snap_values=(60.0, 50.0),
        ),
        root,
        version_id="run-2",
    )

    index_path = resolve_us_microplex_run_index_path(root)
    index_path.unlink()

    rebuilt_path = rebuild_us_microplex_run_index(
        root,
        registry_path=root / "run_registry.jsonl",
    )
    frontier = select_us_microplex_frontier_index_row(rebuilt_path)

    assert rebuilt_path == index_path
    assert frontier is not None
    assert frontier["artifact_id"] == "run-1"
    with duckdb.connect(str(rebuilt_path), read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM target_metrics").fetchone()[0] == 4
