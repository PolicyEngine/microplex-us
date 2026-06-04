"""Tests for PolicyEngine US integration helpers."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from microplex.core import EntityType
from microplex.targets import (
    TargetAggregation,
    TargetFilter,
    TargetProvider,
    TargetQuery,
    TargetSpec,
)

from microplex_us.pipelines.us import USMicroplexBuildConfig, USMicroplexPipeline
from microplex_us.policyengine.us import (
    POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES,
    SAFE_POLICYENGINE_US_EXPORT_VARIABLES,
    PolicyEngineUSConstraint,
    PolicyEngineUSDBTarget,
    PolicyEngineUSDBTargetProvider,
    PolicyEngineUSEntityTableBundle,
    PolicyEngineUSMicrosimulationAdapter,
    PolicyEngineUSQuantityTarget,
    PolicyEngineUSStratum,
    PolicyEngineUSTargetValidationError,
    PolicyEngineUSVariableBinding,
    build_policyengine_us_export_column_names,
    build_policyengine_us_export_variable_maps,
    build_policyengine_us_time_period_arrays,
    compile_policyengine_us_household_linear_constraints,
    compute_marketplace_plan_benchmark_ratio,
    compute_policyengine_us_definition_hash,
    detect_policyengine_pseudo_inputs,
    materialize_policyengine_us_variables,
    materialize_policyengine_us_variables_safely,
    policyengine_us_variables_to_materialize,
    project_frame_to_time_period_arrays,
    resolve_policyengine_excluded_export_variables,
    write_policyengine_us_time_period_dataset,
)


def _create_policyengine_targets_db(path: Path) -> None:
    national_constraints: tuple[PolicyEngineUSConstraint, ...] = ()
    california_senior_constraints = (
        PolicyEngineUSConstraint("state_fips", "==", "06"),
        PolicyEngineUSConstraint("age", ">=", "65"),
    )
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
                WHEN t.stratum_id = 2 THEN 'state'
                ELSE 'national'
            END AS geo_level,
            CASE
                WHEN t.stratum_id = 2 THEN '06'
                ELSE 'US'
            END AS geographic_id,
            CASE
                WHEN t.stratum_id = 2 THEN 'snap'
                ELSE NULL
            END AS domain_variable
        FROM targets AS t;
        """
    )
    conn.executemany(
        """
        INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
        VALUES (?, ?, ?)
        """,
        [
            (
                1,
                compute_policyengine_us_definition_hash(national_constraints),
                None,
            ),
            (
                2,
                compute_policyengine_us_definition_hash(
                    california_senior_constraints,
                    parent_stratum_id=1,
                ),
                1,
            ),
        ],
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            constraint_variable,
            operation,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (2, "state_fips", "==", "06"),
            (2, "age", ">=", "65"),
        ],
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
            (10, "snap", 2024, 1, 0, 114_100_000_000.0, 1, 5.0, "CBO", "National SNAP"),
            (
                11,
                "snap",
                2024,
                2,
                0,
                9_500_000_000.0,
                1,
                10.0,
                "CBO",
                "California senior SNAP",
            ),
        ],
    )
    conn.commit()
    conn.close()


class TestPolicyEngineUSDBTargetProvider:
    def test_load_targets_includes_constraints(self, tmp_path):
        db_path = tmp_path / "policy_data.db"
        _create_policyengine_targets_db(db_path)

        provider = PolicyEngineUSDBTargetProvider(db_path)
        targets = provider.load_targets(period=2024, variables=["snap"])

        assert len(targets) == 2
        unconstrained, constrained = targets
        assert unconstrained.target_id == 10
        assert unconstrained.constraints == ()
        assert constrained.target_id == 11
        assert {
            (c.variable, c.operation, c.value) for c in constrained.constraints
        } == {
            ("age", ">=", "65"),
            ("state_fips", "==", "06"),
        }
        assert constrained.parent_stratum_id == 1
        assert constrained.definition_hash is not None

    def test_load_strata_returns_hierarchy_metadata(self, tmp_path):
        db_path = tmp_path / "policy_data.db"
        _create_policyengine_targets_db(db_path)

        provider = PolicyEngineUSDBTargetProvider(db_path)
        strata = provider.load_strata([2])

        assert strata == {
            1: PolicyEngineUSStratum(
                stratum_id=1,
                definition_hash=compute_policyengine_us_definition_hash(()),
                parent_stratum_id=None,
                constraints=(),
            ),
            2: PolicyEngineUSStratum(
                stratum_id=2,
                definition_hash=compute_policyengine_us_definition_hash(
                    (
                        PolicyEngineUSConstraint("state_fips", "==", "06"),
                        PolicyEngineUSConstraint("age", ">=", "65"),
                    ),
                    parent_stratum_id=1,
                ),
                parent_stratum_id=1,
                constraints=(
                    PolicyEngineUSConstraint("age", ">=", "65"),
                    PolicyEngineUSConstraint("state_fips", "==", "06"),
                ),
            ),
        }

    def test_load_target_set_returns_canonical_targets(self, tmp_path):
        db_path = tmp_path / "policy_data.db"
        _create_policyengine_targets_db(db_path)

        provider = PolicyEngineUSDBTargetProvider(db_path)
        target_set = provider.load_target_set(
            TargetQuery(
                period=2024,
                provider_filters={"variables": ["snap"]},
            )
        )

        assert isinstance(provider, TargetProvider)
        assert len(target_set.targets) == 2
        assert all(isinstance(target, TargetSpec) for target in target_set.targets)
        assert target_set.targets[0].measure == "snap"
        assert target_set.targets[1].metadata["parent_stratum_id"] == 1
        assert target_set.targets[1].metadata["constraint_count"] == 2
        assert target_set.targets[1].metadata["stratum_definition_hash"] is not None

    def test_load_targets_supports_exact_and_null_domain_filters(self, tmp_path):
        db_path = tmp_path / "policy_data.db"
        _create_policyengine_targets_db(db_path)

        provider = PolicyEngineUSDBTargetProvider(db_path)

        national_targets = provider.load_targets(
            period=2024,
            variables=["snap"],
            geo_levels=["national"],
            domain_variable_is_null=True,
        )
        state_targets = provider.load_targets(
            period=2024,
            variables=["snap"],
            geo_levels=["state"],
            domain_variable_values=["snap"],
        )

        assert [target.target_id for target in national_targets] == [10]
        assert [target.target_id for target in state_targets] == [11]

    def test_load_targets_supports_exact_target_cells(self, tmp_path):
        db_path = tmp_path / "policy_data.db"
        _create_policyengine_targets_db(db_path)

        provider = PolicyEngineUSDBTargetProvider(db_path)
        national_targets = provider.load_targets(
            period=2024,
            target_cells=[
                {
                    "variable": "snap",
                    "geo_level": "national",
                    "domain_variable": None,
                }
            ],
        )
        state_targets = provider.load_targets(
            period=2024,
            target_cells=[
                {
                    "variable": "snap",
                    "geo_level": "state",
                    "domain_variable": "snap",
                }
            ],
        )

        assert [target.target_id for target in national_targets] == [10]
        assert [target.target_id for target in state_targets] == [11]

    def test_load_targets_supports_multi_domain_target_cells(self, tmp_path):
        db_path = tmp_path / "policy_data.db"
        conn = sqlite3.connect(db_path)
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
                'eitc,eitc_child_count' AS domain_variable
            FROM targets AS t;
            """
        )
        constraints = (
            PolicyEngineUSConstraint("state_fips", "==", "06"),
            PolicyEngineUSConstraint("tax_unit_is_filer", "==", "1"),
            PolicyEngineUSConstraint("eitc", ">", "0"),
            PolicyEngineUSConstraint("eitc_child_count", "==", "2"),
        )
        conn.execute(
            """
            INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
            VALUES (?, ?, ?)
            """,
            (
                1,
                compute_policyengine_us_definition_hash(constraints),
                None,
            ),
        )
        conn.executemany(
            """
            INSERT INTO stratum_constraints (
                stratum_id,
                constraint_variable,
                operation,
                value
            ) VALUES (?, ?, ?, ?)
            """,
            [(1, c.variable, c.operation, c.value) for c in constraints],
        )
        conn.execute(
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
            (
                20,
                "tax_unit_count",
                2024,
                1,
                0,
                123.0,
                1,
                5.0,
                "IRS SOI",
                "California EITC recipients with two children",
            ),
        )
        conn.commit()
        conn.close()

        provider = PolicyEngineUSDBTargetProvider(db_path)

        targets = provider.load_targets(
            period=2024,
            target_cells=[
                {
                    "variable": "tax_unit_count",
                    "geo_level": "state",
                    "domain_variable": "eitc_child_count",
                }
            ],
        )
        by_domain = provider.load_targets(
            period=2024,
            variables=["tax_unit_count"],
            domain_variables=["eitc_child_count"],
        )

        assert [target.target_id for target in targets] == [20]
        assert [target.target_id for target in by_domain] == [20]

    def test_load_targets_allows_geographic_hierarchy_without_literal_inheritance(
        self,
        tmp_path,
    ):
        db_path = tmp_path / "policy_data.db"
        conn = sqlite3.connect(db_path)
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
                'district' AS geo_level,
                '0601' AS geographic_id,
                NULL AS domain_variable
            FROM targets AS t;
            """
        )
        parent_constraints = (PolicyEngineUSConstraint("state_fips", "==", "06"),)
        child_constraints = (
            PolicyEngineUSConstraint("congressional_district_geoid", "==", "0601"),
        )
        conn.executemany(
            """
            INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
            VALUES (?, ?, ?)
            """,
            [
                (
                    1,
                    compute_policyengine_us_definition_hash(parent_constraints),
                    None,
                ),
                (
                    2,
                    compute_policyengine_us_definition_hash(
                        child_constraints,
                        parent_stratum_id=1,
                    ),
                    1,
                ),
            ],
        )
        conn.execute(
            """
            INSERT INTO stratum_constraints (
                stratum_id,
                constraint_variable,
                operation,
                value
            ) VALUES (?, ?, ?, ?)
            """,
            (1, "state_fips", "==", "06"),
        )
        conn.execute(
            """
            INSERT INTO stratum_constraints (
                stratum_id,
                constraint_variable,
                operation,
                value
            ) VALUES (?, ?, ?, ?)
            """,
            (2, "congressional_district_geoid", "==", "0601"),
        )
        conn.execute(
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
            (1, "household_count", 2024, 2, 0, 100.0, 1, None, "test", "district"),
        )
        conn.commit()
        conn.close()

        provider = PolicyEngineUSDBTargetProvider(db_path)
        targets = provider.load_targets(period=2024)

        assert [target.target_id for target in targets] == [1]

    def test_load_target_set_keeps_best_available_period_targets(self, tmp_path):
        db_path = tmp_path / "policy_data.db"
        conn = sqlite3.connect(db_path)
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
                'national' AS geo_level,
                'US' AS geographic_id,
                NULL AS domain_variable
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
        conn.execute(
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
            (1, "snap", 2023, 1, 0, 100.0, 1, None, "test", "best-period"),
        )
        conn.commit()
        conn.close()

        provider = PolicyEngineUSDBTargetProvider(db_path)
        target_set = provider.load_target_set(
            TargetQuery(
                period=2024,
                provider_filters={"variables": ["snap"]},
            )
        )

        assert len(target_set.targets) == 1
        assert target_set.targets[0].period == 2023

    def test_load_targets_rejects_invalid_parent_child_constraints(self, tmp_path):
        db_path = tmp_path / "policy_data.db"
        _create_policyengine_targets_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO stratum_constraints (
                stratum_id,
                constraint_variable,
                operation,
                value
            ) VALUES (?, ?, ?, ?)
            """,
            (1, "state_fips", "==", "06"),
        )
        conn.execute(
            "DELETE FROM stratum_constraints WHERE stratum_id = 2 AND constraint_variable = 'state_fips'"
        )
        conn.execute(
            """
            UPDATE strata
            SET definition_hash = ?
            WHERE stratum_id = 1
            """,
            (
                compute_policyengine_us_definition_hash(
                    (PolicyEngineUSConstraint("state_fips", "==", "06"),),
                ),
            ),
        )
        conn.execute(
            """
            UPDATE strata
            SET definition_hash = ?
            WHERE stratum_id = 2
            """,
            (
                compute_policyengine_us_definition_hash(
                    (PolicyEngineUSConstraint("age", ">=", "65"),),
                    parent_stratum_id=1,
                ),
            ),
        )
        conn.commit()
        conn.close()

        provider = PolicyEngineUSDBTargetProvider(db_path)

        with pytest.raises(
            PolicyEngineUSTargetValidationError,
            match="missing inherited parent constraints",
        ):
            provider.load_targets(period=2024, variables=["snap"])

    def test_load_targets_rejects_invalid_definition_hash(self, tmp_path):
        db_path = tmp_path / "policy_data.db"
        _create_policyengine_targets_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE strata SET definition_hash = 'broken' WHERE stratum_id = 2"
        )
        conn.commit()
        conn.close()

        provider = PolicyEngineUSDBTargetProvider(db_path)

        with pytest.raises(
            PolicyEngineUSTargetValidationError,
            match="definition_hash",
        ):
            provider.load_targets(period=2024, variables=["snap"])

    def test_to_quantity_targets_selects_only_unconstrained_targets(self, tmp_path):
        db_path = tmp_path / "policy_data.db"
        _create_policyengine_targets_db(db_path)

        provider = PolicyEngineUSDBTargetProvider(db_path)
        specs = provider.to_quantity_targets({"snap": "snap"}, period=2024)

        assert specs == (
            PolicyEngineUSQuantityTarget(
                name="snap",
                variable="snap",
                column="snap",
                period=2024,
            ),
        )


class TestPolicyEngineUSMicrosimulationAdapter:
    def test_compute_targets_supports_sum_and_count_positive(self):
        class FakeSimulation:
            def calculate(self, variable, period=None, map_to=None):
                assert period == 2024
                if variable == "snap":
                    return np.array([10.0, 0.0, 5.0])
                if variable == "in_poverty":
                    return np.array([1.0, 0.0, 1.0, 1.0])
                raise KeyError(variable)

        adapter = PolicyEngineUSMicrosimulationAdapter(simulation=FakeSimulation())
        targets = adapter.compute_targets(
            (
                PolicyEngineUSQuantityTarget(
                    name="snap_total",
                    variable="snap",
                    column="snap",
                    period=2024,
                    aggregation="sum",
                ),
                PolicyEngineUSQuantityTarget(
                    name="poverty_count",
                    variable="in_poverty",
                    column="in_poverty",
                    period=2024,
                    aggregation="count_positive",
                ),
            )
        )

        assert targets == {
            "snap_total": 15.0,
            "poverty_count": 3.0,
        }

    def test_from_dataset_retries_without_dataset_year_if_unsupported(self, tmp_path):
        dataset_path = tmp_path / "microplex.h5"
        dataset_path.write_text("placeholder")

        class FakeSimulation:
            def __init__(self, dataset):
                self.dataset = dataset

            def calculate(self, variable, period=None, map_to=None):
                _ = variable, period, map_to
                return np.array([0.0])

        adapter = PolicyEngineUSMicrosimulationAdapter.from_dataset(
            dataset_path,
            dataset_year=2024,
            simulation_cls=FakeSimulation,
        )

        assert isinstance(adapter.simulation, FakeSimulation)
        assert adapter.simulation.dataset == str(dataset_path)


class TestPolicyEngineUSConstraintCompilation:
    def test_compiles_db_targets_to_household_linear_constraints(self):
        households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "state_fips": ["06", "36"],
                "weight": [1.0, 1.0],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [10, 11, 12],
                "household_id": [1, 1, 2],
                "age": [70, 40, 30],
            }
        )
        spm_units = pd.DataFrame(
            {
                "spm_unit_id": [100, 101],
                "household_id": [1, 2],
                "snap": [100.0, 0.0],
            }
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=households,
            persons=persons,
            spm_units=spm_units,
        )
        targets = (
            PolicyEngineUSDBTarget(
                target_id=20,
                variable="snap",
                period=2024,
                stratum_id=1,
                reform_id=0,
                value=200.0,
                active=True,
                constraints=(PolicyEngineUSConstraint("state_fips", "==", "06"),),
            ),
            PolicyEngineUSDBTarget(
                target_id=21,
                variable="household_count",
                period=2024,
                stratum_id=2,
                reform_id=0,
                value=2.0,
                active=True,
                constraints=(
                    PolicyEngineUSConstraint("snap", ">", "0"),
                    PolicyEngineUSConstraint("state_fips", "==", "06"),
                ),
            ),
            PolicyEngineUSDBTarget(
                target_id=22,
                variable="person_count",
                period=2024,
                stratum_id=3,
                reform_id=0,
                value=2.0,
                active=True,
                constraints=(
                    PolicyEngineUSConstraint("age", ">=", "65"),
                    PolicyEngineUSConstraint("state_fips", "==", "06"),
                ),
            ),
        )
        variable_bindings = {
            "state_fips": PolicyEngineUSVariableBinding(
                entity=EntityType.HOUSEHOLD,
                column="state_fips",
            ),
            "age": PolicyEngineUSVariableBinding(
                entity=EntityType.PERSON,
                column="age",
            ),
            "snap": PolicyEngineUSVariableBinding(
                entity=EntityType.SPM_UNIT,
                column="snap",
            ),
        }

        constraints = compile_policyengine_us_household_linear_constraints(
            targets=targets,
            tables=tables,
            variable_bindings=variable_bindings,
        )

        assert len(constraints) == 3
        np.testing.assert_allclose(constraints[0].coefficients, np.array([100.0, 0.0]))
        np.testing.assert_allclose(constraints[1].coefficients, np.array([1.0, 0.0]))
        np.testing.assert_allclose(constraints[2].coefficients, np.array([1.0, 0.0]))
        assert [constraint.target for constraint in constraints] == [200.0, 2.0, 2.0]

    def test_amount_targets_apply_same_entity_constraints_before_household_aggregation(
        self,
    ):
        households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "state_fips": ["06", "36"],
                "weight": [1.0, 1.0],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [10, 11, 12],
                "household_id": [1, 1, 2],
                "age": [70, 40, 30],
                "medicaid": [1_000.0, 500.0, 300.0],
            }
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=households,
            persons=persons,
        )
        constraints = compile_policyengine_us_household_linear_constraints(
            targets=(
                PolicyEngineUSDBTarget(
                    target_id=23,
                    variable="medicaid",
                    period=2024,
                    stratum_id=4,
                    reform_id=0,
                    value=1_500.0,
                    active=True,
                    constraints=(PolicyEngineUSConstraint("age", ">=", "65"),),
                ),
            ),
            tables=tables,
            variable_bindings={
                "age": PolicyEngineUSVariableBinding(
                    entity=EntityType.PERSON,
                    column="age",
                ),
                "medicaid": PolicyEngineUSVariableBinding(
                    entity=EntityType.PERSON,
                    column="medicaid",
                ),
            },
        )

        np.testing.assert_allclose(
            constraints[0].coefficients, np.array([1_000.0, 0.0])
        )

    def test_amount_targets_exclude_negative_rows_under_positive_same_entity_filter(
        self,
    ):
        households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "weight": [1.0, 1.0],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [10, 11, 12],
                "household_id": [1, 1, 2],
                "self_employment_income": [100.0, -80.0, -40.0],
            }
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=households,
            persons=persons,
        )

        constraints = compile_policyengine_us_household_linear_constraints(
            targets=(
                TargetSpec(
                    name="positive_self_employment_income",
                    entity=EntityType.HOUSEHOLD,
                    value=100.0,
                    period=2024,
                    measure="self_employment_income",
                    aggregation=TargetAggregation.SUM,
                    filters=(
                        TargetFilter(
                            feature="self_employment_income",
                            operator=">",
                            value=0,
                        ),
                    ),
                ),
            ),
            tables=tables,
            variable_bindings={
                "self_employment_income": PolicyEngineUSVariableBinding(
                    entity=EntityType.PERSON,
                    column="self_employment_income",
                ),
            },
        )

        np.testing.assert_allclose(constraints[0].coefficients, np.array([100.0, 0.0]))

    def test_compiled_constraints_run_through_calibrator(self):
        from microplex.calibration import Calibrator

        households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "state_fips": ["06", "36"],
                "weight": [1.0, 1.0],
            }
        )
        spm_units = pd.DataFrame(
            {
                "spm_unit_id": [100, 101],
                "household_id": [1, 2],
                "snap": [100.0, 0.0],
            }
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=households,
            spm_units=spm_units,
        )
        constraints = compile_policyengine_us_household_linear_constraints(
            targets=(
                PolicyEngineUSDBTarget(
                    target_id=30,
                    variable="snap",
                    period=2024,
                    stratum_id=1,
                    reform_id=0,
                    value=200.0,
                    active=True,
                    constraints=(PolicyEngineUSConstraint("state_fips", "==", "06"),),
                ),
                PolicyEngineUSDBTarget(
                    target_id=31,
                    variable="household_count",
                    period=2024,
                    stratum_id=2,
                    reform_id=0,
                    value=2.0,
                    active=True,
                    constraints=(PolicyEngineUSConstraint("snap", ">", "0"),),
                ),
            ),
            tables=tables,
            variable_bindings={
                "state_fips": PolicyEngineUSVariableBinding(
                    entity=EntityType.HOUSEHOLD,
                    column="state_fips",
                ),
                "snap": PolicyEngineUSVariableBinding(
                    entity=EntityType.SPM_UNIT,
                    column="snap",
                ),
            },
        )

        calibrator = Calibrator(method="entropy")
        calibrated = calibrator.fit_transform(
            households,
            {},
            linear_constraints=constraints,
        )
        report = calibrator.validate(calibrated)

        assert report["max_error"] < 1e-6
        np.testing.assert_allclose(
            calibrated["weight"].values,
            np.array([2.0, 1.0]),
            rtol=1e-5,
        )

    def test_canonical_target_specs_compile_to_household_constraints(self):
        households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "state_fips": ["06", "36"],
                "weight": [1.0, 1.0],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [10, 11, 12],
                "household_id": [1, 1, 2],
                "age": [70, 40, 30],
            }
        )
        spm_units = pd.DataFrame(
            {
                "spm_unit_id": [100, 101],
                "household_id": [1, 2],
                "snap": [100.0, 0.0],
            }
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=households,
            persons=persons,
            spm_units=spm_units,
        )

        constraints = compile_policyengine_us_household_linear_constraints(
            targets=(
                TargetSpec(
                    name="snap_california",
                    entity=EntityType.SPM_UNIT,
                    value=100.0,
                    period=2024,
                    measure="snap",
                    aggregation=TargetAggregation.SUM,
                    filters=(
                        TargetFilter(feature="state_fips", operator="==", value="06"),
                    ),
                ),
                TargetSpec(
                    name="senior_households",
                    entity=EntityType.HOUSEHOLD,
                    value=1.0,
                    period=2024,
                    aggregation=TargetAggregation.COUNT,
                    filters=(TargetFilter(feature="age", operator=">=", value=65),),
                ),
            ),
            tables=tables,
            variable_bindings={
                "state_fips": PolicyEngineUSVariableBinding(
                    entity=EntityType.HOUSEHOLD,
                    column="state_fips",
                ),
                "age": PolicyEngineUSVariableBinding(
                    entity=EntityType.PERSON,
                    column="age",
                ),
                "snap": PolicyEngineUSVariableBinding(
                    entity=EntityType.SPM_UNIT,
                    column="snap",
                ),
            },
        )

        assert [constraint.name for constraint in constraints] == [
            "snap_california",
            "senior_households",
        ]
        np.testing.assert_allclose(constraints[0].coefficients, np.array([100.0, 0.0]))
        np.testing.assert_allclose(constraints[1].coefficients, np.array([1.0, 0.0]))

    def test_amount_targets_align_tax_unit_constraints_before_household_aggregation(
        self,
    ):
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "weight": [1.0, 1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [10, 11, 12],
                    "household_id": [1, 1, 2],
                    "tax_unit_id": [100, 101, 200],
                    "dividend_income": [100.0, 200.0, 300.0],
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100, 101, 200],
                    "household_id": [1, 1, 2],
                    "tax_unit_is_filer": [1, 0, 1],
                }
            ),
        )

        constraints = compile_policyengine_us_household_linear_constraints(
            targets=(
                TargetSpec(
                    name="filer_dividend_income",
                    entity=EntityType.PERSON,
                    value=400.0,
                    period=2024,
                    measure="dividend_income",
                    aggregation=TargetAggregation.SUM,
                    filters=(
                        TargetFilter(
                            feature="tax_unit_is_filer",
                            operator="==",
                            value=1,
                        ),
                    ),
                ),
            ),
            tables=tables,
            variable_bindings={
                "dividend_income": PolicyEngineUSVariableBinding(
                    entity=EntityType.PERSON,
                    column="dividend_income",
                ),
                "tax_unit_is_filer": PolicyEngineUSVariableBinding(
                    entity=EntityType.TAX_UNIT,
                    column="tax_unit_is_filer",
                ),
            },
        )

        np.testing.assert_allclose(
            constraints[0].coefficients,
            np.array([100.0, 300.0]),
        )

    def test_count_targets_align_person_constraints_to_tax_unit_rows(self):
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "weight": [1.0, 1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [10, 11, 12],
                    "household_id": [1, 1, 2],
                    "tax_unit_id": [100, 101, 200],
                    "dividend_income": [100.0, 200.0, 300.0],
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100, 101, 200],
                    "household_id": [1, 1, 2],
                    "tax_unit_is_filer": [1, 0, 1],
                }
            ),
        )

        constraints = compile_policyengine_us_household_linear_constraints(
            targets=(
                TargetSpec(
                    name="filer_tax_units_with_dividends",
                    entity=EntityType.TAX_UNIT,
                    value=2.0,
                    period=2024,
                    aggregation=TargetAggregation.COUNT,
                    filters=(
                        TargetFilter(
                            feature="dividend_income",
                            operator=">",
                            value=0.0,
                        ),
                        TargetFilter(
                            feature="tax_unit_is_filer",
                            operator="==",
                            value=1,
                        ),
                    ),
                ),
            ),
            tables=tables,
            variable_bindings={
                "dividend_income": PolicyEngineUSVariableBinding(
                    entity=EntityType.PERSON,
                    column="dividend_income",
                ),
                "tax_unit_is_filer": PolicyEngineUSVariableBinding(
                    entity=EntityType.TAX_UNIT,
                    column="tax_unit_is_filer",
                ),
            },
        )

        np.testing.assert_allclose(
            constraints[0].coefficients,
            np.array([1.0, 1.0]),
        )

    def test_materializes_formula_variables_before_compiling_constraints(
        self, tmp_path
    ):
        households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "household_weight": [1.0, 1.0],
                "state_fips": [6, 36],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [10, 11],
                "household_id": [1, 2],
                "tax_unit_id": [100, 200],
                "spm_unit_id": [1000, 2000],
                "family_id": [5000, 6000],
                "marital_unit_id": [7000, 8000],
                "age": [34, 52],
                "employment_income": [20_000.0, 15_000.0],
            }
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=households,
            persons=persons,
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100, 200],
                    "household_id": [1, 2],
                }
            ),
            spm_units=pd.DataFrame(
                {
                    "spm_unit_id": [1000, 2000],
                    "household_id": [1, 2],
                }
            ),
            families=pd.DataFrame(
                {
                    "family_id": [5000, 6000],
                    "household_id": [1, 2],
                }
            ),
            marital_units=pd.DataFrame(
                {
                    "marital_unit_id": [7000, 8000],
                    "household_id": [1, 2],
                }
            ),
        )

        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity, formulas=None):
                self.entity = entity
                self.formulas = formulas or {}

            def is_input_variable(self):
                return not self.formulas

        class FakeSystem:
            variables = {
                "employment_income": FakeVariable(FakeEntity("person")),
                "state_fips": FakeVariable(FakeEntity("household")),
                "snap": FakeVariable(
                    FakeEntity("household"),
                    formulas={"2024": object()},
                ),
            }

        class FakeSimulation:
            tax_benefit_system = FakeSystem()

            def __init__(self, dataset, dataset_year=None, **kwargs):
                self.dataset = dataset
                self.dataset_year = dataset_year
                _ = kwargs

            def calculate(self, variable, period=None, map_to=None):
                assert Path(self.dataset).exists()
                assert self.dataset_year == 2024
                assert period == 2024
                assert map_to is None
                if variable == "snap":
                    return np.array([120.0, 0.0])
                raise KeyError(variable)

        materialized_tables, materialized_bindings = (
            materialize_policyengine_us_variables(
                tables,
                variables=("snap",),
                period=2024,
                dataset_year=2024,
                simulation_cls=FakeSimulation,
                temp_dir=tmp_path,
            )
        )

        assert materialized_bindings["snap"] == PolicyEngineUSVariableBinding(
            entity=EntityType.HOUSEHOLD,
            column="snap",
        )
        np.testing.assert_allclose(
            materialized_tables.households["snap"].to_numpy(dtype=float),
            np.array([120.0, 0.0]),
        )

        constraints = compile_policyengine_us_household_linear_constraints(
            targets=(
                PolicyEngineUSDBTarget(
                    target_id=32,
                    variable="snap",
                    period=2024,
                    stratum_id=1,
                    reform_id=0,
                    value=240.0,
                    active=True,
                    constraints=(),
                ),
                PolicyEngineUSDBTarget(
                    target_id=33,
                    variable="household_count",
                    period=2024,
                    stratum_id=2,
                    reform_id=0,
                    value=2.0,
                    active=True,
                    constraints=(PolicyEngineUSConstraint("snap", ">", "0"),),
                ),
            ),
            tables=materialized_tables,
            variable_bindings=materialized_bindings,
        )

        np.testing.assert_allclose(constraints[0].coefficients, np.array([120.0, 0.0]))
        np.testing.assert_allclose(constraints[1].coefficients, np.array([1.0, 0.0]))

    def test_variables_to_materialize_can_force_formula_outputs(self):
        targets = [
            TargetSpec(
                name="ssi",
                entity=EntityType.PERSON,
                value=100.0,
                period=2024,
                measure="ssi",
            )
        ]
        bindings = {
            "ssi": PolicyEngineUSVariableBinding(
                entity=EntityType.PERSON,
                column="ssi",
            ),
            "employment_income": PolicyEngineUSVariableBinding(
                entity=EntityType.PERSON,
                column="employment_income",
            ),
        }

        assert policyengine_us_variables_to_materialize(targets, bindings) == set()
        assert policyengine_us_variables_to_materialize(
            targets,
            bindings,
            force_materialize_variables={"ssi"},
        ) == {"ssi"}

    def test_materialization_supports_nested_system_attribute(self, tmp_path):
        households = pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [1.0],
                "state_fips": [6],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [10],
                "household_id": [1],
                "tax_unit_id": [100],
                "spm_unit_id": [1000],
                "family_id": [5000],
                "marital_unit_id": [7000],
                "employment_income": [20_000.0],
            }
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=households,
            persons=persons,
            tax_units=pd.DataFrame({"tax_unit_id": [100], "household_id": [1]}),
            spm_units=pd.DataFrame({"spm_unit_id": [1000], "household_id": [1]}),
            families=pd.DataFrame({"family_id": [5000], "household_id": [1]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [7000], "household_id": [1]}
            ),
        )

        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity, formulas=None):
                self.entity = entity
                self.formulas = formulas or {}

            def is_input_variable(self):
                return not self.formulas

        class FakeTaxBenefitSystem:
            variables = {
                "employment_income": FakeVariable(FakeEntity("person")),
                "state_fips": FakeVariable(FakeEntity("household")),
                "snap": FakeVariable(
                    FakeEntity("household"),
                    formulas={"2024": object()},
                ),
            }

        class FakeSystemModule:
            system = FakeTaxBenefitSystem()

        class FakeSimulation:
            system = FakeSystemModule()

            def __init__(self, dataset, dataset_year=None, **kwargs):
                self.dataset = dataset
                self.dataset_year = dataset_year
                _ = kwargs

            def calculate(self, variable, period=None, map_to=None):
                assert Path(self.dataset).exists()
                assert self.dataset_year == 2024
                assert period == 2024
                assert map_to is None
                if variable == "snap":
                    return np.array([75.0])
                raise KeyError(variable)

        materialized_tables, materialized_bindings = (
            materialize_policyengine_us_variables(
                tables,
                variables=("snap",),
                period=2024,
                dataset_year=2024,
                simulation_cls=FakeSimulation,
                temp_dir=tmp_path,
            )
        )

        assert materialized_bindings["snap"] == PolicyEngineUSVariableBinding(
            entity=EntityType.HOUSEHOLD,
            column="snap",
        )
        np.testing.assert_allclose(
            materialized_tables.households["snap"].to_numpy(dtype=float),
            np.array([75.0]),
        )

    def test_materialization_skips_derived_pipeline_support_columns(self, tmp_path):
        households = pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [1.0],
                "state_fips": [6],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [10, 11],
                "household_id": [1, 1],
                "tax_unit_id": [100, 100],
                "spm_unit_id": [1000, 1000],
                "family_id": [5000, 5000],
                "marital_unit_id": [7000, 7000],
                "age": [34, 12],
                "age_group": ["18-34", "0-17"],
                "employment_income_before_lsr": [20_000.0, 0.0],
            }
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=households,
            persons=persons,
            tax_units=pd.DataFrame({"tax_unit_id": [100], "household_id": [1]}),
            spm_units=pd.DataFrame({"spm_unit_id": [1000], "household_id": [1]}),
            families=pd.DataFrame({"family_id": [5000], "household_id": [1]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [7000], "household_id": [1]}
            ),
        )

        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity, formulas=None):
                self.entity = entity
                self.formulas = formulas or {}

            def is_input_variable(self):
                return not self.formulas

        class FakeTaxBenefitSystem:
            variables = {
                "age": FakeVariable(FakeEntity("person")),
                "age_group": FakeVariable(FakeEntity("person")),
                "employment_income_before_lsr": FakeVariable(FakeEntity("person")),
                "state_fips": FakeVariable(FakeEntity("household")),
                "snap": FakeVariable(
                    FakeEntity("household"),
                    formulas={"2024": object()},
                ),
            }

        class FakeSimulation:
            tax_benefit_system = FakeTaxBenefitSystem()

            def __init__(self, dataset, dataset_year=None, **kwargs):
                _ = dataset_year, kwargs
                with h5py.File(dataset, "r") as handle:
                    assert "age" in handle
                    assert "employment_income_before_lsr" in handle
                    assert "state_fips" in handle
                    assert "age_group" not in handle
                    assert len(handle["age"]["2024"]) == 2
                    assert len(handle["state_fips"]["2024"]) == 1

            def calculate(self, variable, period=None, map_to=None):
                _ = period, map_to
                if variable == "snap":
                    return np.array([10.0])
                raise KeyError(variable)

        materialized_tables, _ = materialize_policyengine_us_variables(
            tables,
            variables=("snap",),
            period=2024,
            dataset_year=2024,
            simulation_cls=FakeSimulation,
            temp_dir=tmp_path,
        )

        np.testing.assert_allclose(
            materialized_tables.households["snap"].to_numpy(dtype=float),
            np.array([10.0]),
        )

    def test_safe_materialization_one_by_one_uses_prior_materialized_outputs(
        self,
        monkeypatch,
    ):
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [1],
                    "household_weight": [1.0],
                    "state_fips": [6],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [10],
                    "household_id": [1],
                    "tax_unit_id": [100],
                    "spm_unit_id": [1000],
                    "family_id": [5000],
                    "marital_unit_id": [7000],
                }
            ),
            tax_units=pd.DataFrame({"tax_unit_id": [100], "household_id": [1]}),
            spm_units=pd.DataFrame({"spm_unit_id": [1000], "household_id": [1]}),
            families=pd.DataFrame({"family_id": [5000], "household_id": [1]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [7000], "household_id": [1]}
            ),
        )

        def fake_materialize(
            incoming_tables,
            *,
            variables,
            period,
            dataset_year=None,
            simulation_cls=None,
            microsimulation_kwargs=None,
            temp_dir=None,
            direct_override_variables=(),
        ):
            _ = (
                period,
                dataset_year,
                simulation_cls,
                microsimulation_kwargs,
                temp_dir,
                direct_override_variables,
            )
            if tuple(variables) == ("a", "b"):
                raise RuntimeError("batch failed")
            if tuple(variables) == ("a",):
                updated = PolicyEngineUSEntityTableBundle(
                    households=incoming_tables.households.assign(a=[1.0]),
                    persons=incoming_tables.persons.copy(),
                    tax_units=incoming_tables.tax_units.copy(),
                    spm_units=incoming_tables.spm_units.copy(),
                    families=incoming_tables.families.copy(),
                    marital_units=incoming_tables.marital_units.copy(),
                )
                return updated, {
                    "a": PolicyEngineUSVariableBinding(
                        entity=EntityType.HOUSEHOLD,
                        column="a",
                    )
                }
            if tuple(variables) == ("b",):
                assert "a" in incoming_tables.households.columns
                updated = PolicyEngineUSEntityTableBundle(
                    households=incoming_tables.households.assign(
                        b=incoming_tables.households["a"] + 1.0
                    ),
                    persons=incoming_tables.persons.copy(),
                    tax_units=incoming_tables.tax_units.copy(),
                    spm_units=incoming_tables.spm_units.copy(),
                    families=incoming_tables.families.copy(),
                    marital_units=incoming_tables.marital_units.copy(),
                )
                return updated, {
                    "b": PolicyEngineUSVariableBinding(
                        entity=EntityType.HOUSEHOLD,
                        column="b",
                    )
                }
            raise AssertionError(f"unexpected variables: {variables}")

        monkeypatch.setattr(
            "microplex_us.policyengine.us.materialize_policyengine_us_variables",
            fake_materialize,
        )

        result = materialize_policyengine_us_variables_safely(
            tables,
            variables=("a", "b"),
            period=2024,
        )

        assert result.materialized_variables == ("a", "b")
        assert result.failed_variables == {}
        np.testing.assert_allclose(
            result.tables.households["b"].to_numpy(dtype=float),
            np.array([2.0]),
        )


class TestPolicyEngineUSProjection:
    def test_builds_structural_time_period_arrays_from_entity_tables(self):
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10, 20],
                    "household_weight": [1.5, 2.5],
                    "state_code": ["CA", "NY"],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "household_id": [10, 10, 20],
                    "tax_unit_id": [100, 100, 200],
                    "spm_unit_id": [1000, 1000, 2000],
                    "family_id": [5000, 5000, 6000],
                    "marital_unit_id": [7000, 7000, 8000],
                    "age": [34, 12, 45],
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100, 200],
                    "household_id": [10, 20],
                }
            ),
            spm_units=pd.DataFrame(
                {
                    "spm_unit_id": [1000, 2000],
                    "household_id": [10, 20],
                    "snap": [1200.0, 300.0],
                }
            ),
            families=pd.DataFrame(
                {
                    "family_id": [5000, 6000],
                    "household_id": [10, 20],
                }
            ),
            marital_units=pd.DataFrame(
                {
                    "marital_unit_id": [7000, 8000],
                    "household_id": [10, 20],
                }
            ),
        )

        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            household_variable_map={"state_code": "state_code"},
            person_variable_map={"age": "age"},
            spm_unit_variable_map={"snap": "snap"},
        )

        expected_keys = {
            "household_id",
            "person_id",
            "tax_unit_id",
            "spm_unit_id",
            "family_id",
            "marital_unit_id",
            "person_household_id",
            "person_tax_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_marital_unit_id",
            "household_weight",
            "state_code",
            "age",
            "snap",
        }
        assert expected_keys.issubset(arrays)
        assert set(arrays["household_id"]) == {"2024"}
        np.testing.assert_array_equal(
            arrays["household_id"]["2024"], np.array([10, 20])
        )
        np.testing.assert_array_equal(
            arrays["person_household_id"]["2024"], np.array([10, 10, 20])
        )
        np.testing.assert_array_equal(
            arrays["person_tax_unit_id"]["2024"], np.array([100, 100, 200])
        )
        np.testing.assert_array_equal(
            arrays["person_spm_unit_id"]["2024"], np.array([1000, 1000, 2000])
        )
        np.testing.assert_array_equal(
            arrays["person_family_id"]["2024"], np.array([5000, 5000, 6000])
        )
        np.testing.assert_array_equal(
            arrays["person_marital_unit_id"]["2024"], np.array([7000, 7000, 8000])
        )
        np.testing.assert_allclose(
            arrays["household_weight"]["2024"], np.array([1.5, 2.5], dtype=np.float32)
        )
        assert "person_weight" not in arrays
        assert "tax_unit_weight" not in arrays
        assert "spm_unit_weight" not in arrays
        assert "family_weight" not in arrays
        assert "marital_unit_weight" not in arrays
        np.testing.assert_array_equal(arrays["age"]["2024"], np.array([34, 12, 45]))
        np.testing.assert_allclose(arrays["snap"]["2024"], np.array([1200.0, 300.0]))

    def test_export_column_names_match_written_time_period_arrays(self):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity):
                self.entity = FakeEntity(entity)

        class FakeSystem:
            variables = {
                "age": FakeVariable("person"),
                "snap": FakeVariable("spm_unit"),
                "state_code": FakeVariable("household"),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.5],
                    "state_code": ["CA"],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "household_id": [10, 10],
                    "tax_unit_id": [100, 100],
                    "spm_unit_id": [1000, 1000],
                    "family_id": [5000, 5000],
                    "marital_unit_id": [7000, 7000],
                    "age": [34, 12],
                }
            ),
            tax_units=pd.DataFrame({"tax_unit_id": [100], "household_id": [10]}),
            spm_units=pd.DataFrame(
                {"spm_unit_id": [1000], "household_id": [10], "snap": [1200.0]}
            ),
            families=pd.DataFrame({"family_id": [5000], "household_id": [10]}),
            marital_units=pd.DataFrame(
                {"marital_unit_id": [7000], "household_id": [10]}
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
        )
        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            household_variable_map=export_maps["household"],
            person_variable_map=export_maps["person"],
            spm_unit_variable_map=export_maps["spm_unit"],
        )
        columns = build_policyengine_us_export_column_names(
            tables,
            tax_benefit_system=FakeSystem(),
        )
        excluded = resolve_policyengine_excluded_export_variables(
            FakeSystem(),
            sorted(arrays),
        )

        assert columns == set(arrays) - excluded

    def test_derives_household_head_export_from_relationship_to_head(self):
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10, 20],
                    "household_weight": [1.0, 1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "household_id": [10, 10, 20],
                    "relationship_to_head": [0, 2, 0],
                }
            ),
        )

        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            person_variable_map={"is_household_head": "is_household_head"},
        )

        np.testing.assert_array_equal(
            arrays["is_household_head"]["2024"],
            np.array([True, False, True]),
        )

    def test_export_variable_maps_include_derived_household_head(self):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity):
                self.entity = FakeEntity(entity)

        class FakeSystem:
            variables = {
                "is_household_head": FakeVariable("person"),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1],
                    "household_id": [10],
                    "relationship_to_head": [0],
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
        )

        assert export_maps["person"]["is_household_head"] == "is_household_head"

    def test_derives_missing_group_tables_from_person_memberships(self):
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10, 20],
                    "weight": [1.5, 2.5],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "household_id": [10, 10, 20],
                    "tax_unit_id": [100, 100, 200],
                    "spm_unit_id": [1000, 1000, 2000],
                    "family_id": [5000, 5000, 6000],
                    "marital_unit_id": [7000, 7000, 8000],
                }
            ),
        )

        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
        )

        np.testing.assert_array_equal(
            arrays["tax_unit_id"]["2024"], np.array([100, 200])
        )
        np.testing.assert_array_equal(
            arrays["spm_unit_id"]["2024"], np.array([1000, 2000])
        )
        np.testing.assert_array_equal(
            arrays["family_id"]["2024"], np.array([5000, 6000])
        )
        np.testing.assert_array_equal(
            arrays["marital_unit_id"]["2024"], np.array([7000, 8000])
        )
        assert "family_weight" not in arrays
        assert "marital_unit_weight" not in arrays

    def test_detects_computed_exports_from_formula_and_aggregate_variables(self):
        class FakeVariable:
            def __init__(self, adds=None, subtracts=None, formulas=None):
                self.adds = adds or []
                self.subtracts = subtracts or []
                self.formulas = formulas or {}

        class FakeSystem:
            variables = {
                "employment_income": FakeVariable(),
                "filing_status": FakeVariable(formulas={"2024": object()}),
                "self_employed_pension_contribution_ald_person": FakeVariable(
                    formulas={"2024": object()}
                ),
                "self_employed_pension_contribution_ald": FakeVariable(
                    adds=["self_employed_pension_contribution_ald_person"]
                ),
            }

        pseudo_inputs = detect_policyengine_pseudo_inputs(
            FakeSystem(),
            [
                "employment_income",
                "filing_status",
                "self_employed_pension_contribution_ald",
            ],
        )

        assert pseudo_inputs == {
            "filing_status",
            "self_employed_pension_contribution_ald",
        }

    def test_build_policyengine_us_export_variable_maps_includes_tax_inputs(self):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity):
                self.entity = FakeEntity(entity)

        class FakeSystem:
            variables = {
                "state_fips": FakeVariable("household"),
                "alimony_income": FakeVariable("person"),
                "child_support_expense": FakeVariable("person"),
                "child_support_received": FakeVariable("person"),
                "disability_benefits": FakeVariable("person"),
                "employment_income_before_lsr": FakeVariable("person"),
                "health_insurance_premiums_without_medicare_part_b": FakeVariable(
                    "person"
                ),
                "is_female": FakeVariable("person"),
                "medicare_part_b_premiums": FakeVariable("person"),
                "other_medical_expenses": FakeVariable("person"),
                "over_the_counter_health_expenses": FakeVariable("person"),
                "rent": FakeVariable("person"),
                "real_estate_taxes": FakeVariable("person"),
                "medicaid": FakeVariable("person"),
                "medicaid_enrolled": FakeVariable("person"),
                "ssi": FakeVariable("person"),
                "ssi_reported": FakeVariable("person"),
                "takes_up_ssi_if_eligible": FakeVariable("person"),
                "self_employment_income_before_lsr": FakeVariable("person"),
                "taxable_interest_income": FakeVariable("person"),
                "qualified_dividend_income": FakeVariable("person"),
                "non_qualified_dividend_income": FakeVariable("person"),
                "short_term_capital_gains": FakeVariable("person"),
                "long_term_capital_gains_before_response": FakeVariable("person"),
                "rental_income": FakeVariable("person"),
                "unemployment_compensation": FakeVariable("person"),
                "snap": FakeVariable("spm_unit"),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.0],
                    "state_fips": [6],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1],
                    "household_id": [10],
                    "alimony_income": [500.0],
                    "child_support_expense": [350.0],
                    "child_support_received": [200.0],
                    "disability_benefits": [300.0],
                    "employment_income_before_lsr": [50_000.0],
                    "health_insurance_premiums_without_medicare_part_b": [900.0],
                    "is_female": [True],
                    "medicare_part_b_premiums": [400.0],
                    "other_medical_expenses": [250.0],
                    "over_the_counter_health_expenses": [75.0],
                    "rent": [1_200.0],
                    "real_estate_taxes": [300.0],
                    "medicaid": [1_200.0],
                    "medicaid_enrolled": [True],
                    "ssi": [400.0],
                    "ssi_reported": [400.0],
                    "takes_up_ssi_if_eligible": [True],
                    "self_employment_income_before_lsr": [2_000.0],
                    "taxable_interest_income": [100.0],
                    "qualified_dividend_income": [40.0],
                    "non_qualified_dividend_income": [60.0],
                    "short_term_capital_gains": [25.0],
                    "long_term_capital_gains_before_response": [75.0],
                    "rental_income": [500.0],
                    "unemployment_compensation": [0.0],
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100],
                    "household_id": [10],
                }
            ),
            spm_units=pd.DataFrame(
                {
                    "spm_unit_id": [1000],
                    "household_id": [10],
                    "snap": [1_800.0],
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
        )

        assert export_maps["household"] == {"state_fips": "state_fips"}
        assert export_maps["tax_unit"] == {}
        assert "snap" not in export_maps["spm_unit"]
        expected_person_exports = {
            "alimony_income": "alimony_income",
            "child_support_expense": "child_support_expense",
            "child_support_received": "child_support_received",
            "disability_benefits": "disability_benefits",
            "employment_income_before_lsr": "employment_income_before_lsr",
            "health_insurance_premiums_without_medicare_part_b": "health_insurance_premiums_without_medicare_part_b",
            "is_female": "is_female",
            "other_medical_expenses": "other_medical_expenses",
            "over_the_counter_health_expenses": "over_the_counter_health_expenses",
            "real_estate_taxes": "real_estate_taxes",
            "self_employment_income_before_lsr": "self_employment_income_before_lsr",
            "takes_up_ssi_if_eligible": "takes_up_ssi_if_eligible",
            "taxable_interest_income": "taxable_interest_income",
            "qualified_dividend_income": "qualified_dividend_income",
            "non_qualified_dividend_income": "non_qualified_dividend_income",
            "short_term_capital_gains": "short_term_capital_gains",
            "long_term_capital_gains_before_response": "long_term_capital_gains_before_response",
            "rental_income": "rental_income",
            "unemployment_compensation": "unemployment_compensation",
        }
        assert expected_person_exports.items() <= export_maps["person"].items()
        assert "medicare_part_b_premiums" not in export_maps["person"].values()
        assert "rent" not in export_maps["person"].values()
        assert "medicaid" not in export_maps["person"].values()

    def test_build_policyengine_us_export_variable_maps_includes_contract_inputs(self):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity):
                self.entity = FakeEntity(entity)

        person_contract_inputs = (
            "alimony_expense",
            "amt_foreign_tax_credit",
            "bank_account_assets",
            "bond_assets",
            "casualty_loss",
            "charitable_cash_donations",
            "charitable_non_cash_donations",
            "early_withdrawal_penalty",
            "educator_expense",
            "excess_withheld_payroll_tax",
            "general_business_credit",
            "investment_income_elected_form_4952",
            "is_household_head",
            "long_term_capital_gains_on_collectibles",
            "miscellaneous_income",
            "non_sch_d_capital_gains",
            "other_credits",
            "own_children_in_household",
            "prior_year_minimum_tax_credit",
            "qualified_tuition_expenses",
            "salt_refund_income",
            "stock_assets",
            "taxable_ira_distributions",
            "tip_income",
            "unreimbursed_business_employee_expenses",
        )
        household_contract_inputs = (
            "auto_loan_balance",
            "auto_loan_interest",
            "block_geoid",
            "congressional_district_geoid",
            "county_fips",
            "tenure_type",
            "tract_geoid",
        )
        tax_unit_contract_inputs = (
            "domestic_production_ald",
            "recapture_of_investment_credit",
            "unrecaptured_section_1250_gain",
            "unreported_payroll_tax",
        )
        spm_unit_contract_inputs = (
            "receives_housing_assistance",
            "spm_unit_pre_subsidy_childcare_expenses",
            "spm_unit_tenure_type",
        )
        legacy_spm_unit_contract_inputs = (
            "spm_unit_energy_subsidy",
            "takes_up_housing_assistance_if_eligible",
        )
        legacy_person_contract_inputs = ("count_under_18", "count_under_6")

        class FakeSystem:
            variables = {
                **{name: FakeVariable("person") for name in person_contract_inputs},
                **{
                    name: FakeVariable("household")
                    for name in household_contract_inputs
                },
                **{name: FakeVariable("tax_unit") for name in tax_unit_contract_inputs},
                **{
                    name: FakeVariable("spm_unit")
                    for name in (
                        *spm_unit_contract_inputs,
                        *legacy_spm_unit_contract_inputs,
                    )
                },
                "self_employed_health_insurance_ald": FakeVariable("tax_unit"),
                "self_employed_pension_contribution_ald": FakeVariable("tax_unit"),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.0],
                    **{name: [5.0] for name in household_contract_inputs},
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1],
                    "household_id": [10],
                    **{name: [1.0] for name in person_contract_inputs},
                    **{name: [0] for name in legacy_person_contract_inputs},
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100],
                    "household_id": [10],
                    **{name: [2.0] for name in tax_unit_contract_inputs},
                    "self_employed_health_insurance_ald": [3.0],
                    "self_employed_pension_contribution_ald": [4.0],
                }
            ),
            spm_units=pd.DataFrame(
                {
                    "spm_unit_id": [1000],
                    "household_id": [10],
                    "receives_housing_assistance": [True],
                    "spm_unit_pre_subsidy_childcare_expenses": [1500.0],
                    "spm_unit_tenure_type": ["RENTER"],
                    **{name: [1.0] for name in legacy_spm_unit_contract_inputs},
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
        )

        assert {
            name: name
            for name in (*person_contract_inputs, *legacy_person_contract_inputs)
        }.items() <= export_maps["person"].items()
        assert export_maps["household"] == {
            name: name for name in household_contract_inputs
        }
        assert export_maps["tax_unit"] == {
            name: name for name in tax_unit_contract_inputs
        }
        assert {
            name: name
            for name in (
                *spm_unit_contract_inputs,
                *legacy_spm_unit_contract_inputs,
            )
        }.items() <= export_maps["spm_unit"].items()

    def test_build_policyengine_us_export_variable_maps_blocks_computed_direct_overrides(
        self,
    ):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity, formulas=None):
                self.entity = FakeEntity(entity)
                self.formulas = formulas or {}

        class FakeSystem:
            variables = {
                "state_fips": FakeVariable("household"),
                "employment_income_before_lsr": FakeVariable("person"),
                "is_female": FakeVariable("person"),
                "medicaid": FakeVariable("person", formulas={"2024": object()}),
                "medicaid_enrolled": FakeVariable(
                    "person",
                    formulas={"2024": object()},
                ),
                "ssi": FakeVariable("person", formulas={"2024": object()}),
                "filing_status": FakeVariable("tax_unit", formulas={"2024": object()}),
                "snap": FakeVariable("spm_unit", formulas={"2024": object()}),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.0],
                    "state_fips": [6],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1],
                    "household_id": [10],
                    "employment_income_before_lsr": [50_000.0],
                    "is_female": [True],
                    "medicaid": [1_200.0],
                    "medicaid_enrolled": [True],
                    "ssi": [400.0],
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100],
                    "household_id": [10],
                    "filing_status": ["SINGLE"],
                }
            ),
            spm_units=pd.DataFrame(
                {
                    "spm_unit_id": [1000],
                    "household_id": [10],
                    "snap": [1_800.0],
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
            direct_override_variables=(
                "filing_status",
                "snap",
                "ssi",
                "medicaid",
                "medicaid_enrolled",
            ),
        )

        assert {
            "employment_income_before_lsr": "employment_income_before_lsr",
            "is_female": "is_female",
        }.items() <= export_maps["person"].items()
        assert "medicaid" not in export_maps["person"].values()
        assert "medicaid_enrolled" not in export_maps["person"].values()
        assert "ssi" not in export_maps["person"].values()
        assert export_maps["tax_unit"] == {}
        assert "snap" not in export_maps["spm_unit"].values()

    def test_build_policyengine_us_export_variable_maps_drops_reported_social_security_retirement_alias(
        self,
    ):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity):
                self.entity = FakeEntity(entity)

        class FakeSystem:
            variables = {
                "social_security_retirement_reported": FakeVariable("person"),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1],
                    "household_id": [10],
                    "social_security_retirement": [12_000.0],
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
        )

        assert (
            "social_security_retirement_reported" not in export_maps["person"].values()
        )

    def test_build_policyengine_us_export_variable_maps_drops_computed_alias_inputs(
        self,
    ):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity):
                self.entity = FakeEntity(entity)

        class FakeSystem:
            variables = {
                "medicare_part_b_premiums_reported": FakeVariable("person"),
                "roth_ira_contributions_desired": FakeVariable("person"),
                "self_employed_pension_contributions_desired": FakeVariable("person"),
                "traditional_ira_contributions_desired": FakeVariable("person"),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1],
                    "household_id": [10],
                    "medicare_part_b_premiums": [1_800.0],
                    "roth_ira_contributions": [2_000.0],
                    "self_employed_pension_contributions": [4_000.0],
                    "traditional_ira_contributions": [3_000.0],
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
        )

        assert "medicare_part_b_premiums_reported" not in export_maps["person"].values()
        assert "roth_ira_contributions_desired" not in export_maps["person"].values()
        assert (
            "self_employed_pension_contributions_desired"
            not in export_maps["person"].values()
        )
        assert (
            "traditional_ira_contributions_desired"
            not in export_maps["person"].values()
        )

    def test_default_policyengine_us_export_surface_avoids_formula_aggregates(self):
        from policyengine_us import CountryTaxBenefitSystem

        tbs = CountryTaxBenefitSystem()

        overlaps = sorted(
            name
            for name in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
            if name in tbs.variables
            and name not in POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES
            and (
                getattr(tbs.variables[name], "formulas", None)
                or getattr(tbs.variables[name], "adds", None)
                or getattr(tbs.variables[name], "subtracts", None)
            )
        )

        assert overlaps == []
        assert "estate_income" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "child_support_expense" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "farm_operations_income" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "farm_rent_income" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "health_savings_account_ald" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "filing_status" not in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "rent" not in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "social_security_retirement" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert (
            "social_security_retirement_reported"
            not in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        )
        assert (
            "medicare_part_b_premiums_reported"
            not in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        )
        assert "traditional_ira_contributions" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert (
            "traditional_ira_contributions_desired"
            in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        )
        assert "roth_ira_contributions" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "roth_ira_contributions_desired" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "traditional_401k_contributions" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "roth_401k_contributions" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert (
            "self_employed_pension_contributions"
            in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        )
        assert (
            "self_employed_pension_contributions_desired"
            in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        )
        assert (
            "spm_unit_capped_work_childcare_expenses"
            in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        )
        assert "non_sch_d_capital_gains" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "receives_wic" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "ssn_card_type" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "tenure_type" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "spm_unit_tenure_type" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "is_separated" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "is_surviving_spouse" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "is_blind" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "takes_up_ssi_if_eligible" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "ssi" not in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert "ssi_reported" not in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert (
            "self_employed_health_insurance_ald"
            not in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        )
        assert (
            "self_employed_pension_contribution_ald"
            not in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        )

    def test_resolve_policyengine_excluded_export_variables_preserves_explicit_overrides(
        self,
    ):
        class FakeVariable:
            def __init__(self, adds=None, subtracts=None, formulas=None):
                self.adds = adds or []
                self.subtracts = subtracts or []
                self.formulas = formulas or {}

        class FakeSystem:
            variables = {
                "employment_income": FakeVariable(),
                "self_employed_pension_contribution_ald_person": FakeVariable(
                    formulas={"2024": object()}
                ),
                "self_employed_pension_contribution_ald": FakeVariable(
                    adds=["self_employed_pension_contribution_ald_person"]
                ),
            }

        excluded = resolve_policyengine_excluded_export_variables(
            FakeSystem(),
            ["employment_income", "self_employed_pension_contribution_ald"],
            direct_override_variables=("self_employed_pension_contribution_ald",),
        )

        assert excluded == {"self_employed_pension_contribution_ald"}

    def test_build_policyengine_us_export_variable_maps_supports_exact_pre_sim_names(
        self,
    ):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity):
                self.entity = FakeEntity(entity)

        class FakeSystem:
            variables = {
                "cps_race": FakeVariable("person"),
                "non_sch_d_capital_gains": FakeVariable("person"),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1],
                    "household_id": [10],
                    "race": [4],
                    "non_sch_d_capital_gains": [250.0],
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
            direct_override_variables=("non_sch_d_capital_gains",),
        )

        assert {
            "race": "cps_race",
            "non_sch_d_capital_gains": "non_sch_d_capital_gains",
        }.items() <= export_maps["person"].items()

    def test_build_policyengine_us_export_variable_maps_prefers_exact_pre_sim_names(
        self,
    ):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity):
                self.entity = FakeEntity(entity)

        class FakeSystem:
            variables = {
                "cps_race": FakeVariable("person"),
                "is_hispanic": FakeVariable("person"),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1],
                    "household_id": [10],
                    "race": [4],
                    "cps_race": [3],
                    "is_hispanic": [False],
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
        )

        assert {
            "cps_race": "cps_race",
            "is_hispanic": "is_hispanic",
        }.items() <= export_maps["person"].items()

    def test_build_policyengine_us_export_variable_maps_aliases_rent_to_pre_subsidy_rent(
        self,
    ):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity, formulas=None):
                self.entity = FakeEntity(entity)
                self.formulas = formulas or {}

        class FakeSystem:
            variables = {
                "pre_subsidy_rent": FakeVariable("person"),
                "rent": FakeVariable("person", formulas={"2024": object()}),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10, 20],
                    "household_weight": [1.0, 1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "household_id": [10, 20],
                    "rent": [14_400.0, 0.0],
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
        )
        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            person_variable_map=export_maps["person"],
        )

        assert export_maps["person"]["rent"] == "pre_subsidy_rent"
        assert "pre_subsidy_rent" not in export_maps["person"]
        assert "rent" not in export_maps["person"].values()
        assert arrays["pre_subsidy_rent"]["2024"].tolist() == [14_400.0, 0.0]

    def test_build_policyengine_us_export_variable_maps_includes_absent_export_defaults(
        self,
    ):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity):
                self.entity = FakeEntity(entity)

        class FakeSystem:
            variables = {
                "auto_loan_balance": FakeVariable("household"),
                "first_home_mortgage_balance": FakeVariable("tax_unit"),
                "immigration_status_str": FakeVariable("person"),
                "net_worth": FakeVariable("household"),
                "spm_unit_pre_subsidy_childcare_expenses": FakeVariable("spm_unit"),
                "spm_unit_tenure_type": FakeVariable("spm_unit"),
                "ssn_card_type": FakeVariable("person"),
                "takes_up_aca_if_eligible": FakeVariable("tax_unit"),
                "takes_up_early_head_start_if_eligible": FakeVariable("person"),
                "takes_up_eitc": FakeVariable("tax_unit"),
                "takes_up_snap_if_eligible": FakeVariable("spm_unit"),
                "tenure_type": FakeVariable("household"),
                "weeks_unemployed": FakeVariable("person"),
                "would_claim_wic": FakeVariable("person"),
                "would_file_taxes_voluntarily": FakeVariable("tax_unit"),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1],
                    "household_id": [10],
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100],
                    "household_id": [10],
                }
            ),
            spm_units=pd.DataFrame(
                {
                    "spm_unit_id": [1000],
                    "household_id": [10],
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
        )

        assert {
            "immigration_status_str": "immigration_status_str",
            "ssn_card_type": "ssn_card_type",
            "takes_up_early_head_start_if_eligible": "takes_up_early_head_start_if_eligible",
            "weeks_unemployed": "weeks_unemployed",
            "would_claim_wic": "would_claim_wic",
        }.items() <= export_maps["person"].items()
        assert export_maps["household"] == {
            "auto_loan_balance": "auto_loan_balance",
            "net_worth": "net_worth",
            "tenure_type": "tenure_type",
        }
        assert export_maps["tax_unit"] == {
            "first_home_mortgage_balance": "first_home_mortgage_balance",
            "takes_up_aca_if_eligible": "takes_up_aca_if_eligible",
            "takes_up_eitc": "takes_up_eitc",
            "would_file_taxes_voluntarily": "would_file_taxes_voluntarily",
        }
        assert {
            "spm_unit_pre_subsidy_childcare_expenses": "spm_unit_pre_subsidy_childcare_expenses",
            "spm_unit_tenure_type": "spm_unit_tenure_type",
            "takes_up_snap_if_eligible": "takes_up_snap_if_eligible",
        }.items() <= export_maps["spm_unit"].items()

    def test_time_period_arrays_derive_ecps_persisted_computed_inputs(self):
        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity, formulas=None):
                self.entity = FakeEntity(entity)
                self.formulas = formulas or {}

        class FakeSystem:
            variables = {
                "fsla_overtime_premium": FakeVariable("person"),
                "has_itin": FakeVariable("person", formulas={"2024": object()}),
                "has_tin": FakeVariable("person", formulas={"2024": object()}),
                "hours_worked_last_week": FakeVariable("person"),
                "in_nyc": FakeVariable("household", formulas={"2024": object()}),
                "meets_ssi_disability_criteria": FakeVariable("person"),
                "weekly_hours_worked_before_lsr": FakeVariable("person"),
            }

        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10, 11],
                    "household_weight": [1.0, 2.0],
                    "state_fips": [36, 6],
                    "county_fips": [61, 1],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "household_id": [10, 11],
                    "ssn_card_type": ["NONE", "CITIZEN"],
                    "age": [40, 70],
                    "difficulty_hearing": [True, False],
                    "ssi": [0.0, 0.0],
                    "employment_income": [0.0, 55_000.0],
                    "hours_worked": [0.0, 50.0],
                    "weeks_worked": [0.0, 52.0],
                    "is_paid_hourly": [False, True],
                    "has_never_worked": [False, False],
                    "is_military": [False, False],
                    "is_executive_administrative_professional": [False, False],
                    "is_farmer_fisher": [False, False],
                    "is_computer_scientist": [False, False],
                }
            ),
        )

        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=FakeSystem(),
        )
        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            household_variable_map=export_maps["household"],
            person_variable_map=export_maps["person"],
        )

        columns = build_policyengine_us_export_column_names(
            tables,
            tax_benefit_system=FakeSystem(),
        )

        assert arrays["in_nyc"]["2024"].tolist() == [True, False]
        assert arrays["has_tin"]["2024"].tolist() == [False, True]
        assert arrays["has_itin"]["2024"].tolist() == [False, True]
        assert arrays["hours_worked_last_week"]["2024"].tolist() == [0.0, 50.0]
        assert arrays["weekly_hours_worked_before_lsr"]["2024"].tolist() == [
            0.0,
            50.0,
        ]
        assert arrays["meets_ssi_disability_criteria"]["2024"].tolist() == [
            True,
            False,
        ]
        np.testing.assert_allclose(
            arrays["fsla_overtime_premium"]["2024"],
            np.array([0.0, 5_000.0], dtype=np.float32),
        )
        assert {
            "fsla_overtime_premium",
            "has_itin",
            "has_tin",
            "hours_worked_last_week",
            "in_nyc",
            "meets_ssi_disability_criteria",
            "weekly_hours_worked_before_lsr",
        }.issubset(columns)

    def test_projects_frame_and_writes_time_period_dataset(self, tmp_path):
        frame = pd.DataFrame(
            {
                "person_id": [1, 2],
                "age": [34, 12],
                "state_code": ["CA", "NY"],
                "self_employed_pension_contribution_ald": [10.0, 20.0],
            }
        )

        arrays = project_frame_to_time_period_arrays(
            frame,
            period=2024,
            column_map={
                "person_id": "person_id",
                "age": "age",
                "state_code": "state_code",
                "self_employed_pension_contribution_ald": (
                    "self_employed_pension_contribution_ald"
                ),
            },
        )
        output_path = tmp_path / "microplex_pe.h5"
        write_policyengine_us_time_period_dataset(
            arrays,
            output_path,
            excluded_variables={"self_employed_pension_contribution_ald"},
        )

        with h5py.File(output_path, "r") as handle:
            assert set(handle.keys()) == {"age", "person_id", "state_code"}
            assert np.array_equal(handle["age"]["2024"][:], np.array([34, 12]))
            assert handle["state_code"]["2024"].dtype.kind == "S"

    def test_writer_rejects_computed_policyengine_variables(self, tmp_path):
        class FakeVariable:
            def __init__(self, formulas=None):
                self.formulas = formulas or {}

        class FakeSystem:
            variables = {
                "age": FakeVariable(),
                "filing_status": FakeVariable(formulas={"2024": object()}),
            }

        arrays = {
            "age": {"2024": np.asarray([34, 12])},
            "filing_status": {"2024": np.asarray([1, 2])},
        }

        with pytest.raises(ValueError, match="filing_status"):
            write_policyengine_us_time_period_dataset(
                arrays,
                tmp_path / "microplex_pe.h5",
                tax_benefit_system=FakeSystem(),
            )

    def test_build_time_period_arrays_defaults_missing_ssn_card_type_to_citizen(self):
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10, 20],
                    "household_weight": [1.0, 1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "household_id": [10, 20],
                    "ssn_card_type": ["NONE", None],
                }
            ),
        )

        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            person_variable_map={"ssn_card_type": "ssn_card_type"},
        )

        assert arrays["ssn_card_type"]["2024"].tolist() == [b"NONE", b"CITIZEN"]

    def test_compute_marketplace_plan_benchmark_ratio_clips_marketplace_takers(self):
        ratio = compute_marketplace_plan_benchmark_ratio(
            reported_premium=np.array([300.0, 50.0, 500.0, 500.0]),
            aca_ptc=np.array([700.0, 0.0, 0.0, 500.0]),
            slcsp=np.array([1_000.0, 1_000.0, 0.0, 1_000.0]),
            takes_up_aca=np.array([True, True, True, False]),
        )

        np.testing.assert_allclose(ratio, np.array([1.0, 0.5, 1.0, 1.0]))

    def test_build_time_period_arrays_derives_marketplace_plan_ratio(self):
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10, 20, 30],
                    "household_weight": [1.0, 1.0, 1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "household_id": [10, 20, 30],
                    "tax_unit_id": [100, 200, 300],
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100, 200, 300],
                    "household_id": [10, 20, 30],
                    "health_insurance_premiums_without_medicare_part_b": [
                        300.0,
                        50.0,
                        200.0,
                    ],
                    "aca_ptc": [700.0, 0.0, 100.0],
                    "slcsp": [1_000.0, 1_000.0, 0.0],
                    "takes_up_aca_if_eligible": [True, True, True],
                }
            ),
        )

        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            tax_unit_variable_map={
                "selected_marketplace_plan_benchmark_ratio": (
                    "selected_marketplace_plan_benchmark_ratio"
                )
            },
        )

        np.testing.assert_allclose(
            arrays["selected_marketplace_plan_benchmark_ratio"]["2024"],
            np.array([1.0, 0.5, 1.0]),
        )

    def test_build_time_period_arrays_defaults_absent_export_inputs(self):
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10],
                    "household_weight": [1.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1],
                    "household_id": [10],
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100],
                    "household_id": [10],
                }
            ),
            spm_units=pd.DataFrame(
                {
                    "spm_unit_id": [1000],
                    "household_id": [10],
                }
            ),
        )

        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            person_variable_map={
                "immigration_status_str": "immigration_status_str",
                "ssn_card_type": "ssn_card_type",
                "takes_up_early_head_start_if_eligible": "takes_up_early_head_start_if_eligible",
                "weeks_unemployed": "weeks_unemployed",
                "would_claim_wic": "would_claim_wic",
            },
            household_variable_map={
                "auto_loan_balance": "auto_loan_balance",
                "net_worth": "net_worth",
                "tenure_type": "tenure_type",
            },
            tax_unit_variable_map={
                "first_home_mortgage_balance": "first_home_mortgage_balance",
                "selected_marketplace_plan_benchmark_ratio": (
                    "selected_marketplace_plan_benchmark_ratio"
                ),
                "takes_up_aca_if_eligible": "takes_up_aca_if_eligible",
                "takes_up_eitc": "takes_up_eitc",
                "would_file_taxes_voluntarily": "would_file_taxes_voluntarily",
            },
            spm_unit_variable_map={
                "spm_unit_pre_subsidy_childcare_expenses": "spm_unit_pre_subsidy_childcare_expenses",
                "spm_unit_tenure_type": "spm_unit_tenure_type",
                "takes_up_snap_if_eligible": "takes_up_snap_if_eligible",
            },
        )

        assert arrays["ssn_card_type"]["2024"].tolist() == [b"CITIZEN"]
        assert arrays["immigration_status_str"]["2024"].tolist() == [b"CITIZEN"]
        assert arrays["auto_loan_balance"]["2024"].tolist() == [0.0]
        assert arrays["net_worth"]["2024"].tolist() == [0]
        assert arrays["tenure_type"]["2024"].tolist() == [b"NONE"]
        assert arrays["takes_up_early_head_start_if_eligible"]["2024"].tolist() == [
            True
        ]
        assert arrays["weeks_unemployed"]["2024"].tolist() == [0]
        assert arrays["would_claim_wic"]["2024"].tolist() == [True]
        assert arrays["first_home_mortgage_balance"]["2024"].tolist() == [0.0]
        assert arrays["selected_marketplace_plan_benchmark_ratio"]["2024"].tolist() == [
            1.0
        ]
        assert arrays["takes_up_aca_if_eligible"]["2024"].tolist() == [True]
        assert arrays["takes_up_eitc"]["2024"].tolist() == [True]
        assert arrays["would_file_taxes_voluntarily"]["2024"].tolist() == [False]
        assert arrays["spm_unit_pre_subsidy_childcare_expenses"]["2024"].tolist() == [0]
        assert arrays["spm_unit_tenure_type"]["2024"].tolist() == [b"RENTER"]
        assert arrays["takes_up_snap_if_eligible"]["2024"].tolist() == [True]

    def test_build_time_period_arrays_normalizes_numeric_tenure_codes(self):
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10, 20, 30],
                    "household_weight": [1.0, 1.0, 1.0],
                    "tenure_type": [0, 1, 2],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "household_id": [10, 20, 30],
                    "spm_unit_id": [100, 200, 300],
                }
            ),
            spm_units=pd.DataFrame(
                {
                    "spm_unit_id": [100, 200, 300],
                    "household_id": [10, 20, 30],
                    "spm_unit_tenure_type": [0, 1, 2],
                }
            ),
        )

        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            household_variable_map={"tenure_type": "tenure_type"},
            spm_unit_variable_map={"spm_unit_tenure_type": "spm_unit_tenure_type"},
        )

        assert arrays["tenure_type"]["2024"].tolist() == [
            b"NONE",
            b"OWNED_WITH_MORTGAGE",
            b"RENTED",
        ]
        assert arrays["spm_unit_tenure_type"]["2024"].tolist() == [
            b"RENTER",
            b"OWNER_WITH_MORTGAGE",
            b"RENTER",
        ]


class TestUSPipelinePolicyEngineTargets:
    def test_build_policyengine_continuous_targets_uses_adapter(self):
        seed_data = pd.DataFrame({"snap": [10.0, 20.0], "income": [1.0, 2.0]})
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())

        class FakeAdapter:
            def compute_targets(self, specs):
                assert len(specs) == 1
                return {"snap_total": 42.0}

        targets = pipeline.build_policyengine_continuous_targets(
            seed_data=seed_data,
            adapter=FakeAdapter(),
            quantity_targets=(
                PolicyEngineUSQuantityTarget(
                    name="snap_total",
                    variable="snap",
                    column="snap",
                    period=2024,
                ),
            ),
        )

        assert targets == {"snap": 42.0}

    def test_build_targets_requires_dataset_when_policyengine_targets_configured(self):
        households = pd.DataFrame(
            {
                "household_id": [1],
                "state_fips": [6],
                "hh_weight": [100.0],
                "tenure": [1],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [10],
                "household_id": [1],
                "age": [34],
                "sex": [1],
                "education": [3],
                "employment_status": [1],
                "income": [55_000.0],
                "snap": [1_200.0],
            }
        )
        config = USMicroplexBuildConfig(
            policyengine_quantity_targets=(
                PolicyEngineUSQuantityTarget(
                    name="snap_total",
                    variable="snap",
                    column="snap",
                    period=2024,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households)

        with pytest.raises(ValueError, match="policyengine_dataset"):
            pipeline.build_targets(seed)


class TestPolicyEngineImportBoundaries:
    def test_policyengine_submodule_import_does_not_require_polars(self):
        src_dir = Path(__file__).resolve().parents[2] / "src"
        code = f"""
import importlib
import sys

sys.path.insert(0, {src_dir.as_posix()!r})
sys.modules['polars'] = None
module = importlib.import_module('microplex_us.policyengine.us')
print(module.__name__)
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "microplex_us.policyengine.us"
