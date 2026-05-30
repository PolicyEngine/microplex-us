"""Tests for PolicyEngine US target comparison helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from microplex.core import EntityType
from microplex.targets import (
    FilterOperator,
    TargetFilter,
    TargetQuery,
    TargetSpec,
    normalize_metric_payload,
)

import microplex_us.policyengine.comparison as comparison_module
from microplex_us.policyengine import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSConstraint,
    PolicyEngineUSDBTargetProvider,
    PolicyEngineUSEntityTableBundle,
    PolicyEngineUSMaterializationError,
    PolicyEngineUSTargetComparisonReport,
    build_policyengine_us_time_period_arrays,
    compare_policyengine_us_target_query_to_baseline,
    compute_policyengine_us_definition_hash,
    evaluate_policyengine_us_target_set,
    load_policyengine_us_entity_tables,
    write_policyengine_us_time_period_dataset,
)
from microplex_us.policyengine.comparison import (
    PolicyEngineUSTargetEvaluation,
    PolicyEngineUSTargetEvaluationReport,
)


def _create_snap_targets_db(path: Path) -> None:
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
        """
    )
    conn.executemany(
        """
        INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
        VALUES (?, ?, ?)
        """,
        [
            (1, compute_policyengine_us_definition_hash(()), None),
            (
                2,
                compute_policyengine_us_definition_hash(
                    (PolicyEngineUSConstraint("state_fips", "==", "06"),),
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
        (2, "state_fips", "==", "06"),
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
            (10, "snap", 2024, 1, 0, 250.0, 1, 0.0, "test", "National SNAP"),
            (11, "snap", 2024, 2, 0, 200.0, 1, 0.0, "test", "California SNAP"),
        ],
    )
    conn.commit()
    conn.close()


def _sample_tables() -> PolicyEngineUSEntityTableBundle:
    return PolicyEngineUSEntityTableBundle(
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
                "tax_unit_id": [100, 100, 200],
                "spm_unit_id": [1000, 1000, 2000],
                "family_id": [5000, 5000, 6000],
                "marital_unit_id": [7000, 7000, 8000],
                "age": [40.0, 10.0, 30.0],
                "employment_income": [30_000.0, 0.0, 20_000.0],
                "employment_income_before_lsr": [30_000.0, 0.0, 20_000.0],
            }
        ),
        tax_units=pd.DataFrame(
            {
                "tax_unit_id": [100, 200],
                "household_id": [1, 2],
                "filing_status": ["JOINT", "SINGLE"],
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


def test_load_policyengine_us_entity_tables_round_trips_written_dataset(tmp_path):
    tables = _sample_tables()
    arrays = build_policyengine_us_time_period_arrays(
        tables,
        period=2024,
        household_variable_map={"state_fips": "state_fips", "snap": "snap"},
        person_variable_map={
            "age": "age",
            "employment_income": "employment_income",
        },
        tax_unit_variable_map={"filing_status": "filing_status"},
    )
    dataset_path = tmp_path / "baseline.h5"
    write_policyengine_us_time_period_dataset(arrays, dataset_path)

    loaded = load_policyengine_us_entity_tables(
        dataset_path,
        period=2024,
        variables=("state_fips", "snap", "age", "employment_income", "filing_status"),
    )

    np.testing.assert_array_equal(
        loaded.households["household_id"].to_numpy(),
        np.array([1, 2]),
    )
    np.testing.assert_allclose(
        loaded.households["household_weight"].to_numpy(dtype=float),
        np.array([2.0, 1.0]),
    )
    np.testing.assert_array_equal(
        loaded.persons["tax_unit_id"].to_numpy(),
        np.array([100, 100, 200]),
    )
    assert loaded.tax_units is not None
    np.testing.assert_array_equal(
        loaded.tax_units["household_id"].to_numpy(),
        np.array([1, 2]),
    )
    assert loaded.tax_units["filing_status"].tolist() == ["JOINT", "SINGLE"]


def test_load_policyengine_us_entity_tables_uses_legacy_contract_entities(tmp_path):
    tables = _sample_tables()
    assert len(tables.households) == len(tables.spm_units)
    tables.spm_units["free_school_meals_reported"] = [1.0, 0.0]
    arrays = build_policyengine_us_time_period_arrays(
        tables,
        period=2024,
        spm_unit_variable_map={
            "free_school_meals_reported": "free_school_meals_reported"
        },
    )
    dataset_path = tmp_path / "baseline_with_equal_household_spm_counts.h5"
    write_policyengine_us_time_period_dataset(arrays, dataset_path)

    loaded = load_policyengine_us_entity_tables(
        dataset_path,
        period=2024,
        variables=("free_school_meals_reported",),
    )

    assert "free_school_meals_reported" not in loaded.households.columns
    assert loaded.spm_units is not None
    assert loaded.spm_units["free_school_meals_reported"].tolist() == [1.0, 0.0]


def test_load_policyengine_us_entity_tables_skips_unsupported_arrays_when_loading_all(
    tmp_path,
):
    tables = _sample_tables()
    arrays = build_policyengine_us_time_period_arrays(
        tables,
        period=2024,
        household_variable_map={"state_fips": "state_fips"},
    )
    dataset_path = tmp_path / "baseline_with_records.h5"
    write_policyengine_us_time_period_dataset(arrays, dataset_path)
    with h5py.File(dataset_path, "a") as handle:
        group = handle.create_group("record_amount")
        group.create_dataset("2024", data=np.array([1.0, 2.0, 3.0, 4.0]))

    loaded = load_policyengine_us_entity_tables(dataset_path, period=2024)

    assert "record_amount" not in loaded.households.columns
    assert loaded.persons is not None
    assert "record_amount" not in loaded.persons.columns


def test_evaluate_policyengine_us_target_set_scores_count_sum_and_mean():
    report = evaluate_policyengine_us_target_set(
        _sample_tables(),
        [
            TargetSpec(
                name="ca_households",
                entity=EntityType.HOUSEHOLD,
                value=2.0,
                period=2024,
                aggregation="count",
                filters=(TargetFilter("state_fips", FilterOperator.EQ, 6),),
            ),
            TargetSpec(
                name="employment_income_before_lsr_total",
                entity=EntityType.PERSON,
                value=80_000.0,
                period=2024,
                measure="employment_income_before_lsr",
                aggregation="sum",
            ),
            TargetSpec(
                name="ca_mean_age",
                entity=EntityType.PERSON,
                value=25.0,
                period=2024,
                measure="age",
                aggregation="mean",
                filters=(TargetFilter("state_fips", FilterOperator.EQ, 6),),
            ),
        ],
        period=2024,
    )

    assert report.supported_target_count == 3
    assert not report.unsupported_targets
    assert report.materialized_variables == ()
    assert report.mean_abs_relative_error == 0.0
    actuals = {evaluation.target.name: evaluation.actual_value for evaluation in report.evaluations}
    assert actuals == {
        "ca_households": 2.0,
        "employment_income_before_lsr_total": 80_000.0,
        "ca_mean_age": 25.0,
    }


def test_policyengine_us_benchmark_metrics_delegate_to_shared_normalization():
    target = TargetSpec(
        name="zero_target_snap",
        entity=EntityType.HOUSEHOLD,
        value=0.0,
        period=2024,
        measure="snap",
        aggregation="sum",
        source="test",
        metadata={"geographic_level": "national"},
    )
    report = PolicyEngineUSTargetEvaluationReport(
        label="candidate",
        period=2024,
        evaluations=[
            PolicyEngineUSTargetEvaluation(
                target=target,
                actual_value=2.5,
            )
        ],
    )

    assert report.benchmark_metrics == [
        normalize_metric_payload(
            {
                "name": "zero_target_snap",
                "estimate": 2.5,
                "target": 0.0,
                "metadata": {
                    "source": "test",
                    "entity": EntityType.HOUSEHOLD.value,
                    "measure": "snap",
                    "aggregation": "sum",
                    "geographic_level": "national",
                },
            }
        )
    ]


def test_evaluate_policyengine_us_target_set_materializes_missing_variables(tmp_path):
    base_tables = _sample_tables()
    tables = PolicyEngineUSEntityTableBundle(
        households=base_tables.households.drop(columns=["snap"]),
        persons=base_tables.persons,
        tax_units=base_tables.tax_units,
        spm_units=base_tables.spm_units,
        families=base_tables.families,
        marital_units=base_tables.marital_units,
    )

    class FakeEntity:
        def __init__(self, key: str):
            self.key = key

    class FakeVariable:
        def __init__(self, entity: FakeEntity, formulas: dict[str, object] | None = None):
            self.entity = entity
            self.formulas = formulas or {}

        def is_input_variable(self) -> bool:
            return not self.formulas

    class FakeTaxBenefitSystem:
        variables = {
            "employment_income": FakeVariable(FakeEntity("person")),
            "state_fips": FakeVariable(FakeEntity("household")),
            "snap": FakeVariable(FakeEntity("household"), formulas={"2024": object()}),
        }

    class FakeSimulation:
        tax_benefit_system = FakeTaxBenefitSystem()

        def __init__(self, dataset, dataset_year=None, **kwargs):
            assert Path(dataset).exists()
            assert dataset_year == 2024
            _ = kwargs

        def calculate(self, variable, period=None, map_to=None):
            assert period == 2024
            assert map_to is None
            if variable == "snap":
                return np.array([120.0, 50.0])
            raise KeyError(variable)

    report = evaluate_policyengine_us_target_set(
        tables,
        [
            TargetSpec(
                name="snap_total",
                entity=EntityType.HOUSEHOLD,
                value=290.0,
                period=2024,
                measure="snap",
                aggregation="sum",
            )
        ],
        period=2024,
        dataset_year=2024,
        simulation_cls=FakeSimulation,
    )

    assert report.materialized_variables == ("snap",)
    assert report.supported_target_count == 1
    assert report.evaluations[0].actual_value == 290.0
    assert report.materialization_failures == {}


def test_evaluate_policyengine_us_target_set_materializes_add_based_variables(tmp_path):
    tables = _sample_tables()

    class FakeEntity:
        def __init__(self, key: str):
            self.key = key

    class FakeVariable:
        def __init__(
            self,
            entity: FakeEntity,
            *,
            adds: list[str] | None = None,
            formulas: dict[str, object] | None = None,
        ):
            self.entity = entity
            self.adds = adds or []
            self.subtracts: list[str] = []
            self.formulas = formulas or {}

        def is_input_variable(self) -> bool:
            return not self.formulas and not self.adds

    class FakeTaxBenefitSystem:
        variables = {
            "employment_income": FakeVariable(
                FakeEntity("person"),
                adds=["employment_income_before_lsr"],
            ),
            "employment_income_before_lsr": FakeVariable(FakeEntity("person")),
        }

    class FakeSimulation:
        tax_benefit_system = FakeTaxBenefitSystem()

        def __init__(self, dataset, dataset_year=None, **kwargs):
            assert Path(dataset).exists()
            assert dataset_year == 2024
            _ = kwargs

        def calculate(self, variable, period=None, map_to=None):
            assert variable == "employment_income"
            assert period == 2024
            assert map_to is None
            return np.array([10.0, 20.0, 30.0])

    report = evaluate_policyengine_us_target_set(
        tables,
        [
            TargetSpec(
                name="employment_income_total",
                entity=EntityType.PERSON,
                value=90.0,
                period=2024,
                measure="employment_income",
                aggregation="sum",
            )
        ],
        period=2024,
        dataset_year=2024,
        simulation_cls=FakeSimulation,
    )

    assert report.materialized_variables == ("employment_income",)
    assert report.evaluations[0].actual_value == 90.0


def test_evaluate_policyengine_us_target_set_skips_failed_materializations(tmp_path):
    base_tables = _sample_tables()
    tables = PolicyEngineUSEntityTableBundle(
        households=base_tables.households.drop(columns=["snap"]),
        persons=base_tables.persons,
        tax_units=base_tables.tax_units,
        spm_units=base_tables.spm_units,
        families=base_tables.families,
        marital_units=base_tables.marital_units,
    )

    class FakeEntity:
        def __init__(self, key: str):
            self.key = key

    class FakeVariable:
        def __init__(self, entity: FakeEntity, formulas: dict[str, object] | None = None):
            self.entity = entity
            self.formulas = formulas or {}

        def is_input_variable(self) -> bool:
            return not self.formulas

    class FakeTaxBenefitSystem:
        variables = {
            "employment_income": FakeVariable(FakeEntity("person")),
            "state_fips": FakeVariable(FakeEntity("household")),
            "snap": FakeVariable(FakeEntity("household"), formulas={"2024": object()}),
            "income_tax": FakeVariable(FakeEntity("person"), formulas={"2024": object()}),
        }

    class FakeSimulation:
        tax_benefit_system = FakeTaxBenefitSystem()

        def __init__(self, dataset, dataset_year=None, **kwargs):
            assert Path(dataset).exists()
            assert dataset_year == 2024
            _ = kwargs

        def calculate(self, variable, period=None, map_to=None):
            assert period == 2024
            assert map_to is None
            if variable == "snap":
                return np.array([120.0, 50.0])
            if variable == "income_tax":
                raise RuntimeError("missing test parameter")
            raise KeyError(variable)

    report = evaluate_policyengine_us_target_set(
        tables,
        [
            TargetSpec(
                name="snap_total",
                entity=EntityType.HOUSEHOLD,
                value=290.0,
                period=2024,
                measure="snap",
                aggregation="sum",
            ),
            TargetSpec(
                name="income_tax_total",
                entity=EntityType.PERSON,
                value=0.0,
                period=2024,
                measure="income_tax",
                aggregation="sum",
            ),
        ],
        period=2024,
        dataset_year=2024,
        simulation_cls=FakeSimulation,
    )

    assert report.materialized_variables == ("snap",)
    assert report.materialization_failures == {
        "income_tax": "RuntimeError: missing test parameter"
    }
    assert report.supported_target_count == 1
    assert report.evaluations[0].target.name == "snap_total"
    assert [target.name for target in report.unsupported_targets] == ["income_tax_total"]


def test_evaluate_policyengine_us_target_set_marks_district_targets_unsupported_when_district_geography_materializes_to_defaults():
    base_tables = _sample_tables()
    tables = PolicyEngineUSEntityTableBundle(
        households=base_tables.households,
        persons=base_tables.persons,
        tax_units=base_tables.tax_units,
        spm_units=base_tables.spm_units,
        families=base_tables.families,
        marital_units=base_tables.marital_units,
    )

    class FakeEntity:
        def __init__(self, key: str):
            self.key = key

    class FakeVariable:
        def __init__(self, entity: FakeEntity, formulas: dict[str, object] | None = None):
            self.entity = entity
            self.formulas = formulas or {}

        def is_input_variable(self) -> bool:
            return not self.formulas

    class FakeTaxBenefitSystem:
        variables = {
            "state_fips": FakeVariable(FakeEntity("household")),
            "congressional_district_geoid": FakeVariable(
                FakeEntity("household"),
                formulas={"2024": object()},
            ),
        }

    class FakeSimulation:
        tax_benefit_system = FakeTaxBenefitSystem()

        def __init__(self, dataset, dataset_year=None, **kwargs):
            assert Path(dataset).exists()
            _ = dataset_year, kwargs

        def calculate(self, variable, period=None, map_to=None):
            assert period == 2024
            assert map_to is None
            if variable == "congressional_district_geoid":
                return np.array([0.0, 0.0])
            raise KeyError(variable)

    report = evaluate_policyengine_us_target_set(
        tables,
        [
            TargetSpec(
                name="district_households",
                entity=EntityType.HOUSEHOLD,
                value=2.0,
                period=2024,
                aggregation="count",
                filters=(
                    TargetFilter(
                        "congressional_district_geoid",
                        FilterOperator.EQ,
                        601,
                    ),
                ),
            )
        ],
        period=2024,
        dataset_year=2024,
        simulation_cls=FakeSimulation,
    )

    assert report.materialized_variables == ("congressional_district_geoid",)
    assert report.supported_target_count == 0
    assert len(report.evaluations) == 0
    assert [target.name for target in report.unsupported_targets] == [
        "district_households"
    ]


def test_evaluate_policyengine_us_target_set_supports_district_targets_with_real_district_geography():
    base_tables = _sample_tables()
    tables = PolicyEngineUSEntityTableBundle(
        households=base_tables.households.assign(
            congressional_district_geoid=np.array([601, 3601])
        ),
        persons=base_tables.persons,
        tax_units=base_tables.tax_units,
        spm_units=base_tables.spm_units,
        families=base_tables.families,
        marital_units=base_tables.marital_units,
    )

    report = evaluate_policyengine_us_target_set(
        tables,
        [
            TargetSpec(
                name="district_households",
                entity=EntityType.HOUSEHOLD,
                value=2.0,
                period=2024,
                aggregation="count",
                filters=(
                    TargetFilter(
                        "congressional_district_geoid",
                        FilterOperator.EQ,
                        601,
                    ),
                ),
            )
        ],
        period=2024,
    )

    assert report.materialized_variables == ()
    assert report.supported_target_count == 1
    assert not report.unsupported_targets
    assert report.evaluations[0].actual_value == 2.0


def test_evaluate_policyengine_us_target_set_raises_on_strict_materialization_failure(
    tmp_path,
):
    base_tables = _sample_tables()
    tables = PolicyEngineUSEntityTableBundle(
        households=base_tables.households.drop(columns=["snap"]),
        persons=base_tables.persons,
        tax_units=base_tables.tax_units,
        spm_units=base_tables.spm_units,
        families=base_tables.families,
        marital_units=base_tables.marital_units,
    )

    class FakeEntity:
        def __init__(self, key: str):
            self.key = key

    class FakeVariable:
        def __init__(self, entity: FakeEntity, formulas: dict[str, object] | None = None):
            self.entity = entity
            self.formulas = formulas or {}

        def is_input_variable(self) -> bool:
            return not self.formulas

    class FakeTaxBenefitSystem:
        variables = {
            "employment_income": FakeVariable(FakeEntity("person")),
            "state_fips": FakeVariable(FakeEntity("household")),
            "snap": FakeVariable(FakeEntity("household"), formulas={"2024": object()}),
        }

    class FailingSimulation:
        tax_benefit_system = FakeTaxBenefitSystem()

        def __init__(self, dataset, dataset_year=None, **kwargs):
            assert Path(dataset).exists()
            assert dataset_year == 2024
            _ = kwargs

        def calculate(self, variable, period=None, map_to=None):
            assert variable == "snap"
            assert period == 2024
            assert map_to is None
            raise RuntimeError("missing test parameter")

    with pytest.raises(PolicyEngineUSMaterializationError, match="candidate"):
        evaluate_policyengine_us_target_set(
            tables,
            [
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=290.0,
                    period=2024,
                    measure="snap",
                    aggregation="sum",
                )
            ],
            period=2024,
            dataset_year=2024,
            simulation_cls=FailingSimulation,
            strict_materialization=True,
        )


def test_evaluate_policyengine_us_target_set_supports_person_to_tax_unit_count_filters():
    report = evaluate_policyengine_us_target_set(
        _sample_tables(),
        [
            TargetSpec(
                name="adult_tax_units",
                entity=EntityType.TAX_UNIT,
                value=1.0,
                period=2024,
                aggregation="count",
                filters=(
                    TargetFilter("age", FilterOperator.GTE, 18),
                ),
            )
        ],
        period=2024,
    )

    assert report.supported_target_count == 1
    assert len(report.evaluations) == 1
    assert report.evaluations[0].target.name == "adult_tax_units"
    assert report.evaluations[0].actual_value == pytest.approx(3.0)
    assert report.unsupported_targets == []


def test_evaluate_policyengine_us_target_set_batches_supported_constraint_compilation(
    monkeypatch,
):
    call_count = 0
    real_compile = (
        comparison_module.compile_supported_policyengine_us_household_linear_constraints
    )

    def record_compile(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(
        comparison_module,
        "compile_supported_policyengine_us_household_linear_constraints",
        record_compile,
    )

    report = evaluate_policyengine_us_target_set(
        _sample_tables(),
        [
            TargetSpec(
                name="ca_households",
                entity=EntityType.HOUSEHOLD,
                value=2.0,
                period=2024,
                aggregation="count",
                filters=(TargetFilter("state_fips", FilterOperator.EQ, 6),),
            ),
            TargetSpec(
                name="snap_total",
                entity=EntityType.HOUSEHOLD,
                value=250.0,
                period=2024,
                measure="snap",
                aggregation="sum",
            ),
        ],
        period=2024,
    )

    assert report.supported_target_count == 2
    assert call_count == 1


def test_compare_policyengine_us_target_query_to_baseline(tmp_path):
    class EmploymentIncomeProvider:
        def load_target_set(self, query=None):
            _ = query
            return [
                TargetSpec(
                    name="employment_income_before_lsr_total",
                    entity=EntityType.PERSON,
                    value=80_000.0,
                    period=2024,
                    measure="employment_income_before_lsr",
                    aggregation="sum",
                )
            ]

    provider = EmploymentIncomeProvider()

    baseline_tables = _sample_tables()
    baseline_arrays = build_policyengine_us_time_period_arrays(
        baseline_tables,
        period=2024,
        household_variable_map={"state_fips": "state_fips"},
        person_variable_map={
            "age": "age",
            "employment_income_before_lsr": "employment_income_before_lsr",
        },
    )
    baseline_path = tmp_path / "enhanced_cps_2024.h5"
    write_policyengine_us_time_period_dataset(baseline_arrays, baseline_path)

    base_candidate = _sample_tables()
    candidate_tables = PolicyEngineUSEntityTableBundle(
        households=base_candidate.households,
        persons=base_candidate.persons.assign(
            employment_income_before_lsr=np.array([20_000.0, 0.0, 20_000.0])
        ),
        tax_units=base_candidate.tax_units,
        spm_units=base_candidate.spm_units,
        families=base_candidate.families,
        marital_units=base_candidate.marital_units,
    )

    report = compare_policyengine_us_target_query_to_baseline(
        candidate_tables,
        provider,
        TargetQuery(
            period=2024,
            provider_filters={"variables": ["employment_income_before_lsr"]},
        ),
        baseline_dataset=baseline_path,
        candidate_label="microplex",
        baseline_label="enhanced_cps",
    )

    assert isinstance(report, PolicyEngineUSTargetComparisonReport)
    assert report.candidate.label == "microplex"
    assert report.baseline is not None
    assert report.baseline.label == "enhanced_cps"
    assert report.candidate.mean_abs_relative_error == pytest.approx(0.25)
    assert report.baseline.mean_abs_relative_error == 0.0
    assert report.mean_abs_relative_error_delta == pytest.approx(0.25)


def test_policyengine_us_comparison_report_uses_common_target_intersection():
    shared_target = TargetSpec(
        name="shared",
        entity=EntityType.HOUSEHOLD,
        value=10.0,
        period=2024,
        measure="snap",
        aggregation="sum",
        source="snap",
    )
    candidate_only_target = TargetSpec(
        name="candidate_only",
        entity=EntityType.HOUSEHOLD,
        value=10.0,
        period=2024,
        measure="snap",
        aggregation="sum",
        source="snap",
    )
    baseline_only_target = TargetSpec(
        name="baseline_only",
        entity=EntityType.HOUSEHOLD,
        value=10.0,
        period=2024,
        measure="snap",
        aggregation="sum",
        source="snap",
    )

    report = PolicyEngineUSTargetComparisonReport(
        candidate=PolicyEngineUSTargetEvaluationReport(
            label="candidate",
            period=2024,
            evaluations=[
                PolicyEngineUSTargetEvaluation(target=shared_target, actual_value=8.0),
                PolicyEngineUSTargetEvaluation(target=candidate_only_target, actual_value=100.0),
            ],
        ),
        baseline=PolicyEngineUSTargetEvaluationReport(
            label="baseline",
            period=2024,
            evaluations=[
                PolicyEngineUSTargetEvaluation(target=shared_target, actual_value=9.0),
                PolicyEngineUSTargetEvaluation(target=baseline_only_target, actual_value=0.0),
            ],
        ),
    )

    assert report.common_target_count == 1
    assert report.mean_abs_relative_error_delta == pytest.approx(0.1)
    assert report.target_win_rate == 0.0


def test_compare_policyengine_us_target_query_to_baseline_evaluates_baseline_first(
    monkeypatch,
    tmp_path,
):
    provider_db = tmp_path / "policy_data.db"
    _create_snap_targets_db(provider_db)
    provider = PolicyEngineUSDBTargetProvider(provider_db)

    baseline_tables = _sample_tables()
    baseline_arrays = build_policyengine_us_time_period_arrays(
        baseline_tables,
        period=2024,
        household_variable_map={"state_fips": "state_fips", "snap": "snap"},
        person_variable_map={"age": "age"},
    )
    baseline_path = tmp_path / "enhanced_cps_2024.h5"
    write_policyengine_us_time_period_dataset(baseline_arrays, baseline_path)

    call_order: list[str] = []

    def fake_evaluate(*args, label: str, **kwargs):
        _ = args, kwargs
        call_order.append(label)
        return type(
            "FakeReport",
            (),
            {
                "label": label,
                "period": 2024,
                "evaluations": [],
                "unsupported_targets": [],
                "materialized_variables": (),
                "materialization_failures": {},
                "mean_abs_relative_error": 0.0,
                "max_abs_relative_error": 0.0,
                "supported_target_count": 0,
            },
        )()

    monkeypatch.setattr(
        "microplex_us.policyengine.comparison.evaluate_policyengine_us_target_set",
        fake_evaluate,
    )

    compare_policyengine_us_target_query_to_baseline(
        _sample_tables(),
        provider,
        TargetQuery(period=2024, provider_filters={"variables": ["snap"]}),
        baseline_dataset=baseline_path,
        candidate_label="microplex",
        baseline_label="enhanced_cps",
    )

    assert call_order == ["enhanced_cps", "microplex"]


def test_compare_policyengine_us_target_query_to_baseline_reuses_cache(monkeypatch):
    class CountingProvider:
        def __init__(self):
            self.load_calls = 0

        def load_target_set(self, query=None):
            _ = query
            self.load_calls += 1
            return [
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=250.0,
                    period=2024,
                    measure="snap",
                    aggregation="sum",
                )
            ]

    provider = CountingProvider()
    cache = PolicyEngineUSComparisonCache()
    load_counts = {"tables": 0}
    eval_counts = {"baseline": 0, "microplex": 0}

    def fake_load_policyengine_us_entity_tables(dataset, period):
        _ = dataset
        _ = period
        load_counts["tables"] += 1
        return _sample_tables()

    def fake_evaluate(*args, label: str, **kwargs):
        _ = args, kwargs
        eval_counts[label] = eval_counts.get(label, 0) + 1
        return type(
            "FakeReport",
            (),
            {
                "label": label,
                "period": 2024,
                "evaluations": [],
                "unsupported_targets": [],
                "materialized_variables": (),
                "materialization_failures": {},
                "mean_abs_relative_error": 0.0,
                "max_abs_relative_error": 0.0,
                "supported_target_count": 0,
            },
        )()

    monkeypatch.setattr(
        "microplex_us.policyengine.comparison.load_policyengine_us_entity_tables",
        fake_load_policyengine_us_entity_tables,
    )
    monkeypatch.setattr(
        "microplex_us.policyengine.comparison.evaluate_policyengine_us_target_set",
        fake_evaluate,
    )

    for _ in range(2):
        compare_policyengine_us_target_query_to_baseline(
            _sample_tables(),
            provider,
            TargetQuery(period=2024, names=("snap_total",)),
            baseline_dataset="/tmp/enhanced_cps_2024.h5",
            baseline_label="baseline",
            cache=cache,
        )

    assert provider.load_calls == 1
    assert load_counts["tables"] == 1
    assert eval_counts["baseline"] == 1
    assert eval_counts["microplex"] == 2


def test_compare_policyengine_us_target_query_to_baseline_raises_on_strict_materialization_failure(
    tmp_path,
):
    provider = PolicyEngineUSDBTargetProvider(tmp_path / "policy_data.db")
    _create_snap_targets_db(provider.db_path)

    baseline_tables = _sample_tables()
    baseline_arrays = build_policyengine_us_time_period_arrays(
        PolicyEngineUSEntityTableBundle(
            households=baseline_tables.households.drop(columns=["snap"]),
            persons=baseline_tables.persons,
            tax_units=baseline_tables.tax_units,
            spm_units=baseline_tables.spm_units,
            families=baseline_tables.families,
            marital_units=baseline_tables.marital_units,
        ),
        period=2024,
        household_variable_map={"state_fips": "state_fips"},
        person_variable_map={"age": "age"},
    )
    baseline_path = tmp_path / "baseline_missing_snap.h5"
    write_policyengine_us_time_period_dataset(baseline_arrays, baseline_path)

    class FakeEntity:
        def __init__(self, key: str):
            self.key = key

    class FakeVariable:
        def __init__(self, entity: FakeEntity, formulas: dict[str, object] | None = None):
            self.entity = entity
            self.formulas = formulas or {}

        def is_input_variable(self) -> bool:
            return not self.formulas

    class FakeTaxBenefitSystem:
        variables = {
            "state_fips": FakeVariable(FakeEntity("household")),
            "snap": FakeVariable(FakeEntity("household"), formulas={"2024": object()}),
        }

    class FailingSimulation:
        tax_benefit_system = FakeTaxBenefitSystem()

        def __init__(self, dataset, dataset_year=None, **kwargs):
            assert Path(dataset).exists()
            assert dataset_year == 2024
            _ = kwargs

        def calculate(self, variable, period=None, map_to=None):
            assert variable == "snap"
            assert period == 2024
            assert map_to is None
            raise RuntimeError("snap materialization unavailable")

    with pytest.raises(PolicyEngineUSMaterializationError, match="enhanced_cps"):
        compare_policyengine_us_target_query_to_baseline(
            _sample_tables(),
            provider,
            TargetQuery(period=2024, provider_filters={"variables": ["snap"]}),
            baseline_dataset=baseline_path,
            dataset_year=2024,
            simulation_cls=FailingSimulation,
            candidate_label="microplex",
            baseline_label="enhanced_cps",
            strict_materialization=True,
        )
