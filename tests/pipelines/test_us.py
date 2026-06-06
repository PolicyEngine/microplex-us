"""Tests for the US microplex pipeline library."""

import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from microplex.calibration import LinearConstraint
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceDescriptor,
    SourceQuery,
    SourceVariableCapability,
    StaticSourceProvider,
    TimeStructure,
)
from microplex.targets import TargetAggregation, TargetQuery, TargetSpec

import microplex_us.pipelines.us as us_pipeline_module
from microplex_us.geography import BlockGeography
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexPipeline,
    USMicroplexTargets,
    _attach_household_census_geographies,
    _normalize_policyengine_constraints_for_microcalibrate,
    _policyengine_target_loss_geography_key,
    _select_feasible_policyengine_calibration_constraints,
    _select_policyengine_deferred_stage_constraints,
    _select_ssi_takeup_by_age_amount,
    _summarize_policyengine_target_fit_report,
    _summarize_weight_diagnostics,
    build_us_microplex,
)
from microplex_us.policyengine.comparison import (
    PolicyEngineUSTargetEvaluation,
    PolicyEngineUSTargetEvaluationReport,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSConstraint,
    PolicyEngineUSEntityTableBundle,
    PolicyEngineUSVariableBinding,
    PolicyEngineUSVariableMaterializationResult,
    build_policyengine_us_export_variable_maps,
    compute_policyengine_us_definition_hash,
)


def _create_policyengine_calibration_db(path) -> None:
    national_constraints: tuple[PolicyEngineUSConstraint, ...] = ()
    california_constraints = (PolicyEngineUSConstraint("state_fips", "==", "6"),)
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
            (
                1,
                compute_policyengine_us_definition_hash(national_constraints),
                None,
            ),
            (
                2,
                compute_policyengine_us_definition_hash(
                    california_constraints,
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
        [(2, "state_fips", "==", "6")],
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
            (1, "household_count", 2024, 1, 0, 450.0, 1, None, "test", "national"),
            (2, "household_count", 2024, 2, 0, 225.0, 1, None, "test", "ca"),
        ],
    )
    conn.commit()
    conn.close()


def _create_policyengine_calibration_db_with_unsupported_target(path) -> None:
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
            (2, compute_policyengine_us_definition_hash(()), None),
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
            (
                10,
                "household_count",
                2024,
                1,
                0,
                450.0,
                1,
                0.0,
                "test",
                "All households",
            ),
            (
                11,
                "income_tax",
                2024,
                2,
                0,
                0.0,
                1,
                0.0,
                "test",
                "Income tax total",
            ),
        ],
    )
    conn.commit()
    conn.close()


def test_select_ssi_takeup_by_age_amount_matches_reported_age_group_amounts():
    selected, summary = _select_ssi_takeup_by_age_amount(
        person_ids=pd.Series([1, 2, 3, 4]),
        ages=pd.Series([70, 70, 40, 40]),
        weights=pd.Series([1.0, 1.0, 1.0, 1.0]),
        reported_ssi=pd.Series([100.0, 0.0, 100.0, 0.0]),
        full_takeup_ssi=pd.Series([80.0, 20.0, 20.0, 80.0]),
    )

    assert selected.tolist() == [True, True, True, True]
    assert summary["reported_amount"] == 200.0
    assert summary["selected_amount"] == 200.0
    assert summary["groups"]["aged"]["selected_amount"] == 100.0
    assert summary["groups"]["under65"]["selected_amount"] == 100.0


class TestUSMicroplexBuildConfig:
    """Test pipeline configuration."""

    def test_defaults(self):
        config = USMicroplexBuildConfig()

        assert config.synthesis_backend == "synthesizer"
        assert config.calibration_backend == "entropy"
        assert config.n_synthetic == 100_000
        assert config.random_seed == 42
        assert config.donor_imputer_authoritative_override_variables == ()
        assert (
            config.policyengine_calibration_deferred_stage_min_active_households == ()
        )
        assert config.policyengine_calibration_deferred_stage_max_constraints == 24
        assert (
            config.policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error
            is None
        )
        assert config.policyengine_calibration_deferred_stage_top_family_count == 8
        assert config.policyengine_calibration_deferred_stage_top_geography_count == 8
        assert config.dependent_tax_leaf_soft_cap_multiplier is None
        assert config.dependent_tax_leaf_soft_cap_base_variables == (
            "employment_income",
            "wage_income",
            "self_employment_income",
        )
        assert config.dependent_tax_leaf_soft_cap_variables == (
            "taxable_interest_income",
            "tax_exempt_interest_income",
            "taxable_pension_income",
            "dividend_income",
            "qualified_dividend_income",
            "non_qualified_dividend_income",
            "partnership_s_corp_income",
            "rental_income",
        )

    def test_custom_values(self):
        config = USMicroplexBuildConfig(
            n_synthetic=250,
            synthesis_backend="seed",
            calibration_backend="ipf",
            synthesizer_epochs=12,
            policyengine_selection_backend="pe_native_loss",
            policyengine_selection_household_budget=500,
            policyengine_selection_state_floor=25,
            policyengine_selection_max_iter=750,
            policyengine_selection_tol=1e-7,
            policyengine_selection_l2_penalty=1e-5,
            policyengine_selection_target_total_weight=150_000_000.0,
        )

        assert config.n_synthetic == 250
        assert config.synthesis_backend == "seed"
        assert config.calibration_backend == "ipf"
        assert config.synthesizer_epochs == 12
        assert config.policyengine_selection_backend == "pe_native_loss"
        assert config.policyengine_selection_household_budget == 500
        assert config.policyengine_selection_state_floor == 25
        assert config.policyengine_selection_max_iter == 750
        assert config.policyengine_selection_tol == 1e-7
        assert config.policyengine_selection_l2_penalty == 1e-5
        assert config.policyengine_selection_target_total_weight == 150_000_000.0
        assert config.policyengine_oracle_relative_error_cap == 10.0

    def test_can_opt_into_authoritative_donor_overrides(self):
        config = USMicroplexBuildConfig(
            donor_imputer_authoritative_override_variables=(
                "self_employment_income",
                "rental_income",
            )
        )

        assert config.donor_imputer_authoritative_override_variables == (
            "self_employment_income",
            "rental_income",
        )

    def test_puf_support_clone_requires_seed_backend_and_no_household_selection(self):
        with pytest.raises(ValueError, match="synthesis_backend='seed'"):
            USMicroplexBuildConfig(puf_support_clone_enabled=True)

        with pytest.raises(ValueError, match="policyengine_selection_household_budget"):
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
                policyengine_selection_household_budget=10,
            )

    def test_initialize_puf_support_clone_calibration_weights_reserves_clone_share(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
                puf_support_clone_prior_weight_share=0.05,
            )
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": ["h1", "h2", "h1__puf_clone", "h2__puf_clone"],
                    "household_weight": [100.0, 200.0, 0.0, 0.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2, 3, 4],
                    "household_id": [
                        "h1",
                        "h2",
                        "h1__puf_clone",
                        "h2__puf_clone",
                    ],
                    "person_is_puf_clone": [0.0, 0.0, 1.0, 1.0],
                    "weight": [100.0, 200.0, 0.0, 0.0],
                }
            ),
            tax_units=pd.DataFrame(),
            spm_units=pd.DataFrame(),
            families=pd.DataFrame(),
            marital_units=pd.DataFrame(),
        )

        updated_tables, summary = pipeline._initialize_puf_clone_calibration_weights(
            tables
        )

        assert summary["applied"] is True
        assert summary["clone_household_count"] == 2
        assert summary["clone_prior_weight_share"] == pytest.approx(0.05)
        assert summary["pre_clone_weight_sum"] == 0.0
        assert summary["pre_clone_original_weight_sum"] == pytest.approx(300.0)
        assert summary["clone_prior_total_weight"] == pytest.approx(300.0 * 0.05 / 0.95)
        assert summary["clone_prior_household_weight"] == pytest.approx(
            300.0 * 0.05 / 0.95 / 2
        )
        assert updated_tables.households["household_weight"].tolist() == [
            pytest.approx(100.0),
            pytest.approx(200.0),
            pytest.approx(300.0 * 0.05 / 0.95 / 2),
            pytest.approx(300.0 * 0.05 / 0.95 / 2),
        ]
        assert updated_tables.persons["weight"].tolist() == [100.0, 200.0, 0.0, 0.0]

    def test_initialize_puf_support_clone_calibration_weights_skips_no_calibration(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                calibration_backend="none",
                puf_support_clone_enabled=True,
            )
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "household_weight": [100.0, 0.0],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "household_id": [1, 2],
                    "person_is_puf_clone": [0.0, 1.0],
                    "weight": [100.0, 0.0],
                }
            ),
            tax_units=pd.DataFrame(),
            spm_units=pd.DataFrame(),
            families=pd.DataFrame(),
            marital_units=pd.DataFrame(),
        )

        updated_tables, summary = pipeline._initialize_puf_clone_calibration_weights(
            tables
        )

        assert summary["applied"] is False
        assert summary["reason"] == "calibration_backend_none"
        assert updated_tables.households["household_weight"].tolist() == [100.0, 0.0]

    def test_rejects_conflicting_policyengine_weight_rescale_modes(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            USMicroplexBuildConfig(
                policyengine_calibration_rescale_to_input_weight_sum=True,
                policyengine_calibration_rescale_to_target_total_weight=True,
                policyengine_calibration_target_total_weight=150_000_000.0,
            )

    def test_rejects_target_rescale_without_target_total_weight(self):
        with pytest.raises(
            ValueError,
            match="requires policyengine_calibration_target_total_weight",
        ):
            USMicroplexBuildConfig(
                policyengine_calibration_rescale_to_target_total_weight=True
            )

    def test_rejects_nonpositive_oracle_relative_error_cap(self):
        with pytest.raises(ValueError, match="must be positive"):
            USMicroplexBuildConfig(policyengine_oracle_relative_error_cap=0.0)

    def test_rejects_nonpositive_deferred_stage_support_floor(self):
        with pytest.raises(ValueError, match="must contain only positive"):
            USMicroplexBuildConfig(
                policyengine_calibration_deferred_stage_min_active_households=(0,)
            )

    def test_rejects_negative_deferred_stage_family_focus_limit(self):
        with pytest.raises(ValueError, match="must be nonnegative"):
            USMicroplexBuildConfig(
                policyengine_calibration_deferred_stage_top_family_count=-1
            )

    def test_rejects_negative_deferred_stage_geography_focus_limit(self):
        with pytest.raises(ValueError, match="must be nonnegative"):
            USMicroplexBuildConfig(
                policyengine_calibration_deferred_stage_top_geography_count=-1
            )

    def test_rejects_nonpositive_deferred_stage_constraint_cap(self):
        with pytest.raises(ValueError, match="must be positive"):
            USMicroplexBuildConfig(
                policyengine_calibration_deferred_stage_max_constraints=0
            )

    def test_rejects_nonpositive_deferred_stage_trigger_threshold(self):
        with pytest.raises(ValueError, match="must be positive"):
            USMicroplexBuildConfig(
                policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error=0.0
            )

    def test_rejects_negative_dependent_tax_leaf_soft_cap_multiplier(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            USMicroplexBuildConfig(dependent_tax_leaf_soft_cap_multiplier=-0.01)


def test_apply_dependent_tax_leaf_soft_caps_only_for_dependents():
    config = USMicroplexBuildConfig(dependent_tax_leaf_soft_cap_multiplier=0.5)
    pipeline = USMicroplexPipeline(config)
    seed_data = pd.DataFrame(
        {
            "is_tax_unit_dependent": [1, 0],
            "employment_income": [100.0, 100.0],
            "wage_income": [0.0, 20.0],
            "self_employment_income": [0.0, 0.0],
            "taxable_interest_income": [80.0, 80.0],
            "rental_income": [120.0, 120.0],
        }
    )

    updated = pipeline._apply_dependent_tax_leaf_soft_caps(seed_data.copy())

    assert updated.loc[0, "taxable_interest_income"] == pytest.approx(50.0)
    assert updated.loc[0, "rental_income"] == pytest.approx(50.0)
    assert updated.loc[1, "taxable_interest_income"] == pytest.approx(80.0)
    assert updated.loc[1, "rental_income"] == pytest.approx(120.0)


def test_summarize_policyengine_target_fit_report_caps_relative_error():
    target = TargetSpec(
        name="tiny_target",
        entity=EntityType.HOUSEHOLD,
        period=2024,
        measure="state_income_tax",
        aggregation=TargetAggregation.SUM,
        value=0.0,
        source="test",
        metadata={},
    )
    report = PolicyEngineUSTargetEvaluationReport(
        label="candidate",
        period=2024,
        evaluations=[
            PolicyEngineUSTargetEvaluation(target=target, actual_value=25.0),
        ],
    )

    summary = _summarize_policyengine_target_fit_report(
        report,
        target_count=1,
        relative_error_cap=10.0,
    )

    assert summary["mean_abs_relative_error"] == pytest.approx(25.0)
    assert summary["capped_mean_abs_relative_error"] == pytest.approx(10.0)
    assert summary["relative_error_cap"] == pytest.approx(10.0)


def test_summarize_policyengine_target_fit_report_penalizes_unsupported_targets():
    supported_target = TargetSpec(
        name="supported_target",
        entity=EntityType.HOUSEHOLD,
        period=2024,
        aggregation=TargetAggregation.COUNT,
        value=100.0,
        source="test",
        metadata={},
    )
    unsupported_target = TargetSpec(
        name="unsupported_target",
        entity=EntityType.TAX_UNIT,
        period=2024,
        aggregation=TargetAggregation.COUNT,
        value=1.0,
        source="test",
        metadata={},
    )
    report = PolicyEngineUSTargetEvaluationReport(
        label="candidate",
        period=2024,
        evaluations=[
            PolicyEngineUSTargetEvaluation(
                target=supported_target,
                actual_value=100.0,
            ),
        ],
        unsupported_targets=[unsupported_target],
    )

    summary = _summarize_policyengine_target_fit_report(
        report,
        target_count=2,
        relative_error_cap=10.0,
    )

    assert summary["supported_only_mean_abs_relative_error"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert summary["unsupported_target_error_penalty"] == pytest.approx(10.0)
    assert summary["mean_abs_relative_error"] == pytest.approx(5.0)
    assert summary["capped_mean_abs_relative_error"] == pytest.approx(5.0)


def test_select_policyengine_deferred_stage_constraints_prioritizes_target_level_loss():
    def _target(name: str) -> TargetSpec:
        return TargetSpec(
            name=name,
            entity=EntityType.HOUSEHOLD,
            period=2024,
            aggregation=TargetAggregation.COUNT,
            value=100.0,
            source="test",
            metadata={
                "variable": "household_count",
                "geo_level": "state",
                "geographic_id": "6",
            },
        )

    compiled_targets = [
        _target("a_low"),
        _target("m_mid"),
        _target("z_high"),
    ]
    compiled_constraints = (
        SimpleNamespace(coefficients=np.array([1.0] * 12)),
        SimpleNamespace(coefficients=np.array([1.0] * 12)),
        SimpleNamespace(coefficients=np.array([1.0] * 12)),
    )
    target_ledger = [
        {
            "target_name": target.name,
            "stage": "solve_later",
            "variable": "household_count",
            "domain_variable": "",
            "geo_level": "state",
            "geographic_id": "6",
        }
        for target in compiled_targets
    ]
    deferred_oracle_loss = {
        "family_ranking": [
            {
                "group": "household_count",
                "capped_loss_share": 1.0,
                "capped_sum_abs_relative_error": 10.0,
            }
        ],
        "geography_ranking": [
            {
                "group": "state:CA",
                "capped_loss_share": 1.0,
                "capped_sum_abs_relative_error": 10.0,
            }
        ],
    }

    selected_targets, _, metadata = _select_policyengine_deferred_stage_constraints(
        compiled_targets=compiled_targets,
        compiled_constraints=compiled_constraints,
        target_ledger=target_ledger,
        deferred_oracle_loss=deferred_oracle_loss,
        deferred_target_priority_lookup={
            "a_low": 0.5,
            "m_mid": 1.0,
            "z_high": 4.0,
        },
        selected_target_names=set(),
        household_count=100,
        min_active_households=10,
        max_constraints=1,
        max_constraints_per_household=None,
        top_family_count=1,
        top_geography_count=1,
    )

    assert [target.name for target in selected_targets] == ["z_high"]
    assert metadata["target_error_priority_available"] is True
    assert metadata["n_focus_eligible_constraints"] == 3


class TestUSMicroplexPipeline:
    """Test orchestration for US microplex builds."""

    def test_policyengine_target_loss_geography_key_normalizes_state_fips(self):
        assert (
            _policyengine_target_loss_geography_key(
                {"geo_level": "state", "geographic_id": "1"}
            )
            == "state:AL"
        )
        assert (
            _policyengine_target_loss_geography_key(
                {"geo_level": "state", "geographic_id": "01"}
            )
            == "state:AL"
        )
        assert (
            _policyengine_target_loss_geography_key(
                {"geo_level": "national", "geographic_id": "usa"}
            )
            == "national:US"
        )

    @pytest.fixture
    def households(self):
        return pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "state_fips": [6, 36, 48],
                "county_fips": [6037, 36061, 48201],
                "hh_weight": [100.0, 150.0, 200.0],
                "tenure": [1, 2, 1],
            }
        )

    @pytest.fixture
    def persons(self):
        return pd.DataFrame(
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

    def test_attach_household_census_geographies_from_state_county(self):
        geography = BlockGeography.from_data(
            pd.DataFrame(
                {
                    "geoid": ["060010201001000", "360610101001000"],
                    "state_fips": ["06", "36"],
                    "county": ["001", "061"],
                    "county_fips": ["06001", "36061"],
                    "tract": ["020100", "010100"],
                    "tract_geoid": ["06001020100", "36061010100"],
                    "cd_id": ["CA-01", "NY-12"],
                    "prob": [1.0, 1.0],
                }
            )
        )
        households = pd.DataFrame(
            {
                "household_id": [10, 20],
                "state_fips": [6, 36],
                "county_fips": [1, 61],
            },
            index=[100, 200],
        )

        result = _attach_household_census_geographies(
            households,
            seed=0,
            geography=geography,
        ).sort_values("household_id")

        assert result["block_geoid"].tolist() == [
            "060010201001000",
            "360610101001000",
        ]
        assert result["county_fips"].tolist() == ["06001", "36061"]
        assert result["tract_geoid"].tolist() == ["06001020100", "36061010100"]
        assert result["congressional_district_geoid"].tolist() == [601, 3612]

    def test_prepare_seed_data(self, persons, households):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())

        seed = pipeline.prepare_seed_data(persons, households)

        assert len(seed) == len(persons)
        assert "state" in seed.columns
        assert "county_fips" in seed.columns
        assert "block_geoid" in seed.columns
        assert "tract_geoid" in seed.columns
        assert "congressional_district_geoid" in seed.columns
        assert "age_group" in seed.columns
        assert "income_bracket" in seed.columns
        assert set(seed["state"]) == {"CA", "NY", "TX"}
        assert set(seed["age_group"].astype(str)) == {"0-17", "18-34", "35-54", "65+"}

    def test_prepare_seed_data_normalizes_social_security_components(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        households = pd.DataFrame(
            {
                "household_id": [1],
                "state_fips": [6],
                "county_fips": [6037],
                "hh_weight": [100.0],
                "tenure": [1],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [10],
                "household_id": [1],
                "age": [68],
                "sex": [1],
                "education": [3],
                "employment_status": [2],
                "income": [1_200.0],
                "gross_social_security": [1_200.0],
                "social_security_disability": [200.0],
            }
        )

        seed = pipeline.prepare_seed_data(persons, households)
        row = seed.iloc[0]

        assert row["social_security"] == 1_200.0
        assert row["social_security_retirement"] == 0.0
        assert row["social_security_disability"] == 200.0
        assert row["social_security_survivors"] == 0.0
        assert row["social_security_dependents"] == 0.0
        assert row["social_security_unclassified"] == 1_000.0

    def test_build_targets(self, persons, households):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        seed = pipeline.prepare_seed_data(persons, households)

        targets = pipeline.build_targets(seed)

        assert isinstance(targets, USMicroplexTargets)
        assert set(targets.marginal.keys()) == {"state", "age_group", "income_bracket"}
        assert targets.marginal["state"]["CA"] == 200.0
        assert targets.marginal["state"]["NY"] == 300.0
        assert targets.marginal["state"]["TX"] == 400.0
        expected_income = float((seed["hh_weight"] * seed["income"]).sum())
        assert targets.continuous["income"] == expected_income

    def test_build_with_bootstrap_backend(self, persons, households):
        config = USMicroplexBuildConfig(
            n_synthetic=12,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            random_seed=7,
        )
        result = build_us_microplex(persons, households, config)

        assert len(result.seed_data) == len(persons)
        assert result.synthetic_data["household_id"].nunique() == 12
        assert result.calibrated_data["household_id"].nunique() == 12
        assert len(result.synthetic_data) > 12
        assert len(result.calibrated_data) > 12
        assert "weight" in result.calibrated_data.columns
        assert result.calibration_summary["max_error"] < 0.05
        assert result.synthesizer is None
        assert result.policyengine_tables is not None
        assert result.source_frame is not None
        assert result.fusion_plan is not None
        assert len(result.policyengine_tables.households) == 12
        assert len(result.policyengine_tables.persons) == len(result.calibrated_data)
        assert len(result.policyengine_tables.tax_units) > 0
        assert len(result.policyengine_tables.spm_units) > 0
        assert len(result.policyengine_tables.families) > 0
        assert len(result.policyengine_tables.marital_units) > 0
        assert result.synthesis_metadata["bootstrap_strata_columns"] == []

    def test_bootstrap_infers_state_strata_from_target_scope(self, persons, households):
        config = USMicroplexBuildConfig(
            synthesis_backend="bootstrap",
            policyengine_calibration_target_geo_levels=("state",),
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households)

        assert pipeline._resolve_bootstrap_strata_columns(seed) == ("state_fips",)

    def test_bootstrap_preserves_state_support_when_state_targets_are_requested(self):
        seed = pd.DataFrame(
            {
                "household_id": [0, 1],
                "person_id": [0, 1],
                "hh_weight": [1000.0, 1.0],
                "state_fips": [6, 36],
                "age": [40, 41],
                "sex": [1, 2],
                "education": [3, 3],
                "employment_status": [1, 1],
                "tenure": [1, 1],
                "income": [50_000.0, 60_000.0],
            }
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                random_seed=1,
                policyengine_calibration_target_geo_levels=("state",),
            )
        )

        synthetic = pipeline._synthesize_bootstrap(
            seed,
            initial_weight=1.0,
            strata_columns=pipeline._resolve_bootstrap_strata_columns(seed),
        )

        assert synthetic["state_fips"].nunique() == 2

    def test_bootstrap_explicit_missing_strata_column_raises(self, persons, households):
        config = USMicroplexBuildConfig(
            synthesis_backend="bootstrap",
            bootstrap_strata_columns=("missing_geo",),
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households)

        with pytest.raises(ValueError, match="bootstrap_strata_columns"):
            pipeline._resolve_bootstrap_strata_columns(seed)

    def test_build_with_synthesizer_backend(self, persons, households):
        config = USMicroplexBuildConfig(
            n_synthetic=10,
            synthesis_backend="synthesizer",
            calibration_backend="entropy",
            synthesizer_epochs=5,
            synthesizer_n_layers=2,
            synthesizer_hidden_dim=16,
            random_seed=11,
        )
        result = build_us_microplex(persons, households, config)

        assert len(result.synthetic_data) == 10
        assert result.synthesizer is not None
        assert result.synthesis_metadata["backend"] == "synthesizer"
        assert result.synthesis_metadata["condition_vars"] == [
            "age",
            "sex",
            "education",
            "employment_status",
            "state_fips",
            "tenure",
        ]
        assert result.synthesis_metadata["target_vars"] == ["income"]
        assert result.fusion_plan is not None
        assert set(result.fusion_plan.output_entities) == {
            EntityType.HOUSEHOLD,
            EntityType.PERSON,
        }
        assert (result.synthetic_data["income"] >= 0).all()
        assert result.policyengine_tables is not None

    def test_build_policyengine_entity_tables(self, persons, households):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        seed = pipeline.prepare_seed_data(persons, households)
        synthetic = pipeline._finalize_synthetic_population(seed, initial_weight=1.0)

        tables = pipeline.build_policyengine_entity_tables(synthetic)

        assert set(tables.households.columns) >= {"household_id", "household_weight"}
        assert set(tables.persons.columns) >= {
            "person_id",
            "household_id",
            "tax_unit_id",
            "spm_unit_id",
            "family_id",
            "marital_unit_id",
        }
        assert set(tables.tax_units.columns) >= {"tax_unit_id", "household_id"}
        assert set(tables.spm_units.columns) >= {"spm_unit_id", "household_id"}
        assert set(tables.families.columns) >= {"family_id", "household_id"}
        assert set(tables.marital_units.columns) >= {"marital_unit_id", "household_id"}
        assert tables.persons["tax_unit_id"].notna().all()
        assert tables.persons["spm_unit_id"].notna().all()
        assert tables.persons["family_id"].notna().all()
        assert tables.persons["marital_unit_id"].notna().all()
        assert set(tables.tax_units["filing_status"]).issubset(
            {"SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"}
        )

    def test_build_policyengine_entity_tables_preserves_household_contract_inputs(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 20],
                "weight": [1.0, 1.0, 2.0],
                "age": [45, 12, 70],
                "income": [60_000.0, 0.0, 25_000.0],
                "relationship_to_head": [0, 2, 0],
                "state_fips": [6, 6, 36],
                "tenure": [1, 1, 2],
                "tenure_type": ["OWNER_WITH_MORTGAGE", "OWNER_WITH_MORTGAGE", "RENTER"],
                "net_worth": [300_000.0, 300_000.0, 50_000.0],
                "auto_loan_balance": [12_000.0, 12_000.0, 0.0],
                "auto_loan_interest": [600.0, 600.0, 0.0],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        households = tables.households.sort_values("household_id").reset_index(
            drop=True
        )

        assert households["tenure_type"].tolist() == [
            "OWNER_WITH_MORTGAGE",
            "RENTER",
        ]
        assert households["net_worth"].tolist() == [300_000.0, 50_000.0]
        assert households["auto_loan_balance"].tolist() == [12_000.0, 0.0]
        assert households["auto_loan_interest"].tolist() == [600.0, 0.0]
        assert tables.spm_units.sort_values("household_id")[
            "spm_unit_tenure_type"
        ].tolist() == [
            "OWNER_WITH_MORTGAGE",
            "RENTER",
        ]

    def test_build_policyengine_entity_tables_preserves_spm_source_inputs(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 20],
                "spm_unit_id": [100, 100, 200],
                "weight": [1.0, 1.0, 2.0],
                "age": [45, 12, 70],
                "income": [60_000.0, 0.0, 25_000.0],
                "relationship_to_head": [0, 2, 0],
                "receives_housing_assistance": [False, True, False],
                "takes_up_housing_assistance_if_eligible": [False, True, False],
                "takes_up_snap_if_eligible": [False, True, False],
                "spm_unit_energy_subsidy": [90.0, 90.0, 0.0],
                "spm_unit_pre_subsidy_childcare_expenses": [1500.0, 1500.0, 0.0],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        spm_units = tables.spm_units.sort_values("household_id").reset_index(drop=True)

        assert len(spm_units) == 2
        assert spm_units["receives_housing_assistance"].tolist() == [True, False]
        assert spm_units["takes_up_housing_assistance_if_eligible"].tolist() == [
            True,
            False,
        ]
        assert spm_units["takes_up_snap_if_eligible"].tolist() == [True, False]
        assert spm_units["spm_unit_energy_subsidy"].tolist() == [90.0, 0.0]
        assert spm_units["spm_unit_pre_subsidy_childcare_expenses"].tolist() == [
            1500.0,
            0.0,
        ]

    def test_build_policyengine_entity_tables_adds_deterministic_snap_takeup(
        self,
        monkeypatch,
    ):
        calls: list[tuple[str, int]] = []

        def fake_load_takeup_rate(variable_name: str, year: int) -> float:
            calls.append((variable_name, year))
            return 0.0

        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_takeup_rate",
            fake_load_takeup_rate,
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 20],
                "spm_unit_id": [100, 100, 200],
                "weight": [1.0, 1.0, 2.0],
                "age": [45, 12, 70],
                "income": [60_000.0, 0.0, 25_000.0],
                "relationship_to_head": [0, 2, 0],
                "takes_up_aca_if_eligible": [True, True, True],
                "would_file_taxes_voluntarily": [False, False, False],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        spm_units = tables.spm_units.sort_values("household_id").reset_index(drop=True)

        assert calls == [
            ("head_start", 2024),
            ("early_head_start", 2024),
            ("dc_ptc", 2024),
            ("snap", 2024),
            ("tanf", 2024),
        ]
        assert spm_units["takes_up_snap_if_eligible"].tolist() == [False, False]

    def test_build_policyengine_entity_tables_recomputes_child_count_contract_inputs(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "household_id": [10, 10, 10, 20],
                "weight": [1.0, 1.0, 1.0, 2.0],
                "age": [45, 4, 17, 18],
                "income": [60_000.0, 0.0, 0.0, 25_000.0],
                "relationship_to_head": [0, 2, 2, 0],
                "count_under_18": [99, 99, 99, 99],
                "count_under_6": [99, 99, 99, 99],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        persons = tables.persons.sort_values("person_id").reset_index(drop=True)

        assert persons["count_under_18"].tolist() == [2, 2, 2, 0]
        assert persons["count_under_6"].tolist() == [1, 1, 1, 0]

    def test_build_policyengine_entity_tables_uses_household_level_spm_fallback(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "household_id": [10, 10, 10, 10],
                "weight": [1.0, 1.0, 1.0, 1.0],
                "age": [45, 43, 12, 30],
                "income": [60_000.0, 15_000.0, 0.0, 20_000.0],
                "relationship_to_head": [0, 1, 2, 3],
                "marital_status": [1, 1, 7, 7],
                "state_fips": [6, 6, 6, 6],
                "tenure": [1, 1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)

        assert len(tables.spm_units) == 1
        assert person_rows["spm_unit_id"].nunique() == 1
        assert len(tables.families) == 2
        assert person_rows["family_id"].nunique() == 2
        assert person_rows.loc[:2, "family_id"].nunique() == 1
        assert person_rows.loc[3, "family_id"] != person_rows.loc[0, "family_id"]

    def test_build_policyengine_entity_tables_uses_family_relationship_for_family_units(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5],
                "household_id": [10, 10, 10, 10, 10],
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
                "age": [45, 43, 12, 70, 30],
                "income": [60_000.0, 15_000.0, 0.0, 5_000.0, 20_000.0],
                "relationship_to_head": [0, 1, 2, 3, 3],
                "family_relationship": [1, 2, 3, 4, 0],
                "marital_status": [1, 1, 7, 4, 7],
                "state_fips": [6, 6, 6, 6, 6],
                "tenure": [1, 1, 1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)

        assert len(tables.spm_units) == 1
        assert len(tables.families) == 2
        assert person_rows.loc[:3, "family_id"].nunique() == 1
        assert person_rows.loc[4, "family_id"] != person_rows.loc[0, "family_id"]

    def test_build_policyengine_entity_tables_preserves_complete_existing_group_ids(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "household_id": [10, 10, 10, 20],
                "weight": [1.0, 1.0, 1.0, 2.0],
                "age": [45, 12, 30, 70],
                "income": [60_000.0, 0.0, 20_000.0, 25_000.0],
                "relationship_to_head": [0, 2, 3, 0],
                "family_id": [1, 1, 2, 1],
                "spm_unit_id": [1, 2, 2, 1],
                "marital_unit_id": [1, 2, 3, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)

        assert len(tables.families) == 3
        assert len(tables.spm_units) == 3
        assert len(tables.marital_units) == 4
        assert person_rows.loc[0, "family_id"] == person_rows.loc[1, "family_id"]
        assert person_rows.loc[0, "family_id"] != person_rows.loc[3, "family_id"]
        assert person_rows.loc[1, "spm_unit_id"] == person_rows.loc[2, "spm_unit_id"]
        assert person_rows.loc[0, "spm_unit_id"] != person_rows.loc[3, "spm_unit_id"]
        assert person_rows["marital_unit_id"].nunique() == 4

    def test_build_policyengine_entity_tables_derives_is_household_head(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 20],
                "weight": [1.0, 1.0, 2.0],
                "age": [45, 12, 70],
                "income": [60_000.0, 0.0, 25_000.0],
                "relationship_to_head": [0, 2, 0],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        persons = tables.persons.sort_values("person_id").reset_index(drop=True)

        assert persons["is_household_head"].tolist() == [True, False, True]

    def test_build_policyengine_entity_tables_derives_tax_input_columns(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "weight": [1.0, 1.0],
                "age": [45, 43],
                "sex": [1, 2],
                "race": [4, 2],
                "hispanic": [2, 1],
                "income": [60_000.0, 15_000.0],
                "wage_income": [50_000.0, 10_000.0],
                "self_employment_income": [5_000.0, 0.0],
                "taxable_interest_income": [100.0, 20.0],
                "ordinary_dividend_income": [80.0, 30.0],
                "qualified_dividend_income": [30.0, 5.0],
                "short_term_capital_gains": [10.0, 0.0],
                "long_term_capital_gains": [40.0, 5.0],
                "rental_income": [200.0, 0.0],
                "gross_social_security": [0.0, 800.0],
                "ssi": [0.0, 600.0],
                "taxable_pension_income": [0.0, 300.0],
                "unemployment_compensation": [0.0, 150.0],
                "medicaid": [0.0, 1_250.0],
                "medicaid_enrolled": [False, True],
                "health_insurance_premiums_without_medicare_part_b": [120.0, 80.0],
                "state_income_tax_paid": [400.0, 50.0],
                "filing_status": ["JOINT", "JOINT"],
                "relationship_to_head": [0, 1],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_unit_rows = tables.tax_units.sort_values("household_id").reset_index(
            drop=True
        )

        assert person_rows["employment_income_before_lsr"].tolist() == [
            50_000.0,
            10_000.0,
        ]
        assert person_rows["self_employment_income_before_lsr"].tolist() == [
            5_000.0,
            0.0,
        ]
        assert person_rows["taxable_interest_income"].tolist() == [100.0, 20.0]
        assert person_rows["dividend_income"].tolist() == [80.0, 30.0]
        assert person_rows["qualified_dividend_income"].tolist() == [30.0, 5.0]
        assert person_rows["non_qualified_dividend_income"].tolist() == [50.0, 25.0]
        assert person_rows["short_term_capital_gains"].tolist() == [10.0, 0.0]
        assert person_rows["long_term_capital_gains_before_response"].tolist() == [
            40.0,
            5.0,
        ]
        assert person_rows["social_security_retirement"].tolist() == [0.0, 800.0]
        assert person_rows["ssi"].tolist() == [0.0, 600.0]
        assert person_rows["takes_up_ssi_if_eligible"].tolist() == [False, True]
        assert person_rows["taxable_private_pension_income"].tolist() == [0.0, 300.0]
        assert person_rows["unemployment_compensation"].tolist() == [0.0, 150.0]
        assert person_rows["is_female"].tolist() == [False, True]
        assert person_rows["cps_race"].tolist() == [4, 2]
        assert person_rows["is_hispanic"].tolist() == [False, True]
        assert person_rows["medicaid"].tolist() == [0.0, 1_250.0]
        assert person_rows["medicaid_enrolled"].tolist() == [False, True]
        assert (
            tax_unit_rows["health_insurance_premiums_without_medicare_part_b"].sum()
            == 200.0
        )
        assert person_rows["state_income_tax_reported"].tolist() == [400.0, 50.0]

    def test_build_policyengine_entity_tables_adds_deterministic_aca_takeup(
        self,
        monkeypatch,
    ):
        calls: list[tuple[str, int]] = []

        def fake_load_takeup_rate(variable_name: str, year: int) -> float:
            calls.append((variable_name, year))
            return 0.0

        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_takeup_rate",
            fake_load_takeup_rate,
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 20, 30],
                "weight": [1.0, 1.0, 1.0],
                "age": [34, 42, 29],
                "sex": [2, 1, 2],
                "income": [40_000.0, 65_000.0, 32_000.0],
                "filing_status": ["SINGLE", "SINGLE", "SINGLE"],
                "relationship_to_head": [0, 0, 0],
                "state_fips": [6, 12, 48],
                "tenure": [1, 1, 1],
                "has_marketplace_health_coverage": [True, False, True],
                "takes_up_snap_if_eligible": [True, True, True],
                "would_file_taxes_voluntarily": [False, False, False],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)

        tax_units = tables.tax_units.sort_values("household_id").reset_index(drop=True)
        assert calls == [
            ("head_start", 2024),
            ("early_head_start", 2024),
            ("aca", 2024),
            ("dc_ptc", 2024),
            ("tanf", 2024),
        ]
        assert tax_units["takes_up_aca_if_eligible"].tolist() == [
            False,
            False,
            False,
        ]

    def test_build_policyengine_entity_tables_preserves_explicit_aca_takeup(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 20],
                "weight": [1.0, 1.0],
                "age": [34, 42],
                "sex": [2, 1],
                "income": [40_000.0, 65_000.0],
                "filing_status": ["SINGLE", "SINGLE"],
                "relationship_to_head": [0, 0],
                "state_fips": [6, 12],
                "tenure": [1, 1],
                "has_marketplace_health_coverage": [False, True],
                "takes_up_aca_if_eligible": [True, False],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)

        tax_units = tables.tax_units.sort_values("household_id").reset_index(drop=True)
        assert tax_units["takes_up_aca_if_eligible"].tolist() == [True, False]

    def test_attach_policyengine_marketplace_ratio_materializes_intermediates(
        self,
        monkeypatch,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
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
                    "tax_unit_id": [100, 200],
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [100, 200],
                    "household_id": [10, 20],
                    "health_insurance_premiums_without_medicare_part_b": [
                        300.0,
                        50.0,
                    ],
                    "takes_up_aca_if_eligible": [True, True],
                }
            ),
        )

        captured_variables: list[tuple[str, ...]] = []

        def fake_materialize(tables_arg, *, variables, **kwargs):
            captured_variables.append(tuple(variables))
            tax_units = tables_arg.tax_units.copy()
            tax_units["aca_ptc"] = [700.0, 0.0]
            tax_units["slcsp"] = [1_000.0, 1_000.0]
            return PolicyEngineUSVariableMaterializationResult(
                tables=PolicyEngineUSEntityTableBundle(
                    households=tables_arg.households,
                    persons=tables_arg.persons,
                    tax_units=tax_units,
                    spm_units=tables_arg.spm_units,
                    families=tables_arg.families,
                    marital_units=tables_arg.marital_units,
                ),
                bindings={
                    "aca_ptc": PolicyEngineUSVariableBinding(
                        entity=EntityType.TAX_UNIT,
                        column="aca_ptc",
                    ),
                    "slcsp": PolicyEngineUSVariableBinding(
                        entity=EntityType.TAX_UNIT,
                        column="slcsp",
                    ),
                },
                materialized_variables=tuple(variables),
            )

        monkeypatch.setattr(
            us_pipeline_module,
            "materialize_policyengine_us_variables_safely",
            fake_materialize,
        )

        updated = pipeline._attach_policyengine_marketplace_plan_benchmark_ratio(
            tables,
            target_period=2024,
        )

        assert captured_variables == [("aca_ptc", "slcsp")]
        np.testing.assert_allclose(
            updated.tax_units["selected_marketplace_plan_benchmark_ratio"],
            np.array([1.0, 0.5]),
        )

    def test_build_policyengine_entity_tables_adds_ecps_stochastic_takeup_inputs(
        self,
        monkeypatch,
    ):
        scalar_calls: list[tuple[str, int]] = []
        medicaid_calls: list[int] = []
        pregnancy_calls: list[int] = []
        eitc_calls: list[int] = []
        voluntary_calls: list[int] = []

        def fake_load_takeup_rate(variable_name: str, year: int) -> float:
            scalar_calls.append((variable_name, year))
            return {
                "head_start": 0.0,
                "early_head_start": 1.0,
                "dc_ptc": 1.0,
                "snap": 1.0,
                "tanf": 0.0,
                "aca": 1.0,
            }[variable_name]

        def fake_load_medicaid_rates(year: int) -> dict[str, float]:
            medicaid_calls.append(year)
            return {"CA": 0.0, "TX": 1.0}

        def fake_load_pregnancy_rates(year: int) -> dict[str, float]:
            pregnancy_calls.append(year)
            return {"CA": 1.0, "TX": 0.0}

        def fake_load_eitc_rates(year: int) -> dict[int, float]:
            eitc_calls.append(year)
            return {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}

        def fake_load_voluntary_rates(
            year: int,
        ) -> dict[str, dict[str, dict[str, float]]]:
            voluntary_calls.append(year)
            return {
                children: {
                    wage: {age: 1.0 for age in ("under_65", "age_65_plus")}
                    for wage in ("zero", "low", "medium", "high")
                }
                for children in ("no_children", "with_children")
            }

        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_takeup_rate",
            fake_load_takeup_rate,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_medicaid_takeup_rates",
            fake_load_medicaid_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_pregnancy_rates",
            fake_load_pregnancy_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_eitc_takeup_rates",
            fake_load_eitc_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_voluntary_filing_rates",
            fake_load_voluntary_rates,
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 20, 20],
                "spm_unit_id": [100, 200, 200],
                "weight": [1.0, 1.0, 1.0],
                "age": [34, 42, 8],
                "sex": [2, 1, 2],
                "income": [40_000.0, 35_000.0, 0.0],
                "relationship_to_head": [0, 0, 2],
                "state_fips": [6, 48, 48],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)

        persons = tables.persons.sort_values("person_id").reset_index(drop=True)
        assert persons["takes_up_medicaid_if_eligible"].tolist() == [
            False,
            True,
            True,
        ]
        assert persons["takes_up_head_start_if_eligible"].tolist() == [
            False,
            False,
            False,
        ]
        assert persons["takes_up_early_head_start_if_eligible"].tolist() == [
            True,
            True,
            True,
        ]
        assert persons["is_pregnant"].tolist() == [True, False, False]

        tax_units = tables.tax_units.sort_values("household_id").reset_index(drop=True)
        assert tax_units["takes_up_aca_if_eligible"].tolist() == [True, True]
        assert tax_units["takes_up_dc_ptc"].tolist() == [True, True]
        assert tax_units["takes_up_eitc"].tolist() == [False, True]
        assert tax_units["would_file_taxes_voluntarily"].tolist() == [True, False]

        spm_units = tables.spm_units.sort_values("household_id").reset_index(drop=True)
        assert spm_units["takes_up_snap_if_eligible"].tolist() == [True, True]
        assert spm_units["takes_up_tanf_if_eligible"].tolist() == [False, False]
        assert scalar_calls == [
            ("head_start", 2024),
            ("early_head_start", 2024),
            ("aca", 2024),
            ("dc_ptc", 2024),
            ("snap", 2024),
            ("tanf", 2024),
        ]
        assert medicaid_calls == [2024]
        assert pregnancy_calls == [2024]
        assert eitc_calls == [2024]
        assert voluntary_calls == [2024]

    def test_attach_policyengine_pregnancy_inputs_assigns_eligible_females(
        self,
        monkeypatch,
    ):
        class FakeRng:
            def random(self, size: int) -> np.ndarray:
                return np.zeros(size)

        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_pregnancy_rates",
            lambda year: {"CA": 0.10, "NY": 0.0},
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_microplex_seeded_rng",
            lambda variable_name, *, salt=None: FakeRng(),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
        persons = pd.DataFrame(
            {
                "age": [20, 44, 45, 30, 20],
                "sex": [2, 2, 2, 1, 2],
                "state_fips": [6, 36, 6, 6, 99],
            }
        )

        result = pipeline._attach_policyengine_pregnancy_inputs(persons)

        assert result["is_pregnant"].tolist() == [
            True,
            False,
            False,
            False,
            True,
        ]

    def test_attach_policyengine_pregnancy_inputs_preserves_explicit_column(
        self,
        monkeypatch,
    ):
        def fail_rates(year: int) -> dict[str, float]:
            raise AssertionError(f"unexpected pregnancy rate load: {year}")

        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_pregnancy_rates",
            fail_rates,
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
        persons = pd.DataFrame({"is_pregnant": [1, 0, True, False]})

        result = pipeline._attach_policyengine_pregnancy_inputs(persons)

        assert result["is_pregnant"].tolist() == [True, False, True, False]

    def test_build_policyengine_entity_tables_adds_wic_takeup_inputs(
        self,
        monkeypatch,
    ):
        wic_takeup_calls: list[int] = []
        wic_risk_calls: list[int] = []

        def fake_wic_takeup_rates(year: int) -> dict[str, float]:
            wic_takeup_calls.append(year)
            return {
                "PREGNANT": 0.0,
                "POSTPARTUM": 1.0,
                "BREASTFEEDING": 0.0,
                "INFANT": 1.0,
                "CHILD": 0.0,
                "NONE": 0.0,
            }

        def fake_wic_risk_rates(year: int) -> dict[str, float]:
            wic_risk_calls.append(year)
            return {
                "PREGNANT": 0.0,
                "POSTPARTUM": 0.0,
                "BREASTFEEDING": 0.0,
                "INFANT": 0.0,
                "CHILD": 1.0,
                "NONE": 0.0,
            }

        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_wic_takeup_rates",
            fake_wic_takeup_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_wic_nutritional_risk_rates",
            fake_wic_risk_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_pregnancy_rates",
            lambda year: {},
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_microplex_seeded_rng",
            lambda variable_name, *, salt=None: np.random.default_rng(0),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "household_id": [10, 10, 30, 40],
                "family_id": [10, 10, 30, 40],
                "spm_unit_id": [10, 10, 30, 40],
                "weight": [1.0, 1.0, 1.0, 1.0],
                "age": [30, 0, 4, 40],
                "sex": [2, 1, 2, 1],
                "income": [40_000.0, 0.0, 0.0, 35_000.0],
                "relationship_to_head": [0, 2, 0, 0],
                "state_fips": [6, 6, 6, 6],
                "own_children_in_household": [1, 0, 0, 0],
                "receives_wic": [False, True, False, False],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)

        persons = tables.persons.sort_values("person_id").reset_index(drop=True)
        assert persons["would_claim_wic"].tolist() == [True, True, False, False]
        assert persons["is_wic_at_nutritional_risk"].tolist() == [
            False,
            True,
            True,
            False,
        ]
        assert wic_takeup_calls == [2024]
        assert wic_risk_calls == [2024]

    def test_build_policyengine_entity_tables_preserves_explicit_stochastic_takeup_inputs(
        self,
        monkeypatch,
    ):
        def fail_scalar_rate(variable_name: str, year: int) -> float:
            raise AssertionError(f"unexpected scalar rate load: {variable_name} {year}")

        def fail_medicaid_rates(year: int) -> dict[str, float]:
            raise AssertionError(f"unexpected Medicaid rate load: {year}")

        def fail_pregnancy_rates(year: int) -> dict[str, float]:
            raise AssertionError(f"unexpected pregnancy rate load: {year}")

        def fail_eitc_rates(year: int) -> dict[int, float]:
            raise AssertionError(f"unexpected EITC rate load: {year}")

        def fail_voluntary_rates(year: int) -> dict:
            raise AssertionError(f"unexpected voluntary filing rate load: {year}")

        def fail_wic_takeup_rates(year: int) -> dict[str, float]:
            raise AssertionError(f"unexpected WIC take-up rate load: {year}")

        def fail_wic_risk_rates(year: int) -> dict[str, float]:
            raise AssertionError(f"unexpected WIC nutritional-risk rate load: {year}")

        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_takeup_rate",
            fail_scalar_rate,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_medicaid_takeup_rates",
            fail_medicaid_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_pregnancy_rates",
            fail_pregnancy_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_eitc_takeup_rates",
            fail_eitc_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_voluntary_filing_rates",
            fail_voluntary_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_wic_takeup_rates",
            fail_wic_takeup_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_wic_nutritional_risk_rates",
            fail_wic_risk_rates,
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "spm_unit_id": [100, 100],
                "weight": [1.0, 1.0],
                "age": [34, 8],
                "sex": [2, 2],
                "income": [40_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "state_fips": [6, 6],
                "takes_up_medicaid_if_eligible": [False, True],
                "is_pregnant": [False, True],
                "takes_up_head_start_if_eligible": [False, True],
                "takes_up_early_head_start_if_eligible": [True, False],
                "takes_up_aca_if_eligible": [False, True],
                "takes_up_dc_ptc": [False, True],
                "takes_up_eitc": [False, True],
                "would_file_taxes_voluntarily": [True, False],
                "takes_up_snap_if_eligible": [False, True],
                "takes_up_tanf_if_eligible": [True, False],
                "would_claim_wic": [False, True],
                "is_wic_at_nutritional_risk": [True, False],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)

        persons = tables.persons.sort_values("person_id").reset_index(drop=True)
        assert persons["takes_up_medicaid_if_eligible"].tolist() == [False, True]
        assert persons["is_pregnant"].tolist() == [False, True]
        assert persons["takes_up_head_start_if_eligible"].tolist() == [False, True]
        assert persons["takes_up_early_head_start_if_eligible"].tolist() == [
            True,
            False,
        ]
        assert persons["would_claim_wic"].tolist() == [False, True]
        assert persons["is_wic_at_nutritional_risk"].tolist() == [True, False]

        tax_units = tables.tax_units.sort_values("household_id").reset_index(drop=True)
        assert tax_units["takes_up_aca_if_eligible"].tolist() == [True]
        assert tax_units["takes_up_dc_ptc"].tolist() == [True]
        assert tax_units["takes_up_eitc"].tolist() == [True]
        assert tax_units["would_file_taxes_voluntarily"].tolist() == [True]

        spm_units = tables.spm_units.sort_values("household_id").reset_index(drop=True)
        assert spm_units["takes_up_snap_if_eligible"].tolist() == [True]
        assert spm_units["takes_up_tanf_if_eligible"].tolist() == [True]

    def test_build_policyengine_entity_tables_uses_eitc_children_for_eitc_takeup(
        self,
        monkeypatch,
    ):
        eitc_calls: list[int] = []

        def fail_scalar_rate(variable_name: str, year: int) -> float:
            raise AssertionError(f"unexpected scalar rate load: {variable_name} {year}")

        def fail_medicaid_rates(year: int) -> dict[str, float]:
            raise AssertionError(f"unexpected Medicaid rate load: {year}")

        def fake_eitc_rates(year: int) -> dict[int, float]:
            eitc_calls.append(year)
            return {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}

        def fail_voluntary_rates(year: int) -> dict:
            raise AssertionError(f"unexpected voluntary filing rate load: {year}")

        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_takeup_rate",
            fail_scalar_rate,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_medicaid_takeup_rates",
            fail_medicaid_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_eitc_takeup_rates",
            fake_eitc_rates,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "_load_microplex_voluntary_filing_rates",
            fail_voluntary_rates,
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
        population = pd.DataFrame(
            {
                "person_id": [1],
                "household_id": [10],
                "spm_unit_id": [100],
                "weight": [1.0],
                "age": [34],
                "sex": [2],
                "income": [40_000.0],
                "relationship_to_head": [0],
                "state_fips": [6],
                "eitc_children": [1],
                "eitc_child_count": [0],
                "takes_up_medicaid_if_eligible": [True],
                "takes_up_head_start_if_eligible": [False],
                "takes_up_early_head_start_if_eligible": [False],
                "takes_up_aca_if_eligible": [True],
                "takes_up_dc_ptc": [False],
                "would_file_taxes_voluntarily": [False],
                "takes_up_snap_if_eligible": [True],
                "takes_up_tanf_if_eligible": [False],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)

        tax_units = tables.tax_units.sort_values("household_id").reset_index(drop=True)
        assert tax_units["takes_up_eitc"].tolist() == [True]
        assert "_mp_eitc_child_count_for_takeup" not in tax_units.columns
        assert eitc_calls == [2024]

    def test_build_policyengine_entity_tables_fallback_employment_excludes_transfer_income(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1],
                "household_id": [10],
                "weight": [1.0],
                "age": [62],
                "sex": [2],
                "income": [18_000.0],
                "ssi": [9_000.0],
                "public_assistance": [3_000.0],
                "gross_social_security": [2_000.0],
                "filing_status": ["SINGLE"],
                "relationship_to_head": [0],
                "state_fips": [6],
                "tenure": [1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_row = tables.persons.iloc[0]

        assert person_row["employment_income_before_lsr"] == 4_000.0
        assert person_row["ssi"] == 9_000.0
        assert person_row["social_security_retirement"] == 2_000.0

    def test_build_policyengine_entity_tables_allocates_social_security_residual_to_retirement(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1],
                "household_id": [10],
                "weight": [1.0],
                "age": [62],
                "sex": [2],
                "income": [2_000.0],
                "gross_social_security": [2_000.0],
                "social_security_disability": [500.0],
                "filing_status": ["SINGLE"],
                "relationship_to_head": [0],
                "state_fips": [6],
                "tenure": [1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_row = tables.persons.iloc[0]

        assert person_row["social_security_retirement"] == 1_500.0
        assert person_row["social_security_disability"] == 500.0

    def test_build_policyengine_entity_tables_derives_dividend_totals_from_atomic_components(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1],
                "household_id": [10],
                "weight": [1.0],
                "age": [45],
                "income": [60_000.0],
                "wage_income": [50_000.0],
                "ordinary_dividend_income": [50.0],
                "dividend_income": [0.0],
                "qualified_dividend_income": [30.0],
                "non_qualified_dividend_income": [12.0],
                "filing_status": ["SINGLE"],
                "relationship_to_head": [0],
                "state_fips": [6],
                "tenure": [1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_row = tables.persons.iloc[0]

        assert person_row["qualified_dividend_income"] == 30.0
        assert person_row["non_qualified_dividend_income"] == 12.0
        assert person_row["ordinary_dividend_income"] == 42.0
        assert person_row["dividend_income"] == 42.0

    def test_build_policyengine_entity_tables_derives_relationships_from_family_relationship(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "family_relationship": [0, 1, 2],
                "marital_status": [1, 1, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["relationship_to_head"].tolist() == [0, 1, 2]
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_derives_relationships_from_one_based_family_relationship(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "family_relationship": [1, 2, 3],
                "marital_status": [1, 1, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["relationship_to_head"].tolist() == [0, 1, 2]
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_uses_spouse_and_dependent_flags_when_relationship_missing(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "is_spouse": [0, 1, 0],
                "is_dependent": [0, 0, 1],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["relationship_to_head"].tolist() == [0, 1, 2]
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_prefers_richer_family_relationship_over_collapsed_relationship_to_head(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "family_relationship": [0, 1, 2],
                "relationship_to_head": [0, 3, 3],
                "marital_status": [1, 1, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["relationship_to_head"].tolist() == [0, 1, 2]
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_repairs_households_without_a_head(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "relationship_to_head": [1, 1, 2],
                "marital_status": [1, 1, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["relationship_to_head"].tolist() == [0, 1, 2]
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_marks_separated_head_as_separate(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "weight": [1.0, 1.0],
                "age": [45, 12],
                "income": [60_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "marital_status": [6, 7],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "SEPARATE"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_splits_separated_spouses_into_two_units(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "weight": [1.0, 1.0],
                "age": [45, 43],
                "income": [60_000.0, 15_000.0],
                "relationship_to_head": [0, 1],
                "marital_status": [6, 6],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)

        assert len(tax_units) == 2
        assert tax_units["filing_status"].tolist() == ["SEPARATE", "SEPARATE"]
        assert person_rows["tax_unit_id"].nunique() == 2

    def test_build_policyengine_entity_tables_splits_separated_spouses_and_keeps_dependents_with_head(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "relationship_to_head": [0, 1, 2],
                "marital_status": [6, 6, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)

        assert len(tax_units) == 2
        assert tax_units.iloc[0]["filing_status"] == "SEPARATE"
        assert tax_units.iloc[0]["n_dependents"] == 1
        assert tax_units.iloc[1]["filing_status"] == "SEPARATE"
        assert tax_units.iloc[1]["n_dependents"] == 0
        dependent_tax_unit_id = int(
            person_rows.loc[person_rows["person_id"] == 3, "tax_unit_id"].iloc[0]
        )
        assert dependent_tax_unit_id == int(tax_units.iloc[0]["tax_unit_id"])

    def test_build_policyengine_entity_tables_splits_spouse_coded_pair_without_marriage_evidence(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "weight": [1.0, 1.0],
                "age": [45, 43],
                "income": [60_000.0, 15_000.0],
                "relationship_to_head": [0, 1],
                "marital_status": [7, 7],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)

        assert len(tax_units) == 2
        assert tax_units["filing_status"].tolist() == ["SINGLE", "SINGLE"]
        assert person_rows["tax_unit_id"].nunique() == 2

    def test_build_policyengine_entity_tables_marks_widowed_head_with_child_as_surviving_spouse(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "weight": [1.0, 1.0],
                "age": [45, 12],
                "income": [60_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "marital_status": [4, 7],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "SURVIVING_SPOUSE"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_prefers_explicit_head_of_household_code(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "weight": [1.0, 1.0],
                "age": [45, 12],
                "income": [60_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "marital_status": [5, 7],
                "filing_status_code": [4, np.nan],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "HEAD_OF_HOUSEHOLD"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_preserves_explicit_role_flag_head_of_household_code(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 100],
                "weight": [1.0, 1.0],
                "age": [45, 12],
                "income": [60_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "person_number": [1, 2],
                "spouse_person_number": [0, 0],
                "tax_unit_is_joint": [0.0, 0.0],
                "tax_unit_count_dependents": [1.0, 1.0],
                "is_tax_unit_head": [1.0, 0.0],
                "is_tax_unit_spouse": [0.0, 0.0],
                "is_tax_unit_dependent": [0.0, 1.0],
                "filing_status_code": [4, np.nan],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "HEAD_OF_HOUSEHOLD"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_does_not_promote_non_hoh_role_flag_codes(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "household_id": [10, 10, 20, 20],
                "tax_unit_id": [100, 100, 200, 200],
                "weight": [1.0, 1.0, 1.0, 1.0],
                "age": [45, 12, 44, 10],
                "income": [60_000.0, 0.0, 55_000.0, 0.0],
                "relationship_to_head": [0, 2, 0, 2],
                "person_number": [1, 2, 1, 2],
                "spouse_person_number": [0, 0, 0, 0],
                "tax_unit_is_joint": [0.0, 0.0, 0.0, 0.0],
                "tax_unit_count_dependents": [1.0, 1.0, 1.0, 1.0],
                "is_tax_unit_head": [1.0, 0.0, 1.0, 0.0],
                "is_tax_unit_spouse": [0.0, 0.0, 0.0, 0.0],
                "is_tax_unit_dependent": [0.0, 1.0, 0.0, 1.0],
                "filing_status_code": [1, np.nan, 2, np.nan],
                "state_fips": [6, 6, 6, 6],
                "tenure": [1, 1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert tax_units["filing_status"].tolist() == ["SINGLE", "SINGLE"]
        assert tax_units["n_dependents"].tolist() == [1, 1]

    def test_build_policyengine_entity_tables_does_not_infer_head_of_household_from_marital_status_alone(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "weight": [1.0, 1.0],
                "age": [45, 12],
                "income": [60_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "marital_status": [5, 7],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "SINGLE"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_can_preserve_existing_tax_unit_ids(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "tax_unit_id": [100, 100, 200],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "relationship_to_head": [0, 1, 2],
                "marital_status": [1, 1, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["tax_unit_id"].tolist() == [100, 100, 200]
        assert tax_units["tax_unit_id"].tolist() == [100, 200]
        assert tax_units["filing_status"].tolist() == ["JOINT", "SINGLE"]
        assert tax_units["n_dependents"].tolist() == [0, 0]

    def test_build_policyengine_entity_tables_prefers_tax_unit_role_flags_over_bad_ids(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "tax_unit_id": [100, 101, 102],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "relationship_to_head": [0, 1, 2],
                "marital_status": [1, 1, 7],
                "person_number": [1, 2, 3],
                "spouse_person_number": [2, 1, 0],
                "tax_unit_is_joint": [1.0, 1.0, 1.0],
                "tax_unit_count_dependents": [1.0, 1.0, 1.0],
                "is_tax_unit_head": [1.0, 0.0, 0.0],
                "is_tax_unit_spouse": [0.0, 1.0, 0.0],
                "is_tax_unit_dependent": [0.0, 0.0, 1.0],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert len(tax_units) == 1
        assert person_rows["tax_unit_id"].nunique() == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_microunit_overrides_bad_cps_tax_unit_ids(
        self,
    ):
        # microunit is the DEFAULT tax-unit constructor: when the high-fidelity CPS
        # fields (person_number + family_relationship) are present it re-partitions
        # the household and intentionally REPLACES the unreliable CPS-provided
        # tax_unit_id (Census TAX_ID) -- even though
        # policyengine_prefer_existing_tax_unit_ids defaults to True (that path is a
        # fallback for households microunit does not construct, not a competing
        # authority). This locks in "replace the CPS tax units, keep the SPM units".
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        assert pipeline.config.policyengine_prefer_existing_tax_unit_ids is True
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                # CPS TAX_ID nonsensically splits the dependent child into its own unit.
                "tax_unit_id": [100, 100, 200],
                # SPM units, by contrast, must be preserved.
                "spm_unit_id": [500, 500, 500],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "person_number": [1, 2, 3],
                "spouse_person_number": [2, 1, 0],
                "family_relationship": [1, 2, 3],  # CPS A_FAMREL: ref, spouse, child
                "marital_status": [1, 1, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units

        # microunit folds couple + child into ONE unit, discarding the [100,100,200]
        # split (which preservation would have kept as two units).
        assert len(tax_units) == 1
        assert person_rows["tax_unit_id"].nunique() == 1
        assert tax_units.iloc[0]["n_dependents"] == 1
        # The SPM unit is untouched (replace tax, keep SPM).
        assert person_rows["spm_unit_id"].nunique() == 1

    def test_build_policyengine_entity_tables_resolves_spouse_head_role_conflicts(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 101],
                "weight": [1.0, 1.0],
                "age": [45, 43],
                "income": [60_000.0, 15_000.0],
                "relationship_to_head": [0, 1],
                "family_relationship": [1, 2],
                "person_number": [1, 2],
                "spouse_person_number": [2, 1],
                "tax_unit_is_joint": [1.0, 1.0],
                "tax_unit_count_dependents": [0.0, 0.0],
                "is_tax_unit_head": [1.0, 1.0],
                "is_tax_unit_spouse": [0.0, 1.0],
                "is_tax_unit_dependent": [0.0, 0.0],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert len(tax_units) == 1
        assert person_rows["tax_unit_id"].nunique() == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"

    def test_build_policyengine_entity_tables_resolves_dependent_head_role_conflicts(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 101],
                "weight": [1.0, 1.0],
                "age": [45, 12],
                "income": [60_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "family_relationship": [1, 3],
                "person_number": [1, 2],
                "spouse_person_number": [0, 0],
                "tax_unit_is_joint": [0.0, 0.0],
                "tax_unit_count_dependents": [1.0, 1.0],
                "is_tax_unit_head": [1.0, 1.0],
                "is_tax_unit_spouse": [0.0, 0.0],
                "is_tax_unit_dependent": [0.0, 1.0],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        # filing_status is PE-computed (delegated; microplex does not export it),
        # so only microunit's partition is asserted here.
        assert len(tax_units) == 1
        assert person_rows["tax_unit_id"].nunique() == 1
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_resolves_spouse_dependent_role_conflicts(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 100],
                "weight": [1.0, 1.0],
                "age": [45, 12],
                "income": [60_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "family_relationship": [1, 3],
                "person_number": [1, 2],
                "spouse_person_number": [0, 0],
                "tax_unit_is_joint": [0.0, 1.0],
                "tax_unit_count_dependents": [1.0, 1.0],
                "is_tax_unit_head": [1.0, 0.0],
                "is_tax_unit_spouse": [0.0, 1.0],
                "is_tax_unit_dependent": [0.0, 1.0],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        # filing_status delegated to PE; assert only microunit's partition.
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_repairs_missing_role_flag_heads(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 101],
                "weight": [1.0, 1.0],
                "age": [45, 12],
                "income": [60_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "family_relationship": [1, 3],
                "person_number": [1, 2],
                "spouse_person_number": [0, 0],
                "tax_unit_is_joint": [0.0, 0.0],
                "tax_unit_count_dependents": [1.0, 1.0],
                "is_tax_unit_head": [0.0, 0.0],
                "is_tax_unit_spouse": [0.0, 0.0],
                "is_tax_unit_dependent": [0.0, 1.0],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        # filing_status delegated to PE; assert only microunit's partition.
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_folds_young_head_hint_dependents(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 101],
                "weight": [1.0, 1.0],
                "age": [45, 22],
                "income": [60_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "family_relationship": [1, 3],
                "person_number": [1, 2],
                "spouse_person_number": [0, 0],
                "tax_unit_is_joint": [0.0, 0.0],
                "tax_unit_count_dependents": [1.0, 1.0],
                "is_tax_unit_head": [1.0, 1.0],
                "is_tax_unit_spouse": [0.0, 0.0],
                "is_tax_unit_dependent": [0.0, 0.0],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        # microunit applies the real qualifying-child age rule: a 19+ non-student
        # own-child is NOT folded as a dependent (it gets its own tax unit), unlike
        # the legacy role-flag heuristic. Threading student enrollment (A_HSCOL) so
        # the qualifying-child-to-24 student extension fires is a tracked follow-up.
        assert len(tax_units) == 2
        assert int(tax_units["n_dependents"].sum()) == 0

    def test_build_policyengine_entity_tables_uses_legacy_path_without_cps_fields(
        self,
    ):
        # Without the high-fidelity CPS fields (person_number/family_relationship),
        # microunit cannot construct, so the legacy role-flag reconstruction (the
        # fallback) handles the conflict. Preserves coverage of that path now that
        # the real-data path defaults to microunit.
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 101],
                "weight": [1.0, 1.0],
                "age": [45, 12],
                "income": [60_000.0, 0.0],
                "relationship_to_head": [0, 2],
                "is_tax_unit_head": [1.0, 1.0],
                "is_tax_unit_spouse": [0.0, 0.0],
                "is_tax_unit_dependent": [0.0, 1.0],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )
        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_keeps_positive_income_adult_heads(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 101],
                "weight": [1.0, 1.0],
                "age": [45, 25],
                "income": [60_000.0, 20_000.0],
                "relationship_to_head": [0, 2],
                "family_relationship": [1, 3],
                "person_number": [1, 2],
                "spouse_person_number": [0, 0],
                "tax_unit_is_joint": [0.0, 0.0],
                "tax_unit_count_dependents": [0.0, 0.0],
                "is_tax_unit_head": [1.0, 1.0],
                "is_tax_unit_spouse": [0.0, 0.0],
                "is_tax_unit_dependent": [0.0, 0.0],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert len(tax_units) == 2
        assert tax_units["filing_status"].tolist() == ["SINGLE", "SINGLE"]
        assert tax_units["n_dependents"].tolist() == [0, 0]

    def test_build_policyengine_entity_tables_preserves_tax_unit_agi_inputs(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_prefer_existing_tax_unit_ids=True)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 100],
                "weight": [1.0, 1.0],
                "age": [45, 43],
                "income": [60_000.0, 15_000.0],
                "relationship_to_head": [0, 1],
                "filing_status": ["JOINT", "JOINT"],
                "domestic_production_ald": [7.0, 2.0],
                "health_savings_account_ald": [60.0, 15.0],
                "recapture_of_investment_credit": [3.0, 4.0],
                "self_employed_health_insurance_ald": [20.0, 5.0],
                "self_employed_pension_contribution_ald": [30.0, 10.0],
                "unrecaptured_section_1250_gain": [11.0, 13.0],
                "unreported_payroll_tax": [17.0, 19.0],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert tax_units["domestic_production_ald"].tolist() == [9.0]
        assert tax_units["health_savings_account_ald"].tolist() == [75.0]
        assert tax_units["recapture_of_investment_credit"].tolist() == [7.0]
        assert tax_units["self_employed_health_insurance_ald"].tolist() == [25.0]
        assert tax_units["self_employed_pension_contribution_ald"].tolist() == [40.0]
        assert tax_units["unrecaptured_section_1250_gain"].tolist() == [24.0]
        assert tax_units["unreported_payroll_tax"].tolist() == [36.0]

    def test_build_policyengine_entity_tables_deduplicates_repeated_tax_unit_ald_values(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_prefer_existing_tax_unit_ids=True)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 100],
                "weight": [1.0, 1.0],
                "age": [45, 43],
                "income": [60_000.0, 15_000.0],
                "relationship_to_head": [0, 1],
                "filing_status": ["JOINT", "JOINT"],
                "self_employed_pension_contribution_ald": [30.0, 30.0],
                "unrecaptured_section_1250_gain": [50.0, 50.0],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert tax_units["self_employed_pension_contribution_ald"].tolist() == [30.0]
        assert tax_units["unrecaptured_section_1250_gain"].tolist() == [50.0]

    def test_build_policyengine_entity_tables_preserved_tax_units_require_reciprocal_spouse_pointer_for_joint(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_prefer_existing_tax_unit_ids=True)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 100],
                "weight": [1.0, 1.0],
                "age": [45, 43],
                "income": [60_000.0, 15_000.0],
                "relationship_to_head": [0, 1],
                "person_number": [1, 2],
                "spouse_person_number": [0, 0],
                "marital_status": [5, 7],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert tax_units["tax_unit_id"].tolist() == [100]
        assert tax_units["filing_status"].tolist() == ["SINGLE"]
        assert tax_units["n_dependents"].tolist() == [1]

    def test_build_policyengine_entity_tables_preserved_tax_units_keep_joint_for_reciprocal_spouse_pointer(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_prefer_existing_tax_unit_ids=True)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 100],
                "weight": [1.0, 1.0],
                "age": [45, 43],
                "income": [60_000.0, 15_000.0],
                "relationship_to_head": [0, 1],
                "person_number": [1, 2],
                "spouse_person_number": [2, 1],
                "marital_status": [1, 1],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert tax_units["tax_unit_id"].tolist() == [100]
        assert tax_units["filing_status"].tolist() == ["JOINT"]
        assert tax_units["n_dependents"].tolist() == [0]

    def test_build_policyengine_entity_tables_falls_back_when_existing_tax_unit_ids_cross_households(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_prefer_existing_tax_unit_ids=True)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 20],
                "tax_unit_id": [100, 100],
                "weight": [1.0, 1.0],
                "age": [45, 39],
                "income": [60_000.0, 40_000.0],
                "relationship_to_head": [0, 0],
                "marital_status": [7, 7],
                "state_fips": [6, 36],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["tax_unit_id"].nunique() == 2
        assert tax_units["household_id"].tolist() == [10, 20]

    def test_build_policyengine_entity_tables_partially_preserves_existing_tax_unit_ids(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_prefer_existing_tax_unit_ids=True)
        )
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5],
                "household_id": [10, 10, 10, 20, 20],
                "tax_unit_id": [100, 100, 200, np.nan, np.nan],
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
                "age": [45, 43, 12, 38, 8],
                "income": [60_000.0, 15_000.0, 0.0, 42_000.0, 0.0],
                "relationship_to_head": [0, 1, 2, 0, 2],
                "marital_status": [1, 1, 7, 7, 7],
                "state_fips": [6, 6, 6, 36, 36],
                "tenure": [1, 1, 1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values(
            ["household_id", "tax_unit_id"]
        ).reset_index(drop=True)

        assert person_rows.loc[:2, "tax_unit_id"].tolist() == [100, 100, 200]
        hh20_person_tax_units = person_rows.loc[
            person_rows["household_id"] == 20, "tax_unit_id"
        ]
        assert hh20_person_tax_units.notna().all()
        assert hh20_person_tax_units.nunique() == 1
        assert int(hh20_person_tax_units.iloc[0]) > 200
        assert tax_units.loc[
            tax_units["household_id"] == 10, "tax_unit_id"
        ].tolist() == [100, 200]
        assert tax_units.loc[
            tax_units["household_id"] == 20, "tax_unit_id"
        ].tolist() == [201]

    def test_build_from_source_providers_accepts_year_specific_query_keys(self):
        households = pd.DataFrame(
            {
                "household_id": ["1"],
                "state_fips": [6],
                "household_weight": [1.0],
                "year": [2024],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": ["1:1"],
                "household_id": ["1"],
                "age": [40],
                "sex": [1],
                "education": [3],
                "employment_status": [1],
                "income": [50_000.0],
                "weight": [1.0],
                "year": [2024],
            }
        )

        descriptor = SourceDescriptor(
            name="toy_source",
            shareability=Shareability.PUBLIC,
            time_structure=TimeStructure.REPEATED_CROSS_SECTION,
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column="household_id",
                    variable_names=("state_fips",),
                    weight_column="household_weight",
                    period_column="year",
                ),
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column="person_id",
                    variable_names=(
                        "age",
                        "sex",
                        "education",
                        "employment_status",
                        "income",
                    ),
                    weight_column="weight",
                    period_column="year",
                ),
            ),
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="toy_source_2024",
                shareability=descriptor.shareability,
                time_structure=descriptor.time_structure,
                observations=descriptor.observations,
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        class YearNamedProvider:
            year = 2024
            _descriptor_cache = None

            @property
            def descriptor(self):
                return self._descriptor_cache or descriptor

            def load_frame(self, query=None):
                self.last_query = query
                self._descriptor_cache = frame.source
                return frame

        provider = YearNamedProvider()
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=1,
                synthesis_backend="bootstrap",
            )
        )

        result = pipeline.build_from_source_providers(
            [provider],
            queries={
                "toy_source_2024": SourceQuery(
                    provider_filters={"sample_n": 1, "random_seed": 7}
                )
            },
        )

        assert provider.last_query is not None
        assert provider.last_query.provider_filters["sample_n"] == 1
        assert result.source_frame is not None
        assert result.source_frame.source.name == "toy_source_2024"

    def test_integrate_donor_sources_models_dividends_compositionally(
        self,
        monkeypatch,
    ):
        captured: dict[str, object] = {}

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [44, 21],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [58_000.0, 13_000.0],
                "qualified_dividend_income": [20.0, 7.0],
                "non_qualified_dividend_income": [8.0, 3.0],
                "ordinary_dividend_income": [28.0, 10.0],
                "dividend_income": [500.0, 200.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "qualified_dividend_income",
                            "non_qualified_dividend_income",
                            "ordinary_dividend_income",
                            "dividend_income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        class FakeSynthesizer:
            def __init__(self, *args, **kwargs):
                _ = args
                captured["init_kwargs"] = dict(kwargs)
                self.target_vars = kwargs.get("target_vars", [])

            def fit(self, *args, **kwargs):
                _ = args
                captured["fit_kwargs"] = dict(kwargs)

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                if "dividend_income" in self.target_vars:
                    result["dividend_income"] = [28.0, 10.0]
                if "qualified_dividend_share" in self.target_vars:
                    result["qualified_dividend_share"] = [20.0 / 28.0, 0.7]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_epochs=7,
                donor_imputer_batch_size=33,
                donor_imputer_learning_rate=5e-4,
                donor_imputer_n_layers=3,
                donor_imputer_hidden_dim=48,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert integration["integrated_variables"] == [
            "non_qualified_dividend_income",
            "qualified_dividend_income",
        ]
        assert integration["seed_data"]["qualified_dividend_income"].round(
            6
        ).tolist() == [
            20.0,
            7.0,
        ]
        assert integration["seed_data"]["non_qualified_dividend_income"].round(
            6
        ).tolist() == [
            8.0,
            3.0,
        ]
        assert integration["seed_data"]["ordinary_dividend_income"].round(
            6
        ).tolist() == [
            28.0,
            10.0,
        ]
        assert integration["seed_data"]["dividend_income"].round(6).tolist() == [
            28.0,
            10.0,
        ]
        assert "qualified_dividend_share" not in integration["seed_data"].columns
        assert captured["init_kwargs"]["n_layers"] == 3
        assert captured["init_kwargs"]["hidden_dim"] == 48
        assert captured["fit_kwargs"]["epochs"] == 7
        assert captured["fit_kwargs"]["batch_size"] == 33
        assert captured["fit_kwargs"]["learning_rate"] == 5e-4

    def test_integrate_donor_sources_models_unrelated_tax_variables_in_separate_blocks(
        self,
        monkeypatch,
    ):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [44, 21],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [58_000.0, 13_000.0],
                "qualified_dividend_income": [20.0, 7.0],
                "non_qualified_dividend_income": [8.0, 3.0],
                "partnership_s_corp_income": [1_000.0, 200.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "qualified_dividend_income",
                            "non_qualified_dividend_income",
                            "partnership_s_corp_income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        target_var_calls: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *args, **kwargs):
                _ = args
                self.target_vars = tuple(kwargs.get("target_vars", []))
                target_var_calls.append(self.target_vars)

            def fit(self, *args, **kwargs):
                _ = args
                _ = kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                if self.target_vars == ("dividend_income", "qualified_dividend_share"):
                    result["dividend_income"] = [28.0, 10.0]
                    result["qualified_dividend_share"] = [20.0 / 28.0, 0.7]
                if self.target_vars == ("partnership_s_corp_income",):
                    result["partnership_s_corp_income"] = [1_000.0, 200.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert target_var_calls == [
            ("dividend_income", "qualified_dividend_share"),
            ("partnership_s_corp_income",),
        ]
        assert integration["seed_data"]["qualified_dividend_income"].round(
            6
        ).tolist() == [
            20.0,
            7.0,
        ]
        assert integration["seed_data"]["non_qualified_dividend_income"].round(
            6
        ).tolist() == [
            8.0,
            3.0,
        ]
        assert integration["seed_data"]["partnership_s_corp_income"].round(
            6
        ).tolist() == [
            1_000.0,
            200.0,
        ]

    def test_integrate_donor_sources_can_use_zi_qrf_backend(self, monkeypatch):
        captured: dict[str, object] = {}

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [44, 21],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [58_000.0, 13_000.0],
                "public_assistance": [200.0, 0.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="benefit_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "public_assistance",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        class FakeQRFImputer:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def fit(self, frame, **kwargs):
                captured["fit_columns"] = list(frame.columns)
                captured["fit_kwargs"] = kwargs
                return self

            def generate(self, frame, seed=None):
                _ = seed
                return frame.assign(public_assistance=[190.0, 10.0])

        monkeypatch.setattr(
            "microplex_us.pipelines.us.ColumnwiseQRFDonorImputer",
            FakeQRFImputer,
        )

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_backend="zi_qrf",
                donor_imputer_qrf_n_estimators=77,
                donor_imputer_qrf_zero_threshold=0.1,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert integration["integrated_variables"] == ["public_assistance"]
        assert captured["init_kwargs"]["n_estimators"] == 77
        assert captured["init_kwargs"]["zero_threshold"] == 0.1
        assert captured["init_kwargs"]["zero_inflated_vars"] == {"public_assistance"}
        assert captured["init_kwargs"]["nonnegative_vars"] == set()
        assert "weight" in captured["fit_columns"]
        assert captured["fit_kwargs"]["weight_col"] == "weight"
        assert set(integration["seed_data"]["public_assistance"].tolist()) <= {
            0.0,
            200.0,
        }

    def test_support_sensitive_donor_vars_do_not_force_clamps(self, monkeypatch):
        captured: dict[str, dict[str, object]] = {}

        class FakeRegimeAwareDonorImputer:
            def __init__(self, **kwargs):
                captured["regime_aware"] = kwargs

        class FakeQRFImputer:
            def __init__(self, **kwargs):
                captured["zi_qrf"] = kwargs

        monkeypatch.setattr(
            "microplex_us.pipelines.us.RegimeAwareDonorImputer",
            FakeRegimeAwareDonorImputer,
        )
        monkeypatch.setattr(
            "microplex_us.pipelines.us.ColumnwiseQRFDonorImputer",
            FakeQRFImputer,
        )

        target_vars = ("partnership_s_corp_income", "public_assistance")

        regime_pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                donor_imputer_backend="regime_aware",
                donor_imputer_qrf_n_estimators=77,
                donor_imputer_qrf_max_train_samples=1234,
            )
        )
        regime_pipeline._build_donor_imputer(
            condition_vars=["age"],
            target_vars=target_vars,
        )

        qrf_pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                donor_imputer_backend="zi_qrf",
            )
        )
        qrf_pipeline._build_donor_imputer(
            condition_vars=["age"],
            target_vars=target_vars,
        )

        assert "nonnegative_vars" not in captured["regime_aware"]
        assert captured["regime_aware"]["n_estimators"] == 77
        assert captured["regime_aware"]["max_train_samples"] == 1234
        assert captured["zi_qrf"]["nonnegative_vars"] == set()
        assert captured["zi_qrf"]["zero_inflated_vars"] == {
            "partnership_s_corp_income",
            "public_assistance",
        }

    def test_integrate_donor_sources_preserves_informative_scaffold_values(
        self, monkeypatch
    ):
        cps_households = pd.DataFrame(
            {
                "household_id": [1],
                "hh_weight": [100.0],
                "state_fips": [6],
                "tenure": [1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10],
                "household_id": [1],
                "age": [45],
                "sex": [1],
                "education": [3],
                "employment_status": [1],
                "income": [60_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101],
                "hh_weight": [80.0],
                "state_fips": [6],
                "tenure": [1],
                "household_weight": [999.0],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001],
                "household_id": [101],
                "age": [44],
                "sex": [1],
                "education": [3],
                "employment_status": [0],
                "income": [5.0],
                "qualified_dividend_income": [20.0],
                "non_qualified_dividend_income": [8.0],
                "tax_unit_id": [12345],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure", "household_weight"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "qualified_dividend_income",
                            "non_qualified_dividend_income",
                            "tax_unit_id",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        class FakeSynthesizer:
            def __init__(self, *args, **kwargs):
                _ = args, kwargs
                self.target_vars = tuple(kwargs.get("target_vars", []))

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                if self.target_vars == ("dividend_income", "qualified_dividend_share"):
                    result["dividend_income"] = [28.0]
                    result["qualified_dividend_share"] = [20.0 / 28.0]
                if self.target_vars == ("income",):
                    result["income"] = [5.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(n_synthetic=1, synthesis_backend="bootstrap")
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)
        seed_data["income"] = [60_000.0]

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert "household_weight" not in integration["integrated_variables"]
        assert "tax_unit_id" not in integration["integrated_variables"]
        assert "income" not in integration["integrated_variables"]
        assert integration["seed_data"]["income"].tolist() == [60_000.0]

    def test_integrate_donor_sources_allows_authoritative_override_for_shared_irs_variables(
        self, monkeypatch
    ):
        captured: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = kwargs
                self.target_vars = tuple(target_vars)
                captured.append(tuple(condition_vars))

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                if self.target_vars == ("self_employment_income",):
                    result["self_employment_income"] = np.linspace(
                        -3.0,
                        3.0,
                        len(result),
                    )
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 110.0, 120.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 28, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 1, 0],
                "income": [60_000.0, 25_000.0, 12_000.0],
                "self_employment_income": [75.0, 100.0, 50.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 29, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 1, 0],
                "income": [58_000.0, 26_000.0, 13_000.0],
                "self_employment_income": [-250.0, 0.0, 500.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "self_employment_income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "self_employment_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "self_employment_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    )
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=3,
                synthesis_backend="bootstrap",
                donor_imputer_authoritative_override_variables=(
                    "self_employment_income",
                ),
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert "self_employment_income" in integration["integrated_variables"]
        assert captured[-1] == (
            "age",
            "education",
            "employment_status",
            "income",
            "sex",
            "state_fips",
            "tenure",
        )
        assert integration["seed_data"]["self_employment_income"].tolist() == [
            -250.0,
            0.0,
            500.0,
        ]

    def test_integrate_donor_sources_appends_puf_support_clone_before_later_donors(
        self, monkeypatch
    ):
        generated_lengths: list[tuple[tuple[str, ...], int]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = condition_vars, kwargs
                self.target_vars = tuple(target_vars)

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                generated_lengths.append((self.target_vars, len(frame)))
                result = frame.copy()
                for target in self.target_vars:
                    result[target] = np.linspace(1.0, float(len(result)), len(result))
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 200.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 62],
                "sex": [1, 2],
                "education": [3, 4],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
                "self_employment_income": [75.0, 50.0],
                "taxpayer_id_type": [1, 2],
            }
        )
        puf_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        puf_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [44, 61],
                "sex": [1, 2],
                "education": [3, 4],
                "employment_status": [1, 0],
                "income": [58_000.0, 13_000.0],
                "self_employment_income": [-250.0, 500.0],
                "taxable_interest_income": [10.0, 20.0],
                "state_income_tax_paid": [400.0, 50.0],
            }
        )
        sipp_households = pd.DataFrame(
            {
                "household_id": [201, 202],
                "hh_weight": [70.0, 75.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        sipp_persons = pd.DataFrame(
            {
                "person_id": [2001, 2002],
                "household_id": [201, 202],
                "age": [45, 62],
                "sex": [1, 2],
                "education": [3, 4],
                "employment_status": [1, 0],
                "income": [59_000.0, 14_000.0],
                "ssi_reported": [0.0, 100.0],
            }
        )

        def frame_for(name, households, persons, capabilities):
            return ObservationFrame(
                source=SourceDescriptor(
                    name=name,
                    shareability=Shareability.PUBLIC
                    if name.startswith("cps")
                    else Shareability.RESTRICTED,
                    time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                    observations=(
                        EntityObservation(
                            entity=EntityType.HOUSEHOLD,
                            key_column="household_id",
                            variable_names=("state_fips", "tenure"),
                            weight_column="hh_weight",
                        ),
                        EntityObservation(
                            entity=EntityType.PERSON,
                            key_column="person_id",
                            variable_names=tuple(
                                column
                                for column in persons.columns
                                if column != "person_id"
                            ),
                        ),
                    ),
                    variable_capabilities={
                        variable: SourceVariableCapability(
                            authoritative=True,
                            usable_as_condition=True,
                        )
                        for variable in capabilities
                    },
                ),
                tables={
                    EntityType.HOUSEHOLD: households,
                    EntityType.PERSON: persons,
                },
                relationships=(
                    EntityRelationship(
                        parent_entity=EntityType.HOUSEHOLD,
                        child_entity=EntityType.PERSON,
                        parent_key="household_id",
                        child_key="household_id",
                        cardinality=RelationshipCardinality.ONE_TO_MANY,
                    ),
                ),
            )

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
                puf_support_clone_overlap_variables=("self_employment_income",),
                puf_support_clone_both_halves_override_variables=(),
            )
        )
        cps_input = pipeline.prepare_source_input(
            frame_for(
                "cps_asec_test", cps_households, cps_persons, ("taxpayer_id_type",)
            )
        )
        puf_input = pipeline.prepare_source_input(
            frame_for(
                "irs_soi_puf_2024",
                puf_households,
                puf_persons,
                (
                    "self_employment_income",
                    "taxable_interest_income",
                    "state_income_tax_paid",
                ),
            )
        )
        sipp_input = pipeline.prepare_source_input(
            frame_for("sipp_2023", sipp_households, sipp_persons, ("ssi_reported",))
        )
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[sipp_input, puf_input],
        )
        result = integration["seed_data"]

        assert integration["processed_donor_source_order"] == [
            "irs_soi_puf_2024",
            "sipp_2023",
        ]
        assert integration["puf_clone_source_order"] == ["irs_soi_puf_2024"]
        assert result["person_is_puf_clone"].tolist() == [0.0, 0.0, 1.0, 1.0]
        assert result["hh_weight"].tolist() == [100.0, 200.0, 0.0, 0.0]
        assert result["self_employment_income"].iloc[:2].tolist() == [75.0, 50.0]
        assert result["self_employment_income"].iloc[2:].tolist() == [-250.0, 500.0]
        assert result["taxpayer_id_type"].tolist() == [1, 2, 1, 2]
        assert result["taxable_interest_income"].iloc[:2].tolist() == [0.0, 0.0]
        assert result["taxable_interest_income"].iloc[2:].tolist() == [10.0, 20.0]
        assert "state_income_tax_paid" in result.columns
        assert "tax_unit_id" not in result.columns
        assert integration["puf_support_clone_summary"][
            "dropped_generated_entity_id_columns"
        ] == ["tax_unit_id"]
        assert result.index.tolist() == [0, 1, 2, 3]
        assert generated_lengths[-1] == (("ssi_reported",), 4)
        assert "ssi_reported" in result.columns

    def test_finalize_puf_support_clone_can_collapse_donor_only_values_to_cps_rows(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
                puf_support_clone_output_mode="collapse_to_scaffold",
                puf_support_clone_both_halves_override_variables=(),
            )
        )
        original = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 62],
                "self_employment_income": [75.0, 50.0],
            }
        )
        clone = pd.DataFrame(
            {
                "person_id": [30, 40],
                "household_id": [3, 4],
                "age": [45, 62],
                us_pipeline_module.PUF_SUPPORT_CLONE_SOURCE_ROW_ID_COLUMN: [0, 1],
                "self_employment_income": [-250.0, 500.0],
                "taxable_interest_income": [10.0, 20.0],
                "partnership_s_corp_income": [-700.0, 1_200.0],
                "state_income_tax_paid": [400.0, 50.0],
                "tax_unit_id": [101, 102],
            }
        )

        result, summary = pipeline._finalize_puf_support_clone_frame(
            original=original,
            imputed_clone=clone,
            donor_source_name="irs_soi_puf_2024",
            integrated_variables=[
                "self_employment_income",
                "taxable_interest_income",
                "partnership_s_corp_income",
                "state_income_tax_paid",
                "tax_unit_id",
            ],
            preclone_columns=set(original.columns),
            donor_seed_columns=set(clone.columns),
            donor_observed=set(clone.columns),
        )

        assert result.index.tolist() == [0, 1]
        assert result["person_is_puf_clone"].tolist() == [0.0, 0.0]
        assert result["person_id"].tolist() == [10, 20]
        assert result["household_id"].tolist() == [1, 2]
        assert result["self_employment_income"].tolist() == [-250.0, 500.0]
        assert result["taxable_interest_income"].tolist() == [10.0, 20.0]
        assert result["partnership_s_corp_income"].tolist() == [-700.0, 1_200.0]
        assert result["state_income_tax_paid"].tolist() == [400.0, 50.0]
        assert "tax_unit_id" not in result.columns
        assert summary["output_mode"] == "collapse_to_scaffold"
        assert summary["clone_row_count"] == 2
        assert summary["emitted_clone_row_count"] == 0
        assert summary["final_row_count"] == 2
        assert summary["dropped_generated_entity_id_columns"] == ["tax_unit_id"]
        assert summary["collapse_copy_variables"] == [
            "partnership_s_corp_income",
            "self_employment_income",
            "state_income_tax_paid",
            "taxable_interest_income",
        ]
        assert summary["overlap_collapse_override_variables"] == [
            "self_employment_income",
        ]
        assert summary["source_row_alignment"] == {
            "enabled": True,
            "column": us_pipeline_module.PUF_SUPPORT_CLONE_SOURCE_ROW_ID_COLUMN,
            "row_count": 2,
            "clone_was_reordered": False,
        }

    def test_finalize_puf_support_clone_preserves_puf_tax_details_by_default(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
                puf_support_clone_output_mode="collapse_to_scaffold",
                puf_support_clone_both_halves_override_variables=(),
            )
        )
        original = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 62],
                "employment_income": [30_000.0, 10_000.0],
                "self_employment_income": [500.0, 250.0],
                "long_term_capital_gains": [1_000.0, 0.0],
                "short_term_capital_gains": [100.0, 0.0],
                "capital_gains": [1_100.0, 0.0],
                "interest_income": [3.0, 4.0],
                # Regression coverage for preclone components: these may exist on
                # the CPS scaffold already, but PUF-integrated leaves must still
                # survive collapse back to the scaffold rows.
                "taxable_interest_income": [3.0, 4.0],
                "tax_exempt_interest_income": [3.0, 4.0],
                "dividend_income": [10.0, 5.0],
                "qualified_dividend_income": [10.0, 5.0],
                "non_qualified_dividend_income": [10.0, 5.0],
                "pension_income": [100.0, 200.0],
                "taxable_pension_income": [100.0, 200.0],
                "tax_exempt_pension_income": [100.0, 200.0],
                "unemployment_compensation": [100.0, 0.0],
                "taxable_unemployment_compensation": [100.0, 0.0],
            }
        )
        clone = pd.DataFrame(
            {
                "person_id": [30, 40],
                "household_id": [3, 4],
                "age": [45, 62],
                us_pipeline_module.PUF_SUPPORT_CLONE_SOURCE_ROW_ID_COLUMN: [0, 1],
                "employment_income": [90_000.0, 20_000.0],
                "self_employment_income": [-4_000.0, 8_000.0],
                "long_term_capital_gains": [50_000.0, -1_000.0],
                "short_term_capital_gains": [2_500.0, -500.0],
                "capital_gains": [52_500.0, -1_500.0],
                "taxable_interest_income": [1_000.0, 0.0],
                "tax_exempt_interest_income": [500.0, 0.0],
                "qualified_dividend_income": [20.0, 0.0],
                "non_qualified_dividend_income": [5.0, 0.0],
                "ordinary_dividend_income": [25.0, 0.0],
                "dividend_income": [25.0, 0.0],
                "taxable_pension_income": [90.0, 0.0],
                "tax_exempt_pension_income": [10.0, 0.0],
                "taxable_unemployment_compensation": [600.0, 700.0],
            }
        )

        result, summary = pipeline._finalize_puf_support_clone_frame(
            original=original,
            imputed_clone=clone,
            donor_source_name="irs_soi_puf_2024",
            integrated_variables=[
                "taxable_interest_income",
                "tax_exempt_interest_income",
                "employment_income",
                "self_employment_income",
                "long_term_capital_gains",
                "short_term_capital_gains",
                "capital_gains",
                "qualified_dividend_income",
                "non_qualified_dividend_income",
                "taxable_pension_income",
                "tax_exempt_pension_income",
                "taxable_unemployment_compensation",
            ],
            preclone_columns=set(original.columns),
            donor_seed_columns=set(clone.columns),
            donor_observed=set(clone.columns),
        )

        assert result["employment_income"].tolist() == [90_000.0, 20_000.0]
        assert result["self_employment_income"].tolist() == [-4_000.0, 8_000.0]
        assert result["long_term_capital_gains"].tolist() == [50_000.0, -1_000.0]
        assert result["short_term_capital_gains"].tolist() == [2_500.0, -500.0]
        assert result["capital_gains"].tolist() == [52_500.0, -1_500.0]
        assert result["taxable_interest_income"].tolist() == [1_000.0, 0.0]
        assert result["tax_exempt_interest_income"].tolist() == [500.0, 0.0]
        assert result["interest_income"].tolist() == [1_500.0, 0.0]
        assert result["taxable_unemployment_compensation"].tolist() == [600.0, 700.0]
        assert result["unemployment_compensation"].tolist() == [600.0, 700.0]
        assert result["dividend_income"].tolist() == [25.0, 0.0]
        assert result["ordinary_dividend_income"].tolist() == [25.0, 0.0]
        assert result["qualified_dividend_income"].tolist() == [20.0, 0.0]
        assert result["non_qualified_dividend_income"].tolist() == [5.0, 0.0]
        assert result["taxable_pension_income"].tolist() == [90.0, 0.0]
        assert result["tax_exempt_pension_income"].tolist() == [10.0, 0.0]
        assert result["pension_income"].tolist() == [100.0, 0.0]
        passthrough = summary["cps_measured_total_passthrough"]
        assert passthrough["enabled"] is False
        assert passthrough["passthrough_variables"] == []
        assert passthrough["dividend_components_scaled_to_cps_total"] is False
        assert set(passthrough["identity_reconciled_variables"]) >= {
            "dividend_income",
            "interest_income",
            "ordinary_dividend_income",
            "pension_income",
            "unemployment_compensation",
        }
        assert set(summary["collapse_copy_variables"]) >= {
            "dividend_income",
            "employment_income",
            "interest_income",
            "long_term_capital_gains",
            "non_qualified_dividend_income",
            "ordinary_dividend_income",
            "pension_income",
            "qualified_dividend_income",
            "self_employment_income",
            "short_term_capital_gains",
            "tax_exempt_interest_income",
            "tax_exempt_pension_income",
            "taxable_interest_income",
            "taxable_pension_income",
            "taxable_unemployment_compensation",
            "unemployment_compensation",
        }
        assert set(summary["overlap_collapse_override_variables"]) >= {
            "capital_gains",
            "employment_income",
            "long_term_capital_gains",
            "self_employment_income",
            "short_term_capital_gains",
            "tax_exempt_interest_income",
            "tax_exempt_pension_income",
            "taxable_interest_income",
            "taxable_pension_income",
            "taxable_unemployment_compensation",
        }
        assert summary["source_row_alignment"]["clone_was_reordered"] is False

    def test_finalize_puf_support_clone_aligns_shuffled_clone_by_source_row_id(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
                puf_support_clone_output_mode="collapse_to_scaffold",
                puf_support_clone_both_halves_override_variables=(),
            )
        )
        original = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 62],
                "self_employment_income": [75.0, 50.0],
            }
        )
        clone = pd.DataFrame(
            {
                "person_id": [40, 30],
                "household_id": [4, 3],
                "age": [62, 45],
                us_pipeline_module.PUF_SUPPORT_CLONE_SOURCE_ROW_ID_COLUMN: [1, 0],
                "self_employment_income": [500.0, -250.0],
            }
        )

        result, summary = pipeline._finalize_puf_support_clone_frame(
            original=original,
            imputed_clone=clone,
            donor_source_name="irs_soi_puf_2024",
            integrated_variables=["self_employment_income"],
            preclone_columns=set(original.columns),
            donor_seed_columns=set(clone.columns),
            donor_observed=set(clone.columns),
        )

        assert result["person_id"].tolist() == [10, 20]
        assert result["self_employment_income"].tolist() == [-250.0, 500.0]
        assert summary["source_row_alignment"] == {
            "enabled": True,
            "column": us_pipeline_module.PUF_SUPPORT_CLONE_SOURCE_ROW_ID_COLUMN,
            "row_count": 2,
            "clone_was_reordered": True,
        }

    def test_finalize_puf_support_clone_can_scale_tax_details_to_cps_totals(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
                puf_support_clone_output_mode="collapse_to_scaffold",
                puf_support_clone_both_halves_override_variables=(),
                puf_support_clone_scale_tax_details_to_cps_totals=True,
            )
        )
        original = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 62],
                "interest_income": [3.0, 4.0],
                "taxable_interest_income": [3.0, 4.0],
                "tax_exempt_interest_income": [3.0, 4.0],
                "dividend_income": [10.0, 5.0],
                "qualified_dividend_income": [10.0, 5.0],
                "non_qualified_dividend_income": [10.0, 5.0],
                "pension_income": [100.0, 200.0],
                "taxable_pension_income": [100.0, 200.0],
                "tax_exempt_pension_income": [100.0, 200.0],
                "unemployment_compensation": [100.0, 0.0],
                "taxable_unemployment_compensation": [100.0, 0.0],
            }
        )
        clone = pd.DataFrame(
            {
                "person_id": [30, 40],
                "household_id": [3, 4],
                "age": [45, 62],
                us_pipeline_module.PUF_SUPPORT_CLONE_SOURCE_ROW_ID_COLUMN: [0, 1],
                "taxable_interest_income": [1_000.0, 0.0],
                "tax_exempt_interest_income": [500.0, 0.0],
                "qualified_dividend_income": [20.0, 0.0],
                "non_qualified_dividend_income": [5.0, 0.0],
                "ordinary_dividend_income": [25.0, 0.0],
                "dividend_income": [25.0, 0.0],
                "taxable_pension_income": [90.0, 0.0],
                "tax_exempt_pension_income": [10.0, 0.0],
                "taxable_unemployment_compensation": [600.0, 700.0],
            }
        )

        result, summary = pipeline._finalize_puf_support_clone_frame(
            original=original,
            imputed_clone=clone,
            donor_source_name="irs_soi_puf_2024",
            integrated_variables=[
                "taxable_interest_income",
                "tax_exempt_interest_income",
                "qualified_dividend_income",
                "non_qualified_dividend_income",
                "taxable_pension_income",
                "tax_exempt_pension_income",
                "taxable_unemployment_compensation",
            ],
            preclone_columns=set(original.columns),
            donor_seed_columns=set(clone.columns),
            donor_observed=set(clone.columns),
        )

        assert result["taxable_interest_income"].round(6).tolist() == [2.0, 2.72]
        assert result["tax_exempt_interest_income"].round(6).tolist() == [1.0, 1.28]
        assert result["interest_income"].round(6).tolist() == [3.0, 4.0]
        assert result["taxable_unemployment_compensation"].tolist() == [100.0, 0.0]
        assert result["unemployment_compensation"].tolist() == [100.0, 0.0]
        assert result["dividend_income"].tolist() == [10.0, 5.0]
        assert result["ordinary_dividend_income"].tolist() == [10.0, 5.0]
        assert result["qualified_dividend_income"].round(6).tolist() == [
            8.0,
            3.9,
        ]
        assert result["non_qualified_dividend_income"].round(6).tolist() == [
            2.0,
            1.1,
        ]
        assert result["taxable_pension_income"].round(6).tolist() == [
            90.0,
            118.0,
        ]
        assert result["tax_exempt_pension_income"].round(6).tolist() == [
            10.0,
            82.0,
        ]
        assert result["pension_income"].round(6).tolist() == [100.0, 200.0]
        passthrough = summary["cps_measured_total_passthrough"]
        assert passthrough["enabled"] is True
        assert passthrough["passthrough_variables"] == [
            "non_qualified_dividend_income",
            "qualified_dividend_income",
            "tax_exempt_interest_income",
            "tax_exempt_pension_income",
            "taxable_interest_income",
            "taxable_pension_income",
            "taxable_unemployment_compensation",
        ]
        assert passthrough["dividend_components_scaled_to_cps_total"] is True
        assert set(passthrough["identity_reconciled_variables"]) >= {
            "dividend_income",
            "interest_income",
            "ordinary_dividend_income",
            "pension_income",
            "unemployment_compensation",
        }

    def test_integrate_donor_sources_collapses_puf_support_clone_before_later_donors(
        self, monkeypatch
    ):
        generated_lengths: list[tuple[tuple[str, ...], int]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = condition_vars, kwargs
                self.target_vars = tuple(target_vars)

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                generated_lengths.append((self.target_vars, len(frame)))
                result = frame.copy()
                for target in self.target_vars:
                    result[target] = np.linspace(1.0, float(len(result)), len(result))
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        def frame_for(name, households, persons, capabilities):
            return ObservationFrame(
                source=SourceDescriptor(
                    name=name,
                    shareability=Shareability.PUBLIC
                    if name.startswith("cps")
                    else Shareability.RESTRICTED,
                    time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                    observations=(
                        EntityObservation(
                            entity=EntityType.HOUSEHOLD,
                            key_column="household_id",
                            variable_names=("state_fips", "tenure"),
                            weight_column="hh_weight",
                        ),
                        EntityObservation(
                            entity=EntityType.PERSON,
                            key_column="person_id",
                            variable_names=tuple(
                                column
                                for column in persons.columns
                                if column != "person_id"
                            ),
                        ),
                    ),
                    variable_capabilities={
                        variable: SourceVariableCapability(
                            authoritative=True,
                            usable_as_condition=True,
                        )
                        for variable in capabilities
                    },
                ),
                tables={
                    EntityType.HOUSEHOLD: households,
                    EntityType.PERSON: persons,
                },
                relationships=(
                    EntityRelationship(
                        parent_entity=EntityType.HOUSEHOLD,
                        child_entity=EntityType.PERSON,
                        parent_key="household_id",
                        child_key="household_id",
                        cardinality=RelationshipCardinality.ONE_TO_MANY,
                    ),
                ),
            )

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 200.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 62],
                "sex": [1, 2],
                "education": [3, 4],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
                "self_employment_income": [75.0, 50.0],
                "taxpayer_id_type": [1, 2],
            }
        )
        puf_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        puf_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [44, 61],
                "sex": [1, 2],
                "education": [3, 4],
                "employment_status": [1, 0],
                "income": [58_000.0, 13_000.0],
                "self_employment_income": [-250.0, 500.0],
                "taxable_interest_income": [10.0, 20.0],
                "state_income_tax_paid": [400.0, 50.0],
            }
        )
        sipp_households = pd.DataFrame(
            {
                "household_id": [201, 202],
                "hh_weight": [70.0, 75.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        sipp_persons = pd.DataFrame(
            {
                "person_id": [2001, 2002],
                "household_id": [201, 202],
                "age": [45, 62],
                "sex": [1, 2],
                "education": [3, 4],
                "employment_status": [1, 0],
                "income": [59_000.0, 14_000.0],
                "ssi_reported": [0.0, 100.0],
            }
        )

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
                puf_support_clone_output_mode="collapse_to_scaffold",
                puf_support_clone_overlap_variables=("self_employment_income",),
                puf_support_clone_both_halves_override_variables=(),
            )
        )
        cps_input = pipeline.prepare_source_input(
            frame_for(
                "cps_asec_test", cps_households, cps_persons, ("taxpayer_id_type",)
            )
        )
        puf_input = pipeline.prepare_source_input(
            frame_for(
                "irs_soi_puf_2024",
                puf_households,
                puf_persons,
                (
                    "self_employment_income",
                    "taxable_interest_income",
                    "state_income_tax_paid",
                ),
            )
        )
        sipp_input = pipeline.prepare_source_input(
            frame_for("sipp_2023", sipp_households, sipp_persons, ("ssi_reported",))
        )
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[sipp_input, puf_input],
        )
        result = integration["seed_data"]

        assert result.index.tolist() == [0, 1]
        assert result["person_is_puf_clone"].tolist() == [0.0, 0.0]
        assert result["hh_weight"].tolist() == [100.0, 200.0]
        assert result["self_employment_income"].tolist() == [-250.0, 500.0]
        assert result["taxable_interest_income"].tolist() == [10.0, 20.0]
        assert sorted(result["state_income_tax_paid"].tolist()) == [50.0, 400.0]
        assert integration["puf_support_clone_summary"]["output_mode"] == (
            "collapse_to_scaffold"
        )
        assert integration["puf_support_clone_summary"]["final_row_count"] == 2
        assert integration["puf_support_clone_summary"]["emitted_clone_row_count"] == 0
        assert generated_lengths[-1] == (("ssi_reported",), 2)

    def test_puf_support_clone_refresh_rematches_cps_only_disability_to_puf_income(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
            )
        )
        original = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [1, 2],
                "age": [40, 40],
                "is_male": [1, 1],
                "state_fips": [6, 6],
                "employment_income": [0.0, 100_000.0],
                "self_employment_income": [0.0, 0.0],
                "social_security": [0.0, 0.0],
                "is_disabled": [1, 0],
                "difficulty_hearing": [1, 0],
                "meets_ssi_disability_criteria": [1, 0],
            }
        )
        clone = original.copy()
        clone["employment_income"] = [100_000.0, 0.0]

        refreshed, summary = pipeline._refresh_puf_support_clone_cps_only_fields(
            original=original,
            clone=clone,
            integrated_variables=["employment_income"],
            preclone_columns=set(original.columns),
        )

        assert refreshed["is_disabled"].tolist() == [0, 1]
        assert refreshed["difficulty_hearing"].tolist() == [0, 1]
        assert refreshed["meets_ssi_disability_criteria"].tolist() == [0, 1]
        assert "employment_income" in summary["condition_variables"]
        assert summary["matched_source_row_count"] == 2
        assert "is_disabled" in summary["refreshed_variables"]

    def test_puf_support_clone_refresh_does_not_overwrite_amount_fields(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
            )
        )
        original = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [1, 2],
                "age": [40, 40],
                "is_male": [1, 1],
                "state_fips": [6, 6],
                "employment_income": [0.0, 100_000.0],
                "self_employment_income": [0.0, 0.0],
                "social_security": [0.0, 0.0],
                "is_disabled": [1, 0],
                "disability_benefits": [4_000.0, 0.0],
                "weekly_hours_worked": [0.0, 40.0],
                "taxable_401k_distributions": [0.0, 2_000.0],
            }
        )
        clone = original.copy()
        clone["employment_income"] = [100_000.0, 0.0]
        clone["disability_benefits"] = [123_456.0, 789_012.0]
        clone["weekly_hours_worked"] = [12.0, 34.0]
        clone["taxable_401k_distributions"] = [56.0, 78.0]

        refreshed, summary = pipeline._refresh_puf_support_clone_cps_only_fields(
            original=original,
            clone=clone,
            integrated_variables=["employment_income"],
            preclone_columns=set(original.columns),
        )

        assert refreshed["is_disabled"].tolist() == [0, 1]
        assert refreshed["disability_benefits"].tolist() == [123_456.0, 789_012.0]
        assert refreshed["weekly_hours_worked"].tolist() == [12.0, 34.0]
        assert refreshed["taxable_401k_distributions"].tolist() == [56.0, 78.0]
        assert "is_disabled" in summary["refreshed_variables"]
        assert "disability_benefits" not in summary["refreshed_variables"]
        assert "weekly_hours_worked" not in summary["refreshed_variables"]
        assert "taxable_401k_distributions" not in summary["refreshed_variables"]

    def test_puf_support_clone_refresh_reconciles_social_security_subcomponents(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
            )
        )
        clone = pd.DataFrame(
            {
                "age": [45, 70, 40],
                "social_security": [12_000.0, 8_000.0, 0.0],
                "social_security_retirement": [0.0, 2_000.0, 100.0],
                "social_security_disability": [3_000.0, 0.0, 50.0],
            }
        )

        reconciled = pipeline._reconcile_puf_support_clone_social_security(clone)

        assert reconciled == [
            "social_security_retirement",
            "social_security_disability",
        ]
        assert clone["social_security_disability"].tolist() == [12_000.0, 0.0, 0.0]
        assert clone["social_security_retirement"].tolist() == [0.0, 8_000.0, 0.0]

    def test_integrate_donor_sources_puf_support_clone_validates_scaffold_and_donor(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                synthesis_backend="seed",
                puf_support_clone_enabled=True,
            )
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_asec_test",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips",),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=("household_id", "age", "income"),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: pd.DataFrame(
                    {"household_id": [1], "hh_weight": [1.0], "state_fips": [6]}
                ),
                EntityType.PERSON: pd.DataFrame(
                    {
                        "person_id": [1],
                        "household_id": [1],
                        "age": [40],
                        "income": [1.0],
                    }
                ),
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        cps_input = pipeline.prepare_source_input(frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        with pytest.raises(ValueError, match="requires exactly one PUF donor"):
            pipeline._integrate_donor_sources(
                seed_data,
                scaffold_input=cps_input,
                donor_inputs=[],
            )

    def test_integrate_donor_sources_zeroes_minor_employment_income_after_authoritative_override(
        self, monkeypatch
    ):
        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = condition_vars, kwargs
                self.target_vars = tuple(target_vars)

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                if self.target_vars == ("employment_income",):
                    result["employment_income"] = np.linspace(1.0, 2.0, len(result))
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [16, 35],
                "sex": [1, 2],
                "education": [1, 3],
                "employment_status": [0, 1],
                "income": [5_000.0, 55_000.0],
                "employment_income": [500.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [90.0, 110.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [17, 36],
                "sex": [1, 2],
                "education": [1, 3],
                "employment_status": [0, 1],
                "income": [6_000.0, 56_000.0],
                "employment_income": [50_000.0, 80_000.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "employment_income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "employment_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "employment_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=False,
                    )
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                donor_imputer_authoritative_override_variables=("employment_income",),
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert "employment_income" in integration["integrated_variables"]
        assert integration["seed_data"]["employment_income"].tolist() == [
            0.0,
            80_000.0,
        ]

    def test_integrate_donor_sources_zeroes_retired_senior_employment_income_without_esi(
        self, monkeypatch
    ):
        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = condition_vars, kwargs
                self.target_vars = tuple(target_vars)

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                if self.target_vars == ("employment_income",):
                    result["employment_income"] = np.linspace(
                        70_000.0, 90_000.0, len(result)
                    )
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 130.0],
                "state_fips": [6, 36, 48],
                "tenure": [1, 2, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [68, 68, 68],
                "sex": [1, 2, 1],
                "education": [3, 3, 3],
                "employment_status": [1, 1, 1],
                "income": [45_000.0, 65_000.0, 50_000.0],
                "employment_income": [30_000.0, 40_000.0, 35_000.0],
                "social_security_retirement": [18_000.0, 18_000.0, 0.0],
                "has_esi": [0.0, 1.0, 0.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [90.0, 110.0, 105.0],
                "state_fips": [6, 36, 48],
                "tenure": [1, 2, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [68, 68, 68],
                "sex": [1, 2, 1],
                "education": [3, 3, 3],
                "employment_status": [1, 1, 1],
                "income": [46_000.0, 66_000.0, 51_000.0],
                "employment_income": [80_000.0, 85_000.0, 82_000.0],
                "social_security_retirement": [19_000.0, 19_000.0, 0.0],
                "has_esi": [0.0, 1.0, 0.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "employment_income",
                            "social_security_retirement",
                            "has_esi",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "employment_income",
                            "social_security_retirement",
                            "has_esi",
                        ),
                    ),
                ),
                variable_capabilities={
                    "employment_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=False,
                    )
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=3,
                synthesis_backend="bootstrap",
                donor_imputer_authoritative_override_variables=("employment_income",),
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert "employment_income" in integration["integrated_variables"]
        employment_income = integration["seed_data"]["employment_income"].tolist()
        assert employment_income[0] == 0.0
        assert employment_income[1] > 0.0
        assert employment_income[2] > 0.0

    def test_integrate_donor_sources_normalizes_social_security_before_senior_wage_guard(
        self, monkeypatch
    ):
        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = condition_vars, kwargs
                self.target_vars = tuple(target_vars)

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                if self.target_vars == ("employment_income",):
                    result["employment_income"] = [70_000.0, 90_000.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [68, 68],
                "sex": [1, 2],
                "education": [3, 3],
                "employment_status": [1, 1],
                "income": [45_000.0, 65_000.0],
                "employment_income": [30_000.0, 40_000.0],
                "social_security": [18_000.0, 0.0],
                "has_esi": [0.0, 0.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [90.0, 110.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [68, 68],
                "sex": [1, 2],
                "education": [3, 3],
                "employment_status": [1, 1],
                "income": [46_000.0, 66_000.0],
                "employment_income": [80_000.0, 85_000.0],
                "social_security": [19_000.0, 0.0],
                "has_esi": [0.0, 0.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "employment_income",
                            "social_security",
                            "has_esi",
                        ),
                    ),
                ),
                variable_capabilities={
                    "employment_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=False,
                    )
                },
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "employment_income",
                            "social_security",
                            "has_esi",
                        ),
                    ),
                ),
                variable_capabilities={
                    "employment_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=False,
                    )
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                donor_imputer_authoritative_override_variables=("employment_income",),
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert integration["seed_data"]["social_security_retirement"].tolist() == [
            0.0,
            0.0,
        ]
        assert integration["seed_data"]["social_security_unclassified"].tolist() == [
            18_000.0,
            0.0,
        ]
        assert integration["seed_data"]["employment_income"].tolist() == [0.0, 85_000.0]

    def test_export_policyengine_dataset(self, persons, households, tmp_path):
        config = USMicroplexBuildConfig(
            n_synthetic=8,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            policyengine_dataset_year=2024,
        )
        result = build_us_microplex(persons, households, config)
        pipeline = USMicroplexPipeline(config)

        output_path = pipeline.export_policyengine_dataset(
            result, tmp_path / "us_microplex.h5"
        )

        assert output_path.exists()
        with h5py.File(output_path, "r") as handle:
            assert "county_fips" in handle
            exported_counties = handle["county_fips"]["2024"][()]
        normalized_counties = {
            str(value.decode() if isinstance(value, bytes) else value).zfill(5)
            for value in np.asarray(exported_counties).tolist()
        }
        assert normalized_counties == {"06037", "36061", "48201"}

    def test_export_policyengine_dataset_passes_direct_overrides(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        captured: list[tuple[str, ...]] = []

        original_build_maps = build_policyengine_us_export_variable_maps

        def _capture_build_maps(*args, **kwargs):
            captured.append(tuple(kwargs.get("direct_override_variables", ())))
            return original_build_maps(*args, **kwargs)

        monkeypatch.setattr(
            "microplex_us.pipelines.us.build_policyengine_us_export_variable_maps",
            _capture_build_maps,
        )

        config = USMicroplexBuildConfig(
            n_synthetic=8,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            policyengine_dataset_year=2024,
            policyengine_direct_override_variables=("filing_status",),
        )
        result = build_us_microplex(persons, households, config)
        pipeline = USMicroplexPipeline(config)

        output_path = pipeline.export_policyengine_dataset(
            result, tmp_path / "us_microplex.h5"
        )

        assert output_path.exists()
        assert captured == [("filing_status",)]

    def test_export_policyengine_dataset_normalizes_checkpoint_person_inputs(
        self,
        tmp_path,
        monkeypatch,
    ):
        captured_persons: list[pd.DataFrame] = []

        def _identity_marketplace_ratio(self, tables, *, target_period):
            return tables

        def _fake_build_maps(tables, **kwargs):
            captured_persons.append(tables.persons.copy())
            return {
                "household": {},
                "person": {
                    "rental_income": "rental_income",
                    "farm_income": "farm_income",
                },
                "tax_unit": {},
                "spm_unit": {},
                "family": {},
            }

        def _fake_arrays(*args, **kwargs):
            return {}

        def _fake_write(arrays, path, **kwargs):
            Path(path).write_text("h5 placeholder")
            return Path(path)

        monkeypatch.setattr(
            USMicroplexPipeline,
            "_attach_policyengine_marketplace_plan_benchmark_ratio",
            _identity_marketplace_ratio,
        )
        monkeypatch.setattr(
            USMicroplexPipeline,
            "_resolve_policyengine_tax_benefit_system",
            lambda self: SimpleNamespace(variables={}),
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "build_policyengine_us_export_variable_maps",
            _fake_build_maps,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "resolve_policyengine_excluded_export_variables",
            lambda *args, **kwargs: set(),
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "build_policyengine_us_time_period_arrays",
            _fake_arrays,
        )
        monkeypatch.setattr(
            us_pipeline_module,
            "write_policyengine_us_time_period_dataset",
            _fake_write,
        )

        config = USMicroplexBuildConfig(policyengine_dataset_year=2024)
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame({"household_id": [1], "household_weight": [1.0]}),
            persons=pd.DataFrame(
                {
                    "person_id": [10, 20, 30],
                    "household_id": [1, 1, 1],
                    "age": [45, 50, 55],
                    "sex": [1, 2, 1],
                    "income": [1_000.0, 1_000.0, 1_000.0],
                    "rental_income": [900.0, 900.0, 900.0],
                    "rental_income_positive": [300.0, 0.0, 50.0],
                    "rental_income_negative": [100.0, 200.0, 0.0],
                    "farm_income": [20.0, 30.0, 40.0],
                    "farm_operations_income": [10.0, -15.0, 0.0],
                }
            ),
        )
        result = USMicroplexBuildResult(
            config=config,
            seed_data=pd.DataFrame(),
            synthetic_data=pd.DataFrame(),
            calibrated_data=pd.DataFrame(),
            targets=USMicroplexTargets(marginal={}, continuous={}),
            calibration_summary={},
            policyengine_tables=tables,
        )

        output_path = USMicroplexPipeline(config).export_policyengine_dataset(
            result,
            tmp_path / "us_microplex.h5",
        )

        assert output_path.exists()
        assert captured_persons[0]["rental_income"].tolist() == [
            200.0,
            -200.0,
            50.0,
        ]
        assert captured_persons[0]["farm_income"].tolist() == [10.0, -15.0, 40.0]

    def test_augment_policyengine_person_inputs_materializes_non_sch_d_capital_gains(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "non_sch_d_capital_gains": [250.0],
                "age": [45],
                "sex": [1],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["non_sch_d_capital_gains"].tolist() == [250.0]

    def test_augment_policyengine_person_inputs_aliases_rent_to_pre_subsidy_rent(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "rent": [14_400.0, 0.0, 9_600.0],
                "pre_subsidy_rent": [0.0, 7_200.0, None],
                "age": [45, 70, 12],
                "sex": [1, 2, 1],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["pre_subsidy_rent"].tolist() == [
            14_400.0,
            7_200.0,
            9_600.0,
        ]

    def test_augment_policyengine_person_inputs_recomposes_signed_rental_income(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "age": [45, 50, 55],
                "sex": [1, 2, 1],
                "income": [1_000.0, 1_000.0, 1_000.0],
                "rental_income": [900.0, 900.0, 900.0],
                "rental_income_positive": [300.0, 0.0, 50.0],
                "rental_income_negative": [100.0, 200.0, 0.0],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["rental_income"].tolist() == [200.0, -200.0, 50.0]
        assert augmented["employment_income_before_lsr"].tolist() == [
            800.0,
            1_200.0,
            950.0,
        ]

    def test_augment_policyengine_person_inputs_prefers_signed_business_losses(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "age": [45, 50, 55],
                "sex": [1, 2, 1],
                "income": [1_000.0, 1_000.0, 1_000.0],
                "self_employment_income_before_lsr": [50.0, 60.0, 70.0],
                "self_employment_income": [100.0, -25.0, 0.0],
                "farm_income": [20.0, 30.0, 40.0],
                "farm_operations_income": [10.0, -15.0, 0.0],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["self_employment_income_before_lsr"].tolist() == [
            100.0,
            -25.0,
            70.0,
        ]
        assert augmented["farm_income"].tolist() == [10.0, -15.0, 40.0]
        assert augmented["employment_income_before_lsr"].tolist() == [
            900.0,
            1_025.0,
            930.0,
        ]

    def test_augment_policyengine_person_inputs_zeros_part_b_without_medicare(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "medicare_part_b_premiums": [100.0, 200.0, -30.0, 400.0],
                "has_medicare": [0, 1, 0, 1],
                "age": [12, 70, 45, 58],
                "sex": [1, 2, 1, 2],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["has_medicare"].tolist() == [False, True, False, True]
        assert augmented["medicare_part_b_premiums"].tolist() == [
            0.0,
            200.0,
            0.0,
            400.0,
        ]

    def test_augment_policyengine_person_inputs_derives_blind_flag(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "difficulty_seeing": [0, 1, None, 2],
                "age": [30, 45, 70, 12],
                "sex": [1, 2, 1, 2],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["is_blind"].tolist() == [False, True, False, True]

    def test_augment_policyengine_person_inputs_uses_reported_ssi_for_takeup_only(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "ssi": [500.0, 0.0, 200.0],
                "ssi_reported": [0.0, 100.0, 0.0],
                "age": [70, 45, 34],
                "sex": [1, 2, 1],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["ssi"].tolist() == [500.0, 0.0, 200.0]
        assert augmented["ssi_reported"].tolist() == [0.0, 100.0, 0.0]
        assert augmented["takes_up_ssi_if_eligible"].tolist() == [
            False,
            True,
            False,
        ]

    def test_augment_policyengine_person_inputs_normalizes_explicit_ssi_takeup(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "takes_up_ssi_if_eligible": [1, 0, None, 2],
                "ssi_reported": [0.0, 100.0, 100.0, 0.0],
                "age": [70, 45, 34, 60],
                "sex": [1, 2, 1, 2],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["takes_up_ssi_if_eligible"].tolist() == [
            True,
            False,
            False,
            True,
        ]

    def test_calibrate_policyengine_ssi_takeup_uses_reported_amounts_by_age(
        self,
        monkeypatch,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                policyengine_dataset_year=2024,
                policyengine_calibration_target_profile="pe_native_broad",
            )
        )
        persons = pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "household_id": [10, 20, 30, 40],
                "age": [70, 70, 40, 40],
                "weight": [1.0, 1.0, 1.0, 1.0],
                "ssi": [100.0, 0.0, 100.0, 0.0],
                "takes_up_ssi_if_eligible": [True, False, True, False],
            }
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [10, 20, 30, 40],
                    "household_weight": [1.0, 1.0, 1.0, 1.0],
                }
            ),
            persons=persons,
            tax_units=pd.DataFrame({"tax_unit_id": [1], "household_id": [10]}),
            spm_units=pd.DataFrame({"spm_unit_id": [1], "household_id": [10]}),
            families=pd.DataFrame({"family_id": [1], "household_id": [10]}),
        )

        def fake_materialize(tables_arg, **kwargs):
            assert kwargs["variables"] == ("ssi",)
            assert tables_arg.persons["takes_up_ssi_if_eligible"].all()
            materialized_persons = tables_arg.persons.copy()
            materialized_persons["ssi"] = [80.0, 20.0, 20.0, 80.0]
            return PolicyEngineUSVariableMaterializationResult(
                tables=PolicyEngineUSEntityTableBundle(
                    households=tables_arg.households,
                    persons=materialized_persons,
                    tax_units=tables_arg.tax_units,
                    spm_units=tables_arg.spm_units,
                    families=tables_arg.families,
                    marital_units=tables_arg.marital_units,
                ),
                bindings={
                    "ssi": PolicyEngineUSVariableBinding(
                        entity=EntityType.PERSON,
                        column="ssi",
                    )
                },
                materialized_variables=("ssi",),
            )

        monkeypatch.setattr(
            us_pipeline_module,
            "materialize_policyengine_us_variables_safely",
            fake_materialize,
        )

        updated_tables, summary = (
            pipeline._calibrate_policyengine_ssi_takeup_from_reported_amounts(
                tables,
                target_period=2024,
            )
        )

        assert updated_tables.persons["takes_up_ssi_if_eligible"].tolist() == [
            True,
            True,
            True,
            True,
        ]
        assert summary["enabled"] is True
        assert summary["reported_amount"] == 200.0
        assert summary["selected_amount"] == 200.0

    def test_augment_policyengine_person_inputs_materializes_agi_parity_inputs(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "estate_income": [22.0],
                "farm_operations_income": [120.0],
                "farm_rent_income": [35.0],
                "health_savings_account_ald": [20.0],
                "self_employed_health_insurance_ald": [15.0],
                "self_employed_pension_contribution_ald": [10.0],
                "age": [45],
                "sex": [1],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["estate_income"].tolist() == [22.0]
        assert augmented["farm_operations_income"].tolist() == [120.0]
        assert augmented["farm_rent_income"].tolist() == [35.0]
        assert augmented["health_savings_account_ald"].tolist() == [20.0]
        assert augmented["self_employed_health_insurance_ald"].tolist() == [15.0]
        assert augmented["self_employed_pension_contribution_ald"].tolist() == [10.0]

    def test_augment_policyengine_person_inputs_materializes_export_support_aliases(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "age": [45, 50],
                "sex": [1, 2],
                "w2_wages_from_qualified_business": [1_000.0, 0.0],
                "unadjusted_basis_qualified_property": [10_000.0, 0.0],
                "business_is_sstb": [1, 0],
                "sstb_self_employment_income": [300.0, 0.0],
                "sstb_w2_wages_from_qualified_business": [200.0, 0.0],
                "sstb_unadjusted_basis_qualified_property": [2_000.0, 0.0],
                "self_employment_income_would_be_qualified": [1, 0],
                "sstb_self_employment_income_would_be_qualified": [1, 0],
                "qualified_reit_and_ptp_income": [75.0, 0.0],
                "qualified_bdc_income": [25.0, 0.0],
                "deductible_mortgage_interest": [900.0, 0.0],
                "investment_income_elected_form_4952": [40.0, 0.0],
                "health_insurance_premiums_without_medicare_part_b": [120.0, 0.0],
                "hours_worked": [37.5, 0.0],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["w2_wages_from_qualified_business"].tolist() == [1_000.0, 0.0]
        assert augmented["unadjusted_basis_qualified_property"].tolist() == [
            10_000.0,
            0.0,
        ]
        assert augmented["business_is_sstb"].tolist() == [True, False]
        assert augmented["sstb_self_employment_income_before_lsr"].tolist() == [
            300.0,
            0.0,
        ]
        assert augmented["sstb_w2_wages_from_qualified_business"].tolist() == [
            200.0,
            0.0,
        ]
        assert augmented["sstb_unadjusted_basis_qualified_property"].tolist() == [
            2_000.0,
            0.0,
        ]
        assert augmented["self_employment_income_would_be_qualified"].tolist() == [
            True,
            False,
        ]
        assert augmented["sstb_self_employment_income_would_be_qualified"].tolist() == [
            True,
            False,
        ]
        assert augmented["qualified_reit_and_ptp_income"].tolist() == [75.0, 0.0]
        assert augmented["qualified_bdc_income"].tolist() == [25.0, 0.0]
        assert augmented["home_mortgage_interest"].tolist() == [900.0, 0.0]
        assert augmented["investment_interest_expense"].tolist() == [40.0, 0.0]
        assert augmented["other_health_insurance_premiums"].tolist() == [120.0, 0.0]
        assert augmented["weekly_hours_worked_before_lsr"].tolist() == [37.5, 0.0]

    def test_augment_policyengine_person_inputs_coalesces_sparse_source_aliases_by_row(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "age": [45, 50, 55],
                "sex": [1, 2, 1],
                "income": [60_000.0, 75_000.0, 0.0],
                "employment_income_before_lsr": [0.0, 70_000.0, 0.0],
                "wage_income": [50_000.0, 80_000.0, 0.0],
                "self_employment_income_before_lsr": [0.0, 200.0, -300.0],
                "self_employment_income": [500.0, 999.0, 50.0],
                "taxable_interest_income": [0.0, 20.0, 0.0],
                "interest_income": [100.0, 999.0, 0.0],
                "ordinary_dividend_income": [0.0, 30.0, 0.0],
                "dividend_income": [80.0, 999.0, 0.0],
                "qualified_dividend_income": [0.0, 5.0, 0.0],
                "non_qualified_dividend_income": [0.0, 25.0, 0.0],
                "tax_exempt_pension_income": [40.0, 0.0, 0.0],
                "long_term_capital_gains_before_response": [0.0, 60.0, -10.0],
                "long_term_capital_gains": [40.0, 999.0, 0.0],
                "capital_gains": [999.0, 999.0, 25.0],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["employment_income_before_lsr"].tolist() == [
            50_000.0,
            70_000.0,
            0.0,
        ]
        assert augmented["self_employment_income_before_lsr"].tolist() == [
            500.0,
            200.0,
            -300.0,
        ]
        assert augmented["taxable_interest_income"].tolist() == [100.0, 20.0, 0.0]
        assert augmented["ordinary_dividend_income"].tolist() == [80.0, 30.0, 0.0]
        assert augmented["dividend_income"].tolist() == [80.0, 30.0, 0.0]
        assert augmented["long_term_capital_gains_before_response"].tolist() == [
            40.0,
            60.0,
            -10.0,
        ]
        assert augmented["tax_exempt_private_pension_income"].tolist() == [
            40.0,
            0.0,
            0.0,
        ]

    def test_augment_policyengine_person_inputs_preserves_tax_exempt_interest_split(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "age": [45, 50, 55],
                "sex": [1, 2, 1],
                "interest_income": [100.0, 50.0, 75.0],
                "taxable_interest_income": [0.0, 20.0, 0.0],
                "tax_exempt_interest_income": [100.0, 30.0, 0.0],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["taxable_interest_income"].tolist() == [0.0, 20.0, 75.0]
        assert augmented["tax_exempt_interest_income"].tolist() == [
            100.0,
            30.0,
            0.0,
        ]
        assert (
            augmented["taxable_interest_income"]
            + augmented["tax_exempt_interest_income"]
        ).tolist() == [100.0, 50.0, 75.0]

    def test_attach_policyengine_tax_unit_source_inputs_derives_mortgage_structure(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(policyengine_dataset_year=2024)
        )
        tax_units = pd.DataFrame(
            {
                "tax_unit_id": [1, 2],
                "deductible_mortgage_interest": [600.0, 0.0],
                "interest_deduction": [700.0, 0.0],
                "scf_mortgage_debt": [8_000.0, 0.0],
            }
        )

        augmented = pipeline._attach_policyengine_tax_unit_source_inputs(tax_units)

        assert augmented["first_home_mortgage_interest"].tolist() == [600.0, 0.0]
        assert augmented["interest_deduction"].tolist() == [700.0, 0.0]
        assert augmented["first_home_mortgage_balance"].tolist() == [10_000.0, 0.0]
        assert augmented["first_home_mortgage_origination_year"].tolist() == [2014, 0]

    def test_build_policyengine_households_preserves_vehicle_exports(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "household_id": [10, 10, 20],
                "weight": [1.0, 1.0, 2.0],
                "household_vehicles_owned": [2.0, 2.0, 1.0],
                "household_vehicles_value": [12_000.0, 12_000.0, 6_000.0],
            }
        )

        households = pipeline._build_policyengine_households(persons)

        assert households["household_vehicles_owned"].tolist() == [2.0, 1.0]
        assert households["household_vehicles_value"].tolist() == [12_000.0, 6_000.0]

    def test_augment_policyengine_person_inputs_derives_marital_status_flags_from_cps_codes(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "age": [45, 52, 38],
                "sex": [1, 2, 1],
                "marital_status": [6, 4, 7],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["is_separated"].tolist() == [True, False, False]
        assert augmented["is_surviving_spouse"].tolist() == [False, True, False]

    def test_augment_policyengine_person_inputs_derives_marital_status_flags_from_filing_status_code(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        persons = pd.DataFrame(
            {
                "age": [45, 52, 38],
                "sex": [1, 2, 1],
                "filing_status_code": [3, 5, 1],
            }
        )

        augmented = pipeline._augment_policyengine_person_inputs(persons)

        assert augmented["is_separated"].tolist() == [True, False, False]
        assert augmented["is_surviving_spouse"].tolist() == [False, True, False]

    def test_calibrate_policyengine_tables_from_db(self, persons, households, tmp_path):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        household_weights = calibrated_tables.households.set_index("household_id")[
            "household_weight"
        ]
        california_weight = calibrated_tables.households.loc[
            calibrated_tables.households["state_fips"] == 6,
            "household_weight",
        ].sum()

        assert summary["backend"] == "policyengine_db_entropy"
        assert summary["n_constraints"] == 2
        assert summary["max_error"] < 1e-6
        assert summary["weight_collapse_suspected"] is False
        assert summary["household_weight_diagnostics"]["total_weight"] == pytest.approx(
            450.0,
            rel=1e-6,
        )
        assert (
            summary["household_weight_diagnostics"]["positive_count"]
            == summary["household_weight_diagnostics"]["row_count"]
        )
        assert household_weights.sum() == pytest.approx(450.0, rel=1e-6)
        assert california_weight == pytest.approx(225.0, rel=1e-6)
        assert calibrated_persons.loc[
            calibrated_persons["state_fips"] == 6, "weight"
        ].iloc[0] == pytest.approx(225.0, rel=1e-6)

    def test_calibrate_policyengine_tables_residualizes_and_appends_forbes_spine(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        forbes_path = tmp_path / "forbes.jsonl"
        forbes_path.write_text(
            json.dumps(
                {
                    "forbes_unit_id": "forbes-1",
                    "name": "Example Founder",
                    "rank": 1,
                    "state_fips": 6,
                    "net_worth": 10_000_000_000.0,
                    "weight": 1.0,
                }
            )
            + "\n"
        )
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
            forbes_fixed_spine_records_path=forbes_path,
            forbes_fixed_spine_snapshot_id="forbes-test-2024",
            forbes_fixed_spine_replicates_per_unit=2,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        contributions = {
            contribution["target_name"]: contribution
            for contribution in summary["fixed_spine"]["residualization"][
                "contributions"
            ]
        }
        for table in (
            calibrated_tables.households,
            calibrated_tables.persons,
            calibrated_tables.tax_units,
        ):
            assert table is not None
            assert not any(column.startswith("forbes_") for column in table.columns)

        assert summary["fixed_spine"]["enabled"] is True
        assert summary["fixed_spine"]["record_metadata_rows"] == 2
        assert (
            summary["fixed_spine"]["source_metadata"]["snapshot_id"]
            == "forbes-test-2024"
        )
        assert summary["fixed_spine"]["residualization"]["supported_target_count"] == 2
        assert contributions["policyengine_us_target_1"]["contribution"] == (
            pytest.approx(1.0)
        )
        assert contributions["policyengine_us_target_2"]["contribution"] == (
            pytest.approx(1.0)
        )
        assert len(calibrated_tables.households) == len(tables.households) + 2
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-6,
        )
        california_weight = calibrated_tables.households.loc[
            calibrated_tables.households["state_fips"].eq(6),
            "household_weight",
        ].sum()
        assert california_weight == pytest.approx(225.0, rel=1e-6)
        assert calibrated_persons["weight"].sum() == pytest.approx(899.0, rel=1e-6)

    def test_calibrate_policyengine_tables_none_backend_preserves_original_weights(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="none",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)
        original_weights = tables.households["household_weight"].astype(float).copy()

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        assert summary["backend"] == "policyengine_db_none"
        assert summary["converged"] is True
        assert summary["max_error"] == 0.0
        assert summary["mean_error"] == 0.0
        assert summary["weight_collapse_suspected"] is False
        calibrated_weights = calibrated_tables.households["household_weight"].astype(
            float
        )
        assert calibrated_weights.tolist() == pytest.approx(original_weights.tolist())

    def test_calibrate_policyengine_tables_from_db_with_sparse_backend(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="sparse",
            target_sparsity=0.0,
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["backend"] == "policyengine_db_sparse"
        assert summary["n_constraints"] == 2
        assert summary["max_error"] < 1e-5
        assert summary["converged"] is True
        assert summary["sparsity"] == pytest.approx(0.0, abs=1e-9)
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-5,
        )

    def test_synthesize_seed_backend_preserves_seed_support(self, persons, households):
        config = USMicroplexBuildConfig(
            synthesis_backend="seed",
            n_synthetic=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households)

        synthetic, synthesizer, metadata = pipeline.synthesize(seed)

        assert synthesizer is None
        assert metadata["backend"] == "seed"
        assert metadata["n_seed_records"] == len(seed)
        assert len(synthetic) == len(seed)
        assert synthetic["household_id"].nunique() == seed["household_id"].nunique()
        assert synthetic["weight"].tolist() == pytest.approx(seed["hh_weight"].tolist())

    def test_calibrate_policyengine_tables_can_prune_to_household_budget(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)

        class StubSparseSelector:
            def __init__(self, **_kwargs):
                pass

            def fit_transform(
                self,
                frame,
                *_args,
                weight_col,
                linear_constraints=None,
                **_kwargs,
            ):
                result = frame.copy()
                result[weight_col] = np.array([10.0, 8.0, 0.0])
                self._constraints = tuple(linear_constraints or ())
                return result

            def validate(self, _frame):
                return {
                    "max_error": 0.1,
                    "mean_error": 0.05,
                    "converged": True,
                    "sparsity": 1 / 3,
                    "linear_errors": {
                        constraint.name: {
                            "actual": float(constraint.target),
                            "target": float(constraint.target),
                            "relative_error": 0.0,
                        }
                        for constraint in getattr(self, "_constraints", ())
                    },
                }

        monkeypatch.setattr(
            "microplex_us.pipelines.us.SparseCalibrator",
            StubSparseSelector,
        )
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
            policyengine_selection_household_budget=2,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        assert len(calibrated_tables.households) == 2
        assert set(calibrated_tables.households["household_id"]) == {1, 2}
        assert set(calibrated_persons["household_id"]) == {1, 2}
        assert summary["selection"]["applied"] is True
        assert summary["selection"]["selected_household_count"] == 2
        assert summary["selection"]["selector_positive_selected_count"] == 2
        assert "pre_selection" in summary["feasibility_filter"]

    def test_calibrate_policyengine_tables_from_db_can_use_pe_native_selection_backend(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)

        def _fake_optimize(**kwargs):
            assert kwargs["max_iter"] == 777
            assert kwargs["tol"] == pytest.approx(1e-7)
            assert kwargs["l2_penalty"] == pytest.approx(1e-5)
            output_path = kwargs["output_dataset_path"]
            with h5py.File(output_path, "w") as handle:
                household_id_group = handle.create_group("household_id")
                household_id_group.create_dataset("2024", data=np.asarray([1, 2, 3]))
                household_weight_group = handle.create_group("household_weight")
                household_weight_group.create_dataset(
                    "2024",
                    data=np.asarray([3.0, 2.0, 0.0], dtype=np.float32),
                )
            return SimpleNamespace(
                to_dict=lambda: {
                    "metric": "enhanced_cps_native_loss_weight_optimization",
                    "initial_loss": 0.9,
                    "optimized_loss": 0.7,
                    "converged": True,
                    "iterations": 12,
                    "positive_household_count": 2,
                    "target_names": ["nation/foo"],
                }
            )

        monkeypatch.setattr(
            "microplex_us.pipelines.us.optimize_policyengine_us_native_loss_dataset",
            _fake_optimize,
        )
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
            policyengine_selection_backend="pe_native_loss",
            policyengine_selection_household_budget=2,
            policyengine_selection_max_iter=777,
            policyengine_selection_tol=1e-7,
            policyengine_selection_l2_penalty=1e-5,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        assert len(calibrated_tables.households) == 2
        assert set(calibrated_tables.households["household_id"]) == {1, 2}
        assert set(calibrated_persons["household_id"]) == {1, 2}
        assert summary["selection"]["applied"] is True
        assert summary["selection"]["backend"] == "pe_native_loss"
        assert summary["selection"]["selected_household_count"] == 2
        assert summary["selection"]["selector_positive_selected_count"] == 2
        assert summary["selection"]["pe_native_optimization"]["optimized_loss"] == 0.7
        assert "target_names" not in summary["selection"]["pe_native_optimization"]

    def test_calibrate_policyengine_tables_pe_native_selection_can_preallocate_state_floor(
        self,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)

        def _unexpected_optimize(**_kwargs):
            raise AssertionError(
                "PE-native optimizer should not run when state floor fills budget"
            )

        monkeypatch.setattr(
            "microplex_us.pipelines.us.optimize_policyengine_us_native_loss_dataset",
            _unexpected_optimize,
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                calibration_backend="entropy",
                policyengine_targets_db=str(db_path),
                policyengine_target_variables=("household_count",),
                policyengine_target_period=2024,
                policyengine_calibration_min_active_households=1,
                policyengine_selection_backend="pe_native_loss",
                policyengine_selection_household_budget=2,
                policyengine_selection_state_floor=1,
            )
        )
        tables = PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame(
                {
                    "household_id": [1, 2, 3],
                    "household_weight": [10.0, 4.0, 8.0],
                    "state_fips": [6, 6, 36],
                }
            ),
            persons=pd.DataFrame(
                {
                    "person_id": [101, 102, 103],
                    "household_id": [1, 2, 3],
                    "weight": [10.0, 4.0, 8.0],
                    "state_fips": [6, 6, 36],
                    "age": [35, 28, 52],
                }
            ),
            tax_units=pd.DataFrame(
                {
                    "tax_unit_id": [11, 12, 13],
                    "household_id": [1, 2, 3],
                }
            ),
            spm_units=pd.DataFrame(
                {
                    "spm_unit_id": [21, 22, 23],
                    "household_id": [1, 2, 3],
                }
            ),
            families=pd.DataFrame(
                {
                    "family_id": [31, 32, 33],
                    "household_id": [1, 2, 3],
                }
            ),
            marital_units=pd.DataFrame(
                {
                    "marital_unit_id": [41, 42, 43],
                    "household_id": [1, 2, 3],
                }
            ),
        )

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        assert set(calibrated_tables.households["household_id"]) == {1, 3}
        assert set(calibrated_persons["household_id"]) == {1, 3}
        assert summary["selection"]["backend"] == "pe_native_loss"
        assert summary["selection"]["state_floor"]["applied"] is True
        assert summary["selection"]["state_floor"]["selected_household_count"] == 2
        assert summary["selection"]["state_floor"]["state_count"] == 2
        assert summary["selection"]["pe_native_optimization"]["budget"] == 0

    def test_selection_optimizer_kwargs_passes_target_total_weight(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                policyengine_selection_backend="pe_native_loss",
                policyengine_selection_household_budget=100,
                policyengine_selection_target_total_weight=150_000_000.0,
            )
        )
        kwargs = pipeline._policyengine_selection_optimizer_kwargs(requested_budget=100)
        assert kwargs["target_total_weight"] == 150_000_000.0

    def test_selection_optimizer_kwargs_omits_target_total_weight_when_none(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                policyengine_selection_backend="pe_native_loss",
                policyengine_selection_household_budget=100,
            )
        )
        kwargs = pipeline._policyengine_selection_optimizer_kwargs(requested_budget=100)
        assert "target_total_weight" not in kwargs

    def test_calibrate_policyengine_tables_from_db_with_hardconcrete_backend(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        seen_constraints = {}

        class StubHardConcreteCalibrator:
            def __init__(self, **_kwargs):
                self._constraints = ()

            def fit_transform(
                self,
                frame,
                *_args,
                weight_col,
                linear_constraints=None,
                **_kwargs,
            ):
                self._constraints = tuple(linear_constraints or ())
                seen_constraints["count"] = len(self._constraints)
                return frame.copy()

            def validate(self, _frame):
                return {
                    "max_error": 0.0,
                    "mean_error": 0.0,
                    "converged": True,
                    "sparsity": 0.25,
                    "linear_errors": {
                        constraint.name: {
                            "actual": float(constraint.target),
                            "target": float(constraint.target),
                            "relative_error": 0.0,
                        }
                        for constraint in self._constraints
                    },
                }

        monkeypatch.setattr(
            "microplex_us.pipelines.us.HardConcreteCalibrator",
            StubHardConcreteCalibrator,
        )
        config = USMicroplexBuildConfig(
            calibration_backend="hardconcrete",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert seen_constraints["count"] == 2
        assert summary["backend"] == "policyengine_db_hardconcrete"
        assert summary["n_constraints"] == 2
        assert summary["converged"] is True
        assert summary["sparsity"] == pytest.approx(0.25)
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-6,
        )

    def test_calibrate_policyengine_tables_from_db_with_pe_l0_backend(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        seen_constraints = {}

        class StubPolicyEngineL0Calibrator:
            def __init__(self, **_kwargs):
                self._constraints = ()

            def fit_transform(
                self,
                frame,
                *_args,
                weight_col,
                linear_constraints=None,
                **_kwargs,
            ):
                self._constraints = tuple(linear_constraints or ())
                seen_constraints["count"] = len(self._constraints)
                result = frame.copy()
                result[weight_col] = result[weight_col].astype(float)
                return result

            def validate(self, _frame):
                return {
                    "max_error": 0.0,
                    "mean_error": 0.0,
                    "converged": True,
                    "sparsity": 0.1,
                    "linear_errors": {
                        constraint.name: {
                            "actual": float(constraint.target),
                            "target": float(constraint.target),
                            "relative_error": 0.0,
                        }
                        for constraint in self._constraints
                    },
                }

        monkeypatch.setattr(
            "microplex_us.pipelines.us.PolicyEngineL0Calibrator",
            StubPolicyEngineL0Calibrator,
        )
        config = USMicroplexBuildConfig(
            calibration_backend="pe_l0",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert seen_constraints["count"] == 2
        assert summary["backend"] == "policyengine_db_pe_l0"
        assert summary["n_constraints"] == 2
        assert summary["converged"] is True
        assert summary["sparsity"] == pytest.approx(0.1)
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-6,
        )

    def test_calibrate_policyengine_tables_flags_weight_collapse(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)

        class CollapsingCalibrator:
            def __init__(self, method, **_kwargs):
                self.method = method

            def fit_transform(
                self,
                frame,
                *_args,
                weight_col,
                **_kwargs,
            ):
                collapsed = frame.copy()
                collapsed[weight_col] = 1e-10
                return collapsed

            def validate(self, _frame):
                return {
                    "max_error": 1.0,
                    "mean_error": 1.0,
                    "converged": False,
                    "linear_errors": {},
                }

        monkeypatch.setattr(
            "microplex_us.pipelines.us.Calibrator",
            CollapsingCalibrator,
        )
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        _, calibrated_persons, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["weight_collapse_suspected"] is True
        assert (
            summary["household_weight_diagnostics"]["tiny_count"]
            == summary["household_weight_diagnostics"]["row_count"]
        )
        assert summary["household_weight_diagnostics"]["total_weight"] == pytest.approx(
            summary["household_weight_diagnostics"]["row_count"] * 1e-10
        )
        assert summary["person_weight_diagnostics"]["tiny_count"] == len(
            calibrated_persons
        )

    def test_calibrate_policyengine_tables_can_rescale_back_to_input_weight_sum(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)

        class ShrinkingCalibrator:
            def __init__(self, method, **_kwargs):
                self.method = method

            def fit_transform(
                self,
                frame,
                *_args,
                weight_col,
                **_kwargs,
            ):
                shrunk = frame.copy()
                shrunk[weight_col] = shrunk[weight_col].astype(float) * 0.25
                return shrunk

            def validate(self, frame):
                values = frame["household_weight"].astype(float).to_numpy()
                return {
                    "max_error": 0.0,
                    "mean_error": 0.0,
                    "converged": True,
                    "linear_errors": {},
                    "sparsity": 0.0,
                    "validated_weight_sum": float(values.sum()),
                }

        monkeypatch.setattr(
            "microplex_us.pipelines.us.Calibrator",
            ShrinkingCalibrator,
        )
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
            policyengine_calibration_rescale_to_input_weight_sum=True,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        assert summary["input_household_weight_sum"] == pytest.approx(450.0, rel=1e-6)
        assert summary["pre_rescale_household_weight_sum"] == pytest.approx(
            112.5, rel=1e-6
        )
        assert summary["post_rescale_household_weight_sum"] == pytest.approx(
            450.0, rel=1e-6
        )
        assert summary["weight_sum_rescaled"] is True
        assert summary["weight_sum_rescale_mode"] == "input_weight_sum"
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-6,
        )
        assert calibrated_persons["weight"].sum() == pytest.approx(900.0, rel=1e-6)

    def test_calibrate_policyengine_tables_can_rescale_to_target_weight_sum(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)

        class ShrinkingCalibrator:
            def __init__(self, method, **_kwargs):
                self.method = method

            def fit_transform(
                self,
                frame,
                *_args,
                weight_col,
                **_kwargs,
            ):
                shrunk = frame.copy()
                shrunk[weight_col] = shrunk[weight_col].astype(float) * 0.25
                return shrunk

            def validate(self, frame):
                values = frame["household_weight"].astype(float).to_numpy()
                return {
                    "max_error": 0.0,
                    "mean_error": 0.0,
                    "converged": True,
                    "linear_errors": {},
                    "sparsity": 0.0,
                    "validated_weight_sum": float(values.sum()),
                }

        monkeypatch.setattr(
            "microplex_us.pipelines.us.Calibrator",
            ShrinkingCalibrator,
        )
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
            policyengine_calibration_target_total_weight=1_000.0,
            policyengine_calibration_rescale_to_target_total_weight=True,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        assert summary["input_household_weight_sum"] == pytest.approx(450.0, rel=1e-6)
        assert summary["pre_rescale_household_weight_sum"] == pytest.approx(
            112.5, rel=1e-6
        )
        assert summary["post_rescale_household_weight_sum"] == pytest.approx(
            1_000.0, rel=1e-6
        )
        assert summary["weight_sum_rescaled"] is True
        assert summary["weight_sum_rescale_mode"] == "target_total_weight"
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            1_000.0,
            rel=1e-6,
        )
        assert calibrated_persons["weight"].sum() == pytest.approx(2_000.0, rel=1e-6)

    def test_summarize_weight_diagnostics_flags_low_effective_sample_ratio(self):
        summary = _summarize_weight_diagnostics([100.0, 100.0] + [1e-10] * 10)

        assert summary["tiny_share"] < 0.95
        assert summary["effective_sample_ratio"] < 0.25
        assert summary["collapse_suspected"] is True

    def test_select_feasible_policyengine_calibration_constraints_caps_budget(self):
        targets = [
            TargetSpec(
                name="national_count",
                entity=EntityType.HOUSEHOLD,
                value=100.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                metadata={"geo_level": "national"},
            ),
            TargetSpec(
                name="state_count",
                entity=EntityType.HOUSEHOLD,
                value=50.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                metadata={"geo_level": "state"},
            ),
            TargetSpec(
                name="state_sum",
                entity=EntityType.HOUSEHOLD,
                value=25.0,
                period=2024,
                measure="snap",
                aggregation=TargetAggregation.SUM,
                metadata={"geo_level": "state"},
            ),
        ]
        constraints = (
            SimpleNamespace(coefficients=np.array([1.0, 1.0])),
            SimpleNamespace(coefficients=np.array([1.0, 0.0])),
            SimpleNamespace(coefficients=np.array([1.0, 1.0])),
        )

        selected_targets, selected_constraints, summary = (
            _select_feasible_policyengine_calibration_constraints(
                targets,
                constraints,
                household_count=2,
                max_constraints=None,
                max_constraints_per_household=1.0,
                min_active_households=1,
            )
        )

        assert [target.name for target in selected_targets] == [
            "national_count",
            "state_count",
        ]
        assert len(selected_constraints) == 2
        assert summary["feasibility_filter_applied"] is True
        assert summary["requested_max_constraints"] == 2
        assert summary["n_constraints_before_feasibility_filter"] == 3
        assert summary["n_constraints_after_feasibility_filter"] == 2
        assert summary["n_constraints_dropped_over_capacity"] == 1
        assert summary["constraint_drop_share"] == pytest.approx(1 / 3)
        assert summary["warning_messages"]

    def test_select_feasible_policyengine_calibration_constraints_drops_low_support_rows(
        self,
    ):
        targets = [
            TargetSpec(
                name="dense_state_count",
                entity=EntityType.HOUSEHOLD,
                value=50.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                metadata={"geo_level": "state"},
            ),
            TargetSpec(
                name="thin_state_count",
                entity=EntityType.HOUSEHOLD,
                value=25.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                metadata={"geo_level": "state"},
            ),
        ]
        constraints = (
            SimpleNamespace(coefficients=np.array([1.0, 1.0, 1.0, 1.0, 1.0])),
            SimpleNamespace(coefficients=np.array([0.0, 0.0, 0.0, 0.0, 1.0])),
        )

        selected_targets, _, summary = (
            _select_feasible_policyengine_calibration_constraints(
                targets,
                constraints,
                household_count=5,
                max_constraints=None,
                max_constraints_per_household=None,
                min_active_households=5,
            )
        )

        assert [target.name for target in selected_targets] == ["dense_state_count"]
        assert summary["n_constraints_dropped_low_support"] == 1
        assert summary["n_constraints_after_feasibility_filter"] == 1

    def test_normalize_microcalibrate_constraints_flips_negative_targets(self):
        constraints = (
            LinearConstraint(
                name="signed_loss",
                coefficients=np.array([-4.0, 1.0, 0.0]),
                target=-12.0,
            ),
            LinearConstraint(
                name="positive_amount",
                coefficients=np.array([2.0, 0.0, 3.0]),
                target=20.0,
            ),
        )

        normalized, summary = _normalize_policyengine_constraints_for_microcalibrate(
            constraints
        )

        assert normalized[0].name == "signed_loss"
        assert normalized[0].target == pytest.approx(12.0)
        np.testing.assert_allclose(normalized[0].coefficients, [4.0, -1.0, -0.0])
        assert normalized[1].name == "positive_amount"
        assert normalized[1].target == pytest.approx(20.0)
        np.testing.assert_allclose(normalized[1].coefficients, [2.0, 0.0, 3.0])
        assert summary == {
            "sign_flipped_constraint_count": 1,
            "sign_flipped_constraint_names": ["signed_loss"],
            "sign_flipped_constraint_names_truncated": False,
        }

    def test_calibrate_policyengine_tables_applies_feasibility_constraint_budget(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_max_constraints_per_household=0.5,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["n_constraints"] == 1
        assert summary["feasibility_filter"]["feasibility_filter_applied"] is True
        assert summary["feasibility_filter"]["requested_max_constraints"] == 1
        assert (
            summary["feasibility_filter"]["n_constraints_before_feasibility_filter"]
            == 2
        )
        assert (
            summary["feasibility_filter"]["n_constraints_after_feasibility_filter"] == 1
        )
        assert summary["target_plan"]["stage_counts"] == {
            "solve_now": 1,
            "solve_later": 1,
            "audit_only": 0,
        }
        assert summary["target_plan"]["reason_counts"]["constraint_capacity"] == 1
        assert summary["oracle_loss"]["full_oracle"]["target_count"] == 2
        assert summary["oracle_loss"]["full_oracle"]["supported_target_count"] == 2
        assert summary["oracle_loss"]["active_solve"]["target_count"] == 1
        assert summary["oracle_loss"]["active_solve"][
            "mean_abs_relative_error"
        ] == pytest.approx(
            0.0,
            abs=1e-12,
        )
        assert summary["oracle_loss"]["active_solve"][
            "capped_mean_abs_relative_error"
        ] == pytest.approx(0.0, abs=1e-12)
        assert summary["oracle_loss"]["deferred"]["target_count"] == 1
        assert (
            summary["oracle_loss"]["deferred"]["family_summaries"]["household_count"][
                "target_count"
            ]
            == 1
        )
        assert summary["oracle_loss"]["deferred"]["family_summaries"][
            "household_count"
        ]["loss_share"] == pytest.approx(1.0, rel=1e-9)
        assert summary["oracle_loss"]["deferred"]["family_summaries"][
            "household_count"
        ]["sum_abs_relative_error"] == pytest.approx(
            summary["oracle_loss"]["deferred"]["mean_abs_relative_error"],
            rel=1e-9,
        )
        assert summary["oracle_loss"]["deferred"]["family_ranking"][0]["group"] == (
            "household_count"
        )
        assert summary["oracle_loss"]["deferred"]["family_ranking"][0][
            "capped_sum_abs_relative_error"
        ] == pytest.approx(
            summary["oracle_loss"]["deferred"]["family_ranking"][0][
                "sum_abs_relative_error"
            ],
            rel=1e-9,
        )
        assert (
            summary["oracle_loss"]["full_oracle"]["geography_summaries"]["unspecified"][
                "target_count"
            ]
            == 2
        )
        assert summary["oracle_loss"]["full_oracle"]["geography_ranking"][0][
            "group"
        ] == ("unspecified")
        assert (
            summary["oracle_loss"]["full_oracle"]["mean_abs_relative_error"]
            > summary["oracle_loss"]["active_solve"]["mean_abs_relative_error"]
        )
        assert summary["full_oracle_mean_abs_relative_error"] == pytest.approx(
            summary["oracle_loss"]["full_oracle"]["mean_abs_relative_error"],
            rel=1e-9,
        )
        assert summary["full_oracle_capped_mean_abs_relative_error"] == pytest.approx(
            summary["oracle_loss"]["full_oracle"]["capped_mean_abs_relative_error"],
            rel=1e-9,
        )
        assert summary["active_solve_mean_abs_relative_error"] == pytest.approx(
            summary["oracle_loss"]["active_solve"]["mean_abs_relative_error"],
            rel=1e-9,
        )
        assert summary["active_solve_capped_mean_abs_relative_error"] == pytest.approx(
            summary["oracle_loss"]["active_solve"]["capped_mean_abs_relative_error"],
            rel=1e-9,
        )
        assert summary["oracle_relative_error_cap"] == pytest.approx(10.0)
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-6,
        )

    def test_calibrate_policyengine_tables_warns_when_many_constraints_are_dropped(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_max_constraints_per_household=0.5,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        with pytest.warns(
            UserWarning,
            match="Calibration feasibility filter dropped",
        ):
            _, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["warnings"]

    def test_calibrate_policyengine_tables_runs_deferred_low_support_stage(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=2,
            policyengine_calibration_deferred_stage_min_active_households=(1,),
            policyengine_calibration_deferred_stage_top_family_count=None,
            policyengine_calibration_deferred_stage_top_geography_count=None,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        _, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["n_constraints"] == 2
        assert summary["n_supported_targets"] == 2
        assert summary["n_calibration_stages_applied"] == 2
        assert summary["final_calibration_stage_index"] == 2
        assert summary["deferred_stage_support_schedule"] == [1]
        assert summary["target_plan"]["stage_counts"] == {
            "solve_now": 2,
            "solve_later": 0,
            "audit_only": 0,
        }
        assert summary["target_plan"]["reason_counts"]["selected_stage_1"] == 1
        assert summary["target_plan"]["reason_counts"]["selected_stage_2"] == 1
        assert summary["oracle_loss"]["deferred"]["target_count"] == 0
        assert summary["oracle_loss"]["active_solve"]["target_count"] == 2
        assert len(summary["calibration_stages"]) == 2
        assert summary["calibration_stages"][1]["kind"] == "deferred"
        assert summary["calibration_stages"][1]["status"] == "applied"
        assert summary["calibration_stages"][1]["min_active_households"] == 1
        assert summary["calibration_stages"][1]["selected_target_count"] == 1
        assert any(
            entry["stage"] == "solve_now" and entry["reason"] == "selected_stage_2"
            for entry in summary["target_ledger"]
        )

    def test_calibrate_policyengine_tables_skips_deferred_stage_below_trigger_threshold(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=3,
            policyengine_calibration_deferred_stage_min_active_households=(2, 1),
            policyengine_calibration_deferred_stage_top_family_count=None,
            policyengine_calibration_deferred_stage_top_geography_count=None,
            policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error=100.0,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        _, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["n_constraints"] == 1
        assert summary["n_calibration_stages_applied"] == 1
        assert summary["final_calibration_stage_index"] == 1
        assert summary["deferred_stage_support_schedule"] == [2, 1]
        assert summary["target_plan"]["stage_counts"] == {
            "solve_now": 1,
            "solve_later": 1,
            "audit_only": 0,
        }
        assert len(summary["calibration_stages"]) == 3
        assert summary["calibration_stages"][1]["status"] == "skipped"
        assert summary["calibration_stages"][1]["skip_reason"] == (
            "trigger_metric_below_threshold"
        )
        assert summary["calibration_stages"][1]["trigger_threshold"] == pytest.approx(
            100.0
        )
        assert summary["calibration_stages"][2]["status"] == "skipped"
        assert summary["calibration_stages"][2]["skip_reason"] == (
            "trigger_metric_below_threshold"
        )
        assert summary["calibration_stages"][2]["trigger_threshold"] == pytest.approx(
            100.0
        )

    def test_calibrate_policyengine_tables_marks_materialization_failures_audit_only(
        self,
        persons,
        households,
        tmp_path,
    ):
        class FakeEntity:
            def __init__(self, key: str):
                self.key = key

        class FakeVariable:
            def __init__(
                self,
                entity: FakeEntity,
                formulas: dict[str, object] | None = None,
            ):
                self.entity = entity
                self.formulas = formulas or {}

            def is_input_variable(self) -> bool:
                return not self.formulas

        class FakeTaxBenefitSystem:
            variables = {
                "state_fips": FakeVariable(FakeEntity("household")),
                "income_tax": FakeVariable(
                    FakeEntity("person"),
                    formulas={"2024": object()},
                ),
            }

        class FakeSimulation:
            tax_benefit_system = FakeTaxBenefitSystem()

            def __init__(self, dataset, dataset_year=None, **kwargs):
                _ = dataset, dataset_year, kwargs

            def calculate(self, variable, period=None, map_to=None):
                assert period == 2024
                assert map_to is None
                if variable == "income_tax":
                    raise RuntimeError("missing test parameter")
                raise KeyError(variable)

        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db_with_unsupported_target(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
            policyengine_simulation_cls=FakeSimulation,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["n_loaded_targets"] == 2
        assert summary["n_supported_targets"] == 1
        assert summary["n_unsupported_targets"] == 0
        assert summary["n_constraints"] == 1
        assert summary["target_plan"]["stage_counts"] == {
            "solve_now": 1,
            "solve_later": 0,
            "audit_only": 1,
        }
        assert summary["target_plan"]["reason_counts"]["materialization_failure"] == 1
        assert summary["oracle_loss"]["full_oracle"]["target_count"] == 2
        assert summary["oracle_loss"]["full_oracle"]["supported_target_count"] == 1
        assert summary["oracle_loss"]["full_oracle"]["unsupported_target_count"] == 1
        assert summary["oracle_loss"]["audit_only"]["target_count"] == 1
        assert summary["oracle_loss"]["audit_only"]["supported_target_count"] == 0
        assert summary["oracle_loss"]["audit_only"]["unsupported_target_count"] == 1
        assert summary["oracle_loss"]["full_oracle"][
            "unsupported_target_error_penalty"
        ] == pytest.approx(10.0)
        assert summary["oracle_loss"]["full_oracle"]["mean_abs_relative_error"] == (
            pytest.approx(5.0)
        )
        assert summary["oracle_loss"]["full_oracle"][
            "capped_mean_abs_relative_error"
        ] == pytest.approx(5.0)
        assert summary["oracle_loss"]["audit_only"]["mean_abs_relative_error"] == (
            pytest.approx(10.0)
        )
        assert summary["oracle_loss"]["audit_only"][
            "capped_mean_abs_relative_error"
        ] == pytest.approx(10.0)
        assert (
            summary["oracle_loss"]["audit_only"]["family_summaries"]["income_tax"][
                "unsupported_target_count"
            ]
            == 1
        )
        assert summary["oracle_loss"]["audit_only"]["family_summaries"]["income_tax"][
            "sum_abs_relative_error"
        ] == pytest.approx(10.0)
        assert summary["oracle_loss"]["audit_only"]["family_summaries"]["income_tax"][
            "capped_sum_abs_relative_error"
        ] == pytest.approx(10.0)
        assert summary["oracle_loss"]["audit_only"]["family_ranking"][0]["group"] == (
            "income_tax"
        )
        assert summary["oracle_loss"]["full_oracle"]["family_summaries"]["income_tax"][
            "supported_target_rate"
        ] == pytest.approx(0.0, abs=1e-12)
        assert (
            summary["oracle_loss"]["full_oracle"]["geography_summaries"]["unspecified"][
                "unsupported_target_count"
            ]
            == 1
        )
        assert summary["oracle_loss"]["full_oracle"]["geography_ranking"][0][
            "group"
        ] == ("unspecified")
        assert any(
            entry["stage"] == "audit_only"
            and entry["reason"] == "materialization_failure"
            for entry in summary["target_ledger"]
        )
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-6,
        )

    def test_policyengine_target_provider_returns_canonical_specs(
        self, persons, households, tmp_path
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)
        provider = pipeline.config.policyengine_targets_db
        assert provider is not None
        bindings = pipeline._infer_policyengine_variable_bindings(tables)

        from microplex_us.policyengine.us import PolicyEngineUSDBTargetProvider

        targets = PolicyEngineUSDBTargetProvider(provider).load_target_set(
            TargetQuery(
                period=2024,
                provider_filters={
                    "variables": ["household_count"],
                    "reform_id": 0,
                    "entity_overrides": {
                        variable: binding.entity
                        for variable, binding in bindings.items()
                    },
                },
            )
        )

        assert targets.targets
        assert all(isinstance(target, TargetSpec) for target in targets.targets)

    def test_calibrate_policyengine_tables_from_db_with_simulated_variable(
        self, persons, households, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "policyengine_targets.db"
        conn = sqlite3.connect(db_path)
        national_constraints: tuple[PolicyEngineUSConstraint, ...] = ()
        snap_positive_constraints = (PolicyEngineUSConstraint("snap", ">", "0"),)
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
                (
                    1,
                    compute_policyengine_us_definition_hash(national_constraints),
                    None,
                ),
                (
                    2,
                    compute_policyengine_us_definition_hash(
                        snap_positive_constraints,
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
            (2, "snap", ">", "0"),
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
                (1, "snap", 2024, 1, 0, 200.0, 1, None, "test", "national"),
                (2, "household_count", 2024, 2, 0, 2.0, 1, None, "test", "positive"),
            ],
        )
        conn.commit()
        conn.close()

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
                assert period == 2024
                assert self.dataset_year == 2024
                assert map_to is None
                if variable == "snap":
                    return [100.0, 0.0, 0.0]
                raise KeyError(variable)

        captured_direct_overrides: list[tuple[str, ...]] = []
        original_materialize = (
            us_pipeline_module.materialize_policyengine_us_variables_safely
        )

        def spy_materialize(*args, **kwargs):
            captured_direct_overrides.append(
                tuple(kwargs.get("direct_override_variables", ()))
            )
            return original_materialize(*args, **kwargs)

        monkeypatch.setattr(
            us_pipeline_module,
            "materialize_policyengine_us_variables_safely",
            spy_materialize,
        )

        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("snap", "household_count"),
            policyengine_target_period=2024,
            policyengine_dataset_year=2024,
            policyengine_simulation_cls=FakeSimulation,
            policyengine_direct_override_variables=("pre_tax_contributions",),
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight", "income": "employment_income"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)
        tables.households["snap"] = 999.0

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        assert summary["backend"] == "policyengine_db_entropy"
        assert captured_direct_overrides == [("pre_tax_contributions",)]
        assert summary["n_constraints"] == 2
        assert summary["materialized_variables"] == ["snap"]
        assert summary["max_error"] < 1e-6
        positive_weight = calibrated_tables.households.loc[
            calibrated_tables.households["snap"] > 0,
            "household_weight",
        ].sum()
        assert (
            calibrated_tables.households["snap"]
            * calibrated_tables.households["household_weight"]
        ).sum() == pytest.approx(
            200.0,
            rel=1e-6,
        )
        assert positive_weight == pytest.approx(2.0, rel=1e-6)
        positive_household_id = int(
            calibrated_tables.households.loc[
                calibrated_tables.households["snap"] > 0,
                "household_id",
            ].iloc[0]
        )
        assert calibrated_persons.loc[
            calibrated_persons["household_id"] == positive_household_id, "weight"
        ].iloc[0] == pytest.approx(2.0, rel=1e-6)

    def test_calibrate_policyengine_tables_skips_failed_materialized_variables(
        self, persons, households, tmp_path
    ):
        db_path = tmp_path / "policyengine_targets.db"
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
            """
        )
        conn.execute(
            """
            INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
            VALUES (?, ?, ?)
            """,
            (1, compute_policyengine_us_definition_hash(()), None),
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
                (1, "snap", 2024, 1, 0, 200.0, 1, None, "test", "national"),
                (
                    2,
                    "adjusted_gross_income",
                    2024,
                    1,
                    0,
                    1_000.0,
                    1,
                    None,
                    "test",
                    "agi",
                ),
            ],
        )
        conn.commit()
        conn.close()

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
                "adjusted_gross_income": FakeVariable(
                    FakeEntity("tax_unit"),
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
                assert period == 2024
                assert self.dataset_year == 2024
                assert map_to is None
                if variable == "snap":
                    return [100.0, 0.0, 0.0]
                if variable == "adjusted_gross_income":
                    raise RuntimeError("invalid state metadata")
                raise KeyError(variable)

        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("snap", "adjusted_gross_income"),
            policyengine_target_period=2024,
            policyengine_dataset_year=2024,
            policyengine_simulation_cls=FakeSimulation,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight", "income": "employment_income"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["n_loaded_targets"] == 2
        assert summary["n_supported_targets"] == 1
        assert summary["n_constraints"] == 1
        assert summary["materialized_variables"] == ["snap"]
        assert summary["materialization_failures"] == {
            "adjusted_gross_income": "RuntimeError: invalid state metadata"
        }

    def test_calibrate_policyengine_tables_uses_calibration_target_filters(
        self, persons, households, tmp_path
    ):
        db_path = tmp_path / "policyengine_targets.db"
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
            """
        )
        conn.execute(
            """
            INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
            VALUES (?, ?, ?)
            """,
            (1, compute_policyengine_us_definition_hash(()), None),
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
                (1, "snap", 2024, 1, 0, 200.0, 1, None, "test", "national"),
                (
                    2,
                    "adjusted_gross_income",
                    2024,
                    1,
                    0,
                    1_000.0,
                    1,
                    None,
                    "test",
                    "agi",
                ),
            ],
        )
        conn.commit()
        conn.close()

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
                "adjusted_gross_income": FakeVariable(
                    FakeEntity("tax_unit"),
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
                assert period == 2024
                assert self.dataset_year == 2024
                assert map_to is None
                if variable == "snap":
                    return [100.0, 0.0, 0.0]
                if variable == "adjusted_gross_income":
                    raise RuntimeError("invalid state metadata")
                raise KeyError(variable)

        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("snap", "adjusted_gross_income"),
            policyengine_calibration_target_variables=("snap",),
            policyengine_target_period=2024,
            policyengine_dataset_year=2024,
            policyengine_simulation_cls=FakeSimulation,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight", "income": "employment_income"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["n_loaded_targets"] == 1
        assert summary["n_supported_targets"] == 1
        assert summary["n_constraints"] == 1
        assert summary["target_variables"] == ["snap"]
        assert summary["materialized_variables"] == ["snap"]
        assert summary["materialization_failures"] == {}
        assert (
            calibrated_tables.households["snap"]
            * calibrated_tables.households["household_weight"]
        ).sum() == pytest.approx(200.0, rel=1e-6)
        assert (
            calibrated_tables.households["snap"]
            * calibrated_tables.households["household_weight"]
        ).sum() == pytest.approx(200.0, rel=1e-6)

    def test_build_policyengine_target_query_includes_named_target_profile(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                policyengine_target_profile="pe_native_broad",
            )
        )

        query = pipeline._build_policyengine_target_query({}, period=2024)

        assert query.provider_filters["target_profile"] == "pe_native_broad"
        assert query.provider_filters["target_cells"]
        assert {
            cell["geo_level"] for cell in query.provider_filters["target_cells"]
        } <= {"national", "state"}

    def test_build_policyengine_target_query_prefers_calibration_profile_override(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                policyengine_target_profile="pe_native_broad",
                policyengine_calibration_target_profile="pe_native_broad",
                policyengine_calibration_target_variables=("snap",),
            )
        )

        query = pipeline._build_policyengine_target_query(
            {},
            period=2024,
            for_calibration=True,
        )

        assert query.provider_filters["target_profile"] == "pe_native_broad"
        assert query.provider_filters["variables"] == ["snap"]
        assert query.provider_filters["target_cells"]

    def test_load_inputs_from_directory(self, persons, households, tmp_path):
        households.rename(columns={"hh_weight": "household_weight"}).to_parquet(
            tmp_path / "cps_asec_households.parquet",
            index=False,
        )
        persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

        config = USMicroplexBuildConfig(
            n_synthetic=8,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
        )
        pipeline = USMicroplexPipeline(config)
        result = pipeline.build_from_data_dir(tmp_path)

        assert result.synthetic_data["household_id"].nunique() == 8
        assert len(result.synthetic_data) > 8
        assert result.seed_data["hh_weight"].sum() == pytest.approx(900.0)

    def test_build_weight_calibrator_respects_iteration_and_tolerance_config(self):
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            calibration_tol=1e-4,
            calibration_max_iter=777,
        )
        pipeline = USMicroplexPipeline(config)

        calibrator = pipeline._build_weight_calibrator()

        assert calibrator.tol == pytest.approx(1e-4)
        assert calibrator.max_iter == 777

    def test_build_from_data_dir_can_prefer_cached_cps_asec_source(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        households.rename(columns={"hh_weight": "household_weight"}).to_parquet(
            tmp_path / "cps_asec_households.parquet",
            index=False,
        )
        persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "cps_asec_2023_processed.parquet").write_text("stub")

        class FakeCachedProvider:
            def __init__(self, *, year, cache_dir, download):
                self.year = year
                self.cache_dir = cache_dir
                self.download = download
                self.descriptor = SourceDescriptor(
                    name="cps_asec",
                    shareability=Shareability.PUBLIC,
                    time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                    observations=(
                        EntityObservation(
                            entity=EntityType.HOUSEHOLD,
                            key_column="household_id",
                            variable_names=("state_fips",),
                        ),
                    ),
                )

        class FakeParquetProvider:
            def __init__(self, *, data_dir):
                self.data_dir = data_dir
                self.descriptor = SourceDescriptor(
                    name="cps_asec_parquet",
                    shareability=Shareability.PUBLIC,
                    time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                    observations=(
                        EntityObservation(
                            entity=EntityType.HOUSEHOLD,
                            key_column="household_id",
                            variable_names=("state_fips",),
                        ),
                    ),
                )

        monkeypatch.setattr(
            "microplex_us.data_sources.cps.CPSASECSourceProvider",
            FakeCachedProvider,
        )
        monkeypatch.setattr(
            "microplex_us.data_sources.cps.CPSASECParquetSourceProvider",
            FakeParquetProvider,
        )

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                prefer_cached_cps_asec_source=True,
                cps_asec_cache_dir=str(cache_dir),
                cps_asec_source_year=2023,
            )
        )
        chosen: dict[str, object] = {}

        def fake_build_from_source_provider(provider):
            chosen["provider"] = provider
            return "cached"

        monkeypatch.setattr(
            pipeline, "build_from_source_provider", fake_build_from_source_provider
        )

        result = pipeline.build_from_data_dir(tmp_path)

        assert result == "cached"
        assert chosen["provider"].descriptor.name == "cps_asec"

    def test_build_from_source_provider(self, persons, households):
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
                name="test_cps",
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
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=8,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_source_provider(provider)

        assert result.synthetic_data["household_id"].nunique() == 8
        assert len(result.synthetic_data) > 8
        assert result.source_frame is not None
        assert result.source_frame.source.name == "test_cps"
        assert result.fusion_plan is not None
        assert result.fusion_plan.source_names == ("test_cps",)
        assert result.seed_data["hh_weight"].sum() == pytest.approx(900.0)
        assert {"person_id", "household_id"}.issubset(result.seed_data.columns)

    def test_build_from_source_provider_requires_household_person_relationship(
        self, persons, households
    ):
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="test_cps",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "hh_weight", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())

        with pytest.raises(
            ValueError,
            match="one-to-many household-to-person relationship",
        ):
            pipeline.build_from_source_provider(provider)

    def test_build_from_frames_prefers_scaffold_with_valid_geography(self):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [50.0, 75.0, 80.0],
                "state_fips": [0, 0, 0],
                "tenure": [1, 2, 1],
                "extra_household_var": [1.0, 2.0, 3.0],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [51, 34, 28],
                "sex": [1, 2, 1],
                "education": [4, 3, 2],
                "employment_status": [1, 1, 0],
                "income": [80_000.0, 40_000.0, 20_000.0],
                "extra_person_var": [9.0, 8.0, 7.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure", "extra_household_var"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "extra_person_var",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_frames([cps_frame, donor_frame])

        assert result.source_frame is not None
        assert result.source_frame.source.name == "cps_like"
        assert result.seed_data["state_fips"].tolist() == [6, 36]

    def test_select_scaffold_prefers_cps_when_puf_support_clone_enabled(self):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
            }
        )
        acs_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [90.0, 110.0, 130.0],
                "state_fips": [6, 36, 48],
                "tenure": [1, 2, 1],
                "rent": [1_000.0, 1_500.0, 900.0],
                "real_estate_taxes": [0.0, 2_000.0, 3_000.0],
            }
        )
        acs_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 74_000.0],
                "extra_person_var": [9.0, 8.0, 7.0],
            }
        )

        def frame(
            name: str,
            households: pd.DataFrame,
            persons: pd.DataFrame,
            household_variables: tuple[str, ...],
            person_variables: tuple[str, ...],
        ) -> ObservationFrame:
            return ObservationFrame(
                source=SourceDescriptor(
                    name=name,
                    shareability=Shareability.PUBLIC,
                    time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                    observations=(
                        EntityObservation(
                            entity=EntityType.HOUSEHOLD,
                            key_column="household_id",
                            variable_names=household_variables,
                            weight_column="hh_weight",
                        ),
                        EntityObservation(
                            entity=EntityType.PERSON,
                            key_column="person_id",
                            variable_names=person_variables,
                        ),
                    ),
                ),
                tables={
                    EntityType.HOUSEHOLD: households,
                    EntityType.PERSON: persons,
                },
                relationships=(
                    EntityRelationship(
                        parent_entity=EntityType.HOUSEHOLD,
                        child_entity=EntityType.PERSON,
                        parent_key="household_id",
                        child_key="household_id",
                        cardinality=RelationshipCardinality.ONE_TO_MANY,
                    ),
                ),
            )

        cps_frame = frame(
            "cps_asec_2025",
            cps_households,
            cps_persons,
            ("state_fips", "tenure"),
            (
                "household_id",
                "age",
                "sex",
                "education",
                "employment_status",
                "income",
            ),
        )
        acs_frame = frame(
            "acs_2022",
            acs_households,
            acs_persons,
            ("state_fips", "tenure", "rent", "real_estate_taxes"),
            (
                "household_id",
                "age",
                "sex",
                "education",
                "employment_status",
                "income",
                "extra_person_var",
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                puf_support_clone_enabled=True,
                synthesis_backend="seed",
                calibration_backend="entropy",
            )
        )
        source_inputs = [
            pipeline.prepare_source_input(cps_frame),
            pipeline.prepare_source_input(acs_frame),
        ]

        selected = pipeline._select_scaffold_source(source_inputs)

        assert selected.frame.source.name == "cps_asec_2025"

    def test_build_from_frames_prefers_scaffold_with_state_program_proxies(self):
        proxy_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        proxy_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
                "has_medicaid": [1, 0],
                "public_assistance": [0.0, 250.0],
                "ssi": [0.0, 0.0],
                "social_security": [0.0, 0.0],
            }
        )
        wider_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [90.0, 110.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
                "extra_household_var": [1.0, 2.0],
            }
        )
        wider_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [44, 21],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [58_000.0, 13_000.0],
                "extra_person_var": [9.0, 8.0],
                "another_extra_var": [5.0, 6.0],
            }
        )

        proxy_frame = ObservationFrame(
            source=SourceDescriptor(
                name="proxy_rich_cps",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "has_medicaid",
                            "public_assistance",
                            "ssi",
                            "social_security",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: proxy_households,
                EntityType.PERSON: proxy_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        wider_frame = ObservationFrame(
            source=SourceDescriptor(
                name="wider_but_proxy_poor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure", "extra_household_var"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "extra_person_var",
                            "another_extra_var",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: wider_households,
                EntityType.PERSON: wider_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_frames([proxy_frame, wider_frame])

        assert result.source_frame is not None
        assert result.source_frame.source.name == "proxy_rich_cps"
        assert result.synthesis_metadata["state_program_support_proxies"][
            "available"
        ] == [
            "has_medicaid",
            "public_assistance",
            "social_security",
            "ssi",
        ]
        assert result.synthesis_metadata["condition_vars"] == [
            "age",
            "sex",
            "education",
            "employment_status",
            "state_fips",
            "tenure",
            "has_medicaid",
        ]
        assert "has_medicaid" not in result.synthesis_metadata["target_vars"]
        assert "public_assistance" in result.synthesis_metadata["target_vars"]
        assert "ssi" in result.synthesis_metadata["target_vars"]
        assert "social_security" in result.synthesis_metadata["target_vars"]

    def test_build_from_source_provider_promotes_state_program_proxies_to_conditions(
        self,
    ):
        households = pd.DataFrame(
            {
                "household_key": [1, 2, 3],
                "household_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        persons = pd.DataFrame(
            {
                "person_key": [10, 11, 12],
                "household_key": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
                "has_medicaid": [1, 0, 1],
                "public_assistance": [0.0, 250.0, 0.0],
                "ssi": [0.0, 0.0, 900.0],
                "social_security": [0.0, 0.0, 1200.0],
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="proxy_rich_single_source",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_key",
                        variable_names=("state_fips", "tenure"),
                        weight_column="household_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=(
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "has_medicaid",
                            "public_assistance",
                            "ssi",
                            "social_security",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_key",
                    child_key="household_key",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_source_provider(StaticSourceProvider(frame))

        assert result.synthesis_metadata["condition_vars"] == [
            "age",
            "sex",
            "education",
            "employment_status",
            "state_fips",
            "tenure",
            "has_medicaid",
        ]
        assert result.synthesis_metadata["target_vars"] == [
            "income",
            "public_assistance",
            "ssi",
            "social_security",
        ]

    def test_build_from_frames_skips_non_numeric_donor_imputation_targets(self):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 41_000.0],
                "taxable_interest_income": [100.0, 50.0, 25.0],
                "all_zero_income": [0.0, 0.0, 0.0],
                "filing_status": ["SINGLE", "JOINT", "SINGLE"],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                            "all_zero_income",
                            "filing_status",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert "taxable_interest_income" in integration["seed_data"].columns
        assert "all_zero_income" not in integration["seed_data"].columns
        assert "filing_status" not in integration["seed_data"].columns
        assert integration["integrated_variables"] == ["taxable_interest_income"]

    def test_integrate_donor_sources_restricts_puf_to_authoritative_variables(self):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [0, 0, 0],
                "tenure": [0, 0, 0],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 41_000.0],
                "employment_income": [55_000.0, 12_500.0, 39_000.0],
                "taxable_interest_income": [0.0, 25.0, 100.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "employment_income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "tenure": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "employment_status": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "employment_income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert "taxable_interest_income" in integration["integrated_variables"]
        assert "employment_income" not in integration["integrated_variables"]
        assert "taxable_interest_income" in integration["seed_data"].columns
        assert "employment_income" not in integration["seed_data"].columns

    def test_integrate_donor_sources_respects_excluded_variables(self, monkeypatch):
        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = condition_vars, kwargs
                self.target_vars = tuple(target_vars)

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [10.0] * len(result)
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [0, 0, 0],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 41_000.0],
                "taxable_interest_income": [0.0, 25.0, 100.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "tenure": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "employment_status": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_excluded_variables=("taxable_interest_income",),
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert integration["integrated_variables"] == []
        assert "taxable_interest_income" not in integration["seed_data"].columns

    def test_default_build_config_excludes_filing_status_code_from_donor_imputation(
        self,
    ):
        config = USMicroplexBuildConfig()

        assert "filing_status_code" in config.donor_imputer_excluded_variables

    def test_build_config_can_opt_back_into_filing_status_code_donor_imputation(self):
        config = USMicroplexBuildConfig(donor_imputer_excluded_variables=())

        assert "filing_status_code" not in config.donor_imputer_excluded_variables

    def test_integrate_donor_sources_drops_constant_donor_conditions(self, monkeypatch):
        captured: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = kwargs
                self.target_vars = tuple(target_vars)
                self.condition_vars = tuple(condition_vars)
                captured.append(self.condition_vars)

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [10.0] * len(result)
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [0, 0, 0],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 41_000.0],
                "taxable_interest_income": [0.0, 25.0, 100.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "tenure": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=True,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "employment_status": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert captured
        assert "state_fips" not in captured[0]
        assert "tenure" in captured[0]

    def test_integrate_donor_sources_selects_top_correlated_condition_vars(
        self,
        monkeypatch,
    ):
        captured: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured.append(tuple(condition_vars))

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [10.0, 20.0, 30.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [25, 45, 65],
                "sex": [1, 2, 1],
                "education": [2, 2, 2],
                "employment_status": [1, 1, 1],
                "income": [30_000.0, 40_000.0, 50_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [24, 44, 64],
                "sex": [1, 1, 1],
                "education": [2, 2, 2],
                "employment_status": [1, 1, 1],
                "income": [10_000.0, 80_000.0, 20_000.0],
                "taxable_interest_income": [5.0, 15.0, 25.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=True,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_max_condition_vars=1,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert captured == [("age",)]

    def test_augment_donor_condition_frame_for_targets_derives_pe_style_puf_predictors(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        frame = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2", "1:3"],
                "household_id": [1, 1, 1],
                "tax_unit_id": ["1001", "1001", "1001"],
                "person_number": [1, 2, 3],
                "spouse_person_number": [2, 1, 0],
                "family_relationship": [1, 2, 3],
                "age": [45, 43, 12],
                "sex": [1, 2, 2],
            }
        )

        result = pipeline._augment_donor_condition_frame_for_targets(
            frame,
            ("taxable_interest_income",),
        )

        assert result["is_male"].tolist() == [1.0, 0.0, 0.0]
        assert result["tax_unit_is_joint"].tolist() == [1.0, 1.0, 1.0]
        assert result["tax_unit_count_dependents"].tolist() == [1.0, 1.0, 1.0]
        assert result["is_tax_unit_head"].tolist() == [1.0, 0.0, 0.0]
        assert result["is_tax_unit_spouse"].tolist() == [0.0, 1.0, 0.0]
        assert result["is_tax_unit_dependent"].tolist() == [0.0, 0.0, 1.0]

    def test_resolve_preferred_donor_condition_vars_uses_available_block_predictors(
        self,
    ):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        donor_frame = pd.DataFrame(
            {
                "age": [30, 45, 70],
                "is_male": [1.0, 0.0, 1.0],
                "income": [20_000.0, 80_000.0, 250_000.0],
                "tax_unit_is_joint": [0.0, 1.0, 1.0],
            }
        )
        current_frame = pd.DataFrame(
            {
                "age": [28, 50, 72],
                "is_male": [0.0, 1.0, 1.0],
                "income": [25_000.0, 90_000.0, 300_000.0],
                "tax_unit_is_joint": [0.0, 0.0, 1.0],
            }
        )

        assert pipeline._resolve_preferred_donor_condition_vars(
            donor_frame=donor_frame,
            current_frame=current_frame,
            donor_block=("dividend_income", "qualified_dividend_share"),
        ) == ["age", "is_male", "tax_unit_is_joint"]

    def test_resolve_challenger_shared_condition_vars_uses_source_native_puf_overlap(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                donor_imputer_condition_selection="pe_plus_puf_native_challenger"
            )
        )
        donor_frame = pd.DataFrame(
            {
                "age": [30, 45, 70],
                "self_employment_income": [0.0, 15_000.0, 0.0],
                "rental_income": [2_000.0, 0.0, 5_000.0],
                "social_security_retirement": [0.0, 0.0, 20_000.0],
                "alimony_income": [0.0, 3_000.0, 0.0],
            }
        )
        current_frame = pd.DataFrame(
            {
                "age": [28, 50, 72],
                "self_employment_income": [0.0, 12_000.0, 0.0],
                "rental_income": [1_500.0, 0.0, 4_000.0],
                "social_security_retirement": [0.0, 0.0, 18_000.0],
                "alimony_income": [0.0, 2_500.0, 0.0],
            }
        )

        assert pipeline._resolve_challenger_shared_condition_vars(
            donor_frame=donor_frame,
            current_frame=current_frame,
            shared_vars=[
                "age",
                "self_employment_income",
                "rental_income",
                "social_security_retirement",
                "alimony_income",
            ],
            donor_block=("taxable_interest_income",),
            donor_source_name="irs_soi_puf_2024",
        ) == [
            "self_employment_income",
            "rental_income",
            "social_security_retirement",
        ]

    def test_select_donor_condition_vars_keeps_all_shared_distinct_from_pe_presets(
        self,
    ):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                donor_imputer_condition_selection="all_shared",
                donor_imputer_max_condition_vars=1,
            )
        )
        donor_frame = pd.DataFrame(
            {
                "age": [30, 45, 70],
                "is_male": [1.0, 0.0, 1.0],
                "tax_unit_is_joint": [0.0, 1.0, 1.0],
                "education": [1.0, 2.0, 3.0],
            }
        )
        current_frame = donor_frame.copy()
        shared_vars = ["age", "is_male", "tax_unit_is_joint", "education"]

        assert (
            pipeline._select_donor_condition_vars(
                donor_frame,
                current_frame,
                shared_vars,
                ("taxable_interest_income",),
            )
            == shared_vars
        )

    def test_integrate_donor_sources_uses_pe_style_puf_predictors_for_generic_irs_vars(
        self,
        monkeypatch,
    ):
        captured: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured.append(tuple(condition_vars))

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [10.0, 20.0, 0.0, 25.0, 15.0, 0.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 90.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2", "1:3", "2:1", "3:1", "3:2"],
                "household_id": [1, 1, 1, 2, 3, 3],
                "age": [45, 43, 12, 61, 38, 10],
                "sex": [1, 2, 2, 1, 2, 1],
                "education": [2, 2, 1, 2, 2, 1],
                "employment_status": [1, 1, 0, 1, 1, 0],
                "income": [80_000.0, 50_000.0, 0.0, 70_000.0, 55_000.0, 0.0],
                "tax_unit_id": ["1001", "1001", "1001", "2001", "3001", "3001"],
                "person_number": [1, 2, 3, 1, 1, 2],
                "spouse_person_number": [2, 1, 0, 0, 0, 0],
                "family_relationship": [1, 2, 3, 1, 1, 3],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 110.0, 95.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": ["101:1", "101:2", "101:3", "102:1", "103:1", "103:2"],
                "household_id": [101, 101, 101, 102, 103, 103],
                "age": [46, 42, 11, 60, 39, 9],
                "sex": [1, 2, 2, 1, 2, 1],
                "education": [2, 2, 1, 2, 2, 1],
                "employment_status": [1, 1, 0, 1, 1, 0],
                "income": [70_000.0, 45_000.0, 0.0, 68_000.0, 52_000.0, 0.0],
                "tax_unit_id": ["2101", "2101", "2101", "2201", "2301", "2301"],
                "person_number": [1, 2, 3, 1, 1, 2],
                "spouse_person_number": [2, 1, 0, 0, 0, 0],
                "is_head": [1, 0, 0, 1, 1, 0],
                "is_spouse": [0, 1, 0, 0, 0, 0],
                "is_dependent": [0, 0, 1, 0, 0, 1],
                "taxable_interest_income": [5.0, 10.0, 0.0, 12.0, 8.0, 0.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "tax_unit_id",
                            "person_number",
                            "spouse_person_number",
                            "family_relationship",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "tax_unit_id",
                            "person_number",
                            "spouse_person_number",
                            "is_head",
                            "is_spouse",
                            "is_dependent",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=True,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_condition_selection="pe_prespecified",
                donor_imputer_max_condition_vars=1,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert captured == [
            (
                "age",
                "is_male",
                "tax_unit_is_joint",
                "tax_unit_count_dependents",
                "is_tax_unit_head",
                "is_tax_unit_spouse",
                "is_tax_unit_dependent",
            )
        ]
        assert integration["conditioning_diagnostics"] == [
            {
                "donor_source": "irs_soi_puf_2024",
                "model_variables": ["taxable_interest_income"],
                "restored_variables": ["taxable_interest_income"],
                "condition_selection": "pe_prespecified",
                "used_condition_surface": False,
                "raw_shared_vars": [
                    "age",
                    "education",
                    "employment_status",
                    "income",
                    "person_number",
                    "sex",
                    "spouse_person_number",
                    "state_fips",
                    "tenure",
                ],
                "shared_vars_after_model_exclusion": [
                    "age",
                    "education",
                    "employment_status",
                    "income",
                    "person_number",
                    "sex",
                    "spouse_person_number",
                    "state_fips",
                    "tenure",
                ],
                "projection_applied": False,
                "entity_compatible_shared_vars": [],
                "shared_vars_for_block": [
                    "age",
                    "education",
                    "employment_status",
                    "income",
                    "person_number",
                    "sex",
                    "spouse_person_number",
                    "state_fips",
                    "tenure",
                ],
                "selected_condition_vars": [
                    "age",
                    "is_male",
                    "tax_unit_is_joint",
                    "tax_unit_count_dependents",
                    "is_tax_unit_head",
                    "is_tax_unit_spouse",
                    "is_tax_unit_dependent",
                ],
                "requested_supplemental_shared_condition_vars": [],
                "requested_challenger_shared_condition_vars": [],
                "raw_supplemental_shared_condition_var_status": [],
                "raw_challenger_shared_condition_var_status": [],
                "supplemental_shared_condition_var_status": [],
                "challenger_shared_condition_var_status": [],
                "dropped_shared_vars": [
                    "education",
                    "employment_status",
                    "income",
                    "person_number",
                    "sex",
                    "spouse_person_number",
                    "state_fips",
                    "tenure",
                ],
            }
        ]

    def test_integrate_donor_sources_pe_plus_puf_native_challenger_widens_pe_surface(
        self,
        monkeypatch,
        caplog,
    ):
        captured: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured.append(tuple(condition_vars))

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [10.0, 20.0, 0.0, 25.0, 15.0, 0.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 90.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2", "1:3", "2:1", "3:1", "3:2"],
                "household_id": [1, 1, 1, 2, 3, 3],
                "age": [45, 43, 12, 61, 38, 10],
                "sex": [1, 2, 2, 1, 2, 1],
                "education": [2, 2, 1, 2, 2, 1],
                "employment_status": [1, 1, 0, 1, 1, 0],
                "income": [80_000.0, 50_000.0, 0.0, 70_000.0, 55_000.0, 0.0],
                "self_employment_income": [0.0, 2_000.0, 0.0, 0.0, 4_000.0, 0.0],
                "rental_income": [500.0, 0.0, 0.0, 0.0, 1_500.0, 0.0],
                "social_security_retirement": [0.0, 0.0, 0.0, 20_000.0, 0.0, 0.0],
                "tax_unit_id": ["1001", "1001", "1001", "2001", "3001", "3001"],
                "person_number": [1, 2, 3, 1, 1, 2],
                "spouse_person_number": [2, 1, 0, 0, 0, 0],
                "family_relationship": [1, 2, 3, 1, 1, 3],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 110.0, 95.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": ["101:1", "101:2", "101:3", "102:1", "103:1", "103:2"],
                "household_id": [101, 101, 101, 102, 103, 103],
                "age": [46, 42, 11, 60, 39, 9],
                "sex": [1, 2, 2, 1, 2, 1],
                "education": [2, 2, 1, 2, 2, 1],
                "employment_status": [1, 1, 0, 1, 1, 0],
                "income": [70_000.0, 45_000.0, 0.0, 68_000.0, 52_000.0, 0.0],
                "self_employment_income": [0.0, 1_000.0, 0.0, 0.0, 3_500.0, 0.0],
                "rental_income": [400.0, 0.0, 0.0, 0.0, 1_200.0, 0.0],
                "social_security_retirement": [0.0, 0.0, 0.0, 18_000.0, 0.0, 0.0],
                "tax_unit_id": ["2101", "2101", "2101", "2201", "2301", "2301"],
                "person_number": [1, 2, 3, 1, 1, 2],
                "spouse_person_number": [2, 1, 0, 0, 0, 0],
                "is_head": [1, 0, 0, 1, 1, 0],
                "is_spouse": [0, 1, 0, 0, 0, 0],
                "is_dependent": [0, 0, 1, 0, 0, 1],
                "taxable_interest_income": [5.0, 10.0, 0.0, 12.0, 8.0, 0.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_2024",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        weight_column="hh_weight",
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "self_employment_income",
                            "rental_income",
                            "social_security_retirement",
                            "tax_unit_id",
                            "person_number",
                            "spouse_person_number",
                            "family_relationship",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        weight_column="hh_weight",
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "self_employment_income",
                            "rental_income",
                            "social_security_retirement",
                            "tax_unit_id",
                            "person_number",
                            "spouse_person_number",
                            "is_head",
                            "is_spouse",
                            "is_dependent",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "employment_status": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_condition_selection="pe_plus_puf_native_challenger",
                donor_imputer_max_condition_vars=1,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)
        caplog.set_level(logging.INFO, logger="microplex_us.pipelines.us")

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert captured == [
            (
                "age",
                "is_male",
                "tax_unit_is_joint",
                "tax_unit_count_dependents",
                "is_tax_unit_head",
                "is_tax_unit_spouse",
                "is_tax_unit_dependent",
                "self_employment_income",
                "rental_income",
                "social_security_retirement",
            )
        ]
        diagnostics = integration["conditioning_diagnostics"][0]
        assert diagnostics["condition_selection"] == "pe_plus_puf_native_challenger"
        assert diagnostics["used_condition_surface"] is False
        assert diagnostics["requested_challenger_shared_condition_vars"] == [
            "self_employment_income",
            "rental_income",
            "social_security_retirement",
        ]
        assert diagnostics["selected_condition_vars"] == list(captured[0])
        assert diagnostics["raw_challenger_shared_condition_var_status"] == [
            {
                "variable": "self_employment_income",
                "selected": True,
                "in_shared_overlap": True,
                "reason": "selected",
            },
            {
                "variable": "rental_income",
                "selected": True,
                "in_shared_overlap": True,
                "reason": "selected",
            },
            {
                "variable": "social_security_retirement",
                "selected": True,
                "in_shared_overlap": True,
                "reason": "selected",
            },
        ]
        assert diagnostics["challenger_shared_condition_var_status"] == [
            {
                "variable": "self_employment_income",
                "selected": True,
                "in_shared_overlap": True,
                "reason": "selected",
            },
            {
                "variable": "rental_income",
                "selected": True,
                "in_shared_overlap": True,
                "reason": "selected",
            },
            {
                "variable": "social_security_retirement",
                "selected": True,
                "in_shared_overlap": True,
                "reason": "selected",
            },
        ]
        log_messages = [record.getMessage() for record in caplog.records]
        assert any(
            "US microplex donor integration: source ready" in message
            and "donor_source=irs_soi_puf_2024" in message
            and "blocks=1" in message
            for message in log_messages
        )
        assert any(
            "US microplex donor integration: block run" in message
            and "block=taxable_interest_income" in message
            and "condition_vars=10" in message
            for message in log_messages
        )
        assert any(
            "US microplex donor integration: block complete" in message
            and "integrated_vars=1" in message
            for message in log_messages
        )

    def test_integrate_donor_sources_uses_pe_prespecified_acs_predictors(
        self,
        monkeypatch,
    ):
        captured: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured.append(tuple(condition_vars))

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["rent"] = [1_200.0, 900.0, 600.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 11, 20],
                "household_id": [1, 1, 2],
                "age": [45, 14, 67],
                "sex": [1, 2, 2],
                "is_head": [1, 0, 1],
                "employment_income": [60_000.0, 0.0, 10_000.0],
                "self_employment_income": [5_000.0, 0.0, 0.0],
                "gross_social_security": [0.0, 0.0, 20_000.0],
                "taxable_pension_income": [0.0, 0.0, 15_000.0],
                "income": [65_000.0, 0.0, 45_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 101, 102],
                "age": [44, 12, 68],
                "sex": [1, 2, 2],
                "is_head": [1, 0, 1],
                "employment_income": [58_000.0, 0.0, 12_000.0],
                "self_employment_income": [4_000.0, 0.0, 0.0],
                "gross_social_security": [0.0, 0.0, 22_000.0],
                "taxable_pension_income": [0.0, 0.0, 16_000.0],
                "income": [62_000.0, 0.0, 50_000.0],
                "rent": [1_100.0, 0.0, 950.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "is_head",
                            "employment_income",
                            "self_employment_income",
                            "gross_social_security",
                            "taxable_pension_income",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="acs_2022",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "is_head",
                            "employment_income",
                            "self_employment_income",
                            "gross_social_security",
                            "taxable_pension_income",
                            "income",
                            "rent",
                        ),
                    ),
                ),
                variable_capabilities={
                    "rent": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_condition_selection="pe_prespecified",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert captured == [
            (
                "is_household_head",
                "age",
                "is_male",
                "tenure_type",
                "employment_income",
                "self_employment_income",
                "social_security",
                "pension_income",
                "household_size",
                "state_fips",
            )
        ]

    def test_integrate_donor_sources_pe_prespecified_falls_back_for_unmapped_sources(
        self,
        monkeypatch,
    ):
        captured: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured.append(tuple(condition_vars))

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [10.0, 20.0, 30.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [25, 45, 65],
                "sex": [1, 2, 1],
                "education": [2, 2, 2],
                "employment_status": [1, 1, 1],
                "income": [30_000.0, 40_000.0, 50_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [24, 44, 64],
                "sex": [1, 1, 1],
                "education": [2, 2, 2],
                "employment_status": [1, 1, 1],
                "income": [10_000.0, 80_000.0, 20_000.0],
                "taxable_interest_income": [5.0, 15.0, 25.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=True,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_condition_selection="pe_prespecified",
                donor_imputer_max_condition_vars=1,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert captured == [("age",)]

    def test_integrate_donor_sources_keeps_person_native_irs_blocks_on_person_rows_when_ids_present(
        self,
        monkeypatch,
    ):
        captured_conditions: list[tuple[str, ...]] = []
        captured_fit_rows: list[int] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured_conditions.append(tuple(condition_vars))

            def fit(self, frame, *args, **kwargs):
                _ = args, kwargs
                captured_fit_rows.append(len(frame))

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = np.zeros(len(result), dtype=float)
                result.loc[result.index[-1], "taxable_interest_income"] = 100.0
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2", "2:1"],
                "household_id": [1, 1, 2],
                "tax_unit_id": [100, 100, 200],
                "age": [45, 43, 19],
                "sex": [1, 2, 1],
                "education": [3, 3, 2],
                "employment_status": [1, 1, 1],
                "income": [60_000.0, 15_000.0, 12_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": ["101:1", "101:2", "102:1"],
                "household_id": [101, 101, 102],
                "tax_unit_id": [900, 900, 901],
                "age": [44, 42, 21],
                "sex": [1, 2, 1],
                "education": [3, 3, 2],
                "employment_status": [1, 1, 1],
                "income": [58_000.0, 14_000.0, 13_000.0],
                "taxable_interest_income": [0.0, 0.0, 100.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "tax_unit_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "tax_unit_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert len(captured_conditions) == 1
        assert {"age", "income", "state_fips", "tenure"}.issubset(
            set(captured_conditions[0])
        )
        assert captured_fit_rows == [3]
        assert integration["seed_data"]["taxable_interest_income"].tolist() == [
            0.0,
            0.0,
            100.0,
        ]

    def test_integrate_donor_sources_allows_person_conditions_for_labor_tax_unit_blocks(
        self,
        monkeypatch,
    ):
        captured_conditions: list[tuple[str, ...]] = []
        captured_fit_rows: list[int] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured_conditions.append(tuple(condition_vars))

            def fit(self, frame, *args, **kwargs):
                _ = args, kwargs
                captured_fit_rows.append(len(frame))

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["self_employment_income"] = np.linspace(
                    0.0,
                    90.0,
                    num=len(result),
                    dtype=float,
                )
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2", "2:1", "3:1"],
                "household_id": [1, 1, 2, 3],
                "tax_unit_id": [100, 100, 200, 300],
                "age": [25, 23, 45, 65],
                "sex": [1, 1, 1, 1],
                "education": [2, 2, 2, 2],
                "employment_status": [1, 1, 1, 1],
                "income": [20_000.0, 5_000.0, 50_000.0, 90_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": ["101:1", "101:2", "102:1", "103:1"],
                "household_id": [101, 101, 102, 103],
                "tax_unit_id": [900, 900, 901, 902],
                "age": [24, 22, 44, 64],
                "sex": [1, 1, 1, 1],
                "education": [2, 2, 2, 2],
                "employment_status": [1, 1, 1, 1],
                "income": [18_000.0, 4_000.0, 52_000.0, 92_000.0],
                "self_employment_income": [0.0, 0.0, 20.0, 100.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "tax_unit_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "tax_unit_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "self_employment_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "self_employment_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_max_condition_vars=1,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert captured_conditions == [("age",)]
        assert captured_fit_rows == [4]

    def test_project_frame_to_entity_uses_variable_projection_aggregation(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        frame = pd.DataFrame(
            {
                "tax_unit_id": [100, 100, 200],
                "age": [25, 45, 65],
                "income": [20_000.0, 5_000.0, 90_000.0],
                "tenure": [1, 1, 2],
            }
        )

        projected = pipeline._project_frame_to_entity(
            frame,
            entity=EntityType.TAX_UNIT,
            variables={"age", "income", "tenure"},
        )

        assert projected["tax_unit_id"].tolist() == [100, 200]
        assert projected["age"].tolist() == [45, 65]
        assert projected["income"].tolist() == [25_000.0, 90_000.0]
        assert projected["tenure"].tolist() == [1, 2]

    def test_integrate_donor_sources_projects_spm_unit_native_blocks_when_ids_missing(
        self,
        monkeypatch,
    ):
        captured_conditions: list[tuple[str, ...]] = []
        captured_fit_rows: list[int] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured_conditions.append(tuple(condition_vars))

            def fit(self, frame, *args, **kwargs):
                _ = args, kwargs
                captured_fit_rows.append(len(frame))

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["snap"] = [120.0, 0.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2", "2:1"],
                "household_id": [1, 1, 2],
                "relationship_to_head": [0, 2, 0],
                "age": [40, 10, 55],
                "sex": [1, 2, 1],
                "education": [3, 1, 4],
                "employment_status": [1, 0, 1],
                "income": [40_000.0, 0.0, 35_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": ["101:1", "101:2", "102:1"],
                "household_id": [101, 101, 102],
                "relationship_to_head": [0, 2, 0],
                "age": [42, 11, 57],
                "sex": [1, 2, 1],
                "education": [3, 1, 4],
                "employment_status": [1, 0, 1],
                "income": [38_000.0, 0.0, 34_000.0],
                "snap": [120.0, 120.0, 0.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "relationship_to_head",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="spm_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "relationship_to_head",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "snap",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)
        monkeypatch.setattr(
            pipeline,
            "build_policyengine_entity_tables",
            lambda _population: pytest.fail(
                "SPM-only donor projection should not build full entity tables"
            ),
        )

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert len(captured_conditions) == 1
        assert {"age", "income", "state_fips", "tenure"}.issubset(
            set(captured_conditions[0])
        )
        assert captured_fit_rows == [2]
        assert "spm_unit_id" in integration["seed_data"].columns
        assert integration["seed_data"]["snap"].tolist() == [120.0, 120.0, 0.0]

    def test_strip_generated_entity_ids_drops_helper_ids_missing_from_scaffold(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_households = pd.DataFrame(
            {
                "household_id": [1],
                "hh_weight": [100.0],
                "state_fips": [6],
                "tenure": [1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2"],
                "household_id": [1, 1],
                "age": [40, 10],
                "sex": [1, 2],
                "education": [3, 1],
                "employment_status": [1, 0],
                "income": [40_000.0, 0.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        frame = cps_persons.assign(
            tax_unit_id=[100, 100],
            family_id=[10, 10],
            spm_unit_id=[20, 20],
            marital_unit_id=[30, 31],
        )

        stripped = pipeline._strip_generated_entity_ids(
            frame,
            scaffold_input=cps_input,
        )

        assert "tax_unit_id" not in stripped.columns
        assert "family_id" not in stripped.columns
        assert "spm_unit_id" not in stripped.columns
        assert "marital_unit_id" not in stripped.columns
        assert stripped["person_id"].tolist() == ["1:1", "1:2"]

    def test_strip_generated_entity_ids_preserves_observed_scaffold_ids(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_households = pd.DataFrame(
            {
                "household_id": [1],
                "hh_weight": [100.0],
                "state_fips": [6],
                "tenure": [1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2"],
                "household_id": [1, 1],
                "tax_unit_id": [100, 100],
                "family_id": [10, 10],
                "age": [40, 10],
                "sex": [1, 2],
                "education": [3, 1],
                "employment_status": [1, 0],
                "income": [40_000.0, 0.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "tax_unit_id",
                            "family_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        frame = cps_persons.assign(
            spm_unit_id=[20, 20],
            marital_unit_id=[30, 31],
        )

        stripped = pipeline._strip_generated_entity_ids(
            frame,
            scaffold_input=cps_input,
        )

        assert stripped["tax_unit_id"].tolist() == [100, 100]
        assert stripped["family_id"].tolist() == [10, 10]
        assert "spm_unit_id" not in stripped.columns
        assert "marital_unit_id" not in stripped.columns

    def test_build_from_frames_drops_generated_entity_ids_before_stage5(
        self,
        monkeypatch,
    ):
        cps_households = pd.DataFrame(
            {
                "household_id": [1],
                "hh_weight": [100.0],
                "state_fips": [6],
                "tenure": [1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2"],
                "household_id": [1, 1],
                "relationship_to_head": [0, 2],
                "age": [40, 10],
                "sex": [1, 2],
                "education": [3, 1],
                "employment_status": [1, 0],
                "income": [40_000.0, 0.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "relationship_to_head",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="seed",
                calibration_backend="entropy",
            )
        )
        original_prepare_seed_data_from_source = pipeline.prepare_seed_data_from_source
        captured_integrate_seed_columns: list[str] = []
        captured_seed_columns: list[str] = []

        def fake_prepare_seed_data_from_source(source_input):
            seed_data = original_prepare_seed_data_from_source(source_input)
            return seed_data.assign(
                tax_unit_id=[100, 100],
                spm_unit_id=[200, 200],
                marital_unit_id=[300, 301],
            )

        def fake_integrate(seed_data, *, scaffold_input, donor_inputs):
            _ = scaffold_input, donor_inputs
            captured_integrate_seed_columns[:] = seed_data.columns.tolist()
            return {
                "seed_data": seed_data,
                "integrated_variables": [],
                "conditioning_diagnostics": [
                    {
                        "donor_source": "test_donor",
                        "model_variables": ["income"],
                        "selected_condition_vars": ["age"],
                    }
                ],
            }

        def fake_synthesize(seed_data, synthesis_variables=None):
            _ = synthesis_variables
            captured_seed_columns[:] = seed_data.columns.tolist()
            synthetic = seed_data.copy()
            synthetic["weight"] = synthetic["hh_weight"].astype(float)
            return synthetic, None, {}

        def fake_calibrate(synthetic_data, targets):
            _ = targets
            return synthetic_data, {}

        monkeypatch.setattr(
            pipeline,
            "prepare_seed_data_from_source",
            fake_prepare_seed_data_from_source,
        )
        monkeypatch.setattr(pipeline, "_integrate_donor_sources", fake_integrate)
        monkeypatch.setattr(pipeline, "synthesize", fake_synthesize)
        monkeypatch.setattr(pipeline, "calibrate", fake_calibrate)

        result = pipeline.build_from_frames([cps_frame])

        assert "tax_unit_id" not in captured_integrate_seed_columns
        assert "spm_unit_id" not in captured_integrate_seed_columns
        assert "marital_unit_id" not in captured_integrate_seed_columns
        assert result.scaffold_seed_data is not None
        assert "tax_unit_id" not in result.scaffold_seed_data.columns
        assert "spm_unit_id" not in result.scaffold_seed_data.columns
        assert "marital_unit_id" not in result.scaffold_seed_data.columns
        assert "tax_unit_id" not in captured_seed_columns
        assert "spm_unit_id" not in captured_seed_columns
        assert "marital_unit_id" not in captured_seed_columns
        assert "tax_unit_id" not in result.seed_data.columns
        assert "spm_unit_id" not in result.seed_data.columns
        assert "marital_unit_id" not in result.seed_data.columns
        assert result.synthesis_metadata["donor_conditioning_diagnostics"] == [
            {
                "donor_source": "test_donor",
                "model_variables": ["income"],
                "selected_condition_vars": ["age"],
            }
        ]

    def test_build_from_frames_rank_matches_generated_donor_values(
        self,
        monkeypatch,
    ):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 41_000.0],
                "taxable_interest_income": [0.0, 0.0, 100.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        class FakeSynthesizer:
            def __init__(self, *args, **kwargs):
                _ = args
                _ = kwargs

            def fit(self, *args, **kwargs):
                _ = args
                _ = kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [1e12, -1e12, 500.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert integration["seed_data"]["taxable_interest_income"].tolist() == [
            100.0,
            0.0,
            0.0,
        ]

    def test_rank_match_donor_values_preserves_zero_inflated_positive_support(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        scores = pd.Series([0.1, 0.2, 0.9, 1.0], dtype=float)
        donor_values = pd.Series([0.0, 0.0, 10.0, 20.0], dtype=float)
        donor_weights = pd.Series([1.0, 1.0, 1.0, 1.0], dtype=float)

        matched = pipeline._rank_match_donor_values(
            scores,
            donor_values=donor_values,
            donor_weights=donor_weights,
            rng=np.random.default_rng(42),
        )

        assert matched.tolist() == [0.0, 0.0, 10.0, 20.0]

    def test_rank_match_donor_values_respects_weighted_positive_rate(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=5,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        scores = pd.Series([0.1, 0.2, 0.3, 0.9, 1.0], dtype=float)
        donor_values = pd.Series([0.0, 0.0, 10.0], dtype=float)
        donor_weights = pd.Series([4.0, 4.0, 2.0], dtype=float)

        matched = pipeline._rank_match_donor_values(
            scores,
            donor_values=donor_values,
            donor_weights=donor_weights,
            rng=np.random.default_rng(42),
        )

        assert (matched > 0).sum() == 1
        assert matched.iloc[-1] > 0.0
        assert matched.iloc[:-1].eq(0.0).all()

    def test_build_from_source_provider_defaults_missing_optional_variables(self):
        households = pd.DataFrame(
            {
                "household_key": [1, 2],
                "household_weight": [125.0, 175.0],
                "region_code": [1, 2],
            }
        )
        persons = pd.DataFrame(
            {
                "person_key": [10, 11],
                "household_key": [1, 2],
                "age": [45, 19],
                "income": [60_000.0, 12_000.0],
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="sparse_provider",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_key",
                        variable_names=("region_code",),
                        weight_column="household_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=("age", "income"),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_key",
                    child_key="household_key",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_source_provider(provider)

        assert result.seed_data["tenure"].eq(0).all()
        assert result.seed_data["employment_status"].eq(0).all()
        assert set(result.seed_data["state"]) == {"UNK"}
        assert result.seed_data["hh_weight"].sum() == pytest.approx(300.0)

    def test_build_from_source_provider_prefers_household_scoped_merge_columns(self):
        households = pd.DataFrame(
            {
                "household_key": [1, 2],
                "household_weight": [125.0, 175.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        persons = pd.DataFrame(
            {
                "person_key": [10, 11],
                "household_key": [1, 2],
                "age": [45, 19],
                "income": [60_000.0, 12_000.0],
                "state_fips": [99, 99],
                "tenure": [9, 9],
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="overlapping_columns",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_key",
                        variable_names=("state_fips", "tenure"),
                        weight_column="household_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=("age", "income", "state_fips", "tenure"),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_key",
                    child_key="household_key",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_source_provider(provider)

        assert result.seed_data["state_fips"].tolist() == [6, 36]
        assert result.seed_data["tenure"].tolist() == [1, 2]

    def test_synthesizer_uses_observed_source_coverage(self):
        households = pd.DataFrame(
            {
                "household_key": [1, 2, 3],
                "household_weight": [100.0, 120.0, 140.0],
                "region_code": [1, 2, 3],
            }
        )
        persons = pd.DataFrame(
            {
                "person_key": [10, 11, 12],
                "household_key": [1, 2, 3],
                "age": [45, 19, 62],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="sparse_provider",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_key",
                        variable_names=("region_code",),
                        weight_column="household_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=("age", "income"),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_key",
                    child_key="household_key",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=3,
                synthesis_backend="synthesizer",
                calibration_backend="entropy",
                synthesizer_epochs=2,
                synthesizer_n_layers=2,
                synthesizer_hidden_dim=8,
                random_seed=5,
            )
        )

        result = pipeline.build_from_source_provider(provider)

        assert result.synthesis_metadata["condition_vars"] == ["age"]
        assert result.synthesis_metadata["target_vars"] == ["income"]
        assert result.synthesizer is not None

    def test_synthesizer_handles_state_program_proxy_condition_vars(self):
        households = pd.DataFrame(
            {
                "household_key": [1, 2, 3, 4],
                "household_weight": [100.0, 120.0, 140.0, 160.0],
                "state_fips": [6, 6, 36, 36],
                "tenure": [1, 2, 1, 2],
            }
        )
        persons = pd.DataFrame(
            {
                "person_key": [10, 11, 12, 13],
                "household_key": [1, 2, 3, 4],
                "age": [45, 19, 62, 35],
                "sex": [1, 2, 1, 2],
                "education": [4, 2, 3, 1],
                "employment_status": [1, 0, 1, 1],
                "income": [60_000.0, 12_000.0, 40_000.0, 22_000.0],
                "has_medicaid": [1.0, 0.0, 0.0, 1.0],
                "public_assistance": [0.0, 150.0, 0.0, 0.0],
                "ssi": [0.0, 0.0, 0.0, 0.0],
                "social_security": [0.0, 0.0, 900.0, 0.0],
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="state_program_proxy_provider",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_key",
                        variable_names=("state_fips", "tenure"),
                        weight_column="household_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=(
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "has_medicaid",
                            "public_assistance",
                            "ssi",
                            "social_security",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_key",
                    child_key="household_key",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="synthesizer",
                calibration_backend="entropy",
                synthesizer_epochs=2,
                synthesizer_n_layers=2,
                synthesizer_hidden_dim=8,
                random_seed=7,
            )
        )

        result = pipeline.build_from_source_provider(provider)

        assert result.synthesizer is not None
        assert result.synthesis_metadata["condition_vars"] == [
            "age",
            "sex",
            "education",
            "employment_status",
            "state_fips",
            "tenure",
            "has_medicaid",
        ]
        assert len(result.synthetic_data) == 4

    def test_constant_has_medicaid_is_not_auto_promoted_to_condition_var(self):
        frame = pd.DataFrame(
            {
                "age": [25, 40, 55, 32],
                "sex": [1, 2, 1, 2],
                "education": [2, 3, 4, 1],
                "employment_status": [1, 1, 0, 1],
                "state_fips": [6, 6, 36, 36],
                "tenure": [1, 2, 1, 2],
                "income": [50_000.0, 30_000.0, 20_000.0, 80_000.0],
                "has_medicaid": [0.0, 0.0, 0.0, 0.0],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        condition_vars = pipeline._resolve_synthesis_condition_vars(
            frame.columns,
            observed_frame=frame,
        )

        assert "has_medicaid" not in condition_vars

    def test_ensure_target_support_handles_bool_destination_columns(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        synthetic_data = pd.DataFrame(
            {
                "person_id": [0, 1],
                "household_id": [0, 1],
                "state_fips": [6, 36],
                "tenure": [1, 2],
                "age": [40, 50],
                "sex": [1, 2],
                "education": [3, 4],
                "employment_status": [1, 1],
                "income": [40_000.0, 60_000.0],
                "has_medicaid": pd.Series([False, False], dtype=bool),
                "weight": [1.0, 1.0],
            }
        )
        seed_data = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [10, 20],
                "state_fips": [6, 36],
                "tenure": [1, 2],
                "age": [41, 51],
                "sex": [1, 2],
                "education": [3, 4],
                "employment_status": [1, 1],
                "income": [42_000.0, 61_000.0],
                "has_medicaid": [1.0, 0.0],
                "weight": [1.0, 1.0],
            }
        )
        targets = USMicroplexTargets(
            marginal={"has_medicaid": ["1.0"]},
            continuous={},
        )

        result = pipeline.ensure_target_support(synthetic_data, seed_data, targets)

        assert pd.to_numeric(result["has_medicaid"], errors="coerce").max() == 1.0

    def test_build_from_missing_directory_raises(self, tmp_path):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())

        with pytest.raises(FileNotFoundError, match="CPS ASEC data files not found"):
            pipeline.build_from_data_dir(tmp_path)


class TestUSMicroplexBuildResult:
    """Test build result helpers."""

    @pytest.fixture
    def result(self):
        config = USMicroplexBuildConfig(
            n_synthetic=3,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
        )
        seed = pd.DataFrame({"income": [1.0], "hh_weight": [1.0]})
        synthetic = pd.DataFrame({"income": [1.0, 2.0, 3.0], "weight": [1.0, 1.0, 1.0]})
        calibrated = synthetic.copy()
        calibrated["weight"] = [0.0, 2.0, 3.0]

        return USMicroplexBuildResult(
            config=config,
            seed_data=seed,
            synthetic_data=synthetic,
            calibrated_data=calibrated,
            targets=USMicroplexTargets(marginal={}, continuous={"income": 6.0}),
            calibration_summary={"max_error": 0.0, "mean_error": 0.0},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=None,
        )

    def test_nonzero_weight_count(self, result):
        assert result.n_nonzero_weights == 2

    def test_total_weighted_population(self, result):
        assert result.total_weighted_population == 5.0
