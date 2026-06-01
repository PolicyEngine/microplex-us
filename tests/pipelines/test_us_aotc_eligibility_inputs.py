"""Tests for the AOTC eligibility-input construction in the US pipeline.

Exercises ``USMicroplexPipeline._construct_aotc_eligibility_inputs`` (and its
call site inside ``build_policyengine_entity_tables``), which mirrors the
enhanced-CPS baseline ``ExtendedCPS._impute_aotc_eligibility_inputs`` at
``PolicyEngine/policyengine-us-data``
``policyengine_us_data/datasets/cps/extended_cps.py:1204-1369``.
"""

import pandas as pd
import pytest

from microplex_us.pipelines.us import USMicroplexBuildConfig, USMicroplexPipeline
from microplex_us.policyengine.us import (
    POLICYENGINE_US_EXPORT_DEFAULTS,
    SAFE_POLICYENGINE_US_EXPORT_VARIABLES,
    build_policyengine_us_export_variable_maps,
    build_policyengine_us_time_period_arrays,
)

AOTC_TRUE_FLAG_COLUMNS = (
    "is_pursuing_credential_for_american_opportunity_credit",
    "attends_eligible_educational_institution_for_american_opportunity_credit",
    "is_enrolled_at_least_half_time_for_american_opportunity_credit",
    "has_american_opportunity_credit_1098_t_or_exception",
    "has_american_opportunity_credit_institution_ein",
)
AOTC_FALSE_FLAG_COLUMNS = (
    "has_completed_first_four_years_of_postsecondary_education",
    "has_felony_drug_conviction",
)
AOTC_PRIOR_YEARS_COLUMN = "american_opportunity_credit_claimed_prior_years"
ALL_AOTC_COLUMNS = (
    AOTC_TRUE_FLAG_COLUMNS + AOTC_FALSE_FLAG_COLUMNS + (AOTC_PRIOR_YEARS_COLUMN,)
)


def _pipeline(year: int = 2024) -> USMicroplexPipeline:
    return USMicroplexPipeline(USMicroplexBuildConfig(policyengine_dataset_year=year))


def test_all_eight_aotc_columns_are_safe_export_variables():
    for column in ALL_AOTC_COLUMNS:
        assert column in SAFE_POLICYENGINE_US_EXPORT_VARIABLES


def test_all_eight_aotc_columns_have_false_or_zero_defaults():
    for column in AOTC_TRUE_FLAG_COLUMNS + AOTC_FALSE_FLAG_COLUMNS:
        assert POLICYENGINE_US_EXPORT_DEFAULTS[column] is False
    assert POLICYENGINE_US_EXPORT_DEFAULTS[AOTC_PRIOR_YEARS_COLUMN] == 0


def test_fallback_marks_tuition_holders_when_no_credit_signal():
    """No credit column -> eCPS fallback aotc_student = tuition > 0.

    This path needs no PolicyEngine-US parameters (no back-solve runs).
    """
    pipeline = _pipeline()
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "household_id": [10, 10, 20],
            "tax_unit_id": [100, 100, 200],
            "age": [45, 19, 50],
            "income": [60_000.0, 0.0, 40_000.0],
            "qualified_tuition_expenses": [0.0, 3_500.0, 0.0],
            "relationship_to_head": [0, 2, 0],
        }
    )

    result = pipeline._construct_aotc_eligibility_inputs(persons)
    by_id = result.set_index("person_id")

    # Student (person 2, positive tuition) gets the five factual flags.
    for column in AOTC_TRUE_FLAG_COLUMNS:
        assert bool(by_id.loc[2, column]) is True
    for column in AOTC_FALSE_FLAG_COLUMNS:
        assert bool(by_id.loc[2, column]) is False
    assert int(by_id.loc[2, AOTC_PRIOR_YEARS_COLUMN]) == 0

    # Non-students (persons 1, 3) keep defaults.
    for person_id in (1, 3):
        for column in AOTC_TRUE_FLAG_COLUMNS + AOTC_FALSE_FLAG_COLUMNS:
            assert bool(by_id.loc[person_id, column]) is False
        assert int(by_id.loc[person_id, AOTC_PRIOR_YEARS_COLUMN]) == 0


def test_no_signal_at_all_leaves_frame_unchanged():
    """Neither a credit nor a tuition column -> nothing to construct."""
    pipeline = _pipeline()
    persons = pd.DataFrame(
        {
            "person_id": [1, 2],
            "household_id": [10, 10],
            "tax_unit_id": [100, 100],
            "age": [40, 38],
            "income": [50_000.0, 45_000.0],
            "relationship_to_head": [0, 1],
        }
    )

    result = pipeline._construct_aotc_eligibility_inputs(persons)

    # The construction returns early; no AOTC columns are added here. The
    # export layer supplies the contract-required columns from defaults.
    for column in ALL_AOTC_COLUMNS:
        assert column not in result.columns


def test_fallback_clamps_existing_prior_years_to_three():
    pipeline = _pipeline()
    persons = pd.DataFrame(
        {
            "person_id": [1],
            "household_id": [10],
            "tax_unit_id": [100],
            "age": [20],
            "income": [0.0],
            "qualified_tuition_expenses": [2_000.0],
            AOTC_PRIOR_YEARS_COLUMN: [7],
            "relationship_to_head": [0],
        }
    )

    result = pipeline._construct_aotc_eligibility_inputs(persons)
    assert int(result.set_index("person_id").loc[1, AOTC_PRIOR_YEARS_COLUMN]) == 3


def test_credit_signal_with_zero_positive_credit_marks_nobody():
    """Credit column present but no positive value -> eCPS early return."""
    pipeline = _pipeline()
    persons = pd.DataFrame(
        {
            "person_id": [1, 2],
            "household_id": [10, 10],
            "tax_unit_id": [100, 100],
            "age": [45, 19],
            "income": [60_000.0, 0.0],
            "qualified_tuition_expenses": [0.0, 3_000.0],
            "american_opportunity_credit": [0.0, 0.0],
            "is_full_time_college_student": [False, True],
            "relationship_to_head": [0, 2],
        }
    )

    result = pipeline._construct_aotc_eligibility_inputs(persons)
    # When a credit signal exists but is all-zero, the credit-driven path
    # returns before writing inputs (it does NOT fall back to tuition>0).
    for column in ALL_AOTC_COLUMNS:
        assert column not in result.columns


class TestCreditDrivenConstruction:
    """Credit-driven back-solve; needs PolicyEngine-US parameters."""

    @pytest.fixture(autouse=True)
    def _require_policyengine_us(self):
        pytest.importorskip("policyengine_us")

    def test_dependent_student_selected_and_tuition_backsolved(self):
        pipeline = _pipeline(2024)
        # Parent filer + full-time college dependent; $2,500 tax-unit credit
        # broadcast across members (PUF tax-unit column on the person frame).
        persons = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "tax_unit_id": [100, 100, 100],
                "age": [50, 19, 16],
                "income": [80_000.0, 0.0, 0.0],
                "is_tax_unit_dependent": [0.0, 1.0, 1.0],
                "is_full_time_college_student": [False, True, False],
                "qualified_tuition_expenses": [0.0, 4_000.0, 0.0],
                "american_opportunity_credit": [2_500.0, 2_500.0, 2_500.0],
                "relationship_to_head": [0, 2, 2],
            }
        )

        result = pipeline._construct_aotc_eligibility_inputs(persons)
        by_id = result.set_index("person_id")

        # The college dependent is the selected student.
        for column in AOTC_TRUE_FLAG_COLUMNS:
            assert bool(by_id.loc[2, column]) is True
        for column in AOTC_FALSE_FLAG_COLUMNS:
            assert bool(by_id.loc[2, column]) is False
        assert int(by_id.loc[2, AOTC_PRIOR_YEARS_COLUMN]) in range(0, 4)

        # $2,500 credit back-solves to $4,000 of qualified expenses.
        assert by_id.loc[2, "qualified_tuition_expenses"] == pytest.approx(4_000.0)

        # Parent and minor are not students.
        for person_id in (1, 3):
            for column in AOTC_TRUE_FLAG_COLUMNS:
                assert bool(by_id.loc[person_id, column]) is False

    def test_partial_credit_backsolves_to_smaller_expenses(self):
        pipeline = _pipeline(2024)
        # Single filer who is the student; $1,250 credit -> $1,250 expenses
        # (inside the 100% first-bracket), OVERWRITING the reported $2,000.
        persons = pd.DataFrame(
            {
                "person_id": [1],
                "household_id": [10],
                "tax_unit_id": [100],
                "age": [28],
                "income": [30_000.0],
                "is_tax_unit_dependent": [0.0],
                "is_full_time_college_student": [True],
                "qualified_tuition_expenses": [2_000.0],
                "american_opportunity_credit": [1_250.0],
                "relationship_to_head": [0],
            }
        )

        result = pipeline._construct_aotc_eligibility_inputs(persons)
        row = result.set_index("person_id").loc[1]
        for column in AOTC_TRUE_FLAG_COLUMNS:
            assert bool(row[column]) is True
        assert row["qualified_tuition_expenses"] == pytest.approx(1_250.0)

    def test_full_time_student_selected_when_no_member_has_tuition(self):
        pipeline = _pipeline(2024)
        # Credit present, nobody has positive tuition: selection falls to the
        # full-time college student (second priority group in eCPS).
        persons = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 100],
                "age": [50, 20],
                "income": [70_000.0, 0.0],
                "is_tax_unit_dependent": [0.0, 1.0],
                "is_full_time_college_student": [False, True],
                "qualified_tuition_expenses": [0.0, 0.0],
                "american_opportunity_credit": [2_500.0, 2_500.0],
                "relationship_to_head": [0, 2],
            }
        )

        result = pipeline._construct_aotc_eligibility_inputs(persons)
        by_id = result.set_index("person_id")
        assert (
            bool(by_id.loc[2, "is_pursuing_credential_for_american_opportunity_credit"])
            is True
        )
        assert (
            bool(by_id.loc[1, "is_pursuing_credential_for_american_opportunity_credit"])
            is False
        )
        # The student's tuition is set to the credit-implied $4,000.
        assert by_id.loc[2, "qualified_tuition_expenses"] == pytest.approx(4_000.0)

    def test_export_includes_all_eight_columns_with_real_values(self):
        pipeline = _pipeline(2024)
        tbs = pipeline._resolve_policyengine_tax_benefit_system()
        persons = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "tax_unit_id": [100, 100],
                "age": [50, 19],
                "sex": [1, 2],
                "income": [80_000.0, 0.0],
                "is_tax_unit_dependent": [0.0, 1.0],
                "is_full_time_college_student": [False, True],
                "qualified_tuition_expenses": [0.0, 4_000.0],
                "american_opportunity_credit": [2_500.0, 2_500.0],
                "relationship_to_head": [0, 2],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(persons)
        export_maps = build_policyengine_us_export_variable_maps(
            tables, tax_benefit_system=tbs
        )
        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            household_variable_map=export_maps["household"],
            person_variable_map=export_maps["person"],
            tax_unit_variable_map=export_maps["tax_unit"],
            spm_unit_variable_map=export_maps["spm_unit"],
            family_variable_map=export_maps["family"],
        )

        for column in ALL_AOTC_COLUMNS:
            assert column in arrays, column

        # The dependent student (second person row) has the True flags.
        for column in AOTC_TRUE_FLAG_COLUMNS:
            assert arrays[column]["2024"].tolist() == [False, True]
        for column in AOTC_FALSE_FLAG_COLUMNS:
            assert arrays[column]["2024"].tolist() == [False, False]
        assert arrays[AOTC_PRIOR_YEARS_COLUMN]["2024"].tolist() == [0, 0]

        # american_opportunity_credit is a PUF calculated output and must not
        # be exported (PolicyEngine-US recomputes it from these inputs).
        assert "american_opportunity_credit" not in arrays


def test_no_signal_export_falls_back_to_defaults():
    """With no AOTC signal, the contract-required columns still export."""
    pytest.importorskip("policyengine_us")
    pipeline = _pipeline(2024)
    tbs = pipeline._resolve_policyengine_tax_benefit_system()
    persons = pd.DataFrame(
        {
            "person_id": [1, 2],
            "household_id": [10, 10],
            "tax_unit_id": [100, 100],
            "age": [40, 38],
            "sex": [1, 2],
            "income": [50_000.0, 45_000.0],
            "is_tax_unit_dependent": [0.0, 0.0],
            "relationship_to_head": [0, 1],
        }
    )

    tables = pipeline.build_policyengine_entity_tables(persons)
    export_maps = build_policyengine_us_export_variable_maps(
        tables, tax_benefit_system=tbs
    )
    arrays = build_policyengine_us_time_period_arrays(
        tables,
        period=2024,
        household_variable_map=export_maps["household"],
        person_variable_map=export_maps["person"],
        tax_unit_variable_map=export_maps["tax_unit"],
        spm_unit_variable_map=export_maps["spm_unit"],
        family_variable_map=export_maps["family"],
    )

    for column in AOTC_TRUE_FLAG_COLUMNS + AOTC_FALSE_FLAG_COLUMNS:
        assert column in arrays
        assert arrays[column]["2024"].tolist() == [False, False]
    assert arrays[AOTC_PRIOR_YEARS_COLUMN]["2024"].tolist() == [0, 0]
