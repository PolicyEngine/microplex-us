"""Tests for atomic vs derived variable normalization."""

from __future__ import annotations

import pandas as pd
import pytest
from microplex.core import EntityType

from microplex_us.variables import (
    ConditionScoreMode,
    DonorMatchStrategy,
    ProjectionAggregation,
    VariableSupportFamily,
    add_dividend_composition_features,
    apply_donor_variable_semantics,
    donor_imputation_block_specs,
    donor_imputation_blocks,
    is_condition_var_compatible_with_entity,
    is_condition_var_compatible_with_targets,
    is_projected_condition_var_compatible,
    normalize_dividend_columns,
    normalize_social_security_columns,
    prune_redundant_variables,
    resolve_condition_entities_for_targets,
    resolve_variable_semantic_capabilities,
    restore_dividend_components_from_composition,
    validate_donor_variable_semantics,
)


def test_normalize_dividend_columns_prefers_atomic_components_over_totals():
    frame = pd.DataFrame(
        {
            "qualified_dividend_income": [30.0],
            "non_qualified_dividend_income": [12.0],
            "ordinary_dividend_income": [80.0],
            "dividend_income": [5.0],
        }
    )

    normalized = normalize_dividend_columns(frame)

    assert normalized["qualified_dividend_income"].tolist() == [30.0]
    assert normalized["non_qualified_dividend_income"].tolist() == [12.0]
    assert normalized["ordinary_dividend_income"].tolist() == [42.0]
    assert normalized["dividend_income"].tolist() == [42.0]


def test_normalize_dividend_columns_coalesces_sparse_total_aliases_by_row():
    frame = pd.DataFrame(
        {
            "ordinary_dividend_income": [0.0, 30.0, 0.0],
            "dividend_income": [80.0, 999.0, 0.0],
            "qualified_dividend_income": [0.0, 5.0, 0.0],
            "non_qualified_dividend_income": [0.0, 25.0, 0.0],
        }
    )

    normalized = normalize_dividend_columns(frame)

    # Row 0 carries only a dividend total (80) with no observed split, so it is
    # allocated by the SOI qualified share instead of defaulting 100% to
    # non-qualified. Rows 1-2 keep their observed components unchanged.
    assert normalized["qualified_dividend_income"].tolist() == pytest.approx(
        [62.4, 5.0, 0.0]
    )
    assert normalized["non_qualified_dividend_income"].tolist() == pytest.approx(
        [17.6, 25.0, 0.0]
    )
    assert normalized["ordinary_dividend_income"].tolist() == [80.0, 30.0, 0.0]
    assert normalized["dividend_income"].tolist() == [80.0, 30.0, 0.0]


def test_normalize_dividend_columns_splits_unsplit_total_by_qualified_share():
    # A row with only a dividend total (e.g. CPS DIV_VAL) and no qualified /
    # non-qualified components must be split by the SOI qualified share, not
    # left entirely non-qualified (which zeroed qualified dividends nationally
    # and inverted the split vs the SOI targets).
    frame = pd.DataFrame(
        {
            "qualified_dividend_income": [0.0],
            "non_qualified_dividend_income": [0.0],
            "dividend_income": [1_000.0],
        }
    )

    normalized = normalize_dividend_columns(frame)

    assert normalized["qualified_dividend_income"].tolist() == pytest.approx([780.0])
    assert normalized["non_qualified_dividend_income"].tolist() == pytest.approx(
        [220.0]
    )
    assert normalized["dividend_income"].tolist() == pytest.approx([1_000.0])


def test_normalize_social_security_columns_tracks_unclassified_residual():
    frame = pd.DataFrame(
        {
            "gross_social_security": [1_000.0, 900.0, 0.0],
            "social_security_disability": [400.0, 0.0, 0.0],
            "social_security_survivors": [100.0, 0.0, 50.0],
        }
    )

    normalized = normalize_social_security_columns(frame)

    assert normalized["social_security"].tolist() == [1_000.0, 900.0, 50.0]
    assert normalized["social_security_retirement"].tolist() == [0.0, 0.0, 0.0]
    assert normalized["social_security_disability"].tolist() == [400.0, 0.0, 0.0]
    assert normalized["social_security_survivors"].tolist() == [100.0, 0.0, 50.0]
    assert normalized["social_security_dependents"].tolist() == [0.0, 0.0, 0.0]
    assert normalized["social_security_unclassified"].tolist() == [500.0, 900.0, 0.0]


def test_prune_redundant_variables_drops_dividend_totals_when_basis_present():
    variables = {
        "income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "dividend_income",
        "ordinary_dividend_income",
    }

    assert prune_redundant_variables(variables) == {
        "income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
    }


def test_dividend_composition_features_derive_total_and_share():
    frame = pd.DataFrame(
        {
            "qualified_dividend_income": [30.0, 0.0],
            "non_qualified_dividend_income": [12.0, 0.0],
        }
    )

    enriched = add_dividend_composition_features(frame)

    assert enriched["dividend_income"].tolist() == [42.0, 0.0]
    assert enriched["ordinary_dividend_income"].tolist() == [42.0, 0.0]
    assert enriched["qualified_dividend_share"].tolist() == [30.0 / 42.0, 0.0]


def test_restore_dividend_components_from_composition_reconstructs_atomic_basis():
    frame = pd.DataFrame(
        {
            "dividend_income": [42.0, 10.0],
            "qualified_dividend_share": [30.0 / 42.0, 0.25],
        }
    )

    restored = restore_dividend_components_from_composition(frame)

    assert restored["qualified_dividend_income"].round(6).tolist() == [30.0, 2.5]
    assert restored["non_qualified_dividend_income"].round(6).tolist() == [12.0, 7.5]
    assert restored["ordinary_dividend_income"].round(6).tolist() == [42.0, 10.0]
    assert restored["dividend_income"].round(6).tolist() == [42.0, 10.0]


def test_donor_imputation_blocks_keep_dividends_in_one_composition_block():
    blocks = donor_imputation_blocks(
        {
            "qualified_dividend_income",
            "non_qualified_dividend_income",
            "taxable_interest_income",
            "partnership_s_corp_income",
        }
    )

    assert blocks == (
        ("dividend_income", "qualified_dividend_share"),
        ("partnership_s_corp_income",),
        ("taxable_interest_income",),
    )


def test_donor_imputation_block_specs_include_match_strategies_and_restored_variables():
    specs = donor_imputation_block_specs(
        {
            "qualified_dividend_income",
            "non_qualified_dividend_income",
            "taxable_interest_income",
        }
    )

    assert specs[0].model_variables == ("dividend_income", "qualified_dividend_share")
    assert specs[0].restored_variables == (
        "qualified_dividend_income",
        "non_qualified_dividend_income",
    )
    assert specs[0].strategy_for("dividend_income") is DonorMatchStrategy.RANK
    assert specs[0].native_entity is EntityType.PERSON
    assert specs[0].condition_entities == (
        EntityType.PERSON,
        EntityType.HOUSEHOLD,
        EntityType.TAX_UNIT,
    )
    assert specs[0].strategy_for("qualified_dividend_share") is DonorMatchStrategy.RANK
    assert specs[1].model_variables == ("taxable_interest_income",)
    assert specs[1].native_entity is EntityType.PERSON
    assert specs[1].condition_entities == (
        EntityType.PERSON,
        EntityType.HOUSEHOLD,
        EntityType.TAX_UNIT,
    )
    assert specs[1].strategy_for("taxable_interest_income") is DonorMatchStrategy.RANK


def test_donor_imputation_block_specs_use_zero_inflated_matching_for_sparse_irs_amounts():
    specs = donor_imputation_block_specs(
        {
            "health_savings_account_ald",
            "self_employed_health_insurance_ald",
            "self_employed_pension_contribution_ald",
            "partnership_s_corp_income",
        }
    )

    by_variable = {spec.model_variables[0]: spec for spec in specs}
    for variable_name in (
        "health_savings_account_ald",
        "self_employed_health_insurance_ald",
        "self_employed_pension_contribution_ald",
    ):
        assert by_variable[variable_name].native_entity is EntityType.PERSON
        assert by_variable[variable_name].condition_entities == (
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        )
        assert (
            by_variable[variable_name].strategy_for(variable_name)
            is DonorMatchStrategy.RANK
        )
    assert by_variable["partnership_s_corp_income"].native_entity is EntityType.PERSON
    assert (
        by_variable["partnership_s_corp_income"].strategy_for(
            "partnership_s_corp_income"
        )
        is DonorMatchStrategy.RANK
    )


def test_condition_var_compatibility_allows_household_controls_for_tax_unit_targets():
    assert is_condition_var_compatible_with_entity(
        "state_fips",
        target_entity=EntityType.TAX_UNIT,
    )
    assert is_condition_var_compatible_with_entity(
        "tenure",
        target_entity=EntityType.TAX_UNIT,
    )
    assert not is_condition_var_compatible_with_entity(
        "age",
        target_entity=EntityType.TAX_UNIT,
    )


def test_resolve_condition_entities_uses_variable_family_policy():
    assert resolve_condition_entities_for_targets(("taxable_interest_income",)) == (
        EntityType.PERSON,
        EntityType.HOUSEHOLD,
        EntityType.TAX_UNIT,
    )
    assert resolve_condition_entities_for_targets(("self_employment_income",)) == (
        EntityType.PERSON,
        EntityType.HOUSEHOLD,
        EntityType.TAX_UNIT,
    )


def test_condition_var_compatibility_with_targets_distinguishes_asset_and_labor_tax_vars():
    assert is_condition_var_compatible_with_targets(
        "age",
        target_variables=("taxable_interest_income",),
    )
    assert is_condition_var_compatible_with_targets(
        "age",
        target_variables=("self_employment_income",),
    )
    assert is_condition_var_compatible_with_targets(
        "tenure",
        target_variables=("taxable_interest_income",),
    )


def test_projected_condition_var_compatibility_promotes_person_features_to_group_entity():
    assert is_projected_condition_var_compatible(
        "age",
        projected_entity=EntityType.TAX_UNIT,
        allowed_condition_entities=(EntityType.HOUSEHOLD, EntityType.TAX_UNIT),
    )
    assert not is_projected_condition_var_compatible(
        "age",
        projected_entity=EntityType.HOUSEHOLD,
        allowed_condition_entities=(EntityType.TAX_UNIT,),
    )


def test_resolve_variable_semantic_capabilities_marks_redundant_dividend_totals():
    capabilities = resolve_variable_semantic_capabilities(
        {
            "qualified_dividend_income",
            "non_qualified_dividend_income",
            "ordinary_dividend_income",
            "dividend_income",
        }
    )

    assert not capabilities["dividend_income"].authoritative
    assert not capabilities["dividend_income"].usable_as_condition
    assert not capabilities["ordinary_dividend_income"].authoritative
    assert not capabilities["ordinary_dividend_income"].usable_as_condition


def test_variable_semantics_define_projection_aggregation_for_person_controls():
    from microplex_us.variables import variable_semantic_spec_for

    assert (
        EntityType.RECORD
        not in variable_semantic_spec_for("age").allowed_condition_entities
    )
    assert (
        variable_semantic_spec_for("age").projection_aggregation
        is ProjectionAggregation.MAX
    )
    assert (
        variable_semantic_spec_for("income").projection_aggregation
        is ProjectionAggregation.SUM
    )


def test_state_program_proxy_semantics_are_registered():
    from microplex_us.variables import variable_semantic_spec_for

    has_medicaid = variable_semantic_spec_for("has_medicaid")
    assert has_medicaid.support_family is VariableSupportFamily.SUPPORT_SENSITIVE
    assert has_medicaid.donor_match_strategy is DonorMatchStrategy.RANK
    assert has_medicaid.condition_score_mode is ConditionScoreMode.VALUE_AND_SUPPORT
    assert has_medicaid.projection_aggregation is ProjectionAggregation.MAX

    for variable_name in ("public_assistance", "ssi", "social_security"):
        spec = variable_semantic_spec_for(variable_name)
        assert spec.support_family is VariableSupportFamily.SUPPORT_SENSITIVE
        assert spec.donor_match_strategy is DonorMatchStrategy.RANK


def test_sparse_irs_ald_semantics_are_registered():
    from microplex_us.variables import variable_semantic_spec_for

    for variable_name in (
        "health_savings_account_ald",
        "self_employed_health_insurance_ald",
        "self_employed_pension_contribution_ald",
    ):
        spec = variable_semantic_spec_for(variable_name)
        assert spec.native_entity is EntityType.PERSON
        assert spec.support_family is VariableSupportFamily.SUPPORT_SENSITIVE
        assert spec.donor_match_strategy is DonorMatchStrategy.RANK
        assert spec.condition_score_mode is ConditionScoreMode.VALUE_AND_SUPPORT


def test_partnership_income_semantics_remain_person_native():
    from microplex_us.variables import variable_semantic_spec_for

    spec = variable_semantic_spec_for("partnership_s_corp_income")
    assert spec.native_entity is EntityType.PERSON
    assert spec.support_family is VariableSupportFamily.SUPPORT_SENSITIVE
    assert spec.donor_match_strategy is DonorMatchStrategy.RANK
    assert spec.condition_score_mode is ConditionScoreMode.VALUE_AND_SUPPORT


def test_sparse_irs_tax_variables_use_puf_irs_predictors():
    from microplex_us.variables import (
        PUF_DIVIDEND_INTEREST_CHALLENGER_SHARED_CONDITION_VARS,
        PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
        PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS,
        PUF_PARTNERSHIP_CHALLENGER_SHARED_CONDITION_VARS,
        PUF_PENSION_CHALLENGER_SHARED_CONDITION_VARS,
        variable_semantic_spec_for,
    )

    for variable_name in (
        "dividend_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "taxable_interest_income",
        "tax_exempt_interest_income",
        "taxable_pension_income",
        "taxable_social_security",
        "student_loan_interest",
        "health_savings_account_ald",
        "self_employed_health_insurance_ald",
        "self_employed_pension_contribution_ald",
        "tax_unit_partnership_s_corp_income",
        "partnership_s_corp_income",
    ):
        assert (
            variable_semantic_spec_for(variable_name).preferred_condition_vars
            == PUF_IRS_TAX_PREFERRED_CONDITION_VARS
        )
        assert (
            variable_semantic_spec_for(variable_name).supplemental_shared_condition_vars
            == PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS
        )

    assert PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS == ()
    assert (
        variable_semantic_spec_for(
            "taxable_interest_income"
        ).challenger_shared_condition_vars
        == PUF_DIVIDEND_INTEREST_CHALLENGER_SHARED_CONDITION_VARS
    )
    assert (
        variable_semantic_spec_for(
            "qualified_dividend_income"
        ).challenger_shared_condition_vars
        == PUF_DIVIDEND_INTEREST_CHALLENGER_SHARED_CONDITION_VARS
    )
    assert (
        variable_semantic_spec_for(
            "taxable_pension_income"
        ).challenger_shared_condition_vars
        == PUF_PENSION_CHALLENGER_SHARED_CONDITION_VARS
    )
    assert (
        variable_semantic_spec_for(
            "partnership_s_corp_income"
        ).challenger_shared_condition_vars
        == PUF_PARTNERSHIP_CHALLENGER_SHARED_CONDITION_VARS
    )


def test_rental_income_components_use_sparse_asset_conditioning():
    from microplex_us.variables import (
        RENTAL_INCOME_COMPONENT_PREFERRED_CONDITION_VARS,
        variable_semantic_spec_for,
    )

    for variable_name in ("rental_income_positive", "rental_income_negative"):
        spec = variable_semantic_spec_for(variable_name)
        assert spec.support_family is VariableSupportFamily.SUPPORT_SENSITIVE
        assert spec.donor_match_strategy is DonorMatchStrategy.RANK
        assert spec.condition_score_mode is ConditionScoreMode.VALUE_AND_SUPPORT
        assert (
            spec.preferred_condition_vars
            == RENTAL_INCOME_COMPONENT_PREFERRED_CONDITION_VARS
        )


def test_person_native_irs_semantics_match_current_policyengine_entities():
    from microplex_us.variables import variable_semantic_spec_for

    for variable_name in (
        "dividend_income",
        "ordinary_dividend_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "taxable_interest_income",
        "tax_exempt_interest_income",
        "taxable_pension_income",
        "taxable_social_security",
        "student_loan_interest",
        "self_employment_income",
    ):
        assert (
            variable_semantic_spec_for(variable_name).native_entity is EntityType.PERSON
        )


def test_self_employment_income_semantics_preserve_signed_support():
    from microplex_us.variables import variable_semantic_spec_for

    spec = variable_semantic_spec_for("self_employment_income")

    assert spec.support_family is VariableSupportFamily.CONTINUOUS
    assert spec.donor_match_strategy is DonorMatchStrategy.RANK


def test_employment_income_donor_semantics_zero_minor_wages():
    frame = pd.DataFrame(
        {
            "age": [16.0, 19.0],
            "employment_income": [50_000.0, 25_000.0],
        }
    )

    adjusted = apply_donor_variable_semantics(frame, ("employment_income",))

    assert adjusted["employment_income"].tolist() == [0.0, 25_000.0]


def test_employment_income_donor_semantics_zero_retired_senior_wages_without_esi():
    frame = pd.DataFrame(
        {
            "age": [68.0, 68.0, 68.0],
            "employment_income": [80_000.0, 80_000.0, 80_000.0],
            "social_security_retirement": [18_000.0, 18_000.0, 0.0],
            "has_esi": [0.0, 1.0, 0.0],
        }
    )

    adjusted = apply_donor_variable_semantics(frame, ("employment_income",))

    assert adjusted["employment_income"].tolist() == [0.0, 80_000.0, 80_000.0]


def test_employment_income_donor_semantics_uses_unclassified_social_security_compatibly():
    frame = pd.DataFrame(
        {
            "age": [68.0, 68.0, 68.0],
            "employment_income": [80_000.0, 80_000.0, 80_000.0],
            "social_security": [18_000.0, 18_000.0, 0.0],
            "has_esi": [0.0, 1.0, 0.0],
        }
    )

    adjusted = apply_donor_variable_semantics(frame, ("employment_income",))

    assert adjusted["social_security_retirement"].tolist() == [0.0, 0.0, 0.0]
    assert adjusted["social_security_unclassified"].tolist() == [
        18_000.0,
        18_000.0,
        0.0,
    ]
    assert adjusted["employment_income"].tolist() == [0.0, 80_000.0, 80_000.0]


def test_validate_donor_variable_semantics_reports_minor_positive_wages():
    frame = pd.DataFrame(
        {
            "age": [16.0, 19.0],
            "employment_income": [50_000.0, 25_000.0],
        }
    )

    reports = validate_donor_variable_semantics(frame, ("employment_income",))

    assert len(reports) == 1
    assert reports[0].name == "minor_positive_employment_income"
    assert reports[0].evaluated is True
    assert reports[0].violating_row_count == 1
    assert reports[0].passed is False


def test_validate_donor_variable_semantics_passes_after_minor_wage_guard():
    frame = pd.DataFrame(
        {
            "age": [16.0, 19.0],
            "employment_income": [50_000.0, 25_000.0],
        }
    )

    adjusted = apply_donor_variable_semantics(frame, ("employment_income",))
    reports = validate_donor_variable_semantics(adjusted, ("employment_income",))

    assert len(reports) == 1
    assert reports[0].violating_row_count == 0
    assert reports[0].passed is True
