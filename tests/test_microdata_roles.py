"""Tests for source-specific microdata variable roles."""

from microplex_us.microdata_roles import (
    MicrodataVariableRole,
    PolicyEngineUSVariableRole,
    blocked_policyengine_us_direct_export_variables,
    is_model_input_microdata_variable,
    is_policyengine_us_direct_export_blocked,
    microdata_variable_role,
    non_model_input_microdata_variables,
    policyengine_us_variable_role,
)


def test_puf_tax_credit_lines_are_reported_outputs_not_model_inputs():
    for variable in (
        "foreign_tax_credit",
        "savers_credit",
        "state_and_local_sales_or_income_tax",
        "state_income_tax_paid",
        "taxable_social_security",
        "taxable_unemployment_compensation",
    ):
        assert (
            microdata_variable_role("irs_soi_puf_2024", variable)
            is MicrodataVariableRole.CALCULATED_TAX_OUTPUT
        )
        assert not is_model_input_microdata_variable("irs_soi_puf_2024", variable)
    assert is_model_input_microdata_variable(
        "irs_soi_puf_2024",
        "taxable_interest_income",
    )


def test_non_model_input_microdata_variables_is_source_specific():
    assert non_model_input_microdata_variables(
        "irs_soi_puf_2024",
        ["savers_credit", "taxable_interest_income", "taxable_social_security"],
    ) == ("savers_credit", "taxable_social_security")
    assert non_model_input_microdata_variables(
        "cps_asec_2024",
        ["savers_credit"],
    ) == ()


def test_policyengine_us_variable_roles_separate_inputs_from_outputs():
    assert (
        policyengine_us_variable_role("takes_up_snap_if_eligible")
        is PolicyEngineUSVariableRole.TAKEUP_INPUT
    )
    assert (
        policyengine_us_variable_role("takes_up_eitc")
        is PolicyEngineUSVariableRole.TAKEUP_INPUT
    )
    assert (
        policyengine_us_variable_role(
            "would_file_if_eligible_for_refundable_credit"
        )
        is PolicyEngineUSVariableRole.TAKEUP_INPUT
    )
    assert (
        policyengine_us_variable_role("would_file_taxes_voluntarily")
        is PolicyEngineUSVariableRole.TAKEUP_INPUT
    )
    assert (
        policyengine_us_variable_role("snap")
        is PolicyEngineUSVariableRole.CALCULATED_OUTPUT
    )
    assert (
        policyengine_us_variable_role("state_income_tax")
        is PolicyEngineUSVariableRole.CALCULATED_OUTPUT
    )
    assert (
        policyengine_us_variable_role("filing_status")
        is PolicyEngineUSVariableRole.PRESERVED_INPUT
    )
    assert (
        policyengine_us_variable_role("snap_reported")
        is PolicyEngineUSVariableRole.REPORTED_OUTPUT
    )
    assert (
        policyengine_us_variable_role("taxable_interest_income")
        is PolicyEngineUSVariableRole.PRESERVED_INPUT
    )
    assert (
        policyengine_us_variable_role("non_sch_d_capital_gains")
        is PolicyEngineUSVariableRole.PRESERVED_INPUT
    )
    assert (
        policyengine_us_variable_role("long_term_capital_gains_before_response")
        is PolicyEngineUSVariableRole.PRESERVED_INPUT
    )
    assert (
        policyengine_us_variable_role("net_capital_gains")
        is PolicyEngineUSVariableRole.CALCULATED_OUTPUT
    )


def test_policyengine_direct_export_guard_blocks_calculated_and_reported_outputs():
    blocked = blocked_policyengine_us_direct_export_variables(
        [
            "takes_up_snap_if_eligible",
            "would_file_taxes_voluntarily",
            "net_capital_gains",
            "non_sch_d_capital_gains",
            "filing_status",
            "snap",
            "snap_reported",
            "state_income_tax",
            "taxable_interest_income",
        ]
    )

    assert blocked == (
        "net_capital_gains",
        "snap",
        "snap_reported",
        "state_income_tax",
    )
    assert not is_policyengine_us_direct_export_blocked("filing_status")
    assert is_policyengine_us_direct_export_blocked("snap")
    assert not is_policyengine_us_direct_export_blocked("takes_up_snap_if_eligible")
    assert not is_policyengine_us_direct_export_blocked(
        "would_file_taxes_voluntarily"
    )
    assert not is_policyengine_us_direct_export_blocked("non_sch_d_capital_gains")
