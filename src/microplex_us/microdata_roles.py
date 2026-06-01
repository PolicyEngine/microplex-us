"""Source-specific microdata variable role metadata.

This is the Microplex-side bridge to the richer Arch source-data contract:
Arch preserves what a source says, while Microplex decides which source columns
are model inputs versus source-reported outputs or diagnostics.
"""

from __future__ import annotations

from enum import Enum


class MicrodataVariableRole(Enum):
    """How Microplex should treat one source-native microdata variable."""

    SOURCE_INPUT = "source_input"
    REPORTED_RETURN_LINE_INPUT = "reported_return_line_input"
    CALCULATED_TAX_OUTPUT = "calculated_tax_output"


class PolicyEngineUSVariableRole(Enum):
    """How Microplex should treat a PolicyEngine US variable at export time."""

    PRESERVED_INPUT = "preserved_input"
    TAKEUP_INPUT = "takeup_input"
    REPORTED_OUTPUT = "reported_output"
    CALCULATED_OUTPUT = "calculated_output"


PUF_CALCULATED_TAX_OUTPUT_VARIABLES: frozenset[str] = frozenset(
    {
        "american_opportunity_credit",
        "amt_foreign_tax_credit",
        "early_withdrawal_penalty",
        "energy_efficient_home_improvement_credit",
        "excess_withheld_payroll_tax",
        "foreign_tax_credit",
        "general_business_credit",
        "other_credits",
        "prior_year_minimum_tax_credit",
        "recapture_of_investment_credit",
        "savers_credit",
        "state_and_local_sales_or_income_tax",
        "state_income_tax_paid",
        "taxable_social_security",
        "taxable_unemployment_compensation",
        "unreported_payroll_tax",
    }
)

POLICYENGINE_US_TAKEUP_INPUT_VARIABLES: frozenset[str] = frozenset(
    {
        "takes_up_aca_if_eligible",
        "takes_up_early_head_start_if_eligible",
        "takes_up_eitc",
        "takes_up_head_start_if_eligible",
        "takes_up_housing_assistance_if_eligible",
        "takes_up_medicaid_if_eligible",
        "takes_up_medicare_if_eligible",
        "takes_up_snap_if_eligible",
        "takes_up_ssi_if_eligible",
        "takes_up_tanf_if_eligible",
        "would_claim_wic",
        "would_file_taxes_voluntarily",
    }
)

POLICYENGINE_US_REPORTED_BENEFIT_AMOUNT_VARIABLES: frozenset[str] = frozenset(
    {
        "snap_reported",
        "ssi_reported",
        "tanf_reported",
    }
)

POLICYENGINE_US_REPORTED_TAX_OUTPUT_VARIABLES: frozenset[str] = frozenset(
    PUF_CALCULATED_TAX_OUTPUT_VARIABLES
    | {
        "state_income_tax_reported",
    }
)

POLICYENGINE_US_REPORTED_OUTPUT_VARIABLES: frozenset[str] = frozenset(
    POLICYENGINE_US_REPORTED_BENEFIT_AMOUNT_VARIABLES
    | POLICYENGINE_US_REPORTED_TAX_OUTPUT_VARIABLES
)

POLICYENGINE_US_CALCULATED_OUTPUT_VARIABLES: frozenset[str] = frozenset(
    {
        "aca_ptc",
        "additional_ctc",
        "assigned_aca_ptc",
        "loss_limited_net_capital_gains",
        "net_capital_gains",
        "chip_enrolled",
        "ctc",
        "early_head_start",
        "eitc",
        "filing_status",
        "head_start",
        "income_tax",
        "income_tax_positive",
        "is_aca_ptc_eligible",
        "medicaid",
        "medicaid_cost",
        "medicaid_enrolled",
        "non_refundable_ctc",
        "premium_tax_credit",
        "refundable_ctc",
        "rent",
        "snap",
        "ssi",
        "state_income_tax",
        "tanf",
        "total_income_tax",
        "wic",
    }
)

POLICYENGINE_US_CONSTRUCTION_INPUT_VARIABLES: frozenset[str] = frozenset()

POLICYENGINE_US_DIRECT_EXPORT_BLOCKED_VARIABLES: frozenset[str] = frozenset(
    POLICYENGINE_US_CALCULATED_OUTPUT_VARIABLES
    | POLICYENGINE_US_REPORTED_OUTPUT_VARIABLES
)


def source_name_matches_prefix(source_name: str, prefix: str) -> bool:
    """Return whether a source name is an exact or year-suffixed source prefix."""
    return source_name == prefix or source_name.startswith(f"{prefix}_")


def microdata_variable_role(
    source_name: str,
    variable_name: str,
) -> MicrodataVariableRole:
    """Resolve the source-specific role for one microdata variable."""
    if (
        source_name_matches_prefix(source_name, "irs_soi_puf")
        and variable_name in PUF_CALCULATED_TAX_OUTPUT_VARIABLES
    ):
        return MicrodataVariableRole.CALCULATED_TAX_OUTPUT
    return MicrodataVariableRole.SOURCE_INPUT


def is_model_input_microdata_variable(
    source_name: str,
    variable_name: str,
) -> bool:
    """Return whether a source column should enter model-ready microdata."""
    return microdata_variable_role(
        source_name,
        variable_name,
    ) is not MicrodataVariableRole.CALCULATED_TAX_OUTPUT


def non_model_input_microdata_variables(
    source_name: str,
    variable_names: list[str] | tuple[str, ...] | set[str] | frozenset[str],
) -> tuple[str, ...]:
    """Return source columns that should stay out of model-ready microdata."""
    return tuple(
        variable_name
        for variable_name in variable_names
        if not is_model_input_microdata_variable(source_name, variable_name)
    )


def policyengine_us_variable_role(variable_name: str) -> PolicyEngineUSVariableRole:
    """Resolve the Microplex role for a PolicyEngine US variable name."""
    if variable_name in POLICYENGINE_US_CONSTRUCTION_INPUT_VARIABLES:
        return PolicyEngineUSVariableRole.PRESERVED_INPUT
    if variable_name in POLICYENGINE_US_CALCULATED_OUTPUT_VARIABLES:
        return PolicyEngineUSVariableRole.CALCULATED_OUTPUT
    if variable_name in POLICYENGINE_US_REPORTED_OUTPUT_VARIABLES:
        return PolicyEngineUSVariableRole.REPORTED_OUTPUT
    if variable_name in POLICYENGINE_US_TAKEUP_INPUT_VARIABLES:
        return PolicyEngineUSVariableRole.TAKEUP_INPUT
    return PolicyEngineUSVariableRole.PRESERVED_INPUT


def is_policyengine_us_direct_export_blocked(variable_name: str) -> bool:
    """Return whether a source column may not override a PE-US variable."""
    return (
        policyengine_us_variable_role(variable_name)
        in {
            PolicyEngineUSVariableRole.CALCULATED_OUTPUT,
            PolicyEngineUSVariableRole.REPORTED_OUTPUT,
        }
    )


def blocked_policyengine_us_direct_export_variables(
    variable_names: list[str] | tuple[str, ...] | set[str] | frozenset[str],
) -> tuple[str, ...]:
    """Return requested direct overrides that violate the variable contract."""
    return tuple(
        sorted(
            variable_name
            for variable_name in variable_names
            if is_policyengine_us_direct_export_blocked(variable_name)
        )
    )
