"""Named target-cell profiles for PolicyEngine US target selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyEngineUSTargetCell:
    """One exact target cell from the PolicyEngine US target DB."""

    variable: str
    geo_level: str | None = None
    domain_variable: str | None = None
    geographic_id: str | None = None

    def to_provider_filter(self) -> dict[str, str | None]:
        return {
            "variable": self.variable,
            "geo_level": self.geo_level,
            "domain_variable": self.domain_variable,
            "geographic_id": self.geographic_id,
        }


PolicyEngineUSTargetCellKey = tuple[str, str | None, str | None, str | None]


def _target_cell_key(cell: PolicyEngineUSTargetCell) -> PolicyEngineUSTargetCellKey:
    return (
        cell.variable,
        cell.geo_level,
        cell.domain_variable,
        cell.geographic_id,
    )


PE_NATIVE_BROAD_TARGET_CELLS: tuple[PolicyEngineUSTargetCell, ...] = (
    PolicyEngineUSTargetCell(
        "aca_ptc", geo_level="national", domain_variable="aca_ptc"
    ),
    PolicyEngineUSTargetCell("adjusted_gross_income", geo_level="national"),
    PolicyEngineUSTargetCell(
        "adjusted_gross_income",
        geo_level="national",
        domain_variable="adjusted_gross_income",
    ),
    PolicyEngineUSTargetCell(
        "adjusted_gross_income",
        geo_level="national",
        domain_variable="adjusted_gross_income,filing_status,income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell(
        "adjusted_gross_income",
        geo_level="national",
        domain_variable="adjusted_gross_income,income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell("alimony_expense", geo_level="national"),
    PolicyEngineUSTargetCell("alimony_income", geo_level="national"),
    PolicyEngineUSTargetCell("charitable_deduction", geo_level="national"),
    PolicyEngineUSTargetCell("childcare_expenses", geo_level="national"),
    PolicyEngineUSTargetCell("child_support_expense", geo_level="national"),
    PolicyEngineUSTargetCell("child_support_received", geo_level="national"),
    PolicyEngineUSTargetCell("deductible_mortgage_interest", geo_level="national"),
    PolicyEngineUSTargetCell("dividend_income", geo_level="national"),
    PolicyEngineUSTargetCell(
        "dividend_income", geo_level="national", domain_variable="dividend_income"
    ),
    PolicyEngineUSTargetCell("employment_income_before_lsr", geo_level="national"),
    PolicyEngineUSTargetCell(
        "employment_income", geo_level="national", domain_variable="employment_income"
    ),
    PolicyEngineUSTargetCell("eitc", geo_level="national"),
    PolicyEngineUSTargetCell(
        "eitc", geo_level="national", domain_variable="eitc_child_count"
    ),
    PolicyEngineUSTargetCell(
        "eitc",
        geo_level="national",
        domain_variable="adjusted_gross_income,eitc,eitc_child_count",
    ),
    PolicyEngineUSTargetCell(
        "health_insurance_premiums_without_medicare_part_b",
        geo_level="national",
    ),
    PolicyEngineUSTargetCell(
        "household_count",
        geo_level="national",
        domain_variable="spm_unit_energy_subsidy_reported",
    ),
    PolicyEngineUSTargetCell(
        "income_tax", geo_level="national", domain_variable="income_tax"
    ),
    PolicyEngineUSTargetCell(
        "income_tax_before_credits",
        geo_level="national",
        domain_variable="income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell("income_tax_positive", geo_level="national"),
    PolicyEngineUSTargetCell("investment_interest_expense", geo_level="national"),
    PolicyEngineUSTargetCell("interest_deduction", geo_level="national"),
    PolicyEngineUSTargetCell("long_term_capital_gains", geo_level="national"),
    PolicyEngineUSTargetCell("medicaid", geo_level="national"),
    PolicyEngineUSTargetCell("medical_expense_deduction", geo_level="national"),
    PolicyEngineUSTargetCell(
        "medical_expense_deduction",
        geo_level="national",
        domain_variable="medical_expense_deduction",
    ),
    PolicyEngineUSTargetCell(
        "medical_expense_deduction",
        geo_level="national",
        domain_variable="medical_expense_deduction,tax_unit_itemizes",
    ),
    PolicyEngineUSTargetCell("medicare_part_b_premiums", geo_level="national"),
    PolicyEngineUSTargetCell(
        "net_capital_gains", geo_level="national", domain_variable="net_capital_gains"
    ),
    PolicyEngineUSTargetCell("net_worth", geo_level="national"),
    PolicyEngineUSTargetCell(
        "non_refundable_ctc",
        geo_level="national",
        domain_variable="adjusted_gross_income,non_refundable_ctc",
    ),
    PolicyEngineUSTargetCell(
        "non_refundable_ctc",
        geo_level="national",
        domain_variable="non_refundable_ctc",
    ),
    PolicyEngineUSTargetCell("other_medical_expenses", geo_level="national"),
    PolicyEngineUSTargetCell("over_the_counter_health_expenses", geo_level="national"),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="national", domain_variable="aca_ptc"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="national", domain_variable="age"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="national", domain_variable="medicaid"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="national", domain_variable="snap"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="national", domain_variable="ssi"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="national", domain_variable="ssi,is_ssi_aged"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="national", domain_variable="ssi,is_blind"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="national", domain_variable="ssi,is_ssi_disabled"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="national", domain_variable="ssn_card_type"
    ),
    PolicyEngineUSTargetCell(
        "qualified_business_income_deduction", geo_level="national"
    ),
    PolicyEngineUSTargetCell(
        "qualified_business_income_deduction",
        geo_level="national",
        domain_variable="qualified_business_income_deduction",
    ),
    PolicyEngineUSTargetCell(
        "qualified_dividend_income",
        geo_level="national",
        domain_variable="qualified_dividend_income",
    ),
    PolicyEngineUSTargetCell("real_estate_taxes", geo_level="national"),
    PolicyEngineUSTargetCell(
        "real_estate_taxes",
        geo_level="national",
        domain_variable="real_estate_taxes",
    ),
    PolicyEngineUSTargetCell(
        "real_estate_taxes",
        geo_level="national",
        domain_variable="real_estate_taxes,tax_unit_itemizes",
    ),
    PolicyEngineUSTargetCell(
        "refundable_ctc",
        geo_level="national",
        domain_variable="adjusted_gross_income,refundable_ctc",
    ),
    PolicyEngineUSTargetCell(
        "refundable_ctc", geo_level="national", domain_variable="refundable_ctc"
    ),
    PolicyEngineUSTargetCell("rent", geo_level="national"),
    PolicyEngineUSTargetCell("rental_income", geo_level="national"),
    PolicyEngineUSTargetCell(
        "rental_income", geo_level="national", domain_variable="rental_income"
    ),
    PolicyEngineUSTargetCell("roth_401k_contributions", geo_level="national"),
    PolicyEngineUSTargetCell("roth_ira_contributions", geo_level="national"),
    PolicyEngineUSTargetCell("salt", geo_level="national", domain_variable="salt"),
    PolicyEngineUSTargetCell(
        "salt", geo_level="national", domain_variable="salt,tax_unit_itemizes"
    ),
    PolicyEngineUSTargetCell("salt_deduction", geo_level="national"),
    PolicyEngineUSTargetCell("salt_refund_income", geo_level="national"),
    PolicyEngineUSTargetCell(
        "self_employed_pension_contribution_ald", geo_level="national"
    ),
    PolicyEngineUSTargetCell(
        "self_employment_income",
        geo_level="national",
    ),
    PolicyEngineUSTargetCell(
        "self_employment_income",
        geo_level="national",
        domain_variable="self_employment_income",
    ),
    PolicyEngineUSTargetCell("short_term_capital_gains", geo_level="national"),
    PolicyEngineUSTargetCell(
        "household_count", geo_level="national", domain_variable="snap"
    ),
    PolicyEngineUSTargetCell("snap", geo_level="national"),
    PolicyEngineUSTargetCell("social_security", geo_level="national"),
    PolicyEngineUSTargetCell("social_security_dependents", geo_level="national"),
    PolicyEngineUSTargetCell("social_security_disability", geo_level="national"),
    PolicyEngineUSTargetCell("social_security_retirement", geo_level="national"),
    PolicyEngineUSTargetCell("social_security_survivors", geo_level="national"),
    PolicyEngineUSTargetCell("spm_unit_capped_housing_subsidy", geo_level="national"),
    PolicyEngineUSTargetCell(
        "spm_unit_capped_work_childcare_expenses", geo_level="national"
    ),
    PolicyEngineUSTargetCell(
        "spm_unit_count", geo_level="national", domain_variable="tanf"
    ),
    PolicyEngineUSTargetCell("ssi", geo_level="national"),
    PolicyEngineUSTargetCell(
        "ssi", geo_level="national", domain_variable="ssi,is_ssi_aged"
    ),
    PolicyEngineUSTargetCell(
        "ssi", geo_level="national", domain_variable="ssi,is_blind"
    ),
    PolicyEngineUSTargetCell(
        "ssi", geo_level="national", domain_variable="ssi,is_ssi_disabled"
    ),
    PolicyEngineUSTargetCell("tanf", geo_level="national"),
    PolicyEngineUSTargetCell("tanf", geo_level="national", domain_variable="tanf"),
    PolicyEngineUSTargetCell(
        "tax_exempt_interest_income",
        geo_level="national",
        domain_variable="tax_exempt_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="national", domain_variable="aca_ptc"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="adjusted_gross_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="adjusted_gross_income,filing_status,income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="adjusted_gross_income,income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="adjusted_gross_income,non_refundable_ctc",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="adjusted_gross_income,refundable_ctc",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="dividend_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="employment_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="eitc_child_count",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="adjusted_gross_income,eitc,eitc_child_count",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="national", domain_variable="income_tax"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="investment_interest_expense",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="long_term_capital_gains",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="medical_expense_deduction",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="medical_expense_deduction,tax_unit_itemizes",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="net_capital_gains",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="non_refundable_ctc",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="qualified_business_income_deduction",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="qualified_dividend_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="real_estate_taxes",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="real_estate_taxes,tax_unit_itemizes",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="refundable_ctc",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="national", domain_variable="rental_income"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="national", domain_variable="salt"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="salt_refund_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="salt,tax_unit_itemizes",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="self_employment_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="short_term_capital_gains",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="tax_exempt_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="tax_unit_partnership_s_corp_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="taxable_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="taxable_ira_distributions",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="taxable_pension_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="taxable_social_security",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="total_self_employment_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="unemployment_compensation",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_partnership_s_corp_income",
        geo_level="national",
        domain_variable="tax_unit_partnership_s_corp_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_interest_income",
        geo_level="national",
        domain_variable="taxable_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_ira_distributions",
        geo_level="national",
        domain_variable="taxable_ira_distributions",
    ),
    PolicyEngineUSTargetCell(
        "taxable_pension_income",
        geo_level="national",
        domain_variable="taxable_pension_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_social_security",
        geo_level="national",
        domain_variable="taxable_social_security",
    ),
    PolicyEngineUSTargetCell("tip_income", geo_level="national"),
    PolicyEngineUSTargetCell(
        "total_self_employment_income",
        geo_level="national",
        domain_variable="total_self_employment_income",
    ),
    PolicyEngineUSTargetCell("traditional_401k_contributions", geo_level="national"),
    PolicyEngineUSTargetCell("traditional_ira_contributions", geo_level="national"),
    PolicyEngineUSTargetCell("unemployment_compensation", geo_level="national"),
    PolicyEngineUSTargetCell(
        "unemployment_compensation",
        geo_level="national",
        domain_variable="unemployment_compensation",
    ),
    PolicyEngineUSTargetCell("aca_ptc", geo_level="state", domain_variable=None),
    PolicyEngineUSTargetCell("aca_ptc", geo_level="state", domain_variable="aca_ptc"),
    PolicyEngineUSTargetCell("adjusted_gross_income", geo_level="state"),
    PolicyEngineUSTargetCell(
        "adjusted_gross_income",
        geo_level="state",
        domain_variable="adjusted_gross_income",
    ),
    PolicyEngineUSTargetCell(
        "dividend_income", geo_level="state", domain_variable="dividend_income"
    ),
    PolicyEngineUSTargetCell("employment_income_before_lsr", geo_level="state"),
    PolicyEngineUSTargetCell(
        "employment_income", geo_level="state", domain_variable="employment_income"
    ),
    PolicyEngineUSTargetCell(
        "eitc", geo_level="state", domain_variable="eitc_child_count"
    ),
    PolicyEngineUSTargetCell(
        "household_count", geo_level="state", domain_variable="snap"
    ),
    PolicyEngineUSTargetCell(
        "income_tax", geo_level="state", domain_variable="income_tax"
    ),
    PolicyEngineUSTargetCell(
        "income_tax_before_credits",
        geo_level="state",
        domain_variable="income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell("investment_interest_expense", geo_level="state"),
    PolicyEngineUSTargetCell("long_term_capital_gains", geo_level="state"),
    PolicyEngineUSTargetCell(
        "medical_expense_deduction",
        geo_level="state",
        domain_variable="medical_expense_deduction",
    ),
    PolicyEngineUSTargetCell(
        "medical_expense_deduction",
        geo_level="state",
        domain_variable="medical_expense_deduction,tax_unit_itemizes",
    ),
    PolicyEngineUSTargetCell(
        "net_capital_gains", geo_level="state", domain_variable="net_capital_gains"
    ),
    PolicyEngineUSTargetCell(
        "non_refundable_ctc",
        geo_level="state",
        domain_variable="non_refundable_ctc",
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="state", domain_variable="aca_ptc"
    ),
    PolicyEngineUSTargetCell(
        "person_count",
        geo_level="state",
        domain_variable="aca_ptc,is_aca_ptc_eligible",
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="state", domain_variable="adjusted_gross_income"
    ),
    PolicyEngineUSTargetCell("person_count", geo_level="state", domain_variable="age"),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="state", domain_variable="is_pregnant"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="state", domain_variable="medicaid_enrolled"
    ),
    PolicyEngineUSTargetCell("person_count", geo_level="state", domain_variable="snap"),
    PolicyEngineUSTargetCell("person_count", geo_level="state", domain_variable="ssi"),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="state", domain_variable="ssi,is_ssi_aged"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="state", domain_variable="ssi,is_blind"
    ),
    PolicyEngineUSTargetCell(
        "person_count", geo_level="state", domain_variable="ssi,is_ssi_disabled"
    ),
    PolicyEngineUSTargetCell(
        "qualified_business_income_deduction",
        geo_level="state",
        domain_variable="qualified_business_income_deduction",
    ),
    PolicyEngineUSTargetCell(
        "qualified_dividend_income",
        geo_level="state",
        domain_variable="qualified_dividend_income",
    ),
    PolicyEngineUSTargetCell(
        "real_estate_taxes", geo_level="state", domain_variable="real_estate_taxes"
    ),
    PolicyEngineUSTargetCell(
        "real_estate_taxes",
        geo_level="state",
        domain_variable="real_estate_taxes,tax_unit_itemizes",
    ),
    PolicyEngineUSTargetCell(
        "refundable_ctc", geo_level="state", domain_variable="refundable_ctc"
    ),
    PolicyEngineUSTargetCell(
        "rental_income", geo_level="state", domain_variable="rental_income"
    ),
    PolicyEngineUSTargetCell("salt", geo_level="state", domain_variable="salt"),
    PolicyEngineUSTargetCell(
        "salt", geo_level="state", domain_variable="salt,tax_unit_itemizes"
    ),
    PolicyEngineUSTargetCell("salt_refund_income", geo_level="state"),
    PolicyEngineUSTargetCell(
        "self_employment_income",
        geo_level="state",
    ),
    PolicyEngineUSTargetCell(
        "self_employment_income",
        geo_level="state",
        domain_variable="self_employment_income",
    ),
    PolicyEngineUSTargetCell("short_term_capital_gains", geo_level="state"),
    PolicyEngineUSTargetCell("snap", geo_level="state", domain_variable="snap"),
    PolicyEngineUSTargetCell(
        "spm_unit_count", geo_level="state", domain_variable="tanf"
    ),
    PolicyEngineUSTargetCell("ssi", geo_level="state"),
    PolicyEngineUSTargetCell("state_income_tax", geo_level="state"),
    PolicyEngineUSTargetCell("tanf", geo_level="state", domain_variable="tanf"),
    PolicyEngineUSTargetCell(
        "tax_exempt_interest_income",
        geo_level="state",
        domain_variable="tax_exempt_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="aca_ptc"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="adjusted_gross_income"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="dividend_income"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="employment_income"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="eitc_child_count"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="income_tax"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="investment_interest_expense",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="long_term_capital_gains",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="medical_expense_deduction",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="medical_expense_deduction,tax_unit_itemizes",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="net_capital_gains"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="non_refundable_ctc"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="qualified_business_income_deduction",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="qualified_dividend_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="real_estate_taxes"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="real_estate_taxes,tax_unit_itemizes",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="refundable_ctc"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="rental_income"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="salt"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="salt_refund_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="salt,tax_unit_itemizes",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="selected_marketplace_plan_benchmark_ratio,used_aca_ptc",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="self_employment_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="short_term_capital_gains",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="tax_exempt_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="tax_unit_partnership_s_corp_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="taxable_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="taxable_ira_distributions",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="taxable_pension_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="taxable_social_security",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="total_self_employment_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="unemployment_compensation",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count", geo_level="state", domain_variable="used_aca_ptc"
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_partnership_s_corp_income",
        geo_level="state",
        domain_variable="tax_unit_partnership_s_corp_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_interest_income",
        geo_level="state",
        domain_variable="taxable_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_ira_distributions",
        geo_level="state",
        domain_variable="taxable_ira_distributions",
    ),
    PolicyEngineUSTargetCell(
        "taxable_pension_income",
        geo_level="state",
        domain_variable="taxable_pension_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_social_security",
        geo_level="state",
        domain_variable="taxable_social_security",
    ),
    PolicyEngineUSTargetCell(
        "total_self_employment_income",
        geo_level="state",
        domain_variable="total_self_employment_income",
    ),
    PolicyEngineUSTargetCell(
        "unemployment_compensation",
        geo_level="state",
        domain_variable="unemployment_compensation",
    ),
)

_PE_NATIVE_BROAD_NO_STATE_ACA_EXCLUDED_CELLS = frozenset(
    {
        ("aca_ptc", "state", None, None),
        ("aca_ptc", "state", "aca_ptc", None),
        ("person_count", "state", "aca_ptc", None),
        ("person_count", "state", "aca_ptc,is_aca_ptc_eligible", None),
        ("tax_unit_count", "state", "aca_ptc", None),
        (
            "tax_unit_count",
            "state",
            "selected_marketplace_plan_benchmark_ratio,used_aca_ptc",
            None,
        ),
        ("tax_unit_count", "state", "used_aca_ptc", None),
    }
)

PE_NATIVE_BROAD_NO_STATE_ACA_TARGET_CELLS: tuple[PolicyEngineUSTargetCell, ...] = tuple(
    cell
    for cell in PE_NATIVE_BROAD_TARGET_CELLS
    if _target_cell_key(cell) not in _PE_NATIVE_BROAD_NO_STATE_ACA_EXCLUDED_CELLS
)

PE_NATIVE_BROAD_SOURCE_BACKED_EXCLUDED_CELL_REASONS: dict[
    PolicyEngineUSTargetCellKey,
    str,
] = {
    (
        "adjusted_gross_income",
        "national",
        "adjusted_gross_income,filing_status,income_tax_before_credits",
        None,
    ): (
        "SOI source packages currently loaded by Arch do not publish adjusted "
        "gross income jointly by AGI band, filing status, and returns with "
        "positive income tax before credits."
    ),
    (
        "adjusted_gross_income",
        "national",
        "adjusted_gross_income,income_tax_before_credits",
        None,
    ): (
        "SOI source packages currently loaded by Arch publish AGI bands and "
        "income-tax-before-credits returns separately, not AGI amounts "
        "restricted to returns with positive income tax before credits."
    ),
    (
        "tax_unit_count",
        "national",
        "adjusted_gross_income,filing_status,income_tax_before_credits",
        None,
    ): (
        "SOI Historic Table 2 does not provide the full AGI by filing-status "
        "by positive-income-tax-before-credits joint count required by this "
        "PolicyEngine cell."
    ),
    (
        "person_count",
        "national",
        "ssn_card_type",
        None,
    ): (
        "PolicyEngine ssn_card_type is a modeled legal-status input; no "
        "accepted primary aggregate source mapping is encoded for Arch."
    ),
    (
        "person_count",
        "state",
        "is_pregnant",
        None,
    ): (
        "The PolicyEngine cell is a pregnancy stock by state; live births are "
        "a flow and are not a defensible direct source fact for this target."
    ),
    (
        "person_count",
        "state",
        "adjusted_gross_income",
        None,
    ): (
        "Loaded SOI state AGI sources provide return counts and AGI amounts, "
        "not filer-person counts by AGI band."
    ),
    (
        "child_support_expense",
        "national",
        None,
        None,
    ): (
        "No accepted primary source mapping is encoded for this "
        "survey/model-input expense variable."
    ),
    (
        "child_support_received",
        "national",
        None,
        None,
    ): (
        "No accepted primary source mapping is encoded for this "
        "survey/model-input receipt variable."
    ),
    (
        "childcare_expenses",
        "national",
        None,
        None,
    ): (
        "IRS child-care credit expenses and W-2 dependent-care benefits are "
        "narrower tax concepts than PolicyEngine childcare_expenses, so they "
        "are not treated as source-equivalent."
    ),
    (
        "health_insurance_premiums_without_medicare_part_b",
        "national",
        None,
        None,
    ): (
        "This premium component is a modeled/survey input; no accepted primary "
        "aggregate source mapping is encoded for Arch."
    ),
    (
        "other_medical_expenses",
        "national",
        None,
        None,
    ): (
        "This out-of-pocket medical expense component is a survey/model input "
        "without an accepted primary aggregate source mapping."
    ),
    (
        "over_the_counter_health_expenses",
        "national",
        None,
        None,
    ): (
        "This out-of-pocket medical expense component is a survey/model input "
        "without an accepted primary aggregate source mapping."
    ),
    (
        "rent",
        "national",
        None,
        None,
    ): (
        "PolicyEngine rent is a household survey/model input; ACS rent tables "
        "do not provide a direct aggregate source fact for this exact variable."
    ),
    (
        "salt",
        "national",
        "salt",
        None,
    ): (
        "SOI Table 2.1 itemized deduction sources cover itemizers; "
        "PolicyEngine salt can be positive outside the itemizer domain."
    ),
    (
        "tax_unit_count",
        "national",
        "salt",
        None,
    ): (
        "SOI Table 2.1 publishes separate component counts, not the union "
        "count of tax units with positive PolicyEngine salt."
    ),
    (
        "tax_unit_count",
        "national",
        "salt,tax_unit_itemizes",
        None,
    ): (
        "SOI Table 2.1 publishes separate component counts, not the union "
        "count of itemizing tax units with positive PolicyEngine salt."
    ),
    (
        "salt",
        "state",
        "salt",
        None,
    ): (
        "Loaded state SOI sources do not provide an exact state-level "
        "PolicyEngine salt amount; total state/local taxes also include "
        "personal property taxes."
    ),
    (
        "salt",
        "state",
        "salt,tax_unit_itemizes",
        None,
    ): (
        "Loaded state SOI sources do not provide state-level itemizer salt "
        "as income-or-sales tax plus real estate tax without personal "
        "property taxes."
    ),
    (
        "tax_unit_count",
        "state",
        "salt",
        None,
    ): (
        "Loaded state SOI sources do not provide the union count of tax units "
        "with positive PolicyEngine salt."
    ),
    (
        "tax_unit_count",
        "state",
        "salt,tax_unit_itemizes",
        None,
    ): (
        "Loaded state SOI sources do not provide the union count of itemizing "
        "tax units with positive PolicyEngine salt."
    ),
    (
        "spm_unit_capped_housing_subsidy",
        "national",
        None,
        None,
    ): (
        "This is a capped SPM model amount rather than a direct publisher source fact."
    ),
    (
        "spm_unit_capped_work_childcare_expenses",
        "national",
        None,
        None,
    ): (
        "This is a capped SPM model amount rather than a direct publisher source fact."
    ),
}

_PENDING_SSI_DETAIL_SOURCE_REASON = (
    "Current Arch SSA source packages cover broad SSI payment totals but do not "
    "yet encode exact recipient-count, state-level, aged, blind, or disabled "
    "SSI source facts for this PolicyEngine cell."
)

_PENDING_IRS_DETAIL_SOURCE_REASON = (
    "Current Arch IRS SOI source packages do not yet encode an exact source "
    "fact for this detailed PolicyEngine tax cell at the requested geography "
    "and domain."
)

_PENDING_BEA_STATE_WAGE_SOURCE_REASON = (
    "Current Arch BEA state packages do not yet include the full component "
    "panel needed to derive state residence-adjusted employment income before "
    "legal-social-responsibility adjustments."
)

_PENDING_ARCH_SOURCE_BACKED_CELL_REASONS: dict[
    PolicyEngineUSTargetCellKey,
    str,
] = {
    **{
        ("person_count", geo_level, domain_variable, None): (
            _PENDING_SSI_DETAIL_SOURCE_REASON
        )
        for geo_level in ("national", "state")
        for domain_variable in (
            "ssi",
            "ssi,is_ssi_aged",
            "ssi,is_blind",
            "ssi,is_ssi_disabled",
        )
    },
    **{
        ("ssi", geo_level, domain_variable, None): (
            _PENDING_SSI_DETAIL_SOURCE_REASON
        )
        for geo_level, domain_variable in (
            ("national", "ssi,is_ssi_aged"),
            ("national", "ssi,is_blind"),
            ("national", "ssi,is_ssi_disabled"),
            ("state", None),
        )
    },
    **{
        (variable, geo_level, None, None): _PENDING_IRS_DETAIL_SOURCE_REASON
        for variable in (
            "long_term_capital_gains",
            "salt_refund_income",
            "short_term_capital_gains",
        )
        for geo_level in ("national", "state")
    },
    (
        "investment_interest_expense",
        "state",
        None,
        None,
    ): _PENDING_IRS_DETAIL_SOURCE_REASON,
    (
        "employment_income_before_lsr",
        "state",
        None,
        None,
    ): _PENDING_BEA_STATE_WAGE_SOURCE_REASON,
    **{
        ("tax_unit_count", geo_level, domain_variable, None): (
            _PENDING_IRS_DETAIL_SOURCE_REASON
        )
        for geo_level in ("national", "state")
        for domain_variable in (
            "investment_interest_expense",
            "long_term_capital_gains",
            "salt_refund_income",
            "short_term_capital_gains",
        )
    },
}

PE_NATIVE_BROAD_SOURCE_BACKED_EXCLUDED_CELL_REASONS = {
    **PE_NATIVE_BROAD_SOURCE_BACKED_EXCLUDED_CELL_REASONS,
    **_PENDING_ARCH_SOURCE_BACKED_CELL_REASONS,
}

PE_NATIVE_BROAD_SOURCE_BACKED_TARGET_CELLS: tuple[PolicyEngineUSTargetCell, ...] = (
    tuple(
        cell
        for cell in PE_NATIVE_BROAD_TARGET_CELLS
        if _target_cell_key(cell)
        not in PE_NATIVE_BROAD_SOURCE_BACKED_EXCLUDED_CELL_REASONS
    )
)

_TARGET_PROFILES: dict[str, tuple[PolicyEngineUSTargetCell, ...]] = {
    "pe_native_broad": PE_NATIVE_BROAD_TARGET_CELLS,
    "pe_native_broad_no_state_aca": PE_NATIVE_BROAD_NO_STATE_ACA_TARGET_CELLS,
    "pe_native_broad_source_backed": PE_NATIVE_BROAD_SOURCE_BACKED_TARGET_CELLS,
}

_TARGET_PROFILE_EXCLUSION_REASONS: dict[
    str,
    dict[PolicyEngineUSTargetCellKey, str],
] = {
    "pe_native_broad": {},
    "pe_native_broad_no_state_aca": {
        cell_key: "State ACA cells are excluded from this profile variant."
        for cell_key in _PE_NATIVE_BROAD_NO_STATE_ACA_EXCLUDED_CELLS
    },
    "pe_native_broad_source_backed": (
        PE_NATIVE_BROAD_SOURCE_BACKED_EXCLUDED_CELL_REASONS
    ),
}


def policyengine_us_target_profile_names() -> tuple[str, ...]:
    return tuple(sorted(_TARGET_PROFILES))


def resolve_policyengine_us_target_profile(
    name: str,
) -> tuple[PolicyEngineUSTargetCell, ...]:
    try:
        return _TARGET_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(policyengine_us_target_profile_names())
        raise ValueError(
            f"Unknown PolicyEngine US target profile '{name}'. Known profiles: {known}"
        ) from exc


def policyengine_us_target_profile_exclusion_reasons(
    name: str,
) -> dict[PolicyEngineUSTargetCellKey, str]:
    if name not in _TARGET_PROFILES:
        known = ", ".join(policyengine_us_target_profile_names())
        raise ValueError(
            f"Unknown PolicyEngine US target profile '{name}'. Known profiles: {known}"
        )
    return dict(_TARGET_PROFILE_EXCLUSION_REASONS.get(name, {}))
