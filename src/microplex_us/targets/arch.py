"""Adapters from Arch target records to core Microplex target specs."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha1
from pathlib import Path
from typing import Any

from microplex.core import EntityType
from microplex.targets import (
    TargetAggregation,
    TargetFilter,
    TargetQuery,
    TargetSet,
    apply_target_query,
    arch_consumer_fact_concept,
    arch_consumer_fact_numeric_value,
    arch_consumer_fact_period,
    arch_consumer_fact_source_record_id,
    load_arch_consumer_fact_jsonl_rows,
)
from microplex.targets import (
    TargetSpec as CanonicalTargetSpec,
)

from microplex_us.geography import (
    US_STATE_ABBR_BY_FIPS,
    normalize_state_legislative_district_id,
)
from microplex_us.microdata_roles import policyengine_us_variable_role
from microplex_us.policyengine.target_profiles import (
    PolicyEngineUSTargetCell,
    resolve_policyengine_us_target_profile,
)

ARCH_SOURCE_ALIASES = {
    "bea": "BEA",
    "bea-nipa": "BEA",
    "bea-regional": "BEA",
    "census-decennial": "CENSUS_DECENNIAL",
    "irs-soi": "IRS_SOI",
    "census-acs": "CENSUS_ACS",
    "census-pep": "CENSUS_PEP",
    "census-stc": "CENSUS_STC",
    "usda-snap": "USDA_SNAP",
    "cms-aca": "CMS_ACA",
    "cms-medicare": "CMS_MEDICARE",
    "cms-medicaid": "CMS_MEDICAID",
    "federal-reserve": "FEDERAL_RESERVE",
    "hhs-acf-liheap": "HHS_ACF_LIHEAP",
    "hhs-acf-tanf": "HHS_ACF_TANF",
}

ARCH_CONSTRAINT_VARIABLE_ALIASES = {
    "eitc_qualifying_children": "eitc_child_count",
    "is_tax_filer": "tax_unit_is_filer",
}

ARCH_POSITIVE_CONSTRAINT_ALIASES = {
    "aca": "aca_ptc",
    "aca_marketplace": "aca_ptc",
    "aca_ptc": "aca_ptc",
    "is_aca_ptc_eligible": "aca_ptc",
    "selected_marketplace_plan_benchmark_ratio": "aca_ptc",
    "total_self_employment_income": "self_employment_income",
    "used_aca_ptc": "aca_ptc",
    "is_medicaid": "medicaid_enrolled",
    "medicaid": "medicaid_enrolled",
    "medicaid_enrolled": "medicaid_enrolled",
    "snap": "snap",
}

ARCH_CONSTRAINT_OPERATOR_ALIASES = {
    "=": "==",
    "eq": "==",
    "<>": "!=",
    "ne": "!=",
    "neq": "!=",
}

ARCH_AMOUNT_VARIABLE_ALIASES = {
    "adjusted_gross_income": "adjusted_gross_income",
    "income_tax_liability": "income_tax",
    "income_tax_before_credits_amount": "income_tax_before_credits",
    "eitc_amount": "eitc",
    "ctc_amount": "non_refundable_ctc",
    "actc_amount": "refundable_ctc",
    "taxable_interest_amount": "taxable_interest_income",
    "tax_exempt_interest_amount": "tax_exempt_interest_income",
    "alimony_received_amount": "alimony_income",
    "personal_dividend_income_amount": "dividend_income",
    "ordinary_dividends_amount": "dividend_income",
    "qualified_dividends_amount": "qualified_dividend_income",
    "long_term_capital_gains_amount": "long_term_capital_gains",
    "short_term_capital_gains_amount": "short_term_capital_gains",
    "wages_salaries_amount": "employment_income",
    "net_capital_gains_amount": "net_capital_gains",
    "taxable_ira_distributions_amount": "taxable_ira_distributions",
    "traditional_ira_contributions": "traditional_ira_contributions",
    "roth_ira_contributions": "roth_ira_contributions",
    "taxable_pension_income_amount": "taxable_pension_income",
    "taxable_social_security_amount": "taxable_social_security",
    "unemployment_insurance_benefits": "unemployment_compensation",
    "unemployment_compensation_amount": "unemployment_compensation",
    "tip_income": "tip_income",
    "rental_income_amount": "rental_income",
    "rental_royalty_income_amount": "rental_income",
    "partnership_scorp_income_amount": "tax_unit_partnership_s_corp_income",
    "schedule_c_income_amount": "self_employment_income",
    "state_local_refunds_amount": "salt_refund_income",
    "qbi_amount": "qualified_business_income_deduction",
    "salt_amount": "salt",
    "limited_state_local_taxes_amount": "salt_deduction",
    "charitable_amount": "charitable_deduction",
    "mortgage_interest_amount": "deductible_mortgage_interest",
    "mortgage_interest_paid_amount": "deductible_mortgage_interest",
    "home_mortgage_personal_seller_amount": "deductible_mortgage_interest",
    "deductible_points_amount": "deductible_mortgage_interest",
    "investment_interest_paid_amount": "investment_interest_expense",
    "interest_paid_deduction_amount": "interest_deduction",
    "medical_amount": "medical_expense_deduction",
    "medical_dental_expense_amount": "medical_expense_deduction",
    "real_estate_taxes_amount": "real_estate_taxes",
    "aca_aptc_amount": "aca_ptc",
    "medicaid_benefits": "medicaid",
    "social_security_benefits": "social_security",
    "social_security_dependents_benefits": "social_security_dependents",
    "social_security_disability_benefits": "social_security_disability",
    "social_security_retirement_benefits": "social_security_retirement",
    "social_security_survivors_benefits": "social_security_survivors",
    "snap_benefits": "snap",
    "state_individual_income_tax_collections": "state_income_tax",
    "ssi_payments": "ssi",
    "ssi_total_payments": "ssi",
    "tanf_cash_assistance": "tanf",
    "medicare_part_b_premiums": "medicare_part_b_premiums",
    "net_worth": "net_worth",
}

ARCH_SELF_DOMAIN_AMOUNT_VARIABLES = frozenset(
    set(ARCH_AMOUNT_VARIABLE_ALIASES.values()) - {"adjusted_gross_income"}
)

ARCH_IRS_SOI_ITEMIZED_DEDUCTION_AMOUNT_VARIABLES = frozenset(
    {
        "medical_amount",
        "medical_dental_expense_amount",
        "real_estate_taxes_amount",
        "salt_amount",
    }
)

ARCH_IRS_SOI_ITEMIZED_DEDUCTION_COUNT_VARIABLES = frozenset(
    {
        "medical_claims",
        "real_estate_taxes_claims",
        "salt_claims",
    }
)

ARCH_IRS_SOI_ITEMIZED_DEDUCTION_TABLE_MARKERS = (
    "itemized",
    "historic table 2",
    "table 2.",
)

ARCH_IRS_SOI_CREDIT_AGI_DOMAIN_VARIABLES = frozenset(
    {
        "actc_amount",
        "actc_claims",
        "ctc_amount",
        "ctc_claims",
    }
)

ARCH_STATE_TO_NATIONAL_ROLLUP_VARIABLES = frozenset(
    {
        "aca_aptc_amount",
        "ctc_amount",
        "ctc_claims",
    }
)

ARCH_NATIONAL_ROLLUP_STATE_FIPS = frozenset(
    state_fips for state_fips in US_STATE_ABBR_BY_FIPS if state_fips != "72"
)

ARCH_POSITIVE_AMOUNT_FILTER_VARIABLES = frozenset(
    {
        # SOI Table 1.4's taxable net capital gains amount is paired with
        # returns with taxable net capital gains; PolicyEngine's variable can be
        # negative, so the amount target must use the same positive domain.
        "net_capital_gains",
    }
)

ARCH_TARGET_CELL_VARIABLE_ALIASES = {
    "income_tax": frozenset({"income_tax_positive"}),
    "self_employment_income": frozenset({"total_self_employment_income"}),
}

ARCH_BROAD_BUSINESS_INCOME_SELF_EMPLOYMENT_BLOCKLIST = frozenset(
    {
        "bea_nipa.proprietors_income_with_inventory_valuation_and_capital_consumption_adjustments",
        "bea_nipa.a041rc_proprietors_income_with_inventory_valuation_and_capital_consumption_adjustments",
        "bea_regional.proprietors_income",
        "bea_regional.sainc5n_line_70_proprietors_income",
        "cbo.income_source:net_business_income",
    }
)

ARCH_COUNT_VARIABLE_ALIASES = {
    "tax_unit_count": ("tax_unit_count", EntityType.TAX_UNIT, None),
    "income_tax_liability_returns": (
        "tax_unit_count",
        EntityType.TAX_UNIT,
        "income_tax",
    ),
    "income_tax_before_credits_returns": (
        "tax_unit_count",
        EntityType.TAX_UNIT,
        "income_tax_before_credits",
    ),
    "household_count": ("household_count", EntityType.HOUSEHOLD, None),
    "population": ("person_count", EntityType.PERSON, None),
    "tax_filer_individual_count": ("person_count", EntityType.PERSON, None),
    "snap_household_count": ("household_count", EntityType.HOUSEHOLD, "snap"),
    "snap_participant_count": ("person_count", EntityType.PERSON, "snap"),
    "aca_marketplace_enrollment": (
        "person_count",
        EntityType.PERSON,
        "aca_ptc",
    ),
    "aca_ptc_returns": ("tax_unit_count", EntityType.TAX_UNIT, "aca_ptc"),
    "medicaid_total_enrollment": (
        "person_count",
        EntityType.PERSON,
        "medicaid_enrolled",
    ),
    "medicaid_enrollment": ("person_count", EntityType.PERSON, "medicaid_enrolled"),
    "liheap_household_count": (
        "household_count",
        EntityType.HOUSEHOLD,
        "spm_unit_energy_subsidy_reported",
    ),
    "tanf_family_count": ("spm_unit_count", EntityType.SPM_UNIT, "tanf"),
    "tanf_recipient_count": ("person_count", EntityType.PERSON, "tanf"),
}

ARCH_FACT_CONCEPT_TO_TARGET = {
    "irs_soi.individual_income_tax_returns": ("tax_unit_count", "COUNT"),
    "irs_soi.returns_with_total_wages": ("wages_salaries_returns", "COUNT"),
    "irs_soi.returns_with_taxable_net_capital_gains": (
        "net_capital_gains_returns",
        "COUNT",
    ),
    "irs_soi.returns_with_taxable_ira_distributions": (
        "taxable_ira_distributions_returns",
        "COUNT",
    ),
    "irs_soi.returns_with_taxable_pension_income": (
        "taxable_pension_income_returns",
        "COUNT",
    ),
    "irs_soi.returns_with_unemployment_compensation": (
        "unemployment_compensation_returns",
        "COUNT",
    ),
    "irs_soi.returns_with_taxable_social_security_benefits": (
        "taxable_social_security_returns",
        "COUNT",
    ),
    "irs_soi.returns_with_income_tax_after_credits": (
        "income_tax_liability_returns",
        "COUNT",
    ),
    "irs_soi.tax_filer_individuals": (
        "tax_filer_individual_count",
        "COUNT",
    ),
    "irs_soi.returns_with_income_tax_before_credits": (
        "income_tax_before_credits_returns",
        "COUNT",
    ),
    "irs_soi.income_tax_before_credits": (
        "income_tax_before_credits_amount",
        "AMOUNT",
    ),
    "irs_soi.income_tax_after_credits": ("income_tax_liability", "AMOUNT"),
    "irs_soi.returns_with_premium_tax_credit": (
        "aca_ptc_returns",
        "COUNT",
    ),
    "irs_soi.premium_tax_credit": ("aca_aptc_amount", "AMOUNT"),
    "irs_soi.returns_with_earned_income_credit": ("eitc_claims", "COUNT"),
    "irs_soi.earned_income_credit": ("eitc_amount", "AMOUNT"),
    "irs_soi.total_earned_income_credit": ("eitc_amount", "AMOUNT"),
    "irs_soi.returns_with_total_earned_income_credit": ("eitc_claims", "COUNT"),
    "irs_soi.returns_with_child_tax_credit": ("ctc_claims", "COUNT"),
    "irs_soi.child_tax_credit": ("ctc_amount", "AMOUNT"),
    "irs_soi.returns_with_additional_child_tax_credit": (
        "actc_claims",
        "COUNT",
    ),
    "irs_soi.additional_child_tax_credit": ("actc_amount", "AMOUNT"),
    "irs_soi.returns_with_real_estate_taxes": (
        "real_estate_taxes_claims",
        "COUNT",
    ),
    "irs_soi.real_estate_taxes": ("real_estate_taxes_amount", "AMOUNT"),
    "irs_soi.returns_with_limited_state_local_taxes": (
        "limited_state_local_taxes_returns",
        "COUNT",
    ),
    "irs_soi.limited_state_local_taxes": (
        "limited_state_local_taxes_amount",
        "AMOUNT",
    ),
    "us:statutes/26/62#adjusted_gross_income": (
        "adjusted_gross_income",
        "AMOUNT",
    ),
    "us:statutes/26/62#input.wages": ("wages_salaries_amount", "AMOUNT"),
    "irs_soi.adjusted_gross_income": ("adjusted_gross_income", "AMOUNT"),
    "irs_soi.total_income_tax": ("income_tax_liability", "AMOUNT"),
    "irs_soi.total_wages": ("wages_salaries_amount", "AMOUNT"),
    "irs_soi.returns_with_ordinary_dividends": (
        "ordinary_dividends_returns",
        "COUNT",
    ),
    "irs_soi.ordinary_dividends": ("ordinary_dividends_amount", "AMOUNT"),
    "irs_soi.returns_with_qualified_dividends": (
        "qualified_dividends_returns",
        "COUNT",
    ),
    "irs_soi.qualified_dividends": ("qualified_dividends_amount", "AMOUNT"),
    "irs_soi.returns_with_qualified_business_income_deduction": (
        "qbi_claims",
        "COUNT",
    ),
    "irs_soi.qualified_business_income_deduction": ("qbi_amount", "AMOUNT"),
    "irs_soi.returns_with_taxable_interest": (
        "taxable_interest_returns",
        "COUNT",
    ),
    "irs_soi.taxable_interest": ("taxable_interest_amount", "AMOUNT"),
    "irs_soi.returns_with_tax_exempt_interest": (
        "tax_exempt_interest_returns",
        "COUNT",
    ),
    "irs_soi.tax_exempt_interest": ("tax_exempt_interest_amount", "AMOUNT"),
    "irs_soi.returns_with_schedule_c_income": (
        "schedule_c_income_returns",
        "COUNT",
    ),
    "irs_soi.schedule_c_income": ("schedule_c_income_amount", "AMOUNT"),
    "irs_soi.taxable_net_capital_gains": ("net_capital_gains_amount", "AMOUNT"),
    "irs_soi.returns_with_partnership_scorp_income": (
        "partnership_scorp_income_returns",
        "COUNT",
    ),
    "irs_soi.partnership_scorp_income": (
        "partnership_scorp_income_amount",
        "AMOUNT",
    ),
    "irs_soi.returns_with_rental_royalty_income": (
        "rental_royalty_income_returns",
        "COUNT",
    ),
    "irs_soi.rental_royalty_income": (
        "rental_royalty_income_amount",
        "AMOUNT",
    ),
    "irs_soi.taxable_ira_distributions": (
        "taxable_ira_distributions_amount",
        "AMOUNT",
    ),
    "irs_soi.taxable_pension_income": ("taxable_pension_income_amount", "AMOUNT"),
    "irs_soi.unemployment_compensation": (
        "unemployment_compensation_amount",
        "AMOUNT",
    ),
    "irs_soi.taxable_social_security_benefits": (
        "taxable_social_security_amount",
        "AMOUNT",
    ),
    "irs_soi.total_itemized_deductions": ("itemized_deductions", "AMOUNT"),
    "irs_soi.returns_with_itemized_deductions": (
        "itemized_deductions_returns",
        "COUNT",
    ),
    "irs_soi.returns_with_medical_dental_expense_deduction": (
        "medical_claims",
        "COUNT",
    ),
    "irs_soi.medical_dental_expense_deduction": (
        "medical_dental_expense_amount",
        "AMOUNT",
    ),
    "irs_soi.standard_deduction": ("standard_deduction", "AMOUNT"),
    "irs_soi.taxable_income": ("taxable_income", "AMOUNT"),
    "irs_soi.total_income": ("total_income", "AMOUNT"),
    "irs_soi.returns_with_total_income": ("total_income_returns", "COUNT"),
    "irs_soi.capital_asset_net_gain_less_loss": (
        "capital_asset_net_gain_less_loss",
        "AMOUNT",
    ),
    "irs_soi.returns_with_capital_asset_net_gain_less_loss": (
        "capital_asset_net_gain_less_loss_returns",
        "COUNT",
    ),
    "irs_soi.tax_credits": ("tax_credits", "AMOUNT"),
    "irs_soi.returns_with_tax_credits": ("tax_credits_returns", "COUNT"),
    "irs_soi.returns_with_taxable_income": ("taxable_income_returns", "COUNT"),
    "irs_soi.returns_with_total_income_tax": (
        "income_tax_liability_returns",
        "COUNT",
    ),
    "irs_soi.individual_income_tax_returns_excluding_dependents": (
        "tax_unit_count",
        "COUNT",
    ),
    "irs_soi.eic_earned_income": ("eic_earned_income", "AMOUNT"),
    "irs_soi.returns_with_eic_earned_income": (
        "eic_earned_income_returns",
        "COUNT",
    ),
    "irs_soi.eic_refundable_portion": ("eitc_refundable_portion", "AMOUNT"),
    "irs_soi.returns_with_eic_refundable_portion": (
        "eitc_refundable_portion_returns",
        "COUNT",
    ),
    "irs_soi.roth_ira_contributions": ("roth_ira_contributions", "AMOUNT"),
    "irs_soi.roth_ira_contributors": ("roth_ira_contributors", "COUNT"),
    "irs_soi.traditional_ira_contributions": (
        "traditional_ira_contributions",
        "AMOUNT",
    ),
    "irs_soi.traditional_ira_contributors": (
        "traditional_ira_contributors",
        "COUNT",
    ),
    "irs_soi.form_w2_social_security_tip_income": ("tip_income", "AMOUNT"),
    "irs_soi.form_w2_social_security_tip_returns": (
        "tip_income_returns",
        "COUNT",
    ),
    "irs_soi.form_w2_social_security_tip_taxpayers": (
        "tip_income_taxpayers",
        "COUNT",
    ),
    "irs_soi.form_w2_401k_elective_deferrals": (
        "traditional_401k_contributions",
        "AMOUNT",
    ),
    "irs_soi.form_w2_designated_roth_401k_contributions": (
        "roth_401k_contributions",
        "AMOUNT",
    ),
    "irs_soi.payments_to_keogh_plan": (
        "self_employed_pension_contribution_ald",
        "AMOUNT",
    ),
    "federal_reserve.z1.households_nonprofits_net_worth": (
        "net_worth",
        "AMOUNT",
    ),
    "cms_medicare.part_b_premium_income": (
        "medicare_part_b_premiums",
        "AMOUNT",
    ),
    "census_decennial.resident_population": ("population", "COUNT"),
    "census_decennial.occupied_housing_units": ("household_count", "COUNT"),
    "census_pep.resident_population": ("population", "COUNT"),
    "census_stc.individual_income_tax_collections": (
        "state_individual_income_tax_collections",
        "AMOUNT",
    ),
    "cms_aca.marketplace_effectuated_enrollment": (
        "aca_marketplace_enrollment",
        "COUNT",
    ),
    "cms_aca.marketplace_plan_selections": (
        "aca_marketplace_plan_selections",
        "COUNT",
    ),
    "cms_aca.aptc_consumers": ("aca_aptc_consumers", "COUNT"),
    "cms_aca.average_monthly_aptc": ("aca_average_monthly_aptc", "RATE"),
    "cms_medicaid.total_medicaid_enrollment": (
        "medicaid_total_enrollment",
        "COUNT",
    ),
    "cms_medicaid.total_medicaid_chip_enrollment": (
        "medicaid_chip_total_enrollment",
        "COUNT",
    ),
    "cms_medicaid.total_chip_enrollment": ("chip_total_enrollment", "COUNT"),
    "cms_medicaid.medicaid_chip_child_enrollment": (
        "medicaid_chip_child_enrollment",
        "COUNT",
    ),
    "cms_medicaid.total_adult_medicaid_enrollment": (
        "adult_medicaid_enrollment",
        "COUNT",
    ),
    "cms_nhe.medicaid_title_xix_expenditures": (
        "medicaid_benefits",
        "AMOUNT",
    ),
    "hhs_acf_tanf.cash_assistance_expenditures": (
        "tanf_cash_assistance",
        "AMOUNT",
    ),
    "hhs_acf_tanf.average_monthly_tanf_total_recipients": (
        "tanf_recipient_count",
        "COUNT",
    ),
    "hhs_acf_tanf.average_monthly_tanf_adult_recipients": (
        "tanf_adult_recipient_count",
        "COUNT",
    ),
    "hhs_acf_tanf.average_monthly_tanf_child_recipients": (
        "tanf_child_recipient_count",
        "COUNT",
    ),
    "hhs_acf_tanf.average_monthly_tanf_total_families": (
        "tanf_family_count",
        "COUNT",
    ),
    "hhs_acf_tanf.average_monthly_tanf_one_parent_families": (
        "tanf_one_parent_family_count",
        "COUNT",
    ),
    "hhs_acf_tanf.average_monthly_tanf_two_parent_families": (
        "tanf_two_parent_family_count",
        "COUNT",
    ),
    "hhs_acf_tanf.average_monthly_tanf_no_parent_families": (
        "tanf_no_parent_family_count",
        "COUNT",
    ),
    "hhs_acf_liheap.households_served_by_state_programs": (
        "liheap_household_count",
        "COUNT",
    ),
    "bea_nipa.wages_and_salaries": ("wages_salaries_amount", "AMOUNT"),
    "bea_nipa.proprietors_income_with_inventory_valuation_and_capital_consumption_adjustments": (
        "proprietors_income_amount",
        "AMOUNT",
    ),
    "bea_nipa.rental_income_of_persons_with_capital_consumption_adjustment": (
        "rental_income_amount",
        "AMOUNT",
    ),
    "bea_nipa.personal_interest_income": (
        "personal_interest_income_amount",
        "RATE",
    ),
    "bea_nipa.personal_dividend_income": (
        "personal_dividend_income_amount",
        "AMOUNT",
    ),
    "bea_nipa.supplements_to_wages_and_salaries": (
        "supplements_to_wages_and_salaries",
        "RATE",
    ),
    "bea_nipa.employer_contributions_for_employee_pension_and_insurance_funds": (
        "employer_pension_and_insurance_contributions",
        "RATE",
    ),
    "bea_nipa.employer_contributions_for_government_social_insurance": (
        "employer_government_social_insurance_contributions",
        "RATE",
    ),
    "bea_nipa.farm_proprietors_income": ("farm_proprietors_income", "RATE"),
    "bea_nipa.nonfarm_proprietors_income": ("nonfarm_proprietors_income", "RATE"),
    "bea_nipa.government_social_benefits_to_persons": (
        "government_social_benefits_to_persons",
        "RATE",
    ),
    "bea_nipa.social_security_benefits": ("social_security_benefits", "AMOUNT"),
    "bea_nipa.medicare_benefits": ("medicare_benefits", "RATE"),
    "bea_nipa.medicaid_benefits": ("medicaid_benefits", "AMOUNT"),
    "bea_nipa.unemployment_insurance_benefits": (
        "unemployment_insurance_benefits",
        "AMOUNT",
    ),
    "bea_nipa.veterans_benefits": ("veterans_benefits", "RATE"),
    "bea_nipa.other_government_social_benefits_to_persons": (
        "other_government_social_benefits_to_persons",
        "RATE",
    ),
    "bea_nipa.other_current_transfer_receipts_from_business_net": (
        "other_current_transfer_receipts_from_business_net",
        "RATE",
    ),
    "bea_nipa.personal_current_transfer_receipts": (
        "personal_current_transfer_receipts",
        "RATE",
    ),
    "bea_nipa.personal_income": ("personal_income", "RATE"),
    "bea_nipa.personal_current_taxes": ("personal_current_taxes", "RATE"),
    "bea_nipa.disposable_personal_income": ("disposable_personal_income", "RATE"),
    "bea_nipa.personal_outlays": ("personal_outlays", "RATE"),
    "bea_nipa.personal_saving": ("personal_saving", "RATE"),
    "bea_nipa.personal_saving_rate": ("personal_saving_rate", "RATE"),
    "bea_regional.personal_income": ("regional_personal_income", "RATE"),
    "bea_regional.dividends_interest_and_rent": (
        "regional_dividends_interest_and_rent",
        "RATE",
    ),
    "bea_regional.personal_current_transfer_receipts": (
        "regional_personal_current_transfer_receipts",
        "RATE",
    ),
    "bea_regional.wages_and_salaries": ("wages_salaries_amount", "AMOUNT"),
    "bea_regional.supplements_to_wages_and_salaries": (
        "regional_supplements_to_wages_and_salaries",
        "RATE",
    ),
    "bea_regional.proprietors_income": ("proprietors_income_amount", "AMOUNT"),
    "usda_snap.total_benefits": ("snap_benefits", "AMOUNT"),
    "usda_snap.average_monthly_households": ("snap_household_count", "COUNT"),
    "usda_snap.average_monthly_persons": ("snap_participant_count", "COUNT"),
    "usda_snap.average_monthly_benefit_per_person": (
        "snap_average_monthly_benefit_per_person",
        "RATE",
    ),
}

ARCH_FACT_DOMAIN_CONSTRAINTS = {
    "all_individual_income_tax_returns": (("is_tax_filer", "==", "1"),),
    "form_w2_items": (),
    "household_balance_sheet": (),
    "individual_income_tax_returns": (("is_tax_filer", "==", "1"),),
    "individual_income_tax_returns_excluding_dependents": (
        ("is_dependent", "==", "0"),
    ),
    "individual_income_tax_returns_with_earned_income_credit": (("eitc", ">", "0"),),
    "individual_income_tax_returns_with_itemized_deductions": (
        ("itemized_deductions", ">", "0"),
    ),
    "individual_retirement_arrangement_contributions": (),
    "compensation_of_employees": (),
    "households": (),
    "aca_marketplace_effectuated_enrollment": (),
    "aca_marketplace_qhp_selections": (),
    "medicaid_chip_enrollment": (),
    "medicare_financing": (),
    "national_health_expenditures": (),
    "personal_current_transfer_receipts": (),
    "personal_income": (),
    "resident_population": (),
    "social_security_and_ssi_payments": (),
    "state_government_tax_collections": (),
    "supplemental_nutrition_assistance_program": (("snap", "==", "1"),),
    "tanf_cash_assistance": (),
    "tanf_caseload": (),
    "liheap_state_programs": (),
}

ARCH_FACT_CONSTRAINT_VARIABLE_ALIASES = {
    "age": "age",
    "us.tax.earned_income_credit_qualifying_children": "eitc_child_count",
    "us_social_security_and_ssi.program_payment_type": "program_payment_type",
    "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income",
    "irs_soi.adjusted_gross_income": "adjusted_gross_income",
}

ARCH_IGNORED_FACT_CONSTRAINT_VARIABLES = frozenset(
    {
        "administering_entity",
        "amount_basis",
        "bea_nipa.series_code",
        "bea_regional.geo_name",
        "bea_regional.line_code",
        "bea_regional.table_name",
        "medicare.financing_component",
        "medicare.part",
        "program",
    }
)

ARCH_ENTITY_HINTS = {
    "adjusted_gross_income": EntityType.TAX_UNIT,
    "income_tax": EntityType.TAX_UNIT,
    "income_tax_positive": EntityType.TAX_UNIT,
    "income_tax_before_credits": EntityType.TAX_UNIT,
    "eitc": EntityType.TAX_UNIT,
    "non_refundable_ctc": EntityType.TAX_UNIT,
    "refundable_ctc": EntityType.TAX_UNIT,
    "qualified_business_income_deduction": EntityType.TAX_UNIT,
    "salt": EntityType.TAX_UNIT,
    "salt_deduction": EntityType.TAX_UNIT,
    "charitable_deduction": EntityType.TAX_UNIT,
    "deductible_mortgage_interest": EntityType.TAX_UNIT,
    "interest_deduction": EntityType.TAX_UNIT,
    "investment_interest_expense": EntityType.PERSON,
    "medical_expense_deduction": EntityType.TAX_UNIT,
    "real_estate_taxes": EntityType.TAX_UNIT,
    "tax_unit_partnership_s_corp_income": EntityType.TAX_UNIT,
    "dividend_income": EntityType.PERSON,
    "employment_income": EntityType.PERSON,
    "qualified_dividend_income": EntityType.PERSON,
    "taxable_interest_income": EntityType.PERSON,
    "tax_exempt_interest_income": EntityType.PERSON,
    "long_term_capital_gains": EntityType.PERSON,
    "short_term_capital_gains": EntityType.PERSON,
    "proprietors_income_amount": EntityType.PERSON,
    "rental_income": EntityType.PERSON,
    "roth_401k_contributions": EntityType.PERSON,
    "self_employment_income": EntityType.PERSON,
    "self_employed_pension_contribution_ald": EntityType.TAX_UNIT,
    "salt_refund_income": EntityType.PERSON,
    "state_income_tax": EntityType.TAX_UNIT,
    "taxable_ira_distributions": EntityType.PERSON,
    "traditional_ira_contributions": EntityType.PERSON,
    "roth_ira_contributions": EntityType.PERSON,
    "taxable_pension_income": EntityType.PERSON,
    "taxable_social_security": EntityType.PERSON,
    "tip_income": EntityType.PERSON,
    "traditional_401k_contributions": EntityType.PERSON,
    "unemployment_compensation": EntityType.PERSON,
    "medicare_part_b_premiums": EntityType.PERSON,
    "medicaid": EntityType.PERSON,
    "net_worth": EntityType.HOUSEHOLD,
    "social_security": EntityType.PERSON,
    "social_security_dependents": EntityType.PERSON,
    "social_security_disability": EntityType.PERSON,
    "social_security_retirement": EntityType.PERSON,
    "social_security_survivors": EntityType.PERSON,
    "snap": EntityType.HOUSEHOLD,
    "ssi": EntityType.PERSON,
    "tanf": EntityType.SPM_UNIT,
}

ARCH_AGI_BRACKET_FILTERS = {
    "under_1": (None, 1),
    "1_to_10k": (1, 10_000),
    "10k_to_25k": (10_000, 25_000),
    "25k_to_50k": (25_000, 50_000),
    "50k_to_75k": (50_000, 75_000),
    "75k_to_100k": (75_000, 100_000),
    "100k_to_200k": (100_000, 200_000),
    "200k_to_500k": (200_000, 500_000),
    "500k_to_1m": (500_000, 1_000_000),
    "1m_plus": (1_000_000, None),
}

ARCH_CURRENT_TAX_VARIABLES = frozenset(
    {
        "tax_unit_count",
        "adjusted_gross_income",
        "income_tax_liability",
    }
)

ARCH_LABEL_WORD_OVERRIDES = {
    "aca": "ACA",
    "actc": "ACTC",
    "agi": "AGI",
    "bls": "BLS",
    "cbo": "CBO",
    "cms": "CMS",
    "ctc": "CTC",
    "eitc": "EITC",
    "irs": "IRS",
    "qbi": "QBI",
    "liheap": "LIHEAP",
    "snap": "SNAP",
    "soi": "SOI",
    "ssi": "SSI",
    "tanf": "TANF",
    "usda": "USDA",
}

ARCH_VARIABLE_LABEL_OVERRIDES = {
    "adjusted_gross_income": "Adjusted gross income",
    "income_tax_liability": "Income tax liability",
    "income_tax_liability_returns": "Returns with income tax after credits",
    "income_tax_before_credits_returns": ("Returns with income tax before credits"),
    "income_tax_before_credits_amount": "Income tax before credits amount",
    "tax_filer_individual_count": "Individuals on tax returns",
    "aca_ptc_returns": "Returns with premium tax credit",
    "aca_aptc_amount": "Premium tax credit amount",
    "eitc_claims": "Returns with earned income credit",
    "eitc_amount": "Earned income credit amount",
    "real_estate_taxes_claims": "Returns with real estate taxes",
    "real_estate_taxes_amount": "Real estate taxes amount",
    "limited_state_local_taxes_returns": ("Returns with limited state and local taxes"),
    "tax_exempt_interest_returns": "Tax-exempt interest returns",
    "tax_exempt_interest_amount": "Tax-exempt interest amount",
    "taxable_interest_amount": "Taxable interest amount",
    "wages_salaries_returns": "Returns with total wages",
    "wages_salaries_amount": "Total wages amount",
    "personal_dividend_income_amount": "Personal dividend income amount",
    "proprietors_income_amount": "Proprietors' income amount",
    "rental_income_amount": "Rental income amount",
    "net_capital_gains_returns": "Returns with taxable net capital gains",
    "net_capital_gains_amount": "Taxable net capital gains amount",
    "taxable_ira_distributions_returns": ("Returns with taxable IRA distributions"),
    "taxable_ira_distributions_amount": "Taxable IRA distributions amount",
    "taxable_pension_income_returns": "Returns with taxable pension income",
    "taxable_pension_income_amount": "Taxable pension income amount",
    "unemployment_compensation_returns": ("Returns with unemployment compensation"),
    "unemployment_compensation_amount": "Unemployment compensation amount",
    "unemployment_insurance_benefits": "Unemployment insurance benefits",
    "taxable_social_security_returns": (
        "Returns with taxable Social Security benefits"
    ),
    "taxable_social_security_amount": "Taxable Social Security benefits amount",
    "ordinary_dividends_amount": "Ordinary dividends amount",
    "qualified_dividends_returns": "Returns with qualified dividends",
    "qualified_dividends_amount": "Qualified dividends amount",
    "long_term_capital_gains_amount": "Long-term capital gains amount",
    "short_term_capital_gains_amount": "Short-term capital gains amount",
    "partnership_scorp_income_returns": "Returns with partnership and S-corp income",
    "partnership_scorp_income_amount": "Partnership and S-corp income amount",
    "schedule_c_income_returns": "Returns with Schedule C income",
    "schedule_c_income_amount": "Schedule C income amount",
    "medical_claims": "Returns with medical expense deduction",
    "medical_dental_expense_amount": "Medical and dental expense amount",
    "tax_unit_count": "Tax unit count",
    "household_count": "Household count",
    "population": "Population count",
    "snap_household_count": "SNAP household count",
    "snap_participant_count": "SNAP participant count",
    "aca_marketplace_enrollment": "ACA marketplace enrollment",
    "state_individual_income_tax_collections": (
        "State individual income tax collections"
    ),
    "limited_state_local_taxes_amount": "Limited state and local taxes amount",
    "interest_paid_deduction_amount": "Interest paid deduction amount",
    "mortgage_interest_paid_amount": "Mortgage interest paid amount",
    "home_mortgage_personal_seller_amount": (
        "Home mortgage from personal seller amount"
    ),
    "deductible_points_amount": "Deductible points amount",
    "investment_interest_paid_amount": "Investment interest paid amount",
    "medicaid_benefits": "Medicaid benefits",
    "medicaid_total_enrollment": "Medicaid enrollment",
    "medicaid_enrollment": "Medicaid enrollment",
    "liheap_household_count": "LIHEAP household count",
    "social_security_benefits": "Social Security benefits",
    "social_security_dependents_benefits": "Social Security dependent benefits",
    "social_security_disability_benefits": "Social Security disability benefits",
    "social_security_retirement_benefits": "Social Security retirement benefits",
    "social_security_survivors_benefits": "Social Security survivor benefits",
    "ssi_payments": "SSI payments",
    "tanf_cash_assistance": "TANF cash assistance",
    "tanf_family_count": "TANF family count",
    "tanf_recipient_count": "TANF recipient count",
    "tip_income": "Tip income",
    "traditional_401k_contributions": "Traditional 401(k) contributions",
    "traditional_ira_contributions": "Traditional IRA contributions",
    "roth_401k_contributions": "Roth 401(k) contributions",
    "roth_ira_contributions": "Roth IRA contributions",
    "self_employed_pension_contribution_ald": (
        "Self-employed pension contribution ALD"
    ),
}

ARCH_AGI_BRACKET_LABELS = {
    "under_1": "under $1",
    "1_to_10k": "$1-$10k",
    "10k_to_25k": "$10k-$25k",
    "25k_to_50k": "$25k-$50k",
    "50k_to_75k": "$50k-$75k",
    "75k_to_100k": "$75k-$100k",
    "100k_to_200k": "$100k-$200k",
    "200k_to_500k": "$200k-$500k",
    "500k_to_1m": "$500k-$1m",
    "1m_plus": "$1m+",
}

ARCH_MODEL_AMOUNT_VARIABLE_HINTS = {
    **{
        model_variable: source_variable
        for source_variable, model_variable in ARCH_AMOUNT_VARIABLE_ALIASES.items()
    },
    "employment_income": "wages_salaries_amount",
    "income_tax_positive": "income_tax_liability",
    "income_tax_before_credits": "income_tax_before_credits_amount",
    "interest_deduction": "interest_paid_deduction_amount",
    "medicare_part_b_premiums": "medicare_part_b_premiums",
    "net_capital_gains": "net_capital_gains_amount",
    "net_worth": "net_worth",
    "real_estate_taxes": "real_estate_taxes_amount",
    "roth_401k_contributions": "roth_401k_contributions",
    "self_employed_pension_contribution_ald": (
        "self_employed_pension_contribution_ald"
    ),
    "total_self_employment_income": "schedule_c_income_amount",
    "taxable_ira_distributions": "taxable_ira_distributions_amount",
    "taxable_pension_income": "taxable_pension_income_amount",
    "taxable_social_security": "taxable_social_security_amount",
    "tip_income": "tip_income",
    "traditional_401k_contributions": "traditional_401k_contributions",
    "unemployment_compensation": "unemployment_compensation_amount",
}

ARCH_MODEL_COUNT_DOMAIN_VARIABLE_HINTS = {
    "adjusted_gross_income": "tax_unit_count",
    "dividend_income": "ordinary_dividends_returns",
    "employment_income": "wages_salaries_returns",
    "eitc": "eitc_claims",
    "income_tax": "income_tax_liability_returns",
    "income_tax_before_credits": "income_tax_before_credits_returns",
    "medical_expense_deduction": "medical_claims",
    "net_capital_gains": "net_capital_gains_returns",
    "non_refundable_ctc": "ctc_claims",
    "qualified_business_income_deduction": "qbi_claims",
    "qualified_dividend_income": "qualified_dividends_returns",
    "real_estate_taxes": "real_estate_taxes_claims",
    "refundable_ctc": "actc_claims",
    "rental_income": "rental_royalty_income_returns",
    "salt": "salt_claims",
    "self_employment_income": "schedule_c_income_returns",
    "total_self_employment_income": "schedule_c_income_returns",
    "tax_exempt_interest_income": "tax_exempt_interest_returns",
    "tax_unit_partnership_s_corp_income": "partnership_scorp_income_returns",
    "taxable_interest_income": "taxable_interest_returns",
    "taxable_ira_distributions": "taxable_ira_distributions_returns",
    "taxable_pension_income": "taxable_pension_income_returns",
    "taxable_social_security": "taxable_social_security_returns",
    "unemployment_compensation": "unemployment_compensation_returns",
}

ARCH_BEA_FULL_POP_AMOUNT_VARIABLES = frozenset(
    {
        "dividend_income",
        "employment_income",
        "rental_income",
        "unemployment_compensation",
    }
)

ARCH_BEA_FULL_POP_AMOUNT_ARCH_VARIABLES = {
    "dividend_income": "personal_dividend_income_amount",
    "employment_income": "wages_salaries_amount",
    "rental_income": "rental_income_amount",
    "unemployment_compensation": "unemployment_insurance_benefits",
}

ARCH_IRS_SOI_GAP_VARIABLES = frozenset(
    {
        *ARCH_MODEL_AMOUNT_VARIABLE_HINTS,
        *ARCH_MODEL_COUNT_DOMAIN_VARIABLE_HINTS,
        "income_tax_positive",
        "interest_deduction",
        "roth_ira_contributions",
        "tax_unit_count",
        "tip_income",
        "traditional_ira_contributions",
    }
)

ARCH_DEPRIORITIZED_SURVEY_OR_MODEL_GAP_VARIABLES = frozenset(
    {
        "alimony_expense",
        "child_support_expense",
        "child_support_received",
        "health_insurance_premiums_without_medicare_part_b",
        "other_medical_expenses",
        "over_the_counter_health_expenses",
        "rent",
        "spm_unit_capped_housing_subsidy",
        "spm_unit_capped_work_childcare_expenses",
    }
)

ARCH_DEPRIORITIZED_SURVEY_OR_MODEL_GAP_DOMAINS = frozenset(
    {
        "ssn_card_type",
    }
)

ARCH_GAP_SOURCE_TABLE_HINTS = {
    "aca_aptc_amount": "CMS Marketplace Open Enrollment public-use files",
    "aca_marketplace_enrollment": "CMS Marketplace Open Enrollment public-use files",
    "employment_income": "IRS SOI Publication 1304 Table 1.4",
    "aca_ptc_returns": "IRS SOI Historic Table 2",
    "eitc_amount": "IRS SOI Historic Table 2",
    "eitc_claims": "IRS SOI Historic Table 2",
    "income_tax_liability": "IRS SOI Publication 1304 Table 1.1 or Historic Table 2",
    "income_tax_before_credits": "IRS SOI Publication 1304 Table 1.1",
    "income_tax_before_credits_returns": "IRS SOI Historic Table 2",
    "tax_filer_individual_count": "IRS SOI Historic Table 2",
    "interest_paid_deduction_amount": "IRS SOI Historic Table 2",
    "limited_state_local_taxes_amount": "IRS SOI Historic Table 2",
    "liheap_household_count": "HHS ACF LIHEAP National Profile",
    "medicaid_benefits": (
        "CMS National Health Expenditures by type of service and source of funds"
    ),
    "net_capital_gains": "IRS SOI Publication 1304 Table 1.4",
    "population": "Census Population Estimates Program Vintage 2024 age-sex files",
    "real_estate_taxes": "IRS SOI itemized deduction tables or ACS state files",
    "roth_ira_contributions": "IRS SOI IRA contribution tables",
    "roth_401k_contributions": "IRS SOI Form W-2 Statistics Table 4.B",
    "self_employed_pension_contribution_ald": "IRS SOI Publication 1304 Table 1.4",
    "state_individual_income_tax_collections": (
        "Census State Tax Collections item T40"
    ),
    "social_security_benefits": "SSA Annual Statistical Supplement",
    "social_security_dependents_benefits": "SSA Annual Statistical Supplement",
    "social_security_disability_benefits": "SSA Annual Statistical Supplement",
    "social_security_retirement_benefits": "SSA Annual Statistical Supplement",
    "social_security_survivors_benefits": "SSA Annual Statistical Supplement",
    "snap_benefits": "USDA FNS SNAP annual state participation and benefit workbooks",
    "snap_household_count": (
        "USDA FNS SNAP annual state participation and benefit workbooks"
    ),
    "snap_participant_count": (
        "USDA FNS SNAP annual state participation and benefit workbooks"
    ),
    "ssi_payments": "SSA Annual Statistical Supplement",
    "tanf_cash_assistance": "ACF TANF Financial Data",
    "tanf_family_count": "ACF TANF Caseload Data",
    "tanf_recipient_count": "ACF TANF Caseload Data",
    "tip_income": "IRS SOI Form W-2 Statistics",
    "traditional_ira_contributions": "IRS SOI IRA contribution tables",
    "traditional_401k_contributions": "IRS SOI Form W-2 Statistics Table 4.B",
    "taxable_ira_distributions": "IRS SOI IRA accumulation/distribution tables",
    "taxable_pension_income": "IRS SOI Publication 1304 Table 1.4",
    "taxable_social_security": "IRS SOI Publication 1304 Table 1.4",
    "unemployment_compensation": "IRS SOI Publication 1304 Table 1.4",
}


@dataclass(frozen=True)
class SOIAgingFactors:
    """Declared factors used to age SOI target records to a model year."""

    source_year: int
    target_year: int
    count_factor: float
    amount_factor: float
    count_method: str
    amount_method: str


@dataclass(frozen=True)
class ArchTargetRecord:
    """A source target record loaded from the Arch SQLite DB."""

    target_id: int
    stratum_id: int
    variable: str
    period: int
    value: float
    target_type: str
    geographic_level: str | None
    geography_id: str | None
    source: str
    source_table: str | None
    source_url: str | None
    notes: str | None
    stratum_name: str | None
    jurisdiction: str
    constraints: tuple[tuple[str, str, str], ...]
    source_period: int | None = None
    aging_factors: SOIAgingFactors | None = None
    aggregate_fact_key: str | None = None
    semantic_fact_key: str | None = None
    source_record_id: str | None = None
    source_cell_keys: tuple[str, ...] = ()
    source_row_keys: tuple[str, ...] = ()
    unit: str | None = None
    concept: str | None = None
    source_concept: str | None = None
    concept_relation: str | None = None
    concept_authority: str | None = None
    concept_evidence_url: str | None = None
    concept_evidence_notes: str | None = None
    legal_vintage: str | None = None
    source_db_path: str | None = None
    source_db_index: int | None = None
    source_target_id: int | None = None
    source_stratum_id: int | None = None


@dataclass(frozen=True)
class ArchTargetCellCoverage:
    """Coverage for one PolicyEngine target cell from an Arch target DB."""

    cell: dict[str, str | None]
    target_ids: tuple[int, ...]
    target_names: tuple[str, ...]
    sources: tuple[str, ...]

    @property
    def covered(self) -> bool:
        return bool(self.target_ids)

    @property
    def target_count(self) -> int:
        return len(self.target_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": dict(self.cell),
            "covered": self.covered,
            "target_count": self.target_count,
            "target_ids": list(self.target_ids),
            "target_names": list(self.target_names),
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class ArchTargetProfileCoverageReport:
    """JSON-ready summary of Arch coverage for a Microplex target profile."""

    profile_name: str
    period: int
    target_cell_count: int
    covered_cell_count: int
    uncovered_cell_count: int
    coverage_rate: float
    by_geo_level: dict[str, dict[str, int]]
    by_variable: dict[str, dict[str, int]]
    cells: tuple[ArchTargetCellCoverage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "period": self.period,
            "target_cell_count": self.target_cell_count,
            "covered_cell_count": self.covered_cell_count,
            "uncovered_cell_count": self.uncovered_cell_count,
            "coverage_rate": self.coverage_rate,
            "by_geo_level": self.by_geo_level,
            "by_variable": self.by_variable,
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True)
class ArchTargetGapQueueRow:
    """One target-profile cell as an Arch authoring task."""

    priority: int
    profile_name: str
    period: int
    variable: str
    geo_level: str | None
    domain_variable: str | None
    geographic_id: str | None
    covered: bool
    target_count: int
    target_ids: tuple[int, ...]
    sources: tuple[str, ...]
    expected_source: str | None
    expected_source_table: str | None
    expected_arch_variable: str | None
    expected_target_type: str | None
    expected_entity: str | None
    expected_aggregation: str | None
    expected_filters: tuple[dict[str, Any], ...]
    gap_category: str
    loader_status: str
    agent_task_kind: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority,
            "profile_name": self.profile_name,
            "period": self.period,
            "cell": {
                "variable": self.variable,
                "geo_level": self.geo_level,
                "domain_variable": self.domain_variable,
                "geographic_id": self.geographic_id,
            },
            "covered": self.covered,
            "target_count": self.target_count,
            "target_ids": list(self.target_ids),
            "sources": list(self.sources),
            "expected_source": self.expected_source,
            "expected_source_table": self.expected_source_table,
            "expected_arch_variable": self.expected_arch_variable,
            "expected_target_type": self.expected_target_type,
            "expected_entity": self.expected_entity,
            "expected_aggregation": self.expected_aggregation,
            "expected_filters": list(self.expected_filters),
            "gap_category": self.gap_category,
            "loader_status": self.loader_status,
            "agent_task_kind": self.agent_task_kind,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ArchTargetGapQueueReport:
    """JSON-ready Arch authoring queue for a Microplex target profile."""

    profile_name: str
    period: int
    row_count: int
    covered_row_count: int
    uncovered_row_count: int
    by_loader_status: dict[str, int]
    by_gap_category: dict[str, int]
    rows: tuple[ArchTargetGapQueueRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "period": self.period,
            "row_count": self.row_count,
            "covered_row_count": self.covered_row_count,
            "uncovered_row_count": self.uncovered_row_count,
            "by_loader_status": self.by_loader_status,
            "by_gap_category": self.by_gap_category,
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class ArchTargetParityRow:
    """One canonical target identity compared across two Arch artifacts."""

    status: str
    identity: tuple[Any, ...]
    incumbent_targets: tuple[CanonicalTargetSpec, ...]
    candidate_targets: tuple[CanonicalTargetSpec, ...]
    absolute_delta: float | None
    relative_delta: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "identity": _arch_target_parity_identity_dict(self.identity),
            "incumbent_target_count": len(self.incumbent_targets),
            "candidate_target_count": len(self.candidate_targets),
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "incumbent_targets": [
                _target_parity_sample(target) for target in self.incumbent_targets
            ],
            "candidate_targets": [
                _target_parity_sample(target) for target in self.candidate_targets
            ],
        }


@dataclass(frozen=True)
class ArchTargetParityReport:
    """JSON-ready parity report between incumbent and candidate Arch artifacts."""

    period: int
    incumbent_artifacts: tuple[str, ...]
    candidate_artifacts: tuple[str, ...]
    value_abs_tolerance: float
    value_rel_tolerance: float
    counts: dict[str, int]
    rows: tuple[ArchTargetParityRow, ...]
    errors: tuple[dict[str, Any], ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self, *, row_limit: int | None = None) -> dict[str, Any]:
        rows = self.rows if row_limit is None else self.rows[: max(0, row_limit)]
        return {
            "valid": self.valid,
            "period": self.period,
            "incumbent_artifacts": list(self.incumbent_artifacts),
            "candidate_artifacts": list(self.candidate_artifacts),
            "value_abs_tolerance": self.value_abs_tolerance,
            "value_rel_tolerance": self.value_rel_tolerance,
            "counts": self.counts,
            "row_count": len(self.rows),
            "rows": [row.to_dict() for row in rows],
            "errors": list(self.errors),
        }


class ArchSQLiteTargetProvider:
    """Read Arch target records from the Arch SQLite DB."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        jurisdiction: str = "us",
        compose_model_year_targets: bool = True,
        age_soi_targets: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.jurisdiction = jurisdiction
        self.compose_model_year_targets = compose_model_year_targets
        self.age_soi_targets = age_soi_targets

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        """Load canonical targets through the core provider protocol."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Arch targets DB not found: {self.db_path}")

        query = query or TargetQuery()
        provider_filters = dict(query.provider_filters)
        period = query.period if isinstance(query.period, int) else None
        jurisdiction = str(provider_filters.get("jurisdiction") or self.jurisdiction)
        variables = _as_string_tuple(provider_filters.get("variables"))
        domain_variables = _as_string_tuple(provider_filters.get("domain_variables"))
        sources = _as_string_tuple(provider_filters.get("sources"))
        geo_levels = _as_string_tuple(provider_filters.get("geo_levels"))
        target_cells = _as_target_cell_filters(provider_filters.get("target_cells"))
        compose_model_year_targets = bool(
            provider_filters.get(
                "compose_model_year_targets",
                self.compose_model_year_targets,
            )
        )
        age_soi_targets = bool(
            provider_filters.get("age_soi_targets", self.age_soi_targets)
        )
        entity_overrides = provider_filters.get("entity_overrides") or {}

        records = (
            self._compose_model_year_records(
                target_year=period,
                jurisdiction=jurisdiction,
                sources=sources,
                age_soi_targets=age_soi_targets,
            )
            if compose_model_year_targets and period is not None
            else self.load_records(
                period=period,
                jurisdiction=jurisdiction,
                sources=sources,
            )
        )
        canonical_targets = TargetSet(
            [
                target
                for record in records
                if _matches_arch_provider_filters(
                    record,
                    variables=variables,
                    domain_variables=domain_variables,
                    geo_levels=geo_levels,
                    target_cells=target_cells,
                    entity_overrides=entity_overrides,
                )
                for target in [
                    arch_target_record_to_canonical_spec(
                        record,
                        entity_overrides=entity_overrides,
                    )
                ]
                if target is not None
            ]
        )
        return apply_target_query(
            canonical_targets,
            TargetQuery(
                period=query.period,
                entity=query.entity,
                names=query.names,
                metadata_filters=query.metadata_filters,
            ),
        )

    def load_records(
        self,
        *,
        period: int | None = None,
        jurisdiction: str | None = None,
        sources: tuple[str, ...] = (),
    ) -> list[ArchTargetRecord]:
        """Load source target records with attached stratum constraints."""
        jurisdiction = jurisdiction or self.jurisdiction
        normalized_sources = tuple(_normalize_arch_source(source) for source in sources)
        clauses = [_jurisdiction_clause(jurisdiction)]
        params: list[Any] = []
        if period is not None:
            clauses.append("t.period = ?")
            params.append(int(period))
        if normalized_sources:
            placeholders = ", ".join("?" for _ in normalized_sources)
            clauses.append(f"t.source IN ({placeholders})")
            params.extend(normalized_sources)
        where_clause = " AND ".join(clauses)
        sql = f"""
            SELECT
                t.id AS target_id,
                t.stratum_id,
                t.variable,
                t.period,
                t.value,
                t.target_type,
                t.geographic_level,
                t.source,
                t.source_table,
                t.source_url,
                t.notes,
                s.name AS stratum_name,
                s.jurisdiction,
                sc.variable AS constraint_variable,
                sc.operator AS constraint_operator,
                sc.value AS constraint_value
            FROM targets AS t
            JOIN strata AS s
                ON s.id = t.stratum_id
            LEFT JOIN stratum_constraints AS sc
                ON sc.stratum_id = s.id
            WHERE {where_clause}
            ORDER BY t.id, sc.variable, sc.operator, sc.value
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            has_parent_id = _sqlite_table_has_column(conn, "strata", "parent_id")
            if has_parent_id:
                sql = f"""
                    WITH target_rows AS (
                        SELECT
                            t.id AS target_id,
                            t.stratum_id,
                            t.variable,
                            t.period,
                            t.value,
                            t.target_type,
                            t.geographic_level,
                            t.source,
                            t.source_table,
                            t.source_url,
                            t.notes,
                            s.name AS stratum_name,
                            s.jurisdiction,
                            s.parent_id
                        FROM targets AS t
                        JOIN strata AS s
                            ON s.id = t.stratum_id
                        WHERE {where_clause}
                    ),
                    ancestor_strata(target_id, stratum_id, depth) AS (
                        SELECT
                            target_id,
                            stratum_id,
                            0 AS depth
                        FROM target_rows
                        UNION ALL
                        SELECT
                            a.target_id,
                            parent.id AS stratum_id,
                            a.depth + 1 AS depth
                        FROM ancestor_strata AS a
                        JOIN strata AS child
                            ON child.id = a.stratum_id
                        JOIN strata AS parent
                            ON parent.id = child.parent_id
                        WHERE child.parent_id IS NOT NULL
                    )
                    SELECT
                        tr.target_id,
                        tr.stratum_id,
                        tr.variable,
                        tr.period,
                        tr.value,
                        tr.target_type,
                        tr.geographic_level,
                        tr.source,
                        tr.source_table,
                        tr.source_url,
                        tr.notes,
                        tr.stratum_name,
                        tr.jurisdiction,
                        sc.variable AS constraint_variable,
                        sc.operator AS constraint_operator,
                        sc.value AS constraint_value
                    FROM target_rows AS tr
                    LEFT JOIN ancestor_strata AS a
                        ON a.target_id = tr.target_id
                    LEFT JOIN stratum_constraints AS sc
                        ON sc.stratum_id = a.stratum_id
                    ORDER BY
                        tr.target_id,
                        a.depth DESC,
                        sc.variable,
                        sc.operator,
                        sc.value
                """
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return _group_arch_target_rows(rows)

    def _compose_model_year_records(
        self,
        *,
        target_year: int,
        jurisdiction: str,
        sources: tuple[str, ...],
        age_soi_targets: bool,
    ) -> list[ArchTargetRecord]:
        current_records = self.load_records(
            period=target_year,
            jurisdiction=jurisdiction,
            sources=sources,
        )
        if sources and _normalize_arch_source("IRS_SOI") not in {
            _normalize_arch_source(source) for source in sources
        }:
            return _with_state_to_national_rollup_records(current_records)

        non_soi_current_records = [
            record for record in current_records if record.source != "IRS_SOI"
        ]
        soi_records = self._latest_soi_records_by_composition(
            target_year=target_year,
            jurisdiction=jurisdiction,
        )
        if age_soi_targets:
            soi_records = self._age_soi_records_by_source_year(
                soi_records,
                target_year=target_year,
                jurisdiction=jurisdiction,
            )
        return _with_state_to_national_rollup_records(
            [*non_soi_current_records, *soi_records]
        )

    def _latest_soi_records_by_composition(
        self,
        *,
        target_year: int,
        jurisdiction: str,
    ) -> list[ArchTargetRecord]:
        """Return the latest SOI records for each target composition."""
        records = [
            record
            for record in self.load_records(
                period=None,
                jurisdiction=jurisdiction,
                sources=("IRS_SOI",),
            )
            if record.period <= target_year
        ]
        latest_period_by_key: dict[
            tuple[str, str, str, tuple[tuple[str, str, str], ...]],
            int,
        ] = {}
        for record in records:
            key = _arch_record_composition_key(record)
            latest_period_by_key[key] = max(
                latest_period_by_key.get(key, record.period),
                record.period,
            )
        return [
            record
            for record in records
            if record.period
            == latest_period_by_key[_arch_record_composition_key(record)]
        ]

    def _age_soi_records_by_source_year(
        self,
        records: list[ArchTargetRecord],
        *,
        target_year: int,
        jurisdiction: str,
    ) -> list[ArchTargetRecord]:
        aged: list[ArchTargetRecord] = []
        source_years = sorted({record.period for record in records})
        for source_year in source_years:
            source_records = [
                record for record in records if record.period == source_year
            ]
            if source_year == target_year:
                aged.extend(source_records)
            else:
                aged.extend(
                    self.age_soi_records(
                        source_records,
                        source_year=source_year,
                        target_year=target_year,
                        jurisdiction=jurisdiction,
                    )
                )
        return aged

    def latest_soi_year(self, target_year: int, *, jurisdiction: str) -> int | None:
        """Return the latest SOI year at or before the model year."""
        variables = tuple(sorted(ARCH_CURRENT_TAX_VARIABLES))
        placeholders = ", ".join("?" for _ in variables)
        sql = f"""
            SELECT DISTINCT t.period
            FROM targets AS t
            JOIN strata AS s
                ON s.id = t.stratum_id
            WHERE {_jurisdiction_clause(jurisdiction)}
              AND t.source = 'IRS_SOI'
              AND t.period <= ?
              AND t.variable IN ({placeholders})
            ORDER BY t.period DESC
        """
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(sql, [int(target_year), *variables]).fetchall()
        finally:
            conn.close()
        return int(rows[0][0]) if rows else None

    def age_soi_records(
        self,
        records: list[ArchTargetRecord],
        *,
        source_year: int,
        target_year: int,
        jurisdiction: str,
    ) -> list[ArchTargetRecord]:
        """Age SOI records with declared Microplex-side factors."""
        needs_count_factor = any(record.target_type == "COUNT" for record in records)
        needs_amount_factor = any(record.target_type == "AMOUNT" for record in records)
        factors = self.get_soi_aging_factors(
            source_year=source_year,
            target_year=target_year,
            jurisdiction=jurisdiction,
            needs_count_factor=needs_count_factor,
            needs_amount_factor=needs_amount_factor,
        )
        aged: list[ArchTargetRecord] = []
        for record in records:
            if record.source != "IRS_SOI":
                aged.append(record)
                continue
            if record.target_type == "COUNT":
                factor = factors.count_factor
            elif record.target_type == "AMOUNT":
                factor = factors.amount_factor
            else:
                factor = 1.0
            aged.append(
                replace(
                    record,
                    value=float(record.value) * factor,
                    period=target_year,
                    source_period=record.period,
                    aging_factors=factors,
                )
            )
        return aged

    def get_soi_aging_factors(
        self,
        *,
        source_year: int,
        target_year: int,
        jurisdiction: str,
        needs_count_factor: bool = True,
        needs_amount_factor: bool = True,
    ) -> SOIAgingFactors:
        """Resolve source-backed factors for SOI count and amount targets."""
        if source_year == target_year:
            return SOIAgingFactors(
                source_year=source_year,
                target_year=target_year,
                count_factor=1.0,
                amount_factor=1.0,
                count_method="identity",
                amount_method="identity",
            )
        if needs_count_factor:
            source_labor_force = self._target_value(
                year=source_year,
                jurisdiction=jurisdiction,
                source="BLS",
                variable="labor_force_count",
            )
            target_labor_force, count_method = self._labor_force_for_year(
                year=target_year,
                jurisdiction=jurisdiction,
            )
            count_factor = target_labor_force / source_labor_force
        else:
            count_factor = 1.0
            count_method = "not_required"

        if needs_amount_factor:
            source_agi = self._soi_total_agi(
                year=source_year, jurisdiction=jurisdiction
            )
            target_agi, amount_method = self._soi_total_agi_for_year(
                target_year=target_year,
                jurisdiction=jurisdiction,
            )
            amount_factor = target_agi / source_agi
        else:
            amount_factor = 1.0
            amount_method = "not_required"

        return SOIAgingFactors(
            source_year=source_year,
            target_year=target_year,
            count_factor=count_factor,
            amount_factor=amount_factor,
            count_method=count_method,
            amount_method=amount_method,
        )

    def _labor_force_for_year(
        self,
        *,
        year: int,
        jurisdiction: str,
    ) -> tuple[float, str]:
        bls_value = self._optional_target_value(
            year=year,
            jurisdiction=jurisdiction,
            source="BLS",
            variable="labor_force_count",
        )
        if bls_value is not None:
            return bls_value, "bls_labor_force_ratio"
        cbo_value = self._optional_target_value(
            year=year,
            jurisdiction=jurisdiction,
            source="CBO",
            variable="labor_force",
        )
        if cbo_value is not None:
            return cbo_value, "cbo_labor_force_ratio"
        raise ValueError(f"No BLS/CBO labor-force target found for {year}.")

    def _soi_total_agi_for_year(
        self,
        *,
        target_year: int,
        jurisdiction: str,
    ) -> tuple[float, str]:
        target_agi = self._optional_soi_total_agi(
            year=target_year,
            jurisdiction=jurisdiction,
        )
        if target_agi is not None:
            return target_agi, "soi_total_agi_ratio"

        available = {
            year: value
            for year in range(target_year - 20, target_year + 1)
            if (
                value := self._optional_soi_total_agi(
                    year=year,
                    jurisdiction=jurisdiction,
                )
            )
            is not None
        }
        if len(available) < 2:
            raise ValueError(
                "Need at least two SOI total AGI years to extrapolate "
                f"aggregate income to {target_year}."
            )
        latest_year = max(available)
        previous_year = max(year for year in available if year < latest_year)
        annual_growth = available[latest_year] / available[previous_year]
        years_forward = target_year - latest_year
        return (
            available[latest_year] * annual_growth**years_forward,
            "soi_total_agi_last_growth_extrapolation",
        )

    def _soi_total_agi(self, *, year: int, jurisdiction: str) -> float:
        value = self._optional_soi_total_agi(year=year, jurisdiction=jurisdiction)
        if value is None:
            raise ValueError(f"No SOI total AGI target found for {year}.")
        return value

    def _optional_soi_total_agi(self, *, year: int, jurisdiction: str) -> float | None:
        records = self.load_records(
            period=year,
            jurisdiction=jurisdiction,
            sources=("IRS_SOI",),
        )
        for record in records:
            if (
                record.variable == "adjusted_gross_income"
                and record.stratum_name == "US All Filers"
            ):
                return float(record.value)
        for record in records:
            if record.variable == "adjusted_gross_income" and record.constraints == (
                ("is_tax_filer", "==", "1"),
            ):
                return float(record.value)
        return None

    def _target_value(
        self,
        *,
        year: int,
        jurisdiction: str,
        source: str,
        variable: str,
    ) -> float:
        value = self._optional_target_value(
            year=year,
            jurisdiction=jurisdiction,
            source=source,
            variable=variable,
        )
        if value is None:
            raise ValueError(f"No {source} {variable} target found for {year}.")
        return value

    def _optional_target_value(
        self,
        *,
        year: int,
        jurisdiction: str,
        source: str,
        variable: str,
    ) -> float | None:
        records = self.load_records(
            period=year,
            jurisdiction=jurisdiction,
            sources=(source,),
        )
        matching = [record for record in records if record.variable == variable]
        if not matching:
            return None
        unconstrained = [record for record in matching if not record.constraints]
        if len(unconstrained) == 1:
            return float(unconstrained[0].value)
        return float(matching[0].value)


class ArchFactSQLiteTargetProvider:
    """Read Arch aggregate facts and expose Microplex canonical targets."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        jurisdiction: str = "us",
        compose_model_year_targets: bool = True,
        age_soi_targets: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.jurisdiction = jurisdiction
        self.compose_model_year_targets = compose_model_year_targets
        self.age_soi_targets = age_soi_targets

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        """Load canonical targets from Arch aggregate fact tables."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Arch facts DB not found: {self.db_path}")

        query = query or TargetQuery()
        provider_filters = dict(query.provider_filters)
        period = query.period if isinstance(query.period, int) else None
        variables = _as_string_tuple(provider_filters.get("variables"))
        domain_variables = _as_string_tuple(provider_filters.get("domain_variables"))
        sources = _as_string_tuple(provider_filters.get("sources"))
        geo_levels = _as_string_tuple(provider_filters.get("geo_levels"))
        target_cells = _as_target_cell_filters(provider_filters.get("target_cells"))
        entity_overrides = provider_filters.get("entity_overrides") or {}
        compose_model_year_targets = bool(
            provider_filters.get(
                "compose_model_year_targets",
                self.compose_model_year_targets,
            )
        )
        age_soi_targets = bool(
            provider_filters.get("age_soi_targets", self.age_soi_targets)
        )

        records = (
            self._compose_model_year_records(
                target_year=period,
                sources=sources,
                age_soi_targets=age_soi_targets,
            )
            if compose_model_year_targets and period is not None
            else self.load_records(period=period, sources=sources)
        )
        canonical_targets = TargetSet(
            [
                target
                for record in records
                if _matches_arch_provider_filters(
                    record,
                    variables=variables,
                    domain_variables=domain_variables,
                    geo_levels=geo_levels,
                    target_cells=target_cells,
                    entity_overrides=entity_overrides,
                )
                for target in [
                    arch_target_record_to_canonical_spec(
                        record,
                        entity_overrides=entity_overrides,
                    )
                ]
                if target is not None
            ]
        )
        return apply_target_query(
            canonical_targets,
            TargetQuery(
                period=query.period,
                entity=query.entity,
                names=query.names,
                metadata_filters=query.metadata_filters,
            ),
        )

    def load_records(
        self,
        *,
        period: int | None = None,
        sources: tuple[str, ...] = (),
    ) -> list[ArchTargetRecord]:
        """Load Arch fact rows with attached fact constraints and lineage."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            clauses = ["1 = 1"]
            params: list[Any] = []
            if period is not None:
                clauses.append("CAST(af.period_value AS INTEGER) = ?")
                params.append(int(period))
            where_clause = " AND ".join(clauses)
            rows = conn.execute(
                f"""
                SELECT
                    af.fact_key,
                    af.source_record_id,
                    af.value_numeric,
                    af.value_text,
                    af.value_json,
                    af.period_value,
                    af.geography_level,
                    af.geography_id,
                    af.geography_name,
                    af.measure_concept,
                    af.measure_source_concept,
                    af.measure_concept_relation,
                    af.measure_concept_authority,
                    af.measure_concept_evidence_url,
                    af.measure_concept_evidence_notes,
                    af.measure_legal_vintage,
                    af.measure_unit,
                    af.aggregation_method,
                    af.domain,
                    af.filters_json,
                    af.label,
                    af.source_name,
                    af.source_table,
                    af.source_url,
                    af.source_method_notes,
                    ac.ordinal AS constraint_ordinal,
                    ac.variable AS constraint_variable,
                    ac.operator AS constraint_operator,
                    ac.value_text AS constraint_value_text,
                    ac.value_numeric AS constraint_value_numeric,
                    ac.value_json AS constraint_value_json
                FROM aggregate_facts AS af
                LEFT JOIN aggregate_constraints AS ac
                    ON ac.fact_key = af.fact_key
                WHERE {where_clause}
                ORDER BY af.fact_key, ac.ordinal
                """,
                params,
            ).fetchall()
            lineage = _load_arch_fact_lineage(conn)
        finally:
            conn.close()

        records = _group_arch_fact_rows(rows, lineage=lineage)
        if sources:
            normalized_sources = {_normalize_arch_source(source) for source in sources}
            records = [
                record
                for record in records
                if _normalize_arch_source(record.source) in normalized_sources
            ]
        return records

    def _compose_model_year_records(
        self,
        *,
        target_year: int,
        sources: tuple[str, ...],
        age_soi_targets: bool,
    ) -> list[ArchTargetRecord]:
        return _compose_arch_model_year_records(
            self.load_records(period=None, sources=()),
            target_year=target_year,
            sources=sources,
            age_soi_targets=age_soi_targets,
        )


class ArchConsumerFactJSONLTargetProvider:
    """Read Arch consumer-contract JSONL facts as Microplex targets."""

    schema_version = "arch.consumer_fact.v1"

    def __init__(
        self,
        path: str | Path,
        *,
        jurisdiction: str = "us",
        compose_model_year_targets: bool = True,
        age_soi_targets: bool = True,
    ) -> None:
        self.path = Path(path)
        self.jurisdiction = jurisdiction
        self.compose_model_year_targets = compose_model_year_targets
        self.age_soi_targets = age_soi_targets

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        """Load canonical targets from Arch consumer-contract JSONL."""
        if not self.path.exists():
            raise FileNotFoundError(f"Arch consumer facts JSONL not found: {self.path}")

        query = query or TargetQuery()
        provider_filters = dict(query.provider_filters)
        period = query.period if isinstance(query.period, int) else None
        variables = _as_string_tuple(provider_filters.get("variables"))
        domain_variables = _as_string_tuple(provider_filters.get("domain_variables"))
        sources = _as_string_tuple(provider_filters.get("sources"))
        geo_levels = _as_string_tuple(provider_filters.get("geo_levels"))
        target_cells = _as_target_cell_filters(provider_filters.get("target_cells"))
        entity_overrides = provider_filters.get("entity_overrides") or {}
        compose_model_year_targets = bool(
            provider_filters.get(
                "compose_model_year_targets",
                self.compose_model_year_targets,
            )
        )
        age_soi_targets = bool(
            provider_filters.get("age_soi_targets", self.age_soi_targets)
        )

        records = (
            self._compose_model_year_records(
                target_year=period,
                sources=sources,
                age_soi_targets=age_soi_targets,
            )
            if compose_model_year_targets and period is not None
            else self.load_records(period=period, sources=sources)
        )
        canonical_targets = TargetSet(
            [
                target
                for record in records
                if _matches_arch_provider_filters(
                    record,
                    variables=variables,
                    domain_variables=domain_variables,
                    geo_levels=geo_levels,
                    target_cells=target_cells,
                    entity_overrides=entity_overrides,
                )
                for target in [
                    arch_target_record_to_canonical_spec(
                        record,
                        entity_overrides=entity_overrides,
                    )
                ]
                if target is not None
            ]
        )
        return apply_target_query(
            canonical_targets,
            TargetQuery(
                period=query.period,
                entity=query.entity,
                names=query.names,
                metadata_filters=query.metadata_filters,
            ),
        )

    def load_records(
        self,
        *,
        period: int | None = None,
        sources: tuple[str, ...] = (),
    ) -> list[ArchTargetRecord]:
        """Load Arch consumer-contract fact rows."""
        if not self.path.exists():
            raise FileNotFoundError(f"Arch consumer facts JSONL not found: {self.path}")

        rows = list(
            load_arch_consumer_fact_jsonl_rows(
                (self.path,),
                period=period,
                schema_version=self.schema_version,
            )
        )

        records = _consumer_fact_rows_to_records(rows)
        if sources:
            normalized_sources = {_normalize_arch_source(source) for source in sources}
            records = [
                record
                for record in records
                if _normalize_arch_source(record.source) in normalized_sources
            ]
        return records

    def _compose_model_year_records(
        self,
        *,
        target_year: int,
        sources: tuple[str, ...],
        age_soi_targets: bool,
    ) -> list[ArchTargetRecord]:
        return _compose_arch_model_year_records(
            self.load_records(period=None, sources=()),
            target_year=target_year,
            sources=sources,
            age_soi_targets=age_soi_targets,
        )


class ArchCompositeSQLiteTargetProvider:
    """Compose multiple Arch SQLite artifacts into one target provider."""

    def __init__(
        self,
        db_paths: tuple[str | Path, ...],
        *,
        jurisdiction: str = "us",
        compose_model_year_targets: bool = True,
        age_soi_targets: bool = True,
    ) -> None:
        paths = tuple(Path(path) for path in db_paths)
        if not paths:
            raise ValueError("At least one Arch targets DB path is required")
        self.db_paths = paths
        self.path = tuple(str(path) for path in paths)
        self.jurisdiction = jurisdiction
        self.compose_model_year_targets = compose_model_year_targets
        self.age_soi_targets = age_soi_targets
        self.providers = tuple(
            resolve_arch_sqlite_target_provider(
                path,
                jurisdiction=jurisdiction,
                compose_model_year_targets=compose_model_year_targets,
                age_soi_targets=age_soi_targets,
            )
            for path in paths
        )

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        """Load and renumber targets across all configured Arch artifacts."""
        query = query or TargetQuery()
        provider_filters = dict(query.provider_filters)
        period = query.period if isinstance(query.period, int) else None
        variables = _as_string_tuple(provider_filters.get("variables"))
        domain_variables = _as_string_tuple(provider_filters.get("domain_variables"))
        sources = _as_string_tuple(provider_filters.get("sources"))
        geo_levels = _as_string_tuple(provider_filters.get("geo_levels"))
        target_cells = _as_target_cell_filters(provider_filters.get("target_cells"))
        entity_overrides = provider_filters.get("entity_overrides") or {}
        compose_model_year_targets = bool(
            provider_filters.get(
                "compose_model_year_targets",
                self.compose_model_year_targets,
            )
        )
        age_soi_targets = bool(
            provider_filters.get("age_soi_targets", self.age_soi_targets)
        )

        records = self.load_records(
            period=period,
            sources=sources,
            compose_model_year_targets=compose_model_year_targets,
            age_soi_targets=age_soi_targets,
        )
        stratum_ids: dict[tuple[tuple[str, str, Any], ...], int] = {}
        targets: list[CanonicalTargetSpec] = []
        for record in records:
            if not _matches_arch_provider_filters(
                record,
                variables=variables,
                domain_variables=domain_variables,
                geo_levels=geo_levels,
                target_cells=target_cells,
                entity_overrides=entity_overrides,
            ):
                continue
            target = arch_target_record_to_canonical_spec(
                record,
                entity_overrides=entity_overrides,
            )
            if target is None:
                continue
            metadata = dict(target.metadata)
            metadata["stratum_id"] = stratum_ids.setdefault(
                _target_filter_tuple(target),
                len(stratum_ids) + 1,
            )
            targets.append(
                replace(
                    target,
                    name=f"arch_target_{metadata['target_id']}",
                    metadata=metadata,
                )
            )
        return apply_target_query(
            TargetSet(targets),
            TargetQuery(
                period=query.period,
                entity=query.entity,
                names=query.names,
                metadata_filters=query.metadata_filters,
            ),
        )

    def load_records(
        self,
        *,
        period: int | None = None,
        sources: tuple[str, ...] = (),
        compose_model_year_targets: bool | None = None,
        age_soi_targets: bool | None = None,
    ) -> list[ArchTargetRecord]:
        """Load and renumber raw records across configured Arch artifacts."""
        records = self._load_all_child_records()
        resolved_compose = (
            self.compose_model_year_targets
            if compose_model_year_targets is None
            else compose_model_year_targets
        )
        resolved_age_soi = (
            self.age_soi_targets if age_soi_targets is None else age_soi_targets
        )
        if resolved_compose and period is not None:
            records = _compose_arch_model_year_records(
                records,
                target_year=period,
                sources=sources,
                age_soi_targets=resolved_age_soi,
            )
        else:
            records = [
                record
                for record in records
                if (period is None or record.period == period)
                and _record_matches_sources(record, sources)
            ]
        return _renumber_arch_records(records)

    def _load_all_child_records(self) -> list[ArchTargetRecord]:
        records: list[ArchTargetRecord] = []
        seen_fact_keys: set[str] = set()
        for source_index, (path, provider) in enumerate(
            zip(self.db_paths, self.providers, strict=True),
            start=1,
        ):
            provider_records = _load_arch_provider_raw_records(
                provider,
                jurisdiction=self.jurisdiction,
            )
            for record in provider_records:
                if record.aggregate_fact_key is not None:
                    if record.aggregate_fact_key in seen_fact_keys:
                        continue
                    seen_fact_keys.add(record.aggregate_fact_key)
                records.append(
                    replace(
                        record,
                        source_db_path=str(path),
                        source_db_index=source_index,
                        source_target_id=record.source_target_id or record.target_id,
                        source_stratum_id=(
                            record.source_stratum_id or record.stratum_id
                        ),
                    )
                )
        return records


def _load_arch_provider_raw_records(
    provider: (
        ArchSQLiteTargetProvider
        | ArchFactSQLiteTargetProvider
        | ArchConsumerFactJSONLTargetProvider
        | ArchCompositeSQLiteTargetProvider
    ),
    *,
    jurisdiction: str,
) -> list[ArchTargetRecord]:
    if isinstance(
        provider,
        (ArchFactSQLiteTargetProvider, ArchConsumerFactJSONLTargetProvider),
    ):
        return provider.load_records(period=None, sources=())
    if isinstance(provider, ArchCompositeSQLiteTargetProvider):
        return provider._load_all_child_records()
    return provider.load_records(period=None, jurisdiction=jurisdiction, sources=())


def _compose_arch_model_year_records(
    records: list[ArchTargetRecord],
    *,
    target_year: int,
    sources: tuple[str, ...],
    age_soi_targets: bool,
) -> list[ArchTargetRecord]:
    current_records = [
        record
        for record in records
        if record.period == target_year and _record_matches_sources(record, sources)
    ]
    normalized_sources = {_normalize_arch_source(source) for source in sources}
    if sources and _normalize_arch_source("IRS_SOI") not in normalized_sources:
        return _with_state_to_national_rollup_records(current_records)

    non_soi_current_records = [
        record
        for record in current_records
        if _normalize_arch_source(record.source) != "IRS_SOI"
    ]
    soi_records = _latest_soi_records_by_composition(
        records,
        target_year=target_year,
    )
    if age_soi_targets:
        soi_records = _age_arch_soi_records_by_source_year(
            soi_records,
            target_year=target_year,
            reference_records=records,
        )
    else:
        soi_records = [
            _carry_forward_arch_record_to_model_year(record, target_year=target_year)
            for record in soi_records
        ]
    return _with_state_to_national_rollup_records(
        [*non_soi_current_records, *soi_records]
    )


def _with_state_to_national_rollup_records(
    records: list[ArchTargetRecord],
) -> list[ArchTargetRecord]:
    rollups = _state_to_national_rollup_records(records)
    if not rollups:
        return records
    return [*records, *rollups]


def _state_to_national_rollup_records(
    records: list[ArchTargetRecord],
) -> list[ArchTargetRecord]:
    existing_national_keys = {
        key
        for record in records
        if _arch_record_geo_level(record) == "national"
        for key in [_state_rollup_group_key(record)]
        if key is not None
    }
    grouped: dict[tuple[Any, ...], list[tuple[str, ArchTargetRecord]]] = {}
    for record in records:
        if _arch_record_geo_level(record) != "state":
            continue
        key = _state_rollup_group_key(record)
        if key is None or key in existing_national_keys:
            continue
        state_fips = _arch_record_state_fips(record)
        if state_fips is None or state_fips not in ARCH_NATIONAL_ROLLUP_STATE_FIPS:
            continue
        grouped.setdefault(key, []).append((state_fips, record))

    rollups: list[ArchTargetRecord] = []
    for key, state_records in grouped.items():
        records_by_state: dict[str, ArchTargetRecord] = {}
        for state_fips, record in state_records:
            if state_fips in records_by_state:
                records_by_state = {}
                break
            records_by_state[state_fips] = record
        if set(records_by_state) != ARCH_NATIONAL_ROLLUP_STATE_FIPS:
            continue
        ordered_records = [
            records_by_state[state_fips]
            for state_fips in sorted(ARCH_NATIONAL_ROLLUP_STATE_FIPS)
        ]
        rollups.append(
            _state_records_to_national_rollup_record(
                key,
                ordered_records,
            )
        )
    return rollups


def _state_rollup_group_key(record: ArchTargetRecord) -> tuple[Any, ...] | None:
    if record.variable not in ARCH_STATE_TO_NATIONAL_ROLLUP_VARIABLES:
        return None
    return (
        _normalize_arch_source(record.source),
        record.source_table,
        record.source_url,
        record.variable,
        record.target_type,
        record.period,
        record.source_period,
        record.aging_factors,
        record.unit,
        record.concept,
        record.source_concept,
        record.concept_relation,
        record.concept_authority,
        record.legal_vintage,
        _non_state_constraints(record.constraints),
    )


def _state_records_to_national_rollup_record(
    key: tuple[Any, ...],
    records: list[ArchTargetRecord],
) -> ArchTargetRecord:
    first = records[0]
    digest = sha1(repr(key).encode("utf-8")).hexdigest()
    source_row_keys = tuple(
        dict.fromkeys(
            source_row_key
            for record in records
            for source_row_key in (
                record.source_row_keys
                or (str(record.source_target_id or record.target_id),)
            )
        )
    )
    source_cell_keys = tuple(
        dict.fromkeys(
            source_cell_key
            for record in records
            for source_cell_key in record.source_cell_keys
        )
    )
    notes = "Microplex national rollup from 51 state targets."
    if first.notes:
        notes = f"{first.notes} {notes}"
    return replace(
        first,
        target_id=-int(digest[:12], 16),
        stratum_id=-int(digest[12:20], 16),
        value=sum(record.value for record in records),
        geographic_level=None,
        geography_id=None,
        stratum_name="US National Rollup",
        constraints=_non_state_constraints(first.constraints),
        notes=notes,
        source_record_id=f"microplex_state_rollup:{digest[:16]}",
        source_cell_keys=source_cell_keys,
        source_row_keys=source_row_keys,
        source_target_id=None,
        source_stratum_id=None,
    )


def _non_state_constraints(
    constraints: tuple[tuple[str, str, str], ...],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        constraint for constraint in constraints if constraint[0] != "state_fips"
    )


def _arch_record_state_fips(record: ArchTargetRecord) -> str | None:
    for variable, operator, value in record.constraints:
        if variable != "state_fips":
            continue
        if _canonical_arch_constraint_operator(operator) != "==":
            continue
        try:
            return str(int(float(value))).zfill(2)
        except (TypeError, ValueError):
            return str(value).zfill(2)
    if _normalize_geo_level(record.geographic_level) == "state":
        geography_id = record.geography_id
        if geography_id is not None:
            return _state_fips_from_arch_geography_id(geography_id)
    return None


def _latest_soi_records_by_composition(
    records: list[ArchTargetRecord],
    *,
    target_year: int,
) -> list[ArchTargetRecord]:
    candidates = [
        record
        for record in records
        if _normalize_arch_source(record.source) == "IRS_SOI"
        and record.period <= target_year
    ]
    latest_period_by_key: dict[
        tuple[str, str, str, tuple[tuple[str, str, str], ...]],
        int,
    ] = {}
    for record in candidates:
        key = _arch_record_composition_key(record)
        latest_period_by_key[key] = max(
            latest_period_by_key.get(key, record.period),
            record.period,
        )
    return [
        record
        for record in candidates
        if record.period == latest_period_by_key[_arch_record_composition_key(record)]
    ]


def _age_arch_soi_records_by_source_year(
    records: list[ArchTargetRecord],
    *,
    target_year: int,
    reference_records: list[ArchTargetRecord],
) -> list[ArchTargetRecord]:
    aged: list[ArchTargetRecord] = []
    for source_year in sorted({record.period for record in records}):
        source_records = [record for record in records if record.period == source_year]
        if source_year == target_year:
            aged.extend(source_records)
            continue
        needs_count_factor = any(
            record.target_type == "COUNT" for record in source_records
        )
        needs_amount_factor = any(
            record.target_type == "AMOUNT" for record in source_records
        )
        factors = _arch_record_soi_aging_factors(
            reference_records,
            source_year=source_year,
            target_year=target_year,
            needs_count_factor=needs_count_factor,
            needs_amount_factor=needs_amount_factor,
        )
        for record in source_records:
            factor = 1.0
            if record.target_type == "COUNT":
                factor = factors.count_factor
            elif record.target_type == "AMOUNT":
                factor = factors.amount_factor
            aged.append(
                replace(
                    record,
                    value=float(record.value) * factor,
                    period=target_year,
                    source_period=record.period,
                    aging_factors=factors,
                )
            )
    return aged


def _carry_forward_arch_record_to_model_year(
    record: ArchTargetRecord,
    *,
    target_year: int,
) -> ArchTargetRecord:
    if record.period == target_year:
        return record
    return replace(record, period=target_year, source_period=record.period)


def _arch_record_soi_aging_factors(
    records: list[ArchTargetRecord],
    *,
    source_year: int,
    target_year: int,
    needs_count_factor: bool,
    needs_amount_factor: bool,
) -> SOIAgingFactors:
    if source_year == target_year:
        return SOIAgingFactors(
            source_year=source_year,
            target_year=target_year,
            count_factor=1.0,
            amount_factor=1.0,
            count_method="identity",
            amount_method="identity",
        )

    if needs_count_factor:
        count_factor, count_method = _arch_record_soi_count_aging_factor(
            records,
            source_year=source_year,
            target_year=target_year,
        )
    else:
        count_factor = 1.0
        count_method = "not_required"

    if needs_amount_factor:
        amount_factor, amount_method = _arch_record_soi_amount_aging_factor(
            records,
            source_year=source_year,
            target_year=target_year,
        )
    else:
        amount_factor = 1.0
        amount_method = "not_required"

    return SOIAgingFactors(
        source_year=source_year,
        target_year=target_year,
        count_factor=count_factor,
        amount_factor=amount_factor,
        count_method=count_method,
        amount_method=amount_method,
    )


def _arch_record_soi_count_aging_factor(
    records: list[ArchTargetRecord],
    *,
    source_year: int,
    target_year: int,
) -> tuple[float, str]:
    source_labor_force = _optional_arch_total_value(
        records,
        year=source_year,
        source="BLS",
        variable="labor_force_count",
    )
    target_labor_force, labor_force_method = _optional_arch_labor_force_for_year(
        records,
        year=target_year,
    )
    if source_labor_force is not None and target_labor_force is not None:
        return target_labor_force / source_labor_force, labor_force_method

    source_count = _optional_arch_soi_total_value(
        records,
        year=source_year,
        variable="tax_unit_count",
    )
    target_count, count_method = _arch_soi_total_for_year(
        records,
        target_year=target_year,
        variable="tax_unit_count",
        exact_method="soi_total_return_count_ratio",
        extrapolation_method="soi_total_return_count_last_growth_extrapolation",
    )
    if source_count is not None and target_count is not None:
        return target_count / source_count, count_method
    return 1.0, "source_fact_carry_forward_no_count_reference"


def _arch_record_soi_amount_aging_factor(
    records: list[ArchTargetRecord],
    *,
    source_year: int,
    target_year: int,
) -> tuple[float, str]:
    source_agi = _optional_arch_soi_total_value(
        records,
        year=source_year,
        variable="adjusted_gross_income",
    )
    target_agi, amount_method = _arch_soi_total_for_year(
        records,
        target_year=target_year,
        variable="adjusted_gross_income",
        exact_method="soi_total_agi_ratio",
        extrapolation_method="soi_total_agi_last_growth_extrapolation",
    )
    if source_agi is not None and target_agi is not None:
        return target_agi / source_agi, amount_method
    return 1.0, "source_fact_carry_forward_no_amount_reference"


def _optional_arch_labor_force_for_year(
    records: list[ArchTargetRecord],
    *,
    year: int,
) -> tuple[float | None, str]:
    bls_value = _optional_arch_total_value(
        records,
        year=year,
        source="BLS",
        variable="labor_force_count",
    )
    if bls_value is not None:
        return bls_value, "bls_labor_force_ratio"
    cbo_value = _optional_arch_total_value(
        records,
        year=year,
        source="CBO",
        variable="labor_force",
    )
    if cbo_value is not None:
        return cbo_value, "cbo_labor_force_ratio"
    return None, "source_fact_carry_forward_no_labor_force_reference"


def _arch_soi_total_for_year(
    records: list[ArchTargetRecord],
    *,
    target_year: int,
    variable: str,
    exact_method: str,
    extrapolation_method: str,
) -> tuple[float | None, str]:
    exact = _optional_arch_soi_total_value(
        records,
        year=target_year,
        variable=variable,
    )
    if exact is not None:
        return exact, exact_method

    available = {
        year: value
        for year in sorted({record.period for record in records})
        if year <= target_year
        for value in [
            _optional_arch_soi_total_value(
                records,
                year=year,
                variable=variable,
            )
        ]
        if value is not None
    }
    if len(available) < 2:
        return None, f"source_fact_carry_forward_no_{variable}_reference"
    latest_year = max(available)
    previous_year = max(year for year in available if year < latest_year)
    annual_growth = available[latest_year] / available[previous_year]
    years_forward = target_year - latest_year
    return available[latest_year] * annual_growth**years_forward, extrapolation_method


def _optional_arch_soi_total_value(
    records: list[ArchTargetRecord],
    *,
    year: int,
    variable: str,
) -> float | None:
    return _optional_arch_total_value(
        records,
        year=year,
        source="IRS_SOI",
        variable=variable,
        require_total_scope=True,
    )


def _optional_arch_total_value(
    records: list[ArchTargetRecord],
    *,
    year: int,
    source: str,
    variable: str,
    require_total_scope: bool = False,
) -> float | None:
    matches = [
        record
        for record in records
        if record.period == year
        and _normalize_arch_source(record.source) == _normalize_arch_source(source)
        and record.variable == variable
    ]
    if require_total_scope:
        matches = [record for record in matches if _arch_record_is_total_scope(record)]
    if not matches:
        return None
    return float(matches[0].value)


def _arch_record_is_total_scope(record: ArchTargetRecord) -> bool:
    if not record.constraints:
        return True
    if tuple(record.constraints) == (("is_tax_filer", "==", "1"),):
        return True
    if tuple(record.constraints) == (("tax_unit_is_filer", "==", "1"),):
        return True
    return str(record.stratum_name or "").lower().endswith("all filers")


def _record_matches_sources(
    record: ArchTargetRecord,
    sources: tuple[str, ...],
) -> bool:
    if not sources:
        return True
    normalized_sources = {_normalize_arch_source(source) for source in sources}
    return _normalize_arch_source(record.source) in normalized_sources


def _renumber_arch_records(records: list[ArchTargetRecord]) -> list[ArchTargetRecord]:
    renumbered: list[ArchTargetRecord] = []
    stratum_ids: dict[tuple[tuple[str, str, str], ...], int] = {}
    for record in records:
        renumbered.append(
            replace(
                record,
                target_id=len(renumbered) + 1,
                stratum_id=stratum_ids.setdefault(
                    record.constraints,
                    len(stratum_ids) + 1,
                ),
            )
        )
    return renumbered


def resolve_arch_sqlite_target_provider(
    db_path: str | Path | tuple[str | Path, ...],
    *,
    jurisdiction: str = "us",
    compose_model_year_targets: bool = True,
    age_soi_targets: bool = True,
) -> (
    ArchSQLiteTargetProvider
    | ArchFactSQLiteTargetProvider
    | ArchConsumerFactJSONLTargetProvider
    | ArchCompositeSQLiteTargetProvider
):
    """Return the Arch provider matching a source artifact's schema."""
    paths = _as_arch_db_path_tuple(db_path)
    if len(paths) > 1:
        return ArchCompositeSQLiteTargetProvider(
            paths,
            jurisdiction=jurisdiction,
            compose_model_year_targets=compose_model_year_targets,
            age_soi_targets=age_soi_targets,
        )
    path = paths[0]
    if not path.exists():
        raise FileNotFoundError(f"Arch targets DB not found: {path}")
    if _looks_like_arch_consumer_fact_jsonl(path):
        return ArchConsumerFactJSONLTargetProvider(
            path,
            jurisdiction=jurisdiction,
            compose_model_year_targets=compose_model_year_targets,
            age_soi_targets=age_soi_targets,
        )
    conn = sqlite3.connect(path)
    try:
        if _sqlite_table_exists(conn, "aggregate_facts"):
            return ArchFactSQLiteTargetProvider(
                path,
                jurisdiction=jurisdiction,
                compose_model_year_targets=compose_model_year_targets,
                age_soi_targets=age_soi_targets,
            )
    finally:
        conn.close()
    return ArchSQLiteTargetProvider(
        path,
        jurisdiction=jurisdiction,
        compose_model_year_targets=compose_model_year_targets,
        age_soi_targets=age_soi_targets,
    )


def summarize_arch_target_profile_coverage(
    provider: (
        ArchSQLiteTargetProvider
        | ArchFactSQLiteTargetProvider
        | ArchConsumerFactJSONLTargetProvider
        | ArchCompositeSQLiteTargetProvider
    ),
    *,
    period: int,
    profile_name: str = "pe_native_broad",
    target_cells: tuple[PolicyEngineUSTargetCell | dict[str, Any], ...] | None = None,
    sources: tuple[str, ...] = (),
    jurisdiction: str | None = None,
    compose_model_year_targets: bool | None = None,
    age_soi_targets: bool | None = None,
    entity_overrides: dict[str, Any] | None = None,
    provider_filters: dict[str, Any] | None = None,
) -> ArchTargetProfileCoverageReport:
    """Summarize how much of a Microplex target profile Arch can satisfy."""

    resolved_cells = (
        tuple(target_cells)
        if target_cells is not None
        else resolve_policyengine_us_target_profile(profile_name)
    )
    cell_filters = tuple(
        _target_cell_to_provider_filter(cell) for cell in resolved_cells
    )
    query_filters: dict[str, Any] = dict(provider_filters or {})
    query_filters["target_profile"] = profile_name
    query_filters["target_cells"] = [dict(cell) for cell in cell_filters]
    if sources:
        query_filters["sources"] = list(sources)
    if jurisdiction is not None:
        query_filters["jurisdiction"] = jurisdiction
    if compose_model_year_targets is not None:
        query_filters["compose_model_year_targets"] = compose_model_year_targets
    if age_soi_targets is not None:
        query_filters["age_soi_targets"] = age_soi_targets
    if entity_overrides is not None:
        query_filters["entity_overrides"] = entity_overrides

    target_set = provider.load_target_set(
        TargetQuery(period=period, provider_filters=query_filters)
    )
    coverage_cells = tuple(
        _coverage_for_arch_target_cell(cell_filter, target_set)
        for cell_filter in cell_filters
    )
    target_cell_count = len(coverage_cells)
    covered_cell_count = sum(1 for cell in coverage_cells if cell.covered)
    uncovered_cell_count = target_cell_count - covered_cell_count
    coverage_rate = covered_cell_count / target_cell_count if target_cell_count else 0.0
    return ArchTargetProfileCoverageReport(
        profile_name=profile_name,
        period=int(period),
        target_cell_count=target_cell_count,
        covered_cell_count=covered_cell_count,
        uncovered_cell_count=uncovered_cell_count,
        coverage_rate=coverage_rate,
        by_geo_level=_summarize_arch_cell_coverage(coverage_cells, field="geo_level"),
        by_variable=_summarize_arch_cell_coverage(coverage_cells, field="variable"),
        cells=coverage_cells,
    )


def summarize_arch_target_gap_queue(
    provider: (
        ArchSQLiteTargetProvider
        | ArchFactSQLiteTargetProvider
        | ArchConsumerFactJSONLTargetProvider
        | ArchCompositeSQLiteTargetProvider
    ),
    *,
    period: int,
    profile_name: str = "pe_native_broad",
    include_covered: bool = False,
    target_cells: tuple[PolicyEngineUSTargetCell | dict[str, Any], ...] | None = None,
    sources: tuple[str, ...] = (),
    jurisdiction: str | None = None,
    compose_model_year_targets: bool | None = None,
    age_soi_targets: bool | None = None,
    entity_overrides: dict[str, Any] | None = None,
    provider_filters: dict[str, Any] | None = None,
) -> ArchTargetGapQueueReport:
    """Build an agent-facing queue of Arch target records to add or review."""

    coverage = summarize_arch_target_profile_coverage(
        provider,
        period=period,
        profile_name=profile_name,
        target_cells=target_cells,
        sources=sources,
        jurisdiction=jurisdiction,
        compose_model_year_targets=compose_model_year_targets,
        age_soi_targets=age_soi_targets,
        entity_overrides=entity_overrides,
        provider_filters=provider_filters,
    )
    catalog = _arch_gap_loaded_variable_catalog(
        provider,
        period=period,
        jurisdiction=jurisdiction,
        sources=sources,
        compose_model_year_targets=compose_model_year_targets,
        age_soi_targets=age_soi_targets,
    )
    variable_uncovered_counts = {
        variable: counts["uncovered_cell_count"]
        for variable, counts in coverage.by_variable.items()
    }
    rows = [
        _arch_gap_queue_row_for_coverage_cell(
            coverage_cell,
            profile_name=profile_name,
            period=period,
            loaded_variable_catalog=catalog,
            variable_uncovered_count=variable_uncovered_counts.get(
                str(coverage_cell.cell.get("variable") or ""),
                0,
            ),
        )
        for coverage_cell in coverage.cells
        if include_covered or not coverage_cell.covered
    ]
    rows = [
        replace(row, priority=priority)
        for priority, row in enumerate(
            sorted(rows, key=_arch_gap_queue_sort_key),
            start=1,
        )
    ]
    by_loader_status: dict[str, int] = {}
    by_gap_category: dict[str, int] = {}
    for row in rows:
        by_loader_status[row.loader_status] = (
            by_loader_status.get(row.loader_status, 0) + 1
        )
        by_gap_category[row.gap_category] = by_gap_category.get(row.gap_category, 0) + 1
    covered_row_count = sum(1 for row in rows if row.covered)
    return ArchTargetGapQueueReport(
        profile_name=profile_name,
        period=int(period),
        row_count=len(rows),
        covered_row_count=covered_row_count,
        uncovered_row_count=len(rows) - covered_row_count,
        by_loader_status=dict(sorted(by_loader_status.items())),
        by_gap_category=dict(sorted(by_gap_category.items())),
        rows=tuple(rows),
    )


def summarize_arch_target_parity(
    incumbent_provider: (
        ArchSQLiteTargetProvider
        | ArchFactSQLiteTargetProvider
        | ArchConsumerFactJSONLTargetProvider
        | ArchCompositeSQLiteTargetProvider
    ),
    candidate_provider: (
        ArchSQLiteTargetProvider
        | ArchFactSQLiteTargetProvider
        | ArchConsumerFactJSONLTargetProvider
        | ArchCompositeSQLiteTargetProvider
    ),
    *,
    period: int,
    sources: tuple[str, ...] = (),
    variables: tuple[str, ...] = (),
    value_abs_tolerance: float = 1e-6,
    value_rel_tolerance: float = 1e-12,
) -> ArchTargetParityReport:
    """Compare canonical Microplex targets loaded from two Arch artifacts."""
    provider_filters: dict[str, Any] = {}
    if sources:
        provider_filters["sources"] = tuple(sources)
    if variables:
        provider_filters["variables"] = tuple(variables)

    query = TargetQuery(period=period, provider_filters=provider_filters)
    incumbent_targets = list(incumbent_provider.load_target_set(query).targets)
    candidate_targets = list(candidate_provider.load_target_set(query).targets)
    rows = _arch_target_parity_rows(
        incumbent_targets,
        candidate_targets,
        value_abs_tolerance=value_abs_tolerance,
        value_rel_tolerance=value_rel_tolerance,
    )
    errors = tuple(
        _arch_target_parity_error(row) for row in rows if row.status != "matched"
    )
    counts = {
        "incumbent_target_count": len(incumbent_targets),
        "candidate_target_count": len(candidate_targets),
        "matched_count": sum(1 for row in rows if row.status == "matched"),
        "value_mismatch_count": sum(
            1 for row in rows if row.status == "value_mismatch"
        ),
        "incumbent_only_count": sum(
            1 for row in rows if row.status == "incumbent_only"
        ),
        "candidate_only_count": sum(
            1 for row in rows if row.status == "candidate_only"
        ),
        "duplicate_identity_count": sum(
            1 for row in rows if row.status == "duplicate_identity"
        ),
    }
    return ArchTargetParityReport(
        period=int(period),
        incumbent_artifacts=_arch_provider_artifacts(incumbent_provider),
        candidate_artifacts=_arch_provider_artifacts(candidate_provider),
        value_abs_tolerance=value_abs_tolerance,
        value_rel_tolerance=value_rel_tolerance,
        counts=counts,
        rows=rows,
        errors=errors,
    )


def main_coverage(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Arch target-profile coverage JSON."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Summarize Arch target DB coverage for a Microplex target profile."
    )
    parser.add_argument(
        "--arch-targets-db",
        required=True,
        action="append",
        help=(
            "Arch targets SQLite DB path or consumer-fact JSONL path. May be "
            "supplied multiple times to combine source-package artifacts."
        ),
    )
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--profile", default="pe_native_broad")
    parser.add_argument("--jurisdiction", default="us")
    parser.add_argument("--source", action="append", dest="sources", default=[])
    parser.add_argument(
        "--no-compose-model-year-targets",
        action="store_false",
        dest="compose_model_year_targets",
        default=True,
    )
    parser.add_argument(
        "--no-age-soi-targets",
        action="store_false",
        dest="age_soi_targets",
        default=True,
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    provider = resolve_arch_sqlite_target_provider(
        _single_or_many_paths(args.arch_targets_db),
        jurisdiction=args.jurisdiction,
        compose_model_year_targets=args.compose_model_year_targets,
        age_soi_targets=args.age_soi_targets,
    )
    report = summarize_arch_target_profile_coverage(
        provider,
        period=args.period,
        profile_name=args.profile,
        sources=tuple(args.sources),
        jurisdiction=args.jurisdiction,
        compose_model_year_targets=args.compose_model_year_targets,
        age_soi_targets=args.age_soi_targets,
    )
    json.dump(report.to_dict(), sys.stdout, indent=args.indent, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def main_gaps(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Arch target-profile gap queue rows."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Emit an agent-facing Arch target gap queue for a profile."
    )
    parser.add_argument(
        "--arch-targets-db",
        required=True,
        action="append",
        help=(
            "Arch targets SQLite DB path or consumer-fact JSONL path. May be "
            "supplied multiple times to combine source-package artifacts."
        ),
    )
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--profile", default="pe_native_broad")
    parser.add_argument("--jurisdiction", default="us")
    parser.add_argument("--source", action="append", dest="sources", default=[])
    parser.add_argument("--include-covered", action="store_true")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--output")
    parser.add_argument(
        "--no-compose-model-year-targets",
        action="store_false",
        dest="compose_model_year_targets",
        default=True,
    )
    parser.add_argument(
        "--no-age-soi-targets",
        action="store_false",
        dest="age_soi_targets",
        default=True,
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    provider = resolve_arch_sqlite_target_provider(
        _single_or_many_paths(args.arch_targets_db),
        jurisdiction=args.jurisdiction,
        compose_model_year_targets=args.compose_model_year_targets,
        age_soi_targets=args.age_soi_targets,
    )
    report = summarize_arch_target_gap_queue(
        provider,
        period=args.period,
        profile_name=args.profile,
        include_covered=args.include_covered,
        sources=tuple(args.sources),
        jurisdiction=args.jurisdiction,
        compose_model_year_targets=args.compose_model_year_targets,
        age_soi_targets=args.age_soi_targets,
    )
    if args.format == "csv":
        output = _arch_target_gap_queue_csv(report)
    else:
        output = json.dumps(report.to_dict(), indent=args.indent, sort_keys=True)
        output += "\n"

    if args.output:
        Path(args.output).write_text(output)
    else:
        sys.stdout.write(output)
    return 0


def main_refresh(argv: list[str] | None = None) -> int:
    """Refresh Arch target coverage and gap snapshots for a target profile."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Write Arch target-profile coverage, gap queue, and summary artifacts."
        )
    )
    parser.add_argument(
        "--arch-targets-db",
        action="append",
        default=[],
        help=(
            "Arch targets SQLite DB path or consumer-fact JSONL path. May be "
            "supplied multiple times. If omitted, --artifact-root is searched."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help=(
            "Directory or file to search for Arch target artifacts when "
            "--arch-targets-db is omitted. May be supplied multiple times."
        ),
    )
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--profile", default="pe_native_broad")
    parser.add_argument("--jurisdiction", default="us")
    parser.add_argument("--source", action="append", dest="sources", default=[])
    parser.add_argument(
        "--output-dir",
        default="artifacts/arch-target-coverage",
        help="Directory for coverage JSON, gap JSON/CSV, and markdown summary.",
    )
    parser.add_argument(
        "--no-compose-model-year-targets",
        action="store_false",
        dest="compose_model_year_targets",
        default=True,
    )
    parser.add_argument(
        "--no-age-soi-targets",
        action="store_false",
        dest="age_soi_targets",
        default=True,
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    artifact_paths = tuple(Path(path) for path in args.arch_targets_db)
    if not artifact_paths:
        discovery_roots = (
            tuple(Path(path) for path in args.artifact_root)
            or _default_arch_target_artifact_roots()
        )
        artifact_paths = discover_arch_target_artifacts(discovery_roots)
    if not artifact_paths:
        roots = args.artifact_root or [
            str(path) for path in _default_arch_target_artifact_roots()
        ]
        raise FileNotFoundError(
            "No Arch target artifacts found. Pass --arch-targets-db or place "
            f"consumer_facts.jsonl / Arch targets DB files under: {', '.join(roots)}"
        )

    provider = resolve_arch_sqlite_target_provider(
        _single_or_many_paths([str(path) for path in artifact_paths]),
        jurisdiction=args.jurisdiction,
        compose_model_year_targets=args.compose_model_year_targets,
        age_soi_targets=args.age_soi_targets,
    )
    coverage = summarize_arch_target_profile_coverage(
        provider,
        period=args.period,
        profile_name=args.profile,
        sources=tuple(args.sources),
        jurisdiction=args.jurisdiction,
        compose_model_year_targets=args.compose_model_year_targets,
        age_soi_targets=args.age_soi_targets,
    )
    gaps = summarize_arch_target_gap_queue(
        provider,
        period=args.period,
        profile_name=args.profile,
        sources=tuple(args.sources),
        jurisdiction=args.jurisdiction,
        compose_model_year_targets=args.compose_model_year_targets,
        age_soi_targets=args.age_soi_targets,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_filename_slug(args.profile)}_{int(args.period)}"
    coverage_path = output_dir / f"{stem}_coverage.json"
    gaps_json_path = output_dir / f"{stem}_gaps.json"
    gaps_csv_path = output_dir / f"{stem}_gaps.csv"
    summary_path = output_dir / f"{stem}_summary.md"

    coverage_path.write_text(
        json.dumps(coverage.to_dict(), indent=args.indent, sort_keys=True) + "\n"
    )
    gaps_json_path.write_text(
        json.dumps(gaps.to_dict(), indent=args.indent, sort_keys=True) + "\n"
    )
    gaps_csv_path.write_text(_arch_target_gap_queue_csv(gaps))
    summary_path.write_text(
        _arch_target_refresh_summary_markdown(
            coverage,
            gaps,
            artifact_paths=artifact_paths,
            output_paths=(
                coverage_path,
                gaps_json_path,
                gaps_csv_path,
                summary_path,
            ),
        )
    )

    json.dump(
        {
            "profile_name": coverage.profile_name,
            "period": coverage.period,
            "target_cell_count": coverage.target_cell_count,
            "covered_cell_count": coverage.covered_cell_count,
            "uncovered_cell_count": coverage.uncovered_cell_count,
            "coverage_rate": coverage.coverage_rate,
            "artifact_paths": [str(path) for path in artifact_paths],
            "output_paths": {
                "coverage": str(coverage_path),
                "gaps_json": str(gaps_json_path),
                "gaps_csv": str(gaps_csv_path),
                "summary": str(summary_path),
            },
        },
        sys.stdout,
        indent=args.indent,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


def main_parity(argv: list[str] | None = None) -> int:
    """CLI entrypoint comparing incumbent and candidate Arch target artifacts."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Compare two Arch target artifacts after loading both through the "
            "Microplex Arch provider."
        )
    )
    parser.add_argument(
        "--incumbent-arch-targets-db",
        required=True,
        action="append",
        help=(
            "Incumbent Arch targets SQLite DB path. May be supplied multiple "
            "times to combine artifacts."
        ),
    )
    parser.add_argument(
        "--candidate-arch-targets-db",
        required=True,
        action="append",
        help=(
            "Candidate Arch targets SQLite DB or consumer-fact JSONL path. May "
            "be supplied multiple times to combine artifacts."
        ),
    )
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--jurisdiction", default="us")
    parser.add_argument("--source", action="append", dest="sources", default=[])
    parser.add_argument("--variable", action="append", dest="variables", default=[])
    parser.add_argument("--value-abs-tolerance", type=float, default=1e-6)
    parser.add_argument("--value-rel-tolerance", type=float, default=1e-12)
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument(
        "--no-compose-model-year-targets",
        action="store_false",
        dest="compose_model_year_targets",
        default=True,
    )
    parser.add_argument(
        "--no-age-soi-targets",
        action="store_false",
        dest="age_soi_targets",
        default=True,
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    try:
        incumbent_provider = resolve_arch_sqlite_target_provider(
            _single_or_many_paths(args.incumbent_arch_targets_db),
            jurisdiction=args.jurisdiction,
            compose_model_year_targets=args.compose_model_year_targets,
            age_soi_targets=args.age_soi_targets,
        )
        candidate_provider = resolve_arch_sqlite_target_provider(
            _single_or_many_paths(args.candidate_arch_targets_db),
            jurisdiction=args.jurisdiction,
            compose_model_year_targets=args.compose_model_year_targets,
            age_soi_targets=args.age_soi_targets,
        )
        payload = summarize_arch_target_parity(
            incumbent_provider,
            candidate_provider,
            period=args.period,
            sources=tuple(args.sources),
            variables=tuple(args.variables),
            value_abs_tolerance=args.value_abs_tolerance,
            value_rel_tolerance=args.value_rel_tolerance,
        ).to_dict(row_limit=args.row_limit)
    except Exception as exc:  # noqa: BLE001 - CLI must return JSON on failures.
        payload = {
            "valid": False,
            "period": args.period,
            "incumbent_artifacts": list(args.incumbent_arch_targets_db),
            "candidate_artifacts": list(args.candidate_arch_targets_db),
            "counts": {
                "incumbent_target_count": 0,
                "candidate_target_count": 0,
                "matched_count": 0,
                "value_mismatch_count": 0,
                "incumbent_only_count": 0,
                "candidate_only_count": 0,
                "duplicate_identity_count": 0,
            },
            "row_count": 0,
            "rows": [],
            "errors": [{"code": "load_failed", "message": str(exc)}],
        }

    json.dump(payload, sys.stdout, indent=args.indent, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if payload["valid"] else 1


def main_smoke(argv: list[str] | None = None) -> int:
    """CLI entrypoint proving Arch artifacts load as Microplex targets."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Load an Arch target artifact, including consumer_facts.jsonl, "
            "through the Microplex Arch provider and emit a JSON smoke report."
        )
    )
    parser.add_argument(
        "--arch-targets-db",
        required=True,
        action="append",
        help=(
            "Arch targets SQLite DB path or consumer-fact JSONL path. May be "
            "supplied multiple times to combine source-package artifacts."
        ),
    )
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--jurisdiction", default="us")
    parser.add_argument("--source", action="append", dest="sources", default=[])
    parser.add_argument("--variable", action="append", dest="variables", default=[])
    parser.add_argument("--expected-target-count", type=int)
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument(
        "--no-compose-model-year-targets",
        action="store_false",
        dest="compose_model_year_targets",
        default=True,
    )
    parser.add_argument(
        "--no-age-soi-targets",
        action="store_false",
        dest="age_soi_targets",
        default=True,
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    errors: list[dict[str, str]] = []
    targets: list[CanonicalTargetSpec] = []
    try:
        provider = resolve_arch_sqlite_target_provider(
            _single_or_many_paths(args.arch_targets_db),
            jurisdiction=args.jurisdiction,
            compose_model_year_targets=args.compose_model_year_targets,
            age_soi_targets=args.age_soi_targets,
        )
        provider_filters: dict[str, Any] = {}
        if args.sources:
            provider_filters["sources"] = tuple(args.sources)
        if args.variables:
            provider_filters["variables"] = tuple(args.variables)
        targets = list(
            provider.load_target_set(
                TargetQuery(
                    period=args.period,
                    provider_filters=provider_filters,
                )
            ).targets
        )
    except Exception as exc:  # noqa: BLE001 - CLI must return JSON on failures.
        errors.append({"code": "load_failed", "message": str(exc)})

    if (
        args.expected_target_count is not None
        and len(targets) != args.expected_target_count
    ):
        errors.append(
            {
                "code": "unexpected_target_count",
                "message": (
                    f"Expected {args.expected_target_count} targets, "
                    f"loaded {len(targets)}."
                ),
            }
        )

    payload = {
        "valid": not errors,
        "period": args.period,
        "target_count": len(targets),
        "by_variable": dict(
            sorted(Counter(_target_variable(target) for target in targets).items())
        ),
        "by_source": dict(
            sorted(Counter(str(target.source) for target in targets).items())
        ),
        "by_aggregation": dict(
            sorted(
                Counter(
                    str(getattr(target.aggregation, "value", target.aggregation))
                    for target in targets
                ).items()
            )
        ),
        "by_filter_count": {
            str(key): value
            for key, value in sorted(
                Counter(len(target.filters) for target in targets).items()
            )
        },
        "sample_targets": [
            _target_smoke_sample(target)
            for target in targets[: max(0, args.sample_limit)]
        ],
        "errors": errors,
    }
    json.dump(payload, sys.stdout, indent=args.indent, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if payload["valid"] else 1


def _target_variable(target: CanonicalTargetSpec) -> str:
    """Return the Microplex variable represented by a canonical target."""
    variable = target.metadata.get("variable") if target.metadata else None
    return str(variable or target.measure or target.name)


def _target_smoke_sample(target: CanonicalTargetSpec) -> dict[str, Any]:
    """Return a compact JSON sample for an Arch target smoke report."""
    return {
        "name": target.name,
        "variable": _target_variable(target),
        "aggregation": str(getattr(target.aggregation, "value", target.aggregation)),
        "measure": target.measure,
        "value": target.value,
        "period": target.period,
        "source": str(target.source),
        "filters": [
            {
                "feature": target_filter.feature,
                "operator": str(
                    getattr(target_filter.operator, "value", target_filter.operator)
                ),
                "value": target_filter.value,
            }
            for target_filter in target.filters
        ],
        "metadata": {
            key: target.metadata[key]
            for key in (
                "arch_aggregate_fact_key",
                "arch_semantic_fact_key",
                "arch_source_record_id",
                "geo_level",
            )
            if key in target.metadata
        },
    }


def _arch_target_parity_rows(
    incumbent_targets: list[CanonicalTargetSpec],
    candidate_targets: list[CanonicalTargetSpec],
    *,
    value_abs_tolerance: float,
    value_rel_tolerance: float,
) -> tuple[ArchTargetParityRow, ...]:
    incumbent_by_identity = _index_arch_targets_by_parity_identity(incumbent_targets)
    candidate_by_identity = _index_arch_targets_by_parity_identity(candidate_targets)
    rows: list[ArchTargetParityRow] = []
    for identity in sorted(
        set(incumbent_by_identity) | set(candidate_by_identity),
        key=str,
    ):
        incumbent_group = tuple(incumbent_by_identity.get(identity, ()))
        candidate_group = tuple(candidate_by_identity.get(identity, ()))
        absolute_delta: float | None = None
        relative_delta: float | None = None
        if len(incumbent_group) != 1 or len(candidate_group) != 1:
            status = _arch_target_parity_nonunique_status(
                incumbent_group,
                candidate_group,
            )
        else:
            incumbent_value = float(incumbent_group[0].value)
            candidate_value = float(candidate_group[0].value)
            absolute_delta = candidate_value - incumbent_value
            relative_delta = (
                absolute_delta / incumbent_value if incumbent_value != 0 else None
            )
            status = (
                "matched"
                if _arch_target_values_match(
                    incumbent_value,
                    candidate_value,
                    abs_tolerance=value_abs_tolerance,
                    rel_tolerance=value_rel_tolerance,
                )
                else "value_mismatch"
            )
        rows.append(
            ArchTargetParityRow(
                status=status,
                identity=identity,
                incumbent_targets=incumbent_group,
                candidate_targets=candidate_group,
                absolute_delta=absolute_delta,
                relative_delta=relative_delta,
            )
        )
    return tuple(sorted(rows, key=_arch_target_parity_row_sort_key))


def _index_arch_targets_by_parity_identity(
    targets: list[CanonicalTargetSpec],
) -> dict[tuple[Any, ...], list[CanonicalTargetSpec]]:
    indexed: dict[tuple[Any, ...], list[CanonicalTargetSpec]] = {}
    for target in targets:
        indexed.setdefault(_arch_target_parity_identity(target), []).append(target)
    return indexed


def _arch_target_parity_identity(target: CanonicalTargetSpec) -> tuple[Any, ...]:
    metadata = target.metadata or {}
    return (
        str(getattr(target.entity, "value", target.entity)),
        str(getattr(target.aggregation, "value", target.aggregation)),
        str(target.measure or ""),
        _arch_target_period_value(target.period),
        str(target.source or ""),
        _target_variable(target),
        str(metadata.get("geo_level") or ""),
        str(_arch_target_geographic_id(target) or ""),
        _target_parity_filter_tuple(target),
    )


def _arch_target_period_value(value: int | str) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _target_parity_filter_tuple(
    target: CanonicalTargetSpec,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (
                str(target_filter.feature),
                str(getattr(target_filter.operator, "value", target_filter.operator)),
                _json_scalar_text(target_filter.value),
            )
            for target_filter in target.filters
        )
    )


def _arch_target_parity_nonunique_status(
    incumbent_targets: tuple[CanonicalTargetSpec, ...],
    candidate_targets: tuple[CanonicalTargetSpec, ...],
) -> str:
    if len(incumbent_targets) > 1 or len(candidate_targets) > 1:
        return "duplicate_identity"
    if not incumbent_targets:
        return "candidate_only"
    if not candidate_targets:
        return "incumbent_only"
    return "duplicate_identity"


def _arch_target_values_match(
    incumbent_value: float,
    candidate_value: float,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> bool:
    delta = abs(candidate_value - incumbent_value)
    if delta <= abs_tolerance:
        return True
    scale = max(abs(incumbent_value), abs(candidate_value), 1.0)
    return delta <= rel_tolerance * scale


def _arch_target_parity_row_sort_key(row: ArchTargetParityRow) -> tuple[int, str]:
    status_rank = {
        "value_mismatch": 0,
        "incumbent_only": 1,
        "candidate_only": 2,
        "duplicate_identity": 3,
        "matched": 4,
    }
    return (
        status_rank.get(row.status, 99),
        json.dumps(_arch_target_parity_identity_dict(row.identity), sort_keys=True),
    )


def _arch_target_parity_error(row: ArchTargetParityRow) -> dict[str, Any]:
    identity = _arch_target_parity_identity_dict(row.identity)
    if row.status == "value_mismatch":
        return {
            "code": "value_mismatch",
            "message": "Candidate target value differs from incumbent target value.",
            "identity": identity,
            "incumbent_value": row.incumbent_targets[0].value,
            "candidate_value": row.candidate_targets[0].value,
            "absolute_delta": row.absolute_delta,
            "relative_delta": row.relative_delta,
        }
    if row.status == "incumbent_only":
        return {
            "code": "missing_candidate_target",
            "message": "Incumbent target identity is absent from the candidate artifact.",
            "identity": identity,
            "incumbent_target_count": len(row.incumbent_targets),
            "candidate_target_count": len(row.candidate_targets),
        }
    if row.status == "candidate_only":
        return {
            "code": "unexpected_candidate_target",
            "message": "Candidate target identity is absent from the incumbent artifact.",
            "identity": identity,
            "incumbent_target_count": len(row.incumbent_targets),
            "candidate_target_count": len(row.candidate_targets),
        }
    return {
        "code": "duplicate_identity",
        "message": "A target identity is not unique in one or both artifacts.",
        "identity": identity,
        "incumbent_target_count": len(row.incumbent_targets),
        "candidate_target_count": len(row.candidate_targets),
    }


def _arch_target_parity_identity_dict(identity: tuple[Any, ...]) -> dict[str, Any]:
    (
        entity,
        aggregation,
        measure,
        period,
        source,
        variable,
        geo_level,
        geographic_id,
        filters,
    ) = identity
    return {
        "entity": entity,
        "aggregation": aggregation,
        "measure": measure or None,
        "period": period,
        "source": source or None,
        "variable": variable,
        "geo_level": geo_level or None,
        "geographic_id": geographic_id or None,
        "filters": [
            {"feature": feature, "operator": operator, "value": value}
            for feature, operator, value in filters
        ],
    }


def _target_parity_sample(target: CanonicalTargetSpec) -> dict[str, Any]:
    sample = _target_smoke_sample(target)
    metadata = dict(sample["metadata"])
    for key in (
        "target_id",
        "source_table",
        "display_label",
        "arch_source_period",
        "arch_model_period",
    ):
        if key in target.metadata:
            metadata[key] = target.metadata[key]
    sample["metadata"] = metadata
    return sample


def _arch_provider_artifacts(
    provider: (
        ArchSQLiteTargetProvider
        | ArchFactSQLiteTargetProvider
        | ArchConsumerFactJSONLTargetProvider
        | ArchCompositeSQLiteTargetProvider
    ),
) -> tuple[str, ...]:
    if isinstance(provider, ArchCompositeSQLiteTargetProvider):
        return tuple(str(path) for path in provider.db_paths)
    path = getattr(provider, "db_path", None) or getattr(provider, "path", None)
    if path is None:
        return ()
    return (str(path),)


def arch_target_record_to_canonical_spec(
    record: ArchTargetRecord,
    *,
    entity_overrides: dict[str, Any] | None = None,
) -> CanonicalTargetSpec | None:
    """Translate one Arch target record into a canonical core target spec."""
    if record.target_type == "RATE":
        return None
    if _should_skip_arch_target_record(record):
        return None

    filters = list(_canonical_filters_for_arch_constraints(record.constraints))
    geography_filter = _target_filter_for_arch_geography(record)
    if geography_filter is not None:
        filters.append(geography_filter)
    entity_overrides = entity_overrides or {}
    source_variable = record.variable
    model_variable: str
    aggregation: TargetAggregation
    measure: str | None
    entity: EntityType

    if record.target_type == "COUNT":
        count_mapping = ARCH_COUNT_VARIABLE_ALIASES.get(source_variable)
        positive_measure = _positive_measure_for_count_record(source_variable)
        if count_mapping is not None:
            model_variable, entity, count_filter_measure = count_mapping
            if count_filter_measure is not None:
                filters.append(
                    TargetFilter(
                        feature=count_filter_measure,
                        operator=">",
                        value=0,
                    )
                )
        elif positive_measure is not None:
            model_variable = positive_measure
            entity = EntityType.TAX_UNIT
            filters.append(
                TargetFilter(feature=positive_measure, operator=">", value=0)
            )
        else:
            return None
        aggregation = TargetAggregation.COUNT
        measure = None
    elif record.target_type == "AMOUNT":
        model_variable = ARCH_AMOUNT_VARIABLE_ALIASES.get(
            source_variable, source_variable
        )
        aggregation = TargetAggregation.SUM
        measure = model_variable
        if _is_blocked_self_employment_binding(record, model_variable):
            raise ValueError(
                "Broad Arch business/proprietors income cannot be exposed as "
                "plain self_employment_income; use a dedicated proprietors-income "
                "target or an explicit composite mapping."
            )
        entity = _entity_for_measure(model_variable, entity_overrides)
        if model_variable in ARCH_POSITIVE_AMOUNT_FILTER_VARIABLES:
            filters.append(
                TargetFilter(
                    feature=model_variable,
                    operator=">",
                    value=0,
                )
            )
    else:
        return None

    filters = list(_dedupe_target_filters(filters))
    display_label = _arch_target_display_label(record)
    metadata = {
        "target_id": record.target_id,
        "stratum_id": record.stratum_id,
        "display_label": display_label,
        "variable": model_variable,
        "model_variable_role": policyengine_us_variable_role(model_variable).value,
        "arch_variable": record.variable,
        "arch_target_type": record.target_type,
        "target_semantic": record.target_type.lower(),
        "source": record.source,
        "source_table": record.source_table,
        "source_url": record.source_url,
        "notes": record.notes,
        "stratum_name": record.stratum_name,
        "jurisdiction": record.jurisdiction,
        "geo_level": _arch_record_geo_level(record),
        "geographic_level": record.geographic_level,
        "geography_id": record.geography_id,
        "constraint_count": len(filters),
        "arch_source_period": record.source_period or record.period,
        "arch_model_period": record.period,
    }
    if record.aggregate_fact_key is not None:
        metadata["arch_aggregate_fact_key"] = record.aggregate_fact_key
    if record.semantic_fact_key is not None:
        metadata["arch_semantic_fact_key"] = record.semantic_fact_key
    if record.source_record_id is not None:
        metadata["arch_source_record_id"] = record.source_record_id
    if record.source_cell_keys:
        metadata["arch_source_cell_keys"] = list(record.source_cell_keys)
    if record.source_row_keys:
        metadata["arch_source_row_keys"] = list(record.source_row_keys)
    if record.unit is not None:
        metadata["unit"] = record.unit
    if record.concept is not None:
        metadata["arch_concept"] = record.concept
    if record.source_concept is not None:
        metadata["arch_source_concept"] = record.source_concept
    if record.concept_relation is not None:
        metadata["arch_concept_relation"] = record.concept_relation
    if record.concept_authority is not None:
        metadata["arch_concept_authority"] = record.concept_authority
    if record.concept_evidence_url is not None:
        metadata["arch_concept_evidence_url"] = record.concept_evidence_url
    if record.concept_evidence_notes is not None:
        metadata["arch_concept_evidence_notes"] = record.concept_evidence_notes
    if record.legal_vintage is not None:
        metadata["arch_legal_vintage"] = record.legal_vintage
    if record.source_db_path is not None:
        metadata["arch_source_db_path"] = record.source_db_path
    if record.source_db_index is not None:
        metadata["arch_source_db_index"] = record.source_db_index
    if record.source_target_id is not None:
        metadata["arch_source_target_id"] = record.source_target_id
    if record.source_stratum_id is not None:
        metadata["arch_source_stratum_id"] = record.source_stratum_id
    if record.aging_factors is not None:
        factors = record.aging_factors
        metadata.update(
            {
                "arch_aged": True,
                "arch_aging_source_year": factors.source_year,
                "arch_aging_target_year": factors.target_year,
                "arch_aging_count_factor": factors.count_factor,
                "arch_aging_amount_factor": factors.amount_factor,
                "arch_aging_count_method": factors.count_method,
                "arch_aging_amount_method": factors.amount_method,
            }
        )

    return CanonicalTargetSpec(
        name=f"arch_target_{record.target_id}",
        entity=entity,
        value=record.value,
        period=record.period,
        measure=measure,
        aggregation=aggregation,
        filters=tuple(filters),
        source=record.source,
        description=display_label,
        metadata=metadata,
    )


def _should_skip_arch_target_record(record: ArchTargetRecord) -> bool:
    return _is_bea_regional_country_record(record)


def _is_blocked_self_employment_binding(
    record: ArchTargetRecord,
    model_variable: str,
) -> bool:
    if model_variable != "self_employment_income":
        return False
    markers = {
        str(value)
        for value in (
            record.variable,
            record.concept,
            record.source_concept,
            record.source_record_id,
        )
        if value is not None
    }
    markers.update(
        f"{variable}:{value}"
        for variable, _, value in record.constraints
        if value is not None
    )
    return bool(markers & ARCH_BROAD_BUSINESS_INCOME_SELF_EMPLOYMENT_BLOCKLIST)


def _is_bea_regional_country_record(record: ArchTargetRecord) -> bool:
    if not _has_bea_regional_lineage(record):
        return False
    if str(record.geography_id) == "0100000US":
        return True
    return _arch_record_geo_level(record) in {"national", "country"}


def _has_bea_regional_lineage(record: ArchTargetRecord) -> bool:
    lineage_values = (
        record.concept,
        record.source_concept,
        record.source_record_id,
    )
    return any(
        str(value).startswith("bea_regional.")
        or str(value).startswith("bea-regional.")
        or ".bea-regional-" in str(value)
        for value in lineage_values
        if value is not None
    )


def _group_arch_target_rows(rows: list[sqlite3.Row]) -> list[ArchTargetRecord]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        target_id = int(row["target_id"])
        item = grouped.setdefault(
            target_id,
            {
                "target_id": target_id,
                "stratum_id": int(row["stratum_id"]),
                "variable": row["variable"],
                "period": int(row["period"]),
                "value": float(row["value"]),
                "target_type": str(row["target_type"]),
                "geographic_level": row["geographic_level"],
                "geography_id": None,
                "source": row["source"],
                "source_table": row["source_table"],
                "source_url": row["source_url"],
                "notes": row["notes"],
                "stratum_name": row["stratum_name"],
                "jurisdiction": row["jurisdiction"],
                "constraints": [],
            },
        )
        if row["constraint_variable"] is not None:
            constraint = (
                str(row["constraint_variable"]),
                str(row["constraint_operator"]),
                str(row["constraint_value"]),
            )
            if constraint not in item["constraints"]:
                item["constraints"].append(constraint)
    return [
        ArchTargetRecord(
            **{
                **item,
                "constraints": tuple(item["constraints"]),
            }
        )
        for item in grouped.values()
    ]


def _load_arch_fact_lineage(
    conn: sqlite3.Connection,
) -> dict[str, dict[str, tuple[str, ...]]]:
    lineage: dict[str, dict[str, tuple[str, ...]]] = {}
    if _sqlite_table_exists(conn, "fact_source_cells"):
        for row in conn.execute(
            """
            SELECT fact_key, source_cell_key
            FROM fact_source_cells
            ORDER BY fact_key, ordinal
            """
        ):
            fact_key = str(row["fact_key"])
            item = lineage.setdefault(fact_key, {})
            item["source_cell_keys"] = (
                *item.get("source_cell_keys", ()),
                str(row["source_cell_key"]),
            )
    if _sqlite_table_exists(conn, "fact_source_rows"):
        for row in conn.execute(
            """
            SELECT fact_key, source_row_key
            FROM fact_source_rows
            ORDER BY fact_key, ordinal
            """
        ):
            fact_key = str(row["fact_key"])
            item = lineage.setdefault(fact_key, {})
            item["source_row_keys"] = (
                *item.get("source_row_keys", ()),
                str(row["source_row_key"]),
            )
    return lineage


def _consumer_fact_rows_to_records(
    rows: list[dict[str, Any]],
) -> list[ArchTargetRecord]:
    records: list[ArchTargetRecord] = []
    stratum_ids: dict[tuple[tuple[str, str, str], ...], int] = {}
    for target_id, row in enumerate(rows, start=1):
        constraints = tuple(
            dict.fromkeys(
                constraint
                for constraint in (
                    *_arch_consumer_fact_domain_constraints(row),
                    *(
                        _arch_consumer_fact_constraint(constraint)
                        for constraint in _consumer_fact_universe_constraints(row).get(
                            "constraints", []
                        )
                    ),
                )
                if constraint is not None
            )
        )
        stratum_id = stratum_ids.setdefault(constraints, len(stratum_ids) + 1)
        variable, target_type = _arch_consumer_fact_target_identity(row)
        source = row.get("source") or {}
        observed_measure = row.get("observed_measure") or {}
        geography = row.get("geography") or {}
        lineage = row.get("lineage") or {}
        concept_alignment = row.get("concept_alignment") or {}
        source_name = (
            source.get("source_name") or observed_measure.get("source_name") or "arch"
        )
        records.append(
            ArchTargetRecord(
                target_id=target_id,
                stratum_id=stratum_id,
                variable=variable,
                period=_consumer_fact_period(row),
                value=_json_numeric_value(row.get("value")),
                target_type=target_type,
                geographic_level=_arch_consumer_fact_geographic_level(row),
                geography_id=geography.get("id"),
                source=_normalize_arch_source(str(source_name)),
                source_table=source.get("source_table")
                or observed_measure.get("source_table"),
                source_url=source.get("url"),
                notes=source.get("method_notes"),
                stratum_name=_arch_consumer_fact_stratum_name(row),
                jurisdiction="US",
                constraints=constraints,
                aggregate_fact_key=row.get("aggregate_fact_key"),
                semantic_fact_key=row.get("semantic_fact_key"),
                source_record_id=arch_consumer_fact_source_record_id(row),
                source_cell_keys=tuple(lineage.get("source_cell_keys") or ()),
                source_row_keys=tuple(lineage.get("source_row_keys") or ()),
                unit=observed_measure.get("unit"),
                concept=_arch_consumer_fact_concept(row),
                source_concept=concept_alignment.get("source_concept")
                or observed_measure.get("source_concept"),
                concept_relation=concept_alignment.get("relation"),
                concept_authority=concept_alignment.get("authority"),
                concept_evidence_url=concept_alignment.get("evidence_url"),
                concept_evidence_notes=concept_alignment.get("evidence_notes"),
                legal_vintage=concept_alignment.get("legal_vintage"),
            )
        )
    return records


def _consumer_fact_period(row: dict[str, Any]) -> int:
    return arch_consumer_fact_period(row)


def _arch_consumer_fact_target_identity(row: dict[str, Any]) -> tuple[str, str]:
    concept = _arch_consumer_fact_concept(row)
    if concept == "ssa.annual_oasdi_or_ssi_payment_amount":
        return (_ssa_payment_variable_from_consumer_fact(row), "AMOUNT")
    try:
        return ARCH_FACT_CONCEPT_TO_TARGET[concept]
    except KeyError as exc:
        raise ValueError(
            f"No Microplex Arch consumer fact mapping for concept {concept!r}"
        ) from exc


def _ssa_payment_variable_from_consumer_fact(row: dict[str, Any]) -> str:
    for constraint in _consumer_fact_universe_constraints(row).get("constraints", []):
        if (
            constraint.get("variable")
            == "us_social_security_and_ssi.program_payment_type"
        ):
            return str(constraint.get("value"))
    raise ValueError("SSA payment fact row has no program payment type constraint.")


def _arch_consumer_fact_concept(row: dict[str, Any]) -> str:
    concept = arch_consumer_fact_concept(row)
    if concept is None:
        raise ValueError("Arch consumer fact row has no mappable concept.")
    return concept


def _arch_consumer_fact_domain_constraints(
    row: dict[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    domain = str(_consumer_fact_universe_constraints(row).get("domain"))
    return _arch_fact_domain_constraints_for_domain(domain)


def _arch_consumer_fact_constraint(
    constraint: dict[str, Any],
) -> tuple[str, str, str] | None:
    variable = str(constraint["variable"])
    if variable in ARCH_IGNORED_FACT_CONSTRAINT_VARIABLES:
        return None
    try:
        mapped_variable = ARCH_FACT_CONSTRAINT_VARIABLE_ALIASES[variable]
    except KeyError as exc:
        raise ValueError(
            f"No Microplex Arch consumer fact constraint mapping for variable {variable!r}"
        ) from exc
    return (
        mapped_variable,
        str(constraint["operator"]),
        _json_scalar_text(constraint.get("value")),
    )


def _consumer_fact_universe_constraints(row: dict[str, Any]) -> dict[str, Any]:
    universe_constraints = row.get("universe_constraints") or {}
    if not isinstance(universe_constraints, dict):
        raise ValueError("Arch consumer fact universe_constraints must be an object.")
    return universe_constraints


def _arch_consumer_fact_geographic_level(row: dict[str, Any]) -> str | None:
    geography = row.get("geography") or {}
    return _arch_geographic_level_from_arch_level(geography.get("level"))


def _arch_consumer_fact_stratum_name(row: dict[str, Any]) -> str:
    dimensions = row.get("dimensions") or {}
    income_range = dimensions.get("income_range")
    geography_name = _arch_consumer_fact_geography_name(row)
    if income_range == "all":
        return f"{geography_name} All Filers"
    if income_range:
        return f"{geography_name} Filers AGI {income_range}"
    return str(row.get("label") or geography_name)


def _arch_consumer_fact_geography_name(row: dict[str, Any]) -> str:
    geography = row.get("geography") or {}
    level = str(geography.get("level") or "").lower()
    if level == "country":
        return "US"
    return str(geography.get("name") or geography.get("id") or "US")


def _group_arch_fact_rows(
    rows: list[sqlite3.Row],
    *,
    lineage: dict[str, dict[str, tuple[str, ...]]],
) -> list[ArchTargetRecord]:
    grouped: dict[str, dict[str, Any]] = {}
    stratum_ids: dict[tuple[tuple[str, str, str], ...], int] = {}
    for row in rows:
        fact_key = str(row["fact_key"])
        item = grouped.setdefault(
            fact_key,
            {
                "row": row,
                "constraints": list(_arch_fact_domain_constraints(row)),
            },
        )
        if row["constraint_variable"] is not None:
            constraint = _arch_fact_constraint(row)
            if constraint is not None:
                item["constraints"].append(constraint)

    records: list[ArchTargetRecord] = []
    for target_id, (fact_key, item) in enumerate(sorted(grouped.items()), start=1):
        row = item["row"]
        constraints = tuple(dict.fromkeys(item["constraints"]))
        stratum_id = stratum_ids.setdefault(constraints, len(stratum_ids) + 1)
        variable, target_type = _arch_fact_target_identity(row)
        period = int(row["period_value"])
        source_name = row["source_name"] or "arch"
        fact_lineage = lineage.get(fact_key, {})
        records.append(
            ArchTargetRecord(
                target_id=target_id,
                stratum_id=stratum_id,
                variable=variable,
                period=period,
                value=_arch_fact_numeric_value(row),
                target_type=target_type,
                geographic_level=_arch_fact_geographic_level(row),
                geography_id=row["geography_id"],
                source=_normalize_arch_source(source_name),
                source_table=row["source_table"],
                source_url=row["source_url"],
                notes=row["source_method_notes"],
                stratum_name=_arch_fact_stratum_name(row),
                jurisdiction="US",
                constraints=constraints,
                aggregate_fact_key=fact_key,
                semantic_fact_key=_arch_fact_semantic_key(row, constraints),
                source_record_id=row["source_record_id"],
                source_cell_keys=fact_lineage.get("source_cell_keys", ()),
                source_row_keys=fact_lineage.get("source_row_keys", ()),
                unit=row["measure_unit"],
                concept=row["measure_concept"],
                source_concept=row["measure_source_concept"],
                concept_relation=row["measure_concept_relation"],
                concept_authority=row["measure_concept_authority"],
                concept_evidence_url=row["measure_concept_evidence_url"],
                concept_evidence_notes=row["measure_concept_evidence_notes"],
                legal_vintage=row["measure_legal_vintage"],
            )
        )
    return records


def _arch_fact_target_identity(row: sqlite3.Row) -> tuple[str, str]:
    concept = str(row["measure_concept"])
    try:
        return ARCH_FACT_CONCEPT_TO_TARGET[concept]
    except KeyError as exc:
        raise ValueError(
            f"No Microplex Arch fact mapping for concept {concept!r}"
        ) from exc


def _arch_fact_domain_constraints(row: sqlite3.Row) -> tuple[tuple[str, str, str], ...]:
    domain = str(row["domain"])
    return _arch_fact_domain_constraints_for_domain(domain)


def _arch_fact_domain_constraints_for_domain(
    domain: str,
) -> tuple[tuple[str, str, str], ...]:
    try:
        return ARCH_FACT_DOMAIN_CONSTRAINTS[domain]
    except KeyError as exc:
        raise ValueError(
            f"No Microplex Arch fact mapping for domain {domain!r}"
        ) from exc


def _arch_fact_constraint(row: sqlite3.Row) -> tuple[str, str, str] | None:
    variable = str(row["constraint_variable"])
    if variable in ARCH_IGNORED_FACT_CONSTRAINT_VARIABLES:
        return None
    try:
        mapped_variable = ARCH_FACT_CONSTRAINT_VARIABLE_ALIASES[variable]
    except KeyError as exc:
        raise ValueError(
            f"No Microplex Arch fact constraint mapping for variable {variable!r}"
        ) from exc
    return (
        mapped_variable,
        str(row["constraint_operator"]),
        _sqlite_json_scalar_text(
            row["constraint_value_text"],
            row["constraint_value_numeric"],
            row["constraint_value_json"],
        ),
    )


def _arch_fact_numeric_value(row: sqlite3.Row) -> float:
    numeric = row["value_numeric"]
    if numeric is not None:
        return float(numeric)
    return float(_sqlite_json_scalar_text(row["value_text"], None, row["value_json"]))


def _sqlite_json_scalar_text(
    text_value: Any,
    numeric_value: Any,
    json_value: Any,
) -> str:
    if text_value is not None:
        return str(text_value)
    if numeric_value is not None:
        numeric = float(numeric_value)
        return str(int(numeric)) if numeric.is_integer() else str(numeric)
    return str(json_value)


def _arch_fact_geographic_level(row: sqlite3.Row) -> str | None:
    return _arch_geographic_level_from_arch_level(row["geography_level"])


def _arch_geographic_level_from_arch_level(level_value: Any) -> str | None:
    level = str(level_value or "").lower()
    if level == "country":
        return "NATIONAL"
    if level == "state":
        return "STATE"
    if level == "county":
        return "COUNTY"
    if level in {"congressional_district", "congressional-district"}:
        return "CONGRESSIONAL_DISTRICT"
    if level in {
        "state_legislative_district_upper",
        "state-legislative-district-upper",
    }:
        return "STATE_LEGISLATIVE_DISTRICT_UPPER"
    if level in {
        "state_legislative_district_lower",
        "state-legislative-district-lower",
    }:
        return "STATE_LEGISLATIVE_DISTRICT_LOWER"
    return level.upper() if level else None


def _json_numeric_value(value: Any) -> float:
    return arch_consumer_fact_numeric_value(value)


def _json_scalar_text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _arch_fact_stratum_name(row: sqlite3.Row) -> str:
    income_range = _json_object_value(row["filters_json"], "income_range")
    geography_name = row["geography_name"] or "US"
    if income_range is None:
        return str(geography_name)
    if income_range == "all":
        return f"{geography_name} All Filers"
    return f"{geography_name} Filers AGI {income_range}"


def _arch_fact_semantic_key(
    row: sqlite3.Row,
    constraints: tuple[tuple[str, str, str], ...],
) -> str:
    constraint_key = ",".join(
        f"{variable}{operator}{value}" for variable, operator, value in constraints
    )
    return "|".join(
        [
            "arch.semantic_fact.v1",
            str(row["measure_concept"]),
            str(row["domain"]),
            f"{row['period_value']}",
            f"{row['geography_level']}:{row['geography_id']}",
            constraint_key,
        ]
    )


def _json_object_value(raw: Any, key: str) -> Any:
    if raw is None:
        return None
    import json

    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get(key)


def _arch_target_display_label(record: ArchTargetRecord) -> str:
    measure_label = _arch_target_measure_label(record)
    scope_label = _arch_target_scope_label(record)
    source_label = _humanize_arch_source(record.source)
    suffix = (
        f" ({source_label}, {record.period})" if source_label else f" ({record.period})"
    )
    if scope_label:
        return f"{measure_label} for {scope_label}{suffix}"
    return f"{measure_label}{suffix}"


def _arch_target_measure_label(record: ArchTargetRecord) -> str:
    source_variable = str(record.variable)
    override = ARCH_VARIABLE_LABEL_OVERRIDES.get(source_variable)
    if override is not None:
        return override
    if record.target_type == "COUNT":
        for suffix in ("_returns", "_claims", "_count"):
            if source_variable.endswith(suffix):
                base = source_variable.removesuffix(suffix)
                return f"{_humanize_snake_label(base)} {suffix.removeprefix('_')}"
        return f"{_humanize_snake_label(source_variable)} count"
    if record.target_type == "AMOUNT":
        if source_variable.endswith("_amount"):
            return f"{_humanize_snake_label(source_variable.removesuffix('_amount'))} amount"
        return f"{_humanize_snake_label(source_variable)} amount"
    return _humanize_snake_label(source_variable)


def _arch_target_scope_label(record: ArchTargetRecord) -> str:
    if record.stratum_name:
        return str(record.stratum_name)
    constraint_labels = [
        label
        for constraint in record.constraints
        for label in [_arch_constraint_display_label(constraint)]
        if label
    ]
    if constraint_labels:
        return ", ".join(constraint_labels)
    jurisdiction = str(record.jurisdiction or "").strip()
    return jurisdiction.upper().replace("_", " ") if jurisdiction else ""


def _arch_constraint_display_label(
    constraint: tuple[str, str, str],
) -> str:
    variable, operator, value = constraint
    canonical_operator = _canonical_arch_constraint_operator(operator)
    value_text = str(value)
    if variable == "agi_bracket":
        return f"AGI {ARCH_AGI_BRACKET_LABELS.get(value_text, value_text)}"
    if variable == "is_tax_filer" and canonical_operator == "==":
        if _truthy_constraint_value(value_text):
            return "tax filers"
        if _falsey_constraint_value(value_text):
            return "non-filers"
    if variable == "state_fips" and canonical_operator == "==":
        return f"state FIPS {str(value_text).zfill(2)}"
    if variable == "congressional_district" and canonical_operator == "==":
        return f"congressional district {str(value_text).zfill(2)}"
    if variable == "sldu_id" and canonical_operator == "==":
        return f"state senate district {value_text}"
    if variable == "sldl_id" and canonical_operator == "==":
        return f"state house district {value_text}"
    positive_feature = ARCH_POSITIVE_CONSTRAINT_ALIASES.get(variable)
    if positive_feature is not None and canonical_operator == "==":
        feature_label = _humanize_snake_label(positive_feature)
        if _truthy_constraint_value(value_text):
            return f"{feature_label} > 0"
        if _falsey_constraint_value(value_text):
            return f"{feature_label} = 0"
    return f"{_humanize_snake_label(variable)} {canonical_operator} {value_text}"


def _truthy_constraint_value(value: str) -> bool:
    try:
        return float(str(value)) == 1.0
    except ValueError:
        return str(value).strip().lower() in {"true", "yes"}


def _falsey_constraint_value(value: str) -> bool:
    try:
        return float(str(value)) == 0.0
    except ValueError:
        return str(value).strip().lower() in {"false", "no"}


def _humanize_arch_source(source: str | None) -> str:
    if not source:
        return ""
    return _humanize_snake_label(str(source))


def _humanize_snake_label(value: str) -> str:
    words = [
        ARCH_LABEL_WORD_OVERRIDES.get(word.lower(), word.lower())
        for word in str(value).replace("-", "_").split("_")
        if word
    ]
    if not words:
        return ""
    label = " ".join(words)
    label = label[0].upper() + label[1:]
    return label.replace("Tax exempt", "Tax-exempt")


def _canonical_filters_for_arch_constraints(
    constraints: tuple[tuple[str, str, str], ...],
) -> tuple[TargetFilter, ...]:
    filters: list[TargetFilter] = []
    equalities = _constraint_equalities(constraints)
    for variable, operator, value in constraints:
        canonical_operator = _canonical_arch_constraint_operator(operator)
        if variable == "agi_bracket":
            filters.extend(_agi_bracket_filters(value))
            continue
        if variable == "congressional_district":
            geoid = _congressional_district_geoid(
                state_fips=equalities.get("state_fips"),
                district=value,
            )
            filters.append(
                TargetFilter(
                    feature="congressional_district_geoid",
                    operator=canonical_operator,
                    value=geoid or value,
                )
            )
            continue
        positive_feature = ARCH_POSITIVE_CONSTRAINT_ALIASES.get(variable)
        if positive_feature is not None:
            filters.append(
                _positive_support_filter_for_arch_constraint(
                    positive_feature,
                    operator=canonical_operator,
                    value=value,
                )
            )
            continue
        feature = ARCH_CONSTRAINT_VARIABLE_ALIASES.get(variable, variable)
        filters.append(
            TargetFilter(feature=feature, operator=canonical_operator, value=value)
        )
    return _dedupe_target_filters(filters)


def _target_filter_for_arch_geography(record: ArchTargetRecord) -> TargetFilter | None:
    geography_id = record.geography_id
    if geography_id is None:
        return None
    geo_level = _arch_record_geo_level(record)
    if geo_level == "state":
        return TargetFilter(
            feature="state_fips",
            operator="==",
            value=_state_fips_from_arch_geography_id(geography_id),
        )
    if geo_level == "county":
        return TargetFilter(
            feature="county_fips",
            operator="==",
            value=_county_fips_from_arch_geography_id(geography_id),
        )
    if geo_level == "district":
        return TargetFilter(
            feature="congressional_district_geoid",
            operator="==",
            value=_congressional_district_from_arch_geography_id(geography_id),
        )
    if geo_level == "sldu":
        return TargetFilter(
            feature="sldu_id",
            operator="==",
            value=_state_legislative_district_from_arch_geography_id(
                geography_id,
                chamber="upper",
            ),
        )
    if geo_level == "sldl":
        return TargetFilter(
            feature="sldl_id",
            operator="==",
            value=_state_legislative_district_from_arch_geography_id(
                geography_id,
                chamber="lower",
            ),
        )
    return None


def _state_fips_from_arch_geography_id(geography_id: Any) -> str:
    raw = str(geography_id)
    if raw.startswith("0400000US"):
        return raw[-2:]
    if raw.isdigit():
        return raw.zfill(2)
    return raw


def _county_fips_from_arch_geography_id(geography_id: Any) -> str:
    raw = str(geography_id)
    if raw.startswith("0500000US"):
        return raw[-5:]
    if raw.isdigit():
        return raw.zfill(5)
    return raw


def _congressional_district_from_arch_geography_id(geography_id: Any) -> str:
    raw = str(geography_id)
    if raw.startswith("5001800US"):
        return raw[-4:]
    return raw


ARCH_STATE_ABBR_BY_FIPS = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
    "72": "PR",
}


def _state_legislative_district_from_arch_geography_id(
    geography_id: Any,
    *,
    chamber: str,
) -> str:
    return normalize_state_legislative_district_id(
        geography_id, chamber=chamber
    ) or str(geography_id)


def _canonical_arch_constraint_operator(operator: str) -> str:
    value = str(operator).strip()
    return ARCH_CONSTRAINT_OPERATOR_ALIASES.get(value.lower(), value)


def _constraint_equalities(
    constraints: tuple[tuple[str, str, str], ...],
) -> dict[str, str]:
    return {
        variable: value
        for variable, operator, value in constraints
        if _canonical_arch_constraint_operator(operator) == "=="
    }


def _congressional_district_geoid(
    *,
    state_fips: str | None,
    district: str,
) -> str | None:
    try:
        district_id = str(int(str(district))).zfill(2)
    except ValueError:
        district_id = str(district)
    if len(district_id) >= 4:
        return district_id
    if state_fips is None:
        return None
    try:
        state_id = str(int(str(state_fips))).zfill(2)
    except ValueError:
        state_id = str(state_fips).zfill(2)
    return f"{state_id}{district_id}"


def _positive_support_filter_for_arch_constraint(
    feature: str,
    *,
    operator: str,
    value: str,
) -> TargetFilter:
    canonical_operator = _canonical_arch_constraint_operator(operator)
    if canonical_operator == "==":
        try:
            numeric_value = float(str(value))
        except ValueError:
            numeric_value = None
        if numeric_value == 1 or str(value).strip().lower() in {"true", "yes"}:
            return TargetFilter(feature=feature, operator=">", value=0)
        if numeric_value == 0 or str(value).strip().lower() in {"false", "no"}:
            return TargetFilter(feature=feature, operator="==", value=0)
    return TargetFilter(feature=feature, operator=canonical_operator, value=value)


def _dedupe_target_filters(filters: list[TargetFilter]) -> tuple[TargetFilter, ...]:
    seen: set[tuple[str, str, Any]] = set()
    deduped: list[TargetFilter] = []
    for target_filter in filters:
        operator = getattr(target_filter.operator, "value", target_filter.operator)
        key = (str(target_filter.feature), str(operator), str(target_filter.value))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target_filter)
    return tuple(deduped)


def _agi_bracket_filters(value: str) -> tuple[TargetFilter, ...]:
    bounds = ARCH_AGI_BRACKET_FILTERS.get(value)
    if bounds is None:
        return (TargetFilter(feature="agi_bracket", operator="==", value=value),)
    lower, upper = bounds
    filters: list[TargetFilter] = []
    if lower is not None:
        filters.append(
            TargetFilter(feature="adjusted_gross_income", operator=">=", value=lower)
        )
    if upper is not None:
        filters.append(
            TargetFilter(feature="adjusted_gross_income", operator="<", value=upper)
        )
    return tuple(filters)


def _positive_measure_for_count_record(source_variable: str) -> str | None:
    if source_variable.endswith("_returns"):
        amount_variable = f"{source_variable.removesuffix('_returns')}_amount"
    elif source_variable.endswith("_claims"):
        amount_variable = f"{source_variable.removesuffix('_claims')}_amount"
    else:
        return None
    return ARCH_AMOUNT_VARIABLE_ALIASES.get(amount_variable)


def _entity_for_measure(
    measure: str,
    entity_overrides: dict[str, Any],
) -> EntityType:
    override = entity_overrides.get(measure)
    if isinstance(override, EntityType):
        return override
    if override is not None:
        return EntityType(override)
    return ARCH_ENTITY_HINTS.get(measure, EntityType.TAX_UNIT)


def _matches_arch_provider_filters(
    record: ArchTargetRecord,
    *,
    variables: tuple[str, ...],
    domain_variables: tuple[str, ...],
    geo_levels: tuple[str, ...],
    target_cells: tuple[dict[str, Any], ...],
    entity_overrides: dict[str, Any] | None = None,
) -> bool:
    target: CanonicalTargetSpec | None = None
    if variables or domain_variables or target_cells:
        target = arch_target_record_to_canonical_spec(
            record,
            entity_overrides=entity_overrides or {},
        )
        if target is None:
            return False
    if variables and target is not None:
        candidate_variables = _arch_target_query_variables(record, target)
        if variables and candidate_variables.isdisjoint(variables):
            return False
    if domain_variables and target is not None:
        candidate_domain_variables = _arch_target_domain_variables(target)
        if candidate_domain_variables.isdisjoint(domain_variables):
            return False
    if geo_levels:
        geo_level = _arch_record_geo_level(record)
        if geo_level not in {_normalize_geo_level(str(level)) for level in geo_levels}:
            return False
    if target_cells and target is not None:
        if not any(_matches_arch_target_cell(target, cell) for cell in target_cells):
            return False
    return True


def _arch_target_query_variables(
    record: ArchTargetRecord,
    target: CanonicalTargetSpec,
) -> set[str]:
    variables = {
        record.variable,
        str(target.metadata.get("variable")),
    }
    if target.measure is not None:
        variables.add(str(target.measure))
    if target.aggregation is TargetAggregation.SUM:
        variables.update(_arch_target_cell_variables(target))
    return {variable for variable in variables if variable}


def _arch_target_cell_variables(target: CanonicalTargetSpec) -> set[str]:
    if target.aggregation is TargetAggregation.COUNT:
        if target.entity is EntityType.HOUSEHOLD:
            return {"household_count"}
        if target.entity is EntityType.PERSON:
            return {"person_count"}
        if target.entity is EntityType.SPM_UNIT:
            return {"spm_unit_count"}
        return {"tax_unit_count"}
    if target.measure is not None:
        variable = str(target.measure)
        return {variable, *ARCH_TARGET_CELL_VARIABLE_ALIASES.get(variable, ())}
    variable = target.metadata.get("variable")
    if variable is None:
        return set()
    variable = str(variable)
    return {variable, *ARCH_TARGET_CELL_VARIABLE_ALIASES.get(variable, ())}


def _arch_target_domain_variables(target: CanonicalTargetSpec) -> set[str]:
    domain_variables: set[str] = set()
    for target_filter in target.filters:
        feature = str(target_filter.feature)
        if feature in {
            "state_fips",
            "county_fips",
            "tract_geoid",
            "congressional_district_geoid",
            "sldu_id",
            "sldl_id",
            "program_payment_type",
            "tax_unit_is_filer",
        }:
            continue
        domain_variables.add(feature)
    variable = str(target.metadata.get("variable") or "")
    if (
        target.aggregation is TargetAggregation.COUNT
        and variable
        and variable not in _arch_target_cell_variables(target)
    ):
        domain_variables.add(variable)
    if (
        target.aggregation is TargetAggregation.SUM
        and variable in ARCH_SELF_DOMAIN_AMOUNT_VARIABLES
        and not domain_variables
    ):
        domain_variables.add(variable)
    return domain_variables


def _matches_arch_target_cell(
    target: CanonicalTargetSpec,
    raw_cell: dict[str, Any],
) -> bool:
    variable = raw_cell.get("variable")
    if variable is None or str(variable) not in _arch_target_cell_variables(target):
        return False

    target_geo_level = _normalize_geo_level(
        str(target.metadata.get("geo_level") or "national")
    )
    geo_level = raw_cell.get("geo_level")
    cell_geo_level = target_geo_level
    if geo_level is not None:
        cell_geo_level = _normalize_geo_level(str(geo_level))
        if target_geo_level != cell_geo_level:
            return False

    geographic_id = raw_cell.get("geographic_id")
    if geographic_id is not None:
        target_geographic_id = _arch_target_geographic_id(target)
        if target_geographic_id is None:
            return False
        if _normalize_target_cell_geographic_id(
            target_geographic_id,
            geo_level=target_geo_level,
        ) != _normalize_target_cell_geographic_id(
            geographic_id,
            geo_level=cell_geo_level,
        ):
            return False

    domain_variable = raw_cell.get("domain_variable")
    if "domain_variable" in raw_cell:
        target_domain_variables = _arch_target_domain_variables(target)
        cell_domain_variables = set(
            _split_target_cell_domain_variables(domain_variable)
        )
        if domain_variable is None or not cell_domain_variables:
            if _target_self_domain_is_redundant(target, target_domain_variables):
                return True
            return not target_domain_variables
        if not _target_domain_variables_match(
            target,
            target_domain_variables=target_domain_variables,
            cell_domain_variables=cell_domain_variables,
        ):
            return False

    return True


def _target_domain_variables_match(
    target: CanonicalTargetSpec,
    *,
    target_domain_variables: set[str],
    cell_domain_variables: set[str],
) -> bool:
    if cell_domain_variables == target_domain_variables:
        return True

    implied_domain_variables = _arch_target_implied_domain_variables(target)
    effective_target_domain_variables = (
        target_domain_variables | implied_domain_variables
    )
    if cell_domain_variables == effective_target_domain_variables:
        return True

    target_variables = _arch_target_cell_variables(target)
    if (
        target.aggregation is TargetAggregation.SUM
        and target_variables.issubset(ARCH_SELF_DOMAIN_AMOUNT_VARIABLES)
        and cell_domain_variables
        == effective_target_domain_variables | target_variables
    ):
        return True

    model_variable = str(target.metadata.get("variable") or "")
    if (
        target.aggregation is TargetAggregation.SUM
        and model_variable
        and model_variable in effective_target_domain_variables
        and cell_domain_variables
        == effective_target_domain_variables - {model_variable}
    ):
        return True

    if (
        target.aggregation is TargetAggregation.COUNT
        and model_variable
        and cell_domain_variables
        == effective_target_domain_variables - {model_variable}
    ):
        return True

    return False


def _arch_target_implied_domain_variables(
    target: CanonicalTargetSpec,
) -> set[str]:
    if str(target.source) != "IRS_SOI":
        return set()
    arch_variable = str(target.metadata.get("arch_variable") or "")
    if arch_variable in ARCH_IRS_SOI_CREDIT_AGI_DOMAIN_VARIABLES:
        return {"adjusted_gross_income"}
    if arch_variable in (
        ARCH_IRS_SOI_ITEMIZED_DEDUCTION_AMOUNT_VARIABLES
        | ARCH_IRS_SOI_ITEMIZED_DEDUCTION_COUNT_VARIABLES
    ):
        source_table = str(target.metadata.get("source_table") or "").lower()
        if any(
            marker in source_table
            for marker in ARCH_IRS_SOI_ITEMIZED_DEDUCTION_TABLE_MARKERS
        ):
            return {"tax_unit_itemizes"}
    return set()


def _target_self_domain_is_redundant(
    target: CanonicalTargetSpec,
    target_domain_variables: set[str],
) -> bool:
    if target.aggregation is not TargetAggregation.SUM:
        return False
    target_variables = _arch_target_cell_variables(target)
    return (
        len(target_domain_variables) == 1
        and target_domain_variables.issubset(target_variables)
        and target_domain_variables.issubset(ARCH_SELF_DOMAIN_AMOUNT_VARIABLES)
    )


def _coverage_for_arch_target_cell(
    cell_filter: dict[str, str | None],
    target_set: TargetSet,
) -> ArchTargetCellCoverage:
    matches = [
        target
        for target in target_set.targets
        if _matches_arch_target_cell(target, cell_filter)
    ]
    return ArchTargetCellCoverage(
        cell=dict(cell_filter),
        target_ids=tuple(
            int(target.metadata["target_id"])
            for target in matches
            if target.metadata.get("target_id") is not None
        ),
        target_names=tuple(str(target.name) for target in matches),
        sources=tuple(
            sorted({str(target.source) for target in matches if target.source})
        ),
    )


def _arch_gap_loaded_variable_catalog(
    provider: (
        ArchSQLiteTargetProvider
        | ArchFactSQLiteTargetProvider
        | ArchConsumerFactJSONLTargetProvider
        | ArchCompositeSQLiteTargetProvider
    ),
    *,
    period: int,
    jurisdiction: str | None,
    sources: tuple[str, ...],
    compose_model_year_targets: bool | None,
    age_soi_targets: bool | None,
) -> dict[tuple[str, str], set[str]]:
    resolved_jurisdiction = jurisdiction or provider.jurisdiction
    if isinstance(
        provider,
        (
            ArchFactSQLiteTargetProvider,
            ArchConsumerFactJSONLTargetProvider,
            ArchCompositeSQLiteTargetProvider,
        ),
    ):
        records = provider.load_records(period=period, sources=sources)
    else:
        resolved_compose = (
            provider.compose_model_year_targets
            if compose_model_year_targets is None
            else compose_model_year_targets
        )
        resolved_age_soi = (
            provider.age_soi_targets if age_soi_targets is None else age_soi_targets
        )
        records = (
            provider._compose_model_year_records(
                target_year=period,
                jurisdiction=resolved_jurisdiction,
                sources=sources,
                age_soi_targets=resolved_age_soi,
            )
            if resolved_compose
            else provider.load_records(
                period=period,
                jurisdiction=resolved_jurisdiction,
                sources=sources,
            )
        )
    catalog: dict[tuple[str, str], set[str]] = {}
    for record in records:
        key = (record.source, record.variable)
        catalog.setdefault(key, set()).add(_arch_record_geo_level(record))
    return catalog


def _arch_gap_queue_row_for_coverage_cell(
    coverage: ArchTargetCellCoverage,
    *,
    profile_name: str,
    period: int,
    loaded_variable_catalog: dict[tuple[str, str], set[str]],
    variable_uncovered_count: int,
) -> ArchTargetGapQueueRow:
    cell = coverage.cell
    expected_source = _arch_gap_expected_source(cell)
    expected_arch_variable = _arch_gap_expected_arch_variable(cell)
    expected_target_type = _arch_gap_expected_target_type(cell)
    expected_entity = _arch_gap_expected_entity(cell)
    expected_aggregation = _arch_gap_expected_aggregation(expected_target_type)
    loader_status = _arch_gap_loader_status(
        coverage,
        expected_source=expected_source,
        expected_arch_variable=expected_arch_variable,
        loaded_variable_catalog=loaded_variable_catalog,
        cell=cell,
    )
    gap_category = _arch_gap_category(
        cell,
        loader_status=loader_status,
        expected_source=expected_source,
        expected_arch_variable=expected_arch_variable,
    )
    return ArchTargetGapQueueRow(
        priority=0,
        profile_name=profile_name,
        period=int(period),
        variable=str(cell.get("variable") or ""),
        geo_level=cell.get("geo_level"),
        domain_variable=cell.get("domain_variable"),
        geographic_id=cell.get("geographic_id"),
        covered=coverage.covered,
        target_count=coverage.target_count,
        target_ids=coverage.target_ids,
        sources=coverage.sources,
        expected_source=expected_source,
        expected_source_table=_arch_gap_expected_source_table(
            expected_source,
            expected_arch_variable,
            cell,
        ),
        expected_arch_variable=expected_arch_variable,
        expected_target_type=expected_target_type,
        expected_entity=expected_entity,
        expected_aggregation=expected_aggregation,
        expected_filters=_arch_gap_expected_filters(cell),
        gap_category=gap_category,
        loader_status=loader_status,
        agent_task_kind=_arch_gap_agent_task_kind(gap_category),
        notes=_arch_gap_notes(
            cell,
            expected_source=expected_source,
            expected_arch_variable=expected_arch_variable,
            gap_category=gap_category,
            variable_uncovered_count=variable_uncovered_count,
        ),
    )


def _arch_gap_queue_sort_key(row: ArchTargetGapQueueRow) -> tuple[Any, ...]:
    source_rank = {
        "IRS_SOI": 0,
        "BEA": 1,
        "CENSUS_ACS": 2,
        "CMS_ACA": 3,
        "CMS_MEDICAID": 4,
        "CMS_MEDICARE": 5,
        "USDA_SNAP": 6,
        "SSA": 7,
        "HHS_ACF_TANF": 8,
        "HHS_ACF_LIHEAP": 9,
        "FEDERAL_RESERVE": 10,
    }.get(str(row.expected_source), 99)
    return (
        row.covered,
        row.loader_status == "needs_source_mapping_review",
        -_arch_gap_notes_uncovered_count(row.notes),
        source_rank,
        str(row.variable),
        str(row.geo_level or ""),
        str(row.domain_variable or ""),
    )


def _arch_gap_notes_uncovered_count(notes: str) -> int:
    if not notes.startswith("profile_variable_uncovered_count="):
        return 0
    raw_count = notes.split(";", 1)[0].split("=", 1)[1]
    try:
        return int(raw_count)
    except ValueError:
        return 0


def _arch_gap_expected_source(cell: dict[str, Any]) -> str | None:
    variable = str(cell.get("variable") or "")
    domain_variables = set(
        _split_target_cell_domain_variables(cell.get("domain_variable"))
    )
    if not domain_variables and variable in ARCH_BEA_FULL_POP_AMOUNT_VARIABLES:
        return "BEA"
    if variable == "tax_unit_count" and "aca_ptc" in domain_variables:
        return "IRS_SOI"
    if variable == "snap" or "snap" in domain_variables:
        return "USDA_SNAP"
    if variable == "tanf" or "tanf" in domain_variables:
        return "HHS_ACF_TANF"
    if "spm_unit_energy_subsidy_reported" in domain_variables:
        return "HHS_ACF_LIHEAP"
    if variable == "aca_ptc" or "aca_ptc" in domain_variables:
        return "CMS_ACA"
    if variable == "medicaid" or "medicaid_enrolled" in domain_variables:
        return "CMS_MEDICAID"
    if variable == "ssi" or variable.startswith("social_security"):
        return "SSA"
    if variable == "state_income_tax":
        return "CENSUS_STC"
    if variable == "medicare_part_b_premiums":
        return "CMS_MEDICARE"
    if variable == "net_worth":
        return "FEDERAL_RESERVE"
    if variable == "person_count":
        if _normalize_geo_level(cell.get("geo_level")) in {"sldu", "sldl"}:
            return "CENSUS_DECENNIAL"
        if "adjusted_gross_income" in domain_variables:
            return "IRS_SOI"
        if "age" in domain_variables or not domain_variables:
            return "CENSUS_PEP"
        return None
    if variable == "household_count":
        if _normalize_geo_level(cell.get("geo_level")) in {"sldu", "sldl"}:
            return "CENSUS_DECENNIAL"
        if not domain_variables:
            return "CENSUS_ACS"
        return None
    if variable in ARCH_IRS_SOI_GAP_VARIABLES:
        return "IRS_SOI"
    if domain_variables & ARCH_IRS_SOI_GAP_VARIABLES:
        return "IRS_SOI"
    return None


def _arch_gap_expected_arch_variable(cell: dict[str, Any]) -> str | None:
    variable = str(cell.get("variable") or "")
    domain_variables = tuple(
        _split_target_cell_domain_variables(cell.get("domain_variable"))
    )
    domain_variable = domain_variables[0] if len(domain_variables) == 1 else None
    if not domain_variables and variable in ARCH_BEA_FULL_POP_AMOUNT_ARCH_VARIABLES:
        return ARCH_BEA_FULL_POP_AMOUNT_ARCH_VARIABLES[variable]
    if variable == "tax_unit_count":
        if set(domain_variables) in (
            {"eitc_child_count"},
            {"adjusted_gross_income", "eitc", "eitc_child_count"},
        ):
            return "eitc_claims"
        if {
            "adjusted_gross_income",
            "income_tax_before_credits",
        }.issubset(domain_variables):
            return "income_tax_before_credits_returns"
        if set(domain_variables) == {"aca_ptc"}:
            return "aca_ptc_returns"
        itemized_domain_variables = set(domain_variables) - {"tax_unit_itemizes"}
        if (
            "tax_unit_itemizes" in domain_variables
            and len(itemized_domain_variables) == 1
        ):
            return ARCH_MODEL_COUNT_DOMAIN_VARIABLE_HINTS.get(
                next(iter(itemized_domain_variables))
            )
        if domain_variable is None:
            return "tax_unit_count" if not domain_variables else None
        return ARCH_MODEL_COUNT_DOMAIN_VARIABLE_HINTS.get(domain_variable)
    if variable == "household_count":
        if domain_variable == "snap":
            return "snap_household_count"
        if domain_variable == "spm_unit_energy_subsidy_reported":
            return "liheap_household_count"
        return "household_count" if domain_variable is None else None
    if variable == "spm_unit_count":
        if domain_variable == "tanf":
            return "tanf_family_count"
        return None
    if variable == "person_count":
        if domain_variable == "snap":
            return "snap_participant_count"
        if domain_variable == "aca_ptc":
            return "aca_marketplace_enrollment"
        if domain_variable == "medicaid_enrolled":
            return "medicaid_total_enrollment"
        if domain_variable == "adjusted_gross_income":
            return "tax_filer_individual_count"
        if domain_variable == "age" or not domain_variables:
            return "population"
        return None
    if variable == "snap":
        return "snap_benefits"
    if variable == "aca_ptc":
        return "aca_aptc_amount"
    if variable == "medicaid":
        return "medicaid_benefits"
    if variable == "tanf":
        return "tanf_cash_assistance"
    if variable == "state_income_tax":
        return "state_individual_income_tax_collections"
    return ARCH_MODEL_AMOUNT_VARIABLE_HINTS.get(variable)


def _arch_gap_expected_target_type(cell: dict[str, Any]) -> str | None:
    variable = str(cell.get("variable") or "")
    if variable in {
        "household_count",
        "person_count",
        "spm_unit_count",
        "tax_unit_count",
    }:
        return "COUNT"
    if _arch_gap_expected_arch_variable(cell) is not None:
        return "AMOUNT"
    return None


def _arch_gap_expected_entity(cell: dict[str, Any]) -> str | None:
    variable = str(cell.get("variable") or "")
    if variable == "tax_unit_count":
        return EntityType.TAX_UNIT.value
    if variable == "person_count":
        return EntityType.PERSON.value
    if variable == "spm_unit_count":
        return EntityType.SPM_UNIT.value
    if variable in {"household_count", "snap"}:
        return EntityType.HOUSEHOLD.value
    entity = ARCH_ENTITY_HINTS.get(variable)
    return entity.value if entity is not None else None


def _arch_gap_expected_aggregation(target_type: str | None) -> str | None:
    if target_type == "COUNT":
        return "count"
    if target_type == "AMOUNT":
        return "sum"
    return None


def _arch_gap_expected_filters(cell: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    filters: list[dict[str, Any]] = []
    geo_level = _normalize_geo_level(cell.get("geo_level"))
    geographic_id = cell.get("geographic_id")
    if geo_level == "state":
        filters.append(
            {
                "kind": "geography",
                "feature": "state_fips",
                "operator": "==",
                "value": (
                    _state_fips_from_arch_geography_id(geographic_id)
                    if geographic_id is not None
                    else "<state_fips>"
                ),
            }
        )
    if geo_level == "sldu":
        filters.append(
            {
                "kind": "geography",
                "feature": "sldu_id",
                "operator": "==",
                "value": (
                    _normalize_target_cell_geographic_id(
                        geographic_id,
                        geo_level=geo_level,
                    )
                    if geographic_id is not None
                    else "<sldu_id>"
                ),
            }
        )
    if geo_level == "sldl":
        filters.append(
            {
                "kind": "geography",
                "feature": "sldl_id",
                "operator": "==",
                "value": (
                    _normalize_target_cell_geographic_id(
                        geographic_id,
                        geo_level=geo_level,
                    )
                    if geographic_id is not None
                    else "<sldl_id>"
                ),
            }
        )
    for domain_variable in _split_target_cell_domain_variables(
        cell.get("domain_variable")
    ):
        filters.append(
            {
                "kind": "domain",
                "feature": domain_variable,
                "operator": ">",
                "value": 0,
            }
        )
    return tuple(filters)


def _arch_gap_expected_source_table(
    expected_source: str | None,
    expected_arch_variable: str | None,
    cell: dict[str, Any],
) -> str | None:
    variable = str(cell.get("variable") or "")
    if expected_source == "BEA":
        geo_level = _normalize_geo_level(cell.get("geo_level"))
        if geo_level == "state" and expected_arch_variable in {
            "proprietors_income_amount",
            "wages_salaries_amount",
        }:
            return "BEA Regional SAINC5N annual state personal income"
        if expected_arch_variable == "wages_salaries_amount":
            return "BEA NIPA annual total wages and salaries"
        if expected_arch_variable in {
            "medicaid_benefits",
            "personal_dividend_income_amount",
            "proprietors_income_amount",
            "rental_income_amount",
            "social_security_benefits",
            "unemployment_insurance_benefits",
        }:
            return "BEA NIPA annual personal income components"
        return "BEA NIPA or Regional personal income tables"
    if expected_arch_variable in ARCH_GAP_SOURCE_TABLE_HINTS:
        return ARCH_GAP_SOURCE_TABLE_HINTS[expected_arch_variable]
    if variable in ARCH_GAP_SOURCE_TABLE_HINTS:
        return ARCH_GAP_SOURCE_TABLE_HINTS[variable]
    if expected_source == "IRS_SOI":
        if expected_arch_variable and (
            expected_arch_variable.startswith("wages_salaries_")
            or expected_arch_variable.startswith("net_capital_gains_")
            or expected_arch_variable.startswith("taxable_ira_distributions_")
            or expected_arch_variable.startswith("taxable_pension_income_")
            or expected_arch_variable.startswith("taxable_social_security_")
            or expected_arch_variable.startswith("unemployment_compensation_")
        ):
            return "IRS SOI Publication 1304 Table 1.4"
        if expected_arch_variable and (
            expected_arch_variable.endswith("_claims")
            or expected_arch_variable
            in {"real_estate_taxes_amount", "real_estate_taxes_claims"}
        ):
            return "IRS SOI itemized deduction or credit tables"
        return "IRS SOI Publication 1304"
    if expected_source == "CENSUS_ACS":
        return "Census ACS summary tables"
    if expected_source == "CENSUS_DECENNIAL":
        return "Census 2020 CD119 state legislative district summary file"
    if expected_source == "CENSUS_PEP":
        return "Census Population Estimates Program age-sex files"
    if expected_source == "CENSUS_STC":
        return "Census State Tax Collections item T40"
    if expected_source == "CMS_ACA":
        return "CMS Marketplace Open Enrollment public-use files"
    if expected_source == "CMS_MEDICAID":
        return "CMS Medicaid enrollment and expenditure reports"
    if expected_source == "CMS_MEDICARE":
        return "CMS Medicare Trustees Report Part B premium income"
    if expected_source == "FEDERAL_RESERVE":
        return "Federal Reserve Financial Accounts Z.1 household net worth"
    if expected_source == "SSA":
        return "SSA Annual Statistical Supplement"
    if expected_source == "HHS_ACF_TANF":
        return "ACF TANF Financial Data"
    if expected_source == "HHS_ACF_LIHEAP":
        return "HHS ACF LIHEAP National Profile"
    return None


def _arch_gap_loader_status(
    coverage: ArchTargetCellCoverage,
    *,
    expected_source: str | None,
    expected_arch_variable: str | None,
    loaded_variable_catalog: dict[tuple[str, str], set[str]],
    cell: dict[str, Any],
) -> str:
    if coverage.covered:
        return "covered"
    if expected_source is None or expected_arch_variable is None:
        return "needs_source_mapping_review"
    loaded_geo_levels = loaded_variable_catalog.get(
        (expected_source, expected_arch_variable)
    )
    if loaded_geo_levels:
        expected_geo_level = _normalize_geo_level(cell.get("geo_level"))
        if expected_geo_level not in loaded_geo_levels:
            return "loaded_arch_variable_missing_geography"
        return "loaded_arch_variable_missing_filter_or_adapter"
    return "missing_arch_target_record"


def _arch_gap_category(
    cell: dict[str, Any],
    *,
    loader_status: str,
    expected_source: str | None,
    expected_arch_variable: str | None,
) -> str:
    if loader_status == "covered":
        return "covered"
    if _arch_gap_is_deprioritized_survey_or_model_input(cell):
        return "survey_or_model_input_deprioritized"
    if loader_status == "missing_arch_target_record":
        return "ready_primary_loader"
    if loader_status == "loaded_arch_variable_missing_geography":
        return "ready_rollup_or_geography"
    if loader_status == "loaded_arch_variable_missing_filter_or_adapter":
        return "adapter_or_constraint_review"
    if expected_source is None or expected_arch_variable is None:
        return "source_mapping_review"
    return "source_mapping_review"


def _arch_gap_is_deprioritized_survey_or_model_input(cell: dict[str, Any]) -> bool:
    variable = str(cell.get("variable") or "")
    if variable in ARCH_DEPRIORITIZED_SURVEY_OR_MODEL_GAP_VARIABLES:
        return True
    domain_variables = set(
        _split_target_cell_domain_variables(cell.get("domain_variable"))
    )
    return bool(domain_variables & ARCH_DEPRIORITIZED_SURVEY_OR_MODEL_GAP_DOMAINS)


def _arch_gap_agent_task_kind(gap_category: str) -> str:
    if gap_category == "covered":
        return "none"
    if gap_category == "survey_or_model_input_deprioritized":
        return "defer_or_review_non_primary_source"
    if gap_category == "ready_rollup_or_geography":
        return "add_arch_rollup_or_geography_records"
    if gap_category == "adapter_or_constraint_review":
        return "review_adapter_or_constraints"
    if gap_category == "ready_primary_loader":
        return "add_arch_source_loader_or_target_record"
    return "review_source_mapping"


def _arch_gap_notes(
    cell: dict[str, Any],
    *,
    expected_source: str | None,
    expected_arch_variable: str | None,
    gap_category: str,
    variable_uncovered_count: int,
) -> str:
    parts = [f"profile_variable_uncovered_count={variable_uncovered_count}"]
    if gap_category == "survey_or_model_input_deprioritized":
        parts.append(
            "survey/model-input proxy deprioritized until primary source review"
        )
    if expected_source is None:
        parts.append("expected_source requires review")
    if expected_arch_variable is None:
        parts.append("expected Arch variable requires review")
    if "," in str(cell.get("domain_variable") or ""):
        parts.append("multi-domain cells may need a grouped source-record spec")
    return "; ".join(parts)


def _arch_target_gap_queue_csv(report: ArchTargetGapQueueReport) -> str:
    import csv
    import io
    import json

    fieldnames = [
        "priority",
        "profile_name",
        "period",
        "variable",
        "geo_level",
        "domain_variable",
        "geographic_id",
        "covered",
        "target_count",
        "target_ids",
        "sources",
        "expected_source",
        "expected_source_table",
        "expected_arch_variable",
        "expected_target_type",
        "expected_entity",
        "expected_aggregation",
        "expected_filters",
        "gap_category",
        "loader_status",
        "agent_task_kind",
        "notes",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in report.rows:
        writer.writerow(
            {
                "priority": row.priority,
                "profile_name": row.profile_name,
                "period": row.period,
                "variable": row.variable,
                "geo_level": row.geo_level,
                "domain_variable": row.domain_variable,
                "geographic_id": row.geographic_id,
                "covered": row.covered,
                "target_count": row.target_count,
                "target_ids": json.dumps(list(row.target_ids)),
                "sources": json.dumps(list(row.sources)),
                "expected_source": row.expected_source,
                "expected_source_table": row.expected_source_table,
                "expected_arch_variable": row.expected_arch_variable,
                "expected_target_type": row.expected_target_type,
                "expected_entity": row.expected_entity,
                "expected_aggregation": row.expected_aggregation,
                "expected_filters": json.dumps(list(row.expected_filters)),
                "gap_category": row.gap_category,
                "loader_status": row.loader_status,
                "agent_task_kind": row.agent_task_kind,
                "notes": row.notes,
            }
        )
    return buffer.getvalue()


def _summarize_arch_cell_coverage(
    coverage_cells: tuple[ArchTargetCellCoverage, ...],
    *,
    field: str,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for coverage in coverage_cells:
        raw_value = coverage.cell.get(field)
        value = (
            _normalize_geo_level(raw_value)
            if field == "geo_level"
            else str(raw_value or "")
        )
        if not value:
            value = "none"
        item = summary.setdefault(
            value,
            {
                "target_cell_count": 0,
                "covered_cell_count": 0,
                "uncovered_cell_count": 0,
            },
        )
        item["target_cell_count"] += 1
        if coverage.covered:
            item["covered_cell_count"] += 1
        else:
            item["uncovered_cell_count"] += 1
    return dict(sorted(summary.items()))


def _target_cell_to_provider_filter(
    cell: PolicyEngineUSTargetCell | dict[str, Any],
) -> dict[str, str | None]:
    if isinstance(cell, PolicyEngineUSTargetCell):
        return cell.to_provider_filter()
    return {
        "variable": cell.get("variable"),
        "geo_level": cell.get("geo_level"),
        "domain_variable": cell.get("domain_variable"),
        "geographic_id": cell.get("geographic_id"),
    }


def _arch_target_geographic_id(target: CanonicalTargetSpec) -> str | None:
    geo_level = str(target.metadata.get("geo_level") or "national").lower()
    feature_by_level = {
        "state": "state_fips",
        "county": "county_fips",
        "tract": "tract_geoid",
        "district": "congressional_district_geoid",
        "congressional_district": "congressional_district_geoid",
        "sldu": "sldu_id",
        "sldl": "sldl_id",
    }
    feature = feature_by_level.get(geo_level)
    if feature is None:
        return None
    for target_filter in target.filters:
        if str(target_filter.feature) != feature:
            continue
        operator = getattr(target_filter.operator, "value", target_filter.operator)
        if _canonical_arch_constraint_operator(str(operator)) == "==":
            return str(target_filter.value)
    return None


def _split_target_cell_domain_variables(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(
        _normalize_target_cell_domain_variable(part)
        for part in str(value).split(",")
        if part.strip()
    )


def _normalize_target_cell_domain_variable(value: Any) -> str:
    raw = str(value).strip()
    return ARCH_POSITIVE_CONSTRAINT_ALIASES.get(raw, raw)


def _normalize_target_cell_geographic_id(
    value: Any,
    *,
    geo_level: str | None = None,
) -> str:
    raw = str(value)
    normalized_geo_level = _normalize_geo_level(geo_level)
    chamber = None
    if normalized_geo_level == "sldu":
        chamber = "upper"
    elif normalized_geo_level == "sldl":
        chamber = "lower"
    normalized_sld = normalize_state_legislative_district_id(raw, chamber=chamber)
    if normalized_sld != raw:
        return str(normalized_sld)
    try:
        return str(int(raw))
    except (TypeError, ValueError):
        return raw


def _arch_record_composition_key(
    record: ArchTargetRecord,
) -> tuple[str, str, str, tuple[tuple[str, str, str], ...]]:
    return (
        record.variable,
        record.target_type,
        _arch_record_geo_level(record),
        tuple(sorted(record.constraints)),
    )


def _arch_record_geo_level(record: ArchTargetRecord) -> str:
    return _geo_level_for_constraints(record.constraints) or _normalize_geo_level(
        record.geographic_level
    )


def _geo_level_for_constraints(
    constraints: tuple[tuple[str, str, str], ...],
) -> str | None:
    constraint_variables = {variable for variable, _, _ in constraints}
    for variable, geo_level in (
        ("tract_geoid", "tract"),
        ("county_fips", "county"),
        ("congressional_district", "district"),
        ("congressional_district_geoid", "district"),
        ("sldu_id", "sldu"),
        ("sldl_id", "sldl"),
        ("state_fips", "state"),
    ):
        if variable in constraint_variables:
            return geo_level
    return None


def _normalize_arch_source(source: str) -> str:
    value = str(source)
    return ARCH_SOURCE_ALIASES.get(value.lower(), value.upper().replace("-", "_"))


def _normalize_geo_level(geo_level: str | None) -> str:
    if not geo_level:
        return "national"
    normalized = geo_level.lower()
    if normalized in {"congressional_district", "congressional-district"}:
        return "district"
    if normalized in {
        "sldu",
        "state_legislative_district_upper",
        "state-legislative-district-upper",
        "state_senate_district",
        "state-senate-district",
    }:
        return "sldu"
    if normalized in {
        "sldl",
        "state_legislative_district_lower",
        "state-legislative-district-lower",
        "state_house_district",
        "state-house-district",
    }:
        return "sldl"
    return normalized


def _sqlite_table_has_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
) -> bool:
    return column in _sqlite_table_columns(conn, table)


def _sqlite_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    names: set[str] = set()
    for row in conn.execute(f"PRAGMA table_info({table})"):
        names.add(str(row["name"] if isinstance(row, sqlite3.Row) else row[1]))
    return names


def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _looks_like_arch_consumer_fact_jsonl(path: Path) -> bool:
    return path.suffix.lower() in {".jsonl", ".ndjson"}


def _as_arch_db_path_tuple(
    value: str | Path | tuple[str | Path, ...],
) -> tuple[Path, ...]:
    if isinstance(value, (str, Path)):
        return (Path(value),)
    paths = tuple(Path(path) for path in value)
    if not paths:
        raise ValueError("At least one Arch targets DB path is required")
    return paths


def _single_or_many_paths(paths: list[str]) -> str | tuple[str, ...]:
    return paths[0] if len(paths) == 1 else tuple(paths)


def _default_arch_target_artifact_roots() -> tuple[Path, ...]:
    candidates = (
        Path.cwd() / "artifacts",
        Path.cwd().parent / "arch",
        Path("/tmp"),
    )
    return tuple(path for path in candidates if path.exists())


def discover_arch_target_artifacts(
    roots: tuple[str | Path, ...],
    *,
    max_depth: int = 6,
) -> tuple[Path, ...]:
    """Find local Arch target artifacts under bounded discovery roots."""

    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if root.is_file():
            candidates = (root,)
        elif root.is_dir():
            candidates = tuple(_walk_arch_target_artifact_candidates(root, max_depth))
        else:
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not _is_arch_target_artifact(resolved):
                continue
            discovered.append(resolved)
            seen.add(resolved)
    return tuple(sorted(discovered, key=lambda path: str(path)))


def _walk_arch_target_artifact_candidates(
    root: Path, max_depth: int
) -> tuple[Path, ...]:
    import os

    skip_dir_names = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "site-packages",
    }
    candidates: list[Path] = []
    root = root.resolve()
    for directory, dirnames, filenames in os.walk(root):
        current = Path(directory)
        try:
            depth = len(current.relative_to(root).parts)
        except ValueError:
            depth = 0
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [
                dirname for dirname in dirnames if dirname not in skip_dir_names
            ]
        for filename in filenames:
            candidate = current / filename
            if _is_arch_target_artifact_candidate_name(candidate):
                candidates.append(candidate)
    return tuple(candidates)


def _is_arch_target_artifact_candidate_name(path: Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in {"consumer_facts.jsonl", "consumer_facts.ndjson"}:
        return True
    if suffix not in {".db", ".sqlite", ".sqlite3"}:
        return False
    return name == "targets.db" or "arch_targets" in name


def _is_arch_target_artifact(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return _is_arch_consumer_fact_jsonl(path)
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        return _is_arch_sqlite_artifact(path)
    return False


def _is_arch_consumer_fact_jsonl(path: Path) -> bool:
    try:
        with path.open() as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                schema_version = str(row.get("schema_version") or "")
                return schema_version.startswith("arch.consumer_fact") or (
                    "aggregate_fact_key" in row and "observed_measure" in row
                )
    except (OSError, json.JSONDecodeError):
        return False
    return False


def _is_arch_sqlite_artifact(path: Path) -> bool:
    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error:
        return False
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "aggregate_facts" in tables:
            return True
        if not {"targets", "strata", "stratum_constraints"}.issubset(tables):
            return False
        target_columns = _sqlite_table_columns(conn, "targets")
        required_target_columns = {
            "id",
            "stratum_id",
            "variable",
            "period",
            "value",
            "target_type",
            "geographic_level",
            "source",
        }
        return required_target_columns.issubset(target_columns)
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _filename_slug(value: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in value)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug.lower() or "profile"


def _arch_target_refresh_summary_markdown(
    coverage: ArchTargetProfileCoverageReport,
    gaps: ArchTargetGapQueueReport,
    *,
    artifact_paths: tuple[Path, ...],
    output_paths: tuple[Path, ...],
) -> str:
    lines = [
        "# Arch Target Coverage Snapshot",
        "",
        f"- Profile: `{coverage.profile_name}`",
        f"- Period: `{coverage.period}`",
        f"- Target cells: `{coverage.target_cell_count}`",
        f"- Covered cells: `{coverage.covered_cell_count}`",
        f"- Uncovered cells: `{coverage.uncovered_cell_count}`",
        f"- Coverage rate: `{coverage.coverage_rate:.1%}`",
        "",
        "## Coverage By Geography",
        "",
        "| Geography | Target cells | Covered | Uncovered |",
        "| --- | ---: | ---: | ---: |",
    ]
    for geo_level, counts in sorted(coverage.by_geo_level.items()):
        lines.append(
            "| {geo_level} | {target_cell_count} | {covered_cell_count} | "
            "{uncovered_cell_count} |".format(geo_level=geo_level, **counts)
        )
    lines.extend(
        [
            "",
            "## Gap Categories",
            "",
            "| Category | Rows |",
            "| --- | ---: |",
        ]
    )
    for category, count in sorted(gaps.by_gap_category.items()):
        lines.append(f"| `{category}` | {count} |")
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            *(f"- `{path}`" for path in artifact_paths),
            "",
            "## Outputs",
            "",
            *(f"- `{path}`" for path in output_paths),
            "",
        ]
    )
    return "\n".join(lines)


def _target_filter_tuple(
    target: CanonicalTargetSpec,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (
                str(target_filter.feature),
                str(getattr(target_filter.operator, "value", target_filter.operator)),
                _json_scalar_text(target_filter.value),
            )
            for target_filter in target.filters
        )
    )


def _jurisdiction_clause(jurisdiction: str) -> str:
    normalized = jurisdiction.upper().replace("-", "_")
    if normalized == "US":
        return "upper(s.jurisdiction) LIKE 'US%'"
    return f"upper(s.jurisdiction) = '{normalized}'"


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _as_target_cell_filters(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        return (dict(value),)
    return tuple(dict(item) for item in value if item is not None)


__all__ = [
    "ArchCompositeSQLiteTargetProvider",
    "ArchConsumerFactJSONLTargetProvider",
    "ArchFactSQLiteTargetProvider",
    "ArchTargetCellCoverage",
    "ArchTargetGapQueueReport",
    "ArchTargetGapQueueRow",
    "ArchTargetParityReport",
    "ArchTargetParityRow",
    "ArchTargetProfileCoverageReport",
    "ArchSQLiteTargetProvider",
    "ArchTargetRecord",
    "SOIAgingFactors",
    "arch_target_record_to_canonical_spec",
    "resolve_arch_sqlite_target_provider",
    "summarize_arch_target_gap_queue",
    "summarize_arch_target_parity",
    "summarize_arch_target_profile_coverage",
]
