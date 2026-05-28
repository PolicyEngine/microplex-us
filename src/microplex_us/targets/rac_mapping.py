"""
RAC Variable Mapping

Maps calibration target variables to PolicyEngine RAC (statute) definitions.
Enables validation of microdata against encoded tax law.
"""

from dataclasses import dataclass


@dataclass
class RACVariable:
    """A variable defined in PolicyEngine RAC."""
    name: str
    statute: str           # e.g., "26/62" for IRC Section 62
    description: str
    entity: str            # Person, TaxUnit, Household
    dtype: str             # Money, Rate, Boolean, Integer
    period: str            # Year, Month


# Map from target variable names to RAC definitions
# Based on policyengine-us/statute structure
RAC_VARIABLE_MAP: dict[str, RACVariable] = {
    # Income (IRC Section 61 - Gross Income)
    "adjusted_gross_income": RACVariable(
        name="adjusted_gross_income",
        statute="26/62",
        description="Adjusted Gross Income (AGI) per IRC Section 62",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "gross_income": RACVariable(
        name="gross_income",
        statute="26/61",
        description="Gross income per IRC Section 61",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "employment_income": RACVariable(
        name="employment_income",
        statute="26/61/a/1",
        description="Compensation for services (wages, salaries, tips)",
        entity="Person",
        dtype="Money",
        period="Year",
    ),
    "self_employment_income": RACVariable(
        name="self_employment_income",
        statute="26/1402",
        description="Net earnings from self-employment",
        entity="Person",
        dtype="Money",
        period="Year",
    ),
    "interest_income": RACVariable(
        name="interest_income",
        statute="26/61/a/4",
        description="Interest income",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "dividend_income": RACVariable(
        name="dividend_income",
        statute="26/61/a/7",
        description="Dividends (ordinary and qualified)",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "qualified_dividend_income": RACVariable(
        name="qualified_dividend_income",
        statute="26/1/h/11",
        description="Qualified dividends taxed at capital gains rates",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "rental_income": RACVariable(
        name="rental_income",
        statute="26/61/a/5",
        description="Rents and royalties",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "capital_gains": RACVariable(
        name="capital_gains",
        statute="26/1222",
        description="Net capital gain",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "social_security_income": RACVariable(
        name="social_security_income",
        statute="26/86",
        description="Social security benefits (portion taxable)",
        entity="Person",
        dtype="Money",
        period="Year",
    ),
    "pension_income": RACVariable(
        name="pension_income",
        statute="26/72",
        description="Annuities and pension distributions",
        entity="Person",
        dtype="Money",
        period="Year",
    ),
    "partnership_s_corp_income": RACVariable(
        name="partnership_s_corp_income",
        statute="26/702",
        description="Partnership and S-corporation income",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "unemployment_compensation": RACVariable(
        name="unemployment_compensation",
        statute="26/85",
        description="Unemployment compensation",
        entity="Person",
        dtype="Money",
        period="Year",
    ),

    # Deductions (IRC Section 63)
    "standard_deduction": RACVariable(
        name="standard_deduction",
        statute="26/63/c",
        description="Standard deduction amount",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "itemized_deductions": RACVariable(
        name="itemized_deductions",
        statute="26/63/d",
        description="Total itemized deductions",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "charitable_deduction": RACVariable(
        name="charitable_deduction",
        statute="26/170",
        description="Charitable contribution deduction",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "salt_deduction": RACVariable(
        name="salt_deduction",
        statute="26/164",
        description="State and local tax deduction",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "mortgage_interest_deduction": RACVariable(
        name="mortgage_interest_deduction",
        statute="26/163/h",
        description="Home mortgage interest deduction",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "medical_expense_deduction": RACVariable(
        name="medical_expense_deduction",
        statute="26/213",
        description="Medical and dental expense deduction",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "qbi_deduction": RACVariable(
        name="qbi_deduction",
        statute="26/199A",
        description="Qualified business income deduction",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),

    # Tax computation (IRC Sections 1, 55)
    "taxable_income": RACVariable(
        name="taxable_income",
        statute="26/63",
        description="Taxable income (AGI minus deductions)",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "income_tax_before_credits": RACVariable(
        name="income_tax_before_credits",
        statute="26/1",
        description="Regular tax on taxable income",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "alternative_minimum_tax": RACVariable(
        name="alternative_minimum_tax",
        statute="26/55",
        description="Alternative minimum tax",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),

    # Credits (IRC Sections 21-54)
    "earned_income_credit": RACVariable(
        name="earned_income_credit",
        statute="26/32",
        description="Earned Income Tax Credit (EITC)",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "child_tax_credit": RACVariable(
        name="child_tax_credit",
        statute="26/24",
        description="Child Tax Credit (CTC)",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "additional_child_tax_credit": RACVariable(
        name="additional_child_tax_credit",
        statute="26/24/h",
        description="Additional (refundable) Child Tax Credit",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "child_care_credit": RACVariable(
        name="child_care_credit",
        statute="26/21",
        description="Child and Dependent Care Credit",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "education_credit": RACVariable(
        name="education_credit",
        statute="26/25A",
        description="American Opportunity and Lifetime Learning Credits",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),
    "premium_tax_credit": RACVariable(
        name="premium_tax_credit",
        statute="26/36B",
        description="Premium Tax Credit (ACA)",
        entity="TaxUnit",
        dtype="Money",
        period="Year",
    ),

    # Benefits (Title 7 - SNAP, Title 42 - SSI/Medicaid)
    "snap_benefit": RACVariable(
        name="snap_benefit",
        statute="7/2017",
        description="SNAP (food stamps) benefit amount",
        entity="Household",
        dtype="Money",
        period="Month",
    ),
    "medicaid_eligible": RACVariable(
        name="medicaid_eligible",
        statute="42/1396a",
        description="Medicaid eligibility",
        entity="Person",
        dtype="Boolean",
        period="Year",
    ),
    "ssi_benefit": RACVariable(
        name="ssi_benefit",
        statute="42/1382",
        description="Supplemental Security Income",
        entity="Person",
        dtype="Money",
        period="Month",
    ),
    "tanf_benefit": RACVariable(
        name="tanf_benefit",
        statute="42/601",
        description="TANF cash assistance",
        entity="Household",
        dtype="Money",
        period="Month",
    ),
    "housing_subsidy": RACVariable(
        name="housing_subsidy",
        statute="42/1437f",
        description="Section 8 housing assistance",
        entity="Household",
        dtype="Money",
        period="Month",
    ),
    "wic": RACVariable(
        name="wic",
        statute="42/1786",
        description="Women, Infants, and Children program",
        entity="Person",
        dtype="Money",
        period="Month",
    ),
    "school_lunch": RACVariable(
        name="school_lunch",
        statute="42/1758",
        description="National School Lunch Program",
        entity="Person",
        dtype="Money",
        period="Year",
    ),
    "liheap": RACVariable(
        name="liheap",
        statute="42/8621",
        description="Low Income Home Energy Assistance Program",
        entity="Household",
        dtype="Money",
        period="Year",
    ),
    "ccdf": RACVariable(
        name="ccdf",
        statute="42/9858",
        description="Child Care and Development Fund",
        entity="Person",
        dtype="Money",
        period="Month",
    ),

    # Demographics (not statutory but needed for calibration)
    "age": RACVariable(
        name="age",
        statute=None,
        description="Age in years",
        entity="Person",
        dtype="Integer",
        period="Year",
    ),
    "is_tax_filer": RACVariable(
        name="is_tax_filer",
        statute="26/6012",
        description="Required to file tax return",
        entity="TaxUnit",
        dtype="Boolean",
        period="Year",
    ),
    "filing_status": RACVariable(
        name="filing_status",
        statute="26/1",
        description="Tax filing status",
        entity="TaxUnit",
        dtype="Categorical",
        period="Year",
    ),
}


# Map from PolicyEngine variable names to our RAC variables
POLICYENGINE_TO_RAC: dict[str, str] = {
    "adjusted_gross_income": "adjusted_gross_income",
    "irs_employment_income": "employment_income",
    "self_employment_income": "self_employment_income",
    "taxable_interest_income": "interest_income",
    "non_qualified_dividend_income": "dividend_income",
    "qualified_dividend_income": "qualified_dividend_income",
    "rental_income": "rental_income",
    "loss_limited_net_capital_gains": "capital_gains",
    "social_security": "social_security_income",
    "pension_income": "pension_income",
    "partnership_s_corp_income": "partnership_s_corp_income",
    "unemployment_compensation": "unemployment_compensation",
    "charitable_deduction": "charitable_deduction",
    "salt_deduction": "salt_deduction",
    "interest_deduction": "mortgage_interest_deduction",
    "medical_expense_deduction": "medical_expense_deduction",
    "qualified_business_income_deduction": "qbi_deduction",
    "taxable_income": "taxable_income",
    "income_tax_before_credits": "income_tax_before_credits",
    "eitc": "earned_income_credit",
    "ctc": "child_tax_credit",
    "refundable_ctc": "additional_child_tax_credit",
    "snap": "snap_benefit",
    "is_medicaid_eligible": "medicaid_eligible",
    "ssi": "ssi_benefit",
    "tanf": "tanf_benefit",
}


# Map from microdata column names (CPS/PUF) to RAC variables
MICRODATA_TO_RAC: dict[str, str] = {
    # CPS columns
    "wage_income": "employment_income",
    "self_employment_income": "self_employment_income",
    "interest_income": "interest_income",
    "dividend_income": "dividend_income",
    "rental_income": "rental_income",
    "social_security_income": "social_security_income",
    "unemployment_compensation": "unemployment_compensation",
    "adjusted_gross_income": "adjusted_gross_income",
    "total_income": "gross_income",
    "head_age": "age",

    # PUF columns (E-codes)
    "E00100": "adjusted_gross_income",
    "E00200": "employment_income",
    "E00300": "interest_income",
    "E00600": "dividend_income",
    "E00650": "qualified_dividend_income",
    "E00900": "self_employment_income",
    "E01000": "capital_gains",
    "E01500": "pension_income",
    "E02300": "unemployment_compensation",
    "E02400": "social_security_income",
}


def get_rac_for_target(target_name: str) -> RACVariable | None:
    """Get RAC variable definition for a target name."""
    return RAC_VARIABLE_MAP.get(target_name)


def get_rac_for_pe_variable(pe_variable: str) -> RACVariable | None:
    """Get RAC variable for a PolicyEngine variable name."""
    rac_name = POLICYENGINE_TO_RAC.get(pe_variable)
    if rac_name:
        return RAC_VARIABLE_MAP.get(rac_name)
    return None


def get_rac_for_microdata_column(column: str) -> RACVariable | None:
    """Get RAC variable for a microdata column name."""
    rac_name = MICRODATA_TO_RAC.get(column)
    if rac_name:
        return RAC_VARIABLE_MAP.get(rac_name)
    return None
