"""
CPS ASEC -> policyengine-us variable mappings.

Maps Census CPS columns to statute-defined variables in policyengine-us.
Each mapping documents:
- The policyengine-us variable it maps to
- The statutory reference (USC section)
- CPS columns used
- Coverage level (full, partial, derived, none)
- Gaps (what the statute requires that CPS can't provide)
"""

from dataclasses import dataclass, field
from enum import Enum

import polars as pl


class CoverageLevel(Enum):
    """How well CPS covers a policyengine-us variable."""

    FULL = "full"  # CPS provides all required data
    PARTIAL = "partial"  # CPS provides some, with known gaps
    DERIVED = "derived"  # Must be computed from other CPS variables
    NONE = "none"  # CPS doesn't have this data


@dataclass
class CoverageGap:
    """A component required by statute but not available in CPS."""

    component: str
    statute_ref: str
    impact: str  # "high", "medium", "low"
    notes: str


@dataclass
class VariableMapping:
    """Metadata for a CPS -> policyengine-us variable mapping."""

    policyengine_us_variable: str
    statute_ref: str
    cps_columns: list[str]
    coverage: CoverageLevel
    entity: str  # "Person", "TaxUnit", "Household"
    gaps: list[CoverageGap] = field(default_factory=list)
    expected_gap_pct: float | None = None  # Expected % undercount vs SOI
    notes: str = ""


# =============================================================================
# MAPPING REGISTRY
# =============================================================================

_MAPPINGS: dict[str, VariableMapping] = {}


def _register(mapping: VariableMapping) -> VariableMapping:
    """Register a mapping in the global registry."""
    _MAPPINGS[mapping.policyengine_us_variable] = mapping
    return mapping


# =============================================================================
# DIRECT MAPPINGS (Full Coverage)
# =============================================================================

_register(VariableMapping(
    policyengine_us_variable="age",
    statute_ref="26 USC 63(f), 24(c)(1)",
    cps_columns=["A_AGE"],
    coverage=CoverageLevel.FULL,
    entity="Person",
    notes="Direct mapping. CPS age is as of survey date (March).",
))


def map_age(persons: pl.DataFrame) -> pl.DataFrame:
    """Map CPS A_AGE to age."""
    return persons.with_columns(
        pl.col("A_AGE").alias("age")
    )


_register(VariableMapping(
    policyengine_us_variable="household_size",
    statute_ref="7 USC 2014(c)",
    cps_columns=["H_NUMPER"],
    coverage=CoverageLevel.FULL,
    entity="Household",
    notes="Direct mapping for SNAP household size.",
))


def map_household_size(households: pl.DataFrame) -> pl.DataFrame:
    """Map CPS H_NUMPER to household_size."""
    return households.with_columns(
        pl.col("H_NUMPER").alias("household_size")
    )


# =============================================================================
# EARNED INCOME (Full Coverage)
# =============================================================================

_register(VariableMapping(
    policyengine_us_variable="earned_income",
    statute_ref="26 USC 32(c)(2) - Earned income defined",
    cps_columns=["WSAL_VAL", "SEMP_VAL"],
    coverage=CoverageLevel.FULL,
    entity="Person",
    notes="Sum of wages/salaries and self-employment income. "
          "32(c)(2) defines earned income for EITC purposes.",
))


def map_earned_income(persons: pl.DataFrame) -> pl.DataFrame:
    """
    Map CPS income to earned_income per 32(c)(2).

    Earned income = wages + salaries + tips + self-employment income
    """
    return persons.with_columns(
        (pl.col("WSAL_VAL").fill_null(0) + pl.col("SEMP_VAL").fill_null(0))
        .alias("earned_income")
    )


# =============================================================================
# FILING STATUS (Derived)
# =============================================================================

_register(VariableMapping(
    policyengine_us_variable="filing_status",
    statute_ref="26 USC 1 (tax rates by status), 2 (definitions)",
    cps_columns=["A_MARITL", "A_AGE", "A_EXPRRP"],
    coverage=CoverageLevel.DERIVED,
    entity="TaxUnit",
    gaps=[
        CoverageGap(
            component="head_of_household",
            statute_ref="26 USC 2(b)",
            impact="medium",
            notes="Requires determining if taxpayer maintains household for "
                  "qualifying person. CPS has household relationships but "
                  "determining HoH status requires assumptions.",
        ),
        CoverageGap(
            component="qualifying_widow",
            statute_ref="26 USC 2(a)",
            impact="low",
            notes="Requires spouse death within 2 years AND dependent child. "
                  "CPS doesn't track year of spouse's death.",
        ),
    ],
    notes="Simplified mapping: married w/ spouse -> joint, others -> single. "
          "Head of household determination requires additional logic.",
))

# CPS A_MARITL codes
_MARITL_MARRIED_SPOUSE_PRESENT = 1
_MARITL_MARRIED_SPOUSE_ABSENT = 2
_MARITL_WIDOWED = 3
_MARITL_DIVORCED = 4
_MARITL_SEPARATED = 5
_MARITL_NEVER_MARRIED = 7


def map_filing_status(persons: pl.DataFrame) -> pl.DataFrame:
    """
    Derive filing status from CPS marital status.

    Simplified mapping:
    - Married, spouse present -> married_joint
    - All others -> single

    TODO: Add head_of_household logic based on:
    - Unmarried
    - Maintains household for qualifying person (child, parent)
    - Pays > 50% of household costs
    """
    return persons.with_columns(
        pl.when(pl.col("A_MARITL") == _MARITL_MARRIED_SPOUSE_PRESENT)
        .then(pl.lit("married_joint"))
        .otherwise(pl.lit("single"))
        .alias("filing_status")
    )


# =============================================================================
# BLINDNESS (Full Coverage)
# =============================================================================

_register(VariableMapping(
    policyengine_us_variable="is_blind",
    statute_ref="26 USC 63(f)(2) - Additional standard deduction for blind",
    cps_columns=["PEDISEYE"],
    coverage=CoverageLevel.FULL,
    entity="Person",
    notes="PEDISEYE: 1 = serious difficulty seeing, 2 = no difficulty. "
          "Tax definition of blind (63(f)(4)) is more specific "
          "(corrected vision <= 20/200 or field <= 20 deg) but CPS proxy is reasonable.",
))


def map_is_blind(persons: pl.DataFrame) -> pl.DataFrame:
    """Map CPS PEDISEYE to is_blind."""
    return persons.with_columns(
        (pl.col("PEDISEYE") == 1).alias("is_blind")
    )


# =============================================================================
# IS DEPENDENT (Derived)
# =============================================================================

_register(VariableMapping(
    policyengine_us_variable="is_dependent",
    statute_ref="26 USC 152 - Dependent defined",
    cps_columns=["A_EXPRRP", "A_AGE", "WSAL_VAL"],
    coverage=CoverageLevel.DERIVED,
    entity="Person",
    gaps=[
        CoverageGap(
            component="support_test",
            statute_ref="26 USC 152(c)(1)(D), 152(d)(1)(C)",
            impact="medium",
            notes="CPS doesn't track who provides >50% support.",
        ),
        CoverageGap(
            component="joint_return_test",
            statute_ref="26 USC 152(c)(1)(E)",
            impact="low",
            notes="Can't determine if dependent filed joint return.",
        ),
    ],
    notes="Simplified: children under 19 (or under 24 if student) are dependents. "
          "Doesn't verify support test or other 152 requirements.",
))


def map_is_dependent(persons: pl.DataFrame) -> pl.DataFrame:
    """
    Derive is_dependent from CPS relationships and age.

    Simplified: Person is dependent if:
    - Has relationship code indicating child/grandchild
    - Age < 19 (or < 24 for students, but we can't identify students easily)
    """
    # A_EXPRRP codes for children: 4 = child, 8 = grandchild
    return persons.with_columns(
        (
            (pl.col("A_EXPRRP").is_in([4, 8])) &
            (pl.col("A_AGE") < 19)
        ).alias("is_dependent")
    )


# =============================================================================
# CTC QUALIFYING CHILDREN (Derived)
# =============================================================================

_register(VariableMapping(
    policyengine_us_variable="ctc_qualifying_children",
    statute_ref="26 USC 24(c) - Qualifying child (under 17, per 152(c))",
    cps_columns=["A_AGE", "A_EXPRRP", "PH_SEQ", "A_LINENO"],
    coverage=CoverageLevel.DERIVED,
    entity="TaxUnit",
    gaps=[
        CoverageGap(
            component="citizenship_test",
            statute_ref="26 USC 24(c)(2)",
            impact="low",
            notes="Child must be US citizen/national/resident. "
                  "CPS has citizenship but we don't filter on it.",
        ),
        CoverageGap(
            component="ssn_requirement",
            statute_ref="26 USC 24(h)(7)",
            impact="low",
            notes="Child must have SSN. Not in CPS.",
        ),
    ],
    notes="Count of children under 17 with qualifying relationship. "
          "Age limit is 17 per 24(c)(1): 'has not attained age 17'.",
))


def map_ctc_qualifying_children(persons: pl.DataFrame) -> pl.DataFrame:
    """
    Count CTC qualifying children per tax unit.

    A child qualifies if:
    - Under age 17 (24(c)(1))
    - Is a qualifying child per 152(c) (relationship, age, residency, support)

    Simplified: count children (A_EXPRRP = 4) under 17 in same household.
    """
    # First, identify qualifying children
    persons_with_flag = persons.with_columns(
        (
            (pl.col("A_EXPRRP") == 4) &  # Child of householder
            (pl.col("A_AGE") < 17)  # Under 17
        ).alias("_is_ctc_child")
    )

    # Count per household (using PH_SEQ as household ID)
    child_counts = (
        persons_with_flag
        .group_by("PH_SEQ")
        .agg(pl.col("_is_ctc_child").sum().alias("ctc_qualifying_children"))
    )

    # Join back and assign to reference person (A_LINENO = 1 typically)
    result = persons_with_flag.join(
        child_counts, on="PH_SEQ", how="left"
    )

    # Only the tax unit head gets the count; others get 0
    # (Simplified: reference person is tax unit head)
    result = result.with_columns(
        pl.when(pl.col("A_LINENO") == 1)
        .then(pl.col("ctc_qualifying_children"))
        .otherwise(0)
        .alias("ctc_qualifying_children")
    )

    return result.drop("_is_ctc_child")


# =============================================================================
# AGI PROXY (Partial Coverage)
# =============================================================================

_register(VariableMapping(
    policyengine_us_variable="adjusted_gross_income",
    statute_ref="26 USC 62(a) - Adjusted gross income defined",
    cps_columns=["WSAL_VAL", "SEMP_VAL", "INT_VAL", "DIV_VAL", "PNSN_VAL"],
    coverage=CoverageLevel.PARTIAL,
    entity="TaxUnit",
    expected_gap_pct=0.15,  # Expect ~15% undercount vs SOI
    gaps=[
        CoverageGap(
            component="capital_gains",
            statute_ref="26 USC 61(a)(3), 1222",
            impact="high",
            notes="CPS does not collect capital gains. SOI 2021: ~$1.2T. "
                  "Major source of income for top brackets.",
        ),
        CoverageGap(
            component="ira_deduction",
            statute_ref="26 USC 62(a)(7)",
            impact="medium",
            notes="Above-the-line IRA deduction not in CPS. Our proxy will "
                  "overstate AGI by this amount.",
        ),
        CoverageGap(
            component="student_loan_interest",
            statute_ref="26 USC 62(a)(17)",
            impact="low",
            notes="Student loan interest deduction not in CPS.",
        ),
        CoverageGap(
            component="self_employment_tax_deduction",
            statute_ref="26 USC 62(a)(1)",
            impact="medium",
            notes="Deductible portion of SE tax not calculable from CPS.",
        ),
        CoverageGap(
            component="interest_dividends_underreporting",
            statute_ref="26 USC 61(a)(4), (7)",
            impact="medium",
            notes="CPS interest/dividends underreported by ~40% vs SOI.",
        ),
    ],
    notes="AGI proxy using available CPS income. Known to undercount vs SOI "
          "due to missing capital gains and underreported investment income. "
          "Also overstates slightly due to missing above-line deductions.",
))


def map_agi_proxy(persons: pl.DataFrame) -> pl.DataFrame:
    """
    Construct AGI proxy from available CPS income variables.

    This is NOT true AGI (62). Missing components include:
    - Capital gains (1222) - NOT in CPS
    - Above-line deductions (62(a)(1)-(21)) - NOT in CPS

    Included (with notes):
    - Wages/salaries (61(a)(1)) - WSAL_VAL
    - Self-employment (61(a)(2)) - SEMP_VAL
    - Interest (61(a)(4)) - INT_VAL (underreported ~40%)
    - Dividends (61(a)(7)) - DIV_VAL (underreported ~40%)
    - Pensions (61(a)(11)) - PNSN_VAL
    """
    income_cols = ["WSAL_VAL", "SEMP_VAL", "INT_VAL", "DIV_VAL", "PNSN_VAL"]

    # Fill nulls with 0 for each column
    expr = pl.lit(0)
    for col in income_cols:
        if col in persons.columns:
            expr = expr + pl.col(col).fill_null(0)

    return persons.with_columns(expr.alias("agi_proxy"))


# =============================================================================
# REGISTRY FUNCTIONS
# =============================================================================

def get_mapping_metadata(variable_name: str) -> VariableMapping:
    """Get metadata for a variable mapping."""
    # Handle aliases
    if variable_name == "agi_proxy":
        variable_name = "adjusted_gross_income"

    if variable_name not in _MAPPINGS:
        raise KeyError(f"No mapping found for '{variable_name}'")

    return _MAPPINGS[variable_name]


def get_all_mappings() -> list[VariableMapping]:
    """Get all registered variable mappings."""
    return list(_MAPPINGS.values())


def coverage_summary() -> dict[str, list[str]]:
    """
    Summarize coverage by level.

    Returns dict mapping coverage level to list of variable names.
    """
    result = {level.value: [] for level in CoverageLevel}

    for mapping in _MAPPINGS.values():
        result[mapping.coverage.value].append(mapping.policyengine_us_variable)

    return result
