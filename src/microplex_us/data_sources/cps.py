"""
CPS ASEC (Annual Social and Economic Supplement) data loading.

The CPS ASEC is the primary source for income and poverty statistics in the US.
Released annually in March, it contains detailed income, employment, and
demographic information for ~100K households.

Data source: https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html
"""

import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceArchetype,
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

from microplex_us.data_sources.cps_age import randomize_cps_topcoded_age_80_84
from microplex_us.data_sources.sampling import (
    sample_frame_with_state_floor,
    sample_frame_without_replacement,
)
from microplex_us.source_registry import resolve_source_variable_capabilities

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "microplex"
CPS_ASEC_PROCESSED_CACHE_VERSION = "20260604_spm_retirement_target_inputs"

CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP = {
    "reported_has_direct_purchase_health_coverage_at_interview": "NOW_DIR",
    "reported_has_marketplace_health_coverage_at_interview": "NOW_MRK",
    "reported_has_subsidized_marketplace_health_coverage_at_interview": "NOW_MRKS",
    "reported_has_unsubsidized_marketplace_health_coverage_at_interview": "NOW_MRKUN",
    "reported_has_non_marketplace_direct_purchase_health_coverage_at_interview": (
        "NOW_NONM"
    ),
    "reported_has_employer_sponsored_health_coverage_at_interview": "NOW_GRP",
    "reported_has_medicare_health_coverage_at_interview": "NOW_MCARE",
    "reported_has_medicaid_health_coverage_at_interview": "NOW_CAID",
    "reported_has_means_tested_health_coverage_at_interview": "NOW_MCAID",
    "reported_has_chip_health_coverage_at_interview": "NOW_PCHIP",
    "reported_has_other_means_tested_health_coverage_at_interview": "NOW_OTHMT",
    "reported_has_tricare_health_coverage_at_interview": "NOW_MIL",
    "reported_has_champva_health_coverage_at_interview": "NOW_CHAMPVA",
    "reported_has_va_health_coverage_at_interview": "NOW_VACARE",
    "reported_has_indian_health_service_coverage_at_interview": "NOW_IHSFLG",
}

CURRENT_HEALTH_COVERAGE_RULE_INPUT_ALIAS_MAP = {
    "has_marketplace_health_coverage_at_interview": (
        "reported_has_marketplace_health_coverage_at_interview"
    ),
    "has_non_marketplace_direct_purchase_health_coverage_at_interview": (
        "reported_has_non_marketplace_direct_purchase_health_coverage_at_interview"
    ),
    "has_medicaid_health_coverage_at_interview": (
        "reported_has_medicaid_health_coverage_at_interview"
    ),
    "has_other_means_tested_health_coverage_at_interview": (
        "reported_has_other_means_tested_health_coverage_at_interview"
    ),
    "has_tricare_health_coverage_at_interview": (
        "reported_has_tricare_health_coverage_at_interview"
    ),
    "has_champva_health_coverage_at_interview": (
        "reported_has_champva_health_coverage_at_interview"
    ),
    "has_va_health_coverage_at_interview": (
        "reported_has_va_health_coverage_at_interview"
    ),
    "has_indian_health_service_coverage_at_interview": (
        "reported_has_indian_health_service_coverage_at_interview"
    ),
}

CENSUS_OCCUPATION_CODE_TO_TTOC = {
    725: 502,
    2350: 507,
    2633: 502,
    2752: 206,
    2755: 207,
    2770: 208,
    2910: 503,
    3602: 501,
    3630: 602,
    4000: 105,
    4010: 106,
    4030: 106,
    4040: 101,
    4055: 107,
    4110: 102,
    4120: 103,
    4130: 104,
    4140: 108,
    4150: 109,
    4160: 106,
    4230: 304,
    4251: 402,
    4350: 506,
    4420: 210,
    4500: 603,
    4510: 603,
    4521: 605,
    4522: 601,
    4600: 508,
    4621: 607,
    4655: 501,
    5130: 203,
    5300: 303,
    6355: 403,
    6442: 404,
    7120: 401,
    7200: 409,
    7315: 405,
    7320: 406,
    7340: 401,
    7540: 408,
    7610: 401,
    7800: 110,
    8510: 401,
    9122: 806,
    9141: 803,
    9142: 802,
    9350: 801,
    9610: 805,
    9620: 809,
}

# CPS ASEC data URLs by year
CPS_URLS = {
    2025: "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asecpub25csv.zip",
    2024: "https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip",
    2023: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip",
}

# Key variable mappings (Census variable name -> our name)
PERSON_VARIABLES = {
    # Demographics
    "A_AGE": "age",
    "A_SEX": "sex",
    "PRDTRACE": "race",
    "PEHSPNON": "hispanic",
    "PRDTHSP": "_cps_hispanic_code",
    "A_HGA": "education",
    "PEDISDRS": "_disability_dressing",
    "PEDISEAR": "_disability_hearing",
    "PEDISEYE": "_disability_vision",
    "PEDISOUT": "_disability_errands",
    "PEDISPHY": "_disability_physical",
    "PEDISREM": "_disability_cognitive",
    "DIS_VAL1": "_disability_income_1",
    "DIS_SC1": "_disability_income_code_1",
    "DIS_VAL2": "_disability_income_2",
    "DIS_SC2": "_disability_income_code_2",
    "RESNSS1": "_social_security_reason_1",
    "RESNSS2": "_social_security_reason_2",
    # Employment
    "A_CLSWKR": "class_of_worker",
    "A_WKSTAT": "work_status",
    "A_HRS1": "hours_worked",
    "A_HSCOL": "_high_school_or_college_status",
    "A_HRLYWK": "_is_paid_hourly_code",
    "A_HRSPAY": "_hourly_pay_cents",
    "A_UNMEM": "_union_member_code",
    "POCCU2": "detailed_occupation_recode",
    "PEIOOCC": "_detailed_census_occupation_code",
    # Income (annual)
    "WSAL_VAL": "wage_income",
    "SEMP_VAL": "self_employment_income",
    # Bundled retirement-contribution total (Census asks a single
    # "how much did you contribute to retirement accounts?" question).
    # Split into account-type-specific desired leaves in _process_persons,
    # mirroring eCPS cps.py:1500-1552. Staging column dropped after split.
    "RETCB_VAL": "_retirement_contributions",
    "INT_VAL": "interest_income",
    "DIV_VAL": "dividend_income",
    "RNT_VAL": "rental_income",
    "ANN_VAL": "_annuity_income",
    "PNSN_VAL": "_pension_income",
    "SS_VAL": "social_security",
    "SSI_VAL": "ssi",
    "UC_VAL": "unemployment_compensation",
    "LKWEEKS": "weeks_unemployed",
    "VET_VAL": "veterans_benefits",
    "WC_VAL": "workers_compensation",
    "DST_SC1": "_retirement_distribution_code_1",
    "DST_SC2": "_retirement_distribution_code_2",
    "DST_SC1_YNG": "_retirement_distribution_code_1_yng",
    "DST_SC2_YNG": "_retirement_distribution_code_2_yng",
    "DST_VAL1": "_retirement_distribution_value_1",
    "DST_VAL2": "_retirement_distribution_value_2",
    "DST_VAL1_YNG": "_retirement_distribution_value_1_yng",
    "DST_VAL2_YNG": "_retirement_distribution_value_2_yng",
    # CPS-derived direct income copies (mirror eCPS cps.py:1493-1495).
    "SRVS_VAL": "survivor_benefits",
    "ED_VAL": "educational_assistance",
    "FIN_VAL": "financial_assistance",
    "PTOTVAL": "total_person_income",
    "OI_OFF": "_other_income_code",
    "OI_VAL": "_other_income_value",
    # Benefits
    "PAW_VAL": "public_assistance",
    "CSP_VAL": "child_support_received",
    "CHSP_VAL": "child_support_expense",
    "MCARE": "has_medicare",
    "MCAID": "has_medicaid",
    "NOW_GRP": "has_esi",
    "NOW_MRK": "has_marketplace_health_coverage",
    **{
        census_name: f"_{leaf}"
        for leaf, census_name in CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP.items()
    },
    "NOW_PRIV": "_reported_has_private_health_coverage_at_interview",
    "NOW_PUB": "_reported_has_public_health_coverage_at_interview",
    "NOW_COV": "_reported_current_health_coverage_code",
    # Employer-sponsored insurance policyholder + premium inputs (eCPS
    # cps.py:197-275). NOW_OWNGRP flags own-name current group (ESI) coverage;
    # NOW_HIPAID is who pays the premium; NOW_GRPFTYP is family vs self-only
    # plan. These seed the ESI policyholder recode and the premium imputation.
    "NOW_OWNGRP": "_now_owngrp",
    "NOW_HIPAID": "_now_hipaid",
    "NOW_GRPFTYP": "_now_grpftyp",
    "PHIP_VAL": "health_insurance_premiums_without_medicare_part_b",
    "POTC_VAL": "over_the_counter_health_expenses",
    "PMED_VAL": "other_medical_expenses",
    "PEMCPREM": "medicare_part_b_premiums",
    "WICYN": "_receives_wic",
    "SPM_CAPHOUSESUB": "spm_unit_capped_housing_subsidy_reported",
    "SPM_ENGVAL": "spm_unit_energy_subsidy",
    "SPM_CAPWKCCXPNS": "spm_unit_capped_work_childcare_expenses",
    "SPM_CHILDCAREXPNS": "spm_unit_pre_subsidy_childcare_expenses",
    # Person relationship-to-householder code (eCPS cps.py:190-195, :1219).
    # Codes 43/44/46/47 mark an unmarried partner of the household head.
    "PERRP": "_person_relationship_to_householder",
    # Identifiers
    "PH_SEQ": "household_id",
    "GESTFIPS": "state_fips",
    "PF_SEQ": "family_id",
    "TAX_ID": "tax_unit_id",
    "SPM_ID": "spm_unit_id",
    "A_LINENO": "person_number",
    "A_SPOUSE": "spouse_person_number",
    "A_FAMREL": "family_relationship",
    "A_MARITL": "marital_status",
    # Weights
    "A_FNLWGT": "weight",
    "MARSUPWT": "march_supplement_weight",
}

HOUSEHOLD_VARIABLES = {
    "H_SEQ": "household_id",
    "GESTFIPS": "state_fips",
    "GTCO": "county_fips",
    "GTCBSA": "cbsa",
    "HRHTYPE": "household_type",
    "H_NUMPER": "household_size",
    "HHINC": "household_income_bracket",
    "HTOTVAL": "household_total_income",
    "HSUP_WGT": "household_weight",
}

PERSON_OBSERVATION_EXCLUDED_COLUMNS = (
    "person_id",
    "household_id",
    "weight",
    "march_supplement_weight",
    "year",
)

HOUSEHOLD_OBSERVATION_EXCLUDED_COLUMNS = (
    "household_id",
    "household_weight",
    "year",
)

CPS_INCOME_ALIAS_COMPONENT_GROUPS = (
    ("wage_income",),
    ("self_employment_income",),
    ("interest_income",),
    ("dividend_income",),
    ("rental_income",),
    ("social_security",),
    ("pension_income", "taxable_pension_income"),
    ("unemployment_compensation",),
    ("alimony_income",),
)
CPS_INCOME_ALIAS_COMPONENTS = tuple(
    column for group in CPS_INCOME_ALIAS_COMPONENT_GROUPS for column in group
)

PERSON_NONNEGATIVE_VALUE_COLUMNS = (
    "wage_income",
    "self_employment_income",
    "interest_income",
    "dividend_income",
    "rental_income",
    "social_security",
    "ssi",
    "unemployment_compensation",
    "public_assistance",
    "total_person_income",
    "alimony_income",
    "child_support_received",
    "child_support_expense",
    "disability_benefits",
    "health_insurance_premiums_without_medicare_part_b",
    "over_the_counter_health_expenses",
    "other_medical_expenses",
    "medicare_part_b_premiums",
    "social_security_disability",
    "social_security_retirement",
    "social_security_survivors",
    "social_security_dependents",
    "spm_unit_energy_subsidy",
    "spm_unit_capped_housing_subsidy_reported",
    "spm_unit_capped_work_childcare_expenses",
    "spm_unit_pre_subsidy_childcare_expenses",
    "hourly_wage",
    "self_employed_pension_contributions",
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
    "taxable_401k_distributions",
    "tax_exempt_401k_distributions",
    "taxable_403b_distributions",
    "tax_exempt_403b_distributions",
    "regular_ira_distributions",
    "roth_ira_distributions",
    "tax_exempt_ira_distributions",
    "taxable_sep_distributions",
    "tax_exempt_sep_distributions",
    "other_type_retirement_account_distributions",
    "keogh_distributions",
    "veterans_benefits",
    "workers_compensation",
    "weeks_unemployed",
)

PERSON_ZERO_DEFAULT_VALUE_COLUMNS = (
    "alimony_income",
    "child_support_received",
    "child_support_expense",
    "disability_benefits",
    "health_insurance_premiums_without_medicare_part_b",
    "over_the_counter_health_expenses",
    "other_medical_expenses",
    "medicare_part_b_premiums",
    "social_security_disability",
    "social_security_retirement",
    "social_security_survivors",
    "social_security_dependents",
    "spm_unit_energy_subsidy",
    "spm_unit_capped_housing_subsidy_reported",
    "spm_unit_capped_work_childcare_expenses",
    "spm_unit_pre_subsidy_childcare_expenses",
    "hourly_wage",
    "self_employed_pension_contributions",
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
    "taxable_401k_distributions",
    "tax_exempt_401k_distributions",
    "taxable_403b_distributions",
    "tax_exempt_403b_distributions",
    "regular_ira_distributions",
    "roth_ira_distributions",
    "tax_exempt_ira_distributions",
    "taxable_sep_distributions",
    "tax_exempt_sep_distributions",
    "other_type_retirement_account_distributions",
    "keogh_distributions",
    "veterans_benefits",
    "workers_compensation",
    "weeks_unemployed",
)

PERSON_CACHE_REQUIRED_COLUMNS = (
    "state_fips",
    "county_fips",
    "cps_race",
    "is_hispanic",
    "is_disabled",
    "has_esi",
    "has_marketplace_health_coverage",
    "alimony_income",
    "child_support_received",
    "child_support_expense",
    "disability_benefits",
    "health_insurance_premiums_without_medicare_part_b",
    "other_medical_expenses",
    "over_the_counter_health_expenses",
    "medicare_part_b_premiums",
    "social_security_disability",
    "social_security_retirement",
    "social_security_survivors",
    "social_security_dependents",
    "receives_wic",
    "spm_unit_pre_subsidy_childcare_expenses",
    "tax_exempt_private_pension_income",
    "regular_ira_distributions",
    "roth_ira_distributions",
    "tax_exempt_ira_distributions",
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "taxable_sep_distributions",
    "other_type_retirement_account_distributions",
    "keogh_distributions",
    "veterans_benefits",
    "workers_compensation",
    "weeks_unemployed",
)

PERSON_CPS_DISABILITY_COLUMNS = (
    "_disability_dressing",
    "_disability_hearing",
    "_disability_vision",
    "_disability_errands",
    "_disability_physical",
    "_disability_cognitive",
)

# eCPS difficulty_* eligibility leaves recoded from the ASEC PEDIS* fields
# (PEDIS{X} == 1 -> True, the same recode eCPS uses for is_blind from PEDISEYE).
# These are eCPS final-H5 contract columns, not pe-us variables, so they export
# via the legacy-contract entity map. Mirrors policyengine-us-data
# datasets/cps/cps.py (unmerged branch claude/document-census-tax-id-replacement)
# which maps each difficulty leaf to its PEDIS source field.
PERSON_CPS_DIFFICULTY_LEAVES = {
    "_disability_dressing": "difficulty_dressing_or_bathing",
    "_disability_hearing": "difficulty_hearing",
    "_disability_vision": "difficulty_seeing",
    "_disability_errands": "difficulty_doing_errands",
    "_disability_physical": "difficulty_walking_or_climbing_stairs",
    "_disability_cognitive": "difficulty_remembering_or_making_decisions",
}

WORKERS_COMP_DISABILITY_CODE = 1
ALIMONY_OTHER_INCOME_CODE = 20
STRIKE_BENEFITS_OTHER_INCOME_CODE = 12
SOCIAL_SECURITY_RETIREMENT_REASON_CODE = 1
SOCIAL_SECURITY_DISABILITY_REASON_CODE = 2
SOCIAL_SECURITY_SURVIVOR_REASON_CODES = (3, 5)
SOCIAL_SECURITY_DEPENDENT_REASON_CODES = (4, 6, 7)
MINIMUM_RETIREMENT_AGE = 62

# Retirement-contribution allocation fractions used to split the single
# bundled CPS RETCB_VAL total into the account-type-specific desired
# contribution leaves the eCPS contract requires. These mirror the eCPS
# split (PolicyEngine/policyengine-us-data
# policyengine_us_data/datasets/cps/cps.py:1500-1552) and trace exactly to
# policyengine_us_data/datasets/cps/imputation_parameters.yaml:
#   se_pension_share_of_retirement_contributions: 0.046  (yaml line 30)
#   dc_share_of_retirement_contributions:        0.908  (yaml line 38)
#   roth_share_of_dc_contributions:              0.15   (yaml line 48)
#   traditional_share_of_ira_contributions:      0.392  (yaml line 55)
# "Desired" means pre-statutory-limit; PolicyEngine-US applies the limits.
SE_PENSION_SHARE_OF_RETIREMENT_CONTRIBUTIONS = 0.046
DC_SHARE_OF_RETIREMENT_CONTRIBUTIONS = 0.908
ROTH_SHARE_OF_DC_CONTRIBUTIONS = 0.15
TRADITIONAL_SHARE_OF_IRA_CONTRIBUTIONS = 0.392
TAXABLE_PENSION_FRACTION = 0.590
TAXABLE_401K_DISTRIBUTION_FRACTION = 1.0
TAXABLE_403B_DISTRIBUTION_FRACTION = 1.0
TAXABLE_SEP_DISTRIBUTION_FRACTION = 1.0
RETIREMENT_CONTRIBUTION_LIMITS_BY_YEAR = {
    2021: {"401k": 19_500, "401k_catch_up": 6_500, "ira": 6_000, "ira_catch_up": 1_000},
    2022: {"401k": 20_500, "401k_catch_up": 6_500, "ira": 6_000, "ira_catch_up": 1_000},
    2023: {"401k": 22_500, "401k_catch_up": 7_500, "ira": 6_500, "ira_catch_up": 1_000},
    2024: {"401k": 23_000, "401k_catch_up": 7_500, "ira": 7_000, "ira_catch_up": 1_000},
    2025: {"401k": 23_500, "401k_catch_up": 7_500, "ira": 7_000, "ira_catch_up": 1_000},
}
RETIREMENT_CATCH_UP_AGE = 50

# Census CPS ASEC 2024 technical documentation, PERRP (relationship to
# household reference person). Codes 43/44/46/47 mark an unmarried partner of
# the household head. Mirrors policyengine-us-data cps.py:190-195, :1219.
# https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar24.pdf
PERRP_UNMARRIED_PARTNER_OF_HOUSEHOLD_HEAD_CODES = (43, 44, 46, 47)

# Employer-sponsored insurance recode/imputation codes and plan-type priors,
# mirrored verbatim from policyengine-us-data cps.py:204-274.
ESI_HAS_CURRENT_OWN_COVERAGE = 1  # NOW_OWNGRP: holds ESI in own name.
ESI_EMPLOYER_PAYS_ALL = 1  # NOW_HIPAID
ESI_EMPLOYER_PAYS_SOME = 2  # NOW_HIPAID
ESI_FAMILY_PLAN = 1  # NOW_GRPFTYP
ESI_SELF_ONLY_PLAN = 2  # NOW_GRPFTYP
# AHRQ MEPS-IC Table IV.A.1 (private sector, 2024) plan-type averages. eCPS
# hardcodes these same constants to seed CPS policyholder premium records;
# national calibration later aligns the aggregate to the BEA full-economy
# employer premium total. These are constants in eCPS, not external data.
ESI_PLAN_PRIORS_2024 = {
    "family": {
        "total_premium": 21_207.52589669509,
        "employee_contribution": 6_490.205059544782,
    },
    "self_only": {
        "total_premium": 8_389.275834815255,
        "employee_contribution": 1_909.5781466113417,
    },
}
PE_CPS_UNDOCUMENTED_TARGET = 13e6
PE_CPS_UNDOCUMENTED_WORKERS_TARGET = 8.3e6
PE_CPS_UNDOCUMENTED_STUDENTS_TARGET = 0.21 * 1.9e6


def derive_treasury_tipped_occupation_code(
    census_occupation_codes: pd.Series | np.ndarray,
) -> np.ndarray:
    """Map CPS detailed occupation codes to Treasury tipped occupation codes."""
    values = pd.Series(census_occupation_codes, copy=False)
    values = pd.to_numeric(values, errors="coerce").fillna(-1).astype(int)
    return (
        values.map(CENSUS_OCCUPATION_CODE_TO_TTOC).fillna(0).astype(np.int16).to_numpy()
    )


def processed_cps_asec_cache_path(*, year: int, cache_dir: Path) -> Path:
    """Return the versioned processed-cache path for one CPS ASEC year."""
    return cache_dir / (
        f"cps_asec_{year}_processed_v{CPS_ASEC_PROCESSED_CACHE_VERSION}.parquet"
    )


def legacy_processed_cps_asec_cache_path(*, year: int, cache_dir: Path) -> Path:
    """Return the legacy unversioned processed-cache path for one CPS ASEC year."""
    return cache_dir / f"cps_asec_{year}_processed.parquet"


@dataclass
class CPSDataset:
    """Container for CPS ASEC data."""

    persons: pl.DataFrame
    households: pl.DataFrame
    year: int
    source: str

    @property
    def n_persons(self) -> int:
        return len(self.persons)

    @property
    def n_households(self) -> int:
        return len(self.households)

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "year": self.year,
            "n_persons": self.n_persons,
            "n_households": self.n_households,
            "states": self.households["state_fips"].n_unique(),
            "total_weight": float(self.persons["weight"].sum()),
        }


def _descriptor_from_tables(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    name: str,
) -> SourceDescriptor:
    household_variables = tuple(
        column
        for column in households.columns
        if column not in HOUSEHOLD_OBSERVATION_EXCLUDED_COLUMNS
    )
    person_variables = tuple(
        column
        for column in persons.columns
        if column not in PERSON_OBSERVATION_EXCLUDED_COLUMNS
    )
    return SourceDescriptor(
        name=name,
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        archetype=SourceArchetype.HOUSEHOLD_INCOME,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=household_variables,
                weight_column="household_weight"
                if "household_weight" in households.columns
                else None,
                period_column="year" if "year" in households.columns else None,
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=person_variables,
                weight_column="weight" if "weight" in persons.columns else None,
                period_column="year" if "year" in persons.columns else None,
            ),
        ),
        variable_capabilities=resolve_source_variable_capabilities(
            name,
            (*household_variables, *person_variables),
        ),
    )


def _ensure_person_ids(persons: pd.DataFrame) -> pd.DataFrame:
    result = persons.copy()
    if "person_id" in result.columns:
        return result
    if "person_number" in result.columns and "household_id" in result.columns:
        result["person_id"] = (
            result["household_id"].astype(str)
            + ":"
            + result["person_number"].astype(str)
        )
        return result
    if "household_id" in result.columns:
        result["person_id"] = (
            result["household_id"].astype(str)
            + ":"
            + result.groupby("household_id").cumcount().add(1).astype(str)
        )
        return result
    result["person_id"] = np.arange(len(result)).astype(str)
    return result


def _add_cps_income_aliases(persons: pd.DataFrame) -> pd.DataFrame:
    """Derive canonical income from CPS components for PE-style donor matching."""
    if "income" in persons.columns:
        return persons
    component_groups = [
        tuple(column for column in group if column in persons.columns)
        for group in CPS_INCOME_ALIAS_COMPONENT_GROUPS
    ]
    component_groups = [group for group in component_groups if group]
    if not component_groups:
        if "total_person_income" not in persons.columns:
            return persons
        result = persons.copy()
        result["income"] = (
            pd.to_numeric(result["total_person_income"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        return result

    result = persons.copy()
    income = pd.Series(0.0, index=result.index, dtype=float)
    for group in component_groups:
        column = group[0]
        income = income + (
            pd.to_numeric(result[column], errors="coerce").fillna(0.0).astype(float)
        )
    result["income"] = income.astype(float)
    return result


def _repair_relationship_to_head(
    persons: pd.DataFrame,
    relationship: pd.Series,
) -> pd.Series:
    """Repair household relationship patterns so each household has one clear head."""
    normalized = relationship.astype(int).copy()
    if "household_id" not in persons.columns:
        return normalized

    ages = pd.to_numeric(persons.get("age", 0), errors="coerce").fillna(0.0)
    grouped = persons.groupby("household_id", sort=False).groups
    for member_index in grouped.values():
        member_index = list(member_index)
        household_relationship = normalized.loc[member_index].copy()
        household_ages = ages.loc[member_index]

        head_index = household_relationship[household_relationship.eq(0)].index.tolist()
        if not head_index:
            spouse_candidates = [
                index
                for index in household_relationship[
                    household_relationship.eq(1)
                ].index.tolist()
                if household_ages.loc[index] >= 18
            ]
            adult_candidates = [
                index
                for index in household_relationship.index.tolist()
                if household_ages.loc[index] >= 18
            ]
            candidate_pool = (
                spouse_candidates
                or adult_candidates
                or household_relationship.index.tolist()
            )
            head_choice = max(
                candidate_pool, key=lambda index: household_ages.loc[index]
            )
            normalized.loc[head_choice] = 0
            head_index = [head_choice]
        elif len(head_index) > 1:
            keep_head = max(head_index, key=lambda index: household_ages.loc[index])
            for index in head_index:
                if index == keep_head:
                    continue
                normalized.loc[index] = 3 if household_ages.loc[index] >= 19 else 2

        spouse_index = normalized.loc[member_index][
            normalized.loc[member_index].eq(1)
        ].index.tolist()
        if len(spouse_index) > 1:
            keep_spouse = max(spouse_index, key=lambda index: household_ages.loc[index])
            for index in spouse_index:
                if index == keep_spouse:
                    continue
                normalized.loc[index] = 3 if household_ages.loc[index] >= 19 else 2

    return normalized


def _normalize_relationship_to_head(persons: pd.DataFrame) -> pd.Series:
    """Normalize available CPS relationship coding to head/spouse/dependent/other."""
    family_normalized: pd.Series | None = None
    if "family_relationship" in persons.columns:
        family_relationship = (
            pd.to_numeric(persons["family_relationship"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )
        unique_values = set(family_relationship.unique().tolist())
        if unique_values.issubset({0, 1, 2, 3, 4}):
            family_normalized = pd.Series(3, index=persons.index, dtype=int)
            household_groups = (
                persons.groupby("household_id", sort=False).groups.values()
                if "household_id" in persons.columns
                else [persons.index]
            )
            for member_index in household_groups:
                member_index = list(member_index)
                household_codes = set(family_relationship.loc[member_index].tolist())
                if 0 in household_codes:
                    mapped = family_relationship.loc[member_index].map(
                        {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
                    )
                else:
                    mapped = family_relationship.loc[member_index].map(
                        {1: 0, 2: 1, 3: 2, 4: 3}
                    )
                family_normalized.loc[member_index] = mapped.fillna(3).astype(int)

    if "relationship_to_head" not in persons.columns:
        if family_normalized is not None:
            return _repair_relationship_to_head(persons, family_normalized)
        order = persons.groupby("household_id").cumcount()
        normalized = pd.Series(3, index=persons.index, dtype=int)
        normalized.loc[order == 0] = 0
        normalized.loc[
            (order == 1)
            & (pd.to_numeric(persons.get("age", 0), errors="coerce").fillna(0) >= 18)
        ] = 1
        normalized.loc[
            pd.to_numeric(persons.get("age", 0), errors="coerce").fillna(0) < 18
        ] = 2
        return _repair_relationship_to_head(persons, normalized)

    relationship = (
        pd.to_numeric(persons["relationship_to_head"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    unique_values = set(relationship.unique().tolist())
    if unique_values.issubset({0, 1, 2, 3}):
        if family_normalized is not None:
            relationship_detail = set(relationship.unique().tolist()) & {1, 2}
            family_detail = set(family_normalized.unique().tolist()) & {1, 2}
            if len(family_detail) > len(relationship_detail):
                return _repair_relationship_to_head(persons, family_normalized)
        return _repair_relationship_to_head(persons, relationship)

    if unique_values.issubset({1, 2, 3, 4}):
        normalized = relationship.map({1: 0, 2: 1, 3: 3, 4: 2}).fillna(3).astype(int)
        return _repair_relationship_to_head(persons, normalized)

    order = persons.groupby("household_id").cumcount()
    normalized = pd.Series(3, index=persons.index, dtype=int)
    normalized.loc[order == 0] = 0
    normalized.loc[
        (order == 1)
        & (pd.to_numeric(persons.get("age", 0), errors="coerce").fillna(0) >= 18)
    ] = 1
    normalized.loc[
        pd.to_numeric(persons.get("age", 0), errors="coerce").fillna(0) < 18
    ] = 2
    return _repair_relationship_to_head(persons, normalized)


def _add_cps_tax_unit_structure_columns(persons: pd.DataFrame) -> pd.DataFrame:
    """Derive PE-style tax-unit role columns from CPS tax-unit identifiers and pointers."""
    if "tax_unit_id" not in persons.columns:
        return persons

    result = persons.copy()
    relationship = _normalize_relationship_to_head(result)
    result["tax_unit_is_joint"] = 0.0
    result["tax_unit_count_dependents"] = 0.0
    result["is_tax_unit_head"] = 0.0
    result["is_tax_unit_spouse"] = 0.0
    result["is_tax_unit_dependent"] = 0.0

    ages = pd.to_numeric(result.get("age", 0), errors="coerce").fillna(0.0)
    spouse_person_number = (
        pd.to_numeric(result.get("spouse_person_number", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    person_number = (
        pd.to_numeric(result.get("person_number", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    valid_tax_unit_ids = result["tax_unit_id"].notna() & result["tax_unit_id"].astype(
        str
    ).str.strip().ne("")
    grouped = result.loc[valid_tax_unit_ids].groupby(
        ["household_id", "tax_unit_id"], sort=False
    )
    for _, unit_persons in grouped:
        member_index = unit_persons.index
        unit_relationship = relationship.loc[member_index]
        dependent_index = unit_relationship[unit_relationship.eq(2)].index.tolist()

        spouse_index: list[int] = []
        by_number = {
            int(number): idx
            for idx, number in person_number.loc[member_index].items()
            if int(number) > 0
        }
        for idx in member_index:
            spouse_number = int(spouse_person_number.loc[idx])
            current_number = int(person_number.loc[idx])
            if spouse_number <= 0 or current_number <= 0:
                continue
            spouse_idx = by_number.get(spouse_number)
            if spouse_idx is None:
                continue
            if int(spouse_person_number.loc[spouse_idx]) != current_number:
                continue
            spouse_index.extend([int(idx), int(spouse_idx)])
        if not spouse_index:
            spouse_index = (
                unit_relationship[unit_relationship.eq(1)].index.astype(int).tolist()
            )
        spouse_index = [
            idx for idx in dict.fromkeys(spouse_index) if idx not in dependent_index
        ]

        head_index: int | None = None
        head_candidates = [
            int(idx)
            for idx in unit_relationship[unit_relationship.eq(0)].index.tolist()
            if int(idx) not in spouse_index
        ]
        if head_candidates:
            head_index = head_candidates[0]
        else:
            nondependent_candidates = [
                int(idx)
                for idx in member_index.tolist()
                if int(idx) not in spouse_index and int(idx) not in dependent_index
            ]
            if nondependent_candidates:
                head_index = max(
                    nondependent_candidates,
                    key=lambda idx: (float(ages.loc[idx]), -int(idx)),
                )
            elif spouse_index:
                head_index = spouse_index[0]
                spouse_index = [idx for idx in spouse_index if idx != head_index]
            else:
                head_index = int(member_index[0])

        spouse_index = [idx for idx in spouse_index if idx != head_index]
        if len(spouse_index) > 1:
            spouse_index = [
                max(
                    spouse_index,
                    key=lambda idx: (float(ages.loc[idx]), -int(idx)),
                )
            ]

        result.loc[member_index, "tax_unit_is_joint"] = float(bool(spouse_index))
        result.loc[member_index, "tax_unit_count_dependents"] = float(
            len(dependent_index)
        )
        result.loc[dependent_index, "is_tax_unit_dependent"] = 1.0
        if head_index is not None:
            result.loc[head_index, "is_tax_unit_head"] = 1.0
        result.loc[spouse_index, "is_tax_unit_spouse"] = 1.0

    return result


def _build_observation_frame(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    source_name: str,
) -> ObservationFrame:
    normalized_households = households.copy()
    normalized_persons = _add_cps_tax_unit_structure_columns(
        _add_cps_income_aliases(_ensure_person_ids(persons))
    )
    descriptor = _descriptor_from_tables(
        households=normalized_households,
        persons=normalized_persons,
        name=source_name,
    )
    frame = ObservationFrame(
        source=descriptor,
        tables={
            EntityType.HOUSEHOLD: normalized_households,
            EntityType.PERSON: normalized_persons,
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
    frame.validate()
    return frame


def _sample_households_and_persons(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    sample_n: int | None,
    random_seed: int,
    state_floor: int | None = None,
    state_age_floor: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample households and keep all linked person records."""
    household_sort_columns = [
        column for column in ("household_id", "year") if column in households.columns
    ]
    person_sort_columns = [
        column
        for column in ("household_id", "person_id", "person_number", "year")
        if column in persons.columns
    ]
    if household_sort_columns:
        households = households.sort_values(
            household_sort_columns,
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        households = households.reset_index(drop=True)
    if person_sort_columns:
        persons = persons.sort_values(
            person_sort_columns,
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        persons = persons.reset_index(drop=True)
    if sample_n is None or sample_n >= len(households):
        return households, persons
    sampled_households = _sample_cps_households(
        households=households,
        persons=persons,
        sample_n=sample_n,
        random_seed=random_seed,
        state_floor=state_floor,
        state_age_floor=state_age_floor,
    )
    sampled_keys = set(sampled_households["household_id"])
    sampled_persons = persons[persons["household_id"].isin(sampled_keys)].copy()
    if household_sort_columns:
        sampled_households = sampled_households.sort_values(
            household_sort_columns,
            kind="mergesort",
        )
    if person_sort_columns:
        sampled_persons = sampled_persons.sort_values(
            person_sort_columns,
            kind="mergesort",
        )
    return sampled_households.reset_index(drop=True), sampled_persons.reset_index(
        drop=True
    )


def _sample_cps_households(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    sample_n: int | None,
    random_seed: int,
    state_floor: int | None = None,
    state_age_floor: int | None = None,
) -> pd.DataFrame:
    """Sample CPS households with optional state or state-age coverage floors."""

    resolved_state_age_floor = int(state_age_floor or 0)
    if (
        resolved_state_age_floor <= 0
        or "state_fips" not in households.columns
        or "age" not in persons.columns
        or "household_id" not in households.columns
        or "household_id" not in persons.columns
    ):
        return sample_frame_with_state_floor(
            households,
            sample_n=sample_n,
            random_seed=random_seed,
            weight_col="household_weight",
            state_floor=state_floor,
        )

    coverage = persons[["household_id", "age"]].merge(
        households[["household_id", "state_fips"]],
        on="household_id",
        how="inner",
    )
    coverage["age_band"] = coverage["age"].map(_cps_age_band_key)
    coverage["state_fips"] = pd.to_numeric(
        coverage["state_fips"], errors="coerce"
    ).astype("Int64")
    coverage = coverage.dropna(subset=["state_fips", "age_band"]).copy()
    if coverage.empty:
        return sample_frame_with_state_floor(
            households,
            sample_n=sample_n,
            random_seed=random_seed,
            weight_col="household_weight",
            state_floor=state_floor,
        )

    rng = np.random.default_rng(random_seed)
    selected_ids: set[int] = set()
    for _, group in coverage.groupby(["state_fips", "age_band"], sort=True):
        group_household_ids = pd.Index(group["household_id"].unique())
        already_selected = [hid for hid in group_household_ids if hid in selected_ids]
        missing = resolved_state_age_floor - len(already_selected)
        if missing <= 0:
            continue
        available_ids = [hid for hid in group_household_ids if hid not in selected_ids]
        if not available_ids:
            continue
        candidate_households = households[
            households["household_id"].isin(available_ids)
        ].copy()
        sampled = sample_frame_without_replacement(
            candidate_households,
            sample_n=min(missing, len(candidate_households)),
            random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            weight_col="household_weight",
        )
        selected_ids.update(int(hid) for hid in sampled["household_id"].tolist())

    if sample_n is not None and len(selected_ids) > sample_n:
        raise ValueError(
            "state_age_floor requires more sampled households than sample_n allows: "
            f"selected={len(selected_ids)}, sample_n={sample_n}"
        )

    if not selected_ids:
        return sample_frame_with_state_floor(
            households,
            sample_n=sample_n,
            random_seed=random_seed,
            weight_col="household_weight",
            state_floor=state_floor,
        )

    selected = households[households["household_id"].isin(selected_ids)].copy()
    remaining_n = int(sample_n) - len(selected)
    if remaining_n <= 0:
        return selected

    remainder = households[~households["household_id"].isin(selected_ids)].copy()
    remainder_sample = sample_frame_without_replacement(
        remainder,
        sample_n=remaining_n,
        random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        weight_col="household_weight",
    )
    return pd.concat([selected, remainder_sample], axis=0, ignore_index=False)


def _cps_age_band_key(age: float | int | None) -> str | None:
    value = pd.to_numeric(pd.Series([age]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    age_int = int(value)
    if age_int < 0:
        return None
    if age_int >= 85:
        return "85_plus"
    lower = (age_int // 5) * 5
    upper = lower + 5
    return f"{lower}_{upper}"


@dataclass
class CPSASECSourceProvider:
    """Source-provider wrapper around the CPS ASEC Census loader."""

    year: int = 2023
    cache_dir: Path | None = None
    download: bool = True
    loader: Callable[..., CPSDataset] | None = None
    _descriptor_cache: SourceDescriptor | None = None

    @property
    def descriptor(self) -> SourceDescriptor:
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        return SourceDescriptor(
            name="cps_asec",
            shareability=Shareability.PUBLIC,
            time_structure=TimeStructure.REPEATED_CROSS_SECTION,
            archetype=SourceArchetype.HOUSEHOLD_INCOME,
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column="household_id",
                    variable_names=("state_fips",),
                    weight_column="household_weight",
                ),
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column="person_id",
                    variable_names=("age",),
                    weight_column="weight",
                ),
            ),
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        query = query or SourceQuery()
        provider_filters = query.provider_filters
        loader = self.loader or load_cps_asec
        dataset = loader(
            year=int(provider_filters.get("year", self.year)),
            cache_dir=provider_filters.get("cache_dir", self.cache_dir),
            download=bool(provider_filters.get("download", self.download)),
        )
        households = dataset.households.to_pandas()
        persons = dataset.persons.to_pandas()
        households, persons = _sample_households_and_persons(
            households=households,
            persons=persons,
            sample_n=provider_filters.get("sample_n"),
            random_seed=int(provider_filters.get("random_seed", 0)),
            state_floor=provider_filters.get("state_floor"),
            state_age_floor=provider_filters.get("state_age_floor"),
        )
        frame = _build_observation_frame(
            households=households,
            persons=persons,
            source_name=f"cps_asec_{dataset.year}",
        )
        self._descriptor_cache = frame.source
        return apply_source_query(frame, query)


@dataclass
class CPSASECParquetSourceProvider:
    """Source-provider wrapper around split CPS household/person parquet files."""

    data_dir: str | Path
    year: int | None = None
    households_filename: str = "cps_asec_households.parquet"
    persons_filename: str = "cps_asec_persons.parquet"
    _descriptor_cache: SourceDescriptor | None = None

    @property
    def descriptor(self) -> SourceDescriptor:
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        return SourceDescriptor(
            name="cps_asec_parquet",
            shareability=Shareability.PUBLIC,
            time_structure=TimeStructure.REPEATED_CROSS_SECTION,
            archetype=SourceArchetype.HOUSEHOLD_INCOME,
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column="household_id",
                    variable_names=("state_fips",),
                ),
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column="person_id",
                    variable_names=("age",),
                ),
            ),
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        data_dir = Path(self.data_dir)
        households_path = data_dir / self.households_filename
        persons_path = data_dir / self.persons_filename
        if not households_path.exists() or not persons_path.exists():
            raise FileNotFoundError(
                f"CPS ASEC data files not found in {data_dir}.\n"
                "Expected household/person parquet files in the source directory."
            )

        households = pd.read_parquet(households_path)
        persons = pd.read_parquet(persons_path)
        query = query or SourceQuery()
        provider_filters = query.provider_filters
        if self.year is not None:
            households = households.copy()
            persons = persons.copy()
            if "year" not in households.columns:
                households["year"] = self.year
            if "year" not in persons.columns:
                persons["year"] = self.year
        households, persons = _sample_households_and_persons(
            households=households,
            persons=persons,
            sample_n=provider_filters.get("sample_n"),
            random_seed=int(provider_filters.get("random_seed", 0)),
            state_floor=provider_filters.get("state_floor"),
            state_age_floor=provider_filters.get("state_age_floor"),
        )
        frame = _build_observation_frame(
            households=households,
            persons=persons,
            source_name="cps_asec_parquet",
        )
        self._descriptor_cache = frame.source
        return apply_source_query(frame, query)


def download_cps_asec(
    year: int,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download CPS ASEC data for a given year.

    Args:
        year: Year of CPS ASEC (e.g., 2023)
        cache_dir: Directory to cache downloads
        force: Re-download even if cached

    Returns:
        Path to downloaded/cached zip file
    """

    import httpx

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)

    if year not in CPS_URLS:
        available = ", ".join(str(y) for y in sorted(CPS_URLS.keys()))
        raise ValueError(f"CPS ASEC for {year} not available. Available: {available}")

    url = CPS_URLS[year]
    filename = f"cps_asec_{year}.zip"
    cache_path = cache_dir / filename

    if cache_path.exists() and not force:
        print(f"Using cached CPS ASEC {year} from {cache_path}")
        return cache_path

    print(f"Downloading CPS ASEC {year} from {url}...")

    with httpx.Client(follow_redirects=True, timeout=300) as client:
        response = client.get(url)
        response.raise_for_status()

        with open(cache_path, "wb") as f:
            f.write(response.content)

    print(f"Downloaded {len(response.content) / 1_000_000:.1f} MB to {cache_path}")
    return cache_path


def _read_cps_asec_raw_files(
    zip_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    # Schema overrides for columns with large IDs that overflow int64.
    schema_overrides = {
        "PERIDNUM": pl.Utf8,
        "H_IDNUM": pl.Utf8,
        "OCCURNUM": pl.Utf8,
        "QSTNUM": pl.Utf8,
    }

    with zipfile.ZipFile(zip_path, "r") as zf:
        person_file = None
        household_file = None

        for name in zf.namelist():
            lower = name.lower()
            if "pppub" in lower and lower.endswith(".csv"):
                person_file = name
            elif "hhpub" in lower and lower.endswith(".csv"):
                household_file = name

        if person_file is None:
            raise ValueError(f"Could not find person file in {zip_path}")

        with zf.open(person_file) as f:
            persons_raw = pl.read_csv(
                f,
                infer_schema_length=10000,
                schema_overrides=schema_overrides,
            )

        if household_file is None:
            households_raw = None
        else:
            with zf.open(household_file) as f:
                households_raw = pl.read_csv(
                    f,
                    infer_schema_length=10000,
                    schema_overrides=schema_overrides,
                )

    return persons_raw, households_raw


def _attach_previous_year_income(
    *,
    persons: pl.DataFrame,
    current_persons_raw: pl.DataFrame,
) -> pl.DataFrame:
    # The EITC/CTC prior-year-earnings election (the COVID-era "lookback")
    # expired after 2021, so employment_income_last_year /
    # self_employment_income_last_year / previous_year_income_available feed no
    # live PolicyEngine-US formula. Rather than load and panel-join the prior
    # ASEC (an extra survey-year dependency that only covered the ~50% rotation
    # overlap), fall back to current-year earnings as a placeholder. These
    # columns can be dropped entirely once the export contract no longer
    # requires them.
    required = {"WSAL_VAL", "SEMP_VAL"}
    if not required.issubset(set(current_persons_raw.columns)) or len(persons) != len(
        current_persons_raw
    ):
        return persons.with_columns(
            [
                pl.lit(-1.0).alias("employment_income_last_year"),
                pl.lit(-1.0).alias("self_employment_income_last_year"),
                pl.lit(False).alias("previous_year_income_available"),
            ]
        )

    current = current_persons_raw.select(["WSAL_VAL", "SEMP_VAL"]).to_pandas()
    employment = (
        pd.to_numeric(current["WSAL_VAL"], errors="coerce").fillna(0.0).to_numpy(float)
    )
    self_employment = (
        pd.to_numeric(current["SEMP_VAL"], errors="coerce").fillna(0.0).to_numpy(float)
    )
    return persons.with_columns(
        [
            pl.Series("employment_income_last_year", employment),
            pl.Series("self_employment_income_last_year", self_employment),
            pl.Series(
                "previous_year_income_available",
                (employment != 0.0) | (self_employment != 0.0),
            ),
        ]
    )


def load_cps_asec(
    year: int = 2023,
    cache_dir: Path | None = None,
    download: bool = True,
) -> CPSDataset:
    """
    Load CPS ASEC data for a given year.

    Args:
        year: Year of CPS ASEC (e.g., 2023)
        cache_dir: Directory for cached data
        download: Whether to download if not cached

    Returns:
        CPSDataset with persons and households DataFrames
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Prefer a versioned processed cache so derivation-logic changes do not
    # silently reuse stale pre-sim columns.
    processed_path = processed_cps_asec_cache_path(year=year, cache_dir=cache_dir)
    legacy_processed_path = legacy_processed_cps_asec_cache_path(
        year=year,
        cache_dir=cache_dir,
    )
    if processed_path.exists():
        print(f"Loading processed CPS ASEC {year} from {processed_path}")
        persons = pl.read_parquet(processed_path)
        if _processed_persons_have_household_geography(persons):
            households = _derive_households(persons)
            return CPSDataset(
                persons=persons,
                households=households,
                year=year,
                source=str(processed_path),
            )
        print(
            f"Cached processed CPS ASEC {year} is missing state_fips; rebuilding from raw source"
        )
    elif legacy_processed_path.exists():
        print(
            "Ignoring legacy CPS ASEC processed cache "
            f"{legacy_processed_path} because cache version "
            f"{CPS_ASEC_PROCESSED_CACHE_VERSION} is required; rebuilding from raw source"
        )

    # Download if needed
    zip_path = cache_dir / f"cps_asec_{year}.zip"
    if not zip_path.exists():
        if not download:
            raise FileNotFoundError(
                f"CPS ASEC {year} not found at {zip_path}. "
                "Set download=True to fetch from Census."
            )
        zip_path = download_cps_asec(year, cache_dir)

    # Extract and parse
    print(f"Parsing CPS ASEC {year}...")

    persons_raw, households_raw = _read_cps_asec_raw_files(zip_path)

    # Process person data
    persons = _process_persons(persons_raw, year)
    persons = _attach_previous_year_income(
        persons=persons,
        current_persons_raw=persons_raw,
    )

    # Process or derive household data
    if households_raw is not None:
        households = _process_households(households_raw, year)
    else:
        households = _derive_households(persons)

    persons = _attach_cps_ssn_card_type(
        persons=persons,
        households=households,
        persons_raw=persons_raw,
    )
    persons = _attach_household_geography_to_persons(
        persons=persons,
        households=households,
    )

    # Cache processed data
    persons.write_parquet(processed_path)
    print(f"Cached processed data to {processed_path}")

    return CPSDataset(
        persons=persons,
        households=households,
        year=year,
        source=str(zip_path),
    )


def _process_persons(df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Process raw person file into clean format."""
    selected = [
        pl.col(census_name).alias(our_name)
        for census_name, our_name in PERSON_VARIABLES.items()
        if census_name in df.columns
    ]
    if not selected:
        raise ValueError("No recognized variables found in person file")
    result = df.select(selected)
    result = randomize_cps_topcoded_age_80_84(result)

    # Scale weights: CPS ASEC weights have 2 implied decimal places
    # See CPS documentation: A_FNLWGT is expressed in units of 1/100
    # Divide by 100 to get actual population representation
    if "weight" in result.columns:
        result = result.with_columns((pl.col("weight") / 100).alias("weight"))
    if "march_supplement_weight" in result.columns:
        result = result.with_columns(
            (pl.col("march_supplement_weight") / 100).alias("march_supplement_weight")
        )

    # Add derived columns
    if "age" in result.columns:
        result = result.with_columns(
            [
                (pl.col("age") >= 18).alias("is_adult"),
                (pl.col("age") < 18).alias("is_child"),
                (pl.col("age") >= 65).alias("is_senior"),
            ]
        )

    if "race" in result.columns and "cps_race" not in result.columns:
        result = result.with_columns(pl.col("race").alias("cps_race"))
    if "_cps_hispanic_code" in result.columns and "is_hispanic" not in result.columns:
        result = result.with_columns(
            (pl.col("_cps_hispanic_code") != 0).alias("is_hispanic")
        ).drop("_cps_hispanic_code")

    health_staging_columns = [
        f"_{leaf}" for leaf in CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP
    ]
    available_health_staging = [
        column for column in health_staging_columns if column in result.columns
    ]
    if available_health_staging:
        result = result.with_columns(
            [
                (pl.col(f"_{leaf}") == 1).alias(leaf)
                for leaf in CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP
                if f"_{leaf}" in result.columns and leaf not in result.columns
            ]
        )
        result = result.with_columns(
            [
                pl.col(reported_leaf).alias(leaf)
                for leaf, reported_leaf in CURRENT_HEALTH_COVERAGE_RULE_INPUT_ALIAS_MAP.items()
                if reported_leaf in result.columns and leaf not in result.columns
            ]
        )
        if (
            "_reported_has_private_health_coverage_at_interview" in result.columns
            and "reported_has_private_health_coverage_at_interview"
            not in result.columns
        ):
            result = result.with_columns(
                (
                    pl.col("_reported_has_private_health_coverage_at_interview") == 1
                ).alias("reported_has_private_health_coverage_at_interview")
            )
        if (
            "_reported_has_public_health_coverage_at_interview" in result.columns
            and "reported_has_public_health_coverage_at_interview" not in result.columns
        ):
            result = result.with_columns(
                (
                    pl.col("_reported_has_public_health_coverage_at_interview") == 1
                ).alias("reported_has_public_health_coverage_at_interview")
            )
        if "_reported_current_health_coverage_code" in result.columns:
            if "reported_is_insured_at_interview" not in result.columns:
                result = result.with_columns(
                    (pl.col("_reported_current_health_coverage_code") == 1).alias(
                        "reported_is_insured_at_interview"
                    )
                )
            if "reported_is_uninsured_at_interview" not in result.columns:
                result = result.with_columns(
                    (pl.col("_reported_current_health_coverage_code") != 1).alias(
                        "reported_is_uninsured_at_interview"
                    )
                )
        coverage_family_columns = [
            "reported_has_employer_sponsored_health_coverage_at_interview",
            "reported_has_marketplace_health_coverage_at_interview",
            "reported_has_non_marketplace_direct_purchase_health_coverage_at_interview",
            "reported_has_medicare_health_coverage_at_interview",
            "reported_has_means_tested_health_coverage_at_interview",
            "reported_has_tricare_health_coverage_at_interview",
            "reported_has_champva_health_coverage_at_interview",
            "reported_has_va_health_coverage_at_interview",
            "reported_has_indian_health_service_coverage_at_interview",
        ]
        available_coverage_family_columns = [
            column for column in coverage_family_columns if column in result.columns
        ]
        if (
            available_coverage_family_columns
            and "reported_has_multiple_health_coverage_at_interview"
            not in result.columns
        ):
            result = result.with_columns(
                (
                    pl.sum_horizontal(
                        *[
                            pl.col(column).cast(pl.Int8)
                            for column in available_coverage_family_columns
                        ]
                    )
                    > 1
                ).alias("reported_has_multiple_health_coverage_at_interview")
            )
        if (
            "reported_has_marketplace_health_coverage_at_interview" in result.columns
            and "has_marketplace_health_coverage" not in result.columns
        ):
            result = result.with_columns(
                pl.col("reported_has_marketplace_health_coverage_at_interview").alias(
                    "has_marketplace_health_coverage"
                )
            )
        if (
            "reported_has_employer_sponsored_health_coverage_at_interview"
            in result.columns
            and "has_esi" not in result.columns
        ):
            result = result.with_columns(
                pl.col(
                    "reported_has_employer_sponsored_health_coverage_at_interview"
                ).alias("has_esi")
            )
        result = result.drop(
            [
                column
                for column in (
                    *available_health_staging,
                    "_reported_has_private_health_coverage_at_interview",
                    "_reported_has_public_health_coverage_at_interview",
                    "_reported_current_health_coverage_code",
                )
                if column in result.columns
            ]
        )

    if (
        "_high_school_or_college_status" in result.columns
        and "is_full_time_college_student" not in result.columns
    ):
        result = result.with_columns(
            (pl.col("_high_school_or_college_status") == 2).alias(
                "is_full_time_college_student"
            )
        ).drop("_high_school_or_college_status")
    elif "_high_school_or_college_status" in result.columns:
        result = result.drop("_high_school_or_college_status")

    if "_is_paid_hourly_code" in result.columns:
        if "is_paid_hourly" not in result.columns:
            result = result.with_columns(
                (pl.col("_is_paid_hourly_code") == 1).alias("is_paid_hourly")
            )
        if (
            "_hourly_pay_cents" in result.columns
            and "hourly_wage" not in result.columns
        ):
            result = result.with_columns(
                pl.when(
                    (pl.col("_is_paid_hourly_code") == 1)
                    & (pl.col("_hourly_pay_cents") > 0)
                )
                .then(pl.col("_hourly_pay_cents") / 100)
                .otherwise(0.0)
                .alias("hourly_wage")
            )
        result = result.drop(
            [
                column
                for column in ("_is_paid_hourly_code", "_hourly_pay_cents")
                if column in result.columns
            ]
        )
    elif "_hourly_pay_cents" in result.columns:
        result = result.drop("_hourly_pay_cents")

    if (
        "_union_member_code" in result.columns
        and "is_union_member_or_covered" not in result.columns
    ):
        result = result.with_columns(
            (pl.col("_union_member_code") == 1).alias("is_union_member_or_covered")
        ).drop("_union_member_code")
    elif "_union_member_code" in result.columns:
        result = result.drop("_union_member_code")

    if "detailed_occupation_recode" in result.columns:
        occupation = pl.col("detailed_occupation_recode")
        occupation_exprs: list[pl.Expr] = []
        if "has_never_worked" not in result.columns:
            occupation_exprs.append((occupation == 53).alias("has_never_worked"))
        if "is_military" not in result.columns:
            occupation_exprs.append((occupation == 52).alias("is_military"))
        if "is_computer_scientist" not in result.columns:
            occupation_exprs.append((occupation == 8).alias("is_computer_scientist"))
        if "is_farmer_fisher" not in result.columns:
            occupation_exprs.append((occupation == 41).alias("is_farmer_fisher"))
        if "is_executive_administrative_professional" not in result.columns:
            occupation_exprs.append(
                occupation.is_in(
                    [
                        1,
                        2,
                        3,
                        5,
                        6,
                        7,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        18,
                        19,
                        25,
                        26,
                        27,
                        28,
                        29,
                        34,
                        36,
                        38,
                        39,
                        40,
                        42,
                        50,
                    ]
                ).alias("is_executive_administrative_professional")
            )
        if occupation_exprs:
            result = result.with_columns(occupation_exprs)

    if "_detailed_census_occupation_code" in result.columns:
        if "treasury_tipped_occupation_code" not in result.columns:
            tipped_codes = derive_treasury_tipped_occupation_code(
                result["_detailed_census_occupation_code"].to_pandas()
            )
            result = result.with_columns(
                pl.Series("treasury_tipped_occupation_code", tipped_codes)
            )
        if (
            "treasury_tipped_occupation_code" in result.columns
            and "is_tipped_occupation" not in result.columns
        ):
            result = result.with_columns(
                (pl.col("treasury_tipped_occupation_code") > 0).alias(
                    "is_tipped_occupation"
                )
            )
        result = result.drop("_detailed_census_occupation_code")

    if {
        "_other_income_code",
        "_other_income_value",
    }.issubset(set(result.columns)) and "alimony_income" not in result.columns:
        other_income_exprs = [
            pl.when(pl.col("_other_income_code") == ALIMONY_OTHER_INCOME_CODE)
            .then(pl.col("_other_income_value"))
            .otherwise(0)
            .alias("alimony_income")
        ]
        if "strike_benefits" not in result.columns:
            other_income_exprs.append(
                pl.when(
                    pl.col("_other_income_code") == STRIKE_BENEFITS_OTHER_INCOME_CODE
                )
                .then(pl.col("_other_income_value"))
                .otherwise(0)
                .alias("strike_benefits")
            )
        result = result.with_columns(other_income_exprs).drop(
            ["_other_income_code", "_other_income_value"]
        )
    else:
        drop_columns = [
            column
            for column in ("_other_income_code", "_other_income_value")
            if column in result.columns
        ]
        if drop_columns:
            result = result.drop(drop_columns)
    if {
        "_social_security_reason_1",
        "_social_security_reason_2",
        "social_security",
        "age",
    }.issubset(set(result.columns)) and (
        "social_security_disability" not in result.columns
        or "social_security_retirement" not in result.columns
        or "social_security_survivors" not in result.columns
        or "social_security_dependents" not in result.columns
    ):
        reason_1 = pl.col("_social_security_reason_1")
        reason_2 = pl.col("_social_security_reason_2")
        has_retirement_reason = (reason_1 == SOCIAL_SECURITY_RETIREMENT_REASON_CODE) | (
            reason_2 == SOCIAL_SECURITY_RETIREMENT_REASON_CODE
        )
        has_disability_reason = (reason_1 == SOCIAL_SECURITY_DISABILITY_REASON_CODE) | (
            reason_2 == SOCIAL_SECURITY_DISABILITY_REASON_CODE
        )
        has_survivor_reason = reason_1.is_in(
            SOCIAL_SECURITY_SURVIVOR_REASON_CODES
        ) | reason_2.is_in(SOCIAL_SECURITY_SURVIVOR_REASON_CODES)
        has_dependent_reason = reason_1.is_in(
            SOCIAL_SECURITY_DEPENDENT_REASON_CODES
        ) | reason_2.is_in(SOCIAL_SECURITY_DEPENDENT_REASON_CODES)
        unclassified_social_security = (
            (pl.col("social_security") > 0)
            & ~has_retirement_reason
            & ~has_disability_reason
            & ~has_survivor_reason
            & ~has_dependent_reason
        )
        derived_columns: list[pl.Expr] = []
        if "social_security_disability" not in result.columns:
            derived_columns.append(
                (
                    pl.when(has_disability_reason & ~has_retirement_reason)
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                    + pl.when(
                        unclassified_social_security
                        & (pl.col("age") < MINIMUM_RETIREMENT_AGE)
                    )
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                ).alias("social_security_disability")
            )
        if "social_security_retirement" not in result.columns:
            derived_columns.append(
                (
                    pl.when(has_retirement_reason & ~has_disability_reason)
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                    + pl.when(
                        unclassified_social_security
                        & (pl.col("age") >= MINIMUM_RETIREMENT_AGE)
                    )
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                ).alias("social_security_retirement")
            )
        if "social_security_survivors" not in result.columns:
            derived_columns.append(
                (
                    pl.when(
                        has_survivor_reason
                        & ~has_retirement_reason
                        & ~has_disability_reason
                    )
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                ).alias("social_security_survivors")
            )
        if "social_security_dependents" not in result.columns:
            derived_columns.append(
                (
                    pl.when(
                        has_dependent_reason
                        & ~has_retirement_reason
                        & ~has_disability_reason
                        & ~has_survivor_reason
                    )
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                ).alias("social_security_dependents")
            )
        result = result.with_columns(derived_columns).drop(
            ["_social_security_reason_1", "_social_security_reason_2"]
        )
    else:
        drop_columns = [
            column
            for column in (
                "_social_security_reason_1",
                "_social_security_reason_2",
            )
            if column in result.columns
        ]
        if drop_columns:
            result = result.drop(drop_columns)

    private_pension_staging = [
        column
        for column in ("_pension_income", "_annuity_income")
        if column in result.columns
    ]
    if private_pension_staging:
        private_pension_total = pl.sum_horizontal(
            *[pl.col(column) for column in private_pension_staging]
        )
        pension_exprs: list[pl.Expr] = []
        if "pension_income" not in result.columns:
            pension_exprs.append(private_pension_total.alias("pension_income"))
        if "taxable_private_pension_income" not in result.columns:
            pension_exprs.append(
                (private_pension_total * TAXABLE_PENSION_FRACTION).alias(
                    "taxable_private_pension_income"
                )
            )
        if "tax_exempt_private_pension_income" not in result.columns:
            pension_exprs.append(
                (private_pension_total * (1 - TAXABLE_PENSION_FRACTION)).alias(
                    "tax_exempt_private_pension_income"
                )
            )
        if "taxable_pension_income" not in result.columns:
            pension_exprs.append(
                (private_pension_total * TAXABLE_PENSION_FRACTION).alias(
                    "taxable_pension_income"
                )
            )
        result = result.with_columns(pension_exprs).drop(private_pension_staging)

    retirement_distribution_pairs = [
        ("_retirement_distribution_code_1", "_retirement_distribution_value_1"),
        ("_retirement_distribution_code_2", "_retirement_distribution_value_2"),
        (
            "_retirement_distribution_code_1_yng",
            "_retirement_distribution_value_1_yng",
        ),
        (
            "_retirement_distribution_code_2_yng",
            "_retirement_distribution_value_2_yng",
        ),
    ]
    available_retirement_distribution_pairs = [
        (code_column, value_column)
        for code_column, value_column in retirement_distribution_pairs
        if code_column in result.columns and value_column in result.columns
    ]
    if available_retirement_distribution_pairs:
        distribution_by_code = {}
        for code in range(1, 8):
            distribution_by_code[code] = pl.sum_horizontal(
                *[
                    pl.when(pl.col(code_column) == code)
                    .then(pl.col(value_column))
                    .otherwise(0.0)
                    for code_column, value_column in available_retirement_distribution_pairs
                ]
            )
        retirement_distribution_exprs: list[pl.Expr] = []
        if "taxable_401k_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                (distribution_by_code[1] * TAXABLE_401K_DISTRIBUTION_FRACTION).alias(
                    "taxable_401k_distributions"
                )
            )
        if "tax_exempt_401k_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                (
                    distribution_by_code[1] * (1 - TAXABLE_401K_DISTRIBUTION_FRACTION)
                ).alias("tax_exempt_401k_distributions")
            )
        if "taxable_403b_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                (distribution_by_code[2] * TAXABLE_403B_DISTRIBUTION_FRACTION).alias(
                    "taxable_403b_distributions"
                )
            )
        if "tax_exempt_403b_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                (
                    distribution_by_code[2] * (1 - TAXABLE_403B_DISTRIBUTION_FRACTION)
                ).alias("tax_exempt_403b_distributions")
            )
        if "roth_ira_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                distribution_by_code[3].alias("roth_ira_distributions")
            )
        if "regular_ira_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                distribution_by_code[4].alias("regular_ira_distributions")
            )
        if "taxable_ira_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                distribution_by_code[4].alias("taxable_ira_distributions")
            )
        if "tax_exempt_ira_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                distribution_by_code[3].alias("tax_exempt_ira_distributions")
            )
        if "keogh_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                distribution_by_code[5].alias("keogh_distributions")
            )
        if "taxable_sep_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                (distribution_by_code[6] * TAXABLE_SEP_DISTRIBUTION_FRACTION).alias(
                    "taxable_sep_distributions"
                )
            )
        if "tax_exempt_sep_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                (
                    distribution_by_code[6] * (1 - TAXABLE_SEP_DISTRIBUTION_FRACTION)
                ).alias("tax_exempt_sep_distributions")
            )
        if "other_type_retirement_account_distributions" not in result.columns:
            retirement_distribution_exprs.append(
                distribution_by_code[7].alias(
                    "other_type_retirement_account_distributions"
                )
            )
        result = result.with_columns(retirement_distribution_exprs).drop(
            [
                column
                for pair in available_retirement_distribution_pairs
                for column in pair
            ]
        )

    # Split the bundled CPS retirement-contribution total (RETCB_VAL, staged
    # as _retirement_contributions) into the five account-type-specific
    # desired contribution leaves the eCPS contract requires. This mirrors
    # PolicyEngine/policyengine-us-data
    # policyengine_us_data/datasets/cps/cps.py:1500-1552 exactly: a
    # proportional split using IRS SOI / BEA-FRED / Vanguard-PSCA shares
    # (see imputation_parameters.yaml). The leaves are "desired"
    # (pre-statutory-limit) inputs; PolicyEngine-US applies the limits.
    _RETIREMENT_CONTRIBUTION_DESIRED_LEAVES = (
        "self_employed_pension_contributions_desired",
        "traditional_401k_contributions_desired",
        "roth_401k_contributions_desired",
        "traditional_ira_contributions_desired",
        "roth_ira_contributions_desired",
    )
    _RETIREMENT_CONTRIBUTION_CAPPED_LEAVES = (
        "self_employed_pension_contributions",
        "traditional_401k_contributions",
        "roth_401k_contributions",
        "traditional_ira_contributions",
        "roth_ira_contributions",
    )
    if {
        "_retirement_contributions",
        "wage_income",
        "self_employment_income",
    }.issubset(set(result.columns)) and any(
        leaf not in result.columns
        for leaf in (
            _RETIREMENT_CONTRIBUTION_DESIRED_LEAVES
            + _RETIREMENT_CONTRIBUTION_CAPPED_LEAVES
        )
    ):
        retirement_contributions = pl.col("_retirement_contributions")
        has_wages = pl.col("wage_income") > 0
        has_se = pl.col("self_employment_income") > 0
        has_earned_income = has_wages | has_se

        # 1) Self-employed pension: a share of the total, gated on SE income.
        #    No statutory limit applied here (PolicyEngine-US applies it).
        se_pension = (
            pl.when(has_se)
            .then(
                retirement_contributions * SE_PENSION_SHARE_OF_RETIREMENT_CONTRIBUTIONS
            )
            .otherwise(0.0)
        )
        remaining = pl.max_horizontal(
            retirement_contributions - se_pension,
            pl.lit(0.0),
        )

        # 2) Split the remainder into a DC (401k) pool and an IRA pool.
        #    DC requires an employer, so it is gated on wages; the IRA pool
        #    takes whatever is left for anyone with earned income.
        dc_pool = (
            pl.when(has_wages)
            .then(remaining * DC_SHARE_OF_RETIREMENT_CONTRIBUTIONS)
            .otherwise(0.0)
        )
        ira_pool = pl.when(has_earned_income).then(remaining - dc_pool).otherwise(0.0)

        derived_retirement_columns: list[pl.Expr] = []
        if "self_employed_pension_contributions_desired" not in result.columns:
            derived_retirement_columns.append(
                se_pension.alias("self_employed_pension_contributions_desired")
            )
        # DC pool: traditional/Roth 401(k) split.
        if "traditional_401k_contributions_desired" not in result.columns:
            derived_retirement_columns.append(
                (dc_pool * (1 - ROTH_SHARE_OF_DC_CONTRIBUTIONS)).alias(
                    "traditional_401k_contributions_desired"
                )
            )
        if "roth_401k_contributions_desired" not in result.columns:
            derived_retirement_columns.append(
                (dc_pool * ROTH_SHARE_OF_DC_CONTRIBUTIONS).alias(
                    "roth_401k_contributions_desired"
                )
            )
        # IRA pool: traditional/Roth IRA split.
        if "traditional_ira_contributions_desired" not in result.columns:
            derived_retirement_columns.append(
                (ira_pool * TRADITIONAL_SHARE_OF_IRA_CONTRIBUTIONS).alias(
                    "traditional_ira_contributions_desired"
                )
            )
        if "roth_ira_contributions_desired" not in result.columns:
            derived_retirement_columns.append(
                (ira_pool * (1 - TRADITIONAL_SHARE_OF_IRA_CONTRIBUTIONS)).alias(
                    "roth_ira_contributions_desired"
                )
            )
        limit_year = max(
            min(year, max(RETIREMENT_CONTRIBUTION_LIMITS_BY_YEAR)),
            min(RETIREMENT_CONTRIBUTION_LIMITS_BY_YEAR),
        )
        limits = RETIREMENT_CONTRIBUTION_LIMITS_BY_YEAR[limit_year]
        catch_up_eligible = pl.col("age") >= RETIREMENT_CATCH_UP_AGE
        limit_401k = pl.lit(float(limits["401k"])) + (
            catch_up_eligible * float(limits["401k_catch_up"])
        )
        limit_ira = pl.lit(float(limits["ira"])) + (
            catch_up_eligible * float(limits["ira_catch_up"])
        )
        capped_se_pension = (
            pl.when(has_se).then(retirement_contributions).otherwise(0.0)
        )
        capped_remaining_after_se = pl.max_horizontal(
            retirement_contributions - capped_se_pension,
            pl.lit(0.0),
        )
        capped_traditional_401k = (
            pl.when(has_wages)
            .then(pl.min_horizontal(capped_remaining_after_se, limit_401k))
            .otherwise(0.0)
        )
        capped_remaining_after_traditional_401k = pl.max_horizontal(
            capped_remaining_after_se - capped_traditional_401k,
            pl.lit(0.0),
        )
        capped_roth_401k = (
            pl.when(has_wages)
            .then(
                pl.min_horizontal(
                    capped_remaining_after_traditional_401k,
                    limit_401k,
                )
            )
            .otherwise(0.0)
        )
        capped_remaining_after_roth_401k = pl.max_horizontal(
            capped_remaining_after_traditional_401k - capped_roth_401k,
            pl.lit(0.0),
        )
        capped_traditional_ira = (
            pl.when(has_wages)
            .then(pl.min_horizontal(capped_remaining_after_roth_401k, limit_ira))
            .otherwise(0.0)
        )
        capped_remaining_after_traditional_ira = pl.max_horizontal(
            capped_remaining_after_roth_401k - capped_traditional_ira,
            pl.lit(0.0),
        )
        capped_roth_ira_limit = limit_ira - capped_traditional_ira
        capped_roth_ira = (
            pl.when(has_wages)
            .then(
                pl.min_horizontal(
                    capped_remaining_after_traditional_ira,
                    capped_roth_ira_limit,
                )
            )
            .otherwise(0.0)
        )
        capped_retirement_columns = {
            "self_employed_pension_contributions": capped_se_pension,
            "traditional_401k_contributions": capped_traditional_401k,
            "roth_401k_contributions": capped_roth_401k,
            "traditional_ira_contributions": capped_traditional_ira,
            "roth_ira_contributions": capped_roth_ira,
        }
        for column, expression in capped_retirement_columns.items():
            if column not in result.columns:
                derived_retirement_columns.append(expression.alias(column))
        result = result.with_columns(derived_retirement_columns).drop(
            "_retirement_contributions"
        )
    elif "_retirement_contributions" in result.columns:
        result = result.drop("_retirement_contributions")
    disability_columns = [
        column for column in PERSON_CPS_DISABILITY_COLUMNS if column in result.columns
    ]
    if disability_columns:
        # eCPS difficulty_* leaves: PEDIS{X} == 1 -> True. Built from the staging
        # columns before they are dropped below (the same staging feeds
        # is_disabled). These are exported as eCPS dataset columns.
        difficulty_exprs = [
            (pl.col(staging) == 1).alias(leaf)
            for staging, leaf in PERSON_CPS_DIFFICULTY_LEAVES.items()
            if staging in result.columns and leaf not in result.columns
        ]
        if difficulty_exprs:
            result = result.with_columns(difficulty_exprs)
    if disability_columns and "is_disabled" not in result.columns:
        result = result.with_columns(
            pl.any_horizontal(
                *[(pl.col(column) == 1) for column in disability_columns]
            ).alias("is_disabled")
        ).drop(disability_columns)
    elif disability_columns:
        result = result.drop(disability_columns)
    if {
        "_disability_income_1",
        "_disability_income_code_1",
        "_disability_income_2",
        "_disability_income_code_2",
    }.issubset(set(result.columns)) and "disability_benefits" not in result.columns:
        result = result.with_columns(
            (
                pl.when(
                    pl.col("_disability_income_code_1") != WORKERS_COMP_DISABILITY_CODE
                )
                .then(pl.col("_disability_income_1"))
                .otherwise(0)
                + pl.when(
                    pl.col("_disability_income_code_2") != WORKERS_COMP_DISABILITY_CODE
                )
                .then(pl.col("_disability_income_2"))
                .otherwise(0)
            ).alias("disability_benefits")
        ).drop(
            [
                "_disability_income_1",
                "_disability_income_code_1",
                "_disability_income_2",
                "_disability_income_code_2",
            ]
        )
    else:
        drop_columns = [
            column
            for column in (
                "_disability_income_1",
                "_disability_income_code_1",
                "_disability_income_2",
                "_disability_income_code_2",
            )
            if column in result.columns
        ]
        if drop_columns:
            result = result.drop(drop_columns)
    if "_receives_wic" in result.columns and "receives_wic" not in result.columns:
        result = result.with_columns(
            (pl.col("_receives_wic") == 1).alias("receives_wic")
        ).drop("_receives_wic")
    elif "_receives_wic" in result.columns:
        result = result.drop("_receives_wic")
    if (
        "spm_unit_capped_housing_subsidy_reported" in result.columns
        and "receives_housing_assistance" not in result.columns
    ):
        result = result.with_columns(
            (pl.col("spm_unit_capped_housing_subsidy_reported") > 0).alias(
                "receives_housing_assistance"
            )
        )
    if (
        "receives_housing_assistance" in result.columns
        and "takes_up_housing_assistance_if_eligible" not in result.columns
    ):
        result = result.with_columns(
            pl.col("receives_housing_assistance").alias(
                "takes_up_housing_assistance_if_eligible"
            )
        )
    # Unmarried partner of the household head (G8). Mirrors eCPS cps.py:1219
    # `perrp.isin(PERRP_UNMARRIED_PARTNER_OF_HOUSEHOLD_HEAD_CODES)`.
    if (
        "_person_relationship_to_householder" in result.columns
        and "is_unmarried_partner_of_household_head" not in result.columns
    ):
        result = result.with_columns(
            pl.col("_person_relationship_to_householder")
            .is_in(PERRP_UNMARRIED_PARTNER_OF_HOUSEHOLD_HEAD_CODES)
            .alias("is_unmarried_partner_of_household_head")
        ).drop("_person_relationship_to_householder")
    elif "_person_relationship_to_householder" in result.columns:
        result = result.drop("_person_relationship_to_householder")
    # Employer-sponsored insurance policyholder + premium (G6). Mirrors eCPS
    # cps.py:1576-1581: the policyholder flag is `NOW_OWNGRP == 1`, and the
    # premium is `impute_employer_sponsored_insurance_premiums(person)`
    # (eCPS cps.py:229-273), reproduced here on the renamed CPS columns.
    _esi_source_columns = {"_now_owngrp", "_now_hipaid", "_now_grpftyp"}
    if _esi_source_columns.issubset(set(result.columns)):
        own_esi = pl.col("_now_owngrp") == ESI_HAS_CURRENT_OWN_COVERAGE
        premium_status = pl.col("_now_hipaid")
        plan_type = pl.col("_now_grpftyp")
        if "reported_owns_employer_sponsored_health_insurance_at_interview" not in (
            result.columns
        ):
            result = result.with_columns(
                own_esi.alias(
                    "reported_owns_employer_sponsored_health_insurance_at_interview"
                )
            )
        if "employer_sponsored_insurance_premiums" not in result.columns:
            # Employee-paid premium (PHIP_VAL), clipped at zero like eCPS.
            employee_paid = (
                pl.when(pl.col("health_insurance_premiums_without_medicare_part_b") > 0)
                .then(pl.col("health_insurance_premiums_without_medicare_part_b"))
                .otherwise(0.0)
                if "health_insurance_premiums_without_medicare_part_b" in result.columns
                else pl.lit(0.0)
            )
            total_premium = (
                pl.when(plan_type == ESI_SELF_ONLY_PLAN)
                .then(ESI_PLAN_PRIORS_2024["self_only"]["total_premium"])
                .otherwise(ESI_PLAN_PRIORS_2024["family"]["total_premium"])
            )
            average_employee_contribution = (
                pl.when(plan_type == ESI_SELF_ONLY_PLAN)
                .then(ESI_PLAN_PRIORS_2024["self_only"]["employee_contribution"])
                .otherwise(ESI_PLAN_PRIORS_2024["family"]["employee_contribution"])
            )
            employee_share = (
                pl.when(employee_paid > 0)
                .then(employee_paid)
                .otherwise(average_employee_contribution)
            )
            employer_paid_when_some = (total_premium - employee_share).clip(
                lower_bound=0.0
            )
            employer_paid = (
                pl.when(premium_status == ESI_EMPLOYER_PAYS_ALL)
                .then(total_premium)
                .when(premium_status == ESI_EMPLOYER_PAYS_SOME)
                .then(employer_paid_when_some)
                .otherwise(0.0)
            )
            valid_owner_with_plan = own_esi & plan_type.is_in(
                [ESI_FAMILY_PLAN, ESI_SELF_ONLY_PLAN]
            )
            result = result.with_columns(
                pl.when(valid_owner_with_plan)
                .then(employer_paid)
                .otherwise(0.0)
                .alias("employer_sponsored_insurance_premiums")
            )
        result = result.drop([c for c in _esi_source_columns if c in result.columns])
    else:
        result = result.drop([c for c in _esi_source_columns if c in result.columns])
    for value_column in PERSON_ZERO_DEFAULT_VALUE_COLUMNS:
        if value_column not in result.columns:
            result = result.with_columns(pl.lit(0.0).alias(value_column))
    for bool_column in (
        "has_medicare",
        "has_medicaid",
        "has_esi",
        "has_marketplace_health_coverage",
        "receives_wic",
    ):
        if bool_column in result.columns:
            result = result.with_columns((pl.col(bool_column) == 1).alias(bool_column))
    if (
        "has_medicare" in result.columns
        and "takes_up_medicare_if_eligible" not in result.columns
    ):
        result = result.with_columns(
            pl.col("has_medicare").alias("takes_up_medicare_if_eligible")
        )
    if "weeks_unemployed" in result.columns:
        result = result.with_columns(
            pl.when(pl.col("weeks_unemployed") == -1)
            .then(0)
            .otherwise(pl.col("weeks_unemployed"))
            .alias("weeks_unemployed")
        )
    for col in PERSON_NONNEGATIVE_VALUE_COLUMNS:
        if col in result.columns:
            result = result.with_columns(
                pl.when(pl.col(col) < 0).then(0).otherwise(pl.col(col)).alias(col)
            )
    if (
        "marital_status" in result.columns
        and "is_surviving_spouse" not in result.columns
    ):
        result = result.with_columns(
            (pl.col("marital_status") == 4).alias("is_surviving_spouse")
        )
    if "marital_status" in result.columns and "is_separated" not in result.columns:
        result = result.with_columns(
            (pl.col("marital_status") == 6).alias("is_separated")
        )
    if {"household_id", "person_number", "spouse_person_number"}.issubset(
        result.columns
    ) and "marital_unit_id" not in result.columns:
        raw_marital_unit_id = pl.col("household_id").cast(
            pl.Int64
        ) * 1_000_000 + pl.max_horizontal(
            pl.col("person_number").cast(pl.Int64),
            pl.col("spouse_person_number").fill_null(0).cast(pl.Int64),
        )
        result = result.with_columns(
            raw_marital_unit_id.rank("dense").cast(pl.Int64).alias("marital_unit_id")
        )

    # Add year
    result = result.with_columns(pl.lit(year).alias("year"))

    return result


def _attach_cps_ssn_card_type(
    *,
    persons: pl.DataFrame,
    households: pl.DataFrame,
    persons_raw: pl.DataFrame,
) -> pl.DataFrame:
    """Derive PE-style CPS SSN card types from raw CPS columns."""
    if "ssn_card_type" in persons.columns:
        return persons

    fallback = persons.with_columns(pl.lit("CITIZEN").alias("ssn_card_type"))
    required_person_columns = {
        "PRCITSHP",
        "PEINUSYR",
        "PENATVTY",
        "A_HSCOL",
        "A_AGE",
        "A_MARITL",
        "A_SPOUSE",
        "MCARE",
        "CAID",
        "PEN_SC1",
        "PEN_SC2",
        "RESNSS1",
        "RESNSS2",
        "IHSFLG",
        "CHAMPVA",
        "MIL",
        "PEIO1COW",
        "A_MJOCC",
        "SS_YN",
        "SPM_ID",
        "SPM_CAPHOUSESUB",
        "PEAFEVER",
        "SSI_YN",
        "WSAL_VAL",
        "SEMP_VAL",
    }
    if not required_person_columns.issubset(set(persons_raw.columns)):
        return fallback
    if not {"household_id"}.issubset(set(persons.columns)):
        return fallback
    if not {"household_id", "household_weight"}.issubset(set(households.columns)):
        return fallback
    if len(persons_raw) != len(persons):
        return fallback

    household_weights = households.select(
        ["household_id", "household_weight"]
    ).to_pandas()
    household_weight_map = dict(
        zip(
            pd.to_numeric(household_weights["household_id"], errors="coerce"),
            pd.to_numeric(
                household_weights["household_weight"], errors="coerce"
            ).fillna(0.0),
        )
    )
    person_household_ids = pd.to_numeric(
        persons["household_id"].to_pandas(),
        errors="coerce",
    )
    person_weights = (
        person_household_ids.map(household_weight_map).fillna(0.0).to_numpy()
    )

    raw = persons_raw.select(sorted(required_person_columns)).to_pandas()

    def numeric_series(column: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(raw[column], errors="coerce").fillna(default)

    def select_random_subset_to_target(
        eligible_ids: np.ndarray,
        current_weighted: float,
        target_weighted: float,
        *,
        random_seed: int,
    ) -> np.ndarray:
        if len(eligible_ids) == 0:
            return np.array([], dtype=int)

        if current_weighted > target_weighted:
            excess_weighted = current_weighted - target_weighted
            total_reassignable_weight = float(np.sum(person_weights[eligible_ids]))
            if total_reassignable_weight <= 0:
                return np.array([], dtype=int)
            share_to_move = min(excess_weighted / total_reassignable_weight, 1.0)
            rng = np.random.default_rng(seed=random_seed)
            random_draw = rng.random(len(eligible_ids))
            return eligible_ids[random_draw < share_to_move]

        needed_weighted = target_weighted - current_weighted
        total_weight = float(np.sum(person_weights[eligible_ids]))
        if total_weight <= 0:
            return np.array([], dtype=int)
        share_to_move = min(needed_weighted / total_weight, 1.0)
        rng = np.random.RandomState(random_seed)
        n_to_move = int(len(eligible_ids) * share_to_move)
        if n_to_move <= 0:
            return np.array([], dtype=int)
        return rng.choice(
            eligible_ids,
            size=n_to_move,
            replace=False,
        )

    prcitshp = numeric_series("PRCITSHP").astype(int)
    peinusyr = numeric_series("PEINUSYR").astype(int)
    birth_country = numeric_series("PENATVTY").astype(int)
    age = numeric_series("A_AGE").astype(int)
    marital = numeric_series("A_MARITL").astype(int)
    spouse_pointer = numeric_series("A_SPOUSE").astype(int)
    medicare = numeric_series("MCARE").astype(int)
    medicaid = numeric_series("CAID").astype(int)
    pension_source_1 = numeric_series("PEN_SC1").astype(int)
    pension_source_2 = numeric_series("PEN_SC2").astype(int)
    social_security_reason_1 = numeric_series("RESNSS1").astype(int)
    social_security_reason_2 = numeric_series("RESNSS2").astype(int)
    ihs = numeric_series("IHSFLG").astype(int)
    champva = numeric_series("CHAMPVA").astype(int)
    military_insurance = numeric_series("MIL").astype(int)
    class_of_worker = numeric_series("PEIO1COW").astype(int)
    major_occupation = numeric_series("A_MJOCC").astype(int)
    social_security_recipient = numeric_series("SS_YN").astype(int)
    spm_unit_id = numeric_series("SPM_ID")
    capped_housing_subsidy = numeric_series("SPM_CAPHOUSESUB")
    veteran = numeric_series("PEAFEVER").astype(int)
    ssi_recipient = numeric_series("SSI_YN").astype(int)
    wage_income = numeric_series("WSAL_VAL")
    self_employment_income = numeric_series("SEMP_VAL")
    student_status = numeric_series("A_HSCOL").astype(int)

    ssn_card_type = np.zeros(len(raw), dtype=np.int64)
    citizens_mask = prcitshp.isin([1, 2, 3, 4]).to_numpy()
    noncitizens = prcitshp.eq(5).to_numpy()
    ssn_card_type[citizens_mask] = 1

    potentially_undocumented = ~np.isin(ssn_card_type, [1, 2])
    arrived_before_1982 = peinusyr.isin([1, 2, 3, 4, 5, 6, 7]).to_numpy()
    is_naturalized = prcitshp.eq(4).to_numpy()
    is_adult = age.ge(18).to_numpy()
    has_five_plus_years = peinusyr.isin(list(range(8, 27))).to_numpy()
    has_three_plus_years = peinusyr.isin(list(range(8, 28))).to_numpy()
    is_married = marital.isin([1, 2]).to_numpy() & spouse_pointer.gt(0).to_numpy()
    eligible_naturalized = (
        is_naturalized
        & is_adult
        & (has_five_plus_years | (has_three_plus_years & is_married))
    )
    has_medicare = medicare.eq(1).to_numpy()
    has_federal_pension = (
        pension_source_1.isin([3]).to_numpy() | pension_source_2.isin([3]).to_numpy()
    )
    has_ss_disability = (
        social_security_reason_1.isin([2]).to_numpy()
        | social_security_reason_2.isin([2]).to_numpy()
    )
    has_ihs = ihs.eq(1).to_numpy()
    has_medicaid = medicaid.eq(1).to_numpy()
    has_champva = champva.eq(1).to_numpy()
    has_military_insurance = military_insurance.eq(1).to_numpy()
    is_government_worker = class_of_worker.isin([1, 2, 3]).to_numpy()
    is_military_occupation = major_occupation.eq(11).to_numpy()
    is_government_employee = is_government_worker | is_military_occupation
    has_social_security = social_security_recipient.eq(1).to_numpy()
    spm_housing_map = (
        pd.DataFrame(
            {
                "SPM_ID": spm_unit_id,
                "SPM_CAPHOUSESUB": capped_housing_subsidy,
            }
        )
        .dropna(subset=["SPM_ID"])
        .groupby("SPM_ID", sort=False)["SPM_CAPHOUSESUB"]
        .max()
    )
    has_housing_assistance = spm_unit_id.map(spm_housing_map).fillna(0).gt(0).to_numpy()
    is_military_connected = veteran.eq(1).to_numpy() | is_military_occupation
    has_ssi = ssi_recipient.eq(1).to_numpy()

    assumed_documented = (
        arrived_before_1982
        | eligible_naturalized
        | has_medicare
        | has_federal_pension
        | has_ss_disability
        | has_ihs
        | has_medicaid
        | has_champva
        | has_military_insurance
        | is_government_employee
        | has_social_security
        | has_housing_assistance
        | is_military_connected
        | has_ssi
    )
    ssn_card_type[potentially_undocumented & assumed_documented] = 3

    worker_mask = (
        (ssn_card_type != 3)
        & noncitizens
        & ((wage_income.gt(0).to_numpy()) | (self_employment_income.gt(0).to_numpy()))
    )
    student_mask = (ssn_card_type != 3) & noncitizens & student_status.eq(2).to_numpy()

    worker_ids = np.flatnonzero(worker_mask)
    selected_workers = select_random_subset_to_target(
        worker_ids,
        current_weighted=float(np.sum(person_weights[worker_ids])),
        target_weighted=PE_CPS_UNDOCUMENTED_WORKERS_TARGET,
        random_seed=0,
    )
    student_ids = np.flatnonzero(student_mask)
    selected_students = select_random_subset_to_target(
        student_ids,
        current_weighted=float(np.sum(person_weights[student_ids])),
        target_weighted=PE_CPS_UNDOCUMENTED_STUDENTS_TARGET,
        random_seed=1,
    )
    ssn_card_type[selected_workers] = 2
    ssn_card_type[selected_students] = 2

    current_undocumented = float(np.sum(person_weights[ssn_card_type == 0]))
    if current_undocumented < PE_CPS_UNDOCUMENTED_TARGET:
        mixed_household_candidates: list[int] = []
        household_values = person_household_ids.to_numpy()
        for household_id in pd.unique(household_values):
            household_mask = household_values == household_id
            household_codes = ssn_card_type[household_mask]
            if not (np.any(household_codes == 0) and np.any(household_codes == 3)):
                continue
            household_indices = np.flatnonzero(household_mask)
            mixed_household_candidates.extend(
                household_indices[household_codes == 3].tolist()
            )
        if mixed_household_candidates:
            selected_indices = select_random_subset_to_target(
                np.asarray(mixed_household_candidates, dtype=int),
                current_weighted=current_undocumented,
                target_weighted=PE_CPS_UNDOCUMENTED_TARGET,
                random_seed=100,
            )
            ssn_card_type[selected_indices] = 0

    code_to_str = {
        0: "NONE",
        1: "CITIZEN",
        2: "NON_CITIZEN_VALID_EAD",
        3: "OTHER_NON_CITIZEN",
    }
    has_valid_ssn = ssn_card_type == 1
    taxpayer_id_type = np.where(
        has_valid_ssn,
        "VALID_SSN",
        np.where(ssn_card_type != 0, "OTHER_TIN", "NONE"),
    )
    immigration_status = _derive_cps_immigration_status(
        ssn_card_type=ssn_card_type,
        birth_country=birth_country.to_numpy(),
        peinusyr=peinusyr.to_numpy(),
        age=age.to_numpy(),
        year=int(persons["year"][0])
        if "year" in persons.columns and len(persons) > 0
        else 2024,
    )
    return persons.with_columns(
        [
            pl.Series(
                "ssn_card_type",
                pd.Series(ssn_card_type).map(code_to_str).tolist(),
            ),
            pl.Series("has_valid_ssn", has_valid_ssn),
            pl.Series("taxpayer_id_type", taxpayer_id_type.tolist()),
            pl.Series("immigration_status_str", immigration_status.tolist()),
        ]
    )


def _derive_cps_immigration_status(
    *,
    ssn_card_type: np.ndarray,
    birth_country: np.ndarray,
    peinusyr: np.ndarray,
    age: np.ndarray,
    year: int,
) -> np.ndarray:
    """Approximate eCPS immigration-status tags from CPS ASEC citizenship inputs."""

    arrival_year_map = {
        1: 1950,
        2: 1955,
        3: 1960,
        4: 1965,
        5: 1970,
        6: 1975,
        7: 1980,
        8: 1982,
        9: 1984,
        10: 1986,
        11: 1988,
        12: 1990,
        13: 1992,
        14: 1994,
        15: 1996,
        16: 1998,
        17: 2000,
        18: 2002,
        19: 2004,
        20: 2006,
        21: 2008,
        22: 2010,
        23: 2012,
        24: 2014,
        25: 2017,
        26: 2019,
        27: 2021,
        28: 2023,
        29: 2024,
    }
    arrival_years = pd.Series(peinusyr).map(arrival_year_map).fillna(2024).to_numpy()
    years_in_us = year - arrival_years
    age_at_entry = np.maximum(0, age - years_in_us)

    result = np.full(len(ssn_card_type), "LEGAL_PERMANENT_RESIDENT", dtype="U32")
    result[ssn_card_type == 1] = "CITIZEN"

    arrived_before_1982 = np.isin(peinusyr, [1, 2, 3, 4, 5, 6, 7])
    result[(ssn_card_type == 0) & ~arrived_before_1982] = "UNDOCUMENTED"

    cofa_birth_country_codes = {511, 512}
    cuban_haitian_birth_country_codes = {327, 332}
    result[
        (ssn_card_type != 0) & np.isin(birth_country, list(cofa_birth_country_codes))
    ] = "LEGAL_PERMANENT_RESIDENT"
    result[
        (ssn_card_type != 0)
        & np.isin(birth_country, list(cuban_haitian_birth_country_codes))
        & (arrival_years >= 1980)
    ] = "CUBAN_HAITIAN_ENTRANT"
    result[
        (ssn_card_type == 2)
        & (arrival_years <= 2007)
        & (age_at_entry < 16)
        & (age >= 15)
    ] = "DACA"
    result[(ssn_card_type == 3) & (years_in_us <= 5)] = "REFUGEE"
    result[(ssn_card_type == 2) & (result == "LEGAL_PERMANENT_RESIDENT")] = "TPS"
    return result


def _processed_persons_have_household_geography(persons: pl.DataFrame) -> bool:
    """Whether cached processed person data can derive household geography."""
    required_columns = set(PERSON_CACHE_REQUIRED_COLUMNS)
    if not required_columns.issubset(set(persons.columns)):
        return False
    return len(persons["state_fips"].drop_nulls()) > 0


def _process_households(df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Process raw household file into clean format."""
    selected = [
        pl.col(census_name).alias(our_name)
        for census_name, our_name in HOUSEHOLD_VARIABLES.items()
        if census_name in df.columns
    ]
    if not selected:
        raise ValueError("No recognized variables found in household file")
    result = df.select(selected)

    # Scale weights: CPS ASEC weights have 2 implied decimal places
    if "household_weight" in result.columns:
        result = result.with_columns(
            (pl.col("household_weight") / 100).alias("household_weight")
        )

    result = result.with_columns(pl.lit(year).alias("year"))

    return result


def _attach_household_geography_to_persons(
    *,
    persons: pl.DataFrame,
    households: pl.DataFrame,
) -> pl.DataFrame:
    """Propagate household geography onto cached person rows when needed."""
    if "household_id" not in households.columns:
        return persons
    geography_columns = [
        column
        for column in ("state_fips", "county_fips")
        if column in households.columns
    ]
    if not geography_columns:
        return persons
    joined = persons.join(
        households.select(["household_id", *geography_columns]).rename(
            {column: f"_household_{column}" for column in geography_columns}
        ),
        on="household_id",
        how="left",
    )
    for column in geography_columns:
        household_column = f"_household_{column}"
        if column in joined.columns:
            joined = joined.with_columns(
                pl.coalesce(column, household_column).alias(column)
            )
        else:
            joined = joined.with_columns(pl.col(household_column).alias(column))
        joined = joined.drop(household_column)
    return joined


def _derive_households(persons: pl.DataFrame) -> pl.DataFrame:
    """Derive household-level data from person records."""
    if "household_id" not in persons.columns:
        raise ValueError("Cannot derive households without household_id")

    aggregations = [
        pl.len().alias("household_size"),
        pl.col("weight").first().alias("household_weight"),
    ]
    if "state_fips" in persons.columns:
        aggregations.append(pl.col("state_fips").first().alias("state_fips"))
    else:
        aggregations.append(pl.lit(None).alias("state_fips"))
    if "county_fips" in persons.columns:
        aggregations.append(pl.col("county_fips").first().alias("county_fips"))
    else:
        aggregations.append(pl.lit(None).alias("county_fips"))
    if "total_person_income" in persons.columns:
        aggregations.append(
            pl.col("total_person_income").sum().alias("household_total_income")
        )
    else:
        aggregations.append(pl.lit(0).alias("household_total_income"))
    if "is_child" in persons.columns:
        aggregations.append(pl.col("is_child").sum().alias("num_children"))
    else:
        aggregations.append(pl.lit(0).alias("num_children"))
    if "is_adult" in persons.columns:
        aggregations.append(pl.col("is_adult").sum().alias("num_adults"))
    else:
        aggregations.append(pl.lit(0).alias("num_adults"))

    households = persons.group_by("household_id").agg(aggregations)

    if "year" in persons.columns:
        year_val = persons.select("year").unique().to_series()[0]
        households = households.with_columns(pl.lit(year_val).alias("year"))

    return households


def get_available_years() -> list[int]:
    """Return list of available CPS ASEC years."""
    return sorted(CPS_URLS.keys())
