"""Library-first US microplex build pipeline."""

from __future__ import annotations

import logging
import sys
import time
import warnings
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
from microplex.calibration import (
    Calibrator,
    HardConcreteCalibrator,
    LinearConstraint,
    SparseCalibrator,
)
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceDescriptor,
    SourceProvider,
    SourceQuery,
    TimeStructure,
)
from microplex.fusion import FusionPlan
from microplex.geography import GeographyQuery
from microplex.hierarchical import TaxUnitOptimizer
from microplex.synthesizer import Synthesizer
from microplex.targets import TargetQuery, TargetSpec

from microplex_us.data_sources.forbes import (
    ForbesFixedSpine,
    ForbesFixedSpineConfig,
    append_forbes_fixed_spine_tables,
    build_forbes_fixed_spine,
    residualize_targets_for_fixed_spine,
)
from microplex_us.geography import (
    BlockGeography,
    normalize_us_county_fips,
)
from microplex_us.pe_source_impute_engine import (
    PE_SOURCE_IMPUTE_BLOCK_ENGINE,
    PESourceImputeBlockRunRequest,
    PESourceImputeConditionedBlockRunRequest,
)
from microplex_us.pipelines.check_export_columns import (
    _format_report as _format_export_column_report,
)
from microplex_us.pipelines.check_export_columns import (
    compute_column_diff,
    load_contract,
)
from microplex_us.pipelines.donor_imputers import (
    ColumnwiseQRFDonorImputer,
    RegimeAwareDonorImputer,
)
from microplex_us.pipelines.pe_l0 import PolicyEngineL0Calibrator
from microplex_us.pipelines.pe_native_optimization import (
    optimize_policyengine_us_native_loss_dataset,
)
from microplex_us.policyengine.aotc import (
    qualifying_expenses_from_american_opportunity_credit,
)
from microplex_us.policyengine.comparison import (
    evaluate_policyengine_us_target_set,
    slice_policyengine_us_target_evaluation_report,
)
from microplex_us.policyengine.target_profiles import (
    PolicyEngineUSTargetCell,
    resolve_policyengine_us_target_profile,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSDBTargetProvider,
    PolicyEngineUSEntityTableBundle,
    PolicyEngineUSMicrosimulationAdapter,
    PolicyEngineUSQuantityTarget,
    PolicyEngineUSVariableBinding,
    build_policyengine_us_export_column_names,
    build_policyengine_us_export_variable_maps,
    build_policyengine_us_time_period_arrays,
    compile_supported_policyengine_us_household_linear_constraints,
    filter_supported_policyengine_us_targets,
    infer_policyengine_us_variable_bindings,
    load_us_pipeline_checkpoint,
    materialize_policyengine_us_variables_safely,
    policyengine_us_formula_variables_for_targets,
    policyengine_us_variables_to_materialize,
    resolve_policyengine_excluded_export_variables,
    save_us_pipeline_checkpoint,
    write_policyengine_us_time_period_dataset,
)
from microplex_us.policyengine.us import (
    subset_policyengine_tables_by_households as _subset_policyengine_tables_by_households,
)
from microplex_us.targets.arch import resolve_arch_sqlite_target_provider
from microplex_us.variables import (
    PE_STYLE_PUF_IRS_DEMOGRAPHIC_PREDICTORS,
    DonorMatchStrategy,
    VariableSupportFamily,
    donor_imputation_block_specs,
    normalize_dividend_columns,
    normalize_social_security_columns,
    prune_redundant_variables,
    score_donor_condition_var,
    social_security_retirement_compatible_amount,
    variable_semantic_spec_for,
)

LOGGER = logging.getLogger(__name__)

PUF_SUPPORT_CLONE_FLAG_COLUMN = "person_is_puf_clone"

PUF_SUPPORT_CLONE_IMPUTED_VARIABLES: tuple[str, ...] = (
    "employment_income",
    "partnership_s_corp_income",
    "social_security",
    "taxable_pension_income",
    "interest_deduction",
    "tax_exempt_pension_income",
    "long_term_capital_gains",
    "unreimbursed_business_employee_expenses",
    "pre_tax_contributions",
    "taxable_ira_distributions",
    "self_employment_income",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "business_is_sstb",
    "short_term_capital_gains",
    "qualified_dividend_income",
    "charitable_cash_donations",
    "self_employed_pension_contribution_ald",
    "unrecaptured_section_1250_gain",
    "taxable_unemployment_compensation",
    "taxable_interest_income",
    "domestic_production_ald",
    "self_employed_health_insurance_ald",
    "rental_income",
    "non_qualified_dividend_income",
    "cdcc_relevant_expenses",
    "tax_exempt_interest_income",
    "salt_refund_income",
    "foreign_tax_credit",
    "estate_income",
    "charitable_non_cash_donations",
    "american_opportunity_credit",
    "miscellaneous_income",
    "alimony_expense",
    "farm_income",
    "partnership_se_income",
    "alimony_income",
    "health_savings_account_ald",
    "non_sch_d_capital_gains",
    "general_business_credit",
    "energy_efficient_home_improvement_credit",
    "traditional_ira_contributions",
    "amt_foreign_tax_credit",
    "excess_withheld_payroll_tax",
    "savers_credit",
    "student_loan_interest",
    "investment_income_elected_form_4952",
    "early_withdrawal_penalty",
    "prior_year_minimum_tax_credit",
    "farm_rent_income",
    "qualified_tuition_expenses",
    "educator_expense",
    "long_term_capital_gains_on_collectibles",
    "other_credits",
    "casualty_loss",
    "unreported_payroll_tax",
    "recapture_of_investment_credit",
    "deductible_mortgage_interest",
    "qualified_reit_and_ptp_income",
    "qualified_bdc_income",
    "farm_operations_income",
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
    "self_employment_income_would_be_qualified",
)

PUF_SUPPORT_CLONE_CPS_REFRESH_CONDITION_VARIABLES: tuple[str, ...] = (
    "age",
    "is_male",
    "state_fips",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
    "employment_income",
    "self_employment_income",
    "social_security",
)

PUF_SUPPORT_CLONE_CPS_REFRESH_INCOME_VARIABLES: frozenset[str] = frozenset(
    {
        "employment_income",
        "self_employment_income",
        "social_security",
    }
)

# Refresh categorical/status fields against the PUF income surface, but never
# overwrite amount fields here. PUF and CPS income amounts must come from donor
# imputation/calibration, not from post-hoc bucket or nearest-neighbor surgery.
PUF_SUPPORT_CLONE_CPS_REFRESH_VARIABLES: tuple[str, ...] = (
    "is_male",
    "cps_race",
    "is_hispanic",
    "detailed_occupation_recode",
    "treasury_tipped_occupation_code",
    "is_disabled",
    "difficulty_seeing",
    "difficulty_hearing",
    "difficulty_walking_or_climbing_stairs",
    "difficulty_dressing_or_bathing",
    "difficulty_doing_errands",
    "difficulty_remembering_or_making_decisions",
    "meets_ssi_disability_criteria",
    "receives_wic",
    "receives_housing_assistance",
    "is_paid_hourly",
    "is_union_member_or_covered",
)

DEFAULT_ACA_TAKEUP_RATE = 0.672
DEFAULT_DC_PTC_TAKEUP_RATE = 0.32
DEFAULT_EARLY_HEAD_START_TAKEUP_RATE = 0.09
DEFAULT_EITC_TAKEUP_RATES_BY_CHILDREN = {0: 0.65, 1: 0.86, 2: 0.85, 3: 0.85}
DEFAULT_HEAD_START_TAKEUP_RATE = 0.30
DEFAULT_MEDICAID_TAKEUP_RATE = 0.93
DEFAULT_MEDICAID_TAKEUP_RATES_BY_STATE = {
    "AK": 0.88,
    "AL": 0.92,
    "AR": 0.79,
    "AZ": 0.95,
    "CA": 0.78,
    "CO": 0.99,
    "CT": 0.89,
    "DC": 0.99,
    "DE": 0.86,
    "FL": 0.98,
    "GA": 0.73,
    "HI": 0.88,
    "IA": 0.84,
    "ID": 0.78,
    "IL": 0.85,
    "IN": 0.99,
    "KS": 0.92,
    "KY": 0.87,
    "LA": 0.79,
    "MA": 0.94,
    "MD": 0.95,
    "ME": 0.92,
    "MI": 0.91,
    "MN": 0.89,
    "MO": 0.89,
    "MS": 0.75,
    "MT": 0.83,
    "NC": 0.94,
    "ND": 0.91,
    "NE": 0.79,
    "NH": 0.84,
    "NJ": 0.74,
    "NM": 0.84,
    "NV": 0.93,
    "NY": 0.86,
    "OH": 0.82,
    "OK": 0.77,
    "OR": 0.92,
    "PA": 0.64,
    "RI": 0.94,
    "SC": 0.93,
    "SD": 0.88,
    "TN": 0.92,
    "TX": 0.76,
    "UT": 0.53,
    "VA": 0.82,
    "VT": 0.93,
    "WA": 0.98,
    "WI": 0.91,
    "WV": 0.83,
    "WY": 0.70,
}
DEFAULT_SNAP_TAKEUP_RATE = 0.82
DEFAULT_TANF_TAKEUP_RATE = 0.22
DEFAULT_VOLUNTARY_FILING_RATE = 0.05
DEFAULT_VOLUNTARY_FILING_RATES = {
    "no_children": {
        "zero": {"under_65": 0.20, "age_65_plus": 0.05},
        "low": {"under_65": 0.24, "age_65_plus": 0.04},
        "medium": {"under_65": 0.0, "age_65_plus": 0.0},
        "high": {"under_65": 0.0, "age_65_plus": 0.005},
    },
    "with_children": {
        "zero": {"under_65": 0.50, "age_65_plus": 0.075},
        "low": {"under_65": 0.60, "age_65_plus": 0.06},
        "medium": {"under_65": 0.0, "age_65_plus": 0.0},
        "high": {"under_65": 0.025, "age_65_plus": 0.0037},
    },
}
WIC_TAKEUP_CATEGORY_PREGNANT = "PREGNANT"
WIC_TAKEUP_CATEGORY_POSTPARTUM = "POSTPARTUM"
WIC_TAKEUP_CATEGORY_BREASTFEEDING = "BREASTFEEDING"
WIC_TAKEUP_CATEGORY_INFANT = "INFANT"
WIC_TAKEUP_CATEGORY_CHILD = "CHILD"
WIC_TAKEUP_CATEGORY_NONE = "NONE"
DEFAULT_WIC_TAKEUP_RATES = {
    WIC_TAKEUP_CATEGORY_PREGNANT: 0.456,
    WIC_TAKEUP_CATEGORY_POSTPARTUM: 0.689,
    WIC_TAKEUP_CATEGORY_BREASTFEEDING: 0.663,
    WIC_TAKEUP_CATEGORY_INFANT: 0.784,
    WIC_TAKEUP_CATEGORY_CHILD: 0.460,
    WIC_TAKEUP_CATEGORY_NONE: 0.0,
}
DEFAULT_WIC_NUTRITIONAL_RISK_RATES = {
    WIC_TAKEUP_CATEGORY_PREGNANT: 0.913,
    WIC_TAKEUP_CATEGORY_POSTPARTUM: 0.933,
    WIC_TAKEUP_CATEGORY_BREASTFEEDING: 0.889,
    WIC_TAKEUP_CATEGORY_INFANT: 0.950,
    WIC_TAKEUP_CATEGORY_CHILD: 0.752,
    WIC_TAKEUP_CATEGORY_NONE: 0.0,
}
EITC_TAKEUP_CHILD_COUNT_HELPER_COLUMN = "_mp_eitc_child_count_for_takeup"
VOLUNTARY_FILING_AGE_HEAD_HELPER_COLUMN = "_mp_voluntary_filing_age_head"
VOLUNTARY_FILING_WAGE_INCOME_HELPER_COLUMN = "_mp_voluntary_filing_wage_income"


def _stable_string_hash(value: str) -> np.uint64:
    """Deterministic string hash for reproducible MP stochastic inputs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow encountered", RuntimeWarning)
        hashed = np.uint64(0)
        for byte in value.encode("utf-8"):
            hashed = hashed * np.uint64(31) + np.uint64(byte)
        hashed = hashed ^ (hashed >> np.uint64(33))
        hashed = hashed * np.uint64(0xFF51AFD7ED558CCD)
        hashed = hashed ^ (hashed >> np.uint64(33))
    return hashed


def _microplex_seeded_rng(
    variable_name: str,
    *,
    salt: str | None = None,
) -> np.random.Generator:
    key = variable_name if salt is None else f"{variable_name}:{salt}"
    seed = int(_stable_string_hash(key)) % (2**63)
    return np.random.default_rng(seed=seed)


def _load_microplex_takeup_rate(variable_name: str, year: int) -> float:
    """Load MP-owned scalar take-up assumptions for PE dataset inputs."""
    if variable_name == "aca":
        return DEFAULT_ACA_TAKEUP_RATE
    if variable_name == "dc_ptc":
        return DEFAULT_DC_PTC_TAKEUP_RATE
    if variable_name == "early_head_start":
        return DEFAULT_EARLY_HEAD_START_TAKEUP_RATE
    if variable_name == "head_start":
        return 0.40 if year <= 2020 else DEFAULT_HEAD_START_TAKEUP_RATE
    if variable_name == "snap":
        return DEFAULT_SNAP_TAKEUP_RATE
    if variable_name == "tanf":
        return DEFAULT_TANF_TAKEUP_RATE
    raise KeyError(f"Unknown Microplex take-up rate: {variable_name!r}")


def _load_microplex_medicaid_takeup_rates(year: int) -> dict[str, float]:
    """Load MP-owned Medicaid take-up rates by state abbreviation."""
    _ = year
    return dict(DEFAULT_MEDICAID_TAKEUP_RATES_BY_STATE)


def _load_microplex_eitc_takeup_rates(year: int) -> dict[int, float]:
    """Load MP-owned EITC take-up rates by capped qualifying-child count."""
    _ = year
    return dict(DEFAULT_EITC_TAKEUP_RATES_BY_CHILDREN)


def _load_microplex_voluntary_filing_rates(year: int) -> dict:
    """Load MP-owned voluntary filing rate table."""
    _ = year
    return {
        children: {wage: dict(age_rates) for wage, age_rates in wage_rates.items()}
        for children, wage_rates in DEFAULT_VOLUNTARY_FILING_RATES.items()
    }


def _load_microplex_wic_takeup_rates(year: int) -> dict[str, float]:
    """Load MP-owned WIC take-up rates by demographic category."""
    _ = year
    return dict(DEFAULT_WIC_TAKEUP_RATES)


def _load_microplex_wic_nutritional_risk_rates(year: int) -> dict[str, float]:
    """Load MP-owned WIC nutritional-risk rates by demographic category."""
    _ = year
    return dict(DEFAULT_WIC_NUTRITIONAL_RISK_RATES)


PUF_SUPPORT_CLONE_OVERRIDDEN_VARIABLES: tuple[str, ...] = (
    "partnership_s_corp_income",
    "interest_deduction",
    "unreimbursed_business_employee_expenses",
    "pre_tax_contributions",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "business_is_sstb",
    "charitable_cash_donations",
    "self_employed_pension_contribution_ald",
    "unrecaptured_section_1250_gain",
    "taxable_unemployment_compensation",
    "domestic_production_ald",
    "self_employed_health_insurance_ald",
    "cdcc_relevant_expenses",
    "salt_refund_income",
    "foreign_tax_credit",
    "estate_income",
    "charitable_non_cash_donations",
    "american_opportunity_credit",
    "miscellaneous_income",
    "alimony_expense",
    "health_savings_account_ald",
    "non_sch_d_capital_gains",
    "general_business_credit",
    "energy_efficient_home_improvement_credit",
    "amt_foreign_tax_credit",
    "excess_withheld_payroll_tax",
    "savers_credit",
    "student_loan_interest",
    "investment_income_elected_form_4952",
    "early_withdrawal_penalty",
    "prior_year_minimum_tax_credit",
    "farm_rent_income",
    "qualified_tuition_expenses",
    "educator_expense",
    "long_term_capital_gains_on_collectibles",
    "other_credits",
    "casualty_loss",
    "unreported_payroll_tax",
    "recapture_of_investment_credit",
    "deductible_mortgage_interest",
    "qualified_reit_and_ptp_income",
    "qualified_bdc_income",
    "farm_operations_income",
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
)

PUF_SUPPORT_CLONE_SPECIAL_VARIABLES: tuple[str, ...] = ("weeks_unemployed",)


@lru_cache(maxsize=1)
def _default_block_geography() -> BlockGeography:
    return BlockGeography()


def _normalize_household_county_fips_series(
    county_fips: pd.Series,
    state_fips: pd.Series,
) -> pd.Series:
    """Normalize CPS county fragments into PE's five-digit county FIPS values."""
    county_numeric = pd.to_numeric(county_fips, errors="coerce")
    state_numeric = pd.to_numeric(state_fips, errors="coerce")
    combined = county_numeric.copy()
    county_fragment_mask = (
        county_numeric.notna()
        & county_numeric.gt(0)
        & county_numeric.lt(1000)
        & state_numeric.notna()
        & state_numeric.gt(0)
    )
    combined.loc[county_fragment_mask] = state_numeric.loc[
        county_fragment_mask
    ].round().astype(int) * 1000 + county_numeric.loc[
        county_fragment_mask
    ].round().astype(int)
    normalized = combined.round().astype("Int64").astype("string").str.zfill(5)
    invalid = combined.isna() | combined.le(0)
    return normalized.mask(invalid).astype("string")


def _normalize_household_state_fips_series(state_fips: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(state_fips, errors="coerce")
    normalized = numeric.round().astype("Int64").astype("string").str.zfill(2)
    return normalized.mask(numeric.isna() | numeric.le(0)).astype("string")


def _congressional_district_geoid_from_cd_id(
    cd_id: Any,
    state_fips: Any,
) -> int:
    try:
        state = int(str(state_fips).strip())
    except (TypeError, ValueError):
        return 0
    cd_text = str(cd_id).strip()
    if not cd_text or cd_text.lower() in {"nan", "none", "<na>"}:
        return 0
    district_token = cd_text.split("-")[-1]
    # eCPS normalizes at-large districts to 01: the raw Census codes "AL"/"ZZ"
    # (at-large) and "98" (DC) map to district 0, which is then bumped to 1
    # (policyengine-us-data db/create_initial_strata.py). Microplex's crosswalk
    # feeds the "-AL" token, but accept the raw Census forms too so the encoder
    # stays faithful to the eCPS 436-CD universe regardless of input convention.
    if district_token.upper() in {"AL", "ZZ"}:
        district = 1
    else:
        try:
            district = int(district_token)
        except ValueError:
            return 0
        if district in (0, 98):
            district = 1
    return state * 100 + district


def _attach_household_census_geographies(
    households: pd.DataFrame,
    *,
    seed: int,
    geography: BlockGeography | None = None,
) -> pd.DataFrame:
    """Attach eCPS-contract block, tract, county, and CD geographies to households."""
    # Intermediate frames are indexed by row label and written back via .loc;
    # a non-unique household-frame index makes those reindex operations ambiguous
    # (ValueError: cannot reindex on an axis with duplicate labels). The caller
    # consumes this result by merging on the household_id column, not the index,
    # so collapsing to a fresh RangeIndex here is both safe and robust.
    result = households.reset_index(drop=True)
    for column, default in (
        ("block_geoid", ""),
        ("tract_geoid", ""),
        ("congressional_district_geoid", 0),
    ):
        if column not in result.columns:
            result[column] = default
    if result.empty or "state_fips" not in result.columns:
        return result

    assigned_blocks = pd.Series(pd.NA, index=result.index, dtype="string")
    state_values = _normalize_household_state_fips_series(result["state_fips"])
    county_values = (
        _normalize_household_county_fips_series(
            result["county_fips"], result["state_fips"]
        )
        if "county_fips" in result.columns
        else pd.Series(pd.NA, index=result.index, dtype="string")
    )
    result["county_fips"] = county_values.fillna("00000")

    try:
        block_geography = geography or _default_block_geography()
        block_data = block_geography.data
    except FileNotFoundError:
        return result

    valid_counties = set(block_data["county_fips"].dropna().astype(str))
    county_mask = county_values.isin(valid_counties)
    if county_mask.any():
        county_query = GeographyQuery(
            partition_columns=("county_fips",),
            partition_normalizers={"county_fips": normalize_us_county_fips},
        )
        county_assigner = block_geography.load_assigner(county_query)
        county_frame = pd.DataFrame(
            {"county_fips": county_values.loc[county_mask]},
            index=result.index[county_mask],
        )
        assigned = county_assigner.assign(
            county_frame,
            random_state=seed,
        )
        assigned_blocks.loc[assigned.index] = assigned["block_geoid"].astype("string")

    remaining_mask = assigned_blocks.isna()
    state_mask = remaining_mask & state_values.notna()
    if state_mask.any():
        state_frame = pd.DataFrame(
            {"state_fips": state_values.loc[state_mask]},
            index=result.index[state_mask],
        )
        assigned = block_geography.assign(
            state_frame,
            random_state=seed + 1,
        )
        assigned_blocks.loc[assigned.index] = assigned["block_geoid"].astype("string")

    assigned_mask = assigned_blocks.notna()
    if not assigned_mask.any():
        return result

    materialized = block_geography.materialize(
        pd.DataFrame(
            {
                "_row_index": assigned_blocks.index[assigned_mask],
                "block_geoid": assigned_blocks.loc[assigned_mask].astype(str),
            },
            index=result.index[assigned_mask],
        ),
        columns=("state_fips", "county_fips", "tract_geoid", "cd_id"),
    )
    row_index = materialized["_row_index"].to_numpy()
    for column in ("block_geoid", "tract_geoid", "county_fips"):
        if column in materialized.columns:
            result.loc[row_index, column] = materialized[column].to_numpy()
    result.loc[row_index, "state_fips"] = (
        pd.to_numeric(materialized["state_fips"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    result.loc[row_index, "congressional_district_geoid"] = [
        _congressional_district_geoid_from_cd_id(cd_id, state_fips)
        for cd_id, state_fips in zip(
            materialized.get("cd_id", pd.Series(index=row_index)),
            materialized["state_fips"],
            strict=False,
        )
    ]
    return result


def _root_logger_has_handlers() -> bool:
    return bool(logging.getLogger().handlers)


def _format_progress_values(values: Iterable[Any], *, limit: int = 6) -> str:
    rendered = [str(value) for value in values]
    if len(rendered) <= limit:
        return ",".join(rendered)
    return ",".join(rendered[:limit]) + f",...(+{len(rendered) - limit})"


def _emit_us_pipeline_progress(message: str, /, **context: object) -> None:
    details = ", ".join(
        f"{key}={value}"
        for key, value in context.items()
        if value is not None and value != ""
    )
    line = f"{message} [{details}]" if details else message
    LOGGER.info(line)
    if not LOGGER.handlers and not _root_logger_has_handlers():
        print(line, file=sys.stderr, flush=True)


STATE_FIPS = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}

AGE_BINS = [0, 18, 35, 55, 65, np.inf]


AGE_LABELS = ["0-17", "18-34", "35-54", "55-64", "65+"]
INCOME_BINS = [-np.inf, 25_000, 50_000, 100_000, np.inf]
INCOME_LABELS = ["<25k", "25-50k", "50-100k", "100k+"]
ENTITY_ID_COLUMNS = {
    EntityType.PERSON: "person_id",
    EntityType.HOUSEHOLD: "household_id",
    EntityType.TAX_UNIT: "tax_unit_id",
    EntityType.SPM_UNIT: "spm_unit_id",
    EntityType.FAMILY: "family_id",
}
TINY_WEIGHT_THRESHOLD = 1e-8
DEFAULT_POLICYENGINE_CALIBRATION_MAX_CONSTRAINTS_PER_HOUSEHOLD = 1.0
DEFAULT_POLICYENGINE_CALIBRATION_MIN_ACTIVE_HOUSEHOLDS = 5
CALIBRATION_FEASIBILITY_DROP_WARNING_THRESHOLD = 0.2
STATE_PROGRAM_SUPPORT_PROXY_VARIABLES = (
    "has_medicaid",
    "public_assistance",
    "ssi",
    "social_security",
)
STATE_PROGRAM_AUTO_CONDITION_VARIABLES = ("has_medicaid",)


def _summarize_weight_diagnostics(
    weights: pd.Series | np.ndarray | list[float],
    *,
    tiny_threshold: float = TINY_WEIGHT_THRESHOLD,
) -> dict[str, Any]:
    """Summarize whether a calibrated weight vector looks numerically healthy."""
    series = (
        pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).astype(float)
    )
    row_count = int(len(series))
    if row_count == 0:
        return {
            "row_count": 0,
            "positive_count": 0,
            "nonpositive_count": 0,
            "tiny_count": 0,
            "tiny_share": 0.0,
            "total_weight": 0.0,
            "min_weight": 0.0,
            "p01_weight": 0.0,
            "p50_weight": 0.0,
            "p99_weight": 0.0,
            "max_weight": 0.0,
            "effective_sample_size": 0.0,
            "collapse_suspected": True,
        }

    total_weight = float(series.sum())
    squared_weight_sum = float(np.square(series).sum())
    positive_count = int((series > 0.0).sum())
    nonpositive_count = row_count - positive_count
    tiny_count = int((series <= tiny_threshold).sum())
    tiny_share = float(tiny_count / row_count)
    effective_sample_size = (
        float((total_weight * total_weight) / squared_weight_sum)
        if squared_weight_sum > 0.0
        else 0.0
    )
    effective_sample_ratio = (
        float(effective_sample_size / positive_count) if positive_count > 0 else 0.0
    )
    collapse_suspected = bool(
        total_weight <= tiny_threshold
        or positive_count == 0
        or tiny_share >= 0.95
        or effective_sample_ratio <= 0.25
    )
    return {
        "row_count": row_count,
        "positive_count": positive_count,
        "nonpositive_count": nonpositive_count,
        "tiny_count": tiny_count,
        "tiny_share": tiny_share,
        "total_weight": total_weight,
        "min_weight": float(series.min()),
        "p01_weight": float(series.quantile(0.01)),
        "p50_weight": float(series.quantile(0.5)),
        "p99_weight": float(series.quantile(0.99)),
        "max_weight": float(series.max()),
        "effective_sample_size": effective_sample_size,
        "effective_sample_ratio": effective_sample_ratio,
        "collapse_suspected": collapse_suspected,
    }


def _state_program_support_proxy_summary(
    available_columns: set[str],
) -> dict[str, list[str]]:
    available = sorted(
        variable
        for variable in STATE_PROGRAM_SUPPORT_PROXY_VARIABLES
        if variable in available_columns
    )
    missing = sorted(
        variable
        for variable in STATE_PROGRAM_SUPPORT_PROXY_VARIABLES
        if variable not in available_columns
    )
    return {
        "available": available,
        "missing": missing,
    }


def _subset_policyengine_linear_constraints(
    constraints: tuple[LinearConstraint, ...] | list[LinearConstraint],
    household_mask: np.ndarray,
) -> tuple[LinearConstraint, ...]:
    mask = np.asarray(household_mask, dtype=bool)
    subset: list[LinearConstraint] = []
    for constraint in constraints:
        coefficients = np.asarray(constraint.coefficients, dtype=float)
        if len(coefficients) != len(mask):
            raise ValueError(
                "PolicyEngine linear constraint coefficients do not match household mask length"
            )
        subset.append(
            LinearConstraint(
                name=constraint.name,
                coefficients=coefficients[mask],
                target=float(constraint.target),
            )
        )
    return tuple(subset)


def _policyengine_target_geo_priority(target: TargetSpec) -> int:
    geo_level = str(target.metadata.get("geo_level", "")).lower()
    return {
        "national": 0,
        "state": 1,
        "district": 2,
    }.get(geo_level, 99)


def _constraint_active_household_count(
    constraint: Any,
    *,
    epsilon: float = 1e-12,
    metadata_lookup: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Count households with nonzero coefficient. Uses ``metadata_lookup`` when provided."""
    if metadata_lookup is not None:
        cached = metadata_lookup.get(getattr(constraint, "name", None))
        if cached is not None and "active_households" in cached:
            return int(cached["active_households"])
    coefficients = np.asarray(getattr(constraint, "coefficients", ()), dtype=float)
    if coefficients.size == 0:
        return 0
    return int(np.count_nonzero(np.abs(coefficients) > epsilon))


def _precompute_constraint_metadata(
    constraints: tuple[Any, ...],
    *,
    epsilon: float = 1e-12,
) -> dict[str, dict[str, Any]]:
    """Per-constraint {active_households, coefficient_mass} scalar metadata."""
    metadata: dict[str, dict[str, Any]] = {}
    for constraint in constraints:
        name = getattr(constraint, "name", None)
        if name is None:
            continue
        coefficients = np.asarray(getattr(constraint, "coefficients", ()), dtype=float)
        if coefficients.size == 0:
            metadata[name] = {
                "active_households": 0,
                "coefficient_mass": 0.0,
            }
            continue
        metadata[name] = {
            "active_households": int(np.count_nonzero(np.abs(coefficients) > epsilon)),
            "coefficient_mass": float(np.abs(coefficients).sum()),
        }
    return metadata


def _strip_constraint_coefficients(
    constraints: tuple[Any, ...],
) -> tuple[LinearConstraint, ...]:
    """Replace each constraint's coefficient array with a zero-length sentinel."""
    return tuple(
        LinearConstraint(
            name=c.name, coefficients=np.zeros(0, dtype=float), target=float(c.target)
        )
        for c in constraints
    )


def _build_policyengine_constraint_records(
    targets: list[TargetSpec],
    constraints: tuple[Any, ...],
    *,
    metadata_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target, constraint in zip(targets, constraints, strict=True):
        aggregation_name = str(
            getattr(getattr(target, "aggregation", None), "name", target.aggregation)
        ).upper()
        name = getattr(constraint, "name", None)
        cached = (
            metadata_lookup.get(name)
            if metadata_lookup is not None and name is not None
            else None
        )
        if cached is not None and "coefficient_mass" in cached:
            coefficient_mass = float(cached["coefficient_mass"])
        else:
            coefficient_mass = float(
                np.abs(
                    np.asarray(getattr(constraint, "coefficients", ()), dtype=float)
                ).sum()
            )
        records.append(
            {
                "target": target,
                "constraint": constraint,
                "active_households": _constraint_active_household_count(
                    constraint, metadata_lookup=metadata_lookup
                ),
                "geo_priority": _policyengine_target_geo_priority(target),
                "aggregation_priority": 0 if aggregation_name == "COUNT" else 1,
                "coefficient_mass": coefficient_mass,
            }
        )
    return records


def _policyengine_target_has_entity_table(
    target: TargetSpec,
    tables: PolicyEngineUSEntityTableBundle,
) -> bool:
    return {
        EntityType.HOUSEHOLD: tables.households,
        EntityType.PERSON: tables.persons,
        EntityType.TAX_UNIT: tables.tax_units,
        EntityType.SPM_UNIT: tables.spm_units,
        EntityType.FAMILY: tables.families,
    }.get(target.entity) is not None


def _policyengine_target_variable_name(target: TargetSpec) -> str:
    metadata = dict(target.metadata or {})
    variable = metadata.get("variable")
    if variable is not None:
        return str(variable)
    if target.measure is not None:
        return str(target.measure)
    aggregation_name = str(
        getattr(getattr(target, "aggregation", None), "name", target.aggregation)
    ).upper()
    if aggregation_name == "COUNT":
        entity_value = (
            target.entity.value
            if isinstance(target.entity, EntityType)
            else str(target.entity)
        )
        return f"{entity_value}_count"
    return "unknown"


def _policyengine_target_family_key(target: TargetSpec) -> str:
    metadata = dict(target.metadata or {})
    geo_level = str(metadata.get("geo_level") or "unspecified")
    domain_variable = str(metadata.get("domain_variable") or "")
    variable = _policyengine_target_variable_name(target)
    parts = [geo_level, variable]
    if domain_variable:
        parts.append(f"domain={domain_variable}")
    return "|".join(parts)


def _policyengine_target_loss_family_key(entry: dict[str, Any]) -> str:
    variable = str(entry.get("variable") or "unknown")
    domain_variable = str(entry.get("domain_variable") or "")
    if domain_variable:
        return f"{variable}|domain={domain_variable}"
    return variable


def _policyengine_target_loss_geography_key(entry: dict[str, Any]) -> str:
    geo_level = str(entry.get("geo_level") or "unspecified")
    geographic_id = entry.get("geographic_id")
    if geographic_id is None or str(geographic_id) == "":
        return geo_level
    geographic_key = str(geographic_id).strip()
    if geo_level == "national":
        return f"{geo_level}:US"
    if geo_level == "state":
        try:
            state_fips = int(geographic_key)
        except (TypeError, ValueError):
            geographic_key = geographic_key.upper()
        else:
            geographic_key = STATE_FIPS.get(state_fips, f"{state_fips:02d}")
    return f"{geo_level}:{geographic_key}"


def _select_ssi_takeup_by_age_amount(
    *,
    person_ids: pd.Series,
    ages: pd.Series,
    weights: pd.Series,
    reported_ssi: pd.Series,
    full_takeup_ssi: pd.Series,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select SSI takeup records to match reported SSI dollars by age group."""
    index = person_ids.index
    person_ids = person_ids.reindex(index)
    age_values = pd.to_numeric(ages.reindex(index), errors="coerce").fillna(0.0)
    weight_values = (
        pd.to_numeric(weights.reindex(index), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    reported_values = (
        pd.to_numeric(reported_ssi.reindex(index), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    full_values = (
        pd.to_numeric(full_takeup_ssi.reindex(index), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    reported_amount = (reported_values * weight_values).to_numpy(dtype=float)
    full_amount = (full_values * weight_values).to_numpy(dtype=float)
    reported_positive = reported_values.to_numpy(dtype=float) > 0.0
    formula_positive = full_values.to_numpy(dtype=float) > 0.0
    age_array = age_values.to_numpy(dtype=float)
    selected = np.zeros(len(index), dtype=bool)
    stable_rank = pd.util.hash_pandas_object(
        person_ids.astype("string"),
        index=False,
    ).to_numpy(dtype=np.uint64)

    def _select_until_amount(candidate_mask: np.ndarray, amount: float) -> None:
        if amount <= 0.0:
            return
        candidate_index = np.flatnonzero(candidate_mask & ~selected)
        if candidate_index.size == 0:
            return
        ordered = candidate_index[
            np.argsort(stable_rank[candidate_index], kind="stable")
        ]
        cumulative = np.cumsum(full_amount[ordered])
        cutoff = int(np.searchsorted(cumulative, amount, side="left"))
        selected[ordered[: min(cutoff + 1, len(ordered))]] = True

    groups = {
        "aged": age_array >= 65,
        "under65": age_array < 65,
    }
    group_summary: dict[str, Any] = {}
    for group_name, group_mask in groups.items():
        source_candidates = group_mask & reported_positive & formula_positive
        other_candidates = group_mask & ~reported_positive & formula_positive
        target_amount = float(reported_amount[group_mask].sum())
        _select_until_amount(source_candidates, target_amount)
        selected_amount = float(full_amount[selected & group_mask].sum())
        _select_until_amount(other_candidates, target_amount - selected_amount)
        selected_amount = float(full_amount[selected & group_mask].sum())
        group_summary[group_name] = {
            "reported_amount": target_amount,
            "reported_recipients": float(
                weight_values.to_numpy(dtype=float)[
                    group_mask & reported_positive
                ].sum()
            ),
            "formula_all_takeup_amount": float(full_amount[group_mask].sum()),
            "formula_all_takeup_recipients": float(
                weight_values.to_numpy(dtype=float)[group_mask & formula_positive].sum()
            ),
            "selected_amount": selected_amount,
            "selected_recipients": float(
                weight_values.to_numpy(dtype=float)[group_mask & selected].sum()
            ),
            "source_candidate_amount": float(full_amount[source_candidates].sum()),
            "other_candidate_amount": float(full_amount[other_candidates].sum()),
        }

    weight_array = weight_values.to_numpy(dtype=float)
    summary = {
        "enabled": True,
        "method": "reported_ssi_amount_by_age_group",
        "reported_amount": float(reported_amount.sum()),
        "reported_recipients": float(weight_array[reported_positive].sum()),
        "formula_all_takeup_amount": float(full_amount.sum()),
        "formula_all_takeup_recipients": float(weight_array[formula_positive].sum()),
        "selected_amount": float(full_amount[selected].sum()),
        "selected_recipients": float(weight_array[selected].sum()),
        "groups": group_summary,
    }
    return selected, summary


def _policyengine_target_ledger_entry(
    *,
    target: TargetSpec,
    stage: str,
    reason: str,
    household_count: int,
    active_households: int | None = None,
    min_active_households: int | None = None,
    missing_features: Iterable[str] = (),
    failed_materializations: Iterable[str] = (),
) -> dict[str, Any]:
    metadata = dict(target.metadata or {})
    required_features = sorted(str(feature) for feature in target.required_features)
    entity_value = (
        target.entity.value
        if isinstance(target.entity, EntityType)
        else str(target.entity)
    )
    aggregation_value = getattr(target.aggregation, "value", str(target.aggregation))
    active_support_share = None
    if active_households is not None and household_count > 0:
        active_support_share = float(active_households / household_count)
    return {
        "target_name": target.name,
        "target_id": metadata.get("target_id"),
        "stratum_id": metadata.get("stratum_id"),
        "stage": stage,
        "reason": reason,
        "family": _policyengine_target_family_key(target),
        "entity": entity_value,
        "aggregation": aggregation_value,
        "measure": target.measure,
        "value": float(target.value),
        "geo_level": metadata.get("geo_level"),
        "geographic_id": metadata.get("geographic_id"),
        "variable": _policyengine_target_variable_name(target),
        "domain_variable": metadata.get("domain_variable"),
        "filters": [
            {
                "feature": target_filter.feature,
                "operator": target_filter.operator,
                "value": target_filter.value,
            }
            for target_filter in target.filters
        ],
        "required_features": required_features,
        "missing_features": sorted(str(feature) for feature in missing_features),
        "failed_materializations": sorted(
            str(feature) for feature in failed_materializations
        ),
        "active_households": active_households,
        "active_support_share": active_support_share,
        "min_active_households": min_active_households,
        "source": target.source,
        "description": target.description,
    }


def _summarize_policyengine_target_ledger(
    ledger: list[dict[str, Any]],
    *,
    compiled_target_count: int,
    preselection_target_count: int,
    final_solve_target_count: int,
) -> dict[str, Any]:
    stage_order = ("solve_now", "solve_later", "audit_only")
    stage_counts = Counter(entry["stage"] for entry in ledger)
    reason_counts = Counter(entry["reason"] for entry in ledger)
    stage_reason_counts: dict[str, Counter[str]] = {
        stage: Counter() for stage in stage_order
    }
    family_stage_counts: dict[str, Counter[str]] = {}
    geo_level_stage_counts: dict[str, Counter[str]] = {}
    for entry in ledger:
        stage = str(entry["stage"])
        stage_reason_counts.setdefault(stage, Counter())[str(entry["reason"])] += 1
        family = str(entry["family"])
        family_stage_counts.setdefault(family, Counter())[stage] += 1
        geo_level = str(entry.get("geo_level") or "unspecified")
        geo_level_stage_counts.setdefault(geo_level, Counter())[stage] += 1
    return {
        "n_targets": len(ledger),
        "n_compile_ready_targets": int(compiled_target_count),
        "n_selected_after_feasibility": int(preselection_target_count),
        "n_selected_for_current_solve": int(final_solve_target_count),
        "stage_counts": {
            stage: int(stage_counts.get(stage, 0)) for stage in stage_order
        },
        "reason_counts": {
            reason: int(count) for reason, count in sorted(reason_counts.items())
        },
        "stage_reason_counts": {
            stage: {
                reason: int(count)
                for reason, count in sorted(
                    stage_reason_counts.get(stage, Counter()).items()
                )
            }
            for stage in stage_order
        },
        "geo_level_stage_counts": {
            geo_level: {stage: int(count) for stage, count in sorted(counter.items())}
            for geo_level, counter in sorted(geo_level_stage_counts.items())
        },
        "family_stage_counts": {
            family: {stage: int(count) for stage, count in sorted(counter.items())}
            for family, counter in sorted(family_stage_counts.items())
        },
    }


def _build_policyengine_calibration_target_ledger(
    *,
    canonical_targets: list[TargetSpec],
    tables: PolicyEngineUSEntityTableBundle,
    bindings: dict[str, PolicyEngineUSVariableBinding],
    compiled_targets: list[TargetSpec],
    structurally_unsupported_targets: list[TargetSpec],
    compiled_constraints: tuple[Any, ...],
    preselection_targets: list[TargetSpec],
    selected_stage_by_name: dict[str, int],
    household_count: int,
    min_active_households: int,
    materialization_failures: dict[str, str],
    compiled_constraint_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    min_required_households = max(1, int(min_active_households))
    structurally_unsupported_names = {
        target.name for target in structurally_unsupported_targets
    }
    preselection_names = {target.name for target in preselection_targets}
    final_solve_names = set(selected_stage_by_name)

    ledger: list[dict[str, Any]] = []
    classified_names: set[str] = set()
    for target in canonical_targets:
        missing_features = sorted(
            str(feature)
            for feature in target.required_features
            if feature not in bindings
        )
        has_entity_table = _policyengine_target_has_entity_table(target, tables)
        if not has_entity_table:
            ledger.append(
                _policyengine_target_ledger_entry(
                    target=target,
                    stage="audit_only",
                    reason="missing_entity_table",
                    household_count=household_count,
                    missing_features=missing_features,
                )
            )
            classified_names.add(target.name)
            continue
        if missing_features:
            failed_materializations = [
                feature
                for feature in missing_features
                if feature in materialization_failures
            ]
            ledger.append(
                _policyengine_target_ledger_entry(
                    target=target,
                    stage="audit_only",
                    reason=(
                        "materialization_failure"
                        if failed_materializations
                        else "missing_required_features"
                    ),
                    household_count=household_count,
                    missing_features=missing_features,
                    failed_materializations=failed_materializations,
                )
            )
            classified_names.add(target.name)
            continue
        if target.name in structurally_unsupported_names:
            ledger.append(
                _policyengine_target_ledger_entry(
                    target=target,
                    stage="audit_only",
                    reason="unsupported_structure",
                    household_count=household_count,
                )
            )
            classified_names.add(target.name)

    for record in _build_policyengine_constraint_records(
        compiled_targets,
        compiled_constraints,
        metadata_lookup=compiled_constraint_metadata,
    ):
        target = record["target"]
        classified_names.add(target.name)
        active_households = int(record["active_households"])
        if target.name in final_solve_names:
            stage = "solve_now"
            reason = f"selected_stage_{int(selected_stage_by_name[target.name])}"
        elif target.name in preselection_names:
            stage = "solve_later"
            reason = "household_budget_selection"
        elif active_households < min_required_households:
            stage = "solve_later"
            reason = "low_household_support"
        else:
            stage = "solve_later"
            reason = "constraint_capacity"
        ledger.append(
            _policyengine_target_ledger_entry(
                target=target,
                stage=stage,
                reason=reason,
                household_count=household_count,
                active_households=active_households,
                min_active_households=min_required_households,
            )
        )

    for target in canonical_targets:
        if target.name in classified_names:
            continue
        ledger.append(
            _policyengine_target_ledger_entry(
                target=target,
                stage="audit_only",
                reason="unclassified",
                household_count=household_count,
            )
        )

    stage_rank = {"solve_now": 0, "solve_later": 1, "audit_only": 2}
    ledger.sort(
        key=lambda entry: (
            stage_rank.get(str(entry["stage"]), 99),
            str(entry["reason"]),
            str(entry["family"]),
            str(entry["target_name"]),
        )
    )
    return (
        _summarize_policyengine_target_ledger(
            ledger,
            compiled_target_count=len(compiled_targets),
            preselection_target_count=len(preselection_targets),
            final_solve_target_count=len(final_solve_names),
        ),
        ledger,
    )


def _ranked_policyengine_group_focus_keys(
    ranking: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    limit: int | None,
) -> list[str]:
    if not ranking:
        return []
    if limit is not None and limit <= 0:
        return []
    selected: list[str] = []
    for row in ranking:
        score = float(row.get("capped_sum_abs_relative_error") or 0.0)
        if score <= 0.0:
            continue
        selected.append(str(row["group"]))
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _select_policyengine_deferred_stage_constraints(
    *,
    compiled_targets: list[TargetSpec],
    compiled_constraints: tuple[LinearConstraint, ...],
    target_ledger: list[dict[str, Any]],
    deferred_oracle_loss: dict[str, Any],
    deferred_target_priority_lookup: dict[str, float] | None,
    selected_target_names: set[str],
    household_count: int,
    min_active_households: int,
    max_constraints: int | None,
    max_constraints_per_household: float | None,
    top_family_count: int | None,
    top_geography_count: int | None,
    compiled_constraint_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[TargetSpec], tuple[LinearConstraint, ...], dict[str, Any]]:
    ledger_by_name = {
        str(entry["target_name"]): entry
        for entry in target_ledger
        if entry.get("target_name") is not None
    }
    family_focus = _ranked_policyengine_group_focus_keys(
        deferred_oracle_loss.get("family_ranking"),
        limit=top_family_count,
    )
    geography_focus = _ranked_policyengine_group_focus_keys(
        deferred_oracle_loss.get("geography_ranking"),
        limit=top_geography_count,
    )
    family_focus_set = set(family_focus)
    geography_focus_set = set(geography_focus)
    family_scores = {
        str(row["group"]): float(row.get("capped_loss_share") or 0.0)
        for row in deferred_oracle_loss.get("family_ranking", ())
    }
    geography_scores = {
        str(row["group"]): float(row.get("capped_loss_share") or 0.0)
        for row in deferred_oracle_loss.get("geography_ranking", ())
    }

    candidate_targets: list[TargetSpec] = []
    candidate_constraints: list[LinearConstraint] = []
    priority_scores: dict[str, float] = {}
    focus_eligible_count = 0
    min_required_households = max(1, int(min_active_households))

    for record in _build_policyengine_constraint_records(
        compiled_targets,
        compiled_constraints,
        metadata_lookup=compiled_constraint_metadata,
    ):
        target = record["target"]
        if target.name in selected_target_names:
            continue
        ledger_entry = ledger_by_name.get(target.name)
        if ledger_entry is None or ledger_entry.get("stage") != "solve_later":
            continue
        if int(record["active_households"]) < min_required_households:
            continue
        family_key = _policyengine_target_loss_family_key(ledger_entry)
        geography_key = _policyengine_target_loss_geography_key(ledger_entry)
        if family_focus_set or geography_focus_set:
            if (
                family_key not in family_focus_set
                and geography_key not in geography_focus_set
            ):
                continue
        focus_eligible_count += 1
        candidate_targets.append(target)
        candidate_constraints.append(record["constraint"])
        target_score = (
            float(deferred_target_priority_lookup.get(target.name, 0.0))
            if deferred_target_priority_lookup is not None
            else 0.0
        )
        priority_scores[target.name] = (
            target_score
            + family_scores.get(family_key, 0.0)
            + geography_scores.get(geography_key, 0.0)
        )

    selected_targets, selected_constraints, feasibility_summary = (
        _select_feasible_policyengine_calibration_constraints(
            candidate_targets,
            tuple(candidate_constraints),
            household_count=household_count,
            max_constraints=max_constraints,
            max_constraints_per_household=max_constraints_per_household,
            min_active_households=min_required_households,
            priority_scores=priority_scores,
        )
    )
    return (
        selected_targets,
        selected_constraints,
        {
            "min_active_households": min_required_households,
            "top_family_count": top_family_count,
            "top_geography_count": top_geography_count,
            "focused_families": family_focus,
            "focused_geographies": geography_focus,
            "n_focus_eligible_constraints": focus_eligible_count,
            "target_error_priority_available": deferred_target_priority_lookup
            is not None,
            "feasibility_filter": feasibility_summary,
        },
    )


def _policyengine_unsupported_target_error_penalty(
    *,
    relative_error_cap: float | None,
) -> float:
    if relative_error_cap is not None:
        return float(relative_error_cap)
    return 1.0


def _policyengine_target_fit_loss_components(
    report: Any,
    *,
    relative_error_cap: float | None = None,
) -> dict[str, Any]:
    supported_abs_relative_errors = [
        abs(evaluation.relative_error)
        for evaluation in report.evaluations
        if evaluation.relative_error is not None
    ]
    capped_supported_abs_relative_errors = [
        min(error, float(relative_error_cap))
        if relative_error_cap is not None
        else error
        for error in supported_abs_relative_errors
    ]
    unsupported_target_count = int(len(report.unsupported_targets))
    unsupported_target_error_penalty = _policyengine_unsupported_target_error_penalty(
        relative_error_cap=relative_error_cap
    )
    penalized_abs_relative_errors = [
        *supported_abs_relative_errors,
        *([unsupported_target_error_penalty] * unsupported_target_count),
    ]
    capped_penalized_abs_relative_errors = [
        *capped_supported_abs_relative_errors,
        *([unsupported_target_error_penalty] * unsupported_target_count),
    ]
    return {
        "supported_abs_relative_errors": supported_abs_relative_errors,
        "capped_supported_abs_relative_errors": capped_supported_abs_relative_errors,
        "penalized_abs_relative_errors": penalized_abs_relative_errors,
        "capped_penalized_abs_relative_errors": capped_penalized_abs_relative_errors,
        "unsupported_target_count": unsupported_target_count,
        "unsupported_target_error_penalty": unsupported_target_error_penalty,
    }


def _summarize_policyengine_target_fit_report(
    report: Any,
    *,
    target_count: int,
    relative_error_cap: float | None = None,
) -> dict[str, Any]:
    supported_target_count = int(report.supported_target_count)
    unsupported_target_count = int(len(report.unsupported_targets))
    supported_target_rate = None
    if target_count > 0:
        supported_target_rate = float(supported_target_count / target_count)
    loss_components = _policyengine_target_fit_loss_components(
        report,
        relative_error_cap=relative_error_cap,
    )
    supported_only_mean_abs_relative_error = report.mean_abs_relative_error
    supported_only_max_abs_relative_error = report.max_abs_relative_error
    supported_only_capped_mean_abs_relative_error = (
        float(
            sum(loss_components["capped_supported_abs_relative_errors"])
            / len(loss_components["capped_supported_abs_relative_errors"])
        )
        if loss_components["capped_supported_abs_relative_errors"]
        else None
    )
    penalized_abs_relative_errors = loss_components["penalized_abs_relative_errors"]
    capped_penalized_abs_relative_errors = loss_components[
        "capped_penalized_abs_relative_errors"
    ]
    mean_abs_relative_error = (
        float(sum(penalized_abs_relative_errors) / target_count)
        if target_count > 0 and penalized_abs_relative_errors
        else None
    )
    max_abs_relative_error = None
    if target_count > 0:
        max_candidates = []
        if supported_only_max_abs_relative_error is not None:
            max_candidates.append(float(supported_only_max_abs_relative_error))
        if unsupported_target_count > 0:
            max_candidates.append(loss_components["unsupported_target_error_penalty"])
        if max_candidates:
            max_abs_relative_error = max(max_candidates)
    capped_mean_abs_relative_error = (
        float(sum(capped_penalized_abs_relative_errors) / target_count)
        if target_count > 0 and capped_penalized_abs_relative_errors
        else None
    )
    return {
        "target_count": int(target_count),
        "supported_target_count": supported_target_count,
        "unsupported_target_count": unsupported_target_count,
        "supported_target_rate": supported_target_rate,
        "mean_abs_relative_error": (
            float(mean_abs_relative_error)
            if mean_abs_relative_error is not None
            else None
        ),
        "supported_only_mean_abs_relative_error": (
            float(supported_only_mean_abs_relative_error)
            if supported_only_mean_abs_relative_error is not None
            else None
        ),
        "max_abs_relative_error": (
            float(max_abs_relative_error)
            if max_abs_relative_error is not None
            else None
        ),
        "supported_only_max_abs_relative_error": (
            float(supported_only_max_abs_relative_error)
            if supported_only_max_abs_relative_error is not None
            else None
        ),
        "relative_error_cap": (
            float(relative_error_cap) if relative_error_cap is not None else None
        ),
        "unsupported_target_error_penalty": (
            loss_components["unsupported_target_error_penalty"]
            if unsupported_target_count > 0
            else None
        ),
        "capped_mean_abs_relative_error": capped_mean_abs_relative_error,
        "supported_only_capped_mean_abs_relative_error": (
            supported_only_capped_mean_abs_relative_error
        ),
    }


def _summarize_policyengine_target_fit_group_reports(
    report: Any,
    *,
    targets_by_group: dict[str, list[TargetSpec]],
    relative_error_cap: float | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    total_loss_components = _policyengine_target_fit_loss_components(
        report,
        relative_error_cap=relative_error_cap,
    )
    total_abs_relative_error = float(
        sum(total_loss_components["penalized_abs_relative_errors"])
    )
    total_capped_abs_relative_error = float(
        sum(total_loss_components["capped_penalized_abs_relative_errors"])
    )
    grouped: list[tuple[str, dict[str, Any]]] = []
    for group_key, group_targets in targets_by_group.items():
        group_report = slice_policyengine_us_target_evaluation_report(
            report,
            group_targets,
        )
        group_loss_components = _policyengine_target_fit_loss_components(
            group_report,
            relative_error_cap=relative_error_cap,
        )
        sum_abs_relative_error = float(
            sum(group_loss_components["penalized_abs_relative_errors"])
        )
        capped_sum_abs_relative_error = float(
            sum(group_loss_components["capped_penalized_abs_relative_errors"])
        )
        summary = _summarize_policyengine_target_fit_report(
            group_report,
            target_count=len(group_targets),
            relative_error_cap=relative_error_cap,
        )
        summary["sum_abs_relative_error"] = sum_abs_relative_error
        summary["loss_share"] = (
            float(sum_abs_relative_error / total_abs_relative_error)
            if total_abs_relative_error > 0.0
            else None
        )
        summary["capped_sum_abs_relative_error"] = capped_sum_abs_relative_error
        summary["capped_loss_share"] = (
            float(capped_sum_abs_relative_error / total_capped_abs_relative_error)
            if total_capped_abs_relative_error > 0.0
            else None
        )
        grouped.append((group_key, summary))

    grouped.sort(
        key=lambda item: (
            -item[1]["capped_sum_abs_relative_error"],
            -item[1]["sum_abs_relative_error"],
            -item[1]["target_count"],
            item[0],
        )
    )
    return (
        {group_key: summary for group_key, summary in grouped},
        [
            {
                "group": group_key,
                **summary,
            }
            for group_key, summary in grouped
        ],
    )


def _summarize_policyengine_target_fit_report_with_groups(
    report: Any,
    *,
    targets: list[TargetSpec],
    ledger_by_name: dict[str, dict[str, Any]],
    relative_error_cap: float | None = None,
) -> dict[str, Any]:
    summary = _summarize_policyengine_target_fit_report(
        report,
        target_count=len(targets),
        relative_error_cap=relative_error_cap,
    )
    family_targets: dict[str, list[TargetSpec]] = {}
    geography_targets: dict[str, list[TargetSpec]] = {}
    for target in targets:
        ledger_entry = ledger_by_name.get(target.name)
        if ledger_entry is None:
            continue
        family_targets.setdefault(
            _policyengine_target_loss_family_key(ledger_entry),
            [],
        ).append(target)
        geography_targets.setdefault(
            _policyengine_target_loss_geography_key(ledger_entry),
            [],
        ).append(target)
    (
        summary["family_summaries"],
        summary["family_ranking"],
    ) = _summarize_policyengine_target_fit_group_reports(
        report,
        targets_by_group=family_targets,
        relative_error_cap=relative_error_cap,
    )
    (
        summary["geography_summaries"],
        summary["geography_ranking"],
    ) = _summarize_policyengine_target_fit_group_reports(
        report,
        targets_by_group=geography_targets,
        relative_error_cap=relative_error_cap,
    )
    return summary


def _evaluate_policyengine_target_fit_summaries(
    *,
    tables: PolicyEngineUSEntityTableBundle,
    canonical_targets: list[TargetSpec],
    final_solve_targets: list[TargetSpec],
    target_ledger: list[dict[str, Any]],
    period: int | str,
    dataset_year: int | None,
    simulation_cls: Any | None,
    direct_override_variables: tuple[str, ...] = (),
    relative_error_cap: float | None = None,
) -> dict[str, dict[str, Any]]:
    summaries, _ = _evaluate_policyengine_target_fit_context(
        tables=tables,
        canonical_targets=canonical_targets,
        final_solve_targets=final_solve_targets,
        target_ledger=target_ledger,
        period=period,
        dataset_year=dataset_year,
        simulation_cls=simulation_cls,
        direct_override_variables=direct_override_variables,
        relative_error_cap=relative_error_cap,
    )
    return summaries


def _policyengine_target_fit_priority_lookup(
    report: Any,
    *,
    relative_error_cap: float | None = None,
) -> dict[str, float]:
    target_scores: dict[str, float] = {}
    for evaluation in report.evaluations:
        abs_relative_error = abs(float(evaluation.relative_error))
        capped_abs_relative_error = (
            min(abs_relative_error, float(relative_error_cap))
            if relative_error_cap is not None
            else abs_relative_error
        )
        target_scores[evaluation.target.name] = float(capped_abs_relative_error)
    unsupported_target_error_penalty = _policyengine_unsupported_target_error_penalty(
        relative_error_cap=relative_error_cap
    )
    for target in report.unsupported_targets:
        target_scores[target.name] = float(unsupported_target_error_penalty)
    return target_scores


def _evaluate_policyengine_target_fit_context(
    *,
    tables: PolicyEngineUSEntityTableBundle,
    canonical_targets: list[TargetSpec],
    final_solve_targets: list[TargetSpec],
    target_ledger: list[dict[str, Any]],
    period: int | str,
    dataset_year: int | None,
    simulation_cls: Any | None,
    direct_override_variables: tuple[str, ...] = (),
    relative_error_cap: float | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    target_by_name = {target.name: target for target in canonical_targets}
    ledger_by_name = {
        str(entry["target_name"]): entry
        for entry in target_ledger
        if entry.get("target_name")
    }
    deferred_targets = [
        target_by_name[entry["target_name"]]
        for entry in target_ledger
        if entry["stage"] == "solve_later" and entry["target_name"] in target_by_name
    ]
    audit_only_targets = [
        target_by_name[entry["target_name"]]
        for entry in target_ledger
        if entry["stage"] == "audit_only" and entry["target_name"] in target_by_name
    ]
    full_report = evaluate_policyengine_us_target_set(
        tables,
        canonical_targets,
        period=period,
        dataset_year=dataset_year,
        simulation_cls=simulation_cls,
        label="policyengine_db_calibration",
        direct_override_variables=direct_override_variables,
    )
    active_solve_report = slice_policyengine_us_target_evaluation_report(
        full_report,
        final_solve_targets,
    )
    deferred_report = slice_policyengine_us_target_evaluation_report(
        full_report,
        deferred_targets,
    )
    audit_only_report = slice_policyengine_us_target_evaluation_report(
        full_report,
        audit_only_targets,
    )
    summaries = {
        "full_oracle": _summarize_policyengine_target_fit_report_with_groups(
            full_report,
            targets=canonical_targets,
            ledger_by_name=ledger_by_name,
            relative_error_cap=relative_error_cap,
        ),
        "active_solve": _summarize_policyengine_target_fit_report_with_groups(
            active_solve_report,
            targets=final_solve_targets,
            ledger_by_name=ledger_by_name,
            relative_error_cap=relative_error_cap,
        ),
        "deferred": _summarize_policyengine_target_fit_report_with_groups(
            deferred_report,
            targets=deferred_targets,
            ledger_by_name=ledger_by_name,
            relative_error_cap=relative_error_cap,
        ),
        "audit_only": _summarize_policyengine_target_fit_report_with_groups(
            audit_only_report,
            targets=audit_only_targets,
            ledger_by_name=ledger_by_name,
            relative_error_cap=relative_error_cap,
        ),
    }
    return summaries, {
        "full_oracle": _policyengine_target_fit_priority_lookup(
            full_report,
            relative_error_cap=relative_error_cap,
        ),
        "active_solve": _policyengine_target_fit_priority_lookup(
            active_solve_report,
            relative_error_cap=relative_error_cap,
        ),
        "deferred": _policyengine_target_fit_priority_lookup(
            deferred_report,
            relative_error_cap=relative_error_cap,
        ),
        "audit_only": _policyengine_target_fit_priority_lookup(
            audit_only_report,
            relative_error_cap=relative_error_cap,
        ),
    }


def _select_feasible_policyengine_calibration_constraints(
    targets: list[TargetSpec],
    constraints: tuple[Any, ...],
    *,
    household_count: int,
    max_constraints: int | None,
    max_constraints_per_household: float | None,
    min_active_households: int,
    priority_scores: dict[str, float] | None = None,
) -> tuple[list[TargetSpec], tuple[Any, ...], dict[str, Any]]:
    selected_targets = list(targets)
    selected_constraints = tuple(constraints)
    requested_max_constraints = max_constraints
    if (
        requested_max_constraints is None
        and max_constraints_per_household is not None
        and household_count > 0
    ):
        requested_max_constraints = max(
            1,
            int(np.floor(max_constraints_per_household * household_count)),
        )

    records = _build_policyengine_constraint_records(targets, constraints)

    min_required_households = max(1, int(min_active_households))
    support_filtered = [
        record
        for record in records
        if record["active_households"] >= min_required_households
    ]
    low_support_dropped = len(records) - len(support_filtered)

    support_filtered.sort(
        key=lambda record: (
            -float(priority_scores.get(record["target"].name, 0.0))
            if priority_scores is not None
            else 0.0,
            record["geo_priority"],
            record["aggregation_priority"],
            -record["active_households"],
            -record["coefficient_mass"],
            record["target"].name,
        )
    )

    over_capacity_dropped = 0
    if (
        requested_max_constraints is not None
        and len(support_filtered) > requested_max_constraints
    ):
        over_capacity_dropped = len(support_filtered) - requested_max_constraints
        support_filtered = support_filtered[:requested_max_constraints]

    selected_targets = [record["target"] for record in support_filtered]
    selected_constraints = tuple(record["constraint"] for record in support_filtered)
    dropped_total = low_support_dropped + over_capacity_dropped
    drop_share = float(dropped_total / len(records)) if records else 0.0
    warning_messages: list[str] = []
    if drop_share > CALIBRATION_FEASIBILITY_DROP_WARNING_THRESHOLD:
        warning_messages.append(
            "Calibration feasibility filter dropped "
            f"{dropped_total}/{len(records)} constraints "
            f"({drop_share:.1%}) before solving."
        )
    diagnostics = {
        "requested_max_constraints": requested_max_constraints,
        "max_constraints_per_household": max_constraints_per_household,
        "min_active_households": min_required_households,
        "n_constraints_before_feasibility_filter": len(constraints),
        "n_constraints_after_feasibility_filter": len(selected_constraints),
        "n_constraints_dropped_low_support": low_support_dropped,
        "n_constraints_dropped_over_capacity": over_capacity_dropped,
        "n_constraints_dropped_total": dropped_total,
        "constraint_drop_share": drop_share,
        "warning_messages": warning_messages,
        "feasibility_filter_applied": bool(
            low_support_dropped > 0 or over_capacity_dropped > 0
        ),
    }
    return selected_targets, selected_constraints, diagnostics


@dataclass(frozen=True)
class USMicroplexBuildConfig:
    """Configuration for the US microplex build pipeline."""

    n_synthetic: int = 100_000
    synthesis_backend: Literal["bootstrap", "synthesizer", "seed"] = "synthesizer"
    calibration_backend: Literal[
        "entropy",
        "ipf",
        "chi2",
        "sparse",
        "hardconcrete",
        "pe_l0",
        "microcalibrate",
        "none",
    ] = "entropy"
    calibration_tol: float = 1e-6
    calibration_max_iter: int = 100
    random_seed: int = 42
    target_sparsity: float = 0.9
    device: str = "cpu"
    synthesizer_condition_vars: tuple[str, ...] = (
        "age",
        "sex",
        "education",
        "employment_status",
        "state_fips",
        "tenure",
    )
    synthesizer_target_vars: tuple[str, ...] = ("income",)
    synthesizer_epochs: int = 100
    synthesizer_batch_size: int = 256
    synthesizer_learning_rate: float = 1e-3
    synthesizer_n_layers: int = 4
    synthesizer_hidden_dim: int = 64
    donor_imputer_epochs: int = 20
    donor_imputer_batch_size: int = 128
    donor_imputer_learning_rate: float = 1e-3
    donor_imputer_n_layers: int = 2
    donor_imputer_hidden_dim: int = 32
    donor_imputer_backend: Literal["maf", "qrf", "zi_qrf", "regime_aware"] = "maf"
    donor_imputer_qrf_n_estimators: int = 100
    donor_imputer_qrf_zero_threshold: float = 0.05
    donor_imputer_condition_selection: Literal[
        "all_shared",
        "top_correlated",
        "pe_prespecified",
        "pe_plus_puf_native_challenger",
    ] = "top_correlated"
    donor_imputer_max_condition_vars: int | None = 8
    donor_imputer_excluded_variables: tuple[str, ...] = ("filing_status_code",)
    donor_imputer_authoritative_override_variables: tuple[str, ...] = ()
    puf_support_clone_enabled: bool = False
    puf_support_clone_source_prefixes: tuple[str, ...] = ("irs_soi_puf",)
    puf_support_clone_zero_initial_weight: bool = True
    puf_support_clone_flag_column: str = PUF_SUPPORT_CLONE_FLAG_COLUMN
    puf_support_clone_prior_weight_share: float = 0.05
    puf_support_clone_overlap_variables: tuple[str, ...] = (
        PUF_SUPPORT_CLONE_IMPUTED_VARIABLES
        + PUF_SUPPORT_CLONE_SPECIAL_VARIABLES
        + ("wage_income", "dividend_income", "capital_gains")
    )
    puf_support_clone_both_halves_override_variables: tuple[str, ...] = (
        PUF_SUPPORT_CLONE_OVERRIDDEN_VARIABLES
    )
    puf_support_clone_refresh_cps_only_fields: bool = True
    puf_support_clone_cps_refresh_variables: tuple[str, ...] = (
        PUF_SUPPORT_CLONE_CPS_REFRESH_VARIABLES
    )
    puf_support_clone_cps_refresh_condition_variables: tuple[str, ...] = (
        PUF_SUPPORT_CLONE_CPS_REFRESH_CONDITION_VARIABLES
    )
    dependent_tax_leaf_soft_cap_multiplier: float | None = None
    dependent_tax_leaf_soft_cap_base_variables: tuple[str, ...] = (
        "employment_income",
        "wage_income",
        "self_employment_income",
    )
    dependent_tax_leaf_soft_cap_variables: tuple[str, ...] = (
        "taxable_interest_income",
        "tax_exempt_interest_income",
        "taxable_pension_income",
        "dividend_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "partnership_s_corp_income",
        "rental_income",
    )
    bootstrap_strata_columns: tuple[str, ...] = ()
    prefer_cached_cps_asec_source: bool = False
    cps_asec_source_year: int = 2023
    cps_asec_cache_dir: str | None = None
    policyengine_dataset: str | None = None
    policyengine_baseline_dataset: str | None = None
    policyengine_dataset_year: int | None = None
    policyengine_direct_override_variables: tuple[str, ...] = ()
    policyengine_export_column_contract_path: str | Path | None = None
    """Optional eCPS export-column contract checked before calibration.

    When set, the pipeline verifies the final H5 column surface from the
    post-imputation PE entity tables, then fails before microsimulation or
    calibration if required columns are missing or forbidden columns would be
    exported.
    """
    policyengine_prefer_existing_tax_unit_ids: bool = True
    policyengine_quantity_targets: tuple[PolicyEngineUSQuantityTarget, ...] = ()
    policyengine_targets_db: str | None = None
    arch_targets_db: str | tuple[str, ...] | None = None
    calibration_target_source: Literal["policyengine", "arch"] = "policyengine"
    policyengine_target_period: int | None = None
    policyengine_target_variables: tuple[str, ...] = ()
    policyengine_target_domains: tuple[str, ...] = ()
    policyengine_target_geo_levels: tuple[str, ...] = ()
    policyengine_target_profile: str | None = None
    policyengine_calibration_target_variables: tuple[str, ...] = ()
    policyengine_calibration_target_domains: tuple[str, ...] = ()
    policyengine_calibration_target_geo_levels: tuple[str, ...] = ()
    policyengine_calibration_target_profile: str | None = None
    policyengine_calibrate_ssi_takeup: bool = True
    policyengine_calibration_rescale_to_input_weight_sum: bool = False
    policyengine_calibration_rescale_to_target_total_weight: bool = False
    policyengine_calibration_target_total_weight: float | None = None
    policyengine_selection_backend: Literal["sparse", "pe_native_loss"] = "sparse"
    policyengine_selection_household_budget: int | None = None
    policyengine_selection_state_floor: int = 0
    policyengine_selection_max_iter: int = 200
    policyengine_selection_tol: float = 1e-8
    policyengine_selection_l2_penalty: float = 0.0
    policyengine_selection_target_total_weight: float | None = None
    policyengine_calibration_max_constraints: int | None = None
    policyengine_calibration_max_constraints_per_household: float | None = (
        DEFAULT_POLICYENGINE_CALIBRATION_MAX_CONSTRAINTS_PER_HOUSEHOLD
    )
    policyengine_calibration_min_active_households: int = (
        DEFAULT_POLICYENGINE_CALIBRATION_MIN_ACTIVE_HOUSEHOLDS
    )
    policyengine_calibration_deferred_stage_min_active_households: tuple[int, ...] = ()
    policyengine_calibration_deferred_stage_max_constraints: int | None = 24
    policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error: (
        float | None
    ) = None
    policyengine_calibration_deferred_stage_top_family_count: int | None = 8
    policyengine_calibration_deferred_stage_top_geography_count: int | None = 8
    policyengine_oracle_relative_error_cap: float | None = 10.0
    policyengine_target_reform_id: int = 0
    policyengine_simulation_cls: Any | None = None
    policyengine_materialize_batch_size: int | None = None
    """Batch size for PolicyEngine variable materialization.

    At 1.5M-household scale a single Microsimulation is 25–35 GB. With
    a batch size of e.g. 100_000, the pipeline splits the entity tables
    into chunks and runs one Microsimulation per chunk, reducing peak
    memory to a few GB. ``None`` (default) keeps the legacy single-pass
    behavior. Safe for per-household scalar variables (all our
    calibration targets); unsafe for population-quantile-dependent
    variables (see docstring on
    :func:`materialize_policyengine_us_variables`).
    """
    pipeline_checkpoint_save_post_imputation_path: str | Path | None = None
    """Write a post-imputation pipeline checkpoint to this directory.

    Saved right after donor imputation + ``build_policyengine_entity_tables``
    and before microsim materializes calibration target variables. The
    ~11 h synthesis + imputation + PE-tables build can be skipped on a
    rerun that loads from this checkpoint, leaving only microsim (~30
    min) + calibration fit (~30 min) to redo.
    """
    pipeline_checkpoint_save_post_microsim_path: str | Path | None = None
    """Write a post-microsim pipeline checkpoint to this directory.

    Saved after ``_resolve_policyengine_calibration_targets`` has
    materialized every calibration target variable onto the bundle, and
    before the L0/microcalibrate fit loop. A rerun that loads from this
    checkpoint skips microsim too, leaving only the ~30 min calibration
    fit — useful for tuning calibration targets or backends.
    """
    capital_gains_lots_enabled: bool = False
    """Write an anchor-preserving synthetic capital-gains lot sidecar artifact."""
    capital_gains_lots_max_lots_per_person: int = 4
    capital_gains_lots_random_seed: int | None = None
    forbes_fixed_spine_records_path: str | Path | None = None
    """Normalized Forbes fixed-spine records to append after calibration."""
    forbes_fixed_spine_snapshot_id: str = "forbes-us-top-tail"
    forbes_fixed_spine_replicates_per_unit: int = 10

    def __post_init__(self) -> None:
        if self.puf_support_clone_enabled:
            if self.synthesis_backend != "seed":
                raise ValueError(
                    "puf_support_clone_enabled requires synthesis_backend='seed' "
                    "until post-synthesis clone construction is implemented"
                )
            if self.policyengine_selection_household_budget is not None:
                raise ValueError(
                    "puf_support_clone_enabled cannot be combined with "
                    "policyengine_selection_household_budget until selector "
                    "clone activation is implemented"
                )
            if not self.puf_support_clone_source_prefixes:
                raise ValueError(
                    "puf_support_clone_source_prefixes must not be empty when "
                    "puf_support_clone_enabled is true"
                )
            if not (0.0 <= self.puf_support_clone_prior_weight_share < 1.0):
                raise ValueError(
                    "puf_support_clone_prior_weight_share must be in [0, 1)"
                )
        if (
            self.policyengine_calibration_rescale_to_input_weight_sum
            and self.policyengine_calibration_rescale_to_target_total_weight
        ):
            raise ValueError(
                "policyengine_calibration_rescale_to_input_weight_sum and "
                "policyengine_calibration_rescale_to_target_total_weight are mutually exclusive"
            )
        if (
            self.policyengine_calibration_rescale_to_target_total_weight
            and self.policyengine_calibration_target_total_weight is None
        ):
            raise ValueError(
                "policyengine_calibration_rescale_to_target_total_weight requires "
                "policyengine_calibration_target_total_weight"
            )
        if (
            self.policyengine_oracle_relative_error_cap is not None
            and float(self.policyengine_oracle_relative_error_cap) <= 0.0
        ):
            raise ValueError(
                "policyengine_oracle_relative_error_cap must be positive when provided"
            )
        if (
            self.dependent_tax_leaf_soft_cap_multiplier is not None
            and float(self.dependent_tax_leaf_soft_cap_multiplier) < 0.0
        ):
            raise ValueError(
                "dependent_tax_leaf_soft_cap_multiplier must be non-negative when provided"
            )
        if self.forbes_fixed_spine_replicates_per_unit < 1:
            raise ValueError(
                "forbes_fixed_spine_replicates_per_unit must be at least 1"
            )
        if any(
            int(value) <= 0
            for value in self.policyengine_calibration_deferred_stage_min_active_households
        ):
            raise ValueError(
                "policyengine_calibration_deferred_stage_min_active_households must contain only positive values"
            )
        if int(self.capital_gains_lots_max_lots_per_person) <= 0:
            raise ValueError("capital_gains_lots_max_lots_per_person must be positive")
        if (
            self.policyengine_calibration_deferred_stage_max_constraints is not None
            and int(self.policyengine_calibration_deferred_stage_max_constraints) <= 0
        ):
            raise ValueError(
                "policyengine_calibration_deferred_stage_max_constraints must be positive when provided"
            )
        if (
            self.policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error
            is not None
            and float(
                self.policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error
            )
            <= 0.0
        ):
            raise ValueError(
                "policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error must be positive when provided"
            )
        if (
            self.policyengine_calibration_deferred_stage_top_family_count is not None
            and int(self.policyengine_calibration_deferred_stage_top_family_count) < 0
        ):
            raise ValueError(
                "policyengine_calibration_deferred_stage_top_family_count must be nonnegative when provided"
            )
        if (
            self.policyengine_calibration_deferred_stage_top_geography_count is not None
            and int(self.policyengine_calibration_deferred_stage_top_geography_count)
            < 0
        ):
            raise ValueError(
                "policyengine_calibration_deferred_stage_top_geography_count must be nonnegative when provided"
            )

    def to_dict(self) -> dict[str, Any]:
        return _normalize_config_value(asdict(self))


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, type) or isinstance(value, FunctionType):
        return f"{value.__module__}.{value.__qualname__}"
    return value


@dataclass(frozen=True)
class USMicroplexTargets:
    """Calibration targets for the US microplex pipeline."""

    marginal: dict[str, dict[str, float]]
    continuous: dict[str, float]


@dataclass(frozen=True)
class USMicroplexSourceInput:
    """Normalized source-planning context for one US build."""

    frame: ObservationFrame
    fusion_plan: FusionPlan
    household_observation: EntityObservation
    person_observation: EntityObservation
    household_person_relationship: EntityRelationship
    households: pd.DataFrame
    persons: pd.DataFrame


@dataclass(frozen=True)
class USMicroplexSynthesisVariables:
    """Observed variables to use during synthesis."""

    condition_vars: tuple[str, ...]
    target_vars: tuple[str, ...]


@dataclass
class USMicroplexBuildResult:
    """Artifacts from a US microplex build."""

    config: USMicroplexBuildConfig
    seed_data: pd.DataFrame
    synthetic_data: pd.DataFrame
    calibrated_data: pd.DataFrame
    targets: USMicroplexTargets
    calibration_summary: dict[str, Any]
    synthesis_metadata: dict[str, Any] = field(default_factory=dict)
    synthesizer: Synthesizer | Any | None = None
    policyengine_tables: PolicyEngineUSEntityTableBundle | None = None
    source_frame: ObservationFrame | None = None
    source_frames: tuple[ObservationFrame, ...] = ()
    fusion_plan: FusionPlan | None = None
    scaffold_seed_data: pd.DataFrame | None = None

    @property
    def n_nonzero_weights(self) -> int:
        if "weight" not in self.calibrated_data.columns:
            return 0
        return int((self.calibrated_data["weight"] > 1e-9).sum())

    @property
    def total_weighted_population(self) -> float:
        if "weight" not in self.calibrated_data.columns:
            return 0.0
        return float(self.calibrated_data["weight"].sum())


class USMicroplexPipeline:
    """End-to-end build orchestration for a US microplex dataset."""

    def __init__(self, config: USMicroplexBuildConfig | None = None):
        self.config = config or USMicroplexBuildConfig()

    def build_from_data_dir(self, data_dir: str | Path) -> USMicroplexBuildResult:
        from microplex_us.data_sources.cps import (
            DEFAULT_CACHE_DIR,
            CPSASECParquetSourceProvider,
            CPSASECSourceProvider,
        )

        if self.config.prefer_cached_cps_asec_source:
            cache_dir = (
                Path(self.config.cps_asec_cache_dir)
                if self.config.cps_asec_cache_dir is not None
                else DEFAULT_CACHE_DIR
            )
            processed_path = (
                cache_dir
                / f"cps_asec_{int(self.config.cps_asec_source_year)}_processed.parquet"
            )
            if processed_path.exists():
                return self.build_from_source_provider(
                    CPSASECSourceProvider(
                        year=int(self.config.cps_asec_source_year),
                        cache_dir=cache_dir,
                        download=False,
                    )
                )

        return self.build_from_source_provider(
            CPSASECParquetSourceProvider(data_dir=data_dir)
        )

    def build_from_source_provider(
        self,
        provider: SourceProvider,
        query: SourceQuery | None = None,
    ) -> USMicroplexBuildResult:
        frame = provider.load_frame(query)
        return self.build_from_frames([frame])

    def build_from_source_providers(
        self,
        providers: list[SourceProvider],
        queries: dict[str, SourceQuery] | None = None,
    ) -> USMicroplexBuildResult:
        if not providers:
            raise ValueError(
                "USMicroplexPipeline requires at least one source provider"
            )

        frames: list[ObservationFrame] = []
        for provider in providers:
            frame = provider.load_frame(
                self._resolve_source_query(provider, queries or {})
            )
            frames.append(frame)
        return self.build_from_frames(frames)

    def build_from_frame(self, frame: ObservationFrame) -> USMicroplexBuildResult:
        return self.build_from_frames([frame])

    def build_from_frames(
        self,
        frames: list[ObservationFrame],
    ) -> USMicroplexBuildResult:
        if not frames:
            raise ValueError(
                "USMicroplexPipeline requires at least one observation frame"
            )

        source_inputs = [self.prepare_source_input(frame) for frame in frames]
        fusion_plan = FusionPlan.from_sources([frame.source for frame in frames])
        scaffold_input = self._select_scaffold_source(source_inputs)
        seed_data = self.prepare_seed_data_from_source(scaffold_input)
        seed_data = self._strip_generated_entity_ids(
            seed_data,
            scaffold_input=scaffold_input,
        )
        scaffold_seed_data = seed_data.copy()
        donor_integration = self._integrate_donor_sources(
            seed_data,
            scaffold_input=scaffold_input,
            donor_inputs=[
                source for source in source_inputs if source is not scaffold_input
            ],
        )
        seed_data = donor_integration["seed_data"]
        seed_data = self._apply_dependent_tax_leaf_soft_caps(seed_data)
        _emit_us_pipeline_progress(
            "US microplex build: seed ready",
            scaffold_source=scaffold_input.frame.source.name,
            sources=_format_progress_values(fusion_plan.source_names),
            rows=int(len(seed_data)),
            columns=int(len(seed_data.columns)),
            donor_integrated_variables=int(
                len(donor_integration["integrated_variables"])
            ),
        )
        _emit_us_pipeline_progress(
            "US microplex build: targets start",
            rows=int(len(seed_data)),
        )
        targets = self.build_targets(seed_data)
        _emit_us_pipeline_progress(
            "US microplex build: targets complete",
            marginal_targets=int(len(targets.marginal)),
            continuous_targets=int(len(targets.continuous)),
        )
        synthesis_variables = self._resolve_synthesis_variables(
            scaffold_input,
            fusion_plan=fusion_plan,
            include_all_observed_targets=len(source_inputs) > 1,
            available_columns=set(seed_data.columns),
            observed_frame=seed_data,
        )
        _emit_us_pipeline_progress(
            "US microplex build: synthesis variables ready",
            condition_vars=int(len(synthesis_variables.condition_vars)),
            target_vars=int(len(synthesis_variables.target_vars)),
        )
        _emit_us_pipeline_progress(
            "US microplex build: synthesis start",
            rows=int(len(seed_data)),
        )
        synthetic_data, synthesizer, synthesis_metadata = self.synthesize(
            seed_data,
            synthesis_variables=synthesis_variables,
        )
        _emit_us_pipeline_progress(
            "US microplex build: synthesis complete",
            rows=int(len(synthetic_data)),
            columns=int(len(synthetic_data.columns)),
        )
        synthesis_metadata = {
            **synthesis_metadata,
            "source_names": fusion_plan.source_names,
            "condition_vars": list(synthesis_variables.condition_vars),
            "target_vars": list(synthesis_variables.target_vars),
            "scaffold_source": scaffold_input.frame.source.name,
            "donor_integrated_variables": donor_integration["integrated_variables"],
            "donor_conditioning_diagnostics": donor_integration.get(
                "conditioning_diagnostics", []
            ),
            "processed_donor_source_order": donor_integration.get(
                "processed_donor_source_order", []
            ),
            "puf_clone_source_order": donor_integration.get(
                "puf_clone_source_order", []
            ),
            "puf_support_clone": donor_integration.get("puf_support_clone_summary"),
            "donor_excluded_variables": list(
                self.config.donor_imputer_excluded_variables
            ),
            "donor_authoritative_override_variables": list(
                self.config.donor_imputer_authoritative_override_variables
            ),
            "state_program_support_proxies": _state_program_support_proxy_summary(
                set(seed_data.columns)
            ),
        }
        _emit_us_pipeline_progress(
            "US microplex build: support enforcement start",
            rows=int(len(synthetic_data)),
        )
        synthetic_data = self.ensure_target_support(synthetic_data, seed_data, targets)
        _emit_us_pipeline_progress(
            "US microplex build: support enforcement complete",
            rows=int(len(synthetic_data)),
            columns=int(len(synthetic_data.columns)),
        )
        if self._has_policyengine_calibration_targets():
            _emit_us_pipeline_progress(
                "US microplex build: policyengine tables start",
                rows=int(len(synthetic_data)),
            )
            synthetic_tables = self.build_policyengine_entity_tables(synthetic_data)
            _emit_us_pipeline_progress(
                "US microplex build: policyengine tables complete",
                households=int(len(synthetic_tables.households)),
                persons=int(len(synthetic_tables.persons)),
            )
            if self.config.pipeline_checkpoint_save_post_imputation_path is not None:
                save_us_pipeline_checkpoint(
                    synthetic_tables,
                    self.config.pipeline_checkpoint_save_post_imputation_path,
                    stage="post_imputation",
                )
                _emit_us_pipeline_progress(
                    "US microplex build: post-imputation checkpoint saved",
                    path=str(self.config.pipeline_checkpoint_save_post_imputation_path),
                )
            self._check_policyengine_export_column_contract(
                synthetic_tables,
                stage="pre_calibration",
            )
            _emit_us_pipeline_progress(
                "US microplex build: policyengine calibration start",
                backend=self.config.calibration_backend,
            )
            (
                policyengine_tables,
                calibrated_data,
                calibration_summary,
            ) = self.calibrate_policyengine_tables(synthetic_tables)
            _emit_us_pipeline_progress(
                "US microplex build: policyengine calibration complete",
                backend=self.config.calibration_backend,
                calibrated_rows=int(len(calibrated_data)),
            )
        else:
            _emit_us_pipeline_progress(
                "US microplex build: calibration start",
                backend=self.config.calibration_backend,
                rows=int(len(synthetic_data)),
            )
            calibrated_data, calibration_summary = self.calibrate(
                synthetic_data, targets
            )
            _emit_us_pipeline_progress(
                "US microplex build: calibration complete",
                backend=self.config.calibration_backend,
                calibrated_rows=int(len(calibrated_data)),
            )
            _emit_us_pipeline_progress(
                "US microplex build: policyengine tables start",
                rows=int(len(calibrated_data)),
            )
            policyengine_tables = self.build_policyengine_entity_tables(calibrated_data)
            _emit_us_pipeline_progress(
                "US microplex build: policyengine tables complete",
                households=int(len(policyengine_tables.households)),
                persons=int(len(policyengine_tables.persons)),
            )

        return USMicroplexBuildResult(
            config=self.config,
            seed_data=seed_data,
            synthetic_data=synthetic_data,
            calibrated_data=calibrated_data,
            targets=targets,
            calibration_summary=calibration_summary,
            synthesis_metadata=synthesis_metadata,
            synthesizer=synthesizer,
            policyengine_tables=policyengine_tables,
            source_frame=scaffold_input.frame,
            source_frames=tuple(frame for frame in frames),
            fusion_plan=fusion_plan,
            scaffold_seed_data=scaffold_seed_data,
        )

    def build(
        self,
        persons: pd.DataFrame,
        households: pd.DataFrame,
    ) -> USMicroplexBuildResult:
        return self.build_from_frame(
            self._build_direct_input_frame(
                persons=persons,
                households=households,
            )
        )

    def _resolve_source_query(
        self,
        provider: SourceProvider,
        queries: dict[str, SourceQuery],
    ) -> SourceQuery | None:
        for key in self._source_query_keys(provider):
            query = queries.get(key)
            if query is not None:
                return query
        return None

    def _source_query_keys(self, provider: SourceProvider) -> tuple[str, ...]:
        base_name = provider.descriptor.name
        keys: list[str] = [base_name]
        for attr_name in ("year", "target_year"):
            attr_value = getattr(provider, attr_name, None)
            if attr_value is None:
                continue
            keys.append(f"{base_name}_{attr_value}")
        descriptor_cache = getattr(provider, "_descriptor_cache", None)
        cached_name = getattr(descriptor_cache, "name", None)
        if cached_name is not None:
            keys.append(cached_name)
        return tuple(dict.fromkeys(keys))

    def prepare_source_input(
        self,
        frame: ObservationFrame,
    ) -> USMicroplexSourceInput:
        """Validate and extract the source-planning context for a US build."""
        frame.validate()
        households = frame.tables.get(EntityType.HOUSEHOLD)
        persons = frame.tables.get(EntityType.PERSON)
        if households is None or persons is None:
            raise ValueError(
                "USMicroplexPipeline requires household and person tables from the source provider"
            )

        fusion_plan = FusionPlan.from_sources([frame.source])
        observations_by_entity = {
            observation.entity: observation for observation in frame.source.observations
        }
        household_observation = observations_by_entity.get(EntityType.HOUSEHOLD)
        person_observation = observations_by_entity.get(EntityType.PERSON)
        if household_observation is None or person_observation is None:
            raise ValueError(
                "USMicroplexPipeline requires household and person observations in the source descriptor"
            )

        relationship = next(
            (
                candidate
                for candidate in frame.relationships
                if candidate.parent_entity == EntityType.HOUSEHOLD
                and candidate.child_entity == EntityType.PERSON
                and candidate.cardinality == RelationshipCardinality.ONE_TO_MANY
            ),
            None,
        )
        if relationship is None:
            raise ValueError(
                "USMicroplexPipeline requires a one-to-many household-to-person relationship"
            )

        return USMicroplexSourceInput(
            frame=frame,
            fusion_plan=fusion_plan,
            household_observation=household_observation,
            person_observation=person_observation,
            household_person_relationship=relationship,
            households=households,
            persons=persons,
        )

    def prepare_seed_data_from_source(
        self,
        source_input: USMicroplexSourceInput,
    ) -> pd.DataFrame:
        """Project an observation frame into the canonical US seed schema."""
        household_coverage = source_input.fusion_plan.variables_for(
            EntityType.HOUSEHOLD
        )
        person_coverage = source_input.fusion_plan.variables_for(EntityType.PERSON)
        relationship = source_input.household_person_relationship

        hh = source_input.households.copy()
        persons_df = source_input.persons.copy()

        household_renames = {
            relationship.parent_key: "household_id",
        }
        if source_input.household_observation.weight_column is not None:
            household_renames[source_input.household_observation.weight_column] = (
                "hh_weight"
            )
        hh = hh.rename(columns=household_renames)

        person_renames = {
            source_input.person_observation.key_column: "person_id",
            relationship.child_key: "household_id",
        }
        persons_df = persons_df.rename(columns=person_renames)

        if "household_id" not in hh.columns:
            raise ValueError(
                "USMicroplexPipeline could not resolve a canonical household_id from the source frame"
            )
        if (
            "household_id" not in persons_df.columns
            or "person_id" not in persons_df.columns
        ):
            raise ValueError(
                "USMicroplexPipeline could not resolve canonical person/household linkage columns"
            )

        if "hh_weight" not in hh.columns:
            hh["hh_weight"] = 1.0
        if "state_fips" not in household_coverage or "state_fips" not in hh.columns:
            hh["state_fips"] = 0
        if "county_fips" not in household_coverage or "county_fips" not in hh.columns:
            hh["county_fips"] = 0
        if "tenure" not in household_coverage or "tenure" not in hh.columns:
            hh["tenure"] = 0
        hh = _attach_household_census_geographies(
            hh,
            seed=self.config.random_seed,
        )

        required_person_defaults = {
            "age": 0,
            "sex": 0,
            "education": 0,
            "employment_status": 0,
            "income": 0.0,
        }
        for column, default in required_person_defaults.items():
            if column not in person_coverage or column not in persons_df.columns:
                persons_df[column] = default

        household_seed_columns = [
            "household_id",
            "state_fips",
            "county_fips",
            "hh_weight",
            "tenure",
            "block_geoid",
            "tract_geoid",
            "congressional_district_geoid",
        ]
        seed_data = persons_df.merge(
            hh[[column for column in household_seed_columns if column in hh.columns]],
            on="household_id",
            how="left",
            suffixes=("", "__household"),
        )
        for column in (
            "state_fips",
            "county_fips",
            "hh_weight",
            "tenure",
            "block_geoid",
            "tract_geoid",
            "congressional_district_geoid",
        ):
            household_column = f"{column}__household"
            if household_column not in seed_data.columns:
                continue
            if column in seed_data.columns:
                seed_data[column] = seed_data[household_column].combine_first(
                    seed_data[column]
                )
            else:
                seed_data[column] = seed_data[household_column]
            seed_data = seed_data.drop(columns=[household_column])
        seed_data["hh_weight"] = seed_data["hh_weight"].fillna(1.0).astype(float)
        seed_data["tenure"] = seed_data["tenure"].fillna(0).astype(int)
        seed_data["state_fips"] = seed_data["state_fips"].fillna(0).astype(int)
        seed_data["county_fips"] = (
            seed_data["county_fips"].map(normalize_us_county_fips).fillna("00000")
        )
        if "block_geoid" in seed_data.columns:
            seed_data["block_geoid"] = seed_data["block_geoid"].fillna("").astype(str)
        if "tract_geoid" in seed_data.columns:
            seed_data["tract_geoid"] = seed_data["tract_geoid"].fillna("").astype(str)
        if "congressional_district_geoid" in seed_data.columns:
            seed_data["congressional_district_geoid"] = (
                pd.to_numeric(
                    seed_data["congressional_district_geoid"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )
        seed_data["income"] = pd.to_numeric(
            seed_data["income"], errors="coerce"
        ).fillna(0.0)
        seed_data = normalize_social_security_columns(seed_data)

        seed_data["state"] = seed_data["state_fips"].map(STATE_FIPS).fillna("UNK")
        seed_data["age_group"] = pd.cut(
            seed_data["age"],
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=False,
        )
        seed_data["income_bracket"] = pd.cut(
            seed_data["income"],
            bins=INCOME_BINS,
            labels=INCOME_LABELS,
        )

        return seed_data.reset_index(drop=True)

    def prepare_seed_data(
        self,
        persons: pd.DataFrame,
        households: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge canonical person and household inputs into a synthesis-ready seed frame."""
        return self.prepare_seed_data_from_source(
            self.prepare_source_input(
                self._build_direct_input_frame(
                    persons=persons,
                    households=households,
                )
            )
        )

    def _build_direct_input_frame(
        self,
        *,
        persons: pd.DataFrame,
        households: pd.DataFrame,
    ) -> ObservationFrame:
        """Wrap direct person/household inputs in an observation frame."""
        household_weight_column = next(
            (
                column
                for column in ("hh_weight", "household_weight")
                if column in households.columns
            ),
            None,
        )
        person_weight_column = "weight" if "weight" in persons.columns else None
        household_columns = tuple(
            column
            for column in households.columns
            if column
            not in {
                "household_id",
                household_weight_column,
            }
        )
        person_columns = tuple(
            column
            for column in persons.columns
            if column
            not in {
                "person_id",
                "household_id",
                person_weight_column,
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="us_microplex_direct_input",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=household_columns,
                        weight_column=household_weight_column,
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=person_columns,
                        weight_column=person_weight_column,
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households.copy(),
                EntityType.PERSON: persons.copy(),
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

    def build_targets(
        self,
        seed_data: pd.DataFrame,
        weight_col: str = "hh_weight",
    ) -> USMicroplexTargets:
        """Build weighted calibration targets from the seed data."""
        weights = seed_data[weight_col].astype(float).values
        marginal: dict[str, dict[str, float]] = {}

        for column in ("state", "age_group", "income_bracket"):
            marginal[column] = {}
            categories = seed_data[column].dropna().astype(str).unique()
            for category in categories:
                mask = seed_data[column].astype(str) == category
                marginal[column][category] = float(weights[mask].sum())

        continuous = {
            "income": float((weights * seed_data["income"].astype(float).values).sum())
        }

        if self.config.policyengine_quantity_targets:
            if self.config.policyengine_dataset is None:
                raise ValueError(
                    "policyengine_dataset is required when policyengine_quantity_targets are configured"
                )
            adapter = PolicyEngineUSMicrosimulationAdapter.from_dataset(
                self.config.policyengine_dataset,
                dataset_year=self.config.policyengine_dataset_year,
            )
            continuous.update(
                self.build_policyengine_continuous_targets(
                    seed_data=seed_data,
                    adapter=adapter,
                    quantity_targets=self.config.policyengine_quantity_targets,
                )
            )

        return USMicroplexTargets(marginal=marginal, continuous=continuous)

    def build_policyengine_continuous_targets(
        self,
        seed_data: pd.DataFrame,
        adapter: PolicyEngineUSMicrosimulationAdapter | Any,
        quantity_targets: tuple[PolicyEngineUSQuantityTarget, ...],
    ) -> dict[str, float]:
        """Compute PE-based continuous totals for columns present in the seed data."""
        missing_columns = sorted(
            {
                target.column
                for target in quantity_targets
                if target.column not in seed_data.columns
            }
        )
        if missing_columns:
            raise ValueError(
                f"PolicyEngine target columns not available in seed data: {missing_columns}"
            )

        computed = adapter.compute_targets(quantity_targets)
        continuous_targets: dict[str, float] = {}
        for target in quantity_targets:
            if target.name not in computed:
                raise ValueError(
                    f"PolicyEngine adapter did not return target '{target.name}'"
                )
            continuous_targets[target.column] = float(computed[target.name])
        return continuous_targets

    def ensure_target_support(
        self,
        synthetic_data: pd.DataFrame,
        seed_data: pd.DataFrame,
        targets: USMicroplexTargets,
    ) -> pd.DataFrame:
        """Ensure every marginal target category has support in the synthetic sample."""
        result = synthetic_data.copy().reset_index(drop=True)
        bool_columns = [
            column
            for column in result.columns
            if pd.api.types.is_bool_dtype(result[column].dtype)
        ]
        if bool_columns:
            result[bool_columns] = result[bool_columns].astype(float)
        replace_idx = 0

        for _ in range(sum(len(v) for v in targets.marginal.values())):
            missing: list[tuple[str, str]] = []
            for column, categories in targets.marginal.items():
                current = result[column].astype(str)
                for category in categories:
                    if not (current == str(category)).any():
                        missing.append((column, str(category)))

            if not missing:
                break

            for column, category in missing:
                exemplars = seed_data[seed_data[column].astype(str) == category]
                if exemplars.empty:
                    continue
                exemplar = exemplars.iloc[0]
                row_idx = replace_idx % len(result)
                for column_name, value in exemplar.items():
                    if column_name in result.columns and column_name not in {
                        "person_id",
                        "household_id",
                        "weight",
                    }:
                        resolved_value = value
                        destination = result[column_name]
                        if pd.api.types.is_bool_dtype(
                            destination.dtype
                        ) and not isinstance(
                            resolved_value,
                            (bool, np.bool_),
                        ):
                            result[column_name] = destination.astype(float)
                            destination = result[column_name]
                        if pd.api.types.is_numeric_dtype(
                            destination.dtype
                        ) and isinstance(
                            value,
                            (bool, np.bool_),
                        ):
                            resolved_value = float(value)
                        result.at[row_idx, column_name] = resolved_value
                replace_idx += 1

        initial_weight = (
            float(result["weight"].mean()) if "weight" in result.columns else 1.0
        )
        base = result.drop(
            columns=["person_id", "state", "age_group", "income_bracket"],
            errors="ignore",
        )
        return self._finalize_synthetic_population(base, initial_weight=initial_weight)

    def synthesize(
        self,
        seed_data: pd.DataFrame,
        synthesis_variables: USMicroplexSynthesisVariables | None = None,
    ) -> tuple[pd.DataFrame, Synthesizer | None, dict[str, Any]]:
        """Generate synthetic records from the seed data."""
        if "hh_weight" in seed_data.columns:
            initial_weight = float(seed_data["hh_weight"].sum()) / max(
                self.config.n_synthetic, 1
            )
        else:
            initial_weight = 1.0
        synthesis_variables = synthesis_variables or USMicroplexSynthesisVariables(
            condition_vars=self._resolve_synthesis_condition_vars(
                seed_data.columns,
                observed_frame=seed_data,
            ),
            target_vars=tuple(
                column
                for column in self.config.synthesizer_target_vars
                if column in seed_data.columns
            ),
        )

        if self.config.synthesis_backend == "seed":
            synthetic = seed_data.copy()
            if "hh_weight" in synthetic.columns and "weight" not in synthetic.columns:
                synthetic["weight"] = (
                    pd.to_numeric(synthetic["hh_weight"], errors="coerce")
                    .fillna(initial_weight)
                    .astype(float)
                )
            synthetic = self._finalize_synthetic_population(
                synthetic,
                initial_weight=float(
                    pd.to_numeric(
                        synthetic.get("weight", pd.Series([initial_weight])),
                        errors="coerce",
                    )
                    .fillna(initial_weight)
                    .mean()
                ),
            )
            return (
                synthetic,
                None,
                {
                    "backend": "seed",
                    "n_seed_records": int(len(seed_data)),
                },
            )

        if self.config.synthesis_backend == "bootstrap":
            bootstrap_strata_columns = self._resolve_bootstrap_strata_columns(seed_data)
            synthetic = self._synthesize_bootstrap(
                seed_data,
                initial_weight=initial_weight,
                strata_columns=bootstrap_strata_columns,
            )
            return (
                synthetic,
                None,
                {
                    "backend": "bootstrap",
                    "bootstrap_strata_columns": list(bootstrap_strata_columns),
                },
            )

        synthesizer = self._fit_synthesizer(seed_data, synthesis_variables)
        synthetic = synthesizer.sample(
            self.config.n_synthetic,
            seed=self.config.random_seed,
        )
        synthetic = self._finalize_synthetic_population(
            synthetic,
            initial_weight=initial_weight,
        )
        return synthetic, synthesizer, {"backend": "synthesizer"}

    def calibrate(
        self,
        synthetic_data: pd.DataFrame,
        targets: USMicroplexTargets,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Calibrate synthetic records to weighted targets."""
        if self.config.calibration_backend == "none":
            return synthetic_data.copy(), {
                "backend": "none",
                "max_error": 0.0,
                "mean_error": 0.0,
                "converged": True,
            }
        calibrator = self._build_weight_calibrator()
        if self.config.calibration_backend in {"entropy", "ipf", "chi2"}:
            calibrated = calibrator.fit_transform(
                synthetic_data,
                targets.marginal,
                targets.continuous,
                weight_col="weight",
            )
            validation = calibrator.validate(calibrated)
            all_errors = []
            for var_errors in validation["marginal_errors"].values():
                all_errors.extend(
                    item["relative_error"] for item in var_errors.values()
                )
            all_errors.extend(
                item["relative_error"]
                for item in validation["continuous_errors"].values()
            )
            summary = {
                "backend": self.config.calibration_backend,
                "max_error": float(validation["max_error"]),
                "mean_error": float(np.mean(all_errors)) if all_errors else 0.0,
                "converged": bool(validation["converged"]),
            }
            return calibrated, summary

        calibrated = calibrator.fit_transform(
            synthetic_data,
            targets.marginal,
            targets.continuous,
            weight_col="weight",
        )
        validation = calibrator.validate(calibrated)
        summary = {
            "backend": self.config.calibration_backend,
            "max_error": float(validation["max_error"]),
            "mean_error": float(validation["mean_error"]),
            "sparsity": float(validation.get("sparsity", 0.0)),
            "converged": bool(validation.get("converged", False)),
        }
        return calibrated, summary

    def _build_weight_calibrator(
        self,
        stage_index: int = 1,
    ) -> (
        Calibrator
        | SparseCalibrator
        | HardConcreteCalibrator
        | PolicyEngineL0Calibrator
    ):
        # Stage 1 selects the sparse support via L0; stages 2+ only
        # refine weights against additional targets. Re-applying the same
        # L0 penalty on warm-started weights compounds sparsity and
        # collapses the support set (v10 went 442k → 1.5k across stages).
        sparsity_pass = stage_index <= 1
        l0_penalty = 1e-4 if sparsity_pass else 0.0
        if self.config.calibration_backend in {"entropy", "ipf", "chi2"}:
            return Calibrator(
                method=self.config.calibration_backend,
                tol=self.config.calibration_tol,
                max_iter=self.config.calibration_max_iter,
            )
        if self.config.calibration_backend == "sparse":
            return SparseCalibrator(
                target_sparsity=self.config.target_sparsity,
                tol=self.config.calibration_tol,
                max_iter=max(self.config.calibration_max_iter, 1_000),
            )
        if self.config.calibration_backend == "hardconcrete":
            if l0_penalty <= 0.0:
                from microplex_us.calibration import (
                    MicrocalibrateAdapter,
                    MicrocalibrateAdapterConfig,
                )

                return MicrocalibrateAdapter(
                    MicrocalibrateAdapterConfig(
                        epochs=max(self.config.calibration_max_iter, 32),
                        learning_rate=1e-3,
                        device=self.config.device,
                        seed=self.config.random_seed,
                        regularize_with_l0=False,
                    )
                )
            return HardConcreteCalibrator(
                lambda_l0=l0_penalty,
                epochs=max(self.config.calibration_max_iter, 500),
                lr=0.1,
                device=self.config.device,
                verbose=False,
            )
        if self.config.calibration_backend == "pe_l0":
            return PolicyEngineL0Calibrator(
                lambda_l0=l0_penalty,
                epochs=max(self.config.calibration_max_iter, 100),
                device=self.config.device,
                tol=self.config.calibration_tol,
            )
        if self.config.calibration_backend == "microcalibrate":
            from microplex_us.calibration import (
                MicrocalibrateAdapter,
                MicrocalibrateAdapterConfig,
            )

            return MicrocalibrateAdapter(
                MicrocalibrateAdapterConfig(
                    epochs=max(self.config.calibration_max_iter, 32),
                    learning_rate=1e-3,
                    device=self.config.device,
                    seed=self.config.random_seed,
                )
            )
        raise ValueError(
            f"Unsupported calibration backend: {self.config.calibration_backend}"
        )

    def _select_policyengine_household_budget(
        self,
        tables: PolicyEngineUSEntityTableBundle,
        supported_targets: list[TargetSpec],
        constraints: tuple[LinearConstraint, ...],
    ) -> tuple[
        PolicyEngineUSEntityTableBundle,
        list[TargetSpec],
        tuple[LinearConstraint, ...],
        dict[str, Any],
    ]:
        requested_budget = self.config.policyengine_selection_household_budget
        household_count = len(tables.households)
        if requested_budget is None or requested_budget >= household_count:
            return (
                tables,
                supported_targets,
                constraints,
                {
                    "applied": False,
                    "requested_household_budget": requested_budget,
                    "input_household_count": household_count,
                },
            )
        if requested_budget <= 0:
            raise ValueError("policyengine_selection_household_budget must be positive")
        if not constraints:
            return (
                tables,
                supported_targets,
                constraints,
                {
                    "applied": False,
                    "requested_household_budget": requested_budget,
                    "input_household_count": household_count,
                    "reason": "no_constraints",
                },
            )

        target_sparsity = max(0.0, 1.0 - (requested_budget / household_count))
        household_ids = tables.households["household_id"].to_numpy(dtype=np.int64)
        selection_backend = self.config.policyengine_selection_backend
        state_floor_positions = np.asarray([], dtype=np.int64)
        state_floor_summary = {
            "applied": False,
            "requested_state_floor": int(
                max(self.config.policyengine_selection_state_floor, 0)
            ),
        }
        if selection_backend == "sparse":
            selector = SparseCalibrator(
                target_sparsity=target_sparsity,
                tol=self.config.calibration_tol,
                max_iter=max(self.config.calibration_max_iter, 1_000),
            )
            selector_result = selector.fit_transform(
                tables.households.copy(),
                {},
                weight_col="household_weight",
                linear_constraints=constraints,
            )
            selector_validation = selector.validate(selector_result)
            selector_weights = (
                pd.to_numeric(selector_result["household_weight"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            selector_metadata = {
                "selector_converged": bool(selector_validation.get("converged", False)),
                "selector_max_error": float(selector_validation.get("max_error", 0.0)),
                "selector_mean_error": float(
                    selector_validation.get("mean_error", 0.0)
                ),
                "selector_sparsity": float(selector_validation.get("sparsity", 0.0)),
            }
        elif selection_backend == "pe_native_loss":
            (
                state_floor_positions,
                state_floor_summary,
            ) = self._select_policyengine_state_floor_positions(
                tables=tables,
                requested_budget=requested_budget,
            )
            state_floor_mask = np.zeros(household_count, dtype=bool)
            state_floor_mask[state_floor_positions] = True
            remaining_budget = requested_budget - int(state_floor_mask.sum())
            if remaining_budget < 0:
                raise ValueError(
                    "policyengine_selection_state_floor selects more households than "
                    "policyengine_selection_household_budget allows"
                )
            remaining_tables = (
                _subset_policyengine_tables_by_households(
                    tables,
                    pd.Index(
                        household_ids[~state_floor_mask],
                        name="household_id",
                    ),
                )
                if state_floor_mask.any()
                else tables
            )
            remaining_household_ids = (
                household_ids[~state_floor_mask]
                if state_floor_mask.any()
                else household_ids
            )
            if remaining_budget == 0 or len(remaining_household_ids) == 0:
                selector_weights = np.zeros(
                    len(remaining_household_ids), dtype=np.float64
                )
                optimization_summary = {
                    "metric": "enhanced_cps_native_loss_weight_optimization",
                    "initial_loss": 0.0,
                    "optimized_loss": 0.0,
                    "loss_delta": 0.0,
                    "initial_weight_sum": 0.0,
                    "optimized_weight_sum": 0.0,
                    "household_count": int(len(remaining_household_ids)),
                    "positive_household_count": 0,
                    "budget": int(remaining_budget),
                    "converged": True,
                    "iterations": 0,
                }
            else:
                selector_weights, optimization_summary = (
                    self._select_policyengine_household_budget_with_pe_native_loss(
                        tables=remaining_tables,
                        requested_budget=remaining_budget,
                        household_ids=remaining_household_ids,
                    )
                )
            if state_floor_mask.any():
                full_selector_weights = np.zeros(household_count, dtype=np.float64)
                full_selector_weights[~state_floor_mask] = selector_weights
                floor_priority = (
                    float(selector_weights.max()) + 1.0
                    if selector_weights.size
                    else 1.0
                )
                full_selector_weights[state_floor_mask] = floor_priority
                selector_weights = full_selector_weights
            selector_metadata = {
                "selector_converged": bool(
                    optimization_summary.get("converged", False)
                ),
                "selector_max_error": 0.0,
                "selector_mean_error": 0.0,
                "selector_sparsity": 0.0,
                "pe_native_optimization": optimization_summary,
                "state_floor": state_floor_summary,
            }
        else:
            raise ValueError(
                f"Unsupported policyengine_selection_backend: {selection_backend}"
            )

        ranking = np.lexsort((household_ids, -selector_weights))
        selected_positions = np.sort(ranking[:requested_budget])
        household_mask = np.zeros(household_count, dtype=bool)
        household_mask[selected_positions] = True
        selected_ids = pd.Index(household_ids[household_mask], name="household_id")

        return (
            _subset_policyengine_tables_by_households(tables, selected_ids),
            supported_targets,
            _subset_policyengine_linear_constraints(constraints, household_mask),
            {
                "applied": True,
                "backend": selection_backend,
                "requested_household_budget": int(requested_budget),
                "input_household_count": int(household_count),
                "selected_household_count": int(household_mask.sum()),
                "target_sparsity": float(target_sparsity),
                "selector_nonzero_count": int((selector_weights > 0.0).sum()),
                "selector_positive_selected_count": int(
                    (selector_weights[household_mask] > 0.0).sum()
                ),
                "selector_weight_diagnostics": _summarize_weight_diagnostics(
                    selector_weights
                ),
                **selector_metadata,
            },
        )

    def _select_policyengine_state_floor_positions(
        self,
        *,
        tables: PolicyEngineUSEntityTableBundle,
        requested_budget: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        requested_floor = int(max(self.config.policyengine_selection_state_floor, 0))
        if requested_floor <= 0:
            return (
                np.asarray([], dtype=np.int64),
                {"applied": False, "requested_state_floor": requested_floor},
            )
        households = tables.households.copy()
        if "state_fips" not in households.columns:
            return (
                np.asarray([], dtype=np.int64),
                {
                    "applied": False,
                    "requested_state_floor": requested_floor,
                    "reason": "missing_state_fips",
                },
            )
        ranked = households.loc[
            :, ["household_id", "state_fips", "household_weight"]
        ].copy()
        ranked["_position"] = np.arange(len(ranked), dtype=np.int64)
        ranked["state_fips"] = pd.to_numeric(ranked["state_fips"], errors="coerce")
        ranked["household_weight"] = pd.to_numeric(
            ranked["household_weight"], errors="coerce"
        ).fillna(0.0)
        ranked = ranked.dropna(subset=["state_fips"])
        if ranked.empty:
            return (
                np.asarray([], dtype=np.int64),
                {
                    "applied": False,
                    "requested_state_floor": requested_floor,
                    "reason": "no_rankable_states",
                },
            )
        ranked["state_fips"] = ranked["state_fips"].astype(int)
        ranked = ranked.sort_values(
            ["state_fips", "household_weight", "household_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        selected = ranked.groupby("state_fips", sort=True).head(requested_floor)
        selected_positions = np.sort(selected["_position"].to_numpy(dtype=np.int64))
        if len(selected_positions) > requested_budget:
            raise ValueError(
                "policyengine_selection_state_floor selects "
                f"{len(selected_positions)} households, exceeding budget "
                f"{requested_budget}"
            )
        counts_by_state = (
            selected.groupby("state_fips")["household_id"].size().astype(int).to_dict()
        )
        return (
            selected_positions,
            {
                "applied": True,
                "requested_state_floor": requested_floor,
                "selected_household_count": int(len(selected_positions)),
                "state_count": int(selected["state_fips"].nunique()),
                "counts_by_state": {
                    str(int(state_fips)): int(count)
                    for state_fips, count in counts_by_state.items()
                },
            },
        )

    def _select_policyengine_household_budget_with_pe_native_loss(
        self,
        *,
        tables: PolicyEngineUSEntityTableBundle,
        requested_budget: int,
        household_ids: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        period = (
            self.config.policyengine_dataset_year
            or self.config.policyengine_target_period
            or 2024
        )
        with TemporaryDirectory(prefix="microplex-us-pe-native-selection-") as temp_dir:
            temp_dir_path = Path(temp_dir)
            selection_build_result = USMicroplexBuildResult(
                config=self.config,
                seed_data=pd.DataFrame(),
                synthetic_data=pd.DataFrame(),
                calibrated_data=pd.DataFrame(),
                targets=USMicroplexTargets(marginal={}, continuous={}),
                calibration_summary={},
                policyengine_tables=tables,
            )
            selection_input_path = self.export_policyengine_dataset(
                selection_build_result,
                temp_dir_path / "selection_candidate.h5",
                period=period,
                direct_override_variables=self.config.policyengine_direct_override_variables,
            )
            selection_output_path = temp_dir_path / "selection_candidate_optimized.h5"
            optimization_result = optimize_policyengine_us_native_loss_dataset(
                input_dataset_path=selection_input_path,
                output_dataset_path=selection_output_path,
                period=period,
                **self._policyengine_selection_optimizer_kwargs(
                    requested_budget=requested_budget
                ),
            )
            with h5py.File(selection_output_path, "r") as handle:
                period_key = str(period)
                optimized_household_ids = handle["household_id"][period_key][:].astype(
                    np.int64,
                    copy=False,
                )
                optimized_household_weights = handle["household_weight"][period_key][
                    :
                ].astype(
                    np.float64,
                    copy=False,
                )
        weight_by_household_id = {
            int(household_id): float(weight)
            for household_id, weight in zip(
                optimized_household_ids,
                optimized_household_weights,
                strict=True,
            )
        }
        selector_weights = np.asarray(
            [
                weight_by_household_id[int(household_id)]
                for household_id in household_ids
            ],
            dtype=np.float64,
        )
        optimization_summary = optimization_result.to_dict()
        optimization_summary.pop("target_names", None)
        return selector_weights, optimization_summary

    def _policyengine_selection_optimizer_kwargs(
        self,
        *,
        requested_budget: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "budget": requested_budget,
            "max_iter": max(self.config.policyengine_selection_max_iter, 1),
            "l2_penalty": float(self.config.policyengine_selection_l2_penalty),
            "tol": float(self.config.policyengine_selection_tol),
        }
        if self.config.policyengine_selection_target_total_weight is not None:
            kwargs["target_total_weight"] = float(
                self.config.policyengine_selection_target_total_weight
            )
        return kwargs

    def _puf_clone_household_summary(
        self,
        tables: PolicyEngineUSEntityTableBundle,
    ) -> dict[str, Any]:
        flag_column = self.config.puf_support_clone_flag_column
        if tables.persons is None or flag_column not in tables.persons.columns:
            return {
                "available": False,
                "clone_household_count": 0,
                "mixed_flag_household_count": 0,
            }
        persons = tables.persons
        if "household_id" not in persons.columns:
            return {
                "available": False,
                "reason": "missing_person_household_id",
                "clone_household_count": 0,
                "mixed_flag_household_count": 0,
            }
        flags = pd.to_numeric(persons[flag_column], errors="coerce").fillna(0.0)
        grouped = flags.groupby(persons["household_id"], sort=False)
        flag_min = grouped.min()
        flag_max = grouped.max()
        clone_household_ids = flag_min.index[(flag_min > 0.5) & (flag_max > 0.5)]
        mixed_household_ids = flag_min.index[(flag_min <= 0.5) & (flag_max > 0.5)]
        activated_count = 0
        weight_sum = 0.0
        weight_share = 0.0
        if "household_id" in tables.households.columns:
            households = tables.households
            weights = pd.to_numeric(
                households.get("household_weight", 0.0),
                errors="coerce",
            ).fillna(0.0)
            household_weights = pd.Series(
                weights.to_numpy(dtype=float),
                index=households["household_id"].to_numpy(),
                dtype=float,
            )
            clone_weights = household_weights.reindex(clone_household_ids).fillna(0.0)
            activated_count = int((clone_weights > 0.0).sum())
            weight_sum = float(clone_weights.sum())
            total_weight = float(weights.sum())
            weight_share = float(weight_sum / total_weight) if total_weight else 0.0
        clone_household_id_values = [
            value.item() if hasattr(value, "item") else value
            for value in clone_household_ids.to_list()
        ]
        return {
            "available": True,
            "flag_column": flag_column,
            "clone_household_count": int(len(clone_household_ids)),
            "mixed_flag_household_count": int(len(mixed_household_ids)),
            "activated_household_count": activated_count,
            "household_weight_sum": weight_sum,
            "household_weight_share": weight_share,
            "clone_household_ids": clone_household_id_values,
        }

    def _initialize_puf_clone_calibration_weights(
        self,
        tables: PolicyEngineUSEntityTableBundle,
    ) -> tuple[PolicyEngineUSEntityTableBundle, dict[str, Any]]:
        if not self.config.puf_support_clone_enabled:
            return tables, {"applied": False}
        summary = self._puf_clone_household_summary(tables)
        if not summary.get("available"):
            return tables, {"applied": False, **summary}
        if summary.get("mixed_flag_household_count", 0):
            raise ValueError(
                "PUF support clone household diagnostics found mixed original/clone "
                "person flags within a household"
            )
        if self.config.calibration_backend == "none":
            return tables, {
                "applied": False,
                "reason": "calibration_backend_none",
                **summary,
            }
        clone_household_ids = set(summary.get("clone_household_ids", []))
        if not clone_household_ids or "household_id" not in tables.households.columns:
            return tables, {"applied": False, **summary}
        households = tables.households.copy()
        weights = pd.to_numeric(
            households["household_weight"],
            errors="coerce",
        ).fillna(0.0)
        clone_mask = households["household_id"].isin(clone_household_ids)
        share = float(self.config.puf_support_clone_prior_weight_share)
        clone_count = int(clone_mask.sum())
        original_weight_sum = float(weights.loc[~clone_mask].sum())
        clone_prior_total = (
            original_weight_sum * share / (1.0 - share)
            if share > 0.0 and original_weight_sum > 0.0 and clone_count
            else 0.0
        )
        clone_prior_weight = (
            clone_prior_total / clone_count
            if clone_count and clone_prior_total
            else 0.0
        )
        if clone_prior_weight > 0.0:
            households.loc[clone_mask, "household_weight"] = clone_prior_weight
        updated_tables = PolicyEngineUSEntityTableBundle(
            households=households,
            persons=tables.persons,
            tax_units=tables.tax_units,
            spm_units=tables.spm_units,
            families=tables.families,
            marital_units=tables.marital_units,
        )
        return updated_tables, {
            "applied": bool(clone_prior_weight > 0.0),
            "clone_prior_weight_share": share,
            "clone_prior_total_weight": clone_prior_total,
            "clone_prior_household_weight": clone_prior_weight,
            "clone_household_count": clone_count,
            "pre_clone_weight_sum": float(weights.loc[clone_mask].sum()),
            "pre_clone_original_weight_sum": original_weight_sum,
        }

    def calibrate_policyengine_tables(
        self,
        tables: PolicyEngineUSEntityTableBundle,
    ) -> tuple[PolicyEngineUSEntityTableBundle, pd.DataFrame, dict[str, Any]]:
        """Calibrate household weights using PolicyEngine US target DB constraints."""
        provider, _source = self._resolve_calibration_target_provider()
        target_period = (
            self.config.policyengine_target_period
            or self.config.policyengine_dataset_year
            or 2024
        )
        forbes_fixed_spine = self._build_forbes_fixed_spine()
        tables, ssi_takeup_summary = (
            self._calibrate_policyengine_ssi_takeup_from_reported_amounts(
                tables,
                target_period=target_period,
            )
        )
        (
            tables,
            bindings,
            canonical_targets,
            compiled_targets,
            unsupported_targets,
            compiled_constraints,
            supported_targets,
            constraints,
            feasibility_filter_summary,
            materialized_variables,
            materialization_failures,
            fixed_spine_residualization_summary,
        ) = self._resolve_policyengine_calibration_targets(
            tables,
            provider=provider,
            target_period=target_period,
            forbes_fixed_spine=forbes_fixed_spine,
        )
        if self.config.pipeline_checkpoint_save_post_microsim_path is not None:
            save_us_pipeline_checkpoint(
                tables,
                self.config.pipeline_checkpoint_save_post_microsim_path,
                stage="post_microsim",
            )
            _emit_us_pipeline_progress(
                "US microplex build: post-microsim checkpoint saved",
                path=str(self.config.pipeline_checkpoint_save_post_microsim_path),
            )
        tables, puf_clone_calibration_initialization = (
            self._initialize_puf_clone_calibration_weights(tables)
        )
        preselection_supported_targets = list(supported_targets)
        target_planning_household_count = len(tables.households)
        if not supported_targets:
            raise ValueError(
                "No supported PolicyEngine DB targets matched current tables"
            )
        compiled_constraint_tables = tables
        selection_summary: dict[str, Any] | None = None
        if self.config.policyengine_selection_household_budget is not None:
            preselection_household_ids = compiled_constraint_tables.households[
                "household_id"
            ].to_numpy(dtype=np.int64)
            (
                tables,
                supported_targets,
                constraints,
                selection_summary,
            ) = self._select_policyengine_household_budget(
                tables,
                supported_targets,
                tuple(constraints),
            )
            if selection_summary.get("applied"):
                (
                    supported_targets,
                    constraints,
                    post_selection_feasibility_summary,
                ) = _select_feasible_policyengine_calibration_constraints(
                    supported_targets,
                    constraints,
                    household_count=len(tables.households),
                    max_constraints=self.config.policyengine_calibration_max_constraints,
                    max_constraints_per_household=(
                        self.config.policyengine_calibration_max_constraints_per_household
                    ),
                    min_active_households=(
                        self.config.policyengine_calibration_min_active_households
                    ),
                )
                feasibility_filter_summary = {
                    **post_selection_feasibility_summary,
                    "pre_selection": feasibility_filter_summary,
                }
                if not supported_targets:
                    raise ValueError(
                        "No supported PolicyEngine DB targets remained after household-budget selection"
                    )
                selected_household_ids = tables.households["household_id"].to_numpy(
                    dtype=np.int64
                )
                selection_mask = np.isin(
                    preselection_household_ids,
                    selected_household_ids,
                )
                compiled_constraints = _subset_policyengine_linear_constraints(
                    compiled_constraints,
                    selection_mask,
                )

        input_household_weight_sum = float(tables.households["household_weight"].sum())

        def _apply_policyengine_constraint_stage(
            stage_tables: PolicyEngineUSEntityTableBundle,
            stage_constraints: tuple[LinearConstraint, ...],
            stage_index: int = 1,
        ) -> tuple[PolicyEngineUSEntityTableBundle, pd.DataFrame, dict[str, Any]]:
            stage_input_household_weight_sum = float(
                stage_tables.households["household_weight"].sum()
            )
            stage_calibrator = None
            if self.config.calibration_backend == "none":
                calibrated_households = stage_tables.households.copy()
                pre_rescale_household_weight_sum = stage_input_household_weight_sum
            else:
                stage_calibrator = self._build_weight_calibrator(
                    stage_index=stage_index
                )
                calibration_constraints = list(stage_constraints)
                if self.config.policyengine_calibration_target_total_weight is not None:
                    n_hh = len(stage_tables.households)
                    calibration_constraints.append(
                        LinearConstraint(
                            name="total_household_weight_sum",
                            coefficients=np.ones(n_hh, dtype=float),
                            target=float(
                                self.config.policyengine_calibration_target_total_weight
                            ),
                        )
                    )
                calibrated_households = stage_calibrator.fit_transform(
                    stage_tables.households.copy(),
                    {},
                    weight_col="household_weight",
                    linear_constraints=tuple(calibration_constraints),
                )
                pre_rescale_household_weight_sum = float(
                    calibrated_households["household_weight"].sum()
                )
            weight_sum_rescaled = False
            weight_sum_rescale_mode: str | None = None
            if (
                self.config.policyengine_calibration_rescale_to_target_total_weight
                and self.config.policyengine_calibration_target_total_weight is not None
                and pre_rescale_household_weight_sum > 0.0
                and not np.isclose(
                    pre_rescale_household_weight_sum,
                    float(self.config.policyengine_calibration_target_total_weight),
                )
            ):
                calibrated_households["household_weight"] = calibrated_households[
                    "household_weight"
                ].astype(float) * (
                    float(self.config.policyengine_calibration_target_total_weight)
                    / pre_rescale_household_weight_sum
                )
                weight_sum_rescaled = True
                weight_sum_rescale_mode = "target_total_weight"
            elif (
                self.config.policyengine_calibration_rescale_to_input_weight_sum
                and pre_rescale_household_weight_sum > 0.0
                and not np.isclose(
                    pre_rescale_household_weight_sum,
                    stage_input_household_weight_sum,
                )
            ):
                calibrated_households["household_weight"] = calibrated_households[
                    "household_weight"
                ].astype(float) * (
                    stage_input_household_weight_sum / pre_rescale_household_weight_sum
                )
                weight_sum_rescaled = True
                weight_sum_rescale_mode = "input_weight_sum"
            if self.config.calibration_backend == "none":
                validation = {
                    "converged": True,
                    "max_error": 0.0,
                    "sparsity": 0.0,
                    "linear_errors": {},
                }
            else:
                validation = stage_calibrator.validate(calibrated_households)

            household_weights = calibrated_households.set_index("household_id")[
                "household_weight"
            ]
            calibrated_persons = (
                stage_tables.persons.copy()
                if stage_tables.persons is not None
                else pd.DataFrame()
            )
            if not calibrated_persons.empty:
                calibrated_persons["weight"] = (
                    calibrated_persons["household_id"]
                    .map(household_weights)
                    .astype(float)
                )

            updated_stage_tables = PolicyEngineUSEntityTableBundle(
                households=calibrated_households,
                persons=calibrated_persons
                if not calibrated_persons.empty
                else stage_tables.persons,
                tax_units=stage_tables.tax_units,
                spm_units=stage_tables.spm_units,
                families=stage_tables.families,
                marital_units=stage_tables.marital_units,
            )
            return (
                updated_stage_tables,
                calibrated_persons,
                {
                    "validation": validation,
                    "input_household_weight_sum": stage_input_household_weight_sum,
                    "pre_rescale_household_weight_sum": pre_rescale_household_weight_sum,
                    "post_rescale_household_weight_sum": float(
                        calibrated_households["household_weight"].sum()
                    ),
                    "weight_sum_rescaled": weight_sum_rescaled,
                    "weight_sum_rescale_mode": weight_sum_rescale_mode,
                    "household_weight_diagnostics": _summarize_weight_diagnostics(
                        calibrated_households["household_weight"]
                    ),
                    "person_weight_diagnostics": (
                        _summarize_weight_diagnostics(calibrated_persons["weight"])
                        if not calibrated_persons.empty
                        and "weight" in calibrated_persons.columns
                        else None
                    ),
                },
            )

        selected_stage_by_name = {target.name: 1 for target in supported_targets}
        all_selected_targets = list(supported_targets)
        all_selected_constraints = list(constraints)
        # Pre-compute the ledger-needed scalars once, while compiled_constraints'
        # coefficient arrays are still live. Downstream calls (ledger +
        # deferred-stage selection) read from this lookup instead of
        # rescanning the ~4k × 1.5M float64 arrays three times. The
        # repeated scans were allocating ~30 GB of transient
        # ``np.abs(...)`` copies on top of the 48 GB baseline, a
        # contributor to the v8 197 GB-compressed jetsam kill.
        compiled_constraint_metadata = _precompute_constraint_metadata(
            compiled_constraints
        )
        updated_tables, calibrated_persons, final_stage_summary = (
            _apply_policyengine_constraint_stage(
                tables,
                tuple(constraints),
            )
        )
        target_plan_summary, target_ledger = (
            _build_policyengine_calibration_target_ledger(
                canonical_targets=canonical_targets,
                tables=tables,
                bindings=bindings,
                compiled_targets=compiled_targets,
                structurally_unsupported_targets=unsupported_targets,
                compiled_constraints=compiled_constraints,
                preselection_targets=preselection_supported_targets,
                selected_stage_by_name=selected_stage_by_name,
                household_count=target_planning_household_count,
                min_active_households=self.config.policyengine_calibration_min_active_households,
                materialization_failures=materialization_failures,
                compiled_constraint_metadata=compiled_constraint_metadata,
            )
        )
        oracle_loss, oracle_target_priority_lookup = (
            _evaluate_policyengine_target_fit_context(
                tables=updated_tables,
                canonical_targets=canonical_targets,
                final_solve_targets=all_selected_targets,
                target_ledger=target_ledger,
                period=target_period,
                dataset_year=self.config.policyengine_dataset_year
                or int(target_period),
                simulation_cls=self.config.policyengine_simulation_cls,
                direct_override_variables=(
                    self.config.policyengine_direct_override_variables
                ),
                relative_error_cap=self.config.policyengine_oracle_relative_error_cap,
            )
        )

        calibration_stages: list[dict[str, Any]] = []
        applied_stage_count = 1
        final_stage_index = 1
        deferred_stage_accept_metric = "full_oracle_capped_mean_abs_relative_error"
        deferred_stage_trigger_metric = "full_oracle_capped_mean_abs_relative_error"

        def _append_stage_summary(
            *,
            stage_index: int,
            kind: str,
            status: str,
            min_active_households: int,
            selected_targets_for_stage: list[TargetSpec],
            stage_metadata: dict[str, Any],
            stage_result: dict[str, Any] | None,
            oracle_loss_snapshot: dict[str, dict[str, Any]],
            pre_oracle_loss_snapshot: dict[str, dict[str, Any]] | None = None,
        ) -> None:
            validation = (
                stage_result.get("validation", {}) if stage_result is not None else {}
            )
            linear_errors = list(validation.get("linear_errors", {}).values())
            stage_summary = {
                "stage_index": stage_index,
                "kind": kind,
                "status": status,
                "min_active_households": int(min_active_households),
                "selected_target_count": len(selected_targets_for_stage),
                "selected_constraint_count": len(selected_targets_for_stage),
                "selected_target_names": [
                    target.name for target in selected_targets_for_stage
                ],
                "post_full_oracle_mean_abs_relative_error": oracle_loss_snapshot[
                    "full_oracle"
                ]["mean_abs_relative_error"],
                "post_full_oracle_capped_mean_abs_relative_error": (
                    oracle_loss_snapshot["full_oracle"][
                        "capped_mean_abs_relative_error"
                    ]
                ),
                "post_active_solve_mean_abs_relative_error": oracle_loss_snapshot[
                    "active_solve"
                ]["mean_abs_relative_error"],
                "post_active_solve_capped_mean_abs_relative_error": (
                    oracle_loss_snapshot["active_solve"][
                        "capped_mean_abs_relative_error"
                    ]
                ),
                **stage_metadata,
            }
            if pre_oracle_loss_snapshot is not None:
                stage_summary.update(
                    {
                        "pre_full_oracle_mean_abs_relative_error": (
                            pre_oracle_loss_snapshot["full_oracle"][
                                "mean_abs_relative_error"
                            ]
                        ),
                        "pre_full_oracle_capped_mean_abs_relative_error": (
                            pre_oracle_loss_snapshot["full_oracle"][
                                "capped_mean_abs_relative_error"
                            ]
                        ),
                        "pre_active_solve_mean_abs_relative_error": (
                            pre_oracle_loss_snapshot["active_solve"][
                                "mean_abs_relative_error"
                            ]
                        ),
                        "pre_active_solve_capped_mean_abs_relative_error": (
                            pre_oracle_loss_snapshot["active_solve"][
                                "capped_mean_abs_relative_error"
                            ]
                        ),
                    }
                )
            if stage_result is not None:
                stage_summary.update(
                    {
                        "input_household_weight_sum": stage_result[
                            "input_household_weight_sum"
                        ],
                        "pre_rescale_household_weight_sum": stage_result[
                            "pre_rescale_household_weight_sum"
                        ],
                        "post_rescale_household_weight_sum": stage_result[
                            "post_rescale_household_weight_sum"
                        ],
                        "weight_sum_rescaled": stage_result["weight_sum_rescaled"],
                        "weight_sum_rescale_mode": stage_result[
                            "weight_sum_rescale_mode"
                        ],
                        "household_weight_diagnostics": stage_result[
                            "household_weight_diagnostics"
                        ],
                        "person_weight_diagnostics": stage_result[
                            "person_weight_diagnostics"
                        ],
                        "max_error": float(validation.get("max_error", 0.0)),
                        "effective_backend": validation.get("backend"),
                        "uses_gates": validation.get("uses_gates"),
                        "mean_error": (
                            float(
                                np.mean(
                                    [error["relative_error"] for error in linear_errors]
                                )
                            )
                            if linear_errors
                            else 0.0
                        ),
                        "converged": bool(validation.get("converged", False)),
                        "sparsity": float(validation.get("sparsity", 0.0)),
                    }
                )
            calibration_stages.append(stage_summary)

        _append_stage_summary(
            stage_index=1,
            kind="initial",
            status="applied",
            min_active_households=self.config.policyengine_calibration_min_active_households,
            selected_targets_for_stage=list(supported_targets),
            stage_metadata={"feasibility_filter": feasibility_filter_summary},
            stage_result=final_stage_summary,
            oracle_loss_snapshot=oracle_loss,
        )

        deferred_stage_schedule: list[int] = []
        for (
            min_active_households
        ) in self.config.policyengine_calibration_deferred_stage_min_active_households:
            resolved_min_active = int(min_active_households)
            if (
                resolved_min_active
                >= self.config.policyengine_calibration_min_active_households
                or resolved_min_active in deferred_stage_schedule
            ):
                continue
            deferred_stage_schedule.append(resolved_min_active)

        if self.config.calibration_backend != "none":
            for stage_index, min_active_households in enumerate(
                deferred_stage_schedule,
                start=2,
            ):
                pre_stage_oracle_loss = oracle_loss
                pre_stage_trigger_metric_value = pre_stage_oracle_loss["full_oracle"][
                    "capped_mean_abs_relative_error"
                ]
                trigger_threshold = self.config.policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error
                if (
                    trigger_threshold is not None
                    and pre_stage_trigger_metric_value is not None
                    and float(pre_stage_trigger_metric_value) < float(trigger_threshold)
                ):
                    _append_stage_summary(
                        stage_index=stage_index,
                        kind="deferred",
                        status="skipped",
                        min_active_households=min_active_households,
                        selected_targets_for_stage=[],
                        stage_metadata={
                            "trigger_metric": deferred_stage_trigger_metric,
                            "trigger_threshold": float(trigger_threshold),
                            "trigger_metric_value": float(
                                pre_stage_trigger_metric_value
                            ),
                            "skip_reason": "trigger_metric_below_threshold",
                        },
                        stage_result=None,
                        oracle_loss_snapshot=oracle_loss,
                        pre_oracle_loss_snapshot=pre_stage_oracle_loss,
                    )
                    continue
                stage_targets, stage_constraints, stage_metadata = (
                    _select_policyengine_deferred_stage_constraints(
                        compiled_targets=compiled_targets,
                        compiled_constraints=compiled_constraints,
                        target_ledger=target_ledger,
                        deferred_oracle_loss=oracle_loss["deferred"],
                        deferred_target_priority_lookup=oracle_target_priority_lookup[
                            "deferred"
                        ],
                        selected_target_names=set(selected_stage_by_name),
                        household_count=target_planning_household_count,
                        min_active_households=min_active_households,
                        max_constraints=(
                            self.config.policyengine_calibration_deferred_stage_max_constraints
                            if self.config.policyengine_calibration_deferred_stage_max_constraints
                            is not None
                            else self.config.policyengine_calibration_max_constraints
                        ),
                        max_constraints_per_household=(
                            self.config.policyengine_calibration_max_constraints_per_household
                        ),
                        top_family_count=(
                            self.config.policyengine_calibration_deferred_stage_top_family_count
                        ),
                        top_geography_count=(
                            self.config.policyengine_calibration_deferred_stage_top_geography_count
                        ),
                        compiled_constraint_metadata=compiled_constraint_metadata,
                    )
                )
                if not stage_targets:
                    _append_stage_summary(
                        stage_index=stage_index,
                        kind="deferred",
                        status="skipped",
                        min_active_households=min_active_households,
                        selected_targets_for_stage=[],
                        stage_metadata=stage_metadata,
                        stage_result=None,
                        oracle_loss_snapshot=oracle_loss,
                        pre_oracle_loss_snapshot=pre_stage_oracle_loss,
                    )
                    continue
                (
                    candidate_tables,
                    candidate_calibrated_persons,
                    candidate_stage_summary,
                ) = _apply_policyengine_constraint_stage(
                    updated_tables,
                    stage_constraints,
                    stage_index=stage_index,
                )
                candidate_selected_stage_by_name = dict(selected_stage_by_name)
                for target in stage_targets:
                    candidate_selected_stage_by_name[target.name] = stage_index
                candidate_all_selected_targets = [
                    *all_selected_targets,
                    *stage_targets,
                ]
                candidate_all_selected_constraints = [
                    *all_selected_constraints,
                    *stage_constraints,
                ]
                candidate_target_plan_summary, candidate_target_ledger = (
                    _build_policyengine_calibration_target_ledger(
                        canonical_targets=canonical_targets,
                        tables=tables,
                        bindings=bindings,
                        compiled_targets=compiled_targets,
                        structurally_unsupported_targets=unsupported_targets,
                        compiled_constraints=compiled_constraints,
                        preselection_targets=preselection_supported_targets,
                        selected_stage_by_name=candidate_selected_stage_by_name,
                        household_count=target_planning_household_count,
                        min_active_households=(
                            self.config.policyengine_calibration_min_active_households
                        ),
                        materialization_failures=materialization_failures,
                        compiled_constraint_metadata=compiled_constraint_metadata,
                    )
                )
                candidate_oracle_loss, candidate_target_priority_lookup = (
                    _evaluate_policyengine_target_fit_context(
                        tables=candidate_tables,
                        canonical_targets=canonical_targets,
                        final_solve_targets=candidate_all_selected_targets,
                        target_ledger=candidate_target_ledger,
                        period=target_period,
                        dataset_year=self.config.policyengine_dataset_year
                        or int(target_period),
                        simulation_cls=self.config.policyengine_simulation_cls,
                        direct_override_variables=(
                            self.config.policyengine_direct_override_variables
                        ),
                        relative_error_cap=(
                            self.config.policyengine_oracle_relative_error_cap
                        ),
                    )
                )
                pre_metric = pre_stage_oracle_loss["full_oracle"][
                    "capped_mean_abs_relative_error"
                ]
                post_metric = candidate_oracle_loss["full_oracle"][
                    "capped_mean_abs_relative_error"
                ]
                stage_improved = (
                    pre_metric is None
                    or post_metric is None
                    or float(post_metric) < float(pre_metric)
                )
                if stage_improved:
                    updated_tables = candidate_tables
                    calibrated_persons = candidate_calibrated_persons
                    final_stage_summary = candidate_stage_summary
                    applied_stage_count += 1
                    final_stage_index = stage_index
                    selected_stage_by_name = candidate_selected_stage_by_name
                    all_selected_targets = candidate_all_selected_targets
                    all_selected_constraints = candidate_all_selected_constraints
                    target_plan_summary = candidate_target_plan_summary
                    target_ledger = candidate_target_ledger
                    oracle_loss = candidate_oracle_loss
                    oracle_target_priority_lookup = candidate_target_priority_lookup
                _append_stage_summary(
                    stage_index=stage_index,
                    kind="deferred",
                    status="applied" if stage_improved else "rejected",
                    min_active_households=min_active_households,
                    selected_targets_for_stage=stage_targets,
                    stage_metadata={
                        **stage_metadata,
                        "accept_metric": deferred_stage_accept_metric,
                        "accepted": stage_improved,
                        "trigger_metric": deferred_stage_trigger_metric,
                        "trigger_threshold": (
                            float(trigger_threshold)
                            if trigger_threshold is not None
                            else None
                        ),
                        "trigger_metric_value": (
                            float(pre_stage_trigger_metric_value)
                            if pre_stage_trigger_metric_value is not None
                            else None
                        ),
                    },
                    stage_result=candidate_stage_summary,
                    oracle_loss_snapshot=candidate_oracle_loss,
                    pre_oracle_loss_snapshot=pre_stage_oracle_loss,
                )

        validation = dict(final_stage_summary["validation"])
        linear_errors = list(validation.get("linear_errors", {}).values())
        household_weight_diagnostics = final_stage_summary[
            "household_weight_diagnostics"
        ]
        person_weight_diagnostics = final_stage_summary["person_weight_diagnostics"]
        summary = {
            "backend": f"policyengine_db_{self.config.calibration_backend}",
            "period": int(target_period),
            "n_loaded_targets": len(canonical_targets),
            "n_supported_targets": len(all_selected_targets),
            "n_unsupported_targets": len(unsupported_targets),
            "n_constraints": len(all_selected_constraints),
            "feasibility_filter": feasibility_filter_summary,
            "calibration_stages": calibration_stages,
            "n_calibration_stages_applied": applied_stage_count,
            "final_calibration_stage_index": final_stage_index,
            "deferred_stage_support_schedule": deferred_stage_schedule,
            "deferred_stage_accept_metric": deferred_stage_accept_metric,
            "deferred_stage_trigger_metric": deferred_stage_trigger_metric,
            "deferred_stage_trigger_threshold": (
                self.config.policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error
            ),
            "target_variables": list(
                self._policyengine_target_scope(for_calibration=True)[0]
            ),
            "target_domains": list(
                self._policyengine_target_scope(for_calibration=True)[1]
            ),
            "target_geo_levels": list(
                self._policyengine_target_scope(for_calibration=True)[2]
            ),
            "target_profile": self._policyengine_target_profile(for_calibration=True),
            "target_cell_count": len(
                self._policyengine_target_cells(for_calibration=True)
            ),
            "materialized_variables": sorted(materialized_variables),
            "materialization_failures": materialization_failures,
            "ssi_takeup": ssi_takeup_summary,
            "max_error": float(validation["max_error"]),
            "mean_error": (
                float(np.mean([error["relative_error"] for error in linear_errors]))
                if linear_errors
                else 0.0
            ),
            "converged": bool(validation["converged"]),
            "sparsity": float(validation.get("sparsity", 0.0)),
            "weight_collapse_suspected": bool(
                household_weight_diagnostics["collapse_suspected"]
                or (
                    person_weight_diagnostics is not None
                    and person_weight_diagnostics["collapse_suspected"]
                )
            ),
            "input_household_weight_sum": input_household_weight_sum,
            "total_weight_constraint_target": self.config.policyengine_calibration_target_total_weight,
            "pre_rescale_household_weight_sum": final_stage_summary[
                "pre_rescale_household_weight_sum"
            ],
            "post_rescale_household_weight_sum": final_stage_summary[
                "post_rescale_household_weight_sum"
            ],
            "weight_sum_rescaled": final_stage_summary["weight_sum_rescaled"],
            "weight_sum_rescale_mode": final_stage_summary["weight_sum_rescale_mode"],
            "household_weight_diagnostics": household_weight_diagnostics,
            "person_weight_diagnostics": person_weight_diagnostics,
            "target_plan": target_plan_summary,
            "target_ledger": target_ledger,
            "oracle_loss": oracle_loss,
            "oracle_relative_error_cap": self.config.policyengine_oracle_relative_error_cap,
            "full_oracle_mean_abs_relative_error": oracle_loss["full_oracle"][
                "mean_abs_relative_error"
            ],
            "full_oracle_capped_mean_abs_relative_error": oracle_loss["full_oracle"][
                "capped_mean_abs_relative_error"
            ],
            "active_solve_mean_abs_relative_error": oracle_loss["active_solve"][
                "mean_abs_relative_error"
            ],
            "active_solve_capped_mean_abs_relative_error": oracle_loss["active_solve"][
                "capped_mean_abs_relative_error"
            ],
            "puf_support_clone": {
                "enabled": bool(self.config.puf_support_clone_enabled),
                "calibration_initialization": puf_clone_calibration_initialization,
                "final_household_diagnostics": self._puf_clone_household_summary(
                    updated_tables
                ),
            },
        }
        if selection_summary is not None:
            summary["selection"] = selection_summary
        if forbes_fixed_spine is not None:
            updated_tables = append_forbes_fixed_spine_tables(
                updated_tables,
                forbes_fixed_spine,
            )
            calibrated_persons = (
                updated_tables.persons.copy()
                if updated_tables.persons is not None
                else pd.DataFrame()
            )
            summary["fixed_spine"] = {
                "enabled": True,
                "source_metadata": forbes_fixed_spine.source_metadata,
                "record_metadata_rows": int(len(forbes_fixed_spine.record_metadata)),
                "residualization": fixed_spine_residualization_summary,
                "post_append_households": int(len(updated_tables.households)),
                "post_append_household_weight_sum": float(
                    updated_tables.households["household_weight"].sum()
                ),
            }
        else:
            summary["fixed_spine"] = {"enabled": False}
        warning_messages = list(feasibility_filter_summary.get("warning_messages", ()))
        for stage in calibration_stages[1:]:
            stage_warnings = stage.get("feasibility_filter", {}).get(
                "warning_messages", ()
            )
            warning_messages.extend(
                f"Deferred calibration stage {stage['stage_index']}: {message}"
                for message in stage_warnings
            )
        if any(
            stage.get("status") == "applied" and not stage.get("converged", True)
            for stage in calibration_stages
        ):
            warning_messages.append(
                "Calibration did not converge on one or more selected constraint sets."
            )
        summary["warnings"] = warning_messages
        for message in warning_messages:
            warnings.warn(message, stacklevel=2)
        return updated_tables, calibrated_persons, summary

    def _check_policyengine_export_column_contract(
        self,
        tables: PolicyEngineUSEntityTableBundle,
        *,
        stage: str,
    ) -> None:
        contract_path = self.config.policyengine_export_column_contract_path
        if contract_path is None:
            return

        tax_benefit_system = self._resolve_policyengine_tax_benefit_system()
        contract = load_contract(Path(contract_path))
        present = build_policyengine_us_export_column_names(
            tables,
            tax_benefit_system=tax_benefit_system,
            direct_override_variables=self.config.policyengine_direct_override_variables,
        )
        diff = compute_column_diff(
            present,
            required=set(contract["required"]),
            forbidden=set(contract["forbidden"]),
            optional=set(contract["ecps_internal_optional"]),
            excluded=set(contract.get("formula_owned_excluded", [])),
        )
        _emit_us_pipeline_progress(
            "US microplex build: policyengine export columns check complete",
            stage=stage,
            status="pass" if diff.ok else "fail",
            columns_present=int(len(present)),
            missing_required=int(len(diff.missing_required)),
            forbidden_present=int(len(diff.forbidden_present)),
        )
        if diff.ok:
            return
        report = _format_export_column_report(
            diff,
            source=f"{stage}:{contract_path}",
            n_present=len(present),
            n_required=len(contract["required"]),
            n_forbidden=len(contract["forbidden"]),
        )
        raise ValueError(report)

    def _build_forbes_fixed_spine(self) -> ForbesFixedSpine | None:
        path = self.config.forbes_fixed_spine_records_path
        if path is None:
            return None
        return build_forbes_fixed_spine(
            path,
            config=ForbesFixedSpineConfig(
                period=(
                    self.config.policyengine_target_period
                    or self.config.policyengine_dataset_year
                    or 2024
                ),
                snapshot_id=self.config.forbes_fixed_spine_snapshot_id,
                replicates_per_unit=self.config.forbes_fixed_spine_replicates_per_unit,
            ),
            source_metadata={
                "configured_by": "USMicroplexBuildConfig.forbes_fixed_spine_records_path"
            },
        )

    def _resolve_policyengine_calibration_targets(
        self,
        tables: PolicyEngineUSEntityTableBundle,
        *,
        provider: PolicyEngineUSDBTargetProvider,
        target_period: int,
        forbes_fixed_spine: ForbesFixedSpine | None = None,
    ) -> tuple[
        PolicyEngineUSEntityTableBundle,
        dict[str, PolicyEngineUSVariableBinding],
        list[TargetSpec],
        list[TargetSpec],
        list[TargetSpec],
        tuple[Any, ...],
        list[TargetSpec],
        tuple[Any, ...],
        dict[str, Any],
        set[str],
        dict[str, str],
        dict[str, Any] | None,
    ]:
        bindings = infer_policyengine_us_variable_bindings(tables)
        canonical_targets = self._load_policyengine_target_set(
            provider,
            bindings=bindings,
            period=target_period,
            for_calibration=True,
        ).targets
        force_materialize_variables = policyengine_us_formula_variables_for_targets(
            canonical_targets,
            simulation_cls=self.config.policyengine_simulation_cls,
            direct_override_variables=self.config.policyengine_direct_override_variables,
        )
        missing_variables = policyengine_us_variables_to_materialize(
            canonical_targets,
            bindings,
            force_materialize_variables=force_materialize_variables,
        )
        materialization_failures: dict[str, str] = {}
        materialized_variables: set[str] = set()
        if missing_variables:
            materialization_result = materialize_policyengine_us_variables_safely(
                tables,
                variables=tuple(sorted(missing_variables)),
                period=target_period,
                dataset_year=self.config.policyengine_dataset_year or target_period,
                simulation_cls=self.config.policyengine_simulation_cls,
                direct_override_variables=self.config.policyengine_direct_override_variables,
                batch_size=self.config.policyengine_materialize_batch_size,
            )
            tables = materialization_result.tables
            unmaterialized_forced_variables = (
                force_materialize_variables
                & missing_variables - set(materialization_result.bindings)
            )
            bindings = {
                variable: binding
                for variable, binding in bindings.items()
                if variable not in unmaterialized_forced_variables
            }
            bindings = {
                **bindings,
                **materialization_result.bindings,
            }
            materialized_variables = set(materialization_result.materialized_variables)
            materialization_failures = dict(materialization_result.failed_variables)
            canonical_targets = self._load_policyengine_target_set(
                provider,
                bindings=bindings,
                period=target_period,
                for_calibration=True,
            ).targets
        fixed_spine_residualization_summary: dict[str, Any] | None = None
        if forbes_fixed_spine is not None:
            residualization_result = residualize_targets_for_fixed_spine(
                canonical_targets,
                forbes_fixed_spine.tables,
            )
            canonical_targets = list(residualization_result.targets.targets)
            fixed_spine_residualization_summary = {
                "target_count": len(canonical_targets),
                "supported_target_count": sum(
                    contribution.status == "supported"
                    for contribution in residualization_result.contributions
                ),
                "unsupported_target_count": sum(
                    contribution.status != "supported"
                    for contribution in residualization_result.contributions
                ),
                "contributions": residualization_result.diagnostics(),
            }
        supported_targets = filter_supported_policyengine_us_targets(
            canonical_targets,
            tables,
            bindings,
        )
        supported_targets, unsupported_targets, constraints = (
            compile_supported_policyengine_us_household_linear_constraints(
                supported_targets,
                tables,
                variable_bindings=bindings,
            )
        )
        compiled_targets = list(supported_targets)
        compiled_constraints = tuple(constraints)
        (
            supported_targets,
            constraints,
            feasibility_filter_summary,
        ) = _select_feasible_policyengine_calibration_constraints(
            supported_targets,
            constraints,
            household_count=len(tables.households),
            max_constraints=self.config.policyengine_calibration_max_constraints,
            max_constraints_per_household=(
                self.config.policyengine_calibration_max_constraints_per_household
            ),
            min_active_households=(
                self.config.policyengine_calibration_min_active_households
            ),
        )
        return (
            tables,
            bindings,
            canonical_targets,
            compiled_targets,
            unsupported_targets,
            compiled_constraints,
            supported_targets,
            constraints,
            feasibility_filter_summary,
            materialized_variables,
            materialization_failures,
            fixed_spine_residualization_summary,
        )

    def _has_policyengine_calibration_targets(self) -> bool:
        if self.config.calibration_target_source == "arch":
            return self.config.arch_targets_db is not None
        return self.config.policyengine_targets_db is not None

    def _resolve_calibration_target_provider(self):
        if self.config.calibration_target_source == "arch":
            if self.config.arch_targets_db is None:
                raise ValueError(
                    "arch_targets_db is required when calibration_target_source='arch'"
                )
            return (
                resolve_arch_sqlite_target_provider(self.config.arch_targets_db),
                "arch",
            )
        if self.config.policyengine_targets_db is None:
            raise ValueError(
                "policyengine_targets_db is required for PolicyEngine DB calibration"
            )
        return (
            PolicyEngineUSDBTargetProvider(self.config.policyengine_targets_db),
            "policyengine",
        )

    def _load_policyengine_target_set(
        self,
        provider: Any,
        *,
        bindings: dict[str, PolicyEngineUSVariableBinding],
        period: int,
        for_calibration: bool,
    ):
        return provider.load_target_set(
            self._build_policyengine_target_query(
                bindings,
                period=period,
                for_calibration=for_calibration,
            )
        )

    def _policyengine_target_scope(
        self,
        *,
        for_calibration: bool,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        variables = (
            self.config.policyengine_calibration_target_variables
            if for_calibration and self.config.policyengine_calibration_target_variables
            else self.config.policyengine_target_variables
        )
        domain_variables = (
            self.config.policyengine_calibration_target_domains
            if for_calibration and self.config.policyengine_calibration_target_domains
            else self.config.policyengine_target_domains
        )
        geo_levels = (
            self.config.policyengine_calibration_target_geo_levels
            if for_calibration
            and self.config.policyengine_calibration_target_geo_levels
            else self.config.policyengine_target_geo_levels
        )
        return variables, domain_variables, geo_levels

    def _policyengine_target_profile(
        self,
        *,
        for_calibration: bool,
    ) -> str | None:
        return (
            self.config.policyengine_calibration_target_profile
            if for_calibration and self.config.policyengine_calibration_target_profile
            else self.config.policyengine_target_profile
        )

    def _policyengine_target_cells(
        self,
        *,
        for_calibration: bool,
    ) -> tuple[PolicyEngineUSTargetCell, ...]:
        profile_name = self._policyengine_target_profile(
            for_calibration=for_calibration
        )
        if profile_name is None:
            return ()
        return resolve_policyengine_us_target_profile(profile_name)

    def _policyengine_calibration_scope_includes_ssi(self) -> bool:
        variables, domain_variables, _geo_levels = self._policyengine_target_scope(
            for_calibration=True
        )
        if "ssi" in variables:
            return True
        if any("ssi" in str(domain).split(",") for domain in domain_variables):
            return True
        for cell in self._policyengine_target_cells(for_calibration=True):
            cell_domains = tuple(
                item.strip()
                for item in str(cell.domain_variable or "").split(",")
                if item.strip()
            )
            if cell.variable == "ssi" or "ssi" in cell_domains:
                return True
        return False

    def _calibrate_policyengine_ssi_takeup_from_reported_amounts(
        self,
        tables: PolicyEngineUSEntityTableBundle,
        *,
        target_period: int,
    ) -> tuple[PolicyEngineUSEntityTableBundle, dict[str, Any]]:
        if not self.config.policyengine_calibrate_ssi_takeup:
            return tables, {"enabled": False, "reason": "disabled_by_config"}
        if not self._policyengine_calibration_scope_includes_ssi():
            return tables, {"enabled": False, "reason": "target_scope_excludes_ssi"}
        if tables.persons is None or tables.persons.empty:
            return tables, {"enabled": False, "reason": "missing_person_table"}
        persons = tables.persons.copy()
        required_columns = {"person_id", "age", "weight", "ssi"}
        missing_columns = sorted(required_columns - set(persons.columns))
        if missing_columns:
            return tables, {
                "enabled": False,
                "reason": "missing_required_columns",
                "missing_columns": missing_columns,
            }
        reported_ssi = (
            pd.to_numeric(persons["ssi"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        if not reported_ssi.gt(0.0).any():
            persons["takes_up_ssi_if_eligible"] = False
            return (
                PolicyEngineUSEntityTableBundle(
                    households=tables.households,
                    persons=persons,
                    tax_units=tables.tax_units,
                    spm_units=tables.spm_units,
                    families=tables.families,
                    marital_units=tables.marital_units,
                ),
                {
                    "enabled": True,
                    "method": "reported_ssi_amount_by_age_group",
                    "reason": "no_reported_positive_ssi",
                    "selected_recipients": 0.0,
                    "selected_amount": 0.0,
                },
            )

        full_takeup_persons = persons.copy()
        full_takeup_persons["takes_up_ssi_if_eligible"] = True
        full_takeup_tables = PolicyEngineUSEntityTableBundle(
            households=tables.households,
            persons=full_takeup_persons,
            tax_units=tables.tax_units,
            spm_units=tables.spm_units,
            families=tables.families,
            marital_units=tables.marital_units,
        )
        materialization_result = materialize_policyengine_us_variables_safely(
            full_takeup_tables,
            variables=("ssi",),
            period=target_period,
            dataset_year=self.config.policyengine_dataset_year or target_period,
            simulation_cls=self.config.policyengine_simulation_cls,
            direct_override_variables=self.config.policyengine_direct_override_variables,
            batch_size=self.config.policyengine_materialize_batch_size,
        )
        materialized_persons = materialization_result.tables.persons
        if (
            materialized_persons is None
            or "ssi" not in materialized_persons.columns
            or "ssi" not in materialization_result.bindings
        ):
            return tables, {
                "enabled": False,
                "reason": "full_takeup_ssi_materialization_failed",
                "materialization_failures": dict(
                    materialization_result.failed_variables
                ),
            }

        selected, selection_summary = _select_ssi_takeup_by_age_amount(
            person_ids=persons["person_id"],
            ages=persons["age"],
            weights=persons["weight"],
            reported_ssi=reported_ssi,
            full_takeup_ssi=materialized_persons["ssi"],
        )
        persons["takes_up_ssi_if_eligible"] = selected
        updated_tables = PolicyEngineUSEntityTableBundle(
            households=tables.households,
            persons=persons,
            tax_units=tables.tax_units,
            spm_units=tables.spm_units,
            families=tables.families,
            marital_units=tables.marital_units,
        )
        return updated_tables, selection_summary

    def _build_policyengine_target_query(
        self,
        bindings: dict[str, PolicyEngineUSVariableBinding],
        *,
        period: int,
        for_calibration: bool = False,
    ) -> TargetQuery:
        variables, domain_variables, geo_levels = self._policyengine_target_scope(
            for_calibration=for_calibration
        )
        profile_name = self._policyengine_target_profile(
            for_calibration=for_calibration
        )
        target_cells = self._policyengine_target_cells(for_calibration=for_calibration)
        return TargetQuery(
            period=period,
            provider_filters={
                "variables": list(variables) if variables else None,
                "domain_variables": (
                    list(domain_variables) if domain_variables else None
                ),
                "geo_levels": list(geo_levels) if geo_levels else None,
                "target_profile": profile_name,
                "target_cells": (
                    [cell.to_provider_filter() for cell in target_cells]
                    if target_cells
                    else None
                ),
                "reform_id": self.config.policyengine_target_reform_id,
                "entity_overrides": {
                    variable: binding.entity for variable, binding in bindings.items()
                },
            },
        )

    def build_policyengine_entity_tables(
        self,
        population: pd.DataFrame,
    ) -> PolicyEngineUSEntityTableBundle:
        """Build a PolicyEngine-oriented multientity bundle from person rows."""
        persons = population.copy().reset_index(drop=True)
        if "person_id" not in persons.columns:
            persons["person_id"] = np.arange(len(persons), dtype=np.int64)
        if "household_id" not in persons.columns:
            persons["household_id"] = np.arange(len(persons), dtype=np.int64)
        if "weight" not in persons.columns:
            persons["weight"] = 1.0
        if "income" not in persons.columns:
            persons["income"] = 0.0
        if "age" not in persons.columns:
            persons["age"] = 0

        persons["person_id"] = persons["person_id"].astype(np.int64)
        persons["household_id"] = persons["household_id"].astype(np.int64)
        persons["weight"] = pd.to_numeric(persons["weight"], errors="coerce").fillna(
            0.0
        )
        persons["income"] = pd.to_numeric(persons["income"], errors="coerce").fillna(
            0.0
        )
        persons["age"] = (
            pd.to_numeric(persons["age"], errors="coerce").fillna(0).astype(int)
        )
        household_ids = persons["household_id"]
        for column, threshold in (("count_under_18", 18), ("count_under_6", 6)):
            under_threshold = persons["age"].lt(threshold).astype(np.int64)
            persons[column] = under_threshold.groupby(
                household_ids, sort=False
            ).transform("sum")
        persons = self._augment_policyengine_person_inputs(persons)
        persons["relationship_to_head"] = self._normalize_relationship_to_head(persons)
        persons = self._assign_policyengine_household_head_flag(persons)
        persons = self._attach_policyengine_person_takeup_inputs(persons)

        households = self._build_policyengine_households(persons)
        tax_units, persons = self._build_policyengine_tax_units(persons)
        tax_units = self._attach_policyengine_tax_unit_takeup_inputs(tax_units)
        persons = self._construct_aotc_eligibility_inputs(persons)
        persons = self._assign_family_and_spm_units(persons)
        persons = self._attach_policyengine_wic_inputs(persons)
        families = self._collapse_group_table(persons, "family_id")
        spm_units = self._collapse_group_table(persons, "spm_unit_id")
        spm_units = self._attach_spm_unit_source_columns(persons, spm_units)
        if "tenure_type" in persons.columns:
            spm_tenure = (
                persons.groupby("spm_unit_id", as_index=False)["tenure_type"]
                .first()
                .rename(columns={"tenure_type": "spm_unit_tenure_type"})
            )
            spm_units = spm_units.merge(spm_tenure, on="spm_unit_id", how="left")
        persons = self._assign_marital_units(persons)
        marital_units = self._collapse_group_table(persons, "marital_unit_id")

        tables = PolicyEngineUSEntityTableBundle(
            households=households,
            persons=persons,
            tax_units=tax_units,
            spm_units=spm_units,
            families=families,
            marital_units=marital_units,
        )
        return tables

    # AOTC eligibility-input columns populated by
    # ``_construct_aotc_eligibility_inputs``, matching the per-student inputs
    # written by the enhanced-CPS baseline ``_impute_aotc_eligibility_inputs``
    # (PolicyEngine/policyengine-us-data, unmerged branch
    # ``codex/fix-aotc-eligibility``).
    _AOTC_TRUE_FLAG_COLUMNS = (
        "is_pursuing_credential_for_american_opportunity_credit",
        "attends_eligible_educational_institution_for_american_opportunity_credit",
        "is_enrolled_at_least_half_time_for_american_opportunity_credit",
        "has_american_opportunity_credit_1098_t_or_exception",
        "has_american_opportunity_credit_institution_ein",
    )
    _AOTC_FALSE_FLAG_COLUMNS = (
        "has_completed_first_four_years_of_postsecondary_education",
        "has_felony_drug_conviction",
    )
    _AOTC_PRIOR_YEARS_COLUMN = "american_opportunity_credit_claimed_prior_years"

    def _construct_aotc_eligibility_inputs(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert the PUF AOTC signal into person eligibility inputs.

        Mirrors the enhanced-CPS baseline
        ``ExtendedCPS._impute_aotc_eligibility_inputs``
        (``PolicyEngine/policyengine-us-data``, unmerged branch
        ``codex/fix-aotc-eligibility``).

        The enhanced CPS operates on a flat ``{variable: {period: array}}``
        payload keyed by ``person_tax_unit_id``; Microplex carries the same
        signals (``american_opportunity_credit``,
        ``qualified_tuition_expenses``, ``is_full_time_college_student``,
        ``is_tax_unit_dependent``) as columns on the person table keyed by
        ``tax_unit_id`` once ``_build_policyengine_tax_units`` has assigned
        authoritative tax units, so the per-tax-unit back-solve is the same
        algorithm applied to a single DataFrame.

        Driven by the PUF-imputed ``american_opportunity_credit`` (PUF
        ``E87521``; see ``data_sources/puf.py`` / ``manifests/puf.json``). For
        each tax unit with positive credit the enhanced-CPS rule applies: if
        any member already reports positive qualified tuition, every such
        member is marked an AOTC student and the reported tuition is left
        unchanged; otherwise a single student is selected by priority
        (full-time college student -> tax-unit dependent -> any member) and
        that student's qualified tuition is back-solved to the minimum amount
        reproducing the unit's credit under PolicyEngine-US. With no credit
        signal it falls back to the enhanced-CPS
        ``aotc_student = qualified_tuition_expenses > 0`` rule. The selected
        students receive the five factual eligibility flags as ``True``,
        ``has_completed_first_four_years_of_postsecondary_education`` and
        ``has_felony_drug_conviction`` as ``False`` (constants the enhanced
        CPS also hard-codes), and
        ``american_opportunity_credit_claimed_prior_years`` clamped to a
        maximum of 3. ``american_opportunity_credit`` is a PUF
        calculated-tax output (see ``microdata_roles.py``) and is not itself
        exported; PolicyEngine-US recomputes the credit from these inputs.
        """
        if persons is None or persons.empty:
            return persons
        if "tax_unit_id" not in persons.columns:
            return persons

        result = persons.copy()
        n = len(result)
        time_period = int(self.config.policyengine_dataset_year or 2024)

        person_tax_unit_ids = result["tax_unit_id"].to_numpy()
        tuition = (
            pd.to_numeric(
                result["qualified_tuition_expenses"],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(dtype=float, copy=True)
            if "qualified_tuition_expenses" in result.columns
            else np.zeros(n, dtype=float)
        )
        if "qualified_tuition_expenses" not in result.columns:
            # No tuition signal and no credit-derived tuition can be
            # back-solved, so there is no student population to mark.
            credit_present = "american_opportunity_credit" in result.columns
            if not credit_present:
                return persons

        credit = (
            pd.to_numeric(
                result["american_opportunity_credit"],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(dtype=float)
            if "american_opportunity_credit" in result.columns
            else None
        )
        full_time = (
            pd.to_numeric(result["is_full_time_college_student"], errors="coerce")
            .fillna(0)
            .astype(bool)
            .to_numpy()
            if "is_full_time_college_student" in result.columns
            else np.zeros(n, dtype=bool)
        )
        dependent = (
            pd.to_numeric(result["is_tax_unit_dependent"], errors="coerce")
            .fillna(0)
            .astype(bool)
            .to_numpy()
            if "is_tax_unit_dependent" in result.columns
            else np.zeros(n, dtype=bool)
        )

        aotc_student = np.zeros(n, dtype=bool)

        if credit is not None:
            positive_credit = credit > 0
            if not positive_credit.any():
                # No positive credit anywhere: nothing to construct. The
                # enhanced CPS returns early here without writing inputs.
                return persons

            # ``american_opportunity_credit`` rides on the person table as the
            # per-tax-unit value repeated across members; collapse to one
            # value per tax unit (the maximum guards against any per-member
            # zero-fill on non-filer rows).
            credit_by_tax_unit: dict[Any, float] = {}
            for tax_unit_id, member_credit in zip(person_tax_unit_ids, credit):
                prior = credit_by_tax_unit.get(tax_unit_id, 0.0)
                if member_credit > prior:
                    credit_by_tax_unit[tax_unit_id] = float(member_credit)

            positive_credit_units = [
                tax_unit_id
                for tax_unit_id, unit_credit in credit_by_tax_unit.items()
                if unit_credit > 0
            ]
            for tax_unit_id in positive_credit_units:
                member_indices = np.flatnonzero(person_tax_unit_ids == tax_unit_id)
                if member_indices.size == 0:
                    continue

                # eCPS rule: if any member already reports positive qualified
                # tuition, every such member is an AOTC student and the reported
                # tuition is left untouched (no back-solve, no rewrite).
                tuition_indices = member_indices[tuition[member_indices] > 0]
                if tuition_indices.size > 0:
                    aotc_student[tuition_indices] = True
                    continue

                # Otherwise select a single student by the eCPS priority
                # (full-time college student -> tax-unit dependent -> any
                # member) and back-solve the minimum qualified tuition that
                # reproduces the unit's credit under PolicyEngine-US.
                preferred = member_indices[full_time[member_indices]]
                if preferred.size == 0:
                    preferred = member_indices[dependent[member_indices]]
                if preferred.size == 0:
                    preferred = member_indices
                selected = preferred[0]
                aotc_student[selected] = True
                tuition[selected] = max(
                    tuition[selected],
                    qualifying_expenses_from_american_opportunity_credit(
                        credit_by_tax_unit[tax_unit_id],
                        time_period,
                    ),
                )
        else:
            aotc_student = tuition > 0
            if not aotc_student.any():
                return persons

        # Five factual eligibility flags -> True for selected students.
        for column in self._AOTC_TRUE_FLAG_COLUMNS:
            values = (
                result[column].fillna(False).astype(bool).to_numpy().copy()
                if column in result.columns
                else np.zeros(n, dtype=bool)
            )
            values[aotc_student] = True
            result[column] = values

        # has_completed_first_four_years / has_felony_drug_conviction -> False.
        for column in self._AOTC_FALSE_FLAG_COLUMNS:
            values = (
                result[column].fillna(False).astype(bool).to_numpy().copy()
                if column in result.columns
                else np.zeros(n, dtype=bool)
            )
            values[aotc_student] = False
            result[column] = values

        # Prior-year claims clamped to the 4-year (max 3 prior) AOTC limit.
        prior_years = (
            pd.to_numeric(result[self._AOTC_PRIOR_YEARS_COLUMN], errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
            .copy()
            if self._AOTC_PRIOR_YEARS_COLUMN in result.columns
            else np.zeros(n, dtype=np.int64)
        )
        prior_years[aotc_student] = np.minimum(prior_years[aotc_student], 3)
        result[self._AOTC_PRIOR_YEARS_COLUMN] = prior_years

        # Write the back-solved per-student tuition the credit implies, so the
        # exported ``qualified_tuition_expenses`` reproduces the PUF credit
        # under PolicyEngine-US (enhanced CPS does the same).
        if "qualified_tuition_expenses" in result.columns:
            result["qualified_tuition_expenses"] = tuition

        return result

    def export_policyengine_dataset(
        self,
        result: USMicroplexBuildResult,
        path: str | Path,
        *,
        period: int | None = None,
        direct_override_variables: tuple[str, ...] | None = None,
    ) -> Path:
        """Export a build result as a PolicyEngine-readable HDF5 dataset."""
        export_period = (
            period
            or self.config.policyengine_dataset_year
            or result.config.policyengine_dataset_year
            or 2024
        )
        export_direct_override_variables = (
            direct_override_variables
            if direct_override_variables is not None
            else (
                self.config.policyengine_direct_override_variables
                or result.config.policyengine_direct_override_variables
            )
        )
        tables = result.policyengine_tables or self.build_policyengine_entity_tables(
            result.calibrated_data
        )
        tax_benefit_system = self._resolve_policyengine_tax_benefit_system()
        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=tax_benefit_system,
            direct_override_variables=export_direct_override_variables,
        )
        excluded_variables = resolve_policyengine_excluded_export_variables(
            tax_benefit_system,
            sorted(
                {
                    target
                    for variable_map in export_maps.values()
                    for target in variable_map.values()
                }
            ),
            direct_override_variables=export_direct_override_variables,
        )
        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=export_period,
            household_variable_map=export_maps["household"],
            person_variable_map=export_maps["person"],
            tax_unit_variable_map=export_maps["tax_unit"],
            spm_unit_variable_map=export_maps["spm_unit"],
            family_variable_map=export_maps["family"],
        )
        return write_policyengine_us_time_period_dataset(
            arrays,
            path,
            excluded_variables=excluded_variables,
            tax_benefit_system=tax_benefit_system,
        )

    def _fit_synthesizer(
        self,
        seed_data: pd.DataFrame,
        synthesis_variables: USMicroplexSynthesisVariables,
    ) -> Synthesizer:
        """Fit a microplex synthesizer on the seed data."""
        condition_vars = list(synthesis_variables.condition_vars)
        target_vars = list(synthesis_variables.target_vars)
        if not target_vars:
            raise ValueError(
                "USMicroplexPipeline requires at least one observed target variable"
            )

        synthesizer = Synthesizer(
            target_vars=target_vars,
            condition_vars=condition_vars,
            n_layers=self.config.synthesizer_n_layers,
            hidden_dim=self.config.synthesizer_hidden_dim,
        )
        synthesizer.fit(
            seed_data[condition_vars + target_vars + ["hh_weight"]].rename(
                columns={"hh_weight": "weight"}
            ),
            weight_col="weight",
            epochs=self.config.synthesizer_epochs,
            batch_size=self.config.synthesizer_batch_size,
            learning_rate=self.config.synthesizer_learning_rate,
            verbose=False,
        )
        return synthesizer

    def _build_donor_imputer(
        self,
        *,
        condition_vars: list[str],
        target_vars: tuple[str, ...],
    ) -> Synthesizer | ColumnwiseQRFDonorImputer:
        backend = self.config.donor_imputer_backend
        if backend == "maf":
            return Synthesizer(
                target_vars=list(target_vars),
                condition_vars=condition_vars,
                n_layers=self.config.donor_imputer_n_layers,
                hidden_dim=self.config.donor_imputer_hidden_dim,
            )

        support_families = {
            variable: variable_semantic_spec_for(variable).support_family
            for variable in target_vars
        }
        nonnegative_vars = {
            variable
            for variable, support_family in support_families.items()
            if support_family is VariableSupportFamily.BOUNDED_SHARE
        }
        if backend == "regime_aware":
            return RegimeAwareDonorImputer(
                condition_vars=condition_vars,
                target_vars=list(target_vars),
                n_estimators=self.config.donor_imputer_qrf_n_estimators,
                seed=self.config.random_seed,
            )
        zero_inflated_vars = (
            {
                variable
                for variable, support_family in support_families.items()
                if support_family
                in {
                    VariableSupportFamily.SUPPORT_SENSITIVE,
                }
            }
            if backend == "zi_qrf"
            else set()
        )
        return ColumnwiseQRFDonorImputer(
            condition_vars=condition_vars,
            target_vars=list(target_vars),
            n_estimators=self.config.donor_imputer_qrf_n_estimators,
            zero_inflated_vars=zero_inflated_vars,
            nonnegative_vars=nonnegative_vars,
            zero_threshold=self.config.donor_imputer_qrf_zero_threshold,
        )

    def _resolve_synthesis_variables(
        self,
        source_input: USMicroplexSourceInput,
        *,
        fusion_plan: FusionPlan | None = None,
        include_all_observed_targets: bool = False,
        available_columns: set[str] | None = None,
        observed_frame: pd.DataFrame | None = None,
    ) -> USMicroplexSynthesisVariables:
        """Select the observed variables to feed into synthesis."""
        active_plan = fusion_plan or source_input.fusion_plan
        available_variables = prune_redundant_variables(
            active_plan.variables_for(EntityType.HOUSEHOLD)
            | active_plan.variables_for(EntityType.PERSON)
        )
        if available_columns is not None:
            available_variables = available_variables & available_columns
        condition_vars = self._resolve_synthesis_condition_vars(
            available_variables,
            observed_frame=observed_frame,
        )
        configured_targets = [
            variable
            for variable in self.config.synthesizer_target_vars
            if variable in available_variables and variable not in condition_vars
        ]
        configured_targets.extend(
            variable
            for variable in STATE_PROGRAM_SUPPORT_PROXY_VARIABLES
            if variable in available_variables
            and variable not in condition_vars
            and variable not in configured_targets
        )
        extra_targets: list[str] = []
        if include_all_observed_targets:
            excluded = {
                "person_id",
                "household_id",
                "hh_weight",
                "weight",
                "state",
                "age_group",
                "income_bracket",
            }
            extra_targets = sorted(
                variable
                for variable in available_variables
                if variable not in excluded
                and variable not in condition_vars
                and variable not in configured_targets
            )
        return USMicroplexSynthesisVariables(
            condition_vars=condition_vars,
            target_vars=tuple(configured_targets + extra_targets),
        )

    def _resolve_synthesis_condition_vars(
        self,
        available_columns: Iterable[str],
        *,
        observed_frame: pd.DataFrame | None = None,
    ) -> tuple[str, ...]:
        available = set(available_columns)
        ordered = list(self.config.synthesizer_condition_vars)
        for variable in STATE_PROGRAM_AUTO_CONDITION_VARIABLES:
            if (
                variable in available
                and variable not in ordered
                and (
                    observed_frame is None
                    or self._is_informative_state_program_proxy(
                        observed_frame,
                        variable,
                    )
                )
            ):
                ordered.append(variable)
        return tuple(variable for variable in ordered if variable in available)

    def _is_informative_state_program_proxy(
        self,
        frame: pd.DataFrame,
        variable: str,
    ) -> bool:
        if variable not in frame.columns:
            return False
        series = pd.to_numeric(frame[variable], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        series = series.dropna()
        if series.empty:
            return False
        return bool(series.nunique(dropna=True) > 1)

    def _select_scaffold_source(
        self,
        source_inputs: list[USMicroplexSourceInput],
    ) -> USMicroplexSourceInput:
        candidates = [
            source
            for source in source_inputs
            if source.household_observation is not None
            and source.household_person_relationship is not None
        ]
        if not candidates:
            raise ValueError(
                "USMicroplexPipeline requires at least one structured source with household and person observations"
            )

        def score(source: USMicroplexSourceInput) -> tuple[int, int, int, int]:
            public_score = int(source.frame.source.shareability == Shareability.PUBLIC)
            geography_score = self._household_geography_coverage(source)
            observed_variables = source.fusion_plan.variables_for(
                EntityType.HOUSEHOLD
            ) | source.fusion_plan.variables_for(EntityType.PERSON)
            support_proxy_score = sum(
                variable in observed_variables
                for variable in STATE_PROGRAM_SUPPORT_PROXY_VARIABLES
            )
            observed_vars = len(observed_variables)
            household_rows = (
                len(source.households) if source.households is not None else 0
            )
            return (
                public_score,
                geography_score,
                support_proxy_score,
                observed_vars,
                household_rows,
            )

        if self.config.puf_support_clone_enabled:
            cps_candidates = [
                source
                for source in candidates
                if self._is_cps_asec_scaffold_source(source.frame.source.name)
            ]
            if cps_candidates:
                return max(cps_candidates, key=score)

        return max(candidates, key=score)

    def _household_geography_coverage(
        self,
        source: USMicroplexSourceInput,
    ) -> int:
        households = source.households
        if households is None or "state_fips" not in households.columns:
            return 0
        state_fips = pd.to_numeric(households["state_fips"], errors="coerce").fillna(0)
        return int((state_fips > 0).sum())

    def _is_puf_support_clone_source(self, source_name: str) -> bool:
        return any(
            source_name.startswith(prefix)
            for prefix in self.config.puf_support_clone_source_prefixes
        )

    def _is_cps_asec_scaffold_source(self, source_name: str) -> bool:
        return source_name.startswith(("cps", "cps_asec"))

    def _ordered_donor_inputs_for_puf_support_clone(
        self,
        *,
        scaffold_input: USMicroplexSourceInput,
        donor_inputs: list[USMicroplexSourceInput],
    ) -> tuple[list[USMicroplexSourceInput], list[str], list[str]]:
        """Return PUF-first donor inputs and clone source order for clone mode."""
        input_order = [donor.frame.source.name for donor in donor_inputs]
        if not self.config.puf_support_clone_enabled:
            return donor_inputs, input_order, []

        scaffold_name = scaffold_input.frame.source.name
        if self._is_puf_support_clone_source(scaffold_name):
            raise ValueError(
                "puf_support_clone_enabled requires the PUF source to be a donor, "
                f"but selected scaffold source is {scaffold_name!r}"
            )
        if not self._is_cps_asec_scaffold_source(scaffold_name):
            raise ValueError(
                "puf_support_clone_enabled requires a CPS/ASEC-shaped scaffold; "
                f"selected scaffold source is {scaffold_name!r}"
            )

        puf_donors = [
            donor
            for donor in donor_inputs
            if self._is_puf_support_clone_source(donor.frame.source.name)
        ]
        if not puf_donors:
            raise ValueError(
                "puf_support_clone_enabled requires exactly one PUF donor source, "
                "but none matched puf_support_clone_source_prefixes"
            )
        if len(puf_donors) > 1:
            raise ValueError(
                "puf_support_clone_enabled requires an unambiguous PUF donor source; "
                "matched " + ", ".join(donor.frame.source.name for donor in puf_donors)
            )

        non_puf_donors = [
            donor
            for donor in donor_inputs
            if not self._is_puf_support_clone_source(donor.frame.source.name)
        ]
        ordered = [*puf_donors, *non_puf_donors]
        return (
            ordered,
            [donor.frame.source.name for donor in ordered],
            [donor.frame.source.name for donor in puf_donors],
        )

    def _prepare_puf_support_clone_frame(self, original: pd.DataFrame) -> pd.DataFrame:
        """Create a zero-stored-weight PUF clone frame from CPS support rows."""
        clone = original.copy()
        structural_id_columns = {"person_id", *ENTITY_ID_COLUMNS.values()}
        for column in sorted(structural_id_columns & set(clone.columns)):
            series = clone[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                finite = numeric[np.isfinite(numeric)]
                offset = int(finite.max()) + 1 if not finite.empty else len(clone)
                clone[column] = numeric.fillna(-1).astype(np.int64) + int(offset)
            else:
                clone[column] = series.astype(str) + "__puf_clone"
        if self.config.puf_support_clone_zero_initial_weight:
            for column in clone.columns:
                if column == "weight" or "_weight" in column:
                    clone[column] = 0.0
        clone[self.config.puf_support_clone_flag_column] = 1.0
        return clone

    def _refresh_puf_support_clone_cps_only_fields(
        self,
        *,
        original: pd.DataFrame,
        clone: pd.DataFrame,
        integrated_variables: Iterable[str],
        preclone_columns: set[str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Refresh copied CPS-only clone fields after PUF income is grafted on.

        PUF support clones start as literal CPS copies, then receive PUF tax and
        income fields. Any remaining copied CPS-only fields can become
        incoherent with the clone's new income surface. Re-match those fields
        from CPS donors using demographic predictors plus PUF-imputed income.
        """
        summary: dict[str, Any] = {
            "enabled": bool(self.config.puf_support_clone_refresh_cps_only_fields),
            "condition_variables": [],
            "refreshed_variables": [],
            "social_security_reconciled_variables": [],
            "matched_source_row_count": 0,
        }
        if not self.config.puf_support_clone_refresh_cps_only_fields:
            return clone, summary
        if original.empty or clone.empty:
            return clone, summary

        integrated_set = set(integrated_variables)
        condition_vars = [
            variable
            for variable in self.config.puf_support_clone_cps_refresh_condition_variables
            if variable in original.columns
            and variable in clone.columns
            and pd.api.types.is_numeric_dtype(original[variable])
            and pd.api.types.is_numeric_dtype(clone[variable])
            and self._is_compatible_donor_condition(
                clone[variable],
                original[variable],
            )
        ]
        if not condition_vars:
            return clone, summary

        refresh_variables = [
            variable
            for variable in self.config.puf_support_clone_cps_refresh_variables
            if variable in preclone_columns
            and variable not in integrated_set
            and variable in original.columns
            and variable in clone.columns
        ]
        if not refresh_variables:
            return clone, summary

        train = original.loc[:, condition_vars].apply(
            lambda series: pd.to_numeric(series, errors="coerce").fillna(0.0)
        )
        test = clone.loc[:, condition_vars].apply(
            lambda series: pd.to_numeric(series, errors="coerce").fillna(0.0)
        )
        for variable in (
            set(condition_vars) & PUF_SUPPORT_CLONE_CPS_REFRESH_INCOME_VARIABLES
        ):
            train[variable] = np.arcsinh(train[variable])
            test[variable] = np.arcsinh(test[variable])
        scale = train.std(ddof=0).replace(0.0, 1.0)
        center = train.mean()
        train_values = ((train - center) / scale).to_numpy(dtype=float)
        test_values = ((test - center) / scale).to_numpy(dtype=float)

        from sklearn.neighbors import NearestNeighbors

        matcher = NearestNeighbors(n_neighbors=1)
        matcher.fit(train_values)
        matched = matcher.kneighbors(test_values, return_distance=False).reshape(-1)

        refreshed = clone.copy()
        for variable in refresh_variables:
            refreshed[variable] = original[variable].to_numpy(copy=True)[matched]

        reconciled_variables = self._reconcile_puf_support_clone_social_security(
            refreshed
        )
        summary["condition_variables"] = condition_vars
        summary["refreshed_variables"] = refresh_variables
        summary["social_security_reconciled_variables"] = reconciled_variables
        summary["matched_source_row_count"] = int(np.unique(matched).size)
        return refreshed, summary

    def _reconcile_puf_support_clone_social_security(
        self,
        clone: pd.DataFrame,
    ) -> list[str]:
        """Scale cloned Social Security components to the PUF-imputed total."""
        if "social_security" not in clone.columns:
            return []
        subcomponents = [
            variable
            for variable in (
                "social_security_retirement",
                "social_security_disability",
                "social_security_survivors",
                "social_security_dependents",
            )
            if variable in clone.columns
        ]
        if not subcomponents:
            return []

        total = pd.to_numeric(clone["social_security"], errors="coerce").fillna(0.0)
        sub_values = {
            variable: pd.to_numeric(clone[variable], errors="coerce").fillna(0.0)
            for variable in subcomponents
        }
        sub_sum = sum(sub_values.values())
        positive_total = total.gt(0.0)
        positive_sub_sum = sub_sum.gt(0.0)
        scale_mask = positive_total & positive_sub_sum
        zero_mask = ~positive_total

        for variable, values in sub_values.items():
            adjusted = values.copy()
            adjusted.loc[zero_mask] = 0.0
            adjusted.loc[scale_mask] = (
                values.loc[scale_mask] * total.loc[scale_mask] / sub_sum.loc[scale_mask]
            )
            clone[variable] = adjusted

        fallback_mask = positive_total & ~positive_sub_sum
        if fallback_mask.any():
            age = pd.to_numeric(clone.get("age", 0.0), errors="coerce").fillna(0.0)
            if "social_security_retirement" in subcomponents:
                clone.loc[
                    fallback_mask & age.ge(62),
                    "social_security_retirement",
                ] = total.loc[fallback_mask & age.ge(62)]
            if "social_security_disability" in subcomponents:
                clone.loc[
                    fallback_mask & age.lt(62),
                    "social_security_disability",
                ] = total.loc[fallback_mask & age.lt(62)]
        return subcomponents

    def _finalize_puf_support_clone_frame(
        self,
        *,
        original: pd.DataFrame,
        imputed_clone: pd.DataFrame,
        donor_source_name: str,
        integrated_variables: list[str],
        preclone_columns: set[str],
        donor_seed_columns: set[str],
        donor_observed: set[str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Concatenate original CPS support and its PUF-imputed support clone."""
        flag_column = self.config.puf_support_clone_flag_column
        original = original.copy()
        clone = imputed_clone.copy()
        original[flag_column] = 0.0
        clone[flag_column] = 1.0

        integrated_set = set(integrated_variables)
        both_halves_override = (
            integrated_set
            & set(self.config.puf_support_clone_both_halves_override_variables)
            & preclone_columns
        )
        for variable in sorted(both_halves_override):
            if variable in original.columns and variable in clone.columns:
                original[variable] = clone[variable].to_numpy(copy=True)

        clone, cps_refresh_summary = self._refresh_puf_support_clone_cps_only_fields(
            original=original,
            clone=clone,
            integrated_variables=integrated_variables,
            preclone_columns=preclone_columns,
        )

        generated_entity_id_columns = sorted(
            set(ENTITY_ID_COLUMNS.values()) & (set(clone.columns) - preclone_columns)
        )
        if generated_entity_id_columns:
            clone = clone.drop(columns=generated_entity_id_columns)

        for column in sorted(set(clone.columns) - set(original.columns)):
            original[column] = 0.0
        for column in sorted(set(original.columns) - set(clone.columns)):
            clone[column] = original[column].to_numpy(copy=True)
        original = original.loc[:, clone.columns]

        combined = pd.concat([original, clone], ignore_index=True, sort=False)
        combined = combined.reset_index(drop=True)

        overlap_variables = sorted(integrated_set & preclone_columns)
        donor_only_variables = sorted(integrated_set - preclone_columns)
        ecps_surface = (
            set(PUF_SUPPORT_CLONE_IMPUTED_VARIABLES)
            | set(PUF_SUPPORT_CLONE_OVERRIDDEN_VARIABLES)
            | set(PUF_SUPPORT_CLONE_SPECIAL_VARIABLES)
        )
        included_surface = sorted(ecps_surface & integrated_set)
        excluded_surface: dict[str, str] = {}
        for variable in sorted(ecps_surface - set(included_surface)):
            if variable not in donor_observed and variable not in donor_seed_columns:
                reason = "missing_puf_source_column"
            elif variable in self.config.donor_imputer_excluded_variables:
                reason = "excluded_by_config"
            elif variable not in preclone_columns:
                reason = "not_present_before_clone"
            else:
                reason = "not_selected_for_imputation"
            excluded_surface[variable] = reason

        clone_weight_sum = 0.0
        for column in ("household_weight", "hh_weight", "weight"):
            if column in clone.columns:
                clone_weight_sum = float(
                    pd.to_numeric(clone[column], errors="coerce").fillna(0.0).sum()
                )
                break

        summary = {
            "enabled": True,
            "donor_source_name": donor_source_name,
            "original_row_count": int(len(original)),
            "clone_row_count": int(len(clone)),
            "final_row_count": int(len(combined)),
            "clone_initial_weight_sum": clone_weight_sum,
            "integrated_variable_count": int(len(integrated_set)),
            "clone_overlap_variable_count": int(len(overlap_variables)),
            "clone_donor_only_variable_count": int(len(donor_only_variables)),
            "overlap_variables": overlap_variables,
            "donor_only_variables": donor_only_variables,
            "both_halves_override_variables": sorted(both_halves_override),
            "cps_only_refresh": cps_refresh_summary,
            "dropped_generated_entity_id_columns": generated_entity_id_columns,
            "variable_surface": {
                "ecps_imputed_variables": list(PUF_SUPPORT_CLONE_IMPUTED_VARIABLES),
                "ecps_overridden_variables": list(
                    PUF_SUPPORT_CLONE_OVERRIDDEN_VARIABLES
                ),
                "ecps_special_variables": list(PUF_SUPPORT_CLONE_SPECIAL_VARIABLES),
                "included_variables": included_surface,
                "excluded_variables": excluded_surface,
            },
        }
        return combined, summary

    def _integrate_donor_sources(
        self,
        seed_data: pd.DataFrame,
        *,
        scaffold_input: USMicroplexSourceInput,
        donor_inputs: list[USMicroplexSourceInput],
    ) -> dict[str, Any]:
        current = seed_data.copy()
        integrated_variables: list[str] = []
        conditioning_diagnostics: list[dict[str, Any]] = []
        donor_inputs, processed_donor_source_order, puf_clone_source_order = (
            self._ordered_donor_inputs_for_puf_support_clone(
                scaffold_input=scaffold_input,
                donor_inputs=donor_inputs,
            )
        )
        puf_support_clone_summary: dict[str, Any] | None = None
        scaffold_observed = prune_redundant_variables(
            scaffold_input.fusion_plan.variables_for(EntityType.HOUSEHOLD)
            | scaffold_input.fusion_plan.variables_for(EntityType.PERSON)
        )
        excluded = {
            "person_id",
            "household_id",
            "hh_weight",
            "weight",
            "household_weight",
            "tax_unit_id",
            "family_id",
            "spm_unit_id",
            "marital_unit_id",
            "state",
            "age_group",
            "income_bracket",
            "is_head",
            "is_spouse",
            "is_dependent",
        }
        rng = np.random.default_rng(self.config.random_seed)
        _emit_us_pipeline_progress(
            "US microplex donor integration: start",
            donor_sources=len(donor_inputs),
            seed_rows=len(current),
            condition_selection=self.config.donor_imputer_condition_selection,
            puf_support_clone_enabled=self.config.puf_support_clone_enabled,
        )

        for donor_input in donor_inputs:
            donor_source_name = donor_input.frame.source.name
            is_puf_support_clone_source = (
                self.config.puf_support_clone_enabled
                and self._is_puf_support_clone_source(donor_source_name)
            )
            source_original_current: pd.DataFrame | None = None
            source_preclone_columns: set[str] = set(current.columns)
            source_integrated_variables: list[str] = []
            if is_puf_support_clone_source:
                source_original_current = current.copy()
                current = self._prepare_puf_support_clone_frame(source_original_current)
                _emit_us_pipeline_progress(
                    "US microplex donor integration: puf support clone prepared",
                    donor_source=donor_source_name,
                    original_rows=len(source_original_current),
                    clone_rows=len(current),
                )
            _emit_us_pipeline_progress(
                "US microplex donor integration: source start",
                donor_source=donor_source_name,
                current_rows=len(current),
            )
            donor_seed = self.prepare_seed_data_from_source(donor_input)
            donor_observed = prune_redundant_variables(
                donor_input.fusion_plan.variables_for(EntityType.HOUSEHOLD)
                | donor_input.fusion_plan.variables_for(EntityType.PERSON)
            )
            numeric_current = {
                column
                for column in current.columns
                if pd.api.types.is_numeric_dtype(current[column])
            }
            numeric_donor = {
                column
                for column in donor_seed.columns
                if pd.api.types.is_numeric_dtype(donor_seed[column])
            }
            shared_vars = sorted(
                variable
                for variable in scaffold_observed & donor_observed
                if variable not in excluded
                and variable in current.columns
                and variable in donor_seed.columns
                and variable in numeric_current
                and variable in numeric_donor
                and scaffold_input.frame.source.allows_conditioning_on(variable)
                and donor_input.frame.source.allows_conditioning_on(variable)
                and self._is_compatible_donor_condition(
                    current[variable],
                    donor_seed[variable],
                )
            )
            raw_shared_var_set = set(shared_vars)
            donor_only_vars = sorted(
                variable
                for variable in donor_observed - scaffold_observed
                if variable not in excluded
                and variable not in self.config.donor_imputer_excluded_variables
                and variable in donor_seed.columns
                and variable in numeric_donor
                and donor_input.frame.source.is_authoritative_for(variable)
                and self._should_integrate_donor_variable(current, variable)
                and self._is_compatible_donor_target(donor_seed[variable])
            )
            donor_override_vars = sorted(
                variable
                for variable in scaffold_observed & donor_observed
                if variable not in excluded
                and variable not in self.config.donor_imputer_excluded_variables
                and variable
                in self.config.donor_imputer_authoritative_override_variables
                and variable in current.columns
                and variable in donor_seed.columns
                and variable in numeric_current
                and variable in numeric_donor
                and donor_input.frame.source.is_authoritative_for(variable)
                and self._is_compatible_donor_target(donor_seed[variable])
            )
            if is_puf_support_clone_source:
                puf_clone_overlap_vars = sorted(
                    variable
                    for variable in set(self.config.puf_support_clone_overlap_variables)
                    if variable not in excluded
                    and variable not in self.config.donor_imputer_excluded_variables
                    and variable in scaffold_observed
                    and variable in donor_observed
                    and variable in current.columns
                    and variable in donor_seed.columns
                    and variable in numeric_current
                    and variable in numeric_donor
                    and donor_input.frame.source.is_authoritative_for(variable)
                    and self._is_compatible_donor_target(donor_seed[variable])
                )
                donor_override_vars = sorted(
                    set(donor_override_vars) | set(puf_clone_overlap_vars)
                )
            donor_target_vars = sorted(set(donor_only_vars) | set(donor_override_vars))
            if not shared_vars or not donor_target_vars:
                if is_puf_support_clone_source:
                    raise ValueError(
                        "PUF support clone donor produced no imputation targets; "
                        f"shared_vars={len(shared_vars)}, "
                        f"donor_target_vars={len(donor_target_vars)}"
                    )
                _emit_us_pipeline_progress(
                    "US microplex donor integration: source skipped",
                    donor_source=donor_source_name,
                    donor_rows=len(donor_seed),
                    shared_vars=len(shared_vars),
                    donor_target_vars=len(donor_target_vars),
                )
                continue

            donor_block_specs = donor_imputation_block_specs(donor_target_vars)
            _emit_us_pipeline_progress(
                "US microplex donor integration: source ready",
                donor_source=donor_source_name,
                donor_rows=len(donor_seed),
                shared_vars=len(shared_vars),
                donor_target_vars=len(donor_target_vars),
                blocks=len(donor_block_specs),
            )
            required_entities = {
                donor_block_spec.native_entity
                for donor_block_spec in donor_block_specs
                if donor_block_spec.native_entity is not EntityType.PERSON
            }
            if required_entities:
                _emit_us_pipeline_progress(
                    "US microplex donor integration: entity ids required",
                    donor_source=donor_source_name,
                    entities=_format_progress_values(
                        sorted(entity.value for entity in required_entities)
                    ),
                    current_rows=len(current),
                    donor_rows=len(donor_seed),
                )
                current = self._ensure_seed_entity_ids(
                    current,
                    entities=required_entities,
                    frame_role="current",
                    donor_source_name=donor_source_name,
                )
                donor_seed = self._ensure_seed_entity_ids(
                    donor_seed,
                    entities=required_entities,
                    frame_role="donor",
                    donor_source_name=donor_source_name,
                )

            for donor_block_spec in donor_block_specs:
                block_label = _format_progress_values(
                    donor_block_spec.model_variables,
                    limit=4,
                )
                _emit_us_pipeline_progress(
                    "US microplex donor integration: block start",
                    donor_source=donor_source_name,
                    block=block_label,
                    restored=_format_progress_values(
                        donor_block_spec.restored_variables,
                        limit=4,
                    ),
                )
                prepared_inputs = PE_SOURCE_IMPUTE_BLOCK_ENGINE.prepare_block_inputs(
                    donor_seed=donor_seed,
                    current_frame=current,
                    shared_vars=shared_vars,
                    donor_block_spec=donor_block_spec,
                    donor_source_name=donor_source_name,
                    prepare_pe_surface=(self._uses_pe_condition_surface()),
                    can_project_to_entity=self._can_project_donor_block_to_entity,
                    project_frame_to_entity=self._project_frame_to_entity,
                    entity_key_fn=self._entity_key_column,
                )
                shared_vars_for_block = list(prepared_inputs.shared_vars_for_block)
                donor_fit_source = prepared_inputs.donor_fit_source
                current_generation_source = prepared_inputs.current_generation_source
                entity_key = prepared_inputs.entity_key
                donor_condition_source = donor_fit_source
                current_condition_source = current_generation_source
                requested_supplemental_vars = (
                    self._resolve_requested_supplemental_shared_condition_vars(
                        donor_block_spec.model_variables
                    )
                )
                requested_challenger_vars = (
                    self._resolve_requested_challenger_shared_condition_vars(
                        donor_block_spec.model_variables,
                        donor_source_name=donor_source_name,
                    )
                )
                if prepared_inputs.condition_surface is not None:
                    surface = prepared_inputs.condition_surface
                    if (
                        self.config.donor_imputer_condition_selection
                        == "pe_plus_puf_native_challenger"
                    ):
                        donor_condition_source = surface.donor_frame.copy()
                        current_condition_source = surface.current_frame.copy()
                        challenger_condition_vars = (
                            self._resolve_challenger_shared_condition_vars(
                                donor_frame=donor_fit_source,
                                current_frame=current_generation_source,
                                shared_vars=shared_vars_for_block,
                                donor_block=donor_block_spec.model_variables,
                                donor_source_name=donor_source_name,
                            )
                        )
                        for variable in challenger_condition_vars:
                            donor_condition_source[variable] = donor_fit_source[
                                variable
                            ]
                            current_condition_source[variable] = (
                                current_generation_source[variable]
                            )
                        donor_condition_vars = list(
                            dict.fromkeys(
                                surface.compatible_predictors(
                                    compatibility_fn=self._is_compatible_donor_condition,
                                )
                                + challenger_condition_vars
                            )
                        )
                        _emit_us_pipeline_progress(
                            "US microplex donor integration: block run",
                            donor_source=donor_source_name,
                            block=block_label,
                            condition_vars=len(donor_condition_vars),
                            donor_rows=len(donor_fit_source),
                            current_rows=len(current_generation_source),
                        )
                        result = PE_SOURCE_IMPUTE_BLOCK_ENGINE.run_conditioned_block(
                            request=PESourceImputeConditionedBlockRunRequest(
                                block_request=PESourceImputeBlockRunRequest(
                                    donor_block_spec=donor_block_spec,
                                    donor_fit_source=donor_fit_source,
                                    current_generation_source=current_generation_source,
                                    current_frame=current,
                                    entity_key=entity_key,
                                ),
                                donor_condition_source=donor_condition_source,
                                current_condition_source=current_condition_source,
                                condition_vars=tuple(donor_condition_vars),
                            ),
                            build_imputer=self._build_donor_imputer,
                            rank_match=self._rank_match_donor_values,
                            fit_kwargs={
                                "epochs": self.config.donor_imputer_epochs,
                                "batch_size": self.config.donor_imputer_batch_size,
                                "learning_rate": self.config.donor_imputer_learning_rate,
                                "verbose": False,
                            },
                            seed=self.config.random_seed,
                            rng=rng,
                        )
                    else:
                        donor_condition_source = surface.donor_frame
                        current_condition_source = surface.current_frame
                        compatible_predictors = surface.compatible_predictors(
                            compatibility_fn=self._is_compatible_donor_condition,
                        )
                        _emit_us_pipeline_progress(
                            "US microplex donor integration: block run",
                            donor_source=donor_source_name,
                            block=block_label,
                            condition_vars=len(compatible_predictors),
                            donor_rows=len(donor_fit_source),
                            current_rows=len(current_generation_source),
                        )
                        result = PE_SOURCE_IMPUTE_BLOCK_ENGINE.run_prepared_block(
                            surface=surface,
                            request=PESourceImputeBlockRunRequest(
                                donor_block_spec=donor_block_spec,
                                donor_fit_source=donor_fit_source,
                                current_generation_source=current_generation_source,
                                current_frame=current,
                                entity_key=entity_key,
                            ),
                            build_imputer=self._build_donor_imputer,
                            rank_match=self._rank_match_donor_values,
                            compatibility_fn=self._is_compatible_donor_condition,
                            fit_kwargs={
                                "epochs": self.config.donor_imputer_epochs,
                                "batch_size": self.config.donor_imputer_batch_size,
                                "learning_rate": self.config.donor_imputer_learning_rate,
                                "verbose": False,
                            },
                            seed=self.config.random_seed,
                            rng=rng,
                        )
                    if result is not None:
                        selected_condition_vars = list(result.condition_vars)
                        conditioning_diagnostics.append(
                            {
                                "donor_source": donor_input.frame.source.name,
                                "model_variables": list(
                                    donor_block_spec.model_variables
                                ),
                                "restored_variables": list(
                                    donor_block_spec.restored_variables
                                ),
                                "condition_selection": (
                                    self.config.donor_imputer_condition_selection
                                ),
                                "used_condition_surface": True,
                                "raw_shared_vars": list(
                                    prepared_inputs.raw_shared_vars
                                ),
                                "shared_vars_after_model_exclusion": list(
                                    prepared_inputs.shared_vars_after_model_exclusion
                                ),
                                "projection_applied": (
                                    prepared_inputs.projection_applied
                                ),
                                "entity_compatible_shared_vars": list(
                                    prepared_inputs.entity_compatible_shared_vars
                                ),
                                "shared_vars_for_block": list(shared_vars_for_block),
                                "selected_condition_vars": selected_condition_vars,
                                "dropped_shared_vars": [
                                    variable
                                    for variable in shared_vars_for_block
                                    if variable not in selected_condition_vars
                                ],
                                "requested_supplemental_shared_condition_vars": (
                                    requested_supplemental_vars
                                ),
                                "requested_challenger_shared_condition_vars": (
                                    requested_challenger_vars
                                ),
                                "raw_supplemental_shared_condition_var_status": (
                                    self._summarize_requested_raw_condition_var_status(
                                        donor_frame=donor_seed,
                                        current_frame=current,
                                        scaffold_source=scaffold_input.frame.source,
                                        donor_source=donor_input.frame.source,
                                        numeric_current=numeric_current,
                                        numeric_donor=numeric_donor,
                                        shared_var_set=raw_shared_var_set,
                                        excluded=excluded,
                                        requested_vars=requested_supplemental_vars,
                                    )
                                ),
                                "raw_challenger_shared_condition_var_status": (
                                    self._summarize_requested_raw_condition_var_status(
                                        donor_frame=donor_seed,
                                        current_frame=current,
                                        scaffold_source=scaffold_input.frame.source,
                                        donor_source=donor_input.frame.source,
                                        numeric_current=numeric_current,
                                        numeric_donor=numeric_donor,
                                        shared_var_set=raw_shared_var_set,
                                        excluded=excluded,
                                        requested_vars=requested_challenger_vars,
                                    )
                                ),
                                "supplemental_shared_condition_var_status": (
                                    self._summarize_requested_condition_var_status(
                                        donor_frame=donor_condition_source,
                                        current_frame=current_condition_source,
                                        shared_vars=shared_vars_for_block,
                                        selected_condition_vars=selected_condition_vars,
                                        requested_vars=requested_supplemental_vars,
                                    )
                                ),
                                "challenger_shared_condition_var_status": (
                                    self._summarize_requested_condition_var_status(
                                        donor_frame=donor_condition_source,
                                        current_frame=current_condition_source,
                                        shared_vars=shared_vars_for_block,
                                        selected_condition_vars=selected_condition_vars,
                                        requested_vars=requested_challenger_vars,
                                    )
                                ),
                            }
                        )
                        current = result.updated_frame
                        integrated_variables.extend(result.integrated_variables)
                        source_integrated_variables.extend(result.integrated_variables)
                        _emit_us_pipeline_progress(
                            "US microplex donor integration: block complete",
                            donor_source=donor_source_name,
                            block=block_label,
                            integrated_vars=len(result.integrated_variables),
                        )
                    continue
                donor_condition_source = (
                    self._augment_donor_condition_frame_for_targets(
                        donor_condition_source,
                        donor_block_spec.model_variables,
                    )
                )
                current_condition_source = (
                    self._augment_donor_condition_frame_for_targets(
                        current_condition_source,
                        donor_block_spec.model_variables,
                    )
                )
                donor_condition_vars = self._select_donor_condition_vars(
                    donor_condition_source,
                    current_condition_source,
                    shared_vars_for_block,
                    donor_block_spec.model_variables,
                    donor_source_name=donor_source_name,
                )
                if not donor_condition_vars:
                    _emit_us_pipeline_progress(
                        "US microplex donor integration: block skipped",
                        donor_source=donor_source_name,
                        block=block_label,
                        reason="no_condition_vars",
                    )
                    continue

                _emit_us_pipeline_progress(
                    "US microplex donor integration: block run",
                    donor_source=donor_source_name,
                    block=block_label,
                    condition_vars=len(donor_condition_vars),
                    donor_rows=len(donor_fit_source),
                    current_rows=len(current_generation_source),
                )
                result = PE_SOURCE_IMPUTE_BLOCK_ENGINE.run_conditioned_block(
                    request=PESourceImputeConditionedBlockRunRequest(
                        block_request=PESourceImputeBlockRunRequest(
                            donor_block_spec=donor_block_spec,
                            donor_fit_source=donor_fit_source,
                            current_generation_source=current_generation_source,
                            current_frame=current,
                            entity_key=entity_key,
                        ),
                        donor_condition_source=donor_condition_source,
                        current_condition_source=current_condition_source,
                        condition_vars=tuple(donor_condition_vars),
                    ),
                    build_imputer=self._build_donor_imputer,
                    rank_match=self._rank_match_donor_values,
                    fit_kwargs={
                        "epochs": self.config.donor_imputer_epochs,
                        "batch_size": self.config.donor_imputer_batch_size,
                        "learning_rate": self.config.donor_imputer_learning_rate,
                        "verbose": False,
                    },
                    seed=self.config.random_seed,
                    rng=rng,
                )
                if result is not None:
                    selected_condition_vars = list(result.condition_vars)
                    conditioning_diagnostics.append(
                        {
                            "donor_source": donor_input.frame.source.name,
                            "model_variables": list(donor_block_spec.model_variables),
                            "restored_variables": list(
                                donor_block_spec.restored_variables
                            ),
                            "condition_selection": (
                                self.config.donor_imputer_condition_selection
                            ),
                            "used_condition_surface": False,
                            "raw_shared_vars": list(prepared_inputs.raw_shared_vars),
                            "shared_vars_after_model_exclusion": list(
                                prepared_inputs.shared_vars_after_model_exclusion
                            ),
                            "projection_applied": prepared_inputs.projection_applied,
                            "entity_compatible_shared_vars": list(
                                prepared_inputs.entity_compatible_shared_vars
                            ),
                            "shared_vars_for_block": list(shared_vars_for_block),
                            "selected_condition_vars": selected_condition_vars,
                            "dropped_shared_vars": [
                                variable
                                for variable in shared_vars_for_block
                                if variable not in selected_condition_vars
                            ],
                            "requested_supplemental_shared_condition_vars": (
                                requested_supplemental_vars
                            ),
                            "requested_challenger_shared_condition_vars": (
                                requested_challenger_vars
                            ),
                            "raw_supplemental_shared_condition_var_status": (
                                self._summarize_requested_raw_condition_var_status(
                                    donor_frame=donor_seed,
                                    current_frame=current,
                                    scaffold_source=scaffold_input.frame.source,
                                    donor_source=donor_input.frame.source,
                                    numeric_current=numeric_current,
                                    numeric_donor=numeric_donor,
                                    shared_var_set=raw_shared_var_set,
                                    excluded=excluded,
                                    requested_vars=requested_supplemental_vars,
                                )
                            ),
                            "raw_challenger_shared_condition_var_status": (
                                self._summarize_requested_raw_condition_var_status(
                                    donor_frame=donor_seed,
                                    current_frame=current,
                                    scaffold_source=scaffold_input.frame.source,
                                    donor_source=donor_input.frame.source,
                                    numeric_current=numeric_current,
                                    numeric_donor=numeric_donor,
                                    shared_var_set=raw_shared_var_set,
                                    excluded=excluded,
                                    requested_vars=requested_challenger_vars,
                                )
                            ),
                            "supplemental_shared_condition_var_status": (
                                self._summarize_requested_condition_var_status(
                                    donor_frame=donor_condition_source,
                                    current_frame=current_condition_source,
                                    shared_vars=shared_vars_for_block,
                                    selected_condition_vars=selected_condition_vars,
                                    requested_vars=requested_supplemental_vars,
                                )
                            ),
                            "challenger_shared_condition_var_status": (
                                self._summarize_requested_condition_var_status(
                                    donor_frame=donor_condition_source,
                                    current_frame=current_condition_source,
                                    shared_vars=shared_vars_for_block,
                                    selected_condition_vars=selected_condition_vars,
                                    requested_vars=requested_challenger_vars,
                                )
                            ),
                        }
                    )
                    current = result.updated_frame
                    integrated_variables.extend(result.integrated_variables)
                    source_integrated_variables.extend(result.integrated_variables)
                    _emit_us_pipeline_progress(
                        "US microplex donor integration: block complete",
                        donor_source=donor_source_name,
                        block=block_label,
                        integrated_vars=len(result.integrated_variables),
                    )

            if is_puf_support_clone_source:
                if source_original_current is None:
                    raise AssertionError("PUF support clone original frame missing")
                current, puf_support_clone_summary = (
                    self._finalize_puf_support_clone_frame(
                        original=source_original_current,
                        imputed_clone=current,
                        donor_source_name=donor_source_name,
                        integrated_variables=source_integrated_variables,
                        preclone_columns=source_preclone_columns,
                        donor_seed_columns=set(donor_seed.columns),
                        donor_observed=donor_observed,
                    )
                )
                _emit_us_pipeline_progress(
                    "US microplex donor integration: puf support clone complete",
                    donor_source=donor_source_name,
                    rows=len(current),
                    integrated_vars=len(source_integrated_variables),
                )

        return {
            "seed_data": current,
            "integrated_variables": sorted(set(integrated_variables)),
            "conditioning_diagnostics": conditioning_diagnostics,
            "processed_donor_source_order": processed_donor_source_order,
            "puf_clone_source_order": puf_clone_source_order,
            "puf_support_clone_summary": puf_support_clone_summary,
        }

    def _apply_dependent_tax_leaf_soft_caps(
        self,
        seed_data: pd.DataFrame,
    ) -> pd.DataFrame:
        multiplier = self.config.dependent_tax_leaf_soft_cap_multiplier
        if multiplier is None:
            return seed_data
        if "is_tax_unit_dependent" in seed_data.columns:
            dependent = (
                pd.to_numeric(
                    seed_data["is_tax_unit_dependent"], errors="coerce"
                ).fillna(0.0)
                > 0
            )
        elif "is_dependent" in seed_data.columns:
            dependent = (
                pd.to_numeric(seed_data["is_dependent"], errors="coerce").fillna(0.0)
                > 0
            )
        else:
            return seed_data
        base_vars = [
            var
            for var in self.config.dependent_tax_leaf_soft_cap_base_variables
            if var in seed_data.columns
        ]
        if not base_vars:
            return seed_data
        base = (
            pd.to_numeric(seed_data[base_vars].sum(axis=1), errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        cap = base * float(multiplier)
        for variable in self.config.dependent_tax_leaf_soft_cap_variables:
            if variable not in seed_data.columns:
                continue
            series = pd.to_numeric(seed_data[variable], errors="coerce").fillna(0.0)
            adjusted = series.where(~dependent, other=series.clip(upper=cap))
            seed_data[variable] = adjusted
        return seed_data

    def _uses_pe_condition_surface(self) -> bool:
        return self.config.donor_imputer_condition_selection in {
            "pe_prespecified",
            "pe_plus_puf_native_challenger",
        }

    def _select_donor_condition_vars(
        self,
        donor_frame: pd.DataFrame,
        current_frame: pd.DataFrame,
        shared_vars: list[str],
        donor_block: tuple[str, ...],
        donor_source_name: str | None = None,
    ) -> list[str]:
        condition_vars = [
            variable for variable in shared_vars if variable in donor_frame.columns
        ]
        if len(condition_vars) <= 1:
            return condition_vars

        max_condition_vars = self.config.donor_imputer_max_condition_vars
        if self.config.donor_imputer_condition_selection in {
            "pe_prespecified",
            "pe_plus_puf_native_challenger",
        }:
            preferred_condition_vars = self._resolve_preferred_donor_condition_vars(
                donor_frame=donor_frame,
                current_frame=current_frame,
                shared_vars=shared_vars,
                donor_block=donor_block,
            )
            if (
                self.config.donor_imputer_condition_selection
                == "pe_plus_puf_native_challenger"
            ):
                for variable in self._resolve_challenger_shared_condition_vars(
                    donor_frame=donor_frame,
                    current_frame=current_frame,
                    shared_vars=shared_vars,
                    donor_block=donor_block,
                    donor_source_name=donor_source_name,
                ):
                    if variable not in preferred_condition_vars:
                        preferred_condition_vars.append(variable)
            if preferred_condition_vars:
                return preferred_condition_vars
        if (
            self.config.donor_imputer_condition_selection == "all_shared"
            or max_condition_vars is None
            or len(condition_vars) <= max_condition_vars
        ):
            return condition_vars

        scored_conditions = [
            (
                score_donor_condition_var(
                    donor_frame[variable],
                    [
                        donor_frame[target]
                        for target in donor_block
                        if target in donor_frame.columns
                    ],
                    score_modes={
                        variable_semantic_spec_for(target).condition_score_mode
                        for target in donor_block
                    },
                ),
                variable,
            )
            for variable in condition_vars
        ]
        scored_conditions = [
            (score, variable) for score, variable in scored_conditions if score > 0.0
        ]
        if not scored_conditions:
            return condition_vars[:max_condition_vars]

        scored_conditions.sort(key=lambda item: (-item[0], item[1]))
        return [variable for _, variable in scored_conditions[:max_condition_vars]]

    def _resolve_preferred_donor_condition_vars(
        self,
        *,
        donor_frame: pd.DataFrame,
        current_frame: pd.DataFrame,
        shared_vars: list[str] | None = None,
        donor_block: tuple[str, ...],
    ) -> list[str]:
        semantic_specs = tuple(
            variable_semantic_spec_for(target_variable)
            for target_variable in donor_block
        )
        preferred_condition_vars = tuple(
            dict.fromkeys(
                variable
                for spec in semantic_specs
                for variable in spec.preferred_condition_vars
            )
        )
        if not preferred_condition_vars:
            return []
        resolved: list[str] = []
        for variable in preferred_condition_vars:
            if (
                variable not in donor_frame.columns
                or variable not in current_frame.columns
            ):
                continue
            if not pd.api.types.is_numeric_dtype(donor_frame[variable]):
                continue
            if not pd.api.types.is_numeric_dtype(current_frame[variable]):
                continue
            if not self._is_compatible_donor_condition(
                current_frame[variable],
                donor_frame[variable],
            ):
                continue
            resolved.append(variable)
        shared_var_set = set(shared_vars or ())
        supplemental_shared_condition_vars = tuple(
            dict.fromkeys(
                variable
                for spec in semantic_specs
                for variable in spec.supplemental_shared_condition_vars
            )
        )
        for variable in supplemental_shared_condition_vars:
            if variable in resolved or variable not in shared_var_set:
                continue
            resolved.append(variable)
        return resolved

    def _resolve_requested_supplemental_shared_condition_vars(
        self,
        donor_block: tuple[str, ...],
    ) -> list[str]:
        return list(
            dict.fromkeys(
                variable
                for target_variable in donor_block
                for variable in variable_semantic_spec_for(
                    target_variable
                ).supplemental_shared_condition_vars
            )
        )

    def _resolve_requested_challenger_shared_condition_vars(
        self,
        donor_block: tuple[str, ...],
        *,
        donor_source_name: str | None,
    ) -> list[str]:
        if (
            self.config.donor_imputer_condition_selection
            != "pe_plus_puf_native_challenger"
            or donor_source_name is None
            or not donor_source_name.startswith("irs_soi_puf")
        ):
            return []
        return list(
            dict.fromkeys(
                variable
                for target_variable in donor_block
                for variable in variable_semantic_spec_for(
                    target_variable
                ).challenger_shared_condition_vars
            )
        )

    def _resolve_challenger_shared_condition_vars(
        self,
        *,
        donor_frame: pd.DataFrame,
        current_frame: pd.DataFrame,
        shared_vars: list[str] | None = None,
        donor_block: tuple[str, ...],
        donor_source_name: str | None,
    ) -> list[str]:
        requested_vars = self._resolve_requested_challenger_shared_condition_vars(
            donor_block,
            donor_source_name=donor_source_name,
        )
        if not requested_vars:
            return []
        shared_var_set = set(shared_vars or ())
        resolved: list[str] = []
        for variable in requested_vars:
            if (
                variable not in shared_var_set
                or variable not in donor_frame.columns
                or variable not in current_frame.columns
                or not pd.api.types.is_numeric_dtype(donor_frame[variable])
                or not pd.api.types.is_numeric_dtype(current_frame[variable])
                or not self._is_compatible_donor_condition(
                    current_frame[variable],
                    donor_frame[variable],
                )
            ):
                continue
            resolved.append(variable)
        return resolved

    def _summarize_requested_condition_var_status(
        self,
        *,
        donor_frame: pd.DataFrame,
        current_frame: pd.DataFrame,
        shared_vars: list[str],
        selected_condition_vars: list[str],
        requested_vars: list[str],
    ) -> list[dict[str, Any]]:
        shared_var_set = set(shared_vars)
        selected_var_set = set(selected_condition_vars)
        statuses: list[dict[str, Any]] = []
        for variable in requested_vars:
            status = {
                "variable": variable,
                "selected": variable in selected_var_set,
                "in_shared_overlap": variable in shared_var_set,
            }
            if variable in selected_var_set:
                status["reason"] = "selected"
            elif variable in shared_var_set:
                status["reason"] = "available_but_not_selected"
            elif variable not in donor_frame.columns:
                status["reason"] = "missing_donor_column"
            elif variable not in current_frame.columns:
                status["reason"] = "missing_current_column"
            elif not pd.api.types.is_numeric_dtype(donor_frame[variable]):
                status["reason"] = "non_numeric_donor_column"
            elif not pd.api.types.is_numeric_dtype(current_frame[variable]):
                status["reason"] = "non_numeric_current_column"
            elif not self._is_compatible_donor_condition(
                current_frame[variable],
                donor_frame[variable],
            ):
                status["reason"] = "incompatible_condition_support"
            else:
                status["reason"] = "excluded_from_block_shared_overlap"
            statuses.append(status)
        return statuses

    def _summarize_requested_raw_condition_var_status(
        self,
        *,
        donor_frame: pd.DataFrame,
        current_frame: pd.DataFrame,
        scaffold_source: SourceDescriptor,
        donor_source: SourceDescriptor,
        numeric_current: set[str],
        numeric_donor: set[str],
        shared_var_set: set[str],
        excluded: set[str],
        requested_vars: list[str],
    ) -> list[dict[str, Any]]:
        statuses: list[dict[str, Any]] = []
        for variable in requested_vars:
            status = {
                "variable": variable,
                "selected": variable in shared_var_set,
                "in_shared_overlap": variable in shared_var_set,
            }
            if variable in shared_var_set:
                status["reason"] = "selected"
            elif variable in excluded:
                status["reason"] = "excluded_variable"
            elif variable not in current_frame.columns:
                status["reason"] = "missing_current_column"
            elif variable not in donor_frame.columns:
                status["reason"] = "missing_donor_column"
            elif variable not in numeric_current:
                status["reason"] = "non_numeric_current_column"
            elif variable not in numeric_donor:
                status["reason"] = "non_numeric_donor_column"
            elif not scaffold_source.allows_conditioning_on(variable):
                status["reason"] = "scaffold_source_disallows_conditioning"
            elif not donor_source.allows_conditioning_on(variable):
                status["reason"] = "donor_source_disallows_conditioning"
            elif not self._is_compatible_donor_condition(
                current_frame[variable],
                donor_frame[variable],
            ):
                status["reason"] = "incompatible_condition_support"
            else:
                status["reason"] = "excluded_from_shared_overlap"
            statuses.append(status)
        return statuses

    def _augment_donor_condition_frame_for_targets(
        self,
        frame: pd.DataFrame,
        target_variables: tuple[str, ...],
    ) -> pd.DataFrame:
        preferred_condition_vars = [
            variable
            for target_variable in target_variables
            for variable in variable_semantic_spec_for(
                target_variable
            ).preferred_condition_vars
        ]
        if not preferred_condition_vars:
            return frame
        if not set(PE_STYLE_PUF_IRS_DEMOGRAPHIC_PREDICTORS) & set(
            preferred_condition_vars
        ):
            return frame
        predictor_frame = self._build_pe_style_puf_irs_condition_frame(frame)
        if predictor_frame.empty:
            return frame
        result = frame.copy()
        for column in predictor_frame.columns:
            result[column] = predictor_frame[column]
        return result

    def _build_pe_style_puf_irs_condition_frame(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        sex = (
            pd.to_numeric(frame["sex"], errors="coerce")
            if "sex" in frame.columns
            else pd.Series(np.nan, index=frame.index, dtype=float)
        )
        if "age" in frame.columns:
            result["age"] = pd.to_numeric(frame["age"], errors="coerce").astype(float)
        if "sex" in frame.columns:
            result["is_male"] = pd.Series(
                np.where(sex == 1, 1.0, np.where(sex == 2, 0.0, np.nan)),
                index=frame.index,
                dtype=float,
            )
        elif "is_male" in frame.columns:
            result["is_male"] = pd.to_numeric(frame["is_male"], errors="coerce").astype(
                float
            )
        if "tax_unit_id" not in frame.columns:
            return result

        relationship = (
            self._normalize_relationship_to_head(frame)
            if "relationship_to_head" not in frame.columns
            else pd.to_numeric(frame["relationship_to_head"], errors="coerce")
            .fillna(3)
            .astype(int)
        )
        result["tax_unit_is_joint"] = 0.0
        result["tax_unit_count_dependents"] = 0.0
        result["is_tax_unit_head"] = 0.0
        result["is_tax_unit_spouse"] = 0.0
        result["is_tax_unit_dependent"] = 0.0

        ages = (
            pd.to_numeric(frame["age"], errors="coerce").fillna(0.0)
            if "age" in frame.columns
            else pd.Series(0.0, index=frame.index, dtype=float)
        )
        spouse_person_number = (
            pd.to_numeric(frame.get("spouse_person_number"), errors="coerce")
            .fillna(0)
            .astype(int)
            if "spouse_person_number" in frame.columns
            else pd.Series(0, index=frame.index, dtype=int)
        )
        person_number = (
            pd.to_numeric(frame.get("person_number"), errors="coerce")
            .fillna(0)
            .astype(int)
            if "person_number" in frame.columns
            else pd.Series(0, index=frame.index, dtype=int)
        )

        tax_unit_ids = frame["tax_unit_id"]
        valid_tax_unit_ids = tax_unit_ids.notna() & tax_unit_ids.astype(
            str
        ).str.strip().ne("")
        for _, unit_persons in frame.loc[valid_tax_unit_ids].groupby(
            "tax_unit_id",
            sort=False,
        ):
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
                    unit_relationship[unit_relationship.eq(1)]
                    .index.astype(int)
                    .tolist()
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

    def _entity_key_column(self, entity: EntityType) -> str | None:
        return ENTITY_ID_COLUMNS.get(entity)

    def _ensure_seed_entity_ids(
        self,
        frame: pd.DataFrame,
        *,
        entities: set[EntityType],
        frame_role: str | None = None,
        donor_source_name: str | None = None,
    ) -> pd.DataFrame:
        missing_columns = [
            self._entity_key_column(entity)
            for entity in entities
            if entity is not EntityType.PERSON
            and self._entity_key_column(entity) not in frame.columns
        ]
        if not missing_columns:
            _emit_us_pipeline_progress(
                "US microplex donor integration: entity ids ready",
                donor_source=donor_source_name,
                frame=frame_role,
                rows=len(frame),
                status="already_present",
                columns=_format_progress_values(
                    sorted(
                        self._entity_key_column(entity) or ""
                        for entity in entities
                        if entity is not EntityType.PERSON
                    )
                ),
            )
            return frame
        started_at = time.perf_counter()
        missing_column_set = set(missing_columns)
        can_use_group_only_path = missing_column_set <= {"family_id", "spm_unit_id"}
        method = (
            "family_spm_only"
            if can_use_group_only_path
            else "policyengine_entity_bundle"
        )
        _emit_us_pipeline_progress(
            "US microplex donor integration: entity ids start",
            donor_source=donor_source_name,
            frame=frame_role,
            rows=len(frame),
            missing_columns=_format_progress_values(missing_columns),
            method=method,
        )
        working = frame.copy()
        original_person_ids = working["person_id"].copy()
        working["person_id"] = np.arange(len(working), dtype=np.int64)
        if "household_id" in working.columns:
            working["household_id"] = pd.factorize(working["household_id"])[0].astype(
                np.int64
            )
        else:
            working["household_id"] = np.arange(len(working), dtype=np.int64)
        if "age" not in working.columns:
            working["age"] = 0
        if can_use_group_only_path:
            working["relationship_to_head"] = self._normalize_relationship_to_head(
                working
            )
            persons = self._assign_family_and_spm_units(working).copy()
        else:
            persons = self.build_policyengine_entity_tables(working).persons.copy()
        persons["source_person_id"] = original_person_ids.to_numpy()
        mapping = persons[["source_person_id", *missing_columns]]
        if mapping["source_person_id"].duplicated().any():
            raise ValueError(
                "PolicyEngine entity table build produced duplicate person mappings"
            )
        result = frame.merge(
            mapping,
            left_on="person_id",
            right_on="source_person_id",
            how="left",
        ).drop(columns=["source_person_id"])
        _emit_us_pipeline_progress(
            "US microplex donor integration: entity ids complete",
            donor_source=donor_source_name,
            frame=frame_role,
            rows=len(result),
            added_columns=_format_progress_values(missing_columns),
            method=method,
            elapsed_seconds=f"{time.perf_counter() - started_at:.3f}",
        )
        return result

    def _strip_generated_entity_ids(
        self,
        frame: pd.DataFrame,
        *,
        scaffold_input: USMicroplexSourceInput,
    ) -> pd.DataFrame:
        scaffold_person_columns = set(scaffold_input.persons.columns)
        ephemeral_entity_ids = [
            column
            for column in ("tax_unit_id", "family_id", "spm_unit_id", "marital_unit_id")
            if column in frame.columns and column not in scaffold_person_columns
        ]
        if not ephemeral_entity_ids:
            return frame
        return frame.drop(columns=ephemeral_entity_ids)

    def _can_project_donor_block_to_entity(
        self,
        current_frame: pd.DataFrame,
        donor_frame: pd.DataFrame,
        entity: EntityType,
    ) -> bool:
        if entity is EntityType.PERSON:
            return False
        entity_key = self._entity_key_column(entity)
        return bool(
            entity_key
            and entity_key in current_frame.columns
            and entity_key in donor_frame.columns
            and current_frame[entity_key].notna().all()
            and donor_frame[entity_key].notna().all()
        )

    def _project_frame_to_entity(
        self,
        frame: pd.DataFrame,
        *,
        entity: EntityType,
        variables: set[str],
    ) -> pd.DataFrame:
        entity_key = self._entity_key_column(entity)
        if entity_key is None:
            raise ValueError(f"Unsupported donor projection entity: {entity}")
        columns = [
            entity_key,
            *[
                variable
                for variable in sorted(variables)
                if variable != entity_key and variable in frame.columns
            ],
        ]
        projected = frame[columns].copy()
        if entity is EntityType.PERSON:
            return projected

        sort_columns = [
            column
            for column in (entity_key, "household_id", "person_id")
            if column in projected.columns
        ]
        if sort_columns:
            projected = projected.sort_values(sort_columns, kind="mergesort")
        aggregations = {
            column: self._projection_aggregation_for(column)
            for column in projected.columns
            if column != entity_key
        }
        return projected.groupby(entity_key, as_index=False).agg(aggregations)

    def _projection_aggregation_for(self, column: str) -> str:
        if column in {"hh_weight", "household_id", "person_id", "year"}:
            return "first"
        return variable_semantic_spec_for(column).projection_aggregation.value

    def _should_integrate_donor_variable(
        self,
        current: pd.DataFrame,
        variable: str,
    ) -> bool:
        if variable not in current.columns:
            return True
        current_values = pd.to_numeric(
            current[variable],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        informative = current_values.dropna()
        if informative.empty:
            return True
        if (informative != 0).any():
            return False
        return informative.nunique() <= 1

    def _is_compatible_donor_condition(
        self,
        current_series: pd.Series,
        donor_series: pd.Series,
    ) -> bool:
        current_values = (
            pd.to_numeric(current_series, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        donor_values = (
            pd.to_numeric(donor_series, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if current_values.empty or donor_values.empty:
            return False
        if current_values.nunique() <= 1:
            return False
        if donor_values.nunique() <= 1:
            return False
        return True

    def _is_compatible_donor_target(self, series: pd.Series) -> bool:
        values = pd.to_numeric(series, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        values = values.dropna()
        if values.empty:
            return False
        if values.nunique() <= 1:
            return False
        return bool((values > 0).any())

    def _rank_match_donor_values(
        self,
        scores: pd.Series,
        *,
        donor_values: pd.Series,
        donor_weights: pd.Series | None,
        rng: np.random.Generator,
        strategy: DonorMatchStrategy = DonorMatchStrategy.RANK,
    ) -> pd.Series:
        """Assign donor values by rank, preserving the donor marginal distribution."""
        if donor_values.empty:
            return pd.Series(0.0, index=scores.index, dtype=float)

        donor_array = donor_values.to_numpy(dtype=float)
        donor_weight_array = None
        if donor_weights is not None and not donor_weights.empty:
            donor_weight_array = donor_weights.to_numpy(dtype=float)
            donor_weight_array = np.clip(donor_weight_array, a_min=0.0, a_max=None)

        if (
            strategy is DonorMatchStrategy.RANK
            and self._is_zero_inflated_positive_distribution(donor_array)
        ):
            return self._rank_match_zero_inflated_positive_values(
                scores,
                donor_values=donor_array,
                donor_weights=donor_weight_array,
                rng=rng,
            )

        sampled_values = self._sample_donor_array(
            donor_array,
            size=len(scores),
            donor_weights=donor_weight_array,
            rng=rng,
        )

        sampled_values = np.sort(sampled_values.astype(float))
        order = np.argsort(scores.to_numpy(dtype=float), kind="mergesort")
        matched = np.empty(len(scores), dtype=float)
        matched[order] = sampled_values
        return pd.Series(matched, index=scores.index, dtype=float)

    def _rank_match_zero_inflated_positive_values(
        self,
        scores: pd.Series,
        *,
        donor_values: np.ndarray,
        donor_weights: np.ndarray | None,
        rng: np.random.Generator,
    ) -> pd.Series:
        matched = np.zeros(len(scores), dtype=float)
        positive_mask = donor_values > 0.0
        positive_values = donor_values[positive_mask]
        if len(positive_values) == 0:
            return pd.Series(matched, index=scores.index, dtype=float)

        positive_rate = self._weighted_positive_rate(
            donor_values,
            donor_weights=donor_weights,
        )
        n_positive = int(round(positive_rate * len(scores)))
        n_positive = min(max(n_positive, 0), len(scores))
        if n_positive == 0:
            return pd.Series(matched, index=scores.index, dtype=float)

        positive_weights = (
            donor_weights[positive_mask] if donor_weights is not None else None
        )
        sampled_positive = self._sample_donor_array(
            positive_values,
            size=n_positive,
            donor_weights=positive_weights,
            rng=rng,
        )
        sampled_positive = np.sort(sampled_positive.astype(float))
        order = np.argsort(scores.to_numpy(dtype=float), kind="mergesort")
        matched[order[-n_positive:]] = sampled_positive
        return pd.Series(matched, index=scores.index, dtype=float)

    def _sample_donor_array(
        self,
        donor_values: np.ndarray,
        *,
        size: int,
        donor_weights: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if len(donor_values) == size:
            return donor_values.copy()

        probabilities = None
        if donor_weights is not None and len(donor_weights) == len(donor_values):
            weight_sum = float(donor_weights.sum())
            if weight_sum > 0.0:
                probabilities = donor_weights / weight_sum
        return rng.choice(
            donor_values,
            size=size,
            replace=True,
            p=probabilities,
        )

    def _weighted_positive_rate(
        self,
        donor_values: np.ndarray,
        *,
        donor_weights: np.ndarray | None,
    ) -> float:
        positive_mask = donor_values > 0.0
        if donor_weights is None or len(donor_weights) != len(donor_values):
            return float(np.mean(positive_mask))
        weight_sum = float(donor_weights.sum())
        if weight_sum <= 0.0:
            return float(np.mean(positive_mask))
        return float(donor_weights[positive_mask].sum() / weight_sum)

    def _is_zero_inflated_positive_distribution(self, donor_values: np.ndarray) -> bool:
        return bool(
            len(donor_values) > 0
            and np.all(donor_values >= 0.0)
            and np.any(donor_values == 0.0)
            and np.any(donor_values > 0.0)
        )

    def _synthesize_bootstrap(
        self,
        seed_data: pd.DataFrame,
        initial_weight: float,
        *,
        strata_columns: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """Generate synthetic households via weighted bootstrap resampling."""
        rng = np.random.default_rng(self.config.random_seed)
        households = (
            seed_data.groupby("household_id", as_index=False)
            .agg(
                {
                    "hh_weight": "first",
                    **{
                        column: "first"
                        for column in strata_columns
                        if column in seed_data.columns
                    },
                }
            )
            .rename(columns={"hh_weight": "household_weight"})
        )
        sampled_households = self._sample_bootstrap_household_ids(
            households,
            rng=rng,
            strata_columns=strata_columns,
        )

        cloned_households: list[pd.DataFrame] = []
        for new_household_id, source_household_id in enumerate(sampled_households):
            household_persons = seed_data[
                seed_data["household_id"] == source_household_id
            ].copy()
            household_persons["household_id"] = new_household_id
            cloned_households.append(household_persons)

        synthetic = pd.concat(cloned_households, ignore_index=True)
        if "income" in synthetic.columns:
            synthetic["income"] = synthetic["income"].astype(float) * rng.lognormal(
                mean=0.0,
                sigma=0.05,
                size=len(synthetic),
            )
            synthetic["income"] = synthetic["income"].clip(lower=0.0)
        return self._finalize_synthetic_population(
            synthetic,
            initial_weight=initial_weight,
        )

    def _resolve_bootstrap_strata_columns(
        self,
        seed_data: pd.DataFrame,
    ) -> tuple[str, ...]:
        if self.config.bootstrap_strata_columns:
            missing_columns = [
                column
                for column in self.config.bootstrap_strata_columns
                if column not in seed_data.columns
            ]
            if missing_columns:
                raise ValueError(
                    "bootstrap_strata_columns are not available in seed data: "
                    f"{missing_columns}"
                )
            return self.config.bootstrap_strata_columns

        requested_geo_levels: set[str] = set()
        for scope in (False, True):
            _, _, geo_levels = self._policyengine_target_scope(for_calibration=scope)
            requested_geo_levels.update(geo_levels)

        inferred_columns: list[str] = []
        if {
            "state",
            "district",
            "county",
        } & requested_geo_levels and "state_fips" in seed_data.columns:
            inferred_columns.append("state_fips")
        if "county" in requested_geo_levels and "county_fips" in seed_data.columns:
            inferred_columns.append("county_fips")
        if (
            "district" in requested_geo_levels
            and "congressional_district_geoid" in seed_data.columns
        ):
            inferred_columns.append("congressional_district_geoid")
        return tuple(dict.fromkeys(inferred_columns))

    def _sample_bootstrap_household_ids(
        self,
        households: pd.DataFrame,
        *,
        rng: np.random.Generator,
        strata_columns: tuple[str, ...],
    ) -> np.ndarray:
        weights = households["household_weight"].astype(float).to_numpy()
        household_ids = households["household_id"].to_numpy()
        if (
            not strata_columns
            or self.config.n_synthetic <= 0
            or len(household_ids) == 0
        ):
            probabilities = weights / weights.sum()
            return rng.choice(
                household_ids,
                size=self.config.n_synthetic,
                replace=True,
                p=probabilities,
            )

        stratum_frame = households.loc[:, list(strata_columns)].copy()
        for column in stratum_frame.columns:
            values = stratum_frame[column]
            if pd.api.types.is_numeric_dtype(values):
                stratum_frame[column] = values.fillna(-1)
            else:
                stratum_frame[column] = values.astype("string").fillna("__missing__")
        stratum_keys = pd.MultiIndex.from_frame(stratum_frame)
        weighted_households = households.assign(_bootstrap_stratum_key=stratum_keys)
        stratum_weights = (
            weighted_households.groupby("_bootstrap_stratum_key", dropna=False)[
                "household_weight"
            ]
            .sum()
            .astype(float)
        )
        stratum_weights = stratum_weights[stratum_weights > 0]
        if stratum_weights.empty:
            probabilities = weights / weights.sum()
            return rng.choice(
                household_ids,
                size=self.config.n_synthetic,
                replace=True,
                p=probabilities,
            )

        n_strata = len(stratum_weights)
        base_counts = pd.Series(0, index=stratum_weights.index, dtype=int)
        remaining = self.config.n_synthetic
        if self.config.n_synthetic >= n_strata:
            base_counts += 1
            remaining -= n_strata

        probabilities = (stratum_weights / stratum_weights.sum()).to_numpy(dtype=float)
        extra_counts = (
            rng.multinomial(remaining, probabilities)
            if remaining > 0
            else np.zeros(n_strata, dtype=int)
        )

        sampled_households: list[np.ndarray] = []
        for stratum_key, sample_count in zip(
            stratum_weights.index,
            base_counts.to_numpy(dtype=int) + extra_counts,
            strict=False,
        ):
            if sample_count <= 0:
                continue
            candidates = weighted_households.loc[
                weighted_households["_bootstrap_stratum_key"] == stratum_key
            ]
            candidate_ids = candidates["household_id"].to_numpy()
            candidate_weights = candidates["household_weight"].astype(float).to_numpy()
            if candidate_weights.sum() <= 0:
                candidate_probabilities = np.full(
                    len(candidate_ids),
                    1.0 / max(len(candidate_ids), 1),
                )
            else:
                candidate_probabilities = candidate_weights / candidate_weights.sum()
            sampled_households.append(
                rng.choice(
                    candidate_ids,
                    size=int(sample_count),
                    replace=True,
                    p=candidate_probabilities,
                )
            )

        if not sampled_households:
            probabilities = weights / weights.sum()
            return rng.choice(
                household_ids,
                size=self.config.n_synthetic,
                replace=True,
                p=probabilities,
            )

        return rng.permutation(np.concatenate(sampled_households))

    def _finalize_synthetic_population(
        self,
        synthetic: pd.DataFrame,
        initial_weight: float,
    ) -> pd.DataFrame:
        """Add derived fields and canonical identifiers to synthetic output."""
        result = synthetic.copy().reset_index(drop=True)
        for column, default in {
            "state_fips": 0,
            "county_fips": "00000",
            "block_geoid": "",
            "tract_geoid": "",
            "congressional_district_geoid": 0,
            "tenure": 0,
            "age": 0,
            "sex": 0,
            "education": 0,
            "employment_status": 0,
            "income": 0.0,
        }.items():
            if column not in result.columns:
                result[column] = default
        result["person_id"] = np.arange(len(result))
        if "household_id" in result.columns:
            result["household_id"] = pd.factorize(result["household_id"])[0].astype(
                np.int64
            )
        else:
            result["household_id"] = np.arange(len(result), dtype=np.int64)
        result["state"] = result["state_fips"].map(STATE_FIPS).fillna("UNK")
        result["age_group"] = pd.cut(
            result["age"],
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=False,
        ).astype(str)
        result["income_bracket"] = pd.cut(
            result["income"],
            bins=INCOME_BINS,
            labels=INCOME_LABELS,
        ).astype(str)
        if "weight" not in result.columns:
            result["weight"] = float(initial_weight)
        else:
            result["weight"] = (
                pd.to_numeric(result["weight"], errors="coerce")
                .fillna(float(initial_weight))
                .astype(float)
            )
        return result

    def _build_policyengine_households(self, persons: pd.DataFrame) -> pd.DataFrame:
        household_columns = [
            column
            for column in (
                "state_fips",
                "county_fips",
                "block_geoid",
                "tract_geoid",
                "congressional_district_geoid",
                "tenure",
                "tenure_type",
                "state",
                "net_worth",
                "auto_loan_balance",
                "auto_loan_interest",
            )
            if column in persons.columns
        ]
        aggregations = {column: "first" for column in household_columns}
        aggregations["weight"] = "mean"
        households = (
            persons.groupby("household_id", as_index=False)
            .agg(aggregations)
            .rename(columns={"weight": "household_weight"})
        )
        return households

    def _build_policyengine_tax_units(
        self,
        persons: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        person_rows = persons.copy()
        tax_unit_rows: list[dict[str, Any]] = []
        person_to_tax_unit: dict[int, int] = {}
        next_tax_unit_id = 0
        preserved_households: set[Any] = set()

        role_based = self._build_policyengine_tax_units_from_role_flags(
            persons,
            start_tax_unit_id=next_tax_unit_id,
        )
        if role_based is not None:
            role_tax_units, role_person_rows, role_households = role_based
            if len(role_households) == person_rows["household_id"].nunique():
                return role_tax_units, role_person_rows
            if not role_tax_units.empty:
                tax_unit_rows.extend(role_tax_units.to_dict(orient="records"))
                person_to_tax_unit.update(
                    {
                        int(person_id): int(tax_unit_id)
                        for person_id, tax_unit_id in zip(
                            role_person_rows["person_id"].tolist(),
                            role_person_rows["tax_unit_id"].tolist(),
                            strict=True,
                        )
                    }
                )
                preserved_households.update(role_households)
                next_tax_unit_id = (
                    int(
                        pd.to_numeric(
                            role_tax_units["tax_unit_id"],
                            errors="coerce",
                        ).max()
                    )
                    + 1
                )

        if self.config.policyengine_prefer_existing_tax_unit_ids:
            remaining_persons = persons.loc[
                ~persons["household_id"].isin(preserved_households)
            ].copy()
            preserved = self._build_policyengine_tax_units_from_existing_ids(
                remaining_persons,
                start_tax_unit_id=next_tax_unit_id,
            )
            if preserved is not None:
                preserved_tax_units, preserved_person_rows, existing_households = (
                    preserved
                )
                if (
                    len(existing_households | preserved_households)
                    == person_rows["household_id"].nunique()
                    and not tax_unit_rows
                ):
                    return preserved_tax_units, preserved_person_rows
                if not preserved_tax_units.empty:
                    tax_unit_rows.extend(preserved_tax_units.to_dict(orient="records"))
                    person_to_tax_unit.update(
                        {
                            int(person_id): int(tax_unit_id)
                            for person_id, tax_unit_id in zip(
                                preserved_person_rows["person_id"].tolist(),
                                preserved_person_rows["tax_unit_id"].tolist(),
                                strict=True,
                            )
                        }
                    )
                    preserved_households.update(existing_households)
                    next_tax_unit_id = (
                        int(
                            pd.to_numeric(
                                preserved_tax_units["tax_unit_id"],
                                errors="coerce",
                            ).max()
                        )
                        + 1
                    )

        optimizer = TaxUnitOptimizer()

        for household_id in person_rows["household_id"].drop_duplicates().tolist():
            if household_id in preserved_households:
                continue
            hh_persons = person_rows[person_rows["household_id"] == household_id].copy()
            if hh_persons.empty:
                continue
            optimized_units = optimizer.optimize_household(
                int(household_id), hh_persons
            )
            optimized_units = self._apply_tax_unit_filing_status_hints(
                hh_persons,
                optimized_units,
            )
            if not optimized_units:
                optimized_units = [
                    {
                        "tax_unit_id": 0,
                        "household_id": int(household_id),
                        "filing_status": "single",
                        "filer_ids": [int(hh_persons.iloc[0]["person_id"])],
                        "dependent_ids": [],
                        "n_dependents": 0,
                        "total_income": float(hh_persons["income"].sum()),
                        "tax_liability": 0.0,
                    }
                ]

            assigned_person_ids: set[int] = set()
            for unit in optimized_units:
                unit_person_ids = [
                    int(person_id)
                    for person_id in list(unit.get("filer_ids", []))
                    + list(unit.get("dependent_ids", []))
                ]
                if not unit_person_ids:
                    continue
                global_tax_unit_id = next_tax_unit_id
                next_tax_unit_id += 1
                for person_id in unit_person_ids:
                    person_to_tax_unit[person_id] = global_tax_unit_id
                    assigned_person_ids.add(person_id)
                unit_persons = hh_persons.loc[
                    hh_persons["person_id"].astype(int).isin(unit_person_ids)
                ].copy()
                tax_unit_rows.append(
                    {
                        "tax_unit_id": global_tax_unit_id,
                        "household_id": int(household_id),
                        "filing_status": self._normalize_policyengine_filing_status(
                            unit.get("filing_status", "single")
                        ),
                        "n_dependents": int(unit.get("n_dependents", 0)),
                        "total_income": float(unit.get("total_income", 0.0)),
                        "tax_liability": float(unit.get("tax_liability", 0.0)),
                        **self._aggregate_policyengine_tax_unit_input_columns(
                            unit_persons
                        ),
                    }
                )

            unassigned = [
                int(person_id)
                for person_id in hh_persons["person_id"].tolist()
                if int(person_id) not in assigned_person_ids
            ]
            for person_id in unassigned:
                global_tax_unit_id = next_tax_unit_id
                next_tax_unit_id += 1
                person_to_tax_unit[person_id] = global_tax_unit_id
                unit_persons = hh_persons.loc[
                    hh_persons["person_id"].astype(int).eq(person_id)
                ].copy()
                tax_unit_rows.append(
                    {
                        "tax_unit_id": global_tax_unit_id,
                        "household_id": int(household_id),
                        "filing_status": "SINGLE",
                        "n_dependents": 0,
                        "total_income": float(
                            hh_persons.loc[
                                hh_persons["person_id"] == person_id, "income"
                            ].iloc[0]
                        ),
                        "tax_liability": 0.0,
                        **self._aggregate_policyengine_tax_unit_input_columns(
                            unit_persons
                        ),
                    }
                )

        person_rows["tax_unit_id"] = person_rows["person_id"].map(person_to_tax_unit)
        tax_units = pd.DataFrame(tax_unit_rows)
        return tax_units, person_rows

    # Raw CPS ASEC columns that ``microunit.construct_tax_units`` consumes to
    # reconstruct tax units. ``microunit`` is the standalone extraction of
    # eCPS's tax-unit logic (issue #113); it is *source-agnostic* and expects
    # this normalized CPS-like contract rather than microplex's collapsed
    # ``relationship_to_head`` coding. We only delegate when the frame actually
    # carries these columns, so the delegation is behavior-preserving on
    # today's frames (which do not carry them) and only becomes active once an
    # upstream change threads CPS columns through to entity construction.
    _MICROUNIT_REQUIRED_CPS_COLUMNS = (
        "PH_SEQ",
        "A_LINENO",
        "A_AGE",
        "A_MARITL",
        "A_SPOUSE",
        "PEPAR1",
        "PEPAR2",
        "A_EXPRRP",
    )

    def _build_policyengine_tax_units_via_microunit(
        self,
        persons: pd.DataFrame,
        *,
        start_tax_unit_id: int = 0,
        allow_normalized_adapter: bool | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, set[Any]] | None:
        """Reconstruct tax units by delegating to ``microunit`` (issue #113).

        This is microplex's **default** tax-unit constructor for CPS-derived
        frames. ``microunit`` is the rules-based engine that *replaces* the
        unreliable CPS-provided ``tax_unit_id`` (Census ``TAX_ID``): when the
        frame carries the real CPS pointer fields, the high-fidelity adapter
        (#115) builds microunit's CPS contract and microunit re-partitions each
        household from scratch, intentionally overriding any incoming
        ``tax_unit_id``. The ``policyengine_prefer_existing_tax_unit_ids`` /
        :meth:`_build_policyengine_tax_units_from_existing_ids` path is a
        **fallback** for the households this method does not construct -- it runs
        *after* this one, on the remaining households -- not a parallel
        authority. SPM/family/marital group IDs are preserved separately (#112)
        and are not touched here, so "keep the source SPM units, replace the tax
        units" holds.

        Delegation runs when ``persons`` carries the raw CPS columns in
        :attr:`_MICROUNIT_REQUIRED_CPS_COLUMNS`, or can synthesize them: the
        high-fidelity adapter (#115) is used by DEFAULT when the real
        ``person_number``/``spouse_person_number``/``family_relationship`` fields
        are present (the production candidate carries them); the coarse
        ``relationship_to_head``-only heuristic stays opt-in. When neither the
        raw columns nor the high-fidelity fields are available (and the coarse
        heuristic is not enabled), we return ``None`` and let the caller fall
        back to the legacy role-flag reconstruction.

        .. warning::
            ``microunit`` *is* eCPS's tax-unit construction. Routing microplex
            through it makes microplex's constructed tax units **converge toward
            eCPS's**. Any loss change from enabling this delegation is an
            *entity-convergence* effect and must be interpreted as such, not as
            a quality improvement. See issue #113.

        Returns the same ``(tax_units, person_rows, households)`` triple shape as
        :meth:`_build_policyengine_tax_units_from_role_flags`, or ``None`` to
        defer to the caller's fallback.
        """
        if "person_id" not in persons.columns or "household_id" not in persons.columns:
            return None
        cps_frame = persons
        if not set(self._MICROUNIT_REQUIRED_CPS_COLUMNS).issubset(persons.columns):
            # microunit is microplex's required tax-unit engine (#113). When the
            # raw CPS columns are absent, synthesize its CPS contract from the
            # normalized frame. The high-fidelity path (real person_number /
            # spouse_person_number / family_relationship, which the production
            # candidate carries) is used by DEFAULT; the coarse
            # relationship_to_head-only heuristic stays opt-in via the config flag
            # so minimal frames don't silently get the lossy reconstruction.
            has_high_fidelity_fields = {
                "person_number",
                "family_relationship",
            }.issubset(persons.columns)
            if allow_normalized_adapter is None:
                allow_normalized_adapter = has_high_fidelity_fields or bool(
                    getattr(self.config, "microunit_construct_from_normalized", False)
                )
            if not allow_normalized_adapter:
                return None
            cps_frame = self._microunit_cps_frame_from_normalized(persons)
            if cps_frame is None:
                return None

        # Imported lazily to match this module's optional-dependency convention:
        # ``microunit`` ships in the ``policyengine`` extra, and the base test
        # suite must import this module without that extra installed.
        from microunit import POLICYENGINE_MODE, construct_tax_units

        # microunit keys its CPS-style frame on (PH_SEQ, A_LINENO); resetting the
        # index keeps row order so the returned per-person TAX_ID and role align
        # positionally back onto person_rows.
        person_rows = cps_frame.reset_index(drop=True).copy()
        try:
            person_assignments, tax_unit = construct_tax_units(
                person_rows.copy(),
                year=self._microunit_reference_year(person_rows),
                mode=POLICYENGINE_MODE,
            )
        except Exception:
            # microunit raises on households it cannot resolve (e.g. no valid
            # reference person). Never let that crash materialization — fall back
            # to the caller's legacy reconstruction for the whole frame.
            LOGGER.warning(
                "microunit tax-unit construction failed; falling back to "
                "legacy reconstruction",
                exc_info=True,
            )
            return None

        tax_id = pd.to_numeric(person_assignments["TAX_ID"], errors="coerce")
        person_rows["tax_unit_id"] = (
            tax_id.to_numpy() + int(start_tax_unit_id)
        ).astype(np.int64)
        # microunit emits an authoritative per-person HEAD/SPOUSE/DEPENDENT role;
        # use it directly for the filer/dependent split rather than re-deriving
        # from the (possibly absent) collapsed relationship_to_head coding.
        person_rows["_microunit_role"] = [
            self._decode_microunit_bytes(role)
            for role in person_assignments["tax_unit_role_input"].tolist()
        ]

        # microunit emits the canonical filing-status vocabulary already, but
        # normalize defensively so this path can never diverge from the legacy
        # paths if microunit ever changes its spelling/casing.
        filing_status_by_unit = {
            int(row_tax_id) + int(start_tax_unit_id): (
                self._normalize_policyengine_filing_status(
                    self._decode_microunit_bytes(filing_value)
                )
            )
            for row_tax_id, filing_value in zip(
                tax_unit["TAX_ID"].tolist(),
                tax_unit["filing_status_input"].tolist(),
                strict=True,
            )
        }

        tax_unit_rows: list[dict[str, Any]] = []
        for unit_id, unit_persons in person_rows.groupby("tax_unit_id", sort=False):
            ordered = unit_persons.sort_values(
                ["_microunit_role", "age", "person_id"],
                ascending=[True, False, True],
            ).reset_index(drop=True)
            is_filer = ordered["_microunit_role"].isin(["HEAD", "SPOUSE"])
            filer_ids = [
                int(person_id) for person_id in ordered.loc[is_filer, "person_id"]
            ]
            dependent_ids = [
                int(person_id) for person_id in ordered.loc[~is_filer, "person_id"]
            ]
            if not filer_ids:
                filer_ids = [int(ordered.iloc[0]["person_id"])]
                dependent_ids = [
                    int(person_id)
                    for person_id in ordered["person_id"].tolist()
                    if int(person_id) not in filer_ids
                ]
            tax_unit_rows.append(
                {
                    "tax_unit_id": int(unit_id),
                    "household_id": int(ordered.iloc[0]["household_id"]),
                    "filing_status": filing_status_by_unit.get(int(unit_id), "SINGLE"),
                    "member_ids": [
                        int(person_id) for person_id in ordered["person_id"]
                    ],
                    "filer_ids": filer_ids,
                    "dependent_ids": dependent_ids,
                    "n_dependents": len(dependent_ids),
                    "total_income": float(
                        pd.to_numeric(ordered.get("income", 0.0), errors="coerce")
                        .fillna(0.0)
                        .sum()
                    ),
                    "tax_liability": 0.0,
                    **self._aggregate_policyengine_tax_unit_input_columns(ordered),
                }
            )

        if not tax_unit_rows:
            return None

        households = set(person_rows["household_id"].drop_duplicates().tolist())
        person_rows = person_rows.drop(columns=["_microunit_role"], errors="ignore")
        return pd.DataFrame(tax_unit_rows), person_rows, households

    def _microunit_cps_frame_from_cps_fields(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame:
        """High-fidelity (#115) build of microunit's CPS contract from the real
        CPS-derived fields microplex carries at materialization: ``person_number``
        (a 1-based within-household line number), ``spouse_person_number`` (a real
        spouse line pointer), ``family_relationship`` (CPS A_FAMREL) and
        ``marital_status``. The household reference person (``person_number == 1``)
        always anchors a valid head, so microunit never lacks one.

        Only ``PEPAR1``/``PEPAR2`` remain heuristic: a child's parents are taken to
        be the household reference person (line 1) and that person's spouse (#115).
        """
        frame = persons.reset_index(drop=True).copy()
        hh = pd.to_numeric(frame["household_id"], errors="coerce")
        pernum = (
            pd.to_numeric(frame["person_number"], errors="coerce").fillna(0).astype(int)
        )
        # ``family_relationship`` arrives in either CPS A_FAMREL 1-based coding
        # (1=reference person, 2=spouse, 3=child, ...) or the optimizer's 0-based
        # coding (0=head, 1=spouse, 2=child); the rest of the pipeline detects
        # this per household (see ``_normalize_relationship_to_head`` and
        # ``data_sources.cps``). The A_EXPRRP / parent-pointer mapping below
        # expects the 1-based scheme, so shift any 0-based household up by one --
        # otherwise a 0-based frame silently mis-codes children as spouses and
        # drops their parent pointers.
        famrel_raw = pd.to_numeric(frame["family_relationship"], errors="coerce")
        zero_based_hh = (famrel_raw == 0).groupby(hh).transform("any").fillna(False)
        famrel = famrel_raw.add(zero_based_hh.astype(int)).fillna(0).astype(int)
        spouse_num = (
            pd.to_numeric(frame.get("spouse_person_number", 0), errors="coerce")
            .fillna(0)
            .astype(int)
        )

        frame["PH_SEQ"] = hh.astype(np.int64)
        frame["A_LINENO"] = pernum
        frame["A_AGE"] = (
            pd.to_numeric(frame["age"], errors="coerce").fillna(0).astype(int)
        )
        frame["A_SPOUSE"] = spouse_num

        is_ref = pernum == 1
        # A_EXPRRP (microunit.CPSRelationshipCode): reference person 1, spouse 3,
        # own child 5, everyone else other-relative 10.
        exprrp = pd.Series(10, index=frame.index, dtype=int)
        exprrp[famrel == 2] = 3
        exprrp[famrel == 3] = 5
        exprrp[is_ref] = 1
        frame["A_EXPRRP"] = exprrp

        # A_MARITL: 1 married spouse present; 4 widowed; else 7 never-married.
        marital = pd.Series(7, index=frame.index, dtype=int)
        marital[spouse_num > 0] = 1
        if "is_surviving_spouse" in frame.columns:
            surviving = (
                pd.to_numeric(frame["is_surviving_spouse"], errors="coerce").fillna(0)
                > 0
            )
            marital[surviving & (spouse_num == 0)] = 4
        frame["A_MARITL"] = marital

        # PEPAR1/PEPAR2: a child's parents are heuristically the household
        # reference person (line 1) and that reference person's spouse.
        ref_spouse_line = frame.loc[is_ref].groupby(hh[is_ref])["A_SPOUSE"].first()
        frame["PEPAR1"] = 0
        frame["PEPAR2"] = 0
        is_child = famrel == 3
        frame.loc[is_child, "PEPAR1"] = 1
        frame.loc[is_child, "PEPAR2"] = (
            hh[is_child].map(ref_spouse_line).fillna(0).astype(int)
        )

        return frame

    def _microunit_cps_frame_from_normalized(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """PROTOTYPE (issue #115): synthesize microunit's CPS-like input contract
        from microplex's normalized person columns.

        ``microunit.construct_tax_units`` needs raw CPS columns (PH_SEQ/A_LINENO/
        A_AGE/A_MARITL/A_SPOUSE/PEPAR1/PEPAR2/A_EXPRRP); at PolicyEngine
        materialization microplex instead carries ``household_id``/``age``/
        ``relationship_to_head``. This builds the former from the latter, mirroring
        the ACS->CPS mapping microunit documents as the consumer's responsibility.

        .. warning::
            HEURISTIC AND UNVALIDATED. The ``relationship_to_head`` -> ``A_EXPRRP``
            and married -> ``A_MARITL`` maps are approximate, and ``PEPAR1``/
            ``PEPAR2`` are inferred by assuming a child's parents are the household
            head and spouse. The fidelity of these maps must be validated against
            the legacy reconstruction before this is trusted (see #115); it is
            gated OFF by default.

        Returns ``persons`` with the eight microunit CPS columns added, or ``None``
        if the prerequisite normalized columns are absent.
        """
        # Prefer the high-fidelity path when microplex carries the real CPS-derived
        # pointer fields (person_number is a 1-based within-household line number;
        # spouse_person_number a real spouse line pointer). Otherwise fall back to
        # the coarse relationship_to_head heuristic below.
        if {
            "person_id",
            "person_number",
            "family_relationship",
            "household_id",
            "age",
        }.issubset(persons.columns):
            return self._microunit_cps_frame_from_cps_fields(persons)

        required = {"person_id", "household_id", "age", "relationship_to_head"}
        if not required.issubset(persons.columns):
            return None

        # CPS A_EXPRRP recode (microunit.CPSRelationshipCode): 1 reference person,
        # 3 husband, 5 own child, 10 other relative.
        exprrp_by_rel = {0: 1, 1: 3, 2: 5, 3: 10}

        frame = persons.reset_index(drop=True).copy()
        rel = (
            pd.to_numeric(frame["relationship_to_head"], errors="coerce")
            .fillna(3)
            .astype(int)
        )
        age = pd.to_numeric(frame["age"], errors="coerce").fillna(0).astype(int)

        # Per-household line numbers (1-based, unique within household): head
        # first, then spouse, then everyone else oldest-first.
        frame = frame.assign(_rel=rel.to_numpy(), _age=age.to_numpy())
        frame = frame.sort_values(
            ["household_id", "_rel", "_age", "person_id"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
        frame["A_LINENO"] = frame.groupby("household_id", sort=False).cumcount() + 1
        # Guarantee exactly one household head (microunit requires a single
        # reference person per PH_SEQ, else it raises). After the head-first sort
        # the line-1 member is the most head-like; make it the head and demote
        # any other rows that mapped to head (multi-family / headless households).
        is_line1 = frame["A_LINENO"] == 1
        frame.loc[is_line1, "_rel"] = 0
        frame.loc[~is_line1 & (frame["_rel"] == 0), "_rel"] = 3
        frame["PH_SEQ"] = pd.to_numeric(frame["household_id"], errors="coerce").astype(
            np.int64
        )
        frame["A_AGE"] = frame["_age"]
        frame["A_EXPRRP"] = frame["_rel"].map(exprrp_by_rel).fillna(10).astype(int)

        # Head/spouse line numbers per household, for spouse pointers + marital.
        head_line = (
            frame.loc[frame["_rel"] == 0].groupby("household_id")["A_LINENO"].first()
        )
        spouse_line = (
            frame.loc[frame["_rel"] == 1].groupby("household_id")["A_LINENO"].first()
        )
        hh = frame["household_id"]
        is_head = frame["_rel"] == 0
        is_spouse = frame["_rel"] == 1
        is_child = frame["_rel"] == 2
        has_spouse = hh.map(spouse_line).notna()

        frame["A_SPOUSE"] = 0
        frame.loc[is_head, "A_SPOUSE"] = (
            hh[is_head].map(spouse_line).fillna(0).astype(int)
        )
        frame.loc[is_spouse, "A_SPOUSE"] = (
            hh[is_spouse].map(head_line).fillna(0).astype(int)
        )

        # A_MARITL: 1 = married, spouse present (head/spouse of a couple); else
        # 7 = never married. microunit only needs the married-vs-not distinction.
        frame["A_MARITL"] = 7
        frame.loc[(is_head | is_spouse) & has_spouse, "A_MARITL"] = 1

        # PEPAR1/PEPAR2: assume a child's parents are the household head + spouse.
        frame["PEPAR1"] = 0
        frame["PEPAR2"] = 0
        frame.loc[is_child, "PEPAR1"] = (
            hh[is_child].map(head_line).fillna(0).astype(int)
        )
        frame.loc[is_child, "PEPAR2"] = (
            hh[is_child].map(spouse_line).fillna(0).astype(int)
        )

        return frame.drop(columns=["_rel", "_age"])

    @staticmethod
    def _decode_microunit_bytes(value: Any) -> str:
        """Decode a ``microunit`` bytes-typed status/role into an upper string."""
        if isinstance(value, bytes):
            return value.decode()
        return str(value)

    def _microunit_reference_year(self, persons: pd.DataFrame) -> int:
        """Year passed to ``microunit`` for its dependency income thresholds.

        Prefers an explicit ``year``/``tax_year`` column when the frame carries
        one; otherwise falls back to the pipeline's configured reference year so
        the only year-dependent behavior (the qualifying-relative gross income
        limit) matches the rest of the pipeline. TODO(#113): thread the dataset
        reference year through entity construction explicitly.
        """
        for column in ("year", "tax_year"):
            if column in persons.columns:
                values = pd.to_numeric(persons[column], errors="coerce").dropna()
                if not values.empty:
                    return int(values.iloc[0])
        configured = getattr(self.config, "reference_year", None)
        if configured is not None:
            return int(configured)
        return 2024

    def _build_policyengine_tax_units_from_role_flags(
        self,
        persons: pd.DataFrame,
        *,
        start_tax_unit_id: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, set[Any]] | None:
        # Issue #113: when the frame carries microunit's CPS-style input
        # columns, delegate the reconstruction to microunit. Otherwise fall
        # through to the legacy role-flag reconstruction below (the current
        # production path, since these columns are not yet threaded through).
        microunit_result = self._build_policyengine_tax_units_via_microunit(
            persons,
            start_tax_unit_id=start_tax_unit_id,
        )
        if microunit_result is not None:
            return microunit_result

        role_columns = {
            "is_tax_unit_head",
            "is_tax_unit_spouse",
            "is_tax_unit_dependent",
        }
        if (
            not role_columns.issubset(persons.columns)
            or "person_id" not in persons.columns
        ):
            return None

        person_rows = persons.copy()
        raw_head_flag = self._role_flag_series(
            person_rows,
            "is_tax_unit_head",
        )
        raw_spouse_flag = self._role_flag_series(
            person_rows,
            "is_tax_unit_spouse",
        )
        raw_dependent_flag = self._role_flag_series(
            person_rows,
            "is_tax_unit_dependent",
        )
        (
            person_rows["_is_tax_unit_head_flag"],
            person_rows["_is_tax_unit_spouse_flag"],
            person_rows["_is_tax_unit_dependent_flag"],
        ) = self._resolve_tax_unit_role_flags(
            person_rows,
            head_flag=raw_head_flag,
            spouse_flag=raw_spouse_flag,
            dependent_flag=raw_dependent_flag,
        )

        tax_unit_rows: list[dict[str, Any]] = []
        person_to_tax_unit: dict[int, int] = {}
        role_households: set[Any] = set()
        next_tax_unit_id = int(start_tax_unit_id)

        for household_id, household_persons in person_rows.groupby(
            "household_id",
            sort=False,
        ):
            ordered = household_persons.sort_values(
                ["relationship_to_head", "age", "person_id"],
                ascending=[True, False, True],
            ).copy()
            ordered = self._cohere_tax_unit_role_flags_for_household(ordered)
            head_rows = ordered.loc[ordered["_is_tax_unit_head_flag"]]
            if head_rows.empty:
                continue

            head_ids = [int(person_id) for person_id in head_rows["person_id"].tolist()]
            head_to_spouses = self._assign_role_flag_spouses(ordered, head_ids)
            head_to_dependents = self._assign_role_flag_dependents(ordered, head_ids)
            assigned_person_ids: set[int] = set()

            for head_id in head_ids:
                spouse_ids = head_to_spouses.get(head_id, [])
                dependent_ids = head_to_dependents.get(head_id, [])
                unit_person_ids = list(
                    dict.fromkeys([head_id, *spouse_ids, *dependent_ids])
                )
                unit_persons = ordered.loc[
                    ordered["person_id"].astype(int).isin(unit_person_ids)
                ].copy()
                if unit_persons.empty:
                    continue
                global_tax_unit_id = next_tax_unit_id
                next_tax_unit_id += 1
                for person_id in unit_person_ids:
                    person_to_tax_unit[int(person_id)] = global_tax_unit_id
                    assigned_person_ids.add(int(person_id))

                filing_status = self._infer_role_flag_tax_unit_filing_status(
                    unit_persons,
                    head_id=head_id,
                    spouse_ids=spouse_ids,
                    dependent_ids=dependent_ids,
                )
                tax_unit_rows.append(
                    {
                        "tax_unit_id": global_tax_unit_id,
                        "household_id": int(household_id),
                        "filing_status": filing_status,
                        "member_ids": [int(person_id) for person_id in unit_person_ids],
                        "filer_ids": [head_id, *spouse_ids],
                        "dependent_ids": dependent_ids,
                        "n_dependents": len(dependent_ids),
                        "total_income": float(
                            pd.to_numeric(
                                unit_persons.get("income", 0.0),
                                errors="coerce",
                            )
                            .fillna(0.0)
                            .sum()
                        ),
                        "tax_liability": 0.0,
                        **self._aggregate_policyengine_tax_unit_input_columns(
                            unit_persons
                        ),
                    }
                )

            unassigned = [
                int(person_id)
                for person_id in ordered["person_id"].tolist()
                if int(person_id) not in assigned_person_ids
            ]
            for person_id in unassigned:
                unit_persons = ordered.loc[
                    ordered["person_id"].astype(int).eq(person_id)
                ].copy()
                global_tax_unit_id = next_tax_unit_id
                next_tax_unit_id += 1
                person_to_tax_unit[person_id] = global_tax_unit_id
                tax_unit_rows.append(
                    {
                        "tax_unit_id": global_tax_unit_id,
                        "household_id": int(household_id),
                        "filing_status": "SINGLE",
                        "member_ids": [person_id],
                        "filer_ids": [person_id],
                        "dependent_ids": [],
                        "n_dependents": 0,
                        "total_income": float(
                            pd.to_numeric(
                                unit_persons.get("income", 0.0),
                                errors="coerce",
                            )
                            .fillna(0.0)
                            .sum()
                        ),
                        "tax_liability": 0.0,
                        **self._aggregate_policyengine_tax_unit_input_columns(
                            unit_persons
                        ),
                    }
                )

            role_households.add(household_id)

        if not tax_unit_rows:
            return None

        result_persons = person_rows.loc[
            person_rows["household_id"].isin(role_households)
        ].copy()
        result_persons["tax_unit_id"] = result_persons["person_id"].map(
            person_to_tax_unit
        )
        result_persons = result_persons.drop(
            columns=[
                "_is_tax_unit_head_flag",
                "_is_tax_unit_spouse_flag",
                "_is_tax_unit_dependent_flag",
            ],
            errors="ignore",
        )
        return pd.DataFrame(tax_unit_rows), result_persons, role_households

    def _build_policyengine_tax_units_from_existing_ids(
        self,
        persons: pd.DataFrame,
        *,
        start_tax_unit_id: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, set[Any]] | None:
        if "tax_unit_id" not in persons.columns or "person_id" not in persons.columns:
            return None

        raw_tax_unit_id = pd.to_numeric(persons["tax_unit_id"], errors="coerce")
        if raw_tax_unit_id.isna().all():
            return None

        person_rows = persons.copy()
        household_has_complete_tax_unit_ids = (
            raw_tax_unit_id.notna()
            .groupby(person_rows["household_id"])
            .transform("all")
        )
        if not bool(household_has_complete_tax_unit_ids.any()):
            return None

        person_rows = person_rows.loc[household_has_complete_tax_unit_ids].copy()
        raw_tax_unit_id = raw_tax_unit_id.loc[person_rows.index]
        preserved_households = set(
            person_rows["household_id"].drop_duplicates().tolist()
        )
        tax_unit_key = pd.DataFrame(
            {
                "household_id": person_rows["household_id"],
                "tax_unit_id": raw_tax_unit_id,
            }
        )

        households_per_tax_unit = (
            tax_unit_key.assign(_household_id=person_rows["household_id"])
            .groupby("tax_unit_id")["_household_id"]
            .nunique()
        )
        if bool((households_per_tax_unit > 1).any()):
            normalized_tax_unit_id = pd.factorize(
                pd.MultiIndex.from_frame(tax_unit_key), sort=False
            )[0].astype(np.int64) + int(start_tax_unit_id)
            person_rows["tax_unit_id"] = normalized_tax_unit_id
        else:
            raw_tax_unit_id = raw_tax_unit_id.astype(np.int64)
            if int(start_tax_unit_id) == 0:
                person_rows["tax_unit_id"] = raw_tax_unit_id
            else:
                raw_min = int(raw_tax_unit_id.min()) if len(raw_tax_unit_id) else 0
                person_rows["tax_unit_id"] = (
                    raw_tax_unit_id - raw_min + int(start_tax_unit_id)
                ).astype(np.int64)

        tax_unit_rows: list[dict[str, Any]] = []
        for tax_unit_id, unit_persons in person_rows.groupby("tax_unit_id", sort=False):
            ordered = unit_persons.sort_values(
                ["relationship_to_head", "age", "person_id"],
                ascending=[True, False, True],
            ).reset_index(drop=True)
            filer_ids, dependent_ids = self._split_preserved_tax_unit_members(ordered)
            if not filer_ids:
                filer_ids = [int(ordered.iloc[0]["person_id"])]
                dependent_ids = [
                    int(person_id)
                    for person_id in ordered["person_id"].tolist()
                    if int(person_id) not in filer_ids
                ]
            filing_status = self._infer_preserved_tax_unit_filing_status(
                ordered,
                filer_ids=filer_ids,
                dependent_ids=dependent_ids,
            )
            tax_unit_rows.append(
                {
                    "tax_unit_id": int(tax_unit_id),
                    "household_id": int(ordered.iloc[0]["household_id"]),
                    "filing_status": filing_status,
                    "member_ids": [
                        int(person_id) for person_id in ordered["person_id"]
                    ],
                    "filer_ids": filer_ids,
                    "dependent_ids": dependent_ids,
                    "n_dependents": len(dependent_ids),
                    "total_income": float(
                        pd.to_numeric(ordered.get("income", 0.0), errors="coerce")
                        .fillna(0.0)
                        .sum()
                    ),
                    "tax_liability": 0.0,
                    **self._aggregate_policyengine_tax_unit_input_columns(ordered),
                }
            )

        return pd.DataFrame(tax_unit_rows), person_rows, preserved_households

    def _role_flag_series(self, frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).gt(0.5)

    def _resolve_tax_unit_role_flags(
        self,
        frame: pd.DataFrame,
        *,
        head_flag: pd.Series,
        spouse_flag: pd.Series,
        dependent_flag: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        relationship = (
            pd.to_numeric(frame["relationship_to_head"], errors="coerce")
            .fillna(-1)
            .astype(int)
            if "relationship_to_head" in frame.columns
            else pd.Series(-1, index=frame.index, dtype=int)
        )
        family_relationship = (
            pd.to_numeric(frame["family_relationship"], errors="coerce")
            .fillna(-1)
            .astype(int)
            if "family_relationship" in frame.columns
            else pd.Series(-1, index=frame.index, dtype=int)
        )
        head_hint = relationship.eq(0) | family_relationship.isin([0, 1])
        spouse_hint = relationship.eq(1) | family_relationship.eq(2)
        dependent_hint = relationship.isin([2, 3]) | family_relationship.isin([3, 4])

        resolved_dependent = (
            dependent_flag
            & (~spouse_flag | dependent_hint | ~spouse_hint)
            & (~head_flag | dependent_hint | ~head_hint)
        )
        resolved_spouse = (
            spouse_flag & ~resolved_dependent & (~head_flag | spouse_hint | ~head_hint)
        )
        resolved_head = head_flag & ~resolved_spouse & ~resolved_dependent
        return resolved_head, resolved_spouse, resolved_dependent

    def _cohere_tax_unit_role_flags_for_household(
        self,
        household_persons: pd.DataFrame,
    ) -> pd.DataFrame:
        if household_persons.empty:
            return household_persons

        result = household_persons.copy()
        relationship = (
            pd.to_numeric(result["relationship_to_head"], errors="coerce")
            .fillna(-1)
            .astype(int)
            if "relationship_to_head" in result.columns
            else pd.Series(-1, index=result.index, dtype=int)
        )
        family_relationship = (
            pd.to_numeric(result["family_relationship"], errors="coerce")
            .fillna(-1)
            .astype(int)
            if "family_relationship" in result.columns
            else pd.Series(-1, index=result.index, dtype=int)
        )
        age = (
            pd.to_numeric(result["age"], errors="coerce").fillna(0.0)
            if "age" in result.columns
            else pd.Series(0.0, index=result.index, dtype=float)
        )
        income = (
            pd.to_numeric(result["income"], errors="coerce").fillna(0.0)
            if "income" in result.columns
            else pd.Series(0.0, index=result.index, dtype=float)
        )
        head_hint = relationship.eq(0) | family_relationship.isin([0, 1])
        spouse_hint = relationship.eq(1) | family_relationship.eq(2)
        dependent_hint = relationship.isin([2, 3]) | family_relationship.isin([3, 4])

        head_flag = result["_is_tax_unit_head_flag"].astype(bool)
        spouse_flag = result["_is_tax_unit_spouse_flag"].astype(bool)
        dependent_flag = result["_is_tax_unit_dependent_flag"].astype(bool)

        rank = pd.Series(4, index=result.index, dtype=int)
        rank.loc[age.ge(18)] = 3
        rank.loc[head_flag] = 2
        rank.loc[head_hint] = 1
        rank.loc[head_flag & head_hint] = 0
        primary_index = (
            pd.DataFrame(
                {
                    "rank": rank,
                    "relationship": relationship.where(relationship.ge(0), 99),
                    "age": -age,
                    "person_id": pd.to_numeric(
                        result["person_id"],
                        errors="coerce",
                    ).fillna(0),
                },
                index=result.index,
            )
            .sort_values(["rank", "relationship", "age", "person_id"])
            .index[0]
        )

        coherent_head = pd.Series(False, index=result.index, dtype=bool)
        coherent_spouse = pd.Series(False, index=result.index, dtype=bool)
        coherent_dependent = pd.Series(False, index=result.index, dtype=bool)
        coherent_head.loc[primary_index] = True

        primary_person_number = self._household_role_person_number(
            result,
            primary_index,
        )
        primary_spouse_number = self._household_role_spouse_number(
            result,
            primary_index,
        )
        spouse_candidates = result.index[
            (result.index != primary_index) & ~dependent_flag
        ]
        spouse_index: Any | None = None
        if primary_spouse_number > 0:
            spouse_index = self._find_household_role_person_number_index(
                result,
                spouse_candidates,
                primary_spouse_number,
            )
            if (
                spouse_index is not None
                and primary_person_number > 0
                and self._household_role_spouse_number(result, spouse_index)
                not in {0, primary_person_number}
            ):
                spouse_index = None
        if spouse_index is None:
            spouse_pool = spouse_candidates[
                (
                    spouse_flag.loc[spouse_candidates]
                    | (
                        spouse_hint.loc[spouse_candidates]
                        & self._role_flag_series(result, "tax_unit_is_joint").loc[
                            spouse_candidates
                        ]
                    )
                )
            ]
            if len(spouse_pool):
                spouse_index = (
                    pd.DataFrame(
                        {
                            "source_spouse": ~spouse_flag.loc[spouse_pool],
                            "relationship": ~spouse_hint.loc[spouse_pool],
                            "age": -age.loc[spouse_pool],
                            "person_id": pd.to_numeric(
                                result.loc[spouse_pool, "person_id"],
                                errors="coerce",
                            ).fillna(0),
                        },
                        index=spouse_pool,
                    )
                    .sort_values(["source_spouse", "relationship", "age", "person_id"])
                    .index[0]
                )
        if spouse_index is not None:
            coherent_spouse.loc[spouse_index] = True

        available = ~(coherent_head | coherent_spouse)
        coherent_dependent.loc[
            available
            & (
                dependent_flag
                | (dependent_hint & (age.lt(24) | income.le(0.0)))
                | (spouse_hint & income.le(0.0))
            )
        ] = True

        available = ~(coherent_head | coherent_spouse | coherent_dependent)
        coherent_head.loc[available & age.ge(18) & (head_flag | income.gt(0.0))] = True

        coherent_dependent.loc[
            ~(coherent_head | coherent_spouse | coherent_dependent)
            & (age.lt(18) | dependent_hint | income.le(0.0))
        ] = True
        coherent_head.loc[~(coherent_head | coherent_spouse | coherent_dependent)] = (
            True
        )

        result["_is_tax_unit_head_flag"] = coherent_head
        result["_is_tax_unit_spouse_flag"] = coherent_spouse
        result["_is_tax_unit_dependent_flag"] = coherent_dependent
        return result

    def _household_role_person_number(
        self,
        household_persons: pd.DataFrame,
        index: Any,
    ) -> int:
        if "person_number" not in household_persons.columns:
            return 0
        value = pd.to_numeric(
            pd.Series([household_persons.loc[index, "person_number"]]),
            errors="coerce",
        ).fillna(0)
        return int(value.iloc[0])

    def _household_role_spouse_number(
        self,
        household_persons: pd.DataFrame,
        index: Any,
    ) -> int:
        if "spouse_person_number" not in household_persons.columns:
            return 0
        value = pd.to_numeric(
            pd.Series([household_persons.loc[index, "spouse_person_number"]]),
            errors="coerce",
        ).fillna(0)
        return int(value.iloc[0])

    def _find_household_role_person_number_index(
        self,
        household_persons: pd.DataFrame,
        candidate_indices: pd.Index,
        person_number: int,
    ) -> Any | None:
        if "person_number" not in household_persons.columns:
            return None
        person_numbers = pd.to_numeric(
            household_persons.loc[candidate_indices, "person_number"],
            errors="coerce",
        ).fillna(0)
        matches = person_numbers.index[person_numbers.astype(int).eq(person_number)]
        return matches[0] if len(matches) else None

    def _assign_role_flag_spouses(
        self,
        household_persons: pd.DataFrame,
        head_ids: list[int],
    ) -> dict[int, list[int]]:
        head_set = set(head_ids)
        assignments: dict[int, list[int]] = {head_id: [] for head_id in head_ids}
        spouse_rows = household_persons.loc[
            household_persons["_is_tax_unit_spouse_flag"]
        ]
        if spouse_rows.empty:
            return assignments

        person_number = (
            pd.to_numeric(
                household_persons.get("person_number"),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            if "person_number" in household_persons.columns
            else pd.Series(0, index=household_persons.index, dtype=int)
        )
        spouse_number = (
            pd.to_numeric(
                household_persons.get("spouse_person_number"),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            if "spouse_person_number" in household_persons.columns
            else pd.Series(0, index=household_persons.index, dtype=int)
        )
        head_by_person_number = {
            int(person_number.loc[index]): int(row["person_id"])
            for index, row in household_persons.iterrows()
            if int(row["person_id"]) in head_set and int(person_number.loc[index]) > 0
        }
        row_by_person_id = {
            int(row["person_id"]): index for index, row in household_persons.iterrows()
        }
        assigned_spouses: set[int] = set()

        for index, row in spouse_rows.iterrows():
            spouse_id = int(row["person_id"])
            pointed_head_id = head_by_person_number.get(int(spouse_number.loc[index]))
            if pointed_head_id is None:
                spouse_person_number = int(person_number.loc[index])
                for head_id in head_ids:
                    head_index = row_by_person_id.get(head_id)
                    if head_index is None:
                        continue
                    if int(spouse_number.loc[head_index]) == spouse_person_number:
                        pointed_head_id = head_id
                        break
            if pointed_head_id is None:
                continue
            if assignments[pointed_head_id]:
                continue
            assignments[pointed_head_id].append(spouse_id)
            assigned_spouses.add(spouse_id)

        unassigned_spouse_ids = [
            int(person_id)
            for person_id in spouse_rows["person_id"].tolist()
            if int(person_id) not in assigned_spouses
        ]
        heads_without_spouse = [
            head_id for head_id in head_ids if not assignments[head_id]
        ]
        for head_id, spouse_id in zip(
            heads_without_spouse,
            unassigned_spouse_ids,
            strict=False,
        ):
            assignments[head_id].append(spouse_id)

        return assignments

    def _assign_role_flag_dependents(
        self,
        household_persons: pd.DataFrame,
        head_ids: list[int],
    ) -> dict[int, list[int]]:
        assignments: dict[int, list[int]] = {head_id: [] for head_id in head_ids}
        dependent_rows = household_persons.loc[
            household_persons["_is_tax_unit_dependent_flag"]
        ].sort_values(["age", "person_id"], ascending=[True, True])
        if dependent_rows.empty:
            return assignments

        target_counts: dict[int, int] = {}
        if "tax_unit_count_dependents" in household_persons.columns:
            count_series = pd.to_numeric(
                household_persons["tax_unit_count_dependents"],
                errors="coerce",
            ).fillna(0)
            for head_id in head_ids:
                head_mask = household_persons["person_id"].astype(int).eq(head_id)
                if not bool(head_mask.any()):
                    target_counts[head_id] = 0
                    continue
                target_counts[head_id] = max(
                    0,
                    int(round(float(count_series.loc[head_mask].iloc[0]))),
                )
        else:
            target_counts = {head_id: 0 for head_id in head_ids}

        for _, dependent in dependent_rows.iterrows():
            dependent_id = int(dependent["person_id"])
            candidates = [
                head_id
                for head_id in head_ids
                if len(assignments[head_id]) < target_counts.get(head_id, 0)
            ]
            head_id = candidates[0] if candidates else head_ids[0]
            assignments[head_id].append(dependent_id)

        return assignments

    def _infer_role_flag_tax_unit_filing_status(
        self,
        unit_persons: pd.DataFrame,
        *,
        head_id: int,
        spouse_ids: list[int],
        dependent_ids: list[int],
    ) -> str:
        if spouse_ids:
            return "JOINT"

        head_rows = unit_persons.loc[unit_persons["person_id"].astype(int).eq(head_id)]
        if head_rows.empty:
            return "SINGLE"
        hinted_status = self._infer_single_filer_filing_status(
            head_rows.iloc[0],
            has_dependents=bool(dependent_ids),
        )
        if hinted_status is not None:
            return hinted_status
        return "SINGLE"

    def _aggregate_policyengine_tax_unit_input_columns(
        self,
        unit_persons: pd.DataFrame,
    ) -> dict[str, Any]:
        columns = (
            "domestic_production_ald",
            "health_savings_account_ald",
            "recapture_of_investment_credit",
            "self_employed_health_insurance_ald",
            "self_employed_pension_contribution_ald",
            "unrecaptured_section_1250_gain",
            "unreported_payroll_tax",
        )
        aggregated: dict[str, Any] = {}
        for column in columns:
            if column not in unit_persons.columns:
                continue
            values = pd.to_numeric(unit_persons[column], errors="coerce").fillna(0.0)
            nonzero_values = values.loc[~np.isclose(values.to_numpy(dtype=float), 0.0)]
            if len(nonzero_values) > 1 and nonzero_values.nunique(dropna=True) == 1:
                aggregated[column] = float(nonzero_values.iloc[0])
                continue
            aggregated[column] = float(values.sum())
        for child_count_column in ("eitc_children", "eitc_child_count"):
            if child_count_column not in unit_persons.columns:
                continue
            values = pd.to_numeric(
                unit_persons[child_count_column], errors="coerce"
            ).fillna(0.0)
            aggregated[EITC_TAKEUP_CHILD_COUNT_HELPER_COLUMN] = float(values.max())
            break
        employment_income = pd.to_numeric(
            unit_persons.get("employment_income", 0.0), errors="coerce"
        )
        if isinstance(employment_income, pd.Series):
            aggregated[VOLUNTARY_FILING_WAGE_INCOME_HELPER_COLUMN] = float(
                employment_income.fillna(0.0).clip(lower=0.0).sum()
            )
        age = pd.to_numeric(unit_persons.get("age", 0.0), errors="coerce").fillna(0.0)
        head_mask = self._normal_bool_series(
            unit_persons.get("is_tax_unit_head", False),
            index=unit_persons.index,
        )
        if not bool(head_mask.any()) and "relationship_to_head" in unit_persons.columns:
            head_mask = (
                pd.to_numeric(unit_persons["relationship_to_head"], errors="coerce")
                .fillna(-1)
                .eq(0)
            )
        head_age = age.loc[head_mask].iloc[0] if bool(head_mask.any()) else age.iloc[0]
        aggregated[VOLUNTARY_FILING_AGE_HEAD_HELPER_COLUMN] = float(head_age)
        for boolean_column in (
            "takes_up_aca_if_eligible",
            "takes_up_dc_ptc",
            "takes_up_eitc",
            "would_file_taxes_voluntarily",
        ):
            value = self._infer_policyengine_bool_for_group(
                unit_persons, boolean_column
            )
            if value is not None:
                aggregated[boolean_column] = value
        return aggregated

    def _infer_policyengine_bool_for_group(
        self,
        group_rows: pd.DataFrame,
        column: str,
    ) -> bool | None:
        if column in group_rows.columns:
            return bool(
                self._normal_bool_series(
                    group_rows[column], index=group_rows.index
                ).any()
            )
        return None

    def _attach_policyengine_aca_takeup(
        self,
        tax_units: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach eCPS-style ACA take-up input before PE materialization."""
        result = tax_units.copy()
        column = "takes_up_aca_if_eligible"
        if column in result.columns:
            result[column] = (
                pd.to_numeric(result[column], errors="coerce")
                .fillna(0.0)
                .ne(0.0)
                .astype(bool)
            )
            return result

        year = int(
            self.config.policyengine_dataset_year
            or self.config.policyengine_target_period
            or 2024
        )
        rate = _load_microplex_takeup_rate("aca", year)
        rng = _microplex_seeded_rng(column)
        result[column] = rng.random(len(result)) < rate
        return result

    def _attach_policyengine_tax_unit_takeup_inputs(
        self,
        tax_units: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach eCPS-style tax-unit stochastic inputs before materialization."""
        result = self._attach_policyengine_aca_takeup(tax_units)
        result = self._attach_policyengine_simple_tax_unit_takeup(
            result,
            column="takes_up_dc_ptc",
            rate_key="dc_ptc",
        )
        result = self._attach_policyengine_eitc_takeup(result)
        return self._attach_policyengine_voluntary_filing(result)

    def _attach_policyengine_simple_tax_unit_takeup(
        self,
        tax_units: pd.DataFrame,
        *,
        column: str,
        rate_key: str,
    ) -> pd.DataFrame:
        result = tax_units.copy()
        if column in result.columns:
            result[column] = self._normal_bool_series(
                result[column], index=result.index
            )
            return result

        year = self._policyengine_takeup_year()
        rate = _load_microplex_takeup_rate(rate_key, year)
        rng = _microplex_seeded_rng(column)
        result[column] = rng.random(len(result)) < rate
        return result

    def _attach_policyengine_eitc_takeup(
        self,
        tax_units: pd.DataFrame,
    ) -> pd.DataFrame:
        result = tax_units.copy()
        column = "takes_up_eitc"
        if column in result.columns:
            result[column] = self._normal_bool_series(
                result[column], index=result.index
            )
            return result

        year = self._policyengine_takeup_year()
        rates = _load_microplex_eitc_takeup_rates(year)
        child_count_column = (
            EITC_TAKEUP_CHILD_COUNT_HELPER_COLUMN
            if EITC_TAKEUP_CHILD_COUNT_HELPER_COLUMN in result.columns
            else "n_dependents"
        )
        raw_dependent_count = (
            result[child_count_column]
            if child_count_column in result.columns
            else pd.Series(0, index=result.index)
        )
        dependent_count = (
            pd.to_numeric(raw_dependent_count, errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=3)
            .astype(int)
        )
        takeup_rate = dependent_count.map(lambda count: rates.get(int(count), 0.85))
        rng = _microplex_seeded_rng(column)
        result[column] = rng.random(len(result)) < takeup_rate.to_numpy(dtype=float)
        return result

    def _attach_policyengine_voluntary_filing(
        self,
        tax_units: pd.DataFrame,
    ) -> pd.DataFrame:
        result = tax_units.copy()
        column = "would_file_taxes_voluntarily"
        if column in result.columns:
            result[column] = self._normal_bool_series(
                result[column], index=result.index
            )
            return result.drop(
                columns=[
                    EITC_TAKEUP_CHILD_COUNT_HELPER_COLUMN,
                    VOLUNTARY_FILING_AGE_HEAD_HELPER_COLUMN,
                    VOLUNTARY_FILING_WAGE_INCOME_HELPER_COLUMN,
                ],
                errors="ignore",
            )

        year = self._policyengine_takeup_year()
        rates = _load_microplex_voluntary_filing_rates(year)
        takes_up_eitc = self._normal_bool_series(
            result.get("takes_up_eitc", False),
            index=result.index,
        )
        child_count = self._tax_unit_child_count_for_takeup(result)
        wage_income = pd.to_numeric(
            result.get(
                VOLUNTARY_FILING_WAGE_INCOME_HELPER_COLUMN,
                pd.Series(0.0, index=result.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        age_head = pd.to_numeric(
            result.get(
                VOLUNTARY_FILING_AGE_HEAD_HELPER_COLUMN,
                pd.Series(0.0, index=result.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        takeup_rate = self._voluntary_filing_rate_by_tax_unit(
            rates,
            child_count=child_count,
            wage_income=wage_income,
            age_head=age_head,
        )
        rng = _microplex_seeded_rng(column)
        result[column] = (~takes_up_eitc.to_numpy(dtype=bool)) & (
            rng.random(len(result)) < takeup_rate.to_numpy(dtype=float)
        )
        result = result.drop(
            columns=[
                EITC_TAKEUP_CHILD_COUNT_HELPER_COLUMN,
                VOLUNTARY_FILING_AGE_HEAD_HELPER_COLUMN,
                VOLUNTARY_FILING_WAGE_INCOME_HELPER_COLUMN,
            ],
            errors="ignore",
        )
        return result

    def _tax_unit_child_count_for_takeup(self, tax_units: pd.DataFrame) -> pd.Series:
        child_count_column = (
            EITC_TAKEUP_CHILD_COUNT_HELPER_COLUMN
            if EITC_TAKEUP_CHILD_COUNT_HELPER_COLUMN in tax_units.columns
            else "n_dependents"
        )
        raw_child_count = (
            tax_units[child_count_column]
            if child_count_column in tax_units.columns
            else pd.Series(0, index=tax_units.index)
        )
        return (
            pd.to_numeric(raw_child_count, errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=3)
            .astype(int)
        )

    @staticmethod
    def _voluntary_filing_rate_by_tax_unit(
        rates: dict,
        *,
        child_count: pd.Series,
        wage_income: pd.Series,
        age_head: pd.Series,
    ) -> pd.Series:
        children_bin = np.where(
            child_count.to_numpy(dtype=int) > 0, "with_children", "no_children"
        )
        wage_values = wage_income.to_numpy(dtype=float)
        wage_bin = np.select(
            [wage_values <= 0.0, wage_values < 15_000.0, wage_values < 30_000.0],
            ["zero", "low", "medium"],
            default="high",
        )
        age_bin = np.where(
            age_head.to_numpy(dtype=float) >= 65.0, "age_65_plus", "under_65"
        )
        values = [
            rates.get(children, {})
            .get(wage, {})
            .get(age, DEFAULT_VOLUNTARY_FILING_RATE)
            for children, wage, age in zip(children_bin, wage_bin, age_bin, strict=True)
        ]
        return pd.Series(values, index=child_count.index, dtype=float)

    def _attach_policyengine_person_takeup_inputs(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach eCPS-style person stochastic inputs before materialization."""
        result = self._attach_policyengine_medicaid_takeup(persons)
        for column, rate_key in (
            ("takes_up_head_start_if_eligible", "head_start"),
            ("takes_up_early_head_start_if_eligible", "early_head_start"),
        ):
            result = self._attach_policyengine_simple_person_takeup(
                result,
                column=column,
                rate_key=rate_key,
            )
        return result

    def _attach_policyengine_simple_person_takeup(
        self,
        persons: pd.DataFrame,
        *,
        column: str,
        rate_key: str,
    ) -> pd.DataFrame:
        result = persons.copy()
        if column in result.columns:
            result[column] = self._normal_bool_series(
                result[column], index=result.index
            )
            return result

        year = self._policyengine_takeup_year()
        rate = _load_microplex_takeup_rate(rate_key, year)
        rng = _microplex_seeded_rng(column)
        result[column] = rng.random(len(result)) < rate
        return result

    def _attach_policyengine_medicaid_takeup(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame:
        result = persons.copy()
        column = "takes_up_medicaid_if_eligible"
        if column in result.columns:
            result[column] = self._normal_bool_series(
                result[column], index=result.index
            )
            return result

        year = self._policyengine_takeup_year()
        rates = _load_microplex_medicaid_takeup_rates(year)
        states = self._person_state_abbreviation(result)
        takeup_rate = states.map(
            lambda state: rates.get(state, DEFAULT_MEDICAID_TAKEUP_RATE)
        )
        rng = _microplex_seeded_rng(column)
        result[column] = rng.random(len(result)) < takeup_rate.to_numpy(dtype=float)
        return result

    def _attach_policyengine_wic_inputs(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame:
        result = persons.copy()
        category = self._policyengine_wic_category_for_takeup(result)
        year = self._policyengine_takeup_year()

        claim_column = "would_claim_wic"
        if claim_column in result.columns:
            result[claim_column] = self._normal_bool_series(
                result[claim_column],
                index=result.index,
            )
        else:
            claim_rates = _load_microplex_wic_takeup_rates(year)
            claim_rate = category.map(
                lambda value: claim_rates.get(str(value), 0.0)
            ).fillna(0.0)
            rng = _microplex_seeded_rng(claim_column)
            result[claim_column] = rng.random(len(result)) < claim_rate.to_numpy(
                dtype=float
            )

        risk_column = "is_wic_at_nutritional_risk"
        if risk_column in result.columns:
            result[risk_column] = self._normal_bool_series(
                result[risk_column],
                index=result.index,
            )
        else:
            risk_rates = _load_microplex_wic_nutritional_risk_rates(year)
            risk_rate = category.map(
                lambda value: risk_rates.get(str(value), 0.0)
            ).fillna(0.0)
            receives_wic = self._normal_bool_series(
                result.get("receives_wic", False),
                index=result.index,
            )
            rng = _microplex_seeded_rng(risk_column)
            result[risk_column] = receives_wic | (
                rng.random(len(result)) < risk_rate.to_numpy(dtype=float)
            )
        return result

    def _policyengine_wic_category_for_takeup(
        self,
        persons: pd.DataFrame,
    ) -> pd.Series:
        index = persons.index
        age = pd.to_numeric(
            persons.get("age", pd.Series(0.0, index=index)),
            errors="coerce",
        ).fillna(0.0)
        pregnant = self._normal_bool_series(
            persons.get("is_pregnant", False),
            index=index,
        )
        breastfeeding = self._normal_bool_series(
            persons.get("is_breastfeeding", False),
            index=index,
        )
        if "is_female" in persons.columns:
            female = self._normal_bool_series(persons["is_female"], index=index)
        elif "sex" in persons.columns:
            female = (
                pd.to_numeric(persons["sex"], errors="coerce")
                .fillna(0)
                .astype(int)
                .eq(2)
            )
        else:
            female = pd.Series(False, index=index)

        own_children = pd.to_numeric(
            persons.get("own_children_in_household", pd.Series(0, index=index)),
            errors="coerce",
        ).fillna(0.0)
        mother = breastfeeding | (female & own_children.gt(0))

        group_column = next(
            (
                column
                for column in ("family_id", "spm_unit_id", "household_id")
                if column in persons.columns
            ),
            None,
        )
        if group_column is None:
            min_age_group = age
        else:
            group_keys = persons[group_column].where(
                persons[group_column].notna(),
                pd.Series(np.arange(len(persons)), index=index),
            )
            min_age_group = age.groupby(group_keys, sort=False).transform("min")

        category = np.select(
            [
                pregnant.to_numpy(dtype=bool),
                (
                    mother.to_numpy(dtype=bool)
                    & breastfeeding.to_numpy(dtype=bool)
                    & min_age_group.lt(1.0).to_numpy(dtype=bool)
                ),
                (
                    mother.to_numpy(dtype=bool)
                    & min_age_group.lt(0.5).to_numpy(dtype=bool)
                ),
                age.lt(1.0).to_numpy(dtype=bool),
                age.lt(5.0).to_numpy(dtype=bool),
            ],
            [
                WIC_TAKEUP_CATEGORY_PREGNANT,
                WIC_TAKEUP_CATEGORY_BREASTFEEDING,
                WIC_TAKEUP_CATEGORY_POSTPARTUM,
                WIC_TAKEUP_CATEGORY_INFANT,
                WIC_TAKEUP_CATEGORY_CHILD,
            ],
            default=WIC_TAKEUP_CATEGORY_NONE,
        )
        return pd.Series(category, index=index, dtype="string")

    def _person_state_abbreviation(self, persons: pd.DataFrame) -> pd.Series:
        if "state" in persons.columns:
            state = persons["state"].astype("string").str.upper()
            known = set(STATE_FIPS.values())
            return state.where(state.isin(known), "CA").fillna("CA")
        if "state_code_str" in persons.columns:
            state = persons["state_code_str"].astype("string").str.upper()
            known = set(STATE_FIPS.values())
            return state.where(state.isin(known), "CA").fillna("CA")
        if "state_fips" in persons.columns:
            state_fips = (
                pd.to_numeric(persons["state_fips"], errors="coerce")
                .fillna(6)
                .astype(int)
            )
            return state_fips.map(lambda value: STATE_FIPS.get(int(value), "CA"))
        return pd.Series("CA", index=persons.index, dtype="string")

    def _attach_policyengine_spm_takeup_inputs(
        self,
        spm_units: pd.DataFrame,
    ) -> pd.DataFrame:
        result = self._attach_policyengine_snap_takeup(spm_units)
        return self._attach_policyengine_tanf_takeup(result)

    def _attach_policyengine_tanf_takeup(
        self,
        spm_units: pd.DataFrame,
    ) -> pd.DataFrame:
        result = spm_units.copy()
        column = "takes_up_tanf_if_eligible"
        if column in result.columns:
            result[column] = self._normal_bool_series(
                result[column], index=result.index
            )
            return result

        year = self._policyengine_takeup_year()
        rate = _load_microplex_takeup_rate("tanf", year)
        rng = _microplex_seeded_rng(column)
        result[column] = rng.random(len(result)) < rate
        return result

    def _policyengine_takeup_year(self) -> int:
        return int(
            self.config.policyengine_dataset_year
            or self.config.policyengine_target_period
            or 2024
        )

    @staticmethod
    def _normal_bool_series(value: Any, *, index: pd.Index) -> pd.Series:
        if isinstance(value, pd.Series):
            series = value.reindex(index)
        else:
            series = pd.Series(value, index=index)
        return pd.to_numeric(series, errors="coerce").fillna(0.0).ne(0.0).astype(bool)

    def _split_preserved_tax_unit_members(
        self,
        unit_persons: pd.DataFrame,
    ) -> tuple[list[int], list[int]]:
        relationship = pd.to_numeric(
            unit_persons.get("relationship_to_head"),
            errors="coerce",
        ).fillna(3)
        head_mask = relationship.eq(0)
        spouse_mask = relationship.eq(1)
        dependent_mask = relationship.eq(2)

        filer_ids: list[int] = []
        spouse_pair_ids = self._find_preserved_tax_unit_spouse_pair(unit_persons)
        if head_mask.any():
            head_id = int(unit_persons.loc[head_mask, "person_id"].iloc[0])
            filer_ids.append(head_id)
            if head_id in spouse_pair_ids:
                filer_ids.extend(
                    [
                        int(person_id)
                        for person_id in spouse_pair_ids
                        if int(person_id) != head_id
                    ]
                )
            elif (
                spouse_mask.any() and "spouse_person_number" not in unit_persons.columns
            ):
                filer_ids.append(
                    int(unit_persons.loc[spouse_mask, "person_id"].iloc[0])
                )
        elif spouse_pair_ids:
            pair_rows = unit_persons.loc[
                unit_persons["person_id"].astype(int).isin(spouse_pair_ids)
            ].copy()
            pair_rows["age"] = pd.to_numeric(
                pair_rows.get("age"), errors="coerce"
            ).fillna(0.0)
            filer_ids.extend(
                pair_rows.sort_values(["age", "person_id"], ascending=[False, True])[
                    "person_id"
                ]
                .astype(int)
                .tolist()[:2]
            )
        elif spouse_mask.any() and "spouse_person_number" not in unit_persons.columns:
            filer_ids.append(int(unit_persons.loc[spouse_mask, "person_id"].iloc[0]))
        if not filer_ids:
            adult_mask = (
                pd.to_numeric(
                    unit_persons.get("age"),
                    errors="coerce",
                )
                .fillna(0)
                .ge(18)
            )
            if adult_mask.any():
                filer_ids.append(int(unit_persons.loc[adult_mask, "person_id"].iloc[0]))
            else:
                filer_ids.append(int(unit_persons.iloc[0]["person_id"]))

        dependent_ids = [
            int(person_id)
            for person_id in unit_persons.loc[dependent_mask, "person_id"].tolist()
            if int(person_id) not in filer_ids
        ]
        if not dependent_ids:
            dependent_ids = [
                int(person_id)
                for person_id in unit_persons["person_id"].tolist()
                if int(person_id) not in filer_ids
            ]
        return filer_ids, dependent_ids

    def _find_preserved_tax_unit_spouse_pair(
        self,
        unit_persons: pd.DataFrame,
    ) -> list[int]:
        required_columns = {"person_number", "spouse_person_number", "person_id"}
        if not required_columns.issubset(unit_persons.columns):
            return []
        pairs: set[tuple[int, int]] = set()
        by_number = {
            int(person_number): {
                "person_id": int(person_id),
                "spouse_person_number": int(spouse_person_number),
                "age": float(age),
            }
            for person_number, spouse_person_number, person_id, age in unit_persons[
                ["person_number", "spouse_person_number", "person_id", "age"]
            ]
            .assign(
                age=lambda frame: pd.to_numeric(frame["age"], errors="coerce").fillna(
                    0.0
                ),
                spouse_person_number=lambda frame: pd.to_numeric(
                    frame["spouse_person_number"], errors="coerce"
                ).fillna(0),
                person_number=lambda frame: pd.to_numeric(
                    frame["person_number"], errors="coerce"
                ).fillna(0),
            )
            .itertuples(index=False, name=None)
        }
        for person_number, data in by_number.items():
            spouse_number = data["spouse_person_number"]
            if spouse_number <= 0:
                continue
            spouse = by_number.get(spouse_number)
            if spouse is None or spouse["spouse_person_number"] != person_number:
                continue
            pair = tuple(sorted((data["person_id"], spouse["person_id"])))
            pairs.add(pair)
        if not pairs:
            return []
        if len(pairs) == 1:
            return list(next(iter(pairs)))

        head_candidates = unit_persons.loc[
            pd.to_numeric(unit_persons.get("relationship_to_head"), errors="coerce")
            .fillna(3)
            .eq(0),
            "person_id",
        ].astype(int)
        if not head_candidates.empty:
            head_id = int(head_candidates.iloc[0])
            for pair in sorted(pairs):
                if head_id in pair:
                    return list(pair)
        best_pair = max(
            pairs,
            key=lambda pair: sum(
                by_number[number]["age"]
                for number in by_number
                if by_number[number]["person_id"] in pair
            ),
        )
        return list(best_pair)

    def _infer_preserved_tax_unit_filing_status(
        self,
        unit_persons: pd.DataFrame,
        *,
        filer_ids: list[int],
        dependent_ids: list[int],
    ) -> str:
        if "filing_status" in unit_persons.columns:
            filing_status_values = (
                unit_persons["filing_status"].dropna().astype(str).str.strip()
            )
            filing_status_values = filing_status_values[filing_status_values != ""]
            if not filing_status_values.empty:
                return self._normalize_policyengine_filing_status(
                    filing_status_values.iloc[0]
                )

        if len(filer_ids) >= 2:
            return "JOINT"

        filer_row = unit_persons.loc[unit_persons["person_id"] == filer_ids[0]].iloc[0]
        hinted_status = self._infer_single_filer_filing_status(
            filer_row,
            has_dependents=bool(dependent_ids),
        )
        return hinted_status or "SINGLE"

    def _apply_tax_unit_filing_status_hints(
        self,
        household_persons: pd.DataFrame,
        optimized_units: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not optimized_units or "person_id" not in household_persons.columns:
            return optimized_units

        person_lookup = household_persons.set_index("person_id", drop=False)
        updated_units: list[dict[str, Any]] = []
        for unit in optimized_units:
            unit_copy = dict(unit)
            filer_ids = [int(person_id) for person_id in unit_copy.get("filer_ids", [])]
            dependent_ids = [
                int(person_id) for person_id in unit_copy.get("dependent_ids", [])
            ]
            if len(filer_ids) == 2:
                separated_split = self._split_joint_tax_unit_for_separated_filers(
                    person_lookup,
                    filer_ids=filer_ids,
                    dependent_ids=dependent_ids,
                )
                if separated_split is not None:
                    updated_units.extend(separated_split)
                    continue
            if len(filer_ids) != 1:
                updated_units.append(unit_copy)
                continue
            filer_id = filer_ids[0]
            if filer_id not in person_lookup.index:
                updated_units.append(unit_copy)
                continue
            filer_row = person_lookup.loc[filer_id]
            hinted_status = self._infer_single_filer_filing_status(
                filer_row,
                has_dependents=bool(dependent_ids),
            )
            if hinted_status is not None:
                unit_copy["filing_status"] = hinted_status
            elif self._normalize_policyengine_filing_status(
                unit_copy.get("filing_status", "single")
            ) in {"HEAD_OF_HOUSEHOLD", "SEPARATE"}:
                unit_copy["filing_status"] = "SINGLE"
            updated_units.append(unit_copy)
        return updated_units

    def _split_joint_tax_unit_for_separated_filers(
        self,
        person_lookup: pd.DataFrame,
        *,
        filer_ids: list[int],
        dependent_ids: list[int],
    ) -> list[dict[str, Any]] | None:
        if len(filer_ids) != 2:
            return None
        if not all(filer_id in person_lookup.index for filer_id in filer_ids):
            return None

        filer_rows = person_lookup.loc[filer_ids]
        if isinstance(filer_rows, pd.Series):
            filer_rows = filer_rows.to_frame().T
        separated_mask = filer_rows.apply(
            lambda row: self._has_explicit_separation_evidence(row), axis=1
        )
        if not bool(
            separated_mask.any()
        ) and self._has_marriage_compatible_joint_evidence(filer_rows):
            return None

        primary_filer_id = self._select_primary_tax_unit_filer(
            filer_rows,
            fallback_id=filer_ids[0],
        )
        secondary_filer_id = next(
            filer_id for filer_id in filer_ids if filer_id != primary_filer_id
        )
        split_units: list[dict[str, Any]] = []
        for filer_id, unit_dependent_ids in (
            (primary_filer_id, dependent_ids),
            (secondary_filer_id, []),
        ):
            filer_row = person_lookup.loc[filer_id]
            total_income = float(
                pd.to_numeric(filer_row.get("income", 0.0), errors="coerce") or 0.0
            )
            if unit_dependent_ids:
                dependent_income = pd.to_numeric(
                    person_lookup.loc[unit_dependent_ids, "income"],
                    errors="coerce",
                ).fillna(0.0)
                total_income += float(dependent_income.sum())
            hinted_status = self._infer_single_filer_filing_status(
                filer_row,
                has_dependents=bool(unit_dependent_ids),
            )
            split_units.append(
                {
                    "filer_ids": [int(filer_id)],
                    "dependent_ids": [
                        int(person_id) for person_id in unit_dependent_ids
                    ],
                    "n_dependents": int(len(unit_dependent_ids)),
                    "total_income": total_income,
                    "tax_liability": 0.0,
                    "filing_status": hinted_status or "SINGLE",
                }
            )
        return split_units

    def _has_marriage_compatible_joint_evidence(
        self,
        filer_rows: pd.DataFrame,
    ) -> bool:
        if "marital_status" not in filer_rows.columns:
            return True
        marital_status = pd.to_numeric(
            pd.Series(filer_rows["marital_status"]),
            errors="coerce",
        )
        observed = marital_status.dropna().astype(int)
        if observed.empty:
            return True
        # CPS spouse-present statuses are the only strong evidence that a
        # spouse-coded pair should survive as one joint PE tax unit.
        return bool(observed.isin({1, 2}).all())

    def _has_explicit_separation_evidence(self, filer_row: pd.Series) -> bool:
        if bool(filer_row.get("is_separated", False)):
            return True
        filing_status_code = self._coerce_policyengine_status_code(
            filer_row.get("filing_status_code")
        )
        if filing_status_code == 3:
            return True
        marital_status = self._coerce_policyengine_status_code(
            filer_row.get("marital_status")
        )
        return marital_status == 6

    def _select_primary_tax_unit_filer(
        self,
        filer_rows: pd.DataFrame,
        *,
        fallback_id: int,
    ) -> int:
        relationship = pd.to_numeric(
            filer_rows.get("relationship_to_head"),
            errors="coerce",
        )
        if relationship is not None:
            head_candidates = filer_rows.loc[relationship.eq(0)]
            if not head_candidates.empty:
                return int(head_candidates.iloc[0]["person_id"])
        is_head = pd.to_numeric(
            filer_rows.get("is_head"),
            errors="coerce",
        )
        if is_head is not None:
            head_candidates = filer_rows.loc[is_head.fillna(0).astype(float) > 0.0]
            if not head_candidates.empty:
                return int(head_candidates.iloc[0]["person_id"])
        if fallback_id in filer_rows["person_id"].astype(int).tolist():
            return int(fallback_id)
        return int(filer_rows.iloc[0]["person_id"])

    def _infer_single_filer_filing_status(
        self,
        filer_row: pd.Series,
        *,
        has_dependents: bool,
    ) -> str | None:
        filing_status_code = self._coerce_policyengine_status_code(
            filer_row.get("filing_status_code")
        )
        if filing_status_code == 3:
            return "SEPARATE"
        if filing_status_code == 4:
            return "HEAD_OF_HOUSEHOLD"
        if filing_status_code == 5:
            return "SURVIVING_SPOUSE"

        marital_status = self._coerce_policyengine_status_code(
            filer_row.get("marital_status")
        )
        if marital_status == 6:
            return "SEPARATE"
        if marital_status == 4 and has_dependents:
            return "SURVIVING_SPOUSE"
        return None

    def _coerce_policyengine_status_code(self, value: Any) -> int | None:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return None
        return int(numeric)

    def _assign_family_and_spm_units(self, persons: pd.DataFrame) -> pd.DataFrame:
        """Assign family and SPM units, preserving authoritative IDs when present.

        NOT delegated to ``microunit`` in this pass (issue #113). At the pinned
        commit ``microunit.units.spm.assign_spm_partition`` is documented as "a
        conservative adapter, not yet the full Census-parity constructor" and is
        not exported from microunit's public API, and microunit has no
        family-unit constructor. The authoritative-ID fast path is preserved
        here. TODO(#113): delegate once microunit grows a Census-parity
        SPM/family constructor.
        """
        result = persons.copy()
        preserved_family_ids = self._normalized_complete_existing_group_ids(
            result,
            "family_id",
        )
        # SPM unit ids from the source are trustworthy and must survive synthesis
        # even when partially missing (a single missing id must not collapse the
        # whole frame to one SPM unit per household). Tax-unit ids, by contrast,
        # are reconstructed, not preserved (see _build_policyengine_tax_units).
        preserved_spm_unit_ids = self._preserve_present_group_ids(
            result,
            "spm_unit_id",
        )
        if preserved_family_ids is not None and preserved_spm_unit_ids is not None:
            result["family_id"] = preserved_family_ids
            result["spm_unit_id"] = preserved_spm_unit_ids
            return result

        family_ids: dict[int, int] = {}
        spm_unit_ids: dict[int, int] = {}
        next_family_id = 0
        next_spm_unit_id = 0

        for _, household_persons in result.groupby("household_id", sort=False):
            household_spm_id = next_spm_unit_id
            next_spm_unit_id += 1
            primary_mask = self._primary_family_member_mask(household_persons)
            if primary_mask.any():
                primary_family_id = next_family_id
                next_family_id += 1
            else:
                primary_family_id = None

            for _, row in household_persons.iterrows():
                spm_unit_ids[int(row.name)] = household_spm_id
                if primary_family_id is not None and bool(primary_mask.loc[row.name]):
                    family_ids[int(row.name)] = primary_family_id
                    continue

                family_ids[int(row.name)] = next_family_id
                next_family_id += 1

        result["family_id"] = (
            preserved_family_ids
            if preserved_family_ids is not None
            else result.index.map(family_ids).astype(np.int64)
        )
        result["spm_unit_id"] = (
            preserved_spm_unit_ids
            if preserved_spm_unit_ids is not None
            else result.index.map(spm_unit_ids).astype(np.int64)
        )
        return result

    def _primary_family_member_mask(
        self,
        household_persons: pd.DataFrame,
    ) -> pd.Series:
        """Identify people who belong to the household's primary family."""

        relationship_primary = household_persons["relationship_to_head"].isin({0, 1, 2})
        if "family_relationship" not in household_persons.columns:
            return relationship_primary

        family_relationship = pd.to_numeric(
            household_persons["family_relationship"],
            errors="coerce",
        )
        # CPS A_FAMREL is a family-membership code: 0 means not in a family;
        # positive values are reference person, spouse, child, or other relative.
        family_member = family_relationship.isin({1, 2, 3, 4})
        return relationship_primary | family_member

    def _assign_marital_units(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assign marital units, preserving authoritative IDs when present.

        NOT delegated to ``microunit`` in this pass (issue #113): microunit does
        not construct marital units at the pinned commit (filing status is its
        only marital-related output; there is no ``construct_marital_units``).
        The authoritative-ID fast path is preserved here. TODO(#113): revisit if
        microunit grows marital-unit support.
        """
        result = persons.copy()
        preserved_marital_unit_ids = self._normalized_complete_existing_group_ids(
            result,
            "marital_unit_id",
        )
        if preserved_marital_unit_ids is not None:
            result["marital_unit_id"] = preserved_marital_unit_ids
            return result

        marital_unit_by_person: dict[int, int] = {}
        next_marital_unit_id = 0

        for tax_unit_id, unit_persons in result.groupby("tax_unit_id", sort=False):
            _ = tax_unit_id
            filers = unit_persons[unit_persons["relationship_to_head"].isin({0, 1})]
            if len(filers) >= 2:
                marital_unit_id = next_marital_unit_id
                next_marital_unit_id += 1
                for person_id in filers.head(2)["person_id"].tolist():
                    marital_unit_by_person[int(person_id)] = marital_unit_id
            elif len(filers) == 1:
                marital_unit_by_person[int(filers.iloc[0]["person_id"])] = (
                    next_marital_unit_id
                )
                next_marital_unit_id += 1

            for person_id in unit_persons["person_id"].tolist():
                if int(person_id) in marital_unit_by_person:
                    continue
                marital_unit_by_person[int(person_id)] = next_marital_unit_id
                next_marital_unit_id += 1

        result["marital_unit_id"] = (
            result["person_id"].map(marital_unit_by_person).astype(np.int64)
        )
        return result

    def _assign_policyengine_household_head_flag(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame:
        result = persons.copy()
        derived = (
            pd.to_numeric(result["relationship_to_head"], errors="coerce")
            .fillna(-1)
            .eq(0)
        )
        if "is_household_head" not in result.columns:
            result["is_household_head"] = derived
            return result

        existing = pd.to_numeric(result["is_household_head"], errors="coerce")
        result["is_household_head"] = existing.where(existing.notna(), derived).gt(0.5)
        return result

    def _normalized_complete_existing_group_ids(
        self,
        persons: pd.DataFrame,
        id_column: str,
    ) -> pd.Series | None:
        if id_column not in persons.columns:
            return None
        raw_ids = persons[id_column]
        if raw_ids.isna().any():
            return None

        raw_key = raw_ids.astype("string")
        key = pd.DataFrame(
            {
                "household_id": persons["household_id"],
                id_column: raw_key,
            },
            index=persons.index,
        )
        raw_numeric = pd.to_numeric(raw_ids, errors="coerce")
        households_per_raw_id = key.groupby(id_column, dropna=False)[
            "household_id"
        ].nunique()
        must_factorize = raw_numeric.isna().any() or bool(
            households_per_raw_id.gt(1).any()
        )
        if must_factorize:
            return pd.Series(
                pd.factorize(pd.MultiIndex.from_frame(key), sort=False)[0].astype(
                    np.int64
                ),
                index=persons.index,
                name=id_column,
            )
        return raw_numeric.astype(np.int64).rename(id_column)

    def _preserve_present_group_ids(
        self,
        persons: pd.DataFrame,
        id_column: str,
    ) -> pd.Series | None:
        """Preserve existing per-person unit ids where present, regenerating only
        the rows that are missing one.

        Unlike :meth:`_normalized_complete_existing_group_ids` (which discards the
        whole column if *any* id is missing), this keeps the authoritative
        grouping for every row that carries an id and collapses rows with a
        missing id into a single per-household fallback unit. Used for SPM units,
        whose source ids are trustworthy and should survive synthesis even when
        partially missing (otherwise a single missing id drops the whole frame to
        one SPM unit per household). Returns ``None`` only when the column is
        absent or entirely empty.
        """
        if id_column not in persons.columns:
            return None
        raw_ids = persons[id_column]
        present = raw_ids.notna()
        if not present.any():
            return None
        hh = persons["household_id"]
        codes = pd.Series(-1, index=persons.index, dtype=np.int64)
        # Present rows: stable unit code from factorizing (household_id, real id).
        present_key = pd.MultiIndex.from_frame(
            pd.DataFrame({"hh": hh[present], "id": raw_ids[present].astype("string")})
        )
        codes.loc[present] = pd.factorize(present_key, sort=False)[0]
        if (~present).any():
            # Missing rows fold into their household's first present unit so they
            # never fabricate a spurious unit; households with no present id at
            # all get one fresh fallback unit each.
            first_present = codes[present].groupby(hh[present]).first()
            miss_hh = hh[~present]
            fallback = miss_hh.map(first_present)
            no_present = fallback.isna()
            if no_present.any():
                fresh = pd.factorize(miss_hh[no_present], sort=False)[0]
                fallback.loc[no_present] = fresh + (int(codes.max()) + 1)
            codes.loc[~present] = fallback.astype(np.int64).to_numpy()
        return codes.rename(id_column)

    def _collapse_group_table(
        self,
        persons: pd.DataFrame,
        id_column: str,
    ) -> pd.DataFrame:
        return (
            persons.groupby(id_column, as_index=False)
            .agg({"household_id": "first"})
            .astype({id_column: np.int64, "household_id": np.int64})
        )

    def _attach_spm_unit_source_columns(
        self,
        persons: pd.DataFrame,
        spm_units: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach observed SPM-unit inputs carried on CPS person rows."""
        if "spm_unit_id" not in persons.columns:
            return self._attach_policyengine_spm_takeup_inputs(spm_units)

        aggregation_by_column = {
            "receives_housing_assistance": "max",
            "takes_up_housing_assistance_if_eligible": "max",
            "takes_up_snap_if_eligible": "max",
            "takes_up_tanf_if_eligible": "max",
            "spm_unit_energy_subsidy": "first",
            "spm_unit_pre_subsidy_childcare_expenses": "first",
        }
        aggregations = {
            column: aggregation
            for column, aggregation in aggregation_by_column.items()
            if column in persons.columns and column not in spm_units.columns
        }
        if not aggregations:
            return self._attach_policyengine_spm_takeup_inputs(spm_units)

        source_values = persons.groupby("spm_unit_id", as_index=False).agg(aggregations)
        merged = spm_units.merge(source_values, on="spm_unit_id", how="left")
        return self._attach_policyengine_spm_takeup_inputs(merged)

    def _attach_policyengine_snap_takeup(
        self,
        spm_units: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach eCPS-style SNAP take-up input before PE materialization."""
        result = spm_units.copy()
        column = "takes_up_snap_if_eligible"
        if column in result.columns:
            result[column] = (
                pd.to_numeric(result[column], errors="coerce")
                .fillna(0.0)
                .ne(0.0)
                .astype(bool)
            )
            return result

        year = int(
            self.config.policyengine_dataset_year
            or self.config.policyengine_target_period
            or 2024
        )
        rate = _load_microplex_takeup_rate("snap", year)
        rng = _microplex_seeded_rng(column)
        result[column] = rng.random(len(result)) < rate
        return result

    def _normalize_relationship_to_head(self, persons: pd.DataFrame) -> pd.Series:
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
                    household_codes = set(
                        family_relationship.loc[member_index].tolist()
                    )
                    if 0 in household_codes:
                        # Some sources already use the optimizer's 0-based coding.
                        mapped = family_relationship.loc[member_index].map(
                            {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
                        )
                    else:
                        # CPS A_FAMREL is 1-based: 1=head, 2=spouse, 3=child, 4=other.
                        mapped = family_relationship.loc[member_index].map(
                            {1: 0, 2: 1, 3: 2, 4: 3}
                        )
                    family_normalized.loc[member_index] = mapped.fillna(3).astype(int)

        if "relationship_to_head" not in persons.columns:
            if family_normalized is not None:
                return self._repair_relationship_to_head(persons, family_normalized)
            if "is_spouse" in persons.columns or "is_dependent" in persons.columns:
                order = persons.groupby("household_id").cumcount()
                normalized = pd.Series(3, index=persons.index, dtype=int)
                normalized.loc[order == 0] = 0
                if "is_spouse" in persons.columns:
                    spouse_mask = (
                        pd.to_numeric(persons["is_spouse"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        > 0
                    )
                    normalized.loc[spouse_mask] = 1
                if "is_dependent" in persons.columns:
                    dependent_mask = (
                        pd.to_numeric(persons["is_dependent"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        > 0
                    )
                    normalized.loc[dependent_mask & ~normalized.eq(1)] = 2
                return self._repair_relationship_to_head(persons, normalized)
            order = persons.groupby("household_id").cumcount()
            normalized = order.map(lambda idx: 0 if idx == 0 else 3).astype(int)
            return self._repair_relationship_to_head(persons, normalized)

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
                    return self._repair_relationship_to_head(persons, family_normalized)
            return self._repair_relationship_to_head(persons, relationship)

        if unique_values.issubset({1, 2, 3, 4}):
            normalized = (
                relationship.map({1: 0, 2: 1, 3: 3, 4: 2}).fillna(3).astype(int)
            )
            return self._repair_relationship_to_head(persons, normalized)

        order = persons.groupby("household_id").cumcount()
        normalized = pd.Series(3, index=persons.index, dtype=int)
        normalized.loc[order == 0] = 0
        normalized.loc[(order == 1) & (persons["age"] >= 18)] = 1
        normalized.loc[persons["age"] < 18] = 2
        return self._repair_relationship_to_head(persons, normalized)

    def _repair_relationship_to_head(
        self,
        persons: pd.DataFrame,
        relationship: pd.Series,
    ) -> pd.Series:
        """Repair household relationship patterns so tax-unit construction has one clear head."""
        normalized = relationship.astype(int).copy()
        if "household_id" not in persons.columns:
            return normalized

        ages = pd.to_numeric(persons.get("age", 0), errors="coerce").fillna(0.0)
        grouped = persons.groupby("household_id", sort=False).groups
        for member_index in grouped.values():
            member_index = list(member_index)
            household_relationship = normalized.loc[member_index].copy()
            household_ages = ages.loc[member_index]

            head_index = household_relationship[
                household_relationship.eq(0)
            ].index.tolist()
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
                head_index = [keep_head]

            spouse_index = normalized.loc[member_index][
                normalized.loc[member_index].eq(1)
            ].index.tolist()
            if len(spouse_index) > 1:
                keep_spouse = max(
                    spouse_index, key=lambda index: household_ages.loc[index]
                )
                for index in spouse_index:
                    if index == keep_spouse:
                        continue
                    normalized.loc[index] = 3 if household_ages.loc[index] >= 19 else 2

        return normalized.astype(int)

    def _infer_policyengine_variable_bindings(
        self,
        tables: PolicyEngineUSEntityTableBundle,
    ) -> dict[str, PolicyEngineUSVariableBinding]:
        return infer_policyengine_us_variable_bindings(tables)

    def _filter_supported_policyengine_targets(
        self,
        targets: list[TargetSpec],
        tables: PolicyEngineUSEntityTableBundle,
        bindings: dict[str, PolicyEngineUSVariableBinding],
    ) -> list[TargetSpec]:
        return filter_supported_policyengine_us_targets(targets, tables, bindings)

    def _policyengine_variables_to_materialize(
        self,
        targets: list[TargetSpec],
        bindings: dict[str, PolicyEngineUSVariableBinding],
    ) -> set[str]:
        return policyengine_us_variables_to_materialize(targets, bindings)

    def _has_policyengine_entity_table(
        self,
        entity: EntityType,
        tables: PolicyEngineUSEntityTableBundle,
    ) -> bool:
        entity_tables = {
            EntityType.HOUSEHOLD: tables.households,
            EntityType.PERSON: tables.persons,
            EntityType.TAX_UNIT: tables.tax_units,
            EntityType.SPM_UNIT: tables.spm_units,
            EntityType.FAMILY: tables.families,
        }
        table = entity_tables.get(entity)
        return table is not None

    def _normalize_policyengine_filing_status(self, value: Any) -> str:
        normalized = str(value).strip().lower()
        mapping = {
            "single": "SINGLE",
            "married_joint": "JOINT",
            "married_filing_jointly": "JOINT",
            "joint": "JOINT",
            "married_filing_separately": "SEPARATE",
            "separate": "SEPARATE",
            "head_of_household": "HEAD_OF_HOUSEHOLD",
            "widow": "SURVIVING_SPOUSE",
            "qualifying_widow": "SURVIVING_SPOUSE",
            "surviving_spouse": "SURVIVING_SPOUSE",
        }
        return mapping.get(normalized, "SINGLE")

    def _augment_policyengine_person_inputs(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame:
        result = normalize_social_security_columns(normalize_dividend_columns(persons))
        zero = pd.Series(0.0, index=result.index, dtype=float)

        def first_present(*columns: str) -> pd.Series:
            for column in columns:
                if column in result.columns:
                    return (
                        pd.to_numeric(
                            result[column],
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .astype(float)
                    )
            return zero.copy()

        def first_nonzero_or_present(*columns: str) -> pd.Series:
            values = zero.copy()
            found = False
            for column in columns:
                if column not in result.columns:
                    continue
                candidate = (
                    pd.to_numeric(
                        result[column],
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .astype(float)
                )
                if not found:
                    values = candidate.copy()
                    found = True
                    continue
                values = values.where(values.ne(0.0), candidate)
            return values if found else zero.copy()

        def has_any(*columns: str) -> bool:
            return any(column in result.columns for column in columns)

        if "is_female" in result.columns:
            result["is_female"] = result["is_female"].fillna(False).astype(bool)
        elif "sex" in result.columns:
            sex = pd.to_numeric(result["sex"], errors="coerce").fillna(0).astype(int)
            result["is_female"] = sex.eq(2)

        if "cps_race" in result.columns:
            result["cps_race"] = (
                pd.to_numeric(result["cps_race"], errors="coerce").fillna(0).astype(int)
            )
        elif "race" in result.columns:
            result["cps_race"] = (
                pd.to_numeric(result["race"], errors="coerce").fillna(0).astype(int)
            )

        if "is_hispanic" in result.columns:
            result["is_hispanic"] = result["is_hispanic"].fillna(False).astype(bool)
        elif "hispanic" in result.columns:
            hispanic = pd.to_numeric(result["hispanic"], errors="coerce")
            observed_codes = set(hispanic.dropna().astype(int).unique().tolist())
            if observed_codes and observed_codes <= {1, 2}:
                result["is_hispanic"] = hispanic.fillna(0).astype(int).eq(1)
            else:
                result["is_hispanic"] = hispanic.fillna(0).astype(int).ne(0)

        marital_status = (
            pd.to_numeric(result["marital_status"], errors="coerce")
            if "marital_status" in result.columns
            else None
        )
        filing_status_code = (
            pd.to_numeric(result["filing_status_code"], errors="coerce")
            if "filing_status_code" in result.columns
            else None
        )
        filing_status_text = (
            result["filing_status"].astype(str).str.strip().str.upper()
            if "filing_status" in result.columns
            else None
        )

        if "is_separated" in result.columns:
            result["is_separated"] = result["is_separated"].fillna(False).astype(bool)
        elif marital_status is not None:
            result["is_separated"] = marital_status.fillna(0).astype(int).eq(6)
        elif filing_status_code is not None:
            result["is_separated"] = filing_status_code.fillna(0).astype(int).eq(3)
        elif filing_status_text is not None:
            result["is_separated"] = filing_status_text.eq("SEPARATE")

        if "is_surviving_spouse" in result.columns:
            result["is_surviving_spouse"] = (
                result["is_surviving_spouse"].fillna(False).astype(bool)
            )
        elif marital_status is not None:
            result["is_surviving_spouse"] = marital_status.fillna(0).astype(int).eq(4)
        elif filing_status_code is not None:
            result["is_surviving_spouse"] = (
                filing_status_code.fillna(0).astype(int).eq(5)
            )
        elif filing_status_text is not None:
            result["is_surviving_spouse"] = filing_status_text.eq("SURVIVING_SPOUSE")

        if "medicaid" in result.columns:
            result["medicaid"] = (
                pd.to_numeric(result["medicaid"], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
        if "medicaid_enrolled" in result.columns:
            result["medicaid_enrolled"] = (
                result["medicaid_enrolled"].fillna(False).astype(bool)
            )
        if "has_medicare" in result.columns:
            result["has_medicare"] = (
                pd.to_numeric(result["has_medicare"], errors="coerce")
                .fillna(0.0)
                .astype(float)
                .ne(0.0)
            )
        if "is_blind" in result.columns:
            result["is_blind"] = (
                pd.to_numeric(result["is_blind"], errors="coerce").fillna(0.0).ne(0.0)
            )
        elif "difficulty_seeing" in result.columns:
            result["is_blind"] = first_present("difficulty_seeing").gt(0.0)
        if "medicare_part_b_premiums" in result.columns:
            medicare_part_b_premiums = (
                pd.to_numeric(
                    result["medicare_part_b_premiums"],
                    errors="coerce",
                )
                .fillna(0.0)
                .clip(lower=0.0)
                .astype(float)
            )
            if "has_medicare" in result.columns:
                medicare_part_b_premiums = medicare_part_b_premiums.where(
                    result["has_medicare"],
                    0.0,
                )
            result["medicare_part_b_premiums"] = medicare_part_b_premiums

        if "takes_up_ssi_if_eligible" in result.columns:
            result["takes_up_ssi_if_eligible"] = (
                pd.to_numeric(
                    result["takes_up_ssi_if_eligible"],
                    errors="coerce",
                )
                .fillna(0.0)
                .ne(0.0)
            )
        elif "ssi_reported" in result.columns:
            result["takes_up_ssi_if_eligible"] = first_present("ssi_reported").gt(0.0)
        elif "ssi" in result.columns:
            result["takes_up_ssi_if_eligible"] = first_present("ssi").gt(0.0)

        known_nonemployment = (
            first_nonzero_or_present(
                "self_employment_income_before_lsr",
                "self_employment_income",
            )
            + first_nonzero_or_present("taxable_interest_income", "interest_income")
            + first_nonzero_or_present("ordinary_dividend_income", "dividend_income")
            + first_present("rental_income")
            + first_present("gross_social_security", "social_security")
            + first_present("ssi")
            + first_present("public_assistance")
            + first_nonzero_or_present("taxable_pension_income", "pension_income")
            + first_present("unemployment_compensation")
        )
        fallback_employment_income = (
            pd.to_numeric(result.get("income", zero), errors="coerce")
            .fillna(0.0)
            .astype(float)
            - known_nonemployment
        ).clip(lower=0.0)

        result["employment_income_before_lsr"] = (
            first_nonzero_or_present(
                "employment_income_before_lsr", "employment_income", "wage_income"
            )
            if has_any(
                "employment_income_before_lsr", "employment_income", "wage_income"
            )
            else fallback_employment_income
        )
        result["self_employment_income_before_lsr"] = first_nonzero_or_present(
            "self_employment_income_before_lsr",
            "self_employment_income",
        )
        result["taxable_interest_income"] = first_nonzero_or_present(
            "taxable_interest_income",
            "interest_income",
        )
        result["tax_exempt_interest_income"] = first_present(
            "tax_exempt_interest_income"
        )
        result["qualified_dividend_income"] = first_present(
            "qualified_dividend_income",
        ).clip(lower=0.0)
        result["non_qualified_dividend_income"] = first_present(
            "non_qualified_dividend_income",
        ).clip(lower=0.0)
        dividend_alias = first_nonzero_or_present(
            "ordinary_dividend_income",
            "dividend_income",
        ).clip(lower=0.0)
        result["ordinary_dividend_income"] = dividend_alias
        if has_any("qualified_dividend_income", "non_qualified_dividend_income"):
            dividend_total = (
                result["qualified_dividend_income"]
                + result["non_qualified_dividend_income"]
            ).clip(lower=0.0)
            result["ordinary_dividend_income"] = dividend_total.where(
                dividend_total.ne(0.0),
                dividend_alias,
            )
            result["dividend_income"] = result["ordinary_dividend_income"]
        else:
            result = normalize_dividend_columns(result)

        result["short_term_capital_gains"] = first_present("short_term_capital_gains")
        result["non_sch_d_capital_gains"] = first_present(
            "non_sch_d_capital_gains",
            "capital_gains_distributions",
        )
        result["long_term_capital_gains_before_response"] = (
            first_nonzero_or_present(
                "long_term_capital_gains_before_response",
                "long_term_capital_gains",
                "capital_gains",
            )
            if has_any(
                "long_term_capital_gains_before_response",
                "long_term_capital_gains",
                "capital_gains",
            )
            else zero.copy()
        )
        result["partnership_s_corp_income"] = first_present("partnership_s_corp_income")
        result["partnership_se_income"] = first_present("partnership_se_income")
        result["estate_income"] = first_present("estate_income")
        result["farm_income"] = first_present("farm_income")
        result["farm_operations_income"] = first_present("farm_operations_income")
        result["farm_rent_income"] = first_present("farm_rent_income")
        result["rental_income"] = first_present("rental_income")
        result["health_savings_account_ald"] = first_present(
            "health_savings_account_ald"
        )
        result["self_employed_health_insurance_ald"] = first_present(
            "self_employed_health_insurance_ald"
        )
        result["self_employed_pension_contribution_ald"] = first_present(
            "self_employed_pension_contribution_ald"
        )
        result["taxable_private_pension_income"] = first_present(
            "taxable_private_pension_income",
            "taxable_pension_income",
            "pension_income",
        )
        result["taxable_public_pension_income"] = first_present(
            "taxable_public_pension_income"
        )
        result["tax_exempt_private_pension_income"] = first_present(
            "tax_exempt_private_pension_income"
        )
        result["tax_exempt_public_pension_income"] = first_present(
            "tax_exempt_public_pension_income"
        )
        result["social_security_retirement"] = (
            social_security_retirement_compatible_amount(result)
        )
        result["social_security_disability"] = first_present(
            "social_security_disability"
        )
        result["social_security_survivors"] = first_present("social_security_survivors")
        result["social_security_dependents"] = first_present(
            "social_security_dependents"
        )
        result["unemployment_compensation"] = first_present("unemployment_compensation")
        result["state_income_tax_reported"] = first_present(
            "state_income_tax_reported",
            "state_income_tax_paid",
        )
        result["student_loan_interest"] = first_present("student_loan_interest")
        return result

    def _resolve_policyengine_tax_benefit_system(self) -> Any:
        simulation_cls = self.config.policyengine_simulation_cls
        if simulation_cls is None:
            import policyengine_us

            return getattr(policyengine_us.system, "system", policyengine_us.system)

        tax_benefit_system = getattr(simulation_cls, "tax_benefit_system", None)
        if tax_benefit_system is None:
            tax_benefit_system = getattr(simulation_cls, "system", None)
        if tax_benefit_system is not None:
            return getattr(tax_benefit_system, "system", tax_benefit_system)
        raise ValueError(
            "policyengine_simulation_cls must expose a tax_benefit_system or system attribute"
        )


def build_us_microplex(
    persons: pd.DataFrame,
    households: pd.DataFrame,
    config: USMicroplexBuildConfig | None = None,
) -> USMicroplexBuildResult:
    """Convenience wrapper for the US microplex pipeline."""
    pipeline = USMicroplexPipeline(config)
    return pipeline.build(persons, households)


@dataclass
class USMicroplexRecalibrateResult:
    """Output of ``recalibrate_policyengine_us_from_checkpoint``.

    Narrower than ``USMicroplexBuildResult`` because synthesis state is
    unavailable when resuming: no ``seed_data``, no ``synthesizer``, no
    source frames. Only calibration output is populated.
    """

    config: USMicroplexBuildConfig
    loaded_stage: str
    checkpoint_path: Path
    policyengine_tables: PolicyEngineUSEntityTableBundle
    calibrated_data: pd.DataFrame
    calibration_summary: dict[str, Any]


def recalibrate_policyengine_us_from_checkpoint(
    config: USMicroplexBuildConfig,
    checkpoint_path: str | Path,
) -> USMicroplexRecalibrateResult:
    """Load a saved pipeline checkpoint and rerun calibration against it.

    Use for fast iteration on calibration config (backend, lambda
    schedule, targets) without paying the ~11 h synthesis + donor
    imputation cost that produced the bundle. Both
    ``post_imputation`` and ``post_microsim`` checkpoints are
    supported: the latter skips microsim too because
    ``infer_policyengine_us_variable_bindings`` picks up the
    materialized target vars as columns on the bundle, so
    ``policyengine_us_variables_to_materialize`` returns an empty set
    and ``_resolve_policyengine_calibration_targets`` short-circuits
    past the materialization call.
    """
    checkpoint_path = Path(checkpoint_path)
    bundle, metadata = load_us_pipeline_checkpoint(checkpoint_path)
    stage = metadata.get("stage")
    if stage not in {"post_imputation", "post_microsim"}:
        raise ValueError(
            f"Cannot resume from checkpoint stage {stage!r}; expected "
            "'post_imputation' or 'post_microsim'."
        )

    pipeline = USMicroplexPipeline(config)
    policyengine_tables, calibrated_data, calibration_summary = (
        pipeline.calibrate_policyengine_tables(bundle)
    )
    return USMicroplexRecalibrateResult(
        config=config,
        loaded_stage=stage,
        checkpoint_path=checkpoint_path,
        policyengine_tables=policyengine_tables,
        calibrated_data=calibrated_data,
        calibration_summary=calibration_summary,
    )
