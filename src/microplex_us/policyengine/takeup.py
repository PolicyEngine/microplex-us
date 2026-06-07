"""Deterministic PolicyEngine-US take-up input generation."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

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
DEFAULT_PREGNANCY_RATE = 0.041
EITC_TAKEUP_CHILD_COUNT_HELPER_COLUMN = "_mp_eitc_child_count_for_takeup"
VOLUNTARY_FILING_AGE_HEAD_HELPER_COLUMN = "_mp_voluntary_filing_age_head"
VOLUNTARY_FILING_WAGE_INCOME_HELPER_COLUMN = "_mp_voluntary_filing_wage_income"

STATE_FIPS_TO_ABBR = {
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

TAX_UNIT_TAKEUP_FEATURES = frozenset(
    {
        "takes_up_aca_if_eligible",
        "takes_up_dc_ptc",
        "takes_up_eitc",
        "would_file_taxes_voluntarily",
    }
)
PERSON_TAKEUP_FEATURES = frozenset(
    {
        "takes_up_early_head_start_if_eligible",
        "takes_up_head_start_if_eligible",
        "takes_up_medicaid_if_eligible",
        "takes_up_medicare_if_eligible",
        "takes_up_ssi_if_eligible",
        "would_claim_wic",
    }
)
SPM_UNIT_TAKEUP_FEATURES = frozenset(
    {
        "takes_up_housing_assistance_if_eligible",
        "takes_up_snap_if_eligible",
        "takes_up_tanf_if_eligible",
    }
)


def rerandomize_policyengine_us_takeup_frames(
    *,
    persons: pd.DataFrame | None,
    tax_units: pd.DataFrame | None,
    spm_units: pd.DataFrame | None,
    features: Iterable[str],
    year: int,
) -> tuple[
    pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, tuple[str, ...]
]:
    """Regenerate requested PolicyEngine-US take-up inputs where supported."""
    requested = tuple(dict.fromkeys(str(feature) for feature in features))
    unsupported: list[str] = []

    tax_units_out = tax_units.copy() if tax_units is not None else None
    tax_features = [
        feature for feature in requested if feature in TAX_UNIT_TAKEUP_FEATURES
    ]
    if tax_features and tax_units_out is None:
        unsupported.extend(tax_features)
    elif tax_units_out is not None:
        tax_units_out = _rerandomize_tax_unit_takeup_features(
            tax_units_out,
            features=tax_features,
            year=year,
        )

    persons_out = persons.copy() if persons is not None else None
    person_features = [
        feature for feature in requested if feature in PERSON_TAKEUP_FEATURES
    ]
    if person_features and persons_out is None:
        unsupported.extend(person_features)
    elif persons_out is not None:
        persons_out, person_unsupported = _rerandomize_person_takeup_features(
            persons_out,
            features=person_features,
            year=year,
        )
        unsupported.extend(person_unsupported)

    spm_units_out = spm_units.copy() if spm_units is not None else None
    spm_features = [
        feature for feature in requested if feature in SPM_UNIT_TAKEUP_FEATURES
    ]
    if spm_features and spm_units_out is None:
        unsupported.extend(spm_features)
    elif spm_units_out is not None:
        spm_units_out, spm_unsupported = _rerandomize_spm_unit_takeup_features(
            spm_units_out,
            features=spm_features,
            year=year,
        )
        unsupported.extend(spm_unsupported)

    known = TAX_UNIT_TAKEUP_FEATURES | PERSON_TAKEUP_FEATURES | SPM_UNIT_TAKEUP_FEATURES
    unsupported.extend(feature for feature in requested if feature not in known)
    return persons_out, tax_units_out, spm_units_out, tuple(dict.fromkeys(unsupported))


def _rerandomize_tax_unit_takeup_features(
    tax_units: pd.DataFrame,
    *,
    features: Iterable[str],
    year: int,
) -> pd.DataFrame:
    requested = set(features)
    result = tax_units.copy()
    if "takes_up_aca_if_eligible" in requested:
        result = _set_scalar_takeup(
            result,
            column="takes_up_aca_if_eligible",
            rate=_load_microplex_takeup_rate("aca", year),
        )
    if "takes_up_dc_ptc" in requested:
        result = _set_scalar_takeup(
            result,
            column="takes_up_dc_ptc",
            rate=_load_microplex_takeup_rate("dc_ptc", year),
        )
    if "takes_up_eitc" in requested:
        result = _set_eitc_takeup(result, year=year)
    if "would_file_taxes_voluntarily" in requested:
        if "takes_up_eitc" not in result.columns:
            result = _set_eitc_takeup(result, year=year)
        result = _set_voluntary_filing(result, year=year)
    return result


def _rerandomize_person_takeup_features(
    persons: pd.DataFrame,
    *,
    features: Iterable[str],
    year: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    requested = set(features)
    unsupported: list[str] = []
    result = persons.copy()
    if "takes_up_medicaid_if_eligible" in requested:
        rates = _load_microplex_medicaid_takeup_rates(year)
        states = _person_state_abbreviation(result)
        takeup_rate = states.map(
            lambda state: rates.get(state, DEFAULT_MEDICAID_TAKEUP_RATE)
        )
        rng = _microplex_seeded_rng("takes_up_medicaid_if_eligible")
        result["takes_up_medicaid_if_eligible"] = rng.random(
            len(result)
        ) < takeup_rate.to_numpy(dtype=float)
    if "takes_up_head_start_if_eligible" in requested:
        result = _set_scalar_takeup(
            result,
            column="takes_up_head_start_if_eligible",
            rate=_load_microplex_takeup_rate("head_start", year),
        )
    if "takes_up_early_head_start_if_eligible" in requested:
        result = _set_scalar_takeup(
            result,
            column="takes_up_early_head_start_if_eligible",
            rate=_load_microplex_takeup_rate("early_head_start", year),
        )
    if "takes_up_ssi_if_eligible" in requested:
        result, supported = _set_ssi_takeup_if_available(result)
        if not supported:
            unsupported.append("takes_up_ssi_if_eligible")
    if "takes_up_medicare_if_eligible" in requested:
        if "takes_up_medicare_if_eligible" in result.columns:
            result["takes_up_medicare_if_eligible"] = _normal_bool_series(
                result["takes_up_medicare_if_eligible"],
                index=result.index,
            )
        else:
            unsupported.append("takes_up_medicare_if_eligible")
    if "would_claim_wic" in requested:
        result = _set_wic_takeup_inputs(result, year=year)
    return result, tuple(unsupported)


def _rerandomize_spm_unit_takeup_features(
    spm_units: pd.DataFrame,
    *,
    features: Iterable[str],
    year: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    requested = set(features)
    unsupported: list[str] = []
    result = spm_units.copy()
    if "takes_up_snap_if_eligible" in requested:
        result = _set_scalar_takeup(
            result,
            column="takes_up_snap_if_eligible",
            rate=_load_microplex_takeup_rate("snap", year),
        )
    if "takes_up_tanf_if_eligible" in requested:
        result = _set_scalar_takeup(
            result,
            column="takes_up_tanf_if_eligible",
            rate=_load_microplex_takeup_rate("tanf", year),
        )
    if "takes_up_housing_assistance_if_eligible" in requested:
        if "takes_up_housing_assistance_if_eligible" in result.columns:
            result["takes_up_housing_assistance_if_eligible"] = _normal_bool_series(
                result["takes_up_housing_assistance_if_eligible"],
                index=result.index,
            )
        else:
            unsupported.append("takes_up_housing_assistance_if_eligible")
    return result, tuple(unsupported)


def _set_scalar_takeup(
    frame: pd.DataFrame,
    *,
    column: str,
    rate: float,
) -> pd.DataFrame:
    result = frame.copy()
    rng = _microplex_seeded_rng(column)
    result[column] = rng.random(len(result)) < float(rate)
    return result


def _set_eitc_takeup(
    tax_units: pd.DataFrame,
    *,
    year: int,
) -> pd.DataFrame:
    result = tax_units.copy()
    rates = _load_microplex_eitc_takeup_rates(year)
    child_count = _tax_unit_child_count_for_takeup(result)
    takeup_rate = child_count.map(lambda count: rates.get(int(count), 0.85))
    rng = _microplex_seeded_rng("takes_up_eitc")
    result["takes_up_eitc"] = rng.random(len(result)) < takeup_rate.to_numpy(
        dtype=float
    )
    return result


def _set_voluntary_filing(
    tax_units: pd.DataFrame,
    *,
    year: int,
) -> pd.DataFrame:
    result = tax_units.copy()
    rates = _load_microplex_voluntary_filing_rates(year)
    takes_up_eitc = _normal_bool_series(
        result.get("takes_up_eitc", False),
        index=result.index,
    )
    child_count = _tax_unit_child_count_for_takeup(result)
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
    takeup_rate = _voluntary_filing_rate_by_tax_unit(
        rates,
        child_count=child_count,
        wage_income=wage_income,
        age_head=age_head,
    )
    rng = _microplex_seeded_rng("would_file_taxes_voluntarily")
    result["would_file_taxes_voluntarily"] = (~takes_up_eitc.to_numpy(dtype=bool)) & (
        rng.random(len(result)) < takeup_rate.to_numpy(dtype=float)
    )
    return result.drop(
        columns=[
            EITC_TAKEUP_CHILD_COUNT_HELPER_COLUMN,
            VOLUNTARY_FILING_AGE_HEAD_HELPER_COLUMN,
            VOLUNTARY_FILING_WAGE_INCOME_HELPER_COLUMN,
        ],
        errors="ignore",
    )


def _tax_unit_child_count_for_takeup(tax_units: pd.DataFrame) -> pd.Series:
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


def _voluntary_filing_rate_by_tax_unit(
    rates: dict,
    *,
    child_count: pd.Series,
    wage_income: pd.Series,
    age_head: pd.Series,
) -> pd.Series:
    children_bin = np.where(
        child_count.to_numpy(dtype=int) > 0,
        "with_children",
        "no_children",
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
        rates.get(children, {}).get(wage, {}).get(age, DEFAULT_VOLUNTARY_FILING_RATE)
        for children, wage, age in zip(children_bin, wage_bin, age_bin, strict=True)
    ]
    return pd.Series(values, index=child_count.index, dtype=float)


def _set_ssi_takeup_if_available(persons: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    result = persons.copy()
    if "takes_up_ssi_if_eligible" in result.columns:
        result["takes_up_ssi_if_eligible"] = _normal_bool_series(
            result["takes_up_ssi_if_eligible"],
            index=result.index,
        )
        return result, True
    if "ssi_reported" in result.columns:
        result["takes_up_ssi_if_eligible"] = _nonzero_series(result["ssi_reported"])
        return result, True
    if "ssi" in result.columns:
        result["takes_up_ssi_if_eligible"] = _nonzero_series(result["ssi"])
        return result, True
    return result, False


def _set_wic_takeup_inputs(
    persons: pd.DataFrame,
    *,
    year: int,
) -> pd.DataFrame:
    result = persons.copy()
    if "is_pregnant" in result.columns:
        result["is_pregnant"] = _normal_bool_series(
            result["is_pregnant"], index=result.index
        )
    else:
        result = _set_pregnancy_inputs(result, year=year)

    category = _wic_category_for_takeup(result)
    claim_rates = _load_microplex_wic_takeup_rates(year)
    claim_rate = category.map(lambda value: claim_rates.get(str(value), 0.0)).fillna(
        0.0
    )
    rng = _microplex_seeded_rng("would_claim_wic")
    result["would_claim_wic"] = rng.random(len(result)) < claim_rate.to_numpy(
        dtype=float
    )

    risk_rates = _load_microplex_wic_nutritional_risk_rates(year)
    risk_rate = category.map(lambda value: risk_rates.get(str(value), 0.0)).fillna(0.0)
    receives_wic = _normal_bool_series(
        result.get("receives_wic", False),
        index=result.index,
    )
    rng = _microplex_seeded_rng("is_wic_at_nutritional_risk")
    result["is_wic_at_nutritional_risk"] = receives_wic | (
        rng.random(len(result)) < risk_rate.to_numpy(dtype=float)
    )
    return result


def _set_pregnancy_inputs(
    persons: pd.DataFrame,
    *,
    year: int,
) -> pd.DataFrame:
    result = persons.copy()
    index = result.index
    age = pd.to_numeric(
        result.get("age", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    if "is_female" in result.columns:
        female = _normal_bool_series(result["is_female"], index=index)
    elif "sex" in result.columns:
        female = (
            pd.to_numeric(result["sex"], errors="coerce").fillna(0).astype(int).eq(2)
        )
    else:
        female = pd.Series(False, index=index)

    rates = _load_microplex_pregnancy_rates(year)
    states = _person_state_abbreviation(result)
    pregnancy_rate = states.map(
        lambda state: rates.get(str(state).upper(), DEFAULT_PREGNANCY_RATE)
    ).fillna(DEFAULT_PREGNANCY_RATE)
    eligible = female & age.ge(15.0) & age.le(44.0)
    rng = _microplex_seeded_rng("is_pregnant")
    result["is_pregnant"] = eligible.to_numpy(dtype=bool) & (
        rng.random(len(result)) < pregnancy_rate.to_numpy(dtype=float)
    )
    return result


def _wic_category_for_takeup(persons: pd.DataFrame) -> pd.Series:
    index = persons.index
    age = pd.to_numeric(
        persons.get("age", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    pregnant = _normal_bool_series(persons.get("is_pregnant", False), index=index)
    breastfeeding = _normal_bool_series(
        persons.get("is_breastfeeding", False),
        index=index,
    )
    if "is_female" in persons.columns:
        female = _normal_bool_series(persons["is_female"], index=index)
    elif "sex" in persons.columns:
        female = (
            pd.to_numeric(persons["sex"], errors="coerce").fillna(0).astype(int).eq(2)
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
            (mother.to_numpy(dtype=bool) & min_age_group.lt(0.5).to_numpy(dtype=bool)),
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


def _person_state_abbreviation(persons: pd.DataFrame) -> pd.Series:
    if "state" in persons.columns:
        state = persons["state"].astype("string").str.upper()
        known = set(STATE_FIPS_TO_ABBR.values())
        return state.where(state.isin(known), "CA").fillna("CA")
    if "state_code_str" in persons.columns:
        state = persons["state_code_str"].astype("string").str.upper()
        known = set(STATE_FIPS_TO_ABBR.values())
        return state.where(state.isin(known), "CA").fillna("CA")
    if "state_fips" in persons.columns:
        state_fips = (
            pd.to_numeric(persons["state_fips"], errors="coerce").fillna(6).astype(int)
        )
        return state_fips.map(lambda value: STATE_FIPS_TO_ABBR.get(int(value), "CA"))
    return pd.Series("CA", index=persons.index, dtype="string")


def _nonzero_series(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").fillna(0.0).ne(0.0)


def _normal_bool_series(value, *, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.reindex(index)
    else:
        series = pd.Series(value, index=index)
    return pd.to_numeric(series, errors="coerce").fillna(0.0).ne(0.0).astype(bool)


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
    variable_name: str, *, salt: str | None = None
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
    _ = year
    return dict(DEFAULT_MEDICAID_TAKEUP_RATES_BY_STATE)


def _load_microplex_eitc_takeup_rates(year: int) -> dict[int, float]:
    _ = year
    return dict(DEFAULT_EITC_TAKEUP_RATES_BY_CHILDREN)


def _load_microplex_voluntary_filing_rates(year: int) -> dict:
    _ = year
    return {
        children: {wage: dict(age_rates) for wage, age_rates in wage_rates.items()}
        for children, wage_rates in DEFAULT_VOLUNTARY_FILING_RATES.items()
    }


def _load_microplex_wic_takeup_rates(year: int) -> dict[str, float]:
    _ = year
    return dict(DEFAULT_WIC_TAKEUP_RATES)


def _load_microplex_wic_nutritional_risk_rates(year: int) -> dict[str, float]:
    _ = year
    return dict(DEFAULT_WIC_NUTRITIONAL_RISK_RATES)


def _load_microplex_pregnancy_rates(year: int) -> dict[str, float]:
    _ = year
    try:
        from policyengine_us_data.db.etl_pregnancy import get_state_pregnancy_rates

        rates = get_state_pregnancy_rates()
    except Exception:
        LOGGER.warning(
            "Failed to load state pregnancy rates; using national fallback",
            exc_info=True,
        )
        return {}

    return {str(state).upper(): float(rate) for state, rate in rates.items()}
