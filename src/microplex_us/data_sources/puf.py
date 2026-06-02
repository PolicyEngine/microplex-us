"""IRS Public Use File (PUF) loader, processing, and source-provider wrapper.

Downloads PUF from HuggingFace, uprates 2015 → target year,
and maps to common variable schema for multi-survey fusion.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

from microplex_us.data_sources.cps import load_cps_asec
from microplex_us.data_sources.share_imputation import (
    GroupedShareModel,
    fit_grouped_share_model,
    predict_grouped_component_shares,
)
from microplex_us.pipelines.pe_native_scores import (
    resolve_policyengine_us_data_repo_root,
)
from microplex_us.source_manifests import load_us_source_manifest
from microplex_us.source_registry import resolve_source_variable_capabilities
from microplex_us.variables import normalize_dividend_columns

try:
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

PUF_VARIABLE_MAP = {
    column_spec.raw_column: column_spec.canonical_name
    for column_spec in load_us_source_manifest("puf")
    .observation_for(EntityType.TAX_UNIT)
    .columns
}

PUF_UPRATING_MODE_INTERPOLATED = "interpolated"
PUF_UPRATING_MODE_PE_SOI = "pe_soi"

PE_ITMDED_GROW_RATE = 0.02
PE_PUF_SOI_END_YEAR = 2021
PE_UPRATING_FACTOR_ALIASES = {
    "weight": "household_weight",
    "gross_social_security": "social_security",
}
PE_SOI_TO_PUF_STRAIGHT_RENAMES = {
    "employment_income": "E00200",
    "capital_gains_distributions": "E01100",
    "taxable_interest_income": "E00300",
    "exempt_interest": "E00400",
    "ordinary_dividends": "E00600",
    "qualified_dividends": "E00650",
    "ira_distributions": "E01400",
    "total_pension_income": "E01500",
    "taxable_pension_income": "E01700",
    "unemployment_compensation": "E02300",
    "total_social_security": "E02400",
    "taxable_social_security": "E02500",
    "medical_expense_deductions_uncapped": "E17500",
    "itemized_state_income_tax_deductions": "E18400",
    "itemized_real_estate_tax_deductions": "E18500",
    "interest_paid_deductions": "E19200",
    "charitable_contributions_deductions": "E19800",
}
PE_SOI_TO_PUF_POS_ONLY_RENAMES = {
    "business_net_profits": "E00900",
    "capital_gains_gross": "E01000",
    "partnership_and_s_corp_income": "E26270",
}
PE_SOI_TO_PUF_NEG_ONLY_RENAMES = {
    "business_net_losses": "E00900",
    "capital_gains_losses": "E01000",
    "partnership_and_s_corp_losses": "E26270",
}
PUF_AGGREGATE_RECIDS = (999996, 999997, 999998, 999999)
PUF_SYNTHETIC_RECID_START = 1_000_000
PUF_AGGREGATE_SCREENED_FIELDS = (
    "E00200",  # Wages
    "P23250",  # Long-term capital gains
    "P22250",  # Short-term capital gains
    "E00650",  # Qualified dividends
    "E00300",  # Taxable interest
    "E26270",  # Partnership / S-corp
    "E00900",  # Business income
    "E02100",  # Farm income
    "E00400",  # Tax-exempt interest
    "E00600",  # Ordinary dividends
)
PUF_AGGREGATE_BUCKET_BOUNDS = {
    999996: (-np.inf, 0.0),
    999997: (0.0, 10_000_000.0),
    999998: (10_000_000.0, 100_000_000.0),
    999999: (100_000_000.0, np.inf),
}
PUF_AGGREGATE_STRUCTURAL_COLUMNS = ("MARS", "XTOT", "DSI", "EIC")
_PUF_AMOUNT_COLUMN_PATTERN = re.compile(r"^(?:[EPT]\d+|S\d{5})$")
_PUF_AGGREGATE_AGI_CAP_100M_PLUS = 1_250_000_000.0
_PUF_AGGREGATE_MAX_AGI_DOMINANCE = 0.20
_PUF_AGGREGATE_SELECTION_POWER = 24
_PUF_NUMERIC_TOL = 1e-9
PE_PUF_REMAINING_RAW_COLUMNS = (
    "E03500",
    "E00800",
    "E20500",
    "E32800",
    "E20100",
    "E03240",
    "E03400",
    "E03220",
    "E26390",
    "E26400",
    "T27800",
    "E27200",
    "E03290",
    "P23250",
    "E24518",
    "E20400",
    "E26270",
    "E03230",
    "E25850",
    "E25860",
    "E00900",
    "E03270",
    "E03300",
    "P22250",
    "E03210",
    "E03150",
    "E24515",
    "E07300",
    "E62900",
    "E01200",
    "E00700",
    "E58990",
    "E07400",
    "E07600",
    "E11200",
    "E87521",
    "E07260",
    "E09900",
    "P08000",
    "E07240",
    "E09700",
    "E09800",
)

# SOI growth factors for uprating 2015 → 2024
# Based on IRS SOI aggregate growth rates
# These should be updated with actual SOI data
UPRATING_FACTORS = {
    "employment_income": 1.45,  # ~4.5% annual wage growth
    "self_employment_income": 1.35,
    "farm_income": 1.20,
    "taxable_interest_income": 2.50,  # Interest rates rose significantly
    "tax_exempt_interest_income": 1.80,
    "ordinary_dividend_income": 1.60,
    "qualified_dividend_income": 1.60,
    "short_term_capital_gains": 1.80,
    "long_term_capital_gains": 2.20,  # Stock market growth
    "non_sch_d_capital_gains": 1.80,
    "partnership_s_corp_income": 1.50,
    "rental_income_positive": 1.40,
    "rental_income_negative": 1.40,
    "ira_distributions": 1.60,
    "total_pension_income": 1.40,
    "taxable_pension_income": 1.40,
    "gross_social_security": 1.45,
    "taxable_social_security": 1.45,
    "unemployment_compensation": 0.30,  # Down from COVID peak
    "alimony_income": 0.50,  # Declining due to tax law change
    "medical_expense_agi_floor": 1.50,
    "state_income_tax_paid": 1.40,
    "real_estate_tax_paid": 1.35,
    "mortgage_interest_paid": 1.30,
    "charitable_cash": 1.40,
    "charitable_noncash": 1.40,
    "student_loan_interest": 1.20,
}

MINIMUM_SOCIAL_SECURITY_RETIREMENT_AGE = 62
SOCIAL_SECURITY_SHARE_AGE_BINS = (-np.inf, 18.0, 30.0, 45.0, 62.0, 75.0, np.inf)
SOCIAL_SECURITY_SHARE_AGE_LABELS = (
    "under_18",
    "18_to_29",
    "30_to_44",
    "45_to_61",
    "62_to_74",
    "75_plus",
)
SOCIAL_SECURITY_SHARE_EXPLICIT_COMPONENTS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
)
SOCIAL_SECURITY_SHARE_IMPLICIT_COMPONENT = "social_security_dependents"
SOCIAL_SECURITY_SHARE_COMPONENTS = (
    *SOCIAL_SECURITY_SHARE_EXPLICIT_COMPONENTS,
    SOCIAL_SECURITY_SHARE_IMPLICIT_COMPONENT,
)
SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE = "grouped_share"
SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF = "pe_qrf"
SOCIAL_SECURITY_SPLIT_STRATEGY_AGE_HEURISTIC = "age_heuristic"
PE_STYLE_SOCIAL_SECURITY_QRF_PREDICTORS = (
    "age",
    "is_male",
    "tax_unit_is_joint",
    "is_tax_unit_head",
    "is_tax_unit_dependent",
)
MIN_PE_STYLE_SOCIAL_SECURITY_QRF_TRAINING_RECORDS = 100
PE_PUF_PERSON_EXPANSION_RANDOM_SEED = 64

JOINT_HEAD_SHARE_ALLOCATION = {
    "employment_income": 0.6,
    "self_employment_income": 0.6,
}

JOINT_EQUAL_SHARE_ALLOCATION = (
    "farm_income",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "ordinary_dividend_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "short_term_capital_gains",
    "long_term_capital_gains",
    "non_sch_d_capital_gains",
    "partnership_s_corp_income",
    "rental_income",
    "ira_distributions",
    "total_pension_income",
    "taxable_pension_income",
    "gross_social_security",
    "taxable_social_security",
    "unemployment_compensation",
    "alimony_income",
    "medical_expense_agi_floor",
    "state_income_tax_paid",
    "real_estate_tax_paid",
    "mortgage_interest_paid",
    "charitable_cash",
    "charitable_noncash",
    "ira_deduction",
    "student_loan_interest",
)

PUF_DEMOGRAPHIC_HELPER_COLUMNS = (
    "_puf_recid",
    "_puf_agerange",
    "_puf_earnsplit",
    "_puf_gender",
    "_puf_agedp1",
    "_puf_agedp2",
    "_puf_agedp3",
)

PUF_PERSON_EXPANSION_PRESERVE_COLUMNS = {
    "weight",
    "household_weight",
    "year",
    "household_id",
    "state_fips",
    "tenure",
    "filing_status",
    "filing_status_code",
    "exemptions_count",
    "eitc_children",
    "ctc_children",
    "age",
    "is_male",
    "employment_status",
    "income",
    "interest_income",
    "dividend_income",
    "capital_gains",
    "pension_income",
    "social_security",
    "social_security_retirement",
    *PUF_DEMOGRAPHIC_HELPER_COLUMNS,
}

MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS = {
    "health_insurance_premiums_without_medicare_part_b": 0.453,
    "other_medical_expenses": 0.325,
    "medicare_part_b_premiums": 0.137,
    "over_the_counter_health_expenses": 0.085,
}


@dataclass(frozen=True)
class PEStyleQRFShareModel:
    """PE-style QRF share model for PUF Social Security components."""

    predictors: tuple[str, ...]
    component_columns: tuple[str, ...]
    share_prediction_columns: tuple[str, ...]
    fitted_model: Any


SocialSecurityShareModel = GroupedShareModel | PEStyleQRFShareModel


@dataclass(frozen=True)
class PEStyleQRFImputationModel:
    """PE-style QRF imputation model for direct PUF variable imputation."""

    predictors: tuple[str, ...]
    imputed_variable: str
    fitted_model: Any


PUF_DEMOGRAPHIC_VARIABLES = (
    "AGEDP1",
    "AGEDP2",
    "AGEDP3",
    "AGERANGE",
    "EARNSPLIT",
    "GENDER",
)

PUF_DEMOGRAPHIC_PREDICTORS = (
    "E00200",
    "MARS",
    "DSI",
    "EIC",
    "XTOT",
)


def download_puf(cache_dir: Path | None = None) -> Path:
    """Download PUF from HuggingFace.

    Returns path to downloaded CSV file.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "microplex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    puf_path = cache_dir / "puf_2015.csv"
    demo_path = cache_dir / "demographics_2015.csv"

    # Prefer an already-present local copy over any remote resolution.
    if puf_path.exists():
        return puf_path, demo_path

    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub required: pip install huggingface_hub")

    # Download PUF 2015
    puf_path = hf_hub_download(
        repo_id="policyengine/irs-soi-puf",
        filename="puf_2015.csv",
        repo_type="model",
        local_dir=cache_dir,
    )

    # Download demographics file
    demo_path = hf_hub_download(
        repo_id="policyengine/irs-soi-puf",
        filename="demographics_2015.csv",
        repo_type="model",
        local_dir=cache_dir,
    )

    return Path(puf_path), Path(demo_path)


def _resolve_policyengine_repo_local_puf_paths(
    policyengine_us_data_repo: str | Path | None,
) -> tuple[Path, Path | None] | None:
    """Resolve raw PUF CSVs from a local policyengine-us-data checkout."""

    if policyengine_us_data_repo is None:
        return None
    try:
        repo_root = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    except FileNotFoundError:
        return None
    candidate_dirs = (
        repo_root / "policyengine_us_data" / "storage",
        repo_root / "data" / "raw",
    )
    for candidate_dir in candidate_dirs:
        puf_path = candidate_dir / "puf_2015.csv"
        demographics_path = candidate_dir / "demographics_2015.csv"
        if puf_path.exists():
            return puf_path, (demographics_path if demographics_path.exists() else None)
    return None


def _puf_aggregate_amount_columns(columns: pd.Index | list[str]) -> list[str]:
    return [column for column in columns if _PUF_AMOUNT_COLUMN_PATTERN.match(column)]


def _puf_aggregate_bucket_mask(df: pd.DataFrame, recid: int) -> pd.Series:
    if "E00100" not in df.columns:
        return pd.Series(True, index=df.index)
    agi = pd.to_numeric(df["E00100"], errors="coerce").fillna(0.0)
    lower, upper = PUF_AGGREGATE_BUCKET_BOUNDS.get(recid, (-np.inf, np.inf))
    return agi.ge(lower) & agi.lt(upper)


def _puf_aggregate_eligibility_scores(
    df: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> pd.Series:
    reference_frame = df if reference is None else reference
    present_fields = [
        field
        for field in PUF_AGGREGATE_SCREENED_FIELDS
        if field in df.columns and field in reference_frame.columns
    ]
    if not present_fields:
        return pd.Series(0.0, index=df.index, dtype=float)

    scores = np.zeros(len(df), dtype=float)
    for raw_field in present_fields:
        values = pd.to_numeric(df[raw_field], errors="coerce").fillna(0.0)
        reference_values = pd.to_numeric(
            reference_frame[raw_field], errors="coerce"
        ).fillna(0.0)
        field_scores = np.zeros(len(df), dtype=float)

        positive_mask = values.gt(0)
        reference_positive = np.sort(
            reference_values[reference_values.gt(0)].to_numpy()
        )
        if bool(positive_mask.any()) and len(reference_positive) > 0:
            field_scores[positive_mask.to_numpy()] = np.searchsorted(
                reference_positive,
                values[positive_mask].to_numpy(),
                side="right",
            ) / len(reference_positive)

        negative_mask = values.lt(0)
        reference_negative = np.sort(
            (-reference_values[reference_values.lt(0)]).to_numpy()
        )
        if bool(negative_mask.any()) and len(reference_negative) > 0:
            negative_scores = np.searchsorted(
                reference_negative,
                (-values[negative_mask]).to_numpy(),
                side="right",
            ) / len(reference_negative)
            field_scores[negative_mask.to_numpy()] = np.maximum(
                field_scores[negative_mask.to_numpy()],
                negative_scores,
            )

        scores = np.maximum(scores, field_scores)
    return pd.Series(scores, index=df.index, dtype=float)


def _choose_puf_aggregate_synthetic_count(pop_weight: float) -> int:
    total_weight = max(1, int(round(pop_weight)))
    target_count = max(20, round(pop_weight / 10))
    return int(min(40, total_weight, target_count))


def _assign_puf_aggregate_weights(
    pop_weight: float,
    n_records: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total_weight = max(1, int(round(pop_weight)))
    n_records = max(1, min(int(n_records), total_weight))
    weights = np.ones(n_records, dtype=int)
    remainder = total_weight - n_records
    if remainder > 0:
        base_extra = remainder // n_records
        weights += base_extra
        leftover = remainder - base_extra * n_records
        if leftover:
            weights[rng.choice(n_records, size=leftover, replace=False)] += 1
    return weights


def _project_puf_weighted_sum_to_bounds(
    values: np.ndarray,
    weights: np.ndarray,
    target_total: float,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iter: int = 50,
) -> np.ndarray:
    projected = np.clip(values.astype(float), lower, upper)

    for _ in range(max_iter):
        residual = float(target_total - np.dot(projected, weights))
        if abs(residual) <= 1e-6:
            return projected

        slack = upper - projected if residual > 0 else projected - lower
        free = slack > _PUF_NUMERIC_TOL
        if not bool(free.any()):
            break

        basis = np.abs(projected[free])
        if basis.sum() <= _PUF_NUMERIC_TOL:
            basis = np.ones(free.sum(), dtype=float)
        denom = float(np.dot(weights[free], basis))
        if denom <= _PUF_NUMERIC_TOL:
            basis = np.ones(free.sum(), dtype=float)
            denom = float(weights[free].sum())

        delta = residual * basis / denom
        if residual > 0:
            delta = np.minimum(delta, slack[free])
        else:
            delta = -np.minimum(-delta, slack[free])
        projected[free] += delta
        projected = np.clip(projected, lower, upper)

    return projected


def _allocate_puf_weighted_values(
    base_values: np.ndarray,
    weights: np.ndarray,
    target_total: float,
    lower: np.ndarray | float | None = None,
    upper: np.ndarray | float | None = None,
) -> np.ndarray:
    base_values = np.asarray(base_values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(base_values)
    if abs(target_total) <= 1e-6:
        return np.zeros(n, dtype=float)

    if target_total > 0 and np.any(base_values > 0):
        active = base_values > 0
    elif target_total < 0 and np.any(base_values < 0):
        active = base_values < 0
    elif np.any(np.abs(base_values) > _PUF_NUMERIC_TOL):
        active = np.abs(base_values) > _PUF_NUMERIC_TOL
    else:
        active = np.ones(n, dtype=bool)

    allocated = np.zeros(n, dtype=float)
    magnitudes = np.abs(base_values[active])
    if magnitudes.sum() <= _PUF_NUMERIC_TOL:
        magnitudes = np.ones(active.sum(), dtype=float)
    denom = float(np.dot(weights[active], magnitudes))
    if denom <= _PUF_NUMERIC_TOL:
        magnitudes = np.ones(active.sum(), dtype=float)
        denom = float(weights[active].sum())

    allocated[active] = np.sign(target_total) * magnitudes * abs(target_total) / denom
    if lower is None and upper is None:
        return allocated

    if lower is None:
        lower_array = np.full(n, -np.inf, dtype=float)
    elif np.isscalar(lower):
        lower_array = np.full(n, float(lower), dtype=float)
    else:
        lower_array = np.asarray(lower, dtype=float)

    if upper is None:
        upper_array = np.full(n, np.inf, dtype=float)
    elif np.isscalar(upper):
        upper_array = np.full(n, float(upper), dtype=float)
    else:
        upper_array = np.asarray(upper, dtype=float)

    return _project_puf_weighted_sum_to_bounds(
        allocated,
        weights,
        target_total,
        lower_array,
        upper_array,
    )


def _allocate_puf_aggregate_agi(
    donor_agi: np.ndarray,
    weights: np.ndarray,
    recid: int,
    target_total: float,
) -> np.ndarray:
    donor_agi = np.asarray(donor_agi, dtype=float)
    weights = np.asarray(weights, dtype=float)
    dominance_cap = _PUF_AGGREGATE_MAX_AGI_DOMINANCE * abs(target_total) / weights
    n = len(donor_agi)

    if recid == 999996:
        lower = -dominance_cap
        upper = np.zeros(n, dtype=float)
    else:
        bucket_lower, bucket_upper = PUF_AGGREGATE_BUCKET_BOUNDS[recid]
        if np.isinf(bucket_upper):
            bucket_upper = _PUF_AGGREGATE_AGI_CAP_100M_PLUS
        lower = np.full(n, max(float(bucket_lower), 0.0), dtype=float)
        upper = np.minimum(np.full(n, float(bucket_upper), dtype=float), dominance_cap)

    return _allocate_puf_weighted_values(
        base_values=np.abs(donor_agi),
        weights=weights,
        target_total=target_total,
        lower=lower,
        upper=upper,
    )


def _sample_puf_aggregate_donors(
    donor_bucket: pd.DataFrame,
    donor_scores: pd.Series,
    target_mean_agi: float,
    n_records: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    scores = donor_scores.loc[donor_bucket.index].to_numpy(dtype=float)
    score_mass = np.clip(scores, 1e-6, None) ** _PUF_AGGREGATE_SELECTION_POWER
    donor_abs_agi = np.abs(donor_bucket["E00100"].to_numpy(dtype=float))
    target_abs_agi = max(abs(float(target_mean_agi)), 1.0)
    agi_distance = np.abs(np.log1p(donor_abs_agi) - np.log1p(target_abs_agi))
    probabilities = score_mass * np.sqrt(1.0 / (1.0 + agi_distance))
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        probabilities = np.ones(len(donor_bucket), dtype=float)
    probabilities = probabilities / probabilities.sum()

    selected_index = rng.choice(
        donor_bucket.index.to_numpy(),
        size=n_records,
        replace=len(donor_bucket) < n_records,
        p=probabilities,
    )
    return donor_bucket.loc[selected_index].reset_index(drop=True).copy()


def _disaggregate_puf_aggregate_bucket(
    recid: int,
    row: pd.Series,
    regular: pd.DataFrame,
    amount_columns: list[str],
    donor_scores: pd.Series,
    next_recid: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pop_weight = float(row["S006"]) / 100.0
    target_mean_agi = float(row["E00100"])
    target_total_agi = pop_weight * target_mean_agi
    donor_bucket = regular[_puf_aggregate_bucket_mask(regular, recid)].copy()
    if donor_bucket.empty:
        donor_bucket = regular.copy()

    total_weight = max(1, int(round(pop_weight)))
    n_records = min(_choose_puf_aggregate_synthetic_count(pop_weight), total_weight)
    synthetic_weights = _assign_puf_aggregate_weights(
        pop_weight, n_records, rng
    ).astype(float)
    selected = _sample_puf_aggregate_donors(
        donor_bucket=donor_bucket,
        donor_scores=donor_scores,
        target_mean_agi=target_mean_agi,
        n_records=n_records,
        rng=rng,
    )
    selected = selected.astype(
        {column: float for column in amount_columns if column in selected.columns}
    )

    synthetic = selected.copy()
    synthetic["RECID"] = np.arange(next_recid, next_recid + n_records, dtype=int)
    synthetic["S006"] = (synthetic_weights.astype(int) * 100).astype(int)

    for column in PUF_AGGREGATE_STRUCTURAL_COLUMNS:
        if column in synthetic.columns:
            synthetic[column] = selected[column].round().astype(int)
    if "MARS" in synthetic.columns and "XTOT" in synthetic.columns:
        joint_mask = synthetic["MARS"] == 2
        synthetic.loc[joint_mask, "XTOT"] = np.maximum(
            synthetic.loc[joint_mask, "XTOT"],
            2,
        )
        synthetic["XTOT"] = synthetic["XTOT"].clip(lower=0, upper=5).astype(int)

    synthetic["E00100"] = _allocate_puf_aggregate_agi(
        donor_agi=selected["E00100"].to_numpy(dtype=float),
        weights=synthetic_weights,
        recid=recid,
        target_total=target_total_agi,
    )
    for column in amount_columns:
        if column == "E00100":
            continue
        target_value = float(row.get(column, 0.0))
        if not np.isfinite(target_value):
            target_value = 0.0
        synthetic[column] = _allocate_puf_weighted_values(
            base_values=selected[column].to_numpy(dtype=float),
            weights=synthetic_weights,
            target_total=pop_weight * target_value,
        )
    return synthetic


def disaggregate_puf_aggregate_records(
    puf: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Replace IRS aggregate PUF rows with calibrated synthetic donor records."""
    if not {"RECID", "MARS", "S006", "E00100"}.issubset(puf.columns):
        return puf

    recid = pd.to_numeric(puf["RECID"], errors="coerce").astype("Int64")
    aggregate_mask = recid.isin(PUF_AGGREGATE_RECIDS) | pd.to_numeric(
        puf["MARS"], errors="coerce"
    ).eq(0)
    if not bool(aggregate_mask.any()):
        return puf

    regular = puf.loc[~aggregate_mask].copy()
    if regular.empty:
        return puf.loc[~pd.to_numeric(puf["MARS"], errors="coerce").eq(0)].copy()

    aggregate_rows = puf.loc[aggregate_mask].copy()
    amount_columns = _puf_aggregate_amount_columns(puf.columns)
    donor_scores = _puf_aggregate_eligibility_scores(regular)
    rng = np.random.default_rng(seed)
    next_recid = PUF_SYNTHETIC_RECID_START
    synthetic_rows: list[pd.DataFrame] = []

    for _, row in aggregate_rows.sort_values("RECID").iterrows():
        row_recid = int(row["RECID"])
        if row_recid not in PUF_AGGREGATE_BUCKET_BOUNDS:
            continue
        synthetic = _disaggregate_puf_aggregate_bucket(
            recid=row_recid,
            row=row,
            regular=regular,
            amount_columns=amount_columns,
            donor_scores=donor_scores,
            next_recid=next_recid,
            rng=rng,
        )
        next_recid += len(synthetic)
        synthetic_rows.append(synthetic[puf.columns])

    if not synthetic_rows:
        return regular.reset_index(drop=True)

    synthetic_frame = pd.concat(synthetic_rows, ignore_index=True)
    print(
        f"Disaggregated {int(aggregate_mask.sum())} aggregate PUF records into "
        f"{len(synthetic_frame)} synthetic records"
    )
    return pd.concat([regular, synthetic_frame], ignore_index=True)


def load_puf_raw(puf_path: Path, demographics_path: Path | None = None) -> pd.DataFrame:
    """Load raw PUF data from CSV."""
    print(f"Loading PUF from {puf_path}...")
    puf = pd.read_csv(puf_path)

    puf = disaggregate_puf_aggregate_records(puf)

    print(f"  Raw records: {len(puf):,}")

    # Load and merge demographics if available
    if demographics_path and demographics_path.exists():
        print(f"Loading demographics from {demographics_path}...")
        demo = pd.read_csv(demographics_path)

        # Demographics file has RECID to match
        if "RECID" in puf.columns and "RECID" in demo.columns:
            puf = puf.merge(demo, on="RECID", how="left", suffixes=("", "_demo"))
            print(f"  After demographics merge: {len(puf):,}")
            puf = _impute_missing_puf_demographics(puf)

    return puf


def _normalize_puf_uprating_mode(mode: str | None) -> str:
    resolved = (mode or PUF_UPRATING_MODE_INTERPOLATED).strip().lower()
    allowed = {
        PUF_UPRATING_MODE_INTERPOLATED,
        PUF_UPRATING_MODE_PE_SOI,
    }
    if resolved not in allowed:
        raise ValueError(
            f"puf uprating mode must be one of {sorted(allowed)}; got {mode!r}"
        )
    return resolved


def _resolve_pe_soi_path(
    *,
    policyengine_us_data_repo: str | Path | None = None,
    soi_path: str | Path | None = None,
) -> Path:
    if soi_path is not None:
        resolved = Path(soi_path)
    elif policyengine_us_data_repo is not None:
        resolved = (
            Path(policyengine_us_data_repo)
            / "policyengine_us_data"
            / "storage"
            / "soi.csv"
        )
    else:
        raise ValueError(
            "PE SOI uprating requires soi_path or policyengine_us_data_repo"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"Could not find PE SOI file at {resolved}")
    return resolved


def _resolve_pe_uprating_factors_path(
    *,
    policyengine_us_data_repo: str | Path | None = None,
) -> Path:
    if policyengine_us_data_repo is None:
        raise ValueError("PE forward uprating requires policyengine_us_data_repo")
    resolved = (
        Path(policyengine_us_data_repo)
        / "policyengine_us_data"
        / "storage"
        / "uprating_factors.csv"
    )
    if not resolved.exists():
        raise FileNotFoundError(f"Could not find PE uprating factors at {resolved}")
    return resolved


@cache
def _load_pe_soi_table(soi_path: str) -> pd.DataFrame:
    return pd.read_csv(soi_path)


@cache
def _load_pe_uprating_factors_table(uprating_factors_path: str) -> pd.DataFrame:
    return pd.read_csv(uprating_factors_path)


def _get_pe_soi_aggregate(
    soi_table: pd.DataFrame,
    variable: str,
    year: int,
    *,
    is_count: bool,
) -> float:
    lookup_variable = (
        "count" if variable == "adjusted_gross_income" and is_count else variable
    )
    rows = soi_table[
        (soi_table["Variable"] == lookup_variable)
        & (soi_table["Year"] == year)
        & (soi_table["Filing status"] == "All")
        & (soi_table["AGI lower bound"] == -np.inf)
        & (soi_table["AGI upper bound"] == np.inf)
        & (soi_table["Count"] == is_count)
        & (~soi_table["Taxable only"])
    ]
    if rows.empty:
        raise ValueError(
            f"Missing SOI aggregate for variable={lookup_variable!r}, year={year}, is_count={is_count}"
        )
    return float(rows.iloc[0]["Value"])


def _get_pe_soi_growth(
    soi_table: pd.DataFrame,
    variable: str,
    from_year: int,
    to_year: int,
) -> float:
    start_value = _get_pe_soi_aggregate(
        soi_table,
        variable,
        from_year,
        is_count=False,
    )
    end_value = _get_pe_soi_aggregate(
        soi_table,
        variable,
        to_year,
        is_count=False,
    )
    start_population = _get_pe_soi_aggregate(
        soi_table,
        "count",
        from_year,
        is_count=True,
    )
    end_population = _get_pe_soi_aggregate(
        soi_table,
        "count",
        to_year,
        is_count=True,
    )
    return (end_value / start_value) / (end_population / start_population)


def uprate_raw_puf_pe_style(
    puf: pd.DataFrame,
    *,
    from_year: int = 2015,
    to_year: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    soi_path: str | Path | None = None,
) -> pd.DataFrame:
    """Uprate raw PUF columns using the PE SOI growth contract."""
    if from_year == to_year:
        return puf.copy()
    resolved_soi_path = _resolve_pe_soi_path(
        policyengine_us_data_repo=policyengine_us_data_repo,
        soi_path=soi_path,
    )
    soi_table = _load_pe_soi_table(str(resolved_soi_path.resolve()))
    result = puf.copy()

    for variable, puf_column in PE_SOI_TO_PUF_STRAIGHT_RENAMES.items():
        if puf_column not in result.columns:
            continue
        growth = _get_pe_soi_growth(soi_table, variable, from_year, to_year)
        if variable in {
            "medical_expense_deductions_uncapped",
            "itemized_state_income_tax_deductions",
            "itemized_real_estate_tax_deductions",
            "interest_paid_deductions",
            "charitable_contributions_deductions",
        }:
            growth = (1.0 + PE_ITMDED_GROW_RATE) ** (to_year - from_year)
        values = pd.to_numeric(result[puf_column], errors="coerce").fillna(0.0)
        result[puf_column] = values * growth

    for variable, puf_column in PE_SOI_TO_PUF_POS_ONLY_RENAMES.items():
        if puf_column not in result.columns:
            continue
        growth = _get_pe_soi_growth(soi_table, variable, from_year, to_year)
        values = pd.to_numeric(result[puf_column], errors="coerce").fillna(0.0)
        result[puf_column] = values.where(values <= 0.0, values * growth)

    for variable, puf_column in PE_SOI_TO_PUF_NEG_ONLY_RENAMES.items():
        if puf_column not in result.columns:
            continue
        growth = _get_pe_soi_growth(soi_table, variable, from_year, to_year)
        values = pd.to_numeric(result[puf_column], errors="coerce").fillna(0.0)
        result[puf_column] = values.where(values >= 0.0, values * growth)

    agi_growth = _get_pe_soi_growth(
        soi_table,
        "adjusted_gross_income",
        from_year,
        to_year,
    )
    for puf_column in PE_PUF_REMAINING_RAW_COLUMNS:
        if puf_column not in result.columns:
            continue
        values = pd.to_numeric(result[puf_column], errors="coerce").fillna(0.0)
        result[puf_column] = values * agi_growth

    if "S006" in result.columns:
        returns_start = _get_pe_soi_aggregate(
            soi_table,
            "count",
            from_year,
            is_count=True,
        )
        returns_end = _get_pe_soi_aggregate(
            soi_table,
            "count",
            to_year,
            is_count=True,
        )
        weights = pd.to_numeric(result["S006"], errors="coerce").fillna(0.0)
        result["S006"] = weights * (returns_end / returns_start)

    return result


def uprate_mapped_puf_with_pe_factors(
    puf: pd.DataFrame,
    *,
    from_year: int = PE_PUF_SOI_END_YEAR,
    to_year: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
) -> pd.DataFrame:
    """Uprate mapped PUF variables using PE's forward factor table."""
    if to_year <= from_year:
        return puf.copy()
    uprating_factors_path = _resolve_pe_uprating_factors_path(
        policyengine_us_data_repo=policyengine_us_data_repo,
    )
    factors = _load_pe_uprating_factors_table(str(uprating_factors_path.resolve()))
    start_column = str(from_year)
    end_column = str(to_year)
    if start_column not in factors.columns or end_column not in factors.columns:
        raise ValueError(f"PE uprating factors do not cover {from_year} -> {to_year}")
    factor_lookup = factors.set_index("Variable")
    result = puf.copy()
    for column in result.columns:
        factor_variable = PE_UPRATING_FACTOR_ALIASES.get(column, column)
        if factor_variable not in factor_lookup.index:
            continue
        start_value = float(factor_lookup.at[factor_variable, start_column])
        end_value = float(factor_lookup.at[factor_variable, end_column])
        growth = end_value / start_value
        values = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        result[column] = values * growth
    if {
        "qualified_dividend_income",
        "non_qualified_dividend_income",
    }.issubset(result.columns):
        result["ordinary_dividend_income"] = result["qualified_dividend_income"].fillna(
            0.0
        ) + result["non_qualified_dividend_income"].fillna(0.0)
    if {
        "taxable_pension_income",
        "tax_exempt_pension_income",
    }.issubset(result.columns):
        result["total_pension_income"] = result["taxable_pension_income"].fillna(
            0.0
        ) + result["tax_exempt_pension_income"].fillna(0.0)
    return result


def _impute_missing_puf_demographics(puf: pd.DataFrame) -> pd.DataFrame:
    if not set(PUF_DEMOGRAPHIC_VARIABLES).issubset(puf.columns):
        return puf

    missing_mask = puf.loc[:, PUF_DEMOGRAPHIC_VARIABLES].isna().all(axis=1)
    if not bool(missing_mask.any()):
        return puf

    observed_mask = ~missing_mask
    if int(observed_mask.sum()) < 100:
        return puf

    try:
        from microimpute.models.qrf import QRF
    except ImportError:
        return puf

    train = (
        puf.loc[
            observed_mask, [*PUF_DEMOGRAPHIC_PREDICTORS, *PUF_DEMOGRAPHIC_VARIABLES]
        ]
        .copy()
        .fillna(0)
    )
    if len(train) > 10_000:
        train = train.sample(n=10_000, random_state=0)

    qrf = QRF(log_level="WARNING", memory_efficient=True)
    fitted_model = qrf.fit(
        X_train=train,
        predictors=list(PUF_DEMOGRAPHIC_PREDICTORS),
        imputed_variables=list(PUF_DEMOGRAPHIC_VARIABLES),
        n_jobs=1,
    )

    predicted = fitted_model.predict(
        X_test=puf.loc[missing_mask, list(PUF_DEMOGRAPHIC_PREDICTORS)].copy().fillna(0)
    )

    result = puf.copy()
    bounds = {
        "AGEDP1": (0, 7),
        "AGEDP2": (0, 7),
        "AGEDP3": (0, 7),
        "AGERANGE": (0, 7),
        "EARNSPLIT": (0, 4),
        "GENDER": (1, 2),
    }
    for column in PUF_DEMOGRAPHIC_VARIABLES:
        if column not in predicted.columns:
            continue
        values = pd.to_numeric(predicted[column], errors="coerce").fillna(0.0)
        lower, upper = bounds[column]
        values = values.round().clip(lower=lower, upper=upper)
        result.loc[missing_mask, column] = values.to_numpy()
    return result


def map_puf_variables(
    puf: pd.DataFrame,
    *,
    random_seed: int = 42,
    impute_pre_tax_contributions: bool = False,
    pre_tax_contribution_model: PEStyleQRFImputationModel | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    pre_tax_training_year: int = 2024,
    require_pre_tax_contribution_model: bool = False,
) -> pd.DataFrame:
    """Map PUF variable codes to common names."""
    result = pd.DataFrame(index=puf.index)
    manifest = load_us_source_manifest("puf")
    observation = manifest.observation_for(EntityType.TAX_UNIT)

    for column_spec in observation.columns:
        if column_spec.raw_column in puf.columns:
            result[column_spec.canonical_name] = puf[column_spec.raw_column].fillna(0)
        else:
            result[column_spec.canonical_name] = 0

    # Fix weight (PUF stores in hundredths)
    if "weight" in result.columns:
        result["weight"] = result["weight"] / 100

    # Preserve rental losses as negative values so downstream PE targets can
    # recover rent-and-royalty loss cells.
    result["rental_income"] = result.get("rental_income_positive", 0).fillna(
        0
    ) + -result.get("rental_income_negative", 0).fillna(0)
    if {"E00600", "E00650"}.issubset(set(puf.columns)):
        result["non_qualified_dividend_income"] = puf["E00600"].fillna(0) - puf[
            "E00650"
        ].fillna(0)
    if {"E26190", "E26180", "E25980", "E25960"}.issubset(set(puf.columns)):
        s_corp_income = puf["E26190"].fillna(0) - puf["E26180"].fillna(0)
        partnership_income = puf["E25980"].fillna(0) - puf["E25960"].fillna(0)
        result["partnership_s_corp_income"] = s_corp_income + partnership_income
    if {
        "E30400",
        "E30500",
        "E00900",
        "E02100",
        "E25940",
        "E25980",
        "E25920",
        "E25960",
    }.issubset(set(puf.columns)):
        se_deduction_factor = 0.9235
        taxable_se = puf["E30400"].fillna(0) + puf["E30500"].fillna(0)
        gross_se = taxable_se / se_deduction_factor
        schedule_c_f_income = puf["E00900"].fillna(0) + puf["E02100"].fillna(0)
        has_partnership = (
            puf["E25940"].fillna(0)
            + puf["E25980"].fillna(0)
            - puf["E25920"].fillna(0)
            - puf["E25960"].fillna(0)
        ) != 0
        result["partnership_se_income"] = np.where(
            has_partnership,
            gross_se - schedule_c_f_income,
            0.0,
        )
    if "T27800" in puf.columns:
        result["farm_income"] = puf["T27800"].fillna(0)
    if {"E26390", "E26400"}.issubset(set(puf.columns)):
        result["estate_income"] = puf["E26390"].fillna(0) - puf["E26400"].fillna(0)
    if {"E01500", "E01700"}.issubset(set(puf.columns)):
        result["tax_exempt_pension_income"] = puf["E01500"].fillna(0) - puf[
            "E01700"
        ].fillna(0)
    medical_expense_floor = result.get("medical_expense_agi_floor")
    if medical_expense_floor is not None:
        for variable, fraction in MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS.items():
            result[variable] = medical_expense_floor.fillna(0) * fraction

    # Map filing status code to string
    filing_status_map = {
        1: "SINGLE",
        2: "JOINT",
        3: "SEPARATE",
        4: "HEAD_OF_HOUSEHOLD",
        5: "SURVIVING_SPOUSE",
    }
    result["filing_status"] = (
        result["filing_status_code"].map(filing_status_map).fillna("UNKNOWN")
    )
    filing_status_code = (
        pd.to_numeric(result["filing_status_code"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    result["is_surviving_spouse"] = filing_status_code.eq(5)

    # Add age from demographics if available
    if "age" in puf.columns:
        result["age"] = puf["age"]
    elif "AGE_HEAD" in puf.columns:
        result["age"] = puf["AGE_HEAD"]
    else:
        # Impute age based on income patterns
        result["age"] = _impute_age(result, random_seed=random_seed)

    # Add sex from demographics if available
    if "is_male" in puf.columns:
        result["is_male"] = puf["is_male"]
    elif "GENDER" in puf.columns:
        result["is_male"] = (puf["GENDER"] == 1).astype(float)
    else:
        # Unknown - will be learned from CPS
        result["is_male"] = np.nan

    if "RECID" in puf.columns:
        result["_puf_recid"] = pd.to_numeric(puf["RECID"], errors="coerce")
    if "AGERANGE" in puf.columns:
        result["_puf_agerange"] = pd.to_numeric(puf["AGERANGE"], errors="coerce")
    if "EARNSPLIT" in puf.columns:
        result["_puf_earnsplit"] = pd.to_numeric(puf["EARNSPLIT"], errors="coerce")
    if "GENDER" in puf.columns:
        result["_puf_gender"] = pd.to_numeric(puf["GENDER"], errors="coerce")
    for dependent_idx in range(1, 4):
        raw_column = f"AGEDP{dependent_idx}"
        if raw_column in puf.columns:
            result[f"_puf_agedp{dependent_idx}"] = pd.to_numeric(
                puf[raw_column],
                errors="coerce",
            )

    if impute_pre_tax_contributions:
        model = pre_tax_contribution_model
        if model is None:
            try:
                model = _default_pe_style_puf_pre_tax_contribution_model(
                    policyengine_us_data_repo=policyengine_us_data_repo,
                    policyengine_us_data_python=policyengine_us_data_python,
                    pre_tax_training_year=pre_tax_training_year,
                )
            except (FileNotFoundError, ImportError, ValueError):
                if require_pre_tax_contribution_model:
                    raise
                model = None
        if model is not None:
            predictor_frame = result.loc[:, model.predictors].copy()
            predictor_frame = predictor_frame.apply(
                lambda column: pd.to_numeric(column, errors="coerce").fillna(0.0)
            )
            try:
                predictions = model.fitted_model.predict(X_test=predictor_frame)
            except (
                FileNotFoundError,
                ImportError,
                ValueError,
            ):
                if require_pre_tax_contribution_model:
                    raise
                model = None
            else:
                result[model.imputed_variable] = pd.to_numeric(
                    predictions[model.imputed_variable],
                    errors="coerce",
                ).fillna(0.0)
        elif require_pre_tax_contribution_model:
            raise RuntimeError(
                "pre_tax_contributions imputation was requested but no PE-style "
                "pre-tax contribution model was available"
            )

    # Mark survey source
    result["_survey"] = "puf"

    return result


def _impute_age(
    df: pd.DataFrame,
    *,
    random_seed: int = 42,
) -> pd.Series:
    """Simple age imputation based on income patterns.

    This is a rough heuristic. The masked MAF will learn
    better age distributions from CPS.
    """
    # Base age on Social Security receipt and pension income
    age = pd.Series(40, index=df.index)  # Default

    # Social Security recipients tend to be older
    has_ss = df.get("gross_social_security", 0) > 0
    age = age.where(~has_ss, 68)

    # Pension recipients also older
    has_pension = df.get("taxable_pension_income", 0) > 0
    age = age.where(~has_pension | has_ss, 62)

    # IRA distributions suggest retirement age
    has_ira = df.get("ira_distributions", 0) > 0
    age = age.where(~has_ira | has_ss | has_pension, 60)

    # High earners tend to be prime working age
    high_wage = df.get("employment_income", 0) > 200_000
    age = age.where(~high_wage, 45)

    # Add some noise
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(0, 5, len(age))
    age = (age + noise).clip(18, 95).astype(int)

    return age


def _decode_puf_filer_age(
    age_range: int | float | None,
    *,
    fallback: float = 40.0,
    rng: np.random.Generator | None = None,
) -> int:
    resolved_fallback = 40.0 if fallback is None or pd.isna(fallback) else fallback
    if age_range is None or pd.isna(age_range):
        return int(resolved_fallback)
    age_code = int(age_range)
    if age_code == 0:
        return int(resolved_fallback)
    age_decode = {
        1: 18,
        2: 26,
        3: 35,
        4: 45,
        5: 55,
        6: 65,
        7: 80,
    }
    lower = age_decode.get(age_code)
    upper = age_decode.get(age_code + 1)
    if lower is None:
        return int(resolved_fallback)
    if upper is None or upper <= lower:
        if age_code == 7:
            if rng is not None:
                return int(rng.integers(low=lower, high=90, endpoint=False))
            return int(lower + (90 - lower) / 2)
        return int(lower)
    return int(lower + (upper - lower) / 2)


def _decode_puf_dependent_age(
    age_range: int | float | None,
    *,
    rng: np.random.Generator | None = None,
) -> int:
    if age_range is None or pd.isna(age_range):
        return 0
    age_code = int(age_range)
    if age_code == 0:
        return 0
    age_decode = {
        0: 0,
        1: 0,
        2: 5,
        3: 13,
        4: 17,
        5: 19,
        6: 25,
        7: 30,
    }
    lower = age_decode.get(age_code, 0)
    upper = age_decode.get(age_code + 1, lower)
    if upper <= lower:
        return int(lower)
    if rng is not None:
        return int(rng.integers(low=lower, high=upper, endpoint=False))
    return int(lower + (upper - lower) / 2)


def _puf_joint_head_share(
    row: pd.Series,
    *,
    default: float = 0.6,
    rng: np.random.Generator | None = None,
) -> float:
    earnsplit = row.get("_puf_earnsplit")
    if earnsplit is None or pd.isna(earnsplit):
        return default
    split_code = int(earnsplit)
    if split_code <= 0:
        return 1.0
    split_decodes = {
        1: 0.0,
        2: 0.25,
        3: 0.75,
        4: 1.0,
        5: 1.0,
    }
    lower = split_decodes.get(split_code)
    upper = split_decodes.get(split_code + 1)
    if lower is None or upper is None:
        return default
    if rng is not None:
        frac = (upper - lower) * rng.random() + lower
        return float(1.0 - frac)
    return float(1.0 - ((lower + upper) / 2.0))


def _puf_spouse_is_male(
    gender_code: int | float | None,
    *,
    rng: np.random.Generator | None = None,
) -> float:
    if gender_code is None or pd.isna(gender_code):
        return 0.0
    resolved_gender = int(gender_code)
    if rng is None:
        return float(resolved_gender != 1)
    is_opposite_gender = bool(rng.random() < 0.96)
    opposite_gender_code = 0.0 if resolved_gender == 1 else 1.0
    same_gender_code = 1.0 - opposite_gender_code
    return opposite_gender_code if is_opposite_gender else same_gender_code


def _puf_dependent_is_male(*, rng: np.random.Generator | None = None) -> float:
    if rng is None:
        return 0.0
    return float(rng.choice([0, 1]))


def _is_puf_numeric_split_column(df: pd.DataFrame, column: str) -> bool:
    if column in PUF_PERSON_EXPANSION_PRESERVE_COLUMNS:
        return False
    if column.startswith("_"):
        return False
    return pd.api.types.is_numeric_dtype(df[column])


def uprate_puf(
    df: pd.DataFrame, from_year: int = 2015, to_year: int = 2024
) -> pd.DataFrame:
    """Uprate PUF income variables from one year to another.

    Uses SOI-based growth factors.
    """
    if from_year == to_year:
        return df

    # Simple scaling - in production, use year-specific factors
    year_factor = (to_year - from_year) / (2024 - 2015)

    result = df.copy()

    for var, factor in UPRATING_FACTORS.items():
        if var in result.columns:
            # Interpolate factor based on years
            scaled_factor = 1 + (factor - 1) * year_factor
            result[var] = result[var] * scaled_factor

    print(f"Uprated PUF from {from_year} to {to_year}")

    return result


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return df[column].fillna(0).astype(float)


def _default_cps_reference_year(target_year: int) -> int:
    return min(max(target_year - 1, 2021), 2023)


def _social_security_age_bucket(ages: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(ages, errors="coerce"),
        bins=SOCIAL_SECURITY_SHARE_AGE_BINS,
        labels=SOCIAL_SECURITY_SHARE_AGE_LABELS,
        right=False,
        include_lowest=True,
    )


def _normalize_social_security_split_strategy(strategy: str | None) -> str:
    resolved = (
        (strategy or SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE).strip().lower()
    )
    allowed = {
        SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE,
        SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF,
        SOCIAL_SECURITY_SPLIT_STRATEGY_AGE_HEURISTIC,
    }
    if resolved not in allowed:
        raise ValueError(
            "social_security_split_strategy must be one of "
            f"{sorted(allowed)}; got {strategy!r}"
        )
    return resolved


def _build_pe_style_social_security_predictor_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    if "age" in frame.columns:
        result["age"] = pd.to_numeric(frame["age"], errors="coerce").astype(float)
    if "is_male" in frame.columns:
        result["is_male"] = pd.to_numeric(frame["is_male"], errors="coerce").astype(
            float
        )
    elif "sex" in frame.columns:
        sex = pd.to_numeric(frame["sex"], errors="coerce")
        result["is_male"] = pd.Series(
            np.where(sex == 1, 1.0, np.where(sex == 2, 0.0, np.nan)),
            index=frame.index,
            dtype=float,
        )
    if "tax_unit_is_joint" in frame.columns:
        result["tax_unit_is_joint"] = pd.to_numeric(
            frame["tax_unit_is_joint"], errors="coerce"
        ).astype(float)
    elif "filing_status" in frame.columns:
        filing_status = frame["filing_status"].astype(str)
        result["tax_unit_is_joint"] = (filing_status == "JOINT").astype(float)
    if "is_tax_unit_head" in frame.columns:
        result["is_tax_unit_head"] = pd.to_numeric(
            frame["is_tax_unit_head"], errors="coerce"
        ).astype(float)
    elif "is_head" in frame.columns:
        result["is_tax_unit_head"] = pd.to_numeric(
            frame["is_head"], errors="coerce"
        ).astype(float)
    if "is_tax_unit_dependent" in frame.columns:
        result["is_tax_unit_dependent"] = pd.to_numeric(
            frame["is_tax_unit_dependent"], errors="coerce"
        ).astype(float)
    elif "is_dependent" in frame.columns:
        result["is_tax_unit_dependent"] = pd.to_numeric(
            frame["is_dependent"], errors="coerce"
        ).astype(float)
    return result


def _fit_puf_social_security_share_model_from_reference(
    reference_persons: pd.DataFrame,
) -> GroupedShareModel:
    work = reference_persons.copy()
    if "weight" not in work.columns:
        work["weight"] = 1.0
    work["age_bucket"] = _social_security_age_bucket(
        work.get("age", pd.Series(np.nan, index=work.index))
    )
    return fit_grouped_share_model(
        work,
        explicit_component_columns=SOCIAL_SECURITY_SHARE_EXPLICIT_COMPONENTS,
        implicit_component_column=SOCIAL_SECURITY_SHARE_IMPLICIT_COMPONENT,
        feature_sets=(("age_bucket",),),
        weight_column="weight",
    )


def _default_puf_social_security_share_model(
    *,
    cps_reference_year: int,
    cache_dir: Path | None,
) -> GroupedShareModel:
    cps_dataset = load_cps_asec(
        year=cps_reference_year,
        cache_dir=cache_dir,
        download=True,
    )
    return _fit_puf_social_security_share_model_from_reference(
        cps_dataset.persons.to_pandas()
    )


def _fit_pe_style_puf_social_security_qrf_model_from_reference(
    reference_persons: pd.DataFrame,
    *,
    min_training_records: int = MIN_PE_STYLE_SOCIAL_SECURITY_QRF_TRAINING_RECORDS,
) -> PEStyleQRFShareModel:
    from microimpute.models.qrf import QRF

    total_social_security = _numeric_series(reference_persons, "social_security")
    has_social_security = total_social_security > 0.0
    if int(has_social_security.sum()) < min_training_records:
        raise ValueError(
            "PE-style QRF Social Security split requires at least "
            f"{min_training_records} positive training rows"
        )

    predictor_frame = _build_pe_style_social_security_predictor_frame(
        reference_persons.loc[has_social_security]
    )
    available_predictors = tuple(
        predictor
        for predictor in PE_STYLE_SOCIAL_SECURITY_QRF_PREDICTORS
        if predictor in predictor_frame.columns
        and predictor_frame[predictor].notna().any()
    )
    if not available_predictors:
        raise ValueError(
            "PE-style QRF Social Security split requires at least one predictor"
        )

    train = predictor_frame.loc[:, available_predictors].copy()
    total = total_social_security.loc[has_social_security].to_numpy(dtype=float)
    share_prediction_columns: list[str] = []
    for component in SOCIAL_SECURITY_SHARE_COMPONENTS:
        share_column = f"{component}_share"
        share_prediction_columns.append(share_column)
        component_values = _numeric_series(
            reference_persons.loc[has_social_security],
            component,
        ).to_numpy(dtype=float)
        train[share_column] = np.where(total > 0.0, component_values / total, 0.0)

    qrf = QRF(log_level="WARNING", memory_efficient=True)
    fitted_model = qrf.fit(
        X_train=train.loc[:, [*available_predictors, *share_prediction_columns]],
        predictors=list(available_predictors),
        imputed_variables=share_prediction_columns,
        n_jobs=1,
    )
    return PEStyleQRFShareModel(
        predictors=available_predictors,
        component_columns=SOCIAL_SECURITY_SHARE_COMPONENTS,
        share_prediction_columns=tuple(share_prediction_columns),
        fitted_model=fitted_model,
    )


def _default_pe_style_puf_social_security_share_model(
    *,
    cps_reference_year: int,
    cache_dir: Path | None,
) -> PEStyleQRFShareModel:
    cps_dataset = load_cps_asec(
        year=cps_reference_year,
        cache_dir=cache_dir,
        download=True,
    )
    return _fit_pe_style_puf_social_security_qrf_model_from_reference(
        cps_dataset.persons.to_pandas()
    )


def _load_pe_extended_cps_pre_tax_training_frame(
    *,
    policyengine_us_data_repo: str | Path,
    training_year: int,
) -> pd.DataFrame:
    import h5py

    repo_root = Path(policyengine_us_data_repo).expanduser().resolve()
    storage_dir = repo_root / "policyengine_us_data" / "storage"
    candidate_paths = (
        storage_dir / f"extended_cps_{int(training_year)}.h5",
        storage_dir / "extended_cps_2024.h5",
    )
    dataset_path = next((path for path in candidate_paths if path.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError(
            "Could not locate an extended CPS training artifact for PE-style "
            f"pre-tax contributions under {storage_dir}"
        )

    with h5py.File(dataset_path, "r") as h5:
        train = pd.DataFrame(
            {
                "employment_income": np.asarray(
                    h5["employment_income"][str(int(training_year))], dtype=float
                )
                if str(int(training_year)) in h5["employment_income"]
                else np.asarray(
                    h5["employment_income"][sorted(h5["employment_income"].keys())[-1]],
                    dtype=float,
                ),
                "age": np.asarray(h5["age"][str(int(training_year))], dtype=float)
                if str(int(training_year)) in h5["age"]
                else np.asarray(
                    h5["age"][sorted(h5["age"].keys())[-1]],
                    dtype=float,
                ),
                "is_male": 1.0
                - np.asarray(h5["is_female"][str(int(training_year))], dtype=float)
                if str(int(training_year)) in h5["is_female"]
                else 1.0
                - np.asarray(
                    h5["is_female"][sorted(h5["is_female"].keys())[-1]],
                    dtype=float,
                ),
                "pre_tax_contributions": np.asarray(
                    h5["pre_tax_contributions"][str(int(training_year))], dtype=float
                )
                if str(int(training_year)) in h5["pre_tax_contributions"]
                else np.asarray(
                    h5["pre_tax_contributions"][
                        sorted(h5["pre_tax_contributions"].keys())[-1]
                    ],
                    dtype=float,
                ),
            }
        )
    if len(train) > 10_000:
        train = train.sample(n=10_000, random_state=0)
    return train


def _load_microplex_cps_pre_tax_training_frame(
    *,
    training_year: int,
) -> pd.DataFrame:
    cps = load_cps_asec(year=int(training_year))
    persons = cps.persons.to_pandas()
    index = persons.index
    employment_income = pd.to_numeric(
        persons.get("employment_income", persons.get("wage_income", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    age = pd.to_numeric(persons.get("age", 0.0), errors="coerce").fillna(0.0)
    if "is_male" in persons.columns:
        is_male = pd.to_numeric(persons["is_male"], errors="coerce").fillna(0.0)
    elif "sex" in persons.columns:
        is_male = (
            pd.to_numeric(persons["sex"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(1)
            .astype(float)
        )
    else:
        is_male = pd.Series(0.0, index=index)

    if "pre_tax_contributions" in persons.columns:
        pre_tax_contributions = pd.to_numeric(
            persons["pre_tax_contributions"],
            errors="coerce",
        ).fillna(0.0)
    else:
        pre_tax_contributions = pd.Series(0.0, index=index)
        for column in (
            "traditional_401k_contributions",
            "traditional_401k_contributions_desired",
            "traditional_403b_contributions",
            "traditional_403b_contributions_desired",
            "pre_tax_health_insurance_premiums",
            "health_savings_account_payroll_contributions",
        ):
            if column in persons.columns:
                pre_tax_contributions = pre_tax_contributions.add(
                    pd.to_numeric(persons[column], errors="coerce").fillna(0.0),
                    fill_value=0.0,
                )

    train = pd.DataFrame(
        {
            "employment_income": employment_income,
            "age": age,
            "is_male": is_male,
            "pre_tax_contributions": pre_tax_contributions,
        },
        index=index,
    )
    train = train.apply(
        lambda column: pd.to_numeric(column, errors="coerce").fillna(0.0)
    )
    if len(train) > 10_000:
        train = train.sample(n=10_000, random_state=0)
    return train


@lru_cache(maxsize=4)
def _default_pe_style_puf_pre_tax_contribution_model(
    *,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    pre_tax_training_year: int = 2024,
) -> PEStyleQRFImputationModel:
    predictors = ("employment_income", "age", "is_male")
    if policyengine_us_data_repo is not None:
        try:
            train = _load_pe_extended_cps_pre_tax_training_frame(
                policyengine_us_data_repo=policyengine_us_data_repo,
                training_year=pre_tax_training_year,
            )
        except (FileNotFoundError, KeyError, OSError, ValueError):
            train = _load_microplex_cps_pre_tax_training_frame(
                training_year=pre_tax_training_year,
            )

        from microimpute.models.qrf import QRF

        train = train.apply(
            lambda column: pd.to_numeric(column, errors="coerce").fillna(0.0)
        )
        qrf = QRF(log_level="WARNING", memory_efficient=True)
        fitted_model = qrf.fit(
            X_train=train,
            predictors=list(predictors),
            imputed_variables=["pre_tax_contributions"],
            n_jobs=1,
        )
        return PEStyleQRFImputationModel(
            predictors=predictors,
            imputed_variable="pre_tax_contributions",
            fitted_model=fitted_model,
        )

    from microimpute.models.qrf import QRF

    train = _load_microplex_cps_pre_tax_training_frame(
        training_year=pre_tax_training_year
    )

    qrf = QRF(log_level="WARNING", memory_efficient=True)
    fitted_model = qrf.fit(
        X_train=train,
        predictors=list(predictors),
        imputed_variables=["pre_tax_contributions"],
        n_jobs=1,
    )
    return PEStyleQRFImputationModel(
        predictors=predictors,
        imputed_variable="pre_tax_contributions",
        fitted_model=fitted_model,
    )


def _strategy_social_security_share_model_loader(
    strategy: str,
) -> Callable[[int, Path | None], SocialSecurityShareModel]:
    if strategy == SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF:
        return lambda year, cache_dir: (
            _default_pe_style_puf_social_security_share_model(
                cps_reference_year=year,
                cache_dir=cache_dir,
            )
        )
    if strategy == SOCIAL_SECURITY_SPLIT_STRATEGY_AGE_HEURISTIC:
        return lambda year, cache_dir: _age_heuristic_puf_social_security_share_model()
    return lambda year, cache_dir: _default_puf_social_security_share_model(
        cps_reference_year=year,
        cache_dir=cache_dir,
    )


def _age_heuristic_puf_social_security_share_model() -> GroupedShareModel:
    reference = pd.DataFrame(
        {
            "age_bucket": list(SOCIAL_SECURITY_SHARE_AGE_LABELS),
            "weight": [1.0] * len(SOCIAL_SECURITY_SHARE_AGE_LABELS),
            "social_security_retirement": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "social_security_disability": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "social_security_survivors": [0.0] * len(SOCIAL_SECURITY_SHARE_AGE_LABELS),
            "social_security_dependents": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    return fit_grouped_share_model(
        reference,
        explicit_component_columns=SOCIAL_SECURITY_SHARE_EXPLICIT_COMPONENTS,
        implicit_component_column=SOCIAL_SECURITY_SHARE_IMPLICIT_COMPONENT,
        feature_sets=(("age_bucket",),),
        weight_column="weight",
    )


def _predict_puf_social_security_component_shares(
    persons: pd.DataFrame,
    *,
    share_model: SocialSecurityShareModel,
) -> pd.DataFrame:
    if isinstance(share_model, GroupedShareModel):
        features = persons.loc[:, []].copy()
        features["age_bucket"] = _social_security_age_bucket(
            persons.get("age", pd.Series(np.nan, index=persons.index))
        )
        return predict_grouped_component_shares(features, share_model)

    predictors = _build_pe_style_social_security_predictor_frame(persons)
    X_test = pd.DataFrame(index=persons.index)
    for predictor in share_model.predictors:
        if predictor in predictors.columns:
            X_test[predictor] = (
                pd.to_numeric(predictors[predictor], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
        else:
            X_test[predictor] = 0.0
    predictions = share_model.fitted_model.predict(X_test=X_test)

    shares = pd.DataFrame(index=persons.index)
    total = np.zeros(len(persons), dtype=float)
    for component, share_column in zip(
        share_model.component_columns,
        share_model.share_prediction_columns,
        strict=True,
    ):
        source_column = (
            share_column if share_column in predictions.columns else component
        )
        if source_column in predictions.columns:
            values = np.clip(
                pd.to_numeric(predictions[source_column], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float),
                0.0,
                1.0,
            )
        else:
            values = np.zeros(len(persons), dtype=float)
        shares[component] = values
        total += values

    positive_total = total > 0.0
    for component in share_model.component_columns:
        shares[component] = np.where(
            positive_total,
            shares[component].to_numpy(dtype=float) / total,
            0.0,
        )
    return shares


def _impute_puf_social_security_components(
    persons: pd.DataFrame,
    *,
    share_model: SocialSecurityShareModel,
) -> pd.DataFrame:
    result = persons.copy()
    total_social_security = _numeric_series(result, "social_security")
    if float(total_social_security.sum()) <= 0.0:
        for component in SOCIAL_SECURITY_SHARE_COMPONENTS:
            result[component] = 0.0
        return result

    shares = _predict_puf_social_security_component_shares(
        result,
        share_model=share_model,
    )
    for component in SOCIAL_SECURITY_SHARE_COMPONENTS:
        result[component] = total_social_security * shares[component]
    return result


def _add_derived_income_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = normalize_dividend_columns(result)
    employment_income = _numeric_series(result, "employment_income")
    self_employment_income = _numeric_series(result, "self_employment_income")
    taxable_interest_income = _numeric_series(result, "taxable_interest_income")
    ordinary_dividend_income = _numeric_series(result, "ordinary_dividend_income")
    short_term_capital_gains = _numeric_series(result, "short_term_capital_gains")
    long_term_capital_gains = _numeric_series(result, "long_term_capital_gains")
    taxable_pension_income = _numeric_series(result, "taxable_pension_income")
    gross_social_security = _numeric_series(result, "gross_social_security")
    if "age" in result.columns:
        ages = pd.to_numeric(result["age"], errors="coerce").fillna(0.0).astype(float)
    else:
        ages = pd.Series(0.0, index=result.index, dtype=float)
    rental_income = _numeric_series(result, "rental_income")
    unemployment_compensation = _numeric_series(
        result,
        "unemployment_compensation",
    )
    alimony_income = _numeric_series(result, "alimony_income")

    result["interest_income"] = taxable_interest_income
    result["dividend_income"] = ordinary_dividend_income
    result["capital_gains"] = short_term_capital_gains + long_term_capital_gains
    result["pension_income"] = taxable_pension_income
    result["social_security"] = gross_social_security
    result["social_security_retirement"] = gross_social_security.where(
        ages >= MINIMUM_SOCIAL_SECURITY_RETIREMENT_AGE, 0.0
    ).astype(float)
    result["income"] = (
        employment_income
        + self_employment_income
        + result["interest_income"]
        + result["dividend_income"]
        + rental_income
        + result["social_security"]
        + result["pension_income"]
        + unemployment_compensation
        + alimony_income
    )
    result["employment_status"] = (
        (employment_income + self_employment_income) > 0
    ).astype(int)
    return result


def _allocate_joint_tax_unit_amounts(
    row: pd.Series,
    head: pd.Series,
    spouse: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    for variable, head_share in JOINT_HEAD_SHARE_ALLOCATION.items():
        if variable not in row.index:
            continue
        amount = float(row[variable])
        head[variable] = amount * head_share
        spouse[variable] = amount * (1.0 - head_share)

    for variable in JOINT_EQUAL_SHARE_ALLOCATION:
        if variable not in row.index:
            continue
        amount = float(row[variable])
        head[variable] = amount * 0.5
        spouse[variable] = amount * 0.5

    return head, spouse


def expand_to_persons(df: pd.DataFrame) -> pd.DataFrame:
    """Expand tax unit records to person-level records.

    Each tax unit becomes 1-2 persons (filer + spouse if joint).
    This enables stacking with CPS person-level data.
    """
    records = []
    split_columns = [
        column for column in df.columns if _is_puf_numeric_split_column(df, column)
    ]
    pe_rng = np.random.default_rng(PE_PUF_PERSON_EXPANSION_RANDOM_SEED)
    pe_age_rng = np.random.default_rng(PE_PUF_PERSON_EXPANSION_RANDOM_SEED + 1)

    for idx, row in df.iterrows():
        filing_status = row.get("filing_status", "SINGLE")
        exemptions = int(
            pd.to_numeric(row.get("exemptions_count", 1), errors="coerce") or 1
        )
        has_pe_demographics = "_puf_agerange" in row.index and not pd.isna(
            row.get("_puf_agerange")
        )
        tax_unit_id = row.get("_puf_recid")
        if tax_unit_id is None or pd.isna(tax_unit_id):
            tax_unit_id = idx
        pe_tax_unit_id = str(int(tax_unit_id)) if pd.notna(tax_unit_id) else str(idx)

        # Create head record
        head = row.copy()
        head["is_head"] = 1
        head["is_spouse"] = 0
        head["is_dependent"] = 0
        head["person_id"] = (
            f"{pe_tax_unit_id}:1" if has_pe_demographics else f"{idx}_head"
        )
        head["tax_unit_id"] = pe_tax_unit_id if has_pe_demographics else str(idx)
        if has_pe_demographics:
            head["age"] = _decode_puf_filer_age(
                row.get("_puf_agerange"),
                fallback=row.get("age", 40.0),
                rng=pe_age_rng,
            )
            if pd.notna(row.get("_puf_gender")):
                head["is_male"] = float(int(row.get("_puf_gender")) == 1)
        records.append(head)

        # Create spouse record if joint filing
        if filing_status == "JOINT":
            spouse = row.copy()
            spouse["is_head"] = 0
            spouse["is_spouse"] = 1
            spouse["is_dependent"] = 0
            spouse["person_id"] = (
                f"{pe_tax_unit_id}:2" if has_pe_demographics else f"{idx}_spouse"
            )
            spouse["tax_unit_id"] = pe_tax_unit_id if has_pe_demographics else str(idx)
            spouse["is_surviving_spouse"] = False

            if has_pe_demographics:
                spouse["age"] = _decode_puf_filer_age(
                    row.get("_puf_agerange"),
                    fallback=row.get("age", 40.0),
                    rng=pe_age_rng,
                )
                if pd.notna(row.get("_puf_gender")):
                    spouse["is_male"] = _puf_spouse_is_male(
                        row.get("_puf_gender"),
                    )
                head_share = _puf_joint_head_share(row, rng=pe_rng)
                for column in split_columns:
                    amount = float(
                        pd.to_numeric(row.get(column), errors="coerce") or 0.0
                    )
                    head[column] = amount * head_share
                    spouse[column] = amount * (1.0 - head_share)
            else:
                head, spouse = _allocate_joint_tax_unit_amounts(row, head, spouse)
            # Spouse weight is same as head (we'll deduplicate in calibration)
            records.append(spouse)
            exemptions -= 1

        exemptions -= 1
        if has_pe_demographics:
            for dependent_idx in range(min(3, max(exemptions, 0))):
                dependent = row.copy()
                dependent["is_head"] = 0
                dependent["is_spouse"] = 0
                dependent["is_dependent"] = 1
                dependent["person_id"] = f"{pe_tax_unit_id}:{dependent_idx + 3}"
                dependent["tax_unit_id"] = pe_tax_unit_id
                dependent["age"] = _decode_puf_dependent_age(
                    row.get(f"_puf_agedp{dependent_idx + 1}"),
                )
                dependent["is_male"] = _puf_dependent_is_male()
                dependent["is_surviving_spouse"] = False
                for column in split_columns:
                    dependent[column] = 0.0
                records.append(dependent)

    result = pd.DataFrame(records).reset_index(drop=True)
    helper_columns = [
        column for column in result.columns if column in PUF_DEMOGRAPHIC_HELPER_COLUMNS
    ]
    if helper_columns:
        result = result.drop(columns=helper_columns)
    result = _add_derived_income_columns(result)
    print(f"Expanded {len(df):,} tax units to {len(result):,} persons")

    return result


def load_puf(
    target_year: int = 2024,
    expand_persons: bool = True,
    cache_dir: Path | None = None,
    social_security_split_strategy: str = SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE,
    uprating_mode: str = PUF_UPRATING_MODE_INTERPOLATED,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    impute_pre_tax_contributions: bool = False,
    pre_tax_training_year: int = 2024,
    soi_path: str | Path | None = None,
    require_pre_tax_contribution_model: bool = False,
) -> pd.DataFrame:
    """Load and process PUF for multi-survey fusion.

    Args:
        target_year: Year to uprate to
        expand_persons: If True, expand tax units to person records
        cache_dir: Directory to cache downloaded files

    Returns:
        DataFrame with common variable names, ready for stacking with CPS
    """
    # Prefer a repo-local raw PUF copy when available to avoid remote auth/cache
    # requirements during rebuild runs.
    local_repo_paths = _resolve_policyengine_repo_local_puf_paths(
        policyengine_us_data_repo
    )
    if local_repo_paths is not None:
        puf_path, demo_path = local_repo_paths
        print(f"Using repo-local PUF from {puf_path}...")
    else:
        puf_path, demo_path = download_puf(cache_dir)

    # Load raw data
    raw = load_puf_raw(puf_path, demo_path)
    resolved_uprating_mode = _normalize_puf_uprating_mode(uprating_mode)
    if resolved_uprating_mode == PUF_UPRATING_MODE_PE_SOI:
        raw_uprating_year = min(int(target_year), PE_PUF_SOI_END_YEAR)
        raw = uprate_raw_puf_pe_style(
            raw,
            from_year=2015,
            to_year=raw_uprating_year,
            policyengine_us_data_repo=policyengine_us_data_repo,
            soi_path=soi_path,
        )

    # Map to common variables
    df = map_puf_variables(
        raw,
        impute_pre_tax_contributions=impute_pre_tax_contributions,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        pre_tax_training_year=pre_tax_training_year,
        require_pre_tax_contribution_model=require_pre_tax_contribution_model,
    )

    # Uprate to target year
    if resolved_uprating_mode == PUF_UPRATING_MODE_PE_SOI:
        if target_year > PE_PUF_SOI_END_YEAR:
            df = uprate_mapped_puf_with_pe_factors(
                df,
                from_year=PE_PUF_SOI_END_YEAR,
                to_year=target_year,
                policyengine_us_data_repo=policyengine_us_data_repo,
            )
    else:
        df = uprate_puf(df, from_year=2015, to_year=target_year)

    # Expand to persons if requested
    if expand_persons:
        df = expand_to_persons(df)
        strategy = _normalize_social_security_split_strategy(
            social_security_split_strategy
        )
        try:
            if strategy == SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF:
                share_model = _default_pe_style_puf_social_security_share_model(
                    cps_reference_year=_default_cps_reference_year(target_year),
                    cache_dir=cache_dir,
                )
            elif strategy == SOCIAL_SECURITY_SPLIT_STRATEGY_AGE_HEURISTIC:
                share_model = _age_heuristic_puf_social_security_share_model()
            else:
                share_model = _default_puf_social_security_share_model(
                    cps_reference_year=_default_cps_reference_year(target_year),
                    cache_dir=cache_dir,
                )
        except (FileNotFoundError, ImportError, ValueError):
            share_model = _age_heuristic_puf_social_security_share_model()
        df = _impute_puf_social_security_components(df, share_model=share_model)

    print(f"\nPUF loaded: {len(df):,} records")
    print(f"  Weight sum: {df['weight'].sum():,.0f}")

    return df


# Variables that PUF has but CPS doesn't (will be NaN in CPS)
PUF_EXCLUSIVE_VARS = [
    "pre_tax_contributions",
    "short_term_capital_gains",
    "long_term_capital_gains",
    "non_sch_d_capital_gains",
    "partnership_s_corp_income",
    "qualified_dividend_income",
    "tax_exempt_interest_income",
    "charitable_cash",
    "charitable_noncash",
    "mortgage_interest_paid",
    "state_income_tax_paid",
    "real_estate_tax_paid",
    "student_loan_interest",
    "ira_deduction",
]

# Variables that both surveys have (may differ in quality)
SHARED_VARS = [
    "employment_income",
    "self_employment_income",
    "taxable_interest_income",
    "ordinary_dividend_income",
    "rental_income",
    "gross_social_security",
    "taxable_pension_income",
    "unemployment_compensation",
    "age",
    "filing_status",
]


def _sample_tax_units(
    tax_units: pd.DataFrame,
    *,
    sample_n: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Sample tax units before expanding them to persons."""
    if sample_n is None or sample_n >= len(tax_units):
        return tax_units.reset_index(drop=True)
    sample_weights: pd.Series | None = None
    weight_column = next(
        (
            candidate
            for candidate in ("weight", "S006", "household_weight")
            if candidate in tax_units.columns
        ),
        None,
    )
    if weight_column is not None:
        candidate_weights = (
            pd.to_numeric(tax_units[weight_column], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        if (
            candidate_weights.sum() > 0.0
            and int((candidate_weights > 0.0).sum()) >= sample_n
        ):
            sample_weights = candidate_weights
    try:
        return tax_units.sample(
            n=sample_n,
            random_state=random_seed,
            replace=False,
            weights=sample_weights,
        ).reset_index(drop=True)
    except ValueError:
        # Match CPS behavior: if weighted sampling without replacement is
        # infeasible at high sample sizes, fall back to deterministic uniform
        # sampling instead of failing the run.
        return tax_units.sample(
            n=sample_n,
            random_state=random_seed,
            replace=False,
            weights=None,
        ).reset_index(drop=True)


def _build_puf_tax_units(
    *,
    raw: pd.DataFrame,
    target_year: int,
    random_seed: int = 42,
    uprating_mode: str = PUF_UPRATING_MODE_INTERPOLATED,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    impute_pre_tax_contributions: bool = False,
    pre_tax_training_year: int = 2024,
    soi_path: str | Path | None = None,
    require_pre_tax_contribution_model: bool = False,
) -> pd.DataFrame:
    """Map raw PUF records into a normalized tax-unit table."""
    resolved_uprating_mode = _normalize_puf_uprating_mode(uprating_mode)
    if resolved_uprating_mode == PUF_UPRATING_MODE_PE_SOI:
        raw_uprating_year = min(int(target_year), PE_PUF_SOI_END_YEAR)
        raw = uprate_raw_puf_pe_style(
            raw,
            from_year=2015,
            to_year=raw_uprating_year,
            policyengine_us_data_repo=policyengine_us_data_repo,
            soi_path=soi_path,
        )
    tax_units = map_puf_variables(
        raw,
        random_seed=random_seed,
        impute_pre_tax_contributions=impute_pre_tax_contributions,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        pre_tax_training_year=pre_tax_training_year,
        require_pre_tax_contribution_model=require_pre_tax_contribution_model,
    )
    if resolved_uprating_mode == PUF_UPRATING_MODE_PE_SOI:
        if target_year > PE_PUF_SOI_END_YEAR:
            tax_units = uprate_mapped_puf_with_pe_factors(
                tax_units,
                from_year=PE_PUF_SOI_END_YEAR,
                to_year=target_year,
                policyengine_us_data_repo=policyengine_us_data_repo,
            )
    else:
        tax_units = uprate_puf(tax_units, from_year=2015, to_year=target_year)
    identifier = (
        raw["RECID"].astype(str).reset_index(drop=True)
        if "RECID" in raw.columns
        else pd.Series(np.arange(len(raw)).astype(str))
    )
    tax_units = tax_units.reset_index(drop=True)
    tax_units["household_id"] = identifier
    tax_units["year"] = target_year
    tax_units["state_fips"] = 0
    tax_units["tenure"] = 0
    tax_units["household_weight"] = tax_units["weight"].astype(float)
    tax_units = _add_derived_income_columns(tax_units)
    is_male = tax_units.get("is_male", pd.Series(np.nan, index=tax_units.index)).fillna(
        0
    )
    tax_units["sex"] = np.where(is_male > 0, 1, np.where(is_male == 0, 2, 0))
    tax_units["education"] = 0
    return tax_units


def _tax_units_to_persons(
    tax_units: pd.DataFrame,
    *,
    expand_persons_flag: bool,
) -> pd.DataFrame:
    """Expand tax units into a person table."""
    if expand_persons_flag:
        persons = expand_to_persons(tax_units)
    else:
        persons = tax_units.copy()
        persons["is_head"] = 1
        persons["is_spouse"] = 0
        persons["is_dependent"] = 0
        persons["person_id"] = persons["household_id"].astype(str) + ":head"
        persons["tax_unit_id"] = persons["household_id"].astype(str)
    persons = persons.reset_index(drop=True)
    persons["person_id"] = persons["person_id"].astype(str)
    persons["household_id"] = persons["household_id"].astype(str)
    persons["year"] = tax_units["year"].iloc[0] if not tax_units.empty else 2024
    if "income" not in persons.columns:
        persons["income"] = tax_units["income"]
    if "employment_status" not in persons.columns:
        persons["employment_status"] = tax_units["employment_status"]
    if "education" not in persons.columns:
        persons["education"] = 0
    if "age" not in persons.columns:
        persons["age"] = 0
    if "sex" not in persons.columns:
        persons["sex"] = 0
    return persons


def _build_puf_observation_frame(
    *,
    tax_units: pd.DataFrame,
    persons: pd.DataFrame,
    source_name: str,
    shareability: Shareability,
) -> ObservationFrame:
    """Build an observation frame from normalized PUF tax units."""
    manifest = load_us_source_manifest("puf")
    households = tax_units[
        ["household_id", "year", "state_fips", "tenure", "household_weight"]
    ].copy()
    person_variable_names = tuple(
        column
        for column in persons.columns
        if column not in {"person_id", "household_id", "weight", "year"}
    )
    descriptor = SourceDescriptor(
        name=source_name,
        shareability=shareability,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        archetype=manifest.archetype,
        population=manifest.population,
        description=manifest.description,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=("state_fips", "tenure"),
                weight_column="household_weight",
                period_column="year",
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=person_variable_names,
                weight_column="weight" if "weight" in persons.columns else None,
                period_column="year",
            ),
        ),
        variable_capabilities=resolve_source_variable_capabilities(
            source_name,
            ("state_fips", "tenure", *person_variable_names),
        ),
    )
    frame = ObservationFrame(
        source=descriptor,
        tables={
            EntityType.HOUSEHOLD: households,
            EntityType.PERSON: persons,
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


@dataclass
class PUFSourceProvider:
    """Source-provider wrapper around the IRS SOI PUF."""

    target_year: int = 2024
    cache_dir: Path | None = None
    puf_path: str | Path | None = None
    demographics_path: str | Path | None = None
    expand_persons: bool = True
    uprating_mode: str = PUF_UPRATING_MODE_INTERPOLATED
    cps_reference_year: int | None = None
    policyengine_us_data_repo: str | Path | None = None
    policyengine_us_data_python: str | Path | None = None
    impute_pre_tax_contributions: bool = False
    pre_tax_training_year: int = 2024
    soi_path: str | Path | None = None
    require_pre_tax_contribution_model: bool = False
    social_security_split_strategy: str = SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE
    shareability: Shareability = Shareability.PUBLIC
    loader: Callable[[Path | None], tuple[Path, Path | None]] | None = None
    social_security_share_model_loader: (
        Callable[[int, Path | None], SocialSecurityShareModel] | None
    ) = None
    _descriptor_cache: SourceDescriptor | None = None
    _social_security_share_model_cache: dict[
        tuple[int, str], SocialSecurityShareModel
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def descriptor(self) -> SourceDescriptor:
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        manifest = load_us_source_manifest("puf")
        person_variables = ("age", "sex", "income")
        return SourceDescriptor(
            name="irs_soi_puf",
            shareability=self.shareability,
            time_structure=TimeStructure.REPEATED_CROSS_SECTION,
            archetype=manifest.archetype,
            population=manifest.population,
            description=manifest.description,
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column="household_id",
                    variable_names=("state_fips", "tenure"),
                    weight_column="household_weight",
                    period_column="year",
                ),
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column="person_id",
                    variable_names=person_variables,
                    weight_column="weight",
                    period_column="year",
                ),
            ),
            variable_capabilities=resolve_source_variable_capabilities(
                "irs_soi_puf",
                ("state_fips", "tenure", *person_variables),
            ),
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        query = query or SourceQuery()
        provider_filters = query.provider_filters
        target_year = int(provider_filters.get("target_year", self.target_year))
        expand_persons_flag = bool(
            provider_filters.get("expand_persons", self.expand_persons)
        )
        cps_reference_year = int(
            provider_filters.get(
                "cps_reference_year",
                self.cps_reference_year or _default_cps_reference_year(target_year),
            )
        )
        uprating_mode = _normalize_puf_uprating_mode(
            provider_filters.get("uprating_mode", self.uprating_mode)
        )
        social_security_split_strategy = _normalize_social_security_split_strategy(
            provider_filters.get(
                "social_security_split_strategy",
                self.social_security_split_strategy,
            )
        )
        puf_path = provider_filters.get("puf_path", self.puf_path)
        demographics_path = provider_filters.get(
            "demographics_path",
            self.demographics_path,
        )
        policyengine_us_data_repo = provider_filters.get(
            "policyengine_us_data_repo",
            self.policyengine_us_data_repo,
        )
        policyengine_us_data_python = provider_filters.get(
            "policyengine_us_data_python",
            self.policyengine_us_data_python,
        )
        impute_pre_tax_contributions = bool(
            provider_filters.get(
                "impute_pre_tax_contributions",
                self.impute_pre_tax_contributions,
            )
        )
        pre_tax_training_year = int(
            provider_filters.get(
                "pre_tax_training_year",
                self.pre_tax_training_year,
            )
        )
        soi_path = provider_filters.get("soi_path", self.soi_path)
        require_pre_tax_contribution_model = bool(
            provider_filters.get(
                "require_pre_tax_contribution_model",
                self.require_pre_tax_contribution_model,
            )
        )
        if puf_path is None:
            local_repo_paths = _resolve_policyengine_repo_local_puf_paths(
                policyengine_us_data_repo
            )
            if local_repo_paths is not None:
                loaded_puf_path, loaded_demographics_path = local_repo_paths
                print(f"Using repo-local PUF from {loaded_puf_path}...")
            else:
                loader = self.loader or download_puf
                loaded_puf_path, loaded_demographics_path = loader(self.cache_dir)
            puf_path = loaded_puf_path
            if demographics_path is None:
                demographics_path = loaded_demographics_path

        raw = load_puf_raw(
            Path(puf_path),
            Path(demographics_path) if demographics_path is not None else None,
        )
        raw = _sample_tax_units(
            raw,
            sample_n=provider_filters.get("sample_n"),
            random_seed=int(provider_filters.get("random_seed", 0)),
        )
        tax_units = _build_puf_tax_units(
            raw=raw,
            target_year=target_year,
            random_seed=int(provider_filters.get("random_seed", 0)),
            uprating_mode=uprating_mode,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            impute_pre_tax_contributions=impute_pre_tax_contributions,
            pre_tax_training_year=pre_tax_training_year,
            soi_path=soi_path,
            require_pre_tax_contribution_model=require_pre_tax_contribution_model,
        )
        persons = _tax_units_to_persons(
            tax_units,
            expand_persons_flag=expand_persons_flag,
        )
        persons = _impute_puf_social_security_components(
            persons,
            share_model=self._load_social_security_share_model(
                cps_reference_year,
                social_security_split_strategy,
            ),
        )
        frame = _build_puf_observation_frame(
            tax_units=tax_units,
            persons=persons,
            source_name=f"irs_soi_puf_{target_year}",
            shareability=self.shareability,
        )
        self._descriptor_cache = frame.source
        return apply_source_query(frame, query)

    def _load_social_security_share_model(
        self,
        cps_reference_year: int,
        strategy: str,
    ) -> SocialSecurityShareModel:
        cache_key = (cps_reference_year, strategy)
        cached = self._social_security_share_model_cache.get(cache_key)
        if cached is not None:
            return cached
        loader = (
            self.social_security_share_model_loader
            or _strategy_social_security_share_model_loader(strategy)
        )
        try:
            model = loader(cps_reference_year, self.cache_dir)
        except (FileNotFoundError, ImportError, ValueError):
            model = _age_heuristic_puf_social_security_share_model()
        self._social_security_share_model_cache[cache_key] = model
        return model


if __name__ == "__main__":
    # Test loading
    df = load_puf(target_year=2024)
    print("\nSample of loaded PUF:")
    print(df.head())

    print("\nIncome variable sums:")
    income_vars = [
        "employment_income",
        "self_employment_income",
        "long_term_capital_gains",
        "partnership_s_corp_income",
        "gross_social_security",
        "taxable_pension_income",
    ]
    for var in income_vars:
        if var in df.columns:
            total = (df[var] * df["weight"]).sum() / 1e9
            print(f"  {var}: ${total:.1f}B")
