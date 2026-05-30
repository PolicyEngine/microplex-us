"""Library-first US microplex build pipeline."""

from __future__ import annotations

import logging
import sys
import time
import warnings
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
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
from microplex_us.pe_source_impute_engine import (
    PE_SOURCE_IMPUTE_BLOCK_ENGINE,
    PESourceImputeBlockRunRequest,
    PESourceImputeConditionedBlockRunRequest,
)
from microplex_us.pipelines.donor_imputers import (
    ColumnwiseQRFDonorImputer,
    RegimeAwareDonorImputer,
)
from microplex_us.pipelines.pe_l0 import PolicyEngineL0Calibrator
from microplex_us.pipelines.pe_native_optimization import (
    optimize_policyengine_us_native_loss_dataset,
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
        pd.to_numeric(weights.reindex(index), errors="coerce").fillna(0.0).clip(lower=0.0)
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
                weight_values.to_numpy(dtype=float)[group_mask & reported_positive].sum()
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

        seed_data = persons_df.merge(
            hh[["household_id", "state_fips", "county_fips", "hh_weight", "tenure"]],
            on="household_id",
            how="left",
            suffixes=("", "__household"),
        )
        for column in ("state_fips", "county_fips", "hh_weight", "tenure"):
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
        seed_data["county_fips"] = seed_data["county_fips"].fillna(0).astype(int)
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
            pd.to_numeric(persons["ssi"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
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

        households = self._build_policyengine_households(persons)
        tax_units, persons = self._build_policyengine_tax_units(persons)
        persons = self._assign_family_and_spm_units(persons)
        families = self._collapse_group_table(persons, "family_id")
        spm_units = self._collapse_group_table(persons, "spm_unit_id")
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
            if support_family
            in {
                VariableSupportFamily.ZERO_INFLATED_POSITIVE,
                VariableSupportFamily.BOUNDED_SHARE,
            }
        }
        if backend == "regime_aware":
            return RegimeAwareDonorImputer(
                condition_vars=condition_vars,
                target_vars=list(target_vars),
                n_estimators=self.config.donor_imputer_qrf_n_estimators,
                nonnegative_vars=nonnegative_vars,
                seed=self.config.random_seed,
            )
        zero_inflated_vars = (
            {
                variable
                for variable, support_family in support_families.items()
                if support_family is VariableSupportFamily.ZERO_INFLATED_POSITIVE
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
        )

        for donor_input in donor_inputs:
            donor_source_name = donor_input.frame.source.name
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
            donor_target_vars = sorted(set(donor_only_vars) | set(donor_override_vars))
            if not shared_vars or not donor_target_vars:
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
                    _emit_us_pipeline_progress(
                        "US microplex donor integration: block complete",
                        donor_source=donor_source_name,
                        block=block_label,
                        integrated_vars=len(result.integrated_variables),
                    )

        return {
            "seed_data": current,
            "integrated_variables": sorted(set(integrated_variables)),
            "conditioning_diagnostics": conditioning_diagnostics,
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

        if strategy is DonorMatchStrategy.ZERO_INFLATED_POSITIVE or (
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

    def _build_policyengine_tax_units_from_role_flags(
        self,
        persons: pd.DataFrame,
        *,
        start_tax_unit_id: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, set[Any]] | None:
        role_columns = {
            "is_tax_unit_head",
            "is_tax_unit_spouse",
            "is_tax_unit_dependent",
        }
        if not role_columns.issubset(persons.columns) or "person_id" not in persons.columns:
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
                        "member_ids": [
                            int(person_id) for person_id in unit_person_ids
                        ],
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
            normalized_tax_unit_id = (
                pd.factorize(pd.MultiIndex.from_frame(tax_unit_key), sort=False)[
                    0
                ].astype(np.int64)
                + int(start_tax_unit_id)
            )
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
            spouse_flag
            & ~resolved_dependent
            & (~head_flag | spouse_hint | ~head_hint)
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
                    .sort_values(
                        ["source_spouse", "relationship", "age", "person_id"]
                    )
                    .index[0]
                )
        if spouse_index is not None:
            coherent_spouse.loc[spouse_index] = True

        available = ~(coherent_head | coherent_spouse)
        coherent_dependent.loc[
            available
            & (
                dependent_flag
                | (
                    dependent_hint
                    & (age.lt(24) | income.le(0.0))
                )
                | (spouse_hint & income.le(0.0))
            )
        ] = True

        available = ~(coherent_head | coherent_spouse | coherent_dependent)
        coherent_head.loc[
            available
            & age.ge(18)
            & (head_flag | income.gt(0.0))
        ] = True

        coherent_dependent.loc[
            ~(coherent_head | coherent_spouse | coherent_dependent)
            & (age.lt(18) | dependent_hint | income.le(0.0))
        ] = True
        coherent_head.loc[~(coherent_head | coherent_spouse | coherent_dependent)] = True

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
            if int(row["person_id"]) in head_set
            and int(person_number.loc[index]) > 0
        }
        row_by_person_id = {
            int(row["person_id"]): index
            for index, row in household_persons.iterrows()
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
    ) -> dict[str, float]:
        columns = (
            "domestic_production_ald",
            "health_savings_account_ald",
            "recapture_of_investment_credit",
            "self_employed_health_insurance_ald",
            "self_employed_pension_contribution_ald",
            "unrecaptured_section_1250_gain",
            "unreported_payroll_tax",
        )
        aggregated: dict[str, float] = {}
        for column in columns:
            if column not in unit_persons.columns:
                continue
            values = pd.to_numeric(unit_persons[column], errors="coerce").fillna(0.0)
            nonzero_values = values.loc[~np.isclose(values.to_numpy(dtype=float), 0.0)]
            if len(nonzero_values) > 1 and nonzero_values.nunique(dropna=True) == 1:
                aggregated[column] = float(nonzero_values.iloc[0])
                continue
            aggregated[column] = float(values.sum())
        return aggregated

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
        result = persons.copy()
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

        result["family_id"] = result.index.map(family_ids).astype(np.int64)
        result["spm_unit_id"] = result.index.map(spm_unit_ids).astype(np.int64)
        return result

    def _primary_family_member_mask(
        self,
        household_persons: pd.DataFrame,
    ) -> pd.Series:
        """Identify people who belong to the household's primary family."""

        relationship_primary = household_persons["relationship_to_head"].isin(
            {0, 1, 2}
        )
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
        result = persons.copy()
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
                pd.to_numeric(result["is_blind"], errors="coerce")
                .fillna(0.0)
                .ne(0.0)
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
            first_present("self_employment_income")
            + first_present("taxable_interest_income", "interest_income")
            + first_present("ordinary_dividend_income", "dividend_income")
            + first_present("rental_income")
            + first_present("gross_social_security", "social_security")
            + first_present("ssi")
            + first_present("public_assistance")
            + first_present("taxable_pension_income", "pension_income")
            + first_present("unemployment_compensation")
        )
        fallback_employment_income = (
            pd.to_numeric(result.get("income", zero), errors="coerce")
            .fillna(0.0)
            .astype(float)
            - known_nonemployment
        ).clip(lower=0.0)

        result["employment_income_before_lsr"] = (
            first_present(
                "employment_income_before_lsr", "employment_income", "wage_income"
            )
            if has_any(
                "employment_income_before_lsr", "employment_income", "wage_income"
            )
            else fallback_employment_income
        )
        result["self_employment_income_before_lsr"] = first_present(
            "self_employment_income_before_lsr",
            "self_employment_income",
        )
        result["taxable_interest_income"] = first_present(
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
        result["ordinary_dividend_income"] = first_present(
            "ordinary_dividend_income",
            "dividend_income",
        ).clip(lower=0.0)
        if has_any("qualified_dividend_income", "non_qualified_dividend_income"):
            dividend_total = (
                result["qualified_dividend_income"]
                + result["non_qualified_dividend_income"]
            ).clip(lower=0.0)
            result["ordinary_dividend_income"] = dividend_total
            result["dividend_income"] = dividend_total
        else:
            result = normalize_dividend_columns(result)

        result["short_term_capital_gains"] = first_present("short_term_capital_gains")
        result["non_sch_d_capital_gains"] = first_present(
            "non_sch_d_capital_gains",
            "capital_gains_distributions",
        )
        result["long_term_capital_gains_before_response"] = (
            first_present(
                "long_term_capital_gains_before_response",
                "long_term_capital_gains",
            )
            if has_any(
                "long_term_capital_gains_before_response",
                "long_term_capital_gains",
            )
            else first_present("capital_gains")
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
