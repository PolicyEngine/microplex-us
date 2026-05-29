"""Concrete checkpoint runner for the PE-US-data rebuild profile."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd
from microplex.core import (
    EntityObservation,
    EntityType,
    ObservationFrame,
    SourceDescriptor,
    SourceQuery,
)
from microplex.targets import assert_valid_benchmark_artifact_manifest

from microplex_us.pipelines.artifacts import (
    USMicroplexArtifactPaths,
    USMicroplexVersionedBuildArtifacts,
    build_and_save_versioned_us_microplex_from_source_providers,
)
from microplex_us.pipelines.imputation_ablation import (
    ImputationAblationSliceSpec,
    ImputationAblationVariant,
    score_imputation_ablation_variants,
)
from microplex_us.pipelines.index_db import append_us_microplex_run_index_entry
from microplex_us.pipelines.pe_us_data_rebuild import (
    PEUSDataRebuildProgram,
    default_policyengine_us_data_rebuild_config,
    default_policyengine_us_data_rebuild_program,
    default_policyengine_us_data_rebuild_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild_audit import (
    build_policyengine_us_data_rebuild_native_audit,
)
from microplex_us.pipelines.pe_us_data_rebuild_parity import (
    build_policyengine_us_data_rebuild_parity_artifact,
    write_policyengine_us_data_rebuild_parity_artifact,
)
from microplex_us.pipelines.registry import (
    append_us_microplex_run_registry_entry,
    build_us_microplex_run_registry_entry,
    load_us_microplex_run_registry,
    select_us_microplex_frontier_entry,
)
from microplex_us.pipelines.stage_artifacts import (
    build_us_stage_artifact_inventory,
    write_us_stage_artifact_inventory,
)
from microplex_us.pipelines.stage_manifest import (
    write_us_stage_manifest,
    write_us_validation_evidence_manifest,
)
from microplex_us.pipelines.stage_readiness import (
    write_us_conditional_readiness_report,
)
from microplex_us.variables import prune_redundant_variables

if TYPE_CHECKING:
    from microplex.core import SourceProvider
    from microplex.targets import TargetProvider

    from microplex_us.pipelines.registry import FrontierMetric
    from microplex_us.pipelines.us import USMicroplexBuildConfig
    from microplex_us.policyengine.harness import (
        PolicyEngineUSComparisonCache,
        PolicyEngineUSHarnessSlice,
    )


DEFAULT_CHECKPOINT_IMPUTATION_ABLATION_EVAL_FRACTION = 0.25
MIN_CHECKPOINT_IMPUTATION_ABLATION_HOUSEHOLDS = 8
LOGGER = logging.getLogger(__name__)


def _root_logger_has_handlers() -> bool:
    return bool(logging.getLogger().handlers)


def _emit_checkpoint_progress(message: str, /, **context: object) -> None:
    details = ", ".join(
        f"{key}={value}"
        for key, value in context.items()
        if value is not None and value != ""
    )
    line = f"{message} [{details}]" if details else message
    LOGGER.info(line)
    if not LOGGER.handlers and not _root_logger_has_handlers():
        print(line, file=sys.stderr, flush=True)


def _resolve_checkpoint_calibration_target_variables(
    calibration_target_variables: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(calibration_target_variables)


@dataclass(frozen=True)
class PEUSDataRebuildCheckpointResult:
    """Saved artifact bundle plus attached PE comparison sidecars."""

    build_config: USMicroplexBuildConfig
    provider_names: tuple[str, ...]
    queries: dict[str, SourceQuery]
    artifacts: USMicroplexVersionedBuildArtifacts
    parity_path: Path
    parity_payload: dict[str, Any]
    native_audit_path: Path | None = None
    native_audit_payload: dict[str, Any] | None = None
    imputation_ablation_path: Path | None = None
    imputation_ablation_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class PEUSDataRebuildCheckpointEvidenceResult:
    """Comparison evidence attached to one saved rebuild artifact."""

    artifact_dir: Path
    manifest_path: Path
    harness_path: Path | None
    native_scores_path: Path | None
    parity_path: Path
    parity_payload: dict[str, Any]
    native_audit_path: Path | None = None
    native_audit_payload: dict[str, Any] | None = None
    imputation_ablation_path: Path | None = None
    imputation_ablation_payload: dict[str, Any] | None = None


def _normalize_path_value(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


def _validate_checkpoint_config_context(
    config: USMicroplexBuildConfig,
    *,
    policyengine_baseline_dataset: str | Path,
    policyengine_targets_db: str | Path,
    target_period: int,
    target_profile: str,
    calibration_target_profile: str | None,
    target_variables: tuple[str, ...],
    target_domains: tuple[str, ...],
    target_geo_levels: tuple[str, ...],
    calibration_target_variables: tuple[str, ...],
    calibration_target_domains: tuple[str, ...],
    calibration_target_geo_levels: tuple[str, ...],
) -> None:
    expected_pairs = {
        "policyengine_baseline_dataset": _normalize_path_value(
            policyengine_baseline_dataset
        ),
        "policyengine_targets_db": _normalize_path_value(policyengine_targets_db),
        "policyengine_dataset_year": int(target_period),
        "policyengine_target_period": int(target_period),
        "policyengine_target_profile": target_profile,
        "policyengine_calibration_target_profile": (
            calibration_target_profile or target_profile
        ),
        "policyengine_target_variables": tuple(target_variables),
        "policyengine_target_domains": tuple(target_domains),
        "policyengine_target_geo_levels": tuple(target_geo_levels),
        "policyengine_calibration_target_variables": (
            _resolve_checkpoint_calibration_target_variables(
                calibration_target_variables
            )
        ),
        "policyengine_calibration_target_domains": tuple(calibration_target_domains),
        "policyengine_calibration_target_geo_levels": tuple(
            calibration_target_geo_levels
        ),
    }
    for key, expected in expected_pairs.items():
        observed = getattr(config, key)
        if observed != expected:
            raise ValueError(
                "Explicit config does not match the requested PE rebuild context for "
                f"{key}: expected {expected!r}, observed {observed!r}"
            )


def _validate_query_keys(
    provider_names: tuple[str, ...],
    queries: dict[str, SourceQuery],
) -> None:
    unexpected = sorted(set(queries) - set(provider_names))
    if unexpected:
        allowed = ", ".join(provider_names)
        unexpected_text = ", ".join(unexpected)
        raise ValueError(
            "Checkpoint queries include unknown provider keys: "
            f"{unexpected_text}. Expected one of: {allowed}"
        )


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def _resolve_policyengine_us_runtime_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("policyengine-us")
    except PackageNotFoundError:
        return None


def _registry_metric_value(entry: Any | None, metric: FrontierMetric) -> float | None:
    if entry is None:
        return None
    return getattr(entry, metric, None)


def _resolve_saved_artifact_path(
    artifact_root: Path,
    relative_or_absolute: str | Path | None,
) -> Path | None:
    if relative_or_absolute is None:
        return None
    candidate = Path(relative_or_absolute)
    if not candidate.is_absolute():
        artifact_relative = artifact_root / candidate
        if artifact_relative.exists():
            return artifact_relative
        cwd_relative = candidate.resolve()
        if cwd_relative.exists():
            return cwd_relative
        candidate = artifact_relative
    return candidate


def _infer_policyengine_baseline_household_weight_sum(
    baseline_dataset: str | Path,
    *,
    target_period: int,
) -> float | None:
    """Best-effort household-weight target inferred from the PE baseline dataset."""

    dataset_path = Path(baseline_dataset).expanduser()
    if not dataset_path.exists():
        return None
    try:
        with h5py.File(dataset_path, "r") as handle:
            weights = handle.get("household_weight")
            if weights is None:
                return None
            period_key = str(int(target_period))
            if period_key not in weights:
                return None
            weight_sum = float(weights[period_key][...].sum())
    except (FileNotFoundError, OSError, ValueError):
        return None
    return weight_sum if weight_sum > 0.0 else None


def _checkpoint_imputation_ablation_variants() -> tuple[ImputationAblationVariant, ...]:
    return (
        ImputationAblationVariant(
            name="broad_common_qrf",
            description="QRF with every compatible shared predictor.",
            condition_selection="all_shared",
        ),
        ImputationAblationVariant(
            name="top_correlated_qrf",
            description="QRF with the production top-correlated predictor selection.",
            condition_selection="top_correlated",
        ),
        ImputationAblationVariant(
            name="structured_pe_conditioning",
            description="PolicyEngine-style structural conditioning and preferred predictors.",
            condition_selection="pe_prespecified",
        ),
    )


def _checkpoint_imputation_ablation_slice_specs() -> tuple[
    ImputationAblationSliceSpec, ...
]:
    return (
        ImputationAblationSliceSpec(
            name="state_by_age",
            columns=("state_fips", "age_group"),
        ),
        ImputationAblationSliceSpec(
            name="sex_by_age",
            columns=("sex", "age_group"),
        ),
        ImputationAblationSliceSpec(
            name="employment_by_income",
            columns=("employment_status", "income_bracket"),
        ),
    )


def _production_imputation_ablation_variant_name(config: Any) -> str:
    condition_selection = getattr(config, "donor_imputer_condition_selection", None)
    if condition_selection == "all_shared":
        return "broad_common_qrf"
    if condition_selection == "top_correlated":
        return "top_correlated_qrf"
    return "structured_pe_conditioning"


def _checkpoint_post_calibration_metrics(
    manifest: dict[str, Any],
    *,
    production_variant: str,
) -> dict[str, dict[str, float]]:
    calibration_summary = dict(manifest.get("calibration", {}))
    harness_summary = dict(manifest.get("policyengine_harness", {}))
    native_scores_summary = dict(manifest.get("policyengine_native_scores", {}))
    metrics: dict[str, float] = {}
    for key in (
        "full_oracle_capped_mean_abs_relative_error",
        "full_oracle_mean_abs_relative_error",
        "active_solve_capped_mean_abs_relative_error",
        "active_solve_mean_abs_relative_error",
    ):
        value = calibration_summary.get(key)
        if value is not None:
            metrics[key] = float(value)
    for key in (
        "candidate_mean_abs_relative_error",
        "mean_abs_relative_error_delta",
        "candidate_composite_parity_loss",
        "composite_parity_loss_delta",
        "target_win_rate",
    ):
        value = harness_summary.get(key)
        if value is not None:
            metrics[key] = float(value)
    for key in (
        "candidate_enhanced_cps_native_loss",
        "enhanced_cps_native_loss_delta",
    ):
        value = native_scores_summary.get(key)
        if value is not None:
            metrics[key] = float(value)
    return {production_variant: metrics} if metrics else {}


def _build_checkpoint_source_descriptor(
    *,
    base_source: SourceDescriptor,
    household_table: pd.DataFrame,
    person_table: pd.DataFrame,
    household_variables: set[str] | None = None,
    person_variables: set[str] | None = None,
    name: str | None = None,
) -> SourceDescriptor | None:
    def _build_observation(
        entity: EntityType,
        table: pd.DataFrame,
        allowed_variables: set[str] | None,
    ) -> EntityObservation | None:
        observation = base_source.observation_for(entity)
        available_columns = set(table.columns)
        if observation.key_column not in available_columns:
            return None
        variable_names = tuple(
            variable
            for variable in observation.variable_names
            if variable in available_columns
            and (allowed_variables is None or variable in allowed_variables)
        )
        if not variable_names:
            return None
        return EntityObservation(
            entity=entity,
            key_column=observation.key_column,
            variable_names=variable_names,
            weight_column=(
                observation.weight_column
                if observation.weight_column in available_columns
                else None
            ),
            period_column=(
                observation.period_column
                if observation.period_column in available_columns
                else None
            ),
        )

    household_observation = _build_observation(
        EntityType.HOUSEHOLD,
        household_table,
        household_variables,
    )
    person_observation = _build_observation(
        EntityType.PERSON,
        person_table,
        person_variables,
    )
    if household_observation is None or person_observation is None:
        return None

    included_variables = set(household_observation.variable_names) | set(
        person_observation.variable_names
    )
    return SourceDescriptor(
        name=name or base_source.name,
        shareability=base_source.shareability,
        time_structure=base_source.time_structure,
        observations=(household_observation, person_observation),
        archetype=base_source.archetype,
        population=base_source.population,
        description=base_source.description,
        variable_capabilities={
            variable: capability
            for variable, capability in base_source.variable_capabilities.items()
            if variable in included_variables
        },
    )


def _household_person_relationship(frame: ObservationFrame) -> Any:
    relationship = next(
        (
            candidate
            for candidate in frame.relationships
            if candidate.parent_entity == EntityType.HOUSEHOLD
            and candidate.child_entity == EntityType.PERSON
        ),
        None,
    )
    if relationship is None:
        raise ValueError(
            "Checkpoint imputation ablation requires a household-to-person relationship"
        )
    return relationship


def _project_checkpoint_table_to_source_schema(
    table: pd.DataFrame,
    observation: EntityObservation,
    *,
    relationship_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    columns = [
        observation.key_column,
        *relationship_columns,
        *observation.variable_names,
    ]
    if observation.weight_column is not None:
        columns.append(observation.weight_column)
    if observation.period_column is not None:
        columns.append(observation.period_column)
    resolved_columns = [
        column for column in dict.fromkeys(columns) if column in table.columns
    ]
    return table.loc[:, resolved_columns].copy()


def _subset_checkpoint_frame_to_households(
    frame: ObservationFrame,
    household_ids: tuple[Any, ...],
    *,
    source: SourceDescriptor,
) -> ObservationFrame | None:
    relationship = _household_person_relationship(frame)
    households = frame.tables[EntityType.HOUSEHOLD]
    persons = frame.tables[EntityType.PERSON]
    household_subset = households.loc[
        households[relationship.parent_key].isin(household_ids)
    ].copy()
    if household_subset.empty:
        return None
    household_id_index = tuple(household_subset[relationship.parent_key].tolist())
    person_subset = persons.loc[
        persons[relationship.child_key].isin(household_id_index)
    ].copy()
    if person_subset.empty:
        return None
    household_observation = source.observation_for(EntityType.HOUSEHOLD)
    person_observation = source.observation_for(EntityType.PERSON)
    subset_frame = ObservationFrame(
        source=source,
        tables={
            EntityType.HOUSEHOLD: _project_checkpoint_table_to_source_schema(
                household_subset,
                household_observation,
                relationship_columns=(relationship.parent_key,),
            ),
            EntityType.PERSON: _project_checkpoint_table_to_source_schema(
                person_subset,
                person_observation,
                relationship_columns=(relationship.child_key,),
            ),
        },
        relationships=(relationship,),
    )
    subset_frame.validate()
    return subset_frame


def _split_checkpoint_household_ids(
    frame: ObservationFrame,
    *,
    eval_fraction: float,
    random_seed: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    relationship = _household_person_relationship(frame)
    household_ids = (
        frame.tables[EntityType.HOUSEHOLD][relationship.parent_key]
        .drop_duplicates()
        .tolist()
    )
    if len(household_ids) < MIN_CHECKPOINT_IMPUTATION_ABLATION_HOUSEHOLDS:
        return None
    shuffled = np.asarray(household_ids, dtype=object)
    np.random.default_rng(random_seed).shuffle(shuffled)
    eval_count = int(np.ceil(len(shuffled) * float(eval_fraction)))
    eval_count = max(1, min(eval_count, len(shuffled) - 1))
    eval_ids = tuple(shuffled[:eval_count].tolist())
    train_ids = tuple(shuffled[eval_count:].tolist())
    if not train_ids or not eval_ids:
        return None
    return train_ids, eval_ids


def _build_checkpoint_holdout_scaffold_source(
    scaffold_source: SourceDescriptor,
    donor_frame: ObservationFrame,
    *,
    masked_target_variables: set[str] | None = None,
) -> SourceDescriptor | None:
    excluded_variables = set(masked_target_variables or ())
    return _build_checkpoint_source_descriptor(
        base_source=scaffold_source,
        household_table=donor_frame.tables[EntityType.HOUSEHOLD],
        person_table=donor_frame.tables[EntityType.PERSON],
        household_variables=set(scaffold_source.variables_for(EntityType.HOUSEHOLD))
        - excluded_variables,
        person_variables=set(scaffold_source.variables_for(EntityType.PERSON))
        - excluded_variables,
        name=f"{donor_frame.source.name}_checkpoint_scaffold",
    )


def _resolve_checkpoint_imputation_targets(
    pipeline: Any,
    *,
    scaffold_input: Any,
    donor_input: Any,
    current_seed: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    scaffold_observed = prune_redundant_variables(
        scaffold_input.fusion_plan.variables_for(EntityType.HOUSEHOLD)
        | scaffold_input.fusion_plan.variables_for(EntityType.PERSON)
    )
    donor_seed = pipeline.prepare_seed_data_from_source(donor_input)
    donor_observed = prune_redundant_variables(
        donor_input.fusion_plan.variables_for(EntityType.HOUSEHOLD)
        | donor_input.fusion_plan.variables_for(EntityType.PERSON)
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
    numeric_current = {
        column
        for column in current_seed.columns
        if pd.api.types.is_numeric_dtype(current_seed[column])
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
        and variable in current_seed.columns
        and variable in donor_seed.columns
        and variable in numeric_current
        and variable in numeric_donor
        and scaffold_input.frame.source.allows_conditioning_on(variable)
        and donor_input.frame.source.allows_conditioning_on(variable)
        and pipeline._is_compatible_donor_condition(
            current_seed[variable],
            donor_seed[variable],
        )
    )
    donor_only_vars = sorted(
        variable
        for variable in donor_observed - scaffold_observed
        if variable not in excluded
        and variable not in pipeline.config.donor_imputer_excluded_variables
        and variable in donor_seed.columns
        and variable in numeric_donor
        and donor_input.frame.source.is_authoritative_for(variable)
        and pipeline._should_integrate_donor_variable(current_seed, variable)
        and pipeline._is_compatible_donor_target(donor_seed[variable])
    )
    donor_override_vars = sorted(
        variable
        for variable in scaffold_observed & donor_observed
        if variable not in excluded
        and variable not in pipeline.config.donor_imputer_excluded_variables
        and variable in pipeline.config.donor_imputer_authoritative_override_variables
        and variable in current_seed.columns
        and variable in donor_seed.columns
        and variable in numeric_current
        and variable in numeric_donor
        and donor_input.frame.source.is_authoritative_for(variable)
        and pipeline._is_compatible_donor_target(donor_seed[variable])
    )
    return shared_vars, sorted(set(donor_only_vars) | set(donor_override_vars))


def _checkpoint_variant_config(
    config: Any,
    variant: ImputationAblationVariant,
) -> Any:
    return replace(
        config,
        donor_imputer_condition_selection=variant.condition_selection,
        donor_imputer_max_condition_vars=(
            None
            if variant.condition_selection == "all_shared"
            else config.donor_imputer_max_condition_vars
        ),
    )


def _prepare_checkpoint_imputation_score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ages = (
        pd.to_numeric(result["age"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if "age" in result.columns
        else pd.Series(np.nan, index=result.index, dtype=float)
    )
    age_groups = pd.cut(
        ages,
        bins=[-np.inf, 18.0, 35.0, 55.0, 65.0, np.inf],
        labels=False,
        right=False,
    )
    result["age_group"] = (
        pd.Series(age_groups, index=result.index).fillna(-1).astype(int)
    )
    incomes = (
        pd.to_numeric(result["income"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if "income" in result.columns
        else pd.Series(np.nan, index=result.index, dtype=float)
    )
    income_brackets = pd.cut(
        incomes,
        bins=[-np.inf, 0.0, 25_000.0, 50_000.0, 100_000.0, np.inf],
        labels=False,
        right=False,
    )
    result["income_bracket"] = (
        pd.Series(
            income_brackets,
            index=result.index,
        )
        .fillna(-1)
        .astype(int)
    )
    return result


def _ensure_checkpoint_target_columns(
    frame: pd.DataFrame,
    *,
    target_variables: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    for variable in target_variables:
        if variable not in result.columns:
            result[variable] = 0.0
    return result


def _mean_checkpoint_metric(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _summarize_checkpoint_imputation_ablation(
    *,
    source_reports: dict[str, dict[str, Any]],
    skipped_sources: list[dict[str, Any]],
    production_variant: str,
) -> dict[str, Any]:
    metric_names = (
        "mean_weighted_mae",
        "mean_total_relative_error",
        "mean_support_f1",
        "mean_slice_total_js_divergence",
        "mean_slice_support_js_divergence",
        "mean_slice_positive_rate_delta",
    )
    variant_metrics: dict[str, dict[str, list[float]]] = {}
    variant_source_counts: dict[str, int] = {}
    target_count = 0
    for source_report in source_reports.values():
        target_count += len(source_report.get("target_variables", ()))
        report_payload = dict(source_report.get("report", {}))
        for variant_name, variant_payload in dict(
            report_payload.get("variants", {})
        ).items():
            aggregate_metrics = dict(variant_payload.get("aggregate_metrics", {}))
            variant_source_counts[variant_name] = (
                variant_source_counts.get(variant_name, 0) + 1
            )
            metric_buckets = variant_metrics.setdefault(
                variant_name,
                {metric_name: [] for metric_name in metric_names},
            )
            for metric_name in metric_names:
                value = aggregate_metrics.get(metric_name)
                if value is not None:
                    metric_buckets[metric_name].append(float(value))
    variant_scorecard: dict[str, dict[str, Any]] = {}
    for variant_name, metric_buckets in variant_metrics.items():
        variant_scorecard[variant_name] = {
            "source_count": variant_source_counts.get(variant_name, 0),
            **{
                metric_name: _mean_checkpoint_metric(metric_values)
                for metric_name, metric_values in metric_buckets.items()
            },
        }

    best_mean_weighted_mae_variant = None
    mae_candidates = [
        (payload.get("mean_weighted_mae"), variant_name)
        for variant_name, payload in variant_scorecard.items()
        if payload.get("mean_weighted_mae") is not None
    ]
    if mae_candidates:
        best_mean_weighted_mae_variant = min(mae_candidates)[1]

    best_mean_support_f1_variant = None
    f1_candidates = [
        (payload.get("mean_support_f1"), variant_name)
        for variant_name, payload in variant_scorecard.items()
        if payload.get("mean_support_f1") is not None
    ]
    if f1_candidates:
        best_mean_support_f1_variant = max(f1_candidates)[1]

    production_scorecard = variant_scorecard.get(production_variant, {})
    return {
        "source_count": len(source_reports),
        "skipped_source_count": len(skipped_sources),
        "target_count": target_count,
        "production_variant": production_variant,
        "production_mean_weighted_mae": production_scorecard.get("mean_weighted_mae"),
        "production_mean_support_f1": production_scorecard.get("mean_support_f1"),
        "best_mean_weighted_mae_variant": best_mean_weighted_mae_variant,
        "best_mean_support_f1_variant": best_mean_support_f1_variant,
        "variant_scorecard": variant_scorecard,
    }


def _build_checkpoint_imputation_ablation_payload(
    build_result: Any,
    *,
    artifact_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if build_result.source_frame is None or not build_result.source_frames:
        return None

    from microplex_us.pipelines.us import USMicroplexPipeline

    pipeline = USMicroplexPipeline(build_result.config)
    scaffold_input = pipeline.prepare_source_input(build_result.source_frame)
    scaffold_seed = pipeline.prepare_seed_data_from_source(scaffold_input)
    production_variant = _production_imputation_ablation_variant_name(
        build_result.config
    )
    variants = _checkpoint_imputation_ablation_variants()
    slice_specs = _checkpoint_imputation_ablation_slice_specs()
    source_reports: dict[str, dict[str, Any]] = {}
    skipped_sources: list[dict[str, Any]] = []

    for source_index, donor_frame in enumerate(build_result.source_frames):
        if donor_frame.source.name == build_result.source_frame.source.name:
            continue
        donor_name = donor_frame.source.name
        try:
            donor_input = pipeline.prepare_source_input(donor_frame)
            shared_vars, target_vars = _resolve_checkpoint_imputation_targets(
                pipeline,
                scaffold_input=scaffold_input,
                donor_input=donor_input,
                current_seed=scaffold_seed,
            )
            if not shared_vars:
                skipped_sources.append(
                    {"source_name": donor_name, "reason": "no_shared_condition_vars"}
                )
                continue
            if not target_vars:
                skipped_sources.append(
                    {"source_name": donor_name, "reason": "no_imputable_target_vars"}
                )
                continue

            donor_subset_source = _build_checkpoint_source_descriptor(
                base_source=donor_frame.source,
                household_table=donor_frame.tables[EntityType.HOUSEHOLD],
                person_table=donor_frame.tables[EntityType.PERSON],
                name=donor_name,
            )
            if donor_subset_source is None:
                skipped_sources.append(
                    {
                        "source_name": donor_name,
                        "reason": "missing_household_or_person_observations",
                    }
                )
                continue

            household_split = _split_checkpoint_household_ids(
                donor_frame,
                eval_fraction=DEFAULT_CHECKPOINT_IMPUTATION_ABLATION_EVAL_FRACTION,
                random_seed=int(build_result.config.random_seed) + source_index,
            )
            if household_split is None:
                skipped_sources.append(
                    {"source_name": donor_name, "reason": "insufficient_households"}
                )
                continue
            train_households, eval_households = household_split

            train_frame = _subset_checkpoint_frame_to_households(
                donor_frame,
                train_households,
                source=donor_subset_source,
            )
            observed_eval_frame = _subset_checkpoint_frame_to_households(
                donor_frame,
                eval_households,
                source=donor_subset_source,
            )
            holdout_scaffold_source = _build_checkpoint_holdout_scaffold_source(
                build_result.source_frame.source,
                donor_frame,
                masked_target_variables=set(target_vars),
            )
            if holdout_scaffold_source is None:
                skipped_sources.append(
                    {
                        "source_name": donor_name,
                        "reason": "no_overlap_with_scaffold_schema",
                    }
                )
                continue
            scaffold_eval_frame = _subset_checkpoint_frame_to_households(
                donor_frame,
                eval_households,
                source=holdout_scaffold_source,
            )
            if (
                train_frame is None
                or observed_eval_frame is None
                or scaffold_eval_frame is None
            ):
                skipped_sources.append(
                    {"source_name": donor_name, "reason": "empty_train_or_eval_split"}
                )
                continue

            observed_eval_seed = _prepare_checkpoint_imputation_score_frame(
                pipeline.prepare_seed_data_from_source(
                    pipeline.prepare_source_input(observed_eval_frame)
                )
            )
            imputed_frames: dict[str, pd.DataFrame] = {}
            for variant in variants:
                variant_pipeline = USMicroplexPipeline(
                    _checkpoint_variant_config(build_result.config, variant)
                )
                scaffold_eval_input = variant_pipeline.prepare_source_input(
                    scaffold_eval_frame
                )
                donor_train_input = variant_pipeline.prepare_source_input(train_frame)
                masked_seed = variant_pipeline.prepare_seed_data_from_source(
                    scaffold_eval_input
                )
                integrated = variant_pipeline._integrate_donor_sources(
                    masked_seed,
                    scaffold_input=scaffold_eval_input,
                    donor_inputs=[donor_train_input],
                )["seed_data"]
                imputed_frames[variant.name] = (
                    _prepare_checkpoint_imputation_score_frame(
                        _ensure_checkpoint_target_columns(
                            integrated,
                            target_variables=target_vars,
                        )
                    )
                )

            report = score_imputation_ablation_variants(
                observed_frame=observed_eval_seed,
                imputed_frames=imputed_frames,
                target_variables=target_vars,
                slice_specs=slice_specs,
                variants=variants,
                weight_column="hh_weight"
                if "hh_weight" in observed_eval_seed.columns
                else None,
                post_calibration_metrics=_checkpoint_post_calibration_metrics(
                    manifest,
                    production_variant=production_variant,
                ),
            )
            source_reports[donor_name] = {
                "source_name": donor_name,
                "shared_variables": shared_vars,
                "target_variables": target_vars,
                "train_household_count": len(train_households),
                "eval_household_count": len(eval_households),
                "report": report.to_dict(),
            }
        except (KeyError, ValueError) as exc:
            skipped_sources.append(
                {
                    "source_name": donor_name,
                    "reason": "source_evaluation_failed",
                    "detail": str(exc),
                }
            )

    if not source_reports:
        return None

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_id": artifact_id,
        "production_variant": production_variant,
        "summary": _summarize_checkpoint_imputation_ablation(
            source_reports=source_reports,
            skipped_sources=skipped_sources,
            production_variant=production_variant,
        ),
        "source_reports": source_reports,
        "skipped_sources": skipped_sources,
    }


def _build_checkpoint_benchmark_stage(
    manifest: dict[str, Any],
    *,
    extra_outputs: tuple[str, ...] = (),
) -> dict[str, Any]:
    artifacts = dict(manifest.get("artifacts", {}))
    calibration_summary = dict(manifest.get("calibration", {}))
    harness_summary = dict(manifest.get("policyengine_harness", {}))
    native_scores_summary = dict(manifest.get("policyengine_native_scores", {}))
    imputation_ablation_summary = dict(manifest.get("imputation_ablation", {}))
    outputs = [
        value
        for value in (
            artifacts.get("policyengine_harness"),
            artifacts.get("policyengine_native_scores"),
            artifacts.get("imputation_ablation"),
            artifacts.get("policyengine_native_audit"),
            *extra_outputs,
        )
        if value
    ]
    return {
        "id": "09_validation_benchmarking",
        "legacyId": "benchmark",
        "step": "09",
        "title": "Validation and benchmarking",
        "summary": (
            "Harness, native-loss, and donor-imputation diagnostics stay attached "
            "to the same artifact bundle."
        ),
        "status": (
            "ready"
            if harness_summary or native_scores_summary or imputation_ablation_summary
            else "missing"
        ),
        "metrics": [
            {
                "label": "Capped full oracle loss",
                "value": calibration_summary.get(
                    "full_oracle_capped_mean_abs_relative_error"
                ),
            },
            {
                "label": "Full oracle loss",
                "value": calibration_summary.get("full_oracle_mean_abs_relative_error"),
            },
            {
                "label": "Harness delta",
                "value": harness_summary.get("mean_abs_relative_error_delta"),
            },
            {
                "label": "Native delta",
                "value": native_scores_summary.get("enhanced_cps_native_loss_delta"),
            },
            {
                "label": "Win rate",
                "value": harness_summary.get("target_win_rate"),
            },
            {
                "label": "Imputation MAE",
                "value": imputation_ablation_summary.get(
                    "production_mean_weighted_mae"
                ),
            },
            {
                "label": "Imputation F1",
                "value": imputation_ablation_summary.get("production_mean_support_f1"),
            },
        ],
        "outputs": list(dict.fromkeys(outputs)),
    }


def _refresh_checkpoint_data_flow_snapshot(
    artifact_root: Path,
    manifest: dict[str, Any],
    *,
    extra_outputs: tuple[str, ...] = (),
) -> Path | None:
    snapshot_path = artifact_root / "data_flow_snapshot.json"
    stage_manifest_path = artifact_root / "stage_manifest.json"
    artifact_inventory_path = (
        artifact_root / "stage_artifacts" / "artifact_inventory.json"
    )
    conditional_readiness_path = (
        artifact_root / "stage_artifacts" / "conditional_readiness.json"
    )
    validation_evidence_path = (
        artifact_root
        / "stage_artifacts"
        / "09_validation_benchmarking"
        / "evidence_manifest.json"
    )
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts.setdefault("stage_manifest", stage_manifest_path.name)
    artifacts.setdefault(
        "artifact_inventory",
        str(artifact_inventory_path.relative_to(artifact_root)),
    )
    artifacts.setdefault(
        "conditional_readiness",
        str(conditional_readiness_path.relative_to(artifact_root)),
    )
    artifacts.setdefault(
        "validation_evidence",
        str(validation_evidence_path.relative_to(artifact_root)),
    )
    manifest["artifacts"] = artifacts
    write_us_validation_evidence_manifest(
        artifact_root,
        validation_evidence_path,
        manifest_payload=manifest,
    )
    write_us_stage_manifest(
        artifact_root,
        stage_manifest_path,
        manifest_payload=manifest,
        assume_existing_artifact_keys=(
            "artifact_inventory",
            "conditional_readiness",
        ),
    )
    readiness_inventory = build_us_stage_artifact_inventory(
        artifact_root,
        manifest_payload=manifest,
        assume_existing_artifact_keys=(
            "artifact_inventory",
            "conditional_readiness",
        ),
    )
    write_us_conditional_readiness_report(
        artifact_root,
        conditional_readiness_path,
        manifest_payload=manifest,
        artifact_inventory=readiness_inventory,
    )
    write_us_stage_artifact_inventory(
        artifact_root,
        artifact_inventory_path,
        manifest_payload=manifest,
        assume_existing_artifact_keys=("artifact_inventory",),
    )
    if not snapshot_path.exists():
        return None
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("schemaVersion") != 1:
        return snapshot_path
    stages = list(snapshot.get("stages", []))
    benchmark_stage = _build_checkpoint_benchmark_stage(
        manifest,
        extra_outputs=extra_outputs,
    )
    replaced = False
    for index, stage in enumerate(stages):
        if isinstance(stage, dict) and stage.get("id") in {
            "benchmark",
            "09_validation_benchmarking",
        }:
            stages[index] = benchmark_stage
            replaced = True
            break
    if not replaced:
        stages.append(benchmark_stage)
    snapshot["stages"] = stages
    _write_json_atomically(snapshot_path, snapshot)
    return snapshot_path


def _attach_checkpoint_registry_and_index(
    artifact_root: Path,
    manifest: dict[str, Any],
    *,
    harness_path: Path | None,
    harness_payload: dict[str, Any] | None,
    run_registry_path: str | Path | None,
    run_index_path: str | Path | None,
    run_registry_metadata: dict[str, Any] | None,
) -> tuple[Path | None, Path | None]:
    if (
        manifest.get("calibration", {}).get(
            "full_oracle_capped_mean_abs_relative_error"
        )
        is None
        and manifest.get("calibration", {}).get("full_oracle_mean_abs_relative_error")
        is None
        and "policyengine_harness" not in manifest
        and "policyengine_native_scores" not in manifest
    ):
        return None, None
    if (
        "policyengine_harness" not in manifest
        and "policyengine_native_scores" not in manifest
    ):
        resolved_harness_payload = None
    else:
        resolved_harness_payload = (
            dict(harness_payload)
            if harness_payload is not None
            else (
                json.loads(harness_path.read_text())
                if harness_path is not None and harness_path.exists()
                else None
            )
        )
    resolved_run_registry_path = Path(
        run_registry_path or artifact_root.parent / "run_registry.jsonl"
    )
    existing_entry = next(
        (
            entry
            for entry in reversed(
                load_us_microplex_run_registry(resolved_run_registry_path)
            )
            if entry.artifact_id == artifact_root.name
        ),
        None,
    )
    if existing_entry is None:
        run_entry = build_us_microplex_run_registry_entry(
            artifact_dir=artifact_root,
            manifest_path=artifact_root / "manifest.json",
            manifest=manifest,
            policyengine_harness_path=harness_path,
            policyengine_harness_payload=resolved_harness_payload,
            metadata=dict(run_registry_metadata or {}),
        )
        recorded_entry = append_us_microplex_run_registry_entry(
            resolved_run_registry_path,
            run_entry,
        )
    else:
        recorded_entry = existing_entry
    resolved_run_index_path = append_us_microplex_run_index_entry(
        run_index_path or artifact_root.parent,
        recorded_entry,
        policyengine_harness_payload=resolved_harness_payload,
    )
    manifest["run_registry"] = {
        "path": str(resolved_run_registry_path),
        "artifact_id": recorded_entry.artifact_id,
        "improved_candidate_frontier": recorded_entry.improved_candidate_frontier,
        "improved_delta_frontier": recorded_entry.improved_delta_frontier,
        "improved_composite_frontier": recorded_entry.improved_composite_frontier,
        "improved_native_frontier": recorded_entry.improved_native_frontier,
        "default_frontier_metric": _checkpoint_default_frontier_metric(manifest),
    }
    manifest["run_index"] = {
        "path": str(resolved_run_index_path),
        "artifact_id": recorded_entry.artifact_id,
    }
    return resolved_run_registry_path, resolved_run_index_path


def _checkpoint_default_frontier_metric(manifest: dict[str, Any]) -> FrontierMetric:
    if (
        dict(manifest.get("calibration", {})).get(
            "full_oracle_capped_mean_abs_relative_error"
        )
        is not None
    ):
        return "full_oracle_capped_mean_abs_relative_error"
    if (
        dict(manifest.get("calibration", {})).get("full_oracle_mean_abs_relative_error")
        is not None
    ):
        return "full_oracle_mean_abs_relative_error"
    if "policyengine_native_scores" in manifest:
        return "enhanced_cps_native_loss_delta"
    return "candidate_composite_parity_loss"


def _load_checkpoint_versioned_artifacts(
    *,
    build_result: Any,
    artifact_root: Path,
    frontier_metric: FrontierMetric,
) -> USMicroplexVersionedBuildArtifacts:
    manifest_path = artifact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = dict(manifest.get("artifacts", {}))
    artifact_paths = USMicroplexArtifactPaths(
        output_dir=artifact_root,
        version_id=artifact_root.name,
        seed_data=artifact_root / str(artifacts["seed_data"]),
        synthetic_data=artifact_root / str(artifacts["synthetic_data"]),
        calibrated_data=artifact_root / str(artifacts["calibrated_data"]),
        targets=artifact_root / str(artifacts["targets"]),
        manifest=manifest_path,
        synthesizer=_resolve_saved_artifact_path(
            artifact_root,
            artifacts.get("synthesizer"),
        ),
        policyengine_dataset=_resolve_saved_artifact_path(
            artifact_root,
            artifacts.get("policyengine_dataset"),
        ),
        data_flow_snapshot=(
            artifact_root / "data_flow_snapshot.json"
            if (artifact_root / "data_flow_snapshot.json").exists()
            else None
        ),
        policyengine_harness=_resolve_saved_artifact_path(
            artifact_root,
            artifacts.get("policyengine_harness"),
        ),
        policyengine_native_scores=_resolve_saved_artifact_path(
            artifact_root,
            artifacts.get("policyengine_native_scores"),
        ),
        policyengine_native_audit=_resolve_saved_artifact_path(
            artifact_root,
            artifacts.get("policyengine_native_audit"),
        ),
        run_registry=_resolve_saved_artifact_path(
            artifact_root,
            dict(manifest.get("run_registry", {})).get("path"),
        ),
        run_index_db=_resolve_saved_artifact_path(
            artifact_root,
            dict(manifest.get("run_index", {})).get("path"),
        ),
    )
    current_entry = None
    frontier_entry = None
    frontier_delta = None
    if artifact_paths.run_registry is not None:
        registry_entries = load_us_microplex_run_registry(artifact_paths.run_registry)
        current_entry = next(
            (
                entry
                for entry in reversed(registry_entries)
                if entry.artifact_id == artifact_root.name
            ),
            None,
        )
        frontier_entry = select_us_microplex_frontier_entry(
            artifact_paths.run_registry,
            metric=frontier_metric,
        )
        current_value = _registry_metric_value(current_entry, frontier_metric)
        frontier_value = _registry_metric_value(frontier_entry, frontier_metric)
        if current_value is not None and frontier_value is not None:
            frontier_delta = current_value - frontier_value
    return USMicroplexVersionedBuildArtifacts(
        build_result=build_result,
        artifact_paths=artifact_paths,
        current_entry=current_entry,
        frontier_entry=frontier_entry,
        frontier_delta=frontier_delta,
    )


def _build_checkpoint_harness_context(
    *,
    manifest: dict[str, Any],
    policyengine_target_provider: TargetProvider | None,
    policyengine_baseline_dataset: str | Path | None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ),
    policyengine_harness_metadata: dict[str, Any] | None,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None,
) -> tuple[
    TargetProvider | None,
    str | Path | None,
    tuple[PolicyEngineUSHarnessSlice, ...],
    dict[str, Any],
]:
    from microplex_us.policyengine.harness import (
        default_policyengine_us_db_all_target_slices,
        default_policyengine_us_harness_slices,
        filter_nonempty_policyengine_us_harness_slices,
    )
    from microplex_us.policyengine.us import PolicyEngineUSDBTargetProvider

    config = dict(manifest.get("config", {}))
    resolved_target_provider = policyengine_target_provider
    if (
        resolved_target_provider is None
        and config.get("policyengine_targets_db") is not None
    ):
        resolved_target_provider = PolicyEngineUSDBTargetProvider(
            config["policyengine_targets_db"]
        )
    resolved_baseline_dataset = policyengine_baseline_dataset or config.get(
        "policyengine_baseline_dataset"
    )
    harness_period = (
        config.get("policyengine_dataset_year")
        or config.get("policyengine_target_period")
        or 2024
    )
    if policyengine_harness_slices is not None:
        resolved_harness_slices = tuple(policyengine_harness_slices)
    elif config.get("policyengine_targets_db") is not None:
        resolved_harness_slices = default_policyengine_us_db_all_target_slices(
            period=int(harness_period),
            reform_id=int(config.get("policyengine_target_reform_id", 0) or 0),
        )
    else:
        resolved_harness_slices = default_policyengine_us_harness_slices(
            period=int(harness_period)
        )
    if resolved_target_provider is not None and resolved_harness_slices:
        resolved_harness_slices = filter_nonempty_policyengine_us_harness_slices(
            resolved_target_provider,
            resolved_harness_slices,
            cache=policyengine_comparison_cache,
        )
    resolved_harness_metadata = {
        "baseline_dataset": (
            Path(resolved_baseline_dataset).name
            if resolved_baseline_dataset is not None
            else None
        ),
        "targets_db": (
            Path(config["policyengine_targets_db"]).name
            if config.get("policyengine_targets_db") is not None
            else None
        ),
        "target_period": config.get("policyengine_target_period"),
        "target_variables": list(config.get("policyengine_target_variables", ())),
        "target_domains": list(config.get("policyengine_target_domains", ())),
        "target_geo_levels": list(config.get("policyengine_target_geo_levels", ())),
        "target_profile": config.get("policyengine_target_profile"),
        "calibration_target_profile": config.get(
            "policyengine_calibration_target_profile"
        ),
        "target_reform_id": config.get("policyengine_target_reform_id"),
        "harness_slice_names": [
            slice_spec.name for slice_spec in resolved_harness_slices
        ],
        "policyengine_us_runtime_version": _resolve_policyengine_us_runtime_version(),
        "harness_suite": (
            "policyengine_us_all_targets"
            if config.get("policyengine_targets_db") is not None
            and policyengine_harness_slices is None
            else None
        ),
        **dict(policyengine_harness_metadata or {}),
    }
    return (
        resolved_target_provider,
        resolved_baseline_dataset,
        resolved_harness_slices,
        resolved_harness_metadata,
    )


def attach_policyengine_us_data_rebuild_checkpoint_evidence(
    artifact_dir: str | Path,
    *,
    build_result: Any | None = None,
    program: PEUSDataRebuildProgram | None = None,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    compute_harness: bool = True,
    compute_native_scores: bool = True,
    compute_native_audit: bool = True,
    compute_imputation_ablation: bool = False,
    require_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    precomputed_imputation_ablation_payload: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> PEUSDataRebuildCheckpointEvidenceResult:
    """Attach PE comparison evidence to an already-saved rebuild artifact."""

    from microplex_us.pipelines.pe_native_scores import compute_us_pe_native_scores
    from microplex_us.policyengine.harness import evaluate_policyengine_us_harness
    from microplex_us.policyengine.us import load_policyengine_us_entity_tables

    artifact_root = Path(artifact_dir)
    manifest_path = artifact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = dict(manifest.get("config", {}))
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_name = artifacts.get("policyengine_dataset")
    dataset_path = (
        artifact_root / dataset_name if isinstance(dataset_name, str) else None
    )
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(
            "Saved rebuild artifact is missing policyengine_dataset output"
        )

    harness_path: Path | None = None
    harness_payload = (
        dict(precomputed_policyengine_harness_payload)
        if precomputed_policyengine_harness_payload is not None
        else None
    )
    if harness_payload is None and compute_harness:
        (
            resolved_target_provider,
            resolved_baseline_dataset,
            resolved_harness_slices,
            resolved_harness_metadata,
        ) = _build_checkpoint_harness_context(
            manifest=manifest,
            policyengine_target_provider=policyengine_target_provider,
            policyengine_baseline_dataset=policyengine_baseline_dataset,
            policyengine_harness_slices=policyengine_harness_slices,
            policyengine_harness_metadata=policyengine_harness_metadata,
            policyengine_comparison_cache=policyengine_comparison_cache,
        )
        if resolved_target_provider is None:
            raise ValueError(
                "Cannot compute rebuild checkpoint harness without a target provider"
            )
        if resolved_baseline_dataset is None:
            raise ValueError(
                "Cannot compute rebuild checkpoint harness without a baseline dataset"
            )
        if not resolved_harness_slices:
            raise ValueError(
                "Cannot compute rebuild checkpoint harness because no nonempty slices resolved"
            )
        candidate_tables = load_policyengine_us_entity_tables(
            dataset_path,
            period=(
                config.get("policyengine_dataset_year")
                or config.get("policyengine_target_period")
                or 2024
            ),
        )
        harness_run = evaluate_policyengine_us_harness(
            candidate_tables,
            resolved_target_provider,
            resolved_harness_slices,
            baseline_dataset=str(resolved_baseline_dataset),
            dataset_year=config.get("policyengine_dataset_year"),
            simulation_cls=None,
            candidate_label="microplex",
            baseline_label="policyengine_us_data",
            metadata=resolved_harness_metadata,
            cache=policyengine_comparison_cache,
        )
        harness_payload = harness_run.to_dict()
    if harness_payload is not None:
        harness_path = artifact_root / "policyengine_harness.json"
        _write_json_atomically(harness_path, harness_payload)
        artifacts["policyengine_harness"] = harness_path.name
        manifest["policyengine_harness"] = dict(harness_payload.get("summary", {}))

    native_scores_path: Path | None = None
    native_scores_payload = (
        dict(precomputed_policyengine_native_scores)
        if precomputed_policyengine_native_scores is not None
        else None
    )
    if native_scores_payload is None and compute_native_scores:
        resolved_baseline_dataset = policyengine_baseline_dataset or config.get(
            "policyengine_baseline_dataset"
        )
        if resolved_baseline_dataset is None:
            raise ValueError(
                "Cannot compute PE-native scores without a baseline dataset"
            )
        native_scores_payload = compute_us_pe_native_scores(
            candidate_dataset_path=dataset_path,
            baseline_dataset_path=resolved_baseline_dataset,
            period=(
                config.get("policyengine_dataset_year")
                or config.get("policyengine_target_period")
                or 2024
            ),
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
    if native_scores_payload is not None:
        native_scores_path = artifact_root / "policyengine_native_scores.json"
        _write_json_atomically(native_scores_path, native_scores_payload)
        artifacts["policyengine_native_scores"] = native_scores_path.name
        manifest["policyengine_native_scores"] = dict(
            native_scores_payload.get("summary", {})
        )
    elif require_policyengine_native_score:
        raise ValueError(
            "require_policyengine_native_score=True but no PE-native scores were computed"
        )

    imputation_ablation_path: Path | None = None
    imputation_ablation_payload = (
        dict(precomputed_imputation_ablation_payload)
        if precomputed_imputation_ablation_payload is not None
        else None
    )
    if (
        imputation_ablation_payload is None
        and compute_imputation_ablation
        and build_result is not None
    ):
        imputation_ablation_payload = _build_checkpoint_imputation_ablation_payload(
            build_result,
            artifact_id=artifact_root.name,
            manifest=manifest,
        )
    if imputation_ablation_payload is not None:
        imputation_ablation_path = artifact_root / "imputation_ablation.json"
        _write_json_atomically(imputation_ablation_path, imputation_ablation_payload)
        artifacts["imputation_ablation"] = imputation_ablation_path.name
        manifest["imputation_ablation"] = dict(
            imputation_ablation_payload.get("summary", {})
        )

    manifest["artifacts"] = artifacts
    _attach_checkpoint_registry_and_index(
        artifact_root,
        manifest,
        harness_path=harness_path,
        harness_payload=harness_payload,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
    )
    assert_valid_benchmark_artifact_manifest(
        manifest,
        artifact_dir=artifact_root,
        manifest_path=manifest_path,
        summary_section=(
            "policyengine_harness" if "policyengine_harness" in manifest else None
        ),
        required_artifact_keys=(
            "seed_data",
            "synthetic_data",
            "calibrated_data",
            "targets",
            *(
                ("policyengine_harness",)
                if artifacts.get("policyengine_harness") is not None
                else ()
            ),
            *(
                ("policyengine_native_scores",)
                if artifacts.get("policyengine_native_scores") is not None
                else ()
            ),
        ),
        required_summary_keys=(
            (
                "candidate_mean_abs_relative_error",
                "baseline_mean_abs_relative_error",
                "mean_abs_relative_error_delta",
            )
            if "policyengine_harness" in manifest
            else ()
        ),
    )
    resolved_program = program or default_policyengine_us_data_rebuild_program()
    parity_path = write_policyengine_us_data_rebuild_parity_artifact(
        artifact_root,
        program=resolved_program,
    )
    parity_payload = build_policyengine_us_data_rebuild_parity_artifact(
        artifact_root,
        program=resolved_program,
    )
    native_audit_path: Path | None = None
    native_audit_payload: dict[str, Any] | None = None
    if compute_native_audit and artifacts.get("policyengine_native_scores") is not None:
        native_audit_payload = build_policyengine_us_data_rebuild_native_audit(
            artifact_root,
            manifest_payload=manifest,
            native_scores_payload=native_scores_payload,
            imputation_ablation_payload=imputation_ablation_payload,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
        native_audit_path = artifact_root / "pe_us_data_rebuild_native_audit.json"
        _write_json_atomically(native_audit_path, native_audit_payload)
        artifacts["policyengine_native_audit"] = native_audit_path.name
        manifest["policyengine_native_audit"] = dict(
            native_audit_payload.get("verdictHints", {})
        )
    stage_manifest_path = artifact_root / "stage_manifest.json"
    validation_evidence_path = (
        artifact_root
        / "stage_artifacts"
        / "09_validation_benchmarking"
        / "evidence_manifest.json"
    )
    artifacts.setdefault("stage_manifest", stage_manifest_path.name)
    artifacts.setdefault(
        "validation_evidence",
        str(validation_evidence_path.relative_to(artifact_root)),
    )
    manifest["artifacts"] = artifacts
    write_us_validation_evidence_manifest(
        artifact_root,
        validation_evidence_path,
        manifest_payload=manifest,
    )
    write_us_stage_manifest(
        artifact_root,
        stage_manifest_path,
        manifest_payload=manifest,
    )
    _refresh_checkpoint_data_flow_snapshot(
        artifact_root,
        manifest,
        extra_outputs=(native_audit_path.name,)
        if native_audit_path is not None
        else (),
    )
    _write_json_atomically(manifest_path, manifest)
    return PEUSDataRebuildCheckpointEvidenceResult(
        artifact_dir=artifact_root,
        manifest_path=manifest_path,
        harness_path=harness_path,
        native_scores_path=native_scores_path,
        parity_path=parity_path,
        parity_payload=parity_payload,
        native_audit_path=native_audit_path,
        native_audit_payload=native_audit_payload,
        imputation_ablation_path=imputation_ablation_path,
        imputation_ablation_payload=imputation_ablation_payload,
    )


def default_policyengine_us_data_rebuild_checkpoint_config(
    *,
    policyengine_baseline_dataset: str | Path,
    policyengine_targets_db: str | Path,
    target_period: int = 2024,
    target_profile: str = "pe_native_broad",
    calibration_target_profile: str | None = None,
    target_variables: tuple[str, ...] = (),
    target_domains: tuple[str, ...] = (),
    target_geo_levels: tuple[str, ...] = (),
    calibration_target_variables: tuple[str, ...] = (),
    calibration_target_domains: tuple[str, ...] = (),
    calibration_target_geo_levels: tuple[str, ...] = (),
    **overrides: Any,
) -> USMicroplexBuildConfig:
    """Return the canonical rebuild config with required PE comparison context."""

    resolved_target_period = int(target_period)
    resolved_baseline_weight_sum = _infer_policyengine_baseline_household_weight_sum(
        policyengine_baseline_dataset,
        target_period=resolved_target_period,
    )
    resolved_overrides = dict(overrides)
    infer_total_weight_targets = (
        resolved_baseline_weight_sum is not None
        and resolved_overrides.get("calibration_backend") != "none"
    )
    if infer_total_weight_targets:
        resolved_overrides.setdefault(
            "policyengine_selection_target_total_weight",
            resolved_baseline_weight_sum,
        )
        if not resolved_overrides.get(
            "policyengine_calibration_rescale_to_input_weight_sum",
            False,
        ):
            resolved_overrides.setdefault(
                "policyengine_calibration_target_total_weight",
                resolved_baseline_weight_sum,
            )
            resolved_overrides.setdefault(
                "policyengine_calibration_rescale_to_target_total_weight",
                True,
            )
    return default_policyengine_us_data_rebuild_config(
        policyengine_baseline_dataset=str(policyengine_baseline_dataset),
        policyengine_targets_db=str(policyengine_targets_db),
        policyengine_dataset_year=resolved_target_period,
        policyengine_target_period=resolved_target_period,
        policyengine_target_profile=target_profile,
        policyengine_calibration_target_profile=(
            calibration_target_profile or target_profile
        ),
        policyengine_target_variables=tuple(target_variables),
        policyengine_target_domains=tuple(target_domains),
        policyengine_target_geo_levels=tuple(target_geo_levels),
        policyengine_calibration_target_variables=(
            _resolve_checkpoint_calibration_target_variables(
                calibration_target_variables
            )
        ),
        policyengine_calibration_target_domains=tuple(calibration_target_domains),
        policyengine_calibration_target_geo_levels=tuple(calibration_target_geo_levels),
        **resolved_overrides,
    )


def default_policyengine_us_data_rebuild_queries(
    providers: tuple[SourceProvider, ...] | list[SourceProvider],
    *,
    cps_sample_n: int | None = None,
    puf_sample_n: int | None = None,
    donor_sample_n: int | None = None,
    cps_state_age_floor: int | None = 1,
    donor_state_age_floor: int | None = 1,
    random_seed: int = 0,
) -> dict[str, SourceQuery]:
    """Return default provider queries for a rebuild checkpoint smoke run."""

    from microplex_us.data_sources.cps import CPSASECSourceProvider
    from microplex_us.data_sources.donor_surveys import DonorSurveySourceProvider
    from microplex_us.data_sources.puf import PUFSourceProvider

    resolved_donor_sample_n = donor_sample_n
    if resolved_donor_sample_n is None:
        source_sample_sizes = tuple(
            int(sample_n)
            for sample_n in (cps_sample_n, puf_sample_n)
            if sample_n is not None
        )
        if source_sample_sizes:
            resolved_donor_sample_n = max(source_sample_sizes)

    queries: dict[str, SourceQuery] = {}
    for provider in providers:
        sample_n: int | None = None
        if isinstance(provider, CPSASECSourceProvider):
            sample_n = cps_sample_n
        elif isinstance(provider, PUFSourceProvider):
            sample_n = puf_sample_n
        elif isinstance(provider, DonorSurveySourceProvider):
            sample_n = resolved_donor_sample_n
        if sample_n is None:
            continue
        provider_filters = {
            "sample_n": int(sample_n),
            "random_seed": int(random_seed),
        }
        if (
            isinstance(provider, CPSASECSourceProvider)
            and cps_state_age_floor is not None
        ):
            provider_filters["state_age_floor"] = int(cps_state_age_floor)
        elif (
            isinstance(provider, DonorSurveySourceProvider)
            and donor_state_age_floor is not None
        ):
            provider_filters["state_age_floor"] = int(donor_state_age_floor)
        queries[provider.descriptor.name] = SourceQuery(
            provider_filters=provider_filters
        )
    return queries


def run_policyengine_us_data_rebuild_checkpoint(
    output_root: str | Path,
    *,
    policyengine_baseline_dataset: str | Path,
    policyengine_targets_db: str | Path,
    target_period: int = 2024,
    target_profile: str = "pe_native_broad",
    calibration_target_profile: str | None = None,
    target_variables: tuple[str, ...] = (),
    target_domains: tuple[str, ...] = (),
    target_geo_levels: tuple[str, ...] = (),
    calibration_target_variables: tuple[str, ...] = (),
    calibration_target_domains: tuple[str, ...] = (),
    calibration_target_geo_levels: tuple[str, ...] = (),
    config: USMicroplexBuildConfig | None = None,
    config_overrides: dict[str, Any] | None = None,
    providers: tuple[SourceProvider, ...] | list[SourceProvider] | None = None,
    queries: dict[str, SourceQuery] | None = None,
    cps_source_year: int = 2023,
    cps_cache_dir: str | Path | None = None,
    cps_download: bool = True,
    puf_target_year: int | None = None,
    puf_cps_reference_year: int | None = None,
    puf_cache_dir: str | Path | None = None,
    puf_path: str | Path | None = None,
    puf_demographics_path: str | Path | None = None,
    puf_expand_persons: bool = True,
    include_donor_surveys: bool = True,
    include_acs: bool | None = None,
    include_sipp: bool | None = None,
    include_scf: bool | None = None,
    acs_year: int = 2022,
    sipp_year: int = 2023,
    scf_year: int = 2022,
    donor_cache_dir: str | Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    cps_sample_n: int | None = None,
    puf_sample_n: int | None = None,
    donor_sample_n: int | None = None,
    query_random_seed: int = 0,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "full_oracle_capped_mean_abs_relative_error",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    defer_native_audit: bool = False,
    defer_imputation_ablation: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    precomputed_imputation_ablation_payload: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> PEUSDataRebuildCheckpointResult:
    """Run one saved rebuild checkpoint and write its PE comparison sidecars."""

    if config is not None and config_overrides:
        raise ValueError(
            "config_overrides cannot be used when an explicit config is supplied"
        )
    resolved_config = config or default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_targets_db=policyengine_targets_db,
        target_period=target_period,
        target_profile=target_profile,
        calibration_target_profile=calibration_target_profile,
        target_variables=target_variables,
        target_domains=target_domains,
        target_geo_levels=target_geo_levels,
        calibration_target_variables=calibration_target_variables,
        calibration_target_domains=calibration_target_domains,
        calibration_target_geo_levels=calibration_target_geo_levels,
        **dict(config_overrides or {}),
    )
    if config is not None:
        _validate_checkpoint_config_context(
            resolved_config,
            policyengine_baseline_dataset=policyengine_baseline_dataset,
            policyengine_targets_db=policyengine_targets_db,
            target_period=target_period,
            target_profile=target_profile,
            calibration_target_profile=calibration_target_profile,
            target_variables=target_variables,
            target_domains=target_domains,
            target_geo_levels=target_geo_levels,
            calibration_target_variables=calibration_target_variables,
            calibration_target_domains=calibration_target_domains,
            calibration_target_geo_levels=calibration_target_geo_levels,
        )
    if providers is None:
        resolved_providers = tuple(
            default_policyengine_us_data_rebuild_source_providers(
                cps_source_year=cps_source_year,
                cps_cache_dir=cps_cache_dir,
                cps_download=cps_download,
                puf_target_year=(
                    int(puf_target_year)
                    if puf_target_year is not None
                    else int(target_period)
                ),
                puf_cps_reference_year=puf_cps_reference_year,
                puf_cache_dir=puf_cache_dir,
                puf_path=puf_path,
                puf_demographics_path=puf_demographics_path,
                puf_expand_persons=puf_expand_persons,
                include_donor_surveys=include_donor_surveys,
                include_acs=include_acs,
                include_sipp=include_sipp,
                include_scf=include_scf,
                acs_year=acs_year,
                sipp_year=sipp_year,
                scf_year=scf_year,
                donor_cache_dir=donor_cache_dir,
                policyengine_us_data_repo=policyengine_us_data_repo,
                policyengine_us_data_python=policyengine_us_data_python,
            )
        )
    else:
        resolved_providers = tuple(providers)
        if not resolved_providers:
            raise ValueError(
                "providers must be None or a non-empty provider sequence for a rebuild checkpoint"
            )
    resolved_queries = (
        dict(queries)
        if queries is not None
        else default_policyengine_us_data_rebuild_queries(
            resolved_providers,
            cps_sample_n=cps_sample_n,
            puf_sample_n=puf_sample_n,
            donor_sample_n=donor_sample_n,
            random_seed=query_random_seed,
        )
    )
    program = default_policyengine_us_data_rebuild_program()
    provider_names = tuple(provider.descriptor.name for provider in resolved_providers)
    _validate_query_keys(provider_names, resolved_queries)
    if (
        policyengine_us_data_python is not None
        and not defer_policyengine_native_score
        and precomputed_policyengine_native_scores is None
    ):
        raise ValueError(
            "policyengine_us_data_python requires defer_policyengine_native_score=True "
            "or precomputed_policyengine_native_scores because the automatic native-score "
            "save path cannot yet honor a custom PE-US-data interpreter"
        )
    resolved_harness_metadata = {
        "rebuild_checkpoint": True,
        "rebuild_program_id": program.program_id,
        "rebuild_provider_names": list(provider_names),
        **dict(policyengine_harness_metadata or {}),
    }
    resolved_registry_metadata = {
        "rebuild_checkpoint": True,
        "rebuild_program_id": program.program_id,
        "rebuild_provider_names": list(provider_names),
        "rebuild_profile_expected": True,
        **dict(run_registry_metadata or {}),
    }
    _emit_checkpoint_progress(
        "PE-US-data rebuild checkpoint: starting build",
        output_root=Path(output_root).expanduser(),
        version_id=version_id or "auto",
        target_profile=resolved_config.policyengine_target_profile,
        donor_condition_selection=resolved_config.donor_imputer_condition_selection,
        providers=",".join(provider_names),
    )

    artifacts = build_and_save_versioned_us_microplex_from_source_providers(
        providers=list(resolved_providers),
        output_root=output_root,
        config=resolved_config,
        queries=resolved_queries or None,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=resolved_config.policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=resolved_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=True,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=True,
        precomputed_policyengine_harness_payload=None,
        precomputed_policyengine_native_scores=None,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=resolved_registry_metadata,
        enable_child_tax_unit_agi_drift=True,
    )
    _emit_checkpoint_progress(
        "PE-US-data rebuild checkpoint: build complete",
        artifact_dir=artifacts.artifact_paths.output_dir,
        frontier_metric=frontier_metric,
    )
    _emit_checkpoint_progress(
        "PE-US-data rebuild checkpoint: attaching PE evidence",
        artifact_dir=artifacts.artifact_paths.output_dir,
        compute_harness=not defer_policyengine_harness,
        compute_native_scores=not defer_policyengine_native_score,
        compute_native_audit=not defer_native_audit,
        compute_imputation_ablation=not defer_imputation_ablation,
    )
    evidence = attach_policyengine_us_data_rebuild_checkpoint_evidence(
        artifacts.artifact_paths.output_dir,
        build_result=artifacts.build_result,
        program=program,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=resolved_config.policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=resolved_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        compute_harness=not defer_policyengine_harness,
        compute_native_scores=not defer_policyengine_native_score,
        compute_native_audit=not defer_native_audit,
        compute_imputation_ablation=not defer_imputation_ablation,
        require_policyengine_native_score=require_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        precomputed_imputation_ablation_payload=precomputed_imputation_ablation_payload,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=resolved_registry_metadata,
    )
    _emit_checkpoint_progress(
        "PE-US-data rebuild checkpoint: evidence complete",
        parity_path=evidence.parity_path,
        native_audit_path=evidence.native_audit_path,
        imputation_ablation_path=evidence.imputation_ablation_path,
    )
    refreshed_artifacts = _load_checkpoint_versioned_artifacts(
        build_result=artifacts.build_result,
        artifact_root=artifacts.artifact_paths.output_dir,
        frontier_metric=frontier_metric,
    )
    _emit_checkpoint_progress(
        "PE-US-data rebuild checkpoint: checkpoint ready",
        artifact_dir=refreshed_artifacts.artifact_paths.output_dir,
    )
    return PEUSDataRebuildCheckpointResult(
        build_config=resolved_config,
        provider_names=provider_names,
        queries=resolved_queries,
        artifacts=refreshed_artifacts,
        parity_path=evidence.parity_path,
        parity_payload=evidence.parity_payload,
        native_audit_path=evidence.native_audit_path,
        native_audit_payload=evidence.native_audit_payload,
        imputation_ablation_path=evidence.imputation_ablation_path,
        imputation_ablation_payload=evidence.imputation_ablation_payload,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for one PE-US-data rebuild checkpoint."""

    parser = argparse.ArgumentParser(
        description="Run a versioned PE-US-data rebuild checkpoint in microplex-us."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--targets-db", required=True)
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument("--version-id")
    parser.add_argument("--target-period", type=int, default=2024)
    parser.add_argument("--target-profile", default="pe_native_broad")
    parser.add_argument("--calibration-target-profile")
    parser.add_argument("--n-synthetic", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--donor-imputer-condition-selection")
    parser.add_argument(
        "--donor-imputer-backend",
        choices=["maf", "qrf", "zi_qrf", "regime_aware"],
        default=None,
        help=(
            "Donor imputer backend. `zi_qrf` activates the zero-inflated "
            "QRF path that skips predict() on gate-predicted-zero rows, "
            "which is a large wall-clock win on heavy-zero PUF tax "
            "variables. See docs/next-run-plan.md."
        ),
    )
    parser.add_argument("--cps-source-year", type=int, default=2023)
    parser.add_argument("--puf-target-year", type=int)
    parser.add_argument("--puf-cps-reference-year", type=int)
    parser.add_argument("--acs-year", type=int, default=2022)
    parser.add_argument("--sipp-year", type=int, default=2023)
    parser.add_argument("--scf-year", type=int, default=2022)
    parser.add_argument("--cps-cache-dir")
    parser.add_argument("--puf-cache-dir")
    parser.add_argument("--donor-cache-dir")
    parser.add_argument("--puf-path")
    parser.add_argument("--puf-demographics-path")
    parser.add_argument("--cps-sample-n", type=int)
    parser.add_argument("--puf-sample-n", type=int)
    parser.add_argument("--donor-sample-n", type=int)
    parser.add_argument("--query-random-seed", type=int, default=0)
    parser.add_argument("--target-variable", action="append", default=[])
    parser.add_argument("--target-domain", action="append", default=[])
    parser.add_argument("--target-geo-level", action="append", default=[])
    parser.add_argument("--calibration-target-variable", action="append", default=[])
    parser.add_argument("--calibration-target-domain", action="append", default=[])
    parser.add_argument("--calibration-target-geo-level", action="append", default=[])
    parser.add_argument(
        "--include-donor-surveys",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-acs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include the ACS donor provider. Defaults to --include-donor-surveys; "
            "use --no-include-acs for an eCPS-shaped run that keeps SIPP/SCF."
        ),
    )
    parser.add_argument(
        "--include-sipp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include SIPP donor providers. Defaults to --include-donor-surveys.",
    )
    parser.add_argument(
        "--include-scf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include the SCF donor provider. Defaults to --include-donor-surveys.",
    )
    parser.add_argument("--no-cps-download", action="store_true")
    parser.add_argument("--no-puf-expand-persons", action="store_true")
    parser.add_argument("--defer-policyengine-harness", action="store_true")
    parser.add_argument("--defer-policyengine-native-score", action="store_true")
    parser.add_argument("--defer-native-audit", action="store_true")
    parser.add_argument("--defer-imputation-ablation", action="store_true")
    parser.add_argument("--require-policyengine-native-score", action="store_true")
    parser.add_argument(
        "--calibration-backend",
        choices=[
            "entropy",
            "ipf",
            "chi2",
            "sparse",
            "hardconcrete",
            "pe_l0",
            "microcalibrate",
            "none",
        ],
        default=None,
        help=(
            "Weighting/calibration backend. Default is the config default "
            "(entropy). Use `microcalibrate` for the identity-preserving "
            "gradient-descent chi-squared backend that survived the v6 OOM."
        ),
    )
    parser.add_argument(
        "--calibration-max-iter",
        type=int,
        default=None,
        help=(
            "Max iterations / epochs for the calibration solver. Passed "
            "through to USMicroplexBuildConfig.calibration_max_iter."
        ),
    )
    parser.add_argument(
        "--policyengine-materialize-batch-size",
        type=int,
        default=None,
        help=(
            "If set, splits PolicyEngine variable materialization into "
            "household chunks of this size. At 1.5M-household scale a "
            "single Microsimulation is 25-35 GB; batch_size=100_000 "
            "drops peak to a few GB. Required for workstation runs; "
            "unset (full-dataset) path targeted Modal GPU."
        ),
    )
    parser.add_argument(
        "--pipeline-checkpoint-save-post-imputation-path",
        type=str,
        default=None,
        help=(
            "If set, save a post-imputation pipeline checkpoint to this "
            "directory (right after donor imputation + PE tables build, "
            "before microsim). A rerun can resume from this checkpoint "
            "to skip the ~11 h synthesis stage."
        ),
    )
    parser.add_argument(
        "--policyengine-export-column-contract-path",
        type=str,
        default=None,
        help=(
            "If set, check the eCPS export-column contract from the "
            "post-imputation PE entity tables before microsimulation and "
            "calibration."
        ),
    )
    parser.add_argument(
        "--pipeline-checkpoint-save-post-microsim-path",
        type=str,
        default=None,
        help=(
            "If set, save a post-microsim pipeline checkpoint to this "
            "directory (after target variables are materialized, before "
            "the calibration fit loop). A rerun can resume from this "
            "checkpoint to skip both synthesis and microsim, leaving "
            "only the calibration fit."
        ),
    )
    parser.add_argument(
        "--capital-gains-lots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write an anchor-preserving synthetic capital-gains lot SQLite "
            "sidecar artifact from PolicyEngine person tables."
        ),
    )
    parser.add_argument("--capital-gains-lots-max-lots-per-person", type=int)
    parser.add_argument("--capital-gains-lots-random-seed", type=int)
    args = parser.parse_args(argv)

    config_overrides = {
        "n_synthetic": int(args.n_synthetic),
        "random_seed": int(args.random_seed),
    }
    if args.donor_imputer_condition_selection is not None:
        config_overrides["donor_imputer_condition_selection"] = (
            args.donor_imputer_condition_selection
        )
    if args.donor_imputer_backend is not None:
        config_overrides["donor_imputer_backend"] = args.donor_imputer_backend
    if args.calibration_backend is not None:
        config_overrides["calibration_backend"] = args.calibration_backend
    if args.calibration_max_iter is not None:
        config_overrides["calibration_max_iter"] = int(args.calibration_max_iter)
    if args.policyengine_materialize_batch_size is not None:
        config_overrides["policyengine_materialize_batch_size"] = int(
            args.policyengine_materialize_batch_size
        )
    if args.pipeline_checkpoint_save_post_imputation_path is not None:
        config_overrides["pipeline_checkpoint_save_post_imputation_path"] = (
            args.pipeline_checkpoint_save_post_imputation_path
        )
    if args.policyengine_export_column_contract_path is not None:
        config_overrides["policyengine_export_column_contract_path"] = (
            args.policyengine_export_column_contract_path
        )
    if args.pipeline_checkpoint_save_post_microsim_path is not None:
        config_overrides["pipeline_checkpoint_save_post_microsim_path"] = (
            args.pipeline_checkpoint_save_post_microsim_path
        )
    if args.capital_gains_lots is not None:
        config_overrides["capital_gains_lots_enabled"] = bool(args.capital_gains_lots)
    if args.capital_gains_lots_max_lots_per_person is not None:
        config_overrides["capital_gains_lots_max_lots_per_person"] = int(
            args.capital_gains_lots_max_lots_per_person
        )
    if args.capital_gains_lots_random_seed is not None:
        config_overrides["capital_gains_lots_random_seed"] = int(
            args.capital_gains_lots_random_seed
        )

    result = run_policyengine_us_data_rebuild_checkpoint(
        output_root=args.output_root,
        policyengine_baseline_dataset=args.baseline_dataset,
        policyengine_targets_db=args.targets_db,
        target_period=args.target_period,
        target_profile=args.target_profile,
        calibration_target_profile=args.calibration_target_profile,
        target_variables=tuple(args.target_variable),
        target_domains=tuple(args.target_domain),
        target_geo_levels=tuple(args.target_geo_level),
        calibration_target_variables=tuple(args.calibration_target_variable),
        calibration_target_domains=tuple(args.calibration_target_domain),
        calibration_target_geo_levels=tuple(args.calibration_target_geo_level),
        config_overrides=config_overrides,
        cps_source_year=args.cps_source_year,
        cps_cache_dir=args.cps_cache_dir,
        cps_download=not args.no_cps_download,
        puf_target_year=args.puf_target_year,
        puf_cps_reference_year=args.puf_cps_reference_year,
        puf_cache_dir=args.puf_cache_dir,
        puf_path=args.puf_path,
        puf_demographics_path=args.puf_demographics_path,
        puf_expand_persons=not args.no_puf_expand_persons,
        include_donor_surveys=args.include_donor_surveys,
        include_acs=args.include_acs,
        include_sipp=args.include_sipp,
        include_scf=args.include_scf,
        acs_year=args.acs_year,
        sipp_year=args.sipp_year,
        scf_year=args.scf_year,
        donor_cache_dir=args.donor_cache_dir,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
        cps_sample_n=args.cps_sample_n,
        puf_sample_n=args.puf_sample_n,
        donor_sample_n=args.donor_sample_n,
        query_random_seed=args.query_random_seed,
        version_id=args.version_id,
        defer_policyengine_harness=args.defer_policyengine_harness,
        require_policyengine_native_score=args.require_policyengine_native_score,
        defer_policyengine_native_score=args.defer_policyengine_native_score,
        defer_native_audit=args.defer_native_audit,
        defer_imputation_ablation=args.defer_imputation_ablation,
    )

    print(result.artifacts.artifact_paths.output_dir)
    print(result.parity_path)
    print(json.dumps(result.parity_payload["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
