"""Checkpoint-specific imputation ablation evidence helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from microplex.core import (
    EntityObservation,
    EntityType,
    ObservationFrame,
    SourceDescriptor,
)

from microplex_us.pipelines.imputation_ablation import (
    ImputationAblationSliceSpec,
    ImputationAblationVariant,
    score_imputation_ablation_variants,
)
from microplex_us.variables import prune_redundant_variables

if TYPE_CHECKING:
    pass

DEFAULT_CHECKPOINT_IMPUTATION_ABLATION_EVAL_FRACTION = 0.25
MIN_CHECKPOINT_IMPUTATION_ABLATION_HOUSEHOLDS = 8


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
