"""Validation and benchmark artifact helpers for saved US Microplex bundles."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from microplex.targets import TargetProvider

from microplex_us.pipelines.us import USMicroplexBuildResult
from microplex_us.policyengine.harness import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSHarnessSlice,
    default_policyengine_us_db_all_target_slices,
    default_policyengine_us_harness_slices,
    filter_nonempty_policyengine_us_harness_slices,
)
from microplex_us.policyengine.us import PolicyEngineUSDBTargetProvider


def _stage9_benchmark_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "policyengine_harness",
        "policyengine_native_scores",
        "policyengine_native_audit",
        "imputation_ablation",
    ):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            summary[key] = dict(value)
    diagnostics = manifest.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        for key in ("child_tax_unit_agi_drift", "capital_gains_lots"):
            value = diagnostics.get(key)
            if isinstance(value, Mapping):
                summary[key] = dict(value)
    return summary


def _summarize_child_tax_unit_agi_drift_ratios(
    payload: dict[str, Any],
    *,
    stage: str,
    variables: tuple[str, ...],
) -> dict[str, Any]:
    stages = dict(payload.get("stages", {}))
    stage_payload = dict(stages.get(stage, {}))
    subsets = dict(stage_payload.get("subsets", {}))
    adults = dict(subsets.get("adults", {}))
    dependents = dict(subsets.get("dependents_under_20", {}))
    ratios: dict[str, float | None] = {}
    for variable in variables:
        adult_sum = adults.get(variable, {}).get("sum")
        child_sum = dependents.get(variable, {}).get("sum")
        if adult_sum in (None, 0):
            ratios[variable] = None
        else:
            ratios[variable] = float(child_sum or 0.0) / float(adult_sum)
    return {
        "stage": stage,
        "dependents_under_20_sum_share": ratios,
    }


def _resolve_policyengine_harness_context(
    result: USMicroplexBuildResult,
    *,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None,
    policyengine_target_provider: TargetProvider | None,
    policyengine_baseline_dataset: str | Path | None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ),
    policyengine_harness_metadata: dict[str, Any] | None,
) -> tuple[
    TargetProvider | None,
    str | Path | None,
    tuple[PolicyEngineUSHarnessSlice, ...],
    dict[str, Any],
]:
    resolved_target_provider = policyengine_target_provider
    if (
        resolved_target_provider is None
        and result.config.policyengine_targets_db is not None
    ):
        resolved_target_provider = PolicyEngineUSDBTargetProvider(
            result.config.policyengine_targets_db
        )

    resolved_baseline_dataset = (
        policyengine_baseline_dataset or result.config.policyengine_baseline_dataset
    )

    harness_period = result.config.policyengine_dataset_year or 2024
    if policyengine_harness_slices is not None:
        resolved_harness_slices = tuple(policyengine_harness_slices)
    elif result.config.policyengine_targets_db is not None:
        resolved_harness_slices = default_policyengine_us_db_all_target_slices(
            period=harness_period,
            reform_id=result.config.policyengine_target_reform_id,
        )
    else:
        resolved_harness_slices = default_policyengine_us_harness_slices(
            period=harness_period
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
            Path(result.config.policyengine_targets_db).name
            if result.config.policyengine_targets_db is not None
            else None
        ),
        "target_period": result.config.policyengine_target_period,
        "target_variables": list(result.config.policyengine_target_variables),
        "target_domains": list(result.config.policyengine_target_domains),
        "target_geo_levels": list(result.config.policyengine_target_geo_levels),
        "target_profile": result.config.policyengine_target_profile,
        "calibration_target_profile": (
            result.config.policyengine_calibration_target_profile
        ),
        "target_reform_id": result.config.policyengine_target_reform_id,
        "harness_slice_names": [
            slice_spec.name for slice_spec in resolved_harness_slices
        ],
        "policyengine_us_runtime_version": _resolve_policyengine_us_runtime_version(),
        "harness_suite": (
            "policyengine_us_all_targets"
            if result.config.policyengine_targets_db is not None
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


def _resolve_policyengine_us_runtime_version() -> str | None:
    try:
        return version("policyengine-us")
    except PackageNotFoundError:
        return None
