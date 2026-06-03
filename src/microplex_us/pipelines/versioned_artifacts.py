"""Versioned build-and-save entrypoints for US Microplex artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microplex.core import SourceProvider, SourceQuery
from microplex.targets import TargetProvider

from microplex_us.pipelines.artifact_types import (
    USMicroplexArtifactPaths,
    USMicroplexVersionedBuildArtifacts,
)
from microplex_us.pipelines.registry import (
    FrontierMetric,
    load_us_microplex_run_registry,
    select_us_microplex_frontier_entry,
)
from microplex_us.pipelines.stage_run import (
    USArtifactRef,
    USDiagnosticOutput,
    USRunProfileOutputs,
    USStageInputOverride,
)
from microplex_us.pipelines.stage_runtime import USStageRuntimeWriter
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexPipeline,
    build_us_microplex,
)
from microplex_us.policyengine.harness import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSHarnessSlice,
)


def _save_us_microplex_artifacts(*args: Any, **kwargs: Any) -> USMicroplexArtifactPaths:
    from microplex_us.pipelines.artifacts import save_us_microplex_artifacts

    return save_us_microplex_artifacts(*args, **kwargs)


def _facade_pipeline_cls() -> type[USMicroplexPipeline]:
    from microplex_us.pipelines import artifacts

    return artifacts.USMicroplexPipeline


def _finalize_via_facade(
    build_result: USMicroplexBuildResult,
    **kwargs: Any,
) -> USMicroplexVersionedBuildArtifacts:
    from microplex_us.pipelines import artifacts

    finalize = artifacts._finalize_versioned_build_artifacts
    if finalize is _finalize_versioned_build_artifacts:
        return _finalize_versioned_build_artifacts(build_result, **kwargs)
    return finalize(build_result, **kwargs)


def save_versioned_us_microplex_artifacts(
    result: USMicroplexBuildResult,
    output_root: str | Path,
    *,
    version_id: str | None = None,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
    stage_runtime_writer: USStageRuntimeWriter | None = None,
) -> USMicroplexArtifactPaths:
    """Persist a build under a stable versioned directory beneath one output root."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_version_id, output_dir = _allocate_versioned_output_dir(
        output_root,
        version_id=version_id,
        result=result,
    )
    paths = _save_us_microplex_artifacts(
        result,
        output_dir,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path or output_root / "run_registry.jsonl",
        run_index_path=run_index_path or output_root,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
        stage_runtime_writer=stage_runtime_writer,
    )
    return replace(paths, version_id=resolved_version_id)


def build_and_save_versioned_us_microplex(
    persons: Any,
    households: Any,
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
) -> USMicroplexVersionedBuildArtifacts:
    """Build a US microplex dataset, save a versioned bundle, and report frontier gap."""
    build_result = build_us_microplex(persons, households, config=config)
    return save_versioned_us_microplex_build_result(
        build_result,
        output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )


def save_versioned_us_microplex_build_result(
    build_result: USMicroplexBuildResult,
    output_root: str | Path,
    *,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
) -> USMicroplexVersionedBuildArtifacts:
    """Save an already-built result as a versioned bundle and report frontier gap."""
    return _finalize_via_facade(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )


def build_and_save_versioned_us_microplex_from_source_provider(
    provider: SourceProvider,
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    query: SourceQuery | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
) -> USMicroplexVersionedBuildArtifacts:
    """Build from one source provider, save a versioned bundle, and report frontier gap."""
    pipeline = _facade_pipeline_cls()(config)
    build_result = pipeline.build_from_source_provider(provider, query=query)
    return _finalize_via_facade(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )


def build_and_save_versioned_us_microplex_from_source_providers(
    providers: list[SourceProvider],
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    queries: dict[str, SourceQuery] | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
) -> USMicroplexVersionedBuildArtifacts:
    """Build from multiple source providers, save a versioned bundle, and report frontier gap."""
    resolved_config = config or USMicroplexBuildConfig()
    _resolved_version_id, preallocated_output_dir, stage_runtime_writer = (
        _initialize_versioned_stage_runtime_writer(
            output_root,
            version_id=version_id,
            config=resolved_config,
            providers=providers,
            queries=queries,
            allow_stage_input_overrides=allow_stage_input_overrides,
            stage_input_overrides=stage_input_overrides,
        )
    )
    pipeline = _facade_pipeline_cls()(
        resolved_config,
        stage_runtime_writer=stage_runtime_writer,
    )
    build_result = pipeline.build_from_source_providers(providers, queries=queries)
    return _finalize_via_facade(
        build_result,
        output_root=output_root,
        version_id=version_id,
        preallocated_output_dir=preallocated_output_dir,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
        stage_runtime_writer=stage_runtime_writer,
    )


def build_and_save_versioned_us_microplex_from_data_dir(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
) -> USMicroplexVersionedBuildArtifacts:
    """Build from a CPS-style parquet directory, save a versioned bundle, and report frontier gap."""
    pipeline = _facade_pipeline_cls()(config)
    build_result = pipeline.build_from_data_dir(data_dir)
    return _finalize_via_facade(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )


def _finalize_versioned_build_artifacts(
    build_result: USMicroplexBuildResult,
    *,
    output_root: str | Path,
    version_id: str | None,
    preallocated_output_dir: str | Path | None = None,
    frontier_metric: FrontierMetric,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None,
    policyengine_target_provider: TargetProvider | None,
    policyengine_baseline_dataset: str | Path | None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ),
    policyengine_harness_metadata: dict[str, Any] | None,
    policyengine_us_data_repo: str | Path | None,
    defer_policyengine_harness: bool,
    require_policyengine_native_score: bool,
    defer_policyengine_native_score: bool,
    precomputed_policyengine_harness_payload: dict[str, Any] | None,
    precomputed_policyengine_native_scores: dict[str, Any] | None,
    run_registry_path: str | Path | None,
    run_index_path: str | Path | None,
    run_registry_metadata: dict[str, Any] | None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
    stage_runtime_writer: USStageRuntimeWriter | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    if preallocated_output_dir is not None:
        output_root_path = Path(output_root)
        output_dir = Path(preallocated_output_dir)
        artifact_paths = _save_us_microplex_artifacts(
            build_result,
            output_dir,
            policyengine_comparison_cache=policyengine_comparison_cache,
            policyengine_target_provider=policyengine_target_provider,
            policyengine_baseline_dataset=policyengine_baseline_dataset,
            policyengine_harness_slices=policyengine_harness_slices,
            policyengine_harness_metadata=policyengine_harness_metadata,
            policyengine_us_data_repo=policyengine_us_data_repo,
            defer_policyengine_harness=defer_policyengine_harness,
            require_policyengine_native_score=require_policyengine_native_score,
            defer_policyengine_native_score=defer_policyengine_native_score,
            precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
            precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
            run_registry_path=run_registry_path
            or output_root_path / "run_registry.jsonl",
            run_index_path=run_index_path or output_root_path,
            run_registry_metadata=run_registry_metadata,
            enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
            child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
            allow_stage_input_overrides=allow_stage_input_overrides,
            stage_input_overrides=stage_input_overrides,
            stage_runtime_writer=stage_runtime_writer,
        )
        artifact_paths = replace(artifact_paths, version_id=output_dir.name)
    else:
        artifact_paths = save_versioned_us_microplex_artifacts(
            build_result,
            output_root,
            version_id=version_id,
            policyengine_comparison_cache=policyengine_comparison_cache,
            policyengine_target_provider=policyengine_target_provider,
            policyengine_baseline_dataset=policyengine_baseline_dataset,
            policyengine_harness_slices=policyengine_harness_slices,
            policyengine_harness_metadata=policyengine_harness_metadata,
            policyengine_us_data_repo=policyengine_us_data_repo,
            defer_policyengine_harness=defer_policyengine_harness,
            require_policyengine_native_score=require_policyengine_native_score,
            defer_policyengine_native_score=defer_policyengine_native_score,
            precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
            precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
            run_registry_path=run_registry_path,
            run_index_path=run_index_path,
            run_registry_metadata=run_registry_metadata,
            enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
            child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
            allow_stage_input_overrides=allow_stage_input_overrides,
            stage_input_overrides=stage_input_overrides,
            stage_runtime_writer=stage_runtime_writer,
        )
    current_entry = None
    frontier_entry = None
    frontier_delta = None
    if (
        artifact_paths.run_registry is not None
        and artifact_paths.version_id is not None
    ):
        registry_entries = load_us_microplex_run_registry(artifact_paths.run_registry)
        current_entry = next(
            (
                entry
                for entry in reversed(registry_entries)
                if entry.artifact_id == artifact_paths.version_id
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


def _allocate_versioned_output_dir(
    output_root: Path,
    *,
    version_id: str | None,
    result: USMicroplexBuildResult,
) -> tuple[str, Path]:
    return _allocate_versioned_output_dir_for_config(
        output_root,
        version_id=version_id,
        config=result.config.to_dict(),
    )


def _allocate_versioned_output_dir_for_config(
    output_root: Path,
    *,
    version_id: str | None,
    config: dict[str, Any],
) -> tuple[str, Path]:
    if version_id is not None:
        output_dir = output_root / version_id
        if output_dir.exists():
            if _version_dir_contains_only_configured_checkpoints(output_dir, config):
                return version_id, output_dir
            raise FileExistsError(
                f"Versioned artifact directory already exists: {output_dir}"
            )
        return version_id, output_dir

    config_hash = _short_config_hash(config)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_version_id = f"{timestamp}-{config_hash}"
    candidate_version_id = base_version_id
    suffix = 2
    output_dir = output_root / candidate_version_id
    while output_dir.exists():
        candidate_version_id = f"{base_version_id}-{suffix}"
        output_dir = output_root / candidate_version_id
        suffix += 1
    return candidate_version_id, output_dir


def _version_dir_contains_only_configured_checkpoints(
    output_dir: Path,
    config: Mapping[str, Any],
) -> bool:
    if not output_dir.is_dir():
        return False

    resolved_output_dir = output_dir.expanduser().resolve(strict=False)
    checkpoint_roots = _configured_checkpoint_roots_inside_version_dir(
        resolved_output_dir,
        config,
    )
    if not checkpoint_roots:
        return False

    return all(
        _path_is_allowed_checkpoint_tree_member(path, checkpoint_roots)
        for path in output_dir.rglob("*")
    )


def _configured_checkpoint_roots_inside_version_dir(
    output_dir: Path,
    config: Mapping[str, Any],
) -> tuple[Path, ...]:
    roots: list[Path] = []
    for key in (
        "pipeline_checkpoint_save_post_imputation_path",
        "pipeline_checkpoint_save_post_microsim_path",
    ):
        checkpoint_path = config.get(key)
        if checkpoint_path is None:
            continue
        resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve(
            strict=False
        )
        if (
            resolved_checkpoint_path != output_dir
            and _path_is_relative_to(resolved_checkpoint_path, output_dir)
        ):
            roots.append(resolved_checkpoint_path)
    return tuple(roots)


def _path_is_allowed_checkpoint_tree_member(
    path: Path,
    checkpoint_roots: tuple[Path, ...],
) -> bool:
    resolved_path = path.expanduser().resolve(strict=False)
    return any(
        _path_is_relative_to(resolved_path, checkpoint_root)
        or _path_is_relative_to(checkpoint_root, resolved_path)
        for checkpoint_root in checkpoint_roots
    )


def _path_is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


def _short_config_hash(config: dict[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _initialize_versioned_stage_runtime_writer(
    output_root: str | Path,
    *,
    version_id: str | None,
    config: USMicroplexBuildConfig,
    providers: list[SourceProvider],
    queries: dict[str, SourceQuery] | None,
    allow_stage_input_overrides: bool,
    stage_input_overrides: tuple[USStageInputOverride, ...],
) -> tuple[str, Path, USStageRuntimeWriter]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    resolved_version_id, output_dir = _allocate_versioned_output_dir_for_config(
        root,
        version_id=version_id,
        config=config.to_dict(),
    )
    provider_query_plan = _provider_query_plan(providers, queries)
    writer = USStageRuntimeWriter(
        output_dir,
        manifest_payload={
            "created_at": datetime.now(UTC).isoformat(),
            "config": config.to_dict(),
            "artifacts": {"manifest": "manifest.json"},
        },
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )
    writer.start_stage(
        "01_run_profile",
        metadata={"version_id": resolved_version_id},
    )
    writer.complete_stage(
        USRunProfileOutputs(
            manifest=USArtifactRef(
                key="manifest",
                path="manifest.json",
                format="json",
                required=True,
                assume_exists=True,
            ),
            resolved_config=config.to_dict(),
            provider_query_plan=provider_query_plan,
            diagnostics={
                "stage_summary": USDiagnosticOutput(
                    key="stage_summary",
                    description="Runtime run-profile summary.",
                    summary={
                        "provider_names": provider_query_plan["provider_names"],
                        "version_id": resolved_version_id,
                    },
                )
            },
        )
    )
    return resolved_version_id, output_dir, writer


def _provider_query_plan(
    providers: list[SourceProvider],
    queries: dict[str, SourceQuery] | None,
) -> dict[str, Any]:
    return {
        "provider_names": [provider.descriptor.name for provider in providers],
        "queries": {
            key: _json_ready_query(query) for key, query in dict(queries or {}).items()
        },
    }


def _json_ready_query(query: SourceQuery) -> dict[str, Any]:
    if hasattr(query, "to_dict"):
        payload = query.to_dict()
        if isinstance(payload, dict):
            return payload
    if hasattr(query, "__dataclass_fields__"):
        return _json_ready(asdict(query))
    return _json_ready(vars(query))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return value


def _registry_metric_value(entry: Any | None, metric: FrontierMetric) -> float | None:
    if entry is None:
        return None
    return getattr(entry, metric, None)
