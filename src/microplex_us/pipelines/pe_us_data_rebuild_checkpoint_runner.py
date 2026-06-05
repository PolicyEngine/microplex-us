"""Top-level PE-US-data checkpoint rebuild runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from microplex.core import SourceQuery

from microplex_us.pipelines.artifacts import (
    build_and_save_versioned_us_microplex_from_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild import (
    default_policyengine_us_data_rebuild_program,
    default_policyengine_us_data_rebuild_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_artifacts import (
    _load_checkpoint_manifest,
    _load_checkpoint_versioned_artifacts,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_common import (
    _emit_checkpoint_progress,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_config import (
    _validate_checkpoint_config_context,
    _validate_query_keys,
    default_policyengine_us_data_rebuild_checkpoint_config,
    default_policyengine_us_data_rebuild_queries,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_evidence import (
    attach_policyengine_us_data_rebuild_checkpoint_evidence,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_resume import (
    _checkpoint_resume_extra_artifact_requirements,
    _is_artifact_backed_checkpoint_resume_stage,
    _resolve_checkpoint_resume_artifact_root,
    _resume_provider_context_from_manifest,
    _run_policyengine_us_data_rebuild_checkpoint_resume,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_types import (
    PEUSDataRebuildCheckpointResult,
)
from microplex_us.pipelines.stage_contracts import (
    US_CANONICAL_STAGE_IDS,
    canonicalize_us_pipeline_stage_id,
)
from microplex_us.pipelines.stage_resume import preflight_us_stage_resume
from microplex_us.pipelines.stage_run import USStageInputOverride
from microplex_us.pipelines.us import USMicroplexBuildConfig

if TYPE_CHECKING:
    from microplex.core import SourceProvider
    from microplex.targets import TargetProvider

    from microplex_us.pipelines.registry import FrontierMetric
    from microplex_us.policyengine.harness import (
        PolicyEngineUSComparisonCache,
        PolicyEngineUSHarnessSlice,
    )


def run_policyengine_us_data_rebuild_checkpoint(
    output_root: str | Path,
    *,
    policyengine_baseline_dataset: str | Path,
    policyengine_targets_db: str | Path,
    arch_targets_db: str | Path | tuple[str | Path, ...] | None = None,
    calibration_target_source: Literal["policyengine", "arch"] = "policyengine",
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
    include_sipp: bool | None = None,
    include_scf: bool | None = None,
    acs_year: int = 2024,
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
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
    resume_from_stage: str | None = None,
) -> PEUSDataRebuildCheckpointResult:
    """Run one saved rebuild checkpoint and write its PE comparison sidecars."""

    if config is not None and config_overrides:
        raise ValueError(
            "config_overrides cannot be used when an explicit config is supplied"
        )
    resolved_config = config or default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_targets_db=policyengine_targets_db,
        arch_targets_db=arch_targets_db,
        calibration_target_source=calibration_target_source,
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
            arch_targets_db=arch_targets_db,
            calibration_target_source=calibration_target_source,
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
    resume_stage_id = (
        canonicalize_us_pipeline_stage_id(resume_from_stage)
        if resume_from_stage is not None
        else None
    )
    if resume_stage_id is not None and resume_stage_id not in US_CANONICAL_STAGE_IDS:
        raise ValueError(f"Unknown US pipeline stage: {resume_from_stage}")
    if (
        resume_stage_id is not None
        and _is_artifact_backed_checkpoint_resume_stage(resume_stage_id)
    ):
        artifact_root = _resolve_checkpoint_resume_artifact_root(
            output_root,
            version_id=version_id,
            resume_from_stage=resume_stage_id,
        )
        preflight = preflight_us_stage_resume(
            artifact_root,
            resume_stage_id,
            extra_required_artifacts=_checkpoint_resume_extra_artifact_requirements(
                resume_stage_id
            ),
        )
        preflight.raise_for_missing()
        manifest = _load_checkpoint_manifest(artifact_root)
        provider_names, resolved_queries = _resume_provider_context_from_manifest(
            artifact_root,
            manifest,
        )
        program = default_policyengine_us_data_rebuild_program()
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
        return _run_policyengine_us_data_rebuild_checkpoint_resume(
            output_root=output_root,
            version_id=version_id,
            resume_from_stage=resume_stage_id,
            resolved_config=resolved_config,
            program=program,
            resolved_providers=(),
            provider_names=provider_names,
            resolved_queries=resolved_queries,
            frontier_metric=frontier_metric,
            policyengine_comparison_cache=policyengine_comparison_cache,
            policyengine_target_provider=policyengine_target_provider,
            policyengine_harness_slices=policyengine_harness_slices,
            resolved_harness_metadata=resolved_harness_metadata,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            defer_policyengine_harness=defer_policyengine_harness,
            require_policyengine_native_score=require_policyengine_native_score,
            defer_policyengine_native_score=defer_policyengine_native_score,
            defer_native_audit=defer_native_audit,
            defer_imputation_ablation=defer_imputation_ablation,
            precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
            precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
            precomputed_imputation_ablation_payload=precomputed_imputation_ablation_payload,
            run_registry_path=run_registry_path,
            run_index_path=run_index_path,
            resolved_registry_metadata=resolved_registry_metadata,
            allow_stage_input_overrides=allow_stage_input_overrides,
            stage_input_overrides=stage_input_overrides,
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
    if resume_from_stage is not None:
        return _run_policyengine_us_data_rebuild_checkpoint_resume(
            output_root=output_root,
            version_id=version_id,
            resume_from_stage=resume_from_stage,
            resolved_config=resolved_config,
            program=program,
            resolved_providers=resolved_providers,
            provider_names=provider_names,
            resolved_queries=resolved_queries,
            frontier_metric=frontier_metric,
            policyengine_comparison_cache=policyengine_comparison_cache,
            policyengine_target_provider=policyengine_target_provider,
            policyengine_harness_slices=policyengine_harness_slices,
            resolved_harness_metadata=resolved_harness_metadata,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            defer_policyengine_harness=defer_policyengine_harness,
            require_policyengine_native_score=require_policyengine_native_score,
            defer_policyengine_native_score=defer_policyengine_native_score,
            defer_native_audit=defer_native_audit,
            defer_imputation_ablation=defer_imputation_ablation,
            precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
            precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
            precomputed_imputation_ablation_payload=precomputed_imputation_ablation_payload,
            run_registry_path=run_registry_path,
            run_index_path=run_index_path,
            resolved_registry_metadata=resolved_registry_metadata,
            allow_stage_input_overrides=allow_stage_input_overrides,
            stage_input_overrides=stage_input_overrides,
        )
    _emit_checkpoint_progress(
        "PE-US-data rebuild checkpoint: starting build",
        output_root=Path(output_root).expanduser(),
        version_id=version_id or "auto",
        target_profile=resolved_config.policyengine_target_profile,
        calibration_target_profile=(
            resolved_config.policyengine_calibration_target_profile
        ),
        calibration_target_source=resolved_config.calibration_target_source,
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
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
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
        native_target_diagnostics_path=getattr(
            evidence,
            "native_target_diagnostics_path",
            None,
        ),
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
        native_target_diagnostics_path=getattr(
            evidence,
            "native_target_diagnostics_path",
            None,
        ),
        native_target_diagnostics_payload=getattr(
            evidence,
            "native_target_diagnostics_payload",
            None,
        ),
        imputation_ablation_path=evidence.imputation_ablation_path,
        imputation_ablation_payload=evidence.imputation_ablation_payload,
    )
