"""Stage resume orchestration for PE-US-data checkpoint rebuilds."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from microplex.core import ObservationFrame, SourceQuery

from microplex_us.pipelines.artifact_io import _stage_artifact_ref, _stage_diagnostics
from microplex_us.pipelines.artifacts import USMicroplexVersionedBuildArtifacts
from microplex_us.pipelines.pe_us_data_rebuild import PEUSDataRebuildProgram
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_artifacts import (
    _load_checkpoint_manifest,
    _load_checkpoint_manifest_if_available,
    _load_checkpoint_versioned_artifacts,
    _load_resume_dataframe_artifact,
    _load_resume_json_artifact,
    _load_resume_policyengine_tables,
    _load_resume_targets,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_common import (
    _emit_checkpoint_progress,
    _write_json_atomically,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_evidence import (
    attach_policyengine_us_data_rebuild_checkpoint_evidence,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_types import (
    PEUSDataRebuildCheckpointResult,
)
from microplex_us.pipelines.stage_contracts import (
    US_CANONICAL_STAGE_IDS,
    canonicalize_us_pipeline_stage_id,
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.stage_policyengine_artifacts import (
    write_us_policyengine_entity_stage_artifact,
)
from microplex_us.pipelines.stage_resume import (
    USStageResumeArtifactRequirement,
    preflight_us_stage_resume,
)
from microplex_us.pipelines.stage_run import (
    USArtifactRef,
    USCalibrationOutputs,
    USDiagnosticOutput,
    USPolicyEngineEntityOutputs,
    USRunProfileOutputs,
    USStageInputOverride,
    resolve_us_manifest_or_contract_artifact_path,
)
from microplex_us.pipelines.stage_runtime import USStageRuntimeWriter
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexPipeline,
    USMicroplexTargets,
)
from microplex_us.pipelines.versioned_artifacts import (
    _finalize_versioned_build_artifacts,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    save_us_pipeline_checkpoint,
)

if TYPE_CHECKING:
    from microplex.core import SourceProvider
    from microplex.targets import TargetProvider

    from microplex_us.pipelines.registry import FrontierMetric
    from microplex_us.policyengine.harness import (
        PolicyEngineUSComparisonCache,
        PolicyEngineUSHarnessSlice,
    )


def _resolve_checkpoint_resume_artifact_root(
    output_root: str | Path,
    *,
    version_id: str | None,
    resume_from_stage: str | None = None,
) -> Path:
    root = Path(output_root).expanduser()
    if version_id is not None:
        return root / version_id
    if (root / "manifest.json").exists():
        return root
    if (
        resume_from_stage is not None
        and canonicalize_us_pipeline_stage_id(resume_from_stage) == "01_run_profile"
    ):
        return root
    raise ValueError(
        "resume_from_stage requires --version-id unless --output-root points "
        "directly at a saved artifact directory with manifest.json"
    )


def _is_artifact_backed_checkpoint_resume_stage(stage_id: str) -> bool:
    return US_CANONICAL_STAGE_IDS.index(stage_id) >= US_CANONICAL_STAGE_IDS.index(
        "06_policyengine_entities"
    )


def _resume_provider_context_from_manifest(
    artifact_root: Path,
    manifest: dict[str, Any],
) -> tuple[tuple[str, ...], dict[str, SourceQuery]]:
    plan = _resume_provider_query_plan_from_manifest(artifact_root, manifest)
    provider_names = _string_tuple(plan.get("provider_names"))
    if not provider_names:
        provider_names = _string_tuple(plan.get("source_names"))
    if not provider_names:
        provider_names = _string_tuple(
            dict(manifest.get("synthesis", {})).get("source_names")
        )
    queries = _resume_queries_from_provider_plan(plan)
    if not provider_names and queries:
        provider_names = tuple(queries)
    return provider_names, queries


def _resume_provider_query_plan_from_manifest(
    artifact_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    stage_manifests = manifest.get("stage_output_manifests")
    stage_manifest_path = None
    if isinstance(stage_manifests, Mapping):
        stage_manifest_path = stage_manifests.get("01_run_profile")
    path = (
        artifact_root / str(stage_manifest_path)
        if isinstance(stage_manifest_path, str)
        else artifact_root / "stage_artifacts" / "manifests" / "01_run_profile.json"
    )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        return {}
    plan = outputs.get("provider_query_plan")
    return dict(plan) if isinstance(plan, Mapping) else {}


def _resume_queries_from_provider_plan(
    provider_query_plan: Mapping[str, Any],
) -> dict[str, SourceQuery]:
    queries_payload = provider_query_plan.get("queries")
    if not isinstance(queries_payload, Mapping):
        return {}
    queries: dict[str, SourceQuery] = {}
    for key, value in queries_payload.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            continue
        provider_filters = value.get("provider_filters")
        if provider_filters is None:
            provider_filters = value.get("providerFilters")
        queries[key] = SourceQuery(
            provider_filters=(
                dict(provider_filters)
                if isinstance(provider_filters, Mapping)
                else dict(value)
            )
        )
    return queries


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if item)


def _checkpoint_resume_extra_artifact_requirements(
    resume_from_stage: str,
) -> tuple[USStageResumeArtifactRequirement, ...]:
    stage_id = canonicalize_us_pipeline_stage_id(resume_from_stage)
    stage_index = US_CANONICAL_STAGE_IDS.index(stage_id)
    requirements: list[USStageResumeArtifactRequirement] = []
    if stage_index >= US_CANONICAL_STAGE_IDS.index("06_policyengine_entities"):
        requirements.extend(
            [
                USStageResumeArtifactRequirement(
                    "05_donor_integration_synthesis",
                    "seed_data",
                    "runner must hydrate seed rows before replaying downstream stages",
                ),
                USStageResumeArtifactRequirement(
                    "05_donor_integration_synthesis",
                    "synthetic_data",
                    "runner must hydrate candidate rows before replaying downstream stages",
                ),
            ]
        )
    if stage_index >= US_CANONICAL_STAGE_IDS.index("08_dataset_assembly"):
        requirements.extend(
            [
                USStageResumeArtifactRequirement(
                    "07_calibration",
                    "calibrated_data",
                    "runner must hydrate calibrated rows before dataset assembly",
                ),
                USStageResumeArtifactRequirement(
                    "07_calibration",
                    "targets",
                    "runner must hydrate target payload before dataset assembly",
                ),
                USStageResumeArtifactRequirement(
                    "07_calibration",
                    "calibration_summary",
                    "runner must hydrate calibration summary before dataset assembly",
                ),
                USStageResumeArtifactRequirement(
                    "07_calibration",
                    "policyengine_entity_tables",
                    "runner must hydrate calibrated PE entity tables before dataset assembly",
                ),
            ]
        )
    if stage_index >= US_CANONICAL_STAGE_IDS.index("09_validation_benchmarking"):
        requirements.append(
            USStageResumeArtifactRequirement(
                "08_dataset_assembly",
                "policyengine_dataset",
                "validation and benchmark evidence require the assembled H5 dataset",
            )
        )
    return tuple(requirements)


def _resume_policyengine_table_summary(
    tables: PolicyEngineUSEntityTableBundle,
) -> dict[str, Any]:
    return {
        "households": int(len(tables.households)),
        "persons": int(len(tables.persons)),
        "tax_units": int(len(tables.tax_units)),
        "spm_units": int(len(tables.spm_units)),
        "families": int(len(tables.families)),
        "marital_units": int(len(tables.marital_units)),
    }


def _resume_target_ledger(targets: USMicroplexTargets) -> dict[str, Any]:
    return {
        "n_marginal_groups": len(targets.marginal),
        "n_continuous": len(targets.continuous),
        "marginal_keys": sorted(targets.marginal.keys()),
        "continuous_keys": sorted(targets.continuous.keys()),
    }


def _load_checkpoint_source_frames(
    providers: tuple[SourceProvider, ...],
    queries: dict[str, SourceQuery],
) -> list[ObservationFrame]:
    pipeline = USMicroplexPipeline()
    frames: list[ObservationFrame] = []
    for provider in providers:
        frame = provider.load_frame(
            pipeline._resolve_source_query(provider, queries or {})
        )
        frames.append(frame)
    return frames


def _complete_resume_run_profile_stage(
    *,
    stage_runtime_writer: USStageRuntimeWriter,
    config: USMicroplexBuildConfig,
    version_id: str,
    provider_names: tuple[str, ...],
    queries: dict[str, SourceQuery],
) -> None:
    stage_runtime_writer.start_stage(
        "01_run_profile",
        metadata={"version_id": version_id, "resume": True},
    )
    stage_runtime_writer.complete_stage(
        USRunProfileOutputs(
            manifest=USArtifactRef(
                key="manifest",
                path="manifest.json",
                format="json",
                required=True,
                assume_exists=True,
            ),
            resolved_config=config.to_dict(),
            provider_query_plan={
                "provider_names": list(provider_names),
                "queries": {
                    key: (
                        query.to_dict()
                        if hasattr(query, "to_dict")
                        else dict(getattr(query, "__dict__", {}))
                    )
                    for key, query in queries.items()
                },
            },
            diagnostics={
                "stage_summary": USDiagnosticOutput(
                    key="stage_summary",
                    description="Runtime run-profile summary.",
                    summary={
                        "provider_names": list(provider_names),
                        "version_id": version_id,
                        "resume": True,
                    },
                )
            },
        )
    )


def _resume_checkpoint_build_from_source_stage(
    *,
    artifact_root: Path,
    resume_from_stage: str,
    config: USMicroplexBuildConfig,
    providers: tuple[SourceProvider, ...],
    queries: dict[str, SourceQuery],
    stage_runtime_writer: USStageRuntimeWriter,
    provider_names: tuple[str, ...],
) -> USMicroplexBuildResult:
    pipeline = USMicroplexPipeline(config, stage_runtime_writer=stage_runtime_writer)
    if resume_from_stage == "01_run_profile":
        _complete_resume_run_profile_stage(
            stage_runtime_writer=stage_runtime_writer,
            config=config,
            version_id=artifact_root.name,
            provider_names=provider_names,
            queries=queries,
        )
        return pipeline.build_from_source_providers(list(providers), queries=queries)
    if resume_from_stage == "02_source_loading":
        return pipeline.build_from_source_providers(list(providers), queries=queries)

    frames = _load_checkpoint_source_frames(providers, queries)
    restored_scaffold_seed_data = None
    if resume_from_stage == "05_donor_integration_synthesis":
        manifest = _load_checkpoint_manifest(artifact_root)
        restored_scaffold_seed_data = _load_resume_dataframe_artifact(
            artifact_root,
            manifest,
            "scaffold_seed_data",
            stage_id="04_seed_scaffold",
        )
    return pipeline.build_from_frames(
        frames,
        resume_from_stage=resume_from_stage,
        restored_scaffold_seed_data=restored_scaffold_seed_data,
    )


def _load_resume_build_result_base(
    *,
    artifact_root: Path,
    config: USMicroplexBuildConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame | None, USMicroplexTargets]:
    manifest = _load_checkpoint_manifest_if_available(artifact_root)
    seed_data = _load_resume_dataframe_artifact(
        artifact_root,
        manifest,
        "seed_data",
        stage_id="05_donor_integration_synthesis",
    )
    synthetic_data = _load_resume_dataframe_artifact(
        artifact_root,
        manifest,
        "synthetic_data",
        stage_id="05_donor_integration_synthesis",
    )
    scaffold_seed_data_path = resolve_us_manifest_or_contract_artifact_path(
        artifact_root,
        manifest,
        "scaffold_seed_data",
        stage_id="04_seed_scaffold",
    )
    scaffold_seed_data = (
        pd.read_parquet(scaffold_seed_data_path)
        if scaffold_seed_data_path.exists()
        else None
    )
    targets = _load_resume_targets(
        artifact_root,
        manifest,
        config=config,
        seed_data=seed_data,
    )
    return manifest, seed_data, synthetic_data, scaffold_seed_data, targets


def _run_checkpoint_policyengine_entity_resume_stage(
    *,
    pipeline: USMicroplexPipeline,
    synthetic_data: pd.DataFrame,
) -> PolicyEngineUSEntityTableBundle:
    pipeline._runtime_start_stage("06_policyengine_entities")
    try:
        synthetic_tables = pipeline.build_policyengine_entity_tables(synthetic_data)
        if pipeline.config.pipeline_checkpoint_save_post_imputation_path is not None:
            save_us_pipeline_checkpoint(
                synthetic_tables,
                pipeline.config.pipeline_checkpoint_save_post_imputation_path,
                stage="post_imputation",
            )
        pipeline._check_policyengine_export_column_contract(
            synthetic_tables,
            stage="pre_calibration",
        )
        if pipeline.stage_runtime_writer is not None:
            write_us_policyengine_entity_stage_artifact(
                synthetic_tables,
                pipeline.stage_runtime_writer.artifact_root,
                stage_id="06_policyengine_entities",
                artifact_key="pre_calibration_policyengine_entity_tables",
                checkpoint_stage="post_microsim",
            )
            entity_summary = _resume_policyengine_table_summary(synthetic_tables)
            pipeline.stage_runtime_writer.complete_stage(
                USPolicyEngineEntityOutputs(
                    pre_calibration_policyengine_entity_tables=_stage_artifact_ref(
                        pipeline.stage_runtime_writer.artifact_root,
                        "06_policyengine_entities",
                        "pre_calibration_policyengine_entity_tables",
                    ),
                    materialized_policyengine_inputs=entity_summary,
                    diagnostics=_stage_diagnostics(
                        "06_policyengine_entities",
                        entity_summary,
                    ),
                )
            )
    except Exception as exc:
        pipeline._runtime_fail_stage("06_policyengine_entities", exc)
        raise
    return synthetic_tables


def _run_checkpoint_calibration_resume_stage(
    *,
    pipeline: USMicroplexPipeline,
    synthetic_data: pd.DataFrame,
    synthetic_tables: PolicyEngineUSEntityTableBundle,
    targets: USMicroplexTargets,
) -> tuple[PolicyEngineUSEntityTableBundle, pd.DataFrame, dict[str, Any]]:
    pipeline._runtime_start_stage("07_calibration")
    try:
        if pipeline._has_policyengine_calibration_targets():
            policyengine_tables, calibrated_data, calibration_summary = (
                pipeline.calibrate_policyengine_tables(synthetic_tables)
            )
        else:
            calibrated_data, calibration_summary = pipeline.calibrate(
                synthetic_data,
                targets,
            )
            policyengine_tables = pipeline.build_policyengine_entity_tables(
                calibrated_data
            )
        if pipeline.stage_runtime_writer is not None:
            artifact_root = pipeline.stage_runtime_writer.artifact_root
            write_us_policyengine_entity_stage_artifact(
                policyengine_tables,
                artifact_root,
                stage_id="07_calibration",
                artifact_key="policyengine_entity_tables",
                checkpoint_stage="post_calibration",
            )
            calibrated_data_path = resolve_us_stage_artifact_contract_path(
                artifact_root,
                "07_calibration",
                "calibrated_data",
            )
            targets_path = resolve_us_stage_artifact_contract_path(
                artifact_root,
                "07_calibration",
                "targets",
            )
            calibration_summary_path = resolve_us_stage_artifact_contract_path(
                artifact_root,
                "07_calibration",
                "calibration_summary",
            )
            calibrated_data_path.parent.mkdir(parents=True, exist_ok=True)
            targets_path.parent.mkdir(parents=True, exist_ok=True)
            calibration_summary_path.parent.mkdir(parents=True, exist_ok=True)
            calibrated_data.to_parquet(calibrated_data_path, index=False)
            _write_json_atomically(
                targets_path,
                {
                    "marginal": targets.marginal,
                    "continuous": targets.continuous,
                },
            )
            _write_json_atomically(calibration_summary_path, calibration_summary)
            target_ledger = _resume_target_ledger(targets)
            pipeline.stage_runtime_writer.complete_stage(
                USCalibrationOutputs(
                    calibrated_data=_stage_artifact_ref(
                        artifact_root,
                        "07_calibration",
                        "calibrated_data",
                    ),
                    targets=_stage_artifact_ref(
                        artifact_root,
                        "07_calibration",
                        "targets",
                    ),
                    calibration_summary=_stage_artifact_ref(
                        artifact_root,
                        "07_calibration",
                        "calibration_summary",
                    ),
                    policyengine_entity_tables=_stage_artifact_ref(
                        artifact_root,
                        "07_calibration",
                        "policyengine_entity_tables",
                    ),
                    target_ledger=target_ledger,
                    diagnostics=_stage_diagnostics(
                        "07_calibration",
                        {
                            "calibrated_rows": int(len(calibrated_data)),
                            "backend": pipeline.config.calibration_backend,
                            **target_ledger,
                        },
                    ),
                )
            )
    except Exception as exc:
        pipeline._runtime_fail_stage("07_calibration", exc)
        raise
    return policyengine_tables, calibrated_data, calibration_summary


def _resume_checkpoint_build_from_saved_stage(
    *,
    artifact_root: Path,
    resume_from_stage: str,
    config: USMicroplexBuildConfig,
    stage_runtime_writer: USStageRuntimeWriter,
) -> USMicroplexBuildResult:
    (
        manifest,
        seed_data,
        synthetic_data,
        scaffold_seed_data,
        targets,
    ) = _load_resume_build_result_base(artifact_root=artifact_root, config=config)
    synthesis_metadata = dict(manifest.get("synthesis", {}))
    synthesis_metadata["stage_resume"] = {
        "source_artifact_dir": str(artifact_root),
        "resume_from_stage": resume_from_stage,
    }
    pipeline = USMicroplexPipeline(config, stage_runtime_writer=stage_runtime_writer)
    stage_index = US_CANONICAL_STAGE_IDS.index(resume_from_stage)

    if stage_index <= US_CANONICAL_STAGE_IDS.index("06_policyengine_entities"):
        pre_calibration_tables = _run_checkpoint_policyengine_entity_resume_stage(
            pipeline=pipeline,
            synthetic_data=synthetic_data,
        )
    else:
        pre_calibration_tables = _load_resume_policyengine_tables(
            artifact_root,
            manifest,
            "pre_calibration_policyengine_entity_tables",
            stage_id="06_policyengine_entities",
            expected_stage="post_microsim",
        )

    if stage_index <= US_CANONICAL_STAGE_IDS.index("07_calibration"):
        policyengine_tables, calibrated_data, calibration_summary = (
            _run_checkpoint_calibration_resume_stage(
                pipeline=pipeline,
                synthetic_data=synthetic_data,
                synthetic_tables=pre_calibration_tables,
                targets=targets,
            )
        )
    else:
        calibrated_data = _load_resume_dataframe_artifact(
            artifact_root,
            manifest,
            "calibrated_data",
            stage_id="07_calibration",
        )
        calibration_summary = _load_resume_json_artifact(
            artifact_root,
            manifest,
            "calibration_summary",
            stage_id="07_calibration",
        )
        policyengine_tables = _load_resume_policyengine_tables(
            artifact_root,
            manifest,
            "policyengine_entity_tables",
            stage_id="07_calibration",
            expected_stage="post_calibration",
        )

    return USMicroplexBuildResult(
        config=config,
        seed_data=seed_data,
        synthetic_data=synthetic_data,
        calibrated_data=calibrated_data,
        targets=targets,
        calibration_summary=calibration_summary,
        synthesis_metadata=synthesis_metadata,
        policyengine_tables=policyengine_tables,
        pre_calibration_policyengine_tables=pre_calibration_tables,
        scaffold_seed_data=scaffold_seed_data,
    )


def _run_policyengine_us_data_rebuild_checkpoint_resume(
    *,
    output_root: str | Path,
    version_id: str | None,
    resume_from_stage: str,
    resolved_config: USMicroplexBuildConfig,
    program: PEUSDataRebuildProgram,
    resolved_providers: tuple[SourceProvider, ...],
    provider_names: tuple[str, ...],
    resolved_queries: dict[str, SourceQuery],
    frontier_metric: FrontierMetric,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None,
    policyengine_target_provider: TargetProvider | None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ),
    resolved_harness_metadata: dict[str, Any],
    policyengine_us_data_repo: str | Path | None,
    policyengine_us_data_python: str | Path | None,
    defer_policyengine_harness: bool,
    require_policyengine_native_score: bool,
    defer_policyengine_native_score: bool,
    defer_native_audit: bool,
    defer_imputation_ablation: bool,
    precomputed_policyengine_harness_payload: dict[str, Any] | None,
    precomputed_policyengine_native_scores: dict[str, Any] | None,
    precomputed_imputation_ablation_payload: dict[str, Any] | None,
    run_registry_path: str | Path | None,
    run_index_path: str | Path | None,
    resolved_registry_metadata: dict[str, Any],
    allow_stage_input_overrides: bool,
    stage_input_overrides: tuple[USStageInputOverride, ...],
) -> PEUSDataRebuildCheckpointResult:
    resume_stage_id = canonicalize_us_pipeline_stage_id(resume_from_stage)
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
    manifest = _load_checkpoint_manifest_if_available(artifact_root)
    stage_runtime_writer = USStageRuntimeWriter(
        artifact_root,
        manifest_payload=manifest,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )
    _emit_checkpoint_progress(
        "PE-US-data rebuild checkpoint: resuming build",
        artifact_dir=artifact_root,
        resume_from_stage=resume_stage_id,
        providers=",".join(provider_names),
    )

    build_result: USMicroplexBuildResult | None = None
    if resume_stage_id in {
        "01_run_profile",
        "02_source_loading",
        "03_source_planning",
        "04_seed_scaffold",
        "05_donor_integration_synthesis",
    }:
        build_result = _resume_checkpoint_build_from_source_stage(
            artifact_root=artifact_root,
            resume_from_stage=resume_stage_id,
            config=resolved_config,
            providers=resolved_providers,
            queries=resolved_queries,
            stage_runtime_writer=stage_runtime_writer,
            provider_names=provider_names,
        )
    elif resume_stage_id in {
        "06_policyengine_entities",
        "07_calibration",
        "08_dataset_assembly",
        "09_validation_benchmarking",
    }:
        build_result = _resume_checkpoint_build_from_saved_stage(
            artifact_root=artifact_root,
            resume_from_stage=resume_stage_id,
            config=resolved_config,
            stage_runtime_writer=stage_runtime_writer,
        )

    artifacts: USMicroplexVersionedBuildArtifacts | None = None
    if build_result is not None and resume_stage_id != "09_validation_benchmarking":
        artifacts = _finalize_versioned_build_artifacts(
            build_result,
            output_root=artifact_root.parent,
            version_id=artifact_root.name,
            preallocated_output_dir=artifact_root,
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
            stage_runtime_writer=stage_runtime_writer,
        )
        _emit_checkpoint_progress(
            "PE-US-data rebuild checkpoint: resumed build complete",
            artifact_dir=artifact_root,
            resume_from_stage=resume_stage_id,
            frontier_metric=frontier_metric,
        )

    _emit_checkpoint_progress(
        "PE-US-data rebuild checkpoint: attaching PE evidence",
        artifact_dir=artifact_root,
        compute_harness=not defer_policyengine_harness,
        compute_native_scores=not defer_policyengine_native_score,
        compute_native_audit=not defer_native_audit,
        compute_imputation_ablation=not defer_imputation_ablation,
    )
    evidence = attach_policyengine_us_data_rebuild_checkpoint_evidence(
        artifact_root,
        build_result=build_result,
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
        build_result=(
            artifacts.build_result
            if artifacts is not None
            else build_result
        ),
        artifact_root=artifact_root,
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
