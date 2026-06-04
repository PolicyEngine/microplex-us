"""Artifact persistence for production pipeline outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microplex.targets import (
    TargetProvider,
    assert_valid_benchmark_artifact_manifest,
)

from microplex_us.pipeline_metadata import pipeline_node
from microplex_us.pipelines.artifact_dataset_assembly import (
    _maybe_write_capital_gains_lot_artifact,
)
from microplex_us.pipelines.artifact_io import (
    _stage_artifact_ref,
    _stage_diagnostics,
    _write_json_atomically,
    _write_json_unless_live_artifact_exists,
    _write_parquet_unless_live_artifact_exists,
)
from microplex_us.pipelines.artifact_replay import (
    replay_and_save_versioned_us_microplex_policyengine_stage,
    replay_us_microplex_policyengine_stage_from_artifact,
)
from microplex_us.pipelines.artifact_source_diagnostics import (
    _build_source_weight_diagnostics,
    _write_us_source_plan_artifact,
)
from microplex_us.pipelines.artifact_types import (
    USMicroplexArtifactPaths,
    USMicroplexVersionedBuildArtifacts,
)
from microplex_us.pipelines.artifact_validation import (
    _resolve_policyengine_harness_context,
    _stage9_benchmark_summary,
    _summarize_child_tax_unit_agi_drift_ratios,
)
from microplex_us.pipelines.index_db import (
    append_us_microplex_run_index_entry,
)
from microplex_us.pipelines.pe_native_scores import (
    compute_us_pe_native_scores,
)
from microplex_us.pipelines.registry import (
    append_us_microplex_run_registry_entry,
    build_us_microplex_run_registry_entry,
)
from microplex_us.pipelines.stage_contracts import (
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.stage_manifest import (
    write_us_policyengine_entity_stage_artifact,
    write_us_validation_evidence_manifest,
)
from microplex_us.pipelines.stage_run import (
    USDatasetAssemblyOutputs,
    USStageInputOverride,
    USValidationBenchmarkingOutputs,
    write_us_stage_run_manifests_from_artifact_manifest,
)
from microplex_us.pipelines.stage_runtime import USStageRuntimeWriter
from microplex_us.pipelines.summarize_child_tax_unit_agi_drift import (
    DEFAULT_VARIABLES as DEFAULT_CHILD_TAX_UNIT_AGI_DRIFT_VARIABLES,
)
from microplex_us.pipelines.summarize_child_tax_unit_agi_drift import (
    summarize_child_tax_unit_agi_drift,
)
from microplex_us.pipelines.us import (
    USMicroplexBuildResult,
    USMicroplexPipeline,
)
from microplex_us.pipelines.versioned_artifacts import (
    _allocate_versioned_output_dir,
    _allocate_versioned_output_dir_for_config,
    _finalize_versioned_build_artifacts,
    _initialize_versioned_stage_runtime_writer,
    _json_ready,
    _json_ready_query,
    _provider_query_plan,
    _registry_metric_value,
    _short_config_hash,
    build_and_save_versioned_us_microplex,
    build_and_save_versioned_us_microplex_from_data_dir,
    build_and_save_versioned_us_microplex_from_source_provider,
    build_and_save_versioned_us_microplex_from_source_providers,
    save_versioned_us_microplex_artifacts,
    save_versioned_us_microplex_build_result,
)
from microplex_us.policyengine.harness import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSHarnessSlice,
    evaluate_policyengine_us_harness,
)

__all__ = [
    "USMicroplexArtifactPaths",
    "USMicroplexVersionedBuildArtifacts",
    "_allocate_versioned_output_dir",
    "_allocate_versioned_output_dir_for_config",
    "_finalize_versioned_build_artifacts",
    "_initialize_versioned_stage_runtime_writer",
    "_json_ready",
    "_json_ready_query",
    "_maybe_write_capital_gains_lot_artifact",
    "_provider_query_plan",
    "_registry_metric_value",
    "_short_config_hash",
    "build_and_save_versioned_us_microplex",
    "build_and_save_versioned_us_microplex_from_data_dir",
    "build_and_save_versioned_us_microplex_from_source_provider",
    "build_and_save_versioned_us_microplex_from_source_providers",
    "replay_and_save_versioned_us_microplex_policyengine_stage",
    "replay_us_microplex_policyengine_stage_from_artifact",
    "save_us_microplex_artifacts",
    "save_versioned_us_microplex_artifacts",
    "save_versioned_us_microplex_build_result",
]


@pipeline_node(
    id="us.artifacts.save_us_microplex_artifacts",
    label="Save artifact bundle",
    description="Persist the build result, stage artifacts, diagnostics, manifests, and optional benchmark evidence.",
    artifacts_in=("build_result",),
    artifacts_out=("artifact_manifest", "stage_artifacts", "policyengine_dataset"),
)
def save_us_microplex_artifacts(
    result: USMicroplexBuildResult,
    output_dir: str | Path,
    *,
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
    """Persist a build result as a reproducible artifact bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaffold_seed_data = (
        result.scaffold_seed_data
        if result.scaffold_seed_data is not None
        else result.seed_data
    )
    pre_calibration_policyengine_tables = result.pre_calibration_policyengine_tables

    seed_data_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "05_donor_integration_synthesis",
        "seed_data",
    )
    synthetic_data_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "05_donor_integration_synthesis",
        "synthetic_data",
    )
    calibrated_data_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "07_calibration",
        "calibrated_data",
    )
    targets_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "07_calibration",
        "targets",
    )
    manifest_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "01_run_profile",
        "manifest",
    )
    source_weight_diagnostics_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "05_donor_integration_synthesis",
        "source_weight_diagnostics",
    )
    synthesizer_path = (
        resolve_us_stage_artifact_contract_path(
            output_dir,
            "05_donor_integration_synthesis",
            "synthesizer",
        )
        if result.synthesizer
        else None
    )
    policyengine_dataset_path = (
        resolve_us_stage_artifact_contract_path(
            output_dir,
            "08_dataset_assembly",
            "policyengine_dataset",
        )
        if result.policyengine_tables is not None
        else None
    )
    data_flow_snapshot_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "08_dataset_assembly",
        "data_flow_snapshot",
    )
    stage_manifest_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "08_dataset_assembly",
        "stage_manifest",
    )
    artifact_inventory_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "08_dataset_assembly",
        "artifact_inventory",
    )
    conditional_readiness_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "08_dataset_assembly",
        "conditional_readiness",
    )
    source_plan_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "03_source_planning",
        "source_plan",
    )
    scaffold_seed_data_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "04_seed_scaffold",
        "scaffold_seed_data",
    )
    policyengine_entity_tables_path = (
        resolve_us_stage_artifact_contract_path(
            output_dir,
            "07_calibration",
            "policyengine_entity_tables",
        )
        if result.policyengine_tables is not None
        else None
    )
    calibration_summary_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "07_calibration",
        "calibration_summary",
    )
    pre_calibration_policyengine_entity_tables_path = (
        resolve_us_stage_artifact_contract_path(
            output_dir,
            "06_policyengine_entities",
            "pre_calibration_policyengine_entity_tables",
        )
        if pre_calibration_policyengine_tables is not None
        else None
    )
    validation_evidence_path = (
        resolve_us_stage_artifact_contract_path(
            output_dir,
            "09_validation_benchmarking",
            "validation_evidence",
        )
        if result.policyengine_tables is not None
        else None
    )
    policyengine_harness_path = None
    policyengine_native_scores_path = None
    resolved_run_registry_path = None
    resolved_run_index_path = None
    harness_payload = None
    live_artifacts = stage_runtime_writer is not None

    if stage_runtime_writer is not None:
        stage_runtime_writer.start_stage("08_dataset_assembly")

    try:
        _write_parquet_unless_live_artifact_exists(
            scaffold_seed_data_path,
            scaffold_seed_data,
            live_artifact=live_artifacts,
        )
        _write_parquet_unless_live_artifact_exists(
            seed_data_path,
            result.seed_data,
            live_artifact=live_artifacts,
        )
        _write_parquet_unless_live_artifact_exists(
            synthetic_data_path,
            result.synthetic_data,
            live_artifact=live_artifacts,
        )
        _write_parquet_unless_live_artifact_exists(
            calibrated_data_path,
            result.calibrated_data,
            live_artifact=live_artifacts,
        )
        _write_json_unless_live_artifact_exists(
            targets_path,
            {
                "marginal": result.targets.marginal,
                "continuous": result.targets.continuous,
            },
            live_artifact=live_artifacts,
        )

        if result.synthesizer is not None and synthesizer_path is not None:
            result.synthesizer.save(synthesizer_path)

        if not (live_artifacts and source_plan_path.exists()):
            _write_us_source_plan_artifact(result, source_plan_path)
        if not (live_artifacts and calibration_summary_path.exists()):
            _write_json_atomically(calibration_summary_path, result.calibration_summary)
        source_weight_diagnostics_payload = _build_source_weight_diagnostics(result)
        _write_json_atomically(
            source_weight_diagnostics_path,
            source_weight_diagnostics_payload,
        )

        if (
            pre_calibration_policyengine_entity_tables_path is not None
            and pre_calibration_policyengine_tables is not None
        ):
            if not (
                live_artifacts
                and pre_calibration_policyengine_entity_tables_path.exists()
            ):
                write_us_policyengine_entity_stage_artifact(
                    pre_calibration_policyengine_tables,
                    output_dir,
                    stage_id="06_policyengine_entities",
                    artifact_key="pre_calibration_policyengine_entity_tables",
                    checkpoint_stage="post_microsim",
                )
        if (
            policyengine_entity_tables_path is not None
            and result.policyengine_tables is not None
        ):
            if not (live_artifacts and policyengine_entity_tables_path.exists()):
                write_us_policyengine_entity_stage_artifact(
                    result.policyengine_tables,
                    output_dir,
                    stage_id="07_calibration",
                    artifact_key="policyengine_entity_tables",
                    checkpoint_stage="post_calibration",
                )
        if (
            result.policyengine_tables is not None
            and policyengine_dataset_path is not None
        ):
            period = result.config.policyengine_dataset_year or 2024
            USMicroplexPipeline(result.config).export_policyengine_dataset(
                result,
                policyengine_dataset_path,
                period=period,
            )
        capital_gains_lots_path, capital_gains_lots_summary = (
            _maybe_write_capital_gains_lot_artifact(result, output_dir)
        )

        if stage_runtime_writer is not None:
            stage_runtime_writer.complete_stage(
                USDatasetAssemblyOutputs(
                    policyengine_dataset=(
                        _stage_artifact_ref(
                            output_dir,
                            "08_dataset_assembly",
                            "policyengine_dataset",
                        )
                        if policyengine_dataset_path is not None
                        else None
                    ),
                    stage_manifest=_stage_artifact_ref(
                        output_dir,
                        "08_dataset_assembly",
                        "stage_manifest",
                        assume_exists=True,
                    ),
                    data_flow_snapshot=_stage_artifact_ref(
                        output_dir,
                        "08_dataset_assembly",
                        "data_flow_snapshot",
                        assume_exists=True,
                    ),
                    artifact_inventory=_stage_artifact_ref(
                        output_dir,
                        "08_dataset_assembly",
                        "artifact_inventory",
                        assume_exists=True,
                    ),
                    conditional_readiness=_stage_artifact_ref(
                        output_dir,
                        "08_dataset_assembly",
                        "conditional_readiness",
                        assume_exists=True,
                    ),
                    diagnostics=_stage_diagnostics(
                        "08_dataset_assembly",
                        {
                            "policyengine_dataset": (
                                str(policyengine_dataset_path.relative_to(output_dir))
                                if policyengine_dataset_path is not None
                                else None
                            ),
                            "has_capital_gains_lots": (
                                capital_gains_lots_path is not None
                            ),
                        },
                    ),
                )
            )
    except Exception as exc:
        if stage_runtime_writer is not None:
            stage_runtime_writer.fail_stage("08_dataset_assembly", exc)
        raise

    try:
        if stage_runtime_writer is not None:
            stage_runtime_writer.start_stage("09_validation_benchmarking")

        (
            resolved_target_provider,
            resolved_baseline_dataset,
            resolved_harness_slices,
            resolved_harness_metadata,
        ) = _resolve_policyengine_harness_context(
            result,
            policyengine_comparison_cache=policyengine_comparison_cache,
            policyengine_target_provider=policyengine_target_provider,
            policyengine_baseline_dataset=policyengine_baseline_dataset,
            policyengine_harness_slices=policyengine_harness_slices,
            policyengine_harness_metadata=policyengine_harness_metadata,
        )

        harness_summary = None
        native_scores_payload = (
            dict(precomputed_policyengine_native_scores)
            if precomputed_policyengine_native_scores is not None
            else None
        )
        if precomputed_policyengine_harness_payload is not None:
            harness_payload = dict(precomputed_policyengine_harness_payload)
            policyengine_harness_path = resolve_us_stage_artifact_contract_path(
                output_dir,
                "09_validation_benchmarking",
                "policyengine_harness",
            )
            policyengine_harness_path.write_text(
                json.dumps(harness_payload, indent=2, sort_keys=True)
            )
            harness_summary = harness_payload.get("summary")
        elif (
            not defer_policyengine_harness
            and result.policyengine_tables is not None
            and resolved_target_provider is not None
            and resolved_baseline_dataset is not None
            and resolved_harness_slices
        ):
            harness_period = result.config.policyengine_dataset_year or 2024
            harness_run = evaluate_policyengine_us_harness(
                result.policyengine_tables,
                resolved_target_provider,
                resolved_harness_slices,
                baseline_dataset=str(resolved_baseline_dataset),
                dataset_year=harness_period,
                simulation_cls=result.config.policyengine_simulation_cls,
                candidate_label="microplex",
                baseline_label="policyengine_us_data",
                metadata=resolved_harness_metadata,
                cache=policyengine_comparison_cache,
            )
            policyengine_harness_path = resolve_us_stage_artifact_contract_path(
                output_dir,
                "09_validation_benchmarking",
                "policyengine_harness",
            )
            harness_run.save(policyengine_harness_path)
            harness_payload = harness_run.to_dict()
            harness_summary = harness_payload["summary"]

        if native_scores_payload is not None:
            policyengine_native_scores_path = resolve_us_stage_artifact_contract_path(
                output_dir,
                "09_validation_benchmarking",
                "policyengine_native_scores",
            )
            policyengine_native_scores_path.write_text(
                json.dumps(native_scores_payload, indent=2, sort_keys=True)
            )
        elif (
            not defer_policyengine_native_score
            and policyengine_dataset_path is not None
            and resolved_baseline_dataset is not None
        ):
            try:
                native_scores_payload = compute_us_pe_native_scores(
                    candidate_dataset_path=policyengine_dataset_path,
                    baseline_dataset_path=resolved_baseline_dataset,
                    period=result.config.policyengine_dataset_year or 2024,
                    policyengine_us_data_repo=policyengine_us_data_repo,
                )
                policyengine_native_scores_path = (
                    resolve_us_stage_artifact_contract_path(
                        output_dir,
                        "09_validation_benchmarking",
                        "policyengine_native_scores",
                    )
                )
                policyengine_native_scores_path.write_text(
                    json.dumps(native_scores_payload, indent=2, sort_keys=True)
                )
            except Exception:
                if require_policyengine_native_score:
                    raise

        child_tax_unit_agi_drift_path = None
        child_tax_unit_agi_drift_summary: dict[str, Any] | None = None
        if enable_child_tax_unit_agi_drift:
            try:
                drift_path = resolve_us_stage_artifact_contract_path(
                    output_dir,
                    "09_validation_benchmarking",
                    "child_tax_unit_agi_drift",
                )
                variables = (
                    child_tax_unit_agi_drift_variables
                    or DEFAULT_CHILD_TAX_UNIT_AGI_DRIFT_VARIABLES
                )
                payload = summarize_child_tax_unit_agi_drift(
                    output_dir,
                    variables=variables,
                )
                drift_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
                child_tax_unit_agi_drift_path = drift_path
                child_tax_unit_agi_drift_summary = (
                    _summarize_child_tax_unit_agi_drift_ratios(
                        payload,
                        stage="calibrated",
                        variables=variables,
                    )
                )
            except Exception as exc:  # pragma: no cover - diagnostic best-effort
                child_tax_unit_agi_drift_summary = {
                    "error": f"{type(exc).__name__}: {exc}",
                }

        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "config": result.config.to_dict(),
            "rows": {
                "seed": int(len(result.seed_data)),
                "synthetic": int(len(result.synthetic_data)),
                "calibrated": int(len(result.calibrated_data)),
            },
            "weights": {
                "nonzero": result.n_nonzero_weights,
                "total": result.total_weighted_population,
            },
            "targets": {
                "n_marginal_groups": len(result.targets.marginal),
                "n_continuous": len(result.targets.continuous),
            },
            "synthesis": result.synthesis_metadata,
            "calibration": result.calibration_summary,
            "artifacts": {
                "seed_data": seed_data_path.name,
                "scaffold_seed_data": str(
                    scaffold_seed_data_path.relative_to(output_dir)
                ),
                "synthetic_data": synthetic_data_path.name,
                "calibrated_data": calibrated_data_path.name,
                "targets": targets_path.name,
                "synthesizer": synthesizer_path.name if synthesizer_path else None,
                "source_plan": str(source_plan_path.relative_to(output_dir)),
                "source_weight_diagnostics": source_weight_diagnostics_path.name,
                "calibration_summary": str(
                    calibration_summary_path.relative_to(output_dir)
                ),
                "pre_calibration_policyengine_entity_tables": (
                    str(
                        pre_calibration_policyengine_entity_tables_path.relative_to(
                            output_dir
                        )
                    )
                    if pre_calibration_policyengine_entity_tables_path is not None
                    and pre_calibration_policyengine_entity_tables_path.exists()
                    else None
                ),
                "policyengine_entity_tables": (
                    str(policyengine_entity_tables_path.relative_to(output_dir))
                    if policyengine_entity_tables_path is not None
                    else None
                ),
                "policyengine_dataset": (
                    policyengine_dataset_path.name
                    if policyengine_dataset_path
                    else None
                ),
                "data_flow_snapshot": data_flow_snapshot_path.name,
                "stage_manifest": stage_manifest_path.name,
                "artifact_inventory": str(
                    artifact_inventory_path.relative_to(output_dir)
                ),
                "conditional_readiness": str(
                    conditional_readiness_path.relative_to(output_dir)
                ),
                "validation_evidence": (
                    str(validation_evidence_path.relative_to(output_dir))
                    if validation_evidence_path is not None
                    else None
                ),
                "policyengine_harness": (
                    policyengine_harness_path.name
                    if policyengine_harness_path
                    else None
                ),
                "policyengine_native_scores": (
                    policyengine_native_scores_path.name
                    if policyengine_native_scores_path is not None
                    else None
                ),
                "capital_gains_lots": (
                    capital_gains_lots_path.name
                    if capital_gains_lots_path is not None
                    else None
                ),
            },
        }
        if harness_summary is not None:
            manifest["policyengine_harness"] = harness_summary
        if native_scores_payload is not None:
            manifest["policyengine_native_scores"] = dict(
                native_scores_payload.get("summary", {})
            )
        if child_tax_unit_agi_drift_path is not None:
            manifest["artifacts"]["child_tax_unit_agi_drift"] = (
                child_tax_unit_agi_drift_path.name
            )
        if child_tax_unit_agi_drift_summary is not None:
            manifest.setdefault("diagnostics", {})["child_tax_unit_agi_drift"] = (
                child_tax_unit_agi_drift_summary
            )
        if capital_gains_lots_summary is not None:
            manifest.setdefault("diagnostics", {})["capital_gains_lots"] = (
                capital_gains_lots_summary
            )
        manifest.setdefault("diagnostics", {})["source_weight_diagnostics"] = dict(
            source_weight_diagnostics_payload.get("summary", {})
        )
        if harness_summary is not None or native_scores_payload is not None:
            resolved_run_registry_path = Path(
                run_registry_path or output_dir.parent / "run_registry.jsonl"
            )
            run_entry = build_us_microplex_run_registry_entry(
                artifact_dir=output_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                policyengine_harness_path=policyengine_harness_path,
                policyengine_harness_payload=harness_payload,
                metadata=dict(run_registry_metadata or {}),
            )
            recorded_entry = append_us_microplex_run_registry_entry(
                resolved_run_registry_path,
                run_entry,
            )
            resolved_run_index_path = append_us_microplex_run_index_entry(
                run_index_path or output_dir.parent,
                recorded_entry,
                policyengine_harness_payload=harness_payload,
            )
            manifest["run_registry"] = {
                "path": str(resolved_run_registry_path),
                "artifact_id": recorded_entry.artifact_id,
                "improved_candidate_frontier": recorded_entry.improved_candidate_frontier,
                "improved_delta_frontier": recorded_entry.improved_delta_frontier,
                "improved_composite_frontier": recorded_entry.improved_composite_frontier,
                "improved_native_frontier": recorded_entry.improved_native_frontier,
                "default_frontier_metric": (
                    "enhanced_cps_native_loss_delta"
                    if native_scores_payload is not None
                    else "candidate_composite_parity_loss"
                ),
            }
            manifest["run_index"] = {
                "path": str(resolved_run_index_path),
                "artifact_id": recorded_entry.artifact_id,
            }
        if stage_runtime_writer is not None:
            stage_runtime_writer.manifest_payload = manifest
            stage9_summary = _stage9_benchmark_summary(manifest)
            if stage9_summary:
                if validation_evidence_path is not None:
                    write_us_validation_evidence_manifest(
                        output_dir,
                        validation_evidence_path,
                        manifest_payload=manifest,
                    )
                stage_runtime_writer.complete_stage(
                    USValidationBenchmarkingOutputs(
                        validation_evidence=_stage_artifact_ref(
                            output_dir,
                            "09_validation_benchmarking",
                            "validation_evidence",
                        ),
                        benchmark_summary=stage9_summary,
                        policyengine_harness=(
                            _stage_artifact_ref(
                                output_dir,
                                "09_validation_benchmarking",
                                "policyengine_harness",
                            )
                            if policyengine_harness_path is not None
                            else None
                        ),
                        policyengine_native_scores=(
                            _stage_artifact_ref(
                                output_dir,
                                "09_validation_benchmarking",
                                "policyengine_native_scores",
                            )
                            if policyengine_native_scores_path is not None
                            else None
                        ),
                        diagnostics=_stage_diagnostics(
                            "09_validation_benchmarking",
                            stage9_summary,
                        ),
                    )
                )
            else:
                stage_runtime_writer.defer_stage(
                    "09_validation_benchmarking",
                    "No validation or benchmark evidence was configured for this run.",
                )
            manifest = stage_runtime_writer.finalize_from_artifact_manifest(manifest)
        else:
            manifest = write_us_stage_run_manifests_from_artifact_manifest(
                output_dir,
                manifest,
                allow_stage_input_overrides=allow_stage_input_overrides,
                stage_input_overrides=stage_input_overrides,
            )
    except Exception as exc:
        if stage_runtime_writer is not None:
            stage_runtime_writer.fail_stage("09_validation_benchmarking", exc)
        raise
    assert_valid_benchmark_artifact_manifest(
        manifest,
        artifact_dir=output_dir,
        manifest_path=manifest_path,
        summary_section=(
            "policyengine_harness" if harness_summary is not None else None
        ),
        required_artifact_keys=(
            "scaffold_seed_data",
            "seed_data",
            "synthetic_data",
            "calibrated_data",
            "targets",
            "source_weight_diagnostics",
            *(
                ("policyengine_native_scores",)
                if native_scores_payload is not None
                else ()
            ),
        ),
        required_summary_keys=(
            (
                "candidate_mean_abs_relative_error",
                "baseline_mean_abs_relative_error",
                "mean_abs_relative_error_delta",
            )
            if harness_summary is not None
            else ()
        ),
    )

    return USMicroplexArtifactPaths(
        output_dir=output_dir,
        version_id=output_dir.name,
        seed_data=seed_data_path,
        synthetic_data=synthetic_data_path,
        calibrated_data=calibrated_data_path,
        targets=targets_path,
        manifest=manifest_path,
        scaffold_seed_data=scaffold_seed_data_path,
        synthesizer=synthesizer_path,
        policyengine_dataset=policyengine_dataset_path,
        data_flow_snapshot=data_flow_snapshot_path,
        stage_manifest=stage_manifest_path,
        artifact_inventory=artifact_inventory_path,
        conditional_readiness=conditional_readiness_path,
        source_plan=source_plan_path,
        pre_calibration_policyengine_entity_tables=(
            pre_calibration_policyengine_entity_tables_path
        ),
        policyengine_entity_tables=policyengine_entity_tables_path,
        calibration_summary=calibration_summary_path,
        validation_evidence=validation_evidence_path,
        policyengine_harness=policyengine_harness_path,
        policyengine_native_scores=policyengine_native_scores_path,
        policyengine_native_audit=None,
        policyengine_native_target_diagnostics=None,
        child_tax_unit_agi_drift=child_tax_unit_agi_drift_path,
        capital_gains_lots=capital_gains_lots_path,
        source_weight_diagnostics=source_weight_diagnostics_path,
        run_registry=resolved_run_registry_path,
        run_index_db=resolved_run_index_path,
    )
