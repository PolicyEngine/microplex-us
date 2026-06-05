"""Evidence attachment for PE-US-data checkpoint rebuilds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from microplex.targets import assert_valid_benchmark_artifact_manifest

from microplex_us.pipelines.index_db import append_us_microplex_run_index_entry
from microplex_us.pipelines.pe_us_data_rebuild import (
    PEUSDataRebuildProgram,
    default_policyengine_us_data_rebuild_program,
)
from microplex_us.pipelines.pe_us_data_rebuild_audit import (
    build_policyengine_us_data_rebuild_native_audit,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_ablation import (
    _build_checkpoint_imputation_ablation_payload,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_common import (
    _resolve_policyengine_us_runtime_version,
    _write_json_atomically,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_types import (
    PEUSDataRebuildCheckpointEvidenceResult,
)
from microplex_us.pipelines.pe_us_data_rebuild_parity import (
    build_policyengine_us_data_rebuild_parity_artifact,
    write_policyengine_us_data_rebuild_parity_artifact,
)
from microplex_us.pipelines.registry import (
    append_us_microplex_run_registry_entry,
    build_us_microplex_run_registry_entry,
    load_us_microplex_run_registry,
)
from microplex_us.pipelines.stage_contracts import (
    canonicalize_us_pipeline_stage_id,
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.stage_metrics import stage_metrics
from microplex_us.pipelines.stage_run import (
    write_us_stage_run_manifests_from_artifact_manifest,
)

if TYPE_CHECKING:
    from microplex.targets import TargetProvider

    from microplex_us.pipelines.registry import FrontierMetric
    from microplex_us.policyengine.harness import (
        PolicyEngineUSComparisonCache,
        PolicyEngineUSHarnessSlice,
    )


def _refresh_checkpoint_data_flow_snapshot(
    artifact_root: Path,
    manifest: dict[str, Any],
    *,
    extra_outputs: tuple[str, ...] = (),
) -> Path | None:
    if extra_outputs:
        manifest.setdefault("diagnostics", {}).setdefault(
            "checkpoint_extra_outputs",
            list(extra_outputs),
        )
    try:
        updated_manifest = write_us_stage_run_manifests_from_artifact_manifest(
            artifact_root,
            manifest,
        )
    except ValueError as exc:
        manifest.setdefault("diagnostics", {})["checkpoint_stage_refresh_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return _patch_checkpoint_data_flow_snapshot_outputs(
            artifact_root,
            manifest=manifest,
            extra_outputs=extra_outputs,
        )
    manifest.clear()
    manifest.update(updated_manifest)
    snapshot_path = resolve_us_stage_artifact_contract_path(
        artifact_root,
        "08_dataset_assembly",
        "data_flow_snapshot",
    )
    if extra_outputs:
        return _patch_checkpoint_data_flow_snapshot_outputs(
            artifact_root,
            manifest=manifest,
            extra_outputs=extra_outputs,
        )
    return snapshot_path if snapshot_path.exists() else None


def _patch_checkpoint_data_flow_snapshot_outputs(
    artifact_root: Path,
    *,
    manifest: dict[str, Any],
    extra_outputs: tuple[str, ...],
) -> Path | None:
    snapshot_path = resolve_us_stage_artifact_contract_path(
        artifact_root,
        "08_dataset_assembly",
        "data_flow_snapshot",
    )
    if not snapshot_path.exists():
        return None
    snapshot = json.loads(snapshot_path.read_text())
    stages = snapshot.get("stages")
    if not isinstance(stages, list):
        return snapshot_path
    validation_stage = None
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        stage_id = str(stage.get("id", ""))
        if canonicalize_us_pipeline_stage_id(stage_id) == "09_validation_benchmarking":
            validation_stage = stage
            stage["id"] = "09_validation_benchmarking"
            break
    if validation_stage is None:
        validation_stage = {
            "id": "09_validation_benchmarking",
            "outputs": [],
            "metrics": [],
            "status": "ready",
        }
        stages.append(validation_stage)
    existing_outputs = list(validation_stage.get("outputs") or ())
    validation_stage["outputs"] = list(
        dict.fromkeys(
            [
                *existing_outputs,
                *_checkpoint_validation_output_names(manifest),
                *extra_outputs,
            ]
        )
    )
    if not validation_stage.get("metrics"):
        validation_stage["metrics"] = stage_metrics(
            "09_validation_benchmarking",
            manifest=manifest,
        )
    if extra_outputs:
        validation_stage["status"] = "ready"
    else:
        validation_stage.setdefault("status", "ready")
    _write_json_atomically(snapshot_path, snapshot)
    return snapshot_path


def _checkpoint_validation_output_names(manifest: dict[str, Any]) -> tuple[str, ...]:
    artifacts = dict(manifest.get("artifacts", {}))
    ordered_keys = (
        "policyengine_harness",
        "policyengine_native_scores",
        "imputation_ablation",
        "policyengine_native_audit",
        "policyengine_native_target_diagnostics",
        "child_tax_unit_agi_drift",
    )
    return tuple(
        str(artifacts[key])
        for key in ordered_keys
        if isinstance(artifacts.get(key), str)
    )


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

    from microplex_us.pipelines.pe_native_scores import (
        build_us_pe_native_target_diagnostics_payload,
        compute_us_pe_native_scores,
    )
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
        harness_path = resolve_us_stage_artifact_contract_path(
            artifact_root,
            "09_validation_benchmarking",
            "policyengine_harness",
        )
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
        native_scores_path = resolve_us_stage_artifact_contract_path(
            artifact_root,
            "09_validation_benchmarking",
            "policyengine_native_scores",
        )
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
        imputation_ablation_path = resolve_us_stage_artifact_contract_path(
            artifact_root,
            "09_validation_benchmarking",
            "imputation_ablation",
        )
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
    native_target_diagnostics_path: Path | None = None
    native_target_diagnostics_payload: dict[str, Any] | None = None
    if compute_native_audit and artifacts.get("policyengine_native_scores") is not None:
        native_audit_payload = build_policyengine_us_data_rebuild_native_audit(
            artifact_root,
            manifest_payload=manifest,
            native_scores_payload=native_scores_payload,
            imputation_ablation_payload=imputation_ablation_payload,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
        native_audit_path = resolve_us_stage_artifact_contract_path(
            artifact_root,
            "09_validation_benchmarking",
            "policyengine_native_audit",
        )
        _write_json_atomically(native_audit_path, native_audit_payload)
        artifacts["policyengine_native_audit"] = native_audit_path.name
        manifest["policyengine_native_audit"] = dict(
            native_audit_payload.get("verdictHints", {})
        )
        target_delta_payload = native_audit_payload.get("targetDelta")
        if isinstance(target_delta_payload, dict):
            native_target_diagnostics_payload = (
                build_us_pe_native_target_diagnostics_payload(
                    period=(
                        config.get("policyengine_dataset_year")
                        or config.get("policyengine_target_period")
                        or 2024
                    ),
                    from_label="policyengine-us-data",
                    to_label="microplex-us",
                    policyengine_us_data_repo=policyengine_us_data_repo,
                    policyengine_us_data_python=policyengine_us_data_python,
                    policyengine_targets_db_path=config.get("policyengine_targets_db"),
                    target_delta_payload=target_delta_payload,
                    artifact_id=str(
                        native_audit_payload.get("artifactId") or artifact_root.name
                    ),
                    run_id=str(
                        native_audit_payload.get("artifactId") or artifact_root.name
                    ),
                )
            )
            native_target_diagnostics_path = resolve_us_stage_artifact_contract_path(
                artifact_root,
                "09_validation_benchmarking",
                "policyengine_native_target_diagnostics",
            )
            _write_json_atomically(
                native_target_diagnostics_path,
                native_target_diagnostics_payload,
            )
            artifacts["policyengine_native_target_diagnostics"] = (
                native_target_diagnostics_path.name
            )
    manifest["artifacts"] = artifacts
    _refresh_checkpoint_data_flow_snapshot(
        artifact_root,
        manifest,
        extra_outputs=tuple(
            path.name
            for path in (
                native_audit_path,
                native_target_diagnostics_path,
            )
            if path is not None
        ),
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
        native_target_diagnostics_path=native_target_diagnostics_path,
        native_target_diagnostics_payload=native_target_diagnostics_payload,
        imputation_ablation_path=imputation_ablation_path,
        imputation_ablation_payload=imputation_ablation_payload,
    )
