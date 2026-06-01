"""Artifact-backed parity summaries for the PE-US-data rebuild track."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microplex_us.pipelines.pe_us_data_rebuild import (
    PEUSDataRebuildProgram,
    default_policyengine_us_data_rebuild_config,
    default_policyengine_us_data_rebuild_program,
)
from microplex_us.pipelines.stage_run import (
    resolve_us_manifest_or_contract_artifact_path,
)

_HARNESS_SUMMARY_KEYS = (
    "candidate_mean_abs_relative_error",
    "baseline_mean_abs_relative_error",
    "mean_abs_relative_error_delta",
    "candidate_composite_parity_loss",
    "baseline_composite_parity_loss",
    "composite_parity_loss_delta",
    "slice_win_rate",
    "target_win_rate",
    "supported_target_rate",
    "baseline_supported_target_rate",
    "tag_summaries",
)

_NATIVE_SUMMARY_KEYS = (
    "candidate_enhanced_cps_native_loss",
    "baseline_enhanced_cps_native_loss",
    "enhanced_cps_native_loss_delta",
    "candidate_beats_baseline",
    "candidate_unweighted_msre",
    "baseline_unweighted_msre",
    "unweighted_msre_delta",
    "n_targets_total",
    "n_targets_kept",
    "n_targets_zero_dropped",
    "n_targets_bad_dropped",
    "n_national_targets",
    "n_state_targets",
)

_IMPUTATION_SUMMARY_KEYS = (
    "source_count",
    "skipped_source_count",
    "target_count",
    "production_variant",
    "production_mean_weighted_mae",
    "production_mean_support_f1",
    "best_mean_weighted_mae_variant",
    "best_mean_support_f1_variant",
    "variant_scorecard",
)

_PROFILE_CONTEXT_KEYS = {
    "cps_asec_cache_dir",
    "policyengine_baseline_dataset",
    "policyengine_dataset",
    "policyengine_targets_db",
}
_POLICYENGINE_HARNESS_BASELINE_LABEL = "policyengine_us_data"
_POLICYENGINE_NATIVE_METRIC = "enhanced_cps_native_loss"


def build_policyengine_us_data_rebuild_parity_artifact(
    artifact_dir: str | Path,
    *,
    program: PEUSDataRebuildProgram | None = None,
    manifest_payload: dict[str, Any] | None = None,
    harness_payload: dict[str, Any] | None = None,
    native_scores_payload: dict[str, Any] | None = None,
    imputation_ablation_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact rebuild-parity sidecar from one saved artifact bundle."""

    artifact_root = Path(artifact_dir)
    manifest_source = _resolve_payload_source(
        artifact_root / "manifest.json",
        override_supplied=manifest_payload is not None,
    )
    manifest = (
        dict(manifest_payload)
        if manifest_payload is not None
        else json.loads((artifact_root / "manifest.json").read_text())
    )
    harness_path = _resolve_stage_artifact_path(
        artifact_root,
        manifest,
        "policyengine_harness",
        stage_id="09_validation_benchmarking",
    )
    harness_source = _resolve_payload_source(
        harness_path,
        override_supplied=harness_payload is not None,
    )
    harness = (
        dict(harness_payload)
        if harness_payload is not None
        else _load_optional_json(harness_path)
    )
    native_scores_path = _resolve_stage_artifact_path(
        artifact_root,
        manifest,
        "policyengine_native_scores",
        stage_id="09_validation_benchmarking",
    )
    native_scores_source = _resolve_payload_source(
        native_scores_path,
        override_supplied=native_scores_payload is not None,
    )
    native_scores = (
        dict(native_scores_payload)
        if native_scores_payload is not None
        else _load_optional_json(native_scores_path)
    )
    imputation_ablation_path = _resolve_stage_artifact_path(
        artifact_root,
        manifest,
        "imputation_ablation",
        stage_id="09_validation_benchmarking",
    )
    imputation_ablation_source = _resolve_payload_source(
        imputation_ablation_path,
        override_supplied=imputation_ablation_payload is not None,
    )
    imputation_ablation = (
        dict(imputation_ablation_payload)
        if imputation_ablation_payload is not None
        else _load_optional_json(imputation_ablation_path)
    )

    resolved_program = program or default_policyengine_us_data_rebuild_program()
    config = _normalize_observed_config(dict(manifest.get("config", {})))
    default_config = default_policyengine_us_data_rebuild_config().to_dict()
    harness_summary = dict(harness.get("summary", {})) if harness is not None else {}
    native_summary = (
        dict(native_scores.get("summary", {})) if native_scores is not None else {}
    )
    imputation_summary = (
        dict(imputation_ablation.get("summary", {}))
        if imputation_ablation is not None
        else {}
    )
    baseline_dataset_path = config.get("policyengine_baseline_dataset")
    harness_is_pe_comparison = bool(
        baseline_dataset_path
        and harness is not None
        and harness.get("baseline_label") == _POLICYENGINE_HARNESS_BASELINE_LABEL
    )
    native_is_pe_comparison = bool(
        baseline_dataset_path
        and native_scores is not None
        and native_scores.get("metric") == _POLICYENGINE_NATIVE_METRIC
    )

    return {
        "schemaVersion": 1,
        "artifactId": artifact_root.name,
        "artifactDir": str(artifact_root.resolve()),
        "evidence": {
            "manifest": manifest_source,
            "policyengineHarness": harness_source,
            "policyengineNativeScores": native_scores_source,
            "imputationAblation": imputation_ablation_source,
        },
        "program": {
            "programId": resolved_program.program_id,
            "title": resolved_program.title,
            "stageStatuses": {
                stage.stage_id: stage.current_status.value for stage in resolved_program.stages
            },
        },
        "profileConformance": _build_profile_conformance(
            observed_config=config,
            expected_config=default_config,
        ),
        "baselineSlice": {
            "baselineDatasetPath": config.get("policyengine_baseline_dataset"),
            "targetsDbPath": config.get("policyengine_targets_db"),
            "datasetYear": config.get("policyengine_dataset_year"),
            "targetPeriod": config.get("policyengine_target_period"),
            "targetProfile": config.get("policyengine_target_profile"),
            "calibrationTargetProfile": config.get("policyengine_calibration_target_profile"),
            "candidateLabel": harness.get("candidate_label") if harness is not None else None,
            "baselineLabel": harness.get("baseline_label") if harness is not None else None,
            "comparisonMetadata": dict(harness.get("metadata", {})) if harness is not None else {},
        },
        "comparison": {
            "policyengineHarness": (
                {
                    "available": True,
                    "isPolicyEngineComparison": harness_is_pe_comparison,
                    "period": harness.get("period"),
                    **{
                        key: harness_summary.get(key)
                        for key in _HARNESS_SUMMARY_KEYS
                    },
                }
                if harness is not None
                else {"available": False}
            ),
            "policyengineNativeScores": (
                {
                    "available": True,
                    "isPolicyEngineComparison": native_is_pe_comparison,
                    "metric": native_scores.get("metric"),
                    "period": native_scores.get("period"),
                    **{
                        key: native_summary.get(key)
                        for key in _NATIVE_SUMMARY_KEYS
                    },
                }
                if native_scores is not None
                else {"available": False}
            ),
            "imputationAblation": (
                {
                    "available": True,
                    **{
                        key: imputation_summary.get(key)
                        for key in _IMPUTATION_SUMMARY_KEYS
                    },
                }
                if imputation_ablation is not None
                else {"available": False}
            ),
        },
        "verdict": {
            "candidateBeatsHarnessMeanAbsRelativeError": (
                _delta_is_better(harness_summary.get("mean_abs_relative_error_delta"))
                if harness_is_pe_comparison
                else None
            ),
            "candidateBeatsHarnessCompositeParityLoss": (
                _delta_is_better(harness_summary.get("composite_parity_loss_delta"))
                if harness_is_pe_comparison
                else None
            ),
            "candidateBeatsNativeBroadLoss": (
                native_summary.get("candidate_beats_baseline")
                if native_is_pe_comparison
                else None
            ),
            "productionImputationVariantIsMaeWinner": (
                imputation_summary.get("production_variant")
                == imputation_summary.get("best_mean_weighted_mae_variant")
                if imputation_ablation is not None
                else None
            ),
            "productionImputationVariantIsSupportWinner": (
                imputation_summary.get("production_variant")
                == imputation_summary.get("best_mean_support_f1_variant")
                if imputation_ablation is not None
                else None
            ),
            "hasRealPolicyEngineComparison": harness_is_pe_comparison
            or native_is_pe_comparison,
            "hasImputationAblation": imputation_ablation is not None,
        },
    }


def write_policyengine_us_data_rebuild_parity_artifact(
    artifact_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    program: PEUSDataRebuildProgram | None = None,
    manifest_payload: dict[str, Any] | None = None,
    harness_payload: dict[str, Any] | None = None,
    native_scores_payload: dict[str, Any] | None = None,
    imputation_ablation_payload: dict[str, Any] | None = None,
) -> Path:
    """Write the PE rebuild parity sidecar for one saved artifact bundle."""

    artifact_root = Path(artifact_dir)
    destination = (
        Path(output_path)
        if output_path is not None
        else artifact_root / "pe_us_data_rebuild_parity.json"
    )
    payload = build_policyengine_us_data_rebuild_parity_artifact(
        artifact_root,
        program=program,
        manifest_payload=manifest_payload,
        harness_payload=harness_payload,
        native_scores_payload=native_scores_payload,
        imputation_ablation_payload=imputation_ablation_payload,
    )
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resolve_stage_artifact_path(
    artifact_root: Path,
    manifest: dict[str, Any],
    artifact_key: str,
    *,
    stage_id: str,
) -> Path:
    return resolve_us_manifest_or_contract_artifact_path(
        artifact_root,
        manifest,
        artifact_key,
        stage_id=stage_id,
    )


def _build_profile_conformance(
    *,
    observed_config: dict[str, Any],
    expected_config: dict[str, Any],
) -> dict[str, Any]:
    differing_keys = []
    matching_keys = []
    observed_only_keys = []
    for key in sorted(set(expected_config) | set(observed_config)):
        if key in _PROFILE_CONTEXT_KEYS:
            continue
        if key not in expected_config:
            observed_only_keys.append(
                {
                    "key": key,
                    "expected": None,
                    "observed": observed_config.get(key),
                }
            )
            continue
        observed = observed_config.get(key)
        expected = expected_config.get(key)
        if observed == expected:
            matching_keys.append(key)
        else:
            differing_keys.append(
                {
                    "key": key,
                    "expected": expected,
                    "observed": observed,
                }
            )
    differing_keys.extend(observed_only_keys)
    return {
        "exactMatch": not differing_keys,
        "matchingKeyCount": len(matching_keys),
        "differingKeyCount": len(differing_keys),
        "differingKeys": differing_keys,
    }


def _delta_is_better(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return float(value) < 0.0
    except (TypeError, ValueError):
        return None


def _normalize_observed_config(observed_config: dict[str, Any]) -> dict[str, Any]:
    from microplex_us.pipelines.us import USMicroplexBuildConfig

    normalized = USMicroplexBuildConfig().to_dict()
    normalized.update(observed_config)
    return normalized


def _resolve_payload_source(path: Path, *, override_supplied: bool) -> dict[str, Any]:
    return {
        "source": "in_memory_override" if override_supplied else "artifact_bundle",
        "file": path.name,
        "exists": path.exists(),
    }
