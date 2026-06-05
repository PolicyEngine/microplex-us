"""Replay helpers for saved US Microplex artifact bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
from microplex.targets import TargetProvider

from microplex_us.pipelines.artifact_io import (
    _resolve_optional_saved_artifact_file,
    _resolve_saved_artifact_file,
)
from microplex_us.pipelines.artifact_types import USMicroplexVersionedBuildArtifacts
from microplex_us.pipelines.registry import FrontierMetric
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexPipeline,
    USMicroplexTargets,
)
from microplex_us.pipelines.versioned_artifacts import (
    _finalize_versioned_build_artifacts,
)
from microplex_us.policyengine.harness import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSHarnessSlice,
)


def _facade_pipeline_cls() -> type[USMicroplexPipeline]:
    from microplex_us.pipelines import artifacts

    return artifacts.USMicroplexPipeline


def _infer_baseline_household_weight_sum(
    baseline_dataset: str | Path,
    *,
    target_period: int,
) -> float | None:
    """Best-effort household-weight total inferred from a PE baseline H5."""

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


def _refresh_baseline_derived_weight_targets(
    config_payload: dict[str, Any],
    *,
    explicit_override_keys: set[str],
) -> None:
    """Refresh derived total-weight knobs after a replay baseline override."""

    if "policyengine_baseline_dataset" not in explicit_override_keys:
        return
    baseline_dataset = config_payload.get("policyengine_baseline_dataset")
    if baseline_dataset in (None, ""):
        return
    target_period = int(
        config_payload.get("policyengine_target_period")
        or config_payload.get("policyengine_dataset_year")
        or 2024
    )
    baseline_weight_sum = _infer_baseline_household_weight_sum(
        baseline_dataset,
        target_period=target_period,
    )
    if baseline_weight_sum is None:
        return

    if "policyengine_selection_target_total_weight" not in explicit_override_keys:
        config_payload["policyengine_selection_target_total_weight"] = (
            baseline_weight_sum
        )
    if config_payload.get("calibration_backend") == "none":
        return
    if config_payload.get("policyengine_calibration_rescale_to_input_weight_sum"):
        if (
            "policyengine_calibration_rescale_to_target_total_weight"
            not in explicit_override_keys
        ):
            config_payload["policyengine_calibration_rescale_to_target_total_weight"] = (
                False
            )
        if "policyengine_calibration_target_total_weight" not in explicit_override_keys:
            config_payload["policyengine_calibration_target_total_weight"] = None
        return
    if "policyengine_calibration_target_total_weight" not in explicit_override_keys:
        config_payload["policyengine_calibration_target_total_weight"] = (
            baseline_weight_sum
        )
    if (
        "policyengine_calibration_rescale_to_target_total_weight"
        not in explicit_override_keys
    ):
        config_payload["policyengine_calibration_rescale_to_target_total_weight"] = True


def replay_us_microplex_policyengine_stage_from_artifact(
    artifact_dir: str | Path,
    *,
    config_overrides: dict[str, Any] | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
) -> USMicroplexBuildResult:
    """Replay calibration/export inputs from a saved artifact without raw ETL.

    This reloads saved seed and synthetic rows, applies optional runtime config
    overrides, and reruns the downstream calibration stage from the saved
    synthetic population. For PE-DB builds, this intentionally calls
    ``calibrate_policyengine_tables`` even when ``calibration_backend="none"``
    so PE target materialization and export-only variables stay on the same
    path as a full pipeline build.
    """

    artifact_root = Path(artifact_dir).expanduser().resolve()
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Saved artifact manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    config_payload = dict(manifest.get("config", {}))
    resolved_config_overrides = dict(config_overrides or {})
    if (
        policyengine_baseline_dataset is not None
        and "policyengine_baseline_dataset" not in resolved_config_overrides
    ):
        resolved_config_overrides["policyengine_baseline_dataset"] = str(
            policyengine_baseline_dataset
        )
    explicit_override_keys = set(resolved_config_overrides)
    config_payload.update(resolved_config_overrides)
    _refresh_baseline_derived_weight_targets(
        config_payload,
        explicit_override_keys=explicit_override_keys,
    )
    config = USMicroplexBuildConfig(**config_payload)

    seed_data = pd.read_parquet(
        _resolve_saved_artifact_file(artifact_root, manifest, "seed_data")
    )
    scaffold_seed_data_path = _resolve_optional_saved_artifact_file(
        artifact_root,
        manifest,
        "scaffold_seed_data",
    )
    scaffold_seed_data = (
        pd.read_parquet(scaffold_seed_data_path)
        if scaffold_seed_data_path is not None
        else None
    )
    synthetic_data = pd.read_parquet(
        _resolve_saved_artifact_file(artifact_root, manifest, "synthetic_data")
    )
    targets_payload = json.loads(
        _resolve_saved_artifact_file(artifact_root, manifest, "targets").read_text()
    )
    targets = USMicroplexTargets(
        marginal=dict(targets_payload.get("marginal", {})),
        continuous=dict(targets_payload.get("continuous", {})),
    )

    pipeline = _facade_pipeline_cls()(config)
    pre_calibration_policyengine_tables = pipeline.build_policyengine_entity_tables(
        synthetic_data
    )
    if config.policyengine_targets_db is not None:
        policyengine_tables, calibrated_data, calibration_summary = (
            pipeline.calibrate_policyengine_tables(pre_calibration_policyengine_tables)
        )
    else:
        calibrated_data, calibration_summary = pipeline.calibrate(
            synthetic_data,
            targets,
        )
        policyengine_tables = pipeline.build_policyengine_entity_tables(calibrated_data)

    synthesis_metadata = dict(manifest.get("synthesis", {}))
    synthesis_metadata["policyengine_stage_replay"] = {
        "source_artifact_dir": str(artifact_root),
        "source_manifest": str(manifest_path),
        "config_override_keys": sorted((config_overrides or {}).keys()),
    }

    return USMicroplexBuildResult(
        config=config,
        seed_data=seed_data,
        synthetic_data=synthetic_data,
        calibrated_data=calibrated_data,
        targets=targets,
        calibration_summary=calibration_summary,
        synthesis_metadata=synthesis_metadata,
        policyengine_tables=policyengine_tables,
        pre_calibration_policyengine_tables=pre_calibration_policyengine_tables,
        scaffold_seed_data=scaffold_seed_data,
    )


def replay_and_save_versioned_us_microplex_policyengine_stage(
    artifact_dir: str | Path,
    output_root: str | Path | None = None,
    *,
    config_overrides: dict[str, Any] | None = None,
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
    defer_policyengine_harness: bool = True,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Replay a saved artifact's policy stage and persist a new versioned bundle."""

    artifact_root = Path(artifact_dir).expanduser().resolve()
    build_result = replay_us_microplex_policyengine_stage_from_artifact(
        artifact_root,
        config_overrides=config_overrides,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
    )
    resolved_output_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else artifact_root.parent
    )
    replay_metadata = {
        "policyengine_stage_replay": True,
        "source_artifact_dir": str(artifact_root),
        **dict(run_registry_metadata or {}),
    }
    return _finalize_versioned_build_artifacts(
        build_result,
        output_root=resolved_output_root,
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
        run_registry_metadata=replay_metadata,
    )
