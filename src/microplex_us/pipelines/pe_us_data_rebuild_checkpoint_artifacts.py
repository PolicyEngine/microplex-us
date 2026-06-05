"""Saved artifact loading helpers for PE-US-data checkpoint rebuilds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from microplex_us.pipelines.artifacts import (
    USMicroplexArtifactPaths,
    USMicroplexVersionedBuildArtifacts,
)
from microplex_us.pipelines.registry import (
    load_us_microplex_run_registry,
    select_us_microplex_frontier_entry,
)
from microplex_us.pipelines.stage_contracts import (
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.stage_policyengine_artifacts import (
    load_us_policyengine_entity_stage_artifact,
)
from microplex_us.pipelines.stage_run import (
    resolve_us_manifest_or_contract_artifact_path,
)
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexPipeline,
    USMicroplexTargets,
)
from microplex_us.policyengine.us import PolicyEngineUSEntityTableBundle

if TYPE_CHECKING:
    from microplex_us.pipelines.registry import FrontierMetric


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


def _resolve_required_saved_artifact_path(
    artifact_root: Path,
    artifacts: dict[str, Any],
    artifact_key: str,
) -> Path:
    path = _resolve_saved_artifact_path(artifact_root, artifacts.get(artifact_key))
    if path is None:
        raise KeyError(f"Saved artifact manifest does not declare {artifact_key!r}")
    return path


def _resolve_saved_stage_artifact_path(
    artifact_root: Path,
    artifacts: dict[str, Any],
    artifact_key: str,
    *,
    stage_id: str,
) -> Path | None:
    declared_path = _resolve_saved_artifact_path(
        artifact_root, artifacts.get(artifact_key)
    )
    if declared_path is not None:
        return declared_path
    contract_path = resolve_us_stage_artifact_contract_path(
        artifact_root,
        stage_id,
        artifact_key,
    )
    return contract_path if contract_path.exists() else None


def _load_checkpoint_manifest_if_available(artifact_root: Path) -> dict[str, Any]:
    path = artifact_root / "manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_checkpoint_manifest(artifact_root: Path) -> dict[str, Any]:
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Saved artifact manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Saved artifact manifest is not an object: {manifest_path}")
    return payload


def _load_resume_dataframe_artifact(
    artifact_root: Path,
    manifest: dict[str, Any],
    artifact_key: str,
    *,
    stage_id: str,
) -> pd.DataFrame:
    path = resolve_us_manifest_or_contract_artifact_path(
        artifact_root,
        manifest,
        artifact_key,
        stage_id=stage_id,
    )
    if not path.exists():
        raise FileNotFoundError(f"Resume artifact not found: {path}")
    return pd.read_parquet(path)


def _load_resume_json_artifact(
    artifact_root: Path,
    manifest: dict[str, Any],
    artifact_key: str,
    *,
    stage_id: str,
) -> dict[str, Any]:
    path = resolve_us_manifest_or_contract_artifact_path(
        artifact_root,
        manifest,
        artifact_key,
        stage_id=stage_id,
    )
    if not path.exists():
        raise FileNotFoundError(f"Resume artifact not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Resume JSON artifact is not an object: {path}")
    return payload


def _load_resume_targets(
    artifact_root: Path,
    manifest: dict[str, Any],
    *,
    config: USMicroplexBuildConfig,
    seed_data: pd.DataFrame,
) -> USMicroplexTargets:
    path = resolve_us_manifest_or_contract_artifact_path(
        artifact_root,
        manifest,
        "targets",
        stage_id="07_calibration",
    )
    if path.exists():
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Resume targets artifact is not an object: {path}")
        return USMicroplexTargets(
            marginal=dict(payload.get("marginal", {})),
            continuous=dict(payload.get("continuous", {})),
        )
    return USMicroplexPipeline(config).build_targets(seed_data)


def _load_resume_policyengine_tables(
    artifact_root: Path,
    manifest: dict[str, Any],
    artifact_key: str,
    *,
    stage_id: str,
    expected_stage: str | None,
) -> PolicyEngineUSEntityTableBundle:
    path = resolve_us_manifest_or_contract_artifact_path(
        artifact_root,
        manifest,
        artifact_key,
        stage_id=stage_id,
    )
    if not path.exists():
        raise FileNotFoundError(f"Resume PE entity artifact not found: {path}")
    bundle, _metadata = load_us_policyengine_entity_stage_artifact(
        path,
        expected_stage=expected_stage,  # type: ignore[arg-type]
    )
    return bundle


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
        seed_data=_resolve_required_saved_artifact_path(
            artifact_root,
            artifacts,
            "seed_data",
        ),
        synthetic_data=_resolve_required_saved_artifact_path(
            artifact_root,
            artifacts,
            "synthetic_data",
        ),
        calibrated_data=_resolve_required_saved_artifact_path(
            artifact_root,
            artifacts,
            "calibrated_data",
        ),
        targets=_resolve_required_saved_artifact_path(
            artifact_root,
            artifacts,
            "targets",
        ),
        manifest=manifest_path,
        scaffold_seed_data=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "scaffold_seed_data",
            stage_id="04_seed_scaffold",
        ),
        synthesizer=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "synthesizer",
            stage_id="05_donor_integration_synthesis",
        ),
        policyengine_dataset=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "policyengine_dataset",
            stage_id="08_dataset_assembly",
        ),
        data_flow_snapshot=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "data_flow_snapshot",
            stage_id="08_dataset_assembly",
        ),
        stage_manifest=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "stage_manifest",
            stage_id="08_dataset_assembly",
        ),
        artifact_inventory=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "artifact_inventory",
            stage_id="08_dataset_assembly",
        ),
        conditional_readiness=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "conditional_readiness",
            stage_id="08_dataset_assembly",
        ),
        source_plan=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "source_plan",
            stage_id="03_source_planning",
        ),
        policyengine_entity_tables=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "policyengine_entity_tables",
            stage_id="07_calibration",
        ),
        calibration_summary=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "calibration_summary",
            stage_id="07_calibration",
        ),
        validation_evidence=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "validation_evidence",
            stage_id="09_validation_benchmarking",
        ),
        policyengine_harness=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "policyengine_harness",
            stage_id="09_validation_benchmarking",
        ),
        policyengine_native_scores=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "policyengine_native_scores",
            stage_id="09_validation_benchmarking",
        ),
        policyengine_native_audit=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "policyengine_native_audit",
            stage_id="09_validation_benchmarking",
        ),
        policyengine_native_target_diagnostics=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "policyengine_native_target_diagnostics",
            stage_id="09_validation_benchmarking",
        ),
        child_tax_unit_agi_drift=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "child_tax_unit_agi_drift",
            stage_id="09_validation_benchmarking",
        ),
        capital_gains_lots=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "capital_gains_lots",
            stage_id="08_dataset_assembly",
        ),
        source_weight_diagnostics=_resolve_saved_stage_artifact_path(
            artifact_root,
            artifacts,
            "source_weight_diagnostics",
            stage_id="05_donor_integration_synthesis",
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
