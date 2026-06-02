"""Saved-run status classification for US pipeline stage manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microplex_us.pipelines.stage_manifest_types import (
    USStageArtifactRecord,
    USStageStatus,
)


def stage_status(
    stage_id: str,
    *,
    artifact_root: Path,
    manifest: dict[str, Any],
    artifacts: list[USStageArtifactRecord],
    assume_existing_artifact_keys: set[str],
) -> USStageStatus:
    """Return the saved-run status for one canonical stage."""

    artifact_map = dict(manifest.get("artifacts", {}))
    synthesis = dict(manifest.get("synthesis", {}))
    calibration = dict(manifest.get("calibration", {}))
    rows = dict(manifest.get("rows", {}))
    if stage_id == "01_run_profile":
        if artifact_missing(artifacts, required_only=True):
            return "incomplete"
        if artifact_exists(artifacts, "manifest"):
            return "ready"
        return "metadata_only" if manifest.get("config") else "missing"
    if stage_id == "02_source_loading":
        return "metadata_only" if synthesis.get("source_names") else "missing"
    if stage_id == "03_source_planning":
        if artifact_missing(artifacts):
            return "incomplete"
        if artifact_exists(artifacts, "source_plan"):
            return "ready"
        return "metadata_only" if synthesis.get("scaffold_source") else "missing"
    if stage_id == "04_seed_scaffold":
        if artifact_missing(artifacts, required_only=True):
            return "incomplete"
        if required_artifacts_exist(artifacts):
            return "ready"
        return (
            "metadata_only"
            if rows.get("seed") or synthesis.get("scaffold_source")
            else "missing"
        )
    if stage_id == "05_donor_integration_synthesis":
        if artifact_missing(artifacts, required_only=True):
            return "incomplete"
        if required_artifacts_exist(artifacts):
            return "ready"
        return (
            "metadata_only" if rows.get("seed") or rows.get("synthetic") else "missing"
        )
    if stage_id == "06_policyengine_entities":
        if artifact_missing(artifacts):
            return "incomplete"
        if artifact_exists(artifacts, "pre_calibration_policyengine_entity_tables"):
            return "ready"
        if manifest_artifact_exists(
            manifest,
            artifact_root,
            "policyengine_dataset",
            assume_existing_artifact_keys=assume_existing_artifact_keys,
        ):
            return "metadata_only"
        return "missing"
    if stage_id == "07_calibration":
        if artifact_missing(artifacts, required_only=True):
            return "incomplete"
        if calibration and required_artifacts_exist(artifacts):
            return "ready"
        return "metadata_only" if calibration and rows.get("calibrated") else "missing"
    if stage_id == "08_dataset_assembly":
        if artifact_missing(artifacts, required_only=True):
            return "incomplete"
        if manifest_artifact_exists(
            manifest,
            artifact_root,
            "policyengine_dataset",
            assume_existing_artifact_keys=assume_existing_artifact_keys,
        ):
            return "ready"
        return "metadata_only" if artifact_map.get("stage_manifest") else "missing"
    if stage_id == "09_validation_benchmarking":
        evidence_keys = (
            "policyengine_harness",
            "policyengine_native_scores",
            "policyengine_native_audit",
            "imputation_ablation",
        )
        evidence_index_keys = ("validation_evidence",)
        if manifest_artifact_missing(
            manifest,
            artifact_root,
            (*evidence_keys, *evidence_index_keys),
            assume_existing_artifact_keys=assume_existing_artifact_keys,
        ):
            return "incomplete"
        has_evidence = any(
            manifest_artifact_exists(
                manifest,
                artifact_root,
                key,
                assume_existing_artifact_keys=assume_existing_artifact_keys,
            )
            for key in evidence_keys
        )
        if not has_evidence:
            has_evidence = validation_evidence_index_has_existing_evidence(
                manifest,
                artifact_root,
                assume_existing_artifact_keys=assume_existing_artifact_keys,
            )
        if has_evidence:
            if not manifest_artifact_exists(
                manifest,
                artifact_root,
                "validation_evidence",
                assume_existing_artifact_keys=assume_existing_artifact_keys,
            ):
                return "incomplete"
            return "ready"
        if manifest_artifact_exists(
            manifest,
            artifact_root,
            "policyengine_dataset",
            assume_existing_artifact_keys=assume_existing_artifact_keys,
        ):
            return "deferred"
        return "missing"
    if any(artifact.get("exists") for artifact in artifacts):
        return "ready"
    return "missing"


def required_artifacts_exist(artifacts: list[USStageArtifactRecord]) -> bool:
    """Return whether all required artifacts exist."""

    required = [artifact for artifact in artifacts if bool(artifact.get("required"))]
    return bool(required) and all(bool(artifact.get("exists")) for artifact in required)


def artifact_exists(artifacts: list[USStageArtifactRecord], key: str) -> bool:
    """Return whether a stage artifact record exists."""

    return any(
        artifact.get("key") == key and bool(artifact.get("exists"))
        for artifact in artifacts
    )


def artifact_missing(
    artifacts: list[USStageArtifactRecord],
    *,
    required_only: bool = False,
) -> bool:
    """Return whether required or referenced stage artifacts are missing."""

    return any(
        not bool(artifact.get("exists"))
        and (
            bool(artifact.get("required"))
            or (not required_only and bool(artifact.get("referenced")))
        )
        for artifact in artifacts
    )


def manifest_artifact_exists(
    manifest: dict[str, Any],
    artifact_root: Path,
    artifact_key: str,
    *,
    assume_existing_artifact_keys: set[str],
) -> bool:
    """Return whether a top-level manifest artifact exists."""

    path = manifest_artifact_path(manifest, artifact_root, artifact_key)
    if path is None:
        return False
    if artifact_key in assume_existing_artifact_keys:
        return True
    return path.exists()


def manifest_artifact_missing(
    manifest: dict[str, Any],
    artifact_root: Path,
    artifact_keys: tuple[str, ...],
    *,
    assume_existing_artifact_keys: set[str],
) -> bool:
    """Return whether any referenced top-level manifest artifact is missing."""

    artifacts = dict(manifest.get("artifacts", {}))
    return any(
        bool(artifacts.get(key))
        and not manifest_artifact_exists(
            manifest,
            artifact_root,
            key,
            assume_existing_artifact_keys=assume_existing_artifact_keys,
        )
        for key in artifact_keys
    )


def validation_evidence_index_has_existing_evidence(
    manifest: dict[str, Any],
    artifact_root: Path,
    *,
    assume_existing_artifact_keys: set[str],
) -> bool:
    """Return whether a validation evidence index points to existing evidence."""

    path = manifest_artifact_path(manifest, artifact_root, "validation_evidence")
    if path is None:
        return False
    if "validation_evidence" in assume_existing_artifact_keys and not path.exists():
        return False
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return False
    for record in evidence:
        if not isinstance(record, dict) or not record.get("path"):
            continue
        evidence_path = Path(str(record["path"]))
        if not evidence_path.is_absolute():
            evidence_path = artifact_root / evidence_path
        if evidence_path.exists():
            return True
    return False


def manifest_artifact_path(
    manifest: dict[str, Any],
    artifact_root: Path,
    artifact_key: str,
) -> Path | None:
    """Return the resolved path for a top-level manifest artifact."""

    artifacts = dict(manifest.get("artifacts", {}))
    filename = artifacts.get(artifact_key)
    if not filename:
        return None
    path = Path(str(filename))
    if not path.is_absolute():
        path = artifact_root / path
    return path


__all__ = [
    "artifact_exists",
    "artifact_missing",
    "manifest_artifact_exists",
    "manifest_artifact_missing",
    "manifest_artifact_path",
    "required_artifacts_exist",
    "stage_status",
    "validation_evidence_index_has_existing_evidence",
]
