"""Build aggregate saved-run stage manifests for US pipeline artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from microplex_us.pipelines.stage_contracts import (
    US_STAGE_CONTRACT_VERSION,
    USPipelineStageContract,
    USStageArtifactContract,
    USStageResourceContract,
    default_us_pipeline_stage_contracts,
)
from microplex_us.pipelines.stage_manifest_types import (
    US_STAGE_MANIFEST_SCHEMA_VERSION,
    USStageArtifactRecord,
    USStageLifecycleStatus,
    USStageManifest,
    USStageRecord,
    USStageResourceRecord,
    USStageValidationRecord,
)
from microplex_us.pipelines.stage_metrics import stage_metrics
from microplex_us.pipelines.stage_status import stage_status


def build_us_stage_manifest(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any],
    assume_existing_artifact_keys: Iterable[str] = (),
) -> USStageManifest:
    """Build the canonical stage manifest from a saved artifact manifest."""

    artifact_root = Path(artifact_dir)
    manifest = dict(manifest_payload)
    artifact_map = dict(manifest.get("artifacts", {}))
    assumed_existing = set(assume_existing_artifact_keys)
    stage_output_manifests = _load_stage_output_manifests(
        artifact_root,
        manifest,
    )
    stages = [
        _stage_record(
            contract,
            artifact_root=artifact_root,
            manifest=manifest,
            stage_output_manifest=stage_output_manifests.get(contract.id),
            assume_existing_artifact_keys=assumed_existing,
        )
        for contract in default_us_pipeline_stage_contracts()
    ]
    return {
        "schemaVersion": US_STAGE_MANIFEST_SCHEMA_VERSION,
        "contractVersion": US_STAGE_CONTRACT_VERSION,
        "generatedAt": _optional_str(manifest.get("created_at")),
        "pipeline": "us_microplex",
        "artifactRoot": ".",
        "manifest": str(artifact_map.get("manifest", "manifest.json")),
        "stages": stages,
    }


def resolve_us_stage_artifact_path(
    artifact_dir: str | Path,
    stage_manifest: dict[str, Any],
    stage_id: str,
    artifact_key: str,
) -> Path:
    """Resolve one artifact path from a stage manifest."""

    for stage in stage_manifest.get("stages", ()):
        if not isinstance(stage, dict) or stage.get("id") != stage_id:
            continue
        for artifact in stage.get("artifacts", ()):
            if (
                isinstance(artifact, dict)
                and artifact.get("key") == artifact_key
                and artifact.get("path")
            ):
                path = Path(str(artifact["path"]))
                if not path.is_absolute():
                    path = Path(artifact_dir) / path
                return path
    raise KeyError(f"Stage artifact not found: {stage_id}.{artifact_key}")


def _stage_record(
    contract: USPipelineStageContract,
    *,
    artifact_root: Path,
    manifest: dict[str, Any],
    stage_output_manifest: dict[str, Any] | None,
    assume_existing_artifact_keys: set[str],
) -> USStageRecord:
    artifacts = [
        _artifact_record(
            artifact,
            artifact_root=artifact_root,
            manifest=manifest,
            assume_existing_artifact_keys=assume_existing_artifact_keys,
        )
        for artifact in contract.artifacts
    ]
    status = stage_status(
        contract.id,
        artifact_root=artifact_root,
        manifest=manifest,
        artifacts=artifacts,
        assume_existing_artifact_keys=assume_existing_artifact_keys,
    )
    return {
        "id": contract.id,
        "step": contract.step,
        "title": contract.title,
        "purpose": contract.purpose,
        "status": status,
        "lifecycleStatus": _stage_lifecycle_status(
            stage_output_manifest,
            saved_status=status,
        ),
        "outputManifest": _stage_output_manifest_ref(manifest, contract.id),
        "startedAt": _runtime_optional_str(stage_output_manifest, "startedAt"),
        "updatedAt": _runtime_optional_str(stage_output_manifest, "updatedAt"),
        "completedAt": _runtime_optional_str(stage_output_manifest, "completedAt"),
        "failedAt": _runtime_optional_str(stage_output_manifest, "failedAt"),
        "deferredReason": _runtime_optional_str(
            stage_output_manifest,
            "deferredReason",
        ),
        "failure": _runtime_mapping_or_none(stage_output_manifest, "failure"),
        "events": _runtime_events(stage_output_manifest),
        "consumes": list(contract.consumes),
        "produces": list(contract.produces),
        "inputs": _resource_records(contract.inputs),
        "outputs": _resource_records(contract.outputs),
        "artifacts": artifacts,
        "diagnostics": list(contract.diagnostics),
        "validations": cast(
            list[USStageValidationRecord],
            [validation.to_dict() for validation in contract.validations],
        ),
        "resume": {
            "mode": contract.resume_mode,
            "notes": contract.resume_notes,
        },
        "metrics": stage_metrics(contract.id, manifest=manifest),
    }


def _load_stage_output_manifests(
    artifact_root: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    stage_manifest_paths = manifest.get("stage_output_manifests")
    if not isinstance(stage_manifest_paths, dict):
        return {}
    payloads: dict[str, dict[str, Any]] = {}
    for stage_id, value in stage_manifest_paths.items():
        if not isinstance(stage_id, str) or value is None:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = artifact_root / path
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads[stage_id] = payload
    return payloads


def _stage_output_manifest_ref(
    manifest: dict[str, Any],
    stage_id: str,
) -> str | None:
    stage_manifest_paths = manifest.get("stage_output_manifests")
    if not isinstance(stage_manifest_paths, dict):
        return None
    value = stage_manifest_paths.get(stage_id)
    return str(value) if value is not None else None


def _stage_lifecycle_status(
    stage_output_manifest: dict[str, Any] | None,
    *,
    saved_status: str,
) -> USStageLifecycleStatus:
    if stage_output_manifest is not None:
        value = stage_output_manifest.get("lifecycleStatus")
        if value in {"pending", "running", "complete", "failed", "deferred"}:
            return cast(USStageLifecycleStatus, value)
        if stage_output_manifest.get("complete") is True:
            return "complete"
        if stage_output_manifest.get("complete") is False:
            return "pending"
    if saved_status == "ready":
        return "complete"
    if saved_status == "deferred":
        return "deferred"
    return "pending"


def _runtime_optional_str(
    stage_output_manifest: dict[str, Any] | None,
    key: str,
) -> str | None:
    if stage_output_manifest is None:
        return None
    value = stage_output_manifest.get(key)
    return str(value) if value is not None else None


def _runtime_mapping_or_none(
    stage_output_manifest: dict[str, Any] | None,
    key: str,
) -> dict[str, Any] | None:
    if stage_output_manifest is None:
        return None
    value = stage_output_manifest.get(key)
    return dict(value) if isinstance(value, dict) else None


def _runtime_events(
    stage_output_manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if stage_output_manifest is None:
        return []
    events = stage_output_manifest.get("events")
    if not isinstance(events, list):
        return []
    return [dict(event) for event in events if isinstance(event, dict)]


def _artifact_record(
    artifact: USStageArtifactContract,
    *,
    artifact_root: Path,
    manifest: dict[str, Any],
    assume_existing_artifact_keys: set[str],
) -> USStageArtifactRecord:
    artifacts = dict(manifest.get("artifacts", {}))
    manifest_path = artifacts.get(artifact.key)
    path = str(manifest_path) if manifest_path else artifact.path_hint
    exists = False
    if path:
        resolved = Path(str(path))
        if not resolved.is_absolute():
            resolved = artifact_root / resolved
        exists = resolved.exists() or artifact.key in assume_existing_artifact_keys
    return {
        **artifact.to_dict(),
        "path": path,
        "exists": exists,
        "referenced": manifest_path is not None,
    }


def _resource_records(
    resources: tuple[USStageResourceContract, ...],
) -> list[USStageResourceRecord]:
    return cast(
        list[USStageResourceRecord],
        [resource.to_dict() for resource in resources],
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "build_us_stage_manifest",
    "resolve_us_stage_artifact_path",
]
