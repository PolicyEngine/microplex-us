"""Build aggregate saved-run stage manifests for US pipeline artifacts."""

from __future__ import annotations

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
    stages = [
        _stage_record(
            contract,
            artifact_root=artifact_root,
            manifest=manifest,
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
    return {
        "id": contract.id,
        "step": contract.step,
        "title": contract.title,
        "purpose": contract.purpose,
        "status": stage_status(
            contract.id,
            artifact_root=artifact_root,
            manifest=manifest,
            artifacts=artifacts,
            assume_existing_artifact_keys=assume_existing_artifact_keys,
        ),
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
