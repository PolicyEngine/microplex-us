"""Artifact inventory helpers for US Microplex saved runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from microplex_us.pipelines.stage_contracts import (
    US_STAGE_CONTRACT_VERSION,
    StageArtifactFormat,
    StageArtifactHashMode,
    StageArtifactResumeRole,
)
from microplex_us.pipelines.stage_manifest import (
    USStageManifest,
    build_us_stage_manifest,
)

US_STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION = 1
DEFAULT_US_STAGE_ARTIFACT_HASH_MAX_BYTES = 25_000_000

USStageArtifactClassification = Literal[
    "contract_only",
    "diagnostic_only",
    "manual_replay",
    "manual_resume",
    "post_artifact_evidence",
    "missing_required",
    "missing_optional",
    "metadata_only",
]

USStageArtifactHashStatus = Literal[
    "hashed",
    "not_requested",
    "missing",
    "too_large",
    "unsupported",
    "error",
]


class USStageArtifactInventoryRecord(TypedDict):
    """Inventory view of one canonical stage artifact."""

    stageId: str
    stageStep: str
    stageTitle: str
    key: str
    description: str
    path: str | None
    exists: bool
    referenced: bool
    required: bool
    resumeRole: StageArtifactResumeRole | None
    format: StageArtifactFormat
    hashMode: StageArtifactHashMode
    classification: USStageArtifactClassification
    sizeBytes: int | None
    fileCount: int | None
    contentHash: str | None
    hashStatus: USStageArtifactHashStatus


class USStageArtifactInventory(TypedDict):
    """Machine-readable artifact inventory for one saved run."""

    schemaVersion: int
    contractVersion: str
    generatedAt: str
    pipeline: str
    artifactRoot: str
    manifest: str
    stageManifest: str | None
    artifacts: list[USStageArtifactInventoryRecord]


def build_us_stage_artifact_inventory(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
    assume_existing_artifact_keys: Iterable[str] = (),
    max_hash_bytes: int | None = DEFAULT_US_STAGE_ARTIFACT_HASH_MAX_BYTES,
) -> USStageArtifactInventory:
    """Build an artifact inventory for one US Microplex saved-run directory."""

    artifact_root = Path(artifact_dir)
    manifest = (
        dict(manifest_payload)
        if manifest_payload is not None
        else json.loads((artifact_root / "manifest.json").read_text())
    )
    stages = (
        dict(stage_manifest)
        if stage_manifest is not None
        else build_us_stage_manifest(
            artifact_root,
            manifest_payload=manifest,
            assume_existing_artifact_keys=assume_existing_artifact_keys,
        )
    )
    artifacts: list[USStageArtifactInventoryRecord] = []
    for stage in stages.get("stages", ()):
        if not isinstance(stage, dict):
            continue
        stage_id = str(stage.get("id", ""))
        stage_step = str(stage.get("step", ""))
        stage_title = str(stage.get("title", ""))
        for artifact in stage.get("artifacts", ()):
            if isinstance(artifact, dict):
                artifacts.append(
                    _inventory_record(
                        artifact,
                        stage_id=stage_id,
                        stage_step=stage_step,
                        stage_title=stage_title,
                        artifact_root=artifact_root,
                        max_hash_bytes=max_hash_bytes,
                    )
                )

    manifest_artifacts = dict(manifest.get("artifacts", {}))
    return {
        "schemaVersion": US_STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION,
        "contractVersion": US_STAGE_CONTRACT_VERSION,
        "generatedAt": datetime.now(UTC).isoformat(),
        "pipeline": "us_microplex",
        "artifactRoot": ".",
        "manifest": str(manifest_artifacts.get("manifest", "manifest.json")),
        "stageManifest": _optional_str(manifest_artifacts.get("stage_manifest")),
        "artifacts": artifacts,
    }


def write_us_stage_artifact_inventory(
    artifact_dir: str | Path,
    output_path: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
    assume_existing_artifact_keys: Iterable[str] = (),
    max_hash_bytes: int | None = DEFAULT_US_STAGE_ARTIFACT_HASH_MAX_BYTES,
) -> Path:
    """Write an artifact inventory sidecar for one saved run."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(
        destination,
        build_us_stage_artifact_inventory(
            artifact_dir,
            manifest_payload=manifest_payload,
            stage_manifest=stage_manifest,
            assume_existing_artifact_keys=assume_existing_artifact_keys,
            max_hash_bytes=max_hash_bytes,
        ),
    )
    return destination


def load_us_stage_artifact_inventory(path: str | Path) -> USStageArtifactInventory:
    """Load a saved artifact inventory and validate its schema version."""

    inventory_path = Path(path)
    payload = json.loads(inventory_path.read_text())
    if payload.get("schemaVersion") != US_STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported US stage artifact inventory schema: "
            f"{payload.get('schemaVersion')!r}"
        )
    return cast(USStageArtifactInventory, payload)


def resolve_us_stage_artifact_from_inventory(
    artifact_dir: str | Path,
    inventory: USStageArtifactInventory | dict[str, Any],
    stage_id: str,
    artifact_key: str,
) -> Path:
    """Resolve one artifact path from a stage artifact inventory."""

    for artifact in inventory.get("artifacts", ()):
        if not isinstance(artifact, dict):
            continue
        if artifact.get("stageId") != stage_id or artifact.get("key") != artifact_key:
            continue
        path_text = artifact.get("path")
        if not path_text:
            raise KeyError(f"Stage artifact has no path: {stage_id}.{artifact_key}")
        path = Path(str(path_text))
        if not path.is_absolute():
            path = Path(artifact_dir) / path
        return path
    raise KeyError(f"Stage artifact not found: {stage_id}.{artifact_key}")


def _inventory_record(
    artifact: dict[str, Any],
    *,
    stage_id: str,
    stage_step: str,
    stage_title: str,
    artifact_root: Path,
    max_hash_bytes: int | None,
) -> USStageArtifactInventoryRecord:
    path_text = _optional_str(artifact.get("path"))
    resolved_path = _resolve_artifact_path(artifact_root, path_text)
    artifact_format = cast(
        StageArtifactFormat,
        artifact.get("format") or "unknown",
    )
    hash_mode = cast(
        StageArtifactHashMode,
        artifact.get("hash_mode") or "none",
    )
    hash_target = _hash_target_path(resolved_path, artifact_format, hash_mode)
    size_bytes, file_count = _artifact_size(hash_target)
    content_hash, hash_status = _artifact_hash(
        hash_target,
        hash_mode=hash_mode,
        max_hash_bytes=max_hash_bytes,
    )
    return {
        "stageId": stage_id,
        "stageStep": stage_step,
        "stageTitle": stage_title,
        "key": str(artifact.get("key", "")),
        "description": str(artifact.get("description", "")),
        "path": path_text,
        "exists": bool(artifact.get("exists")),
        "referenced": bool(artifact.get("referenced")),
        "required": bool(artifact.get("required")),
        "resumeRole": cast(StageArtifactResumeRole | None, artifact.get("resume_role")),
        "format": artifact_format,
        "hashMode": hash_mode,
        "classification": _artifact_classification(artifact),
        "sizeBytes": size_bytes,
        "fileCount": file_count,
        "contentHash": content_hash,
        "hashStatus": hash_status,
    }


def _artifact_classification(
    artifact: Mapping[str, Any],
) -> USStageArtifactClassification:
    if not bool(artifact.get("exists")):
        if not bool(artifact.get("referenced")):
            return "contract_only"
        return "missing_required" if bool(artifact.get("required")) else "missing_optional"
    resume_role = artifact.get("resume_role")
    if resume_role == "diagnostic":
        return "diagnostic_only"
    if resume_role in {"manual_replay", "manual_resume", "post_artifact_evidence"}:
        return cast(USStageArtifactClassification, resume_role)
    return "metadata_only"


def _resolve_artifact_path(artifact_root: Path, path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = artifact_root / path
    return path


def _hash_target_path(
    path: Path | None,
    artifact_format: StageArtifactFormat,
    hash_mode: StageArtifactHashMode,
) -> Path | None:
    if path is None or hash_mode != "directory_sha256":
        return path
    if artifact_format == "policyengine_entity_bundle" and path.name == "metadata.json":
        return path.parent
    return path


def _artifact_size(path: Path | None) -> tuple[int | None, int | None]:
    if path is None or not path.exists():
        return None, None
    if path.is_file():
        return path.stat().st_size, 1
    if path.is_dir():
        total = 0
        count = 0
        for child in _iter_directory_files(path):
            total += child.stat().st_size
            count += 1
        return total, count
    return None, None


def _artifact_hash(
    path: Path | None,
    *,
    hash_mode: StageArtifactHashMode,
    max_hash_bytes: int | None,
) -> tuple[str | None, USStageArtifactHashStatus]:
    if hash_mode == "none":
        return None, "not_requested"
    if path is None or not path.exists():
        return None, "missing"
    try:
        if hash_mode == "file_sha256":
            if not path.is_file():
                return None, "unsupported"
            size = path.stat().st_size
            if max_hash_bytes is not None and size > max_hash_bytes:
                return None, "too_large"
            return _hash_file(path), "hashed"
        if hash_mode == "directory_sha256":
            if not path.is_dir():
                return None, "unsupported"
            size, _ = _artifact_size(path)
            if (
                max_hash_bytes is not None
                and size is not None
                and size > max_hash_bytes
            ):
                return None, "too_large"
            return _hash_directory(path), "hashed"
    except OSError:
        return None, "error"
    return None, "unsupported"


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_directory(path: Path) -> str:
    hasher = hashlib.sha256()
    for child in _iter_directory_files(path):
        relative = child.relative_to(path).as_posix()
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_hash_file(child).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _iter_directory_files(path: Path) -> list[Path]:
    return sorted(child for child in path.rglob("*") if child.is_file())


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


__all__ = [
    "DEFAULT_US_STAGE_ARTIFACT_HASH_MAX_BYTES",
    "US_STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION",
    "USStageArtifactClassification",
    "USStageArtifactHashStatus",
    "USStageArtifactInventory",
    "USStageArtifactInventoryRecord",
    "build_us_stage_artifact_inventory",
    "load_us_stage_artifact_inventory",
    "resolve_us_stage_artifact_from_inventory",
    "write_us_stage_artifact_inventory",
]
