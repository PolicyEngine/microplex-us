"""Artifact inventory helpers for US Microplex saved runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import pandas as pd

from microplex_us.pipelines.stage_contracts import (
    US_STAGE_CONTRACT_VERSION,
    StageArtifactFormat,
    StageArtifactHashMode,
    StageArtifactResumeRole,
)
from microplex_us.pipelines.stage_manifest import (
    USStageManifest,
    build_us_stage_manifest,
    load_us_policyengine_entity_stage_artifact,
)

if TYPE_CHECKING:
    from microplex_us.pipelines.us import USMicroplexTargets
    from microplex_us.policyengine import PolicyEngineUSEntityTableBundle

US_STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION = 1
DEFAULT_US_STAGE_ARTIFACT_HASH_MAX_BYTES: int | None = None

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
    generatedAt: str | None
    pipeline: str
    artifactRoot: str
    manifest: str
    stageManifest: str | None
    artifacts: list[USStageArtifactInventoryRecord]


@dataclass(frozen=True)
class USSeedScaffoldStageArtifacts:
    """Reloaded Stage 4 seed/scaffold artifact."""

    scaffold_seed_data: pd.DataFrame
    artifact_paths: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class USCandidateStageArtifacts:
    """Reloaded Stage 5 candidate artifacts for manual downstream replay."""

    seed_data: pd.DataFrame
    synthetic_data: pd.DataFrame
    artifact_paths: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class USCandidateCalibrationReplayArtifacts:
    """Cross-stage artifacts for manually replaying candidate calibration."""

    candidate: USCandidateStageArtifacts
    targets: USMicroplexTargets
    seed_scaffold: USSeedScaffoldStageArtifacts | None = None
    artifact_paths: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class USPolicyEngineEntityStageArtifacts:
    """Reloaded Stage 6 PolicyEngine entity-table checkpoint."""

    bundle: PolicyEngineUSEntityTableBundle
    metadata: dict[str, Any]
    metadata_path: Path


@dataclass(frozen=True)
class USCalibratedStageArtifacts:
    """Reloaded Stage 7 calibrated data and target metadata."""

    calibrated_data: pd.DataFrame
    targets: USMicroplexTargets
    calibration_summary: dict[str, Any]
    artifact_paths: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class USDatasetAssemblyArtifacts:
    """Resolved Stage 8 dataset assembly artifacts."""

    policyengine_dataset: Path
    manifest: Path
    stage_manifest: Path
    data_flow_snapshot: Path
    artifact_inventory: Path
    conditional_readiness: Path


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
        "generatedAt": _optional_str(manifest.get("created_at")),
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


def resolve_us_stage_artifact_path_checked(
    artifact_dir: str | Path,
    stage_id: str,
    artifact_key: str,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
    expected_format: StageArtifactFormat | None = None,
    require_exists: bool = True,
) -> Path:
    """Resolve one stage artifact path and enforce format/existence checks."""

    artifact_root = Path(artifact_dir)
    record = _stage_artifact_record(
        artifact_root,
        stage_id,
        artifact_key,
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
    )
    actual_format = cast(StageArtifactFormat, record.get("format") or "unknown")
    if expected_format is not None and actual_format != expected_format:
        raise ValueError(
            f"Stage artifact {stage_id}.{artifact_key} has format "
            f"{actual_format!r}, expected {expected_format!r}"
        )
    path_text = record.get("path")
    if not path_text:
        raise KeyError(f"Stage artifact has no path: {stage_id}.{artifact_key}")
    path = Path(str(path_text))
    if not path.is_absolute():
        path = artifact_root / path
    if require_exists and not path.exists():
        raise FileNotFoundError(f"Stage artifact not found: {path}")
    return path


def load_us_stage_parquet_artifact(
    artifact_dir: str | Path,
    stage_id: str,
    artifact_key: str,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load one stage-owned parquet dataframe artifact."""

    path = resolve_us_stage_artifact_path_checked(
        artifact_dir,
        stage_id,
        artifact_key,
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
        expected_format="parquet_dataframe",
    )
    return pd.read_parquet(path)


def load_us_stage_json_artifact(
    artifact_dir: str | Path,
    stage_id: str,
    artifact_key: str,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one stage-owned JSON artifact."""

    path = resolve_us_stage_artifact_path_checked(
        artifact_dir,
        stage_id,
        artifact_key,
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
        expected_format="json",
    )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in stage artifact: {path}")
    return dict(payload)


def load_us_candidate_stage_artifacts(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
) -> USCandidateStageArtifacts:
    """Load the saved Stage 5 candidate population artifacts."""

    seed_path = resolve_us_stage_artifact_path_checked(
        artifact_dir,
        "05_donor_integration_synthesis",
        "seed_data",
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
        expected_format="parquet_dataframe",
    )
    synthetic_path = resolve_us_stage_artifact_path_checked(
        artifact_dir,
        "05_donor_integration_synthesis",
        "synthetic_data",
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
        expected_format="parquet_dataframe",
    )
    return USCandidateStageArtifacts(
        seed_data=pd.read_parquet(seed_path),
        synthetic_data=pd.read_parquet(synthetic_path),
        artifact_paths={
            "seed_data": seed_path,
            "synthetic_data": synthetic_path,
        },
    )


def load_us_seed_scaffold_stage_artifacts(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
) -> USSeedScaffoldStageArtifacts:
    """Load the saved Stage 4 seed/scaffold artifact."""

    scaffold_seed_path = resolve_us_stage_artifact_path_checked(
        artifact_dir,
        "04_seed_scaffold",
        "scaffold_seed_data",
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
        expected_format="parquet_dataframe",
    )
    return USSeedScaffoldStageArtifacts(
        scaffold_seed_data=pd.read_parquet(scaffold_seed_path),
        artifact_paths={"scaffold_seed_data": scaffold_seed_path},
    )


def load_us_candidate_calibration_replay_artifacts(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
    include_seed_scaffold: bool = True,
) -> USCandidateCalibrationReplayArtifacts:
    """Load the cross-stage artifacts needed to manually replay calibration."""

    from microplex_us.pipelines.us import USMicroplexTargets

    candidate = load_us_candidate_stage_artifacts(
        artifact_dir,
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
    )
    targets_path = resolve_us_stage_artifact_path_checked(
        artifact_dir,
        "07_calibration",
        "targets",
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
        expected_format="json",
    )
    seed_scaffold = None
    if include_seed_scaffold:
        try:
            seed_scaffold = load_us_seed_scaffold_stage_artifacts(
                artifact_dir,
                manifest_payload=manifest_payload,
                stage_manifest=stage_manifest,
            )
        except (KeyError, FileNotFoundError):
            seed_scaffold = None
    targets_payload = json.loads(targets_path.read_text())
    artifact_paths = {
        **dict(candidate.artifact_paths),
        "targets": targets_path,
    }
    if seed_scaffold is not None:
        artifact_paths.update(seed_scaffold.artifact_paths)
    return USCandidateCalibrationReplayArtifacts(
        candidate=candidate,
        targets=USMicroplexTargets(
            marginal=dict(targets_payload.get("marginal", {})),
            continuous=dict(targets_payload.get("continuous", {})),
        ),
        seed_scaffold=seed_scaffold,
        artifact_paths=artifact_paths,
    )


def load_us_policyengine_entity_stage_artifacts(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
) -> USPolicyEngineEntityStageArtifacts:
    """Load the saved Stage 6 PolicyEngine entity-table bundle."""

    metadata_path = resolve_us_stage_artifact_path_checked(
        artifact_dir,
        "06_policyengine_entities",
        "policyengine_entity_tables",
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
        expected_format="policyengine_entity_bundle",
    )
    bundle, metadata = load_us_policyengine_entity_stage_artifact(metadata_path)
    return USPolicyEngineEntityStageArtifacts(
        bundle=bundle,
        metadata=metadata,
        metadata_path=metadata_path,
    )


def load_us_calibrated_stage_artifacts(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
) -> USCalibratedStageArtifacts:
    """Load saved Stage 7 calibrated outputs and calibration metadata."""

    from microplex_us.pipelines.us import USMicroplexTargets

    calibrated_path = resolve_us_stage_artifact_path_checked(
        artifact_dir,
        "07_calibration",
        "calibrated_data",
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
        expected_format="parquet_dataframe",
    )
    targets_path = resolve_us_stage_artifact_path_checked(
        artifact_dir,
        "07_calibration",
        "targets",
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
        expected_format="json",
    )
    calibration_summary_path = resolve_us_stage_artifact_path_checked(
        artifact_dir,
        "07_calibration",
        "calibration_summary",
        manifest_payload=manifest_payload,
        stage_manifest=stage_manifest,
        expected_format="json",
    )
    targets_payload = json.loads(targets_path.read_text())
    return USCalibratedStageArtifacts(
        calibrated_data=pd.read_parquet(calibrated_path),
        targets=USMicroplexTargets(
            marginal=dict(targets_payload.get("marginal", {})),
            continuous=dict(targets_payload.get("continuous", {})),
        ),
        calibration_summary=json.loads(calibration_summary_path.read_text()),
        artifact_paths={
            "calibrated_data": calibrated_path,
            "targets": targets_path,
            "calibration_summary": calibration_summary_path,
        },
    )


def load_us_dataset_assembly_artifacts(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
) -> USDatasetAssemblyArtifacts:
    """Resolve saved Stage 8 dataset assembly artifacts."""

    artifact_root = Path(artifact_dir)
    return USDatasetAssemblyArtifacts(
        policyengine_dataset=resolve_us_stage_artifact_path_checked(
            artifact_root,
            "08_dataset_assembly",
            "policyengine_dataset",
            manifest_payload=manifest_payload,
            stage_manifest=stage_manifest,
            expected_format="h5_dataset",
        ),
        manifest=artifact_root / "manifest.json",
        stage_manifest=resolve_us_stage_artifact_path_checked(
            artifact_root,
            "08_dataset_assembly",
            "stage_manifest",
            manifest_payload=manifest_payload,
            stage_manifest=stage_manifest,
            expected_format="json",
        ),
        data_flow_snapshot=resolve_us_stage_artifact_path_checked(
            artifact_root,
            "08_dataset_assembly",
            "data_flow_snapshot",
            manifest_payload=manifest_payload,
            stage_manifest=stage_manifest,
            expected_format="json",
        ),
        artifact_inventory=resolve_us_stage_artifact_path_checked(
            artifact_root,
            "08_dataset_assembly",
            "artifact_inventory",
            manifest_payload=manifest_payload,
            stage_manifest=stage_manifest,
            expected_format="json",
        ),
        conditional_readiness=resolve_us_stage_artifact_path_checked(
            artifact_root,
            "08_dataset_assembly",
            "conditional_readiness",
            manifest_payload=manifest_payload,
            stage_manifest=stage_manifest,
            expected_format="json",
        ),
    )


def _stage_artifact_record(
    artifact_root: Path,
    stage_id: str,
    artifact_key: str,
    *,
    manifest_payload: dict[str, Any] | None,
    stage_manifest: USStageManifest | dict[str, Any] | None,
) -> dict[str, Any]:
    manifest = (
        dict(manifest_payload)
        if manifest_payload is not None
        else json.loads((artifact_root / "manifest.json").read_text())
    )
    stages = (
        dict(stage_manifest)
        if stage_manifest is not None
        else build_us_stage_manifest(artifact_root, manifest_payload=manifest)
    )
    for stage in stages.get("stages", ()):
        if not isinstance(stage, dict) or stage.get("id") != stage_id:
            continue
        for artifact in stage.get("artifacts", ()):
            if isinstance(artifact, dict) and artifact.get("key") == artifact_key:
                return dict(artifact)
    raise KeyError(f"Stage artifact not found: {stage_id}.{artifact_key}")


def _resolve_optional_stage_artifact_path(
    artifact_dir: str | Path,
    stage_id: str,
    artifact_key: str,
    *,
    manifest_payload: dict[str, Any] | None,
    stage_manifest: USStageManifest | dict[str, Any] | None,
    expected_format: StageArtifactFormat,
) -> Path | None:
    try:
        return resolve_us_stage_artifact_path_checked(
            artifact_dir,
            stage_id,
            artifact_key,
            manifest_payload=manifest_payload,
            stage_manifest=stage_manifest,
            expected_format=expected_format,
        )
    except (KeyError, FileNotFoundError):
        return None


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
        if bool(artifact.get("required")):
            return "missing_required"
        if bool(artifact.get("referenced")):
            return "missing_optional"
        return "contract_only"
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
    "USCalibratedStageArtifacts",
    "USCandidateStageArtifacts",
    "USCandidateCalibrationReplayArtifacts",
    "USDatasetAssemblyArtifacts",
    "USPolicyEngineEntityStageArtifacts",
    "USSeedScaffoldStageArtifacts",
    "USStageArtifactClassification",
    "USStageArtifactHashStatus",
    "USStageArtifactInventory",
    "USStageArtifactInventoryRecord",
    "build_us_stage_artifact_inventory",
    "load_us_calibrated_stage_artifacts",
    "load_us_candidate_calibration_replay_artifacts",
    "load_us_candidate_stage_artifacts",
    "load_us_dataset_assembly_artifacts",
    "load_us_policyengine_entity_stage_artifacts",
    "load_us_seed_scaffold_stage_artifacts",
    "load_us_stage_json_artifact",
    "load_us_stage_parquet_artifact",
    "load_us_stage_artifact_inventory",
    "resolve_us_stage_artifact_path_checked",
    "resolve_us_stage_artifact_from_inventory",
    "write_us_stage_artifact_inventory",
]
