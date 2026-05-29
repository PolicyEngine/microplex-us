"""Conditional-readiness reports for US Microplex saved runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from microplex_us.pipelines.stage_artifacts import (
    USStageArtifactInventory,
    USStageArtifactInventoryRecord,
    build_us_stage_artifact_inventory,
    load_us_stage_artifact_inventory,
)
from microplex_us.pipelines.stage_contracts import US_STAGE_CONTRACT_VERSION
from microplex_us.pipelines.stage_manifest import (
    USStageManifest,
    USStageStatus,
    build_us_stage_manifest,
)

US_CONDITIONAL_READINESS_SCHEMA_VERSION = 1
US_CONFIG_REUSE_IGNORED_KEYS = frozenset(
    {
        "pipeline_checkpoint_save_post_imputation_path",
        "pipeline_checkpoint_save_post_microsim_path",
    }
)

USStageReadiness = Literal[
    "manual_replay",
    "manual_resume",
    "post_artifact_evidence",
    "diagnostic_only",
    "metadata_only",
    "must_rerun",
    "not_applicable",
]

USStageCompatibility = Literal[
    "match",
    "mismatch",
    "missing_saved_config",
    "not_evaluated",
]


class USConditionalReadinessStageRecord(TypedDict):
    """Conditional-readiness view of one canonical stage."""

    stageId: str
    stageStep: str
    stageTitle: str
    status: USStageStatus
    readiness: USStageReadiness
    reason: str
    compatibility: USStageCompatibility
    reuseKey: str | None
    availableArtifacts: list[str]
    missingArtifacts: list[str]
    diagnosticArtifacts: list[str]
    reloadableArtifacts: list[str]


class USConditionalReadinessReport(TypedDict):
    """Saved-run conditional-readiness report."""

    schemaVersion: int
    contractVersion: str
    generatedAt: str
    pipeline: str
    artifactRoot: str
    manifest: str
    artifactInventory: str | None
    savedConfigHash: str | None
    requestedConfigHash: str | None
    stages: list[USConditionalReadinessStageRecord]


def build_us_stage_reuse_key(
    stage_id: str,
    manifest_payload: Mapping[str, Any],
    artifact_inventory: USStageArtifactInventory | Mapping[str, Any],
) -> str | None:
    """Return a deterministic reuse key for one stage, if any evidence exists."""

    stage_artifacts = [
        artifact
        for artifact in artifact_inventory.get("artifacts", ())
        if isinstance(artifact, dict) and artifact.get("stageId") == stage_id
    ]
    if not stage_artifacts:
        return None
    evidence = [
        {
            "key": str(artifact.get("key")),
            "path": artifact.get("path"),
            "classification": artifact.get("classification"),
            "hashStatus": artifact.get("hashStatus"),
            "contentHash": artifact.get("contentHash"),
            "sizeBytes": artifact.get("sizeBytes"),
            "fileCount": artifact.get("fileCount"),
        }
        for artifact in stage_artifacts
        if artifact.get("exists") or artifact.get("referenced")
    ]
    if not evidence:
        return None
    payload = {
        "stageId": stage_id,
        "configHash": _config_hash(manifest_payload.get("config")),
        "artifacts": sorted(evidence, key=lambda item: item["key"]),
    }
    return _hash_json(payload)


def build_us_conditional_readiness_report(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
    artifact_inventory: USStageArtifactInventory | dict[str, Any] | None = None,
    requested_config: Mapping[str, Any] | None = None,
) -> USConditionalReadinessReport:
    """Build a report describing which stage outputs could be reused manually."""

    artifact_root = Path(artifact_dir)
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
    inventory = (
        dict(artifact_inventory)
        if artifact_inventory is not None
        else _load_or_build_inventory(artifact_root, manifest_payload=manifest)
    )
    saved_config_hash = _config_hash(manifest.get("config"))
    requested_config_hash = (
        _config_hash(requested_config) if requested_config is not None else None
    )
    compatibility = _config_compatibility(
        saved_config_hash,
        requested_config_hash,
        requested_config_supplied=requested_config is not None,
    )
    return {
        "schemaVersion": US_CONDITIONAL_READINESS_SCHEMA_VERSION,
        "contractVersion": US_STAGE_CONTRACT_VERSION,
        "generatedAt": datetime.now(UTC).isoformat(),
        "pipeline": "us_microplex",
        "artifactRoot": ".",
        "manifest": str(dict(manifest.get("artifacts", {})).get("manifest", "manifest.json")),
        "artifactInventory": _optional_str(
            dict(manifest.get("artifacts", {})).get("artifact_inventory")
        ),
        "savedConfigHash": saved_config_hash,
        "requestedConfigHash": requested_config_hash,
        "stages": [
            _readiness_stage_record(
                stage,
                manifest=manifest,
                inventory=inventory,
                compatibility=compatibility,
            )
            for stage in stages.get("stages", ())
            if isinstance(stage, dict)
        ],
    }


def write_us_conditional_readiness_report(
    artifact_dir: str | Path,
    output_path: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    stage_manifest: USStageManifest | dict[str, Any] | None = None,
    artifact_inventory: USStageArtifactInventory | dict[str, Any] | None = None,
    requested_config: Mapping[str, Any] | None = None,
) -> Path:
    """Write a conditional-readiness report sidecar for one saved run."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(
        destination,
        build_us_conditional_readiness_report(
            artifact_dir,
            manifest_payload=manifest_payload,
            stage_manifest=stage_manifest,
            artifact_inventory=artifact_inventory,
            requested_config=requested_config,
        ),
    )
    return destination


def load_us_conditional_readiness_report(
    path: str | Path,
) -> USConditionalReadinessReport:
    """Load a saved conditional-readiness report."""

    report_path = Path(path)
    payload = json.loads(report_path.read_text())
    if payload.get("schemaVersion") != US_CONDITIONAL_READINESS_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported US conditional-readiness report schema: "
            f"{payload.get('schemaVersion')!r}"
        )
    return cast(USConditionalReadinessReport, payload)


def _readiness_stage_record(
    stage: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
    compatibility: USStageCompatibility,
) -> USConditionalReadinessStageRecord:
    stage_id = str(stage.get("id", ""))
    artifacts = _inventory_artifacts_for_stage(inventory, stage_id)
    available = [
        _artifact_label(artifact)
        for artifact in artifacts
        if bool(artifact.get("exists"))
    ]
    missing = [
        _artifact_label(artifact)
        for artifact in artifacts
        if artifact.get("classification") in {"missing_required", "missing_optional"}
    ]
    diagnostic = [
        _artifact_label(artifact)
        for artifact in artifacts
        if artifact.get("classification") == "diagnostic_only"
    ]
    reloadable = [
        _artifact_label(artifact)
        for artifact in artifacts
        if artifact.get("classification")
        in {"manual_replay", "manual_resume", "post_artifact_evidence"}
    ]
    readiness, reason = _stage_readiness(
        stage,
        artifacts,
        compatibility=compatibility,
        stage8_dataset_available=_stage8_dataset_available(inventory),
    )
    return {
        "stageId": stage_id,
        "stageStep": str(stage.get("step", "")),
        "stageTitle": str(stage.get("title", "")),
        "status": cast(USStageStatus, stage.get("status", "missing")),
        "readiness": readiness,
        "reason": reason,
        "compatibility": compatibility,
        "reuseKey": build_us_stage_reuse_key(stage_id, manifest, inventory),
        "availableArtifacts": available,
        "missingArtifacts": missing,
        "diagnosticArtifacts": diagnostic,
        "reloadableArtifacts": reloadable,
    }


def _stage_readiness(
    stage: Mapping[str, Any],
    artifacts: list[USStageArtifactInventoryRecord],
    *,
    compatibility: USStageCompatibility,
    stage8_dataset_available: bool,
) -> tuple[USStageReadiness, str]:
    stage_id = str(stage.get("id", ""))
    status = stage.get("status")
    if status in {"missing", "incomplete"}:
        return "must_rerun", f"Stage status is {status}."
    if compatibility == "mismatch":
        return "must_rerun", "Requested configuration does not match the saved run."
    if stage_id == "09_validation_benchmarking" and status == "deferred":
        if stage8_dataset_available:
            return (
                "post_artifact_evidence",
                "Stage 8 dataset is available for validation or benchmark evidence.",
            )
        return "must_rerun", "Validation is deferred and no Stage 8 dataset is available."
    classifications = {
        str(artifact.get("classification"))
        for artifact in artifacts
        if bool(artifact.get("exists"))
    }
    for readiness in ("manual_resume", "manual_replay", "post_artifact_evidence"):
        if readiness in classifications:
            return cast(USStageReadiness, readiness), (
                f"Stage has existing {readiness.replace('_', ' ')} artifacts."
            )
    if "diagnostic_only" in classifications:
        return "diagnostic_only", "Stage has diagnostic artifacts but no replay boundary."
    if status == "metadata_only":
        return "metadata_only", "Stage has metadata but no reloadable artifact."
    return "not_applicable", "No reusable artifact boundary is available."


def _inventory_artifacts_for_stage(
    inventory: Mapping[str, Any],
    stage_id: str,
) -> list[USStageArtifactInventoryRecord]:
    return [
        cast(USStageArtifactInventoryRecord, artifact)
        for artifact in inventory.get("artifacts", ())
        if isinstance(artifact, dict) and artifact.get("stageId") == stage_id
    ]


def _stage8_dataset_available(inventory: Mapping[str, Any]) -> bool:
    return any(
        isinstance(artifact, dict)
        and artifact.get("stageId") == "08_dataset_assembly"
        and artifact.get("key") == "policyengine_dataset"
        and bool(artifact.get("exists"))
        for artifact in inventory.get("artifacts", ())
    )


def _load_or_build_inventory(
    artifact_root: Path,
    *,
    manifest_payload: dict[str, Any],
) -> USStageArtifactInventory:
    inventory_name = dict(manifest_payload.get("artifacts", {})).get("artifact_inventory")
    if isinstance(inventory_name, str):
        inventory_path = Path(inventory_name)
        if not inventory_path.is_absolute():
            inventory_path = artifact_root / inventory_path
        if inventory_path.exists():
            return load_us_stage_artifact_inventory(inventory_path)
    return build_us_stage_artifact_inventory(
        artifact_root,
        manifest_payload=manifest_payload,
    )


def _config_compatibility(
    saved_config_hash: str | None,
    requested_config_hash: str | None,
    *,
    requested_config_supplied: bool,
) -> USStageCompatibility:
    if not requested_config_supplied:
        return "not_evaluated"
    if saved_config_hash is None:
        return "missing_saved_config"
    return "match" if saved_config_hash == requested_config_hash else "mismatch"


def _config_hash(config: Any) -> str | None:
    if not isinstance(config, Mapping):
        return None
    return _hash_json(_canonical_config(config))


def _canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _normalize_config_value(value)
        for key, value in sorted(config.items())
        if key not in US_CONFIG_REUSE_IGNORED_KEYS
    }


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_config_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _artifact_label(artifact: Mapping[str, Any]) -> str:
    return f"{artifact.get('stageId')}.{artifact.get('key')}"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


__all__ = [
    "US_CONDITIONAL_READINESS_SCHEMA_VERSION",
    "US_CONFIG_REUSE_IGNORED_KEYS",
    "USConditionalReadinessReport",
    "USConditionalReadinessStageRecord",
    "USStageCompatibility",
    "USStageReadiness",
    "build_us_conditional_readiness_report",
    "build_us_stage_reuse_key",
    "load_us_conditional_readiness_report",
    "write_us_conditional_readiness_report",
]
