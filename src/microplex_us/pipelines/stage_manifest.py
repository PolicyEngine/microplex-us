"""Stage manifest and reusable stage artifact helpers for US builds."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from microplex_us.pipelines.stage_contracts import (
    US_STAGE_CONTRACT_VERSION,
    USPipelineStageContract,
    USStageArtifactContract,
    default_us_pipeline_stage_contracts,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    load_us_pipeline_checkpoint,
    save_us_pipeline_checkpoint,
)

US_STAGE_MANIFEST_SCHEMA_VERSION = 1
US_STAGE_ARTIFACT_ROOT = "stage_artifacts"
US_POLICYENGINE_ENTITY_STAGE_ID = "06_policyengine_entities"
US_VALIDATION_STAGE_ID = "09_validation_benchmarking"


USStageStatus = Literal[
    "ready",
    "metadata_only",
    "deferred",
    "incomplete",
    "missing",
]


class USStageMetric(TypedDict):
    """One compact metric shown for a saved stage."""

    label: str
    value: Any


class USStageArtifactRecord(TypedDict):
    """Saved-run view of one stage artifact contract."""

    key: str
    description: str
    path_hint: str | None
    required: bool
    resume_role: str | None
    path: Any
    exists: bool
    referenced: bool


class USStageResumeRecord(TypedDict):
    """Saved-run resume metadata for one stage."""

    mode: str
    notes: str


class USStageRecord(TypedDict):
    """One stage entry in a US stage manifest."""

    id: str
    step: str
    title: str
    purpose: str
    status: USStageStatus
    consumes: list[str]
    produces: list[str]
    artifacts: list[USStageArtifactRecord]
    diagnostics: list[str]
    validations: list[dict[str, object]]
    resume: USStageResumeRecord
    metrics: list[USStageMetric]


class USStageManifest(TypedDict):
    """Canonical saved-run stage manifest."""

    schemaVersion: int
    contractVersion: str
    generatedAt: Any
    pipeline: str
    artifactRoot: str
    manifest: Any
    stages: list[USStageRecord]


class USDataFlowStageSummary(TypedDict):
    """Stage summary embedded in the site-facing data-flow snapshot."""

    id: Any
    step: Any
    title: Any
    summary: Any
    status: Any
    metrics: list[Any]
    outputs: list[Any]
    resumeMode: Any


class USValidationEvidenceRecord(TypedDict):
    """One validation or benchmarking evidence sidecar."""

    key: str
    path: Any
    exists: bool


class USValidationEvidenceManifest(TypedDict):
    """Stage 9 evidence index."""

    formatVersion: int
    stageId: str
    evidence: list[USValidationEvidenceRecord]
    summaries: dict[str, Any]


def write_us_stage_manifest(
    artifact_dir: str | Path,
    output_path: str | Path,
    *,
    manifest_payload: dict[str, Any],
) -> Path:
    """Write the canonical stage manifest for a saved US artifact bundle."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(
        destination,
        build_us_stage_manifest(
            artifact_dir,
            manifest_payload=manifest_payload,
        ),
    )
    return destination


def load_us_stage_manifest(path: str | Path) -> USStageManifest:
    """Load a saved stage manifest and validate its schema version."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    if payload.get("schemaVersion") != US_STAGE_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported US stage manifest schema: "
            f"{payload.get('schemaVersion')!r}"
        )
    return cast(USStageManifest, payload)


def build_us_stage_manifest(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any],
) -> USStageManifest:
    """Build the canonical stage manifest from a saved artifact manifest."""

    artifact_root = Path(artifact_dir)
    manifest = dict(manifest_payload)
    artifact_map = dict(manifest.get("artifacts", {}))
    stages = [
        _stage_record(contract, artifact_root=artifact_root, manifest=manifest)
        for contract in default_us_pipeline_stage_contracts()
    ]
    return {
        "schemaVersion": US_STAGE_MANIFEST_SCHEMA_VERSION,
        "contractVersion": US_STAGE_CONTRACT_VERSION,
        "generatedAt": manifest.get("created_at"),
        "pipeline": "us_microplex",
        "artifactRoot": ".",
        "manifest": artifact_map.get("manifest", "manifest.json"),
        "stages": stages,
    }


def stage_summary_for_data_flow_snapshot(
    stage_manifest: USStageManifest | dict[str, Any],
) -> list[USDataFlowStageSummary]:
    """Return site-facing stage summaries from a canonical stage manifest."""

    summaries: list[USDataFlowStageSummary] = []
    for stage in stage_manifest.get("stages", ()):
        if not isinstance(stage, dict):
            continue
        summaries.append(
            {
                "id": stage.get("id"),
                "step": stage.get("step"),
                "title": stage.get("title"),
                "summary": stage.get("purpose"),
                "status": stage.get("status"),
                "metrics": list(stage.get("metrics", ())),
                "outputs": [
                    artifact.get("path")
                    for artifact in stage.get("artifacts", ())
                    if isinstance(artifact, dict) and artifact.get("path")
                ],
                "resumeMode": stage.get("resume", {}).get("mode"),
            }
        )
    return summaries


def write_us_policyengine_entity_stage_artifact(
    bundle: PolicyEngineUSEntityTableBundle,
    artifact_dir: str | Path,
) -> Path:
    """Persist a Stage 6 PE entity-table bundle as a pipeline checkpoint."""

    stage_dir = save_us_pipeline_checkpoint(
        bundle,
        artifact_dir,
        stage="post_microsim",
    )
    metadata_path = stage_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["stageId"] = US_POLICYENGINE_ENTITY_STAGE_ID
    _write_json_atomically(metadata_path, metadata)
    return metadata_path


def load_us_policyengine_entity_stage_artifact(
    path: str | Path,
) -> tuple[PolicyEngineUSEntityTableBundle, dict[str, Any]]:
    """Load a Stage 6 PE entity-table bundle artifact."""

    input_path = Path(path)
    checkpoint_dir = input_path if input_path.is_dir() else input_path.parent
    bundle, metadata = load_us_pipeline_checkpoint(
        checkpoint_dir,
        expected_stage="post_microsim",
    )
    return bundle, metadata


def build_us_validation_evidence_manifest(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any],
) -> USValidationEvidenceManifest:
    """Build a compact Stage 9 evidence index from a saved artifact manifest."""

    artifact_root = Path(artifact_dir)
    artifacts = dict(manifest_payload.get("artifacts", {}))
    evidence_keys = (
        "policyengine_harness",
        "policyengine_native_scores",
        "policyengine_native_audit",
        "imputation_ablation",
        "child_tax_unit_agi_drift",
    )
    evidence = []
    for key in evidence_keys:
        filename = artifacts.get(key)
        if not filename:
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = artifact_root / path
        evidence.append(
            {
                "key": key,
                "path": filename,
                "exists": path.exists(),
            }
        )
    return {
        "formatVersion": 1,
        "stageId": US_VALIDATION_STAGE_ID,
        "evidence": evidence,
        "summaries": {
            key: manifest_payload[key]
            for key in (
                "policyengine_harness",
                "policyengine_native_scores",
                "policyengine_native_audit",
                "imputation_ablation",
            )
            if isinstance(manifest_payload.get(key), dict)
        },
    }


def write_us_validation_evidence_manifest(
    artifact_dir: str | Path,
    output_path: str | Path,
    *,
    manifest_payload: dict[str, Any],
) -> Path:
    """Write a Stage 9 evidence manifest for validation/benchmark sidecars."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(
        destination,
        build_us_validation_evidence_manifest(
            artifact_dir,
            manifest_payload=manifest_payload,
        ),
    )
    return destination


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
) -> USStageRecord:
    artifacts = [
        _artifact_record(artifact, artifact_root=artifact_root, manifest=manifest)
        for artifact in contract.artifacts
    ]
    return {
        "id": contract.id,
        "step": contract.step,
        "title": contract.title,
        "purpose": contract.purpose,
        "status": _stage_status(
            contract.id,
            artifact_root=artifact_root,
            manifest=manifest,
            artifacts=artifacts,
        ),
        "consumes": list(contract.consumes),
        "produces": list(contract.produces),
        "artifacts": artifacts,
        "diagnostics": list(contract.diagnostics),
        "validations": [validation.to_dict() for validation in contract.validations],
        "resume": {
            "mode": contract.resume_mode,
            "notes": contract.resume_notes,
        },
        "metrics": _stage_metrics(contract.id, manifest=manifest),
    }


def _artifact_record(
    artifact: USStageArtifactContract,
    *,
    artifact_root: Path,
    manifest: dict[str, Any],
) -> USStageArtifactRecord:
    artifacts = dict(manifest.get("artifacts", {}))
    manifest_path = artifacts.get(artifact.key)
    path = manifest_path or artifact.path_hint
    exists = False
    if path:
        resolved = Path(str(path))
        if not resolved.is_absolute():
            resolved = artifact_root / resolved
        exists = resolved.exists()
    return {
        **artifact.to_dict(),
        "path": path,
        "exists": exists,
        "referenced": manifest_path is not None,
    }


def _stage_status(
    stage_id: str,
    *,
    artifact_root: Path,
    manifest: dict[str, Any],
    artifacts: list[USStageArtifactRecord],
) -> USStageStatus:
    artifact_map = dict(manifest.get("artifacts", {}))
    synthesis = dict(manifest.get("synthesis", {}))
    calibration = dict(manifest.get("calibration", {}))
    rows = dict(manifest.get("rows", {}))
    if stage_id == "01_run_profile":
        if _referenced_artifact_missing(artifacts):
            return "incomplete"
        if _artifact_exists(artifacts, "manifest"):
            return "ready"
        return "metadata_only" if manifest.get("config") else "missing"
    if stage_id == "02_source_loading":
        return "metadata_only" if synthesis.get("source_names") else "missing"
    if stage_id == "03_source_planning":
        if _referenced_artifact_missing(artifacts):
            return "incomplete"
        if _artifact_exists(artifacts, "source_plan"):
            return "ready"
        return "metadata_only" if synthesis.get("scaffold_source") else "missing"
    if stage_id == "04_seed_and_donors":
        if _referenced_artifact_missing(artifacts, required_only=True):
            return "incomplete"
        if _required_artifacts_exist(artifacts):
            return "ready"
        return "metadata_only" if rows.get("seed") else "missing"
    if stage_id == "05_synthesis":
        if _referenced_artifact_missing(artifacts, required_only=True):
            return "incomplete"
        if _required_artifacts_exist(artifacts):
            return "ready"
        return "metadata_only" if rows.get("synthetic") else "missing"
    if stage_id == "06_policyengine_entities":
        if _referenced_artifact_missing(artifacts):
            return "incomplete"
        if _artifact_exists(artifacts, "policyengine_entity_tables"):
            return "ready"
        if _manifest_artifact_exists(
            manifest,
            artifact_root,
            "policyengine_dataset",
        ):
            return "metadata_only"
        return "missing"
    if stage_id == "07_calibration":
        if _referenced_artifact_missing(artifacts, required_only=True):
            return "incomplete"
        if calibration and _required_artifacts_exist(artifacts):
            return "ready"
        return "metadata_only" if calibration and rows.get("calibrated") else "missing"
    if stage_id == "08_dataset_assembly":
        if _manifest_artifact_missing(
            manifest,
            artifact_root,
            ("policyengine_dataset", "stage_manifest", "data_flow_snapshot"),
        ):
            return "incomplete"
        if _manifest_artifact_exists(manifest, artifact_root, "policyengine_dataset"):
            return "ready"
        return "metadata_only" if artifact_map.get("stage_manifest") else "missing"
    if stage_id == "09_validation_benchmarking":
        evidence_keys = (
            "policyengine_harness",
            "policyengine_native_scores",
            "policyengine_native_audit",
            "imputation_ablation",
            "validation_evidence",
        )
        if _manifest_artifact_missing(manifest, artifact_root, evidence_keys):
            return "incomplete"
        has_evidence = any(
            _manifest_artifact_exists(manifest, artifact_root, key)
            for key in evidence_keys
        )
        if has_evidence:
            return "ready"
        if _manifest_artifact_exists(manifest, artifact_root, "policyengine_dataset"):
            return "deferred"
        return "missing"
    if any(artifact.get("exists") for artifact in artifacts):
        return "ready"
    return "missing"


def _required_artifacts_exist(artifacts: list[USStageArtifactRecord]) -> bool:
    required = [artifact for artifact in artifacts if bool(artifact.get("required"))]
    return bool(required) and all(bool(artifact.get("exists")) for artifact in required)


def _artifact_exists(artifacts: list[USStageArtifactRecord], key: str) -> bool:
    return any(
        artifact.get("key") == key and bool(artifact.get("exists"))
        for artifact in artifacts
    )


def _referenced_artifact_missing(
    artifacts: list[USStageArtifactRecord],
    *,
    required_only: bool = False,
) -> bool:
    return any(
        bool(artifact.get("referenced"))
        and not bool(artifact.get("exists"))
        and (not required_only or bool(artifact.get("required")))
        for artifact in artifacts
    )


def _manifest_artifact_exists(
    manifest: dict[str, Any],
    artifact_root: Path,
    artifact_key: str,
) -> bool:
    artifacts = dict(manifest.get("artifacts", {}))
    filename = artifacts.get(artifact_key)
    if not filename:
        return False
    path = Path(str(filename))
    if not path.is_absolute():
        path = artifact_root / path
    return path.exists()


def _manifest_artifact_missing(
    manifest: dict[str, Any],
    artifact_root: Path,
    artifact_keys: tuple[str, ...],
) -> bool:
    artifacts = dict(manifest.get("artifacts", {}))
    return any(
        bool(artifacts.get(key))
        and not _manifest_artifact_exists(manifest, artifact_root, key)
        for key in artifact_keys
    )


def _stage_metrics(stage_id: str, *, manifest: dict[str, Any]) -> list[USStageMetric]:
    synthesis = dict(manifest.get("synthesis", {}))
    calibration = dict(manifest.get("calibration", {}))
    artifacts = dict(manifest.get("artifacts", {}))
    harness = dict(manifest.get("policyengine_harness", {}))
    native_scores = dict(manifest.get("policyengine_native_scores", {}))
    rows = dict(manifest.get("rows", {}))
    config = dict(manifest.get("config", {}))
    if stage_id == "01_run_profile":
        return [
            {"label": "Target period", "value": config.get("policyengine_target_period")},
            {"label": "Backend", "value": config.get("calibration_backend")},
        ]
    if stage_id == "02_source_loading":
        return [
            {"label": "Sources", "value": len(synthesis.get("source_names", ()))},
        ]
    if stage_id == "03_source_planning":
        return [{"label": "Scaffold", "value": synthesis.get("scaffold_source")}]
    if stage_id == "04_seed_and_donors":
        return [
            {"label": "Seed rows", "value": rows.get("seed")},
            {
                "label": "Integrated vars",
                "value": len(synthesis.get("donor_integrated_variables", ())),
            },
        ]
    if stage_id == "05_synthesis":
        return [
            {"label": "Backend", "value": synthesis.get("backend")},
            {"label": "Synthetic rows", "value": rows.get("synthetic")},
        ]
    if stage_id == "06_policyengine_entities":
        return [{"label": "Entity bundle", "value": artifacts.get("policyengine_entity_tables")}]
    if stage_id == "07_calibration":
        return [
            {"label": "Backend", "value": calibration.get("backend")},
            {"label": "Supported", "value": calibration.get("n_supported_targets")},
            {"label": "Converged", "value": calibration.get("converged")},
        ]
    if stage_id == "08_dataset_assembly":
        return [{"label": "Dataset", "value": artifacts.get("policyengine_dataset")}]
    if stage_id == "09_validation_benchmarking":
        return [
            {"label": "Harness delta", "value": harness.get("mean_abs_relative_error_delta")},
            {"label": "Native delta", "value": native_scores.get("enhanced_cps_native_loss_delta")},
            {"label": "Win rate", "value": harness.get("target_win_rate")},
        ]
    return []


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


__all__ = [
    "USDataFlowStageSummary",
    "US_POLICYENGINE_ENTITY_STAGE_ID",
    "US_STAGE_ARTIFACT_ROOT",
    "US_STAGE_MANIFEST_SCHEMA_VERSION",
    "USStageArtifactRecord",
    "USStageManifest",
    "USStageMetric",
    "USStageRecord",
    "USStageStatus",
    "US_VALIDATION_STAGE_ID",
    "USValidationEvidenceManifest",
    "USValidationEvidenceRecord",
    "build_us_stage_manifest",
    "build_us_validation_evidence_manifest",
    "load_us_policyengine_entity_stage_artifact",
    "load_us_stage_manifest",
    "resolve_us_stage_artifact_path",
    "stage_summary_for_data_flow_snapshot",
    "write_us_policyengine_entity_stage_artifact",
    "write_us_stage_manifest",
    "write_us_validation_evidence_manifest",
]
