"""Validation and benchmarking evidence manifests for US saved runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microplex_us.pipeline_metadata import pipeline_node
from microplex_us.pipelines.stage_manifest_io import write_json_atomically
from microplex_us.pipelines.stage_manifest_types import (
    US_VALIDATION_STAGE_ID,
    USValidationEvidenceManifest,
    USValidationEvidenceRecord,
)


@pipeline_node(
    id="us.stage9.build_validation_evidence_manifest",
    label="Build validation evidence manifest",
    description="Index validation and benchmark evidence files written for Stage 9.",
    artifacts_in=("artifact_manifest",),
    artifacts_out=("validation_evidence_manifest",),
)
def build_us_validation_evidence_manifest(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any],
) -> USValidationEvidenceManifest:
    """Build a compact Stage 9 evidence index from a saved artifact manifest."""

    artifact_root = Path(artifact_dir)
    artifacts = dict(manifest_payload.get("artifacts", {}))
    existing = _load_existing_validation_evidence_manifest(artifact_root, artifacts)
    evidence_keys = (
        "policyengine_harness",
        "policyengine_native_scores",
        "policyengine_native_audit",
        "policyengine_native_target_diagnostics",
        "imputation_ablation",
        "child_tax_unit_agi_drift",
    )
    evidence_by_key: dict[str, USValidationEvidenceRecord] = {}
    if existing is not None:
        for record in existing.get("evidence", ()):
            if not isinstance(record, Mapping) or not record.get("key"):
                continue
            key = str(record["key"])
            evidence_by_key[key] = _validation_evidence_record(
                artifact_root,
                key,
                record.get("path"),
            )
    for key in evidence_keys:
        filename = artifacts.get(key)
        if not filename:
            continue
        evidence_by_key[key] = _validation_evidence_record(
            artifact_root,
            key,
            filename,
        )
    summaries: dict[str, Any] = {}
    if existing is not None and isinstance(existing.get("summaries"), Mapping):
        summaries.update(dict(existing["summaries"]))
    summaries.update(
        {
            key: manifest_payload[key]
            for key in (
                "policyengine_harness",
                "policyengine_native_scores",
                "policyengine_native_audit",
                "imputation_ablation",
            )
            if isinstance(manifest_payload.get(key), dict)
        }
    )
    return {
        "formatVersion": 1,
        "stageId": US_VALIDATION_STAGE_ID,
        "evidence": list(evidence_by_key.values()),
        "summaries": summaries,
    }


@pipeline_node(
    id="us.stage9.write_validation_evidence_manifest",
    label="Write validation evidence manifest",
    description="Persist the Stage 9 validation evidence manifest.",
    artifacts_in=("validation_evidence_manifest",),
    artifacts_out=("validation_evidence_manifest_file",),
)
def write_us_validation_evidence_manifest(
    artifact_dir: str | Path,
    output_path: str | Path,
    *,
    manifest_payload: dict[str, Any],
) -> Path:
    """Write a Stage 9 evidence manifest for validation/benchmark sidecars."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomically(
        destination,
        build_us_validation_evidence_manifest(
            artifact_dir,
            manifest_payload=manifest_payload,
        ),
    )
    return destination


def _load_existing_validation_evidence_manifest(
    artifact_root: Path,
    artifacts: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    evidence_name = artifacts.get("validation_evidence")
    if not evidence_name:
        return None
    path = Path(str(evidence_name))
    if not path.is_absolute():
        path = artifact_root / path
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _validation_evidence_record(
    artifact_root: Path,
    key: str,
    path_value: Any,
) -> USValidationEvidenceRecord:
    path_text = str(path_value) if path_value else ""
    path = Path(path_text)
    if path_text and not path.is_absolute():
        path = artifact_root / path
    return {
        "key": key,
        "path": path_text,
        "exists": bool(path_text) and path.exists(),
    }


__all__ = [
    "build_us_validation_evidence_manifest",
    "write_us_validation_evidence_manifest",
]
