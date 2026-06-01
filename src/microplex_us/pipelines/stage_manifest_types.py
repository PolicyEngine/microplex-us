"""Shared saved-run stage manifest schemas for US pipeline artifacts."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from microplex_us.pipelines.stage_contracts import (
    StageArtifactFormat,
    StageArtifactHashMode,
    StageResumeMode,
)

US_STAGE_MANIFEST_SCHEMA_VERSION = 3
SUPPORTED_US_STAGE_MANIFEST_SCHEMA_VERSIONS = frozenset({1, 2, 3})
US_STAGE_ARTIFACT_ROOT = "stage_artifacts"
US_POLICYENGINE_ENTITY_STAGE_ID = "06_policyengine_entities"
US_VALIDATION_STAGE_ID = "09_validation_benchmarking"


USStageMetricValue = str | int | float | bool | None

USStageStatus = Literal[
    "ready",
    "metadata_only",
    "deferred",
    "incomplete",
    "missing",
]

USStageValidationStatus = Literal["planned", "manual", "implemented"]
USStageLifecycleStatus = Literal[
    "pending",
    "running",
    "complete",
    "failed",
    "deferred",
]


class USStageMetric(TypedDict):
    """One compact metric shown for a saved stage."""

    label: str
    value: USStageMetricValue


class USStageArtifactRecord(TypedDict):
    """Saved-run view of one stage artifact contract."""

    key: str
    description: str
    path_hint: str | None
    required: bool
    resume_role: str | None
    format: StageArtifactFormat
    hash_mode: StageArtifactHashMode
    path: str | None
    exists: bool
    referenced: bool


class USStageResumeRecord(TypedDict):
    """Saved-run resume metadata for one stage."""

    mode: StageResumeMode
    notes: str


class USStageValidationRecord(TypedDict):
    """Saved-run view of one planned or implemented validation."""

    key: str
    description: str
    status: USStageValidationStatus


class USStageFailureRecord(TypedDict, total=False):
    """Runtime failure details for one stage."""

    errorType: str
    message: str
    traceback: str | None


class USStageRuntimeEventRecord(TypedDict, total=False):
    """Compact runtime event included in a stage output manifest."""

    event: str
    timestamp: str
    details: dict[str, Any]


class USStageResourceRecord(TypedDict):
    """Saved-run view of one structured stage input or output."""

    key: str
    description: str
    kind: str
    required: bool
    stage_id: str | None
    artifact_key: str | None
    config_key: str | None
    manifest_key: str | None


class USStageRecord(TypedDict):
    """One stage entry in a US stage manifest."""

    id: str
    step: str
    title: str
    purpose: str
    status: USStageStatus
    lifecycleStatus: USStageLifecycleStatus
    outputManifest: str | None
    startedAt: str | None
    updatedAt: str | None
    completedAt: str | None
    failedAt: str | None
    deferredReason: str | None
    failure: USStageFailureRecord | None
    events: list[USStageRuntimeEventRecord]
    consumes: list[str]
    produces: list[str]
    inputs: list[USStageResourceRecord]
    outputs: list[USStageResourceRecord]
    artifacts: list[USStageArtifactRecord]
    diagnostics: list[str]
    validations: list[USStageValidationRecord]
    resume: USStageResumeRecord
    metrics: list[USStageMetric]


class USStageManifest(TypedDict):
    """Canonical saved-run stage manifest."""

    schemaVersion: int
    contractVersion: str
    generatedAt: str | None
    pipeline: str
    artifactRoot: str
    manifest: str
    stages: list[USStageRecord]


class USDataFlowStageSummary(TypedDict):
    """Stage summary embedded in the site-facing data-flow snapshot."""

    id: str
    step: str
    title: str
    summary: str
    status: USStageStatus
    metrics: list[USStageMetric]
    outputs: list[str]
    resumeMode: StageResumeMode


class USValidationEvidenceRecord(TypedDict):
    """One validation or benchmarking evidence sidecar."""

    key: str
    path: str
    exists: bool


class USValidationEvidenceManifest(TypedDict):
    """Stage 9 evidence index."""

    formatVersion: int
    stageId: str
    evidence: list[USValidationEvidenceRecord]
    summaries: dict[str, Any]


__all__ = [
    "SUPPORTED_US_STAGE_MANIFEST_SCHEMA_VERSIONS",
    "USDataFlowStageSummary",
    "US_POLICYENGINE_ENTITY_STAGE_ID",
    "US_STAGE_ARTIFACT_ROOT",
    "US_STAGE_MANIFEST_SCHEMA_VERSION",
    "US_VALIDATION_STAGE_ID",
    "USStageArtifactRecord",
    "USStageFailureRecord",
    "USStageLifecycleStatus",
    "USStageManifest",
    "USStageMetric",
    "USStageMetricValue",
    "USStageRecord",
    "USStageResourceRecord",
    "USStageResumeRecord",
    "USStageRuntimeEventRecord",
    "USStageStatus",
    "USStageValidationRecord",
    "USStageValidationStatus",
    "USValidationEvidenceManifest",
    "USValidationEvidenceRecord",
]
