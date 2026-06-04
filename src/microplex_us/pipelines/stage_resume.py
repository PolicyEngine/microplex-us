"""Preflight checks for resuming canonical US pipeline stages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microplex_us.pipelines.stage_contracts import (
    US_CANONICAL_STAGE_IDS,
    US_STAGE_CONTRACT_VERSION,
    canonicalize_us_pipeline_stage_id,
    get_us_pipeline_stage_contract,
    get_us_stage_artifact_contract,
)
from microplex_us.pipelines.stage_run import (
    resolve_us_manifest_or_contract_artifact_path,
)

_POLICYENGINE_ENTITY_BUNDLE_TABLES = (
    "households",
    "persons",
    "tax_units",
    "spm_units",
    "families",
    "marital_units",
)


@dataclass(frozen=True)
class USStageResumeArtifactRequirement:
    """One durable artifact required before a stage resume can start."""

    stage_id: str
    artifact_key: str
    reason: str

    @property
    def label(self) -> str:
        return f"{self.stage_id}.{self.artifact_key}"


@dataclass(frozen=True)
class USStageResumeMissingRequirement:
    """One missing input detected by the resume preflight."""

    label: str
    reason: str
    path: Path | None = None

    def format(self) -> str:
        path_text = f" ({self.path})" if self.path is not None else ""
        return f"{self.label}: {self.reason}{path_text}"


@dataclass(frozen=True)
class USStageResumePreflightResult:
    """Result of checking whether a saved run can resume at one stage."""

    artifact_root: Path
    resume_from_stage: str
    missing: tuple[USStageResumeMissingRequirement, ...]

    @property
    def ok(self) -> bool:
        return not self.missing

    def raise_for_missing(self) -> None:
        if self.ok:
            return
        raise ValueError(format_us_stage_resume_preflight_error(self))


def preflight_us_stage_resume(
    artifact_root: str | Path,
    resume_from_stage: str,
    *,
    extra_required_artifacts: tuple[USStageResumeArtifactRequirement, ...] = (),
) -> USStageResumePreflightResult:
    """Validate durable inputs before resuming a saved US stage run."""

    root = Path(artifact_root).expanduser()
    stage_id = canonicalize_us_pipeline_stage_id(resume_from_stage)
    if stage_id not in US_CANONICAL_STAGE_IDS:
        raise ValueError(f"Unknown US pipeline stage: {resume_from_stage}")

    missing: list[USStageResumeMissingRequirement] = []
    if not root.exists():
        missing.append(
            USStageResumeMissingRequirement(
                label="artifact_root",
                reason="saved run directory does not exist",
                path=root,
            )
        )
        return USStageResumePreflightResult(root, stage_id, tuple(missing))
    if not root.is_dir():
        missing.append(
            USStageResumeMissingRequirement(
                label="artifact_root",
                reason="saved run path is not a directory",
                path=root,
            )
        )
        return USStageResumePreflightResult(root, stage_id, tuple(missing))

    stage_index = US_CANONICAL_STAGE_IDS.index(stage_id)
    manifest = _load_json_if_available(root / "manifest.json")
    if manifest is None and stage_index > 0:
        missing.append(
            USStageResumeMissingRequirement(
                label="01_run_profile.manifest",
                reason="top-level artifact manifest is missing or unreadable",
                path=root / "manifest.json",
            )
        )

    if stage_index > 0:
        previous_stage_id = US_CANONICAL_STAGE_IDS[stage_index - 1]
        missing.extend(_missing_completed_stage_requirements(root, previous_stage_id))
        for resource in get_us_pipeline_stage_contract(stage_id).inputs:
            if (
                not resource.required
                or resource.stage_id is None
                or resource.kind not in {"artifact", "manifest", "stage_output"}
            ):
                continue
            missing.extend(
                _missing_stage_output_requirement(
                    root,
                    stage_id=resource.stage_id,
                    output_key=resource.key,
                    consumer_stage_id=stage_id,
                )
            )

    if manifest is not None:
        for requirement in extra_required_artifacts:
            path = resolve_us_manifest_or_contract_artifact_path(
                root,
                manifest,
                requirement.artifact_key,
                stage_id=requirement.stage_id,
            )
            if not path.exists():
                missing.append(
                    USStageResumeMissingRequirement(
                        label=requirement.label,
                        reason=requirement.reason,
                        path=path,
                    )
                )
            else:
                missing.extend(
                    _missing_artifact_format_requirements(
                        stage_id=requirement.stage_id,
                        artifact_key=requirement.artifact_key,
                        path=path,
                        label=requirement.label,
                    )
                )

    return USStageResumePreflightResult(root, stage_id, _dedupe_missing(missing))


def format_us_stage_resume_preflight_error(
    result: USStageResumePreflightResult,
) -> str:
    """Return a clear error message for a failed resume preflight."""

    details = "\n".join(f"- {item.format()}" for item in result.missing)
    return (
        "US pipeline resume preflight failed for "
        f"{result.resume_from_stage} at {result.artifact_root}. "
        "The rerun was not started because required durable inputs are missing:\n"
        f"{details}"
    )


def _missing_completed_stage_requirements(
    artifact_root: Path,
    stage_id: str,
) -> tuple[USStageResumeMissingRequirement, ...]:
    path = _stage_manifest_path(artifact_root, stage_id)
    payload = _load_json_if_available(path)
    if payload is None:
        return (
            USStageResumeMissingRequirement(
                label=f"{stage_id}.stage_manifest",
                reason="stage output manifest is missing or unreadable",
                path=path,
            ),
        )
    missing: list[USStageResumeMissingRequirement] = []
    if payload.get("contractVersion") != US_STAGE_CONTRACT_VERSION:
        missing.append(
            USStageResumeMissingRequirement(
                label=f"{stage_id}.contractVersion",
                reason=(
                    "stage output manifest uses stale contract version "
                    f"{payload.get('contractVersion')!r}; expected "
                    f"{US_STAGE_CONTRACT_VERSION!r}"
                ),
                path=path,
            )
        )
    if payload.get("lifecycleStatus") != "complete" or not payload.get("complete"):
        missing.append(
            USStageResumeMissingRequirement(
                label=f"{stage_id}.lifecycleStatus",
                reason="stage is not marked complete",
                path=path,
            )
        )
    for output_key in tuple(payload.get("requiredOutputs") or ()):
        missing.extend(
            _missing_stage_output_requirement(
                artifact_root,
                stage_id=stage_id,
                output_key=str(output_key),
                consumer_stage_id=None,
            )
        )
    return tuple(missing)


def _dedupe_missing(
    missing: list[USStageResumeMissingRequirement],
) -> tuple[USStageResumeMissingRequirement, ...]:
    deduped: list[USStageResumeMissingRequirement] = []
    seen: set[tuple[str, Path | None]] = set()
    for item in missing:
        key = (item.label, item.path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return tuple(deduped)


def _missing_stage_output_requirement(
    artifact_root: Path,
    *,
    stage_id: str,
    output_key: str,
    consumer_stage_id: str | None,
) -> tuple[USStageResumeMissingRequirement, ...]:
    path = _stage_manifest_path(artifact_root, stage_id)
    payload = _load_json_if_available(path)
    if payload is None:
        return (
            USStageResumeMissingRequirement(
                label=f"{stage_id}.{output_key}",
                reason="source stage manifest is missing or unreadable",
                path=path,
            ),
        )
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or output_key not in outputs:
        consumer = f" required by {consumer_stage_id}" if consumer_stage_id else ""
        return (
            USStageResumeMissingRequirement(
                label=f"{stage_id}.{output_key}",
                reason=f"required output is not recorded{consumer}",
                path=path,
            ),
        )
    value = outputs[output_key]
    missing = _missing_serialized_output_requirements(
        artifact_root,
        stage_id=stage_id,
        output_key=output_key,
        value=value,
    )
    if not missing:
        return ()
    consumer = f" required by {consumer_stage_id}" if consumer_stage_id else ""
    return tuple(
        USStageResumeMissingRequirement(
            label=item.label,
            reason=f"{item.reason}{consumer}",
            path=item.path or path,
        )
        for item in missing
    )


def _missing_serialized_output_requirements(
    artifact_root: Path,
    *,
    stage_id: str,
    output_key: str,
    value: Any,
) -> tuple[USStageResumeMissingRequirement, ...]:
    label = f"{stage_id}.{output_key}"
    if value is None:
        return (
            USStageResumeMissingRequirement(
                label=label,
                reason="required output is unavailable",
            ),
        )
    if isinstance(value, Mapping):
        path = _serialized_output_path(artifact_root, value)
        if path is not None:
            if not path.exists():
                return (
                    USStageResumeMissingRequirement(
                        label=label,
                        reason="required output is unavailable",
                        path=path,
                    ),
                )
            return _missing_artifact_format_requirements(
                stage_id=stage_id,
                artifact_key=output_key,
                path=path,
                label=label,
            )
        exists = value.get("exists")
        if exists is not None:
            if bool(exists):
                return ()
            return (
                USStageResumeMissingRequirement(
                    label=label,
                    reason="required output is unavailable",
                ),
            )
        if value:
            return ()
        return (
            USStageResumeMissingRequirement(
                label=label,
                reason="required output is unavailable",
            ),
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        if value:
            return ()
        return (
            USStageResumeMissingRequirement(
                label=label,
                reason="required output is unavailable",
            ),
        )
    if isinstance(value, str):
        if value:
            return ()
        return (
            USStageResumeMissingRequirement(
                label=label,
                reason="required output is unavailable",
            ),
        )
    return ()


def _missing_artifact_format_requirements(
    *,
    stage_id: str,
    artifact_key: str,
    path: Path,
    label: str,
) -> tuple[USStageResumeMissingRequirement, ...]:
    try:
        contract = get_us_stage_artifact_contract(stage_id, artifact_key)
    except KeyError:
        return ()
    if contract.format != "policyengine_entity_bundle":
        return ()
    metadata_path = path / "metadata.json" if path.is_dir() else path
    metadata = _load_json_if_available(metadata_path)
    if metadata is None:
        return (
            USStageResumeMissingRequirement(
                label=label,
                reason="PolicyEngine entity bundle metadata is missing or unreadable",
                path=metadata_path,
            ),
        )
    missing: list[USStageResumeMissingRequirement] = []
    for table_name in _POLICYENGINE_ENTITY_BUNDLE_TABLES:
        if metadata.get(table_name) is None:
            continue
        table_path = metadata_path.parent / f"{table_name}.parquet"
        if not table_path.exists():
            missing.append(
                USStageResumeMissingRequirement(
                    label=label,
                    reason=f"PolicyEngine entity bundle is missing {table_name}.parquet",
                    path=table_path,
                )
            )
    return tuple(missing)


def _serialized_output_path(artifact_root: Path, value: Any) -> Path | None:
    if not isinstance(value, Mapping):
        return None
    path_value = value.get("path")
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = artifact_root / path
    return path


def _stage_manifest_path(artifact_root: Path, stage_id: str) -> Path:
    return artifact_root / "stage_artifacts" / "manifests" / f"{stage_id}.json"


def _load_json_if_available(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


__all__ = [
    "USStageResumeArtifactRequirement",
    "USStageResumeMissingRequirement",
    "USStageResumePreflightResult",
    "format_us_stage_resume_preflight_error",
    "preflight_us_stage_resume",
]
