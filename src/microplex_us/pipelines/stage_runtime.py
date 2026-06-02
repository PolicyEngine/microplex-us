"""Live runtime writer for canonical US pipeline stage manifests."""

from __future__ import annotations

import json
import traceback
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from microplex_us.pipelines.stage_contracts import (
    US_CANONICAL_STAGE_IDS,
    US_STAGE_CONTRACT_VERSION,
    get_us_pipeline_stage_contract,
    get_us_stage_artifact_contract,
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.stage_manifest import write_us_stage_manifest
from microplex_us.pipelines.stage_manifest_types import (
    USStageFailureRecord,
    USStageLifecycleStatus,
    USStageRuntimeEventRecord,
)
from microplex_us.pipelines.stage_run import (
    USArtifactRef,
    USDiagnosticOutput,
    USStageInputOverride,
    USStageOutputManifest,
    USStageRunWriter,
    _serialize_value,
    build_us_stage_output_manifests_from_artifact_manifest,
)

RuntimeUpdateSection = Literal["outputs", "diagnostics", "metadata"]


class USStageRuntimeWriter:
    """Write stage manifests incrementally during a canonical US build."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        manifest_payload: Mapping[str, Any] | None = None,
        allow_stage_input_overrides: bool = False,
        stage_input_overrides: tuple[USStageInputOverride, ...] = (),
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.manifest_payload: dict[str, Any] = dict(manifest_payload or {})
        self.allow_stage_input_overrides = allow_stage_input_overrides
        self.stage_input_overrides = tuple(stage_input_overrides)
        self._run_writer = USStageRunWriter(
            self.artifact_root,
            manifest_payload=self.manifest_payload,
            allow_stage_input_overrides=allow_stage_input_overrides,
            stage_input_overrides=stage_input_overrides,
        )

    @property
    def recorded_stages(self) -> tuple[USStageOutputManifest, ...]:
        """Return completed typed stage manifests recorded by this writer."""

        return self._run_writer.recorded_stages

    def start_stage(
        self,
        stage_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark one stage as running after validating its previous stage seam."""

        self._validate_stage_id(stage_id)
        self._validate_start_transition(stage_id)
        now = _now()
        payload = self._stage_payload(stage_id)
        payload["complete"] = False
        payload["lifecycleStatus"] = "running"
        payload["startedAt"] = payload.get("startedAt") or now
        payload["updatedAt"] = now
        payload["completedAt"] = None
        payload["failedAt"] = None
        payload["deferredReason"] = None
        payload["failure"] = None
        payload["inputOverrides"] = self._serialized_overrides_for_stage(stage_id)
        payload["metadata"] = {
            **dict(payload.get("metadata", {})),
            **dict(metadata or {}),
        }
        payload["events"] = [
            *list(payload.get("events", [])),
            _event("stage_started", now, dict(metadata or {})),
        ]
        self._write_stage_payload(stage_id, payload)
        self._refresh_aggregate()
        return payload

    def update(
        self,
        stage_id: str,
        key: str,
        value: Any,
        *,
        section: RuntimeUpdateSection = "outputs",
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Update one manifest entry, optionally writing a JSON artifact first."""

        self._validate_stage_id(stage_id)
        if section == "outputs":
            self._validate_output_key(stage_id, key)
        payload = self._stage_payload(stage_id)
        written_value = value
        if path is not None:
            written_value = self._write_update_artifact(stage_id, key, value, path)
        bucket = dict(payload.get(section, {}))
        bucket[key] = _runtime_serialize(written_value, self.artifact_root)
        payload[section] = bucket
        now = _now()
        payload["updatedAt"] = now
        payload["events"] = [
            *list(payload.get("events", [])),
            _event("stage_updated", now, {"section": section, "key": key}),
        ]
        self._write_stage_payload(stage_id, payload)
        self._refresh_aggregate()
        return payload

    def record_output(
        self,
        stage_id: str,
        key: str,
        value: Any,
        *,
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Record one stage output entry."""

        return self.update(stage_id, key, value, section="outputs", path=path)

    def record_diagnostic(
        self,
        stage_id: str,
        diagnostic: USDiagnosticOutput,
    ) -> dict[str, Any]:
        """Record one diagnostic output for a running stage."""

        return self.update(
            stage_id,
            diagnostic.key,
            diagnostic,
            section="diagnostics",
        )

    def complete_stage(self, outputs: USStageOutputManifest) -> dict[str, Any]:
        """Validate, record, and write a complete typed stage output manifest."""

        self._validate_stage_id(outputs.stage_id)
        now = _now()
        existing = self._stage_payload(outputs.stage_id)
        stage_started_at = _optional_str(existing.get("startedAt")) or now
        existing_events = tuple(
            dict(event)
            for event in existing.get("events", ())
            if isinstance(event, dict)
        )
        input_stage_manifest = outputs.input_stage_manifest
        if input_stage_manifest is None:
            input_stage_manifest = self._previous_stage_manifest_ref(outputs.stage_id)
        lifecycle_outputs = replace(
            outputs,
            input_stage_manifest=input_stage_manifest,
            lifecycle_status="complete",
            started_at=stage_started_at,
            updated_at=now,
            completed_at=now,
            failed_at=None,
            deferred_reason=None,
            failure=None,
            events=(
                *existing_events,
                *tuple(outputs.events),
                _event("stage_completed", now),
            ),
        )
        self._run_writer.manifest_payload = self.manifest_payload
        self._run_writer.record_stage(lifecycle_outputs)
        payload = lifecycle_outputs.to_dict(
            self.artifact_root,
            input_stage_manifest=input_stage_manifest,
            input_overrides=self._input_overrides_for_stage(outputs.stage_id),
        )
        self._write_stage_payload(outputs.stage_id, payload)
        if outputs.stage_id == "08_dataset_assembly":
            self.manifest_payload = self._run_writer.write_manifest_files()
        else:
            self._refresh_aggregate()
        return payload

    def fail_stage(
        self,
        stage_id: str,
        error: BaseException,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark one stage as failed and persist the failure details."""

        self._validate_stage_id(stage_id)
        now = _now()
        payload = self._stage_payload(stage_id)
        failure: USStageFailureRecord = {
            "errorType": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        payload["complete"] = False
        payload["lifecycleStatus"] = "failed"
        payload["updatedAt"] = now
        payload["failedAt"] = now
        payload["failure"] = failure
        payload["metadata"] = {
            **dict(payload.get("metadata", {})),
            **dict(metadata or {}),
        }
        payload["events"] = [
            *list(payload.get("events", [])),
            _event("stage_failed", now, {"errorType": type(error).__name__}),
        ]
        self._write_stage_payload(stage_id, payload)
        self._refresh_aggregate()
        return payload

    def defer_stage(
        self,
        stage_id: str,
        reason: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark one stage as intentionally deferred."""

        self._validate_stage_id(stage_id)
        now = _now()
        payload = self._stage_payload(stage_id)
        payload["complete"] = False
        payload["lifecycleStatus"] = "deferred"
        payload["updatedAt"] = now
        payload["deferredReason"] = reason
        payload["metadata"] = {
            **dict(payload.get("metadata", {})),
            **dict(metadata or {}),
        }
        payload["events"] = [
            *list(payload.get("events", [])),
            _event("stage_deferred", now, {"reason": reason}),
        ]
        self._write_stage_payload(stage_id, payload)
        self._refresh_aggregate()
        return payload

    def finalize_from_artifact_manifest(
        self,
        manifest_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Finalize typed manifests from a completed saved artifact manifest."""

        self.manifest_payload = dict(manifest_payload)
        self._run_writer = USStageRunWriter(
            self.artifact_root,
            manifest_payload=self.manifest_payload,
            allow_stage_input_overrides=self.allow_stage_input_overrides,
            stage_input_overrides=self.stage_input_overrides,
        )
        for outputs in build_us_stage_output_manifests_from_artifact_manifest(
            self.artifact_root,
            self.manifest_payload,
        ):
            existing = self._stage_payload(outputs.stage_id)
            now = _now()
            lifecycle_status = _final_lifecycle_status(outputs)
            existing_events = tuple(
                dict(event)
                for event in existing.get("events", ())
                if isinstance(event, dict)
            )
            lifecycle_outputs = replace(
                outputs,
                input_stage_manifest=outputs.input_stage_manifest
                or self._previous_stage_manifest_ref(outputs.stage_id),
                lifecycle_status=lifecycle_status,
                started_at=_optional_str(existing.get("startedAt")) or now,
                updated_at=now,
                completed_at=now if lifecycle_status == "complete" else None,
                deferred_reason=(
                    outputs.deferred_reason if lifecycle_status == "deferred" else None
                ),
                events=(
                    *existing_events,
                    *tuple(outputs.events),
                    _event(f"stage_{lifecycle_status}", now),
                ),
            )
            self._run_writer.record_stage(lifecycle_outputs)
        self.manifest_payload = self._run_writer.write_manifest_files()
        return self.manifest_payload

    def _stage_payload(self, stage_id: str) -> dict[str, Any]:
        path = self._stage_output_manifest_path(stage_id)
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict):
                return _ensure_stage_payload_defaults(stage_id, payload)
        return _empty_stage_payload(stage_id)

    def _write_stage_payload(self, stage_id: str, payload: Mapping[str, Any]) -> None:
        path = self._stage_output_manifest_path(stage_id)
        _write_json_atomically(path, payload)
        self._register_stage_output_manifest(stage_id, path)

    def _register_stage_output_manifest(self, stage_id: str, path: Path) -> None:
        stage_paths = dict(self.manifest_payload.get("stage_output_manifests", {}))
        stage_paths[stage_id] = str(path.relative_to(self.artifact_root))
        self.manifest_payload["stage_output_manifests"] = stage_paths

    def _refresh_aggregate(self) -> None:
        stage_manifest_path = resolve_us_stage_artifact_contract_path(
            self.artifact_root,
            "08_dataset_assembly",
            "stage_manifest",
        )
        artifacts = dict(self.manifest_payload.get("artifacts", {}))
        artifacts.setdefault("stage_manifest", stage_manifest_path.name)
        artifacts.setdefault("manifest", "manifest.json")
        self.manifest_payload["artifacts"] = artifacts
        _write_json_atomically(
            self.artifact_root / "manifest.json", self.manifest_payload
        )
        write_us_stage_manifest(
            self.artifact_root,
            stage_manifest_path,
            manifest_payload=self.manifest_payload,
        )

    def _stage_output_manifest_path(self, stage_id: str) -> Path:
        return self.artifact_root / "stage_artifacts" / "manifests" / f"{stage_id}.json"

    def _previous_stage_manifest_ref(self, stage_id: str) -> str | None:
        stage_index = US_CANONICAL_STAGE_IDS.index(stage_id)
        if stage_index == 0:
            return None
        previous_stage_id = US_CANONICAL_STAGE_IDS[stage_index - 1]
        path = self._stage_output_manifest_path(previous_stage_id)
        return str(path.relative_to(self.artifact_root)) if path.exists() else None

    def _validate_start_transition(self, stage_id: str) -> None:
        stage_index = US_CANONICAL_STAGE_IDS.index(stage_id)
        if stage_index == 0:
            return
        previous_stage_id = US_CANONICAL_STAGE_IDS[stage_index - 1]
        previous_payload = self._stage_payload(previous_stage_id)
        if previous_payload.get("lifecycleStatus") == "complete":
            self._validate_completed_stage(previous_stage_id, previous_payload)
            self._validate_required_start_inputs(stage_id)
            return
        contract = get_us_pipeline_stage_contract(stage_id)
        required_previous_inputs = tuple(
            resource
            for resource in contract.inputs
            if resource.required and resource.stage_id == previous_stage_id
        )
        if required_previous_inputs and all(
            self._override_satisfies(stage_id, resource.key)
            for resource in required_previous_inputs
        ):
            self._validate_required_start_inputs(stage_id)
            return
        raise ValueError(
            f"{stage_id} requires {previous_stage_id} to be complete before start, "
            "unless explicit stage input overrides are enabled"
        )

    def _validate_required_start_inputs(self, stage_id: str) -> None:
        contract = get_us_pipeline_stage_contract(stage_id)
        missing_inputs: list[str] = []
        for resource in contract.inputs:
            if (
                not resource.required
                or resource.stage_id is None
                or resource.kind not in {"artifact", "manifest", "stage_output"}
                or self._override_satisfies(stage_id, resource.key)
            ):
                continue
            payload = self._stage_payload(resource.stage_id)
            if payload.get("lifecycleStatus") != "complete":
                missing_inputs.append(f"{resource.stage_id}.{resource.key}")
                continue
            self._validate_completed_stage(resource.stage_id, payload)
            outputs = payload.get("outputs")
            if not isinstance(outputs, Mapping) or not _serialized_output_is_available(
                outputs.get(resource.key)
            ):
                missing_inputs.append(f"{resource.stage_id}.{resource.key}")
        if missing_inputs:
            raise ValueError(
                f"{stage_id} is missing required stage input(s) before start: "
                f"{', '.join(missing_inputs)}"
            )

    def _validate_completed_stage(
        self,
        stage_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if payload.get("contractVersion") != US_STAGE_CONTRACT_VERSION:
            raise ValueError(
                f"{stage_id} uses stale contract version "
                f"{payload.get('contractVersion')!r}; expected "
                f"{US_STAGE_CONTRACT_VERSION!r}"
            )
        missing = tuple(payload.get("missingRequiredOutputs") or ())
        if missing:
            raise ValueError(
                f"{stage_id} is complete but missing required outputs: "
                f"{', '.join(str(item) for item in missing)}"
            )
        outputs = payload.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ValueError(f"{stage_id} has no serialized outputs")
        required_outputs = tuple(payload.get("requiredOutputs") or ())
        for key in required_outputs:
            if not _serialized_output_is_available(outputs.get(str(key))):
                raise ValueError(
                    f"{stage_id} is complete but required output {key!r} is unavailable"
                )

    def _override_satisfies(self, stage_id: str, key: str) -> bool:
        if not self.allow_stage_input_overrides:
            return False
        return any(
            override.stage_id == stage_id and override.key == key
            for override in self.stage_input_overrides
        )

    def _serialized_overrides_for_stage(self, stage_id: str) -> list[dict[str, Any]]:
        return [
            override.to_dict(self.artifact_root)
            for override in self._input_overrides_for_stage(stage_id)
        ]

    def _input_overrides_for_stage(
        self,
        stage_id: str,
    ) -> tuple[USStageInputOverride, ...]:
        return tuple(
            override
            for override in self.stage_input_overrides
            if override.stage_id == stage_id
        )

    def _validate_output_key(self, stage_id: str, key: str) -> None:
        contract = get_us_pipeline_stage_contract(stage_id)
        valid_keys = {resource.key for resource in contract.outputs}
        valid_keys.update(artifact.key for artifact in contract.artifacts)
        if key not in valid_keys:
            valid = ", ".join(sorted(valid_keys)) or "none"
            raise KeyError(f"Unknown output key {stage_id}.{key}; valid keys: {valid}")

    def _write_update_artifact(
        self,
        stage_id: str,
        key: str,
        value: Any,
        path: str | Path,
    ) -> USArtifactRef:
        resolved_path = Path(path)
        if not resolved_path.is_absolute():
            resolved_path = self.artifact_root / resolved_path
        _write_json_atomically(
            resolved_path, _runtime_serialize(value, self.artifact_root)
        )
        artifact_contract = get_us_stage_artifact_contract(stage_id, key)
        return USArtifactRef(
            key=key,
            path=resolved_path,
            format=artifact_contract.format,
            required=artifact_contract.required,
            resume_role=artifact_contract.resume_role,
            exists=True,
        )

    @staticmethod
    def _validate_stage_id(stage_id: str) -> None:
        if stage_id not in US_CANONICAL_STAGE_IDS:
            raise KeyError(f"Unknown US pipeline stage: {stage_id}")


def _empty_stage_payload(stage_id: str) -> dict[str, Any]:
    contract = get_us_pipeline_stage_contract(stage_id)
    return {
        "schemaVersion": 2,
        "contractVersion": US_STAGE_CONTRACT_VERSION,
        "stageId": stage_id,
        "complete": False,
        "lifecycleStatus": "pending",
        "startedAt": None,
        "updatedAt": None,
        "completedAt": None,
        "failedAt": None,
        "deferredReason": None,
        "failure": None,
        "inputStageManifest": None,
        "inputOverrides": [],
        "requiredOutputs": [
            resource.key for resource in contract.outputs if resource.required
        ],
        "missingRequiredOutputs": [
            resource.key for resource in contract.outputs if resource.required
        ],
        "outputs": {},
        "diagnostics": {
            "stage_summary": USDiagnosticOutput(
                key="stage_summary",
                description=f"Runtime diagnostic summary for {stage_id}.",
            ).to_dict(),
        },
        "auxiliaryArtifacts": {},
        "metadata": {},
        "events": [],
    }


def _ensure_stage_payload_defaults(
    stage_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    defaults = _empty_stage_payload(stage_id)
    merged = {**defaults, **payload}
    for key in ("outputs", "diagnostics", "auxiliaryArtifacts", "metadata"):
        if not isinstance(merged.get(key), dict):
            merged[key] = {}
    if not isinstance(merged.get("events"), list):
        merged["events"] = []
    return merged


def _final_lifecycle_status(
    outputs: USStageOutputManifest,
) -> USStageLifecycleStatus:
    if outputs.resolved_lifecycle_status() == "deferred":
        return "deferred"
    return "complete" if outputs.complete else "pending"


def _runtime_serialize(value: Any, artifact_root: str | Path | None) -> Any:
    if isinstance(value, USDiagnosticOutput):
        return value.to_dict(artifact_root)
    return _serialize_value(value, artifact_root)


def _serialized_output_is_available(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        exists = value.get("exists")
        if exists is not None:
            return bool(exists)
        return bool(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value)
    if isinstance(value, str):
        return bool(value)
    return True


def _event(
    event: str,
    timestamp: str,
    details: Mapping[str, Any] | None = None,
) -> USStageRuntimeEventRecord:
    return {
        "event": event,
        "timestamp": timestamp,
        "details": dict(details or {}),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _write_json_atomically(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


__all__ = [
    "RuntimeUpdateSection",
    "USStageRuntimeWriter",
]
