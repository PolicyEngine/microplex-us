"""Shared stage-run writer for US Microplex saved-run manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

from microplex_us.pipelines.data_flow_snapshot import (
    write_us_microplex_data_flow_snapshot,
)
from microplex_us.pipelines.stage_artifacts import (
    build_us_stage_artifact_inventory,
    write_us_stage_artifact_inventory,
)
from microplex_us.pipelines.stage_contracts import (
    US_CANONICAL_STAGE_IDS,
    US_STAGE_CONTRACT_VERSION,
    StageArtifactFormat,
    StageArtifactResumeRole,
    get_us_pipeline_stage_contract,
    get_us_stage_artifact_contract,
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.stage_manifest import (
    write_us_stage_manifest,
    write_us_validation_evidence_manifest,
)
from microplex_us.pipelines.stage_readiness import (
    write_us_conditional_readiness_report,
)

US_STAGE_OUTPUT_MANIFEST_SCHEMA_VERSION = 1

USArtifactCategory = Literal[
    "required_output",
    "diagnostic",
    "auxiliary",
    "derived",
]


@dataclass(frozen=True)
class USArtifactRef:
    """Reference to one artifact owned by a stage output manifest."""

    key: str
    path: str | Path
    format: StageArtifactFormat = "unknown"
    required: bool = False
    category: USArtifactCategory = "required_output"
    resume_role: StageArtifactResumeRole | None = None
    assume_exists: bool = False
    exists: bool | None = None

    def resolved_path(self, artifact_root: str | Path) -> Path:
        path = Path(self.path)
        if not path.is_absolute():
            path = Path(artifact_root) / path
        return path

    def exists_under(self, artifact_root: str | Path) -> bool:
        if self.assume_exists:
            return True
        if self.exists is not None:
            return self.exists
        return self.resolved_path(artifact_root).exists()

    def relative_path(self, artifact_root: str | Path) -> str:
        path = self.resolved_path(artifact_root)
        try:
            return str(path.relative_to(Path(artifact_root)))
        except ValueError:
            return str(path)

    def to_dict(self, artifact_root: str | Path | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = (
            self.relative_path(artifact_root)
            if artifact_root is not None
            else str(self.path)
        )
        if artifact_root is not None:
            payload["exists"] = self.exists_under(artifact_root)
        return payload


@dataclass(frozen=True)
class USAuxiliaryArtifact:
    """Optional artifact declared by a stage contract."""

    key: str
    path: str | Path
    format: StageArtifactFormat = "unknown"
    description: str = ""
    assume_exists: bool = False

    def as_artifact_ref(self) -> USArtifactRef:
        return USArtifactRef(
            key=self.key,
            path=self.path,
            format=self.format,
            category="auxiliary",
            assume_exists=self.assume_exists,
        )


@dataclass(frozen=True)
class USDiagnosticOutput:
    """Diagnostic output exposed by a stage manifest."""

    key: str
    description: str = ""
    path: str | Path | None = None
    summary: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, artifact_root: str | Path | None = None) -> dict[str, Any]:
        path = None
        if self.path is not None:
            resolved = Path(self.path)
            if artifact_root is not None and not resolved.is_absolute():
                resolved = Path(artifact_root) / resolved
            if artifact_root is not None:
                try:
                    path = str(resolved.relative_to(Path(artifact_root)))
                except ValueError:
                    path = str(resolved)
            else:
                path = str(self.path)
        return {
            "key": self.key,
            "description": self.description,
            "path": path,
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class USStageInputOverride:
    """Explicit override for a stage input that is not provided by the prior stage."""

    stage_id: str
    key: str
    path: str | Path
    reason: str | None = None

    def to_dict(self, artifact_root: str | Path | None = None) -> dict[str, Any]:
        path = Path(self.path)
        path_text = str(path)
        if artifact_root is not None and not path.is_absolute():
            path_text = str(path)
        return {
            "stageId": self.stage_id,
            "key": self.key,
            "path": path_text,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class USStageOutputManifest:
    """Base type for one typed stage output manifest."""

    schema_version: int = US_STAGE_OUTPUT_MANIFEST_SCHEMA_VERSION
    contract_version: str = US_STAGE_CONTRACT_VERSION
    input_stage_manifest: str | Path | None = None
    diagnostics: Mapping[str, USDiagnosticOutput] = field(default_factory=dict)
    auxiliary_artifacts: Mapping[str, USAuxiliaryArtifact] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    complete: bool = True
    stage_id: str = field(default="", init=False)

    def required_output_keys(self) -> tuple[str, ...]:
        """Return required output keys from the canonical stage contract."""

        contract = get_us_pipeline_stage_contract(self.stage_id)
        return tuple(resource.key for resource in contract.outputs if resource.required)

    def artifact_refs(self) -> dict[str, USArtifactRef]:
        """Return artifact references carried by this stage output manifest."""

        refs: dict[str, USArtifactRef] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, USArtifactRef):
                refs[value.key] = value
        for artifact in self.auxiliary_artifacts.values():
            refs[artifact.key] = artifact.as_artifact_ref()
        return refs

    def missing_required_outputs(self, artifact_root: str | Path) -> tuple[str, ...]:
        """Return required output keys not provided or not present on disk."""

        missing: list[str] = []
        for key in self.required_output_keys():
            value = getattr(self, key, None)
            if _required_output_is_missing(value, artifact_root):
                missing.append(key)
        return tuple(missing)

    def to_dict(
        self,
        artifact_root: str | Path | None = None,
        *,
        input_stage_manifest: str | None = None,
        input_overrides: tuple[USStageInputOverride, ...] = (),
    ) -> dict[str, Any]:
        """Serialize this typed output manifest."""

        diagnostics = {
            key: diagnostic.to_dict(artifact_root)
            for key, diagnostic in self.diagnostics.items()
        }
        auxiliary = {
            key: artifact.as_artifact_ref().to_dict(artifact_root)
            for key, artifact in self.auxiliary_artifacts.items()
        }
        output_fields = {
            item.name: _serialize_value(getattr(self, item.name), artifact_root)
            for item in fields(self)
            if item.name
            not in {
                "schema_version",
                "contract_version",
                "input_stage_manifest",
                "diagnostics",
                "auxiliary_artifacts",
                "metadata",
                "complete",
                "stage_id",
            }
        }
        return {
            "schemaVersion": self.schema_version,
            "contractVersion": self.contract_version,
            "stageId": self.stage_id,
            "complete": self.complete,
            "inputStageManifest": input_stage_manifest
            or _optional_str(self.input_stage_manifest),
            "inputOverrides": [
                override.to_dict(artifact_root) for override in input_overrides
            ],
            "requiredOutputs": list(self.required_output_keys()),
            "missingRequiredOutputs": (
                list(self.missing_required_outputs(artifact_root))
                if artifact_root is not None
                else []
            ),
            "outputs": output_fields,
            "diagnostics": diagnostics,
            "auxiliaryArtifacts": auxiliary,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class USRunProfileOutputs(USStageOutputManifest):
    stage_id: str = field(default="01_run_profile", init=False)
    manifest: USArtifactRef | None = None
    resolved_config: Mapping[str, Any] = field(default_factory=dict)
    provider_query_plan: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class USSourceLoadingOutputs(USStageOutputManifest):
    stage_id: str = field(default="02_source_loading", init=False)
    observation_frame_summary: Mapping[str, Any] = field(default_factory=dict)
    source_descriptors: tuple[str, ...] = ()
    source_relationships: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class USSourcePlanningOutputs(USStageOutputManifest):
    stage_id: str = field(default="03_source_planning", init=False)
    source_plan: USArtifactRef | None = None
    scaffold_selection: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class USSeedScaffoldOutputs(USStageOutputManifest):
    stage_id: str = field(default="04_seed_scaffold", init=False)
    scaffold_seed_data: USArtifactRef | None = None
    seed_schema_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class USDonorSynthesisOutputs(USStageOutputManifest):
    stage_id: str = field(default="05_donor_integration_synthesis", init=False)
    seed_data: USArtifactRef | None = None
    synthetic_data: USArtifactRef | None = None
    synthesis_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_weight_diagnostics: USArtifactRef | None = None


@dataclass(frozen=True)
class USPolicyEngineEntityOutputs(USStageOutputManifest):
    stage_id: str = field(default="06_policyengine_entities", init=False)
    policyengine_entity_tables: USArtifactRef | None = None
    materialized_policyengine_inputs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class USCalibrationOutputs(USStageOutputManifest):
    stage_id: str = field(default="07_calibration", init=False)
    calibrated_data: USArtifactRef | None = None
    targets: USArtifactRef | None = None
    calibration_summary: USArtifactRef | None = None
    target_ledger: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class USDatasetAssemblyOutputs(USStageOutputManifest):
    stage_id: str = field(default="08_dataset_assembly", init=False)
    policyengine_dataset: USArtifactRef | None = None
    stage_manifest: USArtifactRef | None = None
    data_flow_snapshot: USArtifactRef | None = None
    artifact_inventory: USArtifactRef | None = None
    conditional_readiness: USArtifactRef | None = None


@dataclass(frozen=True)
class USValidationBenchmarkingOutputs(USStageOutputManifest):
    stage_id: str = field(default="09_validation_benchmarking", init=False)
    validation_evidence: USArtifactRef | None = None
    benchmark_summary: Mapping[str, Any] = field(default_factory=dict)
    policyengine_harness: USArtifactRef | None = None
    policyengine_native_scores: USArtifactRef | None = None
    policyengine_native_audit: USArtifactRef | None = None
    imputation_ablation: USArtifactRef | None = None
    child_tax_unit_agi_drift: USArtifactRef | None = None


US_STAGE_OUTPUT_MANIFEST_TYPES: dict[str, type[USStageOutputManifest]] = {
    "01_run_profile": USRunProfileOutputs,
    "02_source_loading": USSourceLoadingOutputs,
    "03_source_planning": USSourcePlanningOutputs,
    "04_seed_scaffold": USSeedScaffoldOutputs,
    "05_donor_integration_synthesis": USDonorSynthesisOutputs,
    "06_policyengine_entities": USPolicyEngineEntityOutputs,
    "07_calibration": USCalibrationOutputs,
    "08_dataset_assembly": USDatasetAssemblyOutputs,
    "09_validation_benchmarking": USValidationBenchmarkingOutputs,
}


class USStageRunWriter:
    """Validate and write typed US stage output manifests as one run."""

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
        if self.stage_input_overrides and not self.allow_stage_input_overrides:
            raise ValueError(
                "Stage input overrides require allow_stage_input_overrides=True"
            )
        for override in self.stage_input_overrides:
            _validate_us_stage_input_override(override)
        self._recorded: dict[str, USStageOutputManifest] = {}

    @property
    def recorded_stages(self) -> tuple[USStageOutputManifest, ...]:
        """Return recorded stages in canonical order."""

        return tuple(
            self._recorded[stage_id]
            for stage_id in US_CANONICAL_STAGE_IDS
            if stage_id in self._recorded
        )

    def update(self, outputs: USStageOutputManifest) -> None:
        """Record one whole typed stage output manifest."""

        self.record_stage(outputs)

    def record_stage(self, outputs: USStageOutputManifest) -> None:
        """Validate and record one whole typed stage output manifest."""

        self.validate_stage(outputs)
        self.validate_transition(outputs)
        self._recorded[outputs.stage_id] = outputs

    def validate_stage(self, outputs: USStageOutputManifest) -> None:
        """Validate one typed stage output manifest against its contract."""

        expected_type = US_STAGE_OUTPUT_MANIFEST_TYPES.get(outputs.stage_id)
        if expected_type is None:
            raise KeyError(f"Unknown US stage output manifest: {outputs.stage_id}")
        if not isinstance(outputs, expected_type):
            raise TypeError(
                f"{outputs.stage_id} must use {expected_type.__name__}, "
                f"got {type(outputs).__name__}"
            )
        get_us_pipeline_stage_contract(outputs.stage_id)
        if not outputs.diagnostics:
            raise ValueError(f"{outputs.stage_id} does not expose diagnostics")
        missing = outputs.missing_required_outputs(self.artifact_root)
        if outputs.complete and missing:
            raise ValueError(
                f"{outputs.stage_id} is marked complete but is missing required "
                f"outputs: {', '.join(missing)}"
            )
        contract_artifact_keys = {
            artifact.key
            for artifact in get_us_pipeline_stage_contract(outputs.stage_id).artifacts
        }
        for artifact in outputs.auxiliary_artifacts.values():
            if artifact.key not in contract_artifact_keys:
                raise KeyError(
                    f"{outputs.stage_id} auxiliary artifact {artifact.key!r} "
                    "is not declared by the stage contract"
                )
        for artifact in outputs.artifact_refs().values():
            if artifact.key not in contract_artifact_keys:
                raise KeyError(
                    f"{outputs.stage_id} artifact {artifact.key!r} is not declared "
                    "by the stage contract"
                )

    def validate_transition(self, outputs: USStageOutputManifest) -> None:
        """Validate that a stage consumes the previous stage output manifest."""

        stage_index = US_CANONICAL_STAGE_IDS.index(outputs.stage_id)
        if stage_index == 0:
            return
        previous_stage_id = US_CANONICAL_STAGE_IDS[stage_index - 1]
        if previous_stage_id in self._recorded:
            return
        if outputs.input_stage_manifest is not None:
            path = Path(outputs.input_stage_manifest)
            if not path.is_absolute():
                path = self.artifact_root / path
            if (
                path == self._stage_output_manifest_path(previous_stage_id)
                and path.exists()
            ):
                return
        if self.allow_stage_input_overrides and self._overrides_for_stage(
            outputs.stage_id
        ):
            return
        raise ValueError(
            f"{outputs.stage_id} requires {previous_stage_id} output manifest "
            "or an explicit stage input override"
        )

    def write_manifest_files(self) -> dict[str, Any]:
        """Write per-stage manifests and derived aggregate run manifests."""

        self.artifact_root.mkdir(parents=True, exist_ok=True)
        manifest = self._materialize_manifest_payload()
        stage_manifest_path = resolve_us_stage_artifact_contract_path(
            self.artifact_root,
            "08_dataset_assembly",
            "stage_manifest",
        )
        data_flow_snapshot_path = resolve_us_stage_artifact_contract_path(
            self.artifact_root,
            "08_dataset_assembly",
            "data_flow_snapshot",
        )
        artifact_inventory_path = resolve_us_stage_artifact_contract_path(
            self.artifact_root,
            "08_dataset_assembly",
            "artifact_inventory",
        )
        conditional_readiness_path = resolve_us_stage_artifact_contract_path(
            self.artifact_root,
            "08_dataset_assembly",
            "conditional_readiness",
        )
        manifest_path = resolve_us_stage_artifact_contract_path(
            self.artifact_root,
            "01_run_profile",
            "manifest",
        )
        validation_evidence_name = dict(manifest.get("artifacts", {})).get(
            "validation_evidence"
        )

        _write_json_atomically(manifest_path, manifest)
        if validation_evidence_name:
            validation_evidence_path = self._resolve_path(validation_evidence_name)
            write_us_validation_evidence_manifest(
                self.artifact_root,
                validation_evidence_path,
                manifest_payload=manifest,
            )
        write_us_microplex_data_flow_snapshot(
            self.artifact_root,
            data_flow_snapshot_path,
            manifest_payload=manifest,
            assume_existing_stage_artifact_keys=(
                "stage_manifest",
                "artifact_inventory",
                "conditional_readiness",
            ),
        )
        write_us_stage_manifest(
            self.artifact_root,
            stage_manifest_path,
            manifest_payload=manifest,
            assume_existing_artifact_keys=(
                "artifact_inventory",
                "conditional_readiness",
            ),
        )
        readiness_inventory = build_us_stage_artifact_inventory(
            self.artifact_root,
            manifest_payload=manifest,
            assume_existing_artifact_keys=(
                "artifact_inventory",
                "conditional_readiness",
            ),
        )
        write_us_conditional_readiness_report(
            self.artifact_root,
            conditional_readiness_path,
            manifest_payload=manifest,
            artifact_inventory=readiness_inventory,
        )
        write_us_stage_artifact_inventory(
            self.artifact_root,
            artifact_inventory_path,
            manifest_payload=manifest,
            assume_existing_artifact_keys=("artifact_inventory",),
        )
        return manifest

    def _materialize_manifest_payload(self) -> dict[str, Any]:
        manifest = dict(self.manifest_payload)
        artifacts = dict(manifest.get("artifacts", {}))
        stage_manifest_paths: dict[str, str] = {}

        for stage_id in US_CANONICAL_STAGE_IDS:
            outputs = self._recorded.get(stage_id)
            if outputs is None:
                continue
            stage_manifest_path = self._stage_output_manifest_path(stage_id)
            stage_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            stage_manifest_paths[stage_id] = str(
                stage_manifest_path.relative_to(self.artifact_root)
            )
            for artifact in outputs.artifact_refs().values():
                artifacts[artifact.key] = artifact.relative_path(self.artifact_root)

        self._ensure_aggregate_artifact_paths(artifacts)
        manifest["artifacts"] = artifacts
        manifest["stage_output_manifests"] = stage_manifest_paths
        manifest.setdefault("diagnostics", {})
        for stage_id, outputs in self._recorded.items():
            manifest["diagnostics"].setdefault(
                stage_id,
                {
                    key: diagnostic.to_dict(self.artifact_root)
                    for key, diagnostic in outputs.diagnostics.items()
                },
            )
        for stage_id, outputs in self._recorded.items():
            stage_manifest_path = self._stage_output_manifest_path(stage_id)
            _write_json_atomically(
                stage_manifest_path,
                outputs.to_dict(
                    self.artifact_root,
                    input_stage_manifest=self._previous_stage_manifest_ref(stage_id),
                    input_overrides=self._overrides_for_stage(stage_id),
                ),
            )
        self.manifest_payload = manifest
        return manifest

    def _ensure_aggregate_artifact_paths(self, artifacts: dict[str, Any]) -> None:
        artifacts.setdefault(
            "stage_manifest",
            resolve_us_stage_artifact_contract_path(
                self.artifact_root,
                "08_dataset_assembly",
                "stage_manifest",
            ).name,
        )
        artifacts.setdefault(
            "data_flow_snapshot",
            resolve_us_stage_artifact_contract_path(
                self.artifact_root,
                "08_dataset_assembly",
                "data_flow_snapshot",
            ).name,
        )
        artifacts.setdefault(
            "artifact_inventory",
            str(
                resolve_us_stage_artifact_contract_path(
                    self.artifact_root,
                    "08_dataset_assembly",
                    "artifact_inventory",
                ).relative_to(self.artifact_root)
            ),
        )
        artifacts.setdefault(
            "conditional_readiness",
            str(
                resolve_us_stage_artifact_contract_path(
                    self.artifact_root,
                    "08_dataset_assembly",
                    "conditional_readiness",
                ).relative_to(self.artifact_root)
            ),
        )

    def _stage_output_manifest_path(self, stage_id: str) -> Path:
        return self.artifact_root / "stage_artifacts" / "manifests" / f"{stage_id}.json"

    def _previous_stage_manifest_ref(self, stage_id: str) -> str | None:
        stage_index = US_CANONICAL_STAGE_IDS.index(stage_id)
        if stage_index == 0:
            return None
        previous_stage_id = US_CANONICAL_STAGE_IDS[stage_index - 1]
        if previous_stage_id not in self._recorded:
            return None
        return str(
            self._stage_output_manifest_path(previous_stage_id).relative_to(
                self.artifact_root
            )
        )

    def _overrides_for_stage(self, stage_id: str) -> tuple[USStageInputOverride, ...]:
        return tuple(
            override
            for override in self.stage_input_overrides
            if override.stage_id == stage_id
        )

    def _resolve_path(self, value: Any) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.artifact_root / path
        return path


def build_us_stage_output_manifests_from_artifact_manifest(
    artifact_root: str | Path,
    manifest_payload: Mapping[str, Any],
) -> tuple[USStageOutputManifest, ...]:
    """Build typed stage output manifests from an existing artifact manifest."""

    root = Path(artifact_root)
    manifest = dict(manifest_payload)
    synthesis = dict(manifest.get("synthesis", {}))
    rows = dict(manifest.get("rows", {}))
    config = dict(manifest.get("config", {}))
    artifacts = dict(manifest.get("artifacts", {}))
    source_names = tuple(
        str(source)
        for source in synthesis.get("source_names", ())
        if isinstance(source, str)
    )
    benchmark_summary = _benchmark_summary(manifest)
    has_benchmark = bool(benchmark_summary)
    has_dataset = (
        _artifact_ref(root, artifacts, "policyengine_dataset", "08_dataset_assembly")
        is not None
    )
    return (
        USRunProfileOutputs(
            manifest=_artifact_ref(
                root,
                {"manifest": artifacts.get("manifest", "manifest.json")},
                "manifest",
                "01_run_profile",
                assume_exists=True,
            ),
            resolved_config=config,
            provider_query_plan={"source_names": list(source_names)},
            diagnostics=_diagnostics("01_run_profile", manifest),
            complete=bool(config),
        ),
        USSourceLoadingOutputs(
            observation_frame_summary={"source_count": len(source_names)},
            source_descriptors=source_names,
            source_relationships={"status": "summarized"},
            diagnostics=_diagnostics("02_source_loading", manifest),
            complete=bool(source_names),
        ),
        USSourcePlanningOutputs(
            source_plan=_artifact_ref(
                root, artifacts, "source_plan", "03_source_planning"
            ),
            scaffold_selection={"scaffold_source": synthesis.get("scaffold_source")},
            diagnostics=_diagnostics("03_source_planning", manifest),
            complete=_artifact_exists(root, artifacts, "source_plan"),
        ),
        USSeedScaffoldOutputs(
            scaffold_seed_data=_artifact_ref(
                root,
                artifacts,
                "scaffold_seed_data",
                "04_seed_scaffold",
            ),
            seed_schema_metadata={"seed_rows": rows.get("seed")},
            diagnostics=_diagnostics("04_seed_scaffold", manifest),
            complete=_artifact_exists(root, artifacts, "scaffold_seed_data"),
        ),
        USDonorSynthesisOutputs(
            seed_data=_artifact_ref(
                root,
                artifacts,
                "seed_data",
                "05_donor_integration_synthesis",
            ),
            synthetic_data=_artifact_ref(
                root,
                artifacts,
                "synthetic_data",
                "05_donor_integration_synthesis",
            ),
            synthesis_metadata=synthesis,
            source_weight_diagnostics=_artifact_ref(
                root,
                artifacts,
                "source_weight_diagnostics",
                "05_donor_integration_synthesis",
                category="diagnostic",
            ),
            diagnostics=_diagnostics("05_donor_integration_synthesis", manifest),
            complete=all(
                _artifact_exists(root, artifacts, key)
                for key in ("seed_data", "synthetic_data")
            ),
        ),
        USPolicyEngineEntityOutputs(
            policyengine_entity_tables=_artifact_ref(
                root,
                artifacts,
                "policyengine_entity_tables",
                "06_policyengine_entities",
            ),
            materialized_policyengine_inputs=_policyengine_entity_metadata_summary(
                root,
                artifacts,
            ),
            diagnostics=_diagnostics("06_policyengine_entities", manifest),
            complete=_artifact_exists(root, artifacts, "policyengine_entity_tables"),
        ),
        USCalibrationOutputs(
            calibrated_data=_artifact_ref(
                root, artifacts, "calibrated_data", "07_calibration"
            ),
            targets=_artifact_ref(root, artifacts, "targets", "07_calibration"),
            calibration_summary=_artifact_ref(
                root,
                artifacts,
                "calibration_summary",
                "07_calibration",
                category="diagnostic",
            ),
            target_ledger={"target_count": manifest.get("targets", {})},
            diagnostics=_diagnostics("07_calibration", manifest),
            complete=all(
                _artifact_exists(root, artifacts, key)
                for key in ("calibrated_data", "targets", "calibration_summary")
            ),
        ),
        USDatasetAssemblyOutputs(
            policyengine_dataset=_artifact_ref(
                root,
                artifacts,
                "policyengine_dataset",
                "08_dataset_assembly",
            ),
            stage_manifest=_derived_artifact_ref(
                root, "stage_manifest", "08_dataset_assembly"
            ),
            data_flow_snapshot=_derived_artifact_ref(
                root,
                "data_flow_snapshot",
                "08_dataset_assembly",
            ),
            artifact_inventory=_derived_artifact_ref(
                root,
                "artifact_inventory",
                "08_dataset_assembly",
            ),
            conditional_readiness=_derived_artifact_ref(
                root,
                "conditional_readiness",
                "08_dataset_assembly",
            ),
            diagnostics=_diagnostics("08_dataset_assembly", manifest),
            complete=bool(has_dataset),
        ),
        USValidationBenchmarkingOutputs(
            validation_evidence=(
                _derived_artifact_ref(
                    root,
                    "validation_evidence",
                    "09_validation_benchmarking",
                )
                if has_dataset or has_benchmark
                else None
            ),
            benchmark_summary=benchmark_summary,
            policyengine_harness=_artifact_ref(
                root,
                artifacts,
                "policyengine_harness",
                "09_validation_benchmarking",
                category="diagnostic",
            ),
            policyengine_native_scores=_artifact_ref(
                root,
                artifacts,
                "policyengine_native_scores",
                "09_validation_benchmarking",
                category="diagnostic",
            ),
            policyengine_native_audit=_artifact_ref(
                root,
                artifacts,
                "policyengine_native_audit",
                "09_validation_benchmarking",
                category="diagnostic",
            ),
            imputation_ablation=_artifact_ref(
                root,
                artifacts,
                "imputation_ablation",
                "09_validation_benchmarking",
                category="diagnostic",
            ),
            child_tax_unit_agi_drift=_artifact_ref(
                root,
                artifacts,
                "child_tax_unit_agi_drift",
                "09_validation_benchmarking",
                category="diagnostic",
            ),
            diagnostics=_diagnostics("09_validation_benchmarking", manifest),
            complete=bool(has_benchmark),
        ),
    )


def write_us_stage_run_manifests_from_artifact_manifest(
    artifact_root: str | Path,
    manifest_payload: Mapping[str, Any],
    *,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
) -> dict[str, Any]:
    """Write typed stage manifests and aggregate outputs from an artifact manifest."""

    writer = USStageRunWriter(
        artifact_root,
        manifest_payload=manifest_payload,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )
    for outputs in build_us_stage_output_manifests_from_artifact_manifest(
        artifact_root,
        manifest_payload,
    ):
        writer.record_stage(outputs)
    return writer.write_manifest_files()


def resolve_us_manifest_or_contract_artifact_path(
    artifact_root: str | Path,
    manifest_payload: Mapping[str, Any],
    artifact_key: str,
    *,
    stage_id: str,
) -> Path:
    """Resolve an artifact from the manifest first, then the stage contract."""

    artifacts = dict(manifest_payload.get("artifacts", {}))
    declared = artifacts.get(artifact_key)
    if declared is not None:
        path = Path(str(declared))
        if not path.is_absolute():
            path = Path(artifact_root) / path
        return path
    return resolve_us_stage_artifact_contract_path(
        artifact_root, stage_id, artifact_key
    )


def parse_us_stage_input_override(value: str) -> USStageInputOverride:
    """Parse STAGE_ID.KEY=PATH into a stage input override."""

    if "=" not in value:
        raise ValueError("Stage input overrides must use STAGE_ID.KEY=PATH syntax")
    left, path = value.split("=", 1)
    if "." not in left:
        raise ValueError("Stage input overrides must use STAGE_ID.KEY=PATH syntax")
    stage_id, key = left.split(".", 1)
    if not stage_id or not key or not path:
        raise ValueError("Stage input overrides must use STAGE_ID.KEY=PATH syntax")
    if stage_id not in US_CANONICAL_STAGE_IDS:
        raise ValueError(f"Unknown US pipeline stage: {stage_id}")
    override = USStageInputOverride(stage_id=stage_id, key=key, path=path)
    _validate_us_stage_input_override(override)
    return override


def _validate_us_stage_input_override(override: USStageInputOverride) -> None:
    if override.stage_id not in US_CANONICAL_STAGE_IDS:
        raise ValueError(f"Unknown US pipeline stage: {override.stage_id}")
    contract = get_us_pipeline_stage_contract(override.stage_id)
    input_keys = {resource.key for resource in contract.inputs}
    if override.key not in input_keys:
        valid_keys = ", ".join(sorted(input_keys)) or "none"
        raise ValueError(
            f"Unknown input override key {override.stage_id}.{override.key}; "
            f"valid keys: {valid_keys}"
        )


def _artifact_ref(
    artifact_root: Path,
    artifacts: Mapping[str, Any],
    artifact_key: str,
    stage_id: str,
    *,
    category: USArtifactCategory = "required_output",
    assume_exists: bool = False,
) -> USArtifactRef | None:
    declared = artifacts.get(artifact_key)
    if declared is None:
        return None
    contract = get_us_stage_artifact_contract(stage_id, artifact_key)
    return USArtifactRef(
        key=artifact_key,
        path=str(declared),
        format=contract.format,
        required=contract.required,
        category=category,
        resume_role=contract.resume_role,
        assume_exists=assume_exists,
        exists=_artifact_path_exists(artifact_root, declared),
    )


def _derived_artifact_ref(
    artifact_root: Path,
    artifact_key: str,
    stage_id: str,
) -> USArtifactRef:
    contract = get_us_stage_artifact_contract(stage_id, artifact_key)
    path = resolve_us_stage_artifact_contract_path(
        artifact_root, stage_id, artifact_key
    )
    return USArtifactRef(
        key=artifact_key,
        path=str(path.relative_to(artifact_root)),
        format=contract.format,
        required=contract.required,
        category="derived",
        resume_role=contract.resume_role,
        assume_exists=True,
    )


def _artifact_exists(
    artifact_root: Path,
    artifacts: Mapping[str, Any],
    artifact_key: str,
) -> bool:
    declared = artifacts.get(artifact_key)
    return declared is not None and _artifact_path_exists(artifact_root, declared)


def _artifact_path_exists(artifact_root: Path, value: Any) -> bool:
    path = Path(str(value))
    if not path.is_absolute():
        path = artifact_root / path
    return path.exists()


def _path_for_manifest(path: Path, artifact_root: Path) -> str:
    try:
        return str(path.relative_to(artifact_root))
    except ValueError:
        return str(path)


def _policyengine_entity_metadata_summary(
    artifact_root: Path,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    declared = artifacts.get("policyengine_entity_tables")
    if declared is None:
        return {}
    path = Path(str(declared))
    if not path.is_absolute():
        path = artifact_root / path
    summary: dict[str, Any] = {
        "metadata_path": _path_for_manifest(path, artifact_root),
    }
    if not path.exists() or not path.is_file():
        return summary
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return summary
    if not isinstance(metadata, Mapping):
        return summary
    stage = metadata.get("stage")
    if stage is not None:
        summary["stage"] = stage
    tables: dict[str, dict[str, Any]] = {}
    for key in (
        "households",
        "persons",
        "tax_units",
        "spm_units",
        "families",
        "marital_units",
    ):
        table_metadata = metadata.get(key)
        if not isinstance(table_metadata, Mapping):
            continue
        columns = table_metadata.get("columns", ())
        column_names = (
            [str(column) for column in columns]
            if isinstance(columns, (list, tuple))
            else []
        )
        tables[key] = {
            "rows": table_metadata.get("rows"),
            "columns": column_names,
        }
    if tables:
        summary["tables"] = tables
    return summary


def _diagnostics(
    stage_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, USDiagnosticOutput]:
    diagnostics = dict(manifest.get("diagnostics", {}))
    stage_diagnostics = diagnostics.get(stage_id)
    summary = (
        dict(stage_diagnostics)
        if isinstance(stage_diagnostics, Mapping)
        else _default_stage_diagnostic_summary(stage_id, manifest)
    )
    return {
        "stage_summary": USDiagnosticOutput(
            key="stage_summary",
            description=f"Saved-run diagnostic summary for {stage_id}.",
            summary=summary,
        )
    }


def _default_stage_diagnostic_summary(
    stage_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = dict(manifest.get("rows", {}))
    synthesis = dict(manifest.get("synthesis", {}))
    calibration = dict(manifest.get("calibration", {}))
    artifacts = dict(manifest.get("artifacts", {}))
    if stage_id == "01_run_profile":
        return {"has_config": isinstance(manifest.get("config"), Mapping)}
    if stage_id == "02_source_loading":
        return {"source_names": list(synthesis.get("source_names", ()))}
    if stage_id == "03_source_planning":
        return {"scaffold_source": synthesis.get("scaffold_source")}
    if stage_id == "04_seed_scaffold":
        return {"seed_rows": rows.get("seed")}
    if stage_id == "05_donor_integration_synthesis":
        return {
            "seed_rows": rows.get("seed"),
            "synthetic_rows": rows.get("synthetic"),
            "backend": synthesis.get("backend"),
        }
    if stage_id == "06_policyengine_entities":
        return {"entity_tables": artifacts.get("policyengine_entity_tables")}
    if stage_id == "07_calibration":
        return {
            "calibrated_rows": rows.get("calibrated"),
            "backend": calibration.get("backend"),
            "converged": calibration.get("converged"),
        }
    if stage_id == "08_dataset_assembly":
        return {"dataset": artifacts.get("policyengine_dataset")}
    if stage_id == "09_validation_benchmarking":
        return _benchmark_summary(manifest)
    return {}


def _benchmark_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "policyengine_harness",
        "policyengine_native_scores",
        "policyengine_native_audit",
        "imputation_ablation",
    ):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            summary[key] = dict(value)
    return summary


def _serialize_value(value: Any, artifact_root: str | Path | None) -> Any:
    if isinstance(value, USArtifactRef):
        return value.to_dict(artifact_root)
    if isinstance(value, USAuxiliaryArtifact):
        return value.as_artifact_ref().to_dict(artifact_root)
    if isinstance(value, USDiagnosticOutput):
        return value.to_dict(artifact_root)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_value(item, artifact_root)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_serialize_value(item, artifact_root) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item, artifact_root) for item in value]
    if is_dataclass(value):
        return {
            str(key): _serialize_value(item, artifact_root)
            for key, item in asdict(value).items()
        }
    return value


def _required_output_is_missing(value: Any, artifact_root: str | Path) -> bool:
    if value is None:
        return True
    if isinstance(value, USArtifactRef):
        return not value.exists_under(artifact_root)
    if isinstance(value, Mapping):
        return not bool(value)
    if isinstance(value, (tuple, list, set, frozenset)):
        return not bool(value)
    if isinstance(value, str):
        return not value
    return False


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


__all__ = [
    "USAuxiliaryArtifact",
    "USArtifactCategory",
    "USArtifactRef",
    "USCalibrationOutputs",
    "USDatasetAssemblyOutputs",
    "USDiagnosticOutput",
    "USDonorSynthesisOutputs",
    "USPolicyEngineEntityOutputs",
    "USRunProfileOutputs",
    "USSeedScaffoldOutputs",
    "USSourceLoadingOutputs",
    "USSourcePlanningOutputs",
    "USStageInputOverride",
    "USStageOutputManifest",
    "USStageRunWriter",
    "USValidationBenchmarkingOutputs",
    "build_us_stage_output_manifests_from_artifact_manifest",
    "parse_us_stage_input_override",
    "resolve_us_manifest_or_contract_artifact_path",
    "write_us_stage_run_manifests_from_artifact_manifest",
]
