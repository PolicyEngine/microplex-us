"""Canonical runtime stage contracts for the US Microplex build."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

US_STAGE_CONTRACT_VERSION = "us-runtime-stages-v1"

StageResumeMode = Literal[
    "none",
    "metadata_only",
    "manual_replay",
    "manual_resume",
    "post_artifact_evidence",
]


@dataclass(frozen=True)
class USStageArtifactContract:
    """One artifact expected or produced by a canonical build stage."""

    key: str
    description: str
    path_hint: str | None = None
    required: bool = False
    resume_role: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class USStageValidationContract:
    """A future validation hook owned by a canonical build stage."""

    key: str
    description: str
    status: Literal["planned", "manual", "implemented"] = "planned"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class USPipelineStageContract:
    """Stable contract for one canonical US Microplex runtime stage."""

    id: str
    step: str
    title: str
    purpose: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    artifacts: tuple[USStageArtifactContract, ...]
    diagnostics: tuple[str, ...]
    validations: tuple[USStageValidationContract, ...]
    resume_mode: StageResumeMode
    resume_notes: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        payload["validations"] = [
            validation.to_dict() for validation in self.validations
        ]
        return payload


def default_us_pipeline_stage_contracts() -> tuple[USPipelineStageContract, ...]:
    """Return the canonical 9-stage US Microplex runtime taxonomy."""

    return (
        USPipelineStageContract(
            id="01_run_profile",
            step="01",
            title="Run profile, config, and source bundle",
            purpose="Resolve the build profile, runtime config, providers, queries, and run-level options.",
            consumes=("user configuration", "provider defaults", "runtime overrides"),
            produces=("resolved build config", "provider/query plan"),
            artifacts=(
                USStageArtifactContract(
                    key="manifest",
                    description="Top-level artifact manifest with resolved config.",
                    path_hint="manifest.json",
                    required=True,
                ),
            ),
            diagnostics=(
                "resolved provider names",
                "sample/query filters",
                "target period",
                "baseline dataset and target DB references",
            ),
            validations=(
                USStageValidationContract(
                    key="config_context",
                    description="Check required paths and context for the selected profile.",
                ),
            ),
            resume_mode="metadata_only",
            resume_notes="The resolved config can be reused, but this stage does not contain reloadable data by itself.",
        ),
        USPipelineStageContract(
            id="02_source_loading",
            step="02",
            title="Source contracts and source loading",
            purpose="Load external datasets into validated Microplex observation frames.",
            consumes=(
                "resolved provider/query plan",
                "source manifests",
                "external datasets",
            ),
            produces=(
                "observation frames",
                "source descriptors",
                "entity relationships",
            ),
            artifacts=(),
            diagnostics=(
                "source row counts",
                "entity coverage",
                "relationship validity",
                "cache/download provenance",
            ),
            validations=(
                USStageValidationContract(
                    key="observation_frame_validity",
                    description="Validate required entity tables and relationships for each source.",
                ),
            ),
            resume_mode="none",
            resume_notes="Full source-frame snapshotting is not implemented yet.",
        ),
        USPipelineStageContract(
            id="03_source_planning",
            step="03",
            title="Source planning, fusion planning, and scaffold selection",
            purpose="Choose the scaffold source and map donor/source coverage before seed construction.",
            consumes=("observation frames", "source descriptors"),
            produces=("fusion plan", "scaffold selection", "donor/source plan"),
            artifacts=(
                USStageArtifactContract(
                    key="source_plan",
                    description="Compact JSON summary of source names, scaffold, and donor variable plan.",
                    path_hint="stage_artifacts/03_source_planning/source_plan.json",
                    resume_role="diagnostic",
                ),
            ),
            diagnostics=(
                "source score summary",
                "coverage matrix",
                "scaffold source",
                "donor source names",
            ),
            validations=(
                USStageValidationContract(
                    key="scaffold_has_households_and_persons",
                    description="Check that the scaffold has household/person observations and a valid relationship.",
                ),
            ),
            resume_mode="metadata_only",
            resume_notes="The source plan explains the build route; raw source frames are not reloadable from this artifact yet.",
        ),
        USPipelineStageContract(
            id="04_seed_scaffold",
            step="04",
            title="Seed/scaffold construction",
            purpose="Project the selected scaffold source into the canonical seed structure.",
            consumes=("source plan", "scaffold frame", "identifier rules"),
            produces=("scaffold-derived seed frame", "seed schema metadata"),
            artifacts=(
                USStageArtifactContract(
                    key="scaffold_seed_data",
                    description="Seed population immediately after scaffold projection and before donor integration.",
                    path_hint="stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet",
                    required=True,
                    resume_role="manual_replay",
                ),
            ),
            diagnostics=(
                "scaffold source",
                "pre-donor seed rows and columns",
                "canonical identifier coverage",
                "required seed column defaults",
            ),
            validations=(
                USStageValidationContract(
                    key="seed_schema",
                    description="Check canonical identifiers and required seed columns.",
                ),
            ),
            resume_mode="manual_replay",
            resume_notes="The pre-donor seed frame is saved for diagnostics and manual replay; automatic donor-stage resume is not implemented yet.",
        ),
        USPipelineStageContract(
            id="05_donor_integration_synthesis",
            step="05",
            title="Donor integration, synthesis, and support enforcement",
            purpose="Integrate donor variables and produce the candidate population that will be calibrated.",
            consumes=(
                "scaffold-derived seed frame",
                "donor frames",
                "synthesis variable plan",
                "target support requirements",
            ),
            produces=(
                "donor-integrated seed frame",
                "synthetic/candidate frame",
                "synthesis metadata",
            ),
            artifacts=(
                USStageArtifactContract(
                    key="seed_data",
                    description="Seed population after donor integration and semantic guards.",
                    path_hint="seed_data.parquet",
                    required=True,
                    resume_role="diagnostic",
                ),
                USStageArtifactContract(
                    key="synthetic_data",
                    description="Candidate population before final calibration.",
                    path_hint="synthetic_data.parquet",
                    required=True,
                    resume_role="manual_replay",
                ),
                USStageArtifactContract(
                    key="synthesizer",
                    description="Optional fitted synthesis model.",
                    path_hint="synthesizer.pt",
                    resume_role="diagnostic",
                ),
            ),
            diagnostics=(
                "donor-integrated variables",
                "conditioning diagnostics",
                "authoritative override variables",
                "synthesis backend",
                "condition variables",
                "target variables",
                "support enforcement changes",
            ),
            validations=(
                USStageValidationContract(
                    key="candidate_support",
                    description="Check that candidate rows support requested marginal target cells.",
                ),
            ),
            resume_mode="manual_replay",
            resume_notes="Existing policy-stage replay can reload synthetic_data.parquet and rerun downstream PE work.",
        ),
        USPipelineStageContract(
            id="06_policyengine_entities",
            step="06",
            title="PolicyEngine entity construction and microsimulation materialization",
            purpose="Convert candidate rows into PE entity tables and materialize PE-facing inputs.",
            consumes=("synthetic/candidate frame", "PE input mapping rules"),
            produces=("PolicyEngine entity table bundle", "materialized PE variables"),
            artifacts=(
                USStageArtifactContract(
                    key="policyengine_entity_tables",
                    description="Reloadable PE entity-table bundle saved as parquet files plus metadata.",
                    path_hint="stage_artifacts/06_policyengine_entities/metadata.json",
                    resume_role="manual_resume",
                ),
            ),
            diagnostics=(
                "entity row counts",
                "ID/link integrity",
                "missing or filled PE inputs",
                "direct override variables",
            ),
            validations=(
                USStageValidationContract(
                    key="entity_integrity",
                    description="Check ID uniqueness and cross-entity links.",
                ),
            ),
            resume_mode="manual_resume",
            resume_notes="The entity-table bundle can be loaded for manual downstream calibration/export workflows.",
        ),
        USPipelineStageContract(
            id="07_calibration",
            step="07",
            title="Target resolution, selection, and calibration",
            purpose="Resolve target constraints, solve weights, and summarize fit quality.",
            consumes=("PE entity table bundle", "target provider/query", "calibration config"),
            produces=("calibrated tables", "calibration summary", "target ledger"),
            artifacts=(
                USStageArtifactContract(
                    key="calibrated_data",
                    description="Calibrated person-level output frame.",
                    path_hint="calibrated_data.parquet",
                    required=True,
                    resume_role="manual_replay",
                ),
                USStageArtifactContract(
                    key="targets",
                    description="Saved target payload used by the build.",
                    path_hint="targets.json",
                    required=True,
                    resume_role="manual_replay",
                ),
                USStageArtifactContract(
                    key="calibration_summary",
                    description="Stage-local calibration summary JSON.",
                    path_hint="stage_artifacts/07_calibration/calibration_summary.json",
                    resume_role="diagnostic",
                ),
            ),
            diagnostics=(
                "supported and unsupported targets",
                "feasibility filter",
                "calibration stages",
                "target ledger",
                "oracle loss",
                "weight diagnostics",
            ),
            validations=(
                USStageValidationContract(
                    key="calibration_fit",
                    description="Check convergence, selected target errors, and weight diagnostics.",
                ),
            ),
            resume_mode="manual_replay",
            resume_notes="Saved calibrated outputs can be reused for export/assembly; full conditional calibration is future work.",
        ),
        USPipelineStageContract(
            id="08_dataset_assembly",
            step="08",
            title="Dataset assembly and publication",
            purpose="Assemble the calibrated output into the distributable PE dataset artifact.",
            consumes=("calibrated entity tables", "export variable maps", "period config"),
            produces=("PolicyEngine H5 dataset", "artifact manifest", "data-flow snapshot"),
            artifacts=(
                USStageArtifactContract(
                    key="policyengine_dataset",
                    description="PolicyEngine-readable H5 dataset.",
                    path_hint="policyengine_us.h5",
                    resume_role="post_artifact_evidence",
                ),
                USStageArtifactContract(
                    key="stage_manifest",
                    description="Canonical stage manifest for the saved run.",
                    path_hint="stage_manifest.json",
                    required=True,
                ),
                USStageArtifactContract(
                    key="data_flow_snapshot",
                    description="Site-facing saved-run pipeline snapshot.",
                    path_hint="data_flow_snapshot.json",
                    required=True,
                ),
            ),
            diagnostics=(
                "exported variable maps",
                "excluded variables",
                "H5 loadability",
                "row counts and weight totals",
            ),
            validations=(
                USStageValidationContract(
                    key="dataset_loadability",
                    description="Check that the assembled H5 can be opened and contains expected arrays.",
                ),
            ),
            resume_mode="post_artifact_evidence",
            resume_notes="The assembled dataset is the input for validation and benchmarking evidence backfills.",
        ),
        USPipelineStageContract(
            id="09_validation_benchmarking",
            step="09",
            title="Validation and benchmarking",
            purpose="Evaluate the assembled dataset and attach benchmark evidence.",
            consumes=("PolicyEngine H5 dataset", "baseline dataset", "target provider/query"),
            produces=("harness evidence", "native scores", "audits", "run registry/index evidence"),
            artifacts=(
                USStageArtifactContract(
                    key="policyengine_harness",
                    description="PolicyEngine harness comparison payload.",
                    path_hint="policyengine_harness.json",
                    resume_role="diagnostic",
                ),
                USStageArtifactContract(
                    key="policyengine_native_scores",
                    description="PE-US-data native score comparison payload.",
                    path_hint="policyengine_native_scores.json",
                    resume_role="diagnostic",
                ),
                USStageArtifactContract(
                    key="validation_evidence",
                    description="Stage-local evidence manifest for validation sidecars.",
                    path_hint="stage_artifacts/09_validation_benchmarking/evidence_manifest.json",
                    resume_role="diagnostic",
                ),
            ),
            diagnostics=(
                "harness deltas",
                "native score deltas",
                "target win rates",
                "audit verdicts",
                "ablation summaries",
            ),
            validations=(
                USStageValidationContract(
                    key="benchmark_completeness",
                    description="Check that configured benchmark evidence was produced.",
                ),
            ),
            resume_mode="post_artifact_evidence",
            resume_notes="Benchmark evidence can be rerun or backfilled against the Stage 8 dataset artifact.",
        ),
    )


def get_us_pipeline_stage_contract(stage_id: str) -> USPipelineStageContract:
    """Return one canonical US pipeline stage contract by ID."""

    for contract in default_us_pipeline_stage_contracts():
        if contract.id == stage_id:
            return contract
    raise KeyError(f"Unknown US pipeline stage contract: {stage_id}")


def serialize_us_pipeline_stage_contracts() -> dict[str, object]:
    """Serialize the canonical US stage contract registry."""

    contracts = default_us_pipeline_stage_contracts()
    return {
        "schemaVersion": 1,
        "contractVersion": US_STAGE_CONTRACT_VERSION,
        "pipeline": "us_microplex",
        "stages": [contract.to_dict() for contract in contracts],
    }


__all__ = [
    "US_STAGE_CONTRACT_VERSION",
    "USPipelineStageContract",
    "USStageArtifactContract",
    "USStageValidationContract",
    "default_us_pipeline_stage_contracts",
    "get_us_pipeline_stage_contract",
    "serialize_us_pipeline_stage_contracts",
]
