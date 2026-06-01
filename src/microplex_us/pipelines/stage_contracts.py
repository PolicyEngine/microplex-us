"""Canonical runtime stage contracts for the US Microplex build."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

US_STAGE_CONTRACT_VERSION = "us-runtime-stages-v2"

StageResumeMode = Literal[
    "none",
    "metadata_only",
    "manual_replay",
    "manual_resume",
    "post_artifact_evidence",
]

StageArtifactResumeRole = Literal[
    "diagnostic",
    "manual_replay",
    "manual_resume",
    "post_artifact_evidence",
]

StageArtifactFormat = Literal[
    "json",
    "parquet_dataframe",
    "policyengine_entity_bundle",
    "h5_dataset",
    "model_file",
    "sqlite",
    "unknown",
]

StageArtifactHashMode = Literal[
    "none",
    "file_sha256",
    "directory_sha256",
]

StageResourceKind = Literal[
    "artifact",
    "config",
    "external_data",
    "manifest",
    "runtime_object",
    "stage_output",
]

US_CANONICAL_STAGE_IDS = (
    "01_run_profile",
    "02_source_loading",
    "03_source_planning",
    "04_seed_scaffold",
    "05_donor_integration_synthesis",
    "06_policyengine_entities",
    "07_calibration",
    "08_dataset_assembly",
    "09_validation_benchmarking",
)

US_LEGACY_STAGE_ID_ALIASES = {
    # Historical run_contract.py IDs from the US Microplex build path.
    "preflight": "01_run_profile",
    "source_loading": "02_source_loading",
    "source_planning": "03_source_planning",
    "seed_scaffold": "04_seed_scaffold",
    "seed_build": "05_donor_integration_synthesis",
    "donor_integration": "05_donor_integration_synthesis",
    "synthesis": "05_donor_integration_synthesis",
    "support_enforcement": "05_donor_integration_synthesis",
    "policyengine_materialization": "06_policyengine_entities",
    "target_build": "07_calibration",
    "calibration": "07_calibration",
    "dataset_assembly": "08_dataset_assembly",
    "finalization": "08_dataset_assembly",
    "validation": "09_validation_benchmarking",
    "benchmark": "09_validation_benchmarking",
    "scoring": "09_validation_benchmarking",
    "policyengine_native_scores": "09_validation_benchmarking",
    # Historical PE-US-data parity plan IDs used in Microplex docs/snapshots.
    "source-contracts": "02_source_loading",
    "cps-construction": "02_source_loading",
    "puf-ingestion-uprating": "02_source_loading",
    "extended-cps-qrf": "05_donor_integration_synthesis",
    "family-imputation-parity": "05_donor_integration_synthesis",
    "entity-export-parity": "06_policyengine_entities",
    "weighting-backend": "07_calibration",
    "targets-and-eval": "09_validation_benchmarking",
}


def canonicalize_us_pipeline_stage_id(stage_id: str) -> str:
    """Return the canonical US runtime stage ID for a current or legacy ID."""

    if stage_id in US_CANONICAL_STAGE_IDS:
        return stage_id
    return US_LEGACY_STAGE_ID_ALIASES.get(stage_id, stage_id)


@dataclass(frozen=True)
class USStageArtifactContract:
    """One artifact expected or produced by a canonical build stage."""

    key: str
    description: str
    path_hint: str | None = None
    required: bool = False
    resume_role: StageArtifactResumeRole | None = None
    format: StageArtifactFormat = "unknown"
    hash_mode: StageArtifactHashMode = "none"

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
class USStageResourceContract:
    """Structured input or output dependency for one canonical build stage."""

    key: str
    description: str
    kind: StageResourceKind
    required: bool = True
    stage_id: str | None = None
    artifact_key: str | None = None
    config_key: str | None = None
    manifest_key: str | None = None

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
    inputs: tuple[USStageResourceContract, ...]
    outputs: tuple[USStageResourceContract, ...]
    artifacts: tuple[USStageArtifactContract, ...]
    diagnostics: tuple[str, ...]
    validations: tuple[USStageValidationContract, ...]
    resume_mode: StageResumeMode
    resume_notes: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["inputs"] = [resource.to_dict() for resource in self.inputs]
        payload["outputs"] = [resource.to_dict() for resource in self.outputs]
        payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        payload["validations"] = [
            validation.to_dict() for validation in self.validations
        ]
        return payload


def _artifact_resource(
    key: str,
    description: str,
    *,
    stage_id: str,
    artifact_key: str | None = None,
    required: bool = True,
) -> USStageResourceContract:
    return USStageResourceContract(
        key=key,
        description=description,
        kind="artifact",
        required=required,
        stage_id=stage_id,
        artifact_key=artifact_key or key,
    )


def _config_resource(
    key: str,
    description: str,
    *,
    config_key: str | None = None,
    required: bool = True,
) -> USStageResourceContract:
    return USStageResourceContract(
        key=key,
        description=description,
        kind="config",
        required=required,
        config_key=config_key or key,
    )


def _external_resource(
    key: str,
    description: str,
    *,
    required: bool = True,
) -> USStageResourceContract:
    return USStageResourceContract(
        key=key,
        description=description,
        kind="external_data",
        required=required,
    )


def _manifest_resource(
    key: str,
    description: str,
    *,
    manifest_key: str | None = None,
    required: bool = True,
) -> USStageResourceContract:
    return USStageResourceContract(
        key=key,
        description=description,
        kind="manifest",
        required=required,
        manifest_key=manifest_key or key,
    )


def _runtime_resource(
    key: str,
    description: str,
    *,
    required: bool = True,
) -> USStageResourceContract:
    return USStageResourceContract(
        key=key,
        description=description,
        kind="runtime_object",
        required=required,
    )


def _stage_output_resource(
    key: str,
    description: str,
    *,
    stage_id: str,
    required: bool = True,
) -> USStageResourceContract:
    return USStageResourceContract(
        key=key,
        description=description,
        kind="stage_output",
        required=required,
        stage_id=stage_id,
    )


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
            inputs=(
                _config_resource(
                    "build_profile",
                    "Selected build profile and runtime overrides.",
                    config_key="profile",
                    required=False,
                ),
                _config_resource(
                    "policyengine_target_period",
                    "Target period used by downstream PolicyEngine export and validation.",
                ),
                _config_resource(
                    "calibration_backend",
                    "Calibration backend selected for this run.",
                ),
                _config_resource(
                    "source_names",
                    "Requested source names or provider defaults.",
                    required=False,
                ),
            ),
            outputs=(
                _artifact_resource(
                    "manifest",
                    "Top-level manifest containing resolved configuration and artifact map.",
                    stage_id="01_run_profile",
                ),
                _stage_output_resource(
                    "resolved_config",
                    "Resolved build configuration recorded for downstream stages.",
                    stage_id="01_run_profile",
                ),
                _stage_output_resource(
                    "provider_query_plan",
                    "Resolved provider and source-query plan for source loading.",
                    stage_id="01_run_profile",
                ),
            ),
            artifacts=(
                USStageArtifactContract(
                    key="manifest",
                    description="Top-level artifact manifest with resolved config.",
                    path_hint="manifest.json",
                    required=True,
                    format="json",
                    hash_mode="file_sha256",
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
            inputs=(
                _stage_output_resource(
                    "provider_query_plan",
                    "Resolved provider and source-query plan from Stage 1.",
                    stage_id="01_run_profile",
                ),
                _external_resource(
                    "source_datasets",
                    "External source datasets requested by the provider/query plan.",
                ),
            ),
            outputs=(
                _stage_output_resource(
                    "observation_frame_summary",
                    "Saved summary of loaded Microplex observation frames with source metadata.",
                    stage_id="02_source_loading",
                ),
                _stage_output_resource(
                    "source_descriptors",
                    "Source descriptors attached to the loaded observation frames.",
                    stage_id="02_source_loading",
                ),
                _stage_output_resource(
                    "source_relationships",
                    "Validated entity relationships in loaded source frames.",
                    stage_id="02_source_loading",
                ),
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
            inputs=(
                _runtime_resource(
                    "observation_frames",
                    "Loaded observation frames from Stage 2.",
                ),
                _runtime_resource(
                    "source_descriptors",
                    "Source descriptors attached to the loaded frames.",
                ),
            ),
            outputs=(
                _artifact_resource(
                    "source_plan",
                    "Saved scaffold and donor/source planning summary.",
                    stage_id="03_source_planning",
                ),
                _stage_output_resource(
                    "scaffold_selection",
                    "Selected scaffold/backbone source and donor plan.",
                    stage_id="03_source_planning",
                ),
            ),
            artifacts=(
                USStageArtifactContract(
                    key="source_plan",
                    description="Compact JSON summary of source names, scaffold, and donor variable plan.",
                    path_hint="stage_artifacts/03_source_planning/source_plan.json",
                    required=True,
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="file_sha256",
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
            inputs=(
                _artifact_resource(
                    "source_plan",
                    "Saved scaffold and donor/source planning summary from Stage 3.",
                    stage_id="03_source_planning",
                ),
                _stage_output_resource(
                    "scaffold_selection",
                    "Selected scaffold/backbone source from Stage 3.",
                    stage_id="03_source_planning",
                ),
                _runtime_resource(
                    "scaffold_frame",
                    "Loaded source frame selected as the population scaffold.",
                ),
            ),
            outputs=(
                _artifact_resource(
                    "scaffold_seed_data",
                    "Scaffold-projected seed population before donor integration.",
                    stage_id="04_seed_scaffold",
                ),
                _stage_output_resource(
                    "seed_schema_metadata",
                    "Canonical identifier and required-column metadata for the seed.",
                    stage_id="04_seed_scaffold",
                ),
            ),
            artifacts=(
                USStageArtifactContract(
                    key="scaffold_seed_data",
                    description="Seed population immediately after scaffold projection and before donor integration.",
                    path_hint="stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet",
                    required=True,
                    resume_role="manual_replay",
                    format="parquet_dataframe",
                    hash_mode="file_sha256",
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
            inputs=(
                _artifact_resource(
                    "scaffold_seed_data",
                    "Scaffold-projected seed population from Stage 4.",
                    stage_id="04_seed_scaffold",
                ),
                _runtime_resource(
                    "donor_frames",
                    "Loaded donor source frames used for variable integration.",
                ),
                _config_resource(
                    "synthesis_backend",
                    "Configured synthesis backend.",
                ),
                _config_resource(
                    "n_synthetic",
                    "Requested synthetic population size.",
                    required=False,
                ),
                _config_resource(
                    "random_seed",
                    "Random seed used by donor integration and synthesis.",
                ),
                _config_resource(
                    "synthesizer_condition_vars",
                    "Configured synthesis conditioning variables.",
                    required=False,
                ),
                _config_resource(
                    "synthesizer_target_vars",
                    "Configured synthesis target variables.",
                    required=False,
                ),
                _config_resource(
                    "synthesizer_epochs",
                    "Configured synthesizer training epochs.",
                    required=False,
                ),
                _config_resource(
                    "synthesizer_batch_size",
                    "Configured synthesizer batch size.",
                    required=False,
                ),
                _config_resource(
                    "synthesizer_learning_rate",
                    "Configured synthesizer learning rate.",
                    required=False,
                ),
                _config_resource(
                    "synthesizer_n_layers",
                    "Configured synthesizer network depth.",
                    required=False,
                ),
                _config_resource(
                    "synthesizer_hidden_dim",
                    "Configured synthesizer hidden dimension.",
                    required=False,
                ),
                _config_resource(
                    "donor_imputer_backend",
                    "Configured donor imputer backend.",
                    required=False,
                ),
                _config_resource(
                    "donor_imputer_condition_selection",
                    "Configured donor imputer condition selection strategy.",
                    required=False,
                ),
                _config_resource(
                    "donor_imputer_max_condition_vars",
                    "Configured donor imputer condition-variable cap.",
                    required=False,
                ),
                _config_resource(
                    "donor_imputer_excluded_variables",
                    "Variables excluded from donor imputation.",
                    required=False,
                ),
                _config_resource(
                    "donor_imputer_authoritative_override_variables",
                    "Variables treated as authoritative donor overrides.",
                    required=False,
                ),
                _config_resource(
                    "bootstrap_strata_columns",
                    "Bootstrap strata columns used by seed/bootstrap synthesis.",
                    required=False,
                ),
            ),
            outputs=(
                _artifact_resource(
                    "seed_data",
                    "Seed population after donor integration and semantic guards.",
                    stage_id="05_donor_integration_synthesis",
                ),
                _artifact_resource(
                    "synthetic_data",
                    "Candidate population before final calibration.",
                    stage_id="05_donor_integration_synthesis",
                ),
                _manifest_resource(
                    "synthesis_metadata",
                    "Synthesis metadata recorded in the saved manifest.",
                    manifest_key="synthesis",
                ),
            ),
            artifacts=(
                USStageArtifactContract(
                    key="seed_data",
                    description="Seed population after donor integration and semantic guards.",
                    path_hint="seed_data.parquet",
                    required=True,
                    resume_role="diagnostic",
                    format="parquet_dataframe",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="synthetic_data",
                    description="Candidate population before final calibration.",
                    path_hint="synthetic_data.parquet",
                    required=True,
                    resume_role="manual_replay",
                    format="parquet_dataframe",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="synthesizer",
                    description="Optional fitted synthesis model.",
                    path_hint="synthesizer.pt",
                    resume_role="diagnostic",
                    format="model_file",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="source_weight_diagnostics",
                    description="Diagnostic summary of source-level contribution weights.",
                    path_hint="source_weight_diagnostics.json",
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="file_sha256",
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
            inputs=(
                _artifact_resource(
                    "synthetic_data",
                    "Candidate population from Stage 5.",
                    stage_id="05_donor_integration_synthesis",
                ),
                _runtime_resource(
                    "policyengine_mapping_rules",
                    "Rules mapping Microplex candidate rows into PolicyEngine entities.",
                ),
            ),
            outputs=(
                _artifact_resource(
                    "policyengine_entity_tables",
                    "Reloadable PolicyEngine entity-table checkpoint.",
                    stage_id="06_policyengine_entities",
                ),
                _stage_output_resource(
                    "materialized_policyengine_inputs",
                    "PolicyEngine-facing variables materialized for calibration/export.",
                    stage_id="06_policyengine_entities",
                ),
            ),
            artifacts=(
                USStageArtifactContract(
                    key="policyengine_entity_tables",
                    description="Reloadable PE entity-table bundle saved as parquet files plus metadata.",
                    path_hint="stage_artifacts/06_policyengine_entities/metadata.json",
                    required=True,
                    resume_role="manual_resume",
                    format="policyengine_entity_bundle",
                    hash_mode="directory_sha256",
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
            consumes=(
                "PE entity table bundle",
                "target provider/query",
                "calibration config",
            ),
            produces=("calibrated tables", "calibration summary", "target ledger"),
            inputs=(
                _artifact_resource(
                    "policyengine_entity_tables",
                    "PolicyEngine entity-table checkpoint from Stage 6.",
                    stage_id="06_policyengine_entities",
                ),
                _external_resource(
                    "target_provider",
                    "Target provider or target database queried for calibration.",
                ),
                _config_resource(
                    "calibration_backend",
                    "Configured calibration backend.",
                ),
                _config_resource(
                    "calibration_tol",
                    "Configured calibration tolerance.",
                    required=False,
                ),
                _config_resource(
                    "calibration_max_iter",
                    "Configured maximum calibration iterations or epochs.",
                    required=False,
                ),
                _config_resource(
                    "target_sparsity",
                    "Configured sparse-target selection pressure.",
                    required=False,
                ),
                _config_resource(
                    "policyengine_quantity_targets",
                    "Configured PolicyEngine quantity targets.",
                    required=False,
                ),
                _config_resource(
                    "policyengine_targets_db",
                    "PolicyEngine target database used for calibration.",
                    required=False,
                ),
                _config_resource(
                    "policyengine_calibration_target_variables",
                    "Configured calibration target variables.",
                    required=False,
                ),
                _config_resource(
                    "policyengine_calibration_target_domains",
                    "Configured calibration target domains.",
                    required=False,
                ),
                _config_resource(
                    "policyengine_calibration_geo_levels",
                    "Configured calibration geography levels.",
                    required=False,
                ),
            ),
            outputs=(
                _artifact_resource(
                    "calibrated_data",
                    "Calibrated output frame.",
                    stage_id="07_calibration",
                ),
                _artifact_resource(
                    "targets",
                    "Target payload used by the build.",
                    stage_id="07_calibration",
                ),
                _artifact_resource(
                    "calibration_summary",
                    "Stage-local calibration summary.",
                    stage_id="07_calibration",
                ),
            ),
            artifacts=(
                USStageArtifactContract(
                    key="calibrated_data",
                    description="Calibrated person-level output frame.",
                    path_hint="calibrated_data.parquet",
                    required=True,
                    resume_role="manual_replay",
                    format="parquet_dataframe",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="targets",
                    description="Saved target payload used by the build.",
                    path_hint="targets.json",
                    required=True,
                    resume_role="manual_replay",
                    format="json",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="calibration_summary",
                    description="Stage-local calibration summary JSON.",
                    path_hint="stage_artifacts/07_calibration/calibration_summary.json",
                    required=True,
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="file_sha256",
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
            consumes=(
                "calibrated entity tables",
                "export variable maps",
                "period config",
            ),
            produces=(
                "PolicyEngine H5 dataset",
                "artifact manifest",
                "data-flow snapshot",
            ),
            inputs=(
                _artifact_resource(
                    "calibrated_data",
                    "Calibrated output frame from Stage 7.",
                    stage_id="07_calibration",
                ),
                _artifact_resource(
                    "policyengine_entity_tables",
                    "PolicyEngine entity-table checkpoint from Stage 6.",
                    stage_id="06_policyengine_entities",
                ),
                _config_resource(
                    "policyengine_dataset_year",
                    "PolicyEngine dataset period used during H5 export.",
                    required=False,
                ),
            ),
            outputs=(
                _artifact_resource(
                    "policyengine_dataset",
                    "PolicyEngine-readable H5 dataset.",
                    stage_id="08_dataset_assembly",
                ),
                _artifact_resource(
                    "stage_manifest",
                    "Canonical saved-run stage manifest.",
                    stage_id="08_dataset_assembly",
                ),
                _artifact_resource(
                    "data_flow_snapshot",
                    "Site-facing saved-run pipeline snapshot.",
                    stage_id="08_dataset_assembly",
                ),
                _artifact_resource(
                    "artifact_inventory",
                    "Stage-owned artifact inventory.",
                    stage_id="08_dataset_assembly",
                ),
                _artifact_resource(
                    "conditional_readiness",
                    "Conditional-readiness report.",
                    stage_id="08_dataset_assembly",
                ),
            ),
            artifacts=(
                USStageArtifactContract(
                    key="policyengine_dataset",
                    description="PolicyEngine-readable H5 dataset.",
                    path_hint="policyengine_us.h5",
                    required=True,
                    resume_role="post_artifact_evidence",
                    format="h5_dataset",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="capital_gains_lots",
                    description="Optional synthetic capital-gains lot sidecar database.",
                    path_hint="capital_gains_lots.sqlite",
                    resume_role="diagnostic",
                    format="sqlite",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="stage_manifest",
                    description="Canonical stage manifest for the saved run.",
                    path_hint="stage_manifest.json",
                    required=True,
                    format="json",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="data_flow_snapshot",
                    description="Site-facing saved-run pipeline snapshot.",
                    path_hint="data_flow_snapshot.json",
                    required=True,
                    format="json",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="artifact_inventory",
                    description="Stage-owned artifact inventory with existence, role, and hash metadata.",
                    path_hint="stage_artifacts/artifact_inventory.json",
                    required=True,
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="none",
                ),
                USStageArtifactContract(
                    key="conditional_readiness",
                    description="Conditional-readiness report for manual reuse decisions.",
                    path_hint="stage_artifacts/conditional_readiness.json",
                    required=True,
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="none",
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
            consumes=(
                "PolicyEngine H5 dataset",
                "baseline dataset",
                "target provider/query",
            ),
            produces=(
                "harness evidence",
                "native scores",
                "audits",
                "run registry/index evidence",
            ),
            inputs=(
                _artifact_resource(
                    "policyengine_dataset",
                    "PolicyEngine-readable H5 dataset from Stage 8.",
                    stage_id="08_dataset_assembly",
                ),
                _external_resource(
                    "baseline_dataset",
                    "Baseline dataset used by validation or comparison harnesses.",
                    required=False,
                ),
                _external_resource(
                    "target_provider",
                    "Target provider or target database used for benchmark evidence.",
                    required=False,
                ),
                _config_resource(
                    "policyengine_dataset_year",
                    "PolicyEngine dataset period used during validation.",
                    required=False,
                ),
            ),
            outputs=(
                _artifact_resource(
                    "validation_evidence",
                    "Stage-local evidence manifest for validation sidecars.",
                    stage_id="09_validation_benchmarking",
                ),
                _stage_output_resource(
                    "benchmark_summary",
                    "Saved summary of validation and benchmark evidence attached to the run.",
                    stage_id="09_validation_benchmarking",
                ),
                _artifact_resource(
                    "policyengine_harness",
                    "PolicyEngine harness comparison payload.",
                    stage_id="09_validation_benchmarking",
                    required=False,
                ),
                _artifact_resource(
                    "policyengine_native_scores",
                    "PE-US-data native score comparison payload.",
                    stage_id="09_validation_benchmarking",
                    required=False,
                ),
                _artifact_resource(
                    "policyengine_native_audit",
                    "PE-US-data native score audit payload.",
                    stage_id="09_validation_benchmarking",
                    required=False,
                ),
            ),
            artifacts=(
                USStageArtifactContract(
                    key="policyengine_harness",
                    description="PolicyEngine harness comparison payload.",
                    path_hint="policyengine_harness.json",
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="policyengine_native_scores",
                    description="PE-US-data native score comparison payload.",
                    path_hint="policyengine_native_scores.json",
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="policyengine_native_audit",
                    description="PE-US-data native score audit payload.",
                    path_hint="pe_us_data_rebuild_native_audit.json",
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="imputation_ablation",
                    description="Imputation ablation benchmark payload.",
                    path_hint="imputation_ablation.json",
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="child_tax_unit_agi_drift",
                    description="Child tax-unit AGI drift diagnostic payload.",
                    path_hint="child_tax_unit_agi_drift.json",
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="file_sha256",
                ),
                USStageArtifactContract(
                    key="validation_evidence",
                    description="Stage-local evidence manifest for validation sidecars.",
                    path_hint="stage_artifacts/09_validation_benchmarking/evidence_manifest.json",
                    required=True,
                    resume_role="diagnostic",
                    format="json",
                    hash_mode="file_sha256",
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


def get_us_stage_artifact_contract(
    stage_id: str,
    artifact_key: str,
) -> USStageArtifactContract:
    """Return one artifact contract from a canonical stage."""

    contract = get_us_pipeline_stage_contract(stage_id)
    for artifact in contract.artifacts:
        if artifact.key == artifact_key:
            return artifact
    raise KeyError(f"Unknown US stage artifact contract: {stage_id}.{artifact_key}")


def resolve_us_stage_artifact_contract_path(
    artifact_dir: str | Path,
    stage_id: str,
    artifact_key: str,
) -> Path:
    """Resolve a stage artifact's canonical path from its contract path hint."""

    artifact = get_us_stage_artifact_contract(stage_id, artifact_key)
    if artifact.path_hint is None:
        raise KeyError(f"US stage artifact has no path hint: {stage_id}.{artifact_key}")
    return Path(artifact_dir) / artifact.path_hint


def config_keys_for_us_pipeline_stage(stage_id: str) -> tuple[str, ...]:
    """Return config keys that affect one canonical stage's reuse checks."""

    contract = get_us_pipeline_stage_contract(stage_id)
    return tuple(
        dict.fromkeys(
            resource.config_key
            for resource in contract.inputs
            if resource.kind == "config" and resource.config_key is not None
        )
    )


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
    "StageArtifactFormat",
    "StageArtifactHashMode",
    "StageArtifactResumeRole",
    "StageResourceKind",
    "StageResumeMode",
    "US_CANONICAL_STAGE_IDS",
    "US_LEGACY_STAGE_ID_ALIASES",
    "US_STAGE_CONTRACT_VERSION",
    "USPipelineStageContract",
    "USStageArtifactContract",
    "USStageResourceContract",
    "USStageValidationContract",
    "canonicalize_us_pipeline_stage_id",
    "config_keys_for_us_pipeline_stage",
    "default_us_pipeline_stage_contracts",
    "get_us_stage_artifact_contract",
    "get_us_pipeline_stage_contract",
    "resolve_us_stage_artifact_contract_path",
    "serialize_us_pipeline_stage_contracts",
]
