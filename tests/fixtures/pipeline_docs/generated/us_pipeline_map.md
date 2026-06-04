# US Pipeline Map

Generated from `microplex_us.pipelines.stage_contracts`, `docs/us_pipeline_internals.map.json`, and `@pipeline_node` decorators.

This page is the static documentation surface for the build path. It lists canonical stages, substages, exact class or method references, source locations, and directed edges.

## Canonical Stages

| Stage | Title | Produces |
| --- | --- | --- |
| `01_run_profile` 01 | Run profile, config, and source bundle | `resolved build config`, `provider/query plan` |
| `02_source_loading` 02 | Source contracts and source loading | `observation frames`, `source descriptors`, `entity relationships` |
| `03_source_planning` 03 | Source planning, fusion planning, and scaffold selection | `fusion plan`, `scaffold selection`, `donor/source plan` |
| `04_seed_scaffold` 04 | Seed/scaffold construction | `scaffold-derived seed frame`, `seed schema metadata` |
| `05_donor_integration_synthesis` 05 | Donor integration, synthesis, and support enforcement | `donor-integrated seed frame`, `synthetic/candidate frame`, `synthesis metadata` |
| `06_policyengine_entities` 06 | PolicyEngine entity construction and microsimulation materialization | `PolicyEngine entity table bundle`, `materialized PE variables` |
| `07_calibration` 07 | Target resolution, selection, and calibration | `calibrated tables`, `calibration summary`, `target ledger` |
| `08_dataset_assembly` 08 | Dataset assembly and publication | `PolicyEngine H5 dataset`, `artifact manifest`, `data-flow snapshot` |
| `09_validation_benchmarking` 09 | Validation and benchmarking | `harness evidence`, `native scores`, `audits`, `run registry/index evidence` |

## 01: Run profile, config, and source bundle

Resolve the build profile, runtime config, providers, queries, and run-level options.

### Resolve run profile

Resolve build configuration, provider query choices, output locations, and optional validation settings before any data is loaded.

- Substage ID: `01a_resolve_run_profile`
- Canonical stage: `01_run_profile`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `config.build_config` USMicroplexBuildConfig | `artifact` | `current` |  |
| `config.provider_queries` Provider query plan | `artifact` | `current` |  |
| `core.source_provider` SourceProvider | `library` | `current` | `microplex.core.SourceProvider`, `microplex.core.SourceQuery` |
| `us.pipeline.resolve_source_query` Resolve source query | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline._resolve_source_query`, `microplex_us.pipelines.us.USMicroplexPipeline._source_query_keys` |
| `core.source_provider_load_frame` Load provider frame | `library` | `current` | `microplex.core.SourceProvider.load_frame` |
| `core.observation_frame` ObservationFrame | `library` | `current` | `microplex.core.ObservationFrame`, `microplex.core.SourceDescriptor` |
| `stage_runtime.source_loading_lifecycle` Source-loading lifecycle writer | `infrastructure` | `current` | `microplex_us.pipelines.stage_runtime.USStageRuntimeWriter.start_stage`, `microplex_us.pipelines.stage_runtime.USStageRuntimeWriter.complete_stage`, `microplex_us.pipelines.stage_runtime.USStageRuntimeWriter.fail_stage` |
| `us.pipeline.source_loading_stage_outputs` Source-loading stage outputs | `process` | `current` | `microplex_us.pipelines.us._source_loading_stage_outputs` |
| `us.pipeline.build_from_frames` Build from frames | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.build_from_frames` |
| `artifact.run_profile` Run profile manifest | `artifact` | `current` |  |
| `us.pipeline.build_from_source_providers` Build from source providers | `entrypoint` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.build_from_source_providers` |

#### Edges

- `config.build_config` -> `config.provider_queries` `data_flow` (resolves)
- `config.build_config` -> `us.pipeline.build_from_source_providers` `data_flow` (starts build)
- `config.provider_queries` -> `us.pipeline.resolve_source_query` `data_flow` (query map)
- `core.source_provider` -> `us.pipeline.resolve_source_query` `uses_library` (descriptor keys)
- `us.pipeline.build_from_source_providers` -> `stage_runtime.source_loading_lifecycle` `uses_utility` (starts Stage 2)
- `us.pipeline.build_from_source_providers` -> `us.pipeline.resolve_source_query` `data_flow` (per provider)
- `us.pipeline.resolve_source_query` -> `core.source_provider_load_frame` `data_flow` (query)
- `core.source_provider` -> `core.source_provider_load_frame` `external_source` (provider implementation)
- `core.source_provider_load_frame` -> `core.observation_frame` `produces_artifact` (returns)
- `core.observation_frame` -> `us.pipeline.source_loading_stage_outputs` `data_flow` (summarizes)
- `us.pipeline.source_loading_stage_outputs` -> `stage_runtime.source_loading_lifecycle` `produces_artifact` (completes Stage 2)
- `core.observation_frame` -> `us.pipeline.build_from_frames` `data_flow` (loaded frames)
- `us.pipeline.build_from_source_providers` -> `artifact.run_profile` `produces_artifact` (initializes)
- `us.pipeline.build_from_frames` -> `artifact.run_profile` `data_flow` (continues run)

## 02: Source contracts and source loading

Load external datasets into validated Microplex observation frames.

### Prepare source input

Ask each configured provider for the requested source data and normalize it into source input records.

- Substage ID: `02a_prepare_source_input`
- Canonical stage: `02_source_loading`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `external.source_providers` Source providers | `external` | `current` |  |
| `artifact.source_frames` Observation frames | `artifact` | `current` |  |
| `us.pipeline.prepare_source_input` Prepare source input | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.prepare_source_input` |

#### Edges

- `external.source_providers` -> `us.pipeline.prepare_source_input` `external_source` (queries)
- `us.pipeline.prepare_source_input` -> `artifact.source_frames` `produces_artifact` (loads)

### Attach source semantics

Carry source descriptors, entity observations, and household-person relationships with loaded tables.

- Substage ID: `02b_attach_source_semantics`
- Canonical stage: `02_source_loading`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `core.observation_frame` ObservationFrame | `library` | `current` | `microplex.core.ObservationFrame`, `microplex.core.SourceDescriptor` |
| `artifact.source_semantics` Source semantics | `artifact` | `current` |  |
| `artifact.source_frames` Observation frames | `artifact` | `current` |  |

#### Edges

- `artifact.source_frames` -> `core.observation_frame` `uses_library` (represented as)
- `core.observation_frame` -> `artifact.source_semantics` `produces_artifact` (describes)

## 03: Source planning, fusion planning, and scaffold selection

Choose the scaffold source and map donor/source coverage before seed construction.

### Build fusion plan

Summarize variable coverage and source contribution roles before choosing the build backbone.

- Substage ID: `03a_build_fusion_plan`
- Canonical stage: `03_source_planning`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `core.fusion_planning` Fusion planning | `library` | `current` | `microplex.fusion.FusionPlan` |
| `artifact.fusion_plan` Fusion plan | `artifact` | `current` |  |
| `artifact.source_frames` Observation frames | `artifact` | `current` |  |

#### Edges

- `artifact.source_frames` -> `core.fusion_planning` `uses_library` (coverage)
- `core.fusion_planning` -> `artifact.fusion_plan` `produces_artifact` (plans)

### Select scaffold source

Pick the household-person source that becomes the structural backbone of the national population.

- Substage ID: `03b_select_scaffold_source`
- Canonical stage: `03_source_planning`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.scaffold_selection` Scaffold selection | `artifact` | `current` |  |
| `artifact.source_plan` Source plan | `artifact` | `current` |  |
| `us.pipeline.select_scaffold_source` Select scaffold source | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline._select_scaffold_source` |
| `artifact.fusion_plan` Fusion plan | `artifact` | `current` |  |

#### Edges

- `artifact.fusion_plan` -> `us.pipeline.select_scaffold_source` `data_flow` (scores)
- `us.pipeline.select_scaffold_source` -> `artifact.scaffold_selection` `produces_artifact` (chooses)
- `artifact.scaffold_selection` -> `artifact.source_plan` `produces_artifact` (records)

## 04: Seed/scaffold construction

Project the selected scaffold source into the canonical seed structure.

### Project scaffold seed

Convert the selected scaffold source into initial household and person seed tables.

- Substage ID: `04a_project_scaffold_seed`
- Canonical stage: `04_seed_scaffold`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.scaffold_seed_data` Scaffold seed data | `artifact` | `current` |  |
| `us.pipeline.prepare_seed_data_from_source` Prepare scaffold seed data | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.prepare_seed_data_from_source` |
| `artifact.scaffold_selection` Scaffold selection | `artifact` | `current` |  |

#### Edges

- `artifact.scaffold_selection` -> `us.pipeline.prepare_seed_data_from_source` `data_flow` (projects)
- `us.pipeline.prepare_seed_data_from_source` -> `artifact.scaffold_seed_data` `produces_artifact` (writes)

### Normalize scaffold identifiers

Remove generated IDs that should not leak across the stage boundary.

- Substage ID: `04b_normalize_scaffold_ids`
- Canonical stage: `04_seed_scaffold`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.normalized_scaffold_seed_data` Normalized scaffold seed | `artifact` | `current` |  |
| `us.pipeline.strip_generated_entity_ids` Strip generated entity ids | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline._strip_generated_entity_ids` |
| `artifact.scaffold_seed_data` Scaffold seed data | `artifact` | `current` |  |

#### Edges

- `artifact.scaffold_seed_data` -> `us.pipeline.strip_generated_entity_ids` `data_flow` (normalizes)
- `us.pipeline.strip_generated_entity_ids` -> `artifact.normalized_scaffold_seed_data` `produces_artifact` (hands off)

## 05: Donor integration, synthesis, and support enforcement

Integrate donor variables and produce the candidate population that will be calibrated.

### Integrate donor sources

Impute donor variables from secondary sources onto the scaffold seed.

- Substage ID: `05a_integrate_donor_sources`
- Canonical stage: `05_donor_integration_synthesis`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.donor_frames` Donor frames | `artifact` | `current` |  |
| `artifact.seed_data` Seed data | `artifact` | `current` |  |
| `us.pipeline.build_donor_imputer` Build donor imputer | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline._build_donor_imputer` |
| `us.pipeline.integrate_donor_sources` Integrate donor sources | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline._integrate_donor_sources` |
| `artifact.normalized_scaffold_seed_data` Normalized scaffold seed | `artifact` | `current` |  |

#### Edges

- `artifact.donor_frames` -> `us.pipeline.build_donor_imputer` `data_flow` (fits)
- `us.pipeline.build_donor_imputer` -> `us.pipeline.integrate_donor_sources` `uses_utility` (imputes)
- `artifact.normalized_scaffold_seed_data` -> `us.pipeline.integrate_donor_sources` `data_flow` (augments)
- `us.pipeline.integrate_donor_sources` -> `artifact.seed_data` `produces_artifact` (produces)

### Build targets and synthesis variables

Build target counts and decide which variables the synthesizer must carry forward.

- Substage ID: `05b_build_targets_and_variables`
- Canonical stage: `05_donor_integration_synthesis`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `core.targets` Target specs | `library` | `current` | `microplex.targets.TargetSpec`, `microplex.targets.TargetQuery` |
| `artifact.targets` Targets | `artifact` | `current` |  |
| `artifact.synthesis_variables` Synthesis variables | `artifact` | `current` |  |
| `us.pipeline.build_targets` Build calibration targets | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.build_targets` |
| `us.pipeline.resolve_synthesis_variables` Resolve synthesis variables | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline._resolve_synthesis_variables` |

#### Edges

- `core.targets` -> `us.pipeline.build_targets` `uses_library` (materializes)
- `us.pipeline.build_targets` -> `artifact.targets` `produces_artifact` (counts)
- `artifact.targets` -> `us.pipeline.resolve_synthesis_variables` `data_flow` (requires)
- `us.pipeline.resolve_synthesis_variables` -> `artifact.synthesis_variables` `produces_artifact` (selects)

### Synthesize population

Generate the synthetic candidate population and enforce target support.

- Substage ID: `05c_synthesize_population`
- Canonical stage: `05_donor_integration_synthesis`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `core.synthesizer` Synthesizer | `library` | `current` | `microplex.synthesizer.Synthesizer` |
| `artifact.synthetic_data` Synthetic data | `artifact` | `current` |  |
| `us.pipeline.synthesize` Synthesize records | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.synthesize` |
| `us.pipeline.ensure_target_support` Ensure target support | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.ensure_target_support` |
| `artifact.seed_data` Seed data | `artifact` | `current` |  |

#### Edges

- `artifact.seed_data` -> `us.pipeline.synthesize` `data_flow` (seeds)
- `core.synthesizer` -> `us.pipeline.synthesize` `uses_library` (runs)
- `us.pipeline.synthesize` -> `us.pipeline.ensure_target_support` `data_flow` (supports)
- `us.pipeline.ensure_target_support` -> `artifact.synthetic_data` `produces_artifact` (outputs)

## 06: PolicyEngine entity construction and microsimulation materialization

Convert candidate rows into PE entity tables and materialize PE-facing inputs.

### Materialize PolicyEngine tables

Convert synthetic records into household, person, tax-unit, family, and SPM entity tables.

- Substage ID: `06a_materialize_policyengine_tables`
- Canonical stage: `06_policyengine_entities`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.policyengine_entity_tables` PolicyEngine entity tables | `artifact` | `current` |  |
| `us.pipeline.build_policyengine_entity_tables` Build PolicyEngine entity tables | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.build_policyengine_entity_tables` |
| `artifact.synthetic_data` Synthetic data | `artifact` | `current` |  |

#### Edges

- `artifact.synthetic_data` -> `us.pipeline.build_policyengine_entity_tables` `data_flow` (maps)
- `us.pipeline.build_policyengine_entity_tables` -> `artifact.policyengine_entity_tables` `produces_artifact` (builds)

### Checkpoint and contract-check entity tables

Optionally persist pre-calibration entity tables and verify the export-column surface.

- Substage ID: `06b_checkpoint_and_contract_check`
- Canonical stage: `06_policyengine_entities`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.policyengine_checkpoint` PolicyEngine checkpoint | `artifact` | `current` |  |
| `us.policyengine.save_us_pipeline_checkpoint` Save PE entity checkpoint | `process` | `current` | `microplex_us.policyengine.us.save_us_pipeline_checkpoint` |
| `us.pipeline.check_policyengine_export_column_contract` Check PE export columns | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline._check_policyengine_export_column_contract` |
| `artifact.policyengine_entity_tables` PolicyEngine entity tables | `artifact` | `current` |  |

#### Edges

- `artifact.policyengine_entity_tables` -> `us.policyengine.save_us_pipeline_checkpoint` `produces_artifact` (optional)
- `us.policyengine.save_us_pipeline_checkpoint` -> `artifact.policyengine_checkpoint` `produces_artifact` (writes)
- `artifact.policyengine_entity_tables` -> `us.pipeline.check_policyengine_export_column_contract` `validates` (checks)

## 07: Target resolution, selection, and calibration

Resolve target constraints, solve weights, and summarize fit quality.

### Resolve PolicyEngine constraints

Load PE target rows, filter unsupported constraints, and choose the calibration household budget.

- Substage ID: `07a_resolve_policyengine_constraints`
- Canonical stage: `07_calibration`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `external.policyengine_targets_db` PolicyEngine targets DB | `external` | `current` |  |
| `artifact.policyengine_targets` PolicyEngine targets | `artifact` | `current` |  |
| `us.pipeline.resolve_policyengine_calibration_targets` Resolve PE calibration targets | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline._resolve_policyengine_calibration_targets` |
| `us.pipeline.select_feasible_policyengine_calibration_constraints` Select feasible PE constraints | `process` | `current` | `microplex_us.pipelines.us._select_feasible_policyengine_calibration_constraints` |
| `us.pipeline.select_policyengine_household_budget` Select household budget | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline._select_policyengine_household_budget` |

#### Edges

- `external.policyengine_targets_db` -> `us.pipeline.resolve_policyengine_calibration_targets` `external_source` (loads)
- `us.pipeline.resolve_policyengine_calibration_targets` -> `us.pipeline.select_feasible_policyengine_calibration_constraints` `data_flow` (filters)
- `us.pipeline.select_policyengine_household_budget` -> `us.pipeline.select_feasible_policyengine_calibration_constraints` `uses_utility` (limits)
- `us.pipeline.select_feasible_policyengine_calibration_constraints` -> `artifact.policyengine_targets` `produces_artifact` (selects)

### Run calibration

Run either PE-native table calibration or the generic dataframe calibration fallback.

- Substage ID: `07b_run_calibration`
- Canonical stage: `07_calibration`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `core.calibrator` Calibrator | `library` | `current` | `microplex.calibration.Calibrator`, `microplex.calibration.LinearConstraint` |
| `artifact.calibrated_data` Calibrated data | `artifact` | `current` |  |
| `artifact.calibration_summary` Calibration summary | `artifact` | `current` |  |
| `us.pipeline.calibrate_policyengine_tables` Calibrate PolicyEngine tables | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.calibrate_policyengine_tables` |
| `us.pipeline.calibrate` Calibrate generic tables | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.calibrate` |
| `artifact.policyengine_targets` PolicyEngine targets | `artifact` | `current` |  |
| `artifact.targets` Targets | `artifact` | `current` |  |

#### Edges

- `artifact.policyengine_targets` -> `us.pipeline.calibrate_policyengine_tables` `conditional` (PE-native path)
- `artifact.targets` -> `us.pipeline.calibrate` `conditional` (generic path)
- `core.calibrator` -> `us.pipeline.calibrate_policyengine_tables` `uses_library` (optimizes)
- `us.pipeline.calibrate_policyengine_tables` -> `artifact.calibrated_data` `produces_artifact` (writes)
- `us.pipeline.calibrate` -> `artifact.calibrated_data` `produces_artifact` (fallback)
- `artifact.calibrated_data` -> `artifact.calibration_summary` `produces_artifact` (summarizes)

## 08: Dataset assembly and publication

Assemble the calibrated output into the distributable PE dataset artifact.

### Export PolicyEngine dataset

Write the final calibrated entity tables into the PolicyEngine H5 dataset format.

- Substage ID: `08a_export_policyengine_dataset`
- Canonical stage: `08_dataset_assembly`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.policyengine_dataset` PolicyEngine H5 dataset | `artifact` | `current` |  |
| `us.pipeline.export_policyengine_dataset` Export PolicyEngine dataset | `process` | `current` | `microplex_us.pipelines.us.USMicroplexPipeline.export_policyengine_dataset` |
| `artifact.calibrated_data` Calibrated data | `artifact` | `current` |  |

#### Edges

- `artifact.calibrated_data` -> `us.pipeline.export_policyengine_dataset` `data_flow` (exports)
- `us.pipeline.export_policyengine_dataset` -> `artifact.policyengine_dataset` `produces_artifact` (writes)

### Save dataset bundle

Persist stage artifacts, manifests, registry/index entries, and versioned publication outputs.

- Substage ID: `08b_save_dataset_bundle`
- Canonical stage: `08_dataset_assembly`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.artifact_manifest` Artifact manifest | `artifact` | `current` |  |
| `artifact.versioned_bundle` Versioned bundle | `artifact` | `current` |  |
| `us.artifacts.save_us_microplex_artifacts` Save artifact bundle | `process` | `current` | `microplex_us.pipelines.artifacts.save_us_microplex_artifacts` |
| `us.artifacts.save_versioned_us_microplex_artifacts` Save versioned artifacts | `process` | `current` | `microplex_us.pipelines.versioned_artifacts.save_versioned_us_microplex_artifacts` |
| `artifact.policyengine_dataset` PolicyEngine H5 dataset | `artifact` | `current` |  |

#### Edges

- `artifact.policyengine_dataset` -> `us.artifacts.save_us_microplex_artifacts` `data_flow` (persists)
- `us.artifacts.save_us_microplex_artifacts` -> `artifact.artifact_manifest` `produces_artifact` (writes)
- `artifact.policyengine_dataset` -> `us.artifacts.save_versioned_us_microplex_artifacts` `conditional` (versioned run)
- `us.artifacts.save_versioned_us_microplex_artifacts` -> `us.artifacts.save_us_microplex_artifacts` `uses_utility` (delegates)
- `us.artifacts.save_versioned_us_microplex_artifacts` -> `artifact.versioned_bundle` `produces_artifact` (publishes)

## 09: Validation and benchmarking

Evaluate the assembled dataset and attach benchmark evidence.

### Load validation inputs

Load the final H5 and previously written artifact manifest for replayable validation.

- Substage ID: `09a_load_validation_inputs`
- Canonical stage: `09_validation_benchmarking`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.validation_inputs` Validation inputs | `artifact` | `current` |  |
| `us.policyengine.load_policyengine_us_entity_tables` Load PE entity tables | `process` | `current` | `microplex_us.policyengine.us.load_policyengine_us_entity_tables` |
| `artifact.policyengine_dataset` PolicyEngine H5 dataset | `artifact` | `current` |  |
| `artifact.artifact_manifest` Artifact manifest | `artifact` | `current` |  |

#### Edges

- `artifact.policyengine_dataset` -> `us.policyengine.load_policyengine_us_entity_tables` `data_flow` (loads)
- `artifact.artifact_manifest` -> `artifact.validation_inputs` `data_flow` (indexes)
- `us.policyengine.load_policyengine_us_entity_tables` -> `artifact.validation_inputs` `produces_artifact` (tables)

### Compute benchmark evidence

Run configured validation and benchmark evidence, including PE target harness and PE-native scores.

- Substage ID: `09b_compute_benchmark_evidence`
- Canonical stage: `09_validation_benchmarking`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.policyengine_harness` PolicyEngine harness | `artifact` | `current` |  |
| `artifact.policyengine_native_scores` PE-native scores | `artifact` | `current` |  |
| `artifact.benchmark_summary` Benchmark summary | `artifact` | `current` |  |
| `us.stage9.evaluate_policyengine_us_harness` Evaluate PE target harness | `process` | `current` | `microplex_us.policyengine.harness.evaluate_policyengine_us_harness` |
| `us.stage9.compute_pe_native_scores` Compute PE-native scores | `process` | `current` | `microplex_us.pipelines.pe_native_scores.compute_us_pe_native_scores` |
| `us.stage9.attach_rebuild_checkpoint_evidence` Attach PE rebuild evidence | `process` | `current` | `microplex_us.pipelines.pe_us_data_rebuild_checkpoint.attach_policyengine_us_data_rebuild_checkpoint_evidence` |
| `artifact.validation_inputs` Validation inputs | `artifact` | `current` |  |

#### Edges

- `artifact.validation_inputs` -> `us.stage9.evaluate_policyengine_us_harness` `validates` (target harness)
- `us.stage9.evaluate_policyengine_us_harness` -> `artifact.policyengine_harness` `produces_artifact` (writes)
- `artifact.validation_inputs` -> `us.stage9.compute_pe_native_scores` `validates` (native score)
- `us.stage9.compute_pe_native_scores` -> `artifact.policyengine_native_scores` `produces_artifact` (writes)
- `us.stage9.attach_rebuild_checkpoint_evidence` -> `artifact.benchmark_summary` `produces_artifact` (summarizes)

### Write validation evidence manifest

Index validation artifacts for saved-run overlays and allow Stage 9 replay without mutating Stage 8 outputs.

- Substage ID: `09c_write_validation_evidence`
- Canonical stage: `09_validation_benchmarking`
- Status: `current`

| Node | Type | Status | API refs |
| --- | --- | --- | --- |
| `artifact.validation_evidence_manifest` Validation evidence manifest | `artifact` | `current` |  |
| `us.stage9.build_validation_evidence_manifest` Build validation evidence manifest | `process` | `current` | `microplex_us.pipelines.stage_validation_evidence.build_us_validation_evidence_manifest` |
| `us.stage9.write_validation_evidence_manifest` Write validation evidence manifest | `process` | `current` | `microplex_us.pipelines.stage_validation_evidence.write_us_validation_evidence_manifest` |
| `us.stage9.replay_validation_benchmarking` Replay Stage 9 validation | `process` | `current` | `microplex_us.pipelines.stage9_replay.replay_us_stage9_validation_benchmarking` |
| `artifact.benchmark_summary` Benchmark summary | `artifact` | `current` |  |
| `artifact.validation_inputs` Validation inputs | `artifact` | `current` |  |

#### Edges

- `artifact.benchmark_summary` -> `us.stage9.build_validation_evidence_manifest` `data_flow` (indexes)
- `us.stage9.build_validation_evidence_manifest` -> `us.stage9.write_validation_evidence_manifest` `data_flow` (persists)
- `us.stage9.write_validation_evidence_manifest` -> `artifact.validation_evidence_manifest` `produces_artifact` (writes)
- `artifact.validation_inputs` -> `us.stage9.replay_validation_benchmarking` `conditional` (replay path)
- `us.stage9.replay_validation_benchmarking` -> `artifact.validation_evidence_manifest` `produces_artifact` (replays)
