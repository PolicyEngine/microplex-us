# Stage contracts and manifests

The canonical stage registry lives in
`microplex_us.pipelines.stage_contracts`. It defines each stage's purpose,
expected inputs, outputs, artifacts, diagnostics, validation placeholders, and
resume mode.

Saved artifact bundles now include a `stage_manifest.json` sidecar. This file is
the machine-readable saved-run overlay for the stage taxonomy. It records the
canonical stages, status for the current run, artifact paths, diagnostics owned
by each stage, and the current resume posture.

The registry exposes two seam layers:

- `inputs` and `outputs` are structured stage resources. They identify artifact,
  config, manifest, runtime, and external-data dependencies with explicit keys.
- `consumes` and `produces` remain short human-readable summaries for diagrams
  and documentation.

Artifact `required` means required for a complete canonical saved bundle. It is
separate from `resume_role`, which says whether an existing artifact is useful
for diagnostics, manual replay, manual resume, or post-artifact validation.
Partial bundles can therefore still expose a valid replay boundary while the
manifest honestly reports that the complete publication bundle is incomplete.

## Legacy run-contract IDs

Older run-contract summaries and dashboard payloads used operational labels
such as `preflight`, `seed_build`, `donor_integration`,
`policyengine_materialization`, `calibration`, and `finalization`. New saved-run
views should report the canonical 9-stage IDs while preserving the old labels as
legacy provenance when present.

`canonicalize_us_pipeline_stage_id` maps those historical IDs into the stage
registry. The dashboard applies that mapping when reading `run_summary.json`, so
old and new runs sort into the same stage taxonomy instead of creating a second
parallel lifecycle.

## Resume artifacts

The first implementation is explicit rather than automatic. It writes reusable
boundary artifacts where the pipeline already has stable outputs:

- Stage 4: `stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet`
- Stage 5: `seed_data.parquet` and `synthetic_data.parquet`
- Stage 6: `stage_artifacts/06_policyengine_entities/`
- Stage 7: `calibrated_data.parquet`, `targets.json`, and
  `stage_artifacts/07_calibration/calibration_summary.json`
- Stage 8: `policyengine_us.h5`
- Stage 9: validation and benchmark evidence sidecars

The Stage 4 artifact is the scaffold-projected seed before donor integration. It
is a diagnostic and manual replay boundary, not an automatic conditional resume
point yet.

Conditional execution is intentionally not implemented yet. The stage manifest
and artifacts are designed to make that possible later without changing the
saved-run contract again.

## Artifact inventory and readiness

Saved bundles also expose two Stage 8 diagnostic sidecars:

- `stage_artifacts/artifact_inventory.json` lists canonical stage artifacts,
  whether each path exists, whether it was referenced by the run manifest, its
  resume role, size/file counts, and content hashes where hashing is practical.
- `stage_artifacts/conditional_readiness.json` summarizes which stage outputs
  are available for manual replay, manual resume, post-artifact evidence, or
  diagnostics only.

These reports are advisory. They do not skip or rerun stages, and they do not
silently accept stale artifacts. If a requested config is supplied to the
readiness builder, config mismatches are reported as `must_rerun`.

## Validation hooks

Each stage contract includes concise validation descriptors. These describe the
checks the stage should eventually own, but they do not run a shared validation
engine yet. That keeps this change focused on contracts, artifacts, and docs
without changing build behavior.
