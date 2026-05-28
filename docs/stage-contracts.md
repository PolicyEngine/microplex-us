# Stage contracts and manifests

The canonical stage registry lives in
`microplex_us.pipelines.stage_contracts`. It defines each stage's purpose,
expected inputs, outputs, artifacts, diagnostics, validation placeholders, and
resume mode.

Saved artifact bundles now include a `stage_manifest.json` sidecar. This file is
the machine-readable saved-run overlay for the stage taxonomy. It records the
canonical stages, status for the current run, artifact paths, diagnostics owned
by each stage, and the current resume posture.

## Resume artifacts

The first implementation is explicit rather than automatic. It writes reusable
boundary artifacts where the pipeline already has stable outputs:

- Stage 4: `seed_data.parquet`
- Stage 5: `synthetic_data.parquet`
- Stage 6: `stage_artifacts/06_policyengine_entities/`
- Stage 7: `calibrated_data.parquet`, `targets.json`, and
  `stage_artifacts/07_calibration/calibration_summary.json`
- Stage 8: `policyengine_us.h5`
- Stage 9: validation and benchmark evidence sidecars

Conditional execution is intentionally not implemented yet. The stage manifest
and artifacts are designed to make that possible later without changing the
saved-run contract again.

## Validation hooks

Each stage contract includes concise validation descriptors. These describe the
checks the stage should eventually own, but they do not run a shared validation
engine yet. That keeps this change focused on contracts, artifacts, and docs
without changing build behavior.
