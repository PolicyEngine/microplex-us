# Hugging Face artifact publishing

Microplex-US publishes completed artifact bundles to Hugging Face dataset repos.
The Hugging Face repos are the stable registry consumed by dashboards and
deployment tooling; local `artifacts/` paths are build outputs, not durable
interfaces.

## Repositories

Use dataset repos under the `policyengine` organization:

- `policyengine/microplex-us-diagnostics`
- `policyengine/microplex-us-deployed-datasets`

The diagnostics repo is the lightweight inspection registry. It stores loss
summaries, per-target diagnostics, audit sidecars, immutable run manifests, and
mutable discovery pointers.

The deployed-datasets repo stores the heavier PolicyEngine H5 bundle. Uploads
land in a staging path first, then a validated run can be promoted to the repo
root as the current deployed dataset.

## Repository layout

Diagnostics files:

```text
runs/<run_id>/manifest.json
runs/<run_id>/policyengine_native_scores.json
runs/<run_id>/pe_us_data_rebuild_native_audit.json
runs/<run_id>/pe_native_target_diagnostics.json
latest.json
run_registry.jsonl
```

Dataset files:

```text
staging/<run_id>/policyengine_us.h5
staging/<run_id>/manifest.json
policyengine_us.h5
manifest.json
```

The `runs/<run_id>/...` and `staging/<run_id>/...` paths are immutable once a
run is published. Only `latest.json`, `run_registry.jsonl`, and the promoted
dataset root files are expected to change.

## Dashboard contract

Dashboards should discover the default diagnostics bundle from:

```text
policyengine/microplex-us-diagnostics/latest.json
```

For reproducibility, dashboards should also support pinned diagnostics runs:

```text
policyengine/microplex-us-diagnostics/runs/<run_id>/...
```

The current deployed PolicyEngine dataset is:

```text
policyengine/microplex-us-deployed-datasets/policyengine_us.h5
```

A pinned staged dataset is:

```text
policyengine/microplex-us-deployed-datasets/staging/<run_id>/policyengine_us.h5
```

## Local publishing

Publish a full bundle after validation:

```bash
export HUGGING_FACE_TOKEN=...
uv run --extra hf --python 3.13 microplex-us-publish-hf-artifacts \
  artifacts/.../<run_id> \
  --run-id <run_id> \
  --publish-dataset \
  --promote-dataset
```

Run without uploading:

```bash
uv run --extra hf --python 3.13 microplex-us-publish-hf-artifacts \
  artifacts/.../<run_id> \
  --run-id <run_id> \
  --publish-dataset \
  --promote-dataset \
  --dry-run
```

The command writes `hf_publish_manifest.json` into the local artifact directory.
That file records the exact Hugging Face paths and operation counts that would
be committed or were committed.

Smoke-check a published bundle:

```bash
uv run --extra hf --python 3.13 microplex-us-smoke-hf-artifact \
  --run-id <run_id>
```

For staging-only dataset publishes that have not promoted root files yet:

```bash
uv run --extra hf --python 3.13 microplex-us-smoke-hf-artifact \
  --run-id <run_id> \
  --no-promoted-dataset
```

## GitHub Actions publishing

The `Publish Hugging Face Artifacts` workflow is a manual workflow. It accepts
either:

- an Actions artifact name containing an unpacked bundle or an archive, or
- an archive URL pointing to a `.zip`, `.tar`, `.tar.gz`, or `.tgz` bundle.

The workflow extracts the bundle, finds `manifest.json`, runs the focused
publisher tests, and invokes `microplex-us-publish-hf-artifacts`.
For real publishes, it then smoke-checks that the expected remote diagnostics
and dataset files are visible on Hugging Face.

The workflow defaults to `dry_run: true`. To publish, set `dry_run: false` and
provide a writable Hugging Face token through the repository secret `HF_TOKEN`.

Promotion is explicit:

- `publish_dataset: true` uploads the H5 and manifest to
  `staging/<run_id>/...`.
- `promote_dataset: true` also writes `policyengine_us.h5` and `manifest.json`
  at the dataset repo root.

## Validation before promotion

Before promoting a run as current, verify:

- `policyengine_us.h5` exists and loads with PolicyEngine-US.
- `manifest.json` points to all expected sidecars.
- `policyengine_native_scores.json` is real run output.
- `pe_us_data_rebuild_native_audit.json` is real run output.
- `pe_native_target_diagnostics.json` contains per-target rows.
- H5 weights are national scale for budget scoring.
- A PolicyEngine smoke test can compute representative baseline and reform
  aggregates.
- The dashboard can read the diagnostics bundle and, where needed, the H5.
