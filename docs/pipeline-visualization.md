# Pipeline visualization

`microplex-us` exposes the canonical US build path as generated graph data plus
saved-run overlays. The graph is static and comes from the stage registry. An
overlay is run-specific evidence derived from `stage_manifest.json`, per-stage
manifests, and validation or benchmark evidence.

## Generated data

The public generated files live under `docs/generated/`:

- [`us_pipeline_graph.json`](./generated/us_pipeline_graph.json): the canonical
  9-stage graph.
- [`us_pipeline_graph.schema.json`](./generated/us_pipeline_graph.schema.json):
  schema for graph consumers.
- [`us_pipeline_overlay.schema.json`](./generated/us_pipeline_overlay.schema.json):
  schema for saved-run overlays.

Small complete and failed overlay fixtures live under
`tests/fixtures/pipeline_docs/generated/`. Those are test fixtures, not
production run evidence.

Regenerate the public graph and schemas with:

```bash
microplex-us-generate-pipeline-docs --output-dir docs/generated
```

Check that the committed graph and schemas are current with:

```bash
microplex-us-generate-pipeline-docs --output-dir docs/generated --check
```

Generate an overlay for a saved run with:

```bash
microplex-us-generate-pipeline-docs \
  --output-dir /tmp/microplex-us-pipeline-overlay \
  --artifact-root /path/to/saved/artifact
```

## Viewer

The separate `pipeline_viewer/` app renders the generated graph with ELK-routed
React Flow edges. It intentionally does not reuse the existing `dashboard/`
target diagnostics page.

Run the viewer locally with:

```bash
cd pipeline_viewer
npm install
npm run dev
```

The viewer ships with small fixture graph and overlay JSON files. Use the file
loaders in the header to inspect a different graph or saved-run overlay.

Build-check the viewer with:

```bash
cd pipeline_viewer
npm run build
```

## Overlay contents

Overlays include concise run evidence:

- stage readiness and lifecycle status
- artifact references and existence flags
- diagnostic keys
- validation hook status
- compact metrics
- resume posture
- failure details when a stage failed

Overlays do not include raw microdata, secrets, large diagnostic tables, or
absolute local paths. Consumers should resolve relative artifact references
against the saved artifact directory they are inspecting.

## Core `microplex` links

This documentation treats core `microplex` APIs as upstream library concepts.
The US docs explain how those concepts appear in the US build and link to the
published core API docs through intersphinx where possible. Shared source,
fusion, synthesis, calibration, and benchmark primitives should remain
documented in core instead of being mirrored here.

Useful core concepts for the US stages:

| US stage | Core concept |
| --- | --- |
| Stage 2 | Observation frames and source descriptors |
| Stage 3 | Fusion planning and source coverage |
| Stage 5 | Synthesis and donor integration primitives |
| Stage 7 | Calibration targets and solvers |
| Stage 9 | Benchmark artifacts and comparison helpers |
