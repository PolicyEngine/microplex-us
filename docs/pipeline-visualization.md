# Pipeline visualization

`microplex-us` exposes the canonical US build path as generated graph data,
substage machinery data, and saved-run overlays. The stage graph is static and
comes from the stage registry. The machinery layer comes from
[`us_pipeline_internals.map.json`](./us_pipeline_internals.map.json) plus
`@pipeline_node` decorators in the runtime code. An overlay is run-specific
evidence derived from `stage_manifest.json`, per-stage manifests, and validation
or benchmark evidence.

## Generated data

The public generated files live under `docs/generated/`:

- [`us_pipeline_graph.json`](./generated/us_pipeline_graph.json): the canonical
  9-stage graph.
- [`us_pipeline_internals.json`](./generated/us_pipeline_internals.json): the
  substage, function, artifact, library, external-source, and validation graph
  for each canonical stage.
- [`us_pipeline_graph.schema.json`](./generated/us_pipeline_graph.schema.json):
  schema for graph consumers.
- [`us_pipeline_internals.schema.json`](./generated/us_pipeline_internals.schema.json):
  schema for substage machinery graph consumers.
- [`us_pipeline_overlay.schema.json`](./generated/us_pipeline_overlay.schema.json):
  schema for saved-run overlays.

Small complete and failed overlay fixtures live under
`tests/fixtures/pipeline_docs/generated/`. Those are test fixtures, not
production run evidence.

Regenerate the public graph, machinery map, and schemas with:

```bash
microplex-us-generate-pipeline-docs --output-dir docs/generated
```

Check that the committed generated files are current with:

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
target diagnostics page. The left menu switches between the full pipeline view
and individual stage views. The full pipeline view shows the canonical stage
graph; individual stage views show the deeper machinery graph and retain the
substage selector for focusing on one function/artifact path.

Run the viewer locally with:

```bash
cd pipeline_viewer
npm install
npm run dev
```

The viewer ships with fixture graph, internals, and overlay JSON files. Use the
file loaders in the header to inspect different generated graph data or a
saved-run overlay.

After regenerating `docs/generated/us_pipeline_internals.json`, update the
viewer fixture with:

```bash
cp docs/generated/us_pipeline_internals.json \
  pipeline_viewer/src/fixtures/us_pipeline_internals.json
```

When durable build machinery changes, update the authored map and decorators:

- add or revise substages, artifact nodes, library nodes, and edges in
  `docs/us_pipeline_internals.map.json`
- add `@pipeline_node(...)` to stable runtime functions that should appear as
  function nodes
- rerun `microplex-us-generate-pipeline-docs --output-dir docs/generated`

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
