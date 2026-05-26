# microplex-us

US-specific survey adapters, calibration targets, pipelines, and PolicyEngine integration
built on top of the generic `microplex` engine.

## Docs

- [Docs index](./docs/README.md)
- [Architecture](./docs/architecture.md)
- [Source semantics](./docs/source-semantics.md)
- [Imputation conditioning contract](./docs/imputation-conditioning-contract.md)
- [Benchmarking](./docs/benchmarking.md)
- [Methodology ledger](./docs/methodology-ledger.md)
- [PolicyEngine oracle compatibility path](./docs/policyengine-oracle-compatibility.md)
- [PE construction parity](./docs/pe-construction-parity.md)
- [Superseding `policyengine-us-data`](./docs/superseding-policyengine-us-data.md)

## Diagnostics dashboard

The static dashboard in `dashboard/` loads the full PE-native per-target
diagnostic JSON written by:

```bash
microplex-us-pe-native-target-diagnostics \
  --from-dataset /path/to/enhanced_cps_2024.h5 \
  --to-dataset /path/to/policyengine_us.h5 \
  --policyengine-targets-db /path/to/policy_data.db \
  --output-path artifacts/pe_native_target_diagnostics_current.json
```

The dashboard uses the exported PolicyEngine design tokens from
`@policyengine/config/theme.css`; run `python scripts/sync_policyengine_theme.py --check`
to verify the local browser-readable token copy is still synced.
When a PolicyEngine target DB is available, the JSON annotates PE-native legacy
labels with structured target IDs and flags legacy-only gaps.

## Current focus

`microplex-us` is being built as a library-first US runtime with
`policyengine-us` as the shared measurement operator and
`policyengine-us-data` as the incumbent comparator, not as the thing we are
trying to clone wholesale:

- canonical source and target metadata
- PE-US-compatible export
- full-target benchmarking against the active targets DB
- run registry and DuckDB index for frontier analysis

The architecture is still evolving, so the docs are deliberately technical and
operational rather than paper-like.

Method-level decomposable-family bakeoffs now live in the sibling eval repo:
`/Users/maxghenis/PolicyEngine/microplex-evals`. `microplex-us` should keep the
runtime helpers and pipeline-adjacent diagnostics, not the long-lived eval
orchestration and artifact curation.
