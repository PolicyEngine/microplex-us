# Canonical US pipeline stages

`microplex-us` uses a 9-stage runtime taxonomy for the canonical US dataset
build. The stages describe the operational lifecycle of a build; they are
separate from parity or migration roadmaps against incumbent data packages.

```text
1. Run profile
   -> 2. Source contracts and loading
   -> 3. Source planning, fusion planning, and scaffold selection
   -> 4. Seed construction and donor integration
   -> 5. Synthesis, candidate population, and support enforcement
   -> 6. PolicyEngine entity construction and microsimulation materialization
   -> 7. Target resolution, selection, and calibration
   -> 8. Dataset assembly and publication
   -> 9. Validation and benchmarking
```

## Stage 1: Run profile, config, and source bundle

Defines the build that is about to run: profile, providers, target period,
target database, baseline dataset, sample filters, random seeds, and defer or
checkpoint options.

## Stage 2: Source contracts and source loading

Turns external datasets into Microplex observation frames with source metadata,
entity tables, and relationships. This includes CPS, PUF, ACS, SIPP, SCF, and
any construction loaders still backed by other packages.

## Stage 3: Source planning, fusion planning, and scaffold selection

Reasons about the source mix: variable coverage, scaffold selection, donor
sources, and variable families that need donor integration or synthetic
generation.

## Stage 4: Seed construction and donor integration

Projects the scaffold into the canonical seed schema and integrates donor
variables, conditioning surfaces, exclusions, and authoritative overrides.

## Stage 5: Synthesis, candidate population, and support enforcement

Produces the candidate population that will be calibrated. This may be seed
passthrough, bootstrap synthesis, or model-backed synthesis, depending on the
selected backend.

## Stage 6: PolicyEngine entity construction and microsimulation materialization

Builds PolicyEngine-facing households, persons, tax units, SPM units, families,
and marital units. This stage owns PE entity integrity and materialized PE input
readiness before calibration/export.

## Stage 7: Target resolution, selection, and calibration

Loads and filters targets, materializes target variables, selects feasible
constraints, solves weights, and records target-fit diagnostics.

## Stage 8: Dataset assembly and publication

Maps calibrated tables to export variables, writes the final H5 dataset, and
records the saved artifact bundle metadata.

## Stage 9: Validation and benchmarking

Evaluates the assembled dataset with harness outputs, native scores, audits,
ablation evidence, and run registry/index evidence.
