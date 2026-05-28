# PolicyEngine Oracle Compatibility Path

This document states the current execution rule for the incumbent-compatibility
track:

> Use `policyengine-us-data` as the incumbent comparator and use
> `policyengine-us` plus the active targets DB as the shared measurement
> oracle. Match incumbent behavior where that sharpens attribution or closes an
> interface contract. Keep `microplex-us` as an independent runtime, and treat
> materially different modeling choices as explicit challenger modes rather
> than calling the whole project a PE-US-data clone.

That is a stricter rule than either "make Microplex mimic PE-US-data wholesale" or
"make it better however we can."

Historical note:

- some internal module names still use `pe_us_data_rebuild`
- that reflects the original implementation thread, not the methodological
  claim
- the claim now is oracle compatibility plus incumbent comparison, not
  wholesale reconstruction

## Current runtime entry points

The incumbent-compatibility track currently uses historically named runtime
entry points in
[`pe_us_data_rebuild.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/pe_us_data_rebuild.py):

- `default_policyengine_us_data_rebuild_config(...)`
- `default_policyengine_us_data_rebuild_source_providers(...)`
- `build_policyengine_us_data_rebuild_pipeline(...)`

And it now has one concrete saved-run checkpoint runner in
[`pe_us_data_rebuild_checkpoint.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/pe_us_data_rebuild_checkpoint.py):

- `default_policyengine_us_data_rebuild_checkpoint_config(...)`
- `default_policyengine_us_data_rebuild_queries(...)`
- `attach_policyengine_us_data_rebuild_checkpoint_evidence(...)`
- `run_policyengine_us_data_rebuild_checkpoint(...)`

These make the incumbent-comparison path callable as a first-class Microplex
profile rather than a loose collection of remembered settings.

That profile now also includes:

- the PE-style PUF Social Security QRF split mode
- the PE-style prespecified donor-predictor mode for source imputations
- opt-in ACS/SCF donor providers plus a block-spec-driven SIPP donor provider
  for the compatibility path
- one shared donor-block manifest,
  [`pe_source_impute_blocks.json`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/manifests/pe_source_impute_blocks.json),
  that now drives both:
  - donor-survey adapter specs in
    [`donor_surveys.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/data_sources/donor_surveys.py)
  - the PE-style prespecified predictor and condition-preparation surface in
    [`us.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/us.py)
  - SIPP donor-block postprocessing such as month filtering, annualization, and
    household child-count features
  - SIPP raw-file extraction details such as file names, delimiters, ID parts,
    raw column mappings, and simple indicator derivations
  - ACS/SCF subprocess dataset-loader details such as dataset module/class,
    table-builder mode, and canonical variable mappings
  - one explicit PE source-impute execution boundary in
    [`pe_source_impute_engine.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pe_source_impute_engine.py)
    so `us.py` no longer owns PE block resolution, PE block-frame preparation /
    entity projection, condition-surface prep, the prespecified block
    fit/generate/match loop, or a second duplicated generic donor execution loop
  - one saved-run parity sidecar in
    [`pe_us_data_rebuild_parity.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/pe_us_data_rebuild_parity.py)
    that records profile conformance, the exact incumbent baseline slice, and
    the harness / PE-native verdicts for one artifact bundle
  - one saved-run native audit sidecar in
    [`pe_us_data_rebuild_audit.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/pe_us_data_rebuild_audit.py)
    that records family regressions, target-level regressions, support audits,
    and imputation-sidecar verdict hints for the same artifact bundle
  - one checkpoint runner that saves a normal versioned Microplex artifact
    bundle first, then attaches harness/native parity evidence from the saved
    dataset, and finally materializes parity / native-audit sidecars from the
    updated bundle instead of relying on an ad hoc notebook or shell sequence

## Current calibration rule

Oracle compatibility does not mean "optimize against every DB row in one flat
solve."

Current rule:

- measure every serious run against the full active PolicyEngine targets DB
- compute full-oracle loss with an explicit penalty for unsupported target rows
- keep supported-only diagnostics visible as a separate channel
- let the calibration planner classify target rows into:
  - `solve_now`
  - `solve_later`
  - `audit_only`
- allow narrow deferred later passes and keep them only when they improve the
  current full-oracle score

This keeps the full DB as the shared oracle without pretending that every DB
row should always be an active calibration constraint in the same numerical
stage.

## Why this rule exists

If we mix:

- oracle-compatibility checks
- model-family changes
- predictor-surface changes
- weighting-backend changes
- calibration-objective changes

in the same pass, then we lose attribution.

We may still end up with a better system, but we will not know whether it is:

- a closer match to incumbent interface behavior
- a materially different model stack
- or both

Related guardrail:

- do not treat a late export-layer port of an upstream PE concept as a
  substitute for upstream construction parity
- if a concept properly belongs in source construction, tax-unit construction,
  or source/family imputation, an export-layer patch can still be useful as a
  probe, but it should be recorded as a challenger implementation boundary
  rather than silently promoted into the default path

The incumbent-compatibility track is meant to answer a simpler question first:

> Can Microplex produce a PE-ingestable dataset under a documented
> incumbent-compatible profile, in a cleaner and more auditable form, while
> keeping the differences from the incumbent attributable?

## What is allowed in the incumbent-compatibility pass

Allowed changes are architectural improvements that should move outputs only on
the margin:

- replacing implicit scripts with explicit source/provider contracts
- turning inline special cases into declarative stage specs
- wrapping incumbent PE weighting backends behind Microplex interfaces
- making parity assumptions explicit in docs, tests, and artifacts
- adding provenance, parity audits, and stage-level artifacts
- reorganizing code so country-pack boundaries and pipeline ownership are clear

These are improvements in:

- maintainability
- provenance
- portability
- reproducibility
- evaluation discipline

without trying to win by silently changing the underlying comparison contract.

## What is not allowed by default in the incumbent-compatibility pass

These should be treated as explicit departures, not silent cleanup:

- changing model class
  - e.g. replacing incumbent QRF stages with grouped-share or forest-share
- materially changing predictor surfaces
  - e.g. replacing a PE-style prespecified predictor set with a broader
    data-driven feature search
- changing fallback heuristics in ways likely to move support or totals
- changing weighting/calibration objectives or optimization backends
- introducing new target surfaces as if they were still measuring the same
  incumbent comparison problem

Any such change can still be good. It just belongs in the challenger phase,
where it is measured as an intentional departure.

## Practical decision rule

When we face a design choice during the incumbent-compatibility pass:

1. Ask whether the incumbent PE-US-data behavior is clear enough to reproduce.
2. If yes, match it inside cleaner Microplex structure.
3. Only deviate if the incumbent choice would create an obvious architectural
   problem.
4. If we deviate, choose the smallest alternative that should change outputs
   only on the margin.
5. Write the deviation down as `intentional` rather than letting it masquerade
   as oracle compatibility.

One important corollary:

- rejecting a late export-layer patch does **not** automatically reject the
  underlying model concept
- sometimes the concept is right, but the implementation point is wrong
- in that case the concept stays in scope, but it must be reintroduced at the
  correct upstream layer rather than as a last-minute PE-input bolt-on

## Examples

### Good incumbent-compatibility changes

- Keep PE's QRF family, but call it through a Microplex method spec instead of
  an inline script.
- Keep the incumbent predictor set, but declare it in one stage config rather
  than scattering it across files.
- Keep PE's donor-survey blocks, but declare them once in a shared manifest
  instead of hardcoding ACS/SIPP/SCF surfaces separately in both provider code
  and pipeline code.
- Keep PE's donor-block postprocessing rules, but attach them to the same block
  specs instead of baking month filters and annualization logic into ad hoc
  loader branches.
- Keep raw donor-file mappings close to the block spec, so file names, raw
  columns, and identifier assembly stop being copied across multiple SIPP
  loaders.
- Keep subprocess dataset-loader mappings close to the block spec too, so ACS
  and SCF import/class/table-shaping contracts stop living as large inline
  script blobs.
- Keep the PE weighting backend, but call it through a Microplex-owned adapter.
- Keep the same CPS reason-code logic, but express it in a source adapter with
  explicit parity tests.

### Too far for the incumbent-compatibility pass

- switching from PE's prespecified QRF predictors to a broader automatic
  feature search because it seems statistically cleaner
- replacing the incumbent SS split model with a new forest-share family before
  we have a parity implementation
- changing the calibration objective because the incumbent one is inconvenient

## Relationship to the parity matrix

This rule is what makes the parity matrix meaningful.

If we follow it, then the matrix statuses:

- `Exact`
- `Close`
- `Compatible, not equivalent`
- `Different`

actually describe the comparison contract.

If we ignore it, the matrix becomes unstable because every "cleanup" quietly
changes the underlying model contract.

## Relationship to later outperformance work

This rule does **not** say we should stop improving the pipeline.

It says:

1. make the incumbent-comparison path explicit and auditable
2. prove parity or intentional difference where parity matters
3. then run challenger methods against that incumbent-compatible baseline

That sequence is what gives later benchmark wins credibility.
