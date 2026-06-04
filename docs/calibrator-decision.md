# Calibrator decision

*Decided: 2026-04-16. Applies to `spec-based-ecps-rewire` and every microplex-us pipeline that follows.*

## Context

Three calibration systems exist in the microplex / PolicyEngine ecosystem:

| System | Location | Method | Scale notes |
|---|---|---|---|
| `microplex.calibration.Calibrator` | microplex core, ~2011 lines | Classical IPF / chi-square / entropy balancing, with `LinearConstraint` for explicit constraint rows | Entropy backend just killed v6 at 1.5M households |
| `microplex.reweighting.Reweighter` | microplex core, 506 lines | Sparse L0/L1/L2 with scipy and cvxpy backends | Unused in production; designed for geographic-hierarchy reweighting; enforces sparsity by construction |
| `microcalibrate` | PolicyEngine external package | Gradient-descent chi-squared with soft penalties and optional feasibility filtering | Used by PE-US-data for its main calibration; has production track record |

v6 died inside `Calibrator.fit_transform(..., backend="entropy")` on a 1.5M-household frame. The underlying problem is not the Calibrator code — it is that entropy calibration instantiates dense-ish structures at `(n_households × n_constraints)` scale, and with ~1,255 constraints that exceeds what a 48 GB machine can hold once scratch memory is included.

## Decision

**Mainline calibrator for all production runs: `microcalibrate` (gradient-descent chi-squared).**

**Optional sparse deployment selector applied *after* mainline calibration: `microplex.reweighting.Reweighter` with L0/HardConcrete backend**, used only when a deployment artifact (web app, embedded tool) needs a ~50k-record subsample of a national build.

**Retire for production use: `microplex.calibration.Calibrator` with `backend="entropy"` at scales above ~200k records.** The classical Calibrator's IPF and chi-square backends stay available for small-scale work, diagnostics, and test harnesses where their explicit constraint semantics are convenient.

## Why `microcalibrate` and not core `Calibrator`

1. **Identity preservation.** `microcalibrate` adjusts per-record weights via gradient descent without materializing dense constraint Jacobians. Every input record survives to the output with a new weight. The rearchitecture's longitudinal extension (SS-model) requires stable entity identity across years; identity-preservation cannot be negotiable.
2. **Scalability at the target scale.** `microcalibrate` is the calibration stack PE-US-data actually uses for production enhanced-CPS builds at full scale. v6's death at 1.5M is direct evidence the entropy path doesn't scale; `microcalibrate`'s gradient-descent pattern does.
3. **Soft-penalty feasibility handling.** The 2026-03-30 review flagged that v2's calibration dropped 65 % of constraints as infeasible and then scored against the full target set, producing a systematic loss inflation. `microcalibrate` supports soft penalty weights on targets the solver cannot feasibly hit, giving principled rather than binary drop behavior.
4. **External track record.** The SS-model methodology doc explicitly names `microcalibrate` as the calibration tool for the longitudinal extension. Picking it now aligns cross-section with the planned longitudinal path.

## Why `Reweighter` stays as a post-mainline optional stage

1. **L0 sparsity serves deployment, not accuracy.** The right use of L0 is to produce a small subsample of a well-calibrated national dataset for constrained deployment targets (web app UI, mobile, static hosting). It is the wrong tool for "calibrate to hit targets" because it sacrifices exact match for sparsity.
2. **Apply after, not instead of, the mainline.** The mainline run produces ~1.5M records with adjusted weights. If a deployment needs 50k records, apply `Reweighter` with appropriate L0 λ as a second pass. The mainline artifact remains the ground-truth output for analysis.
3. **`SparseCalibrator` + `HardConcreteCalibrator` analysis on the `codex/core-semantic-guards` paper work showed HardConcrete dominates the sparse-calibration Pareto frontier**, so when the sparse step does run, HardConcrete is the preferred backend. Core already ships this with multi-seed evaluation.

## Why `Calibrator` is retired at scale

1. v6 proves `Calibrator(backend="entropy")` OOMs at 1.5M × 1.2k-constraint scale on a 48 GB workstation. v4 proved it at 1.5M × similar scale.
2. No architectural fix is cheap. To make entropy work at that scale we would have to rewrite the backend to use sparse constraint matrices and streaming gradient, which is effectively reimplementing `microcalibrate`.
3. `Calibrator` stays available and useful for small-scale test harnesses. It is still the right tool for `n < ~200k`, for unit tests of the calibration layer, and for explicit-constraint diagnostics (the `LinearConstraint` API is clean).

## Implementation implication

The rewired pipeline in `spec-based-ecps-rewire` will import `microcalibrate` as a real dependency (not optional). This is a net-new dependency on microplex-us. The audit entry that proposed "retire `microcalibrate` if `Calibrator` covers the scalability requirement" is overruled by v6's evidence.

## Calibration architecture, in order

```
raw seed data  ─►  donor integration  ─►  seed_ready
                                          │
                                          ▼
                                  synthesize (seed backend = copy)
                                          │
                                          ▼
                                  support enforcement
                                          │
                                          ▼
                                  policyengine entity tables (households, persons, tax_units, ...)
                                          │
                                          ▼
                      ┌──────────────────┴──────────────────┐
                      │  MAINLINE (every run)               │
                      │  microcalibrate.Calibrator          │
                      │    - chi-squared distance           │
                      │    - gradient descent               │
                      │    - soft penalty for infeasibles   │
                      │    - preserves all record IDs       │
                      │                                     │
                      │  Hierarchical in later phases:      │
                      │    national → state → stratum       │
                      └───────────────────┬─────────────────┘
                                          │
                                          ▼
                                  calibrated artifact (full scale)
                                          │
                                          ▼
                      ┌───────────────────┴─────────────────┐
                      │  OPTIONAL SPARSE DEPLOYMENT STEP    │
                      │  microplex.reweighting.Reweighter   │
                      │    - L0 / HardConcrete              │
                      │    - deployment-scale subsample     │
                      │  Only when a deployment artifact    │
                      │  needs to be small.                 │
                      └─────────────────────────────────────┘
```

## Hierarchical calibration — separate decision, deferred

This decision only picks the calibration *backend*. Hierarchical geographic calibration (national → state → stratum, with spatial smoothness priors, optional Fay-Herriot small-area composites) is a structure layered on top of `microcalibrate` and will be decided in its own doc at the start of the local-area gate (G2). Cross-section gate (G1) calibrates at national scale first.

## Does this close out the three-way overlap?

Yes, operationally:

- Production runs: `microcalibrate`.
- Deployment subsampling: `Reweighter`.
- Tests and small-scale diagnostics: `Calibrator`.
- No single-pipeline run crosses all three. Each tool has a distinct and non-overlapping job.

## Empirical support: sparse selection annihilates rare subpopulations

The single cleanest empirical argument for this split comes from
`microplex/benchmarks/results/sparse_coverage.csv`. Measuring rare-subpopulation
preservation at varying sparsity levels (lower `coverage_median` = closer to
oracle):

| Method | `coverage_median` | elderly_selfemp_ratio | young_dividend_ratio |
|---|---:|---:|---:|
| Oracle (full) | 0.009 | 0.94 | 1.11 |
| Generative (10%) | 0.53 | 27.7 | 20.6 |
| Generative (2%) | 0.42 | 22.1 | 32.3 |
| Generative (1%) | 0.25 | 7.2 | 1.7 |
| Weighted (10%) | 0.24 | **0.00** | **0.00** |
| Weighted (2%) | 0.35 | 0.02 | **0.00** |
| Weighted (1%) | 0.65 | **0.00** | **0.00** |

Sparse L0 weighting drops rare subpopulations to **zero representation** at
every sparsity level tested. Generative synthesis preserves them at 7–30× the
oracle ratio. For policy analysis, where rare subpopulations (elderly
self-employed, young dividend earners, disability recipients, top-1% earners)
drive outsized fiscal and distributional effects, sparse-as-mainline is
non-viable on accuracy grounds alone.

This empirical pattern reinforces the decision above: L0/sparse selection is a
**post-calibration deployment tool**, not a calibration method. Apply it after
the mainline `microcalibrate` run has produced a fully-covered adjusted-weight
artifact, and only when a downstream consumer needs a small subsample.

### Scale caveat

`sparse_coverage.csv` was produced on **10,000-row synthetic data with ~7
variables**. Production scale is 1.5M rows × 150+ variables on real joint
microdata. We should not assume the 20–30× generative-vs-weighted gap holds at
that scale — the absolute numbers will shift, and rare-subpopulation
preservation may tighten for both methods. What is expected to hold is the
structural pattern: sparse L0 exactly zeros out records, generative synthesis
does not. The argument against sparse-as-mainline survives any plausible
scale-up because the failure mode (zero representation of rare cells) is not a
noise issue, it is mathematically baked into L0 selection.

## What this unblocks

- Migration step 2 of `docs/core-wiring-audit.md`: "Adopt `Calibrator` end-to-end" is revised to "Adopt `microcalibrate` end-to-end as the production calibrator." That becomes the first real code change in `spec-based-ecps-rewire`.
- The rewired cross-section pipeline can start being written against a concrete calibration contract.

## Revisit conditions

Revisit this decision if any of the following becomes true:

1. A benchmark shows `microcalibrate` produces materially worse loss than a refactored `Calibrator` on representative constraint matrices. (Unlikely — PE uses it successfully.)
2. Licensing / availability of `microcalibrate` becomes a blocker for external consumers of microplex-us. (Mitigate by forking the needed subset into microplex core.)
3. The SS-model longitudinal extension requires a calibration primitive that `microcalibrate` does not provide (e.g., explicit spatial smoothness, per-year temporal regularization). Add the primitive at microplex level rather than swapping backends.

## Update 2026-06-04: optimizers added since the original decision

Two optimizer paths landed after this doc was written and were not covered above. Recording them here so the optimizer landscape is legible in one place.

### `pe_l0` backend (eCPS-parity L0)
`calibration_backend="pe_l0"` selects `microplex_us.pipelines.pe_l0.PolicyEngineL0Calibrator`, a lazy wrapper around PE-US-data's `fit_l0_weights` (the incumbent enhanced-CPS L0 sparse solver). Added to run the MP build under the same calibration the published eCPS uses, for apples-to-apples comparison.

### PE-native-loss refit / scoring — `optimize_pe_native_loss_weights`
`microplex_us.pipelines.pe_native_optimization.optimize_pe_native_loss_weights` refits household weights **directly on the PolicyEngine-native target loss** (used in the eCPS-replacement symmetric-refit comparison and scoring — *not* the dataset build).

- **Algorithm: projected (proximal) gradient descent** — least-squares PE-native loss, gradient step, project onto the nonnegative budget simplex, Lipschitz step size + backtracking line search, monotone-descent acceptance.
- **Naming caveat:** commit history ("Add robust PE-native loss and APG refit") and earlier notes call this "APG". That is a **misnomer** — there is no Nesterov momentum/extrapolation term, so it is **plain projected GD, not accelerated proximal gradient**. Do not assume acceleration from the name.

### Current optimizer landscape (one place)
| Job | Optimizer | Method |
|---|---|---|
| Dataset-build weight calibration (mainline) | `microcalibrate` / `HardConcreteCalibrator` | gradient-descent chi-squared + optional L0 (HardConcrete) |
| Dataset-build (small-scale / diagnostics) | `microplex.calibration.Calibrator` | entropy balancing (scipy KL) / IPF (raking) / chi-square |
| Dataset-build (eCPS-parity) | `pe_l0` → `PolicyEngineL0Calibrator` | PE-US-data `fit_l0_weights` (L0 sparse) |
| eCPS-replacement refit / scoring | `optimize_pe_native_loss_weights` | projected (proximal) gradient descent on the PE-native loss ("APG" in commits = misnomer; not accelerated) |
