# Build manifest spec (v1)

A single, authoritative, human-readable record of **what a microdata dataset is
and how to reproduce it** — designed to work for both `policyengine-us-data`
(eCPS) and `microplex-us`. It is the legibility layer the project keeps needing
and never finished.

## Why this exists (the failures it prevents)

Every mistake made while reviewing these builds came from the same root cause:
*you cannot answer "what is this dataset and how was it made" from one place.*
Concretely, in one day:

| What went wrong | Field that prevents it |
|---|---|
| A build dir named `puf100k-acs100k` was actually CPS+ACS | `population.spines` (authoritative, emitted by the build) |
| Benchmarked a non-default ACS experiment thinking it was the default | `build.id` + `build.derives_from` + `build.delta_label` (no anonymous artifacts) |
| The "best" imputer (`regime_aware`) wasn't the one that ran (`qrf`); env couldn't even import it | `build.environment` + per-variable `method` |
| Confused "ACS as donor" vs "ACS as spine" | `population.spines` vs `donors` (hard split) |
| Couldn't tell which donors the default wires (PUF only? +SCF/SIPP?) | `donors[]` list, emitted not assumed |
| Couldn't tell if partnership income is kept/imputed/overridden | `variables[].treatment` (per-variable provenance) |
| "mp beats eCPS 0.094" was a 3×-records + one-sided-refit artifact | `population.total_households` + the scoreboard refusing to compare without manifests |

The manifest makes each of these a field you read in two minutes, not a forensic
reconstruction off misleading artifacts.

## Design contract (the invariants that make it trustworthy)

1. **Emitted, not hand-written.** The build pipeline *produces* the manifest as
   an output artifact. Hand-maintained docs drift (that's how we got here).
2. **Reproducible from the manifest alone.** `build.reproduce.command` +
   `build.code_ref` + `build.environment` must be sufficient to rebuild it.
3. **Bound to its output.** `outputs.dataset.sha256` ties the manifest to the
   exact H5. A validator recomputes `measured.*` from the H5 and asserts it
   matches `population`/`composition`; mismatch = the manifest is a lie, fail.
4. **No anonymous builds.** Every build has a stable, meaningful `id`. Every
   experiment is a **delta**: `derives_from` a canonical build + a `delta_label`
   naming exactly what changed. (One canonical national build; everything else
   is a named diff.)
5. **Spine vs donor is explicit and exhaustive.** A source either contributes
   **rows** (`population.spines`) or **imputes variables** (`donors`) — never
   ambiguous. Every released household traces to a spine.
6. **Per-variable provenance is mandatory.** For every modeled variable: where
   it comes from and how it's treated (`kept` / `imputed_on_clone` /
   `overridden_everywhere` / `imputed_from_donor` / `derived`).
7. **Unconfirmed = pending, never asserted.** If the build can't determine a
   field, it writes `pending` — it does not guess. (Same rule for humans.)

## Schema (annotated YAML)

```yaml
manifest_version: 1

build:
  id: mp-national-canonical-2024        # stable, meaningful; NOT a timestamp blob
  engine: microplex-us                  # microplex-us | policyengine-us-data
  derives_from: null                    # parent build id, if this is a delta
  delta_label: null                     # e.g. "canonical + acs-spine"; what changed
  period: 2024
  created_at: 2026-05-29T12:00:00Z
  code_ref:
    repo: PolicyEngine/microplex-us
    git_sha: <sha>
    dirty: false                        # uncommitted changes present at build time?
  environment:                          # captures "the imputer that can't import" trap
    python: "3.13"
    key_packages: { microimpute: "1.1.2", policyengine-us: "1.715.1" }
  reproduce:
    command: "uv run --extra policyengine python -m microplex_us.pipelines.pe_us_data_rebuild --config national_canonical"
    config_ref: configs/national_canonical.yaml

population:                             # the ROWS — what's actually in the file
  frame: cps_asec                       # base household frame
  spines:                               # sources that contribute ROWS
    - { source: cps_asec, role: base_candidate, households: 55762, clone_of: null }
    - { source: cps_asec, role: support_clone,  households: 55762, clone_of: cps_asec, clone_strategy: doubled }
  total_households: 111524
  # sources NOT listed here (PUF/ACS/SCF/SIPP) are donors, below — they add no rows

donors:                                 # sources that impute VARIABLES, no rows
  - source: irs_soi_puf
    shareability: restricted            # why it can't be a spine
    method: qrf                         # qrf | regime_aware(sign-classifier) | ...
    predictors: [age, is_male, tax_unit_is_joint, tax_unit_count_dependents,
                 is_tax_unit_head, is_tax_unit_spouse, is_tax_unit_dependent]
    applies_to: [base, clone]           # which rows it touches
    stripped: [MARS==0 aggregate, AGI(recomputed), deps>3]
  - source: acs
    method: qrf
    applies_to: [base, clone]
  - source: scf
    method: qrf
  - source: sipp
    method: qrf

variables:                             # PER-VARIABLE provenance (the killer field)
  # treatment: kept | imputed_on_clone | overridden_everywhere | imputed_from_donor | derived
  employment_income:        { source: cps_asec,   treatment: kept }
  long_term_capital_gains:  { source: cps_asec,   treatment: imputed_on_clone, donor: irs_soi_puf, method: qrf }
  partnership_s_corp_income:{ source: irs_soi_puf, treatment: overridden_everywhere, method: qrf }
  net_worth:                { source: scf,         treatment: imputed_from_donor, method: qrf }
  real_estate_taxes:        { source: acs,         treatment: imputed_from_donor, method: qrf }
  adjusted_gross_income:    { source: runtime,     treatment: derived }

calibration:
  method: l0                            # l0 | projected_gd | entropy
  target_set: { name: pe_native_broad, version: 2024, count: 2818 }
  household_budget: null                # if sparse-select to N nonzero
  holdout: { fraction: 0.2, seed: 20260529 }

outputs:
  dataset: { path: ..., sha256: <hash>, n_households: 111524 }

measured:                               # filled in post-build by the validator
  total_weight: 153768768.0
  nonzero_households: pending
  effective_sample_size: pending
  pe_native_broad_loss: pending         # only if scored; else pending (never guessed)

validation:                            # the binding check
  spec_matches_output: pending          # measured composition == population spec?
  reproduced_clean: pending             # rebuilt from reproduce.command and matched sha?
```

## Worked example — the eCPS-shaped canonical build

A populated manifest for the "match eCPS" national build lives at
`rebuild/manifest/example-canonical-national.yaml`. It encodes what we actually
learned: CPS frame + a doubled CPS clone, PUF imputed via QRF (47 vars
`overridden_everywhere`, the rest `imputed_on_clone`), and ACS/SCF/SIPP donors —
with the donor wiring marked `pending` where we have **not yet confirmed** the
default config activates them (modeling rule #7).

## How it plugs into the scoreboard

The comparison harness (`mp_rebuild/compare.py`) should **require a valid
manifest for each side** and refuse to run otherwise. Then a comparison is
self-describing: "candidate `mp-national-canonical-2024` (manifest sha …,
111,524 hh, PUF+ACS+SCF+SIPP donors) vs baseline `ecps-2024` (manifest sha …,
…) at matched N, held-out." We would never again benchmark a build we can't
describe — which is exactly the mistake that produced the 0.094 headline.

## Data-weekly alignment (2026-05-29): the live-dashboard contract

The team agreed the manifest *is* the architecture — a **live** record emitted as
stages run, written to a central store, that the calibration dashboard + a query
API/CLI read. So the schema gained two blocks:

- **`calibration_diagnostics.targets[]`** — per-target aggregate outputs
  (`target_value` / `estimate` / `relative_error` / `source` / `in_loss_function` /
  `geography` / `pipeline`). This closes the **"Microplex stores no per-variable
  aggregate outputs anywhere"** gap Pavel flagged, and the columns *are* the
  calibration dashboard's columns. Emitted incrementally so a live dashboard
  renders build progress.
- **`run`** — a unique, non-trampled `run_id`; `emitted_at_stage` for live
  emission; and an `upload` policy: **default auto-upload** to a central store
  (Supabase, PolicyEngine org), **opt-out via `incognito`** for purely-experimental
  local runs — so results stop getting lost (the lesson from the messy
  early-Microplex experiments).

Ownership: this contract is what **Pavel's calibration dashboard** and the
**Pavel+Anthony query API/CLI** build against; **Anthony's stage-emit** writes the
incremental snapshots. Rename to "ledger" is deferred (keep `arch` for now); the
central store moves to PolicyEngine Supabase.

## Adoption path (cheapest first)

1. Land the schema + validator (this dir).
2. Write the manifest for **eCPS** by hand once (reverse-engineered) and validate
   it against the shipped H5 — proves the schema captures a real, messy build.
3. Make `pe_us_data_rebuild` **emit** a manifest for the canonical mp build.
4. Wire the scoreboard to require manifests.
5. Then, and only then, compare canonical-mp vs eCPS — both pinned.

This is governance, not a new engine. It's the thing Microplex promised
(provenance + legibility) and is the actual unsolved problem.
