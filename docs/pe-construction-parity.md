# PE Construction Parity

This document tracks whether `microplex-us` matches incumbent
`policyengine-us-data` behavior where that matters at the mapping /
construction / rules layer, while still treating PolicyEngine as the evaluation
oracle rather than the thing Microplex is trying to become.

It is intentionally narrower than benchmark performance:

- benchmark superiority asks whether Microplex produces better downstream data
- construction parity asks whether Microplex is matching PE's incumbent
  interface and rule contracts faithfully enough to support attribution and a
  credible replacement claim

The point is to avoid mixing these claims.

Saved-run parity evidence should now be written as a sidecar with
[`write_policyengine_us_data_rebuild_parity_artifact(...)`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/pe_us_data_rebuild_parity.py),
so one artifact bundle records:
- whether the run actually matched the default incumbent-compatibility profile
- which exact `policyengine-us-data` baseline slice it used
- what the harness and PE-native broad-loss comparisons said

The intended way to create those bundles is now
[`run_policyengine_us_data_rebuild_checkpoint(...)`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/pe_us_data_rebuild_checkpoint.py),
which runs the explicit incumbent-compatibility profile, saves a normal versioned artifact,
attaches harness / optional PE-native evidence from the saved dataset, and then
writes the parity sidecar from that saved bundle.

## Status legend

- `Exact`: same construction contract to the best of the current audit
- `Close`: same high-level rule logic, with only minor implementation
  differences
- `Compatible, not equivalent`: PE-ingestable and semantically aligned, but not
  the same construction contract
- `Different`: materially different construction logic today
- `Not yet audited`: important, but not yet checked closely enough

## Initial audited matrix

| Area | Microplex source | PE source | Status | Notes |
| --- | --- | --- | --- | --- |
| CPS Social Security split | [`cps.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/data_sources/cps.py) | [`cps.py`](/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/datasets/cps/cps.py) | `Close` | Both use `RESNSS1/2`, the same retirement/disability/survivor/dependent priority, and the same age-62 fallback for otherwise unclassified SS. |
| PE total Social Security contract | [`variables.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/variables.py), [`us.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/us.py) | [`social_security.py`](/Users/maxghenis/PolicyEngine/policyengine-us/policyengine_us/variables/gov/ssa/ss/social_security.py) | `Compatible, not equivalent` | PE treats total SS as the sum of the four component inputs. Microplex still carries `social_security_unclassified` internally, then allocates that residual into retirement at PE export points. |
| PUF Social Security split | [`puf.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/data_sources/puf.py), [`share_imputation.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/data_sources/share_imputation.py), [`pe_us_data_rebuild.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/pe_us_data_rebuild.py) | [`puf_impute.py`](/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/calibration/puf_impute.py) | `Compatible, not equivalent` | Microplex now has an explicit PE-style QRF split strategy and an incumbent-compatibility provider bundle that selects it. It is not yet a full line-by-line clone of PE-data's predictor surface and remains a configurable parity mode rather than the only path. |
| Donor-survey source-impute predictor contract | [`us.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/us.py), [`pe_us_data_rebuild.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/pe_us_data_rebuild.py), [`donor_surveys.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/data_sources/donor_surveys.py), [`pe_source_impute_specs.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pe_source_impute_specs.py), [`pe_source_impute_engine.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pe_source_impute_engine.py), [`pe_source_impute_blocks.json`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/manifests/pe_source_impute_blocks.json) | [`source_impute.py`](/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/calibration/source_impute.py) | `Compatible, not equivalent` | Microplex now has an explicit `pe_prespecified` donor-condition mode plus one shared donor-block manifest that drives donor adapters, predictor surfaces, condition-frame derivations, raw/dataset loader mappings, and a centralized PE source-impute engine for block resolution, block-frame preparation / entity projection, prepared condition surfaces, and shared donor-block execution semantics used by both the prespecified PE path and the generic fallback path. SIPP is modeled as one survey with multiple donor blocks rather than separate bespoke provider implementations. The full source-imputation stage is still not a line-by-line PE clone, especially around annualization/sampling details and how generic donor integration replaces the PE inline script. |
| Dividend atomic basis | [`variables.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/variables.py), [`us.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/us.py) | PE consumes the exported variables through `policyengine-us`; line-by-line PE-data construction parity has not yet been audited | `Compatible, not equivalent` | Microplex explicitly normalizes dividend inputs onto a qualified/non-qualified atomic basis and then derives totals. This is a cleaner contract, but not yet audited as a PE-data rule clone. |
| Interest taxable / tax-exempt split | [`puf.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/data_sources/puf.py), [`us.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/us.py) | PE consumes the exported variables through `policyengine-us`; line-by-line PE-data construction parity has not yet been audited | `Not yet audited` | Microplex has explicit taxable and tax-exempt interest handling, but the PE-data construction path for parity purposes has not yet been written up. |
| Pension taxable / tax-exempt split | [`us.py`](/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/us.py) | PE consumes the exported variables through `policyengine-us`; line-by-line PE-data construction parity has not yet been audited | `Not yet audited` | This is an important family because it affects many downstream PE formulas, but the current status is still "compatible export", not audited parity. |
| Formula layer boundary | `microplex-us` export/build path | `policyengine-us` variable formulas | `Different` | Microplex is still primarily a PE-input construction runtime. The formula layer remains in `policyengine-us`, which is intentional. |

## What this means

The current path is strong enough to support an architecture-first program, but
not strong enough to claim general PE construction parity.

The best current reading is:

1. `microplex-us` is already becoming the cleaner US build system.
2. Some high-value constructions are already close to PE, especially CPS Social
   Security.
3. Some important paths now have an explicit incumbent-parity mode, but are not
   yet full line-by-line clones, especially PUF Social Security.
4. Several family constructions and donor-source details still need a real
   audit before we can call them PE-equivalent, even though the donor-block
   contract itself is now centralized.

## Next parity targets

The next high-value parity work should be:

1. Remove or explicitly retire the Social Security residual-to-retirement shim
   at PE export points.
2. Audit dividend, interest, and pension-family construction against the PE
   build path and mark each one as `Exact`, `Close`, or `Intentionally
   different`.
3. Add parity checks where possible so the matrix is backed by code, not only
   prose.
4. Only after the US parity picture is clearer, promote any stable generic
   abstraction up into `microplex`.
