# Methodology Ledger

This document is the living methods record for `microplex-us`.

It is not the paper. It is the shortest place in the repo that should answer:

- what the current canonical pipeline is
- what PolicyEngine is doing in that pipeline
- which methodological choices are considered canonical today
- which choices are explicitly provisional or challenger-only
- where the evidence for those choices is stored

## Core framing

`microplex-us` is not trying to literally recreate `policyengine-us-data`.

Current framing:

- `policyengine-us` is the shared measurement operator
- the active PE-US targets DB is the truth surface we score against
- `policyengine-us-data` is the incumbent comparator and interface reference
- `microplex-us` is an independent US data-construction runtime

That means incumbent-compatibility work exists to improve attribution and
interface confidence, not to define the project as a wrapper around PE-US-data.

## Claim separation

We keep four claims separate:

1. Architecture claim
   - `microplex-us` is a cleaner, more modular, more auditable runtime.
2. Oracle-compatibility claim
   - where important, Microplex matches or intentionally departs from incumbent
     PE-US-data construction behavior.
3. Benchmark claim
   - Microplex produces a better PE-ingestable dataset than the incumbent on
     the active target estate.
4. Paper claim
   - a stable narrative about methodology, evidence, and novelty that can be
     defended externally.

The first three live in code and artifacts now. The fourth should be written
from them later, not invented separately.

## Methods snapshot

### Snapshot as of 2026-04-10

This is the current working methods snapshot, not a claim of finality.

| Area | Current reading | Status | Main evidence |
| --- | --- | --- | --- |
| Measurement contract | `policyengine-us` plus the active targets DB are the oracle. `policyengine-us-data` is the incumbent comparator. | `Canonical` | [benchmarking.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/benchmarking.md) |
| Runtime boundary | Microplex owns source loading, donor integration, synthesis, entity build, export, artifacts, and experiment tracking. PolicyEngine owns measurement/materialization at eval time. | `Canonical` | [architecture.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/architecture.md) |
| Incumbent-compatibility work | PE-style modes are used where they improve attribution or interface confidence, but they do not define the whole project. | `Canonical` | [policyengine-oracle-compatibility.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/policyengine-oracle-compatibility.md) |
| Construction parity claim | Some construction layers are close or compatible, but general PE-construction parity is not yet established. | `Canonical` | [pe-construction-parity.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/pe-construction-parity.md) |
| Imputation evaluation | We currently track both support realism and MAE. Neither should be collapsed into a single unqualified "best" method. | `Canonical` | [pe_us_data_rebuild_parity.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_parity.json), [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json) |
| Current production imputation reading | `structured_pe_conditioning` is the support winner on the current checkpoint ablation; `top_correlated_qrf` is the MAE winner. | `Provisional` | [pe_us_data_rebuild_parity.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_parity.json), [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json) |
| Broad mission metric | The mission metric is PE-native broad loss frontier, but pre-calibration support evidence is retained so unrealistic imputations do not hide behind later weighting. | `Canonical` | [superseding-policyengine-us-data.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/superseding-policyengine-us-data.md), [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json) |
| Full-oracle loss accounting | `full_oracle_*` metrics now score the entire active targets DB, including explicit penalty mass for unsupported rows. Supported-only diagnostics remain separate. | `Canonical` | [policyengine-oracle-compatibility.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/policyengine-oracle-compatibility.md), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_medium/medium-source-corrected-oracle-v1/manifest.json) |
| Calibration target planning | The active targets DB is one catalog, but calibration is staged and support-aware: rows are classified into `solve_now`, `solve_later`, or `audit_only` instead of forcing one flat solve. | `Canonical` | [policyengine-oracle-compatibility.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/policyengine-oracle-compatibility.md), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_medium/medium-source-corrected-oracle-v1/manifest.json) |
| Current deferred calibration policy | Default PE-oracle rebuilds use a dense first pass plus two deferred passes at support `10` and `1`, each capped to 24 constraints, always consider those passes, and narrow them to the top 7 deferred families and top 4 deferred geographies. Within that focus, deferred stages spend capacity by row-level capped error first, then family/geography loss share, and each deferred pass is only kept if it improves capped full-oracle loss. After correcting the upstream EITC-recipient oracle semantics, the support-10 pass improved the matched `2000/2000` large no-donor run from `0.9729` to `0.9498`, the matched donor-inclusive large run from `0.9730` to `0.9502`, and the medium no-donor run from `1.0298` to `1.0291`. With the row-aware selector in place, the support-1 pass further improves the broader donor-inclusive run from `0.8783` to `0.8213`, the matched broader no-donor run from `0.8908` to `0.8362`, and the medium no-donor run from `1.0291` to `1.0029`. Widening deferred family focus from 3 to 4 then improves the broader donor-inclusive run again from `0.8213` to `0.7909`, the matched broader no-donor run from `0.8362` to `0.7996`, and the medium no-donor run from `1.0029` to `0.9969`. A fresh broader donor-inclusive checkpoint through the unmodified default entrypoint reproduces that `0.7909` result exactly. Widening deferred geographies from 4 to 8 on the same broader donor run then regresses capped full-oracle loss from `0.7909` to `0.7992`, so the geography focus should stay at `4`. Fixing raw PUF checkpoint sampling to respect `S006` weights then improves the broader donor-inclusive default again from `0.7909` to `0.7682` and the matched broader no-donor default from `0.7996` to `0.7683` without any calibration-policy change. After promoting the earnsplit-only PUF person-expansion default, widening deferred family focus from 4 to 7 improves the broader donor-inclusive run again from `0.7176` to `0.7045`, and the matched donor-free broader run from `0.7171` to `0.7040`, with the same focused family set including `aca_ptc` and `rental_income`. | `Provisional` | [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_eitc_recipient_oracle_large_nodonors/large-nodonors-eitc-recipient-oracle-v2/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_age_agi_forced_stage2_large_nodonors/large-nodonors-age-agi-forced-stage2-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_age_agi_forced_stage2_large_donors/large-donors-age-agi-forced-stage2-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_age_agi_forced_stage2_medium_nodonors/medium-nodonors-age-agi-forced-stage2-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_default_stage2_large_donors/large-donors-default-stage2-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_rowrank_donors/broader-donors-rowrank-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_rowrank_nodonors/broader-nodonors-rowrank-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_donors/broader-donors-stage3-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_nodonors/broader-nodonors-stage3-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_stage3_nodonors/medium-nodonors-stage3-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_default_stage3_nodonors/medium-nodonors-default-stage3-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_top4family_donors/broader-donors-stage3-top4family-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_top4family_nodonors/broader-nodonors-stage3-top4family-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_stage3_top4family_nodonors/medium-nodonors-stage3-top4family-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_default_top4family_nodonors/medium-nodonors-default-top4family-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_default_top4family_donors_rerun/broader-donors-default-top4family-v2/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_geo8_donors/broader-donors-geo8-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_weight_donors/broader-donors-puf-weight-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_weight_nodonors/broader-nodonors-puf-weight-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_default_nodonors/broader-nodonors-puf-personexpansion-default-v2/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_nodonors/broader-nodonors-puf-personexpansion-family7-v1/manifest.json) |
| Current checkpoint PUF sampling reading | Checkpoint-scale PUF sampling should respect raw `S006` weights before variable mapping rather than uniformly sampling raw PUF records. This is incumbent-alignment work, not a challenger method: it changes the checkpoint source sample so it better reflects the PUF weighting surface before any Microplex-specific synthesis or calibration logic. | `Provisional` | [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_default_top4family_donors_rerun/broader-donors-default-top4family-v2/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_weight_donors/broader-donors-puf-weight-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_top4family_nodonors/broader-nodonors-stage3-top4family-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_weight_nodonors/broader-nodonors-puf-weight-v1/manifest.json) |
| Current checkpoint CPS age-support sampling reading | Checkpoint-scale CPS sampling should guarantee at least one sampled household per observed `state x 5-year age-band` cell. This is also checkpoint-only incumbent-compatibility work: it does not change the full-data runtime, only the sampled source surface used in checkpoint experiments. On the matched broader donor run it improves capped full-oracle loss from `0.7682` to `0.7329`, and on the matched broader no-donor run from `0.7683` to `0.7368`. | `Provisional` | [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_weight_donors/broader-donors-puf-weight-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_donors/broader-donors-cps-stateage1-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_weight_nodonors/broader-nodonors-puf-weight-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_nodonors/broader-nodonors-cps-stateage1-v1/manifest.json) |
| Current checkpoint donor age-support sampling reading | On donor-inclusive checkpoints, donor survey sampling should also guarantee at least one sampled household per observed `state x 5-year age-band` cell when a donor source exposes both state and age. This stays in the same checkpoint-only incumbent-compatibility bucket as the CPS age floor, but the effect is much smaller: on the matched broader donor run it improves capped full-oracle loss from `0.7329149849` to `0.7327632809` with the same selected-constraint count. | `Provisional` | [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_donors/broader-donors-cps-stateage1-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_donor_stateage1_donors/broader-donors-donor-stateage1-v1/manifest.json) |
| Current checkpoint CPS income-support sampling reading | Do not promote checkpoint CPS income-support floors yet. The household-income analogue clearly regressed the matched broader donor run from `0.7329` to `0.7554`, and the more PE-aligned tax-unit-income analogue was a near miss but still regressed the frontier metric from `0.7329` to `0.7372` even while improving uncapped full-oracle and active-solve loss. The accepted upstream checkpoint support change therefore remains the CPS `state x age-band` floor only. | `Provisional` | [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_donors/broader-donors-cps-stateage1-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_income_donors/broader-donors-cps-stateage1-income-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_taxunitincome_donors/broader-donors-cps-stateage1-taxunitincome-v1/manifest.json) |
| Current PUF person-expansion reading | Keep PE-style `EARNSPLIT` randomization in the PUF PE-demographics branch, but do not promote PE-style age-bin and spouse/dependent-sex randomization into the default path yet. The winning split-only version improves the matched broader donor checkpoint from `0.7327632809` to `0.7176041064`, while the age/sex-only version regresses it to `0.7463902007`. A later retest of the full age/sex path on top of the stronger family-7 broader donor default still regresses the mission metric from `0.7044626415` to `0.7111876263`, so this remains a rejected lane rather than an unresolved default question. This keeps the upstream income-splitting alignment that helps the frontier metric without forcing the age/sex piece that currently hurts checkpoint performance. | `Provisional` | [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_ageonly_donors/broader-donors-puf-personexpansion-ageonly-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_earnsplitonly_donors/broader-donors-puf-personexpansion-earnsplitonly-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_default_donors/broader-donors-puf-personexpansion-default-v2/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_rng_donors/broader-donors-puf-personexpansion-rng-v1/manifest.json), [tmp_puf_source_stage_parity_personexpansion_20260412.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_puf_source_stage_parity_personexpansion_20260412.json) |
| Current post-fix residual reading | After the raw PUF weighting fix, the checkpoint CPS `state x age-band` floor, the earnsplit-only PUF person-expansion default, and the wider deferred family gate, ACA PTC and rental mass drop sharply, but the remaining capped-error mass is now led again by age, person AGI, tax-unit AGI, and EITC child-count families. The worst individual rows are still dominated by ACA amount and ACA-eligibility cells, with a thinner stored-input tail now mostly in tax-exempt interest and a few rental states. That keeps the next upstream lane on age/AGI/EITC structure rather than another broad calibration-policy sweep. | `Provisional` | [tmp_broader_puf_personexpansion_family7_donor_drilldown_20260412.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_broader_puf_personexpansion_family7_donor_drilldown_20260412.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_nodonors/broader-nodonors-puf-personexpansion-family7-v1/manifest.json) |
| Current stored-input tail reading | Keep the accepted interest/rental donor-conditioning change, reject the property-cost extension, and reject both export-side rental normalization and direct zero-support-mask propagation in zero-inflated donor rank matching. Each looked locally plausible, but fresh `2000/2000` large no-donor source checkpoints regressed capped full-oracle loss from `1.3274` to `1.3874` and `1.9223` respectively, so the default path stays conservative here. | `Provisional` | [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_current/smoke-nodonors-asset-tail-conditioning-current-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_oldsemantics/smoke-nodonors-asset-tail-old-semantics-v1/manifest.json), [tmp_policyengine_oracle_target_drilldown_asset_tail_smoke_current_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_policyengine_oracle_target_drilldown_asset_tail_smoke_current_20260411.json), [tmp_policyengine_oracle_target_drilldown_asset_tail_smoke_old_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_policyengine_oracle_target_drilldown_asset_tail_smoke_old_20260411.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_rental_export_large_nodonors/large-nodonors-rental-export-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_zero_support_mask_large_nodonors/large-nodonors-zero-support-mask-v1/manifest.json) |
| Current interest-family reading | Do not promote the `interest_income + tax_exempt_interest_share` decomposition into the default path yet. It looked strong on the `400/400` medium no-donor run, but the matched `2000/2000` no-donor confirmation regressed capped full-oracle loss from `1.3274` to `1.3555`, so the default remains separate `taxable_interest_income` and `tax_exempt_interest_income` lanes. | `Provisional` | [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_interest_family_medium_nodonors/medium-nodonors-interest-family-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_interest_family_large_nodonors/large-nodonors-interest-family-v1/manifest.json) |
| Current donor-support sampling reading | Keep donor-support sampling with replacement. Forcing no-replacement support sampling looked cleaner mechanically but made the matched smoke run materially worse on both capped full-oracle and active-solve loss. | `Provisional` | [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_current/smoke-nodonors-asset-tail-conditioning-current-v1/manifest.json), [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_donor_support_sampling_smoke_nodonors/smoke-nodonors-donor-support-sampling-v1/manifest.json) |
| Current benchmark reading | On the current checkpoint artifact, harness metrics improved versus the incumbent comparator, but native broad loss is still much worse than `enhanced_cps_2024`. | `Canonical` | [pe_us_data_rebuild_parity.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_parity.json), [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json) |
| Current cross-run regression reading | Across 66 scored modelpass checkpoint runs, `national_irs_other` appears in the top 3 every time, `state_agi_distribution` in 63/66, and `state_aca_spending` in 54/66. Near-term model work should target those recurring families directly rather than broad tuning. | `Provisional` | [live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json) |
| Current `national_irs_other` drilldown reading | The audited `national_irs_other` failures are concentrated in filing-status-sensitive IRS cells and coincide with large `SINGLE` and `JOINT` overcounts plus `SEPARATE` undercounts. The first remediation step is to preserve source-authoritative filing-status inputs into the PE construction path. | `Provisional` | [live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json) |

## Canonical pipeline

The current broad US pipeline is:

1. Load raw survey/tax sources into canonical observation frames.
2. Apply source semantics and variable semantics.
3. Build donor blocks and donor-condition surfaces.
4. Impute donor-only variables into the scaffold population.
5. Synthesize a candidate population.
6. Build PolicyEngine-ingestable entity tables.
7. Export final H5.
8. Run PolicyEngine materialization and compare implied aggregates to the active
   target DB.
9. Save artifact bundles, sidecars, and registry/index records.

This is a fresh Microplex pipeline with a PolicyEngine evaluation boundary, not
an attempt to make PE-US-data the runtime architecture.

## What is currently canonical

- Source and variable semantics are declared in Microplex-owned registries and
  manifests.
- Final evaluation uses the shared PE-US runtime and active targets DB.
- Artifact discipline is required for serious runs:
  - `manifest.json`
  - `data_flow_snapshot.json`
  - `policyengine_harness.json` when harness evaluation runs
  - `policyengine_native_scores.json` when PE-native broad loss runs
  - `pe_us_data_rebuild_parity.json` for incumbent-compatibility checkpoints
  - `pe_us_data_rebuild_native_audit.json` for target/family/support audit
  - `run_registry.jsonl`
  - `run_index.duckdb`
- Incumbent-compatibility modes are allowed when they improve attribution.
- Materially different model choices should be explicit challenger variants.

## What is still provisional

- The default imputation stack is still under active evaluation.
- Support realism vs MAE tradeoffs are still live methodological questions.
- Full-support candidate construction and selector design are not settled.
- Calibration is still operationally important, but it is not the only or even
  always the dominant methodological lever.
- Held-out evaluation is not yet the default outer loop.

These should not be written up later as if they were settled all along.

## Current open questions

- Should runtime imputation selection prioritize support realism, weighted MAE,
  or a gated combination of the two?
- How much conditioning structure should be imposed before flexible donor/QRF
  prediction begins?
- How much of the remaining broad-loss gap is record construction versus
  selection/calibration?
- Should deferred calibration eligibility stay at a single scalar trigger
  (`full_oracle_capped_mean_abs_relative_error > 2.45`), or should it become
  family-aware once larger source runs accumulate?
- Which incumbent-compatible modes are worth keeping as long-run options, and
  which should remain diagnostic-only?
- When should held-out evaluation become a required gate rather than an optional
  extra?

## Current methodological evidence surfaces

Use these surfaces when writing claims down later:

- [benchmarking.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/benchmarking.md)
  for the truth/comparator/operator contract
- [policyengine-oracle-compatibility.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/policyengine-oracle-compatibility.md)
  for incumbent-compatibility rules
- [pe-construction-parity.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/pe-construction-parity.md)
  for audited construction-layer matching vs intentional difference
- saved artifact bundles for actual run-level evidence
- tests for the code-enforced contract behind those claims

For the current checkpoint-style evidence bundle, the most useful files are:

- [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/manifest.json)
- [data_flow_snapshot.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/data_flow_snapshot.json)
- [policyengine_harness.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/policyengine_harness.json)
- [policyengine_native_scores.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/policyengine_native_scores.json)
- [pe_us_data_rebuild_parity.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_parity.json)
- [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json)
- [imputation_ablation.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/imputation_ablation.json)
- [live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json)
- [live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json)

## Decision log

### 2026-04-10: Project framing

- Decision:
  - describe `policyengine-us` as the oracle/evaluator and
    `policyengine-us-data` as the incumbent comparator
- Why:
  - this matches how the system is actually being used
  - it avoids understating the novelty of the Microplex runtime
  - it keeps incumbent-compatibility work from swallowing the whole project
- Evidence:
  - [benchmarking.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/benchmarking.md)
  - [policyengine-oracle-compatibility.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/policyengine-oracle-compatibility.md)

### 2026-04-10: Imputation evaluation contract

- Decision:
  - keep support realism and MAE as separate evidence channels
  - do not summarize imputation quality using post-calibration loss alone
- Why:
  - the current checkpoint artifact shows a real tradeoff
  - `structured_pe_conditioning` wins support
  - `top_correlated_qrf` wins weighted MAE
  - collapsing the two too early would hide methodology risk
- Evidence:
  - [pe_us_data_rebuild_parity.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_parity.json)
  - [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json)
  - [imputation_ablation.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/imputation_ablation.json)

### 2026-04-10: Artifact contract for headline claims

- Decision:
  - treat sidecars and registry metadata as part of the methodology, not just
    engineering exhaust
- Why:
  - paper-facing claims will need reproducible evidence with exact configs,
    metrics, and comparison slices
  - the artifact bundle is now the canonical storage layer for that evidence
- Evidence:
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/manifest.json)
  - [data_flow_snapshot.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/data_flow_snapshot.json)

### 2026-04-10: Cross-run regression priority

- Decision:
  - prioritize targeted fixes for `national_irs_other`,
    `state_agi_distribution`, and then `state_aca_spending`
- Why:
  - across recent modelpass checkpoint families, the same regressions recur even
    when total loss improves substantially
  - `national_irs_other` appears in the top 3 for all 66 scored runs
  - `state_agi_distribution` appears in the top 3 for 63/66 runs and is the
    largest regressing family in 34 runs
  - `state_aca_spending` appears in the top 3 for 54/66 runs but is more often
    a secondary or tertiary regression
- Evidence:
  - [live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json)

### 2026-04-10: First `national_irs_other` remediation target

- Decision:
  - first fix the preservation of source-authoritative filing-status inputs in
    the PE-oracle rebuild path before attempting more downstream status tuning
- Why:
  - audited `national_irs_other` lead runs show repeated IRS target failures in
    filing-status-sensitive cells, especially `Single`, `Joint`, and high-AGI
    bins
  - those same audited runs show large `SINGLE` and `JOINT` count surpluses,
    large `SEPARATE` deficits, and missing or distorted MFS support bins
  - the saved candidate seed/synthetic/calibrated rows for leading runs retain
    `marital_status` but not `filing_status_code`, so the authoritative PUF tax
    filing code is disappearing before tax-unit construction
- Evidence:
  - [live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json)

### 2026-04-11: Full-oracle accounting means the full DB

- Decision:
  - score `full_oracle_*` metrics over the full active targets DB, not just the
    supported subset
  - penalize unsupported rows explicitly rather than letting them disappear from
    the scalar objective
  - keep supported-only summaries as separate diagnostics
- Why:
  - "measure everything, optimize the feasible subset" only works if the
    measurement metric actually reflects unsupported misses
  - otherwise frontier selection and deferred-stage triggers can be gamed by
    leaving hard rows unsupported
- Evidence:
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_medium/medium-source-corrected-oracle-v1/manifest.json)
  - [policyengine-oracle-compatibility.md](/Users/maxghenis/PolicyEngine/microplex-us/docs/policyengine-oracle-compatibility.md)

### 2026-04-11: Full DB measurement, staged calibration execution

- Decision:
  - keep the full active targets DB as one measurement catalog
  - classify rows into `solve_now`, `solve_later`, or `audit_only`
  - use a dense first pass plus at most one deferred pass by default on the
    incumbent-compatible PE-oracle rebuild path
- Why:
  - one flat broad solve is not numerically credible on thinner artifacts
  - the right execution rule is support-aware staging, not shadow target CSVs
    or pretending all DB rows belong in the same solve
- Evidence:
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_medium/medium-source-corrected-oracle-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_donors/donors-source-corrected-oracle-v1/manifest.json)

### 2026-04-11: Current deferred-stage default

- Decision:
  - default deferred calibration on the incumbent-compatible PE-oracle rebuild
    path uses:
    - one deferred pass at support floor `10`
    - deferred-pass cap `24`
    - trigger threshold `full_oracle_capped_mean_abs_relative_error > 2.45`
- Why:
  - tiny-source evidence still benefits from the deferred pass
  - medium, donor-inclusive, and larger replayed/source artifacts do not justify
    attempting it below that threshold
- Evidence:
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_medium/medium-source-corrected-oracle-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_donors/donors-source-corrected-oracle-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_large_donors/large-donors-source-corrected-oracle-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_large_nodonors/large-nodonors-source-corrected-oracle-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1/manifest.json)
  - [tmp_corrected_oracle_large_replay_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_corrected_oracle_large_replay_20260411.json)
  - [tmp_corrected_oracle_xlarge_replay_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_corrected_oracle_xlarge_replay_20260411.json)

### 2026-04-11: Support person-to-tax-unit count targets in the PE compiler

- Decision:
  - support `person -> tax_unit/family/spm_unit` boolean target filters in the
    PE household-constraint compiler using group-membership `.any()` semantics
- Why:
  - broad-oracle runs were carrying an artificial unsupported wall across 11
    whole `tax_unit_count` families such as `dividend_income`,
    `taxable_interest_income`, and `unemployment_compensation`
  - those targets are defined as `tax_unit_count` with person-entity domain
    filters like `dividend_income > 0` plus tax-unit filters like
    `tax_unit_is_filer == 1`
  - removing that structural limitation dropped unsupported targets on the
    large no-donor replay from `572` to `0`, and the fresh source rerun improved
    capped full-oracle loss from `2.4329` to `1.3274`
- Evidence:
  - [tmp_large_source_cross_entity_fix_replay_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_large_source_cross_entity_fix_replay_20260411.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1/manifest.json)

### 2026-04-11: Residual oracle work should target age, EITC-child-count, AGI, and OR/GA/MO

- Decision:
  - prioritize post-fix model and construction work against the remaining large-run
    oracle leaders rather than more deferred-stage tuning
- Why:
  - fresh donor and no-donor `2000/2000` source runs now share the same top
    full-oracle residual families and geographies
  - the largest remaining families are age counts, `tax_unit_count` for
    `eitc_child_count`, and AGI count families; the leading geographies are
    `state:OR`, `state:GA`, and `state:MO`
  - within those geographies, the worst cells are concentrated in ACA PTC,
    AGI counts, SALT, rental income, tax-exempt interest income, and
    pass-through income
- Evidence:
  - [tmp_policyengine_oracle_regressions_cross_entity_fix_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_policyengine_oracle_regressions_cross_entity_fix_20260411.json)
  - [tmp_policyengine_oracle_target_drilldown_cross_entity_fix_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_policyengine_oracle_target_drilldown_cross_entity_fix_20260411.json)

### 2026-04-11: Keep the interest/rental conditioning change, reject the property-cost extension

- Decision:
  - keep the richer interest/rental donor-conditioning semantics
  - do not promote the property-cost semantic extension into the default pipeline
- Why:
  - on matched `200/200` smoke checkpoints, the accepted interest/rental change
    slightly improves capped full-oracle loss from `1.4417803` to `1.4414441`
    and lowers active-solve capped loss from `1.8878380` to `1.8829362`
  - the accepted change cuts the capped stored-input mass attributed to
    `tax_exempt_interest_income` in the top drilldown from `40` to `20`
  - the follow-on property-cost extension made capped full-oracle loss worse
    (`1.4489770`) and doubled property-side capped mass in the top drilldown,
    so it was reverted
- Evidence:
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_current/smoke-nodonors-asset-tail-conditioning-current-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_oldsemantics/smoke-nodonors-asset-tail-old-semantics-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_v2/smoke-nodonors-asset-tail-conditioning-v2/manifest.json)
  - [tmp_policyengine_oracle_target_drilldown_asset_tail_smoke_current_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_policyengine_oracle_target_drilldown_asset_tail_smoke_current_20260411.json)
  - [tmp_policyengine_oracle_target_drilldown_asset_tail_smoke_old_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_policyengine_oracle_target_drilldown_asset_tail_smoke_old_20260411.json)

### 2026-04-11: Reject rental export normalization from donor-integrated components

- Decision:
  - do not rebuild net `rental_income` at PolicyEngine export from
    `rental_income_positive - rental_income_negative`
  - keep exporting the observed net `rental_income` directly in the default path
- Why:
  - a saved-seed replay looked promising and improved capped full-oracle loss
    from `1.3274` to `1.3169`, which made the export-side normalization look
    like a clean way to use donor-integrated rental components
  - the fresh `2000/2000` large no-donor source checkpoint contradicted that
    replay: capped full-oracle loss worsened from `1.3274` to `1.3874`
  - active-solve capped loss also worsened from `2.6923` to `2.7722`, and the
    number of active constraints fell from `540` to `522`
  - source checkpoints decide default-path changes; replay-only wins are not
    sufficient
- Evidence:
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_rental_export_large_nodonors/large-nodonors-rental-export-v1/manifest.json)

### 2026-04-11: Reject direct zero-support-mask propagation in zero-inflated donor rank matching

- Decision:
  - do not make zero-inflated donor rank matching honor the generated support mask
    directly by replacing the donor positive-rate count with `scores > 0`
  - keep the existing donor-rate-based positive count in the default path
- Why:
  - the idea was structurally coherent: the QRF path already trains a zero model,
    so propagating its zero mask through final donor assignment looked like a way
    to stop rank matching from reintroducing positive tail support
  - the fresh `2000/2000` large no-donor source checkpoint failed badly:
    capped full-oracle loss worsened from `1.3274` to `1.9223`
  - active-solve capped loss worsened from `2.6923` to `4.3296`, and active
    constraints rose from `540` to `703`, so the change was not merely trading
    one metric for another
  - again, source checkpoints decide default-path changes
- Evidence:
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_zero_support_mask_large_nodonors/large-nodonors-zero-support-mask-v1/manifest.json)

### 2026-04-11: Reject interest-family decomposition as the default path

- Decision:
  - do not promote the `interest_income + tax_exempt_interest_share` donor
    block into the default pipeline
  - keep `taxable_interest_income` and `tax_exempt_interest_income` on separate
    donor lanes for now
- Why:
  - the medium no-donor checkpoint was promising: capped full-oracle loss fell
    from `2.3931` to `1.3644`
  - the matched large no-donor confirmation did not hold: capped full-oracle
    loss worsened from `1.3274` to `1.3555`
  - raw full-oracle loss also worsened sharply on the large run, from `2256.6`
    to `16980.7`, and active-solve capped loss worsened from `2.6923` to
    `2.8229`
  - the default path should follow the larger, more representative no-donor run,
    not the thinner medium win
- Evidence:
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_interest_family_medium_nodonors/medium-nodonors-interest-family-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_medium/medium-source-corrected-oracle-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_interest_family_large_nodonors/large-nodonors-interest-family-v1/manifest.json)

### 2026-04-11: Reject donor-support sampling without replacement

- Decision:
  - keep donor-support sampling with replacement in the default donor path
- Why:
  - a no-replacement support sampler sounds cleaner, but the matched smoke run
    was worse on the only metrics that matter here
  - capped full-oracle loss worsened from `1.4414` to `1.6369`
  - active-solve capped loss worsened from `1.8829` to `2.7402`
  - this should remain a rejected experiment unless a stronger construction
    change makes it worthwhile to revisit
- Evidence:
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_current/smoke-nodonors-asset-tail-conditioning-current-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_donor_support_sampling_smoke_nodonors/smoke-nodonors-donor-support-sampling-v1/manifest.json)

### 2026-04-11: Correct upstream EITC-recipient child-count target semantics

- Decision:
  - treat IRS SOI EITC child-count targets as recipient strata that require
    `eitc > 0`, not just filer strata split by `eitc_child_count`
  - keep Microplex compatible with the corrected DB by treating
    `domain_variable` as a set-membership field when target rows carry multiple
    domain constraints such as `eitc,eitc_child_count`
- Why:
  - the active targets DB guide already described `eitc_child_count` as EITC
    recipient strata, and `policyengine-us-data`'s own loss code evaluates
    those cells as `(eitc > 0) * meets_child_criteria`
  - the ETL was the inconsistent layer: it created child-count strata under
    filer strata without the positive-EITC condition
  - after correcting the DB and rerunning the matched `2000/2000` large
    no-donor source checkpoint, capped full-oracle loss fell from `1.0149`
    to `0.9718` on an apples-to-apples corrected-oracle comparison
  - the same comparison moved `tax_unit_count|domain=eitc_child_count` out of
    the top-3 residual families, so this was a real oracle bug, not just a
    cosmetic target renaming
- Evidence:
  - [tmp_eitc_recipient_oracle_large_nodonors_comparison_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_eitc_recipient_oracle_large_nodonors_comparison_20260411.json)
  - [tmp_eitc_recipient_oracle_regression_summary_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_eitc_recipient_oracle_regression_summary_20260411.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_eitc_recipient_oracle_large_nodonors/large-nodonors-eitc-recipient-oracle-v2/manifest.json)

### 2026-04-11: Default to a narrow always-considered deferred stage

- Decision:
  - default PE-oracle rebuilds should always consider one deferred support-10
    calibration pass
  - keep that pass narrow by default: top 3 deferred families, top 4 deferred
    geographies, and at most 24 constraints
  - let the existing capped full-oracle accept/reject rule decide whether the
    stage is retained, instead of gating the attempt behind a hard trigger
- Why:
  - after the EITC-recipient oracle fix, the old `2.45` trigger became the
    brittle heuristic rather than the principled part of the policy
  - on matched `2000/2000` large no-donor and donor-inclusive source runs, the
    same narrow stage-2 pass improved capped full-oracle loss from `0.9729` to
    `0.9498` and from `0.9730` to `0.9502`
  - the same narrow pass also improved the medium no-donor run slightly, from
    `1.0298` to `1.0291`, so the accept/reject rule is carrying the right
    burden and the hard trigger is not buying us much
- Evidence:
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_age_agi_forced_stage2_large_nodonors/large-nodonors-age-agi-forced-stage2-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_age_agi_forced_stage2_large_donors/large-donors-age-agi-forced-stage2-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_age_agi_forced_stage2_medium_nodonors/medium-nodonors-age-agi-forced-stage2-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_default_stage2_large_donors/large-donors-default-stage2-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_default_stage2_donors/broader-donors-default-stage2-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_default_stage2_nodonors/broader-nodonors-default-stage2-v1/manifest.json)

### 2026-04-11: Do not widen deferred stage family focus just to include ACA

- Decision:
  - keep the default deferred-stage family focus at 3 rather than widening it
    to 4 just to admit `aca_ptc|domain=aca_ptc`
- Why:
  - the broader no-donor row-level drilldown made ACA look like the next
    plausible family to admit into stage 2, but the matched `5000/5000`
    checkpoint with `top_family_count = 4` produced the exact same final result
    as `top_family_count = 3`
  - capped full-oracle loss stayed at `0.8908588019931089`
  - active-solve capped loss stayed at `0.8950141021216582`
  - the stage-2 cap remained `24`, so widening the family focus did not
    meaningfully change which cells won capacity
- Evidence:
  - [tmp_broader_nodonor_oracle_drilldown_20260411.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_broader_nodonor_oracle_drilldown_20260411.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_default_stage2_nodonors/broader-nodonors-default-stage2-v1/manifest.json)
  - [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_nodonor_top4family/broader-nodonors-top4family-v1/manifest.json)

### 2026-04-11: Prioritize deferred stage-2 rows by row-level loss inside the focused cap

- Decision:
  - keep the row-aware deferred selector
  - within the existing top-3-family / top-4-geography focus and 24-constraint
    cap, rank candidate stage-2 rows by capped target error plus family and
    geography loss share rather than family/geography share alone
- Why:
  - widening the focused family set did nothing because the bottleneck is the
    24-slot cap, not admission into the focused set
  - the row-aware ranking is neutral on the medium no-donor checkpoint, slightly
    better on the broader no-donor checkpoint, and materially better on the
    broader donor-inclusive checkpoint
  - that is the right direction for the actual objective, capped full-oracle
    loss, without changing the surrounding stage-2 policy
- Evidence:
  - matched medium no-donor row-aware rerun:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_rowrank_nodonors/medium-nodonors-rowrank-v1/manifest.json)
    - unchanged from the prior medium default, `1.0298017982 -> 1.0291445335`
  - matched broader no-donor row-aware rerun:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_rowrank_nodonors/broader-nodonors-rowrank-v1/manifest.json)
    - improves capped full-oracle loss from `0.8908588020` to
      `0.8907527501`
  - matched broader donor-inclusive row-aware rerun:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_rowrank_donors/broader-donors-rowrank-v1/manifest.json)
    - improves capped full-oracle loss from `0.8932869027` to
      `0.8782556650`

### 2026-04-11: Default to an extra ultra-thin deferred stage after the support-10 pass

- Decision:
  - change the canonical PE-oracle rebuild default from one deferred support-10
    pass to two deferred passes at support `10` and `1`
  - keep the same `24`-constraint cap and top-3-family / top-4-geography focus
    on each deferred pass
- Why:
  - the support-1 stage is now solving the right residual class: mostly
    ultra-thin age and AGI rows that remain after the row-aware support-10 pass
  - it improves the actual objective, capped full-oracle loss, on broader
    donor-inclusive, broader no-donor, and medium no-donor reruns
  - the existing accept/reject rule already prevents the stage from sticking if
    it ever becomes harmful on another run
- Evidence:
  - matched broader donor-inclusive rerun with an extra support-1 stage:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_donors/broader-donors-stage3-v1/manifest.json)
    - improves capped full-oracle loss from `0.8782556650` to
      `0.8212707783`
  - matched broader no-donor rerun with the same extra support-1 stage:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_nodonors/broader-nodonors-stage3-v1/manifest.json)
    - improves capped full-oracle loss from `0.8907527501` to
      `0.8362042462`
  - matched medium no-donor rerun with the same extra support-1 stage:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_stage3_nodonors/medium-nodonors-stage3-v1/manifest.json)
    - improves capped full-oracle loss from `1.0291445335` to
      `1.0028694956`
  - fresh medium no-donor checkpoint through the default entrypoint:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_default_stage3_nodonors/medium-nodonors-default-stage3-v1/manifest.json)
    - reproduces the same three-stage result exactly, confirming the default
      schedule is now `(10, 1)` in the real entrypoint path

### 2026-04-11: Default deferred family focus should be 4, not 3

- Decision:
  - change the canonical PE-oracle rebuild default from top-3 deferred
    families to top-4 deferred families, keeping the same top-4 geographies and
    24-constraint cap
- Why:
  - after the row-aware selector and the extra support-1 stage, ACA PTC becomes
    the fourth largest deferred family by capped loss mass and still has many
    cells with support in the teens
  - that means it is being excluded by family admission, not by impossible
    support, and letting it into the focused set materially improves the
    full-oracle objective
- Evidence:
  - matched broader donor-inclusive rerun with top-4 deferred families:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_top4family_donors/broader-donors-stage3-top4family-v1/manifest.json)
    - improves capped full-oracle loss from `0.8212707783` to
      `0.7908917500`
  - matched broader no-donor rerun with top-4 deferred families:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_top4family_nodonors/broader-nodonors-stage3-top4family-v1/manifest.json)
    - improves capped full-oracle loss from `0.8362042462` to
      `0.7995775732`
  - matched medium no-donor rerun with top-4 deferred families:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_stage3_top4family_nodonors/medium-nodonors-stage3-top4family-v1/manifest.json)
    - improves capped full-oracle loss from `1.0028694956` to
      `0.9968822972`
  - fresh medium no-donor checkpoint through the default entrypoint:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_default_top4family_nodonors/medium-nodonors-default-top4family-v1/manifest.json)
    - reproduces the same top-4-family result exactly, confirming the default
      family focus is now `4` in the real entrypoint path

### 2026-04-12: Keep deferred geography focus at 4

- Decision:
  - keep the canonical PE-oracle rebuild default at top-4 deferred geographies
    rather than widening the geography focus further
- Why:
  - the fresh broader donor-inclusive default-entrypoint rerun reproduces the
    existing top-4-family/top-4-geography result exactly, so the default path is
    already stable on the current broader donor benchmark
  - the fresh residual drilldown does show age and AGI pressure spread across
    several states, but widening geography focus to `8` on the same matched
    broader donor run worsens the real objective instead of helping
- Evidence:
  - fresh broader donor-inclusive checkpoint through the unmodified default
    entrypoint:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_default_top4family_donors_rerun/broader-donors-default-top4family-v2/manifest.json)
    - reproduces capped full-oracle loss `0.7908917500` with the default
      top-4-family/top-4-geography policy
  - matched broader donor-inclusive rerun with top-8 deferred geographies:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_geo8_donors/broader-donors-geo8-v1/manifest.json)
    - regresses capped full-oracle loss from `0.7908917500` to
      `0.7991939177`
  - fresh broader donor default drilldown:
    [tmp_broader_default_top4family_donor_drilldown_20260412.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_broader_default_top4family_donor_drilldown_20260412.json)
    - confirms the remaining capped-error mass is still led by age, AGI, ACA,
      and EITC families, so the next work should move upstream rather than
      continuing to widen deferred geography focus

### 2026-04-12: Reject PE-style CPS tax-leaf splits at both tested boundaries

- Decision:
  - reject both tested versions of the CPS AGI-alignment hypothesis:
    - do not materialize PE-style interest/dividend/pension leaf inputs inside
      the CPS source provider for the mixed-source rebuild path
    - do not apply the same split inside the default PolicyEngine export
      builder either
- Why:
  - `policyengine-us-data` does use fixed CPS split assumptions for those leaf
    inputs, but Microplex is not a single-source CPS build; it is a mixed-source
    fusion path where early promotion of estimated tax leafs can distort donor
    integration and downstream calibration
  - the source-side version confirmed that concern directly by creating a large
    new tax-exempt-interest residual family on the broader donor benchmark
  - moving the split later to the export boundary avoids the catastrophic source
    distortion, but it still does not beat the incumbent default on the
    frontier metric
- Evidence:
  - matched broader donor incumbent baseline:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_donors/broader-donors-cps-stateage1-v1/manifest.json)
    - capped full-oracle loss `0.7329149849`
  - source-side CPS leaf-input candidate:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_cps_pe_agi_donors/broader-donors-cps-pe-agi-v1/manifest.json)
    - regresses capped full-oracle loss to `0.9164981002`
    - introduces large new interest-family residuals, especially
      `tax_unit_count|domain=tax_exempt_interest_income`
  - export-side candidate:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_pe_export_cps_agi_donors/broader-donors-pe-export-cps-agi-v1/manifest.json)
    - improves on the source-side candidate but still regresses capped
      full-oracle loss to `0.7998451134`
- Read:
  - the direct PE CPS split assumptions are not plug-compatible with the
    current Microplex broader rebuild path
  - this lane should be treated as explored and rejected for the current
    frontier objective, not as an untested TODO
  - next upstream AGI work should look for better alignment boundaries than
    copying PE CPS tax-leaf splits wholesale

### 2026-04-12: Keep donor checkpoint `state x age-band` support floor

- Decision:
  - keep the donor-side analogue of the accepted CPS checkpoint `state x age-band`
    floor in the default sampled-query path for donor-inclusive checkpoints
- Why:
  - the current checkpoint asymmetry was real: CPS sampling guaranteed
    `state x 5-year age-band` coverage, while donor survey sampling still only
    applied a plain state floor
  - donor survey providers already carry household state and person age for the
    sources where this matters, so the cleanest test was to mirror the CPS
    checkpoint floor there and keep it only if the full-oracle metric moved
  - the improvement is small, but the run is deterministic and the code surface
    is narrow, so this is still worth keeping as a low-risk checkpoint-default
    refinement
- Evidence:
  - matched broader donor baseline with the accepted CPS age floor only:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_donors/broader-donors-cps-stateage1-v1/manifest.json)
    - capped full-oracle loss `0.7329149849`
    - active-solve capped loss `0.8498782563`
    - selected constraints `1059`
  - matched broader donor rerun with donor-side `state x age-band` floor:
    [manifest.json](/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_donor_stateage1_donors/broader-donors-donor-stateage1-v1/manifest.json)
    - capped full-oracle loss `0.7327632809`
    - active-solve capped loss `0.8495978941`
    - selected constraints `1059`
- Read:
  - this is not a large methodological change and should not be described that
    way
  - it is a small but real upstream support improvement on the big metric, and
    it keeps the donor-inclusive checkpoint path more symmetric with the accepted
    CPS checkpoint support rule

## 2026-04-12 keep PE-style PUF person-expansion randomness

- Code:
  - keep PE-style random-in-bin decoding for `_puf_agerange`,
    `_puf_agedp*`, and `_puf_earnsplit` in
    `src/microplex_us/data_sources/puf.py`
  - keep PE-style spouse/dependent sex draws in the same PE-demographics branch
  - keep the seeded PE-demographics regression in
    `tests/test_puf_source_provider.py`
- Why:
  - the previous implementation was a direct parity bug, not a modeling choice:
    it decoded PE demographic helper bins to fixed midpoints, while
    `policyengine-us-data` samples within those coded intervals and uses
    randomized spouse/dependent sex assignment
  - this is upstream alignment work on the exact PUF construction boundary,
    which is a better next step than inventing a new AGI heuristic
- Focused verification:
  - `python -m py_compile src/microplex_us/data_sources/puf.py tests/test_puf_source_provider.py`
  - `uv run pytest tests/test_puf_source_provider.py -q -k 'expand_to_persons or sample_tax_units'`
  - `uv run pytest tests/test_puf_source_provider.py -q -k 'not pre_tax_contributions_via_policyengine_subprocess'`
- Artifacts:
  - source-stage parity candidate:
    `artifacts/tmp_puf_source_stage_parity_personexpansion_20260412.json`
  - legacy source-stage parity reference:
    `artifacts/source_stage_parity_20260408/puf_2024_raw_source_stage_parity.json`
  - matched broader donor checkpoint:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_donors/broader-donors-puf-personexpansion-v1`
  - matched broader no-donor checkpoint:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_nodonors/broader-nodonors-puf-personexpansion-v1`
- Read:
  - raw PUF source-stage parity moves materially closer to PolicyEngine on the
    most relevant variables:
    - age weighted-mean ratio: `1.0367 -> 1.0275`
    - employment-income weighted-mean ratio: `1.2196 -> 0.9996`
    - taxable-interest weighted-mean ratio: `2.2495 -> 1.1774`
  - matched broader no-donor checkpoint:
    - baseline capped full-oracle loss: `0.7368409543`
    - candidate capped full-oracle loss: `0.7336528770`
    - delta: `-0.0031880773`
    - active-solve capped loss: `0.8497778115 -> 0.8005940161`
  - matched broader donor checkpoint:
    - baseline capped full-oracle loss: `0.7327632809`
    - candidate capped full-oracle loss: `0.7342149723`
    - delta: `+0.0014516915` worse
    - active-solve capped loss: `0.8495978941 -> 0.8037192584`
  - conclusion:
    - keep the upstream parity fix
    - do not overclaim it as an unconditional frontier win
    - treat the donor-path regression as the next interaction to investigate,
      rather than reverting a real PE-alignment correction

## 2026-04-12 keep only PE-style `EARNSPLIT` randomization by default

- Code:
  - keep PE-style `EARNSPLIT` sampling in
    `src/microplex_us/data_sources/puf.py`
  - revert default PE-demographics age-bin and spouse/dependent-sex
    randomization in the same file
  - keep the updated PE-demographics regression in
    `tests/test_puf_source_provider.py`
- Why:
  - the first bundled parity fix mixed two conceptually separate changes:
    - age/sex randomization
    - income-split randomization
  - the only clean way to decide what belongs in the default path was a matched
    ablation on the broader donor checkpoint
- Focused verification:
  - `python -m py_compile src/microplex_us/data_sources/puf.py tests/test_puf_source_provider.py`
  - `uv run pytest tests/test_puf_source_provider.py -q -k 'expand_to_persons or sample_tax_units'`
  - `uv run pytest tests/test_puf_source_provider.py -q -k 'not pre_tax_contributions_via_policyengine_subprocess'`
- Artifacts:
  - donor baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_donor_stateage1_donors/broader-donors-donor-stateage1-v1`
  - age/sex-only ablation:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_ageonly_donors/broader-donors-puf-personexpansion-ageonly-v1`
  - earnsplit-only ablation:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_earnsplitonly_donors/broader-donors-puf-personexpansion-earnsplitonly-v1`
  - real code-path confirmation:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_default_donors/broader-donors-puf-personexpansion-default-v2`
- Read:
  - age/sex-only is clearly the wrong half for the current frontier objective:
    - baseline capped full-oracle loss: `0.7327632809`
    - candidate: `0.7463902007`
    - delta: `+0.0136269199` worse
  - earnsplit-only is clearly the right half:
    - candidate: `0.7176041064`
    - delta vs baseline: `-0.0151591745`
    - active-solve capped loss: `0.8495978941 -> 0.7726915403`
  - the real code-path rerun matches the winning ablation exactly
  - conclusion:
    - default to PE-style `EARNSPLIT` randomization
    - do not default to PE-style age/sex randomization yet
    - treat age-bin randomization as an open parity lane rather than a settled
      improvement

## 2026-04-12 widen deferred family focus to 7 after `EARNSPLIT`

- Code:
  - `src/microplex_us/pipelines/pe_us_data_rebuild.py`
  - `tests/pipelines/test_pe_us_data_rebuild.py`
  - `tests/pipelines/test_pe_us_data_rebuild_checkpoint.py`
  - `artifacts/experiment_index.jsonl`
  - `docs/methodology-ledger.md`
- Why:
  - after the accepted `EARNSPLIT` fix, the sharpest surviving rows were no
    longer mostly age/AGI; the worst individual cells were now concentrated in
    `aca_ptc` and `rental_income`
  - the staged selector was still spending its family slots on AGI and EITC
    pairs, so ACA and rental were being excluded from deferred consideration
    even when they were among the highest-error rows
- Focused verification:
  - matched broader donor checkpoint with `top_family_count = 7`
  - donor-free broader confirmation with `top_family_count = 7`
  - `uv run pytest tests/pipelines/test_pe_us_data_rebuild.py tests/pipelines/test_pe_us_data_rebuild_checkpoint.py -q`
- Artifacts:
  - donor baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_default_donors/broader-donors-puf-personexpansion-default-v2`
  - donor family-7 rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1`
  - donor-free baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_default_nodonors/broader-nodonors-puf-personexpansion-default-v2`
  - donor-free confirmation:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_nodonors/broader-nodonors-puf-personexpansion-family7-v1`
- Read:
  - on the broader donor run, widening deferred family focus from `4` to `7`
    improves capped full-oracle loss from `0.7176041064` to `0.7044626415`
  - the selected deferred families now explicitly include:
    - `aca_ptc|domain=aca_ptc`
    - `rental_income|domain=rental_income`
  - the matched donor-free broader run also improves from `0.7170633141` to
    `0.7039665310` with the same focused family set
  - conclusion:
    - promote `top_family_count = 7` into the default rebuild policy
    - keep geography focus at `4`
    - treat ACA/rental as active deferred-calibration families rather than
      residuals that should stay outside the search surface

## 2026-04-12 reject full PUF age/sex randomization again on top of family-7

- Code:
  - `src/microplex_us/data_sources/puf.py` was restored to the earnsplit-only
    default after the retest
  - `tests/test_puf_source_provider.py` was restored to the incumbent
    earnsplit-only regression expectations
  - `artifacts/experiment_index.jsonl`
  - `docs/methodology-ledger.md`
- Why:
  - revisiting upstream person structure was reasonable, but this specific
    PE-style age/sex path had already lost once and needed to beat the current
    stronger family-7 default, not the older top-family-4 baseline
  - the clean test was a one-axis donor rerun with the current default config,
    not another parity argument in the abstract
- Focused verification:
  - `uv run pytest tests/test_puf_source_provider.py -q -k 'expand_to_persons_uses_pe_demographic_helpers_when_present or expand_to_persons_preserves_joint_tax_unit_monetary_totals or expand_to_persons_splits_negative_joint_self_employment_losses or expand_to_persons_clears_status_flags_for_non_head_members'`
- Artifacts:
  - current donor incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1`
  - full-rng retest:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_rng_donors/broader-donors-puf-personexpansion-rng-v1`
- Read:
  - donor incumbent capped full-oracle loss:
    - `0.7044626415`
  - full-rng retest:
    - `0.7111876263`
  - delta:
    - `+0.0067249848` worse
  - conclusion:
    - keep the earnsplit-only default
    - treat full PE-style age/sex randomization as re-rejected for the current
      frontier objective
    - move the next upstream work to AGI or EITC structure, not back into this
      same person-expansion branch

## 2026-04-12 keep CPS tax-unit structure at the source boundary

- Code:
  - `src/microplex_us/data_sources/cps.py`
  - `tests/test_cps_source_provider.py`
  - `artifacts/experiment_index.jsonl`
  - `docs/methodology-ledger.md`
- Why:
  - a direct code review against `policyengine-us-data` showed the main CPS
    structural gap was that source tax-unit semantics were still too flat in
    Microplex even when later pipeline stages could reconstruct similar roles
  - the clean fix was to derive tax-unit head/spouse/dependent roles,
    jointness, and dependent counts from raw `TAX_ID` in the CPS source layer
    instead of leaving that work implicit downstream
- Verification:
  - `python -m py_compile src/microplex_us/data_sources/cps.py tests/test_cps_source_provider.py`
  - `uv run pytest tests/test_cps_source_provider.py -q -k 'derives_tax_unit_roles_from_tax_id or caches_household_geography_on_persons or derives_survivor_and_dependent_social_security or loads_observation_frame or canonical_income_alias'`
- Artifacts:
  - donor incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1`
  - source-structure rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_taxunit_structure_donors/broader-donors-cps-taxunit-structure-v1`
- Read:
  - frontier metric is neutral:
    - `0.7044626415 -> 0.7044626415`
  - conclusion:
    - keep the source-layer CPS tax-unit derivation
    - treat it as architecture cleanup and PE-boundary alignment, not as an
      independent frontier gain

## 2026-04-12 reject direct CPS student flag on the broader donor checkpoint

- Code:
  - `src/microplex_us/data_sources/cps.py` was restored after the test
  - `tests/test_cps_source_provider.py` was restored after the test
  - `artifacts/experiment_index.jsonl`
  - `docs/methodology-ledger.md`
- Why:
  - after moving tax-unit structure to the source boundary, the next narrow
    EITC-side parity hypothesis was to expose `is_full_time_college_student`
    directly from CPS `A_HSCOL`, because `policyengine-us` uses that input in
    qualifying-child logic
  - the clean test was a one-axis broader donor rerun, not an argument from
    policy parity alone
- Verification:
  - `python -m py_compile src/microplex_us/data_sources/cps.py tests/test_cps_source_provider.py`
  - `uv run pytest tests/test_cps_source_provider.py -q -k 'derives_tax_unit_roles_from_tax_id or caches_household_geography_on_persons or derives_survivor_and_dependent_social_security or loads_observation_frame or canonical_income_alias'`
- Artifacts:
  - donor incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1`
  - student-input rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_student_donors/broader-donors-cps-student-v1`
- Read:
  - direct CPS student input is strongly harmful on the broader donor frontier:
    - `0.7044626415 -> 0.7815651801`
  - conclusion:
    - do not promote `is_full_time_college_student` into the current mixed-source
      broader default
    - treat this as another case where direct PE CPS inputs are not
      automatically plug-compatible with the broader Microplex path

## 2026-04-12 reject partial preserved tax units as the broader mixed-source default

- Code:
  - `src/microplex_us/pipelines/us.py`
  - `tests/pipelines/test_us.py`
  - `artifacts/experiment_index.jsonl`
  - `docs/methodology-ledger.md`
- Why:
  - after the CPS tax-unit structure cleanup, the strongest remaining direct
    alignment hypothesis was to keep authoritative source tax-unit IDs for
    households that already have them and only optimize donor households with
    missing tax-unit IDs
  - that is a coherent architectural boundary, but it still had to beat the
    broader donor frontier metric rather than just look more PE-like on paper
- Verification:
  - `python -m py_compile src/microplex_us/pipelines/us.py tests/pipelines/test_us.py`
  - `uv run pytest tests/pipelines/test_us.py -q -k 'preserve_existing_tax_unit_ids or falls_back_when_existing_tax_unit_ids_cross_households or partially_preserves_existing_tax_unit_ids'`
- Artifacts:
  - donor incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1`
  - partial-preservation rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_partial_preserve_taxunits_donors/broader-donors-partial-preserve-taxunits-v1`
- Read:
  - capped full-oracle loss regresses slightly:
    - `0.7044626415 -> 0.7055670761`
  - active-solve capped loss improves materially:
    - `0.7909211525 -> 0.7648463685`
  - conclusion:
    - keep the mixed-preservation code path as an optional capability
    - do not promote `policyengine_prefer_existing_tax_unit_ids=True` into the
      current broader default
    - move the next upstream work off this boundary and back to the remaining
      AGI and EITC input/eligibility lanes

## 2026-04-12 keep PE-style CPS `ssn_card_type` in the broader donor default

- implemented PE-style CPS `ssn_card_type` derivation in
  `src/microplex_us/data_sources/cps.py`
  - use the raw CPS immigration, benefits, work, and housing-assistance fields
    to assign:
    - `CITIZEN`
    - `NON_CITIZEN_VALID_EAD`
    - `OTHER_NON_CITIZEN`
    - `NONE`
  - added a safe fallback so if a future CPS extract is missing one of the raw
    helper fields, Microplex still emits `ssn_card_type = CITIZEN` rather than
    silently dropping the column
- allowed `ssn_card_type` into the PE export surface in
  `src/microplex_us/policyengine/us.py`
  - mixed-source missing rows now backfill to `CITIZEN` at export time
- focused verification:
  - `python -m py_compile src/microplex_us/data_sources/cps.py src/microplex_us/policyengine/us.py tests/test_cps_source_provider.py tests/policyengine/test_us.py`
  - `uv run pytest tests/test_cps_source_provider.py -q -k 'ssn_card_type or derives_tax_unit_roles_from_tax_id'`
  - `uv run pytest tests/policyengine/test_us.py -q -k 'default_policyengine_us_export_surface or defaults_missing_ssn_card_type_to_citizen'`
- artifact comparison:
  - incumbent broader donor default:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1`
  - `ssn_card_type` rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
- read:
  - capped full-oracle loss improves:
    - `0.7044626415 -> 0.6955460`
  - active-solve capped loss also improves:
    - `0.7909211525 -> 0.7813926586`
  - the direct `ssn_card_type` family improves sharply:
    - `person_count|domain=ssn_card_type`
    - `1.0000 -> 0.3786`
  - EITC child-count families improve:
    - `eitc|domain=eitc,eitc_child_count`
    - `0.8283 -> 0.7499`
    - `tax_unit_count|domain=eitc,eitc_child_count`
    - `0.8154 -> 0.7408`
  - the aggregate `eitc` row itself gets worse:
    - `0.1066 -> 0.2954`
  - conclusion:
    - keep this change because it clears the frontier bar and the direction of
      movement is specifically consistent with the intended EITC-identification
      lane
    - describe it narrowly: it improves the full-oracle metric and the
      identification / child-count families, not “all EITC targets”

## 2026-04-12 reject PE-style EITC take-up and voluntary filing inputs

- implemented a PE-style `takes_up_eitc` /
  `would_file_taxes_voluntarily` tax-unit input path in
  `src/microplex_us/pipelines/us.py`
  - the prototype used materialized `eitc_child_count` to assign PE-style
    take-up rates and voluntary-filing draws before export
  - a review pass also hardened the prototype so materialization failures fell
    back explicitly instead of silently dropping the new columns
- temporarily exposed those variables in `src/microplex_us/policyengine/us.py`
  so the PE export surface could carry them
- focused verification before the checkpoint:
  - `python -m py_compile src/microplex_us/pipelines/us.py src/microplex_us/policyengine/us.py tests/pipelines/test_us.py tests/policyengine/test_us.py`
  - `uv run pytest tests/pipelines/test_us.py -q -k 'build_policyengine_entity_tables'`
  - `uv run pytest tests/policyengine/test_us.py -q -k 'default_policyengine_us_export_surface or defaults_missing_ssn_card_type_to_citizen'`
- artifact comparison:
  - incumbent broader donor default:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
  - take-up rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_takeup_donors/broader-donors-takeup-v1`
- read:
  - capped full-oracle loss regresses:
    - `0.6955460 -> 0.7041134`
  - active-solve capped loss regresses:
    - `0.7813927 -> 0.7896826`
  - EITC child-count families improve:
    - `eitc|domain=eitc,eitc_child_count`
    - `0.7499 -> 0.7030`
    - `tax_unit_count|domain=eitc,eitc_child_count`
    - `0.7408 -> 0.6757`
  - but the aggregate `eitc` family gets worse:
    - `0.2954 -> 0.4010`
  - ACA amount and count families also get worse:
    - `aca_ptc|domain=aca_ptc`
    - `2.3488 -> 2.5737`
    - `tax_unit_count|domain=aca_ptc`
    - `1.1521 -> 1.3708`
  - conclusion:
    - reject the change on the current broader donor frontier metric
    - revert the code path and keep the broader runtime at the
      `ssn_card_type` incumbent
    - do **not** interpret this as rejecting the conceptual separation between:
      - filing because required
      - filing voluntarily for non-credit reasons
      - filing to claim refundable credits
      - taking up EITC conditional on filing / eligibility
    - the rejection is narrower: the current late export-layer port of
      `takes_up_eitc` and `would_file_taxes_voluntarily` is not yet the right
      implementation in the broader mixed-source runtime
    - if this lane is revisited later, treat it as a challenger path that
      needs upstream filer / take-up calibration evidence rather than another
      direct PE-input port

## 2026-04-12 reject stronger `state x age-band` checkpoint floors

- tested a matched broader donor checkpoint with stronger upstream checkpoint
  sampling support:
  - CPS `state_age_floor = 2`
  - donor `state_age_floor = 2`
- artifact comparison:
  - incumbent broader donor default:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
  - stronger-floor rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_stateage2_donors/broader-donors-stateage2-v1`
- read:
  - capped full-oracle loss regresses sharply:
    - `0.6955460 -> 0.7361964`
  - active-solve capped loss also regresses:
    - `0.7813927 -> 0.8371045`
  - the target family that motivated the run does improve:
    - `person_count|domain=age`
    - `0.4681 -> 0.4480`
  - but the broader frontier gets worse because AGI, EITC-child-count, and ACA
    families all move in the wrong direction:
    - `person_count|domain=adjusted_gross_income`
    - `0.7119 -> 0.7553`
    - `tax_unit_count|domain=adjusted_gross_income`
    - `0.6372 -> 0.6618`
    - `eitc|domain=eitc,eitc_child_count`
    - `0.7499 -> 0.8880`
    - `tax_unit_count|domain=eitc,eitc_child_count`
    - `0.7408 -> 0.8755`
    - `aca_ptc|domain=aca_ptc`
    - `2.3488 -> 2.9982`
  - conclusion:
    - reject stronger checkpoint age-floor heuristics
    - keep the accepted `state_age_floor = 1` incumbent
    - move the next parity work to upstream PUF age/AGI construction rather
      than stronger checkpoint support heuristics

## 2026-04-12 reject high-AGI-preserving PUF checkpoint samples

- tested a matched broader donor checkpoint with a checkpoint-only PUF sampling
  change:
  - preserve the top raw PUF AGI tail whenever `sample_n` is active
  - keep the rest of the broader donor runtime unchanged
- artifact comparison:
  - incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
  - candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_agi_tail_donors/broader-donors-puf-agi-tail-v1`
- metric read:
  - capped full-oracle loss:
    - `0.6955460 -> 1.1132009`
  - active-solve capped loss:
    - `0.7813927 -> 1.9290`
  - selected constraints:
    - `1031 -> 1163`
  - a fast raw PUF source-stage proxy did improve taxable-interest and
    dividend parity, but it simultaneously worsened self-employment and rental
    structure enough that the real broader checkpoint failed outright
- action:
  - reject high-AGI-preserving checkpoint PUF sampling
  - revert the checkpoint-only sampler code path completely
  - keep the broader donor incumbent on the accepted `ssn_card_type` runtime
  - continue the next parity work in upstream construction/imputation rather
    than checkpoint-only tail heuristics

## Update rule

Update this document when any of the following changes:

- the canonical measurement contract
- the default runtime pipeline shape
- the default imputation or selection method family
- the meaning of the parity/audit sidecars
- the set of artifacts required for a headline claim
- the boundary between incumbent-compatibility work and challenger work

## Paper extraction rule

When writing the eventual paper:

1. Start from this ledger, not from memory.
2. Pull claims only from code-backed docs and artifact-backed evidence.
3. Preserve the distinction between canonical, provisional, and open items.
4. Cite the exact artifact family that supported each headline claim.
5. Avoid rewriting temporary engineering names like `pe_us_data_rebuild` into
   misleading methodological claims.

## Naming note

Some internal module names still say `pe_us_data_rebuild`.

Treat that as historical naming, not as the canonical project description. The
canonical description is:

- Microplex is the runtime
- PolicyEngine is the oracle/evaluator
- PE-US-data is the incumbent comparator

## 2026-04-12 reject standalone ACA take-up construction patch, keep the concept

- traced the ACA residual lane and confirmed that
  `takes_up_aca_if_eligible` is a real PE construction-stage input rather than
  a made-up Microplex feature
  - PE-US-data assigns it during CPS construction
  - PE-US uses it directly in the ACA PTC formula
- implemented the narrowest plausible version in
  `src/microplex_us/pipelines/us.py` and `src/microplex_us/policyengine/us.py`
  as a direct probe:
  - add a deterministic PE-style `takes_up_aca_if_eligible` draw during
    tax-unit construction
  - expose that variable on the PE export surface
- verification before evaluation:
  - `python -m py_compile src/microplex_us/pipelines/us.py src/microplex_us/policyengine/us.py tests/pipelines/test_us.py tests/policyengine/test_us.py`
  - `uv run pytest tests/pipelines/test_us.py -q -k 'aca_takeup or export_policyengine_dataset or derives_tax_input_columns'`
  - `uv run pytest tests/policyengine/test_us.py -q -k 'default_policyengine_us_export_surface_avoids_formula_aggregates'`
- evaluation method:
  - reevaluated the incumbent broader donor synthetic population in memory
    against the shared oracle instead of running a fresh saved checkpoint,
    because disk pressure made a large rerun unreliable
  - baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
  - saved readout:
    `artifacts/tmp_broader_aca_takeup_recalibration_20260412.json`
- metric read:
  - capped full-oracle loss regresses:
    - `0.6955460 -> 0.8211989`
  - active-solve capped loss improves:
    - `0.7813927 -> 0.7013644`
  - the intended ACA families improve sharply:
    - `aca_ptc|domain=aca_ptc`
    - `2.3488 -> 0.5529`
    - `tax_unit_count|domain=aca_ptc`
    - `1.1521 -> 0.7112`
    - `person_count|domain=aca_ptc,is_aca_ptc_eligible`
    - `1.0994 -> 0.7771`
- action:
  - reject this implementation from the default broader runtime and revert it
  - keep the concept in scope as required upstream parity work
  - interpret the result narrowly:
    - this is not evidence against separate ACA take-up behavior
    - it is evidence that a standalone tax-unit/export-boundary patch is the
      wrong implementation boundary in the current mixed-source runtime

## 2026-04-12 ACA child gap is mostly Medicaid crowd-out, not missing ACA knobs

- ACA-specific review conclusion:
  - beyond raw `has_marketplace_health_coverage` / `has_esi`, the only real
    ACA-specific upstream input is `takes_up_aca_if_eligible`
  - there is no large hidden ACA-specific construction surface still missing
    from Microplex before export
- diagnostic comparison:
  - compared the incumbent broader donor artifact
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1/policyengine_us.h5`
    against PE's `enhanced_cps_2024.h5`
  - saved readout:
    `artifacts/tmp_broader_aca_eligibility_decomposition_20260412.json`
- read:
  - the incumbent has higher under-20 Medicaid/CHIP eligibility than the PE
    baseline:
    - `eligible_share_under20`: `0.4909 -> 0.6094`
    - `medicaid_share_under20`: `0.3930 -> 0.5278`
  - the dominant driver is much lower child-unit `medicaid_income_level` in
    the incumbent:
    - median under-20 `medicaid_income_level`:
      `15.1512 -> 1.6054`
    - p75 under-20 `medicaid_income_level`:
      `364.3831 -> 3.9464`
  - child filing-status mix is not the main failure mode:
    - the incumbent actually places more under-20s in `JOINT` units than the
      PE baseline
  - current interpretation:
    - the next lane is AGI / tax-unit construction and imputation for child
      units
    - ACA should no longer be treated as primarily an ACA-specific export/input
      problem

## 2026-04-13 reject source tax-unit preservation as the broader donor default

- hypothesis:
  - because the seeded integrated microdata already has near-PE under-20
    singleton-tax-unit structure, preserving source `tax_unit_id` values in the
    PE rebuild path might be a direct parity win and should beat the current
    optimizer-driven rebuild on the big metric
- code path under test:
  - flipped `policyengine_prefer_existing_tax_unit_ids` to `True` only in
    `src/microplex_us/pipelines/pe_us_data_rebuild.py`
  - left the generic `USMicroplexBuildConfig` default unchanged; this was only
    a PE rebuild / checkpoint default probe
  - updated the default-config assertions in
    `tests/pipelines/test_pe_us_data_rebuild.py`
    and
    `tests/pipelines/test_pe_us_data_rebuild_checkpoint.py`
- verification:
  - focused config tests passed
  - an explorer review found no concrete code-level regression path from the
    default flip
  - matched broader donor source rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_preserve_taxunits_default_donors/broader-donors-preserve-taxunits-default-v1`
- read:
  - the synthetic-data proxy was slightly positive:
    - optimizer: `0.63654`
    - preserve existing IDs: `0.63583`
  - but the real broader donor checkpoint still loses on the mission metric:
    - incumbent:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
    - candidate:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_preserve_taxunits_default_donors/broader-donors-preserve-taxunits-default-v1`
    - capped full-oracle loss:
      `0.6955 -> 0.6977`
    - active-solve capped loss:
      `0.7814 -> 0.7624`
    - selected constraints:
      `1031 -> 1019`
- decision:
  - reject the default flip and revert it from the canonical PE rebuild path
  - keep source-tax-unit preservation as an optional structural probe rather
    than the default
- interpretation:
  - this is another case where a promising structural parity clue clears a
    local or proxy test but still misses on the real broader frontier metric
  - the child-unit AGI / Medicaid-income miss is still best treated as an
    upstream construction / source-impute problem, not as a rebuild-default
    switch we can justify today

## 2026-04-13 reject minor-household source tax-unit preservation

- hypothesis:
  - if full source-tax-unit preservation is too broad, preserve source
    `tax_unit_id` values only in households with minors and let the optimizer
    rebuild adult-only households
- code path under test:
  - added an opt-in experiment flag in `src/microplex_us/pipelines/us.py` so
    preserved tax units applied only to households with at least one person
    under age 20
  - added a focused household-level regression in
    `tests/pipelines/test_us.py`
- verification:
  - focused `py_compile` and preservation tests passed before the real run
  - matched broader donor source rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_minorhousehold_preserve_taxunits_donors/broader-donors-minorhousehold-preserve-taxunits-v1`
- read:
  - it materially fixes the exact child-structure symptom:
    - under-20 singleton-tax-unit share:
      `0.1538 -> 0.0345`
    - under-20 mean `medicaid_income_level`:
      `2.7279 -> 3.0408`
    - under-20 median `medicaid_income_level`:
      `1.5131 -> 1.8068`
  - but it still loses on the broader donor mission metric:
    - capped full-oracle loss:
      `0.6955 -> 0.6985`
    - active-solve capped loss:
      `0.7814 -> 0.7614`
    - selected constraints:
      `1031 -> 1031`
- decision:
  - reject the experiment and revert the code path
- interpretation:
  - tax-unit assignment is only part of the child-lane miss
  - the remaining gap is in child-linked AGI component construction, not just
    which adults children are attached to

## 2026-04-13 under-20 AGI miss is now clearly a component-construction problem

- diagnostic comparison:
  - compared the PE baseline, the broader donor incumbent, and the rejected
    minor-household-preservation rerun on person-mapped under-20 tax-unit
    aggregates
- read:
  - the rejected preservation rerun raises under-20 mapped AGI and Medicaid
    MAGI, but both remain far below the PE baseline:
    - under-20 mapped `adjusted_gross_income`:
      - PE baseline: `137623.5`
      - incumbent: `85755.2`
      - minor-preserve rerun: `98230.0`
    - under-20 mapped `medicaid_magi`:
      - PE baseline: `140533.9`
      - incumbent: `86338.8`
      - minor-preserve rerun: `98586.5`
  - the surviving gap looks like AGI composition, not simple child attachment:
    - under-20 mapped `tax_unit_partnership_s_corp_income`:
      - PE baseline: `23323.0`
      - incumbent: `9568.7`
      - minor-preserve rerun: `10710.1`
    - under-20 mapped `net_capital_gains`:
      - PE baseline: `3200.0`
      - incumbent: `534.3`
      - minor-preserve rerun: `945.7`
    - under-20 mapped `qualified_dividend_income`:
      - PE baseline: `47.2`
      - incumbent: `0.0`
      - minor-preserve rerun: `0.0`
    - under-20 mapped `tax_exempt_interest_income`:
      - PE baseline: `4.68`
      - incumbent: `0.0`
      - minor-preserve rerun: `0.0`
- action:
  - move the next direct-path lane to AGI component construction / source-impute
    parity for child-linked tax units
  - stop spending more effort on source-tax-unit preservation variants

## 2026-04-13 reject PE-style sequential PUF joint-QRF imputation in the current donor runtime

- hypothesis:
  - the child-linked AGI miss might be coming from a real architecture gap:
    PE imputes PUF tax variables with one sequential QRF over a joint block,
    while Microplex currently donor-imputes those leaves mostly as independent
    blocks
  - a PE-like grouped sequential-QRF challenger for the main PUF AGI leaves
    could therefore be a more direct parity move than more tax-unit heuristics
- code path under test:
  - added a non-default `sequential_qrf` donor-imputer backend
  - grouped the main PUF AGI component leaves into one joint donor block when
    that backend was selected
  - added focused regressions, then ran matched medium and broader donor
    checkpoints
- verification:
  - focused `py_compile` and the new block/backend regression slice passed
    before the real runs
  - matched medium donor rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_sequential_puf_joint_medium/medium-donors-sequential-puf-joint-v1`
  - matched broader donor rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_sequential_puf_joint_donors/broader-donors-sequential-puf-joint-v1`
- read:
  - the broader donor frontier metric regresses:
    - capped full-oracle loss:
      `0.6955 -> 0.7190`
    - active-solve capped loss:
      `0.7814 -> 0.7757`
    - selected constraints:
      `1031 -> 999`
  - the medium donor rerun is also not attractive:
    - capped full-oracle loss:
      `0.9426`
    - active-solve capped loss:
      `0.6618`
  - a direct matched CPS+PUF stage probe on a `1000/1000` sample shows the
    PE-like backend changes the child-linked AGI composition aggressively, but
    not in a clearly correct direction:
    - under-20 linked `qualified_dividend_income`:
      `40.0 -> 1199.0`
    - under-20 linked `taxable_interest_income`:
      `507.2 -> 1634.6`
    - under-20 linked `tax_exempt_interest_income`:
      `4.66 -> 249.4`
    - under-20 linked `taxable_pension_income`:
      `9118.5 -> 19317.6`
- decision:
  - reject the challenger and revert the experiment code
- interpretation:
  - the parity observation is still useful: PE really does use a more joint
    QRF architecture for this lane
  - but a direct port into the current donor/rank-match runtime is not
    numerically safe enough to keep
  - keep the next lane on narrower upstream AGI construction / source-impute
    parity for child-linked units, not on a wholesale donor-backend swap

## 2026-04-13 reject post-donor zeroing of PUF tax leaves on dependent rows

- diagnosis:
  - the child-linked AGI misallocation is not coming from raw PUF person
    expansion
  - direct inspection of `PUFSourceProvider(..., expand_persons=True)` on a
    matched sample showed under-20 dependent rows carry zero
    `partnership_s_corp_income`, `taxable_pension_income`,
    `taxable_interest_income`, `qualified_dividend_income`, and
    `tax_exempt_interest_income`
  - the incumbent broader donor seed artifact instead carried large dependent
    mass on some of those leaves, especially:
    - under-20 `partnership_s_corp_income`: `4.09M`
    - under-20 `taxable_pension_income`: `17.77M`
    - under-20 `taxable_interest_income`: `33.98k`
  - so the structural clue was real: donor integration is creating dependent-row
    mass that is not present in raw expanded PUF
- tested:
  - added a post-donor semantic guard that zeroed the affected PUF tax leaves on
    rows with `is_tax_unit_dependent > 0`
  - verified locally that the guard nearly removed the seeded child mass:
    - under-20 `partnership_s_corp_income`: `4.09M -> 87.3k`
    - under-20 `taxable_pension_income`: `17.77M -> 172.6k`
    - under-20 `taxable_interest_income`: `33.98k -> 3.28k`
  - ran a matched broader donor checkpoint:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_dependent_zero_tax_leaves_donors/broader-donors-dependent-zero-tax-leaves-v1`
- read:
  - the real frontier result is decisively worse:
    - capped full-oracle loss:
      `0.6955 -> 1.1372`
    - active-solve capped loss:
      `0.7814 -> 1.6581`
  - the first calibration stage was already much worse than the incumbent:
    - post-stage-1 capped full-oracle loss:
      `1.3660`
  - later deferred stages improved on that bad starting point, but still never
    recovered:
    - post-stage-2 capped full-oracle loss:
      `1.2460`
    - final capped full-oracle loss:
      `1.1372`
- decision:
  - reject the guard and revert the code
- interpretation:
  - the structural diagnosis still holds: donor integration is where the
    dependent-row mass is being created
  - but a blunt post-donor zeroing rule destroys too much signal elsewhere and
    is not a valid repair
  - the next lane should target narrower donor-impute/source-impute parity for
    these leaves, not post-hoc dependent suppression

## 2026-04-13 reject dependent-role partitioning inside donor imputation

- hypothesis:
  - the blunt post-donor zeroing guard failed because it acted too late
  - a narrower parity move would be to keep the donor-impute path but partition
    fitting and matching by `is_tax_unit_dependent` for the leaves that were
    actually exploding on child-linked rows:
    - `partnership_s_corp_income`
    - `taxable_pension_income`
    - `taxable_interest_income`
- tested:
  - added a block-level exact-match partition on `is_tax_unit_dependent` for
    those singleton donor blocks
  - verified the block-planning assertions locally, then ran a matched broader
    donor checkpoint:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_dependent_partition_tax_leaves_donors/broader-donors-dependent-partition-tax-leaves-v1`
  - also requested an independent code review of the partition implementation
- read:
  - the frontier result is again decisively worse:
    - capped full-oracle loss:
      `0.6955 -> 1.2406`
    - active-solve capped loss:
      `0.7814 -> 1.6943`
  - the seeded child-dependent mass is still strongly suppressed:
    - under-20 `partnership_s_corp_income`: `74.5k`
    - under-20 `taxable_pension_income`: `257.4k`
    - under-20 `taxable_interest_income`: `3.33k`
  - so the narrower support change did move the child rows, but still did not
    improve the real oracle objective
- review findings:
  - null partition keys would fall through to the global donor fallback instead
    of staying partitioned
  - `is_tax_unit_dependent` partition labels were lossy after entity projection
    because the projected value could come from a `FIRST`-style collapse rather
    than the unit’s real dependent composition
  - empty donor partitions also fell back silently to the global donor pool,
    which weakened the exact-match semantics
- decision:
  - reject the experiment and revert the code
- interpretation:
  - the structural clue is still right: donor integration is the failure point
  - but neither blunt post-donor zeroing nor this first exact-partition repair
    is a safe or effective solution
  - the next lane should move closer to PE source-impute structure itself:
    leaf-specific block design and condition-surface parity for these AGI
    components, rather than more role-suppression heuristics

## 2026-04-13 reject richer singleton condition surfaces for PUF child-linked tax leaves

- hypothesis:
  - the previous parity attempts may have failed because the current
    `pe_prespecified` donor path was forcing these sparse PUF leaves onto a
    demographic-only condition surface
  - a narrower repair would keep the existing donor backend and singleton block
    structure, but enrich the preferred condition surface for
    `partnership_s_corp_income`, `taxable_interest_income`, and
    `taxable_pension_income` with current income state
- code path under test:
  - expanded the preferred condition vars for those leaves to include
    `income`, `employment_income`, `self_employment_income`, and for pension
    also `social_security`
  - added focused regressions confirming that only those leaves changed their
    preferred-condition surface and that the pipeline resolved the extra income
    predictor when it was available
  - ran a matched broader donor checkpoint:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_income_aware_puf_tax_leaves_donors/broader-donors-income-aware-puf-tax-leaves-v1`
- verification:
  - focused `py_compile` passed
  - focused `tests/test_variables.py` and `tests/pipelines/test_us.py` slices
    passed before the real rerun
- read:
  - the broader donor frontier metric still regresses:
    - capped full-oracle loss:
      `0.6955 -> 0.7420`
    - active-solve capped loss:
      `0.7814 -> 0.8499`
    - selected constraints:
      `1031 -> 1027`
  - staged calibration improves the candidate internally, but the final result
    still loses to the incumbent:
    - post-stage-1 capped full-oracle loss:
      `0.8326`
    - post-stage-2 capped full-oracle loss:
      `0.7879`
    - final capped full-oracle loss:
      `0.7420`
- PE code read:
  - PolicyEngine does not solve this lane with richer singleton donor surfaces
  - these leaves sit inside one sequential PUF QRF pass, with
    `partnership_s_corp_income` also included in the override pass
  - the only donor-survey block directly touching one of them is the ACS path
    for `taxable_pension_income`
- decision:
  - reject the richer singleton condition-surface patch and revert the code
- interpretation:
  - this was a reasonable approximation attempt, but it still tried to emulate a
    joint sequential-QRF lane with a patched singleton-donor runtime
  - local code read also confirms the ownership seam: provider order is
    `CPS -> PUF -> ACS -> SIPP -> SCF`, these leaves are mapped directly by the
    PUF adapter before person expansion, and the current rebuild does not treat
    them as explicit direct-override variables
  - the next lane should stop broadening singleton condition surfaces and move
    toward the actual structure gap: how these PUF leaves enter the build before
    donor integration and how much of that lane should remain PUF-native rather
    than generic donor-imputed

## 2026-04-13 reject a standalone PUF-native QRF hook for the main child-linked AGI leaves

- hypothesis:
  - the richer singleton-condition experiment lost because it was still trying
    to fix a PUF-owned lane inside the generic donor runtime
  - a narrower and more PE-aligned repair would move these leaves into a
    provider-owned QRF hook at PUF tax-unit load time for
    `partnership_s_corp_income`, `taxable_interest_income`, and
    `taxable_pension_income`, then let the normal donor integration stack use
    the rebuilt PUF support
- code path under test:
  - added a temporary PE-style QRF hook in `map_puf_variables()` /
    `_build_puf_tax_units()` for exactly those three leaves
  - trained the temporary models from the PE extended CPS artifact and passed
    them through the PUF provider only; no calibration defaults or donor-engine
    logic changed
  - ran a matched broader donor checkpoint:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_puf_tax_leaf_qrf_donors/broader-donors-puf-tax-leaf-qrf-v1`
- verification:
  - focused `py_compile` passed
  - focused `tests/test_puf_source_provider.py` slices passed before the real
    rerun
- read:
  - the broader donor frontier metric regresses sharply:
    - capped full-oracle loss:
      `0.6955 -> 0.8729`
    - active-solve capped loss:
      `0.7814 -> 1.1545`
    - selected constraints:
      `1031 -> 1064`
  - the run completes cleanly, so this is a real model loss rather than a
    harness artifact
- decision:
  - reject the standalone PUF-native QRF hook and revert the code
- interpretation:
  - this confirms that the structure problem is not just “put a QRF on the PUF
    side”
  - moving the hook to the provider boundary without also reproducing the rest
    of PolicyEngine’s sequential clone/impute shape still gives the wrong
    runtime behavior
  - the next lane should stay structural, but it needs to revisit the ownership
    boundary more carefully than “PUF provider QRF for three leaves”

## 2026-04-13 add a child tax-unit AGI drift summary tool

- motivation:
  - the sequential PUF joint experiment surfaced large child-linked AGI shifts
    that were hard to isolate from the full-oracle metrics
  - we need a repeatable summary to compare child vs adult income components
    across seed, calibrated, and synthetic stages before touching calibration
- tool:
  - `python -m microplex_us.pipelines.summarize_child_tax_unit_agi_drift <artifact>`
  - summarizes per-person subsets (all, under-20, dependents-under-20, adults)
    and per-tax-unit subsets (all, with-children, without-children)
  - uses the income variables that exist in the current artifact surfaces
    (total/income/employment/wage/self-employment/social-security/SSI/
    public-assistance/pension/dividend/rental/tax-leaf components)
- initial read:
  - wrote the latest summary to
    `artifacts/tmp_child_tax_unit_agi_drift_20260413.json`
  - this will be the baseline diagnostic for upcoming PUF AGI ownership
    experiments before we touch calibration boundaries

## 2026-04-13 child AGI drift comparison (calibrated stage)

- scope:
  - compared calibrated-stage child/adult income shares for three artifacts:
    - `broader-donors-ssn-card-type-v1`
    - `broader-donors-puf-personexpansion-family7-v1`
    - `broader-donors-sequential-puf-joint-v1`
  - metric: dependents-under-20 sum divided by adult sum for each variable
- read (dependents-under-20 sum share; calibrated stage):
  - broader donors ssn-card-type:
    - taxable interest: `0.0085`
    - taxable pension: `0.8507`
    - dividends: `0.0000`
    - partnership/S-corp: `0.9633`
    - rental: `0.0009`
    - wage: `0.0046`
    - employment: `0.0126`
  - broader donors puf-personexpansion family7:
    - taxable interest: `0.0000`
    - taxable pension: `0.0000`
    - dividends: `0.0000`
    - partnership/S-corp: `0.0000`
    - rental: `0.0000`
    - wage: `0.0000`
    - employment: `0.0000`
  - broader donors sequential PUF joint:
    - taxable interest: `0.3036`
    - taxable pension: `0.0960`
    - dividends: `0.1239`
    - partnership/S-corp: `0.2482`
    - rental: `0.0031`
    - wage: `0.0040`
    - employment: `0.0085`
- interpretation:
  - the sequential PUF joint path shifts significant child-linked mass into the
    interest/dividend/partnership lanes relative to the family7 baseline, while
    the SSN-card-type baseline already shows outsized child shares for pension
    and partnership components
  - the next structural fixes should aim to move child-linked mass away from
    these PUF tax leaves without collapsing legitimate child wage/employment
    mass

## 2026-04-14 dependent tax-leaf soft cap (broader donors)

- goal:
  - reduce dependent-row tax-leaf spikes by softly capping PUF tax leaves on
    dependents at a fraction of base earned income
  - configuration: `dependent_tax_leaf_soft_cap_multiplier=0.1`, base variables
    `employment_income`, `wage_income`, `self_employment_income`
  - capped variables: `taxable_interest_income`, `tax_exempt_interest_income`,
    `taxable_pension_income`, `dividend_income`,
    `qualified_dividend_income`, `non_qualified_dividend_income`,
    `partnership_s_corp_income`, `rental_income`
- run:
  - `artifacts/live_pe_us_data_rebuild_checkpoint_20260414_dependent_tax_leaf_soft_cap/broader-donors-dependent-tax-leaf-softcap-v1`
- result:
  - full-oracle capped loss:
    `0.6955 -> 1.1498`
  - active-solve capped loss:
    `0.7814 -> 1.6832`
  - candidate beats harness MAE and composite parity loss but still loses the
    native broad loss check
- decision:
  - reject the dependent tax-leaf soft cap guard
- interpretation:
  - the soft cap removes too much mass in the dependent tail without improving
    the full-oracle fit; this needs a structural donor/conditioning fix rather
    than a post-hoc clip

## 2026-04-14 donor conditioning diagnostics + structured supplement lane

- motivation:
  - the dependent soft-cap failure reinforced that the problem is in donor
    conditioning structure, not in post-hoc clipping
  - we needed artifact-level evidence for which predictors the
    `pe_prespecified` lane actually keeps and which shared predictors it drops
- instrumentation:
  - artifacts now carry `synthesis.donor_conditioning_diagnostics`
  - added `python -m microplex_us.pipelines.summarize_donor_conditioning
    <artifact>` to inspect selected vs dropped donor predictors by block
- current structural hypothesis:
  - keep the PE-style structural predictor backbone for the problematic
    zero-inflated PUF tax leaves
  - admit a narrow supplemental shared set
    (`employment_status`, `income`, `state_fips`) instead of reopening the full
    broad-common predictor surface
- status:
  - checkpoint run in progress; do not treat this as accepted or rejected yet

## 2026-04-14 structured PUF shared supplement lane (broader donors)

- run:
  - `artifacts/live_pe_us_data_rebuild_checkpoint_20260414_structured_puf_shared_supplement/broader-donors-structured-puf-shared-supplement-v1`
- result:
  - full-oracle capped loss:
    `0.6955 -> 1.1739`
  - active-solve capped loss:
    `0.7814 -> 1.7118`
  - native broad loss:
    `0.0202 -> 9.6703`
  - harness MAE/composite parity still beat the incumbent slice, but the run
    failed the native broad loss gate again
- diagnostic read:
  - the new donor-conditioning diagnostics show that for the four problematic
    PUF tax-leaf blocks in this run (`qualified/non-qualified dividend`,
    `partnership_s_corp_income`, `taxable_interest_income`,
    `taxable_pension_income`), the selected condition vars remained the pure
    PE structural set
  - the intended supplemental shared vars did not enter those blocks on the
    real artifact because they were not in the actual compatible shared overlap
    for those runs
- decision:
  - reject this exact supplement patch as a real fix
- interpretation:
  - this was more diagnostic than corrective: the structured lane is still too
    narrow in practice, but the immediate blocker is not just "allow three more
    vars in semantics metadata"
  - the next experiment needs to inspect why those income/state/employment
    features are absent from compatible overlap on the live PUF blocks, rather
    than assuming they can simply be appended to the preferred list

## 2026-04-14 structured supplement diagnostic smoke

- run:
  - `artifacts/live_pe_us_data_rebuild_checkpoint_20260414_structured_puf_shared_supplement_diag_smoke/broader-donors-structured-puf-shared-supplement-diagnostic-smoke-v1`
- question:
  - for the problematic PUF tax-leaf blocks, why do the requested supplemental
    shared predictors fail to enter the live `pe_prespecified` condition set?
- read:
  - `employment_status` failed with `incompatible_condition_support`
  - `state_fips` failed with `incompatible_condition_support`
  - `income` failed with `excluded_from_block_shared_overlap`
  - this pattern repeated across the four main problematic blocks:
    dividend split, `partnership_s_corp_income`, `taxable_interest_income`,
    and `taxable_pension_income`
- interpretation:
  - the main blocker is upstream of the preferred-list merge
  - `income` appears to be dropped before block-level shared-overlap selection
  - `employment_status` and `state_fips` survive as columns but fail the live
    compatibility check on the prepared donor/current condition frames
- status:
  - superseded by the raw-overlap confirmation below
- immediate next step at the time:
  - instrument the block-preparation path itself so we can distinguish a true
    overlap / compatibility failure from an earlier source-capability gate

## 2026-04-14 raw overlap gate confirmation

- run:
  - `artifacts/live_pe_us_data_rebuild_checkpoint_20260414_structured_puf_shared_supplement_diag_smoke/broader-donors-structured-puf-shared-supplement-diagnostic-smoke-v2`
- question:
  - after instrumenting raw overlap, are these supplemental PUF tax-leaf vars
    really failing in block preparation, or are they blocked earlier by source
    capability policy?
- read:
  - across all four problematic PUF tax-leaf blocks, the raw supplemental
    statuses for `employment_status`, `income`, and `state_fips` are all
    `donor_source_disallows_conditioning`
  - the prepared-stage readout remains:
    - `employment_status` -> `incompatible_condition_support`
    - `income` -> `excluded_from_block_shared_overlap`
    - `state_fips` -> `incompatible_condition_support`
  - so the raw overlap never actually admitted those vars into the PUF donor
    conditioning pool in the first place
- alignment read:
  - local `policyengine-us-data` evidence resolves the PE question:
    `policyengine_us_data/calibration/puf_impute.py` trains the PUF clone QRF on
    `DEMOGRAPHIC_PREDICTORS` only, which matches the structural
    `age` / tax-unit-role backbone and does not use `income`,
    `employment_status`, or `state_fips`
- interpretation:
  - the prior supplemental-shared experiment was not just ineffective; it was
    also off the PE-aligned path
  - the PUF source policy is doing the right thing by blocking those derived /
    non-geographic convenience columns as donor conditions
- action:
  - keep the instrumentation and summarizer
  - revert the PUF IRS tax-leaf semantics back to structural-only PE-style
    conditioning
  - treat any future widening as an explicit challenger experiment using
    source-native PUF predictors, not as a PE-alignment patch

## 2026-04-14 PUF native challenger diagnostic smoke

- run:
  - `artifacts/live_pe_us_data_rebuild_checkpoint_20260414_pe_plus_puf_native_challenger_diag_smoke/puf-native-challenger-diag-smoke-v1`
- question:
  - if we add an explicit non-default challenger lane that keeps the PE
    structural backbone but appends a narrow source-native PUF overlap, do
    those vars actually enter the four problematic tax-leaf blocks on a live
    artifact?
- setup:
  - `donor_imputer_condition_selection = pe_plus_puf_native_challenger`
  - keep the PE structural predictors for the PUF IRS tax-leaf family
  - append only explicit source-native challengers:
    - dividend / taxable-interest blocks:
      `self_employment_income`, `rental_income`,
      `social_security_retirement`
    - taxable-pension block:
      `social_security_retirement`, `social_security_disability`,
      `unemployment_compensation`
    - partnership block:
      `self_employment_income`, `rental_income`, `alimony_income`
- read:
  - the challenger vars now enter the live artifact for all four targeted
    blocks
  - selected sets were:
    - dividend split:
      PE structural backbone + `self_employment_income`, `rental_income`,
      `social_security_retirement`
    - `taxable_interest_income`:
      PE structural backbone + `self_employment_income`, `rental_income`,
      `social_security_retirement`
    - `taxable_pension_income`:
      PE structural backbone + `social_security_retirement`,
      `social_security_disability`, `unemployment_compensation`
    - `partnership_s_corp_income`:
      PE structural backbone + `self_employment_income`, `rental_income`
      while `alimony_income` failed with `incompatible_condition_support`
- interpretation:
  - this clears the immediate blocker from the earlier failed supplement patch:
    we now have a real opt-in challenger lane whose native PUF predictors are
    visible in live `donor_conditioning_diagnostics`
  - the next real question is no longer "can the vars get in?" but "does this
    challenger help or hurt the PE-oracle losses once we run a full checkpoint"
- next step:
  - run one matched broader checkpoint with this challenger mode and compare it
    against the structural-only PE-aligned default
