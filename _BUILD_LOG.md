# _BUILD_LOG.md

Append-only notes for agents working in `microplex-us`.

## 2026-04-11

- Corrected upstream EITC-recipient oracle semantics:
  - the active PE targets DB now builds IRS SOI EITC child-count strata with
    `eitc > 0` in addition to `eitc_child_count`
  - Microplex's PE target-provider matching now treats `domain_variable` as a
    set-membership field for target-cell selection, so corrected rows like
    `eitc,eitc_child_count` still match the intended target profile
- Fresh evidence after the EITC-recipient oracle fix:
  - corrected-oracle apples-to-apples reevaluation of the pre-fix large
    no-donor artifact:
    - artifact:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1`
    - corrected capped full-oracle loss `1.0149`
    - corrected full-oracle loss `1.3233`
  - matched large no-donor source rerun against the corrected oracle:
    - artifact:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_eitc_recipient_oracle_large_nodonors/large-nodonors-eitc-recipient-oracle-v2`
    - `4609` calibrated rows
    - capped full-oracle loss `0.9729`
    - full-oracle loss `1.2352`
    - active-solve capped loss `1.2345`
    - `420` active constraints
    - deferred stage still skipped
  - focused deferred-stage confirmations:
    - matched large no-donor source rerun with a forced narrow stage 2:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_age_agi_forced_stage2_large_nodonors/large-nodonors-age-agi-forced-stage2-v1`
      - capped full-oracle loss improves from `0.9729` to `0.9498`
      - active-solve capped loss improves from `1.2345` to `1.1237`
      - stage 2 selects `24` constraints from the top 3 deferred families and
        top 4 deferred geographies
    - matched large donor-inclusive source rerun with the same narrow stage 2:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_age_agi_forced_stage2_large_donors/large-donors-age-agi-forced-stage2-v1`
      - capped full-oracle loss improves from `0.9730` to `0.9502`
      - active-solve capped loss improves from `1.2333` to `1.1238`
      - stage 2 again selects `24` constraints from the same focused set
    - fresh canonical donor-inclusive checkpoint through the default entrypoint:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_default_stage2_large_donors/large-donors-default-stage2-v1`
      - reproduces the same donor-stage result exactly
      - `trigger_threshold` is now `null`
      - stage 2 keeps the same `24` focused constraints and the same
        `0.9502` capped full-oracle loss
    - broader canonical donor-inclusive checkpoint:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_default_stage2_donors/broader-donors-default-stage2-v1`
      - `5000` CPS + `5000` PUF source sample
      - `12092` calibrated rows
      - stage 1 reaches `0.9080` capped full-oracle loss
      - stage 2 still helps, improving to `0.8933`
      - the focused deferred geographies shift to `KY`, `MS`, `WV`, and `DC`
    - matched broader canonical no-donor checkpoint:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_default_stage2_nodonors/broader-nodonors-default-stage2-v1`
      - `5000` CPS + `5000` PUF source sample
      - `12092` calibrated rows
      - stage 1 reaches `0.9056` capped full-oracle loss
      - stage 2 still helps, improving to `0.8909`
      - the focused deferred geographies are `KY`, `MS`, `WV`, and `AZ`
      - donor surveys remain effectively neutral at this broader scale, with a
        slight edge to the no-donor run:
        - donors: `0.8933`
        - no donors: `0.8909`
    - broader no-donor row-level drilldown and selector check:
      - drilldown artifact:
        `artifacts/tmp_broader_nodonor_oracle_drilldown_20260411.json`
      - age and AGI remain the dominant deferred families
      - ACA is the next family down and its worst rows are capped at `10.0`,
        but widening deferred family focus from 3 to 4 does nothing under the
        current `24`-constraint cap
      - matched top-4-family run:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_nodonor_top4family/broader-nodonors-top4family-v1`
      - result is identical to the default broader no-donor run:
        - capped full-oracle loss `0.8909`
        - active-solve capped loss `0.8950`
    - deferred selector switched from family/geography-share-only priority to
      row-level deferred capped error plus family/geography loss share within
      the same focused stage-2 cap
      - focused regression coverage:
        - `python -m py_compile src/microplex_us/pipelines/us.py tests/pipelines/test_us.py`
        - `uv run pytest tests/pipelines/test_us.py -q -k 'prioritizes_target_level_loss or deferred_stage or feasibility_constraint_budget or materialization_failures_audit_only'`
        - `uv run pytest tests/pipelines/test_pe_us_data_rebuild.py tests/pipelines/test_pe_us_data_rebuild_checkpoint.py -q`
      - matched medium no-donor rerun:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_rowrank_nodonors/medium-nodonors-rowrank-v1`
        - unchanged headline result vs the prior medium default:
          `1.0298017982 -> 1.0291445335`
      - matched broader no-donor rerun:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_rowrank_nodonors/broader-nodonors-rowrank-v1`
        - capped full-oracle loss improves from `0.8908588020` to
          `0.8907527501`
        - active-solve capped loss worsens slightly from `0.8950` to `0.9152`,
          but the default objective is full-oracle capped loss
      - matched broader donor-inclusive rerun:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_rowrank_donors/broader-donors-rowrank-v1`
        - capped full-oracle loss improves from `0.8932869027` to
          `0.8782556650`
        - active-solve capped loss improves from `0.8969` to `0.8814`
      - read:
        - the surrounding stage-2 policy was already right; the missed piece was
          which rows got the fixed `24` slots
        - keep the row-aware selector and stop spending time on wider family
          admission experiments for now
    - medium no-donor source rerun with the same narrow stage 2:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_age_agi_forced_stage2_medium_nodonors/medium-nodonors-age-agi-forced-stage2-v1`
      - capped full-oracle loss improves from `1.0298` to `1.0291`
      - active-solve capped loss improves from `0.7356` to `0.7048`
      - stage 2 only finds `7` eligible focused constraints and still helps
    - extra ultra-thin support-1 deferred stage after the row-aware stage 2:
      - matched broader donor-inclusive rerun:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_donors/broader-donors-stage3-v1`
        - capped full-oracle loss improves from `0.8782556650` to
          `0.8212707783`
        - active-solve capped loss improves from `0.8813634527` to
          `0.8343080918`
      - matched broader no-donor rerun:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_nodonors/broader-nodonors-stage3-v1`
        - capped full-oracle loss improves from `0.8907527501` to
          `0.8362042462`
        - active-solve capped loss improves from `0.9151883609` to
          `0.8766713154`
      - matched medium no-donor rerun:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_stage3_nodonors/medium-nodonors-stage3-v1`
        - capped full-oracle loss improves from `1.0291445335` to
          `1.0028694956`
        - active-solve capped loss worsens slightly from `0.7047951546` to
          `0.7148843510`, but the full-oracle objective still improves
      - fresh default-entrypoint medium no-donor confirmation:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_default_stage3_nodonors/medium-nodonors-default-stage3-v1`
        - reproduces the same three-stage result exactly
      - read:
        - the support-1 pass is now doing real work on the residual
          ultra-thin age and AGI cells, not just adding noisy extra constraints
        - promote the default deferred-stage schedule from `(10,)` to `(10, 1)`
    - deferred family focus widened from `3` to `4` after the new stage-3
      residual drilldown showed ACA PTC as the next supported deferred family:
      - added capped-error-mass rankings to the oracle drilldown helper so
        family prioritization is based on loss contribution, not row counts
      - broader donor-inclusive rerun with top-4 deferred families:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_top4family_donors/broader-donors-stage3-top4family-v1`
        - capped full-oracle loss improves from `0.8212707783` to
          `0.7908917500`
      - broader no-donor rerun with top-4 deferred families:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_broader_stage3_top4family_nodonors/broader-nodonors-stage3-top4family-v1`
        - capped full-oracle loss improves from `0.8362042462` to
          `0.7995775732`
      - medium no-donor rerun with top-4 deferred families:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_stage3_top4family_nodonors/medium-nodonors-stage3-top4family-v1`
        - capped full-oracle loss improves from `1.0028694956` to
          `0.9968822972`
      - fresh default-entrypoint medium no-donor confirmation:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_medium_default_top4family_nodonors/medium-nodonors-default-top4family-v1`
        - reproduces the same top-4-family result exactly
      - read:
        - once stage 3 is in place, ACA is no longer a side issue; it is the
          next admitted high-support deferred family
        - promote the default deferred family focus from `3` to `4`
    - fresh broader donor-inclusive default-entrypoint confirmation:
      - `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_default_top4family_donors_rerun/broader-donors-default-top4family-v2`
      - reproduces the existing broader donor default exactly at
        `0.7908917500` capped full-oracle loss
    - rejected wider deferred geography focus:
      - `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_geo8_donors/broader-donors-geo8-v1`
      - widening deferred geographies from `4` to `8` worsens capped
        full-oracle loss from `0.7908917500` to `0.7991939177`
      - read:
        - the current deferred calibration policy is stable on the broader donor
          default path
        - stop widening calibration focus and move upstream to age/AGI structure
    - fresh broader donor drilldown:
      - `artifacts/tmp_broader_default_top4family_donor_drilldown_20260412.json`
      - capped-error mass is still led by `person_count|domain=age`,
        `person_count|domain=adjusted_gross_income`,
        `tax_unit_count|domain=adjusted_gross_income`, and
        `aca_ptc|domain=aca_ptc`
    - state-floor source-sampling prototype:
      - added optional source-side `state_floor` sampling support for CPS and
        donor household samplers
      - matched broader donor rerun with `state_floor=2` on CPS and donor
        sources:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_statefloor2_donors/broader-donors-statefloor2-v1`
      - read:
        - this is a no-op at the current broader `5000/5000` scale; the big
          metric, selected constraints, and deferred geographies are identical
          to the current default artifact
        - the remaining age/AGI problem is therefore not plain state-level
          undercoverage; if we stay upstream, the next sharper idea is
          state-by-age or state-by-AGI support structure rather than a generic
          state floor
    - raw PUF checkpoint sampling should use `S006` weights:
      - fixed `_sample_tax_units()` so checkpoint-scale PUF samples respect raw
        `S006` weights before variable mapping instead of uniformly sampling raw
        PUF rows
      - matched broader donor rerun:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_weight_donors/broader-donors-puf-weight-v1`
        - improves capped full-oracle loss from `0.7908917500` to
          `0.7681656356`
      - matched broader no-donor rerun:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_weight_nodonors/broader-nodonors-puf-weight-v1`
        - improves capped full-oracle loss from `0.7995775732` to
          `0.7683205208`
      - read:
        - this is a direct incumbent-alignment fix, not a challenger modeling
          tweak
        - it improves the big metric more than the recent calibration-planner
          experiments
        - after the fix, age and AGI still dominate capped-error mass, but the
          worst individual cells shift toward ACA PTC and rental/interest tails
    - experiment index:
      - created `artifacts/experiment_index.jsonl`
      - records the intervention artifact, baseline artifact, big metric delta,
        and kept/rejected decision for the recent matched experiments
    - top-3 deferred families is now rejected again under the improved upstream
      source sample:
      - `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_weight_top3family_donors/broader-donors-puf-weight-top3family-v1`
      - regresses capped full-oracle loss from `0.7681656356` to
        `0.8021818710`
      - read:
        - ACA still belongs in the focused deferred family set under the new
          source sample, even though ACA-family loss itself remains ugly
    - CPS `state x age-band` checkpoint floor:
      - added optional `state_age_floor` support to CPS checkpoint sampling and
        promoted `state_age_floor=1` into the default checkpoint query builder
      - matched broader donor rerun:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_donors/broader-donors-cps-stateage1-v1`
        - improves capped full-oracle loss from `0.7681656356` to
          `0.7329149849`
      - matched broader no-donor rerun:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_nodonors/broader-nodonors-cps-stateage1-v1`
        - improves capped full-oracle loss from `0.7683205208` to
          `0.7368409543`
      - stage attribution on the broader donor artifact:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_weight_donors/tmp_broader_puf_weight_donor_stage_attribution_20260412.json`
      - read:
        - `seed` and `synthetic` are identical on the PE oracle for this path,
          so the remaining age/AGI miss is entering before synthesis
        - calibration still reduces age/AGI/EITC substantially, but it worsens
          ACA and rental
        - the state-age floor is the first upstream CPS support tweak that
          materially improves the big metric on both donor and no-donor runs
  - comparative read:
    - this is a real improvement under the corrected oracle, not a stale-manifest
      artifact
    - `tax_unit_count|domain=eitc_child_count` drops out of the top-3 residual
      families after the rerun
    - the remaining leading families are now age counts and AGI count families,
      with leading geographies `OR`, `WI`, and `MI`
- Durable comparison artifact:
  - `artifacts/tmp_eitc_recipient_oracle_large_nodonors_comparison_20260411.json`

- Corrected full-oracle accounting:
  - `full_oracle_*` metrics now include explicit penalty mass for unsupported
    targets instead of silently scoring only the supported subset
  - supported-only summaries remain available as separate diagnostics
- Corrected deferred-stage control flow:
  - a skipped deferred stage no longer aborts later scheduled stages
- Current default PE-oracle rebuild policy:
  - dense first calibration pass
  - one deferred support-10 pass
  - deferred-pass cap `24`
  - deferred pass always considered
  - deferred pass focused to the top 3 deferred families and top 4 deferred
    geographies
  - deferred pass only retained if capped full-oracle loss improves
- Fresh evidence after the correction:
  - medium source checkpoint:
    - artifact:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_medium/medium-source-corrected-oracle-v1`
    - `918` calibrated rows
    - capped full-oracle loss `2.3931`
    - stage 2 skipped under the new `2.45` trigger
  - donor-inclusive source checkpoint:
    - artifact:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_donors/donors-source-corrected-oracle-v1`
    - `918` calibrated rows
    - capped full-oracle loss `2.3940`
    - active-solve capped loss `2.0969`
    - stage 2 also skipped under the new `2.45` trigger
  - larger donor-inclusive source checkpoint:
    - artifact:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_large_donors/large-donors-source-corrected-oracle-v1`
    - source mix:
      `cps_asec_2023 + irs_soi_puf_2024 + acs_2022 + sipp_tips_2023 + sipp_assets_2023 + scf_2022`
    - `4859` calibrated rows
    - `490` active constraints after the feasibility filter
    - capped full-oracle loss `2.4331`
    - active-solve capped loss `2.7178`
    - deferred stage still skipped under the new `2.45` trigger
  - matched larger no-donor source checkpoint:
    - artifact:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_corrected_oracle_source_large_nodonors/large-nodonors-source-corrected-oracle-v1`
    - source mix:
      `cps_asec_2023 + irs_soi_puf_2024`
    - `4859` calibrated rows
    - `487` active constraints after the feasibility filter
    - capped full-oracle loss `2.4329`
    - active-solve capped loss `2.7284`
    - deferred stage also skipped under the new `2.45` trigger
- larger replayed saved artifacts:
  - `4859` rows: capped full-oracle loss `0.6803`, stage 2 skipped
  - `24686` rows: capped full-oracle loss `1.9845`, stage 2 skipped
- Current interpretation:
  - the corrected metric still preserves the useful tiny-run stage-2 gain
  - at medium and above, the deferred pass should usually not fire under the
    current incumbent-compatible default
  - the fresh `4859`-row donor-inclusive source build lands very close to the
    trigger, so `2.45` now looks like a real boundary value rather than a loose
    conservative skip rule
  - at this `2000/2000` source scale, donor surveys are basically neutral on
    corrected full-oracle loss:
    - donors: `2.4331`
    - no donors: `2.4329`
    - donors slightly improve active-solve loss but do not improve the
      full-oracle score
  - follow-up compiler diagnosis:
    - the dominant remaining full-oracle families were not actually calibration
      misses; they were `tax_unit_count` targets with person-entity domain
      filters such as `dividend_income > 0` and `tax_unit_is_filer == 1`
    - PE defines those domain variables on `person`, while the old compiler only
      supported cross-entity filters for household targets
    - extending the compiler to align `person -> tax_unit/family/spm_unit`
      boolean filters removes that structural unsupported wall
  - replay after the compiler fix on the saved `4859`-row large source
    artifacts:
    - supported targets move from `4070` to `4642`
    - unsupported targets drop from `572` to `0`
    - capped full-oracle replay loss falls from about `2.43` to about `1.33`
    - donor vs no-donor remains effectively neutral on the replayed full-oracle
      metric:
      - donors: `1.3267`
      - no donors: `1.3264`
  - fresh large no-donor source rerun after the compiler fix:
    - artifact:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1`
    - active constraints rise from `487` to `540`
    - supported targets rise from `487` to `540` within the solve
    - unsupported targets drop from `572` to `0` on the full oracle
    - capped full-oracle loss falls from `2.4329` to `1.3274`
    - active-solve capped loss improves slightly from `2.7284` to `2.6923`
    - deferred stage still skips, now because the trigger metric is
      `1.3274 < 2.45`
  - fresh large donor-inclusive source rerun after the compiler fix:
    - artifact:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_donors/large-donors-cross-entity-fix-v1`
    - capped full-oracle loss lands at `1.3277`
    - active-solve capped loss lands at `2.6825`
    - unsupported targets remain `0`
    - donor inclusion is still basically neutral on the broad oracle at this scale
  - added saved-artifact oracle summaries:
    - recurring family/geography summary:
      `artifacts/tmp_policyengine_oracle_regressions_cross_entity_fix_20260411.json`
    - exact worst-cell drilldown:
      `artifacts/tmp_policyengine_oracle_target_drilldown_cross_entity_fix_20260411.json`
  - residual reading after the compiler fix:
    - the largest remaining full-oracle families are now
      `person_count|domain=age`,
      `tax_unit_count|domain=eitc_child_count`,
      `person_count|domain=adjusted_gross_income`,
      `tax_unit_count|domain=adjusted_gross_income`,
      `tax_unit_count|domain=salt`, and `aca_ptc|domain=aca_ptc`
    - the leading geographies are `state:OR`, `state:GA`, and `state:MO`
    - concrete worst cells inside those geographies include:
      - `tax_exempt_interest_income` in `OR`
      - AGI count targets in `OR` and `MO`
      - ACA PTC in `OR`, `GA`, and `MO`
      - EITC child-count and SALT targets in `GA`
      - pass-through income in `MO`
  - next work should target those residual families/geographies directly, not
    more deferred-stage threshold tuning
  - controlled smoke A/B on stored-input tails:
    - accepted interest/rental conditioning change:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_current/smoke-nodonors-asset-tail-conditioning-current-v1`
    - matched old-semantics baseline:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_oldsemantics/smoke-nodonors-asset-tail-old-semantics-v1`
    - rejected property-cost extension:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_v2/smoke-nodonors-asset-tail-conditioning-v2`
    - outcome:
      - the accepted change is a small honest win on the smoke A/B:
        capped full-oracle loss improves from `1.4417803` to `1.4414441`
      - active-solve capped loss also improves from `1.8878380` to `1.8829362`
      - the capped stored-input mass attributed to
        `tax_exempt_interest_income` in the top drilldown falls from `40` to `20`
      - extending the same pattern to property-tax variables was worse and was
        reverted: capped full-oracle loss rose to `1.4489770`
  - tested a separate interest-family decomposition path and rejected it:
    - medium no-donor candidate:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_interest_family_medium_nodonors/medium-nodonors-interest-family-v1`
    - matched large no-donor confirmation:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_interest_family_large_nodonors/large-nodonors-interest-family-v1`
    - matched large no-donor baseline:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1`
    - reading:
      - the idea looked good at medium scale
      - it does not hold at `2000/2000`
      - capped full-oracle loss worsens from `1.3274` to `1.3555`
      - raw full-oracle loss worsens from `2256.6` to `16980.7`
      - active-solve capped loss worsens from `2.6923` to `2.8229`
      - reverted the code change; default path stays on separate
        `taxable_interest_income` and `tax_exempt_interest_income`
  - tested donor-support sampling without replacement and rejected it:
    - rejected smoke artifact:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_donor_support_sampling_smoke_nodonors/smoke-nodonors-donor-support-sampling-v1`
    - baseline smoke artifact:
      `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_asset_tail_conditioning_smoke_nodonors_current/smoke-nodonors-asset-tail-conditioning-current-v1`
    - reading:
      - capped full-oracle loss worsens from `1.4414` to `1.6369`
      - active-solve capped loss worsens from `1.8829` to `2.7402`
      - keep donor-support sampling with replacement
  - rejected rental export normalization from donor-integrated components:
    - the saved large no-donor seed already carries
      `rental_income_positive` and `rental_income_negative`
    - replaying that saved seed with export-side normalization looked promising:
      - capped full-oracle loss improves from `1.3274` to `1.3169`
      - active-solve capped loss improves from `2.6923` to `2.6877`
    - but the fresh `2000/2000` large no-donor source checkpoint failed:
      - baseline:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1`
      - candidate:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_rental_export_large_nodonors/large-nodonors-rental-export-v1`
      - capped full-oracle loss worsens from `1.3274` to `1.3874`
      - active-solve capped loss worsens from `2.6923` to `2.7722`
      - active constraints fall from `540` to `522`
    - verdict: do not keep this change in the default path; source checkpoints
      override replay-only wins
  - rejected direct zero-support-mask propagation in zero-inflated donor rank
    matching:
    - idea:
      - the QRF path already trains a zero model for zero-inflated positives
      - let final donor rank matching use the generated `scores > 0` support mask
        instead of donor positive-rate counts
    - rationale:
      - this looked like a clean way to stop final rank matching from
        reintroducing positive tail support after the zero model had already
        predicted zeros
    - but the fresh `2000/2000` large no-donor source checkpoint failed:
      - baseline:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_cross_entity_fix_large_nodonors/large-nodonors-cross-entity-fix-v1`
      - candidate:
        `artifacts/live_pe_us_data_rebuild_checkpoint_20260411_zero_support_mask_large_nodonors/large-nodonors-zero-support-mask-v1`
      - capped full-oracle loss worsens from `1.3274` to `1.9223`
      - active-solve capped loss worsens from `2.6923` to `4.3296`
      - active constraints rise from `540` to `703`
    - verdict: reject; do not replace donor-rate positive counts with the
      generated zero mask in the default path

## 2026-03-28

- The US country pack now consumes more shared core benchmark infrastructure.
- `benchmark_metrics` in `src/microplex_us/policyengine/comparison.py` now delegates to shared `normalize_metric_payload(...)` instead of hand-building `TargetMetric`.
- `src/microplex_us/policyengine/harness.py` now builds suites from shared result-oriented core helpers rather than local payload plumbing.
- `src/microplex_us/pipelines/local_reweighting.py` remains the thin adapter over core reweighting bundles and solver.
- `_materialize_policyengine_us_variables_one_by_one(...)` in `src/microplex_us/policyengine/us.py` was fixed to chain successful materialized outputs forward, so dependency chains work in fallback mode.
- US-specific legacy targets DB implementation now lives here instead of core:
  - `src/microplex_us/targets_database.py`
- `src/microplex_us/pipelines/experiments.py` now has a first-class `n_synthetic` sweep helper:
  - `build_us_n_synthetic_sweep_experiments(...)`
  - `run_us_microplex_n_synthetic_sweep(...)`
- The performance-session experiment path now respects experiment-level `n_synthetic` and `random_seed` overrides instead of silently using the outer harness defaults.
- The corrected parity benchmark showed the real local gap is `state_programs_core`, not district slices.
- Current diagnosis:
  - `cps_puf_500_auto_conditions_support_match` loses on state Medicaid and SNAP targets.
  - the larger saved `cps_5000_puf_500_nsynthetic_5000_state_stratified_bootstrap` artifact is not a healthy counterexample; its calibrated weights collapse to near-zero mass, so it should not be used as evidence that scaling fixed the state gap.
- Worst current `state_programs_core` misses for `cps_puf_500_auto_conditions_support_match` are concentrated in a small set of zero/near-zero states:
  - Medicaid: GA (`state_fips=13`), WV (`54`), AZ (`4`), OR (`41`), VT (`50`), TX (`48`), AK (`2`), RI (`44`)
  - SNAP: IA (`19`), OR (`41`), NH (`33`), WI (`55`)
  - candidate zeros by source: Medicaid `10/51`, SNAP `8/51`
- The saved `500_best` and `5000_state_stratified` artifacts are not comparable on scaffold richness:
  - `500_best` seed carries `has_medicaid`, `public_assistance`, and `ssi`
  - `5000_state_stratified` seed does not
- `src/microplex_us/pipelines/us.py` now prefers scaffold sources that carry state-program support proxies (`has_medicaid`, `public_assistance`, `ssi`, `social_security`) before falling back to raw observed-column count.
- `synthesis_metadata` now records `state_program_support_proxies.available/missing` so artifact triage can see whether a run ever had Medicaid/SNAP support proxies in the scaffolded seed.
- `src/microplex_us/pipelines/us.py` now records explicit household/person weight diagnostics in `calibration_summary`, including effective sample size, tiny-weight share, and a `weight_collapse_suspected` flag so broken calibration runs are obvious in saved manifests.
- `src/microplex_us/pipelines/registry.py` now carries `calibration_converged` and `weight_collapse_suspected`, and frontier selection ignores runs flagged as weight-collapsed.
- A direct CPS scaffold A/B on `state_programs_core` confirms scaffold richness matters at fixed `n_synthetic=500`:
  - stripped parquet CPS scaffold (`cps_asec_parquet`): candidate MARE `1.1675`, composite parity loss `1.0630`
  - rich cached CPS scaffold (`cps_asec_2023`): candidate MARE `0.7861`, composite parity loss `0.7257`
  - both compared against the same PE baseline (`0.4682` MARE, `0.4530` composite)
- The rich cached CPS scaffold is materially better specifically because it carries `has_medicaid`, `public_assistance`, `ssi`, and `social_security`. This is now a confirmed causal lever, not just a suspicion from artifact comparison.
- The next empirical question is whether that scaffold gain survives once PUF is added back in and `n_synthetic` is increased beyond `500`.
- PE-US bridge fix landed after that A/B:
  - `src/microplex_us/policyengine/us.py` now exports `ssi` into temporary PE datasets when available.
  - `src/microplex_us/pipelines/us.py` no longer lets fallback `employment_income_before_lsr` absorb `ssi` or `public_assistance` when explicit wages are missing.
- Interpretation: older state-program benchmark runs understate what a rich CPS scaffold can do, because they were dropping a program-relevant PE input (`ssi`) at the export boundary.
- Direct-override policy alignment:
  - do not model around `*_reported` variables here
  - PE rules should remain canonical by default; direct program overrides should be explicit, not automatic
  - `src/microplex_us/policyengine/us.py` now supports explicit direct-override variable names in `build_policyengine_us_export_variable_maps(...)`, so callers can intentionally short-circuit with values like `snap` or `ssi` when they mean to
- Slack context for that policy lives in:
  - `#us-snap` thread on PR `policyengine-us#7858` removing `snap_reported`
  - `#mfb-policy-engine` thread stating that callers should pass direct values like `snap`/`tanf` when they want to short-circuit, rather than rely on `*_reported`
- Tonight's post-diagnosis empirical check on `state_programs_core`:
  - current rich CPS-only run (`n_synthetic=500`, default PE rules): candidate MARE `0.9530`, baseline MARE `0.4682`, candidate composite `0.8616`, baseline composite `0.4530`
  - explicit `candidate_direct_override_variables=('ssi',)` made no observable difference on that slice
  - mixed rich CPS + PUF runs are better than current CPS-only:
    - `n_synthetic=500`: candidate MARE `0.8198`, composite `0.7495`
    - `n_synthetic=2000`: candidate MARE `0.7808`, composite `0.7129`
  - but both still lose clearly to the PE baseline on `state_programs_core`
- Interpretation:
  - richer scaffold and more rows help
  - explicit `ssi` short-circuiting is not the lever
  - the remaining gap still looks like real state-program support / structure, not a simple PE-bridge switch
- Canonical artifact discipline tightened:
  - `src/microplex_us/pipelines/site_snapshot.py` now builds a site-facing snapshot directly from one saved artifact bundle (`manifest.json` + `policyengine_harness.json`).
  - Canonical website input now lives at `artifacts/site_snapshot_us.json`, not in `tmp_*.json` diagnostics.
  - New blessed version-bump benchmark command:
    - `uv run microplex-us-version-bump-benchmark --output-root ... --cps-parquet-dir ... --targets-db ... --baseline-dataset ...`
  - The command can also refresh the canonical site snapshot with `--site-snapshot-path /Users/maxghenis/PolicyEngine/microplex-us/artifacts/site_snapshot_us.json`.
- Enforcement direction:
  - scratch diagnostics can still exist, but the website should only read the canonical snapshot file
  - versioned benchmark runs should emit manifest + harness + registry entry, then optionally refresh the canonical snapshot

## Current review bar

- Prefer pushing reusable benchmark/evaluation abstractions into `microplex`.
- PE-US materialization changes need focused regression coverage.
- Be skeptical of any benchmark delta that does not clearly state whether it is common-target or full-set based.

## Known remaining risks

- `src/microplex_us/policyengine/us.py` is still a large concentration of concerns.
- Composite-loss reporting and generic suite MARE are both present; do not conflate them.
- Future tax-unit endogeneity work will likely force another boundary review with core.

## 2026-03-29

- US artifact persistence and site snapshot generation now validate saved bundles against the shared core manifest contract before using them.
- The shared contract is intentionally structural:
  - top-level manifest keys
  - required benchmark summary keys for harness-backed bundles
  - referenced artifact files must exist
- This means the website snapshot path now fails fast on incomplete saved bundles instead of quietly reading partial manifests.
- Canonical version-bump benchmarking now refreshes the site snapshot by default.
  - `uv run microplex-us-version-bump-benchmark ...` writes to `artifacts/site_snapshot_us.json` unless `--site-snapshot-path` overrides it.
- Added deterministic snapshot freshness check:
  - `uv run microplex-us-check-site-snapshot artifacts/site_snapshot_us.json`
- Added GitHub Actions workflow:
  - `.github/workflows/site-snapshot.yml`
- CI design is intentionally narrow:
  - checkout `microplex-us` plus sibling core `microplex`
  - run focused snapshot/version-benchmark tests
  - regenerate the canonical snapshot from its source artifact and fail if the committed JSON differs

## 2026-03-29 state-program follow-up

- US `state_programs_core` diagnosis tightened:
  - the remaining gap is concentrated in repeated low-mass states across both Medicaid and SNAP, not just one program family
  - on the `n=2000` diagnostic slice, candidate MARE is still materially worse than baseline:
    - overall `0.8252` vs `0.4682`
    - Medicaid `0.8766` vs `0.3098`
    - SNAP `0.7738` vs `0.6265`
  - current failure mode is severe under-support, not unsupported targets:
    - `supported_target_rate = 1.0`
    - `candidate_zero_count = 0` for both domains in the focused diagnostics
    - worst states are often at `~0.1%` to `~3%` of target mass
- The pipeline now preserves state-program support proxies through synthesis by default instead of only carrying them implicitly in richer multi-source target sets:
  - `src/microplex_us/pipelines/us.py` now auto-promotes available `has_medicaid`, `public_assistance`, `ssi`, and `social_security` columns into `condition_vars`
  - this applies to the normal single-source CPS path as well as multi-source runs
  - focused regression coverage now pins both paths in `tests/pipelines/test_us.py`
- The PE-US parity suite semantics were corrected for the state SNAP leg:
  - `src/microplex_us/policyengine/harness.py` now uses `household_count` with domain `snap` in `state_programs_core`
  - this matches the slice description (`recipiency`) and aligns with the district SNAP slice instead of treating state SNAP as a dollar-total benchmark
  - focused regression coverage now pins the slice filters in `tests/policyengine/test_harness.py`
- Current interpretation:
  - household-weight-only calibration is not failing to compile these targets
  - the bigger ceiling is synthetic support expressiveness and source coverage
  - real CPS/PUF source coverage is still structurally thin for this problem:
    - real CPS carries proxies like `has_medicaid`, `public_assistance`, `ssi`, `social_security`
    - real CPS/PUF does not provide real `snap` values for donor integration
    - Medicaid still enters as proxy support rather than a native target-aligned source variable
- Likely next move:
  - rerun the corrected comparable state slice after the proxy-preservation fix
  - then decide whether the next investment is:
    - stronger source/backbone support for program participation, or
    - a richer non-household weight entity path for US local calibration
- Focused rerun on the saved `n=2000` candidate with the corrected `state_programs_core` semantics:
  - candidate MARE `0.8492`
  - PE baseline MARE `0.7298`
  - delta `+0.1194` (PE still better)
  - candidate composite parity loss `0.7754`
  - PE baseline composite parity loss `0.7408`
  - supported targets `102` for both
  - target win rate `29.41%`
- Interpretation of that rerun:
  - the old state SNAP amount/count mismatch was materially inflating the apparent local gap
  - correcting the slice semantics narrows the loss substantially
  - but it does not remove the underlying state-program weakness
  - next reruns should use the corrected count-based state SNAP slice as canonical
- Fresh real-source rerun after the proxy-preserving synthesis change:
  - output saved at `artifacts/tmp_state_programs_corrected_rerun_20260329.json`
  - source mix: `cps_asec_2023 + irs_soi_puf_2024`
  - sample size: `500` source households / tax units
  - corrected state slice only
  - results:
    - `n_synthetic=500`: candidate MARE `0.9619`, baseline MARE `0.7298`, delta `+0.2321`, candidate composite `0.8678`
    - `n_synthetic=2000`: candidate MARE `0.8729`, baseline MARE `0.7298`, delta `+0.1432`, candidate composite `0.7925`
  - both runs preserved the proxies in synthesis `condition_vars`:
    - `age`, `sex`, `education`, `employment_status`, `state_fips`, `tenure`, `has_medicaid`, `public_assistance`, `ssi`, `social_security`
  - both runs were healthy enough numerically:
    - no weight collapse
    - all `102` corrected state targets supported
- Interpretation of the fresh rerun:
  - preserving the CPS state-program proxies through synthesis is not enough to beat PE on the corrected state slice
  - scaling from `500` to `2000` still helps, but only modestly
  - the remaining gap now looks even more like a structural source/backbone problem than a lost-proxy problem
  - specifically:
    - real CPS/PUF still lacks true SNAP donor support
    - Medicaid still enters mostly as proxy support rather than a target-native source variable
    - household-weight-only calibration can rescale what exists, but cannot create the missing state-program structure

2026-03-29
- Scope reviewed:
  - US `state_programs_core` after focused Claude review
  - DB calibration feasibility vs solver non-convergence
  - proxy semantics and synthesizer-path safety
- What changed:
  - DB calibration now applies a feasibility filter before solving:
    - config supports `policyengine_calibration_max_constraints`
    - config supports `policyengine_calibration_max_constraints_per_household`
    - config supports `policyengine_calibration_min_active_households`
  - calibration summaries now record:
    - `n_constraints_before_feasibility_filter`
    - `n_constraints_after_feasibility_filter`
    - low-support / over-capacity drops
  - weight diagnostics now flag low effective-sample-ratio collapse, not just tiny-weight share
  - registered semantic specs for:
    - `has_medicaid`
    - `public_assistance`
    - `ssi`
    - `social_security`
  - fixed a core synthesizer bug where zero-inflated variables with all-zero training support could crash on inverse transform during sampling
- New canonical bootstrap rerun with the feasibility filter:
  - output saved at `artifacts/tmp_state_programs_feasible_bootstrap_rerun_20260329.json`
  - exact calibration DB: `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/calibration/policy_data.db`
  - corrected state-only calibration + benchmark scope:
    - variables: `household_count`, `person_count`
    - domains: `snap`, `medicaid_enrolled`
    - geo level: `state`
  - results:
    - `n_synthetic=500`
      - candidate MARE `0.9232`
      - PE baseline MARE `0.7386`
      - delta `+0.1846`
      - candidate composite `0.8358`
      - PE composite `0.7704`
      - target win rate `33.33%`
      - feasibility filter reduced constraints `102 -> 81`
    - `n_synthetic=2000`
      - candidate MARE `0.7335`
      - PE baseline MARE `0.7386`
      - delta `-0.0051`
      - candidate composite `0.6770`
      - PE composite `0.7704`
      - target win rate `37.25%`
      - feasibility filter reduced constraints `102 -> 100`
- Interpretation:
  - the Claude review was directionally right that calibration feasibility mattered more than the earlier “backbone only” diagnosis
  - once the state-program solve stops trying to absorb an infeasible flat constraint set, the `n=2000` CPS+PUF bootstrap run slightly beats PE on the corrected state slice
  - this does not prove the final production architecture is solved, but it does show the immediate local gap was not just a source-support story
  - remaining open issues:
    - synthesizer-backed state-program reruns still need a clean end-to-end pass
    - proxy preservation alone is not the main lever; feasible calibration is
- Follow-up synthesizer unblock:
  - fixed core zero-inflated inverse-transform handling when a target has all-zero training support
  - fixed `ensure_target_support()` to coerce boolean exemplar values before writing back into numeric synthetic columns
  - added a real synthesizer-path regression with the promoted state-program proxy condition vars
  - synthesizer rerun output saved at `artifacts/tmp_state_programs_feasible_synth_rerun_20260329.json`
  - results:
    - `n_synthetic=500`
      - candidate MARE `0.8918`
      - PE baseline MARE `0.7386`
      - delta `+0.1533`
      - candidate composite `0.8143`
      - PE composite `0.7704`
      - target win rate `29.41%`
    - `n_synthetic=2000`
      - candidate MARE `0.6811`
      - PE baseline MARE `0.7386`
      - delta `-0.0574`
      - candidate composite `0.6481`
      - PE composite `0.7704`
      - target win rate `42.16%`
- Updated interpretation:
  - feasible calibration was the main missing lever
  - once the solve is narrowed to the corrected state-program target estate, both bootstrap and synthesizer improve sharply
  - the synthesizer path now also clears PE at `n=2000`, and by a healthier margin than bootstrap
 - the remaining US state-program work should now focus on:
    - stabilizing this feasible-target calibration path
    - deciding whether to keep the default cap at `1.0 * household_count` or tune it lower
    - then broadening back out carefully instead of returning to the flat 3,611-constraint solve

2026-03-29 — focused code review (Claude agent team)
- Scope: state-program accuracy work across microplex-us and microplex core
- Top findings:
  1. **Critical**: all saved artifacts show `converged: false` — headline n=2000 results are on unconverged weights. The "win" vs PE is narrow and not reliable.
  2. **High**: `min_active_households=1` lets degenerate single-household constraints through. Raise to 5-10.
  3. **High**: `has_medicaid` uses `BOUNDED_SHARE` but is binary — should be `ZERO_INFLATED_POSITIVE`.
  4. **High**: `ensure_target_support()` bool fix is correct but only guarantees 1 exemplar per category — not enough for calibration.
  5. **Medium**: zero project-level tests in microplex-us; zero direct unit tests for core transform fix.
- Diagnosis assessment: calibration infeasibility was a real blocker, but the deeper root cause is sparse small-state sample coverage (n=2000 across 51 states). Feasibility filtering delays the reckoning but doesn't resolve it.
- Benchmark assessment: corrected state-only path is valid as a diagnostic slice but should not replace the full canonical benchmark. Results are directionally encouraging but not credible until calibration converges.
- Top 3 next fixes:
  1. Add small-state oversampling floor (min 10 households/state) to bootstrap/synthesis
  2. Raise `min_active_households` to 5-10, warn when >20% constraints dropped
  3. Write regression tests for feasibility filter, ensure_target_support, condition var promotion, harness slice stability

2026-03-29
- Review handoff workflow:
  - durable pending Claude review request now lives at `reviews/PENDING_CLAUDE_REVIEW.md`
  - full Claude reviews should be written under `reviews/`
  - `_BUILD_LOG.md` should keep only concise review summaries
  - intended short Claude instruction is now just:
    - `Please execute the pending review request in /Users/maxghenis/PolicyEngine/microplex-us/reviews/PENDING_CLAUDE_REVIEW.md`

2026-03-29
- Follow-up after focused review findings:
  - tightened calibration feasibility defaults:
    - `policyengine_calibration_min_active_households` now defaults to `5`
    - feasibility diagnostics now record total dropped constraints, drop share, and warning messages
    - calibration summaries now surface warnings for heavy feasibility dropping and non-convergence
  - adjusted proxy handling:
    - `has_medicaid` now uses `ZERO_INFLATED_POSITIVE` semantics
    - only `has_medicaid` is auto-promoted into synthesis condition vars by default
    - `public_assistance`, `ssi`, and `social_security` now remain synthesis targets instead of inflating the condition space
  - core transform fallback now warns when a zero-inflated variable has no positive training support
- Focused verification:
  - `microplex-us` focused pipeline tests: `13 passed`
  - `microplex-us` variable semantics tests: `13 passed`
  - `microplex` synthesizer tests: `17 passed`
  - Ruff clean on touched files
- Updated corrected state-only reruns with stricter defaults:
  - bootstrap artifact: `artifacts/tmp_state_programs_feasible_bootstrap_rerun_20260329.json`
    - `n=2000`: candidate MARE `0.8094`, PE MARE `0.7386`
    - `n=2000`: candidate composite `0.7408`, PE composite `0.7704`
    - `n=2000`: `converged=false`, feasibility filter dropped `25/102` constraints (`24.5%`)
    - interpretation: bootstrap no longer beats PE under the stricter floor
  - synthesizer artifact: `artifacts/tmp_state_programs_feasible_synth_rerun_20260329.json`
    - `n=2000`: candidate MARE `0.6910`, PE MARE `0.7386`
    - `n=2000`: candidate composite `0.6537`, PE composite `0.7704`
    - `n=2000`: `converged=false`, feasibility filter dropped `3/102` constraints (`2.9%`)
    - interpretation: synthesizer still edges PE on the corrected state slice, but the solve is still unconverged, so this remains directional evidence rather than a settled win

2026-03-29
- PE-native mission-metric setup:
  - `microplex-us` now has a real broad PE-native scorer in `src/microplex_us/pipelines/pe_native_scores.py`
  - saved artifacts can persist `policyengine_native_scores.json` plus a `policyengine_native_scores` summary block in `manifest.json`
  - `run_registry.jsonl` now understands:
    - `candidate_enhanced_cps_native_loss`
    - `baseline_enhanced_cps_native_loss`
    - `enhanced_cps_native_loss_delta`
    - unweighted MSRE companions
  - canonical US version-bump flow now requires native scoring and ranks on `candidate_enhanced_cps_native_loss`
- Important boundary:
  - the exact broad `enhanced_cps` native loss is now the primary PE mission metric
  - PE local validation does not expose one single final scalar; the correct follow-up is a `validate_staging.py` wrapper plus saved `validation_results.csv` / summary JSON, not a fake “local PE loss”
- Focused verification:
  - `tests/pipelines/test_pe_native_scores.py`
  - `tests/pipelines/test_version_benchmark.py`
  - `tests/pipelines/test_artifacts.py`
  - `tests/pipelines/test_registry.py`
  - result: `13 passed`
  - Ruff clean on scorer/artifact/registry/version-benchmark files

2026-03-29
- PE-native mission loop tightened:
  - canonical saved US version-bump flow now ranks frontier runs on `enhanced_cps_native_loss_delta`, not absolute candidate native loss
  - saved native-score summaries now include an explicit `candidate_beats_baseline` flag
  - `run_registry.jsonl` carries that boolean as `candidate_beats_baseline_native_loss`
  - saved artifacts append to the registry even when only PE-native scoring is available and harness scoring is absent
  - `microplex-us-version-benchmark` now supports `--require-beat-pe-native-loss` to fail fast when a run still loses on PE's own broad native loss
- Focused verification:
  - `tests/pipelines/test_pe_native_scores.py`
  - `tests/pipelines/test_version_benchmark.py`
  - `tests/pipelines/test_registry.py -k "native_loss_frontier_selection or append_and_load_us_microplex_run_registry"`
  - `tests/pipelines/test_artifacts.py -k "policyengine_native_scores_when_available"`
  - Ruff clean on the touched scorer/artifact/registry/version-benchmark files

2026-03-29
- Historical PE-native backfill support:
  - added `src/microplex_us/pipelines/backfill_pe_native_scores.py`
  - new CLI: `microplex-us-backfill-pe-native-scores`
  - backfill upgrades old bundles by writing `policyengine_native_scores.json`, updating `manifest.json`, and rebuilding `run_registry.jsonl` / `run_index.duckdb` for that artifact root
- Focused verification:
  - `tests/pipelines/test_backfill_pe_native_scores.py`
  - `tests/pipelines/test_pe_native_scores.py`
  - `tests/pipelines/test_version_benchmark.py`
  - `tests/pipelines/test_artifacts.py -k "policyengine_native_scores_when_available"`
  - `tests/pipelines/test_registry.py -k "native_loss_frontier_selection or append_and_load_us_microplex_run_registry"`
  - Ruff clean on the touched backfill/scorer/artifact/registry/version-benchmark files
- Important mission finding:
  - backfilled `/artifacts/live_cps_puf_three_fixes_20260326/20260326T131756Z-4eaab451`
  - despite beating PE on its own narrow saved harness (`candidate MARE 0.1737` vs baseline `0.1881`), it is catastrophic on PE's true broad native loss:
    - candidate native loss `27.8382`
    - PE baseline native loss `0.01748`
    - delta `+27.8207`
  - implication: the mission is not “go back to the older narrow tax-target config”; current broad/native-aligned candidates are much closer to PE even when they still lose

2026-03-29
- PE-native target-estate and local mission-loop wiring:
  - added named exact-cell target profile support in `src/microplex_us/policyengine/target_profiles.py`
  - added first mission profile: `pe_native_broad`
  - provider now accepts exact `target_cells` filters through `TargetQuery.provider_filters`
  - `USMicroplexBuildConfig` and local performance configs now carry `policyengine_target_profile` / `policyengine_calibration_target_profile`
  - canonical `microplex-us-version-benchmark` now defaults both target-profile flags to `pe_native_broad`
  - local performance harness can now optionally export the candidate and score PE-native broad loss directly via `evaluate_pe_native_loss=True`
- Important finding:
  - for the current production target DB, `pe_native_broad` is exactly the active `national+state` surface:
    - all geos: `37,755`
    - national+state: `4,183`
    - `pe_native_broad` profile: `4,183`
  - so the value of the profile today is not a smaller target estate; it is making the mission surface explicit and future-stable, while excluding district/local drift from the canonical version-bump path
- Focused verification:
  - targeted provider/pipeline/profile/version-benchmark/performance tests: `22 passed`
  - `tests/pipelines/test_performance.py`: `13 passed`
  - Ruff clean on touched target-profile/provider/pipeline/performance/version-benchmark files

2026-03-29
- Mission-loop throughput fix:
  - `run_us_microplex_performance_harness()` was already computing PE-native scores, but `save_us_microplex_artifacts()` ignored them and recomputed the full PE-native scorer again while writing the bundle
  - added `precomputed_policyengine_harness_payload` / `precomputed_policyengine_native_scores` passthrough support to artifact saving
  - `run_us_microplex_source_experiments()` now forwards `performance_result.parity_run.to_dict()` and `performance_result.pe_native_scores` into the artifact saver
  - implication: future sweeps stop paying the PE-native scorer twice per candidate
- PE-native broad target mix (from current scorer outputs + `policyengine-us-data` calibration targets):
  - kept targets: `2,853`
  - split: `677 national` / `2,176 state`
  - state-heavy families are the real mission surface:
    - age by state: `900`
    - AGI bins by state: `918`
    - SNAP state cost/households: `102`
    - ACA spending/enrollment: `102`
    - Medicaid enrollment: `51`
    - real estate taxes by state: `51`
    - state population: `51`
  - implication: beating PE on the broad native loss requires state age/AGI structure, not just fixing SNAP/Medicaid
- Focused verification:
  - `tests/pipelines/test_artifacts.py -k "precomputed_policyengine_native_scores or writes_policyengine_native_scores_when_available"`: `2 passed`
  - `tests/pipelines/test_experiments.py -k "performance_session"`: `1 passed`
  - Ruff clean on touched artifact/experiment files

2026-03-29
- Performance-harness scope fix for PE-native broad runs:
  - found a real mission-loop bug: `USMicroplexPerformanceHarnessConfig` had hardcoded default target filters for five national tax variables, and those defaults were still applied even when `target_profile='pe_native_broad'`
  - effect: the first live `cps+puf-rich` "broad" run under `/artifacts/live_pe_native_cps_puf_rich_sweep_20260329` was not actually broad; it calibrated only 5 national targets and produced a misleading PE-native score (`candidate native loss 1.1437` vs baseline `0.02024`)
  - fixed `src/microplex_us/pipelines/performance.py` so named target profiles can own the scope unless the caller explicitly overrides variables/domains/geo levels
  - parity/cache paths now read the resolved build scope, not stale config defaults
  - relaunched the true broad mission run at `/artifacts/live_pe_native_cps_puf_rich_broad_fixed_20260329`
- Focused verification:
  - `tests/pipelines/test_performance.py -k "preserves_target_profiles or warm_us_microplex_parity_cache"`: `3 passed`
  - Ruff clean on touched performance/test files

2026-03-29
- Corrected broad PE-native result (`cps+puf-rich`, `sample_n=500`, `n_synthetic=2000`):
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_native_cps_puf_rich_broad_fixed_20260329/20260329T175330Z-057066af`
  - the scope is now correct: `policyengine_target_profile='pe_native_broad'` with no extra variable/geo filters
  - PE-native broad loss is still far from PE:
    - candidate native loss `0.95856`
    - PE baseline native loss `0.02024`
    - delta `+0.93832`
    - kept targets `2,817` (`641 national`, `2,176 state`)
  - calibration remains the dominant failure mode on the broad mission surface:
    - `converged=false`
    - `1,413` supported constraints out of `4,183` loaded targets
    - feasibility filter dropped `2,198 / 3,611` candidate constraints (`60.9%`)
    - mean error `0.9234`
  - implication: the PE-native mission is still primarily a scale/support problem; fixing the profile bug was necessary, but not enough
- Next live run:
  - launched a larger broad mission candidate at `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_native_cps_puf_rich_broad_scaled_20260329`
  - config: `sample_n=5000`, `n_synthetic=10000`, `target_profile='pe_native_broad'`, native loss only

2026-03-29
- PE-native scorer instrumentation:
  - `src/microplex_us/pipelines/pe_native_scores.py` now supports `family_breakdown` in both single-candidate and batch native-loss scoring
  - current family classifier covers the broad PE-native estate at the level we care about operationally:
    - `state_age_distribution`
    - `state_agi_distribution`
    - `state_snap_cost`
    - `state_snap_households`
    - `state_medicaid_enrollment`
    - `state_aca_spending`
    - `state_aca_enrollment`
    - `state_population`
    - `state_population_under_5`
    - `state_real_estate_taxes`
    - plus national census / IRS / JCT / SSA / net-worth families
  - goal: stop treating PE-native broad loss as one opaque scalar and identify which families dominate the mission gap
- Focused verification:
  - `tests/pipelines/test_pe_native_scores.py`: `3 passed`
  - Ruff clean on touched native-score files

2026-03-29
- Wired sparse/L0-style calibration into the actual PE-backed DB solve path:
  - `src/microplex/calibration.py` now lets `SparseCalibrator` and `HardConcreteCalibrator` accept explicit `LinearConstraint` rows and report `linear_errors` / `converged` in the same shape as the classical calibrator
  - `src/microplex_us/pipelines/us.py` now builds calibrators through one shared backend factory, so `policyengine_targets_db` calibration can use `sparse` and `hardconcrete` instead of hard-rejecting everything except `entropy/ipf/chi2`
  - added focused regressions in:
    - `microplex/tests/test_sparse_calibrator.py`
    - `microplex/tests/test_sparse_calibration_comparison.py`
    - `microplex-us/tests/pipelines/test_us.py`
- Focused verification:
  - `microplex/tests/test_sparse_calibrator.py`, `microplex/tests/test_sparse_calibration_comparison.py`, `microplex/tests/test_calibration.py`: `48 passed`
  - `microplex-us/tests/pipelines/test_us.py -k calibrate_policyengine_tables_from_db`: `4 passed`
  - Ruff clean on touched core + US files
- Mission follow-up:
  - attempted a broad sparse-vs-entropy sweep at `sample_n=5000`, `n_synthetic=10000`, but the first broad PE-native score alone was slow enough that it is not a practical overnight tuning loop yet
  - replaced it with a smaller first broad sparse diagnostic at `sample_n=1000`, `n_synthetic=2000`, `target_sparsity=0.1`; result pending in `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pe_native_broad_sparse_n2000_20260329.json`

2026-03-29
- First broad sparse PE-native diagnostic landed:
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pe_native_broad_sparse_n2000_20260329.json`
  - result is much worse than entropy on the mission surface:
    - candidate native loss `633.9884`
    - PE baseline native loss `0.0202`
    - delta `+633.9681`
  - calibration summary:
    - backend `policyengine_db_sparse`
    - supported constraints `1,314 / 4,183`
    - feasibility filter dropped `2,297 / 3,611` candidate constraints (`63.6%`)
  - the dominant family blowups are not just Medicaid/SNAP:
    - `state_agi_distribution`
    - `state_age_distribution`
    - `state_aca_spending`
    - `state_aca_enrollment`
    - `state_medicaid_enrollment`
  - implication: the current sparse/L0-style solve path is not ready for the broad PE-native mission loop; it is a diagnostic branch, not a candidate frontier path
- Throughput fix for future mission sweeps:
  - `src/microplex_us/pipelines/artifacts.py` now supports deferring native scoring when saving a batch of experiment bundles
  - `src/microplex_us/pipelines/backfill_pe_native_scores.py` now has grouped batch backfill via `compute_batch_us_pe_native_scores(...)`
  - `src/microplex_us/pipelines/experiments.py` now saves multi-experiment performance batches first, batch-scores native loss once per baseline, rebuilds the registry, and refreshes experiment results/frontier entries from the rebuilt registry
  - goal: stop paying the fixed PE-native baseline/scorer cost candidate-by-candidate in experiment sweeps
- Focused verification:
  - `tests/pipelines/test_experiments.py`, `tests/pipelines/test_backfill_pe_native_scores.py`: `10 passed`
  - Ruff clean on touched artifact/backfill/experiment files

2026-03-29
- Native-only experiment throughput fix:
  - the first batched `pe_native_broad` source/synthesis compare showed that `save_us_microplex_artifacts(...)` was still generating full `policyengine_harness.json` sidecars even when the performance run had `evaluate_parity=False`
  - that was wasted work for the PE-native mission loop and produced huge harness files (`~100MB`) before native batch scoring even started
  - fixed by threading `defer_policyengine_harness` through:
    - `src/microplex_us/pipelines/artifacts.py`
    - `src/microplex_us/pipelines/experiments.py`
  - performance-session experiment batches now skip harness generation when there is no precomputed parity payload, while still deferring native scoring and backfilling it in batch
- Focused verification:
  - `tests/pipelines/test_experiments.py::test_run_us_microplex_source_experiments_can_use_performance_session`
  - `tests/pipelines/test_artifacts.py::TestSaveUSMicroplexArtifacts::test_can_defer_policyengine_harness_generation`
  - Ruff clean on touched artifact/experiment files
 - Current live run:
   - relaunched the four-way PE-native broad compare on the no-harness path at `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_native_broad_entropy_batch_noharness_20260329`
   - matrix:
     - `cps-only-bootstrap`
     - `cps-only-synthesizer`
     - `cps-puf-bootstrap`
     - `cps-puf-synthesizer`
   - shared config:
     - `sample_n=1000`
     - `n_synthetic=2000`
     - `calibration_backend='entropy'`
     - `target_profile='pe_native_broad'`

2026-03-29
- First live donor-imputer A/B on the real PE-native broad mission path:
  - added explicit donor-imputer backend switching in `src/microplex_us/pipelines/us.py`
    - runtime now supports `donor_imputer_backend='maf' | 'qrf' | 'zi_qrf'`
    - `qrf` / `zi_qrf` use a new columnwise forest-based donor imputer rather than the existing flow-based `Synthesizer`
  - added focused route coverage in `tests/pipelines/test_us.py`
- Smoke-test result on `cps_asec_2023 + puf_2024`, `sample_n=500`, `n_synthetic=2000`, `target_profile='pe_native_broad'`, `calibration_backend='entropy'`:
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_donor_backend_ab_pe_native_broad_20260329.json`
  - `maf`:
    - candidate native loss `0.8958`
    - baseline native loss `0.02024`
    - delta `+0.8755`
    - calibration `converged=false`
    - supported constraints `1,391`
    - feasibility filter dropped `2,220 / 3,611` constraints (`61.5%`)
  - `zi_qrf`:
    - candidate native loss `0.9278`
    - baseline native loss `0.02024`
    - delta `+0.9076`
    - calibration `converged=false`
    - supported constraints `1,459`
    - feasibility filter dropped `2,152 / 3,611` constraints (`59.6%`)
- Immediate read:
  - the widened imputation eval winner (`zi_qrf`) did not improve total PE-native broad loss on the live runtime path; it made the smoke-test result slightly worse than `maf`
  - translation caveat is likely real: the runtime donor-imputed variables on this path are mostly PUF tax variables (`capital_gains`, `dividends`, `interest`, `pension`, etc.), not the broader survey-support surfaces emphasized by the widened eval
  - next control is plain `qrf` on the same path to see whether the miss is the zero-inflated gate or the whole forest donor-imputer branch
- Plain `qrf` control on the same config:
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_donor_backend_qrf_pe_native_broad_20260329.json`
  - candidate native loss `0.8931`
  - baseline native loss `0.02024`
  - delta `+0.8728`
  - calibration `converged=false`
  - supported constraints `1,398`
  - feasibility filter dropped `2,213 / 3,611` constraints (`61.3%`)
- Current runtime read:
  - `qrf` is slightly better than `maf` on PE-native broad total loss in this smoke test (`0.8931` vs `0.8958`)
  - `zi_qrf` is worse than both (`0.9278`)
  - none of these are remotely close to PE yet, so this is only a runtime-direction result, not a candidate-frontier change
- QRF control on the same live path:
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_donor_backend_qrf_pe_native_broad_20260329.json`
  - `qrf`:
    - candidate native loss `0.8931`
    - baseline native loss `0.02024`
    - delta `+0.8728`
    - calibration `converged=false`
    - supported constraints `1,398`
    - feasibility filter dropped `2,213 / 3,611` constraints (`61.3%`)
- Updated read:
  - on the current live PE-native broad smoke test, plain `qrf` slightly beat the existing `maf` runtime donor path, while `zi_qrf` was worse
  - ordering on this path was `qrf` (`0.8931`) better than `maf` (`0.8958`) better than `zi_qrf` (`0.9278`)
  - the `qrf` vs `maf` gap is tiny and all three runs remain `converged=false`, so this is not enough to justify a production switch
  - the widened eval is still useful, but it should not directly drive the PE-native production switch without a closer mission-surface benchmark

2026-03-29
- Broad PE-native family diagnosis:
  - the huge broad-loss gap is not primarily a donor-imputer issue
  - in `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_native_broad_entropy_batch_noharness_20260329/20260329T210427Z-057066af/policyengine_native_scores.json`, the top loss contributors are:
    - `national_irs_other` `+0.2839`
    - `state_agi_distribution` `+0.1893`
    - `state_age_distribution` `+0.1860`
    - `national_population_by_age` `+0.0605`
    - `national_census_other` `+0.0445`
    - `state_aca_spending` `+0.0333`
  - donor-imputer choice only moves total broad loss by about `0.035` end-to-end (`0.8931` to `0.9278`), while the gap to PE is still about `0.87`
  - current live donor-imputation only affects a 31-variable PUF tax block, so most of the broad native-loss delta is coming from seams outside the donor-imputer switch
- Failed bootstrap-target-scope experiments:
  - tried auto-inferencing profile-driven bootstrap strata from `pe_native_broad`
  - full profile strata (`state_fips`, `age_group`, `income_bracket`) made broad loss worse:
    - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_profile_strata_pe_native_broad_20260329.json`
    - candidate native loss `0.9371`
    - delta `+0.9169`
  - narrower state-only profile strata also made broad loss worse:
    - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_state_strata_pe_native_broad_20260329.json`
    - candidate native loss `0.9373`
    - delta `+0.9170`
 - conclusion: bootstrap stratification is not the missing broad-native lever here; the attempted default inference was reverted after the smoke tests

2026-03-29
- Broad PE-native structural export diagnosis:
  - found a real upstream household-structure bug on the broad path: saved calibrated rows still carried healthy `family_relationship`, but `relationship_to_head` had already collapsed to mostly `{0,3}`, and `build_policyengine_entity_tables()` was preserving that bad column
  - first fix: when `family_relationship` is richer than `relationship_to_head`, prefer it during PE-entity construction
  - second fix: repair incoherent household relationship patterns before tax-unit construction so each household has exactly one head and at most one spouse
  - before the repair on the saved broad artifact (`20260329T210427Z-057066af`):
    - `4774` tax units for `4774` people
    - filing status all `SINGLE`
    - `1170 / 2000` households had no head at all
  - after the repair on the same saved artifact:
    - `4650` tax units for `4774` people
    - filing status distribution `{'SINGLE': 4529, 'JOINT': 119, 'HEAD_OF_HOUSEHOLD': 2}`
    - `0 / 2000` households with no head
    - `0 / 2000` households with multiple heads
  - quick PE probe on the repaired `cps+puf` broad export:
    - `income_tax_sum` moved from `105.41B` to `104.01B`
    - `tax_unit_is_filer_sum` moved from `4.889M` to `4.793M`
    - raw IRS person-income sums like `qualified_dividend_income`, `taxable_interest_income`, and `taxable_pension_income` were unchanged, so this fix primarily affects filing/tax-unit structure rather than person-level donor values
- Broad donor/entity semantics diagnosis:
  - several IRS donor-integrated inputs in `variables.py` were still marked tax-unit-native even though current `policyengine_us` defines them as person variables
  - patched the confirmed person-native set:
    - `dividend_income`
    - `ordinary_dividend_income`
    - `qualified_dividend_income`
    - `non_qualified_dividend_income`
    - `taxable_interest_income`
    - `tax_exempt_interest_income`
    - `taxable_pension_income`
    - `taxable_social_security`
    - `self_employment_income`
    - `student_loan_interest`
  - also moved `DIVIDEND_DONOR_BLOCK_SPEC` to `native_entity=PERSON`
  - this stops the donor path from projecting those inputs onto tax units with default `FIRST`
- Verification:
  - focused relationship tests in `tests/pipelines/test_us.py`: passed (`4`)
  - focused variable-semantics tests in `tests/test_variables.py`: passed (`4`)
  - Ruff clean on touched files
- Next step:
  - clean PE-native broad rescoring is still running on the repaired `cps+puf` export to quantify how much the broad loss actually moves from these two structural fixes

2026-03-29
- Broad PE-native rescore on repaired `cps+puf` export:
  - persisted repaired export:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_cps_puf_broad_relationship_entity_fix_20260329.h5`
  - direct PE-native broad scoring under `policyengine-us-data` showed:
    - candidate loss `0.9386384097643049`
    - same kept-target surface as before (`2817` = `641` national + `2176` state)
  - comparison to the saved pre-fix `cps+puf` broad artifact (`20260329T210540Z-057066af`):
    - pre-fix candidate loss `0.9369853544124408`
    - post-fix candidate loss `0.9386384097643049`
    - change `+0.0016530553518641` (slightly worse)
 - interpretation:
    - the relationship/head repair and confirmed person-native IRS semantic fixes corrected real structural bugs
    - but on this saved `cps+puf` broad candidate they did not improve the mission metric
    - broad PE-native loss is still dominated by seams outside this export-structure fix, especially the already-identified `national_irs_other`, `state_agi_distribution`, and `state_age_distribution` families

2026-03-29
- PE pre-sim parity audit against `source_imputed_stratified_extended_cps_2024.h5`:
  - added reusable audit helper:
    - `src/microplex_us/pipelines/pre_sim_parity.py`
    - `tests/pipelines/test_pre_sim_parity.py`
  - real audit artifact written to:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pre_sim_parity_audit_20260329.json`
  - saved broad candidate audited:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_native_broad_entropy_batch_noharness_20260329/20260329T210427Z-057066af/policyengine_us.h5`
  - key findings:
    - candidate schema recall vs PE pre-sim input surface is only `35 / 165 = 21.2%`
    - missing critical pre-sim inputs include:
      - `county_fips`
      - `cps_race`
      - `is_hispanic`
      - `is_disabled`
      - `rent`
      - `real_estate_taxes`
      - `net_worth`
      - `has_esi`
      - `has_marketplace_health_coverage`
    - candidate tax-unit structure is still pathological pre-sim:
      - `share_multi_person_tax_units = 0.0`
      - reference `share_multi_person_tax_units = 0.446`
    - candidate state-by-age pre-sim support recall is only `0.627`
      - `576 / 918` nonempty `(state, 5-year-age-bin)` cells
      - worst missing states by cell count include DC (`11`), WY (`56`), SD (`46`), VT (`50`)
    - several mission-relevant IRS donor inputs have zero positive support in the candidate while PE pre-sim has real mass, notably:
      - `long_term_capital_gains_before_response`
      - `partnership_s_corp_income`
      - `farm_income`
  - interpretation:
    - the broad PE-native gap is not just calibration
    - we are feeding PE a far thinner and structurally weaker pre-sim dataset than PE-US-data feeds itself
 - next step:
    - build a parity-focused fix list around missing pre-sim inputs and tax-unit structure before spending more cycles on donor-backend A/B tests

2026-03-29
- PE pre-sim parity follow-up:
  - re-exported the saved broad candidate under current code to isolate export/handoff vs upstream candidate quality:
    - candidate source tables:
      - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_native_broad_entropy_batch_noharness_20260329/20260329T210427Z-057066af/calibrated_data.parquet`
    - re-exported H5:
      - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pre_sim_parity_reexport_20260329.h5`
    - updated audit:
      - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pre_sim_parity_reexport_20260329.json`
  - compared with the original saved candidate H5 audit:
    - common PE pre-sim vars improved from `35` to `39`
    - schema recall improved from `0.2121` to `0.2364`
    - recovered exactly these PE inputs in the H5 handoff:
      - `cps_race`
      - `is_hispanic`
      - `rent`
      - `real_estate_taxes`
    - missing critical vars dropped from:
      - `county_fips`, `cps_race`, `is_hispanic`, `is_disabled`, `rent`, `real_estate_taxes`, `net_worth`, `has_esi`, `has_marketplace_health_coverage`
      - to:
      - `county_fips`, `is_disabled`, `net_worth`, `has_esi`, `has_marketplace_health_coverage`
    - candidate tax-unit structure improved slightly under current entity-table/export code:
      - `share_multi_person_tax_units` from `0.0` to `0.0260`
  - interpretation:
    - the export bridge was a real part of the problem, but not the dominant one
    - after current-code re-export, the remaining broad gap is clearly upstream of H5 writing
- CPS pre-sim source-surface restoration:
  - updated `src/microplex_us/data_sources/cps.py` so raw CPS loads now carry the same core CPS-derived pre-sim inputs that `policyengine-us-data` uses:
    - `county_fips` from household `GTCO`
    - `cps_race` from `PRDTRACE`
    - `is_hispanic` from `PRDTHSP != 0`
    - `is_disabled` from the CPS disability flags (`PEDISDRS`, `PEDISEAR`, `PEDISEYE`, `PEDISOUT`, `PEDISPHY`, `PEDISREM`)
    - `has_esi` from `NOW_GRP == 1`
    - `has_marketplace_health_coverage` from `NOW_MRK == 1`
  - also tightened processed-cache freshness so stale cached CPS parquet will rebuild if those PE-style pre-sim columns are missing
  - verified in `tests/test_cps_source_provider.py` (`6 passed`, Ruff clean)
  - this is aimed at future broad reruns; it does not retroactively change the already-saved broad artifact

2026-03-29
- Fresh current-code parity audit correction:
  - the earlier `tmp_pre_sim_parity_reexport_20260329.h5/json` pair turned out to be stale for entity-structure conclusions
  - rebuilt a fresh current-code export directly from the saved broad `calibrated_data.parquet`:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_tax_unit_recheck_20260329.h5`
    - audit:
      - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pre_sim_parity_reexport_fresh_20260329.json`
  - corrected fresh-audit findings:
    - schema overlap is unchanged from the re-export check:
      - `39 / 165` common PE pre-sim vars
      - schema recall `0.2364`
      - missing critical vars remain:
        - `county_fips`
        - `is_disabled`
        - `net_worth`
        - `has_esi`
        - `has_marketplace_health_coverage`
    - but entity structure is substantially healthier than the stale re-export audit implied:
      - `tax_unit_rows = 2807`
      - mean tax-unit size `1.7007`
      - `share_multi_person_tax_units = 0.3997`
      - `share_multi_person_households = 0.687`
    - state-age support recall is still only `0.627`
 - interpretation:
    - current code no longer appears to be collapsing tax-unit membership at the PE export boundary
    - the remaining pre-sim parity gap is now more clearly about:
      - missing CPS-derived inputs that are not yet present upstream (`county_fips`, `is_disabled`, `has_esi`, `has_marketplace_health_coverage`)
      - missing wealth input (`net_worth`)
      - thin `(state, age)` support before calibration

2026-03-29
- CPS pre-sim parity smoke test on the real broad mission metric:
  - ran a fresh CPS-only broad PE-native smoke build with the updated raw CPS loader and real PE targets DB:
    - provider: `CPSASECSourceProvider(year=2023)`
    - calibration DB: `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/calibration/policy_data.db`
    - PE baseline: `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5`
    - config: `sample_n=500`, `n_synthetic=2000`, `target_profile='pe_native_broad'`, `calibration_target_profile='pe_native_broad'`, `evaluate_pe_native_loss=True`
  - result:
    - candidate broad PE-native loss `0.9058149122381814`
    - PE baseline `0.020243908529428433`
    - delta `+0.885571003708753`
    - calibration still `converged=false`
    - feasibility filter still dropped `2506 / 3611` constraints (`69.4%`)
  - comparison to the earlier CPS-only broad bootstrap frontier run:
    - earlier saved candidate loss `0.9233365911702252`
    - improvement from restored CPS pre-sim inputs `-0.0175216789320438`
 - interpretation:
    - restoring PE-style CPS pre-sim inputs is directionally correct and measurably improves the real mission metric
    - but it is not remotely sufficient on its own; the remaining broad gap is still dominated by other structural issues

2026-03-29
- PE export + relationship parity corrections:
  - updated `src/microplex_us/policyengine/us.py` so the PE export whitelist now includes pre-sim inputs we already carry upstream:
    - `cps_race`
    - `is_hispanic`
    - `is_disabled`
    - `rent`
    - `real_estate_taxes`
    - `has_esi`
    - `has_marketplace_health_coverage`
    - `net_worth`
  - added a narrow export alias only for `race -> cps_race`; dropped the lossy raw `hispanic -> is_hispanic` rename
  - updated `src/microplex_us/pipelines/us.py` so PE-oriented person-input augmentation now derives exact PE-native columns before export:
    - `cps_race` from `race`
    - `is_hispanic` from CPS-coded `hispanic`
  - fixed `family_relationship` normalization to handle the common CPS 1-based coding per household:
    - `1=head`, `2=spouse`, `3=child`, `4=other`
    - this was the real reason rebuilt tax units had been collapsing toward singletons on many CPS-shaped households
  - fixed `prepare_seed_data_from_source()` to preserve household `county_fips` instead of dropping it during the household-person merge
  - focused verification:
    - `tests/test_cps_source_provider.py`: `6 passed`
    - `tests/pipelines/test_pre_sim_parity.py`: `1 passed`
    - `tests/pipelines/test_us.py -k 'prepare_seed_data or build_policyengine_entity_tables or derives_tax_input_columns'`: `10 passed`
    - `tests/policyengine/test_us.py -k 'export_variable_maps or projects_frame'`: `5 passed`
    - Ruff clean on touched CPS / pipeline / PE-export files
- fresh current-code re-export from the saved broad candidate:
  - candidate H5:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pre_sim_parity_export_fix_candidate_20260329.h5`
  - parity audit:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pre_sim_parity_export_fix_audit_20260329.json`
  - native score:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pre_sim_parity_export_fix_native_score_20260329.json`
  - key results:
    - schema recall remains `0.2364` (`39 / 165`)
    - missing critical vars are now:
      - `county_fips`
      - `is_disabled`
      - `net_worth`
      - `has_esi`
      - `has_marketplace_health_coverage`
    - candidate tax-unit structure is now materially healthier under current code:
      - `tax_unit_rows = 2807`
      - mean tax-unit size `1.7007`
      - `share_multi_person_tax_units = 0.3997`
    - broad PE-native loss on the repaired re-export is:
      - candidate `0.9339483631287737`
      - PE baseline `0.020243908529428433`
      - delta `+0.9137044545993452`
 - interpretation:
    - the PE handoff really was broken in specific ways, and the repaired handoff is more faithful now
    - but even a substantially healthier export/tax-unit structure only buys a small broad-loss improvement on the saved candidate
    - the dominant remaining gap is still upstream of export, especially:
      - missing pre-sim input surfaces
      - thin state-age support
      - weak IRS / AGI cell mass before calibration

2026-03-29
- current-code CPS-only broad PE-native drilldown:
  - built and exported the exact current-code CPS-only candidate H5:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_cps_only_currentcode_candidate_20260329.h5`
  - broad smoke result:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_cps_only_currentcode_pe_native_broad_20260329.json`
    - candidate broad PE-native loss `0.9159877997083388`
    - PE baseline `0.020243908529428433`
    - delta `+0.8957438911789103`
  - exact worst targets:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pe_native_broad_worst_targets_currentcode_cps_20260329.json`
  - pre-sim surface compare against PE's source-imputed CPS:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pre_sim_surface_compare_currentcode_cps_20260329.json`
  - state-mass compare:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_state_mass_compare_currentcode_cps_20260329.json`
- main findings:
  - `state_age_distribution` is a real large driver, not a scorer artifact:
    - current-code candidate has only `434` nonempty `(state, 5-year-age-bin)` cells vs `911` in PE's source-imputed CPS
    - many large exact cells are literally zero, e.g.:
      - `state/census/age/PA/20-24`: candidate `0.0` vs target `798,935`
      - `state/census/age/FL/40-44`: candidate `0.0000275` vs target `1,434,863`
      - `state/census/age/TX/15-19`: candidate `0.0000626` vs target `2,198,388`
  - `national_irs_other` is being driven by literal zeroed IRS surfaces:
    - candidate has `0.0` on high-value exact targets where PE baseline is near-target, e.g.:
      - `nation/irs/total pension income/total/AGI in 20k-25k/taxable/All`
      - `nation/irs/qualified dividends/total/AGI in -inf-inf/taxable/All`
      - `nation/irs/partnership and s corp income/total/AGI in 75k-100k/taxable/All`
      - `nation/irs/adjusted gross income/total/AGI in 500k-1m/taxable/Single`
      - `nation/irs/capital gains gross/total/AGI in 30k-40k/taxable/All`
    - pre-sim IRS surface compare confirms the upstream mass problem:
      - candidate weighted positive-share is `0.0` for `capital_gains_gross`
      - candidate weighted positive-share is `0.0` for `partnership_and_s_corp_income`
      - candidate weighted positive-share is `0.0` for `total_pension_income`
      - candidate has no tax-unit mass above `$1m` AGI, while PE reference has weighted share `0.0597`
  - `state_agi_distribution` is a mix of state-mass collapse and AGI-tail distortion:
    - worst exact misses include:
      - `state/MD/adjusted_gross_income/count/-inf_1`: candidate `127,417` vs target `40,530`
      - `state/MS/adjusted_gross_income/count/500000_inf`: candidate `23,033` vs target `8,170`
      - many state amount cells are still exactly zero, e.g.:
        - `state/WY/adjusted_gross_income/amount/100000_200000`
        - `state/WV/adjusted_gross_income/amount/500000_inf`
        - `state/DC/adjusted_gross_income/amount/75000_100000`
  - weighted state mass itself is heavily distorted before calibration:
    - candidate state share ratios vs PE reference are effectively zero in some states:
      - TN (`~6.1e-10`)
      - SD (`~6.4e-10`)
      - NV (`~9.5e-10`)
    - large states are also badly underweighted:
      - TX share ratio `0.0929`
      - FL `0.3971`
    - while some states are materially overweighted:
      - VA `3.06`
      - MA `2.36`
      - GA `2.30`
- interpretation:
  - the dominant broad-loss problem is now clearly upstream population/state allocation and missing IRS surface mass before calibration
  - PE-native scorer correctness looks much less suspicious than candidate structure/support
  - the next high-leverage fixes are:
    - restore missing IRS/tax-unit mass (`capital_gains_gross`, `partnership_and_s_corp_income`, `total_pension_income`, high-AGI filers)
    - repair state allocation before calibration
    - then revisit ACA/coverage surfaces, which also show extreme exact misses (`nation/irs/aca_spending/hi`, `state/irs/aca_enrollment/hi`)

## 2026-03-29 weighted-source sampling checkpoint

- current-code donor path diagnosis:
  - the critical PUF IRS variables are *not* disappearing in the live `cps+puf` build anymore
  - a direct mini-build trace shows `qualified_dividend_income`, `long_term_capital_gains`, `partnership_s_corp_income`, `total_pension_income`, `taxable_pension_income`, and `taxable_interest_income` all survive:
    - raw PUF frame
    - donor integration into `seed_data`
    - bootstrap `synthetic_data`
    - `calibrated_data`
  - that means the old zero-surface failure was a saved-artifact issue, not the current-code seam
- source loader fix:
  - `CPSASEC` and `PUF` `sample_n` subsampling now use weight-aware sampling without replacement when there are enough positive-weight rows
  - this is now covered by focused provider regressions for both CPS and PUF
- mission-surface effect:
  - patched `cps+puf + qrf + bootstrap` broad PE-native smoke:
    - candidate loss `0.8894089161`
    - PE baseline `0.0202439085`
    - delta `+0.8691650076`
  - prior comparable `qrf` smoke was `0.8930645879`
  - so the weighted-source patch improved broad loss by about `0.00366`
- remaining constraints:
  - the same patched candidate still drops `2387 / 3611` calibration constraints (`66.1%`)
  - a patched `cps+puf` pre-sim audit still only reaches `453 / 918` nonempty `(state, age-bin)` cells, support recall `0.493`
- interpretation:
  - weight-aware source sampling is a real but small win
  - it is not enough to close the broad-loss gap
  - the remaining bottleneck is still structural state support / state allocation plus unconverged broad calibration, not donor-variable passage

## 2026-03-29 weighted-source scale checkpoint

- broad PE-native result on weighted `cps+puf + qrf + bootstrap` with `sample_n=1000`, `n_synthetic=2000`:
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_weighted_sample1000_pe_native_broad_20260329.json`
  - candidate loss `0.8696287975`
  - PE baseline `0.0202439085`
  - delta `+0.8493848890`
- this improves the weighted `sample_n=500` comparable run (`0.8894089161`) by about `0.01978`
- calibration also got a little healthier:
  - dropped constraints improved from `2387 / 3611` to `2301 / 3611`
  - feasibility-drop share improved from `66.1%` to `63.7%`
- family improvements are concentrated exactly where we need them:
  - `state_age_distribution`: `-0.00579` loss-contribution delta improvement
  - `state_agi_distribution`: `-0.00579`
  - `national_irs_other`: `-0.00438`
  - `national_population_by_age`: `-0.00158`
- pre-sim support also improved materially at this scale:
  - `sample_n=500`: state-age support recall `0.464`, nonempty cells `426`
  - `sample_n=1000`: state-age support recall `0.598`, nonempty cells `549`
- interpretation:
  - scaling the source sample is a much stronger lever than the small weighted-subsampling patch alone
  - the next main-line bet should stay on this axis: weighted-source path + larger `sample_n`
  - state-stratified bootstrap still looks like the wrong direction at this sample size

## 2026-03-29 broad-loop reversal checkpoint

- weighted `cps+puf + qrf + bootstrap` with `sample_n=1000`, `n_synthetic=5000` regressed materially:
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_weighted_sample1000_n5000_pe_native_broad_20260329.json`
  - candidate loss `0.8907772820` vs the stronger `sample_n=1000`, `n_synthetic=2000` result `0.8696287975`
  - calibration feasibility looked *broader* but fit quality got worse:
    - dropped constraints improved to `1807 / 3611` (`50.0%`)
    - but `weight_collapse_suspected = true`
    - household effective sample ratio collapsed to `0.165`
    - median household weight collapsed to `~1.37e-08`
- family-level regression from `1000/2000` to `1000/5000` is narrow, not broad-based:
  - `national_irs_other`: `+0.01510`
  - `state_agi_distribution`: `+0.00899`
  - `state_aca_spending`: `+0.00133`
  - meanwhile `state_age_distribution` *improved* slightly (`-0.00293`)
- exact target regressions confirm the failure mode is filer/tax/ACA structure, not generic state-age support:
  - huge regressions in:
    - high-AGI IRS bins (`1m+`, `500k-1m`)
    - Head of Household bins
    - business/capital-gains/taxable-interest cells
    - state ACA spending cells
    - a few extreme state high-AGI cells like `state/VT/adjusted_gross_income/amount/500000_inf`
- interpretation:
  - more synthetic rows from the same support base destabilize broad PE-native fit
  - this is not a monotone “more `n_synthetic` is better” regime
  - for broad PE-native loss, the current bottleneck is tax/filer structure stability plus calibration interaction

## 2026-03-29 source-mix and donor-path checkpoint

- weighted `cps+puf + qrf + bootstrap` with `sample_n=2000`, `n_synthetic=2000` was worse, not better:
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_weighted_sample2000_pe_native_broad_20260329.json`
  - candidate loss `0.9251676593`
  - supported constraints `1280` vs `1310` on the better `1000/2000` run
  - household calibrated weight total `6.24M` vs `10.37M` on the better `1000/2000` run
  - mean constraint error `0.879` vs `0.795`
- the raw weighted CPS source sample is not the obvious culprit:
  - `sample_n=1000`: weight sum `4.37M`, `50` states
  - `sample_n=2000`: weight sum `8.70M`, all `51` states
- the raw PUF source is effectively national-only in this path, which is expected:
  - `state_count = 1` on the sampled PUF household table
- donor-condition audit for the PUF path on the current best `cps+puf` run:
  - scaffold: `cps_asec_2023`
  - selected donor condition vars are only:
    - `age`
    - `interest_income`
    - `rental_income`
    - `self_employment_income`
    - `sex`
    - `social_security`
    - `unemployment_compensation`
  - importantly, `state_fips` is *not* entering the PUF donor match
- `cps-only` isolation at the same `sample_n=2000`, `n_synthetic=2000` size:
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_cps_only_sample2000_pe_native_broad_20260329.json`
  - candidate loss `0.8846092807`
  - this is much better than `cps+puf` at the same size (`0.9251676593`)
  - but still worse than the best current broad run (`cps+puf`, `1000/2000`, `0.8696287975`)
- pre-sim parity at `sample_n=2000`, `n_synthetic=2000` also points the same way:
  - `cps+puf`: state-age support recall `0.6100`, multi-person tax-unit share `0.3885`
  - `cps-only`: state-age support recall `0.6296`, multi-person tax-unit share `0.4090`
- interpretation:
  - the current PUF donor path is harming the broad PE-native mission surface at `sample_n=2000`
  - the harm is not coming from `state_fips` being used in donor matching
  - the sharper hypothesis is that donor-imputing tax/filer surfaces like `filing_status_code` from only a weak seven-variable numeric condition set is destabilizing `national_irs_other` and related ACA/high-AGI families

## 2026-03-29 diagnostics tooling checkpoint

- added reusable PE-native target-delta comparison helper in `src/microplex_us/pipelines/pe_native_scores.py`
  - purpose: compare exact target-level weighted-loss deltas between two candidate H5s without ad hoc one-off scripts
  - exported via `src/microplex_us/pipelines/__init__.py`
  - covered in `tests/pipelines/test_pe_native_scores.py`
- focused verification:
  - `pytest -q tests/pipelines/test_pe_native_scores.py` -> `4 passed`
  - `ruff check` on the touched scorer/export/test files -> clean

## In flight

- direct ablation still running:
  - `cps+puf`, weighted `qrf + bootstrap`, `sample_n=1000`, `n_synthetic=2000`
  - but skip donor integration of `filing_status_code`
  - output target: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_no_filing_status_pe_native_broad_20260329.json`
- this is the cleanest immediate test of the current filer-structure hypothesis.

## 2026-03-29 filing-status donor checkpoint

- the `filing_status_code` ablation landed and improved the real mission metric:
  - baseline broad run: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_weighted_sample1000_pe_native_broad_20260329.json`
    - candidate loss `0.8696287975`
  - no-filing ablation: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_no_filing_status_pe_native_broad_20260329.json`
    - candidate loss `0.8596198236`
  - improvement: `-0.010009`
- the gains are concentrated in the same broad families we already care about:
  - `national_irs_other`: `-0.00358`
  - `state_aca_spending`: `-0.00281`
  - `state_agi_distribution`: `-0.00136`
  - `national_population_by_age`: `-0.00091`
- pre-sim parity did **not** improve on state-age support:
  - best broad run: state-age support recall `0.5980`
  - no-filing ablation: `0.5643`
  - interpretation: this is a tax/filer-structure win, not a generic coverage win
- exported tax-unit structure changed modestly in the healthier direction:
  - best broad run:
    - `filing_status` shares `SINGLE 59.6%`, `JOINT 35.1%`, `HOH 5.3%`
    - mean tax-unit size `1.7266`
    - multi-person tax-unit share `0.4038`
  - no-filing ablation:
    - `filing_status` shares `SINGLE 58.0%`, `JOINT 37.8%`, `HOH 4.2%`
    - mean tax-unit size `1.7432`
    - multi-person tax-unit share `0.4199`
- raw PUF confirms why the donor path is risky here:
  - `filing_status_code` exists only in PUF, not in the CPS scaffold seed
  - raw sampled PUF distribution is strongly categorical and skewed:
    - `JOINT 1112`, `SINGLE 316`, `HOH 103`, `SEPARATE 25`
  - current donor logic was treating `filing_status_code` as a generic continuous donor target under weak shared numeric conditions
- code change:
  - `src/microplex_us/pipelines/us.py` now supports `donor_imputer_excluded_variables`
  - exclusion remains opt-in; do **not** make `filing_status_code` the default exclusion until the result is reproducible
  - `synthesis_metadata` now records `donor_excluded_variables`
  - focused test added in `tests/pipelines/test_us.py`
- next likely tax/filer ablation candidates, if broad loss plateaus here:
  - `eitc_children`
  - `exemptions_count`
  - possibly other PUF-only count/categorical surfaces before touching zero-inflated amount variables

## 2026-03-29 filing-status reproducibility warning

- the supported-path rerun of the same broad `qrf + bootstrap` idea with opt-in exclusion
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_excluded_filing_status_config_pe_native_broad_20260329.json`
  - candidate loss `1.3717579152`
  - this is much worse than both:
    - the earlier one-off no-filing artifact `0.8596198236`
    - the ordinary broad run `0.8696287975`
- family comparison against the earlier no-filing artifact says the regression is dominated by:
  - `national_irs_other` `+0.4980`
  - `state_aca_spending` `+0.0040`
  - `state_age_distribution` `+0.0031`
  - `national_population_by_age` `+0.0019`
  - `state_agi_distribution` `+0.0017`
- pre-sim parity also diverged materially:
  - earlier no-filing artifact:
    - state-age support recall `0.5643`
    - state count `50`
    - mean tax-unit size `1.7432`
    - multi-person tax-unit share `0.4199`
  - supported-path rerun:
    - state-age support recall `0.5795`
    - state count `48`
    - mean tax-unit size `1.6550`
    - multi-person tax-unit share `0.3808`
- interpretation:
  - the `filing_status_code` exclusion hook is worth keeping for controlled ablations
  - but the win is **not yet reproducible enough** to set as the default mission path
  - treat this as a reproducibility / run-path discrepancy that needs explanation before widening tax/filer exclusions

## 2026-03-29 deterministic PUF age fix

- found a concrete reproducibility bug in `src/microplex_us/data_sources/puf.py`
  - the live PUF path does **not** have `age` or `AGE_HEAD` after the demographics merge
  - so `map_puf_variables()` falls back to `_impute_age()`
  - `_impute_age()` was adding Gaussian noise with unseeded `np.random.normal(...)`
- that means identical broad `cps+puf + qrf + bootstrap + entropy` runs could differ before donor integration and calibration even with the same configured seed
- patch:
  - `map_puf_variables(..., random_seed=...)`
  - `_impute_age(..., random_seed=...)`
  - `_build_puf_tax_units(..., random_seed=...)`
  - `PUFSourceProvider.load_frame()` now passes provider `random_seed` through to the age-imputation fallback
- regression coverage:
  - `tests/test_puf_source_provider.py::test_map_puf_variables_seed_controls_age_imputation`
  - `tests/test_puf_source_provider.py::test_puf_source_provider_age_imputation_is_reproducible_with_same_seed`
- validation after the patch:
  - two same-seed exported H5s from the broad baseline path
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_postfix_rebuild_a_20260329.h5`
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_postfix_rebuild_b_20260329.h5`
  - have identical pre-sim parity metrics:
    - state-age nonempty cells `571`
    - state-age support recall `0.6220`
    - mean tax-unit size `1.7212`
    - multi-person tax-unit share `0.4013`
  - and identical exported variable arrays across the full common H5 surface (`different_variable_count = 0`)
- implication:
  - same-config A/Bs on the patched path are now much more trustworthy
  - do not interpret older `cps+puf` broad comparisons as fully clean unless they were built after this fix

## 2026-03-30 filing-status exclusion confirmed on deterministic path

- direct PE-native broad rescoring of the deterministic no-filing artifact:
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_postfix_no_filing_20260329.h5`
  - candidate loss `0.8677052580`
  - PE baseline `0.0202439085`
  - delta `+0.8474613495`
- this is a real improvement over the deterministic patched baseline export:
  - patched baseline `0.9286499637`
  - improvement from excluding donor-imputed `filing_status_code`: `0.0609447056` (`6.56%`)
- top remaining family deltas on the improved no-filing candidate are still:
  - `national_irs_other` `+0.2473`
  - `state_agi_distribution` `+0.1822`
  - `state_age_distribution` `+0.1807`
  - `national_population_by_age` `+0.0560`
  - `national_census_other` `+0.0449`
  - `state_aca_spending` `+0.0315`
- compared with the deterministic patched baseline, excluding `filing_status_code`:
  - strongly improves several IRS/HOH/high-income cells
  - but also worsens some ACA spending / ACA enrollment state cells
- pre-sim signal:
  - `filing_status` is exported and used directly in PE-US-data SOI loss masks
  - `exemptions_count` and `eitc_children` are **not** on the exported H5 input surface right now, so they are not the immediate next exclusion candidates
- action:
  - restore `donor_imputer_excluded_variables=("filing_status_code",)` as the default in `USMicroplexBuildConfig`
  - keep investigating the ACA regressions, because this fix helps broad loss overall but is not yet sufficient on its own

## 2026-03-30 leafified default PE export surface

- tightened `SAFE_POLICYENGINE_US_EXPORT_VARIABLES` in `/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/policyengine/us.py`
  - dropped default export of PE computed/add variables that already have leaf inputs on our surface:
    - `employment_income`
    - `self_employment_income`
    - `pension_income`
    - `social_security`
    - `interest_income`
    - `dividend_income`
    - `capital_gains`
    - `filing_status`
  - kept leaf replacements already present on the surface, plus `rent` as the deliberate stored-input exception
- added a regression in `/Users/maxghenis/PolicyEngine/microplex-us/tests/policyengine/test_us.py`
  - default export-map test no longer expects tax-unit `filing_status`
  - new guard checks that the default export whitelist does not overlap PE formula/add/subtract variables except the explicit `rent` exception
- focused verification:
  - `pytest -q tests/policyengine/test_us.py -k 'export_variable_maps or avoids_formula_aggregates'` -> `5 passed`
  - `ruff check src/microplex_us/policyengine/us.py tests/policyengine/test_us.py` -> clean
- post-change audit against live `policyengine-us` metadata:
  - default computed-variable overlap is now only `[('rent', True, False)]`
- interpretation:
  - this aligns `microplex-us` much more closely with the PE-US-data “store leaf inputs, not recomputed aggregates” rule
  - `filing_status` remains available as an explicit direct override if we intentionally want to bypass PE, but it is no longer part of the default pre-sim export contract

## 2026-03-30 scorer env fix + leafified/state-floor follow-up

- fixed a real PE-native rescoring portability bug in `/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/pe_native_scores.py`
  - the scorer now automatically includes a sibling `/Users/maxghenis/PolicyEngine/microimpute` checkout on `PYTHONPATH` when resolving a local `policyengine-us-data` repo
  - added regression coverage in `/Users/maxghenis/PolicyEngine/microplex-us/tests/pipelines/test_pe_native_scores.py`
- focused verification:
  - `pytest -q tests/pipelines/test_pe_native_scores.py tests/test_cps_source_provider.py` -> `12 passed`
  - `ruff check src/microplex_us/pipelines/pe_native_scores.py src/microplex_us/data_sources/cps.py tests/pipelines/test_pe_native_scores.py tests/test_cps_source_provider.py` -> clean
- direct candidate-only PE-native broad rescoring of the leafified export:
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_leafified_export_pe_native_broad_20260330.h5`
  - candidate loss `0.8892950182`
  - this is worse than the deterministic no-filing checkpoint `0.8677052580`
  - interpretation: leafifying the export surface is the right correctness/control-surface fix, but it does not improve the mission metric by itself
- checked a CPS source-sampling state-floor experiment and reverted it
  - temporary artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_leafified_statefloor_export_pe_native_broad_20260330.h5`
  - pre-sim effect:
    - all `51` states survive through seed, synthetic, and calibrated tables
    - exported H5 state-age support recall improved from about `0.5708` to `0.5871`
  - mission effect:
    - candidate loss worsened to `0.9147484499`
  - action:
    - do **not** keep a one-household-per-state floor in default CPS source subsampling
- additional seam confirmed from the live build:
  - `rent` and `real_estate_taxes` are absent from `seed_data`, `synthetic_data`, and `calibrated_data` on the current `cps+puf` path
  - the exported H5 now includes those arrays, but they are all-zero placeholders rather than populated pre-sim inputs

## 2026-03-30 PE-native helper root-cause fix

- the remaining scorer-helper failure under nested `uv run` was not mainly a `PYTHONPATH` problem
- root cause:
  - `/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/pe_native_scores.py` was calling `.resolve()` on `/Users/maxghenis/PolicyEngine/policyengine-us-data/.venv/bin/python`
  - that followed the venv symlink to the underlying Homebrew/system Python binary and silently stripped the venv context
  - effect: the helper subprocess imported global `policyengine_us`, then failed deep inside local `microimpute` with missing `statsmodels`
- fixes now in place:
  - preserve the `.venv/bin/python` path instead of resolving the symlink target
  - build a minimal subprocess env rather than inheriting the full outer process env
  - still include sibling local `microimpute` on `PYTHONPATH`
- regression coverage:
  - `tests/pipelines/test_pe_native_scores.py` now checks both:
    - sibling `microimpute` inclusion on `PYTHONPATH`
    - preservation of the `.venv/bin/python` symlink path
- direct candidate-only broad rescoring remains the trustworthy numeric checkpoint for the leafified export:
  - candidate artifact `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_leafified_export_pe_native_broad_20260330.h5`
  - candidate loss `0.8892950182`

## 2026-03-30 joint-return A/B + export direct-override path

- ruled out a tempting but wrong IRS fix on the live broad path
  - artifact: `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_joint_allocation_head_preserving_ab_20260330.json`
  - config: `cps_asec_2023 + irs_soi_puf_2024`, `sample_n=1000`, `n_synthetic=2000`, `bootstrap + qrf + entropy`, `donor_imputer_excluded_variables=('filing_status_code',)`
  - result:
    - current split baseline: candidate loss `0.8659920427`
    - head-preserving equal-share joint allocation: candidate loss `0.8784570742`
  - interpretation:
    - keeping the “equal-share” PUF joint-return variables entirely on the head makes broad PE-native loss worse
    - the dominant IRS gap is not coming from that specific PUF personization rule
- checked deeper PE role structure on the old better candidate vs the newer leafified export
  - the leafified candidate does **not** lose overall tax-unit dependents or HOH-eligible mass relative to the older better candidate
  - the regressions are therefore about AGI mass allocation within filing statuses, not a simple collapse of dependent/HOH structure
- added first-class direct-export override plumbing for PE-native experiments
  - `/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/us.py`
    - `USMicroplexBuildConfig` now includes `policyengine_direct_override_variables`
    - `export_policyengine_dataset(...)` accepts explicit `direct_override_variables` and defaults to the build config value
  - `/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/performance.py`
    - PE-native scoring path now forwards `build_config.policyengine_direct_override_variables` into export
  - focused verification:
    - `pytest -q tests/pipelines/test_performance.py -k 'native_loss or export_direct_overrides'` -> passed
    - `pytest -q tests/pipelines/test_us.py -k 'export_policyengine_dataset'` -> passed
    - `ruff check src/microplex_us/pipelines/us.py src/microplex_us/pipelines/performance.py tests/pipelines/test_us.py tests/pipelines/test_performance.py` -> clean
- current pending high-signal run:
  - export-policy A/B on the same built candidate tables:
    - default leafified export
    - leafified export + explicit direct override `('filing_status',)`
  - exported datasets already written:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_leaf_default_export_ab_20260330.h5`
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_leaf_filing_override_export_ab_20260330.h5`
  - broad PE-native scores are still running; this is the cleanest test of whether `filing_status` should remain a temporary deliberate exception while deeper tax-unit structure is fixed

## 2026-03-30 repeatability + exact-target diagnosis + parity-input patch

- confirmed the current nominally best broad config is still not reproducible under the same seed:
  - repeated `cps_asec_2023 + irs_soi_puf_2024`, `sample_n=1000`, `n_synthetic=2000`, `bootstrap + qrf + entropy`, `donor_imputer_excluded_variables=('filing_status_code',)` landed at:
    - loss `0.8643217352`, `n_constraints=1234`, `mean_error=0.77098`
    - loss `0.8810677038`, `n_constraints=1252`, `mean_error=0.79746`
  - implication: there is still a real nondeterminism bug in the live build path, not just scorer noise
- exact broad target deltas on the current best saved H5 (`/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_best_broad_target_deltas_20260330.json`) show many hard-zero regressions against PE's enhanced CPS, including:
  - `nation/irs/aca_spending/la`
  - `nation/census/medicare_part_b_premiums/age_20_to_29`
  - `nation/irs/aca_spending/nh`
  - `nation/irs/aca_spending/tx`
  - `nation/irs/adjusted gross income/total/AGI in 500k-1m/taxable/Head of Household`
  - `nation/census/child_support_received`
  - `nation/irs/total social security/total/AGI in 10k-15k/taxable/All`
- traced the zeroed-out targets back to missing pre-sim inputs rather than donor-imputer choice:
  - current best candidate H5 did not export `child_support_received`, `medicare_part_b_premiums`, `other_medical_expenses`, `health_insurance_premiums_without_medicare_part_b`, `alimony_income`, or `disability_benefits`
  - `policyengine-us-data` does source these already:
    - CPS: `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/datasets/cps/cps.py`
    - PUF: `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/datasets/puf/puf.py`
- implemented a parity-input patch on the Microplex-US side:
  - CPS now derives and keeps:
    - `alimony_income`
    - `child_support_received`
    - `disability_benefits`
    - `health_insurance_premiums_without_medicare_part_b`
    - `other_medical_expenses`
    - `over_the_counter_health_expenses`
    - `medicare_part_b_premiums`
  - PUF now maps `alimony_income` under the PE-native name and derives the PE-style medical-expense category breakout from `medical_expense_agi_floor`
  - default PE export surface now includes those new pre-sim inputs
  - focused verification passed:
    - `tests/test_cps_source_provider.py`
    - `tests/test_puf_source_provider.py`
    - `tests/policyengine/test_us.py`
    - Ruff clean
- structural donor-variable ablation did **not** help:
  - excluding `eitc_children`, `exemptions_count`, and `is_male` in addition to `filing_status_code` worsened broad loss from `0.8791992898` to `0.9247766974`
  - implication: do not generalize a blanket “exclude count/binary donor vars” policy
- current pending mission run:
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_parity_inputs_broad_pe_native_20260330.json`
  - same broad config as the current best path, but with the new CPS/PUF parity inputs on the runtime surface

## 2026-03-30 CPS repeatability fix

- isolated the remaining same-seed drift to the CPS provider rather than PUF or the PE-native scorer
  - repeated `CPSASECSourceProvider(year=2023)` loads with `sample_n=1000`, `random_seed=42` were producing different household/person samples from the same cached processed parquet
  - root cause: household sampling depended on unstable row order from derived CPS households; same `random_state` on different row order yields different samples
- fixed `/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/data_sources/cps.py`
  - canonicalize household order by `household_id` before sampling
  - canonicalize person order by `household_id`, `person_id`, `person_number` before sampling
  - sort sampled household/person outputs before returning
- added regression coverage in `/Users/maxghenis/PolicyEngine/microplex-us/tests/test_cps_source_provider.py`
  - repeated same-seed loads from cached processed CPS data now return identical household/person selections
- direct repeatability check after the patch:
  - provider repeatability artifact `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_provider_repeatability_20260330.json`
    - CPS: `same_households=true`, `same_persons=true`
    - PUF: `same_households=true`, `same_persons=true`
  - pre-calibration repeatability artifact `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_repeatability_precal_20260330.json`
    - `same_seed_same_seed_data=true`
    - `same_seed_same_integrated_seed=true`
    - `same_seed_same_synthetic=true`
- focused verification:
  - `pytest -q tests/test_cps_source_provider.py -k 'sampling or deterministic or derives_policyengine_value_inputs'` -> `4 passed`
  - `ruff check src/microplex_us/data_sources/cps.py tests/test_cps_source_provider.py` -> clean
- current pending mission rerun:
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_parity_inputs_broad_pe_native_20260330.json`
  - this is the first broad PE-native rerun on a deterministic `cps+puf + qrf + bootstrap + entropy` path after the parity-input patch

## 2026-03-30 parity-input broad blow-up + stale CPS cache diagnosis

- the first deterministic broad rerun after the parity-input patch landed at:
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_parity_inputs_broad_pe_native_20260330.json`
  - candidate broad PE-native loss `7.433075015991533`
  - PE baseline `0.020243908529428433`
  - delta `+7.412831107462105`
- family breakdown showed the blow-up was overwhelmingly concentrated in `national_census_other`
  - contribution delta `+6.582284784720224`
  - other major regressions remained `national_irs_other`, `state_agi_distribution`, and `state_age_distribution`
- direct H5/input inspection showed the parity-input runtime was still not actually carrying all of the new CPS-derived inputs:
  - exported candidate H5 had `child_support_received = 0` everywhere and no `disability_benefits`
  - stage audit confirmed the problem was upstream of export on the live cache-backed path:
    - `seed_data` and `synthetic_data` were missing `child_support_received` and `disability_benefits`
- root cause:
  - `/Users/maxghenis/.cache/microplex/cps_asec_2023_processed.parquet` was stale relative to the new CPS loader contract
  - `load_cps_asec()` cache validation only required the older geography / coverage columns, so it silently reused a processed cache that predated the new PE-native derived inputs
- fix now in place:
  - `/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/data_sources/cps.py`
    - extended `PERSON_CACHE_REQUIRED_COLUMNS` to require:
      - `alimony_income`
      - `child_support_received`
      - `disability_benefits`
      - `health_insurance_premiums_without_medicare_part_b`
      - `other_medical_expenses`
      - `over_the_counter_health_expenses`
      - `medicare_part_b_premiums`
  - `/Users/maxghenis/PolicyEngine/microplex-us/tests/test_cps_source_provider.py`
    - updated stale-cache and deterministic-cache fixtures to match the stricter processed-cache contract
    - focused verification:
      - `pytest -q tests/test_cps_source_provider.py -k 'deterministic or stale_processed_cache_without_pe_presim_inputs or derives_policyengine_value_inputs'` -> passed
      - `ruff check src/microplex_us/data_sources/cps.py tests/test_cps_source_provider.py` -> clean
- live-path verification after rebuilding the actual cached CPS parquet:
  - `load_cps_asec(year=2023)` now rebuilds the stale cache and returns all new derived inputs
  - on the broad runtime path:
    - `child_support_received` is now present in `seed_data`, `synthetic_data`, and `calibrated_data`
    - `disability_benefits` is now present in `seed_data`, `synthetic_data`, and `calibrated_data`
- current pending clean rerun:
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_parity_inputs_broad_pe_native_20260330_v2.json`
  - this is the first broad PE-native rerun on:
    - deterministic CPS sampling
    - rebuilt live CPS processed cache
    - actual carriage of the new CPS-derived PE inputs

## 2026-03-30 focused Claude review — broad PE-native loss checkpoint

- Scope: v2 clean broad result after deterministic CPS + rebuilt cache fixes
- Full review: `reviews/2026-03-30-claude-broad-native-loss-checkpoint-review.md`
- Top findings:
  1. **HIGH**: Calibration-vs-scoring target mismatch dominates loss — calibrated against 1,255 constraints, scored against 2,817 targets. Top 3 families (`national_irs_other`, `state_agi_distribution`, `state_age_distribution`) account for 72% of the 0.855 delta.
  2. **HIGH**: Calibration never converges — all saved artifacts show `converged=false`. A/B comparisons unreliable unless delta exceeds ~0.02-0.03.
  3. **MEDIUM**: Cache invalidation checks column presence, not derivation correctness — same bug class as the 7.43 blow-up, different future trigger.
- 7.43 blow-up: fully explained by stale CPS processed cache missing new PE-derived inputs. No deeper bug.
- v2 result (candidate 0.875, PE baseline 0.020): trustworthy for family-level diagnosis, not for precision claims.
- Top next fixes:
  1. Increase source `sample_n` to 2000-3000 (steepest support-recall curve)
  2. Diagnose calibration convergence with 10x solver iterations
  3. Add cache derivation version to prevent stale-cache class bugs
  4. Split `national_irs_other` in the family classifier for sub-family diagnosis

## 2026-03-30 follow-up to Claude broad-loss review

- Landed the two most direct correctness/investigation fixes from the review:
  - `src/microplex_us/data_sources/cps.py`
    - added a versioned processed-cache path:
      - `cps_asec_{year}_processed_v20260330.parquet`
    - legacy unversioned processed caches are now ignored and rebuilt from raw source
    - minimal CPS inputs now still materialize the PE-facing value leaves as zero columns:
      - `alimony_income`
      - `child_support_received`
      - `disability_benefits`
      - `health_insurance_premiums_without_medicare_part_b`
      - `other_medical_expenses`
      - `over_the_counter_health_expenses`
      - `medicare_part_b_premiums`
  - `src/microplex_us/pipelines/us.py`
    - `USMicroplexBuildConfig` now carries:
      - `calibration_tol`
      - `calibration_max_iter`
    - entropy / IPF / chi2 calibrators now honor those settings
  - `src/microplex_us/pipelines/performance.py`
    - calibration cache keys now include `calibration_tol` and `calibration_max_iter`
    - precalibration cache keys exclude them so only the calibration stage reruns when these change
- Focused verification:
  - `pytest -q tests/test_cps_source_provider.py tests/pipelines/test_us.py -k 'cache or deterministic or tolerance_config or stale_processed_cache or derives_policyengine_value_inputs or build_weight_calibrator'` -> `7 passed`
  - `pytest -q tests/pipelines/test_performance.py -k 'calibration_cache_key_includes_iteration_and_tolerance_settings or preserves_target_profiles or can_evaluate_native_loss'` -> `3 passed`
  - `ruff check src/microplex_us/data_sources/cps.py src/microplex_us/pipelines/us.py src/microplex_us/pipelines/performance.py tests/test_cps_source_provider.py tests/pipelines/test_us.py tests/pipelines/test_performance.py` -> clean
- Running now:
  - deterministic broad PE-native smoke on the current path with:
    - `sample_n=2000`
    - `n_synthetic=2000`
    - `donor_imputer_backend='qrf'`
    - `donor_imputer_excluded_variables=('filing_status_code',)`
    - `calibration_backend='entropy'`
    - `calibration_max_iter=1000`
  - output target:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_weighted_sample2000_iter1000_pe_native_broad_20260330.json`
- Result:
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_weighted_sample2000_iter1000_pe_native_broad_20260330.json`
  - candidate PE-native broad loss: `0.8830832791543215`
  - PE baseline: `0.020243908529428433`
  - delta: `+0.862839370624893`
  - calibration still did not converge:
    - `converged=false`
    - `mean_error=0.8053911891798184`
    - `max_error=1.5450053947105458`
    - `n_constraints=1263`
  - feasibility filter still dropped `2348 / 3611` constraints (`65.0%`)
  - conclusion:
    - increasing entropy solve effort from `100` to `1000` iterations on the current deterministic `sample_n=2000 / n_synthetic=2000` path did not help the mission metric
    - next lever should stay on source support (`sample_n=3000`) rather than more entropy iterations

## 2026-03-30 full-support PE-scale path

- Code:
  - `src/microplex_us/pipelines/us.py`
    - added `synthesis_backend='seed'` to preserve the full donor-integrated support surface instead of resampling it before PE-table calibration
    - added `policyengine_selection_household_budget` and a sparse household selector that prunes PE tables to a fixed household budget before the final calibration pass
  - `src/microplex_us/pipelines/performance.py`
    - `sample_n` can now be `None` for full-source runs
    - calibration cache keys now include `policyengine_selection_household_budget`, while precalibration cache keys still do not
- Focused verification:
  - `pytest -q tests/pipelines/test_us.py -k 'synthesize_seed_backend_preserves_seed_support or household_budget or sparse_backend or calibrate_policyengine_tables_from_db'` -> `6 passed`
  - `pytest -q tests/pipelines/test_performance.py -k 'household_budget_selection or full_source_queries or preserves_target_profiles or native_loss'` -> `4 passed`
  - `ruff check src/microplex_us/pipelines/us.py src/microplex_us/pipelines/performance.py tests/pipelines/test_us.py tests/pipelines/test_performance.py` -> clean
- PE-scale source-subsampled comparison point:
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_qrf_weighted_sample29999_n29999_pe_native_broad_20260330.json`
  - config:
    - `sample_n=29999`
    - `n_synthetic=29999`
    - `bootstrap + qrf + entropy`
    - `donor_imputer_excluded_variables=('filing_status_code',)`
  - result:
    - candidate PE-native broad loss: `0.9547853569761191`
    - PE baseline: `0.020243908529428433`
    - delta: `+0.9345414484466906`
    - `converged=false`
    - `n_constraints=3300`
  - read:
    - matching PE's row count by source-side weighted subsampling is worse than the smaller deterministic broad path
    - the better next experiment is full CPS + full PUF support, then prune to `29,999` households with the new sparse selection stage
- Running now:
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_fullsource_seed_sparse29999_pe_native_broad_20260330.json`
  - config:
    - `sample_n=None` (full sources)
    - `synthesis_backend='seed'`
    - `policyengine_selection_household_budget=29999`
    - `donor_imputer_backend='qrf'`
    - `donor_imputer_excluded_variables=('filing_status_code',)`

## 2026-03-31 direct PE-native objective path

- Code:
  - `src/microplex_us/pipelines/pe_native_optimization.py`
    - added direct PE-native loss-matrix extraction from `policyengine-us-data`
    - added projected gradient weight optimization on the exact broad PE-native objective for a fixed exported candidate
    - added H5 rewrite utilities to propagate optimized household weights to person and group weight arrays
  - `src/microplex_us/pipelines/performance.py`
    - added opt-in `optimize_pe_native_loss` harness mode so exported candidates can be weight-optimized before PE-native scoring
  - `src/microplex_us/pipelines/__init__.py`
    - exported the direct PE-native optimization helpers
- Focused verification:
  - `pytest -q tests/pipelines/test_pe_native_optimization.py tests/pipelines/test_performance.py -k 'native_loss or pe_native_optimization'` -> `5 passed`
  - `ruff check src/microplex_us/pipelines/pe_native_optimization.py src/microplex_us/pipelines/performance.py src/microplex_us/pipelines/__init__.py tests/pipelines/test_pe_native_optimization.py tests/pipelines/test_performance.py` -> clean
- First same-candidate direct-objective A/B:
  - input candidate:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/live_pe_native_broad_entropy_batch_noharness_20260329/20260329T210427Z-057066af/policyengine_us.h5`
  - optimized output:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pe_native_direct_opt_20260331.h5`
  - summary:
    - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_pe_native_direct_opt_20260331.json`
  - result:
    - raw candidate PE-native broad loss: `0.9233365911702252`
    - direct-objective optimized loss: `0.9229024219474923`
    - improvement: `-0.00043416922273291814`
    - baseline PE loss: `0.020243908529428433`
    - optimizer status:
      - `converged=false`
      - `iterations=200`
      - `positive_household_count=1993 / 2000`
- Read:
  - optimizing the exact PE-native broad objective on a fixed exported candidate helps only trivially
  - objective mismatch is real but not the main blocker on the current path
  - the next large gain must come from better records or a budgeted selector over a larger support set, not just replacing entropy with a better weight objective after export

## 2026-03-31 PE-native optimizer score-consistency guard

- Code:
  - `src/microplex_us/pipelines/performance.py`
    - added a hard consistency check for `optimize_pe_native_loss=True`
    - the rescored `candidate_enhanced_cps_native_loss` must now match the optimizer's internal `optimized_loss` within `pe_native_score_consistency_tol` (default `1e-6`)
    - mismatches now raise immediately instead of silently attaching stale/incorrect optimization metadata
- Focused verification:
  - `pytest -q tests/pipelines/test_performance.py -k 'optimize_native_loss or consistency'` -> `1 passed`
  - `ruff check src/microplex_us/pipelines/performance.py tests/pipelines/test_performance.py` -> clean
- Read:
  - this does not change the diagnosis; it just makes the direct-objective path trustworthy for future larger-candidate selector work

## 2026-03-31 focused Claude review — direct PE-native optimizer

- Scope: code review + architectural diagnosis of `pe_native_optimization.py`, harness integration, and first A/B result
- Full review: `reviews/2026-03-31-claude-direct-pe-native-optimizer-review.md`
- Top findings:
  1. **Objective alignment is correct**: optimizer's `||M^T w - s||^2` proven algebraically identical to the scorer's native loss. Initial losses match within float64 noise (2e-16).
  2. **No serious correctness bugs**: gradient, Lipschitz estimate, simplex projection, H5 weight rewrite, and harness integration are all correct.
  3. **MINOR**: weight-sum drift ~9e-6 relative after 200 iterations (cosmetic). No cross-validation between optimizer's internal loss and rescored loss (worth adding as guard).
- Objective alignment confirmed: the direct optimizer minimizes the exact same function the scorer evaluates.
- Tiny gain (0.92334 → 0.92290) definitively confirms record support is the bottleneck:
  - The best achievable loss with 2000 households is ~0.923 — entropy was already near-optimal for this support
  - Only 0.05% of the 0.903 gap to PE is attributable to the weight objective
  - The other 99.95% is structural (support, state coverage, missing IRS mass)
- Top next fix: full-support + budgeted household selection path (already prototyped). Do not invest further in direct weight optimization on small candidates.

## 2026-03-31 full-support PE-native-loss selector at PE row budget

- Scope: full CPS + full PUF support, `synthesis_backend='seed'`, `policyengine_selection_backend='pe_native_loss'`, household budget `29,999`
- Artifact:
  - `artifacts/tmp_fullsource_seed_pe_native_selector29999_20260331.json`
  - `artifacts/tmp_fullsource_seed_pe_native_selector29999_20260331.h5`
- Result:
  1. candidate PE-native broad loss `0.6333835740352115`
  2. PE baseline `0.020243908529428433`
  3. delta `+0.613139665505783`
- Comparison:
  1. materially better than earlier full-support sparse selector (`0.8960`)
  2. materially better than source-sampled `29,999` run (`0.9548`)
  3. still far from full PE baseline
- Diagnostics:
  1. final calibration still `converged=false`
  2. supported targets `2575 / 4183`
  3. feasibility filter dropped `887 / 3462` post-selection constraints (`25.6%`)
  4. selector optimization itself did not converge in `200` iterations, but still produced a much stronger selected population
  5. selector kept exactly `29,999` positive-weight households from `56,839` input households
- Read:
  - budgeted selection on a full-support candidate is the first PE-scale change that clearly moved the frontier in the right direction
  - this is still not enough to beat full PE, but it is strong evidence that candidate construction + selection is a better lever than source-side subsampling or post-export weight tuning

## 2026-03-31 harness output contract

- Code:
  - `src/microplex_us/pipelines/performance.py`
    - added `output_json_path` and `output_policyengine_dataset_path` to the local harness config
    - harness can now persist one self-contained JSON summary and one final PE-ingestable H5 without ad hoc wrapper scripts
    - when `optimize_pe_native_loss=True`, the persisted H5 is the optimized dataset that was actually scored, not the pre-optimization export
- Focused verification:
  - `pytest -q tests/pipelines/test_performance.py -k 'write_output_bundle or writes_optimized_dataset_output or can_optimize_native_loss or can_evaluate_native_loss'` -> `4 passed`
  - `ruff check src/microplex_us/pipelines/performance.py tests/pipelines/test_performance.py` -> clean
- Read:
  - long PE-scale runs no longer need bespoke `uv run python <<PY` wrappers just to save a JSON summary and exported dataset

## 2026-03-31 harness target-delta output

- Code:
  - `src/microplex_us/pipelines/performance.py`
    - added `output_pe_native_target_delta_path` and `pe_native_target_delta_top_k`
    - local harness can now emit the exact PE-native top regressions / improvements against the PE baseline as part of a normal run
    - target-delta output follows the final scored dataset, so optimized runs analyze the optimized H5 rather than the pre-optimization export
- Focused verification:
  - `pytest -q tests/pipelines/test_performance.py -k 'write_pe_native_target_delta_output or rejects_nonpositive_target_delta_top_k or write_output_bundle or writes_optimized_dataset_output or can_optimize_native_loss or can_evaluate_native_loss'` -> `6 passed`
  - `ruff check src/microplex_us/pipelines/performance.py tests/pipelines/test_performance.py` -> clean
- Read:
  - the ad hoc exact-target analysis wrapper can now be replaced by a first-class harness output

## 2026-03-31 harness batch native scoring

- Code:
  - `src/microplex_us/pipelines/performance.py`
    - added `USMicroplexPerformanceHarnessRequest` and `USMicroplexPerformanceSession.run_batch(...)`
    - shared-session batch runs now export candidates once, group compatible requests by baseline/repo/period, and score PE-native loss through `compute_batch_us_pe_native_scores(...)`
    - keeps direct PE-native optimizer runs on the single-candidate path, but removes repeated scorer subprocess overhead for normal multi-candidate native-loss A/Bs
- Focused verification:
  - `pytest -q tests/pipelines/test_performance.py -k 'run_batch_uses_native_batch_scorer or write_pe_native_target_delta_output or rejects_nonpositive_target_delta_top_k or write_output_bundle or writes_optimized_dataset_output or can_optimize_native_loss or can_evaluate_native_loss or reuses_comparison_cache or reuses_loaded_frames or reuses_precalibration_state or reuses_calibration_state'` -> `11 passed`
  - `ruff check src/microplex_us/pipelines/performance.py src/microplex_us/pipelines/__init__.py src/microplex_us/__init__.py tests/pipelines/test_performance.py` -> clean
- Read:
  - the local performance harness now has a real multi-candidate PE-native path instead of relying on separate experiment/backfill machinery

## 2026-03-31 harness matched-N PE baseline

- Code:
  - `src/microplex_us/pipelines/performance.py`
    - added `evaluate_matched_pe_native_loss`
    - harness can now sample the full PE baseline down to a matched household count, rescale the sampled baseline weights back to the original total, and score `Microplex@N` against that raw `PE@N`
    - default matched household count follows the candidate household count; optional output path persists the sampled PE baseline H5
- Focused verification:
  - `pytest -q tests/pipelines/test_performance.py -k 'evaluate_matched_native_loss or rejects_nonpositive_matched_baseline_household_count or run_batch_uses_native_batch_scorer or write_pe_native_target_delta_output or rejects_nonpositive_target_delta_top_k or write_output_bundle or writes_optimized_dataset_output or can_optimize_native_loss or can_evaluate_native_loss or reuses_comparison_cache or reuses_loaded_frames or reuses_precalibration_state or reuses_calibration_state'` -> `13 passed`
  - `ruff check src/microplex_us/pipelines/performance.py tests/pipelines/test_performance.py` -> clean
- Read:
  - matched-size raw PE baselines are now a first-class harness comparator instead of a separate notebook-style script

## 2026-03-31 harness matched-N reweighted PE baseline

- Code:
  - `src/microplex_us/pipelines/performance.py`
    - added `reweight_matched_pe_native_loss`
    - matched-size PE baseline path can now run PE's own `enhanced_cps.reweight(...)` on the sampled baseline H5 before rescoring
    - this gives the local harness a fairer `PE@N_reweighted` comparator than simple weight rescaling alone
- Focused verification:
  - `pytest -q tests/pipelines/test_performance.py -k 'reweight_matched_native_loss or evaluate_matched_native_loss or rejects_reweighted_matched_loss_without_matched_loss or rejects_nonpositive_matched_baseline_household_count or run_batch_uses_native_batch_scorer or write_pe_native_target_delta_output or rejects_nonpositive_target_delta_top_k or write_output_bundle or writes_optimized_dataset_output or can_optimize_native_loss or can_evaluate_native_loss or reuses_comparison_cache or reuses_loaded_frames or reuses_precalibration_state or reuses_calibration_state'` -> `15 passed`
  - `ruff check src/microplex_us/pipelines/performance.py tests/pipelines/test_performance.py` -> clean
- Read:
  - the local harness can now emit `Microplex@N`, raw `PE@N`, and reweighted `PE@N` from one comparable evaluation surface

## 2026-03-31 matched PE baseline fidelity fix

- Scope: repaired matched-`N` PE baseline generation in `src/microplex_us/pipelines/performance.py`
- Root cause:
  - the harness matched-baseline writer was lossy
  - full-count `PE@29999` collapsed to `17` variables instead of `167`
  - smaller matched baselines silently dropped non-annual variables such as `is_household_head` (`ETERNITY`) and `receives_wic` (monthly)
- Fix:
  - `N == full_N` now short-circuits to a byte-for-byte copy of the original PE baseline H5
  - smaller matched baselines are now sampled directly at the H5 array level, preserving all variables and all stored periods
- Focused verification:
  - `pytest -q tests/pipelines/test_performance.py -k 'matched_native_loss or write_matched_policyengine_us_baseline_dataset_preserves_variables'` -> `3 passed`
  - `ruff check src/microplex_us/pipelines/performance.py tests/pipelines/test_performance.py` -> clean
  - direct schema diff now matches full PE exactly at `N=2000`, `N=3000`, and `N=29999` (`167` vars, no missing, no extra)
- Consequence:
  - the earlier harness-produced raw `PE@29999` comparator was invalid and should not be used

## 2026-03-31 filing-status experiments falsified

- Scope: tested two ways to push separated / surviving-spouse structure into PE on the `29,999` full-support selector path
  - direct `filing_status` override
  - exporting person-level `is_separated` / `is_surviving_spouse`
- Results:
  - prior `statusfix` baseline: `0.6362298466`
  - direct `filing_status` override: `0.6539544578`
  - leaf-input export: `0.9793611801`
  - PE baseline: `0.0202439085`
- Root cause read:
  - PE's `filing_status` formula uses tax-unit structure plus person-level leaf inputs
  - direct override carried existing synthesized MFJ structural errors straight into PE
  - the leaf-input experiment was worse because coarse CPS `marital_status` / `filing_status_code` hints were not precise enough to safely synthesize `is_separated` and `is_surviving_spouse`
  - that path inflated separated-filer structure and caused severe weight collapse
- Code consequence:
  - reverted `is_separated` / `is_surviving_spouse` from the default PE export surface
  - kept only passthrough normalization if those columns ever exist from a more trustworthy source
- Read:
  - the filing-status seam is real, but these two fixes are not the right fix
  - next work should shift back to the larger `national_irs_other`, `state_agi_distribution`, and `state_age_distribution` support problems

## 2026-03-31 signed IRS surface repair

- Scope: repair signed-income and missing-leaf seams that were still zeroing major IRS loss terms on the `29,999` full-support selector path
- Root cause:
  - raw mapped PUF `self_employment_income` is signed, but Microplex marked it as `ZERO_INFLATED_POSITIVE`, so donor matching could never emit losses
  - raw mapped PUF `rental_income_negative` is a positive loss amount, and `map_puf_variables()` was adding it instead of subtracting it
  - `capital_gains_distributions` existed in PUF but never reached PE because the export surface omitted the correct PE input alias `non_sch_d_capital_gains`
- Code:
  - `src/microplex_us/data_sources/puf.py`
    - preserve rental losses as negative values when combining positive and negative rental components
  - `src/microplex_us/variables.py`
    - stop treating `self_employment_income` as a positive-only donor target; preserve signed support
  - `src/microplex_us/policyengine/us.py`
    - export `capital_gains_distributions` through the PE input alias `non_sch_d_capital_gains`
- Focused verification:
  - `pytest -q tests/test_puf_source_provider.py -k 'rental_loss_sign or preserve_joint_tax_unit_monetary_totals or splits_negative_joint_self_employment_losses or maps_policyengine_medical_and_alimony_inputs'` -> `3 passed`
  - `pytest -q tests/policyengine/test_us.py -k 'default_policyengine_us_export_surface_avoids_formula_aggregates or supports_pre_sim_aliases'` -> `2 passed`
  - `pytest -q tests/test_variables.py -k 'self_employment_income_semantics_preserve_signed_support or person_native_irs_semantics_match_current_policyengine_entities or donor_imputation_block_specs_include_match_strategies'` -> `3 passed`
  - `ruff check src/microplex_us/data_sources/puf.py src/microplex_us/policyengine/us.py src/microplex_us/variables.py tests/test_puf_source_provider.py tests/policyengine/test_us.py tests/test_variables.py` -> clean
- Read:
  - the remaining IRS gap is not just “more support”; several high-loss cells were impossible to hit because losses or leaves were being structurally erased before PE saw them

## 2026-03-31 authoritative donor override for shared IRS variables

- Scope: allow PUF to replace weak shared CPS scaffold values for a narrow signed-IRS allowlist instead of only filling donor-only variables
- Root cause:
  - even after restoring signed PUF support, donor integration only modeled `donor_observed - scaffold_observed`
  - `self_employment_income` and `rental_income` exist on both CPS and PUF, so PUF could not overwrite the CPS scaffold despite being the more authoritative IRS-style source
  - when a shared variable becomes a donor target, it also must be removed from donor conditions for that block; otherwise the imputer just learns back the scaffold value being replaced
- Code:
  - `src/microplex_us/pipelines/us.py`
    - add `donor_imputer_authoritative_override_variables`, defaulting to `self_employment_income` and `rental_income`
    - allow authoritative donors to model and overwrite those shared variables
    - exclude block target variables from the donor condition set
- Focused verification:
  - `pytest -q tests/pipelines/test_us.py -k 'authoritative_override_for_shared_irs_variables or preserves_informative_scaffold_values or defaults'` -> `4 passed`
  - `ruff check src/microplex_us/pipelines/us.py tests/pipelines/test_us.py` -> clean
- Cheap export spotcheck:
  - `artifacts/tmp_signed_income_override_spotcheck_20260331.h5`
  - `self_employment_income_before_lsr`: `31` negative rows, `62` positive rows, min `-14175.0`
  - `rental_income`: `14` negative rows, `32` positive rows, min `-243450.0`
  - `non_sch_d_capital_gains`: `24` positive rows
- Read:
  - the signed IRS surfaces now survive into a real PE export, which is the prerequisite for the next full `29,999` selector rerun

## 2026-03-31 signed-support and shared-override PE-scale readout

- Full `29,999` selector results:
  - prior strong selector: `0.6333835740`
  - `statusfix` baseline: `0.6362298466`
  - signed-support fixes only: `0.9762246696`
  - signed-support + `self_employment_income` authoritative override: `0.9317965866`
  - signed-support + `rental_income` authoritative override: `0.9831478185`
  - signed-support + both overrides: `0.9686514499`
  - PE baseline: `0.0202439085`
- Read:
  - restoring signed IRS support was necessary for representability, but not a win on the current selector/calibration path
  - all shared authoritative override variants were worse than the pre-override baseline
  - `self_employment_income` override was harmful; `rental_income` override was worse
  - keep `donor_imputer_authoritative_override_variables` opt-in only, not default
- Code consequence:
  - revert the default override allowlist to `()`
  - retain the override mechanism for future bounded A/Bs only

## 2026-03-31 capital gains distributions export moved back to opt-in

- Evidence:
  - the no-override `signedirsfix` run (`0.9762246696`) was still far worse than `statusfix` (`0.6362298466`)
  - in `tmp_fullsupport_selector29999_signedirsfix_20260331.h5`, `self_employment_income_before_lsr` and `rental_income` still had `0` negative rows, so the signed-income support repairs were not yet affecting the default path
  - the main new default-path change was exporting `capital_gains_distributions` as `non_sch_d_capital_gains`
- Code consequence:
  - remove `non_sch_d_capital_gains` from `SAFE_POLICYENGINE_US_EXPORT_VARIABLES`
  - keep the alias available for explicit opt-in through `direct_override_variables`
- Focused verification:
  - `pytest -q tests/policyengine/test_us.py -k 'default_policyengine_us_export_surface_avoids_formula_aggregates or supports_pre_sim_aliases'` -> `2 passed`
  - `ruff check src/microplex_us/policyengine/us.py tests/policyengine/test_us.py` -> clean
- Read:
  - until a direct H5 ablation proves otherwise, `non_sch_d_capital_gains` should not be on the default PE export surface

## 2026-04-01 PE-native support audit on the trusted `statusfix` path

- Code:
  - `src/microplex_us/pipelines/pe_native_scores.py`
    - add `compute_us_pe_native_support_audit(...)`
    - compare candidate vs baseline on stored-variable presence, filing-status support, high-AGI MFS support, state marketplace enrollment, and state age-bucket support
  - `src/microplex_us/pipelines/performance.py`
    - add `output_pe_native_support_audit_path`
    - allow the harness to emit a durable support-audit JSON next to the normal PE-native score outputs
  - `tests/pipelines/test_pe_native_scores.py`
  - `tests/pipelines/test_performance.py`
- Focused verification:
  - `uv run pytest -q tests/pipelines/test_pe_native_scores.py -k 'support_audit or target_deltas'` -> `2 passed`
  - `uv run pytest -q tests/pipelines/test_performance.py -k 'support_audit or target_delta_output'` -> `2 passed`
  - `ruff check src/microplex_us/pipelines/pe_native_scores.py src/microplex_us/pipelines/performance.py tests/pipelines/test_pe_native_scores.py tests/pipelines/test_performance.py` -> clean
- Artifact:
  - `artifacts/tmp_fullsupport_selector29999_statusfix_support_audit_20260401.json`
- Read:
  - the trusted `statusfix` candidate is not just missing a few leaves; it is structurally underweighted after calibration
  - candidate PE household-weight sum: `41.17M`
  - same run's selection optimizer preserved `135.40M` total weight before entropy calibration
  - full `enhanced_cps_2024` baseline PE household-weight sum: `149.96M`
  - support gaps are therefore broad, not isolated:
    - `child_support_expense` is entirely absent on the candidate export (`stored=false`, `weighted_nonzero=0.0`) while baseline has `2.63M` weighted nonzero support
    - `has_marketplace_health_coverage`: candidate `2.54M` weighted nonzero vs baseline `11.74M`
    - `has_esi`: candidate `63.61M` vs baseline `185.45M`
    - `medicare_part_b_premiums`: candidate `11.54M` vs baseline `49.53M`
    - `self_employment_income_before_lsr`: candidate `3.74M` vs baseline `25.53M`
    - `rental_income`: candidate `3.04M` vs baseline `13.21M`
  - filing-status support is still structurally incomplete:
    - `SEPARATE` weighted count `0.0` vs baseline `6.53M`
    - `SURVIVING_SPOUSE` weighted count `0.0` vs baseline `1.74M`
    - MFS support in `75k+` AGI bins is exactly zero across the board
  - ACA and state-age failures are clearly structural:
    - biggest marketplace enrollment gaps include `GA`, `CA`, `TX`, `IL`, `NY`
    - biggest state-age bucket gaps are concentrated in `TX`, `CA`, and `FL`
- Next hypothesis:
  - the best current selector path is being undone by post-selection entropy calibration collapsing total mass
  - the next decisive experiment is to renormalize the final calibrated `statusfix` weights back toward the pre-calibration/selection total and rescore before changing record construction again

## 2026-04-01 source-backed `child_support_expense` added to CPS + PE export

- Code:
  - `src/microplex_us/data_sources/cps.py`
    - map CPS `CHSP_VAL -> child_support_expense`
    - treat it as a nonnegative zero-default PE pre-sim input
    - require it in the processed CPS cache contract
    - bump CPS processed-cache version to `20260401`
  - `src/microplex_us/policyengine/us.py`
    - add `child_support_expense` to `SAFE_POLICYENGINE_US_EXPORT_VARIABLES`
  - `tests/test_cps_source_provider.py`
  - `tests/policyengine/test_us.py`
- Why:
  - the new PE-native support audit showed the trusted `statusfix` candidate exported no `child_support_expense` at all, while the full PE baseline had `2.63M` weighted nonzero support
  - `policyengine-us-data` already sources this directly from CPS (`CHSP_VAL`), so this is a clean parity miss rather than a speculative new feature
- Focused verification:
  - `uv run pytest -q tests/test_cps_source_provider.py -k 'policyengine_value_inputs or stale_processed_cache_without_pe_presim_inputs or caches_household_geography_on_persons'` -> `3 passed`
  - `uv run pytest -q tests/policyengine/test_us.py -k 'export_variable_maps_includes_tax_inputs or default_policyengine_us_export_surface_avoids_formula_aggregates'` -> `2 passed`
  - `ruff check src/microplex_us/data_sources/cps.py src/microplex_us/policyengine/us.py tests/test_cps_source_provider.py tests/policyengine/test_us.py` -> clean
- Read:
  - this is a safe source-backed fix and should stay
  - it may help some SNAP / expense surfaces, but it is not expected to explain the full `statusfix` gap by itself

## 2026-04-12 reject checkpoint CPS `state x household-income-band` floor

- Code:
  - no retained code changes; the temporary `state_income_floor` experiment in
    `src/microplex_us/data_sources/cps.py` and
    `src/microplex_us/pipelines/pe_us_data_rebuild_checkpoint.py` was reverted
    after the benchmark run regressed
- Why:
  - the next clean AGI-side upstream hypothesis was to mirror the accepted CPS
    `state x age-band` support floor with a coarse `state x household-income-band`
    floor during checkpoint sampling
  - this stayed within the same architecture: better sampled source support
    before synthesis/calibration, same PE oracle, same downstream calibration
    planner
- Focused verification:
  - `python -m py_compile src/microplex_us/data_sources/cps.py src/microplex_us/pipelines/pe_us_data_rebuild_checkpoint.py tests/test_cps_source_provider.py tests/pipelines/test_pe_us_data_rebuild_checkpoint.py`
  - `uv run pytest tests/test_cps_source_provider.py tests/pipelines/test_pe_us_data_rebuild_checkpoint.py -q -k 'state_age_floor or default_policyengine_us_data_rebuild_queries'`
- Artifact:
  - `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_income_donors/broader-donors-cps-stateage1-income-v1`
- Read:
  - the hypothesis lost on the mission metric and should not stay in the code
    surface
  - matched broader donor baseline with the accepted CPS age floor:
    `full_oracle_capped_mean_abs_relative_error = 0.7329149849`
  - candidate with the added income-band floor:
    `full_oracle_capped_mean_abs_relative_error = 0.7554346215`
  - delta: `+0.0225196366` worse
  - the candidate also worsened active-solve capped loss (`0.8499 -> 0.8586`)
    while increasing selected constraints (`1059 -> 1086`)
  - conclusion: keep the accepted checkpoint CPS `state x age-band` floor, and
    do not add the `state x household-income-band` floor

## 2026-04-12 reject checkpoint CPS `state x tax-unit-income-band` floor

- Code:
  - no retained code changes; the temporary `state_tax_unit_income_floor`
    experiment in `src/microplex_us/data_sources/cps.py` and
    `src/microplex_us/pipelines/pe_us_data_rebuild_checkpoint.py` was reverted
    after the benchmark run
- Why:
  - the household-income analogue was too blunt, so the next cleaner AGI-side
    upstream hypothesis was a CPS `state x tax-unit-income-band` floor built
    from summed `total_person_income` within each CPS tax unit
  - this is closer to the PE AGI target surface than household income while
    still staying entirely in checkpoint-scale source sampling
- Focused verification:
  - `python -m py_compile src/microplex_us/data_sources/cps.py src/microplex_us/pipelines/pe_us_data_rebuild_checkpoint.py tests/test_cps_source_provider.py tests/pipelines/test_pe_us_data_rebuild_checkpoint.py`
  - `uv run pytest tests/test_cps_source_provider.py tests/pipelines/test_pe_us_data_rebuild_checkpoint.py -q -k 'state_age_floor or default_policyengine_us_data_rebuild_queries'`
- Artifact:
  - `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_taxunitincome_donors/broader-donors-cps-stateage1-taxunitincome-v1`
- Read:
  - this was a near miss but still not a keeper on the mission metric
  - matched broader donor baseline with the accepted CPS age floor:
    `full_oracle_capped_mean_abs_relative_error = 0.7329149849`
  - candidate with the added tax-unit-income floor:
    `full_oracle_capped_mean_abs_relative_error = 0.7372298992`
  - delta: `+0.0043149143` worse
  - unlike the household-income version, this candidate did improve some
    secondary diagnostics:
    - `full_oracle_mean_abs_relative_error`: `0.8169 -> 0.8134`
    - `active_solve_capped_mean_abs_relative_error`: `0.8499 -> 0.8047`
  - conclusion: still reject for the current frontier objective; if this idea
    comes back later, it should come back with tighter AGI-band design or a
    clearer target-family-specific objective rather than as a default checkpoint
    support rule

## 2026-04-12 reject PolicyEngine-style CPS tax-leaf splits on the broader donor checkpoint

- Code:
  - no retained runtime code changes from this lane
  - the temporary CPS-source leaf-input materialization and the temporary
    export-side split fallback were both reverted after the benchmark runs
  - retained code state only bumps the CPS processed-cache version in
    `src/microplex_us/data_sources/cps.py` to avoid reusing the rejected
    source-side cache schema
- Why:
  - the next direct AGI-alignment hypothesis was to reuse the same CPS tax-input
    split assumptions as `policyengine-us-data` for interest, dividends, and
    pension income
  - two boundaries were tested:
    - source-side: materialize those leaf inputs directly in the CPS provider
      before Microplex donor integration
    - export-side: keep the CPS source on gross aggregates but apply the same
      split only when building the final PolicyEngine export surface
- Focused verification:
  - source/provider and semantic regression slice:
    `uv run pytest tests/test_cps_source_provider.py tests/test_variables.py tests/pipelines/test_us.py -q -k 'policyengine_value_inputs or atomic_variable_semantics or prune_redundant_variables or sparse_irs_tax_variables_use_puf_irs_predictors or person_native_irs_semantics or derives_tax_input_columns or fallback_employment_excludes_transfer_income'`
  - after reversion: `7 passed`
- Artifacts:
  - source-side candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_cps_pe_agi_donors/broader-donors-cps-pe-agi-v1`
  - export-side candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_pe_export_cps_agi_donors/broader-donors-pe-export-cps-agi-v1`
  - matched incumbent baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_donors/broader-donors-cps-stateage1-v1`
- Read:
  - the source-side version is clearly wrong for the mixed-source Microplex
    pipeline:
    - baseline capped full-oracle loss: `0.7329149849`
    - source-side candidate: `0.9164981002`
    - delta: `+0.1835831153` worse
    - top residual families now included
      `tax_unit_count|domain=tax_exempt_interest_income` and
      `tax_exempt_interest_income|domain=tax_exempt_interest_income`, which is
      a strong sign that the source surface was polluted by estimated leafs too
      early
  - the export-side version is better than the source-side one but still not a
    keeper:
    - export-side candidate: `0.7998451134`
    - delta vs baseline: `+0.0669301285` worse
  - conclusion:
    - do not promote PE-style CPS tax leafs into the source provider
    - do not apply the export-side split by default either
    - the clean alignment boundary for this lane is still unresolved, so the
      default path stays on gross CPS tax aggregates for now

## 2026-04-12 keep donor checkpoint `state x age-band` floor

- Code:
  - keep donor survey checkpoint sampling support for `state_age_floor` in
    `src/microplex_us/data_sources/donor_surveys.py`
  - keep the default checkpoint query builder passing `state_age_floor=1` to
    donor survey providers in
    `src/microplex_us/pipelines/pe_us_data_rebuild_checkpoint.py`
  - keep the new donor sampling/query regressions in
    `tests/test_donor_survey_source_providers.py` and
    `tests/pipelines/test_pe_us_data_rebuild_checkpoint.py`
- Why:
  - after accepting the CPS checkpoint `state x age-band` floor, donor-inclusive
    checkpoints still had an upstream asymmetry: CPS sampling guaranteed
    `state x age` coverage, donor survey sampling only guaranteed a plain state
    floor
  - the next clean test was to mirror the same age-band support floor on donor
    survey checkpoint sampling, but only keep it if the full-oracle metric moved
- Focused verification:
  - `python -m py_compile src/microplex_us/data_sources/donor_surveys.py src/microplex_us/pipelines/pe_us_data_rebuild_checkpoint.py tests/test_donor_survey_source_providers.py tests/pipelines/test_pe_us_data_rebuild_checkpoint.py`
  - `uv run pytest tests/test_donor_survey_source_providers.py tests/pipelines/test_pe_us_data_rebuild_checkpoint.py -q -k 'state_age_floor or default_policyengine_us_data_rebuild_queries or forwards_state_age_floor'`
- Artifacts:
  - baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_stateage1_donors/broader-donors-cps-stateage1-v1`
  - candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_donor_stateage1_donors/broader-donors-donor-stateage1-v1`
- Read:
  - the gain is small but real on the deterministic broader donor benchmark
  - baseline capped full-oracle loss: `0.7329149849`
  - candidate capped full-oracle loss: `0.7327632809`
  - delta: `-0.0001517041`
  - active-solve capped loss also improved slightly: `0.8498782563 -> 0.8495978941`
  - selected constraints stayed flat at `1059`
  - conclusion: keep this as a low-risk checkpoint-default refinement, not as a
    headline methodological change

## 2026-04-12 keep PE-style PUF person-expansion randomness

- Code:
  - `src/microplex_us/data_sources/puf.py`
  - `tests/test_puf_source_provider.py`
  - `artifacts/experiment_index.jsonl`
  - `docs/methodology-ledger.md`
- Why:
  - the PE-demographics branch in Microplex was decoding `_puf_agerange`,
    `_puf_agedp*`, and `_puf_earnsplit` to fixed midpoints, while
    `policyengine-us-data` samples inside those coded bins and also randomizes
    spouse/dependent sex assignment
  - that is a direct upstream parity bug, not a new modeling idea
- Focused verification:
  - `python -m py_compile src/microplex_us/data_sources/puf.py tests/test_puf_source_provider.py`
  - `uv run pytest tests/test_puf_source_provider.py -q -k 'expand_to_persons or sample_tax_units'`
  - `uv run pytest tests/test_puf_source_provider.py -q -k 'not pre_tax_contributions_via_policyengine_subprocess'`
- Artifacts:
  - source-stage parity candidate:
    `artifacts/tmp_puf_source_stage_parity_personexpansion_20260412.json`
  - donor checkpoint:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_donors/broader-donors-puf-personexpansion-v1`
  - no-donor checkpoint:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_nodonors/broader-nodonors-puf-personexpansion-v1`
- Read:
  - raw PUF source-stage parity improved materially on the direct PE boundary:
    - age weighted-mean ratio: `1.0367 -> 1.0275`
    - employment-income weighted-mean ratio: `1.2196 -> 0.9996`
    - taxable-interest weighted-mean ratio: `2.2495 -> 1.1774`
  - matched broader no-donor checkpoint improved on the mission metric:
    - `0.7368409543 -> 0.7336528770`
    - active-solve capped loss: `0.8497778115 -> 0.8005940161`
  - matched broader donor checkpoint regressed slightly on capped full-oracle
    loss while still improving active-solve loss:
    - `0.7327632809 -> 0.7342149723`
    - active-solve capped loss: `0.8495978941 -> 0.8037192584`
  - conclusion:
    - keep the parity fix
    - log the donor-path regression explicitly
    - treat the donor interaction as the next thing to explain, not as a reason
      to restore the old midpoint-decoding bug

## 2026-04-12 split PUF person-expansion parity fix; keep only `EARNSPLIT`

- Code:
  - `src/microplex_us/data_sources/puf.py`
  - `tests/test_puf_source_provider.py`
  - `artifacts/experiment_index.jsonl`
  - `docs/methodology-ledger.md`
- Why:
  - the bundled parity fix was too coarse; it mixed age/sex randomization with
    income-split randomization, and the broader donor checkpoint gave only a
    slightly negative net result
  - the next direct move was a matched ablation, not more speculation
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
  - age/sex-only is clearly harmful on the broader donor frontier:
    - `0.7327632809 -> 0.7463902007`
  - earnsplit-only is clearly beneficial:
    - `0.7327632809 -> 0.7176041064`
    - active-solve capped loss: `0.8495978941 -> 0.7726915403`
  - the real code-path rerun matches the earnsplit-only ablation exactly
  - conclusion:
    - keep PE-style `EARNSPLIT` randomization in the default path
    - revert PE-style age/sex randomization for now
    - treat age-bin randomization as an unresolved parity lane, not a current
      default

## 2026-04-12 widen deferred family focus to 7 after `EARNSPLIT`

- Code:
  - `src/microplex_us/pipelines/pe_us_data_rebuild.py`
  - `tests/pipelines/test_pe_us_data_rebuild.py`
  - `tests/pipelines/test_pe_us_data_rebuild_checkpoint.py`
  - `artifacts/experiment_index.jsonl`
  - `docs/methodology-ledger.md`
- Why:
  - after the accepted `EARNSPLIT` fix, the strongest surviving individual
    rows were ACA PTC and rental tails, but the staged selector was still
    filling its family slots with AGI and EITC pairs
  - the clean test was a one-axis rerun with wider deferred family focus, not
    another ad hoc selector change
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
  - donor run improves on the mission metric:
    - `0.7176041064 -> 0.7044626415`
  - donor-free broader run also improves:
    - `0.7170633141 -> 0.7039665310`
  - the widened focus set includes `aca_ptc` and `rental_income` in both
    deferred passes
  - fresh residual drilldown now shows:
    - ACA/rental mass down sharply
    - remaining mass led again by age, AGI, and EITC families
    - top individual rows still concentrated in ACA amount and eligibility cells
  - conclusion:
    - promote `policyengine_calibration_deferred_stage_top_family_count = 7`
      into the default rebuild policy
    - keep the geography gate at `4`

## 2026-04-12 reject full PUF age/sex randomization again on top of family-7

- Code:
  - `src/microplex_us/data_sources/puf.py` was restored to the earnsplit-only
    default after the retest
  - `tests/test_puf_source_provider.py` was restored to the incumbent
    earnsplit-only regression expectations
  - `artifacts/experiment_index.jsonl`
  - `docs/methodology-ledger.md`
- Verification:
  - `uv run pytest tests/test_puf_source_provider.py -q -k 'expand_to_persons_uses_pe_demographic_helpers_when_present or expand_to_persons_preserves_joint_tax_unit_monetary_totals or expand_to_persons_splits_negative_joint_self_employment_losses or expand_to_persons_clears_status_flags_for_non_head_members'`
- Artifacts:
  - current donor incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1`
  - full-rng retest:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_rng_donors/broader-donors-puf-personexpansion-rng-v1`
- Read:
  - broader donor default still loses with full age/sex randomization:
    - `0.7044626415 -> 0.7111876263`
  - conclusion:
    - keep earnsplit-only PUF person expansion in the default path
    - do not reopen this same parity lane until there is a new interaction
      hypothesis stronger than “try the rejected thing again”

## 2026-04-12 keep CPS tax-unit structure at the source boundary

- implemented source-layer CPS tax-unit role derivation keyed by raw `TAX_ID`
  in `src/microplex_us/data_sources/cps.py`
  - derive:
    - `is_tax_unit_head`
    - `is_tax_unit_spouse`
    - `is_tax_unit_dependent`
    - `tax_unit_is_joint`
    - `tax_unit_count_dependents`
  - added a focused provider regression in
    `tests/test_cps_source_provider.py`
- focused verification:
  - `python -m py_compile src/microplex_us/data_sources/cps.py tests/test_cps_source_provider.py`
  - `uv run pytest tests/test_cps_source_provider.py -q -k 'derives_tax_unit_roles_from_tax_id or caches_household_geography_on_persons or derives_survivor_and_dependent_social_security or loads_observation_frame or canonical_income_alias'`
- artifact comparison:
  - incumbent broader donor default:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1`
  - source-structure rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_taxunit_structure_donors/broader-donors-cps-taxunit-structure-v1`
- read:
  - capped full-oracle loss is exactly unchanged:
    - `0.7044626415 -> 0.7044626415`
  - conclusion:
    - keep this change because it moves CPS tax-unit semantics to the correct
      source boundary and removes downstream reconstruction pressure
    - do not sell it as a frontier gain; it is architecture cleanup

## 2026-04-12 reject direct CPS student flag on the broader donor checkpoint

- tested a narrow EITC-side parity hypothesis:
  - materialize `is_full_time_college_student` directly from CPS `A_HSCOL`
    in the processed CPS cache
- result on the matched broader donor rerun:
  - incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1`
  - student-input rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_cps_student_donors/broader-donors-cps-student-v1`
  - capped full-oracle loss:
    - `0.7044626415 -> 0.7815651801`
- action:
  - reverted the student-field addition in `src/microplex_us/data_sources/cps.py`
    and the temporary student assertions in `tests/test_cps_source_provider.py`
  - reran the focused CPS verification slice after the revert
- interpretation:
  - this is another case where a direct PE CPS input is not automatically
    plug-compatible with the current mixed-source broader Microplex path
  - next upstream work should stay on age/AGI/EITC structure, but not through
    this direct student-field promotion

## 2026-04-12 reject partial preserved tax units as the broader donor default

- implemented a mixed-preservation path in `src/microplex_us/pipelines/us.py`
  - households with complete source `tax_unit_id` values can now keep those IDs
  - unresolved households still fall back to `TaxUnitOptimizer`
  - added a mixed-household regression in `tests/pipelines/test_us.py`
- focused verification:
  - `python -m py_compile src/microplex_us/pipelines/us.py tests/pipelines/test_us.py`
  - `uv run pytest tests/pipelines/test_us.py -q -k 'preserve_existing_tax_unit_ids or falls_back_when_existing_tax_unit_ids_cross_households or partially_preserves_existing_tax_unit_ids'`
- artifact comparison:
  - incumbent broader donor default:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_puf_personexpansion_family7_donors/broader-donors-puf-personexpansion-family7-v1`
  - partial-preservation rerun:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_partial_preserve_taxunits_donors/broader-donors-partial-preserve-taxunits-v1`
- read:
  - capped full-oracle loss regresses slightly:
    - `0.7044626415 -> 0.7055670761`
  - active-solve capped loss improves:
    - `0.7909211525 -> 0.7648463685`
  - conclusion:
    - do not flip the broader default to preserved tax units
    - keep the code path available for future targeted runs, but move the next
      upstream work off this boundary and back to AGI/EITC inputs

## 2026-04-12 keep PE-style CPS `ssn_card_type`

- changed:
  - derive PE-style CPS `ssn_card_type` from raw CPS immigration / benefits /
    work / housing-assistance fields in
    `src/microplex_us/data_sources/cps.py`
  - add mixed-source export support plus `CITIZEN` fallback in
    `src/microplex_us/policyengine/us.py`
  - bump the processed CPS cache version so the new column is materialized in
    rebuilt caches
  - add focused regressions in
    `tests/test_cps_source_provider.py`
    and
    `tests/policyengine/test_us.py`
- verification:
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
  - active-solve capped loss improves:
    - `0.7909211525 -> 0.7813926586`
  - direct `ssn_card_type` family improves sharply:
    - `1.0000 -> 0.3786`
  - EITC child-count families improve:
    - `0.8283 -> 0.7499`
    - `0.8154 -> 0.7408`
  - aggregate `eitc` gets worse:
    - `0.1066 -> 0.2954`
- conclusion:
  - keep it
  - interpret it narrowly as an identification / child-count improvement
    rather than a blanket EITC win

## 2026-04-12 reject PE-style EITC take-up and voluntary filing inputs

- prototyped PE-style `takes_up_eitc` and
  `would_file_taxes_voluntarily` tax-unit inputs in
  `src/microplex_us/pipelines/us.py`, exposed them in
  `src/microplex_us/policyengine/us.py`, and added review-driven fallback and
  determinism checks before the checkpoint
- verification before the run:
  - `python -m py_compile src/microplex_us/pipelines/us.py src/microplex_us/policyengine/us.py tests/pipelines/test_us.py tests/policyengine/test_us.py`
  - `uv run pytest tests/pipelines/test_us.py -q -k 'build_policyengine_entity_tables'`
  - `uv run pytest tests/policyengine/test_us.py -q -k 'default_policyengine_us_export_surface or defaults_missing_ssn_card_type_to_citizen'`
- artifact comparison:
  - incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
  - candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_takeup_donors/broader-donors-takeup-v1`
- metric read:
  - capped full-oracle loss:
    - `0.6955460 -> 0.7041134`
  - active-solve capped loss:
    - `0.7813927 -> 0.7896826`
  - EITC child-count families improved, but aggregate `eitc` worsened:
    - `0.2954 -> 0.4010`
  - ACA amount / count families also worsened:
    - `2.3488 -> 2.5737`
    - `1.1521 -> 1.3708`
- action:
  - revert the take-up / voluntary-filing code path
  - keep `broader-donors-ssn-card-type-v1` as the incumbent broader donor
    runtime
  - do not read this as “drop the concept”; the separation between filing
    propensity and EITC take-up remains a structural requirement, but the
    attempted late export-layer implementation is not good enough yet

## 2026-04-12 reject `state_age_floor = 2` on broader donor checkpoints

- tested a matched broader donor checkpoint with:
  - `cps_state_age_floor = 2`
  - `donor_state_age_floor = 2`
- artifact comparison:
  - incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
  - candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_stateage2_donors/broader-donors-stateage2-v1`
- metric read:
  - capped full-oracle loss:
    - `0.6955460 -> 0.7361964`
  - active-solve capped loss:
    - `0.7813927 -> 0.8371045`
  - age improves slightly:
    - `0.4681 -> 0.4480`
  - but AGI, EITC child-count, and ACA all regress hard enough to dominate
    the frontier:
    - `0.7119 -> 0.7553`
    - `0.6372 -> 0.6618`
    - `0.7499 -> 0.8880`
    - `0.7408 -> 0.8755`
    - `2.3488 -> 2.9982`
- action:
  - reject stronger checkpoint age-floor heuristics
  - keep the accepted floor-1 incumbent
  - move the next experiment to upstream PUF age/AGI construction instead

## 2026-04-12 reject high-AGI-preserving PUF checkpoint samples

- prototyped a checkpoint-only PUF sampler that preserved the top AGI tail
  whenever `sample_n` was active, then ran the matched broader donor checkpoint
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
- action:
  - reject it
  - revert the sampler path completely
  - treat the fast source-stage improvement on dividends / interest as a false
    friend unless it survives the real broader checkpoint

## 2026-04-12 reject standalone ACA take-up construction patch

- traced the ACA residual lane and confirmed that
  `takes_up_aca_if_eligible` is a real PE construction input, not a made-up
  Microplex feature
- implemented the narrow probe in `src/microplex_us/pipelines/us.py` and
  exposed it in `src/microplex_us/policyengine/us.py`, then verified the local
  code path with focused `py_compile` and pytest slices
- because disk pressure made a fresh broader rerun unreliable, reevaluated the
  incumbent broader donor synthetic population in memory against the shared
  oracle and saved the readout in
  `artifacts/tmp_broader_aca_takeup_recalibration_20260412.json`
- read:
  - capped full-oracle loss:
    - `0.6955460 -> 0.8211989`
  - active-solve capped loss:
    - `0.7813927 -> 0.7013644`
  - ACA families improve sharply:
    - `aca_ptc|domain=aca_ptc`
    - `2.3488 -> 0.5529`
    - `tax_unit_count|domain=aca_ptc`
    - `1.1521 -> 0.7112`
    - `person_count|domain=aca_ptc,is_aca_ptc_eligible`
    - `1.0994 -> 0.7771`
- action:
  - revert the patch from the default path
  - keep the concept documented as required future parity work
  - interpret this as “wrong implementation boundary right now,” not “wrong
    concept”

## 2026-04-12 ACA child gap is mostly Medicaid crowd-out, not missing ACA knobs

- ACA-specific review conclusion:
  - beyond raw `has_marketplace_health_coverage` / `has_esi`, the only real
    ACA-specific upstream input is `takes_up_aca_if_eligible`
  - so there is no large hidden ACA-specific construction surface still
    missing from Microplex
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
  - the key driver is much lower child-unit `medicaid_income_level` in the
    incumbent:
    - median under-20 `medicaid_income_level`:
      `15.1512 -> 1.6054`
    - p75 under-20 `medicaid_income_level`:
      `364.3831 -> 3.9464`
  - filing-status mix is not the main failure mode; child tax units are simply
    too low-income relative to the PE baseline
- action:
  - move the next lane to AGI / tax-unit construction and imputation for child
    units
  - stop treating ACA as primarily an ACA-specific export/input problem

## 2026-04-13 child-unit income miss is already present before synthesis

- stage-localized the incumbent broader donor artifact by comparing
  `seed_data.parquet`, `synthetic_data.parquet`, and `calibrated_data.parquet`
  on under-20 tax-unit income aggregates
- read:
  - `seed` and `synthetic` are effectively identical on the child-unit income
    surface:
    - weighted mean under-20 tax-unit income:
      `110304.6 -> 110304.6`
    - weighted mean under-20 tax-unit employment income:
      `68829.3 -> 68829.3`
  - calibration only nudges those values:
    - weighted mean under-20 tax-unit income:
      `110304.6 -> 108967.8`
    - weighted mean under-20 tax-unit employment income:
      `68829.3 -> 65923.5`
- action:
  - treat the current child-unit AGI / Medicaid-income miss as entering in the
    seeded integrated microdata before synthesis
  - keep the next debugging lane on upstream construction / source-impute
    parity rather than calibration

## 2026-04-13 reject source tax-unit preservation as the broader donor default

- tested:
  - flipped `policyengine_prefer_existing_tax_unit_ids=True` only in the
    canonical PE rebuild default
  - left the generic build-config default unchanged
  - ran the focused rebuild/checkpoint config tests
  - got an explorer review; no concrete code-level regressions were identified
- synthetic proxy read:
  - preserving source tax-unit IDs still looked slightly better on the cached
    synthetic-policyengine comparison:
    - `0.63654 -> 0.63583`
- real decision run:
  - incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
  - candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_preserve_taxunits_default_donors/broader-donors-preserve-taxunits-default-v1`
- read:
  - capped full-oracle loss regresses slightly:
    - `0.6955 -> 0.6977`
  - active-solve capped loss improves:
    - `0.7814 -> 0.7624`
  - selected constraints fall slightly:
    - `1031 -> 1019`
- action:
  - reverted the default flip in `src/microplex_us/pipelines/pe_us_data_rebuild.py`
    and the matching config assertions in the rebuild/checkpoint tests
  - kept the optional preservation path available in `src/microplex_us/pipelines/us.py`
- interpretation:
  - the structural clue is still real, but the broader donor frontier metric
    does not justify making this the default rebuild path yet
  - keep the next lane on upstream child-unit AGI / Medicaid-income
    construction and source-impute parity

## 2026-04-13 reject minor-household source tax-unit preservation

- tested:
  - added an opt-in experiment flag that preserved source `tax_unit_id` values
    only for households containing a minor and left adult-only households on
    the optimizer rebuild path
  - added a focused preservation regression in `tests/pipelines/test_us.py`
- real decision run:
  - incumbent:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
  - candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_minorhousehold_preserve_taxunits_donors/broader-donors-minorhousehold-preserve-taxunits-v1`
- read:
  - the child symptom improves sharply:
    - under-20 singleton-tax-unit share:
      `0.1538 -> 0.0345`
    - under-20 mean `medicaid_income_level`:
      `2.7279 -> 3.0408`
  - but the broader donor frontier metric still regresses:
    - capped full-oracle loss:
      `0.6955 -> 0.6985`
    - active-solve capped loss:
      `0.7814 -> 0.7614`
- action:
  - reverted the experiment flag and its targeted test
- interpretation:
  - preserving child tax-unit structure helps, but it is not the main blocker
    anymore
  - the next upstream lane has to be AGI component construction for child-linked
    tax units

## 2026-04-13 under-20 AGI miss is now clearly a component-construction problem

- compared PE baseline, the incumbent broader donor artifact, and the rejected
  minor-household-preservation rerun on person-mapped under-20 tax-unit
  aggregates
- read:
  - under-20 mapped AGI / Medicaid MAGI improve with the rejected structure
    probe, but remain far below the PE baseline:
    - `adjusted_gross_income`:
      `137623.5` (PE) vs `85755.2` (incumbent) vs `98230.0` (minor-preserve)
    - `medicaid_magi`:
      `140533.9` (PE) vs `86338.8` (incumbent) vs `98586.5` (minor-preserve)
  - the remaining miss is in AGI composition:
    - `tax_unit_partnership_s_corp_income` stays far too low:
      `23323.0` (PE) vs `9568.7` vs `10710.1`
    - `net_capital_gains` stays far too low:
      `3200.0` (PE) vs `534.3` vs `945.7`
    - `qualified_dividend_income` remains zero in both Microplex artifacts
    - `tax_exempt_interest_income` remains zero in both Microplex artifacts
- action:
  - move the next direct-path work off tax-unit-preservation variants and onto
    AGI component construction / source-impute parity for child-linked units

## 2026-04-13 reject PE-style sequential PUF joint-QRF imputation in the current donor runtime

- tested:
  - added a non-default `sequential_qrf` donor-imputer backend for the main PUF
    AGI leaf lane and grouped the key tax variables into one joint block when
    that backend was selected
  - added focused regressions, verified the challenger path locally, then ran
    matched medium and broader donor checkpoints
- real decision runs:
  - medium candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_sequential_puf_joint_medium/medium-donors-sequential-puf-joint-v1`
  - broader candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_sequential_puf_joint_donors/broader-donors-sequential-puf-joint-v1`
  - incumbent baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
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
    challenger changes child-linked AGI composition aggressively rather than
    cleanly fixing the miss:
    - under-20 linked `qualified_dividend_income`:
      `40.0 -> 1199.0`
    - under-20 linked `taxable_interest_income`:
      `507.2 -> 1634.6`
    - under-20 linked `tax_exempt_interest_income`:
      `4.66 -> 249.4`
    - under-20 linked `taxable_pension_income`:
      `9118.5 -> 19317.6`
- action:
  - rejected the challenger, reverted the experiment code, and kept the
    incumbent donor-impute backend
- interpretation:
  - the parity clue is still useful because PolicyEngine really does use a more
    joint QRF architecture for this lane
  - but the direct port into the current donor/rank-match runtime is not
    numerically safe enough to keep
  - the next lane remains narrower AGI component construction / source-impute
    parity for child-linked tax units, not a backend replacement

## 2026-04-13 reject post-donor zeroing of PUF tax leaves on dependent rows

- tested:
  - added a post-donor semantic guard that zeroed selected PE-style PUF tax
    leaves on rows with `is_tax_unit_dependent > 0`
  - rationale: raw expanded PUF dependents already carry zero for these leaves,
    while the incumbent broader donor seed artifact was assigning large
    dependent-row mass on `partnership_s_corp_income`,
    `taxable_pension_income`, and `taxable_interest_income`
- local diagnostic read:
  - the guard did what it was intended to do on the incumbent seed artifact:
    - under-20 `partnership_s_corp_income`:
      `4.09M -> 87.3k`
    - under-20 `taxable_pension_income`:
      `17.77M -> 172.6k`
    - under-20 `taxable_interest_income`:
      `33.98k -> 3.28k`
- real decision run:
  - candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_dependent_zero_tax_leaves_donors/broader-donors-dependent-zero-tax-leaves-v1`
  - incumbent baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
- read:
  - the broader donor frontier metric regresses badly:
    - capped full-oracle loss:
      `0.6955 -> 1.1372`
    - active-solve capped loss:
      `0.7814 -> 1.6581`
  - the run starts from a much worse first calibration stage:
    - post-stage-1 capped full-oracle loss:
      `1.3660`
  - deferred stages improve that bad candidate but do not rescue it:
    - post-stage-2 capped full-oracle loss:
      `1.2460`
    - final capped full-oracle loss:
      `1.1372`
- action:
  - rejected the guard and reverted the code
- interpretation:
  - the structural clue is still useful because the dependent-row mass is being
    created during donor integration, not in raw PUF expansion
  - but blunt post-donor zeroing is the wrong repair and should not stay in the
    default path
  - the next lane remains narrower donor-impute/source-impute parity for these
    child-linked tax leaves

## 2026-04-13 reject dependent-role partitioning inside donor imputation

- tested:
  - added an exact-match partition on `is_tax_unit_dependent` for the three PUF
    leaves that were actually exploding on child-linked rows:
    `partnership_s_corp_income`, `taxable_pension_income`,
    `taxable_interest_income`
  - rationale: move the repair to the actual failure point inside donor
    imputation, instead of zeroing rows after integration
- real decision run:
  - candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_dependent_partition_tax_leaves_donors/broader-donors-dependent-partition-tax-leaves-v1`
  - incumbent baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
- read:
  - the broader donor frontier metric regresses even more:
    - capped full-oracle loss:
      `0.6955 -> 1.2406`
    - active-solve capped loss:
      `0.7814 -> 1.6943`
  - the child-dependent mass is strongly suppressed, but that still does not
    help the shared objective:
    - under-20 `partnership_s_corp_income`:
      `74.5k`
    - under-20 `taxable_pension_income`:
      `257.4k`
    - under-20 `taxable_interest_income`:
      `3.33k`
- review:
  - an independent review also found correctness risks in the partition
    implementation:
    - null partition keys would fall through to a global donor fallback
    - projected partition labels were lossy after entity projection
    - empty donor partitions silently disabled exact-match isolation
- action:
  - rejected the experiment and reverted the code
- interpretation:
  - the failure point is still donor integration
  - but role-suppression heuristics, even inside donor fitting/matching, are not
    the right repair
  - the next lane should move closer to PE source-impute structure for these AGI
    leaves rather than adding more support heuristics

## 2026-04-13 reject richer singleton condition surfaces for the PUF child-linked tax leaves

- tested:
  - expanded the preferred donor-condition surface for
    `partnership_s_corp_income`, `taxable_interest_income`, and
    `taxable_pension_income` beyond the PE-style demographic predictors to also
    use current income state
  - kept the current donor backend and singleton block structure unchanged
  - added focused regressions that the richer predictors resolved only for these
    leaves and that `income` was actually added to the resolved condition set
    when available
- verification:
  - `python -m py_compile src/microplex_us/variables.py tests/test_variables.py tests/pipelines/test_us.py`
  - `uv run pytest tests/test_variables.py tests/pipelines/test_us.py -q -k 'puf_irs_predictors or pe_style_puf_predictors_for_generic_irs_vars or donor_imputation_block_specs or augment_donor_condition_frame_for_targets_derives_pe_style_puf_predictors'`
- real decision run:
  - candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_income_aware_puf_tax_leaves_donors/broader-donors-income-aware-puf-tax-leaves-v1`
  - incumbent baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
- read:
  - the broader donor frontier metric regresses:
    - capped full-oracle loss:
      `0.6955 -> 0.7420`
    - active-solve capped loss:
      `0.7814 -> 0.8499`
    - selected constraints:
      `1031 -> 1027`
  - the candidate does improve across deferred stages, but never catches the
    incumbent:
    - post-stage-1 capped full-oracle loss:
      `0.8326`
    - post-stage-2 capped full-oracle loss:
      `0.7879`
    - final capped full-oracle loss:
      `0.7420`
- PE code read:
  - this explains why the shortcut loses: PolicyEngine does not solve these
    leaves with richer singleton donor surfaces
  - they live inside one sequential PUF QRF pass, with only
    `taxable_pension_income` also touching the separate ACS donor path
- action:
  - rejected the richer singleton condition-surface patch and reverted the code
- interpretation:
  - widening singleton condition surfaces is still the wrong abstraction for
    this lane
  - local code read confirms these are PUF-native leaves entering the build
    through the PUF provider before the donor-survey sources, not current
    explicit direct-override variables
  - the next step should move toward the real structure gap in how PUF tax
    leaves enter the build, not pile more predictors onto the generic donor path

## 2026-04-13 reject standalone PUF-native QRF hook for three child-linked AGI leaves

- tested:
  - added a temporary PUF-provider QRF hook at tax-unit load time for
    `partnership_s_corp_income`, `taxable_interest_income`, and
    `taxable_pension_income`
  - kept the rest of the donor integration and calibration path unchanged
- verification:
  - focused `py_compile` passed
  - focused `tests/test_puf_source_provider.py` slices passed before the real
    rerun
- real decision run:
  - candidate:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260413_puf_tax_leaf_qrf_donors/broader-donors-puf-tax-leaf-qrf-v1`
  - incumbent baseline:
    `artifacts/live_pe_us_data_rebuild_checkpoint_20260412_broader_ssn_card_type_donors/broader-donors-ssn-card-type-v1`
- read:
  - the broader donor frontier metric regresses hard:
    - capped full-oracle loss:
      `0.6955 -> 0.8729`
    - active-solve capped loss:
      `0.7814 -> 1.1545`
    - selected constraints:
      `1031 -> 1064`
- action:
  - rejected the provider-hook experiment and reverted the code
- interpretation:
  - the right lesson is not “more QRF earlier”
  - a standalone PUF-side QRF hook, without the rest of PolicyEngine’s
    sequential clone/impute structure, is still the wrong runtime shape for
    this lane
