# PUF handling: Enhanced CPS vs Microplex

How each system uses the IRS SOI PUF — what it strips, what it imputes, what
predictors it conditions on, and how "clones" work. Sourced from a file-level
read of both repos (2026-05-29). Citations are `file:line`.

## TL;DR

- **Neither system inserts raw PUF rows into the released data.** Both use PUF as
  **QRF training data** to impute tax variables onto CPS-based households.
- **The imputation method is nearly identical**: quantile regression forest (QRF)
  conditioned on the **same 7 demographic predictors**.
- **The real difference is the household population:**
  - **eCPS** = CPS **doubled**. Half the rows are CPS-as-reported; the other half
    are "PUF clones" = the same CPS demographic skeletons carrying QRF-imputed PUF
    tax variables, starting at **zero weight**, then L0-reweighted. 100% CPS-derived,
    tax-focused.
  - **Microplex** = **CPS + ACS** spines, each with a generic donor-support clone
    (4 groups, ~311k households), PUF donor-imputed onto all of them. **~⅔ of the
    pool is ACS** (200k of 311k), which carries geography/demographics, not the
    tax detail the PE-native targets need.
- **Implication:** at matched record budget, mp spends most of its rows on ACS
  (tax-poor) while eCPS spends all of them on CPS+PUF-flavored rows. That dilution
  is a strong candidate for *why* mp trails eCPS on the IRS-SOI-heavy target estate.
  **Dropping ACS → CPS + PUF-clone (eCPS-shaped) is well-motivated for the national
  milestone**; re-add ACS later for local/subnational work.
- **Surprise:** mp's supposed imputation edge — a −/0/+ "regime-aware" sign
  classifier before QRF — is **implemented but not the production default** (the
  rebuild config uses plain `qrf`, and the installed `microimpute==1.1.2` can't even
  import the `zero_inflated` module). The May-16 candidate's build dir is named
  `...regime_aware...`, so that build *may* have used it, but it is in flux, not a
  settled advantage.

---

## Two ways eCPS uses PUF (confirmed in `puf_impute.py`)

1. **Imputed onto the ORIGINAL CPS rows** — the `OVERRIDDEN_IMPUTED_VARIABLES`
   list (47 vars) is QRF-predicted and *overwrites* the CPS value on every record.
   The PUF-authoritative / not-reliably-on-CPS set: **partnership_s_corp_income**,
   QBI inputs (w2_wages, UBIA, sstb), deductible_mortgage_interest, charitable,
   foreign_tax_credit, savers_credit, student_loan_interest, many credits/ALDs.
2. **Stripped PUF-trained clone** — the doubled half has *all 68* `IMPUTED_VARIABLES`
   replaced by PUF-QRF, keeping only demographics + IDs.

The ~21 vars imputed-but-NOT-overridden (employment_income, social_security,
pensions, interest, dividends, capital gains, rental, IRA distributions) are the
items CPS reports well → original CPS keeps them; only the clone gets the PUF version.

## Sources vs population rows (don't conflate)

eCPS is **not** CPS+PUF only. `policyengine_us_data/datasets/` = `acs, cps, forbes,
org, puf, scf, sipp`. It imputes from many sources as **donors**: ACS (residence
value/property), SCF (net worth), SIPP (assets), CPS-ORG (wages), Forbes (top tail),
PUF (tax). But its household **rows** are all CPS-derived (CPS + CPS clones).
Microplex has the **same donor families** (`ACSSourceProvider`, `SIPPSourceProvider`
+ Tips/Assets, `SCFSourceProvider`). So the two don't differ on *which sources* —
they differ on **donor** (impute onto CPS rows; eCPS's approach for everything but
CPS) vs **spine** (own rows in the population). The tested mp build's problem was
making **ACS a spine** (+200k ACS rows), not a missing source. (Earlier note that
"eCPS is CPS+PUF / mp should add SCF-SIPP" was wrong — both already have them.)

## Enhanced CPS (policyengine-us-data)

### Clone mechanics — "PUF clones" are doubled CPS rows
- `puf_clone_dataset()` (`calibration/puf_impute.py:158,239-306`) **doubles** the CPS:
  first half = original CPS households; second half = "PUF copies."
- Every variable concatenated `[first_half, second_half]`. `*_id` offset to stay
  unique; **`*_weight` second half × 0** (`puf_impute.py:253-256`) → PUF clones start
  at zero weight and only earn weight from L0 calibration.
- PUF-clone geography **mirrors its CPS source row** (`state_fips`, `block_geoid`,
  `cd_geoid`, `county_fips`, `tract_geoid` concatenated `[x,x]`, `puf_impute.py:266-289`).
- The "50/50 CPS vs PUF-clone split" is a **calibration target on weighted totals**,
  added by **PR #1150 (not shipped)** — `household_is_puf_clone`,
  `PUF_CLONE_HOUSEHOLD_COUNT_TARGET_SHARE`, ±0.10 guard.
- A separate local-area orchestrator (`unified_calibration.py`) clones **10×** via
  `assign_random_geography(n_clones=10)`; classic national eCPS uses the 2× ExtendedCPS path.

### What is PUF-actual vs imputed on the clones
- The PUF-clone half does **not** carry the filer's real PUF dollars. The 68
  `IMPUTED_VARIABLES` on the PUF half = **QRF predictions**; raw PUF micro-records
  are only **training data**. So a PUF clone = "CPS skeleton + QRF-imputed PUF tax
  vars @ weight 0."

### Method + predictors
- **QRF** (`microimpute.models.qrf.QRF` → `RandomForestQuantileRegressor`).
- **Predictors** = `DEMOGRAPHIC_PREDICTORS` (`puf_impute.py:25-33`):
  `age, is_male, tax_unit_is_joint, tax_unit_count_dependents, is_tax_unit_head,
  is_tax_unit_spouse, is_tax_unit_dependent`. (Geography deliberately removed as a
  predictor.)
- **Imputed outputs** = 68 `IMPUTED_VARIABLES` (`puf_impute.py:35-103`): employment_income,
  long/short_term_capital_gains, qualified/non-qualified dividends, partnership_s_corp_income,
  self_employment_income, taxable_pension_income, taxable_ira_distributions, social_security,
  rental_income, farm_income, deductible_mortgage_interest, charitable_*, QBI fields
  (w2_wages_from_qualified_business, unadjusted_basis_qualified_property, business_is_sstb,
  qualified_reit_and_ptp_income), plus credits/ALDs.
- Batched **10 vars/QRF model** (`puf_impute.py:526-592`); `weeks_unemployed` is a
  downstream QRF conditioned on imputed UC. **No covariance/copula restoration.**

### What is stripped / excluded
- `MARS==0` aggregate record dropped (`puf.py:564`).
- AGI (`E00100`) **not carried** — recomputed (`puf.py:313,466`).
- Only `FINANCIAL_SUBSET` columns survive; unmapped raw E-codes dropped.
- Dependents capped at 3/unit (`puf.py:627`).
- QRF training subsample ~20k, **but top 0.5% AGI preserved** (`PUF_TOP_PERCENTILE=99.5`).
- **[PR #1150 only, not shipped]** Forbes-record exclusion + $10M-AGI/$10M-component
  top-tail donor exclusion.

---

## Microplex (microplex-us)

### PUF role — donor only, ZERO PUF rows
- `PUFSourceProvider.shareability = Shareability.RESTRICTED` (`data_sources/puf.py:2380`).
- Export gate (`pipelines/us.py:803-805`): a source can be a household spine/clone only
  if `shareability is Shareability.PUBLIC`; RESTRICTED PUF fails, as does the
  `tax_microdata` / tax-unit population check (`us.py:789-795`). Enforced before
  appending spines (`us.py:4162`) and support clones (`us.py:4310`).
- "donor_attribute_support_clone" rows are **copies of the CPS/ACS scaffold**
  (`_build_donor_support_clone_seed`: `clone = seed_data.copy()`, `us.py:4410`), **not PUF**.
- Confirmed empirically: `source_spine_composition.json` shows spines `acs_pums` + `cps_asec`
  only (4 groups, 311,524 hh) — **no `irs_soi_puf` group**.

### Method + predictors
- Two backends (`us.py:7566-7613`): **`qrf`** (production — `pe_us_data_rebuild.py:88`) and
  **`regime_aware`** (the −/0/+ sign classifier).
- **`regime_aware`** (`pipelines/donor_imputers.py:114-239`): per target, a
  `ZeroInflatedImputer` auto-detects one of 7 regimes (THREE_SIGN / ZI_POSITIVE /
  ZI_NEGATIVE / SIGN_ONLY / …); a gate classifier (default `hist_gb`) predicts the
  sign class (neg/zero/pos) and routes each row to a sign-specific QRF, so predictions
  can't land in the dead band between max(neg) and min(pos) and negatives survive.
  **Caveat: not the production default, and `microimpute==1.1.2` lacks the
  `zero_inflated` module → not runnable in this checkout.**
- **`qrf`** (production): plain QRF with a random per-row quantile draw (0.05–0.95) —
  essentially the same as eCPS.
- **Predictors** = `PUF_IRS_TAX_PREFERRED_CONDITION_VARS` (`variables.py:76-84`):
  `age, is_male, tax_unit_is_joint, tax_unit_count_dependents, is_tax_unit_head,
  is_tax_unit_spouse, is_tax_unit_dependent` — **identical to eCPS**. Optional
  "challenger" widenings (dividends/pension/partnership add a few income predictors)
  behind a flag.
- **Imputed outputs** (`variables.py:321-738`): dividend_income, qualified/non_qualified
  dividends, taxable/tax_exempt interest, long_term_capital_gains, taxable/tax_exempt
  pension, taxable_ira_distributions, traditional/roth IRA contributions, taxable_social_security,
  student_loan_interest, HSA/SEHI/SEP ALDs, partnership_s_corp_income, plus tax-unit
  deduction lines (SALT, real_estate_tax, mortgage_interest, charitable, interest/IRA
  deduction). Three-sign (−/0/+) targets: long_term_capital_gains, capital_gains,
  partnership_s_corp_income.

### What is stripped / excluded
- `MARS != 0` filter drops the SOI aggregate record (`data_sources/puf.py:451-452`).
- 16 `PUF_CALCULATED_TAX_OUTPUT_VARIABLES` marked `usable_as_condition=False` — kept out
  of model inputs (`source_registry.py:55-115`, `microdata_roles.py:30-49`).
- Geography/tenure/income/employment_status marked non-authoritative scaffold filler
  for PUF (`source_registry.py:69-96`).
- Demographic helper cols dropped after person expansion (`puf.py:262-270`).
- **No Forbes/top-tail exclusion** — the opposite: `DEFAULT_PUF_TAIL_PRESERVE_RAW_COLUMNS`
  + `_select_tail_preserved_tax_units` (`puf.py:145-159,2158`) preserve the tail.
- Roth synthesized from traditional IRA × 39/25; medical split by fixed fractions; SOI uprating.

---

## Side-by-side

| | Enhanced CPS | Microplex (production) |
|---|---|---|
| Raw PUF rows in output | No | No (gated by `Shareability.PUBLIC`) |
| "Clone" rows | CPS doubled; 2nd half = PUF-flavored, weight 0 | CPS & ACS scaffold copies (donor-support) |
| PUF role | QRF training data | QRF training data (donor) |
| Imputation | QRF, per-quantile | QRF, per-quantile (`qrf`); −/0/+ regime-aware exists but off |
| Predictors | 7 demographics | **same** 7 demographics |
| Population | ~100% CPS-derived | CPS (~36%) + **ACS (~64%)** |
| Geography | block assigned to CPS; clone mirrors | source-or-assigned per spine |
| Tail | top 0.5% AGI kept | tail preserved |

## Implications for the build direction

1. **Drop ACS for the national milestone — well-motivated.** ACS is ~⅔ of mp's pool
   and contributes geography, not the income/tax detail that dominates the PE-native
   broad targets (IRS SOI). eCPS is all-CPS+PUF-flavored. Matching that shape
   (CPS + PUF-imputed CPS clone, no ACS) removes the dilution and is the apples-to-apples
   way to "at least match eCPS." Re-add ACS for local/subnational (Phase 5), where it earns its keep.
2. **"PUF clones": mp already has the functional equivalent** (donor-support clones carry
   PUF-imputed tax vars), but it isn't *PUF-targeted* the way eCPS's 1:1 CPS doubling is,
   and it's split across ACS. To mirror eCPS: double the CPS spine, impute PUF tax vars
   onto the copy, start at zero weight, L0-reweight.
3. **No imputation edge in the shipped build.** The −/0/+ regime-aware classifier is the
   one genuine methodological differentiator, but production uses plain `qrf` and the env
   can't run regime_aware. If we believe the sign classifier helps (it should, for
   −/0/+ vars like cap gains / partnership income), getting it *actually wired and
   runnable* is a concrete lever — and a fair comparison must state which imputer the
   candidate used.

## Caveats
- eCPS Forbes/$10M exclusions are **PR #1150 (in-flight)**, not shipped.
- mp's `regime_aware` is wired but not the default and not runnable under `microimpute==1.1.2`;
  the May-16 candidate's build dir is named `regime_aware`, so it may have used an earlier
  microimpute — verify per-artifact before claiming which imputer produced a given candidate.
- Did not execute either build; this is a read of wiring + specs, not a runtime trace.
