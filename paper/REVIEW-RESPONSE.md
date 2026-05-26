# Consolidated referee review and revision plan

*Five subagent referee reviews ran in parallel on 2026-04-17 evening on the paper scaffold. This doc synthesizes their findings into an ordered revision plan.*

## Reviewer verdicts

| Reviewer | Verdict | Main issue |
|---|---|---|
| Citation | Minor revisions | Synthcity author mismatch; identity-preservation framing overstated vs Dekkers 2015 |
| Methodology | Major revisions | Single-seed, non-converged calibration presented as final, correlated "robustness checks" |
| Domain | Major revisions | 36 "target columns" are inputs not policy outputs; ecosystem under-represented |
| Stylistic | Major revisions | 4 of 7 body sections are stubs; solo-authored "we"; documentation register |
| Reproducibility | Major revisions | No code/data availability statement; 2 of 4 robustness checks used pre-snap data |

Four of five reviewers reach Major Revisions. The draft is not submittable in its current state but is recoverable within 1–2 weeks of focused work.

## Critical findings (blocker before submission)

### B1. Two "independent robustness checks" used the pre-snap broken pipeline [RESOLVED]

The reproducibility reviewer identified that `artifacts/embedding_prdc_compare.json` (Apr 17 08:03) and `artifacts/calibrate_on_synthesizer.json` (Apr 17 08:06) predated the snap fixes (harness-side at 12:06, upstream-core at 12:20). Both scripts called `method.fit` and `method.generate` directly without invoking `_snap_categorical_shared_cols`.

**Resolution (2026-04-17 21:15/21:17)**: both scripts were re-run against the post-fix upstream `microplex` (commit `81a5e10`, "Only smooth-noise continuous shared cols, not categorical ones"). The pre-fix artifacts were preserved with a `.pre-snap.json` suffix for audit; the post-fix artifacts replaced the original `.json` filenames. Comparison:

| Artifact | Pre-snap coverage (ZI-QRF, 40k raw) | Post-snap coverage (ZI-QRF, 40k raw) |
|---|---:|---:|
| `embedding_prdc_compare.json` | 0.348 | 0.982 |
| `calibrate_on_synthesizer.json` | pre-cal rel-err 0.256 | pre-cal rel-err 0.317, post-cal 0.105 |

Ordering is preserved (ZI-QRF > ZI-QDNN > ZI-MAF) under both regimes; absolute post-snap numbers are the ones reported in §5. Paper text at lines 252–268 already references the post-snap artifacts.

### B2. The 36 "target columns" are input variables, not policy outputs

The domain reviewer's single most important finding: the paper uses `employment_income_last_year`, `snap_reported`, `ssi_reported`, etc. — CPS-reported amounts — as "targets." A tax-microsim reviewer expects "targets" to mean policy outputs: federal income tax liability, state income tax, computed EITC/CTC, SNAP benefits under program rules, SSI amounts.

Two options:

- **Rename**. Call them "conditioning income and benefit columns" or "target income components." Do this at minimum; the current language is misleading.
- **Add downstream validation**. Run `policyengine-us` (and/or TAXSIM, Tax-Calculator, TPC — whichever the reviewer population cares about most) on microplex-us output data and report computed federal tax, EITC disbursed, CTC disbursed, SNAP/SSI/ACA PTC aggregates against external benchmarks (IRS SOI tables, USDA SNAP totals, SSA SSI totals, CBO SNAP outlays). This is the test a tax-microsim reviewer actually wants.

Recommendation: do both. Rename immediately; add the downstream validation as a major new results subsection.

### B3. Four of seven body sections are stubs

Architecture (§3), Methods (§4), rare-cell subsection (§5.3), Discussion (§6), Conclusion (§8) are either parenthetical placeholders or explicit TBD. Not submittable in this state.

**Action**: work through these in order. Methods first (reviewer can't evaluate anything else until they know what was done). Architecture second. Results-rare-cell third. Discussion and Conclusion last.

### B4. No Code and Data Availability statement

Standard requirement at every target venue. Must state data source (HuggingFace URL with pinned revision), code repository, software versions, Python version, OS tested, hardware, expected wall time, license.

**Action**: add `## Code and Data Availability` section after Limitations. One paragraph.

### B5. Conflicts of Interest disclosure missing

Author founded PolicyEngine and previously led Enhanced CPS work (cited extensively in this paper). The `AFFILIATION.md` rule is followed in the byline and acknowledgments, but silence on the prior affiliation is a disclosure gap. Per domain reviewer: "Silence on the question will read worse than acknowledgement."

**Action**: add explicit COI statement. Template: "The author founded PolicyEngine and previously led work on Enhanced CPS [@ghenis2024ecps]. The present work is conducted at PolicyEngine, the non-profit organization that publishes Enhanced CPS. PolicyEngine's Enhanced CPS is cited as the incumbent public tool against which microplex-us is measured."

## High-priority revisions (before review circulation)

### H1. Convert first-person plural to first-person singular (or third-person)

Solo-authored paper uses "we" throughout both documents. Per the project's global style rule and the target venues' conventions, this should be "I" or third-person recast. The stylistic reviewer identified ~20 instances needing judgment-based conversion (global find-and-replace won't work).

### H2. Self-contain the Related Work section

Line 56 of `index.qmd` says "A full literature review for this paper is maintained in `literature-review.qmd`." This is a documentation move, not an academic one. Self-contain §2 with 400–600 words of prose. Keep `literature-review.qmd` as supplementary material.

### H3. Remove all documentation-register artifacts

- `*(This section is being written against the spec-based-ecps-rewire branch...)*` — convert to outline-as-prose.
- `[report low]` editorial marker at line ~100 — resolve.
- `77,006 × 50 scale` — rewrite as "77,006 records across 50 columns."
- "keeps every record alive" — "preserves all records" or "retains positive weight on every record."
- "mainline" — "primary calibration mechanism."
- Artifact paths referenced in body text — remove.

### H4. Tables need captions, numbers, cross-reference labels

All three tables are bare Markdown pipe-tables with no caption, no number, no Quarto `{#tbl-...}` label. Required for IJM / NTJ / JASA.

### H5. Add at least one figure

Pipeline schematic (source providers → donor blocks → chained QRF → calibration → L0 post-step) is the obvious first figure. Methods papers at the target tier with zero figures are unusual.

### H6. Quantify or soften "widely-used upstream benchmark base class"

Abstract claims the noise-injection defect "systematically biased earlier synthesizer comparisons." Evidence cited is one pre/post table on three methods using one base class. Either name the affected published benchmarks or soften to "introduced systematic bias into synthesizer comparisons using this base class."

### H7. Citation form consistency

Audit every `[@key]` vs `@key` for correct parenthetical vs textual intent. Pandoc renders them differently.

## Medium-priority revisions (quality improvements)

### M1. Uncertainty quantification

Every headline table is a single-seed point estimate. Methodology reviewer correctly notes this is weak for a methods paper. ZI-QRF runs in 37 seconds — running 5-10 seeds is trivial compute. Report means with standard errors, or at least ordering-stability counts ("ordering preserved in 10/10 seeds").

### M2. Rerun with calibration converged

All three entries in `artifacts/calibrate_on_synthesizer.json` have `"calibration_converged": false` at 200 epochs. The docs acknowledge this; the paper does not. Rerun at 1000-2000 epochs or report the epoch budget and frame as "fraction of pre-cal gap closed" rather than absolute post-cal error.

### M3. Formal definition of identity preservation

Currently asserted as an architectural property but never defined. Add Definition 1 in §3: *A weight-adjustment procedure $\phi: w \to w'$ is identity-preserving if $\forall i: w_i' > 0$ and $\phi$ does not drop records.* Either cite that `microcalibrate`'s gradient step satisfies this, or prove it.

### M4. Embedding-PRDC circularity

Autoencoder is fit on holdout only. Potential bias toward methods that match holdout idiosyncrasies. Re-run with AE fit on train (or an independent third partition). Report both.

### M5. Soften "novel to PolicyEngine" Forbes claim

Domain reviewer identified the SCF + Forbes precedent: Bricker-Henriques-Hansen-Moore (2016), Vermeulen (2018), Kennickell (2019). The tax-microsim integration remains novel; the broader pattern has precedent. Rewrite: "While top-wealth augmentation from Forbes-style lists is established practice in distributional national accounts [cites], its integration into a production tax-microsim pipeline is to our knowledge first done in policyengine-us-data."

### M6. Cross-sectional motivation for identity preservation

Domain reviewer: "Identity preservation also matters cross-sectionally for interpretability, subgroup analysis, confidentiality auditing, reproducibility and provenance." Add two paragraphs in Discussion making the cross-section case alongside the longitudinal case.

### M7. ZI-QRF substrate circularity

ECPS itself is QRF-constructed. ZI-QRF's win may be partly method-substrate match. Either add a non-ECPS robustness check (raw CPS ASEC or SCF) or explicitly note the circularity as a limitation.

### M8. Target-set expansion

Add Medicaid/CHIP, ACA PTC, mortgage interest, charitable contributions, medical expenses, property tax. Rerun at the expanded target set.

### M9. Snap heuristic cardinality guard

Stylistic and methodology reviewers flag that `_snap_categorical_shared_cols` fires on any integer-valued column, which could accidentally snap continuous-but-rounded columns (currency stored in dollars). Add cardinality threshold (e.g., snap only when `n_unique <= 50`).

### M10. Decouple PRDC seed from split seed

Currently both are `self.config.seed`. Use `seed + k` for the PRDC subsample. Average PRDC over 5+ subsample seeds per split to separate metric noise from split noise.

## Low-priority revisions (cosmetic)

### L1. Fix citation errors

- Synthcity: author list should be Qian, Davis, van der Schaar for the NeurIPS 2023 D&B paper (not Cebere). Citation reviewer flagged as MAJOR but fix is trivial.
- Add TabPFGen (Ma et al., arXiv 2406.05216, 2024) — referenced in lit review but not cited.
- Add CTAB-GAN+ (Zhao et al. 2023, Frontiers in Big Data).
- Add Auten-Splinter (2024) as DINA counterweight to PSZ 2018.
- Add Meyer-Mok-Sullivan on CPS benefit under-reporting.
- Add Czajka-Hirabayashi-Moffitt-Scholz (1992) for statistical matching lineage.
- Add Ruggles (2025 PNAS) as engagement point.
- Remove `zhang2017privbayes` (unused) or cite.

### L2. URL / DOI completeness

Add URLs/DOIs for: patki2016sdv (IEEE DOI 10.1109/DSAA.2016.49), xu2019modeling (NeurIPS proceedings), naeem2020prdc (PMLR), kotelnikov2023tabddpm (PMLR), borisov2023great (OpenReview), and others listed by the citation reviewer.

### L3. Bibliography cleanup

- `solatorio2023realtabformer` should be `@misc` not `@article` with `journal = {arXiv preprint}`.
- `dementen2014liam2` needs `{de Menten}, Gaetan` brace protection.
- Standardize URL-only vs DOI-only policy (document the rule once).

### L4. Table formatting

- Pick one bolding rule (all best-per-column or none).
- Spell out abbreviated headers ("Fit (s)" → "Fit time (s)") or footnote them.
- Expand "Pre-cal" / "Post-cal" to "Before calibration" / "After calibration."

### L5. Abstract cleanup

- Expand ZI-QRF / ZI-QDNN / ZI-MAF / PRDC on first use.
- Replace "keeps every record alive," "mainline," "77,006 × 50 scale" per H3.
- Either support or drop "widely-used" (H6).

### L6. Remove unused references from `.bib`

`ruggles2025synth` (cited in lit review but not index.qmd; consider citing in index.qmd per domain reviewer M1), `zhang2017privbayes`.

### L7. Cite each data product on first reference

CPS ASEC, ACS, PUF, SCF, SIPP need primary-source citations on first use.

### L8. Repository hygiene

- Add `LICENSE` file at repo root.
- Add regression test for ordering (e.g., `test_stage1_10k_ordering`).
- Move paper tables to Quarto chunks that read from `../artifacts/*.json` to auto-update.

## Revision order

Roughly the sequence to work through:

1. **Rerun pre-snap artifacts** (B1). Half-hour compute.
2. **Rename target columns + add downstream tax-output validation** (B2). Several days; the downstream run is non-trivial.
3. **Draft §3 Architecture** (B3). One to two days.
4. **Draft §4 Methods** (B3). One day.
5. **Add Code and Data Availability statement + COI** (B4, B5). One hour.
6. **Convert voice to first-person singular** (H1). Several hours, judgment-by-judgment.
7. **Self-contain Related Work** (H2). Half-day.
8. **Strip documentation register** (H3). Hours.
9. **Table captions, numbering, labels** (H4). Hour.
10. **Pipeline diagram** (H5). Hour (one TikZ / mermaid / svg figure).
11. **Soften the "widely-used" claim** (H6). Minutes.
12. **Citation form audit** (H7). Hour.
13. **Draft §5.3 rare-cell + §6 Discussion + §8 Conclusion** (B3 cont.). Two days.
14. **Medium-priority revisions** (M1–M10). Several days.
15. **Low-priority / cosmetic** (L1–L8). Final pass.

Total budget estimate: 2–3 weeks to a submittable draft, assuming the downstream tax-output validation is the bottleneck.

## What the reviewers got wrong

Two minor issues where the reviews overstated the gap:

- Reproducibility reviewer said `zi_maf_tuning.json` is missing; it is present at `artifacts/zi_maf_tuning.json` (verified). The reviewer's grep missed it.
- Citation reviewer flagged the identity-preservation framing as overstating the gap vs Dekkers (2015). Dekkers does discuss identity under static vs dynamic ageing; what the paper claims is novel is the cross-sectional calibration-layer framing, which Dekkers does NOT discuss. But the reviewer's point stands that the literature review should cite Dekkers and clarify which layer the claim refers to.

## Reviews kept for reference

Full reviewer outputs are preserved in the `a*` agent IDs noted by the subagent framework. If a rebuttal is needed later, those sessions can be resumed via `SendMessage`.
