# `microplex-us` paper

Quarto manuscript and supporting materials.

## Affiliation

PolicyEngine-only. See `AFFILIATION.md` — this work is intentionally independent of PolicyEngine for tax-and-organization reasons.

## Contents

- `_quarto.yml` — project config, HTML + PDF outputs.
- `index.qmd` — main manuscript.
- `literature-review.qmd` — standalone literature survey, cited by the main paper.
- `references.bib` — BibTeX bibliography, confirmed citations only.
- `AFFILIATION.md` — hard rule on affiliation independence. Re-read before adding any acknowledgment or author line.

## Build

```bash
cd paper
quarto render             # both HTML and PDF
quarto render index.qmd   # main paper only
quarto preview            # live-reload local server
```

Output lands in `_output/`.

## Cross-references and figures

Figures and tables are sourced from `../artifacts/` (`stage1_77k_snap.json`, `zi_maf_tuning.json`, `embedding_prdc_compare.json`, `calibrate_on_synthesizer.json`). When final figures land, they should be generated as Quarto chunks rather than hand-placed PNGs so they re-render against the latest artifact set.

## Citation style

APA via Quarto's built-in CSL. Change in `_quarto.yml` if the target journal has a different requirement.
