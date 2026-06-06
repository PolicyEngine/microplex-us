# Next v8 pipeline run plan

> Superseded for release-candidate builds as of 2026-06-06. PE-US-data PUF
> support clone rebuilds must use `--donor-imputer-backend regime_aware`, which
> routes through MicroImpute chained donor imputations. The older `qrf` and
> `zi_qrf` backends remain useful only for explicit non-release experiments with
> `puf_support_clone_enabled=False`; release-profile config now fails closed if
> either backend is requested.

## Summary

v7 (2026-04-18 12:19 PM, artifact `live_pe_us_data_rebuild_checkpoint_20260418_microcalibrate_modular`) uses the default `donor_imputer_backend="qrf"`. That path leaves `zero_inflated_vars` empty in `ColumnwiseQRFDonorImputer`, so the imputer fits no zero-classifier and the QRF runs `predict()` over all 3.37 M rows for every target column — including columns that are 99 % zero.

v8 originally planned to flip to `--donor-imputer-backend zi_qrf`, which activates the `ZERO_INFLATED_POSITIVE`-whitelist path. On whitelisted columns the imputer fits a `RandomForestClassifier` zero-gate, then only invokes QRF `predict()` on rows the gate sends to the positive branch. On a 97 %-zero column this cuts QRF predict to ~3 % of rows — a large wall-clock win on donor integration. That is no longer sufficient for MP/eCPS release candidates because it does not use MicroImpute chained imputations across related donor targets.

## What `zi_qrf` actually covers

The whitelist is populated from variables whose `VariableSupportFamily` is `ZERO_INFLATED_POSITIVE`. Grep over `src/microplex_us/variables.py`:

- `dividend_income`, `ordinary_dividend_income`, `qualified_dividend_income`, `non_qualified_dividend_income`
- `taxable_interest_income`, `tax_exempt_interest_income`
- `taxable_pension_income`
- (plus the rest of the PUF-side tax variables marked with `support_family=VariableSupportFamily.ZERO_INFLATED_POSITIVE` — run `grep -n ZERO_INFLATED_POSITIVE src/microplex_us/variables.py | head -30` for the full list)

Benefit variables `ssi_reported`, `tanf_reported`, `snap_reported`, `unemployment_compensation`, `social_security_disability` are currently marked `CONTINUOUS` even though they have high zero fractions. They will *not* get the zero-gate under `zi_qrf`. If we want to speed those up too, the fix is a one-line support-family reclassification in `variables.py`, not a code change.

## Pre-launch verification

Run `uv run pytest tests/pipelines/test_zi_qrf_backend.py -v`. Five tests pin the guarantees v8 relies on:

1. `test_zi_whitelist_produces_zero_classifier` — given a whitelist, `fit()` trains the RF gate on heavy-zero columns and not on dense columns.
2. `test_empty_whitelist_means_no_gates` — documents v7 behavior (no gates ever fitted).
3. `test_generate_calls_qrf_only_on_predicted_positive_rows` — proves QRF `predict` is called on a strict subset; the wall-clock optimization is real.
4. `test_zi_qrf_backend_populates_whitelist` — `backend="zi_qrf"` in the factory wires the whitelist from the semantic specs correctly.
5. `test_qrf_backend_leaves_whitelist_empty` — `backend="qrf"` (v7) leaves optimization off, regression-pin.

## Launch command for v8

```bash
HF_TOKEN=$(cat ~/.huggingface/token) \
HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token) \
uv run python -m microplex_us.pipelines.pe_us_data_rebuild_checkpoint \
  --output-root artifacts/live_pe_us_data_rebuild_checkpoint_<date>_zi_qrf_modular \
  --baseline-dataset /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5 \
  --targets-db /Users/maxghenis/PolicyEngine/policyengine-us-data-aca-agi-db/policyengine_us_data/storage/calibration/policy_data.db \
  --policyengine-us-data-repo /Users/maxghenis/PolicyEngine/policyengine-us-data \
  --calibration-backend microcalibrate \
  --donor-imputer-backend regime_aware \
  --version-id microcalibrate-regime-aware-v8 \
  --n-synthetic 100000 \
  --defer-policyengine-harness \
  --defer-policyengine-native-score \
  --defer-native-audit \
  --defer-imputation-ablation
```

## Subtle consequence of the gate

With the gate active, the post-ZI QRF is fit *only* on rows with `y > 0`. It cannot produce zero at prediction time — its minimum leaf value equals the smallest positive training value. This is the standard two-component zero-inflated mixture:

$$P(y \mid x) = P(y = 0 \mid x) \cdot \delta_0(y) + P(y > 0 \mid x) \cdot f_{\text{pos}}(y \mid x)$$

Zeros come exclusively from the gate path (`values[:] = 0.0`). Nonzero draws come exclusively from the QRF path. The final synthetic distribution has the correct zero mass and a strictly positive continuous tail, but the boundary between them is sharp: no "small positive values just above zero" exist if the training data has a visible gap at that boundary. For PUF variables like dividend/interest income the gap is unobservable in distributional tests, but the asymmetry is worth remembering if we ever inspect column-level support coverage near zero.

## Open follow-ups after v8 succeeds

- Extend `ZERO_INFLATED_POSITIVE` support_family classification to the benefit variables (`ssi_reported`, `tanf_reported`, `snap_reported`, `unemployment_compensation`, `social_security_disability`) so `zi_qrf` gates those too. That's the largest remaining gap; those are the 98 %-zero columns currently running QRF predict on all 3.37 M rows.
- Run a small benchmark comparing v7 (`qrf`) vs v8 (`zi_qrf`) donor-integration wall time on the same source set to quantify the actual speedup.
