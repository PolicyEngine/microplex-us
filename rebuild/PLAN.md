# Scoreboard-first rebuild

Clean-slate rebuild of the microplex-US calibration + benchmark loop, built
outside-in: the measurement harness first, then the simplest baseline, then an
honest comparison. Branch `claude/scoreboard-first-rebuild` off `origin/main`.

This is a parallel experiment (a Codex effort is running independently); the two
are meant to be compared.

## Why this exists

An independent audit of the `mp120k_latest_us_data_refit_20260528` artifact
found the "Microplex beats eCPS" headline (0.094 vs 0.166 PE-native broad loss)
is not a sound comparison:

1. **Unmatched N** — 120k candidate households vs 41,314 eCPS (~2.9x weights).
2. **One-sided refit** — only the candidate is reweighted (`--score-candidate-only`);
   eCPS is scored with shipped weights.
3. **The refit is not a valid equalizer** — applying the *same* refit to eCPS
   drives its loss 0.166 -> 0.544 (confirmed monotonic across two optimizer
   configs, so it's structural, not instability). The fitting objective the
   optimizer minimises diverges from the native loss that is scored.

So there is currently no valid configuration in which the candidate beats eCPS.
Full findings: `~/microplex-vs-policyengine-us-data-review.md` (Entries 1-4).

## What "achieved" means

- [x] Scoreboard with invariants passing: fit==score, monotone non-increase
      (operator can't worsen a dataset), symmetric comparison, held-out split.
      (22 fast tests green.)
- [x] End-to-end recovery: refit(eCPS) recovers ~0.166, not 0.544.
      RESULT: shipped 0.16637 (reproduces baseline exactly), refit 0.16420
      (improves, never worse). See results/ecps_recovery.json.
- [~] An HONEST Microplex@N vs eCPS@N comparison on **held-out** targets, both
      calibrated by the identical operator. Symmetric + held-out harness DONE
      (mp_rebuild/compare.py); first run launched at UNMATCHED N (candidate
      ~120k vs eCPS ~41k). Matched-N is the next step.
- [ ] Matched N: candidate subsampled to 41,314 (careful multi-entity H5).
- [ ] A dumb reweighted-CPS baseline built end-to-end to a PE-ingestable H5.
- [ ] Short writeup: the honest number vs the old 0.094 claim, and vs Codex.

## The honesty contract (important)

We are NOT chasing a big loss win. Max's prior: mp and eCPS should be ~parity on
loss (the GD/reweighting is similar); mp's real edge is in **target cleanliness**
and **imputation quality** (a -/0/+ sign classifier before QRF, richer
predictors). An honest near-parity loss with quantified cleanliness/imputation
advantages is a SUCCESS. If a "win" only appears via extra degrees of freedom or
one-sided refitting, that is a bug, not a result.

## Design

- `mp_rebuild/scoreboard.py` — `CalibrationProblem` (one loss for fit + score),
  `fit` (GD + Armijo backtracking => never increases loss), `score`,
  `split_targets` (held-out), `compare` (symmetric, no one-sided refit).
- One loss is the source of truth; there is no separate fitting vs scoring
  matrix that can diverge. The bug that produced 0.544 is structurally excluded.

## Choices / non-destructive note

- Built as a standalone package under `rebuild/`; it does **not** import the
  existing 14.7k-line `src/microplex_us/pipelines/us.py` monolith. PE-native
  scoring will be re-derived cleanly (reusing only raw data + target
  definitions), partly so we don't reimport whatever causes the fit/score
  divergence.
- The existing code is left in place (not deleted) — it's a reference and is
  recoverable from `origin/main` regardless.

## Status log

- **Iter 1** — worktree + scoreboard core + fast invariant tests (green). The
  non-increase / recovery property holds by construction. eCPS integration test
  scaffolded and skipped pending the PE-native loader.
- **Iter 2** — `mp_rebuild/pe_native.py`: clean loader reusing the canonical
  `build_loss_matrix` + exact PE-native filtering/scaling (reproduces ~0.166).
  Added `reduction="sum"` to `CalibrationProblem` for the pre-scaled PE-native
  system. Live eCPS recovery test (`MPR_RUN_SLOW=1`) + standalone
  `run_recovery_check.py` driver. Fast suite still green (19 passed, 1 skipped).
  Launched the recovery check in background (build matrix + fit + score eCPS) —
  expect shipped_loss ~0.166 and refit_loss <= shipped (vs the old 0.544).
  Reuses only data helpers (`_ENHANCED_CPS_BAD_TARGETS`, family classifier),
  not the old optimizer.
  ENV (important): run PE-native scoring from the microplex-us main checkout as
  `cd ~/CosilicoAI/microplex-us && PYTHONPATH=/Users/maxghenis/PolicyEngine/policyengine-us-data uv run --extra dev --extra policyengine python <script>`.
  `policyengine_us_data` is NOT installed in the uv env (nor the bare .venv); it
  must be injected from its repo source via PYTHONPATH. (`policyengine_us` and
  `microplex_us` ARE in the uv env.) The recovery driver adds its own dir to
  sys.path for `mp_rebuild`.
  Caveat: us-data's `utils/loss.py` has uncommitted edits, so the live target
  set may differ slightly from the original 0.166 baseline -- recovery
  (non-increase) holds regardless; baseline reproduction is checked with abs=0.03.
- **Iter 3** — RECOVERY PROVEN (shipped 0.16637 == published baseline to 10
  digits; refit 0.16420, never worse; old harness gave 0.544). Built the
  symmetric, shared-scaling, held-out comparison harness: refactored
  `pe_native.py` to expose raw (unscaled) pieces + a `pe_native_scaling` helper
  (scaling depends only on the target set, so both datasets get identical
  scaling on their common targets); `compare.py` fits both on a TRAIN target
  split and scores on a disjoint HOLDOUT split. Renamed scoreboard `compare`
  -> `compare_problems` (collision with the compare module). 22 fast tests
  green. Launched the first honest comparison (candidate mp_top120000 vs eCPS,
  unmatched N) in background -> `~/microplex-review-rescore/clean_comparison.log`
  (COMPARISON_RESULT json). Honest expectation: ~parity on held-out.
  TODO next: matched-N candidate loader (>50k households needs batching),
  reweighted-CPS baseline, then the honest Microplex@N vs eCPS@N held-out run.
