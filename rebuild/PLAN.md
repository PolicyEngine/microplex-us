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

- [ ] Scoreboard with invariants passing: fit==score, monotone non-increase
      (operator can't worsen a dataset), symmetric comparison, held-out split,
      matched-N as a first-class axis. (unit level: DONE; integration: pending)
- [ ] End-to-end recovery: refit(eCPS) recovers ~0.166, not 0.544.
- [ ] A dumb reweighted-CPS baseline built end-to-end to a PE-ingestable H5.
- [ ] An HONEST Microplex@N vs eCPS@N comparison on **held-out** targets at
      matched N, both calibrated by the identical operator, committed + pushed.
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
