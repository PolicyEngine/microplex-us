# Pending Claude Review

Please do a focused code review of the current US state-program accuracy work across:

- `/Users/maxghenis/PolicyEngine/microplex-us`
- `/Users/maxghenis/PolicyEngine/microplex`

Use agent teams if available in your environment.
Suggested split:
- one agent for `/Users/maxghenis/PolicyEngine/microplex-us`
- one agent for `/Users/maxghenis/PolicyEngine/microplex`
- one integrating agent to synthesize the calibration/benchmark conclusion

## Read first

- `/Users/maxghenis/PolicyEngine/microplex-us/AGENTS.md`
- `/Users/maxghenis/PolicyEngine/microplex-us/_WORKSPACE.md`
- `/Users/maxghenis/PolicyEngine/microplex-us/_BUILD_LOG.md`
- `/Users/maxghenis/PolicyEngine/microplex/AGENTS.md`
- `/Users/maxghenis/PolicyEngine/microplex/_WORKSPACE.md`
- `/Users/maxghenis/PolicyEngine/microplex/_BUILD_LOG.md`

Then inspect recent changes with git diff/status and review the changed files and saved artifacts.

## Review mindset

- Findings first, ordered by severity.
- Prioritize bugs, behavioral regressions, benchmark-validity risks, abstraction mistakes, silent incompatibilities, and missing tests.
- Be skeptical and concrete.
- I want actionable review comments, not a broad summary.

## Important recent changes

- We investigated the US `state_programs_core` gap against PE.
- Earlier diagnosis leaned toward source/backbone support, but the recent diagnosis shifted toward calibration feasibility.
- `microplex-us` now has a calibration feasibility filter and better weight-collapse diagnostics in `/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/us.py`.
- Explicit semantic specs were added for `has_medicaid`, `public_assistance`, `ssi`, and `social_security` in `/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/variables.py`.
- A core synthesizer bug for all-zero zero-inflated variables was fixed in `/Users/maxghenis/PolicyEngine/microplex/src/microplex/transforms.py`.
- A boolean-to-float support-fill bug was fixed in `ensure_target_support()` in `/Users/maxghenis/PolicyEngine/microplex-us/src/microplex_us/pipelines/us.py`.
- We ran corrected state-only reruns against the real PE-US-data calibration DB using:
  - variables: `household_count`, `person_count`
  - domains: `snap`, `medicaid_enrolled`
  - geography: `state`
- Key artifacts:
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_state_programs_n2000_diagnostics_20260329.json`
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_cps_puf_rich_state_sweep_20260329.json`
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_state_programs_feasible_bootstrap_rerun_20260329.json`
  - `/Users/maxghenis/PolicyEngine/microplex-us/artifacts/tmp_state_programs_feasible_synth_rerun_20260329.json`

## Current headline result to verify

On the corrected feasible state-only target estate, `n=2000` now appears to beat PE.

Bootstrap rerun:
- Microplex MARE `0.7335`
- PE MARE `0.7386`
- Microplex composite `0.6770`
- PE composite `0.7704`

Synthesizer rerun:
- Microplex MARE `0.6811`
- PE MARE `0.7386`
- Microplex composite `0.6481`
- PE composite `0.7704`
- target win rate `42.16%`

## Focus especially on

1. Whether the new diagnosis is actually correct: was the main blocker calibration infeasibility rather than source/backbone support?
2. Whether the new calibration feasibility filter in `us.py` is mathematically and operationally sound, or whether it is just hiding targets we should still be solving.
3. Whether the corrected state-only calibration scope is the right canonical target estate for this question, or whether it is too favorable, too narrow, or no longer comparable to PE.
4. Whether the bootstrap and synthesizer reruns are genuinely apples-to-apples against PE.
5. Whether the new proxy semantic specs are correct and sufficient.
6. Whether the all-zero zero-inflated transform fix in core is correct and safe.
7. Whether the `ensure_target_support()` bool/numeric coercion fix is correct or risks masking real support problems.
8. Whether there are missing tests that would let this diagnosis flip again incorrectly.
9. Whether the next correct operational step is to make this corrected state-only feasible calibration path part of the canonical US benchmark/version-benchmark flow.

## Important context

- We are intentionally trying to keep `microplex` generic and `microplex-us` thin where possible.
- We want to beat PE on real targets, but not by benchmarking an invalid or overly favorable target estate.
- PE rules remain the canonical runtime for program calculations.
- The question here is specifically whether we now have a sound US state-program benchmark path and a real result, not just a debugging artifact.

## Please return

1. Findings first, with severity and file/line references.
2. Then a short section on architectural risks.
3. Then the top 3 next fixes.
4. Then explicitly answer:
   - Is the new diagnosis actually right?
   - Is the corrected feasible state-only benchmark path sound and comparable to PE?
   - Do the new `n=2000` results actually support the claim that Microplex now beats PE on the corrected US state-program slice?
   - What is the next highest-leverage fix?

## After the review

1. Write the full review to:
   - `/Users/maxghenis/PolicyEngine/microplex-us/reviews/2026-03-29-claude-state-program-review.md`
2. Append a concise summary to:
   - `/Users/maxghenis/PolicyEngine/microplex-us/_BUILD_LOG.md`

Keep the `_BUILD_LOG.md` append short:
- date
- scope reviewed
- top findings
- top 1-3 next fixes
