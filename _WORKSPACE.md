# _WORKSPACE.md

This file is the durable local context for `microplex-us`.

## Repo role

`microplex-us` is the US country pack. It should specialize core `microplex`, not fork it conceptually.

Core repo:

- [`/Users/maxghenis/PolicyEngine/microplex`](/Users/maxghenis/PolicyEngine/microplex)

Sibling country pack:

- [`/Users/maxghenis/PolicyEngine/microplex-uk`](/Users/maxghenis/PolicyEngine/microplex-uk)

## Current high-value modules

### PolicyEngine-US

- `src/microplex_us/policyengine/us.py`
- `src/microplex_us/policyengine/harness.py`
- `src/microplex_us/policyengine/comparison.py`

### Local reweighting

- `src/microplex_us/pipelines/local_reweighting.py`

### Source semantics / manifests

- `src/microplex_us/variables.py`
- `src/microplex_us/data_sources/`
- `src/microplex_us/manifests/`

## Current architectural boundary

US should keep local:

- PE-US microsimulation/materialization details
- US target database/provider specifics
- raw CPS/PUF and other US source mappings

US should not keep local if it generalizes:

- benchmark math
- benchmark suite/result types
- reweighting math
- generic target querying/filtering
- long-lived eval-repo benchmark orchestration for method bakeoffs

## Important current caveat

US tax filing units may eventually be policy-endogenous. Avoid hard-baking tax-unit structure too deeply into shared abstractions.

## Current mission metric

- The real US mission is no longer generic parity improvement. It is to beat PolicyEngine on the PE-native broad loss frontier.
- The main comparator should be matched-size PE baselines:
  - `Microplex@N` vs `PE@N`
  - ideally `PE@N` should be reweighted/recalibrated after sampling
- Full `enhanced_cps_2024` remains the stretch reference, not the only pass/fail bar.

## Current benchmark guidance

- Use common-target comparisons when claiming candidate vs baseline wins.
- Composite parity loss remains useful as a diagnostic, but it is not the US mission metric.
- PE-native broad loss is the canonical mission score for US frontier work.
- Do not assume larger `N` should help automatically on the current path; non-monotonicity has already shown that record support and optimizer alignment are still imperfect.

## Current diagnostic read

- Post-export direct optimization on the exact PE-native broad objective is now available.
- On a fixed `2000`-household exported candidate, direct PE-native optimization improved loss only trivially (`0.92334 -> 0.92290`).
- Current read: objective mismatch is real, but the larger bottleneck is still record construction/support, not just the final weight objective.
- The next high-leverage path is full-support candidate construction plus budgeted household selection, not more small-candidate entropy or donor A/B loops.

## Selection backends

- Household-budgeted selection now has two backends in the US pipeline:
  - `sparse`
  - `pe_native_loss`
- `pe_native_loss` is the cleaner experimental backend because it ranks/selects households using the actual PE-native loss surface on an exported candidate.
- Until the full-support `pe_native_loss` selector run lands, do not port this architecture to UK.

## High-signal tests

- `tests/policyengine/test_comparison.py`
- `tests/policyengine/test_harness.py`
- `tests/policyengine/test_us.py`
- `tests/pipelines/test_local_reweighting.py`
- `tests/test_share_imputation.py`

## Working rule

If the same helper starts appearing in both PE-US and PE-UK benchmark/reweighting flows, promote it to core.

## Review handoff

- Current durable Claude request:
  - `/Users/maxghenis/PolicyEngine/microplex-us/reviews/PENDING_CLAUDE_REVIEW.md`
- Full saved reviews belong under:
  - `/Users/maxghenis/PolicyEngine/microplex-us/reviews/`
- `_BUILD_LOG.md` should only keep a concise review summary, not the full review body.
