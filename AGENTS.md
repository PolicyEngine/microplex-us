# AGENTS.md

This repo is the US country pack for `microplex`. Keep it thin where possible and push shared abstractions upstream into core.

## Default posture

- Prefer spec-driven behavior over ad hoc logic in large pipeline files.
- If a seam is useful for both UK and US, move it to `microplex` instead of polishing a US-only local helper.
- Keep PolicyEngine-US execution details local unless there is a clean shared protocol.

## Regression discipline (every fix ships a guard)

A bug is not fixed until it ships an **automated guard on the run path** — a
gate/assertion/CI test that fails loud if it recurs. Do not rely on a human or
agent re-noticing a known failure mode. Definition of done for any fix: "I added
the check that would have caught this."

In particular, the eCPS-replacement comparison must refuse to emit a verdict when
its inputs or result are invalid (see `ecps_replacement_comparison.py` gates):
the baseline must be production-pinned and score sanely (baseline-sanity gate),
and the refit must materially reduce loss (refit-effectiveness gate). When a new
comparison failure mode is found, add a gate there rather than fixing it once.

## Current architectural intent

- `microplex-us` owns:
  - US source manifests and raw source adapters
  - PolicyEngine-US execution/materialization
  - US-specific target providers and benchmark harnesses
  - US-local pipeline orchestration
- `microplex` core owns:
  - targets specs/providers/protocols
  - reweighting bundles and solver
  - benchmark metrics/comparisons/suites
  - shared result-based benchmark builders

## Current mission notes

- For US, the canonical mission metric is the PE-native broad loss frontier, not composite parity.
- When evaluating progress, prefer:
  - matched-size `Microplex@N` vs `PE@N`
  - full `enhanced_cps_2024` only as a stretch reference
- Recent direct-objective testing showed that changing only the post-export weight objective moves loss very little on the same fixed candidate.
- Bias effort toward:
  - better candidate records
  - fuller support coverage
  - budgeted selection on larger candidates
- Bias away from:
  - repeated small-candidate donor-backend A/Bs
  - more entropy tuning without evidence that the candidate population itself improved

## Review checklist

When reviewing recent changes here, check:

1. Is this still duplicating something that should now live in core?
2. Is the US harness using shared core benchmarking helpers instead of rebuilding them inline?
3. Are any benchmark claims relying on non-common-target comparisons?
4. Is the work using PE-native broad loss when it claims mission progress?
5. Does PE-US materialization handle dependency chains and partial failures safely?
6. Is this baking in fixed tax-unit structure more deeply than necessary?

## Be careful around

- `src/microplex_us/policyengine/us.py`
  - Large file with execution/materialization logic and remaining monolith risk.
- `src/microplex_us/policyengine/harness.py`
  - Should keep delegating more suite/result logic to core.
- `src/microplex_us/pipelines/local_reweighting.py`
  - Should remain a thin adapter over core bundle/reweighting surfaces.

## Standard commands

- Install, production/runtime: `./scripts/install.sh --prod`
- Install, development: `./scripts/install.sh --dev`
- Install, Intel macOS development: `./scripts/install.sh --dev-intel-mac`
- Ruff: `uv run ruff check src tests`
- Focused comparison/harness tests: `uv run pytest -q tests/policyengine/test_comparison.py tests/policyengine/test_harness.py`
- Local reweighting tests: `uv run pytest -q tests/pipelines/test_local_reweighting.py`

## Environment guidance

- Production macOS installs require Apple Silicon (`arm64`).
- Intel macOS (`x86_64`) is development/testing-only. If uv/PyPI fails on
  `torch` wheels there, use `./scripts/install.sh --dev-intel-mac`.
- Do not add torch stubs or no-torch runtime workarounds for Intel macOS; use
  the conda-forge developer environment instead.

## Claude/Codex review shortcut

For a quick review, read:

1. [`/Users/maxghenis/PolicyEngine/microplex-us/AGENTS.md`](/Users/maxghenis/PolicyEngine/microplex-us/AGENTS.md)
2. [`/Users/maxghenis/PolicyEngine/microplex-us/_WORKSPACE.md`](/Users/maxghenis/PolicyEngine/microplex-us/_WORKSPACE.md)
3. [`/Users/maxghenis/PolicyEngine/microplex-us/_BUILD_LOG.md`](/Users/maxghenis/PolicyEngine/microplex-us/_BUILD_LOG.md)

Then inspect changed files and return findings first.

## Review handoff

To avoid rebuilding long prompts in chat:

1. Treat [`/Users/maxghenis/PolicyEngine/microplex-us/reviews/PENDING_CLAUDE_REVIEW.md`](/Users/maxghenis/PolicyEngine/microplex-us/reviews/PENDING_CLAUDE_REVIEW.md) as the current review request.
2. Read that file after the standard repo context files above.
3. Write the full review to a dated file under [`/Users/maxghenis/PolicyEngine/microplex-us/reviews/`](/Users/maxghenis/PolicyEngine/microplex-us/reviews/).
4. Append only a concise summary to [`/Users/maxghenis/PolicyEngine/microplex-us/_BUILD_LOG.md`](/Users/maxghenis/PolicyEngine/microplex-us/_BUILD_LOG.md).

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **microplex-us** (4778 symbols, 12879 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/microplex-us/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/microplex-us/context` | Codebase overview, check index freshness |
| `gitnexus://repo/microplex-us/clusters` | All functional areas |
| `gitnexus://repo/microplex-us/processes` | All execution flows |
| `gitnexus://repo/microplex-us/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
