# Arch Target Gap Queue

The Arch target gap queue is a Microplex-side review tool. It compares a
Microplex target profile to a queryable Arch target DB and emits rows that help
humans or agents decide what Arch source work is missing.

The queue does not make Arch own Microplex target selection. Profile membership,
source aging, reconciliation, activation, and model-variable aliases remain in
`microplex-us`.

## Boundary Rules

- Arch stores publisher/source facts with provenance, constraints, periods,
  geography, and source lineage.
- Arch should not duplicate a source fact only because Microplex names a model
  variable differently.
- Microplex adapters may map one Arch source fact into simulator-specific target
  semantics. For example, Arch
  `irs_soi.returns_with_income_tax_after_credits` can satisfy the
  PolicyEngine `income_tax_positive` count target because SOI Table 1.1 reports
  the count of returns with positive income tax after credits.
- A gap row is an authoring hint, not proof that a source exists.
- Rows marked as source-mapping review or deprioritized must be reviewed before
  assigning loader work to agents.

## Categories

`gap_category` is the high-level agent-readiness taxonomy:

| Category | Meaning | Default action |
| --- | --- | --- |
| `covered` | An Arch target record already satisfies the target cell. | No task. |
| `ready_primary_loader` | The expected publisher source and Arch variable shape are known, but the record is missing. | Assign source-loader/spec work. |
| `ready_rollup_or_geography` | The Arch variable exists but not at the requested geography. | Add rollup/geography records or review source geography. |
| `adapter_or_constraint_review` | The Arch variable exists at the geography, but filters or adapter matching do not cover the cell. | Review constraints and adapter mapping. |
| `source_mapping_review` | The queue cannot identify a defensible source fact or Arch variable shape. | Human source-mapping review first. |
| `survey_or_model_input_deprioritized` | The cell is currently treated as a survey/model-input proxy rather than a primary administrative source task. | Defer unless a primary source is identified. |

`loader_status` is the lower-level diagnostic used to derive the category. Use
`gap_category` for agent routing and `loader_status` for debugging why a cell
landed there.

## Current PolicyEngine Broad Profile Boundary

The current Arch-backed PE broad profile coverage intentionally stops before
survey-heavy or model-input cells such as rent, net worth, child support,
medical-premium subcomponents, SPM expenses, and `ssn_card_type`. Those rows are
not ready for automated source-loader agents under the primary-source-first
policy.

## Current Local Snapshot

Snapshot date: 2026-05-19.

Inputs:

- `/Users/maxghenis/CosilicoAI/arch/arch/fixtures/consumer_facts.jsonl`
- `/Users/maxghenis/CosilicoAI/arch/macro/targets.db`

Command:

```bash
uv run microplex-us-arch-target-refresh \
  --artifact-root /Users/maxghenis/CosilicoAI/arch \
  --period 2024 \
  --profile pe_native_broad \
  --output-dir artifacts/arch-target-coverage
```

Coverage:

- 189 target cells in `pe_native_broad`
- 138 covered
- 51 uncovered
- 73.0% coverage
- national: 79 of 116 covered
- state: 59 of 73 covered

Gap categories:

| Category | Rows |
| --- | ---: |
| `source_mapping_review` | 26 |
| `survey_or_model_input_deprioritized` | 12 |
| `adapter_or_constraint_review` | 10 |
| `ready_rollup_or_geography` | 3 |

Generated outputs:

- `artifacts/arch-target-coverage/pe_native_broad_2024_coverage.json`
- `artifacts/arch-target-coverage/pe_native_broad_2024_gaps.json`
- `artifacts/arch-target-coverage/pe_native_broad_2024_gaps.csv`
- `artifacts/arch-target-coverage/pe_native_broad_2024_summary.md`

Remaining work is concentrated in:

- source-mapping review for the newly expanded PE parity cells, especially
  domains whose expected Arch concept is not yet encoded in the gap taxonomy
- adapter or constraint review where Arch has the variable at the right
  geography but the Microplex adapter does not yet match the PE target cell
- a small rollup/geography queue for variables loaded in Arch but not at the
  requested national or state target geography
- survey/model-input proxy cells that remain deprioritized until a primary
  publisher source is identified
