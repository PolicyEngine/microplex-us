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

## Current PolicyEngine Profile Boundary

`pe_native_broad` keeps the raw PolicyEngine parity surface intact. It includes
all currently tracked broad target cells, including survey/model-input rows and
cells whose publisher-source semantics still need review.

`pe_native_broad_source_backed` is the Arch-backed calibration/profile boundary.
It excludes only cells with explicit reasons in
`src/microplex_us/policyengine/target_profiles.py`, such as:

- SOI multi-domain cells that would require joint AGI, filing status, and
  positive income-tax-before-credits facts not currently published by the loaded
  SOI packages
- survey-heavy or model-input cells such as rent, net worth, child support,
  medical-premium subcomponents, SPM capped expenses, and `ssn_card_type`
- source-near but non-equivalent rows such as `childcare_expenses`, where IRS
  credit expenses and W-2 dependent-care benefits are narrower tax concepts
- pregnancy stock by state, where live births are a flow rather than a direct
  source fact for the PolicyEngine target

## Current Local Snapshot

Snapshot date: 2026-05-22.

Inputs:

- `/Users/maxghenis/CosilicoAI/arch/arch/fixtures/consumer_facts.jsonl`
- `/Users/maxghenis/CosilicoAI/arch/macro/targets.db`
- `/tmp/arch-suite-hhs-acf-tanf-caseload-2024/consumer_facts.jsonl`
- `/tmp/arch-suite-soi-historic-table-2-2022/consumer_facts.jsonl`
- `/tmp/arch-suite-hhs-acf-liheap-fy2024-national-profile/consumer_facts.jsonl`
- `/tmp/arch-suite-soi-historic-table-2-state-agi-2022/consumer_facts.jsonl`
- `/tmp/arch-suite-soi-w2-statistics-2020/consumer_facts.jsonl`
- `/tmp/arch-suite-soi-table-1-4-2023/consumer_facts.jsonl`

Command:

```bash
uv run microplex-us-arch-target-refresh \
  --arch-targets-db /Users/maxghenis/CosilicoAI/arch/arch/fixtures/consumer_facts.jsonl \
  --arch-targets-db /Users/maxghenis/CosilicoAI/arch/macro/targets.db \
  --arch-targets-db /tmp/arch-suite-hhs-acf-tanf-caseload-2024/consumer_facts.jsonl \
  --arch-targets-db /tmp/arch-suite-soi-historic-table-2-2022/consumer_facts.jsonl \
  --arch-targets-db /tmp/arch-suite-hhs-acf-liheap-fy2024-national-profile/consumer_facts.jsonl \
  --arch-targets-db /tmp/arch-suite-soi-historic-table-2-state-agi-2022/consumer_facts.jsonl \
  --arch-targets-db /tmp/arch-suite-soi-w2-statistics-2020/consumer_facts.jsonl \
  --arch-targets-db /tmp/arch-suite-soi-table-1-4-2023/consumer_facts.jsonl \
  --period 2024 \
  --profile pe_native_broad_source_backed \
  --output-dir artifacts/arch-target-coverage
```

Coverage:

- 172 target cells in `pe_native_broad_source_backed`
- 172 covered
- 0 uncovered
- 100.0% coverage

The raw `pe_native_broad` profile remains at 172 of 189 covered with 17
explicitly reviewed rows outside the source-backed boundary:

| Category | Rows |
| --- | ---: |
| `survey_or_model_input_deprioritized` | 12 |
| `adapter_or_constraint_review` | 3 |
| `source_mapping_review` | 2 |

Generated outputs:

- `artifacts/arch-target-coverage/pe_native_broad_source_backed_2024_coverage.json`
- `artifacts/arch-target-coverage/pe_native_broad_source_backed_2024_gaps.json`
- `artifacts/arch-target-coverage/pe_native_broad_source_backed_2024_gaps.csv`
- `artifacts/arch-target-coverage/pe_native_broad_source_backed_2024_summary.md`

Remaining work is concentrated in:

- the raw `pe_native_broad` cells excluded from the source-backed profile, if a
  future primary publisher source can support them without changing semantics
- UK profile parity, which should follow the same pattern: keep the raw PE
  target surface intact and expose a source-backed profile with explicit
  exclusions where source equivalence is not defensible
