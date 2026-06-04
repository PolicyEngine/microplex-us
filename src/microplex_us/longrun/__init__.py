"""Long-run (75-year) household projection + reweighting, ported from
policyengine-us-data (which is being deprecated in favor of microplex-us).

Builds year-specific projected datasets from a base dataset by uprating via
PolicyEngine and reweighting to SSA Trustees age/SS/payroll/TOB targets.

PORT STATUS (claude/port-75y-longrun):
- [done] relocate engine into microplex_us.longrun
- [todo] rebase base_dataset on the promoted MP dataset (not eCPS)
- [todo] income anchoring: regularize forward weights toward the income-rich MP
  base + add CBO long-run income-tax-receipts %GDP target (fixes the
  unconstrained-income-tax / tax>GDP under-determination)
"""
