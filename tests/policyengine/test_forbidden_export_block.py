"""Regression test: the eCPS-contract forbidden columns are excluded from export.

The contract's ``forbidden`` set (transient ``*_reported`` takeup inputs and the
PUF reported/calculated tax-credit outputs) must never reach the written H5.
``resolve_policyengine_excluded_export_variables`` is the single chokepoint that
the materialize/persist path uses to build the ``excluded`` set, so we assert it
excludes every forbidden column that appears in the exported inputs, and that the
exclusion is sourced from the committed contract (single source of truth).
"""

import json
from pathlib import Path

from microplex_us.policyengine.us import (
    _contract_forbidden_export_columns,
    resolve_policyengine_excluded_export_variables,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "microplex_us"
    / "pipelines"
    / "ecps_export_contract.json"
)


def _contract_forbidden() -> set[str]:
    return set(json.loads(CONTRACT_PATH.read_text())["forbidden"])


class _NoComputedVariables:
    """Tax-benefit-system stub whose variables are all plain inputs.

    Isolates the forbidden-block behaviour from PolicyEngine-computed detection:
    with no known variables, ``detect_policyengine_computed_export_variables``
    contributes nothing, so anything excluded comes from the forbidden contract.
    """

    variables: dict = {}


def test_helper_matches_committed_contract_forbidden():
    assert _contract_forbidden_export_columns() == _contract_forbidden()
    # Non-trivial: the contract forbids the *_reported + PUF tax-credit family.
    assert "snap_reported" in _contract_forbidden_export_columns()
    assert "general_business_credit" in _contract_forbidden_export_columns()


def test_forbidden_columns_present_in_export_are_excluded():
    forbidden = sorted(_contract_forbidden())
    # Mix forbidden columns with legitimate inputs.
    exported = forbidden + ["age", "employment_income", "household_weight"]
    excluded = resolve_policyengine_excluded_export_variables(
        _NoComputedVariables(), exported
    )
    # Every forbidden column that was offered for export must be excluded.
    assert set(forbidden).issubset(excluded)
    # Legitimate inputs must NOT be excluded.
    assert "age" not in excluded
    assert "employment_income" not in excluded
    assert "household_weight" not in excluded


def test_forbidden_columns_absent_are_not_spuriously_excluded():
    # If no forbidden column is exported, none are added to the excluded set.
    exported = ["age", "employment_income", "household_weight"]
    excluded = resolve_policyengine_excluded_export_variables(
        _NoComputedVariables(), exported
    )
    assert excluded.isdisjoint(_contract_forbidden())
