"""Tests for delegating PolicyEngine tax-unit reconstruction to ``microunit``.

These exercise the reconstruction-from-scratch path added in issue #113, where
``USMicroplexPipeline._build_policyengine_tax_units_from_role_flags`` delegates to
:func:`microunit.construct_tax_units` when the person frame carries the raw CPS
columns ``microunit`` requires. The authoritative-ID path (#112) and the legacy
role-flag fallback are validated by ``test_us_entities.py``.

``microunit`` ships in the ``policyengine`` optional dependency group, so these
tests skip when it is not installed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from microplex_us.pipelines.us import USMicroplexPipeline

pytest.importorskip("microunit", reason="microunit is in the policyengine extra")


@pytest.fixture
def pipeline() -> USMicroplexPipeline:
    return USMicroplexPipeline()


def _unit_field(tax_units: pd.DataFrame, unit_id: int, field: str):
    """Read one tax unit's field robustly.

    ``tax_units`` has list-valued columns (``filer_ids`` etc.); ``DataFrame.iterrows``
    coerces a row to a single-dtype Series and mangles those, so we filter to the
    single matching row and read the column's first (only) value instead.
    """
    rows = tax_units.loc[tax_units["tax_unit_id"] == unit_id]
    assert len(rows) == 1, f"expected exactly one unit {unit_id}, found {len(rows)}"
    return rows[field].iloc[0]


def _cps_person_frame() -> pd.DataFrame:
    """A two-household CPS-like frame microunit can consume directly.

    Household 10 is a married couple (lines 1 and 2) with one own child
    (line 3, parent pointers to lines 1 and 2). Household 20 is a single adult.
    The columns are the raw CPS ASEC names microunit expects; ``person_id`` and
    ``household_id`` are microplex's own keys used to map results back.
    """
    return pd.DataFrame(
        [
            {
                "person_id": 1,
                "household_id": 10,
                "relationship_to_head": 0,
                "age": 40,
                "income": 50000.0,
                "PH_SEQ": 10,
                "A_LINENO": 1,
                "A_AGE": 40,
                "A_MARITL": 1,
                "A_SPOUSE": 2,
                "PEPAR1": 0,
                "PEPAR2": 0,
                "A_EXPRRP": 1,
                "WSAL_VAL": 50000,
            },
            {
                "person_id": 2,
                "household_id": 10,
                "relationship_to_head": 1,
                "age": 38,
                "income": 30000.0,
                "PH_SEQ": 10,
                "A_LINENO": 2,
                "A_AGE": 38,
                "A_MARITL": 1,
                "A_SPOUSE": 1,
                "PEPAR1": 0,
                "PEPAR2": 0,
                "A_EXPRRP": 3,
                "WSAL_VAL": 30000,
            },
            {
                "person_id": 3,
                "household_id": 10,
                "relationship_to_head": 2,
                "age": 10,
                "income": 0.0,
                "PH_SEQ": 10,
                "A_LINENO": 3,
                "A_AGE": 10,
                "A_MARITL": 7,
                "A_SPOUSE": 0,
                "PEPAR1": 1,
                "PEPAR2": 2,
                "A_EXPRRP": 5,
                "WSAL_VAL": 0,
            },
            {
                "person_id": 4,
                "household_id": 20,
                "relationship_to_head": 0,
                "age": 25,
                "income": 45000.0,
                "PH_SEQ": 20,
                "A_LINENO": 1,
                "A_AGE": 25,
                "A_MARITL": 7,
                "A_SPOUSE": 0,
                "PEPAR1": 0,
                "PEPAR2": 0,
                "A_EXPRRP": 1,
                "WSAL_VAL": 45000,
            },
        ]
    )


def test_microunit_path_groups_married_couple_with_child(pipeline):
    persons = _cps_person_frame()

    result = pipeline._build_policyengine_tax_units_via_microunit(persons)

    assert result is not None, "microunit path must activate when CPS columns exist"
    tax_units, person_rows, households = result

    # Two units: the married-with-child family and the single adult.
    assert households == {10, 20}
    assert person_rows["tax_unit_id"].nunique() == 2
    assert len(tax_units) == 2

    # The three household-10 members share one unit; the single adult is alone.
    unit_by_person = dict(
        zip(person_rows["person_id"], person_rows["tax_unit_id"], strict=True)
    )
    assert unit_by_person[1] == unit_by_person[2] == unit_by_person[3]
    assert unit_by_person[4] != unit_by_person[1]

    family_id = int(unit_by_person[1])
    single_id = int(unit_by_person[4])

    assert _unit_field(tax_units, family_id, "filing_status") == "JOINT"
    assert int(_unit_field(tax_units, family_id, "n_dependents")) == 1
    assert sorted(_unit_field(tax_units, family_id, "filer_ids")) == [1, 2]
    assert list(_unit_field(tax_units, family_id, "dependent_ids")) == [3]

    assert _unit_field(tax_units, single_id, "filing_status") == "SINGLE"
    assert int(_unit_field(tax_units, single_id, "n_dependents")) == 0
    assert list(_unit_field(tax_units, single_id, "filer_ids")) == [4]
    assert list(_unit_field(tax_units, single_id, "dependent_ids")) == []

    # The temporary role column must not leak into the returned person frame.
    assert "_microunit_role" not in person_rows.columns


def test_microunit_path_honors_start_tax_unit_id(pipeline):
    persons = _cps_person_frame()

    result = pipeline._build_policyengine_tax_units_via_microunit(
        persons, start_tax_unit_id=100
    )

    assert result is not None
    _, person_rows, _ = result
    assert int(person_rows["tax_unit_id"].min()) >= 100


def test_role_flags_entry_delegates_to_microunit_when_cps_columns_present(pipeline):
    persons = _cps_person_frame()

    via_entry = pipeline._build_policyengine_tax_units_from_role_flags(persons)
    via_direct = pipeline._build_policyengine_tax_units_via_microunit(persons)

    assert via_entry is not None
    assert via_direct is not None
    _, entry_persons, entry_households = via_entry
    _, direct_persons, direct_households = via_direct

    # The entry method must route through microunit (identical assignments),
    # not the legacy role-flag reconstruction, when CPS columns are present.
    assert entry_households == direct_households
    pd.testing.assert_series_equal(
        entry_persons.set_index("person_id")["tax_unit_id"],
        direct_persons.set_index("person_id")["tax_unit_id"],
    )


def test_role_flags_entry_falls_back_without_cps_columns(pipeline):
    # No CPS columns: only microplex role flags. microunit must NOT be used;
    # the legacy role-flag reconstruction handles it (behavior-preserving).
    persons = pd.DataFrame(
        [
            {
                "person_id": 1,
                "household_id": 10,
                "relationship_to_head": 0,
                "age": 40,
                "income": 50000.0,
                "is_tax_unit_head": 1,
                "is_tax_unit_spouse": 0,
                "is_tax_unit_dependent": 0,
            },
            {
                "person_id": 2,
                "household_id": 10,
                "relationship_to_head": 1,
                "age": 38,
                "income": 30000.0,
                "is_tax_unit_head": 0,
                "is_tax_unit_spouse": 1,
                "is_tax_unit_dependent": 0,
            },
        ]
    )

    # The dedicated microunit adapter declines (missing CPS columns)...
    assert pipeline._build_policyengine_tax_units_via_microunit(persons) is None

    # ...but the role-flag entry still succeeds via the legacy path.
    result = pipeline._build_policyengine_tax_units_from_role_flags(persons)
    assert result is not None
    tax_units, person_rows, households = result
    assert households == {10}
    assert person_rows["tax_unit_id"].nunique() == 1
    assert tax_units["filing_status"].iloc[0] == "JOINT"
