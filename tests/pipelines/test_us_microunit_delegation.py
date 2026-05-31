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


def test_microunit_path_preserves_alignment_under_unsorted_input(pipeline):
    """microunit groups internally by PH_SEQ but returns per-person results in
    input row order; the delegation maps TAX_ID/role back positionally, so the
    mapping must stay correct even when input rows are not pre-sorted by
    PH_SEQ/A_LINENO.

    Regression guard: if a future microunit change stopped preserving input row
    order, the positional mapping would misassign people to units/roles and this
    test would fail while the pre-sorted fixture test still passed.
    """
    persons = _cps_person_frame()
    # Input order != PH_SEQ order: single adult (HH 20) first, then the HH 10
    # members out of A_LINENO order (2, 1, 3).
    shuffled = persons.iloc[[3, 1, 0, 2]].reset_index(drop=True)

    result = pipeline._build_policyengine_tax_units_via_microunit(shuffled)
    assert result is not None
    tax_units, person_rows, _ = result

    unit_by_person = dict(
        zip(person_rows["person_id"], person_rows["tax_unit_id"], strict=True)
    )
    # Couple + child still share one unit; the single adult stays separate.
    assert unit_by_person[1] == unit_by_person[2] == unit_by_person[3]
    assert unit_by_person[4] != unit_by_person[1]

    # Roles/filing must stay correctly aligned, not just the grouping.
    family_id = int(unit_by_person[1])
    assert _unit_field(tax_units, family_id, "filing_status") == "JOINT"
    assert sorted(_unit_field(tax_units, family_id, "filer_ids")) == [1, 2]
    assert list(_unit_field(tax_units, family_id, "dependent_ids")) == [3]


def _normalized_person_frame() -> pd.DataFrame:
    """A purely NORMALIZED microplex frame (no raw CPS columns).

    The same population as ``_cps_person_frame`` expressed only in microplex's
    materialization columns: a married couple with one child (HH 10) and a
    single adult (HH 20). Used to exercise the prototype adapter (#115) that
    synthesizes microunit's CPS contract from these columns.
    """
    return pd.DataFrame(
        [
            {
                "person_id": 1,
                "household_id": 10,
                "relationship_to_head": 0,
                "age": 40,
                "income": 50000.0,
            },
            {
                "person_id": 2,
                "household_id": 10,
                "relationship_to_head": 1,
                "age": 38,
                "income": 30000.0,
            },
            {
                "person_id": 3,
                "household_id": 10,
                "relationship_to_head": 2,
                "age": 10,
                "income": 0.0,
            },
            {
                "person_id": 4,
                "household_id": 20,
                "relationship_to_head": 0,
                "age": 25,
                "income": 45000.0,
            },
        ]
    )


def test_normalized_adapter_is_off_by_default(pipeline):
    persons = _normalized_person_frame()
    # No raw CPS columns and the adapter is not enabled -> declines (the default
    # behavior is unchanged; the legacy role-flag path handles such frames).
    assert pipeline._build_policyengine_tax_units_via_microunit(persons) is None


def test_normalized_adapter_activates_delegation_when_enabled(pipeline):
    persons = _normalized_person_frame()

    result = pipeline._build_policyengine_tax_units_via_microunit(
        persons, allow_normalized_adapter=True
    )

    assert result is not None, (
        "adapter must let the delegation fire from a normalized frame"
    )
    tax_units, person_rows, households = result
    assert households == {10, 20}

    unit_by_person = dict(
        zip(person_rows["person_id"], person_rows["tax_unit_id"], strict=True)
    )
    # Couple + child collapse into one unit; the single adult stays separate.
    assert unit_by_person[1] == unit_by_person[2] == unit_by_person[3]
    assert unit_by_person[4] != unit_by_person[1]

    family_id = int(unit_by_person[1])
    assert _unit_field(tax_units, family_id, "filing_status") == "JOINT"
    assert int(_unit_field(tax_units, family_id, "n_dependents")) == 1


def test_normalized_adapter_builds_microunit_cps_contract(pipeline):
    persons = _normalized_person_frame()
    frame = pipeline._microunit_cps_frame_from_normalized(persons)
    assert frame is not None
    # All eight microunit-required CPS columns are present.
    for col in (
        "PH_SEQ",
        "A_LINENO",
        "A_AGE",
        "A_MARITL",
        "A_SPOUSE",
        "PEPAR1",
        "PEPAR2",
        "A_EXPRRP",
    ):
        assert col in frame.columns, f"missing {col}"
    by_pid = frame.set_index("person_id")
    # Couple are married (A_MARITL == 1) and point at each other's line numbers.
    assert int(by_pid.loc[1, "A_MARITL"]) == 1
    assert int(by_pid.loc[1, "A_SPOUSE"]) == int(by_pid.loc[2, "A_LINENO"])
    assert int(by_pid.loc[2, "A_SPOUSE"]) == int(by_pid.loc[1, "A_LINENO"])
    # The child's parent pointers reference the head and spouse line numbers.
    assert int(by_pid.loc[3, "PEPAR1"]) == int(by_pid.loc[1, "A_LINENO"])
    assert int(by_pid.loc[3, "PEPAR2"]) == int(by_pid.loc[2, "A_LINENO"])
    # The single adult is never-married with no spouse/parent pointers.
    assert int(by_pid.loc[4, "A_MARITL"]) == 7
    assert int(by_pid.loc[4, "A_SPOUSE"]) == 0


def test_microunit_path_fails_safe_to_none(pipeline, monkeypatch):
    # If microunit raises on a household it cannot resolve, the delegation must
    # fall back (return None) rather than crash materialization.
    import microunit

    def _boom(*args, **kwargs):
        raise ValueError("synthetic microunit failure")

    monkeypatch.setattr(microunit, "construct_tax_units", _boom)
    persons = _cps_person_frame()  # raw CPS columns present -> reaches microunit
    assert pipeline._build_policyengine_tax_units_via_microunit(persons) is None


def test_normalized_adapter_promotes_a_head_when_none_marked(pipeline):
    # A household where nobody is marked head (all "other") must still get a
    # single reference person so microunit can construct it.
    persons = pd.DataFrame(
        [
            {
                "person_id": 1,
                "household_id": 30,
                "relationship_to_head": 3,
                "age": 70,
                "income": 20000.0,
            },
            {
                "person_id": 2,
                "household_id": 30,
                "relationship_to_head": 3,
                "age": 35,
                "income": 15000.0,
            },
        ]
    )
    frame = pipeline._microunit_cps_frame_from_normalized(persons)
    assert frame is not None
    # Exactly one reference person (A_EXPRRP == 1) in the household.
    assert int((frame["A_EXPRRP"] == 1).sum()) == 1
    # And the delegation must not raise (fail-safe covers microunit edge cases).
    pipeline._build_policyengine_tax_units_via_microunit(
        persons, allow_normalized_adapter=True
    )


def _cps_fields_frame() -> pd.DataFrame:
    """A frame in microplex's real CPS-derived fields (no relationship_to_head):
    person_number = within-household line, spouse_person_number = spouse line,
    family_relationship = CPS A_FAMREL. Couple+child (HH 10), single (HH 20)."""
    return pd.DataFrame(
        [
            {
                "person_id": 1,
                "household_id": 10,
                "age": 40,
                "income": 50000.0,
                "person_number": 1,
                "spouse_person_number": 2,
                "family_relationship": 1,
            },
            {
                "person_id": 2,
                "household_id": 10,
                "age": 38,
                "income": 30000.0,
                "person_number": 2,
                "spouse_person_number": 1,
                "family_relationship": 2,
            },
            {
                "person_id": 3,
                "household_id": 10,
                "age": 10,
                "income": 0.0,
                "person_number": 3,
                "spouse_person_number": 0,
                "family_relationship": 3,
            },
            {
                "person_id": 4,
                "household_id": 20,
                "age": 25,
                "income": 45000.0,
                "person_number": 1,
                "spouse_person_number": 0,
                "family_relationship": 0,
            },
        ]
    )


def test_high_fidelity_adapter_uses_real_pointers(pipeline):
    persons = _cps_fields_frame()
    # The normalized adapter must dispatch to the high-fidelity path when the real
    # CPS-derived fields are present (no relationship_to_head needed).
    frame = pipeline._microunit_cps_frame_from_normalized(persons)
    assert frame is not None
    by = frame.set_index("person_id")
    # Real line/spouse pointers, not heuristics.
    assert int(by.loc[1, "A_LINENO"]) == 1 and int(by.loc[2, "A_LINENO"]) == 2
    assert int(by.loc[1, "A_SPOUSE"]) == 2 and int(by.loc[2, "A_SPOUSE"]) == 1
    # A_EXPRRP from family relationship: ref=1, spouse=3, own child=5.
    assert int(by.loc[1, "A_EXPRRP"]) == 1
    assert int(by.loc[2, "A_EXPRRP"]) == 3
    assert int(by.loc[3, "A_EXPRRP"]) == 5
    # Child's parent pointers reference the head (line 1) and spouse (line 2).
    assert int(by.loc[3, "PEPAR1"]) == 1 and int(by.loc[3, "PEPAR2"]) == 2


def test_high_fidelity_adapter_activates_delegation(pipeline):
    persons = _cps_fields_frame()
    result = pipeline._build_policyengine_tax_units_via_microunit(
        persons, allow_normalized_adapter=True
    )
    assert result is not None
    tax_units, person_rows, households = result
    unit_by_person = dict(
        zip(person_rows["person_id"], person_rows["tax_unit_id"], strict=True)
    )
    # Couple + child form one unit; single adult separate.
    assert unit_by_person[1] == unit_by_person[2] == unit_by_person[3]
    assert unit_by_person[4] != unit_by_person[1]
    family_id = int(unit_by_person[1])
    assert _unit_field(tax_units, family_id, "filing_status") == "JOINT"


def _zero_based_cps_fields_frame() -> pd.DataFrame:
    """Same couple+child household as ``_cps_fields_frame`` but with
    family_relationship in the optimizer's 0-based coding (0=head, 1=spouse,
    2=child) instead of CPS A_FAMREL 1-based (1=ref, 2=spouse, 3=child)."""
    return pd.DataFrame(
        [
            {
                "person_id": 1,
                "household_id": 10,
                "age": 40,
                "income": 50000.0,
                "person_number": 1,
                "spouse_person_number": 2,
                "family_relationship": 0,
            },
            {
                "person_id": 2,
                "household_id": 10,
                "age": 38,
                "income": 30000.0,
                "person_number": 2,
                "spouse_person_number": 1,
                "family_relationship": 1,
            },
            {
                "person_id": 3,
                "household_id": 10,
                "age": 10,
                "income": 0.0,
                "person_number": 3,
                "spouse_person_number": 0,
                "family_relationship": 2,
            },
        ]
    )


def test_high_fidelity_adapter_normalizes_zero_based_family_relationship(pipeline):
    # A 0-based family_relationship frame must map to the SAME A_EXPRRP / parent
    # pointers as the 1-based CPS A_FAMREL coding. Without per-household scheme
    # normalization the child (code 2) is mis-read as a spouse and loses parent
    # pointers, so microunit mis-partitions the household.
    frame = pipeline._microunit_cps_frame_from_normalized(
        _zero_based_cps_fields_frame()
    )
    assert frame is not None
    by = frame.set_index("person_id")
    assert int(by.loc[1, "A_EXPRRP"]) == 1  # head
    assert int(by.loc[2, "A_EXPRRP"]) == 3  # spouse, not other-relative
    assert int(by.loc[3, "A_EXPRRP"]) == 5  # own child, not spouse
    assert int(by.loc[3, "PEPAR1"]) == 1 and int(by.loc[3, "PEPAR2"]) == 2

    # And the partition matches the 1-based frame: couple + child = one unit.
    result = pipeline._build_policyengine_tax_units_via_microunit(
        _zero_based_cps_fields_frame(), allow_normalized_adapter=True
    )
    assert result is not None
    _, person_rows, _ = result
    unit_by_person = dict(
        zip(person_rows["person_id"], person_rows["tax_unit_id"], strict=True)
    )
    assert unit_by_person[1] == unit_by_person[2] == unit_by_person[3]
