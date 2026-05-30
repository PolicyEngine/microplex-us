"""SPM-unit-id preservation (#113).

The authoritative SPM unit ids carried by the source records are eCPS-quality
(~1.04 units/household). Synthesis can leave some records without an id; the old
all-or-nothing logic discarded the whole column on any missing id, collapsing to
one SPM unit per household (~1.00). These tests pin the preserve-present-fill-
missing behavior that keeps the authoritative structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microplex_us.pipelines.us import USMicroplexPipeline


def _pipe() -> USMicroplexPipeline:
    return USMicroplexPipeline()


def test_preserve_present_keeps_distinct_units():
    # Two SPM units in one household, all ids present -> both preserved.
    persons = pd.DataFrame(
        {"household_id": [1, 1, 1, 1], "spm_unit_id": [10, 10, 11, 11]}
    )
    out = _pipe()._preserve_present_group_ids(persons, "spm_unit_id")
    assert out.nunique() == 2
    assert out.iloc[0] == out.iloc[1]
    assert out.iloc[2] == out.iloc[3]
    assert out.iloc[0] != out.iloc[2]


def test_missing_rows_fold_into_present_unit_not_new():
    # A missing id in a household that has present ids must NOT fabricate a unit.
    persons = pd.DataFrame(
        {"household_id": [1, 1, 1], "spm_unit_id": [10.0, 10.0, np.nan]}
    )
    out = _pipe()._preserve_present_group_ids(persons, "spm_unit_id")
    assert out.nunique() == 1  # folded into the present unit, not split to 2


def test_fully_missing_household_gets_one_fallback_when_others_present():
    persons = pd.DataFrame(
        {"household_id": [1, 1, 2, 2], "spm_unit_id": [10.0, 10.0, np.nan, np.nan]}
    )
    out = _pipe()._preserve_present_group_ids(persons, "spm_unit_id")
    assert out.iloc[0] == out.iloc[1]  # hh1 preserved
    assert out.iloc[2] == out.iloc[3]  # hh2 -> one fallback unit
    assert out.iloc[0] != out.iloc[2]


def test_all_missing_returns_none():
    persons = pd.DataFrame({"household_id": [1, 1], "spm_unit_id": [np.nan, np.nan]})
    # Entirely empty column -> None so the caller regenerates from scratch.
    assert _pipe()._preserve_present_group_ids(persons, "spm_unit_id") is None


def test_assign_family_and_spm_preserves_partial_spm():
    # End to end: a partially-missing SPM column keeps the present structure
    # rather than collapsing to one unit per household.
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "household_id": [1, 1, 1, 2, 2],
            "relationship_to_head": [0, 1, 2, 0, 2],
            "spm_unit_id": [10.0, 10.0, 11.0, np.nan, np.nan],
        }
    )
    out = _pipe()._assign_family_and_spm_units(persons)
    per_hh = out.groupby("household_id")["spm_unit_id"].nunique()
    assert int(per_hh.loc[1]) == 2  # two present units preserved
    assert int(per_hh.loc[2]) == 1  # fully-missing household -> one fallback
