"""Tests for the eCPS disability-difficulty recodes (Gate-1 export gap).

The Enhanced CPS exports six person-level ``difficulty_*`` columns recoded from
the ASEC ``PEDIS*`` fields (``PEDIS{X} == 1`` -> ``True``), the same recode it
uses for ``is_blind`` from ``PEDISEYE``. They are eCPS final-H5 contract columns
with no PolicyEngine-US variable, so Microplex exports them as dataset columns
via the legacy-contract entity map. Mirrors policyengine-us-data
``datasets/cps/cps.py`` (unmerged branch ``claude/document-census-tax-id-replacement``).

Microplex already ingested the six ``PEDIS*`` fields into ``_disability_*``
staging columns (used to compute ``is_disabled``) but never produced the
``difficulty_*`` leaves, so they were absent from the export. These tests drive
the real ``_process_persons`` and assert the recode, the staging cleanup, and
the contract/export wiring.
"""

import json
from pathlib import Path

import polars as pl

from microplex_us.data_sources.cps import (
    PERSON_CPS_DIFFICULTY_LEAVES,
    PERSON_VARIABLES,
    _process_persons,
)

_PEDIS_TO_LEAF = {
    "PEDISDRS": "difficulty_dressing_or_bathing",
    "PEDISEAR": "difficulty_hearing",
    "PEDISEYE": "difficulty_seeing",
    "PEDISOUT": "difficulty_doing_errands",
    "PEDISPHY": "difficulty_walking_or_climbing_stairs",
    "PEDISREM": "difficulty_remembering_or_making_decisions",
}

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "microplex_us"
    / "pipelines"
    / "ecps_export_contract.json"
)


def _raw_person_frame(rows: list[dict]) -> pl.DataFrame:
    """Raw CPS-style person frame carrying the six PEDIS* disability fields.

    Census column names are used because ``_process_persons`` selects/renames
    via ``PERSON_VARIABLES``. PEDIS* default to 2 ("no") when unspecified.
    """
    n = len(rows)
    base = {
        "PH_SEQ": [1] * n,
        "A_LINENO": list(range(1, n + 1)),
        "A_FNLWGT": [100.0] * n,
        "A_AGE": [40] * n,
    }
    for pedis in _PEDIS_TO_LEAF:
        base[pedis] = [row.get(pedis, 2) for row in rows]
    return pl.DataFrame(base)


def test_person_variables_maps_the_six_pedis_fields():
    for pedis in _PEDIS_TO_LEAF:
        assert pedis in PERSON_VARIABLES
        assert PERSON_VARIABLES[pedis].startswith("_disability_")


def test_difficulty_leaf_map_covers_all_six_staging_columns():
    assert set(PERSON_CPS_DIFFICULTY_LEAVES.values()) == set(_PEDIS_TO_LEAF.values())


def test_difficulty_leaves_recode_pedis_equals_one():
    """PEDIS{X} == 1 -> True; codes 2 ("no") and 0 ("not in universe") -> False."""
    rows = [
        {p: 1 for p in _PEDIS_TO_LEAF},  # all difficulties
        {p: 2 for p in _PEDIS_TO_LEAF},  # explicit "no"
        {p: 0 for p in _PEDIS_TO_LEAF},  # not in universe
        {"PEDISEYE": 1},  # only vision difficulty
    ]
    result = _process_persons(_raw_person_frame(rows), 2023)

    for leaf in _PEDIS_TO_LEAF.values():
        assert leaf in result.columns, f"{leaf} not produced"
        assert result.schema[leaf] == pl.Boolean, f"{leaf} not boolean"

    # Row 0: every difficulty True; rows 1-2: every difficulty False.
    for leaf in _PEDIS_TO_LEAF.values():
        values = result[leaf].to_list()
        assert values[0] is True, f"{leaf} row0"
        assert values[1] is False and values[2] is False, f"{leaf} rows1-2"

    # Row 3 isolates vision: difficulty_seeing True, the rest False.
    assert result["difficulty_seeing"].to_list() == [True, False, False, True]
    assert result["difficulty_hearing"].to_list() == [True, False, False, False]


def test_difficulty_seeing_tracks_pediseye_like_is_blind():
    # eCPS derives both difficulty_seeing and is_blind from PEDISEYE == 1, so the
    # two must agree on every row.
    rows = [{"PEDISEYE": 1}, {"PEDISEYE": 2}, {"PEDISEYE": 0}]
    result = _process_persons(_raw_person_frame(rows), 2023)
    assert result["difficulty_seeing"].to_list() == [True, False, False]


def test_staging_disability_columns_do_not_leak():
    result = _process_persons(_raw_person_frame([{"PEDISEYE": 1}]), 2023)
    for staging in PERSON_CPS_DIFFICULTY_LEAVES:  # keys are the _disability_* staging
        assert staging not in result.columns
    # is_disabled is still computed from the same staging signal.
    assert "is_disabled" in result.columns
    assert result["is_disabled"].to_list() == [True]


def test_difficulty_leaves_are_contract_required():
    required = set(json.loads(_CONTRACT_PATH.read_text())["required"])
    for leaf in _PEDIS_TO_LEAF.values():
        assert leaf in required, f"{leaf} not in contract required"


def test_difficulty_leaves_export_wiring():
    from microplex_us.policyengine.us import (
        POLICYENGINE_US_EXPORT_DEFAULTS,
        POLICYENGINE_US_LEGACY_CONTRACT_VARIABLE_ENTITIES,
        SAFE_POLICYENGINE_US_EXPORT_VARIABLES,
    )

    for leaf in _PEDIS_TO_LEAF.values():
        assert leaf in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert POLICYENGINE_US_EXPORT_DEFAULTS[leaf] is False
        # Not a pe-us variable -> routed as a person-level dataset column.
        assert POLICYENGINE_US_LEGACY_CONTRACT_VARIABLE_ENTITIES[leaf] == "person"
