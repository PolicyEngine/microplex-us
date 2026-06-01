"""Tests for the CPS-derived direct income copies (G7 export-parity gap).

The Enhanced CPS exports three person-level income leaves as direct copies of
raw ASEC fields (``policyengine_us_data/datasets/cps/cps.py:1493-1495``):

- ``survivor_benefits``       <- ``SRVS_VAL``
- ``educational_assistance``  <- ``ED_VAL``
- ``financial_assistance``    <- ``FIN_VAL``

Microplex produced none of them: the raw fields were not mapped in
``PERSON_VARIABLES`` and the leaves were absent from the export allowlist, so
they never reached the H5. These tests exercise the real ``_process_persons``
(no stubbing) to prove the rename happens, plus assert allowlist membership and
that no alias remaps the leaves.
"""

import polars as pl

from microplex_us.data_sources.cps import PERSON_VARIABLES, _process_persons

_COPIES = {
    "SRVS_VAL": "survivor_benefits",
    "ED_VAL": "educational_assistance",
    "FIN_VAL": "financial_assistance",
}


def _raw_person_frame(rows: list[dict]) -> pl.DataFrame:
    """Raw CPS-style person frame carrying the income-copy fields.

    Census column names are used because ``_process_persons`` selects/renames
    via ``PERSON_VARIABLES``.
    """
    n = len(rows)
    return pl.DataFrame(
        {
            "PH_SEQ": [1] * n,
            "A_LINENO": list(range(1, n + 1)),
            "A_FNLWGT": [100.0] * n,
            "A_AGE": [row.get("age", 40) for row in rows],
            "SRVS_VAL": [row.get("srvs", 0.0) for row in rows],
            "ED_VAL": [row.get("ed", 0.0) for row in rows],
            "FIN_VAL": [row.get("fin", 0.0) for row in rows],
        }
    )


def test_person_variables_maps_the_three_raw_fields():
    for census, leaf in _COPIES.items():
        assert PERSON_VARIABLES.get(census) == leaf


def test_process_persons_copies_raw_fields_to_leaves():
    """The raw ASEC values are copied verbatim onto the pe-us input leaves."""
    rows = [
        {"srvs": 12_000.0, "ed": 0.0, "fin": 0.0},
        {"srvs": 0.0, "ed": 5_000.0, "fin": 0.0},
        {"srvs": 0.0, "ed": 0.0, "fin": 3_200.0},
        {"srvs": 800.0, "ed": 1_100.0, "fin": 450.0},
        {"srvs": 0.0, "ed": 0.0, "fin": 0.0},  # non-recipient
    ]
    result = _process_persons(_raw_person_frame(rows), 2023)

    for census, leaf in _COPIES.items():
        assert leaf in result.columns, f"{leaf} not produced"
        got = result[leaf].to_list()
        expected = [row.get(_FIELD_FOR[census], 0.0) for row in rows]
        assert got == expected, f"{leaf}: {got} != {expected}"


_FIELD_FOR = {"SRVS_VAL": "srvs", "ED_VAL": "ed", "FIN_VAL": "fin"}


def test_copies_are_non_degenerate():
    """Each leaf carries distinct nonzero values, not a constant/zero fill."""
    rows = [
        {"srvs": 9_000.0, "ed": 2_000.0, "fin": 1_500.0},
        {"srvs": 21_000.0, "ed": 6_500.0, "fin": 4_000.0},
        {"srvs": 0.0, "ed": 0.0, "fin": 0.0},
    ]
    result = _process_persons(_raw_person_frame(rows), 2023)
    for leaf in _COPIES.values():
        values = [v for v in result[leaf].to_list() if v > 0]
        assert len(values) >= 2, f"{leaf} should be positive for several records"
        assert len(set(values)) >= 2, f"{leaf} should not be a single constant"


def test_copies_in_export_allowlist_and_not_aliased():
    from microplex_us.policyengine.us import (
        POLICYENGINE_US_EXPORT_COLUMN_ALIASES,
        SAFE_POLICYENGINE_US_EXPORT_VARIABLES,
    )

    for leaf in _COPIES.values():
        assert leaf in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        assert POLICYENGINE_US_EXPORT_COLUMN_ALIASES.get(leaf) is None


if __name__ == "__main__":
    import traceback

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"SUMMARY passed={passed} failed={failed}")
    raise SystemExit(1 if failed else 0)
