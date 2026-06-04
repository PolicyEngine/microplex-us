"""Tests for the CPS retirement-contribution desired-leaf split (G2 gap).

The Enhanced CPS splits the single bundled CPS retirement-contribution total
(``RETCB_VAL``) into five account-type-specific *desired* (pre-statutory-limit)
contribution leaves, using a proportional split with IRS SOI / BEA-FRED /
Vanguard-PSCA shares (PolicyEngine/policyengine-us-data
``policyengine_us_data/datasets/cps/cps.py:1500-1552``; share values in
``policyengine_us_data/datasets/cps/imputation_parameters.yaml``):

- ``self_employed_pension_contributions_desired`` (se share, gated on SE income)
- ``traditional_401k_contributions_desired``      (DC pool, gated on wages)
- ``roth_401k_contributions_desired``             (DC pool, gated on wages)
- ``traditional_ira_contributions_desired``       (IRA pool, any earned income)
- ``roth_ira_contributions_desired``              (IRA pool, any earned income)

Microplex produced none of them: ``RETCB_VAL`` was not mapped in
``PERSON_VARIABLES`` and the leaves were absent from the export allowlist, so
they never reached the H5. These tests exercise the real ``_process_persons``
(no stubbing) to prove the split happens with the exact eCPS gating and share
math, that the five leaves reconcile to ``RETCB_VAL`` for earned-income
records, that gating zeroes the right pools, and that the leaves are in the
PolicyEngine-US export allowlist and not aliased away.
"""

import numpy as np
import polars as pl

from microplex_us.data_sources.cps import (
    DC_SHARE_OF_RETIREMENT_CONTRIBUTIONS,
    PERSON_VARIABLES,
    RETIREMENT_CONTRIBUTION_LIMITS_BY_YEAR,
    ROTH_SHARE_OF_DC_CONTRIBUTIONS,
    SE_PENSION_SHARE_OF_RETIREMENT_CONTRIBUTIONS,
    TRADITIONAL_SHARE_OF_IRA_CONTRIBUTIONS,
    _process_persons,
)

_LEAVES = (
    "self_employed_pension_contributions_desired",
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
)
_CAPPED_LEAVES = (
    "self_employed_pension_contributions",
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
)


def _raw_person_frame(rows: list[dict]) -> pl.DataFrame:
    """Raw CPS-style person frame carrying RETCB_VAL plus the earned-income
    fields that gate the split.

    Census column names are used because ``_process_persons`` selects/renames
    via ``PERSON_VARIABLES`` (RETCB_VAL -> _retirement_contributions ->
    desired leaves).
    """
    n = len(rows)
    return pl.DataFrame(
        {
            "PH_SEQ": [1] * n,
            "A_LINENO": list(range(1, n + 1)),
            "A_FNLWGT": [100.0] * n,
            "A_AGE": [row.get("age", 40) for row in rows],
            "WSAL_VAL": [float(row.get("wages", 0.0)) for row in rows],
            "SEMP_VAL": [float(row.get("se", 0.0)) for row in rows],
            "RETCB_VAL": [float(row.get("retcb", 0.0)) for row in rows],
        }
    )


def _expected_split(retcb: float, wages: float, se: float) -> dict[str, float]:
    """Recompute the eCPS split independently (cps.py:1514-1552)."""
    has_wages = wages > 0
    has_se = se > 0
    has_earned = has_wages or has_se
    se_pension = retcb * SE_PENSION_SHARE_OF_RETIREMENT_CONTRIBUTIONS if has_se else 0.0
    remaining = max(retcb - se_pension, 0.0)
    dc_pool = remaining * DC_SHARE_OF_RETIREMENT_CONTRIBUTIONS if has_wages else 0.0
    ira_pool = (remaining - dc_pool) if has_earned else 0.0
    return {
        "self_employed_pension_contributions_desired": se_pension,
        "traditional_401k_contributions_desired": dc_pool
        * (1 - ROTH_SHARE_OF_DC_CONTRIBUTIONS),
        "roth_401k_contributions_desired": dc_pool * ROTH_SHARE_OF_DC_CONTRIBUTIONS,
        "traditional_ira_contributions_desired": ira_pool
        * TRADITIONAL_SHARE_OF_IRA_CONTRIBUTIONS,
        "roth_ira_contributions_desired": ira_pool
        * (1 - TRADITIONAL_SHARE_OF_IRA_CONTRIBUTIONS),
    }


def _expected_capped_split(
    retcb: float,
    wages: float,
    se: float,
    age: int,
    *,
    year: int,
) -> dict[str, float]:
    """Recompute the final account leaves by capping each desired pool."""
    limits = RETIREMENT_CONTRIBUTION_LIMITS_BY_YEAR[year]
    catch_up = age >= 50
    limit_401k = limits["401k"] + catch_up * limits["401k_catch_up"]
    limit_ira = limits["ira"] + catch_up * limits["ira_catch_up"]
    desired = _expected_split(retcb, wages, se)
    traditional_401k = min(
        desired["traditional_401k_contributions_desired"],
        limit_401k,
    )
    roth_401k = min(
        desired["roth_401k_contributions_desired"],
        max(limit_401k - traditional_401k, 0.0),
    )
    traditional_ira = min(
        desired["traditional_ira_contributions_desired"],
        limit_ira,
    )
    roth_ira = min(
        desired["roth_ira_contributions_desired"],
        max(limit_ira - traditional_ira, 0.0),
    )
    return {
        "self_employed_pension_contributions": desired[
            "self_employed_pension_contributions_desired"
        ],
        "traditional_401k_contributions": traditional_401k,
        "roth_401k_contributions": roth_401k,
        "traditional_ira_contributions": traditional_ira,
        "roth_ira_contributions": roth_ira,
    }


def test_share_constants_trace_to_imputation_parameters_yaml():
    """The four scalars must equal the eCPS imputation_parameters.yaml values."""
    assert SE_PENSION_SHARE_OF_RETIREMENT_CONTRIBUTIONS == 0.046  # yaml line 30
    assert DC_SHARE_OF_RETIREMENT_CONTRIBUTIONS == 0.908  # yaml line 38
    assert ROTH_SHARE_OF_DC_CONTRIBUTIONS == 0.15  # yaml line 48
    assert TRADITIONAL_SHARE_OF_IRA_CONTRIBUTIONS == 0.392  # yaml line 55


def test_person_variables_stages_retcb_val():
    """RETCB_VAL is mapped to the staging column consumed by the split."""
    assert PERSON_VARIABLES.get("RETCB_VAL") == "_retirement_contributions"


def test_process_persons_produces_retirement_leaves_and_drops_staging():
    rows = [
        {"wages": 50_000.0, "se": 0.0, "retcb": 10_000.0},
        {"wages": 0.0, "se": 80_000.0, "retcb": 5_000.0},
        {"wages": 30_000.0, "se": 10_000.0, "retcb": 2_000.0},
    ]
    out = _process_persons(_raw_person_frame(rows), 2023)
    for leaf in _LEAVES + _CAPPED_LEAVES:
        assert leaf in out.columns, f"{leaf} not produced"
    # Staging column must not leak into the processed frame.
    assert "_retirement_contributions" not in out.columns


def test_split_matches_ecps_math_exactly():
    """Every leaf equals the independent eCPS recomputation, row by row."""
    rows = [
        {"wages": 50_000.0, "se": 0.0, "retcb": 10_000.0},  # wages only
        {"wages": 0.0, "se": 80_000.0, "retcb": 5_000.0},  # SE only
        {"wages": 30_000.0, "se": 10_000.0, "retcb": 2_000.0},  # both
        {"wages": 90_000.0, "se": 0.0, "retcb": 23_000.0},  # high earner
    ]
    out = _process_persons(_raw_person_frame(rows), 2023)
    for i, row in enumerate(rows):
        expected = _expected_split(row["retcb"], row["wages"], row["se"])
        for leaf in _LEAVES:
            got = out[leaf].to_list()[i]
            assert got == expected[leaf], f"row {i} {leaf}: {got} != {expected[leaf]}"


def test_capped_split_matches_desired_account_pools_with_limits():
    rows = [
        {"wages": 50_000.0, "se": 0.0, "retcb": 10_000.0, "age": 40},
        {"wages": 0.0, "se": 80_000.0, "retcb": 5_000.0, "age": 40},
        {"wages": 90_000.0, "se": 0.0, "retcb": 80_000.0, "age": 52},
    ]
    out = _process_persons(_raw_person_frame(rows), 2023)
    for i, row in enumerate(rows):
        expected = _expected_capped_split(
            row["retcb"],
            row["wages"],
            row["se"],
            row["age"],
            year=2023,
        )
        for leaf in _CAPPED_LEAVES:
            got = out[leaf].to_list()[i]
            assert got == expected[leaf], f"row {i} {leaf}: {got} != {expected[leaf]}"


def test_capped_split_preserves_ira_support_below_401k_limit():
    rows = [
        {"wages": 50_000.0, "se": 0.0, "retcb": 10_000.0, "age": 40},
    ]
    out = _process_persons(_raw_person_frame(rows), 2024)

    assert out["traditional_401k_contributions"].to_list()[0] > 0
    assert out["roth_401k_contributions"].to_list()[0] > 0
    assert out["traditional_ira_contributions"].to_list()[0] > 0
    assert out["roth_ira_contributions"].to_list()[0] > 0


def test_five_leaves_reconcile_to_retcb_for_earned_income_records():
    """For anyone with earned income the five leaves sum back to RETCB_VAL."""
    rows = [
        {"wages": 50_000.0, "se": 0.0, "retcb": 10_000.0},
        {"wages": 0.0, "se": 80_000.0, "retcb": 5_000.0},
        {"wages": 30_000.0, "se": 10_000.0, "retcb": 2_000.0},
        {"wages": 20_000.0, "se": 0.0, "retcb": 7_345.67},
    ]
    out = _process_persons(_raw_person_frame(rows), 2023)
    total = np.zeros(len(rows))
    for leaf in _LEAVES:
        total += np.array(out[leaf].to_list(), dtype=float)
    retcb = np.array([row["retcb"] for row in rows], dtype=float)
    assert np.allclose(total, retcb, atol=1e-6), f"{total} != {retcb}"


def test_se_pension_gated_on_self_employment_income():
    """Self-employed pension is nonzero only when SE income is positive."""
    rows = [
        {"wages": 40_000.0, "se": 0.0, "retcb": 6_000.0},  # no SE -> 0 se pension
        {"wages": 0.0, "se": 40_000.0, "retcb": 6_000.0},  # SE -> nonzero se pension
        {"wages": 40_000.0, "se": 40_000.0, "retcb": 6_000.0},  # both -> nonzero
    ]
    out = _process_persons(_raw_person_frame(rows), 2023)
    se = out["self_employed_pension_contributions_desired"].to_list()
    assert se[0] == 0.0
    assert se[1] > 0.0
    assert se[2] > 0.0
    # Exact value when SE present: se_share * RETCB.
    assert se[1] == 6_000.0 * SE_PENSION_SHARE_OF_RETIREMENT_CONTRIBUTIONS


def test_dc_401k_gated_on_wages():
    """401(k) pool is nonzero only when wages are positive."""
    rows = [
        {"wages": 0.0, "se": 50_000.0, "retcb": 8_000.0},  # SE only -> 401k == 0
        {"wages": 50_000.0, "se": 0.0, "retcb": 8_000.0},  # wages -> 401k > 0
    ]
    out = _process_persons(_raw_person_frame(rows), 2023)
    trad = out["traditional_401k_contributions_desired"].to_list()
    roth = out["roth_401k_contributions_desired"].to_list()
    assert trad[0] == 0.0 and roth[0] == 0.0, "no 401(k) without an employer/wages"
    assert trad[1] > 0.0 and roth[1] > 0.0, "401(k) expected with wages"


def test_no_earned_income_yields_zero_everywhere():
    """A record with RETCB but no wages and no SE produces all-zero leaves."""
    rows = [{"wages": 0.0, "se": 0.0, "retcb": 9_000.0}]
    out = _process_persons(_raw_person_frame(rows), 2023)
    for leaf in _LEAVES:
        assert out[leaf].to_list()[0] == 0.0, (
            f"{leaf} should be 0 without earned income"
        )


def test_zero_retcb_yields_zero_everywhere():
    """A worker who contributed nothing has zero across all five leaves."""
    rows = [{"wages": 60_000.0, "se": 20_000.0, "retcb": 0.0}]
    out = _process_persons(_raw_person_frame(rows), 2023)
    for leaf in _LEAVES:
        assert out[leaf].to_list()[0] == 0.0, f"{leaf} should be 0 with zero RETCB"


def test_split_is_non_degenerate():
    """Across records each leaf takes several distinct positive values."""
    rows = [
        {"wages": 50_000.0, "se": 5_000.0, "retcb": 12_000.0},
        {"wages": 80_000.0, "se": 0.0, "retcb": 20_000.0},
        {"wages": 0.0, "se": 70_000.0, "retcb": 9_000.0},
        {"wages": 35_000.0, "se": 15_000.0, "retcb": 6_500.0},
    ]
    out = _process_persons(_raw_person_frame(rows), 2023)
    for leaf in _LEAVES:
        positive = [v for v in out[leaf].to_list() if v > 0]
        assert len(positive) >= 2, f"{leaf} should be positive for several records"
        assert len(set(positive)) >= 2, f"{leaf} should not be a single constant"


def test_leaves_in_export_allowlist_and_not_aliased():
    from microplex_us.policyengine.us import (
        POLICYENGINE_US_EXPORT_COLUMN_ALIASES,
        SAFE_POLICYENGINE_US_EXPORT_VARIABLES,
    )

    for leaf in _LEAVES + _CAPPED_LEAVES:
        assert leaf in SAFE_POLICYENGINE_US_EXPORT_VARIABLES, f"{leaf} not exported"
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
