"""Tests for the CPS-derived recodes closing the G6/G8/G9 eCPS export gaps.

The Enhanced CPS exports four person-level leaves that Microplex did not:

- ``is_unmarried_partner_of_household_head`` (G8) -- a recode of the ASEC
  relationship-to-householder code ``PERRP``: codes 43/44/46/47 mark an
  unmarried partner of the household head. Mirrors eCPS
  ``policyengine_us_data/datasets/cps/cps.py:1219``
  (``perrp.isin(PERRP_UNMARRIED_PARTNER_OF_HOUSEHOLD_HEAD_CODES)``).

- ``reported_owns_employer_sponsored_health_insurance_at_interview`` (G6a) --
  the ESI policyholder flag ``NOW_OWNGRP == 1``. Mirrors eCPS cps.py:1576-1578.

- ``employer_sponsored_insurance_premiums`` (G6b) -- annual employer-paid ESI
  premium imputed from ``NOW_OWNGRP``/``NOW_HIPAID``/``NOW_GRPFTYP``/``PHIP_VAL``
  plus the MEPS-IC 2024 plan-type priors. Reproduces eCPS
  ``impute_employer_sponsored_insurance_premiums`` (cps.py:229-273) verbatim;
  the expected values below are the eCPS reference function's own outputs.

- ``sstb_self_employment_income_would_be_qualified`` (G9) -- the SSTB QBI
  qualification flag. eCPS computes it in its PUF QBI simulation as
  ``np.where(business_is_sstb, self_employment_income_would_be_qualified,
  False)`` (puf.py:768). Microplex runs no PUF QBI simulation and already
  exports ``business_is_sstb=False`` for every record, so that expression
  collapses to a constant ``False``; G9 is therefore exported as a constant
  default rather than a CPS recode.

The recode tests exercise the real ``_process_persons`` (no stubbing). The
ESI-premium expectations are cross-checked against the eCPS reference
implementation in ``tests/.../test_employer_sponsored_insurance_premiums.py``.
"""

import math

import polars as pl

from microplex_us.data_sources.cps import (
    ESI_PLAN_PRIORS_2024,
    PERRP_UNMARRIED_PARTNER_OF_HOUSEHOLD_HEAD_CODES,
    PERSON_VARIABLES,
    _process_persons,
)

# Raw ASEC fields that seed the new leaves, mapped via PERSON_VARIABLES to the
# underscore-prefixed staging columns consumed inside _process_persons.
_RAW_STAGING_COLUMNS = {
    "PERRP": "_person_relationship_to_householder",
    "NOW_OWNGRP": "_now_owngrp",
    "NOW_HIPAID": "_now_hipaid",
    "NOW_GRPFTYP": "_now_grpftyp",
}

# Leaves produced by the new recodes (G6/G8). G9 is a constant default, not a
# recode, so it is asserted via the export config, not here.
_RECODE_LEAVES = (
    "is_unmarried_partner_of_household_head",
    "reported_owns_employer_sponsored_health_insurance_at_interview",
    "employer_sponsored_insurance_premiums",
)


def _raw_person_frame(rows: list[dict]) -> pl.DataFrame:
    """Raw CPS-style person frame carrying the new recode source fields.

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
            "PERRP": [row.get("perrp", 40) for row in rows],
            "NOW_OWNGRP": [row.get("owngrp", 0) for row in rows],
            "NOW_HIPAID": [row.get("hipaid", 0) for row in rows],
            "NOW_GRPFTYP": [row.get("grpftyp", 0) for row in rows],
            "PHIP_VAL": [row.get("phip", 0.0) for row in rows],
        }
    )


# ---------------------------------------------------------------------------
# G8: is_unmarried_partner_of_household_head (PERRP recode)
# ---------------------------------------------------------------------------


def test_person_variables_maps_the_new_raw_fields():
    for census, staging in _RAW_STAGING_COLUMNS.items():
        assert PERSON_VARIABLES.get(census) == staging


def test_unmarried_partner_codes_match_ecps():
    # eCPS PERRP_UNMARRIED_PARTNER_OF_HOUSEHOLD_HEAD_CODES = {43, 44, 46, 47}.
    assert set(PERRP_UNMARRIED_PARTNER_OF_HOUSEHOLD_HEAD_CODES) == {43, 44, 46, 47}


def test_unmarried_partner_flag_recode():
    rows = [
        {"perrp": 43},  # opposite-sex partner with relatives -> True
        {"perrp": 44},  # opposite-sex partner without relatives -> True
        {"perrp": 46},  # same-sex partner with relatives -> True
        {"perrp": 47},  # same-sex partner without relatives -> True
        {"perrp": 40},  # reference person -> False
        {"perrp": 1},  # spouse -> False
        {"perrp": 45},  # adjacent code, deliberately excluded -> False
    ]
    result = _process_persons(_raw_person_frame(rows), 2023)
    assert "is_unmarried_partner_of_household_head" in result.columns
    assert result["is_unmarried_partner_of_household_head"].to_list() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    # The raw staging column must not leak into the processed frame.
    assert "_person_relationship_to_householder" not in result.columns


def test_unmarried_partner_flag_is_boolean_dtype():
    result = _process_persons(_raw_person_frame([{"perrp": 43}, {"perrp": 40}]), 2023)
    assert result.schema["is_unmarried_partner_of_household_head"] == pl.Boolean


# ---------------------------------------------------------------------------
# G6a: reported_owns_employer_sponsored_health_insurance_at_interview
# ---------------------------------------------------------------------------


def test_esi_policyholder_flag_recode():
    rows = [
        {"owngrp": 1},  # holds ESI in own name -> True
        {"owngrp": 0},  # does not -> False
        {"owngrp": 2},  # any non-1 code -> False
    ]
    result = _process_persons(_raw_person_frame(rows), 2023)
    col = "reported_owns_employer_sponsored_health_insurance_at_interview"
    assert col in result.columns
    assert result[col].to_list() == [True, False, False]
    assert result.schema[col] == pl.Boolean


# ---------------------------------------------------------------------------
# G6b: employer_sponsored_insurance_premiums (MEPS-prior imputation)
# ---------------------------------------------------------------------------


def test_esi_premium_matches_ecps_reference_fixture():
    """Reproduce the eCPS unit-test fixture exactly.

    Fixture and expectations are lifted from policyengine-us-data
    ``tests/unit/test_employer_sponsored_insurance_premiums.py``::

        NOW_OWNGRP  = [1, 1, 1, 0, 1]
        NOW_HIPAID  = [1, 2, 2, 1, 2]
        NOW_GRPFTYP = [2, 2, 1, 2, 1]
        PHIP_VAL    = [0, 1_200, 0, 0, 50_000]
    """
    rows = [
        {"owngrp": 1, "hipaid": 1, "grpftyp": 2, "phip": 0},
        {"owngrp": 1, "hipaid": 2, "grpftyp": 2, "phip": 1_200},
        {"owngrp": 1, "hipaid": 2, "grpftyp": 1, "phip": 0},
        {"owngrp": 0, "hipaid": 1, "grpftyp": 2, "phip": 0},
        {"owngrp": 1, "hipaid": 2, "grpftyp": 2, "phip": 50_000},
    ]
    result = _process_persons(_raw_person_frame(rows), 2023)
    premiums = result["employer_sponsored_insurance_premiums"].to_list()

    self_only_total = ESI_PLAN_PRIORS_2024["self_only"]["total_premium"]
    family_total = ESI_PLAN_PRIORS_2024["family"]["total_premium"]
    family_employee = ESI_PLAN_PRIORS_2024["family"]["employee_contribution"]
    expected = [
        self_only_total,  # employer pays all, self-only plan
        self_only_total - 1_200,  # employer pays some, self-only, $1.2k employee
        family_total - family_employee,  # employer pays some, family, avg employee
        0.0,  # not an own-name ESI holder
        0.0,  # employer pays some but employee paid exceeds total -> clip at 0
    ]
    assert len(premiums) == len(expected)
    for got, want in zip(premiums, expected):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-6), (got, want)


def test_esi_premium_is_zero_for_non_owners():
    # Even with a paid-premium status and a real plan type, a non-owner
    # (NOW_OWNGRP != 1) must get a zero employer premium.
    rows = [{"owngrp": 0, "hipaid": 1, "grpftyp": 1, "phip": 0}]
    result = _process_persons(_raw_person_frame(rows), 2023)
    assert result["employer_sponsored_insurance_premiums"].to_list() == [0.0]


def test_esi_premium_zero_when_no_employer_contribution():
    # NOW_HIPAID code other than 1/2 (e.g. 3 = employee pays all) -> no
    # employer-paid premium even for a valid owner with a plan.
    rows = [{"owngrp": 1, "hipaid": 3, "grpftyp": 1, "phip": 5_000}]
    result = _process_persons(_raw_person_frame(rows), 2023)
    assert result["employer_sponsored_insurance_premiums"].to_list() == [0.0]


def test_esi_staging_columns_are_dropped():
    result = _process_persons(_raw_person_frame([{"owngrp": 1}]), 2023)
    for staging in ("_now_owngrp", "_now_hipaid", "_now_grpftyp"):
        assert staging not in result.columns


def test_esi_premium_priors_match_ecps_meps_constants():
    # MEPS-IC Table IV.A.1 (private sector, 2024) constants, copied from eCPS.
    assert math.isclose(
        ESI_PLAN_PRIORS_2024["family"]["total_premium"], 21_207.52589669509
    )
    assert math.isclose(
        ESI_PLAN_PRIORS_2024["family"]["employee_contribution"], 6_490.205059544782
    )
    assert math.isclose(
        ESI_PLAN_PRIORS_2024["self_only"]["total_premium"], 8_389.275834815255
    )
    assert math.isclose(
        ESI_PLAN_PRIORS_2024["self_only"]["employee_contribution"],
        1_909.5781466113417,
    )


# ---------------------------------------------------------------------------
# Export-config wiring (all four leaves)
# ---------------------------------------------------------------------------


def test_recode_leaves_in_export_allowlist_and_not_aliased():
    from microplex_us.policyengine.us import (
        POLICYENGINE_US_EXPORT_COLUMN_ALIASES,
        SAFE_POLICYENGINE_US_EXPORT_VARIABLES,
    )

    for leaf in _RECODE_LEAVES:
        assert leaf in SAFE_POLICYENGINE_US_EXPORT_VARIABLES, leaf
        assert POLICYENGINE_US_EXPORT_COLUMN_ALIASES.get(leaf) is None, leaf


def test_esi_policyholder_has_legacy_person_entity():
    # The policyholder flag is not (yet) a released pe-us input variable, so the
    # export entity is pinned in the legacy-contract map (person), like its
    # reported_has_* siblings, to keep it on the eCPS-parity surface.
    from microplex_us.policyengine.us import (
        POLICYENGINE_US_LEGACY_CONTRACT_VARIABLE_ENTITIES,
    )

    assert (
        POLICYENGINE_US_LEGACY_CONTRACT_VARIABLE_ENTITIES.get(
            "reported_owns_employer_sponsored_health_insurance_at_interview"
        )
        == "person"
    )


def test_sstb_qbi_flag_exported_as_constant_false_default():
    # G9: eCPS's np.where(business_is_sstb, ..., False) collapses to False for
    # every record given MP's business_is_sstb=False default. Export it as the
    # same constant False, overriding the pe-us default_value=True.
    from microplex_us.policyengine.us import POLICYENGINE_US_EXPORT_DEFAULTS

    assert (
        POLICYENGINE_US_EXPORT_DEFAULTS["sstb_self_employment_income_would_be_qualified"]
        is False
    )
    # It must be internally consistent with MP's existing SSTB treatment.
    assert POLICYENGINE_US_EXPORT_DEFAULTS["business_is_sstb"] is False


def test_all_four_columns_are_required_by_the_ecps_contract():
    # Every column this change adds is a REQUIRED (not forbidden) column in the
    # frozen eCPS export-parity contract.
    import json
    from importlib import resources

    contract = json.loads(
        resources.files("microplex_us.pipelines")
        .joinpath("ecps_export_contract.json")
        .read_text()
    )
    required = set(contract["required"])
    forbidden = set(contract["forbidden"])
    for col in (
        "is_unmarried_partner_of_household_head",
        "reported_owns_employer_sponsored_health_insurance_at_interview",
        "employer_sponsored_insurance_premiums",
        "sstb_self_employment_income_would_be_qualified",
    ):
        assert col in required, col
        assert col not in forbidden, col


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
