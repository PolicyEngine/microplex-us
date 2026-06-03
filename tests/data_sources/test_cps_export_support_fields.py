"""Tests for CPS-backed eCPS export-support fields.

These fields are required by the eCPS export contract and are populated in the
incumbent enhanced CPS. Microplex previously exported many of them only through
constant defaults, so the presence gate passed while the support gate failed.
"""

import numpy as np
import polars as pl
import pytest

from microplex_us.data_sources.cps import (
    CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP,
    PERSON_VARIABLES,
    TAXABLE_PENSION_FRACTION,
    _attach_cps_ssn_card_type,
    _derive_cps_immigration_status,
    _process_persons,
)


def _raw_person_frame(rows: list[dict]) -> pl.DataFrame:
    n = len(rows)
    return pl.DataFrame(
        {
            "PH_SEQ": [1] * n,
            "A_LINENO": list(range(1, n + 1)),
            "A_FNLWGT": [100.0] * n,
            "A_AGE": [row.get("age", 40) for row in rows],
            "A_HSCOL": [row.get("school", 0) for row in rows],
            "A_HRLYWK": [row.get("hourly_code", 0) for row in rows],
            "A_HRSPAY": [row.get("hourly_cents", -1) for row in rows],
            "A_UNMEM": [row.get("union", 0) for row in rows],
            "POCCU2": [row.get("poccu2", 0) for row in rows],
            "PEIOOCC": [row.get("peioocc", -1) for row in rows],
            "PNSN_VAL": [row.get("pension", 0.0) for row in rows],
            "ANN_VAL": [row.get("annuity", 0.0) for row in rows],
            "LKWEEKS": [row.get("weeks_unemployed", -1) for row in rows],
            "VET_VAL": [row.get("veterans_benefits", 0.0) for row in rows],
            "WC_VAL": [row.get("workers_compensation", 0.0) for row in rows],
            "DST_SC1": [row.get("dst_sc1", 0) for row in rows],
            "DST_VAL1": [row.get("dst_val1", 0.0) for row in rows],
            "DST_SC2": [row.get("dst_sc2", 0) for row in rows],
            "DST_VAL2": [row.get("dst_val2", 0.0) for row in rows],
            "DST_SC1_YNG": [row.get("dst_sc1_yng", 0) for row in rows],
            "DST_VAL1_YNG": [row.get("dst_val1_yng", 0.0) for row in rows],
            "DST_SC2_YNG": [row.get("dst_sc2_yng", 0) for row in rows],
            "DST_VAL2_YNG": [row.get("dst_val2_yng", 0.0) for row in rows],
            "NOW_DIR": [row.get("now_dir", 2) for row in rows],
            "NOW_MRK": [row.get("now_mrk", 2) for row in rows],
            "NOW_MRKS": [row.get("now_mrks", 2) for row in rows],
            "NOW_MRKUN": [row.get("now_mrkun", 2) for row in rows],
            "NOW_NONM": [row.get("now_nonm", 2) for row in rows],
            "NOW_GRP": [row.get("now_grp", 2) for row in rows],
            "NOW_MCARE": [row.get("now_mcare", 2) for row in rows],
            "NOW_CAID": [row.get("now_caid", 2) for row in rows],
            "NOW_MCAID": [row.get("now_mcaid", 2) for row in rows],
            "NOW_PCHIP": [row.get("now_pchip", 2) for row in rows],
            "NOW_OTHMT": [row.get("now_othmt", 2) for row in rows],
            "NOW_MIL": [row.get("now_mil", 2) for row in rows],
            "NOW_CHAMPVA": [row.get("now_champva", 2) for row in rows],
            "NOW_VACARE": [row.get("now_vacare", 2) for row in rows],
            "NOW_IHSFLG": [row.get("now_ihs", 2) for row in rows],
            "NOW_PRIV": [row.get("now_priv", 2) for row in rows],
            "NOW_PUB": [row.get("now_pub", 2) for row in rows],
            "NOW_COV": [row.get("now_cov", 2) for row in rows],
        }
    )


def test_person_variables_maps_current_health_coverage_sources():
    for leaf, census_column in CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP.items():
        assert PERSON_VARIABLES.get(census_column) == f"_{leaf}"


def test_process_persons_populates_health_coverage_support_fields():
    out = _process_persons(
        _raw_person_frame(
            [
                {"now_mrk": 1, "now_grp": 1, "now_priv": 1, "now_cov": 1},
                {"now_caid": 1, "now_mcaid": 1, "now_pub": 1, "now_cov": 1},
                {"now_cov": 2},
            ]
        ),
        2025,
    )

    assert out["reported_has_marketplace_health_coverage_at_interview"].to_list() == [
        True,
        False,
        False,
    ]
    assert out[
        "reported_has_employer_sponsored_health_coverage_at_interview"
    ].to_list() == [
        True,
        False,
        False,
    ]
    assert out["reported_has_medicaid_health_coverage_at_interview"].to_list() == [
        False,
        True,
        False,
    ]
    assert out["reported_has_multiple_health_coverage_at_interview"].to_list() == [
        True,
        False,
        False,
    ]
    assert out["has_marketplace_health_coverage_at_interview"].to_list() == [
        True,
        False,
        False,
    ]
    assert out["has_marketplace_health_coverage"].to_list() == [True, False, False]
    assert out["has_esi"].to_list() == [True, False, False]
    assert out["reported_is_insured_at_interview"].to_list() == [True, True, False]
    assert out["reported_is_uninsured_at_interview"].to_list() == [False, False, True]


def test_process_persons_populates_labor_occupation_and_tipped_fields():
    out = _process_persons(
        _raw_person_frame(
            [
                {
                    "school": 2,
                    "hourly_code": 1,
                    "hourly_cents": 2150,
                    "union": 1,
                    "poccu2": 8,
                    "peioocc": 4000,
                    "weeks_unemployed": 12,
                    "veterans_benefits": 700.0,
                    "workers_compensation": 300.0,
                },
                {"hourly_code": 2, "hourly_cents": -1, "poccu2": 52, "peioocc": -1},
                {"poccu2": 53, "weeks_unemployed": -1},
            ]
        ),
        2025,
    )

    assert out["is_full_time_college_student"].to_list() == [True, False, False]
    assert out["is_paid_hourly"].to_list() == [True, False, False]
    assert out["hourly_wage"].to_list() == [21.5, 0.0, 0.0]
    assert out["is_union_member_or_covered"].to_list() == [True, False, False]
    assert out["detailed_occupation_recode"].to_list() == [8, 52, 53]
    assert out["is_computer_scientist"].to_list() == [True, False, False]
    assert out["is_military"].to_list() == [False, True, False]
    assert out["has_never_worked"].to_list() == [False, False, True]
    assert out["treasury_tipped_occupation_code"].to_list() == [105, 0, 0]
    assert out["is_tipped_occupation"].to_list() == [True, False, False]
    assert out["weeks_unemployed"].to_list() == [12, 0, 0]
    assert out["veterans_benefits"].to_list() == [700.0, 0.0, 0.0]
    assert out["workers_compensation"].to_list() == [300.0, 0.0, 0.0]


def test_process_persons_populates_pension_and_retirement_distribution_leaves():
    out = _process_persons(
        _raw_person_frame(
            [
                {
                    "pension": 10_000.0,
                    "annuity": 2_000.0,
                    "dst_sc1": 1,
                    "dst_val1": 1_500.0,
                    "dst_sc2": 4,
                    "dst_val2": 2_500.0,
                },
                {
                    "dst_sc1": 2,
                    "dst_val1": 600.0,
                    "dst_sc2": 3,
                    "dst_val2": 700.0,
                    "dst_sc1_yng": 6,
                    "dst_val1_yng": 800.0,
                    "dst_sc2_yng": 7,
                    "dst_val2_yng": 900.0,
                },
            ]
        ),
        2025,
    )

    total_pension = 12_000.0
    assert out["pension_income"].to_list() == [total_pension, 0.0]
    assert out["taxable_private_pension_income"].to_list() == pytest.approx(
        [total_pension * TAXABLE_PENSION_FRACTION, 0.0]
    )
    assert out["tax_exempt_private_pension_income"].to_list() == pytest.approx(
        [total_pension * (1 - TAXABLE_PENSION_FRACTION), 0.0]
    )
    assert out["taxable_401k_distributions"].to_list() == [1_500.0, 0.0]
    assert out["regular_ira_distributions"].to_list() == [2_500.0, 0.0]
    assert out["taxable_ira_distributions"].to_list() == [2_500.0, 0.0]
    assert out["taxable_403b_distributions"].to_list() == [0.0, 600.0]
    assert out["roth_ira_distributions"].to_list() == [0.0, 700.0]
    assert out["tax_exempt_ira_distributions"].to_list() == [0.0, 700.0]
    assert out["taxable_sep_distributions"].to_list() == [0.0, 800.0]
    assert out["other_type_retirement_account_distributions"].to_list() == [0.0, 900.0]


def test_derive_cps_immigration_status_varies_from_ssn_card_type():
    status = _derive_cps_immigration_status(
        ssn_card_type=np.array([1, 0, 2, 3]),
        birth_country=np.array([57, 57, 57, 332]),
        peinusyr=np.array([0, 29, 28, 20]),
        age=np.array([40, 30, 30, 40]),
        year=2024,
    )

    assert status.tolist() == [
        "CITIZEN",
        "UNDOCUMENTED",
        "TPS",
        "CUBAN_HAITIAN_ENTRANT",
    ]


def test_attach_cps_ssn_card_type_persists_identification_exports():
    persons = pl.DataFrame(
        {
            "household_id": [1, 2],
            "year": [2025, 2025],
            "age": [40, 30],
        }
    )
    households = pl.DataFrame({"household_id": [1, 2], "household_weight": [1.0, 1.0]})
    raw = pl.DataFrame(
        {
            "PRCITSHP": [1, 5],
            "PEINUSYR": [0, 29],
            "PENATVTY": [57, 57],
            "A_HSCOL": [0, 0],
            "A_AGE": [40, 30],
            "A_MARITL": [0, 0],
            "A_SPOUSE": [0, 0],
            "MCARE": [0, 0],
            "CAID": [0, 0],
            "PEN_SC1": [0, 0],
            "PEN_SC2": [0, 0],
            "RESNSS1": [0, 0],
            "RESNSS2": [0, 0],
            "IHSFLG": [0, 0],
            "CHAMPVA": [0, 0],
            "MIL": [0, 0],
            "PEIO1COW": [0, 0],
            "A_MJOCC": [0, 0],
            "SS_YN": [0, 0],
            "SPM_ID": [1, 2],
            "SPM_CAPHOUSESUB": [0.0, 0.0],
            "PEAFEVER": [0, 0],
            "SSI_YN": [0, 0],
            "WSAL_VAL": [0.0, 0.0],
            "SEMP_VAL": [0.0, 0.0],
        }
    )

    out = _attach_cps_ssn_card_type(
        persons=persons,
        households=households,
        persons_raw=raw,
    )

    assert out["ssn_card_type"].to_list() == ["CITIZEN", "NONE"]
    assert out["has_valid_ssn"].to_list() == [True, False]
    assert out["taxpayer_id_type"].to_list() == ["VALID_SSN", "NONE"]
    assert out["immigration_status_str"].to_list() == ["CITIZEN", "UNDOCUMENTED"]
