"""Tests for CPS source-provider implementations."""

import zipfile

import pandas as pd
import polars as pl
from microplex.core import EntityType, SourceArchetype, SourceProvider, SourceQuery

from microplex_us.data_sources import CPSASECParquetSourceProvider
from microplex_us.data_sources.cps import (
    CPS_ASEC_PROCESSED_CACHE_VERSION,
    PERSON_CACHE_REQUIRED_COLUMNS,
    CPSASECSourceProvider,
    _attach_cps_ssn_card_type,
    _cps_age_band_key,
    _sample_households_and_persons,
    get_available_years,
    load_cps_asec,
    processed_cps_asec_cache_path,
)


def test_cps_asec_available_years_include_latest_survey():
    assert max(get_available_years()) == 2025


def test_cps_parquet_source_provider_loads_observation_frame(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "state_fips": [6, 36],
            "household_weight": [1.0, 2.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "person_number": [1, 2, 1],
            "age": [34, 12, 52],
            "weight": [1.0, 1.0, 2.0],
        }
    )
    households.to_parquet(tmp_path / "cps_asec_households.parquet", index=False)
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

    provider = CPSASECParquetSourceProvider(data_dir=tmp_path, year=2024)
    frame = provider.load_frame(SourceQuery(period=2024))

    assert isinstance(provider, SourceProvider)
    assert set(frame.tables) == {EntityType.HOUSEHOLD, EntityType.PERSON}
    assert frame.tables[EntityType.PERSON]["person_id"].tolist() == [
        "1:1",
        "1:2",
        "2:1",
    ]
    assert frame.tables[EntityType.HOUSEHOLD]["year"].tolist() == [2024, 2024]
    assert frame.source.archetype is SourceArchetype.HOUSEHOLD_INCOME


def test_cps_parquet_source_provider_derives_canonical_income_alias(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1],
            "state_fips": [6],
            "household_weight": [1.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1, 1],
            "person_number": [1, 2],
            "age": [44, 41],
            "weight": [1.0, 1.0],
            "wage_income": [50_000.0, 10_000.0],
            "self_employment_income": [5_000.0, 0.0],
            "interest_income": [100.0, 20.0],
            "dividend_income": [50.0, 10.0],
            "social_security": [0.0, 3_000.0],
            "pension_income": [1_000.0, 4_000.0],
            "taxable_pension_income": [800.0, 3_000.0],
            "ssi": [1_000.0, 2_000.0],
        }
    )
    households.to_parquet(tmp_path / "cps_asec_households.parquet", index=False)
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

    provider = CPSASECParquetSourceProvider(data_dir=tmp_path, year=2024)
    frame = provider.load_frame(SourceQuery(period=2024))

    assert "income" in frame.source.observations[1].variable_names
    assert frame.tables[EntityType.PERSON]["income"].tolist() == [56_150.0, 17_030.0]


def test_cps_parquet_source_provider_derives_tax_unit_roles_from_tax_id(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1],
            "state_fips": [6],
            "household_weight": [1.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1, 1, 1, 1],
            "person_number": [1, 2, 3, 4],
            "spouse_person_number": [2, 1, 0, 0],
            "family_relationship": [1, 2, 3, 1],
            "tax_unit_id": [100, 100, 100, 101],
            "age": [40, 38, 10, 22],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    households.to_parquet(tmp_path / "cps_asec_households.parquet", index=False)
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

    provider = CPSASECParquetSourceProvider(data_dir=tmp_path, year=2024)
    frame = provider.load_frame(SourceQuery(period=2024))
    result = (
        frame.tables[EntityType.PERSON]
        .sort_values("person_number")
        .reset_index(drop=True)
    )

    assert result["tax_unit_id"].tolist() == [100, 100, 100, 101]
    assert result["tax_unit_is_joint"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert result["tax_unit_count_dependents"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert result["is_tax_unit_head"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert result["is_tax_unit_spouse"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert result["is_tax_unit_dependent"].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_attach_cps_ssn_card_type_derives_pe_style_categories():
    persons = pl.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "person_number": [1, 1, 1, 1],
            "age": [30, 40, 28, 35],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    households = pl.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "household_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    persons_raw = pl.DataFrame(
        {
            "PRCITSHP": [1, 5, 5, 5],
            "PEINUSYR": [0, 20, 20, 20],
            "PENATVTY": [57, 303, 303, 303],
            "A_HSCOL": [0, 0, 0, 0],
            "A_AGE": [30, 40, 28, 35],
            "A_MARITL": [0, 0, 0, 0],
            "A_SPOUSE": [0, 0, 0, 0],
            "MCARE": [0, 1, 0, 0],
            "CAID": [0, 0, 0, 0],
            "PEN_SC1": [0, 0, 0, 0],
            "PEN_SC2": [0, 0, 0, 0],
            "RESNSS1": [0, 0, 0, 0],
            "RESNSS2": [0, 0, 0, 0],
            "IHSFLG": [0, 0, 0, 0],
            "CHAMPVA": [0, 0, 0, 0],
            "MIL": [0, 0, 0, 0],
            "PEIO1COW": [0, 0, 0, 0],
            "A_MJOCC": [0, 0, 0, 0],
            "SS_YN": [0, 0, 0, 0],
            "SPM_ID": [11, 22, 33, 44],
            "SPM_CAPHOUSESUB": [0.0, 0.0, 0.0, 0.0],
            "PEAFEVER": [0, 0, 0, 0],
            "SSI_YN": [0, 0, 0, 0],
            "WSAL_VAL": [0.0, 0.0, 20_000.0, 0.0],
            "SEMP_VAL": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = _attach_cps_ssn_card_type(
        persons=persons,
        households=households,
        persons_raw=persons_raw,
    )

    assert result["ssn_card_type"].to_list() == [
        "CITIZEN",
        "OTHER_NON_CITIZEN",
        "NON_CITIZEN_VALID_EAD",
        "NONE",
    ]


def test_attach_cps_ssn_card_type_falls_back_to_citizen_when_raw_fields_missing():
    persons = pl.DataFrame(
        {
            "household_id": [1, 2],
            "person_number": [1, 1],
            "age": [30, 40],
            "weight": [1.0, 1.0],
        }
    )
    households = pl.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [1.0, 1.0],
        }
    )
    persons_raw = pl.DataFrame(
        {
            "PRCITSHP": [1, 5],
            "PEINUSYR": [0, 20],
        }
    )

    result = _attach_cps_ssn_card_type(
        persons=persons,
        households=households,
        persons_raw=persons_raw,
    )

    assert result["ssn_card_type"].to_list() == ["CITIZEN", "CITIZEN"]


def test_cps_parquet_source_provider_supports_household_sampling(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "state_fips": [6, 36, 48],
            "household_weight": [1.0, 2.0, 3.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1, 1, 2, 3],
            "person_number": [1, 2, 1, 1],
            "age": [34, 12, 52, 40],
            "weight": [1.0, 1.0, 2.0, 3.0],
        }
    )
    households.to_parquet(tmp_path / "cps_asec_households.parquet", index=False)
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

    provider = CPSASECParquetSourceProvider(data_dir=tmp_path, year=2024)
    frame = provider.load_frame(
        SourceQuery(
            period=2024,
            provider_filters={"sample_n": 2, "random_seed": 0},
        )
    )

    assert len(frame.tables[EntityType.HOUSEHOLD]) == 2
    assert frame.tables[EntityType.PERSON]["household_id"].nunique() == 2


def test_cps_parquet_source_provider_sampling_respects_household_weights(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "state_fips": [6, 36, 48],
            "household_weight": [0.0, 0.0, 100.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "person_number": [1, 1, 1],
            "age": [34, 52, 40],
            "weight": [0.0, 0.0, 100.0],
        }
    )
    households.to_parquet(tmp_path / "cps_asec_households.parquet", index=False)
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

    provider = CPSASECParquetSourceProvider(data_dir=tmp_path, year=2024)
    frame = provider.load_frame(
        SourceQuery(
            period=2024,
            provider_filters={"sample_n": 1, "random_seed": 0},
        )
    )

    assert frame.tables[EntityType.HOUSEHOLD]["household_id"].tolist() == [3]
    assert frame.tables[EntityType.PERSON]["household_id"].tolist() == [3]


def test_cps_parquet_source_provider_applies_generic_atomic_variable_semantics(
    tmp_path,
):
    households = pd.DataFrame(
        {
            "household_id": [1],
            "state_fips": [6],
            "household_weight": [1.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1],
            "person_number": [1],
            "age": [34],
            "weight": [1.0],
            "qualified_dividend_income": [30.0],
            "non_qualified_dividend_income": [12.0],
            "dividend_income": [42.0],
            "ordinary_dividend_income": [42.0],
        }
    )
    households.to_parquet(tmp_path / "cps_asec_households.parquet", index=False)
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

    provider = CPSASECParquetSourceProvider(data_dir=tmp_path, year=2024)
    frame = provider.load_frame(SourceQuery(period=2024))
    descriptor = frame.source

    assert not descriptor.is_authoritative_for("dividend_income")
    assert not descriptor.allows_conditioning_on("dividend_income")
    assert not descriptor.is_authoritative_for("ordinary_dividend_income")
    assert descriptor.is_authoritative_for("qualified_dividend_income")
    assert descriptor.allows_conditioning_on("qualified_dividend_income")


def test_load_cps_asec_rebuilds_stale_processed_cache_without_state_fips(tmp_path):
    stale_processed = pl.DataFrame(
        {
            "household_id": [1, 1, 2],
            "person_number": [1, 2, 1],
            "age": [34, 12, 52],
            "weight": [1.0, 1.0, 2.0],
            "year": [2023, 2023, 2023],
        }
    )
    stale_processed.write_parquet(tmp_path / "cps_asec_2023_processed.parquet")

    person_rows = pd.DataFrame(
        {
            "PH_SEQ": [1, 1, 2],
            "GESTFIPS": [6, 6, 36],
            "A_LINENO": [1, 2, 1],
            "A_AGE": [34, 12, 52],
            "A_FNLWGT": [100, 100, 200],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))

    dataset = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)

    assert "state_fips" in dataset.persons.columns
    assert sorted(dataset.households["state_fips"].to_list()) == [6, 36]


def test_load_cps_asec_caches_household_geography_on_persons(tmp_path):
    person_rows = pd.DataFrame(
        {
            "PH_SEQ": [1, 1, 2],
            "A_LINENO": [1, 2, 1],
            "A_SPOUSE": [2, 1, 0],
            "A_AGE": [34, 12, 52],
            "A_FNLWGT": [100, 100, 200],
            "TAX_ID": [100, 100, 200],
            "SPM_ID": [10, 10, 20],
            "A_MARITL": [1, 1, 6],
            "PRDTRACE": [4, 4, 1],
            "PRDTHSP": [0, 1, 0],
            "PEHSPNON": [2, 1, 2],
            "PEDISDRS": [0, 1, 0],
            "PEDISEAR": [0, 0, 0],
            "PEDISEYE": [0, 0, 0],
            "PEDISOUT": [0, 0, 0],
            "PEDISPHY": [0, 0, 0],
            "PEDISREM": [0, 0, 0],
            "RESNSS1": [0, 2, 0],
            "RESNSS2": [0, 0, 0],
            "SS_VAL": [0, 9000, 0],
            "WICYN": [1, 2, 0],
            "NOW_MRK": [1, 0, 0],
            "NOW_GRP": [0, 1, 0],
        }
    )
    household_rows = pd.DataFrame(
        {
            "H_SEQ": [1, 2],
            "GESTFIPS": [6, 36],
            "GTCO": [1, 61],
            "HSUP_WGT": [100, 200],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))
        archive.writestr("hhpub23.csv", household_rows.to_csv(index=False))

    first = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)
    cached_persons = pl.read_parquet(
        processed_cps_asec_cache_path(year=2023, cache_dir=tmp_path)
    )
    second = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)

    assert "state_fips" in first.persons.columns
    assert "county_fips" in first.persons.columns
    assert "cps_race" in first.persons.columns
    assert "is_hispanic" in first.persons.columns
    assert "is_disabled" in first.persons.columns
    assert "social_security_disability" in first.persons.columns
    assert "social_security_retirement" in first.persons.columns
    assert "social_security_survivors" in first.persons.columns
    assert "social_security_dependents" in first.persons.columns
    assert "receives_wic" in first.persons.columns
    assert "spm_unit_pre_subsidy_childcare_expenses" in first.persons.columns
    assert "has_marketplace_health_coverage" in first.persons.columns
    assert "has_esi" in first.persons.columns
    assert "tax_unit_id" in first.persons.columns
    assert "spm_unit_id" in first.persons.columns
    assert "spouse_person_number" in first.persons.columns
    assert "marital_unit_id" in first.persons.columns
    assert "is_surviving_spouse" in first.persons.columns
    assert "is_separated" in first.persons.columns
    assert cached_persons["state_fips"].to_list() == [6, 6, 36]
    assert cached_persons["county_fips"].to_list() == [1, 1, 61]
    assert cached_persons["cps_race"].to_list() == [4, 4, 1]
    assert cached_persons["is_hispanic"].to_list() == [False, True, False]
    assert cached_persons["is_disabled"].to_list() == [False, True, False]
    assert cached_persons["social_security_disability"].to_list() == [0.0, 9000.0, 0.0]
    assert cached_persons["social_security_retirement"].to_list() == [0.0, 0.0, 0.0]
    assert cached_persons["social_security_survivors"].to_list() == [0.0, 0.0, 0.0]
    assert cached_persons["social_security_dependents"].to_list() == [0.0, 0.0, 0.0]
    assert cached_persons["receives_wic"].to_list() == [True, False, False]
    assert cached_persons["spm_unit_pre_subsidy_childcare_expenses"].to_list() == [
        0.0,
        0.0,
        0.0,
    ]
    assert cached_persons["has_marketplace_health_coverage"].to_list() == [
        True,
        False,
        False,
    ]
    assert cached_persons["has_esi"].to_list() == [False, True, False]
    assert cached_persons["tax_unit_id"].to_list() == [100, 100, 200]
    assert cached_persons["spm_unit_id"].to_list() == [10, 10, 20]
    assert cached_persons["spouse_person_number"].to_list() == [2, 1, 0]
    assert cached_persons["is_surviving_spouse"].to_list() == [False, False, False]
    assert cached_persons["is_separated"].to_list() == [False, False, True]
    assert cached_persons["marital_unit_id"].to_list() == [1, 1, 2]
    assert second.source.endswith(
        f"cps_asec_2023_processed_v{CPS_ASEC_PROCESSED_CACHE_VERSION}.parquet"
    )
    assert sorted(second.households["state_fips"].to_list()) == [6, 36]
    assert sorted(second.households["county_fips"].to_list()) == [1, 61]


def test_load_cps_asec_derives_policyengine_value_inputs(tmp_path):
    person_rows = pd.DataFrame(
        {
            "PH_SEQ": [1, 1],
            "A_LINENO": [1, 2],
            "A_AGE": [34, 62],
            "A_FNLWGT": [100, 100],
            "OI_OFF": [20, 12],
            "OI_VAL": [1200, 800],
            "CSP_VAL": [300, -1],
            "CHSP_VAL": [700, -1],
            "DIS_VAL1": [500, 400],
            "DIS_SC1": [2, 1],
            "DIS_VAL2": [50, 25],
            "DIS_SC2": [3, 2],
            "RESNSS1": [2, 1],
            "RESNSS2": [0, 0],
            "SS_VAL": [1200, 800],
            "MCARE": [1, 2],
            "MCAID": [2, 1],
            "WICYN": [1, 2],
            "SPM_CAPHOUSESUB": [700, 0],
            "SPM_ENGVAL": [90, -1],
            "SPM_CAPWKCCXPNS": [1200, -1],
            "SPM_CHILDCAREXPNS": [1500, -1],
            "PHIP_VAL": [900, -1],
            "POTC_VAL": [120, -1],
            "PMED_VAL": [450, -1],
            "PEMCPREM": [600, -1],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))

    dataset = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)
    persons = (
        dataset.persons.to_pandas()
        .sort_values(["household_id", "person_number"])
        .reset_index(drop=True)
    )

    assert persons["alimony_income"].tolist() == [1200, 0]
    assert persons["child_support_received"].tolist() == [300, 0]
    assert persons["child_support_expense"].tolist() == [700, 0]
    assert persons["disability_benefits"].tolist() == [550, 25]
    assert persons["social_security_disability"].tolist() == [1200, 0]
    assert persons["social_security_retirement"].tolist() == [0, 800]
    assert persons["social_security_survivors"].tolist() == [0, 0]
    assert persons["social_security_dependents"].tolist() == [0, 0]
    assert persons["has_medicare"].tolist() == [True, False]
    assert persons["takes_up_medicare_if_eligible"].tolist() == [True, False]
    assert persons["has_medicaid"].tolist() == [False, True]
    assert persons["receives_wic"].tolist() == [True, False]
    assert persons["receives_housing_assistance"].tolist() == [True, False]
    assert persons["takes_up_housing_assistance_if_eligible"].tolist() == [True, False]
    assert persons["spm_unit_energy_subsidy"].tolist() == [90, 0]
    assert persons["spm_unit_capped_housing_subsidy_reported"].tolist() == [700, 0]
    assert persons["spm_unit_capped_work_childcare_expenses"].tolist() == [1200, 0]
    assert persons["spm_unit_pre_subsidy_childcare_expenses"].tolist() == [1500, 0]
    assert persons["health_insurance_premiums_without_medicare_part_b"].tolist() == [
        900,
        0,
    ]
    assert persons["over_the_counter_health_expenses"].tolist() == [120, 0]
    assert persons["other_medical_expenses"].tolist() == [450, 0]
    assert persons["medicare_part_b_premiums"].tolist() == [600, 0]


def test_load_cps_asec_falls_back_last_year_income_to_current_earnings(tmp_path):
    # The prior-year-earnings lookback (EITC/CTC prior-year election) expired,
    # so last-year income is a placeholder set to current-year earnings
    # (WSAL_VAL / SEMP_VAL) with no prior-ASEC dependency.
    # previous_year_income_available tracks whether the row has any earnings.
    current_person_rows = pd.DataFrame(
        {
            "PERIDNUM": ["A", "B", "C", "D"],
            "PH_SEQ": [1, 1, 2, 2],
            "A_LINENO": [1, 2, 1, 2],
            "A_AGE": [34, 31, 45, 17],
            "A_FNLWGT": [100, 100, 200, 200],
            "WSAL_VAL": [60_000, 10_000, 20_000, 0],
            "SEMP_VAL": [5_000, 0, -3_000, 0],
            "I_ERNVAL": [0, 1, 0, 0],
            "I_SEVAL": [0, 0, 0, 0],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", current_person_rows.to_csv(index=False))

    dataset = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)
    persons = (
        dataset.persons.to_pandas()
        .sort_values(["household_id", "person_number"])
        .reset_index(drop=True)
    )

    assert persons["employment_income_last_year"].tolist() == [
        60_000.0,
        10_000.0,
        20_000.0,
        0.0,
    ]
    assert persons["self_employment_income_last_year"].tolist() == [
        5_000.0,
        0.0,
        -3_000.0,
        0.0,
    ]
    assert persons["self_employment_income"].tolist() == [
        5_000.0,
        0.0,
        -3_000.0,
        0.0,
    ]
    assert persons["previous_year_income_available"].tolist() == [
        True,
        True,
        True,
        False,
    ]


def test_load_cps_asec_derives_survivor_and_dependent_social_security(tmp_path):
    person_rows = pd.DataFrame(
        {
            "PH_SEQ": [1, 1, 1, 1],
            "A_LINENO": [1, 2, 3, 4],
            "A_AGE": [70, 40, 12, 10],
            "A_FNLWGT": [100, 100, 100, 100],
            "RESNSS1": [3, 5, 4, 6],
            "RESNSS2": [0, 0, 0, 0],
            "SS_VAL": [1000, 1100, 1200, 1300],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))

    dataset = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)
    persons = (
        dataset.persons.to_pandas().sort_values("person_number").reset_index(drop=True)
    )

    assert persons["social_security_survivors"].tolist() == [1000.0, 1100.0, 0.0, 0.0]
    assert persons["social_security_dependents"].tolist() == [0.0, 0.0, 1200.0, 1300.0]
    assert persons["social_security_retirement"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert persons["social_security_disability"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_cps_source_provider_repeat_loads_are_deterministic_for_cached_processed_data(
    tmp_path,
):
    cached_persons = pl.DataFrame(
        {
            "household_id": [2, 1, 2, 3, 1],
            "person_number": [1, 2, 2, 1, 1],
            "person_id": ["2:1", "1:2", "2:2", "3:1", "1:1"],
            "age": [52, 12, 49, 40, 34],
            "weight": [200.0, 100.0, 200.0, 300.0, 100.0],
            "state_fips": [36, 6, 36, 48, 6],
            "county_fips": [61, 1, 61, 201, 1],
            "cps_race": [1, 4, 1, 2, 4],
            "is_hispanic": [False, True, False, False, True],
            "is_disabled": [False, False, False, True, False],
            "social_security_disability": [0.0] * 5,
            "social_security_retirement": [0.0] * 5,
            "social_security_survivors": [0.0] * 5,
            "social_security_dependents": [0.0] * 5,
            "has_esi": [True, False, True, False, False],
            "has_marketplace_health_coverage": [False, True, False, False, True],
            "receives_wic": [False] * 5,
            "alimony_income": [0.0, 0.0, 0.0, 0.0, 0.0],
            "child_support_received": [0.0, 0.0, 0.0, 0.0, 0.0],
            "child_support_expense": [0.0, 0.0, 0.0, 0.0, 0.0],
            "disability_benefits": [0.0, 0.0, 0.0, 0.0, 0.0],
            "health_insurance_premiums_without_medicare_part_b": [0.0] * 5,
            "other_medical_expenses": [0.0] * 5,
            "over_the_counter_health_expenses": [0.0] * 5,
            "medicare_part_b_premiums": [0.0] * 5,
            "spm_unit_pre_subsidy_childcare_expenses": [0.0] * 5,
            "year": [2023, 2023, 2023, 2023, 2023],
        }
    )
    for column in PERSON_CACHE_REQUIRED_COLUMNS:
        if column not in cached_persons.columns:
            cached_persons = cached_persons.with_columns(pl.lit(0).alias(column))
    cached_persons.write_parquet(
        processed_cps_asec_cache_path(year=2023, cache_dir=tmp_path)
    )

    provider = CPSASECSourceProvider(year=2023, cache_dir=tmp_path, download=False)
    query = SourceQuery(provider_filters={"sample_n": 2, "random_seed": 42})

    first = provider.load_frame(query)
    second = provider.load_frame(query)

    first_households = first.tables[EntityType.HOUSEHOLD]
    second_households = second.tables[EntityType.HOUSEHOLD]
    first_persons = first.tables[EntityType.PERSON]
    second_persons = second.tables[EntityType.PERSON]

    assert (
        first_households["household_id"].tolist()
        == second_households["household_id"].tolist()
    )
    assert first_persons["person_id"].tolist() == second_persons["person_id"].tolist()
    assert (
        first_households["household_weight"].tolist()
        == second_households["household_weight"].tolist()
    )
    assert first_persons["weight"].tolist() == second_persons["weight"].tolist()


def test_load_cps_asec_rebuilds_stale_processed_cache_without_pe_presim_inputs(
    tmp_path,
):
    stale_processed = pl.DataFrame(
        {
            "household_id": [1, 1, 2],
            "person_number": [1, 2, 1],
            "age": [34, 12, 52],
            "weight": [1.0, 1.0, 2.0],
            "state_fips": [6, 6, 36],
            "year": [2023, 2023, 2023],
        }
    )
    stale_processed.write_parquet(tmp_path / "cps_asec_2023_processed.parquet")

    person_rows = pd.DataFrame(
        {
            "PH_SEQ": [1, 1, 2],
            "GESTFIPS": [6, 6, 36],
            "A_LINENO": [1, 2, 1],
            "A_AGE": [34, 12, 52],
            "A_FNLWGT": [100, 100, 200],
            "PRDTRACE": [4, 4, 1],
            "PRDTHSP": [0, 1, 0],
            "PEHSPNON": [2, 1, 2],
            "PEDISDRS": [0, 1, 0],
            "PEDISEAR": [0, 0, 0],
            "PEDISEYE": [0, 0, 0],
            "PEDISOUT": [0, 0, 0],
            "PEDISPHY": [0, 0, 0],
            "PEDISREM": [0, 0, 0],
            "NOW_MRK": [1, 0, 0],
            "NOW_GRP": [0, 1, 0],
            "OI_OFF": [20, 0, 0],
            "OI_VAL": [1200, 0, 0],
            "CSP_VAL": [300, 0, 0],
            "CHSP_VAL": [700, 0, 0],
            "DIS_VAL1": [500, 0, 0],
            "DIS_SC1": [2, 0, 0],
            "DIS_VAL2": [50, 0, 0],
            "DIS_SC2": [3, 0, 0],
            "PHIP_VAL": [900, 0, 0],
            "POTC_VAL": [120, 0, 0],
            "PMED_VAL": [450, 0, 0],
            "PEMCPREM": [600, 0, 0],
        }
    )
    household_rows = pd.DataFrame(
        {
            "H_SEQ": [1, 2],
            "GESTFIPS": [6, 36],
            "GTCO": [1, 61],
            "HSUP_WGT": [100, 200],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))
        archive.writestr("hhpub23.csv", household_rows.to_csv(index=False))

    dataset = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)

    assert dataset.source.endswith("cps_asec_2023.zip")
    assert dataset.persons["county_fips"].to_list() == [1, 1, 61]
    assert dataset.persons["cps_race"].to_list() == [4, 4, 1]
    assert dataset.persons["is_hispanic"].to_list() == [False, True, False]
    assert dataset.persons["is_disabled"].to_list() == [False, True, False]
    assert dataset.persons["has_marketplace_health_coverage"].to_list() == [
        True,
        False,
        False,
    ]
    assert dataset.persons["has_esi"].to_list() == [False, True, False]
    assert dataset.persons["alimony_income"].to_list() == [1200, 0, 0]
    assert dataset.persons["child_support_received"].to_list() == [300, 0, 0]
    assert dataset.persons["child_support_expense"].to_list() == [700, 0, 0]
    assert dataset.persons["disability_benefits"].to_list() == [550, 0, 0]
    assert dataset.persons[
        "health_insurance_premiums_without_medicare_part_b"
    ].to_list() == [900, 0, 0]
    assert dataset.persons["other_medical_expenses"].to_list() == [450, 0, 0]
    assert dataset.persons["over_the_counter_health_expenses"].to_list() == [120, 0, 0]
    assert dataset.persons["medicare_part_b_premiums"].to_list() == [600, 0, 0]


def test_cps_sampling_falls_back_to_uniform_when_weighted_sampling_is_infeasible(
    monkeypatch,
):
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "year": [2023, 2023, 2023],
            "household_weight": [10.0, 20.0, 30.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "person_id": ["1:1", "2:1", "3:1"],
            "person_number": [1, 1, 1],
            "year": [2023, 2023, 2023],
        }
    )

    original_sample = pd.DataFrame.sample

    def flaky_sample(self, *args, **kwargs):
        if kwargs.get("weights") is not None:
            raise ValueError("Weighted sampling cannot be achieved with replace=False.")
        return original_sample(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sample", flaky_sample)

    sampled_households, sampled_persons = _sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=2,
        random_seed=42,
    )

    assert len(sampled_households) == 2
    assert len(sampled_persons) == 2
    assert set(sampled_persons["household_id"]) == set(
        sampled_households["household_id"]
    )


def test_sample_households_and_persons_state_floor_preserves_state_coverage() -> None:
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5, 6],
            "state_fips": [6, 6, 36, 36, 48, 48],
            "household_weight": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            "year": [2024] * 6,
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [11, 21, 31, 41, 51, 61],
            "household_id": [1, 2, 3, 4, 5, 6],
            "person_number": [1, 1, 1, 1, 1, 1],
            "year": [2024] * 6,
        }
    )

    sampled_households, sampled_persons = _sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=3,
        random_seed=7,
        state_floor=1,
    )

    assert len(sampled_households) == 3
    assert sampled_households["state_fips"].nunique() == 3
    assert set(sampled_persons["household_id"]) == set(
        sampled_households["household_id"]
    )


def test_sample_households_and_persons_state_age_floor_preserves_age_band_coverage() -> (
    None
):
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5, 6],
            "state_fips": [6, 6, 6, 36, 36, 36],
            "household_weight": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            "year": [2024] * 6,
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [11, 21, 31, 41, 51, 61],
            "household_id": [1, 2, 3, 4, 5, 6],
            "person_number": [1, 1, 1, 1, 1, 1],
            "age": [2, 7, 7, 4, 87, 87],
            "year": [2024] * 6,
        }
    )

    sampled_households, sampled_persons = _sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=4,
        random_seed=7,
        state_age_floor=1,
    )

    observed_keys = {
        (int(state), _cps_age_band_key(age))
        for state, age in persons.merge(
            households[["household_id", "state_fips"]],
            on="household_id",
            how="left",
        )[["state_fips", "age"]].itertuples(index=False, name=None)
    }
    sampled_keys = {
        (int(state), _cps_age_band_key(age))
        for state, age in sampled_persons.merge(
            sampled_households[["household_id", "state_fips"]],
            on="household_id",
            how="left",
        )[["state_fips", "age"]].itertuples(index=False, name=None)
    }

    assert len(sampled_households) == 4
    assert observed_keys.issubset(sampled_keys)
    assert set(sampled_persons["household_id"]) == set(
        sampled_households["household_id"]
    )
