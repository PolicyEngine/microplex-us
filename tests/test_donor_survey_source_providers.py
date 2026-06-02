"""Tests for PE-style donor survey source providers."""

from __future__ import annotations

import h5py
import pandas as pd
import pytest
from microplex.core import EntityType

import microplex_us.data_sources.donor_surveys as donor_surveys
from microplex_us.data_sources.donor_surveys import (
    ACSSourceProvider,
    DonorSurveyTables,
    SCFSourceProvider,
    SIPPSourceProvider,
)


def _acs_tables(**_kwargs) -> DonorSurveyTables:
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [100.0, 120.0],
            "state_fips": [6, 36],
            "tenure": [1, 2],
            "year": [2022, 2022],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [11, 12, 21],
            "household_id": [1, 1, 2],
            "age": [45, 12, 68],
            "sex": [1, 2, 2],
            "is_male": [1.0, 0.0, 0.0],
            "is_household_head": [1.0, 0.0, 1.0],
            "tenure_type": [1, 1, 2],
            "employment_income": [50_000.0, 0.0, 12_000.0],
            "self_employment_income": [5_000.0, 0.0, 0.0],
            "social_security": [0.0, 0.0, 20_000.0],
            "taxable_pension_income": [0.0, 0.0, 15_000.0],
            "rent": [1_200.0, 0.0, 950.0],
            "real_estate_taxes": [3_000.0, 0.0, 0.0],
            "income": [55_000.0, 0.0, 47_000.0],
            "weight": [100.0, 100.0, 120.0],
            "year": [2022, 2022, 2022],
        }
    )
    return DonorSurveyTables(households=households, persons=persons)


def _sipp_tips_tables(**_kwargs) -> DonorSurveyTables:
    households = pd.DataFrame(
        {
            "household_id": ["100:1", "101:1"],
            "household_weight": [80.0, 90.0],
            "state_fips": [0, 0],
            "tenure": [0, 0],
            "year": [2023, 2023],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": ["100:1:1", "100:1:2", "101:1:1"],
            "household_id": ["100:1", "100:1", "101:1"],
            "age": [35, 8, 50],
            "sex": [1, 2, 2],
            "employment_income": [40_000.0, 0.0, 25_000.0],
            "income": [40_000.0, 0.0, 25_000.0],
            "tip_income": [900.0, 0.0, 250.0],
            "count_under_18": [1.0, 1.0, 0.0],
            "count_under_6": [0.0, 0.0, 0.0],
            "weight": [80.0, 80.0, 90.0],
            "year": [2023, 2023, 2023],
        }
    )
    return DonorSurveyTables(households=households, persons=persons)


def _sipp_assets_tables(**_kwargs) -> DonorSurveyTables:
    households = pd.DataFrame(
        {
            "household_id": ["100", "101"],
            "household_weight": [80.0, 90.0],
            "state_fips": [0, 0],
            "tenure": [0, 0],
            "year": [2023, 2023],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": ["100:1", "101:1"],
            "household_id": ["100", "101"],
            "age": [35, 50],
            "sex": [1, 2],
            "is_female": [0.0, 1.0],
            "is_married": [1.0, 0.0],
            "employment_income": [40_000.0, 25_000.0],
            "income": [40_000.0, 25_000.0],
            "count_under_18": [1.0, 0.0],
            "bank_account_assets": [2_500.0, 10_000.0],
            "stock_assets": [0.0, 4_000.0],
            "bond_assets": [0.0, 1_500.0],
            "weight": [80.0, 90.0],
            "year": [2023, 2023],
        }
    )
    return DonorSurveyTables(households=households, persons=persons)


def test_sample_households_and_persons_prefers_positive_weight_households() -> None:
    households = pd.DataFrame(
        {
            "household_id": ["h1", "h2", "h3"],
            "household_weight": [10.0, 0.0, 20.0],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": ["p1", "p2", "p3"],
            "household_id": ["h1", "h2", "h3"],
        }
    )

    sampled_households, sampled_persons = donor_surveys._sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=2,
        random_seed=0,
    )

    assert sampled_households["household_id"].tolist() == ["h1", "h3"]
    assert sampled_persons["household_id"].tolist() == ["h1", "h3"]


def test_sample_households_and_persons_falls_back_when_weighted_sampling_errors(
    monkeypatch,
) -> None:
    households = pd.DataFrame(
        {
            "household_id": ["h1", "h2", "h3"],
            "household_weight": [10.0, 5.0, 20.0],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": ["p1", "p2", "p3"],
            "household_id": ["h1", "h2", "h3"],
        }
    )
    original_sample = pd.DataFrame.sample

    def _flaky_sample(self, *args, **kwargs):
        if kwargs.get("weights") is not None:
            raise ValueError("weighted sampling failed")
        return original_sample(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sample", _flaky_sample)

    sampled_households, sampled_persons = donor_surveys._sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=2,
        random_seed=0,
    )

    assert len(sampled_households) == 2
    assert set(sampled_persons["household_id"]) == set(
        sampled_households["household_id"]
    )


def test_sample_households_and_persons_state_floor_preserves_states() -> None:
    households = pd.DataFrame(
        {
            "household_id": ["h1", "h2", "h3", "h4"],
            "household_weight": [10.0, 9.0, 8.0, 7.0],
            "state_fips": [6, 6, 36, 48],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": ["p1", "p2", "p3", "p4"],
            "household_id": ["h1", "h2", "h3", "h4"],
        }
    )

    sampled_households, sampled_persons = donor_surveys._sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=3,
        random_seed=0,
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
            "household_id": ["h1", "h2", "h3", "h4"],
            "household_weight": [10.0, 9.0, 8.0, 7.0],
            "state_fips": [6, 6, 36, 36],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": ["p1", "p2", "p3", "p4"],
            "household_id": ["h1", "h2", "h3", "h4"],
            "age": [7, 42, 10, 67],
        }
    )

    sampled_households, sampled_persons = donor_surveys._sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=4,
        random_seed=0,
        state_age_floor=1,
    )

    sampled = sampled_persons.merge(
        sampled_households[["household_id", "state_fips"]],
        on="household_id",
        how="left",
    )
    sampled["age_band"] = sampled["age"].map(donor_surveys._donor_age_band_key)

    assert len(sampled_households) == 4
    assert {
        (6, "5_10"),
        (6, "40_45"),
        (36, "10_15"),
        (36, "65_70"),
    }.issubset(set(zip(sampled["state_fips"], sampled["age_band"], strict=False)))


def _scf_tables(**_kwargs) -> DonorSurveyTables:
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 12.0],
            "state_fips": [0, 0],
            "tenure": [0, 0],
            "year": [2022, 2022],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [1, 2],
            "household_id": [1, 2],
            "age": [45, 68],
            "sex": [1, 2],
            "is_female": [0.0, 1.0],
            "cps_race": [1, 2],
            "is_married": [1.0, 0.0],
            "own_children_in_household": [1.0, 0.0],
            "employment_income": [75_000.0, 0.0],
            "income": [75_000.0, 0.0],
            "interest_dividend_income": [1_200.0, 400.0],
            "social_security_pension_income": [0.0, 18_000.0],
            "net_worth": [350_000.0, 180_000.0],
            "auto_loan_balance": [8_000.0, 0.0],
            "auto_loan_interest": [550.0, 0.0],
            "weight": [10.0, 12.0],
            "year": [2022, 2022],
        }
    )
    return DonorSurveyTables(households=households, persons=persons)


def _write_uprating_factors(
    repo_root,
    rows: dict[str, tuple[float, float, float]],
) -> None:
    storage_dir = repo_root / "policyengine_us_data" / "storage"
    storage_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Variable": variable,
                "2022": values[0],
                "2023": values[1],
                "2024": values[2],
            }
            for variable, values in rows.items()
        ]
    ).to_csv(storage_dir / "uprating_factors.csv", index=False)


def test_acs_source_provider_builds_observation_frame_from_injected_loader() -> None:
    provider = ACSSourceProvider(loader=_acs_tables)

    frame = provider.load_frame()

    assert frame.source.name == "acs_2022"
    assert frame.source.observes("rent", EntityType.PERSON)
    assert frame.source.allows_conditioning_on("state_fips") is True
    assert list(frame.tables[EntityType.HOUSEHOLD]["household_id"]) == [1, 2]


def test_acs_source_provider_uses_manifest_backed_dataset_loader(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_loader(
        *, spec, year, sample_n, random_seed, **_kwargs
    ) -> DonorSurveyTables:
        captured["spec"] = spec
        captured["year"] = year
        captured["sample_n"] = sample_n
        captured["random_seed"] = random_seed
        return _acs_tables()

    monkeypatch.setattr(
        donor_surveys,
        "_run_policyengine_dataset_loader_from_spec",
        _fake_loader,
    )

    provider = ACSSourceProvider()
    frame = provider.load_frame()

    assert frame.source.name == "acs_2022"
    assert captured["spec"].key == "acs"
    assert captured["year"] == 2022
    assert captured["sample_n"] is None
    assert captured["random_seed"] == 0


def test_acs_source_provider_can_load_newer_storage_h5(
    tmp_path,
) -> None:
    storage_dir = tmp_path / "policyengine_us_data" / "storage"
    storage_dir.mkdir(parents=True)
    h5_path = storage_dir / "acs_2024.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("household_id", data=[1, 2])
        h5.create_dataset("person_household_id", data=[1, 1, 2])
        h5.create_dataset("person_id", data=[11, 12, 21])
        h5.create_dataset("age", data=[45, 12, 68])
        h5.create_dataset("is_male", data=[True, False, False])
        h5.create_dataset("is_household_head", data=[True, False, True])
        h5.create_dataset("state_fips", data=[6, 36])
        h5.create_dataset(
            "tenure_type",
            data=[b"OWNED_WITH_MORTGAGE", b"RENTED"],
        )
        h5.create_dataset("employment_income", data=[50_000.0, 0.0, 12_000.0])
        h5.create_dataset("self_employment_income", data=[5_000.0, 0.0, 0.0])
        h5.create_dataset("social_security", data=[0.0, 0.0, 20_000.0])
        h5.create_dataset(
            "taxable_private_pension_income",
            data=[0.0, 0.0, 15_000.0],
        )
        h5.create_dataset("rent", data=[1_200.0, 0.0, 950.0])
        h5.create_dataset("real_estate_taxes", data=[3_000.0, 0.0, 0.0])
        h5.create_dataset("household_weight", data=[100.0, 120.0])

    frame = ACSSourceProvider(
        year=2024, policyengine_us_data_repo=tmp_path
    ).load_frame()

    assert frame.source.name == "acs_2024"
    households = frame.tables[EntityType.HOUSEHOLD]
    persons = frame.tables[EntityType.PERSON]
    assert households["household_weight"].tolist() == [100.0, 120.0]
    assert persons["rent"].tolist() == [1_200.0, 0.0, 950.0]
    assert persons["tenure"].tolist() == [1, 1, 2]


def test_acs_source_provider_forwards_state_age_floor_query_filter() -> None:
    captured: dict[str, object] = {}

    def _loader(**kwargs) -> DonorSurveyTables:
        captured.update(kwargs)
        return _acs_tables()

    provider = ACSSourceProvider(loader=_loader)
    provider.load_frame(
        query=donor_surveys.SourceQuery(
            provider_filters={
                "sample_n": 2,
                "random_seed": 3,
                "state_age_floor": 1,
            }
        )
    )

    assert captured["sample_n"] == 2
    assert captured["random_seed"] == 3
    assert captured["state_age_floor"] == 1


def test_acs_source_provider_deduplicates_households_from_dataset_loader(
    monkeypatch,
) -> None:
    def _fake_loader(
        *, spec, year, sample_n, random_seed, **_kwargs
    ) -> DonorSurveyTables:
        households = pd.DataFrame(
            {
                "household_id": [1, 1, 2],
                "household_weight": [100.0, 100.0, 120.0],
                "state_fips": [6, 6, 36],
                "tenure": [1, 1, 2],
                "year": [2022, 2022, 2022],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [11, 12, 21],
                "household_id": [1, 1, 2],
                "age": [45, 12, 68],
                "sex": [1, 2, 2],
                "is_male": [1.0, 0.0, 0.0],
                "is_household_head": [1.0, 0.0, 1.0],
                "tenure_type": [1, 1, 2],
                "employment_income": [50_000.0, 0.0, 12_000.0],
                "self_employment_income": [5_000.0, 0.0, 0.0],
                "social_security": [0.0, 0.0, 20_000.0],
                "taxable_pension_income": [0.0, 0.0, 15_000.0],
                "rent": [1_200.0, 0.0, 950.0],
                "real_estate_taxes": [3_000.0, 0.0, 0.0],
                "income": [55_000.0, 0.0, 47_000.0],
                "weight": [100.0, 100.0, 120.0],
                "year": [2022, 2022, 2022],
            }
        )
        return DonorSurveyTables(households=households, persons=persons)

    monkeypatch.setattr(
        donor_surveys,
        "_run_policyengine_dataset_loader_from_spec",
        _fake_loader,
    )

    frame = ACSSourceProvider().load_frame()

    assert frame.tables[EntityType.HOUSEHOLD]["household_id"].tolist() == [1, 2]
    assert frame.tables[EntityType.PERSON]["household_id"].tolist() == [1, 1, 2]


def test_acs_source_provider_makes_duplicate_person_ids_household_scoped(
    monkeypatch,
) -> None:
    def _fake_loader(
        *, spec, year, sample_n, random_seed, **_kwargs
    ) -> DonorSurveyTables:
        households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "household_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
                "year": [2022, 2022],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [1, 2, 1],
                "household_id": [1, 1, 2],
                "age": [45, 12, 68],
                "sex": [1, 2, 2],
                "is_male": [1.0, 0.0, 0.0],
                "is_household_head": [1.0, 0.0, 1.0],
                "tenure_type": [1, 1, 2],
                "employment_income": [50_000.0, 0.0, 12_000.0],
                "self_employment_income": [5_000.0, 0.0, 0.0],
                "social_security": [0.0, 0.0, 20_000.0],
                "taxable_pension_income": [0.0, 0.0, 15_000.0],
                "rent": [1_200.0, 0.0, 950.0],
                "real_estate_taxes": [3_000.0, 0.0, 0.0],
                "income": [55_000.0, 0.0, 47_000.0],
                "weight": [100.0, 100.0, 120.0],
                "year": [2022, 2022, 2022],
            }
        )
        return DonorSurveyTables(households=households, persons=persons)

    monkeypatch.setattr(
        donor_surveys,
        "_run_policyengine_dataset_loader_from_spec",
        _fake_loader,
    )

    frame = ACSSourceProvider().load_frame()
    person_ids = frame.tables[EntityType.PERSON]["person_id"].tolist()

    assert person_ids == ["1:1", "1:2", "2:1"]
    assert len(person_ids) == len(set(person_ids))


def test_sipp_and_scf_provider_fillers_are_not_usable_as_conditions() -> None:
    tips_provider = SIPPSourceProvider(block="tips", loader=_sipp_tips_tables)
    assets_provider = SIPPSourceProvider(block="assets", loader=_sipp_assets_tables)
    scf_provider = SCFSourceProvider(loader=_scf_tables)

    tips_frame = tips_provider.load_frame()
    assets_frame = assets_provider.load_frame()
    scf_frame = scf_provider.load_frame()

    assert tips_frame.source.name == "sipp_tips_2023"
    assert assets_frame.source.name == "sipp_assets_2023"
    assert scf_frame.source.name == "scf_2022"
    assert tips_frame.source.allows_conditioning_on("state_fips") is False
    assert assets_frame.source.is_authoritative_for("tenure") is False
    assert scf_frame.source.allows_conditioning_on("state_fips") is False
    assert scf_frame.source.observes("net_worth", EntityType.PERSON)


def test_sipp_provider_uprates_amounts_to_target_year(tmp_path) -> None:
    _write_uprating_factors(
        tmp_path,
        {
            "employment_income_before_lsr": (1.0, 1.0, 1.1),
            "tip_income": (1.0, 1.0, 1.25),
        },
    )

    provider = SIPPSourceProvider(
        block="tips",
        loader=_sipp_tips_tables,
        policyengine_us_data_repo=tmp_path,
        target_year=2024,
    )

    frame = provider.load_frame()
    households = frame.tables[EntityType.HOUSEHOLD]
    persons = frame.tables[EntityType.PERSON]

    assert frame.source.name == "sipp_tips_2023"
    assert households["year"].tolist() == [2024, 2024]
    assert persons["year"].tolist() == [2024, 2024, 2024]
    assert persons["employment_income"].tolist() == pytest.approx(
        [44_000.0, 0.0, 27_500.0]
    )
    assert persons["income"].tolist() == pytest.approx([44_000.0, 0.0, 27_500.0])
    assert persons["tip_income"].tolist() == [1_125.0, 0.0, 312.5]
    assert persons["weight"].tolist() == [80.0, 80.0, 90.0]


def test_sipp_asset_provider_uprates_liquid_assets_to_target_year(tmp_path) -> None:
    _write_uprating_factors(
        tmp_path,
        {
            "employment_income_before_lsr": (1.0, 1.0, 1.1),
            "bank_account_assets": (1.0, 1.0, 1.2),
            "stock_assets": (1.0, 1.0, 1.3),
            "bond_assets": (1.0, 1.0, 1.4),
        },
    )

    provider = SIPPSourceProvider(
        block="assets",
        loader=_sipp_assets_tables,
        policyengine_us_data_repo=tmp_path,
        target_year=2024,
    )

    persons = provider.load_frame().tables[EntityType.PERSON]

    assert persons["employment_income"].tolist() == pytest.approx([44_000.0, 27_500.0])
    assert persons["income"].tolist() == pytest.approx([44_000.0, 27_500.0])
    assert persons["bank_account_assets"].tolist() == [3_000.0, 12_000.0]
    assert persons["stock_assets"].tolist() == [0.0, 5_200.0]
    assert persons["bond_assets"].tolist() == [0.0, 2_100.0]


def test_scf_provider_uprates_amounts_to_target_year(tmp_path) -> None:
    _write_uprating_factors(
        tmp_path,
        {
            "employment_income_before_lsr": (1.0, 1.0, 1.1),
            "taxable_interest_income": (1.0, 1.0, 2.0),
            "social_security_retirement": (1.0, 1.0, 1.5),
            "net_worth": (1.0, 1.0, 1.2),
            "auto_loan_balance": (1.0, 1.0, 1.3),
            "auto_loan_interest": (1.0, 1.0, 1.4),
        },
    )

    provider = SCFSourceProvider(
        loader=_scf_tables,
        policyengine_us_data_repo=tmp_path,
        target_year=2024,
    )

    frame = provider.load_frame()
    households = frame.tables[EntityType.HOUSEHOLD]
    persons = frame.tables[EntityType.PERSON]

    assert frame.source.name == "scf_2022"
    assert households["year"].tolist() == [2024, 2024]
    assert persons["year"].tolist() == [2024, 2024]
    assert persons["employment_income"].tolist() == [82_500.0, 0.0]
    assert persons["income"].tolist() == [82_500.0, 0.0]
    assert persons["interest_dividend_income"].tolist() == [2_400.0, 800.0]
    assert persons["social_security_pension_income"].tolist() == [0.0, 27_000.0]
    assert persons["net_worth"].tolist() == [420_000.0, 216_000.0]
    assert persons["auto_loan_balance"].tolist() == [10_400.0, 0.0]
    assert persons["auto_loan_interest"].tolist() == [770.0, 0.0]
    assert persons["weight"].tolist() == [10.0, 12.0]


def test_scf_source_provider_uses_manifest_backed_dataset_loader(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_loader(
        *, spec, year, sample_n, random_seed, **_kwargs
    ) -> DonorSurveyTables:
        captured["spec"] = spec
        captured["year"] = year
        captured["sample_n"] = sample_n
        captured["random_seed"] = random_seed
        return _scf_tables()

    monkeypatch.setattr(
        donor_surveys,
        "_run_policyengine_dataset_loader_from_spec",
        _fake_loader,
    )

    provider = SCFSourceProvider()
    frame = provider.load_frame()

    assert frame.source.name == "scf_2022"
    assert captured["spec"].key == "scf"
    assert captured["year"] == 2022
    assert captured["sample_n"] is None
    assert captured["random_seed"] == 0


def test_sipp_tips_provider_uses_manifest_backed_raw_loader(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "pu2023_slim.csv"
    pd.DataFrame(
        {
            "SSUID": ["100", "100", "101"],
            "MONTHCODE": [1, 1, 2],
            "PNUM": [1, 2, 1],
            "WPFINWGT": [80.0, 80.0, 90.0],
            "TAGE": [35, 8, 50],
            "ESEX": [1, 2, 2],
            "TPTOTINC": [1000.0, 0.0, 500.0],
            "TXAMT1": [10.0, 0.0, 5.0],
            "TXAMT2": [2.0, 0.0, 0.0],
        }
    ).to_csv(path, index=False)

    monkeypatch.setattr(
        donor_surveys,
        "_download_policyengine_us_data_file",
        lambda **_kwargs: path,
    )

    frame = SIPPSourceProvider(block="tips").load_frame()
    persons = frame.tables[EntityType.PERSON]

    assert frame.source.name == "sipp_tips_2023"
    assert persons["tip_income"].tolist() == [144.0, 0.0, 60.0]
    assert persons["employment_income"].tolist() == [12000.0, 0.0, 6000.0]
    assert persons["count_under_18"].tolist() == [1.0, 1.0, 0.0]
    assert persons["count_under_6"].tolist() == [0.0, 0.0, 0.0]


def test_sipp_assets_provider_uses_manifest_backed_raw_loader(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "pu2023.csv"
    pd.DataFrame(
        {
            "SSUID": ["100", "100", "101"],
            "PNUM": [1, 2, 1],
            "MONTHCODE": [11, 12, 12],
            "WPFINWGT": [80.0, 80.0, 90.0],
            "TAGE": [10, 35, 50],
            "ESEX": [1, 2, 2],
            "EMS": [2, 1, 0],
            "TPTOTINC": [100.0, 200.0, 300.0],
            "TVAL_BANK": [1.0, 2.0, 3.0],
            "TVAL_STMF": [4.0, 5.0, 6.0],
            "TVAL_BOND": [7.0, 8.0, 9.0],
        }
    ).to_csv(path, index=False, sep="|")

    monkeypatch.setattr(
        donor_surveys,
        "_download_policyengine_us_data_file",
        lambda **_kwargs: path,
    )

    frame = SIPPSourceProvider(block="assets").load_frame()
    persons = frame.tables[EntityType.PERSON]

    assert frame.source.name == "sipp_assets_2023"
    assert persons["person_id"].tolist() == ["100:2", "101:1"]
    assert persons["employment_income"].tolist() == [2400.0, 3600.0]
    assert persons["is_female"].tolist() == [1.0, 1.0]
    assert persons["is_married"].tolist() == [1.0, 0.0]
    assert persons["count_under_18"].tolist() == [0.0, 0.0]
