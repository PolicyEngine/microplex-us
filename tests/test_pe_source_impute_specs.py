"""Tests for shared PE source-impute donor block specs."""

from __future__ import annotations

import pandas as pd
from microplex.core import SourceArchetype

from microplex_us.pe_source_impute_specs import (
    apply_pe_source_impute_loader_postprocess,
    get_pe_source_impute_block_spec,
    load_pe_source_impute_block_specs,
    prepare_pe_source_impute_condition_frame,
    resolve_pe_source_impute_block_key,
    resolve_sipp_source_impute_block_spec,
)


def test_load_pe_source_impute_block_specs_reads_manifest() -> None:
    specs = load_pe_source_impute_block_specs()

    assert set(specs) == {"acs", "sipp_tips", "sipp_assets", "scf"}
    assert specs["acs"].archetype is SourceArchetype.HOUSEHOLD_INCOME
    assert specs["scf"].archetype is SourceArchetype.WEALTH
    assert specs["sipp_assets"].target_variables == (
        "bank_account_assets",
        "stock_assets",
        "bond_assets",
        "household_vehicles_owned",
        "household_vehicles_value",
    )
    assert specs["sipp_tips"].raw_loader is not None
    assert specs["sipp_tips"].raw_loader.filename == "pu2023_slim.csv"
    assert specs["sipp_assets"].raw_loader is not None
    assert specs["sipp_assets"].raw_loader.usecols[0] == "SSUID"
    assert specs["sipp_tips"].annualized_variables == (
        "tip_income",
        "employment_income",
    )
    assert specs["sipp_assets"].required_monthcode == 12
    assert specs["acs"].dataset_loader is not None
    assert specs["acs"].dataset_loader.module == "policyengine_us_data.datasets.acs.acs"
    assert specs["scf"].dataset_loader is not None
    assert specs["scf"].dataset_loader.builder_kind == "single_person_households"


def test_resolve_pe_source_impute_block_key_uses_source_name_and_targets() -> None:
    assert (
        resolve_pe_source_impute_block_key(
            donor_source_name="acs_2022",
            donor_block=("rent",),
        )
        == "acs"
    )
    assert (
        resolve_pe_source_impute_block_key(
            donor_source_name="sipp_assets_2023",
            donor_block=("stock_assets", "bond_assets"),
        )
        == "sipp_assets"
    )
    assert (
        resolve_pe_source_impute_block_key(
            donor_source_name="scf_2022",
            donor_block=("tip_income",),
        )
        is None
    )
    assert (
        resolve_pe_source_impute_block_key(
            donor_source_name="sipp_2023",
            donor_block=("tip_income",),
        )
        is None
    )


def test_resolve_sipp_source_impute_block_spec_and_named_lookup() -> None:
    tips = resolve_sipp_source_impute_block_spec("tips")
    scf = get_pe_source_impute_block_spec("scf")

    assert tips.key == "sipp_tips"
    assert tips.descriptor_name == "sipp_tips"
    assert scf.descriptor_name == "scf"
    assert tips.matches_source_name("sipp_tips_2023") is True
    assert tips.matches_source_name("sipp_2023") is False


def test_prepare_pe_source_impute_condition_frame_derives_manifest_backed_predictors() -> (
    None
):
    spec = get_pe_source_impute_block_spec("acs")
    frame = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "age": [45, 12, 70],
            "sex": [1, 2, 2],
            "is_head": [1.0, 0.0, 1.0],
            "tenure": [1, 1, 2],
            "employment_income": [50_000.0, 0.0, 12_000.0],
            "self_employment_income": [5_000.0, 0.0, 0.0],
            "gross_social_security": [0.0, 0.0, 20_000.0],
            "taxable_pension_income": [0.0, 0.0, 15_000.0],
            "state_fips": [6, 6, 36],
        }
    )

    prepared = prepare_pe_source_impute_condition_frame(frame, spec)

    assert prepared["is_male"].tolist() == [1.0, 0.0, 0.0]
    assert prepared["is_household_head"].tolist() == [1.0, 0.0, 1.0]
    assert prepared["tenure_type"].tolist() == [1.0, 1.0, 2.0]
    assert prepared["social_security"].tolist() == [0.0, 0.0, 20_000.0]
    assert prepared["pension_income"].tolist() == [0.0, 0.0, 15_000.0]
    assert prepared["household_size"].tolist() == [2.0, 2.0, 1.0]


def test_apply_pe_source_impute_loader_postprocess_uses_manifest_rules() -> None:
    spec = get_pe_source_impute_block_spec("sipp_assets")
    frame = pd.DataFrame(
        {
            "MONTHCODE": [11, 12, 12],
            "household_id": ["100", "100", "101"],
            "age": [10, 35, 5],
            "employment_income": [100.0, 200.0, 300.0],
        }
    )

    postprocessed = apply_pe_source_impute_loader_postprocess(frame, spec)

    assert postprocessed["household_id"].tolist() == ["100", "101"]
    assert postprocessed["employment_income"].tolist() == [2_400.0, 3_600.0]
    assert postprocessed["count_under_18"].tolist() == [0.0, 1.0]
