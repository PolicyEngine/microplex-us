"""Tests for Census block-derived target providers."""

from __future__ import annotations

import pandas as pd
from microplex.core import EntityType
from microplex.targets import TargetAggregation, TargetFilter, TargetQuery

from microplex_us.targets.census_blocks import (
    CENSUS_BLOCK_POPULATION_SOURCE,
    CensusBlockPopulationTargetProvider,
    build_census_block_population_targets,
)


def _sample_blocks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geoid": [
                "060010201001000",
                "060010201001001",
                "060030101001000",
                "360610001001000",
            ],
            "state_fips": ["06", "06", "06", "36"],
            "county": ["001", "001", "003", "061"],
            "tract": ["020100", "020100", "010100", "000100"],
            "population": [10, 20, 5, 7],
            "cd_id": ["CA-12", "CA-12", "CA-03", "NY-10"],
            "sldu_id": ["CA-SD-09", "CA-SD-09", "CA-SD-01", "NY-SD-30"],
            "sldl_id": ["CA-HD-18", "CA-HD-18", "CA-HD-05", "NY-AD-65"],
            "cbsa_code": ["41860", "41860", None, "35620"],
            "spm_metro_area": ["41860", "41860", "", "35620"],
        }
    )


def test_build_census_block_population_targets_rolls_parent_geographies() -> None:
    targets = build_census_block_population_targets(
        _sample_blocks(),
        geo_levels=("national", "state", "county", "tract", "cd", "sldu", "sldl"),
    )

    by_name = {target.name: target for target in targets}

    assert by_name["census_block_population_national"].value == 42
    assert by_name["census_block_population_state_06"].value == 35
    assert by_name["census_block_population_county_06001"].value == 30
    assert by_name["census_block_population_tract_06001020100"].value == 30
    assert by_name["census_block_population_cd_CA_12"].value == 30
    assert by_name["census_block_population_sldu_CA_SLDU_009"].value == 30
    assert by_name["census_block_population_sldl_CA_SLDL_018"].value == 30

    county = by_name["census_block_population_county_06001"]
    assert county.entity is EntityType.PERSON
    assert county.aggregation is TargetAggregation.COUNT
    assert county.source == CENSUS_BLOCK_POPULATION_SOURCE
    assert county.filters == (
        TargetFilter(feature="county_fips", operator="==", value="06001"),
    )
    assert county.metadata["variable"] == "person_count"
    assert county.metadata["geo_level"] == "county"
    assert county.metadata["geographic_id"] == "06001"
    assert county.metadata["block_rollup"] is True


def test_census_block_provider_filters_by_geo_level_and_id() -> None:
    provider = CensusBlockPopulationTargetProvider(block_probabilities=_sample_blocks())

    target_set = provider.load_target_set(
        TargetQuery(
            provider_filters={
                "geo_levels": ["county", "cd"],
                "geographic_ids": ["06001", "CA-03"],
                "variables": ["person_count"],
            },
        )
    )

    targets = sorted(target_set.targets, key=lambda target: target.name)

    assert [target.name for target in targets] == [
        "census_block_population_cd_CA_03",
        "census_block_population_county_06001",
    ]
    assert [target.value for target in targets] == [5, 30]


def test_census_block_provider_normalizes_legacy_sld_ids() -> None:
    provider = CensusBlockPopulationTargetProvider(block_probabilities=_sample_blocks())

    target_set = provider.load_target_set(
        TargetQuery(
            provider_filters={
                "geo_levels": ["sldu", "sldl"],
                "geographic_ids": ["CA-SD-09", "NY-AD-65"],
            }
        )
    )
    by_name = {target.name: target for target in target_set.targets}

    assert by_name["census_block_population_sldu_CA_SLDU_009"].value == 30
    assert by_name["census_block_population_sldl_NY_SLDL_065"].value == 7


def test_census_block_targets_use_geo_level_to_normalize_bare_sld_ids() -> None:
    targets = build_census_block_population_targets(
        _sample_blocks(),
        geo_levels=("sldu", "sldl"),
        geographic_ids=("06009", "36065"),
    )
    by_name = {target.name: target for target in targets}

    assert by_name["census_block_population_sldu_CA_SLDU_009"].value == 30
    assert by_name["census_block_population_sldl_NY_SLDL_065"].value == 7

    provider = CensusBlockPopulationTargetProvider(block_probabilities=_sample_blocks())
    target_set = provider.load_target_set(
        TargetQuery(
            provider_filters={
                "geo_levels": ["sldu", "sldl"],
                "geographic_ids": ["06009", "36065"],
            }
        )
    )
    provider_by_name = {target.name: target for target in target_set.targets}

    assert provider_by_name["census_block_population_sldu_CA_SLDU_009"].value == 30
    assert provider_by_name["census_block_population_sldl_NY_SLDL_065"].value == 7


def test_census_block_targets_resolve_all_before_bare_sld_filter_expansion() -> None:
    targets = build_census_block_population_targets(
        _sample_blocks(),
        geo_levels=("all",),
        geographic_ids=("06009",),
    )

    assert {
        target.name: target.value
        for target in targets
        if target.metadata["geo_level"] == "sldu"
    } == {"census_block_population_sldu_CA_SLDU_009": 30}


def test_census_block_provider_ignores_unrelated_variables() -> None:
    provider = CensusBlockPopulationTargetProvider(block_probabilities=_sample_blocks())

    target_set = provider.load_target_set(
        TargetQuery(provider_filters={"variables": ["household_count"]})
    )

    assert target_set.targets == []
