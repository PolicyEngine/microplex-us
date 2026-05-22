"""Census block-derived target providers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from microplex.core import EntityType
from microplex.targets import (
    TabularRollupSpec,
    TabularRollupTargetProvider,
    TargetAggregation,
    TargetQuery,
    TargetSpec,
    as_string_tuple,
    build_tabular_rollup_targets,
)

from microplex_us.geography import (
    load_block_probabilities,
    normalize_state_legislative_district_id,
)

CENSUS_BLOCK_POPULATION_VARIABLE = "person_count"
CENSUS_BLOCK_POPULATION_SOURCE = "Census 2020 PL 94-171"
CENSUS_BLOCK_POPULATION_UNITS = "persons"
CENSUS_BLOCK_TARGET_PERIOD = 2024
CENSUS_BLOCK_SOURCE_YEAR = 2020
CENSUS_BLOCK_GEOGRAPHY_YEAR = 2020

DEFAULT_CENSUS_BLOCK_POPULATION_GEO_LEVELS: tuple[str, ...] = (
    "national",
    "state",
    "county",
    "cd",
    "sldu",
    "sldl",
    "cbsa",
    "spm_metro_area",
)


CensusBlockPopulationRollup = TabularRollupSpec


CENSUS_BLOCK_POPULATION_ROLLUPS: dict[str, CensusBlockPopulationRollup] = {
    "national": CensusBlockPopulationRollup(
        geo_level="national",
        source_column=None,
        filter_feature=None,
        group_name="census_block_population_national",
        name_prefix="census_block_population_national",
    ),
    "state": CensusBlockPopulationRollup(
        geo_level="state",
        source_column="state_fips",
        filter_feature="state_fips",
        group_name="census_block_population_state",
        name_prefix="census_block_population_state",
    ),
    "county": CensusBlockPopulationRollup(
        geo_level="county",
        source_column="county_fips",
        filter_feature="county_fips",
        group_name="census_block_population_county",
        name_prefix="census_block_population_county",
    ),
    "tract": CensusBlockPopulationRollup(
        geo_level="tract",
        source_column="tract_geoid",
        filter_feature="tract_geoid",
        group_name="census_block_population_tract",
        name_prefix="census_block_population_tract",
    ),
    "block": CensusBlockPopulationRollup(
        geo_level="block",
        source_column="geoid",
        filter_feature="block_geoid",
        group_name="census_block_population_block",
        name_prefix="census_block_population_block",
    ),
    "cd": CensusBlockPopulationRollup(
        geo_level="cd",
        source_column="cd_id",
        filter_feature="cd_id",
        group_name="census_block_population_cd",
        name_prefix="census_block_population_cd",
    ),
    "sldu": CensusBlockPopulationRollup(
        geo_level="sldu",
        source_column="sldu_id",
        filter_feature="sldu_id",
        group_name="census_block_population_sldu",
        name_prefix="census_block_population_sldu",
    ),
    "sldl": CensusBlockPopulationRollup(
        geo_level="sldl",
        source_column="sldl_id",
        filter_feature="sldl_id",
        group_name="census_block_population_sldl",
        name_prefix="census_block_population_sldl",
    ),
    "cbsa": CensusBlockPopulationRollup(
        geo_level="cbsa",
        source_column="cbsa_code",
        filter_feature="cbsa_code",
        group_name="census_block_population_cbsa",
        name_prefix="census_block_population_cbsa",
    ),
    "spm_metro_area": CensusBlockPopulationRollup(
        geo_level="spm_metro_area",
        source_column="spm_metro_area",
        filter_feature="spm_metro_area",
        group_name="census_block_population_spm_metro_area",
        name_prefix="census_block_population_spm_metro_area",
    ),
}
CENSUS_BLOCK_POPULATION_GEO_LEVELS: tuple[str, ...] = tuple(
    CENSUS_BLOCK_POPULATION_ROLLUPS
)


class CensusBlockPopulationTargetProvider(TabularRollupTargetProvider):
    """Build population count targets by rolling Census blocks to parent geos."""

    def __init__(
        self,
        block_probabilities: pd.DataFrame | None = None,
        *,
        block_probabilities_path: str | Path | None = None,
        default_geo_levels: Iterable[str] = DEFAULT_CENSUS_BLOCK_POPULATION_GEO_LEVELS,
        period: int = CENSUS_BLOCK_TARGET_PERIOD,
    ) -> None:
        super().__init__(
            block_probabilities,
            data_path=block_probabilities_path,
            data_loader=load_block_probabilities,
            prepare_data=_prepare_block_probabilities,
            rollups=CENSUS_BLOCK_POPULATION_ROLLUPS,
            value_column="population",
            variable=CENSUS_BLOCK_POPULATION_VARIABLE,
            variable_aliases=("population",),
            entity=EntityType.PERSON,
            aggregation=TargetAggregation.COUNT,
            period=period,
            source=CENSUS_BLOCK_POPULATION_SOURCE,
            units=CENSUS_BLOCK_POPULATION_UNITS,
            default_geo_levels=default_geo_levels,
            min_value=0.0,
            normalize_geographic_id=_normalize_census_block_geographic_id,
            base_metadata={
                "source_year": CENSUS_BLOCK_SOURCE_YEAR,
                "geography_year": CENSUS_BLOCK_GEOGRAPHY_YEAR,
                "source_artifact": "census_2020_pl_94_171_state_files",
                "support_artifact": "block_probabilities.parquet",
                "block_rollup": True,
            },
        )

    def load_target_set(self, query: TargetQuery | None = None):
        """Load Census block rollup targets with US SLD ID alias support."""
        if query is None or "geographic_ids" not in query.provider_filters:
            return super().load_target_set(query)
        provider_filters = dict(query.provider_filters)
        geo_levels = _requested_census_block_geo_levels(
            provider_filters,
            default_geo_levels=self.default_geo_levels,
        )
        provider_filters["geographic_ids"] = _expand_census_block_geographic_ids(
            provider_filters["geographic_ids"],
            geo_levels=geo_levels,
        )
        return super().load_target_set(
            TargetQuery(
                period=query.period,
                entity=query.entity,
                names=query.names,
                metadata_filters=query.metadata_filters,
                provider_filters=provider_filters,
            )
        )


def build_census_block_population_targets(
    block_probabilities: pd.DataFrame,
    *,
    geo_levels: Iterable[str] = DEFAULT_CENSUS_BLOCK_POPULATION_GEO_LEVELS,
    geographic_ids: Iterable[str] | None = None,
    period: int = CENSUS_BLOCK_TARGET_PERIOD,
) -> list[TargetSpec]:
    """Roll block-level Census population counts to canonical target specs."""
    requested_geo_levels = as_string_tuple(geo_levels)
    resolved_geo_levels = (
        CENSUS_BLOCK_POPULATION_GEO_LEVELS
        if requested_geo_levels == ("all",)
        else requested_geo_levels
    )
    return build_tabular_rollup_targets(
        _prepare_block_probabilities(block_probabilities),
        rollups=CENSUS_BLOCK_POPULATION_ROLLUPS,
        value_column="population",
        variable=CENSUS_BLOCK_POPULATION_VARIABLE,
        entity=EntityType.PERSON,
        aggregation=TargetAggregation.COUNT,
        period=period,
        source=CENSUS_BLOCK_POPULATION_SOURCE,
        units=CENSUS_BLOCK_POPULATION_UNITS,
        geo_levels=resolved_geo_levels,
        geographic_ids=_expand_census_block_geographic_ids(
            geographic_ids,
            geo_levels=resolved_geo_levels,
        ),
        min_value=0.0,
        normalize_geographic_id=_normalize_census_block_geographic_id,
        base_metadata={
            "source_year": CENSUS_BLOCK_SOURCE_YEAR,
            "geography_year": CENSUS_BLOCK_GEOGRAPHY_YEAR,
            "source_artifact": "census_2020_pl_94_171_state_files",
            "support_artifact": "block_probabilities.parquet",
            "block_rollup": True,
        },
    )


def _prepare_block_probabilities(block_probabilities: pd.DataFrame) -> pd.DataFrame:
    if "population" not in block_probabilities.columns:
        raise ValueError("Block probabilities must include a population column")
    blocks = block_probabilities.copy()
    blocks["population"] = pd.to_numeric(blocks["population"], errors="coerce")
    if "state_fips" in blocks.columns:
        blocks["state_fips"] = _zero_pad_series(blocks["state_fips"], 2)
    if "county_fips" in blocks.columns:
        blocks["county_fips"] = _zero_pad_series(blocks["county_fips"], 5)
    elif {"state_fips", "county"}.issubset(blocks.columns):
        blocks["county_fips"] = blocks["state_fips"] + _zero_pad_series(
            blocks["county"], 3
        )
    if "tract_geoid" not in blocks.columns and {
        "state_fips",
        "county",
        "tract",
    }.issubset(blocks.columns):
        blocks["tract_geoid"] = (
            blocks["state_fips"]
            + _zero_pad_series(blocks["county"], 3)
            + _zero_pad_series(blocks["tract"], 6)
        )
    if "sldu_id" in blocks.columns:
        blocks["sldu_id"] = blocks["sldu_id"].map(
            lambda value: (
                normalize_state_legislative_district_id(
                    value,
                    chamber="upper",
                )
                or ""
            )
        )
    if "sldl_id" in blocks.columns:
        blocks["sldl_id"] = blocks["sldl_id"].map(
            lambda value: (
                normalize_state_legislative_district_id(
                    value,
                    chamber="lower",
                )
                or ""
            )
        )
    for column in (
        "geoid",
        "tract_geoid",
        "cd_id",
        "cbsa_code",
        "spm_metro_area",
    ):
        if column in blocks.columns:
            blocks[column] = blocks[column].map(_normalize_geographic_id)
    return blocks


def _zero_pad_series(values: pd.Series, width: int) -> pd.Series:
    text = values.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    numeric_text = numeric.round().astype("Int64").astype("string").str.zfill(width)
    return text.where(numeric.isna(), numeric_text).str.zfill(width)


def _normalize_geographic_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _normalize_census_block_geographic_id(value: Any) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    normalized_sld = normalize_state_legislative_district_id(value)
    if normalized_sld is not None and normalized_sld != raw:
        return normalized_sld
    return _normalize_geographic_id(value)


def _requested_census_block_geo_levels(
    provider_filters: dict[str, Any],
    *,
    default_geo_levels: Iterable[str],
) -> tuple[str, ...]:
    if "geo_levels" in provider_filters:
        requested = as_string_tuple(provider_filters["geo_levels"])
    elif "geographic_levels" in provider_filters:
        requested = as_string_tuple(provider_filters["geographic_levels"])
    else:
        requested = tuple(default_geo_levels)
    return (
        tuple(CENSUS_BLOCK_POPULATION_ROLLUPS) if requested == ("all",) else requested
    )


def _expand_census_block_geographic_ids(
    geographic_ids: Iterable[str] | Any | None,
    *,
    geo_levels: Iterable[str],
) -> tuple[str, ...] | None:
    if geographic_ids is None:
        return None
    levels = set(as_string_tuple(geo_levels))
    include_upper = "sldu" in levels
    include_lower = "sldl" in levels
    expanded: list[str] = []
    for value in as_string_tuple(geographic_ids):
        normalized = _normalize_census_block_geographic_id(value)
        if normalized:
            expanded.append(normalized)
        if include_upper:
            upper = normalize_state_legislative_district_id(value, chamber="upper")
            if upper:
                expanded.append(upper)
        if include_lower:
            lower = normalize_state_legislative_district_id(value, chamber="lower")
            if lower:
                expanded.append(lower)
    return tuple(dict.fromkeys(expanded))


__all__ = [
    "CENSUS_BLOCK_GEOGRAPHY_YEAR",
    "CENSUS_BLOCK_POPULATION_GEO_LEVELS",
    "CENSUS_BLOCK_POPULATION_ROLLUPS",
    "CENSUS_BLOCK_POPULATION_SOURCE",
    "CENSUS_BLOCK_POPULATION_UNITS",
    "CENSUS_BLOCK_POPULATION_VARIABLE",
    "CENSUS_BLOCK_SOURCE_YEAR",
    "CENSUS_BLOCK_TARGET_PERIOD",
    "DEFAULT_CENSUS_BLOCK_POPULATION_GEO_LEVELS",
    "CensusBlockPopulationRollup",
    "CensusBlockPopulationTargetProvider",
    "build_census_block_population_targets",
]
