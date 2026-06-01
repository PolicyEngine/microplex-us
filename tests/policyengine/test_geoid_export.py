"""Tests for Census block GEOID export wiring (G5).

Covers the three GEOID leaves the enhanced-CPS export carries:
``block_geoid``, ``tract_geoid``, and ``congressional_district_geoid``.

The contract being verified end-to-end:

* ``block_geoid`` is a real 15-digit Census block GEOID drawn from the
  population-weighted crosswalk (never fabricated).
* ``tract_geoid == block_geoid[:11]`` exactly.
* ``congressional_district_geoid`` is the integer ``SSDD`` GEOID derived from
  the block's crosswalk ``cd_id`` (at-large states use district ``1``, matching
  the enhanced-CPS calibration-target universe).
* All three are members of ``SAFE_POLICYENGINE_US_EXPORT_VARIABLES`` and survive
  the PolicyEngine-US export map / forbidden-column guard.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microplex_us.geography import (
    AT_LARGE_DISTRICT_NUMBER,
    BLOCK_GEOID_LEN,
    TRACT_GEOID_LEN,
    US_STATE_ABBR_BY_FIPS,
    BlockGeography,
    assign_household_block_geography,
    cd_id_series_to_congressional_district_geoid,
    cd_id_to_congressional_district_geoid,
    default_runtime_block_probabilities_path,
)
from microplex_us.policyengine.us import (
    SAFE_POLICYENGINE_US_EXPORT_VARIABLES,
    PolicyEngineUSEntityTableBundle,
    build_policyengine_us_export_variable_maps,
    build_policyengine_us_time_period_arrays,
    resolve_policyengine_excluded_export_variables,
    write_policyengine_us_time_period_dataset,
)

GEOID_LEAVES = ("block_geoid", "tract_geoid", "congressional_district_geoid")


def _runtime_crosswalk_path() -> Path | None:
    return default_runtime_block_probabilities_path()


def _require_crosswalk() -> BlockGeography:
    path = _runtime_crosswalk_path()
    if path is None or not Path(path).exists():
        pytest.skip("Block probabilities crosswalk not available")
    return BlockGeography(path, lazy_load=False)


# ---------------------------------------------------------------------------
# congressional_district_geoid integer conversion
# ---------------------------------------------------------------------------


class TestCDIdConversion:
    def test_standard_districts_use_ssdd(self) -> None:
        # state_fips * 100 + district, matching the enhanced-CPS targets.
        assert cd_id_to_congressional_district_geoid("CA-01") == 601
        assert cd_id_to_congressional_district_geoid("CA-52") == 652
        assert cd_id_to_congressional_district_geoid("NC-01") == 3701
        assert cd_id_to_congressional_district_geoid("AL-02") == 102
        assert cd_id_to_congressional_district_geoid("NY-12") == 3612

    def test_at_large_states_use_district_one(self) -> None:
        # At-large encoding matches the enhanced-CPS calibration targets exactly.
        assert cd_id_to_congressional_district_geoid("AK-AL") == 201
        assert cd_id_to_congressional_district_geoid("DE-AL") == 1001
        assert cd_id_to_congressional_district_geoid("DC-AL") == 1101
        assert cd_id_to_congressional_district_geoid("ND-AL") == 3801
        assert cd_id_to_congressional_district_geoid("SD-AL") == 4601
        assert cd_id_to_congressional_district_geoid("VT-AL") == 5001
        assert cd_id_to_congressional_district_geoid("WY-AL") == 5601
        assert AT_LARGE_DISTRICT_NUMBER == 1

    def test_unparseable_returns_none(self) -> None:
        assert cd_id_to_congressional_district_geoid(None) is None
        assert cd_id_to_congressional_district_geoid(float("nan")) is None
        assert cd_id_to_congressional_district_geoid("") is None
        assert cd_id_to_congressional_district_geoid("ZZ-01") is None
        assert cd_id_to_congressional_district_geoid("06013") is None

    def test_state_component_recovers_fips(self) -> None:
        # geoid // 100 must recover the state FIPS for every state.
        for fips, abbr in US_STATE_ABBR_BY_FIPS.items():
            geoid = cd_id_to_congressional_district_geoid(f"{abbr}-01")
            assert geoid is not None
            assert geoid // 100 == int(fips)

    def test_series_conversion_is_nullable_int(self) -> None:
        series = pd.Series(["CA-01", "DC-AL", None, "ZZ-99"])
        result = cd_id_series_to_congressional_district_geoid(series)
        assert str(result.dtype) == "Int64"
        assert result.tolist()[:2] == [601, 1101]
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])


# ---------------------------------------------------------------------------
# Household block-geography assignment (real crosswalk)
# ---------------------------------------------------------------------------


class TestHouseholdBlockAssignment:
    def test_state_partitioned_draw_produces_valid_geoids(self) -> None:
        geography = _require_crosswalk()
        households = pd.DataFrame(
            {
                "household_id": [1, 2, 3, 4, 5, 6],
                "state_fips": [6, 36, 48, 1, 11, 56],
            }
        )
        out = assign_household_block_geography(
            households, block_geography=geography, random_state=42
        )

        for leaf in GEOID_LEAVES:
            assert leaf in out.columns

        block = out["block_geoid"].astype(str)
        tract = out["tract_geoid"].astype(str)
        cd = out["congressional_district_geoid"]

        # block_geoid: real 15-digit GEOIDs.
        assert (block.str.len() == BLOCK_GEOID_LEN).all()
        assert block.str.isdigit().all()

        # tract_geoid is the TRUE 11-char prefix of block_geoid.
        assert (tract == block.str[:TRACT_GEOID_LEN]).all()
        assert (tract.str.len() == TRACT_GEOID_LEN).all()

        # block state prefix matches the requested household state.
        assert (block.str[:2].astype(int) == out["state_fips"].astype(int)).all()

        # congressional_district_geoid resolves for every household and is
        # consistent with the household state.
        assert cd.notna().all()
        # assign_household_block_geography fills unresolved CDs with 0 and exports
        # a plain int64 column (matching PE-US's integer congressional_district_geoid).
        assert cd.dtype.kind in {"i", "u"}
        assert ((cd // 100).astype(int) == out["state_fips"].astype(int)).all()
        # 3- or 4-digit SSDD values only.
        assert cd.astype(int).astype(str).str.len().isin([3, 4]).all()

    def test_at_large_state_resolves_to_district_one(self) -> None:
        geography = _require_crosswalk()
        households = pd.DataFrame({"household_id": [1], "state_fips": [11]})  # DC
        out = assign_household_block_geography(
            households, block_geography=geography, random_state=0
        )
        assert int(out["congressional_district_geoid"].iloc[0]) == 1101

    def test_county_partitioned_draw_lands_in_county(self) -> None:
        geography = _require_crosswalk()
        # Disclosed CPS counties: LA County, NY County (Manhattan), Cook County.
        households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "state_fips": [6, 36, 17],
                "county_fips": ["06037", "36061", "17031"],
            }
        )
        out = assign_household_block_geography(
            households, block_geography=geography, random_state=42
        )
        block = out["block_geoid"].astype(str)
        tract = out["tract_geoid"].astype(str)
        # Block's 5-char county prefix matches the household's CPS county.
        assert (block.str[:5] == out["county_fips"]).all()
        assert (tract == block.str[:TRACT_GEOID_LEN]).all()
        assert out["congressional_district_geoid"].notna().all()

    def test_suppressed_county_falls_back_to_state(self) -> None:
        geography = _require_crosswalk()
        # CPS suppresses county for confidentiality in many records (GTCO == 0).
        households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "state_fips": [6, 48],
                "county_fips": ["00000", "0"],
            }
        )
        out = assign_household_block_geography(
            households, block_geography=geography, random_state=1
        )
        block = out["block_geoid"].astype(str)
        assert (block.str.len() == BLOCK_GEOID_LEN).all()
        # Still lands in the correct state via the state-partition fallback.
        assert (block.str[:2].astype(int) == out["state_fips"].astype(int)).all()

    def test_cd_partitioned_draw_preserves_existing_cd(self) -> None:
        geography = _require_crosswalk()
        households = pd.DataFrame(
            {
                "household_id": [10, 11, 12],
                "state_fips": [6, 6, 37],
                "congressional_district_geoid": [649, 612, 3701],
            }
        )
        out = assign_household_block_geography(
            households, block_geography=geography, random_state=7
        )
        # CD preserved exactly; block drawn from WITHIN that CD.
        assert out["congressional_district_geoid"].tolist() == [649, 612, 3701]
        block = out["block_geoid"].astype(str)
        tract = out["tract_geoid"].astype(str)
        assert (block.str.len() == BLOCK_GEOID_LEN).all()
        assert (tract == block.str[:TRACT_GEOID_LEN]).all()

    def test_reproducible_with_seed(self) -> None:
        geography = _require_crosswalk()
        households = pd.DataFrame(
            {"household_id": list(range(20)), "state_fips": [6] * 20}
        )
        first = assign_household_block_geography(
            households, block_geography=geography, random_state=123
        )
        second = assign_household_block_geography(
            households, block_geography=geography, random_state=123
        )
        pd.testing.assert_series_equal(first["block_geoid"], second["block_geoid"])

    def test_does_not_clobber_state_or_county(self) -> None:
        geography = _require_crosswalk()
        households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "state_fips": [6, 36],
                "county_fips": ["06037", "36061"],
            }
        )
        out = assign_household_block_geography(
            households, block_geography=geography, random_state=1
        )
        assert out["state_fips"].tolist() == [6, 36]
        assert out["county_fips"].tolist() == ["06037", "36061"]

    def test_missing_state_column_raises(self) -> None:
        geography = _require_crosswalk()
        with pytest.raises(ValueError, match="state_fips"):
            assign_household_block_geography(
                pd.DataFrame({"household_id": [1]}), block_geography=geography
            )


# ---------------------------------------------------------------------------
# Allowlist membership + export wiring (no PolicyEngine-US dependency)
# ---------------------------------------------------------------------------


class _FakeEntity:
    def __init__(self, key: str) -> None:
        self.key = key


class _FakeVariable:
    def __init__(self, entity: _FakeEntity, formulas: dict | None = None) -> None:
        self.entity = entity
        self.formulas = formulas or {}


class _FakeSystem:
    """Mimics the storable-INPUT geo variables in policyengine-us."""

    variables = {
        "employment_income": _FakeVariable(_FakeEntity("person")),
        "state_fips": _FakeVariable(_FakeEntity("household")),
        "county_fips": _FakeVariable(_FakeEntity("household")),
        "block_geoid": _FakeVariable(_FakeEntity("household")),
        "tract_geoid": _FakeVariable(_FakeEntity("household")),
        "congressional_district_geoid": _FakeVariable(_FakeEntity("household")),
    }


class TestGeoidExportWiring:
    def test_all_three_leaves_in_allowlist(self) -> None:
        for leaf in GEOID_LEAVES:
            assert leaf in SAFE_POLICYENGINE_US_EXPORT_VARIABLES

    def _tables(self) -> PolicyEngineUSEntityTableBundle:
        households = pd.DataFrame(
            {
                "household_id": [0, 1, 2],
                "household_weight": [100.0, 200.0, 300.0],
                "state_fips": [6, 36, 11],
                "county_fips": ["06037", "36061", "11001"],
                "block_geoid": [
                    "060372073021001",
                    "360610001001000",
                    "110010091022000",
                ],
                "tract_geoid": ["06037207302", "36061000100", "11001009102"],
                "congressional_district_geoid": [630, 3612, 1101],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": [0, 1, 2, 3],
                "household_id": [0, 0, 1, 2],
                "age": [40, 10, 30, 55],
                "is_household_head": [True, False, True, True],
            }
        )
        return PolicyEngineUSEntityTableBundle(households=households, persons=persons)

    def test_export_map_includes_geoids(self) -> None:
        tables = self._tables()
        export_maps = build_policyengine_us_export_variable_maps(
            tables, tax_benefit_system=_FakeSystem()
        )
        for leaf in GEOID_LEAVES:
            assert export_maps["household"].get(leaf) == leaf

    def test_geoids_not_excluded_by_guard(self) -> None:
        tables = self._tables()
        export_maps = build_policyengine_us_export_variable_maps(
            tables, tax_benefit_system=_FakeSystem()
        )
        excluded = resolve_policyengine_excluded_export_variables(
            _FakeSystem(),
            sorted({t for m in export_maps.values() for t in m.values()}),
        )
        for leaf in GEOID_LEAVES:
            assert leaf not in excluded

    def test_written_h5_carries_geoids_with_tract_prefix(self, tmp_path: Path) -> None:
        h5py = pytest.importorskip("h5py")
        tables = self._tables()
        export_maps = build_policyengine_us_export_variable_maps(
            tables, tax_benefit_system=_FakeSystem()
        )
        excluded = resolve_policyengine_excluded_export_variables(
            _FakeSystem(),
            sorted({t for m in export_maps.values() for t in m.values()}),
        )
        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=2024,
            household_variable_map=export_maps["household"],
            person_variable_map=export_maps["person"],
            tax_unit_variable_map=export_maps["tax_unit"],
            spm_unit_variable_map=export_maps["spm_unit"],
            family_variable_map=export_maps["family"],
        )
        out = tmp_path / "pe.h5"
        write_policyengine_us_time_period_dataset(
            arrays, out, excluded_variables=excluded
        )
        with h5py.File(out, "r") as handle:
            for leaf in GEOID_LEAVES:
                assert leaf in handle, f"{leaf} missing from H5"
            block = [
                value.decode() if isinstance(value, bytes) else value
                for value in np.asarray(handle["block_geoid"]["2024"]).tolist()
            ]
            tract = [
                value.decode() if isinstance(value, bytes) else value
                for value in np.asarray(handle["tract_geoid"]["2024"]).tolist()
            ]
            cd = np.asarray(handle["congressional_district_geoid"]["2024"])

            assert all(len(b) == BLOCK_GEOID_LEN for b in block)
            assert all(t == b[:TRACT_GEOID_LEN] for t, b in zip(tract, block))
            assert cd.dtype.kind in {"i", "u"}
            assert cd.tolist() == [630, 3612, 1101]


# ---------------------------------------------------------------------------
# Cross-check against the enhanced-CPS calibration-target CD universe
# ---------------------------------------------------------------------------


def _ecps_targets_db() -> Path | None:
    candidate = Path(
        "/Users/maxghenis/PolicyEngine/policyengine-us-data/"
        "policyengine_us_data/storage/calibration/policy_data.db"
    )
    return candidate if candidate.exists() else None


class TestEnhancedCPSParity:
    def test_cd_universe_matches_enhanced_cps_targets(self) -> None:
        db_path = _ecps_targets_db()
        if db_path is None:
            pytest.skip("Enhanced-CPS calibration-target DB not available")
        geography = _require_crosswalk()

        connection = sqlite3.connect(db_path)
        try:
            rows = connection.execute(
                "SELECT DISTINCT value FROM stratum_constraints "
                "WHERE constraint_variable = 'congressional_district_geoid'"
            ).fetchall()
        finally:
            connection.close()
        target_cds = {int(row[0]) for row in rows}

        crosswalk_cds = {
            cd_id_to_congressional_district_geoid(cd_id)
            for cd_id in geography.data["cd_id"].dropna().unique()
        }
        crosswalk_cds.discard(None)

        # Every block-crosswalk CD maps to a real enhanced-CPS target CD and the
        # universes coincide exactly (at-large encoding agrees).
        assert crosswalk_cds == target_cds
