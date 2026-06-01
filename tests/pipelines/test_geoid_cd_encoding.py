"""Geoid carry-over guards for the CPS block-geography export (PR #129).

Two regressions are pinned here, both surfaced while reconciling PR #129
against the closed PR #130:

1. ``congressional_district_geoid`` encoding must match the eCPS 436-CD
   calibration universe, which encodes at-large districts (and DC) as district
   ``01`` -- e.g. AK->201, WY->5601, DC->1101. The raw Census block crosswalk
   carries at-large as ``00``/``98``; eCPS normalizes those to ``01`` in
   ``policyengine-us-data db/create_initial_strata.py`` before writing the
   targets DB and the dataset. ``_congressional_district_geoid_from_cd_id`` must
   reproduce the ``01`` convention regardless of which input form it is fed.

2. ``_attach_household_census_geographies`` must not assume a unique household
   -frame index. It assigns blocks and writes them back via ``.loc[row_index]``;
   a duplicate index previously raised
   ``ValueError: cannot reindex on an axis with duplicate labels``.
"""

import pandas as pd
import pytest

from microplex_us.pipelines.us import (
    _attach_household_census_geographies,
    _congressional_district_geoid_from_cd_id,
    _default_block_geography,
)

# state_fips for the 7 single-seat / at-large jurisdictions in the 119th Congress.
_AT_LARGE_STATES = {
    "AK": 2,
    "DE": 10,
    "DC": 11,
    "ND": 38,
    "SD": 46,
    "VT": 50,
    "WY": 56,
}


def test_multi_district_encoding_is_ssdd():
    cases = {
        ("CA-52", 6): 652,
        ("NY-12", 36): 3612,
        ("TX-38", 48): 4838,
        ("FL-28", 12): 1228,
        ("AL-02", 1): 102,
    }
    for (cd_id, state), expected in cases.items():
        assert _congressional_district_geoid_from_cd_id(cd_id, state) == expected


def test_at_large_uses_district_01_via_AL_token():
    # Microplex's crosswalk feeds the "<ST>-AL" token; every at-large state and
    # DC must encode to district 01 (state*100 + 1), matching the eCPS universe.
    for abbr, state in _AT_LARGE_STATES.items():
        assert (
            _congressional_district_geoid_from_cd_id(f"{abbr}-AL", state)
            == state * 100 + 1
        )


def test_raw_census_at_large_forms_normalize_to_01():
    # Hardening: even if a raw Census form leaks through (DC as "98", at-large as
    # "ZZ" or "00"), the encoder must still produce district 01, never 1198/5600.
    assert _congressional_district_geoid_from_cd_id("DC-98", 11) == 1101
    assert _congressional_district_geoid_from_cd_id("WY-ZZ", 56) == 5601
    assert _congressional_district_geoid_from_cd_id("AK-00", 2) == 201
    assert _congressional_district_geoid_from_cd_id("98", 11) == 1101


def test_no_at_large_geoid_ends_in_00_or_98():
    # The invariant that distinguishes the eCPS universe from the raw crosswalk.
    for abbr, state in _AT_LARGE_STATES.items():
        for token in (f"{abbr}-AL", f"{abbr}-00", f"{abbr}-98", f"{abbr}-ZZ"):
            geoid = _congressional_district_geoid_from_cd_id(token, state)
            assert geoid % 100 not in (0, 98), f"{token} -> {geoid}"


def test_invalid_inputs_return_zero():
    assert _congressional_district_geoid_from_cd_id("", 6) == 0
    assert _congressional_district_geoid_from_cd_id("nan", 6) == 0
    assert _congressional_district_geoid_from_cd_id("<NA>", 6) == 0
    assert _congressional_district_geoid_from_cd_id("CA-12", "not-a-state") == 0


# --- duplicate-index robustness -------------------------------------------------


class _StubAssigner:
    def __init__(self, block_geoid: str):
        self._block_geoid = block_geoid

    def assign(self, frame: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
        out = frame.copy()
        out["block_geoid"] = self._block_geoid
        return out


class _StubBlockGeography:
    """Minimal BlockGeography surface for exercising the assignment write-back.

    Returns a single deterministic CA block so the test focuses on index
    handling, not the probabilistic draw.
    """

    _BLOCK = "060371000001000"

    def __init__(self):
        self.data = pd.DataFrame({"county_fips": ["06037"]})

    def load_assigner(self, query) -> _StubAssigner:  # noqa: ANN001 - query unused
        return _StubAssigner(self._BLOCK)

    def assign(self, frame: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
        out = frame.copy()
        out["block_geoid"] = self._BLOCK
        return out

    def materialize(self, frame: pd.DataFrame, columns=()) -> pd.DataFrame:
        out = frame.copy()
        out["state_fips"] = "06"
        out["county_fips"] = "06037"
        out["tract_geoid"] = "06037100000"
        out["cd_id"] = "CA-37"
        return out


def test_attach_geographies_handles_duplicate_household_index():
    # Duplicate labels [0, 0, 1] previously broke the .loc[row_index] write-back.
    households = pd.DataFrame(
        {
            "household_id": [10, 11, 12],
            "state_fips": [6, 6, 6],
            "county_fips": [37, 37, 37],  # CPS fragment -> 06037
        },
        index=[0, 0, 1],
    )
    result = _attach_household_census_geographies(
        households, seed=0, geography=_StubBlockGeography()
    )
    assert len(result) == 3
    assert (result["block_geoid"] == "060371000001000").all()
    assert (result["tract_geoid"] == "06037100000").all()
    # CA-37 -> 6 * 100 + 37
    assert (result["congressional_district_geoid"] == 637).all()
    # household_id is preserved (consumed downstream via merge on this column).
    assert sorted(result["household_id"]) == [10, 11, 12]


# --- live universe parity (skips when the crosswalk parquet is unavailable) -----


def test_cd_encoder_reproduces_ecps_436_cd_universe():
    """Run the encoder over the real block crosswalk's distinct (state, cd_id).

    Verified during review to equal the eCPS calibration target universe in
    policy_data.db exactly (436 CDs, at-large=01). Skips in environments without
    the crosswalk parquet (e.g. CI).
    """
    try:
        data = _default_block_geography().data
    except (FileNotFoundError, OSError):
        pytest.skip("block crosswalk parquet not available")
    if "cd_id" not in data.columns or "state_fips" not in data.columns:
        pytest.skip("crosswalk lacks cd_id/state_fips columns")

    pairs = data[["state_fips", "cd_id"]].dropna().drop_duplicates()
    geoids = {
        _congressional_district_geoid_from_cd_id(cd_id, state)
        for state, cd_id in zip(pairs["state_fips"], pairs["cd_id"], strict=False)
    }
    geoids.discard(0)

    assert len(geoids) == 436, f"expected 436-CD universe, got {len(geoids)}"
    # at-large districts encode to 01, so nothing ends in 00.
    assert not any(g % 100 == 0 for g in geoids)
    for g in geoids:
        state, district = divmod(g, 100)
        assert 1 <= state <= 78, f"invalid state in {g}"
        assert 1 <= district <= 53, f"invalid district in {g}"
