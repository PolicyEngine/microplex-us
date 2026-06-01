"""US-specific Census block geography helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from microplex.geography import (
    AtomicGeographyCrosswalk,
    GeographyProvider,
    GeographyQuery,
    ProbabilisticAtomicGeographyAssigner,
    nearest_numeric_partition_key,
)

STATE_LEN = 2
COUNTY_LEN = 3
TRACT_LEN = 6
BLOCK_LEN = 4

STATE_GEOID_LEN = STATE_LEN
COUNTY_GEOID_LEN = STATE_LEN + COUNTY_LEN
TRACT_GEOID_LEN = STATE_LEN + COUNTY_LEN + TRACT_LEN
BLOCK_GEOID_LEN = STATE_LEN + COUNTY_LEN + TRACT_LEN + BLOCK_LEN

# At-large (single-district) states use district number 1 in the integer
# ``congressional_district_geoid`` (SSDD) convention. This matches the
# enhanced-CPS calibration targets exactly: AK->201, DE->1001, DC->1101,
# ND->3801, SD->4601, VT->5001, WY->5601. The block crosswalk encodes these
# districts with the ``"AL"`` suffix (e.g. ``"DC-AL"``).
AT_LARGE_DISTRICT_NUMBER = 1
AT_LARGE_DISTRICT_SUFFIX = "AL"

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR_CANDIDATES = (
    PACKAGE_ROOT / "data",
    PACKAGE_ROOT.parent / "microplex" / "data",
)
DEFAULT_DATA_DIR = next(
    (
        candidate
        for candidate in DEFAULT_DATA_DIR_CANDIDATES
        if (candidate / "block_probabilities.parquet").exists()
    ),
    next(
        (candidate for candidate in DEFAULT_DATA_DIR_CANDIDATES if candidate.exists()),
        DEFAULT_DATA_DIR_CANDIDATES[0],
    ),
)
DEFAULT_BLOCK_PROBABILITIES_PATH = DEFAULT_DATA_DIR / "block_probabilities.parquet"
DEFAULT_BLOCK_PROBABILITIES_SPM_GEOGRAPHY_PATH = (
    DEFAULT_DATA_DIR / "block_probabilities_spm_geography.parquet"
)
SPM_METRO_AREA_COLUMN = "spm_metro_area"
CENSUS_CBSA_DELINEATION_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/geographies/"
    "reference-files/2020/delineation-files/list1_2020.xls"
)
US_STATE_ABBR_BY_FIPS = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
    "72": "PR",
}

US_STATE_FIPS_BY_ABBR: dict[str, int] = {
    abbr: int(fips) for fips, abbr in US_STATE_ABBR_BY_FIPS.items()
}


def load_block_probabilities(path: str | Path | None = None) -> pd.DataFrame:
    """Load US Census block probabilities from parquet."""
    path = default_runtime_block_probabilities_path() if path is None else Path(path)
    if path is None:
        path = DEFAULT_BLOCK_PROBABILITIES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Block probabilities file not found at {path}.\n"
            "Run the US geography preparation pipeline first."
        )
    return pd.read_parquet(path)


def default_runtime_block_probabilities_path() -> Path | None:
    """Return the preferred runtime block probabilities file when available."""
    if DEFAULT_BLOCK_PROBABILITIES_SPM_GEOGRAPHY_PATH.exists():
        return DEFAULT_BLOCK_PROBABILITIES_SPM_GEOGRAPHY_PATH
    if DEFAULT_BLOCK_PROBABILITIES_PATH.exists():
        return DEFAULT_BLOCK_PROBABILITIES_PATH
    return None


def normalize_us_state_fips(value: Any) -> str:
    """Normalize US state FIPS values to two-character strings."""
    return str(int(round(float(value)))).zfill(2)


def _normalize_us_state_fips_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    normalized = numeric.round().astype("Int64").astype("string").str.zfill(2)
    return normalized.mask(numeric.isna())


def normalize_us_county_fips(value: Any) -> str | None:
    """Normalize US county FIPS values to five-character strings."""
    try:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return str(int(round(float(numeric)))).zfill(COUNTY_GEOID_LEN)
        digits = "".join(character for character in text if character.isdigit())
        return digits.zfill(COUNTY_GEOID_LEN) if digits else None
    except (TypeError, ValueError, OverflowError):
        return None


def normalize_state_legislative_district_id(
    value: Any,
    *,
    chamber: str | None = None,
) -> str | None:
    """Normalize state legislative district IDs to STATE-SLDU/SLDL-NNN."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raw = str(value).strip()
    if not raw:
        return None

    labeled = _normalize_labeled_state_legislative_district_id(raw)
    if labeled is not None:
        return labeled

    if raw.startswith("610U900US"):
        raw = raw[-5:]
        chamber = "upper"
    elif raw.startswith("620L900US"):
        raw = raw[-5:]
        chamber = "lower"

    if raw.isdigit() and len(raw) >= 5 and chamber in {"upper", "lower"}:
        state_fips = raw[:2]
        district_code = raw[2:]
        state_abbr = US_STATE_ABBR_BY_FIPS.get(state_fips, state_fips)
        chamber_label = "SLDU" if chamber == "upper" else "SLDL"
        return f"{state_abbr}-{chamber_label}-{_normalize_sld_district(district_code)}"

    return raw


def _normalize_labeled_state_legislative_district_id(raw: str) -> str | None:
    parts = raw.split("-")
    if len(parts) != 3:
        return None
    state, chamber_label, district_code = (part.strip() for part in parts)
    if not state or not chamber_label or not district_code:
        return None
    canonical_chamber_label = {
        "SLDU": "SLDU",
        "SD": "SLDU",
        "SLDL": "SLDL",
        "HD": "SLDL",
        "AD": "SLDL",
    }.get(chamber_label.upper())
    if canonical_chamber_label is None:
        return None
    return (
        f"{state.upper()}-{canonical_chamber_label}-"
        f"{_normalize_sld_district(district_code)}"
    )


def _normalize_sld_district(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return f"{int(text):03d}" if text.isdigit() else text


@lru_cache(maxsize=8)
def _spm_metro_area_codes(year: int) -> frozenset[str]:
    try:
        from spm_calculator.geoadj import list_metro_areas
    except ImportError:
        return frozenset()
    return frozenset(str(area["code"]) for area in list_metro_areas(year))


def state_nonmetro_spm_area_code(
    state_fips: Any,
    *,
    year: int = 2024,
) -> str | None:
    """Return the Census SPM state-nonmetro area code when one exists."""
    return _state_spm_area_code(state_fips, suffix=2, year=year)


def _state_spm_area_code(
    state_fips: Any,
    *,
    suffix: int,
    year: int,
) -> str | None:
    try:
        normalized_state = normalize_us_state_fips(state_fips)
    except (TypeError, ValueError, OverflowError):
        return None
    if normalized_state == "00":
        return None
    code = str(int(normalized_state) * 1_000 + suffix)
    return code if code in _spm_metro_area_codes(year) else None


def _state_spm_area_code_series(
    values: pd.Series,
    *,
    suffix: int,
    year: int,
) -> pd.Series:
    states = _normalize_us_state_fips_series(values)
    numeric_states = pd.to_numeric(states, errors="coerce")
    codes = (numeric_states * 1_000 + suffix).round().astype("Int64").astype("string")
    valid_codes = _spm_metro_area_codes(year)
    return codes.where(
        states.notna() & states.ne("00") & codes.isin(valid_codes)
    ).astype("string")


@lru_cache(maxsize=1)
def _census_cbsa_crosswalk() -> dict[str, str] | None:
    """Load county-to-CBSA crosswalk from Census' official delineation file."""
    try:
        delineation = pd.read_excel(
            CENSUS_CBSA_DELINEATION_URL,
            header=2,
            dtype=str,
        )
    except Exception:
        return None

    required_columns = {"CBSA Code", "FIPS State Code", "FIPS County Code"}
    if not required_columns.issubset(delineation.columns):
        return None

    rows = delineation.dropna(
        subset=["CBSA Code", "FIPS State Code", "FIPS County Code"]
    ).copy()
    rows["county_fips"] = rows["FIPS State Code"].astype(str).str.zfill(2) + rows[
        "FIPS County Code"
    ].astype(str).str.zfill(3)
    rows["cbsa_code"] = rows["CBSA Code"].astype(str).str.strip()
    return dict(zip(rows["county_fips"], rows["cbsa_code"], strict=False))


def _normalize_census_area_code(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"} or text in {"0", "00000"}:
        return None
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return str(int(round(float(numeric)))) if pd.notna(numeric) else text


def _normalize_census_area_code_series(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    invalid = (
        result.isna()
        | result.str.lower().isin({"", "nan", "none", "<na>"})
        | result.isin({"0", "00000"})
    )
    numeric = pd.to_numeric(result, errors="coerce")
    numeric_codes = numeric.round().astype("Int64").astype("string")
    result = result.mask(numeric.notna(), numeric_codes)
    return result.mask(invalid).astype("string")


def _normalize_spm_area_code(value: Any, *, year: int) -> str | None:
    code = _normalize_census_area_code(value)
    if code is None:
        return None
    return code if code in _spm_metro_area_codes(year) else None


def add_spm_metro_area_geography(
    frame: pd.DataFrame,
    *,
    year: int = 2024,
    county_column: str = "county_fips",
    state_column: str = "state_fips",
    cbsa_column: str = "cbsa_code",
    spm_metro_area_column: str = SPM_METRO_AREA_COLUMN,
    derive_cbsa_from_primary_source: bool = True,
) -> pd.DataFrame:
    """Attach Census SPM metro/nonmetro area IDs from block-derived geography.

    The final SPM threshold geography is a Census metropolitan area code when
    SPM publishes one; otherwise it is a state metro/nonmetro SPM code. We only
    classify blank, micropolitan, or unsupported CBSA values into a state area
    when they came from a trusted CBSA source: either an existing CBSA column or
    the Census delineation input.
    """
    if frame.empty:
        result = frame.copy()
        if spm_metro_area_column not in result.columns:
            result[spm_metro_area_column] = pd.Series(dtype="string")
        return result

    result = frame.copy()
    trusted_cbsa_source = cbsa_column in result.columns
    if cbsa_column in result.columns:
        cbsa_values = _normalize_census_area_code_series(result[cbsa_column])
    else:
        cbsa_values = pd.Series(pd.NA, index=result.index, dtype="string")

    if (
        derive_cbsa_from_primary_source
        and county_column in result.columns
        and cbsa_values.isna().any()
    ):
        cbsa_crosswalk = _census_cbsa_crosswalk()
        if cbsa_crosswalk is not None:
            trusted_cbsa_source = True
            county_values = result[county_column].map(normalize_us_county_fips)
            derived_cbsa = county_values.map(cbsa_crosswalk)
            cbsa_values = cbsa_values.combine_first(
                _normalize_census_area_code_series(derived_cbsa)
            )

    result[cbsa_column] = cbsa_values.astype("string")

    spm_area = cbsa_values.where(cbsa_values.isin(_spm_metro_area_codes(year))).astype(
        "string"
    )
    if trusted_cbsa_source and state_column in result.columns:
        nonmetro_codes = _state_spm_area_code_series(
            result[state_column],
            suffix=2,
            year=year,
        )
        state_metro_codes = _state_spm_area_code_series(
            result[state_column],
            suffix=1,
            year=year,
        )
        state_fallback_codes = nonmetro_codes.combine_first(
            state_metro_codes.where(cbsa_values.notna())
        )
        spm_area = spm_area.combine_first(state_fallback_codes)

    result[spm_metro_area_column] = spm_area
    return result


def derive_geographies(
    block_geoids: list[str] | np.ndarray | pd.Series,
    include_cd: bool = False,
    include_sld: bool = False,
    include_spm_metro_area: bool = False,
    block_data: pd.DataFrame | None = None,
    year: int = 2024,
) -> pd.DataFrame:
    """Derive parent geographies from Census block GEOIDs."""
    geoids = pd.Series(block_geoids).astype(str)
    result = pd.DataFrame(
        {
            "block_geoid": geoids,
            "state_fips": geoids.str[:STATE_GEOID_LEN],
            "county_fips": geoids.str[:COUNTY_GEOID_LEN],
            "tract_geoid": geoids.str[:TRACT_GEOID_LEN],
        }
    )
    if include_cd or include_sld:
        block_data = load_block_probabilities() if block_data is None else block_data
    if include_cd:
        result["cd_id"] = geoids.map(
            dict(zip(block_data["geoid"], block_data["cd_id"]))
        )
    if include_sld:
        if "sldu_id" in block_data.columns:
            result["sldu_id"] = geoids.map(
                dict(zip(block_data["geoid"], block_data["sldu_id"]))
            )
        if "sldl_id" in block_data.columns:
            result["sldl_id"] = geoids.map(
                dict(zip(block_data["geoid"], block_data["sldl_id"]))
            )
    if include_spm_metro_area:
        result = add_spm_metro_area_geography(result, year=year)
    return result


def cd_id_to_congressional_district_geoid(cd_id: Any) -> int | None:
    """Convert a block-crosswalk ``cd_id`` to the PE-US integer GEOID.

    The block probabilities crosswalk stores congressional districts as
    ``"<state_abbr>-<district>"`` strings, e.g. ``"CA-01"`` or the at-large
    sentinel ``"DC-AL"``. PolicyEngine-US (and the enhanced-CPS calibration
    targets) store ``congressional_district_geoid`` as an ``SSDD`` integer:
    ``state_fips * 100 + district_number``.

    At-large (single-district) states are encoded as district ``1`` to match the
    enhanced-CPS calibration-target universe exactly (e.g. ``"AK-AL"`` -> 201,
    ``"DC-AL"`` -> 1101, ``"WY-AL"`` -> 5601). Returns ``None`` for missing or
    unparseable values rather than fabricating a geoid.
    """
    if cd_id is None:
        return None
    try:
        if pd.isna(cd_id):
            return None
    except (TypeError, ValueError):
        pass
    text = str(cd_id).strip()
    if "-" not in text:
        return None
    abbr, district = text.rsplit("-", 1)
    state_fips = US_STATE_FIPS_BY_ABBR.get(abbr.strip().upper())
    if state_fips is None:
        return None
    district = district.strip().upper()
    if district == AT_LARGE_DISTRICT_SUFFIX:
        district_number = AT_LARGE_DISTRICT_NUMBER
    elif district.isdigit():
        district_number = int(district)
    else:
        return None
    return state_fips * 100 + district_number


def cd_id_series_to_congressional_district_geoid(values: pd.Series) -> pd.Series:
    """Vectorized :func:`cd_id_to_congressional_district_geoid` over a Series.

    Returns a nullable ``Int64`` Series so unmapped districts surface as ``<NA>``
    rather than silently becoming a real geoid.
    """
    mapping = {
        cd_id: cd_id_to_congressional_district_geoid(cd_id)
        for cd_id in pd.Series(values).dropna().astype(str).unique()
    }
    return pd.Series(values).astype("string").map(mapping).astype("Int64")


def assign_household_block_geography(
    households: pd.DataFrame,
    *,
    block_geography: BlockGeography | None = None,
    block_data: pd.DataFrame | None = None,
    state_column: str = "state_fips",
    county_column: str = "county_fips",
    cd_geoid_column: str = "congressional_district_geoid",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Attach Census block geography (``block_geoid``/``tract_geoid``/CD) to households.

    Each household is assigned a real 15-digit Census ``block_geoid`` via a
    population-weighted draw from the block-probabilities crosswalk, mirroring the
    enhanced-CPS block-assignment method. ``tract_geoid`` is then the true
    11-character prefix of ``block_geoid`` and ``congressional_district_geoid`` is
    looked up from the block's crosswalk ``cd_id``.

    The block draw is partitioned using the most specific geography available for
    each household, in priority order:

    1. ``county_fips`` when present and matching a crosswalk county (the CPS source
       county is the finest real signal),
    2. then an existing ``congressional_district_geoid`` (eCPS-style, preserves a
       known CD),
    3. then ``state_fips``.

    Households whose partition has no matching blocks keep empty/NA geography rather
    than receiving a fabricated geoid.

    Returns a copy of ``households`` with ``block_geoid`` (str), ``tract_geoid``
    (str), and ``congressional_district_geoid`` (Int64) columns added or updated.
    Existing ``state_fips``/``county_fips`` columns are left untouched.
    """
    if state_column not in households.columns:
        raise ValueError(
            f"Household table must contain '{state_column}' to assign block geography"
        )

    geography = block_geography
    if geography is None:
        geography = (
            BlockGeography.from_data(block_data)
            if block_data is not None
            else BlockGeography(lazy_load=False)
        )

    crosswalk = geography.data
    cd_id_by_geoid = dict(zip(crosswalk["geoid"].astype(str), crosswalk["cd_id"]))

    result = households.copy()
    n = len(result)
    block_geoids = pd.Series([pd.NA] * n, index=result.index, dtype="string")
    unresolved = pd.Series([True] * n, index=result.index)
    rng_seed = random_state

    # --- Partition 1: county_fips (most specific CPS-source geography) ---------
    if county_column in result.columns:
        county_values = result[county_column].map(normalize_us_county_fips)
        county_blocks = _county_block_index(crosswalk)
        if county_blocks:
            resolvable = (
                unresolved
                & county_values.notna()
                & county_values.isin(county_blocks.keys())
            )
            for county_fips, group in result.loc[resolvable].groupby(
                county_values.loc[resolvable]
            ):
                frame = county_blocks.get(str(county_fips))
                if frame is None or frame.empty:
                    continue
                block_geoids.loc[group.index] = _weighted_block_sample(
                    frame, len(group), rng_seed
                )
                unresolved.loc[group.index] = False

    # --- Partition 2: existing congressional_district_geoid (eCPS-style) -------
    if unresolved.any() and cd_geoid_column in result.columns:
        existing_cd = pd.to_numeric(result[cd_geoid_column], errors="coerce")
        cd_blocks = _cd_geoid_block_index(crosswalk)
        if cd_blocks:
            resolvable = unresolved & existing_cd.notna()
            for cd_geoid, group in result.loc[resolvable].groupby(
                existing_cd.loc[resolvable].astype(int)
            ):
                frame = cd_blocks.get(int(cd_geoid))
                if frame is None or frame.empty:
                    continue
                block_geoids.loc[group.index] = _weighted_block_sample(
                    frame, len(group), rng_seed
                )
                unresolved.loc[group.index] = False

    # --- Partition 3: state_fips (always-available fallback) -------------------
    if unresolved.any():
        state_values = _normalize_us_state_fips_series(result[state_column])
        state_blocks = _state_block_index(crosswalk)
        resolvable = unresolved & state_values.notna()
        for state_fips, group in result.loc[resolvable].groupby(
            state_values.loc[resolvable]
        ):
            frame = state_blocks.get(str(state_fips))
            if frame is None or frame.empty:
                continue
            block_geoids.loc[group.index] = _weighted_block_sample(
                frame, len(group), rng_seed
            )
            unresolved.loc[group.index] = False

    result["block_geoid"] = block_geoids.fillna("").astype("string")
    tract = block_geoids.str[:TRACT_GEOID_LEN]
    result["tract_geoid"] = tract.fillna("").astype("string")
    cd_id_series = block_geoids.map(cd_id_by_geoid)
    # Unresolved households fall back to the policyengine-us defaults: empty
    # block/tract strings and CD geoid 0 (PE-US ``congressional_district_geoid``
    # default_value), so the export stays a clean integer array. In practice the
    # state-partition fallback resolves every household with a valid state FIPS.
    cd_geoids = cd_id_series_to_congressional_district_geoid(cd_id_series)
    result[cd_geoid_column] = cd_geoids.fillna(0).astype("int64")
    return result


def _block_sample_columns(crosswalk: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("geoid", "prob", "population", "national_prob")
        if column in crosswalk.columns
    ]


def _county_block_index(crosswalk: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "county_fips" in crosswalk.columns:
        county = crosswalk["county_fips"].astype(str).str.zfill(COUNTY_GEOID_LEN)
    elif {"state_fips", "county"}.issubset(crosswalk.columns):
        county = crosswalk["state_fips"].astype(str).str.zfill(STATE_GEOID_LEN) + (
            crosswalk["county"].astype(str).str.zfill(COUNTY_LEN)
        )
    else:
        return {}
    columns = _block_sample_columns(crosswalk)
    return {
        str(county_fips): group[columns].copy()
        for county_fips, group in crosswalk.groupby(county, sort=False)
    }


def _state_block_index(crosswalk: pd.DataFrame) -> dict[str, pd.DataFrame]:
    state = crosswalk["state_fips"].astype(str).str.zfill(STATE_GEOID_LEN)
    columns = _block_sample_columns(crosswalk)
    return {
        str(state_fips): group[columns].copy()
        for state_fips, group in crosswalk.groupby(state, sort=False)
    }


def _cd_geoid_block_index(crosswalk: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if "cd_id" not in crosswalk.columns:
        return {}
    columns = _block_sample_columns(crosswalk)
    frames: dict[int, list[pd.DataFrame]] = {}
    for cd_id, group in crosswalk.groupby("cd_id", sort=False):
        geoid = cd_id_to_congressional_district_geoid(cd_id)
        if geoid is None:
            continue
        frames.setdefault(int(geoid), []).append(group[columns])
    return {
        geoid: pd.concat(parts, ignore_index=True) for geoid, parts in frames.items()
    }


def _weighted_block_sample(
    frame: pd.DataFrame,
    n: int,
    random_state: int | None,
) -> np.ndarray:
    """Draw ``n`` block GEOIDs weighted by block population/probability."""
    geoids = frame["geoid"].astype(str).to_numpy()
    weights = _block_sampling_weights(frame)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(geoids), size=n, replace=True, p=weights)
    return geoids[indices]


def _block_sampling_weights(frame: pd.DataFrame) -> np.ndarray:
    if "prob" in frame.columns:
        weights = pd.to_numeric(frame["prob"], errors="coerce").to_numpy(dtype=float)
    elif "population" in frame.columns:
        weights = pd.to_numeric(
            frame["population"], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        weights = np.ones(len(frame), dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = weights.sum()
    if total <= 0:
        return np.full(len(frame), 1.0 / len(frame))
    return weights / total


class BlockGeography(GeographyProvider):
    """US atomic-geography provider backed by Census blocks."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        lazy_load: bool = True,
    ):
        self._data_path = data_path
        self._data: pd.DataFrame | None = None
        self._cd_lookup: dict[str, str] | None = None
        self._sldu_lookup: dict[str, str] | None = None
        self._sldl_lookup: dict[str, str] | None = None
        self._state_blocks: dict[str, pd.DataFrame] | None = None
        if not lazy_load:
            self._load_data()

    @classmethod
    def from_data(cls, data: pd.DataFrame) -> BlockGeography:
        instance = cls(lazy_load=True)
        instance._data = data.copy()
        return instance

    def _load_data(self) -> None:
        if self._data is None:
            self._data = load_block_probabilities(self._data_path)

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._load_data()
        return self._data

    @staticmethod
    @lru_cache(maxsize=100000)
    def get_state(block_geoid: str) -> str:
        return block_geoid[:STATE_GEOID_LEN]

    @staticmethod
    @lru_cache(maxsize=100000)
    def get_county(block_geoid: str) -> str:
        return block_geoid[:COUNTY_GEOID_LEN]

    @staticmethod
    @lru_cache(maxsize=100000)
    def get_tract(block_geoid: str) -> str:
        return block_geoid[:TRACT_GEOID_LEN]

    def get_cd(self, block_geoid: str) -> str | None:
        if self._cd_lookup is None:
            self._build_lookups()
        return self._cd_lookup.get(block_geoid)

    def get_sldu(self, block_geoid: str) -> str | None:
        if self._sldu_lookup is None:
            self._build_lookups()
        return self._sldu_lookup.get(block_geoid)

    def get_sldl(self, block_geoid: str) -> str | None:
        if self._sldl_lookup is None:
            self._build_lookups()
        return self._sldl_lookup.get(block_geoid)

    def _build_lookups(self) -> None:
        self._cd_lookup = dict(zip(self.data["geoid"], self.data["cd_id"]))
        self._sldu_lookup = (
            dict(zip(self.data["geoid"], self.data["sldu_id"]))
            if "sldu_id" in self.data.columns
            else {}
        )
        self._sldl_lookup = (
            dict(zip(self.data["geoid"], self.data["sldl_id"]))
            if "sldl_id" in self.data.columns
            else {}
        )

    def get_all_geographies(self, block_geoid: str) -> dict[str, str | None]:
        return {
            "state_fips": self.get_state(block_geoid),
            "county_fips": self.get_county(block_geoid),
            "tract_geoid": self.get_tract(block_geoid),
            "cd_id": self.get_cd(block_geoid),
            "sldu_id": self.get_sldu(block_geoid),
            "sldl_id": self.get_sldl(block_geoid),
            "spm_metro_area": self._get_spm_metro_area(block_geoid),
        }

    def _get_spm_metro_area(self, block_geoid: str) -> str | None:
        if SPM_METRO_AREA_COLUMN in self.data.columns:
            match = self.data.loc[self.data["geoid"].astype(str).eq(block_geoid)]
            if not match.empty:
                value = match[SPM_METRO_AREA_COLUMN].iloc[0]
                return None if pd.isna(value) else str(value)

        rows = {
            "state_fips": [self.get_state(block_geoid)],
            "county_fips": [self.get_county(block_geoid)],
        }
        if "cbsa_code" in self.data.columns:
            match = self.data.loc[self.data["geoid"].astype(str).eq(block_geoid)]
            if not match.empty:
                rows["cbsa_code"] = [match["cbsa_code"].iloc[0]]
        value = add_spm_metro_area_geography(
            pd.DataFrame(rows),
            derive_cbsa_from_primary_source=False,
        )["spm_metro_area"].iloc[0]
        return None if pd.isna(value) else str(value)

    def to_crosswalk(self) -> AtomicGeographyCrosswalk:
        crosswalk = self.data.copy()
        if "county_fips" not in crosswalk.columns and {"state_fips", "county"}.issubset(
            crosswalk.columns
        ):
            crosswalk["county_fips"] = crosswalk["state_fips"].astype(str) + crosswalk[
                "county"
            ].astype(str)
        if "tract_geoid" not in crosswalk.columns and {
            "state_fips",
            "county",
            "tract",
        }.issubset(crosswalk.columns):
            crosswalk["tract_geoid"] = (
                crosswalk["state_fips"].astype(str)
                + crosswalk["county"].astype(str)
                + crosswalk["tract"].astype(str)
            )
        geography_columns = tuple(
            column
            for column in (
                "state_fips",
                "county_fips",
                "tract_geoid",
                "cd_id",
                "sldu_id",
                "sldl_id",
                "cbsa_code",
                SPM_METRO_AREA_COLUMN,
            )
            if column in crosswalk.columns
        )
        return AtomicGeographyCrosswalk(
            data=crosswalk.rename(columns={"geoid": "block_geoid"}),
            atomic_id_column="block_geoid",
            geography_columns=geography_columns,
            probability_column="prob" if "prob" in crosswalk.columns else None,
        )

    def load_crosswalk(
        self, query: GeographyQuery | None = None
    ) -> AtomicGeographyCrosswalk:
        query = query or GeographyQuery()
        crosswalk = self.to_crosswalk()
        if not query.geography_columns and query.probability_column is None:
            return crosswalk
        return AtomicGeographyCrosswalk(
            data=crosswalk.data.copy(),
            atomic_id_column=crosswalk.atomic_id_column,
            geography_columns=tuple(query.geography_columns)
            or crosswalk.geography_columns,
            probability_column=query.probability_column or crosswalk.probability_column,
        )

    def load_assigner(
        self,
        query: GeographyQuery | None = None,
    ) -> ProbabilisticAtomicGeographyAssigner:
        query = query or GeographyQuery()
        partition_columns = tuple(query.partition_columns) or ("state_fips",)
        partition_normalizers = dict(query.partition_normalizers)
        fallback_resolver = query.fallback_resolver
        if partition_columns == ("state_fips",):
            partition_normalizers.setdefault("state_fips", normalize_us_state_fips)
            if fallback_resolver is None:
                fallback_resolver = nearest_numeric_partition_key
        return ProbabilisticAtomicGeographyAssigner(
            crosswalk=self.load_crosswalk(query),
            partition_columns=partition_columns,
            probability_column=query.probability_column,
            partition_normalizers=partition_normalizers,
            fallback_resolver=fallback_resolver,
        )

    def assign(
        self,
        frame: pd.DataFrame,
        *,
        state_column: str = "state_fips",
        atomic_id_column: str = "block_geoid",
        random_state: int | None = None,
    ) -> pd.DataFrame:
        working = frame.copy()
        if state_column != "state_fips":
            working = working.rename(columns={state_column: "state_fips"})
        assigned = self.load_assigner().assign(
            working,
            atomic_id_column=atomic_id_column,
            random_state=random_state,
        )
        if state_column != "state_fips":
            assigned = assigned.rename(columns={"state_fips": state_column})
        return assigned

    def materialize(
        self,
        frame: pd.DataFrame,
        *,
        columns: tuple[str, ...] | list[str] | None = None,
        atomic_id_column: str = "block_geoid",
    ) -> pd.DataFrame:
        return self.to_crosswalk().materialize(
            frame,
            columns=columns,
            atomic_id_column=atomic_id_column,
        )

    def sample_blocks(
        self,
        state_fips: str,
        n: int,
        replace: bool = True,
        random_state: int | None = None,
    ) -> np.ndarray:
        if self._state_blocks is None:
            self._build_state_index()
        if state_fips not in self._state_blocks:
            raise ValueError(f"State FIPS '{state_fips}' not found in block data.")
        state_df = self._state_blocks[state_fips]
        if random_state is not None:
            np.random.seed(random_state)
        sampled_indices = np.random.choice(
            len(state_df),
            size=n,
            replace=replace,
            p=state_df["prob"].values,
        )
        geoids = state_df["geoid"].astype(str).to_numpy()
        return np.asarray(geoids[sampled_indices])

    def _build_state_index(self) -> None:
        self._state_blocks = {}
        for state_fips, group in self.data.groupby("state_fips"):
            self._state_blocks[state_fips] = group[["geoid", "prob"]].copy()

    def sample_blocks_national(
        self,
        n: int,
        replace: bool = True,
        random_state: int | None = None,
    ) -> np.ndarray:
        if random_state is not None:
            np.random.seed(random_state)
        sampled_indices = np.random.choice(
            len(self.data),
            size=n,
            replace=replace,
            p=self.data["national_prob"].values,
        )
        geoids = self.data["geoid"].astype(str).to_numpy()
        return np.asarray(geoids[sampled_indices])

    def get_blocks_in_state(self, state_fips: str) -> pd.DataFrame:
        return self.data[self.data["state_fips"] == state_fips].copy()

    def get_blocks_in_county(self, county_fips: str) -> pd.DataFrame:
        state = county_fips[:STATE_GEOID_LEN]
        county = county_fips[STATE_GEOID_LEN:]
        return self.data[
            (self.data["state_fips"] == state) & (self.data["county"] == county)
        ].copy()

    def get_blocks_in_tract(self, tract_geoid: str) -> pd.DataFrame:
        return self.data[self.data["tract_geoid"] == tract_geoid].copy()

    def get_blocks_in_cd(self, cd_id: str) -> pd.DataFrame:
        return self.data[self.data["cd_id"] == cd_id].copy()

    def get_blocks_in_sldu(self, sldu_id: str) -> pd.DataFrame:
        if "sldu_id" not in self.data.columns:
            return pd.DataFrame()
        return self.data[self.data["sldu_id"] == sldu_id].copy()

    def get_blocks_in_sldl(self, sldl_id: str) -> pd.DataFrame:
        if "sldl_id" not in self.data.columns:
            return pd.DataFrame()
        return self.data[self.data["sldl_id"] == sldl_id].copy()

    @property
    def states(self) -> list[str]:
        return sorted(self.data["state_fips"].unique())

    @property
    def n_blocks(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self._data is None:
            return "BlockGeography(not loaded)"
        return f"BlockGeography({self.n_blocks:,} blocks, {len(self.states)} states)"


__all__ = [
    "STATE_LEN",
    "COUNTY_LEN",
    "TRACT_LEN",
    "BLOCK_LEN",
    "STATE_GEOID_LEN",
    "COUNTY_GEOID_LEN",
    "TRACT_GEOID_LEN",
    "BLOCK_GEOID_LEN",
    "DEFAULT_DATA_DIR",
    "DEFAULT_BLOCK_PROBABILITIES_PATH",
    "DEFAULT_BLOCK_PROBABILITIES_SPM_GEOGRAPHY_PATH",
    "CENSUS_CBSA_DELINEATION_URL",
    "SPM_METRO_AREA_COLUMN",
    "default_runtime_block_probabilities_path",
    "load_block_probabilities",
    "normalize_us_state_fips",
    "normalize_us_county_fips",
    "normalize_state_legislative_district_id",
    "state_nonmetro_spm_area_code",
    "add_spm_metro_area_geography",
    "derive_geographies",
    "cd_id_to_congressional_district_geoid",
    "cd_id_series_to_congressional_district_geoid",
    "assign_household_block_geography",
    "AT_LARGE_DISTRICT_NUMBER",
    "AT_LARGE_DISTRICT_SUFFIX",
    "US_STATE_ABBR_BY_FIPS",
    "US_STATE_FIPS_BY_ABBR",
    "BlockGeography",
]
