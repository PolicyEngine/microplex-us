"""CPS ASEC age recodes."""

import warnings

import numpy as np
import polars as pl

CPS_AGE_80_84_RANDOMIZATION_KEY = "age_randomization_80_84"


def _stable_string_hash(value: str) -> np.uint64:
    """Return a deterministic hash compatible with policyengine-us-data."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow encountered", RuntimeWarning)
        result = np.uint64(0)
        for byte in value.encode("utf-8"):
            result = result * np.uint64(31) + np.uint64(byte)
        result = result ^ (result >> np.uint64(33))
        result = result * np.uint64(0xFF51AFD7ED558CCD)
        result = result ^ (result >> np.uint64(33))
    return result


def cps_seeded_rng(key: str) -> np.random.Generator:
    """Create a deterministic CPS recode RNG without importing us-data."""
    seed = int(_stable_string_hash(key)) % (2**63)
    return np.random.default_rng(seed=seed)


def randomize_cps_topcoded_age_80_84(
    frame: pl.DataFrame,
    *,
    age_column: str = "age",
) -> pl.DataFrame:
    """Spread CPS A_AGE==80, meaning ages 80-84, across integer ages 80-84."""
    if age_column not in frame.columns:
        return frame
    ages = frame[age_column].to_numpy().astype(np.int64, copy=True)
    age_80 = ages == 80
    if not age_80.any():
        return frame

    rng = cps_seeded_rng(CPS_AGE_80_84_RANDOMIZATION_KEY)
    draws = rng.integers(80, 85, len(ages), dtype=np.int64)
    ages[age_80] = draws[age_80]
    return frame.with_columns(pl.Series(age_column, ages))
