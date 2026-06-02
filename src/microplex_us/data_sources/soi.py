"""IRS SOI target-table artifact resolution for US source adapters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "microplex"
SOI_TARGETS_POLICYENGINE_US_DATA_REF = "f7458313c86fa580fb1e43a2f18252d67cf76e4a"
SOI_TARGETS_CACHE_FILENAME = (
    f"soi_targets_pe_us_data_{SOI_TARGETS_POLICYENGINE_US_DATA_REF}.csv"
)
# The HF policyengine-us-data model publishes current raw inputs/target DB
# artifacts, but not this historical PE-style long target table.
SOI_TARGETS_URL = (
    "https://raw.githubusercontent.com/PolicyEngine/policyengine-us-data/"
    f"{SOI_TARGETS_POLICYENGINE_US_DATA_REF}/"
    "policyengine_us_data/storage/calibration_targets/soi_targets.csv"
)

PE_SOI_TARGETS_REQUIRED_COLUMNS = frozenset(
    {
        "Year",
        "Variable",
        "Filing status",
        "AGI lower bound",
        "AGI upper bound",
        "Count",
        "Taxable only",
        "Value",
    }
)


def _cache_safe_ref(ref: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in ref
    )


def _soi_targets_cache_filename(revision: str) -> str:
    return f"soi_targets_pe_us_data_{_cache_safe_ref(revision)}.csv"


def validate_pe_soi_targets_file(path: str | Path) -> Path:
    """Validate the PE-style long SOI target table schema."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Could not find PE SOI targets file at {resolved}")

    try:
        columns = set(pd.read_csv(resolved, nrows=0).columns)
    except Exception as exc:
        raise ValueError(f"Could not read PE SOI targets file at {resolved}") from exc

    missing = sorted(PE_SOI_TARGETS_REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(
            "PE SOI targets file is missing required columns "
            f"{missing}: {resolved}"
        )
    return resolved


def pe_soi_targets_cache_path(
    cache_dir: str | Path | None = None,
    *,
    revision: str = SOI_TARGETS_POLICYENGINE_US_DATA_REF,
) -> Path:
    """Return the cache path for the PE-style SOI targets table."""

    resolved_cache_dir = DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir)
    return resolved_cache_dir / _soi_targets_cache_filename(revision)


def download_pe_soi_targets(
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
    revision: str = SOI_TARGETS_POLICYENGINE_US_DATA_REF,
    url: str = SOI_TARGETS_URL,
) -> Path:
    """Resolve the PE-style SOI targets table into the microplex cache."""

    resolved_cache_dir = DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    destination = pe_soi_targets_cache_path(resolved_cache_dir, revision=revision)
    if destination.exists() and not force:
        return validate_pe_soi_targets_file(destination)

    response = requests.get(url, timeout=300)
    response.raise_for_status()
    destination.write_bytes(response.content)
    try:
        return validate_pe_soi_targets_file(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
