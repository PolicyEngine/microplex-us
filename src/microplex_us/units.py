"""SPM unit assignment helpers for US donor tables."""

from __future__ import annotations

from typing import Any

import pandas as pd
from microunit.units import assign_spm_partition


def attach_spm_unit_id(
    persons: pd.DataFrame,
    *,
    person_col: str = "person_id",
    household_col: str = "household_id",
    family_col: str = "family_id",
    output_col: str = "spm_unit_id",
) -> pd.DataFrame:
    """Ensure a person table has an SPM unit ID column."""

    result = persons.copy()
    if output_col in result.columns:
        return result

    partition = assign_spm_partition(
        result,
        person_col=person_col,
        household_col=household_col,
        family_col=family_col,
    )
    unit_by_person = dict(
        zip(
            partition.person_id.tolist(),
            partition.unit_id.tolist(),
            strict=True,
        )
    )
    result[output_col] = result[person_col].map(unit_by_person)
    _preserve_id_dtype(result, output_col, partition.unit_id)

    return result


def _preserve_id_dtype(
    frame: pd.DataFrame,
    column: str,
    source: pd.Series[Any],
) -> None:
    """Keep integer-like IDs integer after the person-id map roundtrip."""

    if pd.api.types.is_integer_dtype(source.dtype):
        frame[column] = frame[column].astype(source.dtype)
