"""Dataset-assembly artifact helpers for saved US Microplex bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from microplex_us.capital_gains_lots import (
    SyntheticCapitalGainsLotConfig,
    generate_synthetic_capital_gains_lots,
    synthetic_capital_gains_lot_metadata,
    validate_capital_gains_lot_anchors,
    write_capital_gains_lots_sqlite,
)
from microplex_us.pipelines.stage_contracts import (
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.us import USMicroplexBuildResult


def _maybe_write_capital_gains_lot_artifact(
    result: USMicroplexBuildResult,
    output_dir: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    if (
        not result.config.capital_gains_lots_enabled
        or result.policyengine_tables is None
    ):
        return None, None
    persons = result.policyengine_tables.persons
    gain_column = "long_term_capital_gains_before_response"
    if gain_column not in persons.columns:
        return None, {
            "enabled": True,
            "written": False,
            "reason": f"missing {gain_column}",
        }

    period = result.config.policyengine_dataset_year or 2024
    lot_config = SyntheticCapitalGainsLotConfig(
        random_seed=(
            result.config.capital_gains_lots_random_seed
            if result.config.capital_gains_lots_random_seed is not None
            else result.config.random_seed
        ),
        max_lots_per_person=int(result.config.capital_gains_lots_max_lots_per_person),
    )
    lots = generate_synthetic_capital_gains_lots(
        persons,
        period=period,
        config=lot_config,
        gain_column=gain_column,
    )
    validate_capital_gains_lot_anchors(persons, lots, gain_column=gain_column)
    metadata = synthetic_capital_gains_lot_metadata(
        lot_config,
        period=period,
        source_gain_column=gain_column,
    )
    nonzero_people = int(
        pd.to_numeric(persons[gain_column], errors="coerce").fillna(0.0).ne(0.0).sum()
    )
    metadata.update(
        {
            "person_rows": int(len(persons)),
            "nonzero_person_rows": nonzero_people,
            "lot_rows": int(len(lots)),
        }
    )
    path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "08_dataset_assembly",
        "capital_gains_lots",
    )
    write_capital_gains_lots_sqlite(lots, path, metadata=metadata)
    return path, {
        "enabled": True,
        "written": True,
        "path": path.name,
        "person_rows": int(len(persons)),
        "nonzero_person_rows": nonzero_people,
        "lot_rows": int(len(lots)),
        "source_gain_column": gain_column,
        "max_lots_per_person": int(lot_config.max_lots_per_person),
    }
