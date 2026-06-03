"""Source-plan and source-weight diagnostics for saved artifact bundles."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from microplex_us.data_sources.forbes import ForbesFixedSpineConfig
from microplex_us.pipelines.artifact_io import _write_json_atomically
from microplex_us.pipelines.us import USMicroplexBuildResult


def _write_us_source_plan_artifact(
    result: USMicroplexBuildResult,
    output_path: Path,
) -> None:
    synthesis = dict(result.synthesis_metadata)
    source_names = tuple(
        dict.fromkeys(
            value
            for value in (
                *list(synthesis.get("source_names", ())),
                synthesis.get("scaffold_source"),
            )
            if isinstance(value, str) and value
        )
    )
    payload = {
        "formatVersion": 1,
        "stageId": "03_source_planning",
        "sourceNames": list(source_names),
        "scaffoldSource": synthesis.get("scaffold_source"),
        "donorIntegratedVariables": list(
            synthesis.get("donor_integrated_variables", ())
        ),
        "conditionVars": list(synthesis.get("condition_vars", ())),
        "targetVars": list(synthesis.get("target_vars", ())),
        "donorAuthoritativeOverrideVariables": list(
            synthesis.get("donor_authoritative_override_variables", ())
        ),
        "donorExcludedVariables": list(synthesis.get("donor_excluded_variables", ())),
    }
    if result.fusion_plan is not None:
        payload["fusionPlan"] = {
            "sourceNames": list(result.fusion_plan.source_names),
        }
    _write_json_atomically(output_path, payload)


def _build_source_weight_diagnostics(
    result: USMicroplexBuildResult,
) -> dict[str, Any]:
    """Summarize source-weight provenance without exporting diagnostics to H5."""

    entity_summaries = _entity_weight_summaries(result)
    household_summary = entity_summaries["households"]
    total_household_weight = household_summary["weight_sum"]
    source_names = _source_names_for_diagnostics(result)
    scaffold_source = _scaffold_source_for_diagnostics(result)
    donor_sources = [
        source_name
        for source_name in source_names
        if scaffold_source is None or source_name != scaffold_source
    ]
    sources: list[dict[str, Any]] = []

    fixed_spine_entry = _fixed_spine_source_entry(
        result,
        total_entity_summaries=entity_summaries,
    )
    fixed_entity_summaries = (
        {
            entity: {
                "count": fixed_spine_entry.get(f"{prefix}_count", 0),
                "weight_sum": fixed_spine_entry.get(f"{prefix}_weight_sum", 0.0),
                "available": fixed_spine_entry.get(f"{prefix}_weight_sum") is not None,
            }
            for entity, prefix in _SOURCE_DIAGNOSTIC_ENTITY_PREFIXES.items()
        }
        if fixed_spine_entry is not None
        else {}
    )
    ordinary_entity_summaries = _subtract_entity_summaries(
        entity_summaries,
        fixed_entity_summaries,
    )

    sources.append(
        {
            "source_name": scaffold_source or "microplex_synthetic_population",
            "source_class": "synthetic_population",
            "source_role": "scaffold",
            "source_names": source_names,
            **_source_entity_fields(ordinary_entity_summaries, entity_summaries),
        }
    )

    donor_integrated_variables = list(
        result.synthesis_metadata.get("donor_integrated_variables", ())
    )
    for source_name in donor_sources:
        sources.append(
            {
                "source_name": source_name,
                "source_class": "donor_imputation",
                "source_role": "donor",
                "integrated_variable_count": len(donor_integrated_variables),
                "row_contribution": "variables_imputed_into_synthetic_rows",
                **_source_entity_fields(
                    _zero_entity_summaries(),
                    entity_summaries,
                ),
            }
        )

    if fixed_spine_entry is not None:
        sources.append(fixed_spine_entry)

    numeric_shares = [
        float(source["household_weight_share"])
        for source in sources
        if isinstance(source.get("household_weight_share"), int | float)
    ]
    summary = {
        "diagnostic_scope": "saved_artifact_entity_weight_by_source_rows",
        "household_count": household_summary["count"],
        "total_household_weight": total_household_weight,
        "person_count": entity_summaries["persons"]["count"],
        "total_person_weight": entity_summaries["persons"]["weight_sum"],
        "tax_unit_count": entity_summaries["tax_units"]["count"],
        "total_tax_unit_weight": entity_summaries["tax_units"]["weight_sum"],
        "source_entry_count": len(sources),
        "donor_source_count": len(donor_sources),
        "donor_integrated_variable_count": len(donor_integrated_variables),
        "support_rows_appended": False,
        "donor_rows_appended": False,
        "support_household_weight_sum": 0.0,
        "support_household_weight_share": 0.0,
        "puf_support_household_weight_sum": 0.0,
        "puf_support_household_weight_share": 0.0,
        "max_source_household_weight_share": (
            max(numeric_shares) if numeric_shares else None
        ),
        "fixed_spine_enabled": bool(
            isinstance(result.calibration_summary.get("fixed_spine"), dict)
            and result.calibration_summary.get("fixed_spine", {}).get("enabled")
        ),
        "h5_exported": False,
    }

    return {
        "formatVersion": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "sources": sources,
        "notes": [
            "Donor sources contribute imputed variables to synthetic rows; they are not appended as weighted source rows.",
            "Source diagnostics are written as a sidecar and are intentionally not exported into PolicyEngine H5 variables.",
        ],
    }


_SOURCE_DIAGNOSTIC_ENTITY_PREFIXES = {
    "households": "household",
    "persons": "person",
    "tax_units": "tax_unit",
}


def _entity_weight_summaries(
    result: USMicroplexBuildResult,
) -> dict[str, dict[str, Any]]:
    summaries = _zero_entity_summaries()
    if result.policyengine_tables is not None:
        for entity in _SOURCE_DIAGNOSTIC_ENTITY_PREFIXES:
            frame, weights = _policyengine_entity_weights(result, entity)
            if frame is None or weights is None:
                continue
            summaries[entity] = {
                "count": int(len(frame)),
                "weight_sum": float(weights.sum()),
                "available": True,
            }
        return summaries

    frame = result.calibrated_data
    if frame.empty:
        return summaries
    weight_column = (
        "household_weight" if "household_weight" in frame.columns else "weight"
    )
    if weight_column not in frame.columns:
        summaries["persons"] = {
            "count": int(len(frame)),
            "weight_sum": 0.0,
            "available": False,
        }
        return summaries

    weights = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0.0)
    summaries["persons"] = {
        "count": int(len(frame)),
        "weight_sum": float(weights.sum()),
        "available": True,
    }
    if "household_id" in frame.columns:
        household_weights = weights.groupby(frame["household_id"], sort=False).first()
        summaries["households"] = {
            "count": int(len(household_weights)),
            "weight_sum": float(household_weights.sum()),
            "available": True,
        }
    return summaries


def _zero_entity_summaries() -> dict[str, dict[str, Any]]:
    return {
        entity: {"count": 0, "weight_sum": 0.0, "available": False}
        for entity in _SOURCE_DIAGNOSTIC_ENTITY_PREFIXES
    }


def _subtract_entity_summaries(
    total: dict[str, dict[str, Any]],
    subtract: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entity in _SOURCE_DIAGNOSTIC_ENTITY_PREFIXES:
        total_summary = total.get(entity, {})
        subtract_summary = subtract.get(entity, {})
        total_count = int(total_summary.get("count", 0) or 0)
        subtract_count = int(subtract_summary.get("count", 0) or 0)
        total_weight = float(total_summary.get("weight_sum", 0.0) or 0.0)
        subtract_weight = float(subtract_summary.get("weight_sum", 0.0) or 0.0)
        result[entity] = {
            "count": max(total_count - subtract_count, 0),
            "weight_sum": max(total_weight - subtract_weight, 0.0),
            "available": bool(total_summary.get("available", False)),
        }
    return result


def _source_entity_fields(
    source: dict[str, dict[str, Any]],
    total: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for entity, prefix in _SOURCE_DIAGNOSTIC_ENTITY_PREFIXES.items():
        source_summary = source.get(entity, {})
        total_summary = total.get(entity, {})
        source_weight = source_summary.get("weight_sum")
        fields[f"{prefix}_count"] = int(source_summary.get("count", 0) or 0)
        fields[f"{prefix}_weight_sum"] = (
            float(source_weight) if source_weight is not None else None
        )
        fields[f"{prefix}_weight_share"] = _weight_share(
            float(source_weight or 0.0),
            float(total_summary.get("weight_sum", 0.0) or 0.0),
        )
    return fields


def _policyengine_entity_weights(
    result: USMicroplexBuildResult,
    entity: str,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    tables = result.policyengine_tables
    if tables is None:
        return None, None
    households = tables.households
    if households is None or "household_weight" not in households.columns:
        household_weight_by_id = None
    else:
        household_weights = pd.to_numeric(
            households["household_weight"],
            errors="coerce",
        ).fillna(0.0)
        household_weight_by_id = pd.Series(
            household_weights.to_numpy(dtype=float),
            index=households["household_id"],
        )
    if entity == "households":
        if households is None or household_weight_by_id is None:
            return None, None
        return households, household_weights
    if entity == "persons":
        return _frame_and_entity_weights(
            tables.persons,
            direct_weight_columns=("weight", "person_weight", "household_weight"),
            household_weight_by_id=household_weight_by_id,
        )
    if entity == "tax_units":
        return _frame_and_entity_weights(
            tables.tax_units,
            direct_weight_columns=("tax_unit_weight", "household_weight"),
            household_weight_by_id=household_weight_by_id,
        )
    return None, None


def _frame_and_entity_weights(
    frame: pd.DataFrame | None,
    *,
    direct_weight_columns: tuple[str, ...],
    household_weight_by_id: pd.Series | None,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    if frame is None:
        return None, None
    for column in direct_weight_columns:
        if column in frame.columns:
            return (
                frame,
                pd.to_numeric(frame[column], errors="coerce").fillna(0.0),
            )
    if household_weight_by_id is not None and "household_id" in frame.columns:
        return (
            frame,
            frame["household_id"].map(household_weight_by_id).fillna(0.0),
        )
    return frame, pd.Series(0.0, index=frame.index, dtype=float)


def _source_names_for_diagnostics(result: USMicroplexBuildResult) -> list[str]:
    synthesis = dict(result.synthesis_metadata)
    names: list[str] = []
    if result.fusion_plan is not None:
        names.extend(str(name) for name in result.fusion_plan.source_names)
    names.extend(str(name) for name in synthesis.get("source_names", ()) if name)
    scaffold_source = synthesis.get("scaffold_source")
    if scaffold_source:
        names.append(str(scaffold_source))
    for frame in result.source_frames:
        source = getattr(frame, "source", None)
        source_name = getattr(source, "name", None)
        if source_name:
            names.append(str(source_name))
    return list(dict.fromkeys(names))


def _scaffold_source_for_diagnostics(result: USMicroplexBuildResult) -> str | None:
    scaffold_source = result.synthesis_metadata.get("scaffold_source")
    if scaffold_source:
        return str(scaffold_source)
    source_names = _source_names_for_diagnostics(result)
    return source_names[0] if source_names else None


def _fixed_spine_source_entry(
    result: USMicroplexBuildResult,
    *,
    total_entity_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    fixed_spine = result.calibration_summary.get("fixed_spine")
    if not isinstance(fixed_spine, dict) or not fixed_spine.get("enabled"):
        return None

    source_metadata = dict(fixed_spine.get("source_metadata", {}))
    entry: dict[str, Any] = {
        "source_name": source_metadata.get("source", "forbes_fixed_spine"),
        "source_class": "fixed_spine",
        "source_role": "post_calibration_append",
        "source_metadata": source_metadata,
    }
    fixed_spine_config = ForbesFixedSpineConfig()
    fixed_entity_summaries = _fixed_spine_entity_summaries(
        result,
        fixed_spine_config=fixed_spine_config,
    )
    entry.update(
        {
            **_source_entity_fields(
                fixed_entity_summaries,
                total_entity_summaries,
            ),
            "household_id_detection": {
                "method": "forbes_default_household_id_floor",
                "minimum_household_id": fixed_spine_config.household_id_start,
            },
        }
    )
    return entry


def _fixed_spine_entity_summaries(
    result: USMicroplexBuildResult,
    *,
    fixed_spine_config: ForbesFixedSpineConfig,
) -> dict[str, dict[str, Any]]:
    summaries = _zero_entity_summaries()
    id_floors = {
        "households": ("household_id", fixed_spine_config.household_id_start),
        "persons": ("person_id", fixed_spine_config.person_id_start),
        "tax_units": ("tax_unit_id", fixed_spine_config.tax_unit_id_start),
    }
    for entity, (id_column, id_floor) in id_floors.items():
        frame, weights = _policyengine_entity_weights(result, entity)
        if frame is None or weights is None or id_column not in frame.columns:
            continue
        ids = pd.to_numeric(frame[id_column], errors="coerce")
        fixed_mask = ids >= id_floor
        fixed_weights = weights.loc[fixed_mask]
        summaries[entity] = {
            "count": int(fixed_mask.sum()),
            "weight_sum": float(fixed_weights.sum()),
            "available": True,
        }
    return summaries


def _weight_share(value: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(value) / float(denominator)
