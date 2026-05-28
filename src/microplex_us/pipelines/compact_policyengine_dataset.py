"""Compact PolicyEngine time-period H5 datasets by household weight."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ENTITY_ID_VARIABLES = {
    "household": "household_id",
    "person": "person_id",
    "tax_unit": "tax_unit_id",
    "spm_unit": "spm_unit_id",
    "family": "family_id",
    "marital_unit": "marital_unit_id",
}

PERSON_ENTITY_LINK_VARIABLES = {
    "household": "person_household_id",
    "tax_unit": "person_tax_unit_id",
    "spm_unit": "person_spm_unit_id",
    "family": "person_family_id",
    "marital_unit": "person_marital_unit_id",
}

STRUCTURAL_VARIABLE_ENTITIES = {
    "household_id": "household",
    "household_weight": "household",
    "person_id": "person",
    "person_household_id": "person",
    "person_tax_unit_id": "person",
    "person_spm_unit_id": "person",
    "person_family_id": "person",
    "person_marital_unit_id": "person",
    "person_weight": "person",
    "tax_unit_id": "tax_unit",
    "spm_unit_id": "spm_unit",
    "family_id": "family",
    "marital_unit_id": "marital_unit",
}


def compact_policyengine_dataset_by_household_weight(
    *,
    input_dataset_path: str | Path,
    output_dataset_path: str | Path,
    households: int,
    period: int = 2024,
    weights_path: str | Path | None = None,
    rescale_to_total: bool = True,
    target_total_weight: float | None = None,
) -> dict[str, Any]:
    """Write a household-subset PE H5, keeping the largest household weights."""

    input_path = Path(input_dataset_path).expanduser()
    output_path = Path(output_dataset_path).expanduser()
    if households <= 0:
        raise ValueError("households must be positive")

    period_key = str(period)
    with h5py.File(input_path, "r") as source:
        household_ids = _period_array(source, "household_id", period_key)
        source_household_weights = np.asarray(
            _period_array(source, "household_weight", period_key),
            dtype=np.float64,
        )
        if household_ids.shape[0] != source_household_weights.shape[0]:
            raise ValueError("household_id and household_weight lengths differ")

        selection_weights = (
            np.load(Path(weights_path).expanduser()).astype(np.float64)
            if weights_path is not None
            else source_household_weights
        )
        if selection_weights.ndim != 1:
            raise ValueError("selection weights must be a one-dimensional array")
        if selection_weights.shape[0] != household_ids.shape[0]:
            raise ValueError(
                "selection weights length does not match household_id length: "
                f"{selection_weights.shape[0]} vs {household_ids.shape[0]}"
            )
        if households > household_ids.shape[0]:
            raise ValueError(
                "households cannot exceed source household count: "
                f"{households} > {household_ids.shape[0]}"
            )

        selected_by_weight = np.argsort(-selection_weights, kind="stable")[:households]
        selected_source_order = np.sort(selected_by_weight)
        selected_household_ids = household_ids[selected_source_order]
        selected_weights = source_household_weights[selected_source_order].astype(
            np.float64,
            copy=True,
        )
        original_selected_weight_sum = float(selected_weights.sum())
        resolved_target_total = (
            float(target_total_weight)
            if target_total_weight is not None
            else float(source_household_weights.sum())
        )
        if rescale_to_total:
            if original_selected_weight_sum <= 0:
                raise ValueError("selected household weights sum to zero")
            selected_weights *= resolved_target_total / original_selected_weight_sum

        metadata = _build_metadata(source, period_key)
        masks = _build_entity_masks(metadata, selected_household_ids)
        _write_compacted_dataset(
            source,
            output_path,
            period_key=period_key,
            metadata=metadata,
            masks=masks,
        )

    with h5py.File(output_path, "r+") as output:
        weight_dataset = output["household_weight"][period_key]
        weight_dataset[...] = selected_weights.astype(weight_dataset.dtype)
        entity_counts = {
            entity: int(len(output[variable][period_key]))
            for entity, variable in ENTITY_ID_VARIABLES.items()
            if variable in output and period_key in output[variable]
        }
        output_weight_sum = float(
            np.asarray(output["household_weight"][period_key], dtype=np.float64).sum()
        )

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "input_dataset": str(input_path.resolve()),
        "output_dataset": str(output_path.resolve()),
        "period": int(period),
        "selection_method": "largest_household_weight",
        "source_households": int(household_ids.shape[0]),
        "selected_households": int(households),
        "source_weight_sum": float(source_household_weights.sum()),
        "selected_weight_sum_before_rescale": original_selected_weight_sum,
        "output_weight_sum": output_weight_sum,
        "target_total_weight": resolved_target_total if rescale_to_total else None,
        "rescale_to_total": bool(rescale_to_total),
        "selection_weight_min_kept": float(selection_weights[selected_by_weight[-1]]),
        "selection_weight_max_kept": float(selection_weights[selected_by_weight[0]]),
        "entity_counts": entity_counts,
        "source_size_bytes": int(input_path.stat().st_size),
        "output_size_bytes": int(output_path.stat().st_size),
        "source_size_ratio": float(
            output_path.stat().st_size / input_path.stat().st_size
        ),
    }
    return summary


def _period_array(source: h5py.File, variable: str, period_key: str) -> np.ndarray:
    if variable not in source or period_key not in source[variable]:
        raise ValueError(f"{source.filename} is missing {variable}/{period_key}")
    return np.asarray(source[variable][period_key])


def _copy_attrs(
    source: h5py.Group | h5py.Dataset, destination: h5py.Group | h5py.Dataset
) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _build_metadata(source: h5py.File, period_key: str) -> dict[str, Any]:
    entity_ids = {
        entity: _period_array(source, variable, period_key)
        for entity, variable in ENTITY_ID_VARIABLES.items()
        if variable in source and period_key in source[variable]
    }
    person_links = {
        entity: _period_array(source, variable, period_key)
        for entity, variable in PERSON_ENTITY_LINK_VARIABLES.items()
        if variable in source and period_key in source[variable]
    }
    if "household" not in entity_ids or "person" not in entity_ids:
        raise ValueError("input dataset must include household_id and person_id")
    if "household" not in person_links:
        raise ValueError("input dataset must include person_household_id")

    entity_lengths = {entity: int(len(values)) for entity, values in entity_ids.items()}
    length_entities: dict[int, list[str]] = {}
    for entity, length in entity_lengths.items():
        length_entities.setdefault(length, []).append(entity)

    policyengine_variable_entities = _load_policyengine_variable_entities()
    variable_entities: dict[str, str] = {}
    for variable in source.keys():
        if period_key not in source[variable]:
            continue
        dataset = source[variable][period_key]
        entity = _infer_variable_entity(
            variable,
            int(len(dataset)) if dataset.shape else 0,
            entity_lengths=entity_lengths,
            length_entities=length_entities,
            policyengine_variable_entities=policyengine_variable_entities,
        )
        variable_entities[variable] = entity

    return {
        "entity_ids": entity_ids,
        "person_links": person_links,
        "variable_entities": variable_entities,
    }


def _infer_variable_entity(
    variable: str,
    array_length: int,
    *,
    entity_lengths: dict[str, int],
    length_entities: dict[int, list[str]],
    policyengine_variable_entities: dict[str, str],
) -> str:
    structural_entity = STRUCTURAL_VARIABLE_ENTITIES.get(variable)
    if structural_entity is not None:
        return structural_entity

    policyengine_entity = policyengine_variable_entities.get(variable)
    if policyengine_entity in entity_lengths:
        return policyengine_entity

    matching_entities = length_entities.get(array_length, [])
    if len(matching_entities) == 1:
        return matching_entities[0]

    raise ValueError(
        f"Could not infer entity for {variable!r} with length {array_length}; "
        f"matches={matching_entities}"
    )


def _load_policyengine_variable_entities() -> dict[str, str]:
    try:
        from policyengine_us import Microsimulation  # noqa: PLC0415
    except Exception:
        return {}
    try:
        variables = Microsimulation().tax_benefit_system.variables
    except Exception:
        return {}
    return {name: str(definition.entity.key) for name, definition in variables.items()}


def _build_entity_masks(
    metadata: dict[str, Any],
    selected_household_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    household_mask = np.isin(
        metadata["entity_ids"]["household"],
        selected_household_ids,
    )
    person_mask = np.isin(
        metadata["person_links"]["household"],
        selected_household_ids,
    )
    masks = {"household": household_mask, "person": person_mask}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        if entity not in metadata["entity_ids"]:
            continue
        if entity not in metadata["person_links"]:
            raise ValueError(
                f"input dataset includes {ENTITY_ID_VARIABLES[entity]} but lacks "
                f"{PERSON_ENTITY_LINK_VARIABLES[entity]}"
            )
        selected_entity_ids = np.unique(metadata["person_links"][entity][person_mask])
        masks[entity] = np.isin(metadata["entity_ids"][entity], selected_entity_ids)
    return masks


def _write_compacted_dataset(
    source: h5py.File,
    output_path: Path,
    *,
    period_key: str,
    metadata: dict[str, Any],
    masks: dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as output:
        _copy_attrs(source, output)
        for variable in source.keys():
            if period_key not in source[variable]:
                continue
            entity = metadata["variable_entities"][variable]
            group = output.create_group(variable)
            _copy_attrs(source[variable], group)
            for source_period_key in source[variable].keys():
                dataset = source[variable][source_period_key]
                values = np.asarray(dataset)
                if values.shape:
                    values = values[masks[entity]]
                output_dataset = group.create_dataset(source_period_key, data=values)
                _copy_attrs(dataset, output_dataset)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compact a PolicyEngine US H5 by keeping top household weights."
    )
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--households", type=int, required=True)
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--weights-npy")
    parser.add_argument("--target-total-weight", type=float)
    parser.add_argument("--no-rescale", action="store_true")
    parser.add_argument("--summary-json")
    args = parser.parse_args(argv)

    summary = compact_policyengine_dataset_by_household_weight(
        input_dataset_path=args.input_dataset,
        output_dataset_path=args.output_dataset,
        households=args.households,
        period=args.period,
        weights_path=args.weights_npy,
        rescale_to_total=not args.no_rescale,
        target_total_weight=args.target_total_weight,
    )
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(summary_path)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
