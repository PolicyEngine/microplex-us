"""Lightweight readiness audit for exported PolicyEngine-US datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from microplex_us.pipelines.stage_contracts import (
    resolve_us_stage_artifact_contract_path,
)

DEFAULT_PERIOD = 2024
DEFAULT_REQUIRED_VARIABLES: dict[str, str] = {
    "household_id": "household",
    "household_weight": "household",
    "person_id": "person",
    "person_household_id": "person",
    "tax_unit_id": "tax_unit",
    "person_tax_unit_id": "person",
    "spm_unit_id": "spm_unit",
    "person_spm_unit_id": "person",
    "state_fips": "household",
    "county_fips": "household",
    "congressional_district_geoid": "household",
    "spm_unit_spm_threshold": "spm_unit",
    "spm_unit_tenure_type": "spm_unit",
}
DEFAULT_EXPECTED_MATERIALIZED_VARIABLES = (
    "income_tax",
    "income_tax_positive",
    "eitc",
    "ctc",
    "refundable_ctc",
    "non_refundable_ctc",
    "snap",
    "ssi",
    "tanf",
    "medicaid",
    "aca_ptc",
)
DEFAULT_EXPECTED_SPINES = ("cps_asec", "acs_pums")


def build_policyengine_us_dataset_readiness_audit(
    path: str | Path,
    *,
    period: int | str = DEFAULT_PERIOD,
    expected_materialized_variables: tuple[str, ...] = DEFAULT_EXPECTED_MATERIALIZED_VARIABLES,
    required_variables: dict[str, str] | None = None,
    expected_spines: tuple[str, ...] = DEFAULT_EXPECTED_SPINES,
    minimum_nonmissing_share: float = 0.999,
) -> dict[str, Any]:
    """Inspect a saved artifact bundle or ``policyengine_us.h5`` export.

    This audit intentionally avoids running PolicyEngine. It only checks that the
    exported H5 has the structural, geography, SPM-threshold, and materialized
    policy-output arrays we expect before expensive native scoring starts.
    """

    input_path = Path(path).expanduser()
    artifact_dir = input_path if input_path.is_dir() else None
    dataset_path = _resolve_dataset_path(input_path)
    manifest = _load_optional_json(artifact_dir / "manifest.json") if artifact_dir else None
    source_spine_composition = (
        _load_optional_json(artifact_dir / "source_spine_composition.json")
        if artifact_dir
        else None
    )
    period_key = str(period)
    required = dict(required_variables or DEFAULT_REQUIRED_VARIABLES)
    expected_variables = tuple(dict.fromkeys(expected_materialized_variables))
    issues: list[dict[str, Any]] = []

    with h5py.File(dataset_path, "r") as handle:
        entity_counts = _entity_counts(handle, period_key)
        variable_summaries = {
            variable: _variable_summary(
                handle,
                variable,
                period_key=period_key,
                entity_counts=entity_counts,
                preferred_entity=required.get(variable),
            )
            for variable in sorted(set(required) | set(expected_variables))
        }

    for variable, expected_entity in required.items():
        summary = variable_summaries[variable]
        _append_variable_presence_issues(
            issues,
            variable=variable,
            summary=summary,
            expected_entity=expected_entity,
            minimum_nonmissing_share=minimum_nonmissing_share,
            required=True,
        )
    for variable in expected_variables:
        summary = variable_summaries[variable]
        _append_variable_presence_issues(
            issues,
            variable=variable,
            summary=summary,
            expected_entity=None,
            minimum_nonmissing_share=0.0,
            required=True,
        )

    _append_source_spine_issues(
        issues,
        source_spine_composition=source_spine_composition,
        expected_spines=expected_spines,
    )
    valid = not any(issue["severity"] == "error" for issue in issues)
    return {
        "schemaVersion": 1,
        "valid": valid,
        "inputPath": str(input_path),
        "artifactDir": str(artifact_dir) if artifact_dir is not None else None,
        "datasetPath": str(dataset_path),
        "period": int(period) if str(period).isdigit() else str(period),
        "entityCounts": entity_counts,
        "requiredVariables": required,
        "expectedMaterializedVariables": list(expected_variables),
        "variableSummaries": variable_summaries,
        "sourceSpineComposition": _source_spine_summary(source_spine_composition),
        "manifestSummary": _manifest_summary(manifest),
        "issues": issues,
    }


def write_policyengine_us_dataset_readiness_audit(
    path: str | Path,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Write a readiness audit JSON sidecar."""

    input_path = Path(path).expanduser()
    destination = (
        Path(output_path)
        if output_path is not None
        else (
            input_path / "policyengine_dataset_readiness.json"
            if input_path.is_dir()
            else input_path.with_name(f"{input_path.stem}_readiness.json")
        )
    )
    payload = build_policyengine_us_dataset_readiness_audit(input_path, **kwargs)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def _resolve_dataset_path(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset or artifact directory not found: {path}")
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        dataset_name = dict(manifest.get("artifacts", {})).get("policyengine_dataset")
        if isinstance(dataset_name, str) and dataset_name:
            dataset_path = Path(dataset_name)
            if not dataset_path.is_absolute():
                dataset_path = path / dataset_path
            if dataset_path.exists():
                return dataset_path.resolve()
    dataset_path = resolve_us_stage_artifact_contract_path(
        path,
        "08_dataset_assembly",
        "policyengine_dataset",
    )
    if dataset_path.exists():
        return dataset_path.resolve()
    raise FileNotFoundError(f"No policyengine_us.h5 export found under {path}")


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _entity_counts(handle: h5py.File, period_key: str) -> dict[str, int | None]:
    variables = {
        "household": "household_id",
        "person": "person_id",
        "tax_unit": "tax_unit_id",
        "spm_unit": "spm_unit_id",
    }
    return {
        entity: _dataset_length(handle, variable, period_key)
        for entity, variable in variables.items()
    }


def _dataset_length(
    handle: h5py.File,
    variable: str,
    period_key: str,
) -> int | None:
    if variable not in handle or period_key not in handle[variable]:
        return None
    return int(len(handle[variable][period_key]))


def _variable_summary(
    handle: h5py.File,
    variable: str,
    *,
    period_key: str,
    entity_counts: dict[str, int | None],
    preferred_entity: str | None = None,
) -> dict[str, Any]:
    if variable not in handle:
        return {"exists": False, "hasPeriod": False}
    group = handle[variable]
    if period_key not in group:
        return {
            "exists": True,
            "hasPeriod": False,
            "availablePeriods": sorted(str(key) for key in group.keys()),
        }
    values = np.asarray(group[period_key])
    length = int(values.shape[0]) if values.shape else 1
    profile = _array_profile(values)
    return {
        "exists": True,
        "hasPeriod": True,
        "length": length,
        "dtype": str(values.dtype),
        "entity": _infer_entity(
            length,
            entity_counts,
            preferred_entity=preferred_entity,
        ),
        **profile,
    }


def _array_profile(values: np.ndarray) -> dict[str, Any]:
    flat = np.ravel(values)
    if flat.dtype.kind in {"b", "i", "u", "f"}:
        numeric = flat.astype(float, copy=False)
        finite = np.isfinite(numeric)
        positive = finite & (numeric > 0.0)
        nonzero = finite & (numeric != 0.0)
        return {
            "finiteCount": int(finite.sum()),
            "finiteShare": _share(finite.sum(), len(flat)),
            "nonmissingCount": int(finite.sum()),
            "nonmissingShare": _share(finite.sum(), len(flat)),
            "positiveCount": int(positive.sum()),
            "positiveShare": _share(positive.sum(), len(flat)),
            "nonzeroCount": int(nonzero.sum()),
            "nonzeroShare": _share(nonzero.sum(), len(flat)),
        }
    decoded = _decode_string_array(flat)
    nonmissing = np.array(
        [bool(value.strip()) and value.strip().lower() != "nan" for value in decoded],
        dtype=bool,
    )
    return {
        "nonmissingCount": int(nonmissing.sum()),
        "nonmissingShare": _share(nonmissing.sum(), len(flat)),
    }


def _decode_string_array(values: np.ndarray) -> list[str]:
    result: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            result.append(value.decode("utf-8", errors="replace"))
        else:
            result.append(str(value))
    return result


def _share(numerator: int | np.integer, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _infer_entity(
    length: int,
    entity_counts: dict[str, int | None],
    *,
    preferred_entity: str | None = None,
) -> str | None:
    matches = [
        entity
        for entity, count in entity_counts.items()
        if count is not None and int(count) == length
    ]
    if preferred_entity is not None and preferred_entity in matches:
        return preferred_entity
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return "|".join(matches)
    return None


def _append_variable_presence_issues(
    issues: list[dict[str, Any]],
    *,
    variable: str,
    summary: dict[str, Any],
    expected_entity: str | None,
    minimum_nonmissing_share: float,
    required: bool,
) -> None:
    severity = "error" if required else "warning"
    if not summary.get("exists"):
        issues.append(
            {
                "severity": severity,
                "code": "missing_variable",
                "variable": variable,
                "message": f"Dataset is missing {variable!r}",
            }
        )
        return
    if not summary.get("hasPeriod"):
        issues.append(
            {
                "severity": severity,
                "code": "missing_period",
                "variable": variable,
                "message": f"Dataset variable {variable!r} is missing the requested period",
            }
        )
        return
    entity = summary.get("entity")
    if expected_entity is not None and entity != expected_entity:
        issues.append(
            {
                "severity": "error",
                "code": "entity_length_mismatch",
                "variable": variable,
                "expectedEntity": expected_entity,
                "observedEntity": entity,
                "message": (
                    f"Variable {variable!r} has length matching {entity!r}, "
                    f"expected {expected_entity!r}"
                ),
            }
        )
    nonmissing_share = summary.get("nonmissingShare")
    if (
        isinstance(nonmissing_share, int | float)
        and nonmissing_share < minimum_nonmissing_share
    ):
        issues.append(
            {
                "severity": "error" if required else "warning",
                "code": "low_nonmissing_share",
                "variable": variable,
                "nonmissingShare": float(nonmissing_share),
                "minimumNonmissingShare": minimum_nonmissing_share,
                "message": (
                    f"Variable {variable!r} nonmissing share "
                    f"{nonmissing_share:.4f} is below {minimum_nonmissing_share:.4f}"
                ),
            }
        )
    if variable == "spm_unit_spm_threshold":
        positive_share = summary.get("positiveShare")
        if isinstance(positive_share, int | float) and positive_share < 0.999:
            issues.append(
                {
                    "severity": "error",
                    "code": "low_positive_spm_threshold_share",
                    "variable": variable,
                    "positiveShare": float(positive_share),
                    "message": "SPM thresholds should be positive for nearly all SPM units",
                }
            )


def _append_source_spine_issues(
    issues: list[dict[str, Any]],
    *,
    source_spine_composition: dict[str, Any] | None,
    expected_spines: tuple[str, ...],
) -> None:
    if not expected_spines:
        return
    if source_spine_composition is None:
        issues.append(
            {
                "severity": "warning",
                "code": "missing_source_spine_composition",
                "message": "Artifact has no source_spine_composition.json sidecar",
            }
        )
        return
    observed = {
        str(group.get("spine"))
        for group in source_spine_composition.get("groups", ())
        if group.get("spine") is not None
    }
    missing = sorted(set(expected_spines) - observed)
    if missing:
        issues.append(
            {
                "severity": "error",
                "code": "missing_expected_spines",
                "missingSpines": missing,
                "observedSpines": sorted(observed),
                "message": "Source-spine composition is missing expected spines",
            }
        )


def _source_spine_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "householdCount": payload.get("household_count"),
        "nonzeroHouseholdCount": payload.get("nonzero_household_count"),
        "totalActiveWeight": payload.get("total_active_weight"),
        "effectiveSampleSize": payload.get("effective_sample_size"),
        "groups": [
            {
                "spine": group.get("spine"),
                "householdCount": group.get("household_count"),
                "nonzeroHouseholdCount": group.get("nonzero_household_count"),
                "totalActiveWeight": group.get("total_active_weight"),
                "totalSourceWeight": group.get("total_source_weight"),
            }
            for group in payload.get("groups", ())
        ],
    }


def _manifest_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    artifacts = dict(payload.get("artifacts", {}))
    return {
        "rows": payload.get("rows"),
        "weights": payload.get("weights"),
        "policyengineDataset": artifacts.get("policyengine_dataset"),
        "policyengineNativeScores": artifacts.get("policyengine_native_scores"),
        "sourceSpineComposition": artifacts.get("source_spine_composition"),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for dataset readiness audits."""

    parser = argparse.ArgumentParser(
        description="Audit a saved MicroPlex PE-US H5 export before native scoring.",
    )
    parser.add_argument("path", help="Artifact directory or policyengine_us.h5 path")
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--output-path")
    parser.add_argument(
        "--expected-materialized-variable",
        action="append",
        default=None,
        help="Calculated PolicyEngine variable expected in the H5. Repeatable.",
    )
    parser.add_argument(
        "--expected-spine",
        action="append",
        default=None,
        help="Source spine expected in source_spine_composition.json. Repeatable.",
    )
    args = parser.parse_args(argv)

    output = write_policyengine_us_dataset_readiness_audit(
        args.path,
        output_path=args.output_path,
        period=args.period,
        expected_materialized_variables=tuple(
            args.expected_materialized_variable
            if args.expected_materialized_variable is not None
            else DEFAULT_EXPECTED_MATERIALIZED_VARIABLES
        ),
        expected_spines=tuple(
            args.expected_spine
            if args.expected_spine is not None
            else DEFAULT_EXPECTED_SPINES
        ),
    )
    payload = json.loads(output.read_text())
    print(
        json.dumps(
            {
                "output": str(output),
                "valid": payload["valid"],
                "datasetPath": payload["datasetPath"],
                "entityCounts": payload["entityCounts"],
                "issueCount": len(payload["issues"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
