"""Summarize the worst calibration-oracle target cells for one saved artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from microplex_us.pipelines.stage_contracts import (
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexPipeline,
    _policyengine_target_ledger_entry,
    _policyengine_target_loss_family_key,
    _policyengine_target_loss_geography_key,
)
from microplex_us.policyengine import (
    PolicyEngineUSDBTargetProvider,
    evaluate_policyengine_us_target_set,
    load_policyengine_us_entity_tables,
)


def summarize_us_policyengine_oracle_target_drilldown(
    artifact_dir: str | Path,
    *,
    family: str | None = None,
    geography: str | None = None,
    stage: str | None = None,
    top_k: int | None = 25,
) -> dict[str, Any]:
    """Evaluate one saved artifact against its oracle targets and list the worst cells."""

    bundle_dir = Path(artifact_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config_payload = dict(manifest.get("config", {}))
    config = USMicroplexBuildConfig(**config_payload)
    if config.policyengine_targets_db is None:
        raise ValueError("Artifact config does not define policyengine_targets_db")

    dataset_name = dict(manifest.get("artifacts", {})).get("policyengine_dataset")
    dataset_path = (
        _resolve_manifest_artifact_path(bundle_dir, str(dataset_name))
        if dataset_name is not None
        else resolve_us_stage_artifact_contract_path(
            bundle_dir,
            "08_dataset_assembly",
            "policyengine_dataset",
        )
    )
    if not dataset_path.exists():
        raise FileNotFoundError(f"PolicyEngine dataset not found: {dataset_path}")

    target_db_path = Path(config.policyengine_targets_db).expanduser()
    if not target_db_path.is_absolute():
        target_db_path = (bundle_dir / target_db_path).resolve()
    period = int(
        config.policyengine_target_period or config.policyengine_dataset_year or 2024
    )
    pipeline = USMicroplexPipeline(config)
    tables = load_policyengine_us_entity_tables(dataset_path, period=period)
    provider = PolicyEngineUSDBTargetProvider(target_db_path)
    (
        tables,
        _bindings,
        canonical_targets,
        _compiled_targets,
        _unsupported_targets,
        _compiled_constraints,
        _supported_targets,
        _constraints,
        _feasibility_filter_summary,
        calibration_materialized_variables,
        _materialization_failures,
        _fixed_spine_residualization_summary,
    ) = pipeline._resolve_policyengine_calibration_targets(
        tables,
        provider=provider,
        target_period=period,
    )

    simulation_cls = (
        config.policyengine_simulation_cls
        if isinstance(config.policyengine_simulation_cls, type)
        else None
    )
    report = evaluate_policyengine_us_target_set(
        tables,
        canonical_targets,
        period=period,
        dataset_year=config.policyengine_dataset_year or period,
        simulation_cls=simulation_cls,
        label=bundle_dir.name,
        strict_materialization=False,
        direct_override_variables=tuple(config.policyengine_direct_override_variables),
    )

    relative_error_cap = manifest.get("calibration", {}).get(
        "oracle_relative_error_cap",
        config.policyengine_oracle_relative_error_cap,
    )
    if relative_error_cap is not None:
        relative_error_cap = float(relative_error_cap)
    target_ledger = list(manifest.get("calibration", {}).get("target_ledger", ()))
    materialized_variables = {
        str(variable)
        for variable in manifest.get("calibration", {}).get(
            "materialized_variables", ()
        )
    }
    materialized_variables.update(
        str(variable) for variable in calibration_materialized_variables
    )
    materialized_variables.update(
        str(variable) for variable in report.materialized_variables
    )
    ledger_by_name = {
        str(entry["target_name"]): dict(entry)
        for entry in target_ledger
        if entry.get("target_name") is not None
    }

    rows: list[dict[str, Any]] = []
    for evaluation in report.evaluations:
        rows.append(
            _oracle_target_row(
                target=evaluation.target,
                ledger_entry=ledger_by_name.get(evaluation.target.name),
                actual_value=float(evaluation.actual_value),
                relative_error=evaluation.relative_error,
                relative_error_cap=relative_error_cap,
                unsupported=False,
                household_count=len(tables.households),
                materialized_variables=materialized_variables,
            )
        )
    unsupported_penalty = relative_error_cap if relative_error_cap is not None else 1.0
    for target in report.unsupported_targets:
        rows.append(
            _oracle_target_row(
                target=target,
                ledger_entry=ledger_by_name.get(target.name),
                actual_value=None,
                relative_error=None,
                relative_error_cap=relative_error_cap,
                unsupported=True,
                unsupported_penalty=unsupported_penalty,
                household_count=len(tables.households),
                materialized_variables=materialized_variables,
            )
        )

    filtered_rows = [
        row
        for row in rows
        if (family is None or row["loss_family"] == family)
        and (geography is None or row["loss_geography"] == geography)
        and (stage is None or row["stage"] == stage)
    ]
    filtered_rows.sort(
        key=lambda row: (
            -float(row.get("capped_abs_relative_error") or -1.0),
            -float(row.get("abs_relative_error") or -1.0),
            row["target_name"],
        )
    )

    return {
        "artifactDir": str(bundle_dir),
        "datasetPath": str(dataset_path),
        "targetDbPath": str(target_db_path),
        "period": period,
        "filters": {
            "family": family,
            "geography": geography,
            "stage": stage,
            "topK": int(top_k) if top_k is not None else None,
        },
        "summary": {
            "targetCount": len(filtered_rows),
            "supportedTargetCount": sum(
                1 for row in filtered_rows if not row["unsupported"]
            ),
            "unsupportedTargetCount": sum(
                1 for row in filtered_rows if row["unsupported"]
            ),
            "stageCounts": {
                key: int(value)
                for key, value in sorted(
                    Counter(str(row["stage"]) for row in filtered_rows).items()
                )
            },
            "largestFamilies": _top_counts(filtered_rows, "loss_family", top_k=10),
            "largestGeographies": _top_counts(
                filtered_rows, "loss_geography", top_k=10
            ),
            "largestFamiliesByCappedError": _top_error_mass(
                filtered_rows,
                "loss_family",
                top_k=10,
            ),
            "largestGeographiesByCappedError": _top_error_mass(
                filtered_rows,
                "loss_geography",
                top_k=10,
            ),
        },
        "topRows": filtered_rows[:top_k] if top_k is not None else filtered_rows,
    }


def _resolve_manifest_artifact_path(bundle_dir: Path, artifact_name: str) -> Path:
    artifact_path = Path(artifact_name)
    if artifact_path.is_absolute():
        return artifact_path
    return (bundle_dir / artifact_path).resolve()


def _oracle_target_row(
    *,
    target: Any,
    ledger_entry: dict[str, Any] | None,
    actual_value: float | None,
    relative_error: float | None,
    relative_error_cap: float | None,
    unsupported: bool,
    household_count: int,
    materialized_variables: set[str],
    unsupported_penalty: float | None = None,
) -> dict[str, Any]:
    entry = (
        dict(ledger_entry)
        if ledger_entry is not None
        else _policyengine_target_ledger_entry(
            target=target,
            stage="unknown",
            reason="missing_manifest_ledger_entry",
            household_count=household_count,
        )
    )
    abs_relative_error = (
        abs(float(relative_error)) if relative_error is not None else None
    )
    capped_abs_relative_error = abs_relative_error
    if capped_abs_relative_error is not None and relative_error_cap is not None:
        capped_abs_relative_error = min(
            capped_abs_relative_error, float(relative_error_cap)
        )
    if unsupported:
        capped_abs_relative_error = (
            float(unsupported_penalty)
            if unsupported_penalty is not None
            else capped_abs_relative_error
        )
    variable = str(entry.get("variable") or "")
    domain_variable = entry.get("domain_variable")
    driver_variable = str(domain_variable or variable or "")
    driver_is_materialized = driver_variable in materialized_variables
    variable_is_materialized = variable in materialized_variables if variable else False
    domain_is_materialized = (
        str(domain_variable) in materialized_variables
        if domain_variable is not None
        else False
    )
    target_value = float(target.value)
    absolute_error = (
        abs(float(actual_value) - target_value) if actual_value is not None else None
    )
    return {
        "target_name": target.name,
        "stage": str(entry.get("stage") or "unknown"),
        "reason": str(entry.get("reason") or "unknown"),
        "loss_family": _policyengine_target_loss_family_key(entry),
        "loss_geography": _policyengine_target_loss_geography_key(entry),
        "family": str(entry.get("family") or ""),
        "variable": variable,
        "domain_variable": domain_variable,
        "driver_variable": driver_variable,
        "driver_is_materialized": driver_is_materialized,
        "variable_is_materialized": variable_is_materialized,
        "domain_is_materialized": domain_is_materialized,
        "provenance_class": (
            "policyengine_materialized" if driver_is_materialized else "stored_input"
        ),
        "geo_level": entry.get("geo_level"),
        "geographic_id": entry.get("geographic_id"),
        "unsupported": bool(unsupported),
        "target_value": target_value,
        "actual_value": float(actual_value) if actual_value is not None else None,
        "absolute_error": absolute_error,
        "relative_error": float(relative_error) if relative_error is not None else None,
        "abs_relative_error": abs_relative_error,
        "capped_abs_relative_error": capped_abs_relative_error,
        "active_households": entry.get("active_households"),
        "active_support_share": entry.get("active_support_share"),
        "filters": list(entry.get("filters") or ()),
    }


def _top_counts(
    rows: list[dict[str, Any]],
    key: str,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    counter = Counter(str(row[key]) for row in rows if row.get(key) is not None)
    return [
        {"group": group, "count": int(count)}
        for group, count in sorted(
            counter.items(), key=lambda item: (-int(item[1]), item[0])
        )[:top_k]
    ]


def _top_error_mass(
    rows: list[dict[str, Any]],
    key: str,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float | int]] = {}
    for row in rows:
        group = row.get(key)
        capped_error = row.get("capped_abs_relative_error")
        if group is None or capped_error is None:
            continue
        bucket = grouped.setdefault(
            str(group),
            {"cappedErrorMass": 0.0, "count": 0},
        )
        bucket["cappedErrorMass"] = float(bucket["cappedErrorMass"]) + float(
            capped_error
        )
        bucket["count"] = int(bucket["count"]) + 1
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            -float(item[1]["cappedErrorMass"]),
            -int(item[1]["count"]),
            item[0],
        ),
    )
    return [
        {
            "group": group,
            "cappedErrorMass": float(metrics["cappedErrorMass"]),
            "count": int(metrics["count"]),
            "meanCappedError": float(metrics["cappedErrorMass"])
            / int(metrics["count"]),
        }
        for group, metrics in ranked[:top_k]
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for one-artifact oracle drilldowns."""

    parser = argparse.ArgumentParser(
        description="Summarize the worst calibration-oracle target cells for one saved artifact.",
    )
    parser.add_argument("artifact_dir", help="Saved artifact bundle directory.")
    parser.add_argument("--family", help="Exact loss-family key to filter to.")
    parser.add_argument("--geography", help="Exact loss-geography key to filter to.")
    parser.add_argument("--stage", help="Exact target-ledger stage to filter to.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of rows to keep in the output.",
    )
    args = parser.parse_args(argv)

    payload = summarize_us_policyengine_oracle_target_drilldown(
        args.artifact_dir,
        family=args.family,
        geography=args.geography,
        stage=args.stage,
        top_k=args.top_k,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
