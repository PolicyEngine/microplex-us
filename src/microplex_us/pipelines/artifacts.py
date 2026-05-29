"""Artifact persistence for production pipeline outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
from microplex.core import SourceProvider, SourceQuery
from microplex.targets import (
    TargetProvider,
    assert_valid_benchmark_artifact_manifest,
)

from microplex_us.capital_gains_lots import (
    SyntheticCapitalGainsLotConfig,
    generate_synthetic_capital_gains_lots,
    synthetic_capital_gains_lot_metadata,
    validate_capital_gains_lot_anchors,
    write_capital_gains_lots_sqlite,
)
from microplex_us.data_sources.forbes import ForbesFixedSpineConfig
from microplex_us.pipelines.index_db import (
    append_us_microplex_run_index_entry,
)
from microplex_us.pipelines.pe_native_scores import (
    compute_us_pe_native_scores,
)
from microplex_us.pipelines.registry import (
    FrontierMetric,
    append_us_microplex_run_registry_entry,
    build_us_microplex_run_registry_entry,
    load_us_microplex_run_registry,
    select_us_microplex_frontier_entry,
)
from microplex_us.pipelines.stage_contracts import (
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.stage_manifest import (
    write_us_policyengine_entity_stage_artifact,
)
from microplex_us.pipelines.stage_run import (
    USStageInputOverride,
    write_us_stage_run_manifests_from_artifact_manifest,
)
from microplex_us.pipelines.summarize_child_tax_unit_agi_drift import (
    DEFAULT_VARIABLES as DEFAULT_CHILD_TAX_UNIT_AGI_DRIFT_VARIABLES,
)
from microplex_us.pipelines.summarize_child_tax_unit_agi_drift import (
    summarize_child_tax_unit_agi_drift,
)
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexPipeline,
    USMicroplexTargets,
    build_us_microplex,
)
from microplex_us.policyengine.harness import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSHarnessSlice,
    default_policyengine_us_db_all_target_slices,
    default_policyengine_us_harness_slices,
    evaluate_policyengine_us_harness,
    filter_nonempty_policyengine_us_harness_slices,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSDBTargetProvider,
)


@dataclass(frozen=True)
class USMicroplexArtifactPaths:
    """Filesystem locations for persisted pipeline artifacts."""

    output_dir: Path
    seed_data: Path
    synthetic_data: Path
    calibrated_data: Path
    targets: Path
    manifest: Path
    version_id: str | None = None
    scaffold_seed_data: Path | None = None
    synthesizer: Path | None = None
    policyengine_dataset: Path | None = None
    data_flow_snapshot: Path | None = None
    stage_manifest: Path | None = None
    artifact_inventory: Path | None = None
    conditional_readiness: Path | None = None
    source_plan: Path | None = None
    policyengine_entity_tables: Path | None = None
    calibration_summary: Path | None = None
    validation_evidence: Path | None = None
    policyengine_harness: Path | None = None
    policyengine_native_scores: Path | None = None
    policyengine_native_audit: Path | None = None
    child_tax_unit_agi_drift: Path | None = None
    capital_gains_lots: Path | None = None
    source_weight_diagnostics: Path | None = None
    run_registry: Path | None = None
    run_index_db: Path | None = None


@dataclass(frozen=True)
class USMicroplexVersionedBuildArtifacts:
    """End-to-end build, save, and frontier-tracking result."""

    build_result: USMicroplexBuildResult
    artifact_paths: USMicroplexArtifactPaths
    current_entry: Any | None = None
    frontier_entry: Any | None = None
    frontier_delta: float | None = None


def replay_us_microplex_policyengine_stage_from_artifact(
    artifact_dir: str | Path,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> USMicroplexBuildResult:
    """Replay calibration/export inputs from a saved artifact without raw ETL.

    This reloads saved seed and synthetic rows, applies optional runtime config
    overrides, and reruns the downstream calibration stage from the saved
    synthetic population. For PE-DB builds, this intentionally calls
    ``calibrate_policyengine_tables`` even when ``calibration_backend="none"``
    so PE target materialization and export-only variables stay on the same
    path as a full pipeline build.
    """

    artifact_root = Path(artifact_dir).expanduser().resolve()
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Saved artifact manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    config_payload = dict(manifest.get("config", {}))
    config_payload.update(dict(config_overrides or {}))
    config = USMicroplexBuildConfig(**config_payload)

    seed_data = pd.read_parquet(
        _resolve_saved_artifact_file(artifact_root, manifest, "seed_data")
    )
    scaffold_seed_data_path = _resolve_optional_saved_artifact_file(
        artifact_root,
        manifest,
        "scaffold_seed_data",
    )
    scaffold_seed_data = (
        pd.read_parquet(scaffold_seed_data_path)
        if scaffold_seed_data_path is not None
        else None
    )
    synthetic_data = pd.read_parquet(
        _resolve_saved_artifact_file(artifact_root, manifest, "synthetic_data")
    )
    targets_payload = json.loads(
        _resolve_saved_artifact_file(artifact_root, manifest, "targets").read_text()
    )
    targets = USMicroplexTargets(
        marginal=dict(targets_payload.get("marginal", {})),
        continuous=dict(targets_payload.get("continuous", {})),
    )

    pipeline = USMicroplexPipeline(config)
    if config.policyengine_targets_db is not None:
        synthetic_tables = pipeline.build_policyengine_entity_tables(synthetic_data)
        policyengine_tables, calibrated_data, calibration_summary = (
            pipeline.calibrate_policyengine_tables(synthetic_tables)
        )
    else:
        calibrated_data, calibration_summary = pipeline.calibrate(
            synthetic_data,
            targets,
        )
        policyengine_tables = pipeline.build_policyengine_entity_tables(calibrated_data)

    synthesis_metadata = dict(manifest.get("synthesis", {}))
    synthesis_metadata["policyengine_stage_replay"] = {
        "source_artifact_dir": str(artifact_root),
        "source_manifest": str(manifest_path),
        "config_override_keys": sorted((config_overrides or {}).keys()),
    }

    return USMicroplexBuildResult(
        config=config,
        seed_data=seed_data,
        synthetic_data=synthetic_data,
        calibrated_data=calibrated_data,
        targets=targets,
        calibration_summary=calibration_summary,
        synthesis_metadata=synthesis_metadata,
        policyengine_tables=policyengine_tables,
        scaffold_seed_data=scaffold_seed_data,
    )


def replay_and_save_versioned_us_microplex_policyengine_stage(
    artifact_dir: str | Path,
    output_root: str | Path | None = None,
    *,
    config_overrides: dict[str, Any] | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = True,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Replay a saved artifact's policy stage and persist a new versioned bundle."""

    artifact_root = Path(artifact_dir).expanduser().resolve()
    build_result = replay_us_microplex_policyengine_stage_from_artifact(
        artifact_root,
        config_overrides=config_overrides,
    )
    resolved_output_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else artifact_root.parent
    )
    replay_metadata = {
        "policyengine_stage_replay": True,
        "source_artifact_dir": str(artifact_root),
        **dict(run_registry_metadata or {}),
    }
    return _finalize_versioned_build_artifacts(
        build_result,
        output_root=resolved_output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=replay_metadata,
    )


def _resolve_saved_artifact_file(
    artifact_root: Path,
    manifest: dict[str, Any],
    artifact_key: str,
) -> Path:
    artifacts = dict(manifest.get("artifacts", {}))
    filename = artifacts.get(artifact_key)
    if not filename:
        filename = "targets.json" if artifact_key == "targets" else f"{artifact_key}.parquet"
    path = Path(filename)
    if not path.is_absolute():
        path = artifact_root / path
    if not path.exists():
        raise FileNotFoundError(f"Saved artifact file not found: {path}")
    return path


def _resolve_optional_saved_artifact_file(
    artifact_root: Path,
    manifest: dict[str, Any],
    artifact_key: str,
) -> Path | None:
    artifacts = dict(manifest.get("artifacts", {}))
    filename = artifacts.get(artifact_key)
    if not filename:
        return None
    path = Path(str(filename))
    if not path.is_absolute():
        path = artifact_root / path
    if not path.exists():
        raise FileNotFoundError(f"Saved optional artifact file not found: {path}")
    return path


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
        "donorExcludedVariables": list(
            synthesis.get("donor_excluded_variables", ())
        ),
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
                "available": fixed_spine_entry.get(f"{prefix}_weight_sum")
                is not None,
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


def _summarize_child_tax_unit_agi_drift_ratios(
    payload: dict[str, Any],
    *,
    stage: str,
    variables: tuple[str, ...],
) -> dict[str, Any]:
    stages = dict(payload.get("stages", {}))
    stage_payload = dict(stages.get(stage, {}))
    subsets = dict(stage_payload.get("subsets", {}))
    adults = dict(subsets.get("adults", {}))
    dependents = dict(subsets.get("dependents_under_20", {}))
    ratios: dict[str, float | None] = {}
    for variable in variables:
        adult_sum = adults.get(variable, {}).get("sum")
        child_sum = dependents.get(variable, {}).get("sum")
        if adult_sum in (None, 0):
            ratios[variable] = None
        else:
            ratios[variable] = float(child_sum or 0.0) / float(adult_sum)
    return {
        "stage": stage,
        "dependents_under_20_sum_share": ratios,
    }


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


def save_us_microplex_artifacts(
    result: USMicroplexBuildResult,
    output_dir: str | Path,
    *,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
) -> USMicroplexArtifactPaths:
    """Persist a build result as a reproducible artifact bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_data_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "05_donor_integration_synthesis",
        "seed_data",
    )
    synthetic_data_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "05_donor_integration_synthesis",
        "synthetic_data",
    )
    calibrated_data_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "07_calibration",
        "calibrated_data",
    )
    targets_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "07_calibration",
        "targets",
    )
    manifest_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "01_run_profile",
        "manifest",
    )
    source_weight_diagnostics_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "05_donor_integration_synthesis",
        "source_weight_diagnostics",
    )
    synthesizer_path = (
        resolve_us_stage_artifact_contract_path(
            output_dir,
            "05_donor_integration_synthesis",
            "synthesizer",
        )
        if result.synthesizer
        else None
    )
    policyengine_dataset_path = (
        resolve_us_stage_artifact_contract_path(
            output_dir,
            "08_dataset_assembly",
            "policyengine_dataset",
        )
        if result.policyengine_tables is not None
        else None
    )
    data_flow_snapshot_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "08_dataset_assembly",
        "data_flow_snapshot",
    )
    stage_manifest_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "08_dataset_assembly",
        "stage_manifest",
    )
    artifact_inventory_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "08_dataset_assembly",
        "artifact_inventory",
    )
    conditional_readiness_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "08_dataset_assembly",
        "conditional_readiness",
    )
    source_plan_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "03_source_planning",
        "source_plan",
    )
    scaffold_seed_data_path = (
        resolve_us_stage_artifact_contract_path(
            output_dir,
            "04_seed_scaffold",
            "scaffold_seed_data",
        )
        if result.scaffold_seed_data is not None
        else None
    )
    policyengine_entity_tables_path = (
        resolve_us_stage_artifact_contract_path(
            output_dir,
            "06_policyengine_entities",
            "policyengine_entity_tables",
        )
        if result.policyengine_tables is not None
        else None
    )
    calibration_summary_path = resolve_us_stage_artifact_contract_path(
        output_dir,
        "07_calibration",
        "calibration_summary",
    )
    validation_evidence_path = (
        resolve_us_stage_artifact_contract_path(
            output_dir,
            "09_validation_benchmarking",
            "validation_evidence",
        )
        if result.policyengine_tables is not None
        else None
    )
    policyengine_harness_path = None
    policyengine_native_scores_path = None
    resolved_run_registry_path = None
    resolved_run_index_path = None
    harness_payload = None

    if result.scaffold_seed_data is not None and scaffold_seed_data_path is not None:
        scaffold_seed_data_path.parent.mkdir(parents=True, exist_ok=True)
        result.scaffold_seed_data.to_parquet(scaffold_seed_data_path, index=False)
    result.seed_data.to_parquet(seed_data_path, index=False)
    result.synthetic_data.to_parquet(synthetic_data_path, index=False)
    result.calibrated_data.to_parquet(calibrated_data_path, index=False)
    targets_path.write_text(
        json.dumps(
            {
                "marginal": result.targets.marginal,
                "continuous": result.targets.continuous,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if result.synthesizer is not None and synthesizer_path is not None:
        result.synthesizer.save(synthesizer_path)

    _write_us_source_plan_artifact(result, source_plan_path)
    _write_json_atomically(calibration_summary_path, result.calibration_summary)
    source_weight_diagnostics_payload = _build_source_weight_diagnostics(result)
    _write_json_atomically(
        source_weight_diagnostics_path,
        source_weight_diagnostics_payload,
    )

    if result.policyengine_tables is not None and policyengine_dataset_path is not None:
        if policyengine_entity_tables_path is not None:
            write_us_policyengine_entity_stage_artifact(
                result.policyengine_tables,
                output_dir,
            )
        period = result.config.policyengine_dataset_year or 2024
        USMicroplexPipeline(result.config).export_policyengine_dataset(
            result,
            policyengine_dataset_path,
            period=period,
        )
    capital_gains_lots_path, capital_gains_lots_summary = (
        _maybe_write_capital_gains_lot_artifact(result, output_dir)
    )

    (
        resolved_target_provider,
        resolved_baseline_dataset,
        resolved_harness_slices,
        resolved_harness_metadata,
    ) = _resolve_policyengine_harness_context(
        result,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
    )

    harness_summary = None
    native_scores_payload = (
        dict(precomputed_policyengine_native_scores)
        if precomputed_policyengine_native_scores is not None
        else None
    )
    if precomputed_policyengine_harness_payload is not None:
        harness_payload = dict(precomputed_policyengine_harness_payload)
        policyengine_harness_path = resolve_us_stage_artifact_contract_path(
            output_dir,
            "09_validation_benchmarking",
            "policyengine_harness",
        )
        policyengine_harness_path.write_text(
            json.dumps(harness_payload, indent=2, sort_keys=True)
        )
        harness_summary = harness_payload.get("summary")
    elif (
        not defer_policyengine_harness
        and result.policyengine_tables is not None
        and resolved_target_provider is not None
        and resolved_baseline_dataset is not None
        and resolved_harness_slices
    ):
        harness_period = result.config.policyengine_dataset_year or 2024
        harness_run = evaluate_policyengine_us_harness(
            result.policyengine_tables,
            resolved_target_provider,
            resolved_harness_slices,
            baseline_dataset=str(resolved_baseline_dataset),
            dataset_year=harness_period,
            simulation_cls=result.config.policyengine_simulation_cls,
            candidate_label="microplex",
            baseline_label="policyengine_us_data",
            metadata=resolved_harness_metadata,
            cache=policyengine_comparison_cache,
        )
        policyengine_harness_path = resolve_us_stage_artifact_contract_path(
            output_dir,
            "09_validation_benchmarking",
            "policyengine_harness",
        )
        harness_run.save(policyengine_harness_path)
        harness_payload = harness_run.to_dict()
        harness_summary = harness_payload["summary"]

    if native_scores_payload is not None:
        policyengine_native_scores_path = resolve_us_stage_artifact_contract_path(
            output_dir,
            "09_validation_benchmarking",
            "policyengine_native_scores",
        )
        policyengine_native_scores_path.write_text(
            json.dumps(native_scores_payload, indent=2, sort_keys=True)
        )
    elif (
        not defer_policyengine_native_score
        and policyengine_dataset_path is not None
        and resolved_baseline_dataset is not None
    ):
        try:
            native_scores_payload = compute_us_pe_native_scores(
                candidate_dataset_path=policyengine_dataset_path,
                baseline_dataset_path=resolved_baseline_dataset,
                period=result.config.policyengine_dataset_year or 2024,
                policyengine_us_data_repo=policyengine_us_data_repo,
            )
            policyengine_native_scores_path = resolve_us_stage_artifact_contract_path(
                output_dir,
                "09_validation_benchmarking",
                "policyengine_native_scores",
            )
            policyengine_native_scores_path.write_text(
                json.dumps(native_scores_payload, indent=2, sort_keys=True)
            )
        except Exception:
            if require_policyengine_native_score:
                raise

    child_tax_unit_agi_drift_path = None
    child_tax_unit_agi_drift_summary: dict[str, Any] | None = None
    if enable_child_tax_unit_agi_drift:
        try:
            drift_path = resolve_us_stage_artifact_contract_path(
                output_dir,
                "09_validation_benchmarking",
                "child_tax_unit_agi_drift",
            )
            variables = (
                child_tax_unit_agi_drift_variables
                or DEFAULT_CHILD_TAX_UNIT_AGI_DRIFT_VARIABLES
            )
            payload = summarize_child_tax_unit_agi_drift(
                output_dir,
                variables=variables,
            )
            drift_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True)
            )
            child_tax_unit_agi_drift_path = drift_path
            child_tax_unit_agi_drift_summary = _summarize_child_tax_unit_agi_drift_ratios(
                payload,
                stage="calibrated",
                variables=variables,
            )
        except Exception as exc:  # pragma: no cover - diagnostic best-effort
            child_tax_unit_agi_drift_summary = {
                "error": f"{type(exc).__name__}: {exc}",
            }

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": result.config.to_dict(),
        "rows": {
            "seed": int(len(result.seed_data)),
            "synthetic": int(len(result.synthetic_data)),
            "calibrated": int(len(result.calibrated_data)),
        },
        "weights": {
            "nonzero": result.n_nonzero_weights,
            "total": result.total_weighted_population,
        },
        "targets": {
            "n_marginal_groups": len(result.targets.marginal),
            "n_continuous": len(result.targets.continuous),
        },
        "synthesis": result.synthesis_metadata,
        "calibration": result.calibration_summary,
        "artifacts": {
            "seed_data": seed_data_path.name,
            "scaffold_seed_data": (
                str(scaffold_seed_data_path.relative_to(output_dir))
                if scaffold_seed_data_path is not None
                else None
            ),
            "synthetic_data": synthetic_data_path.name,
            "calibrated_data": calibrated_data_path.name,
            "targets": targets_path.name,
            "synthesizer": synthesizer_path.name if synthesizer_path else None,
            "source_plan": str(source_plan_path.relative_to(output_dir)),
            "source_weight_diagnostics": source_weight_diagnostics_path.name,
            "calibration_summary": str(
                calibration_summary_path.relative_to(output_dir)
            ),
            "policyengine_entity_tables": (
                str(policyengine_entity_tables_path.relative_to(output_dir))
                if policyengine_entity_tables_path is not None
                else None
            ),
            "policyengine_dataset": (
                policyengine_dataset_path.name if policyengine_dataset_path else None
            ),
            "data_flow_snapshot": data_flow_snapshot_path.name,
            "stage_manifest": stage_manifest_path.name,
            "artifact_inventory": str(
                artifact_inventory_path.relative_to(output_dir)
            ),
            "conditional_readiness": str(
                conditional_readiness_path.relative_to(output_dir)
            ),
            "validation_evidence": (
                str(validation_evidence_path.relative_to(output_dir))
                if validation_evidence_path is not None
                else None
            ),
            "policyengine_harness": (
                policyengine_harness_path.name if policyengine_harness_path else None
            ),
            "policyengine_native_scores": (
                policyengine_native_scores_path.name
                if policyengine_native_scores_path is not None
                else None
            ),
            "capital_gains_lots": (
                capital_gains_lots_path.name
                if capital_gains_lots_path is not None
                else None
            ),
        },
    }
    if harness_summary is not None:
        manifest["policyengine_harness"] = harness_summary
    if native_scores_payload is not None:
        manifest["policyengine_native_scores"] = dict(
            native_scores_payload.get("summary", {})
        )
    if child_tax_unit_agi_drift_path is not None:
        manifest["artifacts"]["child_tax_unit_agi_drift"] = (
            child_tax_unit_agi_drift_path.name
        )
    if child_tax_unit_agi_drift_summary is not None:
        manifest.setdefault("diagnostics", {})[
            "child_tax_unit_agi_drift"
        ] = child_tax_unit_agi_drift_summary
    if capital_gains_lots_summary is not None:
        manifest.setdefault("diagnostics", {})[
            "capital_gains_lots"
        ] = capital_gains_lots_summary
    manifest.setdefault("diagnostics", {})["source_weight_diagnostics"] = dict(
        source_weight_diagnostics_payload.get("summary", {})
    )
    if harness_summary is not None or native_scores_payload is not None:
        resolved_run_registry_path = Path(run_registry_path or output_dir.parent / "run_registry.jsonl")
        run_entry = build_us_microplex_run_registry_entry(
            artifact_dir=output_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            policyengine_harness_path=policyengine_harness_path,
            policyengine_harness_payload=harness_payload,
            metadata=dict(run_registry_metadata or {}),
        )
        recorded_entry = append_us_microplex_run_registry_entry(
            resolved_run_registry_path,
            run_entry,
        )
        resolved_run_index_path = append_us_microplex_run_index_entry(
            run_index_path or output_dir.parent,
            recorded_entry,
            policyengine_harness_payload=harness_payload,
        )
        manifest["run_registry"] = {
            "path": str(resolved_run_registry_path),
            "artifact_id": recorded_entry.artifact_id,
            "improved_candidate_frontier": recorded_entry.improved_candidate_frontier,
            "improved_delta_frontier": recorded_entry.improved_delta_frontier,
            "improved_composite_frontier": recorded_entry.improved_composite_frontier,
            "improved_native_frontier": recorded_entry.improved_native_frontier,
            "default_frontier_metric": (
                "enhanced_cps_native_loss_delta"
                if native_scores_payload is not None
                else "candidate_composite_parity_loss"
            ),
        }
        manifest["run_index"] = {
            "path": str(resolved_run_index_path),
            "artifact_id": recorded_entry.artifact_id,
        }
    manifest = write_us_stage_run_manifests_from_artifact_manifest(
        output_dir,
        manifest,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )
    assert_valid_benchmark_artifact_manifest(
        manifest,
        artifact_dir=output_dir,
        manifest_path=manifest_path,
        summary_section=(
            "policyengine_harness" if harness_summary is not None else None
        ),
        required_artifact_keys=(
            *(("scaffold_seed_data",) if scaffold_seed_data_path is not None else ()),
            "seed_data",
            "synthetic_data",
            "calibrated_data",
            "targets",
            "source_weight_diagnostics",
            *(
                ("policyengine_native_scores",)
                if native_scores_payload is not None
                else ()
            ),
        ),
        required_summary_keys=(
            (
                "candidate_mean_abs_relative_error",
                "baseline_mean_abs_relative_error",
                "mean_abs_relative_error_delta",
            )
            if harness_summary is not None
            else ()
        ),
    )

    return USMicroplexArtifactPaths(
        output_dir=output_dir,
        version_id=output_dir.name,
        seed_data=seed_data_path,
        synthetic_data=synthetic_data_path,
        calibrated_data=calibrated_data_path,
        targets=targets_path,
        manifest=manifest_path,
        scaffold_seed_data=scaffold_seed_data_path,
        synthesizer=synthesizer_path,
        policyengine_dataset=policyengine_dataset_path,
        data_flow_snapshot=data_flow_snapshot_path,
        stage_manifest=stage_manifest_path,
        artifact_inventory=artifact_inventory_path,
        conditional_readiness=conditional_readiness_path,
        source_plan=source_plan_path,
        policyengine_entity_tables=policyengine_entity_tables_path,
        calibration_summary=calibration_summary_path,
        validation_evidence=validation_evidence_path,
        policyengine_harness=policyengine_harness_path,
        policyengine_native_scores=policyengine_native_scores_path,
        policyengine_native_audit=None,
        child_tax_unit_agi_drift=child_tax_unit_agi_drift_path,
        capital_gains_lots=capital_gains_lots_path,
        source_weight_diagnostics=source_weight_diagnostics_path,
        run_registry=resolved_run_registry_path,
        run_index_db=resolved_run_index_path,
    )


def save_versioned_us_microplex_artifacts(
    result: USMicroplexBuildResult,
    output_root: str | Path,
    *,
    version_id: str | None = None,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
) -> USMicroplexArtifactPaths:
    """Persist a build under a stable versioned directory beneath one output root."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_version_id, output_dir = _allocate_versioned_output_dir(
        output_root,
        version_id=version_id,
        result=result,
    )
    paths = save_us_microplex_artifacts(
        result,
        output_dir,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path or output_root / "run_registry.jsonl",
        run_index_path=run_index_path or output_root,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )
    return replace(paths, version_id=resolved_version_id)


def build_and_save_versioned_us_microplex(
    persons: Any,
    households: Any,
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
) -> USMicroplexVersionedBuildArtifacts:
    """Build a US microplex dataset, save a versioned bundle, and report frontier gap."""
    build_result = build_us_microplex(persons, households, config=config)
    return save_versioned_us_microplex_build_result(
        build_result,
        output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )


def save_versioned_us_microplex_build_result(
    build_result: USMicroplexBuildResult,
    output_root: str | Path,
    *,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Save an already-built result as a versioned bundle and report frontier gap."""
    return _finalize_versioned_build_artifacts(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
    )


def build_and_save_versioned_us_microplex_from_source_provider(
    provider: SourceProvider,
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    query: SourceQuery | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Build from one source provider, save a versioned bundle, and report frontier gap."""
    pipeline = USMicroplexPipeline(config)
    build_result = pipeline.build_from_source_provider(provider, query=query)
    return _finalize_versioned_build_artifacts(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
    )


def build_and_save_versioned_us_microplex_from_source_providers(
    providers: list[SourceProvider],
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    queries: dict[str, SourceQuery] | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Build from multiple source providers, save a versioned bundle, and report frontier gap."""
    pipeline = USMicroplexPipeline(config)
    build_result = pipeline.build_from_source_providers(providers, queries=queries)
    return _finalize_versioned_build_artifacts(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
    )


def build_and_save_versioned_us_microplex_from_data_dir(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Build from a CPS-style parquet directory, save a versioned bundle, and report frontier gap."""
    pipeline = USMicroplexPipeline(config)
    build_result = pipeline.build_from_data_dir(data_dir)
    return _finalize_versioned_build_artifacts(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
    )


def _finalize_versioned_build_artifacts(
    build_result: USMicroplexBuildResult,
    *,
    output_root: str | Path,
    version_id: str | None,
    frontier_metric: FrontierMetric,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None,
    policyengine_target_provider: TargetProvider | None,
    policyengine_baseline_dataset: str | Path | None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ),
    policyengine_harness_metadata: dict[str, Any] | None,
    policyengine_us_data_repo: str | Path | None,
    defer_policyengine_harness: bool,
    require_policyengine_native_score: bool,
    defer_policyengine_native_score: bool,
    precomputed_policyengine_harness_payload: dict[str, Any] | None,
    precomputed_policyengine_native_scores: dict[str, Any] | None,
    run_registry_path: str | Path | None,
    run_index_path: str | Path | None,
    run_registry_metadata: dict[str, Any] | None,
    enable_child_tax_unit_agi_drift: bool = False,
    child_tax_unit_agi_drift_variables: tuple[str, ...] | None = None,
    allow_stage_input_overrides: bool = False,
    stage_input_overrides: tuple[USStageInputOverride, ...] = (),
) -> USMicroplexVersionedBuildArtifacts:
    artifact_paths = save_versioned_us_microplex_artifacts(
        build_result,
        output_root,
        version_id=version_id,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
        enable_child_tax_unit_agi_drift=enable_child_tax_unit_agi_drift,
        child_tax_unit_agi_drift_variables=child_tax_unit_agi_drift_variables,
        allow_stage_input_overrides=allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
    )
    current_entry = None
    frontier_entry = None
    frontier_delta = None
    if artifact_paths.run_registry is not None and artifact_paths.version_id is not None:
        registry_entries = load_us_microplex_run_registry(artifact_paths.run_registry)
        current_entry = next(
            (
                entry
                for entry in reversed(registry_entries)
                if entry.artifact_id == artifact_paths.version_id
            ),
            None,
        )
        frontier_entry = select_us_microplex_frontier_entry(
            artifact_paths.run_registry,
            metric=frontier_metric,
        )
        current_value = _registry_metric_value(current_entry, frontier_metric)
        frontier_value = _registry_metric_value(frontier_entry, frontier_metric)
        if current_value is not None and frontier_value is not None:
            frontier_delta = current_value - frontier_value
    return USMicroplexVersionedBuildArtifacts(
        build_result=build_result,
        artifact_paths=artifact_paths,
        current_entry=current_entry,
        frontier_entry=frontier_entry,
        frontier_delta=frontier_delta,
    )


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def _resolve_policyengine_harness_context(
    result: USMicroplexBuildResult,
    *,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None,
    policyengine_target_provider: TargetProvider | None,
    policyengine_baseline_dataset: str | Path | None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ),
    policyengine_harness_metadata: dict[str, Any] | None,
) -> tuple[
    TargetProvider | None,
    str | Path | None,
    tuple[PolicyEngineUSHarnessSlice, ...],
    dict[str, Any],
]:
    resolved_target_provider = policyengine_target_provider
    if resolved_target_provider is None and result.config.policyengine_targets_db is not None:
        resolved_target_provider = PolicyEngineUSDBTargetProvider(
            result.config.policyengine_targets_db
        )

    resolved_baseline_dataset = (
        policyengine_baseline_dataset or result.config.policyengine_baseline_dataset
    )

    harness_period = result.config.policyengine_dataset_year or 2024
    if policyengine_harness_slices is not None:
        resolved_harness_slices = tuple(policyengine_harness_slices)
    elif result.config.policyengine_targets_db is not None:
        resolved_harness_slices = default_policyengine_us_db_all_target_slices(
            period=harness_period,
            reform_id=result.config.policyengine_target_reform_id,
        )
    else:
        resolved_harness_slices = default_policyengine_us_harness_slices(
            period=harness_period
        )
    if resolved_target_provider is not None and resolved_harness_slices:
        resolved_harness_slices = filter_nonempty_policyengine_us_harness_slices(
            resolved_target_provider,
            resolved_harness_slices,
            cache=policyengine_comparison_cache,
        )

    resolved_harness_metadata = {
        "baseline_dataset": (
            Path(resolved_baseline_dataset).name
            if resolved_baseline_dataset is not None
            else None
        ),
        "targets_db": (
            Path(result.config.policyengine_targets_db).name
            if result.config.policyengine_targets_db is not None
            else None
        ),
        "target_period": result.config.policyengine_target_period,
        "target_variables": list(result.config.policyengine_target_variables),
        "target_domains": list(result.config.policyengine_target_domains),
        "target_geo_levels": list(result.config.policyengine_target_geo_levels),
        "target_profile": result.config.policyengine_target_profile,
        "calibration_target_profile": (
            result.config.policyengine_calibration_target_profile
        ),
        "target_reform_id": result.config.policyengine_target_reform_id,
        "harness_slice_names": [slice_spec.name for slice_spec in resolved_harness_slices],
        "policyengine_us_runtime_version": _resolve_policyengine_us_runtime_version(),
        "harness_suite": (
            "policyengine_us_all_targets"
            if result.config.policyengine_targets_db is not None
            and policyengine_harness_slices is None
            else None
        ),
        **dict(policyengine_harness_metadata or {}),
    }
    return (
        resolved_target_provider,
        resolved_baseline_dataset,
        resolved_harness_slices,
        resolved_harness_metadata,
    )


def _resolve_policyengine_us_runtime_version() -> str | None:
    try:
        return version("policyengine-us")
    except PackageNotFoundError:
        return None


def _allocate_versioned_output_dir(
    output_root: Path,
    *,
    version_id: str | None,
    result: USMicroplexBuildResult,
) -> tuple[str, Path]:
    if version_id is not None:
        output_dir = output_root / version_id
        if output_dir.exists():
            raise FileExistsError(f"Versioned artifact directory already exists: {output_dir}")
        return version_id, output_dir

    config_hash = _short_config_hash(result.config.to_dict())
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_version_id = f"{timestamp}-{config_hash}"
    candidate_version_id = base_version_id
    suffix = 2
    output_dir = output_root / candidate_version_id
    while output_dir.exists():
        candidate_version_id = f"{base_version_id}-{suffix}"
        output_dir = output_root / candidate_version_id
        suffix += 1
    return candidate_version_id, output_dir


def _short_config_hash(config: dict[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _registry_metric_value(entry: Any | None, metric: FrontierMetric) -> float | None:
    if entry is None:
        return None
    return getattr(entry, metric, None)
