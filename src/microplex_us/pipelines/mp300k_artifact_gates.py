"""Persistent artifact quality gates for mp-300k candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

GateStatus = Literal["pass", "fail", "unmeasured"]

_ENTITY_ID_ARRAYS = {
    "household": "household_id",
    "person": "person_id",
    "tax_unit": "tax_unit_id",
    "spm_unit": "spm_unit_id",
    "family": "family_id",
    "marital_unit": "marital_unit_id",
}
_PERSON_LINK_ARRAYS = {
    "household": "person_household_id",
    "tax_unit": "person_tax_unit_id",
    "spm_unit": "person_spm_unit_id",
    "family": "person_family_id",
    "marital_unit": "person_marital_unit_id",
}
_REQUIRED_PERIOD_ARRAYS = (
    "household_id",
    "household_weight",
    "person_id",
    "person_household_id",
    "tax_unit_id",
    "person_tax_unit_id",
    "spm_unit_id",
    "person_spm_unit_id",
    "family_id",
    "person_family_id",
    "marital_unit_id",
    "person_marital_unit_id",
)
_DEFAULT_REQUIRED_GATES = (
    "candidate_artifact",
    "compatibility",
    "column_contract",
    "export_support",
    "export_lineage",
    "artifact_size",
    "runtime",
    "source_weight_diagnostics",
    "ecps_comparison",
    "arch_target_coverage",
    "benchmark_manifest",
)
_DEFAULT_ARCH_COVERAGE_PROFILE = "pe_native_broad_source_backed"
_PROTECTED_ECPS_TARGET_FAMILIES = (
    "ssi",
    "snap",
    "wages",
    "self_employment_income",
    "capital_gains",
    "interest",
    "dividends",
    "retirement_income",
    "disability",
    "household_net_income",
)
_CORE_BENCHMARK_ECPS_TARGET_FAMILIES = (
    "state_agi_distribution",
    "state_age_distribution",
    "national_ssa",
    "national_irs_other",
    "state_aca_spending",
)
_PROTECTED_FAMILY_RELATIVE_LOSS_TOLERANCE = 0.05
_PROTECTED_FAMILY_ABSOLUTE_LOSS_TOLERANCE = 0.005
_DEFAULT_MAX_SUPPORT_WEIGHT_SHARE = 0.25
_KNOWN_PEUS_COMPUTED_ECPS_CONTRACT_COLUMNS = frozenset(
    {
        "medicare_enrolled",
        "roth_401k_contributions",
        "roth_ira_contributions",
        "self_employed_health_insurance_ald",
        "self_employed_pension_contribution_ald",
        "self_employed_pension_contributions",
        "spm_unit_capped_work_childcare_expenses",
        "traditional_401k_contributions",
        "traditional_ira_contributions",
    }
)
_FORBIDDEN_SOURCE_DIAGNOSTIC_VARIABLES = frozenset(
    {
        "ssi_reported",
        "ssi_amount_reported",
        "ssdi_reported",
        "snap_reported",
        "tanf_reported",
        "wic_reported",
        "source_dataset",
        "source_survey",
        "donor_source",
        "imputation_source",
    }
)
_FORBIDDEN_SOURCE_DIAGNOSTIC_PREFIXES = (
    "diagnostic_",
    "source_dataset_",
    "source_survey_",
    "donor_source_",
    "imputation_source_",
)
_FORBIDDEN_SOURCE_DIAGNOSTIC_SUFFIXES = (
    "_diagnostic",
    "_diagnostics",
    "_source_dataset",
    "_source_survey",
    "_donor_source",
    "_imputation_source",
)
_REQUIRED_BENCHMARK_MANIFEST_EVIDENCE = {
    "baseline_dataset.path": (
        ("baseline_dataset", "path"),
        ("baseline_dataset_path",),
        ("enhanced_cps", "path"),
        ("enhanced_cps_path",),
    ),
    "baseline_dataset.sha256": (
        ("baseline_dataset", "sha256"),
        ("baseline_dataset_sha256",),
        ("enhanced_cps", "sha256"),
        ("enhanced_cps_sha256",),
    ),
    "policyengine_us_data.commit": (
        ("policyengine_us_data", "commit"),
        ("policyengine_us_data", "commit_sha"),
        ("policyengine_us_data_commit",),
        ("policyengine_us_data_commit_sha",),
        ("us_data_commit",),
    ),
    "policyengine_us.version": (
        ("policyengine_us", "version"),
        ("policyengine_us_version",),
    ),
    "target_db.path": (
        ("target_db", "path"),
        ("target_db_path",),
        ("targets_db", "path"),
        ("policyengine_targets_db",),
    ),
    "target_db.sha256": (
        ("target_db", "sha256"),
        ("target_db_sha256",),
        ("targets_db", "sha256"),
        ("policyengine_targets_db_sha256",),
    ),
}
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def build_mp300k_artifact_gate_report(
    artifact_dir: str | Path,
    *,
    candidate_dataset_path: str | Path | None = None,
    baseline_dataset_path: str | Path | None = None,
    ecps_comparison_payload: Any = None,
    arch_coverage_payload: dict[str, Any] | None = None,
    runtime_smoke_payload: dict[str, Any] | None = None,
    source_weight_diagnostics_payload: dict[str, Any] | None = None,
    source_weight_diagnostics_path: str | Path | None = None,
    benchmark_manifest_path: str | Path | None = None,
    period: int = 2024,
    arch_coverage_profile: str = _DEFAULT_ARCH_COVERAGE_PROFILE,
    artifact_size_ratio_threshold: float = 2.0,
    runtime_ratio_threshold: float = 1.25,
    max_support_weight_share: float = _DEFAULT_MAX_SUPPORT_WEIGHT_SHARE,
    compute_native_scores: bool = True,
    require_ecps_comparison: bool = True,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> dict[str, Any]:
    """Build a CI-friendly artifact gate report for one mp-300k candidate.

    The report is evidence-driven. It can consume precomputed runtime and
    eCPS-comparison payloads, or compute the PE-native broad score when a
    baseline dataset is available.
    """

    artifact_root = Path(artifact_dir).expanduser()
    manifest_path = artifact_root / "manifest.json"
    manifest = _load_manifest(manifest_path)
    candidate_dataset = _resolve_candidate_dataset_path(
        artifact_root,
        manifest,
        candidate_dataset_path,
    )
    baseline_dataset = (
        Path(baseline_dataset_path).expanduser()
        if baseline_dataset_path is not None
        else _manifest_baseline_dataset(artifact_root, manifest)
    )

    candidate_gate = _candidate_artifact_gate(
        manifest_path=manifest_path,
        candidate_dataset=candidate_dataset,
    )
    compatibility_gate = _compatibility_gate(candidate_dataset, period=period)
    column_contract_gate = _column_contract_gate(
        candidate_dataset,
        baseline_dataset=baseline_dataset,
        period=period,
    )
    export_support_gate = _export_support_gate(
        candidate_dataset,
        baseline_dataset=baseline_dataset,
        period=period,
    )
    export_lineage_gate = _export_lineage_gate(
        baseline_dataset=baseline_dataset,
        period=period,
    )
    artifact_size_gate = _artifact_size_gate(
        candidate_dataset,
        baseline_dataset=baseline_dataset,
        artifact_size_ratio_threshold=artifact_size_ratio_threshold,
    )
    resolved_ecps_comparison = _resolve_ecps_comparison_payload(
        ecps_comparison_payload,
        candidate_dataset=candidate_dataset,
        baseline_dataset=baseline_dataset,
        period=period,
        compute_native_scores=compute_native_scores,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )
    ecps_comparison_gate = _ecps_comparison_gate(resolved_ecps_comparison)
    arch_coverage_gate = _arch_target_coverage_gate(
        arch_coverage_payload,
        expected_period=period,
        expected_profile=arch_coverage_profile,
    )
    runtime_gate = _runtime_gate(
        runtime_smoke_payload,
        runtime_ratio_threshold=runtime_ratio_threshold,
    )
    resolved_source_weight_diagnostics = _resolve_source_weight_diagnostics_payload(
        artifact_root,
        manifest,
        source_weight_diagnostics_payload=source_weight_diagnostics_payload,
        source_weight_diagnostics_path=source_weight_diagnostics_path,
    )
    source_weight_diagnostics_gate = _source_weight_diagnostics_gate(
        resolved_source_weight_diagnostics,
        max_support_weight_share=max_support_weight_share,
    )
    benchmark_gate, benchmark_descriptor = _benchmark_manifest_gate(
        benchmark_manifest_path
    )
    gates = {
        "candidate_artifact": candidate_gate,
        "compatibility": compatibility_gate,
        "column_contract": column_contract_gate,
        "export_support": export_support_gate,
        "export_lineage": export_lineage_gate,
        "artifact_size": artifact_size_gate,
        "runtime": runtime_gate,
        "source_weight_diagnostics": source_weight_diagnostics_gate,
        "ecps_comparison": ecps_comparison_gate,
        "arch_target_coverage": arch_coverage_gate,
        "benchmark_manifest": benchmark_gate,
    }
    required_gates = _required_gate_names(
        require_ecps_comparison=require_ecps_comparison,
    )
    summary = _summarize_gates(gates, required_gates=required_gates)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "product": manifest.get("product") or "mp-300k",
        "gate_set": "artifact_ci",
        "artifact_id": artifact_root.name,
        "artifact_dir": str(artifact_root.resolve()),
        "period": int(period),
        "required_gates": required_gates,
        "summary": summary,
        "manifest": _file_descriptor(manifest_path),
        "candidate_dataset": _optional_file_descriptor(candidate_dataset),
        "baseline_dataset": (
            _optional_file_descriptor(baseline_dataset)
            if baseline_dataset is not None
            else None
        ),
        "gates": gates,
        "ecps_comparison_payload": resolved_ecps_comparison,
        "arch_coverage": arch_coverage_payload,
        "runtime_smoke": runtime_smoke_payload,
        "source_weight_diagnostics": resolved_source_weight_diagnostics,
        "benchmark_manifest": benchmark_descriptor,
    }


def write_mp300k_artifact_gate_report(
    artifact_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    update_manifest: bool = True,
    **kwargs: Any,
) -> Path:
    """Write ``mp300k_artifact_gates.json`` and reference it in manifest."""

    artifact_root = Path(artifact_dir).expanduser()
    report_path = (
        Path(output_path).expanduser()
        if output_path is not None
        else artifact_root / "mp300k_artifact_gates.json"
    )
    report = build_mp300k_artifact_gate_report(artifact_root, **kwargs)
    _write_json_atomically(report_path, report)
    if update_manifest:
        manifest_path = artifact_root / "manifest.json"
        manifest = _load_manifest(manifest_path)
        artifacts = dict(manifest.get("artifacts", {}))
        artifacts["mp300k_artifact_gates"] = _relative_or_absolute(
            report_path,
            base_dir=artifact_root,
        )
        manifest["artifacts"] = artifacts
        manifest["mp300k_artifact_gates"] = {
            "status": report["summary"]["status"],
            "passing_required_gate_count": report["summary"][
                "passing_required_gate_count"
            ],
            "failed_required_gate_count": report["summary"][
                "failed_required_gate_count"
            ],
            "unmeasured_required_gate_count": report["summary"][
                "unmeasured_required_gate_count"
            ],
        }
        _write_json_atomically(manifest_path, manifest)
    return report_path


def _resolve_candidate_dataset_path(
    artifact_root: Path,
    manifest: dict[str, Any],
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser()
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_name = artifacts.get("policyengine_dataset")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(
            "manifest.artifacts.policyengine_dataset is required when "
            "candidate_dataset_path is not supplied"
        )
    dataset_path = Path(dataset_name).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = artifact_root / dataset_path
    return dataset_path


def _manifest_baseline_dataset(
    artifact_root: Path, manifest: dict[str, Any]
) -> Path | None:
    config = dict(manifest.get("config", {}))
    value = config.get("policyengine_baseline_dataset")
    if value is None:
        return None
    baseline_path = Path(value).expanduser()
    if not baseline_path.is_absolute():
        baseline_path = artifact_root / baseline_path
    return baseline_path


def _candidate_artifact_gate(
    *,
    manifest_path: Path,
    candidate_dataset: Path,
) -> dict[str, Any]:
    missing = [
        str(path) for path in (manifest_path, candidate_dataset) if not path.exists()
    ]
    if missing:
        return _gate(
            "fail",
            "required candidate artifact files are missing",
            details={"missing": missing},
        )
    return _gate(
        "pass",
        "manifest and candidate H5 exist",
        metrics={
            "manifest_size_bytes": manifest_path.stat().st_size,
            "candidate_size_bytes": candidate_dataset.stat().st_size,
        },
    )


def _artifact_size_gate(
    candidate_dataset: Path,
    *,
    baseline_dataset: Path | None,
    artifact_size_ratio_threshold: float,
) -> dict[str, Any]:
    threshold = float(artifact_size_ratio_threshold)
    if baseline_dataset is None:
        return _gate(
            "unmeasured",
            "baseline H5 has not been attached for artifact-size comparison",
            metrics={"artifact_size_ratio_threshold": threshold},
        )
    if not candidate_dataset.exists() or not baseline_dataset.exists():
        missing = [
            str(path)
            for path in (candidate_dataset, baseline_dataset)
            if not path.exists()
        ]
        return _gate(
            "fail",
            "artifact-size comparison files are missing",
            details={"missing": missing},
            metrics={"artifact_size_ratio_threshold": threshold},
        )
    candidate_size = candidate_dataset.stat().st_size
    baseline_size = baseline_dataset.stat().st_size
    if baseline_size <= 0:
        return _gate(
            "fail",
            "baseline H5 size is nonpositive",
            metrics={
                "candidate_size_bytes": candidate_size,
                "baseline_size_bytes": baseline_size,
                "artifact_size_ratio_threshold": threshold,
            },
        )
    ratio = candidate_size / baseline_size
    return _gate(
        "pass" if ratio <= threshold else "fail",
        (
            "candidate H5 size is inside the artifact-size threshold"
            if ratio <= threshold
            else "candidate H5 size exceeds the artifact-size threshold"
        ),
        metrics={
            "candidate_size_bytes": candidate_size,
            "baseline_size_bytes": baseline_size,
            "artifact_size_ratio": ratio,
            "artifact_size_ratio_threshold": threshold,
        },
    )


def _column_contract_gate(
    candidate_dataset: Path,
    *,
    baseline_dataset: Path | None,
    period: int,
) -> dict[str, Any]:
    if baseline_dataset is None:
        return _gate(
            "unmeasured",
            "pinned eCPS baseline H5 has not been attached for column-contract comparison",
        )
    if not candidate_dataset.exists() or not baseline_dataset.exists():
        missing = [
            str(path)
            for path in (candidate_dataset, baseline_dataset)
            if not path.exists()
        ]
        return _gate(
            "fail",
            "column-contract comparison files are missing",
            details={"missing": missing},
        )

    period_key = str(int(period))
    candidate_columns = _h5_period_columns(candidate_dataset, period_key=period_key)
    baseline_columns = _h5_period_columns(baseline_dataset, period_key=period_key)
    excluded_baseline_computed_columns = sorted(
        _computed_policyengine_us_export_columns(baseline_columns)
    )
    contract_columns = sorted(
        set(baseline_columns) - set(excluded_baseline_computed_columns)
    )
    candidate_column_set = set(candidate_columns)
    contract_column_set = set(contract_columns)
    missing_contract_columns = sorted(contract_column_set - candidate_column_set)
    extra_candidate_columns = sorted(set(candidate_columns) - set(contract_columns))
    satisfied_count = len(contract_columns) - len(missing_contract_columns)
    contract_share = (
        float(satisfied_count / len(contract_columns)) if contract_columns else None
    )
    metrics = {
        "period": int(period),
        "baseline_column_count": len(baseline_columns),
        "candidate_column_count": len(candidate_columns),
        "excluded_baseline_computed_column_count": len(
            excluded_baseline_computed_columns
        ),
        "contract_column_count": len(contract_columns),
        "candidate_contract_column_count": satisfied_count,
        "missing_contract_column_count": len(missing_contract_columns),
        "extra_candidate_column_count": len(extra_candidate_columns),
        "column_contract_share": contract_share,
    }
    details = {
        "missing_contract_columns": missing_contract_columns,
        "extra_candidate_columns": extra_candidate_columns,
        "excluded_baseline_computed_columns": excluded_baseline_computed_columns,
    }
    if missing_contract_columns or extra_candidate_columns:
        return _gate(
            "fail",
            "candidate H5 leaf-input column set differs from the pinned eCPS contract",
            metrics=metrics,
            details=details,
        )
    return _gate(
        "pass",
        "candidate H5 leaf-input column set matches the pinned eCPS contract",
        metrics=metrics,
        details=details,
    )


def _export_support_gate(
    candidate_dataset: Path,
    *,
    baseline_dataset: Path | None,
    period: int,
) -> dict[str, Any]:
    if baseline_dataset is None:
        return _gate(
            "unmeasured",
            "pinned eCPS baseline H5 has not been attached for export-support comparison",
        )
    if not candidate_dataset.exists() or not baseline_dataset.exists():
        missing = [
            str(path)
            for path in (candidate_dataset, baseline_dataset)
            if not path.exists()
        ]
        return _gate(
            "fail",
            "export-support comparison files are missing",
            details={"missing": missing},
        )

    period_key = str(int(period))
    baseline_columns = _h5_period_columns(baseline_dataset, period_key=period_key)
    excluded_baseline_computed_columns = sorted(
        _computed_policyengine_us_export_columns(baseline_columns)
    )
    required_columns = set(baseline_columns) - set(excluded_baseline_computed_columns)
    from microplex_us.pipelines.check_export_columns import (
        compute_support_diff,
        support_diff_to_dict,
    )

    support_diff = compute_support_diff(
        candidate_dataset,
        baseline_h5=baseline_dataset,
        period=period,
        required_columns=required_columns,
    )
    metrics = {
        "period": int(period),
        "checked_export_column_count": len(support_diff.checked_columns),
        "ecps_populated_export_column_count": len(
            support_diff.baseline_populated_columns
        ),
        "ecps_filler_export_column_count": len(support_diff.baseline_filler_columns),
        "unsupported_populated_export_column_count": len(support_diff.issues),
        "excluded_baseline_computed_column_count": len(
            excluded_baseline_computed_columns
        ),
    }
    details = {
        **support_diff_to_dict(support_diff),
        "excluded_baseline_computed_columns": excluded_baseline_computed_columns,
    }
    if support_diff.issues:
        return _gate(
            "fail",
            "candidate export columns lack support for eCPS-populated columns",
            metrics=metrics,
            details=details,
        )
    return _gate(
        "pass",
        "candidate export columns have support for every eCPS-populated export",
        metrics=metrics,
        details=details,
    )


def _export_lineage_gate(
    *,
    baseline_dataset: Path | None,
    period: int,
) -> dict[str, Any]:
    if baseline_dataset is None:
        return _gate(
            "unmeasured",
            "pinned eCPS baseline H5 has not been attached for export-lineage comparison",
        )
    if not baseline_dataset.exists():
        return _gate(
            "fail",
            "export-lineage comparison file is missing",
            details={"missing": [str(baseline_dataset)]},
        )

    from microplex_us.pipelines.export_lineage_manifest import (
        build_export_lineage_manifest,
    )

    manifest = build_export_lineage_manifest(
        support_baseline=baseline_dataset,
        period=period,
    )
    summary = dict(manifest["summary"])
    columns = list(manifest["columns"])
    default_only_columns = [
        column["column"]
        for column in columns
        if column["export_path_status"] == "default_only"
    ]
    details = {
        "issues": manifest["issues"],
        "default_only_columns": default_only_columns,
    }
    if manifest["issues"]:
        return _gate(
            "fail",
            "required eCPS-populated exports lack source/code lineage",
            metrics=summary,
            details=details,
        )
    return _gate(
        "pass",
        "every eCPS-populated required export has source/code lineage",
        metrics=summary,
        details=details,
    )


def _computed_policyengine_us_export_columns(columns: list[str]) -> set[str]:
    try:
        import policyengine_us

        from microplex_us.policyengine.us import (
            detect_policyengine_computed_export_variables,
        )
    except ImportError:
        return set(columns) & _KNOWN_PEUS_COMPUTED_ECPS_CONTRACT_COLUMNS

    tax_benefit_system = getattr(
        policyengine_us.system,
        "system",
        policyengine_us.system,
    )
    return detect_policyengine_computed_export_variables(
        tax_benefit_system,
        tuple(columns),
    )


def _h5_top_level_columns(candidate_dataset: Path) -> set[str]:
    """Return base column names at the top level of an exported H5.

    Accepts both shapes a column can take: a group ``<column>/<period>``
    (the eCPS export layout) or a flat dataset ``<column>``. Names are
    collapsed to the base column via ``split("/")[0]`` so the two are
    comparable. Shared by the fast column-parity CLI
    (``check_export_columns``) so it reads columns the same way the
    artifact path does.
    """
    with h5py.File(candidate_dataset, "r") as handle:
        return {name.split("/")[0] for name in handle.keys()}


def _h5_period_columns(path: Path, *, period_key: str) -> list[str]:
    with h5py.File(path, "r") as handle:
        return sorted(
            name
            for name, value in handle.items()
            if isinstance(value, h5py.Group) and period_key in value
        )


def _compatibility_gate(candidate_dataset: Path, *, period: int) -> dict[str, Any]:
    try:
        inspection = _inspect_h5_contract(candidate_dataset, period=period)
        if inspection["failures"]:
            return _gate(
                "fail",
                "candidate H5 violates the PolicyEngine table contract",
                metrics=inspection["metrics"],
                details=inspection["details"],
            )
        from microplex_us.policyengine.us import load_policyengine_us_entity_tables

        tables = load_policyengine_us_entity_tables(
            candidate_dataset,
            period=period,
            variables=(),
        )
        household_weight_sum = float(tables.households["household_weight"].sum())
        if household_weight_sum <= 0:
            return _gate(
                "fail",
                "candidate H5 has nonpositive household weight sum",
                metrics={"household_weight_sum": household_weight_sum},
            )
        return _gate(
            "pass",
            "candidate H5 satisfies the structural PolicyEngine table contract",
            metrics={
                **inspection["metrics"],
                "household_count": int(len(tables.households)),
                "person_count": int(len(tables.persons))
                if tables.persons is not None
                else 0,
                "household_weight_sum": household_weight_sum,
            },
        )
    except Exception as exc:  # noqa: BLE001 - this is a gate report boundary.
        return _gate(
            "fail",
            "candidate H5 failed the structural PolicyEngine table contract",
            details={"error": str(exc)},
        )


def _inspect_h5_contract(
    candidate_dataset: Path, *, period: int
) -> dict[str, dict[str, Any] | list[str]]:
    period_key = str(int(period))
    failures: list[str] = []
    details: dict[str, Any] = {}
    metrics: dict[str, Any] = {"period": int(period)}
    with h5py.File(candidate_dataset, "r") as handle:
        period_groups = sorted(
            name for name, value in handle.items() if isinstance(value, h5py.Group)
        )
        metrics["variable_count"] = len(period_groups)
        metrics["required_structural_array_count"] = len(_REQUIRED_PERIOD_ARRAYS)
        missing = [
            variable
            for variable in _REQUIRED_PERIOD_ARRAYS
            if variable not in handle or period_key not in handle[variable]
        ]
        if missing:
            failures.append("missing_required_period_arrays")
            details["missing_arrays"] = missing

        variables_missing_period = [
            variable for variable in period_groups if period_key not in handle[variable]
        ]
        if variables_missing_period:
            failures.append("variables_missing_requested_period")
            details["variables_missing_period"] = variables_missing_period

        forbidden_variables = [
            variable
            for variable in period_groups
            if _is_forbidden_source_diagnostic_variable(variable)
        ]
        if forbidden_variables:
            failures.append("source_diagnostic_variables_exported")
            details["forbidden_source_diagnostic_variables"] = forbidden_variables

        arrays = {
            variable: np.asarray(handle[variable][period_key])
            for variable in _REQUIRED_PERIOD_ARRAYS
            if variable in handle and period_key in handle[variable]
        }
        _inspect_structural_dtypes(arrays, failures=failures, details=details)
        _inspect_structural_lengths(arrays, failures=failures, details=details)
        _inspect_entity_ids(arrays, failures=failures, details=details)
        _inspect_person_links(arrays, failures=failures, details=details)
        _inspect_household_weights(arrays, failures=failures, details=details)
        nonfinite = _nonfinite_numeric_period_arrays(
            handle,
            period_key=period_key,
        )
        if nonfinite:
            failures.append("nonfinite_numeric_period_arrays")
            details["nonfinite_numeric_arrays"] = nonfinite
            metrics["nonfinite_numeric_array_count"] = len(nonfinite)
        else:
            metrics["nonfinite_numeric_array_count"] = 0

    metrics.update(_structural_count_metrics(arrays))
    return {"failures": failures, "details": details, "metrics": metrics}


def _is_forbidden_source_diagnostic_variable(variable: str) -> bool:
    return (
        variable in _FORBIDDEN_SOURCE_DIAGNOSTIC_VARIABLES
        or variable.startswith(_FORBIDDEN_SOURCE_DIAGNOSTIC_PREFIXES)
        or variable.endswith(_FORBIDDEN_SOURCE_DIAGNOSTIC_SUFFIXES)
    )


def _inspect_structural_dtypes(
    arrays: dict[str, np.ndarray],
    *,
    failures: list[str],
    details: dict[str, Any],
) -> None:
    invalid_id_dtypes = {
        variable: str(values.dtype)
        for variable, values in arrays.items()
        if variable != "household_weight" and not _is_valid_id_dtype(values.dtype)
    }
    if invalid_id_dtypes:
        failures.append("invalid_structural_id_dtypes")
        details["invalid_structural_id_dtypes"] = invalid_id_dtypes

    household_weight = arrays.get("household_weight")
    if household_weight is not None and not _is_valid_weight_dtype(
        household_weight.dtype
    ):
        failures.append("invalid_household_weight_dtype")
        details["invalid_household_weight_dtype"] = str(household_weight.dtype)


def _is_valid_id_dtype(dtype: np.dtype[Any]) -> bool:
    return np.issubdtype(dtype, np.integer) or dtype.kind in {"S", "U", "O"}


def _is_valid_weight_dtype(dtype: np.dtype[Any]) -> bool:
    return np.issubdtype(dtype, np.number) and dtype.kind != "b"


def _inspect_structural_lengths(
    arrays: dict[str, np.ndarray],
    *,
    failures: list[str],
    details: dict[str, Any],
) -> None:
    household_count = _array_length(arrays.get("household_id"))
    person_count = _array_length(arrays.get("person_id"))
    length_mismatches: dict[str, dict[str, int | None]] = {}
    if household_count is not None:
        _record_length_mismatch(
            length_mismatches,
            "household_weight",
            actual=_array_length(arrays.get("household_weight")),
            expected=household_count,
        )
    if person_count is not None:
        for variable in _PERSON_LINK_ARRAYS.values():
            _record_length_mismatch(
                length_mismatches,
                variable,
                actual=_array_length(arrays.get(variable)),
                expected=person_count,
            )
    if length_mismatches:
        failures.append("structural_array_length_mismatches")
        details["structural_array_length_mismatches"] = length_mismatches


def _array_length(values: np.ndarray | None) -> int | None:
    if values is None:
        return None
    return int(values.shape[0]) if values.ndim else 1


def _record_length_mismatch(
    length_mismatches: dict[str, dict[str, int | None]],
    variable: str,
    *,
    actual: int | None,
    expected: int,
) -> None:
    if actual != expected:
        length_mismatches[variable] = {
            "actual": actual,
            "expected": expected,
        }


def _inspect_entity_ids(
    arrays: dict[str, np.ndarray],
    *,
    failures: list[str],
    details: dict[str, Any],
) -> None:
    duplicate_ids: dict[str, int] = {}
    empty_entities: list[str] = []
    for variable in _ENTITY_ID_ARRAYS.values():
        values = arrays.get(variable)
        if values is None:
            continue
        if _array_length(values) == 0:
            empty_entities.append(variable)
            continue
        unique_count = len(np.unique(values))
        if unique_count != len(values):
            duplicate_ids[variable] = int(len(values) - unique_count)
    if empty_entities:
        failures.append("empty_entity_id_arrays")
        details["empty_entity_id_arrays"] = empty_entities
    if duplicate_ids:
        failures.append("duplicate_entity_ids")
        details["duplicate_entity_ids"] = duplicate_ids


def _inspect_person_links(
    arrays: dict[str, np.ndarray],
    *,
    failures: list[str],
    details: dict[str, Any],
) -> None:
    invalid_links: dict[str, list[Any]] = {}
    for entity, link_variable in _PERSON_LINK_ARRAYS.items():
        id_variable = _ENTITY_ID_ARRAYS[entity]
        link_values = arrays.get(link_variable)
        id_values = arrays.get(id_variable)
        if link_values is None or id_values is None:
            continue
        missing_values = link_values[~np.isin(link_values, id_values)]
        if missing_values.size:
            invalid_links[link_variable] = _jsonable_sample(missing_values)
    if invalid_links:
        failures.append("invalid_person_entity_links")
        details["invalid_person_entity_links"] = invalid_links


def _inspect_household_weights(
    arrays: dict[str, np.ndarray],
    *,
    failures: list[str],
    details: dict[str, Any],
) -> None:
    household_weight = arrays.get("household_weight")
    if household_weight is None:
        return
    values = np.asarray(household_weight, dtype=np.float64)
    if not np.isfinite(values).all():
        failures.append("nonfinite_household_weights")
        details["nonfinite_household_weight_count"] = int(
            np.size(values) - np.isfinite(values).sum()
        )
    negative_count = int((values < 0).sum())
    if negative_count:
        failures.append("negative_household_weights")
        details["negative_household_weight_count"] = negative_count


def _nonfinite_numeric_period_arrays(
    handle: h5py.File,
    *,
    period_key: str,
) -> dict[str, int]:
    nonfinite: dict[str, int] = {}
    for variable, group in handle.items():
        if not isinstance(group, h5py.Group) or period_key not in group:
            continue
        dataset = group[period_key]
        if not np.issubdtype(dataset.dtype, np.floating):
            continue
        values = np.asarray(dataset)
        finite = np.isfinite(values)
        if not finite.all():
            nonfinite[variable] = int(np.size(values) - finite.sum())
    return nonfinite


def _structural_count_metrics(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for entity, variable in _ENTITY_ID_ARRAYS.items():
        count = _array_length(arrays.get(variable))
        if count is not None:
            metrics[f"{entity}_count"] = count
    return metrics


def _jsonable_sample(values: np.ndarray, *, limit: int = 5) -> list[Any]:
    sample = np.unique(values)[:limit].tolist()
    result: list[Any] = []
    for value in sample:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        result.append(value)
    return result


def _resolve_ecps_comparison_payload(
    ecps_comparison_payload: Any,
    *,
    candidate_dataset: Path,
    baseline_dataset: Path | None,
    period: int,
    compute_native_scores: bool,
    policyengine_us_data_repo: str | Path | None,
    policyengine_us_data_python: str | Path | None,
) -> Any:
    if ecps_comparison_payload is not None:
        return ecps_comparison_payload
    if not compute_native_scores or baseline_dataset is None:
        return None
    from microplex_us.pipelines.pe_native_scores import compute_us_pe_native_scores

    return compute_us_pe_native_scores(
        candidate_dataset_path=candidate_dataset,
        baseline_dataset_path=baseline_dataset,
        period=period,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )


def _ecps_comparison_gate(
    ecps_comparison_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if ecps_comparison_payload is None:
        return _gate(
            "unmeasured",
            "PE-native eCPS comparison has not been attached",
        )
    summary = _ecps_comparison_summary(ecps_comparison_payload)
    candidate_loss = summary.get("candidate_enhanced_cps_native_loss")
    baseline_loss = summary.get("baseline_enhanced_cps_native_loss")
    loss_delta = summary.get("enhanced_cps_native_loss_delta")
    reported_candidate_beats = summary.get("candidate_beats_baseline")
    details: dict[str, Any] = {}
    if candidate_loss is not None and baseline_loss is not None:
        computed_loss_delta = float(candidate_loss) - float(baseline_loss)
        if (
            loss_delta is not None
            and abs(float(loss_delta) - computed_loss_delta) > 1e-12
        ):
            details["reported_loss_delta"] = loss_delta
            details["computed_loss_delta"] = computed_loss_delta
        loss_delta = computed_loss_delta
    candidate_beats = None
    if loss_delta is not None:
        candidate_beats = float(loss_delta) < 0.0
    if (
        reported_candidate_beats is not None
        and candidate_beats is not None
        and bool(reported_candidate_beats) != candidate_beats
    ):
        details["reported_candidate_beats_baseline"] = reported_candidate_beats
        details["computed_candidate_beats_baseline"] = candidate_beats
    contract = _ecps_comparison_contract_summary(
        ecps_comparison_payload,
        summary,
    )
    details.update(contract["details"])
    missing_requirements = list(contract["missing_requirements"])
    status: GateStatus
    if candidate_beats is None:
        status = "unmeasured"
    elif missing_requirements:
        status = "fail"
    else:
        status = "pass" if bool(candidate_beats) else "fail"
    return _gate(
        status,
        (
            "candidate beats pinned eCPS under the release comparison contract"
            if status == "pass"
            else (
                (
                    "eCPS comparison is missing release-contract evidence: "
                    + ", ".join(missing_requirements)
                )
                if missing_requirements
                else "candidate does not beat pinned eCPS on PE-native broad loss"
                if status == "fail"
                else "PE-native eCPS comparison payload is incomplete"
            )
        ),
        metrics={
            "candidate_enhanced_cps_native_loss": candidate_loss,
            "baseline_enhanced_cps_native_loss": baseline_loss,
            "enhanced_cps_native_loss_delta": loss_delta,
            "n_targets_kept": summary.get("n_targets_kept"),
            "matched_household_count": contract["matched_household_count"],
            "holdout_target_fraction": contract["holdout_target_fraction"],
        },
        details=details,
    )


def _ecps_comparison_summary(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        for item in payload:
            summary = _ecps_comparison_summary(item)
            if summary:
                return summary
        return {}
    if not isinstance(payload, dict):
        return {}
    for key in ("summary", "broad_loss", "loss_summary"):
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    if "candidate_enhanced_cps_native_loss" in payload:
        return dict(payload)
    if (
        "best_variant_loss" in payload
        and "baseline_enhanced_cps_native_loss" in payload
    ):
        best_variant_loss = payload.get("best_variant_loss")
        baseline_loss = payload.get("baseline_enhanced_cps_native_loss")
        loss_delta = None
        candidate_beats = None
        if best_variant_loss is not None and baseline_loss is not None:
            loss_delta = float(best_variant_loss) - float(baseline_loss)
            candidate_beats = loss_delta < 0.0
        return {
            "candidate_enhanced_cps_native_loss": best_variant_loss,
            "baseline_enhanced_cps_native_loss": baseline_loss,
            "enhanced_cps_native_loss_delta": loss_delta,
            "candidate_beats_baseline": candidate_beats,
            "best_variant_label": payload.get("best_variant_label"),
        }
    return {}


def _ecps_comparison_contract_summary(
    payload: Any,
    summary: dict[str, Any],
) -> dict[str, Any]:
    candidate_households = _first_nested_present(
        payload,
        summary,
        "candidate_household_count",
        "candidate_households",
        "candidate_n_households",
        "candidate_record_count",
    )
    baseline_households = _first_nested_present(
        payload,
        summary,
        "baseline_household_count",
        "baseline_households",
        "baseline_n_households",
        "baseline_record_count",
    )
    matched_household_count = None
    if candidate_households is not None and baseline_households is not None:
        matched_household_count = int(candidate_households) == int(baseline_households)
    else:
        matched_household_count = _first_nested_present(
            payload,
            summary,
            "matched_household_count",
            "matched_n",
            "matched_record_count",
        )
        if matched_household_count is not None:
            matched_household_count = bool(matched_household_count)

    symmetric_refit = _first_nested_present(
        payload,
        summary,
        "symmetric_refit",
        "symmetric_reweight",
        "refit_both",
    )
    candidate_refit_config = _first_nested_present(
        payload,
        summary,
        "candidate_refit_config",
        "candidate_fit_config",
    )
    baseline_refit_config = _first_nested_present(
        payload,
        summary,
        "baseline_refit_config",
        "baseline_fit_config",
    )
    if symmetric_refit is None and (
        isinstance(candidate_refit_config, dict)
        and isinstance(baseline_refit_config, dict)
    ):
        symmetric_refit = candidate_refit_config == baseline_refit_config
    if symmetric_refit is not None:
        symmetric_refit = bool(symmetric_refit)

    score_candidate_only = _first_nested_present(
        payload,
        summary,
        "score_candidate_only",
    )
    if score_candidate_only is not None and bool(score_candidate_only):
        symmetric_refit = False

    objective_identity = _first_nested_present(
        payload,
        summary,
        "refit_objective_matches_scoring",
        "objective_identity_recovery_passed",
    )
    if objective_identity is not None:
        objective_identity = bool(objective_identity)

    ecps_refit_recovery = _first_nested_present(
        payload,
        summary,
        "ecps_refit_recovery_passed",
        "baseline_refit_recovery_passed",
    )
    if ecps_refit_recovery is not None:
        ecps_refit_recovery = bool(ecps_refit_recovery)

    holdout_target_fraction = _first_nested_present(
        payload,
        summary,
        "holdout_target_fraction",
    )
    holdout_targets = _first_nested_present(
        payload,
        summary,
        "holdout_targets",
        "n_holdout_targets",
    )
    has_holdout_targets = False
    if holdout_target_fraction is not None:
        has_holdout_targets = float(holdout_target_fraction) > 0.0
    elif holdout_targets is not None:
        has_holdout_targets = int(holdout_targets) > 0

    protected_summary = _protected_family_floor_summary(payload, summary)
    core_benchmark_summary = _core_benchmark_family_floor_summary(payload, summary)

    requirements = {
        "matched_household_count": matched_household_count is True,
        "symmetric_refit": symmetric_refit is True,
        "refit_objective_matches_scoring": objective_identity is True,
        "ecps_refit_recovery": ecps_refit_recovery is True,
        "holdout_target_split": has_holdout_targets,
        "protected_family_floors": protected_summary["passed"] is True,
        "core_benchmark_family_floors": core_benchmark_summary["passed"] is True,
    }
    return {
        "matched_household_count": matched_household_count,
        "holdout_target_fraction": holdout_target_fraction,
        "missing_requirements": [
            key for key, passed in requirements.items() if not passed
        ],
        "details": {
            "candidate_household_count": candidate_households,
            "baseline_household_count": baseline_households,
            "symmetric_refit": symmetric_refit,
            "score_candidate_only": score_candidate_only,
            "refit_objective_matches_scoring": objective_identity,
            "ecps_refit_recovery_passed": ecps_refit_recovery,
            "holdout_targets": holdout_targets,
            "protected_family_floor": protected_summary,
            "core_benchmark_family_floor": core_benchmark_summary,
        },
    }


def _first_nested_present(
    payload: Any,
    summary: dict[str, Any],
    *keys: str,
) -> Any:
    candidates: list[dict[str, Any]] = [summary]
    if isinstance(payload, dict):
        candidates.append(payload)
        for nested_key in (
            "comparison_contract",
            "ecps_comparison_contract",
            "release_contract",
            "validation",
            "metadata",
        ):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                candidates.append(nested)
    for candidate in candidates:
        value = _first_present(candidate, *keys)
        if value is not None:
            return value
    return None


def _protected_family_floor_summary(
    payload: Any,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return _family_floor_summary(
        payload,
        summary,
        families=_PROTECTED_ECPS_TARGET_FAMILIES,
        explicit_pass_keys=(
            "protected_family_floors_passed",
            "protected_family_floor_passed",
        ),
        row_keys=(
            "protected_family_losses",
            "protected_family_floor_results",
            "family_loss_comparison",
            "family_breakdown",
        ),
    )


def _core_benchmark_family_floor_summary(
    payload: Any,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return _family_floor_summary(
        payload,
        summary,
        families=_CORE_BENCHMARK_ECPS_TARGET_FAMILIES,
        explicit_pass_keys=(
            "core_benchmark_family_floors_passed",
            "core_benchmark_family_floor_passed",
        ),
        row_keys=(
            "core_benchmark_family_losses",
            "core_benchmark_family_floor_results",
            "family_loss_comparison",
            "family_breakdown",
        ),
    )


def _family_floor_summary(
    payload: Any,
    summary: dict[str, Any],
    *,
    families: tuple[str, ...],
    explicit_pass_keys: tuple[str, ...],
    row_keys: tuple[str, ...],
) -> dict[str, Any]:
    explicit = _first_nested_present(
        payload,
        summary,
        *explicit_pass_keys,
    )
    family_rows = _first_nested_present(payload, summary, *row_keys)
    if family_rows is None and isinstance(payload, dict):
        score_payload = payload.get("score")
        if isinstance(score_payload, dict):
            family_rows = _first_present(score_payload, *row_keys)
            if family_rows is None:
                score_summary = score_payload.get("summary")
                if isinstance(score_summary, dict):
                    family_rows = _first_present(score_summary, *row_keys)
    if family_rows is None:
        return {
            "passed": None,
            "reported_passed": explicit,
            "missing_families": list(families),
            "regressions": [],
        }

    rows_by_family = _family_loss_rows_by_name(family_rows)
    missing: list[str] = []
    regressions: list[dict[str, Any]] = []
    for family in families:
        row = rows_by_family.get(family)
        if row is None:
            missing.append(family)
            continue
        candidate_loss = _first_present(
            row,
            "candidate_loss",
            "candidate_family_loss",
            "candidate_loss_contribution",
        )
        baseline_loss = _first_present(
            row,
            "baseline_loss",
            "baseline_family_loss",
            "baseline_loss_contribution",
        )
        if candidate_loss is None or baseline_loss is None:
            missing.append(family)
            continue
        delta = float(candidate_loss) - float(baseline_loss)
        tolerance = max(
            _PROTECTED_FAMILY_ABSOLUTE_LOSS_TOLERANCE,
            _PROTECTED_FAMILY_RELATIVE_LOSS_TOLERANCE * abs(float(baseline_loss)),
        )
        if delta > tolerance:
            regressions.append(
                {
                    "family": family,
                    "candidate_loss": float(candidate_loss),
                    "baseline_loss": float(baseline_loss),
                    "loss_delta": delta,
                    "allowed_delta": tolerance,
                }
            )
    passed = not missing and not regressions
    if explicit is not None:
        passed = passed and bool(explicit)
    return {
        "passed": passed,
        "missing_families": sorted(set(missing)),
        "regressions": regressions,
        "relative_tolerance": _PROTECTED_FAMILY_RELATIVE_LOSS_TOLERANCE,
        "absolute_tolerance": _PROTECTED_FAMILY_ABSOLUTE_LOSS_TOLERANCE,
    }


def _family_loss_rows_by_name(family_rows: Any) -> dict[str, dict[str, Any]]:
    if isinstance(family_rows, dict):
        return {
            str(family): dict(row)
            for family, row in family_rows.items()
            if isinstance(row, dict)
        }
    if isinstance(family_rows, list):
        rows: dict[str, dict[str, Any]] = {}
        for row in family_rows:
            if not isinstance(row, dict):
                continue
            family = _first_present(row, "family", "target_family", "name")
            if family is not None:
                rows[str(family)] = dict(row)
        return rows
    return {}


def _runtime_gate(
    runtime_smoke_payload: dict[str, Any] | None,
    *,
    runtime_ratio_threshold: float,
) -> dict[str, Any]:
    if runtime_smoke_payload is None:
        return _gate("unmeasured", "runtime smoke benchmark has not been attached")
    payload = dict(runtime_smoke_payload)
    threshold = float(runtime_ratio_threshold)
    ratio = _first_present(payload, "runtime_ratio", "median_runtime_ratio")
    candidate_seconds = _first_present(payload, "candidate_seconds")
    baseline_seconds = _first_present(payload, "baseline_seconds")
    candidate_payload = payload.get("candidate")
    baseline_payload = payload.get("baseline")
    if isinstance(candidate_payload, dict):
        if candidate_seconds is None:
            candidate_seconds = _first_present(
                candidate_payload,
                "median_elapsed_seconds",
                "elapsed_seconds",
            )
    if isinstance(baseline_payload, dict):
        if baseline_seconds is None:
            baseline_seconds = _first_present(
                baseline_payload,
                "median_elapsed_seconds",
                "elapsed_seconds",
            )
    if ratio is None and candidate_seconds is not None and baseline_seconds:
        ratio = float(candidate_seconds) / float(baseline_seconds)
    passes = _first_present(
        payload,
        "passes_runtime_gate",
        "passes_runtime_ratio_1_25x",
    )
    details: dict[str, Any] = {}
    reported_threshold = payload.get("runtime_ratio_threshold")
    reported_threshold_matches = False
    try:
        reported_threshold_matches = float(reported_threshold) == threshold
    except (TypeError, ValueError):
        pass
    if reported_threshold is not None and not reported_threshold_matches:
        details["reported_runtime_ratio_threshold"] = reported_threshold
        details["enforced_runtime_ratio_threshold"] = threshold
    if ratio is None:
        return _gate(
            "unmeasured",
            "runtime smoke payload is missing ratio or candidate/baseline seconds",
            metrics={
                "candidate_seconds": candidate_seconds,
                "baseline_seconds": baseline_seconds,
                "runtime_ratio": ratio,
                "runtime_ratio_threshold": threshold,
            },
        )
    derived_passes = float(ratio) <= threshold
    if passes is not None and bool(passes) != derived_passes:
        details["reported_passes_runtime_gate"] = passes
        details["computed_passes_runtime_gate"] = derived_passes
    return _gate(
        "pass" if derived_passes else "fail",
        (
            "candidate runtime is inside the smoke benchmark threshold"
            if derived_passes
            else "candidate runtime exceeds the smoke benchmark threshold"
        ),
        metrics={
            "candidate_seconds": candidate_seconds,
            "baseline_seconds": baseline_seconds,
            "runtime_ratio": ratio,
            "runtime_ratio_threshold": threshold,
        },
        details=details,
    )


def _resolve_source_weight_diagnostics_payload(
    artifact_root: Path,
    manifest: dict[str, Any],
    *,
    source_weight_diagnostics_payload: dict[str, Any] | None,
    source_weight_diagnostics_path: str | Path | None,
) -> dict[str, Any] | None:
    if source_weight_diagnostics_payload is not None:
        return source_weight_diagnostics_payload
    path = _resolve_optional_manifest_artifact_path(
        artifact_root,
        manifest,
        source_weight_diagnostics_path,
        "source_weight_diagnostics",
        "source_diagnostics",
    )
    if path is None:
        return None
    return _load_json_file(path)


def _resolve_optional_manifest_artifact_path(
    artifact_root: Path,
    manifest: dict[str, Any],
    explicit_path: str | Path | None,
    *artifact_keys: str,
) -> Path | None:
    if explicit_path is not None:
        return Path(explicit_path).expanduser()
    artifacts = dict(manifest.get("artifacts", {}))
    for key in artifact_keys:
        value = artifacts.get(key)
        if isinstance(value, str) and value:
            path = Path(value).expanduser()
            return path if path.is_absolute() else artifact_root / path
    return None


def _source_weight_diagnostics_gate(
    payload: dict[str, Any] | None,
    *,
    max_support_weight_share: float,
) -> dict[str, Any]:
    threshold = float(max_support_weight_share)
    if payload is None:
        return _gate(
            "unmeasured",
            "source-weight diagnostics sidecar has not been attached",
            metrics={"max_support_weight_share": threshold},
        )
    if not isinstance(payload, dict):
        return _gate(
            "fail",
            "source-weight diagnostics sidecar must be a JSON object",
            metrics={"max_support_weight_share": threshold},
        )

    entries = _source_weight_entries(payload)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    support_share = _first_present(
        payload,
        "support_household_weight_share",
        "support_weight_share",
        "puf_support_household_weight_share",
        "puf_clone_household_weight_share",
        "clone_household_weight_share",
    )
    puf_support_share = _first_present(
        payload,
        "puf_support_household_weight_share",
        "puf_clone_household_weight_share",
    )
    max_source_share = _first_present(
        payload,
        "max_source_household_weight_share",
        "largest_source_household_weight_share",
    )
    if isinstance(summary, dict):
        support_share = (
            support_share
            if support_share is not None
            else _first_present(
                summary,
                "support_household_weight_share",
                "support_weight_share",
                "puf_support_household_weight_share",
                "puf_clone_household_weight_share",
                "clone_household_weight_share",
            )
        )
        puf_support_share = (
            puf_support_share
            if puf_support_share is not None
            else _first_present(
                summary,
                "puf_support_household_weight_share",
                "puf_clone_household_weight_share",
            )
        )
        max_source_share = (
            max_source_share
            if max_source_share is not None
            else _first_present(
                summary,
                "max_source_household_weight_share",
                "largest_source_household_weight_share",
            )
        )

    if entries:
        entry_shares = [_entry_household_weight_share(entry) for entry in entries]
        numeric_entry_shares = [
            share for share in entry_shares if share is not None and share >= 0
        ]
        if max_source_share is None and numeric_entry_shares:
            max_source_share = max(numeric_entry_shares)
        if support_share is None:
            support_share = sum(
                share
                for entry, share in zip(entries, entry_shares, strict=True)
                if share is not None and _is_support_source_entry(entry)
            )
        if puf_support_share is None:
            puf_support_share = sum(
                share
                for entry, share in zip(entries, entry_shares, strict=True)
                if share is not None
                and _is_support_source_entry(entry)
                and _is_puf_source_entry(entry)
            )

    support_share_float = _optional_float(support_share)
    puf_support_share_float = _optional_float(puf_support_share)
    max_source_share_float = _optional_float(max_source_share)
    failures: list[str] = []
    if not entries and support_share_float is None and puf_support_share_float is None:
        failures.append("source_breakdown")
    if support_share_float is not None and support_share_float > threshold:
        failures.append("support_household_weight_share")
    if puf_support_share_float is not None and puf_support_share_float > threshold:
        failures.append("puf_support_household_weight_share")

    status: GateStatus = "pass" if not failures else "fail"
    return _gate(
        status,
        (
            "source/support weight diagnostics are within dominance thresholds"
            if status == "pass"
            else "source/support weight diagnostics are missing or exceed dominance thresholds"
        ),
        metrics={
            "source_entry_count": len(entries),
            "support_household_weight_share": support_share_float,
            "puf_support_household_weight_share": puf_support_share_float,
            "max_source_household_weight_share": max_source_share_float,
            "max_support_weight_share": threshold,
        },
        details={"failures": failures} if failures else None,
    )


def _source_weight_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("sources", "source_classes", "source_weight_shares"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            entries: list[dict[str, Any]] = []
            for name, item in value.items():
                if isinstance(item, dict):
                    entries.append({"source_name": name, **item})
            return entries
    return []


def _entry_household_weight_share(entry: dict[str, Any]) -> float | None:
    return _optional_float(
        _first_present(
            entry,
            "household_weight_share",
            "weight_share",
            "share",
        )
    )


def _is_support_source_entry(entry: dict[str, Any]) -> bool:
    source_class = str(
        _first_present(entry, "source_class", "class", "kind", "category") or ""
    ).lower()
    source_name = str(_first_present(entry, "source_name", "name") or "").lower()
    if "fixed" in source_class or "forbes" in source_name:
        return False
    support_tokens = ("support", "clone", "donor_replay")
    return any(token in source_class for token in support_tokens) or any(
        token in source_name for token in support_tokens
    )


def _is_puf_source_entry(entry: dict[str, Any]) -> bool:
    source_name = str(_first_present(entry, "source_name", "name") or "").lower()
    source_class = str(
        _first_present(entry, "source_class", "class", "kind", "category") or ""
    ).lower()
    return "puf" in source_name or "irs_soi" in source_name or "puf" in source_class


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _arch_target_coverage_gate(
    arch_coverage_payload: dict[str, Any] | None,
    *,
    expected_period: int,
    expected_profile: str,
) -> dict[str, Any]:
    if arch_coverage_payload is None:
        return _gate(
            "unmeasured",
            "Arch target coverage report has not been attached",
            metrics={
                "expected_profile": expected_profile,
                "expected_period": int(expected_period),
            },
        )
    if not isinstance(arch_coverage_payload, dict):
        return _gate(
            "fail",
            "Arch target coverage report must be a JSON object",
            metrics={
                "expected_profile": expected_profile,
                "expected_period": int(expected_period),
            },
        )
    payload = dict(arch_coverage_payload)
    profile_name = payload.get("profile_name")
    period = payload.get("period")
    target_cell_count = payload.get("target_cell_count")
    covered_cell_count = payload.get("covered_cell_count")
    uncovered_cell_count = payload.get("uncovered_cell_count")
    coverage_rate = payload.get("coverage_rate")
    failures: list[str] = []
    if profile_name != expected_profile:
        failures.append("profile_name")
    if period is None or int(period) != int(expected_period):
        failures.append("period")
    if target_cell_count is None or int(target_cell_count) <= 0:
        failures.append("target_cell_count")
    if uncovered_cell_count is None or int(uncovered_cell_count) != 0:
        failures.append("uncovered_cell_count")
    if (
        covered_cell_count is not None
        and target_cell_count is not None
        and int(covered_cell_count) != int(target_cell_count)
    ):
        failures.append("covered_cell_count")
    if coverage_rate is None or float(coverage_rate) < 1.0:
        failures.append("coverage_rate")
    status: GateStatus = "pass" if not failures else "fail"
    return _gate(
        status,
        (
            "Arch source-backed target coverage is complete"
            if status == "pass"
            else "Arch source-backed target coverage is incomplete or mismatched"
        ),
        metrics={
            "profile_name": profile_name,
            "expected_profile": expected_profile,
            "period": period,
            "expected_period": int(expected_period),
            "target_cell_count": target_cell_count,
            "covered_cell_count": covered_cell_count,
            "uncovered_cell_count": uncovered_cell_count,
            "coverage_rate": coverage_rate,
        },
        details={"failures": failures} if failures else None,
    )


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _benchmark_manifest_gate(
    benchmark_manifest_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if benchmark_manifest_path is None:
        return (
            _gate(
                "unmeasured",
                "frozen microsimulation benchmark manifest has not been attached",
            ),
            None,
        )
    manifest_path = Path(benchmark_manifest_path).expanduser()
    if not manifest_path.exists():
        return (
            _gate(
                "fail",
                "frozen microsimulation benchmark manifest path does not exist",
                details={"path": str(manifest_path)},
            ),
            None,
        )
    descriptor = _file_descriptor(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return (
            _gate(
                "fail",
                "frozen microsimulation benchmark manifest is not valid JSON",
                details={"path": str(manifest_path), "error": str(exc)},
            ),
            descriptor,
        )
    evidence = _benchmark_manifest_evidence(payload)
    if evidence["missing"]:
        return (
            _gate(
                "fail",
                "frozen microsimulation benchmark manifest is missing pinned evidence",
                metrics={
                    "required_evidence_count": len(
                        _REQUIRED_BENCHMARK_MANIFEST_EVIDENCE
                    ),
                    "present_evidence_count": len(evidence["present"]),
                },
                details={
                    **descriptor,
                    "missing_evidence": evidence["missing"],
                    "present_evidence": evidence["present"],
                },
            ),
            descriptor,
        )
    return (
        _gate(
            "pass",
            "frozen microsimulation benchmark manifest pins baseline, target, and package evidence",
            metrics={
                "required_evidence_count": len(_REQUIRED_BENCHMARK_MANIFEST_EVIDENCE),
                "present_evidence_count": len(evidence["present"]),
            },
            details={**descriptor, "present_evidence": evidence["present"]},
        ),
        descriptor,
    )


def _benchmark_manifest_evidence(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "present": {},
            "missing": list(_REQUIRED_BENCHMARK_MANIFEST_EVIDENCE),
        }
    present: dict[str, Any] = {}
    missing: list[str] = []
    for evidence_name, paths in _REQUIRED_BENCHMARK_MANIFEST_EVIDENCE.items():
        value = _first_nested_path_value(payload, paths)
        if not _valid_benchmark_evidence_value(evidence_name, value):
            missing.append(evidence_name)
            continue
        present[evidence_name] = value
    if _first_nested_path_value(payload, (("policyengine_us_data", "dirty"),)) is True:
        missing.append("policyengine_us_data.clean")
    return {"present": present, "missing": missing}


def _first_nested_path_value(
    payload: dict[str, Any],
    paths: tuple[tuple[str, ...], ...],
) -> Any:
    for path in paths:
        current: Any = payload
        for part in path:
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    return None


def _valid_benchmark_evidence_value(name: str, value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if name.endswith(".sha256"):
        return len(value) == 64 and bool(_HEX_RE.fullmatch(value))
    if name.endswith(".commit"):
        return 7 <= len(value) <= 40 and bool(_HEX_RE.fullmatch(value))
    return True


def _required_gate_names(*, require_ecps_comparison: bool) -> list[str]:
    required = list(_DEFAULT_REQUIRED_GATES)
    if not require_ecps_comparison:
        required.remove("ecps_comparison")
    return required


def _summarize_gates(
    gates: dict[str, dict[str, Any]], *, required_gates: list[str]
) -> dict[str, Any]:
    statuses = {name: gate["status"] for name, gate in gates.items()}
    required = set(required_gates)
    failed_required = [
        name
        for name in required_gates
        if statuses.get(name) == "fail" or statuses.get(name) is None
    ]
    unmeasured_required = [
        name
        for name in required_gates
        if statuses.get(name) == "unmeasured" and name not in failed_required
    ]
    passing_required = [name for name in required_gates if statuses.get(name) == "pass"]
    failed_optional = [
        name
        for name, status in statuses.items()
        if name not in required and status == "fail"
    ]
    unmeasured_optional = [
        name
        for name, status in statuses.items()
        if name not in required and status == "unmeasured"
    ]
    if failed_required:
        overall_status = "failed"
    elif not unmeasured_required:
        overall_status = "passed"
    else:
        overall_status = "incomplete"
    return {
        "status": overall_status,
        "passing_required_gates": passing_required,
        "failed_required_gates": failed_required,
        "unmeasured_required_gates": unmeasured_required,
        "failed_optional_gates": failed_optional,
        "unmeasured_optional_gates": unmeasured_optional,
        "passing_required_gate_count": len(passing_required),
        "failed_required_gate_count": len(failed_required),
        "unmeasured_required_gate_count": len(unmeasured_required),
    }


def _gate(
    status: GateStatus,
    summary: str,
    *,
    metrics: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "summary": summary}
    if metrics:
        payload["metrics"] = metrics
    if details:
        payload["details"] = details
    return payload


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text())


def _load_json_file(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).expanduser().read_text())


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _optional_file_descriptor(path: Path) -> dict[str, Any]:
    if path.exists():
        return _file_descriptor(path)
    return {"path": str(path.resolve()), "exists": False}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path, *, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for enforcing mp-300k artifact gates."""

    parser = argparse.ArgumentParser(
        description="Run persistent mp-300k artifact gates against an artifact bundle."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--candidate-dataset")
    parser.add_argument("--baseline-dataset")
    parser.add_argument(
        "--ecps-comparison-json",
        "--native-scores-json",
        dest="ecps_comparison_json",
    )
    parser.add_argument("--runtime-smoke-json")
    parser.add_argument("--source-weight-diagnostics-json")
    parser.add_argument("--arch-coverage-json")
    parser.add_argument("--benchmark-manifest")
    parser.add_argument("--output-json")
    parser.add_argument("--target-period", type=int, default=2024)
    parser.add_argument(
        "--arch-coverage-profile",
        default=_DEFAULT_ARCH_COVERAGE_PROFILE,
    )
    parser.add_argument("--artifact-size-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--runtime-ratio-threshold", type=float, default=1.25)
    parser.add_argument(
        "--max-support-weight-share",
        type=float,
        default=_DEFAULT_MAX_SUPPORT_WEIGHT_SHARE,
    )
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument(
        "--skip-ecps-computation",
        "--skip-native-score",
        dest="skip_ecps_computation",
        action="store_true",
        help=(
            "Do not compute PE-native scores when --ecps-comparison-json is absent. "
            "The eCPS comparison gate will remain unmeasured."
        ),
    )
    parser.add_argument(
        "--no-require-ecps-comparison",
        action="store_true",
        help=(
            "Keep reporting the eCPS comparison gate, but do not make it block "
            "the overall status. This is the intended deprecation path once eCPS "
            "is no longer the comparator."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return exit code 0 when required gates are unmeasured but not failed.",
    )
    parser.add_argument(
        "--no-update-manifest",
        action="store_true",
        help="Write the gate report without adding it to manifest.json.",
    )
    args = parser.parse_args(argv)

    report_path = write_mp300k_artifact_gate_report(
        args.artifact_dir,
        output_path=args.output_json,
        update_manifest=not args.no_update_manifest,
        candidate_dataset_path=args.candidate_dataset,
        baseline_dataset_path=args.baseline_dataset,
        ecps_comparison_payload=_load_json_file(args.ecps_comparison_json),
        arch_coverage_payload=_load_json_file(args.arch_coverage_json),
        runtime_smoke_payload=_load_json_file(args.runtime_smoke_json),
        source_weight_diagnostics_payload=_load_json_file(
            args.source_weight_diagnostics_json
        ),
        benchmark_manifest_path=args.benchmark_manifest,
        period=args.target_period,
        arch_coverage_profile=args.arch_coverage_profile,
        artifact_size_ratio_threshold=args.artifact_size_ratio_threshold,
        runtime_ratio_threshold=args.runtime_ratio_threshold,
        max_support_weight_share=args.max_support_weight_share,
        compute_native_scores=not args.skip_ecps_computation,
        require_ecps_comparison=not args.no_require_ecps_comparison,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
    )
    print(report_path)
    report = json.loads(report_path.read_text())
    status = report["summary"]["status"]
    if status == "passed" or (status == "incomplete" and args.allow_incomplete):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
