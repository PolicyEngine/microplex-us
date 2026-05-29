"""Native-loss-driven audit helpers for PE-US-data rebuild artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from microplex_us.pipelines.pe_native_scores import (
    compare_us_pe_native_target_deltas,
    compute_us_pe_native_support_audit,
)


def build_policyengine_us_data_rebuild_native_audit(
    artifact_dir: str | Path,
    *,
    top_k: int = 15,
    manifest_payload: dict[str, Any] | None = None,
    native_scores_payload: dict[str, Any] | None = None,
    imputation_ablation_payload: dict[str, Any] | None = None,
    target_delta_payload: dict[str, Any] | None = None,
    support_audit_payload: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> dict[str, Any]:
    """Build a saved-artifact audit focused on native-loss regressions."""

    artifact_root = Path(artifact_dir)
    manifest = (
        dict(manifest_payload)
        if manifest_payload is not None
        else json.loads((artifact_root / "manifest.json").read_text())
    )
    native_scores = (
        dict(native_scores_payload)
        if native_scores_payload is not None
        else json.loads((artifact_root / "policyengine_native_scores.json").read_text())
    )
    imputation_ablation = (
        dict(imputation_ablation_payload)
        if imputation_ablation_payload is not None
        else _load_optional_json(artifact_root / "imputation_ablation.json")
    )
    config = dict(manifest.get("config", {}))
    artifacts = dict(manifest.get("artifacts", {}))
    candidate_dataset_path = _resolve_candidate_dataset_path(artifact_root, artifacts)
    baseline_dataset_path = _resolve_baseline_dataset_path(config)
    period = int(
        native_scores.get("period")
        or config.get("policyengine_dataset_year")
        or config.get("policyengine_target_period")
        or 2024
    )

    target_delta = (
        dict(target_delta_payload)
        if target_delta_payload is not None
        else compare_us_pe_native_target_deltas(
            from_dataset_path=baseline_dataset_path,
            to_dataset_path=candidate_dataset_path,
            period=period,
            top_k=top_k,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
    )
    support_audit = (
        dict(support_audit_payload)
        if support_audit_payload is not None
        else compute_us_pe_native_support_audit(
            candidate_dataset_path=candidate_dataset_path,
            baseline_dataset_path=baseline_dataset_path,
            period=period,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
    )

    family_breakdown = list(
        native_scores.get("family_breakdown")
        or dict(native_scores.get("broad_loss", {})).get("family_breakdown", ())
    )
    top_family_regressions = sorted(
        [
            row
            for row in family_breakdown
            if float(row.get("loss_contribution_delta", 0.0)) > 0.0
        ],
        key=lambda row: float(row.get("loss_contribution_delta", 0.0)),
        reverse=True,
    )[:top_k]
    top_family_improvements = [
        row
        for row in sorted(
            family_breakdown,
            key=lambda row: float(row.get("loss_contribution_delta", 0.0)),
        )
        if float(row.get("loss_contribution_delta", 0.0)) < 0.0
    ][:top_k]

    support_summary = _build_support_summary(support_audit, top_k=top_k)
    imputation_summary = (
        dict(imputation_ablation.get("summary", {}))
        if imputation_ablation is not None
        else None
    )
    production_variant = (
        imputation_summary.get("production_variant")
        if imputation_summary is not None
        else None
    )

    return {
        "schemaVersion": 1,
        "artifactId": artifact_root.name,
        "artifactDir": str(artifact_root.resolve()),
        "period": period,
        "candidateDatasetPath": str(candidate_dataset_path),
        "baselineDatasetPath": str(baseline_dataset_path),
        "nativeBroadLossSummary": dict(native_scores.get("summary", {})),
        "topFamilyRegressions": top_family_regressions,
        "topFamilyImprovements": top_family_improvements,
        "topTargetRegressions": list(target_delta.get("top_regressions", ())),
        "topTargetImprovements": list(target_delta.get("top_improvements", ())),
        "supportAuditSummary": support_summary,
        "imputationAblationSummary": imputation_summary,
        "supportAudit": support_audit,
        "targetDelta": target_delta,
        "verdictHints": {
            "largestRegressingFamily": (
                top_family_regressions[0]["family"] if top_family_regressions else None
            ),
            "largestRegressingTarget": (
                target_delta.get("top_regressions", [{}])[0].get("target_name")
                if target_delta.get("top_regressions")
                else None
            ),
            "missingStoredCriticalInputs": support_summary[
                "missingStoredCriticalInputs"
            ],
            "productionImputationVariant": production_variant,
            "productionImputationVariantIsMaeWinner": (
                production_variant
                == imputation_summary.get("best_mean_weighted_mae_variant")
                if imputation_summary is not None
                else None
            ),
            "productionImputationVariantIsSupportWinner": (
                production_variant
                == imputation_summary.get("best_mean_support_f1_variant")
                if imputation_summary is not None
                else None
            ),
        },
    }


def write_policyengine_us_data_rebuild_native_audit(
    artifact_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    top_k: int = 15,
    manifest_payload: dict[str, Any] | None = None,
    native_scores_payload: dict[str, Any] | None = None,
    imputation_ablation_payload: dict[str, Any] | None = None,
    target_delta_payload: dict[str, Any] | None = None,
    support_audit_payload: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> Path:
    """Write the native-loss-driven rebuild audit sidecar for one artifact bundle."""

    artifact_root = Path(artifact_dir)
    destination = (
        Path(output_path)
        if output_path is not None
        else artifact_root / "pe_us_data_rebuild_native_audit.json"
    )
    payload = build_policyengine_us_data_rebuild_native_audit(
        artifact_root,
        top_k=top_k,
        manifest_payload=manifest_payload,
        native_scores_payload=native_scores_payload,
        imputation_ablation_payload=imputation_ablation_payload,
        target_delta_payload=target_delta_payload,
        support_audit_payload=support_audit_payload,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def _resolve_candidate_dataset_path(
    artifact_root: Path,
    artifacts: dict[str, Any],
) -> Path:
    dataset_name = artifacts.get("policyengine_dataset")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise FileNotFoundError(
            "Artifact bundle is missing artifacts.policyengine_dataset in manifest.json"
        )
    dataset_path = artifact_root / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Artifact bundle is missing saved policyengine dataset: {dataset_path}"
        )
    return dataset_path


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resolve_baseline_dataset_path(config: dict[str, Any]) -> Path:
    baseline_dataset = config.get("policyengine_baseline_dataset")
    if not isinstance(baseline_dataset, str) or not baseline_dataset:
        raise ValueError(
            "Artifact config is missing policyengine_baseline_dataset for rebuild audit"
        )
    return Path(baseline_dataset).expanduser().resolve()


def _build_support_summary(
    support_audit: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    comparisons = dict(support_audit.get("comparisons", {}))
    critical_rows = list(comparisons.get("critical_input_support", ()))
    missing_stored = [
        row["variable"]
        for row in critical_rows
        if bool(row.get("baseline_stored")) and not bool(row.get("candidate_stored"))
    ]
    critical_support_gaps = sorted(
        critical_rows,
        key=lambda row: float(row.get("weighted_nonzero_delta", 0.0)),
    )[:top_k]
    filing_status_gaps = sorted(
        list(comparisons.get("filing_status_weighted_delta", ())),
        key=lambda row: abs(float(row.get("weighted_count_delta", 0.0))),
        reverse=True,
    )[:top_k]
    mfs_high_agi_gaps = sorted(
        list(comparisons.get("mfs_high_agi_delta", ())),
        key=lambda row: abs(float(row.get("weighted_count_delta", 0.0))),
        reverse=True,
    )[:top_k]
    hoh_agi_gaps = sorted(
        list(comparisons.get("hoh_agi_delta", ())),
        key=lambda row: abs(float(row.get("weighted_count_delta", 0.0))),
        reverse=True,
    )[:top_k]
    ssi_by_age_gaps = sorted(
        list(comparisons.get("ssi_by_age_delta", ())),
        key=lambda row: abs(float(row.get("weighted_recipient_delta", 0.0))),
        reverse=True,
    )[:top_k]
    medicare_part_b_by_age_gaps = sorted(
        list(comparisons.get("medicare_part_b_premiums_by_age_delta", ())),
        key=lambda row: abs(float(row.get("weighted_positive_delta", 0.0))),
        reverse=True,
    )[:top_k]

    return {
        "missingStoredCriticalInputs": missing_stored,
        "topCriticalInputSupportGaps": critical_support_gaps,
        "topFilingStatusGaps": filing_status_gaps,
        "topMFSAgiGaps": mfs_high_agi_gaps,
        "topHoHAgiGaps": hoh_agi_gaps,
        "topSSIByAgeGaps": ssi_by_age_gaps,
        "topMedicarePartBByAgeGaps": medicare_part_b_by_age_gaps,
        "topAcaPtcSpendingGaps": list(
            comparisons.get("state_aca_ptc_spending_top_gaps", ())
        )[:top_k],
        "topMarketplaceEnrollmentGaps": list(
            comparisons.get("state_marketplace_enrollment_top_gaps", ())
        )[:top_k],
        "topAgeBucketGaps": list(
            comparisons.get("state_age_bucket_top_gaps", ())
        )[:top_k],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI for writing one native-loss rebuild audit sidecar."""

    parser = argparse.ArgumentParser(
        description="Write a native-loss-driven audit sidecar for one PE rebuild artifact.",
    )
    parser.add_argument("artifact_dir")
    parser.add_argument("--output-path")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    args = parser.parse_args(argv)

    destination = write_policyengine_us_data_rebuild_native_audit(
        args.artifact_dir,
        output_path=args.output_path,
        top_k=args.top_k,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
