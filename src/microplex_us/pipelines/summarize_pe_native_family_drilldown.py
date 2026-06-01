"""Summarize one PE-native regression family across saved native-audit sidecars."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from microplex_us.pipelines.stage_contracts import (
    get_us_stage_artifact_contract,
    resolve_us_stage_artifact_contract_path,
)


def classify_pe_native_target_family(target_name: str) -> str:
    """Classify one PE target name into the broad-loss family buckets."""

    parts = target_name.split("/")
    if target_name.startswith("state/census/age/"):
        return "state_age_distribution"
    if target_name.startswith("state/census/population_by_state/"):
        return "state_population"
    if target_name.startswith("state/census/population_under_5_by_state/"):
        return "state_population_under_5"
    if target_name.startswith("nation/irs/aca_spending/"):
        return "state_aca_spending"
    if target_name.startswith("state/irs/aca_enrollment/"):
        return "state_aca_enrollment"
    if target_name.startswith("irs/medicaid_enrollment/"):
        return "state_medicaid_enrollment"
    if target_name.endswith("/snap-cost"):
        return "state_snap_cost"
    if target_name.endswith("/snap-hhs"):
        return "state_snap_households"
    if target_name.startswith("state/real_estate_taxes/"):
        return "state_real_estate_taxes"
    if len(parts) >= 3 and parts[0] == "state" and parts[2] == "adjusted_gross_income":
        return "state_agi_distribution"
    if target_name.startswith("nation/jct/"):
        return "national_tax_expenditures"
    if target_name.startswith("nation/net_worth/"):
        return "national_net_worth"
    if target_name.startswith("nation/ssa/"):
        return "national_ssa"
    if target_name.startswith("nation/census/population_by_age/"):
        return "national_population_by_age"
    if target_name == "nation/census/infants":
        return "national_infants"
    if target_name.startswith("nation/census/agi_in_spm_threshold_decile_"):
        return "national_spm_threshold_agi"
    if target_name.startswith("nation/census/count_in_spm_threshold_decile_"):
        return "national_spm_threshold_count"
    if target_name.startswith("nation/census/"):
        return "national_census_other"
    if target_name.startswith("nation/irs/"):
        return "national_irs_other"
    return "other"


def summarize_us_pe_native_family_drilldown(
    artifact_roots: list[str | Path] | tuple[str | Path, ...],
    *,
    family: str,
    top_k: int = 10,
) -> dict[str, Any]:
    """Summarize one regression family across saved native-audit sidecars."""

    normalized_roots = [Path(root) for root in artifact_roots]
    matching_target_counts: Counter[str] = Counter()
    matching_target_delta_sum: defaultdict[str, float] = defaultdict(float)
    lead_target_counts: Counter[str] = Counter()
    lead_target_delta_sum: defaultdict[str, float] = defaultdict(float)
    filing_gap_rows: defaultdict[str, list[float]] = defaultdict(list)
    mfs_gap_rows: defaultdict[str, list[float]] = defaultdict(list)
    matching_audits: list[dict[str, Any]] = []
    lead_audits: list[dict[str, Any]] = []
    total_audits = 0

    for artifact_root in normalized_roots:
        root_key = artifact_root.name
        for bundle_dir in _iter_native_audit_bundle_dirs(artifact_root):
            total_audits += 1
            payload = json.loads(
                resolve_us_stage_artifact_contract_path(
                    bundle_dir,
                    "09_validation_benchmarking",
                    "policyengine_native_audit",
                ).read_text()
            )
            verdict_hints = dict(payload.get("verdictHints", {}))
            support_summary = dict(payload.get("supportAuditSummary", {}))
            matching_targets = [
                row
                for row in list(payload.get("topTargetRegressions", ()))
                if classify_pe_native_target_family(str(row.get("target_name", ""))) == family
            ]
            if not matching_targets:
                continue

            largest_regressing_family = verdict_hints.get("largestRegressingFamily")
            is_lead_audit = largest_regressing_family == family
            audit_row = {
                "artifactRoot": root_key,
                "artifactPath": str(bundle_dir.relative_to(artifact_root)),
                "largestRegressingFamily": largest_regressing_family,
                "largestRegressingTarget": verdict_hints.get("largestRegressingTarget"),
                "matchingTargets": [
                    {
                        "target": row.get("target_name"),
                        "weightedTermDelta": float(row.get("weighted_term_delta", 0.0)),
                    }
                    for row in matching_targets[:top_k]
                ],
            }
            matching_audits.append(audit_row)

            for row in matching_targets:
                target_name = str(row.get("target_name"))
                matching_target_counts[target_name] += 1
                matching_target_delta_sum[target_name] += float(
                    row.get("weighted_term_delta", 0.0)
                )
                if is_lead_audit:
                    lead_target_counts[target_name] += 1
                    lead_target_delta_sum[target_name] += float(
                        row.get("weighted_term_delta", 0.0)
                    )

            if not is_lead_audit:
                continue

            lead_audits.append(
                {
                    **audit_row,
                    "topFilingStatusGaps": [
                        {
                            "filingStatus": row.get("filing_status"),
                            "weightedCountDelta": float(
                                row.get("weighted_count_delta", 0.0)
                            ),
                        }
                        for row in list(
                            support_summary.get("topFilingStatusGaps", ())
                        )[:top_k]
                    ],
                    "topMFSAgiGaps": [
                        {
                            "agiBin": row.get("agi_bin"),
                            "weightedCountDelta": float(
                                row.get("weighted_count_delta", 0.0)
                            ),
                        }
                        for row in list(support_summary.get("topMFSAgiGaps", ()))[:top_k]
                    ],
                }
            )

            for row in list(support_summary.get("topFilingStatusGaps", ())):
                status = str(row.get("filing_status"))
                filing_gap_rows[status].append(float(row.get("weighted_count_delta", 0.0)))
            for row in list(support_summary.get("topMFSAgiGaps", ())):
                agi_bin = str(row.get("agi_bin"))
                mfs_gap_rows[agi_bin].append(float(row.get("weighted_count_delta", 0.0)))

    matching_audits.sort(
        key=lambda row: (
            row["largestRegressingFamily"] != family,
            row["artifactRoot"],
            row["artifactPath"],
        )
    )
    lead_audits.sort(key=lambda row: (row["artifactRoot"], row["artifactPath"]))

    return {
        "artifactRoots": [str(root) for root in normalized_roots],
        "family": family,
        "totalAudits": total_audits,
        "auditsWithMatchingTargets": len(matching_audits),
        "auditsWhereFamilyLeads": len(lead_audits),
        "matchingTargetCounts": _build_target_rows(
            matching_target_counts,
            matching_target_delta_sum,
        )[:top_k],
        "leadTargetCounts": _build_target_rows(
            lead_target_counts,
            lead_target_delta_sum,
        )[:top_k],
        "leadFilingStatusGapSummary": _build_gap_rows(
            filing_gap_rows,
            gap_key="filingStatus",
        )[:top_k],
        "leadMFSAgiGapSummary": _build_gap_rows(
            mfs_gap_rows,
            gap_key="agiBin",
        )[:top_k],
        "matchingAudits": matching_audits[:top_k],
        "leadAudits": lead_audits[:top_k],
    }


def _iter_native_audit_bundle_dirs(artifact_root: Path) -> tuple[Path, ...]:
    audit_hint = get_us_stage_artifact_contract(
        "09_validation_benchmarking",
        "policyengine_native_audit",
    ).path_hint
    dataset_hint = get_us_stage_artifact_contract(
        "08_dataset_assembly",
        "policyengine_dataset",
    ).path_hint
    if audit_hint is None or dataset_hint is None:
        return ()
    return tuple(
        sorted(
            path.parent
            for path in artifact_root.rglob(audit_hint)
            if (path.parent / dataset_hint).exists()
        )
    )


def _build_target_rows(
    counts: Counter[str],
    delta_sum: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for target, count in counts.items():
        total_delta = float(delta_sum[target])
        rows.append(
            {
                "target": target,
                "count": int(count),
                "weightedTermDeltaSum": total_delta,
                "weightedTermDeltaMean": total_delta / float(count),
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["count"]),
            -float(row["weightedTermDeltaSum"]),
            str(row["target"]),
        )
    )
    return rows


def _build_gap_rows(
    values_by_key: dict[str, list[float]],
    *,
    gap_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for key, values in values_by_key.items():
        if not values:
            continue
        rows.append(
            {
                gap_key: key,
                "count": len(values),
                "positiveCount": sum(1 for value in values if value > 0.0),
                "negativeCount": sum(1 for value in values if value < 0.0),
                "weightedCountDeltaSum": float(sum(values)),
                "meanAbsWeightedCountDelta": float(
                    sum(abs(value) for value in values) / float(len(values))
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["meanAbsWeightedCountDelta"]),
            str(row[gap_key]),
        )
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for one-family native-audit drilldowns."""

    parser = argparse.ArgumentParser(
        description="Summarize one PE-native regression family across saved native audits.",
    )
    parser.add_argument("family", help="Broad-loss family to summarize.")
    parser.add_argument(
        "artifact_roots",
        nargs="+",
        help="One or more artifact roots to scan.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of rows to keep for each ranked section.",
    )
    args = parser.parse_args(argv)

    payload = summarize_us_pe_native_family_drilldown(
        args.artifact_roots,
        family=args.family,
        top_k=args.top_k,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
