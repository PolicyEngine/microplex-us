"""Summarize recurring PE-native regression families across saved US artifacts."""

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


def _sorted_counter_items(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))


def summarize_us_pe_native_regressions(
    artifact_roots: list[str | Path] | tuple[str | Path, ...],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Summarize recurring PE-native regression families from saved artifacts."""

    rows: list[dict[str, Any]] = []
    largest_family_counts: Counter[str] = Counter()
    top3_family_counts: Counter[str] = Counter()
    family_rank_counts: dict[str, Counter[int]] = defaultdict(Counter)
    family_counts_by_root: dict[str, Counter[str]] = defaultdict(Counter)
    target_counts_from_audits: Counter[str] = Counter()
    missing_critical_inputs_counts: Counter[str] = Counter()

    normalized_roots = [Path(root) for root in artifact_roots]
    for artifact_root in normalized_roots:
        root_key = artifact_root.name
        for bundle_dir in _iter_scored_bundle_dirs(artifact_root):
            scores_payload = json.loads(
                resolve_us_stage_artifact_contract_path(
                    bundle_dir,
                    "09_validation_benchmarking",
                    "policyengine_native_scores",
                ).read_text()
            )
            summary = dict(scores_payload.get("summary", {}))
            positive_families = [
                row
                for row in list(scores_payload.get("family_breakdown", ()))
                if float(row.get("loss_contribution_delta", 0.0)) > 0.0
            ]
            positive_families.sort(
                key=lambda row: float(row.get("loss_contribution_delta", 0.0)),
                reverse=True,
            )

            largest_family = positive_families[0] if positive_families else {}
            top3_families = [row.get("family") for row in positive_families[:3]]

            audit_path = resolve_us_stage_artifact_contract_path(
                bundle_dir,
                "09_validation_benchmarking",
                "policyengine_native_audit",
            )
            audit_payload = json.loads(audit_path.read_text()) if audit_path.exists() else None
            verdict_hints = dict((audit_payload or {}).get("verdictHints", {}))
            support_summary = dict((audit_payload or {}).get("supportAuditSummary", {}))

            rows.append(
                {
                    "artifactRoot": root_key,
                    "artifactPath": str(bundle_dir.relative_to(artifact_root)),
                    "lossDelta": float(
                        summary.get("enhanced_cps_native_loss_delta", 0.0)
                    ),
                    "candidateBeatsBaseline": bool(
                        summary.get("candidate_beats_baseline", False)
                    ),
                    "largestRegressingFamily": largest_family.get("family"),
                    "largestRegressingFamilyDelta": (
                        float(largest_family.get("loss_contribution_delta", 0.0))
                        if largest_family
                        else None
                    ),
                    "top3Families": top3_families,
                    "largestRegressingTarget": verdict_hints.get(
                        "largestRegressingTarget"
                    ),
                    "missingStoredCriticalInputs": list(
                        support_summary.get("missingStoredCriticalInputs", ())
                    ),
                    "auditAvailable": audit_payload is not None,
                }
            )

            if largest_family.get("family"):
                largest_family_counts[str(largest_family["family"])] += 1

            for rank, family in enumerate(top3_families, start=1):
                if not family:
                    continue
                family_str = str(family)
                top3_family_counts[family_str] += 1
                family_rank_counts[family_str][rank] += 1
                family_counts_by_root[root_key][family_str] += 1

            target_name = verdict_hints.get("largestRegressingTarget")
            if target_name:
                target_counts_from_audits[str(target_name)] += 1

            for variable in support_summary.get("missingStoredCriticalInputs", ()):
                missing_critical_inputs_counts[str(variable)] += 1

    rows.sort(key=lambda row: (-float(row["lossDelta"]), row["artifactRoot"], row["artifactPath"]))
    best_rows = sorted(
        rows,
        key=lambda row: (float(row["lossDelta"]), row["artifactRoot"], row["artifactPath"]),
    )

    return {
        "artifactRoots": [str(root) for root in normalized_roots],
        "totalScoredRuns": len(rows),
        "totalAuditedRuns": sum(1 for row in rows if bool(row["auditAvailable"])),
        "largestFamilyCounts": [
            {"family": family, "count": count}
            for family, count in _sorted_counter_items(largest_family_counts)
        ],
        "top3FamilyCounts": [
            {
                "family": family,
                "top3Count": top3_family_counts[family],
                "rank1Count": family_rank_counts[family][1],
                "rank2Count": family_rank_counts[family][2],
                "rank3Count": family_rank_counts[family][3],
            }
            for family, _count in _sorted_counter_items(top3_family_counts)
        ],
        "familyCountsByRoot": {
            root: [
                {"family": family, "count": count}
                for family, count in _sorted_counter_items(counter)
            ]
            for root, counter in sorted(family_counts_by_root.items())
        },
        "targetCountsFromAudits": [
            {"target": target, "count": count}
            for target, count in _sorted_counter_items(target_counts_from_audits)
        ],
        "missingCriticalInputsCounts": [
            {"variable": variable, "count": count}
            for variable, count in _sorted_counter_items(missing_critical_inputs_counts)
        ],
        "worstRuns": rows[:top_k],
        "bestRuns": best_rows[:top_k],
    }


def _iter_scored_bundle_dirs(artifact_root: Path) -> tuple[Path, ...]:
    scores_hint = get_us_stage_artifact_contract(
        "09_validation_benchmarking",
        "policyengine_native_scores",
    ).path_hint
    dataset_hint = get_us_stage_artifact_contract(
        "08_dataset_assembly",
        "policyengine_dataset",
    ).path_hint
    if scores_hint is None or dataset_hint is None:
        return ()
    return tuple(
        sorted(
            path.parent
            for path in artifact_root.rglob(scores_hint)
            if (path.parent / dataset_hint).exists()
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for PE-native regression summary over saved artifacts."""

    parser = argparse.ArgumentParser(
        description="Summarize recurring PE-native regression families for saved US artifacts.",
    )
    parser.add_argument(
        "artifact_roots",
        nargs="+",
        help="One or more artifact roots to scan.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of best/worst runs to include in the output.",
    )
    args = parser.parse_args(argv)

    payload = summarize_us_pe_native_regressions(args.artifact_roots, top_k=args.top_k)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
