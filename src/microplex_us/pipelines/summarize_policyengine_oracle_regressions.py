"""Summarize recurring calibration-oracle regression families across saved artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from microplex_us.pipelines.stage_contracts import get_us_stage_artifact_contract


def _sorted_counter_items(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))


def summarize_us_policyengine_oracle_regressions(
    artifact_roots: list[str | Path] | tuple[str | Path, ...],
    *,
    loss_scope: str = "full_oracle",
    top_k: int = 10,
) -> dict[str, Any]:
    """Summarize recurring calibration-oracle regression families for saved artifacts."""

    rows: list[dict[str, Any]] = []
    largest_family_counts: Counter[str] = Counter()
    largest_geography_counts: Counter[str] = Counter()
    top3_family_counts: Counter[str] = Counter()
    top3_geography_counts: Counter[str] = Counter()
    family_rank_counts: dict[str, Counter[int]] = defaultdict(Counter)
    geography_rank_counts: dict[str, Counter[int]] = defaultdict(Counter)
    family_counts_by_root: dict[str, Counter[str]] = defaultdict(Counter)
    geography_counts_by_root: dict[str, Counter[str]] = defaultdict(Counter)

    normalized_roots = [Path(root) for root in artifact_roots]
    for artifact_root in normalized_roots:
        root_key = artifact_root.name
        for bundle_dir in _iter_oracle_bundle_dirs(artifact_root):
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            calibration = dict(manifest.get("calibration", {}))
            oracle_loss = dict(calibration.get("oracle_loss", {}))
            scope_summary = dict(oracle_loss.get(loss_scope, {}))
            family_ranking = [
                row
                for row in list(scope_summary.get("family_ranking", ()))
                if float(row.get("capped_sum_abs_relative_error", 0.0)) > 0.0
            ]
            geography_ranking = [
                row
                for row in list(scope_summary.get("geography_ranking", ()))
                if float(row.get("capped_sum_abs_relative_error", 0.0)) > 0.0
            ]

            largest_family = family_ranking[0] if family_ranking else {}
            largest_geography = geography_ranking[0] if geography_ranking else {}
            top3_families = [row.get("group") for row in family_ranking[:3] if row.get("group")]
            top3_geographies = [
                row.get("group") for row in geography_ranking[:3] if row.get("group")
            ]

            rows.append(
                {
                    "artifactRoot": root_key,
                    "artifactPath": str(bundle_dir.relative_to(artifact_root)),
                    "lossScope": loss_scope,
                    "scopeCappedLoss": scope_summary.get(
                        "capped_mean_abs_relative_error"
                    ),
                    "scopeLoss": scope_summary.get("mean_abs_relative_error"),
                    "activeSolveCappedLoss": calibration.get(
                        "active_solve_capped_mean_abs_relative_error"
                    ),
                    "nConstraints": calibration.get("n_constraints"),
                    "nSupportedTargets": calibration.get("n_supported_targets"),
                    "nUnsupportedTargets": calibration.get("n_unsupported_targets"),
                    "nCalibrationStagesApplied": calibration.get(
                        "n_calibration_stages_applied"
                    ),
                    "largestFamily": largest_family.get("group"),
                    "largestFamilyCappedLossShare": largest_family.get(
                        "capped_loss_share"
                    ),
                    "largestGeography": largest_geography.get("group"),
                    "largestGeographyCappedLossShare": largest_geography.get(
                        "capped_loss_share"
                    ),
                    "top3Families": top3_families,
                    "top3Geographies": top3_geographies,
                }
            )

            if largest_family.get("group"):
                largest_family_counts[str(largest_family["group"])] += 1
            if largest_geography.get("group"):
                largest_geography_counts[str(largest_geography["group"])] += 1

            for rank, family in enumerate(top3_families, start=1):
                family_str = str(family)
                top3_family_counts[family_str] += 1
                family_rank_counts[family_str][rank] += 1
                family_counts_by_root[root_key][family_str] += 1

            for rank, geography in enumerate(top3_geographies, start=1):
                geography_str = str(geography)
                top3_geography_counts[geography_str] += 1
                geography_rank_counts[geography_str][rank] += 1
                geography_counts_by_root[root_key][geography_str] += 1

    rows.sort(
        key=lambda row: (
            -float(row["scopeCappedLoss"] or 0.0),
            row["artifactRoot"],
            row["artifactPath"],
        )
    )
    best_rows = sorted(
        rows,
        key=lambda row: (
            float(row["scopeCappedLoss"] or 0.0),
            row["artifactRoot"],
            row["artifactPath"],
        ),
    )

    return {
        "artifactRoots": [str(root) for root in normalized_roots],
        "lossScope": loss_scope,
        "totalScoredRuns": len(rows),
        "largestFamilyCounts": [
            {"group": group, "count": count}
            for group, count in _sorted_counter_items(largest_family_counts)
        ],
        "largestGeographyCounts": [
            {"group": group, "count": count}
            for group, count in _sorted_counter_items(largest_geography_counts)
        ],
        "top3FamilyCounts": [
            {
                "group": group,
                "top3Count": top3_family_counts[group],
                "rank1Count": family_rank_counts[group][1],
                "rank2Count": family_rank_counts[group][2],
                "rank3Count": family_rank_counts[group][3],
            }
            for group, _count in _sorted_counter_items(top3_family_counts)
        ],
        "top3GeographyCounts": [
            {
                "group": group,
                "top3Count": top3_geography_counts[group],
                "rank1Count": geography_rank_counts[group][1],
                "rank2Count": geography_rank_counts[group][2],
                "rank3Count": geography_rank_counts[group][3],
            }
            for group, _count in _sorted_counter_items(top3_geography_counts)
        ],
        "familyCountsByRoot": {
            root: [
                {"group": group, "count": count}
                for group, count in _sorted_counter_items(counter)
            ]
            for root, counter in sorted(family_counts_by_root.items())
        },
        "geographyCountsByRoot": {
            root: [
                {"group": group, "count": count}
                for group, count in _sorted_counter_items(counter)
            ]
            for root, counter in sorted(geography_counts_by_root.items())
        },
        "worstRuns": rows[:top_k],
        "bestRuns": best_rows[:top_k],
    }


def _iter_oracle_bundle_dirs(artifact_root: Path) -> tuple[Path, ...]:
    dataset_hint = get_us_stage_artifact_contract(
        "08_dataset_assembly",
        "policyengine_dataset",
    ).path_hint
    if dataset_hint is None:
        return ()
    return tuple(
        sorted(
            path.parent
            for path in artifact_root.rglob("manifest.json")
            if (path.parent / dataset_hint).exists()
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for oracle-regression summaries over saved artifacts."""

    parser = argparse.ArgumentParser(
        description=(
            "Summarize recurring calibration-oracle regression families for saved US artifacts."
        ),
    )
    parser.add_argument(
        "artifact_roots",
        nargs="+",
        help="One or more artifact roots to scan.",
    )
    parser.add_argument(
        "--loss-scope",
        default="full_oracle",
        help="Which calibration oracle-loss scope to summarize.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of best/worst runs to include in the output.",
    )
    args = parser.parse_args(argv)

    payload = summarize_us_policyengine_oracle_regressions(
        args.artifact_roots,
        loss_scope=args.loss_scope,
        top_k=args.top_k,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
