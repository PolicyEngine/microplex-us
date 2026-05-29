"""Canonical site-snapshot helpers for saved US microplex artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from microplex.targets import assert_valid_benchmark_artifact_manifest

from microplex_us.pipelines.data_flow_snapshot import (
    require_saved_us_microplex_data_flow_snapshot,
    write_us_microplex_data_flow_snapshot,
)
from microplex_us.pipelines.stage_contracts import (
    resolve_us_stage_artifact_contract_path,
)

FOCUS_TAG_PRIORITY: tuple[str, ...] = (
    "state",
    "local",
    "parity",
    "all_targets",
    "national",
    "tax",
    "benchmark",
)


def build_us_microplex_site_snapshot(
    artifact_dir: str | Path,
    *,
    snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one site-facing snapshot from a versioned US artifact bundle."""
    artifact_root = Path(artifact_dir)
    manifest = json.loads((artifact_root / "manifest.json").read_text())
    artifacts = dict(manifest.get("artifacts", {}))
    assert_valid_benchmark_artifact_manifest(
        manifest,
        artifact_dir=artifact_root,
        manifest_path=artifact_root / "manifest.json",
        summary_section="policyengine_harness",
        required_artifact_keys=(
            "seed_data",
            "synthetic_data",
            "calibrated_data",
            "targets",
            "policyengine_harness",
        ),
        required_summary_keys=(
            "candidate_mean_abs_relative_error",
            "baseline_mean_abs_relative_error",
            "mean_abs_relative_error_delta",
        ),
    )
    harness_path = _resolve_manifest_artifact_path(
        artifact_root,
        artifacts,
        "policyengine_harness",
        stage_id="09_validation_benchmarking",
    )
    harness = json.loads(harness_path.read_text())
    summary = dict(harness.get("summary", {}))
    tag_summaries = {
        key: dict(value)
        for key, value in dict(summary.get("tag_summaries", {})).items()
    }
    focus_tag = _select_focus_tag(tag_summaries)
    focus_summary = tag_summaries.get(focus_tag, summary)
    synthesis = dict(manifest.get("synthesis", {}))
    calibration = dict(manifest.get("calibration", {}))
    config = dict(manifest.get("config", {}))
    data_flow_path = _resolve_manifest_artifact_path(
        artifact_root,
        artifacts,
        "data_flow_snapshot",
        stage_id="08_dataset_assembly",
    )
    data_flow_snapshot = require_saved_us_microplex_data_flow_snapshot(artifact_root)

    source_artifact = {
        "artifactRef": _artifact_ref(artifact_root),
        "manifestFile": "manifest.json",
        "harnessFile": _artifact_path_for_manifest(artifact_root, harness_path),
        "dataFlowFile": _artifact_path_for_manifest(artifact_root, data_flow_path),
        "versionId": artifact_root.name,
    }
    if snapshot_path is not None:
        source_artifact["artifactPath"] = _artifact_path_from_snapshot(
            artifact_root,
            Path(snapshot_path),
        )

    return {
        "generatedAt": manifest.get("created_at"),
        "sourceArtifact": source_artifact,
        "currentRun": {
            "id": artifact_root.name,
            "benchmarkTag": focus_tag,
            "nSynthetic": config.get("n_synthetic"),
            "scaffoldSource": synthesis.get("scaffold_source"),
            "candidateMeanAbsRelativeError": focus_summary.get(
                "candidate_mean_abs_relative_error"
            ),
            "baselineMeanAbsRelativeError": focus_summary.get(
                "baseline_mean_abs_relative_error"
            ),
            "meanAbsRelativeErrorDelta": focus_summary.get(
                "mean_abs_relative_error_delta"
            ),
            "candidateCompositeParityLoss": focus_summary.get(
                "candidate_composite_parity_loss"
            ),
            "baselineCompositeParityLoss": focus_summary.get(
                "baseline_composite_parity_loss"
            ),
            "targetWinRate": focus_summary.get("target_win_rate"),
            "sliceWinRate": focus_summary.get("slice_win_rate"),
            "supportedTargetRate": focus_summary.get("supported_target_rate"),
            "calibration": {
                "loadedTargets": calibration.get("n_loaded_targets"),
                "supportedTargets": calibration.get("n_supported_targets"),
                "converged": calibration.get("converged"),
                "weightCollapseSuspected": calibration.get(
                    "weight_collapse_suspected"
                ),
                "householdEffectiveSampleSize": _nested_metric(
                    calibration,
                    "household_weight_diagnostics",
                    "effective_sample_size",
                ),
                "personEffectiveSampleSize": _nested_metric(
                    calibration,
                    "person_weight_diagnostics",
                    "effective_sample_size",
                ),
                "householdTinyWeightShare": _nested_metric(
                    calibration,
                    "household_weight_diagnostics",
                    "tiny_share",
                ),
                "personTinyWeightShare": _nested_metric(
                    calibration,
                    "person_weight_diagnostics",
                    "tiny_share",
                ),
            },
            "supportProxies": dict(
                synthesis.get(
                    "state_program_support_proxies",
                    {"available": [], "missing": []},
                )
            ),
            "availableTags": list(tag_summaries.keys()),
        },
        "summary": summary,
        "tagSummaries": tag_summaries,
        "parityScorecard": {
            key: dict(value)
            for key, value in dict(summary.get("parity_scorecard", {})).items()
        },
        "attributeCellSummaries": {
            key: dict(value)
            for key, value in dict(summary.get("attribute_cell_summaries", {})).items()
        },
        "dataFlow": data_flow_snapshot,
    }


def write_us_microplex_site_snapshot(
    artifact_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Write the canonical US site snapshot JSON for one saved artifact bundle."""
    artifact_root = Path(artifact_dir)
    write_us_microplex_data_flow_snapshot(
        artifact_root,
        resolve_us_stage_artifact_contract_path(
            artifact_root,
            "08_dataset_assembly",
            "data_flow_snapshot",
        ),
    )
    snapshot = build_us_microplex_site_snapshot(
        artifact_root,
        snapshot_path=output_path,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    return destination


def _artifact_ref(artifact_root: Path) -> str:
    for parent in artifact_root.parents:
        if parent.name == "artifacts":
            return str(artifact_root.relative_to(parent))
    return artifact_root.name


def _resolve_manifest_artifact_path(
    artifact_root: Path,
    artifacts: dict[str, Any],
    artifact_key: str,
    *,
    stage_id: str,
) -> Path:
    declared = artifacts.get(artifact_key)
    if declared is not None:
        path = Path(str(declared))
        if not path.is_absolute():
            path = artifact_root / path
        return path
    return resolve_us_stage_artifact_contract_path(artifact_root, stage_id, artifact_key)


def _artifact_path_for_manifest(artifact_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(artifact_root))
    except ValueError:
        return str(path)


def _artifact_path_from_snapshot(artifact_root: Path, snapshot_path: Path) -> str:
    return os.path.relpath(artifact_root, snapshot_path.parent)


def _nested_metric(
    payload: dict[str, Any],
    section: str,
    key: str,
) -> float | int | None:
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        return None
    return section_payload.get(key)


def _select_focus_tag(tag_summaries: dict[str, dict[str, Any]]) -> str:
    for candidate in FOCUS_TAG_PRIORITY:
        if candidate in tag_summaries:
            return candidate
    if tag_summaries:
        return next(iter(tag_summaries))
    return "summary"
