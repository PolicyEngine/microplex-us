"""Data-flow snapshot adapters for saved US stage manifests."""

from __future__ import annotations

from typing import Any, cast

from microplex_us.pipelines.stage_contracts import StageResumeMode
from microplex_us.pipelines.stage_manifest_types import (
    USDataFlowStageSummary,
    USStageManifest,
    USStageMetric,
    USStageStatus,
)


def stage_summary_for_data_flow_snapshot(
    stage_manifest: USStageManifest | dict[str, Any],
) -> list[USDataFlowStageSummary]:
    """Return site-facing stage summaries from a canonical stage manifest."""

    summaries: list[USDataFlowStageSummary] = []
    for stage in stage_manifest.get("stages", ()):
        if not isinstance(stage, dict):
            continue
        resume = stage.get("resume", {})
        summaries.append(
            {
                "id": str(stage.get("id", "")),
                "step": str(stage.get("step", "")),
                "title": str(stage.get("title", "")),
                "summary": str(stage.get("purpose", "")),
                "status": cast(USStageStatus, stage.get("status", "missing")),
                "metrics": cast(list[USStageMetric], list(stage.get("metrics", ()))),
                "outputs": _stage_output_paths_for_data_flow(stage),
                "resumeMode": cast(
                    StageResumeMode,
                    resume.get("mode", "none") if isinstance(resume, dict) else "none",
                ),
            }
        )
    return summaries


def _stage_output_paths_for_data_flow(stage: dict[str, Any]) -> list[str]:
    """Return artifact paths that a saved run actually referenced or produced."""

    outputs: list[str] = []
    for artifact in stage.get("artifacts", ()):
        if not isinstance(artifact, dict):
            continue
        path = artifact.get("path")
        if not path:
            continue
        if bool(artifact.get("exists")) or bool(artifact.get("referenced")):
            outputs.append(str(path))
    return outputs


__all__ = ["stage_summary_for_data_flow_snapshot"]
