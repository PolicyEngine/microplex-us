"""Tests for live US stage runtime manifest updates."""

import json

import pytest

from microplex_us.pipelines.stage_run import (
    USArtifactRef,
    USDiagnosticOutput,
    USRunProfileOutputs,
    USSourceLoadingOutputs,
)
from microplex_us.pipelines.stage_runtime import USStageRuntimeWriter


def _diagnostics(stage_id: str) -> dict[str, USDiagnosticOutput]:
    return {
        "stage_summary": USDiagnosticOutput(
            key="stage_summary",
            description=f"Summary for {stage_id}.",
            summary={"stage": stage_id},
        )
    }


def test_runtime_writer_requires_previous_stage_completion_to_start(tmp_path):
    writer = USStageRuntimeWriter(tmp_path)

    with pytest.raises(ValueError, match="01_run_profile to be complete"):
        writer.start_stage("02_source_loading")

    writer.start_stage("01_run_profile")

    with pytest.raises(ValueError, match="01_run_profile to be complete"):
        writer.start_stage("02_source_loading")


def test_runtime_writer_completes_stage_and_exposes_lifecycle(tmp_path):
    writer = USStageRuntimeWriter(tmp_path)
    writer.start_stage("01_run_profile", metadata={"profile": "test"})
    writer.complete_stage(
        USRunProfileOutputs(
            manifest=USArtifactRef(
                key="manifest",
                path="manifest.json",
                format="json",
                required=True,
                assume_exists=True,
            ),
            resolved_config={"calibration_backend": "none"},
            provider_query_plan={"source_names": ["unit"]},
            diagnostics=_diagnostics("01_run_profile"),
        )
    )

    writer.start_stage("02_source_loading")
    writer.complete_stage(
        USSourceLoadingOutputs(
            observation_frame_summary={"source_count": 1},
            source_descriptors=("unit",),
            source_relationships={"household_person": "ok"},
            diagnostics=_diagnostics("02_source_loading"),
        )
    )

    stage2_path = tmp_path / "stage_artifacts" / "manifests" / "02_source_loading.json"
    stage2 = json.loads(stage2_path.read_text())
    aggregate = json.loads((tmp_path / "stage_manifest.json").read_text())
    aggregate_stage2 = {stage["id"]: stage for stage in aggregate["stages"]}[
        "02_source_loading"
    ]

    assert stage2["lifecycleStatus"] == "complete"
    assert stage2["inputStageManifest"] == (
        "stage_artifacts/manifests/01_run_profile.json"
    )
    assert aggregate_stage2["lifecycleStatus"] == "complete"
    assert aggregate_stage2["outputManifest"] == (
        "stage_artifacts/manifests/02_source_loading.json"
    )
    assert aggregate_stage2["completedAt"] is not None
    assert [event["event"] for event in stage2["events"]] == [
        "stage_started",
        "stage_completed",
    ]


def test_runtime_writer_update_writes_json_artifact_reference(tmp_path):
    writer = USStageRuntimeWriter(tmp_path)
    payload = writer.record_output(
        "03_source_planning",
        "source_plan",
        {"scaffoldSource": "cps"},
        path="stage_artifacts/03_source_planning/source_plan.json",
    )

    source_plan_path = (
        tmp_path / "stage_artifacts" / "03_source_planning" / "source_plan.json"
    )

    assert json.loads(source_plan_path.read_text()) == {"scaffoldSource": "cps"}
    assert payload["outputs"]["source_plan"]["path"] == (
        "stage_artifacts/03_source_planning/source_plan.json"
    )
    assert payload["outputs"]["source_plan"]["exists"] is True
