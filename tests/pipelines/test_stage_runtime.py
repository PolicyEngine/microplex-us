"""Tests for live US stage runtime manifest updates."""

import json

import pytest
from microplex.core import RelationshipCardinality

from microplex_us.pipelines.stage_contracts import US_STAGE_CONTRACT_VERSION
from microplex_us.pipelines.stage_run import (
    USArtifactRef,
    USDiagnosticOutput,
    USRunProfileOutputs,
    USSourceLoadingOutputs,
    USStageInputOverride,
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


def test_runtime_writer_serializes_enum_outputs(tmp_path):
    writer = USStageRuntimeWriter(
        tmp_path,
        allow_stage_input_overrides=True,
        stage_input_overrides=(
            USStageInputOverride(
                stage_id="02_source_loading",
                key="provider_query_plan",
                path="overrides/provider_query_plan.json",
            ),
        ),
    )
    writer.start_stage("02_source_loading")
    writer.complete_stage(
        USSourceLoadingOutputs(
            observation_frame_summary={"source_count": 1},
            source_descriptors=("unit",),
            source_relationships={
                "unit": [{"cardinality": RelationshipCardinality.ONE_TO_MANY}]
            },
            diagnostics=_diagnostics("02_source_loading"),
        )
    )

    stage2 = json.loads(
        (
            tmp_path / "stage_artifacts" / "manifests" / "02_source_loading.json"
        ).read_text()
    )

    assert stage2["outputs"]["source_relationships"]["unit"][0]["cardinality"] == (
        "one_to_many"
    )


def test_runtime_writer_records_overrides_in_running_manifest(tmp_path):
    writer = USStageRuntimeWriter(
        tmp_path,
        allow_stage_input_overrides=True,
        stage_input_overrides=(
            USStageInputOverride(
                stage_id="02_source_loading",
                key="provider_query_plan",
                path="overrides/provider_query_plan.json",
                reason="unit test",
            ),
        ),
    )

    writer.start_stage("02_source_loading")
    stage2 = json.loads(
        (
            tmp_path / "stage_artifacts" / "manifests" / "02_source_loading.json"
        ).read_text()
    )

    assert stage2["inputOverrides"] == [
        {
            "stageId": "02_source_loading",
            "key": "provider_query_plan",
            "path": "overrides/provider_query_plan.json",
            "reason": "unit test",
        }
    ]


def test_runtime_writer_refreshes_root_manifest_on_stage_start(tmp_path):
    writer = USStageRuntimeWriter(tmp_path)

    writer.start_stage("01_run_profile")
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert manifest["stage_output_manifests"]["01_run_profile"] == (
        "stage_artifacts/manifests/01_run_profile.json"
    )


def test_runtime_writer_rejects_stale_complete_previous_manifest(tmp_path):
    writer = USStageRuntimeWriter(tmp_path)
    stage1_path = tmp_path / "stage_artifacts" / "manifests" / "01_run_profile.json"
    stage1_path.parent.mkdir(parents=True)
    stage1_path.write_text(
        json.dumps(
            {
                "stageId": "01_run_profile",
                "contractVersion": US_STAGE_CONTRACT_VERSION,
                "lifecycleStatus": "complete",
                "requiredOutputs": ["manifest"],
                "missingRequiredOutputs": ["manifest"],
                "outputs": {},
            }
        )
    )

    with pytest.raises(ValueError, match="missing required outputs"):
        writer.start_stage("02_source_loading")


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
