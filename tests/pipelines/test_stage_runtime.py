"""Tests for live US stage runtime manifest updates."""

import json
import signal
import time

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
from microplex_us.pipelines.stage_runtime import (
    USStageInterruptedError,
    USStageRuntimeWriter,
    runtime_stage_interrupt_handler,
)


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


def test_runtime_writer_finalize_preserves_completed_stage_lifecycle(tmp_path):
    writer = USStageRuntimeWriter(
        tmp_path,
        manifest_payload={
            "config": {"calibration_backend": "none"},
            "artifacts": {"manifest": "manifest.json"},
        },
    )
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
    stage1_path = tmp_path / "stage_artifacts" / "manifests" / "01_run_profile.json"
    before = json.loads(stage1_path.read_text())

    writer.finalize_from_artifact_manifest(
        {
            "config": {"calibration_backend": "none"},
            "artifacts": {"manifest": "manifest.json"},
            "synthesis": {"source_names": ["unit"]},
        }
    )

    after = json.loads(stage1_path.read_text())
    assert after["lifecycleStatus"] == "complete"
    assert after["startedAt"] == before["startedAt"]
    assert after["updatedAt"] == before["updatedAt"]
    assert after["completedAt"] == before["completedAt"]
    assert after["events"] == before["events"]


def test_runtime_writer_finalize_preserves_live_stage_outputs(tmp_path):
    writer = USStageRuntimeWriter(
        tmp_path,
        manifest_payload={
            "config": {"calibration_backend": "none"},
            "artifacts": {"manifest": "manifest.json"},
        },
    )
    writer.start_stage("01_run_profile")
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
    writer.record_metadata(
        "02_source_loading",
        "sourceLoadingProgress",
        {"providerCount": 1, "completedProviderCount": 1},
    )
    writer.complete_stage(
        USSourceLoadingOutputs(
            observation_frame_summary={
                "source_count": 1,
                "frames": [{"source": "unit", "rows": {"household": 2}}],
            },
            source_descriptors=("unit",),
            source_relationships={
                "household_person": [
                    {"cardinality": RelationshipCardinality.ONE_TO_MANY}
                ]
            },
            diagnostics=_diagnostics("02_source_loading"),
        )
    )
    stage2_path = tmp_path / "stage_artifacts" / "manifests" / "02_source_loading.json"
    before = json.loads(stage2_path.read_text())

    writer.finalize_from_artifact_manifest(
        {
            "config": {"calibration_backend": "none"},
            "artifacts": {"manifest": "manifest.json"},
            "synthesis": {"source_names": ["unit"]},
        }
    )

    after = json.loads(stage2_path.read_text())
    assert after["outputs"]["observation_frame_summary"] == before["outputs"][
        "observation_frame_summary"
    ]
    assert after["outputs"]["source_relationships"] == before["outputs"][
        "source_relationships"
    ]
    assert after["metadata"]["sourceLoadingProgress"] == {
        "providerCount": 1,
        "completedProviderCount": 1,
    }


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


def test_runtime_writer_heartbeat_updates_manifest(tmp_path):
    writer = USStageRuntimeWriter(tmp_path)
    writer.start_stage("01_run_profile")

    payload = writer.heartbeat_stage("01_run_profile", {"provider": "unit"})

    assert payload["metadata"]["lastHeartbeat"]["provider"] == "unit"
    assert payload["updatedAt"] is not None
    assert [event["event"] for event in payload["events"]] == [
        "stage_started",
        "stage_heartbeat",
    ]


def test_runtime_writer_auto_heartbeat_updates_while_context_is_open(tmp_path):
    writer = USStageRuntimeWriter(tmp_path)
    writer.start_stage("01_run_profile")
    stage1_path = tmp_path / "stage_artifacts" / "manifests" / "01_run_profile.json"

    with writer.auto_heartbeat(
        "01_run_profile",
        interval_seconds=0.01,
        details_factory=lambda: {"provider": "unit"},
    ):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            payload = json.loads(stage1_path.read_text())
            if any(event["event"] == "stage_heartbeat" for event in payload["events"]):
                break
            time.sleep(0.01)

    payload = json.loads(stage1_path.read_text())
    assert payload["metadata"]["lastHeartbeat"]["provider"] == "unit"
    assert any(event["event"] == "stage_heartbeat" for event in payload["events"])


def test_runtime_writer_complete_preserves_running_metadata(tmp_path):
    writer = USStageRuntimeWriter(tmp_path)
    writer.start_stage("01_run_profile")
    writer.record_metadata(
        "01_run_profile",
        "sourceLoadingProgress",
        {"currentProviderName": "unit"},
    )

    payload = writer.complete_stage(
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

    assert payload["metadata"]["sourceLoadingProgress"] == {
        "currentProviderName": "unit"
    }


def test_runtime_writer_best_effort_failure_does_not_raise_handler_error(
    tmp_path,
    monkeypatch,
):
    writer = USStageRuntimeWriter(tmp_path)

    def broken_fail_stage(*args, **kwargs):
        raise RuntimeError("handler broke")

    monkeypatch.setattr(writer, "fail_stage", broken_fail_stage)

    writer.best_effort_fail_stage("01_run_profile", ValueError("original"))


def test_runtime_interrupt_handler_records_failed_stage(tmp_path):
    writer = USStageRuntimeWriter(tmp_path)
    writer.start_stage("01_run_profile")

    with pytest.raises(USStageInterruptedError) as exc_info:
        with runtime_stage_interrupt_handler(writer):
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)

    assert exc_info.value.signal_number == signal.SIGTERM
    stage1 = json.loads(
        (
            tmp_path / "stage_artifacts" / "manifests" / "01_run_profile.json"
        ).read_text()
    )
    assert stage1["lifecycleStatus"] == "failed"
    assert stage1["failure"]["errorType"] == "USStageInterruptedError"
    assert stage1["metadata"]["interrupted"] is True
    assert stage1["metadata"]["signalName"] == "SIGTERM"


def test_runtime_interrupt_handler_continues_when_signal_handlers_unavailable(
    tmp_path,
    monkeypatch,
):
    writer = USStageRuntimeWriter(tmp_path)
    entered = False

    def unavailable_signal_handler(*args, **kwargs):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(signal, "signal", unavailable_signal_handler)

    with runtime_stage_interrupt_handler(writer):
        entered = True

    assert entered is True
