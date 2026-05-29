"""Tests for typed US stage-run output manifests."""

import json

import pytest

from microplex_us.pipelines.stage_contracts import US_CANONICAL_STAGE_IDS
from microplex_us.pipelines.stage_run import (
    US_STAGE_OUTPUT_MANIFEST_TYPES,
    USArtifactRef,
    USAuxiliaryArtifact,
    USDiagnosticOutput,
    USRunProfileOutputs,
    USSourceLoadingOutputs,
    USStageInputOverride,
    USStageRunWriter,
    build_us_stage_output_manifests_from_artifact_manifest,
    parse_us_stage_input_override,
    write_us_stage_run_manifests_from_artifact_manifest,
)


def test_every_canonical_stage_has_typed_output_manifest():
    assert tuple(US_STAGE_OUTPUT_MANIFEST_TYPES) == US_CANONICAL_STAGE_IDS


def test_stage_run_writer_records_typed_stage_manifests(tmp_path):
    _write_artifact_bundle_files(tmp_path)
    manifest = _artifact_manifest()

    updated_manifest = write_us_stage_run_manifests_from_artifact_manifest(
        tmp_path,
        manifest,
    )

    assert (tmp_path / "manifest.json").exists()
    assert (
        tmp_path / "stage_artifacts" / "05_donor_integration_synthesis" / "manifest.json"
    ).exists()
    assert (
        tmp_path / "stage_artifacts" / "09_validation_benchmarking" / "manifest.json"
    ).exists()
    assert (
        updated_manifest["stage_output_manifests"]["07_calibration"]
        == "stage_artifacts/07_calibration/manifest.json"
    )
    stage5_manifest = json.loads(
        (
            tmp_path
            / "stage_artifacts"
            / "05_donor_integration_synthesis"
            / "manifest.json"
        ).read_text()
    )
    assert stage5_manifest["stageId"] == "05_donor_integration_synthesis"
    assert stage5_manifest["diagnostics"]
    assert stage5_manifest["inputStageManifest"] == (
        "stage_artifacts/04_seed_scaffold/manifest.json"
    )


def test_stage_run_writer_rejects_missing_diagnostics(tmp_path):
    writer = USStageRunWriter(tmp_path)
    output = USRunProfileOutputs(
        manifest=USArtifactRef(
            key="manifest",
            path="manifest.json",
            format="json",
            required=True,
            assume_exists=True,
        ),
        resolved_config={"n_synthetic": 10},
        provider_query_plan={"source_names": ["source"]},
    )

    with pytest.raises(ValueError, match="does not expose diagnostics"):
        writer.record_stage(output)


def test_stage_run_writer_requires_prior_stage_or_override(tmp_path):
    output = USSourceLoadingOutputs(
        observation_frame_summary={"source_count": 1},
        source_descriptors=("source",),
        source_relationships={"status": "summarized"},
        diagnostics={
            "stage_summary": USDiagnosticOutput(
                key="stage_summary",
                summary={"source_names": ["source"]},
            )
        },
    )

    with pytest.raises(ValueError, match="requires 01_run_profile"):
        USStageRunWriter(tmp_path).record_stage(output)

    writer = USStageRunWriter(
        tmp_path,
        allow_stage_input_overrides=True,
        stage_input_overrides=(
            USStageInputOverride(
                stage_id="02_source_loading",
                key="provider_query_plan",
                path="overrides/provider_query_plan.json",
                reason="test override",
            ),
        ),
    )
    writer.record_stage(output)
    assert writer.recorded_stages == (output,)


def test_stage_run_writer_rejects_arbitrary_input_manifest(tmp_path):
    arbitrary_manifest = tmp_path / "arbitrary.json"
    arbitrary_manifest.write_text("{}")
    output = USSourceLoadingOutputs(
        input_stage_manifest="arbitrary.json",
        observation_frame_summary={"source_count": 1},
        source_descriptors=("source",),
        source_relationships={"status": "summarized"},
        diagnostics={
            "stage_summary": USDiagnosticOutput(
                key="stage_summary",
                summary={"source_names": ["source"]},
            )
        },
    )

    with pytest.raises(ValueError, match="requires 01_run_profile"):
        USStageRunWriter(tmp_path).record_stage(output)


def test_stage_run_writer_rejects_empty_required_structured_outputs(tmp_path):
    output = USRunProfileOutputs(
        manifest=USArtifactRef(
            key="manifest",
            path="manifest.json",
            format="json",
            required=True,
            assume_exists=True,
        ),
        resolved_config={},
        provider_query_plan={"source_names": ["source"]},
        diagnostics={
            "stage_summary": USDiagnosticOutput(
                key="stage_summary",
                summary={"has_config": False},
            )
        },
    )

    with pytest.raises(ValueError, match="resolved_config"):
        USStageRunWriter(tmp_path).record_stage(output)


def test_stage_run_writer_rejects_undeclared_auxiliary_artifact(tmp_path):
    writer = USStageRunWriter(tmp_path)
    output = USRunProfileOutputs(
        manifest=USArtifactRef(
            key="manifest",
            path="manifest.json",
            format="json",
            required=True,
            assume_exists=True,
        ),
        resolved_config={"n_synthetic": 10},
        provider_query_plan={"source_names": ["source"]},
        diagnostics={
            "stage_summary": USDiagnosticOutput(
                key="stage_summary",
                summary={"has_config": True},
            )
        },
        auxiliary_artifacts={
            "not_declared": USAuxiliaryArtifact(
                key="not_declared",
                path="not_declared.json",
                format="json",
            )
        },
    )

    with pytest.raises(KeyError, match="not declared"):
        writer.update(output)


def test_parse_us_stage_input_override():
    override = parse_us_stage_input_override(
        "02_source_loading.provider_query_plan=overrides/provider_query_plan.json"
    )

    assert override == USStageInputOverride(
        stage_id="02_source_loading",
        key="provider_query_plan",
        path="overrides/provider_query_plan.json",
    )

    with pytest.raises(ValueError, match="STAGE_ID.KEY=PATH"):
        parse_us_stage_input_override("02_source_loading=missing-key")

    with pytest.raises(ValueError, match="Unknown US pipeline stage"):
        parse_us_stage_input_override("unknown_stage.provider_query_plan=override.json")


def test_build_stage_outputs_from_manifest_exposes_diagnostics(tmp_path):
    _write_artifact_bundle_files(tmp_path)
    outputs = build_us_stage_output_manifests_from_artifact_manifest(
        tmp_path,
        _artifact_manifest(),
    )

    assert len(outputs) == 9
    assert all(output.diagnostics for output in outputs)


def _write_artifact_bundle_files(root):
    for relative in (
        "seed_data.parquet",
        "synthetic_data.parquet",
        "calibrated_data.parquet",
        "targets.json",
        "policyengine_us.h5",
        "policyengine_native_scores.json",
        "source_weight_diagnostics.json",
        "stage_artifacts/03_source_planning/source_plan.json",
        "stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet",
        "stage_artifacts/06_policyengine_entities/metadata.json",
        "stage_artifacts/07_calibration/calibration_summary.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")


def _artifact_manifest():
    return {
        "created_at": "2026-05-30T00:00:00+00:00",
        "config": {"n_synthetic": 10, "calibration_backend": "entropy"},
        "rows": {"seed": 1, "synthetic": 1, "calibrated": 1},
        "synthesis": {
            "source_names": ["source"],
            "scaffold_source": "source",
            "backend": "seed",
        },
        "calibration": {"backend": "entropy", "converged": True},
        "policyengine_native_scores": {"enhanced_cps_native_loss_delta": -0.1},
        "artifacts": {
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "calibrated_data": "calibrated_data.parquet",
            "targets": "targets.json",
            "policyengine_dataset": "policyengine_us.h5",
            "policyengine_native_scores": "policyengine_native_scores.json",
            "source_weight_diagnostics": "source_weight_diagnostics.json",
            "source_plan": "stage_artifacts/03_source_planning/source_plan.json",
            "scaffold_seed_data": (
                "stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet"
            ),
            "policyengine_entity_tables": (
                "stage_artifacts/06_policyengine_entities/metadata.json"
            ),
            "calibration_summary": (
                "stage_artifacts/07_calibration/calibration_summary.json"
            ),
        },
    }
