"""Tests for typed US stage-run output manifests."""

import json
from dataclasses import fields

import pytest

from microplex_us.pipelines.stage_contracts import (
    US_CANONICAL_STAGE_IDS,
    get_us_pipeline_stage_contract,
)
from microplex_us.pipelines.stage_run import (
    US_STAGE_OUTPUT_MANIFEST_TYPES,
    USArtifactRef,
    USAuxiliaryArtifact,
    USDiagnosticOutput,
    USRunProfileOutputs,
    USSourceLoadingOutputs,
    USSourcePlanningOutputs,
    USStageInputOverride,
    USStageRunWriter,
    build_us_stage_output_manifests_from_artifact_manifest,
    parse_us_stage_input_override,
    write_us_stage_run_manifests_from_artifact_manifest,
)

_BASE_STAGE_MANIFEST_FIELDS = {
    "schema_version",
    "contract_version",
    "input_stage_manifest",
    "diagnostics",
    "auxiliary_artifacts",
    "metadata",
    "complete",
    "stage_id",
}


def test_every_canonical_stage_has_typed_output_manifest():
    assert tuple(US_STAGE_OUTPUT_MANIFEST_TYPES) == US_CANONICAL_STAGE_IDS


def test_stage_output_manifests_use_contract_outputs_as_required_source():
    for stage_id, manifest_type in US_STAGE_OUTPUT_MANIFEST_TYPES.items():
        contract = get_us_pipeline_stage_contract(stage_id)
        expected = tuple(
            resource.key for resource in contract.outputs if resource.required
        )
        output = manifest_type()

        assert output.required_output_keys() == expected
        assert set(expected) <= {item.name for item in fields(manifest_type)}


def test_stage_output_manifest_fields_are_declared_by_contracts():
    for stage_id, manifest_type in US_STAGE_OUTPUT_MANIFEST_TYPES.items():
        contract = get_us_pipeline_stage_contract(stage_id)
        contract_output_keys = {resource.key for resource in contract.outputs}
        contract_artifact_keys = {artifact.key for artifact in contract.artifacts}
        typed_output_fields = {
            item.name
            for item in fields(manifest_type)
            if item.name not in _BASE_STAGE_MANIFEST_FIELDS
        }

        assert contract_output_keys <= typed_output_fields
        assert typed_output_fields <= contract_output_keys | contract_artifact_keys


def test_stage_run_writer_records_typed_stage_manifests(tmp_path):
    _write_artifact_bundle_files(tmp_path)
    manifest = _artifact_manifest()

    updated_manifest = write_us_stage_run_manifests_from_artifact_manifest(
        tmp_path,
        manifest,
    )

    assert (tmp_path / "manifest.json").exists()
    assert (
        tmp_path
        / "stage_artifacts"
        / "manifests"
        / "05_donor_integration_synthesis.json"
    ).exists()
    assert (
        tmp_path / "stage_artifacts" / "manifests" / "09_validation_benchmarking.json"
    ).exists()
    assert (
        updated_manifest["stage_output_manifests"]["07_calibration"]
        == "stage_artifacts/manifests/07_calibration.json"
    )
    stage5_manifest = json.loads(
        (
            tmp_path
            / "stage_artifacts"
            / "manifests"
            / "05_donor_integration_synthesis.json"
        ).read_text()
    )
    assert stage5_manifest["stageId"] == "05_donor_integration_synthesis"
    assert stage5_manifest["diagnostics"]
    assert stage5_manifest["inputStageManifest"] == (
        "stage_artifacts/manifests/04_seed_scaffold.json"
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

    with pytest.raises(ValueError, match="require allow_stage_input_overrides"):
        USStageRunWriter(
            tmp_path,
            stage_input_overrides=(
                USStageInputOverride(
                    stage_id="02_source_loading",
                    key="provider_query_plan",
                    path="overrides/provider_query_plan.json",
                ),
            ),
        )

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


def test_stage_run_writer_requires_specific_input_override(tmp_path):
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

    writer = USStageRunWriter(
        tmp_path,
        allow_stage_input_overrides=True,
        stage_input_overrides=(
            USStageInputOverride(
                stage_id="02_source_loading",
                key="source_datasets",
                path="overrides/source_datasets.json",
            ),
        ),
    )

    with pytest.raises(ValueError, match="provider_query_plan"):
        writer.record_stage(output)


def test_stage_run_writer_validates_required_inputs_from_prior_manifest(tmp_path):
    writer = USStageRunWriter(tmp_path)
    writer.record_stage(
        USRunProfileOutputs(
            manifest=USArtifactRef(
                key="manifest",
                path="manifest.json",
                format="json",
                required=True,
                assume_exists=True,
            ),
            resolved_config={"n_synthetic": 10},
            provider_query_plan={},
            diagnostics={
                "stage_summary": USDiagnosticOutput(
                    key="stage_summary",
                    summary={"has_config": True},
                )
            },
            complete=False,
        )
    )
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

    with pytest.raises(ValueError, match="01_run_profile.provider_query_plan"):
        writer.record_stage(output)


def test_stage_run_writer_requires_prior_stage_even_without_stage_bound_inputs(
    tmp_path,
):
    output = USSourcePlanningOutputs(
        scaffold_selection={"scaffold_source": "source"},
        diagnostics={
            "stage_summary": USDiagnosticOutput(
                key="stage_summary",
                summary={"scaffold_source": "source"},
            )
        },
        complete=False,
    )

    with pytest.raises(ValueError, match="requires 02_source_loading"):
        USStageRunWriter(tmp_path).record_stage(output)


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

    with pytest.raises(ValueError, match="Unknown input override key"):
        parse_us_stage_input_override("02_source_loading.not_an_input=override.json")


def test_build_stage_outputs_from_manifest_exposes_diagnostics(tmp_path):
    _write_artifact_bundle_files(tmp_path)
    outputs = build_us_stage_output_manifests_from_artifact_manifest(
        tmp_path,
        _artifact_manifest(),
    )

    assert len(outputs) == 9
    assert all(output.diagnostics for output in outputs)
    stage6 = outputs[5]
    assert "policyengine_dataset" not in stage6.materialized_policyengine_inputs
    assert stage6.materialized_policyengine_inputs["tables"]["households"]["rows"] == 1


def test_build_stage_outputs_treats_missing_declared_dataset_as_incomplete(
    tmp_path,
):
    _write_artifact_bundle_files(tmp_path)
    (tmp_path / "policyengine_us.h5").unlink()

    outputs = build_us_stage_output_manifests_from_artifact_manifest(
        tmp_path,
        _artifact_manifest(),
    )

    stage8 = outputs[7]
    assert stage8.complete is False
    assert stage8.missing_required_outputs(tmp_path) == ("policyengine_dataset",)


def test_build_stage_outputs_hydrates_stage9_summary_from_validation_evidence(
    tmp_path,
):
    _write_artifact_bundle_files(tmp_path)
    evidence_path = _write_validation_evidence_manifest(tmp_path)
    manifest = _artifact_manifest()
    manifest.pop("policyengine_native_scores")
    manifest["artifacts"]["validation_evidence"] = str(
        evidence_path.relative_to(tmp_path)
    )

    outputs = build_us_stage_output_manifests_from_artifact_manifest(
        tmp_path,
        manifest,
    )

    stage9 = outputs[8]
    assert stage9.complete is True
    assert stage9.benchmark_summary == {
        "policyengine_native_scores": {
            "enhanced_cps_native_loss_delta": -0.1,
        }
    }
    assert stage9.diagnostics["stage_summary"].summary == stage9.benchmark_summary


def test_build_stage_outputs_does_not_complete_stage9_from_stale_evidence_summary(
    tmp_path,
):
    _write_artifact_bundle_files(tmp_path)
    evidence_path = _write_validation_evidence_manifest(tmp_path)
    (tmp_path / "policyengine_native_scores.json").unlink()
    manifest = _artifact_manifest()
    manifest.pop("policyengine_native_scores")
    manifest["artifacts"]["validation_evidence"] = str(
        evidence_path.relative_to(tmp_path)
    )

    outputs = build_us_stage_output_manifests_from_artifact_manifest(
        tmp_path,
        manifest,
    )

    stage9 = outputs[8]
    assert stage9.complete is False
    assert stage9.benchmark_summary == {}


def test_stage_run_writer_preserves_existing_validation_evidence_summary(
    tmp_path,
):
    _write_artifact_bundle_files(tmp_path)
    evidence_path = _write_validation_evidence_manifest(tmp_path)
    manifest = _artifact_manifest()
    manifest.pop("policyengine_native_scores")
    manifest["artifacts"]["validation_evidence"] = str(
        evidence_path.relative_to(tmp_path)
    )

    write_us_stage_run_manifests_from_artifact_manifest(tmp_path, manifest)

    stage9_manifest = json.loads(
        (
            tmp_path
            / "stage_artifacts"
            / "manifests"
            / "09_validation_benchmarking.json"
        ).read_text()
    )
    rewritten_evidence = json.loads(evidence_path.read_text())

    assert stage9_manifest["complete"] is True
    assert stage9_manifest["outputs"]["benchmark_summary"] == {
        "policyengine_native_scores": {
            "enhanced_cps_native_loss_delta": -0.1,
        }
    }
    assert rewritten_evidence["summaries"] == {
        "policyengine_native_scores": {
            "enhanced_cps_native_loss_delta": -0.1,
        }
    }
    assert any(
        record["key"] == "policyengine_native_scores"
        and record["path"] == "policyengine_native_scores.json"
        and record["exists"] is True
        for record in rewritten_evidence["evidence"]
    )


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
    (
        root / "stage_artifacts" / "06_policyengine_entities" / "metadata.json"
    ).write_text(
        json.dumps(
            {
                "format_version": 1,
                "stage": "post_microsim",
                "households": {"rows": 1, "columns": ["household_id"]},
                "persons": {"rows": 1, "columns": ["person_id"]},
            }
        )
    )


def _write_validation_evidence_manifest(root):
    evidence_path = (
        root
        / "stage_artifacts"
        / "09_validation_benchmarking"
        / "evidence_manifest.json"
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(
            {
                "formatVersion": 1,
                "stageId": "09_validation_benchmarking",
                "evidence": [
                    {
                        "key": "policyengine_native_scores",
                        "path": "policyengine_native_scores.json",
                        "exists": True,
                    }
                ],
                "summaries": {
                    "policyengine_native_scores": {
                        "enhanced_cps_native_loss_delta": -0.1,
                    }
                },
            }
        )
    )
    return evidence_path


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
