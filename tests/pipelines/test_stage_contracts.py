"""Tests for canonical US pipeline stage contracts."""

import pytest

from microplex_us.pipelines.stage_contracts import (
    canonicalize_us_pipeline_stage_id,
    config_keys_for_us_pipeline_stage,
    default_us_pipeline_stage_contracts,
    get_us_pipeline_stage_contract,
    get_us_stage_artifact_contract,
    resolve_us_stage_artifact_contract_path,
    serialize_us_pipeline_stage_contracts,
)


def test_default_us_pipeline_stage_contracts_are_stable_and_complete():
    contracts = default_us_pipeline_stage_contracts()

    assert [contract.step for contract in contracts] == [
        f"{index:02d}" for index in range(1, 10)
    ]
    assert [contract.id for contract in contracts] == [
        "01_run_profile",
        "02_source_loading",
        "03_source_planning",
        "04_seed_scaffold",
        "05_donor_integration_synthesis",
        "06_policyengine_entities",
        "07_calibration",
        "08_dataset_assembly",
        "09_validation_benchmarking",
    ]
    assert len({contract.id for contract in contracts}) == 9
    for contract in contracts:
        assert contract.title
        assert contract.purpose
        assert contract.consumes
        assert contract.produces
        assert contract.inputs
        assert contract.outputs
        assert contract.diagnostics
        assert contract.validations
        assert contract.resume_mode
        for artifact in contract.artifacts:
            assert artifact.format
            assert artifact.hash_mode
            if artifact.resume_role is not None:
                assert artifact.resume_role in {
                    "diagnostic",
                    "manual_replay",
                    "manual_resume",
                    "post_artifact_evidence",
                }


def test_get_us_pipeline_stage_contract_returns_one_stage():
    contract = get_us_pipeline_stage_contract("08_dataset_assembly")

    assert contract.step == "08"
    assert contract.title == "Dataset assembly and publication"


def test_get_us_pipeline_stage_contract_rejects_unknown_stage():
    with pytest.raises(KeyError, match="Unknown US pipeline stage"):
        get_us_pipeline_stage_contract("bogus")


def test_serialize_us_pipeline_stage_contracts_is_json_ready():
    payload = serialize_us_pipeline_stage_contracts()

    assert payload["schemaVersion"] == 1
    assert payload["contractVersion"] == "us-runtime-stages-v2"
    assert len(payload["stages"]) == 9
    assert payload["stages"][5]["id"] == "06_policyengine_entities"
    assert payload["stages"][5]["inputs"][0]["artifact_key"] == "synthetic_data"
    assert payload["stages"][7]["artifacts"][-1]["key"] == "conditional_readiness"
    assert payload["stages"][7]["artifacts"][-1]["format"] == "json"


def test_canonicalize_us_pipeline_stage_id_maps_legacy_runtime_ids():
    assert (
        canonicalize_us_pipeline_stage_id("policyengine_materialization")
        == "06_policyengine_entities"
    )
    assert canonicalize_us_pipeline_stage_id("target_build") == "07_calibration"
    assert canonicalize_us_pipeline_stage_id("finalization") == "08_dataset_assembly"
    assert canonicalize_us_pipeline_stage_id("benchmark") == "09_validation_benchmarking"
    assert canonicalize_us_pipeline_stage_id("08_dataset_assembly") == "08_dataset_assembly"
    assert canonicalize_us_pipeline_stage_id("custom-stage") == "custom-stage"


def test_stage_contracts_expose_config_scope_and_canonical_paths(tmp_path):
    assert "n_synthetic" in config_keys_for_us_pipeline_stage(
        "05_donor_integration_synthesis"
    )
    assert resolve_us_stage_artifact_contract_path(
        tmp_path,
        "08_dataset_assembly",
        "artifact_inventory",
    ) == (tmp_path / "stage_artifacts" / "artifact_inventory.json")


def test_required_stage_inputs_reference_prior_outputs_and_artifacts():
    contracts = default_us_pipeline_stage_contracts()
    contracts_by_id = {contract.id: contract for contract in contracts}

    for contract in contracts:
        for resource in contract.inputs:
            if not resource.required:
                continue
            if resource.kind == "stage_output":
                assert resource.stage_id is not None
                upstream = contracts_by_id[resource.stage_id]
                assert any(
                    output.key == resource.key
                    and output.kind == "stage_output"
                    and output.stage_id == resource.stage_id
                    for output in upstream.outputs
                )
            if resource.kind == "artifact":
                assert resource.stage_id is not None
                artifact = get_us_stage_artifact_contract(
                    resource.stage_id,
                    resource.artifact_key or resource.key,
                )
                assert artifact.required


def test_source_planning_seam_exposes_descriptors_for_stage3():
    stage2 = get_us_pipeline_stage_contract("02_source_loading")
    stage3 = get_us_pipeline_stage_contract("03_source_planning")

    stage2_outputs = {resource.key for resource in stage2.outputs}
    stage3_inputs = {resource.key for resource in stage3.inputs}

    assert "source_descriptors" in stage2_outputs
    assert "source_descriptors" in stage3_inputs
