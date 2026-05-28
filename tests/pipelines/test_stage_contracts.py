"""Tests for canonical US pipeline stage contracts."""

import pytest

from microplex_us.pipelines.stage_contracts import (
    canonicalize_us_pipeline_stage_id,
    default_us_pipeline_stage_contracts,
    get_us_pipeline_stage_contract,
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
        assert contract.diagnostics
        assert contract.validations
        assert contract.resume_mode


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
    assert payload["contractVersion"] == "us-runtime-stages-v1"
    assert len(payload["stages"]) == 9
    assert payload["stages"][5]["id"] == "06_policyengine_entities"


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
