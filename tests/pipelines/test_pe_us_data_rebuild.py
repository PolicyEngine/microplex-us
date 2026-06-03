"""Tests for the PE-US-data rebuild program spec."""

from __future__ import annotations

from microplex_us.data_sources.cps import CPSASECSourceProvider
from microplex_us.data_sources.donor_surveys import (
    ACSSourceProvider,
    SCFSourceProvider,
    SIPPSourceProvider,
)
from microplex_us.data_sources.puf import (
    PUF_UPRATING_MODE_PE_SOI,
    SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF,
    PUFSourceProvider,
)
from microplex_us.pipelines.pe_us_data_rebuild import (
    PEUSDataRebuildStatus,
    build_policyengine_us_data_rebuild_markdown,
    build_policyengine_us_data_rebuild_pipeline,
    default_policyengine_us_data_rebuild_config,
    default_policyengine_us_data_rebuild_program,
    default_policyengine_us_data_rebuild_source_providers,
)
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexPipeline,
)


def test_default_policyengine_us_data_rebuild_program_has_expected_core_stages() -> (
    None
):
    program = default_policyengine_us_data_rebuild_program()

    assert program.program_id == "pe-us-data-rebuild-v1"
    assert "change results only on the margin" in program.principle

    stage_ids = [stage.stage_id for stage in program.stages]
    assert len(stage_ids) == len(set(stage_ids))
    assert stage_ids == [
        "source-contracts",
        "cps-construction",
        "puf-ingestion-uprating",
        "extended-cps-qrf",
        "family-imputation-parity",
        "entity-export-parity",
        "weighting-backend",
        "targets-and-eval",
    ]

    weighting_stage = next(
        stage for stage in program.stages if stage.stage_id == "weighting-backend"
    )
    assert weighting_stage.current_status is PEUSDataRebuildStatus.CLOSE
    assert "policyengine_us_data.calibration.unified_calibration" in (
        weighting_stage.pe_owner_modules
    )


def test_build_policyengine_us_data_rebuild_markdown_mentions_parity_rule() -> None:
    markdown = build_policyengine_us_data_rebuild_markdown()

    assert "# Rebuild PE-US-data in Microplex" in markdown
    assert "Structural improvements are allowed" in markdown
    assert "### CPS construction parity" in markdown
    assert "`status`: `partial`" in markdown


def test_default_policyengine_us_data_rebuild_config_uses_incumbent_defaults() -> None:
    config = default_policyengine_us_data_rebuild_config(
        random_seed=123,
        cps_asec_source_year=2022,
    )

    assert isinstance(config, USMicroplexBuildConfig)
    assert config.synthesis_backend == "seed"
    assert config.calibration_backend == "entropy"
    assert config.policyengine_calibration_min_active_households == 20
    assert config.policyengine_calibration_deferred_stage_min_active_households == (
        10,
        1,
    )
    assert config.policyengine_calibration_deferred_stage_max_constraints == 24
    assert (
        config.policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error
        is None
    )
    assert config.policyengine_calibration_deferred_stage_top_family_count == 7
    assert config.policyengine_calibration_deferred_stage_top_geography_count == 4
    assert config.donor_imputer_backend == "qrf"
    assert config.donor_imputer_condition_selection == "pe_prespecified"
    assert config.donor_imputer_excluded_variables == ()
    assert config.puf_support_clone_enabled is True
    assert config.puf_support_clone_output_mode == "collapse_to_scaffold"
    assert config.policyengine_direct_override_variables == (
        "health_savings_account_ald",
        "non_sch_d_capital_gains",
    )
    assert config.policyengine_prefer_existing_tax_unit_ids is True
    assert config.random_seed == 123
    assert config.cps_asec_source_year == 2022


def test_default_policyengine_us_data_rebuild_config_respects_calibration_support_override() -> (
    None
):
    config = default_policyengine_us_data_rebuild_config(
        policyengine_calibration_min_active_households=5
    )

    assert config.policyengine_calibration_min_active_households == 5


def test_default_policyengine_us_data_rebuild_source_providers_use_pe_style_bundle() -> (
    None
):
    providers = default_policyengine_us_data_rebuild_source_providers(
        cps_source_year=2022,
        puf_target_year=2024,
        cps_download=False,
        puf_expand_persons=False,
        policyengine_us_data_python="/tmp/pe-python",
    )

    assert len(providers) == 6
    cps_provider, puf_provider = providers[:2]
    assert isinstance(cps_provider, CPSASECSourceProvider)
    assert cps_provider.year == 2022
    assert cps_provider.download is False
    assert isinstance(puf_provider, PUFSourceProvider)
    assert puf_provider.target_year == 2024
    assert puf_provider.cps_reference_year == 2022
    assert puf_provider.expand_persons is False
    assert puf_provider.uprating_mode == PUF_UPRATING_MODE_PE_SOI
    assert puf_provider.policyengine_us_data_python == "/tmp/pe-python"
    assert puf_provider.impute_pre_tax_contributions is False
    assert puf_provider.require_pre_tax_contribution_model is False
    assert puf_provider.social_security_split_strategy == (
        SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF
    )
    assert isinstance(providers[2], ACSSourceProvider)
    assert providers[2].year == 2024
    assert isinstance(providers[3], SIPPSourceProvider)
    assert providers[3].block == "tips"
    assert providers[3].target_year == 2024
    assert providers[3].policyengine_us_data_python == "/tmp/pe-python"
    assert isinstance(providers[4], SIPPSourceProvider)
    assert providers[4].block == "assets"
    assert providers[4].target_year == 2024
    assert providers[4].policyengine_us_data_python == "/tmp/pe-python"
    assert isinstance(providers[5], SCFSourceProvider)
    assert providers[5].target_year == 2024


def test_default_policyengine_us_data_rebuild_source_providers_keeps_acs_when_donor_surveys_disabled() -> (
    None
):
    # include_donor_surveys=False disables the SIPP/SCF donors, but the ACS donor is
    # always enabled (it supplies the rent / real_estate_taxes imputation), so it
    # remains alongside the CPS spine and PUF.
    providers = default_policyengine_us_data_rebuild_source_providers(
        include_donor_surveys=False,
        cps_download=False,
    )

    assert len(providers) == 3
    assert isinstance(providers[0], CPSASECSourceProvider)
    assert isinstance(providers[1], PUFSourceProvider)
    assert isinstance(providers[2], ACSSourceProvider)


def test_default_policyengine_us_data_rebuild_source_providers_can_include_donor_surveys() -> (
    None
):
    providers = default_policyengine_us_data_rebuild_source_providers(
        include_donor_surveys=True,
        cps_download=False,
    )

    assert len(providers) == 6
    assert isinstance(providers[0], CPSASECSourceProvider)
    assert isinstance(providers[1], PUFSourceProvider)
    assert isinstance(providers[2], ACSSourceProvider)
    assert isinstance(providers[3], SIPPSourceProvider)
    assert providers[3].block == "tips"
    assert providers[3].target_year == 2024
    assert isinstance(providers[4], SIPPSourceProvider)
    assert providers[4].block == "assets"
    assert providers[4].target_year == 2024
    assert isinstance(providers[5], SCFSourceProvider)
    assert providers[5].target_year == 2024


def test_build_policyengine_us_data_rebuild_pipeline_returns_configured_pipeline() -> (
    None
):
    pipeline = build_policyengine_us_data_rebuild_pipeline(
        random_seed=321,
        calibration_max_iter=77,
    )

    assert isinstance(pipeline, USMicroplexPipeline)
    assert pipeline.config.random_seed == 321
    assert pipeline.config.calibration_max_iter == 77
    assert pipeline.config.synthesis_backend == "seed"
    assert pipeline.config.calibration_backend == "entropy"
