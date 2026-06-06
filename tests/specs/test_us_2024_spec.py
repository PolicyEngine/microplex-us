from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from microplex.spec import DEMOGRAPHICS_TOKEN, ImputationOrder, SpineMethod, load_spec

from microplex_us.pipelines.us import (
    PUF_SUPPORT_CLONE_IMPUTED_VARIABLES,
    PUF_SUPPORT_CLONE_OVERRIDDEN_VARIABLES,
    PUF_SUPPORT_CLONE_SPECIAL_VARIABLES,
)
from microplex_us.variables import PE_STYLE_PUF_IRS_DEMOGRAPHIC_PREDICTORS

SPEC_PATH = Path(str(files("microplex_us.specs").joinpath("us-2024.yaml")))


def _spec():
    return load_spec(SPEC_PATH)


def test_us_2024_spec_loads_and_names_release_surface() -> None:
    spec = _spec()

    assert spec.meta.country == "us"
    assert spec.meta.model_year == 2024
    assert spec.meta.policyengine_model == "policyengine-us"
    assert spec.sources["cps_asec"].dataset == "cps_asec_2025_calendar_2024"
    assert spec.sources["puf"].dataset == "puf_2024"
    assert set(spec.sources) == {"cps_asec", "puf", "acs", "sipp", "scf"}

    assert spec.targets is not None
    assert spec.targets.arch.country == "us"
    assert spec.targets.arch.model_year == 2024
    assert spec.targets.arch.target_profile == "pe_native_broad"
    assert (
        spec.targets.arch.resolved_calibration_target_profile
        == "pe_native_broad_source_backed"
    )
    assert spec.calibrate is not None
    assert spec.calibrate.loss == "pe_native_bucketed_huber_v1"
    assert spec.calibrate.method.value == "apg"


def test_us_2024_spec_declares_ecps_clone_spine() -> None:
    spec = _spec()

    assert spec.spine.base == "cps_asec"
    assert spec.spine.method is SpineMethod.CLONE
    assert spec.spine.clone.seed == 20260529
    assert spec.spine.passthrough_half.name == "cps_keep"
    assert spec.spine.passthrough_half.keep == "all"
    assert spec.spine.synthetic_half.name == "synthetic_puf"
    assert spec.spine.synthetic_half.strip_to == [DEMOGRAPHICS_TOKEN]


def test_us_2024_spec_declares_demographic_only_puf_synthesis() -> None:
    spec = _spec()
    all_puf_vars = list(
        PUF_SUPPORT_CLONE_IMPUTED_VARIABLES + PUF_SUPPORT_CLONE_SPECIAL_VARIABLES
    )

    synthetic, cps_fill, cps_override = spec.imputation

    assert synthetic.onto == "synthetic_puf"
    assert synthetic.from_ == "puf"
    assert synthetic.vars == all_puf_vars
    assert synthetic.condition_on == [DEMOGRAPHICS_TOKEN]
    assert synthetic.order is ImputationOrder.SPINE_FIRST
    assert synthetic.synthesize is True

    assert cps_fill.onto == "cps_keep"
    assert cps_fill.from_ == "puf"
    assert cps_fill.vars == all_puf_vars
    assert cps_fill.condition_on == [DEMOGRAPHICS_TOKEN]
    assert cps_fill.synthesize is False

    assert cps_override.onto == "cps_keep"
    assert cps_override.from_ == "puf"
    assert cps_override.vars == list(PUF_SUPPORT_CLONE_OVERRIDDEN_VARIABLES)
    assert cps_override.condition_on == [DEMOGRAPHICS_TOKEN]
    assert cps_override.synthesize is True

    assert set(PUF_SUPPORT_CLONE_OVERRIDDEN_VARIABLES).issubset(
        PUF_SUPPORT_CLONE_IMPUTED_VARIABLES
    )
    assert "employment_income" in synthetic.vars
    assert "employment_income" not in cps_override.vars
    assert "employment_income" not in synthetic.condition_on
    assert tuple(PE_STYLE_PUF_IRS_DEMOGRAPHIC_PREDICTORS) == (
        "age",
        "is_male",
        "tax_unit_is_joint",
        "tax_unit_count_dependents",
        "is_tax_unit_head",
        "is_tax_unit_spouse",
        "is_tax_unit_dependent",
    )


def test_us_2024_spec_keeps_forbes_out_of_replication_baseline() -> None:
    assert "forbes" not in SPEC_PATH.read_text(encoding="utf-8").lower()
