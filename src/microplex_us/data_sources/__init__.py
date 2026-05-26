"""Data-source convenience exports for microplex-us.

The package root resolves providers lazily so importing ``microplex_us.data_sources``
does not require optional survey, benchmark, or core integration dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _exports(module: str, names: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    return {name: (module, name) for name in names}


_EXPORTS: dict[str, tuple[str, str]] = {
    **_exports(
        "microplex_us.data_sources.cps",
        (
            "CPSDataset",
            "CPSASECSourceProvider",
            "CPSASECParquetSourceProvider",
            "download_cps_asec",
            "get_available_years",
            "PERSON_VARIABLES",
            "HOUSEHOLD_VARIABLES",
        ),
    ),
    "load_cps_asec_polars": (
        "microplex_us.data_sources.cps",
        "load_cps_asec",
    ),
    **_exports(
        "microplex_us.data_sources.cps_mappings",
        (
            "CoverageLevel",
            "CoverageGap",
            "VariableMapping",
            "map_age",
            "map_earned_income",
            "map_filing_status",
            "map_is_blind",
            "map_is_dependent",
            "map_ctc_qualifying_children",
            "map_agi_proxy",
            "map_household_size",
            "get_mapping_metadata",
            "get_all_mappings",
            "coverage_summary",
        ),
    ),
    **_exports(
        "microplex_us.data_sources.cps_transform",
        (
            "TransformedDataset",
            "transform_cps_to_policyengine",
        ),
    ),
    **_exports(
        "microplex_us.data_sources.donor_surveys",
        (
            "ACSSourceProvider",
            "DonorSurveyProviderSpec",
            "DonorSurveySourceProvider",
            "SIPPSourceProvider",
            "SIPPTipsSourceProvider",
            "SIPPAssetsSourceProvider",
            "SCFSourceProvider",
            "resolve_sipp_donor_survey_spec",
        ),
    ),
    **_exports(
        "microplex_us.data_sources.family_imputation_benchmark",
        (
            "DecomposableFamilyBenchmarkSpec",
            "FamilyImputationMethodBenchmark",
            "FamilyImputationBenchmarkResult",
            "benchmark_decomposable_family_imputers",
            "reconcile_component_predictions_to_total",
        ),
    ),
    **_exports(
        "microplex_us.data_sources.puf",
        (
            "load_puf",
            "PUFSourceProvider",
            "download_puf",
            "map_puf_variables",
            "uprate_puf",
            "expand_to_persons",
            "PUF_VARIABLE_MAP",
            "UPRATING_FACTORS",
            "PUF_EXCLUSIVE_VARS",
            "SHARED_VARS",
        ),
    ),
    **_exports(
        "microplex_us.data_sources.psid",
        (
            "PSIDDataset",
            "PSIDSourceProvider",
            "load_psid_panel",
            "extract_transition_rates",
            "get_age_specific_rates",
            "calibrate_marriage_rates",
            "calibrate_divorce_rates",
            "create_psid_fusion_source",
            "PSID_TO_MICROPLEX_VARS",
        ),
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve data-source exports on first access."""
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
