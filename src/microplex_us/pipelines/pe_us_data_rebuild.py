"""Program spec for rebuilding the PE-US-data pipeline inside Microplex."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from microplex.core import SourceProvider

    from microplex_us.pipelines.us import USMicroplexBuildConfig, USMicroplexPipeline


class PEUSDataRebuildStatus(str, Enum):
    """Parity-rebuild status for one PE-US-data stage."""

    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    CLOSE = "close"
    EXACT = "exact"
    INTENTIONALLY_DIFFERENT = "intentionally_different"


@dataclass(frozen=True)
class PEUSDataRebuildStage:
    """One stage in the PE-US-data rebuild program."""

    stage_id: str
    title: str
    goal: str
    pe_owner_modules: tuple[str, ...]
    microplex_owner_modules: tuple[str, ...]
    parity_contract: str
    current_status: PEUSDataRebuildStatus
    notes: str
    next_steps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["current_status"] = self.current_status.value
        return payload


@dataclass(frozen=True)
class PEUSDataRebuildProgram:
    """Durable spec for the architecture-first PE-US-data rebuild track."""

    program_id: str
    title: str
    objective: str
    principle: str
    stages: tuple[PEUSDataRebuildStage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "title": self.title,
            "objective": self.objective,
            "principle": self.principle,
            "stages": [stage.to_dict() for stage in self.stages],
        }


def default_policyengine_us_data_rebuild_config(
    **overrides: Any,
) -> USMicroplexBuildConfig:
    """Return the incumbent-parity runtime config for the PE-US-data rebuild."""

    from microplex_us.pipelines.us import USMicroplexBuildConfig

    defaults = USMicroplexBuildConfig(
        synthesis_backend="seed",
        calibration_backend="entropy",
        policyengine_calibration_min_active_households=20,
        policyengine_calibration_deferred_stage_min_active_households=(10, 1),
        policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error=None,
        policyengine_calibration_deferred_stage_top_family_count=7,
        policyengine_calibration_deferred_stage_top_geography_count=4,
        donor_imputer_backend="qrf",
        donor_imputer_condition_selection="pe_prespecified",
        donor_imputer_qrf_zero_threshold=0.05,
        donor_imputer_excluded_variables=(),
        puf_support_clone_enabled=True,
        prefer_cached_cps_asec_source=False,
        policyengine_direct_override_variables=(
            "health_savings_account_ald",
            "non_sch_d_capital_gains",
        ),
        policyengine_prefer_existing_tax_unit_ids=True,
    )
    return replace(defaults, **overrides)


def default_policyengine_us_data_rebuild_source_providers(
    *,
    cps_source_year: int = 2023,
    cps_cache_dir: str | Path | None = None,
    cps_download: bool = True,
    puf_target_year: int = 2024,
    puf_cps_reference_year: int | None = None,
    puf_cache_dir: str | Path | None = None,
    puf_path: str | Path | None = None,
    puf_demographics_path: str | Path | None = None,
    puf_expand_persons: bool = True,
    include_donor_surveys: bool = True,
    include_acs: bool | None = None,
    include_sipp: bool | None = None,
    include_scf: bool | None = None,
    acs_year: int = 2022,
    sipp_year: int = 2023,
    scf_year: int = 2022,
    donor_cache_dir: str | Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> tuple[SourceProvider, ...]:
    """Return the canonical CPS+PUF provider bundle for the rebuild track."""

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

    cps_cache = None if cps_cache_dir is None else Path(cps_cache_dir)
    puf_cache = None if puf_cache_dir is None else Path(puf_cache_dir)
    donor_cache = None if donor_cache_dir is None else Path(donor_cache_dir)
    providers: list[SourceProvider] = [
        CPSASECSourceProvider(
            year=int(cps_source_year),
            cache_dir=cps_cache,
            download=bool(cps_download),
        ),
        PUFSourceProvider(
            target_year=int(puf_target_year),
            cache_dir=puf_cache,
            puf_path=puf_path,
            demographics_path=puf_demographics_path,
            expand_persons=bool(puf_expand_persons),
            uprating_mode=PUF_UPRATING_MODE_PE_SOI,
            cps_reference_year=(
                int(puf_cps_reference_year)
                if puf_cps_reference_year is not None
                else int(cps_source_year)
            ),
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            social_security_split_strategy=SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF,
        ),
    ]
    resolved_include_acs = include_donor_surveys if include_acs is None else include_acs
    resolved_include_sipp = (
        include_donor_surveys if include_sipp is None else include_sipp
    )
    resolved_include_scf = include_donor_surveys if include_scf is None else include_scf
    if resolved_include_acs:
        providers.append(
            ACSSourceProvider(
                year=int(acs_year),
                policyengine_us_data_repo=policyengine_us_data_repo,
                policyengine_us_data_python=policyengine_us_data_python,
            )
        )
    if resolved_include_sipp:
        providers.extend(
            [
                SIPPSourceProvider(
                    block="tips",
                    year=int(sipp_year),
                    cache_dir=donor_cache,
                ),
                SIPPSourceProvider(
                    block="assets",
                    year=int(sipp_year),
                    cache_dir=donor_cache,
                ),
            ]
        )
    if resolved_include_scf:
        providers.append(
            SCFSourceProvider(
                year=int(scf_year),
                policyengine_us_data_repo=policyengine_us_data_repo,
                policyengine_us_data_python=policyengine_us_data_python,
            )
        )
    return tuple(providers)


def build_policyengine_us_data_rebuild_pipeline(
    **config_overrides: Any,
) -> USMicroplexPipeline:
    """Build a USMicroplexPipeline configured for the incumbent parity path."""

    from microplex_us.pipelines.us import USMicroplexPipeline

    return USMicroplexPipeline(
        config=default_policyengine_us_data_rebuild_config(**config_overrides)
    )


def default_policyengine_us_data_rebuild_program() -> PEUSDataRebuildProgram:
    """Return the current PE-US-data rebuild program for `microplex-us`."""

    return PEUSDataRebuildProgram(
        program_id="pe-us-data-rebuild-v1",
        title="Rebuild PE-US-data in Microplex",
        objective=(
            "Reproduce the incumbent PolicyEngine US-data pipeline using the same "
            "sources, imputation families, and weighting backends where useful, "
            "but in the cleaner Microplex runtime structure."
        ),
        principle=(
            "Architecture-first replacement: first make the PE-US-data build path "
            "explicit, modular, and parity-auditable inside Microplex; then change "
            "models only once the incumbent pipeline is faithfully reproducible. "
            "Structural improvements are allowed when they mainly improve "
            "maintainability, provenance, or modularity and should change "
            "results only on the margin."
        ),
        stages=(
            PEUSDataRebuildStage(
                stage_id="source-contracts",
                title="Canonical source contracts",
                goal=(
                    "Load the same incumbent public/source datasets through explicit "
                    "Microplex source descriptors and manifests."
                ),
                pe_owner_modules=(
                    "policyengine_us_data.datasets.cps",
                    "policyengine_us_data.datasets.puf",
                ),
                microplex_owner_modules=(
                    "microplex_us.data_sources.cps",
                    "microplex_us.data_sources.puf",
                    "microplex_us.source_manifests",
                    "microplex_us.source_registry",
                ),
                parity_contract=(
                    "Use the same raw sources and year conventions as PE-US-data, "
                    "but express them through Microplex source contracts."
                ),
                current_status=PEUSDataRebuildStatus.PARTIAL,
                notes=(
                    "PUF already has an external manifest-backed contract. CPS is "
                    "partly descriptor-backed but still needs a cleaner source-level "
                    "parity contract."
                ),
                next_steps=(
                    "Externalize CPS source contracts to the same level as PUF.",
                    "Document year-by-year source selection and fallback policy.",
                ),
            ),
            PEUSDataRebuildStage(
                stage_id="cps-construction",
                title="CPS construction parity",
                goal=(
                    "Reproduce PE-US-data CPS variable construction, mappings, and "
                    "source-backed rules inside Microplex."
                ),
                pe_owner_modules=("policyengine_us_data.datasets.cps.cps",),
                microplex_owner_modules=("microplex_us.data_sources.cps",),
                parity_contract=(
                    "Same CPS mappings and rule-based derivations unless an "
                    "intentional difference is written down."
                ),
                current_status=PEUSDataRebuildStatus.PARTIAL,
                notes=(
                    "Social Security reason-code logic is already close. Broader CPS "
                    "family-level parity is not yet fully audited."
                ),
                next_steps=(
                    "Audit dividends, interest, pensions, and transfer-income rules.",
                    "Back parity claims with focused tests where feasible.",
                ),
            ),
            PEUSDataRebuildStage(
                stage_id="puf-ingestion-uprating",
                title="PUF ingestion and uprating parity",
                goal=(
                    "Mirror PE-US-data's PUF ingest, demographics handling, and "
                    "uprating flow in a modular Microplex adapter."
                ),
                pe_owner_modules=(
                    "policyengine_us_data.datasets.puf.puf",
                    "policyengine_us_data.datasets.puf.uprate_puf",
                ),
                microplex_owner_modules=("microplex_us.data_sources.puf",),
                parity_contract=(
                    "Same PUF source and uprating semantics before we intentionally "
                    "depart from the incumbent modeling choices."
                ),
                current_status=PEUSDataRebuildStatus.PARTIAL,
                notes=(
                    "Microplex has a clean PUF adapter, but it is not yet a "
                    "line-by-line PE-US-data clone on demographics and uprating "
                    "behavior."
                ),
                next_steps=(
                    "Write explicit parity notes for demographics completion and uprating.",
                    "Decide which PE-data heuristics are copied versus retired.",
                ),
            ),
            PEUSDataRebuildStage(
                stage_id="extended-cps-qrf",
                title="Extended CPS splice and CPS-only imputation",
                goal=(
                    "Rebuild the PE-US-data CPS/PUF splice logic and the CPS-only "
                    "QRF imputation stages inside Microplex."
                ),
                pe_owner_modules=(
                    "policyengine_us_data.datasets.cps.extended_cps",
                    "policyengine_us_data.calibration.source_impute",
                ),
                microplex_owner_modules=(
                    "microplex_us.data_sources.family_imputation_benchmark",
                    "microplex_us.data_sources.puf",
                    "microplex_us.pe_source_impute_engine",
                    "microplex_us.pipelines.us",
                ),
                parity_contract=(
                    "Use the same model family and training/prediction split where "
                    "the intent is parity, even if the code is reorganized."
                ),
                current_status=PEUSDataRebuildStatus.PARTIAL,
                notes=(
                    "The donor-survey side now has an explicit PE-style "
                    "prespecified predictor mode, real ACS/SIPP/SCF donor "
                    "providers, and one shared donor-block manifest for "
                    "provider specs, predictor surfaces, condition prep, SIPP "
                    "postprocessing rules, raw SIPP extraction details, "
                    "ACS/SCF subprocess dataset-loader mappings, and a "
                    "centralized PE source-impute block engine. The remaining "
                    "gap is the full extended CPS splice and line-by-line "
                    "stage parity."
                ),
                next_steps=(
                    "Isolate PE-data stage-1 and stage-2 QRF splice contracts.",
                    "Implement them behind Microplex method specs rather than inline scripts.",
                    "Audit annualization, sampling, and donor-row preparation details against PE-data.",
                ),
            ),
            PEUSDataRebuildStage(
                stage_id="family-imputation-parity",
                title="Family imputation parity",
                goal=(
                    "Recreate PE-US-data family imputations using the incumbent model "
                    "families and fallback heuristics before optimizing beyond them."
                ),
                pe_owner_modules=(
                    "policyengine_us_data.calibration.puf_impute",
                    "policyengine_us_data.calibration.source_impute",
                ),
                microplex_owner_modules=(
                    "microplex_us.data_sources.puf",
                    "microplex_us.data_sources.share_imputation",
                    "microplex_us.data_sources.family_imputation_benchmark",
                ),
                parity_contract=(
                    "Match PE-data on model class, feature surface, and fallback "
                    "rules unless a difference is intentional and benchmarked."
                ),
                current_status=PEUSDataRebuildStatus.PARTIAL,
                notes=(
                    "Microplex currently has its own grouped-share / forest-family "
                    "search machinery. That is useful for later improvement, but not "
                    "yet the same as rebuilding the incumbent pipeline."
                ),
                next_steps=(
                    "Add explicit PE-style QRF family methods as first-class runtime options.",
                    "Separate 'incumbent rebuild' from 'challenger search' in method configs.",
                ),
            ),
            PEUSDataRebuildStage(
                stage_id="entity-export-parity",
                title="PE-ingestable entity and export parity",
                goal=(
                    "Build the same PE-ingestable entity tables and input surface, "
                    "with compatibility shims made explicit."
                ),
                pe_owner_modules=(
                    "policyengine_us_data datasets and H5 build path",
                    "policyengine_us.variables.gov.ssa.ss",
                ),
                microplex_owner_modules=(
                    "microplex_us.pipelines.us",
                    "microplex_us.policyengine.us",
                    "microplex_us.pipelines.pre_sim_parity",
                ),
                parity_contract=(
                    "PE should ingest the resulting dataset without relying on hidden "
                    "construction assumptions."
                ),
                current_status=PEUSDataRebuildStatus.PARTIAL,
                notes=(
                    "Microplex export compatibility is fairly strong, but some "
                    "compatibility shims still exist, especially the Social Security "
                    "residual-to-retirement bridge."
                ),
                next_steps=(
                    "Retire or explicitly own the Social Security export shim.",
                    "Expand pre-sim parity audits over more critical input variables.",
                ),
            ),
            PEUSDataRebuildStage(
                stage_id="weighting-backend",
                title="Weighting and calibration backend parity",
                goal=(
                    "Use the same incumbent PE-US-data weighting/calibration backend "
                    "inside a Microplex-owned interface."
                ),
                pe_owner_modules=(
                    "policyengine_us_data.calibration.unified_calibration",
                ),
                microplex_owner_modules=(
                    "microplex_us.pipelines.pe_l0",
                    "microplex_us.unified_calibration",
                    "microplex_us.pipelines.local_reweighting",
                ),
                parity_contract=(
                    "Weight optimization should be callable through Microplex while "
                    "still allowing the incumbent PE optimizer when parity is desired."
                ),
                current_status=PEUSDataRebuildStatus.CLOSE,
                notes=(
                    "The L0 adapter already wraps the PE-US-data optimizer, which is "
                    "the right structural direction."
                ),
                next_steps=(
                    "Make the incumbent optimizer path an explicit parity mode in the main build flow.",
                ),
            ),
            PEUSDataRebuildStage(
                stage_id="targets-and-eval",
                title="Target DB and benchmark parity",
                goal=(
                    "Keep the same PE target estate and measurement operator while "
                    "comparing incumbent and Microplex builds."
                ),
                pe_owner_modules=(
                    "policyengine_us_data.db",
                    "policyengine_us",
                ),
                microplex_owner_modules=(
                    "microplex_us.policyengine.harness",
                    "microplex_us.policyengine.comparison",
                    "microplex_us.pipelines.performance",
                ),
                parity_contract=(
                    "The target DB and PE formulas remain the shared truth/measurement layer."
                ),
                current_status=PEUSDataRebuildStatus.CLOSE,
                notes=(
                    "This is already one of the strongest parts of the current "
                    "Microplex architecture."
                ),
                next_steps=(
                    "Use this layer for scheduled integrated parity checkpoints, not only for final validation.",
                ),
            ),
        ),
    )


def build_policyengine_us_data_rebuild_markdown(
    program: PEUSDataRebuildProgram | None = None,
) -> str:
    """Render the rebuild program as Markdown."""

    resolved = program or default_policyengine_us_data_rebuild_program()
    lines = [
        f"# {resolved.title}",
        "",
        resolved.objective,
        "",
        f"Principle: {resolved.principle}",
        "",
        "## Stages",
        "",
    ]
    for stage in resolved.stages:
        lines.extend(
            [
                f"### {stage.title}",
                f"- `stage_id`: `{stage.stage_id}`",
                f"- `status`: `{stage.current_status.value}`",
                f"- goal: {stage.goal}",
                f"- parity contract: {stage.parity_contract}",
                "- PE owners:",
                *[f"  - `{module}`" for module in stage.pe_owner_modules],
                "- Microplex owners:",
                *[f"  - `{module}`" for module in stage.microplex_owner_modules],
                f"- notes: {stage.notes}",
            ]
        )
        if stage.next_steps:
            lines.append("- next steps:")
            lines.extend([f"  - {step}" for step in stage.next_steps])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
