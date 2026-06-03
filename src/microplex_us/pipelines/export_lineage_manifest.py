"""Generate static lineage coverage for eCPS-required PE-US exports.

Column presence and H5 support parity answer whether a finished artifact has
the right shape and nonzero/variant data. This module answers the cheaper
pre-build question: does Microplex have an intended source or construction path
for each required export column?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from microplex_us.microdata_roles import POLICYENGINE_US_TAKEUP_INPUT_VARIABLES
from microplex_us.pipelines.check_export_columns import (
    DEFAULT_CONTRACT_PATH,
    _h5_column_values,
    _support_requirement,
    _support_stats,
    load_contract,
)
from microplex_us.policyengine.us import (
    POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES,
    POLICYENGINE_US_EXPORT_COLUMN_ALIASES,
    POLICYENGINE_US_EXPORT_DEFAULTS,
    POLICYENGINE_US_LEGACY_CONTRACT_VARIABLE_ENTITIES,
    POLICYENGINE_US_NUMERIC_ENUM_EXPORT_MAPS,
    POLICYENGINE_US_STRUCTURAL_EXPORT_COLUMNS,
)
from microplex_us.variables import VARIABLE_SEMANTIC_SPECS

SCHEMA_VERSION = 1

SOURCE_BACKED_EVIDENCE_KINDS = frozenset(
    {
        "cps_raw_mapping",
        "cps_derived_recode",
        "puf_raw_mapping",
        "puf_support_clone_imputation",
        "puf_support_clone_override",
        "puf_support_clone_refresh",
        "pe_source_impute_target",
        "pe_source_impute_observed",
        "pipeline_constructed",
        "semantic_derived",
        "policyengine_export_alias",
        "takeup_assumption",
        "allowed_computed_export",
        "enum_map",
        "structural_export",
    }
)
DEFAULT_ONLY_EVIDENCE_KIND = "policyengine_export_default"


@dataclass(frozen=True)
class ExportLineageEvidence:
    """One static source/code path that can populate an export column."""

    kind: str
    source: str
    detail: str
    raw_columns: tuple[str, ...] = ()
    source_variables: tuple[str, ...] = ()


@dataclass
class ExportLineageEntry:
    """Lineage coverage for one required export column."""

    column: str
    required: bool
    entity: str | None
    evidence: list[ExportLineageEvidence] = field(default_factory=list)
    export_path_status: str = "unknown"
    has_source_lineage: bool = False
    ecps_support_requirement: str | None = None
    ecps_support_stats: dict[str, Any] | None = None
    issue: str | None = None


def build_export_lineage_manifest(
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    support_baseline: Path | None = None,
    period: int = 2024,
) -> dict[str, Any]:
    """Return per-column lineage coverage for the eCPS export contract."""
    contract = load_contract(contract_path)
    required_columns = sorted(str(column) for column in contract["required"])
    evidence_index = _build_static_evidence_index()
    baseline_support = (
        _baseline_support_by_column(
            support_baseline,
            required_columns=required_columns,
            period=period,
        )
        if support_baseline is not None
        else {}
    )

    entries: list[ExportLineageEntry] = []
    for column in required_columns:
        evidence = sorted(
            evidence_index.get(column, []),
            key=lambda item: (item.kind, item.source, item.detail),
        )
        has_source_lineage = any(
            item.kind in SOURCE_BACKED_EVIDENCE_KINDS for item in evidence
        )
        status = _export_path_status(evidence, has_source_lineage=has_source_lineage)
        entry = ExportLineageEntry(
            column=column,
            required=True,
            entity=_infer_export_entity(column, evidence),
            evidence=evidence,
            export_path_status=status,
            has_source_lineage=has_source_lineage,
        )

        support_info = baseline_support.get(column)
        if support_info is not None:
            entry.ecps_support_requirement = support_info["requirement"]
            entry.ecps_support_stats = support_info["stats"]
            if support_info["requirement"] is not None and not has_source_lineage:
                entry.issue = "ecps_populated_export_has_no_source_lineage"
        elif not evidence:
            entry.issue = "required_export_has_no_static_lineage"
        entries.append(entry)

    issue_entries = [entry for entry in entries if entry.issue]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_path": str(contract_path),
        "support_baseline": str(support_baseline) if support_baseline else None,
        "period": int(period),
        "summary": {
            "required_export_count": len(entries),
            "source_lineage_count": sum(entry.has_source_lineage for entry in entries),
            "default_only_count": sum(
                entry.export_path_status == "default_only" for entry in entries
            ),
            "unknown_count": sum(
                entry.export_path_status == "unknown" for entry in entries
            ),
            "ecps_populated_checked_count": sum(
                entry.ecps_support_requirement is not None for entry in entries
            ),
            "issue_count": len(issue_entries),
        },
        "issues": [
            {
                "column": entry.column,
                "issue": entry.issue,
                "export_path_status": entry.export_path_status,
                "ecps_support_requirement": entry.ecps_support_requirement,
            }
            for entry in issue_entries
        ],
        "columns": [_entry_to_dict(entry) for entry in entries],
    }
    return payload


def _entry_to_dict(entry: ExportLineageEntry) -> dict[str, Any]:
    payload = asdict(entry)
    payload["evidence"] = [asdict(item) for item in entry.evidence]
    return payload


def _export_path_status(
    evidence: list[ExportLineageEvidence],
    *,
    has_source_lineage: bool,
) -> str:
    if has_source_lineage:
        if any(item.kind == "structural_export" for item in evidence):
            return "structural"
        return "source_or_constructed"
    if evidence and all(item.kind == DEFAULT_ONLY_EVIDENCE_KIND for item in evidence):
        return "default_only"
    if evidence:
        return "documented_no_source"
    return "unknown"


def _baseline_support_by_column(
    baseline_h5: Path,
    *,
    required_columns: list[str],
    period: int,
) -> dict[str, dict[str, Any]]:
    import h5py

    period_key = str(int(period))
    support: dict[str, dict[str, Any]] = {}
    with h5py.File(baseline_h5, "r") as handle:
        for column in required_columns:
            values = _h5_column_values(handle, column, period_key=period_key)
            if values is None:
                continue
            stats = _support_stats(column, values)
            support[column] = {
                "requirement": _support_requirement(stats),
                "stats": asdict(stats),
            }
    return support


def _build_static_evidence_index() -> dict[str, list[ExportLineageEvidence]]:
    index: dict[str, list[ExportLineageEvidence]] = {}
    _add_structural_evidence(index)
    _add_policyengine_export_evidence(index)
    _add_cps_evidence(index)
    _add_puf_manifest_evidence(index)
    _add_pe_source_impute_spec_evidence(index)
    _add_puf_support_clone_evidence(index)
    _add_pipeline_constructed_evidence(index)
    _add_semantic_evidence(index)
    return index


def _append(
    index: dict[str, list[ExportLineageEvidence]],
    column: str,
    evidence: ExportLineageEvidence,
) -> None:
    index.setdefault(str(column), []).append(evidence)


def _add_structural_evidence(index: dict[str, list[ExportLineageEvidence]]) -> None:
    for column in POLICYENGINE_US_STRUCTURAL_EXPORT_COLUMNS:
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="structural_export",
                source="policyengine_us_export",
                detail="Entity id/link/weight column emitted by PE-US H5 writer.",
            ),
        )


def _add_policyengine_export_evidence(
    index: dict[str, list[ExportLineageEvidence]],
) -> None:
    for source_column, target_column in POLICYENGINE_US_EXPORT_COLUMN_ALIASES.items():
        _append(
            index,
            target_column,
            ExportLineageEvidence(
                kind="policyengine_export_alias",
                source="POLICYENGINE_US_EXPORT_COLUMN_ALIASES",
                detail=f"Exports source column {source_column!r} as {target_column!r}.",
                source_variables=(source_column,),
            ),
        )
    for column, default in POLICYENGINE_US_EXPORT_DEFAULTS.items():
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind=DEFAULT_ONLY_EVIDENCE_KIND,
                source="POLICYENGINE_US_EXPORT_DEFAULTS",
                detail=f"Default exported when no source column is present: {default!r}.",
            ),
        )
    for column in POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES:
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="allowed_computed_export",
                source="POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES",
                detail="Computed/overridable PE-US export allowed by the H5 writer.",
            ),
        )
    for column in POLICYENGINE_US_NUMERIC_ENUM_EXPORT_MAPS:
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="enum_map",
                source="POLICYENGINE_US_NUMERIC_ENUM_EXPORT_MAPS",
                detail="Numeric source code mapped to PE-US enum export value.",
            ),
        )
    for column in POLICYENGINE_US_TAKEUP_INPUT_VARIABLES:
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="takeup_assumption",
                source="microplex_us.pipelines.us",
                detail="Policy take-up input generated from Microplex take-up assumptions/source proxies.",
            ),
        )


def _add_cps_evidence(index: dict[str, list[ExportLineageEvidence]]) -> None:
    from microplex_us.data_sources.cps import (
        CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP,
        CURRENT_HEALTH_COVERAGE_RULE_INPUT_ALIAS_MAP,
        HOUSEHOLD_VARIABLES,
        PERSON_VARIABLES,
    )

    for raw_column, column in PERSON_VARIABLES.items():
        if str(column).startswith("_"):
            continue
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="cps_raw_mapping",
                source="CPS ASEC person",
                detail=f"Mapped from CPS ASEC raw person column {raw_column}.",
                raw_columns=(raw_column,),
            ),
        )
    for raw_column, column in HOUSEHOLD_VARIABLES.items():
        if str(column).startswith("_"):
            continue
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="cps_raw_mapping",
                source="CPS ASEC household",
                detail=f"Mapped from CPS ASEC raw household column {raw_column}.",
                raw_columns=(raw_column,),
            ),
        )
    for column, raw_column in CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP.items():
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="cps_derived_recode",
                source="CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP",
                detail=f"Current health coverage report recoded from {raw_column}.",
                raw_columns=(raw_column,),
            ),
        )
    for column, raw_column in {
        "reported_has_private_health_coverage_at_interview": "NOW_PRIV",
        "reported_has_public_health_coverage_at_interview": "NOW_PUB",
        "reported_is_insured_at_interview": "NOW_COV",
        "reported_is_uninsured_at_interview": "NOW_COV",
    }.items():
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="cps_derived_recode",
                source="CPS current health coverage recodes",
                detail=f"Derived from CPS ASEC current-coverage raw column {raw_column}.",
                raw_columns=(raw_column,),
            ),
        )
    for column, source_columns in {
        "reported_has_multiple_health_coverage_at_interview": tuple(
            CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP
        ),
        "has_esi": ("reported_has_employer_sponsored_health_coverage_at_interview",),
        "has_marketplace_health_coverage": (
            "reported_has_marketplace_health_coverage_at_interview",
        ),
    }.items():
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="cps_derived_recode",
                source="CPS current health coverage recodes",
                detail="Derived from CPS ASEC current health coverage indicators.",
                source_variables=tuple(source_columns),
            ),
        )
    for column, source_column in CURRENT_HEALTH_COVERAGE_RULE_INPUT_ALIAS_MAP.items():
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="cps_derived_recode",
                source="CURRENT_HEALTH_COVERAGE_RULE_INPUT_ALIAS_MAP",
                detail=f"Rule input aliases reported coverage column {source_column}.",
                source_variables=(source_column,),
            ),
        )


def _add_puf_manifest_evidence(index: dict[str, list[ExportLineageEvidence]]) -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "manifests" / "puf.json"
    payload = json.loads(manifest_path.read_text())
    for observation in payload.get("observations", []):
        for mapping in observation.get("columns", []):
            column = mapping.get("canonical_name")
            raw_column = mapping.get("raw_column")
            if not column:
                continue
            _append(
                index,
                str(column),
                ExportLineageEvidence(
                    kind="puf_raw_mapping",
                    source="manifests/puf.json",
                    detail=f"Mapped from IRS SOI PUF raw column {raw_column}.",
                    raw_columns=(str(raw_column),) if raw_column is not None else (),
                ),
            )


def _add_pe_source_impute_spec_evidence(
    index: dict[str, list[ExportLineageEvidence]],
) -> None:
    spec_path = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "pe_source_impute_blocks.json"
    )
    payload = json.loads(spec_path.read_text())
    for block_name, block in payload.get("blocks", {}).items():
        source = str(block.get("survey_name") or block_name)
        block_label = f"{source}:{block_name}"
        for column in block.get("target_variables", []):
            _append(
                index,
                str(column),
                ExportLineageEvidence(
                    kind="pe_source_impute_target",
                    source=block_label,
                    detail="Target variable populated by PE-source donor imputation block.",
                ),
            )
        for column in (
            *block.get("person_variables", []),
            *block.get("household_variables", []),
        ):
            _append(
                index,
                str(column),
                ExportLineageEvidence(
                    kind="pe_source_impute_observed",
                    source=block_label,
                    detail="Observed variable available in donor source spec.",
                ),
            )


def _add_puf_support_clone_evidence(
    index: dict[str, list[ExportLineageEvidence]],
) -> None:
    from microplex_us.pipelines.us import (
        PUF_SUPPORT_CLONE_CPS_REFRESH_VARIABLES,
        PUF_SUPPORT_CLONE_IMPUTED_VARIABLES,
        PUF_SUPPORT_CLONE_OVERRIDDEN_VARIABLES,
        PUF_SUPPORT_CLONE_SPECIAL_VARIABLES,
    )

    for column in PUF_SUPPORT_CLONE_IMPUTED_VARIABLES:
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="puf_support_clone_imputation",
                source="PUF_SUPPORT_CLONE_IMPUTED_VARIABLES",
                detail="PUF donor variable imputed onto the CPS support-clone surface.",
            ),
        )
    for column in PUF_SUPPORT_CLONE_OVERRIDDEN_VARIABLES:
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="puf_support_clone_override",
                source="PUF_SUPPORT_CLONE_OVERRIDDEN_VARIABLES",
                detail="PUF donor variable may override/collapse onto the CPS scaffold.",
            ),
        )
    for column in PUF_SUPPORT_CLONE_SPECIAL_VARIABLES:
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="puf_support_clone_imputation",
                source="PUF_SUPPORT_CLONE_SPECIAL_VARIABLES",
                detail="Special support-clone variable populated through PUF/CPS support handling.",
            ),
        )
    for column in PUF_SUPPORT_CLONE_CPS_REFRESH_VARIABLES:
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="puf_support_clone_refresh",
                source="PUF_SUPPORT_CLONE_CPS_REFRESH_VARIABLES",
                detail="CPS-only status/categorical field refreshed after PUF support matching.",
            ),
        )


def _add_pipeline_constructed_evidence(
    index: dict[str, list[ExportLineageEvidence]],
) -> None:
    constructed: dict[str, tuple[str, tuple[str, ...]]] = {
        "block_geoid": (
            "Representative Census block assignment",
            ("state_fips", "county_fips", "spm_unit_size"),
        ),
        "tract_geoid": (
            "Representative Census block assignment",
            ("block_geoid",),
        ),
        "congressional_district_geoid": (
            "Representative Census block assignment",
            ("block_geoid", "state_fips"),
        ),
        "employment_income_before_lsr": (
            "USMicroplexPipeline income normalizer",
            ("employment_income", "wage_income", "income"),
        ),
        "weekly_hours_worked_before_lsr": (
            "CPS weekly-hours export support",
            ("A_HRS1", "hours_worked", "hours_worked_last_week"),
        ),
        "selected_marketplace_plan_benchmark_ratio": (
            "PE-US ACA selected Marketplace plan ratio construction",
            (
                "health_insurance_premiums_without_medicare_part_b",
                "takes_up_aca_if_eligible",
                "aca_ptc",
                "slcsp",
            ),
        ),
        "self_employment_income_before_lsr": (
            "USMicroplexPipeline income normalizer",
            ("self_employment_income",),
        ),
        "long_term_capital_gains_before_response": (
            "USMicroplexPipeline investment-income normalizer",
            ("long_term_capital_gains", "capital_gains"),
        ),
        "taxable_private_pension_income": (
            "CPS private pension taxable/exempt split",
            ("PEN_VAL", "ANN_VAL"),
        ),
        "social_security_disability": (
            "CPS Social Security reason-code split",
            ("SS_VAL", "SSKIND1", "SSKIND2", "age"),
        ),
        "social_security_survivors": (
            "CPS Social Security reason-code split",
            ("SS_VAL", "SSKIND1", "SSKIND2"),
        ),
        "social_security_dependents": (
            "CPS Social Security reason-code split",
            ("SS_VAL", "SSKIND1", "SSKIND2"),
        ),
        "disability_benefits": (
            "CPS disability-income workers-comp split",
            ("DSAB_VAL1", "DSAB_VAL2", "DSAB_ON1", "DSAB_ON2"),
        ),
        "employer_sponsored_insurance_premiums": (
            "CPS employer-sponsored insurance premium imputation",
            ("NOW_OWNGRP", "NOW_HIPAID", "NOW_GRPFTYP", "PHIP_VAL"),
        ),
        "reported_owns_employer_sponsored_health_insurance_at_interview": (
            "CPS employer-sponsored insurance policyholder recode",
            ("NOW_OWNGRP",),
        ),
        "takes_up_dc_ptc": (
            "Microplex tax-unit take-up-rate construction",
            ("DEFAULT_DC_PTC_TAKEUP_RATE",),
        ),
        "is_unmarried_partner_of_household_head": (
            "CPS relationship-to-householder recode",
            ("PERRP",),
        ),
        "is_separated": (
            "CPS marital-status recode",
            ("A_MARITL",),
        ),
        "is_surviving_spouse": (
            "CPS marital-status recode",
            ("A_MARITL",),
        ),
        "is_blind": (
            "CPS difficulty-seeing recode",
            ("PEDISEYE",),
        ),
        "ssn_card_type": (
            "CPS nativity/legal-status tax-id replacement",
            ("PRCITSHP", "PEINUSYR", "PENATVTY", "PEMLR"),
        ),
        "immigration_status_str": (
            "CPS nativity/legal-status tax-id replacement",
            ("PRCITSHP", "PEINUSYR", "PENATVTY", "PEMLR"),
        ),
        "is_pregnant": (
            "USMicroplexPipeline pregnancy-rate construction",
            ("age", "is_female", "state_fips"),
        ),
        "has_valid_ssn": ("CPS tax-id replacement construction", ("ssn_card_type",)),
        "taxpayer_id_type": ("CPS tax-id replacement construction", ("ssn_card_type",)),
        "has_tin": ("PE-US export identity construction", ("taxpayer_id_type",)),
        "has_itin": ("PE-US export identity construction", ("taxpayer_id_type",)),
        "hourly_wage": ("CPS hourly work recode", ("_hourly_pay_cents",)),
        "is_paid_hourly": ("CPS hourly work recode", ("_is_paid_hourly_code",)),
        "is_union_member_or_covered": ("CPS union recode", ("A_UNMEM",)),
        "is_tipped_occupation": (
            "CPS occupation to Treasury tipped-occupation recode",
            ("detailed_occupation_recode",),
        ),
        "treasury_tipped_occupation_code": (
            "CPS occupation to Treasury tipped-occupation recode",
            ("detailed_occupation_recode",),
        ),
        "is_computer_scientist": (
            "CPS detailed occupation recode",
            ("detailed_occupation_recode",),
        ),
        "is_executive_administrative_professional": (
            "CPS detailed occupation recode",
            ("detailed_occupation_recode",),
        ),
        "is_farmer_fisher": (
            "CPS detailed occupation recode",
            ("detailed_occupation_recode",),
        ),
        "is_military": ("CPS class/occupation recode", ("class_of_worker",)),
        "is_full_time_college_student": (
            "CPS school-enrollment recode",
            ("_high_school_or_college_status",),
        ),
        "has_never_worked": ("CPS work-status recode", ("work_status",)),
        "previous_year_income_available": (
            "Prior CPS ASEC PERIDNUM join",
            ("PERIDNUM",),
        ),
        "self_employment_income_last_year": (
            "Prior CPS ASEC PERIDNUM join",
            ("PERIDNUM", "SEMP_VAL"),
        ),
        "tax_exempt_private_pension_income": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "taxable_401k_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "tax_exempt_401k_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "taxable_403b_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "tax_exempt_403b_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "regular_ira_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "roth_ira_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "tax_exempt_ira_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "taxable_sep_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "tax_exempt_sep_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "other_type_retirement_account_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "keogh_distributions": (
            "CPS retirement distribution split",
            ("DST_SC*", "DST_VAL*"),
        ),
        "self_employed_pension_contributions_desired": (
            "CPS retirement-contribution desired-account split",
            ("RETCB_VAL", "wage_income", "self_employment_income"),
        ),
        "traditional_401k_contributions_desired": (
            "CPS retirement-contribution desired-account split",
            ("RETCB_VAL", "wage_income", "self_employment_income"),
        ),
        "roth_401k_contributions_desired": (
            "CPS retirement-contribution desired-account split",
            ("RETCB_VAL", "wage_income", "self_employment_income"),
        ),
        "traditional_ira_contributions_desired": (
            "CPS retirement-contribution desired-account split",
            ("RETCB_VAL", "wage_income", "self_employment_income"),
        ),
        "roth_ira_contributions_desired": (
            "CPS retirement-contribution desired-account split",
            ("RETCB_VAL", "wage_income", "self_employment_income"),
        ),
        "first_home_mortgage_balance": (
            "Tax-unit mortgage support construction",
            ("first_home_mortgage_interest", "scf_mortgage_debt"),
        ),
        "first_home_mortgage_interest": (
            "Tax-unit mortgage support construction",
            ("deductible_mortgage_interest", "mortgage_interest_paid"),
        ),
        "first_home_mortgage_origination_year": (
            "Tax-unit mortgage support construction",
            ("first_home_mortgage_interest",),
        ),
    }
    aotc_columns = (
        "is_pursuing_credential_for_american_opportunity_credit",
        "attends_eligible_educational_institution_for_american_opportunity_credit",
        "is_enrolled_at_least_half_time_for_american_opportunity_credit",
        "has_american_opportunity_credit_1098_t_or_exception",
        "has_american_opportunity_credit_institution_ein",
        "has_completed_first_four_years_of_postsecondary_education",
        "has_felony_drug_conviction",
        "american_opportunity_credit_claimed_prior_years",
    )
    for column in aotc_columns:
        constructed[column] = (
            "USMicroplexPipeline._construct_aotc_eligibility_inputs",
            ("american_opportunity_credit", "qualified_tuition_expenses"),
        )
    for column, (detail, variables) in constructed.items():
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="pipeline_constructed",
                source="microplex_us.pipelines.us",
                detail=detail,
                source_variables=tuple(variables),
            ),
        )


def _add_semantic_evidence(index: dict[str, list[ExportLineageEvidence]]) -> None:
    for column, spec in VARIABLE_SEMANTIC_SPECS.items():
        if not spec.derived_from:
            continue
        _append(
            index,
            column,
            ExportLineageEvidence(
                kind="semantic_derived",
                source="VARIABLE_SEMANTIC_SPECS",
                detail="Variable semantic spec declares derived source variables.",
                source_variables=tuple(spec.derived_from),
            ),
        )


def _infer_export_entity(
    column: str,
    evidence: list[ExportLineageEvidence],
) -> str | None:
    if column in POLICYENGINE_US_STRUCTURAL_EXPORT_COLUMNS:
        if column == "household_weight" or column == "household_id":
            return "household"
        if column.startswith("person_") or column == "person_id":
            return "person"
        if column.endswith("_id"):
            return column.removesuffix("_id")
    if column in POLICYENGINE_US_LEGACY_CONTRACT_VARIABLE_ENTITIES:
        return POLICYENGINE_US_LEGACY_CONTRACT_VARIABLE_ENTITIES[column]
    for item in evidence:
        if item.kind.startswith("pe_source_impute"):
            if "household" in item.detail.lower():
                return "household"
            return "person"
    return None


def _format_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "eCPS export lineage manifest",
        f"  required exports:       {summary['required_export_count']}",
        f"  source lineage:         {summary['source_lineage_count']}",
        f"  default only:           {summary['default_only_count']}",
        f"  unknown:                {summary['unknown_count']}",
        f"  eCPS populated checked: {summary['ecps_populated_checked_count']}",
        f"  issues:                 {summary['issue_count']}",
    ]
    if payload["issues"]:
        lines.append("")
        lines.append("  issue columns:")
        for issue in payload["issues"]:
            lines.append(
                "    - "
                f"{issue['column']} ({issue['issue']}; "
                f"{issue['export_path_status']})"
            )
    lines.append("")
    lines.append("  RESULT: " + ("PASS" if not payload["issues"] else "FAIL"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="export_lineage_manifest",
        description=(
            "Generate a static lineage manifest for required eCPS export columns."
        ),
    )
    parser.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT_PATH),
        help="eCPS export contract JSON path.",
    )
    parser.add_argument(
        "--support-baseline",
        metavar="H5",
        help=(
            "Optional eCPS baseline H5. When supplied, the manifest flags "
            "eCPS-populated required exports that have no source lineage."
        ),
    )
    parser.add_argument(
        "--period",
        type=int,
        default=2024,
        help="Period to inspect in --support-baseline (default: 2024).",
    )
    parser.add_argument(
        "--output",
        metavar="JSON",
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit 1 if the manifest contains lineage issues.",
    )
    args = parser.parse_args(argv)

    payload = build_export_lineage_manifest(
        contract_path=Path(args.contract),
        support_baseline=Path(args.support_baseline) if args.support_baseline else None,
        period=int(args.period),
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(_format_report(payload))
    return 1 if args.fail_on_issues and payload["issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
