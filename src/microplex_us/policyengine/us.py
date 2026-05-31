"""PolicyEngine US integration helpers for targets, simulation, and export."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
from microplex.calibration import LinearConstraint
from microplex.core import EntityType
from microplex.targets import (
    TargetAggregation,
    TargetFilter,
    TargetQuery,
    TargetSet,
    TargetSpec,
    apply_target_query,
)

from microplex_us.microdata_roles import POLICYENGINE_US_TAKEUP_INPUT_VARIABLES
from microplex_us.policyengine.target_profiles import PolicyEngineUSTargetCell

GEOGRAPHIC_CONSTRAINT_VARIABLES: set[str] = {
    "state_fips",
    "congressional_district_geoid",
}


@dataclass(frozen=True)
class PolicyEngineUSConstraint:
    """A single stratum constraint from the PolicyEngine targets DB."""

    variable: str
    operation: str
    value: str


@dataclass(frozen=True)
class PolicyEngineUSStratum:
    """A stratum definition from the PolicyEngine US targets DB."""

    stratum_id: int
    definition_hash: str | None = None
    parent_stratum_id: int | None = None
    constraints: tuple[PolicyEngineUSConstraint, ...] = ()


class PolicyEngineUSTargetValidationError(ValueError):
    """Raised when imported PolicyEngine target metadata is inconsistent."""


@dataclass(frozen=True)
class PolicyEngineUSDBTarget:
    """A target row from the PolicyEngine US targets database."""

    target_id: int
    variable: str
    period: int
    stratum_id: int
    reform_id: int
    value: float
    active: bool
    tolerance: float | None = None
    source: str | None = None
    notes: str | None = None
    geo_level: str | None = None
    geographic_id: str | None = None
    domain_variable: str | None = None
    definition_hash: str | None = None
    parent_stratum_id: int | None = None
    constraints: tuple[PolicyEngineUSConstraint, ...] = ()

    @property
    def is_unconstrained(self) -> bool:
        """Whether this target applies nationally without stratum filters."""
        return not self.constraints

    @property
    def domain_variables(self) -> tuple[str, ...]:
        """Domain-variable hints parsed from the target_overview view."""
        if self.domain_variable is None:
            return ()
        values = [
            item.strip()
            for item in str(self.domain_variable).split(",")
            if item.strip()
        ]
        return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class PolicyEngineUSQuantityTarget:
    """A PE-computed quantity used as a Microplex calibration total."""

    name: str
    variable: str
    column: str
    period: int | None = None
    map_to: str | None = None
    aggregation: Literal["sum", "mean", "count_positive"] = "sum"


@dataclass(frozen=True)
class PolicyEngineUSVariableBinding:
    """How a PolicyEngine variable is represented in Microplex entity tables."""

    entity: EntityType
    column: str | None = None
    household_id_column: str = "household_id"


@dataclass(frozen=True)
class PolicyEngineUSEntityTableBundle:
    """Entity tables aligned to household weights for PE-style calibration."""

    households: pd.DataFrame
    persons: pd.DataFrame | None = None
    tax_units: pd.DataFrame | None = None
    spm_units: pd.DataFrame | None = None
    families: pd.DataFrame | None = None
    marital_units: pd.DataFrame | None = None

    def table_for(self, entity: EntityType) -> pd.DataFrame:
        if entity is EntityType.HOUSEHOLD:
            return self.households
        if entity is EntityType.PERSON and self.persons is not None:
            return self.persons
        if entity is EntityType.TAX_UNIT and self.tax_units is not None:
            return self.tax_units
        if entity is EntityType.SPM_UNIT and self.spm_units is not None:
            return self.spm_units
        if entity is EntityType.FAMILY and self.families is not None:
            return self.families
        raise KeyError(f"No table available for entity '{entity.value}'")


_PIPELINE_CHECKPOINT_TABLES: tuple[str, ...] = (
    "households",
    "persons",
    "tax_units",
    "spm_units",
    "families",
    "marital_units",
)

_ALLOWED_CHECKPOINT_STAGES: frozenset[str] = frozenset(
    {"post_imputation", "post_microsim"}
)


def save_us_pipeline_checkpoint(
    bundle: PolicyEngineUSEntityTableBundle,
    path: str | Path,
    *,
    stage: Literal["post_imputation", "post_microsim"],
) -> Path:
    """Persist a pipeline-stage bundle to ``path`` as parquet + metadata.

    Writes one parquet file per non-None entity table plus a
    ``metadata.json`` index tagged with the pipeline ``stage``. Two
    stages are supported:

    * ``"post_imputation"`` — after donor imputation, before PE microsim
      materializes target variables. Resuming from here reruns
      microsim + calibration.
    * ``"post_microsim"`` — after microsim materialization, before the
      calibration fit loop. Resuming from here reruns only calibration.
    """
    import json
    import shutil

    if stage not in _ALLOWED_CHECKPOINT_STAGES:
        raise ValueError(
            f"stage must be one of {sorted(_ALLOWED_CHECKPOINT_STAGES)}; got {stage!r}"
        )

    checkpoint_dir = Path(path)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)

    metadata: dict[str, Any] = {"format_version": 1, "stage": stage}
    for table_name in _PIPELINE_CHECKPOINT_TABLES:
        frame = getattr(bundle, table_name)
        if frame is None:
            metadata[table_name] = None
            continue
        frame.to_parquet(checkpoint_dir / f"{table_name}.parquet", index=False)
        metadata[table_name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }

    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return checkpoint_dir


def load_us_pipeline_checkpoint(
    path: str | Path,
    *,
    expected_stage: Literal["post_imputation", "post_microsim"] | None = None,
) -> tuple[PolicyEngineUSEntityTableBundle, dict[str, Any]]:
    """Load a pipeline-stage bundle previously saved by ``save_us_pipeline_checkpoint``.

    Returns ``(bundle, metadata)`` so callers can inspect the saved
    stage. If ``expected_stage`` is provided, a mismatch raises a clear
    error — protects against running recalibration from a post-microsim
    checkpoint when a post-imputation checkpoint was expected or vice
    versa.
    """
    import json

    checkpoint_dir = Path(path)
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"US pipeline checkpoint not found at {checkpoint_dir}")
    metadata = json.loads(metadata_path.read_text())

    saved_stage = metadata.get("stage")
    if expected_stage is not None and saved_stage != expected_stage:
        raise ValueError(
            f"Checkpoint at {checkpoint_dir} has stage {saved_stage!r}, "
            f"expected {expected_stage!r}"
        )

    tables: dict[str, pd.DataFrame | None] = {}
    for table_name in _PIPELINE_CHECKPOINT_TABLES:
        if metadata.get(table_name) is None:
            tables[table_name] = None
            continue
        tables[table_name] = pd.read_parquet(checkpoint_dir / f"{table_name}.parquet")
    return PolicyEngineUSEntityTableBundle(**tables), metadata


@dataclass(frozen=True)
class PolicyEngineUSVariableMaterializationResult:
    """Materialized PE variables plus any per-variable failures."""

    tables: PolicyEngineUSEntityTableBundle
    bindings: dict[str, PolicyEngineUSVariableBinding]
    materialized_variables: tuple[str, ...] = ()
    failed_variables: dict[str, str] = field(default_factory=dict)


DEFAULT_POLICYENGINE_US_VARIABLE_BINDINGS: dict[str, PolicyEngineUSVariableBinding] = {
    "household_count": PolicyEngineUSVariableBinding(entity=EntityType.HOUSEHOLD),
    "person_count": PolicyEngineUSVariableBinding(entity=EntityType.PERSON),
    "tax_unit_count": PolicyEngineUSVariableBinding(entity=EntityType.TAX_UNIT),
    "spm_unit_count": PolicyEngineUSVariableBinding(entity=EntityType.SPM_UNIT),
}

POLICYENGINE_US_ENTITY_KEY_TO_ENTITY_TYPE: dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "household": EntityType.HOUSEHOLD,
    "tax_unit": EntityType.TAX_UNIT,
    "spm_unit": EntityType.SPM_UNIT,
    "family": EntityType.FAMILY,
}

ENTITY_TYPE_TO_POLICYENGINE_US_ENTITY_KEY: dict[EntityType, str] = {
    entity_type: entity_key
    for entity_key, entity_type in POLICYENGINE_US_ENTITY_KEY_TO_ENTITY_TYPE.items()
}

SAFE_POLICYENGINE_US_EXPORT_VARIABLES: set[str] = {
    "age",
    "alimony_expense",
    "alimony_income",
    "amt_foreign_tax_credit",
    "auto_loan_balance",
    "auto_loan_interest",
    "bank_account_assets",
    "bond_assets",
    "casualty_loss",
    "child_support_expense",
    "child_support_received",
    "charitable_cash_donations",
    "charitable_non_cash_donations",
    "receives_wic",
    "cps_race",
    "disability_benefits",
    "domestic_production_ald",
    "early_withdrawal_penalty",
    "educator_expense",
    "excess_withheld_payroll_tax",
    "general_business_credit",
    "health_insurance_premiums_without_medicare_part_b",
    "investment_income_elected_form_4952",
    "is_female",
    "is_hispanic",
    "is_blind",
    "is_disabled",
    "is_household_head",
    "long_term_capital_gains_on_collectibles",
    "employment_income_before_lsr",
    "miscellaneous_income",
    "non_sch_d_capital_gains",
    "other_medical_expenses",
    "other_credits",
    "over_the_counter_health_expenses",
    "own_children_in_household",
    "prior_year_minimum_tax_credit",
    "qualified_tuition_expenses",
    "recapture_of_investment_credit",
    "salt_refund_income",
    "self_employment_income_before_lsr",
    "social_security_disability",
    "social_security_retirement",
    "social_security_survivors",
    "social_security_dependents",
    "stock_assets",
    "taxable_ira_distributions",
    "tip_income",
    "unemployment_compensation",
    "unrecaptured_section_1250_gain",
    "unreimbursed_business_employee_expenses",
    "unreported_payroll_tax",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "real_estate_taxes",
    "rental_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
    "partnership_s_corp_income",
    "partnership_se_income",
    "estate_income",
    "farm_income",
    "farm_operations_income",
    "farm_rent_income",
    "has_esi",
    "has_marketplace_health_coverage",
    "health_savings_account_ald",
    "is_separated",
    "is_surviving_spouse",
    "net_worth",
    "ssn_card_type",
    "spm_unit_tenure_type",
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
    "student_loan_interest",
    "tenure_type",
    "state_fips",
    "county_fips",
} | set(POLICYENGINE_US_TAKEUP_INPUT_VARIABLES)

POLICYENGINE_US_EXPORT_COLUMN_ALIASES: dict[str, str] = {
    "race": "cps_race",
}

POLICYENGINE_US_EXPORT_DEFAULTS: dict[str, Any] = {
    "auto_loan_balance": 0.0,
    "auto_loan_interest": 0.0,
    "business_is_sstb": False,
    "detailed_occupation_recode": 0,
    "domestic_production_ald": 0,
    "estate_income_would_be_qualified": True,
    "farm_operations_income_would_be_qualified": True,
    "farm_rent_income_would_be_qualified": True,
    "first_home_mortgage_balance": 0.0,
    "first_home_mortgage_interest": 0.0,
    "first_home_mortgage_origination_year": 0,
    "free_school_meals_reported": 0.0,
    "has_champva_health_coverage_at_interview": False,
    "has_indian_health_service_coverage_at_interview": False,
    "has_marketplace_health_coverage_at_interview": False,
    "has_medicaid_health_coverage_at_interview": False,
    "has_never_worked": False,
    "has_non_marketplace_direct_purchase_health_coverage_at_interview": False,
    "has_other_means_tested_health_coverage_at_interview": False,
    "has_tricare_health_coverage_at_interview": False,
    "has_valid_ssn": True,
    "has_va_health_coverage_at_interview": False,
    "home_mortgage_interest": 0,
    "hourly_wage": 0,
    "hours_worked_last_week": 0,
    "household_vehicles_owned": 0,
    "household_vehicles_value": 0,
    "immigration_status_str": "CITIZEN",
    "investment_interest_expense": 0,
    "is_computer_scientist": False,
    "is_executive_administrative_professional": False,
    "is_farmer_fisher": False,
    "is_blind": False,
    "is_full_time_college_student": False,
    "is_military": False,
    "is_paid_hourly": False,
    "is_pregnant": False,
    "is_tipped_occupation": False,
    "is_union_member_or_covered": False,
    "is_wic_at_nutritional_risk": True,
    "keogh_distributions": 0,
    "net_worth": 0,
    "other_health_insurance_premiums": 0,
    "other_type_retirement_account_distributions": 0,
    "partnership_s_corp_income_would_be_qualified": True,
    "pre_subsidy_rent": 0,
    "previous_year_income_available": False,
    "qualified_bdc_income": 0,
    "qualified_reit_and_ptp_income": 0,
    "recapture_of_investment_credit": 0,
    "reduced_price_school_meals_reported": 0,
    "regular_ira_distributions": 0,
    "reported_has_champva_health_coverage_at_interview": False,
    "reported_has_chip_health_coverage_at_interview": False,
    "reported_has_direct_purchase_health_coverage_at_interview": False,
    "reported_has_employer_sponsored_health_coverage_at_interview": False,
    "reported_has_indian_health_service_coverage_at_interview": False,
    "reported_has_marketplace_health_coverage_at_interview": False,
    "reported_has_means_tested_health_coverage_at_interview": False,
    "reported_has_medicaid_health_coverage_at_interview": False,
    "reported_has_medicare_health_coverage_at_interview": False,
    "reported_has_multiple_health_coverage_at_interview": False,
    "reported_has_non_marketplace_direct_purchase_health_coverage_at_interview": False,
    "reported_has_other_means_tested_health_coverage_at_interview": False,
    "reported_has_private_health_coverage_at_interview": False,
    "reported_has_public_health_coverage_at_interview": False,
    "reported_has_subsidized_marketplace_health_coverage_at_interview": False,
    "reported_has_tricare_health_coverage_at_interview": False,
    "reported_has_unsubsidized_marketplace_health_coverage_at_interview": False,
    "reported_has_va_health_coverage_at_interview": False,
    "reported_is_insured_at_interview": False,
    "reported_is_uninsured_at_interview": False,
    "rental_income_would_be_qualified": True,
    "roth_ira_distributions": 0,
    "second_home_mortgage_balance": 0.0,
    "second_home_mortgage_interest": 0.0,
    "second_home_mortgage_origination_year": 0,
    "selected_marketplace_plan_benchmark_ratio": 1.0,
    "self_employment_income_last_year": 0,
    "self_employment_income_would_be_qualified": True,
    "snap_reported": 0.0,
    "spm_unit_broadband_subsidy_reported": 0.0,
    "spm_unit_capped_housing_subsidy_reported": 0.0,
    "spm_unit_energy_subsidy_reported": 0.0,
    "spm_unit_federal_tax_reported": 0.0,
    "spm_unit_net_income_reported": 0.0,
    "spm_unit_payroll_tax_reported": 0.0,
    "spm_unit_state_tax_reported": 0.0,
    "spm_unit_total_income_reported": 0.0,
    "spm_unit_wic_reported": 0.0,
    "spm_unit_pre_subsidy_childcare_expenses": 0,
    "spm_unit_tenure_type": "RENTER",
    "ssi_reported": 0.0,
    "ssn_card_type": "CITIZEN",
    "sstb_self_employment_income_before_lsr": 0,
    "sstb_unadjusted_basis_qualified_property": 0.0,
    "sstb_w2_wages_from_qualified_business": 0.0,
    "strike_benefits": 0,
    "takes_up_aca_if_eligible": True,
    "takes_up_dc_ptc": True,
    "takes_up_early_head_start_if_eligible": True,
    "takes_up_eitc": True,
    "takes_up_head_start_if_eligible": True,
    "takes_up_medicaid_if_eligible": True,
    "takes_up_snap_if_eligible": True,
    "takes_up_ssi_if_eligible": True,
    "takes_up_tanf_if_eligible": True,
    "tax_exempt_401k_distributions": 0,
    "tax_exempt_403b_distributions": 0,
    "tax_exempt_ira_distributions": 0,
    "tax_exempt_sep_distributions": 0,
    "taxable_401k_distributions": 0,
    "taxable_403b_distributions": 0,
    "taxable_sep_distributions": 0,
    "tanf_reported": 0.0,
    "taxpayer_id_type": "VALID_SSN",
    "tenure_type": "NONE",
    "treasury_tipped_occupation_code": 0,
    "unadjusted_basis_qualified_property": 0,
    "unrecaptured_section_1250_gain": 0,
    "unreported_payroll_tax": 0,
    "veterans_benefits": 0,
    "w2_wages_from_qualified_business": 0,
    "weekly_hours_worked_before_lsr": 40.0,
    "weeks_unemployed": 0,
    "workers_compensation": 0,
    "would_claim_wic": True,
    "would_file_taxes_voluntarily": False,
}

POLICYENGINE_US_NUMERIC_ENUM_EXPORT_MAPS: dict[str, dict[float, str]] = {
    # Microplex imports ACS tenure as compact source codes. PolicyEngine-US
    # expects enum member names in its HDF5 inputs.
    "tenure_type": {
        0.0: "NONE",
        1.0: "OWNED_WITH_MORTGAGE",
        2.0: "RENTED",
    },
    # SPM tenure has no NONE value; no-cash/unknown tenure should not make the
    # exported dataset unreadable, so use the same renter fallback as PE-US.
    "spm_unit_tenure_type": {
        0.0: "RENTER",
        1.0: "OWNER_WITH_MORTGAGE",
        2.0: "RENTER",
    },
}

POLICYENGINE_US_LEGACY_CONTRACT_VARIABLE_ENTITIES: dict[str, str] = {
    "count_under_18": "person",
    "count_under_6": "person",
    "free_school_meals_reported": "spm_unit",
    "has_valid_ssn": "person",
    "is_tipped_occupation": "person",
    "other_type_retirement_account_distributions": "person",
    "reduced_price_school_meals_reported": "spm_unit",
    "regular_ira_distributions": "person",
    "reported_has_champva_health_coverage_at_interview": "person",
    "reported_has_chip_health_coverage_at_interview": "person",
    "reported_has_direct_purchase_health_coverage_at_interview": "person",
    "reported_has_employer_sponsored_health_coverage_at_interview": "person",
    "reported_has_indian_health_service_coverage_at_interview": "person",
    "reported_has_marketplace_health_coverage_at_interview": "person",
    "reported_has_means_tested_health_coverage_at_interview": "person",
    "reported_has_medicaid_health_coverage_at_interview": "person",
    "reported_has_medicare_health_coverage_at_interview": "person",
    "reported_has_multiple_health_coverage_at_interview": "person",
    "reported_has_non_marketplace_direct_purchase_health_coverage_at_interview": "person",
    "reported_has_other_means_tested_health_coverage_at_interview": "person",
    "reported_has_private_health_coverage_at_interview": "person",
    "reported_has_public_health_coverage_at_interview": "person",
    "reported_has_subsidized_marketplace_health_coverage_at_interview": "person",
    "reported_has_tricare_health_coverage_at_interview": "person",
    "reported_has_unsubsidized_marketplace_health_coverage_at_interview": "person",
    "reported_has_va_health_coverage_at_interview": "person",
    "reported_is_insured_at_interview": "person",
    "reported_is_uninsured_at_interview": "person",
    "roth_ira_distributions": "person",
    "snap_reported": "spm_unit",
    "spm_unit_broadband_subsidy_reported": "spm_unit",
    "spm_unit_capped_housing_subsidy_reported": "spm_unit",
    "spm_unit_energy_subsidy_reported": "spm_unit",
    "spm_unit_federal_tax_reported": "spm_unit",
    "spm_unit_payroll_tax_reported": "spm_unit",
    "spm_unit_state_tax_reported": "spm_unit",
    "spm_unit_wic_reported": "spm_unit",
    "tanf_reported": "person",
    "taxpayer_id_type": "person",
}

POLICYENGINE_US_STRUCTURAL_COMPUTED_EXPORT_VARIABLES: frozenset[str] = frozenset(
    {
        # Aligned with policyengine-us-data's final-H5 validation contract.
        # These are structural/cache fields, not model outputs.
        "person_id",
        "has_tin",
        "has_itin",
        "in_nyc",
    }
)

POLICYENGINE_US_DATA_OVERRIDABLE_COMPUTED_EXPORT_VARIABLES: frozenset[str] = frozenset(
    {
        # policyengine-us-data intentionally persists stronger source-data
        # inputs for these fallback formulas.
        "fsla_overtime_premium",
        "meets_ssi_disability_criteria",
    }
)

POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES: frozenset[str] = (
    POLICYENGINE_US_STRUCTURAL_COMPUTED_EXPORT_VARIABLES
    | POLICYENGINE_US_DATA_OVERRIDABLE_COMPUTED_EXPORT_VARIABLES
)


def compute_policyengine_us_definition_hash(
    constraints: tuple[PolicyEngineUSConstraint, ...] | list[PolicyEngineUSConstraint],
    *,
    parent_stratum_id: int | None = None,
) -> str:
    """Replicate policyengine-us-data's stratum definition hash logic."""
    parent_prefix = str(parent_stratum_id) if parent_stratum_id is not None else ""
    constraint_strings = sorted(
        f"{constraint.variable}|{constraint.operation}|{constraint.value}"
        for constraint in constraints
    )
    if not constraint_strings:
        fingerprint_text = parent_prefix
    else:
        fingerprint_text = parent_prefix + "\n" + "\n".join(constraint_strings)
    return hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()


class PolicyEngineUSDBTargetProvider:
    """Read PolicyEngine US target rows from the policyengine-us-data SQLite DB."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        validate: bool = True,
    ):
        self.db_path = Path(db_path)
        self.validate = validate

    def load_strata(
        self,
        stratum_ids: list[int] | tuple[int, ...] | None = None,
        *,
        include_ancestors: bool = True,
    ) -> dict[int, PolicyEngineUSStratum]:
        """Load strata with constraints and optional ancestor chain."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"PolicyEngine targets DB not found: {self.db_path}"
            )

        available_columns = self._table_columns("strata")
        has_definition_hash = "definition_hash" in available_columns
        has_parent_stratum_id = "parent_stratum_id" in available_columns

        remaining_ids = (
            {int(stratum_id) for stratum_id in stratum_ids}
            if stratum_ids is not None
            else None
        )
        loaded: dict[int, PolicyEngineUSStratum] = {}

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            while remaining_ids is None or remaining_ids:
                query, params = self._build_strata_query(
                    remaining_ids,
                    has_definition_hash=has_definition_hash,
                    has_parent_stratum_id=has_parent_stratum_id,
                )
                rows = conn.execute(query, params).fetchall()
                grouped = self._group_stratum_rows(rows)
                newly_loaded = {
                    stratum_id: stratum
                    for stratum_id, stratum in grouped.items()
                    if stratum_id not in loaded
                }
                loaded.update(newly_loaded)

                if remaining_ids is None or not include_ancestors:
                    break

                next_remaining = {
                    stratum.parent_stratum_id
                    for stratum in newly_loaded.values()
                    if stratum.parent_stratum_id is not None
                    and stratum.parent_stratum_id not in loaded
                }
                if not next_remaining:
                    break
                remaining_ids = next_remaining
        finally:
            conn.close()

        return loaded

    def load_targets(
        self,
        period: int | None = None,
        variables: list[str] | None = None,
        domain_variables: list[str] | None = None,
        domain_variable_values: list[str] | None = None,
        domain_variable_is_null: bool | None = None,
        geo_levels: list[str] | None = None,
        target_cells: list[dict[str, Any]] | None = None,
        target_ids: list[int] | None = None,
        stratum_ids: list[int] | None = None,
        reform_id: int = 0,
        active_only: bool = True,
        best_period: bool = True,
    ) -> list[PolicyEngineUSDBTarget]:
        """Load target rows with attached stratum constraints."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"PolicyEngine targets DB not found: {self.db_path}"
            )

        if self._has_target_overview_view() and best_period:
            return self._load_targets_via_target_overview(
                period=period,
                variables=variables,
                domain_variables=domain_variables,
                domain_variable_values=domain_variable_values,
                domain_variable_is_null=domain_variable_is_null,
                geo_levels=geo_levels,
                target_cells=target_cells,
                target_ids=target_ids,
                stratum_ids=stratum_ids,
                reform_id=reform_id,
                active_only=active_only,
            )
        if (
            domain_variables
            or domain_variable_values
            or geo_levels
            or domain_variable_is_null is not None
            or target_cells
        ):
            raise ValueError("domain/geography filters require a target_overview view")

        strata_columns = self._table_columns("strata")
        definition_hash_select = (
            "s.definition_hash AS definition_hash"
            if "definition_hash" in strata_columns
            else "NULL AS definition_hash"
        )
        parent_stratum_id_select = (
            "s.parent_stratum_id AS parent_stratum_id"
            if "parent_stratum_id" in strata_columns
            else "NULL AS parent_stratum_id"
        )
        clauses = ["t.reform_id = ?"]
        params: list[Any] = [reform_id]
        if active_only:
            clauses.append("t.active = 1")
        if period is not None:
            clauses.append("t.period = ?")
            params.append(period)
        if variables:
            placeholders = ", ".join("?" for _ in variables)
            clauses.append(f"t.variable IN ({placeholders})")
            params.extend(variables)
        if target_ids:
            placeholders = ", ".join("?" for _ in target_ids)
            clauses.append(f"t.target_id IN ({placeholders})")
            params.extend(target_ids)
        if stratum_ids:
            placeholders = ", ".join("?" for _ in stratum_ids)
            clauses.append(f"t.stratum_id IN ({placeholders})")
            params.extend(stratum_ids)

        where_clause = " AND ".join(clauses)
        query = f"""
            SELECT
                t.target_id,
                t.variable,
                t.period,
                t.stratum_id,
                t.reform_id,
                t.value AS target_value,
                t.active,
                t.tolerance,
                t.source,
                t.notes,
                NULL AS geo_level,
                NULL AS geographic_id,
                NULL AS domain_variable,
                {definition_hash_select},
                {parent_stratum_id_select},
                sc.constraint_variable,
                sc.operation,
                sc.value AS constraint_value
            FROM targets AS t
            JOIN strata AS s
                ON t.stratum_id = s.stratum_id
            LEFT JOIN stratum_constraints AS sc
                ON t.stratum_id = sc.stratum_id
            WHERE {where_clause}
            ORDER BY t.target_id, sc.constraint_variable, sc.operation, sc.value
        """

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        targets = self._group_target_rows(rows)
        if self.validate:
            self._validate_targets(targets)
        return targets

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        """Load canonical targets through the core provider protocol."""
        from microplex_us.targets import policyengine_db_targets_to_canonical_set

        query = query or TargetQuery()
        provider_filters = query.provider_filters
        best_period = bool(provider_filters.get("best_period", True))
        canonical_targets = policyengine_db_targets_to_canonical_set(
            self.load_targets(
                period=query.period if isinstance(query.period, int) else None,
                variables=provider_filters.get("variables"),
                domain_variables=provider_filters.get("domain_variables"),
                domain_variable_values=provider_filters.get("domain_variable_values"),
                domain_variable_is_null=provider_filters.get("domain_variable_is_null"),
                geo_levels=provider_filters.get("geo_levels"),
                target_cells=provider_filters.get("target_cells"),
                target_ids=provider_filters.get("target_ids"),
                stratum_ids=provider_filters.get("stratum_ids"),
                reform_id=int(provider_filters.get("reform_id", 0)),
                active_only=bool(provider_filters.get("active_only", True)),
                best_period=best_period,
            ),
            default_entity=provider_filters.get("default_entity", EntityType.HOUSEHOLD),
            entity_overrides=provider_filters.get("entity_overrides"),
        )
        return apply_target_query(
            canonical_targets,
            TargetQuery(
                period=None if best_period else query.period,
                entity=query.entity,
                names=query.names,
                metadata_filters=query.metadata_filters,
            ),
        )

    def _has_target_overview_view(self) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'view' AND name = 'target_overview'
                """
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def _load_targets_via_target_overview(
        self,
        *,
        period: int | None,
        variables: list[str] | None,
        domain_variables: list[str] | None,
        domain_variable_values: list[str] | None,
        domain_variable_is_null: bool | None,
        geo_levels: list[str] | None,
        target_cells: list[dict[str, Any]] | None,
        target_ids: list[int] | None,
        stratum_ids: list[int] | None,
        reform_id: int,
        active_only: bool,
    ) -> list[PolicyEngineUSDBTarget]:
        strata_columns = self._table_columns("strata")
        definition_hash_select = (
            "s.definition_hash AS definition_hash"
            if "definition_hash" in strata_columns
            else "NULL AS definition_hash"
        )
        parent_stratum_id_select = (
            "s.parent_stratum_id AS parent_stratum_id"
            if "parent_stratum_id" in strata_columns
            else "NULL AS parent_stratum_id"
        )
        clauses = ["t.reform_id = ?"]
        params: list[Any] = [reform_id]
        if active_only:
            clauses.append("tv.active = 1")
        if variables:
            placeholders = ", ".join("?" for _ in variables)
            clauses.append(f"tv.variable IN ({placeholders})")
            params.extend(variables)
        if target_ids:
            placeholders = ", ".join("?" for _ in target_ids)
            clauses.append(f"tv.target_id IN ({placeholders})")
            params.extend(target_ids)
        if stratum_ids:
            placeholders = ", ".join("?" for _ in stratum_ids)
            clauses.append(f"tv.stratum_id IN ({placeholders})")
            params.extend(stratum_ids)
        if geo_levels:
            placeholders = ", ".join("?" for _ in geo_levels)
            clauses.append(f"tv.geo_level IN ({placeholders})")
            params.extend(geo_levels)
        if domain_variables:
            domain_clauses = [
                "instr(',' || coalesce(tv.domain_variable, '') || ',', ',' || ? || ',') > 0"
                for _ in domain_variables
            ]
            clauses.append("(" + " OR ".join(domain_clauses) + ")")
            params.extend(domain_variables)
        if domain_variable_values:
            placeholders = ", ".join("?" for _ in domain_variable_values)
            clauses.append(f"coalesce(tv.domain_variable, '') IN ({placeholders})")
            params.extend(domain_variable_values)
        if domain_variable_is_null is True:
            clauses.append("coalesce(tv.domain_variable, '') = ''")
        elif domain_variable_is_null is False:
            clauses.append("coalesce(tv.domain_variable, '') <> ''")
        if target_cells:
            clauses.append(self._build_target_cell_clause(target_cells, params))

        time_period = period if period is not None else 9999
        where_clause = " AND ".join(clauses) if clauses else "1=1"
        query = f"""
            WITH filtered_targets AS (
                SELECT
                    tv.target_id,
                    tv.stratum_id,
                    tv.variable,
                    tv.value,
                    tv.period,
                    tv.active,
                    tv.geo_level,
                    tv.geographic_id,
                    tv.domain_variable
                FROM target_overview tv
                JOIN targets t ON tv.target_id = t.target_id
                WHERE {where_clause}
            ),
            best_periods AS (
                SELECT
                    stratum_id,
                    variable,
                    CASE
                        WHEN MAX(CASE WHEN period <= ? THEN period END) IS NOT NULL
                        THEN MAX(CASE WHEN period <= ? THEN period END)
                        ELSE MIN(period)
                    END AS best_period
                FROM filtered_targets
                GROUP BY stratum_id, variable
            )
            SELECT
                ft.target_id,
                ft.variable,
                ft.period,
                ft.stratum_id,
                t.reform_id,
                ft.value AS target_value,
                t.active,
                t.tolerance,
                t.source,
                t.notes,
                ft.geo_level,
                ft.geographic_id,
                ft.domain_variable,
                {definition_hash_select},
                {parent_stratum_id_select},
                sc.constraint_variable,
                sc.operation,
                sc.value AS constraint_value
            FROM filtered_targets ft
            JOIN best_periods bp
                ON ft.stratum_id = bp.stratum_id
                AND ft.variable = bp.variable
                AND ft.period = bp.best_period
            JOIN targets t
                ON ft.target_id = t.target_id
            JOIN strata s
                ON ft.stratum_id = s.stratum_id
            LEFT JOIN stratum_constraints sc
                ON ft.stratum_id = sc.stratum_id
            ORDER BY ft.target_id, sc.constraint_variable, sc.operation, sc.value
        """

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(query, [*params, time_period, time_period]).fetchall()
        finally:
            conn.close()

        targets = self._group_target_rows(rows)
        if self.validate:
            self._validate_targets(targets)
        return targets

    def _build_target_cell_clause(
        self,
        target_cells: list[dict[str, Any]],
        params: list[Any],
    ) -> str:
        cell_clauses: list[str] = []
        for raw_cell in target_cells:
            cell = PolicyEngineUSTargetCell(
                variable=str(raw_cell["variable"]),
                geo_level=(
                    None
                    if raw_cell.get("geo_level") is None
                    else str(raw_cell["geo_level"])
                ),
                domain_variable=(
                    None
                    if "domain_variable" in raw_cell
                    and raw_cell.get("domain_variable") is None
                    else (
                        str(raw_cell["domain_variable"])
                        if raw_cell.get("domain_variable") is not None
                        else None
                    )
                ),
                geographic_id=(
                    None
                    if raw_cell.get("geographic_id") is None
                    else str(raw_cell["geographic_id"])
                ),
            )
            subclauses = ["tv.variable = ?"]
            params.append(cell.variable)
            if cell.geo_level is not None:
                subclauses.append("tv.geo_level = ?")
                params.append(cell.geo_level)
            if "domain_variable" in raw_cell:
                if cell.domain_variable is None:
                    subclauses.append("coalesce(tv.domain_variable, '') = ''")
                else:
                    subclauses.append(
                        "instr(',' || coalesce(tv.domain_variable, '') || ',', ',' || ? || ',') > 0"
                    )
                    params.append(cell.domain_variable)
            if cell.geographic_id is not None:
                subclauses.append("coalesce(tv.geographic_id, '') = ?")
                params.append(cell.geographic_id)
            cell_clauses.append("(" + " AND ".join(subclauses) + ")")
        return "(" + " OR ".join(cell_clauses) + ")"

    def _group_target_rows(
        self,
        rows: list[sqlite3.Row],
    ) -> list[PolicyEngineUSDBTarget]:
        grouped: dict[int, dict[str, Any]] = {}
        for row in rows:
            target_id = int(row["target_id"])
            item = grouped.setdefault(
                target_id,
                {
                    "target_id": target_id,
                    "variable": row["variable"],
                    "period": int(row["period"]),
                    "stratum_id": int(row["stratum_id"]),
                    "reform_id": int(row["reform_id"]),
                    "value": float(row["target_value"]),
                    "active": bool(row["active"]),
                    "tolerance": (
                        float(row["tolerance"])
                        if row["tolerance"] is not None
                        else None
                    ),
                    "source": row["source"],
                    "notes": row["notes"],
                    "geo_level": row["geo_level"],
                    "geographic_id": row["geographic_id"],
                    "domain_variable": row["domain_variable"],
                    "definition_hash": row["definition_hash"],
                    "parent_stratum_id": (
                        int(row["parent_stratum_id"])
                        if row["parent_stratum_id"] is not None
                        else None
                    ),
                    "constraints": [],
                },
            )
            if row["constraint_variable"] is not None:
                item["constraints"].append(
                    PolicyEngineUSConstraint(
                        variable=row["constraint_variable"],
                        operation=row["operation"],
                        value=row["constraint_value"],
                    )
                )

        return [
            PolicyEngineUSDBTarget(
                target_id=item["target_id"],
                variable=item["variable"],
                period=item["period"],
                stratum_id=item["stratum_id"],
                reform_id=item["reform_id"],
                value=item["value"],
                active=item["active"],
                tolerance=item["tolerance"],
                source=item["source"],
                notes=item["notes"],
                geo_level=item["geo_level"],
                geographic_id=item["geographic_id"],
                domain_variable=item["domain_variable"],
                definition_hash=item["definition_hash"],
                parent_stratum_id=item["parent_stratum_id"],
                constraints=tuple(item["constraints"]),
            )
            for item in grouped.values()
        ]

    def _build_strata_query(
        self,
        stratum_ids: set[int] | None,
        *,
        has_definition_hash: bool,
        has_parent_stratum_id: bool,
    ) -> tuple[str, list[Any]]:
        clauses = []
        params: list[Any] = []
        if stratum_ids is not None:
            if not stratum_ids:
                return (
                    """
                    SELECT
                        s.stratum_id,
                        NULL AS definition_hash,
                        NULL AS parent_stratum_id,
                        NULL AS constraint_variable,
                        NULL AS operation,
                        NULL AS constraint_value
                    FROM strata AS s
                    WHERE 1 = 0
                    """,
                    [],
                )
            placeholders = ", ".join("?" for _ in sorted(stratum_ids))
            clauses.append(f"s.stratum_id IN ({placeholders})")
            params.extend(sorted(stratum_ids))

        where_clause = "WHERE " + " AND ".join(clauses) if clauses else ""
        definition_hash_select = (
            "s.definition_hash AS definition_hash"
            if has_definition_hash
            else "NULL AS definition_hash"
        )
        parent_stratum_id_select = (
            "s.parent_stratum_id AS parent_stratum_id"
            if has_parent_stratum_id
            else "NULL AS parent_stratum_id"
        )
        query = f"""
            SELECT
                s.stratum_id,
                {definition_hash_select},
                {parent_stratum_id_select},
                sc.constraint_variable,
                sc.operation,
                sc.value AS constraint_value
            FROM strata AS s
            LEFT JOIN stratum_constraints AS sc
                ON s.stratum_id = sc.stratum_id
            {where_clause}
            ORDER BY s.stratum_id, sc.constraint_variable, sc.operation, sc.value
        """
        return query, params

    def _group_stratum_rows(
        self,
        rows: list[sqlite3.Row],
    ) -> dict[int, PolicyEngineUSStratum]:
        grouped: dict[int, dict[str, Any]] = {}
        for row in rows:
            stratum_id = int(row["stratum_id"])
            item = grouped.setdefault(
                stratum_id,
                {
                    "stratum_id": stratum_id,
                    "definition_hash": row["definition_hash"],
                    "parent_stratum_id": (
                        int(row["parent_stratum_id"])
                        if row["parent_stratum_id"] is not None
                        else None
                    ),
                    "constraints": [],
                },
            )
            if row["constraint_variable"] is not None:
                item["constraints"].append(
                    PolicyEngineUSConstraint(
                        variable=row["constraint_variable"],
                        operation=row["operation"],
                        value=row["constraint_value"],
                    )
                )
        return {
            stratum_id: PolicyEngineUSStratum(
                stratum_id=stratum_id,
                definition_hash=item["definition_hash"],
                parent_stratum_id=item["parent_stratum_id"],
                constraints=tuple(item["constraints"]),
            )
            for stratum_id, item in grouped.items()
        }

    def _table_columns(self, table_name: str) -> set[str]:
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        finally:
            conn.close()
        return {str(row[1]) for row in rows}

    def _validate_targets(self, targets: list[PolicyEngineUSDBTarget]) -> None:
        if not targets:
            return
        strata = self.load_strata([target.stratum_id for target in targets])
        missing_strata = sorted(
            {target.stratum_id for target in targets if target.stratum_id not in strata}
        )
        if missing_strata:
            raise PolicyEngineUSTargetValidationError(
                f"Missing strata for target rows: {missing_strata}"
            )
        self._validate_strata(strata)

    def _validate_strata(
        self,
        strata: dict[int, PolicyEngineUSStratum],
    ) -> None:
        errors: list[str] = []
        for stratum in strata.values():
            if stratum.definition_hash is not None:
                expected_hash = compute_policyengine_us_definition_hash(
                    stratum.constraints,
                    parent_stratum_id=stratum.parent_stratum_id,
                )
                if stratum.definition_hash != expected_hash:
                    errors.append(
                        "Stratum "
                        f"{stratum.stratum_id} has definition_hash "
                        f"{stratum.definition_hash!r}, expected {expected_hash!r}"
                    )

            if stratum.parent_stratum_id is None:
                continue
            parent = strata.get(stratum.parent_stratum_id)
            if parent is None:
                errors.append(
                    f"Stratum {stratum.stratum_id} references missing parent "
                    f"{stratum.parent_stratum_id}"
                )
                continue

            parent_vars = {constraint.variable for constraint in parent.constraints}
            child_vars = {constraint.variable for constraint in stratum.constraints}
            geographic_error = self._validate_geographic_consistency(parent, stratum)
            if geographic_error is not None:
                errors.append(geographic_error)

            if (
                parent_vars <= GEOGRAPHIC_CONSTRAINT_VARIABLES
                and child_vars <= GEOGRAPHIC_CONSTRAINT_VARIABLES
            ):
                continue

            parent_constraints = {
                self._constraint_signature(constraint)
                for constraint in parent.constraints
            }
            child_constraints = {
                self._constraint_signature(constraint)
                for constraint in stratum.constraints
            }
            missing_parent_constraints = sorted(parent_constraints - child_constraints)
            if missing_parent_constraints:
                errors.append(
                    f"Stratum {stratum.stratum_id} is missing inherited parent "
                    f"constraints from {stratum.parent_stratum_id}: "
                    f"{missing_parent_constraints}"
                )

        if errors:
            raise PolicyEngineUSTargetValidationError("\n".join(errors))

    def _constraint_signature(
        self,
        constraint: PolicyEngineUSConstraint,
    ) -> tuple[str, str, str]:
        return (
            constraint.variable,
            constraint.operation,
            self._normalize_constraint_value(constraint.variable, constraint.value),
        )

    def _normalize_constraint_value(self, variable: str, value: str) -> str:
        if variable in GEOGRAPHIC_CONSTRAINT_VARIABLES:
            return str(int(value))
        return str(value)

    def _validate_geographic_consistency(
        self,
        parent: PolicyEngineUSStratum,
        child: PolicyEngineUSStratum,
    ) -> str | None:
        parent_equalities = {
            constraint.variable: constraint.value
            for constraint in parent.constraints
            if constraint.operation == "=="
        }
        child_equalities = {
            constraint.variable: constraint.value
            for constraint in child.constraints
            if constraint.operation == "=="
        }

        for variable in GEOGRAPHIC_CONSTRAINT_VARIABLES:
            if variable not in parent_equalities or variable not in child_equalities:
                continue
            if int(parent_equalities[variable]) != int(child_equalities[variable]):
                return (
                    f"Stratum {child.stratum_id} has geographic constraint "
                    f"{variable}={child_equalities[variable]!r} but parent "
                    f"{parent.stratum_id} has {variable}={parent_equalities[variable]!r}"
                )

        if (
            "state_fips" in parent_equalities
            and "congressional_district_geoid" in child_equalities
        ):
            parent_state = int(parent_equalities["state_fips"])
            district_state = (
                int(child_equalities["congressional_district_geoid"]) // 100
            )
            if district_state != parent_state:
                return (
                    f"Stratum {child.stratum_id} has congressional_district_geoid="
                    f"{child_equalities['congressional_district_geoid']!r} which belongs "
                    f"to state {district_state}, not parent state {parent_state}"
                )
        return None

    def to_quantity_targets(
        self,
        variable_column_map: dict[str, str],
        period: int | None = None,
        reform_id: int = 0,
    ) -> tuple[PolicyEngineUSQuantityTarget, ...]:
        """Convert unconstrained DB rows into quantity targets for calibration."""
        quantity_targets: list[PolicyEngineUSQuantityTarget] = []
        for target in self.load_targets(period=period, reform_id=reform_id):
            if not target.is_unconstrained:
                continue
            column = variable_column_map.get(target.variable)
            if column is None:
                continue
            quantity_targets.append(
                PolicyEngineUSQuantityTarget(
                    name=target.variable,
                    variable=target.variable,
                    column=column,
                    period=target.period,
                )
            )
        return tuple(quantity_targets)


def compile_policyengine_us_household_linear_constraints(
    targets: (
        tuple[PolicyEngineUSDBTarget | TargetSpec, ...]
        | list[PolicyEngineUSDBTarget | TargetSpec]
    ),
    tables: PolicyEngineUSEntityTableBundle,
    *,
    variable_bindings: dict[str, PolicyEngineUSVariableBinding] | None = None,
    household_id_column: str = "household_id",
) -> tuple[LinearConstraint, ...]:
    """Compile PE target rows into household-level linear calibration rows."""
    households = tables.households
    if household_id_column not in households.columns:
        raise ValueError(
            f"Household table must contain '{household_id_column}' for calibration"
        )
    if households[household_id_column].duplicated().any():
        raise ValueError("Household calibration table must have unique household ids")

    household_ids = pd.Index(households[household_id_column], name=household_id_column)
    bindings = {
        **DEFAULT_POLICYENGINE_US_VARIABLE_BINDINGS,
        **(variable_bindings or {}),
    }

    compiled: list[LinearConstraint] = []
    for target in targets:
        coefficients = _compile_household_coefficients(
            target=target,
            tables=tables,
            bindings=bindings,
            household_ids=household_ids,
            household_id_column=household_id_column,
        )
        compiled.append(
            LinearConstraint(
                name=_policyengine_target_name(target),
                coefficients=coefficients,
                target=_target_value(target),
            )
        )

    return tuple(compiled)


class PolicyEngineUSMicrosimulationAdapter:
    """Thin wrapper around a PolicyEngine US microsimulation instance."""

    def __init__(self, simulation: Any):
        self.simulation = simulation

    @property
    def tax_benefit_system(self) -> Any:
        """Return the underlying PolicyEngine tax-benefit system."""
        tax_benefit_system = getattr(
            self.simulation,
            "tax_benefit_system",
            getattr(self.simulation, "system", None),
        )
        return getattr(tax_benefit_system, "system", tax_benefit_system)

    @classmethod
    def from_dataset(
        cls,
        dataset: str | Path | Any,
        *,
        dataset_year: int | None = None,
        simulation_cls: Any | None = None,
        **kwargs: Any,
    ) -> PolicyEngineUSMicrosimulationAdapter:
        """Construct an adapter from a dataset path or PolicyEngine dataset class."""
        if simulation_cls is None:
            try:
                import policyengine_us
            except ImportError as exc:
                raise ImportError(
                    "policyengine_us is required to build a microsimulation adapter"
                ) from exc
            simulation_cls = policyengine_us.Microsimulation

        sim_kwargs = dict(kwargs)
        sim_kwargs["dataset"] = str(dataset) if isinstance(dataset, Path) else dataset
        if dataset_year is not None:
            sim_kwargs["dataset_year"] = dataset_year
        try:
            simulation = simulation_cls(**sim_kwargs)
        except TypeError as exc:
            if dataset_year is None or "dataset_year" not in str(exc):
                raise
            sim_kwargs.pop("dataset_year", None)
            simulation = simulation_cls(**sim_kwargs)
        return cls(simulation=simulation)

    def calculate(
        self,
        variable: str,
        *,
        period: int | None = None,
        map_to: str | None = None,
    ) -> Any:
        """Calculate a PolicyEngine variable."""
        if map_to is None:
            return self.simulation.calculate(variable, period)
        return self.simulation.calculate(variable, period, map_to=map_to)

    def variable_entity(self, variable: str) -> EntityType:
        """Resolve the Microplex entity type for a PolicyEngine variable."""
        return _resolve_policyengine_variable_entity(
            variable,
            tax_benefit_system=self.tax_benefit_system,
        )

    def compute_targets(
        self,
        quantity_targets: tuple[PolicyEngineUSQuantityTarget, ...],
    ) -> dict[str, float]:
        """Compute aggregate targets from the configured PE quantity specs."""
        results: dict[str, float] = {}
        for target in quantity_targets:
            values = self.calculate(
                target.variable,
                period=target.period,
                map_to=target.map_to,
            )
            results[target.name] = self._aggregate(values, target.aggregation)
        return results

    def _aggregate(
        self,
        values: Any,
        aggregation: Literal["sum", "mean", "count_positive"],
    ) -> float:
        if aggregation == "sum":
            if hasattr(values, "sum"):
                return float(values.sum())
            return float(np.asarray(values).sum())
        if aggregation == "mean":
            if hasattr(values, "mean"):
                return float(values.mean())
            return float(np.asarray(values).mean())
        if aggregation == "count_positive":
            positive = values > 0
            if hasattr(positive, "sum"):
                return float(positive.sum())
            return float(np.asarray(positive).sum())
        raise ValueError(f"Unsupported aggregation: {aggregation}")


def _policyengine_us_variable_is_computed_export(metadata: Any) -> bool:
    """Return whether PE-US should compute this variable in final datasets."""
    return (
        bool(getattr(metadata, "formulas", {}) or {})
        or bool(getattr(metadata, "adds", None))
        or bool(getattr(metadata, "subtracts", None))
    )


def detect_policyengine_computed_export_variables(
    tax_benefit_system: Any,
    input_variables: list[str] | tuple[str, ...],
) -> set[str]:
    """Detect exported variables computed by PolicyEngine-US."""
    computed_exports: set[str] = set()
    variables = getattr(tax_benefit_system, "variables", {})

    for variable_name in input_variables:
        if variable_name in POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES:
            continue
        variable = variables.get(variable_name)
        if variable is None:
            continue
        if _policyengine_us_variable_is_computed_export(variable):
            computed_exports.add(variable_name)

    return computed_exports


def detect_policyengine_pseudo_inputs(
    tax_benefit_system: Any,
    input_variables: list[str] | tuple[str, ...],
) -> set[str]:
    """Detect PE-computed variables that must not be persisted as inputs."""
    return detect_policyengine_computed_export_variables(
        tax_benefit_system,
        input_variables,
    )


def resolve_policyengine_excluded_export_variables(
    tax_benefit_system: Any,
    exported_inputs: list[str] | tuple[str, ...],
    *,
    direct_override_variables: tuple[str, ...] = (),
) -> set[str]:
    """Resolve PE-computed exports to exclude from final H5 datasets."""
    return detect_policyengine_computed_export_variables(
        tax_benefit_system,
        exported_inputs,
    )


def subset_policyengine_tables_by_households(
    tables: PolicyEngineUSEntityTableBundle,
    household_ids: np.ndarray | pd.Index,
) -> PolicyEngineUSEntityTableBundle:
    """Slice an entity bundle to a subset of household_ids, preserving order.

    The returned bundle's ``households`` frame is reordered to match the
    order of ``household_ids``; related entity tables retain their own
    internal order but are filtered to only rows whose ``household_id``
    is in the selection.
    """
    selected = pd.Index(household_ids, name="household_id")
    order = pd.Series(np.arange(len(selected)), index=selected)

    households = tables.households.loc[
        tables.households["household_id"].isin(selected)
    ].copy()
    households = (
        households.assign(_hh_order=households["household_id"].map(order))
        .sort_values("_hh_order")
        .drop(columns="_hh_order")
        .reset_index(drop=True)
    )

    def _slice(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        return df.loc[df["household_id"].isin(selected)].reset_index(drop=True)

    return PolicyEngineUSEntityTableBundle(
        households=households,
        persons=_slice(tables.persons),
        tax_units=_slice(tables.tax_units),
        spm_units=_slice(tables.spm_units),
        families=_slice(tables.families),
        marital_units=_slice(tables.marital_units),
    )


def _concat_bundles(
    bundles: list[PolicyEngineUSEntityTableBundle],
) -> PolicyEngineUSEntityTableBundle:
    """Concatenate a list of entity bundles into one, preserving order."""

    def _join(field: str) -> pd.DataFrame | None:
        frames = [getattr(b, field) for b in bundles if getattr(b, field) is not None]
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    return PolicyEngineUSEntityTableBundle(
        households=_join("households"),
        persons=_join("persons"),
        tax_units=_join("tax_units"),
        spm_units=_join("spm_units"),
        families=_join("families"),
        marital_units=_join("marital_units"),
    )


def materialize_policyengine_us_variables(
    tables: PolicyEngineUSEntityTableBundle,
    *,
    variables: tuple[str, ...] | list[str],
    period: int,
    dataset_year: int | None = None,
    simulation_cls: Any | None = None,
    microsimulation_kwargs: dict[str, Any] | None = None,
    temp_dir: str | Path | None = None,
    direct_override_variables: tuple[str, ...] = (),
    batch_size: int | None = None,
) -> tuple[PolicyEngineUSEntityTableBundle, dict[str, PolicyEngineUSVariableBinding]]:
    """Calculate PolicyEngine variables on a temporary export and attach them to tables.

    Memory control: when ``batch_size`` is set, the function loops over
    disjoint household chunks of that size, materializing variables on
    each chunk (one temp h5 + one Microsimulation per chunk) and
    concatenating results. Peak Microsimulation working set drops from
    O(n_households) to O(batch_size) with no change in output — this is
    additive for the per-household scalar variables we use as calibration
    targets (employment income, EITC, CTC, federal income tax, etc.), and
    the per-chunk Microsims are independent of each other.

    Variables with cross-household semantics (national quantile
    thresholds, poverty rates that depend on the full income
    distribution) would be incorrect under batching and are not supported
    when ``batch_size`` is not ``None``. Use ``batch_size=None`` for
    those.
    """
    if batch_size is not None and batch_size > 0:
        n_households = len(tables.households)
        if n_households > batch_size:
            chunk_bundles: list[PolicyEngineUSEntityTableBundle] = []
            chunk_bindings: dict[str, PolicyEngineUSVariableBinding] = {}
            household_ids = tables.households["household_id"].to_numpy()
            for start in range(0, n_households, batch_size):
                end = min(start + batch_size, n_households)
                chunk_ids = household_ids[start:end]
                chunk_tables = subset_policyengine_tables_by_households(
                    tables, chunk_ids
                )
                chunk_result, chunk_binding = materialize_policyengine_us_variables(
                    chunk_tables,
                    variables=variables,
                    period=period,
                    dataset_year=dataset_year,
                    simulation_cls=simulation_cls,
                    microsimulation_kwargs=microsimulation_kwargs,
                    temp_dir=temp_dir,
                    direct_override_variables=direct_override_variables,
                    batch_size=None,
                )
                chunk_bundles.append(chunk_result)
                chunk_bindings.update(chunk_binding)
            return _concat_bundles(chunk_bundles), chunk_bindings
    requested_variables = tuple(dict.fromkeys(str(variable) for variable in variables))
    if not requested_variables:
        return tables, {}

    tax_benefit_system = _resolve_policyengine_us_tax_benefit_system(
        simulation_cls=simulation_cls
    )
    export_maps = build_policyengine_us_export_variable_maps(
        tables,
        tax_benefit_system=tax_benefit_system,
        direct_override_variables=direct_override_variables,
    )
    exported_inputs = sorted(
        {
            target
            for variable_map in export_maps.values()
            for target in variable_map.values()
        }
    )
    excluded_variables = resolve_policyengine_excluded_export_variables(
        tax_benefit_system,
        exported_inputs,
        direct_override_variables=direct_override_variables,
    )
    arrays = build_policyengine_us_time_period_arrays(
        tables,
        period=period,
        household_variable_map=export_maps["household"],
        person_variable_map=export_maps["person"],
        tax_unit_variable_map=export_maps["tax_unit"],
        spm_unit_variable_map=export_maps["spm_unit"],
        family_variable_map=export_maps["family"],
    )

    temp_parent = Path(temp_dir) if temp_dir is not None else None
    with TemporaryDirectory(dir=temp_parent) as directory:
        dataset_path = Path(directory) / "policyengine_us_materialize.h5"
        write_policyengine_us_time_period_dataset(
            arrays,
            dataset_path,
            excluded_variables=excluded_variables,
            tax_benefit_system=tax_benefit_system,
        )
        adapter = PolicyEngineUSMicrosimulationAdapter.from_dataset(
            dataset_path,
            dataset_year=dataset_year or period,
            simulation_cls=simulation_cls,
            **(microsimulation_kwargs or {}),
        )
        return _attach_policyengine_variables_to_tables(
            tables,
            variables=requested_variables,
            period=period,
            adapter=adapter,
        )


def materialize_policyengine_us_variables_safely(
    tables: PolicyEngineUSEntityTableBundle,
    *,
    variables: tuple[str, ...] | list[str],
    period: int,
    dataset_year: int | None = None,
    simulation_cls: Any | None = None,
    microsimulation_kwargs: dict[str, Any] | None = None,
    temp_dir: str | Path | None = None,
    direct_override_variables: tuple[str, ...] = (),
    batch_size: int | None = None,
) -> PolicyEngineUSVariableMaterializationResult:
    """Materialize PE variables, degrading to per-variable failures when needed.

    ``batch_size`` forwards to :func:`materialize_policyengine_us_variables`.
    With a non-``None`` positive value, the full-dataset Microsimulation
    (25–35 GB peak at 1.5M households) is replaced with N per-chunk
    Microsims (each ~2–3 GB). Results are concatenated; output is
    identical for per-household scalar variables.
    """
    requested_variables = tuple(dict.fromkeys(str(variable) for variable in variables))
    if not requested_variables:
        return PolicyEngineUSVariableMaterializationResult(
            tables=tables,
            bindings={},
        )

    try:
        materialized_tables, materialized_bindings = (
            materialize_policyengine_us_variables(
                tables,
                variables=requested_variables,
                period=period,
                dataset_year=dataset_year,
                simulation_cls=simulation_cls,
                microsimulation_kwargs=microsimulation_kwargs,
                temp_dir=temp_dir,
                direct_override_variables=direct_override_variables,
                batch_size=batch_size,
            )
        )
    except Exception:
        return _materialize_policyengine_us_variables_one_by_one(
            tables,
            requested_variables,
            period=period,
            dataset_year=dataset_year,
            simulation_cls=simulation_cls,
            microsimulation_kwargs=microsimulation_kwargs,
            temp_dir=temp_dir,
            direct_override_variables=direct_override_variables,
        )

    return PolicyEngineUSVariableMaterializationResult(
        tables=materialized_tables,
        bindings=materialized_bindings,
        materialized_variables=requested_variables,
    )


def _materialize_policyengine_us_variables_one_by_one(
    tables: PolicyEngineUSEntityTableBundle,
    requested_variables: tuple[str, ...],
    *,
    period: int,
    dataset_year: int | None,
    simulation_cls: Any | None,
    microsimulation_kwargs: dict[str, Any] | None,
    temp_dir: str | Path | None,
    direct_override_variables: tuple[str, ...],
) -> PolicyEngineUSVariableMaterializationResult:
    working_tables = _copy_policyengine_us_entity_tables(tables)
    bindings: dict[str, PolicyEngineUSVariableBinding] = {}
    materialized_variables: list[str] = []
    failed_variables: dict[str, str] = {}

    for variable in requested_variables:
        try:
            materialized_tables, materialized_bindings = (
                materialize_policyengine_us_variables(
                    working_tables,
                    variables=(variable,),
                    period=period,
                    dataset_year=dataset_year,
                    simulation_cls=simulation_cls,
                    microsimulation_kwargs=microsimulation_kwargs,
                    temp_dir=temp_dir,
                    direct_override_variables=direct_override_variables,
                )
            )
        except Exception as exc:
            failed_variables[variable] = f"{type(exc).__name__}: {exc}"
            continue
        working_tables = _merge_materialized_policyengine_bindings(
            working_tables,
            source_tables=materialized_tables,
            bindings=materialized_bindings,
        )
        bindings.update(materialized_bindings)
        materialized_variables.append(variable)

    return PolicyEngineUSVariableMaterializationResult(
        tables=working_tables,
        bindings=bindings,
        materialized_variables=tuple(materialized_variables),
        failed_variables=failed_variables,
    )


def _copy_policyengine_us_entity_tables(
    tables: PolicyEngineUSEntityTableBundle,
) -> PolicyEngineUSEntityTableBundle:
    return PolicyEngineUSEntityTableBundle(
        households=tables.households.copy(),
        persons=tables.persons.copy() if tables.persons is not None else None,
        tax_units=tables.tax_units.copy() if tables.tax_units is not None else None,
        spm_units=tables.spm_units.copy() if tables.spm_units is not None else None,
        families=tables.families.copy() if tables.families is not None else None,
        marital_units=(
            tables.marital_units.copy() if tables.marital_units is not None else None
        ),
    )


def _merge_materialized_policyengine_bindings(
    destination_tables: PolicyEngineUSEntityTableBundle,
    *,
    source_tables: PolicyEngineUSEntityTableBundle,
    bindings: dict[str, PolicyEngineUSVariableBinding],
) -> PolicyEngineUSEntityTableBundle:
    merged_tables = _copy_policyengine_us_entity_tables(destination_tables)
    for binding in bindings.values():
        if binding.column is None:
            continue
        source_table = source_tables.table_for(binding.entity)
        destination_table = merged_tables.table_for(binding.entity)
        destination_table[binding.column] = source_table[binding.column].to_numpy(
            copy=True
        )
    return merged_tables


def load_policyengine_us_entity_tables(
    dataset: str | Path | Any,
    *,
    period: int | str,
    variables: tuple[str, ...] | list[str] | None = None,
) -> PolicyEngineUSEntityTableBundle:
    """Load a PE-US time-period dataset into a multientity table bundle."""
    period_key = str(period)
    requested_variables = (
        None if variables is None else {str(variable) for variable in variables}
    )
    try:
        tax_benefit_system = _resolve_policyengine_us_tax_benefit_system(
            simulation_cls=None
        )
    except (ImportError, ValueError):
        tax_benefit_system = None
    arrays = _load_policyengine_us_period_arrays(
        dataset,
        period_key=period_key,
        variables=requested_variables,
    )

    required_structural = {
        "household_id",
        "person_id",
        "person_household_id",
    }
    missing = sorted(required_structural - set(arrays))
    if missing:
        raise ValueError(
            "PolicyEngine US dataset is missing required structural arrays: "
            + ", ".join(missing)
        )

    households = pd.DataFrame(
        {"household_id": _normalize_id_value(arrays["household_id"])}
    )
    household_weight = arrays.get("household_weight")
    households["household_weight"] = (
        _normalize_weight_value(household_weight)
        if household_weight is not None
        else np.ones(len(households), dtype=float)
    )

    persons = pd.DataFrame(
        {
            "person_id": _normalize_id_value(arrays["person_id"]),
            "household_id": _normalize_id_value(arrays["person_household_id"]),
        }
    )
    if "person_weight" in arrays:
        persons["weight"] = _normalize_weight_value(arrays["person_weight"])

    group_specs = (
        ("tax_unit", "tax_unit_id", "person_tax_unit_id"),
        ("spm_unit", "spm_unit_id", "person_spm_unit_id"),
        ("family", "family_id", "person_family_id"),
        ("marital_unit", "marital_unit_id", "person_marital_unit_id"),
    )
    group_tables: dict[str, pd.DataFrame | None] = {}
    entity_lengths = {
        EntityType.HOUSEHOLD: len(households),
        EntityType.PERSON: len(persons),
    }
    excluded_variable_names = {
        "household_id",
        "household_weight",
        "person_id",
        "person_household_id",
        "person_weight",
    }
    for group_name, id_column, membership_column in group_specs:
        group_ids = arrays.get(id_column)
        membership = arrays.get(membership_column)
        if membership is not None:
            persons[id_column] = _normalize_id_value(membership)
        if group_ids is None:
            group_tables[group_name] = None
            continue
        group_table = pd.DataFrame({id_column: _normalize_id_value(group_ids)})
        if membership is not None:
            group_table["household_id"] = group_table[id_column].map(
                _build_group_household_map(
                    group_name=group_name,
                    group_ids=pd.Series(_normalize_id_value(membership)),
                    household_ids=persons["household_id"],
                )
            )
        group_tables[group_name] = group_table
        entity_type = _policyengine_group_entity_type(group_name)
        if entity_type is not None:
            entity_lengths[entity_type] = len(group_table)
        excluded_variable_names.add(id_column)
        excluded_variable_names.add(membership_column)

    group_entity_to_table = {
        EntityType.TAX_UNIT: group_tables["tax_unit"],
        EntityType.SPM_UNIT: group_tables["spm_unit"],
        EntityType.FAMILY: group_tables["family"],
    }
    for variable_name, values in arrays.items():
        if variable_name in excluded_variable_names:
            continue
        decoded = _decode_policyengine_array(values)
        prefixed_table = _resolve_prefixed_policyengine_table(
            variable_name=variable_name,
            households=households,
            persons=persons,
            group_tables=group_tables,
        )
        if prefixed_table is not None:
            prefixed_table[variable_name] = decoded
            continue
        try:
            entity = _infer_policyengine_array_entity(
                variable_name=variable_name,
                values=values,
                entity_lengths=entity_lengths,
                tax_benefit_system=tax_benefit_system,
            )
        except ValueError:
            if requested_variables is None:
                continue
            raise
        if entity is EntityType.HOUSEHOLD:
            households[variable_name] = decoded
            continue
        if entity is EntityType.PERSON:
            persons[variable_name] = decoded
            continue
        group_table = group_entity_to_table.get(entity)
        if group_table is None:
            raise ValueError(
                f"Loaded variable '{variable_name}' for entity '{entity.value}' "
                "but no structural table exists for that entity"
            )
        group_table[variable_name] = decoded

    return PolicyEngineUSEntityTableBundle(
        households=households,
        persons=persons,
        tax_units=group_tables["tax_unit"],
        spm_units=group_tables["spm_unit"],
        families=group_tables["family"],
        marital_units=group_tables["marital_unit"],
    )


def infer_policyengine_us_variable_bindings(
    tables: PolicyEngineUSEntityTableBundle,
) -> dict[str, PolicyEngineUSVariableBinding]:
    """Infer variable bindings from currently materialized PE-style tables."""
    bindings: dict[str, PolicyEngineUSVariableBinding] = {}
    table_specs = (
        (
            tables.households,
            EntityType.HOUSEHOLD,
            "household_id",
            {"household_id", "household_weight", "weight"},
        ),
        (
            tables.persons,
            EntityType.PERSON,
            "household_id",
            {"person_id", "household_id", "weight"},
        ),
        (
            tables.tax_units,
            EntityType.TAX_UNIT,
            "household_id",
            {"tax_unit_id", "household_id"},
        ),
        (
            tables.spm_units,
            EntityType.SPM_UNIT,
            "household_id",
            {"spm_unit_id", "household_id"},
        ),
        (
            tables.families,
            EntityType.FAMILY,
            "household_id",
            {"family_id", "household_id"},
        ),
    )
    for table, entity, household_id_column, excluded_columns in table_specs:
        if table is None:
            continue
        for column in table.columns:
            if column in excluded_columns or column.endswith("_id"):
                continue
            bindings.setdefault(
                column,
                PolicyEngineUSVariableBinding(
                    entity=entity,
                    column=column,
                    household_id_column=household_id_column,
                ),
            )
    return bindings


def filter_supported_policyengine_us_targets(
    targets: list[TargetSpec],
    tables: PolicyEngineUSEntityTableBundle,
    bindings: dict[str, PolicyEngineUSVariableBinding],
) -> list[TargetSpec]:
    """Return the targets that can be evaluated with current tables/bindings."""
    supported: list[TargetSpec] = []
    for target in targets:
        if not _has_policyengine_entity_table(target.entity, tables):
            continue
        if any(feature not in bindings for feature in target.required_features):
            continue
        supported.append(target)
    return supported


def is_unsupported_policyengine_us_target_error(error: ValueError) -> bool:
    """Return whether a target-compilation failure indicates unsupported structure."""
    message = str(error)
    return (
        "Cross-entity constraints are only supported against household targets "
        "or household metadata" in message
    )


def compile_supported_policyengine_us_household_linear_constraints(
    targets: list[TargetSpec],
    tables: PolicyEngineUSEntityTableBundle,
    *,
    variable_bindings: dict[str, PolicyEngineUSVariableBinding],
    household_id_column: str = "household_id",
) -> tuple[list[TargetSpec], list[TargetSpec], tuple[LinearConstraint, ...]]:
    """Compile the subset of targets that the current household compiler can handle."""
    filtered_targets = filter_supported_policyengine_us_targets(
        targets,
        tables,
        variable_bindings,
    )
    if not filtered_targets:
        return [], [], ()

    try:
        batched_constraints = compile_policyengine_us_household_linear_constraints(
            filtered_targets,
            tables,
            variable_bindings=variable_bindings,
            household_id_column=household_id_column,
        )
    except ValueError as error:
        if not is_unsupported_policyengine_us_target_error(error):
            raise
    else:
        return filtered_targets, [], batched_constraints

    supported_targets: list[TargetSpec] = []
    unsupported_targets: list[TargetSpec] = []
    constraints: list[LinearConstraint] = []
    for target in filtered_targets:
        try:
            constraint = compile_policyengine_us_household_linear_constraints(
                [target],
                tables,
                variable_bindings=variable_bindings,
                household_id_column=household_id_column,
            )[0]
        except ValueError as error:
            if is_unsupported_policyengine_us_target_error(error):
                unsupported_targets.append(target)
                continue
            raise
        supported_targets.append(target)
        constraints.append(constraint)
    return supported_targets, unsupported_targets, tuple(constraints)


def _policyengine_us_target_required_variables(targets: list[TargetSpec]) -> set[str]:
    return {feature for target in targets for feature in target.required_features}


def policyengine_us_formula_variables_for_targets(
    targets: list[TargetSpec],
    *,
    simulation_cls: Any | None = None,
    tax_benefit_system: Any | None = None,
    direct_override_variables: tuple[str, ...] = (),
) -> set[str]:
    """Return target features that should be recalculated by PolicyEngine."""
    required_variables = _policyengine_us_target_required_variables(targets)
    if not required_variables:
        return set()
    if tax_benefit_system is None:
        tax_benefit_system = _resolve_policyengine_us_tax_benefit_system(simulation_cls)
    variables = getattr(tax_benefit_system, "variables", {})
    direct_overrides = set(direct_override_variables)
    formula_variables: set[str] = set()
    for variable in required_variables:
        if variable in direct_overrides:
            continue
        variable_metadata = variables.get(variable)
        if variable_metadata is None:
            continue
        if _policyengine_us_variable_is_calculated(variable_metadata):
            formula_variables.add(variable)
    return formula_variables


def _policyengine_us_variable_is_calculated(variable_metadata: Any) -> bool:
    if getattr(variable_metadata, "formulas", {}):
        return True
    if getattr(variable_metadata, "adds", ()) or getattr(
        variable_metadata, "subtracts", ()
    ):
        return True
    is_input_variable = getattr(variable_metadata, "is_input_variable", None)
    if callable(is_input_variable):
        try:
            return not bool(is_input_variable())
        except TypeError:
            return False
    return False


def policyengine_us_variables_to_materialize(
    targets: list[TargetSpec],
    bindings: dict[str, PolicyEngineUSVariableBinding],
    *,
    force_materialize_variables: set[str] | tuple[str, ...] | None = None,
) -> set[str]:
    """Compute the missing features required to score the given targets."""
    requested_variables = _policyengine_us_target_required_variables(targets)
    force_variables = set(force_materialize_variables or ())
    return {
        variable
        for variable in requested_variables
        if variable not in bindings or variable in force_variables
    }


def _load_policyengine_us_period_arrays(
    dataset: str | Path | Any,
    *,
    period_key: str,
    variables: set[str] | None,
) -> dict[str, np.ndarray]:
    source = _resolve_policyengine_us_dataset_source(dataset)
    structural_variables = {
        "household_id",
        "household_weight",
        "person_id",
        "person_household_id",
        "person_weight",
        "tax_unit_id",
        "person_tax_unit_id",
        "spm_unit_id",
        "person_spm_unit_id",
        "family_id",
        "person_family_id",
        "marital_unit_id",
        "person_marital_unit_id",
    }
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f"PolicyEngine dataset not found: {source}")
        with h5py.File(source, "r") as handle:
            requested = (
                set(handle.keys())
                if variables is None
                else structural_variables | variables
            )
            return {
                variable: np.asarray(handle[variable][period_key])
                for variable in requested
                if variable in handle and period_key in handle[variable]
            }

    loaded = source.load_dataset()
    requested = set(loaded) if variables is None else structural_variables | variables
    arrays: dict[str, np.ndarray] = {}
    for variable in requested:
        variable_periods = loaded.get(variable)
        if variable_periods is None:
            continue
        value = variable_periods.get(period_key)
        if value is None:
            value = variable_periods.get(int(period_key))
        if value is None:
            continue
        arrays[variable] = np.asarray(value)
    return arrays


def _resolve_policyengine_us_dataset_source(dataset: str | Path | Any) -> Path | Any:
    if isinstance(dataset, (str, Path)):
        return Path(dataset)

    file_path = getattr(dataset, "file_path", None)
    if file_path is not None:
        return Path(file_path)

    if hasattr(dataset, "load_dataset"):
        return dataset

    raise TypeError(
        "dataset must be a path, a dataset-like object with file_path, "
        "or an object exposing load_dataset()"
    )


def _infer_policyengine_array_entity(
    *,
    variable_name: str,
    values: np.ndarray,
    entity_lengths: dict[EntityType, int],
    tax_benefit_system: Any | None,
) -> EntityType:
    legacy_entity_key = POLICYENGINE_US_LEGACY_CONTRACT_VARIABLE_ENTITIES.get(
        variable_name
    )
    if legacy_entity_key in POLICYENGINE_US_ENTITY_KEY_TO_ENTITY_TYPE:
        return POLICYENGINE_US_ENTITY_KEY_TO_ENTITY_TYPE[legacy_entity_key]
    if tax_benefit_system is not None:
        try:
            return _resolve_policyengine_variable_entity(
                variable_name,
                tax_benefit_system=tax_benefit_system,
            )
        except (KeyError, ValueError):
            pass
    matching_entities = [
        entity for entity, length in entity_lengths.items() if len(values) == length
    ]
    if len(matching_entities) == 1:
        return matching_entities[0]
    if not matching_entities:
        raise ValueError(
            f"Cannot infer PolicyEngine entity for variable '{variable_name}' "
            f"with length {len(values)}"
        )
    raise ValueError(
        f"Ambiguous PolicyEngine entity for variable '{variable_name}' "
        f"with length {len(values)}: {[entity.value for entity in matching_entities]}"
    )


def _decode_policyengine_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind != "S":
        return values
    return values.astype(str)


def _resolve_prefixed_policyengine_table(
    *,
    variable_name: str,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    group_tables: dict[str, pd.DataFrame | None],
) -> pd.DataFrame | None:
    if variable_name.startswith("household_"):
        return households
    if variable_name.startswith("person_"):
        return persons
    for prefix, group_name in (
        ("tax_unit_", "tax_unit"),
        ("spm_unit_", "spm_unit"),
        ("family_", "family"),
        ("marital_unit_", "marital_unit"),
    ):
        if variable_name.startswith(prefix):
            return group_tables.get(group_name)
    return None


def _policyengine_group_entity_type(group_name: str) -> EntityType | None:
    mapping = {
        "tax_unit": EntityType.TAX_UNIT,
        "spm_unit": EntityType.SPM_UNIT,
        "family": EntityType.FAMILY,
    }
    return mapping.get(group_name)


def _has_policyengine_entity_table(
    entity: EntityType,
    tables: PolicyEngineUSEntityTableBundle,
) -> bool:
    entity_tables = {
        EntityType.HOUSEHOLD: tables.households,
        EntityType.PERSON: tables.persons,
        EntityType.TAX_UNIT: tables.tax_units,
        EntityType.SPM_UNIT: tables.spm_units,
        EntityType.FAMILY: tables.families,
    }
    return entity_tables.get(entity) is not None


def _compile_household_coefficients(
    *,
    target: PolicyEngineUSDBTarget | TargetSpec,
    tables: PolicyEngineUSEntityTableBundle,
    bindings: dict[str, PolicyEngineUSVariableBinding],
    household_ids: pd.Index,
    household_id_column: str,
) -> np.ndarray:
    target_binding = _resolve_target_binding(target, bindings, tables)
    target_table = tables.table_for(target_binding.entity)
    target_household_ids = _household_ids_for_entity_table(
        target_table,
        target_binding,
        household_id_column,
    )

    if _target_aggregation(target) is TargetAggregation.COUNT:
        mask = pd.Series(True, index=target_table.index, dtype=bool)
        for constraint in _target_constraints(target):
            mask &= _evaluate_constraint_mask(
                target_rows=target_table,
                target_binding=target_binding,
                target_household_ids=target_household_ids,
                constraint=constraint,
                tables=tables,
                bindings=bindings,
                household_id_column=household_id_column,
            )
        values = np.ones(len(target_table), dtype=float)
        contributions = pd.Series(
            np.where(mask.to_numpy(), values, 0.0),
            index=target_household_ids.to_numpy(),
            dtype=float,
        )
        grouped = contributions.groupby(level=0).sum()
        return grouped.reindex(household_ids, fill_value=0.0).to_numpy(dtype=float)

    target_measure = _target_measure(target)
    if target_binding.column is None or target_measure is None:
        raise ValueError(
            f"Target '{_policyengine_target_name(target)}' has no source column"
        )

    target_values = pd.to_numeric(
        target_table[target_binding.column], errors="coerce"
    ).fillna(0.0)
    row_mask = pd.Series(True, index=target_table.index, dtype=bool)
    household_constraints: list[PolicyEngineUSConstraint | TargetFilter] = []
    for constraint in _target_constraints(target):
        constraint_binding = _resolve_binding(
            _constraint_feature(constraint),
            bindings,
            tables,
        )
        if constraint_binding.entity in {
            target_binding.entity,
            EntityType.HOUSEHOLD,
        } or _can_align_constraint_to_target_rows(
            target_rows=target_table,
            constraint_binding=constraint_binding,
        ):
            row_mask &= _evaluate_constraint_mask(
                target_rows=target_table,
                target_binding=target_binding,
                target_household_ids=target_household_ids,
                constraint=constraint,
                tables=tables,
                bindings=bindings,
                household_id_column=household_id_column,
            )
        else:
            household_constraints.append(constraint)

    household_totals = (
        target_values.where(row_mask, 0.0).groupby(target_household_ids).sum()
    )
    household_mask = pd.Series(True, index=household_ids, dtype=bool)
    for constraint in household_constraints:
        household_mask &= _evaluate_constraint_on_households(
            constraint=constraint,
            tables=tables,
            bindings=bindings,
            household_ids=household_ids,
            household_id_column=household_id_column,
        )

    return (
        household_totals.reindex(household_ids, fill_value=0.0).astype(float)
        * household_mask.astype(float)
    ).to_numpy(dtype=float)


def _entity_primary_id_column(entity: EntityType) -> str:
    return {
        EntityType.HOUSEHOLD: "household_id",
        EntityType.PERSON: "person_id",
        EntityType.TAX_UNIT: "tax_unit_id",
        EntityType.SPM_UNIT: "spm_unit_id",
        EntityType.FAMILY: "family_id",
    }[entity]


def _can_align_constraint_to_target_rows(
    *,
    target_rows: pd.DataFrame,
    constraint_binding: PolicyEngineUSVariableBinding,
) -> bool:
    return _entity_primary_id_column(constraint_binding.entity) in target_rows.columns


def _resolve_binding(
    variable: str,
    bindings: dict[str, PolicyEngineUSVariableBinding],
    tables: PolicyEngineUSEntityTableBundle,
) -> PolicyEngineUSVariableBinding:
    if variable in bindings:
        return bindings[variable]
    if variable in tables.households.columns:
        return PolicyEngineUSVariableBinding(
            entity=EntityType.HOUSEHOLD, column=variable
        )
    raise KeyError(f"No PolicyEngine binding configured for variable '{variable}'")


def _resolve_target_binding(
    target: PolicyEngineUSDBTarget | TargetSpec,
    bindings: dict[str, PolicyEngineUSVariableBinding],
    tables: PolicyEngineUSEntityTableBundle,
) -> PolicyEngineUSVariableBinding:
    if isinstance(target, TargetSpec):
        if target.aggregation is TargetAggregation.COUNT:
            return PolicyEngineUSVariableBinding(entity=target.entity)
        if target.measure is None:
            raise ValueError(f"Target '{target.name}' is missing a measure")
        binding = _resolve_binding(target.measure, bindings, tables)
        return PolicyEngineUSVariableBinding(
            entity=binding.entity,
            column=binding.column,
            household_id_column=binding.household_id_column,
        )

    return _resolve_binding(target.variable, bindings, tables)


def _require_binding_column(
    binding: PolicyEngineUSVariableBinding,
    *,
    feature: str,
) -> str:
    if binding.column is None:
        raise ValueError(
            f"Constraint variable '{feature}' does not map to a source column"
        )
    return binding.column


def _household_ids_for_entity_table(
    table: pd.DataFrame,
    binding: PolicyEngineUSVariableBinding,
    household_id_column: str,
) -> pd.Series:
    if binding.entity is EntityType.HOUSEHOLD:
        if household_id_column not in table.columns:
            raise ValueError(
                f"Household table is missing household id column '{household_id_column}'"
            )
        return table[household_id_column]

    if binding.household_id_column not in table.columns:
        raise ValueError(
            f"Entity table for '{binding.entity.value}' is missing "
            f"household link column '{binding.household_id_column}'"
        )
    return table[binding.household_id_column]


def _apply_constraint_filter(
    values: pd.Series,
    constraint: PolicyEngineUSConstraint | TargetFilter,
) -> pd.Series:
    return _apply_constraint(
        values,
        _constraint_operator(constraint),
        _constraint_value(constraint),
    )


def _evaluate_constraint_mask(
    *,
    target_rows: pd.DataFrame,
    target_binding: PolicyEngineUSVariableBinding,
    target_household_ids: pd.Series,
    constraint: PolicyEngineUSConstraint | TargetFilter,
    tables: PolicyEngineUSEntityTableBundle,
    bindings: dict[str, PolicyEngineUSVariableBinding],
    household_id_column: str,
) -> pd.Series:
    constraint_binding = _resolve_binding(
        _constraint_feature(constraint), bindings, tables
    )
    constraint_column = _require_binding_column(
        constraint_binding,
        feature=_constraint_feature(constraint),
    )

    if constraint_binding.entity is target_binding.entity:
        return _apply_constraint_filter(target_rows[constraint_column], constraint)

    if constraint_binding.entity is EntityType.HOUSEHOLD:
        household_values = tables.households.set_index(household_id_column)[
            constraint_column
        ]
        aligned = target_household_ids.map(household_values)
        return _apply_constraint_filter(aligned, constraint)

    aligned = _align_related_entity_constraint_values(
        target_rows=target_rows,
        constraint_binding=constraint_binding,
        feature=_constraint_feature(constraint),
        tables=tables,
    )
    if aligned is not None:
        return _apply_constraint_filter(aligned, constraint)

    aligned_mask = _align_related_entity_constraint_mask(
        target_rows=target_rows,
        target_binding=target_binding,
        constraint_binding=constraint_binding,
        constraint=constraint,
        tables=tables,
    )
    if aligned_mask is not None:
        return aligned_mask

    if target_binding.entity is EntityType.HOUSEHOLD:
        aligned = _evaluate_constraint_on_households(
            constraint=constraint,
            tables=tables,
            bindings=bindings,
            household_ids=pd.Index(target_household_ids),
            household_id_column=household_id_column,
        ).reindex(pd.Index(target_household_ids), fill_value=False)
        return pd.Series(aligned.to_numpy(), index=target_rows.index, dtype=bool)

    raise ValueError(
        "Cross-entity constraints are only supported against household targets "
        "or household metadata"
    )


def _align_related_entity_constraint_values(
    *,
    target_rows: pd.DataFrame,
    constraint_binding: PolicyEngineUSVariableBinding,
    feature: str,
    tables: PolicyEngineUSEntityTableBundle,
) -> pd.Series | None:
    related_id_column = _entity_primary_id_column(constraint_binding.entity)
    if related_id_column not in target_rows.columns:
        return None

    related_table = tables.table_for(constraint_binding.entity)
    if related_id_column not in related_table.columns:
        raise ValueError(
            f"Entity table for '{constraint_binding.entity.value}' is missing "
            f"primary id column '{related_id_column}'"
        )

    constraint_column = _require_binding_column(
        constraint_binding,
        feature=feature,
    )
    related_values = related_table.set_index(related_id_column)[constraint_column]
    return target_rows[related_id_column].map(related_values)


def _align_related_entity_constraint_mask(
    *,
    target_rows: pd.DataFrame,
    target_binding: PolicyEngineUSVariableBinding,
    constraint_binding: PolicyEngineUSVariableBinding,
    constraint: PolicyEngineUSConstraint | TargetFilter,
    tables: PolicyEngineUSEntityTableBundle,
) -> pd.Series | None:
    if constraint_binding.entity is not EntityType.PERSON:
        return None
    if target_binding.entity not in {
        EntityType.TAX_UNIT,
        EntityType.SPM_UNIT,
        EntityType.FAMILY,
    }:
        return None
    persons = tables.persons
    if persons is None:
        return None
    related_id_column = _entity_primary_id_column(target_binding.entity)
    if (
        related_id_column not in target_rows.columns
        or related_id_column not in persons.columns
    ):
        return None
    constraint_column = _require_binding_column(
        constraint_binding,
        feature=_constraint_feature(constraint),
    )
    person_matches = _apply_constraint_filter(persons[constraint_column], constraint)
    group_matches = person_matches.groupby(persons[related_id_column]).any()
    aligned = target_rows[related_id_column].map(group_matches).fillna(False)
    return pd.Series(aligned.to_numpy(dtype=bool), index=target_rows.index, dtype=bool)


def _evaluate_constraint_on_households(
    *,
    constraint: PolicyEngineUSConstraint | TargetFilter,
    tables: PolicyEngineUSEntityTableBundle,
    bindings: dict[str, PolicyEngineUSVariableBinding],
    household_ids: pd.Index,
    household_id_column: str,
) -> pd.Series:
    binding = _resolve_binding(_constraint_feature(constraint), bindings, tables)
    binding_column = _require_binding_column(
        binding,
        feature=_constraint_feature(constraint),
    )
    if binding.entity is EntityType.HOUSEHOLD:
        values = tables.households.set_index(household_id_column)[binding_column]
        return _apply_constraint_filter(values, constraint).reindex(
            household_ids,
            fill_value=False,
        )

    table = tables.table_for(binding.entity)
    related_household_ids = _household_ids_for_entity_table(
        table, binding, household_id_column
    )
    row_matches = _apply_constraint_filter(table[binding_column], constraint)
    return (
        row_matches.groupby(related_household_ids)
        .any()
        .reindex(
            household_ids,
            fill_value=False,
        )
    )


def _apply_constraint(series: pd.Series, operation: str, raw_value: Any) -> pd.Series:
    operation = "==" if operation == "=" else operation
    if operation not in {"==", "!=", ">", ">=", "<", "<=", "in", "not_in"}:
        raise ValueError(f"Unsupported PolicyEngine constraint operation: {operation}")

    if operation in {"in", "not_in"}:
        if not isinstance(raw_value, (list, tuple, set, frozenset)):
            raw_values = [raw_value]
        else:
            raw_values = list(raw_value)
        value = [_coerce_constraint_value(series, item) for item in raw_values]
    else:
        value = _coerce_constraint_value(series, raw_value)
    if operation == "==":
        return series == value
    if operation == "!=":
        return series != value
    if operation == "in":
        return series.isin(value)
    if operation == "not_in":
        return ~series.isin(value)
    if operation == ">":
        return series > value
    if operation == ">=":
        return series >= value
    if operation == "<":
        return series < value
    return series <= value


def _coerce_constraint_value(series: pd.Series, raw_value: Any) -> Any:
    if pd.api.types.is_bool_dtype(series):
        return str(raw_value).strip().lower() in {"1", "true", "t", "yes"}
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(pd.Series([raw_value]), errors="raise").iloc[0]
    return str(raw_value)


def _policyengine_target_name(target: PolicyEngineUSDBTarget | TargetSpec) -> str:
    if isinstance(target, TargetSpec):
        return target.name
    return f"target_{target.target_id}_{target.variable}"


def _target_value(target: PolicyEngineUSDBTarget | TargetSpec) -> float:
    return float(target.value)


def _target_aggregation(
    target: PolicyEngineUSDBTarget | TargetSpec,
) -> TargetAggregation:
    if isinstance(target, TargetSpec):
        return target.aggregation
    if target.variable in DEFAULT_POLICYENGINE_US_VARIABLE_BINDINGS:
        return TargetAggregation.COUNT
    return TargetAggregation.SUM


def _target_measure(target: PolicyEngineUSDBTarget | TargetSpec) -> str | None:
    if isinstance(target, TargetSpec):
        return target.measure
    if target.variable in DEFAULT_POLICYENGINE_US_VARIABLE_BINDINGS:
        return None
    return target.variable


def _target_constraints(
    target: PolicyEngineUSDBTarget | TargetSpec,
) -> tuple[PolicyEngineUSConstraint | TargetFilter, ...]:
    if isinstance(target, TargetSpec):
        return target.filters
    return target.constraints


def _constraint_feature(constraint: PolicyEngineUSConstraint | TargetFilter) -> str:
    return (
        constraint.feature
        if isinstance(constraint, TargetFilter)
        else constraint.variable
    )


def _constraint_operator(constraint: PolicyEngineUSConstraint | TargetFilter) -> str:
    if isinstance(constraint, TargetFilter):
        return str(constraint.operator.value)
    return str(constraint.operation)


def _constraint_value(constraint: PolicyEngineUSConstraint | TargetFilter) -> Any:
    return constraint.value


def build_policyengine_us_export_variable_maps(
    tables: PolicyEngineUSEntityTableBundle,
    *,
    tax_benefit_system: Any,
    direct_override_variables: tuple[str, ...] = (),
) -> dict[str, dict[str, str]]:
    """Infer PE export variable maps from entity-table columns."""
    variable_metadata = getattr(tax_benefit_system, "variables", {})
    allowed_variables_by_entity = _group_policyengine_us_export_variables_by_entity(
        variable_metadata,
        direct_override_variables=direct_override_variables,
    )
    person_table = _with_policyengine_person_export_derivatives(tables.persons)
    table_specs = (
        (
            "household",
            tables.households,
            {"household_id", "household_weight", "weight"},
        ),
        ("person", person_table, {"person_id", "household_id"}),
        ("tax_unit", tables.tax_units, {"tax_unit_id", "household_id"}),
        ("spm_unit", tables.spm_units, {"spm_unit_id", "household_id"}),
        ("family", tables.families, {"family_id", "household_id"}),
    )
    export_maps: dict[str, dict[str, str]] = {}
    for entity_key, table, structural_columns in table_specs:
        export_maps[entity_key] = _infer_policyengine_us_table_variable_map(
            table=table,
            allowed_variables=allowed_variables_by_entity.get(entity_key, set()),
            excluded_columns=structural_columns,
        )
    return export_maps


def build_policyengine_us_time_period_arrays(
    tables: PolicyEngineUSEntityTableBundle,
    *,
    period: int,
    household_variable_map: dict[str, str] | None = None,
    person_variable_map: dict[str, str] | None = None,
    tax_unit_variable_map: dict[str, str] | None = None,
    spm_unit_variable_map: dict[str, str] | None = None,
    family_variable_map: dict[str, str] | None = None,
    marital_unit_variable_map: dict[str, str] | None = None,
    household_id_column: str = "household_id",
    person_id_column: str = "person_id",
    household_weight_column: str = "household_weight",
) -> dict[str, dict[str, np.ndarray]]:
    """Build a PE-US TIME_PERIOD_ARRAYS payload from multientity tables."""
    if tables.persons is None:
        raise ValueError("PolicyEngine US export requires a person table")

    period_key = str(period)
    households = _prepare_household_export_table(
        tables.households,
        household_id_column=household_id_column,
        household_weight_column=household_weight_column,
    )
    person_table = _with_policyengine_person_export_derivatives(tables.persons)
    persons = _prepare_person_export_table(
        person_table,
        person_id_column=person_id_column,
        household_id_column=household_id_column,
        household_ids=pd.Index(households[household_id_column]),
    )

    arrays: dict[str, dict[str, np.ndarray]] = {
        "household_id": {
            period_key: _normalize_id_value(households[household_id_column]),
        },
        "person_id": {
            period_key: _normalize_id_value(persons[person_id_column]),
        },
        "person_household_id": {
            period_key: _normalize_id_value(persons[household_id_column]),
        },
        "household_weight": {
            period_key: _normalize_weight_value(households[household_weight_column]),
        },
    }

    arrays.update(
        _project_table_to_time_period_arrays(
            households,
            period_key=period_key,
            column_map=household_variable_map,
            excluded_columns={household_id_column, household_weight_column},
        )
    )
    arrays.update(
        _project_table_to_time_period_arrays(
            persons,
            period_key=period_key,
            column_map=person_variable_map,
            excluded_columns={person_id_column, household_id_column},
        )
    )

    group_specs = (
        (
            "tax_unit",
            "tax_unit_id",
            tables.tax_units,
            tax_unit_variable_map,
            "household",
        ),
        (
            "spm_unit",
            "spm_unit_id",
            tables.spm_units,
            spm_unit_variable_map,
            "household",
        ),
        ("family", "family_id", tables.families, family_variable_map, "household"),
        (
            "marital_unit",
            "marital_unit_id",
            tables.marital_units,
            marital_unit_variable_map,
            "tax_unit",
        ),
    )
    for group_name, id_column, provided_table, variable_map, fallback in group_specs:
        person_group_ids = _resolve_person_group_ids(
            group_name=group_name,
            id_column=id_column,
            persons=persons,
            provided_table=provided_table,
            person_id_column=person_id_column,
            household_id_column=household_id_column,
            fallback=fallback,
        )
        group_table = _resolve_group_export_table(
            group_name=group_name,
            id_column=id_column,
            provided_table=provided_table,
            person_group_ids=person_group_ids,
            person_household_ids=persons[household_id_column],
            household_id_column=household_id_column,
        )
        arrays[f"{group_name}_id"] = {
            period_key: _normalize_id_value(group_table[id_column]),
        }
        arrays[f"person_{group_name}_id"] = {
            period_key: _normalize_id_value(person_group_ids),
        }
        arrays.update(
            _project_table_to_time_period_arrays(
                group_table,
                period_key=period_key,
                column_map=variable_map,
                excluded_columns={id_column, household_id_column},
            )
        )

    return arrays


def _resolve_policyengine_us_tax_benefit_system(simulation_cls: Any | None) -> Any:
    if simulation_cls is None:
        try:
            import policyengine_us
        except ImportError as exc:
            raise ImportError(
                "policyengine_us is required to materialize PolicyEngine US variables"
            ) from exc
        return getattr(policyengine_us.system, "system", policyengine_us.system)

    tax_benefit_system = getattr(simulation_cls, "tax_benefit_system", None)
    if tax_benefit_system is None:
        tax_benefit_system = getattr(simulation_cls, "system", None)
    if tax_benefit_system is not None:
        tax_benefit_system = getattr(tax_benefit_system, "system", tax_benefit_system)
    if tax_benefit_system is None:
        raise ValueError(
            "simulation_cls must expose a 'tax_benefit_system' attribute to materialize variables"
        )
    return tax_benefit_system


def _resolve_policyengine_variable_entity(
    variable: str,
    *,
    tax_benefit_system: Any,
) -> EntityType:
    variables = getattr(tax_benefit_system, "variables", {})
    variable_metadata = variables.get(variable)
    if variable_metadata is None:
        raise KeyError(
            f"PolicyEngine variable '{variable}' not found in tax-benefit system"
        )
    entity_key = getattr(getattr(variable_metadata, "entity", None), "key", None)
    if entity_key not in POLICYENGINE_US_ENTITY_KEY_TO_ENTITY_TYPE:
        raise ValueError(
            f"Unsupported PolicyEngine entity '{entity_key}' for variable '{variable}'"
        )
    return POLICYENGINE_US_ENTITY_KEY_TO_ENTITY_TYPE[entity_key]


def _infer_policyengine_us_table_variable_map(
    *,
    table: pd.DataFrame | None,
    allowed_variables: set[str],
    excluded_columns: set[str],
) -> dict[str, str]:
    if table is None:
        return {}
    variable_map = {
        column: column
        for column in table.columns
        if column in allowed_variables
        and column not in excluded_columns
        and not column.endswith("_id")
    }
    exported_targets = set(variable_map.values())
    available_columns = set(table.columns)
    for source_column, target_variable in POLICYENGINE_US_EXPORT_COLUMN_ALIASES.items():
        if target_variable not in allowed_variables:
            continue
        if source_column not in available_columns or source_column in excluded_columns:
            continue
        if source_column.endswith("_id") or target_variable in exported_targets:
            continue
        variable_map[source_column] = target_variable
        exported_targets.add(target_variable)
    for target_variable in sorted(
        set(POLICYENGINE_US_EXPORT_DEFAULTS) & allowed_variables
    ):
        if target_variable in exported_targets:
            continue
        variable_map[target_variable] = target_variable
        exported_targets.add(target_variable)
    return variable_map


def _group_policyengine_us_export_variables_by_entity(
    variable_metadata: dict[str, Any],
    *,
    direct_override_variables: tuple[str, ...] = (),
) -> dict[str, set[str]]:
    allowed_variable_names = (
        SAFE_POLICYENGINE_US_EXPORT_VARIABLES
        | set(POLICYENGINE_US_EXPORT_DEFAULTS)
        | set(direct_override_variables)
    )
    allowed_variables_by_entity: dict[str, set[str]] = {
        entity_key: set() for entity_key in POLICYENGINE_US_ENTITY_KEY_TO_ENTITY_TYPE
    }
    for variable_name, metadata in variable_metadata.items():
        if variable_name not in allowed_variable_names:
            continue
        if (
            variable_name not in POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES
            and _policyengine_us_variable_is_computed_export(metadata)
        ):
            continue
        entity_key = getattr(getattr(metadata, "entity", None), "key", None)
        if entity_key not in allowed_variables_by_entity:
            continue
        allowed_variables_by_entity[entity_key].add(variable_name)
    for (
        variable_name,
        entity_key,
    ) in POLICYENGINE_US_LEGACY_CONTRACT_VARIABLE_ENTITIES.items():
        if entity_key in allowed_variables_by_entity:
            allowed_variables_by_entity[entity_key].add(variable_name)
    return allowed_variables_by_entity


def _attach_policyengine_variables_to_tables(
    tables: PolicyEngineUSEntityTableBundle,
    *,
    variables: tuple[str, ...],
    period: int,
    adapter: PolicyEngineUSMicrosimulationAdapter,
) -> tuple[PolicyEngineUSEntityTableBundle, dict[str, PolicyEngineUSVariableBinding]]:
    households = tables.households.copy()
    persons = tables.persons.copy() if tables.persons is not None else None
    tax_units = tables.tax_units.copy() if tables.tax_units is not None else None
    spm_units = tables.spm_units.copy() if tables.spm_units is not None else None
    families = tables.families.copy() if tables.families is not None else None
    marital_units = (
        tables.marital_units.copy() if tables.marital_units is not None else None
    )
    bindings: dict[str, PolicyEngineUSVariableBinding] = {}

    for variable in variables:
        entity = adapter.variable_entity(variable)
        entity_key = ENTITY_TYPE_TO_POLICYENGINE_US_ENTITY_KEY[entity]
        table = _table_for_policyengine_entity(
            entity=entity,
            households=households,
            persons=persons,
            tax_units=tax_units,
            spm_units=spm_units,
            families=families,
        )
        values = _coerce_policyengine_calculation_values(
            adapter.calculate(variable, period=period)
        )
        if len(values) != len(table):
            values = _coerce_policyengine_calculation_values(
                adapter.calculate(variable, period=period, map_to=entity_key)
            )
        if len(values) != len(table):
            raise ValueError(
                f"PolicyEngine variable '{variable}' returned {len(values)} values for "
                f"{entity.value}, expected {len(table)}"
            )
        table[variable] = values
        bindings[variable] = PolicyEngineUSVariableBinding(
            entity=entity, column=variable
        )

    return (
        PolicyEngineUSEntityTableBundle(
            households=households,
            persons=persons,
            tax_units=tax_units,
            spm_units=spm_units,
            families=families,
            marital_units=marital_units,
        ),
        bindings,
    )


def _table_for_policyengine_entity(
    *,
    entity: EntityType,
    households: pd.DataFrame,
    persons: pd.DataFrame | None,
    tax_units: pd.DataFrame | None,
    spm_units: pd.DataFrame | None,
    families: pd.DataFrame | None,
) -> pd.DataFrame:
    if entity is EntityType.HOUSEHOLD:
        return households
    if entity is EntityType.PERSON and persons is not None:
        return persons
    if entity is EntityType.TAX_UNIT and tax_units is not None:
        return tax_units
    if entity is EntityType.SPM_UNIT and spm_units is not None:
        return spm_units
    if entity is EntityType.FAMILY and families is not None:
        return families
    raise ValueError(f"No table available to materialize '{entity.value}' variables")


def _coerce_policyengine_calculation_values(values: Any) -> np.ndarray:
    if hasattr(values, "values"):
        return np.asarray(values.values)
    return np.asarray(values)


def project_frame_to_time_period_arrays(
    frame: pd.DataFrame,
    *,
    period: int,
    column_map: dict[str, str],
) -> dict[str, dict[str, np.ndarray]]:
    """Project a richer Microplex frame into PE-style time-period arrays."""
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for source_column, target_variable in column_map.items():
        if source_column not in frame.columns:
            raise ValueError(f"Projection source column not found: {source_column}")
        arrays[target_variable] = {
            str(period): _normalize_h5_value(frame[source_column])
        }
    return arrays


def write_policyengine_us_time_period_dataset(
    data: dict[str, dict[str, np.ndarray]],
    path: str | Path,
    *,
    excluded_variables: set[str] | None = None,
    tax_benefit_system: Any | None = None,
) -> Path:
    """Write PolicyEngine-readable time-period arrays to HDF5."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded = excluded_variables or set()
    written_variables = [variable for variable in data if variable not in excluded]
    if tax_benefit_system is not None:
        computed_exports = detect_policyengine_computed_export_variables(
            tax_benefit_system,
            written_variables,
        )
        if computed_exports:
            formatted = ", ".join(sorted(computed_exports)[:10])
            suffix = "" if len(computed_exports) <= 10 else ", ..."
            raise ValueError(
                "PolicyEngine US export contains computed variables: "
                f"{formatted}{suffix}. Drop construction-only intermediates "
                "or export the underlying leaf input instead."
            )

    with h5py.File(output_path, "w") as handle:
        for variable, periods in data.items():
            if variable in excluded:
                continue
            group = handle.create_group(variable)
            for period, values in periods.items():
                group.create_dataset(str(period), data=_normalize_h5_value(values))

    return output_path


def _normalize_h5_value(values: Any) -> np.ndarray:
    """Normalize values so h5py can persist them predictably."""
    array = np.asarray(values)
    if array.dtype.kind in {"U", "O"}:
        return array.astype("S")
    return array


def _prepare_household_export_table(
    households: pd.DataFrame,
    *,
    household_id_column: str,
    household_weight_column: str,
) -> pd.DataFrame:
    household_table = households.copy()
    if household_id_column not in household_table.columns:
        raise ValueError(
            f"Household table must contain '{household_id_column}' for export"
        )
    if household_weight_column not in household_table.columns:
        if "weight" not in household_table.columns:
            raise ValueError(
                f"Household table must contain '{household_weight_column}' or 'weight'"
            )
        household_table[household_weight_column] = household_table["weight"]
    household_table[household_id_column] = _normalize_id_value(
        household_table[household_id_column]
    )
    if pd.Index(household_table[household_id_column]).duplicated().any():
        raise ValueError("Household export table must have unique household ids")
    household_table[household_weight_column] = _normalize_weight_value(
        household_table[household_weight_column]
    )
    return household_table


def _prepare_person_export_table(
    persons: pd.DataFrame,
    *,
    person_id_column: str,
    household_id_column: str,
    household_ids: pd.Index,
) -> pd.DataFrame:
    person_table = persons.copy()
    missing_columns = {
        column
        for column in (person_id_column, household_id_column)
        if column not in person_table.columns
    }
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Person table is missing required export columns: {missing}")
    person_table[person_id_column] = _normalize_id_value(person_table[person_id_column])
    person_table[household_id_column] = _normalize_id_value(
        person_table[household_id_column]
    )
    if pd.Index(person_table[person_id_column]).duplicated().any():
        raise ValueError("Person export table must have unique person ids")
    if not pd.Index(person_table[household_id_column]).isin(household_ids).all():
        raise ValueError("Every exported person must belong to an exported household")
    return person_table


def _with_policyengine_person_export_derivatives(
    persons: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if persons is None or "is_household_head" in persons.columns:
        return persons
    if "relationship_to_head" not in persons.columns:
        return persons

    person_table = persons.copy()
    relationship = pd.to_numeric(
        person_table["relationship_to_head"],
        errors="coerce",
    )
    person_table["is_household_head"] = relationship.eq(0).fillna(False)
    return person_table


def _resolve_person_group_ids(
    *,
    group_name: str,
    id_column: str,
    persons: pd.DataFrame,
    provided_table: pd.DataFrame | None,
    person_id_column: str,
    household_id_column: str,
    fallback: Literal["household", "tax_unit"],
) -> pd.Series:
    if id_column in persons.columns:
        return pd.Series(
            _normalize_id_value(persons[id_column]),
            index=persons.index,
            name=id_column,
        )

    if provided_table is not None:
        resolved = _extract_membership_ids_from_group_table(
            group_name=group_name,
            id_column=id_column,
            provided_table=provided_table,
            persons=persons,
            person_id_column=person_id_column,
            household_id_column=household_id_column,
        )
        if resolved is not None:
            return resolved

    if fallback == "tax_unit" and "tax_unit_id" in persons.columns:
        return pd.Series(
            _normalize_id_value(persons["tax_unit_id"]),
            index=persons.index,
            name=id_column,
        )

    return pd.Series(
        _normalize_id_value(persons[household_id_column]),
        index=persons.index,
        name=id_column,
    )


def _extract_membership_ids_from_group_table(
    *,
    group_name: str,
    id_column: str,
    provided_table: pd.DataFrame,
    persons: pd.DataFrame,
    person_id_column: str,
    household_id_column: str,
) -> pd.Series | None:
    if id_column not in provided_table.columns:
        return None

    mapping: dict[int, int] = {}
    for member_column in ("member_ids", "filer_ids", "dependent_ids"):
        if member_column not in provided_table.columns:
            continue
        for _, row in provided_table[[id_column, member_column]].iterrows():
            group_id = int(_normalize_id_value([row[id_column]])[0])
            members = row[member_column]
            if not isinstance(members, (list, tuple, np.ndarray, pd.Series)):
                continue
            for member_id in members:
                if pd.isna(member_id):
                    continue
                mapping[int(member_id)] = group_id
    if mapping:
        membership = persons[person_id_column].map(mapping)
        if membership.isna().any():
            missing = persons.loc[membership.isna(), person_id_column].tolist()
            raise ValueError(
                f"Could not derive '{group_name}' membership for persons: {missing}"
            )
        return pd.Series(
            _normalize_id_value(membership),
            index=persons.index,
            name=id_column,
        )

    if (
        household_id_column in provided_table.columns
        and not provided_table[household_id_column].duplicated().any()
    ):
        household_map = (
            provided_table[[id_column, household_id_column]]
            .assign(
                **{
                    id_column: _normalize_id_value(provided_table[id_column]),
                    household_id_column: _normalize_id_value(
                        provided_table[household_id_column]
                    ),
                }
            )
            .set_index(household_id_column)[id_column]
        )
        membership = persons[household_id_column].map(household_map)
        if membership.notna().all():
            return pd.Series(
                _normalize_id_value(membership),
                index=persons.index,
                name=id_column,
            )

    return None


def _resolve_group_export_table(
    *,
    group_name: str,
    id_column: str,
    provided_table: pd.DataFrame | None,
    person_group_ids: pd.Series,
    person_household_ids: pd.Series,
    household_id_column: str,
) -> pd.DataFrame:
    if provided_table is None:
        group_table = pd.DataFrame({id_column: pd.unique(person_group_ids)})
    else:
        group_table = provided_table.copy()
        if id_column not in group_table.columns:
            group_table[id_column] = pd.unique(person_group_ids)

    group_table[id_column] = _normalize_id_value(group_table[id_column])
    if pd.Index(group_table[id_column]).duplicated().any():
        raise ValueError(f"{group_name} export table must have unique ids")

    household_map = _build_group_household_map(
        group_name=group_name,
        group_ids=person_group_ids,
        household_ids=person_household_ids,
    )
    if household_id_column in group_table.columns:
        normalized_households = _normalize_id_value(group_table[household_id_column])
        group_table[household_id_column] = normalized_households
        expected = group_table[id_column].map(household_map)
        mismatch = expected.notna() & pd.Series(
            normalized_households, index=group_table.index
        ).ne(expected)
        if mismatch.any():
            raise ValueError(
                f"{group_name} export table household links are inconsistent with person memberships"
            )
    else:
        group_table[household_id_column] = group_table[id_column].map(household_map)

    if group_table[household_id_column].isna().any():
        missing = group_table.loc[
            group_table[household_id_column].isna(), id_column
        ].tolist()
        raise ValueError(
            f"Could not derive household links for {group_name} ids: {missing}"
        )
    return group_table


def _build_group_household_map(
    *,
    group_name: str,
    group_ids: pd.Series,
    household_ids: pd.Series,
) -> pd.Series:
    mapping = pd.DataFrame(
        {
            "group_id": _normalize_id_value(group_ids),
            "household_id": _normalize_id_value(household_ids),
        }
    ).drop_duplicates()
    if mapping.groupby("group_id")["household_id"].nunique().gt(1).any():
        raise ValueError(
            f"{group_name} members must all belong to the same household for PE export"
        )
    return mapping.set_index("group_id")["household_id"]


def _project_table_to_time_period_arrays(
    table: pd.DataFrame,
    *,
    period_key: str,
    column_map: dict[str, str] | None,
    excluded_columns: set[str],
) -> dict[str, dict[str, np.ndarray]]:
    if not column_map:
        return {}

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for source_column, target_variable in column_map.items():
        if source_column in excluded_columns:
            continue
        has_default = target_variable in POLICYENGINE_US_EXPORT_DEFAULTS
        if source_column not in table.columns and not has_default:
            raise ValueError(f"Projection source column not found: {source_column}")
        values = (
            pd.Series(table[source_column])
            if source_column in table.columns
            else pd.Series(
                POLICYENGINE_US_EXPORT_DEFAULTS[target_variable],
                index=table.index,
            )
        )
        if has_default:
            default_value = POLICYENGINE_US_EXPORT_DEFAULTS[target_variable]
            values = values.where(values.notna(), other=default_value)
            if isinstance(default_value, str):
                string_values = values.astype("string")
                values = string_values.where(
                    string_values.notna() & string_values.ne(""),
                    other=default_value,
                )
        values = _normalize_policyengine_us_export_enum_values(target_variable, values)
        arrays[target_variable] = {
            period_key: _normalize_h5_value(values),
        }
    return arrays


def _normalize_policyengine_us_export_enum_values(
    target_variable: str,
    values: pd.Series,
) -> pd.Series:
    enum_map = POLICYENGINE_US_NUMERIC_ENUM_EXPORT_MAPS.get(target_variable)
    if enum_map is None:
        return values

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_mask = numeric.notna()
    if not numeric_mask.any():
        return values

    normalized = values.copy()
    mapped = numeric.map(enum_map)
    if mapped[numeric_mask].isna().any():
        bad_values = sorted(
            {
                str(value)
                for value in values[numeric_mask & mapped.isna()].dropna().unique()
            }
        )
        raise ValueError(
            f"Unsupported numeric {target_variable} value(s) for PE-US export: "
            + ", ".join(bad_values)
        )
    normalized.loc[numeric_mask] = mapped[numeric_mask]
    return normalized


def _normalize_id_value(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="raise").astype(np.int64).to_numpy()


def _normalize_weight_value(values: Any) -> np.ndarray:
    return (
        pd.to_numeric(pd.Series(values), errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )
