"""Helpers for working with atomic vs derived variables and donor specs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from microplex.core import EntityType
from microplex.core.semantics import (
    FrameSemanticCheck,
    FrameSemanticCheckReport,
    FrameSemanticTransform,
    SemanticTransformStage,
    apply_frame_semantic_transforms,
    evaluate_frame_semantic_checks,
)

try:
    from microplex.core import SourceVariableCapability
except ImportError:

    @dataclass(frozen=True)
    class SourceVariableCapability:
        """Compatibility shim for older core branches."""

        authoritative: bool = True
        usable_as_condition: bool = True
        notes: str | None = None


class DonorMatchStrategy(Enum):
    """How donor-generated scores should be mapped back onto donor support."""

    RANK = "rank"


class VariableSupportFamily(Enum):
    """Statistical support family for one variable."""

    CONTINUOUS = "continuous"
    SUPPORT_SENSITIVE = "support_sensitive"
    BOUNDED_SHARE = "bounded_share"


class ConditionScoreMode(Enum):
    """How donor condition variables should be scored for one target family."""

    VALUE_ONLY = "value_only"
    VALUE_AND_SUPPORT = "value_and_support"


class ProjectionAggregation(Enum):
    """How person-native features should be projected onto a group entity."""

    FIRST = "first"
    SUM = "sum"
    MAX = "max"
    MEAN = "mean"


PE_STYLE_PUF_IRS_DEMOGRAPHIC_PREDICTORS = (
    "age",
    "is_male",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
)
PUF_IRS_TAX_PREFERRED_CONDITION_VARS = (
    "age",
    "is_male",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
)
# Keep PE-aligned PUF tax-leaf conditioning structural by default.
PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS: tuple[str, ...] = ()
# Explicit challenger widening for live PUF tax-leaf blocks. These vars are only
# meant for opt-in experiments that retain the PE structural backbone while
# layering in a narrow source-native raw-overlap surface.
PUF_DIVIDEND_INTEREST_CHALLENGER_SHARED_CONDITION_VARS = (
    "self_employment_income",
    "rental_income",
    "social_security_retirement",
)
PUF_PENSION_CHALLENGER_SHARED_CONDITION_VARS = (
    "social_security_retirement",
    "social_security_disability",
    "unemployment_compensation",
)
PUF_PARTNERSHIP_CHALLENGER_SHARED_CONDITION_VARS = (
    "self_employment_income",
    "rental_income",
    "alimony_income",
)
RENTAL_INCOME_COMPONENT_PREFERRED_CONDITION_VARS = (
    "state_fips",
    "tenure",
    "age",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "income",
    "employment_income",
    "self_employment_income",
    "real_estate_taxes",
)


@dataclass(frozen=True)
class DonorImputationBlockSpec:
    """Declarative donor-model spec for one imputation block."""

    model_variables: tuple[str, ...]
    restored_variables: tuple[str, ...]
    native_entity: EntityType = EntityType.PERSON
    condition_entities: tuple[EntityType, ...] = ()
    match_strategies: Mapping[str, DonorMatchStrategy] = field(default_factory=dict)
    prepare_frame: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    restore_frame: Callable[[pd.DataFrame], pd.DataFrame] | None = None

    def strategy_for(self, variable_name: str) -> DonorMatchStrategy:
        return self.match_strategies.get(variable_name, DonorMatchStrategy.RANK)


@dataclass(frozen=True)
class VariableSemanticSpec:
    """Declarative semantics for variables that can be derived from an atomic basis."""

    native_entity: EntityType = EntityType.PERSON
    condition_entities: tuple[EntityType, ...] = ()
    projection_aggregation: ProjectionAggregation = ProjectionAggregation.FIRST
    support_family: VariableSupportFamily = VariableSupportFamily.CONTINUOUS
    derived_from: tuple[str, ...] = ()
    donor_match_strategy: DonorMatchStrategy = DonorMatchStrategy.RANK
    donor_transform: FrameSemanticTransform | None = None
    donor_check: FrameSemanticCheck | None = None
    preferred_condition_vars: tuple[str, ...] = ()
    supplemental_shared_condition_vars: tuple[str, ...] = ()
    challenger_shared_condition_vars: tuple[str, ...] = ()
    notes: str | None = None

    def is_redundant_given(self, variable_names: Iterable[str]) -> bool:
        """Return whether this variable is redundant given the observed variables."""
        if not self.derived_from:
            return False
        available = set(variable_names)
        return set(self.derived_from).issubset(available)

    @property
    def condition_score_mode(self) -> ConditionScoreMode:
        if self.support_family is VariableSupportFamily.SUPPORT_SENSITIVE:
            return ConditionScoreMode.VALUE_AND_SUPPORT
        return ConditionScoreMode.VALUE_ONLY

    @property
    def allowed_condition_entities(self) -> tuple[EntityType, ...]:
        if self.condition_entities:
            return self.condition_entities
        if self.native_entity is EntityType.PERSON:
            record_entity = getattr(EntityType, "RECORD", None)
            return tuple(entity for entity in EntityType if entity is not record_entity)
        return (EntityType.HOUSEHOLD, self.native_entity)


def zero_minor_employment_income(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce zero employment income for minors on donor-integrated seed frames."""
    if "employment_income" not in frame.columns or "age" not in frame.columns:
        return frame
    ages = pd.to_numeric(frame["age"], errors="coerce")
    if ages.isna().all():
        return frame
    result = frame.copy()
    minor_mask = ages.lt(18).fillna(False)
    if not minor_mask.any():
        return result
    result["employment_income"] = (
        pd.to_numeric(result["employment_income"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    result.loc[minor_mask, "employment_income"] = 0.0
    return result


def suppress_retired_senior_employment_income_without_esi(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Suppress donor-overridden wage income for retired seniors without ESI."""
    required_columns = {"employment_income", "age", "has_esi"}
    if not required_columns.issubset(frame.columns):
        return frame
    ages = pd.to_numeric(frame["age"], errors="coerce")
    if ages.isna().all():
        return frame
    social_security_income = social_security_retirement_compatible_amount(frame)
    has_esi = (
        pd.to_numeric(frame["has_esi"], errors="coerce")
        .fillna(0.0)
        .astype(float)
        .gt(0.0)
    )
    retired_senior_mask = (
        ages.ge(65).fillna(False) & social_security_income.gt(0.0) & ~has_esi
    )
    if not retired_senior_mask.any():
        return frame
    result = frame.copy()
    result["employment_income"] = (
        pd.to_numeric(result["employment_income"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    result.loc[retired_senior_mask, "employment_income"] = 0.0
    return result


def normalize_employment_income_donor_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply donor-side employment income semantic guards in a stable order."""
    adjusted = normalize_social_security_columns(frame)
    adjusted = zero_minor_employment_income(adjusted)
    return suppress_retired_senior_employment_income_without_esi(adjusted)


def minor_positive_employment_income_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows where minors still carry positive employment income."""
    if "employment_income" not in frame.columns or "age" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    ages = pd.to_numeric(frame["age"], errors="coerce")
    income = pd.to_numeric(frame["employment_income"], errors="coerce").fillna(0.0)
    return ages.lt(18).fillna(False) & income.gt(0.0)


VARIABLE_SEMANTIC_SPECS: dict[str, VariableSemanticSpec] = {
    "age": VariableSemanticSpec(
        projection_aggregation=ProjectionAggregation.MAX,
    ),
    "income": VariableSemanticSpec(
        projection_aggregation=ProjectionAggregation.SUM,
    ),
    "state_fips": VariableSemanticSpec(native_entity=EntityType.HOUSEHOLD),
    "tenure": VariableSemanticSpec(native_entity=EntityType.HOUSEHOLD),
    "state": VariableSemanticSpec(native_entity=EntityType.HOUSEHOLD),
    "household_vehicles_owned": VariableSemanticSpec(
        native_entity=EntityType.HOUSEHOLD,
        projection_aggregation=ProjectionAggregation.MAX,
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        notes="Household vehicle count from the SIPP asset donor.",
    ),
    "household_vehicles_value": VariableSemanticSpec(
        native_entity=EntityType.HOUSEHOLD,
        projection_aggregation=ProjectionAggregation.MAX,
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        notes="Household vehicle value from the SIPP asset donor.",
    ),
    "dividend_income": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        derived_from=(
            "qualified_dividend_income",
            "non_qualified_dividend_income",
        ),
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
        supplemental_shared_condition_vars=PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS,
        challenger_shared_condition_vars=(
            PUF_DIVIDEND_INTEREST_CHALLENGER_SHARED_CONDITION_VARS
        ),
        notes="Dividend totals are derived from the qualified and non-qualified atomic basis.",
    ),
    "ordinary_dividend_income": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        derived_from=(
            "qualified_dividend_income",
            "non_qualified_dividend_income",
        ),
        notes="Ordinary dividend totals are derived from the qualified and non-qualified atomic basis.",
    ),
    "qualified_dividend_income": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
        supplemental_shared_condition_vars=PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS,
        challenger_shared_condition_vars=(
            PUF_DIVIDEND_INTEREST_CHALLENGER_SHARED_CONDITION_VARS
        ),
    ),
    "non_qualified_dividend_income": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
        supplemental_shared_condition_vars=PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS,
        challenger_shared_condition_vars=(
            PUF_DIVIDEND_INTEREST_CHALLENGER_SHARED_CONDITION_VARS
        ),
    ),
    "taxable_interest_income": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
        supplemental_shared_condition_vars=PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS,
        challenger_shared_condition_vars=(
            PUF_DIVIDEND_INTEREST_CHALLENGER_SHARED_CONDITION_VARS
        ),
    ),
    "tax_exempt_interest_income": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
        supplemental_shared_condition_vars=PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS,
    ),
    "taxable_pension_income": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
        supplemental_shared_condition_vars=PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS,
        challenger_shared_condition_vars=PUF_PENSION_CHALLENGER_SHARED_CONDITION_VARS,
    ),
    "taxable_social_security": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
    ),
    "state_income_tax_paid": VariableSemanticSpec(
        native_entity=EntityType.TAX_UNIT,
        condition_entities=(EntityType.HOUSEHOLD, EntityType.TAX_UNIT),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
    ),
    "real_estate_tax_paid": VariableSemanticSpec(
        native_entity=EntityType.TAX_UNIT,
        condition_entities=(EntityType.HOUSEHOLD, EntityType.TAX_UNIT),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
    ),
    "mortgage_interest_paid": VariableSemanticSpec(
        native_entity=EntityType.TAX_UNIT,
        condition_entities=(EntityType.HOUSEHOLD, EntityType.TAX_UNIT),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
    ),
    "charitable_cash": VariableSemanticSpec(
        native_entity=EntityType.TAX_UNIT,
        condition_entities=(EntityType.HOUSEHOLD, EntityType.TAX_UNIT),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
    ),
    "charitable_noncash": VariableSemanticSpec(
        native_entity=EntityType.TAX_UNIT,
        condition_entities=(EntityType.HOUSEHOLD, EntityType.TAX_UNIT),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
    ),
    "student_loan_interest": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
    ),
    "ira_deduction": VariableSemanticSpec(
        native_entity=EntityType.TAX_UNIT,
        condition_entities=(EntityType.HOUSEHOLD, EntityType.TAX_UNIT),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
    ),
    "health_savings_account_ald": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
    ),
    "self_employed_health_insurance_ald": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
    ),
    "self_employed_pension_contribution_ald": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
    ),
    "qualified_dividend_share": VariableSemanticSpec(
        native_entity=EntityType.TAX_UNIT,
        condition_entities=(EntityType.HOUSEHOLD, EntityType.TAX_UNIT),
        support_family=VariableSupportFamily.BOUNDED_SHARE,
    ),
    "tax_unit_partnership_s_corp_income": VariableSemanticSpec(
        native_entity=EntityType.TAX_UNIT,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
        supplemental_shared_condition_vars=PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS,
        challenger_shared_condition_vars=(
            PUF_PARTNERSHIP_CHALLENGER_SHARED_CONDITION_VARS
        ),
    ),
    "employment_income": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        donor_transform=FrameSemanticTransform(
            name="normalize_employment_income_donor_values",
            required_columns=("employment_income", "age"),
            transform_frame=normalize_employment_income_donor_values,
            stage=SemanticTransformStage.POST_DONOR_INTEGRATION,
            notes=(
                "Employment income donor overrides should not assign positive wages "
                "to minors and should suppress implausible retired-senior wages "
                "when retirement Social Security is present without ESI."
            ),
        ),
        donor_check=FrameSemanticCheck(
            name="minor_positive_employment_income",
            required_columns=("employment_income", "age"),
            violation_mask=minor_positive_employment_income_mask,
            stage=SemanticTransformStage.POST_DONOR_INTEGRATION,
            notes="Minors should not retain positive donor-overridden wage income.",
        ),
        notes=(
            "Employment income donor overrides should respect basic wage support "
            "semantics for minors and retired seniors."
        ),
    ),
    "self_employment_income": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        notes="Self-employment income is signed and must preserve losses.",
    ),
    "rental_income_positive": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=RENTAL_INCOME_COMPONENT_PREFERRED_CONDITION_VARS,
        notes=(
            "Positive rental-income support should track geography and property-like "
            "predictors instead of generic labor-income conditioning."
        ),
    ),
    "rental_income_negative": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=RENTAL_INCOME_COMPONENT_PREFERRED_CONDITION_VARS,
        notes=(
            "Rental-loss support should track geography and property-like "
            "predictors instead of generic labor-income conditioning."
        ),
    ),
    "partnership_s_corp_income": VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.TAX_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=PUF_IRS_TAX_PREFERRED_CONDITION_VARS,
        supplemental_shared_condition_vars=PUF_IRS_TAX_SUPPLEMENTAL_SHARED_CONDITION_VARS,
        challenger_shared_condition_vars=(
            PUF_PARTNERSHIP_CHALLENGER_SHARED_CONDITION_VARS
        ),
    ),
    "has_medicaid": VariableSemanticSpec(
        projection_aggregation=ProjectionAggregation.MAX,
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        notes="Binary proxy for Medicaid participation on the CPS scaffold.",
    ),
    "public_assistance": VariableSemanticSpec(
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        notes="Public assistance amounts are sparse and should preserve support.",
    ),
    "ssi": VariableSemanticSpec(
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        notes="SSI amounts are sparse and should preserve support.",
    ),
    "social_security": VariableSemanticSpec(
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        notes="Reported Social Security amounts are sparse and support-sensitive.",
    ),
    "snap": VariableSemanticSpec(
        native_entity=EntityType.SPM_UNIT,
        condition_entities=(
            EntityType.PERSON,
            EntityType.HOUSEHOLD,
            EntityType.SPM_UNIT,
        ),
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
    ),
}

# SCF net-worth component leaves (G1). Person-entity, zero-inflated positive
# magnitudes (debt leaves are stored as positive balances; the -1 sign is
# applied only at net-worth reconciliation, mirroring eCPS
# NET_WORTH_COMPONENT_SIGNS). Imputed from the SCF donor on SCF_PREDICTORS
# (age, is_female, cps_race, is_married, own_children_in_household,
# employment_income, interest_dividend_income, social_security_pension_income).
SCF_NET_WORTH_COMPONENT_LEAVES: tuple[str, ...] = (
    "scf_certificates_of_deposit",
    "scf_savings_bonds",
    "scf_retirement_assets",
    "scf_cash_value_life_insurance",
    "scf_other_managed_assets",
    "scf_other_financial_assets",
    "scf_primary_residence_value",
    "scf_other_residential_real_estate",
    "scf_nonresidential_real_estate_equity",
    "scf_business_equity",
    "scf_other_nonfinancial_assets",
    "scf_mortgage_debt",
    "scf_other_residential_debt",
    "scf_other_lines_of_credit",
    "scf_credit_card_debt",
    "scf_vehicle_installment_debt",
    "scf_student_loan_debt",
    "scf_other_installment_debt",
    "scf_other_debt",
)
SCF_COMPONENT_PREFERRED_CONDITION_VARS: tuple[str, ...] = (
    "age",
    "is_female",
    "cps_race",
    "is_married",
    "own_children_in_household",
    "employment_income",
    "interest_dividend_income",
    "social_security_pension_income",
)
for _scf_component_leaf in SCF_NET_WORTH_COMPONENT_LEAVES:
    VARIABLE_SEMANTIC_SPECS[_scf_component_leaf] = VariableSemanticSpec(
        native_entity=EntityType.PERSON,
        support_family=VariableSupportFamily.SUPPORT_SENSITIVE,
        preferred_condition_vars=SCF_COMPONENT_PREFERRED_CONDITION_VARS,
        notes="SCF balance-sheet component leaf; positive magnitude.",
    )

DIVIDEND_COMPONENT_COLUMNS = (
    "qualified_dividend_income",
    "non_qualified_dividend_income",
)
DIVIDEND_TOTAL_COLUMNS = (
    "ordinary_dividend_income",
    "dividend_income",
)
DIVIDEND_SHARE_COLUMN = "qualified_dividend_share"
DIVIDEND_COMPOSITION_MODEL_COLUMNS = (
    "dividend_income",
    DIVIDEND_SHARE_COLUMN,
)
SOCIAL_SECURITY_COMPONENT_COLUMNS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
)
SOCIAL_SECURITY_UNCLASSIFIED_COLUMN = "social_security_unclassified"


def _nonnegative_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return (
        pd.to_numeric(frame[column], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(float)
    )


# Share of a dividend total that is qualified when no observed qualified/
# non-qualified breakdown is available (e.g. CPS DIV_VAL, which reports only a
# total). Basis: SOI 2015 PUF E00650/E00600 = $204.0B/$260.9B = 0.782 qualified.
# Splitting an unsplit total by this share avoids zeroing
# qualified_dividend_income on every CPS-native dividend row (which previously
# dumped 100% into non-qualified and inverted the national qualified vs
# non-qualified split relative to the SOI targets).
UNSPLIT_DIVIDEND_QUALIFIED_SHARE = 0.78


def normalize_dividend_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize dividends onto an atomic basis, then derive totals."""
    result = frame.copy()
    qualified = _nonnegative_series(result, "qualified_dividend_income")
    non_qualified = _nonnegative_series(result, "non_qualified_dividend_income")
    ordinary_total = _nonnegative_series(result, "ordinary_dividend_income")
    dividend_total = _nonnegative_series(result, "dividend_income")
    if "ordinary_dividend_income" in result.columns:
        total = ordinary_total.where(ordinary_total.ne(0.0), dividend_total)
    else:
        total = dividend_total

    has_qualified = "qualified_dividend_income" in result.columns
    has_non_qualified = "non_qualified_dividend_income" in result.columns

    if has_qualified and has_non_qualified:
        component_total = qualified + non_qualified
        total_only = component_total.eq(0.0) & total.gt(0.0)
        # Allocate an unsplit total by the SOI qualified share rather than
        # defaulting the whole amount to non-qualified.
        qualified = qualified.where(
            ~total_only, total * UNSPLIT_DIVIDEND_QUALIFIED_SHARE
        )
        non_qualified = non_qualified.where(
            ~total_only, total * (1.0 - UNSPLIT_DIVIDEND_QUALIFIED_SHARE)
        )
        component_total = qualified + non_qualified
        normalized_total = component_total.where(component_total.ne(0.0), total)
    elif has_qualified:
        normalized_total = np.maximum(
            total.to_numpy(dtype=float), qualified.to_numpy(dtype=float)
        )
        non_qualified = pd.Series(
            normalized_total - qualified.to_numpy(dtype=float),
            index=result.index,
            dtype=float,
        )
        normalized_total = pd.Series(normalized_total, index=result.index, dtype=float)
    elif has_non_qualified:
        normalized_total = np.maximum(
            total.to_numpy(dtype=float),
            non_qualified.to_numpy(dtype=float),
        )
        qualified = pd.Series(
            normalized_total - non_qualified.to_numpy(dtype=float),
            index=result.index,
            dtype=float,
        )
        normalized_total = pd.Series(normalized_total, index=result.index, dtype=float)
    else:
        normalized_total = total.astype(float)
        qualified = normalized_total * UNSPLIT_DIVIDEND_QUALIFIED_SHARE
        non_qualified = normalized_total * (1.0 - UNSPLIT_DIVIDEND_QUALIFIED_SHARE)

    result["qualified_dividend_income"] = qualified.astype(float)
    result["non_qualified_dividend_income"] = non_qualified.astype(float)
    result["ordinary_dividend_income"] = normalized_total.astype(float)
    result["dividend_income"] = normalized_total.astype(float)
    return result


def normalize_social_security_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Social Security onto an explicit component basis.

    Preserve any observed component columns and store any remaining gross
    Social Security residual as an explicit unclassified amount.
    """
    result = frame.copy()
    component_series = {
        column: _nonnegative_series(result, column)
        for column in SOCIAL_SECURITY_COMPONENT_COLUMNS
    }
    component_sum = sum(
        component_series.values(), start=pd.Series(0.0, index=result.index)
    )
    existing_unclassified = _nonnegative_series(
        result, SOCIAL_SECURITY_UNCLASSIFIED_COLUMN
    )

    if "social_security" in result.columns:
        observed_total = _nonnegative_series(result, "social_security")
    else:
        observed_total = _nonnegative_series(result, "gross_social_security")
    normalized_total = pd.Series(
        np.maximum(
            observed_total.to_numpy(dtype=float),
            (component_sum + existing_unclassified).to_numpy(dtype=float),
        ),
        index=result.index,
        dtype=float,
    )
    unclassified = pd.Series(
        np.maximum(
            normalized_total.to_numpy(dtype=float)
            - component_sum.to_numpy(dtype=float),
            0.0,
        ),
        index=result.index,
        dtype=float,
    )

    for column, values in component_series.items():
        result[column] = values.astype(float)
    result[SOCIAL_SECURITY_UNCLASSIFIED_COLUMN] = unclassified.astype(float)
    result["social_security"] = normalized_total.astype(float)
    return result


def social_security_retirement_compatible_amount(frame: pd.DataFrame) -> pd.Series:
    """Return the PE-compatible retirement component amount.

    PolicyEngine models total Social Security as the sum of the four component
    variables. Until we have a better backward allocator, treat any
    unclassified residual as retirement at compatibility points.
    """
    retirement = _nonnegative_series(frame, "social_security_retirement")
    unclassified = _nonnegative_series(frame, SOCIAL_SECURITY_UNCLASSIFIED_COLUMN)
    return (retirement + unclassified).astype(float)


def add_dividend_composition_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add dividend total/share features derived from the atomic basis."""
    result = normalize_dividend_columns(frame)
    total = _nonnegative_series(result, "dividend_income")
    qualified = _nonnegative_series(result, "qualified_dividend_income")
    share_values = np.divide(
        qualified.to_numpy(dtype=float),
        total.to_numpy(dtype=float),
        out=np.zeros(len(result), dtype=float),
        where=total.to_numpy(dtype=float) > 0.0,
    )
    result[DIVIDEND_SHARE_COLUMN] = pd.Series(
        np.clip(share_values, 0.0, 1.0),
        index=result.index,
        dtype=float,
    )
    return result


def restore_dividend_components_from_composition(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct dividend components from total + qualified share."""
    result = frame.copy()
    total = _nonnegative_series(result, "dividend_income")
    share = (
        pd.to_numeric(result.get(DIVIDEND_SHARE_COLUMN, 0.0), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    qualified = pd.Series(
        total.to_numpy(dtype=float) * share.to_numpy(dtype=float),
        index=result.index,
        dtype=float,
    )
    non_qualified = pd.Series(
        total.to_numpy(dtype=float) - qualified.to_numpy(dtype=float),
        index=result.index,
        dtype=float,
    )
    result["qualified_dividend_income"] = qualified
    result["non_qualified_dividend_income"] = non_qualified
    result["ordinary_dividend_income"] = total
    result["dividend_income"] = total
    if DIVIDEND_SHARE_COLUMN in result.columns:
        result = result.drop(columns=[DIVIDEND_SHARE_COLUMN])
    return result


DIVIDEND_DONOR_BLOCK_SPEC = DonorImputationBlockSpec(
    native_entity=EntityType.PERSON,
    condition_entities=(
        EntityType.PERSON,
        EntityType.HOUSEHOLD,
        EntityType.TAX_UNIT,
    ),
    model_variables=DIVIDEND_COMPOSITION_MODEL_COLUMNS,
    restored_variables=DIVIDEND_COMPONENT_COLUMNS,
    match_strategies={DIVIDEND_SHARE_COLUMN: DonorMatchStrategy.RANK},
    prepare_frame=add_dividend_composition_features,
    restore_frame=restore_dividend_components_from_composition,
)


def variable_semantic_spec_for(variable_name: str) -> VariableSemanticSpec:
    """Return semantic metadata for one variable."""
    return VARIABLE_SEMANTIC_SPECS.get(variable_name, VariableSemanticSpec())


def score_donor_condition_var(
    condition_series: pd.Series,
    target_series_list: Iterable[pd.Series],
    *,
    score_modes: Iterable[ConditionScoreMode],
) -> float:
    """Score one shared conditioning variable against one donor target block."""
    condition = pd.to_numeric(
        condition_series,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    if condition.dropna().nunique() <= 1:
        return 0.0

    include_support = ConditionScoreMode.VALUE_AND_SUPPORT in set(score_modes)
    best_score = 0.0
    for target_series in target_series_list:
        target = pd.to_numeric(
            target_series,
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        aligned = pd.concat(
            [condition.rename("condition"), target.rename("target")],
            axis=1,
        ).dropna()
        if len(aligned) < 3 or aligned["target"].nunique() <= 1:
            continue

        value_correlation = aligned["condition"].corr(
            aligned["target"],
            method="spearman",
        )
        if pd.notna(value_correlation):
            best_score = max(best_score, abs(float(value_correlation)))

        if not include_support:
            continue
        support = (aligned["target"].abs() > 0).astype(float)
        if 0.0 < float(support.mean()) < 1.0:
            support_correlation = aligned["condition"].corr(
                support,
                method="spearman",
            )
            if pd.notna(support_correlation):
                best_score = max(best_score, abs(float(support_correlation)))

    return best_score


def is_condition_var_compatible_with_entity(
    condition_variable: str,
    *,
    target_entity: EntityType,
) -> bool:
    """Return whether a condition variable is semantically compatible with a target entity."""
    condition_entity = variable_semantic_spec_for(condition_variable).native_entity
    allowed_entities = VariableSemanticSpec(
        native_entity=target_entity
    ).allowed_condition_entities
    return condition_entity in set(allowed_entities)


def resolve_condition_entities_for_targets(
    target_variables: Iterable[str],
) -> tuple[EntityType, ...]:
    """Return the shared condition-entity policy for one donor target block."""
    target_variables = tuple(dict.fromkeys(target_variables))
    if not target_variables:
        return (EntityType.PERSON, EntityType.HOUSEHOLD)
    allowed_by_target = [
        variable_semantic_spec_for(variable).allowed_condition_entities
        for variable in target_variables
    ]
    shared = set(allowed_by_target[0])
    for allowed_entities in allowed_by_target[1:]:
        shared &= set(allowed_entities)
    if not shared:
        return (EntityType.HOUSEHOLD,)
    return tuple(entity for entity in allowed_by_target[0] if entity in shared)


def is_condition_var_compatible_with_targets(
    condition_variable: str,
    *,
    target_variables: Iterable[str],
) -> bool:
    """Return whether a condition variable is compatible with one donor target block."""
    condition_entity = variable_semantic_spec_for(condition_variable).native_entity
    return condition_entity in set(
        resolve_condition_entities_for_targets(target_variables)
    )


def is_projected_condition_var_compatible(
    condition_variable: str,
    *,
    projected_entity: EntityType,
    allowed_condition_entities: Iterable[EntityType],
) -> bool:
    """Return whether a condition variable remains compatible after projection."""
    condition_entity = variable_semantic_spec_for(condition_variable).native_entity
    record_entity = getattr(EntityType, "RECORD", None)
    allowed_entities = {
        entity for entity in allowed_condition_entities if entity is not record_entity
    }
    if condition_entity in allowed_entities:
        return True
    return (
        condition_entity is EntityType.PERSON and projected_entity in allowed_entities
    )


def donor_imputation_block_specs(
    variable_names: Iterable[str],
) -> tuple[DonorImputationBlockSpec, ...]:
    """Plan donor-imputation model blocks and matching strategies."""
    remaining = set(variable_names)
    block_specs: list[DonorImputationBlockSpec] = []
    if set(DIVIDEND_COMPONENT_COLUMNS).issubset(remaining):
        block_specs.append(DIVIDEND_DONOR_BLOCK_SPEC)
        remaining.difference_update(DIVIDEND_COMPONENT_COLUMNS)
    for variable in sorted(remaining):
        spec = variable_semantic_spec_for(variable)
        block_specs.append(
            DonorImputationBlockSpec(
                native_entity=spec.native_entity,
                condition_entities=resolve_condition_entities_for_targets((variable,)),
                model_variables=(variable,),
                restored_variables=(variable,),
                match_strategies={variable: spec.donor_match_strategy},
            )
        )
    return tuple(block_specs)


def donor_imputation_blocks(
    variable_names: Iterable[str],
) -> tuple[tuple[str, ...], ...]:
    """Plan donor-imputation model blocks without coupling unrelated variables."""
    return tuple(
        block_spec.model_variables
        for block_spec in donor_imputation_block_specs(variable_names)
    )


# Suffixes/tokens that mark a target as an adjustment, deduction, credit, or
# loss - economically *downstream* of the income that drives it, so it should
# be imputed late in the chain (conditioned on the already-imputed income
# spine). Derived from the naming conventions in VARIABLE_SEMANTIC_SPECS.
_DONOR_CHAIN_DEDUCTION_SUFFIXES: tuple[str, ...] = (
    "_ald",  # above-the-line deductions (IRA/HSA/SEP/SE health, etc.)
    "_deduction",
    "_paid",  # itemized paid amounts (mortgage interest, real estate tax)
    "_interest",  # student_loan_interest
    "_debt",  # SCF debt leaves are liabilities, downstream of assets
    "_share",  # qualified_dividend_share is a ratio derived from a total
    "_negative",  # rental_income_negative is the loss leg
)
_DONOR_CHAIN_DEDUCTION_NAMES: frozenset[str] = frozenset(
    {
        "charitable_cash",
        "charitable_noncash",
        "state_income_tax_paid",
        "real_estate_tax_paid",
        "mortgage_interest_paid",
        "ira_deduction",
    }
)


def _donor_chain_conditioner_reference_counts() -> dict[str, int]:
    """Count how often each variable is named as another target's conditioner.

    A donor target that many other targets prefer to condition on is part of
    the income/AGI spine and should be imputed early so later targets can
    chain on its already-imputed value. We read this purely from the existing
    ``preferred_condition_vars`` / supplemental / challenger metadata rather
    than hand-maintaining a spine list.
    """
    counts: dict[str, int] = {}
    for spec in VARIABLE_SEMANTIC_SPECS.values():
        referenced = (
            spec.preferred_condition_vars
            + spec.supplemental_shared_condition_vars
            + spec.challenger_shared_condition_vars
        )
        for variable in referenced:
            counts[variable] = counts.get(variable, 0) + 1
    return counts


def _donor_chain_tier(variable_name: str, *, reference_counts: dict[str, int]) -> int:
    """Coarse economic tier for chain ordering (lower imputes earlier)."""
    spec = variable_semantic_spec_for(variable_name)
    # Tier 0: structural geo/anchor predictors (state/tenure). These are
    # rarely donor targets, but if present they pin geography first.
    if variable_name in {"state_fips", "state", "tenure"}:
        return 0
    # Tier 3: adjustments / deductions / credits / losses - downstream of the
    # income that drives them.
    if variable_name in _DONOR_CHAIN_DEDUCTION_NAMES or any(
        variable_name.endswith(suffix) for suffix in _DONOR_CHAIN_DEDUCTION_SUFFIXES
    ):
        return 3
    # Tier 1: the income spine - variables other targets condition on.
    if reference_counts.get(variable_name, 0) > 0:
        return 1
    # Tier 2: remaining income / receipt magnitudes (dividends, interest,
    # pensions, partnership/S-corp, social security, benefits, asset leaves).
    _ = spec
    return 2


def donor_imputation_chain_order(variable_names: Iterable[str]) -> tuple[str, ...]:
    """Order donor targets "spine first" for sequential chained imputation.

    Income/AGI-driving variables (those other targets prefer to condition on,
    plus geography) are imputed first; dependent deductions, credits, and
    losses last. Within a tier, more-referenced (larger, more upstream)
    variables come first, then a stable alphabetical tiebreak so the order is
    deterministic. The dividend composition pair is kept contiguous in its
    declared order.
    """
    requested = list(dict.fromkeys(variable_names))
    reference_counts = _donor_chain_conditioner_reference_counts()

    def sort_key(variable_name: str) -> tuple[int, int, str]:
        return (
            _donor_chain_tier(variable_name, reference_counts=reference_counts),
            -reference_counts.get(variable_name, 0),
            variable_name,
        )

    return tuple(sorted(requested, key=sort_key))


def _build_donor_chain_block(
    *,
    native_entity: EntityType,
    variable_names: Iterable[str],
) -> DonorImputationBlockSpec:
    """Build one chained donor block for a set of same-native-entity targets.

    ``model_variables`` is the "spine first" chain order (see
    ``donor_imputation_chain_order``). When both dividend components are
    present in this group, they are modelled compositionally - as
    ``dividend_income`` plus ``qualified_dividend_share`` - exactly as
    ``DIVIDEND_DONOR_BLOCK_SPEC`` does, kept contiguous in their declared
    order, and reconstructed back to the component columns on restore.

    Because every target in the group shares ``native_entity``, projection
    onto that entity preserves each variable's grain (no grain change versus
    the legacy per-variable blocks); only the modelling is consolidated into
    one chained call.
    """
    requested = list(dict.fromkeys(variable_names))
    ordered = list(donor_imputation_chain_order(requested))

    models_dividends_compositionally = set(DIVIDEND_COMPONENT_COLUMNS).issubset(
        set(requested)
    )

    model_variables: list[str] = []
    restored_variables: list[str] = []
    inserted_dividend_block = False
    for variable in ordered:
        if models_dividends_compositionally and variable in DIVIDEND_COMPONENT_COLUMNS:
            # Splice the dividend composition pair in (once), contiguous and
            # in declared order, where the first component would have landed.
            if not inserted_dividend_block:
                model_variables.extend(DIVIDEND_COMPOSITION_MODEL_COLUMNS)
                restored_variables.extend(DIVIDEND_COMPONENT_COLUMNS)
                inserted_dividend_block = True
            continue
        model_variables.append(variable)
        restored_variables.append(variable)

    # All targets in this group share ``native_entity``, so the condition
    # policy is that entity's allowed conditioners. Resolve over the requested
    # group targets (not ``model_variables``) so the dividend reparametrization
    # column ``qualified_dividend_share`` - which has a different native entity
    # - does not narrow the PERSON group's conditioning surface (the legacy
    # DIVIDEND_DONOR_BLOCK_SPEC allowed PERSON/HOUSEHOLD/TAX_UNIT conditioning).
    condition_entities = resolve_condition_entities_for_targets(requested)

    match_strategies = {
        variable: variable_semantic_spec_for(variable).donor_match_strategy
        for variable in model_variables
    }

    return DonorImputationBlockSpec(
        native_entity=native_entity,
        condition_entities=condition_entities,
        model_variables=tuple(model_variables),
        restored_variables=tuple(restored_variables),
        match_strategies=match_strategies,
        prepare_frame=(
            add_dividend_composition_features
            if models_dividends_compositionally
            else None
        ),
        restore_frame=(
            restore_dividend_components_from_composition
            if models_dividends_compositionally
            else None
        ),
    )


def _group_donor_targets_by_native_entity(
    variable_names: Iterable[str],
) -> "list[tuple[EntityType, list[str]]]":
    """Group donor targets by their semantic native entity, deterministically.

    Returns ``(native_entity, [variables])`` pairs. Groups are ordered by the
    entity's position in ``EntityType`` so the block sequence is stable across
    runs. The dividend components are always assigned to the PERSON group
    (their native entity) so the compositional reparametrization stays intact.
    """
    requested = list(dict.fromkeys(variable_names))
    groups: dict[EntityType, list[str]] = {}
    for variable in requested:
        native_entity = variable_semantic_spec_for(variable).native_entity
        groups.setdefault(native_entity, []).append(variable)
    entity_order = {entity: index for index, entity in enumerate(EntityType)}
    return [
        (native_entity, groups[native_entity])
        for native_entity in sorted(
            groups, key=lambda entity: entity_order.get(entity, len(entity_order))
        )
    ]


def donor_imputation_chain_block_specs(
    variable_names: Iterable[str],
) -> tuple[DonorImputationBlockSpec, ...]:
    """Plan chained donor blocks: one per native-entity group.

    Targets are grouped by their semantic native entity (PERSON / TAX_UNIT /
    SPM_UNIT / HOUSEHOLD / ...) and each group becomes ONE
    ``DonorImputationBlockSpec`` whose ``model_variables`` is the group's
    "spine first" chain order. microimpute's ``Imputer`` then chains the
    group's targets in a single fit, while each block keeps its own entity
    grain - so there is no projection-grain change versus the legacy
    per-variable blocks; only the modelling is consolidated.

    For the PUF donor this is typically ~1-3 blocks (e.g. a PERSON income
    chain plus a TAX_UNIT deduction chain).
    """
    return tuple(
        _build_donor_chain_block(
            native_entity=native_entity,
            variable_names=group_variables,
        )
        for native_entity, group_variables in _group_donor_targets_by_native_entity(
            variable_names
        )
    )


def donor_imputation_chain_block_spec(
    variable_names: Iterable[str],
) -> DonorImputationBlockSpec:
    """Build a single consolidated chained donor block over all targets.

    Kept for API symmetry and unit testing of the chain-order / dividend
    splicing logic. The pipeline uses the plural
    ``donor_imputation_chain_block_specs`` (one block per native-entity group)
    so that each block preserves its own grain; this single-block form forces
    one block and therefore resolves ``native_entity`` as the broadest shared
    entity across all targets, which can change grain for mixed-entity sets.
    """
    requested = list(dict.fromkeys(variable_names))
    native_entity = _resolve_chain_native_entity(requested)
    return _build_donor_chain_block(
        native_entity=native_entity,
        variable_names=requested,
    )


def _resolve_chain_native_entity(model_variables: Iterable[str]) -> EntityType:
    """Pick the broadest native entity that can host a whole chain block.

    Used only by the single-block ``donor_imputation_chain_block_spec`` helper.
    PERSON-native targets can be hosted on PERSON; when the block mixes PERSON
    with a coarser entity (e.g. TAX_UNIT deductions), fall back to the entity
    that is a valid conditioning entity for every target so projection stays
    well-defined.
    """
    entities = {
        variable_semantic_spec_for(variable).native_entity
        for variable in model_variables
    }
    if not entities:
        return EntityType.PERSON
    if len(entities) == 1:
        return next(iter(entities))
    shared = resolve_condition_entities_for_targets(model_variables)
    # Prefer the finest shared entity that is not the catch-all RECORD entity.
    record_entity = getattr(EntityType, "RECORD", None)
    for entity in shared:
        if entity is not record_entity and entity in entities:
            return entity
    if EntityType.HOUSEHOLD in shared:
        return EntityType.HOUSEHOLD
    return next(iter(shared)) if shared else EntityType.PERSON


def apply_donor_variable_semantics(
    frame: pd.DataFrame,
    variable_names: Iterable[str],
) -> pd.DataFrame:
    """Apply post-imputation semantic guards for donor-integrated variables."""
    transforms: list[FrameSemanticTransform] = []
    seen_transform_names: set[str] = set()
    for variable_name in tuple(dict.fromkeys(variable_names)):
        transform = variable_semantic_spec_for(variable_name).donor_transform
        if transform is None or transform.name in seen_transform_names:
            continue
        transforms.append(transform)
        seen_transform_names.add(transform.name)
    return apply_frame_semantic_transforms(frame, transforms)


def validate_donor_variable_semantics(
    frame: pd.DataFrame,
    variable_names: Iterable[str],
) -> tuple[FrameSemanticCheckReport, ...]:
    """Evaluate semantic checks for donor-integrated variables."""
    checks: list[FrameSemanticCheck] = []
    seen_check_names: set[str] = set()
    for variable_name in tuple(dict.fromkeys(variable_names)):
        check = variable_semantic_spec_for(variable_name).donor_check
        if check is None or check.name in seen_check_names:
            continue
        checks.append(check)
        seen_check_names.add(check.name)
    return evaluate_frame_semantic_checks(frame, checks)


def resolve_variable_semantic_capabilities(
    variable_names: Iterable[str],
) -> dict[str, SourceVariableCapability]:
    """Resolve generic capabilities implied by variable semantics alone."""
    available = tuple(dict.fromkeys(variable_names))
    resolved: dict[str, SourceVariableCapability] = {}
    for variable, spec in VARIABLE_SEMANTIC_SPECS.items():
        if variable not in available or not spec.is_redundant_given(available):
            continue
        resolved[variable] = SourceVariableCapability(
            authoritative=False,
            usable_as_condition=False,
            notes=spec.notes,
        )
    return resolved


def prune_redundant_variables(variable_names: Iterable[str]) -> set[str]:
    """Drop derived variables when their atomic basis is already present."""
    result = set(variable_names)
    for variable in resolve_variable_semantic_capabilities(result):
        result.discard(variable)
    return result
