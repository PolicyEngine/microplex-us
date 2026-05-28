"""Adapters from US-specific target representations to core microplex target specs."""

from __future__ import annotations

from collections.abc import Iterable

from microplex.core import EntityType
from microplex.targets import (
    TargetAggregation,
    TargetFilter,
    TargetSet,
)
from microplex.targets import (
    TargetSpec as CanonicalTargetSpec,
)

from microplex_us.microdata_roles import policyengine_us_variable_role
from microplex_us.policyengine.us import (
    PolicyEngineUSConstraint,
    PolicyEngineUSDBTarget,
)

POLICYENGINE_US_COUNT_ENTITIES: dict[str, EntityType] = {
    "household_count": EntityType.HOUSEHOLD,
    "person_count": EntityType.PERSON,
    "tax_unit_count": EntityType.TAX_UNIT,
    "spm_unit_count": EntityType.SPM_UNIT,
    "family_count": EntityType.FAMILY,
}

POLICYENGINE_US_ACTUAL_ACA_PTC_VARIABLE = "assigned_aca_ptc"


def policyengine_db_target_to_canonical_spec(
    target: PolicyEngineUSDBTarget,
    *,
    default_entity: EntityType | str = EntityType.HOUSEHOLD,
    entity_overrides: dict[str, EntityType] | None = None,
) -> CanonicalTargetSpec:
    """Translate a PolicyEngine US DB target row into the canonical core spec."""
    resolved_default_entity = (
        default_entity
        if isinstance(default_entity, EntityType)
        else EntityType(default_entity)
    )
    resolved_entity = (
        (entity_overrides or {}).get(target.variable)
        or POLICYENGINE_US_COUNT_ENTITIES.get(target.variable)
        or resolved_default_entity
    )
    aggregation = (
        TargetAggregation.COUNT
        if target.variable.endswith("_count")
        else TargetAggregation.SUM
    )
    measure_variable = _policyengine_db_target_measure_variable(target)
    measure = None if aggregation is TargetAggregation.COUNT else measure_variable
    model_variable = measure_variable if measure is not None else target.variable
    filters = tuple(
        _policyengine_db_constraint_to_target_filter(target, constraint)
        for constraint in target.constraints
    )

    return CanonicalTargetSpec(
        name=f"policyengine_us_target_{target.target_id}",
        entity=resolved_entity,
        value=target.value,
        period=target.period,
        measure=measure,
        aggregation=aggregation,
        filters=filters,
        tolerance=target.tolerance,
        source=target.source,
        description=target.notes,
        metadata={
            "target_id": target.target_id,
            "variable": target.variable,
            "stratum_id": target.stratum_id,
            "stratum_definition_hash": target.definition_hash,
            "parent_stratum_id": target.parent_stratum_id,
            "reform_id": target.reform_id,
            "active": target.active,
            "geo_level": target.geo_level,
            "geographic_id": target.geographic_id,
            "domain_variable": target.domain_variable,
            "domain_variables": target.domain_variables,
            "model_variable_role": policyengine_us_variable_role(model_variable).value,
            "target_semantic": (
                "count" if aggregation is TargetAggregation.COUNT else "amount"
            ),
            "constraint_count": len(target.constraints),
        },
    )


def _policyengine_db_target_uses_aca_ptc(target: PolicyEngineUSDBTarget) -> bool:
    return (
        target.variable == "aca_ptc"
        or "aca_ptc" in target.domain_variables
        or any(constraint.variable == "aca_ptc" for constraint in target.constraints)
    )


def _policyengine_db_target_measure_variable(target: PolicyEngineUSDBTarget) -> str:
    if target.variable == "aca_ptc":
        return POLICYENGINE_US_ACTUAL_ACA_PTC_VARIABLE
    return target.variable


def _policyengine_db_constraint_to_target_filter(
    target: PolicyEngineUSDBTarget,
    constraint: PolicyEngineUSConstraint,
) -> TargetFilter:
    feature = constraint.variable
    if feature == "aca_ptc" and _policyengine_db_target_uses_aca_ptc(target):
        feature = POLICYENGINE_US_ACTUAL_ACA_PTC_VARIABLE
    return TargetFilter(
        feature=feature,
        operator=constraint.operation,
        value=constraint.value,
    )


def policyengine_db_targets_to_canonical_set(
    targets: Iterable[PolicyEngineUSDBTarget],
    *,
    default_entity: EntityType | str = EntityType.HOUSEHOLD,
    entity_overrides: dict[str, EntityType] | None = None,
) -> TargetSet:
    """Translate a sequence of PolicyEngine US DB targets into a canonical target set."""
    return TargetSet(
        [
            policyengine_db_target_to_canonical_spec(
                target,
                default_entity=default_entity,
                entity_overrides=entity_overrides,
            )
            for target in targets
        ]
    )
