"""Tests for translating US targets into the canonical core target spec."""

from microplex.core import EntityType
from microplex.targets import TargetAggregation

from microplex_us.policyengine.us import (
    PolicyEngineUSConstraint,
    PolicyEngineUSDBTarget,
)
from microplex_us.targets import (
    policyengine_db_target_to_canonical_spec,
)


class TestPolicyEngineTargetAdapters:
    def test_policyengine_count_target_inferrs_entity_and_filters(self):
        target = PolicyEngineUSDBTarget(
            target_id=1,
            variable="person_count",
            period=2024,
            stratum_id=2,
            reform_id=0,
            value=250.0,
            active=True,
            geo_level="state",
            geographic_id="06",
            constraints=(
                PolicyEngineUSConstraint("state_fips", "==", "06"),
                PolicyEngineUSConstraint("age", ">=", "65"),
            ),
        )

        canonical = policyengine_db_target_to_canonical_spec(target)

        assert canonical.entity is EntityType.PERSON
        assert canonical.aggregation is TargetAggregation.COUNT
        assert canonical.measure is None
        assert canonical.filters[0].feature == "state_fips"
        assert canonical.filters[1].feature == "age"
        assert canonical.metadata["target_id"] == 1

    def test_policyengine_sum_target_uses_override_for_entity(self):
        target = PolicyEngineUSDBTarget(
            target_id=2,
            variable="snap",
            period=2024,
            stratum_id=1,
            reform_id=0,
            value=10_000.0,
            active=True,
        )

        canonical = policyengine_db_target_to_canonical_spec(
            target,
            entity_overrides={"snap": EntityType.SPM_UNIT},
        )

        assert canonical.entity is EntityType.SPM_UNIT
        assert canonical.aggregation is TargetAggregation.SUM
        assert canonical.measure == "snap"

    def test_calculated_output_targets_require_policyengine_materialization(self):
        target = PolicyEngineUSDBTarget(
            target_id=3,
            variable="snap",
            period=2024,
            stratum_id=1,
            reform_id=0,
            value=10_000.0,
            active=True,
        )

        canonical = policyengine_db_target_to_canonical_spec(target)

        assert canonical.sim_modifier_names == ("policyengine_us_materialize",)
        assert canonical.sim_modifiers[0].parameters == {"features": ["snap"]}

    def test_takeup_input_constraints_require_rerandomized_takeup_handler(self):
        target = PolicyEngineUSDBTarget(
            target_id=4,
            variable="household_count",
            period=2024,
            stratum_id=1,
            reform_id=0,
            value=10_000.0,
            active=True,
            constraints=(
                PolicyEngineUSConstraint("takes_up_snap_if_eligible", "==", "1"),
            ),
        )

        canonical = policyengine_db_target_to_canonical_spec(target)

        assert canonical.sim_modifier_names == ("rerandomize_takeup",)
        assert canonical.sim_modifiers[0].parameters == {
            "features": ["takes_up_snap_if_eligible"]
        }
