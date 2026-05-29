"""Tests for the Forbes fixed-spine source contract."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from microplex.core import EntityType
from microplex.targets import (
    TargetAggregation,
    TargetFilter,
    TargetSet,
    TargetSpec,
)

from microplex_us.data_sources.forbes import (
    ForbesFixedSpineConfig,
    append_forbes_fixed_spine_tables,
    build_forbes_fixed_spine,
    fixed_spine_contribution_diagnostics_json,
    read_forbes_fixed_spine_records,
    residualize_targets_for_fixed_spine,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    build_policyengine_us_export_variable_maps,
)


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "forbes_unit_id": "forbes-1",
                "name": "Example Founder",
                "rank": 1,
                "state_fips": 6,
                "age": 71,
                "is_female": 0,
                "net_worth": 10_000_000_000.0,
                "employment_income_before_lsr": 2_000_000.0,
                "taxable_interest_income": 40_000_000.0,
                "qualified_dividend_income": 80_000_000.0,
                "long_term_capital_gains_before_response": 500_000_000.0,
                "weight": 1.0,
            }
        ]
    )


def test_build_forbes_fixed_spine_splits_weights_and_keeps_metadata_separate():
    spine = build_forbes_fixed_spine(
        _records(),
        config=ForbesFixedSpineConfig(
            snapshot_id="forbes-test-2024",
            replicates_per_unit=4,
        ),
    )

    assert len(spine.tables.households) == 4
    assert len(spine.tables.persons) == 4
    assert spine.tables.households["household_weight"].tolist() == pytest.approx(
        [0.25, 0.25, 0.25, 0.25]
    )
    assert spine.tables.households["net_worth"].tolist() == pytest.approx(
        [10_000_000_000.0] * 4
    )
    assert spine.tables.households["state_fips"].tolist() == [6] * 4

    for table in (
        spine.tables.households,
        spine.tables.persons,
        spine.tables.tax_units,
        spine.tables.spm_units,
        spine.tables.families,
        spine.tables.marital_units,
    ):
        assert table is not None
        assert not any(column.startswith("forbes_") for column in table.columns)

    assert spine.record_metadata["forbes_name"].tolist() == ["Example Founder"] * 4
    assert spine.record_metadata["replicate_index"].tolist() == [0, 1, 2, 3]
    assert spine.source_metadata["snapshot_id"] == "forbes-test-2024"
    assert spine.source_metadata["record_count"] == 4


def test_forbes_fixed_spine_export_maps_do_not_include_source_diagnostics():
    class FakeEntity:
        def __init__(self, key):
            self.key = key

    class FakeVariable:
        def __init__(self, entity):
            self.entity = FakeEntity(entity)

    class FakeSystem:
        variables = {
            "state_fips": FakeVariable("household"),
            "net_worth": FakeVariable("household"),
            "age": FakeVariable("person"),
            "taxable_interest_income": FakeVariable("person"),
            "qualified_dividend_income": FakeVariable("person"),
            "long_term_capital_gains_before_response": FakeVariable("person"),
            "forbes_rank": FakeVariable("household"),
            "forbes_name": FakeVariable("person"),
        }

    spine = build_forbes_fixed_spine(_records())

    export_maps = build_policyengine_us_export_variable_maps(
        spine.tables,
        tax_benefit_system=FakeSystem(),
    )
    exported_variables = {
        variable
        for entity_map in export_maps.values()
        for variable in entity_map.values()
    }

    assert "net_worth" in exported_variables
    assert "long_term_capital_gains_before_response" in exported_variables
    assert not any(variable.startswith("forbes_") for variable in exported_variables)


def test_read_forbes_fixed_spine_records_tracks_source_checksum(tmp_path):
    path = tmp_path / "forbes.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in _records().to_dict("records"))
    )

    records = read_forbes_fixed_spine_records(path)
    spine = build_forbes_fixed_spine(
        records,
        config=ForbesFixedSpineConfig(replicates_per_unit=2),
        source_path=path,
    )

    assert len(records) == 1
    assert spine.source_metadata["source_path"] == str(path)
    assert isinstance(spine.source_metadata["source_sha256"], str)
    assert len(spine.source_metadata["source_sha256"]) == 64


def test_append_forbes_fixed_spine_tables_keeps_fixed_weights_post_calibration():
    base = PolicyEngineUSEntityTableBundle(
        households=pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [99.0],
                "state_fips": [6],
            }
        ),
        persons=pd.DataFrame(
            {
                "person_id": [10],
                "household_id": [1],
                "weight": [99.0],
            }
        ),
    )
    spine = build_forbes_fixed_spine(
        _records(),
        config=ForbesFixedSpineConfig(replicates_per_unit=2),
    )

    appended = append_forbes_fixed_spine_tables(base, spine)

    assert appended.households["household_weight"].sum() == pytest.approx(100.0)
    assert appended.persons["weight"].sum() == pytest.approx(100.0)
    assert not any(
        column.startswith("forbes_") for column in appended.households.columns
    )


def test_residualize_targets_for_fixed_spine_subtracts_additive_contributions():
    spine = build_forbes_fixed_spine(
        _records(),
        config=ForbesFixedSpineConfig(replicates_per_unit=5),
    )
    targets = TargetSet(
        [
            TargetSpec(
                name="national_net_worth",
                entity=EntityType.HOUSEHOLD,
                value=15_000_000_000.0,
                period=2024,
                measure="net_worth",
                aggregation=TargetAggregation.SUM,
            ),
            TargetSpec(
                name="ca_ltcg",
                entity=EntityType.PERSON,
                value=800_000_000.0,
                period=2024,
                measure="long_term_capital_gains_before_response",
                aggregation=TargetAggregation.SUM,
                filters=(TargetFilter("state_fips", "==", "06"),),
            ),
            TargetSpec(
                name="ca_top_tail_person_count",
                entity=EntityType.PERSON,
                value=12.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                filters=(
                    TargetFilter("state_fips", "==", "06"),
                    TargetFilter("long_term_capital_gains_before_response", ">", 0),
                ),
            ),
        ]
    )

    result = residualize_targets_for_fixed_spine(targets, spine.tables)
    residuals = {target.name: target.value for target in result.targets.targets}

    assert residuals["national_net_worth"] == pytest.approx(5_000_000_000.0)
    assert residuals["ca_ltcg"] == pytest.approx(300_000_000.0)
    assert residuals["ca_top_tail_person_count"] == pytest.approx(11.0)
    assert [item.status for item in result.contributions] == [
        "supported",
        "supported",
        "supported",
    ]
    assert result.targets.targets[0].metadata["fixed_spine_residualization"] == {
        "original_value": 15_000_000_000.0,
        "fixed_spine_contribution": 10_000_000_000.0,
        "residual_value": 5_000_000_000.0,
        "clamped": False,
    }


def test_residualize_targets_for_fixed_spine_reports_unsupported_mean_targets():
    spine = build_forbes_fixed_spine(_records())
    targets = TargetSet(
        [
            TargetSpec(
                name="mean_net_worth",
                entity=EntityType.HOUSEHOLD,
                value=100.0,
                period=2024,
                measure="net_worth",
                aggregation=TargetAggregation.MEAN,
            )
        ]
    )

    result = residualize_targets_for_fixed_spine(targets, spine.tables)

    assert result.targets.targets[0].value == 100.0
    assert result.contributions[0].status == "unsupported"
    assert "not additive" in result.contributions[0].reason
    diagnostics = json.loads(fixed_spine_contribution_diagnostics_json(result))
    assert diagnostics[0]["target_name"] == "mean_net_worth"
