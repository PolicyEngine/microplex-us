"""Tests for loading US calibration targets from Supabase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
from microplex.core import EntityType
from microplex.targets import FilterOperator, TargetAggregation, TargetQuery

from microplex_us.calibration_harness import CalibrationHarness
from microplex_us.supabase_targets import (
    SUPABASE_SUPPORTED_BY_COLUMN_MAP_KEY,
    SUPABASE_TARGET_TYPE_KEY,
    SUPABASE_VARIABLE_KEY,
    SupabaseTargetLoader,
    SupabaseTargetProvider,
)
from microplex_us.target_registry import (
    US_TARGET_CATEGORY_KEY,
    US_TARGET_LEVEL_KEY,
    TargetCategory,
    TargetLevel,
)

SUPABASE_URL = "https://test.supabase.co"
SUPABASE_KEY = "test-key"


@dataclass
class MockResponse:
    payload: list[dict[str, Any]]

    def json(self) -> list[dict[str, Any]]:
        return self.payload

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def loader() -> SupabaseTargetLoader:
    return SupabaseTargetLoader(SUPABASE_URL, SUPABASE_KEY)


@pytest.fixture
def provider() -> SupabaseTargetProvider:
    return SupabaseTargetProvider(SUPABASE_URL, SUPABASE_KEY)


@pytest.fixture
def request_queue(monkeypatch: pytest.MonkeyPatch):
    calls = []
    responses: list[MockResponse] = []

    def fake_get(
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any],
        timeout: int,
    ) -> MockResponse:
        calls.append(
            {
                "url": url,
                "headers": headers,
                "params": params,
                "timeout": timeout,
            }
        )
        return responses.pop(0)

    def queue(*payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        responses[:] = [MockResponse(payload) for payload in payloads]
        return calls

    monkeypatch.setattr("microplex_us.supabase_targets.requests.get", fake_get)
    return queue


def test_missing_service_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLICYENGINE_SUPABASE_SERVICE_KEY", raising=False)

    with pytest.raises(ValueError, match="POLICYENGINE_SUPABASE_SERVICE_KEY"):
        SupabaseTargetLoader(SUPABASE_URL)


def test_load_all_targets(loader: SupabaseTargetLoader, request_queue) -> None:
    calls = request_queue(
        [
            {
                "id": "t1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
            {
                "id": "t2",
                "variable": "snap_spending",
                "value": 103100000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "USDA SNAP", "institution": "USDA"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
        ]
    )

    targets = loader.load_all()

    assert [target["variable"] for target in targets] == [
        "employment_income",
        "snap_spending",
    ]
    assert calls[0]["url"] == f"{SUPABASE_URL}/rest/v1/targets"
    assert calls[0]["params"]["limit"] == 1000
    assert calls[0]["params"]["offset"] == 0


def test_load_by_institution(
    loader: SupabaseTargetLoader,
    request_queue,
) -> None:
    request_queue(
        [{"id": "src-1", "institution": "IRS", "name": "IRS SOI"}],
        [
            {
                "id": "t1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            }
        ],
    )

    targets = loader.load_by_institution("IRS")

    assert len(targets) == 1
    assert targets[0]["source"]["institution"] == "IRS"


def test_load_by_period(loader: SupabaseTargetLoader, request_queue) -> None:
    calls = request_queue(
        [
            {
                "id": "t1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            }
        ]
    )

    targets = loader.load_by_period(2024)

    assert len(targets) == 1
    assert calls[0]["params"]["period"] == "eq.2024"


def test_cps_column_mapping(loader: SupabaseTargetLoader) -> None:
    mapping = loader.get_cps_column_map()

    assert mapping["employment_income"] == "employment_income"
    assert mapping["self_employment_income"] == "self_employment_income"
    assert mapping["dividend_income"] == "dividend_income"
    assert mapping["snap_spending"] == "snap"
    assert mapping["ssi_spending"] == "ssi"
    assert mapping["eitc_spending"] == "eitc"


def test_build_continuous_targets(
    loader: SupabaseTargetLoader,
    request_queue,
) -> None:
    request_queue(
        [
            {
                "id": "t1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
            {
                "id": "t2",
                "variable": "snap_spending",
                "value": 103100000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "USDA SNAP", "institution": "USDA"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
        ]
    )

    constraints = loader.build_calibration_constraints()

    assert constraints["employment_income"] == 9022400000000
    assert constraints["snap"] == 103100000000


def test_build_state_targets(
    loader: SupabaseTargetLoader,
    request_queue,
) -> None:
    request_queue(
        [
            {
                "id": "t1",
                "variable": "medicaid_enrollment",
                "value": 14000000,
                "target_type": "count",
                "period": 2024,
                "source": {"name": "CMS Medicaid", "institution": "HHS"},
                "stratum": {"name": "California", "jurisdiction": "us-ca"},
            }
        ]
    )

    constraints = loader.build_calibration_constraints(include_states=True)

    assert constraints["medicaid_ca"] == 14000000


def test_get_summary(loader: SupabaseTargetLoader, request_queue) -> None:
    request_queue(
        [
            {
                "id": "t1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
            {
                "id": "t2",
                "variable": "person_count",
                "value": 330000000,
                "target_type": "count",
                "period": 2024,
                "source": {"name": "Census", "institution": "Census"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
        ]
    )

    summary = loader.get_summary()

    assert summary == {
        "total": 2,
        "by_institution": {"IRS": 1, "Census": 1},
        "by_variable": {"employment_income": 1, "person_count": 1},
        "by_type": {"amount": 1, "count": 1},
    }


def test_target_from_row_builds_national_sum_spec(
    provider: SupabaseTargetProvider,
) -> None:
    spec = provider.target_from_row(
        {
            "id": "target-1",
            "variable": "employment_income",
            "value": 9022400000000,
            "target_type": "amount",
            "period": 2024,
            "source": {"name": "IRS SOI", "institution": "IRS"},
            "stratum": {"name": "National", "jurisdiction": "us"},
        }
    )

    assert spec.name == "employment_income"
    assert spec.entity is EntityType.PERSON
    assert spec.aggregation is TargetAggregation.SUM
    assert spec.measure == "employment_income"
    assert spec.filters == ()
    assert spec.value == 9022400000000
    assert spec.source == "IRS SOI"
    assert spec.metadata[SUPABASE_VARIABLE_KEY] == "employment_income"
    assert spec.metadata[SUPABASE_TARGET_TYPE_KEY] == "amount"
    assert spec.metadata[SUPABASE_SUPPORTED_BY_COLUMN_MAP_KEY] is True
    assert spec.metadata[US_TARGET_CATEGORY_KEY] == TargetCategory.INCOME.value
    assert spec.metadata[US_TARGET_LEVEL_KEY] == TargetLevel.NATIONAL.value


def test_target_from_row_builds_state_count_spec(
    provider: SupabaseTargetProvider,
) -> None:
    spec = provider.target_from_row(
        {
            "id": "target-2",
            "variable": "medicaid_enrollment",
            "value": 14000000,
            "target_type": "count",
            "period": 2024,
            "source": {"name": "CMS Medicaid", "institution": "HHS"},
            "stratum": {"name": "California", "jurisdiction": "us-ca"},
        }
    )

    assert spec.name == "medicaid_enrollment_us_ca"
    assert spec.entity is EntityType.PERSON
    assert spec.aggregation is TargetAggregation.COUNT
    assert spec.measure is None
    assert spec.filters[0].feature == "medicaid"
    assert spec.filters[0].operator is FilterOperator.GT
    assert spec.filters[0].value == 0
    assert spec.filters[1].feature == "state_fips"
    assert spec.filters[1].operator is FilterOperator.EQ
    assert spec.filters[1].value == "06"
    assert spec.required_features == ("medicaid", "state_fips")
    assert spec.metadata[US_TARGET_CATEGORY_KEY] == TargetCategory.HEALTH.value
    assert spec.metadata[US_TARGET_LEVEL_KEY] == TargetLevel.STATE.value


def test_target_from_row_keeps_unsupported_variables_classifiable(
    provider: SupabaseTargetProvider,
) -> None:
    spec = provider.target_from_row(
        {
            "id": "target-3",
            "variable": "unknown_cash_income",
            "value": 100,
            "target_type": "amount",
            "period": 2024,
            "source": {"name": "Unknown", "institution": "Other"},
            "stratum": {"name": "National", "jurisdiction": "us"},
        }
    )

    assert spec.measure == "unknown_cash_income"
    assert spec.required_features == ("unknown_cash_income",)
    assert spec.metadata[SUPABASE_SUPPORTED_BY_COLUMN_MAP_KEY] is False


def test_load_target_set_filters_rows_with_core_query(
    provider: SupabaseTargetProvider,
    request_queue,
) -> None:
    calls = request_queue(
        [
            {
                "id": "target-1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
            {
                "id": "target-2",
                "variable": "snap_spending",
                "value": 103100000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "USDA SNAP", "institution": "USDA"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
            {
                "id": "target-3",
                "variable": "unknown_cash_income",
                "value": 100,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "Unknown", "institution": "Other"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
        ]
    )

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            entity=EntityType.PERSON,
            metadata_filters={US_TARGET_CATEGORY_KEY: TargetCategory.INCOME.value},
            provider_filters={"include_unsupported": False},
        )
    )

    assert [target.name for target in target_set.targets] == ["employment_income"]
    assert calls[0]["params"]["period"] == "eq.2024"


def test_calibration_harness_can_use_supabase_target_provider(
    provider: SupabaseTargetProvider,
    request_queue,
) -> None:
    request_queue(
        [
            {
                "id": "target-1",
                "variable": "employment_income",
                "value": 30,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
            {
                "id": "target-2",
                "variable": "unknown_cash_income",
                "value": 100,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "Unknown", "institution": "Other"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
        ]
    )
    harness = CalibrationHarness(target_provider=provider)
    frame = pd.DataFrame(
        {
            "employment_income": [10.0, 20.0],
            "weight": [1.0, 1.0],
        }
    )

    result = harness.run_experiment(
        frame,
        "supabase_income",
        categories=[TargetCategory.INCOME],
        only_available=True,
        period=2024,
        provider_filters={"include_unsupported": False},
        entity=EntityType.PERSON,
        verbose=False,
    )

    assert result.targets_used == ["employment_income"]
    assert result.errors == {"employment_income": 0.0}
