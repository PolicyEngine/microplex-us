"""US Supabase calibration target loader."""

from __future__ import annotations

import os
from typing import Any

import requests
from microplex.core import EntityType
from microplex.targets import (
    FilterOperator,
    TargetAggregation,
    TargetFilter,
    TargetQuery,
    TargetSet,
    TargetSpec,
    apply_target_query,
)

from microplex_us.target_registry import (
    US_TARGET_AVAILABLE_KEY,
    US_TARGET_CATEGORY_KEY,
    US_TARGET_GROUP_KEY,
    US_TARGET_IMPUTATION_KEY,
    US_TARGET_LEVEL_KEY,
    TargetCategory,
    TargetLevel,
)

SUPABASE_TARGET_ID_KEY = "supabase_target_id"
SUPABASE_VARIABLE_KEY = "supabase_variable"
SUPABASE_TARGET_TYPE_KEY = "supabase_target_type"
SUPABASE_JURISDICTION_KEY = "supabase_jurisdiction"
SUPABASE_STRATUM_NAME_KEY = "supabase_stratum_name"
SUPABASE_SOURCE_INSTITUTION_KEY = "supabase_source_institution"
SUPABASE_SUPPORTED_BY_COLUMN_MAP_KEY = "supabase_supported_by_column_map"

_COUNT_ALL_VARIABLES = {
    "family_count",
    "household_count",
    "person_count",
    "spm_unit_count",
    "tax_unit_count",
}

_COUNT_ENTITY_MAP = {
    "family_count": EntityType.FAMILY,
    "household_count": EntityType.HOUSEHOLD,
    "person_count": EntityType.PERSON,
    "spm_unit_count": EntityType.SPM_UNIT,
    "tax_unit_count": EntityType.TAX_UNIT,
}

_INCOME_VARIABLES = {
    "alimony_income",
    "dividend_income",
    "employment_income",
    "farm_income",
    "interest_income",
    "long_term_capital_gains",
    "partnership_s_corp_income",
    "rental_income",
    "self_employment_income",
    "short_term_capital_gains",
    "social_security",
    "tax_exempt_pension_income",
    "taxable_pension_income",
    "unemployment_compensation",
}

_BENEFIT_VARIABLES = {
    "eitc_spending",
    "snap_households",
    "snap_spending",
    "social_security_spending",
    "ssi_spending",
    "unemployment_spending",
}

_HEALTH_VARIABLES = {
    "aca_enrollment",
    "health_insurance_premiums",
    "medicaid_enrollment",
    "other_medical_expenses",
}

_TAX_UNIT_VARIABLES = {
    "eitc_spending",
}

_HOUSEHOLD_VARIABLES = {
    "snap_households",
    "snap_spending",
}


class SupabaseTargetLoader:
    """Load US calibration targets from the microplex Supabase schema."""

    # Mapping from Supabase variable names to CPS column names.
    CPS_COLUMN_MAP = {
        "employment_income": "employment_income",
        "self_employment_income": "self_employment_income",
        "dividend_income": "dividend_income",
        "interest_income": "interest_income",
        "rental_income": "rental_income",
        "social_security": "social_security",
        "unemployment_compensation": "unemployment_compensation",
        "taxable_pension_income": "taxable_pension_income",
        "tax_exempt_pension_income": "tax_exempt_pension_income",
        "long_term_capital_gains": "long_term_capital_gains",
        "short_term_capital_gains": "short_term_capital_gains",
        "partnership_s_corp_income": "partnership_s_corp_income",
        "farm_income": "farm_income",
        "alimony_income": "alimony_income",
        "snap_spending": "snap",
        "ssi_spending": "ssi",
        "eitc_spending": "eitc",
        "social_security_spending": "social_security",
        "unemployment_spending": "unemployment_compensation",
        "medicaid_enrollment": "medicaid",
        "aca_enrollment": "aca",
        "snap_households": "snap",
        "health_insurance_premiums": "health_insurance_premiums",
        "other_medical_expenses": "medical_expenses",
    }

    STATE_FIPS = {
        "01": "al",
        "02": "ak",
        "04": "az",
        "05": "ar",
        "06": "ca",
        "08": "co",
        "09": "ct",
        "10": "de",
        "11": "dc",
        "12": "fl",
        "13": "ga",
        "15": "hi",
        "16": "id",
        "17": "il",
        "18": "in",
        "19": "ia",
        "20": "ks",
        "21": "ky",
        "22": "la",
        "23": "me",
        "24": "md",
        "25": "ma",
        "26": "mi",
        "27": "mn",
        "28": "ms",
        "29": "mo",
        "30": "mt",
        "31": "ne",
        "32": "nv",
        "33": "nh",
        "34": "nj",
        "35": "nm",
        "36": "ny",
        "37": "nc",
        "38": "nd",
        "39": "oh",
        "40": "ok",
        "41": "or",
        "42": "pa",
        "44": "ri",
        "45": "sc",
        "46": "sd",
        "47": "tn",
        "48": "tx",
        "49": "ut",
        "50": "vt",
        "51": "va",
        "53": "wa",
        "54": "wv",
        "55": "wi",
        "56": "wy",
    }

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        schema: str = "microplex",
    ) -> None:
        """Initialize the loader.

        Args:
            url: Supabase URL. Defaults to SUPABASE_URL env var.
            key: Supabase key. Defaults to POLICYENGINE_SUPABASE_SERVICE_KEY env var.
            schema: Schema to use. Defaults to 'microplex'.
        """
        self.url = url or os.environ.get(
            "SUPABASE_URL",
            "https://nsupqhfchdtqclomlrgs.supabase.co",
        )
        self.key = key or os.environ.get("POLICYENGINE_SUPABASE_SERVICE_KEY")
        if not self.key:
            raise ValueError(
                "Supabase service key must be provided via the key argument or "
                "POLICYENGINE_SUPABASE_SERVICE_KEY."
            )
        self.base_url = f"{self.url}/rest/v1"
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept-Profile": schema,
            "Content-Profile": schema,
        }
        self._cache = {}

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        paginate: bool = True,
    ) -> list[dict[str, Any]]:
        """Make a GET request to Supabase with optional pagination."""
        url = f"{self.base_url}/{endpoint}"
        params = params or {}

        if not paginate:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        all_results = []
        offset = 0
        limit = 1000

        while True:
            page_params = {**params, "limit": limit, "offset": offset}
            response = requests.get(
                url,
                headers=self.headers,
                params=page_params,
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()

            if not results:
                break

            all_results.extend(results)
            offset += limit

            if len(results) < limit:
                break

        return all_results

    def load_all(self, period: int | None = None) -> list[dict[str, Any]]:
        """Load all targets with source and stratum info."""
        params = {
            "select": "id,variable,value,target_type,period,notes,source:sources(id,name,institution),stratum:strata(id,name,jurisdiction)",
        }
        if period:
            params["period"] = f"eq.{period}"

        return self._get("targets", params)

    def load_by_institution(
        self,
        institution: str,
        period: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load targets from a specific source institution."""
        sources = self._get("sources", {"institution": f"eq.{institution}"})
        source_ids = [source["id"] for source in sources]

        if not source_ids:
            return []

        params = {
            "select": "id,variable,value,target_type,period,notes,source:sources(id,name,institution),stratum:strata(id,name,jurisdiction)",
            "source_id": f"in.({','.join(source_ids)})",
        }
        if period:
            params["period"] = f"eq.{period}"

        return self._get("targets", params)

    def load_by_period(self, period: int) -> list[dict[str, Any]]:
        """Load targets for a specific year."""
        return self.load_all(period=period)

    def get_cps_column_map(self) -> dict[str, str]:
        """Get the mapping from Supabase variable names to CPS columns."""
        return self.CPS_COLUMN_MAP.copy()

    def _parse_jurisdiction(self, jurisdiction: str) -> str | None:
        """Parse jurisdiction to get the state code when applicable."""
        if jurisdiction in {"us", "us-national"}:
            return None

        if jurisdiction.startswith("us-") and len(jurisdiction) == 5:
            suffix = jurisdiction[3:].lower()
            if suffix in self.STATE_FIPS:
                return self.STATE_FIPS[suffix]
            if suffix in _state_abbr_to_fips(self.STATE_FIPS):
                return suffix

        return None

    def build_calibration_constraints(
        self,
        period: int = 2024,
        include_states: bool = False,
        target_types: list[str] | None = None,
    ) -> dict[str, float]:
        """Build a CPS-column calibration constraint dict from Supabase targets."""
        targets = self.load_all(period=period)
        constraints = {}

        for target in targets:
            variable = target["variable"]
            value = target["value"]
            target_type = target.get("target_type", "amount")
            stratum = target.get("stratum", {})
            jurisdiction = stratum.get("jurisdiction", "us")

            if target_types and target_type not in target_types:
                continue

            cps_col = self.CPS_COLUMN_MAP.get(variable)
            if not cps_col:
                continue

            state = self._parse_jurisdiction(jurisdiction)

            if state and include_states:
                constraints[f"{cps_col}_{state}"] = value
            elif not state and cps_col not in constraints:
                constraints[cps_col] = value

        return constraints

    def get_summary(self) -> dict[str, Any]:
        """Get summary counts for available targets in Supabase."""
        targets = self.load_all()

        by_institution = {}
        by_variable = {}
        by_type = {}

        for target in targets:
            institution = target.get("source", {}).get("institution", "Unknown")
            by_institution[institution] = by_institution.get(institution, 0) + 1

            variable = target["variable"]
            by_variable[variable] = by_variable.get(variable, 0) + 1

            target_type = target.get("target_type", "amount")
            by_type[target_type] = by_type.get(target_type, 0) + 1

        return {
            "total": len(targets),
            "by_institution": by_institution,
            "by_variable": by_variable,
            "by_type": by_type,
        }


class SupabaseTargetProvider(SupabaseTargetLoader):
    """Load Supabase targets as canonical core target specs."""

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        """Load a canonical target set through the core provider protocol."""
        query = query or TargetQuery()
        provider_filters = query.provider_filters
        period = _query_period(query.period)
        institution = provider_filters.get("institution")
        target_types = _as_string_set(provider_filters.get("target_types"))
        include_unsupported = bool(provider_filters.get("include_unsupported", True))
        include_states = bool(provider_filters.get("include_states", True))

        if institution:
            rows = self.load_by_institution(str(institution), period=period)
        else:
            rows = self.load_all(period=period)

        specs: list[TargetSpec] = []
        for row in rows:
            target_type = _target_type(row)
            if target_types and target_type not in target_types:
                continue

            spec = self.target_from_row(row)
            if (
                not include_states
                and spec.metadata.get(US_TARGET_LEVEL_KEY) == TargetLevel.STATE.value
            ):
                continue
            if (
                not include_unsupported
                and not spec.metadata[SUPABASE_SUPPORTED_BY_COLUMN_MAP_KEY]
            ):
                continue
            specs.append(spec)

        return apply_target_query(
            TargetSet(specs),
            TargetQuery(
                period=period if period is not None else query.period,
                entity=query.entity,
                names=query.names,
                metadata_filters=query.metadata_filters,
            ),
        )

    def target_from_row(self, row: dict[str, Any]) -> TargetSpec:
        """Translate one Supabase target row into the canonical target IR."""
        variable = str(row["variable"])
        jurisdiction = _target_jurisdiction(row)
        state_fips, state_abbr = _jurisdiction_state(jurisdiction, self.STATE_FIPS)
        target_type = _target_type(row)
        aggregation = _aggregation_for_target_type(target_type)
        measure = self.CPS_COLUMN_MAP.get(variable, variable)
        supported = variable in self.CPS_COLUMN_MAP
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        source_name = source.get("name") or source.get("institution")
        source_institution = source.get("institution")
        stratum = row.get("stratum") if isinstance(row.get("stratum"), dict) else {}
        category = _category_for_variable(variable)
        level = TargetLevel.STATE if state_fips is not None else TargetLevel.NATIONAL

        filters: list[TargetFilter] = []
        if aggregation is TargetAggregation.COUNT and variable not in _COUNT_ALL_VARIABLES:
            filters.append(
                TargetFilter(
                    feature=measure,
                    operator=FilterOperator.GT,
                    value=0,
                )
            )

        if state_fips is not None:
            filters.append(
                TargetFilter(
                    feature="state_fips",
                    operator=FilterOperator.EQ,
                    value=state_fips,
                )
            )

        metadata: dict[str, Any] = {
            SUPABASE_TARGET_ID_KEY: row.get("id"),
            SUPABASE_VARIABLE_KEY: variable,
            SUPABASE_TARGET_TYPE_KEY: target_type,
            SUPABASE_JURISDICTION_KEY: jurisdiction,
            SUPABASE_STRATUM_NAME_KEY: stratum.get("name"),
            SUPABASE_SOURCE_INSTITUTION_KEY: source_institution,
            SUPABASE_SUPPORTED_BY_COLUMN_MAP_KEY: supported,
            US_TARGET_LEVEL_KEY: level.value,
            US_TARGET_GROUP_KEY: _group_for_category(category),
            US_TARGET_AVAILABLE_KEY: supported,
            US_TARGET_IMPUTATION_KEY: not supported,
        }
        if category is not None:
            metadata[US_TARGET_CATEGORY_KEY] = category.value
        if state_fips is not None:
            metadata["state_fips"] = state_fips
            metadata["state_abbr"] = state_abbr

        return TargetSpec(
            name=_target_name(variable, jurisdiction),
            entity=_entity_for_variable(variable),
            value=float(row["value"]),
            period=int(row["period"]),
            measure=None if aggregation is TargetAggregation.COUNT else measure,
            aggregation=aggregation,
            filters=tuple(filters),
            source=source_name,
            units=_units_for_target_type(target_type),
            description=row.get("notes"),
            metadata=metadata,
        )


def _target_type(row: dict[str, Any]) -> str:
    return str(row.get("target_type") or "amount").lower()


def _aggregation_for_target_type(target_type: str) -> TargetAggregation:
    if target_type == "count":
        return TargetAggregation.COUNT
    if target_type == "mean":
        return TargetAggregation.MEAN
    return TargetAggregation.SUM


def _target_jurisdiction(row: dict[str, Any]) -> str:
    stratum = row.get("stratum") if isinstance(row.get("stratum"), dict) else {}
    return str(stratum.get("jurisdiction") or "us")


def _target_name(variable: str, jurisdiction: str) -> str:
    if jurisdiction in {"us", "us-national"}:
        return variable
    return f"{variable}_{jurisdiction.replace('-', '_')}"


def _query_period(period: int | str | None) -> int | None:
    if isinstance(period, int):
        return period
    if isinstance(period, str) and period.isdigit():
        return int(period)
    return None


def _as_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def _state_abbr_to_fips(state_fips: dict[str, str]) -> dict[str, str]:
    return {abbr: fips for fips, abbr in state_fips.items()}


def _jurisdiction_state(
    jurisdiction: str,
    state_fips: dict[str, str],
) -> tuple[str | None, str | None]:
    if not jurisdiction.startswith("us-") or len(jurisdiction) != 5:
        return None, None

    suffix = jurisdiction[3:].lower()
    if suffix in state_fips:
        return suffix, state_fips[suffix]

    abbr_to_fips = _state_abbr_to_fips(state_fips)
    if suffix in abbr_to_fips:
        return abbr_to_fips[suffix], suffix

    return None, None


def _category_for_variable(variable: str) -> TargetCategory | None:
    if variable in _INCOME_VARIABLES:
        return TargetCategory.INCOME
    if variable in _BENEFIT_VARIABLES:
        return TargetCategory.BENEFITS
    if variable in _HEALTH_VARIABLES:
        return TargetCategory.HEALTH
    if variable.endswith("_tax") or variable.endswith("_credit"):
        return TargetCategory.TAX
    if variable in _COUNT_ALL_VARIABLES:
        return TargetCategory.DEMOGRAPHICS
    return None


def _entity_for_variable(variable: str) -> EntityType:
    if variable in _COUNT_ENTITY_MAP:
        return _COUNT_ENTITY_MAP[variable]
    if variable in _TAX_UNIT_VARIABLES:
        return EntityType.TAX_UNIT
    if variable in _HOUSEHOLD_VARIABLES:
        return EntityType.HOUSEHOLD
    return EntityType.PERSON


def _group_for_category(category: TargetCategory | None) -> str:
    if category is None:
        return "supabase_targets"
    return f"supabase_{category.value}"


def _units_for_target_type(target_type: str) -> str | None:
    return "USD" if target_type == "amount" else None


__all__ = [
    "SUPABASE_JURISDICTION_KEY",
    "SUPABASE_SOURCE_INSTITUTION_KEY",
    "SUPABASE_STRATUM_NAME_KEY",
    "SUPABASE_SUPPORTED_BY_COLUMN_MAP_KEY",
    "SUPABASE_TARGET_ID_KEY",
    "SUPABASE_TARGET_TYPE_KEY",
    "SUPABASE_VARIABLE_KEY",
    "SupabaseTargetLoader",
    "SupabaseTargetProvider",
]
