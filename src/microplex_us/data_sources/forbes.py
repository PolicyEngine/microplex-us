"""Forbes fixed-spine source support for Microplex top-tail units."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from microplex.core import EntityType
from microplex.targets import TargetAggregation, TargetSet, TargetSpec

from microplex_us.policyengine.us import (
    DEFAULT_POLICYENGINE_US_VARIABLE_BINDINGS,
    PolicyEngineUSEntityTableBundle,
    PolicyEngineUSVariableBinding,
    compile_supported_policyengine_us_household_linear_constraints,
)

FORBES_HOUSEHOLD_VARIABLES: tuple[str, ...] = (
    "state_fips",
    "net_worth",
)

FORBES_PERSON_VARIABLES: tuple[str, ...] = (
    "age",
    "is_female",
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
    "partnership_s_corp_income",
    "partnership_se_income",
    "estate_income",
    "farm_income",
    "rental_income",
)

FORBES_SOURCE_METADATA_COLUMNS: tuple[str, ...] = (
    "forbes_unit_id",
    "forbes_name",
    "forbes_rank",
    "forbes_snapshot_id",
    "replicate_index",
    "replicate_count",
    "replicate_weight",
    "household_id",
    "person_id",
    "tax_unit_id",
    "spm_unit_id",
    "family_id",
    "marital_unit_id",
)


@dataclass(frozen=True)
class ForbesFixedSpineConfig:
    """Controls deterministic Forbes fixed-spine construction."""

    period: int = 2024
    snapshot_id: str = "forbes-us-top-tail"
    replicates_per_unit: int = 10
    default_unit_weight: float = 1.0
    household_id_start: int = 90_000_000_000
    person_id_start: int = 90_000_000_000
    tax_unit_id_start: int = 90_000_000_000
    spm_unit_id_start: int = 90_000_000_000
    family_id_start: int = 90_000_000_000
    marital_unit_id_start: int = 90_000_000_000
    unit_id_column: str = "forbes_unit_id"
    name_column: str = "name"
    rank_column: str = "rank"
    unit_weight_column: str = "weight"

    def __post_init__(self) -> None:
        if self.replicates_per_unit < 1:
            raise ValueError("replicates_per_unit must be at least 1")
        if self.default_unit_weight <= 0:
            raise ValueError("default_unit_weight must be positive")


@dataclass(frozen=True)
class ForbesFixedSpine:
    """Constructed fixed spine plus source-owned diagnostic metadata."""

    tables: PolicyEngineUSEntityTableBundle
    record_metadata: pd.DataFrame
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class FixedSpineTargetContribution:
    """Fixed-spine contribution to one target."""

    target_name: str
    target_value: float
    contribution: float
    residual_value: float
    status: str
    reason: str | None = None
    clamped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FixedSpineResidualizationResult:
    """Residualized targets and diagnostic contribution records."""

    targets: TargetSet
    contributions: tuple[FixedSpineTargetContribution, ...]

    def diagnostics(self) -> list[dict[str, Any]]:
        return [contribution.to_dict() for contribution in self.contributions]


def read_forbes_fixed_spine_records(path: str | Path) -> pd.DataFrame:
    """Read normalized Forbes spine records from JSON, JSONL, or CSV."""

    input_path = Path(path)
    suffixes = tuple(suffix.lower() for suffix in input_path.suffixes)
    if suffixes[-1:] == (".csv",):
        return pd.read_csv(input_path)
    if suffixes[-1:] == (".jsonl",):
        return pd.read_json(input_path, lines=True)
    if suffixes[-1:] == (".json",):
        return pd.read_json(input_path)
    raise ValueError(
        f"Forbes fixed-spine records must be .csv, .json, or .jsonl; got {input_path}"
    )


def build_forbes_fixed_spine(
    records: pd.DataFrame | str | Path,
    *,
    config: ForbesFixedSpineConfig | None = None,
    source_path: str | Path | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> ForbesFixedSpine:
    """Build deterministic PolicyEngine entity tables for Forbes fixed units.

    The returned entity tables contain only model-facing variables. Source fields
    such as names, ranks, and replicate provenance are kept in ``record_metadata``
    and ``source_metadata`` so they can be audited without entering the exported
    PolicyEngine dataset.
    """

    resolved_config = config or ForbesFixedSpineConfig()
    if isinstance(records, (str, Path)):
        source_path = records if source_path is None else source_path
        records = read_forbes_fixed_spine_records(records)
    normalized = _normalize_forbes_records(records, resolved_config)

    household_rows: list[dict[str, Any]] = []
    person_rows: list[dict[str, Any]] = []
    tax_unit_rows: list[dict[str, Any]] = []
    spm_unit_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    marital_unit_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for unit_position, (_, unit) in enumerate(normalized.iterrows()):
        unit_weight = _numeric_or_default(
            unit.get(resolved_config.unit_weight_column),
            resolved_config.default_unit_weight,
        )
        replicate_weight = unit_weight / float(resolved_config.replicates_per_unit)
        for replicate_index in range(resolved_config.replicates_per_unit):
            row_index = (
                unit_position * resolved_config.replicates_per_unit + replicate_index
            )
            household_id = resolved_config.household_id_start + row_index
            person_id = resolved_config.person_id_start + row_index
            tax_unit_id = resolved_config.tax_unit_id_start + row_index
            spm_unit_id = resolved_config.spm_unit_id_start + row_index
            family_id = resolved_config.family_id_start + row_index
            marital_unit_id = resolved_config.marital_unit_id_start + row_index
            state_fips = _normalize_state_fips(unit.get("state_fips"))

            household = {
                "household_id": household_id,
                "household_weight": replicate_weight,
                "state_fips": state_fips,
            }
            for variable in FORBES_HOUSEHOLD_VARIABLES:
                if variable == "state_fips":
                    continue
                if variable in unit:
                    household[variable] = _numeric_or_default(unit.get(variable), 0.0)
            household_rows.append(household)

            person = {
                "person_id": person_id,
                "household_id": household_id,
                "tax_unit_id": tax_unit_id,
                "spm_unit_id": spm_unit_id,
                "family_id": family_id,
                "marital_unit_id": marital_unit_id,
                "state_fips": state_fips,
            }
            for variable in FORBES_PERSON_VARIABLES:
                if variable == "state_fips":
                    continue
                if variable in unit:
                    person[variable] = _numeric_or_default(unit.get(variable), 0.0)
            person_rows.append(person)

            tax_unit_rows.append(
                {
                    "tax_unit_id": tax_unit_id,
                    "household_id": household_id,
                    "state_fips": state_fips,
                }
            )
            spm_unit_rows.append(
                {
                    "spm_unit_id": spm_unit_id,
                    "household_id": household_id,
                    "state_fips": state_fips,
                }
            )
            family_rows.append(
                {
                    "family_id": family_id,
                    "household_id": household_id,
                    "state_fips": state_fips,
                }
            )
            marital_unit_rows.append(
                {
                    "marital_unit_id": marital_unit_id,
                    "household_id": household_id,
                    "state_fips": state_fips,
                }
            )
            metadata_rows.append(
                {
                    "forbes_unit_id": unit[resolved_config.unit_id_column],
                    "forbes_name": unit.get(resolved_config.name_column),
                    "forbes_rank": _optional_int(unit.get(resolved_config.rank_column)),
                    "forbes_snapshot_id": resolved_config.snapshot_id,
                    "replicate_index": replicate_index,
                    "replicate_count": resolved_config.replicates_per_unit,
                    "replicate_weight": replicate_weight,
                    "household_id": household_id,
                    "person_id": person_id,
                    "tax_unit_id": tax_unit_id,
                    "spm_unit_id": spm_unit_id,
                    "family_id": family_id,
                    "marital_unit_id": marital_unit_id,
                }
            )

    tables = PolicyEngineUSEntityTableBundle(
        households=pd.DataFrame(household_rows),
        persons=pd.DataFrame(person_rows),
        tax_units=pd.DataFrame(tax_unit_rows),
        spm_units=pd.DataFrame(spm_unit_rows),
        families=pd.DataFrame(family_rows),
        marital_units=pd.DataFrame(marital_unit_rows),
    )
    metadata = {
        **dict(source_metadata or {}),
        "source": "forbes_fixed_spine",
        "snapshot_id": resolved_config.snapshot_id,
        "period": resolved_config.period,
        "unit_count": int(len(normalized)),
        "replicates_per_unit": int(resolved_config.replicates_per_unit),
        "record_count": int(len(household_rows)),
        "source_path": str(source_path) if source_path is not None else None,
        "source_sha256": _sha256_file(source_path) if source_path is not None else None,
    }
    return ForbesFixedSpine(
        tables=tables,
        record_metadata=pd.DataFrame(
            metadata_rows, columns=FORBES_SOURCE_METADATA_COLUMNS
        ),
        source_metadata=metadata,
    )


def forbes_fixed_spine_variable_bindings(
    tables: PolicyEngineUSEntityTableBundle,
) -> dict[str, PolicyEngineUSVariableBinding]:
    """Return variable bindings for scoring fixed-spine target contributions."""

    bindings = dict(DEFAULT_POLICYENGINE_US_VARIABLE_BINDINGS)
    household_columns = set(tables.households.columns)
    person_columns = set(tables.persons.columns if tables.persons is not None else ())

    for variable in FORBES_HOUSEHOLD_VARIABLES:
        if variable in household_columns:
            bindings[variable] = PolicyEngineUSVariableBinding(
                entity=EntityType.HOUSEHOLD,
                column=variable,
            )
    for variable in FORBES_PERSON_VARIABLES:
        if variable in person_columns and variable not in bindings:
            bindings[variable] = PolicyEngineUSVariableBinding(
                entity=EntityType.PERSON,
                column=variable,
            )
    return bindings


def residualize_targets_for_fixed_spine(
    targets: TargetSet | Iterable[TargetSpec],
    fixed_spine_tables: PolicyEngineUSEntityTableBundle,
    *,
    variable_bindings: Mapping[str, PolicyEngineUSVariableBinding] | None = None,
    clamp_negative_residuals: bool = True,
) -> FixedSpineResidualizationResult:
    """Subtract fixed-spine contributions from additive calibration targets."""

    target_list = list(targets.targets if isinstance(targets, TargetSet) else targets)
    bindings = {
        **forbes_fixed_spine_variable_bindings(fixed_spine_tables),
        **dict(variable_bindings or {}),
    }
    residualized_targets: list[TargetSpec] = []
    contributions: list[FixedSpineTargetContribution] = []

    for target in target_list:
        contribution = _fixed_spine_target_contribution(
            target,
            fixed_spine_tables,
            variable_bindings=bindings,
        )
        if contribution.status != "supported":
            residualized_targets.append(target)
            contributions.append(contribution)
            continue

        residual_value = float(target.value) - contribution.contribution
        clamped = False
        if clamp_negative_residuals and residual_value < 0.0:
            residual_value = 0.0
            clamped = True
        metadata = dict(target.metadata)
        metadata["fixed_spine_residualization"] = {
            "original_value": float(target.value),
            "fixed_spine_contribution": contribution.contribution,
            "residual_value": residual_value,
            "clamped": clamped,
        }
        residualized_targets.append(
            replace(target, value=residual_value, metadata=metadata)
        )
        contributions.append(
            replace(contribution, residual_value=residual_value, clamped=clamped)
        )

    return FixedSpineResidualizationResult(
        targets=TargetSet(residualized_targets),
        contributions=tuple(contributions),
    )


def fixed_spine_contribution_diagnostics_json(
    result: FixedSpineResidualizationResult,
) -> str:
    """Serialize fixed-spine residualization diagnostics for artifact manifests."""

    return json.dumps(result.diagnostics(), indent=2, sort_keys=True)


def _fixed_spine_target_contribution(
    target: TargetSpec,
    tables: PolicyEngineUSEntityTableBundle,
    *,
    variable_bindings: Mapping[str, PolicyEngineUSVariableBinding],
) -> FixedSpineTargetContribution:
    if target.aggregation is TargetAggregation.MEAN:
        return FixedSpineTargetContribution(
            target_name=target.name,
            target_value=float(target.value),
            contribution=0.0,
            residual_value=float(target.value),
            status="unsupported",
            reason="mean targets are not additive residualization targets",
        )

    weights = _household_weights(tables)
    try:
        supported, unsupported, constraints = (
            compile_supported_policyengine_us_household_linear_constraints(
                [target],
                tables,
                variable_bindings=dict(variable_bindings),
            )
        )
    except (KeyError, ValueError) as error:
        return FixedSpineTargetContribution(
            target_name=target.name,
            target_value=float(target.value),
            contribution=0.0,
            residual_value=float(target.value),
            status="unsupported",
            reason=str(error),
        )

    if unsupported or not supported or not constraints:
        return FixedSpineTargetContribution(
            target_name=target.name,
            target_value=float(target.value),
            contribution=0.0,
            residual_value=float(target.value),
            status="unsupported",
            reason="target cannot be compiled against fixed-spine entity tables",
        )

    contribution = float(np.dot(weights, constraints[0].coefficients))
    return FixedSpineTargetContribution(
        target_name=target.name,
        target_value=float(target.value),
        contribution=contribution,
        residual_value=float(target.value) - contribution,
        status="supported",
    )


def _normalize_forbes_records(
    records: pd.DataFrame,
    config: ForbesFixedSpineConfig,
) -> pd.DataFrame:
    result = records.copy()
    if result.empty:
        raise ValueError("Forbes fixed spine requires at least one source record")
    if "net_worth" not in result.columns:
        raise ValueError("Forbes fixed spine records must include 'net_worth'")
    if config.unit_id_column not in result.columns:
        result[config.unit_id_column] = np.arange(1, len(result) + 1)
    if config.name_column not in result.columns:
        result[config.name_column] = result[config.unit_id_column].astype(str)
    if config.rank_column not in result.columns:
        result[config.rank_column] = np.arange(1, len(result) + 1)
    if config.unit_weight_column not in result.columns:
        result[config.unit_weight_column] = config.default_unit_weight
    if "state_fips" not in result.columns:
        result["state_fips"] = None
    return result


def _household_weights(tables: PolicyEngineUSEntityTableBundle) -> np.ndarray:
    households = tables.households
    if "household_weight" in households.columns:
        return households["household_weight"].to_numpy(dtype=float, copy=False)
    if "weight" in households.columns:
        return households["weight"].to_numpy(dtype=float, copy=False)
    raise ValueError("Fixed-spine households must include household_weight or weight")


def _normalize_state_fips(value: Any) -> str:
    if value is None or pd.isna(value):
        return "00"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "00"
        numeric = pd.to_numeric(pd.Series([stripped]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return stripped
        return f"{int(numeric):02d}"
    return f"{int(value):02d}"


def _numeric_or_default(value: Any, default: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _optional_int(value: Any) -> int | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(numeric)


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
