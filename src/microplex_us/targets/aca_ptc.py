"""ACA PTC target-construction helpers for US target sources."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from microplex.targets import (
    arch_consumer_fact_concept,
    arch_consumer_fact_numeric_value,
    arch_consumer_fact_period,
    arch_consumer_fact_source_record_id,
    load_arch_consumer_fact_jsonl_rows,
)

ACAPTCBaseAPTCPolicy = Literal[
    "oep",
    "effectuated",
    "oep_with_effectuated_fallback",
]

ACA_MARKETPLACE_EFFECTUATED_ENROLLMENT_CONCEPT = (
    "cms_aca.marketplace_effectuated_enrollment"
)
ACA_AVERAGE_MONTHLY_APTC_CONCEPT = "cms_aca.average_monthly_aptc"


@dataclass(frozen=True)
class ACAPTCMultiplierInput:
    """Publisher-source inputs for one state's ACA PTC multiplier row."""

    state: str
    enroll_base: float
    enroll_target: float
    aptc_base: float
    aptc_target: float
    base_year: int = 2022
    target_year: int = 2024
    enroll_base_source_record_id: str | None = None
    enroll_target_source_record_id: str | None = None
    aptc_base_source_record_id: str | None = None
    aptc_target_source_record_id: str | None = None
    aptc_base_source_kind: str | None = None
    aptc_target_source_kind: str | None = None


@dataclass(frozen=True)
class ACAPTCMultiplierRow:
    """PE-compatible ACA PTC multiplier row for one state."""

    state: str
    enroll_base: float
    enroll_target: float
    vol_mult: float
    aptc_base: float
    aptc_target: float
    val_mult: float
    base_year: int = 2022
    target_year: int = 2024
    enroll_base_source_record_id: str | None = None
    enroll_target_source_record_id: str | None = None
    aptc_base_source_record_id: str | None = None
    aptc_target_source_record_id: str | None = None
    aptc_base_source_kind: str | None = None
    aptc_target_source_kind: str | None = None

    @property
    def amount_mult(self) -> float:
        """Multiplier PE applies to the ACA PTC amount target."""

        return self.vol_mult * self.val_mult

    def target_factors(self) -> dict[str, float]:
        """Return the variable factors consumed by PE's state uprating path."""

        return {
            "tax_unit_count": self.vol_mult,
            "aca_ptc": self.amount_mult,
        }

    def to_policyengine_csv_row(self) -> dict[str, float | int | str]:
        """Return a row with PE's incumbent ACA multiplier CSV column names."""

        return {
            "state": self.state,
            f"enroll_{self.base_year}": _source_csv_number(self.enroll_base),
            f"enroll_{self.target_year}": _source_csv_number(self.enroll_target),
            "vol_mult": self.vol_mult,
            f"aptc_{self.base_year}": _source_csv_number(self.aptc_base),
            f"aptc_{self.target_year}": _source_csv_number(self.aptc_target),
            "val_mult": self.val_mult,
        }


@dataclass(frozen=True)
class _ACAStateFact:
    state: str
    period: int
    value: float
    concept: str
    source_record_id: str | None
    source_kind: str | None


def build_aca_ptc_multiplier_rows(
    inputs: Iterable[ACAPTCMultiplierInput],
) -> tuple[ACAPTCMultiplierRow, ...]:
    """Build state ACA PTC multiplier rows from explicit source inputs."""

    rows = []
    for item in inputs:
        _validate_positive_source_value(item.enroll_base, "enroll_base", item.state)
        _validate_positive_source_value(item.enroll_target, "enroll_target", item.state)
        _validate_positive_source_value(item.aptc_base, "aptc_base", item.state)
        _validate_positive_source_value(item.aptc_target, "aptc_target", item.state)
        rows.append(
            ACAPTCMultiplierRow(
                state=item.state,
                base_year=item.base_year,
                target_year=item.target_year,
                enroll_base=item.enroll_base,
                enroll_target=item.enroll_target,
                vol_mult=item.enroll_target / item.enroll_base,
                aptc_base=item.aptc_base,
                aptc_target=item.aptc_target,
                val_mult=item.aptc_target / item.aptc_base,
                enroll_base_source_record_id=item.enroll_base_source_record_id,
                enroll_target_source_record_id=item.enroll_target_source_record_id,
                aptc_base_source_record_id=item.aptc_base_source_record_id,
                aptc_target_source_record_id=item.aptc_target_source_record_id,
                aptc_base_source_kind=item.aptc_base_source_kind,
                aptc_target_source_kind=item.aptc_target_source_kind,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.state))


def aca_ptc_multiplier_inputs_from_arch_consumer_facts(
    rows: Iterable[Mapping[str, Any]],
    *,
    base_year: int = 2022,
    target_year: int = 2024,
    base_aptc_policy: ACAPTCBaseAPTCPolicy = "oep_with_effectuated_fallback",
) -> tuple[ACAPTCMultiplierInput, ...]:
    """Collect PE-style ACA PTC multiplier inputs from Arch consumer facts.

    The publisher-source recipe uses KFF full-year effectuated enrollment for
    the volume ratio, CMS OEP average APTC where available for the base-year
    value ratio base, CMS full-year 2022 APTC as the fallback for missing OEP
    state values, and CMS OEP average APTC for the target-year value ratio.
    """

    enrollment: dict[tuple[int, str], _ACAStateFact] = {}
    oep_aptc: dict[tuple[int, str], _ACAStateFact] = {}
    effectuated_aptc: dict[tuple[int, str], _ACAStateFact] = {}

    for row in rows:
        fact = _aca_state_fact_from_arch_consumer_fact(row)
        if fact is None:
            continue
        key = (fact.period, fact.state)
        if fact.concept == ACA_MARKETPLACE_EFFECTUATED_ENROLLMENT_CONCEPT:
            enrollment[key] = fact
        elif fact.concept == ACA_AVERAGE_MONTHLY_APTC_CONCEPT:
            if fact.source_kind == "oep":
                oep_aptc[key] = fact
            elif fact.source_kind == "effectuated":
                effectuated_aptc[key] = fact

    states = sorted(
        {
            state
            for period, state in enrollment
            if period == base_year and (target_year, state) in enrollment
        }
    )
    inputs = []
    missing: list[str] = []
    for state in states:
        enroll_base = enrollment[(base_year, state)]
        enroll_target = enrollment[(target_year, state)]
        aptc_base = _select_base_aptc_fact(
            state,
            base_year=base_year,
            policy=base_aptc_policy,
            oep_aptc=oep_aptc,
            effectuated_aptc=effectuated_aptc,
        )
        aptc_target = oep_aptc.get((target_year, state))
        if aptc_base is None:
            missing.append(f"{state} {base_year} average APTC")
            continue
        if aptc_target is None:
            missing.append(f"{state} {target_year} OEP average APTC")
            continue
        inputs.append(
            ACAPTCMultiplierInput(
                state=state,
                base_year=base_year,
                target_year=target_year,
                enroll_base=enroll_base.value,
                enroll_target=enroll_target.value,
                aptc_base=aptc_base.value,
                aptc_target=aptc_target.value,
                enroll_base_source_record_id=enroll_base.source_record_id,
                enroll_target_source_record_id=enroll_target.source_record_id,
                aptc_base_source_record_id=aptc_base.source_record_id,
                aptc_target_source_record_id=aptc_target.source_record_id,
                aptc_base_source_kind=aptc_base.source_kind,
                aptc_target_source_kind=aptc_target.source_kind,
            )
        )

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        raise ValueError(f"Missing ACA PTC source facts: {preview}{suffix}")
    return tuple(inputs)


def write_policyengine_aca_ptc_multiplier_csv(
    rows: Iterable[ACAPTCMultiplierRow],
    path: str | Path,
) -> None:
    """Write PE-compatible ACA PTC multiplier rows."""

    rows = tuple(rows)
    if not rows:
        raise ValueError("Cannot write ACA PTC multiplier CSV with no rows.")
    year_pairs = {(row.base_year, row.target_year) for row in rows}
    if len(year_pairs) != 1:
        raise ValueError("ACA PTC multiplier CSV rows must use one year pair.")
    base_year, target_year = next(iter(year_pairs))
    fieldnames = [
        "state",
        f"enroll_{base_year}",
        f"enroll_{target_year}",
        "vol_mult",
        f"aptc_{base_year}",
        f"aptc_{target_year}",
        "val_mult",
    ]
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_policyengine_csv_row())


def main(argv: list[str] | None = None) -> int:
    """Build a PE-compatible ACA PTC multiplier CSV from Arch consumer facts."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a PE-compatible ACA PTC multiplier CSV from Arch "
            "consumer_facts.jsonl files."
        )
    )
    parser.add_argument(
        "consumer_facts",
        nargs="+",
        help="Arch consumer_facts.jsonl path(s) containing ACA source facts.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2022,
        help="Source year for the multiplier denominator.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=2024,
        help="Target year for the multiplier numerator.",
    )
    parser.add_argument(
        "--base-aptc-policy",
        choices=("oep", "effectuated", "oep_with_effectuated_fallback"),
        default="oep_with_effectuated_fallback",
        help="Source selection policy for base-year average monthly APTC.",
    )
    args = parser.parse_args(argv)

    consumer_fact_rows = load_arch_consumer_fact_jsonl_rows(args.consumer_facts)
    inputs = aca_ptc_multiplier_inputs_from_arch_consumer_facts(
        consumer_fact_rows,
        base_year=args.base_year,
        target_year=args.target_year,
        base_aptc_policy=args.base_aptc_policy,
    )
    rows = build_aca_ptc_multiplier_rows(inputs)
    write_policyengine_aca_ptc_multiplier_csv(rows, args.out)
    print(f"Wrote {len(rows)} ACA PTC multiplier rows to {args.out}")
    return 0


def _select_base_aptc_fact(
    state: str,
    *,
    base_year: int,
    policy: ACAPTCBaseAPTCPolicy,
    oep_aptc: Mapping[tuple[int, str], _ACAStateFact],
    effectuated_aptc: Mapping[tuple[int, str], _ACAStateFact],
) -> _ACAStateFact | None:
    key = (base_year, state)
    if policy == "oep":
        return oep_aptc.get(key)
    if policy == "effectuated":
        return effectuated_aptc.get(key)
    if policy == "oep_with_effectuated_fallback":
        return oep_aptc.get(key) or effectuated_aptc.get(key)
    raise ValueError(f"Unsupported ACA PTC base APTC policy: {policy}")


def _aca_state_fact_from_arch_consumer_fact(
    row: Mapping[str, Any],
) -> _ACAStateFact | None:
    concept = _arch_consumer_fact_concept(row)
    if concept not in {
        ACA_MARKETPLACE_EFFECTUATED_ENROLLMENT_CONCEPT,
        ACA_AVERAGE_MONTHLY_APTC_CONCEPT,
    }:
        return None
    geography = _mapping(row.get("geography"))
    if str(geography.get("level") or "").lower() != "state":
        return None
    state = _arch_consumer_fact_state(row, geography)
    if not state:
        return None
    return _ACAStateFact(
        state=state,
        period=_arch_consumer_fact_period(row),
        value=_json_numeric_value(row.get("value")),
        concept=concept,
        source_record_id=_arch_consumer_fact_source_record_id(row),
        source_kind=_aca_source_kind(row),
    )


def _arch_consumer_fact_concept(row: Mapping[str, Any]) -> str | None:
    return arch_consumer_fact_concept(row)


def _arch_consumer_fact_period(row: Mapping[str, Any]) -> int:
    return arch_consumer_fact_period(row)


def _arch_consumer_fact_state(
    row: Mapping[str, Any],
    geography: Mapping[str, Any],
) -> str | None:
    name = geography.get("name")
    if name:
        return str(name)
    source_record_id = _arch_consumer_fact_source_record_id(row) or ""
    for token in source_record_id.split("."):
        state = _STATE_ABBR_TO_NAME.get(token.lower())
        if state is not None:
            return state
    return None


def _arch_consumer_fact_source_record_id(row: Mapping[str, Any]) -> str | None:
    source_record_id = arch_consumer_fact_source_record_id(row)
    if source_record_id is not None:
        return source_record_id
    fallback = row.get("source_record_id")
    return str(fallback) if fallback else None


def _aca_source_kind(row: Mapping[str, Any]) -> str | None:
    source_record_id = (_arch_consumer_fact_source_record_id(row) or "").lower()
    if ".oep" in source_record_id:
        return "oep"
    if ".effectuated_enrollment." in source_record_id:
        return "effectuated"
    source = _mapping(row.get("source"))
    source_table = str(source.get("source_table") or "").lower()
    if "open enrollment" in source_table or "oep" in source_table:
        return "oep"
    if "effectuated enrollment" in source_table:
        return "effectuated"
    return None


def _validate_positive_source_value(value: float, label: str, state: str) -> None:
    if value <= 0:
        raise ValueError(f"{state} {label} must be positive; got {value}.")


def _json_numeric_value(value: Any) -> float:
    return arch_consumer_fact_numeric_value(value)


def _source_csv_number(value: float) -> float | int:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


_STATE_ABBR_TO_NAME = {
    "ak": "Alaska",
    "al": "Alabama",
    "ar": "Arkansas",
    "az": "Arizona",
    "ca": "California",
    "co": "Colorado",
    "ct": "Connecticut",
    "dc": "District of Columbia",
    "de": "Delaware",
    "fl": "Florida",
    "ga": "Georgia",
    "hi": "Hawaii",
    "ia": "Iowa",
    "id": "Idaho",
    "il": "Illinois",
    "in": "Indiana",
    "ks": "Kansas",
    "ky": "Kentucky",
    "la": "Louisiana",
    "ma": "Massachusetts",
    "md": "Maryland",
    "me": "Maine",
    "mi": "Michigan",
    "mn": "Minnesota",
    "mo": "Missouri",
    "ms": "Mississippi",
    "mt": "Montana",
    "nc": "North Carolina",
    "nd": "North Dakota",
    "ne": "Nebraska",
    "nh": "New Hampshire",
    "nj": "New Jersey",
    "nm": "New Mexico",
    "nv": "Nevada",
    "ny": "New York",
    "oh": "Ohio",
    "ok": "Oklahoma",
    "or": "Oregon",
    "pa": "Pennsylvania",
    "ri": "Rhode Island",
    "sc": "South Carolina",
    "sd": "South Dakota",
    "tn": "Tennessee",
    "tx": "Texas",
    "ut": "Utah",
    "va": "Virginia",
    "vt": "Vermont",
    "wa": "Washington",
    "wi": "Wisconsin",
    "wv": "West Virginia",
    "wy": "Wyoming",
}
