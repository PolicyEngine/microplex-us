from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from microplex_us.targets import (
    ACA_AVERAGE_MONTHLY_APTC_CONCEPT,
    ACA_MARKETPLACE_EFFECTUATED_ENROLLMENT_CONCEPT,
    ACAPTCMultiplierInput,
    aca_ptc_multiplier_inputs_from_arch_consumer_facts,
    build_aca_ptc_multiplier_rows,
    load_arch_consumer_fact_jsonl_rows,
    write_policyengine_aca_ptc_multiplier_csv,
)
from microplex_us.targets.aca_ptc import main


def test_build_aca_ptc_multiplier_rows_matches_policyengine_formula() -> None:
    rows = build_aca_ptc_multiplier_rows(
        [
            ACAPTCMultiplierInput(
                state="California",
                enroll_base=1_701_375,
                enroll_target=1_795_695,
                aptc_base=459,
                aptc_target=526,
            )
        ]
    )

    row = rows[0]
    assert row.vol_mult == pytest.approx(1_795_695 / 1_701_375)
    assert row.val_mult == pytest.approx(526 / 459)
    assert row.amount_mult == pytest.approx(
        (1_795_695 / 1_701_375) * (526 / 459)
    )
    assert row.target_factors() == {
        "tax_unit_count": pytest.approx(1_795_695 / 1_701_375),
        "aca_ptc": pytest.approx((1_795_695 / 1_701_375) * (526 / 459)),
    }


def test_arch_consumer_fact_inputs_use_oep_with_effectuated_fallback() -> None:
    facts = [
        _enrollment_fact("California", "ca", 2022, 1_701_375),
        _enrollment_fact("California", "ca", 2024, 1_795_695),
        _oep_aptc_fact("California", "ca", 2022, 459),
        _effectuated_aptc_fact("California", "ca", 2022, 469.44),
        _oep_aptc_fact("California", "ca", 2024, 526),
        _enrollment_fact("Nevada", "nv", 2022, 90_397),
        _enrollment_fact("Nevada", "nv", 2024, 92_949),
        _effectuated_aptc_fact("Nevada", "nv", 2022, 429.75),
        _oep_aptc_fact("Nevada", "nv", 2024, 438),
    ]

    inputs = aca_ptc_multiplier_inputs_from_arch_consumer_facts(facts)

    by_state = {item.state: item for item in inputs}
    assert by_state["California"].aptc_base == 459
    assert by_state["California"].aptc_base_source_kind == "oep"
    assert by_state["Nevada"].aptc_base == 429.75
    assert by_state["Nevada"].aptc_base_source_kind == "effectuated"

    rows = build_aca_ptc_multiplier_rows(inputs)
    nevada = {row.state: row for row in rows}["Nevada"]
    assert nevada.vol_mult == pytest.approx(92_949 / 90_397)
    assert nevada.val_mult == pytest.approx(438 / 429.75)
    assert nevada.val_mult != pytest.approx(438 / 435)


def test_arch_consumer_fact_inputs_can_require_oep_base_aptc() -> None:
    facts = [
        _enrollment_fact("Nevada", "nv", 2022, 90_397),
        _enrollment_fact("Nevada", "nv", 2024, 92_949),
        _effectuated_aptc_fact("Nevada", "nv", 2022, 429.75),
        _oep_aptc_fact("Nevada", "nv", 2024, 438),
    ]

    with pytest.raises(ValueError, match="Nevada 2022 average APTC"):
        aca_ptc_multiplier_inputs_from_arch_consumer_facts(
            facts,
            base_aptc_policy="oep",
        )


def test_write_policyengine_aca_ptc_multiplier_csv(tmp_path: Path) -> None:
    rows = build_aca_ptc_multiplier_rows(
        [
            ACAPTCMultiplierInput(
                state="Nevada",
                enroll_base=90_397,
                enroll_target=92_949,
                aptc_base=429.75,
                aptc_target=438,
            )
        ]
    )
    path = tmp_path / "aca_ptc_multipliers_2022_2024.csv"

    write_policyengine_aca_ptc_multiplier_csv(rows, path)

    with path.open() as file:
        records = list(csv.DictReader(file))
    assert records[0]["state"] == "Nevada"
    assert records[0]["enroll_2022"] == "90397"
    assert records[0]["aptc_2024"] == "438"
    assert float(records[0]["enroll_2022"]) == 90_397
    assert float(records[0]["aptc_2022"]) == 429.75
    assert float(records[0]["vol_mult"]) == pytest.approx(92_949 / 90_397)
    assert float(records[0]["val_mult"]) == pytest.approx(438 / 429.75)


def test_main_builds_policyengine_csv_from_consumer_fact_jsonl(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    consumer_facts = tmp_path / "consumer_facts.jsonl"
    _write_jsonl(
        consumer_facts,
        [
            _enrollment_fact("Nevada", "nv", 2022, 90_397),
            _enrollment_fact("Nevada", "nv", 2024, 92_949),
            _effectuated_aptc_fact("Nevada", "nv", 2022, 429.75),
            _oep_aptc_fact("Nevada", "nv", 2024, 438),
        ],
    )
    out = tmp_path / "aca_ptc_multipliers_2022_2024.csv"

    assert main([str(consumer_facts), "--out", str(out)]) == 0

    captured = capsys.readouterr()
    assert f"Wrote 1 ACA PTC multiplier rows to {out}" in captured.out
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["state"] == "Nevada"
    assert rows[0]["aptc_2022"] == "429.75"


def test_load_arch_consumer_fact_jsonl_rows_rejects_non_consumer_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "facts.jsonl"
    path.write_text(json.dumps({"schema_version": "arch.fact.v1"}) + "\n")

    with pytest.raises(ValueError, match="Unsupported Arch consumer fact schema"):
        load_arch_consumer_fact_jsonl_rows([path])


def _enrollment_fact(
    state: str,
    state_abbr: str,
    year: int,
    value: float,
) -> dict[str, Any]:
    return _fact(
        state=state,
        period=year,
        value=value,
        concept=ACA_MARKETPLACE_EFFECTUATED_ENROLLMENT_CONCEPT,
        source_record_id=(
            f"kff.marketplace_effectuated_enrollment.{year}.state."
            f"{state_abbr}.total_effectuated_marketplace_enrollment"
        ),
    )


def _oep_aptc_fact(
    state: str,
    state_abbr: str,
    year: int,
    value: float,
) -> dict[str, Any]:
    return _fact(
        state=state,
        period=year,
        value=value,
        concept=ACA_AVERAGE_MONTHLY_APTC_CONCEPT,
        source_record_id=(
            f"cms_aca.oep{year}.state_marketplace."
            f"{state_abbr}.average_monthly_aptc"
        ),
    )


def _effectuated_aptc_fact(
    state: str,
    state_abbr: str,
    year: int,
    value: float,
) -> dict[str, Any]:
    return _fact(
        state=state,
        period=year,
        value=value,
        concept=ACA_AVERAGE_MONTHLY_APTC_CONCEPT,
        source_record_id=(
            f"cms_aca.effectuated_enrollment.{year}.state_marketplace."
            f"{state_abbr}.average_monthly_aptc"
        ),
    )


def _fact(
    *,
    state: str,
    period: int,
    value: float,
    concept: str,
    source_record_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "arch.consumer_fact.v1",
        "period": {"type": "calendar_year", "value": period},
        "geography": {"level": "state", "name": state},
        "observed_measure": {"source_concept": concept},
        "lineage": {"source_record_id": source_record_id},
        "value": value,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
