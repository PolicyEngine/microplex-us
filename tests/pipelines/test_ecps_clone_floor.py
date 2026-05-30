"""Tests for the clone-floor baseline gate.

The gate is the CI decision point that refuses to benchmark a candidate against
a degraded or unverifiable baseline Enhanced CPS. These tests pin its behaviour
with tiny fixtures: a healthy share passes, a degraded share fails loudly, and a
missing / malformed sidecar fails closed.
"""

from __future__ import annotations

import json
from pathlib import Path

from microplex_us.pipelines.ecps_clone_floor import (
    CloneFloorGateResult,
    evaluate_clone_floor_gate,
    extract_clone_weight_share,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


# --------------------------------------------------------------------------- #
# Happy / degraded paths
# --------------------------------------------------------------------------- #
def test_gate_passes_above_floor(tmp_path: Path) -> None:
    """A healthy 10% clone share passes the default 5% floor."""
    p = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    _write(p, {"clone_household_weight_share": 0.10})
    result = evaluate_clone_floor_gate(p)
    assert isinstance(result, CloneFloorGateResult)
    assert result.passed is True
    assert result.clone_weight_share == 0.10
    assert "healthy" in result.message


def test_gate_fails_below_floor_loudly(tmp_path: Path) -> None:
    """A degraded 2% clone share fails loudly with a specific message."""
    p = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    _write(p, {"clone_household_weight_share": 0.02})
    result = evaluate_clone_floor_gate(p)
    assert result.passed is False
    assert result.clone_weight_share == 0.02
    assert "degraded" in result.message
    assert "2.0%" in result.message
    assert "refusing to benchmark" in result.message


def test_gate_exactly_at_floor_passes(tmp_path: Path) -> None:
    """A share exactly at the floor passes (>= comparison)."""
    p = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    _write(p, {"clone_household_weight_share": 0.05})
    assert evaluate_clone_floor_gate(p).passed is True


# --------------------------------------------------------------------------- #
# Fail-closed paths
# --------------------------------------------------------------------------- #
def test_gate_fails_closed_when_file_missing(tmp_path: Path) -> None:
    """A missing sidecar fails closed (refuse, don't silently benchmark)."""
    p = tmp_path / "does_not_exist.clone_diagnostics.json"
    result = evaluate_clone_floor_gate(p)
    assert result.passed is False
    assert result.clone_weight_share is None
    assert "not found" in result.message
    assert "fail closed" in result.message


def test_gate_fails_closed_when_field_missing(tmp_path: Path) -> None:
    """A sidecar exposing no clone share fails closed, not silently.

    Documented choice: a missing clone-share field is treated like a bad
    baseline (fail closed), never as an implicit pass.
    """
    p = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    _write(p, {"total_households": 41000, "some_other_field": 1})
    result = evaluate_clone_floor_gate(p)
    assert result.passed is False
    assert result.clone_weight_share is None
    assert "no clone" in result.message
    assert "fail closed" in result.message


def test_gate_fails_closed_when_json_malformed(tmp_path: Path) -> None:
    """An unparseable sidecar fails closed with a malformed message."""
    p = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    p.write_text("{not valid json")
    result = evaluate_clone_floor_gate(p)
    assert result.passed is False
    assert result.clone_weight_share is None
    assert "malformed" in result.message
    assert "fail closed" in result.message


def test_gate_fails_closed_when_share_non_numeric(tmp_path: Path) -> None:
    """A non-numeric clone-share value is treated as absent (fail closed)."""
    p = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    _write(p, {"clone_household_weight_share": "lots"})
    result = evaluate_clone_floor_gate(p)
    assert result.passed is False
    assert result.clone_weight_share is None


# --------------------------------------------------------------------------- #
# Custom floor + schema flexibility
# --------------------------------------------------------------------------- #
def test_gate_respects_custom_floor(tmp_path: Path) -> None:
    """A 4% share passes a 3% floor but fails the default 5% floor."""
    p = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    _write(p, {"clone_household_weight_share": 0.04})
    assert evaluate_clone_floor_gate(p, floor=0.03).passed is True
    assert evaluate_clone_floor_gate(p, floor=0.05).passed is False


def test_gate_reads_share_from_summary(tmp_path: Path) -> None:
    """The share may live under a nested 'summary' object."""
    p = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    _write(p, {"summary": {"support_household_weight_share": 0.08}})
    result = evaluate_clone_floor_gate(p)
    assert result.passed is True
    assert result.clone_weight_share == 0.08


def test_extract_share_sums_clone_source_rows() -> None:
    """When only a sources list is present, clone rows are summed."""
    payload = {
        "sources": [
            {
                "source_name": "cps_asec",
                "source_class": "base",
                "household_weight_share": 0.90,
            },
            {
                "source_name": "irs_soi_puf_support_clone",
                "source_class": "puf_support",
                "household_weight_share": 0.07,
            },
            {
                "source_name": "donor_replay_psid",
                "source_class": "donor_replay",
                "household_weight_share": 0.03,
            },
            {
                "source_name": "forbes_400_fixed",
                "source_class": "fixed",
                "household_weight_share": 0.001,
            },
        ]
    }
    # 0.07 + 0.03 = 0.10; base and forbes/fixed are excluded.
    assert extract_clone_weight_share(payload) == 0.10


def test_gate_passes_via_summed_clone_source_rows(tmp_path: Path) -> None:
    """End-to-end: a sources-list sidecar with 10% clone share passes."""
    p = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    _write(
        p,
        {
            "sources": [
                {"source_name": "cps_asec", "household_weight_share": 0.90},
                {
                    "source_name": "puf_support_clone",
                    "household_weight_share": 0.10,
                },
            ]
        },
    )
    assert evaluate_clone_floor_gate(p).passed is True
