"""Tests for non-gating Microplex transparency sidecars."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "microplex_us"
    / "pipelines"
    / "transparency_sidecars.py"
)
_spec = importlib.util.spec_from_file_location("transparency_sidecars", _MODULE_PATH)
sidecars = importlib.util.module_from_spec(_spec)
sys.modules["transparency_sidecars"] = sidecars
_spec.loader.exec_module(sidecars)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_sidecars_parse_active_donor_block_without_h5(tmp_path):
    artifact_root = tmp_path / "artifact"
    log_path = artifact_root / "logs" / "gate1_build.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "\n".join(
            [
                "[2026-06-01T07:34:56-04:00] Starting Gate-1 fresh eCPS-shaped Microplex build.",
                "[2026-06-01T07:34:56-04:00] Shape: CPS/ASEC survey year 2025 spine (calendar/income year 2024) + PUF 2024 clones.",
                "PE-US-data rebuild checkpoint: starting build [output_root=/tmp/run, version_id=mp-test, target_profile=pe_native_broad, providers=cps_asec,irs_soi_puf]",
                "Downloading CPS ASEC 2025 from https://example.test/asec.zip...",
                "US microplex donor integration: source ready [donor_source=irs_soi_puf_2024, donor_rows=232699, shared_vars=14, donor_target_vars=71, blocks=70]",
                "US microplex donor integration: block start [donor_source=irs_soi_puf_2024, block=capital_gains, restored=capital_gains]",
                "US microplex donor integration: block run [donor_source=irs_soi_puf_2024, block=capital_gains, condition_vars=8, donor_rows=232699, current_rows=142125]",
            ]
        )
    )
    summary = sidecars.write_transparency_sidecars(artifact_root)

    assert summary["dataset_available"] is False
    assert summary["production_performance_gate"] == "loss"
    imputation = json.loads(
        (artifact_root / "transparency" / "imputation_manifest.json").read_text()
    )
    source = imputation["donor_integration"]["sources"][0]
    assert source["donor_source"] == "irs_soi_puf_2024"
    assert source["ready"]["blocks"] == 70
    assert source["active_blocks"] == ["capital_gains"]

    source_manifest = json.loads(
        (artifact_root / "transparency" / "source_manifest.json").read_text()
    )
    assert source_manifest["build_config"]["version_id"] == "mp-test"
    assert source_manifest["source_events"][0]["message"].startswith(
        "Downloading CPS ASEC 2025"
    )


def test_sidecars_summarize_h5_columns_rows_and_calibration(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    h5_path = artifact_root / "policyengine_us.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("age/2024", data=np.array([30, 40, 50]))
        h5.create_dataset("snap/2024", data=np.array([0, 1, 0]))
        h5.create_dataset("employment_income/2024", data=np.array([1, 2, 3]))
        h5.create_dataset("snap_reported/2024", data=np.array([0, 1, 0]))
        h5.create_dataset("household_weight/2024", data=np.array([1.0, 2.0]))
    contract = _write_json(
        tmp_path / "contract.json",
        {
            "required": ["age", "snap", "employment_income", "state_code"],
            "forbidden": ["snap_reported"],
            "ecps_internal_optional": [],
            "formula_owned_excluded": ["weeks_worked"],
        },
    )
    _write_json(
        artifact_root / "calibration_summary.json",
        {
            "backend": "policyengine_db_entropy",
            "period": 2024,
            "converged": False,
            "n_loaded_targets": 10,
            "n_supported_targets": 9,
        },
    )

    sidecars.write_transparency_sidecars(artifact_root, contract_path=contract)

    columns = json.loads(
        (artifact_root / "transparency" / "column_manifest.json").read_text()
    )
    assert columns["available"] is True
    assert columns["missing_required"] == ["state_code"]
    assert columns["forbidden_present"] == ["snap_reported"]
    assert columns["diagnostic_status"] == "needs_attention"

    rows = json.loads(
        (artifact_root / "transparency" / "row_count_manifest.json").read_text()
    )
    assert rows["available"] is True
    assert rows["shape_counts"][0]["shape"] == "3"
    assert rows["shape_counts"][0]["variable_count"] == 4

    calibration = json.loads(
        (artifact_root / "transparency" / "calibration_trace.json").read_text()
    )
    assert calibration["available"] is True
    assert calibration["summaries"][0]["n_supported_targets"] == 9
