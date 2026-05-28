"""Tests for compact PolicyEngine H5 export."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from microplex_us.pipelines.compact_policyengine_dataset import (
    compact_policyengine_dataset_by_household_weight,
    main,
)
from microplex_us.policyengine.us import write_policyengine_us_time_period_dataset


def _write_dataset(path: Path) -> Path:
    arrays = {
        "household_id": {"2024": np.asarray([10, 20, 30, 40])},
        "household_weight": {"2024": np.asarray([1.0, 100.0, 5.0, 50.0])},
        "person_id": {"2024": np.asarray([1, 2, 3, 4, 5])},
        "person_household_id": {"2024": np.asarray([10, 20, 20, 30, 40])},
        "person_tax_unit_id": {"2024": np.asarray([100, 200, 200, 100, 400])},
        "tax_unit_id": {"2024": np.asarray([100, 200, 400])},
        "employment_income": {"2024": np.asarray([10.0, 20.0, 30.0, 40.0, 50.0])},
        "tax_unit_dependents": {"2024": np.asarray([0, 1, 1])},
        "household_net_worth": {"2024": np.asarray([1.0, 2.0, 3.0, 4.0])},
    }
    return write_policyengine_us_time_period_dataset(arrays, path)


def test_compact_policyengine_dataset_keeps_linked_entities_and_rescales(tmp_path):
    source = _write_dataset(tmp_path / "source.h5")
    output = tmp_path / "compact.h5"

    summary = compact_policyengine_dataset_by_household_weight(
        input_dataset_path=source,
        output_dataset_path=output,
        households=2,
        period=2024,
    )

    with h5py.File(output, "r") as handle:
        assert handle["household_id"]["2024"][:].tolist() == [20, 40]
        assert handle["household_weight"]["2024"][:].tolist() == pytest.approx(
            [104.0, 52.0]
        )
        assert handle["person_id"]["2024"][:].tolist() == [2, 3, 5]
        assert handle["person_household_id"]["2024"][:].tolist() == [20, 20, 40]
        assert handle["tax_unit_id"]["2024"][:].tolist() == [200, 400]
        assert handle["employment_income"]["2024"][:].tolist() == [
            20.0,
            30.0,
            50.0,
        ]
        assert handle["tax_unit_dependents"]["2024"][:].tolist() == [1, 1]
        assert handle["household_net_worth"]["2024"][:].tolist() == [2.0, 4.0]

    assert summary["source_households"] == 4
    assert summary["selected_households"] == 2
    assert summary["source_weight_sum"] == pytest.approx(156.0)
    assert summary["selected_weight_sum_before_rescale"] == pytest.approx(150.0)
    assert summary["output_weight_sum"] == pytest.approx(156.0)
    assert summary["entity_counts"] == {
        "household": 2,
        "person": 3,
        "tax_unit": 2,
    }


def test_compact_policyengine_dataset_can_select_from_external_weights(tmp_path):
    source = _write_dataset(tmp_path / "source.h5")
    selection_weights = np.asarray([100.0, 1.0, 50.0, 2.0])
    weights_path = tmp_path / "weights.npy"
    np.save(weights_path, selection_weights)

    summary = compact_policyengine_dataset_by_household_weight(
        input_dataset_path=source,
        output_dataset_path=tmp_path / "compact.h5",
        households=2,
        period=2024,
        weights_path=weights_path,
        rescale_to_total=False,
    )

    with h5py.File(tmp_path / "compact.h5", "r") as handle:
        assert handle["household_id"]["2024"][:].tolist() == [10, 30]
        assert handle["household_weight"]["2024"][:].tolist() == [1.0, 5.0]

    assert summary["output_weight_sum"] == pytest.approx(6.0)
    assert summary["target_total_weight"] is None
    assert summary["rescale_to_total"] is False


def test_compact_policyengine_dataset_cli_writes_summary(tmp_path):
    source = _write_dataset(tmp_path / "source.h5")
    output = tmp_path / "compact.h5"
    summary_path = tmp_path / "summary.json"

    exit_code = main(
        [
            "--input-dataset",
            str(source),
            "--output-dataset",
            str(output),
            "--households",
            "2",
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert output.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["selected_households"] == 2
    assert payload["output_dataset"] == str(output.resolve())


def test_compact_policyengine_dataset_rejects_mismatched_weights(tmp_path):
    source = _write_dataset(tmp_path / "source.h5")
    weights_path = tmp_path / "weights.npy"
    np.save(weights_path, np.asarray([1.0, 2.0]))

    with pytest.raises(ValueError, match="selection weights length"):
        compact_policyengine_dataset_by_household_weight(
            input_dataset_path=source,
            output_dataset_path=tmp_path / "compact.h5",
            households=2,
            weights_path=weights_path,
        )
