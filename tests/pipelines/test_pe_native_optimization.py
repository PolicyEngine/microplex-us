"""Tests for direct PE-native weight optimization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from microplex_us.pipelines import pe_native_optimization as pe_opt
from microplex_us.pipelines.pe_native_optimization import (
    PolicyEngineUSNativeWeightOptimizationResult,
    optimize_pe_native_loss_weights,
    optimize_policyengine_us_native_loss_dataset,
    rewrite_policyengine_us_dataset_weights,
)


def _write_time_period_array(handle: h5py.File, name: str, values: np.ndarray) -> None:
    group = handle.create_group(name)
    group.create_dataset("2024", data=values)


def _build_stub_dataset(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        _write_time_period_array(
            handle,
            "household_id",
            np.asarray([10, 20], dtype=np.int64),
        )
        _write_time_period_array(
            handle,
            "household_weight",
            np.asarray([1.0, 2.0], dtype=np.float32),
        )
        _write_time_period_array(
            handle,
            "person_household_id",
            np.asarray([10, 10, 20], dtype=np.int64),
        )
        _write_time_period_array(
            handle,
            "person_weight",
            np.asarray([1.0, 1.0, 2.0], dtype=np.float32),
        )
        _write_time_period_array(
            handle,
            "tax_unit_id",
            np.asarray([100, 200], dtype=np.int64),
        )
        _write_time_period_array(
            handle,
            "person_tax_unit_id",
            np.asarray([100, 100, 200], dtype=np.int64),
        )
        _write_time_period_array(
            handle,
            "tax_unit_weight",
            np.asarray([1.0, 2.0], dtype=np.float32),
        )
    return path


def test_optimize_pe_native_loss_weights_reduces_objective_and_respects_budget():
    scaled_matrix = np.eye(3, dtype=np.float64)
    scaled_target = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    initial_weights = np.asarray([1.0 / 3.0] * 3, dtype=np.float64)

    optimized_weights, summary = optimize_pe_native_loss_weights(
        scaled_matrix=scaled_matrix,
        scaled_target=scaled_target,
        initial_weights=initial_weights,
        budget=1,
        max_iter=200,
    )

    assert np.allclose(optimized_weights, np.asarray([1.0, 0.0, 0.0]), atol=1e-6)
    assert summary["optimized_loss"] < summary["initial_loss"]
    assert summary["positive_household_count"] == 1
    assert np.isclose(summary["optimized_weight_sum"], initial_weights.sum())
    assert summary["loss_history"][0]["iteration"] == 0
    assert summary["loss_history"][0]["objective_loss"] == summary["initial_loss"]
    assert summary["loss_history"][-1]["objective_loss"] == summary["optimized_loss"]
    assert all(
        next_row["objective_loss"] <= row["objective_loss"] + 1e-12
        for row, next_row in zip(
            summary["loss_history"],
            summary["loss_history"][1:],
            strict=False,
        )
    )


def test_optimize_pe_native_loss_weights_respects_target_total_weight():
    """When target_total_weight is given, the simplex projection uses that total."""
    scaled_matrix = np.eye(3, dtype=np.float64)
    scaled_target = np.asarray([5.0, 0.0, 0.0], dtype=np.float64)
    initial_weights = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

    optimized_weights, summary = optimize_pe_native_loss_weights(
        scaled_matrix=scaled_matrix,
        scaled_target=scaled_target,
        initial_weights=initial_weights,
        target_total_weight=5.0,
        max_iter=200,
    )

    assert np.isclose(optimized_weights.sum(), 5.0, atol=1e-6)
    assert summary["initial_weight_sum"] == 3.0
    assert summary["target_total_weight"] == 5.0
    assert np.isclose(summary["optimized_weight_sum"], 5.0, atol=1e-6)
    assert summary["optimized_loss"] < summary["initial_loss"]


def test_optimize_pe_native_loss_weights_rejects_objective_increasing_steps(
    monkeypatch,
):
    """An overlarge initial step must not make an already-good fit worse."""
    scaled_matrix = np.eye(2, dtype=np.float64)
    scaled_target = np.asarray([0.6, 0.4], dtype=np.float64)
    initial_weights = np.asarray([0.5, 0.5], dtype=np.float64)
    initial_loss = np.square(scaled_matrix.T @ initial_weights - scaled_target).sum()
    monkeypatch.setattr(
        pe_opt,
        "_estimate_quadratic_lipschitz",
        lambda matrix, l2_penalty: 0.01,
    )

    optimized_weights, summary = optimize_pe_native_loss_weights(
        scaled_matrix=scaled_matrix,
        scaled_target=scaled_target,
        initial_weights=initial_weights,
        max_iter=20,
    )

    optimized_loss = np.square(
        scaled_matrix.T @ optimized_weights - scaled_target
    ).sum()
    assert optimized_loss <= initial_loss + 1e-12
    assert summary["optimized_loss"] <= summary["initial_loss"] + 1e-12
    assert summary["line_search_backtracking_steps"] > 0


def test_rewrite_policyengine_us_dataset_weights_updates_group_weights(tmp_path: Path):
    source_path = _build_stub_dataset(tmp_path / "input.h5")
    output_path = tmp_path / "output.h5"

    rewritten = rewrite_policyengine_us_dataset_weights(
        input_dataset_path=source_path,
        output_dataset_path=output_path,
        household_weights=np.asarray([7.0, 3.0], dtype=np.float64),
    )

    assert rewritten == output_path.resolve()
    with h5py.File(output_path, "r") as handle:
        assert np.allclose(handle["household_weight"]["2024"][:], np.asarray([7.0, 3.0]))
        assert np.allclose(
            handle["person_weight"]["2024"][:],
            np.asarray([7.0, 7.0, 3.0]),
        )
        assert np.allclose(handle["tax_unit_weight"]["2024"][:], np.asarray([7.0, 3.0]))


def test_rewrite_policyengine_us_dataset_weights_skips_empty_derived_weight_group(
    tmp_path: Path,
):
    """Published datasets (e.g. the production enhanced CPS) can leave derived
    entity-weight groups empty because PolicyEngine computes those weights from
    household_weight at runtime. Rewriting must skip such groups instead of
    raising KeyError on the missing period dataset."""
    source_path = _build_stub_dataset(tmp_path / "input.h5")
    # Mirror the production layout: a derived-weight group that exists but has
    # no value for the period, with its id arrays present.
    with h5py.File(source_path, "a") as handle:
        handle.create_group("family_weight")  # empty: no "2024" dataset
        _write_time_period_array(
            handle, "family_id", np.asarray([1000, 2000], dtype=np.int64)
        )
        _write_time_period_array(
            handle,
            "person_family_id",
            np.asarray([1000, 1000, 2000], dtype=np.int64),
        )
    output_path = tmp_path / "output.h5"

    rewritten = rewrite_policyengine_us_dataset_weights(
        input_dataset_path=source_path,
        output_dataset_path=output_path,
        household_weights=np.asarray([7.0, 3.0], dtype=np.float64),
    )

    with h5py.File(rewritten, "r") as handle:
        # The primary household weights are still rewritten.
        assert np.allclose(handle["household_weight"]["2024"][:], np.asarray([7.0, 3.0]))
        # The empty derived-weight group is left untouched (still has no period).
        assert "2024" not in handle["family_weight"]
        # A populated sibling derived group is still propagated in the same
        # call: the skip is per-group, not all-or-nothing.
        assert np.allclose(handle["tax_unit_weight"]["2024"][:], np.asarray([7.0, 3.0]))


def test_optimize_policyengine_us_native_loss_dataset_rewrites_dataset(tmp_path: Path, monkeypatch):
    source_path = _build_stub_dataset(tmp_path / "input.h5")
    output_path = tmp_path / "optimized.h5"

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_optimization.resolve_policyengine_us_data_repo_root",
        lambda repo: Path("/tmp/policyengine-us-data"),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_optimization.build_policyengine_us_data_subprocess_env",
        lambda repo: {"PATH": "/usr/bin"},
    )

    def _fake_run(args, **kwargs):
        prefix = Path(args[-1])
        np.save(
            prefix.with_suffix(".matrix.npy"),
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        )
        np.save(
            prefix.with_suffix(".target.npy"),
            np.asarray([1.0, 2.0], dtype=np.float64),
        )
        np.save(
            prefix.with_suffix(".weights.npy"),
            np.asarray([1.0, 2.0], dtype=np.float64),
        )
        prefix.with_suffix(".meta.json").write_text(
            json.dumps({"target_names": ["nation/foo", "state/bar"]})
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_optimization.subprocess.run",
        _fake_run,
    )

    result = optimize_policyengine_us_native_loss_dataset(
        input_dataset_path=source_path,
        output_dataset_path=output_path,
        max_iter=100,
    )

    assert isinstance(result, PolicyEngineUSNativeWeightOptimizationResult)
    assert result.output_dataset == str(output_path.resolve())
    assert result.optimized_loss <= result.initial_loss
    with h5py.File(output_path, "r") as handle:
        assert np.allclose(
            handle["household_weight"]["2024"][:],
            np.asarray([1.0, 2.0]),
        )
