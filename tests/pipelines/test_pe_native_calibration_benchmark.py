"""Tests for PE-native calibration strategy benchmarking."""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np

from microplex_us.pipelines.pe_native_calibration_benchmark import (
    build_policyengine_us_native_calibration_benchmark,
    compute_household_weight_diagnostics,
)


def _write_dataset(path: Path, weights: list[float]) -> Path:
    household_ids = np.arange(1, len(weights) + 1, dtype=np.int64)
    with h5py.File(path, "w") as handle:
        household_id = handle.create_group("household_id")
        household_id.create_dataset("2024", data=household_ids)
        household_weight = handle.create_group("household_weight")
        household_weight.create_dataset(
            "2024",
            data=np.asarray(weights, dtype=np.float32),
        )
    return path


def test_compute_household_weight_diagnostics_compares_reference_by_id(
    tmp_path: Path,
) -> None:
    candidate = _write_dataset(tmp_path / "candidate.h5", [3.0, 0.0, 9.0])
    reference = tmp_path / "reference.h5"
    with h5py.File(reference, "w") as handle:
        household_id = handle.create_group("household_id")
        household_id.create_dataset("2024", data=np.asarray([3, 1, 2]))
        household_weight = handle.create_group("household_weight")
        household_weight.create_dataset(
            "2024",
            data=np.asarray([6.0, 2.0, 1.0], dtype=np.float32),
        )

    diagnostics = compute_household_weight_diagnostics(
        candidate,
        reference_dataset_path=reference,
    )

    assert diagnostics["household_count"] == 3
    assert diagnostics["positive_household_count"] == 2
    assert diagnostics["weight_sum"] == 12.0
    assert diagnostics["reference_alignment"] == "matched_by_household_id"
    assert diagnostics["reference_weight_sum"] == 9.0
    assert diagnostics["weight_sum_delta"] == 3.0
    assert diagnostics["changed_household_count"] == 3
    assert np.isclose(diagnostics["effective_sample_size"], 1.6)


def test_build_policyengine_us_native_calibration_benchmark_scores_variants(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_dataset = _write_dataset(tmp_path / "input.h5", [1.0, 1.0])
    baseline_dataset = _write_dataset(tmp_path / "baseline.h5", [2.0, 2.0])
    existing_dataset = _write_dataset(tmp_path / "current_weight_diff.h5", [1.2, 0.8])
    output_dir = tmp_path / "benchmark"

    def fake_extract(**kwargs):
        assert kwargs["target_scope_filter"] == "national"
        return {
            "scaled_matrix": np.eye(2),
            "scaled_target": np.asarray([1.0, 0.0]),
            "initial_weights": np.asarray([1.0, 1.0]),
            "metadata": {
                "target_names": ["nation/fake", "state/fake"],
                "skip_tax_expenditure_targets": True,
            },
        }

    def fake_optimize_weights(**kwargs):
        penalty = float(kwargs["l2_penalty"])
        weights = np.asarray([1.9, 0.1] if penalty == 0.0 else [1.4, 0.6])
        return weights, {
            "initial_loss": 1.25,
            "optimized_loss": 0.5 if penalty == 0.0 else 0.75,
            "loss_delta": -0.75 if penalty == 0.0 else -0.5,
            "initial_weight_sum": 2.0,
            "optimized_weight_sum": float(weights.sum()),
            "household_count": 2,
            "positive_household_count": 2,
            "budget": None,
            "iterations": 3,
            "converged": True,
        }

    def fake_rewrite(**kwargs):
        output_path = Path(kwargs["output_dataset_path"])
        shutil.copy2(kwargs["input_dataset_path"], output_path)
        with h5py.File(output_path, "r+") as handle:
            handle["household_weight"]["2024"][...] = np.asarray(
                kwargs["household_weights"],
                dtype=np.float32,
            )
        return output_path.resolve()

    def fake_scores(**kwargs):
        assert kwargs["target_scope_filter"] == "national"
        results = []
        for candidate_path in kwargs["candidate_dataset_paths"]:
            path = Path(candidate_path).resolve()
            if path.name == "input.h5":
                loss = 1.0
            elif path.name == "current_weight_diff.h5":
                loss = 0.8
            elif "unconstrained" in path.name:
                loss = 0.4
            else:
                loss = 0.6
            results.append(
                {
                    "metric": "enhanced_cps_native_loss",
                    "period": 2024,
                    "summary": {
                        "candidate_enhanced_cps_native_loss": loss,
                        "baseline_enhanced_cps_native_loss": 0.5,
                        "enhanced_cps_native_loss_delta": loss - 0.5,
                        "candidate_beats_baseline": loss < 0.5,
                        "candidate_unweighted_msre": loss + 0.1,
                        "baseline_unweighted_msre": 0.7,
                        "unweighted_msre_delta": loss - 0.6,
                        "n_targets_total": 4,
                        "n_targets_kept": 3,
                        "n_targets_zero_dropped": 1,
                        "n_targets_bad_dropped": 0,
                        "n_national_targets": 1,
                        "n_state_targets": 2,
                        "skip_tax_expenditure_targets": True,
                    },
                    "broad_loss": {
                        "metric": "enhanced_cps_native_loss",
                        "period": 2024,
                        "candidate_dataset": str(path),
                        "baseline_dataset": str(baseline_dataset.resolve()),
                        "candidate_enhanced_cps_native_loss": loss,
                        "baseline_enhanced_cps_native_loss": 0.5,
                        "enhanced_cps_native_loss_delta": loss - 0.5,
                        "candidate_beats_baseline": loss < 0.5,
                        "candidate_unweighted_msre": loss + 0.1,
                        "baseline_unweighted_msre": 0.7,
                        "unweighted_msre_delta": loss - 0.6,
                        "n_targets_total": 4,
                        "n_targets_kept": 3,
                        "n_targets_zero_dropped": 1,
                        "n_targets_bad_dropped": 0,
                        "n_national_targets": 1,
                        "n_state_targets": 2,
                        "candidate_weight_sum": 2.0,
                        "baseline_weight_sum": 4.0,
                        "skip_tax_expenditure_targets": True,
                        "family_breakdown": [],
                    },
                    "family_breakdown": [],
                }
            )
        return results

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_calibration_benchmark."
        "_extract_pe_native_loss_inputs",
        fake_extract,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_calibration_benchmark."
        "optimize_pe_native_loss_weights",
        fake_optimize_weights,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_calibration_benchmark."
        "rewrite_policyengine_us_dataset_weights",
        fake_rewrite,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_calibration_benchmark."
        "compute_batch_us_pe_native_scores",
        fake_scores,
    )

    payload = build_policyengine_us_native_calibration_benchmark(
        input_dataset_path=input_dataset,
        baseline_dataset_path=baseline_dataset,
        output_dir=output_dir,
        l2_penalties=(0.0, 1e-8),
        max_iter=5,
        target_total_weight_source="baseline",
        existing_candidates={"current_weight_diff": existing_dataset},
        skip_tax_expenditure_targets=True,
        target_scope_filter="national",
    )

    assert payload["variant_count"] == 4
    assert payload["target_scope_filter"] == "national"
    assert payload["target_total_weight"] == 4.0
    assert payload["target_total_weight_resolved_from"] == "baseline"
    assert payload["best_variant_label"] == "pe_native_unconstrained_baseline_total"
    assert [row["label"] for row in payload["ranking"][:2]] == [
        "pe_native_unconstrained_baseline_total",
        "pe_native_l2_1e-08_baseline_total",
    ]
    unconstrained = next(
        row for row in payload["rows"] if row["label"].startswith("pe_native_unconstrained")
    )
    assert unconstrained["optimization"]["l2_penalty"] == 0.0
    assert unconstrained["weight_diagnostics"]["reference_alignment"] == "same_order"
    assert unconstrained["weight_diagnostics"]["changed_household_count"] == 2
