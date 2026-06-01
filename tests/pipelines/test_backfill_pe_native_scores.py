"""Tests for historical PE-native score backfill."""

from __future__ import annotations

import json

from microplex_us.pipelines.backfill_pe_native_scores import (
    backfill_us_pe_native_scores_bundles,
    backfill_us_pe_native_scores_root,
)
from microplex_us.pipelines.registry import load_us_microplex_run_registry


def test_backfill_us_pe_native_scores_root_updates_manifest_and_registry(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_root = tmp_path / "live_runs"
    bundle_dir = artifact_root / "run-1"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "policyengine_us.h5").write_text("candidate")
    (tmp_path / "baseline.h5").write_text("baseline")

    manifest = {
        "created_at": "2026-03-29T12:00:00+00:00",
        "config": {
            "synthesis_backend": "bootstrap",
            "calibration_backend": "entropy",
            "policyengine_baseline_dataset": str((tmp_path / "baseline.h5").resolve()),
            "policyengine_dataset_year": 2024,
        },
        "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
        "weights": {"nonzero": 20, "total": 1000.0},
        "synthesis": {"source_names": ["cps", "puf"]},
        "calibration": {
            "converged": True,
            "weight_collapse_suspected": False,
        },
        "artifacts": {
            "policyengine_dataset": "policyengine_us.h5",
        },
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_scores.compute_us_pe_native_scores",
        lambda **_kwargs: {
            "metric": "enhanced_cps_native_loss",
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.25,
                "baseline_enhanced_cps_native_loss": 0.5,
                "enhanced_cps_native_loss_delta": -0.25,
                "candidate_beats_baseline": True,
                "candidate_unweighted_msre": 0.3,
                "baseline_unweighted_msre": 0.6,
                "unweighted_msre_delta": -0.3,
                "n_targets_total": 2865,
                "n_targets_kept": 2853,
                "n_targets_zero_dropped": 10,
                "n_targets_bad_dropped": 10,
                "n_national_targets": 677,
                "n_state_targets": 2176,
            },
            "broad_loss": {
                "metric": "enhanced_cps_native_loss",
            },
        },
    )

    manifest_paths = backfill_us_pe_native_scores_root(artifact_root)

    assert manifest_paths == [manifest_path]
    sidecar_path = bundle_dir / "policyengine_native_scores.json"
    assert sidecar_path.exists()

    updated_manifest = json.loads(manifest_path.read_text())
    stage_manifest = json.loads((bundle_dir / "stage_manifest.json").read_text())
    validation_evidence = json.loads(
        (
            bundle_dir
            / "stage_artifacts"
            / "09_validation_benchmarking"
            / "evidence_manifest.json"
        ).read_text()
    )
    assert (
        updated_manifest["artifacts"]["policyengine_native_scores"]
        == "policyengine_native_scores.json"
    )
    assert updated_manifest["artifacts"]["stage_manifest"] == "stage_manifest.json"
    assert (
        updated_manifest["artifacts"]["validation_evidence"]
        == "stage_artifacts/09_validation_benchmarking/evidence_manifest.json"
    )
    assert updated_manifest["policyengine_native_scores"]["candidate_beats_baseline"] is True
    assert (
        updated_manifest["run_registry"]["default_frontier_metric"]
        == "enhanced_cps_native_loss_delta"
    )
    stage9 = next(
        stage
        for stage in stage_manifest["stages"]
        if stage["id"] == "09_validation_benchmarking"
    )
    assert stage9["status"] == "ready"
    assert validation_evidence["evidence"][0]["key"] == "policyengine_native_scores"
    assert validation_evidence["evidence"][0]["exists"] is True

    registry_path = artifact_root / "run_registry.jsonl"
    assert registry_path.exists()
    registry_entries = load_us_microplex_run_registry(registry_path)
    assert len(registry_entries) == 1
    assert registry_entries[0].candidate_beats_baseline_native_loss is True


def test_backfill_us_pe_native_scores_bundles_uses_batch_scorer(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_root = tmp_path / "live_runs"
    bundle_dirs = [artifact_root / "run-1", artifact_root / "run-2"]
    baseline_path = tmp_path / "baseline.h5"
    baseline_path.write_text("baseline")

    for index, bundle_dir in enumerate(bundle_dirs, start=1):
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "policyengine_us.h5").write_text(f"candidate-{index}")
        manifest = {
            "created_at": f"2026-03-29T12:00:0{index}+00:00",
            "config": {
                "synthesis_backend": "bootstrap",
                "calibration_backend": "entropy",
                "policyengine_baseline_dataset": str(baseline_path.resolve()),
                "policyengine_dataset_year": 2024,
            },
            "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
            "weights": {"nonzero": 20, "total": 1000.0},
            "synthesis": {"source_names": ["cps", "puf"]},
            "calibration": {
                "converged": True,
                "weight_collapse_suspected": False,
            },
            "artifacts": {
                "policyengine_dataset": "policyengine_us.h5",
            },
        }
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )

    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_scores.compute_batch_us_pe_native_scores",
        lambda **_kwargs: [
            {
                "metric": "enhanced_cps_native_loss",
                "summary": {
                    "candidate_enhanced_cps_native_loss": 0.25,
                    "baseline_enhanced_cps_native_loss": 0.5,
                    "enhanced_cps_native_loss_delta": -0.25,
                    "candidate_beats_baseline": True,
                    "candidate_unweighted_msre": 0.3,
                    "baseline_unweighted_msre": 0.6,
                    "unweighted_msre_delta": -0.3,
                    "n_targets_total": 2865,
                    "n_targets_kept": 2853,
                    "n_targets_zero_dropped": 10,
                    "n_targets_bad_dropped": 10,
                    "n_national_targets": 677,
                    "n_state_targets": 2176,
                },
                "broad_loss": {"metric": "enhanced_cps_native_loss"},
            },
            {
                "metric": "enhanced_cps_native_loss",
                "summary": {
                    "candidate_enhanced_cps_native_loss": 0.75,
                    "baseline_enhanced_cps_native_loss": 0.5,
                    "enhanced_cps_native_loss_delta": 0.25,
                    "candidate_beats_baseline": False,
                    "candidate_unweighted_msre": 0.8,
                    "baseline_unweighted_msre": 0.6,
                    "unweighted_msre_delta": 0.2,
                    "n_targets_total": 2865,
                    "n_targets_kept": 2853,
                    "n_targets_zero_dropped": 10,
                    "n_targets_bad_dropped": 10,
                    "n_national_targets": 677,
                    "n_state_targets": 2176,
                },
                "broad_loss": {"metric": "enhanced_cps_native_loss"},
            },
        ],
    )

    manifest_paths = backfill_us_pe_native_scores_bundles(bundle_dirs)

    assert len(manifest_paths) == 2
    registry_entries = load_us_microplex_run_registry(artifact_root / "run_registry.jsonl")
    assert len(registry_entries) == 2
    assert registry_entries[0].candidate_beats_baseline_native_loss is True
    assert registry_entries[1].candidate_beats_baseline_native_loss is False
