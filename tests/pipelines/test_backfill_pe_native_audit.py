"""Tests for historical PE rebuild native-audit backfill."""

from __future__ import annotations

import json
from pathlib import Path

from microplex_us.pipelines.backfill_pe_native_audit import (
    backfill_us_pe_native_audit_bundle,
    backfill_us_pe_native_audit_bundles,
    backfill_us_pe_native_audit_root,
)


def test_backfill_us_pe_native_audit_root_updates_manifest_and_snapshot(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_root = tmp_path / "live_runs"
    bundle_dir = artifact_root / "run-1"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "policyengine_us.h5").write_text("candidate")
    (bundle_dir / "policyengine_native_scores.json").write_text(
        json.dumps(
            {
                "metric": "enhanced_cps_native_loss",
                "summary": {
                    "enhanced_cps_native_loss_delta": 0.25,
                },
            }
        )
    )
    (bundle_dir / "data_flow_snapshot.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "stages": [
                    {
                        "id": "benchmark",
                        "outputs": ["policyengine_native_scores.json"],
                        "metrics": [],
                        "status": "ready",
                    }
                ],
            }
        )
    )

    manifest = {
        "created_at": "2026-03-29T12:00:00+00:00",
        "config": {
            "policyengine_baseline_dataset": str((tmp_path / "baseline.h5").resolve()),
            "policyengine_dataset_year": 2024,
        },
        "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
        "weights": {"nonzero": 20, "total": 1000.0},
        "synthesis": {"source_names": ["cps", "puf"]},
        "calibration": {"converged": True},
        "artifacts": {
            "policyengine_dataset": "policyengine_us.h5",
            "policyengine_native_scores": "policyengine_native_scores.json",
        },
        "policyengine_native_scores": {
            "enhanced_cps_native_loss_delta": 0.25,
        },
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.compute_batch_us_pe_native_target_deltas",
        lambda **_kwargs: [
            {
                "metric": "enhanced_cps_native_loss_target_delta",
                "to_dataset": str((bundle_dir / "policyengine_us.h5").resolve()),
                "top_regressions": [{"target_name": "nation/irs/example"}],
                "top_improvements": [],
            }
        ],
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.compute_batch_us_pe_native_support_audits",
        lambda **_kwargs: [
            {
                "metric": "enhanced_cps_support_audit",
                "candidate_dataset": str((bundle_dir / "policyengine_us.h5").resolve()),
                "comparisons": {
                    "critical_input_support": [
                        {
                            "variable": "has_esi",
                            "candidate_stored": False,
                            "baseline_stored": True,
                            "weighted_nonzero_delta": -10.0,
                        }
                    ]
                },
            }
        ],
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.build_policyengine_us_data_rebuild_native_audit",
        lambda *args, **kwargs: {
            "artifactId": "run-1",
            "verdictHints": {
                "largestRegressingFamily": "national_irs_other",
                "productionImputationVariant": "structured_pe_conditioning",
                "productionImputationVariantIsMaeWinner": False,
                "productionImputationVariantIsSupportWinner": True,
            },
        },
    )

    manifest_paths = backfill_us_pe_native_audit_root(artifact_root)

    assert manifest_paths == [manifest_path]
    updated_manifest = json.loads(manifest_path.read_text())
    assert (
        updated_manifest["artifacts"]["policyengine_native_audit"]
        == "pe_us_data_rebuild_native_audit.json"
    )
    assert (
        updated_manifest["policyengine_native_audit"][
            "productionImputationVariantIsSupportWinner"
        ]
        is True
    )
    snapshot = json.loads((bundle_dir / "data_flow_snapshot.json").read_text())
    benchmark = next(
        stage
        for stage in snapshot["stages"]
        if stage["id"] == "09_validation_benchmarking"
    )
    assert benchmark["outputs"] == [
        "policyengine_native_scores.json",
        "pe_us_data_rebuild_native_audit.json",
    ]


def test_backfill_us_pe_native_audit_bundle_reuses_existing_sidecar_without_recomputing(
    monkeypatch,
    tmp_path,
) -> None:
    bundle_dir = tmp_path / "run-1"
    bundle_dir.mkdir()
    (bundle_dir / "policyengine_us.h5").write_text("candidate")
    (bundle_dir / "policyengine_native_scores.json").write_text(
        json.dumps({"metric": "enhanced_cps_native_loss", "summary": {}})
    )
    (bundle_dir / "pe_us_data_rebuild_native_audit.json").write_text(
        json.dumps(
            {
                "artifactId": "run-1",
                "verdictHints": {
                    "largestRegressingFamily": "national_irs_other",
                    "productionImputationVariantIsMaeWinner": False,
                },
            }
        )
    )
    manifest = {
        "created_at": "2026-03-29T12:00:00+00:00",
        "config": {
            "policyengine_baseline_dataset": str((tmp_path / "baseline.h5").resolve()),
            "policyengine_dataset_year": 2024,
        },
        "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
        "weights": {"nonzero": 20, "total": 1000.0},
        "synthesis": {"source_names": ["cps", "puf"]},
        "calibration": {"converged": True},
        "artifacts": {
            "policyengine_dataset": "policyengine_us.h5",
            "policyengine_native_scores": "policyengine_native_scores.json",
        },
        "policyengine_native_scores": {},
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.build_policyengine_us_data_rebuild_native_audit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not recompute when audit sidecar already exists")
        ),
    )

    returned_manifest_path = backfill_us_pe_native_audit_bundle(bundle_dir)

    assert returned_manifest_path == manifest_path
    updated_manifest = json.loads(manifest_path.read_text())
    assert (
        updated_manifest["artifacts"]["policyengine_native_audit"]
        == "pe_us_data_rebuild_native_audit.json"
    )
    assert (
        updated_manifest["policyengine_native_audit"]["largestRegressingFamily"]
        == "national_irs_other"
    )


def test_backfill_us_pe_native_audit_bundles_uses_grouped_batch_helpers(
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
        (bundle_dir / "policyengine_native_scores.json").write_text(
            json.dumps(
                {
                    "metric": "enhanced_cps_native_loss",
                    "period": 2024,
                    "summary": {
                        "enhanced_cps_native_loss_delta": float(index),
                    },
                }
            )
        )
        manifest = {
            "created_at": f"2026-03-29T12:00:0{index}+00:00",
            "config": {
                "policyengine_baseline_dataset": str(baseline_path.resolve()),
                "policyengine_dataset_year": 2024,
            },
            "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
            "weights": {"nonzero": 20, "total": 1000.0},
            "synthesis": {"source_names": ["cps", "puf"]},
            "calibration": {"converged": True},
            "artifacts": {
                "policyengine_dataset": "policyengine_us.h5",
                "policyengine_native_scores": "policyengine_native_scores.json",
            },
            "policyengine_native_scores": {
                "enhanced_cps_native_loss_delta": float(index),
            },
        }
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )

    captured: dict[str, object] = {}

    def fake_batch_target_deltas(**kwargs):
        captured["target_kwargs"] = kwargs
        return [
            {
                "metric": "enhanced_cps_native_loss_target_delta",
                "to_dataset": str((bundle_dirs[0] / "policyengine_us.h5").resolve()),
                "top_regressions": [{"target_name": "target-a"}],
                "top_improvements": [],
            },
            {
                "metric": "enhanced_cps_native_loss_target_delta",
                "to_dataset": str((bundle_dirs[1] / "policyengine_us.h5").resolve()),
                "top_regressions": [{"target_name": "target-b"}],
                "top_improvements": [],
            },
        ]

    def fake_batch_support_audits(**kwargs):
        captured["support_kwargs"] = kwargs
        return [
            {
                "metric": "enhanced_cps_support_audit",
                "candidate_dataset": str((bundle_dirs[0] / "policyengine_us.h5").resolve()),
                "comparisons": {
                    "critical_input_support": [
                        {
                            "variable": "has_esi",
                            "candidate_stored": False,
                            "baseline_stored": True,
                            "weighted_nonzero_delta": -10.0,
                        }
                    ]
                },
            },
            {
                "metric": "enhanced_cps_support_audit",
                "candidate_dataset": str((bundle_dirs[1] / "policyengine_us.h5").resolve()),
                "comparisons": {
                    "critical_input_support": [
                        {
                            "variable": "rental_income",
                            "candidate_stored": True,
                            "baseline_stored": True,
                            "weighted_nonzero_delta": -2.0,
                        }
                    ]
                },
            },
        ]

    def fake_build_audit(
        artifact_dir,
        *,
        manifest_payload,
        native_scores_payload,
        target_delta_payload,
        support_audit_payload,
        **_kwargs,
    ):
        return {
            "artifactId": Path(artifact_dir).name,
            "nativeBroadLossSummary": dict(native_scores_payload.get("summary", {})),
            "topTargetRegressions": list(target_delta_payload.get("top_regressions", ())),
            "supportAuditSummary": {
                "missingStoredCriticalInputs": [
                    row["variable"]
                    for row in support_audit_payload["comparisons"]["critical_input_support"]
                    if row.get("baseline_stored") and not row.get("candidate_stored")
                ]
            },
            "verdictHints": {
                "largestRegressingTarget": target_delta_payload["top_regressions"][0]["target_name"],
                "missingStoredCriticalInputs": [
                    row["variable"]
                    for row in support_audit_payload["comparisons"]["critical_input_support"]
                    if row.get("baseline_stored") and not row.get("candidate_stored")
                ],
            },
        }

    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.compute_batch_us_pe_native_target_deltas",
        fake_batch_target_deltas,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.compute_batch_us_pe_native_support_audits",
        fake_batch_support_audits,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.build_policyengine_us_data_rebuild_native_audit",
        fake_build_audit,
    )

    manifest_paths = backfill_us_pe_native_audit_bundles(bundle_dirs)

    assert len(manifest_paths) == 2
    assert captured["target_kwargs"]["baseline_dataset_path"] == baseline_path.resolve()
    assert captured["support_kwargs"]["baseline_dataset_path"] == baseline_path.resolve()
    assert captured["target_kwargs"]["candidate_dataset_paths"] == [
        bundle_dirs[0] / "policyengine_us.h5",
        bundle_dirs[1] / "policyengine_us.h5",
    ]
    updated_manifest = json.loads((bundle_dirs[0] / "manifest.json").read_text())
    assert updated_manifest["policyengine_native_audit"]["largestRegressingTarget"] == "target-a"
    assert updated_manifest["policyengine_native_audit"]["missingStoredCriticalInputs"] == [
        "has_esi"
    ]


def test_backfill_us_pe_native_audit_bundles_skips_missing_native_scores(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_root = tmp_path / "live_runs"
    skipped_bundle = artifact_root / "run-missing-scores"
    ready_bundle = artifact_root / "run-ready"
    baseline_path = tmp_path / "baseline.h5"
    baseline_path.write_text("baseline")

    for bundle_dir in (skipped_bundle, ready_bundle):
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "policyengine_us.h5").write_text(bundle_dir.name)
        manifest = {
            "created_at": "2026-03-29T12:00:00+00:00",
            "config": {
                "policyengine_baseline_dataset": str(baseline_path.resolve()),
                "policyengine_dataset_year": 2024,
            },
            "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
            "weights": {"nonzero": 20, "total": 1000.0},
            "synthesis": {"source_names": ["cps", "puf"]},
            "calibration": {"converged": True},
            "artifacts": {
                "policyengine_dataset": "policyengine_us.h5",
            },
        }
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )

    (ready_bundle / "policyengine_native_scores.json").write_text(
        json.dumps(
            {
                "metric": "enhanced_cps_native_loss",
                "period": 2024,
                "summary": {"enhanced_cps_native_loss_delta": 0.5},
            }
        )
    )
    ready_manifest = json.loads((ready_bundle / "manifest.json").read_text())
    ready_manifest["artifacts"]["policyengine_native_scores"] = "policyengine_native_scores.json"
    (ready_bundle / "manifest.json").write_text(
        json.dumps(ready_manifest, indent=2, sort_keys=True)
    )

    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.compute_batch_us_pe_native_target_deltas",
        lambda **_kwargs: [
            {
                "metric": "enhanced_cps_native_loss_target_delta",
                "to_dataset": str((ready_bundle / "policyengine_us.h5").resolve()),
                "top_regressions": [{"target_name": "target-ready"}],
                "top_improvements": [],
            }
        ],
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.compute_batch_us_pe_native_support_audits",
        lambda **_kwargs: [
            {
                "metric": "enhanced_cps_support_audit",
                "candidate_dataset": str((ready_bundle / "policyengine_us.h5").resolve()),
                "comparisons": {"critical_input_support": []},
            }
        ],
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.build_policyengine_us_data_rebuild_native_audit",
        lambda artifact_dir, **_kwargs: {
            "artifactId": Path(artifact_dir).name,
            "verdictHints": {"largestRegressingTarget": "target-ready"},
        },
    )

    manifest_paths = backfill_us_pe_native_audit_bundles([skipped_bundle, ready_bundle])

    assert manifest_paths == [ready_bundle / "manifest.json"]
    assert not (skipped_bundle / "pe_us_data_rebuild_native_audit.json").exists()
    assert (ready_bundle / "pe_us_data_rebuild_native_audit.json").exists()
