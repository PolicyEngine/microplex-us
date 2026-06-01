"""Display metrics for saved US pipeline stage manifests."""

from __future__ import annotations

from typing import Any

from microplex_us.pipelines.stage_manifest_types import USStageMetric


def stage_metrics(stage_id: str, *, manifest: dict[str, Any]) -> list[USStageMetric]:
    """Return compact display metrics for one saved stage."""

    synthesis = dict(manifest.get("synthesis", {}))
    calibration = dict(manifest.get("calibration", {}))
    artifacts = dict(manifest.get("artifacts", {}))
    harness = dict(manifest.get("policyengine_harness", {}))
    native_scores = dict(manifest.get("policyengine_native_scores", {}))
    rows = dict(manifest.get("rows", {}))
    config = dict(manifest.get("config", {}))
    if stage_id == "01_run_profile":
        return [
            {
                "label": "Target period",
                "value": config.get("policyengine_target_period"),
            },
            {"label": "Backend", "value": config.get("calibration_backend")},
        ]
    if stage_id == "02_source_loading":
        return [
            {"label": "Sources", "value": len(synthesis.get("source_names", ()))},
        ]
    if stage_id == "03_source_planning":
        return [{"label": "Scaffold", "value": synthesis.get("scaffold_source")}]
    if stage_id == "04_seed_scaffold":
        return [
            {"label": "Seed rows", "value": rows.get("seed")},
            {"label": "Scaffold", "value": synthesis.get("scaffold_source")},
        ]
    if stage_id == "05_donor_integration_synthesis":
        return [
            {"label": "Seed rows", "value": rows.get("seed")},
            {
                "label": "Integrated vars",
                "value": len(synthesis.get("donor_integrated_variables", ())),
            },
            {"label": "Backend", "value": synthesis.get("backend")},
            {"label": "Synthetic rows", "value": rows.get("synthetic")},
        ]
    if stage_id == "06_policyengine_entities":
        return [
            {
                "label": "Entity bundle",
                "value": artifacts.get("policyengine_entity_tables"),
            }
        ]
    if stage_id == "07_calibration":
        return [
            {"label": "Backend", "value": calibration.get("backend")},
            {"label": "Supported", "value": calibration.get("n_supported_targets")},
            {"label": "Converged", "value": calibration.get("converged")},
        ]
    if stage_id == "08_dataset_assembly":
        return [{"label": "Dataset", "value": artifacts.get("policyengine_dataset")}]
    if stage_id == "09_validation_benchmarking":
        imputation_ablation = dict(manifest.get("imputation_ablation", {}))
        return [
            {
                "label": "Capped full oracle loss",
                "value": calibration.get("full_oracle_capped_mean_abs_relative_error"),
            },
            {
                "label": "Full oracle loss",
                "value": calibration.get("full_oracle_mean_abs_relative_error"),
            },
            {
                "label": "Harness delta",
                "value": harness.get("mean_abs_relative_error_delta"),
            },
            {
                "label": "Native delta",
                "value": native_scores.get("enhanced_cps_native_loss_delta"),
            },
            {"label": "Win rate", "value": harness.get("target_win_rate")},
            {
                "label": "Imputation MAE",
                "value": imputation_ablation.get("production_mean_weighted_mae"),
            },
            {
                "label": "Imputation F1",
                "value": imputation_ablation.get("production_mean_support_f1"),
            },
        ]
    return []


__all__ = ["stage_metrics"]
