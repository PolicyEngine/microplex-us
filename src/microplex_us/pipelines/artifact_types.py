"""Shared result types for saved US Microplex artifact bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microplex_us.pipelines.us import USMicroplexBuildResult


@dataclass(frozen=True)
class USMicroplexArtifactPaths:
    """Filesystem locations for persisted pipeline artifacts."""

    output_dir: Path
    seed_data: Path
    synthetic_data: Path
    calibrated_data: Path
    targets: Path
    manifest: Path
    version_id: str | None = None
    scaffold_seed_data: Path | None = None
    synthesizer: Path | None = None
    policyengine_dataset: Path | None = None
    data_flow_snapshot: Path | None = None
    stage_manifest: Path | None = None
    artifact_inventory: Path | None = None
    conditional_readiness: Path | None = None
    source_plan: Path | None = None
    pre_calibration_policyengine_entity_tables: Path | None = None
    policyengine_entity_tables: Path | None = None
    calibration_summary: Path | None = None
    validation_evidence: Path | None = None
    policyengine_harness: Path | None = None
    policyengine_native_scores: Path | None = None
    policyengine_native_audit: Path | None = None
    policyengine_native_target_diagnostics: Path | None = None
    child_tax_unit_agi_drift: Path | None = None
    capital_gains_lots: Path | None = None
    source_weight_diagnostics: Path | None = None
    run_registry: Path | None = None
    run_index_db: Path | None = None


@dataclass(frozen=True)
class USMicroplexVersionedBuildArtifacts:
    """End-to-end build, save, and frontier-tracking result."""

    build_result: USMicroplexBuildResult
    artifact_paths: USMicroplexArtifactPaths
    current_entry: Any | None = None
    frontier_entry: Any | None = None
    frontier_delta: float | None = None
