"""Result types for the PE-US-data checkpoint runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from microplex.core import SourceQuery

    from microplex_us.pipelines.artifacts import USMicroplexVersionedBuildArtifacts
    from microplex_us.pipelines.us import USMicroplexBuildConfig


@dataclass(frozen=True)
class PEUSDataRebuildCheckpointResult:
    """Saved artifact bundle plus attached PE comparison sidecars."""

    build_config: USMicroplexBuildConfig
    provider_names: tuple[str, ...]
    queries: dict[str, SourceQuery]
    artifacts: USMicroplexVersionedBuildArtifacts
    parity_path: Path
    parity_payload: dict[str, Any]
    native_audit_path: Path | None = None
    native_audit_payload: dict[str, Any] | None = None
    native_target_diagnostics_path: Path | None = None
    native_target_diagnostics_payload: dict[str, Any] | None = None
    imputation_ablation_path: Path | None = None
    imputation_ablation_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class PEUSDataRebuildCheckpointEvidenceResult:
    """Comparison evidence attached to one saved rebuild artifact."""

    artifact_dir: Path
    manifest_path: Path
    harness_path: Path | None
    native_scores_path: Path | None
    parity_path: Path
    parity_payload: dict[str, Any]
    native_audit_path: Path | None = None
    native_audit_payload: dict[str, Any] | None = None
    native_target_diagnostics_path: Path | None = None
    native_target_diagnostics_payload: dict[str, Any] | None = None
    imputation_ablation_path: Path | None = None
    imputation_ablation_payload: dict[str, Any] | None = None
