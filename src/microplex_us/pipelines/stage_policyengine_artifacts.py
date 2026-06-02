"""PolicyEngine entity stage artifact I/O for US saved runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microplex_us.pipelines.stage_contracts import (
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.stage_manifest_io import write_json_atomically
from microplex_us.pipelines.stage_manifest_types import US_POLICYENGINE_ENTITY_STAGE_ID
from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    USPipelineCheckpointStage,
    load_us_pipeline_checkpoint,
    save_us_pipeline_checkpoint,
)


def write_us_policyengine_entity_stage_artifact(
    bundle: PolicyEngineUSEntityTableBundle,
    artifact_root: str | Path,
    *,
    stage_id: str = US_POLICYENGINE_ENTITY_STAGE_ID,
    artifact_key: str = "pre_calibration_policyengine_entity_tables",
    checkpoint_stage: USPipelineCheckpointStage = "post_microsim",
) -> Path:
    """Persist a PE entity-table checkpoint under a saved-run root."""

    metadata_path = resolve_us_stage_artifact_contract_path(
        artifact_root,
        stage_id,
        artifact_key,
    )
    stage_dir = save_us_pipeline_checkpoint(
        bundle,
        metadata_path.parent,
        stage=checkpoint_stage,
    )
    metadata_path = stage_dir / metadata_path.name
    metadata = json.loads(metadata_path.read_text())
    metadata["stageId"] = stage_id
    metadata["artifactKey"] = artifact_key
    write_json_atomically(metadata_path, metadata)
    return metadata_path


def load_us_policyengine_entity_stage_artifact(
    path: str | Path,
    *,
    expected_stage: USPipelineCheckpointStage | None = "post_microsim",
) -> tuple[PolicyEngineUSEntityTableBundle, dict[str, Any]]:
    """Load a PE entity-table bundle artifact."""

    input_path = Path(path)
    checkpoint_dir = input_path if input_path.is_dir() else input_path.parent
    bundle, metadata = load_us_pipeline_checkpoint(
        checkpoint_dir,
        expected_stage=expected_stage,
    )
    return bundle, metadata


__all__ = [
    "load_us_policyengine_entity_stage_artifact",
    "write_us_policyengine_entity_stage_artifact",
]
