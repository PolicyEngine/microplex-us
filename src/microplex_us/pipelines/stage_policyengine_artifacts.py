"""PolicyEngine entity stage artifact I/O for US saved runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microplex_us.pipelines.stage_manifest_io import write_json_atomically
from microplex_us.pipelines.stage_manifest_types import (
    US_POLICYENGINE_ENTITY_STAGE_ID,
    US_STAGE_ARTIFACT_ROOT,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    load_us_pipeline_checkpoint,
    save_us_pipeline_checkpoint,
)


def write_us_policyengine_entity_stage_artifact(
    bundle: PolicyEngineUSEntityTableBundle,
    artifact_root: str | Path,
) -> Path:
    """Persist a Stage 6 PE entity-table checkpoint under a saved-run root."""

    stage_dir = save_us_pipeline_checkpoint(
        bundle,
        Path(artifact_root) / US_STAGE_ARTIFACT_ROOT / US_POLICYENGINE_ENTITY_STAGE_ID,
        stage="post_microsim",
    )
    metadata_path = stage_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["stageId"] = US_POLICYENGINE_ENTITY_STAGE_ID
    write_json_atomically(metadata_path, metadata)
    return metadata_path


def load_us_policyengine_entity_stage_artifact(
    path: str | Path,
) -> tuple[PolicyEngineUSEntityTableBundle, dict[str, Any]]:
    """Load a Stage 6 PE entity-table bundle artifact."""

    input_path = Path(path)
    checkpoint_dir = input_path if input_path.is_dir() else input_path.parent
    bundle, metadata = load_us_pipeline_checkpoint(
        checkpoint_dir,
        expected_stage="post_microsim",
    )
    return bundle, metadata


__all__ = [
    "load_us_policyengine_entity_stage_artifact",
    "write_us_policyengine_entity_stage_artifact",
]
