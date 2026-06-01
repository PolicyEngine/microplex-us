"""I/O helpers for saved-run US stage manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from microplex_us.pipelines.stage_manifest_builder import build_us_stage_manifest
from microplex_us.pipelines.stage_manifest_types import (
    SUPPORTED_US_STAGE_MANIFEST_SCHEMA_VERSIONS,
    USStageManifest,
)


def write_us_stage_manifest(
    artifact_dir: str | Path,
    output_path: str | Path,
    *,
    manifest_payload: dict[str, Any],
    assume_existing_artifact_keys: Iterable[str] = (),
) -> Path:
    """Write the canonical stage manifest for a saved US artifact bundle."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomically(
        destination,
        build_us_stage_manifest(
            artifact_dir,
            manifest_payload=manifest_payload,
            assume_existing_artifact_keys=(
                *tuple(assume_existing_artifact_keys),
                "stage_manifest",
            ),
        ),
    )
    return destination


def load_us_stage_manifest(path: str | Path) -> USStageManifest:
    """Load a saved stage manifest and validate its schema version."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    if payload.get("schemaVersion") not in SUPPORTED_US_STAGE_MANIFEST_SCHEMA_VERSIONS:
        raise RuntimeError(
            f"Unsupported US stage manifest schema: {payload.get('schemaVersion')!r}"
        )
    return cast(USStageManifest, payload)


def write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically through a sibling temporary file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


__all__ = [
    "load_us_stage_manifest",
    "write_json_atomically",
    "write_us_stage_manifest",
]
