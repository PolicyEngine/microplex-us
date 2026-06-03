"""Low-level filesystem helpers for saved US Microplex artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from microplex_us.pipelines.stage_contracts import (
    get_us_stage_artifact_contract,
    resolve_us_stage_artifact_contract_path,
)
from microplex_us.pipelines.stage_run import USArtifactRef, USDiagnosticOutput


def _stage_artifact_ref(
    artifact_root: str | Path,
    stage_id: str,
    artifact_key: str,
    *,
    assume_exists: bool = False,
) -> USArtifactRef:
    contract = get_us_stage_artifact_contract(stage_id, artifact_key)
    return USArtifactRef(
        key=artifact_key,
        path=resolve_us_stage_artifact_contract_path(
            artifact_root,
            stage_id,
            artifact_key,
        ),
        format=contract.format,
        required=contract.required,
        resume_role=contract.resume_role,
        assume_exists=assume_exists,
    )


def _stage_diagnostics(
    stage_id: str,
    summary: Mapping[str, Any],
) -> dict[str, USDiagnosticOutput]:
    return {
        "stage_summary": USDiagnosticOutput(
            key="stage_summary",
            description=f"Runtime diagnostic summary for {stage_id}.",
            summary=dict(summary),
        )
    }


def _write_parquet_unless_live_artifact_exists(
    path: Path,
    frame: pd.DataFrame,
    *,
    live_artifact: bool,
) -> None:
    if live_artifact and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_json_unless_live_artifact_exists(
    path: Path,
    payload: Mapping[str, Any],
    *,
    live_artifact: bool,
) -> None:
    if live_artifact and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _resolve_saved_artifact_file(
    artifact_root: Path,
    manifest: dict[str, Any],
    artifact_key: str,
) -> Path:
    artifacts = dict(manifest.get("artifacts", {}))
    filename = artifacts.get(artifact_key)
    if not filename:
        filename = (
            "targets.json" if artifact_key == "targets" else f"{artifact_key}.parquet"
        )
    path = Path(filename)
    if not path.is_absolute():
        path = artifact_root / path
    if not path.exists():
        raise FileNotFoundError(f"Saved artifact file not found: {path}")
    return path


def _resolve_optional_saved_artifact_file(
    artifact_root: Path,
    manifest: dict[str, Any],
    artifact_key: str,
) -> Path | None:
    artifacts = dict(manifest.get("artifacts", {}))
    filename = artifacts.get(artifact_key)
    if not filename:
        return None
    path = Path(str(filename))
    if not path.is_absolute():
        path = artifact_root / path
    if not path.exists():
        raise FileNotFoundError(f"Saved optional artifact file not found: {path}")
    return path


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)
