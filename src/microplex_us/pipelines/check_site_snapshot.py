"""Validate that the committed US site snapshot matches its source artifact."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from microplex_us.pipelines.data_flow_snapshot import (
    build_us_microplex_data_flow_snapshot,
)
from microplex_us.pipelines.site_snapshot import build_us_microplex_site_snapshot
from microplex_us.pipelines.stage_contracts import (
    resolve_us_stage_artifact_contract_path,
)


def check_us_microplex_site_snapshot(
    snapshot_path: str | Path = "artifacts/site_snapshot_us.json",
) -> Path:
    """Raise if the saved US site snapshot is stale or inconsistent."""
    snapshot_file = Path(snapshot_path)
    snapshot = json.loads(snapshot_file.read_text())
    artifact_dir = _resolve_artifact_dir(snapshot_file, snapshot["sourceArtifact"])
    _check_data_flow_snapshot_current(artifact_dir)
    regenerated = build_us_microplex_site_snapshot(
        artifact_dir,
        snapshot_path=snapshot_file,
    )
    if snapshot != regenerated:
        raise SystemExit(_snapshot_diff(snapshot, regenerated, snapshot_file))
    return snapshot_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that the canonical US site snapshot is up to date."
    )
    parser.add_argument(
        "snapshot_path",
        nargs="?",
        default="artifacts/site_snapshot_us.json",
        help="Path to the saved US site snapshot JSON.",
    )
    args = parser.parse_args(argv)
    checked = check_us_microplex_site_snapshot(args.snapshot_path)
    print(checked)
    return 0


def _snapshot_diff(
    saved: dict,
    regenerated: dict,
    snapshot_file: Path,
) -> str:
    saved_text = json.dumps(saved, indent=2, sort_keys=True).splitlines()
    regenerated_text = json.dumps(regenerated, indent=2, sort_keys=True).splitlines()
    diff = "\n".join(
        difflib.unified_diff(
            saved_text,
            regenerated_text,
            fromfile=str(snapshot_file),
            tofile=f"{snapshot_file} (regenerated)",
            lineterm="",
        )
    )
    return f"US site snapshot is stale or inconsistent:\n{diff}"


def _resolve_artifact_dir(snapshot_file: Path, source_artifact: dict) -> Path:
    artifact_path = source_artifact.get("artifactPath")
    if isinstance(artifact_path, str) and artifact_path:
        artifact_dir = (snapshot_file.parent / artifact_path).resolve()
        if (artifact_dir / "manifest.json").exists():
            return artifact_dir

    artifact_ref = source_artifact.get("artifactRef")
    if not isinstance(artifact_ref, str) or not artifact_ref:
        raise SystemExit("US site snapshot is missing sourceArtifact.artifactRef")
    artifact_dir = (snapshot_file.parent / artifact_ref).resolve()
    if not (artifact_dir / "manifest.json").exists():
        raise SystemExit(
            f"US site snapshot artifactRef does not resolve to a manifest: {artifact_ref}"
        )
    return artifact_dir


def _check_data_flow_snapshot_current(artifact_dir: Path) -> None:
    snapshot_path = resolve_us_stage_artifact_contract_path(
        artifact_dir,
        "08_dataset_assembly",
        "data_flow_snapshot",
    )
    if not snapshot_path.exists():
        raise SystemExit("US data-flow snapshot is missing from the artifact bundle.")
    frozen_snapshot = json.loads(snapshot_path.read_text())
    fresh_snapshot = build_us_microplex_data_flow_snapshot(
        artifact_dir,
        prefer_saved=False,
    )
    if frozen_snapshot != fresh_snapshot:
        raise SystemExit(
            "US data-flow snapshot is stale or inconsistent with current code."
        )


if __name__ == "__main__":
    raise SystemExit(main())
