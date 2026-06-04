"""Safe Stage 9 validation and benchmarking replay helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microplex_us.pipeline_metadata import pipeline_node
from microplex_us.pipelines.pe_native_scores import compute_us_pe_native_scores
from microplex_us.pipelines.stage_manifest_io import write_json_atomically
from microplex_us.pipelines.stage_validation_evidence import (
    build_us_validation_evidence_manifest,
)


@dataclass(frozen=True)
class USStage9ReplayResult:
    """Artifacts written by a Stage 9 replay."""

    output_dir: Path
    replay_manifest: Path
    validation_evidence: Path
    policyengine_harness: Path | None = None
    policyengine_native_scores: Path | None = None


@pipeline_node(
    id="us.stage9.replay_validation_benchmarking",
    label="Replay Stage 9 validation",
    description="Rerun validation and benchmark evidence against an existing Stage 8 dataset bundle.",
    artifacts_in=("policyengine_dataset", "artifact_manifest"),
    artifacts_out=("validation_evidence_manifest",),
)
def replay_us_stage9_validation_benchmarking(
    artifact_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    baseline_dataset: str | Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    period: int | None = None,
    precomputed_policyengine_harness: str | Path | dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: str | Path | dict[str, Any] | None = None,
    run_id: str | None = None,
    allow_overwrite: bool = False,
) -> USStage9ReplayResult:
    """Rerun safe Stage 9 evidence against an existing Stage 8 dataset.

    The original artifact bundle is left untouched. New evidence is written under
    a replay directory and indexed by a replay-local evidence manifest.
    """

    artifact_root = Path(artifact_dir).expanduser().resolve()
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Saved artifact manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    dataset_path = _validated_stage8_dataset_path(artifact_root, manifest)

    resolved_output_dir = _resolve_replay_output_dir(
        artifact_root,
        output_dir=output_dir,
        run_id=run_id,
    )
    if resolved_output_dir.exists() and any(resolved_output_dir.iterdir()):
        if not allow_overwrite:
            raise FileExistsError(
                f"Stage 9 replay output directory already exists and is not empty: "
                f"{resolved_output_dir}"
            )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    replay_manifest_payload = dict(manifest)
    replay_artifacts = dict(manifest.get("artifacts", {}))
    summaries: dict[str, Any] = {}

    harness_path = None
    harness_payload = _load_optional_payload(precomputed_policyengine_harness)
    if harness_payload is not None:
        harness_path = resolved_output_dir / "policyengine_harness.json"
        write_json_atomically(harness_path, harness_payload)
        replay_artifacts["policyengine_harness"] = _relative_to_root(
            harness_path,
            artifact_root,
        )
        if isinstance(harness_payload.get("summary"), dict):
            summaries["policyengine_harness"] = dict(harness_payload["summary"])

    native_scores_path = None
    native_scores_payload = _load_optional_payload(
        precomputed_policyengine_native_scores
    )
    if native_scores_payload is None and baseline_dataset is not None:
        native_scores_payload = compute_us_pe_native_scores(
            candidate_dataset_path=dataset_path,
            baseline_dataset_path=baseline_dataset,
            period=period
            or int(
                dict(manifest.get("config", {})).get(
                    "policyengine_dataset_year",
                    2024,
                )
            ),
            policyengine_us_data_repo=policyengine_us_data_repo,
        )
    if native_scores_payload is not None:
        native_scores_path = resolved_output_dir / "policyengine_native_scores.json"
        write_json_atomically(native_scores_path, native_scores_payload)
        replay_artifacts["policyengine_native_scores"] = _relative_to_root(
            native_scores_path,
            artifact_root,
        )
        if isinstance(native_scores_payload.get("summary"), dict):
            summaries["policyengine_native_scores"] = dict(
                native_scores_payload["summary"]
            )

    if not summaries:
        raise ValueError(
            "Stage 9 replay did not produce evidence. Supply precomputed evidence "
            "or a baseline dataset for native scoring."
        )

    evidence_path = resolved_output_dir / "evidence_manifest.json"
    replay_artifacts["validation_evidence"] = _relative_to_root(
        evidence_path,
        artifact_root,
    )
    replay_manifest_payload["artifacts"] = replay_artifacts
    replay_manifest_payload.update(summaries)
    replay_manifest_payload["stage9_replay"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_artifact_dir": str(artifact_root),
        "source_manifest": str(manifest_path),
        "source_policyengine_dataset": _relative_to_root(dataset_path, artifact_root),
        "output_dir": _relative_to_root(resolved_output_dir, artifact_root),
    }
    write_json_atomically(
        evidence_path,
        build_us_validation_evidence_manifest(
            artifact_root,
            manifest_payload=replay_manifest_payload,
        ),
    )
    replay_manifest_path = resolved_output_dir / "replay_manifest.json"
    write_json_atomically(replay_manifest_path, replay_manifest_payload)
    return USStage9ReplayResult(
        output_dir=resolved_output_dir,
        replay_manifest=replay_manifest_path,
        validation_evidence=evidence_path,
        policyengine_harness=harness_path,
        policyengine_native_scores=native_scores_path,
    )


def _validated_stage8_dataset_path(
    artifact_root: Path,
    manifest: dict[str, Any],
) -> Path:
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_value = artifacts.get("policyengine_dataset")
    if not dataset_value:
        raise ValueError("Stage 8 policyengine_dataset artifact is not declared")
    dataset_path = Path(str(dataset_value))
    if not dataset_path.is_absolute():
        dataset_path = artifact_root / dataset_path
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Stage 8 dataset artifact is missing: {dataset_path}")

    stage_manifest_paths = dict(manifest.get("stage_output_manifests", {}))
    stage8_manifest_value = stage_manifest_paths.get("08_dataset_assembly")
    if not stage8_manifest_value:
        raise ValueError("Stage 8 output manifest is not declared")
    stage8_manifest_path = Path(str(stage8_manifest_value))
    if not stage8_manifest_path.is_absolute():
        stage8_manifest_path = artifact_root / stage8_manifest_path
    if not stage8_manifest_path.exists():
        raise FileNotFoundError(
            f"Stage 8 output manifest is missing: {stage8_manifest_path}"
        )
    stage8_manifest = json.loads(stage8_manifest_path.read_text())
    if stage8_manifest.get("lifecycleStatus") != "complete":
        raise ValueError("Stage 8 must be complete before Stage 9 replay")
    stage8_outputs = stage8_manifest.get("outputs")
    if isinstance(stage8_outputs, dict):
        serialized_dataset = stage8_outputs.get("policyengine_dataset")
        if isinstance(serialized_dataset, dict):
            output_path = serialized_dataset.get("path")
            if (
                output_path
                and _resolve_artifact_path(
                    artifact_root,
                    output_path,
                )
                != dataset_path
            ):
                raise ValueError(
                    "Stage 8 dataset output does not match the root manifest "
                    "policyengine_dataset artifact"
                )
    return dataset_path


def _resolve_replay_output_dir(
    artifact_root: Path,
    *,
    output_dir: str | Path | None,
    run_id: str | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        artifact_root
        / "stage_artifacts"
        / "09_validation_benchmarking"
        / "replays"
        / resolved_run_id
    )


def _load_optional_payload(
    value: str | Path | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    return json.loads(Path(value).expanduser().read_text())


def _relative_to_root(path: Path, artifact_root: Path) -> str:
    try:
        return str(path.relative_to(artifact_root))
    except ValueError:
        return str(path)


def _resolve_artifact_path(artifact_root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = artifact_root / path
    return path.expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rerun Stage 9 validation evidence against a saved Stage 8 dataset."
    )
    parser.add_argument("artifact_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--baseline-dataset")
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--period", type=int)
    parser.add_argument("--precomputed-policyengine-harness")
    parser.add_argument("--precomputed-policyengine-native-scores")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = replay_us_stage9_validation_benchmarking(
        args.artifact_dir,
        output_dir=args.output_dir,
        baseline_dataset=args.baseline_dataset,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        period=args.period,
        precomputed_policyengine_harness=args.precomputed_policyengine_harness,
        precomputed_policyengine_native_scores=args.precomputed_policyengine_native_scores,
        run_id=args.run_id,
        allow_overwrite=args.allow_overwrite,
    )
    print(result.validation_evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "USStage9ReplayResult",
    "main",
    "replay_us_stage9_validation_benchmarking",
]
