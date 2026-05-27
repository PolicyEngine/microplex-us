"""Package mp-300k artifact-gate inputs for CI handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def package_mp300k_gate_inputs(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    candidate_dataset_path: str | Path | None = None,
    ecps_comparison_path: str | Path | None = None,
    runtime_smoke_path: str | Path | None = None,
    benchmark_manifest_path: str | Path | None = None,
    archive_name: str = "artifact.tar.gz",
) -> dict[str, Any]:
    """Package an artifact archive plus gate evidence for GitHub Actions.

    The output directory is intended to be uploaded as a single Actions artifact
    and consumed by ``mp300k-artifact-gates.yml`` through ``gate_inputs_artifact``.
    """

    artifact_root = Path(artifact_dir).expanduser()
    output_root = Path(output_dir).expanduser()
    manifest_path = artifact_root / "manifest.json"
    manifest = _load_manifest(manifest_path)
    candidate_dataset = _resolve_candidate_dataset_path(
        artifact_root,
        manifest,
        candidate_dataset_path,
    )
    if not candidate_dataset.exists():
        raise FileNotFoundError(f"candidate dataset not found: {candidate_dataset}")

    output_root.mkdir(parents=True, exist_ok=True)
    archive_path = output_root / archive_name
    stage_parent = output_root / ".staging"
    if stage_parent.exists():
        shutil.rmtree(stage_parent)
    stage_root = stage_parent / artifact_root.name
    stage_root.mkdir(parents=True)

    candidate_relpath = _candidate_archive_relpath(
        manifest,
        candidate_dataset=candidate_dataset,
        explicit_candidate_path=candidate_dataset_path,
    )
    staged_candidate = stage_root / candidate_relpath
    staged_candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_dataset, staged_candidate)

    staged_manifest = _manifest_for_archive(
        manifest,
        source_artifact_dir=artifact_root,
        source_candidate_dataset=candidate_dataset,
        candidate_relpath=candidate_relpath,
    )
    _write_json(stage_root / "manifest.json", staged_manifest)
    _write_archive(archive_path, stage_root)

    evidence = {
        "ecps_comparison": _copy_optional_evidence(
            ecps_comparison_path,
            output_root / "ecps_comparison.json",
        ),
        "runtime_smoke": _copy_optional_evidence(
            runtime_smoke_path,
            output_root / "runtime_smoke.json",
        ),
        "benchmark_manifest": _copy_optional_evidence(
            benchmark_manifest_path,
            output_root / "benchmark_manifest.json",
        ),
    }
    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_artifact_dir": str(artifact_root.resolve()),
        "source_manifest": _file_descriptor(manifest_path),
        "source_candidate_dataset": _file_descriptor(candidate_dataset),
        "artifact_archive": _file_descriptor(archive_path),
        "evidence": evidence,
        "workflow_call": {
            "uses": "./.github/workflows/mp300k-artifact-gates.yml",
            "with": {"gate_inputs_artifact": output_root.name},
        },
    }
    _write_json(output_root / "gate_inputs.json", metadata)
    shutil.rmtree(stage_parent)
    return metadata


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text())


def _resolve_candidate_dataset_path(
    artifact_root: Path,
    manifest: dict[str, Any],
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser()
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_name = artifacts.get("policyengine_dataset")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(
            "manifest.artifacts.policyengine_dataset is required when "
            "candidate_dataset_path is not supplied"
        )
    dataset_path = Path(dataset_name).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = artifact_root / dataset_path
    return dataset_path


def _candidate_archive_relpath(
    manifest: dict[str, Any],
    *,
    candidate_dataset: Path,
    explicit_candidate_path: str | Path | None,
) -> Path:
    if explicit_candidate_path is not None:
        return Path(candidate_dataset.name)
    dataset_name = dict(manifest.get("artifacts", {})).get("policyengine_dataset")
    if isinstance(dataset_name, str) and dataset_name:
        relpath = Path(dataset_name)
        if not relpath.is_absolute():
            return relpath
    return Path(candidate_dataset.name)


def _manifest_for_archive(
    manifest: dict[str, Any],
    *,
    source_artifact_dir: Path,
    source_candidate_dataset: Path,
    candidate_relpath: Path,
) -> dict[str, Any]:
    updated = dict(manifest)
    artifacts = dict(updated.get("artifacts", {}))
    artifacts["policyengine_dataset"] = str(candidate_relpath)
    updated["artifacts"] = artifacts
    updated["mp300k_gate_inputs"] = {
        "packaged_at": datetime.now(UTC).isoformat(),
        "source_artifact_dir": str(source_artifact_dir.resolve()),
        "source_candidate_dataset": str(source_candidate_dataset.resolve()),
    }
    return updated


def _copy_optional_evidence(
    source_path: str | Path | None,
    destination_path: Path,
) -> dict[str, Any] | None:
    if source_path is None:
        return None
    source = Path(source_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"evidence file not found: {source}")
    shutil.copy2(source, destination_path)
    return _file_descriptor(destination_path)


def _write_archive(archive_path: Path, stage_root: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(stage_root, arcname=stage_root.name, recursive=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package mp-300k artifact-gate inputs for CI."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-dataset")
    parser.add_argument("--ecps-comparison-json")
    parser.add_argument("--runtime-smoke-json")
    parser.add_argument("--benchmark-manifest")
    parser.add_argument("--archive-name", default="artifact.tar.gz")
    args = parser.parse_args(argv)

    package_mp300k_gate_inputs(
        args.artifact_dir,
        args.output_dir,
        candidate_dataset_path=args.candidate_dataset,
        ecps_comparison_path=args.ecps_comparison_json,
        runtime_smoke_path=args.runtime_smoke_json,
        benchmark_manifest_path=args.benchmark_manifest,
        archive_name=args.archive_name,
    )
    print(Path(args.output_dir).expanduser() / "gate_inputs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
