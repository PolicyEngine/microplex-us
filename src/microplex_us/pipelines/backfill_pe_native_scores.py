"""Backfill PE-native broad-loss scores for historical US artifact bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microplex.targets import assert_valid_benchmark_artifact_manifest

from microplex_us.pipelines.index_db import rebuild_us_microplex_run_index
from microplex_us.pipelines.pe_native_scores import (
    compute_batch_us_pe_native_scores,
    compute_us_pe_native_scores,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint import (
    _refresh_checkpoint_data_flow_snapshot,
)
from microplex_us.pipelines.registry import (
    append_us_microplex_run_registry_entry,
    build_us_microplex_run_registry_entry,
)
from microplex_us.pipelines.stage_contracts import (
    get_us_stage_artifact_contract,
    resolve_us_stage_artifact_contract_path,
)


def discover_us_candidate_artifact_dirs(artifact_root: str | Path) -> tuple[Path, ...]:
    """Return saved US artifact bundle directories with a PE dataset and manifest."""

    root = Path(artifact_root)
    dataset_hint = get_us_stage_artifact_contract(
        "08_dataset_assembly",
        "policyengine_dataset",
    ).path_hint
    if dataset_hint is None:
        raise RuntimeError("Stage 8 policyengine_dataset artifact has no path hint")
    return tuple(
        sorted(
            path.parent
            for path in root.rglob(dataset_hint)
            if (path.parent / "manifest.json").exists()
        )
    )


def backfill_us_pe_native_scores_bundle(
    artifact_dir: str | Path,
    *,
    baseline_dataset: str | Path | None = None,
    force: bool = False,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> Path:
    """Backfill PE-native broad-loss sidecar + manifest summary for one bundle."""

    bundle_dir = Path(artifact_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_name = artifacts.get("policyengine_dataset")
    if not dataset_name:
        raise ValueError(f"{bundle_dir} does not declare a policyengine_dataset artifact")

    native_sidecar_path = resolve_us_stage_artifact_contract_path(
        bundle_dir,
        "09_validation_benchmarking",
        "policyengine_native_scores",
    )
    if native_sidecar_path.exists() and not force:
        payload = json.loads(native_sidecar_path.read_text())
    else:
        resolved_baseline = _resolve_baseline_dataset(
            manifest,
            baseline_dataset=baseline_dataset,
        )
        payload = compute_us_pe_native_scores(
            candidate_dataset_path=bundle_dir / dataset_name,
            baseline_dataset_path=resolved_baseline,
            period=int(manifest.get("config", {}).get("policyengine_dataset_year", 2024)),
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
    return _write_native_scores_payload_to_bundle(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        payload=payload,
    )


def backfill_us_pe_native_scores_bundles(
    artifact_dirs: list[str | Path] | tuple[str | Path, ...],
    *,
    baseline_dataset: str | Path | None = None,
    force: bool = False,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    rebuild_registry: bool = True,
) -> list[Path]:
    """Backfill PE-native scores for a batch of saved bundles with grouped batch scoring."""

    if not artifact_dirs:
        return []

    manifest_paths: list[Path] = []
    grouped_pending: dict[Path, list[tuple[Path, Path, dict[str, object], Path]]] = {}

    for artifact_dir in artifact_dirs:
        bundle_dir = Path(artifact_dir)
        manifest_path = bundle_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest_paths.append(manifest_path)

        native_sidecar_path = resolve_us_stage_artifact_contract_path(
            bundle_dir,
            "09_validation_benchmarking",
            "policyengine_native_scores",
        )
        if native_sidecar_path.exists() and not force:
            _write_native_scores_payload_to_bundle(
                bundle_dir=bundle_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                payload=json.loads(native_sidecar_path.read_text()),
            )
            continue

        dataset_name = dict(manifest.get("artifacts", {})).get("policyengine_dataset")
        if not dataset_name:
            raise ValueError(
                f"{bundle_dir} does not declare a policyengine_dataset artifact"
            )
        resolved_baseline = _resolve_baseline_dataset(
            manifest,
            baseline_dataset=baseline_dataset,
        )
        grouped_pending.setdefault(resolved_baseline, []).append(
            (bundle_dir, manifest_path, manifest, bundle_dir / dataset_name)
        )

    for resolved_baseline, rows in grouped_pending.items():
        payloads = compute_batch_us_pe_native_scores(
            candidate_dataset_paths=[candidate_path for *_rest, candidate_path in rows],
            baseline_dataset_path=resolved_baseline,
            period=int(rows[0][2].get("config", {}).get("policyengine_dataset_year", 2024)),
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
        if len(payloads) != len(rows):
            raise ValueError(
                "PE-native batch backfill returned a different number of payloads than bundles"
            )
        for (bundle_dir, manifest_path, manifest, _candidate_path), payload in zip(
            rows,
            payloads,
            strict=True,
        ):
            _write_native_scores_payload_to_bundle(
                bundle_dir=bundle_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                payload=payload,
            )

    if rebuild_registry and manifest_paths:
        manifest_groups: dict[Path, list[Path]] = {}
        for manifest_path in manifest_paths:
            manifest_groups.setdefault(manifest_path.parent.parent, []).append(manifest_path)
        for artifact_root, root_manifest_paths in manifest_groups.items():
            rebuild_us_pe_native_run_registry(
                artifact_root,
                manifest_paths=root_manifest_paths,
            )
    return manifest_paths


def backfill_us_pe_native_scores_root(
    artifact_root: str | Path,
    *,
    baseline_dataset: str | Path | None = None,
    force: bool = False,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    rebuild_registry: bool = True,
) -> list[Path]:
    """Backfill every artifact bundle under one saved-output root."""

    manifest_paths: list[Path] = []
    for artifact_dir in discover_us_candidate_artifact_dirs(artifact_root):
        manifest_paths.append(
            backfill_us_pe_native_scores_bundle(
                artifact_dir,
                baseline_dataset=baseline_dataset,
                force=force,
                policyengine_us_data_repo=policyengine_us_data_repo,
                policyengine_us_data_python=policyengine_us_data_python,
            )
        )
    if rebuild_registry and manifest_paths:
        rebuild_us_pe_native_run_registry(artifact_root, manifest_paths=manifest_paths)
    return manifest_paths


def _write_native_scores_payload_to_bundle(
    *,
    bundle_dir: Path,
    manifest_path: Path,
    manifest: dict,
    payload: dict,
) -> Path:
    native_sidecar_path = resolve_us_stage_artifact_contract_path(
        bundle_dir,
        "09_validation_benchmarking",
        "policyengine_native_scores",
    )
    native_sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    artifacts = dict(manifest.get("artifacts", {}))
    artifacts["policyengine_native_scores"] = str(
        native_sidecar_path.relative_to(bundle_dir)
    )
    manifest["artifacts"] = artifacts
    manifest["policyengine_native_scores"] = dict(payload.get("summary", {}))
    if "run_registry" in manifest:
        manifest["run_registry"]["default_frontier_metric"] = (
            "enhanced_cps_native_loss_delta"
        )

    _refresh_checkpoint_data_flow_snapshot(bundle_dir, manifest)
    assert_valid_benchmark_artifact_manifest(
        manifest,
        artifact_dir=bundle_dir,
        manifest_path=manifest_path,
        summary_section=(
            "policyengine_harness"
            if manifest.get("policyengine_harness") is not None
            else None
        ),
        required_artifact_keys=("policyengine_dataset", "policyengine_native_scores"),
        required_summary_keys=(
            (
                "candidate_mean_abs_relative_error",
                "baseline_mean_abs_relative_error",
                "mean_abs_relative_error_delta",
            )
            if manifest.get("policyengine_harness") is not None
            else ()
        ),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def rebuild_us_pe_native_run_registry(
    artifact_root: str | Path,
    *,
    manifest_paths: list[Path] | tuple[Path, ...] | None = None,
) -> Path:
    """Rebuild one run_registry.jsonl from saved manifests under one artifact root."""

    root = Path(artifact_root)
    registry_path = root / "run_registry.jsonl"
    if registry_path.exists():
        registry_path.unlink()

    manifests = (
        list(manifest_paths)
        if manifest_paths is not None
        else [path for path in root.rglob("manifest.json")]
    )
    manifest_rows: list[tuple[str, Path, dict]] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("policyengine_harness") is None
            and manifest.get("policyengine_native_scores") is None
        ):
            continue
        manifest_rows.append(
            (
                str(manifest.get("created_at", "")),
                manifest_path,
                manifest,
            )
        )

    recorded_entries = []
    for _, manifest_path, manifest in sorted(manifest_rows, key=lambda item: item[0]):
        bundle_dir = manifest_path.parent
        harness_path = _resolve_optional_artifact_path(
            bundle_dir,
            manifest.get("artifacts", {}).get("policyengine_harness"),
        )
        harness_payload = (
            json.loads(harness_path.read_text()) if harness_path is not None else None
        )
        recorded = append_us_microplex_run_registry_entry(
            registry_path,
            build_us_microplex_run_registry_entry(
                artifact_dir=bundle_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                policyengine_harness_path=harness_path,
                policyengine_harness_payload=harness_payload,
            ),
        )
        recorded_entries.append((manifest_path, manifest, recorded))

    index_path = (
        rebuild_us_microplex_run_index(root, registry_path=registry_path)
        if recorded_entries
        else root / "run_index.duckdb"
    )

    for manifest_path, manifest, recorded in recorded_entries:
        manifest["run_registry"] = {
            "path": str(registry_path),
            "artifact_id": recorded.artifact_id,
            "improved_candidate_frontier": recorded.improved_candidate_frontier,
            "improved_delta_frontier": recorded.improved_delta_frontier,
            "improved_composite_frontier": recorded.improved_composite_frontier,
            "improved_native_frontier": recorded.improved_native_frontier,
            "default_frontier_metric": (
                "enhanced_cps_native_loss_delta"
                if manifest.get("policyengine_native_scores") is not None
                else "candidate_composite_parity_loss"
            ),
        }
        manifest["run_index"] = {
            "path": str(index_path),
            "artifact_id": recorded.artifact_id,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return registry_path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for historical PE-native score backfill."""

    parser = argparse.ArgumentParser(
        description="Backfill PE-native broad-loss scores for US artifact bundles."
    )
    parser.add_argument(
        "artifact_root",
        nargs="?",
        default="artifacts",
        help="Artifact output root to scan (defaults to ./artifacts).",
    )
    parser.add_argument("--baseline-dataset")
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute native scores even if a sidecar already exists.",
    )
    parser.add_argument(
        "--skip-registry-rebuild",
        action="store_true",
        help="Do not rebuild run_registry.jsonl / run_index.duckdb after backfill.",
    )
    args = parser.parse_args(argv)

    manifest_paths = backfill_us_pe_native_scores_root(
        args.artifact_root,
        baseline_dataset=args.baseline_dataset,
        force=args.force,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
        rebuild_registry=not args.skip_registry_rebuild,
    )
    print(
        json.dumps(
            {
                "artifact_root": str(Path(args.artifact_root).resolve()),
                "backfilled_count": len(manifest_paths),
                "manifest_paths": [str(path) for path in manifest_paths],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_baseline_dataset(
    manifest: dict,
    *,
    baseline_dataset: str | Path | None = None,
) -> Path:
    if baseline_dataset is not None:
        return Path(baseline_dataset).expanduser().resolve()
    config = dict(manifest.get("config", {}))
    configured = config.get("policyengine_baseline_dataset")
    if not configured:
        raise ValueError("Manifest does not include policyengine_baseline_dataset")
    return Path(configured).expanduser().resolve()


def _resolve_optional_artifact_path(
    bundle_dir: Path,
    artifact_name: str | None,
) -> Path | None:
    if not artifact_name:
        return None
    path = bundle_dir / artifact_name
    if not path.exists():
        return None
    return path


if __name__ == "__main__":
    raise SystemExit(main())
