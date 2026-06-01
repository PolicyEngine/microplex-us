"""Backfill PE rebuild native-audit sidecars for historical US artifact bundles."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from microplex.targets import assert_valid_benchmark_artifact_manifest

from microplex_us.pipelines.backfill_pe_native_scores import (
    discover_us_candidate_artifact_dirs,
)
from microplex_us.pipelines.pe_native_scores import (
    compute_batch_us_pe_native_support_audits,
    compute_batch_us_pe_native_target_deltas,
)
from microplex_us.pipelines.pe_us_data_rebuild_audit import (
    build_policyengine_us_data_rebuild_native_audit,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint import (
    _refresh_checkpoint_data_flow_snapshot,
)
from microplex_us.pipelines.stage_contracts import (
    get_us_stage_artifact_contract,
    resolve_us_stage_artifact_contract_path,
)


def backfill_us_pe_native_audit_bundle(
    artifact_dir: str | Path,
    *,
    force: bool = False,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> Path:
    """Backfill PE rebuild native-audit sidecar + manifest summary for one bundle."""

    bundle_dir = Path(artifact_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_name = artifacts.get("policyengine_dataset")
    if not dataset_name:
        raise ValueError(f"{bundle_dir} does not declare a policyengine_dataset artifact")

    native_scores_path = _resolve_required_native_scores_path(bundle_dir, artifacts)
    native_scores_payload = json.loads(native_scores_path.read_text())
    native_audit_path = resolve_us_stage_artifact_contract_path(
        bundle_dir,
        "09_validation_benchmarking",
        "policyengine_native_audit",
    )
    if native_audit_path.exists() and not force:
        payload = json.loads(native_audit_path.read_text())
    else:
        payload = build_policyengine_us_data_rebuild_native_audit(
            bundle_dir,
            manifest_payload=manifest,
            native_scores_payload=native_scores_payload,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
    return _write_native_audit_payload_to_bundle(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        payload=payload,
    )


def backfill_us_pe_native_audit_bundles(
    artifact_dirs: list[str | Path] | tuple[str | Path, ...],
    *,
    force: bool = False,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> list[Path]:
    """Backfill PE rebuild native audits for a batch of saved bundles."""

    if not artifact_dirs:
        return []

    manifest_paths: list[Path] = []
    grouped_pending: dict[
        tuple[Path, int],
        list[tuple[Path, Path, dict, dict, Path]],
    ] = {}

    for artifact_dir in artifact_dirs:
        bundle_dir = Path(artifact_dir)
        manifest_path = bundle_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        artifacts = dict(manifest.get("artifacts", {}))
        dataset_name = artifacts.get("policyengine_dataset")
        if not dataset_name:
            raise ValueError(
                f"{bundle_dir} does not declare a policyengine_dataset artifact"
            )

        native_scores_path = _resolve_optional_native_scores_path(bundle_dir, artifacts)
        if native_scores_path is None:
            continue
        manifest_paths.append(manifest_path)
        native_scores_payload = json.loads(native_scores_path.read_text())
        native_audit_path = resolve_us_stage_artifact_contract_path(
            bundle_dir,
            "09_validation_benchmarking",
            "policyengine_native_audit",
        )
        if native_audit_path.exists() and not force:
            _write_native_audit_payload_to_bundle(
                bundle_dir=bundle_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                payload=json.loads(native_audit_path.read_text()),
            )
            continue

        period = int(
            native_scores_payload.get("period")
            or manifest.get("config", {}).get("policyengine_dataset_year", 2024)
        )
        baseline_dataset = _resolve_baseline_dataset(manifest)
        grouped_pending.setdefault((baseline_dataset, period), []).append(
            (
                bundle_dir,
                manifest_path,
                manifest,
                native_scores_payload,
                bundle_dir / str(dataset_name),
            )
        )

    for (baseline_dataset, period), rows in grouped_pending.items():
        candidate_dataset_paths = [candidate_path for *_rest, candidate_path in rows]
        with ThreadPoolExecutor(max_workers=2) as executor:
            target_future = executor.submit(
                compute_batch_us_pe_native_target_deltas,
                candidate_dataset_paths=candidate_dataset_paths,
                baseline_dataset_path=baseline_dataset,
                period=period,
                top_k=15,
                policyengine_us_data_repo=policyengine_us_data_repo,
                policyengine_us_data_python=policyengine_us_data_python,
            )
            support_future = executor.submit(
                compute_batch_us_pe_native_support_audits,
                candidate_dataset_paths=candidate_dataset_paths,
                baseline_dataset_path=baseline_dataset,
                period=period,
                policyengine_us_data_repo=policyengine_us_data_repo,
                policyengine_us_data_python=policyengine_us_data_python,
            )
            target_delta_payloads = target_future.result()
            support_audit_payloads = support_future.result()

        target_payload_by_candidate = {
            str(Path(payload["to_dataset"]).expanduser().resolve()): payload
            for payload in target_delta_payloads
        }
        support_payload_by_candidate = {
            str(Path(payload["candidate_dataset"]).expanduser().resolve()): payload
            for payload in support_audit_payloads
        }

        if len(target_payload_by_candidate) != len(rows):
            raise ValueError(
                "PE-native batch target-delta backfill returned a different number "
                "of payloads than bundles"
            )
        if len(support_payload_by_candidate) != len(rows):
            raise ValueError(
                "PE-native batch support-audit backfill returned a different number "
                "of payloads than bundles"
            )

        for (
            bundle_dir,
            manifest_path,
            manifest,
            native_scores_payload,
            candidate_dataset_path,
        ) in rows:
            candidate_key = str(candidate_dataset_path.expanduser().resolve())
            target_delta_payload = target_payload_by_candidate.get(candidate_key)
            support_audit_payload = support_payload_by_candidate.get(candidate_key)
            if target_delta_payload is None or support_audit_payload is None:
                raise ValueError(
                    "PE-native batch audit backfill did not return payloads for "
                    f"{candidate_dataset_path}"
                )
            payload = build_policyengine_us_data_rebuild_native_audit(
                bundle_dir,
                manifest_payload=manifest,
                native_scores_payload=native_scores_payload,
                target_delta_payload=target_delta_payload,
                support_audit_payload=support_audit_payload,
                policyengine_us_data_repo=policyengine_us_data_repo,
                policyengine_us_data_python=policyengine_us_data_python,
            )
            _write_native_audit_payload_to_bundle(
                bundle_dir=bundle_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                payload=payload,
            )

    return manifest_paths


def backfill_us_pe_native_audit_root(
    artifact_root: str | Path,
    *,
    force: bool = False,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> list[Path]:
    """Backfill every eligible artifact bundle under one saved-output root."""

    return backfill_us_pe_native_audit_bundles(
        discover_us_candidate_artifact_dirs(artifact_root),
        force=force,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )


def _write_native_audit_payload_to_bundle(
    *,
    bundle_dir: Path,
    manifest_path: Path,
    manifest: dict,
    payload: dict,
) -> Path:
    native_audit_path = resolve_us_stage_artifact_contract_path(
        bundle_dir,
        "09_validation_benchmarking",
        "policyengine_native_audit",
    )
    native_audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    artifacts = dict(manifest.get("artifacts", {}))
    artifacts["policyengine_native_audit"] = str(
        native_audit_path.relative_to(bundle_dir)
    )
    manifest["artifacts"] = artifacts
    manifest["policyengine_native_audit"] = dict(payload.get("verdictHints", {}))

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
        required_artifact_keys=(
            "policyengine_dataset",
            "policyengine_native_scores",
            "policyengine_native_audit",
        ),
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


def _resolve_required_native_scores_path(
    bundle_dir: Path,
    artifacts: dict,
) -> Path:
    path = _resolve_optional_native_scores_path(bundle_dir, artifacts)
    if path is not None:
        return path
    raise ValueError(
        f"{bundle_dir} is missing policyengine_native_scores.json; backfill native scores first"
    )


def _resolve_optional_native_scores_path(
    bundle_dir: Path,
    artifacts: dict,
) -> Path | None:
    artifact_name = (
        artifacts.get("policyengine_native_scores")
        or get_us_stage_artifact_contract(
            "09_validation_benchmarking",
            "policyengine_native_scores",
        ).path_hint
    )
    if artifact_name is None:
        return None
    path = bundle_dir / str(artifact_name)
    if path.exists():
        return path
    return None


def _resolve_baseline_dataset(manifest: dict) -> Path:
    config = dict(manifest.get("config", {}))
    configured = config.get("policyengine_baseline_dataset")
    if not configured:
        raise ValueError("Manifest does not include policyengine_baseline_dataset")
    return Path(str(configured)).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for historical PE rebuild native-audit backfill."""

    parser = argparse.ArgumentParser(
        description="Backfill PE rebuild native-audit sidecars for US artifact bundles.",
    )
    parser.add_argument(
        "artifact_root",
        nargs="?",
        default="artifacts",
        help="Artifact output root to scan (defaults to ./artifacts).",
    )
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute native audits even if a sidecar already exists.",
    )
    args = parser.parse_args(argv)

    manifest_paths = backfill_us_pe_native_audit_root(
        args.artifact_root,
        force=args.force,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
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


if __name__ == "__main__":
    raise SystemExit(main())
