"""Pinned benchmark manifests for Microplex replacement artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_PE_US_DATA_REPO = Path.home() / "PolicyEngine" / "policyengine-us-data"
FROZEN_PRODUCTION_ECPS_CERTIFICATE_TYPE = "frozen_production_ecps_baseline"
FROZEN_PRODUCTION_ECPS_PERIOD = 2024
FROZEN_PRODUCTION_ECPS_BASELINE_SHA256 = (
    "7af7026224f84cb6a91743fd8fa7ac506bad8c78e011fa58b6901894db4b4290"
)
FROZEN_PRODUCTION_ECPS_TARGET_DB_SHA256 = (
    "5d14671156c36cd7fff680d5c4d77ec7fb2026ea866b1e12378d9e9c9fb803dc"
)
FROZEN_PRODUCTION_ECPS_TARGET_PROFILE = "pe_native_broad"
FROZEN_PRODUCTION_ECPS_TARGET_SCOPE = "all"
FROZEN_PRODUCTION_ECPS_TARGET_COUNT = 3701
FROZEN_PRODUCTION_ECPS_TARGET_NAMES_SHA256 = (
    "a49a85a021ef65d5cd5b26d6d605c726ea5ca191ec98d9b5d9cc8b7d5665c25f"
)
FROZEN_PRODUCTION_ECPS_SCORING_CONFIG_SHA256 = (
    "3e67b0ca1f869e4c68f7eba513517b7d4c8dd9aaa195b98c51c100fe65dbabde"
)
FROZEN_PRODUCTION_ECPS_BASELINE_ENHANCED_CPS_NATIVE_LOSS = (
    0.0558541199034061
)
FROZEN_PRODUCTION_ECPS_BASELINE_HOLDOUT_LOSS = 0.01266396784689227
FROZEN_PRODUCTION_ECPS_BASELINE_UNWEIGHTED_MSRE = 3.4642345028776615
FROZEN_PRODUCTION_ECPS_RESOURCE_NAME = (
    "frozen_production_ecps_2024_benchmark_manifest.json"
)
FROZEN_PRODUCTION_ECPS_REQUIRED_EVIDENCE = {
    "certificate_type": FROZEN_PRODUCTION_ECPS_CERTIFICATE_TYPE,
    "period": FROZEN_PRODUCTION_ECPS_PERIOD,
    "baseline_dataset.sha256": FROZEN_PRODUCTION_ECPS_BASELINE_SHA256,
    "target_db.sha256": FROZEN_PRODUCTION_ECPS_TARGET_DB_SHA256,
    "target_surface.target_profile": FROZEN_PRODUCTION_ECPS_TARGET_PROFILE,
    "target_surface.target_scope": FROZEN_PRODUCTION_ECPS_TARGET_SCOPE,
}


def load_frozen_production_ecps_benchmark_manifest() -> dict[str, Any]:
    """Load the source-controlled 2024 production eCPS benchmark manifest."""

    payload = json.loads(_frozen_production_ecps_resource_bytes().decode())
    _assert_manifest_uses_frozen_production_pins(payload)
    return payload


def frozen_production_ecps_benchmark_manifest_descriptor() -> dict[str, Any]:
    """Return file-like evidence for the packaged production eCPS manifest."""

    payload = _frozen_production_ecps_resource_bytes()
    return {
        "path": (
            "package:microplex_us.pipelines/"
            f"{FROZEN_PRODUCTION_ECPS_RESOURCE_NAME}"
        ),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "packaged_default": True,
    }


def _frozen_production_ecps_resource_bytes() -> bytes:
    return (
        importlib.resources.files(__package__)
        .joinpath(FROZEN_PRODUCTION_ECPS_RESOURCE_NAME)
        .read_bytes()
    )


def build_mp_benchmark_manifest(
    *,
    baseline_dataset_path: str | Path,
    target_db_path: str | Path,
    period: int = 2024,
    target_profile: str = "pe_native_broad",
    target_scope: str = "all",
    target_count: int,
    target_names_sha256: str,
    scoring_config_sha256: str,
    baseline_enhanced_cps_native_loss: float | None = None,
    baseline_holdout_loss: float | None = None,
    baseline_unweighted_msre: float | None = None,
    certificate_type: str = "frozen_production_ecps_baseline",
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_commit: str | None = None,
    policyengine_us_version: str | None = None,
    allow_dirty_policyengine_us_data: bool = False,
    enforce_production_pins: bool = True,
) -> dict[str, Any]:
    """Build the frozen comparison manifest required by MP release gates."""

    baseline_dataset = _file_descriptor(Path(baseline_dataset_path).expanduser())
    target_db = _file_descriptor(Path(target_db_path).expanduser())
    repo_path = (
        Path(policyengine_us_data_repo).expanduser()
        if policyengine_us_data_repo is not None
        else _DEFAULT_PE_US_DATA_REPO
    )
    repo_descriptor = _policyengine_us_data_descriptor(
        repo_path,
        explicit_commit=policyengine_us_data_commit,
        allow_dirty=allow_dirty_policyengine_us_data,
    )
    version = policyengine_us_version or _installed_policyengine_us_version()
    baseline_metrics = _baseline_metrics_descriptor(
        certificate_type=certificate_type,
        baseline_enhanced_cps_native_loss=baseline_enhanced_cps_native_loss,
        baseline_holdout_loss=baseline_holdout_loss,
        baseline_unweighted_msre=baseline_unweighted_msre,
    )
    manifest = {
        "schema_version": 1,
        "certificate_type": str(certificate_type),
        "generated_at": datetime.now(UTC).isoformat(),
        "period": int(period),
        "target_profile": str(target_profile),
        "target_scope": str(target_scope),
        "target_surface": {
            "target_profile": str(target_profile),
            "target_scope": str(target_scope),
            "target_count": int(target_count),
            "target_names_sha256": str(target_names_sha256),
        },
        "scoring_config": {"sha256": str(scoring_config_sha256)},
        "baseline_metrics": baseline_metrics,
        "baseline_dataset": baseline_dataset,
        "policyengine_us_data": repo_descriptor,
        "policyengine_us": {"version": version},
        "target_db": target_db,
    }
    if enforce_production_pins:
        _assert_manifest_uses_frozen_production_pins(manifest)
    return manifest


def _baseline_metrics_descriptor(
    *,
    certificate_type: str,
    baseline_enhanced_cps_native_loss: float | None,
    baseline_holdout_loss: float | None,
    baseline_unweighted_msre: float | None,
) -> dict[str, float]:
    metric_values = {
        "baseline_enhanced_cps_native_loss": baseline_enhanced_cps_native_loss,
        "baseline_holdout_loss": baseline_holdout_loss,
        "baseline_unweighted_msre": baseline_unweighted_msre,
    }
    missing = [
        name
        for name, value in metric_values.items()
        if value is None
    ]
    if missing and certificate_type == FROZEN_PRODUCTION_ECPS_CERTIFICATE_TYPE:
        raise ValueError(
            "frozen production eCPS benchmark manifests must pin baseline "
            "metrics: " + ", ".join(missing)
        )
    return {
        name: float(value)
        for name, value in metric_values.items()
        if value is not None
    }


def write_mp_benchmark_manifest(
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Write a pinned benchmark manifest JSON file."""

    path = Path(output_path).expanduser()
    payload = build_mp_benchmark_manifest(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _policyengine_us_data_descriptor(
    repo_path: Path,
    *,
    explicit_commit: str | None,
    allow_dirty: bool,
) -> dict[str, Any]:
    repo = repo_path.resolve()
    commit = explicit_commit or _git_output(repo, "rev-parse", "HEAD")
    dirty = None if explicit_commit is not None else _git_dirty(repo)
    if dirty and not allow_dirty:
        raise ValueError(
            "policyengine-us-data repo has uncommitted changes; commit or pass "
            "--allow-dirty-policyengine-us-data to make the dirty state explicit"
        )
    descriptor: dict[str, Any] = {
        "repo": str(repo),
        "commit": commit,
    }
    if dirty is not None:
        descriptor["dirty"] = dirty
    return descriptor


def _installed_policyengine_us_version() -> str:
    try:
        return importlib.metadata.version("policyengine-us")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(
            "policyengine-us is not installed; pass --policyengine-us-version"
        ) from exc


def _file_descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"benchmark file not found: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_production_pin_mismatches(
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return mismatches against the hard-pinned production eCPS surface."""

    mismatches: list[dict[str, Any]] = []
    certificate_type = evidence.get("certificate_type")
    if certificate_type != FROZEN_PRODUCTION_ECPS_CERTIFICATE_TYPE:
        return mismatches
    for field, expected in FROZEN_PRODUCTION_ECPS_REQUIRED_EVIDENCE.items():
        actual = evidence.get(field)
        if actual is None:
            continue
        if str(actual) != str(expected):
            mismatches.append(
                {
                    "field": field,
                    "expected_production_pin": expected,
                    "actual": actual,
                }
            )
    return mismatches


def _assert_manifest_uses_frozen_production_pins(
    manifest: dict[str, Any],
) -> None:
    evidence = {
        "certificate_type": manifest.get("certificate_type"),
        "period": manifest.get("period"),
        "baseline_dataset.sha256": (manifest.get("baseline_dataset") or {}).get(
            "sha256"
        ),
        "target_db.sha256": (manifest.get("target_db") or {}).get("sha256"),
        "target_surface.target_profile": (
            manifest.get("target_surface") or {}
        ).get("target_profile"),
        "target_surface.target_scope": (
            manifest.get("target_surface") or {}
        ).get("target_scope"),
    }
    mismatches = frozen_production_pin_mismatches(evidence)
    if mismatches:
        details = ", ".join(
            f"{item['field']}={item['actual']!r} "
            f"(expected {item['expected_production_pin']!r})"
            for item in mismatches
        )
        raise ValueError(
            "frozen production eCPS benchmark manifest does not use the "
            f"release-pinned baseline/target surface: {details}"
        )


def _git_output(repo_path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"git {' '.join(args)} failed for {repo_path}: {detail}")
    return completed.stdout.strip()


def _git_dirty(repo_path: Path) -> bool:
    return bool(_git_output(repo_path, "status", "--porcelain"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a pinned benchmark manifest for MP replacement gates."
    )
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--target-db", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--target-profile", default="pe_native_broad")
    parser.add_argument("--target-scope", default="all")
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--target-names-sha256", required=True)
    parser.add_argument("--scoring-config-sha256", required=True)
    parser.add_argument("--baseline-enhanced-cps-native-loss", type=float, required=True)
    parser.add_argument("--baseline-holdout-loss", type=float, required=True)
    parser.add_argument("--baseline-unweighted-msre", type=float, required=True)
    parser.add_argument(
        "--certificate-type",
        default="frozen_production_ecps_baseline",
    )
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-commit")
    parser.add_argument("--policyengine-us-version")
    parser.add_argument(
        "--allow-dirty-policyengine-us-data",
        action="store_true",
        help=(
            "Allow a dirty policyengine-us-data repo and record that dirty state "
            "in the manifest."
        ),
    )
    parser.add_argument(
        "--allow-noncanonical-production-pins",
        action="store_true",
        help=(
            "Allow writing an experimental manifest whose frozen-production "
            "fields do not match the canonical production eCPS baseline, target "
            "DB, and all-target surface. Release gates still reject it."
        ),
    )
    args = parser.parse_args(argv)

    written = write_mp_benchmark_manifest(
        args.output_json,
        baseline_dataset_path=args.baseline_dataset,
        target_db_path=args.target_db,
        period=args.period,
        target_profile=args.target_profile,
        target_scope=args.target_scope,
        target_count=args.target_count,
        target_names_sha256=args.target_names_sha256,
        scoring_config_sha256=args.scoring_config_sha256,
        baseline_enhanced_cps_native_loss=args.baseline_enhanced_cps_native_loss,
        baseline_holdout_loss=args.baseline_holdout_loss,
        baseline_unweighted_msre=args.baseline_unweighted_msre,
        certificate_type=args.certificate_type,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_commit=args.policyengine_us_data_commit,
        policyengine_us_version=args.policyengine_us_version,
        allow_dirty_policyengine_us_data=args.allow_dirty_policyengine_us_data,
        enforce_production_pins=not args.allow_noncanonical_production_pins,
    )
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
