"""Pinned benchmark manifests for Microplex replacement artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_PE_US_DATA_REPO = Path.home() / "PolicyEngine" / "policyengine-us-data"


def build_mp_benchmark_manifest(
    *,
    baseline_dataset_path: str | Path,
    target_db_path: str | Path,
    period: int = 2024,
    target_profile: str = "pe_native_broad",
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_commit: str | None = None,
    policyengine_us_version: str | None = None,
    allow_dirty_policyengine_us_data: bool = False,
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
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "period": int(period),
        "target_profile": str(target_profile),
        "baseline_dataset": baseline_dataset,
        "policyengine_us_data": repo_descriptor,
        "policyengine_us": {"version": version},
        "target_db": target_db,
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
    args = parser.parse_args(argv)

    written = write_mp_benchmark_manifest(
        args.output_json,
        baseline_dataset_path=args.baseline_dataset,
        target_db_path=args.target_db,
        period=args.period,
        target_profile=args.target_profile,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_commit=args.policyengine_us_data_commit,
        policyengine_us_version=args.policyengine_us_version,
        allow_dirty_policyengine_us_data=args.allow_dirty_policyengine_us_data,
    )
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
