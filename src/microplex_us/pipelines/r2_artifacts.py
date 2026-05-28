"""Archive Microplex artifact directories to Cloudflare R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

R2_ARCHIVE_MANIFEST_FILENAME = "r2_archive_manifest.json"
R2_ARCHIVE_INDEX_FILENAME = "r2_archive_index.jsonl"
DEFAULT_R2_PREFIX = "microplex-us/artifacts"
DEFAULT_REGION = "auto"
SUMMARY_FILENAMES = frozenset(
    {
        "manifest.json",
        "summary.md",
        "scores.json",
        "fit_summary.json",
        "target_deltas_top50.json",
        "matrix_residual_drilldown_top100.json",
        "calibration_summary.json",
        "source_spine_composition.json",
        "support_audit.json",
        "run_manifest.json",
    }
)


@dataclass(frozen=True)
class R2ArchiveConfig:
    """R2 destination and credentials for one archive operation."""

    bucket: str
    endpoint_url: str
    prefix: str = DEFAULT_R2_PREFIX
    region: str = DEFAULT_REGION
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None

    @classmethod
    def from_env(cls) -> R2ArchiveConfig:
        """Build config from Microplex-specific env vars with R2/AWS fallbacks."""
        bucket = _first_env("MICROPLEX_R2_BUCKET", "R2_BUCKET", "AWS_BUCKET")
        if not bucket:
            raise ValueError(
                "Missing R2 bucket. Set MICROPLEX_R2_BUCKET or pass --bucket."
            )
        endpoint_url = _first_env("MICROPLEX_R2_ENDPOINT_URL", "R2_ENDPOINT_URL")
        account_id = _first_env("MICROPLEX_R2_ACCOUNT_ID", "CLOUDFLARE_ACCOUNT_ID")
        if not endpoint_url:
            if not account_id:
                raise ValueError(
                    "Missing R2 endpoint. Set MICROPLEX_R2_ENDPOINT_URL, "
                    "R2_ENDPOINT_URL, or CLOUDFLARE_ACCOUNT_ID."
                )
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        return cls(
            bucket=bucket,
            endpoint_url=endpoint_url,
            prefix=(
                _first_env("MICROPLEX_R2_PREFIX", "R2_PREFIX")
                or DEFAULT_R2_PREFIX
            ),
            region=_first_env("MICROPLEX_R2_REGION", "AWS_DEFAULT_REGION")
            or DEFAULT_REGION,
            access_key_id=_first_env(
                "MICROPLEX_R2_ACCESS_KEY_ID",
                "R2_ACCESS_KEY_ID",
                "AWS_ACCESS_KEY_ID",
            ),
            secret_access_key=_first_env(
                "MICROPLEX_R2_SECRET_ACCESS_KEY",
                "R2_SECRET_ACCESS_KEY",
                "AWS_SECRET_ACCESS_KEY",
            ),
            session_token=_first_env(
                "MICROPLEX_R2_SESSION_TOKEN",
                "R2_SESSION_TOKEN",
                "AWS_SESSION_TOKEN",
            ),
        )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_r2_prefix(value: str) -> str:
    """Normalize an R2 key prefix without changing internal separators."""
    return value.strip("/")


def build_r2_object_key(prefix: str, artifact_id: str, relative_path: str) -> str:
    """Return a stable R2 object key for one artifact file."""
    parts = [
        normalize_r2_prefix(prefix),
        artifact_id.strip("/"),
        relative_path.replace(os.sep, "/").strip("/"),
    ]
    return "/".join(part for part in parts if part)


def iter_artifact_files(artifact_dir: str | Path) -> list[Path]:
    """List regular artifact files, excluding the local R2 archive sidecar."""
    root = Path(artifact_dir)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != R2_ARCHIVE_MANIFEST_FILENAME
    )


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive_manifest(
    artifact_dir: str | Path,
    config: R2ArchiveConfig,
    *,
    artifact_id: str | None = None,
    hash_files: bool = True,
    status: str = "planned",
) -> dict[str, Any]:
    """Build the local manifest describing files and destination object keys."""
    root = Path(artifact_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Artifact directory not found: {root}")
    resolved_artifact_id = artifact_id or root.name
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in iter_artifact_files(root):
        relative_path = path.relative_to(root).as_posix()
        size_bytes = path.stat().st_size
        total_bytes += size_bytes
        entry: dict[str, Any] = {
            "path": relative_path,
            "size_bytes": size_bytes,
            "object_key": build_r2_object_key(
                config.prefix,
                resolved_artifact_id,
                relative_path,
            ),
            "status": status,
            "summary": path.name in SUMMARY_FILENAMES,
        }
        if hash_files:
            entry["sha256"] = file_sha256(path)
        files.append(entry)
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_id": resolved_artifact_id,
        "artifact_dir": str(root),
        "r2": {
            "bucket": config.bucket,
            "endpoint_url": config.endpoint_url,
            "prefix": normalize_r2_prefix(config.prefix),
            "region": config.region,
            "manifest_object_key": build_r2_object_key(
                config.prefix,
                resolved_artifact_id,
                R2_ARCHIVE_MANIFEST_FILENAME,
            ),
        },
        "summary_files": [
            entry["path"] for entry in files if bool(entry.get("summary"))
        ],
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def create_r2_s3_client(config: R2ArchiveConfig) -> Any:
    """Create a boto3 S3 client configured for Cloudflare R2."""
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - exercised by CLI environment.
        raise RuntimeError(
            "boto3 is required for R2 uploads. Install the optional extra with "
            "`uv sync --extra r2` or run through `uv run --extra r2 ...`."
        ) from error
    client_kwargs: dict[str, Any] = {
        "service_name": "s3",
        "endpoint_url": config.endpoint_url,
        "region_name": config.region,
    }
    if config.access_key_id is not None:
        client_kwargs["aws_access_key_id"] = config.access_key_id
    if config.secret_access_key is not None:
        client_kwargs["aws_secret_access_key"] = config.secret_access_key
    if config.session_token is not None:
        client_kwargs["aws_session_token"] = config.session_token
    return boto3.client(**client_kwargs)


def upload_artifact_manifest_to_r2(
    artifact_dir: str | Path,
    config: R2ArchiveConfig,
    *,
    artifact_id: str | None = None,
    client: Any | None = None,
    dry_run: bool = False,
    force: bool = False,
    hash_files: bool = True,
) -> dict[str, Any]:
    """Upload an artifact directory to R2 and write a local upload manifest."""
    root = Path(artifact_dir).resolve()
    manifest = build_archive_manifest(
        root,
        config,
        artifact_id=artifact_id,
        hash_files=hash_files,
        status="dry_run" if dry_run else "pending",
    )
    local_manifest_path = root / R2_ARCHIVE_MANIFEST_FILENAME
    if dry_run:
        _write_json(local_manifest_path, manifest)
        return manifest
    s3 = client or create_r2_s3_client(config)
    for entry in manifest["files"]:
        path = root / entry["path"]
        object_key = entry["object_key"]
        if not force and _object_exists(s3, config.bucket, object_key):
            entry["status"] = "already_exists"
            continue
        s3.upload_file(str(path), config.bucket, object_key)
        entry["status"] = "uploaded"
        entry["uploaded_at"] = datetime.now(UTC).isoformat()
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["status"] = "uploaded"
    _write_json(local_manifest_path, manifest)
    manifest_key = manifest["r2"]["manifest_object_key"]
    s3.upload_file(str(local_manifest_path), config.bucket, manifest_key)
    return manifest


def append_archive_index_entry(
    index_path: str | Path,
    manifest: dict[str, Any],
    *,
    pruned_local: bool = False,
) -> Path:
    """Append a compact archive record to a local JSONL index."""
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "artifact_id": manifest["artifact_id"],
        "artifact_dir": manifest["artifact_dir"],
        "bucket": manifest["r2"]["bucket"],
        "prefix": manifest["r2"]["prefix"],
        "manifest_object_key": manifest["r2"]["manifest_object_key"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "status": manifest.get("status"),
        "pruned_local": pruned_local,
    }
    with path.open("a") as file:
        file.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def _object_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # noqa: BLE001 - boto3 exposes provider-specific errors.
        response = getattr(error, "response", None)
        code = None
        if isinstance(response, dict):
            code = str(response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        # Some fakes and S3-compatible clients use a generic missing-object error.
        if error.__class__.__name__ in {"NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path)
    temp_path = resolved.with_suffix(resolved.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp_path.replace(resolved)


def _build_config_from_args(args: argparse.Namespace) -> R2ArchiveConfig:
    env_config: R2ArchiveConfig | None = None
    if args.bucket is None or args.endpoint_url is None:
        try:
            env_config = R2ArchiveConfig.from_env()
        except ValueError:
            if args.bucket is None or args.endpoint_url is None:
                raise
    bucket = args.bucket or (env_config.bucket if env_config is not None else None)
    endpoint_url = args.endpoint_url or (
        env_config.endpoint_url if env_config is not None else None
    )
    if bucket is None or endpoint_url is None:
        raise ValueError("Both bucket and endpoint URL are required.")
    return R2ArchiveConfig(
        bucket=bucket,
        endpoint_url=endpoint_url,
        prefix=args.prefix
        or (env_config.prefix if env_config is not None else DEFAULT_R2_PREFIX),
        region=args.region
        or (env_config.region if env_config is not None else DEFAULT_REGION),
        access_key_id=(
            args.access_key_id
            or (env_config.access_key_id if env_config is not None else None)
        ),
        secret_access_key=(
            args.secret_access_key
            or (env_config.secret_access_key if env_config is not None else None)
        ),
        session_token=(
            args.session_token
            or (env_config.session_token if env_config is not None else None)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Archive a Microplex artifact directory to Cloudflare R2."
    )
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--artifact-id", default=None)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--endpoint-url", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--access-key-id", default=None)
    parser.add_argument("--secret-access-key", default=None)
    parser.add_argument("--session-token", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the local archive manifest without uploading to R2.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload files even when an object with the same key already exists.",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA-256 file hashing when building the archive manifest.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help=(
            "Optional local JSONL archive index. Defaults to "
            "<artifact-parent>/r2_archive_index.jsonl when uploading."
        ),
    )
    parser.add_argument(
        "--mark-pruned-local",
        action="store_true",
        help="Mark the local archive-index row as pruned after external cleanup.",
    )
    args = parser.parse_args(argv)
    try:
        config = _build_config_from_args(args)
        manifest = upload_artifact_manifest_to_r2(
            args.artifact_dir,
            config,
            artifact_id=args.artifact_id,
            dry_run=args.dry_run,
            force=args.force,
            hash_files=not args.no_hash,
        )
    except Exception as error:  # noqa: BLE001 - CLI should report a concise failure.
        print(f"R2 archive failed: {error}", file=sys.stderr)
        return 1
    if not args.dry_run:
        index_path = args.index_path or args.artifact_dir.parent / R2_ARCHIVE_INDEX_FILENAME
        append_archive_index_entry(
            index_path,
            manifest,
            pruned_local=args.mark_pruned_local,
        )
    uploaded = sum(
        1
        for entry in manifest["files"]
        if entry["status"] in {"uploaded", "already_exists"}
    )
    mode = "planned" if args.dry_run else "archived"
    print(
        f"R2 artifact {mode}: {manifest['artifact_id']} "
        f"({uploaded}/{manifest['file_count']} files, "
        f"{manifest['total_bytes']} bytes)"
    )
    print(args.artifact_dir / R2_ARCHIVE_MANIFEST_FILENAME)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
