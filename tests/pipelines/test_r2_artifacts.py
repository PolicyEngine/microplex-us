"""Tests for R2 artifact archiving."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microplex_us.pipelines.r2_artifacts import (
    R2_ARCHIVE_MANIFEST_FILENAME,
    R2ArchiveConfig,
    append_archive_index_entry,
    build_archive_manifest,
    build_r2_object_key,
    upload_artifact_manifest_to_r2,
)


class MissingObjectError(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "404"}}
        super().__init__("missing")


class FakeS3Client:
    def __init__(self, *, existing_keys: set[str] | None = None) -> None:
        self.existing_keys = existing_keys or set()
        self.head_calls: list[tuple[str, str]] = []
        self.upload_calls: list[tuple[str, str, str]] = []

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        self.head_calls.append((Bucket, Key))
        if Key not in self.existing_keys:
            raise MissingObjectError()
        return {}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.upload_calls.append((filename, bucket, key))
        self.existing_keys.add(key)


def test_build_r2_object_key_normalizes_prefix() -> None:
    assert (
        build_r2_object_key("/microplex-us/artifacts/", "run-a", "scores.json")
        == "microplex-us/artifacts/run-a/scores.json"
    )


def test_build_archive_manifest_hashes_files_and_excludes_r2_sidecar(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "run-a"
    artifact_dir.mkdir()
    (artifact_dir / "scores.json").write_text('{"loss": 0.1}\n')
    (artifact_dir / "data").mkdir()
    (artifact_dir / "data" / "weights.npy").write_bytes(b"weights")
    (artifact_dir / R2_ARCHIVE_MANIFEST_FILENAME).write_text("{}")
    config = R2ArchiveConfig(
        bucket="microplex-artifacts",
        endpoint_url="https://example.r2.cloudflarestorage.com",
        prefix="experiments",
    )

    manifest = build_archive_manifest(artifact_dir, config)

    assert manifest["artifact_id"] == "run-a"
    assert manifest["file_count"] == 2
    files = {entry["path"]: entry for entry in manifest["files"]}
    assert files["scores.json"]["summary"] is True
    assert files["scores.json"]["object_key"] == "experiments/run-a/scores.json"
    assert len(files["scores.json"]["sha256"]) == 64
    assert "r2_archive_manifest.json" not in files
    assert files["data/weights.npy"]["summary"] is False


def test_upload_artifact_manifest_to_r2_uploads_files_and_sidecar(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "run-a"
    artifact_dir.mkdir()
    (artifact_dir / "scores.json").write_text('{"loss": 0.1}\n')
    (artifact_dir / "summary.md").write_text("# Run\n")
    config = R2ArchiveConfig(
        bucket="microplex-artifacts",
        endpoint_url="https://example.r2.cloudflarestorage.com",
        prefix="experiments",
    )
    client = FakeS3Client()

    manifest = upload_artifact_manifest_to_r2(
        artifact_dir,
        config,
        client=client,
        hash_files=False,
    )

    assert manifest["status"] == "uploaded"
    assert {entry["status"] for entry in manifest["files"]} == {"uploaded"}
    uploaded_keys = [key for _, _, key in client.upload_calls]
    assert "experiments/run-a/scores.json" in uploaded_keys
    assert "experiments/run-a/summary.md" in uploaded_keys
    assert "experiments/run-a/r2_archive_manifest.json" in uploaded_keys
    local_manifest = json.loads(
        (artifact_dir / R2_ARCHIVE_MANIFEST_FILENAME).read_text()
    )
    assert local_manifest["r2"]["bucket"] == "microplex-artifacts"


def test_upload_artifact_manifest_to_r2_skips_existing_objects(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "run-a"
    artifact_dir.mkdir()
    (artifact_dir / "scores.json").write_text('{"loss": 0.1}\n')
    config = R2ArchiveConfig(
        bucket="microplex-artifacts",
        endpoint_url="https://example.r2.cloudflarestorage.com",
        prefix="experiments",
    )
    client = FakeS3Client(existing_keys={"experiments/run-a/scores.json"})

    manifest = upload_artifact_manifest_to_r2(
        artifact_dir,
        config,
        client=client,
        hash_files=False,
    )

    assert manifest["files"][0]["status"] == "already_exists"
    uploaded_keys = [key for _, _, key in client.upload_calls]
    assert uploaded_keys == ["experiments/run-a/r2_archive_manifest.json"]


def test_append_archive_index_entry_records_compact_upload(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "run-a"
    artifact_dir.mkdir()
    (artifact_dir / "scores.json").write_text('{"loss": 0.1}\n')
    config = R2ArchiveConfig(
        bucket="microplex-artifacts",
        endpoint_url="https://example.r2.cloudflarestorage.com",
        prefix="experiments",
    )
    manifest = build_archive_manifest(artifact_dir, config, hash_files=False)

    index_path = append_archive_index_entry(
        tmp_path / "r2_archive_index.jsonl",
        manifest,
        pruned_local=True,
    )

    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert rows == [
        {
            "recorded_at": rows[0]["recorded_at"],
            "artifact_id": "run-a",
            "artifact_dir": str(artifact_dir.resolve()),
            "bucket": "microplex-artifacts",
            "prefix": "experiments",
            "manifest_object_key": "experiments/run-a/r2_archive_manifest.json",
            "file_count": 1,
            "total_bytes": 14,
            "status": None,
            "pruned_local": True,
        }
    ]


def test_r2_archive_config_from_env_uses_account_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MICROPLEX_R2_BUCKET", "microplex-artifacts")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "abc123")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")

    config = R2ArchiveConfig.from_env()

    assert config.endpoint_url == "https://abc123.r2.cloudflarestorage.com"
    assert config.access_key_id == "key"
    assert config.secret_access_key == "secret"
