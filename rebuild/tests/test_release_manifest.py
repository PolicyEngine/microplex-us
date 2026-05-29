"""Tests for the Microplex release-manifest emitter.

Core structure checks run with stdlib only. Schema validation against the real
policyengine-bundles `data-release-manifest.schema.json` runs when jsonschema and
the bundles repo are available (skipped otherwise, so the fast suite stays fast).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mp_rebuild.release_manifest import (
    DEFAULT_BUNDLES_SCHEMA,
    build_release_manifest,
    sha256_and_size,
)

SAMPLE_BUILD_MANIFEST = {
    "manifest_version": 1,
    "build": {
        "id": "mp-national-canonical-2024",
        "engine": "microplex-us",
        "created_at": "2026-05-29T12:00:00Z",
        "code_ref": {"repo": "PolicyEngine/microplex-us", "git_sha": "abc123", "dirty": False},
        "environment": {"key_packages": {"policyengine-us": "1.715.1", "policyengine-core": "3.26.1"}},
        "reproduce": {"command": "uv run ... --config national_canonical"},
    },
}


def _manifest_for(tmp_path: Path):
    h5 = tmp_path / "mp_300k_2024.h5"
    h5.write_bytes(b"not a real h5, just bytes for hashing" * 100)
    manifest = build_release_manifest(
        data_package_name="microplex-us",
        data_package_version="0.1.0",
        artifacts={"mp_300k_2024": h5},
        repo_id="policyengine/microplex-us",
        revision="abc123",
        compatible_model_packages=[("policyengine-us", ">=1.715,<2")],
        compatible_core_packages=[("policyengine-core", ">=3.26,<4")],
        default_datasets={"us": "mp_300k_2024"},
        build_manifest=SAMPLE_BUILD_MANIFEST,
        certified=True,
    )
    return manifest, h5


def test_required_fields_and_data_package(tmp_path):
    manifest, _ = _manifest_for(tmp_path)
    assert manifest["schema_version"] == 1
    assert manifest["data_package"] == {"name": "microplex-us", "version": "0.1.0"}
    assert manifest["default_datasets"] == {"us": "mp_300k_2024"}


def test_artifact_sha256_matches_file(tmp_path):
    manifest, h5 = _manifest_for(tmp_path)
    art = manifest["artifacts"]["mp_300k_2024"]
    expected_sha, expected_size = sha256_and_size(h5)
    assert art["sha256"] == expected_sha == hashlib.sha256(h5.read_bytes()).hexdigest()
    assert art["size_bytes"] == expected_size
    assert art["status"] == "certified"
    assert art["uri"] == "hf://policyengine/microplex-us/mp_300k_2024.h5"
    assert art["repo_id"] == "policyengine/microplex-us"


def test_compatibility_specifiers(tmp_path):
    manifest, _ = _manifest_for(tmp_path)
    assert manifest["compatible_model_packages"] == [
        {"name": "policyengine-us", "specifier": ">=1.715,<2"}
    ]
    assert manifest["compatible_core_packages"][0]["name"] == "policyengine-core"


def test_build_provenance_fed_from_build_manifest(tmp_path):
    manifest, _ = _manifest_for(tmp_path)
    build = manifest["build"]
    assert build["build_id"] == "mp-national-canonical-2024"
    assert build["metadata"]["model_package_version"] == "1.715.1"
    assert build["metadata"]["core_package_version"] == "3.26.1"
    assert build["metadata"]["engine"] == "microplex-us"


def test_validates_against_bundles_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    if not DEFAULT_BUNDLES_SCHEMA.exists():
        pytest.skip(f"bundles schema not present at {DEFAULT_BUNDLES_SCHEMA}")
    import json

    schema = json.loads(DEFAULT_BUNDLES_SCHEMA.read_text())
    manifest, _ = _manifest_for(tmp_path)
    jsonschema.validate(manifest, schema)  # raises if non-conforming
