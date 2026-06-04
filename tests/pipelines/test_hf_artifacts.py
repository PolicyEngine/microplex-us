"""Tests for Hugging Face artifact publishing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microplex_us.pipelines import hf_artifacts
from microplex_us.pipelines.hf_artifacts import (
    HF_PUBLISH_MANIFEST_FILENAME,
    HuggingFacePublishConfig,
    build_hf_repo_path,
    dataset_repo_paths,
    diagnostics_repo_paths,
    publish_microplex_artifact_to_hf,
    smoke_published_hf_artifact,
)


class FakeHfApi:
    def __init__(self) -> None:
        self.commits: list[dict[str, Any]] = []
        self.files_by_repo: dict[str, list[str]] = {}

    def create_commit(self, **kwargs: Any) -> None:
        self.commits.append(kwargs)

    def list_repo_files(self, **kwargs: Any) -> list[str]:
        return self.files_by_repo[kwargs["repo_id"]]


def _fake_add(path_in_repo: str, local_path: Path) -> dict[str, str]:
    return {
        "path_in_repo": path_in_repo,
        "path_or_fileobj": str(local_path),
    }


def _fake_add_text(path_in_repo: str, text: str) -> dict[str, str]:
    return {
        "path_in_repo": path_in_repo,
        "path_or_fileobj": text,
    }


def _fake_add_bytes(path_in_repo: str, payload: dict[str, Any]) -> dict[str, str]:
    return _fake_add_text(path_in_repo, json.dumps(payload, sort_keys=True))


def _patch_operations(monkeypatch) -> None:
    monkeypatch.setattr(hf_artifacts, "_commit_add", _fake_add)
    monkeypatch.setattr(hf_artifacts, "_commit_add_text", _fake_add_text)
    monkeypatch.setattr(hf_artifacts, "_commit_add_bytes", _fake_add_bytes)


def _write_bundle(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "run-a"
    artifact_dir.mkdir()
    (artifact_dir / "policyengine_us.h5").write_bytes(b"h5")
    (artifact_dir / "policyengine_native_scores.json").write_text(
        json.dumps(
            {
                "summary": {
                    "candidate_enhanced_cps_native_loss": 0.3,
                    "enhanced_cps_native_loss_delta": -0.1,
                }
            }
        )
    )
    (artifact_dir / "pe_us_data_rebuild_native_audit.json").write_text("{}")
    (artifact_dir / "pe_native_target_diagnostics.json").write_text("{}")
    manifest = {
        "created_at": "2026-06-03T00:00:00+00:00",
        "artifacts": {
            "policyengine_dataset": "policyengine_us.h5",
            "policyengine_native_scores": "policyengine_native_scores.json",
            "policyengine_native_audit": "pe_us_data_rebuild_native_audit.json",
            "policyengine_native_target_diagnostics": (
                "pe_native_target_diagnostics.json"
            ),
        },
        "policyengine_native_scores": {
            "candidate_enhanced_cps_native_loss": 0.3,
            "enhanced_cps_native_loss_delta": -0.1,
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    return artifact_dir


def test_build_hf_repo_path_normalizes_parts() -> None:
    assert build_hf_repo_path("/runs/", "run-a", "/manifest.json") == (
        "runs/run-a/manifest.json"
    )


def test_diagnostics_repo_paths_use_stable_layout() -> None:
    paths = diagnostics_repo_paths("artifact/run-a", run_id="run-a")

    assert paths["manifest"] == "runs/run-a/manifest.json"
    assert paths["policyengine_native_scores"] == (
        "runs/run-a/policyengine_native_scores.json"
    )
    assert paths["policyengine_native_audit"] == (
        "runs/run-a/pe_us_data_rebuild_native_audit.json"
    )
    assert paths["policyengine_native_target_diagnostics"] == (
        "runs/run-a/pe_native_target_diagnostics.json"
    )
    assert paths["latest"] == "latest.json"
    assert paths["run_registry"] == "run_registry.jsonl"


def test_dataset_repo_paths_can_include_promoted_current_files() -> None:
    paths = dataset_repo_paths("artifact/run-a", run_id="run-a", promote=True)

    assert paths["policyengine_dataset"] == "staging/run-a/policyengine_us.h5"
    assert paths["manifest"] == "staging/run-a/manifest.json"
    assert paths["promoted_policyengine_dataset"] == "policyengine_us.h5"
    assert paths["promoted_manifest"] == "manifest.json"


def test_publish_diagnostics_dry_run_writes_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_operations(monkeypatch)
    artifact_dir = _write_bundle(tmp_path)
    config = HuggingFacePublishConfig(
        diagnostics_repo="policyengine/microplex-us-diagnostics",
        dataset_repo="policyengine/microplex-us-deployed-datasets",
        token="token",
    )

    result = publish_microplex_artifact_to_hf(
        artifact_dir,
        config,
        dry_run=True,
        registry_loader=lambda _config: "",
    )

    assert result["status"] == "dry_run"
    assert result["diagnostics"]["paths"]["latest"] == "latest.json"
    registry_entry = result["diagnostics"]["run_registry_entry"]
    assert registry_entry["run_id"] == "run-a"
    assert registry_entry["candidate_enhanced_cps_native_loss"] == 0.3
    local_manifest = json.loads(
        (artifact_dir / HF_PUBLISH_MANIFEST_FILENAME).read_text()
    )
    assert local_manifest["diagnostics"]["operation_count"] == 6


def test_publish_full_bundle_calls_expected_repos(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_operations(monkeypatch)
    artifact_dir = _write_bundle(tmp_path)
    api = FakeHfApi()
    config = HuggingFacePublishConfig(
        diagnostics_repo="policyengine/microplex-us-diagnostics",
        dataset_repo="policyengine/microplex-us-deployed-datasets",
        token="token",
    )

    result = publish_microplex_artifact_to_hf(
        artifact_dir,
        config,
        publish_dataset=True,
        promote_dataset=True,
        api=api,
        registry_loader=lambda _config: (
            '{"run_id":"old-run","manifest":"runs/old-run/manifest.json"}\n'
        ),
    )

    assert result["status"] == "published"
    assert [commit["repo_id"] for commit in api.commits] == [
        "policyengine/microplex-us-diagnostics",
        "policyengine/microplex-us-deployed-datasets",
    ]
    diagnostics_paths = {
        op["path_in_repo"] for op in api.commits[0]["operations"]
    }
    assert "latest.json" in diagnostics_paths
    assert "run_registry.jsonl" in diagnostics_paths
    dataset_paths = {op["path_in_repo"] for op in api.commits[1]["operations"]}
    assert "staging/run-a/policyengine_us.h5" in dataset_paths
    assert "staging/run-a/manifest.json" in dataset_paths
    assert "policyengine_us.h5" in dataset_paths
    assert "manifest.json" in dataset_paths


def test_smoke_published_hf_artifact_passes_when_expected_files_exist() -> None:
    api = FakeHfApi()
    api.files_by_repo = {
        "policyengine/microplex-us-diagnostics": [
            "latest.json",
            "run_registry.jsonl",
            "runs/run-a/manifest.json",
            "runs/run-a/policyengine_native_scores.json",
            "runs/run-a/pe_us_data_rebuild_native_audit.json",
            "runs/run-a/pe_native_target_diagnostics.json",
        ],
        "policyengine/microplex-us-deployed-datasets": [
            "staging/run-a/policyengine_us.h5",
            "staging/run-a/manifest.json",
            "policyengine_us.h5",
            "manifest.json",
        ],
    }
    config = HuggingFacePublishConfig(
        diagnostics_repo="policyengine/microplex-us-diagnostics",
        dataset_repo="policyengine/microplex-us-deployed-datasets",
    )

    result = smoke_published_hf_artifact(
        config,
        api=api,
        latest_loader=lambda _config: {"run_id": "run-a"},
    )

    assert result["status"] == "passed"
    assert result["missing_count"] == 0
    assert result["run_id"] == "run-a"


def test_smoke_published_hf_artifact_reports_missing_files() -> None:
    api = FakeHfApi()
    api.files_by_repo = {
        "policyengine/microplex-us-diagnostics": ["latest.json"],
        "policyengine/microplex-us-deployed-datasets": [
            "staging/run-a/policyengine_us.h5"
        ],
    }
    config = HuggingFacePublishConfig(
        diagnostics_repo="policyengine/microplex-us-diagnostics",
        dataset_repo="policyengine/microplex-us-deployed-datasets",
    )

    result = smoke_published_hf_artifact(
        config,
        run_id="run-a",
        api=api,
        latest_loader=lambda _config: {"run_id": "ignored"},
    )

    assert result["status"] == "failed"
    assert "runs/run-a/manifest.json" in result["diagnostics"]["missing"]
    assert "manifest.json" in result["dataset"]["missing"]
