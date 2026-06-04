"""Publish Microplex artifact bundles to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HF_PUBLISH_MANIFEST_FILENAME = "hf_publish_manifest.json"
DEFAULT_HF_REPO_TYPE = "dataset"
DEFAULT_DIAGNOSTICS_REPO = "policyengine/microplex-us-diagnostics"
DEFAULT_DATASET_REPO = "policyengine/microplex-us-deployed-datasets"
DIAGNOSTICS_RUN_PREFIX = "runs"
DATASET_STAGING_PREFIX = "staging"
DIAGNOSTICS_ARTIFACT_KEYS = (
    "policyengine_native_scores",
    "policyengine_native_audit",
    "policyengine_native_target_diagnostics",
)
DATASET_ARTIFACT_KEYS = ("policyengine_dataset",)


@dataclass(frozen=True)
class HuggingFacePublishConfig:
    """Destination repos and auth settings for Hugging Face artifact publishing."""

    diagnostics_repo: str | None = DEFAULT_DIAGNOSTICS_REPO
    dataset_repo: str | None = DEFAULT_DATASET_REPO
    repo_type: str = DEFAULT_HF_REPO_TYPE
    token: str | None = None
    diagnostics_run_prefix: str = DIAGNOSTICS_RUN_PREFIX
    dataset_staging_prefix: str = DATASET_STAGING_PREFIX

    @classmethod
    def from_env(cls) -> HuggingFacePublishConfig:
        return cls(
            diagnostics_repo=os.environ.get(
                "MICROPLEX_HF_DIAGNOSTICS_REPO",
                DEFAULT_DIAGNOSTICS_REPO,
            ),
            dataset_repo=os.environ.get(
                "MICROPLEX_HF_DATASET_REPO",
                DEFAULT_DATASET_REPO,
            ),
            repo_type=os.environ.get("MICROPLEX_HF_REPO_TYPE", DEFAULT_HF_REPO_TYPE),
            token=_first_env("MICROPLEX_HF_TOKEN", "HUGGING_FACE_TOKEN", "HF_TOKEN"),
            diagnostics_run_prefix=os.environ.get(
                "MICROPLEX_HF_DIAGNOSTICS_RUN_PREFIX",
                DIAGNOSTICS_RUN_PREFIX,
            ),
            dataset_staging_prefix=os.environ.get(
                "MICROPLEX_HF_DATASET_STAGING_PREFIX",
                DATASET_STAGING_PREFIX,
            ),
        )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_hf_prefix(prefix: str) -> str:
    """Normalize a Hugging Face repo path prefix."""
    return prefix.strip("/")


def build_hf_repo_path(*parts: str | None) -> str:
    """Join repo path parts using POSIX separators."""
    return "/".join(str(part).strip("/") for part in parts if part)


def resolve_bundle_run_id(artifact_dir: str | Path, run_id: str | None = None) -> str:
    """Return the explicit run ID or the bundle directory name."""
    return run_id or Path(artifact_dir).resolve().name


def resolve_manifest_artifact_path(
    artifact_dir: str | Path,
    manifest: dict[str, Any],
    artifact_key: str,
) -> Path:
    """Resolve one artifact path from a bundle manifest."""
    artifact_name = dict(manifest.get("artifacts", {})).get(artifact_key)
    if not isinstance(artifact_name, str) or not artifact_name:
        raise FileNotFoundError(
            f"Manifest is missing artifacts.{artifact_key} for {artifact_dir}"
        )
    path = Path(artifact_name)
    if not path.is_absolute():
        path = Path(artifact_dir) / path
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest artifact {artifact_key!r} does not exist: {path}"
        )
    return path


def diagnostics_repo_paths(
    artifact_dir: str | Path,
    *,
    run_id: str | None = None,
    run_prefix: str = DIAGNOSTICS_RUN_PREFIX,
) -> dict[str, str]:
    """Return Hugging Face repo paths for diagnostics files."""
    resolved_run_id = resolve_bundle_run_id(artifact_dir, run_id)
    prefix = build_hf_repo_path(normalize_hf_prefix(run_prefix), resolved_run_id)
    return {
        "manifest": build_hf_repo_path(prefix, "manifest.json"),
        "policyengine_native_scores": build_hf_repo_path(
            prefix,
            "policyengine_native_scores.json",
        ),
        "policyengine_native_audit": build_hf_repo_path(
            prefix,
            "pe_us_data_rebuild_native_audit.json",
        ),
        "policyengine_native_target_diagnostics": build_hf_repo_path(
            prefix,
            "pe_native_target_diagnostics.json",
        ),
        "latest": "latest.json",
        "run_registry": "run_registry.jsonl",
    }


def dataset_repo_paths(
    artifact_dir: str | Path,
    *,
    run_id: str | None = None,
    staging_prefix: str = DATASET_STAGING_PREFIX,
    promote: bool = False,
) -> dict[str, str]:
    """Return Hugging Face repo paths for deployed dataset files."""
    resolved_run_id = resolve_bundle_run_id(artifact_dir, run_id)
    prefix = build_hf_repo_path(normalize_hf_prefix(staging_prefix), resolved_run_id)
    paths = {
        "policyengine_dataset": build_hf_repo_path(prefix, "policyengine_us.h5"),
        "manifest": build_hf_repo_path(prefix, "manifest.json"),
    }
    if promote:
        paths.update(
            {
                "promoted_policyengine_dataset": "policyengine_us.h5",
                "promoted_manifest": "manifest.json",
            }
        )
    return paths


def build_latest_payload(
    *,
    run_id: str,
    diagnostics_repo: str,
    repo_type: str,
    paths: dict[str, str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build the diagnostics repo's latest-run pointer."""
    return {
        "schema_version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "artifact_id": run_id,
        "repo_id": diagnostics_repo,
        "repo_type": repo_type,
        "paths": {
            "manifest": paths["manifest"],
            "policyengine_native_scores": paths["policyengine_native_scores"],
            "policyengine_native_audit": paths["policyengine_native_audit"],
            "policyengine_native_target_diagnostics": (
                paths["policyengine_native_target_diagnostics"]
            ),
        },
        "summary": {
            "created_at": manifest.get("created_at"),
            "policyengine_native_scores": manifest.get("policyengine_native_scores"),
            "policyengine_native_audit": manifest.get("policyengine_native_audit"),
        },
    }


def build_run_registry_entry(
    *,
    run_id: str,
    diagnostics_repo: str,
    repo_type: str,
    paths: dict[str, str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build one compact diagnostics registry row."""
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "artifact_id": run_id,
        "repo_id": diagnostics_repo,
        "repo_type": repo_type,
        "manifest": paths["manifest"],
        "policyengine_native_scores": paths["policyengine_native_scores"],
        "policyengine_native_audit": paths["policyengine_native_audit"],
        "policyengine_native_target_diagnostics": (
            paths["policyengine_native_target_diagnostics"]
        ),
        "candidate_enhanced_cps_native_loss": _nested_get(
            manifest,
            "policyengine_native_scores",
            "candidate_enhanced_cps_native_loss",
        ),
        "enhanced_cps_native_loss_delta": _nested_get(
            manifest,
            "policyengine_native_scores",
            "enhanced_cps_native_loss_delta",
        ),
    }


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def build_diagnostics_operations(
    artifact_dir: str | Path,
    config: HuggingFacePublishConfig,
    *,
    run_id: str | None = None,
    registry_text: str = "",
) -> tuple[list[Any], dict[str, Any]]:
    """Build Hugging Face commit operations for diagnostics JSON files."""
    if config.diagnostics_repo is None:
        raise ValueError("diagnostics_repo is required to publish diagnostics")
    root = Path(artifact_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    resolved_run_id = resolve_bundle_run_id(root, run_id)
    paths = diagnostics_repo_paths(
        root,
        run_id=resolved_run_id,
        run_prefix=config.diagnostics_run_prefix,
    )
    files = {
        "manifest": manifest_path,
        "policyengine_native_scores": resolve_manifest_artifact_path(
            root,
            manifest,
            "policyengine_native_scores",
        ),
        "policyengine_native_audit": resolve_manifest_artifact_path(
            root,
            manifest,
            "policyengine_native_audit",
        ),
        "policyengine_native_target_diagnostics": resolve_manifest_artifact_path(
            root,
            manifest,
            "policyengine_native_target_diagnostics",
        ),
    }
    latest = build_latest_payload(
        run_id=resolved_run_id,
        diagnostics_repo=config.diagnostics_repo,
        repo_type=config.repo_type,
        paths=paths,
        manifest=manifest,
    )
    registry_entry = build_run_registry_entry(
        run_id=resolved_run_id,
        diagnostics_repo=config.diagnostics_repo,
        repo_type=config.repo_type,
        paths=paths,
        manifest=manifest,
    )
    registry_text = _append_registry_jsonl(registry_text, registry_entry)

    operations = [
        _commit_add(paths[key], path)
        for key, path in files.items()
    ]
    operations.extend(
        [
            _commit_add_bytes(paths["latest"], latest),
            _commit_add_text(paths["run_registry"], registry_text),
        ]
    )
    payload = {
        "repo_id": config.diagnostics_repo,
        "repo_type": config.repo_type,
        "run_id": resolved_run_id,
        "paths": paths,
        "files": {key: str(path) for key, path in files.items()},
        "latest": latest,
        "run_registry_entry": registry_entry,
        "operation_count": len(operations),
    }
    return operations, payload


def build_dataset_operations(
    artifact_dir: str | Path,
    config: HuggingFacePublishConfig,
    *,
    run_id: str | None = None,
    promote: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """Build Hugging Face commit operations for the deployed dataset repo."""
    if config.dataset_repo is None:
        raise ValueError("dataset_repo is required to publish datasets")
    root = Path(artifact_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    resolved_run_id = resolve_bundle_run_id(root, run_id)
    paths = dataset_repo_paths(
        root,
        run_id=resolved_run_id,
        staging_prefix=config.dataset_staging_prefix,
        promote=promote,
    )
    dataset_path = resolve_manifest_artifact_path(root, manifest, "policyengine_dataset")
    operations = [
        _commit_add(paths["policyengine_dataset"], dataset_path),
        _commit_add(paths["manifest"], manifest_path),
    ]
    if promote:
        operations.extend(
            [
                _commit_add(paths["promoted_policyengine_dataset"], dataset_path),
                _commit_add(paths["promoted_manifest"], manifest_path),
            ]
        )
    payload = {
        "repo_id": config.dataset_repo,
        "repo_type": config.repo_type,
        "run_id": resolved_run_id,
        "paths": paths,
        "files": {
            "policyengine_dataset": str(dataset_path),
            "manifest": str(manifest_path),
        },
        "promoted": bool(promote),
        "operation_count": len(operations),
    }
    return operations, payload


def publish_microplex_artifact_to_hf(
    artifact_dir: str | Path,
    config: HuggingFacePublishConfig,
    *,
    run_id: str | None = None,
    publish_diagnostics: bool = True,
    publish_dataset: bool = False,
    promote_dataset: bool = False,
    dry_run: bool = False,
    api: Any | None = None,
    registry_loader: Callable[[HuggingFacePublishConfig], str] | None = None,
) -> dict[str, Any]:
    """Publish a completed Microplex bundle to configured Hugging Face repos."""
    if not publish_diagnostics and not publish_dataset:
        raise ValueError("At least one of publish_diagnostics or publish_dataset is required")
    root = Path(artifact_dir).resolve()
    resolved_run_id = resolve_bundle_run_id(root, run_id)
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_dir": str(root),
        "run_id": resolved_run_id,
        "dry_run": dry_run,
        "diagnostics": None,
        "dataset": None,
    }
    hf_api = None if dry_run else api or create_hf_api()

    if publish_diagnostics:
        registry_text = (
            registry_loader(config)
            if registry_loader is not None
            else load_existing_registry_text(config)
        )
        operations, diagnostics_payload = build_diagnostics_operations(
            root,
            config,
            run_id=resolved_run_id,
            registry_text=registry_text,
        )
        result["diagnostics"] = diagnostics_payload
        if not dry_run:
            hf_api.create_commit(
                repo_id=config.diagnostics_repo,
                repo_type=config.repo_type,
                operations=operations,
                commit_message=f"Publish Microplex diagnostics {resolved_run_id}",
                token=config.token,
            )

    if publish_dataset:
        operations, dataset_payload = build_dataset_operations(
            root,
            config,
            run_id=resolved_run_id,
            promote=promote_dataset,
        )
        result["dataset"] = dataset_payload
        if not dry_run:
            hf_api.create_commit(
                repo_id=config.dataset_repo,
                repo_type=config.repo_type,
                operations=operations,
                commit_message=f"Publish Microplex dataset {resolved_run_id}",
                token=config.token,
            )

    result["status"] = "dry_run" if dry_run else "published"
    _write_json(root / HF_PUBLISH_MANIFEST_FILENAME, result)
    return result


def smoke_published_hf_artifact(
    config: HuggingFacePublishConfig,
    *,
    run_id: str | None = None,
    check_dataset: bool = True,
    check_promoted_dataset: bool = True,
    api: Any | None = None,
    latest_loader: Callable[[HuggingFacePublishConfig], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify that a published Hugging Face artifact exposes expected files."""
    if config.diagnostics_repo is None:
        raise ValueError("diagnostics_repo is required for smoke checks")
    hf_api = api or create_hf_api()
    latest = (
        latest_loader(config)
        if latest_loader is not None
        else load_hf_json(config, config.diagnostics_repo, "latest.json")
    )
    resolved_run_id = run_id or latest.get("run_id")
    if not isinstance(resolved_run_id, str) or not resolved_run_id:
        raise ValueError("run_id is required when latest.json does not define run_id")

    diagnostics_paths = diagnostics_repo_paths(
        ".",
        run_id=resolved_run_id,
        run_prefix=config.diagnostics_run_prefix,
    )
    expected_diagnostics = set(diagnostics_paths.values())
    diagnostics_files = set(
        hf_api.list_repo_files(
            repo_id=config.diagnostics_repo,
            repo_type=config.repo_type,
            token=config.token,
        )
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "run_id": resolved_run_id,
        "diagnostics": {
            "repo_id": config.diagnostics_repo,
            "expected": sorted(expected_diagnostics),
            "missing": sorted(expected_diagnostics - diagnostics_files),
        },
        "dataset": None,
    }

    if check_dataset:
        if config.dataset_repo is None:
            raise ValueError("dataset_repo is required when check_dataset is true")
        dataset_paths = dataset_repo_paths(
            ".",
            run_id=resolved_run_id,
            staging_prefix=config.dataset_staging_prefix,
            promote=check_promoted_dataset,
        )
        expected_dataset = set(dataset_paths.values())
        dataset_files = set(
            hf_api.list_repo_files(
                repo_id=config.dataset_repo,
                repo_type=config.repo_type,
                token=config.token,
            )
        )
        result["dataset"] = {
            "repo_id": config.dataset_repo,
            "expected": sorted(expected_dataset),
            "missing": sorted(expected_dataset - dataset_files),
        }

    missing = list(result["diagnostics"]["missing"])
    if result["dataset"] is not None:
        missing.extend(result["dataset"]["missing"])
    result["status"] = "passed" if not missing else "failed"
    result["missing_count"] = len(missing)
    return result


def create_hf_api() -> Any:
    """Create a Hugging Face API client lazily."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - exercised by CLI environment.
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face uploads. Install the "
            "optional extra with `uv sync --extra hf` or run through "
            "`uv run --extra hf ...`."
        ) from error
    return HfApi()


def load_hf_json(
    config: HuggingFacePublishConfig,
    repo_id: str,
    filename: str,
) -> dict[str, Any]:
    """Download one JSON file from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face smoke checks."
        ) from error
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=config.repo_type,
        token=config.token,
    )
    return json.loads(Path(path).read_text())


def _commit_add(path_in_repo: str, local_path: Path) -> Any:
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face commit operations."
        ) from error
    return CommitOperationAdd(
        path_in_repo=path_in_repo,
        path_or_fileobj=str(local_path),
    )


def _commit_add_bytes(path_in_repo: str, payload: dict[str, Any]) -> Any:
    return _commit_add_text(
        path_in_repo,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _commit_add_text(path_in_repo: str, text: str) -> Any:
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face commit operations."
        ) from error
    return CommitOperationAdd(
        path_in_repo=path_in_repo,
        path_or_fileobj=text.encode("utf-8"),
    )


def load_existing_registry_text(config: HuggingFacePublishConfig) -> str:
    """Download the existing diagnostics registry JSONL, or return empty text."""
    if config.diagnostics_repo is None:
        return ""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return ""
    try:
        path = hf_hub_download(
            repo_id=config.diagnostics_repo,
            filename="run_registry.jsonl",
            repo_type=config.repo_type,
            token=config.token,
        )
    except Exception:
        return ""
    return Path(path).read_text()


def _append_registry_jsonl(existing_text: str, entry: dict[str, Any]) -> str:
    lines = [line for line in existing_text.splitlines() if line.strip()]
    lines = [
        line
        for line in lines
        if json.loads(line).get("run_id") != entry["run_id"]
    ]
    lines.append(json.dumps(entry, sort_keys=True))
    return "\n".join(lines) + "\n"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path)
    temp_path = resolved.with_suffix(resolved.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp_path.replace(resolved)


def _build_config_from_args(args: argparse.Namespace) -> HuggingFacePublishConfig:
    env_config = HuggingFacePublishConfig.from_env()
    return HuggingFacePublishConfig(
        diagnostics_repo=args.diagnostics_repo or env_config.diagnostics_repo,
        dataset_repo=args.dataset_repo or env_config.dataset_repo,
        repo_type=args.repo_type or env_config.repo_type,
        token=args.token or env_config.token,
        diagnostics_run_prefix=(
            args.diagnostics_run_prefix or env_config.diagnostics_run_prefix
        ),
        dataset_staging_prefix=(
            args.dataset_staging_prefix or env_config.dataset_staging_prefix
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish a Microplex artifact bundle to Hugging Face."
    )
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--diagnostics-repo", default=None)
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--repo-type", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--diagnostics-run-prefix", default=None)
    parser.add_argument("--dataset-staging-prefix", default=None)
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Do not publish diagnostics JSON files.",
    )
    parser.add_argument(
        "--publish-dataset",
        action="store_true",
        help="Also publish policyengine_us.h5 and manifest.json to the dataset repo.",
    )
    parser.add_argument(
        "--promote-dataset",
        action="store_true",
        help="Also write policyengine_us.h5 and manifest.json at the dataset repo root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write hf_publish_manifest.json without uploading.",
    )
    args = parser.parse_args(argv)
    try:
        config = _build_config_from_args(args)
        result = publish_microplex_artifact_to_hf(
            args.artifact_dir,
            config,
            run_id=args.run_id,
            publish_diagnostics=not args.no_diagnostics,
            publish_dataset=args.publish_dataset or args.promote_dataset,
            promote_dataset=args.promote_dataset,
            dry_run=args.dry_run,
        )
    except Exception as error:  # noqa: BLE001 - CLI should report concise failure.
        print(f"Hugging Face publish failed: {error}", file=sys.stderr)
        return 1
    mode = "planned" if args.dry_run else "published"
    print(f"Hugging Face artifact {mode}: {result['run_id']}")
    print(args.artifact_dir / HF_PUBLISH_MANIFEST_FILENAME)
    return 0


def main_smoke(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-check a published Microplex Hugging Face artifact."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--diagnostics-repo", default=None)
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--repo-type", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--diagnostics-run-prefix", default=None)
    parser.add_argument("--dataset-staging-prefix", default=None)
    parser.add_argument(
        "--no-dataset",
        action="store_true",
        help="Only check diagnostics files.",
    )
    parser.add_argument(
        "--no-promoted-dataset",
        action="store_true",
        help="Do not require root policyengine_us.h5 and manifest.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full smoke-check payload as JSON.",
    )
    args = parser.parse_args(argv)
    try:
        config = _build_config_from_args(args)
        result = smoke_published_hf_artifact(
            config,
            run_id=args.run_id,
            check_dataset=not args.no_dataset,
            check_promoted_dataset=not args.no_promoted_dataset,
        )
    except Exception as error:  # noqa: BLE001 - CLI should report concise failure.
        print(f"Hugging Face smoke check failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["status"] == "passed":
        print(f"Hugging Face artifact smoke check passed: {result['run_id']}")
    else:
        print(
            f"Hugging Face artifact smoke check failed: {result['run_id']} "
            f"({result['missing_count']} missing files)",
            file=sys.stderr,
        )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
