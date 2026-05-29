"""Emit a policyengine-bundles ``DataReleaseManifest`` for a Microplex-US dataset.

This is the plumbing that makes a Microplex H5 *certifiable into a bundle* exactly
like the Enhanced CPS. `policyengine-bundles` generation (`generate_bundle`) reads,
per country, a ``data_release_manifest_uri``, hashes it, validates it against the
``DataReleaseManifest`` model, and pins it. So the single artifact a data package
must publish to be bundle-eligible is this manifest. eCPS already emits one
(`policyengine_us_data/utils/release_manifest.py`); Microplex emits none today —
this closes that gap.

Conforms to `policyengine-bundles/schemas/data-release-manifest.schema.json`
(verified field names from `policyengine_bundles/models.py`). The `build`
(provenance) and compatibility fields are fed from the upstream build manifest
(see `rebuild/docs/build-manifest-spec.md`), so provenance *composes into*
certification instead of duplicating it.

Intended home when promoted: `src/microplex_us/release_manifest.py` (mirroring the
eCPS util). Kept here under `rebuild/` while the approach is proven.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
DEFAULT_BUNDLES_SCHEMA = (
    Path.home()
    / "PolicyEngine/policyengine-bundles/schemas/data-release-manifest.schema.json"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_and_size(path: str | Path, *, chunk: int = 1 << 20) -> tuple[str, int]:
    """Stream a file to a SHA256 hex digest + byte size (never loads it whole)."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _specifiers(pairs: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"name": name, "specifier": spec} for name, spec in pairs]


def build_data_artifact(
    *,
    key: str,
    path: str | Path | None = None,
    uri: str | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    kind: str = "dataset",
    certified: bool = False,
    status: str = "unverified",
    missing_reason: str | None = None,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> dict[str, Any]:
    """One ``DataArtifact`` entry. Computes sha256/size from ``path`` if given.

    The bundles ``DataArtifact`` model requires a ``missing_reason`` whenever the
    status is not ``certified``; we auto-fill one so non-certified manifests stay
    valid (a candidate dataset is legitimately ``unverified`` until a bundle run
    certifies it).
    """
    if path is not None and (sha256 is None or size_bytes is None):
        sha256, size_bytes = sha256_and_size(path)
    resolved_status = "certified" if certified else status
    if resolved_status != "certified" and not missing_reason:
        missing_reason = "not yet certified"
    return {
        "kind": kind,
        "uri": uri,
        "path": (Path(path).name if path is not None else None),
        "repo_id": repo_id,
        "revision": revision,
        "status": resolved_status,
        "sha256": sha256,
        "missing_reason": (None if resolved_status == "certified" else missing_reason),
        "size_bytes": size_bytes,
        "release_manifest_artifact_key": key,
        "preservation_mirrors": [],
        "metadata": {},
    }


def _ensure_missing_reason(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Backfill ``missing_reason`` for a caller-supplied artifact dict.

    The dict-passthrough form in :func:`build_release_manifest` would otherwise let
    a non-certified artifact reach the manifest without a reason, which the bundles
    model rejects. Mirrors :func:`build_data_artifact`'s rule; preserves an explicit
    reason and leaves certified artifacts untouched.
    """
    art = dict(artifact)
    if art.get("status", "certified") != "certified" and not art.get("missing_reason"):
        art["missing_reason"] = "not yet certified"
    return art


def _build_info(build_manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Map the upstream build manifest's provenance into ``DataBuildInfo``.

    Exact model/core versions go in ``metadata`` rather than the typed
    ``built_with_model_package`` field (a ``RuntimeComponentMetadata`` we don't
    populate here); the authoritative compatibility lives in
    ``compatible_model_packages`` at the top level.
    """
    if not build_manifest:
        return None
    build = build_manifest.get("build", {})
    env = build.get("environment", {}).get("key_packages", {})
    return {
        "build_id": build.get("id"),
        "built_at": build.get("created_at"),
        "built_with_model_package": None,
        "built_with_core_package": None,
        "metadata": {
            "engine": build.get("engine"),
            "code_ref": build.get("code_ref"),
            "model_package_version": env.get("policyengine-us"),
            "core_package_version": env.get("policyengine-core"),
            "reproduce": build.get("reproduce"),
            "build_manifest_schema_version": build_manifest.get("manifest_version"),
        },
    }


def build_release_manifest(
    *,
    data_package_name: str,
    data_package_version: str,
    artifacts: Mapping[str, str | Path | dict],
    repo_id: str,
    compatible_model_packages: Iterable[tuple[str, str]],
    compatible_core_packages: Iterable[tuple[str, str]],
    default_datasets: Mapping[str, str],
    revision: str | None = None,
    build_manifest: Mapping[str, Any] | None = None,
    certified: bool = False,
    preservation_dois: Iterable[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a bundles-compatible ``DataReleaseManifest`` dict.

    ``artifacts`` maps a key (e.g. ``"mp_300k_2024"``) to either a local H5 path
    (sha256/size computed + an ``hf://<repo_id>/<file>`` URI synthesised) or a
    fully-formed ``DataArtifact`` dict.
    """
    artifact_entries: dict[str, Any] = {}
    for key, spec in artifacts.items():
        if isinstance(spec, dict):
            artifact_entries[key] = _ensure_missing_reason(spec)
            continue
        artifact_entries[key] = build_data_artifact(
            key=key,
            path=spec,
            repo_id=repo_id,
            revision=revision,
            uri=f"hf://{repo_id}/{Path(spec).name}",
            certified=certified,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "data_package": {"name": data_package_name, "version": data_package_version},
        "compatible_model_packages": _specifiers(compatible_model_packages),
        "compatible_core_packages": _specifiers(compatible_core_packages),
        "default_datasets": dict(default_datasets),
        "build": _build_info(build_manifest),
        "artifacts": artifact_entries,
        "preservation_dois": list(preservation_dois or []),
        "created_at": _now_iso(),
        "metadata": dict(metadata or {}),
    }


def write_release_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    return out
