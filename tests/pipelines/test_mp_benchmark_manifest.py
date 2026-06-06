"""Tests for pinned MP replacement benchmark manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from microplex_us.pipelines.mp_benchmark_manifest import (
    FROZEN_PRODUCTION_ECPS_BASELINE_ENHANCED_CPS_NATIVE_LOSS,
    FROZEN_PRODUCTION_ECPS_BASELINE_HOLDOUT_LOSS,
    FROZEN_PRODUCTION_ECPS_BASELINE_SHA256,
    FROZEN_PRODUCTION_ECPS_BASELINE_UNWEIGHTED_MSRE,
    FROZEN_PRODUCTION_ECPS_SCORING_CONFIG_SHA256,
    FROZEN_PRODUCTION_ECPS_TARGET_COUNT,
    FROZEN_PRODUCTION_ECPS_TARGET_DB_SHA256,
    FROZEN_PRODUCTION_ECPS_TARGET_NAMES_SHA256,
    FROZEN_PRODUCTION_ECPS_TARGET_PROFILE,
    FROZEN_PRODUCTION_ECPS_TARGET_SCOPE,
    build_mp_benchmark_manifest,
    load_frozen_production_ecps_benchmark_manifest,
    main,
)


def _write_file(path: Path, contents: bytes) -> Path:
    path.write_bytes(contents)
    return path


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def test_build_mp_benchmark_manifest_pins_release_inputs(tmp_path):
    baseline_contents = b"baseline h5"
    target_contents = b"target db"
    baseline = _write_file(tmp_path / "enhanced_cps_2024.h5", baseline_contents)
    target_db = _write_file(tmp_path / "policyengine_targets.db", target_contents)

    manifest = build_mp_benchmark_manifest(
        baseline_dataset_path=baseline,
        target_db_path=target_db,
        period=2024,
        target_profile="pe_native_broad",
        target_scope="national",
        target_count=150,
        target_names_sha256="d" * 64,
        scoring_config_sha256="e" * 64,
        baseline_enhanced_cps_native_loss=0.2,
        baseline_holdout_loss=0.04,
        baseline_unweighted_msre=0.17,
        policyengine_us_data_commit="b" * 40,
        policyengine_us_version="1.587.0",
        enforce_production_pins=False,
    )

    assert manifest["schema_version"] == 1
    assert manifest["certificate_type"] == "frozen_production_ecps_baseline"
    assert manifest["period"] == 2024
    assert manifest["target_profile"] == "pe_native_broad"
    assert manifest["target_scope"] == "national"
    assert manifest["target_surface"] == {
        "target_profile": "pe_native_broad",
        "target_scope": "national",
        "target_count": 150,
        "target_names_sha256": "d" * 64,
    }
    assert manifest["scoring_config"] == {"sha256": "e" * 64}
    assert manifest["baseline_metrics"] == {
        "baseline_enhanced_cps_native_loss": 0.2,
        "baseline_holdout_loss": 0.04,
        "baseline_unweighted_msre": 0.17,
    }
    assert manifest["baseline_dataset"]["path"] == str(baseline.resolve())
    assert manifest["baseline_dataset"]["sha256"] == _sha256(baseline_contents)
    assert manifest["target_db"]["path"] == str(target_db.resolve())
    assert manifest["target_db"]["sha256"] == _sha256(target_contents)
    assert manifest["policyengine_us_data"]["commit"] == "b" * 40
    assert manifest["policyengine_us"]["version"] == "1.587.0"


def test_packaged_frozen_production_manifest_matches_canonical_surface():
    manifest = load_frozen_production_ecps_benchmark_manifest()

    assert manifest["certificate_type"] == "frozen_production_ecps_baseline"
    assert manifest["period"] == 2024
    assert manifest["baseline_dataset"]["sha256"] == (
        FROZEN_PRODUCTION_ECPS_BASELINE_SHA256
    )
    assert manifest["target_db"]["sha256"] == FROZEN_PRODUCTION_ECPS_TARGET_DB_SHA256
    assert manifest["target_surface"] == {
        "target_profile": FROZEN_PRODUCTION_ECPS_TARGET_PROFILE,
        "target_scope": FROZEN_PRODUCTION_ECPS_TARGET_SCOPE,
        "target_count": FROZEN_PRODUCTION_ECPS_TARGET_COUNT,
        "target_names_sha256": FROZEN_PRODUCTION_ECPS_TARGET_NAMES_SHA256,
    }
    assert manifest["scoring_config"]["sha256"] == (
        FROZEN_PRODUCTION_ECPS_SCORING_CONFIG_SHA256
    )
    assert manifest["baseline_metrics"]["baseline_enhanced_cps_native_loss"] == (
        FROZEN_PRODUCTION_ECPS_BASELINE_ENHANCED_CPS_NATIVE_LOSS
    )
    assert manifest["baseline_metrics"]["baseline_holdout_loss"] == (
        FROZEN_PRODUCTION_ECPS_BASELINE_HOLDOUT_LOSS
    )
    assert manifest["baseline_metrics"]["baseline_unweighted_msre"] == (
        FROZEN_PRODUCTION_ECPS_BASELINE_UNWEIGHTED_MSRE
    )


def test_main_writes_mp_benchmark_manifest(tmp_path, capsys):
    baseline = _write_file(tmp_path / "enhanced_cps_2024.h5", b"baseline")
    target_db = _write_file(tmp_path / "policyengine_targets.db", b"targets")
    output = tmp_path / "benchmark_manifest.json"

    exit_code = main(
        [
            "--baseline-dataset",
            str(baseline),
            "--target-db",
            str(target_db),
            "--output-json",
            str(output),
            "--policyengine-us-data-commit",
            "c" * 40,
            "--policyengine-us-version",
            "1.587.0",
            "--target-scope",
            "national",
            "--target-count",
            "150",
            "--target-names-sha256",
            "d" * 64,
            "--scoring-config-sha256",
            "e" * 64,
            "--baseline-enhanced-cps-native-loss",
            "0.2",
            "--baseline-holdout-loss",
            "0.04",
            "--baseline-unweighted-msre",
            "0.17",
            "--allow-noncanonical-production-pins",
        ]
    )

    printed_path = Path(capsys.readouterr().out.strip())
    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert printed_path == output
    assert payload["policyengine_us_data"]["commit"] == "c" * 40


def test_build_mp_benchmark_manifest_rejects_noncanonical_release_pins(tmp_path):
    baseline = _write_file(tmp_path / "enhanced_cps_2024.h5", b"baseline")
    target_db = _write_file(tmp_path / "policyengine_targets.db", b"targets")

    with pytest.raises(ValueError, match="release-pinned baseline/target surface"):
        build_mp_benchmark_manifest(
            baseline_dataset_path=baseline,
            target_db_path=target_db,
            period=2024,
            target_profile="pe_native_broad",
            target_scope="all",
            target_count=150,
            target_names_sha256="d" * 64,
            scoring_config_sha256="e" * 64,
            baseline_enhanced_cps_native_loss=0.2,
            baseline_holdout_loss=0.04,
            baseline_unweighted_msre=0.17,
            policyengine_us_data_commit="b" * 40,
            policyengine_us_version="1.587.0",
        )


def test_build_mp_benchmark_manifest_requires_baseline_metrics(tmp_path):
    baseline = _write_file(tmp_path / "enhanced_cps_2024.h5", b"baseline")
    target_db = _write_file(tmp_path / "policyengine_targets.db", b"targets")

    with pytest.raises(ValueError, match="pin baseline metrics"):
        build_mp_benchmark_manifest(
            baseline_dataset_path=baseline,
            target_db_path=target_db,
            target_count=150,
            target_names_sha256="d" * 64,
            scoring_config_sha256="e" * 64,
            policyengine_us_data_commit="b" * 40,
            policyengine_us_version="1.587.0",
            enforce_production_pins=False,
        )


def test_dirty_policyengine_us_data_repo_is_rejected_unless_explicit(tmp_path):
    baseline = _write_file(tmp_path / "enhanced_cps_2024.h5", b"baseline")
    target_db = _write_file(tmp_path / "policyengine_targets.db", b"targets")
    repo = tmp_path / "policyengine-us-data"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    _write_file(repo / "tracked.txt", b"clean")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Codex",
            "-c",
            "user.email=codex@example.com",
            "commit",
            "-m",
            "init",
        ],
        check=True,
        capture_output=True,
    )
    (repo / "tracked.txt").write_text("dirty")

    with pytest.raises(ValueError, match="uncommitted changes"):
        build_mp_benchmark_manifest(
            baseline_dataset_path=baseline,
            target_db_path=target_db,
            policyengine_us_data_repo=repo,
            policyengine_us_version="1.587.0",
            target_count=150,
            target_names_sha256="d" * 64,
            scoring_config_sha256="e" * 64,
            baseline_enhanced_cps_native_loss=0.2,
            baseline_holdout_loss=0.04,
            baseline_unweighted_msre=0.17,
            enforce_production_pins=False,
        )

    manifest = build_mp_benchmark_manifest(
        baseline_dataset_path=baseline,
        target_db_path=target_db,
        policyengine_us_data_repo=repo,
        policyengine_us_version="1.587.0",
        target_count=150,
        target_names_sha256="d" * 64,
        scoring_config_sha256="e" * 64,
        baseline_enhanced_cps_native_loss=0.2,
        baseline_holdout_loss=0.04,
        baseline_unweighted_msre=0.17,
        allow_dirty_policyengine_us_data=True,
        enforce_production_pins=False,
    )

    assert manifest["policyengine_us_data"]["dirty"] is True
    assert len(manifest["policyengine_us_data"]["commit"]) == 40
