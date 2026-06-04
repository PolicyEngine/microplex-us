"""Tests for generated US pipeline diagram and overlay data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from microplex_us.pipelines.pipeline_docs import (
    US_PIPELINE_GRAPH_SCHEMA_VERSION,
    US_PIPELINE_INTERNALS_SCHEMA_VERSION,
    US_PIPELINE_OVERLAY_SCHEMA_VERSION,
    build_us_pipeline_docs_payloads,
    build_us_pipeline_graph,
    build_us_pipeline_internals,
    build_us_pipeline_overlay,
    write_us_pipeline_docs,
)
from microplex_us.pipelines.stage_contracts import US_CANONICAL_STAGE_IDS


def test_pipeline_graph_is_generated_from_canonical_stage_registry() -> None:
    graph = build_us_pipeline_graph()

    assert graph["schemaVersion"] == US_PIPELINE_GRAPH_SCHEMA_VERSION
    assert [node["id"] for node in graph["nodes"]] == list(US_CANONICAL_STAGE_IDS)
    assert len(graph["edges"]) == len(US_CANONICAL_STAGE_IDS) - 1

    stage_ids = set(US_CANONICAL_STAGE_IDS)
    for edge in graph["edges"]:
        assert edge["source"] in stage_ids
        assert edge["target"] in stage_ids
        assert edge["viewer"]["routing"] == "elk_orthogonal"


def test_pipeline_overlay_uses_saved_stage_manifest_without_absolute_paths() -> None:
    fixture = _fixture_path("complete_run")

    overlay = build_us_pipeline_overlay(fixture)

    assert overlay["schemaVersion"] == US_PIPELINE_OVERLAY_SCHEMA_VERSION
    assert overlay["artifactRoot"] == "complete_run"
    assert [stage["id"] for stage in overlay["stages"]] == list(US_CANONICAL_STAGE_IDS)
    stage8 = _stage(overlay, "08_dataset_assembly")
    assert stage8["status"] == "ready"
    assert all(
        not str(artifact.get("path", "")).startswith("/")
        for stage in overlay["stages"]
        for artifact in stage["artifacts"]
    )


def test_pipeline_overlay_exposes_partial_and_failed_lifecycle_state() -> None:
    fixture = _fixture_path("failed_run")

    overlay = build_us_pipeline_overlay(fixture)

    stage5 = _stage(overlay, "05_donor_integration_synthesis")
    stage6 = _stage(overlay, "06_policyengine_entities")
    assert stage5["lifecycleStatus"] == "failed"
    assert stage5["failure"]["errorType"] == "RuntimeError"
    assert stage6["lifecycleStatus"] == "pending"


def test_pipeline_internals_include_substages_and_decorated_objects() -> None:
    internals = build_us_pipeline_internals()

    assert internals["schemaVersion"] == US_PIPELINE_INTERNALS_SCHEMA_VERSION
    assert [stage["id"] for stage in internals["stages"]] == list(
        US_CANONICAL_STAGE_IDS
    )
    assert all(stage["substages"] for stage in internals["stages"])

    stage5 = _internals_stage(internals, "05_donor_integration_synthesis")
    donor_node = _internals_node(stage5, "us.pipeline.integrate_donor_sources")
    assert donor_node["sourceFile"] == "src/microplex_us/pipelines/us.py"
    assert isinstance(donor_node["line"], int)
    assert donor_node["signature"].startswith("_integrate_donor_sources(")
    assert donor_node["pydoc"].endswith("USMicroplexPipeline._integrate_donor_sources")


def test_pipeline_internals_edges_reference_materialized_nodes() -> None:
    internals = build_us_pipeline_internals()

    for stage in internals["stages"]:
        for substage in stage["substages"]:
            node_ids = {node["id"] for node in substage["nodes"]}
            for edge in substage["edges"]:
                assert edge["source"] in node_ids, (substage["id"], edge)
                assert edge["target"] in node_ids, (substage["id"], edge)


def test_pipeline_markdown_docs_include_specific_code_references() -> None:
    payloads = build_us_pipeline_docs_payloads()
    markdown = payloads["us_pipeline_map.md"]

    assert isinstance(markdown, str)
    assert "| Node | Type | Status | API refs |" in markdown
    assert "`microplex.core.SourceProvider.load_frame`" in markdown
    assert "#### Edges" in markdown
    assert (
        "`microplex_us.pipelines.us.USMicroplexPipeline._integrate_donor_sources`"
        in markdown
    )


def test_pipeline_viewer_internals_fixture_is_current() -> None:
    repo_root = Path(__file__).parents[2]
    generated = json.loads(
        (repo_root / "docs/generated/us_pipeline_internals.json").read_text()
    )
    fixture = json.loads(
        (
            repo_root / "pipeline_viewer/src/fixtures/us_pipeline_internals.json"
        ).read_text()
    )

    assert fixture == generated


def test_write_us_pipeline_docs_check_detects_stale_generated_files(tmp_path) -> None:
    output_dir = tmp_path / "generated"
    fixture = _fixture_path("complete_run")
    write_us_pipeline_docs(output_dir, artifact_root=fixture)

    graph_path = output_dir / "us_pipeline_graph.json"
    graph = json.loads(graph_path.read_text())
    graph["nodes"][0]["title"] = "stale"
    graph_path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")

    with pytest.raises(SystemExit, match="stale or missing"):
        write_us_pipeline_docs(output_dir, artifact_root=fixture, check=True)


def test_committed_pipeline_docs_fixtures_are_current() -> None:
    generated = _fixture_path("generated")
    complete = _fixture_path("complete_run")
    failed = _fixture_path("failed_run")

    write_us_pipeline_docs(generated, check=True)
    write_us_pipeline_docs(
        generated / "complete_run",
        artifact_root=complete,
        check=True,
    )
    write_us_pipeline_docs(
        generated / "failed_run",
        artifact_root=failed,
        check=True,
    )


def test_pipeline_docs_import_does_not_eagerly_import_microplex() -> None:
    env = os.environ.copy()
    src_path = str(Path(__file__).parents[2] / "src")
    if env.get("PYTHONPATH"):
        src_path = os.pathsep.join((src_path, env["PYTHONPATH"]))
    env["PYTHONPATH"] = src_path
    code = "\n".join(
        (
            "import sys",
            "import microplex_us.pipelines.pipeline_docs",
            "assert 'microplex' not in sys.modules",
            "assert 'torch' not in sys.modules",
        )
    )

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _fixture_path(name: str) -> Path:
    return Path(__file__).parents[1] / "fixtures" / "pipeline_docs" / name


def _stage(overlay: dict, stage_id: str) -> dict:
    return next(stage for stage in overlay["stages"] if stage["id"] == stage_id)


def _internals_stage(internals: dict, stage_id: str) -> dict:
    return next(stage for stage in internals["stages"] if stage["id"] == stage_id)


def _internals_node(stage: dict, node_id: str) -> dict:
    for substage in stage["substages"]:
        for node in substage["nodes"]:
            if node["id"] == node_id:
                return node
    raise AssertionError(f"node not found: {node_id}")
