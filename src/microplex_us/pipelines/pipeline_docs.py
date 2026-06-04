"""Generate canonical US pipeline diagram and saved-run overlay data."""

from __future__ import annotations

import argparse
import difflib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from microplex_us.pipeline_extractor import collect_pipeline_nodes
from microplex_us.pipeline_schema import PipelineEdge, PipelineNode
from microplex_us.pipelines.stage_contracts import (
    US_STAGE_CONTRACT_VERSION,
    USPipelineStageContract,
    default_us_pipeline_stage_contracts,
)
from microplex_us.pipelines.stage_manifest_builder import build_us_stage_manifest
from microplex_us.pipelines.stage_manifest_io import load_us_stage_manifest
from microplex_us.pipelines.stage_manifest_types import (
    USStageLifecycleStatus,
    USStageStatus,
)

US_PIPELINE_GRAPH_SCHEMA_VERSION = 1
US_PIPELINE_OVERLAY_SCHEMA_VERSION = 1
US_PIPELINE_INTERNALS_SCHEMA_VERSION = 1
DEFAULT_US_PIPELINE_INTERNALS_MAP = "docs/us_pipeline_internals.map.json"


class USPipelineGraphNode(TypedDict):
    """One visual node in the canonical US pipeline graph."""

    id: str
    order: int
    step: str
    title: str
    purpose: str
    consumes: list[str]
    produces: list[str]
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    diagnostics: list[str]
    validations: list[dict[str, Any]]
    resume: dict[str, Any]
    viewer: dict[str, Any]


class USPipelineGraphEdge(TypedDict):
    """One directed visual edge in the canonical US pipeline graph."""

    id: str
    source: str
    target: str
    label: str
    resourceKeys: list[str]
    viewer: dict[str, Any]


class USPipelineGraph(TypedDict):
    """Canonical US pipeline graph generated from the stage registry."""

    schemaVersion: int
    contractVersion: str
    generatedFrom: str
    pipeline: str
    nodes: list[USPipelineGraphNode]
    edges: list[USPipelineGraphEdge]


class USPipelineOverlayStage(TypedDict):
    """One saved-run overlay record for a canonical stage."""

    id: str
    title: str
    status: USStageStatus
    lifecycleStatus: USStageLifecycleStatus
    outputManifest: str | None
    startedAt: str | None
    updatedAt: str | None
    completedAt: str | None
    failedAt: str | None
    deferredReason: str | None
    failure: dict[str, Any] | None
    artifacts: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    validations: list[dict[str, Any]]
    metrics: list[dict[str, Any]]
    resume: dict[str, Any]
    benchmarkSummary: dict[str, Any]


class USPipelineOverlay(TypedDict):
    """Saved-run overlay generated from a stage manifest."""

    schemaVersion: int
    graphSchemaVersion: int
    contractVersion: str
    generatedAt: str | None
    pipeline: str
    artifactRoot: str
    manifest: str
    stages: list[USPipelineOverlayStage]


class USPipelineInternals(TypedDict):
    """Static substage and machinery graph generated from docs metadata."""

    schemaVersion: int
    graphSchemaVersion: int
    contractVersion: str
    generatedFrom: list[str]
    pipeline: str
    stages: list[dict[str, Any]]
    apiNodes: list[dict[str, Any]]


def build_us_pipeline_graph() -> USPipelineGraph:
    """Build canonical US pipeline graph data from stage contracts."""

    contracts = default_us_pipeline_stage_contracts()
    nodes = [_graph_node(contract, order) for order, contract in enumerate(contracts)]
    edges = [
        _graph_edge(
            source=contracts[index],
            target=contracts[index + 1],
        )
        for index in range(len(contracts) - 1)
    ]
    return {
        "schemaVersion": US_PIPELINE_GRAPH_SCHEMA_VERSION,
        "contractVersion": US_STAGE_CONTRACT_VERSION,
        "generatedFrom": "microplex_us.pipelines.stage_contracts",
        "pipeline": "us_microplex",
        "nodes": nodes,
        "edges": edges,
    }


def build_us_pipeline_internals(
    *,
    repo_root: str | Path | None = None,
    map_path: str | Path | None = None,
) -> USPipelineInternals:
    """Build stage machinery graph data from the authored map and decorators."""

    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    source_map_path = (
        Path(map_path)
        if map_path is not None
        else root / DEFAULT_US_PIPELINE_INTERNALS_MAP
    )
    source_map = _load_pipeline_internals_map(source_map_path)
    contract_list = default_us_pipeline_stage_contracts()
    contracts = {contract.id: contract for contract in contract_list}
    expected_stage_ids = [contract.id for contract in contract_list]
    decorated_nodes = collect_pipeline_nodes(repo_root=root)
    authored_nodes = _pipeline_internals_authored_nodes(source_map)
    referenced_decorated_node_ids: set[str] = set()
    stages: list[dict[str, Any]] = []
    stage_payloads = _list_of_mappings(source_map.get("stages"))
    actual_stage_ids = [str(stage.get("id", "")) for stage in stage_payloads]
    if actual_stage_ids != expected_stage_ids:
        raise ValueError(
            "pipeline internals map stage ids must match canonical stage order: "
            f"{', '.join(expected_stage_ids)}"
        )
    for stage_payload in stage_payloads:
        stage_id = str(stage_payload.get("id", ""))
        if stage_id not in contracts:
            raise ValueError(f"unknown pipeline internals stage id: {stage_id}")
        contract = contracts[stage_id]
        substages: list[dict[str, Any]] = []
        for substage_payload in _list_of_mappings(stage_payload.get("substages")):
            substage, referenced_ids = _pipeline_internals_substage(
                stage_id=stage_id,
                payload=substage_payload,
                decorated_nodes=decorated_nodes,
                authored_nodes=authored_nodes,
            )
            substages.append(substage)
            referenced_decorated_node_ids.update(referenced_ids)
        stages.append(
            {
                "id": stage_id,
                "step": contract.step,
                "title": contract.title,
                "purpose": contract.purpose,
                "substages": substages,
            }
        )
    api_nodes = [
        decorated_nodes[node_id].to_node()
        for node_id in sorted(set(decorated_nodes) - referenced_decorated_node_ids)
    ]
    return {
        "schemaVersion": US_PIPELINE_INTERNALS_SCHEMA_VERSION,
        "graphSchemaVersion": US_PIPELINE_GRAPH_SCHEMA_VERSION,
        "contractVersion": US_STAGE_CONTRACT_VERSION,
        "generatedFrom": [
            "microplex_us.pipelines.stage_contracts",
            DEFAULT_US_PIPELINE_INTERNALS_MAP,
            "microplex_us.pipeline_metadata.pipeline_node",
        ],
        "pipeline": str(source_map.get("pipeline", "us_microplex")),
        "stages": stages,
        "apiNodes": api_nodes,
    }


def build_us_pipeline_overlay(
    artifact_dir: str | Path,
    *,
    manifest_payload: Mapping[str, Any] | None = None,
    prefer_saved_stage_manifest: bool = True,
) -> USPipelineOverlay:
    """Build a diagram overlay from one saved or live artifact directory."""

    artifact_root = Path(artifact_dir)
    manifest = (
        dict(manifest_payload)
        if manifest_payload is not None
        else json.loads((artifact_root / "manifest.json").read_text())
    )
    stage_manifest = _load_or_build_stage_manifest(
        artifact_root,
        manifest,
        prefer_saved=prefer_saved_stage_manifest,
    )
    return {
        "schemaVersion": US_PIPELINE_OVERLAY_SCHEMA_VERSION,
        "graphSchemaVersion": US_PIPELINE_GRAPH_SCHEMA_VERSION,
        "contractVersion": str(stage_manifest.get("contractVersion")),
        "generatedAt": _optional_str(stage_manifest.get("generatedAt")),
        "pipeline": str(stage_manifest.get("pipeline", "us_microplex")),
        "artifactRoot": artifact_root.name,
        "manifest": _relative_manifest_ref(stage_manifest),
        "stages": [
            _overlay_stage(stage)
            for stage in stage_manifest.get("stages", ())
            if isinstance(stage, dict)
        ],
    }


def pipeline_internals_json_schema() -> dict[str, Any]:
    """Return the JSON schema for substage machinery graph data."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://policyengine.org/schemas/microplex-us/pipeline-internals-v1.json",
        "title": "Microplex-US pipeline internals",
        "type": "object",
        "required": [
            "schemaVersion",
            "graphSchemaVersion",
            "contractVersion",
            "generatedFrom",
            "pipeline",
            "stages",
            "apiNodes",
        ],
        "properties": {
            "schemaVersion": {"const": US_PIPELINE_INTERNALS_SCHEMA_VERSION},
            "graphSchemaVersion": {"const": US_PIPELINE_GRAPH_SCHEMA_VERSION},
            "contractVersion": {"type": "string"},
            "generatedFrom": {"type": "array", "items": {"type": "string"}},
            "pipeline": {"type": "string"},
            "stages": {"type": "array", "items": {"$ref": "#/$defs/stage"}},
            "apiNodes": {"type": "array", "items": {"$ref": "#/$defs/node"}},
        },
        "$defs": {
            "stage": {
                "type": "object",
                "required": ["id", "step", "title", "purpose", "substages"],
                "properties": {
                    "id": {"type": "string"},
                    "step": {"type": "string"},
                    "title": {"type": "string"},
                    "purpose": {"type": "string"},
                    "substages": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/substage"},
                    },
                },
            },
            "substage": {
                "type": "object",
                "required": [
                    "id",
                    "stageId",
                    "title",
                    "description",
                    "nodes",
                    "edges",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "stageId": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "nodes": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                    "edges": {"type": "array", "items": {"$ref": "#/$defs/edge"}},
                },
            },
            "node": {
                "type": "object",
                "required": ["id", "label", "nodeType", "status"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "nodeType": {"type": "string"},
                    "description": {"type": "string"},
                    "sourceFile": {"type": ["string", "null"]},
                    "line": {"type": ["integer", "null"]},
                    "objectPath": {"type": ["string", "null"]},
                    "signature": {"type": ["string", "null"]},
                    "pydoc": {"type": ["string", "null"]},
                    "status": {"type": "string"},
                },
            },
            "edge": {
                "type": "object",
                "required": ["id", "source", "target", "edgeType", "label"],
                "properties": {
                    "id": {"type": "string"},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "edgeType": {"type": "string"},
                    "label": {"type": "string"},
                },
            },
        },
    }


def pipeline_graph_json_schema() -> dict[str, Any]:
    """Return the JSON schema for canonical pipeline graph data."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://policyengine.org/schemas/microplex-us/pipeline-graph-v1.json",
        "title": "Microplex-US pipeline graph",
        "type": "object",
        "required": [
            "schemaVersion",
            "contractVersion",
            "generatedFrom",
            "pipeline",
            "nodes",
            "edges",
        ],
        "properties": {
            "schemaVersion": {"const": US_PIPELINE_GRAPH_SCHEMA_VERSION},
            "contractVersion": {"type": "string"},
            "generatedFrom": {"type": "string"},
            "pipeline": {"type": "string"},
            "nodes": {
                "type": "array",
                "items": {"$ref": "#/$defs/node"},
            },
            "edges": {
                "type": "array",
                "items": {"$ref": "#/$defs/edge"},
            },
        },
        "$defs": {
            "node": {
                "type": "object",
                "required": ["id", "order", "title", "purpose", "viewer"],
                "properties": {
                    "id": {"type": "string"},
                    "order": {"type": "integer"},
                    "step": {"type": "string"},
                    "title": {"type": "string"},
                    "purpose": {"type": "string"},
                    "viewer": {"type": "object"},
                },
            },
            "edge": {
                "type": "object",
                "required": ["id", "source", "target", "label", "viewer"],
                "properties": {
                    "id": {"type": "string"},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "label": {"type": "string"},
                    "resourceKeys": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "viewer": {"type": "object"},
                },
            },
        },
    }


def pipeline_overlay_json_schema() -> dict[str, Any]:
    """Return the JSON schema for saved-run pipeline overlays."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://policyengine.org/schemas/microplex-us/pipeline-overlay-v1.json",
        "title": "Microplex-US pipeline overlay",
        "type": "object",
        "required": [
            "schemaVersion",
            "graphSchemaVersion",
            "contractVersion",
            "pipeline",
            "artifactRoot",
            "manifest",
            "stages",
        ],
        "properties": {
            "schemaVersion": {"const": US_PIPELINE_OVERLAY_SCHEMA_VERSION},
            "graphSchemaVersion": {"const": US_PIPELINE_GRAPH_SCHEMA_VERSION},
            "contractVersion": {"type": "string"},
            "generatedAt": {"type": ["string", "null"]},
            "pipeline": {"type": "string"},
            "artifactRoot": {"type": "string"},
            "manifest": {"type": "string"},
            "stages": {
                "type": "array",
                "items": {"$ref": "#/$defs/stage"},
            },
        },
        "$defs": {
            "stage": {
                "type": "object",
                "required": [
                    "id",
                    "title",
                    "status",
                    "lifecycleStatus",
                    "artifacts",
                    "diagnostics",
                    "validations",
                    "metrics",
                    "resume",
                    "benchmarkSummary",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "status": {"type": "string"},
                    "lifecycleStatus": {"type": "string"},
                    "outputManifest": {"type": ["string", "null"]},
                    "artifacts": {"type": "array"},
                    "diagnostics": {"type": "array"},
                    "validations": {"type": "array"},
                    "metrics": {"type": "array"},
                    "resume": {"type": "object"},
                    "benchmarkSummary": {"type": "object"},
                },
            },
        },
    }


def render_us_pipeline_markdown(
    *,
    graph: USPipelineGraph,
    internals: USPipelineInternals,
) -> str:
    """Render the canonical US pipeline map as generated Markdown docs."""

    lines = [
        "# US Pipeline Map",
        "",
        "Generated from `microplex_us.pipelines.stage_contracts`, "
        f"`{DEFAULT_US_PIPELINE_INTERNALS_MAP}`, and `@pipeline_node` "
        "decorators.",
        "",
        "This page is the static documentation surface for the build path. "
        "It lists canonical stages, substages, exact class or method references, "
        "source locations, and directed edges.",
        "",
        "## Canonical Stages",
        "",
        "| Stage | Title | Produces |",
        "| --- | --- | --- |",
    ]
    for stage in graph["nodes"]:
        produces = ", ".join(f"`{key}`" for key in stage.get("produces", ()))
        lines.append(
            f"| `{stage['id']}` {stage.get('step', '')} | "
            f"{_markdown_cell(stage.get('title', ''))} | "
            f"{produces or '`none`'} |"
        )
    lines.append("")

    internals_by_stage = {stage["id"]: stage for stage in internals["stages"]}
    for stage in graph["nodes"]:
        stage_id = stage["id"]
        internals_stage = internals_by_stage.get(stage_id, {})
        lines.extend(
            [
                f"## {stage.get('step', stage_id)}: {stage.get('title', stage_id)}",
                "",
                str(stage.get("purpose", "")),
                "",
            ]
        )
        substages = internals_stage.get("substages", ())
        for substage in substages if isinstance(substages, list) else ():
            _append_markdown_substage(lines, stage_id=stage_id, substage=substage)

    api_nodes = internals.get("apiNodes", ())
    if isinstance(api_nodes, list) and api_nodes:
        lines.extend(["## Pydoc API Surface", ""])
        for node in api_nodes:
            object_path = _node_code_reference(node)
            if not object_path:
                continue
            lines.extend(
                [
                    f"### `{object_path}`",
                    "",
                ]
            )
            signature = _optional_str(node.get("signature"))
            if signature:
                lines.extend(["```python", signature, "```", ""])
            docstring = _first_docstring_line(node)
            if docstring:
                lines.extend([docstring, ""])
            source = _node_source_reference(node)
            if source:
                lines.extend([f"- Source: `{source}`", ""])

    return "\n".join(lines).rstrip() + "\n"


def build_us_pipeline_docs_payloads(
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, dict[str, Any] | str]:
    """Return the generated docs payloads keyed by output filename."""

    graph = build_us_pipeline_graph()
    internals = build_us_pipeline_internals()
    payloads = {
        "us_pipeline_graph.schema.json": pipeline_graph_json_schema(),
        "us_pipeline_overlay.schema.json": pipeline_overlay_json_schema(),
        "us_pipeline_internals.schema.json": pipeline_internals_json_schema(),
        "us_pipeline_graph.json": cast(dict[str, Any], graph),
        "us_pipeline_internals.json": cast(dict[str, Any], internals),
        "us_pipeline_map.md": render_us_pipeline_markdown(
            graph=graph,
            internals=internals,
        ),
    }
    if artifact_root is not None:
        payloads["us_pipeline_overlay.json"] = cast(
            dict[str, Any],
            build_us_pipeline_overlay(artifact_root),
        )
    return payloads


def write_us_pipeline_docs(
    output_dir: str | Path,
    *,
    artifact_root: str | Path | None = None,
    check: bool = False,
) -> dict[str, Path]:
    """Write or check generated US pipeline docs data."""

    destination = Path(output_dir)
    payloads = build_us_pipeline_docs_payloads(artifact_root=artifact_root)
    paths = {filename: destination / filename for filename in payloads}
    mismatches: list[str] = []
    for filename, payload in payloads.items():
        path = paths[filename]
        content = payload if isinstance(payload, str) else _json_dump(payload)
        if check:
            if not path.exists():
                mismatches.append(f"missing generated file: {path}")
                continue
            existing = path.read_text()
            if existing != content:
                mismatches.append(_json_diff(path, existing, content))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    if mismatches:
        raise SystemExit(
            "US pipeline docs are stale or missing:\n" + "\n".join(mismatches)
        )
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for generated US pipeline docs data."""

    parser = argparse.ArgumentParser(
        description="Generate or check US Microplex pipeline graph and overlay data.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/generated",
        help="Directory for generated graph/schema/overlay JSON.",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Optional saved-run artifact directory used to generate an overlay.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated files differ from files already on disk.",
    )
    args = parser.parse_args(argv)
    paths = write_us_pipeline_docs(
        args.output_dir,
        artifact_root=args.artifact_root,
        check=args.check,
    )
    if not args.check:
        for path in paths.values():
            print(path)


def _graph_node(
    contract: USPipelineStageContract,
    order: int,
) -> USPipelineGraphNode:
    return {
        "id": contract.id,
        "order": order,
        "step": contract.step,
        "title": contract.title,
        "purpose": contract.purpose,
        "consumes": list(contract.consumes),
        "produces": list(contract.produces),
        "inputs": [resource.to_dict() for resource in contract.inputs],
        "outputs": [resource.to_dict() for resource in contract.outputs],
        "artifacts": [artifact.to_dict() for artifact in contract.artifacts],
        "diagnostics": list(contract.diagnostics),
        "validations": [validation.to_dict() for validation in contract.validations],
        "resume": {
            "mode": contract.resume_mode,
            "notes": contract.resume_notes,
        },
        "viewer": {
            "type": "stage",
            "elkPartition": order,
            "nodeKind": _node_kind(contract.id),
        },
    }


def _graph_edge(
    *,
    source: USPipelineStageContract,
    target: USPipelineStageContract,
) -> USPipelineGraphEdge:
    resource_keys = [
        resource.key
        for resource in target.inputs
        if resource.stage_id == source.id and resource.required
    ]
    label = ", ".join(resource_keys) if resource_keys else "stage handoff"
    return {
        "id": f"{source.id}__{target.id}",
        "source": source.id,
        "target": target.id,
        "label": label,
        "resourceKeys": resource_keys,
        "viewer": {
            "edgeKind": "stage_handoff",
            "routing": "elk_orthogonal",
        },
    }


def _load_pipeline_internals_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"pipeline internals map not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline internals map must be a JSON object: {path}")
    if payload.get("schemaVersion") != US_PIPELINE_INTERNALS_SCHEMA_VERSION:
        raise ValueError(
            "pipeline internals map schemaVersion must be "
            f"{US_PIPELINE_INTERNALS_SCHEMA_VERSION}"
        )
    return payload


def _pipeline_internals_substage(
    *,
    stage_id: str,
    payload: Mapping[str, Any],
    decorated_nodes: Mapping[str, Any],
    authored_nodes: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    substage_id = str(payload.get("id", ""))
    if not substage_id:
        raise ValueError(f"pipeline internals substage for {stage_id} requires an id")
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node_payload in _list_of_mappings(payload.get("nodes")):
        node = PipelineNode.from_mapping(node_payload).to_dict()
        node_id = str(node["id"])
        if node_id in nodes_by_id:
            raise ValueError(f"duplicate node id '{node_id}' in {substage_id}")
        nodes_by_id[node_id] = node
    edges = [
        PipelineEdge.from_mapping(edge_payload).to_dict()
        for edge_payload in _list_of_mappings(payload.get("edges"))
    ]
    referenced_node_ids_ordered = _referenced_node_ids(edges)
    referenced_node_ids = set(referenced_node_ids_ordered)
    referenced_decorated_node_ids: set[str] = set()
    for node_id in referenced_node_ids_ordered:
        if node_id in decorated_nodes:
            nodes_by_id.setdefault(node_id, decorated_nodes[node_id].to_node())
            referenced_decorated_node_ids.add(node_id)
    for node_id in referenced_node_ids_ordered:
        if node_id in authored_nodes:
            nodes_by_id.setdefault(node_id, authored_nodes[node_id])
    missing = sorted(referenced_node_ids - set(nodes_by_id))
    if missing:
        raise ValueError(
            f"{substage_id} references unknown pipeline node ids: {', '.join(missing)}"
        )
    return (
        {
            "id": substage_id,
            "stageId": stage_id,
            "title": str(payload.get("title", substage_id)),
            "description": str(payload.get("description", "")),
            "status": str(payload.get("status", "current")),
            "nodes": list(nodes_by_id.values()),
            "edges": edges,
        },
        referenced_decorated_node_ids,
    )


def _pipeline_internals_authored_nodes(
    source_map: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for stage_payload in _list_of_mappings(source_map.get("stages")):
        for substage_payload in _list_of_mappings(stage_payload.get("substages")):
            for node_payload in _list_of_mappings(substage_payload.get("nodes")):
                node = PipelineNode.from_mapping(node_payload).to_dict()
                nodes_by_id.setdefault(str(node["id"]), node)
    return nodes_by_id


def _referenced_node_ids(edges: Sequence[Mapping[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for edge in edges:
        for endpoint in ("source", "target"):
            node_id = str(edge[endpoint])
            if node_id in seen:
                continue
            ordered.append(node_id)
            seen.add(node_id)
    return ordered


def _load_or_build_stage_manifest(
    artifact_root: Path,
    manifest: dict[str, Any],
    *,
    prefer_saved: bool,
) -> dict[str, Any]:
    if prefer_saved:
        stage_manifest_ref = dict(manifest.get("artifacts", {})).get("stage_manifest")
        if isinstance(stage_manifest_ref, str) and stage_manifest_ref:
            stage_manifest_path = Path(stage_manifest_ref)
            if not stage_manifest_path.is_absolute():
                stage_manifest_path = artifact_root / stage_manifest_path
            if stage_manifest_path.exists():
                return dict(load_us_stage_manifest(stage_manifest_path))
    return dict(build_us_stage_manifest(artifact_root, manifest_payload=manifest))


def _overlay_stage(stage: Mapping[str, Any]) -> USPipelineOverlayStage:
    return {
        "id": str(stage.get("id", "")),
        "title": str(stage.get("title", "")),
        "status": cast(USStageStatus, stage.get("status", "missing")),
        "lifecycleStatus": cast(
            USStageLifecycleStatus,
            stage.get("lifecycleStatus", "pending"),
        ),
        "outputManifest": _optional_str(stage.get("outputManifest")),
        "startedAt": _optional_str(stage.get("startedAt")),
        "updatedAt": _optional_str(stage.get("updatedAt")),
        "completedAt": _optional_str(stage.get("completedAt")),
        "failedAt": _optional_str(stage.get("failedAt")),
        "deferredReason": _optional_str(stage.get("deferredReason")),
        "failure": _mapping_or_none(stage.get("failure")),
        "artifacts": _overlay_artifacts(stage.get("artifacts")),
        "diagnostics": _overlay_diagnostics(stage),
        "validations": _list_of_mappings(stage.get("validations")),
        "metrics": _list_of_mappings(stage.get("metrics")),
        "resume": _mapping_or_empty(stage.get("resume")),
        "benchmarkSummary": _benchmark_summary(stage),
    }


def _overlay_artifacts(value: Any) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for artifact in value if isinstance(value, list) else ():
        if not isinstance(artifact, dict):
            continue
        artifacts.append(
            {
                "key": str(artifact.get("key", "")),
                "description": str(artifact.get("description", "")),
                "path": _safe_relative_ref(artifact.get("path")),
                "required": bool(artifact.get("required", False)),
                "resumeRole": artifact.get("resume_role"),
                "format": str(artifact.get("format", "unknown")),
                "exists": bool(artifact.get("exists", False)),
                "referenced": bool(artifact.get("referenced", False)),
            }
        )
    return artifacts


def _overlay_diagnostics(stage: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for key in stage.get("diagnostics", ()) if isinstance(stage, dict) else ():
        diagnostics.append({"key": str(key)})
    return diagnostics


def _benchmark_summary(stage: Mapping[str, Any]) -> dict[str, Any]:
    if stage.get("id") != "09_validation_benchmarking":
        return {}
    metrics = stage.get("metrics")
    if not isinstance(metrics, list):
        return {}
    return {
        str(metric["label"]): metric.get("value")
        for metric in metrics
        if isinstance(metric, dict) and "label" in metric
    }


def _node_kind(
    stage_id: str,
) -> Literal[
    "configuration",
    "source",
    "planning",
    "build",
    "calibration",
    "publication",
    "validation",
]:
    if stage_id == "01_run_profile":
        return "configuration"
    if stage_id == "02_source_loading":
        return "source"
    if stage_id == "03_source_planning":
        return "planning"
    if stage_id == "07_calibration":
        return "calibration"
    if stage_id == "08_dataset_assembly":
        return "publication"
    if stage_id == "09_validation_benchmarking":
        return "validation"
    return "build"


def _relative_manifest_ref(stage_manifest: Mapping[str, Any]) -> str:
    value = stage_manifest.get("manifest")
    return _safe_relative_ref(value) or "manifest.json"


def _safe_relative_ref(value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path.name
    return str(path)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _append_markdown_substage(
    lines: list[str],
    *,
    stage_id: str,
    substage: Mapping[str, Any],
) -> None:
    substage_id = str(substage.get("id", "unknown"))
    title = str(substage.get("title", substage_id))
    description = str(substage.get("description", ""))
    lines.extend(
        [
            f"### {title}",
            "",
            description,
            "",
            f"- Substage ID: `{substage_id}`",
            f"- Canonical stage: `{stage_id}`",
            f"- Status: `{substage.get('status', 'current')}`",
            "",
            "| Node | Type | Status | API refs |",
            "| --- | --- | --- | --- |",
        ]
    )
    for node in _list_of_mappings(substage.get("nodes")):
        refs = _node_markdown_refs(node)
        lines.append(
            f"| `{node.get('id', '')}` {_markdown_cell(node.get('label', ''))} | "
            f"`{node.get('nodeType', 'process')}` | "
            f"`{node.get('status', 'current')}` | "
            f"{refs or ''} |"
        )
    lines.extend(["", "#### Edges", ""])
    for edge in _list_of_mappings(substage.get("edges")):
        label = f" ({edge['label']})" if edge.get("label") else ""
        lines.append(
            f"- `{edge.get('source', '')}` -> `{edge.get('target', '')}` "
            f"`{edge.get('edgeType', 'data_flow')}`{label}"
        )
    lines.append("")


def _node_markdown_refs(node: Mapping[str, Any]) -> str:
    refs: list[str] = []
    object_path = _optional_str(node.get("objectPath"))
    if object_path:
        refs.append(object_path)
    for ref in _tuple_of_str(node.get("apiRefs")):
        if ref not in refs:
            refs.append(ref)
    pydoc = _optional_str(node.get("pydoc"))
    if pydoc and pydoc not in refs:
        refs.append(pydoc)
    return ", ".join(f"`{ref}`" for ref in refs)


def _node_code_reference(node: Mapping[str, Any]) -> str | None:
    object_path = _optional_str(node.get("objectPath"))
    if object_path:
        return object_path
    api_refs = _tuple_of_str(node.get("apiRefs"))
    if api_refs:
        return api_refs[0]
    return _optional_str(node.get("pydoc"))


def _node_source_reference(node: Mapping[str, Any]) -> str | None:
    source_file = _optional_str(node.get("sourceFile"))
    if source_file is None:
        return None
    line = node.get("line")
    return f"{source_file}:{line}" if isinstance(line, int) else source_file


def _first_docstring_line(node: Mapping[str, Any]) -> str:
    docstring = _optional_str(node.get("docstring")) or _optional_str(
        node.get("description")
    )
    return docstring.splitlines()[0] if docstring else ""


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _json_dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _json_diff(path: Path, existing: str, expected: str) -> str:
    diff = difflib.unified_diff(
        existing.splitlines(),
        expected.splitlines(),
        fromfile=str(path),
        tofile=f"{path} (regenerated)",
        lineterm="",
    )
    return "\n".join(diff)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


__all__ = [
    "US_PIPELINE_GRAPH_SCHEMA_VERSION",
    "US_PIPELINE_INTERNALS_SCHEMA_VERSION",
    "US_PIPELINE_OVERLAY_SCHEMA_VERSION",
    "USPipelineGraph",
    "USPipelineGraphEdge",
    "USPipelineGraphNode",
    "USPipelineInternals",
    "USPipelineOverlay",
    "USPipelineOverlayStage",
    "build_us_pipeline_docs_payloads",
    "build_us_pipeline_graph",
    "build_us_pipeline_internals",
    "build_us_pipeline_overlay",
    "pipeline_graph_json_schema",
    "pipeline_internals_json_schema",
    "pipeline_overlay_json_schema",
    "render_us_pipeline_markdown",
    "write_us_pipeline_docs",
]
