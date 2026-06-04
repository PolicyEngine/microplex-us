"""Static pipeline documentation schema helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal[
    "entrypoint",
    "stage",
    "substage",
    "process",
    "library",
    "artifact",
    "utility",
    "external",
    "validation",
    "infrastructure",
]
EdgeType = Literal[
    "data_flow",
    "produces_artifact",
    "uses_library",
    "uses_utility",
    "external_source",
    "validates",
    "documents",
    "conditional",
    "informational",
]
NodeStatus = Literal["current", "planned", "legacy", "missing"]
Stability = Literal["stable", "evolving", "experimental"]


@dataclass(frozen=True)
class PipelineNode:
    """One documented pipeline function, artifact, library, or external input."""

    id: str
    label: str
    node_type: NodeType = "process"
    description: str = ""
    details: str = ""
    source_file: str | None = None
    line: int | None = None
    object_path: str | None = None
    kind: str | None = None
    signature: str | None = None
    docstring: str | None = None
    status: NodeStatus = "current"
    stability: Stability = "stable"
    artifacts_in: tuple[str, ...] = field(default_factory=tuple)
    artifacts_out: tuple[str, ...] = field(default_factory=tuple)
    api_refs: tuple[str, ...] = field(default_factory=tuple)
    pydoc: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> PipelineNode:
        """Build a node from either Python-style or JSON-style keys."""

        return cls(
            id=str(value["id"]),
            label=str(value["label"]),
            node_type=value.get("node_type", value.get("nodeType", "process")),
            description=str(value.get("description", "")),
            details=str(value.get("details", "")),
            source_file=_optional_str(
                value.get("source_file", value.get("sourceFile"))
            ),
            line=_optional_int(value.get("line")),
            object_path=_optional_str(
                value.get("object_path", value.get("objectPath"))
            ),
            kind=_optional_str(value.get("kind")),
            signature=_optional_str(value.get("signature")),
            docstring=_optional_str(value.get("docstring")),
            status=value.get("status", "current"),
            stability=value.get("stability", "stable"),
            artifacts_in=_tuple_of_str(
                value.get("artifacts_in", value.get("artifactsIn", ()))
            ),
            artifacts_out=_tuple_of_str(
                value.get("artifacts_out", value.get("artifactsOut", ()))
            ),
            api_refs=_tuple_of_str(value.get("api_refs", value.get("apiRefs", ()))),
            pydoc=_optional_str(value.get("pydoc")),
            notes=_tuple_of_str(value.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready node payload."""

        return {
            "id": self.id,
            "label": self.label,
            "nodeType": self.node_type,
            "description": self.description,
            "details": self.details,
            "sourceFile": self.source_file,
            "line": self.line,
            "objectPath": self.object_path,
            "kind": self.kind,
            "signature": self.signature,
            "docstring": self.docstring,
            "status": self.status,
            "stability": self.stability,
            "artifactsIn": list(self.artifacts_in),
            "artifactsOut": list(self.artifacts_out),
            "apiRefs": list(self.api_refs),
            "pydoc": self.pydoc,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PipelineEdge:
    """One directed static pipeline documentation edge."""

    source: str
    target: str
    edge_type: EdgeType = "data_flow"
    label: str = ""
    status: NodeStatus = "current"
    stability: Stability = "stable"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> PipelineEdge:
        """Build an edge from either Python-style or JSON-style keys."""

        return cls(
            source=str(value["source"]),
            target=str(value["target"]),
            edge_type=value.get("edge_type", value.get("edgeType", "data_flow")),
            label=str(value.get("label", "")),
            status=value.get("status", "current"),
            stability=value.get("stability", "stable"),
            notes=_tuple_of_str(value.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready edge payload."""

        return {
            "id": f"{self.source}__{self.target}",
            "source": self.source,
            "target": self.target,
            "edgeType": self.edge_type,
            "label": self.label,
            "status": self.status,
            "stability": self.stability,
            "notes": list(self.notes),
        }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


__all__ = [
    "EdgeType",
    "NodeStatus",
    "NodeType",
    "PipelineEdge",
    "PipelineNode",
    "Stability",
]
