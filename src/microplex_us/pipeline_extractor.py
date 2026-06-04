"""Static extractor for pipeline documentation decorators."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microplex_us.pipeline_schema import PipelineNode


@dataclass(frozen=True)
class DocumentedObject:
    """One function, method, or class discovered by the static extractor."""

    id: str
    object_path: str
    source_file: str
    line: int
    kind: str
    signature: str | None
    docstring: str
    metadata: dict[str, Any]

    def to_node(self) -> dict[str, Any]:
        """Return this decorated object as a pipeline node payload."""

        metadata = dict(self.metadata)
        metadata.setdefault("label", self.id)
        node = PipelineNode.from_mapping(metadata)
        payload = node.to_dict()
        payload.update(
            {
                "sourceFile": self.source_file,
                "line": self.line,
                "objectPath": self.object_path,
                "kind": self.kind,
                "signature": self.signature,
                "docstring": self.docstring,
                "pydoc": node.pydoc or self.object_path,
            }
        )
        return payload


def collect_pipeline_nodes(
    *,
    repo_root: str | Path,
    source_roots: tuple[str, ...] = ("src/microplex_us",),
) -> dict[str, DocumentedObject]:
    """Collect all ``@pipeline_node`` metadata without importing package code."""

    root = Path(repo_root)
    objects: dict[str, DocumentedObject] = {}
    for source_root in source_roots:
        for path in sorted((root / source_root).rglob("*.py")):
            if not path.is_file():
                continue
            visitor = DecoratedObjectVisitor(
                module_path=_module_path(path, root),
                source_file=str(path.relative_to(root)),
            )
            visitor.visit(ast.parse(path.read_text(), filename=str(path)))
            for documented in visitor.objects:
                if documented.id in objects:
                    existing = objects[documented.id]
                    raise ValueError(
                        "duplicate pipeline node id "
                        f"'{documented.id}' in {existing.source_file} "
                        f"and {documented.source_file}"
                    )
                objects[documented.id] = documented
    return objects


class DecoratedObjectVisitor(ast.NodeVisitor):
    """AST visitor that finds objects decorated with ``@pipeline_node``."""

    def __init__(self, *, module_path: str, source_file: str) -> None:
        self.module_path = module_path
        self.source_file = source_file
        self.objects: list[DocumentedObject] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_node(node, kind="class")
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self._class_stack else "function"
        self._record_node(node, kind=kind)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = "async_method" if self._class_stack else "async_function"
        self._record_node(node, kind=kind)
        self.generic_visit(node)

    def _record_node(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        kind: str,
    ) -> None:
        metadata = _pipeline_node_metadata(node.decorator_list)
        if metadata is None:
            return
        node_id = metadata.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(
                f"{self.source_file}:{node.lineno} pipeline_node requires an id"
            )
        self.objects.append(
            DocumentedObject(
                id=node_id,
                object_path=self._object_path(node.name),
                source_file=self.source_file,
                line=node.lineno,
                kind=kind,
                signature=_signature(node)
                if not isinstance(node, ast.ClassDef)
                else None,
                docstring=ast.get_docstring(node) or "",
                metadata=metadata,
            )
        )

    def _object_path(self, object_name: str) -> str:
        parts = [self.module_path, *self._class_stack, object_name]
        return ".".join(parts)


def _pipeline_node_metadata(
    decorators: list[ast.expr],
) -> dict[str, Any] | None:
    for decorator in decorators:
        if not isinstance(decorator, ast.Call):
            continue
        if _call_name(decorator.func) != "pipeline_node":
            continue
        if decorator.args:
            first_arg = decorator.args[0]
            if (
                isinstance(first_arg, ast.Call)
                and _call_name(first_arg.func) == "PipelineNode"
            ):
                return _literal_kwargs(first_arg)
            return _literal_value(first_arg)
        return _literal_kwargs(decorator)
    return None


def _literal_kwargs(call: ast.Call) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        metadata[keyword.arg] = _literal_value(keyword.value)
    return metadata


def _literal_value(value: ast.expr) -> Any:
    try:
        return ast.literal_eval(value)
    except ValueError as exc:
        raise ValueError("pipeline_node metadata must be literal") from exc


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    parts: list[str] = []
    parts.extend(arg.arg for arg in args.posonlyargs)
    if args.posonlyargs:
        parts.append("/")
    parts.extend(arg.arg for arg in args.args)
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    parts.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    return f"{node.name}({', '.join(parts)})"


def _module_path(path: Path, repo_root: Path) -> str:
    relative = path.relative_to(repo_root)
    if relative.parts and relative.parts[0] == "src":
        relative = Path(*relative.parts[1:])
    return ".".join(relative.with_suffix("").parts)


__all__ = ["DocumentedObject", "collect_pipeline_nodes"]
