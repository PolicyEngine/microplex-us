"""No-op decorators for static pipeline documentation extraction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, overload

from microplex_us.pipeline_schema import PipelineNode

PipelineTarget = TypeVar("PipelineTarget")


@overload
def pipeline_node(
    node: PipelineNode, /
) -> Callable[[PipelineTarget], PipelineTarget]: ...


@overload
def pipeline_node(**metadata: Any) -> Callable[[PipelineTarget], PipelineTarget]: ...


def pipeline_node(
    *args: Any, **metadata: Any
) -> Callable[[PipelineTarget], PipelineTarget]:
    """Attach static pipeline docs metadata without changing runtime behavior."""

    if len(args) > 1:
        raise TypeError("pipeline_node accepts at most one positional argument")
    if args and metadata:
        raise TypeError(
            "pipeline_node accepts either a PipelineNode or keyword metadata"
        )
    if args:
        node = args[0]
        if not isinstance(node, PipelineNode):
            raise TypeError("pipeline_node positional argument must be a PipelineNode")
        payload = node.to_dict()
    else:
        payload = PipelineNode.from_mapping(dict(metadata)).to_dict()

    def decorate(target: PipelineTarget) -> PipelineTarget:
        setattr(target, "_pipeline_node", payload)
        return target

    return decorate


__all__ = ["pipeline_node"]
