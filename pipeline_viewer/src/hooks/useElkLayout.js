import { useEffect, useMemo, useState } from "react";
import ELK from "elkjs/lib/elk.bundled.js";

const elk = new ELK();

const ELK_OPTIONS = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
  "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
  "elk.layered.spacing.edgeNodeBetweenLayers": "36",
  "elk.layered.spacing.nodeNodeBetweenLayers": "130",
  "elk.partitioning.activate": "true",
  "elk.spacing.edgeEdge": "24",
  "elk.spacing.edgeNode": "36",
  "elk.spacing.nodeNode": "70",
};

function estimateNodeSize(node) {
  const purpose =
    node.data?.docstring ||
    node.data?.details ||
    node.data?.purpose ||
    node.data?.description ||
    "";
  const codeRef = node.data?.objectPath || node.data?.pydoc || "";
  const signature = node.data?.signature || "";
  const width = node.data?.nodeWidth || (node.type === "pipelineInternal" ? 330 : 300);
  const baseHeight = node.type === "pipelineInternal" ? 138 : 82;
  const minHeight = node.type === "pipelineInternal" ? 154 : 112;
  return {
    width,
    height: Math.max(
      minHeight,
      baseHeight +
        Math.ceil(String(purpose).length / 60) * 18 +
        Math.ceil(String(codeRef).length / 48) * 12 +
        Math.ceil(String(signature).length / 48) * 12,
    ),
  };
}

function assignHandles(sourcePos, targetPos) {
  const dx = targetPos.x - sourcePos.x;
  const dy = targetPos.y - sourcePos.y;
  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0
      ? { sourceHandle: "sr", targetHandle: "tl" }
      : { sourceHandle: "sl", targetHandle: "tr" };
  }
  return dy >= 0
    ? { sourceHandle: "sb", targetHandle: "tt" }
    : { sourceHandle: "st", targetHandle: "tb" };
}

export default function useElkLayout(initialNodes, initialEdges) {
  const layoutKey = useMemo(
    () =>
      JSON.stringify({
        nodes: initialNodes.map((node) => [node.id, node.data?.overlay?.lifecycleStatus]),
        edges: initialEdges.map((edge) => edge.id),
      }),
    [initialNodes, initialEdges],
  );
  const [layoutState, setLayoutState] = useState({
    key: "",
    nodes: [],
    edges: [],
    layoutDone: false,
  });

  useEffect(() => {
    let cancelled = false;
    async function runLayout() {
      setLayoutState((current) => ({
        ...current,
        key: layoutKey,
        layoutDone: false,
      }));
      const graph = {
        id: "root",
        layoutOptions: ELK_OPTIONS,
        children: initialNodes.map((node, index) => {
          const size = estimateNodeSize(node);
          return {
            id: node.id,
            width: size.width,
            height: size.height,
            layoutOptions: {
              "elk.partitioning.partition": String(node.data.order ?? index),
            },
          };
        }),
        edges: initialEdges.map((edge) => ({
          id: edge.id,
          sources: [edge.source],
          targets: [edge.target],
        })),
      };
      try {
        const result = await elk.layout(graph);
        if (cancelled) return;
        const positionMap = Object.fromEntries(
          (result.children || []).map((child) => [child.id, { x: child.x, y: child.y }]),
        );
        const routeMap = Object.fromEntries(
          (result.edges || [])
            .filter((edge) => edge.sections?.length)
            .map((edge) => [edge.id, edge.sections[0]]),
        );
        const nodes = initialNodes.map((node) => ({
          ...node,
          position: positionMap[node.id] || { x: 0, y: 0 },
        }));
        const edges = initialEdges.map((edge) => {
          const handles = assignHandles(
            positionMap[edge.source] || { x: 0, y: 0 },
            positionMap[edge.target] || { x: 0, y: 0 },
          );
          return {
            ...edge,
            ...handles,
            type: "elk",
            data: {
              ...(edge.data || {}),
              elkRoute: routeMap[edge.id] || null,
            },
          };
        });
        setLayoutState({ key: layoutKey, nodes, edges, layoutDone: true });
      } catch (error) {
        if (cancelled) return;
        console.error("ELK layout failed", error);
        setLayoutState({
          key: layoutKey,
          nodes: initialNodes.map((node, index) => ({
            ...node,
            position: { x: 0, y: index * 180 },
          })),
          edges: initialEdges,
          layoutDone: true,
        });
      }
    }
    runLayout();
    return () => {
      cancelled = true;
    };
  }, [layoutKey, initialNodes, initialEdges]);

  return layoutState.key === layoutKey
    ? layoutState
    : { nodes: [], edges: [], layoutDone: false };
}
