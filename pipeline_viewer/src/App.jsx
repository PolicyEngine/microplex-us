import { useMemo, useState } from "react";
import { Background, Controls, ReactFlow, ReactFlowProvider } from "@xyflow/react";
import defaultGraph from "./fixtures/us_pipeline_graph.json";
import defaultOverlay from "./fixtures/us_pipeline_overlay.json";
import ElkEdge from "./components/ElkEdge.jsx";
import PipelineStageNode from "./components/PipelineStageNode.jsx";
import useElkLayout from "./hooks/useElkLayout.js";

const nodeTypes = { pipelineStage: PipelineStageNode };
const edgeTypes = { elk: ElkEdge };

const LIFECYCLE_LABELS = {
  pending: "Pending",
  running: "Running",
  complete: "Complete",
  failed: "Failed",
  deferred: "Deferred",
};

function PipelineFlow({ graph, overlay, selectedStageId, setSelectedStageId }) {
  const overlayByStage = useMemo(
    () => new Map((overlay?.stages || []).map((stage) => [stage.id, stage])),
    [overlay],
  );
  const initialNodes = useMemo(
    () =>
      graph.nodes.map((node) => {
        const stageOverlay = overlayByStage.get(node.id);
        return {
          id: node.id,
          type: "pipelineStage",
          data: {
            ...node,
            overlay: stageOverlay || null,
            selected: selectedStageId === node.id,
          },
        };
      }),
    [graph, overlayByStage, selectedStageId],
  );
  const initialEdges = useMemo(
    () =>
      graph.edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.label,
        data: edge.viewer,
        style: {
          stroke: "#475569",
          strokeWidth: 2,
        },
        markerEnd: {
          type: "arrowclosed",
          color: "#475569",
          width: 16,
          height: 16,
        },
      })),
    [graph],
  );
  const { nodes, edges, layoutDone } = useElkLayout(initialNodes, initialEdges);

  if (!layoutDone) {
    return <div className="loading">Computing ELK layout...</div>;
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      minZoom={0.15}
      maxZoom={2}
      fitView
      fitViewOptions={{ padding: 0.16 }}
      onNodeClick={(_, node) => setSelectedStageId(node.id)}
    >
      <Background color="#d8dee9" gap={24} size={1} />
      <Controls position="bottom-right" />
    </ReactFlow>
  );
}

export default function App() {
  const [graph, setGraph] = useState(defaultGraph);
  const [overlay, setOverlay] = useState(defaultOverlay);
  const [selectedStageId, setSelectedStageId] = useState(defaultGraph.nodes[0].id);
  const selectedStage = useMemo(
    () => overlay?.stages?.find((stage) => stage.id === selectedStageId),
    [overlay, selectedStageId],
  );
  const selectedNode = useMemo(
    () => graph.nodes.find((node) => node.id === selectedStageId),
    [graph, selectedStageId],
  );

  async function loadJsonFile(file, setter) {
    const text = await file.text();
    setter(JSON.parse(text));
  }

  return (
    <main className="viewer-shell">
      <header className="viewer-header">
        <div>
          <p className="eyebrow">Microplex US</p>
          <h1>Pipeline Viewer</h1>
        </div>
        <div className="loaders">
          <label>
            Graph JSON
            <input
              type="file"
              accept="application/json,.json"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) loadJsonFile(file, setGraph);
              }}
            />
          </label>
          <label>
            Overlay JSON
            <input
              type="file"
              accept="application/json,.json"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) loadJsonFile(file, setOverlay);
              }}
            />
          </label>
        </div>
      </header>

      <section className="workspace">
        <section className="diagram-panel">
          <ReactFlowProvider>
            <PipelineFlow
              graph={graph}
              overlay={overlay}
              selectedStageId={selectedStageId}
              setSelectedStageId={setSelectedStageId}
            />
          </ReactFlowProvider>
        </section>
        <StageDetails stage={selectedStage} node={selectedNode} />
      </section>
    </main>
  );
}

function StageDetails({ stage, node }) {
  if (!node) {
    return <aside className="details-panel">Select a stage.</aside>;
  }
  const lifecycle = stage?.lifecycleStatus || "pending";
  const readiness = stage?.status || "missing";
  const artifacts = stage?.artifacts || [];
  const diagnostics = stage?.diagnostics || [];
  const validations = stage?.validations || [];
  const metrics = stage?.metrics || [];
  return (
    <aside className="details-panel">
      <p className="eyebrow">{node.step}</p>
      <h2>{node.title}</h2>
      <p className="purpose">{node.purpose}</p>
      <div className="status-row">
        <span className={`status-pill ${lifecycle}`}>{LIFECYCLE_LABELS[lifecycle]}</span>
        <span className="status-pill neutral">{readiness.replaceAll("_", " ")}</span>
      </div>
      {metrics.length > 0 && (
        <DetailSection title="Metrics">
          {metrics.map((metric) => (
            <div className="detail-row" key={metric.label}>
              <span>{metric.label}</span>
              <strong>{String(metric.value ?? "-")}</strong>
            </div>
          ))}
        </DetailSection>
      )}
      <DetailSection title="Artifacts">
        {artifacts.length ? (
          artifacts.map((artifact) => (
            <div className="artifact-row" key={artifact.key}>
              <span>{artifact.key}</span>
              <small>{artifact.path || "not written"}</small>
              <b>{artifact.exists ? "exists" : "missing"}</b>
            </div>
          ))
        ) : (
          <p className="muted">No stage artifact records.</p>
        )}
      </DetailSection>
      <DetailSection title="Diagnostics">
        {diagnostics.length ? (
          diagnostics.map((diagnostic) => (
            <span className="token" key={diagnostic.key}>
              {diagnostic.key}
            </span>
          ))
        ) : (
          <p className="muted">No diagnostics declared.</p>
        )}
      </DetailSection>
      <DetailSection title="Validation Hooks">
        {validations.length ? (
          validations.map((validation) => (
            <div className="detail-row" key={validation.key}>
              <span>{validation.key}</span>
              <strong>{validation.status}</strong>
            </div>
          ))
        ) : (
          <p className="muted">No validation hooks declared.</p>
        )}
      </DetailSection>
      {stage?.failure && (
        <DetailSection title="Failure">
          <p className="failure-message">
            {stage.failure.errorType}: {stage.failure.message}
          </p>
        </DetailSection>
      )}
    </aside>
  );
}

function DetailSection({ title, children }) {
  return (
    <section className="detail-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}
