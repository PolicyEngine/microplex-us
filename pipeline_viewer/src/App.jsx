import { useMemo, useState } from "react";
import { Background, Controls, ReactFlow, ReactFlowProvider } from "@xyflow/react";
import defaultGraph from "./fixtures/us_pipeline_graph.json";
import defaultInternals from "./fixtures/us_pipeline_internals.json";
import defaultOverlay from "./fixtures/us_pipeline_overlay.json";
import ElkEdge from "./components/ElkEdge.jsx";
import PipelineInternalNode from "./components/PipelineInternalNode.jsx";
import PipelineStageNode from "./components/PipelineStageNode.jsx";
import useElkLayout from "./hooks/useElkLayout.js";

const nodeTypes = {
  pipelineInternal: PipelineInternalNode,
  pipelineStage: PipelineStageNode,
};
const edgeTypes = { elk: ElkEdge };

const LIFECYCLE_LABELS = {
  pending: "Pending",
  running: "Running",
  complete: "Complete",
  failed: "Failed",
  deferred: "Deferred",
};

function PipelineOverviewFlow({ graph, overlay, setStageView }) {
  const overlayByStage = useMemo(
    () => new Map((overlay?.stages || []).map((stage) => [stage.id, stage])),
    [overlay],
  );
  const initialNodes = useMemo(() => {
    return graph.nodes.map((stage, stageIndex) => {
      const stageOverlay = overlayByStage.get(stage.id);
      return {
        id: stage.id,
        type: "pipelineStage",
        data: {
          ...stage,
          overlay: stageOverlay || null,
          selected: false,
          order: stageIndex * 100,
        },
      };
    });
  }, [graph, overlayByStage]);
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
    return <div className="loading">Computing pipeline overview layout...</div>;
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      minZoom={0.08}
      maxZoom={2}
      fitView
      fitViewOptions={{ padding: 0.14 }}
      onNodeClick={(_, node) => {
        setStageView(node.id, "all");
      }}
    >
      <Background color="#d8dee9" gap={24} size={1} />
      <Controls position="bottom-right" />
    </ReactFlow>
  );
}

function StageMachineryFlow({ internals, selectedStageId, selectedSubstageId }) {
  const selectedStage = useMemo(
    () => internals.stages.find((stage) => stage.id === selectedStageId),
    [internals, selectedStageId],
  );
  const substages = useMemo(() => {
    const stageSubstages = selectedStage?.substages || [];
    if (selectedSubstageId === "all") return stageSubstages;
    return stageSubstages.filter((substage) => substage.id === selectedSubstageId);
  }, [selectedStage, selectedSubstageId]);
  const initialNodes = useMemo(() => {
    const nodesById = new Map();
    substages.forEach((substage, substageIndex) => {
      nodesById.set(`substage:${substage.id}`, {
        id: `substage:${substage.id}`,
        type: "pipelineInternal",
        data: {
          id: substage.id,
          label: substage.title,
          nodeType: "substage",
          description: substage.description,
          order: substageIndex * 10,
          nodeWidth: 280,
        },
      });
      substage.nodes.forEach((node, nodeIndex) => {
        if (nodesById.has(node.id)) return;
        nodesById.set(node.id, {
          id: node.id,
          type: "pipelineInternal",
          data: {
            ...node,
            order: substageIndex * 10 + nodeIndex + 1,
          },
        });
      });
    });
    return Array.from(nodesById.values());
  }, [substages]);
  const initialEdges = useMemo(() => {
    const edges = [];
    substages.forEach((substage) => {
      const entryTarget = substage.edges[0]?.source || substage.nodes[0]?.id;
      if (entryTarget) {
        edges.push({
          id: `substage:${substage.id}__${entryTarget}`,
          source: `substage:${substage.id}`,
          target: entryTarget,
          label: substage.id,
          data: { edgeKind: "substage_entry" },
          style: {
            stroke: "#94a3b8",
            strokeDasharray: "4 4",
            strokeWidth: 1.4,
          },
          markerEnd: {
            type: "arrowclosed",
            color: "#94a3b8",
            width: 14,
            height: 14,
          },
        });
      }
      substage.edges.forEach((edge) => {
        edges.push({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          data: { edgeKind: edge.edgeType },
          style: edgeStyle(edge.edgeType),
          markerEnd: {
            type: "arrowclosed",
            color: edgeColor(edge.edgeType),
            width: 14,
            height: 14,
          },
        });
      });
    });
    return edges;
  }, [substages]);
  const { nodes, edges, layoutDone } = useElkLayout(initialNodes, initialEdges);

  if (!selectedStage) {
    return <div className="loading">Select a stage.</div>;
  }
  if (!layoutDone) {
    return <div className="loading">Computing stage machinery layout...</div>;
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      minZoom={0.15}
      maxZoom={2.4}
      fitView
      fitViewOptions={{ padding: 0.18 }}
    >
      <Background color="#d8dee9" gap={22} size={1} />
      <Controls position="bottom-right" />
    </ReactFlow>
  );
}

export default function App() {
  const [graph, setGraph] = useState(defaultGraph);
  const [internals, setInternals] = useState(defaultInternals);
  const [overlay, setOverlay] = useState(defaultOverlay);
  const [activeView, setActiveView] = useState("pipeline");
  const [selectedSubstageId, setSelectedSubstageId] = useState("all");
  const selectedStageId = activeView === "pipeline" ? null : activeView;
  const selectedStage = useMemo(
    () => overlay?.stages?.find((stage) => stage.id === selectedStageId),
    [overlay, selectedStageId],
  );
  const selectedNode = useMemo(
    () => graph.nodes.find((node) => node.id === selectedStageId),
    [graph, selectedStageId],
  );
  const selectedInternalsStage = useMemo(
    () => internals.stages.find((stage) => stage.id === selectedStageId),
    [internals, selectedStageId],
  );
  const pipelineSummary = useMemo(
    () => buildPipelineSummary(graph, internals),
    [graph, internals],
  );

  async function loadJsonFile(file, setter) {
    const text = await file.text();
    setter(JSON.parse(text));
  }

  function setStageView(stageId, substageId = "all") {
    setActiveView(stageId);
    setSelectedSubstageId(substageId);
  }

  return (
    <main className="viewer-shell">
      <header className="viewer-header">
        <div>
          <p className="eyebrow">Microplex US</p>
          <h1>Pipeline Viewer</h1>
        </div>
        <details className="data-loaders">
          <summary>Advanced data</summary>
          <p>
            Defaults load from <code>pipeline_viewer/src/fixtures</code>. Regenerated
            sources live in <code>docs/generated</code>.
          </p>
          <div className="loader-grid">
            <label>
              Stage graph
              <small>
                <code>docs/generated/us_pipeline_graph.json</code>
              </small>
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
              Saved-run overlay
              <small>
                generated with <code>--artifact-root</code>
              </small>
              <input
                type="file"
                accept="application/json,.json"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) loadJsonFile(file, setOverlay);
                }}
              />
            </label>
            <label>
              Machinery internals
              <small>
                <code>docs/generated/us_pipeline_internals.json</code>
              </small>
              <input
                type="file"
                accept="application/json,.json"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) loadJsonFile(file, setInternals);
                }}
              />
            </label>
          </div>
        </details>
      </header>

      <section className="workspace">
        <PipelineNav
          activeView={activeView}
          graph={graph}
          internals={internals}
          overlay={overlay}
          setActiveView={setActiveView}
          setStageView={setStageView}
        />
        <section className="content-panel">
          {activeView === "pipeline" ? (
            <section className="diagram-panel overview-diagram">
              <div className="panel-toolbar">
                <div>
                  <p className="eyebrow">Entire Pipeline</p>
                  <strong>Canonical stages</strong>
                </div>
              </div>
              <ReactFlowProvider>
                <PipelineOverviewFlow
                  graph={graph}
                  overlay={overlay}
                  setStageView={setStageView}
                />
              </ReactFlowProvider>
            </section>
          ) : (
            <section className="diagram-panel machinery-diagram">
              <div className="panel-toolbar">
                <div>
                  <p className="eyebrow">Stage Machinery</p>
                  <strong>{selectedInternalsStage?.title || "No stage selected"}</strong>
                </div>
                <label>
                  Substage
                  <select
                    value={selectedSubstageId}
                    onChange={(event) => setSelectedSubstageId(event.target.value)}
                  >
                    <option value="all">All substages</option>
                    {(selectedInternalsStage?.substages || []).map((substage) => (
                      <option value={substage.id} key={substage.id}>
                        {substage.id}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <ReactFlowProvider>
                <StageMachineryFlow
                  internals={internals}
                  selectedStageId={selectedStageId}
                  selectedSubstageId={selectedSubstageId}
                />
              </ReactFlowProvider>
            </section>
          )}
        </section>
        {activeView === "pipeline" ? (
          <PipelineSummaryPanel summary={pipelineSummary} />
        ) : (
          <StageDetails
            stage={selectedStage}
            node={selectedNode}
            internalsStage={selectedInternalsStage}
          />
        )}
      </section>
    </main>
  );
}

function PipelineNav({
  activeView,
  graph,
  internals,
  overlay,
  setActiveView,
  setStageView,
}) {
  const overlayByStage = useMemo(
    () => new Map((overlay?.stages || []).map((stage) => [stage.id, stage])),
    [overlay],
  );
  const internalsByStage = useMemo(
    () => new Map((internals?.stages || []).map((stage) => [stage.id, stage])),
    [internals],
  );
  return (
    <nav className="pipeline-nav" aria-label="Pipeline views">
      <button
        className={`nav-item ${activeView === "pipeline" ? "active" : ""}`}
        type="button"
        onClick={() => setActiveView("pipeline")}
      >
        <span>Entire pipeline</span>
        <small>{graph.nodes.length} stages</small>
      </button>
      <div className="nav-section-label">Stages</div>
      {graph.nodes.map((stage) => {
        const lifecycle = overlayByStage.get(stage.id)?.lifecycleStatus || "pending";
        const substageCount = internalsByStage.get(stage.id)?.substages?.length || 0;
        return (
          <button
            className={`nav-item ${activeView === stage.id ? "active" : ""}`}
            type="button"
            key={stage.id}
            onClick={() => setStageView(stage.id)}
          >
            <span>{stage.step}</span>
            <strong>{stage.title}</strong>
            <small>
              {substageCount} substages / {lifecycle}
            </small>
          </button>
        );
      })}
    </nav>
  );
}

function PipelineSummaryPanel({ summary }) {
  return (
    <aside className="details-panel">
      <p className="eyebrow">Pipeline</p>
      <h2>Complete Build Path</h2>
      <p className="purpose">
        Canonical stages with the substage layer generated from the authored
        internals map.
      </p>
      <DetailSection title="Structure">
        <div className="detail-row">
          <span>Stages</span>
          <strong>{summary.stageCount}</strong>
        </div>
        <div className="detail-row">
          <span>Substages</span>
          <strong>{summary.substageCount}</strong>
        </div>
        <div className="detail-row">
          <span>Machinery nodes</span>
          <strong>{summary.nodeCount}</strong>
        </div>
        <div className="detail-row">
          <span>Machinery edges</span>
          <strong>{summary.edgeCount}</strong>
        </div>
      </DetailSection>
      <DetailSection title="Stages">
        {summary.stages.map((stage) => (
          <div className="detail-row" key={stage.id}>
            <span>{stage.step}</span>
            <strong>{stage.substageCount} substages</strong>
          </div>
        ))}
      </DetailSection>
    </aside>
  );
}

function StageDetails({ stage, node, internalsStage }) {
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
      <DetailSection title="Machinery">
        {internalsStage?.substages?.length ? (
          internalsStage.substages.map((substage) => (
            <div className="detail-row" key={substage.id}>
              <span>{substage.id}</span>
              <strong>
                {substage.nodes.length} nodes, {substage.edges.length} edges
              </strong>
            </div>
          ))
        ) : (
          <p className="muted">No stage machinery map.</p>
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

function edgeColor(edgeType) {
  if (edgeType === "validates") return "#7c3aed";
  if (edgeType === "conditional") return "#b45309";
  if (edgeType === "external_source") return "#0f766e";
  if (edgeType === "uses_library" || edgeType === "uses_utility") return "#2563eb";
  if (edgeType === "produces_artifact") return "#15803d";
  return "#475569";
}

function edgeStyle(edgeType) {
  const color = edgeColor(edgeType);
  return {
    stroke: color,
    strokeDasharray: edgeType === "conditional" ? "6 5" : undefined,
    strokeWidth: 1.8,
  };
}

function buildPipelineSummary(graph, internals) {
  const stages = graph.nodes.map((stage) => {
    const internalsStage = internals.stages.find((item) => item.id === stage.id);
    return {
      id: stage.id,
      step: stage.step,
      title: stage.title,
      substageCount: internalsStage?.substages?.length || 0,
    };
  });
  return {
    stageCount: graph.nodes.length,
    substageCount: internals.stages.reduce(
      (count, stage) => count + stage.substages.length,
      0,
    ),
    nodeCount: internals.stages.reduce(
      (count, stage) =>
        count +
        stage.substages.reduce(
          (substageCount, substage) => substageCount + substage.nodes.length,
          0,
        ),
      0,
    ),
    edgeCount: internals.stages.reduce(
      (count, stage) =>
        count +
        stage.substages.reduce(
          (substageCount, substage) => substageCount + substage.edges.length,
          0,
        ),
      0,
    ),
    stages,
  };
}

function DetailSection({ title, children }) {
  return (
    <section className="detail-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}
