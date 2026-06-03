import { Handle, Position } from "@xyflow/react";

const HANDLE_CLASS = "stage-handle";

function StageHandles() {
  return (
    <>
      <Handle type="target" position={Position.Top} id="tt" className={HANDLE_CLASS} />
      <Handle type="source" position={Position.Top} id="st" className={HANDLE_CLASS} />
      <Handle type="target" position={Position.Right} id="tr" className={HANDLE_CLASS} />
      <Handle type="source" position={Position.Right} id="sr" className={HANDLE_CLASS} />
      <Handle type="target" position={Position.Bottom} id="tb" className={HANDLE_CLASS} />
      <Handle type="source" position={Position.Bottom} id="sb" className={HANDLE_CLASS} />
      <Handle type="target" position={Position.Left} id="tl" className={HANDLE_CLASS} />
      <Handle type="source" position={Position.Left} id="sl" className={HANDLE_CLASS} />
    </>
  );
}

export default function PipelineStageNode({ data }) {
  const lifecycle = data.overlay?.lifecycleStatus || "pending";
  const readiness = data.overlay?.status || "missing";
  return (
    <div className={`stage-node ${lifecycle} ${data.selected ? "selected" : ""}`}>
      <StageHandles />
      <div className="stage-node-topline">
        <span>{data.step}</span>
        <span>{readiness.replaceAll("_", " ")}</span>
      </div>
      <strong>{data.title}</strong>
      <p>{data.purpose}</p>
      <div className="stage-node-footer">
        <span>{lifecycle}</span>
        <span>{data.artifacts.length} artifacts</span>
      </div>
    </div>
  );
}
