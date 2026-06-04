import { Handle, Position } from "@xyflow/react";

const HANDLE_CLASS = "stage-handle";

function NodeHandles() {
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

export default function PipelineInternalNode({ data }) {
  const source = data.sourceFile && data.line ? `${data.sourceFile}:${data.line}` : null;
  return (
    <div className={`internal-node ${data.nodeType || "process"}`}>
      <NodeHandles />
      <div className="internal-node-topline">
        <span>{(data.nodeType || "node").replaceAll("_", " ")}</span>
        {data.stability && <span>{data.stability}</span>}
      </div>
      <strong>{data.label}</strong>
      {data.description && <p>{data.description}</p>}
      {source && <small>{source}</small>}
    </div>
  );
}
