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
  const codeReference =
    data.objectPath ||
    (Array.isArray(data.apiRefs) && data.apiRefs.length ? data.apiRefs[0] : null) ||
    data.pydoc ||
    null;
  const explanation = firstTextLine(data.docstring || data.details || data.description);
  const kind = data.kind ? data.kind.replaceAll("_", " ") : null;
  return (
    <div className={`internal-node ${data.nodeType || "process"}`}>
      <NodeHandles />
      {codeReference && <code className="internal-node-api-ref">{codeReference}</code>}
      <div className="internal-node-topline">
        <span>{(data.nodeType || "node").replaceAll("_", " ")}</span>
        {kind && <span>{kind}</span>}
      </div>
      <strong>{data.label}</strong>
      {explanation && <p>{explanation}</p>}
    </div>
  );
}

function firstTextLine(value) {
  if (!value) return "";
  return String(value).split("\n").find((line) => line.trim())?.trim() || "";
}
