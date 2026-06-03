import { EdgeLabelRenderer, getSmoothStepPath } from "@xyflow/react";

export default function ElkEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  label,
  data,
}) {
  let edgePath;
  let labelX;
  let labelY;

  if (data?.elkRoute) {
    const { startPoint, endPoint, bendPoints = [] } = data.elkRoute;
    const points = [startPoint, ...bendPoints, endPoint];
    edgePath = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
    labelX = (points[0].x + points[points.length - 1].x) / 2;
    labelY = (points[0].y + points[points.length - 1].y) / 2;
  } else {
    const [path, x, y] = getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
    });
    edgePath = path;
    labelX = x;
    labelY = y;
  }

  return (
    <>
      <path
        id={id}
        className="react-flow__edge-path"
        d={edgePath}
        style={style}
        markerEnd={markerEnd}
        fill="none"
      />
      {label && (
        <EdgeLabelRenderer>
          <div
            className="edge-label nodrag nopan"
            style={{
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            }}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
