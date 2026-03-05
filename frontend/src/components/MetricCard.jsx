import React from "react";
import MiniBarChart from "./MiniBarChart";

function MetricCard({ label, value, unit, color, data }) {
  return (
    <div className="card flex items-center justify-between p-5">
      <div>
        <div className="text-xs font-medium uppercase tracking-wider text-[#94a3b8] mb-1">
          {label}
        </div>
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-2xl md:text-3xl font-semibold" style={{ color }}>
            {value}
          </span>
          <span className="text-sm text-[#64748b]">{unit}</span>
        </div>
      </div>
      <MiniBarChart data={data} color={color} />
    </div>
  );
}

export default MetricCard;
