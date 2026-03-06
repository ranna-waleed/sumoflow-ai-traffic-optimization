import React from "react";
import { ResponsiveContainer, BarChart, Bar, Tooltip } from "recharts";

function MiniBarChart({ data, color }) {
  const chartData = data.map((v, idx) => ({ idx, value: v }));

  return (
    <div className="h-12 w-20">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.05)" }}
            contentStyle={{
              backgroundColor: "#161c24",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: "8px",
              fontSize: "11px",
            }}
            formatter={(value) => [value, ""]}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} fill={color} fillOpacity={0.9} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default MiniBarChart;
